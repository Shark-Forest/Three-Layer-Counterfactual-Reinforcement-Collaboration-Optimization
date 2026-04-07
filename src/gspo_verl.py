import torch
from torch.optim import Adam
from src.config import *
from src.model_loader import get_gpt2, get_prompt_token_limit, generate_response

# ==============================================
# 严格对齐arxiv:2507.18071 核心公式
# 纯PyTorch实现，移除所有verl依赖
# ==============================================

# 这份文件实现的是“一个 Agent 的 GSPO 策略更新”。
#
# 可以把它粗略理解成：
# 1. 先让当前策略生成一组候选回答 y_1, ..., y_G
# 2. 对每个候选回答计算奖励 r_i
# 3. 把组内奖励标准化，得到 advantage A_i
# 4. 计算当前策略和旧策略在这个回答上的概率比值 s_i(theta)
# 5. 用类似 PPO 的 clipped surrogate objective 做一次梯度更新
#
# 论文里最核心的几个量如下：
#
# (1) 组内优势（group-relative advantage）
#     A_i = (r_i - mean(r)) / std(r)
#
# (2) 序列级重要性比率（sequence-level importance ratio）
#     s_i(theta) = exp((log pi_theta(y_i|x) - log pi_old(y_i|x)) / |y_i|)
#
#     注意这里除以 |y_i|，是为了把“整段回答的 log probability 差值”
#     归一化到“每个 token 的平均差值”，避免长回答天然数值更大。
#
# (3) GSPO / PPO 风格的裁剪目标
#     L_i(theta) = min(s_i(theta) * A_i,
#                      clip(s_i(theta), 1-eps, 1+eps) * A_i)
#
#     实现里我们最终最小化的是负号后的 loss：
#     loss = - mean_i L_i(theta)
#
# 下面的代码会把这些公式一步一步翻译成 PyTorch。

class GSPOAgentPolicy:
    """
    内层Agent的GSPO策略：对应你的π₀(合作)/π₁(独立)
    完全严格对齐arxiv:2507.18071论文的GSPO更新逻辑
    """
    def __init__(self, is_coop: bool):
        self.is_coop = is_coop
        # get_gpt2 只是历史命名遗留，实际上这里会加载 config.py 中当前配置的模型；
        # 现在实验里实际加载的是 Phi-3，而不是 GPT-2。
        self.tokenizer, self.model, self.device = get_gpt2(
            independent=GSPO_USE_INDEPENDENT_MODEL,
        )
        # 现在仍然控制可训练范围，但不再只训 lm_head：
        # 我们额外解冻最后若干层 decoder block 和最终 norm，
        # 让 Phi-3 有足够的表示能力去适应 GSPO 更新。
        self.trainable_params = self._configure_trainable_params()
        # last_update 只做诊断记录。
        # 这里的 selected_text 仅对应“batch 采样路径里真正执行的那个候选”，
        # 不会因为组内 best_text 更高分就替换掉本轮真实执行结果。
        self.last_update = None
        
        # 论文GSPO核心超参
        # G: 每次更新时采样多少个候选回答
        self.G = GSPO_NUM_CANDIDATES
        # clip_eps: PPO/GSPO 裁剪区间中的 epsilon
        self.clip_eps = GSPO_CLIP_EPS
        self.lr = GSPO_LR

        # 旧策略的 log-prob 在“采样候选时”直接缓存下来，
        # 更新时不再额外跑一份 CPU 大模型前向。
        #
        # 这里继续使用 Adam，但前提是：
        # 训练副本不要再停留在 fp16。
        # 我们现在会在 model_loader.get_gpt2(independent=True) 中把 GSPO 副本
        # 转成更稳的训练精度（默认 bf16），避免 Adam 状态直接在 fp16 上溢出。
        self.optimizer = Adam(self.trainable_params, lr=self.lr)

        # 模式对应的Prompt模板
        # is_coop=False 时通常表示“独立求解/直接给答案”
        # is_coop=True 时通常表示“协作思考/不给最终答案”
        self.prompt_template = PI0_PROMPT if is_coop else PI1_PROMPT

    def _move_optimizer_state(self, device):
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def _activate_model(self):
        target_device = self.device
        current_device = next(self.model.parameters()).device
        if str(current_device) != str(target_device):
            self.model.to(target_device)
            self._move_optimizer_state(target_device)
        return target_device

    def _offload_model(self, release_cuda_cache=True):
        if not GSPO_USE_INDEPENDENT_MODEL:
            return
        current_device = next(self.model.parameters()).device
        if current_device.type != "cpu":
            self.model.to("cpu")
            self._move_optimizer_state("cpu")
            if release_cuda_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def prepare_for_teardown(self):
        """
        进程退出前的保守收尾：
        - 显式切回 eval，避免还留在训练态
        - 只把模型/优化器状态迁回 CPU
        - 不在这里额外调用 empty_cache，避免在异常 GPU runtime 状态下再次触发全局显存清理
        """
        self.model.eval()
        self._offload_model(release_cuda_cache=False)

    def _configure_trainable_params(self):
        """
        Phi-3 全参数 GSPO 的优化器状态显存过大，这里采用折中方案：
        - 始终训练 lm_head
        - 训练最终 norm
        - 训练最后 GSPO_TRAIN_LAST_N_LAYERS 层 decoder block

        这样比“只训 lm_head”更有学习能力，
        又比全参数微调更省显存。
        """
        trainable = []
        num_hidden_layers = getattr(self.model.config, "num_hidden_layers", 0)
        last_layer_start = max(0, num_hidden_layers - GSPO_TRAIN_LAST_N_LAYERS)

        for name, param in self.model.named_parameters():
            should_train = False
            if name.startswith("lm_head"):
                should_train = True
            elif name.startswith("model.norm"):
                should_train = True
            elif name.startswith("model.layers."):
                parts = name.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    layer_idx = int(parts[2])
                    should_train = layer_idx >= last_layer_start
            param.requires_grad_(should_train)
            if should_train:
                trainable.append(param)

        if not trainable:
            raise RuntimeError("未找到可训练的 Phi-3 末端参数，无法执行 GSPO 更新。")
        return trainable

    def _get_sequence_log_likelihood(self, model, input_ids, attention_mask, labels):
        """
        计算一整段回答 y_i 在给定输入 x 下的对数似然 log pi(y_i | x)。

        这里要注意两点：
        1. 我们算的是“整段回答”的 log probability，不是单个 token。
        2. causal LM 的第 t 个位置，预测的是第 t+1 个 token，
           所以一定要做标准的 shift。
        """
        model_device = next(model.parameters()).device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        labels = labels.to(model_device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        # 标准的“左移 logits、右移 labels”写法：
        # shift_logits[:, t] 对应预测 shift_labels[:, t]
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)

        # labels 中被设成 -100 的位置表示“这部分不参与 loss / log-prob 计算”。
        # 在本实验里，prompt 部分都会被屏蔽成 -100，只保留生成回答部分。
        #
        # 但 gather 不能直接索引 -100，所以先把无效位置临时替换成 0，
        # 再用 valid_mask 把它们乘回 0。
        valid_mask = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~valid_mask, 0)

        token_log_probs = log_probs.gather(
            dim=-1,
            index=safe_labels.unsqueeze(-1),
        ).squeeze(-1)
        # 只保留有效 token 的对数概率，prompt 那部分会被置零。
        token_log_probs = token_log_probs * valid_mask
        # 序列总 log probability = 各 token log probability 之和
        seq_log_prob = token_log_probs.sum(dim=-1)
        # seq_length = 真正参与比较的生成 token 数
        seq_length = valid_mask.sum(dim=-1)
        return seq_log_prob, seq_length

    def compute_importance_ratio(self, input_ids, attention_mask, labels, old_log_prob=None, old_seq_len=None):
        """
        论文公式(7)：序列级重要性比率 s_i(theta)

        公式直观上是在问：
        “当前策略 pi_theta 相比旧策略 pi_old，
         对这条回答 y_i 是更偏好还是更不偏好？”

        具体实现：
            log_ratio = (log pi_theta(y_i|x) - log pi_old(y_i|x)) / |y_i|
            s_i(theta) = exp(log_ratio)

        如果 s_i > 1，说明当前策略更偏好这条回答；
        如果 s_i < 1，说明当前策略比旧策略更不偏好这条回答。
        """
        log_prob_theta, seq_len = self._get_sequence_log_likelihood(self.model, input_ids, attention_mask, labels)
        if old_log_prob is None or old_seq_len is None:
            with torch.no_grad():
                log_prob_old, old_seq_len = self._get_sequence_log_likelihood(
                    self.model,
                    input_ids,
                    attention_mask,
                    labels,
                )
                log_prob_old = log_prob_old.detach()
                old_seq_len = old_seq_len.detach()
        else:
            log_prob_old = old_log_prob.to(log_prob_theta.device)
            old_seq_len = old_seq_len.to(log_prob_theta.device)

        norm_seq_len = old_seq_len.clamp(min=1)
        log_ratio = (log_prob_theta - log_prob_old) / norm_seq_len
        s_i = torch.exp(log_ratio)
        return s_i, seq_len

    def act(self, query_context, prompt_override=None):
        """
        用当前策略生成 G 个候选回答。

        query_context: 题目 + 历史上下文
        prompt_override: 允许 run_all.py 在外部指定本轮 prompt。
                         这对“单智能体 GSPO 要严格复用 single_llm 的 prompt 节奏”
                         很重要。
        """
        # 候选采样属于“在线 rollout”，必须关闭 dropout 等训练时随机性，
        # 否则 update() 之后模型若停留在 train 模式，会直接污染后续真实轨迹。
        device = self._activate_model()
        self.model.eval()
        try:
            prompt_template = prompt_override or self.prompt_template
            prompt = prompt_template.format(context=query_context)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=get_prompt_token_limit(self.model, self.tokenizer),
            ).to(device)

            candidates = []
            input_len = inputs.input_ids.shape[1]
            for _ in range(self.G):
                # 每次循环都重新采样一次，因此最终会得到 G 个随机候选。
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        num_return_sequences=1
                    )[0]
                # output = prompt token + 新生成 token
                # 我们只把新生成部分当成候选回答文本。
                generated_ids = output[input_len:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                full_input_ids = output.unsqueeze(0)
                full_attention_mask = torch.ones_like(full_input_ids)
                full_labels = full_input_ids.clone()
                # prompt 部分不参与序列 log-likelihood 比较，设成 -100。
                full_labels[:, :input_len] = -100
                with torch.no_grad():
                    old_log_prob, old_seq_len = self._get_sequence_log_likelihood(
                        self.model,
                        full_input_ids,
                        full_attention_mask,
                        full_labels,
                    )

                candidates.append({
                    "text": generated_text,
                    # 候选缓存到 CPU，避免跨多轮延迟更新时长期占用显存。
                    "input_ids": full_input_ids.cpu(),
                    "attention_mask": full_attention_mask.cpu(),
                    "labels": full_labels.cpu(),
                    "old_log_prob": old_log_prob.detach().cpu(),
                    "old_seq_len": old_seq_len.detach().cpu(),
                })
            return candidates
        finally:
            self._offload_model()

    def sample_candidates(self, query_context, prompt_override=None, selected_idx=0):
        """
        延迟奖励版本的第一步：只负责采样候选，不立刻更新参数。

        selected_idx 只在“训练 / batch 采样”路径里有意义，
        表示这次真实训练分支采用哪一个候选。
        验证/测试阶段不走这里，而是直接调用 sample_text 做单次采样。
        """
        candidates = self.act(query_context, prompt_override=prompt_override)
        if not candidates:
            raise RuntimeError("GSPO 采样未返回任何候选。")

        selected_idx = max(0, min(selected_idx, len(candidates) - 1))
        batch = {
            "query_context": query_context,
            "prompt_override": prompt_override,
            "candidates": candidates,
            "selected_idx": selected_idx,
            "selected_text": candidates[selected_idx]["text"],
        }
        self.last_update = {
            "selected_text": batch["selected_text"],
            # best_text 仅用于诊断或日志，不参与真实轨迹上的 incumbent 更新。
            "best_text": batch["selected_text"],
            "candidates": [cand["text"] for cand in candidates],
            "rewards": None,
        }
        return batch

    def sample_text(self, query_context, prompt_override=None):
        """
        只采样一条文本。

        用途：
        - 验证/测试阶段的真实单轨迹执行
        - 反事实 reroll

        它不构造候选 batch，也不参与 GSPO 参数更新。
        """
        # 反事实 rollout / 真实轨迹都应使用 eval 模式，
        # 避免把训练态 dropout 噪声混进策略行为。
        device = self._activate_model()
        self.model.eval()
        try:
            prompt_template = prompt_override or self.prompt_template
            prompt = prompt_template.format(context=query_context)
            return generate_response(
                prompt,
                self.model,
                self.tokenizer,
                device,
            )
        finally:
            self._offload_model()

    def compute_advantage(self, rewards):
        """
        论文公式(6)：组内优势估计

            A_i = (r_i - mean(r)) / std(r)

        这一步的作用是“只比较同一组候选内部谁更好”，
        让优化更关注相对排序，而不是绝对奖励值大小。
        """
        model_device = next(self.model.parameters()).device
        rewards = torch.tensor(rewards, dtype=torch.float32).to(model_device)
        mean_r = rewards.mean()
        std_r = rewards.std().clamp(min=1e-8)
        advantage = (rewards - mean_r) / std_r
        return advantage

    def update_from_cached(self, batch, rewards):
        """
        延迟奖励版本的第二步：针对之前缓存的 candidates，使用现在得到的 rewards 做更新。
        """
        self._activate_model()
        self.model.train()
        try:
            candidates = batch["candidates"]
            if len(candidates) == 0:
                return 0.0

            if len(rewards) != len(candidates):
                raise ValueError("缓存候选数与奖励数不一致，无法执行 GSPO 更新。")

            best_idx = max(range(len(rewards)), key=lambda idx: rewards[idx])
            avg_reward = sum(rewards) / len(rewards)

            reward_std = float(torch.tensor(rewards, dtype=torch.float32).std().item())
            if GSPO_SKIP_ZERO_SIGNAL_UPDATE and reward_std < GSPO_MIN_REWARD_STD:
                self.last_update = {
                    "avg_reward": avg_reward,
                    "best_reward": rewards[best_idx],
                    # 仅诊断：batch 路径里真正执行的仍是 selected_text。
                    "best_text": candidates[best_idx]["text"],
                    "selected_text": batch["selected_text"],
                    "selected_reward": rewards[batch["selected_idx"]],
                    "candidates": [cand["text"] for cand in candidates],
                    "rewards": rewards,
                    "skipped_update": True,
                }
                return avg_reward

            advantage = self.compute_advantage(rewards)

            candidate_losses = []
            for cand, adv in zip(candidates, advantage):
                s_i, _ = self.compute_importance_ratio(
                    cand["input_ids"],
                    cand["attention_mask"],
                    cand["labels"],
                    old_log_prob=cand["old_log_prob"],
                    old_seq_len=cand["old_seq_len"],
                )
                s_i_clipped = torch.clamp(s_i, 1 - self.clip_eps, 1 + self.clip_eps)
                surr1 = s_i * adv
                surr2 = s_i_clipped * adv
                candidate_losses.append(-torch.min(surr1, surr2).mean())

            policy_loss = torch.stack(candidate_losses).mean()

            if not torch.isfinite(policy_loss):
                self.last_update = {
                    "avg_reward": avg_reward,
                    "best_reward": rewards[best_idx],
                    # 仅诊断：batch 路径里真正执行的仍是 selected_text。
                    "best_text": candidates[best_idx]["text"],
                    "selected_text": batch["selected_text"],
                    "selected_reward": rewards[batch["selected_idx"]],
                    "candidates": [cand["text"] for cand in candidates],
                    "rewards": rewards,
                    "skipped_update": True,
                }
                return avg_reward

            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trainable_params, GSPO_MAX_GRAD_NORM)
            self.optimizer.step()

            self.last_update = {
                "avg_reward": avg_reward,
                "best_reward": rewards[best_idx],
                # 仅诊断：batch 路径里真正执行的仍是 selected_text。
                "best_text": candidates[best_idx]["text"],
                "selected_text": batch["selected_text"],
                "selected_reward": rewards[batch["selected_idx"]],
                "candidates": [cand["text"] for cand in candidates],
                "rewards": rewards,
            }
            return self.last_update["avg_reward"]
        finally:
            # 不管更新是否成功，后续轨迹采样都必须回到 eval 模式。
            self.model.eval()
            self._offload_model()

    def update(self, query_context, reward_func, prompt_override=None):
        """
        GSPO 的一次完整更新。

        这个函数可以按“采样 -> 打分 -> 标准化 -> 计算概率比 -> clipped loss -> 更新参数”
        这条主线来读。
        """
        batch = self.sample_candidates(
            query_context,
            prompt_override=prompt_override,
        )
        rewards = [reward_func(cand["text"]) for cand in batch["candidates"]]
        return self.update_from_cached(batch, rewards)
