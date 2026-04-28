import copy
import os
import random
import re

import numpy as np
import torch
from torch.optim import Adam
from transformers import LogitsProcessor, LogitsProcessorList
from src28.config import *
from src28.model_loader import (
    build_generation_inputs,
    generate_response,
    get_gpt2,
    get_prompt_token_limit,
)

_PEFT_IMPORT_ERROR = None
try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception as exc:
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    _PEFT_IMPORT_ERROR = exc

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

_GSPO_LOGIT_CLAMP = 80.0
_GSPO_TRANSITION_SCORE_CLAMP = 80.0
_GSPO_LOG_RATIO_CLAMP = 2.0
_PROPOSAL_CANDIDATE_FINAL_PATTERN = re.compile(
    r"^\s*candidate\s*[:：]\s*(-?\$?\d[\d,]*\.?\d*)\s*$",
    re.IGNORECASE,
)
_PROPOSAL_EMPTY_CANDIDATE_PATTERN = re.compile(
    r"^\s*candidate\s*[:：]\s*$",
    re.IGNORECASE,
)
_PROPOSAL_NATURAL_FINAL_PATTERN = re.compile(
    r"^\s*(?:therefore,\s*)?(?:the\s+)?(?:candidate|answer|final answer)\s*(?:is\s+|[:：]\s*)(-?\$?\d[\d,]*\.?\d*)\s*\.?\s*$",
    re.IGNORECASE,
)
_REASON_LINE_PATTERN = re.compile(
    r"^\s*reason\s*[:：]\s*(.+?)\s*$",
    re.IGNORECASE,
)
_PROPOSAL_REASON_FINAL_EQUALS_PATTERN = re.compile(
    r"=\s*(-?\$?\d[\d,]*\.?\d*)\b"
)
_PROPOSAL_REASON_TRAILING_NUMBER_PATTERN = re.compile(
    r"(-?\$?\d[\d,]*\.?\d*)\s*\.?\s*$"
)
_PROPOSAL_ANSWER_CUE_PATTERN = re.compile(
    r"(candidate|final answer|the answer is|answer[:：]|答案是|最终答案|答案[:：])",
    re.IGNORECASE,
)
_PROPOSAL_BARE_NUMBER_LINE_PATTERN = re.compile(
    r"^\s*-?\$?\d[\d,]*\.?\d*\s*$"
)
_PROPOSAL_NUMBER_PATTERN = re.compile(r"(?<![\w/.-])-?\$?\d[\d,]*\.?\d*")
_REVIEW_JUDGMENT_PATTERN = re.compile(
    r"^\s*judg(?:e)?ment\s*[:：]\s*(right|wrong)\s*\.?\s*$",
    re.IGNORECASE,
)
_REVIEW_NATURAL_JUDGMENT_PATTERN = re.compile(
    r"^\s*(?:therefore,\s*)?(?:the\s+)?judg(?:e)?ment\s+(?:is\s+)?(right|wrong)\s*\.?\s*$",
    re.IGNORECASE,
)
_REVIEW_JUDGMENT_PREFIX_PATTERN = re.compile(
    r"^\s*judg(?:e)?ment\s*[:：]\s*([A-Za-z]*)\s*$",
    re.IGNORECASE,
)
_REVIEW_INLINE_TAIL_JUDGMENT_PATTERN = re.compile(
    r"^(?P<reason>.*?)(?:\s+|^)(?:judg(?:e)?ment)\s*[:：]\s*(?P<judgment>right|wrong)\s*\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_REVIEW_INLINE_TAIL_NATURAL_JUDGMENT_PATTERN = re.compile(
    r"^(?P<reason>.*?)(?:\s+|^)(?:the\s+)?judg(?:e)?ment\s+(?:is\s+)?(?P<judgment>right|wrong)\s*\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_finetune_mode(value):
    mode = str(value or "last_n_layers").strip().lower()
    alias_map = {
        "last_n_layers": "last_n_layers",
        "last_layers": "last_n_layers",
        "partial": "last_n_layers",
        "lora": "lora",
    }
    if mode not in alias_map:
        raise ValueError(f"不支持的 GSPO_FINETUNE_MODE: {value}")
    return alias_map[mode]


def _parse_target_module_names(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]

def _policy_role_label(is_coop):
    if THREE_LAYER_MIDDLE_ACTION_SCHEMA == "proposal_review":
        return "reviewer" if is_coop else "proposer"
    return "pi0" if is_coop else "pi1"

def _extract_proposal_number_tokens(text):
    values = []
    for token in _PROPOSAL_NUMBER_PATTERN.findall(text or ""):
        cleaned = token.replace("$", "").replace(",", "").strip()
        if not cleaned:
            continue
        try:
            values.append(float(cleaned))
        except ValueError:
            continue
    return values

def _format_candidate_number(value):
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def _extract_reason_final_number(lines):
    for line in lines:
        match = _REASON_LINE_PATTERN.match(line)
        if not match:
            continue
        reason_text = match.group(1).strip()
        eq_matches = _PROPOSAL_REASON_FINAL_EQUALS_PATTERN.findall(reason_text)
        if eq_matches:
            try:
                return _format_candidate_number(float(eq_matches[-1].replace("$", "").replace(",", "").strip()))
            except ValueError:
                pass
        trailing_match = _PROPOSAL_REASON_TRAILING_NUMBER_PATTERN.search(reason_text)
        if trailing_match:
            try:
                return _format_candidate_number(
                    float(trailing_match.group(1).replace("$", "").replace(",", "").strip())
                )
            except ValueError:
                pass
    return None

def _normalize_candidate_value_text(value):
    cleaned = str(value or "").replace("$", "").replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return _format_candidate_number(float(cleaned))
    except ValueError:
        return None

def _extract_explicit_candidate_value(lines):
    for line_idx, line in enumerate(lines):
        if _PROPOSAL_EMPTY_CANDIDATE_PATTERN.match(line):
            return line_idx, None, True
        match = _PROPOSAL_CANDIDATE_FINAL_PATTERN.match(line)
        if match:
            return line_idx, match.group(1), True
    return None, None, False

def _is_prefix_truncated_candidate_value(candidate_value, reason_value):
    candidate_text = _normalize_candidate_value_text(candidate_value)
    reason_text = _normalize_candidate_value_text(reason_value)
    if not candidate_text or not reason_text or candidate_text == reason_text:
        return False
    return len(candidate_text) < len(reason_text) and reason_text.startswith(candidate_text)

def _split_nonempty_lines(text):
    return [
        line.strip()
        for line in str(text or "").replace("\r\n", "\n").split("\n")
        if line.strip()
    ]

def _extract_proposal_final_value_anywhere(lines):
    for line in reversed(lines):
        if _PROPOSAL_ANSWER_CUE_PATTERN.search(line):
            values = _extract_proposal_number_tokens(line)
            if values:
                return _format_candidate_number(values[-1])
        if "\\boxed" in line.lower():
            values = _extract_proposal_number_tokens(line)
            if values:
                return _format_candidate_number(values[-1])
    return None

def _find_proposal_final_line_index(lines):
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if _PROPOSAL_ANSWER_CUE_PATTERN.search(line):
            values = _extract_proposal_number_tokens(line)
            if values:
                return idx
        if "\\boxed" in line.lower():
            values = _extract_proposal_number_tokens(line)
            if values:
                return idx
    return None

def _extract_proposal_final_value(text):
    lines = _split_nonempty_lines(text)
    if not lines:
        return None
    last_line = lines[-1]
    if _PROPOSAL_ANSWER_CUE_PATTERN.search(last_line):
        values = _extract_proposal_number_tokens(last_line)
        if values:
            return _format_candidate_number(values[-1])
    if "\\boxed" in last_line.lower():
        values = _extract_proposal_number_tokens(last_line)
        if values:
            return _format_candidate_number(values[-1])
    return None

def _extract_review_judgment_token(lines):
    if not lines:
        return None
    first_line = lines[0]
    match = _REVIEW_JUDGMENT_PATTERN.match(first_line)
    if match:
        return match.group(1).upper()
    match = _REVIEW_NATURAL_JUDGMENT_PATTERN.match(first_line)
    if match:
        return match.group(1).upper()
    inline_reason, inline_judgment = _split_inline_tail_review_judgment(first_line)
    if inline_judgment is not None and not inline_reason:
        return inline_judgment
    normalized = first_line.strip().lower().rstrip(".")
    if normalized == "right":
        return "RIGHT"
    if normalized == "wrong":
        return "WRONG"
    return None

def _extract_review_judgment_token_anywhere(lines):
    for line in reversed(lines):
        match = _REVIEW_JUDGMENT_PATTERN.match(line)
        if match:
            return match.group(1).upper()
        match = _REVIEW_NATURAL_JUDGMENT_PATTERN.match(line)
        if match:
            return match.group(1).upper()
        inline_reason, inline_judgment = _split_inline_tail_review_judgment(line)
        if inline_judgment is not None:
            return inline_judgment
        normalized = line.strip().lower().rstrip(".")
        if normalized == "right":
            return "RIGHT"
        if normalized == "wrong":
            return "WRONG"
    return None

def _split_inline_tail_review_judgment(text):
    cleaned_text = str(text or "").strip()
    if not cleaned_text:
        return None, None
    for pattern in (
        _REVIEW_INLINE_TAIL_JUDGMENT_PATTERN,
        _REVIEW_INLINE_TAIL_NATURAL_JUDGMENT_PATTERN,
    ):
        match = pattern.match(cleaned_text)
        if not match:
            continue
        reason = (match.group("reason") or "").strip()
        judgment = (match.group("judgment") or "").strip().upper()
        if judgment in {"RIGHT", "WRONG"}:
            return reason, judgment
    return None, None

def _proposal_needs_completion(text):
    cleaned_text = str(text or "").strip()
    if not cleaned_text:
        return False
    lines = _split_nonempty_lines(cleaned_text)
    if not lines:
        return False
    return (
        _extract_proposal_final_value(cleaned_text) is None
        and _extract_proposal_final_value_anywhere(lines) is None
    )

def _review_needs_completion(text):
    cleaned_text = str(text or "").strip()
    if not cleaned_text:
        return False
    lines = _split_nonempty_lines(cleaned_text)
    if not lines:
        return False
    return (
        _extract_review_judgment_token(lines) is None
        and _extract_review_judgment_token_anywhere(lines) is None
    )

def _proposal_completion_suffix(text):
    cleaned_text = str(text or "").rstrip()
    if not cleaned_text:
        return ""
    last_line = _split_nonempty_lines(cleaned_text)[-1]
    lowered = last_line.lower().rstrip()
    if lowered.endswith("final answer:"):
        return " "
    if lowered.endswith("final answer"):
        return ": "
    if lowered.endswith("answer:"):
        return " "
    return "\nFinal answer: "

def _review_completion_suffix(text):
    cleaned_text = str(text or "").rstrip()
    if not cleaned_text:
        return ""
    last_line = _split_nonempty_lines(cleaned_text)[-1]
    lowered = last_line.lower().rstrip()
    if lowered.endswith("judgment:") or lowered.endswith("judgement:"):
        return " "
    if lowered.endswith("judgment") or lowered.endswith("judgement"):
        return ": "
    return "\nJudgment: "


def _infer_default_lora_target_modules(model_type, available_suffixes):
    if model_type == "phi3":
        ordered = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    elif model_type.startswith("qwen2"):
        ordered = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        ordered = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "qkv_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "gate_up_proj",
        ]

    resolved = [name for name in ordered if name in available_suffixes]
    if not resolved:
        raise ValueError(
            f"无法为 model_type={model_type or '<unknown>'} 自动推断 LoRA target modules。"
        )
    return resolved


def _move_state_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _move_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_state_to_cpu(item) for item in value)
    return copy.deepcopy(value)


class _FiniteLogitsProcessor(LogitsProcessor):
    """
    generate 采样前的最后一道防线：
    - 把 nan/inf logits 替换成 0
    - 把过大绝对值裁到稳定区间，避免 softmax 概率张量炸掉
    """

    def __init__(self, owner, stage):
        self.owner = owner
        self.stage = stage

    def __call__(self, input_ids, scores):
        if not torch.is_tensor(scores):
            return scores

        finite_mask = torch.isfinite(scores)
        if not bool(finite_mask.all()):
            non_finite_count = int((~finite_mask).sum().item())
            self.owner._log_numeric_guard(
                f"{self.stage}:logits",
                f"{self.stage} 发现 {non_finite_count} 个非有限 logits，已替换并裁剪。",
            )

        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores.clamp(min=-_GSPO_LOGIT_CLAMP, max=_GSPO_LOGIT_CLAMP)

class GSPOAgentPolicy:
    """
    内层Agent的GSPO策略：对应你的π₀(合作)/π₁(独立)
    完全严格对齐arxiv:2507.18071论文的GSPO更新逻辑
    """
    def __init__(self, is_coop: bool):
        self.is_coop = is_coop
        keep_model_on_device_raw = str(
            os.environ.get("MAS_KEEP_POLICY_ON_DEVICE", "")
        ).strip().lower()
        if keep_model_on_device_raw:
            self.keep_model_on_device = keep_model_on_device_raw in {"1", "true", "yes", "y", "on"}
        else:
            self.keep_model_on_device = bool(
                GSPO_USE_INDEPENDENT_MODEL
                and THREE_LAYER_MIDDLE_ACTION_SCHEMA == "proposal_review"
                and torch.cuda.is_available()
                and torch.cuda.device_count() >= 2
            )
        # get_gpt2 只是历史命名遗留，实际上这里会加载 config.py 中当前配置的模型；
        # 现在实验里实际加载的是 Phi-3，而不是 GPT-2。
        self.tokenizer, self.model, self.device = get_gpt2(
            independent=GSPO_USE_INDEPENDENT_MODEL,
        )
        self.base_model_type = str(
            getattr(getattr(self.model, "config", None), "model_type", "") or ""
        ).lower()
        self.finetune_mode = _normalize_finetune_mode(GSPO_FINETUNE_MODE)
        self.lora_target_modules = []
        self.model = self._configure_finetune_adaptation()
        # 现在仍然控制可训练范围，但不再只训 lm_head：
        # 我们额外解冻最后若干层 decoder block 和最终 norm，
        # 让 Phi-3 有足够的表示能力去适应 GSPO 更新。
        self.trainable_params = self._configure_trainable_params()
        self._print_trainable_summary()
        # last_update 只做诊断记录。
        # 这里的 selected_text 仅对应“batch 采样路径里真正执行的那个候选”，
        # 不会因为组内 best_text 更高分就替换掉本轮真实执行结果。
        self.last_update = None
        self._numeric_guard_counts = {}
        
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

        if self.keep_model_on_device:
            self._activate_model()
            self.model.eval()

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

    def _offload_model(self, release_cuda_cache=True, force=False):
        if not GSPO_USE_INDEPENDENT_MODEL:
            return
        if self.keep_model_on_device and not force:
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
        self._offload_model(release_cuda_cache=False, force=True)

    def _resolve_lora_target_modules(self):
        override_modules = _parse_target_module_names(GSPO_LORA_TARGET_MODULES)
        available_suffixes = {
            name.rsplit(".", 1)[-1]
            for name, _ in self.model.named_modules()
            if name
        }
        if override_modules:
            missing = [name for name in override_modules if name not in available_suffixes]
            if missing:
                raise ValueError(
                    "LoRA target modules 在当前模型中不存在: " + ", ".join(missing)
                )
            return override_modules
        return _infer_default_lora_target_modules(
            self.base_model_type,
            available_suffixes,
        )

    def _configure_finetune_adaptation(self):
        if self.finetune_mode != "lora":
            return self.model

        if get_peft_model is None or LoraConfig is None or TaskType is None:
            raise RuntimeError(
                "当前环境无法启用 LoRA，请检查 peft 安装是否与 transformers 版本兼容。"
                f" 原始错误: {_PEFT_IMPORT_ERROR!r}"
            )

        self.lora_target_modules = self._resolve_lora_target_modules()
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=GSPO_LORA_R,
            lora_alpha=GSPO_LORA_ALPHA,
            lora_dropout=GSPO_LORA_DROPOUT,
            bias=GSPO_LORA_BIAS,
            target_modules=self.lora_target_modules,
        )
        return get_peft_model(self.model, lora_config)

    def _print_trainable_summary(self):
        total_params = sum(param.numel() for param in self.model.parameters())
        trainable_params = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
        role = _policy_role_label(self.is_coop)
        extra = (
            f"target_modules={self.lora_target_modules}"
            if self.finetune_mode == "lora"
            else f"last_n_layers={GSPO_TRAIN_LAST_N_LAYERS}"
        )
        ratio = 0.0 if total_params <= 0 else trainable_params / total_params
        print(
            f"[GSPO训练参数][{role}] model_type={self.base_model_type or '<unknown>'} "
            f"mode={self.finetune_mode} trainable={trainable_params}/{total_params} "
            f"({ratio:.4%}) {extra}",
            flush=True,
        )

    def get_checkpoint_state(self):
        trainable_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_state[name] = param.detach().cpu()

        checkpoint_state = {
            "is_coop": bool(self.is_coop),
            "finetune_mode": self.finetune_mode,
            "model_type": self.base_model_type,
            "lora_target_modules": list(self.lora_target_modules),
            "trainable_state": trainable_state,
            "optimizer_state_dict": _move_state_to_cpu(self.optimizer.state_dict()),
            "last_update": copy.deepcopy(self.last_update),
            "numeric_guard_counts": dict(self._numeric_guard_counts),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state().cpu(),
        }

        if torch.cuda.is_available() and str(self.device).startswith("cuda:"):
            checkpoint_state["cuda_rng_state"] = torch.cuda.get_rng_state(self.device).cpu()
        else:
            checkpoint_state["cuda_rng_state"] = None

        return checkpoint_state

    def load_checkpoint_state(self, checkpoint_state):
        if checkpoint_state is None:
            return

        checkpoint_finetune_mode = checkpoint_state.get("finetune_mode")
        if checkpoint_finetune_mode is not None:
            if _normalize_finetune_mode(checkpoint_finetune_mode) != self.finetune_mode:
                raise ValueError(
                    "checkpoint 训练模式与当前配置不一致："
                    f" checkpoint={checkpoint_finetune_mode} current={self.finetune_mode}"
                )

        checkpoint_model_type = str(checkpoint_state.get("model_type") or "").lower()
        if checkpoint_model_type and checkpoint_model_type != self.base_model_type:
            raise ValueError(
                "checkpoint 底模类型与当前模型不一致："
                f" checkpoint={checkpoint_model_type} current={self.base_model_type}"
            )

        checkpoint_lora_target_modules = list(
            checkpoint_state.get("lora_target_modules") or []
        )
        if (
            self.finetune_mode == "lora"
            and checkpoint_lora_target_modules
            and checkpoint_lora_target_modules != list(self.lora_target_modules)
        ):
            raise ValueError(
                "checkpoint LoRA target modules 与当前配置不一致："
                f" checkpoint={checkpoint_lora_target_modules} current={self.lora_target_modules}"
            )

        device = self._activate_model()
        try:
            trainable_state = checkpoint_state.get("trainable_state", {})
            named_params = dict(self.model.named_parameters())
            with torch.no_grad():
                for name, tensor in trainable_state.items():
                    if name not in named_params:
                        raise KeyError(f"checkpoint 中存在当前模型没有的参数：{name}")
                    param = named_params[name]
                    param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))

            optimizer_state_dict = checkpoint_state.get("optimizer_state_dict")
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
                self._move_optimizer_state(device)

            self.last_update = copy.deepcopy(checkpoint_state.get("last_update"))
            self._numeric_guard_counts = dict(
                checkpoint_state.get("numeric_guard_counts", {})
            )

            python_random_state = checkpoint_state.get("python_random_state")
            if python_random_state is not None:
                random.setstate(python_random_state)

            numpy_random_state = checkpoint_state.get("numpy_random_state")
            if numpy_random_state is not None:
                np.random.set_state(numpy_random_state)

            torch_rng_state = checkpoint_state.get("torch_rng_state")
            if torch_rng_state is not None:
                torch.set_rng_state(torch_rng_state.cpu())

            cuda_rng_state = checkpoint_state.get("cuda_rng_state")
            if cuda_rng_state is not None and torch.cuda.is_available() and str(self.device).startswith("cuda:"):
                torch.cuda.set_rng_state(cuda_rng_state.cpu(), device=self.device)

            self._sanitize_trainable_params("checkpoint/load")
            self._sanitize_optimizer_state("checkpoint/load")
            self.model.eval()
        finally:
            self._offload_model(release_cuda_cache=False)

    def save_checkpoint(self, path):
        device = self._activate_model()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.get_checkpoint_state(), path)
            return {
                "path": path,
                "device": str(device),
                "size_bytes": int(os.path.getsize(path)),
            }
        finally:
            self._offload_model(release_cuda_cache=False)

    def load_checkpoint(self, path):
        # 这里读取的是本地训练过程中保存的完整状态，含 optimizer/RNG 等对象。
        checkpoint_state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_checkpoint_state(checkpoint_state)
        return {"path": path}

    def _configure_trainable_params(self):
        """
        训练参数选择支持两种模式：
        - last_n_layers: 训练 lm_head + 最终 norm + 最后若干层 decoder block
        - lora: 仅训练 LoRA 注入出来的 adapter 参数
        """
        if self.finetune_mode == "lora":
            trainable = [
                param
                for _, param in self.model.named_parameters()
                if param.requires_grad
            ]
            if not trainable:
                raise RuntimeError("LoRA 模式下未找到可训练参数，无法执行 GSPO 更新。")
            return trainable

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
            raise RuntimeError("未找到可训练的末端参数，无法执行 GSPO 更新。")
        return trainable

    def _log_numeric_guard(self, key, message):
        count = self._numeric_guard_counts.get(key, 0) + 1
        self._numeric_guard_counts[key] = count
        if count <= 3 or count in {5, 10, 20, 50, 100}:
            role = _policy_role_label(self.is_coop)
            suffix = "" if count == 1 else f" (第{count}次)"
            print(f"[GSPO数值防护][{role}] {message}{suffix}", flush=True)

    def _build_generate_kwargs(
        self,
        stage,
        num_return_sequences,
        return_dict_in_generate,
        output_scores,
        force_do_sample=None,
    ):
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        use_sampling = True
        if force_do_sample is not None:
            use_sampling = bool(force_do_sample)
        elif stage.startswith("sample_text_batch") and not GSPO_EVAL_DO_SAMPLE:
            use_sampling = False

        if not use_sampling and int(num_return_sequences) > 1:
            raise ValueError(
                "do_sample=False 时 num_return_sequences 必须为 1；"
                "若需要多条确定性候选，请拆成多次生成。"
            )

        kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "pad_token_id": pad_token_id,
            "num_return_sequences": num_return_sequences,
            "return_dict_in_generate": return_dict_in_generate,
            "output_scores": output_scores,
            "remove_invalid_values": True,
            "renormalize_logits": True,
            "logits_processor": LogitsProcessorList([
                _FiniteLogitsProcessor(self, stage),
            ]),
        }
        if use_sampling:
            kwargs["temperature"] = TEMPERATURE
            kwargs["do_sample"] = True
        else:
            kwargs["do_sample"] = False
        return kwargs

    def _sanitize_logits_tensor(self, logits, stage):
        finite_mask = torch.isfinite(logits)
        if not bool(finite_mask.all()):
            non_finite_count = int((~finite_mask).sum().item())
            self._log_numeric_guard(
                f"{stage}:forward_logits",
                f"{stage} 发现 {non_finite_count} 个非有限前向 logits，已替换并裁剪。",
            )
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        return logits.clamp(min=-_GSPO_LOGIT_CLAMP, max=_GSPO_LOGIT_CLAMP)

    def _sanitize_transition_scores(self, transition_scores, stage):
        finite_mask = torch.isfinite(transition_scores)
        if not bool(finite_mask.all()):
            non_finite_count = int((~finite_mask).sum().item())
            self._log_numeric_guard(
                f"{stage}:transition_scores",
                f"{stage} 发现 {non_finite_count} 个非有限 transition score，已替换并裁剪。",
            )

        generated_token_mask = finite_mask & transition_scores.ne(0)
        transition_scores = torch.nan_to_num(transition_scores, nan=0.0, posinf=0.0, neginf=0.0)
        transition_scores = transition_scores.clamp(min=-_GSPO_TRANSITION_SCORE_CLAMP, max=0.0)
        return transition_scores, generated_token_mask

    def _should_canonicalize_proposal_output(self):
        return (
            THREE_LAYER_MIDDLE_ACTION_SCHEMA == "proposal_review"
            and not self.is_coop
        )

    def _should_canonicalize_review_output(self):
        return (
            THREE_LAYER_MIDDLE_ACTION_SCHEMA == "proposal_review"
            and self.is_coop
            and PROPOSAL_REVIEW_OUTPUT_MODE == "structured_judgment"
        )

    def _append_forced_suffix_prefix(
        self,
        full_input_ids,
        full_attention_mask,
        old_log_prob,
        old_seq_len,
        input_len,
        suffix_text,
        stage,
    ):
        suffix_text = str(suffix_text or "")
        if not suffix_text:
            decoded_text = self.tokenizer.decode(
                full_input_ids[0, input_len:],
                skip_special_tokens=True,
            ).strip()
            return (
                full_input_ids,
                full_attention_mask,
                old_log_prob,
                old_seq_len,
                decoded_text,
            )

        suffix_inputs = self.tokenizer(
            suffix_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        suffix_ids = suffix_inputs.get("input_ids")
        if suffix_ids is None or int(suffix_ids.numel()) <= 0:
            decoded_text = self.tokenizer.decode(
                full_input_ids[0, input_len:],
                skip_special_tokens=True,
            ).strip()
            return (
                full_input_ids,
                full_attention_mask,
                old_log_prob,
                old_seq_len,
                decoded_text,
            )

        device = next(self.model.parameters()).device
        prefix_input_ids = full_input_ids.to(device)
        prefix_attention_mask = full_attention_mask.to(device)
        suffix_ids = suffix_ids.to(device)
        suffix_attention_mask = torch.ones_like(suffix_ids, device=device)

        combined_input_ids = torch.cat([prefix_input_ids, suffix_ids], dim=1)
        combined_attention_mask = torch.cat(
            [prefix_attention_mask, suffix_attention_mask],
            dim=1,
        )

        model_outputs = None
        logits = None
        token_log_probs = None
        try:
            with torch.no_grad():
                model_outputs = self.model(
                    input_ids=combined_input_ids,
                    attention_mask=combined_attention_mask,
                )
            logits = self._sanitize_logits_tensor(
                model_outputs.logits,
                stage=f"{stage}/forward_logits",
            )
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = combined_input_ids[:, 1:]
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

            prefix_len = int(prefix_input_ids.shape[1])
            suffix_len = int(suffix_ids.shape[1])
            start_idx = max(0, prefix_len - 1)
            end_idx = start_idx + suffix_len
            suffix_log_prob = token_log_probs[:, start_idx:end_idx].sum(dim=1).detach().cpu()
            suffix_seq_len = torch.tensor(
                [suffix_len],
                dtype=old_seq_len.dtype,
            )

            updated_full_input_ids = combined_input_ids.cpu()
            updated_full_attention_mask = torch.ones_like(updated_full_input_ids)
            updated_old_log_prob = old_log_prob + suffix_log_prob
            updated_old_seq_len = old_seq_len + suffix_seq_len
            updated_text = self.tokenizer.decode(
                updated_full_input_ids[0, input_len:],
                skip_special_tokens=True,
            ).strip()
            return (
                updated_full_input_ids,
                updated_full_attention_mask,
                updated_old_log_prob,
                updated_old_seq_len,
                updated_text,
            )
        finally:
            del model_outputs
            del logits
            del token_log_probs

    def _generate_greedy_continuation(
        self,
        full_input_ids,
        full_attention_mask,
        input_len,
        stage,
        max_new_tokens,
    ):
        if int(max_new_tokens or 0) <= 0:
            return full_input_ids, full_attention_mask, None, None, None

        device = next(self.model.parameters()).device
        prefix_input_ids = full_input_ids.to(device)
        prefix_attention_mask = full_attention_mask.to(device)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        continuation_output = None
        continuation_sequences = None
        continuation_scores = None
        continuation_mask = None
        try:
            with torch.no_grad():
                continuation_output = self.model.generate(
                    input_ids=prefix_input_ids,
                    attention_mask=prefix_attention_mask,
                    max_new_tokens=int(max_new_tokens),
                    pad_token_id=pad_token_id,
                    num_return_sequences=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    remove_invalid_values=True,
                    renormalize_logits=True,
                    logits_processor=LogitsProcessorList([
                        _FiniteLogitsProcessor(self, stage),
                    ]),
                    do_sample=False,
                )

            continuation_sequences = continuation_output.sequences
            continuation_scores = self.model.compute_transition_scores(
                continuation_sequences,
                continuation_output.scores,
                normalize_logits=True,
            )
            continuation_scores, continuation_mask = self._sanitize_transition_scores(
                continuation_scores,
                stage=f"{stage}/transition_scores",
            )
            continuation_token_count = int(continuation_mask.sum(dim=1)[0].item())
            if continuation_token_count <= 0:
                return full_input_ids, full_attention_mask, None, None, None

            updated_total_len = int(prefix_input_ids.shape[1]) + continuation_token_count
            updated_full_input_ids = continuation_sequences[0, :updated_total_len].unsqueeze(0).cpu()
            updated_attention_mask = torch.ones_like(updated_full_input_ids)
            continuation_log_prob = continuation_scores.sum(dim=1).detach().cpu()
            continuation_seq_len = continuation_mask.sum(dim=1).detach().cpu()
            updated_text = self.tokenizer.decode(
                updated_full_input_ids[0, input_len:],
                skip_special_tokens=True,
            ).strip()
            return (
                updated_full_input_ids,
                updated_attention_mask,
                continuation_log_prob,
                continuation_seq_len,
                updated_text,
            )
        finally:
            del continuation_output
            del continuation_sequences
            del continuation_scores
            del continuation_mask

    def _maybe_complete_proposal_candidate(
        self,
        full_input_ids,
        full_attention_mask,
        full_labels,
        old_log_prob,
        old_seq_len,
        input_len,
        generated_text,
    ):
        cleaned_text = str(generated_text or "").strip()
        if not cleaned_text or not self._should_canonicalize_proposal_output():
            return (
                full_input_ids,
                full_attention_mask,
                full_labels,
                old_log_prob,
                old_seq_len,
                cleaned_text,
            )
        if int(PROPOSAL_COMPLETION_MAX_NEW_TOKENS or 0) <= 0:
            return (
                full_input_ids,
                full_attention_mask,
                full_labels,
                old_log_prob,
                old_seq_len,
                cleaned_text,
            )
        if not _proposal_needs_completion(cleaned_text):
            return (
                full_input_ids,
                full_attention_mask,
                full_labels,
                old_log_prob,
                old_seq_len,
                cleaned_text,
            )

        (
            full_input_ids,
            full_attention_mask,
            old_log_prob,
            old_seq_len,
            cleaned_text,
        ) = self._append_forced_suffix_prefix(
            full_input_ids,
            full_attention_mask,
            old_log_prob,
            old_seq_len,
            input_len,
            _proposal_completion_suffix(cleaned_text),
            stage="proposal_completion_suffix",
        )
        (
            updated_full_input_ids,
            updated_attention_mask,
            continuation_log_prob,
            continuation_seq_len,
            updated_text,
        ) = self._generate_greedy_continuation(
            full_input_ids,
            full_attention_mask,
            input_len,
            stage="proposal_completion",
            max_new_tokens=PROPOSAL_COMPLETION_MAX_NEW_TOKENS,
        )
        if updated_text is None:
            return (
                full_input_ids,
                full_attention_mask,
                full_labels,
                old_log_prob,
                old_seq_len,
                cleaned_text,
            )

        updated_labels = updated_full_input_ids.clone()
        updated_labels[:, :input_len] = -100
        updated_old_log_prob = old_log_prob + continuation_log_prob
        updated_old_seq_len = old_seq_len + continuation_seq_len
        return (
            updated_full_input_ids,
            updated_attention_mask,
            updated_labels,
            updated_old_log_prob,
            updated_old_seq_len,
            updated_text or cleaned_text,
        )

    def _maybe_complete_review_verdict_candidate(
        self,
        full_input_ids,
        full_attention_mask,
        full_labels,
        old_log_prob,
        old_seq_len,
        input_len,
        generated_text,
    ):
        cleaned_text = str(generated_text or "").strip()
        if not cleaned_text or not self._should_canonicalize_review_output():
            return (
                full_input_ids,
                full_attention_mask,
                full_labels,
                old_log_prob,
                old_seq_len,
                cleaned_text,
            )
        if not _review_needs_completion(cleaned_text):
            return (
                full_input_ids,
                full_attention_mask,
                full_labels,
                old_log_prob,
                old_seq_len,
                cleaned_text,
            )

        (
            full_input_ids,
            full_attention_mask,
            old_log_prob,
            old_seq_len,
            cleaned_text,
        ) = self._append_forced_suffix_prefix(
            full_input_ids,
            full_attention_mask,
            old_log_prob,
            old_seq_len,
            input_len,
            _review_completion_suffix(cleaned_text),
            stage="review_completion_suffix",
        )
        (
            updated_full_input_ids,
            updated_attention_mask,
            continuation_log_prob,
            continuation_seq_len,
            updated_text,
        ) = self._generate_greedy_continuation(
            full_input_ids,
            full_attention_mask,
            input_len,
            stage="review_completion",
            max_new_tokens=REVIEW_COMPLETION_MAX_NEW_TOKENS,
        )
        if updated_text is None:
            return (
                full_input_ids,
                full_attention_mask,
                full_labels,
                old_log_prob,
                old_seq_len,
                cleaned_text,
            )

        updated_labels = updated_full_input_ids.clone()
        updated_labels[:, :input_len] = -100
        updated_old_log_prob = old_log_prob + continuation_log_prob
        updated_old_seq_len = old_seq_len + continuation_seq_len
        return (
            updated_full_input_ids,
            updated_attention_mask,
            updated_labels,
            updated_old_log_prob,
            updated_old_seq_len,
            updated_text or cleaned_text,
        )

    def _maybe_complete_proposal_candidate_text_only(
        self,
        full_input_ids,
        full_attention_mask,
        input_len,
        generated_text,
    ):
        cleaned_text = str(generated_text or "").strip()
        if not cleaned_text or not self._should_canonicalize_proposal_output():
            return cleaned_text
        if not _proposal_needs_completion(cleaned_text):
            return cleaned_text

        (
            proposal_prefix_ids,
            proposal_prefix_mask,
            _,
            _,
            _,
        ) = self._append_forced_suffix_prefix(
            full_input_ids,
            full_attention_mask,
            torch.zeros(1, dtype=torch.float32),
            torch.zeros(1, dtype=torch.long),
            input_len,
            _proposal_completion_suffix(cleaned_text),
            stage="proposal_completion_text_only_suffix",
        )
        _, _, _, _, updated_text = self._generate_greedy_continuation(
            proposal_prefix_ids,
            proposal_prefix_mask,
            input_len,
            stage="proposal_completion_text_only",
            max_new_tokens=PROPOSAL_COMPLETION_MAX_NEW_TOKENS,
        )
        return updated_text or cleaned_text

    def _maybe_complete_review_verdict_text_only(
        self,
        full_input_ids,
        full_attention_mask,
        input_len,
        generated_text,
    ):
        cleaned_text = str(generated_text or "").strip()
        if not cleaned_text or not self._should_canonicalize_review_output():
            return cleaned_text
        if not _review_needs_completion(cleaned_text):
            return cleaned_text

        (
            review_prefix_ids,
            review_prefix_mask,
            _,
            _,
            _,
        ) = self._append_forced_suffix_prefix(
            full_input_ids,
            full_attention_mask,
            torch.zeros(1, dtype=torch.float32),
            torch.zeros(1, dtype=torch.long),
            input_len,
            _review_completion_suffix(cleaned_text),
            stage="review_completion_text_only_suffix",
        )
        _, _, _, _, updated_text = self._generate_greedy_continuation(
            review_prefix_ids,
            review_prefix_mask,
            input_len,
            stage="review_completion_text_only",
            max_new_tokens=REVIEW_COMPLETION_MAX_NEW_TOKENS,
        )
        return updated_text or cleaned_text

    def _canonicalize_proposal_output(self, text):
        cleaned_text = str(text or "").strip()
        if not cleaned_text or not self._should_canonicalize_proposal_output():
            return cleaned_text

        lines = _split_nonempty_lines(cleaned_text)
        if not lines:
            return cleaned_text
        final_line_idx = _find_proposal_final_line_index(lines)
        if final_line_idx is None:
            return "\n".join(lines)
        return "\n".join(lines[:final_line_idx + 1])

    def _canonicalize_review_output(self, text):
        cleaned_text = str(text or "").strip()
        if not cleaned_text or not self._should_canonicalize_review_output():
            return cleaned_text

        lines = _split_nonempty_lines(cleaned_text)
        if not lines:
            return cleaned_text

        judgment = _extract_review_judgment_token_anywhere(lines)
        if judgment is None:
            return "\n".join(lines)

        kept_lines = []
        for line in lines:
            inline_reason, inline_judgment = _split_inline_tail_review_judgment(line)
            if inline_judgment is not None:
                if inline_reason:
                    kept_lines.append(inline_reason)
                continue
            normalized = line.strip().lower().rstrip(".")
            if _REVIEW_JUDGMENT_PATTERN.match(line) or _REVIEW_NATURAL_JUDGMENT_PATTERN.match(line):
                continue
            if normalized in {"right", "wrong"}:
                continue
            kept_lines.append(line)

        normalized_lines = [judgment]
        normalized_lines.extend(kept_lines)
        return "\n".join(normalized_lines)

    def _postprocess_generated_text(self, text):
        cleaned_text = str(text or "").strip()
        if not cleaned_text:
            return cleaned_text
        if self._should_canonicalize_proposal_output():
            return self._canonicalize_proposal_output(cleaned_text)
        if self._should_canonicalize_review_output():
            return self._canonicalize_review_output(cleaned_text)
        return cleaned_text

    def _resolve_sample_candidate_chunk_size(self):
        chunk_size = int(GSPO_SAMPLE_CANDIDATE_CHUNK_SIZE)
        if chunk_size <= 0:
            return int(self.G)
        return max(1, chunk_size)

    def _resolve_num_greedy_candidates(self):
        return max(0, min(int(GSPO_NUM_GREEDY_CANDIDATES or 0), int(self.G)))

    def _build_candidate_generation_plan(self):
        chunk_size = self._resolve_sample_candidate_chunk_size()
        greedy_remaining = self._resolve_num_greedy_candidates()
        sampled_remaining = max(0, int(self.G) - greedy_remaining)
        plan = []

        while greedy_remaining > 0:
            plan.append({
                "stage": "act/generate_greedy",
                "num_return_sequences": 1,
                "do_sample": False,
                "generation_mode": "greedy",
            })
            greedy_remaining -= 1

        while sampled_remaining > 0:
            current_chunk_size = min(chunk_size, sampled_remaining)
            plan.append({
                "stage": "act/generate_sampled",
                "num_return_sequences": current_chunk_size,
                "do_sample": True,
                "generation_mode": "sampled",
            })
            sampled_remaining -= current_chunk_size

        return plan

    def _sanitize_trainable_params(self, stage):
        bad_param_tensors = 0
        bad_param_values = 0
        params_with_bad_values = []

        with torch.no_grad():
            for param in self.trainable_params:
                data = param.data
                if not data.is_floating_point():
                    continue
                finite_mask = torch.isfinite(data)
                if bool(finite_mask.all()):
                    continue

                bad_param_tensors += 1
                bad_param_values += int((~finite_mask).sum().item())
                param.data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                params_with_bad_values.append(param)

        for param in params_with_bad_values:
            self.optimizer.state.pop(param, None)

        if bad_param_tensors > 0:
            self._log_numeric_guard(
                f"{stage}:params",
                f"{stage} 检测到 {bad_param_tensors} 个参数张量含非有限值，共 {bad_param_values} 个元素；已替换为 0 并清空对应优化器状态。",
            )
        return bad_param_tensors

    def _sanitize_optimizer_state(self, stage):
        bad_state_tensors = 0
        bad_state_values = 0
        cleared_states = 0

        for _, state in list(self.optimizer.state.items()):
            clear_state = False
            for value in state.values():
                if not torch.is_tensor(value) or not value.is_floating_point():
                    continue
                finite_mask = torch.isfinite(value)
                if bool(finite_mask.all()):
                    continue
                clear_state = True
                bad_state_tensors += 1
                bad_state_values += int((~finite_mask).sum().item())

            if clear_state:
                state.clear()
                cleared_states += 1

        if cleared_states > 0:
            self._log_numeric_guard(
                f"{stage}:optimizer_state",
                f"{stage} 检测到 {bad_state_tensors} 个 Adam 状态张量含非有限值，共 {bad_state_values} 个元素；已清空 {cleared_states} 组状态。",
            )
        return cleared_states

    def _collect_grad_anomalies(self):
        bad_grad_tensors = 0
        bad_grad_values = 0

        for param in self.trainable_params:
            grad = param.grad
            if grad is None or not grad.is_floating_point():
                continue
            finite_mask = torch.isfinite(grad)
            if bool(finite_mask.all()):
                continue
            bad_grad_tensors += 1
            bad_grad_values += int((~finite_mask).sum().item())
        return bad_grad_tensors, bad_grad_values

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

        # 这里只需要 logits 来手工计算序列 log-prob。
        # 不把 labels 传给底模，避免模型内部再额外走一遍 loss/shift_logits 路径，
        # 尤其是 Phi-3 float32 下会白白放大一次峰值显存。
        # 同时显式关闭 KV cache；训练时这条前向不需要 past_key_values。
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = self._sanitize_logits_tensor(outputs.logits, stage="gspo_forward")
        # 标准的“左移 logits、右移 labels”写法：
        # shift_logits[:, t] 对应预测 shift_labels[:, t]
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        if not bool(torch.isfinite(log_probs).all()):
            non_finite_count = int((~torch.isfinite(log_probs)).sum().item())
            self._log_numeric_guard(
                "gspo_forward:log_probs",
                f"gspo_forward 发现 {non_finite_count} 个非有限 log_probs，已替换为 0。",
            )
            log_probs = torch.nan_to_num(log_probs, nan=0.0, posinf=0.0, neginf=0.0)

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

    def _collate_cached_candidates(self, candidates):
        """
        把缓存的候选序列拼成一个 batch，一次前向完成 GSPO 比率计算。
        """
        if not candidates:
            raise ValueError("没有可拼接的 candidates。")

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        batch_size = len(candidates)
        max_seq_len = max(int(cand["input_ids"].shape[1]) for cand in candidates)

        input_ids = torch.full(
            (batch_size, max_seq_len),
            pad_token_id,
            dtype=candidates[0]["input_ids"].dtype,
        )
        attention_mask = torch.zeros(
            (batch_size, max_seq_len),
            dtype=candidates[0]["attention_mask"].dtype,
        )
        labels = torch.full(
            (batch_size, max_seq_len),
            -100,
            dtype=candidates[0]["labels"].dtype,
        )
        old_log_probs = torch.empty(
            batch_size,
            dtype=candidates[0]["old_log_prob"].dtype,
        )
        old_seq_lens = torch.empty(
            batch_size,
            dtype=candidates[0]["old_seq_len"].dtype,
        )

        for row_idx, cand in enumerate(candidates):
            seq_len = int(cand["input_ids"].shape[1])
            input_ids[row_idx, :seq_len] = cand["input_ids"][0]
            attention_mask[row_idx, :seq_len] = cand["attention_mask"][0]
            labels[row_idx, :seq_len] = cand["labels"][0]
            old_log_probs[row_idx] = cand["old_log_prob"].reshape(-1)[0]
            old_seq_lens[row_idx] = cand["old_seq_len"].reshape(-1)[0]

        return input_ids, attention_mask, labels, old_log_probs, old_seq_lens

    def _resolve_update_candidate_chunk_size(self, num_candidates):
        if num_candidates <= 0:
            return 1
        configured = int(GSPO_UPDATE_CANDIDATE_CHUNK_SIZE or 0)
        if configured <= 0:
            return num_candidates
        return max(1, min(configured, num_candidates))

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
        if not bool(torch.isfinite(log_ratio).all()):
            non_finite_count = int((~torch.isfinite(log_ratio)).sum().item())
            self._log_numeric_guard(
                "importance_ratio:log_ratio",
                f"importance_ratio 发现 {non_finite_count} 个非有限 log_ratio，已替换并裁剪。",
            )
        log_ratio = torch.nan_to_num(
            log_ratio,
            nan=0.0,
            posinf=_GSPO_LOG_RATIO_CLAMP,
            neginf=-_GSPO_LOG_RATIO_CLAMP,
        )
        log_ratio = log_ratio.clamp(min=-_GSPO_LOG_RATIO_CLAMP, max=_GSPO_LOG_RATIO_CLAMP)
        s_i = torch.exp(log_ratio)
        if not bool(torch.isfinite(s_i).all()):
            non_finite_count = int((~torch.isfinite(s_i)).sum().item())
            self._log_numeric_guard(
                "importance_ratio:s_i",
                f"importance_ratio 发现 {non_finite_count} 个非有限重要性比率，已替换为 1。",
            )
            s_i = torch.nan_to_num(s_i, nan=1.0, posinf=1.0, neginf=1.0)
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
        inputs = None
        try:
            self._sanitize_trainable_params("act/before_generate")
            prompt_template = prompt_override or self.prompt_template
            prompt = prompt_template.format(context=query_context)
            inputs = build_generation_inputs(prompt, self.tokenizer, self.model).to(device)

            input_len = inputs.input_ids.shape[1]
            candidates = []
            for generation_plan in self._build_candidate_generation_plan():
                current_chunk_size = int(generation_plan["num_return_sequences"])
                generation_output = None
                sequences = None
                transition_scores = None
                generated_token_mask = None
                old_log_probs = None
                old_seq_lens = None
                try:
                    with torch.no_grad():
                        generation_output = self.model.generate(
                            **inputs,
                            **self._build_generate_kwargs(
                                stage=generation_plan["stage"],
                                num_return_sequences=current_chunk_size,
                                return_dict_in_generate=True,
                                output_scores=True,
                                force_do_sample=generation_plan["do_sample"],
                            ),
                        )

                    sequences = generation_output.sequences
                    transition_scores = self.model.compute_transition_scores(
                        sequences,
                        generation_output.scores,
                        normalize_logits=True,
                    )
                    transition_scores, generated_token_mask = self._sanitize_transition_scores(
                        transition_scores,
                        stage="act/transition_scores",
                    )
                    old_log_probs = transition_scores.sum(dim=1)
                    old_seq_lens = generated_token_mask.sum(dim=1)

                    for row_idx in range(sequences.shape[0]):
                        generated_token_count = int(old_seq_lens[row_idx].item())
                        total_len = input_len + generated_token_count
                        full_input_ids = sequences[row_idx, :total_len].unsqueeze(0).cpu()
                        full_attention_mask = torch.ones_like(full_input_ids)
                        full_labels = full_input_ids.clone()
                        full_labels[:, :input_len] = -100
                        old_log_prob = old_log_probs[row_idx:row_idx + 1].detach().cpu()
                        old_seq_len = old_seq_lens[row_idx:row_idx + 1].detach().cpu()
                        generated_text = self.tokenizer.decode(
                            full_input_ids[0, input_len:],
                            skip_special_tokens=True,
                        ).strip()
                        (
                            full_input_ids,
                            full_attention_mask,
                            full_labels,
                            old_log_prob,
                            old_seq_len,
                            generated_text,
                        ) = (
                            self._maybe_complete_review_verdict_candidate(
                                full_input_ids,
                                full_attention_mask,
                                full_labels,
                                old_log_prob,
                                old_seq_len,
                                input_len,
                                generated_text,
                            )
                            if self._should_canonicalize_review_output() else
                            self._maybe_complete_proposal_candidate(
                                full_input_ids,
                                full_attention_mask,
                                full_labels,
                                old_log_prob,
                                old_seq_len,
                                input_len,
                                generated_text,
                            )
                        )
                        generated_text = self._postprocess_generated_text(generated_text)

                        candidates.append({
                            "text": generated_text,
                            "generation_mode": generation_plan["generation_mode"],
                            # 候选缓存到 CPU，避免跨多轮延迟更新时长期占用显存。
                            "input_ids": full_input_ids,
                            "attention_mask": full_attention_mask,
                            "labels": full_labels,
                            "old_log_prob": old_log_prob,
                            "old_seq_len": old_seq_len,
                        })
                finally:
                    del generation_output
                    del sequences
                    del transition_scores
                    del generated_token_mask
                    del old_log_probs
                    del old_seq_lens
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            return candidates
        finally:
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
        texts = self.sample_text_batch(
            [query_context],
            prompt_override=prompt_override,
        )
        return texts[0]

    def sample_text_batch(self, query_contexts, prompt_override=None):
        """
        批量采样多条文本。

        主要用于：
        - comment 候选的“带 comment 后下一次 answer”并行评估
        - worker 常驻 GPU 时，减少多次小 generate 的串行开销
        """
        if not query_contexts:
            return []

        device = self._activate_model()
        self.model.eval()
        try:
            self._sanitize_trainable_params("sample_text_batch/before_generate")
            prompt_template = prompt_override or self.prompt_template
            results = []
            prompts = [
                prompt_template.format(context=query_context)
                for query_context in query_contexts
            ]
            chunk_size = int(GSPO_SAMPLE_TEXT_BATCH_SIZE)
            if chunk_size <= 0:
                chunk_size = len(prompts)

            for chunk_start in range(0, len(prompts), chunk_size):
                chunk_prompts = prompts[chunk_start:chunk_start + chunk_size]
                inputs = build_generation_inputs(chunk_prompts, self.tokenizer, self.model)
                input_lens = inputs["attention_mask"].sum(dim=1).tolist()
                inputs = {key: value.to(device) for key, value in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **self._build_generate_kwargs(
                            stage="sample_text_batch/generate",
                            num_return_sequences=1,
                            return_dict_in_generate=False,
                            output_scores=False,
                        ),
                    )

                for row_idx, input_len in enumerate(input_lens):
                    full_output_ids = outputs[row_idx].unsqueeze(0).cpu()
                    full_attention_mask = torch.ones_like(full_output_ids)
                    generated_ids = full_output_ids[0, int(input_len):]
                    generated_text = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                    ).strip()
                    if self._should_canonicalize_proposal_output():
                        generated_text = self._maybe_complete_proposal_candidate_text_only(
                            full_output_ids,
                            full_attention_mask,
                            int(input_len),
                            generated_text,
                        )
                    elif self._should_canonicalize_review_output():
                        generated_text = self._maybe_complete_review_verdict_text_only(
                            full_output_ids,
                            full_attention_mask,
                            int(input_len),
                            generated_text,
                        )
                    results.append(self._postprocess_generated_text(generated_text))

                del outputs
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return results
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
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        mean_r = rewards.mean()
        std_r = rewards.std().clamp(min=1e-8)
        advantage = (rewards - mean_r) / std_r
        return advantage

    def update_from_cached(self, batch, rewards):
        """
        延迟奖励版本的第二步：针对之前缓存的 candidates，使用现在得到的 rewards 做更新。
        """
        self._activate_model()
        # 这里仍然需要梯度，但不需要 dropout 噪声；
        # 用 eval 模式可让“当前策略”和 rollout 时缓存的旧策略更一致。
        self.model.eval()
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
                    "skip_reason": "zero_signal",
                }
                return avg_reward

            self._sanitize_trainable_params("update/before_forward")
            advantage = self.compute_advantage(rewards)
            if not bool(torch.isfinite(advantage).all()):
                self.last_update = {
                    "avg_reward": avg_reward,
                    "best_reward": rewards[best_idx],
                    "best_text": candidates[best_idx]["text"],
                    "selected_text": batch["selected_text"],
                    "selected_reward": rewards[batch["selected_idx"]],
                    "candidates": [cand["text"] for cand in candidates],
                    "rewards": rewards,
                    "skipped_update": True,
                    "skip_reason": "nonfinite_advantage",
                }
                self._log_numeric_guard(
                    "update:advantage",
                    "update/before_forward 发现非有限 advantage，已跳过本次更新。",
                )
                return avg_reward
            self.optimizer.zero_grad(set_to_none=True)
            total_candidates = len(candidates)
            chunk_size = self._resolve_update_candidate_chunk_size(total_candidates)
            policy_loss_total = 0.0

            for chunk_start in range(0, total_candidates, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_candidates)
                chunk_candidates = candidates[chunk_start:chunk_end]
                chunk_advantage = advantage[chunk_start:chunk_end]
                (
                    input_ids,
                    attention_mask,
                    labels,
                    old_log_probs,
                    old_seq_lens,
                ) = self._collate_cached_candidates(chunk_candidates)
                s_i, _ = self.compute_importance_ratio(
                    input_ids,
                    attention_mask,
                    labels,
                    old_log_prob=old_log_probs,
                    old_seq_len=old_seq_lens,
                )
                s_i_clipped = torch.clamp(s_i, 1 - self.clip_eps, 1 + self.clip_eps)
                surr1 = s_i * chunk_advantage
                surr2 = s_i_clipped * chunk_advantage
                chunk_objective = torch.min(surr1, surr2)
                chunk_loss = -chunk_objective.sum() / max(total_candidates, 1)

                if not torch.isfinite(chunk_loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    self.last_update = {
                        "avg_reward": avg_reward,
                        "best_reward": rewards[best_idx],
                        "best_text": candidates[best_idx]["text"],
                        "selected_text": batch["selected_text"],
                        "selected_reward": rewards[batch["selected_idx"]],
                        "candidates": [cand["text"] for cand in candidates],
                        "rewards": rewards,
                        "skipped_update": True,
                        "skip_reason": "nonfinite_policy_loss",
                    }
                    self._log_numeric_guard(
                        "update:policy_loss",
                        "update/before_backward 发现非有限 policy_loss，已跳过本次更新。",
                    )
                    return avg_reward

                chunk_loss.backward()
                policy_loss_total += float(chunk_loss.detach().cpu().item())

                del input_ids
                del attention_mask
                del labels
                del old_log_probs
                del old_seq_lens
                del s_i
                del s_i_clipped
                del surr1
                del surr2
                del chunk_objective
                del chunk_loss
                del chunk_advantage

            bad_grad_tensors, bad_grad_values = self._collect_grad_anomalies()
            if bad_grad_tensors > 0:
                self.optimizer.zero_grad(set_to_none=True)
                self._sanitize_optimizer_state("update/nonfinite_grad")
                self.last_update = {
                    "avg_reward": avg_reward,
                    "best_reward": rewards[best_idx],
                    "best_text": candidates[best_idx]["text"],
                    "selected_text": batch["selected_text"],
                    "selected_reward": rewards[batch["selected_idx"]],
                    "candidates": [cand["text"] for cand in candidates],
                    "rewards": rewards,
                    "skipped_update": True,
                    "skip_reason": "nonfinite_grad",
                }
                self._log_numeric_guard(
                    "update:gradients",
                    f"update/backward 检测到 {bad_grad_tensors} 个梯度张量含非有限值，共 {bad_grad_values} 个元素；已跳过本次更新。",
                )
                return avg_reward

            grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, GSPO_MAX_GRAD_NORM)
            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                self._sanitize_optimizer_state("update/nonfinite_grad_norm")
                self.last_update = {
                    "avg_reward": avg_reward,
                    "best_reward": rewards[best_idx],
                    "best_text": candidates[best_idx]["text"],
                    "selected_text": batch["selected_text"],
                    "selected_reward": rewards[batch["selected_idx"]],
                    "candidates": [cand["text"] for cand in candidates],
                    "rewards": rewards,
                    "skipped_update": True,
                    "skip_reason": "nonfinite_grad_norm",
                }
                self._log_numeric_guard(
                    "update:grad_norm",
                    "update/clip_grad_norm 检测到非有限梯度范数，已跳过本次更新。",
                )
                return avg_reward

            self.optimizer.step()
            bad_params_after_step = self._sanitize_trainable_params("update/after_step")
            bad_states_after_step = self._sanitize_optimizer_state("update/after_step")

            self.last_update = {
                "avg_reward": avg_reward,
                "best_reward": rewards[best_idx],
                # 仅诊断：batch 路径里真正执行的仍是 selected_text。
                "best_text": candidates[best_idx]["text"],
                "selected_text": batch["selected_text"],
                "selected_reward": rewards[batch["selected_idx"]],
                "candidates": [cand["text"] for cand in candidates],
                "rewards": rewards,
                "policy_loss_total": policy_loss_total,
                "sanitized_after_step": bool(bad_params_after_step or bad_states_after_step),
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
