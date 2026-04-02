import os
import sys
import gc
import time
import numpy as np
import torch
from tqdm import tqdm

# 确保能找到src目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import *
from src.data_loader import load_gsm8k_motac, load_gsm8k_splits, compute_accuracy, extract_pred_num
from src.model_loader import generate_response, get_gpt2
from src.gspo_verl import GSPOAgentPolicy
from src.cfr_core import CFRBehaviorSelector
from src.verifier import OnlineAnswerVerifier
from src.metrics_logger import (
    init_logs,
    log_single,
    log_polling,
    log_single_gspo,
    log_dual_gspo,
    log_middle_layer,
    log_middle_layer_no_silent,
    log_three_layer,
    log_three_layer_no_silent,
    load_logs,
)
from src.plotter import plot_accuracy_comparison, plot_three_layer_details

# ============================================================
# 这份 run_all.py 是整个实验的“总调度器”。
#
# 如果你刚接触 Python，可以先抓住下面这条主线：
# 1. 先定义一些小工具函数（拼上下文、算准确率、算奖励）
# 2. 再分别定义多组实验函数
# 3. 最后在 main() 里按统一的数据划分顺序运行 8 组实验
#
# 你可以把它理解成：
# - 上半部分：实验需要的“零件”
# - 中间部分：6 种实验怎么跑
# - 最下面：总开关 main()
#
# 这份文件里有几处“和之前相比很重要的改动”，这里先提前说明：
#
# 改动1：按“样本优先”运行，而不是按“轮次优先”运行
# - 改前容易被理解成：先把 50 条样本都跑第 1 轮，再都跑第 2 轮……
# - 改后现在是：一条样本会连续跑完 5 轮，然后才进入下一条样本
#
# 改动2：每一轮输出都会真的拼回上下文
# - 改前如果不把历史输出写回，后面轮次其实看不到前面轮次说了什么
# - 改后 history 会不断追加，模拟真实多轮对话
#
# 改动3：评论轮不输出新答案时，准确率沿用上一轮答案
# - 这符合你的实验设定：评论轮本身不一定给最终答案
# - 所以第 2/4 轮如果没有新答案，就用上一轮已有答案来统计“当前答案准确率”
#
# 改动4：GSPO 奖励改成“延迟更新”
# - comment 先只缓存，不立刻更新
# - 等这个段落里真的出现 answer，再回头统一给这段里的 step 分配奖励
#
# 改动5：日志里同时保留两个指标
# - accuracy: 当前这一轮手里那份答案是否正确（主指标）
# - best_so_far_accuracy: 到当前轮为止，历史上是否曾经答对过（辅助指标）
# ============================================================

def cleanup_cuda_memory(clear_cuda_cache=True):
    """
    手动回收显存。

    Python 里对象删除后，不一定会立刻把 GPU 显存还回去，
    所以这里至少会做一件事：
    - gc.collect(): 让 Python 垃圾回收器尽快处理无引用对象

    是否再做第二件事，由 clear_cuda_cache 控制：
    - torch.cuda.empty_cache(): 把 PyTorch 缓存的空闲显存释放回 allocator

    这样在“已知 GPU runtime 可能卡在收尾”的路径里，
    我们可以只做 Python 层回收，暂时跳过全局 empty_cache。
    """
    gc.collect()
    if clear_cuda_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()

def teardown_three_layer_runtime(agents, outer_cfr, middle_cfr, verifier):
    """
    三层实验的保守收尾路径。

    目标有两个：
    1. 把可能卡住的位置拆成更细的阶段打印，便于精确定位
    2. 避免在最终收尾时再次触发全局 empty_cache，把卡点尽量缩小到
       “哪一个 agent / policy 的下放或对象释放”。
    """
    print("[三层teardown] start", flush=True)

    for agent_idx, agent_bundle in enumerate(agents):
        for policy_name in ("pi0", "pi1", "pi2"):
            policy = agent_bundle.get(policy_name)
            if policy is None:
                continue
            print(
                f"[三层teardown] offload agent={agent_idx} policy={policy_name} start",
                flush=True,
            )
            policy.prepare_for_teardown()
            print(
                f"[三层teardown] offload agent={agent_idx} policy={policy_name} done",
                flush=True,
            )

    print("[三层teardown] drop_refs start", flush=True)
    agents = None
    outer_cfr = None
    middle_cfr = None
    verifier = None
    print("[三层teardown] drop_refs done", flush=True)

    print("[三层teardown] gc_only_cleanup start", flush=True)
    cleanup_cuda_memory(clear_cuda_cache=False)
    print("[三层teardown] gc_only_cleanup done", flush=True)
    print("[三层teardown] finished", flush=True)

    return agents, outer_cfr, middle_cfr, verifier

def is_exact_match(pred_num, gt):
    """
    判断一个预测是否与标准答案一致。

    这里和 data_loader.compute_accuracy() 保持同一判定标准：
    - 都允许 1e-3 以内的浮点误差
    """
    return pred_num is not None and gt is not None and abs(pred_num - gt) < 1e-3

def render_history_turn(turn):
    """
    把结构化 turn 渲染成统一文本块。

    一个 turn 是一个字典，例如：
        {
            "round": 3,
            "speaker": "solver",
            "kind": "answer",
            "text": "......"
        }
    """
    clean_text = turn["text"].strip()
    return f"[Round {turn['round']} | {turn['speaker']} | {turn['kind']}]\n{clean_text}"

def select_relevant_turns(history):
    """
    从完整历史里抽取“当前轮真正有帮助的那部分”。

    统一策略：
    - 始终保留最近一次 answer
    - 再保留这次 answer 之后的最近若干条 comment
    - 如果还没有 answer，就只保留最近若干条 comment

    这样所有实验都共用同一套上下文裁剪规则，比较更公平。
    """
    if not history:
        return []

    last_answer_idx = None
    for idx in range(len(history) - 1, -1, -1):
        if history[idx]["kind"] == "answer":
            last_answer_idx = idx
            break

    if last_answer_idx is None:
        comments = [turn for turn in history if turn["kind"] == "comment"]
        return comments[-STRUCTURED_CONTEXT_MAX_COMMENTS:]

    selected = [history[last_answer_idx]]
    trailing_comments = [
        turn
        for turn in history[last_answer_idx + 1:]
        if turn["kind"] == "comment"
    ]
    selected.extend(trailing_comments[-STRUCTURED_CONTEXT_MAX_COMMENTS:])
    return selected

def build_context(question, history):
    """
    把“题目”和“结构化历史”拼成这轮真正喂给模型的上下文。

    Python 语法说明：
    - if history: 表示“如果 history 非空”
    - f"...{question}..." 是 f-string，意思是在字符串里直接插入变量值
    """
    if isinstance(history, str):
        if history:
            return f"{question}\n{history}"
        return question

    if history:
        relevant_turns = select_relevant_turns(history)
        if not relevant_turns:
            return question

        latest_answer = None
        recent_comments = []
        for turn in relevant_turns:
            if turn["kind"] == "answer":
                latest_answer = turn
            else:
                recent_comments.append(turn)

        parts = [f"[Original Question]\n{question}"]
        if latest_answer is not None:
            parts.append(
                "[Incumbent Answer]\n"
                f"{render_history_turn(latest_answer)}"
            )
        if recent_comments:
            parts.append(
                "[Recent Comments]\n"
                + "\n\n".join(render_history_turn(turn) for turn in recent_comments)
            )
        return "\n\n".join(parts)
    return question

def append_round_output(history, rnd, speaker, text, kind):
    """
    把某一轮输出以“结构化 turn”的形式追加到 history。

    改动前：
    - history 是一个不断变长的大字符串

    改动后：
    - history 是一个列表
    - 列表里的每一项都明确记录 round / speaker / kind / text

    这样 build_context() 就能统一做“只取最近相关 answer/comment”，
    不需要再从脏字符串里硬拆结构。
    """
    turn = {
        "round": rnd,
        "speaker": speaker,
        "kind": kind,
        "text": text.strip(),
    }
    if history:
        return [*history, turn]
    return [turn]

def update_last_pred(last_pred, current_pred, should_use_current_answer):
    """
    更新“当前手里那份答案”。

    这是理解你实验统计逻辑的关键函数之一。

    改动前容易出现的误解：
    - 每一轮都必须有一个新的答案

    改动后现在的逻辑是：
    - 如果这一轮是 answer 轮，并且抽取到了数字答案，就更新 last_pred
    - 如果这一轮是 comment 轮，或者没有抽取到新答案，就保留上一轮答案

    所以：
    - 第 1/3/5 轮常常会刷新答案
    - 第 2/4 轮通常沿用前一轮答案
    """
    if should_use_current_answer and current_pred is not None:
        return current_pred
    return last_pred

def compute_round_accuracies(round_preds, round_gts):
    """
    计算每一轮的准确率。

    Python 语法说明：
    - 这里用了“列表推导式”（list comprehension）
    - zip(round_preds, round_gts) 会把两个列表按位置一一配对
    - for preds, gts in ... 的意思是：每次取出一轮的预测和对应真值
    """
    return [
        compute_accuracy(preds, gts)
        for preds, gts in zip(round_preds, round_gts)
    ]

def format_accuracy_snapshot(accs):
    return " | ".join(
        f"R{idx+1}={acc:.4f}"
        for idx, acc in enumerate(accs)
    )

def compute_best_so_far_accuracies(best_round_hits):
    """
    计算辅助指标 best-so-far accuracy。

    它回答的问题不是：
    - “当前这一轮答案对不对？”

    而是：
    - “到当前这一轮为止，这条样本历史上有没有至少答对过一次？”

    所以如果某条样本：
    - 第 1 轮答对
    - 第 3 轮改错
    - 第 5 轮还是错

    那么：
    - 当前 accuracy 可能下降
    - 但 best-so-far 依然记这条样本“曾经成功过”
    """
    return [
        (sum(hits) / len(hits)) if hits else 0.0
        for hits in best_round_hits
    ]

def maybe_report_accuracy(exp_name, processed, total, round_preds, round_gts, best_round_hits=None):
    """
    控制“什么时候打印一次中途统计”。

    这里不是每处理 1 条样本都打印，
    而是在这些节点打印：
    - 第 1 条
    - 每 5 条
    - 最后一条

    这样终端不会被刷爆，但又能看到实时趋势。
    """
    if processed == 1 or processed == total or processed % 5 == 0:
        accs = compute_round_accuracies(round_preds, round_gts)
        message = f"[{exp_name}] 已处理 {processed}/{total} 条 | 当前准确率 {format_accuracy_snapshot(accs)}"
        if best_round_hits is not None:
            best_accs = compute_best_so_far_accuracies(best_round_hits)
            message += f" | Best-so-far {format_accuracy_snapshot(best_accs)}"
        print(message)

def build_split_size_message(train_data, val_data, test_data):
    return (
        f"train={len(train_data)} | "
        f"val={len(val_data)} | "
        f"test={len(test_data)}"
    )

def reward_from_pred(pred_num, gt):
    """
    混合奖励：exact match 为主，dense reward 为辅。

    为什么这么改：
    - final accuracy 看的是真正答对没有
    - 但如果训练里只用 0/1 reward，那么两个都答错的候选没有任何区分信号
    - 所以这里采用折中：
      1. exact match 决定主方向
      2. dense reward 只负责在“都答错时”提供细粒度比较

    公式：
        reward = w_exact * exact_match + w_dense * dense_score
    """
    if pred_num is None or gt is None:
        return REWARD_MISSING_PENALTY

    exact_reward = 1.0 if is_exact_match(pred_num, gt) else 0.0
    scale = max(abs(gt), 1.0)
    rel_error = abs(pred_num - gt) / scale
    dense_reward = 1.0 / (1.0 + rel_error)
    return REWARD_EXACT_WEIGHT * exact_reward + REWARD_DENSE_WEIGHT * dense_reward

def reward_delta_from_preds(previous_pred, current_pred, gt):
    """
    计算“新答案相对 incumbent 的增量收益”。
    """
    previous_reward = reward_from_pred(previous_pred, gt)
    current_reward = reward_from_pred(current_pred, gt)
    return current_reward - previous_reward

def rollout_terminal_absolute_reward(pred_num, incumbent_pred, gt):
    """
    rollout 的短程“绝对效用”。

    规则：
    - 有解析答案：返回该答案的绝对 reward
    - 无解析答案但已有 incumbent：视为维持 incumbent，返回 incumbent 的绝对 reward
    - 无解析答案且没有 incumbent：给重罚，显式打击“整段 rollout 最后仍无答案”
    """
    if pred_num is None:
        if incumbent_pred is not None:
            return reward_from_pred(incumbent_pred, gt)
        return ROLLOUT_NO_ANSWER_PENALTY
    return reward_from_pred(pred_num, gt)

def rollout_terminal_delta_reward(pred_num, incumbent_pred, gt):
    """
    rollout 的短程“相对 incumbent 的增量效用”。

    规则：
    - 有解析答案：reward(answer) - reward(incumbent)
    - 无解析答案但已有 incumbent：记 0，表示没有改进也没有破坏 incumbent
    - 无解析答案且没有 incumbent：给重罚
    """
    if pred_num is None:
        if incumbent_pred is not None:
            return 0.0
        return ROLLOUT_NO_ANSWER_PENALTY
    return reward_delta_from_preds(incumbent_pred, pred_num, gt)

def is_late_round(rnd):
    """
    进入后期轮次后，系统应当更偏向 stabilize。

    当前定义：
    - 当回合位置达到总轮数的 80% 及以后，认为进入 late stage
    - 5 轮实验里对应第 4、5 轮
    """
    return (rnd / max(NUM_ROUNDS, 1)) >= 0.8

def compute_phase(verifier, question, history, incumbent_text, rnd):
    """
    把外层状态压缩成两个 phase：
    - search
    - stabilize

    这里不再使用启发式关键词，也不再把状态切得很碎。
    phase 只由两部分决定：
    1. verifier scorer 对当前 incumbent 的绝对打分
    2. 当前是否已经进入后期轮次
    """
    keep_prob, _ = verifier.predict_keep_prob(
        question,
        history,
        incumbent_text,
    )
    late_flag = 1.0 if is_late_round(rnd) else 0.0
    stability_score = keep_prob + VERIFIER_STABILIZE_ALPHA * late_flag
    phase = "stabilize" if stability_score >= VERIFIER_PHASE_THRESHOLD else "search"
    return phase, keep_prob, stability_score

def build_outer_state(phase):
    return (
        "outer",
        f"phase={phase}",
    )

def build_middle_state(selected_agent, phase):
    return (
        "middle",
        f"agent={selected_agent}",
        f"phase={phase}",
    )

def get_middle_allowed_actions(rnd, incumbent_pred, allow_silent=True):
    """
    中层动作空间：
    - silent
    - comment
    - answer

    只保留“物理上必须有”的约束：
    - 最后一轮如果还没有 incumbent，必须 answer，避免整条样本没有答案

    其余情况下，不再用手工规则把策略锁死给某个动作，
    而是让 middle CFR 在 phase 条件下自己学。
    """
    if incumbent_pred is None and rnd <= NO_INCUMBENT_FORCE_ANSWER_ROUNDS:
        return [MIDDLE_ACTION_ANSWER]
    if rnd == NUM_ROUNDS and incumbent_pred is None:
        return [MIDDLE_ACTION_ANSWER]
    if allow_silent:
        return [
            MIDDLE_ACTION_SILENT,
            MIDDLE_ACTION_COMMENT,
            MIDDLE_ACTION_ANSWER,
        ]
    return [
        MIDDLE_ACTION_COMMENT,
        MIDDLE_ACTION_ANSWER,
    ]

def _candidate_consensus_counts(candidates):
    counts = {}
    for cand in candidates:
        pred = extract_pred_num(cand["text"])
        if pred is None:
            continue
        key = round(float(pred), 6)
        counts[key] = counts.get(key, 0) + 1
    return counts

def set_sampled_batch_selected_candidate(agent, batch, selected_idx):
    selected_idx = max(0, min(int(selected_idx), len(batch["candidates"]) - 1))
    batch["selected_idx"] = selected_idx
    batch["selected_text"] = batch["candidates"][selected_idx]["text"]
    if getattr(agent, "last_update", None) is not None:
        agent.last_update["selected_text"] = batch["selected_text"]
        agent.last_update["best_text"] = batch["selected_text"]
    return selected_idx

def select_answer_candidate_index(
    verifier,
    question,
    history,
    incumbent_text,
    incumbent_pred,
    batch,
):
    """
    在同一批候选里做推理时可用的候选选择。

    不使用 reward / 真值，只使用：
    - 是否能抽取出数字答案
    - verifier 对“是否值得接管 incumbent”的概率估计
    - 是否保持 incumbent 不变
    - 同批候选里的数值一致性（出现次数）
    """
    candidates = batch["candidates"]
    if not candidates:
        return 0, None

    consensus_counts = _candidate_consensus_counts(candidates)
    best_idx = 0
    best_meta = None
    accept_gate = verifier.get_accept_gate(question, history, incumbent_text)
    accept_barrier = accept_gate["barrier"]
    gate_keep_prob = accept_gate["keep_prob"]

    for idx, cand in enumerate(candidates):
        candidate_pred = extract_pred_num(cand["text"])
        parseable = candidate_pred is not None
        consensus_count = 0
        accept_prob = 0.0
        accept_margin = 0.0
        keep_bonus = 0.0
        change_penalty = 0.0

        if parseable:
            consensus_count = consensus_counts.get(round(float(candidate_pred), 6), 0)
            if incumbent_text is not None:
                accept_prob, _ = verifier.predict_accept_prob(
                    question,
                    history,
                    incumbent_text,
                    cand["text"],
                )
                accept_margin = accept_prob - accept_barrier
                if incumbent_pred is not None:
                    if abs(candidate_pred - incumbent_pred) <= 1e-6:
                        keep_bonus = ANSWER_CANDIDATE_KEEP_BONUS
                    else:
                        change_penalty = ANSWER_CANDIDATE_CHANGE_PENALTY

        score = (
            1 if parseable else 0,
            accept_margin + keep_bonus - change_penalty,
            consensus_count,
            accept_prob,
            -idx,
        )
        if best_meta is None or score > best_meta["score"]:
            best_idx = idx
            best_meta = {
                "score": score,
                "candidate_pred": candidate_pred,
                "consensus_count": consensus_count,
                "accept_prob": accept_prob,
                "accept_margin": accept_margin,
                "accept_barrier": accept_barrier,
                "gate_keep_prob": gate_keep_prob,
                "keep_bonus": keep_bonus,
                "change_penalty": change_penalty,
            }

    return best_idx, best_meta

def maybe_print_answer_candidate_selection_debug(sample_idx, rnd, batch, selected_idx, meta):
    if not THREE_LAYER_DEBUG_PRINT or meta is None:
        return

    selected_pred_str = "None" if meta["candidate_pred"] is None else f"{meta['candidate_pred']:.4f}"
    print(
        f"[候选选择] sample={sample_idx} rnd={rnd} selected_idx={selected_idx} "
        f"selected_pred={selected_pred_str} consensus={meta['consensus_count']} "
        f"accept_prob={meta['accept_prob']:.4f} accept_margin={meta['accept_margin']:.4f} "
        f"accept_barrier={meta['accept_barrier']:.4f} gate_keep_prob={meta['gate_keep_prob']:.4f} "
        f"keep_bonus={meta['keep_bonus']:.4f} change_penalty={meta['change_penalty']:.4f}"
    )

def get_search_answer_constraint_profile(phase, keep_prob, incumbent_pred, rnd):
    """
    基于“当前 incumbent 有多不稳”动态调整 search 阶段的动作约束。

    只用推理时可见信号：
    - keep_prob
    - incumbent 是否存在
    - 当前轮次

    目标：
    - incumbent 缺失或明显不稳时，提高 answer 尝试率
    - 同时保留 comment 探索，不把策略锁死成纯答题
    """
    answer_floor = SEARCH_MIN_ANSWER_PROB
    silent_cap = SEARCH_MAX_SILENT_PROB

    if phase != "search":
        return answer_floor, silent_cap

    if incumbent_pred is None:
        answer_floor = max(answer_floor, SEARCH_NO_INCUMBENT_MIN_ANSWER_PROB)
        silent_cap = min(silent_cap, SEARCH_NO_INCUMBENT_MAX_SILENT_PROB)
    elif keep_prob >= SEARCH_STABLE_KEEP_PROB and rnd < NUM_ROUNDS:
        answer_floor = min(answer_floor, SEARCH_STABLE_MAX_ANSWER_PROB)
        silent_cap = min(silent_cap, SEARCH_MAX_SILENT_PROB)
    elif keep_prob <= SEARCH_LOW_KEEP_PROB:
        answer_floor = max(answer_floor, SEARCH_UNSTABLE_MIN_ANSWER_PROB)
        silent_cap = min(silent_cap, SEARCH_UNSTABLE_MAX_SILENT_PROB)
    elif keep_prob < SEARCH_RELAX_KEEP_PROB:
        mix = (
            (SEARCH_RELAX_KEEP_PROB - keep_prob)
            / max(SEARCH_RELAX_KEEP_PROB - SEARCH_LOW_KEEP_PROB, 1e-6)
        )
        answer_floor = max(
            answer_floor,
            SEARCH_MIN_ANSWER_PROB + mix * (
                SEARCH_UNSTABLE_MIN_ANSWER_PROB - SEARCH_MIN_ANSWER_PROB
            ),
        )
        silent_cap = min(
            silent_cap,
            SEARCH_MAX_SILENT_PROB - mix * (
                SEARCH_MAX_SILENT_PROB - SEARCH_UNSTABLE_MAX_SILENT_PROB
            ),
        )

    if phase == "search" and is_late_round(rnd):
        answer_floor = min(1.0, answer_floor + SEARCH_LATE_ROUND_ANSWER_BONUS)
        silent_cap = min(silent_cap, SEARCH_UNSTABLE_MAX_SILENT_PROB)

    return float(answer_floor), float(max(0.0, silent_cap))

def apply_search_middle_strategy_constraints(
    strategy,
    allowed_actions,
    phase,
    keep_prob,
    incumbent_pred,
    rnd,
):
    """
    仅在 search 状态下，对中层策略做两类约束：
    - 给 pi0 / pi1 保留最低探索概率
    - 不直接改 CFR 本体，只在采样/期望时使用这份约束后的策略
    """
    adjusted = np.zeros(CFR_NUM_ACTIONS, dtype=np.float64)
    allowed_set = set(allowed_actions)
    if not allowed_set:
        adjusted[MIDDLE_ACTION_SILENT] = 1.0
        return adjusted

    if phase != "search":
        adjusted[:] = strategy
        return adjusted

    floors = np.zeros(CFR_NUM_ACTIONS, dtype=np.float64)
    answer_floor, silent_cap = get_search_answer_constraint_profile(
        phase,
        keep_prob,
        incumbent_pred,
        rnd,
    )
    if MIDDLE_ACTION_COMMENT in allowed_set:
        floors[MIDDLE_ACTION_COMMENT] = SEARCH_MIN_COMMENT_PROB
    if MIDDLE_ACTION_ANSWER in allowed_set:
        floors[MIDDLE_ACTION_ANSWER] = answer_floor
    if (
        phase == "search"
        and incumbent_pred is not None
        and keep_prob >= SEARCH_STABLE_KEEP_PROB
        and rnd < NUM_ROUNDS
        and MIDDLE_ACTION_COMMENT in allowed_set
    ):
        floors[MIDDLE_ACTION_COMMENT] = max(
            floors[MIDDLE_ACTION_COMMENT],
            SEARCH_STABLE_MIN_COMMENT_PROB,
        )

    floor_sum = floors.sum()
    if floor_sum >= 1.0:
        adjusted[:] = floors / floor_sum
        return adjusted

    base = np.array(strategy, dtype=np.float64)
    base[~np.isfinite(base)] = 0.0
    base[base < 0.0] = 0.0
    for act in range(CFR_NUM_ACTIONS):
        if act not in allowed_set:
            base[act] = 0.0

    if MIDDLE_ACTION_SILENT in allowed_set:
        base[MIDDLE_ACTION_SILENT] = min(
            base[MIDDLE_ACTION_SILENT],
            silent_cap,
        )

    adjusted[:] = floors
    reserved_mass = adjusted.sum()
    remaining_mass = max(1.0 - reserved_mass, 0.0)

    extra = np.zeros(CFR_NUM_ACTIONS, dtype=np.float64)
    for act in allowed_set:
        if act == MIDDLE_ACTION_SILENT:
            extra[act] = base[act]
        else:
            extra[act] = max(base[act] - floors[act], 0.0)

    extra_total = extra.sum()
    if extra_total > 0.0:
        adjusted += remaining_mass * (extra / extra_total)
    else:
        active_actions = [act for act in allowed_set if act != MIDDLE_ACTION_SILENT]
        if not active_actions:
            active_actions = list(allowed_set)
        share = remaining_mass / max(len(active_actions), 1)
        for act in active_actions:
            adjusted[act] += share

    if MIDDLE_ACTION_ANSWER in allowed_set and incumbent_pred is not None and keep_prob >= SEARCH_STABLE_KEEP_PROB and rnd < NUM_ROUNDS:
        if adjusted[MIDDLE_ACTION_ANSWER] > SEARCH_STABLE_MAX_ANSWER_PROB:
            overflow = adjusted[MIDDLE_ACTION_ANSWER] - SEARCH_STABLE_MAX_ANSWER_PROB
            adjusted[MIDDLE_ACTION_ANSWER] = SEARCH_STABLE_MAX_ANSWER_PROB
            redistribute_targets = [act for act in allowed_set if act == MIDDLE_ACTION_COMMENT]
            if redistribute_targets:
                share = overflow / len(redistribute_targets)
                for act in redistribute_targets:
                    adjusted[act] += share

    if MIDDLE_ACTION_SILENT in allowed_set and adjusted[MIDDLE_ACTION_SILENT] > silent_cap:
        overflow = adjusted[MIDDLE_ACTION_SILENT] - silent_cap
        adjusted[MIDDLE_ACTION_SILENT] = silent_cap
        redistribute_targets = [
            act for act in allowed_set
            if act in (MIDDLE_ACTION_COMMENT, MIDDLE_ACTION_ANSWER)
        ]
        if redistribute_targets:
            share = overflow / len(redistribute_targets)
            for act in redistribute_targets:
                adjusted[act] += share

    total = adjusted.sum()
    if total > 0.0:
        adjusted /= total
    else:
        fallback_action = next(iter(allowed_set))
        adjusted[fallback_action] = 1.0
    return adjusted

def classify_round_transition(previous_pred, current_pred, gt):
    previous_reward = reward_from_pred(previous_pred, gt)
    current_reward = reward_from_pred(current_pred, gt)
    has_previous = previous_pred is not None

    improved = has_previous and current_reward > previous_reward + 1e-9
    degraded = has_previous and current_reward < previous_reward - 1e-9
    stalled_wrong = (
        has_previous
        and current_reward <= previous_reward + 1e-9
        and not is_exact_match(current_pred, gt)
    )
    preserved_correct = (
        has_previous
        and is_exact_match(previous_pred, gt)
        and is_exact_match(current_pred, gt)
    )
    return {
        "improved": improved,
        "degraded": degraded,
        "stalled_wrong": stalled_wrong,
        "preserved_correct": preserved_correct,
    }

def get_updated_agent_text(agent):
    if getattr(agent, "last_update", None) and agent.last_update.get("best_text") is not None:
        return agent.last_update["best_text"]
    raise RuntimeError("GSPO 更新后未记录候选输出，无法将本轮结果拼接进上下文。")

def get_selected_agent_text(agent):
    if getattr(agent, "last_update", None):
        if agent.last_update.get("selected_text") is not None:
            return agent.last_update["selected_text"]
        if agent.last_update.get("best_text") is not None:
            return agent.last_update["best_text"]
    raise RuntimeError("GSPO 采样后未记录本轮选中的输出，无法拼接进上下文。")

def get_single_gspo_round_prompt(rnd):
    """
    单智能体 GSPO 复用单 LLM 的 prompt 节奏。

    也就是说：
    - 第 1 轮用 SINGLE_PROMPTS[0]
    - 第 2 轮用 SINGLE_PROMPTS[1]
    - ...

    这里 rnd 是从 1 开始计数，所以要写 rnd - 1。
    """
    return SINGLE_PROMPTS[rnd - 1]

def get_next_answer_prompt(rnd):
    next_idx = min(rnd, NUM_ROUNDS - 1)
    return SINGLE_PROMPTS[next_idx]

def get_three_layer_action_name(act):
    if act == MIDDLE_ACTION_SILENT:
        return "pi2"
    if act == MIDDLE_ACTION_COMMENT:
        return "pi0"
    return "pi1"

def choose_outer_agent(scheduler, phase, num_agents=None):
    state = build_outer_state(phase)
    allowed_agents = list(range(NUM_AGENTS if num_agents is None else num_agents))
    strategy = scheduler.get_current_strategy(state, allowed_agents)
    selected_agent = scheduler.get_action(
        state_key=state,
        allowed_actions=allowed_agents,
    )
    return state, allowed_agents, strategy, selected_agent

def get_constrained_middle_strategy(
    selector,
    phase,
    rnd,
    selected_agent,
    incumbent_pred,
    keep_prob,
    allow_silent=True,
    allowed_actions=None,
):
    state = build_middle_state(selected_agent, phase)
    if allowed_actions is None:
        allowed_actions = get_middle_allowed_actions(
            rnd,
            incumbent_pred,
            allow_silent=allow_silent,
        )
    raw_strategy = selector.get_current_strategy(state, allowed_actions)
    strategy = apply_search_middle_strategy_constraints(
        raw_strategy,
        allowed_actions,
        phase,
        keep_prob=keep_prob,
        incumbent_pred=incumbent_pred,
        rnd=rnd,
    )
    return state, allowed_actions, strategy

def choose_middle_action(
    selector,
    phase,
    rnd,
    selected_agent,
    incumbent_pred,
    keep_prob,
    allow_silent=True,
):
    state, allowed_actions, strategy = get_constrained_middle_strategy(
        selector,
        phase,
        rnd,
        selected_agent,
        incumbent_pred,
        keep_prob,
        allow_silent=allow_silent,
    )
    act = int(np.random.choice(CFR_NUM_ACTIONS, p=strategy))
    return state, allowed_actions, strategy, act

def maybe_print_three_layer_round_debug(
    sample_idx,
    rnd,
    phase,
    keep_prob,
    stability_score,
    selected_agent,
    act,
    candidate_pred,
    incumbent_pred,
    gt,
    realized_value,
    outer_strategy,
    middle_strategy,
    accepted,
    verifier_prob,
    accept_threshold,
    accept_keep_prob,
    accept_barrier,
    accept_margin,
    accept_reason,
):
    """
    三层策略的逐轮调试打印。

    这个打印主要是为了解答类似下面这类问题：
    - 为什么 R4 不等于 R3？
    - 为什么某一轮 reward 涨了，但 accuracy 没涨？

    打印字段解释：
    - act: pi2 / pi0 / pi1
    - candidate_pred: 本轮答案候选里提取出的数字
    - incumbent_pred: 本轮结束后系统保留的 incumbent
    - accepted: verifier 是否允许本轮候选覆盖 incumbent
    """
    if not THREE_LAYER_DEBUG_PRINT:
        return

    hit = bool(compute_accuracy([incumbent_pred], [gt]))
    current_pred_str = "None" if candidate_pred is None else f"{candidate_pred:.4f}"
    incumbent_pred_str = "None" if incumbent_pred is None else f"{incumbent_pred:.4f}"
    outer_strategy_str = ", ".join(f"{prob:.4f}" for prob in outer_strategy)
    middle_strategy_str = ", ".join(f"{prob:.4f}" for prob in middle_strategy)
    print(
        f"[三层调试] sample={sample_idx} rnd={rnd} phase={phase} "
        f"keep_prob={keep_prob:.4f} stability_score={stability_score:.4f} agent={selected_agent} "
        f"act={get_three_layer_action_name(act)} candidate_pred={current_pred_str} "
        f"incumbent_pred={incumbent_pred_str} gt={gt:.4f} hit={int(hit)} "
        f"realized_value={realized_value:.4f} accepted={int(accepted)} "
        f"verifier_prob={verifier_prob:.4f} accept_threshold={accept_threshold:.4f} "
        f"accept_keep_prob={accept_keep_prob:.4f} accept_barrier={accept_barrier:.4f} "
        f"accept_margin={accept_margin:.4f} accept_reason={accept_reason} "
        f"outer=[{outer_strategy_str}] "
        f"middle=[{middle_strategy_str}]"
    )

def maybe_print_three_layer_stage(sample_idx, rnd, stage, start_time):
    if not THREE_LAYER_DEBUG_PRINT:
        return
    elapsed = time.perf_counter() - start_time
    print(f"[三层阶段] sample={sample_idx} rnd={rnd} stage={stage} elapsed={elapsed:.2f}s")

def maybe_print_accept_training_debug(
    sample_idx,
    rnd,
    previous_pred,
    candidate_pred,
    accepted,
    update_stats,
):
    if not THREE_LAYER_DEBUG_PRINT or update_stats is None:
        return

    previous_pred_str = "None" if previous_pred is None else f"{previous_pred:.4f}"
    candidate_pred_str = "None" if candidate_pred is None else f"{candidate_pred:.4f}"
    batch_summary = (
        f"batch_parseable={update_stats['num_parseable']} supervised={update_stats['num_supervised']} "
        f"pos={update_stats['num_positive']} neg={update_stats['num_negative']} zero={update_stats['num_zero']} "
        f"margin_step_l2={update_stats['margin_step_l2']:.6f} "
        f"pairwise_pairs={update_stats['pairwise_num_pairs']} "
        f"pairwise_violations={update_stats['pairwise_num_violations']} "
        f"pairwise_step_l2={update_stats['pairwise_step_l2']:.6f}"
    )
    selected_stats = update_stats.get("selected")
    if selected_stats is None:
        print(
            f"[accept训练] sample={sample_idx} rnd={rnd} prev_pred={previous_pred_str} "
            f"candidate_pred={candidate_pred_str} accepted={int(accepted)} "
            f"selected=none {batch_summary}"
        )
        return

    print(
        f"[accept训练] sample={sample_idx} rnd={rnd} prev_pred={previous_pred_str} "
        f"candidate_pred={candidate_pred_str} accepted={int(accepted)} "
        f"reward={selected_stats['reward']:.4f} "
        f"delta_reward={selected_stats['delta_reward']:.4f} "
        f"target_prob={selected_stats['target_prob']:.4f} "
        f"target_margin={selected_stats['target_margin']:.4f} "
        f"pre_prob={selected_stats['pre_prob']:.4f} post_prob={selected_stats['post_prob']:.4f} "
        f"pre_logit={selected_stats['pre_logit']:.4f} post_logit={selected_stats['post_logit']:.4f} "
        f"keep_prob={selected_stats['keep_prob']:.4f} accept_threshold={selected_stats['accept_threshold']:.4f} "
        f"accept_barrier={selected_stats['accept_barrier']:.4f} "
        f"post_keep_prob={selected_stats['post_keep_prob']:.4f} "
        f"pre_margin={selected_stats['pre_margin']:.4f} post_margin={selected_stats['post_margin']:.4f} "
        f"feature_norm={selected_stats['feature_norm']:.4f} "
        f"step_l2={selected_stats['step_l2']:.6f} {batch_summary}"
    )

def print_three_layer_sample_monitor(
    sample_idx,
    gt,
    final_pred,
    first_improve_round,
    first_degrade_round,
    first_stalled_wrong_round,
):
    final_pred_str = "None" if final_pred is None else f"{final_pred:.4f}"
    print(
        f"[三层样本监控] sample={sample_idx} gt={gt:.4f} final_pred={final_pred_str} "
        f"first_improve={first_improve_round} first_degrade={first_degrade_round} "
        f"first_stalled_wrong={first_stalled_wrong_round}"
    )

def average_rollout_reward(question, history, answer_prompt, model, tokenizer, device, gt, num_samples):
    """
    多次 rollout 取平均奖励。

    为什么要这样做：
    - 单次生成随机性很大
    - 评论/comment 的价值往往要看“它是否帮助下一步 answer 更好”
    - 所以我们会采样多次下一步 answer，再把奖励取平均，降低高方差
    """
    rewards = []
    for _ in range(num_samples):
        res = generate_response(
            answer_prompt.format(context=build_context(question, history)),
            model,
            tokenizer,
            device,
        )
        pred = extract_pred_num(res)
        rewards.append(reward_from_pred(pred, gt))
    return sum(rewards) / len(rewards) if rewards else 0.0

def sample_gspo_step(agent, question, history, rnd, speaker, kind, prompt_override=None):
    """
    GSPO 版本里，一轮 step 不只是“生成一条文本”，
    还要把训练更新所需的缓存一起带回来。

    返回的 step 是一个 dict（字典）。

    Python 语法说明：
    - dict 就是 {键: 值, 键: 值, ...}
    - 后面可以通过 step["selected_text"] 这种写法取字段
    """
    ctx = build_context(question, history)
    batch = agent.sample_candidates(ctx, prompt_override=prompt_override)
    selected_text = batch["selected_text"]
    return {
        "agent": agent,
        "question": question,
        "pre_history": history,
        "round": rnd,
        "speaker": speaker,
        "kind": kind,
        "prompt_override": prompt_override,
        "batch": batch,
        "selected_text": selected_text,
    }

def rollout_from_step(question, gt, step, future_steps, current_text, num_samples):
    """
    从某个 step 开始，向后模拟未来轨迹，并估计它会带来多少奖励。

    这是 deferred reward / with-without 奖励的核心组成部分。

    直觉上可以把它理解成：
    - “如果我把这一步内容设成 current_text”
    - “然后后面继续往下模拟”
    - “最后能得到多好的答案？”
    """
    rewards = []
    for _ in range(num_samples):
        history = append_round_output(
            step["pre_history"],
            step["round"],
            step["speaker"],
            current_text,
            step["kind"],
        )
        if step["kind"] == "answer":
            pred = extract_pred_num(current_text)
            rewards.append(reward_from_pred(pred, gt))
            continue

        final_pred = None
        for future_step in future_steps:
            future_ctx = build_context(question, history)
            future_text = future_step["agent"].sample_text(
                future_ctx,
                prompt_override=future_step["prompt_override"],
            )
            history = append_round_output(
                history,
                future_step["round"],
                future_step["speaker"],
                future_text,
                future_step["kind"],
            )
            if future_step["kind"] == "answer":
                final_pred = extract_pred_num(future_text)
                break

        rewards.append(reward_from_pred(final_pred, gt))
    return sum(rewards) / len(rewards) if rewards else 0.0

def compute_deferred_rewards_for_step(segment_steps, step_idx, gt):
    """
    给一个缓存段里的某一步计算奖励。

    两类情况：
    1. 如果这一步本身是 answer
       那就直接看候选答案本身离真值有多近
    2. 如果这一步是 comment
       那就比较：
       - 带着这个 comment 往后模拟，结果怎样
       - 不带这个 comment 往后模拟，结果怎样
       二者差值就是这个 comment 的边际价值
    """
    step = segment_steps[step_idx]
    if step["kind"] == "answer":
        return [
            reward_from_pred(extract_pred_num(cand["text"]), gt)
            for cand in step["batch"]["candidates"]
        ]

    future_steps = segment_steps[step_idx + 1:]
    without_reward = rollout_from_step(
        step["question"],
        gt,
        step,
        future_steps,
        "",
        GSPO_COMMENT_EVAL_SAMPLES,
    )
    return [
        rollout_from_step(
            step["question"],
            gt,
            step,
            future_steps,
            cand["text"],
            GSPO_COMMENT_EVAL_SAMPLES,
        ) - without_reward
        for cand in step["batch"]["candidates"]
    ]

def apply_deferred_segment_updates(segment_steps, gt):
    """
    对一个“从若干 comment 到一个 answer 结束”的缓存段统一做 GSPO 更新。

    这是这版 GSPO 与“每轮立刻更新”的最大区别之一。

    改前直觉：
    - 一生成 comment 就立刻给奖励并更新

    改后现在：
    - comment 先放进 pending_segment
    - 等到这个段里真正出现 answer
    - 再从后往前统一计算这段里每一步的奖励并更新
    """
    if not segment_steps:
        return 0.0

    computed_rewards = []
    for step_idx in reversed(range(len(segment_steps))):
        computed_rewards.append(
            (step_idx, compute_deferred_rewards_for_step(segment_steps, step_idx, gt))
        )

    for step_idx, rewards in computed_rewards:
        segment_steps[step_idx]["agent"].update_from_cached(
            segment_steps[step_idx]["batch"],
            rewards,
        )

    answer_step = segment_steps[-1]
    return reward_from_pred(extract_pred_num(answer_step["selected_text"]), gt)

def get_three_layer_action_spec(act):
    """
    根据 act 取出三层策略这一轮该用哪个 policy / prompt / speaker / kind。
    """
    if act == MIDDLE_ACTION_SILENT:
        return "pi2", None, "pi2", "silent"
    if act == MIDDLE_ACTION_COMMENT:
        return "pi0", PI0_PROMPT, "pi0", "comment"
    return "pi1", PI1_PROMPT, "pi1", "answer"

def evaluate_answer_revision(
    verifier,
    question,
    rnd,
    history,
    incumbent_text,
    incumbent_pred,
    candidate_text,
    candidate_pred,
    gt,
):
    """
    评估一个候选答案相对 incumbent 的收益。

    这里要分清两个“值”：
    1. delta_reward
       候选答案如果真的替换 incumbent，理论上会比 incumbent 好多少/差多少
    2. effective_delta
       这是环境里“实际生效”的收益
       只有 verifier 点头接受，这次修订才真的改写 incumbent；
       否则从环境视角看，这轮没有改答案，所以收益按 0 记

    你可以把它理解成：
    - delta_reward: 候选答案的客观好坏
    - effective_delta: 候选答案在“经过 verifier 审批后”真正产生的效果

    effective_delta:
    - 如果 verifier 接受，就把相对 incumbent 的提升/下降计入环境回报
    - 如果 verifier 拒绝，就视为本轮没有真正修改 incumbent，收益记 0
    """
    delta_reward = reward_delta_from_preds(incumbent_pred, candidate_pred, gt)
    if (
        incumbent_pred is not None
        and candidate_pred is not None
        and abs(candidate_pred - incumbent_pred) <= 1e-6
    ):
        gate = verifier.get_accept_gate(question, history, incumbent_text)
        return {
            "delta_reward": delta_reward,
            "effective_delta": 0.0,
            "accept": False,
            "verifier_prob": 0.0,
            "accept_threshold": gate["threshold"],
            "accept_keep_prob": gate["keep_prob"],
            "accept_barrier": gate["barrier"],
            "accept_margin": -gate["barrier"],
            "accept_reason": "same_pred",
            "improved": False,
        }

    decision = verifier.should_accept(
        question,
        history,
        incumbent_text,
        candidate_text,
        candidate_pred,
    )
    accept = decision["accept"]
    effective_delta = delta_reward if accept else 0.0
    return {
        "delta_reward": delta_reward,
        "effective_delta": effective_delta,
        "accept": accept,
        "verifier_prob": decision["accept_prob"],
        "accept_threshold": decision["accept_threshold"],
        "accept_keep_prob": decision["keep_prob"],
        "accept_barrier": decision["accept_barrier"],
        "accept_margin": decision["accept_margin"],
        "accept_reason": decision["reason"],
        "improved": delta_reward > 0.0,
    }

def build_accept_update_payload(
    question,
    history,
    incumbent_text,
    incumbent_pred,
    incumbent_reward,
    batch,
    gt,
):
    """
    为 accept_head 组装整批监督样本。

    这里不只喂最终 selected candidate，
    而是把这一轮真正采样出来的所有 parseable answer candidates 都送进 verifier：
    - 点式监督：每个 candidate 相对 incumbent 是正收益还是负收益
    - 排序监督：同一批里，更优 candidate 的分数应高于更差 candidate

    这不会引入训练/推理不一致：
    - 推理时仍只使用可观测信号选候选、再由 verifier 审批
    - 训练时只是把“本轮已经实际采样到的 batch”利用得更充分
    """
    if incumbent_text is None or batch is None:
        return None

    candidate_updates = []
    selected_idx = int(batch.get("selected_idx", -1))
    for idx, cand in enumerate(batch.get("candidates", [])):
        candidate_pred = extract_pred_num(cand["text"])
        if candidate_pred is None:
            continue
        candidate_updates.append({
            "current_text": cand["text"],
            "candidate_pred": candidate_pred,
            "reward": reward_from_pred(candidate_pred, gt),
            "delta_reward": reward_delta_from_preds(incumbent_pred, candidate_pred, gt),
            "selected": idx == selected_idx,
        })

    if not candidate_updates:
        return None
    return (
        question,
        history,
        incumbent_text,
        incumbent_reward,
        candidate_updates,
    )

def compute_answer_candidate_rewards(
    question,
    batch,
    verifier,
    rnd,
    history,
    incumbent_text,
    incumbent_pred,
    gt,
):
    """
    给 answer policy 这一批候选答案逐个打分。

    对共享策略栈里的内层 GSPO，answer 的价值直接定义为答案质量 r。
    """
    rewards = []
    for cand in batch["candidates"]:
        candidate_pred = extract_pred_num(cand["text"])
        rewards.append(reward_from_pred(candidate_pred, gt))
    return rewards

def average_next_answer_reward(
    agent_bundle,
    question,
    history,
    rnd,
    incumbent_pred,
    gt,
    num_samples,
):
    """
    估计“从当前历史继续 rollout，到最近一次 answer 出现时”的绝对 reward。

    这个量不再掺入 incumbent baseline，
    主要给 comment 的 with/without 比较使用：

        reward(with comment next_answer) - reward(without comment next_answer)
    """
    rewards = []
    for _ in range(num_samples):
        answer_text = agent_bundle["pi1"].sample_text(
            build_context(question, history),
            prompt_override=PI1_PROMPT,
        )
        answer_pred = extract_pred_num(answer_text)
        rewards.append(
            rollout_terminal_absolute_reward(answer_pred, incumbent_pred, gt)
        )
    return sum(rewards) / len(rewards) if rewards else 0.0

def average_answer_delta_reward(
    agent_bundle,
    question,
    history,
    rnd,
    incumbent_pred,
    gt,
    num_samples,
):
    """
    估计“如果现在尝试 answer，平均能带来多少相对 incumbent 的提升”：

        reward(next_answer) - reward(incumbent)

    这个量用于：
    - 中层 CFR 中 action=answer 的反事实估值
    """
    rewards = []
    for _ in range(num_samples):
        answer_text = agent_bundle["pi1"].sample_text(
            build_context(question, history),
            prompt_override=PI1_PROMPT,
        )
        answer_pred = extract_pred_num(answer_text)
        rewards.append(
            rollout_terminal_delta_reward(answer_pred, incumbent_pred, gt)
        )
    return sum(rewards) / len(rewards) if rewards else 0.0

def average_next_answer_delta_reward(
    agent_bundle,
    question,
    history,
    rnd,
    incumbent_pred,
    gt,
    num_samples,
):
    """
    统一的短程终端效用近似：

        reward(next_answer) - reward(current_incumbent)

    这里的 next_answer 指“从当前 history 继续 rollout，最近一次出现的 answer”。
    中层/外层 CFR 的反事实比较统一使用这个量，而不是再混用：
    - answer 的直接 delta
    - comment 的 with/without 差
    - silent 的手工惩罚
    """
    return average_answer_delta_reward(
        agent_bundle,
        question,
        history,
        rnd,
        incumbent_pred,
        gt,
        num_samples,
    )

def compute_comment_candidate_rewards(step, agent_bundle, verifier, incumbent_text, incumbent_pred, gt):
    """
    给 comment policy 的候选 comment 打分。

    算法是典型的 with-comment / without-comment 对比：
    - without_comment:
      不加这条 comment，直接看下一步 answer 的平均绝对 reward
    - with_comment:
      把这条 comment 拼进历史，再看下一步 answer 的平均绝对 reward
    - 两者差值：
      就是这条 comment 对后续 answer 的边际帮助
    """
    without_comment = average_next_answer_reward(
        agent_bundle,
        step["question"],
        step["pre_history"],
        min(step["round"] + 1, NUM_ROUNDS),
        incumbent_pred,
        gt,
        GSPO_COMMENT_EVAL_SAMPLES,
    )

    rewards = []
    for cand in step["batch"]["candidates"]:
        next_history = append_round_output(
            step["pre_history"],
            step["round"],
            step["speaker"],
            cand["text"],
            "comment",
        )
        with_comment = average_next_answer_reward(
            agent_bundle,
            step["question"],
            next_history,
            min(step["round"] + 1, NUM_ROUNDS),
            incumbent_pred,
            gt,
            GSPO_COMMENT_EVAL_SAMPLES,
        )
        rewards.append(with_comment - without_comment)
    return rewards

def average_policy_stack_answer_reward(agent_bundle, question, history, gt, num_samples):
    """
    给共享策略栈估计“下一次 answer 自身的质量 r”。

    这里不再回退到 incumbent reward：
    - answer 的价值就是 answer utterance 自身的质量
    - 如果这次 answer 不可解析，就直接按 r(None) 记分
    """
    rewards = []
    num_samples = max(int(num_samples), 1)
    for _ in range(num_samples):
        answer_text = agent_bundle["pi1"].sample_text(
            build_context(question, history),
            prompt_override=PI1_PROMPT,
        )
        answer_pred = extract_pred_num(answer_text)
        rewards.append(reward_from_pred(answer_pred, gt))
    return sum(rewards) / len(rewards) if rewards else 0.0

def estimate_policy_stack_answer_value(agent_bundle, question, history, gt, num_samples=1):
    return average_policy_stack_answer_reward(
        agent_bundle,
        question,
        history,
        gt,
        num_samples,
    )

def estimate_policy_stack_comment_value(
    agent_bundle,
    question,
    history,
    rnd,
    incumbent_pred,
    gt,
    num_samples,
    comment_text=None,
    speaker="pi0_cf",
):
    """
    comment 的价值恢复为旧定义：

        with_comment 的下一次 answer 的 r
        - without_comment 的下一次 answer 的 r
    """
    without_comment_reward = average_next_answer_reward(
        agent_bundle,
        question,
        history,
        min(rnd + 1, NUM_ROUNDS),
        incumbent_pred,
        gt,
        num_samples,
    )
    if comment_text is None:
        comment_text = agent_bundle["pi0"].sample_text(
            build_context(question, history),
            prompt_override=PI0_PROMPT,
        )
    next_history = append_round_output(
        history,
        rnd,
        speaker,
        comment_text,
        "comment",
    )
    with_comment_reward = average_next_answer_reward(
        agent_bundle,
        question,
        next_history,
        min(rnd + 1, NUM_ROUNDS),
        incumbent_pred,
        gt,
        num_samples,
    )
    return with_comment_reward - without_comment_reward

def compute_policy_stack_comment_candidate_rewards(step, agent_bundle, incumbent_pred, gt):
    rewards = []
    for cand in step["batch"]["candidates"]:
        rewards.append(
            estimate_policy_stack_comment_value(
                agent_bundle,
                step["question"],
                step["pre_history"],
                step["round"],
                incumbent_pred,
                gt,
                GSPO_COMMENT_EVAL_SAMPLES,
                comment_text=cand["text"],
                speaker=step["speaker"],
            )
        )
    return rewards

def build_non_silent_strategy(strategy, allowed_actions):
    non_silent_actions = [
        act
        for act in allowed_actions
        if act != MIDDLE_ACTION_SILENT
    ]
    non_silent_strategy = np.zeros(CFR_NUM_ACTIONS, dtype=np.float64)
    if not non_silent_actions:
        return non_silent_strategy, non_silent_actions

    mass = float(sum(strategy[act] for act in non_silent_actions))
    if mass > 0.0:
        for act in non_silent_actions:
            non_silent_strategy[act] = strategy[act] / mass
    else:
        share = 1.0 / len(non_silent_actions)
        for act in non_silent_actions:
            non_silent_strategy[act] = share
    return non_silent_strategy, non_silent_actions

def sample_policy_stack_answer_absolute_reward(agent_bundle, question, history, incumbent_pred, gt):
    """
    单次 rollout 一个 answer，并返回“最近 answer 的绝对 reward”。
    """
    answer_text = agent_bundle["pi1"].sample_text(
        build_context(question, history),
        prompt_override=PI1_PROMPT,
    )
    answer_pred = extract_pred_num(answer_text)
    return rollout_terminal_absolute_reward(answer_pred, incumbent_pred, gt)

def sample_policy_stack_next_answer_reward_after_comment(
    agent_bundle,
    question,
    history,
    rnd,
    incumbent_pred,
    gt,
):
    comment_text = agent_bundle["pi0"].sample_text(
        build_context(question, history),
        prompt_override=PI0_PROMPT,
    )
    next_history = append_round_output(
        history,
        rnd,
        "pi0_silent_cf",
        comment_text,
        "comment",
    )
    return sample_policy_stack_answer_absolute_reward(
        agent_bundle,
        question,
        next_history,
        incumbent_pred,
        gt,
    )

def estimate_policy_stack_silent_without_reward(
    agent_bundle,
    question,
    history,
    rnd,
    incumbent_pred,
    gt,
    strategy,
    allowed_actions,
    num_samples,
):
    """
    不选 silent 时，把当前策略在 comment / answer 上重新归一化后抽样 rollout，
    估计得到的“最近 answer 的 r”。
    """
    non_silent_strategy, non_silent_actions = build_non_silent_strategy(
        strategy,
        allowed_actions,
    )
    if not non_silent_actions:
        return average_next_answer_reward(
            agent_bundle,
            question,
            history,
            min(rnd + 1, NUM_ROUNDS),
            incumbent_pred,
            gt,
            num_samples,
        )

    rewards = []
    num_samples = max(int(num_samples), 1)
    for _ in range(num_samples):
        sampled_action = int(np.random.choice(CFR_NUM_ACTIONS, p=non_silent_strategy))
        if sampled_action == MIDDLE_ACTION_COMMENT:
            rewards.append(
                sample_policy_stack_next_answer_reward_after_comment(
                    agent_bundle,
                    question,
                    history,
                    rnd,
                    incumbent_pred,
                    gt,
                )
            )
        else:
            rewards.append(
                sample_policy_stack_answer_absolute_reward(
                    agent_bundle,
                    question,
                    history,
                    incumbent_pred,
                    gt,
                )
            )
    return sum(rewards) / len(rewards) if rewards else 0.0

def estimate_policy_stack_silent_value(
    agent_bundle,
    question,
    history,
    rnd,
    incumbent_pred,
    gt,
    strategy,
    allowed_actions=None,
):
    """
    silent 的价值定义为：

        with_silent 的下一次 answer 的 r
        - 按当前 comment / answer 归一化策略抽样 rollout 得到的 answer 的 r
    """
    if allowed_actions is None:
        allowed_actions = get_middle_allowed_actions(
            rnd,
            incumbent_pred,
            allow_silent=True,
        )
    with_silent_reward = average_next_answer_reward(
        agent_bundle,
        question,
        history,
        min(rnd + 1, NUM_ROUNDS),
        incumbent_pred,
        gt,
        GSPO_COMMENT_EVAL_SAMPLES,
    )
    without_silent_reward = estimate_policy_stack_silent_without_reward(
        agent_bundle,
        question,
        history,
        rnd,
        incumbent_pred,
        gt,
        strategy,
        allowed_actions,
        GSPO_COMMENT_EVAL_SAMPLES,
    )
    return with_silent_reward - without_silent_reward

def estimate_outer_answer_value(
    agent_bundle,
    verifier,
    question,
    history,
    incumbent_text,
    incumbent_pred,
    gt,
    rnd,
    num_samples=1,
):
    """
    外层 agent 的价值看“本轮结束后最近 answer 的 r”。

    如果 answer 没被接受，就仍然保持当前 incumbent 的 r。
    """
    current_reward = reward_from_pred(incumbent_pred, gt)
    rewards = []
    num_samples = max(int(num_samples), 1)
    for _ in range(num_samples):
        candidate_text = agent_bundle["pi1"].sample_text(
            build_context(question, history),
            prompt_override=PI1_PROMPT,
        )
        candidate_pred = extract_pred_num(candidate_text)
        decision = evaluate_answer_revision(
            verifier,
            question,
            rnd,
            history,
            incumbent_text,
            incumbent_pred,
            candidate_text,
            candidate_pred,
            gt,
        )
        rewards.append(
            reward_from_pred(candidate_pred, gt) if decision["accept"] else current_reward
        )
    return sum(rewards) / len(rewards) if rewards else current_reward

def estimate_outer_agent_value(
    middle_cfr,
    agents,
    verifier,
    question,
    gt,
    phase,
    agent_idx,
    rnd,
    history,
    incumbent_text,
    incumbent_pred,
    keep_prob,
    allow_silent=True,
):
    """
    外层每个 agent 的价值定义为“本轮结束后最近 answer 的 r”。
    """
    current_reward = reward_from_pred(incumbent_pred, gt)
    _, allowed_actions, middle_strategy = get_constrained_middle_strategy(
        middle_cfr,
        phase,
        rnd,
        agent_idx,
        incumbent_pred,
        keep_prob,
        allow_silent=allow_silent,
    )
    action_values = np.zeros(CFR_NUM_ACTIONS, dtype=np.float64)
    if MIDDLE_ACTION_SILENT in allowed_actions:
        action_values[MIDDLE_ACTION_SILENT] = current_reward
    if MIDDLE_ACTION_COMMENT in allowed_actions:
        action_values[MIDDLE_ACTION_COMMENT] = current_reward
    if MIDDLE_ACTION_ANSWER in allowed_actions:
        action_values[MIDDLE_ACTION_ANSWER] = estimate_outer_answer_value(
            agents[agent_idx],
            verifier,
            question,
            history,
            incumbent_text,
            incumbent_pred,
            gt,
            rnd,
            num_samples=1,
        )
    return float(np.dot(action_values, middle_strategy))

def estimate_middle_action_values(
    question,
    gt,
    agents,
    middle_cfr,
    phase,
    selected_agent,
    rnd,
    history,
    incumbent_text,
    incumbent_pred,
    verifier,
    keep_prob,
    allow_silent=True,
    known_action=None,
    known_value=None,
):
    """
    估计某个 agent 在当前 round/state 下的中层三动作价值。

    优化点：
    - 如果某个动作已经在真实轨迹里执行过，并且它的 realized value 已知，
      那么这里直接复用，不再为这个动作再跑一遍反事实 rollout。
    - 这样中层 regret 更新和外层调度更新就能共享同一轮里的价值估计，
      避免 selected agent 的 chosen action 被重复计算。
    """
    allowed_actions = get_middle_allowed_actions(
        rnd,
        incumbent_pred,
        allow_silent=allow_silent,
    )
    values = np.zeros(CFR_NUM_ACTIONS, dtype=np.float64)
    agent_bundle = agents[selected_agent]
    _, allowed_actions, middle_strategy = get_constrained_middle_strategy(
        middle_cfr,
        phase,
        rnd,
        selected_agent,
        incumbent_pred,
        keep_prob,
        allow_silent=allow_silent,
        allowed_actions=allowed_actions,
    )

    if MIDDLE_ACTION_COMMENT in allowed_actions:
        if known_action == MIDDLE_ACTION_COMMENT and known_value is not None:
            values[MIDDLE_ACTION_COMMENT] = known_value
        else:
            values[MIDDLE_ACTION_COMMENT] = estimate_policy_stack_comment_value(
                agent_bundle,
                question,
                history,
                rnd,
                incumbent_pred,
                gt,
                GSPO_COMMENT_EVAL_SAMPLES,
            )
    if MIDDLE_ACTION_ANSWER in allowed_actions:
        if known_action == MIDDLE_ACTION_ANSWER and known_value is not None:
            values[MIDDLE_ACTION_ANSWER] = known_value
        else:
            values[MIDDLE_ACTION_ANSWER] = estimate_policy_stack_answer_value(
                agent_bundle,
                question,
                history,
                gt,
                num_samples=1,
            )
    if MIDDLE_ACTION_SILENT in allowed_actions:
        if known_action == MIDDLE_ACTION_SILENT and known_value is not None:
            values[MIDDLE_ACTION_SILENT] = known_value
        else:
            values[MIDDLE_ACTION_SILENT] = estimate_policy_stack_silent_value(
                agent_bundle,
                question,
                history,
                rnd,
                incumbent_pred,
                gt,
                middle_strategy,
                allowed_actions=allowed_actions,
            )
    return values, allowed_actions

def get_cached_middle_estimate(
    cache,
    question,
    gt,
    agents,
    middle_cfr,
    phase,
    agent_idx,
    rnd,
    history,
    incumbent_text,
    incumbent_pred,
    verifier,
    keep_prob,
    allow_silent=True,
    known_action=None,
    known_value=None,
):
    """
    每轮每个 agent 的中层动作价值只估一次，并在本轮内复用。

    这样：
    - 中层 regret 更新会用到它
    - 外层调度 regret 也直接复用同一份 value vector
    """
    cache_key = (agent_idx, phase, rnd, bool(allow_silent))
    cached = cache.get(cache_key)
    if cached is not None:
        values, allowed_actions = cached
        if known_action is not None and known_value is not None:
            values = np.array(values, copy=True)
            values[known_action] = known_value
        return values, allowed_actions

    values, allowed_actions = estimate_middle_action_values(
        question,
        gt,
        agents,
        middle_cfr,
        phase,
        agent_idx,
        rnd,
        history,
        incumbent_text,
        incumbent_pred,
        verifier,
        keep_prob,
        allow_silent,
        known_action=known_action,
        known_value=known_value,
    )
    cache[cache_key] = (np.array(values, copy=True), list(allowed_actions))
    return values, allowed_actions

def init_three_layer_round_buffers():
    """
    初始化“三层策略实验”的整轮统计容器。

    这些列表都是“按轮次存储”的：
    - 下标 0 对应第 1 轮
    - 下标 1 对应第 2 轮
    - ...

    每个元素本身又是一个列表，用来累计所有样本在该轮的统计值。

    这样设计的好处是：
    - 跑完所有样本后，可以很容易按“第 r 轮”做平均
    - 画图和写 CSV 都会更方便
    """
    return {
        "round_preds": [[] for _ in range(NUM_ROUNDS)],
        "round_gts": [[] for _ in range(NUM_ROUNDS)],
        "best_round_hits": [[] for _ in range(NUM_ROUNDS)],
        "round_rewards": [[] for _ in range(NUM_ROUNDS)],
        "round_regrets": [[] for _ in range(NUM_ROUNDS)],
        "round_outer_probs": [[] for _ in range(NUM_ROUNDS)],
        "round_middle_probs": [[] for _ in range(NUM_ROUNDS)],
        "round_search_flags": [[] for _ in range(NUM_ROUNDS)],
        "round_stabilize_flags": [[] for _ in range(NUM_ROUNDS)],
        "round_improve_flags": [[] for _ in range(NUM_ROUNDS)],
        "round_degrade_flags": [[] for _ in range(NUM_ROUNDS)],
        "round_stalled_wrong_flags": [[] for _ in range(NUM_ROUNDS)],
        "round_preserved_correct_flags": [[] for _ in range(NUM_ROUNDS)],
    }

def init_three_layer_sample_state():
    """
    初始化“单条样本”在三层实验里的运行状态。

    这里存的是“会随着轮次变化”的状态变量：
    - history:
      当前样本到目前为止的结构化对话历史
    - incumbent_text / incumbent_pred:
      当前系统正式保留的答案文本和数字答案
    - incumbent_reward:
      当前 incumbent 对应的 reward，用于 keep_head 训练和 phase 判断
    - best_correct:
      这条样本是否历史上曾经答对过，用来算 best-so-far
    - first_*:
      第一次改善 / 第一次退化 / 第一次卡住错误答案分别发生在哪一轮
    """
    return {
        "history": [],
        "incumbent_text": None,
        "incumbent_pred": None,
        "incumbent_reward": 0.0,
        "best_correct": False,
        "first_improve_round": None,
        "first_degrade_round": None,
        "first_stalled_wrong_round": None,
    }

def run_three_layer_realized_action(
    agents,
    verifier,
    sample_idx,
    question,
    gt,
    rnd,
    selected_agent,
    act,
    phase,
    pre_history,
    pre_incumbent_text,
    pre_incumbent_pred,
    middle_strategy,
):
    """
    执行三层策略在“真实轨迹”里的这一步动作。

    这一步只做一件事：
    - 在已经确定了 `selected_agent` 和 `act` 之后，
      真正去采样文本、算 reward、决定是否覆盖 incumbent。

    返回一个字典，里面把这一轮后续还会用到的所有中间结果都带出来：
    - history:
      这一轮执行完后的新历史
    - realized_value:
      中层 / 内层学习用的动作价值
    - outer_value:
      外层调度器看的“本轮结束后最近 answer 的 r”
    - accept_update_payload:
      给 accept_head 做在线更新时需要的监督信号
    - gspo_update_payload:
      给内层 GSPO policy 做更新时需要的缓存批次和 reward

    你可以把它理解成“把长主循环里最核心的那段 if/elif/else 单独搬出来”。
    """
    current_pred = None
    accepted = False
    has_answer_candidates = False
    incumbent_changed = False
    verifier_prob = 0.0
    accept_threshold = 0.0
    accept_keep_prob = 0.0
    accept_barrier = VERIFIER_ACCEPT_THRESHOLD
    accept_margin = -VERIFIER_ACCEPT_THRESHOLD
    accept_reason = "not_answer"
    realized_middle_value = 0.0
    realized_outer_value = 0.0
    step = None
    accept_update_payload = None
    gspo_update_payload = None
    history = pre_history
    incumbent_text = pre_incumbent_text
    incumbent_pred = pre_incumbent_pred
    incumbent_reward = reward_from_pred(incumbent_pred, gt)
    realized_outer_value = incumbent_reward

    if act == MIDDLE_ACTION_SILENT:
        realized_middle_value = estimate_policy_stack_silent_value(
            agents[selected_agent],
            question,
            pre_history,
            rnd,
            pre_incumbent_pred,
            gt,
            middle_strategy,
        )
        history = append_round_output(
            history,
            rnd,
            f"agent{selected_agent}_pi2",
            "pi2 选择不发言，本轮不修改 incumbent。",
            "silent",
        )
    elif act == MIDDLE_ACTION_COMMENT:
        step = sample_gspo_step(
            agents[selected_agent]["pi0"],
            question,
            pre_history,
            rnd,
            f"agent{selected_agent}_pi0",
            "comment",
            prompt_override=PI0_PROMPT,
        )
        comment_rewards = compute_policy_stack_comment_candidate_rewards(
            step,
            agents[selected_agent],
            pre_incumbent_pred,
            gt,
        )
        gspo_update_payload = (
            step["agent"],
            step["batch"],
            comment_rewards,
        )
        selected_idx = int(step["batch"].get("selected_idx", 0))
        if comment_rewards:
            selected_idx = max(0, min(selected_idx, len(comment_rewards) - 1))
            realized_middle_value = comment_rewards[selected_idx]
        history = append_round_output(
            history,
            rnd,
            f"agent{selected_agent}_pi0",
            step["selected_text"],
            "comment",
        )
    else:
        step = sample_gspo_step(
            agents[selected_agent]["pi1"],
            question,
            pre_history,
            rnd,
            f"agent{selected_agent}_pi1",
            "answer",
            prompt_override=PI1_PROMPT,
        )
        has_answer_candidates = bool(step["batch"]["candidates"])
        selected_idx = 0
        selected_meta = None
        set_sampled_batch_selected_candidate(
            step["agent"],
            step["batch"],
            selected_idx,
        )
        step["selected_text"] = step["batch"]["selected_text"]
        maybe_print_answer_candidate_selection_debug(
            sample_idx,
            rnd,
            step["batch"],
            selected_idx,
            selected_meta,
        )
        answer_rewards = compute_answer_candidate_rewards(
            question,
            step["batch"],
            verifier,
            rnd,
            pre_history,
            pre_incumbent_text,
            pre_incumbent_pred,
            gt,
        )
        gspo_update_payload = (
            step["agent"],
            step["batch"],
            answer_rewards,
        )
        current_pred = extract_pred_num(step["selected_text"])
        decision = evaluate_answer_revision(
            verifier,
            question,
            rnd,
            pre_history,
            pre_incumbent_text,
            pre_incumbent_pred,
            step["selected_text"],
            current_pred,
            gt,
        )
        realized_middle_value = answer_rewards[step["batch"]["selected_idx"]]
        accepted = decision["accept"]
        verifier_prob = decision["verifier_prob"]
        accept_threshold = decision["accept_threshold"]
        accept_keep_prob = decision["accept_keep_prob"]
        accept_barrier = decision["accept_barrier"]
        accept_margin = decision["accept_margin"]
        accept_reason = decision["accept_reason"]
        accept_update_payload = (
            build_accept_update_payload(
                question,
                pre_history,
                pre_incumbent_text,
                pre_incumbent_pred,
                reward_from_pred(pre_incumbent_pred, gt),
                step["batch"],
                gt,
            )
        )

        if accepted:
            incumbent_text = step["selected_text"]
            incumbent_pred = current_pred
            incumbent_reward = reward_from_pred(incumbent_pred, gt)
            incumbent_changed = (
                pre_incumbent_text != incumbent_text
                or pre_incumbent_pred != incumbent_pred
            )
            history = append_round_output(
                history,
                rnd,
                f"agent{selected_agent}_pi1",
                step["selected_text"],
                "answer",
            )
        else:
            history = append_round_output(
                history,
                rnd,
                f"agent{selected_agent}_verifier",
                "候选修订未通过 verifier，保持 incumbent 不变。",
                "meta",
            )
        realized_outer_value = incumbent_reward

    return {
        "history": history,
        "incumbent_text": incumbent_text,
        "incumbent_pred": incumbent_pred,
        "incumbent_reward": incumbent_reward,
        "current_pred": current_pred,
        "accepted": accepted,
        "verifier_prob": verifier_prob,
        "accept_threshold": accept_threshold,
        "accept_keep_prob": accept_keep_prob,
        "accept_barrier": accept_barrier,
        "accept_margin": accept_margin,
        "accept_reason": accept_reason,
        "realized_value": realized_middle_value,
        "outer_value": realized_outer_value,
        "step": step,
        "accept_update_payload": accept_update_payload,
        "gspo_update_payload": gspo_update_payload,
        "has_answer_candidates": has_answer_candidates,
        "incumbent_changed": incumbent_changed,
    }

def update_three_layer_regrets(
    outer_cfr,
    middle_cfr,
    agents,
    verifier,
    question,
    gt,
    rnd,
    phase,
    selected_agent,
    act,
    outer_state,
    allowed_agents,
    middle_state,
    allowed_actions,
    pre_history,
    pre_incumbent_text,
    pre_incumbent_pred,
    keep_prob,
    allow_silent,
    realized_value,
    outer_realized_value,
):
    """
    统一做“中层 regret 更新 + 外层 regret 更新”。

    为什么要单独拆出来：
    - 这部分最像“算法核心”，但和真实文本生成逻辑混在一起时很难读
    - 拆开后你可以单独把它看成“CFR 如何做反事实比较”

    这里最重要的两个量是：
    - middle_action_values:
      当前 selected_agent 在本 state 下，三种动作各自会有多大价值
    - outer_action_values:
      当前 phase 下，如果这轮换不同 agent 出场，
      “本轮结束后最近 answer 的 r”分别是多少
    """
    middle_estimate_cache = {}

    if len(allowed_actions) == 1:
        middle_action_values = np.zeros(CFR_NUM_ACTIONS, dtype=np.float64)
        middle_action_values[act] = realized_value
    else:
        middle_action_values, allowed_actions = get_cached_middle_estimate(
            middle_estimate_cache,
            question,
            gt,
            agents,
            middle_cfr,
            phase,
            selected_agent,
            rnd,
            pre_history,
            pre_incumbent_text,
            pre_incumbent_pred,
            verifier,
            keep_prob,
            allow_silent,
            known_action=act,
            known_value=realized_value,
        )
        middle_cfr.update_regret(
            middle_state,
            act,
            middle_action_values,
            allowed_actions=allowed_actions,
        )

    if outer_cfr is None:
        return

    outer_action_values = np.zeros(len(allowed_agents), dtype=np.float64)
    for agent_idx in allowed_agents:
        if agent_idx == selected_agent:
            outer_action_values[agent_idx] = outer_realized_value
            continue

        outer_action_values[agent_idx] = estimate_outer_agent_value(
            middle_cfr,
            agents,
            verifier,
            question,
            gt,
            phase,
            agent_idx,
            rnd,
            pre_history,
            pre_incumbent_text,
            pre_incumbent_pred,
            keep_prob,
            allow_silent,
        )

    outer_cfr.update_regret(
        outer_state,
        selected_agent,
        outer_action_values,
        allowed_actions=allowed_agents,
    )

def apply_three_layer_policy_updates(
    verifier,
    gspo_update_payload,
    accept_update_payload,
    question,
    history,
    incumbent_text,
    incumbent_reward,
    should_update_keep,
):
    """
    统一做“真实轨迹执行完之后”的在线学习更新。

    这里分三类学习器：
    - verifier scorer:
      学“每条 answer utterance 本身的绝对质量分数”
      同时学“更优 utterance 的分数应高于更差 utterance”
    - GSPO policy:
      学“这轮真正生成的文本在组内候选里是否值得被偏好”

    注意时序：
    - 这些更新影响的是“下一轮以后”的行为
    - 不会回头修改本轮已经发生的 accepted / rejected 决策
    """
    accept_update_stats = None
    if accept_update_payload is not None:
        accept_update_stats = verifier.update_accept_batch(*accept_update_payload)

    if should_update_keep and incumbent_text is not None:
        verifier.update_keep(
            question,
            history,
            incumbent_text,
            incumbent_reward,
        )

    if gspo_update_payload is not None:
        gspo_update_payload[0].update_from_cached(
            gspo_update_payload[1],
            gspo_update_payload[2],
        )

    return accept_update_stats

def record_three_layer_round(
    buffers,
    rnd,
    gt,
    incumbent_pred,
    best_correct,
    realized_value,
    outer_cfr,
    middle_cfr,
    outer_strategy,
    middle_strategy,
    phase,
    transition,
):
    """
    把当前这一轮的结果写进全局统计容器。

    这个函数不做任何决策，只做“记账”：
    - 本轮准确率要用什么 pred
    - 本轮 improve/degrade flag 是什么
    - 本轮 outer/middle 策略概率长什么样

    你可以把它理解成三层实验的“日志整理员”。
    """
    buffers["round_preds"][rnd-1].append(incumbent_pred)
    buffers["round_gts"][rnd-1].append(gt)
    buffers["best_round_hits"][rnd-1].append(1.0 if best_correct else 0.0)
    buffers["round_rewards"][rnd-1].append(realized_value)
    total_regret = middle_cfr.get_total_regret()
    if outer_cfr is not None:
        total_regret += outer_cfr.get_total_regret()
    buffers["round_regrets"][rnd-1].append(total_regret)
    buffers["round_outer_probs"][rnd-1].append(np.array(outer_strategy, copy=True))
    buffers["round_middle_probs"][rnd-1].append(middle_strategy.copy())
    buffers["round_search_flags"][rnd-1].append(1.0 if phase == "search" else 0.0)
    buffers["round_stabilize_flags"][rnd-1].append(1.0 if phase == "stabilize" else 0.0)
    buffers["round_improve_flags"][rnd-1].append(1.0 if transition["improved"] else 0.0)
    buffers["round_degrade_flags"][rnd-1].append(1.0 if transition["degraded"] else 0.0)
    buffers["round_stalled_wrong_flags"][rnd-1].append(1.0 if transition["stalled_wrong"] else 0.0)
    buffers["round_preserved_correct_flags"][rnd-1].append(1.0 if transition["preserved_correct"] else 0.0)

def finalize_policy_stack_experiment(buffers, exp_name, log_fn, log_metrics=True):
    """
    在所有样本都跑完后，按轮次输出汇总统计并写日志。

    这里的工作基本都是“把前面累计好的列表做平均”。
    """
    total_reward = 0.0
    for rnd in range(1, NUM_ROUNDS+1):
        acc = compute_accuracy(buffers["round_preds"][rnd-1], buffers["round_gts"][rnd-1])
        best_acc = compute_best_so_far_accuracies(buffers["best_round_hits"])[rnd-1]
        avg_reward = (
            sum(buffers["round_rewards"][rnd-1]) / len(buffers["round_rewards"][rnd-1])
            if buffers["round_rewards"][rnd-1] else 0.0
        )
        total_reward += avg_reward
        avg_regret = (
            sum(buffers["round_regrets"][rnd-1]) / len(buffers["round_regrets"][rnd-1])
            if buffers["round_regrets"][rnd-1] else 0.0
        )
        outer_probs = (
            np.mean(buffers["round_outer_probs"][rnd-1], axis=0)
            if buffers["round_outer_probs"][rnd-1]
            else np.ones(1)
        )
        middle_probs = (
            np.mean(buffers["round_middle_probs"][rnd-1], axis=0)
            if buffers["round_middle_probs"][rnd-1]
            else np.ones(CFR_NUM_ACTIONS) / max(CFR_NUM_ACTIONS, 1)
        )
        search_rate = (
            sum(buffers["round_search_flags"][rnd-1]) / len(buffers["round_search_flags"][rnd-1])
            if buffers["round_search_flags"][rnd-1] else 0.0
        )
        stabilize_rate = (
            sum(buffers["round_stabilize_flags"][rnd-1]) / len(buffers["round_stabilize_flags"][rnd-1])
            if buffers["round_stabilize_flags"][rnd-1] else 0.0
        )
        improve_rate = (
            sum(buffers["round_improve_flags"][rnd-1]) / len(buffers["round_improve_flags"][rnd-1])
            if buffers["round_improve_flags"][rnd-1] else 0.0
        )
        degrade_rate = (
            sum(buffers["round_degrade_flags"][rnd-1]) / len(buffers["round_degrade_flags"][rnd-1])
            if buffers["round_degrade_flags"][rnd-1] else 0.0
        )
        stalled_wrong_rate = (
            sum(buffers["round_stalled_wrong_flags"][rnd-1]) / len(buffers["round_stalled_wrong_flags"][rnd-1])
            if buffers["round_stalled_wrong_flags"][rnd-1] else 0.0
        )
        preserved_correct_rate = (
            sum(buffers["round_preserved_correct_flags"][rnd-1]) / len(buffers["round_preserved_correct_flags"][rnd-1])
            if buffers["round_preserved_correct_flags"][rnd-1] else 0.0
        )

        print(
            f"{exp_name} 第{rnd}轮准确率：{acc:.4f} | Best-so-far：{best_acc:.4f} "
            f"| 累计奖励：{total_reward:.4f} | search={search_rate:.2f} "
            f"stabilize={stabilize_rate:.2f} | improve={improve_rate:.2f} "
            f"degrade={degrade_rate:.2f} stalled_wrong={stalled_wrong_rate:.2f}"
        )
        if log_metrics:
            log_fn(
                rnd,
                acc,
                best_acc,
                total_reward,
                avg_regret,
                outer_probs,
                middle_probs,
                search_rate,
                stabilize_rate,
                improve_rate,
                degrade_rate,
                stalled_wrong_rate,
                preserved_correct_rate,
            )

def finalize_three_layer_experiment(buffers):
    finalize_policy_stack_experiment(buffers, "全量策略", log_three_layer)

# ==============================================
# 实验1：单LLM
# ==============================================
def run_single_llm(data, exp_name="单LLM", log_metrics=True, log_key="single"):
    """
    实验1：单模型自己连续跑 5 轮。

    注意这里的外层循环：
        for idx, item in enumerate(...):

    这表示“按样本循环”。

    也就是说当前真实运行顺序是：
    - 样本1跑完5轮
    - 样本2跑完5轮
    - ...

    而不是：
    - 所有样本先跑第1轮
    - 再统一跑第2轮
    """
    print(f"\n=== 开始运行{exp_name}实验 ===")
    round_preds = [[] for _ in range(NUM_ROUNDS)]
    round_gts = [[] for _ in range(NUM_ROUNDS)]
    best_round_hits = [[] for _ in range(NUM_ROUNDS)]

    total_samples = len(data)
    for idx, item in enumerate(tqdm(data, desc="单LLM 按样本运行"), start=1):
        q = item["question"]
        gt = item["ground_truth"]
        history = []
        last_pred = None
        best_correct = False

        for rnd in range(1, NUM_ROUNDS+1):
            prompt = SINGLE_PROMPTS[rnd-1]
            ctx = build_context(q, history)
            res = generate_response(prompt.format(context=ctx))

            current_pred = extract_pred_num(res)
            is_answer_round = (rnd % 2 == 1)
            last_pred = update_last_pred(last_pred, current_pred, is_answer_round)
            history = append_round_output(
                history,
                rnd,
                "single_llm",
                res,
                "answer" if is_answer_round else "comment",
            )

            round_preds[rnd-1].append(last_pred)
            round_gts[rnd-1].append(gt)
            best_correct = best_correct or bool(compute_accuracy([last_pred], [gt]))
            best_round_hits[rnd-1].append(1.0 if best_correct else 0.0)

        maybe_report_accuracy(exp_name, idx, total_samples, round_preds, round_gts, best_round_hits)

    for rnd in range(1, NUM_ROUNDS+1):
        acc = compute_accuracy(round_preds[rnd-1], round_gts[rnd-1])
        best_acc = compute_best_so_far_accuracies(best_round_hits)[rnd-1]
        if log_metrics:
            log_single(rnd, acc, best_acc)
        print(f"{exp_name} 第{rnd}轮准确率：{acc:.4f} | Best-so-far：{best_acc:.4f}")

    return {
        "round_preds": round_preds,
        "round_gts": round_gts,
        "best_round_hits": best_round_hits,
    }

# ==============================================
# 实验2：双LLM轮询（解答+评论）
# ==============================================
def run_polling_two_llms(data, exp_name="双LLM轮询", log_metrics=True, log_key="polling"):
    """
    实验2：两个角色轮流说话。

    轮次角色固定为：
    - 第1轮 solver
    - 第2轮 commenter
    - 第3轮 solver
    - 第4轮 commenter
    - 第5轮 solver

    这里的“轮询”只是角色轮询，不是样本调度轮询。
    """
    print(f"\n=== 开始运行{exp_name}实验 ===")
    round_role = ["solver", "commenter", "solver", "commenter", "solver"]
    round_preds = [[] for _ in range(NUM_ROUNDS)]
    round_gts = [[] for _ in range(NUM_ROUNDS)]
    best_round_hits = [[] for _ in range(NUM_ROUNDS)]

    total_samples = len(data)
    for idx, item in enumerate(tqdm(data, desc="双LLM轮询 按样本运行"), start=1):
        q = item["question"]
        gt = item["ground_truth"]
        history = []
        last_pred = None
        best_correct = False

        for rnd in range(1, NUM_ROUNDS+1):
            role = round_role[rnd-1]
            ctx = build_context(q, history)
            if role == "solver":
                res = generate_response(SOLVER_PROMPT.format(context=ctx))
                current_pred = extract_pred_num(res)
                last_pred = update_last_pred(last_pred, current_pred, True)
            else:
                res = generate_response(COMMENTER_PROMPT.format(context=ctx))
            history = append_round_output(
                history,
                rnd,
                role,
                res,
                "answer" if role == "solver" else "comment",
            )
            round_preds[rnd-1].append(last_pred)
            round_gts[rnd-1].append(gt)
            best_correct = best_correct or bool(compute_accuracy([last_pred], [gt]))
            best_round_hits[rnd-1].append(1.0 if best_correct else 0.0)

        maybe_report_accuracy(exp_name, idx, total_samples, round_preds, round_gts, best_round_hits)

    for rnd in range(1, NUM_ROUNDS+1):
        acc = compute_accuracy(round_preds[rnd-1], round_gts[rnd-1])
        best_acc = compute_best_so_far_accuracies(best_round_hits)[rnd-1]
        if log_metrics:
            log_polling(rnd, acc, best_acc)
        print(f"{exp_name} 第{rnd}轮准确率：{acc:.4f} | Best-so-far：{best_acc:.4f}")

    return {
        "round_preds": round_preds,
        "round_gts": round_gts,
        "best_round_hits": best_round_hits,
    }

# ==============================================
# 实验3：单LLM GSPO
# ==============================================
def run_single_gspo(
    data,
    agent=None,
    update_params=True,
    log_metrics=True,
    exp_name="单LLM GSPO",
    teardown=True,
):
    """
    实验3：单智能体 GSPO。

    这个实验最容易误解的地方是：
    - 它训练时会采样多个候选
    - 但真实轨迹推进时，仍然只使用 selected_text 这一条

    所以它不是“best-of-G 测试时重排”，
    而是“单次真实动作 + 组内候选训练更新”。
    """
    print(f"\n=== 开始运行{exp_name}实验 ===")
    if agent is None:
        agent = GSPOAgentPolicy(is_coop=False)
    round_preds = [[] for _ in range(NUM_ROUNDS)]
    round_gts = [[] for _ in range(NUM_ROUNDS)]
    best_round_hits = [[] for _ in range(NUM_ROUNDS)]

    total_samples = len(data)
    for idx, item in enumerate(tqdm(data, desc="单LLM GSPO 按样本运行"), start=1):
        q = item["question"]
        gt = item["ground_truth"]
        history = []
        last_pred = None
        best_correct = False

        for rnd in range(1, NUM_ROUNDS+1):
            prompt = get_single_gspo_round_prompt(rnd)
            is_answer_round = (rnd % 2 == 1)
            speaker = "single_gspo_answer" if is_answer_round else "single_gspo_comment"
            step = sample_gspo_step(
                agent,
                q,
                history,
                rnd,
                speaker,
                "answer" if is_answer_round else "comment",
                prompt_override=prompt,
            )
            res = step["selected_text"]
            # 这里明确使用 selected_text，而不是 best_text。
            # 这保证了和 baseline 的“单次真实采样”比较是公平的。
            current_pred = extract_pred_num(res)
            last_pred = update_last_pred(last_pred, current_pred, is_answer_round)
            history = append_round_output(
                history,
                rnd,
                speaker,
                res,
                "answer" if is_answer_round else "comment",
            )
            if update_params:
                if is_answer_round:
                    rewards = [
                        reward_from_pred(extract_pred_num(cand["text"]), gt)
                        for cand in step["batch"]["candidates"]
                    ]
                else:
                    rewards = compute_comment_candidate_rewards(
                        step,
                        {"pi1": agent},
                        None,
                        None,
                        last_pred,
                        gt,
                    )
                agent.update_from_cached(step["batch"], rewards)

            round_preds[rnd-1].append(last_pred)
            round_gts[rnd-1].append(gt)
            best_correct = best_correct or bool(compute_accuracy([last_pred], [gt]))
            best_round_hits[rnd-1].append(1.0 if best_correct else 0.0)

        maybe_report_accuracy(exp_name, idx, total_samples, round_preds, round_gts, best_round_hits)

    for rnd in range(1, NUM_ROUNDS+1):
        acc = compute_accuracy(round_preds[rnd-1], round_gts[rnd-1])
        best_acc = compute_best_so_far_accuracies(best_round_hits)[rnd-1]
        if log_metrics:
            log_single_gspo(rnd, acc, best_acc)
        print(f"{exp_name} 第{rnd}轮准确率：{acc:.4f} | Best-so-far：{best_acc:.4f}")

    if teardown and agent is not None:
        agent.prepare_for_teardown()
        cleanup_cuda_memory(clear_cuda_cache=False)

    return agent, {
        "round_preds": round_preds,
        "round_gts": round_gts,
        "best_round_hits": best_round_hits,
    }

# ==============================================
# 实验4：双LLM GSPO轮询
# ==============================================
def run_dual_gspo(
    data,
    solver=None,
    commenter=None,
    update_params=True,
    log_metrics=True,
    exp_name="双LLM GSPO轮询",
    teardown=True,
):
    """
    实验4：双智能体 GSPO。

    与 run_polling_two_llms 的差别在于：
    - 角色顺序仍然一样
    - 但 solver / commenter 都不再是固定推理模型
    - 而是各自带一个会在线更新的 GSPO policy
    """
    print(f"\n=== 开始运行{exp_name}实验 ===")
    if solver is None:
        solver = GSPOAgentPolicy(is_coop=False)
    if commenter is None:
        commenter = GSPOAgentPolicy(is_coop=True)
    round_role = ["solver", "commenter", "solver", "commenter", "solver"]
    round_preds = [[] for _ in range(NUM_ROUNDS)]
    round_gts = [[] for _ in range(NUM_ROUNDS)]
    best_round_hits = [[] for _ in range(NUM_ROUNDS)]

    total_samples = len(data)
    for idx, item in enumerate(tqdm(data, desc="双LLM GSPO轮询 按样本运行"), start=1):
        q = item["question"]
        gt = item["ground_truth"]
        history = []
        last_pred = None
        best_correct = False

        for rnd in range(1, NUM_ROUNDS+1):
            role = round_role[rnd-1]
            if role == "solver":
                step = sample_gspo_step(
                    solver,
                    q,
                    history,
                    rnd,
                    role,
                    "answer",
                    prompt_override=SOLVER_PROMPT,
                )
                res = step["selected_text"]
                current_pred = extract_pred_num(res)
                last_pred = update_last_pred(last_pred, current_pred, True)
            else:
                step = sample_gspo_step(
                    commenter,
                    q,
                    history,
                    rnd,
                    role,
                    "comment",
                    prompt_override=COMMENTER_PROMPT,
                )
                res = step["selected_text"]

            history = append_round_output(
                history,
                rnd,
                role,
                res,
                "answer" if role == "solver" else "comment",
            )
            if update_params:
                if role == "solver":
                    rewards = [
                        reward_from_pred(extract_pred_num(cand["text"]), gt)
                        for cand in step["batch"]["candidates"]
                    ]
                    solver.update_from_cached(step["batch"], rewards)
                else:
                    rewards = compute_comment_candidate_rewards(
                        step,
                        {"pi1": solver},
                        None,
                        None,
                        last_pred,
                        gt,
                    )
                    commenter.update_from_cached(step["batch"], rewards)
            round_preds[rnd-1].append(last_pred)
            round_gts[rnd-1].append(gt)
            best_correct = best_correct or bool(compute_accuracy([last_pred], [gt]))
            best_round_hits[rnd-1].append(1.0 if best_correct else 0.0)

        maybe_report_accuracy(exp_name, idx, total_samples, round_preds, round_gts, best_round_hits)

    for rnd in range(1, NUM_ROUNDS+1):
        acc = compute_accuracy(round_preds[rnd-1], round_gts[rnd-1])
        best_acc = compute_best_so_far_accuracies(best_round_hits)[rnd-1]
        if log_metrics:
            log_dual_gspo(rnd, acc, best_acc)
        print(f"{exp_name} 第{rnd}轮准确率：{acc:.4f} | Best-so-far：{best_acc:.4f}")

    if teardown:
        if solver is not None:
            solver.prepare_for_teardown()
        if commenter is not None:
            commenter.prepare_for_teardown()
        cleanup_cuda_memory(clear_cuda_cache=False)

    return (solver, commenter), {
        "round_preds": round_preds,
        "round_gts": round_gts,
        "best_round_hits": best_round_hits,
    }

def build_policy_stack_runtime(use_outer_scheduler, num_agents, allow_silent=True):
    outer_cfr = None
    if use_outer_scheduler:
        outer_cfr = CFRBehaviorSelector(num_actions=num_agents, default_action=0)
    middle_cfr = CFRBehaviorSelector(
        num_actions=CFR_NUM_ACTIONS,
        default_action=MIDDLE_ACTION_SILENT,
    )
    verifier = OnlineAnswerVerifier()
    agents = []
    for _ in range(num_agents):
        pi0 = GSPOAgentPolicy(is_coop=True)
        pi1 = GSPOAgentPolicy(is_coop=False)
        agents.append({"pi0": pi0, "pi1": pi1, "pi2": None})
    return {
        "agents": agents,
        "outer_cfr": outer_cfr,
        "middle_cfr": middle_cfr,
        "verifier": verifier,
        "use_outer_scheduler": use_outer_scheduler,
        "num_agents": num_agents,
        "allow_silent": allow_silent,
    }

def run_policy_stack_experiment(
    data,
    runtime=None,
    use_outer_scheduler=True,
    num_agents=None,
    allow_silent=True,
    update_params=True,
    log_metrics=True,
    exp_name="全量策略",
    log_fn=log_three_layer,
    teardown=True,
):
    print(f"\n=== 开始运行{exp_name}实验 ===")
    if runtime is None:
        resolved_num_agents = NUM_AGENTS if num_agents is None else num_agents
        runtime = build_policy_stack_runtime(
            use_outer_scheduler,
            resolved_num_agents,
            allow_silent=allow_silent,
        )

    agents = runtime["agents"]
    outer_cfr = runtime["outer_cfr"]
    middle_cfr = runtime["middle_cfr"]
    verifier = runtime["verifier"]
    resolved_num_agents = runtime["num_agents"]
    resolved_allow_silent = runtime.get("allow_silent", allow_silent)

    buffers = init_three_layer_round_buffers()
    total_samples = len(data)
    for idx, item in enumerate(tqdm(data, desc=f"{exp_name} 按样本运行"), start=1):
        q, gt = item["question"], item["ground_truth"]
        sample_state = init_three_layer_sample_state()
        sample_state["incumbent_reward"] = reward_from_pred(
            sample_state["incumbent_pred"],
            gt,
        )

        for rnd in range(1, NUM_ROUNDS+1):
            round_start_time = time.perf_counter()
            pre_history = sample_state["history"]
            pre_incumbent_text = sample_state["incumbent_text"]
            pre_incumbent_pred = sample_state["incumbent_pred"]

            phase, keep_prob, stability_score = compute_phase(
                verifier,
                q,
                pre_history,
                pre_incumbent_text,
                rnd,
            )

            if use_outer_scheduler:
                outer_state, allowed_agents, outer_strategy, selected_agent = choose_outer_agent(
                    outer_cfr,
                    phase,
                    num_agents=resolved_num_agents,
                )
            else:
                outer_state = None
                allowed_agents = [0]
                outer_strategy = np.array([1.0], dtype=np.float64)
                selected_agent = 0

            middle_state, allowed_actions, middle_strategy, act = choose_middle_action(
                middle_cfr,
                phase,
                rnd,
                selected_agent,
                pre_incumbent_pred,
                keep_prob,
                allow_silent=resolved_allow_silent,
            )

            round_result = run_three_layer_realized_action(
                agents,
                verifier,
                idx,
                q,
                gt,
                rnd,
                selected_agent,
                act,
                phase,
                pre_history,
                pre_incumbent_text,
                pre_incumbent_pred,
                middle_strategy,
            )
            sample_state["history"] = round_result["history"]
            sample_state["incumbent_text"] = round_result["incumbent_text"]
            sample_state["incumbent_pred"] = round_result["incumbent_pred"]
            sample_state["incumbent_reward"] = round_result["incumbent_reward"]
            maybe_print_three_layer_stage(idx, rnd, "realized_action", round_start_time)

            if update_params:
                update_three_layer_regrets(
                    outer_cfr,
                    middle_cfr,
                    agents,
                    verifier,
                    q,
                    gt,
                    rnd,
                    phase,
                    selected_agent,
                    act,
                    outer_state,
                    allowed_agents,
                    middle_state,
                    allowed_actions,
                    pre_history,
                    pre_incumbent_text,
                    pre_incumbent_pred,
                    keep_prob,
                    resolved_allow_silent,
                    round_result["realized_value"],
                    round_result["outer_value"],
                )
                maybe_print_three_layer_stage(idx, rnd, "middle_regret", round_start_time)
                if use_outer_scheduler:
                    maybe_print_three_layer_stage(idx, rnd, "outer_regret", round_start_time)

                accept_update_stats = apply_three_layer_policy_updates(
                    verifier,
                    round_result["gspo_update_payload"],
                    round_result["accept_update_payload"],
                    q,
                    sample_state["history"],
                    sample_state["incumbent_text"],
                    sample_state["incumbent_reward"],
                    should_update_keep=(
                        round_result["has_answer_candidates"]
                        and round_result["incumbent_changed"]
                    ),
                )
                maybe_print_three_layer_stage(idx, rnd, "policy_update", round_start_time)
                maybe_print_accept_training_debug(
                    idx,
                    rnd,
                    pre_incumbent_pred,
                    round_result["current_pred"],
                    round_result["accepted"],
                    accept_update_stats,
                )

            if act != MIDDLE_ACTION_ANSWER:
                sample_state["incumbent_reward"] = reward_from_pred(
                    sample_state["incumbent_pred"],
                    gt,
                )

            transition = classify_round_transition(
                pre_incumbent_pred,
                sample_state["incumbent_pred"],
                gt,
            )
            if transition["improved"] and sample_state["first_improve_round"] is None:
                sample_state["first_improve_round"] = rnd
            if transition["degraded"] and sample_state["first_degrade_round"] is None:
                sample_state["first_degrade_round"] = rnd
            if transition["stalled_wrong"] and sample_state["first_stalled_wrong_round"] is None:
                sample_state["first_stalled_wrong_round"] = rnd

            sample_state["best_correct"] = sample_state["best_correct"] or bool(
                compute_accuracy([sample_state["incumbent_pred"]], [gt])
            )
            record_three_layer_round(
                buffers,
                rnd,
                gt,
                sample_state["incumbent_pred"],
                sample_state["best_correct"],
                round_result["realized_value"],
                outer_cfr,
                middle_cfr,
                outer_strategy,
                middle_strategy,
                phase,
                transition,
            )
            maybe_print_three_layer_round_debug(
                idx,
                rnd,
                phase,
                keep_prob,
                stability_score,
                selected_agent,
                act,
                round_result["current_pred"],
                sample_state["incumbent_pred"],
                gt,
                round_result["realized_value"],
                outer_strategy,
                middle_strategy,
                round_result["accepted"],
                round_result["verifier_prob"],
                round_result["accept_threshold"],
                round_result["accept_keep_prob"],
                round_result["accept_barrier"],
                round_result["accept_margin"],
                round_result["accept_reason"],
            )

        maybe_report_accuracy(
            exp_name,
            idx,
            total_samples,
            buffers["round_preds"],
            buffers["round_gts"],
            buffers["best_round_hits"],
        )
        print_three_layer_sample_monitor(
            idx,
            gt,
            sample_state["incumbent_pred"],
            sample_state["first_improve_round"],
            sample_state["first_degrade_round"],
            sample_state["first_stalled_wrong_round"],
        )

    finalize_policy_stack_experiment(
        buffers,
        exp_name,
        log_fn,
        log_metrics=log_metrics,
    )

    if teardown:
        agents, outer_cfr, middle_cfr, verifier = teardown_three_layer_runtime(
            agents,
            outer_cfr,
            middle_cfr,
            verifier,
        )
        runtime = None
    else:
        runtime = {
            "agents": agents,
            "outer_cfr": outer_cfr,
            "middle_cfr": middle_cfr,
            "verifier": verifier,
            "use_outer_scheduler": use_outer_scheduler,
            "num_agents": resolved_num_agents,
            "allow_silent": resolved_allow_silent,
        }

    return runtime, buffers

# ==============================================
# 实验5：保留中间层策略和最内层GSPO
# ==============================================
def run_middle_layer(
    data,
    runtime=None,
    update_params=True,
    log_metrics=True,
    exp_name="中间层策略+GSPO",
    teardown=True,
):
    return run_policy_stack_experiment(
        data,
        runtime=runtime,
        use_outer_scheduler=False,
        num_agents=1,
        allow_silent=True,
        update_params=update_params,
        log_metrics=log_metrics,
        exp_name=exp_name,
        log_fn=log_middle_layer,
        teardown=teardown,
    )

# ==============================================
# 实验6：保留中间层策略和最内层GSPO（中层无沉默）
# ==============================================
def run_middle_layer_no_silent(
    data,
    runtime=None,
    update_params=True,
    log_metrics=True,
    exp_name="中间层策略+GSPO（无沉默）",
    teardown=True,
):
    return run_policy_stack_experiment(
        data,
        runtime=runtime,
        use_outer_scheduler=False,
        num_agents=1,
        allow_silent=False,
        update_params=update_params,
        log_metrics=log_metrics,
        exp_name=exp_name,
        log_fn=log_middle_layer_no_silent,
        teardown=teardown,
    )

# ==============================================
# 实验7：全量策略（调度器 + 中间层策略 + GSPO）
# ==============================================
def run_three_layer(
    data,
    runtime=None,
    update_params=True,
    log_metrics=True,
    exp_name="全量策略",
    teardown=True,
):
    return run_policy_stack_experiment(
        data,
        runtime=runtime,
        use_outer_scheduler=True,
        num_agents=NUM_AGENTS,
        allow_silent=True,
        update_params=update_params,
        log_metrics=log_metrics,
        exp_name=exp_name,
        log_fn=log_three_layer,
        teardown=teardown,
    )

# ==============================================
# 实验8：全量策略（中层无沉默）
# ==============================================
def run_three_layer_no_silent(
    data,
    runtime=None,
    update_params=True,
    log_metrics=True,
    exp_name="全量策略（中层无沉默）",
    teardown=True,
):
    return run_policy_stack_experiment(
        data,
        runtime=runtime,
        use_outer_scheduler=True,
        num_agents=NUM_AGENTS,
        allow_silent=False,
        update_params=update_params,
        log_metrics=log_metrics,
        exp_name=exp_name,
        log_fn=log_three_layer_no_silent,
        teardown=teardown,
    )

def run_experiment_suite(train_data, val_data, test_data):
    print(f"数据划分：{build_split_size_message(train_data, val_data, test_data)}")

    run_single_llm(test_data, exp_name="单LLM", log_metrics=True)
    run_polling_two_llms(test_data, exp_name="双LLM轮询", log_metrics=True)

    single_agent, _ = run_single_gspo(
        train_data,
        update_params=True,
        log_metrics=False,
        exp_name="单LLM GSPO 训练",
        teardown=False,
    )
    run_single_gspo(
        val_data,
        agent=single_agent,
        update_params=False,
        log_metrics=False,
        exp_name="单LLM GSPO 验证",
        teardown=False,
    )
    run_single_gspo(
        test_data,
        agent=single_agent,
        update_params=False,
        log_metrics=True,
        exp_name="单LLM GSPO",
        teardown=True,
    )

    dual_runtime, _ = run_dual_gspo(
        train_data,
        update_params=True,
        log_metrics=False,
        exp_name="双LLM GSPO轮询 训练",
        teardown=False,
    )
    run_dual_gspo(
        val_data,
        solver=dual_runtime[0],
        commenter=dual_runtime[1],
        update_params=False,
        log_metrics=False,
        exp_name="双LLM GSPO轮询 验证",
        teardown=False,
    )
    run_dual_gspo(
        test_data,
        solver=dual_runtime[0],
        commenter=dual_runtime[1],
        update_params=False,
        log_metrics=True,
        exp_name="双LLM GSPO轮询",
        teardown=True,
    )

    middle_runtime, _ = run_middle_layer(
        train_data,
        update_params=True,
        log_metrics=False,
        exp_name="中间层策略+GSPO 训练",
        teardown=False,
    )
    run_middle_layer(
        val_data,
        runtime=middle_runtime,
        update_params=False,
        log_metrics=False,
        exp_name="中间层策略+GSPO 验证",
        teardown=False,
    )
    run_middle_layer(
        test_data,
        runtime=middle_runtime,
        update_params=False,
        log_metrics=True,
        exp_name="中间层策略+GSPO",
        teardown=True,
    )

    middle_no_silent_runtime, _ = run_middle_layer_no_silent(
        train_data,
        update_params=True,
        log_metrics=False,
        exp_name="中间层策略+GSPO（无沉默） 训练",
        teardown=False,
    )
    run_middle_layer_no_silent(
        val_data,
        runtime=middle_no_silent_runtime,
        update_params=False,
        log_metrics=False,
        exp_name="中间层策略+GSPO（无沉默） 验证",
        teardown=False,
    )
    run_middle_layer_no_silent(
        test_data,
        runtime=middle_no_silent_runtime,
        update_params=False,
        log_metrics=True,
        exp_name="中间层策略+GSPO（无沉默）",
        teardown=True,
    )

    full_runtime, _ = run_three_layer(
        train_data,
        update_params=True,
        log_metrics=False,
        exp_name="全量策略 训练",
        teardown=False,
    )
    run_three_layer(
        val_data,
        runtime=full_runtime,
        update_params=False,
        log_metrics=False,
        exp_name="全量策略 验证",
        teardown=False,
    )
    run_three_layer(
        test_data,
        runtime=full_runtime,
        update_params=False,
        log_metrics=True,
        exp_name="全量策略",
        teardown=True,
    )

    full_no_silent_runtime, _ = run_three_layer_no_silent(
        train_data,
        update_params=True,
        log_metrics=False,
        exp_name="全量策略（中层无沉默） 训练",
        teardown=False,
    )
    run_three_layer_no_silent(
        val_data,
        runtime=full_no_silent_runtime,
        update_params=False,
        log_metrics=False,
        exp_name="全量策略（中层无沉默） 验证",
        teardown=False,
    )
    run_three_layer_no_silent(
        test_data,
        runtime=full_no_silent_runtime,
        update_params=False,
        log_metrics=True,
        exp_name="全量策略（中层无沉默）",
        teardown=True,
    )

# ==============================================
# 主函数
# ==============================================
def main():
    """
    整个脚本的入口函数。

    如果你把这个文件当程序看，
    main() 就像“总流程按钮”：
    - 初始化
    - 加载数据
    - 依次跑 8 个实验
    - 画图
    - 输出结果路径
    """
    print("="*50)
    print("  多智能体GSM8K实验（GSPO+CFR）")
    print("="*50)
    
    print("\n[1/5] 初始化日志...")
    init_logs()
    
    print("\n[2/5] 加载GSM8K数据集并划分 train/val/test...")
    splits = load_gsm8k_splits()
    print("成功加载数据：")
    print(f"  - train: {len(splits['train'])}")
    print(f"  - val:   {len(splits['val'])}")
    print(f"  - test:  {len(splits['test'])}")
    
    print("\n[3/5] 运行实验...")
    run_experiment_suite(
        splits["train"],
        splits["val"],
        splits["test"],
    )
    
    print("\n[4/5] 生成可视化图表...")
    plot_accuracy_comparison()
    plot_three_layer_details()
    
    print("\n[5/5] 所有实验完成！")
    print("="*50)
    print("  结果文件：")
    print(f"  - 实验日志：{LOG_DIR}/")
    print(f"  - 可视化图：{PLOT_DIR}/")
    print("="*50)

if __name__ == "__main__":
    main()
