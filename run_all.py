import os
import sys
import gc
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
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
from src.parallel_runtime import create_three_layer_parallel_controller
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
# 改动3：准确率始终看“最近一次真正产生的 answer”
# - comment / silent 轮本身不产生新答案，所以准确率沿用上一轮 answer
# - 但只要这一轮产生了新的 answer，就必须用这次最新 answer 来统计
#   即使这次 answer 不可解析，准确率也应按这次最新 answer 记分，而不是偷偷沿用旧答案
#
# 改动4：GSPO 奖励改成“延迟更新”
# - comment 先只缓存，不立刻更新
# - 等这个段落里真的出现 answer，再回头统一给这段里的 step 分配奖励
#
# 改动5：日志里同时保留两个指标
# - accuracy: 当前这一轮手里那份答案是否正确（主指标）
# - best_so_far_accuracy: 到当前轮为止，历史上是否曾经答对过（辅助指标）
# ============================================================

REVIEW_REASON_PATTERN = re.compile(
    r"^\s*reason\s*[:：]\s*(.+?)\s*$",
    re.IGNORECASE,
)
REVIEW_JUDGMENT_TOKEN_PATTERN = (
    r"(?:right|wrong|correct|incorrect|accept|reject|accepted|rejected)"
)
REVIEW_JUDGMENT_LINE_PATTERN = re.compile(
    rf"^\s*judg(?:e)?ment\s*[:：]\s*({REVIEW_JUDGMENT_TOKEN_PATTERN})\s*\.?\s*$",
    re.IGNORECASE,
)
REVIEW_NATURAL_JUDGMENT_PATTERN = re.compile(
    rf"^\s*(?:therefore,\s*)?(?:the\s+)?judg(?:e)?ment\s+(?:is\s+)?({REVIEW_JUDGMENT_TOKEN_PATTERN})\s*\.?\s*$",
    re.IGNORECASE,
)
REVIEW_INLINE_TAIL_JUDGMENT_PATTERN = re.compile(
    rf"^(?P<reason>.*?)(?:\s+|^)(?:judg(?:e)?ment)\s*[:：]\s*(?P<judgment>{REVIEW_JUDGMENT_TOKEN_PATTERN})\s*\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)
REVIEW_INLINE_TAIL_NATURAL_JUDGMENT_PATTERN = re.compile(
    rf"^(?P<reason>.*?)(?:\s+|^)(?:the\s+)?judg(?:e)?ment\s+(?:is\s+)?(?P<judgment>{REVIEW_JUDGMENT_TOKEN_PATTERN})\s*\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)
TEXT_NUMBER_PATTERN = re.compile(
    r"(?<![\w/.-])-?\$?\d[\d,]*\.?\d*"
)
PROPOSAL_CANDIDATE_PATTERN = re.compile(
    r"^\s*candidate\s*[:：]\s*(-?\$?\d[\d,]*\.?\d*)\s*$",
    re.IGNORECASE,
)
PROPOSAL_EMPTY_CANDIDATE_PATTERN = re.compile(
    r"^\s*candidate\s*[:：]\s*$",
    re.IGNORECASE,
)
PROPOSAL_ANSWER_LINE_PATTERN = re.compile(
    r"^\s*(?:therefore,\s*)?(?:the\s+)?(?:candidate|answer|final answer)\s*(?:is\s+|[:：]\s*)(-?\$?\d[\d,]*\.?\d*)\s*\.?\s*$",
    re.IGNORECASE,
)
PROPOSAL_ANSWER_PREFIX_PATTERN = re.compile(
    r"^\s*(?:therefore,\s*)?(?:the\s+)?(?:candidate|answer|final answer)\s*(?:is\s+|[:：]\s*)(?P<tail>.+?)\s*$",
    re.IGNORECASE,
)
PROPOSAL_HASH_ANSWER_PATTERN = re.compile(
    r"^\s*####\s*(?P<tail>.+?)\s*$",
    re.IGNORECASE,
)
PROPOSAL_REASON_FINAL_EQUALS_PATTERN = re.compile(
    r"=\s*(-?\$?\d[\d,]*\.?\d*)\b"
)
PROPOSAL_REASON_TRAILING_NUMBER_PATTERN = re.compile(
    r"(-?\$?\d[\d,]*\.?\d*)\s*\.?\s*$"
)
PROPOSAL_LEADING_STANDALONE_NUMBER_PATTERN = re.compile(
    r"^\s*(-?\$?\d[\d,]*\.?\d*)\b"
)
REVIEW_REASON_PREFERRED_ANSWER_PATTERN = re.compile(
    r"(?:should\s+be|should\s+instead\s+be|correct\s+answer\s+is|answer\s+is|gives|equals)\s*(-?\$?\d[\d,]*\.?\d*)",
    re.IGNORECASE,
)
REVIEW_REASON_X_NOT_Y_PATTERN = re.compile(
    r"(-?\$?\d[\d,]*\.?\d*)\s*,?\s*(?:not|rather than|instead of)\s*(-?\$?\d[\d,]*\.?\d*)",
    re.IGNORECASE,
)
REVIEW_REASON_NOT_X_BUT_Y_PATTERN = re.compile(
    r"not\s*(-?\$?\d[\d,]*\.?\d*)\s*(?:but|instead)\s*(-?\$?\d[\d,]*\.?\d*)",
    re.IGNORECASE,
)
REVIEW_TARGET_ANSWER_CUE_PATTERN = re.compile(
    r"(?:correct\s+(?:final\s+)?answer\s*(?:is|should\s+be|[:：])|revised\s+answer\s*[:：]|replacement\s+answer\s*[:：]|final\s+answer\s+(?:is|should\s+be|[:：])|answer\s+(?:is|should\s+be)|result\s+(?:is|should\s+be)|replace\s+it\s+with|should\s+replace\s+it)",
    re.IGNORECASE,
)
REVIEW_EXPLICIT_ACCEPT_PATTERN = re.compile(
    r"(?:current\s+pending\s+final\s+answer|pending\s+final\s+answer|pending\s+answer|current\s+answer|final\s+answer|answer|calculation|solution|this)\s+(?:is\s+|looks\s+|seems\s+|appears\s+|remains\s+)?(?:correct|right|valid|accurate)\b|(?:i\s+agree\s+with\s+(?:the\s+)?(?:pending\s+)?(?:answer|solution))|(?:matches?\s+the\s+(?:problem|given\s+information)|consistent\s+with\s+the\s+(?:problem|given\s+information)|correct\s+as\s+written|checks?\s+out|looks?\s+good)",
    re.IGNORECASE,
)
REVIEW_EXPLICIT_REJECT_PATTERN = re.compile(
    r"(?:current\s+pending\s+final\s+answer|pending\s+final\s+answer|pending\s+answer|current\s+answer|final\s+answer|answer|calculation|solution|this)\s+(?:is\s+|looks\s+|seems\s+|appears\s+)?(?:incorrect|wrong|invalid)\b|(?:i\s+disagree\s+with\s+(?:the\s+)?(?:pending\s+)?(?:answer|solution))|(?:cannot\s+be\s+confirmed\s+as\s+correct|not\s+correct|should\s+be\s+rejected|does\s+not\s+match|does\s+not\s+account\s+for|misses?|left\s+out|should\s+include|should\s+account\s+for)",
    re.IGNORECASE,
)

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

def teardown_three_layer_runtime(
    agents,
    outer_cfr,
    middle_cfr,
    parallel_controller=None,
):
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
    print("[三层teardown] drop_refs done", flush=True)

    print("[三层teardown] gc_only_cleanup start", flush=True)
    cleanup_cuda_memory(clear_cuda_cache=False)
    print("[三层teardown] gc_only_cleanup done", flush=True)
    if parallel_controller is not None:
        print("[三层teardown] worker_shutdown start", flush=True)
        parallel_controller.shutdown()
        parallel_controller = None
        print("[三层teardown] worker_shutdown done", flush=True)
    print("[三层teardown] finished", flush=True)

    return agents, outer_cfr, middle_cfr, parallel_controller


def get_run_root_dir():
    normalized_log_dir = os.path.normpath(LOG_DIR)
    if os.path.basename(normalized_log_dir) == "logs":
        return os.path.dirname(normalized_log_dir)
    return normalized_log_dir


def capture_main_process_rng_state():
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state().cpu(),
        "cuda_rng_state_all": None,
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = [
            torch.cuda.get_rng_state(device_idx).cpu()
            for device_idx in range(torch.cuda.device_count())
        ]
    return state


def restore_main_process_rng_state(state):
    if not state:
        return
    python_random_state = state.get("python_random_state")
    if python_random_state is not None:
        random.setstate(python_random_state)
    numpy_random_state = state.get("numpy_random_state")
    if numpy_random_state is not None:
        np.random.set_state(numpy_random_state)
    torch_rng_state = state.get("torch_rng_state")
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state.cpu())
    cuda_rng_state_all = state.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        for device_idx, rng_state in enumerate(cuda_rng_state_all):
            if device_idx < torch.cuda.device_count():
                torch.cuda.set_rng_state(rng_state.cpu(), device=device_idx)


def iter_runtime_policies(agents):
    for agent_idx, agent_bundle in enumerate(agents):
        for policy_name in ("pi0", "pi1", "pi2"):
            policy = agent_bundle.get(policy_name)
            if policy is not None:
                yield agent_idx, policy_name, policy


def resolve_checkpoint_meta_path(checkpoint_path):
    if not checkpoint_path:
        return None

    resolved_path = os.path.abspath(checkpoint_path)
    if os.path.isdir(resolved_path):
        meta_path = os.path.join(resolved_path, "meta.pt")
        latest_path = os.path.join(resolved_path, "latest_checkpoint.txt")
        if os.path.isfile(meta_path):
            return meta_path
        if os.path.isfile(latest_path):
            with open(latest_path, "r", encoding="utf-8") as handle:
                pointed_path = handle.read().strip()
            if not pointed_path:
                raise ValueError(f"checkpoint 指针文件为空：{latest_path}")
            return resolve_checkpoint_meta_path(pointed_path)
        raise FileNotFoundError(f"checkpoint 目录下未找到 meta.pt：{resolved_path}")

    if os.path.isfile(resolved_path):
        if resolved_path.endswith(".pt"):
            return resolved_path
        with open(resolved_path, "r", encoding="utf-8") as handle:
            pointed_path = handle.read().strip()
        if not pointed_path:
            raise ValueError(f"checkpoint 指针文件为空：{resolved_path}")
        return resolve_checkpoint_meta_path(pointed_path)

    raise FileNotFoundError(f"checkpoint 路径不存在：{resolved_path}")


def save_policy_stack_checkpoint(
    runtime,
    buffers,
    sample_index_completed,
    total_samples,
    num_rounds,
    exp_name,
    checkpoint_dir,
):
    checkpoint_root = checkpoint_dir or CHECKPOINT_DIR
    checkpoint_name = f"sample_{int(sample_index_completed):06d}"
    checkpoint_path = os.path.join(checkpoint_root, checkpoint_name)
    policies_dir = os.path.join(checkpoint_path, "policies")
    os.makedirs(policies_dir, exist_ok=True)

    print(
        f"[checkpoint] start exp={exp_name} sample={sample_index_completed}/{total_samples} dir={checkpoint_path}",
        flush=True,
    )

    policy_files = []
    for agent_idx, policy_name, policy in iter_runtime_policies(runtime["agents"]):
        filename = f"agent{agent_idx}_{policy_name}.pt"
        policy_path = os.path.join(policies_dir, filename)
        policy.save_checkpoint(policy_path)
        policy_files.append(
            {
                "agent_idx": int(agent_idx),
                "policy_name": policy_name,
                "path": policy_path,
            }
        )

    meta = {
        "checkpoint_version": 1,
        "exp_name": exp_name,
        "sample_index_completed": int(sample_index_completed),
        "total_samples": int(total_samples),
        "num_rounds": int(num_rounds),
        "use_outer_scheduler": bool(runtime["use_outer_scheduler"]),
        "num_agents": int(runtime["num_agents"]),
        "allow_silent": bool(runtime["allow_silent"]),
        "parallel_mode": runtime.get("parallel_mode", "serial"),
        "outer_cfr_state": (
            runtime["outer_cfr"].get_checkpoint_state()
            if runtime["outer_cfr"] is not None else None
        ),
        "middle_cfr_state": runtime["middle_cfr"].get_checkpoint_state(),
        "buffers": buffers,
        "main_rng_state": capture_main_process_rng_state(),
        "policy_files": policy_files,
    }
    meta_path = os.path.join(checkpoint_path, "meta.pt")
    torch.save(meta, meta_path)

    latest_pointer_path = os.path.join(checkpoint_root, "latest_checkpoint.txt")
    os.makedirs(checkpoint_root, exist_ok=True)
    with open(latest_pointer_path, "w", encoding="utf-8") as handle:
        handle.write(meta_path)

    print(
        f"[checkpoint] saved exp={exp_name} sample={sample_index_completed}/{total_samples} meta={meta_path}",
        flush=True,
    )
    return meta_path


def load_policy_stack_checkpoint(
    runtime,
    checkpoint_path,
    expected_total_samples=None,
    expected_num_rounds=None,
):
    meta_path = resolve_checkpoint_meta_path(checkpoint_path)
    # 这里的 meta checkpoint 由本仓库本地生成，包含 numpy/python RNG 等非纯 tensor 状态。
    # PyTorch 2.6+ 默认 weights_only=True，会导致这类状态恢复失败，因此显式关闭。
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)

    if expected_total_samples is not None and int(meta["total_samples"]) != int(expected_total_samples):
        raise ValueError(
            f"checkpoint 样本数不匹配：当前={expected_total_samples} checkpoint={meta['total_samples']}"
        )
    if expected_num_rounds is not None and int(meta["num_rounds"]) != int(expected_num_rounds):
        raise ValueError(
            f"checkpoint 轮次数不匹配：当前={expected_num_rounds} checkpoint={meta['num_rounds']}"
        )

    outer_cfr_state = meta.get("outer_cfr_state")
    if runtime["outer_cfr"] is not None and outer_cfr_state is not None:
        runtime["outer_cfr"].load_checkpoint_state(outer_cfr_state)
    runtime["middle_cfr"].load_checkpoint_state(meta["middle_cfr_state"])

    for policy_record in meta.get("policy_files", []):
        agent_idx = int(policy_record["agent_idx"])
        policy_name = policy_record["policy_name"]
        policy = runtime["agents"][agent_idx][policy_name]
        policy.load_checkpoint(policy_record["path"])

    restore_main_process_rng_state(meta.get("main_rng_state"))
    print(
        f"[checkpoint] loaded meta={meta_path} sample={meta['sample_index_completed']}/{meta['total_samples']}",
        flush=True,
    )
    return {
        "meta_path": meta_path,
        "sample_index_completed": int(meta["sample_index_completed"]),
        "buffers": meta["buffers"],
    }

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

    if STRUCTURED_CONTEXT_MODE == "recent_window":
        visible_turns = [
            turn for turn in history
            if turn["kind"] in {"answer", "comment"}
        ]
        return visible_turns[-STRUCTURED_CONTEXT_MAX_TURNS:]

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

def is_proposal_review_schema():
    return THREE_LAYER_MIDDLE_ACTION_SCHEMA == "proposal_review"

def has_pending_candidate(pending_candidate_text=None, pending_candidate_pred=None):
    return bool((pending_candidate_text or "").strip())

def build_proposal_review_context(
    question,
    history,
    incumbent_text,
    pending_candidate_text,
    pending_vote_score=0,
    include_latest_review_reason=True,
):
    parts = [f"QUESTION:\n{question.strip()}"]
    if pending_candidate_text:
        parts.append(f"CURRENT PENDING SOLUTION:\n{pending_candidate_text.strip()}")
        if include_latest_review_reason:
            latest_review_feedback = get_latest_pending_review_feedback(
                history,
                pending_candidate_text,
            )
            if latest_review_feedback:
                parts.append(f"LATEST REVIEW FEEDBACK:\n{latest_review_feedback}")
    return "\n\n".join(parts)

def _split_nonempty_lines(text):
    return [
        line.strip()
        for line in str(text or "").replace("\r\n", "\n").split("\n")
        if line.strip()
    ]

def _last_nonempty_line(text):
    lines = _split_nonempty_lines(text)
    if not lines:
        return None
    return lines[-1]

def _normalize_review_judgment_label(label):
    cleaned = str(label or "").strip().lower()
    cleaned = re.sub(r"^[\s:：-]+|[\s\.\!\?;:：-]+$", "", cleaned)
    mapping = {
        "right": "right",
        "correct": "right",
        "accept": "right",
        "accepted": "right",
        "wrong": "wrong",
        "incorrect": "wrong",
        "reject": "wrong",
        "rejected": "wrong",
    }
    return mapping.get(cleaned)

def _is_bare_review_judgment(text):
    return _normalize_review_judgment_label(text) in {"right", "wrong"}

def _split_inline_tail_review_judgment(text):
    cleaned_text = str(text or "").strip()
    if not cleaned_text:
        return None, None
    for pattern in (
        REVIEW_INLINE_TAIL_JUDGMENT_PATTERN,
        REVIEW_INLINE_TAIL_NATURAL_JUDGMENT_PATTERN,
    ):
        match = pattern.match(cleaned_text)
        if not match:
            continue
        reason = (match.group("reason") or "").strip()
        judgment = _normalize_review_judgment_label(match.group("judgment"))
        if judgment in {"right", "wrong"}:
            return reason, judgment
    return None, None

def _extract_review_judgment_from_lines(lines):
    if not lines:
        return None
    first_line = lines[0]
    match = REVIEW_JUDGMENT_LINE_PATTERN.match(first_line)
    if match:
        return _normalize_review_judgment_label(match.group(1))
    match = REVIEW_NATURAL_JUDGMENT_PATTERN.match(first_line)
    if match:
        return _normalize_review_judgment_label(match.group(1))
    inline_reason, inline_judgment = _split_inline_tail_review_judgment(first_line)
    if inline_judgment is not None and not inline_reason:
        return inline_judgment
    return _normalize_review_judgment_label(first_line)

def _get_review_intent_segments(text):
    segments = []
    for line in _split_nonempty_lines(text):
        cleaned = line.strip()
        if not cleaned:
            continue
        inline_reason, inline_judgment = _split_inline_tail_review_judgment(cleaned)
        if inline_judgment is not None:
            if inline_reason:
                segments.append(inline_reason.strip())
            continue
        match = REVIEW_REASON_PATTERN.match(cleaned)
        if match:
            cleaned = match.group(1).strip()
        if cleaned:
            segments.append(cleaned)
    return segments

def _extract_review_target_pred(text):
    lines = _split_nonempty_lines(text)
    for line in reversed(lines):
        match = PROPOSAL_ANSWER_LINE_PATTERN.match(line)
        if match:
            pred = _parse_numeric_token(match.group(1))
            if pred is not None:
                return pred
        if "\\boxed" in line.lower():
            pred = extract_pred_num(line)
            if pred is not None:
                return pred

    reason = _clean_review_feedback_text(_extract_raw_review_reason(text))
    candidate_texts = []
    if reason:
        candidate_texts.append(reason)
    candidate_texts.extend(_get_review_intent_segments(text))
    candidate_texts.extend(lines)
    seen = set()
    for candidate_text in candidate_texts:
        if not candidate_text or candidate_text in seen:
            continue
        seen.add(candidate_text)
        match = PROPOSAL_ANSWER_LINE_PATTERN.match(candidate_text)
        if match:
            pred = _parse_numeric_token(match.group(1))
            if pred is not None:
                return pred
        preferred_matches = REVIEW_REASON_PREFERRED_ANSWER_PATTERN.findall(candidate_text)
        if preferred_matches:
            pred = _parse_numeric_token(preferred_matches[-1])
            if pred is not None:
                return pred
        comparison_match = REVIEW_REASON_X_NOT_Y_PATTERN.search(candidate_text)
        if comparison_match:
            pred = _parse_numeric_token(comparison_match.group(1))
            if pred is not None:
                return pred
        negation_match = REVIEW_REASON_NOT_X_BUT_Y_PATTERN.search(candidate_text)
        if negation_match:
            pred = _parse_numeric_token(negation_match.group(2))
            if pred is not None:
                return pred
        if not REVIEW_TARGET_ANSWER_CUE_PATTERN.search(candidate_text):
            continue
        pred = _extract_reason_pred_from_text(candidate_text)
        if pred is not None:
            return pred
    return None

def _extract_raw_review_reason(text):
    lines = _split_nonempty_lines(text)
    if not lines:
        return None
    reason_lines = []
    for idx, line in enumerate(lines):
        if idx == 0:
            match = REVIEW_JUDGMENT_LINE_PATTERN.match(line)
            if match:
                continue
            match = REVIEW_NATURAL_JUDGMENT_PATTERN.match(line)
            if match:
                continue
            if _is_bare_review_judgment(line):
                continue
        inline_reason, inline_judgment = _split_inline_tail_review_judgment(line)
        if inline_judgment is not None:
            if inline_reason:
                reason_lines.append(inline_reason)
            continue
        match = REVIEW_REASON_PATTERN.match(line)
        if match:
            reason = match.group(1).strip()
            if reason:
                reason_lines.append(reason)
            continue
        if REVIEW_JUDGMENT_LINE_PATTERN.match(line) or REVIEW_NATURAL_JUDGMENT_PATTERN.match(line):
            continue
        if _is_bare_review_judgment(line):
            continue
        reason_lines.append(line.strip())
    reason = " ".join(part for part in reason_lines if part).strip()
    return reason or None

def _clean_review_feedback_text(reason):
    cleaned = str(reason or "").replace("\r\n", "\n").strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"^(?:reason|feedback|review)\s*[:：-]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        rf"^(?:judg(?:e)?ment\s*[:：-]\s*)?(?:{REVIEW_JUDGMENT_TOKEN_PATTERN})\s*[:：,;\-]*\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    if not cleaned:
        return None
    segments = [
        segment.strip(" \t\r\n-:;,.")
        for segment in re.split(r"(?<=[.!?])\s+|\s*;\s*", cleaned)
        if segment and segment.strip()
    ]
    for segment in segments:
        if _extract_reason_pred_from_text(segment) is not None:
            return segment
        if not review_reason_is_shell(segment):
            return segment
    if not review_reason_is_shell(cleaned):
        return cleaned.strip(" \t\r\n-:;,.")
    return None

def _build_canonical_review_feedback(
    judgment,
    cleaned_reason,
    pending_pred,
    target_pred,
    prefer_canonical=False,
):
    if (
        judgment == "wrong"
        and pending_pred is not None
        and target_pred is not None
        and not same_numeric_prediction(target_pred, pending_pred)
    ):
        if cleaned_reason and not prefer_canonical:
            return cleaned_reason
        target_text = _normalize_numeric_text(target_pred)
        pending_text = _normalize_numeric_text(pending_pred)
        if target_text and pending_text:
            return f"The final answer should be {target_text}, not {pending_text}."
    if cleaned_reason and not prefer_canonical:
        return cleaned_reason
    if judgment == "right":
        return "The pending final answer matches the problem."
    if judgment == "wrong":
        return "The pending final answer does not match the problem."
    return cleaned_reason

def _parse_review_feedback_info(text, pending_candidate_text=None):
    pending_pred = parse_proposal_candidate_pred(pending_candidate_text)
    lines = _split_nonempty_lines(text)
    structured_judgment = _extract_review_judgment_from_lines(lines) if lines else None
    raw_reason = _extract_raw_review_reason(text)
    target_pred = _extract_review_target_pred(text)
    review_text = " ".join(lines)
    has_explicit_accept = bool(REVIEW_EXPLICIT_ACCEPT_PATTERN.search(review_text))
    has_explicit_reject = bool(REVIEW_EXPLICIT_REJECT_PATTERN.search(review_text))
    target_signal = None
    if target_pred is not None and pending_pred is not None:
        target_signal = (
            "right"
            if same_numeric_prediction(target_pred, pending_pred)
            else "wrong"
        )
    explicit_signal = None
    if has_explicit_accept and not has_explicit_reject:
        explicit_signal = "right"
    elif has_explicit_reject and not has_explicit_accept:
        explicit_signal = "wrong"

    raw_judgment = None
    # Prefer the explicit first-line judgment. Numbers mentioned in the reason
    # are only an auxiliary signal and should not override the declared stance.
    if structured_judgment in {"right", "wrong"}:
        raw_judgment = structured_judgment
    elif explicit_signal is not None:
        raw_judgment = explicit_signal
    elif target_signal is not None:
        raw_judgment = target_signal

    contradiction = False
    signals = [
        signal
        for signal in (structured_judgment, explicit_signal, target_signal)
        if signal in {"right", "wrong"}
    ]
    if has_explicit_accept and has_explicit_reject:
        contradiction = True
    if raw_judgment is not None and any(signal != raw_judgment for signal in signals):
        contradiction = True

    cleaned_reason = _clean_review_feedback_text(raw_reason)
    feedback_text = _build_canonical_review_feedback(
        raw_judgment,
        cleaned_reason,
        pending_pred,
        target_pred,
        prefer_canonical=contradiction and target_signal is not None,
    )
    raw_verdict = review_judgment_to_action(raw_judgment)
    if contradiction:
        normalized_verdict = None
        effectively_valid = False
    else:
        normalized_verdict = (
            normalize_review_verdict(raw_verdict, pending_candidate_text)
            if pending_pred is not None and raw_verdict in {"accept", "reject"}
            else None
        )
        effectively_valid = (
            pending_pred is not None
            and raw_judgment in {"right", "wrong"}
            and normalized_verdict in {"accept", "reject"}
        )
    return {
        "raw_judgment": raw_judgment,
        "normalized_judgment": raw_judgment if effectively_valid else None,
        "raw_verdict": raw_verdict,
        "normalized_verdict": normalized_verdict,
        "reason": feedback_text,
        "reason_pred": target_pred,
        "pending_pred": pending_pred,
        "raw_mismatch": contradiction,
        "accept_reason_mismatch": contradiction,
        "effectively_valid": effectively_valid,
        "feedback_label": raw_judgment.upper() if raw_judgment in {"right", "wrong"} else "UNKNOWN",
        "feedback_text": feedback_text,
        "structured_judgment": structured_judgment,
        "raw_reason": raw_reason,
    }

def parse_review_answer_pred(text):
    return _parse_review_feedback_info(text).get("reason_pred")

def infer_review_judgment(text, pending_candidate_text=None):
    return _parse_review_feedback_info(
        text,
        pending_candidate_text=pending_candidate_text,
    ).get("raw_judgment")

def parse_review_judgment(text, pending_candidate_text=None):
    info = _parse_review_feedback_info(
        text,
        pending_candidate_text=pending_candidate_text,
    )
    if pending_candidate_text is not None:
        return info["normalized_judgment"]
    if info["raw_mismatch"]:
        return None
    return info["raw_judgment"]

def review_judgment_to_action(judgment):
    if judgment == "right":
        return "accept"
    if judgment == "wrong":
        return "reject"
    return None

def parse_review_verdict(text, pending_candidate_text=None):
    info = _parse_review_feedback_info(
        text,
        pending_candidate_text=pending_candidate_text,
    )
    if pending_candidate_text is not None:
        return info["normalized_verdict"]
    if info["raw_mismatch"]:
        return None
    return info["raw_verdict"]

def is_valid_review_verdict(verdict):
    return verdict in {"accept", "reject"}

def is_valid_review_text(text):
    return parse_review_judgment(text) in {"right", "wrong"}


def get_effective_review_info(text, pending_candidate_text):
    return _parse_review_feedback_info(
        text,
        pending_candidate_text=pending_candidate_text,
    )


def is_effectively_valid_review_text(text, pending_candidate_text):
    return get_effective_review_info(text, pending_candidate_text)["effectively_valid"]

def invalid_review_reward():
    return -float(REVIEW_INVALID_JUDGMENT_PENALTY)

def binary_exact_reward(pred_num, gt):
    return 1.0 if is_exact_match(pred_num, gt) else 0.0

def signed_exact_reward(pred_num, gt):
    return 1.0 if is_exact_match(pred_num, gt) else -1.0

def get_proposal_review_hold_reward(proposal_mode, incumbent_pred, pending_candidate_pred, gt):
    if proposal_mode == "keep_pending":
        return binary_exact_reward(pending_candidate_pred, gt)
    if proposal_mode == "stop":
        return binary_exact_reward(incumbent_pred, gt)
    return 0.0

def get_proposal_supervision_reward(
    proposal_mode,
    candidate_pred,
    incumbent_pred,
    pending_candidate_pred,
    gt,
):
    if proposal_mode in {"keep_pending", "stop"}:
        return get_proposal_review_hold_reward(
            proposal_mode,
            incumbent_pred,
            pending_candidate_pred,
            gt,
        )
    return binary_exact_reward(candidate_pred, gt)

def get_proposal_middle_value(
    proposal_mode,
    candidate_pred,
    incumbent_pred,
    pending_candidate_pred,
    gt,
    proposal_outcome=None,
):
    if proposal_outcome is not None:
        if proposal_outcome.get("same_answer_refresh_as_keep"):
            return signed_exact_reward(
                resolve_final_pred(incumbent_pred, pending_candidate_pred),
                gt,
            )
        return signed_exact_reward(proposal_outcome.get("current_pred"), gt)
    if proposal_mode == "keep_pending":
        return signed_exact_reward(pending_candidate_pred, gt)
    if proposal_mode == "stop":
        return signed_exact_reward(incumbent_pred, gt)
    return signed_exact_reward(candidate_pred, gt)

def get_review_supervision_reward(review_info, pending_candidate_pred, gt):
    if pending_candidate_pred is None:
        return invalid_review_reward()
    if not review_info["effectively_valid"]:
        return invalid_review_reward()
    target_verdict = "accept" if is_exact_match(pending_candidate_pred, gt) else "reject"
    return 1.0 if review_info["normalized_verdict"] == target_verdict else 0.0

def parse_review_reason(text, pending_candidate_text=None):
    info = _parse_review_feedback_info(
        text,
        pending_candidate_text=pending_candidate_text,
    )
    return info.get("reason")

def _compact_review_reason_text(reason):
    return re.sub(r"[^A-Za-z0-9]+", "", str(reason or "")).strip().lower()

def review_reason_is_shell(reason):
    compact_reason = _compact_review_reason_text(reason)
    if not compact_reason:
        return True
    if len(compact_reason) < int(REVIEW_MIN_REASON_COMPACT_CHARS):
        return True
    return compact_reason in {
        "reason",
        "candidate",
        "judgment",
        "right",
        "wrong",
        "correct",
        "incorrect",
        "calculation",
        "arithmetic",
    }

def get_review_format_reward_adjustment(text):
    return 0.0

def parse_proposal_candidate_pred(text):
    return _extract_proposal_final_pred(text)

def _parse_numeric_token(token):
    token = str(token or "").replace("$", "").replace(",", "").strip()
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None

def _extract_numeric_tokens_from_text(text):
    values = []
    for token in TEXT_NUMBER_PATTERN.findall(text or ""):
        numeric = _parse_numeric_token(token)
        if numeric is not None:
            values.append(numeric)
    return values

def _collect_proposal_candidate_lines(lines):
    candidate_lines = []
    candidate_lines.extend(lines[:2])
    if lines and lines[-1] not in candidate_lines:
        candidate_lines.append(lines[-1])
    return candidate_lines

def _extract_explicit_proposal_candidate(lines):
    for line in _collect_proposal_candidate_lines(lines):
        if PROPOSAL_EMPTY_CANDIDATE_PATTERN.match(line):
            return True, None, ""
        match = PROPOSAL_CANDIDATE_PATTERN.match(line)
        if match:
            return True, _parse_numeric_token(match.group(1)), match.group(1)
        match = PROPOSAL_ANSWER_LINE_PATTERN.match(line)
        if match:
            return True, _parse_numeric_token(match.group(1)), match.group(1)
    return False, None, None

def _normalize_numeric_text(value):
    if value is None:
        return None
    numeric = float(value)
    if abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return f"{numeric:g}"

def _is_numeric_prefix_truncation(candidate_pred, reason_pred):
    candidate_text = _normalize_numeric_text(candidate_pred)
    reason_text = _normalize_numeric_text(reason_pred)
    if not candidate_text or not reason_text or candidate_text == reason_text:
        return False
    return len(candidate_text) < len(reason_text) and reason_text.startswith(candidate_text)

def _extract_reason_texts(lines):
    reason_texts = []
    for line in lines:
        match = REVIEW_REASON_PATTERN.match(line)
        if not match:
            continue
        reason_text = match.group(1).strip()
        if reason_text:
            reason_texts.append(reason_text)
    return reason_texts

def _extract_reason_tail_segment(reason_text):
    if not reason_text:
        return ""
    segments = [
        segment.strip()
        for segment in reason_text.split(";")
        if segment and segment.strip()
    ]
    if segments:
        return segments[-1]
    return reason_text.strip()

def _segment_contains_numeric_value(text, target_value):
    if target_value is None or not text:
        return False
    for numeric in _extract_numeric_tokens_from_text(text):
        if same_numeric_prediction(numeric, target_value):
            return True
    return False

def _extract_reason_pred_from_text(reason_text):
    if not reason_text:
        return None
    eq_matches = PROPOSAL_REASON_FINAL_EQUALS_PATTERN.findall(reason_text)
    if eq_matches:
        pred = _parse_numeric_token(eq_matches[-1])
        if pred is not None:
            return pred
    trailing_match = PROPOSAL_REASON_TRAILING_NUMBER_PATTERN.search(reason_text)
    if trailing_match:
        pred = _parse_numeric_token(trailing_match.group(1))
        if pred is not None:
            return pred
    numeric_tokens = _extract_numeric_tokens_from_text(reason_text)
    if numeric_tokens:
        return numeric_tokens[-1]
    return None

def _extract_reason_final_pred(lines, candidate_pred=None):
    for reason_text in _extract_reason_texts(lines):
        pred = _extract_reason_pred_from_text(reason_text)
        if pred is not None:
            return pred
    return None

def _split_nonempty_lines(text):
    return [line.strip() for line in str(text or "").replace("\r\n", "\n").split("\n") if line.strip()]

def _extract_last_numeric_after_equals(text):
    matches = PROPOSAL_REASON_FINAL_EQUALS_PATTERN.findall(text or "")
    if not matches:
        return None
    return _parse_numeric_token(matches[-1])

def _extract_first_numeric_in_text(text):
    matches = TEXT_NUMBER_PATTERN.findall(text or "")
    if not matches:
        return None
    return _parse_numeric_token(matches[0])

def _extract_leading_standalone_answer_pred(answer_tail):
    match = PROPOSAL_LEADING_STANDALONE_NUMBER_PATTERN.match(answer_tail or "")
    if not match:
        return None
    pred = _parse_numeric_token(match.group(1))
    if pred is None:
        return None
    rest = str(answer_tail or "")[match.end():].strip()
    if not rest:
        return pred
    if rest.startswith("("):
        return pred
    if _extract_numeric_tokens_from_text(rest):
        return None
    if re.search(r"[+\-*/=]", rest):
        return None
    return pred

def _extract_proposal_answer_cue_pred(text):
    lines = _split_nonempty_lines(text)
    if not lines:
        return None
    candidate_lines = []
    candidate_lines.extend(lines[-3:])
    if len(lines) >= 1:
        candidate_lines.append(lines[-1])
    seen = set()
    ordered_lines = []
    for line in candidate_lines:
        if line not in seen:
            seen.add(line)
            ordered_lines.append(line)

    for line in reversed(ordered_lines):
        lowered = line.lower()
        if "\\boxed" in lowered:
            pred = extract_pred_num(line)
            if pred is not None:
                return pred
        match = PROPOSAL_CANDIDATE_PATTERN.match(line)
        if match:
            pred = _parse_numeric_token(match.group(1))
            if pred is not None:
                return pred
        match = PROPOSAL_ANSWER_LINE_PATTERN.match(line)
        if match:
            pred = _parse_numeric_token(match.group(1))
            if pred is not None:
                return pred
        tail_match = PROPOSAL_ANSWER_PREFIX_PATTERN.match(line)
        if tail_match:
            answer_tail = tail_match.group("tail")
        else:
            hash_match = PROPOSAL_HASH_ANSWER_PATTERN.match(line)
            answer_tail = hash_match.group("tail") if hash_match else None
        if answer_tail is None:
            continue
        pred = _extract_last_numeric_after_equals(answer_tail)
        if pred is not None:
            return pred
        pred = _extract_leading_standalone_answer_pred(answer_tail)
        if pred is not None:
            return pred
    return None

def parse_proposal_reason_pred(text):
    return _extract_strict_proposal_final_pred(text)

def _extract_strict_proposal_final_pred(text):
    last_line = _last_nonempty_line(text)
    if not last_line:
        return None
    match = PROPOSAL_CANDIDATE_PATTERN.match(last_line)
    if match:
        return _parse_numeric_token(match.group(1))
    match = PROPOSAL_ANSWER_LINE_PATTERN.match(last_line)
    if match:
        return _parse_numeric_token(match.group(1))
    if "\\boxed" in last_line.lower():
        return extract_pred_num(last_line)
    return None

def _extract_proposal_final_pred(text):
    strict_pred = _extract_strict_proposal_final_pred(text)
    if strict_pred is not None:
        return strict_pred
    return _extract_proposal_answer_cue_pred(text)

def get_proposal_format_info(text):
    resolved_pred = _extract_proposal_final_pred(text)
    info = {
        "candidate_present": resolved_pred is not None,
        "candidate_raw": None,
        "candidate_pred": resolved_pred,
        "reason_pred": resolved_pred,
        "status": "parseable" if resolved_pred is not None else "missing_final_answer",
        "resolved_pred": resolved_pred,
    }
    return info

def get_proposal_format_reward_adjustment(text):
    return 0.0


def proposal_is_state_eligible(text):
    return parse_proposal_candidate_pred(text) is not None


def proposal_has_self_inconsistency(text):
    return not proposal_is_state_eligible(text)

def normalize_review_verdict(verdict, pending_candidate_text):
    if verdict == "accept" and parse_proposal_candidate_pred(pending_candidate_text) is None:
        return "reject"
    return verdict

def get_latest_pending_review_feedback(history, pending_candidate_text):
    if not history:
        return None
    last_proposal_idx = None
    for idx in range(len(history) - 1, -1, -1):
        turn = history[idx]
        if turn.get("kind") == "proposal":
            last_proposal_idx = idx
            break
    if last_proposal_idx is None:
        return None
    for idx in range(len(history) - 1, last_proposal_idx, -1):
        turn = history[idx]
        if turn.get("kind") != "review":
            continue
        review_info = get_effective_review_info(
            turn.get("text"),
            pending_candidate_text,
        )
        if not review_info.get("effectively_valid") or review_info.get("raw_mismatch"):
            continue
        label = review_info.get("normalized_judgment")
        feedback_text = review_info.get("feedback_text") or review_info.get("reason")
        if label in {"right", "wrong"} and feedback_text:
            return f"{label.upper()}\n{feedback_text}"
        if label in {"right", "wrong"}:
            return label.upper()
    return None

def resolve_final_text(incumbent_text, pending_candidate_text):
    if pending_candidate_text is not None and str(pending_candidate_text).strip():
        return pending_candidate_text
    if incumbent_text is not None and str(incumbent_text).strip():
        return incumbent_text
    return None

def resolve_final_pred(incumbent_pred, pending_candidate_pred):
    if pending_candidate_pred is not None:
        return pending_candidate_pred
    if incumbent_pred is not None:
        return incumbent_pred
    return None

def resolve_final_reward(incumbent_pred, pending_candidate_pred, gt):
    return reward_from_pred(
        resolve_final_pred(incumbent_pred, pending_candidate_pred),
        gt,
    )

def compute_terminal_delta_reward(
    pre_incumbent_pred,
    pre_pending_candidate_pred,
    post_incumbent_pred,
    post_pending_candidate_pred,
    gt,
    penalty=0.0,
):
    return (
        resolve_final_reward(post_incumbent_pred, post_pending_candidate_pred, gt)
        - resolve_final_reward(pre_incumbent_pred, pre_pending_candidate_pred, gt)
        + float(penalty)
    )

def apply_review_verdict_to_state(
    verdict,
    pre_incumbent_text,
    pre_incumbent_pred,
    pre_pending_candidate_text,
    pre_pending_candidate_pred,
    pre_pending_vote_score=0,
):
    verdict = normalize_review_verdict(verdict, pre_pending_candidate_text)
    if (
        not has_pending_candidate(pre_pending_candidate_text)
        or pre_pending_candidate_pred is None
    ):
        return {
            "incumbent_text": pre_incumbent_text,
            "incumbent_pred": pre_incumbent_pred,
            "pending_candidate_text": pre_pending_candidate_text,
            "pending_candidate_pred": pre_pending_candidate_pred,
            "pending_vote_score": 0,
        }
    if PROPOSAL_REVIEW_DISABLE_VOTE_UPDATES:
        return {
            "incumbent_text": pre_incumbent_text,
            "incumbent_pred": pre_incumbent_pred,
            "pending_candidate_text": pre_pending_candidate_text,
            "pending_candidate_pred": pre_pending_candidate_pred,
            "pending_vote_score": int(pre_pending_vote_score or 0),
        }
    pending_vote_score = int(pre_pending_vote_score or 0)
    if verdict == "accept":
        pending_vote_score += 1
    elif verdict == "reject":
        pending_vote_score -= 1
    if PROPOSAL_REVIEW_APPLY_VOTE_THRESHOLDS:
        accept_threshold = get_proposal_review_accept_threshold(pre_incumbent_pred)
        drop_threshold = get_proposal_review_drop_threshold(pre_incumbent_pred)
        if pending_vote_score >= accept_threshold:
            return {
                "incumbent_text": pre_pending_candidate_text,
                "incumbent_pred": pre_pending_candidate_pred,
                "pending_candidate_text": None,
                "pending_candidate_pred": None,
                "pending_vote_score": 0,
            }
        if pending_vote_score <= drop_threshold:
            return {
                "incumbent_text": pre_incumbent_text,
                "incumbent_pred": pre_incumbent_pred,
                "pending_candidate_text": None,
                "pending_candidate_pred": None,
                "pending_vote_score": 0,
            }
    return {
        "incumbent_text": pre_incumbent_text,
        "incumbent_pred": pre_incumbent_pred,
        "pending_candidate_text": pre_pending_candidate_text,
        "pending_candidate_pred": pre_pending_candidate_pred,
        "pending_vote_score": pending_vote_score,
    }

def get_proposal_review_proposal_mode(
    rnd,
    incumbent_pred,
    pending_candidate_text,
    act,
):
    del incumbent_pred
    if get_proposal_review_stage(rnd) != "proposal":
        return "review"
    if has_pending_candidate(pending_candidate_text):
        if PROPOSAL_REVIEW_DISABLE_REFRESH:
            return "keep_pending"
        return "keep_pending" if act == MIDDLE_ACTION_COMMENT else "refresh_pending"
    return "propose_initial"

def get_proposal_review_proposer_prompt(proposal_mode):
    if PROPOSAL_REVIEW_USE_LEGACY_PROMPTS:
        return PI1_PROMPT
    if proposal_mode == "refresh_pending":
        return PI1_PROMPT_PROPOSAL_REVIEW_REFRESH
    return PI1_PROMPT_PROPOSAL_REVIEW


def proposal_refresh_keeps_pending(
    proposal_mode,
    answer_pred,
    pre_pending_candidate_text,
    pre_pending_candidate_pred,
):
    if proposal_mode != "refresh_pending":
        return False
    if not has_pending_candidate(pre_pending_candidate_text, pre_pending_candidate_pred):
        return False
    if answer_pred is None or pre_pending_candidate_pred is None:
        return False
    return same_numeric_prediction(answer_pred, pre_pending_candidate_pred)


def get_same_answer_refresh_reward_adjustment(
    proposal_mode,
    answer_pred,
    pre_pending_candidate_text,
    pre_pending_candidate_pred,
):
    return 0.0

def rollout_proposal_review_terminal_value(
    role_bundle,
    question,
    incumbent_text,
    incumbent_pred,
    pending_candidate_text,
    pending_candidate_pred,
    pending_vote_score,
    gt,
    history,
    start_round,
    controller_selector=None,
    use_average_strategy=False,
    total_rounds=NUM_ROUNDS,
):
    sim_history = list(history) if history else []
    sim_incumbent_text = incumbent_text
    sim_incumbent_pred = incumbent_pred
    sim_pending_candidate_text = pending_candidate_text
    sim_pending_candidate_pred = pending_candidate_pred
    sim_pending_vote_score = int(pending_vote_score or 0)
    sim_reward_adjustment = 0.0

    for sim_rnd in range(int(start_round), int(total_rounds) + 1):
        stage = get_proposal_review_stage(sim_rnd)
        if stage == "review":
            if (
                not has_pending_candidate(sim_pending_candidate_text)
                or sim_pending_candidate_pred is None
            ):
                continue
            review_text = role_bundle["pi0"].sample_text(
                build_sampling_context(
                    question,
                    sim_history,
                    incumbent_text=sim_incumbent_text,
                    pending_candidate_text=sim_pending_candidate_text,
                    pending_vote_score=sim_pending_vote_score,
                    include_latest_review_reason=False,
                ),
                prompt_override=PI0_PROMPT,
            )
            review_outcome = resolve_review_action_outcome(
                sim_history,
                sim_rnd,
                "reviewer_rollout",
                review_text,
                parse_review_verdict(review_text),
                sim_incumbent_text,
                sim_incumbent_pred,
                sim_pending_candidate_text,
                sim_pending_candidate_pred,
                sim_pending_vote_score,
            )
            sim_history = review_outcome["history"]
            sim_incumbent_text = review_outcome["incumbent_text"]
            sim_incumbent_pred = review_outcome["incumbent_pred"]
            sim_pending_candidate_text = review_outcome["pending_candidate_text"]
            sim_pending_candidate_pred = review_outcome["pending_candidate_pred"]
            sim_pending_vote_score = review_outcome.get("pending_vote_score", 0)
            continue

        allowed_actions = get_middle_allowed_actions(
            sim_rnd,
            sim_incumbent_pred,
            allow_silent=False,
            total_rounds=total_rounds,
            phase=None,
            pending_candidate_text=sim_pending_candidate_text,
            pending_vote_score=sim_pending_vote_score,
        )
        if len(allowed_actions) == 1:
            act = int(allowed_actions[0])
        else:
            if controller_selector is not None:
                _, _, strategy = get_constrained_middle_strategy(
                    controller_selector,
                    None,
                    sim_rnd,
                    0,
                    sim_incumbent_pred,
                    allow_silent=False,
                    use_average_strategy=use_average_strategy,
                    total_rounds=total_rounds,
                    pending_candidate_text=sim_pending_candidate_text,
                    pending_vote_score=sim_pending_vote_score,
                )
            else:
                state = build_middle_state(
                    0,
                    None,
                    incumbent_pred=sim_incumbent_pred,
                    rnd=sim_rnd,
                    total_rounds=total_rounds,
                    pending_candidate_text=sim_pending_candidate_text,
                    pending_vote_score=sim_pending_vote_score,
                )
                strategy = normalize_middle_strategy(
                    build_middle_fallback_strategy(
                        state,
                        allowed_actions,
                        CFR_NUM_ACTIONS,
                        MIDDLE_ACTION_ANSWER,
                    ),
                    allowed_actions,
                )
            if THREE_LAYER_EVAL_DETERMINISTIC_ACTIONS:
                act = int(np.argmax(strategy))
            else:
                act = int(np.random.choice(CFR_NUM_ACTIONS, p=strategy))

        proposal_mode = get_proposal_review_proposal_mode(
            sim_rnd,
            sim_incumbent_pred,
            sim_pending_candidate_text,
            act,
        )
        if proposal_mode in {"keep_pending", "stop"}:
            continue

        proposal_text = role_bundle["pi1"].sample_text(
            build_sampling_context(
                question,
                sim_history,
                incumbent_text=sim_incumbent_text,
                pending_candidate_text=sim_pending_candidate_text,
                pending_vote_score=sim_pending_vote_score,
            ),
            prompt_override=get_proposal_review_proposer_prompt(proposal_mode),
        )
        proposal_outcome = resolve_proposal_action_outcome(
            sim_history,
            sim_rnd,
            "proposer_rollout",
            proposal_mode,
            proposal_text,
            parse_proposal_candidate_pred(proposal_text),
            sim_incumbent_text,
            sim_incumbent_pred,
            sim_pending_candidate_text,
            sim_pending_candidate_pred,
            sim_pending_vote_score,
        )
        sim_history = proposal_outcome["history"]
        sim_incumbent_text = proposal_outcome["incumbent_text"]
        sim_incumbent_pred = proposal_outcome["incumbent_pred"]
        sim_pending_candidate_text = proposal_outcome["pending_candidate_text"]
        sim_pending_candidate_pred = proposal_outcome["pending_candidate_pred"]
        sim_pending_vote_score = proposal_outcome.get("pending_vote_score", 0)
        sim_reward_adjustment += float(proposal_outcome.get("reward_adjustment", 0.0))

    return resolve_final_reward(
        sim_incumbent_pred,
        sim_pending_candidate_pred,
        gt,
    ) + sim_reward_adjustment

def compute_proposal_candidate_rewards(
    batch,
    role_bundle,
    controller_selector,
    use_average_strategy,
    question,
    pre_history,
    rnd,
    incumbent_text,
    incumbent_pred,
    pending_candidate_text,
    pending_candidate_pred,
    pending_vote_score,
    gt,
    total_rounds=NUM_ROUNDS,
):
    proposal_mode = get_proposal_review_proposal_mode(
        rnd,
        incumbent_pred,
        pending_candidate_text,
        MIDDLE_ACTION_ANSWER,
    )
    rewards = []
    for cand in batch["candidates"]:
        candidate_text = cand["text"]
        candidate_pred = parse_proposal_candidate_pred(candidate_text)
        proposal_outcome = resolve_proposal_action_outcome(
            pre_history,
            rnd,
            "pi1_cf_batch",
            proposal_mode,
            candidate_text,
            candidate_pred,
            incumbent_text,
            incumbent_pred,
            pending_candidate_text,
            pending_candidate_pred,
            pending_vote_score,
        )
        rewards.append(
            get_proposal_supervision_reward(
                proposal_mode,
                candidate_pred,
                incumbent_pred,
                pending_candidate_pred,
                gt,
            )
            + float(proposal_outcome.get("reward_adjustment", 0.0))
        )
    return rewards

def compute_proposal_controller_candidate_values(
    batch,
    pre_history,
    rnd,
    incumbent_text,
    incumbent_pred,
    pending_candidate_text,
    pending_candidate_pred,
    pending_vote_score,
    gt,
):
    proposal_mode = get_proposal_review_proposal_mode(
        rnd,
        incumbent_pred,
        pending_candidate_text,
        MIDDLE_ACTION_ANSWER,
    )
    values = []
    outcomes = []
    for cand in batch["candidates"]:
        candidate_text = cand["text"]
        candidate_pred = parse_proposal_candidate_pred(candidate_text)
        proposal_outcome = resolve_proposal_action_outcome(
            pre_history,
            rnd,
            "pi1_controller_batch",
            proposal_mode,
            candidate_text,
            candidate_pred,
            incumbent_text,
            incumbent_pred,
            pending_candidate_text,
            pending_candidate_pred,
            pending_vote_score,
        )
        values.append(
            get_proposal_middle_value(
                proposal_mode,
                candidate_pred,
                incumbent_pred,
                pending_candidate_pred,
                gt,
                proposal_outcome=proposal_outcome,
            )
        )
        outcomes.append(proposal_outcome)
    return values, outcomes

def compute_proposal_candidate_reward(
    role_bundle,
    controller_selector,
    use_average_strategy,
    question,
    pre_history,
    rnd,
    candidate_text,
    candidate_pred,
    incumbent_text,
    incumbent_pred,
    pending_candidate_text,
    pending_candidate_pred,
    pending_vote_score,
    gt,
    total_rounds=NUM_ROUNDS,
):
    batch = {
        "candidates": [{
            "text": candidate_text,
            "pred": candidate_pred,
        }],
    }
    rewards = compute_proposal_candidate_rewards(
        batch,
        role_bundle,
        controller_selector,
        use_average_strategy,
        question,
        pre_history,
        rnd,
        incumbent_text,
        incumbent_pred,
        pending_candidate_text,
        pending_candidate_pred,
        pending_vote_score,
        gt,
        total_rounds=total_rounds,
    )
    return rewards[0] if rewards else 0.0

def compute_review_reward(
    role_bundle,
    controller_selector,
    use_average_strategy,
    question,
    pre_history,
    rnd,
    verdict,
    incumbent_text,
    incumbent_pred,
    pending_candidate_text,
    pending_candidate_pred,
    pending_vote_score,
    gt,
    total_rounds=NUM_ROUNDS,
    review_text=None,
):
    review_info = get_effective_review_info(review_text, pending_candidate_text)
    return get_review_supervision_reward(
        review_info,
        pending_candidate_pred,
        gt,
    )

def compute_review_candidate_rewards(
    batch,
    role_bundle,
    controller_selector,
    use_average_strategy,
    question,
    pre_history,
    rnd,
    incumbent_text,
    incumbent_pred,
    pending_candidate_text,
    pending_candidate_pred,
    pending_vote_score,
    gt,
    total_rounds=NUM_ROUNDS,
):
    verdict_reward_cache = {}
    rewards = []
    for cand in batch["candidates"]:
        review_info = get_effective_review_info(cand["text"], pending_candidate_text)
        cache_key = (
            review_info["normalized_verdict"],
            bool(review_info["effectively_valid"]),
        )
        if cache_key not in verdict_reward_cache:
            verdict_reward_cache[cache_key] = get_review_supervision_reward(
                review_info,
                pending_candidate_pred,
                gt,
            )
        rewards.append(verdict_reward_cache[cache_key])
    return rewards

def build_sampling_context(
    question,
    history,
    incumbent_text=None,
    pending_candidate_text=None,
    pending_vote_score=0,
    include_latest_review_reason=True,
):
    if is_proposal_review_schema():
        return build_proposal_review_context(
            question,
            history,
            incumbent_text,
            pending_candidate_text,
            pending_vote_score=pending_vote_score,
            include_latest_review_reason=include_latest_review_reason,
        )
    return build_context(question, history)

def resolve_proposal_action_outcome(
    pre_history,
    rnd,
    speaker,
    proposal_mode,
    answer_text,
    answer_pred,
    pre_incumbent_text,
    pre_incumbent_pred,
    pre_pending_candidate_text,
    pre_pending_candidate_pred,
    pre_pending_vote_score=0,
):
    history = list(pre_history) if pre_history else []
    effective_answer_pred = (
        answer_pred
        if answer_pred is not None else parse_proposal_candidate_pred(answer_text)
    )
    same_answer_refresh_as_keep = proposal_refresh_keeps_pending(
        proposal_mode,
        effective_answer_pred,
        pre_pending_candidate_text,
        pre_pending_candidate_pred,
    )
    reward_adjustment = get_same_answer_refresh_reward_adjustment(
        proposal_mode,
        effective_answer_pred,
        pre_pending_candidate_text,
        pre_pending_candidate_pred,
    )
    if (
        proposal_mode == "refresh_pending"
        and PROPOSAL_REVIEW_REFRESH_REQUIRES_NEGATIVE_PENDING
        and int(pre_pending_vote_score or 0) >= 0
    ):
        return {
            "history": history,
            "incumbent_text": pre_incumbent_text,
            "incumbent_pred": pre_incumbent_pred,
            "pending_candidate_text": pre_pending_candidate_text,
            "pending_candidate_pred": pre_pending_candidate_pred,
            "pending_vote_score": int(pre_pending_vote_score or 0),
            "current_pred": resolve_final_pred(
                pre_incumbent_pred,
                pre_pending_candidate_pred,
            ),
            "same_answer_refresh_as_keep": True,
            "refresh_blocked_by_vote": True,
            "reward_adjustment": reward_adjustment,
        }
    if same_answer_refresh_as_keep:
        return {
            "history": history,
            "incumbent_text": pre_incumbent_text,
            "incumbent_pred": pre_incumbent_pred,
            "pending_candidate_text": pre_pending_candidate_text,
            "pending_candidate_pred": pre_pending_candidate_pred,
            "pending_vote_score": int(pre_pending_vote_score or 0),
            "current_pred": resolve_final_pred(
                pre_incumbent_pred,
                pre_pending_candidate_pred,
            ),
            "same_answer_refresh_as_keep": True,
            "reward_adjustment": reward_adjustment,
        }
    if (
        answer_text is not None
        and str(answer_text).strip()
        and not same_answer_refresh_as_keep
    ):
        history = append_round_output(
            pre_history,
            rnd,
            speaker,
            answer_text,
            "proposal",
        )
    if proposal_mode in {"keep_pending", "stop"}:
        return {
            "history": history,
            "incumbent_text": pre_incumbent_text,
            "incumbent_pred": pre_incumbent_pred,
            "pending_candidate_text": pre_pending_candidate_text,
            "pending_candidate_pred": pre_pending_candidate_pred,
            "pending_vote_score": int(pre_pending_vote_score or 0),
            "current_pred": resolve_final_pred(
                pre_incumbent_pred,
                pre_pending_candidate_pred,
            ),
            "same_answer_refresh_as_keep": same_answer_refresh_as_keep,
            "reward_adjustment": reward_adjustment,
        }
    if effective_answer_pred is None:
        if proposal_mode == "refresh_pending":
            return {
                "history": history,
                "incumbent_text": pre_incumbent_text,
                "incumbent_pred": pre_incumbent_pred,
                "pending_candidate_text": pre_pending_candidate_text,
                "pending_candidate_pred": pre_pending_candidate_pred,
                "pending_vote_score": int(pre_pending_vote_score or 0),
                "current_pred": resolve_final_pred(
                    pre_incumbent_pred,
                    pre_pending_candidate_pred,
                ),
                "same_answer_refresh_as_keep": False,
                "reward_adjustment": 0.0,
            }
        return {
            "history": history,
            "incumbent_text": pre_incumbent_text,
            "incumbent_pred": pre_incumbent_pred,
            "pending_candidate_text": None,
            "pending_candidate_pred": None,
            "pending_vote_score": 0,
            "current_pred": resolve_final_pred(
                pre_incumbent_pred,
                None,
            ),
            "same_answer_refresh_as_keep": False,
            "reward_adjustment": 0.0,
        }
    return {
        "history": history,
        "incumbent_text": pre_incumbent_text,
        "incumbent_pred": pre_incumbent_pred,
        "pending_candidate_text": answer_text,
        "pending_candidate_pred": effective_answer_pred,
        "pending_vote_score": 0,
        "current_pred": resolve_final_pred(
            pre_incumbent_pred,
            effective_answer_pred,
        ),
        "same_answer_refresh_as_keep": False,
        "reward_adjustment": 0.0,
    }

def resolve_review_action_outcome(
    pre_history,
    rnd,
    speaker,
    review_text,
    verdict,
    pre_incumbent_text,
    pre_incumbent_pred,
    pre_pending_candidate_text,
    pre_pending_candidate_pred,
    pre_pending_vote_score=0,
):
    history = append_round_output(
        pre_history,
        rnd,
        speaker,
        review_text,
        "review",
    )
    review_info = get_effective_review_info(review_text, pre_pending_candidate_text)
    effective_verdict = (
        review_info["normalized_verdict"]
        if review_info["effectively_valid"] else None
    )
    post_state = apply_review_verdict_to_state(
        effective_verdict,
        pre_incumbent_text,
        pre_incumbent_pred,
        pre_pending_candidate_text,
        pre_pending_candidate_pred,
        pre_pending_vote_score,
    )
    return {
        "history": history,
        "incumbent_text": post_state["incumbent_text"],
        "incumbent_pred": post_state["incumbent_pred"],
        "pending_candidate_text": post_state["pending_candidate_text"],
        "pending_candidate_pred": post_state["pending_candidate_pred"],
        "pending_vote_score": post_state["pending_vote_score"],
        "current_pred": resolve_final_pred(
            post_state["incumbent_pred"],
            post_state["pending_candidate_pred"],
        ),
    }

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
    - 如果这一轮是 answer 轮，就直接把“当前这次 answer 的解析结果”记为最新答案
    - 如果这一轮是 comment 轮，就保留上一轮答案

    所以：
    - 第 1/3/5 轮会刷新“当前答案”，即使这次 answer 不可解析也会刷新成 None
    - 第 2/4 轮通常沿用前一轮答案
    """
    if should_use_current_answer:
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

def append_live_accuracy_snapshot(exp_name, processed, total, accs, best_accs=None):
    record = {
        "timestamp": time.time(),
        "exp_name": exp_name,
        "processed": int(processed),
        "total": int(total),
        "round_accuracy": [float(value) for value in accs],
        "best_so_far_accuracy": (
            [float(value) for value in best_accs]
            if best_accs is not None else None
        ),
    }
    live_log_path = os.path.join(LOG_DIR, "live_accuracy.jsonl")
    with open(live_log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()

def append_policy_stack_candidate_snapshot(record):
    snapshot_path = os.path.join(LOG_DIR, "policy_stack_candidate_dump.jsonl")
    with open(snapshot_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()

def maybe_dump_policy_stack_candidates(
    exp_name,
    sample_idx,
    branch_idx,
    question,
    gt,
    rnd,
    act,
    pre_incumbent_pred,
    pre_pending_candidate_text,
    pre_pending_candidate_pred,
    pre_pending_vote_score,
    round_result,
):
    step = round_result.get("step")
    batch = step.get("batch") if isinstance(step, dict) else None
    if not batch:
        return
    realized_step = round_result.get("realized_step")

    stage = get_proposal_review_stage(rnd) if is_proposal_review_schema() else None
    rewards = None
    batch_middle_values = round_result.get("batch_middle_values")
    if batch_middle_values is None:
        batch_middle_values = round_result.get("candidate_middle_values")
    gspo_update_payload = round_result.get("gspo_update_payload")
    if gspo_update_payload is not None and len(gspo_update_payload) >= 3:
        rewards = gspo_update_payload[2]

    candidate_records = []
    for cand_idx, cand in enumerate(batch.get("candidates", [])):
        text = cand.get("text")
        candidate_record = {
            "idx": int(cand_idx),
            "text": text,
        }
        generation_mode = cand.get("generation_mode")
        if generation_mode is not None:
            candidate_record["generation_mode"] = generation_mode
        if stage == "proposal":
            proposal_info = get_proposal_format_info(text)
            candidate_record["parsed_pred"] = proposal_info["resolved_pred"]
            candidate_record["proposal_status"] = proposal_info["status"]
            candidate_record["state_eligible"] = proposal_is_state_eligible(text)
        elif stage == "review":
            review_info = get_effective_review_info(text, pre_pending_candidate_text)
            candidate_record["parsed_judgment"] = review_info["raw_judgment"]
            candidate_record["normalized_judgment"] = review_info["normalized_judgment"]
            candidate_record["judgment_action"] = review_info["normalized_verdict"]
            candidate_record["parsed_verdict"] = review_info["raw_verdict"]
            candidate_record["normalized_verdict"] = review_info["normalized_verdict"]
            candidate_record["effectively_valid"] = review_info["effectively_valid"]
            candidate_record["parsed_reason"] = review_info["reason"]
            candidate_record["reason_pred"] = review_info["reason_pred"]
            candidate_record["pending_pred"] = review_info["pending_pred"]
            candidate_record["accept_reason_mismatch"] = review_info["accept_reason_mismatch"]
        if rewards is not None and cand_idx < len(rewards):
            candidate_record["reward"] = float(rewards[cand_idx])
        if batch_middle_values is not None and cand_idx < len(batch_middle_values):
            candidate_record["middle_value"] = float(batch_middle_values[cand_idx])
            candidate_record["middle_value_source"] = "gspo_batch_diagnostic_only"
        candidate_records.append(candidate_record)

    selected_text = (
        realized_step.get("selected_text")
        if isinstance(realized_step, dict) and realized_step.get("selected_text") is not None
        else batch.get("selected_text")
    )
    effective_act = int(round_result.get("effective_action", act))
    regret_act = int(round_result.get("regret_action", effective_act))
    record = {
        "timestamp": time.time(),
        "exp_name": exp_name,
        "sample_idx": int(sample_idx),
        "branch_idx": int(branch_idx),
        "round": int(rnd),
        "stage": stage,
        "action": get_three_layer_action_name(
            act,
            rnd=rnd,
            incumbent_pred=pre_incumbent_pred,
            pending_candidate_text=pre_pending_candidate_text,
        ),
        "question": question,
        "ground_truth": gt,
        "pre_incumbent_pred": pre_incumbent_pred,
        "pre_pending_candidate_pred": pre_pending_candidate_pred,
        "pre_pending_vote_score": int(pre_pending_vote_score or 0),
        "selected_idx": int(round_result.get("selected_outcome_idx", 0)),
        "selected_text": selected_text,
        "selected_generation_mode": (
            "single_sample"
            if isinstance(realized_step, dict) and realized_step.get("selected_text") is not None
            else (
                batch.get("candidates", [{}])[int(batch.get("selected_idx", 0))].get("generation_mode")
                if batch.get("candidates") else None
            )
        ),
        "selected_source": (
            "realized_step"
            if isinstance(realized_step, dict) and realized_step.get("selected_text") is not None
            else "batch"
        ),
        "batch_selected_idx": int(batch.get("selected_idx", 0)),
        "batch_selected_text": batch.get("selected_text"),
        "batch_selected_generation_mode": (
            batch.get("candidates", [{}])[int(batch.get("selected_idx", 0))].get("generation_mode")
            if batch.get("candidates") else None
        ),
        "selected_realized_value": float(round_result.get("realized_value", 0.0)),
        "selected_realized_value_source": (
            "single_sample_post_gspo_update"
            if round_result.get("gspo_update_applied", False)
            else "single_sample"
        ),
        "selected_outcome_idx": int(round_result.get("selected_outcome_idx", 0)),
        "candidates": candidate_records,
        "effective_action": get_three_layer_action_name(
            effective_act,
            rnd=rnd,
            incumbent_pred=pre_incumbent_pred,
            pending_candidate_text=pre_pending_candidate_text,
        ),
        "regret_action": get_three_layer_action_name(
            regret_act,
            rnd=rnd,
            incumbent_pred=pre_incumbent_pred,
            pending_candidate_text=pre_pending_candidate_text,
        ),
        "selected_same_answer_refresh_as_keep": bool(
            round_result.get("same_answer_refresh_as_keep", False)
        ),
        "gspo_update_applied": bool(round_result.get("gspo_update_applied", False)),
    }
    if stage == "proposal":
        selected_info = get_proposal_format_info(selected_text)
        record["selected_parsed_pred"] = selected_info["resolved_pred"]
        record["selected_proposal_status"] = selected_info["status"]
        record["selected_state_eligible"] = proposal_is_state_eligible(selected_text)
    elif stage == "review":
        selected_review_info = get_effective_review_info(selected_text, pre_pending_candidate_text)
        record["selected_parsed_judgment"] = selected_review_info["raw_judgment"]
        record["selected_normalized_judgment"] = selected_review_info["normalized_judgment"]
        record["selected_judgment_action"] = selected_review_info["normalized_verdict"]
        record["selected_parsed_verdict"] = selected_review_info["raw_verdict"]
        record["selected_normalized_verdict"] = selected_review_info["normalized_verdict"]
        record["selected_effectively_valid"] = selected_review_info["effectively_valid"]
        record["selected_parsed_reason"] = selected_review_info["reason"]
        record["selected_reason_pred"] = selected_review_info["reason_pred"]
        record["selected_pending_pred"] = selected_review_info["pending_pred"]
        record["selected_accept_reason_mismatch"] = selected_review_info["accept_reason_mismatch"]
    append_policy_stack_candidate_snapshot(record)

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
        best_accs = None
        if best_round_hits is not None:
            best_accs = compute_best_so_far_accuracies(best_round_hits)
            message += f" | Best-so-far {format_accuracy_snapshot(best_accs)}"
        append_live_accuracy_snapshot(
            exp_name,
            processed,
            total,
            accs,
            best_accs=best_accs,
        )
        print(message, flush=True)

def build_split_size_message(train_data, val_data, test_data):
    return (
        f"train={len(train_data)} | "
        f"val={len(val_data)} | "
        f"test={len(test_data)}"
    )

def resolve_num_rounds(update_params):
    return TRAIN_NUM_ROUNDS if update_params else INFER_NUM_ROUNDS

def is_proposal_stage_round(rnd):
    return int(rnd) % 2 == 1

def get_proposal_review_stage(rnd):
    return "proposal" if is_proposal_stage_round(rnd) else "review"

def get_pending_vote_score_bucket(pending_vote_score, pending_candidate_text=None):
    if not has_pending_candidate(pending_candidate_text):
        return "none"
    score = int(pending_vote_score or 0)
    if score <= -1:
        return "neg"
    return "nonneg"

def get_pending_controller_score_bucket(pending_vote_score, pending_candidate_text=None):
    if not has_pending_candidate(pending_candidate_text):
        return "none"
    score = int(pending_vote_score or 0)
    return "score1" if score >= 1 else "score0"

def get_remaining_proposal_rounds_bucket(rnd, total_rounds=NUM_ROUNDS):
    if rnd is None:
        return "unknown"
    current_round = int(rnd)
    last_round = int(total_rounds)
    remaining = 0
    for sim_rnd in range(current_round, last_round + 1):
        if get_proposal_review_stage(sim_rnd) == "proposal":
            remaining += 1
    return "last_chance" if remaining <= 1 else "not_last"

def get_proposal_review_time_bucket(rnd, total_rounds=NUM_ROUNDS):
    remaining_bucket = get_remaining_proposal_rounds_bucket(rnd, total_rounds=total_rounds)
    return "last" if remaining_bucket == "last_chance" else "not_last"

def get_proposal_review_accept_threshold(incumbent_pred):
    if incumbent_pred is None:
        return PROPOSAL_REVIEW_BOOTSTRAP_ACCEPT_SCORE
    return PROPOSAL_REVIEW_REPLACE_ACCEPT_SCORE

def get_proposal_review_drop_threshold(incumbent_pred):
    if incumbent_pred is None:
        return PROPOSAL_REVIEW_BOOTSTRAP_DROP_SCORE
    return PROPOSAL_REVIEW_DROP_SCORE

def get_round_role(rnd):
    if is_proposal_review_schema():
        return "proposer" if is_proposal_stage_round(rnd) else "reviewer"
    return "solver" if rnd % 2 == 1 else "commenter"

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

def compute_phase(prev_phase_value, rnd, eps=PHASE_Q_EPS):
    """
    把外层状态压缩成两个 phase：
    - search
    - stabilize

    当前规则：
    - 第 1 轮固定为 search
    - 从第 2 轮开始，看上一轮该真实分支自己的单个 phase 更新信号
    - 若该值 > eps，则继续 search
    - 否则进入 stabilize
    """
    if rnd <= 1:
        return "search", 0.0

    phase_score = float(prev_phase_value)
    phase = "search" if phase_score > eps else "stabilize"
    return phase, phase_score

def same_numeric_prediction(pred_a, pred_b, tol=1e-9):
    if pred_a is None or pred_b is None:
        return False
    return abs(float(pred_a) - float(pred_b)) < tol

def compute_phase_update_signal(
    act,
    current_pred=None,
    incumbent_pred=None,
):
    """
    训练/推理统一使用的无标签 phase 更新信号。

    目标：
    - train / val / test 共用同一套 phase 转移规则
    - phase 只依赖真实轨迹当前可观测到的动作和答案变化
    - 不再把 reward/value 或 ground-truth 喂回后续轮次

    规则：
    - silent: 0.0，表示本轮没有引入新信息，下一轮偏 stabilize
    - comment: 1.0，表示仍在探索，下一轮进入 search
    - answer:
      - 若当前 answer 不可解析，记 1.0，表示问题仍未稳定
      - 若这是第一条可解析 answer，记 1.0
      - 若与上一轮 incumbent 数值答案相同，记 0.0
      - 若给出了不同于 incumbent 的新答案，记 1.0

    因而 phase 实际上是一个“轨迹是否发生实质变化”的二值状态机。
    """
    if act == MIDDLE_ACTION_SILENT:
        return 0.0
    if act == MIDDLE_ACTION_COMMENT:
        return 1.0

    if current_pred is None:
        return 1.0
    if incumbent_pred is None:
        return 1.0
    if same_numeric_prediction(current_pred, incumbent_pred):
        return 0.0
    return 1.0

def get_round_bucket(rnd, total_rounds=NUM_ROUNDS):
    late_threshold = max(2, int(total_rounds) - 1)
    return "late" if int(rnd) >= late_threshold else "early"


def build_outer_state(
    phase,
    incumbent_pred=None,
    rnd=None,
    total_rounds=NUM_ROUNDS,
    pending_candidate_text=None,
    pending_vote_score=0,
):
    if is_proposal_review_schema():
        return (
            "outer",
            f"has_incumbent={int(incumbent_pred is not None)}",
            f"has_pending={int(has_pending_candidate(pending_candidate_text))}",
            f"pending_vote_bucket={get_pending_vote_score_bucket(pending_vote_score, pending_candidate_text)}",
            f"remaining_proposal_rounds_bucket={get_remaining_proposal_rounds_bucket(rnd, total_rounds=total_rounds)}",
        )
    state = [
        "outer",
        f"phase={phase}",
    ]
    if THREE_LAYER_STATE_KEY_MODE == "expanded":
        state.append(f"has_incumbent={int(incumbent_pred is not None)}")
        if rnd is not None:
            state.append(f"round_bucket={get_round_bucket(rnd, total_rounds=total_rounds)}")
    return tuple(state)


def build_middle_state(
    selected_agent,
    phase,
    incumbent_pred=None,
    rnd=None,
    total_rounds=NUM_ROUNDS,
    pending_candidate_text=None,
    pending_vote_score=0,
):
    if is_proposal_review_schema():
        if not has_pending_candidate(pending_candidate_text):
            return (
                "controller",
                "mode=bootstrap",
            )
        return (
            "controller",
            f"pending_score_bucket={get_pending_controller_score_bucket(pending_vote_score, pending_candidate_text)}",
            f"proposal_time_bucket={get_proposal_review_time_bucket(rnd, total_rounds=total_rounds)}",
        )
    state = [
        "middle",
        f"agent={selected_agent}",
        f"phase={phase}",
    ]
    if THREE_LAYER_STATE_KEY_MODE == "expanded":
        state.append(f"has_incumbent={int(incumbent_pred is not None)}")
        if rnd is not None:
            state.append(f"round_bucket={get_round_bucket(rnd, total_rounds=total_rounds)}")
    return tuple(state)


def resolve_forced_middle_action(rnd, allowed_actions):
    if is_proposal_review_schema():
        return None
    mode = THREE_LAYER_ACTION_OVERRIDE_MODE
    if mode == "always_answer":
        desired_action = MIDDLE_ACTION_ANSWER
    elif mode == "fixed_answer_comment":
        desired_action = MIDDLE_ACTION_ANSWER if rnd % 2 == 1 else MIDDLE_ACTION_COMMENT
    else:
        return None

    if desired_action in allowed_actions:
        return desired_action
    if MIDDLE_ACTION_ANSWER in allowed_actions:
        return MIDDLE_ACTION_ANSWER
    if MIDDLE_ACTION_COMMENT in allowed_actions:
        return MIDDLE_ACTION_COMMENT
    if allowed_actions:
        return allowed_actions[0]
    return None

def build_middle_fallback_strategy(state_key, allowed_actions, num_actions, default_action):
    """
    中层策略的初始化分布。

    只在该 state 下所有正遗憾都为 0 时生效：
    - search: answer / comment 各 1/2，silent 为 0
    - stabilize: answer / comment / silent 各 1/3

    无沉默实验会通过 allowed_actions 的 mask 自动退化成：
    - search: answer / comment 各 1/2
    - stabilize: answer / comment 各 1/2
    """
    if is_proposal_review_schema():
        mode = "pending_control"
        if isinstance(state_key, (list, tuple)):
            for item in state_key:
                if isinstance(item, str) and item.startswith("stage="):
                    # 兼容旧版状态键；新版 proposal/review 由 allowed_actions 本身区分
                    if item.split("=", 1)[1] == "review":
                        fallback = np.zeros(num_actions, dtype=np.float64)
                        fallback[MIDDLE_ACTION_COMMENT] = 1.0
                        return fallback
                if isinstance(item, str) and item.startswith("mode="):
                    mode = item.split("=", 1)[1]
        fallback = np.zeros(num_actions, dtype=np.float64)
        if mode == "bootstrap":
            fallback[MIDDLE_ACTION_ANSWER] = 1.0
        elif mode == "stop":
            fallback[MIDDLE_ACTION_COMMENT] = 1.0
        else:
            fallback[MIDDLE_ACTION_COMMENT] = 0.5
            fallback[MIDDLE_ACTION_ANSWER] = 0.5
        return fallback

    phase = "stabilize"
    if isinstance(state_key, (list, tuple)):
        for item in state_key:
            if isinstance(item, str) and item.startswith("phase="):
                phase = item.split("=", 1)[1]
                break

    fallback = np.zeros(num_actions, dtype=np.float64)
    if phase == "search":
        fallback[MIDDLE_ACTION_COMMENT] = 0.5
        fallback[MIDDLE_ACTION_ANSWER] = 0.5
    else:
        fallback[MIDDLE_ACTION_SILENT] = 1.0 / 3.0
        fallback[MIDDLE_ACTION_COMMENT] = 1.0 / 3.0
        fallback[MIDDLE_ACTION_ANSWER] = 1.0 / 3.0
    return fallback

def normalize_middle_strategy(strategy, allowed_actions):
    mask = np.zeros(CFR_NUM_ACTIONS, dtype=bool)
    for action in allowed_actions:
        mask[action] = True

    normalized = np.zeros(CFR_NUM_ACTIONS, dtype=np.float64)
    normalized[mask] = np.maximum(np.asarray(strategy, dtype=np.float64)[mask], 0.0)
    total = normalized.sum()
    if total > 0:
        normalized /= total
        return normalized

    normalized[mask] = 1.0 / max(mask.sum(), 1)
    return normalized

def project_middle_strategy_with_bounds(
    strategy,
    allowed_actions,
    lower_bounds=None,
    upper_bounds=None,
):
    """
    给中层策略施加 floor / cap 约束，并重新归一化。

    这一步用于把“学出来的 regret-matching 分布”和
    “实验上必须保留的最小探索/最大静默”折中到一起。
    """
    strategy = normalize_middle_strategy(strategy, allowed_actions)
    lower_bounds = lower_bounds or {}
    upper_bounds = upper_bounds or {}

    mask = np.zeros(CFR_NUM_ACTIONS, dtype=bool)
    for action in allowed_actions:
        mask[action] = True

    floors = np.zeros(CFR_NUM_ACTIONS, dtype=np.float64)
    caps = np.zeros(CFR_NUM_ACTIONS, dtype=np.float64)
    caps[mask] = 1.0

    for action, value in lower_bounds.items():
        if 0 <= action < CFR_NUM_ACTIONS and mask[action]:
            floors[action] = max(floors[action], min(float(value), 1.0))
    for action, value in upper_bounds.items():
        if 0 <= action < CFR_NUM_ACTIONS and mask[action]:
            caps[action] = min(caps[action], max(float(value), 0.0))

    for action in range(CFR_NUM_ACTIONS):
        if floors[action] > caps[action]:
            floors[action] = caps[action]

    floor_total = floors.sum()
    if floor_total >= 1.0 - 1e-12:
        return normalize_middle_strategy(floors, allowed_actions)

    result = floors.copy()
    remaining = 1.0 - floor_total
    capacities = caps - floors
    base = np.maximum(strategy - floors, 0.0)
    available = [action for action in allowed_actions if capacities[action] > 1e-12]

    while remaining > 1e-12 and available:
        weights = np.array([base[action] for action in available], dtype=np.float64)
        weight_total = float(weights.sum())
        if weight_total <= 1e-12:
            weights = np.ones(len(available), dtype=np.float64)
            weight_total = float(len(available))

        consumed = 0.0
        next_available = []
        for idx, action in enumerate(available):
            add = remaining * (weights[idx] / weight_total)
            add = min(add, capacities[action])
            if add > 0:
                result[action] += add
                capacities[action] -= add
                consumed += add
            if capacities[action] > 1e-12:
                next_available.append(action)

        if consumed <= 1e-12:
            break
        remaining -= consumed
        available = next_available

    return normalize_middle_strategy(result, allowed_actions)

def apply_middle_action_runtime_gates(
    allowed_actions,
    phase,
    incumbent_pred,
):
    gated_actions = list(allowed_actions)
    if (
        THREE_LAYER_DISABLE_ANSWER_ON_STABILIZE_WITH_INCUMBENT
        and phase == "stabilize"
        and incumbent_pred is not None
        and MIDDLE_ACTION_ANSWER in gated_actions
    ):
        without_answer = [
            action for action in gated_actions
            if action != MIDDLE_ACTION_ANSWER
        ]
        if without_answer:
            gated_actions = without_answer
    return gated_actions


def get_middle_allowed_actions(
    rnd,
    incumbent_pred,
    allow_silent=True,
    total_rounds=NUM_ROUNDS,
    phase=None,
    pending_candidate_text=None,
    pending_vote_score=0,
):
    """
    中层动作空间：
    - silent
    - comment
    - answer

    下一版把几个对结果影响最大的约束重新接回：
    - 如果还没有 incumbent，就禁止 silent
    - 如果已经到最后一轮，禁止 comment
    - 若最后一轮仍没有 incumbent，则只允许 answer
    """
    if is_proposal_review_schema():
        del allow_silent, total_rounds, phase, pending_vote_score, incumbent_pred
        if get_proposal_review_stage(rnd) == "review":
            return [MIDDLE_ACTION_COMMENT]
        if has_pending_candidate(pending_candidate_text):
            return [MIDDLE_ACTION_COMMENT, MIDDLE_ACTION_ANSWER]
        return [MIDDLE_ACTION_ANSWER]

    if rnd >= total_rounds:
        if incumbent_pred is None:
            return apply_middle_action_runtime_gates(
                [MIDDLE_ACTION_ANSWER],
                phase,
                incumbent_pred,
            )
        if allow_silent:
            return apply_middle_action_runtime_gates(
                [MIDDLE_ACTION_SILENT, MIDDLE_ACTION_ANSWER],
                phase,
                incumbent_pred,
            )
        return apply_middle_action_runtime_gates(
            [MIDDLE_ACTION_ANSWER],
            phase,
            incumbent_pred,
        )

    if incumbent_pred is None:
        return apply_middle_action_runtime_gates([
            MIDDLE_ACTION_COMMENT,
            MIDDLE_ACTION_ANSWER,
        ], phase, incumbent_pred)

    if allow_silent:
        return apply_middle_action_runtime_gates([
            MIDDLE_ACTION_SILENT,
            MIDDLE_ACTION_COMMENT,
            MIDDLE_ACTION_ANSWER,
        ], phase, incumbent_pred)
    return apply_middle_action_runtime_gates([
        MIDDLE_ACTION_COMMENT,
        MIDDLE_ACTION_ANSWER,
    ], phase, incumbent_pred)

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

def get_single_gspo_round_prompt(rnd):
    """
    单智能体 GSPO 复用单 LLM 的 prompt 节奏。

    也就是说：
    - 第 1 轮用 SINGLE_PROMPTS[0]
    - 第 2 轮用 SINGLE_PROMPTS[1]
    - ...

    这里 rnd 是从 1 开始计数，所以要写 rnd - 1。
    """
    if rnd <= len(SINGLE_PROMPTS):
        return SINGLE_PROMPTS[rnd - 1]

    desired_parity = rnd % 2
    for idx in range(len(SINGLE_PROMPTS) - 1, -1, -1):
        if (idx + 1) % 2 == desired_parity:
            return SINGLE_PROMPTS[idx]
    return SINGLE_PROMPTS[-1]

def get_next_answer_prompt(rnd, total_rounds=NUM_ROUNDS):
    next_round = min(rnd + 1, total_rounds)
    return get_single_gspo_round_prompt(next_round)

def get_three_layer_action_name(
    act,
    rnd=None,
    incumbent_pred=None,
    pending_candidate_text=None,
):
    if is_proposal_review_schema():
        stage = get_proposal_review_stage(rnd) if rnd is not None else None
        if stage == "review":
            return "review"
        if incumbent_pred is not None:
            return "stop"
        if has_pending_candidate(pending_candidate_text):
            return "keep_pending" if act == MIDDLE_ACTION_COMMENT else "refresh_pending"
        return "propose_initial"
    if act == MIDDLE_ACTION_SILENT:
        return "pi2"
    if act == MIDDLE_ACTION_COMMENT:
        return "pi0"
    return "pi1"

def resolve_effective_parallel_mode(
    parallel_mode,
    proposal_review_mode,
    num_role_bundles,
):
    if parallel_mode != "serial":
        return parallel_mode
    if not proposal_review_mode:
        return parallel_mode
    if int(num_role_bundles) != 1:
        return parallel_mode
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        return "three_layer_workers"
    return parallel_mode

def choose_outer_agent(
    scheduler,
    phase,
    num_agents=None,
    use_average_strategy=False,
    incumbent_pred=None,
    rnd=None,
    total_rounds=NUM_ROUNDS,
    pending_candidate_text=None,
    pending_vote_score=0,
):
    state = build_outer_state(
        phase,
        incumbent_pred=incumbent_pred,
        rnd=rnd,
        total_rounds=total_rounds,
        pending_candidate_text=pending_candidate_text,
        pending_vote_score=pending_vote_score,
    )
    allowed_agents = list(range(NUM_AGENTS if num_agents is None else num_agents))
    strategy = scheduler.get_strategy(
        state,
        allowed_agents,
        use_average_strategy=use_average_strategy,
    )
    selected_agent = scheduler.get_action(
        state_key=state,
        allowed_actions=allowed_agents,
        explore=not THREE_LAYER_EVAL_DETERMINISTIC_ACTIONS,
        use_average_strategy=use_average_strategy,
    )
    return state, allowed_agents, strategy, selected_agent

def get_constrained_middle_strategy(
    selector,
    phase,
    rnd,
    selected_agent,
    incumbent_pred,
    allow_silent=True,
    allowed_actions=None,
    use_average_strategy=False,
    total_rounds=NUM_ROUNDS,
    pending_candidate_text=None,
    pending_vote_score=0,
):
    state = build_middle_state(
        selected_agent,
        phase,
        incumbent_pred=incumbent_pred,
        rnd=rnd,
        total_rounds=total_rounds,
        pending_candidate_text=pending_candidate_text,
        pending_vote_score=pending_vote_score,
    )
    if allowed_actions is None:
        allowed_actions = get_middle_allowed_actions(
            rnd,
            incumbent_pred,
            allow_silent=allow_silent,
            total_rounds=total_rounds,
            phase=phase,
            pending_candidate_text=pending_candidate_text,
            pending_vote_score=pending_vote_score,
        )
    if is_proposal_review_schema():
        strategy = selector.get_strategy(
            state,
            allowed_actions,
            use_average_strategy=use_average_strategy,
        )
        return state, allowed_actions, normalize_middle_strategy(strategy, allowed_actions)
    strategy = selector.get_strategy(
        state,
        allowed_actions,
        use_average_strategy=use_average_strategy,
    )
    lower_bounds = {}
    upper_bounds = {}

    if phase == "search":
        if MIDDLE_ACTION_COMMENT in allowed_actions:
            lower_bounds[MIDDLE_ACTION_COMMENT] = SEARCH_MIN_COMMENT_PROB
        if MIDDLE_ACTION_ANSWER in allowed_actions:
            lower_bounds[MIDDLE_ACTION_ANSWER] = SEARCH_MIN_ANSWER_PROB
        if MIDDLE_ACTION_SILENT in allowed_actions:
            upper_bounds[MIDDLE_ACTION_SILENT] = SEARCH_MAX_SILENT_PROB

        if rnd >= total_rounds - 1 and MIDDLE_ACTION_ANSWER in allowed_actions:
            lower_bounds[MIDDLE_ACTION_ANSWER] = max(
                lower_bounds.get(MIDDLE_ACTION_ANSWER, 0.0),
                min(1.0, SEARCH_MIN_ANSWER_PROB + SEARCH_LATE_ROUND_ANSWER_BONUS),
            )

    if incumbent_pred is None:
        if rnd <= NO_INCUMBENT_FORCE_ANSWER_ROUNDS and MIDDLE_ACTION_ANSWER in allowed_actions:
            lower_bounds[MIDDLE_ACTION_ANSWER] = 1.0
            upper_bounds[MIDDLE_ACTION_COMMENT] = 0.0
            upper_bounds[MIDDLE_ACTION_SILENT] = 0.0
        elif MIDDLE_ACTION_ANSWER in allowed_actions:
            lower_bounds[MIDDLE_ACTION_ANSWER] = max(
                lower_bounds.get(MIDDLE_ACTION_ANSWER, 0.0),
                SEARCH_NO_INCUMBENT_MIN_ANSWER_PROB,
            )
            if MIDDLE_ACTION_SILENT in allowed_actions:
                upper_bounds[MIDDLE_ACTION_SILENT] = min(
                    upper_bounds.get(MIDDLE_ACTION_SILENT, 1.0),
                    SEARCH_NO_INCUMBENT_MAX_SILENT_PROB,
                )

    strategy = project_middle_strategy_with_bounds(
        strategy,
        allowed_actions,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    return state, allowed_actions, strategy

def choose_middle_action(
    selector,
    phase,
    rnd,
    selected_agent,
    incumbent_pred,
    allow_silent=True,
    use_average_strategy=False,
    total_rounds=NUM_ROUNDS,
    pending_candidate_text=None,
    pending_vote_score=0,
):
    state, allowed_actions, strategy = get_constrained_middle_strategy(
        selector,
        phase,
        rnd,
        selected_agent,
        incumbent_pred,
        allow_silent=allow_silent,
        use_average_strategy=use_average_strategy,
        total_rounds=total_rounds,
        pending_candidate_text=pending_candidate_text,
        pending_vote_score=pending_vote_score,
    )
    forced_action = resolve_forced_middle_action(rnd, allowed_actions)
    if forced_action is not None:
        act = int(forced_action)
    elif is_proposal_review_schema() and PROPOSAL_REVIEW_CONTROLLER_OVERRIDE_MODE != "learned":
        mode = PROPOSAL_REVIEW_CONTROLLER_OVERRIDE_MODE
        if mode == "always_keep":
            act = MIDDLE_ACTION_COMMENT if MIDDLE_ACTION_COMMENT in allowed_actions else allowed_actions[0]
        elif mode == "always_refresh":
            act = MIDDLE_ACTION_ANSWER if MIDDLE_ACTION_ANSWER in allowed_actions else allowed_actions[0]
        elif mode == "fixed_keep_refresh":
            desired = MIDDLE_ACTION_COMMENT if rnd % 4 == 1 else MIDDLE_ACTION_ANSWER
            act = desired if desired in allowed_actions else allowed_actions[0]
        elif mode == "uniform":
            act = int(np.random.choice(allowed_actions))
        else:
            raise ValueError(f"Unsupported PROPOSAL_REVIEW_CONTROLLER_OVERRIDE_MODE: {mode}")
    elif THREE_LAYER_EVAL_DETERMINISTIC_ACTIONS:
        act = int(np.argmax(strategy))
    else:
        act = int(np.random.choice(CFR_NUM_ACTIONS, p=strategy))
    return state, allowed_actions, strategy, act

def maybe_print_three_layer_round_debug(
    sample_idx,
    rnd,
    phase,
    phase_score,
    selected_agent,
    act,
    pre_incumbent_pred,
    pre_pending_candidate_text,
    candidate_pred,
    tracked_pred,
    gt,
    realized_value,
    outer_strategy,
    middle_strategy,
):
    """
    三层策略的逐轮调试打印。

    这个打印主要是为了解答类似下面这类问题：
    - 为什么 R4 不等于 R3？
    - 为什么某一轮 reward 涨了，但 accuracy 没涨？

    打印字段解释：
    - act: pi2 / pi0 / pi1
    - candidate_pred: 本轮答案候选里提取出的数字
    - tracked_pred: 本轮统计口径下用于记准确率的最终预测
    - phase_score: 上一轮实际执行动作的 value，用于决定本轮 search / stabilize
    """
    if not THREE_LAYER_DEBUG_PRINT:
        return

    hit = bool(compute_accuracy([tracked_pred], [gt]))
    current_pred_str = "None" if candidate_pred is None else f"{candidate_pred:.4f}"
    tracked_pred_str = "None" if tracked_pred is None else f"{tracked_pred:.4f}"
    outer_strategy_str = ", ".join(f"{prob:.4f}" for prob in outer_strategy)
    middle_strategy_str = ", ".join(f"{prob:.4f}" for prob in middle_strategy)
    actor_label = "role_bundle" if is_proposal_review_schema() else "agent"
    scheduler_label = "scheduler" if is_proposal_review_schema() else "outer"
    controller_label = "controller" if is_proposal_review_schema() else "middle"
    print(
        f"[三层调试] sample={sample_idx} rnd={rnd} phase={phase} "
        f"phase_score={phase_score:.4f} {actor_label}={selected_agent} "
        f"act={get_three_layer_action_name(act, rnd=rnd, incumbent_pred=pre_incumbent_pred, pending_candidate_text=pre_pending_candidate_text)} candidate_pred={current_pred_str} "
        f"final_pred={tracked_pred_str} gt={gt:.4f} hit={int(hit)} "
        f"realized_value={realized_value:.4f} "
        f"{scheduler_label}=[{outer_strategy_str}] "
        f"{controller_label}=[{middle_strategy_str}]"
    )

def maybe_print_three_layer_stage(sample_idx, rnd, stage, start_time):
    if not THREE_LAYER_DEBUG_PRINT:
        return
    elapsed = time.perf_counter() - start_time
    print(f"[三层阶段] sample={sample_idx} rnd={rnd} stage={stage} elapsed={elapsed:.2f}s")

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

def sample_gspo_step(
    agent,
    question,
    history,
    rnd,
    speaker,
    kind,
    prompt_override=None,
    total_rounds=None,
    context_override=None,
):
    """
    GSPO 版本里，一轮 step 不只是“生成一条文本”，
    还要把训练更新所需的缓存一起带回来。

    返回的 step 是一个 dict（字典）。

    Python 语法说明：
    - dict 就是 {键: 值, 键: 值, ...}
    - 后面可以通过 step["selected_text"] 这种写法取字段
    - 这个 helper 对应训练 / batch 采样路径；验证/测试单轨迹请用 sample_single_text_step
    """
    ctx = context_override if context_override is not None else build_context(question, history)
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
        "total_rounds": total_rounds,
        "context_override": context_override,
        "batch": batch,
        "selected_text": selected_text,
    }

def sample_single_text_step(
    agent,
    question,
    history,
    rnd,
    speaker,
    kind,
    prompt_override=None,
    total_rounds=None,
    context_override=None,
):
    """
    单轨迹采样。

    与 sample_gspo_step 的区别：
    - 不构造候选 batch
    - 不存在 selected_idx
    - proposal-review 训练里也会用它估计 controller 的
      真实执行 / 反事实动作 value
    """
    ctx = context_override if context_override is not None else build_context(question, history)
    selected_text = agent.sample_text(ctx, prompt_override=prompt_override)
    return {
        "agent": agent,
        "question": question,
        "pre_history": history,
        "round": rnd,
        "speaker": speaker,
        "kind": kind,
        "prompt_override": prompt_override,
        "total_rounds": total_rounds,
        "context_override": context_override,
        "batch": None,
        "selected_text": selected_text,
    }

def rollout_from_step(question, gt, step, future_steps, current_text, num_samples=None):
    """
    从某个 step 开始，向后模拟未来轨迹，并估计它会带来多少奖励。

    这是 deferred reward / with-without 奖励的核心组成部分。

    直觉上可以把它理解成：
    - “如果我把这一步内容设成 current_text”
    - “然后后面继续往下模拟”
    - “最后能得到多好的答案？”
    """
    history = append_round_output(
        step["pre_history"],
        step["round"],
        step["speaker"],
        current_text,
        step["kind"],
    )
    if step["kind"] == "answer":
        pred = extract_pred_num(current_text)
        return reward_from_pred(pred, gt)

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

    return reward_from_pred(final_pred, gt)

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
    )
    return [
        rollout_from_step(
            step["question"],
            gt,
            step,
            future_steps,
            cand["text"],
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

def compute_answer_candidate_rewards(
    batch,
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
    num_samples=None,
):
    """
    估计“从当前历史继续 rollout，到最近一次 answer 出现时”的绝对 reward。

    当前明确只采样 1 条虚拟轨迹：
    - 不会在一个 comment 候选下再展开一整组 answer
    - 从而避免 comment->answer 的组合在 rollout 中指数膨胀

    这个量不再掺入 incumbent baseline，
    主要给 comment 的 with/without 比较与 silent 基线使用。
    """
    answer_text = agent_bundle["pi1"].sample_text(
        build_context(question, history),
        prompt_override=PI1_PROMPT,
    )
    answer_pred = extract_pred_num(answer_text)
    return rollout_terminal_absolute_reward(answer_pred, incumbent_pred, gt)

def batch_next_answer_rewards(
    agent_bundle,
    question,
    histories,
    incumbent_pred,
    gt,
):
    if not histories:
        return []

    contexts = [build_context(question, history) for history in histories]
    answer_texts = agent_bundle["pi1"].sample_text_batch(
        contexts,
        prompt_override=PI1_PROMPT,
    )
    rewards = []
    for answer_text in answer_texts:
        answer_pred = extract_pred_num(answer_text)
        rewards.append(
            rollout_terminal_absolute_reward(
                answer_pred,
                incumbent_pred,
                gt,
            )
        )
    return rewards

def average_answer_delta_reward(
    agent_bundle,
    question,
    history,
    rnd,
    incumbent_pred,
    gt,
    num_samples=None,
):
    """
    估计“如果现在尝试 answer，平均能带来多少相对 incumbent 的提升”：

        reward(next_answer) - reward(incumbent)

    这个量用于：
    - 中层 CFR 中 action=answer 的反事实估值
    """
    answer_text = agent_bundle["pi1"].sample_text(
        build_context(question, history),
        prompt_override=PI1_PROMPT,
    )
    answer_pred = extract_pred_num(answer_text)
    return rollout_terminal_delta_reward(answer_pred, incumbent_pred, gt)

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

def compute_comment_candidate_rewards(step, agent_bundle, incumbent_pred, gt):
    """
    给 comment policy 的候选 comment 打分。

    算法是典型的 with-comment / without-comment 对比：
    - without_comment:
      不加这条 comment，直接 rollout 1 个下一步 answer 的绝对 reward
    - with_comment:
      把这条 comment 拼进历史，再 rollout 1 个下一步 answer 的绝对 reward
    - 两者差值：
      就是这条 comment 对后续 answer 的边际帮助
    """
    total_rounds = step.get("total_rounds") or NUM_ROUNDS
    without_comment = average_next_answer_reward(
        agent_bundle,
        step["question"],
        step["pre_history"],
        min(step["round"] + 1, total_rounds),
        incumbent_pred,
        gt,
    )

    next_histories = []
    for cand in step["batch"]["candidates"]:
        next_histories.append(
            append_round_output(
                step["pre_history"],
                step["round"],
                step["speaker"],
                cand["text"],
                "comment",
            )
        )
    with_comment_rewards = batch_next_answer_rewards(
        agent_bundle,
        step["question"],
        next_histories,
        incumbent_pred,
        gt,
    )
    return [reward - without_comment for reward in with_comment_rewards]

def average_policy_stack_answer_reward(agent_bundle, question, history, gt, num_samples=None):
    """
    给共享策略栈估计“下一次 answer 自身的质量 r”。

    这里不再回退到 incumbent reward：
    - answer 的价值就是 answer utterance 自身的质量
    - 如果这次 answer 不可解析，就直接按 r(None) 记分
    """
    answer_text = agent_bundle["pi1"].sample_text(
        build_context(question, history),
        prompt_override=PI1_PROMPT,
    )
    answer_pred = extract_pred_num(answer_text)
    return reward_from_pred(answer_pred, gt)

def estimate_middle_silent_baseline(
    agent_bundle,
    question,
    history,
    rnd,
    incumbent_pred,
    gt,
    num_samples=None,
    total_rounds=NUM_ROUNDS,
):
    """
    中层动作 value 的共享静默基线：

        B(s) = E[next_answer_reward | 当前状态 s 下本轮选择 silent]
    """
    return average_next_answer_reward(
        agent_bundle,
        question,
        history,
        min(rnd + 1, total_rounds),
        incumbent_pred,
        gt,
    )

def compute_centered_answer_candidate_values(batch, gt, silent_baseline):
    values = []
    for cand in batch["candidates"]:
        candidate_pred = extract_pred_num(cand["text"])
        values.append(reward_from_pred(candidate_pred, gt) - silent_baseline)
    return values

def sample_counterfactual_policy_batch(
    agent,
    question,
    history,
    prompt_override=None,
    context_override=None,
):
    """
    只读反事实采样：
    - 不写 agent.last_update
    - 不污染真实轨迹这一步已经记录的 selected_text
    """
    query_context = (
        context_override
        if context_override is not None else build_context(question, history)
    )
    candidates = agent.act(query_context, prompt_override=prompt_override)
    if not candidates:
        raise RuntimeError("反事实采样未返回任何候选。")
    return {
        "query_context": query_context,
        "prompt_override": prompt_override,
        "candidates": candidates,
        "selected_idx": 0,
        "selected_text": candidates[0]["text"],
    }

def estimate_policy_stack_comment_value(
    agent_bundle,
    question,
    history,
    rnd,
    incumbent_pred,
    gt,
    num_samples=None,
    comment_text=None,
    speaker="pi0_cf",
    silent_baseline=None,
    total_rounds=NUM_ROUNDS,
):
    """
    comment 的价值恢复为旧定义：

        with_comment 的下一次 answer 的 r
        - without_comment 的下一次 answer 的 r
    """
    if silent_baseline is None:
        silent_baseline = estimate_middle_silent_baseline(
            agent_bundle,
            question,
            history,
            rnd,
            incumbent_pred,
            gt,
            total_rounds=total_rounds,
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
        min(rnd + 1, total_rounds),
        incumbent_pred,
        gt,
    )
    return with_comment_reward - silent_baseline

def compute_policy_stack_comment_candidate_rewards(
    step,
    agent_bundle,
    incumbent_pred,
    gt,
    silent_baseline=None,
):
    total_rounds = step.get("total_rounds") or NUM_ROUNDS
    if silent_baseline is None:
        silent_baseline = estimate_middle_silent_baseline(
            agent_bundle,
            step["question"],
            step["pre_history"],
            step["round"],
            incumbent_pred,
            gt,
            total_rounds=total_rounds,
        )
    next_histories = []
    for cand in step["batch"]["candidates"]:
        next_histories.append(
            append_round_output(
                step["pre_history"],
                step["round"],
                step["speaker"],
                cand["text"],
                "comment",
            )
        )
    with_comment_rewards = batch_next_answer_rewards(
        agent_bundle,
        step["question"],
        next_histories,
        incumbent_pred,
        gt,
    )
    return [reward - silent_baseline for reward in with_comment_rewards]

def resolve_parallel_mode():
    parallel_mode = os.environ.get("MAS_PARALLEL_MODE", "serial").strip() or "serial"
    if parallel_mode not in {"serial", "three_layer_workers"}:
        raise ValueError(f"不支持的 MAS_PARALLEL_MODE: {parallel_mode}")
    return parallel_mode

def estimate_policy_stack_silent_value(
    agent_bundle,
    question,
    history,
    rnd,
    incumbent_pred,
    gt,
):
    """
    中层策略现在退化为最朴素的 regret matching。

    在这个版本里：
    - comment 继续使用 GSPO 层定义的边际价值
    - answer 继续使用 GSPO 层定义的答案质量
    - silent 作为 no-op 基线动作，其额外价值固定记为 0

    这样：
    - comment > 0 表示“比静默更有帮助”
    - comment < 0 表示“这条 comment 还不如不说”
    - regret 更新只需直接比较三动作 value，不再混入额外 rollout 基线
    """
    return 0.0

def estimate_policy_stack_answer_value(
    agent_bundle,
    question,
    history,
    rnd,
    incumbent_pred,
    gt,
    silent_baseline=None,
    total_rounds=NUM_ROUNDS,
):
    if silent_baseline is None:
        silent_baseline = estimate_middle_silent_baseline(
            agent_bundle,
            question,
            history,
            rnd,
            incumbent_pred,
            gt,
            total_rounds=total_rounds,
        )
    batch = sample_counterfactual_policy_batch(
        agent_bundle["pi1"],
        question,
        history,
        prompt_override=PI1_PROMPT,
    )
    values = compute_centered_answer_candidate_values(batch, gt, silent_baseline)
    return sum(values) / len(values) if values else 0.0

def estimate_policy_stack_comment_value_mean(
    agent_bundle,
    question,
    history,
    rnd,
    incumbent_pred,
    gt,
    silent_baseline=None,
    total_rounds=NUM_ROUNDS,
):
    if silent_baseline is None:
        silent_baseline = estimate_middle_silent_baseline(
            agent_bundle,
            question,
            history,
            rnd,
            incumbent_pred,
            gt,
            total_rounds=total_rounds,
        )
    batch = sample_counterfactual_policy_batch(
        agent_bundle["pi0"],
        question,
        history,
        prompt_override=PI0_PROMPT,
    )
    step = {
        "question": question,
        "pre_history": history,
        "round": rnd,
        "speaker": "pi0_cf_batch",
        "total_rounds": total_rounds,
        "batch": batch,
    }
    rewards = compute_policy_stack_comment_candidate_rewards(
        step,
        agent_bundle,
        incumbent_pred,
        gt,
        silent_baseline=silent_baseline,
    )
    return sum(rewards) / len(rewards) if rewards else 0.0

def compute_outer_agent_value(middle_action_values, middle_strategy):
    """
    外层每个 agent 的动作价值：

        V_outer(agent, s) = sum_a pi_middle(a | agent, s) * Q_middle(agent, a, s)

    也就是当前中层策略下，该 agent 三个中层动作 value 的加权平均。
    """
    return float(
        np.dot(
            np.asarray(middle_action_values, dtype=np.float64),
            np.asarray(middle_strategy, dtype=np.float64),
        )
    )

def estimate_middle_action_values(
    question,
    gt,
    agents,
    phase,
    selected_agent,
    rnd,
    history,
    incumbent_text,
    incumbent_pred,
    pending_candidate_text=None,
    pending_candidate_pred=None,
    allow_silent=True,
    known_action=None,
    known_value=None,
    silent_baseline=None,
    total_rounds=NUM_ROUNDS,
    pending_vote_score=0,
    controller_selector=None,
    controller_use_average_strategy=False,
):
    """
    估计某个 agent 在当前 round/state 下的中层三动作价值。

    proposal-review 语义下，这里的 value 与真实轨迹保持一致：
    - 如果某个动作已经在真实轨迹里执行过，并且它的 realized value 已知，
      那么这里直接复用，不再为这个动作再跑一遍反事实 rollout。
    - 对未选中的动作，只额外采样 1 条单轨迹文本作为反事实 realized sample。
    - proposal keep/refresh 的 controller value 使用 signed exact 规则：
      保住正确答案记 +1，保住错误答案记 -1；
      refresh 后答案正确记 +1，错误记 -1；
      same-answer refresh 按 keep 语义回流。
    """
    allowed_actions = get_middle_allowed_actions(
        rnd,
        incumbent_pred,
        allow_silent=allow_silent,
        total_rounds=total_rounds,
        phase=phase,
        pending_candidate_text=pending_candidate_text,
        pending_vote_score=pending_vote_score,
    )
    values = np.zeros(CFR_NUM_ACTIONS, dtype=np.float64)
    agent_bundle = agents[selected_agent]
    if is_proposal_review_schema():
        stage = get_proposal_review_stage(rnd)
        context_override = build_sampling_context(
            question,
            history,
            incumbent_text=incumbent_text,
            pending_candidate_text=pending_candidate_text,
            pending_vote_score=pending_vote_score,
            include_latest_review_reason=(
                stage == "proposal"
                and not PROPOSAL_REVIEW_DISABLE_REVIEW_FEEDBACK_CONTEXT
            ),
        )
        if PROPOSAL_REVIEW_DISABLE_COUNTERFACTUAL_VALUES:
            if known_action is not None and known_value is not None:
                values[int(known_action)] = float(known_value)
            return values, allowed_actions
        if MIDDLE_ACTION_COMMENT in allowed_actions:
            if known_action == MIDDLE_ACTION_COMMENT and known_value is not None:
                values[MIDDLE_ACTION_COMMENT] = known_value
            elif stage == "proposal":
                proposal_mode = get_proposal_review_proposal_mode(
                    rnd,
                    incumbent_pred,
                    pending_candidate_text,
                    MIDDLE_ACTION_COMMENT,
                )
                resolved = resolve_proposal_action_outcome(
                    history,
                    rnd,
                    f"controller_cf_keep_agent{selected_agent}",
                    proposal_mode,
                    None,
                    None,
                    incumbent_text,
                    incumbent_pred,
                    pending_candidate_text,
                    pending_candidate_pred,
                    pending_vote_score,
                )
                values[MIDDLE_ACTION_COMMENT] = float(
                    get_proposal_middle_value(
                        proposal_mode,
                        None,
                        incumbent_pred,
                        pending_candidate_pred,
                        gt,
                        proposal_outcome=resolved,
                    )
                )
            elif (
                not has_pending_candidate(pending_candidate_text)
                or pending_candidate_pred is None
            ):
                values[MIDDLE_ACTION_COMMENT] = 0.0
            else:
                step = sample_single_text_step(
                    agent_bundle["pi0"],
                    question,
                    history,
                    rnd,
                    f"reviewer_cf_agent{selected_agent}",
                    "review",
                    prompt_override=PI0_PROMPT,
                    total_rounds=total_rounds,
                    context_override=context_override,
                )
                values[MIDDLE_ACTION_COMMENT] = float(
                    compute_review_reward(
                        agent_bundle,
                        controller_selector,
                        controller_use_average_strategy,
                        question,
                        history,
                        rnd,
                        parse_review_verdict(step["selected_text"]),
                        incumbent_text,
                        incumbent_pred,
                        pending_candidate_text,
                        pending_candidate_pred,
                        pending_vote_score,
                        gt,
                        total_rounds=total_rounds,
                        review_text=step["selected_text"],
                    )
                )
        if MIDDLE_ACTION_ANSWER in allowed_actions:
            if known_action == MIDDLE_ACTION_ANSWER and known_value is not None:
                values[MIDDLE_ACTION_ANSWER] = known_value
            else:
                proposal_mode = get_proposal_review_proposal_mode(
                    rnd,
                    incumbent_pred,
                    pending_candidate_text,
                    MIDDLE_ACTION_ANSWER,
                )
                step = sample_single_text_step(
                    agent_bundle["pi1"],
                    question,
                    history,
                    rnd,
                    f"proposer_cf_agent{selected_agent}",
                    "proposal",
                    prompt_override=get_proposal_review_proposer_prompt(proposal_mode),
                    total_rounds=total_rounds,
                    context_override=context_override,
                )
                resolved = resolve_proposal_action_outcome(
                    history,
                    rnd,
                    f"proposer_cf_agent{selected_agent}",
                    proposal_mode,
                    step["selected_text"],
                    parse_proposal_candidate_pred(step["selected_text"]),
                    incumbent_text,
                    incumbent_pred,
                    pending_candidate_text,
                    pending_candidate_pred,
                    pending_vote_score,
                )
                values[MIDDLE_ACTION_ANSWER] = float(
                    get_proposal_middle_value(
                        proposal_mode,
                        parse_proposal_candidate_pred(step["selected_text"]),
                        incumbent_pred,
                        pending_candidate_pred,
                        gt,
                        proposal_outcome=resolved,
                    )
                )
        return values, allowed_actions

    if silent_baseline is None:
        silent_baseline = estimate_middle_silent_baseline(
            agent_bundle,
            question,
            history,
            rnd,
            incumbent_pred,
            gt,
            total_rounds=total_rounds,
        )

    if MIDDLE_ACTION_COMMENT in allowed_actions:
        if known_action == MIDDLE_ACTION_COMMENT and known_value is not None:
            values[MIDDLE_ACTION_COMMENT] = known_value
        else:
            values[MIDDLE_ACTION_COMMENT] = estimate_policy_stack_comment_value_mean(
                agent_bundle,
                question,
                history,
                rnd,
                incumbent_pred,
                gt,
                silent_baseline=silent_baseline,
                total_rounds=total_rounds,
            )
    if MIDDLE_ACTION_ANSWER in allowed_actions:
        if known_action == MIDDLE_ACTION_ANSWER and known_value is not None:
            values[MIDDLE_ACTION_ANSWER] = known_value
        else:
            values[MIDDLE_ACTION_ANSWER] = estimate_policy_stack_answer_value(
                agent_bundle,
                question,
                history,
                rnd,
                incumbent_pred,
                gt,
                silent_baseline=silent_baseline,
                total_rounds=total_rounds,
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
            )
    return values, allowed_actions

def get_cached_middle_estimate(
    cache,
    question,
    gt,
    agents,
    phase,
    agent_idx,
    rnd,
    history,
    incumbent_text,
    incumbent_pred,
    pending_candidate_text=None,
    pending_candidate_pred=None,
    allow_silent=True,
    known_action=None,
    known_value=None,
    silent_baseline=None,
    total_rounds=NUM_ROUNDS,
    pending_vote_score=0,
    controller_selector=None,
    controller_use_average_strategy=False,
):
    """
    每轮每个 agent 的中层动作价值只估一次，并在本轮内复用。

    这样：
    - 中层 regret 更新会用到它
    - 外层调度 regret 也直接复用同一份 value vector
    """
    cache_key = (
        agent_idx,
        phase,
        rnd,
        bool(allow_silent),
        int(total_rounds),
        int(incumbent_pred is not None),
        int(has_pending_candidate(pending_candidate_text, pending_candidate_pred)),
        int(pending_vote_score or 0),
    )
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
        phase,
        agent_idx,
        rnd,
        history,
        incumbent_text,
        incumbent_pred,
        pending_candidate_text=pending_candidate_text,
        pending_candidate_pred=pending_candidate_pred,
        allow_silent=allow_silent,
        known_action=known_action,
        known_value=known_value,
        silent_baseline=silent_baseline,
        total_rounds=total_rounds,
        pending_vote_score=pending_vote_score,
        controller_selector=controller_selector,
        controller_use_average_strategy=controller_use_average_strategy,
    )
    cache[cache_key] = (np.array(values, copy=True), list(allowed_actions))
    return values, allowed_actions

def init_three_layer_round_buffers(num_rounds):
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
        "round_preds": [[] for _ in range(num_rounds)],
        "round_texts": [[] for _ in range(num_rounds)],
        "round_gts": [[] for _ in range(num_rounds)],
        "best_round_hits": [[] for _ in range(num_rounds)],
        "round_rewards": [[] for _ in range(num_rounds)],
        "round_regrets": [[] for _ in range(num_rounds)],
        "round_outer_probs": [[] for _ in range(num_rounds)],
        "round_middle_probs": [[] for _ in range(num_rounds)],
        "round_search_flags": [[] for _ in range(num_rounds)],
        "round_stabilize_flags": [[] for _ in range(num_rounds)],
        "round_improve_flags": [[] for _ in range(num_rounds)],
        "round_degrade_flags": [[] for _ in range(num_rounds)],
        "round_stalled_wrong_flags": [[] for _ in range(num_rounds)],
        "round_preserved_correct_flags": [[] for _ in range(num_rounds)],
    }

def init_three_layer_sample_state():
    """
    初始化“单条样本”在三层实验里的运行状态。

    这里存的是“会随着轮次变化”的状态变量：
    - history:
      当前样本到目前为止的结构化对话历史
    - incumbent_text / incumbent_pred:
      当前最新一次 answer 的文本和数字答案；如果后续轮次 comment/silent，
      它只作为 latest_answer fallback 被沿用
    - prev_phase_value:
      上一轮用于决定下一轮 phase 的信号。
      train / val / test 统一使用无标签“轨迹变化”信号，
      避免 reward/value 或 ground-truth 泄露到后续决策。
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
        "pending_candidate_text": None,
        "pending_candidate_pred": None,
        "pending_vote_score": 0,
        "final_text": None,
        "final_pred": None,
        "prev_phase_value": 0.0,
        "best_correct": False,
        "first_improve_round": None,
        "first_degrade_round": None,
        "first_stalled_wrong_round": None,
    }

def clone_branch_state_with_outcome(parent_state, outcome, gt, rnd):
    previous_final_pred = resolve_final_pred(
        parent_state["incumbent_pred"],
        parent_state.get("pending_candidate_pred"),
    )
    current_final_pred = resolve_final_pred(
        outcome["incumbent_pred"],
        outcome.get("pending_candidate_pred"),
    )
    transition = classify_round_transition(
        previous_final_pred,
        current_final_pred,
        gt,
    )
    final_text = resolve_final_text(
        outcome["incumbent_text"],
        outcome.get("pending_candidate_text"),
    )
    child_state = {
        "history": outcome["history"],
        "incumbent_text": outcome["incumbent_text"],
        "incumbent_pred": outcome["incumbent_pred"],
        "incumbent_reward": reward_from_pred(outcome["incumbent_pred"], gt),
        "pending_candidate_text": outcome.get("pending_candidate_text"),
        "pending_candidate_pred": outcome.get("pending_candidate_pred"),
        "pending_vote_score": int(outcome.get("pending_vote_score", 0) or 0),
        "final_text": final_text,
        "final_pred": current_final_pred,
        "prev_phase_value": float(
            outcome.get("phase_update_signal", outcome["realized_value"])
        ),
        "best_correct": parent_state["best_correct"] or bool(
            compute_accuracy([current_final_pred], [gt])
        ),
        "first_improve_round": parent_state["first_improve_round"],
        "first_degrade_round": parent_state["first_degrade_round"],
        "first_stalled_wrong_round": parent_state["first_stalled_wrong_round"],
    }
    if transition["improved"] and child_state["first_improve_round"] is None:
        child_state["first_improve_round"] = rnd
    if transition["degraded"] and child_state["first_degrade_round"] is None:
        child_state["first_degrade_round"] = rnd
    if transition["stalled_wrong"] and child_state["first_stalled_wrong_round"] is None:
        child_state["first_stalled_wrong_round"] = rnd
    return child_state, transition


def sync_step_selected_candidate(step, selected_idx):
    if step is None or step.get("batch") is None:
        return 0
    candidates = step["batch"].get("candidates", [])
    if not candidates:
        return 0
    selected_idx = max(0, min(int(selected_idx), len(candidates) - 1))
    selected_text = candidates[selected_idx]["text"]
    step["batch"]["selected_idx"] = selected_idx
    step["batch"]["selected_text"] = selected_text
    step["selected_text"] = selected_text
    return selected_idx


def get_realized_candidate_idx(step, candidate_values):
    if step is None or step.get("batch") is None:
        return 0
    candidates = step["batch"].get("candidates", [])
    if not candidates:
        return 0

    max_idx = min(len(candidates), len(candidate_values)) - 1
    if max_idx < 0:
        return max(0, min(int(step["batch"].get("selected_idx", 0)), len(candidates) - 1))

    selected_idx = int(step["batch"].get("selected_idx", 0))
    selected_idx = max(0, min(selected_idx, max_idx))
    if THREE_LAYER_REALIZED_BRANCH_SELECTION == "reward_best" and candidate_values:
        candidate_indices = list(range(max_idx + 1))
        if step.get("kind") == "proposal":
            eligible_proposal_indices = [
                idx
                for idx in candidate_indices
                if proposal_is_state_eligible(candidates[idx].get("text"))
            ]
            if eligible_proposal_indices:
                candidate_indices = eligible_proposal_indices
            else:
                candidate_indices = [selected_idx]
        elif step.get("kind") == "review":
            valid_review_indices = [
                idx
                for idx in candidate_indices
                if is_effectively_valid_review_text(
                    candidates[idx].get("text"),
                    step.get("pending_candidate_text"),
                )
            ]
            if valid_review_indices:
                candidate_indices = valid_review_indices
            else:
                candidate_indices = [selected_idx]
        selected_idx = max(
            candidate_indices,
            key=lambda idx: float(candidate_values[idx]),
        )
    return selected_idx


def get_realized_candidate_value(step, candidate_values, default=0.0):
    if not candidate_values:
        return float(default)
    selected_idx = get_realized_candidate_idx(step, candidate_values)
    selected_idx = max(0, min(selected_idx, len(candidate_values) - 1))
    return float(candidate_values[selected_idx])


def choose_realized_candidate_idx(step, candidate_values):
    return sync_step_selected_candidate(
        step,
        get_realized_candidate_idx(step, candidate_values),
    )


def resolve_answer_incumbent_state(
    pre_incumbent_text,
    pre_incumbent_pred,
    answer_text,
    answer_pred,
    gt,
    phase=None,
):
    if (
        THREE_LAYER_KEEP_INCUMBENT_ON_STABILIZE_DIFFERENT_ANSWER
        and phase == "stabilize"
        and pre_incumbent_pred is not None
        and answer_pred is not None
        and not same_numeric_prediction(answer_pred, pre_incumbent_pred)
    ):
        return (
            pre_incumbent_text,
            pre_incumbent_pred,
            reward_from_pred(pre_incumbent_pred, gt),
        )
    if (
        THREE_LAYER_KEEP_INCUMBENT_ON_MISSING_ANSWER
        and answer_pred is None
        and pre_incumbent_pred is not None
    ):
        return (
            pre_incumbent_text,
            pre_incumbent_pred,
            reward_from_pred(pre_incumbent_pred, gt),
        )
    return (
        answer_text,
        answer_pred,
        reward_from_pred(answer_pred, gt),
    )

def materialize_round_children(parent_state, round_result, gt, rnd, expand_all):
    outcomes = round_result["candidate_outcomes"]
    if not expand_all:
        selected_outcome_idx = int(round_result.get("selected_outcome_idx", 0))
        selected_outcome_idx = max(0, min(selected_outcome_idx, len(outcomes) - 1))
        outcomes = [outcomes[selected_outcome_idx]] if outcomes else []

    children = []
    for outcome in outcomes:
        child_state, transition = clone_branch_state_with_outcome(
            parent_state,
            outcome,
            gt,
            rnd,
        )
        children.append({
            "state": child_state,
            "transition": transition,
            "outcome": outcome,
        })
    return children

def get_parallel_agent_device_set(parallel_controller, agent_idx):
    """
    返回某个 agent 在并行 worker 模式下会占用到的策略设备集合。

    regret 阶段的中层 value 估计会同时触发：
    - pi1 的 silent baseline / answer 反事实生成
    - pi0 的 comment 候选生成

    如果两个 agent 的这些 worker 映射到了同一张卡，就不能再在
    ThreadPoolExecutor 里并发估值，否则很容易在 eager generate 路径里
    因为显存碎片或 allocator 峰值而炸掉。
    """
    if parallel_controller is None:
        return frozenset()
    device_map = getattr(parallel_controller, "device_map", None) or {}
    devices = set()
    for policy_name in ("pi0", "pi1"):
        device = device_map.get(f"agent{agent_idx}.{policy_name}")
        if device is not None:
            devices.add(str(device))
    return frozenset(devices)


def build_regret_context_waves(allowed_agents, parallel_controller):
    """
    把 agent 划成若干个“可安全并发”的 wave。

    同一个 wave 内，任意两个 agent 的策略设备集合都必须不相交；
    否则退到后续 wave，最终按 wave 逐批执行。

    这样仍然保留“设备不冲突时尽量并发”的收益，但会自动把
    共卡的 counterfactual generate 串行化，避免 03 这类实验在
    regret 阶段因为共享 GPU worker 并发 act() 而崩掉。
    """
    pending_agents = [int(agent_idx) for agent_idx in allowed_agents]
    if len(pending_agents) <= 1 or parallel_controller is None:
        return [pending_agents]

    agent_devices = {
        agent_idx: get_parallel_agent_device_set(parallel_controller, agent_idx)
        for agent_idx in pending_agents
    }
    if not all(agent_devices.values()):
        return [pending_agents]

    waves = []
    while pending_agents:
        used_devices = set()
        wave = []
        deferred = []
        for agent_idx in pending_agents:
            device_set = agent_devices[agent_idx]
            if used_devices.intersection(device_set):
                deferred.append(agent_idx)
                continue
            wave.append(agent_idx)
            used_devices.update(device_set)
        if not wave:
            wave = [pending_agents[0]]
            deferred = pending_agents[1:]
        waves.append(wave)
        pending_agents = deferred
    return waves

def run_three_layer_proposal_review_action(
    role_bundles,
    question,
    gt,
    rnd,
    selected_role_bundle_idx,
    act,
    pre_history,
    pre_incumbent_text,
    pre_incumbent_pred,
    pre_pending_candidate_text,
    pre_pending_candidate_pred,
    pre_pending_vote_score=0,
    use_candidate_batch=True,
    total_rounds=NUM_ROUNDS,
    controller_selector=None,
    use_average_strategy=False,
    estimate_realized_value=True,
):
    context_override = build_sampling_context(
        question,
        pre_history,
        incumbent_text=pre_incumbent_text,
        pending_candidate_text=pre_pending_candidate_text,
        pending_vote_score=pre_pending_vote_score,
        include_latest_review_reason=(
            get_proposal_review_stage(rnd) == "proposal"
            and not PROPOSAL_REVIEW_DISABLE_REVIEW_FEEDBACK_CONTEXT
        ),
    )
    candidate_outcomes = []
    candidate_middle_values = []
    batch_middle_values = None
    selected_outcome_idx = 0
    realized_middle_value = 0.0
    gspo_update_payload = None
    gspo_update_applied = False
    step = None
    realized_step = None
    role_bundle = role_bundles[selected_role_bundle_idx]
    stage = get_proposal_review_stage(rnd)

    if stage == "review":
        if (
            not has_pending_candidate(pre_pending_candidate_text)
            or pre_pending_candidate_pred is None
        ):
            resolved = {
                "history": pre_history,
                "incumbent_text": pre_incumbent_text,
                "incumbent_pred": pre_incumbent_pred,
                "pending_candidate_text": pre_pending_candidate_text,
                "pending_candidate_pred": pre_pending_candidate_pred,
                "pending_vote_score": int(pre_pending_vote_score or 0),
                "current_pred": resolve_final_pred(
                    pre_incumbent_pred,
                    pre_pending_candidate_pred,
                ),
            }
            candidate_middle_values = [0.0]
            candidate_outcomes = [{
                **resolved,
                "realized_value": 0.0,
                "phase_update_signal": 0.0,
            }]
            return {
                **resolved,
                "realized_value": 0.0,
                "phase_update_signal": 0.0,
                "candidate_middle_values": candidate_middle_values,
                "batch_middle_values": batch_middle_values,
                "candidate_outcomes": candidate_outcomes,
                "selected_outcome_idx": 0,
                "step": None,
                "realized_step": None,
                "gspo_update_payload": None,
                "gspo_update_applied": False,
                "silent_baseline": None,
                "effective_action": act,
                "regret_action": act,
                "same_answer_refresh_as_keep": False,
            }
        if use_candidate_batch:
            step = sample_gspo_step(
                role_bundle["pi0"],
                question,
                pre_history,
                rnd,
                f"reviewer_bundle{selected_role_bundle_idx}",
                "review",
                prompt_override=PI0_PROMPT,
                total_rounds=total_rounds,
                context_override=context_override,
            )
            step["pending_candidate_text"] = pre_pending_candidate_text
            review_rewards = compute_review_candidate_rewards(
                step["batch"],
                role_bundle,
                controller_selector,
                use_average_strategy,
                question,
                pre_history,
                rnd,
                pre_incumbent_text,
                pre_incumbent_pred,
                pre_pending_candidate_text,
                pre_pending_candidate_pred,
                pre_pending_vote_score,
                gt,
                total_rounds=total_rounds,
            )
            gspo_update_payload = (
                step["agent"],
                step["batch"],
                review_rewards,
            )
            batch_middle_values = [float(v) for v in review_rewards]
            apply_three_layer_policy_updates(gspo_update_payload)
            gspo_update_applied = True
            realized_step = sample_single_text_step(
                role_bundle["pi0"],
                question,
                pre_history,
                rnd,
                f"reviewer_bundle{selected_role_bundle_idx}",
                "review",
                prompt_override=PI0_PROMPT,
                total_rounds=total_rounds,
                context_override=context_override,
            )
            if estimate_realized_value:
                realized_middle_value = float(
                    compute_review_reward(
                        role_bundle,
                        controller_selector,
                        use_average_strategy,
                        question,
                        pre_history,
                        rnd,
                        parse_review_verdict(realized_step["selected_text"]),
                        pre_incumbent_text,
                        pre_incumbent_pred,
                        pre_pending_candidate_text,
                        pre_pending_candidate_pred,
                        pre_pending_vote_score,
                        gt,
                        total_rounds=total_rounds,
                        review_text=realized_step["selected_text"],
                    )
                )
            candidate_middle_values = [realized_middle_value]
            selected_outcome_idx = 0
        else:
            step = sample_single_text_step(
                role_bundle["pi0"],
                question,
                pre_history,
                rnd,
                f"reviewer_bundle{selected_role_bundle_idx}",
                "review",
                prompt_override=PI0_PROMPT,
                total_rounds=total_rounds,
                context_override=context_override,
            )
            if estimate_realized_value:
                realized_middle_value = float(
                    compute_review_reward(
                        role_bundle,
                        controller_selector,
                        use_average_strategy,
                        question,
                        pre_history,
                        rnd,
                        parse_review_verdict(step["selected_text"]),
                        pre_incumbent_text,
                        pre_incumbent_pred,
                        pre_pending_candidate_text,
                        pre_pending_candidate_pred,
                        pre_pending_vote_score,
                        gt,
                        total_rounds=total_rounds,
                        review_text=step["selected_text"],
                    )
                )
            else:
                realized_middle_value = 0.0
            candidate_middle_values = [realized_middle_value]
            realized_step = step

        resolved = resolve_review_action_outcome(
            pre_history,
            rnd,
            f"reviewer_bundle{selected_role_bundle_idx}",
            realized_step["selected_text"],
            parse_review_verdict(realized_step["selected_text"]),
            pre_incumbent_text,
            pre_incumbent_pred,
            pre_pending_candidate_text,
            pre_pending_candidate_pred,
            pre_pending_vote_score,
        )
        if not use_candidate_batch and not estimate_realized_value:
            realized_middle_value = float(
                compute_terminal_delta_reward(
                    pre_incumbent_pred,
                    pre_pending_candidate_pred,
                    resolved["incumbent_pred"],
                    resolved["pending_candidate_pred"],
                    gt,
                )
            )
            candidate_middle_values = [realized_middle_value]
        candidate_outcomes.append({
            **resolved,
            "realized_value": realized_middle_value,
            "phase_update_signal": 0.0,
        })
        return {
            **resolved,
            "realized_value": realized_middle_value,
            "phase_update_signal": 0.0,
            "candidate_middle_values": candidate_middle_values,
            "batch_middle_values": batch_middle_values,
            "candidate_outcomes": candidate_outcomes,
            "selected_outcome_idx": selected_outcome_idx,
            "step": step,
            "realized_step": realized_step,
            "gspo_update_payload": gspo_update_payload,
            "gspo_update_applied": gspo_update_applied,
            "silent_baseline": None,
            "effective_action": act,
            "regret_action": act,
            "same_answer_refresh_as_keep": False,
        }

    proposal_mode = get_proposal_review_proposal_mode(
        rnd,
        pre_incumbent_pred,
        pre_pending_candidate_text,
        act,
    )
    if proposal_mode in {"keep_pending", "stop"}:
        resolved = resolve_proposal_action_outcome(
            pre_history,
            rnd,
            f"controller_bundle{selected_role_bundle_idx}",
            proposal_mode,
            None,
            None,
            pre_incumbent_text,
            pre_incumbent_pred,
            pre_pending_candidate_text,
            pre_pending_candidate_pred,
            pre_pending_vote_score,
        )
        realized_middle_value = float(
            get_proposal_middle_value(
                proposal_mode,
                None,
                pre_incumbent_pred,
                pre_pending_candidate_pred,
                gt,
                proposal_outcome=resolved,
            )
        )
        candidate_middle_values = [realized_middle_value]
        candidate_outcomes.append({
            **resolved,
            "realized_value": realized_middle_value,
            "phase_update_signal": 0.0,
        })
        return {
            **resolved,
            "realized_value": realized_middle_value,
            "phase_update_signal": 0.0,
            "candidate_middle_values": candidate_middle_values,
            "batch_middle_values": batch_middle_values,
            "candidate_outcomes": candidate_outcomes,
            "selected_outcome_idx": 0,
            "step": None,
            "realized_step": None,
            "gspo_update_payload": None,
            "gspo_update_applied": False,
            "silent_baseline": None,
            "effective_action": act,
            "regret_action": act,
            "same_answer_refresh_as_keep": False,
        }

    proposal_prompt = get_proposal_review_proposer_prompt(proposal_mode)

    if use_candidate_batch:
        step = sample_gspo_step(
            role_bundle["pi1"],
            question,
            pre_history,
            rnd,
            f"proposer_bundle{selected_role_bundle_idx}",
            "proposal",
            prompt_override=proposal_prompt,
            total_rounds=total_rounds,
            context_override=context_override,
        )
        proposal_rewards = compute_proposal_candidate_rewards(
            step["batch"],
            role_bundle,
            controller_selector,
            use_average_strategy,
            question,
            pre_history,
            rnd,
            pre_incumbent_text,
            pre_incumbent_pred,
            pre_pending_candidate_text,
            pre_pending_candidate_pred,
            pre_pending_vote_score,
            gt,
            total_rounds=total_rounds,
        )
        gspo_update_payload = (
            step["agent"],
            step["batch"],
            proposal_rewards,
        )
        batch_middle_values, _ = compute_proposal_controller_candidate_values(
            step["batch"],
            pre_history,
            rnd,
            pre_incumbent_text,
            pre_incumbent_pred,
            pre_pending_candidate_text,
            pre_pending_candidate_pred,
            pre_pending_vote_score,
            gt,
        )
        apply_three_layer_policy_updates(gspo_update_payload)
        gspo_update_applied = True
        realized_step = sample_single_text_step(
            role_bundle["pi1"],
            question,
            pre_history,
            rnd,
            f"proposer_bundle{selected_role_bundle_idx}",
            "proposal",
            prompt_override=proposal_prompt,
            total_rounds=total_rounds,
            context_override=context_override,
        )
        selected_outcome_idx = 0
    else:
        step = sample_single_text_step(
            role_bundle["pi1"],
            question,
            pre_history,
            rnd,
            f"proposer_bundle{selected_role_bundle_idx}",
            "proposal",
            prompt_override=proposal_prompt,
            total_rounds=total_rounds,
            context_override=context_override,
        )
        realized_middle_value = 0.0
        candidate_middle_values = [realized_middle_value]
        realized_step = step

    resolved = resolve_proposal_action_outcome(
        pre_history,
        rnd,
        f"proposer_bundle{selected_role_bundle_idx}",
        proposal_mode,
        realized_step["selected_text"],
        parse_proposal_candidate_pred(realized_step["selected_text"]),
        pre_incumbent_text,
        pre_incumbent_pred,
        pre_pending_candidate_text,
        pre_pending_candidate_pred,
        pre_pending_vote_score,
    )
    if not use_candidate_batch:
        realized_middle_value = float(
            get_proposal_middle_value(
                proposal_mode,
                parse_proposal_candidate_pred(realized_step["selected_text"]),
                pre_incumbent_pred,
                pre_pending_candidate_pred,
                gt,
                proposal_outcome=resolved,
            )
        )
        candidate_middle_values = [realized_middle_value]
    if use_candidate_batch:
        realized_middle_value = float(
            get_proposal_middle_value(
                proposal_mode,
                parse_proposal_candidate_pred(realized_step["selected_text"]),
                pre_incumbent_pred,
                pre_pending_candidate_pred,
                gt,
                proposal_outcome=resolved,
            )
        )
        candidate_middle_values = [realized_middle_value]
    candidate_outcomes.append({
        **resolved,
        "realized_value": realized_middle_value,
        "phase_update_signal": 0.0,
    })
    effective_action = act
    regret_action = act
    if resolved.get("same_answer_refresh_as_keep"):
        effective_action = MIDDLE_ACTION_COMMENT
        regret_action = MIDDLE_ACTION_COMMENT
    return {
        **resolved,
        "realized_value": realized_middle_value,
        "phase_update_signal": 0.0,
        "candidate_middle_values": candidate_middle_values,
        "batch_middle_values": batch_middle_values,
        "candidate_outcomes": candidate_outcomes,
        "selected_outcome_idx": selected_outcome_idx,
        "step": step,
        "realized_step": realized_step,
        "gspo_update_payload": gspo_update_payload,
        "gspo_update_applied": gspo_update_applied,
        "silent_baseline": None,
        "effective_action": effective_action,
        "regret_action": regret_action,
    }

def run_three_layer_realized_action(
    agents,
    question,
    gt,
    phase,
    rnd,
    selected_agent,
    act,
    pre_history,
    pre_incumbent_text,
    pre_incumbent_pred,
    pre_pending_candidate_text=None,
    pre_pending_candidate_pred=None,
    pre_pending_vote_score=0,
    use_candidate_batch=True,
    total_rounds=NUM_ROUNDS,
    controller_selector=None,
    use_average_strategy=False,
    estimate_realized_value=True,
):
    """
    执行三层策略在“真实轨迹”里的这一步动作。

    这一步只做一件事：
    - 在已经确定了 `selected_agent` 和 `act` 之后，
      真正去采样文本、算 reward，并在 answer 动作下直接更新 latest_answer。

    返回一个字典，里面把这一轮后续还会用到的所有中间结果都带出来：
    - history:
      这一轮执行完后的新历史
    - realized_value:
      中层 / 内层学习用的动作价值
    - phase_update_signal:
      下一轮 phase 使用的无标签状态转移信号。
    - gspo_update_payload:
      给内层 GSPO policy 做更新时需要的缓存批次和 reward

    你可以把它理解成“把长主循环里最核心的那段 if/elif/else 单独搬出来”。
    """
    if is_proposal_review_schema():
        return run_three_layer_proposal_review_action(
            agents,
            question,
            gt,
            rnd,
            selected_agent,
            act,
            pre_history,
            pre_incumbent_text,
            pre_incumbent_pred,
            pre_pending_candidate_text,
            pre_pending_candidate_pred,
            pre_pending_vote_score,
            use_candidate_batch=use_candidate_batch,
            total_rounds=total_rounds,
            controller_selector=controller_selector,
            use_average_strategy=use_average_strategy,
            estimate_realized_value=estimate_realized_value,
        )

    current_pred = None
    realized_middle_value = 0.0
    candidate_middle_values = []
    candidate_outcomes = []
    step = None
    gspo_update_payload = None
    history = pre_history
    incumbent_text = pre_incumbent_text
    incumbent_pred = pre_incumbent_pred
    incumbent_reward = reward_from_pred(incumbent_pred, gt)
    silent_baseline = None
    phase_update_signal = 0.0
    selected_outcome_idx = 0

    if act == MIDDLE_ACTION_SILENT:
        realized_middle_value = estimate_policy_stack_silent_value(
            agents[selected_agent],
            question,
            pre_history,
            rnd,
            pre_incumbent_pred,
            gt,
        )
        history = append_round_output(
            history,
            rnd,
            f"agent{selected_agent}_pi2",
            "pi2 选择不发言，本轮不修改 incumbent。",
            "silent",
        )
        phase_update_signal = compute_phase_update_signal(MIDDLE_ACTION_SILENT)
        candidate_middle_values = [realized_middle_value]
        candidate_outcomes = [{
            "history": history,
            "incumbent_text": incumbent_text,
            "incumbent_pred": incumbent_pred,
            "current_pred": None,
            "realized_value": realized_middle_value,
            "phase_update_signal": phase_update_signal,
        }]
    elif act == MIDDLE_ACTION_COMMENT:
        silent_baseline = estimate_middle_silent_baseline(
            agents[selected_agent],
            question,
            pre_history,
            rnd,
            pre_incumbent_pred,
            gt,
            total_rounds=total_rounds,
        )
        if use_candidate_batch:
            step = sample_gspo_step(
                agents[selected_agent]["pi0"],
                question,
                pre_history,
                rnd,
                f"agent{selected_agent}_pi0",
                "comment",
                prompt_override=PI0_PROMPT,
                total_rounds=total_rounds,
            )
            comment_rewards = compute_policy_stack_comment_candidate_rewards(
                step,
                agents[selected_agent],
                pre_incumbent_pred,
                gt,
                silent_baseline=silent_baseline,
            )
            gspo_update_payload = (
                step["agent"],
                step["batch"],
                comment_rewards,
            )
            candidate_middle_values = [float(v) for v in comment_rewards]
            selected_idx = choose_realized_candidate_idx(
                step,
                candidate_middle_values,
            )
            if comment_rewards:
                realized_middle_value = comment_rewards[selected_idx]
            selected_outcome_idx = selected_idx
        else:
            step = sample_single_text_step(
                agents[selected_agent]["pi0"],
                question,
                pre_history,
                rnd,
                f"agent{selected_agent}_pi0",
                "comment",
                prompt_override=PI0_PROMPT,
                total_rounds=total_rounds,
            )
            realized_middle_value = float(
                estimate_policy_stack_comment_value(
                    agents[selected_agent],
                    question,
                    pre_history,
                    rnd,
                    pre_incumbent_pred,
                    gt,
                    comment_text=step["selected_text"],
                    speaker=step["speaker"],
                    silent_baseline=silent_baseline,
                    total_rounds=total_rounds,
                )
            )
            candidate_middle_values = [realized_middle_value]
            selected_outcome_idx = 0
        phase_update_signal = compute_phase_update_signal(
            MIDDLE_ACTION_COMMENT,
            current_pred=None,
            incumbent_pred=pre_incumbent_pred,
        )
        history = append_round_output(
            history,
            rnd,
            f"agent{selected_agent}_pi0",
            step["selected_text"],
            "comment",
        )
        if use_candidate_batch:
            for cand, cand_value in zip(step["batch"]["candidates"], candidate_middle_values):
                candidate_outcomes.append({
                    "history": append_round_output(
                        pre_history,
                        rnd,
                        f"agent{selected_agent}_pi0",
                        cand["text"],
                        "comment",
                    ),
                    "incumbent_text": pre_incumbent_text,
                    "incumbent_pred": pre_incumbent_pred,
                    "current_pred": None,
                    "realized_value": cand_value,
                    "phase_update_signal": phase_update_signal,
                })
        else:
            candidate_outcomes.append({
                "history": history,
                "incumbent_text": pre_incumbent_text,
                "incumbent_pred": pre_incumbent_pred,
                "current_pred": None,
                "realized_value": realized_middle_value,
                "phase_update_signal": phase_update_signal,
            })
    else:
        silent_baseline = estimate_middle_silent_baseline(
            agents[selected_agent],
            question,
            pre_history,
            rnd,
            pre_incumbent_pred,
            gt,
            total_rounds=total_rounds,
        )
        if use_candidate_batch:
            step = sample_gspo_step(
                agents[selected_agent]["pi1"],
                question,
                pre_history,
                rnd,
                f"agent{selected_agent}_pi1",
                "answer",
                prompt_override=PI1_PROMPT,
                total_rounds=total_rounds,
            )
            answer_rewards = compute_answer_candidate_rewards(
                step["batch"],
                gt,
            )
            gspo_update_payload = (
                step["agent"],
                step["batch"],
                answer_rewards,
            )
            middle_answer_values = compute_centered_answer_candidate_values(
                step["batch"],
                gt,
                silent_baseline,
            )
            candidate_middle_values = [float(v) for v in middle_answer_values]
            selected_idx = choose_realized_candidate_idx(
                step,
                candidate_middle_values,
            )
            if middle_answer_values:
                realized_middle_value = middle_answer_values[selected_idx]
            current_pred = extract_pred_num(step["selected_text"])
            selected_outcome_idx = selected_idx
        else:
            step = sample_single_text_step(
                agents[selected_agent]["pi1"],
                question,
                pre_history,
                rnd,
                f"agent{selected_agent}_pi1",
                "answer",
                prompt_override=PI1_PROMPT,
                total_rounds=total_rounds,
            )
            current_pred = extract_pred_num(step["selected_text"])
            realized_middle_value = float(
                reward_from_pred(current_pred, gt) - silent_baseline
            )
            candidate_middle_values = [realized_middle_value]
            selected_outcome_idx = 0
        incumbent_text, incumbent_pred, incumbent_reward = resolve_answer_incumbent_state(
            pre_incumbent_text,
            pre_incumbent_pred,
            step["selected_text"],
            current_pred,
            gt,
            phase=phase,
        )
        phase_update_signal = compute_phase_update_signal(
            MIDDLE_ACTION_ANSWER,
            current_pred=incumbent_pred,
            incumbent_pred=pre_incumbent_pred,
        )
        history = append_round_output(
            history,
            rnd,
            f"agent{selected_agent}_pi1",
            step["selected_text"],
            "answer",
        )
        if use_candidate_batch:
            for cand, cand_value in zip(step["batch"]["candidates"], candidate_middle_values):
                cand_pred = extract_pred_num(cand["text"])
                cand_incumbent_text, cand_incumbent_pred, _ = resolve_answer_incumbent_state(
                    pre_incumbent_text,
                    pre_incumbent_pred,
                    cand["text"],
                    cand_pred,
                    gt,
                    phase=phase,
                )
                candidate_outcomes.append({
                    "history": append_round_output(
                        pre_history,
                        rnd,
                        f"agent{selected_agent}_pi1",
                        cand["text"],
                        "answer",
                    ),
                    "incumbent_text": cand_incumbent_text,
                    "incumbent_pred": cand_incumbent_pred,
                    "current_pred": cand_pred,
                    "realized_value": cand_value,
                    "phase_update_signal": compute_phase_update_signal(
                        MIDDLE_ACTION_ANSWER,
                        current_pred=cand_incumbent_pred,
                        incumbent_pred=pre_incumbent_pred,
                    ),
                })
        else:
            candidate_outcomes.append({
                "history": history,
                "incumbent_text": incumbent_text,
                "incumbent_pred": incumbent_pred,
                "current_pred": current_pred,
                "realized_value": realized_middle_value,
                "phase_update_signal": phase_update_signal,
            })
    return {
        "history": history,
        "incumbent_text": incumbent_text,
        "incumbent_pred": incumbent_pred,
        "incumbent_reward": incumbent_reward,
        "silent_baseline": silent_baseline,
        "current_pred": current_pred,
        "realized_value": realized_middle_value,
        "phase_update_signal": phase_update_signal,
        "candidate_middle_values": candidate_middle_values,
        "candidate_outcomes": candidate_outcomes,
        "selected_outcome_idx": selected_outcome_idx,
        "step": step,
        "gspo_update_payload": gspo_update_payload,
    }

def prepare_three_layer_regret_context(
    outer_cfr,
    middle_cfr,
    agents,
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
    selected_middle_strategy,
    pre_history,
    pre_incumbent_text,
    pre_incumbent_pred,
    pre_pending_candidate_text,
    pre_pending_candidate_pred,
    pre_pending_vote_score,
    allow_silent,
    known_action=None,
    known_value=None,
    known_silent_baseline=None,
    parallel_controller=None,
    total_rounds=NUM_ROUNDS,
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
      各 agent 在“当前中层策略下的期望中层 value”分别是多少
    """
    selected_allowed_actions = list(allowed_actions)
    base_middle_action_values = np.zeros(CFR_NUM_ACTIONS, dtype=np.float64)
    outer_agent_context = {}

    if parallel_controller is not None and outer_cfr is not None and len(allowed_agents) > 1:
        def compute_agent_context(agent_idx):
            agent_allowed_actions = None
            if agent_idx == selected_agent:
                agent_middle_strategy = np.array(selected_middle_strategy, copy=True)
                agent_middle_values = np.zeros(CFR_NUM_ACTIONS, dtype=np.float64)
                if len(selected_allowed_actions) > 1:
                    agent_middle_values, agent_allowed_actions = get_cached_middle_estimate(
                        {},
                        question,
                        gt,
                        agents,
                        phase,
                        agent_idx,
                        rnd,
                        pre_history,
                        pre_incumbent_text,
                        pre_incumbent_pred,
                        pending_candidate_text=pre_pending_candidate_text,
                        pending_candidate_pred=pre_pending_candidate_pred,
                        pending_vote_score=pre_pending_vote_score,
                        allow_silent=allow_silent,
                        known_action=known_action,
                        known_value=known_value,
                        silent_baseline=known_silent_baseline,
                        total_rounds=total_rounds,
                        controller_selector=middle_cfr,
                        controller_use_average_strategy=False,
                    )
                else:
                    agent_allowed_actions = list(selected_allowed_actions)
            else:
                _, _, agent_middle_strategy = get_constrained_middle_strategy(
                    middle_cfr,
                    phase,
                    rnd,
                    agent_idx,
                    pre_incumbent_pred,
                    allow_silent=allow_silent,
                    total_rounds=total_rounds,
                    pending_candidate_text=pre_pending_candidate_text,
                    pending_vote_score=pre_pending_vote_score,
                )
                agent_middle_values, _ = get_cached_middle_estimate(
                    {},
                    question,
                    gt,
                    agents,
                    phase,
                    agent_idx,
                    rnd,
                    pre_history,
                    pre_incumbent_text,
                    pre_incumbent_pred,
                    pending_candidate_text=pre_pending_candidate_text,
                    pending_candidate_pred=pre_pending_candidate_pred,
                    pending_vote_score=pre_pending_vote_score,
                    allow_silent=allow_silent,
                    total_rounds=total_rounds,
                    controller_selector=middle_cfr,
                    controller_use_average_strategy=False,
                )
            return {
                "agent_idx": agent_idx,
                "middle_values": np.array(agent_middle_values, copy=True),
                "allowed_actions": (
                    list(agent_allowed_actions)
                    if agent_allowed_actions is not None else None
                ),
                "middle_strategy": np.array(agent_middle_strategy, copy=True),
            }

        for wave in build_regret_context_waves(allowed_agents, parallel_controller):
            if len(wave) == 1:
                payload_iter = [compute_agent_context(wave[0])]
            else:
                with ThreadPoolExecutor(max_workers=len(wave)) as executor:
                    payload_iter = list(executor.map(compute_agent_context, wave))
            for payload in payload_iter:
                agent_idx = payload["agent_idx"]
                if agent_idx == selected_agent:
                    base_middle_action_values = payload["middle_values"]
                    if payload["allowed_actions"] is not None:
                        selected_allowed_actions = payload["allowed_actions"]
                    continue
                outer_agent_context[agent_idx] = {
                    "middle_values": payload["middle_values"],
                    "middle_strategy": payload["middle_strategy"],
                }
    else:
        middle_estimate_cache = {}
        if len(selected_allowed_actions) > 1:
            base_middle_action_values, selected_allowed_actions = get_cached_middle_estimate(
                middle_estimate_cache,
                question,
                gt,
                agents,
                phase,
                selected_agent,
                rnd,
                pre_history,
                pre_incumbent_text,
                pre_incumbent_pred,
                pending_candidate_text=pre_pending_candidate_text,
                pending_candidate_pred=pre_pending_candidate_pred,
                pending_vote_score=pre_pending_vote_score,
                allow_silent=allow_silent,
                known_action=known_action,
                known_value=known_value,
                silent_baseline=known_silent_baseline,
                total_rounds=total_rounds,
                controller_selector=middle_cfr,
                controller_use_average_strategy=False,
            )

    if outer_cfr is not None and not outer_agent_context:
        for agent_idx in allowed_agents:
            if agent_idx == selected_agent:
                continue
            _, _, agent_middle_strategy = get_constrained_middle_strategy(
                middle_cfr,
                phase,
                rnd,
                agent_idx,
                pre_incumbent_pred,
                allow_silent=allow_silent,
                total_rounds=total_rounds,
                pending_candidate_text=pre_pending_candidate_text,
                pending_vote_score=pre_pending_vote_score,
            )
            agent_middle_values, _ = get_cached_middle_estimate(
                middle_estimate_cache,
                question,
                gt,
                agents,
                phase,
                agent_idx,
                rnd,
                pre_history,
                pre_incumbent_text,
                pre_incumbent_pred,
                pending_candidate_text=pre_pending_candidate_text,
                pending_candidate_pred=pre_pending_candidate_pred,
                pending_vote_score=pre_pending_vote_score,
                allow_silent=allow_silent,
                total_rounds=total_rounds,
                controller_selector=middle_cfr,
                controller_use_average_strategy=False,
            )
            outer_agent_context[agent_idx] = {
                "middle_values": np.array(agent_middle_values, copy=True),
                "middle_strategy": np.array(agent_middle_strategy, copy=True),
            }

    return {
        "outer_cfr": outer_cfr,
        "middle_cfr": middle_cfr,
        "outer_state": outer_state,
        "middle_state": middle_state,
        "selected_agent": selected_agent,
        "act": act,
        "allowed_agents": list(allowed_agents),
        "selected_allowed_actions": list(selected_allowed_actions),
        "selected_middle_strategy": np.array(selected_middle_strategy, copy=True),
        "base_middle_action_values": np.array(base_middle_action_values, copy=True),
        "outer_agent_context": outer_agent_context,
    }

def apply_prepared_three_layer_regret_update(
    regret_context,
    realized_value,
    update_regrets=True,
    strategy_sum_override=None,
):
    selected_allowed_actions = regret_context["selected_allowed_actions"]
    middle_action_values = np.array(
        regret_context["base_middle_action_values"],
        copy=True,
    )
    middle_action_values[regret_context["act"]] = float(realized_value)

    outer_action_values = None
    if regret_context["outer_cfr"] is not None:
        outer_action_values = np.zeros(len(regret_context["allowed_agents"]), dtype=np.float64)
        for agent_idx in regret_context["allowed_agents"]:
            if agent_idx == regret_context["selected_agent"]:
                agent_middle_values = middle_action_values
                agent_middle_strategy = regret_context["selected_middle_strategy"]
            else:
                agent_middle_values = regret_context["outer_agent_context"][agent_idx]["middle_values"]
                agent_middle_strategy = regret_context["outer_agent_context"][agent_idx]["middle_strategy"]
            outer_action_values[agent_idx] = compute_outer_agent_value(
                agent_middle_values,
                agent_middle_strategy,
            )

    if update_regrets and len(selected_allowed_actions) > 1:
        regret_context["middle_cfr"].update_regret(
            regret_context["middle_state"],
            regret_context["act"],
            middle_action_values,
            allowed_actions=selected_allowed_actions,
            strategy_sum_override=strategy_sum_override,
        )

    if update_regrets and regret_context["outer_cfr"] is not None:
        regret_context["outer_cfr"].update_regret(
            regret_context["outer_state"],
            regret_context["selected_agent"],
            outer_action_values,
            allowed_actions=regret_context["allowed_agents"],
        )

    return middle_action_values, outer_action_values

def update_three_layer_regrets(
    outer_cfr,
    middle_cfr,
    agents,
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
    selected_middle_strategy,
    pre_history,
    pre_incumbent_text,
    pre_incumbent_pred,
    pre_pending_candidate_text,
    pre_pending_candidate_pred,
    pre_pending_vote_score,
    allow_silent,
    realized_value,
    update_regrets=True,
    strategy_sum_override=None,
):
    regret_context = prepare_three_layer_regret_context(
        outer_cfr,
        middle_cfr,
        agents,
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
        selected_middle_strategy,
        pre_history,
        pre_incumbent_text,
        pre_incumbent_pred,
        pre_pending_candidate_text,
        pre_pending_candidate_pred,
        pre_pending_vote_score,
        allow_silent,
    )
    return apply_prepared_three_layer_regret_update(
        regret_context,
        realized_value,
        update_regrets=update_regrets,
        strategy_sum_override=strategy_sum_override,
    )

def apply_three_layer_policy_updates(
    gspo_update_payload,
):
    """
    统一做 proposal/review 内层 policy 的 GSPO 在线更新。

    当前共享策略栈只保留 GSPO 更新：
    - 真实轨迹里产生的一批 candidates
    - 配上本轮定义好的 reward/value
    - 做一次组内相对偏好更新

    注意时序：
    - proposal-review 的 26 号实验里，selected 单样本可以在这次更新后再采，
      让 controller 的 realized value 与新的 4+1 语义保持一致
    """
    if gspo_update_payload is not None:
        if THREE_LAYER_DISABLE_ALL_GSPO_UPDATES:
            return
        policy = gspo_update_payload[0]
        if THREE_LAYER_DISABLE_PI0_UPDATES and getattr(policy, "is_coop", False):
            return
        if THREE_LAYER_DISABLE_PI1_UPDATES and not getattr(policy, "is_coop", False):
            return
        policy.update_from_cached(
            gspo_update_payload[1],
            gspo_update_payload[2],
        )

def record_three_layer_round(
    buffers,
    rnd,
    gt,
    final_text,
    final_pred,
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
    buffers["round_preds"][rnd-1].append(final_pred)
    buffers["round_texts"][rnd-1].append(final_text)
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
    num_rounds = len(buffers["round_preds"])
    total_reward = 0.0
    for rnd in range(1, num_rounds + 1):
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

        if is_proposal_review_schema():
            print(
                f"{exp_name} 第{rnd}轮准确率：{acc:.4f} | Best-so-far：{best_acc:.4f} "
                f"| 累计奖励：{total_reward:.4f} | controller(base)={middle_probs[MIDDLE_ACTION_COMMENT]:.2f} "
                f"controller(change)={middle_probs[MIDDLE_ACTION_ANSWER]:.2f} "
                f"| improve={improve_rate:.2f} degrade={degrade_rate:.2f} "
                f"stalled_wrong={stalled_wrong_rate:.2f}"
            )
        else:
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
    num_rounds = INFER_NUM_ROUNDS
    round_preds = [[] for _ in range(num_rounds)]
    round_gts = [[] for _ in range(num_rounds)]
    best_round_hits = [[] for _ in range(num_rounds)]

    total_samples = len(data)
    for idx, item in enumerate(tqdm(data, desc="单LLM 按样本运行"), start=1):
        q = item["question"]
        gt = item["ground_truth"]
        history = []
        last_pred = None
        best_correct = False

        for rnd in range(1, num_rounds + 1):
            prompt = get_single_gspo_round_prompt(rnd)
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

    for rnd in range(1, num_rounds + 1):
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
    num_rounds = INFER_NUM_ROUNDS
    round_preds = [[] for _ in range(num_rounds)]
    round_gts = [[] for _ in range(num_rounds)]
    best_round_hits = [[] for _ in range(num_rounds)]

    total_samples = len(data)
    for idx, item in enumerate(tqdm(data, desc="双LLM轮询 按样本运行"), start=1):
        q = item["question"]
        gt = item["ground_truth"]
        history = []
        last_pred = None
        best_correct = False

        for rnd in range(1, num_rounds + 1):
            role = get_round_role(rnd)
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

    for rnd in range(1, num_rounds + 1):
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
    - 训练时会采样多个候选，并用整批 candidates 做 GSPO 更新
    - 验证/测试时不再伪造 batch，真实轨迹就是每轮单次采样出来的那一条文本

    所以它不是“best-of-G 测试时重排”，
    而是“训练用组内候选更新，推理时单次真实采样”。
    """
    print(f"\n=== 开始运行{exp_name}实验 ===")
    if agent is None:
        agent = GSPOAgentPolicy(is_coop=False)
    num_rounds = resolve_num_rounds(update_params)
    round_preds = [[] for _ in range(num_rounds)]
    round_gts = [[] for _ in range(num_rounds)]
    best_round_hits = [[] for _ in range(num_rounds)]

    total_samples = len(data)
    for idx, item in enumerate(tqdm(data, desc="单LLM GSPO 按样本运行"), start=1):
        q = item["question"]
        gt = item["ground_truth"]
        history = []
        last_pred = None
        best_correct = False

        for rnd in range(1, num_rounds + 1):
            prompt = get_single_gspo_round_prompt(rnd)
            is_answer_round = (rnd % 2 == 1)
            speaker = "single_gspo_answer" if is_answer_round else "single_gspo_comment"
            if update_params:
                step = sample_gspo_step(
                    agent,
                    q,
                    history,
                    rnd,
                    speaker,
                    "answer" if is_answer_round else "comment",
                    prompt_override=prompt,
                    total_rounds=num_rounds,
                )
            else:
                step = sample_single_text_step(
                    agent,
                    q,
                    history,
                    rnd,
                    speaker,
                    "answer" if is_answer_round else "comment",
                    prompt_override=prompt,
                    total_rounds=num_rounds,
                )
            res = step["selected_text"]
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
                        last_pred,
                        gt,
                    )
                agent.update_from_cached(step["batch"], rewards)

            round_preds[rnd-1].append(last_pred)
            round_gts[rnd-1].append(gt)
            best_correct = best_correct or bool(compute_accuracy([last_pred], [gt]))
            best_round_hits[rnd-1].append(1.0 if best_correct else 0.0)

        maybe_report_accuracy(exp_name, idx, total_samples, round_preds, round_gts, best_round_hits)

    for rnd in range(1, num_rounds + 1):
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
    - 训练时使用候选 batch 更新；验证/测试时每轮只真实采样 1 条文本
    """
    print(f"\n=== 开始运行{exp_name}实验 ===")
    if solver is None:
        solver = GSPOAgentPolicy(is_coop=False)
    if commenter is None:
        commenter = GSPOAgentPolicy(is_coop=True)
    num_rounds = resolve_num_rounds(update_params)
    round_preds = [[] for _ in range(num_rounds)]
    round_gts = [[] for _ in range(num_rounds)]
    best_round_hits = [[] for _ in range(num_rounds)]

    total_samples = len(data)
    for idx, item in enumerate(tqdm(data, desc="双LLM GSPO轮询 按样本运行"), start=1):
        q = item["question"]
        gt = item["ground_truth"]
        history = []
        last_pred = None
        best_correct = False

        for rnd in range(1, num_rounds + 1):
            role = get_round_role(rnd)
            if role == "solver":
                if update_params:
                    step = sample_gspo_step(
                        solver,
                        q,
                        history,
                        rnd,
                        role,
                        "answer",
                        prompt_override=SOLVER_PROMPT,
                        total_rounds=num_rounds,
                    )
                else:
                    step = sample_single_text_step(
                        solver,
                        q,
                        history,
                        rnd,
                        role,
                        "answer",
                        prompt_override=SOLVER_PROMPT,
                        total_rounds=num_rounds,
                    )
                res = step["selected_text"]
                current_pred = extract_pred_num(res)
                last_pred = update_last_pred(last_pred, current_pred, True)
            else:
                if update_params:
                    step = sample_gspo_step(
                        commenter,
                        q,
                        history,
                        rnd,
                        role,
                        "comment",
                        prompt_override=COMMENTER_PROMPT,
                        total_rounds=num_rounds,
                    )
                else:
                    step = sample_single_text_step(
                        commenter,
                        q,
                        history,
                        rnd,
                        role,
                        "comment",
                        prompt_override=COMMENTER_PROMPT,
                        total_rounds=num_rounds,
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
                        last_pred,
                        gt,
                    )
                    commenter.update_from_cached(step["batch"], rewards)
            round_preds[rnd-1].append(last_pred)
            round_gts[rnd-1].append(gt)
            best_correct = best_correct or bool(compute_accuracy([last_pred], [gt]))
            best_round_hits[rnd-1].append(1.0 if best_correct else 0.0)

        maybe_report_accuracy(exp_name, idx, total_samples, round_preds, round_gts, best_round_hits)

    for rnd in range(1, num_rounds + 1):
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

def build_parallel_worker_module_overrides():
    return {
        "MAX_NEW_TOKENS": int(MAX_NEW_TOKENS),
        "CHAT_SYSTEM_PROMPT": CHAT_SYSTEM_PROMPT,
        "THREE_LAYER_MIDDLE_ACTION_SCHEMA": THREE_LAYER_MIDDLE_ACTION_SCHEMA,
        "PROPOSAL_COMPLETION_MAX_NEW_TOKENS": int(PROPOSAL_COMPLETION_MAX_NEW_TOKENS),
        "REVIEW_COMPLETION_MAX_NEW_TOKENS": int(REVIEW_COMPLETION_MAX_NEW_TOKENS),
        "GSPO_EVAL_DO_SAMPLE": bool(GSPO_EVAL_DO_SAMPLE),
        "TEMPERATURE": float(TEMPERATURE),
        "GSPO_NUM_CANDIDATES": int(GSPO_NUM_CANDIDATES),
        "GSPO_NUM_GREEDY_CANDIDATES": int(GSPO_NUM_GREEDY_CANDIDATES),
        "PI0_PROMPT": PI0_PROMPT,
        "PI1_PROMPT": PI1_PROMPT,
        "PROPOSAL_REVIEW_CONTROLLER_OVERRIDE_MODE": PROPOSAL_REVIEW_CONTROLLER_OVERRIDE_MODE,
        "PROPOSAL_REVIEW_DISABLE_CONTROLLER_REGRET": bool(PROPOSAL_REVIEW_DISABLE_CONTROLLER_REGRET),
        "PROPOSAL_REVIEW_DISABLE_COUNTERFACTUAL_VALUES": bool(PROPOSAL_REVIEW_DISABLE_COUNTERFACTUAL_VALUES),
        "PROPOSAL_REVIEW_DISABLE_VOTE_UPDATES": bool(PROPOSAL_REVIEW_DISABLE_VOTE_UPDATES),
        "PROPOSAL_REVIEW_APPLY_VOTE_THRESHOLDS": bool(PROPOSAL_REVIEW_APPLY_VOTE_THRESHOLDS),
        "PROPOSAL_REVIEW_REFRESH_REQUIRES_NEGATIVE_PENDING": bool(PROPOSAL_REVIEW_REFRESH_REQUIRES_NEGATIVE_PENDING),
        "PROPOSAL_REVIEW_DISABLE_REFRESH": bool(PROPOSAL_REVIEW_DISABLE_REFRESH),
        "PROPOSAL_REVIEW_DISABLE_REVIEW_FEEDBACK_CONTEXT": bool(PROPOSAL_REVIEW_DISABLE_REVIEW_FEEDBACK_CONTEXT),
        "PROPOSAL_REVIEW_USE_LEGACY_PROMPTS": bool(PROPOSAL_REVIEW_USE_LEGACY_PROMPTS),
        "THREE_LAYER_DISABLE_PI0_UPDATES": bool(THREE_LAYER_DISABLE_PI0_UPDATES),
        "THREE_LAYER_DISABLE_PI1_UPDATES": bool(THREE_LAYER_DISABLE_PI1_UPDATES),
        "THREE_LAYER_DISABLE_ALL_GSPO_UPDATES": bool(THREE_LAYER_DISABLE_ALL_GSPO_UPDATES),
    }

def build_policy_stack_runtime(
    use_outer_scheduler,
    num_agents,
    allow_silent=True,
    parallel_mode="serial",
):
    outer_cfr = None
    if use_outer_scheduler:
        outer_cfr = CFRBehaviorSelector(num_actions=num_agents, default_action=0)
    middle_default_action = (
        MIDDLE_ACTION_ANSWER if is_proposal_review_schema() else MIDDLE_ACTION_SILENT
    )
    middle_cfr = CFRBehaviorSelector(
        num_actions=CFR_NUM_ACTIONS,
        default_action=middle_default_action,
        fallback_strategy_fn=build_middle_fallback_strategy,
    )
    parallel_controller = None
    if parallel_mode == "three_layer_workers":
        parallel_controller = create_three_layer_parallel_controller(
            project_root=os.path.dirname(os.path.abspath(__file__)),
            num_agents=num_agents,
        )
        parallel_controller.apply_module_overrides(
            build_parallel_worker_module_overrides()
        )
        agents = parallel_controller.build_agents()
    else:
        agents = []
        for _ in range(num_agents):
            pi0 = GSPOAgentPolicy(is_coop=True)
            pi1 = GSPOAgentPolicy(is_coop=False)
            agents.append({"pi0": pi0, "pi1": pi1, "pi2": None})
    return {
        "agents": agents,
        "outer_cfr": outer_cfr,
        "middle_cfr": middle_cfr,
        "use_outer_scheduler": use_outer_scheduler,
        "num_agents": num_agents,
        "allow_silent": allow_silent,
        "parallel_mode": parallel_mode,
        "parallel_controller": parallel_controller,
    }

def run_policy_stack_experiment(
    data,
    runtime=None,
    use_outer_scheduler=True,
    num_agents=None,
    allow_silent=True,
    parallel_mode="serial",
    update_params=True,
    log_metrics=True,
    exp_name="全量策略",
    log_fn=log_three_layer,
    teardown=True,
    checkpoint_options=None,
    force_use_average_strategy=None,
):
    print(f"\n=== 开始运行{exp_name}实验 ===")
    proposal_review_mode = is_proposal_review_schema()
    effective_use_outer_scheduler = (
        use_outer_scheduler and not proposal_review_mode
    )
    effective_num_agents = (
        1 if proposal_review_mode
        else (NUM_AGENTS if num_agents is None else num_agents)
    )
    effective_parallel_mode = resolve_effective_parallel_mode(
        parallel_mode,
        proposal_review_mode,
        effective_num_agents,
    )
    if runtime is None:
        runtime = build_policy_stack_runtime(
            effective_use_outer_scheduler,
            effective_num_agents,
            allow_silent=allow_silent,
            parallel_mode=effective_parallel_mode,
        )

    agents = runtime["agents"]
    outer_cfr = runtime["outer_cfr"]
    middle_cfr = runtime["middle_cfr"]
    resolved_num_agents = runtime["num_agents"]
    resolved_allow_silent = runtime.get("allow_silent", allow_silent)
    resolved_parallel_mode = runtime.get("parallel_mode", effective_parallel_mode)
    parallel_controller = runtime.get("parallel_controller")
    if force_use_average_strategy is None:
        use_average_strategy = not update_params
    else:
        use_average_strategy = bool(force_use_average_strategy)
    num_rounds = resolve_num_rounds(update_params)
    expand_all_train_branches = bool(
        update_params and THREE_LAYER_EXPAND_ALL_TRAIN_BRANCHES
    )
    checkpoint_options = checkpoint_options or {}
    checkpoint_every_samples = int(checkpoint_options.get("every_samples", 0) or 0)
    checkpoint_dir = checkpoint_options.get("checkpoint_dir") or CHECKPOINT_DIR
    resume_checkpoint_path = checkpoint_options.get("resume_path")

    buffers = init_three_layer_round_buffers(num_rounds)
    total_samples = len(data)
    start_sample_offset = 0
    if update_params and resume_checkpoint_path:
        checkpoint_resume_state = load_policy_stack_checkpoint(
            runtime,
            resume_checkpoint_path,
            expected_total_samples=total_samples,
            expected_num_rounds=num_rounds,
        )
        buffers = checkpoint_resume_state["buffers"]
        start_sample_offset = checkpoint_resume_state["sample_index_completed"]

    if start_sample_offset >= total_samples:
        print(
            f"[checkpoint] {exp_name} 已在 checkpoint 中完成全部 {total_samples} 条样本，直接跳过训练循环。",
            flush=True,
        )

    for idx, item in enumerate(
        tqdm(
            data[start_sample_offset:],
            desc=f"{exp_name} 按样本运行",
            initial=start_sample_offset,
            total=total_samples,
        ),
        start=start_sample_offset + 1,
    ):
        q, gt = item["question"], item["ground_truth"]
        active_branches = [init_three_layer_sample_state()]
        representative_state = active_branches[0]

        for rnd in range(1, num_rounds + 1):
            round_start_time = time.perf_counter()
            next_branches = []
            representative_meta = None
            round_records = []

            for branch_idx, branch_state in enumerate(active_branches):
                pre_history = branch_state["history"]
                pre_incumbent_text = branch_state["incumbent_text"]
                pre_incumbent_pred = branch_state["incumbent_pred"]
                pre_pending_candidate_text = branch_state.get("pending_candidate_text")
                pre_pending_candidate_pred = branch_state.get("pending_candidate_pred")
                pre_pending_vote_score = branch_state.get("pending_vote_score", 0)

                if proposal_review_mode:
                    phase, phase_score = None, 0.0
                else:
                    phase, phase_score = compute_phase(
                        branch_state["prev_phase_value"],
                        rnd,
                    )

                if effective_use_outer_scheduler:
                    outer_state, allowed_agents, outer_strategy, selected_agent = choose_outer_agent(
                        outer_cfr,
                        phase,
                        num_agents=resolved_num_agents,
                        use_average_strategy=use_average_strategy,
                        incumbent_pred=pre_incumbent_pred,
                        rnd=rnd,
                        total_rounds=num_rounds,
                        pending_candidate_text=pre_pending_candidate_text,
                        pending_vote_score=pre_pending_vote_score,
                    )
                else:
                    outer_state = None
                    allowed_agents = list(range(resolved_num_agents))
                    selected_agent = (rnd - 1) % max(resolved_num_agents, 1)
                    outer_strategy = np.zeros(resolved_num_agents, dtype=np.float64)
                    outer_strategy[selected_agent] = 1.0

                middle_state, allowed_actions, middle_strategy, act = choose_middle_action(
                    middle_cfr,
                    phase,
                    rnd,
                    selected_agent,
                    pre_incumbent_pred,
                    allow_silent=resolved_allow_silent,
                    use_average_strategy=use_average_strategy,
                    total_rounds=num_rounds,
                    pending_candidate_text=pre_pending_candidate_text,
                    pending_vote_score=pre_pending_vote_score,
                )

                round_result = run_three_layer_realized_action(
                    agents,
                    q,
                    gt,
                    phase,
                    rnd,
                    selected_agent,
                    act,
                    pre_history,
                    pre_incumbent_text,
                    pre_incumbent_pred,
                    pre_pending_candidate_text,
                    pre_pending_candidate_pred,
                    pre_pending_vote_score,
                    use_candidate_batch=update_params,
                    total_rounds=num_rounds,
                    controller_selector=middle_cfr,
                    use_average_strategy=use_average_strategy,
                    estimate_realized_value=(update_params or not proposal_review_mode),
                )
                if branch_idx == 0:
                    maybe_print_three_layer_stage(idx, rnd, "realized_action", round_start_time)
                if update_params:
                    maybe_dump_policy_stack_candidates(
                        exp_name=exp_name,
                        sample_idx=idx,
                        branch_idx=branch_idx,
                        question=q,
                        gt=gt,
                        rnd=rnd,
                        act=act,
                        pre_incumbent_pred=pre_incumbent_pred,
                        pre_pending_candidate_text=pre_pending_candidate_text,
                        pre_pending_candidate_pred=pre_pending_candidate_pred,
                        pre_pending_vote_score=pre_pending_vote_score,
                        round_result=round_result,
                    )

                if update_params:
                    regret_act = int(round_result.get("regret_action", act))
                    strategy_sum_override = None
                    if (
                        act == MIDDLE_ACTION_ANSWER
                        and regret_act == MIDDLE_ACTION_COMMENT
                        and round_result.get("same_answer_refresh_as_keep")
                    ):
                        strategy_sum_override = np.zeros(CFR_NUM_ACTIONS, dtype=np.float64)
                        strategy_sum_override[MIDDLE_ACTION_COMMENT] = 1.0
                    regret_context = prepare_three_layer_regret_context(
                        outer_cfr,
                        middle_cfr,
                        agents,
                        q,
                        gt,
                        rnd,
                        phase,
                        selected_agent,
                        regret_act,
                        outer_state,
                        allowed_agents,
                        middle_state,
                        allowed_actions,
                        middle_strategy,
                        pre_history,
                        pre_incumbent_text,
                        pre_incumbent_pred,
                        pre_pending_candidate_text,
                        pre_pending_candidate_pred,
                        pre_pending_vote_score,
                        resolved_allow_silent,
                        known_action=regret_act,
                        known_value=round_result["realized_value"],
                        known_silent_baseline=round_result.get("silent_baseline"),
                        parallel_controller=parallel_controller,
                        total_rounds=num_rounds,
                    )
                    if THREE_LAYER_REGRET_UPDATE_MODE == "selected_only":
                        realized_values_for_regret = [round_result["realized_value"]]
                    else:
                        realized_values_for_regret = round_result["candidate_middle_values"]
                    for realized_value in realized_values_for_regret:
                        apply_prepared_three_layer_regret_update(
                            regret_context,
                            realized_value,
                            update_regrets=not (
                                proposal_review_mode
                                and PROPOSAL_REVIEW_DISABLE_CONTROLLER_REGRET
                            ),
                            strategy_sum_override=strategy_sum_override,
                        )
                    if branch_idx == 0:
                        maybe_print_three_layer_stage(idx, rnd, "middle_regret", round_start_time)
                        if effective_use_outer_scheduler:
                            maybe_print_three_layer_stage(idx, rnd, "outer_regret", round_start_time)

                    if not round_result.get("gspo_update_applied", False):
                        apply_three_layer_policy_updates(
                            round_result["gspo_update_payload"],
                        )
                    if branch_idx == 0:
                        maybe_print_three_layer_stage(idx, rnd, "policy_update", round_start_time)

                branch_children = materialize_round_children(
                    branch_state,
                    round_result,
                    gt,
                    rnd,
                    expand_all=expand_all_train_branches,
                )
                next_branches.extend(child["state"] for child in branch_children)
                for child in branch_children:
                    round_records.append({
                        "final_text": child["state"]["final_text"],
                        "final_pred": child["state"]["final_pred"],
                        "best_correct": child["state"]["best_correct"],
                        "realized_value": child["outcome"]["realized_value"],
                        "outer_strategy": np.array(outer_strategy, copy=True),
                        "middle_strategy": np.array(middle_strategy, copy=True),
                        "phase": phase,
                        "transition": child["transition"],
                    })

                if representative_meta is None and branch_children:
                    representative_meta = {
                        "state": branch_children[0]["state"],
                        "transition": branch_children[0]["transition"],
                        "phase": phase,
                        "phase_score": phase_score,
                        "selected_agent": selected_agent,
                        "act": act,
                        "pre_incumbent_pred": pre_incumbent_pred,
                        "pre_pending_candidate_text": pre_pending_candidate_text,
                        "outer_strategy": outer_strategy,
                        "middle_strategy": middle_strategy,
                        "current_pred": branch_children[0]["outcome"]["current_pred"],
                        "realized_value": branch_children[0]["outcome"]["realized_value"],
                    }

            active_branches = next_branches
            representative_state = representative_meta["state"]
            for round_record in round_records:
                record_three_layer_round(
                    buffers,
                    rnd,
                    gt,
                    round_record["final_text"],
                    round_record["final_pred"],
                    round_record["best_correct"],
                    round_record["realized_value"],
                    outer_cfr,
                    middle_cfr,
                    round_record["outer_strategy"],
                    round_record["middle_strategy"],
                    round_record["phase"],
                    round_record["transition"],
                )
            maybe_print_three_layer_round_debug(
                idx,
                rnd,
                representative_meta["phase"],
                representative_meta["phase_score"],
                representative_meta["selected_agent"],
                representative_meta["act"],
                representative_meta["pre_incumbent_pred"],
                representative_meta["pre_pending_candidate_text"],
                representative_meta["current_pred"],
                representative_state["final_pred"],
                gt,
                representative_meta["realized_value"],
                representative_meta["outer_strategy"],
                representative_meta["middle_strategy"],
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
            representative_state["final_pred"],
            representative_state["first_improve_round"],
            representative_state["first_degrade_round"],
            representative_state["first_stalled_wrong_round"],
        )

        if update_params and checkpoint_every_samples > 0 and idx % checkpoint_every_samples == 0:
            save_policy_stack_checkpoint(
                runtime,
                buffers,
                sample_index_completed=idx,
                total_samples=total_samples,
                num_rounds=num_rounds,
                exp_name=exp_name,
                checkpoint_dir=checkpoint_dir,
            )

    if update_params and checkpoint_every_samples > 0 and total_samples > 0:
        last_completed = start_sample_offset if start_sample_offset >= total_samples else total_samples
        if last_completed > 0:
            save_policy_stack_checkpoint(
                runtime,
                buffers,
                sample_index_completed=last_completed,
                total_samples=total_samples,
                num_rounds=num_rounds,
                exp_name=exp_name,
                checkpoint_dir=checkpoint_dir,
            )

    finalize_policy_stack_experiment(
        buffers,
        exp_name,
        log_fn,
        log_metrics=log_metrics,
    )

    if teardown:
        agents, outer_cfr, middle_cfr, parallel_controller = teardown_three_layer_runtime(
            agents,
            outer_cfr,
            middle_cfr,
            parallel_controller=parallel_controller,
        )
        runtime = None
    else:
        runtime = {
            "agents": agents,
            "outer_cfr": outer_cfr,
            "middle_cfr": middle_cfr,
            "use_outer_scheduler": effective_use_outer_scheduler,
            "num_agents": resolved_num_agents,
            "allow_silent": resolved_allow_silent,
            "parallel_mode": resolved_parallel_mode,
            "parallel_controller": parallel_controller,
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
    checkpoint_options=None,
    force_use_average_strategy=None,
):
    parallel_mode = runtime.get("parallel_mode", "serial") if runtime is not None else resolve_parallel_mode()
    return run_policy_stack_experiment(
        data,
        runtime=runtime,
        use_outer_scheduler=False,
        num_agents=NUM_AGENTS,
        allow_silent=True,
        parallel_mode=parallel_mode,
        update_params=update_params,
        log_metrics=log_metrics,
        exp_name=exp_name,
        log_fn=log_middle_layer,
        teardown=teardown,
        checkpoint_options=checkpoint_options,
        force_use_average_strategy=force_use_average_strategy,
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
    checkpoint_options=None,
    force_use_average_strategy=None,
):
    parallel_mode = runtime.get("parallel_mode", "serial") if runtime is not None else resolve_parallel_mode()
    return run_policy_stack_experiment(
        data,
        runtime=runtime,
        use_outer_scheduler=False,
        num_agents=NUM_AGENTS,
        allow_silent=False,
        parallel_mode=parallel_mode,
        update_params=update_params,
        log_metrics=log_metrics,
        exp_name=exp_name,
        log_fn=log_middle_layer_no_silent,
        teardown=teardown,
        checkpoint_options=checkpoint_options,
        force_use_average_strategy=force_use_average_strategy,
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
    checkpoint_options=None,
    force_use_average_strategy=None,
):
    parallel_mode = runtime.get("parallel_mode", "serial") if runtime is not None else resolve_parallel_mode()
    return run_policy_stack_experiment(
        data,
        runtime=runtime,
        use_outer_scheduler=True,
        num_agents=NUM_AGENTS,
        allow_silent=True,
        parallel_mode=parallel_mode,
        update_params=update_params,
        log_metrics=log_metrics,
        exp_name=exp_name,
        log_fn=log_three_layer,
        teardown=teardown,
        checkpoint_options=checkpoint_options,
        force_use_average_strategy=force_use_average_strategy,
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
    checkpoint_options=None,
):
    parallel_mode = runtime.get("parallel_mode", "serial") if runtime is not None else resolve_parallel_mode()
    return run_policy_stack_experiment(
        data,
        runtime=runtime,
        use_outer_scheduler=True,
        num_agents=NUM_AGENTS,
        allow_silent=False,
        parallel_mode=parallel_mode,
        update_params=update_params,
        log_metrics=log_metrics,
        exp_name=exp_name,
        log_fn=log_three_layer_no_silent,
        teardown=teardown,
        checkpoint_options=checkpoint_options,
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
