import argparse
import csv
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"


def _optional_sample_count_env(name):
    value = os.environ.get(name)
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"", "none", "all", "full", "0"}:
        return None
    return int(value)


TRAIN_SAMPLE_COUNT = _optional_sample_count_env("PRIORITY_DIAG_TRAIN_SAMPLES")
VAL_SAMPLE_COUNT = _optional_sample_count_env("PRIORITY_DIAG_VAL_SAMPLES")
TEST_SAMPLE_COUNT = _optional_sample_count_env("PRIORITY_DIAG_TEST_SAMPLES")
CHECKPOINT_EVERY_SAMPLES = int(
    os.environ.get("PRIORITY_DIAG_CHECKPOINT_EVERY_SAMPLES", "10") or "10"
)
DETERMINISTIC_EVAL_OVERRIDES = {
    "THREE_LAYER_EVAL_DETERMINISTIC_ACTIONS": True,
    "GSPO_EVAL_DO_SAMPLE": False,
}
INNER_DETERMINISTIC_EVAL_OVERRIDES = {
    "GSPO_EVAL_DO_SAMPLE": False,
}
CONTROLLER_STOCHASTIC_EVAL_OVERRIDES = {
    "THREE_LAYER_EVAL_DETERMINISTIC_ACTIONS": False,
}
CONTROLLER_GREEDY_EVAL_OVERRIDES = {
    "THREE_LAYER_EVAL_DETERMINISTIC_ACTIONS": True,
}


def build_experiment_specs():
    return [
        {
            "name": "01_baseline_current",
            "family": "baseline",
            "kind": "train_eval",
            "description": "当前全量三层策略，作为所有后续对照的基线。",
        },
        {
            "name": "02_deterministic_eval",
            "family": "deterministic_eval",
            "kind": "eval_only",
            "baseline": "01_baseline_current",
            "description": "加载基线 checkpoint，仅把评估改成确定性动作/文本。",
            "module_overrides": {
                "THREE_LAYER_EVAL_DETERMINISTIC_ACTIONS": True,
                "GSPO_EVAL_DO_SAMPLE": False,
            },
        },
        {
            "name": "03_traj_selected_only",
            "family": "trajectory_consistency",
            "kind": "train_eval",
            "description": "真实轨迹只走 selected candidate，regret 也只用 selected value。",
            "module_overrides": {
                "THREE_LAYER_REGRET_UPDATE_MODE": "selected_only",
            },
        },
        {
            "name": "04_traj_reward_best",
            "family": "trajectory_consistency",
            "kind": "train_eval",
            "description": "真实轨迹沿 reward-best candidate 继续，regret 只用 selected value。",
            "module_overrides": {
                "THREE_LAYER_REGRET_UPDATE_MODE": "selected_only",
                "THREE_LAYER_REALIZED_BRANCH_SELECTION": "reward_best",
            },
        },
        {
            "name": "05_traj_expand_all",
            "family": "trajectory_consistency",
            "kind": "train_eval",
            "description": "训练时展开所有分支，检查当前线性近似是否是主要问题。",
            "module_overrides": {
                "THREE_LAYER_EXPAND_ALL_TRAIN_BRANCHES": True,
            },
        },
        {
            "name": "06_num_agents_1",
            "family": "num_agents_control",
            "kind": "train_eval",
            "description": "把同构双 agent 退化成单 agent，对比是否只是数据被切碎。",
            "module_overrides": {
                "NUM_AGENTS": 1,
            },
        },
        {
            "name": "07_prompt_context_relaxed",
            "family": "prompt_context",
            "kind": "train_eval",
            "description": "放松 incumbent 锚定，并把上下文改成 recent window。",
            "module_overrides": {
                "STRUCTURED_CONTEXT_MODE": "recent_window",
                "STRUCTURED_CONTEXT_MAX_TURNS": 6,
            },
            "use_relaxed_prompts": True,
        },
        {
            "name": "08_incumbent_guard",
            "family": "incumbent_guard",
            "kind": "train_eval",
            "description": "answer 不可解析时保留旧 incumbent，检查 final 掉点是否来自后续覆盖。",
            "module_overrides": {
                "THREE_LAYER_KEEP_INCUMBENT_ON_MISSING_ANSWER": True,
            },
        },
        {
            "name": "09_comment_always_answer",
            "family": "comment_effectiveness",
            "kind": "train_eval",
            "description": "中层每轮强制 answer，作为 no-comment 对照。",
            "module_overrides": {
                "THREE_LAYER_ACTION_OVERRIDE_MODE": "always_answer",
            },
        },
        {
            "name": "10_comment_fixed_schedule",
            "family": "comment_effectiveness",
            "kind": "train_eval",
            "description": "固定 answer/comment 节奏，检查 learned comment 是否真的有贡献。",
            "module_overrides": {
                "THREE_LAYER_ACTION_OVERRIDE_MODE": "fixed_answer_comment",
            },
        },
        {
            "name": "11_comment_freeze_pi0",
            "family": "comment_effectiveness",
            "kind": "train_eval",
            "description": "冻结 pi0 更新，只保留当前 comment 采样行为。",
            "module_overrides": {
                "THREE_LAYER_DISABLE_PI0_UPDATES": True,
            },
        },
        {
            "name": "12_state_expanded",
            "family": "state_representation",
            "kind": "train_eval",
            "description": "给高层 state 加入 has_incumbent 和 round bucket。",
            "module_overrides": {
                "THREE_LAYER_STATE_KEY_MODE": "expanded",
            },
        },
        {
            "name": "13_surrogate_fidelity",
            "family": "surrogate_fidelity",
            "kind": "surrogate_eval",
            "baseline": "01_baseline_current",
            "description": "测量中层 surrogate value 与最终 reward delta 的相关性。",
            "module_overrides": {
                "THREE_LAYER_EVAL_DETERMINISTIC_ACTIONS": True,
                "GSPO_EVAL_DO_SAMPLE": False,
            },
        },
        {
            "name": "14_reward_best_state_expanded",
            "family": "trajectory_consistency",
            "kind": "train_eval",
            "description": "在 reward-best 真实轨迹上扩展高层 state，检查 coarse state 是否吞掉增益。",
            "module_overrides": {
                "THREE_LAYER_REGRET_UPDATE_MODE": "selected_only",
                "THREE_LAYER_REALIZED_BRANCH_SELECTION": "reward_best",
                "THREE_LAYER_STATE_KEY_MODE": "expanded",
            },
        },
        {
            "name": "15_eval_14_reward_best_state_expanded",
            "family": "trajectory_consistency",
            "kind": "eval_only",
            "baseline": "14_reward_best_state_expanded",
            "description": "直接从 14 的当前 checkpoint 做 deterministic 测试，不继续剩余训练。",
            "module_overrides": {
                "THREE_LAYER_REGRET_UPDATE_MODE": "selected_only",
                "THREE_LAYER_REALIZED_BRANCH_SELECTION": "reward_best",
                "THREE_LAYER_STATE_KEY_MODE": "expanded",
            },
        },
        {
            "name": "16_reward_best_state_expanded_stabilize_no_answer",
            "family": "incumbent_guard",
            "kind": "train_eval",
            "description": "在 14 的基础上，当已进入 stabilize 且已有 incumbent 时，直接禁用中层 answer 动作。",
            "env_overrides": {
                "MAS_TRAIN_NUM_ROUNDS": "3",
                "MAS_INFER_NUM_ROUNDS": "3",
            },
            "module_overrides": {
                "THREE_LAYER_REGRET_UPDATE_MODE": "selected_only",
                "THREE_LAYER_REALIZED_BRANCH_SELECTION": "reward_best",
                "THREE_LAYER_STATE_KEY_MODE": "expanded",
                "THREE_LAYER_DISABLE_ANSWER_ON_STABILIZE_WITH_INCUMBENT": True,
            },
        },
        {
            "name": "17_reward_best_state_expanded_keep_gate",
            "family": "incumbent_guard",
            "kind": "train_eval",
            "description": "在 14 的基础上，保留 stabilize 阶段的 answer 采样，但不同于 incumbent 的新答案不覆盖 incumbent。",
            "module_overrides": {
                "THREE_LAYER_REGRET_UPDATE_MODE": "selected_only",
                "THREE_LAYER_REALIZED_BRANCH_SELECTION": "reward_best",
                "THREE_LAYER_STATE_KEY_MODE": "expanded",
                "THREE_LAYER_KEEP_INCUMBENT_ON_STABILIZE_DIFFERENT_ANSWER": True,
            },
        },
        {
            "name": "18_reward_best_proposal_review_light_state",
            "family": "proposal_review",
            "kind": "train_eval",
            "description": "控制层只决定 propose/review；proposer 负责产生 candidate，reviewer 输出 ACCEPT/REJECT；state 只看 incumbent/pending，真实轨迹沿 reward-best 分支。",
            "env_overrides": {
                "MAS_TRAIN_NUM_ROUNDS": "3",
                "MAS_INFER_NUM_ROUNDS": "3",
                "MAS_PARALLEL_MODE": "three_layer_workers",
                "MAS_GSPO_FINETUNE_MODE": "lora",
                "MAS_GSPO_LORA_R": "8",
                "MAS_GSPO_LORA_ALPHA": "16",
                "MAS_GSPO_LORA_DROPOUT": "0.0",
            },
            "module_overrides": {
                "THREE_LAYER_MIDDLE_ACTION_SCHEMA": "proposal_review",
                "THREE_LAYER_REGRET_UPDATE_MODE": "selected_only",
                "THREE_LAYER_REALIZED_BRANCH_SELECTION": "reward_best",
            },
            "run_validation": False,
            "use_proposal_review_prompts": True,
        },
        {
            "name": "19_vote_accum_proposal_review_round_robin",
            "family": "proposal_review",
            "kind": "train_eval",
            "description": "6轮 proposal/review 轮询；review 只投票，pending 用净票累积决定升级/丢弃；controller 只在 proposal stage 学 keep/refresh/challenge/stop。",
            "env_overrides": {
                "MAS_TRAIN_NUM_ROUNDS": "6",
                "MAS_INFER_NUM_ROUNDS": "6",
                "MAS_PARALLEL_MODE": "three_layer_workers",
                "MAS_GSPO_FINETUNE_MODE": "lora",
                "MAS_MAX_NEW_TOKENS": "256",
                "MAS_GSPO_TRAIN_DTYPE": "auto",
                "MAS_GSPO_NUM_GREEDY_CANDIDATES": "1",
                "MAS_GSPO_LORA_R": "8",
                "MAS_GSPO_LORA_ALPHA": "16",
                "MAS_GSPO_LORA_DROPOUT": "0.0",
                "MAS_GSPO_SAMPLE_CANDIDATE_CHUNK_SIZE": "1",
                "MAS_GSPO_UPDATE_CANDIDATE_CHUNK_SIZE": "1",
                "MAS_TRAIN_MAX_NEW_TOKENS": "192",
                "MAS_EVAL_MAX_NEW_TOKENS": "256",
            },
            "module_overrides": {
                "THREE_LAYER_MIDDLE_ACTION_SCHEMA": "proposal_review",
                "THREE_LAYER_REGRET_UPDATE_MODE": "selected_only",
                "THREE_LAYER_REALIZED_BRANCH_SELECTION": "reward_best",
            },
            "run_validation": False,
            "use_proposal_review_prompts": True,
        },
        {
            "name": "25_local_review_reward_intent_parser",
            "family": "proposal_review",
            "kind": "train_eval",
            "description": "reviewer 改成自由短评，由后端硬解析器推断 accept/reject 并直接按判对给 1/0 奖励；proposal stage 的 keep 值改成保持当前答案是否正确的局部奖励，refresh 仍沿用原始 proposal reward。",
            "env_overrides": {
                "MAS_TRAIN_NUM_ROUNDS": "6",
                "MAS_INFER_NUM_ROUNDS": "6",
                "MAS_PARALLEL_MODE": "three_layer_workers",
                "MAS_GSPO_FINETUNE_MODE": "lora",
                "MAS_MAX_NEW_TOKENS": "256",
                "MAS_GSPO_TRAIN_DTYPE": "auto",
                "MAS_GSPO_NUM_GREEDY_CANDIDATES": "1",
                "MAS_GSPO_LORA_R": "8",
                "MAS_GSPO_LORA_ALPHA": "16",
                "MAS_GSPO_LORA_DROPOUT": "0.0",
                "MAS_GSPO_SAMPLE_CANDIDATE_CHUNK_SIZE": "1",
                "MAS_GSPO_UPDATE_CANDIDATE_CHUNK_SIZE": "1",
                "MAS_TRAIN_MAX_NEW_TOKENS": "256",
                "MAS_EVAL_MAX_NEW_TOKENS": "256",
                "MAS_PROPOSAL_COMPLETION_MAX_NEW_TOKENS": "32",
                "MAS_REVIEW_COMPLETION_MAX_NEW_TOKENS": "16",
                "MAS_PROPOSAL_REFRESH_SAME_PRED_PENALTY": "0.0",
            },
            "module_overrides": {
                "THREE_LAYER_MIDDLE_ACTION_SCHEMA": "proposal_review",
                "THREE_LAYER_REGRET_UPDATE_MODE": "selected_only",
                "THREE_LAYER_REALIZED_BRANCH_SELECTION": "reward_best",
                "PROPOSAL_REVIEW_BOOTSTRAP_ACCEPT_SCORE": 2,
                "PROPOSAL_REVIEW_BOOTSTRAP_DROP_SCORE": -1,
                "PROPOSAL_REVIEW_REPLACE_ACCEPT_SCORE": 2,
                "PROPOSAL_REVIEW_DROP_SCORE": -1,
            },
            "run_validation": False,
            "use_proposal_review_prompts": True,
        },
        {
            "name": "28_single_pending_neg1_replace_round5",
            "family": "proposal_review",
            "kind": "train_eval",
            "description": "复制 27，仅改两点：不再区分 pending/incumbent，只保留单个 pending；只有当 pending 净票数 <= -1 时，refresh 的新答案才允许替代当前 pending。训练和测试轮数均为 5。",
            "env_overrides": {
                "MAS_TRAIN_NUM_ROUNDS": "5",
                "MAS_INFER_NUM_ROUNDS": "5",
                "MAS_PARALLEL_MODE": "three_layer_workers",
                "MAS_GSPO_FINETUNE_MODE": "lora",
                "MAS_MAX_NEW_TOKENS": "256",
                "MAS_GSPO_TRAIN_DTYPE": "auto",
                "MAS_GSPO_NUM_GREEDY_CANDIDATES": "1",
                "MAS_GSPO_LORA_R": "8",
                "MAS_GSPO_LORA_ALPHA": "16",
                "MAS_GSPO_LORA_DROPOUT": "0.0",
                "MAS_GSPO_SAMPLE_CANDIDATE_CHUNK_SIZE": "1",
                "MAS_GSPO_UPDATE_CANDIDATE_CHUNK_SIZE": "1",
                "MAS_TRAIN_MAX_NEW_TOKENS": "256",
                "MAS_EVAL_MAX_NEW_TOKENS": "256",
                "MAS_PROPOSAL_COMPLETION_MAX_NEW_TOKENS": "32",
                "MAS_REVIEW_COMPLETION_MAX_NEW_TOKENS": "16",
                "MAS_PROPOSAL_REFRESH_SAME_PRED_PENALTY": "0.0",
            },
            "module_overrides": {
                "THREE_LAYER_MIDDLE_ACTION_SCHEMA": "proposal_review",
                "THREE_LAYER_REGRET_UPDATE_MODE": "selected_only",
                "THREE_LAYER_REALIZED_BRANCH_SELECTION": "first",
                "PROPOSAL_REVIEW_BOOTSTRAP_ACCEPT_SCORE": 2,
                "PROPOSAL_REVIEW_BOOTSTRAP_DROP_SCORE": -1,
                "PROPOSAL_REVIEW_REPLACE_ACCEPT_SCORE": 2,
                "PROPOSAL_REVIEW_DROP_SCORE": -1,
            },
            "run_validation": False,
            "use_proposal_review_prompts": True,
        },
    ]


def spec_by_name(name):
    for spec in build_experiment_specs():
        if spec["name"] == name:
            return spec
    raise KeyError(f"unknown experiment: {name}")


def read_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def last_row(rows):
    return rows[-1] if rows else {}


def rankdata(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[cursor][1]:
            end += 1
        avg_rank = (cursor + end) / 2.0 + 1.0
        for pos in range(cursor, end + 1):
            ranks[indexed[pos][0]] = avg_rank
        cursor = end + 1
    return ranks


def pearson_corr(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    diff_x = [x - mean_x for x in xs]
    diff_y = [y - mean_y for y in ys]
    denom_x = sum(value * value for value in diff_x) ** 0.5
    denom_y = sum(value * value for value in diff_y) ** 0.5
    if denom_x <= 1e-12 or denom_y <= 1e-12:
        return None
    numer = sum(dx * dy for dx, dy in zip(diff_x, diff_y))
    return float(numer / (denom_x * denom_y))


def spearman_corr(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson_corr(rankdata(xs), rankdata(ys))


def build_child_env(root_dir: Path, spec_name: str):
    spec = spec_by_name(spec_name)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["MAS_GLOBAL_SEED"] = env.get("MAS_GLOBAL_SEED", "42")
    env["MAS_THREE_LAYER_DEBUG_PRINT"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Muxi 上 comment 估值的 batch generate 峰值较高，默认拆成单条 micro-batch。
    env.setdefault("MAS_GSPO_SAMPLE_TEXT_BATCH_SIZE", "1")
    env["PRIORITY_DIAG_ROOT_DIR"] = str(root_dir)
    env["PRIORITY_DIAG_SPEC_NAME"] = spec_name
    for name, value in spec.get("env_overrides", {}).items():
        env[name] = str(value)
    return env


def summarize_train_eval(exp_dir: Path, spec, evaluations=None):
    rows = read_csv_rows(exp_dir / "logs" / "three_layer.csv")
    checkpoint_dir = exp_dir / "checkpoints"
    latest_pointer = checkpoint_dir / "latest_checkpoint.txt"
    primary_eval = None
    if evaluations:
        primary_eval = evaluations.get("controller_stochastic") or next(
            iter(evaluations.values()),
            None,
        )
    summary = {
        "name": spec["name"],
        "family": spec["family"],
        "kind": spec["kind"],
        "description": spec["description"],
        "main_evaluation": "controller_stochastic" if evaluations else None,
        "metric_rows": (
            int(primary_eval.get("metric_rows", 0))
            if primary_eval is not None else len(rows)
        ),
        "final_row": (
            primary_eval.get("final_row", {})
            if primary_eval is not None else last_row(rows)
        ),
        "evaluations": evaluations,
        "checkpoint_dir": str(checkpoint_dir),
        "latest_checkpoint": latest_pointer.read_text(encoding="utf-8").strip()
        if latest_pointer.exists() else None,
    }
    summary_path = exp_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def summarize_buffers(run_all, buffers):
    rows = []
    best_accs = run_all.compute_best_so_far_accuracies(buffers["best_round_hits"])
    num_rounds = len(buffers["round_preds"])
    total_reward = 0.0
    for rnd in range(1, num_rounds + 1):
        acc = run_all.compute_accuracy(
            buffers["round_preds"][rnd - 1],
            buffers["round_gts"][rnd - 1],
        )
        best_acc = best_accs[rnd - 1]
        avg_reward = (
            sum(buffers["round_rewards"][rnd - 1]) / len(buffers["round_rewards"][rnd - 1])
            if buffers["round_rewards"][rnd - 1] else 0.0
        )
        total_reward += avg_reward
        avg_regret = (
            sum(buffers["round_regrets"][rnd - 1]) / len(buffers["round_regrets"][rnd - 1])
            if buffers["round_regrets"][rnd - 1] else 0.0
        )
        outer_probs = (
            np.mean(buffers["round_outer_probs"][rnd - 1], axis=0).tolist()
            if buffers["round_outer_probs"][rnd - 1] else [1.0]
        )
        controller_probs = (
            np.mean(buffers["round_middle_probs"][rnd - 1], axis=0).tolist()
            if buffers["round_middle_probs"][rnd - 1] else [0.0, 0.0, 0.0]
        )
        search_rate = (
            sum(buffers["round_search_flags"][rnd - 1]) / len(buffers["round_search_flags"][rnd - 1])
            if buffers["round_search_flags"][rnd - 1] else 0.0
        )
        stabilize_rate = (
            sum(buffers["round_stabilize_flags"][rnd - 1]) / len(buffers["round_stabilize_flags"][rnd - 1])
            if buffers["round_stabilize_flags"][rnd - 1] else 0.0
        )
        improve_rate = (
            sum(buffers["round_improve_flags"][rnd - 1]) / len(buffers["round_improve_flags"][rnd - 1])
            if buffers["round_improve_flags"][rnd - 1] else 0.0
        )
        degrade_rate = (
            sum(buffers["round_degrade_flags"][rnd - 1]) / len(buffers["round_degrade_flags"][rnd - 1])
            if buffers["round_degrade_flags"][rnd - 1] else 0.0
        )
        stalled_wrong_rate = (
            sum(buffers["round_stalled_wrong_flags"][rnd - 1]) / len(buffers["round_stalled_wrong_flags"][rnd - 1])
            if buffers["round_stalled_wrong_flags"][rnd - 1] else 0.0
        )
        preserved_correct_rate = (
            sum(buffers["round_preserved_correct_flags"][rnd - 1]) / len(buffers["round_preserved_correct_flags"][rnd - 1])
            if buffers["round_preserved_correct_flags"][rnd - 1] else 0.0
        )
        rows.append({
            "round": float(rnd),
            "accuracy": float(acc),
            "best_so_far_accuracy": float(best_acc),
            "total_reward": float(total_reward),
            "cfr_regret": float(avg_regret),
            "gspo_agent0_prob": float(outer_probs[0]) if len(outer_probs) > 0 else 0.0,
            "gspo_agent1_prob": float(outer_probs[1]) if len(outer_probs) > 1 else 0.0,
            "cfr_silent_prob": float(controller_probs[0]) if len(controller_probs) > 0 else 0.0,
            "cfr_comment_prob": float(controller_probs[1]) if len(controller_probs) > 1 else 0.0,
            "cfr_answer_prob": float(controller_probs[2]) if len(controller_probs) > 2 else 0.0,
            "search_rate": float(search_rate),
            "stabilize_rate": float(stabilize_rate),
            "improve_rate": float(improve_rate),
            "degrade_rate": float(degrade_rate),
            "stalled_wrong_rate": float(stalled_wrong_rate),
            "preserved_correct_rate": float(preserved_correct_rate),
        })
    return {
        "metric_rows": len(rows),
        "rows": rows,
        "final_row": rows[-1] if rows else {},
    }


def set_global_seed(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_overrides(modules, overrides):
    if not overrides:
        return
    for name, value in overrides.items():
        for module in modules:
            if hasattr(module, name):
                setattr(module, name, value)


def enable_deterministic_eval(modules):
    apply_overrides(modules, DETERMINISTIC_EVAL_OVERRIDES)


def set_generation_token_limit(modules, max_new_tokens):
    apply_overrides(modules, {
        "MAX_NEW_TOKENS": int(max_new_tokens),
    })


PARALLEL_WORKER_OVERRIDE_KEYS = [
    "MAX_NEW_TOKENS",
    "CHAT_SYSTEM_PROMPT",
    "THREE_LAYER_MIDDLE_ACTION_SCHEMA",
    "PROPOSAL_COMPLETION_MAX_NEW_TOKENS",
    "REVIEW_COMPLETION_MAX_NEW_TOKENS",
    "GSPO_EVAL_DO_SAMPLE",
    "TEMPERATURE",
    "GSPO_NUM_CANDIDATES",
    "GSPO_NUM_GREEDY_CANDIDATES",
    "PI0_PROMPT",
    "PI1_PROMPT",
]


def collect_parallel_worker_overrides(modules):
    overrides = {}
    for name in PARALLEL_WORKER_OVERRIDE_KEYS:
        for module in modules:
            if hasattr(module, name):
                overrides[name] = getattr(module, name)
                break
    return overrides


def sync_parallel_worker_overrides(runtime, modules):
    if runtime is None:
        return
    if not isinstance(runtime, dict):
        return
    controller = runtime.get("parallel_controller")
    if controller is None or not hasattr(controller, "apply_module_overrides"):
        return
    controller.apply_module_overrides(
        collect_parallel_worker_overrides(modules)
    )


def set_eval_mode(modules, controller_deterministic, inner_deterministic=True):
    controller_overrides = (
        CONTROLLER_GREEDY_EVAL_OVERRIDES
        if controller_deterministic
        else CONTROLLER_STOCHASTIC_EVAL_OVERRIDES
    )
    apply_overrides(modules, controller_overrides)
    if inner_deterministic:
        apply_overrides(modules, INNER_DETERMINISTIC_EVAL_OVERRIDES)


def run_dual_controller_eval(
    config,
    gspo_verl,
    model_loader,
    run_all,
    runtime,
    test_data,
    exp_name,
    torch,
    np,
):
    modules = [config, gspo_verl, model_loader, run_all]
    set_generation_token_limit(modules, config.EVAL_MAX_NEW_TOKENS)
    eval_modes = [
        ("controller_stochastic", False),
        ("controller_greedy", True),
    ]
    summaries = {}
    runtime_handle = runtime

    for idx, (label, controller_deterministic) in enumerate(eval_modes):
        set_eval_mode(
            modules,
            controller_deterministic=controller_deterministic,
            inner_deterministic=True,
        )
        sync_parallel_worker_overrides(runtime_handle, modules)
        set_global_seed(config.GLOBAL_SEED, torch, np)
        runtime_handle, buffers = run_all.run_three_layer(
            test_data,
            runtime=runtime_handle,
            update_params=False,
            log_metrics=False,
            exp_name=f"{exp_name}[{label}]",
            teardown=(idx == len(eval_modes) - 1),
        )
        summaries[label] = summarize_buffers(run_all, buffers)
        summaries[label]["controller_deterministic"] = bool(controller_deterministic)
        summaries[label]["inner_deterministic"] = True

    return summaries


def configure_child_environment(exp_dir: Path):
    os.environ["MAS_LOG_DIR"] = str(exp_dir / "logs")
    os.environ["MAS_PLOT_DIR"] = str(exp_dir / "plots")
    os.environ["MAS_CHECKPOINT_DIR"] = str(exp_dir / "checkpoints")
    os.environ["MAS_CHECKPOINT_EVERY_SAMPLES"] = str(CHECKPOINT_EVERY_SAMPLES)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "plots").mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)


def resolve_resume_path(checkpoint_dir: Path):
    latest_pointer = checkpoint_dir / "latest_checkpoint.txt"
    if latest_pointer.exists():
        return str(checkpoint_dir)
    return None


def load_exact_splits(data_loader):
    splits = data_loader.load_gsm8k_splits()

    def maybe_limit(items, limit):
        return list(items) if limit is None else list(items)[:limit]

    train_data = maybe_limit(splits["train"], TRAIN_SAMPLE_COUNT)
    val_data = maybe_limit(splits["val"], VAL_SAMPLE_COUNT)
    test_data = maybe_limit(splits["test"], TEST_SAMPLE_COUNT)
    return train_data, val_data, test_data


def build_runtime_from_checkpoint(run_all, checkpoint_dir: Path):
    proposal_review_mode = bool(run_all.is_proposal_review_schema())
    runtime = run_all.build_policy_stack_runtime(
        use_outer_scheduler=(not proposal_review_mode),
        num_agents=(1 if proposal_review_mode else run_all.NUM_AGENTS),
        allow_silent=True,
        parallel_mode=run_all.resolve_effective_parallel_mode(
            run_all.resolve_parallel_mode(),
            proposal_review_mode,
            1 if proposal_review_mode else run_all.NUM_AGENTS,
        ),
    )
    run_all.load_policy_stack_checkpoint(
        runtime,
        str(checkpoint_dir),
        expected_total_samples=TRAIN_SAMPLE_COUNT,
        expected_num_rounds=run_all.TRAIN_NUM_ROUNDS,
    )
    return runtime


def run_surrogate_fidelity(run_all, runtime, data):
    records = []
    num_rounds = run_all.resolve_num_rounds(update_params=False)
    proposal_review_mode = bool(run_all.is_proposal_review_schema())

    for item in data:
        q = item["question"]
        gt = item["ground_truth"]
        branch_state = run_all.init_three_layer_sample_state()
        per_round_records = []

        for rnd in range(1, num_rounds + 1):
            pre_history = branch_state["history"]
            pre_incumbent_text = branch_state["incumbent_text"]
            pre_incumbent_pred = branch_state["incumbent_pred"]
            pre_pending_candidate_text = branch_state.get("pending_candidate_text")
            pre_pending_candidate_pred = branch_state.get("pending_candidate_pred")
            pre_pending_vote_score = branch_state.get("pending_vote_score", 0)

            if proposal_review_mode:
                phase, _ = None, 0.0
                outer_state = None
                allowed_agents = [0]
                selected_role_bundle_idx = 0
            else:
                phase, _ = run_all.compute_phase(branch_state["prev_phase_value"], rnd)
                outer_state, allowed_agents, _, selected_role_bundle_idx = run_all.choose_outer_agent(
                    runtime["outer_cfr"],
                    phase,
                    num_agents=runtime["num_agents"],
                    use_average_strategy=True,
                    incumbent_pred=pre_incumbent_pred,
                    rnd=rnd,
                    total_rounds=num_rounds,
                    pending_candidate_text=pre_pending_candidate_text,
                )
            controller_state, allowed_actions, controller_strategy, act = run_all.choose_middle_action(
                runtime["middle_cfr"],
                phase,
                rnd,
                selected_role_bundle_idx,
                pre_incumbent_pred,
                allow_silent=runtime["allow_silent"],
                use_average_strategy=True,
                total_rounds=num_rounds,
                pending_candidate_text=pre_pending_candidate_text,
                pending_vote_score=pre_pending_vote_score,
            )
            estimated_values, _ = run_all.estimate_middle_action_values(
                q,
                gt,
                runtime["agents"],
                phase,
                selected_role_bundle_idx,
                rnd,
                pre_history,
                pre_incumbent_text,
                pre_incumbent_pred,
                pending_candidate_text=pre_pending_candidate_text,
                pending_candidate_pred=pre_pending_candidate_pred,
                allow_silent=runtime["allow_silent"],
                total_rounds=num_rounds,
                pending_vote_score=pre_pending_vote_score,
                controller_selector=runtime["middle_cfr"],
                controller_use_average_strategy=True,
            )
            round_result = run_all.run_three_layer_realized_action(
                runtime["agents"],
                q,
                gt,
                phase,
                rnd,
                selected_role_bundle_idx,
                act,
                pre_history,
                pre_incumbent_text,
                pre_incumbent_pred,
                pre_pending_candidate_text,
                pre_pending_candidate_pred,
                pre_pending_vote_score,
                use_candidate_batch=False,
                total_rounds=num_rounds,
                controller_selector=runtime["middle_cfr"],
                use_average_strategy=True,
            )
            selected_outcome = round_result["candidate_outcomes"][0]
            child_state, _ = run_all.clone_branch_state_with_outcome(
                branch_state,
                selected_outcome,
                gt,
                rnd,
            )
            per_round_records.append({
                "round": rnd,
                "estimated_value": float(estimated_values[act]),
                "pre_reward": float(
                    run_all.reward_from_pred(
                        run_all.resolve_final_pred(
                            pre_incumbent_pred,
                            pre_pending_candidate_pred,
                        ),
                        gt,
                    )
                ),
                "chosen_action": int(act),
                "outer_state": outer_state,
                "allowed_agents": allowed_agents,
                "controller_state": controller_state,
                "allowed_actions": allowed_actions,
                "controller_strategy": [float(value) for value in controller_strategy],
            })
            branch_state = child_state

        final_reward = float(
            run_all.reward_from_pred(
                run_all.resolve_final_pred(
                    branch_state["incumbent_pred"],
                    branch_state.get("pending_candidate_pred"),
                ),
                gt,
            )
        )
        for record in per_round_records:
            records.append({
                "round": record["round"],
                "estimated_value": record["estimated_value"],
                "final_delta": final_reward - record["pre_reward"],
                "chosen_action": record["chosen_action"],
            })

    estimated = [record["estimated_value"] for record in records]
    final_delta = [record["final_delta"] for record in records]
    summary = {
        "num_records": len(records),
        "pearson_estimated_vs_final_delta": pearson_corr(estimated, final_delta),
        "spearman_estimated_vs_final_delta": spearman_corr(estimated, final_delta),
        "mean_estimated_value": (sum(estimated) / len(estimated)) if estimated else None,
        "mean_final_delta": (sum(final_delta) / len(final_delta)) if final_delta else None,
        "records_preview": records[:20],
    }
    return summary


def run_child(spec_name: str, root_dir: Path, require_cuda: bool):
    spec = spec_by_name(spec_name)
    exp_dir = root_dir / spec["name"]
    configure_child_environment(exp_dir)
    for name, value in spec.get("env_overrides", {}).items():
        os.environ[name] = str(value)

    import numpy as np
    import torch

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("当前环境不可见 CUDA，无法启动 100/100 的 Phi-3 诊断实验。")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    import src28.config as config
    import src28.data_loader as data_loader
    import src28.gspo_verl as gspo_verl
    import src28.model_loader as model_loader
    import run_all_28 as run_all

    common_overrides = {
        "GLOBAL_SEED": int(os.environ.get("MAS_GLOBAL_SEED", "42") or "42"),
        "THREE_LAYER_DEBUG_PRINT": False,
    }
    apply_overrides(
        [config, data_loader, gspo_verl, model_loader, run_all],
        common_overrides,
    )
    apply_overrides(
        [config, data_loader, gspo_verl, model_loader, run_all],
        spec.get("module_overrides"),
    )

    if spec.get("use_relaxed_prompts"):
        prompt_overrides = {
            "PI0_PROMPT": config.PI0_PROMPT_RELAXED,
            "PI1_PROMPT": config.PI1_PROMPT_RELAXED,
        }
        apply_overrides([config, gspo_verl, run_all], prompt_overrides)
    if spec.get("use_proposal_review_prompts"):
        prompt_overrides = {
            "PI0_PROMPT": config.PI0_PROMPT_PROPOSAL_REVIEW,
            "PI1_PROMPT": config.PI1_PROMPT_PROPOSAL_REVIEW,
        }
        apply_overrides([config, gspo_verl, run_all], prompt_overrides)

    set_global_seed(config.GLOBAL_SEED, torch, np)

    train_data, val_data, test_data = load_exact_splits(data_loader)
    run_all.init_logs()

    metadata = {
        "name": spec["name"],
        "family": spec["family"],
        "kind": spec["kind"],
        "description": spec["description"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "global_seed": config.GLOBAL_SEED,
        "num_agents": run_all.NUM_AGENTS,
        "train_num_rounds": run_all.TRAIN_NUM_ROUNDS,
        "infer_num_rounds": run_all.INFER_NUM_ROUNDS,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "env_overrides": spec.get("env_overrides", {}),
        "module_overrides": spec.get("module_overrides", {}),
        "deterministic_eval_overrides": DETERMINISTIC_EVAL_OVERRIDES,
        "use_relaxed_prompts": bool(spec.get("use_relaxed_prompts", False)),
        "use_proposal_review_prompts": bool(spec.get("use_proposal_review_prompts", False)),
        "run_validation": bool(spec.get("run_validation", True)),
        "gspo_finetune_mode": config.GSPO_FINETUNE_MODE,
        "gspo_lora_r": getattr(config, "GSPO_LORA_R", None),
        "gspo_lora_alpha": getattr(config, "GSPO_LORA_ALPHA", None),
        "gspo_lora_dropout": getattr(config, "GSPO_LORA_DROPOUT", None),
        "train_max_new_tokens": getattr(
            config,
            "TRAIN_MAX_NEW_TOKENS",
            getattr(config, "MAX_NEW_TOKENS", None),
        ),
        "eval_max_new_tokens": getattr(
            config,
            "EVAL_MAX_NEW_TOKENS",
            getattr(config, "MAX_NEW_TOKENS", None),
        ),
        "parallel_mode": run_all.resolve_effective_parallel_mode(
            run_all.resolve_parallel_mode(),
            getattr(config, "THREE_LAYER_MIDDLE_ACTION_SCHEMA", "legacy") == "proposal_review",
            1 if getattr(config, "THREE_LAYER_MIDDLE_ACTION_SCHEMA", "legacy") == "proposal_review"
            else run_all.NUM_AGENTS,
        ),
        "effective_use_outer_scheduler": not (
            getattr(config, "THREE_LAYER_MIDDLE_ACTION_SCHEMA", "legacy") == "proposal_review"
        ),
        "effective_num_agents": (
            1 if getattr(config, "THREE_LAYER_MIDDLE_ACTION_SCHEMA", "legacy") == "proposal_review"
            else run_all.NUM_AGENTS
        ),
        "effective_runtime_architecture": (
            "controller + proposer + reviewer"
            if getattr(config, "THREE_LAYER_MIDDLE_ACTION_SCHEMA", "legacy") == "proposal_review"
            else "outer_scheduler + middle_selector + policy_stack"
        ),
        "evaluation_protocols": [
            {
                "name": "controller_stochastic",
                "controller_deterministic": False,
                "inner_deterministic": True,
                "controller_policy_source": "average_strategy_sample",
            },
            {
                "name": "controller_greedy",
                "controller_deterministic": True,
                "inner_deterministic": True,
                "controller_policy_source": "average_strategy_argmax",
            },
        ],
        "require_cuda": require_cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    (exp_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if spec["kind"] == "train_eval":
        generation_modules = [config, gspo_verl, model_loader, run_all]
        set_generation_token_limit(generation_modules, config.TRAIN_MAX_NEW_TOKENS)
        checkpoint_dir = exp_dir / "checkpoints"
        checkpoint_options = {
            "checkpoint_dir": str(checkpoint_dir),
            "every_samples": CHECKPOINT_EVERY_SAMPLES,
            "resume_path": resolve_resume_path(checkpoint_dir),
        }
        runtime, _ = run_all.run_three_layer(
            train_data,
            update_params=True,
            log_metrics=False,
            exp_name=f"{spec['name']} 训练",
            teardown=False,
            checkpoint_options=checkpoint_options,
        )
        if spec.get("run_validation", True):
            set_generation_token_limit(generation_modules, config.EVAL_MAX_NEW_TOKENS)
            set_eval_mode(
                [config, gspo_verl, run_all],
                controller_deterministic=False,
                inner_deterministic=True,
            )
            sync_parallel_worker_overrides(runtime, generation_modules)
            set_global_seed(config.GLOBAL_SEED, torch, np)
            run_all.run_three_layer(
                val_data,
                runtime=runtime,
                update_params=False,
                log_metrics=False,
                exp_name=f"{spec['name']} 验证",
                teardown=False,
            )
        eval_summaries = run_dual_controller_eval(
            config,
            gspo_verl,
            model_loader,
            run_all,
            runtime,
            test_data,
            spec["name"],
            torch,
            np,
        )
        return summarize_train_eval(exp_dir, spec, evaluations=eval_summaries)

    baseline_name = spec.get("baseline")
    if not baseline_name:
        raise ValueError(f"{spec['name']} 缺少 baseline 依赖。")
    baseline_checkpoint_dir = root_dir / baseline_name / "checkpoints"
    runtime = build_runtime_from_checkpoint(run_all, baseline_checkpoint_dir)

    if spec["kind"] == "eval_only":
        eval_summaries = run_dual_controller_eval(
            config,
            gspo_verl,
            model_loader,
            run_all,
            runtime,
            test_data,
            spec["name"],
            torch,
            np,
        )
        return summarize_train_eval(exp_dir, spec, evaluations=eval_summaries)

    if spec["kind"] == "surrogate_eval":
        set_generation_token_limit(
            [config, gspo_verl, model_loader, run_all],
            config.EVAL_MAX_NEW_TOKENS,
        )
        sync_parallel_worker_overrides(
            runtime,
            [config, gspo_verl, model_loader, run_all],
        )
        summary = {
            "name": spec["name"],
            "family": spec["family"],
            "kind": spec["kind"],
            "description": spec["description"],
            "baseline_checkpoint_dir": str(baseline_checkpoint_dir),
            "diagnostics": run_surrogate_fidelity(run_all, runtime, test_data),
        }
        run_all.teardown_three_layer_runtime(
            runtime["agents"],
            runtime["outer_cfr"],
            runtime["middle_cfr"],
            parallel_controller=runtime.get("parallel_controller"),
        )
        summary_path = exp_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    raise ValueError(f"unsupported experiment kind: {spec['kind']}")


def summarize_overall(root_dir: Path, selected_names):
    grouped = {}
    experiments = []
    for name in selected_names:
        spec = spec_by_name(name)
        summary_path = root_dir / name / "summary.json"
        summary = None
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        experiments.append({
            "name": name,
            "family": spec["family"],
            "summary_path": str(summary_path),
            "summary": summary,
        })
        grouped.setdefault(spec["family"], []).append({
            "name": name,
            "summary": summary,
        })

    overall = {
        "run_dir": str(root_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "train_sample_count": TRAIN_SAMPLE_COUNT,
        "test_sample_count": TEST_SAMPLE_COUNT,
        "val_sample_count": VAL_SAMPLE_COUNT,
        "checkpoint_every_samples": CHECKPOINT_EVERY_SAMPLES,
        "experiments": experiments,
        "families": grouped,
    }
    overall_path = root_dir / "priority_diagnostics_summary.json"
    overall_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    return overall


def parse_args():
    parser = argparse.ArgumentParser(description="Run prioritized three-layer diagnostic experiments.")
    parser.add_argument(
        "--child",
        default=None,
        help="Internal use: run one child experiment by name.",
    )
    parser.add_argument(
        "--root-dir",
        default=None,
        help="Existing root dir to use. Parent mode defaults to a timestamped run directory.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional subset of experiment names to run.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow child jobs to run without CUDA. This is usually impractical for Phi-3.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_names = args.experiments or ["28_single_pending_neg1_replace_round5"]

    if args.child:
        summary = run_child(
            spec_name=args.child,
            root_dir=Path(args.root_dir or os.environ["PRIORITY_DIAG_ROOT_DIR"]).resolve(),
            require_cuda=not args.allow_cpu,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    run_root = Path(args.root_dir).resolve() if args.root_dir else (
        RUNS_DIR / f"priority_diagnostics_{datetime.now().strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    progress = []
    overall_returncode = 0
    for name in selected_names:
        env = build_child_env(run_root, name)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            name,
            "--root-dir",
            str(run_root),
        ]
        if args.allow_cpu:
            cmd.append("--allow-cpu")

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
        )
        progress.append({
            "name": name,
            "returncode": result.returncode,
        })
        progress_path = run_root / "progress.json"
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
        if result.returncode != 0:
            overall_returncode = result.returncode
            break

    overall = summarize_overall(run_root, selected_names)
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    if overall_returncode != 0:
        raise SystemExit(overall_returncode)


if __name__ == "__main__":
    main()
