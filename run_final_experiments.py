import argparse
import contextlib
import copy
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
CHECKPOINT_EVERY_SAMPLES = int(
    os.environ.get("FINAL_CHECKPOINT_EVERY_SAMPLES", "50") or "50"
)

BASE_ENV = {
    "MAS_TRAIN_NUM_ROUNDS": "5",
    "MAS_INFER_NUM_ROUNDS": "5",
    "MAS_PARALLEL_MODE": "three_layer_workers",
    "MAS_THREE_LAYER_MIDDLE_ACTION_SCHEMA": "proposal_review",
    "MAS_GSPO_FINETUNE_MODE": "lora",
    "MAS_GSPO_TRAIN_DTYPE": "auto",
    "MAS_GSPO_NUM_GREEDY_CANDIDATES": "1",
    "MAS_GSPO_LORA_R": "8",
    "MAS_GSPO_LORA_ALPHA": "16",
    "MAS_GSPO_LORA_DROPOUT": "0.0",
    "MAS_GSPO_SAMPLE_CANDIDATE_CHUNK_SIZE": "1",
    "MAS_GSPO_UPDATE_CANDIDATE_CHUNK_SIZE": "1",
    "MAS_MAX_NEW_TOKENS": "256",
    "MAS_TRAIN_MAX_NEW_TOKENS": "256",
    "MAS_EVAL_MAX_NEW_TOKENS": "256",
    "MAS_PROPOSAL_COMPLETION_MAX_NEW_TOKENS": "32",
    "MAS_REVIEW_COMPLETION_MAX_NEW_TOKENS": "16",
    "MAS_PROPOSAL_REFRESH_SAME_PRED_PENALTY": "0.0",
}

BASE_MODULE_OVERRIDES = {
    "THREE_LAYER_MIDDLE_ACTION_SCHEMA": "proposal_review",
    "THREE_LAYER_REGRET_UPDATE_MODE": "selected_only",
    "THREE_LAYER_REALIZED_BRANCH_SELECTION": "first",
    "PROPOSAL_REVIEW_BOOTSTRAP_ACCEPT_SCORE": 2,
    "PROPOSAL_REVIEW_BOOTSTRAP_DROP_SCORE": -1,
    "PROPOSAL_REVIEW_REPLACE_ACCEPT_SCORE": 2,
    "PROPOSAL_REVIEW_DROP_SCORE": -1,
}


def build_experiment_specs():
    return [
        {
            "name": "01_main_exp30",
            "paper_label": "Main",
            "family": "main",
            "description": "实验30逻辑的主实验：proposal-review，5轮，learned keep/refresh controller，refresh 可直接取代当前 pending。",
        },
        {
            "name": "02_fixed_keep_refresh_controller",
            "paper_label": "Fixed controller",
            "family": "controller",
            "description": "去掉 learned controller，proposal stage 在 keep/refresh 之间按固定周期切换。",
            "module_overrides": {
                "PROPOSAL_REVIEW_CONTROLLER_OVERRIDE_MODE": "fixed_keep_refresh",
                "PROPOSAL_REVIEW_DISABLE_CONTROLLER_REGRET": True,
            },
        },
        {
            "name": "03_always_refresh_controller",
            "paper_label": "Always refresh",
            "family": "controller",
            "description": "controller 不学习；只要允许 refresh 就固定 refresh，否则执行唯一可行动作。",
            "module_overrides": {
                "PROPOSAL_REVIEW_CONTROLLER_OVERRIDE_MODE": "always_refresh",
                "PROPOSAL_REVIEW_DISABLE_CONTROLLER_REGRET": True,
            },
        },
        {
            "name": "04_always_keep_controller",
            "paper_label": "Always keep",
            "family": "controller",
            "description": "controller 不学习；有 pending 时固定 keep。",
            "module_overrides": {
                "PROPOSAL_REVIEW_CONTROLLER_OVERRIDE_MODE": "always_keep",
                "PROPOSAL_REVIEW_DISABLE_CONTROLLER_REGRET": True,
            },
        },
        {
            "name": "05_no_counterfactual_controller_values",
            "paper_label": "No counterfactual values",
            "family": "controller_learning",
            "description": "保留 controller regret matching，但未选动作的反事实 value 置零，只用真实动作 value 更新。",
            "module_overrides": {
                "PROPOSAL_REVIEW_DISABLE_COUNTERFACTUAL_VALUES": True,
            },
        },
        {
            "name": "06_no_controller_regret_update",
            "paper_label": "Frozen controller",
            "family": "controller_learning",
            "description": "完全冻结 controller 的 CFR regret/average strategy 更新，只保留初始化策略。",
            "module_overrides": {
                "PROPOSAL_REVIEW_DISABLE_CONTROLLER_REGRET": True,
            },
        },
        {
            "name": "07_vote_threshold_state_machine",
            "paper_label": "Vote threshold state machine",
            "family": "vote_refresh",
            "description": "启用投票阈值状态机：净票>=+2 升级 incumbent，净票<=-1 丢弃 pending。",
            "module_overrides": {
                "PROPOSAL_REVIEW_APPLY_VOTE_THRESHOLDS": True,
            },
        },
        {
            "name": "08_no_vote_updates",
            "paper_label": "No vote accumulation",
            "family": "vote_refresh",
            "description": "reviewer 仍生成评审文本并训练，但评审 verdict 不改变 pending_vote_score。",
            "module_overrides": {
                "PROPOSAL_REVIEW_DISABLE_VOTE_UPDATES": True,
            },
        },
        {
            "name": "09_refresh_only_negative_pending",
            "paper_label": "Protected nonnegative pending",
            "family": "vote_refresh",
            "description": "恢复保护机制：pending 净票>=0 时，即使 controller 选择 refresh，新答案也不能取代旧 pending。",
            "module_overrides": {
                "PROPOSAL_REVIEW_REFRESH_REQUIRES_NEGATIVE_PENDING": True,
            },
        },
        {
            "name": "10_no_refresh_action",
            "paper_label": "No refresh",
            "family": "vote_refresh",
            "description": "禁用 refresh，pending 存在时 proposal stage 只能 keep。",
            "module_overrides": {
                "PROPOSAL_REVIEW_DISABLE_REFRESH": True,
                "PROPOSAL_REVIEW_DISABLE_CONTROLLER_REGRET": True,
            },
        },
        {
            "name": "11_reviewer_frozen",
            "paper_label": "Reviewer frozen",
            "family": "inner_learning",
            "description": "冻结 reviewer/pi0 的 GSPO 更新，只训练 proposer/pi1 和 controller。",
            "module_overrides": {
                "THREE_LAYER_DISABLE_PI0_UPDATES": True,
            },
        },
        {
            "name": "12_proposer_frozen",
            "paper_label": "Proposer frozen",
            "family": "inner_learning",
            "description": "冻结 proposer/pi1 的 GSPO 更新，只训练 reviewer/pi0 和 controller。",
            "module_overrides": {
                "THREE_LAYER_DISABLE_PI1_UPDATES": True,
            },
        },
        {
            "name": "13_no_inner_gspo_updates",
            "paper_label": "No GSPO",
            "family": "inner_learning",
            "description": "冻结 proposer 和 reviewer 的全部 GSPO 更新，只学习 controller。",
            "module_overrides": {
                "THREE_LAYER_DISABLE_ALL_GSPO_UPDATES": True,
            },
        },
        {
            "name": "14_no_review_feedback_context",
            "paper_label": "No review feedback in proposer context",
            "family": "protocol",
            "description": "保留 pending solution，但 refresh/proposal context 不加入 latest review feedback。",
            "module_overrides": {
                "PROPOSAL_REVIEW_DISABLE_REVIEW_FEEDBACK_CONTEXT": True,
            },
        },
        {
            "name": "15_unstructured_legacy_prompts",
            "paper_label": "Legacy prompts",
            "family": "protocol",
            "description": "仍使用 proposal-review 状态机，但 proposer 使用旧式 answer prompt，去掉结构化 proposal-review proposer prompt。",
            "module_overrides": {
                "PROPOSAL_REVIEW_USE_LEGACY_PROMPTS": True,
            },
            "use_legacy_prompt_override": True,
        },
    ]


def spec_by_name(name):
    for spec in build_experiment_specs():
        if spec["name"] == name:
            return spec
    raise KeyError(f"unknown experiment: {name}")


def configure_child_environment(exp_dir):
    os.environ["MAS_LOG_DIR"] = str(exp_dir / "logs")
    os.environ["MAS_PLOT_DIR"] = str(exp_dir / "plots")
    os.environ["MAS_CHECKPOINT_DIR"] = str(exp_dir / "checkpoints")
    os.environ["MAS_CHECKPOINT_EVERY_SAMPLES"] = str(CHECKPOINT_EVERY_SAMPLES)
    for subdir in ("logs", "plots", "checkpoints"):
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)


def apply_overrides(modules, overrides):
    if not overrides:
        return
    for module in modules:
        for name, value in dict(overrides).items():
            if hasattr(module, name):
                setattr(module, name, copy.deepcopy(value))


def set_global_seed(seed, torch_module, np_module):
    random.seed(seed)
    np_module.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def set_generation_token_limit(modules, max_new_tokens):
    apply_overrides(modules, {"MAX_NEW_TOKENS": int(max_new_tokens)})


def collect_parallel_worker_overrides(modules):
    names = [
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
        "PROPOSAL_REVIEW_CONTROLLER_OVERRIDE_MODE",
        "PROPOSAL_REVIEW_DISABLE_CONTROLLER_REGRET",
        "PROPOSAL_REVIEW_DISABLE_COUNTERFACTUAL_VALUES",
        "PROPOSAL_REVIEW_DISABLE_VOTE_UPDATES",
        "PROPOSAL_REVIEW_APPLY_VOTE_THRESHOLDS",
        "PROPOSAL_REVIEW_REFRESH_REQUIRES_NEGATIVE_PENDING",
        "PROPOSAL_REVIEW_DISABLE_REFRESH",
        "PROPOSAL_REVIEW_DISABLE_REVIEW_FEEDBACK_CONTEXT",
        "PROPOSAL_REVIEW_USE_LEGACY_PROMPTS",
        "THREE_LAYER_DISABLE_PI0_UPDATES",
        "THREE_LAYER_DISABLE_PI1_UPDATES",
        "THREE_LAYER_DISABLE_ALL_GSPO_UPDATES",
    ]
    overrides = {}
    for name in names:
        for module in modules:
            if hasattr(module, name):
                overrides[name] = getattr(module, name)
                break
    return overrides


def sync_parallel_worker_overrides(runtime, modules):
    if not isinstance(runtime, dict):
        return
    controller = runtime.get("parallel_controller")
    if controller is not None and hasattr(controller, "apply_module_overrides"):
        controller.apply_module_overrides(collect_parallel_worker_overrides(modules))


def set_eval_mode(modules, controller_deterministic, inner_deterministic=True):
    apply_overrides(
        modules,
        {"THREE_LAYER_EVAL_DETERMINISTIC_ACTIONS": bool(controller_deterministic)},
    )
    if inner_deterministic:
        apply_overrides(modules, {"GSPO_EVAL_DO_SAMPLE": False})


class UsageCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_calls = 0
        self.llm_api_calls = 0
        self.active_agents = set()
        self.active_policies = set()

    @property
    def tokens(self):
        return int(self.prompt_tokens + self.completion_tokens)

    def snapshot(self):
        return {
            "tokens": self.tokens,
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "llm_calls": int(self.llm_calls),
            "llm_api_calls": int(self.llm_api_calls),
            "active_agents": set(self.active_agents),
            "active_policies": set(self.active_policies),
        }

    def add(self, policy_key, prompt_tokens, completion_tokens, llm_calls, llm_api_calls=1):
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)
        self.llm_calls += int(llm_calls or 0)
        self.llm_api_calls += int(llm_api_calls or 0)
        self.active_policies.add(policy_key)
        self.active_agents.add(policy_key.split(".", 1)[0])

    def record(self):
        return {
            "tokens": self.tokens,
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "llm_calls": int(self.llm_calls),
            "llm_api_calls": int(self.llm_api_calls),
            "active_agents": sorted(self.active_agents),
            "active_agents_count": len(self.active_agents),
            "active_policies": sorted(self.active_policies),
            "active_policies_count": len(self.active_policies),
        }


def counter_delta(before, after):
    return {
        "tokens": int(after["tokens"] - before["tokens"]),
        "prompt_tokens": int(after["prompt_tokens"] - before["prompt_tokens"]),
        "completion_tokens": int(after["completion_tokens"] - before["completion_tokens"]),
        "llm_calls": int(after["llm_calls"] - before["llm_calls"]),
        "llm_api_calls": int(after["llm_api_calls"] - before["llm_api_calls"]),
        "active_agents": sorted(after["active_agents"] - before["active_agents"]),
        "active_agents_count": len(after["active_agents"] - before["active_agents"]),
        "active_policies": sorted(after["active_policies"] - before["active_policies"]),
        "active_policies_count": len(after["active_policies"] - before["active_policies"]),
    }


def count_token_ids(tokenizer, text, add_special_tokens=True):
    if tokenizer is None or text is None:
        return 0
    encoded = tokenizer(str(text), add_special_tokens=add_special_tokens)
    ids = encoded.get("input_ids") if isinstance(encoded, dict) else encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        return len(ids[0])
    return len(ids)


def render_prompt_for_counting(model_loader, tokenizer, prompt):
    try:
        rendered = model_loader.render_prompts_for_generation([prompt], tokenizer)
        return rendered[0] if rendered else prompt
    except Exception:
        return prompt


def format_prompt(prompt_template, query_context):
    if not prompt_template:
        return str(query_context)
    try:
        return prompt_template.format(context=query_context)
    except Exception:
        return f"{prompt_template}\n{query_context}"


class CountingPolicyWrapper:
    def __init__(self, policy, policy_key, default_prompt, counter, tokenizer, model_loader):
        self._policy = policy
        self._policy_key = policy_key
        self._default_prompt = default_prompt
        self._counter = counter
        self._tokenizer = tokenizer
        self._model_loader = model_loader

    def __getattr__(self, name):
        return getattr(self._policy, name)

    def _prompt_tokens(self, query_context, prompt_override):
        prompt = format_prompt(prompt_override or self._default_prompt, query_context)
        rendered = render_prompt_for_counting(self._model_loader, self._tokenizer, prompt)
        return count_token_ids(self._tokenizer, rendered, add_special_tokens=True)

    def _record(self, contexts, outputs, prompt_override=None):
        contexts = list(contexts or [])
        outputs = ["" if value is None else str(value) for value in list(outputs or [])]
        self._counter.add(
            self._policy_key,
            prompt_tokens=sum(self._prompt_tokens(ctx, prompt_override) for ctx in contexts),
            completion_tokens=sum(
                count_token_ids(self._tokenizer, text, add_special_tokens=False)
                for text in outputs
            ),
            llm_calls=max(len(contexts), len(outputs)),
            llm_api_calls=1,
        )

    def act(self, query_context, prompt_override=None):
        result = self._policy.act(query_context, prompt_override=prompt_override)
        texts = [candidate.get("text", "") for candidate in (result or [])]
        self._record([query_context for _ in texts], texts, prompt_override=prompt_override)
        return result

    def sample_candidates(self, query_context, prompt_override=None, selected_idx=0):
        result = self._policy.sample_candidates(
            query_context,
            prompt_override=prompt_override,
            selected_idx=selected_idx,
        )
        candidates = result.get("candidates", []) if isinstance(result, dict) else []
        texts = [candidate.get("text", "") for candidate in candidates]
        self._record([query_context for _ in texts], texts, prompt_override=prompt_override)
        return result

    def sample_text(self, query_context, prompt_override=None):
        result = self._policy.sample_text(query_context, prompt_override=prompt_override)
        self._record([query_context], [result], prompt_override=prompt_override)
        return result

    def sample_text_batch(self, query_contexts, prompt_override=None):
        contexts = list(query_contexts or [])
        result = self._policy.sample_text_batch(contexts, prompt_override=prompt_override)
        self._record(contexts, result, prompt_override=prompt_override)
        return result


def instrument_runtime(runtime, counter, tokenizer, model_loader, config):
    for agent_idx, agent in enumerate(runtime.get("agents", [])):
        for policy_name in ("pi0", "pi1"):
            policy = agent.get(policy_name)
            if policy is None or isinstance(policy, CountingPolicyWrapper):
                continue
            default_prompt = getattr(policy, "prompt_template", None)
            if default_prompt is None:
                default_prompt = getattr(
                    config,
                    "PI0_PROMPT" if policy_name == "pi0" else "PI1_PROMPT",
                    None,
                )
            agent[policy_name] = CountingPolicyWrapper(
                policy,
                f"agent{agent_idx}.{policy_name}",
                default_prompt,
                counter,
                tokenizer,
                model_loader,
            )


def first_item(values):
    return values[0] if values else None


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_correct(run_all, pred, gt):
    return bool(run_all.is_exact_match(pred, gt))


def build_sample_record(run_all, sample_index, item, buffers, usage):
    round_preds = [first_item(values) for values in buffers.get("round_preds", [])]
    round_gts = [first_item(values) for values in buffers.get("round_gts", [])]
    round_texts = [first_item(values) for values in buffers.get("round_texts", [])]
    gt = round_gts[-1] if round_gts else item.get("ground_truth")

    first_round = None
    first_pred = None
    first_text = None
    for rnd, pred in enumerate(round_preds, start=1):
        if pred is not None:
            first_round = rnd
            first_pred = pred
            first_text = round_texts[rnd - 1] if rnd - 1 < len(round_texts) else None
            break

    final_pred = round_preds[-1] if round_preds else None
    final_text = round_texts[-1] if round_texts else None
    first_ok = None if first_pred is None else is_correct(run_all, first_pred, gt)
    final_ok = is_correct(run_all, final_pred, gt)
    correction = first_pred is not None and not first_ok and final_ok
    preservation = first_ok is True and final_ok

    record = {
        "sample_index": int(sample_index),
        "question": item.get("question"),
        "ground_truth": safe_float(gt),
        "round_preds": [safe_float(value) for value in round_preds],
        "round_correct": [is_correct(run_all, pred, gt) for pred in round_preds],
        "first_parseable_round": first_round,
        "first_parseable_pred": safe_float(first_pred),
        "first_parseable_correct": first_ok,
        "first_parseable_text": first_text,
        "final_pred": safe_float(final_pred),
        "final_correct": bool(final_ok),
        "final_text": final_text,
        "correction": bool(correction),
        "preservation": bool(preservation),
    }
    record.update(usage)
    return record


def summarize_eval_records(records, num_rounds):
    size = len(records)
    counts = {
        "final_correct": sum(1 for record in records if record.get("final_correct")),
        "first_parseable": sum(
            1 for record in records
            if record.get("first_parseable_pred") is not None
        ),
        "correction": sum(1 for record in records if record.get("correction")),
        "preservation": sum(1 for record in records if record.get("preservation")),
    }
    round_correct_counts = [0 for _ in range(num_rounds)]
    for record in records:
        for rnd, correct in enumerate(record.get("round_correct", [])[:num_rounds]):
            if correct:
                round_correct_counts[rnd] += 1

    usage_totals = {
        key: sum(int(record.get(key, 0) or 0) for record in records)
        for key in (
            "tokens",
            "prompt_tokens",
            "completion_tokens",
            "llm_calls",
            "llm_api_calls",
        )
    }
    usage_totals["active_agents"] = sum(
        int(record.get("active_agents_count", 0) or 0) for record in records
    )
    usage_totals["active_policies"] = sum(
        int(record.get("active_policies_count", 0) or 0) for record in records
    )

    round_accuracy = [
        float(correct / size) if size else 0.0
        for correct in round_correct_counts
    ]
    round_accuracy_named = {
        f"r{idx + 1}": float(value)
        for idx, value in enumerate(round_accuracy)
    }

    return {
        "eval_size": int(size),
        "round_accuracy": round_accuracy,
        "accuracy_r1_to_r5": round_accuracy_named,
        "accuracy": float(counts["final_correct"] / size) if size else 0.0,
        "correction_rate": float(counts["correction"] / size) if size else 0.0,
        "preservation_rate": float(counts["preservation"] / size) if size else 0.0,
        "first_parseable_rate": float(counts["first_parseable"] / size) if size else 0.0,
        "tokens_per_task": float(usage_totals["tokens"] / size) if size else 0.0,
        "prompt_tokens_per_task": float(usage_totals["prompt_tokens"] / size) if size else 0.0,
        "completion_tokens_per_task": float(usage_totals["completion_tokens"] / size) if size else 0.0,
        "llm_calls_per_task": float(usage_totals["llm_calls"] / size) if size else 0.0,
        "llm_api_calls_per_task": float(usage_totals["llm_api_calls"] / size) if size else 0.0,
        "active_agents_per_task": float(usage_totals["active_agents"] / size) if size else 0.0,
        "active_policies_per_task": float(usage_totals["active_policies"] / size) if size else 0.0,
        "counts": counts,
        "usage_totals": usage_totals,
        "records": records,
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_runtime_from_checkpoint(run_all, checkpoint_dir, train_size):
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
        expected_total_samples=int(train_size),
        expected_num_rounds=run_all.TRAIN_NUM_ROUNDS,
    )
    return runtime


def plot_main_training_curves(exp_dir):
    live_path = exp_dir / "logs" / "live_accuracy.jsonl"
    if not live_path.exists():
        return None
    rows = []
    for line in live_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "训练" in str(row.get("exp_name", "")):
            rows.append(row)
    if not rows:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [row["processed"] for row in rows]
    max_rounds = max(len(row.get("round_accuracy") or []) for row in rows)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    plt.figure(figsize=(8, 5))
    for rnd in range(max_rounds):
        ys = [
            (row.get("round_accuracy") or [None] * max_rounds)[rnd]
            if rnd < len(row.get("round_accuracy") or [])
            else None
            for row in rows
        ]
        plt.plot(xs, ys, label=f"R{rnd + 1}", color=colors[rnd % len(colors)], linewidth=2)
    plt.xlabel("Training samples")
    plt.ylabel("Training accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plot_path = exp_dir / "plots" / "main_train_r1_r5_accuracy.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return str(plot_path)


def run_eval(
    *,
    mode,
    runtime,
    test_data,
    exp_name,
    exp_dir,
    config,
    gspo_verl,
    model_loader,
    run_all,
    counter,
    torch,
    np,
):
    controller_deterministic = mode == "controller_greedy"
    modules = [config, gspo_verl, model_loader, run_all]
    set_generation_token_limit(modules, config.EVAL_MAX_NEW_TOKENS)
    set_eval_mode(modules, controller_deterministic=controller_deterministic)
    sync_parallel_worker_overrides(runtime, modules)
    set_global_seed(config.GLOBAL_SEED, torch, np)

    records = []
    raw_log_path = exp_dir / f"raw_eval_{mode}.log"
    raw_log_path.write_text("", encoding="utf-8")
    with raw_log_path.open("a", encoding="utf-8") as raw_log:
        for sample_index, item in enumerate(test_data, start=1):
            counter.reset()
            with contextlib.redirect_stdout(raw_log), contextlib.redirect_stderr(raw_log):
                runtime, buffers = run_all.run_three_layer(
                    [item],
                    runtime=runtime,
                    update_params=False,
                    log_metrics=False,
                    exp_name=f"{exp_name}[{mode}] sample{sample_index:05d}",
                    teardown=False,
                    force_use_average_strategy=True,
                )
            raw_log.flush()
            record = build_sample_record(
                run_all,
                sample_index,
                item,
                buffers,
                counter.record(),
            )
            records.append(record)
            if sample_index == 1 or sample_index == len(test_data) or sample_index % 25 == 0:
                partial = summarize_eval_records(
                    records,
                    num_rounds=run_all.resolve_num_rounds(update_params=False),
                )
                print(
                    f"[{exp_name}][{mode}] {sample_index}/{len(test_data)} "
                    f"acc={partial['accuracy']:.4f} "
                    f"R={partial['round_accuracy']} "
                    f"corr={partial['correction_rate']:.4f} "
                    f"pres={partial['preservation_rate']:.4f} "
                    f"tokens/task={partial['tokens_per_task']:.1f} "
                    f"llm_calls/task={partial['llm_calls_per_task']:.2f}",
                    flush=True,
                )

    summary = summarize_eval_records(
        records,
        num_rounds=run_all.resolve_num_rounds(update_params=False),
    )
    summary["mode"] = mode
    summary["controller_deterministic"] = bool(controller_deterministic)
    summary["inner_deterministic"] = True
    summary["usage_note"] = "Usage is counted per sample from actual policy generation calls."
    write_json(exp_dir / f"summary_{mode}.json", {k: v for k, v in summary.items() if k != "records"})
    write_jsonl(exp_dir / f"samples_{mode}.jsonl", summary["records"])
    return runtime, summary


def parse_device_groups(raw_value):
    if raw_value is None:
        return []
    groups = []
    for raw_group in str(raw_value).split(";"):
        group = raw_group.strip()
        if group:
            groups.append(group)
    return groups


def parse_experiment_device_map(raw_value):
    if raw_value is None or str(raw_value).strip() == "":
        return {}
    mapping = {}
    for raw_item in str(raw_value).split(";"):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "--experiment-device-map 条目必须是 experiment_name=gpu_ids，"
                f"收到: {item}"
            )
        name, group = item.split("=", 1)
        name = name.strip()
        group = group.strip()
        if not name or not group:
            raise ValueError(
                "--experiment-device-map 条目不能为空，"
                f"收到: {item}"
            )
        mapping[name] = group
    return mapping


def build_child_env(run_root, spec, args, device_group=None, slot_idx=None):
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update({str(k): str(v) for k, v in spec.get("env_overrides", {}).items()})
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["FINAL_RUN_ROOT"] = str(run_root)
    env["MAS_GLOBAL_SEED"] = env.get("MAS_GLOBAL_SEED", "42")
    env["MAS_THREE_LAYER_DEBUG_PRINT"] = "0"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("MAS_GSPO_SAMPLE_TEXT_BATCH_SIZE", "1")
    if args.parallel_mode:
        env["MAS_PARALLEL_MODE"] = str(args.parallel_mode)
    if args.policy_device_map:
        env["MAS_POLICY_DEVICE_MAP"] = str(args.policy_device_map)
    if device_group:
        env["CUDA_VISIBLE_DEVICES"] = str(device_group)
        env["FINAL_EXPERIMENT_DEVICE_GROUP"] = str(device_group)
    elif args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        env["FINAL_EXPERIMENT_DEVICE_GROUP"] = str(args.cuda_visible_devices)
    if slot_idx is not None:
        env["FINAL_EXPERIMENT_SLOT"] = str(slot_idx)
    return env


def run_child(spec_name, root_dir, args):
    spec = spec_by_name(spec_name)
    exp_dir = root_dir / spec["name"]
    configure_child_environment(exp_dir)
    os.environ["MAS_CHECKPOINT_EVERY_SAMPLES"] = str(args.checkpoint_every)
    for name, value in spec.get("env_overrides", {}).items():
        os.environ[name] = str(value)

    import torch

    if not args.allow_cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    import src.config as config
    import src.data_loader as data_loader
    import src.gspo_verl as gspo_verl
    import src.model_loader as model_loader
    import run_all

    common_overrides = {
        "GLOBAL_SEED": int(os.environ.get("MAS_GLOBAL_SEED", "42") or "42"),
        "THREE_LAYER_DEBUG_PRINT": False,
    }
    module_overrides = dict(BASE_MODULE_OVERRIDES)
    module_overrides.update(spec.get("module_overrides", {}))
    modules_all = [config, data_loader, gspo_verl, model_loader, run_all]
    apply_overrides(modules_all, common_overrides)
    apply_overrides(modules_all, module_overrides)
    prompt_overrides = {
        "PI0_PROMPT": config.PI0_PROMPT_PROPOSAL_REVIEW,
        "PI1_PROMPT": (
            config.PI1_PROMPT if spec.get("use_legacy_prompt_override")
            else config.PI1_PROMPT_PROPOSAL_REVIEW
        ),
    }
    apply_overrides([config, gspo_verl, run_all], prompt_overrides)

    set_global_seed(config.GLOBAL_SEED, torch, np)
    splits = data_loader.load_gsm8k_official_splits(
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )
    train_data = splits["train"]
    test_data = splits["test"]
    run_all.init_logs()

    metadata = {
        "name": spec["name"],
        "paper_label": spec.get("paper_label"),
        "family": spec["family"],
        "description": spec["description"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "train_split": "official_gsm8k_train",
        "test_split": "official_gsm8k_test",
        "train_size": len(train_data),
        "test_size": len(test_data),
        "train_limit": args.train_limit,
        "test_limit": args.test_limit,
        "num_rounds": {
            "train": int(config.TRAIN_NUM_ROUNDS),
            "test": int(config.INFER_NUM_ROUNDS),
        },
        "data_protocol": (
            "Use the official GSM8K train split for online training and the "
            "official GSM8K test split for evaluation. No validation split is "
            "carved out in the final suite."
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
        "env_overrides": {**BASE_ENV, **spec.get("env_overrides", {})},
        "module_overrides": module_overrides,
        "parallel": {
            "parallel_mode": os.environ.get("MAS_PARALLEL_MODE"),
            "policy_device_map": os.environ.get("MAS_POLICY_DEVICE_MAP"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_group": os.environ.get("FINAL_EXPERIMENT_DEVICE_GROUP"),
            "slot": os.environ.get("FINAL_EXPERIMENT_SLOT"),
        },
        "checkpoint_every_samples": args.checkpoint_every,
    }
    write_json(exp_dir / "metadata.json", metadata)

    generation_modules = [config, gspo_verl, model_loader, run_all]
    set_generation_token_limit(generation_modules, config.TRAIN_MAX_NEW_TOKENS)
    checkpoint_options = {
        "checkpoint_dir": str(exp_dir / "checkpoints"),
        "every_samples": args.checkpoint_every,
        "resume_path": None,
    }
    runtime, train_buffers = run_all.run_three_layer(
        train_data,
        update_params=True,
        log_metrics=False,
        exp_name=f"{spec['name']} 训练",
        teardown=False,
        checkpoint_options=checkpoint_options,
        force_use_average_strategy=False,
    )
    train_round_accuracy = run_all.compute_round_accuracies(
        train_buffers["round_preds"],
        train_buffers["round_gts"],
    )
    train_plot = plot_main_training_curves(exp_dir) if spec["name"] == "01_main_exp30" else None

    tokenizer = model_loader._ensure_tokenizer()
    counter = UsageCounter()
    instrument_runtime(runtime, counter, tokenizer, model_loader, config)
    evals = {}
    try:
        for mode in ("controller_stochastic", "controller_greedy"):
            runtime, eval_summary = run_eval(
                mode=mode,
                runtime=runtime,
                test_data=test_data,
                exp_name=spec["name"],
                exp_dir=exp_dir,
                config=config,
                gspo_verl=gspo_verl,
                model_loader=model_loader,
                run_all=run_all,
                counter=counter,
                torch=torch,
                np=np,
            )
            evals[mode] = {k: v for k, v in eval_summary.items() if k != "records"}
    finally:
        if runtime is not None:
            run_all.teardown_three_layer_runtime(
                runtime["agents"],
                runtime["outer_cfr"],
                runtime["middle_cfr"],
                parallel_controller=runtime.get("parallel_controller"),
            )

    summary = {
        **metadata,
        "kind": "train_eval_full_gsm8k",
        "train_round_accuracy": [float(value) for value in train_round_accuracy],
        "train_curve_plot": train_plot,
        "evaluations": evals,
        "main_evaluation": "controller_stochastic",
        "main_metrics": evals.get("controller_stochastic", {}),
    }
    write_json(exp_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def summarize_overall(root_dir, selected_names, checkpoint_every):
    experiments = []
    for name in selected_names:
        spec = spec_by_name(name)
        path = root_dir / name / "summary.json"
        experiments.append({
            "name": name,
            "paper_label": spec.get("paper_label"),
            "family": spec["family"],
            "summary_path": str(path),
            "summary": json.loads(path.read_text(encoding="utf-8")) if path.exists() else None,
        })
    overall = {
        "run_dir": str(root_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_every_samples": int(checkpoint_every),
        "experiments": experiments,
    }
    write_json(root_dir / "final_experiments_summary.json", overall)
    return overall


def parse_args():
    parser = argparse.ArgumentParser(description="Run final GSM8K main and ablation experiments.")
    parser.add_argument("--child", default=None, help="Internal: run one experiment.")
    parser.add_argument("--root-dir", default=None, help="Run root. Defaults to timestamped final/runs directory.")
    parser.add_argument("--experiments", nargs="*", default=None, help="Subset of experiment names.")
    parser.add_argument("--train-limit", type=int, default=None, help="Smoke-test cap for official train split.")
    parser.add_argument("--test-limit", type=int, default=None, help="Smoke-test cap for official test split.")
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY_SAMPLES)
    parser.add_argument(
        "--parallel-mode",
        choices=["three_layer_workers", "serial"],
        default=None,
        help=(
            "Internal policy execution mode. Default keeps MAS_PARALLEL_MODE/BASE_ENV, "
            "which is three_layer_workers for this final suite."
        ),
    )
    parser.add_argument(
        "--policy-device-map",
        default=None,
        help=(
            "Explicit MAS_POLICY_DEVICE_MAP, e.g. agent0.pi0:0,agent0.pi1:1. "
            "When omitted, policy workers are assigned round-robin over visible GPUs."
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Set CUDA_VISIBLE_DEVICES for every experiment child, e.g. 0,1.",
    )
    parser.add_argument(
        "--experiment-workers",
        type=int,
        default=1,
        help="Number of experiment child processes to run concurrently. Default: 1.",
    )
    parser.add_argument(
        "--device-groups",
        default=None,
        help=(
            "Semicolon-separated CUDA_VISIBLE_DEVICES groups used for concurrent "
            "experiments, e.g. '0,1;2,3'. Requires --experiment-workers > 1."
        ),
    )
    parser.add_argument(
        "--experiment-device-map",
        default=None,
        help=(
            "Per-experiment CUDA_VISIBLE_DEVICES map, e.g. "
            "'01_main_exp30=0,1;02_fixed_keep_refresh_controller=2,3'. "
            "Overrides --device-groups for listed experiments."
        ),
    )
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    selected_names = args.experiments or [spec["name"] for spec in build_experiment_specs()]
    root_dir = Path(args.root_dir).resolve() if args.root_dir else (
        RUNS_DIR / f"final_{datetime.now().strftime('%Y%m%dT%H%M%SZ')}"
    )
    root_dir.mkdir(parents=True, exist_ok=True)

    if args.child:
        run_child(args.child, root_dir, args)
        return

    experiment_workers = max(1, int(args.experiment_workers or 1))
    device_groups = parse_device_groups(args.device_groups)
    experiment_device_map = parse_experiment_device_map(args.experiment_device_map)
    unknown_mapped = sorted(set(experiment_device_map) - set(selected_names))
    if unknown_mapped:
        raise ValueError(
            "--experiment-device-map 包含未运行的实验名: "
            + ", ".join(unknown_mapped)
        )
    if (
        device_groups
        and not experiment_device_map
        and experiment_workers > len(device_groups)
    ):
        raise ValueError(
            "--experiment-workers 不能大于 --device-groups 中的组数；"
            f"当前 workers={experiment_workers}, groups={len(device_groups)}"
        )

    def build_child_cmd(name):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            name,
            "--root-dir",
            str(root_dir),
            "--checkpoint-every",
            str(args.checkpoint_every),
        ]
        if args.train_limit is not None:
            cmd.extend(["--train-limit", str(args.train_limit)])
        if args.test_limit is not None:
            cmd.extend(["--test-limit", str(args.test_limit)])
        if args.parallel_mode is not None:
            cmd.extend(["--parallel-mode", str(args.parallel_mode)])
        if args.policy_device_map is not None:
            cmd.extend(["--policy-device-map", str(args.policy_device_map)])
        if args.cuda_visible_devices is not None:
            cmd.extend(["--cuda-visible-devices", str(args.cuda_visible_devices)])
        if args.allow_cpu:
            cmd.append("--allow-cpu")
        return cmd

    progress = []
    return_code = 0
    pending = list(selected_names)
    running = []
    next_slot = 0

    while pending or running:
        while pending and len(running) < experiment_workers and return_code == 0:
            name = pending.pop(0)
            group_idx = None
            device_group = None
            if name in experiment_device_map:
                device_group = experiment_device_map[name]
            elif device_groups:
                used_group_indices = {
                    item.get("group_index")
                    for item in running
                    if item.get("group_index") is not None
                }
                free_group_indices = [
                    idx for idx in range(len(device_groups))
                    if idx not in used_group_indices
                ]
                if not free_group_indices:
                    break
                group_idx = free_group_indices[0]
                device_group = device_groups[group_idx]
            child_env = build_child_env(
                root_dir,
                spec_by_name(name),
                args,
                device_group=device_group,
                slot_idx=next_slot,
            )
            proc = subprocess.Popen(
                build_child_cmd(name),
                cwd=str(PROJECT_ROOT),
                env=child_env,
            )
            running.append({
                "name": name,
                "process": proc,
                "device_group": device_group,
                "group_index": group_idx,
                "slot": next_slot,
            })
            next_slot += 1

        if not running:
            break

        time.sleep(5)
        still_running = []
        for item in running:
            proc = item["process"]
            rc = proc.poll()
            if rc is None:
                still_running.append(item)
                continue
            progress.append({
                "name": item["name"],
                "returncode": int(rc),
                "device_group": item.get("device_group"),
                "group_index": item.get("group_index"),
                "slot": item.get("slot"),
            })
            write_json(root_dir / "progress.json", progress)
            if rc != 0 and return_code == 0:
                return_code = int(rc)
        running = still_running

        if return_code != 0 and running:
            for item in running:
                item["process"].terminate()
            for item in running:
                item["process"].wait()
            break

    overall = summarize_overall(
        root_dir,
        [item["name"] for item in progress],
        args.checkpoint_every,
    )
    print(json.dumps(overall, ensure_ascii=False, indent=2), flush=True)
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
