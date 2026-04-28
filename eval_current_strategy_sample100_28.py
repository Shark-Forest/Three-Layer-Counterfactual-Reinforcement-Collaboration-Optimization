import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import run_priority_diagnostics_28 as diag


SPEC_NAME = "28_single_pending_neg1_replace_round5"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate experiment 28 sample_000100 using current strategy."
    )
    parser.add_argument(
        "--exp-dir",
        required=True,
        help="Path to the experiment 28 directory.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="sample_000100",
        help="Checkpoint directory name under exp-dir/checkpoints.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Optional cap on samples from the selected split. Default evaluates the full split.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Optional output subdirectory name under the experiment directory.",
    )
    return parser.parse_args()


def run_current_strategy_dual_eval(
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
    diag.set_generation_token_limit(modules, config.EVAL_MAX_NEW_TOKENS)
    eval_modes = [
        ("controller_stochastic", False, "current_strategy_sample"),
        ("controller_greedy", True, "current_strategy_argmax"),
    ]
    summaries = {}
    runtime_handle = runtime

    for idx, (label, controller_deterministic, policy_source) in enumerate(eval_modes):
        diag.set_eval_mode(
            modules,
            controller_deterministic=controller_deterministic,
            inner_deterministic=True,
        )
        diag.sync_parallel_worker_overrides(runtime_handle, modules)
        diag.set_global_seed(config.GLOBAL_SEED, torch, np)
        runtime_handle, buffers = run_all.run_three_layer(
            test_data,
            runtime=runtime_handle,
            update_params=False,
            log_metrics=False,
            exp_name=f"{exp_name}[{label}]",
            teardown=(idx == len(eval_modes) - 1),
            force_use_average_strategy=False,
        )
        summaries[label] = diag.summarize_buffers(run_all, buffers)
        summaries[label]["controller_deterministic"] = bool(controller_deterministic)
        summaries[label]["inner_deterministic"] = True
        summaries[label]["controller_policy_source"] = policy_source

    return summaries


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"experiment dir not found: {exp_dir}")

    checkpoint_dir = exp_dir / "checkpoints" / args.checkpoint_name
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {checkpoint_dir}")

    output_subdir = (
        args.output_subdir
        if args.output_subdir
        else f"current_strategy_eval_{args.checkpoint_name}"
    )
    eval_dir = exp_dir / output_subdir
    diag.configure_child_environment(eval_dir)

    spec = diag.spec_by_name(SPEC_NAME)
    for name, value in spec.get("env_overrides", {}).items():
        os.environ[name] = str(value)

    if str(diag.PROJECT_ROOT) not in sys.path:
        sys.path.append(str(diag.PROJECT_ROOT))

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current environment.")

    import src28.config as config
    import src28.data_loader as data_loader
    import src28.gspo_verl as gspo_verl
    import src28.model_loader as model_loader
    import run_all_28 as run_all

    common_overrides = {
        "GLOBAL_SEED": int(os.environ.get("MAS_GLOBAL_SEED", "42") or "42"),
        "THREE_LAYER_DEBUG_PRINT": False,
    }
    diag.apply_overrides(
        [config, data_loader, gspo_verl, model_loader, run_all],
        common_overrides,
    )
    diag.apply_overrides(
        [config, data_loader, gspo_verl, model_loader, run_all],
        spec.get("module_overrides"),
    )
    if spec.get("use_proposal_review_prompts"):
        prompt_overrides = {
            "PI0_PROMPT": config.PI0_PROMPT_PROPOSAL_REVIEW,
            "PI1_PROMPT": config.PI1_PROMPT_PROPOSAL_REVIEW,
        }
        diag.apply_overrides([config, gspo_verl, run_all], prompt_overrides)

    diag.set_global_seed(config.GLOBAL_SEED, torch, np)
    train_data, _, test_data = diag.load_exact_splits(data_loader)
    eval_data = train_data if args.split == "train" else test_data
    if args.test_samples is not None:
        eval_data = eval_data[: args.test_samples]
    run_all.init_logs()

    runtime = diag.build_runtime_from_checkpoint(run_all, checkpoint_dir)
    summaries = run_current_strategy_dual_eval(
        config,
        gspo_verl,
        model_loader,
        run_all,
        runtime,
        eval_data,
            f"{SPEC_NAME}_current_strategy_{args.split}",
        torch,
        np,
    )

    primary_eval = summaries.get("controller_stochastic") or next(
        iter(summaries.values()),
        None,
    )
    summary = {
        "name": f"{SPEC_NAME}_current_strategy_{args.checkpoint_name}",
        "family": spec["family"],
        "kind": "eval_only_current_strategy",
        "description": "从指定 checkpoint 恢复当前在线策略，对所选 GSM8K split 分别做 controller-stochastic 和 controller-greedy 评测。",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_experiment": SPEC_NAME,
        "checkpoint_dir": str(checkpoint_dir),
        "split": args.split,
        "eval_size": len(eval_data),
        "main_evaluation": "controller_stochastic",
        "metric_rows": int(primary_eval.get("metric_rows", 0)) if primary_eval else 0,
        "final_row": primary_eval.get("final_row", {}) if primary_eval else {},
        "strategy_source": "current_strategy",
        "evaluations": summaries,
    }
    summary_path = eval_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
