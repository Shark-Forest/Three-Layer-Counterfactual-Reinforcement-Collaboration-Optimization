import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"

EXPERIMENT_TO_FUNC = {
    "single_llm": "run_single_llm",
    "polling_two_llms": "run_polling_two_llms",
    "single_gspo": "run_single_gspo",
    "dual_gspo": "run_dual_gspo",
    "middle_layer": "run_middle_layer",
    "middle_layer_no_silent": "run_middle_layer_no_silent",
    "three_layer": "run_three_layer",
    "three_layer_no_silent": "run_three_layer_no_silent",
}


def read_metric_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def last_row(rows):
    return rows[-1] if rows else {}


def summarize_stage(run_dir: Path):
    logs_dir = run_dir / "logs"
    summary = {
        "run_dir": str(run_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": {},
    }

    for name in EXPERIMENT_TO_FUNC:
        rows = read_metric_csv(logs_dir / f"{name}.csv")
        summary["metrics"][name] = {
            "num_rows": len(rows),
            "last_row": last_row(rows),
        }

    summary_path = run_dir / "stage_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_common_preamble(sample_count: int, num_agents: int, debug_print: bool):
    debug_literal = "True" if debug_print else "False"
    return f"""
import sys
import os
import torch
sys.path.append('{PROJECT_ROOT}')

if str(os.environ.get('MAS_REQUIRE_CUDA', '0')).strip().lower() in ('1', 'true', 'yes', 'y', 'on') and not torch.cuda.is_available():
    raise RuntimeError(
        'MAS_REQUIRE_CUDA=1, 但当前子进程 torch.cuda.is_available()=False。'
        ' 请确认是在 GPU 节点上运行，并检查 CUDA_VISIBLE_DEVICES / 驱动 / torch 安装。'
    )

import src.config as config
import src.gspo_verl as gspo_verl
import src.model_loader as model_loader
import run_all
from src.data_loader import load_gsm8k_splits

config.GSPO_NUM_CANDIDATES = 2
gspo_verl.GSPO_NUM_CANDIDATES = 2
config.GSPO_COMMENT_EVAL_SAMPLES = 1
run_all.GSPO_COMMENT_EVAL_SAMPLES = 1
config.MAX_NEW_TOKENS = 96
model_loader.MAX_NEW_TOKENS = 96
gspo_verl.MAX_NEW_TOKENS = 96
config.NUM_AGENTS = {num_agents}
run_all.NUM_AGENTS = {num_agents}
config.THREE_LAYER_DEBUG_PRINT = {debug_literal}
run_all.THREE_LAYER_DEBUG_PRINT = {debug_literal}

print('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'))
model_loader.print_runtime_device_summary_once()
splits = load_gsm8k_splits(
    train_val_total_limit={sample_count},
    test_limit={sample_count},
)
print('SPLIT_SIZES', {{
    'train': len(splits['train']),
    'val': len(splits['val']),
    'test': len(splits['test']),
}})
"""


def build_init_code(sample_count: int, num_agents: int, debug_print: bool):
    return (
        build_common_preamble(sample_count, num_agents, debug_print)
        + "\nrun_all.init_logs()\nprint('LOGS_INITIALIZED')\n"
    )


def build_experiment_code(sample_count: int, num_agents: int, debug_print: bool, experiment: str):
    func_name = EXPERIMENT_TO_FUNC[experiment]
    if experiment == "single_llm":
        body = "run_all.run_single_llm(splits['test'])"
    elif experiment == "polling_two_llms":
        body = "run_all.run_polling_two_llms(splits['test'])"
    elif experiment == "single_gspo":
        body = """
agent, _ = run_all.run_single_gspo(
    splits['train'],
    update_params=True,
    log_metrics=False,
    exp_name='单LLM GSPO 训练',
    teardown=False,
)
run_all.run_single_gspo(
    splits['val'],
    agent=agent,
    update_params=False,
    log_metrics=False,
    exp_name='单LLM GSPO 验证',
    teardown=False,
)
run_all.run_single_gspo(
    splits['test'],
    agent=agent,
    update_params=False,
    log_metrics=True,
    exp_name='单LLM GSPO',
    teardown=True,
)
"""
    elif experiment == "dual_gspo":
        body = """
(solver, commenter), _ = run_all.run_dual_gspo(
    splits['train'],
    update_params=True,
    log_metrics=False,
    exp_name='双LLM GSPO轮询 训练',
    teardown=False,
)
run_all.run_dual_gspo(
    splits['val'],
    solver=solver,
    commenter=commenter,
    update_params=False,
    log_metrics=False,
    exp_name='双LLM GSPO轮询 验证',
    teardown=False,
)
run_all.run_dual_gspo(
    splits['test'],
    solver=solver,
    commenter=commenter,
    update_params=False,
    log_metrics=True,
    exp_name='双LLM GSPO轮询',
    teardown=True,
)
"""
    elif experiment == "middle_layer":
        body = """
runtime, _ = run_all.run_middle_layer(
    splits['train'],
    update_params=True,
    log_metrics=False,
    exp_name='中间层策略+GSPO 训练',
    teardown=False,
)
run_all.run_middle_layer(
    splits['val'],
    runtime=runtime,
    update_params=False,
    log_metrics=False,
    exp_name='中间层策略+GSPO 验证',
    teardown=False,
)
run_all.run_middle_layer(
    splits['test'],
    runtime=runtime,
    update_params=False,
    log_metrics=True,
    exp_name='中间层策略+GSPO',
    teardown=True,
)
"""
    elif experiment == "middle_layer_no_silent":
        body = """
runtime, _ = run_all.run_middle_layer_no_silent(
    splits['train'],
    update_params=True,
    log_metrics=False,
    exp_name='中间层策略+GSPO（无沉默） 训练',
    teardown=False,
)
run_all.run_middle_layer_no_silent(
    splits['val'],
    runtime=runtime,
    update_params=False,
    log_metrics=False,
    exp_name='中间层策略+GSPO（无沉默） 验证',
    teardown=False,
)
run_all.run_middle_layer_no_silent(
    splits['test'],
    runtime=runtime,
    update_params=False,
    log_metrics=True,
    exp_name='中间层策略+GSPO（无沉默）',
    teardown=True,
)
"""
    else:
        body = """
runtime, _ = run_all.run_three_layer(
    splits['train'],
    update_params=True,
    log_metrics=False,
    exp_name='全量策略 训练',
    teardown=False,
)
run_all.run_three_layer(
    splits['val'],
    runtime=runtime,
    update_params=False,
    log_metrics=False,
    exp_name='全量策略 验证',
    teardown=False,
)
run_all.run_three_layer(
    splits['test'],
    runtime=runtime,
    update_params=False,
    log_metrics=True,
    exp_name='全量策略',
    teardown=True,
)
"""
        if experiment == "three_layer_no_silent":
            body = """
runtime, _ = run_all.run_three_layer_no_silent(
    splits['train'],
    update_params=True,
    log_metrics=False,
    exp_name='全量策略（中层无沉默） 训练',
    teardown=False,
)
run_all.run_three_layer_no_silent(
    splits['val'],
    runtime=runtime,
    update_params=False,
    log_metrics=False,
    exp_name='全量策略（中层无沉默） 验证',
    teardown=False,
)
run_all.run_three_layer_no_silent(
    splits['test'],
    runtime=runtime,
    update_params=False,
    log_metrics=True,
    exp_name='全量策略（中层无沉默）',
    teardown=True,
)
"""
    return (
        build_common_preamble(sample_count, num_agents, debug_print)
        + f"\nprint('RUN_EXPERIMENT', '{experiment}')\n{body}\n"
    )


def build_child_env(run_dir: Path, visible_devices: Optional[str], require_cuda: bool):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["MAS_LOG_DIR"] = str(run_dir / "logs")
    env["MAS_PLOT_DIR"] = str(run_dir / "plots")
    if visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible_devices
    if require_cuda:
        env["MAS_REQUIRE_CUDA"] = "1"
    return env


def run_python_code(code: str, run_dir: Path, visible_devices: Optional[str], require_cuda: bool):
    env = build_child_env(run_dir, visible_devices, require_cuda)

    cmd = [sys.executable, "-c", code]
    return subprocess.run(cmd, cwd=run_dir, env=env, text=True)


def run_single_stage(sample_count: int, num_agents: int, debug_print: bool, visible_devices: Optional[str], require_cuda: bool, experiments: list[str]):
    run_dir = RUNS_DIR / f"staged_isolated_{sample_count}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    init_result = run_python_code(
        build_init_code(sample_count, num_agents, debug_print),
        run_dir,
        visible_devices,
        require_cuda,
    )
    stage_status = [{"experiment": "init_logs", "returncode": init_result.returncode}]
    if init_result.returncode != 0:
        summary = summarize_stage(run_dir)
        return init_result.returncode, stage_status, summary

    for experiment in experiments:
        result = run_python_code(
            build_experiment_code(sample_count, num_agents, debug_print, experiment),
            run_dir,
            visible_devices,
            require_cuda,
        )
        stage_status.append({"experiment": experiment, "returncode": result.returncode})
        if result.returncode != 0:
            break

    summary = summarize_stage(run_dir)
    return stage_status[-1]["returncode"], stage_status, summary


def main():
    parser = argparse.ArgumentParser(description="Run isolated MAS experiment suites.")
    parser.add_argument(
        "--sample-counts",
        nargs="+",
        type=int,
        default=[5, 10, 20, 50],
        help="Sample counts to run in order.",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=2,
        help="Number of outer agents to use for staged runs.",
    )
    parser.add_argument(
        "--debug-print",
        action="store_true",
        help="Enable per-round three-layer debug printing.",
    )
    parser.add_argument(
        "--visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for child stage processes.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail fast if the child process cannot see CUDA.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENT_TO_FUNC.keys()),
        choices=list(EXPERIMENT_TO_FUNC.keys()),
        help="Experiments to run for each stage.",
    )
    args = parser.parse_args()

    overall = []
    for sample_count in args.sample_counts:
        print(f"\n=== Isolated Stage {sample_count} samples ===")
        code, stage_status, summary = run_single_stage(
            sample_count=sample_count,
            num_agents=args.num_agents,
            debug_print=args.debug_print,
            visible_devices=args.visible_devices,
            require_cuda=args.require_cuda,
            experiments=args.experiments,
        )
        overall.append({
            "sample_count": sample_count,
            "returncode": code,
            "stage_status": stage_status,
            "summary": summary,
        })
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if code != 0:
            break

    overall_path = RUNS_DIR / "staged_isolated_progress.json"
    overall_path.write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nProgress:")
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
