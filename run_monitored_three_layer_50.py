import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import src.config as config
import src.gspo_verl as gspo_verl
import src.model_loader as model_loader
import run_all

from src.data_loader import load_gsm8k_motac


def main():
    # 这版 runner 的目标不是最终成绩，而是先验证 50 条样本上的训练链路是否稳定。
    # 因此保留三层训练结构，但压缩 rollout 成本，便于持续监控。
    config.GSPO_NUM_CANDIDATES = 2
    gspo_verl.GSPO_NUM_CANDIDATES = 2

    config.GSPO_COMMENT_EVAL_SAMPLES = 1
    run_all.GSPO_COMMENT_EVAL_SAMPLES = 1

    config.MAX_NEW_TOKENS = 96
    model_loader.MAX_NEW_TOKENS = 96
    gspo_verl.MAX_NEW_TOKENS = 96

    config.NUM_AGENTS = 2
    run_all.NUM_AGENTS = 2

    config.THREE_LAYER_DEBUG_PRINT = True
    run_all.THREE_LAYER_DEBUG_PRINT = True

    run_all.init_logs()
    data = load_gsm8k_motac()[:50]
    print("SAMPLE_COUNT", len(data), flush=True)
    print("OVERRIDES", {
        "GSPO_NUM_CANDIDATES": gspo_verl.GSPO_NUM_CANDIDATES,
        "GSPO_COMMENT_EVAL_SAMPLES": run_all.GSPO_COMMENT_EVAL_SAMPLES,
        "MAX_NEW_TOKENS": model_loader.MAX_NEW_TOKENS,
        "NUM_AGENTS": run_all.NUM_AGENTS,
        "THREE_LAYER_DEBUG_PRINT": run_all.THREE_LAYER_DEBUG_PRINT,
        "MAS_PARALLEL_MODE": os.environ.get("MAS_PARALLEL_MODE", "serial"),
        "MAS_POLICY_DEVICE_MAP": os.environ.get("MAS_POLICY_DEVICE_MAP", "<auto>"),
        "MAS_GLOBAL_SEED": os.environ.get("MAS_GLOBAL_SEED", "<unset>"),
    }, flush=True)
    run_all.run_three_layer(data)


if __name__ == "__main__":
    main()
