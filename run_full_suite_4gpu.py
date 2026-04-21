import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import src.config as config
import src.data_loader as data_loader
import src.gspo_verl as gspo_verl
import src.model_loader as model_loader
import run_all


def _maybe_override_limits():
    train_val_total_limit = os.environ.get("MAS_TRAIN_VAL_TOTAL_LIMIT")
    test_sample_limit = os.environ.get("MAS_TEST_SAMPLE_LIMIT")

    if train_val_total_limit is not None:
        limit = int(train_val_total_limit)
        config.TRAIN_VAL_TOTAL_LIMIT = limit
        data_loader.TRAIN_VAL_TOTAL_LIMIT = limit
        run_all.TRAIN_VAL_TOTAL_LIMIT = limit

    if test_sample_limit is not None:
        limit = int(test_sample_limit)
        config.TEST_SAMPLE_LIMIT = limit
        data_loader.TEST_SAMPLE_LIMIT = limit
        run_all.TEST_SAMPLE_LIMIT = limit


def _apply_runtime_overrides():
    # 完整 suite 默认关闭逐轮调试打印，否则日志量会非常大，
    # 在全量 train/val/test 上会显著拖慢运行。
    debug_print = os.environ.get("MAS_THREE_LAYER_DEBUG_PRINT", "0").strip().lower()
    debug_enabled = debug_print in {"1", "true", "yes", "y", "on"}
    config.THREE_LAYER_DEBUG_PRINT = debug_enabled
    run_all.THREE_LAYER_DEBUG_PRINT = debug_enabled

    # 允许外部在不改代码的情况下做限样本 smoke，
    # 但默认仍然是完整 train/val/test。
    _maybe_override_limits()

    print(
        "FULL_SUITE_OVERRIDES",
        {
            "TRAIN_VAL_TOTAL_LIMIT": config.TRAIN_VAL_TOTAL_LIMIT,
            "TEST_SAMPLE_LIMIT": config.TEST_SAMPLE_LIMIT,
            "THREE_LAYER_DEBUG_PRINT": config.THREE_LAYER_DEBUG_PRINT,
            "MAS_PARALLEL_MODE": os.environ.get("MAS_PARALLEL_MODE", "serial"),
            "MAS_POLICY_DEVICE_MAP": os.environ.get("MAS_POLICY_DEVICE_MAP", "<auto>"),
            "MAS_GLOBAL_SEED": os.environ.get("MAS_GLOBAL_SEED", "<unset>"),
            "GSPO_TRAIN_DTYPE": config.GSPO_TRAIN_DTYPE,
            "NUM_AGENTS": config.NUM_AGENTS,
            "GSPO_NUM_CANDIDATES": config.GSPO_NUM_CANDIDATES,
            "MAX_NEW_TOKENS": config.MAX_NEW_TOKENS,
        },
        flush=True,
    )


def main():
    _apply_runtime_overrides()
    run_all.main()


if __name__ == "__main__":
    main()
