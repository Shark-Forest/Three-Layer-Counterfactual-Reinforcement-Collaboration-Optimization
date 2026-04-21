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
    model_scope = os.environ.get("MAS_MODEL_SCOPE")
    if model_scope is not None:
        value = model_scope.strip()
        config.GPT2_MODEL_SCOPE = value
        model_loader.GPT2_MODEL_SCOPE = value

    gspo_finetune_mode = os.environ.get("MAS_GSPO_FINETUNE_MODE")
    if gspo_finetune_mode is not None:
        value = gspo_finetune_mode.strip().lower()
        config.GSPO_FINETUNE_MODE = value
        gspo_verl.GSPO_FINETUNE_MODE = value

    gspo_train_dtype = os.environ.get("MAS_GSPO_TRAIN_DTYPE")
    if gspo_train_dtype is not None:
        value = gspo_train_dtype.strip()
        config.GSPO_TRAIN_DTYPE = value
        model_loader.GSPO_TRAIN_DTYPE = value

    gspo_lora_r = os.environ.get("MAS_GSPO_LORA_R")
    if gspo_lora_r is not None:
        value = int(gspo_lora_r)
        config.GSPO_LORA_R = value
        gspo_verl.GSPO_LORA_R = value

    gspo_lora_alpha = os.environ.get("MAS_GSPO_LORA_ALPHA")
    if gspo_lora_alpha is not None:
        value = int(gspo_lora_alpha)
        config.GSPO_LORA_ALPHA = value
        gspo_verl.GSPO_LORA_ALPHA = value

    gspo_lora_dropout = os.environ.get("MAS_GSPO_LORA_DROPOUT")
    if gspo_lora_dropout is not None:
        value = float(gspo_lora_dropout)
        config.GSPO_LORA_DROPOUT = value
        gspo_verl.GSPO_LORA_DROPOUT = value

    gspo_lora_bias = os.environ.get("MAS_GSPO_LORA_BIAS")
    if gspo_lora_bias is not None:
        value = gspo_lora_bias.strip().lower()
        config.GSPO_LORA_BIAS = value
        gspo_verl.GSPO_LORA_BIAS = value

    gspo_lora_target_modules = os.environ.get("MAS_GSPO_LORA_TARGET_MODULES")
    if gspo_lora_target_modules is not None:
        value = gspo_lora_target_modules.strip()
        config.GSPO_LORA_TARGET_MODULES = value
        gspo_verl.GSPO_LORA_TARGET_MODULES = value

    debug_print = os.environ.get("MAS_THREE_LAYER_DEBUG_PRINT", "0").strip().lower()
    debug_enabled = debug_print in {"1", "true", "yes", "y", "on"}
    config.THREE_LAYER_DEBUG_PRINT = debug_enabled
    run_all.THREE_LAYER_DEBUG_PRINT = debug_enabled

    expand_all_train_branches = os.environ.get("MAS_EXPAND_ALL_TRAIN_BRANCHES")
    if expand_all_train_branches is not None:
        enabled = expand_all_train_branches.strip().lower() in {"1", "true", "yes", "y", "on"}
        config.THREE_LAYER_EXPAND_ALL_TRAIN_BRANCHES = enabled
        run_all.THREE_LAYER_EXPAND_ALL_TRAIN_BRANCHES = enabled

    _maybe_override_limits()

    gspo_num_candidates = os.environ.get("MAS_GSPO_NUM_CANDIDATES")
    if gspo_num_candidates is not None:
        value = int(gspo_num_candidates)
        config.GSPO_NUM_CANDIDATES = value
        gspo_verl.GSPO_NUM_CANDIDATES = value

    max_new_tokens = os.environ.get("MAS_MAX_NEW_TOKENS")
    if max_new_tokens is not None:
        value = int(max_new_tokens)
        config.MAX_NEW_TOKENS = value
        model_loader.MAX_NEW_TOKENS = value
        gspo_verl.MAX_NEW_TOKENS = value

    train_num_rounds = os.environ.get("MAS_TRAIN_NUM_ROUNDS")
    if train_num_rounds is not None:
        value = int(train_num_rounds)
        config.TRAIN_NUM_ROUNDS = value
        run_all.TRAIN_NUM_ROUNDS = value

    infer_num_rounds = os.environ.get("MAS_INFER_NUM_ROUNDS")
    if infer_num_rounds is not None:
        value = int(infer_num_rounds)
        config.INFER_NUM_ROUNDS = value
        run_all.INFER_NUM_ROUNDS = value

    # 兼容仍然引用 NUM_ROUNDS 的旧代码路径。
    config.NUM_ROUNDS = config.TRAIN_NUM_ROUNDS
    run_all.NUM_ROUNDS = config.NUM_ROUNDS
    gspo_verl.NUM_ROUNDS = config.NUM_ROUNDS
    model_loader.NUM_ROUNDS = config.NUM_ROUNDS

    print(
        "THREE_LAYER_ONLY_OVERRIDES",
        {
            "MODEL_SCOPE": config.GPT2_MODEL_SCOPE,
            "TRAIN_VAL_TOTAL_LIMIT": config.TRAIN_VAL_TOTAL_LIMIT,
            "TEST_SAMPLE_LIMIT": config.TEST_SAMPLE_LIMIT,
            "THREE_LAYER_DEBUG_PRINT": config.THREE_LAYER_DEBUG_PRINT,
            "MAS_PARALLEL_MODE": os.environ.get("MAS_PARALLEL_MODE", "serial"),
            "MAS_POLICY_DEVICE_MAP": os.environ.get("MAS_POLICY_DEVICE_MAP", "<auto>"),
            "MAS_GLOBAL_SEED": os.environ.get("MAS_GLOBAL_SEED", "<unset>"),
            "GSPO_FINETUNE_MODE": config.GSPO_FINETUNE_MODE,
            "GSPO_TRAIN_DTYPE": config.GSPO_TRAIN_DTYPE,
            "GSPO_LORA_R": config.GSPO_LORA_R,
            "GSPO_LORA_ALPHA": config.GSPO_LORA_ALPHA,
            "GSPO_LORA_DROPOUT": config.GSPO_LORA_DROPOUT,
            "GSPO_LORA_BIAS": config.GSPO_LORA_BIAS,
            "GSPO_LORA_TARGET_MODULES": config.GSPO_LORA_TARGET_MODULES or "<auto>",
            "NUM_AGENTS": config.NUM_AGENTS,
            "GSPO_NUM_CANDIDATES": config.GSPO_NUM_CANDIDATES,
            "MAX_NEW_TOKENS": config.MAX_NEW_TOKENS,
            "TRAIN_NUM_ROUNDS": config.TRAIN_NUM_ROUNDS,
            "INFER_NUM_ROUNDS": config.INFER_NUM_ROUNDS,
            "THREE_LAYER_EXPAND_ALL_TRAIN_BRANCHES": config.THREE_LAYER_EXPAND_ALL_TRAIN_BRANCHES,
            "CHECKPOINT_DIR": config.CHECKPOINT_DIR,
            "CHECKPOINT_EVERY_SAMPLES": config.CHECKPOINT_EVERY_SAMPLES,
            "RESUME_CHECKPOINT_PATH": config.RESUME_CHECKPOINT_PATH,
        },
        flush=True,
    )


def main():
    _apply_runtime_overrides()
    run_all.init_logs()
    split_data = data_loader.load_gsm8k_splits(
        train_val_total_limit=config.TRAIN_VAL_TOTAL_LIMIT,
        test_limit=config.TEST_SAMPLE_LIMIT,
        train_ratio=config.TRAIN_RATIO,
        split_seed=config.DATA_SPLIT_SEED,
    )
    train_data = split_data["train"]
    val_data = split_data["val"]
    test_data = split_data["test"]
    print(
        f"数据划分：{run_all.build_split_size_message(train_data, val_data, test_data)}",
        flush=True,
    )

    checkpoint_options = {
        "checkpoint_dir": config.CHECKPOINT_DIR,
        "every_samples": config.CHECKPOINT_EVERY_SAMPLES,
        "resume_path": config.RESUME_CHECKPOINT_PATH,
    }

    runtime, _ = run_all.run_three_layer(
        train_data,
        update_params=True,
        log_metrics=False,
        exp_name="全量策略 训练",
        teardown=False,
        checkpoint_options=checkpoint_options,
    )
    run_all.run_three_layer(
        val_data,
        runtime=runtime,
        update_params=False,
        log_metrics=False,
        exp_name="全量策略 验证",
        teardown=False,
    )
    run_all.run_three_layer(
        test_data,
        runtime=runtime,
        update_params=False,
        log_metrics=True,
        exp_name="全量策略",
        teardown=True,
    )


if __name__ == "__main__":
    main()
