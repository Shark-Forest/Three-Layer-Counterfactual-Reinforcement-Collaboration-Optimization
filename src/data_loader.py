import os
import re
import numpy as np
import ssl
from src.config import *
# 解决SSL报错
ssl._create_default_https_context = ssl._create_unverified_context
# 【关键】配置Hugging Face国内镜像（魔塔内部调用datasets时也会走这个镜像）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 注意顺序：
# 必须先 import config，并在 config 里把 HF_HOME / HF_DATASETS_CACHE /
# HF_MODULES_CACHE / TMPDIR 等环境变量设好，再 import MsDataset。
# 否则 datasets 可能在 import 时就把缓存路径锁定到 /root/.cache。
from modelscope.msdatasets import MsDataset

PROJECT_DATA_CACHE = os.path.abspath(DATA_CACHE)
PROJECT_DATA_HF_HOME = os.path.join(PROJECT_DATA_CACHE, "_hf_home")
PROJECT_DATASET_CACHE = os.path.join(PROJECT_DATA_CACHE, "_hf_datasets")
PROJECT_DATA_MODULES_CACHE = os.path.join(PROJECT_DATA_CACHE, "_hf_modules")
PROJECT_DATA_TMPDIR = os.path.join(PROJECT_DATA_CACHE, "_tmp")

os.makedirs(PROJECT_DATA_CACHE, exist_ok=True)
os.makedirs(PROJECT_DATA_HF_HOME, exist_ok=True)
os.makedirs(PROJECT_DATASET_CACHE, exist_ok=True)
os.makedirs(PROJECT_DATA_MODULES_CACHE, exist_ok=True)
os.makedirs(PROJECT_DATA_TMPDIR, exist_ok=True)

# 再次显式覆盖，确保数据集链路绝不写到 /root/.cache。
os.environ["HF_HOME"] = PROJECT_DATA_HF_HOME
os.environ["HF_DATASETS_CACHE"] = PROJECT_DATASET_CACHE
os.environ["HF_MODULES_CACHE"] = PROJECT_DATA_MODULES_CACHE
os.environ["TMPDIR"] = PROJECT_DATA_TMPDIR

# 更稳的数字匹配：
# - 支持逗号分隔，如 70,000
# - 支持负号，如 -3.5
# - 支持前缀货币符号，如 $18
NUMBER_PATTERN = re.compile(r"(?<![\w/.-])-?\$?\d[\d,]*\.?\d*")

# 多轮上下文里常见的“历史标记”。
# 如果模型把历史整段抄回来，我们不希望把这些行里的轮次编号当成最终答案。
ROUND_MARKER_PATTERN = re.compile(r"^\s*\[Round\s+\d+\s*\|", re.IGNORECASE)

# 很多评论输出会写成：
# 1. ...
# 2. ...
# 3. ...
# 这里的 1/2/3 只是列表编号，不是题目答案，所以抽取前先去掉。
LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[\.\)、:：])\s*")

# 这些词通常意味着“这一句在给最终结论”。
ANSWER_CUE_PATTERN = re.compile(
    r"(最终答案|答案是|答案为|答案[:：]|因此答案|所以答案|故答案|结论[:：]|"
    r"final answer|the answer is|answer[:：]|therefore|thus)",
    re.IGNORECASE,
)


def _normalize_number_token(token):
    """把 '$70,000' 这类字符串转成 float(70000.0)。"""
    cleaned = token.replace("$", "").replace(",", "").strip()
    return float(cleaned) if cleaned else None


def _extract_numbers(text):
    """从一段文本里提取所有数字，返回 float 列表。"""
    values = []
    for token in NUMBER_PATTERN.findall(text):
        try:
            values.append(_normalize_number_token(token))
        except ValueError:
            continue
    return values


def _clean_candidate_line(line):
    """
    清理一行候选文本，尽量去掉：
    - 多轮历史标签
    - 列表编号 1. / 2. / 3.
    """
    cleaned = line.strip()
    if not cleaned:
        return ""
    if ROUND_MARKER_PATTERN.match(cleaned):
        return ""
    cleaned = LIST_PREFIX_PATTERN.sub("", cleaned)
    return cleaned.strip()

def _load_gsm8k_raw_dataset():
    return MsDataset.load(
        "gsm8k",
        subset_name="main",
        cache_dir=DATA_CACHE,
        namespace="modelscope",
        # modelscope 新版本默认不再信任远端数据集脚本，需要显式开启。
        trust_remote_code=True,
    )


def _process_gsm8k_items(items):
    processed = []
    for item in items:
        q = item["question"].strip()
        ans_text = item["answer"]
        nums = _extract_numbers(ans_text)
        gt = nums[-1] if nums else None
        if gt is not None:
            processed.append({"question": q, "ground_truth": gt})
    return processed


def _maybe_limit_items(items, limit):
    if limit is None:
        return items
    return items[:max(int(limit), 0)]


def load_gsm8k_splits(
    train_val_total_limit=TRAIN_VAL_TOTAL_LIMIT,
    test_limit=TEST_SAMPLE_LIMIT,
    train_ratio=TRAIN_RATIO,
    split_seed=DATA_SPLIT_SEED,
):
    """
    加载 GSM8K，并把官方 train split 再切成 train/val。

    约定：
    - 官方 train -> 再按 train_ratio 划分成 train / val
    - 官方 test  -> 作为最终 test
    """
    dataset = _load_gsm8k_raw_dataset()
    train_val_data = _process_gsm8k_items(
        _maybe_limit_items(list(dataset["train"]), train_val_total_limit)
    )
    test_data = _process_gsm8k_items(
        _maybe_limit_items(list(dataset["test"]), test_limit)
    )

    if len(train_val_data) <= 1:
        return {
            "train": train_val_data,
            "val": [],
            "test": test_data,
        }

    rng = np.random.default_rng(split_seed)
    indices = np.arange(len(train_val_data))
    rng.shuffle(indices)

    split_idx = int(len(indices) * float(train_ratio))
    split_idx = min(max(split_idx, 1), len(indices) - 1)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    return {
        "train": [train_val_data[idx] for idx in train_indices],
        "val": [train_val_data[idx] for idx in val_indices],
        "test": test_data,
    }


def load_gsm8k_motac(test_limit=TEST_SAMPLE_LIMIT):
    """
    兼容旧代码：返回官方 test split。
    """
    return load_gsm8k_splits(
        train_val_total_limit=TRAIN_VAL_TOTAL_LIMIT,
        test_limit=test_limit,
    )["test"]

def compute_accuracy(preds, gts):
    correct = 0
    total = len([g for g in gts if g is not None])
    for p, g in zip(preds, gts):
        if p is not None and abs(p - g) < 1e-3:
            correct += 1
    return correct / total if total > 0 else 0.0

def extract_pred_num(text):
    """
    更稳地从模型输出里抽取“最终答案数字”。

    旧逻辑的问题：
    - 直接取整段文本最后一个数字
    - 多轮实验里模型经常会抄历史、抄列表、抄 [Round 4 | ...]
    - 于是很容易把真正答案 18 错提成尾部列表编号 1/2/3/4

    新逻辑按“从可靠到保底”的顺序提取：
    1. 倒序找带有“答案/最终答案/final answer”等提示词的行
    2. 再倒序找最后一个有数字的、且不像历史标记/列表编号的行
    3. 最后才退回到整段文本中的最后一个数字
    """
    if not text:
        return None

    lines = [line for line in text.replace("\r\n", "\n").split("\n") if line.strip()]

    # 先优先找“答案句”。
    for raw_line in reversed(lines):
        cleaned_line = _clean_candidate_line(raw_line)
        if not cleaned_line:
            continue
        if not ANSWER_CUE_PATTERN.search(cleaned_line):
            continue
        nums = _extract_numbers(cleaned_line)
        if nums:
            return nums[-1]

    # 如果没有明显答案句，就找最后一个“像结论”的数字行。
    for raw_line in reversed(lines):
        cleaned_line = _clean_candidate_line(raw_line)
        if not cleaned_line:
            continue
        nums = _extract_numbers(cleaned_line)
        if nums:
            return nums[-1]

    # 最后的兜底：整个文本里实在只有零散数字时，才退回旧行为。
    nums = _extract_numbers(text)
    return nums[-1] if nums else None
