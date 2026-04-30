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

try:
    from math_verify import parse as math_verify_parse
    from math_verify import verify as math_verify_verify
except Exception:
    math_verify_parse = None
    math_verify_verify = None

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
LATEX_FRACTION_PATTERN = re.compile(
    r"\\(?:dfrac|frac)\s*\{\s*([+-]?\d+(?:\.\d+)?)\s*\}\s*\{\s*([+-]?\d+(?:\.\d+)?)\s*\}"
)
SLASH_FRACTION_PATTERN = re.compile(
    r"(?<![\w.])([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)(?![\w.])"
)

# 这些词通常意味着“这一句在给最终结论”。
ANSWER_CUE_PATTERN = re.compile(
    r"(最终答案|答案是|答案为|答案[:：]|因此答案|所以答案|故答案|结论[:：]|"
    r"候选答案[:：]|candidate answer[:：]|candidate[:：]|"
    r"final answer|the answer is|answer[:：]|therefore|thus)",
    re.IGNORECASE,
)
OPTION_NUMBER_PATTERN = re.compile(r"\b(?:option|choice)\s*[:#-]?\s*([1-4])\b", re.IGNORECASE)
OPTION_LETTER_PATTERN = re.compile(r"\b(?:option|choice)\s*[:#-]?\s*([A-D])\b", re.IGNORECASE)
BARE_OPTION_PATTERN = re.compile(r"^\s*(?:final answer|answer|答案)?\s*[:：]?\s*\(?([A-D])\)?\.?\s*$", re.IGNORECASE)
ANSWER_PREFIX_PATTERN = re.compile(
    r"^\s*(?:therefore,\s*)?(?:the\s+)?(?:final answer|answer|答案|最终答案)\s*(?:is\s+|[:：]\s*)(?P<tail>.+?)\s*$",
    re.IGNORECASE,
)

ANSWER_FORMAT_GSM8K = "gsm8k_numeric"
ANSWER_FORMAT_MATH500 = "math500"
ANSWER_FORMAT_GPQA_DIAMOND = "gpqa_diamond"
_ACTIVE_ANSWER_FORMAT = ANSWER_FORMAT_GSM8K


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


def _extract_scalar_values(text):
    """
    Extract scalar numeric values, giving simple fractions priority.

    This keeps MATH-style final answers such as 14/3 or \\frac{14}{3} aligned
    with the numeric reward/evaluation code instead of reading only 14 or 3.
    """
    fraction_values = []
    for match in LATEX_FRACTION_PATTERN.finditer(text):
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if abs(denominator) > 1e-12:
            fraction_values.append((match.start(), numerator / denominator))
    for match in SLASH_FRACTION_PATTERN.finditer(text):
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if abs(denominator) > 1e-12:
            fraction_values.append((match.start(), numerator / denominator))
    if fraction_values:
        return [value for _, value in sorted(fraction_values)]
    return _extract_numbers(text)


def _extract_option_value(text):
    stripped = str(text or "").strip()
    match = OPTION_NUMBER_PATTERN.search(stripped)
    if match:
        return float(match.group(1))
    match = OPTION_LETTER_PATTERN.search(stripped)
    if match:
        return float(ord(match.group(1).upper()) - ord("A") + 1)
    direct = BARE_OPTION_PATTERN.fullmatch(stripped)
    if direct:
        return float(ord(direct.group(1).upper()) - ord("A") + 1)
    return None


def normalize_answer_format(answer_format=None):
    normalized = str(answer_format or _ACTIVE_ANSWER_FORMAT or ANSWER_FORMAT_GSM8K).strip().lower().replace("-", "_")
    aliases = {
        "gsm8k": ANSWER_FORMAT_GSM8K,
        "gsm8k_numeric": ANSWER_FORMAT_GSM8K,
        "numeric": ANSWER_FORMAT_GSM8K,
        "math": ANSWER_FORMAT_MATH500,
        "math500": ANSWER_FORMAT_MATH500,
        "math_500": ANSWER_FORMAT_MATH500,
        "gpqa": ANSWER_FORMAT_GPQA_DIAMOND,
        "gpqa_diamond": ANSWER_FORMAT_GPQA_DIAMOND,
    }
    return aliases.get(normalized, ANSWER_FORMAT_GSM8K)


def set_active_answer_format(answer_format):
    global _ACTIVE_ANSWER_FORMAT
    _ACTIVE_ANSWER_FORMAT = normalize_answer_format(answer_format)
    return _ACTIVE_ANSWER_FORMAT


def get_active_answer_format():
    return _ACTIVE_ANSWER_FORMAT


def get_answer_format_from_item(item):
    if isinstance(item, dict):
        return normalize_answer_format(item.get("answer_format") or item.get("source_dataset"))
    return normalize_answer_format(None)


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


def _load_modelscope_dataset_candidates(candidates, dataset_label):
    errors = []
    for kwargs in candidates:
        load_kwargs = {
            key: value
            for key, value in kwargs.items()
            if value is not None and str(value).strip() != ""
        }
        load_kwargs.setdefault("cache_dir", DATA_CACHE)
        load_kwargs.setdefault("trust_remote_code", True)
        try:
            return MsDataset.load(**load_kwargs)
        except Exception as exc:
            errors.append(f"{load_kwargs}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"无法从 ModelScope 加载 {dataset_label}。尝试过："
        + " | ".join(errors)
    )


def _load_math500_raw_dataset():
    return _load_modelscope_dataset_candidates(
        [
            {
                "dataset_name": os.environ.get(
                    "MAS_MATH500_DATASET",
                    "AI-ModelScope/MATH-500",
                ),
                "split": os.environ.get("MAS_MATH500_SPLIT", "test"),
            },
            {
                "dataset_name": "modelscope/R1-Distill-Math-Test",
                "split": "test",
            },
        ],
        "MATH-500",
    )


def _load_gpqa_diamond_raw_dataset():
    return _load_modelscope_dataset_candidates(
        [
            {
                "dataset_name": os.environ.get(
                    "MAS_GPQA_DATASET",
                    "AI-ModelScope/GPQA",
                ),
                "subset_name": os.environ.get(
                    "MAS_GPQA_SUBSET",
                    "gpqa_diamond",
                ),
            },
            {
                "dataset_name": "modelscope/R1-Distill-Math-Test",
                "split": "test",
            },
        ],
        "GPQA-Diamond",
    )


def _dataset_to_items(dataset, preferred_splits=("test", "validation", "train")):
    if hasattr(dataset, "keys") and not isinstance(dataset, list):
        keys = list(dataset.keys())
        for split in preferred_splits:
            if split in keys:
                return list(dataset[split])
        if keys:
            return list(dataset[keys[0]])
    return list(dataset)


def _process_gsm8k_items(items):
    processed = []
    for item in items:
        q = item["question"].strip()
        ans_text = item["answer"]
        nums = _extract_numbers(ans_text)
        gt = nums[-1] if nums else None
        if gt is not None:
            processed.append({
                "question": q,
                "ground_truth": gt,
                "answer_format": ANSWER_FORMAT_GSM8K,
                "source_dataset": "gsm8k",
            })
    return processed


def _balanced_brace_content(text, start_idx):
    if start_idx < 0 or start_idx >= len(text) or text[start_idx] != "{":
        return None
    depth = 0
    for idx in range(start_idx, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx + 1:idx]
    return None


def _extract_boxed_content(text):
    for marker in ("\\boxed", "\\fbox"):
        pos = text.find(marker)
        if pos < 0:
            continue
        brace_pos = text.find("{", pos + len(marker))
        content = _balanced_brace_content(text, brace_pos)
        if content is not None:
            return content
    return text


def _parse_single_numeric_ground_truth(value):
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = _extract_boxed_content(text)
    cleaned = (
        text.replace("$", "")
        .replace("\\left", "")
        .replace("\\right", "")
        .replace("\\,", "")
        .replace("\\!", "")
        .strip()
    )

    numeric_match = re.fullmatch(r"[+-]?\$?\d[\d,]*(?:\.\d+)?", cleaned)
    if numeric_match:
        return _normalize_number_token(cleaned)

    fraction_match = re.fullmatch(LATEX_FRACTION_PATTERN, cleaned)
    if not fraction_match:
        fraction_match = re.fullmatch(
            r"([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)",
            cleaned,
        )
    if fraction_match:
        numerator = float(fraction_match.group(1))
        denominator = float(fraction_match.group(2))
        if abs(denominator) > 1e-12:
            return numerator / denominator

    if re.search(r"[A-Za-z\\π]|[,()]", cleaned):
        return None
    nums = _extract_scalar_values(cleaned)
    return nums[-1] if len(nums) == 1 else None


def _raw_nested_item(item):
    prompt = item.get("prompt") if isinstance(item, dict) else None
    if isinstance(prompt, dict) and isinstance(prompt.get("raw_input"), dict):
        return prompt["raw_input"]
    return item


def _process_math500_items(items):
    processed = []
    for item in items:
        if str(item.get("dataset_name", "")).lower() not in {"", "math_500", "math500"}:
            continue
        raw = _raw_nested_item(item)
        q = str(raw.get("problem") or raw.get("question") or "").strip()
        ans_text = raw.get("answer")
        if q and ans_text is not None and str(ans_text).strip():
            processed.append({
                "question": q,
                "ground_truth": str(ans_text).strip(),
                "answer_format": ANSWER_FORMAT_MATH500,
                "source_dataset": "math500",
                "source_id": raw.get("unique_id") or raw.get("id"),
                "ground_truth_raw": str(ans_text),
            })
    return processed


def _first_nonempty(item, keys):
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _process_gpqa_diamond_items(items):
    processed = []
    rng = np.random.default_rng(DATA_SPLIT_SEED)
    for raw_idx, item in enumerate(items):
        if str(item.get("dataset_name", "")).lower() not in {"", "gpqa_diamond", "gpqa-diamond"}:
            continue
        raw = _raw_nested_item(item)
        question = _first_nonempty(
            raw,
            ["Question", "Extra Revised Question", "Pre-Revision Question", "question"],
        )
        correct = _first_nonempty(
            raw,
            [
                "Correct Answer",
                "Extra Revised Correct Answer",
                "Pre-Revision Correct Answer",
                "answer",
            ],
        )
        incorrects = []
        for idx in range(1, 4):
            wrong = _first_nonempty(
                raw,
                [
                    f"Incorrect Answer {idx}",
                    f"Extra Revised Incorrect Answer {idx}",
                    f"Pre-Revision Incorrect Answer {idx}",
                ],
            )
            if wrong:
                incorrects.append(wrong)
        if not question or not correct or len(incorrects) < 1:
            continue
        options = [("correct", correct)] + [("incorrect", wrong) for wrong in incorrects]
        order = list(rng.permutation(len(options)))
        ordered_options = [options[idx] for idx in order]
        correct_number = 1 + next(
            out_idx
            for out_idx, original_idx in enumerate(order)
            if original_idx == 0
        )
        correct_number = int(correct_number)
        option_labels = ["A", "B", "C", "D"]
        option_lines = [
            f"{option_labels[idx]}. {text.strip()}"
            for idx, (_, text) in enumerate(ordered_options)
        ]
        q = (
            question.strip()
            + "\n\nOptions:\n"
            + "\n".join(option_lines)
            + "\n\nAnswer with only the option letter (A-D)."
        )
        processed.append({
            "question": q,
            "ground_truth": float(correct_number),
            "answer_format": ANSWER_FORMAT_GPQA_DIAMOND,
            "source_dataset": "gpqa-diamond",
            "source_id": raw.get("Record ID") or raw.get("id") or raw_idx,
            "ground_truth_raw": correct,
            "ground_truth_choice": option_labels[correct_number - 1],
        })
    return processed


def _maybe_limit_items(items, limit):
    if limit is None:
        return items
    return items[:max(int(limit), 0)]


def normalize_final_test_dataset_name(name):
    normalized = str(name or "gsm8k").strip().lower().replace("_", "-")
    aliases = {
        "gsm8k": "gsm8k",
        "math": "math500",
        "math-500": "math500",
        "math500": "math500",
        "gpqa": "gpqa-diamond",
        "gpqa-diamond": "gpqa-diamond",
    }
    if normalized not in aliases:
        raise ValueError(
            f"不支持的测试集: {name}. 可选: gsm8k, math500, gpqa-diamond"
        )
    return aliases[normalized]


def load_final_test_data(test_dataset="gsm8k", test_limit=None):
    test_dataset = normalize_final_test_dataset_name(test_dataset)
    if test_dataset == "gsm8k":
        raw_items = list(_load_gsm8k_raw_dataset()["test"])
        processed = _process_gsm8k_items(raw_items)
    elif test_dataset == "math500":
        raw_items = _dataset_to_items(
            _load_math500_raw_dataset(),
            preferred_splits=("test", "train"),
        )
        processed = _process_math500_items(raw_items)
    elif test_dataset == "gpqa-diamond":
        raw_items = _dataset_to_items(
            _load_gpqa_diamond_raw_dataset(),
            preferred_splits=("test", "train"),
        )
        processed = _process_gpqa_diamond_items(raw_items)
    else:
        raise ValueError(f"不支持的测试集: {test_dataset}")
    return {
        "name": test_dataset,
        "raw_size": len(raw_items),
        "processed_size": len(processed),
        "data": _maybe_limit_items(processed, test_limit),
    }


def load_final_train_test_splits(train_limit=None, test_limit=None, test_dataset="gsm8k"):
    """
    Final-suite data protocol:
    - training always uses the official GSM8K train split;
    - evaluation uses the selected test set.

    Each processed item carries an answer_format field. GSM8K uses numeric
    extraction/reward, MATH-500 keeps symbolic answers for math_verify-based
    equivalence, and GPQA-Diamond is converted to option-letter tasks.
    """
    gsm8k = _load_gsm8k_raw_dataset()
    train_data = _process_gsm8k_items(
        _maybe_limit_items(list(gsm8k["train"]), train_limit)
    )
    if normalize_final_test_dataset_name(test_dataset) == "gsm8k":
        raw_test_items = list(gsm8k["test"])
        processed_test = _process_gsm8k_items(raw_test_items)
        test_info = {
            "name": "gsm8k",
            "raw_size": len(raw_test_items),
            "processed_size": len(processed_test),
            "data": _maybe_limit_items(processed_test, test_limit),
        }
    else:
        test_info = load_final_test_data(test_dataset, test_limit=test_limit)
    return {
        "train": train_data,
        "test": test_info["data"],
        "train_dataset": "gsm8k",
        "test_dataset": test_info["name"],
        "test_raw_size": test_info["raw_size"],
        "test_processed_size": test_info["processed_size"],
    }


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


def load_gsm8k_official_splits(train_limit=None, test_limit=None):
    """
    Load the official GSM8K train/test split without carving out a validation set.

    The final paper experiments use the full official train split for online
    training and the full official test split for evaluation. Optional limits are
    only for smoke tests.
    """
    dataset = _load_gsm8k_raw_dataset()
    train_data = _process_gsm8k_items(
        _maybe_limit_items(list(dataset["train"]), train_limit)
    )
    test_data = _process_gsm8k_items(
        _maybe_limit_items(list(dataset["test"]), test_limit)
    )
    return {
        "train": train_data,
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

def _coerce_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    parsed = _parse_single_numeric_ground_truth(value)
    return parsed


def _extract_numeric_answer(text):
    if not text:
        return None

    lines = [line for line in text.replace("\r\n", "\n").split("\n") if line.strip()]

    for raw_line in reversed(lines):
        cleaned_line = _clean_candidate_line(raw_line)
        if not cleaned_line:
            continue
        if not ANSWER_CUE_PATTERN.search(cleaned_line):
            continue
        nums = _extract_scalar_values(cleaned_line)
        if nums:
            return nums[-1]

    for raw_line in reversed(lines):
        cleaned_line = _clean_candidate_line(raw_line)
        if not cleaned_line:
            continue
        nums = _extract_scalar_values(cleaned_line)
        if nums:
            return nums[-1]

    nums = _extract_scalar_values(text)
    return nums[-1] if nums else None


def _extract_gpqa_answer(text):
    if not text:
        return None
    lines = [line for line in str(text).replace("\r\n", "\n").split("\n") if line.strip()]
    for raw_line in reversed(lines):
        cleaned_line = _clean_candidate_line(raw_line)
        if not cleaned_line:
            continue
        if ANSWER_CUE_PATTERN.search(cleaned_line):
            option_value = _extract_option_value(cleaned_line)
            if option_value is not None:
                return option_value
    for raw_line in reversed(lines[-3:]):
        cleaned_line = _clean_candidate_line(raw_line)
        option_value = _extract_option_value(cleaned_line)
        if option_value is not None:
            return option_value
    return None


def _strip_answer_tail(text):
    cleaned = str(text or "").strip()
    match = ANSWER_PREFIX_PATTERN.match(cleaned)
    if match:
        cleaned = match.group("tail").strip()
    cleaned = re.sub(r"^#+\s*", "", cleaned).strip()
    cleaned = cleaned.rstrip(".。").strip()
    return cleaned or None


def _last_boxed_content(text):
    raw = str(text or "")
    last_content = None
    for marker in ("\\boxed", "\\fbox"):
        search_start = 0
        while True:
            pos = raw.find(marker, search_start)
            if pos < 0:
                break
            brace_pos = raw.find("{", pos + len(marker))
            content = _balanced_brace_content(raw, brace_pos)
            if content is not None:
                last_content = content.strip()
            search_start = pos + len(marker)
    return last_content


def _extract_math500_answer(text):
    if not text:
        return None
    boxed = _last_boxed_content(text)
    if boxed:
        return boxed
    lines = [line for line in str(text).replace("\r\n", "\n").split("\n") if line.strip()]
    for raw_line in reversed(lines):
        cleaned_line = _clean_candidate_line(raw_line)
        if not cleaned_line:
            continue
        if ANSWER_CUE_PATTERN.search(cleaned_line):
            return _strip_answer_tail(cleaned_line)
    if lines:
        tail = _strip_answer_tail(lines[-1])
        if tail and len(tail) <= 160:
            return tail
    return None


def extract_pred_num(text, answer_format=None):
    """
    Dataset-aware final-answer extractor.

    The name is kept for compatibility with the existing MAS code. The return
    value is numeric for GSM8K/GPQA and a raw symbolic answer string for
    MATH-500.
    """
    fmt = normalize_answer_format(answer_format)
    if fmt == ANSWER_FORMAT_GPQA_DIAMOND:
        return _extract_gpqa_answer(text)
    if fmt == ANSWER_FORMAT_MATH500:
        return _extract_math500_answer(text)
    return _extract_numeric_answer(text)


def _normalize_math_answer_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    boxed = _last_boxed_content(text)
    if boxed:
        text = boxed
    text = (
        text.replace("$", "")
        .replace("\\left", "")
        .replace("\\right", "")
        .replace("\\,", "")
        .replace("\\!", "")
        .strip()
    )
    text = re.sub(r"\s+", "", text)
    return text.rstrip(".。")


def _parse_math_answer(value):
    if value is None or math_verify_parse is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    candidates = [text]
    if "\\boxed" not in text:
        candidates.append(f"\\boxed{{{text}}}")
    parsed = []
    for candidate in candidates:
        try:
            parsed.extend(math_verify_parse(candidate, raise_on_error=False))
        except Exception:
            continue
    return parsed


def _math_answers_match(pred, gt):
    pred_norm = _normalize_math_answer_text(pred)
    gt_norm = _normalize_math_answer_text(gt)
    if not pred_norm or not gt_norm:
        return False
    if pred_norm.lower() == gt_norm.lower():
        return True
    if math_verify_parse is None or math_verify_verify is None:
        return False
    pred_parsed = _parse_math_answer(pred)
    gt_parsed = _parse_math_answer(gt)
    if not pred_parsed or not gt_parsed:
        return False
    try:
        return bool(math_verify_verify(gt_parsed, pred_parsed, raise_on_error=False))
    except Exception:
        return False


def answers_match(pred, gt, answer_format=None):
    fmt = normalize_answer_format(answer_format)
    if pred is None or gt is None:
        return False
    if fmt == ANSWER_FORMAT_MATH500:
        return _math_answers_match(pred, gt)
    if fmt == ANSWER_FORMAT_GPQA_DIAMOND:
        pred_num = _coerce_float(pred)
        if pred_num is None:
            pred_num = _extract_gpqa_answer(str(pred))
        gt_num = _coerce_float(gt)
        if gt_num is None:
            gt_num = _extract_gpqa_answer(str(gt))
        return pred_num is not None and gt_num is not None and abs(pred_num - gt_num) < 1e-3
    pred_num = _coerce_float(pred)
    gt_num = _coerce_float(gt)
    return pred_num is not None and gt_num is not None and abs(pred_num - gt_num) < 1e-3


def same_answer_value(pred_a, pred_b, answer_format=None):
    fmt = normalize_answer_format(answer_format)
    if pred_a is None or pred_b is None:
        return False
    if fmt == ANSWER_FORMAT_MATH500:
        return _math_answers_match(pred_a, pred_b)
    return answers_match(pred_a, pred_b, fmt)


def answer_reward(
    pred,
    gt,
    answer_format=None,
    missing_penalty=REWARD_MISSING_PENALTY,
    exact_weight=REWARD_EXACT_WEIGHT,
    dense_weight=REWARD_DENSE_WEIGHT,
):
    fmt = normalize_answer_format(answer_format)
    if pred is None or gt is None:
        return float(missing_penalty)
    if fmt in {ANSWER_FORMAT_MATH500, ANSWER_FORMAT_GPQA_DIAMOND}:
        return 1.0 if answers_match(pred, gt, fmt) else 0.0
    pred_num = _coerce_float(pred)
    gt_num = _coerce_float(gt)
    if pred_num is None or gt_num is None:
        return float(missing_penalty)
    exact_reward = 1.0 if abs(pred_num - gt_num) < 1e-3 else 0.0
    scale = max(abs(gt_num), 1.0)
    rel_error = abs(pred_num - gt_num) / scale
    dense_reward = 1.0 / (1.0 + rel_error)
    return float(exact_weight) * exact_reward + float(dense_weight) * dense_reward


def compute_accuracy(preds, gts):
    total = len([g for g in gts if g is not None])
    correct = 0
    for p, g in zip(preds, gts):
        if answers_match(p, g):
            correct += 1
    return correct / total if total > 0 else 0.0
