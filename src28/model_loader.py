import os
import json
import torch
import ssl
from copy import deepcopy
# 【关键修改1】解决魔塔的SSL证书报错
ssl._create_default_https_context = ssl._create_unverified_context

from modelscope import AutoModelForCausalLM, AutoTokenizer
from src28.config import *

try:
    from transformers.cache_utils import DynamicCache
except Exception:
    DynamicCache = None

if (
    DynamicCache is not None
    and not hasattr(DynamicCache, "seen_tokens")
    and hasattr(DynamicCache, "get_seq_length")
):
    # 兼容较新的 transformers：
    # 新版 DynamicCache 去掉了 seen_tokens，
    # 但 Phi-3 的远端 modeling 代码仍会访问该属性。
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

if (
    DynamicCache is not None
    and not hasattr(DynamicCache, "get_max_length")
):
    # 兼容较新的 transformers：
    # 新版 DynamicCache 把最大 cache 长度暴露成 max_cache_len /
    # get_max_cache_shape()，但 Phi-3 远端代码仍调用 get_max_length()。
    def _compat_get_max_length(self):
        try:
            max_cache_len = getattr(self, "max_cache_len", None)
        except ValueError:
            # 空 DynamicCache 上访问 max_cache_len 时，
            # 新版 transformers 可能因为内部列表为空直接抛 ValueError。
            # 这类场景本质上等价于“当前没有最大 cache 长度约束信息”。
            max_cache_len = None
        if max_cache_len is not None:
            if isinstance(max_cache_len, (int, float)) and max_cache_len <= 0:
                return None
            return max_cache_len
        get_max_cache_shape = getattr(self, "get_max_cache_shape", None)
        if callable(get_max_cache_shape):
            max_shape = get_max_cache_shape()
            if isinstance(max_shape, (tuple, list)) and len(max_shape) > 0:
                candidate = max_shape[-1]
                if isinstance(candidate, (int, float)) and candidate <= 0:
                    return None
                return candidate
            if isinstance(max_shape, (int, float)) and max_shape <= 0:
                return None
            return max_shape
        return None

    DynamicCache.get_max_length = _compat_get_max_length

if (
    DynamicCache is not None
    and not hasattr(DynamicCache, "get_usable_length")
):
    # 兼容旧版 cache 接口：
    # get_usable_length(new_seq_length, layer_idx=None) 语义上返回
    # “在当前 cache 和最大 cache 限制下，已有多少历史 token 仍可继续复用”。
    def _compat_get_usable_length(self, new_seq_length, layer_idx=None):
        # Phi-3 会在同一次 forward 中按 layer 逐层查询 cache 长度。
        # 如果这里只读“全局长度”，那么前一层刚写入 cache 后，
        # 后续尚未写入的层也会被误判成“已经有同样长的 past”，
        # 从而把 kv_seq_len 错算成当前序列长度的两倍。
        #
        # 因此这里必须优先读取“当前 layer 自己”的 cache 长度。
        if layer_idx is None:
            current_length = self.get_seq_length()
        else:
            try:
                current_length = self.get_seq_length(layer_idx)
            except TypeError:
                current_length = self.get_seq_length()
        max_length = self.get_max_length() if hasattr(self, "get_max_length") else None
        if max_length is not None and current_length + new_seq_length > max_length:
            return max(max_length - new_seq_length, 0)
        return current_length

    DynamicCache.get_usable_length = _compat_get_usable_length

PROJECT_MODEL_CACHE = os.path.abspath(MODEL_CACHE)
PROJECT_HF_HOME = os.path.join(PROJECT_MODEL_CACHE, "_hf_home")
PROJECT_TMPDIR = os.path.join(PROJECT_MODEL_CACHE, "_tmp")

os.makedirs(PROJECT_MODEL_CACHE, exist_ok=True)
os.makedirs(PROJECT_HF_HOME, exist_ok=True)
os.makedirs(PROJECT_TMPDIR, exist_ok=True)

# 把模型相关缓存和临时文件固定到项目盘，避免再次写爆根分区 `/`。
os.environ.setdefault("MODELSCOPE_CACHE", PROJECT_MODEL_CACHE)
os.environ.setdefault("HF_HOME", PROJECT_HF_HOME)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(PROJECT_HF_HOME, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(PROJECT_HF_HOME, "transformers"))
os.environ.setdefault("TMPDIR", PROJECT_TMPDIR)

# 全局单例对象：
# - _tokenizer: 所有实验共享一个 tokenizer
# - _model:     仅 pure inference 路径复用的共享底模
_tokenizer, _model = None, None
_independent_device_cursor = 0
_runtime_device_summary_printed = False
# 独立 GSPO 副本至少需要比较宽松的可用显存。
# 这里留出一部分余量，避免刚好卡在加载末尾或第一次 forward 时爆掉。
MIN_FREE_GB_FOR_GSPO_REPLICA = 16.0

def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

def _parse_forced_device():
    forced_device = os.environ.get("MAS_FORCE_DEVICE", "").strip()
    return forced_device or None

def _validate_runtime_device_request():
    forced_device = _parse_forced_device()
    require_cuda = _env_flag("MAS_REQUIRE_CUDA", default=False)

    if forced_device is not None:
        if forced_device == "cpu":
            if require_cuda:
                raise RuntimeError("MAS_REQUIRE_CUDA=1 但 MAS_FORCE_DEVICE=cpu，配置冲突。")
            return forced_device
        if not forced_device.startswith("cuda"):
            raise ValueError(f"不支持的 MAS_FORCE_DEVICE: {forced_device}")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "MAS_FORCE_DEVICE 指向 CUDA，但当前 Python 环境里 torch.cuda.is_available()=False。"
            )
        if ":" in forced_device:
            try:
                device_idx = int(forced_device.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"无法解析 MAS_FORCE_DEVICE={forced_device}") from exc
            device_count = torch.cuda.device_count()
            if device_idx < 0 or device_idx >= device_count:
                raise RuntimeError(
                    f"MAS_FORCE_DEVICE={forced_device} 超出可见 GPU 范围，当前仅有 {device_count} 张可见 GPU。"
                )
        return forced_device

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "MAS_REQUIRE_CUDA=1，但当前 Python 环境里 torch.cuda.is_available()=False。"
            " 请确认是在有 GPU 的节点上运行，并且 CUDA/驱动/torch 安装正确。"
        )
    return None

_forced_device = _validate_runtime_device_request()

def pick_best_device():
    if _forced_device is not None:
        return _forced_device
    if not torch.cuda.is_available():
        return "cpu"

    best_idx = 0
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        with torch.cuda.device(idx):
            free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes > best_free:
            best_idx = idx
            best_free = free_bytes
    return f"cuda:{best_idx}"

def pick_independent_device():
    """
    给独立 GSPO agent 选择设备。

    和 pick_best_device() 不同，这里更偏向“轮转分配”：
    - three_layer / dual_gspo 往往会连续创建多个独立副本
    - 如果每次都只看瞬时空闲显存，多个副本仍可能连续挤到同一张卡
    - 这里先按当前空闲显存给 GPU 排序，再按轮转顺序取下一张，
      更容易把 pi0 / pi1 分散到不同设备
    """
    global _independent_device_cursor

    if _forced_device is not None:
        return _forced_device
    if not torch.cuda.is_available():
        return "cpu"

    free_by_idx = []
    for idx in range(torch.cuda.device_count()):
        with torch.cuda.device(idx):
            free_bytes, _ = torch.cuda.mem_get_info()
        free_by_idx.append((free_bytes, idx))

    free_by_idx.sort(reverse=True)
    min_free_bytes = int(MIN_FREE_GB_FOR_GSPO_REPLICA * (1024 ** 3))
    viable_indices = [idx for free_bytes, idx in free_by_idx if free_bytes >= min_free_bytes]

    # 优先只在“显存明显足够”的 GPU 里轮转。
    # 如果一张都没有，再退回到所有 GPU 里选当前最优的一张。
    ordered_indices = viable_indices if viable_indices else [idx for _, idx in free_by_idx]
    target_idx = ordered_indices[_independent_device_cursor % len(ordered_indices)]
    _independent_device_cursor += 1
    return f"cuda:{target_idx}"

_device = pick_best_device()

def build_runtime_device_summary():
    summary = {
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "mas_require_cuda": _env_flag("MAS_REQUIRE_CUDA", default=False),
        "mas_force_device": _forced_device or "<auto>",
        "selected_default_device": _device,
    }
    if torch.cuda.is_available():
        gpu_names = []
        for idx in range(torch.cuda.device_count()):
            try:
                gpu_names.append(torch.cuda.get_device_name(idx))
            except Exception:
                gpu_names.append(f"cuda:{idx}")
        summary["gpu_names"] = gpu_names
    return summary

def print_runtime_device_summary_once():
    global _runtime_device_summary_printed
    if _runtime_device_summary_printed:
        return
    print(
        "[runtime_device] " + json.dumps(build_runtime_device_summary(), ensure_ascii=False),
        flush=True,
    )
    _runtime_device_summary_printed = True

def resolve_gspo_train_dtype(device):
    """
    解析 GSPO 独立训练副本应使用的数据类型。

    这里单独做一个函数，而不是把判断逻辑散落在 get_gpt2() 里，
    是为了让“为什么 GSPO 用这个精度”更容易读懂。
    """
    preference = str(GSPO_TRAIN_DTYPE).lower()

    if preference == "float16":
        return torch.float16
    if preference == "float32":
        return torch.float32
    if preference == "bfloat16":
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # 如果用户强制要求 bf16，但当前设备不支持，
        # 我们退回 fp32，优先保证训练稳定性。
        return torch.float32
    if preference == "auto":
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32

    raise ValueError(f"不支持的 GSPO_TRAIN_DTYPE: {GSPO_TRAIN_DTYPE}")

def get_model_load_kwargs(device=None):
    resolved_device = device or _device
    kwargs = {
        "cache_dir": MODEL_CACHE,
        "revision": "master",
        "trust_remote_code": True,
    }
    if str(resolved_device).startswith("cuda"):
        kwargs["torch_dtype"] = torch.float16
    return kwargs

def get_model_context_window(model, tokenizer):
    candidates = [
        getattr(model.config, "n_positions", None),
        getattr(model.config, "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, int) and 0 < candidate < 1000000:
            return candidate
    return 1024

def get_prompt_token_limit(model, tokenizer):
    context_window = get_model_context_window(model, tokenizer)
    # 为生成阶段预留 MAX_NEW_TOKENS 的空间，避免把总长度上限误用成提示词上限。
    return max(1, context_window - MAX_NEW_TOKENS)

def tokenizer_has_chat_template(tokenizer):
    template = getattr(tokenizer, "chat_template", None)
    return bool(template and str(template).strip())

def render_prompts_for_generation(prompts, tokenizer):
    """
    对带 chat template 的 instruct 模型，优先包装成 system+user 消息再生成。
    """
    prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
    if not tokenizer_has_chat_template(tokenizer):
        return prompt_list

    rendered = []
    for prompt in prompt_list:
        messages = []
        if CHAT_SYSTEM_PROMPT:
            messages.append({"role": "system", "content": CHAT_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": prompt})
        rendered.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return rendered

def build_generation_inputs(prompts, tokenizer, model):
    prompt_list = render_prompts_for_generation(prompts, tokenizer)
    prompt_token_limit = get_prompt_token_limit(model, tokenizer)
    batched = len(prompt_list) > 1
    return tokenizer(
        prompt_list if batched else prompt_list[0],
        return_tensors="pt",
        padding=batched,
        truncation=True,
        max_length=prompt_token_limit,
    )

def _ensure_tokenizer():
    global _tokenizer
    print_runtime_device_summary_once()
    if _tokenizer is None:
        load_kwargs = get_model_load_kwargs()
        _tokenizer = AutoTokenizer.from_pretrained(
            GPT2_MODEL_SCOPE,
            **load_kwargs,
        )
        _tokenizer.truncation_side = "left"
        _tokenizer.padding_side = "left"
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer

def _load_shared_model():
    global _model
    tokenizer = _ensure_tokenizer()
    if _model is None:
        load_kwargs = get_model_load_kwargs()
        _model = AutoModelForCausalLM.from_pretrained(
            GPT2_MODEL_SCOPE,
            **load_kwargs,
        )
        _model.config.pad_token_id = _tokenizer.pad_token_id
        _model.to(_device)
        _model.eval()
    return tokenizer, _model, _device

def _load_independent_model():
    """
    为 GSPO agent 单独加载一份模型副本。

    和旧实现相比，这里不再：
    - 先把共享底模放到 GPU
    - 再在 GPU 上 deepcopy

    改成“直接从缓存权重加载独立副本到当前最空的设备”，
    好处是：
    - 避免额外常驻一个共享 GPU 底模
    - 更容易把多个 GSPO agent 分散到不同 GPU
    - 降低 three_layer / dual_gspo 在初始化阶段 OOM 或 queue stall 的概率
    """
    tokenizer = _ensure_tokenizer()
    runtime_device = pick_independent_device()
    train_dtype = resolve_gspo_train_dtype(runtime_device)
    print(
        f"[runtime_device] loading_independent_model runtime_device={runtime_device} train_dtype={train_dtype}",
        flush=True,
    )
    load_kwargs = get_model_load_kwargs("cpu")
    load_kwargs["torch_dtype"] = train_dtype
    model = AutoModelForCausalLM.from_pretrained(
        GPT2_MODEL_SCOPE,
        **load_kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    # 独立 GSPO 策略默认常驻 CPU，真正 rollout / update 时再临时迁移到目标 GPU。
    # 否则 2 agents x (pi0, pi1) 在当前机器上会在初始化阶段直接 OOM。
    model.to(device="cpu", dtype=train_dtype)
    model.eval()
    return tokenizer, model, runtime_device

def get_gpt2(independent=False):
    """
    懒加载当前配置的魔塔因果语言模型。

    - independent=False: 返回全局共享单例，适合纯推理基线。
    - independent=True:  基于共享底模深拷贝出独立副本，适合 GSPO 多 agent 训练，
                         避免不同 agent 共享同一份会被原地更新的活跃策略。
    """
    if independent:
        return _load_independent_model()
    return _load_shared_model()

def generate_response(prompt, model=None, tokenizer=None, device=None):
    """GPT2生成推理（固定参数）"""
    if model is None or tokenizer is None or device is None:
        tokenizer, model, device = get_gpt2()

    inputs = build_generation_inputs(prompt, tokenizer, model)
    input_len = inputs["input_ids"].shape[1]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_ids = outputs[0][input_len:]
    res = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return res
