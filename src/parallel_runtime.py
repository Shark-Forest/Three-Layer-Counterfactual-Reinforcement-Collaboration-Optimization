import copy
import importlib
import os
import queue
import random
import sys
import threading
import traceback

import numpy as np
import torch
import torch.multiprocessing as mp


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _normalize_device(device_value):
    value = str(device_value).strip()
    if value == "":
        raise ValueError("设备映射里出现空设备。")
    if value == "cpu":
        return value
    if value.startswith("cuda:"):
        return value
    return f"cuda:{int(value)}"


def build_policy_keys(num_agents):
    keys = []
    for agent_idx in range(num_agents):
        keys.append(f"agent{agent_idx}.pi0")
        keys.append(f"agent{agent_idx}.pi1")
    return keys


def parse_policy_device_map(spec, num_agents):
    policy_keys = build_policy_keys(num_agents)
    if not spec:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device_count = torch.cuda.device_count()
            return {
                key: f"cuda:{idx % device_count}"
                for idx, key in enumerate(policy_keys)
            }
        return {key: "cpu" for key in policy_keys}

    resolved = {}
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"无法解析 MAS_POLICY_DEVICE_MAP 条目: {item}，期望格式为 agent0.pi0:0"
            )
        policy_key, device_value = item.rsplit(":", 1)
        policy_key = policy_key.strip()
        if policy_key not in policy_keys:
            raise ValueError(f"MAS_POLICY_DEVICE_MAP 包含未知策略: {policy_key}")
        resolved[policy_key] = _normalize_device(device_value)

    missing = [key for key in policy_keys if key not in resolved]
    if missing:
        raise ValueError(
            "MAS_POLICY_DEVICE_MAP 缺少策略映射: " + ", ".join(missing)
        )
    return resolved


def _tensor_dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _restore_tensor_dtype(dtype_name):
    return getattr(torch, str(dtype_name))


def _encode_ipc_payload(value):
    if torch.is_tensor(value):
        return {
            "__ipc_tensor__": True,
            "dtype": _tensor_dtype_name(value.dtype),
            "data": value.detach().cpu().tolist(),
        }
    if isinstance(value, dict):
        return {key: _encode_ipc_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_ipc_payload(item) for item in value]
    if isinstance(value, tuple):
        return {
            "__ipc_tuple__": True,
            "items": [_encode_ipc_payload(item) for item in value],
        }
    return value


def _decode_ipc_payload(value):
    if isinstance(value, dict):
        if value.get("__ipc_tensor__"):
            return torch.tensor(
                value["data"],
                dtype=_restore_tensor_dtype(value["dtype"]),
            )
        if value.get("__ipc_tuple__"):
            return tuple(_decode_ipc_payload(item) for item in value["items"])
        return {key: _decode_ipc_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_ipc_payload(item) for item in value]
    return value


class PolicyWorkerClient:
    def __init__(self, policy_key, process, request_queue, response_queue):
        self.policy_key = policy_key
        self.process = process
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._call_lock = threading.Lock()

    def wait_until_ready(self, timeout_s=1800):
        try:
            response = self.response_queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise RuntimeError(f"worker 启动超时: {self.policy_key}") from exc
        if not response.get("ok", False):
            raise RuntimeError(
                f"worker 启动失败: {self.policy_key}\n{response.get('error', '<unknown>')}"
            )
        if response.get("event") != "ready":
            raise RuntimeError(
                f"worker 启动响应异常: {self.policy_key} -> {response}"
            )
        return response

    def call(self, op, **kwargs):
        with self._call_lock:
            self.request_queue.put({"op": op, "kwargs": kwargs})
            response = self.response_queue.get()

        if not response.get("ok", False):
            raise RuntimeError(
                f"worker 执行失败: {self.policy_key} op={op}\n{response.get('error', '<unknown>')}"
            )
        return response

    def shutdown(self):
        if not self.process.is_alive():
            return
        try:
            self.call("shutdown")
        except Exception:
            pass
        self.process.join(timeout=30)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=10)


class ParallelGSPOPolicyProxy:
    def __init__(self, policy_key, client, is_coop):
        self.policy_key = policy_key
        self.client = client
        self.is_coop = bool(is_coop)
        self.last_update = None

    def _sync_last_update(self, response):
        if "last_update" in response:
            self.last_update = copy.deepcopy(response["last_update"])

    def act(self, query_context, prompt_override=None):
        response = self.client.call(
            "act",
            query_context=query_context,
            prompt_override=prompt_override,
        )
        return _decode_ipc_payload(response["result"])

    def sample_candidates(self, query_context, prompt_override=None, selected_idx=0):
        response = self.client.call(
            "sample_candidates",
            query_context=query_context,
            prompt_override=prompt_override,
            selected_idx=selected_idx,
        )
        self._sync_last_update(response)
        return _decode_ipc_payload(response["result"])

    def sample_text(self, query_context, prompt_override=None):
        response = self.client.call(
            "sample_text",
            query_context=query_context,
            prompt_override=prompt_override,
        )
        return response["result"]

    def sample_text_batch(self, query_contexts, prompt_override=None):
        response = self.client.call(
            "sample_text_batch",
            query_contexts=query_contexts,
            prompt_override=prompt_override,
        )
        return _decode_ipc_payload(response["result"])

    def update_from_cached(self, batch, rewards):
        response = self.client.call(
            "update_from_cached",
            batch=_encode_ipc_payload(batch),
            rewards=rewards,
        )
        self._sync_last_update(response)
        return response["result"]

    def prepare_for_teardown(self):
        response = self.client.call("prepare_for_teardown")
        self._sync_last_update(response)

    def save_checkpoint(self, path):
        response = self.client.call("save_checkpoint", path=path)
        self._sync_last_update(response)
        return response["result"]

    def load_checkpoint(self, path):
        response = self.client.call("load_checkpoint", path=path)
        self._sync_last_update(response)
        return response["result"]


def _apply_worker_module_overrides(overrides, policy=None):
    if not overrides:
        return {}

    applied = {}
    for name, value in dict(overrides).items():
        copied_value = copy.deepcopy(value)
        applied[name] = copied_value
        for module_name in ("src.config", "src.model_loader", "src.gspo_verl", "run_all"):
            module = sys.modules.get(module_name)
            if module is not None:
                setattr(module, name, copied_value)

    if policy is not None:
        gspo_module = sys.modules.get("src.gspo_verl")
        if gspo_module is not None:
            prompt_name = "PI0_PROMPT" if policy.is_coop else "PI1_PROMPT"
            if hasattr(gspo_module, prompt_name):
                policy.prompt_template = getattr(gspo_module, prompt_name)
        if "GSPO_NUM_CANDIDATES" in applied:
            policy.G = int(applied["GSPO_NUM_CANDIDATES"])

    return applied


class ThreeLayerWorkerController:
    def __init__(self, project_root, num_agents, device_map, base_seed):
        self.project_root = os.path.abspath(project_root)
        self.num_agents = num_agents
        self.device_map = dict(device_map)
        self.base_seed = base_seed
        self.ctx = mp.get_context("spawn")
        self.clients = {}

        worker_specs = []
        for agent_idx in range(num_agents):
            worker_specs.append((f"agent{agent_idx}.pi0", True))
            worker_specs.append((f"agent{agent_idx}.pi1", False))

        for worker_idx, (policy_key, is_coop) in enumerate(worker_specs):
            device = self.device_map[policy_key]
            request_queue = self.ctx.Queue()
            response_queue = self.ctx.Queue()
            process = self.ctx.Process(
                target=_policy_worker_main,
                args=(
                    self.project_root,
                    policy_key,
                    is_coop,
                    device,
                    self.base_seed + worker_idx,
                    request_queue,
                    response_queue,
                ),
            )
            process.start()
            client = PolicyWorkerClient(
                policy_key=policy_key,
                process=process,
                request_queue=request_queue,
                response_queue=response_queue,
            )
            ready = client.wait_until_ready()
            print(
                "[three_layer_parallel] worker_ready "
                f"policy={policy_key} "
                f"requested_device={ready.get('device')} "
                f"env_force_device={ready.get('env_force_device')} "
                f"policy_device={ready.get('policy_device')} "
                f"current_cuda_device={ready.get('current_cuda_device')} "
                f"max_new_tokens={ready.get('max_new_tokens')} "
                f"schema={ready.get('middle_action_schema')}",
                flush=True,
            )
            self.clients[policy_key] = client

    def build_agents(self):
        agents = []
        for agent_idx in range(self.num_agents):
            agents.append(
                {
                    "pi0": ParallelGSPOPolicyProxy(
                        f"agent{agent_idx}.pi0",
                        self.clients[f"agent{agent_idx}.pi0"],
                        True,
                    ),
                    "pi1": ParallelGSPOPolicyProxy(
                        f"agent{agent_idx}.pi1",
                        self.clients[f"agent{agent_idx}.pi1"],
                        False,
                    ),
                    "pi2": None,
                }
            )
        return agents

    def shutdown(self):
        for policy_key in reversed(build_policy_keys(self.num_agents)):
            client = self.clients.get(policy_key)
            if client is None:
                continue
            client.shutdown()

    def apply_module_overrides(self, overrides):
        if not overrides:
            return
        for policy_key in build_policy_keys(self.num_agents):
            client = self.clients.get(policy_key)
            if client is None:
                continue
            client.call("apply_module_overrides", overrides=copy.deepcopy(overrides))


def create_three_layer_parallel_controller(project_root, num_agents):
    base_seed = _env_int("MAS_GLOBAL_SEED", 1234)
    device_map = parse_policy_device_map(
        os.environ.get("MAS_POLICY_DEVICE_MAP", "").strip(),
        num_agents,
    )
    print(
        f"[three_layer_parallel] mode=three_layer_workers device_map={device_map} base_seed={base_seed}",
        flush=True,
    )
    return ThreeLayerWorkerController(
        project_root=project_root,
        num_agents=num_agents,
        device_map=device_map,
        base_seed=base_seed,
    )


def _set_worker_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _policy_worker_main(
    project_root,
    policy_key,
    is_coop,
    device,
    seed,
    request_queue,
    response_queue,
):
    try:
        os.environ["MAS_FORCE_DEVICE"] = device
        os.environ["MAS_KEEP_POLICY_ON_DEVICE"] = "1"
        os.environ.setdefault("MAS_REQUIRE_CUDA", "1")
        if project_root not in sys.path:
            sys.path.append(project_root)

        stale_module_names = []
        for module_name in list(sys.modules.keys()):
            if module_name == "src.parallel_runtime":
                continue
            if module_name == "src" or module_name.startswith("src."):
                stale_module_names.append(module_name)
            elif module_name in {
                "run_all",
                "run_full_suite_4gpu",
                "run_monitored_three_layer_50",
                "run_three_layer_only_4gpu",
            }:
                stale_module_names.append(module_name)
        for module_name in stale_module_names:
            sys.modules.pop(module_name, None)
        importlib.invalidate_caches()

        from src.gspo_verl import GSPOAgentPolicy

        if str(device).startswith("cuda:"):
            torch.cuda.set_device(int(str(device).split(":", 1)[1]))
        _set_worker_seed(seed)
        policy = GSPOAgentPolicy(is_coop=is_coop)
        response_queue.put(
            {
                "ok": True,
                "event": "ready",
                "device": device,
                "env_force_device": os.environ.get("MAS_FORCE_DEVICE"),
                "policy_device": str(getattr(policy, "device", "<unknown>")),
                "current_cuda_device": (
                    torch.cuda.current_device()
                    if torch.cuda.is_available()
                    else None
                ),
                "max_new_tokens": getattr(sys.modules.get("src.gspo_verl"), "MAX_NEW_TOKENS", None),
                "middle_action_schema": getattr(
                    sys.modules.get("src.gspo_verl"),
                    "THREE_LAYER_MIDDLE_ACTION_SCHEMA",
                    None,
                ),
            }
        )
    except Exception:
        response_queue.put(
            {
                "ok": False,
                "event": "ready",
                "error": traceback.format_exc(),
            }
        )
        return

    while True:
        request = request_queue.get()
        op = request.get("op")
        kwargs = _decode_ipc_payload(request.get("kwargs", {}))
        try:
            if op == "shutdown":
                response_queue.put({"ok": True, "result": None})
                break
            if op == "act":
                result = policy.act(**kwargs)
            elif op == "sample_candidates":
                result = policy.sample_candidates(**kwargs)
            elif op == "sample_text":
                result = policy.sample_text(**kwargs)
            elif op == "sample_text_batch":
                result = policy.sample_text_batch(**kwargs)
            elif op == "update_from_cached":
                result = policy.update_from_cached(**kwargs)
            elif op == "prepare_for_teardown":
                policy.prepare_for_teardown()
                result = None
            elif op == "apply_module_overrides":
                result = _apply_worker_module_overrides(
                    kwargs.get("overrides"),
                    policy=policy,
                )
            elif op == "save_checkpoint":
                result = policy.save_checkpoint(**kwargs)
            elif op == "load_checkpoint":
                result = policy.load_checkpoint(**kwargs)
            else:
                raise ValueError(f"不支持的 worker 操作: {op}")

            response_queue.put(
                {
                    "ok": True,
                    "result": _encode_ipc_payload(result),
                    "last_update": copy.deepcopy(getattr(policy, "last_update", None)),
                }
            )
        except Exception:
            response_queue.put(
                {
                    "ok": False,
                    "error": traceback.format_exc(),
                }
            )
