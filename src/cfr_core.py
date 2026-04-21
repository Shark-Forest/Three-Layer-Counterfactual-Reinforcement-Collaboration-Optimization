import numpy as np


class CFRBehaviorSelector:
    """
    轻量级的状态条件化 CFR / regret-matching 策略。

    设计目标：
    - 外层调度：按状态选择“这轮让哪个 agent 动”
    - 中层动作：按状态选择“静默 / 评论 / 输出答案”

    这里不做完整博弈树 CFR，而是保留实验需要的核心：
    - 每个状态维护一组 regret_sum / strategy_sum
    - 按 regret-matching 取当前策略
    - 用反事实 action value 更新遗憾
    """

    def __init__(self, num_actions: int, default_action: int = 0, fallback_strategy_fn=None):
        self.num_actions = num_actions
        self.default_action = default_action
        self.fallback_strategy_fn = fallback_strategy_fn
        self.regret_sum_by_state = {}
        self.strategy_sum_by_state = {}
        self.iter_by_state = {}

    def _normalize_state_key(self, state_key):
        if state_key is None:
            return ("__default__",)
        if isinstance(state_key, (list, tuple)):
            return tuple(state_key)
        return (state_key,)

    def _ensure_state(self, state_key):
        key = self._normalize_state_key(state_key)
        if key not in self.regret_sum_by_state:
            self.regret_sum_by_state[key] = np.zeros(self.num_actions, dtype=np.float64)
            self.strategy_sum_by_state[key] = np.zeros(self.num_actions, dtype=np.float64)
            self.iter_by_state[key] = 0
        return key

    def _build_mask(self, allowed_actions):
        mask = np.zeros(self.num_actions, dtype=bool)
        if allowed_actions is None:
            mask[:] = True
            return mask
        for action in allowed_actions:
            if 0 <= action < self.num_actions:
                mask[action] = True
        return mask

    def get_current_strategy(self, state_key=None, allowed_actions=None):
        key = self._ensure_state(state_key)
        mask = self._build_mask(allowed_actions)

        pos_regret = np.maximum(self.regret_sum_by_state[key], 0.0) * mask
        total = pos_regret.sum()
        if total > 0:
            return pos_regret / total

        fallback = None
        if self.fallback_strategy_fn is not None:
            fallback = self.fallback_strategy_fn(
                state_key=key,
                allowed_actions=allowed_actions,
                num_actions=self.num_actions,
                default_action=self.default_action,
            )
            if fallback is not None:
                fallback = np.asarray(fallback, dtype=np.float64)
                if fallback.shape != (self.num_actions,):
                    raise ValueError("fallback_strategy_fn 返回的策略维度不正确。")
                fallback = np.maximum(fallback, 0.0) * mask
                total = fallback.sum()
                if total > 0:
                    return fallback / total

        fallback = np.zeros(self.num_actions, dtype=np.float64)
        if mask.any():
            fallback[mask] = 1.0 / mask.sum()
        else:
            fallback[self.default_action] = 1.0
        return fallback

    def get_average_strategy(self, state_key=None, allowed_actions=None):
        key = self._ensure_state(state_key)
        mask = self._build_mask(allowed_actions)
        strategy_sum = self.strategy_sum_by_state[key] * mask
        total = strategy_sum.sum()
        if total > 0:
            return strategy_sum / total
        return self.get_current_strategy(state_key, allowed_actions)

    def get_strategy(self, state_key=None, allowed_actions=None, use_average_strategy=False):
        if use_average_strategy:
            return self.get_average_strategy(state_key, allowed_actions)
        return self.get_current_strategy(state_key, allowed_actions)

    def get_action(
        self,
        state_key=None,
        explore=True,
        allowed_actions=None,
        force_action=None,
        use_average_strategy=False,
    ):
        if force_action is not None:
            return force_action

        strategy = self.get_strategy(
            state_key,
            allowed_actions,
            use_average_strategy=use_average_strategy,
        )
        if explore:
            return int(np.random.choice(self.num_actions, p=strategy))
        return int(np.argmax(strategy))

    def update_regret(self, state_key, chosen_action, action_values, allowed_actions=None):
        key = self._ensure_state(state_key)
        mask = self._build_mask(allowed_actions)
        if not mask.any():
            return

        action_values = np.asarray(action_values, dtype=np.float64)
        chosen_value = action_values[chosen_action]
        regrets = np.zeros(self.num_actions, dtype=np.float64)
        regrets[mask] = action_values[mask] - chosen_value
        self.regret_sum_by_state[key] += regrets
        self.strategy_sum_by_state[key] += self.get_current_strategy(state_key, allowed_actions)
        self.iter_by_state[key] += 1

    def get_total_regret(self):
        if not self.regret_sum_by_state:
            return 0.0
        return float(sum(np.abs(values).sum() for values in self.regret_sum_by_state.values()))

    def get_checkpoint_state(self):
        return {
            "num_actions": int(self.num_actions),
            "default_action": int(self.default_action),
            "regret_sum_by_state": {
                key: np.array(values, copy=True)
                for key, values in self.regret_sum_by_state.items()
            },
            "strategy_sum_by_state": {
                key: np.array(values, copy=True)
                for key, values in self.strategy_sum_by_state.items()
            },
            "iter_by_state": dict(self.iter_by_state),
        }

    def load_checkpoint_state(self, state):
        if state is None:
            return

        num_actions = int(state["num_actions"])
        if num_actions != self.num_actions:
            raise ValueError(
                f"CFR checkpoint 动作维度不匹配：当前={self.num_actions} checkpoint={num_actions}"
            )

        self.default_action = int(state.get("default_action", self.default_action))
        self.regret_sum_by_state = {
            tuple(key): np.asarray(values, dtype=np.float64)
            for key, values in state.get("regret_sum_by_state", {}).items()
        }
        self.strategy_sum_by_state = {
            tuple(key): np.asarray(values, dtype=np.float64)
            for key, values in state.get("strategy_sum_by_state", {}).items()
        }
        self.iter_by_state = {
            tuple(key): int(value)
            for key, value in state.get("iter_by_state", {}).items()
        }
