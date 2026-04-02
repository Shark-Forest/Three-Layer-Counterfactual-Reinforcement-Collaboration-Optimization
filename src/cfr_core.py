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

    def __init__(self, num_actions: int, default_action: int = 0):
        self.num_actions = num_actions
        self.default_action = default_action
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

    def get_action(self, state_key=None, explore=True, allowed_actions=None, force_action=None):
        if force_action is not None:
            return force_action

        strategy = self.get_current_strategy(state_key, allowed_actions)
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
