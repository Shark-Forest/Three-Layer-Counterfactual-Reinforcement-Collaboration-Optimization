# Experiment Settings

本文件记录 `three_level_experiment` 当前默认实验设置，作为推送 GitHub 前的仓库内说明。

## 1. 数据设置

- 默认使用 GSM8K 全量数据。
- 官方 `train` split 会再次划分为 `train / val`。
- 划分比例：`0.8 / 0.2`。
- 随机种子：`42`。
- 官方 `test` split 保持为最终测试集，不参与这一步 `8:2` 划分。
- 相关配置：
  - `TRAIN_VAL_TOTAL_LIMIT = None`
  - `TEST_SAMPLE_LIMIT = None`
  - `TRAIN_RATIO = 0.8`
  - `DATA_SPLIT_SEED = 42`

## 2. 轮次与模型

- 固定多轮交互轮数：`NUM_ROUNDS = 5`
- 主 LLM：`LLM-Research/Phi-3-mini-4k-instruct`
- 默认生成长度：`MAX_NEW_TOKENS = 128`
- 默认采样温度：`TEMPERATURE = 0.3`
- 生成仍使用采样，不是贪心解码。

## 3. 八组实验

### 3.1 只测试、不训练

1. `单LLM`
   - 只在 `test` 上评估。
   - 不更新任何参数。

2. `双LLM轮询`
   - 只在 `test` 上评估。
   - 不更新任何参数。

### 3.2 需要训练的实验

3. `单LLM GSPO`
   - `train` 训练
   - `val` 验证
   - `test` 最终评估

4. `双LLM GSPO轮询`
   - `train` 训练
   - `val` 验证
   - `test` 最终评估

5. `中间层策略+GSPO`
   - `train` 训练
   - `val` 验证
   - `test` 最终评估

6. `中间层策略+GSPO（无沉默）`
   - `train` 训练
   - `val` 验证
   - `test` 最终评估

7. `全量策略`
   - `train` 训练
   - `val` 验证
   - `test` 最终评估

8. `全量策略（中层无沉默）`
   - `train` 训练
   - `val` 验证
   - `test` 最终评估

## 4. 三层结构

### 4.1 外层调度器

- 仅在 `全量策略` 与 `全量策略（中层无沉默）` 中启用。
- 状态：`phase`
- 动作：选择哪个 agent 出场
- 算法：`CFR / regret matching`
- 默认外层 agent 数：`NUM_AGENTS = 2`

### 4.2 中间层动作策略

- 动作空间：
  - `silent`
  - `comment`
  - `answer`
- 无沉默消融版只保留：
  - `comment`
  - `answer`
- 算法：`CFR / regret matching`
- 当前直接使用 raw regret-matching：
  - 正遗憾归一化为当前策略
  - 若所有正遗憾都为 `0`，则在允许动作上均匀分布
  - 不再额外施加 `search` 状态下的中层动作概率约束
- 在不启用外层调度器的实验里：
  - 仍然保留多个 agent bundle
  - 但不学习调度策略
  - 改为按轮次 round-robin 轮询 agent

### 4.3 最内层文本策略

- `pi0`：comment policy
- `pi1`：answer policy
- 算法：`GSPO`

## 5. 当前 answer 进入真实轨迹的规则

### 5.1 单LLM / 双LLM轮询

- 每轮只生成一次。
- `answer` 直接由这次生成文本经过 `extract_pred_num()` 抽取。

### 5.2 单LLM GSPO / 双LLM GSPO轮询

- 一次会采样一批候选。
- 真实轨迹默认使用该批次的第一个候选。
- 更新时会利用整批候选做 GSPO。

### 5.3 共享策略栈实验

适用于：

- `中间层策略+GSPO`
- `中间层策略+GSPO（无沉默）`
- `全量策略`
- `全量策略（中层无沉默）`

当前规则已经统一成训练/验证/测试一致：

1. `pi1` 先采样一批 answer 候选。
2. 如果这一批里存在可解析数字答案，只在这些可解析候选里选择 `verifier` 分数最高的那个。
3. 如果整批都不可解析，则退回到全候选里选择 `verifier` 分数最高的那个。
4. 选中的 `selected_text` 再交给 verifier 判断是否允许接管当前 incumbent。

这意味着：

- 训练轨迹和推理轨迹共用同一套选答逻辑。
- `train` 与 `val/test` 的差别只在于是否做参数更新，而不在于真实轨迹如何选 answer。

## 6. verifier 设置

- verifier 使用单 scorer 结构。
- `keep_prob = scorer(question, incumbent_answer)`
- `accept_prob = scorer(question, candidate_answer)`
- 接管条件：`accept_prob > keep_prob`
- verifier 不直接看 comment，只看题目和 answer utterance。

### 6.1 verifier embedding 来源

- 优先使用项目目录下的本地缓存。
- 如果本地缓存不存在，则回退到 `MAS_VERIFIER_EMBED_MODEL` 指定的模型。
- 当前默认回退模型：`gpt2`

## 7. 奖励定义

### 7.1 基础答案质量

- `reward_from_pred(pred, gt)` 为混合奖励：
  - exact match 主导
  - dense reward 辅助
  - 不可解析答案有惩罚

### 7.2 comment value

- `with comment` 的下一次 answer 的 `r`
- 减去当前状态下 `silent` 的基线 reward
- 对单个 comment candidate 而言，只继续 rollout `1` 个 next answer
- `silent` 基线也只继续 rollout `1` 个 next answer

### 7.3 answer value

- `这次 answer` 的 reward
- 减去当前状态下 `silent` 的基线 reward

### 7.4 silent value

- 固定记为 `0`
- 解释为 no-op 基线动作的额外价值为 0
- 因此：
  - `comment > 0` 表示“比静默更有帮助”
  - `comment < 0` 表示“这条 comment 还不如不说”
  - `answer > 0` 表示“本轮直接答，比静默后再答更好”

### 7.5 中层反事实估计

- 已选动作：
  - 直接使用真实轨迹里被选中的那个 candidate 的单个 value
- 未选动作：
  - 额外做 counterfactual Monte Carlo 采样
  - `answer`：采样 `1` 个 counterfactual answer batch，并对组内 candidates 的 value 取均值
  - `comment`：采样 `1` 个 counterfactual comment batch，并对组内 candidates 的 value 取均值
  - comment batch 里的每个 candidate 也只继续 rollout `1` 个 next answer
- `silent`：
  - 固定为 `0`
- regret 更新：
  - 对每个备选动作计算 `action_value(other) - action_value(chosen)`

### 7.6 外层 agent value

- 外层 agent `i` 的动作价值定义为：
  - `V_outer(i, s) = Σ_a π_middle(a | i, s) · Q_middle(i, a, s)`
- 含义：
  - 先估计该 agent 在当前状态下的中层三动作 value
  - 再按照该 agent 当前中层策略的动作概率做加权平均
- 已选中的 agent 也使用同一口径，而不是直接用“本轮结束后的 incumbent reward”
- 未选中的 agent 复用同样的中层反事实 Monte Carlo 估计

## 8. 关键默认超参数

### 8.1 GSPO

- `GSPO_LR = 5e-6`
- `GSPO_NUM_CANDIDATES = 4`
- `GSPO_CLIP_EPS = 0.1`
- `GSPO_TRAIN_DTYPE = "bfloat16"`
- `GSPO_COMMENT_EVAL_SAMPLES = 1`
- `GSPO_MAX_GRAD_NORM = 0.5`
- `GSPO_USE_INDEPENDENT_MODEL = True`
- `GSPO_TRAIN_LAST_N_LAYERS = 2`

### 8.2 verifier

- `VERIFIER_LR = 0.1`
- `VERIFIER_PHASE_THRESHOLD = 0.85`
- `VERIFIER_STABILIZE_ALPHA = 0.10`
- `VERIFIER_MAX_LENGTH = 256`
- `VERIFIER_CACHE_SIZE = 256`

### 8.3 search 约束

- `search / stabilize` phase 仍然保留，用于构造中层/外层状态
- 但当前中层动作采样已退化为 raw regret-matching，不再额外使用旧的 search 概率 floor / cap 约束
- 相关旧配置仍保留在 `src/config.py`，便于回溯旧实验

## 9. 路径与缓存

默认都写在仓库内部，便于整体迁移：

- `data/`
- `models/`
- `logs/`
- `plots/`
- `runs/`

支持环境变量覆盖：

- `MAS_LOG_DIR`
- `MAS_PLOT_DIR`
- `MAS_MODEL_CACHE`
- `MAS_DATA_CACHE`
- `MAS_REQUIRE_CUDA`
- `MAS_FORCE_DEVICE`
- `CUDA_VISIBLE_DEVICES`

## 10. 环境与命令

推荐环境：

- Python `3.10`
- conda 环境名：`lcs`
- `transformers==4.41.2`

说明：

- `Phi-3-mini-4k-instruct` 当前远端代码与 `transformers 5.x` 不兼容。
- 因此仓库依赖已固定到 `transformers==4.41.2`，避免 fresh clone 后被解析到 5.x 产生启动错误。

典型命令：

```bash
conda create -n lcs python=3.10 pip
conda run -n lcs pip install -r requirements.txt
conda run -n lcs python run_all.py
```

烟测推荐：

```bash
conda run -n lcs python run_staged_suite.py --sample-counts 5 --num-agents 2 --require-cuda
```

## 11. 本次仓库整理后的已验证结果

验证日期：

- `2026-04-03`

已完成：

- 已创建 conda 环境 `lcs`
- 已执行 `conda run -n lcs pip install -r requirements.txt`
- 已确认 `transformers` 实际版本为 `4.41.2`
- 已确认 `conda run -n lcs python -c "import run_all"` 可以成功导入

已验证的烟测：

1. 入口级烟测

```bash
timeout 120s conda run --no-capture-output -n lcs python -u run_staged_isolated.py --sample-counts 1 --num-agents 1 --experiments single_llm
```

观察到的结果：

- 能正常进入 `run_staged_isolated.py`
- 能正常初始化日志目录
- 能正常加载 GSM8K split
- 能正常进入 `single_llm`
- 能正常开始下载 / 加载 `Phi-3-mini-4k-instruct`
- 未再出现此前 `transformers 5.x` 下的 `rope_scaling['type']` 相关启动错误

2. 函数级烟测

- 已直接验证共享策略栈的 answer 选择逻辑：
  - 若 batch 中存在可解析数字答案，只在可解析候选里选 `verifier` 分数最高者
  - 若 batch 全部不可解析，则在全候选里选 `verifier` 分数最高者
  - 选中的候选会正确写回 `selected_text` / `best_text`

当前限制：

- 本次会话中 `torch.cuda.is_available() == False`
- 因此无法在当前会话里完成一个“贴近正式实验速度”的 GPU 烟测
- `single_llm` 的 1 样本入口烟测在 CPU-only 条件下 120 秒内未跑完，这属于运行环境限制，不是已确认的代码启动错误

正式跑实验时建议：

- 在可见 GPU 的节点上执行
- 使用 `--require-cuda`，避免误在 CPU-only 环境里启动长时间实验

如果只做最小链路验证，也可以先去掉 `--require-cuda` 或改用 isolated runner。
