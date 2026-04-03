# Three-Level MAS Experiment on GSM8K

本项目实现了一个面向 GSM8K 数学题的多智能体实验框架，用于比较从单模型基线到三层策略栈的 8 组实验设置，并分析在多轮交互中，`comment / answer / silence` 行为、外层调度器、在线 verifier、以及内层 GSPO 更新之间的作用关系。

当前代码以 `three_level_experiment` 目录作为实验根目录。模型缓存、数据缓存、日志、图表和 staged 运行输出都默认落在该目录下，便于独立打包成一个 Git 仓库。

## 1. 项目目标

这个项目关注的问题不是“单次生成能否直接答对 GSM8K”，而是：

1. 多轮交互是否能逐步改善当前答案。
2. 不同行为类型是否应该被区别对待。
3. 哪一层负责“选谁来发言”、哪一层负责“发什么类型的话”、哪一层负责“把这句话说好”。
4. 在存在 verifier 的前提下，如何同时训练外层调度、中层动作策略和内层文本策略。

## 2. 八组实验

当前主流程一共运行 8 组实验：

1. `单LLM`
   单模型固定提示词，多轮轮换 comment / answer 风格，但没有参数更新。

2. `双LLM轮询`
   两个固定角色轮询：
   - solver 负责 answer
   - commenter 负责 comment
   没有在线训练。

3. `单LLM GSPO`
   单个可训练 GSPO 智能体，交替执行 answer / comment 风格回合。

4. `双LLM GSPO轮询`
   solver 和 commenter 都是独立的 GSPO policy。
   这个实验里的 `comment` 奖励仍保留旧定义：
   `with comment 的下一次 answer 的 reward - without comment 的下一次 answer 的 reward`

5. `中间层策略+GSPO`
   去掉最外层调度器，只保留：
   - 中层 CFR 动作策略：在 `silent / comment / answer` 中选动作
   - 最内层 GSPO：优化 `pi0(comment)` 和 `pi1(answer)` 的文本输出
   这个实验用于研究“只保留中层和内层”时，系统是否已经足够有效。

6. `中间层策略+GSPO（无沉默）`
   与 `中间层策略+GSPO` 相同，但中层动作空间中移除 `silent`，只保留：
   - `comment`
   - `answer`
   这个实验用于分析“是否必须显式保留沉默动作”。

7. `全量策略`
   完整三层结构：
   - 外层：调度哪个 agent
   - 中层：选 `silent / comment / answer`
   - 内层：生成 comment 或 answer 文本
   同时使用 verifier 决定 phase 和 answer 接管。

8. `全量策略（中层无沉默）`
   与 `全量策略` 相同，但中层动作空间中移除 `silent`，只允许：
   - `comment`
   - `answer`

八组实验的统一入口在 [run_all.py](run_all.py)。

## 3. 三层结构概览

### 3.1 最外层：agent 调度器

最外层负责在当前状态下决定“本轮由哪个 agent 出场”。

- 训练算法：CFR / regret matching
- 状态来源：`phase`
- 动作空间：agent 索引
- 仅在 `全量策略` 中启用

在 `中间层策略+GSPO` 和 `中间层策略+GSPO（无沉默）` 中，这一层被移除；
但系统仍保留多个 agent bundle，并按轮次做 round-robin 轮询，而不是固定只用一个 bundle。

### 3.2 中间层：动作策略

完整动作空间下，中间层负责在已选定 agent 后，从下面三个动作中选一个：

- `silent`
- `comment`
- `answer`

这层也用 CFR / regret matching 更新。

其中：

- `comment` 对应 `pi0`
- `answer` 对应 `pi1`
- `silent` 没有文本输出策略参数，因此不会进入最内层 GSPO 更新

当前中层直接使用最朴素的 regret-matching：

- 每个状态维护 `silent / comment / answer` 的累计遗憾
- 采样时直接按“正遗憾归一化”得到当前策略
- 如果所有正遗憾都为 0，则在允许动作上均匀采样
- 不再额外施加 `search` 状态下的概率 floor / cap 约束

另外还有两组“无沉默”消融：

- `中间层策略+GSPO（无沉默）`
- `全量策略（中层无沉默）`

这两组的中层动作空间只保留 `comment / answer`，不允许选择 `silent`。

### 3.3 最内层：文本策略

最内层是实际生成自然语言内容的策略：

- `pi0`：comment policy
- `pi1`：answer policy

训练算法为 GSPO。它对同一上下文采样多个候选，在组内依据 reward/value 更新策略。

对共享策略栈路径（`中间层策略+GSPO`、`中间层策略+GSPO（无沉默）`、`全量策略`、`全量策略（中层无沉默）`）而言，
`pi1(answer)` 进入真实轨迹的规则现在统一为：

- 先采样一批 answer 候选
- 如果其中存在可解析数字答案，只在这些可解析候选里选择 `verifier` 分数最高的那个
- 如果整批都不可解析，则退回到全候选里选择 `verifier` 分数最高的那个

也就是说，这几条共享策略栈实验里，训练、验证、测试都使用同一套“真实轨迹选答”机制，避免训练/推理不一致。

`单LLM GSPO` 和 `双LLM GSPO轮询` 不带 verifier 候选重排，仍默认使用该次采样 batch 的第一个候选推进真实轨迹。

## 4. verifier 的作用

verifier 实现在 [src/verifier.py](src/verifier.py)。

当前版本是一个轻量在线 verifier：

- backbone：本地 embedding 模型
- head：一个线性 scorer

它承担两个职责：

1. 决定外层状态 `phase`
   - `search`
   - `stabilize`

2. 在共享策略栈实验中，为 answer 候选提供打分，并决定被选中的 answer 是否可以接管当前 incumbent

当前逻辑下：

- `keep_prob = scorer(question, incumbent_answer)`
- `accept_prob = scorer(question, candidate_answer)`
- 真实轨迹里先按 `accept_prob` 选择候选 answer
- 如果 `accept_prob > keep_prob`，则候选答案允许接管 incumbent

也就是说，verifier 不直接看 comment，而是只看“题目 + 答案 utterance 本身”。

## 5. 八组实验的训练与评估流程

### 5.1 统一数据划分

GSM8K 现在统一采用：

- 官方 `train` 再划分成 `train / val`
- 划分比例：`8 : 2`
- 官方 `test` 保持为最终测试集

对应实现见 [src/data_loader.py](src/data_loader.py#L120)。

### 5.2 哪些实验会训练参数

会在线更新参数的有 6 组：

- `单LLM GSPO`
- `双LLM GSPO轮询`
- `中间层策略+GSPO`
- `中间层策略+GSPO（无沉默）`
- `全量策略`
- `全量策略（中层无沉默）`

它们统一采用：

1. `train` 上训练
2. `val` 上只验证，不更新
3. `test` 上最终评估

其中，共享策略栈路径在 `train / val / test` 三个阶段都共用同一套 answer 进入真实轨迹的选择规则；差别只在于 `train` 会额外做 CFR / GSPO / verifier 参数更新，而 `val / test` 不更新。

### 5.3 哪些实验只做测试

不训练参数的基线实验有 2 组：

- `单LLM`
- `双LLM轮询`

这两组直接在 `test` 上评估。

## 6. 当前 reward / value 定义

### 6.1 基础答案质量 `r`

底层答案质量由 `reward_from_pred(pred, gt)` 给出，定义在 [run_all.py](run_all.py#L339)。

这是一个混合奖励：

- exact match 为主
- dense reward 为辅
- 不可解析答案有缺失惩罚

因此：

- 完全答对的答案 reward 最高
- 答错但数值接近真值，仍有细粒度区分

### 6.2 双LLM GSPO轮询中的 comment 奖励

`双LLM GSPO轮询` 里的 comment 奖励保留旧定义：

`with comment 的下一次 answer 的 reward - without comment 的下一次 answer 的 reward`

这里的 `with/without` 都只继续 rollout 1 个虚拟 answer，
不会在每个 comment candidate 下再展开一整组 answer candidates。

对应实现见 [run_all.py](run_all.py#L1198) 和 [run_all.py](run_all.py#L2463)。

### 6.3 中间层策略+GSPO / 中间层策略+GSPO（无沉默） / 全量策略 / 全量策略（中层无沉默）中的 value 定义

这四条共享策略栈路径的 value 定义如下。

#### answer 的 value

`answer` 的中层 value 定义为：

`这次 answer 的 reward - 当前状态下 silent 的基线 reward`

其中 silent 基线是：

`B(s) = E[next_answer_reward | 当前状态 s 下本轮选择 silent]`

实现见 [run_all.py](run_all.py#L1277)。

#### comment 的 value

`comment` 的中层 value 定义为：

`with comment 的下一次 answer 的 r - 当前状态下 silent 的基线 reward`

也就是说，这里仍然是“comment 是否提升后续 answer”；
只是现在 answer 和 comment 都统一减去同一个 silent 基线，进入同一坐标系里比较。

这里的虚拟 rollout 也都只取 1 次：

- 一个 comment candidate 只继续生成 1 个 next answer
- `silent` 基线也只继续生成 1 个 next answer

实现见 [run_all.py](run_all.py#L1436)。

#### silent 的 value

`silent` 的 value 现在固定定义为 `0`。

含义是：

- `silent` 被当作 no-op 基线动作
- `comment` 的 value 继续表示“比不说话额外带来多少帮助”
- `answer` 的 value 表示“这次直接答，比本轮静默后再答好多少”

这样中层的 regret 比较就退化成最直接的三动作 value 比较，不再混入额外 rollout 基线。

在两组“无沉默”实验中，中层动作空间不包含 `silent`，因此不会对 `silent` 做选择或更新；但其余 value 和反事实遗憾的定义与共享策略栈保持一致。

### 6.4 中层 CFR 的反事实遗憾

中层三动作的反事实比较使用上述 value：

- 已选动作：
  - 使用这轮真实采样里真正被选中的那个 candidate 的 value
- 未选动作：
  - 额外做 counterfactual Monte Carlo 采样
  - `answer`：采样 1 个 counterfactual answer batch，并对组内 candidates 的 value 取均值
  - `comment`：采样 1 个 counterfactual comment batch，并对组内 candidates 的 value 取均值
  - 其中 comment 单个 candidate 的 value 本身是“该 comment 下 rollout 1 个 next answer 的 reward 减去当前 silent 基线”
- `silent` 固定使用 `0`
- regret 由 `action_value(other) - action_value(chosen)` 形成

因此当前实现里：

- 真实轨迹：comment / answer 仍然各自产生一组 candidates
- 虚拟 rollout：每条 comment 轨迹只继续产生 1 个 answer
- 反事实动作价值：只采样 1 个 counterfactual batch，不再做 batch 均值外再套一层 batch 均值

当前中层的动作采样也直接使用 raw regret-matching 策略，不再对 `search` 状态施加额外的动作概率约束。

对应更新逻辑见 [run_all.py](run_all.py#L1941)。

### 6.5 外层调度器的价值

外层每个 agent 的价值定义为：

`V_outer(i, s) = Σ_a π_middle(a | i, s) · Q_middle(i, a, s)`

也就是：

- 先估计该 agent 在当前状态下的中层三动作 value
- 再按这个 agent 当前的中层策略概率，对 `silent / comment / answer` 做加权平均
- 已选中的 agent 也使用同一口径
- 其中被真实选中的中层动作仍复用这轮真实轨迹里的单个 realized value
- 未被真实选中的中层动作则使用上面 6.4 的 counterfactual Monte Carlo 均值

因此外层比较的是：

- “如果当前 phase 下换另一个 agent 出场，它在自己当前中层策略下的期望动作价值是多少”

而不再直接比较“本轮结束后 incumbent answer 的 reward”。

实现见 [run_all.py](run_all.py#L1476) 和 [run_all.py](run_all.py#L1941)。

## 7. 代码结构

当前建议直接把本目录作为 Git 仓库根目录。核心结构如下：

```text
three_level_experiment/
├── README.md
├── .gitignore
├── requirements.txt
├── run_all.py
├── run_staged_suite.py
├── run_staged_isolated.py
├── run_monitored_three_layer_50.py
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── model_loader.py
│   ├── gspo_verl.py
│   ├── cfr_core.py
│   ├── verifier.py
│   ├── metrics_logger.py
│   └── plotter.py
├── data/      # 数据缓存，默认忽略
├── models/    # 模型缓存，默认忽略
├── logs/      # 实验日志，默认忽略
├── plots/     # 结果图，默认忽略
└── runs/      # staged 输出，默认忽略
```

其中：

- [run_all.py](run_all.py)
  主入口，完整运行 8 组实验

- [run_staged_suite.py](run_staged_suite.py)
  用较小样本分阶段跑整套实验，适合 smoke / 快速验证

- [run_staged_isolated.py](run_staged_isolated.py)
  把实验拆到独立子进程中分别运行，适合做更稳的隔离式调试

- [src/config.py](src/config.py)
  集中管理：
  - 数据划分
  - 模型名
  - 生成参数
  - GSPO 超参数
  - CFR 超参数
  - verifier 超参数
  - 缓存目录和输出目录

## 8. 环境依赖

依赖见 [requirements.txt](requirements.txt)。

核心依赖包括：

- `torch`
- `transformers`
- `datasets`
- `modelscope`
- `pandas`
- `numpy`
- `matplotlib`
- `tqdm`

建议使用 Python 3.10 或 3.11。

## 9. 安装方式

在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

如果你要在 GPU 上运行，需确保：

- `torch.cuda.is_available()` 为 `True`
- 驱动、CUDA、PyTorch 版本匹配

## 10. 模型与缓存

项目会把大部分缓存固定写到项目目录里，而不是系统全局缓存：

- 数据缓存：`data/`
- 模型缓存：`models/`
- 日志：`logs/`
- 图表：`plots/`
- staged 输出：`runs/`

这样做的好处是：

1. 目录可整体迁移
2. 不容易把系统盘缓存写乱
3. 更适合按项目管理实验

第一次运行时，模型和数据可能会下载或解包到这些目录下。

## 11. 快速开始

### 11.1 跑完整 8 组实验

```bash
python run_all.py
```

默认流程：

1. 初始化日志
2. 加载 GSM8K 并切分 `train / val / test`
3. 依次运行 8 组实验
4. 生成对比图

### 11.2 跑 staged 小样本整套实验

```bash
python run_staged_suite.py --sample-counts 5 10 20 50 --num-agents 2
```

如果要求必须看到 CUDA：

```bash
python run_staged_suite.py --sample-counts 5 --require-cuda
```

### 11.3 跑 isolated staged 实验

```bash
python run_staged_isolated.py --sample-counts 5 --experiments single_llm dual_gspo middle_layer middle_layer_no_silent three_layer three_layer_no_silent
```

这个脚本会把每个实验放到独立子进程里执行，便于隔离显存状态和定位问题。

可选实验 ID 一共 8 个：

- `single_llm`
- `polling_two_llms`
- `single_gspo`
- `dual_gspo`
- `middle_layer`
- `middle_layer_no_silent`
- `three_layer`
- `three_layer_no_silent`

## 12. 主要输出文件

### 12.1 日志 CSV

默认写到 `logs/`：

- `single_llm.csv`
- `polling_two_llms.csv`
- `single_gspo.csv`
- `dual_gspo.csv`
- `middle_layer.csv`
- `middle_layer_no_silent.csv`
- `three_layer.csv`
- `three_layer_no_silent.csv`

### 12.2 图表

默认写到 `plots/`：

- `accuracy_comparison.png`
- `three_layer_details.png`

### 12.3 staged 输出

默认写到 `runs/`：

- `staged_suite_*`
- `staged_isolated_*`
- `stage_summary.json`
- `*_progress.json`

## 13. 环境变量

项目支持通过环境变量覆盖目录：

- `MAS_LOG_DIR`
- `MAS_PLOT_DIR`
- `MAS_MODEL_CACHE`
- `MAS_DATA_CACHE`

还支持一些运行控制变量：

- `MAS_REQUIRE_CUDA`
- `MAS_FORCE_DEVICE`
- `CUDA_VISIBLE_DEVICES`

这些变量在多卡调试、staged 运行、以及服务器环境中都很有用。

## 14. 运行注意事项

1. 当前代码依赖 `modelscope`。如果环境缺少该包，`run_all.py` 无法正常 import。
2. verifier 默认优先复用项目目录下的本地 embedding 缓存；如果本地缓存不存在，会回退到 `MAS_VERIFIER_EMBED_MODEL`（默认 `gpt2`）并下载到项目缓存目录。
3. 第一次加载模型和数据时，缓存目录会明显增大，因此不建议把 `data/`、`models/`、`runs/` 直接提交到 Git。
4. `run_staged_suite.py` 和 `run_staged_isolated.py` 默认是小样本 smoke / staging 工具，不等同于最终正式实验。
5. 由于存在在线训练和采样，实验结果会对随机性、GPU 环境和模型缓存状态敏感。

## 15. 当前仓库整理建议

本仓库建议只跟踪以下内容：

- 源代码
- README
- 依赖文件
- 必要的轻量配置文件

建议忽略：

- `data/`
- `models/`
- `logs/`
- `plots/`
- `runs/`
- `__pycache__/`
- 各类 `*.pyc`

这正是当前 `.gitignore` 的设计目标。

## 16. 后续 GitHub 推送

当前这一步已经把目录整理成适合作为仓库根目录的结构。  
如果后续要真正推送到 GitHub，还需要补齐：

- 远程仓库 URL
- 认证方式
  - SSH
  - PAT
  - 或本机已完成 `gh auth login`

然后再执行：

```bash
git remote add origin <your-repo-url>
git add .
git commit -m "Initial project import"
git push -u origin main
```

## 17. 一句话总结

这是一个把“选哪个 agent”“选什么行为”“把内容生成好”拆成三层的 GSM8K 多智能体实验框架：

- 外层和中层用 CFR 学策略
- 内层用 GSPO 学文本
- verifier 负责状态判断和答案接管
- 现在统一支持 8 组实验、`train/val/test` 流程，以及针对 `comment / answer / silent` 的显式价值定义
