# Three-Level MAS Experiment on GSM8K

本项目实现了一个面向 GSM8K 数学题的多智能体实验框架，用于比较从单模型基线到三层策略栈的 8 组实验设置，并分析在多轮交互中，`comment / answer / silence` 行为、外层调度器、phase 状态机、以及内层 GSPO 更新之间的作用关系。

当前代码以 `three_level_experiment` 目录作为实验根目录。模型缓存、数据缓存、日志、图表和 staged 运行输出都默认落在该目录下，便于独立打包成一个 Git 仓库。

## 1. 项目目标

这个项目关注的问题不是“单次生成能否直接答对 GSM8K”，而是：

1. 多轮交互是否能逐步改善当前答案。
2. 不同行为类型是否应该被区别对待。
3. 哪一层负责“选谁来发言”、哪一层负责“发什么类型的话”、哪一层负责“把这句话说好”。
4. 如何同时训练外层调度、中层动作策略和内层文本策略，并保持训练与推理的一致性。

当前轮次配置也已拆成两项：

- `TRAIN_NUM_ROUNDS`：训练阶段使用的轮次
- `INFER_NUM_ROUNDS`：验证 / 测试 / 基线推理阶段使用的轮次

默认两者保持一致；当前默认值都是 `5`。

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
   `phase` 由上一轮该真实分支自己的单个 realized middle value 决定，answer 直接更新 latest answer。

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
- `train` 时按当前 regret-matching 策略采样动作
- `val/test` 时按训练阶段累计下来的、同一 `phase` 状态对应的 `strategy_sum` 归一化平均策略采样动作
- 如果某个状态还没有平均策略统计量，则回退到当前策略

当前 `phase` 规则很简单：

- 第 1 轮固定 `search`
- 从第 2 轮开始，若上一轮该真实分支自己的单个 realized middle value 大于 `0.05`，则继续 `search`
- 否则进入 `stabilize`

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
- `train` 时按“正遗憾归一化”得到当前策略并采样
- `val/test` 时在同一 `search / stabilize` 状态下改用训练阶段累计 `strategy_sum` 归一化得到的平均策略并采样
- 双层 / 三层共享策略栈在最后一轮也不强制 `answer`，仍由中层策略自己决定
- 如果所有正遗憾都为 0，则使用状态相关初始化分布：
  - `search`：`comment = 1/2`，`answer = 1/2`，`silent = 0`
  - `stabilize`：`comment = 1/3`，`answer = 1/3`，`silent = 1/3`
- 对无沉默实验，动作 mask 会自动把上面的初始化分布重新归一化到 `comment / answer`
- 如果某个状态还没有平均策略统计量，则平均策略自动回退到当前策略
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
- 训练时，这一批 candidates 会展开成多条真实训练分支
- 验证/测试时，真实轨迹继续使用该 batch 的 `selected_text`
- 当前实现里 `selected_idx` 默认就是第一个候选

也就是说，这几条共享策略栈实验里：

- 内层 GSPO 一直用整批 candidates 更新
- 训练阶段的中层/外层也会把这批 candidates 展开成多条真实训练分支
- 验证/测试阶段仍保留单分支真实轨迹，用于可控评估

`单LLM GSPO` 和 `双LLM GSPO轮询` 也默认使用该次采样 batch 的第一个候选推进真实轨迹。

## 4. phase 与 latest answer

当前默认实验不再使用 verifier。

共享策略栈里和“状态切换 / 答案保留”相关的规则现在是：

1. `phase`
   - 第 1 轮固定为 `search`
   - 从第 2 轮开始，看上一轮该真实分支自己的单个 realized middle value
   - 若该值大于 `0.05`，则本轮继续 `search`
   - 否则本轮为 `stabilize`

2. `latest_answer`
   - 旧代码里的 `incumbent` 变量现在只表示“最近一次 answer 的文本/数值”
   - 它不再是经过 verifier 审批后保留下来的 incumbent
   - 它只在后续 comment / silent 轮次中作为 fallback 被沿用

仓库里仍保留 [src/verifier.py](src/verifier.py) 作为旧实验兼容代码，但 `run_all.py` 当前默认 8 组实验不会实例化或更新它。

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

其中，共享策略栈路径在 `train / val / test` 三个阶段都共用同一套 batch 采样与 value 定义；差别在于：

- `train` 会把真实 batch 展开成多条训练分支，并更新 CFR / GSPO
- `val / test` 不更新参数，也不展开多分支，只沿 `selected_text` 单路径评估
- `train` 的外层/中层动作来自当前 regret-matching 策略
- `val / test` 的外层/中层动作来自训练全过程累计得到的平均策略
- 三个阶段的 `phase` 都使用“上一轮该真实分支自己的单个 realized middle value”
- 轮次数量也分开配置：`train` 用 `TRAIN_NUM_ROUNDS`，`val / test` 用 `INFER_NUM_ROUNDS`

### 5.3 哪些实验只做测试

不训练参数的基线实验有 2 组：

- `单LLM`
- `双LLM轮询`

这两组直接在 `test` 上评估。
它们没有中层策略，因此 answer / comment 的轮次切换完全由 prompt 节奏决定；默认 5 轮下自然是最后一轮 answer，并不是代码额外强制。

统一统计口径补充：

- 无论训练还是验证 / 测试，准确率始终使用“最近一次真正产生的 answer”
- comment / silent 轮因为没有新 answer，才沿用上一条 answer
- 如果新的 answer 不可解析，也按这次最新 answer 记分，不回退到更早的旧答案

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
  - 对每一条真实训练分支，使用这条分支自己对应的单个 realized value
- 未选动作：
  - 额外做 counterfactual Monte Carlo 采样
  - `answer`：采样 1 个 counterfactual answer batch，并对组内 candidates 的 value 取均值
  - `comment`：采样 1 个 counterfactual comment batch，并对组内 candidates 的 value 取均值
  - 其中 comment 单个 candidate 的 value 本身是“该 comment 下 rollout 1 个 next answer 的 reward 减去当前 silent 基线”
- `silent` 固定使用 `0`
- regret 由 `action_value(other) - action_value(chosen)` 形成

因此当前实现里：

- 训练真实轨迹：comment / answer 每轮都会先产生一组 candidates，再展开成多条真实训练分支
- 验证/测试真实轨迹：仍然只沿 `selected_text` 保留单分支
- 虚拟 rollout：每条 comment 轨迹只继续产生 1 个 answer
- 反事实动作价值：只采样 1 个 counterfactual batch，不再做 batch 均值外再套一层 batch 均值

当前中层在训练时直接使用 raw regret-matching 当前策略，在验证/测试时改用训练阶段累计得到的同状态平均策略；两者都不再对 `search` 状态施加额外的动作概率约束。

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
│   ├── verifier.py      # 旧实验兼容代码，默认主流程未使用
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
  - phase epsilon
  - 以及保留的旧 verifier 超参数
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
2. 默认 `run_all.py` 主流程已经不依赖 verifier embedding 缓存；只有你手动调用旧的 `src/verifier.py` 路径时，才会用到相关 verifier 模型配置。
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
- `phase` 由上一轮该真实分支自己的单个 realized middle value 决定
- `latest_answer` 只负责在 comment / silent 轮次下做兜底
- 现在统一支持 8 组实验、`train/val/test` 流程，以及针对 `comment / answer / silent` 的显式价值定义
