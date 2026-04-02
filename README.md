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

在 `中间层策略+GSPO` 和 `中间层策略+GSPO（无沉默）` 中，这一层被移除，系统固定只有一个 agent bundle。

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

另外还有两组“无沉默”消融：

- `中间层策略+GSPO（无沉默）`
- `全量策略（中层无沉默）`

这两组的中层动作空间只保留 `comment / answer`，不允许选择 `silent`。

### 3.3 最内层：文本策略

最内层是实际生成自然语言内容的策略：

- `pi0`：comment policy
- `pi1`：answer policy

训练算法为 GSPO。它对同一上下文采样多个候选，在组内依据 reward/value 更新策略。

## 4. verifier 的作用

verifier 实现在 [src/verifier.py](src/verifier.py)。

当前版本是一个轻量在线 verifier：

- backbone：本地 embedding 模型
- head：一个线性 scorer

它承担两个职责：

1. 决定外层状态 `phase`
   - `search`
   - `stabilize`

2. 决定 answer 候选是否可以接管当前 incumbent

当前逻辑下：

- `keep_prob = scorer(question, incumbent_answer)`
- `accept_prob = scorer(question, candidate_answer)`
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

对应实现见 [run_all.py](run_all.py#L1397) 和 [run_all.py](run_all.py#L2669)。

### 6.3 中间层策略+GSPO / 中间层策略+GSPO（无沉默） / 全量策略 / 全量策略（中层无沉默）中的 value 定义

这四条共享策略栈路径的 value 定义如下。

#### answer 的 value

`answer` 的 value 直接定义为答案质量 `r`。

实现见 [run_all.py](run_all.py#L1288)。

#### comment 的 value

`comment` 的 value 当前恢复为旧定义：

`with comment 的下一次 answer 的 r - without comment 的下一次 answer 的 r`

实现见 [run_all.py](run_all.py#L1468)。

#### silent 的 value

`silent` 的 value 定义为：

`with silent 的下一次 answer 的 r - 按当前 comment/answer 归一化策略抽样 rollout 得到的 answer 的 r`

也就是说：

- `with silent`
  本轮静默，不立刻发 comment / answer，然后看下一次 answer 的结果

- `without silent`
  因为这一轮已经选定了该 agent，所以如果不选 silent，就只可能在 `comment` 和 `answer` 之间选择
  此时把当前策略在 `comment / answer` 上重新归一化后进行抽样 rollout，得到下一次 answer 的 reward

主体实现见：

- [run_all.py](run_all.py#L1593)
- [run_all.py](run_all.py#L1650)

在两组“无沉默”实验中，中层动作空间不包含 `silent`，因此不会对 `silent` 做选择或更新；但其余 value 和反事实遗憾的定义与共享策略栈保持一致。

### 6.4 中层 CFR 的反事实遗憾

中层三动作的反事实比较使用上述 value：

- `comment` 和 `answer` 的反事实基于它们各自 value
- `silent` 的反事实基于上面的 silent value
- regret 由 `action_value(other) - action_value(chosen)` 形成

对应更新逻辑见 [run_all.py](run_all.py#L2204)。

### 6.5 外层调度器的价值

外层每个 agent 的价值定义为：

`本轮结束后最近 answer 的 r`

所以：

- 若本轮选该 agent 后最终没有接受新答案，则价值通常回到 incumbent 的 `r`
- 若接受了更好的 answer，则价值提高

实现见 [run_all.py](run_all.py#L1735) 和 [run_all.py](run_all.py#L2204)。

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
2. verifier 依赖本地 embedding 模型路径；如果本地模型不存在，verifier 初始化会失败。
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
