# Final Multi-Benchmark Ablation Suite

这个目录是论文最终消融实验入口。`01_main_exp30` 以实验 30 的 proposal-review 逻辑为基础，并采用实验 31 的细粒度中层投票 state；`02` 起是围绕论文需要解释的关键机制做的消融。所有实验的训练都固定使用 GSM8K 官方全量 `train` split，不切 validation；测试集可在 GSM8K、MATH-500 和 GPQA-Diamond 中选择。

## 主实验逻辑

`01_main_exp30` 的核心设置：

- 5 轮训练，5 轮测试。
- 单个 proposal-review agent bundle：`pi1` 是 proposer，`pi0` 是 reviewer。
- 中层 controller 只在 proposal stage 选择 `keep` 或 `refresh`。
- reviewer stage 只产生评审文本和净票更新。
- 中层 controller 使用 selected-only regret matching。
- 瞬时遗憾使用期望基线：`re(a, s) = v_a - sum_b Pi(b|s) v_b`。
- refresh 语义沿用实验 30：只要 controller 选择 `refresh`，新 pending 可以取代旧 pending；不要求旧 pending 的净票数小于 0。
- 默认不启用投票阈值状态机；reviewer verdict 只累积 `pending_vote_score`，不会自动 promote/drop pending。
- 中层 controller state 使用实验 31 的细粒度票数分桶：有 pending 时按精确净票数写入 `pending_score_bucket`，例如 `vote_0`、`vote_plus_1`、`vote_plus_2`、`vote_minus_1`，再与 `proposal_time_bucket` 组合成 state。

## 实验编号

| 编号 | 名称 | 消融问题 |
| --- | --- | --- |
| 01 | `01_main_exp30` | 主实验，完整 proposal-review + learned controller + refresh 可替换任意 pending + 细粒度投票 state。 |
| 02 | `02_fixed_keep_refresh_controller` | 去掉 learned controller，proposal stage 固定 keep/refresh 周期。 |
| 03 | `03_always_refresh_controller` | controller 固定 refresh，检验是否只是持续重采样带来收益。 |
| 04 | `04_always_keep_controller` | controller 固定 keep，检验 refresh 的必要性。 |
| 05 | `05_no_counterfactual_controller_values` | 保留 controller 更新，但未选动作反事实 value 置零。 |
| 06 | `06_no_controller_regret_update` | 冻结中层 controller regret/average strategy。 |
| 07 | `07_vote_threshold_state_machine` | 启用旧投票阈值状态机：净票 `>= +2` promote，`<= -1` drop。 |
| 08 | `08_no_vote_updates` | reviewer 仍生成和训练，但 verdict 不再改变净票。 |
| 09 | `09_refresh_only_negative_pending` | 恢复保护：pending 净票 `>= 0` 时 refresh 不能替换旧 pending。 |
| 10 | `10_no_refresh_action` | 禁用 refresh；有 pending 时只能 keep。 |
| 11 | `11_reviewer_frozen` | 冻结 reviewer/pi0 的 GSPO 更新。 |
| 12 | `12_proposer_frozen` | 冻结 proposer/pi1 的 GSPO 更新。 |
| 13 | `13_no_inner_gspo_updates` | 冻结 proposer 和 reviewer 的全部 GSPO 更新，只学习 controller。 |
| 14 | `14_no_review_feedback_context` | proposer refresh 时不看 latest review feedback。 |
| 15 | `15_unstructured_legacy_prompts` | 保留状态机，但 proposer 使用旧式非结构化 answer prompt。 |

这些消融都在训练和测试两阶段同时生效，不是只在测试时改策略。

细粒度投票 state 默认启用于 `01-15` 全部实验。这样主实验和所有消融在 state 表达能力上保持一致，实验差异只来自各自的消融开关。对于固定 controller、冻结 controller 或禁用 refresh 的实验，细粒度 state 不会改变固定动作本身，主要用于保持记录、checkpoint 和测试策略读取时的状态定义一致。

## 底模和测试集

所有实验都支持选择底模。默认是 Phi-3-mini-4k-instruct：

| 参数值 | ModelScope 模型 | 说明 |
| --- | --- | --- |
| `phi3-mini-4k-instruct` | [LLM-Research/Phi-3-mini-4k-instruct](https://modelscope.cn/models/LLM-Research/Phi-3-mini-4k-instruct) | 默认底模。 |
| `qwen2.5-7b-instruct` | [Qwen/Qwen2.5-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct) | Qwen2.5 7B Instruct。 |

训练数据固定为 [GSM8K train](https://modelscope.cn/datasets/modelscope/gsm8k)。测试集通过 `--test-dataset` 或 `--test-datasets` 选择：

| 参数值 | ModelScope 数据集 | 处理方式 |
| --- | --- | --- |
| `gsm8k` | [modelscope/gsm8k](https://modelscope.cn/datasets/modelscope/gsm8k) | 官方 test split，数值答案。 |
| `math500` | [AI-ModelScope/MATH-500](https://modelscope.cn/datasets/AI-ModelScope/MATH-500) | 使用专用 MATH prompt，要求 `Final answer: \boxed{...}`；答案抽取保留符号/分数/坐标等原格式，并用 `math_verify` 做等价判断。 |
| `gpqa-diamond` | [AI-ModelScope/GPQA](https://modelscope.cn/datasets/AI-ModelScope/GPQA) | 使用 `gpqa_diamond` subset，选项打乱为 `A-D`；使用专用 GPQA prompt，要求 `Final answer: <A/B/C/D>`，答案抽取只认最终选项。 |

prompt 会同时随测试集和底模自动切换，但 proposal-review 协议不变：reviewer 仍然第一行输出 `RIGHT/WRONG`、第二行输出短理由；proposer 仍然 single-CoT 风格求解，只改变推理强度和最终答案格式要求。训练阶段样本始终是 GSM8K，因此训练时使用 GSM8K 数值抽取和当前底模对应的 GSM8K prompt；测试阶段按当前测试集切换到对应抽取器和 prompt。

底模 prompt profile：

- `phi3-mini-4k-instruct` 使用默认精简 prompt，强调短推理、格式稳定和避免无根据改答案。
- `qwen2.5-7b-instruct` 使用 Qwen 专用 prompt，仍保持同一 proposal-review 协议和 final-line 格式，但允许更充分的 decisive check、算术/符号核验、选项排除和错误定位，以发挥 7B instruct 模型更强的推理能力。

所有 final 实验的生成长度默认统一为 `256`：`MAS_MAX_NEW_TOKENS=256`、`MAS_TRAIN_MAX_NEW_TOKENS=256`、`MAS_EVAL_MAX_NEW_TOKENS=256`、`MAS_PROPOSAL_COMPLETION_MAX_NEW_TOKENS=256`、`MAS_REVIEW_COMPLETION_MAX_NEW_TOKENS=256`。

## 指标

每个实验都会测试两种 controller 模式：

- `controller_stochastic`：按训练累计 average strategy 采样 controller 动作。
- `controller_greedy`：选训练累计 average strategy 概率最大的 controller 动作。

这里的 controller 策略源与实验 30 保持一致：测试阶段使用训练中累计得到的 average strategy；`stochastic` 从 average strategy 采样，`greedy` 取 average strategy 的 argmax。

每种模式都会写出：

- `accuracy_r1_to_r5` / `round_accuracy`：第 1 到第 5 轮准确率。
- `accuracy`：第 5 轮 final answer 准确率。
- `tokens_per_task`：每题平均 prompt + completion token 数。
- `llm_calls_per_task`：每题平均模型生成次数。
- `active_agents_per_task`：每题实际调用过的 agent 数。
- `correction_rate`：first parseable answer 错误、final answer 正确的样本比例。
- `preservation_rate`：first parseable answer 正确且 final answer 仍正确的样本比例。

逐样本记录位于 `samples_controller_stochastic.jsonl` 和 `samples_controller_greedy.jsonl`。原始 `run_all` 测试输出位于 `raw_eval_*.log`。

## 运行

进入目录：

```bash
cd /mnt/paper2any/lcs/MAS/final
```

运行全套最终实验：

```bash
conda run --no-capture-output -n lcs-metax python -u run_final_experiments.py
```

只运行主实验：

```bash
conda run --no-capture-output -n lcs-metax python -u run_final_experiments.py --experiments 01_main_exp30
```

指定底模和测试集：

```bash
conda run --no-capture-output -n lcs-metax python -u run_final_experiments.py \
  --experiments 01_main_exp30 \
  --base-model qwen2.5-7b-instruct \
  --test-dataset gpqa-diamond
```

跑两个底模和三个测试集的网格：

```bash
conda run --no-capture-output -n lcs-metax python -u run_final_experiments.py \
  --experiments 01_main_exp30 02_fixed_keep_refresh_controller \
  --base-models phi3-mini-4k-instruct qwen2.5-7b-instruct \
  --test-datasets gsm8k math500 gpqa-diamond
```

也可以直接运行编号脚本：

```bash
conda run --no-capture-output -n lcs-metax python -u 01_main_exp30.py
conda run --no-capture-output -n lcs-metax python -u 11_reviewer_frozen.py
```

小样本 smoke test：

```bash
conda run --no-capture-output -n lcs-metax python -u run_final_experiments.py \
  --experiments 01_main_exp30 \
  --train-limit 2 \
  --test-limit 2 \
  --checkpoint-every 1
```

## 并行和多卡

final suite 有两层并行：

1. 单个实验内部的 policy worker 多卡并行。
2. 多个实验之间的并行启动。

### 单个实验内部多卡

默认已经启用：

```text
MAS_PARALLEL_MODE=three_layer_workers
```

在 proposal-review 实验中，一个实验会启动两个 policy worker：

- `agent0.pi0`：reviewer
- `agent0.pi1`：proposer

如果不指定设备映射，代码会把 policy worker 按当前可见 GPU round-robin 分配。例如机器可见 `cuda:0,cuda:1` 时，通常会把 reviewer 和 proposer 分到两张卡。

显式指定两张卡：

```bash
conda run --no-capture-output -n lcs-metax python -u run_final_experiments.py \
  --experiments 01_main_exp30 \
  --cuda-visible-devices 0,1 \
  --policy-device-map agent0.pi0:0,agent0.pi1:1
```

只想单卡跑一个实验：

```bash
conda run --no-capture-output -n lcs-metax python -u run_final_experiments.py \
  --experiments 01_main_exp30 \
  --cuda-visible-devices 0 \
  --policy-device-map agent0.pi0:0,agent0.pi1:0
```

关闭 policy worker 并行，改为串行模式：

```bash
conda run --no-capture-output -n lcs-metax python -u run_final_experiments.py \
  --experiments 01_main_exp30 \
  --parallel-mode serial
```

### 多个实验并行

`--experiment-workers` 控制同时启动多少个实验子进程。GPU 分配有两种方式：

- `--device-groups`：按空闲 slot 自动给实验分配 GPU 组。
- `--experiment-device-map`：按实验名显式指定每个实验可见哪些 GPU。

两组 GPU 同时跑两个实验，每个实验内部用两张卡：

```bash
conda run --no-capture-output -n lcs-metax python -u run_final_experiments.py \
  --experiment-workers 2 \
  --device-groups '0,1;2,3'
```

上面命令的含义：

- 同时跑 2 个实验。
- 第一个运行中的实验看到 `CUDA_VISIBLE_DEVICES=0,1`。
- 第二个运行中的实验看到 `CUDA_VISIBLE_DEVICES=2,3`。
- 每个实验内部仍用 `three_layer_workers`，并在自己可见的 GPU 组内 round-robin 放置 proposer/reviewer。

按实验名自由指定 GPU：

```bash
conda run --no-capture-output -n lcs-metax python -u run_final_experiments.py \
  --experiments 01_main_exp30 02_fixed_keep_refresh_controller 03_always_refresh_controller \
  --experiment-workers 3 \
  --experiment-device-map '01_main_exp30=0,1;02_fixed_keep_refresh_controller=2;03_always_refresh_controller=3'
```

上面命令会同步跑 3 个实验：

- `01_main_exp30` 只能看到 GPU `0,1`。
- `02_fixed_keep_refresh_controller` 只能看到 GPU `2`。
- `03_always_refresh_controller` 只能看到 GPU `3`。

这种方式允许每个实验自由决定可用卡；如果多个实验显式写成同一组 GPU，代码不会阻止，但会共享显存，通常不建议。

当同时跑多个底模/测试集组合时，输出目录会带上组合后缀，例如：

```text
01_main_exp30__model-qwen2.5-7b-instruct__test-gpqa-diamond
```

`--experiment-device-map` 可以写实验名，也可以写完整 `run_name`。写实验名会影响该实验的所有底模/测试集组合；写完整 `run_name` 只影响那一个组合：

```bash
conda run --no-capture-output -n lcs-metax python -u run_final_experiments.py \
  --experiments 01_main_exp30 \
  --base-models phi3-mini-4k-instruct qwen2.5-7b-instruct \
  --test-datasets gsm8k gpqa-diamond \
  --experiment-workers 2 \
  --experiment-device-map '01_main_exp30__model-qwen2.5-7b-instruct__test-gpqa-diamond=2,3'
```

如果每个实验只给一张卡，可用：

```bash
conda run --no-capture-output -n lcs-metax python -u run_final_experiments.py \
  --experiment-workers 4 \
  --device-groups '0;1;2;3' \
  --policy-device-map agent0.pi0:0,agent0.pi1:0
```

注意：`--policy-device-map` 的编号是子进程视角下的可见 GPU 编号。比如 `--device-groups '2,3'` 的子进程内部仍把这两张卡看成 `0,1`，因此映射写 `agent0.pi0:0,agent0.pi1:1`，不是 `2,3`。

### 参数速查

- `--base-model phi3-mini-4k-instruct|qwen2.5-7b-instruct`：单个底模。
- `--base-models phi3-mini-4k-instruct qwen2.5-7b-instruct`：按底模展开网格，覆盖 `--base-model`。
- `--test-dataset gsm8k|math500|gpqa-diamond`：单个测试集；训练仍固定是 GSM8K train。
- `--test-datasets gsm8k math500 gpqa-diamond`：按测试集展开网格，覆盖 `--test-dataset`。
- `--parallel-mode three_layer_workers|serial`：单实验内部是否启用 policy workers。
- `--policy-device-map agent0.pi0:0,agent0.pi1:1`：显式指定 reviewer/proposer 的 GPU。
- `--cuda-visible-devices 0,1`：所有实验子进程共同使用的可见 GPU。
- `--experiment-workers N`：同时启动 N 个实验。
- `--device-groups '0,1;2,3'`：多实验并行时，每个实验分配一个 GPU 组。
- `--experiment-device-map '01_main_exp30=0,1;02_fixed_keep_refresh_controller=2'`：按实验名指定 GPU 组，覆盖 `--device-groups` 对这些实验的自动分配。
- token 上限默认全部是 `256`；如需临时覆盖，可在启动前设置 `MAS_MAX_NEW_TOKENS`、`MAS_TRAIN_MAX_NEW_TOKENS`、`MAS_EVAL_MAX_NEW_TOKENS`、`MAS_PROPOSAL_COMPLETION_MAX_NEW_TOKENS`、`MAS_REVIEW_COMPLETION_MAX_NEW_TOKENS`。

输出目录默认是：

```text
/mnt/paper2any/lcs/MAS/final/runs/final_YYYYMMDDTHHMMSSZ/
```

每个实验一个子目录，包含：

- `metadata.json`：实验设置、底模、数据 split、ModelScope 链接、模块开关。
- `summary.json`：训练 R1-R5 和两种测试模式的汇总。
- `summary_controller_stochastic.json` / `summary_controller_greedy.json`：单模式测试汇总。
- `samples_controller_stochastic.jsonl` / `samples_controller_greedy.jsonl`：逐样本预测与 usage。
- `checkpoints/`：在线训练 checkpoint，默认每 50 条训练样本保存一次，训练结束再保存最终 checkpoint。
- `logs/live_accuracy.jsonl`：训练时实时 R1-R5 准确率。

## 主实验训练曲线

`01_main_exp30` 会额外生成：

```text
plots/main_train_r1_r5_accuracy.png
```

图中用不同颜色画出训练过程中 R1 到 R5 accuracy 随训练样本数增加的变化曲线。

## 实现入口

- `run_final_experiments.py`：最终实验总入口、编号、指标统计、训练曲线绘图。
- `run_all.py`：实验 30 逻辑和消融开关的实际执行位置。
- `src/config.py`：消融开关默认值；默认值保持实验 30。
- `src/data_loader.py`：`load_final_train_test_splits()` 固定加载 GSM8K train，并按参数加载 GSM8K/MATH500/GPQA-Diamond 测试集。
