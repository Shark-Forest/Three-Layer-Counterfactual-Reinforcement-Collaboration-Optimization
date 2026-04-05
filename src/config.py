import os

# 统一把“实验自己的文件”锚定到 three_level_experiment 目录下。
# 这样无论你从哪个工作目录启动脚本，模型缓存都不会再跑到 /mnt/paper2any 根目录去。
EXPERIMENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据集配置（魔塔社区）
GSM8K_MODEL_SCOPE = "modelscope/datasets--gsm8k"
# 主流程默认使用 GSM8K 官方完整 split：
# - 官方 train 再切成 train / val
# - 官方 test 作为最终测试集
#
# 如需做 smoke / staged 小样本实验，可在脚本里显式传入 limit。
TRAIN_VAL_TOTAL_LIMIT = None
TEST_SAMPLE_LIMIT = None
TRAIN_RATIO = 0.8
DATA_SPLIT_SEED = 42
# 训练与推理轮次分开配置，默认保持一致。
TRAIN_NUM_ROUNDS = 5
INFER_NUM_ROUNDS = TRAIN_NUM_ROUNDS
# 兼容旧代码保留；新代码请优先使用 TRAIN_NUM_ROUNDS / INFER_NUM_ROUNDS。
NUM_ROUNDS = TRAIN_NUM_ROUNDS

# 模型配置（魔塔社区 Phi-3）
# 这里选用更强但仍相对可运行的 Phi-3 Mini Instruct 版本。
GPT2_MODEL_SCOPE = "LLM-Research/Phi-3-mini-4k-instruct"
MAX_NEW_TOKENS = 128
# 降低生成随机性，减少早期在线训练时“偶然好/偶然坏”答案对更新方向的干扰。
TEMPERATURE = 0.3

# 算法超参
# 再把学习率降一档，让在线更新更保守，避免少量样本就把策略带偏。
GSPO_LR = 5e-6
# 候选数略增，提升组内相对比较的稳定性；5条样本复验时这点额外开销可接受。
GSPO_NUM_CANDIDATES = 4
# 缩小裁剪区间，减少一次更新把策略推太远。
GSPO_CLIP_EPS = 0.1
GSPO_USE_INDEPENDENT_MODEL = True  # 每个GSPO agent使用独立Phi-3副本，避免多agent共享同一活跃策略
GSPO_TRAIN_LAST_N_LAYERS = 2       # 除lm_head外，再解冻最后N层decoder block
# GSPO 在线更新的训练精度。
# 为什么这里单独配：
# - 之前 GSPO 独立副本沿用了 fp16 权重，并直接用 Adam 做在线更新
# - 在本实验里这会很容易把 exp_avg / exp_avg_sq 等优化器状态也压到 fp16，
#   数值范围不够，出现 inf / nan 后会把后续生成直接打坏
# - bfloat16 的指数范围更大，通常比 fp16 更适合做这种在线微调
#
# 可选值：
# - "bfloat16": 训练时把 GSPO 独立副本转成 bf16
# - "float32":  训练时转成 fp32，更稳但更占显存
# - "float16":  保留 fp16，不推荐
# - "auto":     CUDA 支持 bf16 时用 bf16，否则退回 fp32
GSPO_TRAIN_DTYPE = "bfloat16"
# 虚拟 rollout 明确只采样 1 个 next answer：
# - 真实轨迹里的 comment / answer 仍然是一整组候选
# - 但在 comment 奖励评估、silent 基线、反事实轨迹里，
#   每条虚拟轨迹只继续生成 1 个 answer，避免 comment 下再展开一组 answer
#   导致计算量近似 g^g 爆炸
GSPO_COMMENT_EVAL_SAMPLES = 1
# 如果一组候选奖励几乎完全一样，说明这次更新没有足够区分信号，直接跳过更稳。
GSPO_SKIP_ZERO_SIGNAL_UPDATE = True
GSPO_MIN_REWARD_STD = 0.05
# 梯度裁剪，避免某一小批在线更新把策略猛地推偏。
GSPO_MAX_GRAD_NORM = 0.5
# 中层动作空间：
# 0=静默，1=评论，2=输出答案
CFR_NUM_ACTIONS = 3
MIDDLE_ACTION_SILENT = 0
MIDDLE_ACTION_COMMENT = 1
MIDDLE_ACTION_ANSWER = 2
# phase 切换规则：
# - 第 1 轮固定 search
# - 从第 2 轮开始，若上一轮该真实分支自己的单个 realized middle value > PHASE_Q_EPS，则继续 search
# - 否则进入 stabilize
PHASE_Q_EPS = 0.05
NUM_AGENTS = 2           # 外层可调度的 agent 数
# 三层策略逐轮调试打印：
# - True:  每一轮打印 act / pred / reward / CFR 概率，便于排查 R3/R4/R5 为什么变化
# - False: 关闭逐轮打印，只保留汇总统计
THREE_LAYER_DEBUG_PRINT = True

# 奖励形状：
# - final accuracy 看 exact match
# - RL 训练仍保留少量 dense reward，避免全错时完全没梯度信号
REWARD_EXACT_WEIGHT = 0.8
REWARD_DENSE_WEIGHT = 0.2
REWARD_MISSING_PENALTY = -0.2
# rollout 到结尾仍没得到可解析答案时：
# - 若没有 incumbent，给更重的终端惩罚
# - 若已有 incumbent，则在 delta 口径下记为 0，在 absolute 口径下回退到 incumbent reward
ROLLOUT_NO_ANSWER_PENALTY = -1.0

# 下面这组 verifier 配置保留为历史兼容项。
# 当前默认 8 组实验中，共享策略栈已不再使用 verifier；
# 这里保留常量主要是为了：
# - 兼容旧实验记录
# - 兼容仍可能单独引用 src/verifier.py 的旧脚本
VERIFIER_LR = 0.1
# 这些 accept-specific 阈值保留名字仅为兼容旧日志/旧实验脚本；
# 单 scorer 版本下，真正的 accept 边界由 keep_prob 本身给出，不再使用这些值。
VERIFIER_ACCEPT_THRESHOLD = 0.50
VERIFIER_ACCEPT_RELAX_KEEP_PROB = 0.55
VERIFIER_ACCEPT_LOW_KEEP_PROB = 0.45
VERIFIER_ACCEPT_RELAXED_THRESHOLD = 0.40
VERIFIER_ACCEPT_POSITIVE_WEIGHT = 3.0
VERIFIER_ACCEPT_NEGATIVE_WEIGHT = 1.0
VERIFIER_ACCEPT_SKIP_ZERO_DELTA = True
VERIFIER_ACCEPT_ZERO_DELTA_EPS = 1e-6
# 先保留这个配置名以兼容旧代码/旧实验记录，但当前版本不再使用 gray margin。
VERIFIER_MARGIN = 0.05
# verifier embedding 优先复用项目目录下的本地缓存；
# fresh clone 若该目录为空，则回退到远端模型 ID 自动下载到 MODEL_CACHE。
DEFAULT_VERIFIER_EMBED_MODEL_PATH = os.path.join(
    EXPERIMENT_ROOT,
    "models",
    "AI-ModelScope",
    "gpt2",
)
VERIFIER_EMBED_MODEL_PATH = os.environ.get(
    "MAS_VERIFIER_EMBED_MODEL_PATH",
    DEFAULT_VERIFIER_EMBED_MODEL_PATH,
)
VERIFIER_EMBED_MODEL_ID = os.environ.get("MAS_VERIFIER_EMBED_MODEL", "gpt2")
VERIFIER_MAX_LENGTH = 256
VERIFIER_CACHE_SIZE = 256
# 兼容旧代码保留；单 scorer 版本的 keep 更新直接拟合 incumbent reward，
# 不再把 reward 先二值化成 should_keep 标签。
VERIFIER_KEEP_THRESHOLD = 0.85
# 用 incumbent 的 scorer 输出构造外层 phase：
# stability_score = keep_prob + VERIFIER_STABILIZE_ALPHA * late_flag
# 若该分数 >= VERIFIER_PHASE_THRESHOLD，则进入 stabilize，否则 search。
VERIFIER_PHASE_THRESHOLD = 0.85
VERIFIER_STABILIZE_ALPHA = 0.10

# search 状态下的中层 CFR 动作探索控制。
# 目标：
# - incumbent 不稳时，少一点 pi2=静默
# - 同时确保 pi0=评论、pi1=答案 都保留最低探索概率
SEARCH_MIN_COMMENT_PROB = 0.20
SEARCH_MIN_ANSWER_PROB = 0.20
SEARCH_MAX_SILENT_PROB = 0.20
SEARCH_RELAX_KEEP_PROB = 0.55
SEARCH_LOW_KEEP_PROB = 0.45
SEARCH_UNSTABLE_MIN_ANSWER_PROB = 0.55
SEARCH_UNSTABLE_MAX_SILENT_PROB = 0.05
SEARCH_NO_INCUMBENT_MIN_ANSWER_PROB = 0.60
SEARCH_NO_INCUMBENT_MAX_SILENT_PROB = 0.00
SEARCH_LATE_ROUND_ANSWER_BONUS = 0.10
SEARCH_SILENT_PENALTY = 0.03
# 旧版本曾用它在“无 incumbent 的前几轮”强制 answer。
# 当前共享策略栈实验已经不再对最后一轮施加强制 answer 约束；
# 这里保留这个配置名仅为兼容旧实验记录。
NO_INCUMBENT_FORCE_ANSWER_ROUNDS = 2
ANSWER_CANDIDATE_CHANGE_BONUS = 0.01
ANSWER_CANDIDATE_CHANGE_PENALTY = 0.03
ANSWER_CANDIDATE_KEEP_BONUS = 0.02
SEARCH_STABLE_KEEP_PROB = 0.55
SEARCH_STABLE_MAX_ANSWER_PROB = 0.35
SEARCH_STABLE_MIN_COMMENT_PROB = 0.45

# 统一结构化上下文时保留的最近历史规模。
STRUCTURED_CONTEXT_MAX_COMMENTS = 2

# 路径配置
# 这里让日志 / 图 / 模型 / 数据都固定落在实验目录下，
# 避免“同一份代码，换个启动目录就写到别处去”的问题。
#
# 同时允许用环境变量覆盖这些目录，便于：
# - staged 运行把日志/图写进各自的 run_dir
# - 在多机/多卡环境里把缓存定向到更合适的盘
LOG_DIR = os.environ.get("MAS_LOG_DIR", os.path.join(EXPERIMENT_ROOT, "logs"))
PLOT_DIR = os.environ.get("MAS_PLOT_DIR", os.path.join(EXPERIMENT_ROOT, "plots"))
MODEL_CACHE = os.environ.get("MAS_MODEL_CACHE", os.path.join(EXPERIMENT_ROOT, "models"))
DATA_CACHE = os.environ.get("MAS_DATA_CACHE", os.path.join(EXPERIMENT_ROOT, "data"))

# 在最早阶段就把 Hugging Face / ModelScope / datasets 的缓存统一重定向到实验目录。
# 这样后面不管是谁先 import：
# - transformers
# - datasets
# - modelscope
# 都不应该再回落到 /root/.cache。
HF_HOME_DIR = os.path.join(DATA_CACHE, "_hf_home")
HF_DATASETS_CACHE_DIR = os.path.join(DATA_CACHE, "_hf_datasets")
HF_MODULES_CACHE_DIR = os.path.join(DATA_CACHE, "_hf_modules")
MODELSCOPE_CACHE_DIR = MODEL_CACHE
TMPDIR_DIR = os.path.join(DATA_CACHE, "_tmp")

for path in [
    LOG_DIR,
    PLOT_DIR,
    MODEL_CACHE,
    DATA_CACHE,
    HF_HOME_DIR,
    HF_DATASETS_CACHE_DIR,
    HF_MODULES_CACHE_DIR,
    TMPDIR_DIR,
]:
    os.makedirs(path, exist_ok=True)

os.environ["HF_HOME"] = HF_HOME_DIR
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE_DIR
os.environ["HF_MODULES_CACHE"] = HF_MODULES_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME_DIR, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME_DIR, "transformers")
os.environ["MODELSCOPE_CACHE"] = MODELSCOPE_CACHE_DIR
os.environ["TMPDIR"] = TMPDIR_DIR

# π0/π1弱Prompt定义（仅行为约束，无固定角色）
PI0_PROMPT = (
    "请只围绕“当前最开始那道数学题”做协作思考和补充，不要给出最终答案。"
    "如果上下文里已经有上一版答案，只有在你能明确指出那份答案里的具体错误、漏项或单位问题时，才提出修订意见；"
    "如果暂时没发现明确错误，就直接说明“当前答案暂未发现明确错误，建议保持”。"
    "如果上下文里出现了别的题目、例子、课堂故事、铅笔/橡皮/学生等无关内容，一律视为噪声并忽略，绝对不要把它们当成当前题目继续展开。"
    "不要改写题目，不要引入新题，不要编造新的数字或场景。"
    "只输出1到2句简短建议，帮助检查当前原题的已知条件、计算步骤或单位；不要复述整段历史，也不要使用1. 2. 3.这类编号列表。\n"
    "上下文：{context}\n思考："
)
PI1_PROMPT = (
    "请只根据“当前最开始那道数学题”作答。"
    "如果上下文里已经有上一版答案，默认应当保持这份答案；只有在你能明确指出上一版答案的具体算术错误、漏项或单位错误时，才允许修改。"
    "不要为了“再想一遍”而随意换一个新数字。"
    "如果上下文里混入了别的题目、例子或无关数字，一律忽略，只保留与原题直接相关的信息。"
    "给出必要步骤，且最后一行必须单独写成“最终答案：<数字>”，不要在这之后继续输出别的编号、列表、额外题目或新内容。\n"
    "上下文：{context}\n答案："
)
# 单LLM轮询提示词（角色一致，表述不同）
SINGLE_PROMPTS = [
    "请只根据“当前最开始那道数学题”作答。如果上下文里混入了别的题目、例子或无关数字，一律忽略，只保留与原题直接相关的信息。给出必要步骤，最后一行必须单独写成“最终答案：<数字>”，不要在这之后继续输出别的编号、列表、额外题目或新内容。\n上下文：{context}\n答案：",
    "请只围绕“当前最开始那道数学题”做协作思考和补充，不要给出最终答案。如果上下文里已经有上一版答案，只有在你能明确指出那份答案里的具体错误、漏项或单位问题时，才提出修订意见；如果暂时没发现明确错误，就直接说明“当前答案暂未发现明确错误，建议保持”。如果上下文里出现了别的题目、例子、课堂故事、铅笔/橡皮/学生等无关内容，一律视为噪声并忽略，绝对不要把它们当成当前题目继续展开。不要改写题目，不要引入新题，不要编造新的数字或场景。只输出1到2句简短建议，不要复述整段历史，也不要使用1. 2. 3.这类编号列表。\n上下文：{context}\n思考：",
    "请只根据“当前最开始那道数学题”作答。如果上下文里已经有上一版答案，默认应当保持这份答案；只有在你能明确指出上一版答案的具体算术错误、漏项或单位错误时，才允许修改。不要为了“再想一遍”而随意换一个新数字。如果上下文里混入了别的题目、例子或无关数字，一律忽略，只保留与原题直接相关的信息。给出必要步骤，最后一行必须单独写成“最终答案：<数字>”，不要在这之后继续输出别的编号、列表、额外题目或新内容。\n上下文：{context}\n答案：",
    "请只围绕“当前最开始那道数学题”做协作思考和补充，不要给出最终答案。如果上下文里已经有上一版答案，只有在你能明确指出那份答案里的具体错误、漏项或单位问题时，才提出修订意见；如果暂时没发现明确错误，就直接说明“当前答案暂未发现明确错误，建议保持”。如果上下文里出现了别的题目、例子、课堂故事、铅笔/橡皮/学生等无关内容，一律视为噪声并忽略，绝对不要把它们当成当前题目继续展开。不要改写题目，不要引入新题，不要编造新的数字或场景。只输出1到2句简短建议，不要复述整段历史，也不要使用1. 2. 3.这类编号列表。\n上下文：{context}\n思考：",
    "请只根据“当前最开始那道数学题”作答。如果上下文里已经有上一版答案，默认应当保持这份答案；只有在你能明确指出上一版答案的具体算术错误、漏项或单位错误时，才允许修改。不要为了“再想一遍”而随意换一个新数字。如果上下文里混入了别的题目、例子或无关数字，一律忽略，只保留与原题直接相关的信息。给出必要步骤，最后一行必须单独写成“最终答案：<数字>”，不要在这之后继续输出别的编号、列表、额外题目或新内容。\n上下文：{context}\n答案：",
]
# 轮询双LLM提示词（解答者+评论员）
SOLVER_PROMPT = (
    "请只根据“当前最开始那道数学题”作答。"
    "如果上下文里已经有上一版答案，默认应当保持这份答案；只有在你能明确指出上一版答案的具体算术错误、漏项或单位错误时，才允许修改。"
    "不要为了“再想一遍”而随意换一个新数字。"
    "如果上下文里混入了别的题目、例子或无关数字，一律忽略，只保留与原题直接相关的信息。"
    "给出必要步骤，最后一行必须单独写成“最终答案：<数字>”，不要在这之后继续输出别的编号、列表、额外题目或新内容。\n"
    "上下文：{context}\n答案："
)
COMMENTER_PROMPT = (
    "请只围绕“当前最开始那道数学题”做协作思考和补充，不要给出最终答案。"
    "如果上下文里已经有上一版答案，只有在你能明确指出那份答案里的具体错误、漏项或单位问题时，才提出修订意见；"
    "如果暂时没发现明确错误，就直接说明“当前答案暂未发现明确错误，建议保持”。"
    "如果上下文里出现了别的题目、例子、课堂故事、铅笔/橡皮/学生等无关内容，一律视为噪声并忽略，绝对不要把它们当成当前题目继续展开。"
    "不要改写题目，不要引入新题，不要编造新的数字或场景。"
    "只输出1到2句简短建议，不要复述整段历史，也不要使用1. 2. 3.这类编号列表。\n"
    "上下文：{context}\n思考："
)
