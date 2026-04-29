import os


def _env_flag(name, default="0"):
    value = os.environ.get(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

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
TRAIN_NUM_ROUNDS = int(os.environ.get("MAS_TRAIN_NUM_ROUNDS", "5") or "5")
INFER_NUM_ROUNDS = int(
    os.environ.get("MAS_INFER_NUM_ROUNDS", str(TRAIN_NUM_ROUNDS)) or str(TRAIN_NUM_ROUNDS)
)
# 兼容旧代码保留；新代码请优先使用 TRAIN_NUM_ROUNDS / INFER_NUM_ROUNDS。
NUM_ROUNDS = TRAIN_NUM_ROUNDS
GLOBAL_SEED = int(os.environ.get("MAS_GLOBAL_SEED", "42") or "42")

# 模型配置
# 默认仍保持当前稳定跑通的 Phi-3，
# 但允许通过 MAS_MODEL_SCOPE 在启动时切到别的底模，例如 Qwen2.5-7B-Instruct。
GPT2_MODEL_SCOPE = os.environ.get(
    "MAS_MODEL_SCOPE",
    "LLM-Research/Phi-3-mini-4k-instruct",
)
CHAT_SYSTEM_PROMPT = os.environ.get(
    "MAS_CHAT_SYSTEM_PROMPT",
    (
        "You are a careful reasoning assistant. "
        "Solve or review the user's math task directly. "
        "Keep the reasoning concise enough to finish within the token budget. "
        "Follow the required final-line format exactly."
    ),
).strip()
MAX_NEW_TOKENS = int(os.environ.get("MAS_MAX_NEW_TOKENS", "128") or "128")
TRAIN_MAX_NEW_TOKENS = int(
    os.environ.get("MAS_TRAIN_MAX_NEW_TOKENS", str(MAX_NEW_TOKENS)) or str(MAX_NEW_TOKENS)
)
EVAL_MAX_NEW_TOKENS = int(
    os.environ.get("MAS_EVAL_MAX_NEW_TOKENS", str(MAX_NEW_TOKENS)) or str(MAX_NEW_TOKENS)
)
PROPOSAL_COMPLETION_MAX_NEW_TOKENS = int(
    os.environ.get("MAS_PROPOSAL_COMPLETION_MAX_NEW_TOKENS", "32") or "32"
)
REVIEW_COMPLETION_MAX_NEW_TOKENS = int(
    os.environ.get("MAS_REVIEW_COMPLETION_MAX_NEW_TOKENS", "16") or "16"
)
PROPOSAL_REVIEW_OUTPUT_MODE = os.environ.get(
    "MAS_PROPOSAL_REVIEW_OUTPUT_MODE",
    "structured_judgment",
).strip().lower() or "structured_judgment"
# 降低生成随机性，减少早期在线训练时“偶然好/偶然坏”答案对更新方向的干扰。
TEMPERATURE = 0.3

# 算法超参
# 再把学习率降一档，让在线更新更保守，避免少量样本就把策略带偏。
GSPO_LR = 5e-6
# 候选数略增，提升组内相对比较的稳定性；5条样本复验时这点额外开销可接受。
GSPO_NUM_CANDIDATES = 4
# 训练阶段允许在一组 GSPO candidates 里混入少量 greedy 候选，
# 用来缩小“训练时全采样、评测时偏确定性”带来的分布偏移。
# 例如设为 1 且 G=4 时，就是 1 greedy + 3 sampled。
GSPO_NUM_GREEDY_CANDIDATES = int(
    os.environ.get("MAS_GSPO_NUM_GREEDY_CANDIDATES", "0") or "0"
)
# 缩小裁剪区间，减少一次更新把策略推太远。
GSPO_CLIP_EPS = 0.1
GSPO_USE_INDEPENDENT_MODEL = True  # 每个GSPO agent使用独立Phi-3副本，避免多agent共享同一活跃策略
GSPO_UPDATE_CANDIDATE_CHUNK_SIZE = int(
    os.environ.get("MAS_GSPO_UPDATE_CANDIDATE_CHUNK_SIZE", "0") or "0"
)
# GSPO 默认改成 LoRA：
# - 用户现在明确希望不再走“只训最后几层”的路径
# - 如需复现实验，可用 MAS_GSPO_FINETUNE_MODE=last_n_layers 覆盖
GSPO_FINETUNE_MODE = os.environ.get("MAS_GSPO_FINETUNE_MODE", "lora").strip().lower()
GSPO_TRAIN_LAST_N_LAYERS = 2       # 除lm_head外，再解冻最后N层decoder block
# LoRA 默认做轻量化，优先减少在线更新耗时与优化器开销。
GSPO_LORA_R = int(os.environ.get("MAS_GSPO_LORA_R", "8") or "8")
GSPO_LORA_ALPHA = int(os.environ.get("MAS_GSPO_LORA_ALPHA", "16") or "16")
GSPO_LORA_DROPOUT = float(os.environ.get("MAS_GSPO_LORA_DROPOUT", "0.0") or "0.0")
GSPO_LORA_BIAS = os.environ.get("MAS_GSPO_LORA_BIAS", "none").strip().lower() or "none"
GSPO_LORA_TARGET_MODULES = os.environ.get("MAS_GSPO_LORA_TARGET_MODULES", "").strip()
# GSPO 在线更新的训练精度。
# 这里默认仍保守使用 float32：
# - LoRA 已经显著减轻了反向与优化器开销
# - proposal/review 这条链路的耗时热点更多在生成，不在 adapter 参数量
# - 先优先保证现有机器上的训练稳定性
#
# 可选值：
# - "bfloat16": 训练时把 GSPO 独立副本转成 bf16
# - "float32":  训练时转成 fp32，更稳但更占显存
# - "float16":  保留 fp16，不推荐
# - "auto":     CUDA 支持 bf16 时用 bf16，否则退回 fp32
GSPO_TRAIN_DTYPE = os.environ.get("MAS_GSPO_TRAIN_DTYPE", "float32")
GSPO_EVAL_DO_SAMPLE = _env_flag("MAS_GSPO_EVAL_DO_SAMPLE", "1")
# 虚拟 rollout 明确只采样 1 个 next answer：
# - 真实轨迹里的 comment / answer 仍然是一整组候选
# - 但在 comment 奖励评估、silent 基线、反事实轨迹里，
#   每条虚拟轨迹只继续生成 1 个 answer，避免 comment 下再展开一组 answer
#   导致计算量近似 g^g 爆炸
GSPO_COMMENT_EVAL_SAMPLES = 1
# 批量 next-answer 评估时，每次送进 generate 的最大 context 数。
# 只影响执行时的 micro-batch，不改变实验逻辑。
# 设为 0 或负数表示“不额外分块”，一次性处理全部 contexts。
GSPO_SAMPLE_TEXT_BATCH_SIZE = int(
    os.environ.get("MAS_GSPO_SAMPLE_TEXT_BATCH_SIZE", "0") or "0"
)
GSPO_SAMPLE_CANDIDATE_CHUNK_SIZE = int(
    os.environ.get("MAS_GSPO_SAMPLE_CANDIDATE_CHUNK_SIZE", "0") or "0"
)
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
# 中层动作语义模式：
# - "legacy": 旧版 silent / comment / answer
# - "proposal_review": 奇数轮是 proposal stage，偶数轮是 review stage；
#   无 pending 时强制 proposer 先提出一个 pending；
#   controller 只在“无 incumbent 且已有 pending”的 proposal stage 决定 keep/refresh；
#   一旦形成 incumbent，后续 proposal stage 直接 stop；
#   reviewer 在 review stage 只评审当前 pending，并输出单张 ACCEPT / REJECT 投票
THREE_LAYER_MIDDLE_ACTION_SCHEMA = (
    os.environ.get("MAS_THREE_LAYER_MIDDLE_ACTION_SCHEMA", "legacy").strip().lower()
    or "legacy"
)
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
# 训练阶段是否把每轮所有候选都展开成下一轮分支。
# 打开后会把 5 轮、4 个候选的真实轨迹扩成近似树搜索，计算量会急剧膨胀；
# 默认关闭，只沿“真实选中的候选”继续后续轮次，把复杂度压回线性。
THREE_LAYER_EXPAND_ALL_TRAIN_BRANCHES = False
# 真实训练轨迹在一批 candidates 中选择哪条分支继续：
# - "first": 保持当前默认行为，继续第 0 个候选
# - "reward_best": 奖励算完后，沿本批 reward/value 最好的候选继续
THREE_LAYER_REALIZED_BRANCH_SELECTION = (
    os.environ.get("MAS_THREE_LAYER_REALIZED_BRANCH_SELECTION", "first").strip().lower()
    or "first"
)
# 中层/外层 CFR 更新在“已选动作”上使用哪些 realized value：
# - "all_candidates": 保持当前行为，对同一动作下整批 candidate values 逐个更新
# - "selected_only": 只用真实继续下去的那一个 candidate value 更新
THREE_LAYER_REGRET_UPDATE_MODE = (
    os.environ.get("MAS_THREE_LAYER_REGRET_UPDATE_MODE", "all_candidates").strip().lower()
    or "all_candidates"
)
# answer 不可解析时，是否保留已有 incumbent，而不是被新的坏 answer 覆盖掉。
THREE_LAYER_KEEP_INCUMBENT_ON_MISSING_ANSWER = _env_flag(
    "MAS_THREE_LAYER_KEEP_INCUMBENT_ON_MISSING_ANSWER",
    "0",
)
# 诊断开关：若当前已经进入 stabilize 且存在 incumbent，
# 是否直接把中层的 answer 动作从可选动作中移除。
# 用于检查“最终掉点是否主要来自后续轮继续改答案”。
THREE_LAYER_DISABLE_ANSWER_ON_STABILIZE_WITH_INCUMBENT = _env_flag(
    "MAS_THREE_LAYER_DISABLE_ANSWER_ON_STABILIZE_WITH_INCUMBENT",
    "0",
)
# 诊断开关：即使本轮仍然选择了 answer，
# 只要当前已在 stabilize 且存在 incumbent，
# 对“不同于 incumbent 的新数值答案”就不做覆盖，继续保留旧 incumbent。
# 这是一个纯基于可观测状态的保守 gate，不依赖 ground-truth。
THREE_LAYER_KEEP_INCUMBENT_ON_STABILIZE_DIFFERENT_ANSWER = _env_flag(
    "MAS_THREE_LAYER_KEEP_INCUMBENT_ON_STABILIZE_DIFFERENT_ANSWER",
    "0",
)
# 评估阶段是否让外层 / 中层动作走确定性 argmax，而不是继续按平均策略采样。
THREE_LAYER_EVAL_DETERMINISTIC_ACTIONS = _env_flag(
    "MAS_THREE_LAYER_EVAL_DETERMINISTIC_ACTIONS",
    "0",
)
# 中层动作的诊断覆盖模式：
# - "learned": 保持当前 learned policy
# - "always_answer": 每轮都强制 answer
# - "fixed_answer_comment": 奇数轮 answer，偶数轮 comment
THREE_LAYER_ACTION_OVERRIDE_MODE = (
    os.environ.get("MAS_THREE_LAYER_ACTION_OVERRIDE_MODE", "learned").strip().lower()
    or "learned"
)
# final paper ablations. Defaults exactly preserve experiment 30.
PROPOSAL_REVIEW_CONTROLLER_OVERRIDE_MODE = (
    os.environ.get("MAS_PROPOSAL_REVIEW_CONTROLLER_OVERRIDE_MODE", "learned").strip().lower()
    or "learned"
)
PROPOSAL_REVIEW_DISABLE_CONTROLLER_REGRET = _env_flag(
    "MAS_PROPOSAL_REVIEW_DISABLE_CONTROLLER_REGRET",
    "0",
)
PROPOSAL_REVIEW_DISABLE_COUNTERFACTUAL_VALUES = _env_flag(
    "MAS_PROPOSAL_REVIEW_DISABLE_COUNTERFACTUAL_VALUES",
    "0",
)
PROPOSAL_REVIEW_DISABLE_VOTE_UPDATES = _env_flag(
    "MAS_PROPOSAL_REVIEW_DISABLE_VOTE_UPDATES",
    "0",
)
PROPOSAL_REVIEW_APPLY_VOTE_THRESHOLDS = _env_flag(
    "MAS_PROPOSAL_REVIEW_APPLY_VOTE_THRESHOLDS",
    "0",
)
PROPOSAL_REVIEW_REFRESH_REQUIRES_NEGATIVE_PENDING = _env_flag(
    "MAS_PROPOSAL_REVIEW_REFRESH_REQUIRES_NEGATIVE_PENDING",
    "0",
)
PROPOSAL_REVIEW_DISABLE_REFRESH = _env_flag(
    "MAS_PROPOSAL_REVIEW_DISABLE_REFRESH",
    "0",
)
PROPOSAL_REVIEW_DISABLE_REVIEW_FEEDBACK_CONTEXT = _env_flag(
    "MAS_PROPOSAL_REVIEW_DISABLE_REVIEW_FEEDBACK_CONTEXT",
    "0",
)
PROPOSAL_REVIEW_USE_LEGACY_PROMPTS = _env_flag(
    "MAS_PROPOSAL_REVIEW_USE_LEGACY_PROMPTS",
    "0",
)
THREE_LAYER_DISABLE_PI1_UPDATES = _env_flag(
    "MAS_THREE_LAYER_DISABLE_PI1_UPDATES",
    "0",
)
THREE_LAYER_DISABLE_ALL_GSPO_UPDATES = _env_flag(
    "MAS_THREE_LAYER_DISABLE_ALL_GSPO_UPDATES",
    "0",
)
# comment/reviewer effectiveness diagnostics.
THREE_LAYER_DISABLE_PI0_UPDATES = _env_flag(
    "MAS_THREE_LAYER_DISABLE_PI0_UPDATES",
    "0",
)
# 状态 key 的诊断模式：
# - "phase_only": 只用当前 phase
# - "expanded": 额外拼入 has_incumbent / round bucket
THREE_LAYER_STATE_KEY_MODE = (
    os.environ.get("MAS_THREE_LAYER_STATE_KEY_MODE", "phase_only").strip().lower()
    or "phase_only"
)

# 奖励形状：
# - final accuracy 看 exact match
# - RL 训练仍保留少量 dense reward，避免全错时完全没梯度信号
REWARD_EXACT_WEIGHT = 0.8
REWARD_DENSE_WEIGHT = 0.2
REWARD_MISSING_PENALTY = -0.2
PROPOSAL_FORMAT_BONUS = float(
    os.environ.get("MAS_PROPOSAL_FORMAT_BONUS", "0.05") or "0.05"
)
PROPOSAL_CANDIDATE_ONLY_PENALTY = float(
    os.environ.get("MAS_PROPOSAL_CANDIDATE_ONLY_PENALTY", "0.10") or "0.10"
)
PROPOSAL_TRUNCATION_PENALTY = float(
    os.environ.get("MAS_PROPOSAL_TRUNCATION_PENALTY", "0.10") or "0.10"
)
PROPOSAL_CONFLICT_PENALTY = float(
    os.environ.get("MAS_PROPOSAL_CONFLICT_PENALTY", "0.30") or "0.30"
)
REVIEW_SHELL_REASON_PENALTY = float(
    os.environ.get("MAS_REVIEW_SHELL_REASON_PENALTY", "0.05") or "0.05"
)
REVIEW_INVALID_JUDGMENT_PENALTY = float(
    os.environ.get("MAS_REVIEW_INVALID_JUDGMENT_PENALTY", "0.05") or "0.05"
)
PROPOSAL_REFRESH_SAME_PRED_PENALTY = float(
    os.environ.get("MAS_PROPOSAL_REFRESH_SAME_PRED_PENALTY", "0.0") or "0.0"
)
REVIEW_MIN_REASON_COMPACT_CHARS = int(
    os.environ.get("MAS_REVIEW_MIN_REASON_COMPACT_CHARS", "8") or "8"
)
# rollout 到结尾仍没得到可解析答案时：
# - 若没有 incumbent，给更重的终端惩罚
# - 若已有 incumbent，则在 delta 口径下记为 0，在 absolute 口径下回退到 incumbent reward
ROLLOUT_NO_ANSWER_PENALTY = -1.0

# proposal-review 协议的 pending 投票阈值：
# - 首个 proposal 先形成 pending，而不是立刻成为 incumbent
# - 没有 incumbent 时，pending 需要累计到 +2 净票才升级为 incumbent
# - 没有 incumbent 时，pending 只要到 -1 净票就立刻丢弃
# - 新版协议下，形成 incumbent 后不再 reopen 新 pending；
#   replace/drop 阈值仅保留为历史兼容项
PROPOSAL_REVIEW_BOOTSTRAP_ACCEPT_SCORE = int(
    os.environ.get("MAS_PROPOSAL_REVIEW_BOOTSTRAP_ACCEPT_SCORE", "2") or "2"
)
PROPOSAL_REVIEW_BOOTSTRAP_DROP_SCORE = int(
    os.environ.get("MAS_PROPOSAL_REVIEW_BOOTSTRAP_DROP_SCORE", "-1") or "-1"
)
PROPOSAL_REVIEW_REPLACE_ACCEPT_SCORE = int(
    os.environ.get("MAS_PROPOSAL_REVIEW_REPLACE_ACCEPT_SCORE", "2") or "2"
)
PROPOSAL_REVIEW_DROP_SCORE = int(
    os.environ.get("MAS_PROPOSAL_REVIEW_DROP_SCORE", "-1") or "-1"
)

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
SEARCH_MIN_COMMENT_PROB = 0.25
SEARCH_MIN_ANSWER_PROB = 0.35
SEARCH_MAX_SILENT_PROB = 0.10
SEARCH_RELAX_KEEP_PROB = 0.55
SEARCH_LOW_KEEP_PROB = 0.45
SEARCH_UNSTABLE_MIN_ANSWER_PROB = 0.55
SEARCH_UNSTABLE_MAX_SILENT_PROB = 0.05
SEARCH_NO_INCUMBENT_MIN_ANSWER_PROB = 0.75
SEARCH_NO_INCUMBENT_MAX_SILENT_PROB = 0.00
SEARCH_LATE_ROUND_ANSWER_BONUS = 0.15
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
STRUCTURED_CONTEXT_MODE = (
    os.environ.get("MAS_STRUCTURED_CONTEXT_MODE", "incumbent_comments").strip().lower()
    or "incumbent_comments"
)
STRUCTURED_CONTEXT_MAX_TURNS = int(
    os.environ.get("MAS_STRUCTURED_CONTEXT_MAX_TURNS", "6") or "6"
)

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
CHECKPOINT_DIR = os.environ.get(
    "MAS_CHECKPOINT_DIR",
    os.path.join(os.path.dirname(LOG_DIR), "checkpoints"),
)
CHECKPOINT_EVERY_SAMPLES = int(
    os.environ.get("MAS_CHECKPOINT_EVERY_SAMPLES", "0") or "0"
)
RESUME_CHECKPOINT_PATH = os.environ.get("MAS_RESUME_CHECKPOINT", "").strip() or None

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
    CHECKPOINT_DIR,
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
PI0_PROMPT_RELAXED = (
    "请只围绕当前最开始那道数学题补充关键信息，不要给出最终答案。"
    "优先指出当前解法里最可能出错的步骤、遗漏条件、单位或算式；"
    "如果你认为需要推翻上一版答案，可以直接说明哪一步不成立以及应该重算什么。"
    "允许列出简短的检查点，但不要引入新题、不要编造新的数字或场景。"
    "尽量给出能直接帮助下一轮重算的具体提示，而不是只说“建议保持”。\n"
    "上下文：{context}\n思考："
)
PI1_PROMPT_RELAXED = (
    "请重新求解当前最开始那道数学题。"
    "如果上下文里已有上一版答案，把它当作待检验候选，而不是默认正确；"
    "当你发现算术、建模、单位或条件使用有问题时，应直接改正并重算。"
    "如果上下文里混入别的题目、例子或无关数字，一律忽略，只保留与原题直接相关的信息。"
    "给出必要步骤，最后一行必须单独写成“最终答案：<数字>”。\n"
    "上下文：{context}\n答案："
)
PI0_PROMPT_PROPOSAL_REVIEW = (
    "{context}\n\n"
    "You are a careful reasoning assistant.\n"
    "Check whether the current pending final answer is correct for the original math problem.\n"
    "First line: RIGHT or WRONG.\n"
    "Second line: one short reason.\n"
    "Do not write a new solution or a new final answer.\n\n"
    "Review:\n"
)
PI1_PROMPT_PROPOSAL_REVIEW = (
    "{context}\n\n"
    "Solve the original math problem directly.\n"
    "If there is a pending solution or short review feedback, use it only as a hint.\n"
    "Keep the reasoning concise.\n"
    "End with exactly one final line: Final answer: <number>.\n\n"
    "Solution:\n"
)
PI1_PROMPT_PROPOSAL_REVIEW_REFRESH = (
    "{context}\n\n"
    "Solve the original math problem directly.\n"
    "Use the current pending solution and the latest review only as hints.\n"
    "If the latest review says WRONG, fix the decisive mistake and recompute.\n"
    "If the latest review says RIGHT, keep the answer unless you find a clear error.\n"
    "Keep the reasoning concise.\n"
    "End with exactly one final line: Final answer: <number>.\n\n"
    "Solution:\n"
)

if THREE_LAYER_MIDDLE_ACTION_SCHEMA == "proposal_review":
    PI0_PROMPT = PI0_PROMPT_PROPOSAL_REVIEW
    PI1_PROMPT = PI1_PROMPT_PROPOSAL_REVIEW
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
