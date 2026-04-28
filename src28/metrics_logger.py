import os
import pandas as pd
from src28.config import *

os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATHS = {
    "single": f"{LOG_DIR}/single_llm.csv",
    "polling": f"{LOG_DIR}/polling_two_llms.csv",
    "single_gspo": f"{LOG_DIR}/single_gspo.csv",
    "dual_gspo": f"{LOG_DIR}/dual_gspo.csv",
    "middle_layer": f"{LOG_DIR}/middle_layer.csv",
    "middle_layer_no_silent": f"{LOG_DIR}/middle_layer_no_silent.csv",
    "three_layer": f"{LOG_DIR}/three_layer.csv",
    "three_layer_no_silent": f"{LOG_DIR}/three_layer_no_silent.csv",
}

def init_logs():
    """初始化日志文件"""
    pd.DataFrame(columns=["round", "accuracy", "best_so_far_accuracy"]).to_csv(LOG_PATHS["single"], index=False)
    pd.DataFrame(columns=["round", "accuracy", "best_so_far_accuracy"]).to_csv(LOG_PATHS["polling"], index=False)
    pd.DataFrame(columns=["round", "accuracy", "best_so_far_accuracy"]).to_csv(LOG_PATHS["single_gspo"], index=False)
    pd.DataFrame(columns=["round", "accuracy", "best_so_far_accuracy"]).to_csv(LOG_PATHS["dual_gspo"], index=False)
    pd.DataFrame(columns=[
        "round", "accuracy", "best_so_far_accuracy", "total_reward", "cfr_regret",
        "gspo_agent0_prob", "gspo_agent1_prob",
        "cfr_silent_prob", "cfr_comment_prob", "cfr_answer_prob",
        "search_rate", "stabilize_rate",
        "improve_rate", "degrade_rate", "stalled_wrong_rate", "preserved_correct_rate"
    ]).to_csv(LOG_PATHS["middle_layer"], index=False)
    pd.DataFrame(columns=[
        "round", "accuracy", "best_so_far_accuracy", "total_reward", "cfr_regret",
        "gspo_agent0_prob", "gspo_agent1_prob",
        "cfr_silent_prob", "cfr_comment_prob", "cfr_answer_prob",
        "search_rate", "stabilize_rate",
        "improve_rate", "degrade_rate", "stalled_wrong_rate", "preserved_correct_rate"
    ]).to_csv(LOG_PATHS["middle_layer_no_silent"], index=False)
    pd.DataFrame(columns=[
        "round", "accuracy", "best_so_far_accuracy", "total_reward", "cfr_regret",
        "gspo_agent0_prob", "gspo_agent1_prob",
        "cfr_silent_prob", "cfr_comment_prob", "cfr_answer_prob",
        "search_rate", "stabilize_rate",
        "improve_rate", "degrade_rate", "stalled_wrong_rate", "preserved_correct_rate"
    ]).to_csv(LOG_PATHS["three_layer"], index=False)
    pd.DataFrame(columns=[
        "round", "accuracy", "best_so_far_accuracy", "total_reward", "cfr_regret",
        "gspo_agent0_prob", "gspo_agent1_prob",
        "cfr_silent_prob", "cfr_comment_prob", "cfr_answer_prob",
        "search_rate", "stabilize_rate",
        "improve_rate", "degrade_rate", "stalled_wrong_rate", "preserved_correct_rate"
    ]).to_csv(LOG_PATHS["three_layer_no_silent"], index=False)

def log_single(round_num, acc, best_acc):
    df = pd.read_csv(LOG_PATHS["single"])
    df.loc[len(df)] = [round_num, acc, best_acc]
    df.to_csv(LOG_PATHS["single"], index=False)

def log_polling(round_num, acc, best_acc):
    df = pd.read_csv(LOG_PATHS["polling"])
    df.loc[len(df)] = [round_num, acc, best_acc]
    df.to_csv(LOG_PATHS["polling"], index=False)

def log_single_gspo(round_num, acc, best_acc):
    df = pd.read_csv(LOG_PATHS["single_gspo"])
    df.loc[len(df)] = [round_num, acc, best_acc]
    df.to_csv(LOG_PATHS["single_gspo"], index=False)

def log_dual_gspo(round_num, acc, best_acc):
    df = pd.read_csv(LOG_PATHS["dual_gspo"])
    df.loc[len(df)] = [round_num, acc, best_acc]
    df.to_csv(LOG_PATHS["dual_gspo"], index=False)

def _log_policy_stack(
    log_key,
    round_num,
    acc,
    best_acc,
    total_reward,
    regret,
    gspo_probs,
    cfr_probs,
    search_rate,
    stabilize_rate,
    improve_rate,
    degrade_rate,
    stalled_wrong_rate,
    preserved_correct_rate,
):
    df = pd.read_csv(LOG_PATHS[log_key])
    # 兼容诊断时临时把 NUM_AGENTS 降成 1 的情况：
    # 旧版日志格式固定保留 agent0 / agent1 两列，
    # 因此这里不足的部分用 0.0 补齐，避免 1-agent smoke 时写日志报错。
    agent0_prob = gspo_probs[0] if len(gspo_probs) > 0 else 0.0
    agent1_prob = gspo_probs[1] if len(gspo_probs) > 1 else 0.0
    df.loc[len(df)] = [
        round_num, acc, best_acc, total_reward, regret,
        agent0_prob, agent1_prob,
        cfr_probs[0] if len(cfr_probs) > 0 else 0.0,
        cfr_probs[1] if len(cfr_probs) > 1 else 0.0,
        cfr_probs[2] if len(cfr_probs) > 2 else 0.0,
        search_rate,
        stabilize_rate,
        improve_rate,
        degrade_rate,
        stalled_wrong_rate,
        preserved_correct_rate,
    ]
    df.to_csv(LOG_PATHS[log_key], index=False)

def log_middle_layer(
    round_num,
    acc,
    best_acc,
    total_reward,
    regret,
    gspo_probs,
    cfr_probs,
    search_rate,
    stabilize_rate,
    improve_rate,
    degrade_rate,
    stalled_wrong_rate,
    preserved_correct_rate,
):
    _log_policy_stack(
        "middle_layer",
        round_num,
        acc,
        best_acc,
        total_reward,
        regret,
        gspo_probs,
        cfr_probs,
        search_rate,
        stabilize_rate,
        improve_rate,
        degrade_rate,
        stalled_wrong_rate,
        preserved_correct_rate,
    )

def log_middle_layer_no_silent(
    round_num,
    acc,
    best_acc,
    total_reward,
    regret,
    gspo_probs,
    cfr_probs,
    search_rate,
    stabilize_rate,
    improve_rate,
    degrade_rate,
    stalled_wrong_rate,
    preserved_correct_rate,
):
    _log_policy_stack(
        "middle_layer_no_silent",
        round_num,
        acc,
        best_acc,
        total_reward,
        regret,
        gspo_probs,
        cfr_probs,
        search_rate,
        stabilize_rate,
        improve_rate,
        degrade_rate,
        stalled_wrong_rate,
        preserved_correct_rate,
    )

def log_three_layer(
    round_num,
    acc,
    best_acc,
    total_reward,
    regret,
    gspo_probs,
    cfr_probs,
    search_rate,
    stabilize_rate,
    improve_rate,
    degrade_rate,
    stalled_wrong_rate,
    preserved_correct_rate,
):
    _log_policy_stack(
        "three_layer",
        round_num,
        acc,
        best_acc,
        total_reward,
        regret,
        gspo_probs,
        cfr_probs,
        search_rate,
        stabilize_rate,
        improve_rate,
        degrade_rate,
        stalled_wrong_rate,
        preserved_correct_rate,
    )

def log_three_layer_no_silent(
    round_num,
    acc,
    best_acc,
    total_reward,
    regret,
    gspo_probs,
    cfr_probs,
    search_rate,
    stabilize_rate,
    improve_rate,
    degrade_rate,
    stalled_wrong_rate,
    preserved_correct_rate,
):
    _log_policy_stack(
        "three_layer_no_silent",
        round_num,
        acc,
        best_acc,
        total_reward,
        regret,
        gspo_probs,
        cfr_probs,
        search_rate,
        stabilize_rate,
        improve_rate,
        degrade_rate,
        stalled_wrong_rate,
        preserved_correct_rate,
    )

def load_logs(exp_name):
    return pd.read_csv(LOG_PATHS[exp_name])
