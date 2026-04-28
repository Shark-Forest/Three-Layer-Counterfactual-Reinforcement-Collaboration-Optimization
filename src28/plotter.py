import os
import matplotlib.pyplot as plt
from matplotlib import font_manager
from src28.metrics_logger import load_logs
from src28.config import *

os.makedirs(PLOT_DIR, exist_ok=True)

def _infer_logged_num_rounds(*logs):
    max_round = 0
    for log in logs:
        rounds = log.get("round", [])
        if rounds:
            max_round = max(max_round, int(max(rounds)))
    return max_round

def _pick_cjk_font():
    candidates = [
        "SimHei",
        "WenQuanYi Zen Hei",
        "WenQuanYi Zen Hei Mono",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available:
            return font_name
    return "DejaVu Sans"

plt.rcParams["font.sans-serif"] = [_pick_cjk_font()]
plt.rcParams["axes.unicode_minus"] = False

def plot_accuracy_comparison():
    """多组实验准确率对比图"""
    single = load_logs("single")
    polling = load_logs("polling")
    single_gspo = load_logs("single_gspo")
    dual_gspo = load_logs("dual_gspo")
    middle = load_logs("middle_layer")
    middle_no_silent = load_logs("middle_layer_no_silent")
    three = load_logs("three_layer")
    three_no_silent = load_logs("three_layer_no_silent")
    num_rounds = _infer_logged_num_rounds(
        single,
        polling,
        single_gspo,
        dual_gspo,
        middle,
        middle_no_silent,
        three,
        three_no_silent,
    )

    plt.figure(figsize=(13, 8))
    plt.plot(single["round"], single["accuracy"], "o-", label="单LLM", linewidth=2)
    plt.plot(polling["round"], polling["accuracy"], "s-", label="双LLM轮询", linewidth=2)
    plt.plot(single_gspo["round"], single_gspo["accuracy"], "d-", label="单LLM GSPO", linewidth=2)
    plt.plot(dual_gspo["round"], dual_gspo["accuracy"], "x-", label="双LLM GSPO轮询", linewidth=2)
    plt.plot(middle["round"], middle["accuracy"], "v-", label="中间层策略+GSPO", linewidth=2)
    plt.plot(middle_no_silent["round"], middle_no_silent["accuracy"], "P-", label="中间层策略+GSPO（无沉默）", linewidth=2)
    plt.plot(three["round"], three["accuracy"], "^-", label="全量策略", linewidth=2)
    plt.plot(three_no_silent["round"], three_no_silent["accuracy"], "*-", label="全量策略（中层无沉默）", linewidth=2)

    plt.xlabel("对话轮次", fontsize=12)
    plt.ylabel("GSM8K解题准确率", fontsize=12)
    plt.title(f"多组实验{num_rounds}轮准确率对比", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xticks(range(1, num_rounds + 1))
    plt.savefig(f"{PLOT_DIR}/accuracy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_three_layer_details():
    """全量策略全指标可视化"""
    df = load_logs("three_layer")
    rounds = df["round"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].plot(rounds, df["accuracy"], "b-o", label="准确率")
    axes[0,0].plot(rounds, df["total_reward"], "r-s", label="累计奖励")
    axes[0,0].set_title("准确率与累计奖励")
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(rounds, df["cfr_regret"], "g-^", label="CFR累计遗憾")
    axes[0,1].set_title("CFR反事实遗憾变化")
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)

    axes[1,0].plot(rounds, df["gspo_agent0_prob"], "o-", label="Agent0选中概率")
    axes[1,0].plot(rounds, df["gspo_agent1_prob"], "s-", label="Agent1选中概率")
    axes[1,0].set_title("GSPO调度策略变化")
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)

    axes[1,1].plot(rounds, df["cfr_silent_prob"], "o-", label="静默概率")
    axes[1,1].plot(rounds, df["cfr_comment_prob"], "s-", label="评论概率")
    axes[1,1].plot(rounds, df["cfr_answer_prob"], "^-", label="答案概率")
    axes[1,1].set_title("中层动作策略变化")
    axes[1,1].legend()
    axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/three_layer_details.png", dpi=300, bbox_inches="tight")
    plt.close()
