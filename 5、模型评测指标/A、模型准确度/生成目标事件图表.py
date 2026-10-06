"""Generate target-event-only comparison data and figures for H2 and H2-plus.

The primary metric is positive-event recall, TP / (TP + FN). It excludes all
background true negatives. The script reads the frozen exact-INT8 summaries so
the published figures stay traceable to the formal Validation/Test evaluations.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Charts and pictures" / "目标事件指标"
MODELS = ("H2", "H2-plus")
ROLES = ("Validation", "Test")
EXPECTED_PROTOCOL = "evuav-33e2aff-official-compatible-v1+step3-density-20000"
EXPECTED_THRESHOLDS = {"H2": 0.30, "H2-plus": 0.40}
EVUAV_PAPER = {
    "model": "EV-SpSegNet（EV-UAV 原论文）",
    "acc": 0.6502,
    "pd": 0.7753,
    "event_ap": None,
    "source": "Chen et al., ICCV 2025, Table 2",
    "source_url": "https://arxiv.org/html/2506.23575",
}
SUMMARY_PATHS = {
    ("Validation", "H2"): ROOT / "0、原始H2模型" / "统一int8评测" / "validation" / "int8" / "summary.json",
    ("Validation", "H2-plus"): ROOT / "1、H2-plus模型" / "validation" / "int8" / "summary.json",
    ("Test", "H2"): ROOT / "0、原始H2模型" / "统一int8评测" / "test" / "int8" / "summary.json",
    ("Test", "H2-plus"): ROOT / "1、H2-plus模型" / "test" / "int8" / "summary.json",
}


def configure_font() -> None:
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    installed = {item.name for item in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = [next((name for name in preferred if name in installed), "DejaVu Sans")]
    plt.rcParams["axes.unicode_minus"] = False


def load_summary(role: str, model: str) -> dict:
    path = SUMMARY_PATHS[(role, model)]
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    threshold = float(data["working_point"]["threshold"])
    protocol = data["evaluation"]["metric_protocol"]
    if not np.isclose(threshold, EXPECTED_THRESHOLDS[model]):
        raise ValueError(f"Unexpected threshold in {path}: {threshold}")
    if protocol != EXPECTED_PROTOCOL:
        raise ValueError(f"Unexpected metric protocol in {path}: {protocol}")
    return data


def metrics_from_summary(data: dict) -> dict:
    working = data["working_point"]
    overall = working["overall"]["event"]
    sequence_recalls = np.asarray(
        [row["event"]["recall"] for row in working["per_sequence"].values()],
        dtype=np.float64,
    )
    result = {
        "event_ap": float(working["average_precision"]),
        "overall_recall": float(overall["recall"]),
        "macro_sequence_recall": float(sequence_recalls.mean()),
        "high_density_recall": float(working["high_density"]["event"]["recall"]),
        "ordinary_density_recall": float(working["ordinary_density"]["event"]["recall"]),
        "true_positive_events": int(overall["tp"]),
        "false_negative_events": int(overall["fn"]),
        "target_events": int(overall["tp"] + overall["fn"]),
        "overall_accuracy_including_background": float(overall["overall_accuracy"]),
        "official_pd": float(working["overall"]["localization"]["pd"]),
    }
    result.update(
        {
            "overall_miss_rate": 1.0 - result["overall_recall"],
            "macro_sequence_miss_rate": 1.0 - result["macro_sequence_recall"],
            "high_density_miss_rate": 1.0 - result["high_density_recall"],
            "ordinary_density_miss_rate": 1.0 - result["ordinary_density_recall"],
        }
    )
    return result


def build_rows(all_metrics: dict) -> list[dict]:
    definitions = [
        ("overall_recall", "Target-event recall", "rate", "all target events", "higher"),
        ("overall_miss_rate", "Target-event miss rate", "rate", "all target events", "lower"),
        ("macro_sequence_recall", "Macro sequence target-event recall", "rate", "equal weight per sequence", "higher"),
        ("macro_sequence_miss_rate", "Macro sequence target-event miss rate", "rate", "equal weight per sequence", "lower"),
        ("high_density_recall", "High-density target-event recall", "rate", "target events in high-density sequences", "higher"),
        ("high_density_miss_rate", "High-density target-event miss rate", "rate", "target events in high-density sequences", "lower"),
        ("ordinary_density_recall", "Ordinary-density target-event recall", "rate", "target events in ordinary-density sequences", "higher"),
        ("ordinary_density_miss_rate", "Ordinary-density target-event miss rate", "rate", "target events in ordinary-density sequences", "lower"),
        ("true_positive_events", "Detected target events", "count", "all target events", "higher"),
        ("false_negative_events", "Missed target events", "count", "all target events", "lower"),
    ]
    rows: list[dict] = []
    for role in ROLES:
        h2 = all_metrics[(role, "H2")]
        plus = all_metrics[(role, "H2-plus")]
        for key, label, measure_type, population, direction in definitions:
            h2_value = h2[key]
            plus_value = plus[key]
            difference = plus_value - h2_value
            result = "better" if (difference > 0) == (direction == "higher") else "worse" if difference else "same"
            rows.append(
                {
                    "data_role": role,
                    "metric": label,
                    "measure_type": measure_type,
                    "population": population,
                    "numeric_path": "exact_int8_replay",
                    "metric_protocol": EXPECTED_PROTOCOL,
                    "h2_threshold": EXPECTED_THRESHOLDS["H2"],
                    "h2_plus_threshold": EXPECTED_THRESHOLDS["H2-plus"],
                    "h2": h2_value,
                    "h2_plus": plus_value,
                    "difference": difference,
                    "difference_percentage_points": difference * 100.0 if measure_type == "rate" else "",
                    "preferred_direction": direction,
                    "h2_plus_result": result,
                }
            )
    return rows


def write_csv(rows: list[dict]) -> Path:
    path = OUTPUT / "H2_vs_H2-plus_目标事件指标.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_benchmark_csv(all_metrics: dict, role: str) -> Path:
    path = OUTPUT / f"EV-UAV_vs_H2_vs_H2-plus_{role}_无TN指标.csv"
    rows = []
    metric_definitions = (
        ("ACC / Target-event recall", "acc", "target events", "EV-UAV paper ACC; H2/H2-plus official_acc=recall"),
        ("Pd", "pd", "50 ms target windows", "EV-UAV-compatible localization Pd"),
        ("Event AP", "event_ap", "event-level precision-recall curve", "TN-independent event AP; not bounding-box COCO mAP"),
    )
    paper_has_values = role == "Test"
    model_rows = [
        {
            "model": EVUAV_PAPER["model"],
            "acc": EVUAV_PAPER["acc"] if paper_has_values else None,
            "pd": EVUAV_PAPER["pd"] if paper_has_values else None,
            "event_ap": EVUAV_PAPER["event_ap"],
            "numeric_path": "paper_reported" if paper_has_values else "paper_not_reported",
            "source": EVUAV_PAPER["source"] if paper_has_values else "EV-UAV paper does not report Validation metrics",
            "source_url": EVUAV_PAPER["source_url"],
        },
        {
            "model": "H2",
            "acc": all_metrics[(role, "H2")]["overall_recall"],
            "pd": all_metrics[(role, "H2")]["official_pd"],
            "event_ap": all_metrics[(role, "H2")]["event_ap"],
            "numeric_path": "exact_int8_replay",
            "source": f"Frozen formal {role} summary.json",
            "source_url": "",
        },
        {
            "model": "H2-plus",
            "acc": all_metrics[(role, "H2-plus")]["overall_recall"],
            "pd": all_metrics[(role, "H2-plus")]["official_pd"],
            "event_ap": all_metrics[(role, "H2-plus")]["event_ap"],
            "numeric_path": "exact_int8_replay",
            "source": f"Frozen formal {role} summary.json",
            "source_url": "",
        },
    ]
    for model in model_rows:
        for metric, key, population, note in metric_definitions:
            value = model[key]
            rows.append(
                {
                    "evaluation_set": f"EV-UAV {role}",
                    "model": model["model"],
                    "metric": metric,
                    "value": "" if value is None else value,
                    "display_percent": "not reported" if value is None else f"{value * 100:.4f}%",
                    "depends_on_background_tn": "false",
                    "population_or_curve": population,
                    "numeric_path": model["numeric_path"],
                    "metric_protocol": model["numeric_path"] if model["numeric_path"].startswith("paper_") else EXPECTED_PROTOCOL,
                    "source": model["source"],
                    "source_url": model["source_url"],
                    "comparison_note": note,
                }
            )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def draw_role(role: str, all_metrics: dict) -> tuple[Path, ...]:
    configure_font()
    h2 = all_metrics[(role, "H2")]
    plus = all_metrics[(role, "H2-plus")]
    category_keys = (
        "overall_recall",
        "macro_sequence_recall",
        "high_density_recall",
        "ordinary_density_recall",
    )
    category_labels = (
        "总体目标事件召回率\n（官方 ACC）",
        "序列宏平均召回率\n（每序列等权）",
        "高密度目标事件召回率",
        "普通密度目标事件召回率",
    )
    h2_values = np.asarray([h2[key] * 100 for key in category_keys])
    plus_values = np.asarray([plus[key] * 100 for key in category_keys])
    y = np.arange(len(category_keys))
    height = 0.32

    figure = plt.figure(figsize=(16, 9), facecolor="#F7F8FA")
    grid = figure.add_gridspec(1, 2, width_ratios=(2.15, 1.0), left=0.07, right=0.96, top=0.80, bottom=0.16, wspace=0.12)
    ax = figure.add_subplot(grid[0, 0])
    info = figure.add_subplot(grid[0, 1])
    ax.set_facecolor("white")
    info.set_facecolor("white")

    h2_color = "#355C7D"
    plus_color = "#D95F02"
    bars_h2 = ax.barh(y - height / 2, h2_values, height=height, color=h2_color, label="H2", zorder=3)
    bars_plus = ax.barh(y + height / 2, plus_values, height=height, color=plus_color, label="H2-plus", zorder=3)
    ax.set_yticks(y, category_labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("召回率（%）— 分母仅含目标事件，不含背景事件", fontsize=12, labelpad=10)
    ax.xaxis.grid(True, color="#D8DEE7", linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#94A3B8")
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=12)

    for bars in (bars_h2, bars_plus):
        for bar in bars:
            ax.text(
                bar.get_width() + 0.7,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.2f}%",
                va="center",
                ha="left",
                fontsize=10,
                color="#0F172A",
            )

    for index, key in enumerate(category_keys):
        delta = (plus[key] - h2[key]) * 100
        color = "#16803C" if delta > 0 else "#B42318" if delta < 0 else "#475569"
        ax.text(
            98.8,
            index,
            f"Δ {delta:+.2f} pp",
            ha="right",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": color, "linewidth": 0.8},
        )

    info.set_xticks([])
    info.set_yticks([])
    for spine in info.spines.values():
        spine.set_color("#D8DEE7")
    info.text(0.06, 0.93, "目标事件命中与漏检", fontsize=15, fontweight="bold", color="#172B4D", transform=info.transAxes)
    info.text(0.06, 0.85, "固定工作点；仅统计 label=1 的事件", fontsize=10.5, color="#64748B", transform=info.transAxes)

    card_y = (0.62, 0.36)
    for y0, model, values, color in zip(card_y, MODELS, (h2, plus), (h2_color, plus_color)):
        info.add_patch(plt.Rectangle((0.06, y0), 0.88, 0.20, transform=info.transAxes, facecolor="#F8FAFC", edgecolor="#D8DEE7", linewidth=1))
        info.add_patch(plt.Rectangle((0.06, y0), 0.018, 0.20, transform=info.transAxes, facecolor=color, edgecolor=color))
        info.text(0.11, y0 + 0.145, model, fontsize=14, fontweight="bold", color=color, transform=info.transAxes)
        info.text(0.11, y0 + 0.082, f"命中 {values['true_positive_events']:,} / {values['target_events']:,}", fontsize=12, color="#0F172A", transform=info.transAxes)
        info.text(0.11, y0 + 0.030, f"漏检 {values['false_negative_events']:,}（{pct(values['overall_miss_rate'])}）", fontsize=12, color="#0F172A", transform=info.transAxes)

    delta_recall = (plus["overall_recall"] - h2["overall_recall"]) * 100
    delta_misses = plus["false_negative_events"] - h2["false_negative_events"]
    direction = "少" if delta_misses < 0 else "多" if delta_misses > 0 else "相同"
    conclusion_color = "#16803C" if delta_recall > 0 else "#B42318" if delta_recall < 0 else "#475569"
    info.text(0.06, 0.29, "H2-plus 相对 H2", fontsize=12, fontweight="bold", color="#334155", transform=info.transAxes)
    info.text(0.06, 0.22, f"总体召回率 {delta_recall:+.3f} pp", fontsize=14, fontweight="bold", color=conclusion_color, transform=info.transAxes)
    info.text(0.06, 0.16, f"漏检目标事件 {abs(delta_misses):,} 个（{direction}）", fontsize=12, color="#0F172A", transform=info.transAxes)
    info.text(0.06, 0.06, "Pd 未作为主图指标：官方兼容规则在 50 ms\n目标窗内命中比例 ≥0.0001 即计为检出，区分度偏弱。", fontsize=10.2, color="#64748B", linespacing=1.45, transform=info.transAxes)

    split_cn = "验证集" if role == "Validation" else "测试集"
    figure.suptitle(f"H2 与 H2-plus：{split_cn}目标事件指标（exact INT8）", x=0.07, y=0.95, ha="left", fontsize=23, fontweight="bold", color="#172B4D")
    figure.text(0.07, 0.875, "核心指标为 Target-event Recall = TP / (TP + FN)，背景真负例 TN 不进入计算。", fontsize=13, color="#475569")
    figure.text(0.07, 0.035, "工作点：H2=0.30，H2-plus=0.40。高密度分层阈值沿用冻结协议；序列宏平均为补充稳健性统计。", fontsize=10.5, color="#64748B")

    stem = f"H2_vs_H2-plus_{role}_目标事件指标"
    png_path = OUTPUT / f"{stem}.png"
    figure.savefig(png_path, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return (png_path,)


def draw_benchmark(all_metrics: dict, role: str) -> tuple[Path, ...]:
    configure_font()
    h2 = all_metrics[(role, "H2")]
    plus = all_metrics[(role, "H2-plus")]
    paper_has_values = role == "Test"
    models = (EVUAV_PAPER["model"], "H2", "H2-plus")
    colors = ("#7C8798", "#355C7D", "#D95F02")
    acc = np.asarray([EVUAV_PAPER["acc"] if paper_has_values else np.nan, h2["overall_recall"], plus["overall_recall"]]) * 100
    pd = np.asarray([EVUAV_PAPER["pd"] if paper_has_values else np.nan, h2["official_pd"], plus["official_pd"]]) * 100
    ap = np.asarray([h2["event_ap"], plus["event_ap"]]) * 100

    figure = plt.figure(figsize=(16, 9), facecolor="#F7F8FA")
    grid = figure.add_gridspec(1, 2, width_ratios=(1.65, 1.0), left=0.07, right=0.96, top=0.79, bottom=0.18, wspace=0.15)
    ax = figure.add_subplot(grid[0, 0])
    ap_ax = figure.add_subplot(grid[0, 1])
    for axis in (ax, ap_ax):
        axis.set_facecolor("white")
        axis.xaxis.grid(True, color="#D8DEE7", linewidth=0.8, zorder=0)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#94A3B8")
        axis.tick_params(axis="y", length=0)

    y = np.arange(2)
    bar_height = 0.22
    offsets = (-bar_height, 0.0, bar_height)
    for index, (model, color, offset) in enumerate(zip(models, colors, offsets)):
        values = np.asarray([acc[index], pd[index]])
        bars = ax.barh(y + offset, values, height=bar_height * 0.88, color=color, label=model, zorder=3)
        for bar in bars:
            if not np.isfinite(bar.get_width()):
                continue
            ax.text(bar.get_width() + 0.7, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.2f}%", va="center", fontsize=10.5, color="#0F172A")
    if not paper_has_values:
        ax.text(3, 0.50, "EV-SpSegNet Validation ACC/Pd：原论文未报告", va="center", fontsize=10.5, fontweight="bold", color=colors[0])
    ax.set_yticks(y, ("ACC / 目标事件召回率", "Pd / 目标窗检出率"), fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("指标值（%）", fontsize=12, labelpad=10)
    ax.set_title("ACC 与 Pd（论文 Table 2 口径）", fontsize=16, fontweight="bold", color="#172B4D", pad=35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False, fontsize=10.5)

    ap_models = ("H2", "H2-plus")
    ap_colors = colors[1:]
    ap_y = np.arange(2)
    bars = ap_ax.barh(ap_y, ap, height=0.46, color=ap_colors, zorder=3)
    ap_ax.set_yticks(ap_y, ap_models, fontsize=12)
    ap_ax.invert_yaxis()
    ap_ax.set_ylim(2.55, -0.5)
    ap_ax.set_xlim(0, 100)
    ap_ax.set_xlabel("Event AP（%）", fontsize=12, labelpad=10)
    ap_ax.set_title("不依赖背景 TN 的 Event AP", fontsize=16, fontweight="bold", color="#172B4D", pad=20)
    for bar in bars:
        ap_ax.text(bar.get_width() + 0.7, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.4f}%", va="center", fontsize=11, color="#0F172A")
    delta_ap = (plus["event_ap"] - h2["event_ap"]) * 100
    ap_ax.text(
        5,
        1.52,
        f"H2-plus 相对 H2：{delta_ap:+.4f} pp",
        fontsize=13,
        fontweight="bold",
        color="#16803C" if delta_ap > 0 else "#B42318",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ap_ax.text(
        5,
        1.92,
        f"EV-SpSegNet：原论文未报告 {role} AP/mAP",
        fontsize=12,
        fontweight="bold",
        color="#475569",
    )
    ap_ax.text(
        5,
        2.20,
        "这里统计的是事件级 PR 曲线面积（Event AP）。\n它不使用 TN，但不等同于需要边界框匹配的 COCO mAP。",
        fontsize=10.5,
        color="#64748B",
        linespacing=1.45,
    )

    figure.suptitle(f"EV-UAV 论文基线与正式模型对比（{role}）", x=0.07, y=0.95, ha="left", fontsize=23, fontweight="bold", color="#172B4D")
    subtitle = (
        "EV-SpSegNet 采用论文 Table 2 报告值；H2/H2-plus 采用冻结 exact INT8 Test 结果。"
        if paper_has_values
        else "EV-UAV 原论文未报告 Validation 指标；H2/H2-plus 采用冻结 exact INT8 Validation 结果。"
    )
    figure.text(0.07, 0.875, subtitle, fontsize=13, color="#475569")
    figure.text(
        0.07,
        0.055,
        "ACC、Pd、Event AP 均不把背景真负例 TN 计入分子。EV-SpSegNet 未报告的指标保持空缺，不使用 Test 数值代替 Validation。"
        if not paper_has_values
        else "ACC、Pd、Event AP 均不把背景真负例 TN 计入分子。EV-SpSegNet 与正式模型的实现和数值路径不同，跨论文差值用于基准定位，不替代同代码复现。",
        fontsize=10.5,
        color="#64748B",
    )
    figure.text(
        0.07,
        0.025,
        "来源：Chen et al., Event-based Tiny Object Detection: A Benchmark Dataset and Baseline, ICCV 2025。",
        fontsize=10,
        color="#64748B",
    )

    stem = f"EV-UAV_vs_H2_vs_H2-plus_{role}_无TN指标"
    png_path = OUTPUT / f"{stem}.png"
    figure.savefig(png_path, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return (png_path,)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_metrics = {
        (role, model): metrics_from_summary(load_summary(role, model))
        for role in ROLES
        for model in MODELS
    }
    rows = build_rows(all_metrics)
    csv_path = write_csv(rows)
    benchmark_csv_paths = [write_benchmark_csv(all_metrics, role) for role in ROLES]
    figure_paths = [path for role in ROLES for path in draw_role(role, all_metrics)]
    benchmark_paths = [path for role in ROLES for path in draw_benchmark(all_metrics, role)]
    print(csv_path)
    for path in benchmark_csv_paths:
        print(path)
    for path in figure_paths:
        print(path)
    for path in benchmark_paths:
        print(path)


if __name__ == "__main__":
    main()
