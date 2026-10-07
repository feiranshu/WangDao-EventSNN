"""从冻结的正式结果生成 A 类统一汇总表和 Validation/Test 图表。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
ACCURACY_ROOT = PROJECT_ROOT / "5、模型评测指标" / "A、模型准确度"
DATA_ROOT = ACCURACY_ROOT / "02、评测数据"
CHART_ROOT = ACCURACY_ROOT / "01、对比图表"
PAPER_REFERENCE = DATA_ROOT / "EV-SpSegNet" / "论文参考.json"
ROLES = ("Validation", "Test")
MODELS = ("EV-SpSegNet", "H2", "H2-plus")
FORMAL_MODELS = ("H2", "H2-plus")
COLORS = {"EV-SpSegNet": "#8A94A3", "H2": "#355C7D", "H2-plus": "#D95F02"}


def configure_font() -> None:
    preferred = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans")
    installed = {item.name for item in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = [next((name for name in preferred if name in installed), "DejaVu Sans")]
    plt.rcParams["axes.unicode_minus"] = False


def event_dir(model: str, role: str) -> Path:
    return DATA_ROOT / model / role / "事件级"


def track_dir(model: str, role: str) -> Path:
    return DATA_ROOT / model / role / "目标轨迹级"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def event_metrics(model: str, role: str) -> dict:
    data = read_json(event_dir(model, role) / "summary.json")
    working = data["working_point"]
    event = working["overall"]["event"]
    localization = working["overall"]["localization"]
    recalls = [float(row["event"]["recall"]) for row in working["per_sequence"].values()]
    return {
        "Event AP": float(working["average_precision"]),
        "Event IoU": float(event["iou"]),
        "Precision": float(event["precision"]),
        "ACC / Target-event Recall": float(event["recall"]),
        "F1": float(event["f1"]),
        "Pd": float(localization["pd"]),
        "Fa": float(localization["fa"]),
        "Macro sequence recall": float(np.mean(recalls)),
        "High-density recall": float(working["high_density"]["event"]["recall"]),
        "Ordinary-density recall": float(working["ordinary_density"]["event"]["recall"]),
        "Detected target events": int(event["tp"]),
        "Missed target events": int(event["fn"]),
        "Target events": int(event["tp"] + event["fn"]),
        "threshold": float(working["threshold"]),
        "protocol": data["evaluation"]["metric_protocol"],
    }


def track_metrics(model: str, role: str) -> dict:
    data = read_json(track_dir(model, role) / "summary.json")
    return {
        **{key: float(value) for key, value in data["metrics"]["averaged"].items()},
        "LongestMissP95ms": float(data["longest_continuous_miss_p95_ms"]),
        "protocol": data["metric_protocol"],
    }


def add_row(rows: list[dict], role: str, model: str, category: str, metric: str,
            value: float | int | None, unit: str, direction: str, status: str,
            numeric_path: str, protocol: str, source: str) -> None:
    rows.append({
        "data_role": role,
        "model": model,
        "category": category,
        "metric": metric,
        "value": "" if value is None else value,
        "unit": unit,
        "preferred_direction": direction,
        "status": status,
        "numeric_path": numeric_path,
        "metric_protocol": protocol,
        "source": source,
    })


def build_summary_rows(events: dict, tracks: dict, paper: dict) -> list[dict]:
    rows: list[dict] = []
    event_defs = (
        ("Event AP", "rate", "higher"), ("Event IoU", "rate", "higher"),
        ("Precision", "rate", "higher"), ("ACC / Target-event Recall", "rate", "higher"),
        ("F1", "rate", "higher"), ("Pd", "rate", "higher"), ("Fa", "rate", "lower"),
        ("Macro sequence recall", "rate", "higher"),
        ("High-density recall", "rate", "higher"),
        ("Ordinary-density recall", "rate", "higher"),
        ("Detected target events", "count", "higher"),
        ("Missed target events", "count", "lower"), ("Target events", "count", "context"),
    )
    track_defs = tuple(
        (name, "ms" if name == "LongestMissP95ms" else "rate", "lower" if name == "LongestMissP95ms" else "higher")
        for name in ("HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "AssRe", "AssPr", "LongestMissP95ms")
    )
    for role in ROLES:
        for metric in ("ACC / Target-event Recall", "Pd", "Event AP"):
            key = {"ACC / Target-event Recall": "acc", "Pd": "pd", "Event AP": "event_ap"}[metric]
            value = paper["paper_reported"][key] if role == "Test" else None
            add_row(rows, role, "EV-SpSegNet", "事件级", metric, value, "rate", "higher",
                    "paper_reported" if value is not None else "not_reported",
                    "paper_reported" if value is not None else "unavailable",
                    "ICCV 2025 paper", "EV-UAV paper Table 2")
        for model in FORMAL_MODELS:
            source = str(event_dir(model, role) / "summary.json")
            for metric, unit, direction in event_defs:
                add_row(rows, role, model, "事件级", metric, events[(model, role)][metric], unit,
                        direction, "complete", "exact_int8_replay", events[(model, role)]["protocol"], source)
        for metric, unit, direction in track_defs:
            add_row(rows, role, "EV-SpSegNet", "目标轨迹级", metric, None, unit, direction,
                    "pending_cuda_evaluation", "official_float_cuda_import_pending",
                    "evuav-observable-event-track-hota-v1", str(DATA_ROOT / "EV-SpSegNet" / "实测状态.json"))
            for model in FORMAL_MODELS:
                add_row(rows, role, model, "目标轨迹级", metric, tracks[(model, role)][metric], unit,
                        direction, "complete", "exact_int8_replay", tracks[(model, role)]["protocol"],
                        str(track_dir(model, role) / "summary.json"))
    return rows


def write_summary_csv(rows: list[dict]) -> Path:
    path = CHART_ROOT / "指标汇总.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def style_axis(ax, title: str, ylabel: str = "指标值（%）") -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def grouped_chart(role: str, metrics: tuple[str, ...], values: dict, filename: str,
                  title: str, models: tuple[str, ...] = FORMAL_MODELS) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    x = np.arange(len(metrics))
    width = 0.72 / len(models)
    for index, model in enumerate(models):
        model_values = [values[(model, role)].get(metric, np.nan) * 100 for metric in metrics]
        bars = ax.bar(x + (index - (len(models) - 1) / 2) * width, model_values, width,
                      label=model, color=COLORS[model])
        for bar, value in zip(bars, model_values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.2f}",
                        ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 108)
    style_axis(ax, title)
    ax.legend()
    fig.tight_layout()
    path = CHART_ROOT / role / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_compatibility(role: str, events: dict, paper: dict) -> Path:
    metrics = ("ACC / Target-event Recall", "Pd", "Event AP")
    values = {
        ("EV-SpSegNet", role): {
            metric: (paper["paper_reported"][{"ACC / Target-event Recall": "acc", "Pd": "pd", "Event AP": "event_ap"}[metric]]
                     if role == "Test" else None) for metric in metrics
        },
        **{(model, role): events[(model, role)] for model in FORMAL_MODELS},
    }
    fig, ax = plt.subplots(figsize=(12, 6.7))
    x = np.arange(len(metrics))
    width = 0.22
    for index, model in enumerate(MODELS):
        raw = [values[(model, role)].get(metric) for metric in metrics]
        plotted = [np.nan if item is None else item * 100 for item in raw]
        bars = ax.bar(x + (index - 1) * width, plotted, width, label=model, color=COLORS[model])
        for metric_index, (bar, value) in enumerate(zip(bars, plotted)):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.2f}", ha="center", fontsize=8, rotation=90)
            else:
                ax.text(x[metric_index] + (index - 1) * width, 2, "N/A", ha="center", fontsize=8, color="#666")
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 108)
    style_axis(ax, f"{role}：EV-UAV 兼容指标与无 TN 指标")
    ax.legend()
    fig.tight_layout()
    path = CHART_ROOT / role / "02_EV-UAV兼容指标.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_track_diagnostic(role: str, tracks: dict) -> Path:
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": (3, 1)})
    metrics = ("DetRe", "DetPr", "AssRe", "AssPr")
    x = np.arange(len(metrics))
    width = 0.32
    for index, model in enumerate(FORMAL_MODELS):
        vals = [tracks[(model, role)][metric] * 100 for metric in metrics]
        left.bar(x + (index - 0.5) * width, vals, width, label=model, color=COLORS[model])
    left.set_xticks(x, metrics)
    left.set_ylim(0, 105)
    style_axis(left, "检测与身份关联分项")
    left.legend()
    misses = [tracks[(model, role)]["LongestMissP95ms"] for model in FORMAL_MODELS]
    bars = right.bar(FORMAL_MODELS, misses, color=[COLORS[m] for m in FORMAL_MODELS])
    for bar, value in zip(bars, misses):
        right.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom")
    style_axis(right, "最长连续漏检 P95", "ms（越低越好）")
    fig.suptitle(f"{role}：目标轨迹诊断指标", fontweight="bold")
    fig.tight_layout()
    path = CHART_ROOT / role / "04_目标轨迹诊断指标.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def read_pr(model: str, role: str) -> tuple[np.ndarray, np.ndarray]:
    with (event_dir(model, role) / "pr_curve.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return np.asarray([float(row["recall"]) for row in rows]), np.asarray([float(row["precision"]) for row in rows])


def draw_pr(role: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7))
    for model in FORMAL_MODELS:
        recall, precision = read_pr(model, role)
        ax.plot(recall, precision, label=model, color=COLORS[model], linewidth=2)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{role}：事件级 PR 曲线", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = CHART_ROOT / role / "05_PR曲线.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    configure_font()
    for role in ROLES:
        (CHART_ROOT / role).mkdir(parents=True, exist_ok=True)
    events = {(model, role): event_metrics(model, role) for model in FORMAL_MODELS for role in ROLES}
    tracks = {(model, role): track_metrics(model, role) for model in FORMAL_MODELS for role in ROLES}
    paper = read_json(PAPER_REFERENCE)
    generated = [write_summary_csv(build_summary_rows(events, tracks, paper))]
    for role in ROLES:
        generated.append(grouped_chart(role, ("Event AP", "Event IoU", "Precision", "F1"), events,
                                       "01_事件级检测指标.png", f"{role}：事件级检测指标"))
        generated.append(draw_compatibility(role, events, paper))
        generated.append(grouped_chart(role, ("HOTA", "DetA", "AssA", "LocA"), tracks,
                                       "03_目标轨迹主指标.png", f"{role}：可观测事件轨迹主指标"))
        generated.append(draw_track_diagnostic(role, tracks))
        generated.append(draw_pr(role))
        generated.append(grouped_chart(role, ("ACC / Target-event Recall", "Macro sequence recall",
                                               "High-density recall", "Ordinary-density recall"), events,
                                       "06_分层检测与漏检分析.png", f"{role}：目标事件分层召回"))
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
