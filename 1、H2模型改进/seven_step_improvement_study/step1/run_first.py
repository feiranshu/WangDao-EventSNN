"""Run revised step 1: frozen-H2 diagnosis, density strata and label review.

The script is deliberately self-contained at the workflow level while reusing
the frozen inference and official-compatible metrics from ``software/wangdao``.
It never trains a model or reads Validation/Test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import zlib

import numpy as np
import torch


IMPLEMENTATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPLEMENTATION / "software"))

from wangdao.h2.constants import WINDOW_MS, WINDOWS_PER_SEQUENCE  # noqa: E402
from wangdao.h2_research import FrozenH2Pipeline, select_fair_working_point  # noqa: E402
from wangdao.h2_refiner import diagnostic_strata, error_decomposition  # noqa: E402
from wangdao.phase1_common import (  # noqa: E402
    PROTOCOL_ID,
    THRESHOLD_GRID,
    aggregate_scores,
    load_sequence,
    role_paths,
    score_predictions,
    sha256,
)


H2_F_THRESHOLD = 0.2800000011920929
FEATURE_NAMES = (
    "x", "y", "sequence_time", "polarity",
    "1px_2ms_current_same", "1px_2ms_current_other",
    "1px_2ms_lag1_same", "1px_2ms_lag1_other",
    "1px_2ms_lag2_same", "1px_2ms_lag2_other",
    "2px_5ms_current_same", "2px_5ms_current_other",
    "2px_5ms_lag1_same", "2px_5ms_lag1_other",
    "2px_5ms_lag2_same", "2px_5ms_lag2_other",
    "4px_20ms_current_same", "4px_20ms_current_other",
    "4px_20ms_lag1_same", "4px_20ms_lag1_other",
    "4px_20ms_lag2_same", "4px_20ms_lag2_other",
    "8px_50ms_current_same", "8px_50ms_current_other",
    "8px_50ms_lag1_same", "8px_50ms_lag1_other",
    "8px_50ms_lag2_same", "8px_50ms_lag2_other",
    "proposal_cell_score", "proposal_3x3_mean",
    "proposal_component_size", "within_100ms",
)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def csv_dump(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_errors(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keys = (
        "events", "target_events", "candidate_events", "proposal_false_negatives",
        "refiner_false_negatives", "tp", "fp", "fn", "tn",
    )
    result = {key: sum(int(row[key]) for row in rows) for key in keys}
    kept = result["target_events"] - result["proposal_false_negatives"]
    result.update(
        candidate_fraction=(result["candidate_events"] / result["events"] if result["events"] else None),
        proposal_recall=(kept / result["target_events"] if result["target_events"] else None),
        refiner_conditional_recall=(result["tp"] / kept if kept else None),
    )
    return result


def assigned_windows(events: np.ndarray) -> np.ndarray:
    relative = events["t"].astype(np.float64) - float(events["t"][0])
    return np.clip(np.floor(relative / WINDOW_MS), 0, WINDOWS_PER_SEQUENCE - 1).astype(np.int64)


def failure_windows(
    events: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> List[Dict[str, Any]]:
    windows = assigned_windows(events)
    retained = np.zeros(len(events), dtype=bool)
    retained[indices] = True
    target = labels.astype(bool)
    predicted = probability >= threshold
    rows = []
    for window in range(WINDOWS_PER_SEQUENCE):
        member = windows == window
        rows.append({
            "window": window,
            "start_ms_relative": float(window * WINDOW_MS),
            "events": int(member.sum()),
            "targets": int(np.sum(target & member)),
            "proposal_fn": int(np.sum(target & ~retained & member)),
            "refiner_fn": int(np.sum(target & retained & ~predicted & member)),
            "fp": int(np.sum(~target & predicted & member)),
        })
    return rows


def quantiles(values: np.ndarray) -> Dict[str, Any]:
    if not len(values):
        return {key: None for key in ("min", "p25", "median", "p75", "max", "mean")}
    array = values.astype(np.float64, copy=False)
    q = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "min": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
        "p75": float(q[3]), "max": float(q[4]), "mean": float(array.mean()),
    }


def feature_rows(
    file_name: str,
    features: np.ndarray,
    candidate_labels: np.ndarray,
    candidate_probability: np.ndarray,
) -> List[Dict[str, Any]]:
    target = candidate_labels.astype(bool)
    predicted = candidate_probability >= H2_F_THRESHOLD
    groups = {
        "target_candidate": target,
        "TP": target & predicted,
        "FN_in_gate": target & ~predicted,
        "FP": ~target & predicted,
        "TN_in_gate": ~target & ~predicted,
    }
    rows: List[Dict[str, Any]] = []
    for group, mask in groups.items():
        selected = features[mask]
        for column, name in enumerate(FEATURE_NAMES):
            stats = quantiles(selected[:, column] if len(selected) else np.empty(0))
            rows.append({
                "file": file_name,
                "group": group,
                "events": int(len(selected)),
                "feature_index": column,
                "feature": name,
                **stats,
                "uint8_255_fraction": (
                    float(np.mean(selected[:, column] == 255)) if len(selected) else None
                ),
            })
    return rows


def density_reference() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sequence_rows: List[Dict[str, Any]] = []
    window_rates: List[np.ndarray] = []
    for path in role_paths("train_main"):
        events, _, labels, _ = load_sequence(path)
        duration_s = max((float(events["t"][-1]) - float(events["t"][0])) / 1000.0, 1.0e-9)
        windows = assigned_windows(events)
        counts = np.bincount(windows, minlength=WINDOWS_PER_SEQUENCE)
        rates = counts.astype(np.float64) / (WINDOW_MS / 1000.0)
        window_rates.append(rates)
        sequence_rows.append({
            "role": "train_main",
            "file": path.name,
            "events": int(len(events)),
            "target_events": int(np.sum(labels == 1)),
            "duration_s": duration_s,
            "event_rate_eps": float(len(events) / duration_s),
            "window_rate_median_eps": float(np.median(rates)),
            "window_rate_p75_eps": float(np.quantile(rates, 0.75)),
            "window_rate_max_eps": float(np.max(rates)),
        })
    sequence_values = np.asarray([row["event_rate_eps"] for row in sequence_rows])
    all_windows = np.concatenate(window_rates)
    reference = {
        "definition": "high-density sequence iff event_rate_eps >= Train-main sequence q75",
        "sequence_event_rate_q75_eps": float(np.quantile(sequence_values, 0.75)),
        "window_event_rate_q75_eps": float(np.quantile(all_windows, 0.75)),
        "train_main_sequences": len(sequence_rows),
        "train_main_windows": int(len(all_windows)),
    }
    for row in sequence_rows:
        row["high_density"] = bool(row["event_rate_eps"] >= reference["sequence_event_rate_q75_eps"])
    return reference, sequence_rows


def event_curve(labels: np.ndarray, probabilities: np.ndarray) -> List[Dict[str, Any]]:
    thresholds = sorted(set([round(index / 100.0, 2) for index in range(1, 100)] + [H2_F_THRESHOLD]))
    target = labels.astype(bool)
    rows = []
    for threshold in thresholds:
        predicted = probabilities >= threshold
        tp = int(np.sum(target & predicted))
        fp = int(np.sum(~target & predicted))
        fn = int(np.sum(target & ~predicted))
        tn = int(np.sum(~target & ~predicted))
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        iou = tp / (tp + fp + fn) if tp + fp + fn else None
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall else None
        )
        rows.append({
            "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "iou": iou,
        })
    return rows


def exact_pr_curve(
    labels: np.ndarray, probabilities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    order = np.argsort(-probabilities, kind="stable")
    scores = probabilities[order]
    target = labels[order] == 1
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    positives = int(target.sum())
    if not positives:
        return scores[ends], np.zeros(len(ends)), np.zeros(len(ends)), None
    tp = np.cumsum(target, dtype=np.int64)[ends]
    precision = tp / (ends + 1)
    recall = tp / positives
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return scores[ends], precision, recall, ap


def decimate_curve(
    thresholds: np.ndarray, precision: np.ndarray, recall: np.ndarray, limit: int = 2500
) -> np.ndarray:
    if len(thresholds) <= limit:
        return np.arange(len(thresholds), dtype=np.int64)
    chosen = np.linspace(0, len(thresholds) - 1, limit, dtype=np.int64)
    return np.unique(np.r_[0, chosen, len(thresholds) - 1])


def svg_plot(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    x: np.ndarray,
    y: np.ndarray,
    markers: Sequence[Mapping[str, Any]],
    annotation: str,
) -> None:
    width, height = 900, 650
    left, right, top, bottom = 95, 35, 70, 85
    plot_width, plot_height = width - left - right, height - top - bottom
    def px(value: float) -> float:
        return left + min(max(float(value), 0.0), 1.0) * plot_width
    def py(value: float) -> float:
        return top + (1.0 - min(max(float(value), 0.0), 1.0)) * plot_height
    points = " ".join(f"{px(a):.2f},{py(b):.2f}" for a, b in zip(x, y))
    items = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="35" text-anchor="middle" font-size="24" font-family="Arial, sans-serif">{title}</text>',
    ]
    for tick in np.linspace(0.0, 1.0, 11):
        xx, yy = px(tick), py(tick)
        items.extend([
            f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{top+plot_height}" stroke="#e5e7eb" stroke-width="1"/>',
            f'<line x1="{left}" y1="{yy:.2f}" x2="{left+plot_width}" y2="{yy:.2f}" stroke="#e5e7eb" stroke-width="1"/>',
            f'<text x="{xx:.2f}" y="{top+plot_height+28}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif">{tick:.1f}</text>',
            f'<text x="{left-14}" y="{yy+5:.2f}" text-anchor="end" font-size="13" font-family="Arial, sans-serif">{tick:.1f}</text>',
        ])
    items.extend([
        f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="black" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="black" stroke-width="2"/>',
        f'<polyline points="{points}" fill="none" stroke="#1565c0" stroke-width="2.5" stroke-linejoin="round"/>',
    ])
    colors = ("#d32f2f", "#2e7d32", "#8e24aa", "#ef6c00")
    for index, marker in enumerate(markers):
        color = colors[index % len(colors)]
        xx, yy = px(float(marker["x"])), py(float(marker["y"]))
        label_dx = float(marker.get("label_dx", 10.0))
        label_dy = float(marker.get("label_dy", -10.0))
        items.extend([
            f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="6" fill="{color}" stroke="white" stroke-width="2"/>',
            f'<text x="{xx+label_dx:.2f}" y="{yy+label_dy:.2f}" font-size="14" fill="{color}" font-family="Arial, sans-serif">{marker["label"]}</text>',
        ])
    items.extend([
        f'<text x="{left+plot_width/2}" y="{height-25}" text-anchor="middle" font-size="18" font-family="Arial, sans-serif">{x_label}</text>',
        f'<text x="24" y="{top+plot_height/2}" text-anchor="middle" transform="rotate(-90 24 {top+plot_height/2})" font-size="18" font-family="Arial, sans-serif">{y_label}</text>',
        f'<text x="{left+12}" y="{top+24}" font-size="16" font-family="Arial, sans-serif" fill="#263238">{annotation}</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(items) + "\n")


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def save_png(path: Path, image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG input must be HxWx3 uint8")
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += png_chunk(b"IDAT", zlib.compress(raw, 9))
    payload += png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


DIGITS = {
    "0": ("111", "101", "101", "101", "111"), "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"), "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"), "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"), "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"), "9": ("111", "101", "111", "001", "111"),
}


def draw_number(canvas: np.ndarray, x: int, y: int, value: int) -> None:
    cursor = x
    for character in str(int(value)):
        for row, bits in enumerate(DIGITS[character]):
            for column, bit in enumerate(bits):
                if bit == "1":
                    canvas[y + row, cursor + column] = (0, 0, 0)
        cursor += 4


def draw_disk(canvas: np.ndarray, x: int, y: int, radius: int, color: Tuple[int, int, int]) -> None:
    height, width, _ = canvas.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                yy, xx = y + dy, x + dx
                if 0 <= yy < height and 0 <= xx < width:
                    canvas[yy, xx] = color


def render_review(
    path: Path,
    events: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    windows_to_show: Sequence[int],
) -> None:
    sensor_width, sensor_height, header = 346, 260, 14
    columns = 5
    rows = int(math.ceil(len(windows_to_show) / columns))
    canvas = np.full((rows * (sensor_height + header), columns * sensor_width, 3), 255, dtype=np.uint8)
    assigned = assigned_windows(events)
    target = labels.astype(bool)
    predicted = probabilities >= H2_F_THRESHOLD
    for panel, window in enumerate(windows_to_show):
        row, column = divmod(panel, columns)
        ox, oy = column * sensor_width, row * (sensor_height + header)
        canvas[oy + header - 1, ox:ox + sensor_width] = (0, 0, 0)
        draw_number(canvas, ox + 3, oy + 4, window)
        member = assigned == window
        background = np.flatnonzero(member & ~target)
        for index in background:
            canvas[oy + header + int(events["y"][index]), ox + int(events["x"][index])] = (214, 214, 214)
        classes = (
            (member & ~target & predicted, (0, 105, 255), 1),
            (member & target & ~predicted, (230, 0, 0), 2),
            (member & target & predicted, (0, 170, 50), 2),
        )
        for mask, color, radius in classes:
            for index in np.flatnonzero(mask):
                draw_disk(
                    canvas,
                    ox + int(events["x"][index]),
                    oy + header + int(events["y"][index]),
                    radius,
                    color,
                )
    save_png(path, canvas)


def review_records(
    file_name: str,
    events: np.ndarray,
    labels: np.ndarray,
    identities: np.ndarray,
    windows: Sequence[int],
    reasons: Mapping[int, str],
) -> List[Dict[str, Any]]:
    assigned = assigned_windows(events)
    target = labels.astype(bool)
    rows: List[Dict[str, Any]] = []
    for window in windows:
        member = assigned == window
        ids = [int(value) for value in np.unique(identities[member & target])]
        boxes = []
        for identity in ids:
            selected = member & target & (identities == identity)
            boxes.append({
                "id": identity,
                "events": int(selected.sum()),
                "x_min": int(events["x"][selected].min()),
                "x_max": int(events["x"][selected].max()),
                "y_min": int(events["y"][selected].min()),
                "y_max": int(events["y"][selected].max()),
                "cx": float(events["x"][selected].mean()),
                "cy": float(events["y"][selected].mean()),
            })
        rows.append({
            "file": file_name,
            "window": int(window),
            "selection_reason": reasons[int(window)],
            "events": int(member.sum()),
            "target_events": int(np.sum(member & target)),
            "target_ids": "|".join(str(value) for value in ids),
            "target_boxes_json": json.dumps(boxes, ensure_ascii=False, separators=(",", ":")),
            "background_identity_consistent": bool(np.all(identities[member & ~target] == 0)),
            "target_identity_consistent": bool(np.all(identities[member & target] > 0)),
            "semantic_review_status": "requires_original_annotation_or_human_review",
        })
    return rows


def select_review_windows(rows: Sequence[Mapping[str, Any]], seed: int) -> Tuple[List[int], Dict[int, str]]:
    reasons: Dict[int, str] = {}
    by_fn = sorted(rows, key=lambda row: (row["proposal_fn"] + row["refiner_fn"], row["fp"]), reverse=True)
    by_fp = sorted(rows, key=lambda row: (row["fp"], row["proposal_fn"] + row["refiner_fn"]), reverse=True)
    for row in by_fn[:5]:
        reasons[int(row["window"])] = "top_fn"
    for row in by_fp:
        if int(row["window"]) not in reasons:
            reasons[int(row["window"])] = "top_fp"
        if sum(value == "top_fp" for value in reasons.values()) >= 5:
            break
    rng = np.random.default_rng(seed)
    for window in rng.permutation(WINDOWS_PER_SEQUENCE):
        value = int(window)
        if value not in reasons:
            reasons[value] = "fixed_random"
        if sum(reason == "fixed_random" for reason in reasons.values()) >= 5:
            break
    windows = sorted(reasons, key=lambda value: (reasons[value], value))
    return windows, reasons


def markdown_report(report: Mapping[str, Any], sequence_rows: Sequence[Mapping[str, Any]]) -> str:
    fixed = report["working_points"]["H2-F"]
    fair = report["working_points"]["H2-R"]
    errors = report["errors_at_H2-F"]
    lines = [
        "# 第一步结果：H2基线复核、错误诊断与标注审查",
        "",
        "## 结论",
        "",
        f"冻结H2在完整10条Train-dev上的H2-F Event IoU为{100*fixed['event']['iou']:.4f}%，"
        f"H2-R公平工作点阈值为{fair['threshold']:.2f}、Event IoU为{100*fair['metrics']['event']['iou']:.4f}%。",
        "",
        f"H2-F共有{errors['fn']:,}个FN，其中Proposal造成{errors['proposal_false_negatives']:,}个，"
        f"Refiner造成{errors['refiner_false_negatives']:,}个。Refiner占总FN的"
        f"{100*errors['refiner_false_negatives']/errors['fn']:.2f}%，后续仍应优先改Refiner。",
        "",
        f"高密度序列阈值冻结为Train-main序列事件率75%分位数："
        f"{report['density_reference']['sequence_event_rate_q75_eps']:.2f} events/s。",
        "",
        f"高密度组由`train_089/092`组成，H2-F Event IoU为"
        f"{100*report['density_groups']['high_density']['H2-F']['event']['iou']:.4f}%，"
        f"产生{report['density_groups']['high_density']['errors_at_H2-F']['fn']:,}个FN；"
        f"普通密度组IoU为"
        f"{100*report['density_groups']['ordinary_density']['H2-F']['event']['iou']:.4f}%，"
        f"产生{report['density_groups']['ordinary_density']['errors_at_H2-F']['fn']:,}个FN。",
        "",
        "NPZ字段、坐标、时间、标签和身份关系通过内部一致性检查。审查包已生成重点窗口图；"
        "没有原始图像或原始框标注时，不能据此证明语义标注完全正确。",
        "",
        "## 逐序列摘要",
        "",
        "| 序列 | 高密度 | 事件数 | 目标数 | Proposal召回 | Refiner条件召回 | H2-F IoU | H2-R IoU | H2-F FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sequence_rows:
        lines.append(
            f"| {row['file']} | {'是' if row['high_density'] else '否'} | {row['events']:,} | "
            f"{row['target_events']:,} | {100*row['proposal_recall']:.2f}% | "
            f"{100*row['refiner_conditional_recall']:.2f}% | {100*row['h2_f_iou']:.2f}% | "
            f"{100*row['h2_r_iou']:.2f}% | {row['h2_f_fn']:,} |"
        )
    lines.extend([
        "",
        "## 文件说明",
        "",
        "- `step1_report.json`：完整统计、协议和哈希。",
        "- `../../../../指标/原始H2模型/`：Event PR/AP、F1、工作点、逐序列和密度分组指标。",
        "- `feature_diagnostics.csv`：各序列TP/FN/FP/TN的32维输入分布。",
        "- `label_review/`：`089/092`的15个固定审查窗口和可视化。",
        "",
        "## 下一步依据",
        "",
        "A1/A2正式训练应继续冻结Proposal。后续每个候选必须同时报告整体、密度分组、"
        "`train_089/092`和逐序列结果，避免总体指标掩盖高密度场景。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument(
        "--metrics-dir", type=Path,
        default=Path(__file__).resolve().parents[3] / "指标" / "原始H2模型",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    metrics_output = args.metrics_dir.resolve()
    if output.exists() or metrics_output.exists():
        parser.error(f"output or metrics directory already exists: {output}, {metrics_output}")
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch size and threads must be positive")
    output.mkdir(parents=True)
    (output / "label_review").mkdir()
    metrics_output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()

    weight_path = IMPLEMENTATION / "software/weights/H2_D1_seed42.npz"
    index_path = IMPLEMENTATION / "data/index.json"
    pipeline = FrozenH2Pipeline(weight_path, args.batch_size)
    density, density_rows = density_reference()

    threshold_scores: Dict[str, List[Dict[str, Any]]] = {
        str(value): [] for value in THRESHOLD_GRID
    }
    ungated_scores: List[Dict[str, Any]] = []
    sequence_state: List[Dict[str, Any]] = []
    feature_diagnostics: List[Dict[str, Any]] = []
    all_labels: List[np.ndarray] = []
    all_probability: List[np.ndarray] = []

    for path in role_paths("train_dev"):
        events, loc, labels, identities = load_sequence(path)
        inputs = events[["x", "y", "t", "p"]].copy()
        prepared = pipeline.prepare(inputs, include_all=True)
        inference = pipeline.infer_prepared(prepared, len(events))
        probability = inference["probabilities"]
        ungated = pipeline.ungated_probability(prepared, len(events))
        for threshold in THRESHOLD_GRID:
            threshold_scores[str(threshold)].append(
                score_predictions(loc, labels, identities, probability, threshold)
            )
        ungated_scores.append(score_predictions(loc, labels, identities, ungated, H2_F_THRESHOLD))
        errors = error_decomposition(labels, prepared.indices, probability)
        strata_masks = diagnostic_strata(events, labels, identities)
        strata = {
            name: error_decomposition(labels, prepared.indices, probability, mask=mask)
            for name, mask in strata_masks.items()
        }
        windows = failure_windows(events, labels, prepared.indices, probability, H2_F_THRESHOLD)
        feature_diagnostics.extend(
            feature_rows(
                path.name,
                prepared.features,
                labels[prepared.indices],
                inference["candidate_probabilities"],
            )
        )
        duration_s = max((float(events["t"][-1]) - float(events["t"][0])) / 1000.0, 1.0e-9)
        event_rate = float(len(events) / duration_s)
        sequence_state.append({
            "path": path,
            "events_array": events,
            "loc": loc,
            "labels": labels,
            "identities": identities,
            "probability": probability,
            "errors": errors,
            "strata": strata,
            "failure_windows": windows,
            "event_rate_eps": event_rate,
            "high_density": bool(event_rate >= density["sequence_event_rate_q75_eps"]),
            "candidate_indices": prepared.indices,
            "timings": inference["timings"],
        })
        all_labels.append(labels.astype(np.uint8))
        all_probability.append(probability.astype(np.float32))
        print(path.name, json.dumps(errors, ensure_ascii=False), flush=True)

    aggregate_curve = {
        key: aggregate_scores(value) for key, value in threshold_scores.items()
    }
    fixed = aggregate_curve[str(H2_F_THRESHOLD)]
    fair = select_fair_working_point(
        aggregate_curve, fixed["localization"]["pd"], fixed["localization"]["fa"]
    )
    if not fair["eligible"]:
        raise RuntimeError("H2-R fair working point is unavailable")
    h2_r_threshold = float(fair["threshold"])

    labels_joined = np.concatenate(all_labels)
    probability_joined = np.concatenate(all_probability)
    curve_rows = event_curve(labels_joined, probability_joined)
    official_lookup = {round(float(key), 8): value for key, value in aggregate_curve.items()}
    for row in curve_rows:
        official = official_lookup.get(round(float(row["threshold"]), 8))
        row["pd"] = official["localization"]["pd"] if official else None
        row["fa"] = official["localization"]["fa"] if official else None
        row["fa_times_1e4"] = official["localization"]["fa_times_1e4"] if official else None
    exact_thresholds, exact_precision, exact_recall, ap = exact_pr_curve(
        labels_joined, probability_joined
    )
    exact_points = int(len(exact_thresholds))

    sequence_rows: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []
    per_sequence_report: List[Dict[str, Any]] = []
    for position, state in enumerate(sequence_state):
        h2_f = score_predictions(
            state["loc"], state["labels"], state["identities"],
            state["probability"], H2_F_THRESHOLD,
        )
        h2_r = score_predictions(
            state["loc"], state["labels"], state["identities"],
            state["probability"], h2_r_threshold,
        )
        errors = state["errors"]
        row = {
            "file": state["path"].name,
            "events": errors["events"],
            "target_events": errors["target_events"],
            "target_fraction": errors["target_events"] / errors["events"],
            "event_rate_eps": state["event_rate_eps"],
            "high_density": state["high_density"],
            "candidate_events": errors["candidate_events"],
            "candidate_fraction": errors["candidate_fraction"],
            "proposal_recall": errors["proposal_recall"],
            "refiner_conditional_recall": errors["refiner_conditional_recall"],
            "proposal_fn": errors["proposal_false_negatives"],
            "refiner_fn": errors["refiner_false_negatives"],
            "h2_f_threshold": H2_F_THRESHOLD,
            "h2_f_tp": h2_f["event"]["tp"],
            "h2_f_fp": h2_f["event"]["fp"],
            "h2_f_fn": h2_f["event"]["fn"],
            "h2_f_iou": h2_f["event"]["iou"],
            "h2_f_precision": h2_f["event"]["precision"],
            "h2_f_recall": h2_f["event"]["recall"],
            "h2_f_pd": h2_f["localization"]["pd"],
            "h2_f_fa_times_1e4": h2_f["localization"]["fa_times_1e4"],
            "h2_r_threshold": h2_r_threshold,
            "h2_r_tp": h2_r["event"]["tp"],
            "h2_r_fp": h2_r["event"]["fp"],
            "h2_r_fn": h2_r["event"]["fn"],
            "h2_r_iou": h2_r["event"]["iou"],
            "h2_r_precision": h2_r["event"]["precision"],
            "h2_r_recall": h2_r["event"]["recall"],
            "h2_r_pd": h2_r["localization"]["pd"],
            "h2_r_fa_times_1e4": h2_r["localization"]["fa_times_1e4"],
        }
        sequence_rows.append(row)
        per_sequence_report.append({
            **{key: value for key, value in row.items() if key not in ("h2_f_threshold", "h2_r_threshold")},
            "H2-F": h2_f,
            "H2-R": h2_r,
            "errors_at_H2-F": errors,
            "diagnostic_strata_at_H2-F": state["strata"],
            "failure_windows_at_H2-F": state["failure_windows"],
            "inference_timings_s": state["timings"],
        })
        if state["path"].name in ("train_089.npz", "train_092.npz"):
            windows, reasons = select_review_windows(state["failure_windows"], 20261002 + position)
            review_rows.extend(
                review_records(
                    state["path"].name, state["events_array"], state["labels"],
                    state["identities"], windows, reasons,
                )
            )
            render_review(
                output / "label_review" / f"{state['path'].stem}_review.png",
                state["events_array"], state["labels"], state["probability"], windows,
            )

    density_output = list(density_rows)
    for row in sequence_rows:
        density_output.append({
            "role": "train_dev", "file": row["file"], "events": row["events"],
            "target_events": row["target_events"], "duration_s": row["events"] / row["event_rate_eps"],
            "event_rate_eps": row["event_rate_eps"], "window_rate_median_eps": None,
            "window_rate_p75_eps": None, "window_rate_max_eps": None,
            "high_density": row["high_density"],
        })

    grouped: Dict[str, Any] = {}
    for name, desired in (("high_density", True), ("ordinary_density", False)):
        members = [index for index, row in enumerate(sequence_rows) if bool(row["high_density"]) == desired]
        grouped[name] = {
            "sequence_count": len(members),
            "sequences": [sequence_rows[index]["file"] for index in members],
            "H2-F": aggregate_scores([
                threshold_scores[str(H2_F_THRESHOLD)][index] for index in members
            ]) if members else None,
            "H2-R": aggregate_scores([
                score_predictions(
                    sequence_state[index]["loc"], sequence_state[index]["labels"],
                    sequence_state[index]["identities"], sequence_state[index]["probability"],
                    h2_r_threshold,
                ) for index in members
            ]) if members else None,
            "errors_at_H2-F": aggregate_errors([
                sequence_state[index]["errors"] for index in members
            ]) if members else None,
        }

    errors = aggregate_errors([state["errors"] for state in sequence_state])
    report = {
        "schema_version": 1,
        "status": "complete",
        "stage": "revised_step1_h2_diagnosis",
        "conclusion": "Frozen-H2 baseline, density-stratified error diagnosis and NPZ-internal label review complete.",
        "scope": "Train-main density reference plus Train-dev diagnosis; no training, Validation or Test",
        "test_used": False,
        "validation_used": False,
        "metric_protocol": PROTOCOL_ID,
        "data_index": str(index_path.relative_to(IMPLEMENTATION)).replace("\\", "/"),
        "data_index_sha256": sha256(index_path),
        "h2_weights": str(weight_path.relative_to(IMPLEMENTATION)).replace("\\", "/"),
        "h2_weights_sha256": sha256(weight_path),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "sequence_count": len(sequence_rows),
        "event_count": int(len(labels_joined)),
        "target_events": int(np.sum(labels_joined == 1)),
        "density_reference": density,
        "working_points": {
            "H2-F": fixed,
            "H2-R": fair,
        },
        "event_average_precision": ap,
        "event_pr_exact_point_count": exact_points,
        "event_ap_definition": "event average precision; not bounding-box mAP",
        "errors_at_H2-F": errors,
        "ungated_all_event_diagnostic_at_H2-F": aggregate_scores(ungated_scores),
        "density_groups": grouped,
        "per_sequence": per_sequence_report,
        "label_review": {
            "files": ["train_089.npz", "train_092.npz"],
            "windows_per_file": 15,
            "selection": "5 top FN, 5 non-overlapping top FP, 5 fixed-seed random windows",
            "automatic_scope": "NPZ internal consistency, track/event geometry and H2 error overlays",
            "semantic_status": "requires original annotations or explicit human judgment",
            "colors": {"background": "gray", "FN": "red", "TP": "green", "FP": "blue"},
        },
        "elapsed_s": time.perf_counter() - started,
    }

    json_dump(output / "step1_report.json", report)
    csv_dump(output / "feature_diagnostics.csv", feature_diagnostics, list(feature_diagnostics[0]))
    csv_dump(output / "label_review/review_manifest.csv", review_rows, list(review_rows[0]))
    with (output / "REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(markdown_report(report, sequence_rows))
    csv_dump(metrics_output / "per_sequence_metrics.csv", sequence_rows, list(sequence_rows[0]))
    csv_dump(metrics_output / "threshold_metrics.csv", curve_rows, list(curve_rows[0]))
    csv_dump(metrics_output / "density_metrics.csv", density_output, list(density_output[0]))
    operating_rows = []
    confusion_rows = []
    for name, threshold, metrics in (
        ("H2-F", H2_F_THRESHOLD, fixed),
        ("H2-R", h2_r_threshold, fair["metrics"]),
    ):
        operating_rows.append({
            "model": name, "threshold": threshold,
            **metrics["event"],
            "pd": metrics["localization"]["pd"],
            "fa": metrics["localization"]["fa"],
            "fa_times_1e4": metrics["localization"]["fa_times_1e4"],
        })
        confusion_rows.append({
            "model": name, "threshold": threshold,
            "tp": metrics["event"]["tp"], "fp": metrics["event"]["fp"],
            "fn": metrics["event"]["fn"], "tn": metrics["event"]["tn"],
        })
    csv_dump(metrics_output / "operating_points.csv", operating_rows, list(operating_rows[0]))
    csv_dump(metrics_output / "confusion_matrix.csv", confusion_rows, list(confusion_rows[0]))
    np.savez_compressed(
        metrics_output / "pr_curve_exact.npz",
        thresholds=exact_thresholds.astype(np.float32),
        precision=exact_precision.astype(np.float64),
        recall=exact_recall.astype(np.float64),
        average_precision=np.asarray(ap, dtype=np.float64),
        labels=np.asarray("event PR; complete H2 pipeline; rejected events score zero"),
    )
    selected = decimate_curve(exact_thresholds, exact_precision, exact_recall)
    plot_rows = [
        {"threshold": float(exact_thresholds[index]),
         "precision": float(exact_precision[index]),
         "recall": float(exact_recall[index])}
        for index in selected
    ]
    csv_dump(metrics_output / "pr_curve_plot.csv", plot_rows, list(plot_rows[0]))
    svg_plot(
        metrics_output / "H2_PR_curve.svg", "H2 Original Model: Event PR Curve",
        "Recall", "Precision", exact_recall[selected], exact_precision[selected],
        [
            {"x": fixed["event"]["recall"], "y": fixed["event"]["precision"], "label": "H2-F (0.28)", "label_dy": -12},
            {"x": fair["metrics"]["event"]["recall"], "y": fair["metrics"]["event"]["precision"], "label": f"H2-R ({h2_r_threshold:.2f})", "label_dy": 22},
        ],
        f"Event AP = {ap:.4f}; exact points = {exact_points:,}",
    )
    f1_x = np.asarray([row["threshold"] for row in curve_rows], dtype=np.float64)
    f1_y = np.asarray([row["f1"] for row in curve_rows], dtype=np.float64)
    svg_plot(
        metrics_output / "H2_F1_threshold.svg", "H2 Original Model: F1 vs Threshold",
        "Event threshold", "F1", f1_x, f1_y,
        [
            {"x": H2_F_THRESHOLD, "y": fixed["event"]["f1"], "label": "H2-F", "label_dy": -12},
            {"x": h2_r_threshold, "y": fair["metrics"]["event"]["f1"], "label": "H2-R", "label_dy": 22},
        ],
        "Full-event classification; thresholds 0.01-0.99 plus 0.28",
    )
    metric_summary = {
        "schema_version": 1,
        "model": "H2 original frozen ANN",
        "data_role": "train_dev",
        "sequence_count": len(sequence_rows),
        "events": int(len(labels_joined)),
        "target_events": int(np.sum(labels_joined == 1)),
        "event_average_precision": ap,
        "exact_pr_points": exact_points,
        "working_points": report["working_points"],
        "density_reference": density,
        "density_groups": grouped,
        "metric_protocol": PROTOCOL_ID,
        "data_index_sha256": report["data_index_sha256"],
        "h2_weights_sha256": report["h2_weights_sha256"],
        "validation_used": False,
        "test_used": False,
    }
    json_dump(metrics_output / "summary.json", metric_summary)
    with (metrics_output / "README.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "# H2原始模型指标\n\n"
            f"完整10条Train-dev，Event AP={ap:.6f}，精确PR点数={exact_points:,}。\n\n"
            "`H2_PR_curve.svg`是PR曲线；`pr_curve_exact.npz`保存全部精确点；"
            "`pr_curve_plot.csv`是用于绘图的等距抽样点。H2-F与H2-R工作点见"
            "`operating_points.csv`。PR/AP衡量事件排序能力，最终模型仍需同时满足IoU、Pd和Fa约束。\n"
        )
    print("Saved:", output, flush=True)


if __name__ == "__main__":
    main()

