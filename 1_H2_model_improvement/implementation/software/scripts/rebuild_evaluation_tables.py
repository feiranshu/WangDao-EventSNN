"""Build registry-ordered per-metric evaluation CSVs and validate coverage."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CSV_FIELDS = (
    "evaluation_id", "model_name", "evaluation_track", "model_role", "platform_id",
    "value", "unit", "status", "evidence_type", "protocol_id", "source_file", "notes",
)


@dataclass(frozen=True)
class Metric:
    code: str
    slug: str
    source_key: str
    unit: str
    level: str

    @property
    def directory(self) -> str:
        return f"{self.code}_{self.slug}"


A_METRICS = (
    Metric("A01", "event_ap", "Event AP", "rate", "01_event_level"),
    Metric("A02", "event_iou", "Event IoU", "rate", "01_event_level"),
    Metric("A03", "precision", "Precision", "rate", "01_event_level"),
    Metric("A04", "target_event_recall_acc", "ACC / Target-event Recall", "rate", "01_event_level"),
    Metric("A05", "f1", "F1", "rate", "01_event_level"),
    Metric("A06", "pd", "Pd", "rate", "01_event_level"),
    Metric("A07", "fa", "Fa", "rate", "01_event_level"),
    Metric("A08", "macro_sequence_recall", "Macro sequence recall", "rate", "01_event_level"),
    Metric("A09", "high_density_recall", "High-density recall", "rate", "01_event_level"),
    Metric("A10", "ordinary_density_recall", "Ordinary-density recall", "rate", "01_event_level"),
    Metric("A11", "detected_target_events", "Detected target events", "count", "01_event_level"),
    Metric("A12", "missed_target_events", "Missed target events", "count", "01_event_level"),
    Metric("A13", "target_events", "Target events", "count", "01_event_level"),
    Metric("A14", "hota", "HOTA", "rate", "02_target_track_level"),
    Metric("A15", "deta", "DetA", "rate", "02_target_track_level"),
    Metric("A16", "detre", "DetRe", "rate", "02_target_track_level"),
    Metric("A17", "detpr", "DetPr", "rate", "02_target_track_level"),
    Metric("A18", "assa", "AssA", "rate", "02_target_track_level"),
    Metric("A19", "assre", "AssRe", "rate", "02_target_track_level"),
    Metric("A20", "asspr", "AssPr", "rate", "02_target_track_level"),
    Metric("A21", "loca", "LocA", "rate", "02_target_track_level"),
    Metric("A22", "longest_miss_p95_ms", "LongestMissP95ms", "ms", "02_target_track_level"),
)

B_METRICS = (
    Metric("B01", "parameter_count", "total_weight_parameters", "count", ""),
    Metric("B02", "static_storage", "total_tensor_payload_bytes", "byte", ""),
    Metric("B03", "dense_mac", "dense_macs_per_sequence", "MAC/sequence", ""),
    Metric("B04a", "effective_mac", "effective_macs_per_sequence", "MAC/sequence", "B04_effective_operations"),
    Metric("B04b", "effective_ac", "effective_acs_per_sequence", "AC/sequence", "B04_effective_operations"),
    Metric("B04c", "mac_plus_ac", "effective_ops_per_sequence", "op/sequence", "B04_effective_operations"),
)

C_COMMON = ("C04a", "C06a", "C06b", "C06c", "C06d", "C06e")
C_SPLIT = (
    "C01a", "C01b", "C01c", "C01d", "C02a", "C02b", "C02c", "C02d",
    "C03a", "C03b", "C03c", "C03d", "C04b", "C04c", "C05a", "C05b", "C07",
)

RUNTIME_PROTOCOL = "full-sequence-runtime-v1"

C_METRICS = (
    ("C01a", "C01_kernel_latency/C01a_mean", "kernel_latency_mean_ms", "ms", "split"),
    ("C01b", "C01_kernel_latency/C01b_p50", "kernel_latency_p50_ms", "ms", "split"),
    ("C01c", "C01_kernel_latency/C01c_p95", "kernel_latency_p95_ms", "ms", "split"),
    ("C01d", "C01_kernel_latency/C01d_p99", "kernel_latency_p99_ms", "ms", "split"),
    ("C02a", "C02_end_to_end_latency/C02a_mean", "end_to_end_latency_mean_ms", "ms", "split"),
    ("C02b", "C02_end_to_end_latency/C02b_p50", "end_to_end_latency_p50_ms", "ms", "split"),
    ("C02c", "C02_end_to_end_latency/C02c_p95", "end_to_end_latency_p95_ms", "ms", "split"),
    ("C02d", "C02_end_to_end_latency/C02d_p99", "end_to_end_latency_p99_ms", "ms", "split"),
    ("C03a", "C03_throughput_and_realtime/C03a_sequence_throughput", "sequence_throughput", "sequence/s", "split"),
    ("C03b", "C03_throughput_and_realtime/C03b_event_throughput", "event_throughput", "event/s", "split"),
    ("C03c", "C03_throughput_and_realtime/C03c_realtime_factor", "realtime_factor", "ratio", "split"),
    ("C03d", "C03_throughput_and_realtime/C03d_timeout_rate", "timeout_rate", "rate", "split"),
    ("C04a", "C04_power/C04a_idle_power", "idle_power_w", "W", "common"),
    ("C04b", "C04_power/C04b_run_power", "run_power_w", "W", "split"),
    ("C04c", "C04_power/C04c_dynamic_power", "dynamic_power_w", "W", "split"),
    ("C05a", "C05_energy/C05a_total_energy_per_sequence", "total_energy_per_sequence_j", "J/sequence", "split"),
    ("C05b", "C05_energy/C05b_dynamic_energy_per_sequence", "dynamic_energy_per_sequence_j", "J/sequence", "split"),
    ("C06a", "C06_hardware_resources/C06a_lut", "lut", "count", "common"),
    ("C06b", "C06_hardware_resources/C06b_ff", "ff", "count", "common"),
    ("C06c", "C06_hardware_resources/C06c_bram36k", "bram36k", "count", "common"),
    ("C06d", "C06_hardware_resources/C06d_dsp", "dsp", "count", "common"),
    ("C06e", "C06_hardware_resources/C06e_clock_frequency", "clock_frequency_mhz", "MHz", "common"),
    ("C07", "C07_runtime", "runtime_ms_per_sequence", "ms/sequence", "split"),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_registry(root: Path) -> list[dict[str, str]]:
    with (root / "model_registry.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"evaluation_id", "model_name", "evaluation_track", "model_role", "platform_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("model_registry.csv is empty or missing required columns")
    return sorted(rows, key=lambda row: int(row.get("display_order", "0") or 0))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: tuple[str, ...] = CSV_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_row(model: dict[str, str], value: Any, unit: str, status: str,
             evidence_type: str, protocol_id: str, source_file: Path | str,
             notes: str = "") -> dict[str, Any]:
    return {
        "evaluation_id": model["evaluation_id"], "model_name": model["model_name"],
        "evaluation_track": model["evaluation_track"], "model_role": model["model_role"],
        "platform_id": model["platform_id"], "value": "" if value is None else value,
        "unit": unit, "status": status, "evidence_type": evidence_type,
        "protocol_id": protocol_id, "source_file": str(source_file).replace("\\", "/"),
        "notes": notes,
    }


def event_metrics(path: Path) -> tuple[dict[str, Any], str]:
    data = read_json(path)
    working = data["working_point"]
    event = working["overall"]["event"]
    localization = working["overall"]["localization"]
    recalls = [float(row["event"]["recall"]) for row in working["per_sequence"].values()]
    values = {
        "Event AP": float(working["average_precision"]), "Event IoU": float(event["iou"]),
        "Precision": float(event["precision"]), "ACC / Target-event Recall": float(event["recall"]),
        "F1": float(event["f1"]), "Pd": float(localization["pd"]), "Fa": float(localization["fa"]),
        "Macro sequence recall": sum(recalls) / len(recalls),
        "High-density recall": float(working["high_density"]["event"]["recall"]),
        "Ordinary-density recall": float(working["ordinary_density"]["event"]["recall"]),
        "Detected target events": int(event["tp"]), "Missed target events": int(event["fn"]),
        "Target events": int(event["tp"] + event["fn"]),
    }
    return values, str(data["evaluation"]["metric_protocol"])


def track_metrics(path: Path) -> tuple[dict[str, Any], str]:
    data = read_json(path)
    values = {key: float(value) for key, value in data["metrics"]["averaged"].items()}
    values["LongestMissP95ms"] = float(data["longest_continuous_miss_p95_ms"])
    return values, str(data["metric_protocol"])


def record_root(category_root: Path, model: dict[str, str]) -> Path:
    branch = "external_references" if model["model_role"] == "external_reference" else "formal"
    return category_root / "02_model_records" / branch / model["evaluation_id"]


def paper_accuracy(model_root: Path, split: str) -> tuple[dict[str, Any], dict[str, str]]:
    paper = read_json(model_root / "paper_reference.json")
    values: dict[str, Any] = {}
    statuses: dict[str, str] = {}
    if split == "test":
        mapping = {
            "Event AP": "event_ap", "Event IoU": "event_iou", "ACC / Target-event Recall": "acc",
            "Pd": "pd", "Fa": "fa",
        }
        for metric, key in mapping.items():
            value = paper.get("paper_reported", {}).get(key)
            values[metric] = value
            statuses[metric] = "paper_reported" if value is not None else "not_reported"
    for metric in A_METRICS:
        statuses.setdefault(metric.source_key, "pending" if metric.level == "02_target_track_level" else "not_reported")
    return values, statuses


def build_accuracy(root: Path) -> list[Path]:
    category = root / "A_model_accuracy"
    registry = read_registry(root)
    outputs: list[Path] = []
    for split in ("validation", "test"):
        cache: dict[str, tuple[dict[str, Any], dict[str, str], str]] = {}
        for model in registry:
            base = record_root(category, model)
            if model["model_role"] == "external_reference":
                values, statuses = paper_accuracy(base, split)
                cache[model["evaluation_id"]] = (values, statuses, "ICCV-2025-EV-UAV-Table-2")
                continue
            event_path = base / split / "event_level" / "summary.json"
            track_path = base / split / "target_track_level" / "summary.json"
            values: dict[str, Any] = {}
            protocols = []
            if event_path.exists():
                event_values, event_protocol = event_metrics(event_path)
                values.update(event_values); protocols.append(event_protocol)
            if track_path.exists():
                track_values, track_protocol = track_metrics(track_path)
                values.update(track_values); protocols.append(track_protocol)
            paper_values: dict[str, Any] = {}
            paper_statuses: dict[str, str] = {}
            if (base / "paper_reference.json").exists():
                paper_values, paper_statuses = paper_accuracy(base, split)
                for key, value in paper_values.items():
                    if key not in values and value is not None:
                        values[key] = value
            statuses = {}
            for metric in A_METRICS:
                if ((metric.level == "01_event_level" and event_path.exists()) or
                        (metric.level == "02_target_track_level" and track_path.exists())) and metric.source_key in values:
                    statuses[metric.source_key] = "complete"
                elif paper_statuses.get(metric.source_key) == "paper_reported":
                    statuses[metric.source_key] = "paper_reported"
                else:
                    statuses[metric.source_key] = "pending"
            cache[model["evaluation_id"]] = (values, statuses, ";".join(protocols))

        for metric in A_METRICS:
            rows = []
            for model in registry:
                values, statuses, protocol = cache[model["evaluation_id"]]
                base = record_root(category, model)
                status = statuses[metric.source_key]
                if status == "paper_reported":
                    source = base / "paper_reference.json"
                elif model["model_role"] == "external_reference":
                    source = base / ("measured_status.json" if metric.level == "02_target_track_level" else "paper_reference.json")
                else:
                    source = base / split / ("target_track_level" if metric.level == "02_target_track_level" else "event_level") / "summary.json"
                evidence = "paper_reported" if status == "paper_reported" else ("pending_import" if status == "pending" else ("measured" if status == "complete" else "not_reported"))
                if status == "paper_reported":
                    row_protocol = "ICCV-2025-EV-UAV-Table-2"
                elif status == "pending" and metric.level == "02_target_track_level":
                    row_protocol = "evuav-observable-event-track-hota-v1"
                elif status == "pending" and model["evaluation_id"] == "EMS-YOLO":
                    row_protocol = "evuav-event-metrics-v1+ems-yolo-sliding-box-to-event-v1"
                elif status == "pending" and model["evaluation_id"] == "EV-SpSegNet":
                    row_protocol = "evuav-33e2aff-official-compatible-v1+step3-density-20000"
                else:
                    row_protocol = protocol
                if status == "pending":
                    notes = "待服务器导入同口径逐事件概率"
                    if model["evaluation_id"] == "EMS-YOLO":
                        notes += "；阈值0.407来自历史Validation选择，Test为主要无偏比较"
                else:
                    notes = "论文 Test 表 2 直接报告；服务器实测后由同口径结果替换" if status == "paper_reported" else ""
                rows.append(make_row(model, values.get(metric.source_key), metric.unit, status, evidence, row_protocol, source, notes))
            path = category / "01_metric_comparisons" / metric.level / metric.directory / f"{split}.csv"
            write_csv(path, rows); outputs.append(path)
    return outputs


def load_complexity_values(category: Path, model: dict[str, str], split: str) -> tuple[dict[str, Any], str, Path]:
    base = record_root(category, model)
    if model["model_role"] == "external_reference":
        paper_path = base / "common" / "paper_reference.json"
        paper = read_json(paper_path)
        return {"total_weight_parameters": paper["paper"]["reported_weight_parameters"]}, "ICCV-2025-EV-UAV-Table-2", paper_path
    if split == "common":
        path = base / "common" / "model_profile.json"
        if not path.exists() and (base / "common" / "paper_reference.json").exists():
            paper_path = base / "common" / "paper_reference.json"; paper = read_json(paper_path)
            return {"total_weight_parameters": paper["paper"]["reported_weight_parameters"]}, "ICCV-2025-EV-UAV-Table-2", paper_path
        data = read_json(path)
        return data["summary"], data["protocol"], path
    path = base / split / "summary.json"
    if not path.exists():
        paper_path = base / "common" / "paper_reference.json"
        fallback: dict[str, Any] = {}
        if paper_path.exists():
            paper = read_json(paper_path)
            if paper.get("paper", {}).get("architecture") == "ANN":
                fallback["effective_acs_per_sequence"] = 0.0
        protocol = "pending-same-protocol-server-measurement"
        if paper_path.exists():
            paper = read_json(paper_path)
            protocol = paper.get("pending_protocol", protocol)
        if model["evaluation_id"] == "EV-SpSegNet" and protocol == "pending-same-protocol-server-measurement":
            protocol = "evspsegnet-official-float-sparse-operator-complexity-v1"
        return fallback, protocol, paper_path if paper_path.exists() else path
    data = read_json(path)
    values = dict(data["summary"]); sequences = int(values["sequences"])
    values.setdefault("effective_acs_per_sequence", float(values.get("effective_acs", 0)) / sequences)
    values["effective_ops_per_sequence"] = float(values["effective_macs_per_sequence"]) + float(values["effective_acs_per_sequence"])
    return values, data["protocol"], path


def b_metric_path(category: Path, metric: Metric, split: str) -> Path:
    base = category / "01_metric_comparisons"
    return (base / metric.level / metric.directory if metric.level else base / metric.directory) / f"{split}.csv"


def build_complexity(root: Path) -> list[Path]:
    category = root / "B_model_complexity"
    registry = [model for model in read_registry(root) if model["evaluation_track"] == "software" or model["model_role"] == "external_reference"]
    outputs = []
    for metric in B_METRICS:
        splits = ("common",) if metric.code in {"B01", "B02"} else ("validation", "test")
        for split in splits:
            rows = []
            for model in registry:
                values, protocol, source = load_complexity_values(category, model, split)
                value = values.get(metric.source_key)
                paper_source = source.name == "paper_reference.json"
                architecture_defined = metric.code == "B04b" and paper_source and value == 0 and model["model_role"] == "formal"
                status = "complete" if architecture_defined else ("paper_reported" if paper_source and value is not None else ("complete" if value is not None else ("pending" if model["model_role"] == "formal" else "not_reported")))
                evidence = "architecture_defined" if architecture_defined else ("paper_reported" if status == "paper_reported" else ("pending_cuda_evaluation" if status == "pending" else ("derived" if metric.code == "B04c" and status == "complete" else ("measured" if status == "complete" else "not_reported"))))
                note = "B04c = B04a + B04b；这是与 B03 稠密 MAC 的正式对比口径" if metric.code == "B04c" else ""
                if status == "paper_reported": note = "论文约数；服务器严格加载 checkpoint 后以实测 B01 替换"
                if status == "pending": note = (note + "；" if note else "") + "待冻结模型 CUDA 服务器包实测导入"
                if architecture_defined: note = "普通 ANN 不执行 SNN 脉冲累加 AC；按定义为 0"
                rows.append(make_row(model, value, metric.unit, status, evidence, protocol, source, note))
            path = b_metric_path(category, metric, split); write_csv(path, rows); outputs.append(path)
    return outputs


def build_deployment(root: Path) -> list[Path]:
    """Build board C tables, plus C07 runtime for formal software and board models."""
    category = root / "C_deployment_efficiency_and_hardware_cost"
    formal_models = [model for model in read_registry(root) if model["model_role"] == "formal" and model["evaluation_track"] in {"software", "board"}]
    outputs: list[Path] = []
    for code, folder, source_key, unit, scope in C_METRICS:
        models = [model for model in formal_models if code == "C07" or model["evaluation_track"] == "board"]
        if not models:
            continue
        splits = ("common",) if scope == "common" else ("validation", "test")
        for split in splits:
            rows = []
            for model in models:
                source = record_root(category, model) / split / "summary.json"
                data = read_json(source) if source.exists() else {}
                summary = data.get("summary", data.get("metrics", {}))
                value = summary.get(source_key, summary.get(code))
                if code == "C07" and value is not None:
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                        raise ValueError(f"{source}: runtime_ms_per_sequence must be a finite positive number or null")
                status = "complete" if value is not None else "pending"
                measured_evidence = "software_measured" if model["evaluation_track"] == "software" else "board_measured"
                pending_evidence = "pending_runtime_measurement" if code == "C07" else "pending_board_measurement"
                evidence = data.get("evidence_type", measured_evidence if status == "complete" else pending_evidence)
                protocol = data.get("protocol_id", data.get("protocol", ""))
                if code == "C07":
                    protocol = data.get("runtime_protocol_id", protocol or RUNTIME_PROTOCOL)
                missing_note = "待固定平台完整序列 runtime 实测；当前无源记录或指标值" if code == "C07" else "缺少板端正式源记录或指标值"
                rows.append(make_row(model, value, unit, status, evidence, protocol, source, "" if status == "complete" else missing_note))
            path = category / "01_metric_comparisons" / Path(folder) / f"{split}.csv"
            write_csv(path, rows); outputs.append(path)
    return outputs


def find_metric_file(category: Path, code: str, split: str) -> Path | None:
    candidates = list((category / "01_metric_comparisons").glob(f"**/{code}_*/{split}.csv"))
    return candidates[0] if len(candidates) == 1 else None


def row_is_complete(path: Path | None, evaluation_id: str) -> bool:
    if path is None or not path.exists(): return False
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return any(row["evaluation_id"] == evaluation_id and row["status"] == "complete" and row["value"] != "" for row in csv.DictReader(stream))


def group_status(category: Path, codes: Iterable[str], split: str, evaluation_id: str) -> str:
    return "complete" if all(row_is_complete(find_metric_file(category, code, split), evaluation_id) for code in codes) else "incomplete"


def build_coverage(root: Path, fail_on_incomplete: bool = True) -> tuple[Path, bool]:
    registry = read_registry(root); a_root = root / "A_model_accuracy"; b_root = root / "B_model_complexity"; c_root = root / "C_deployment_efficiency_and_hardware_cost"
    fields = ("evaluation_id", "model_name", "evaluation_track", "model_role", "a_validation", "a_test", "b_common", "b_validation", "b_test", "c_common", "c_validation", "c_test", "overall_status", "details")
    rows = []; all_ok = True
    for model in registry:
        identity = {key: model[key] for key in fields[:4]}
        if model["model_role"] == "external_reference":
            rows.append({**identity, **{key: "excluded" for key in fields[4:12]}, "overall_status": "excluded_external_reference", "details": "外部参照不参与正式准入完整性判定"}); continue
        evaluation_id = model["evaluation_id"]
        a_val = group_status(a_root, (m.code for m in A_METRICS), "validation", evaluation_id)
        a_test = group_status(a_root, (m.code for m in A_METRICS), "test", evaluation_id)
        if model["evaluation_track"] == "software":
            b_common = group_status(b_root, ("B01", "B02"), "common", evaluation_id)
            b_val = group_status(b_root, ("B03", "B04a", "B04b", "B04c"), "validation", evaluation_id)
            b_test = group_status(b_root, ("B03", "B04a", "B04b", "B04c"), "test", evaluation_id)
            c_common = c_val = c_test = "not_required"; required = (a_val, a_test, b_common, b_val, b_test)
            details = "软件正式模型必测 A(Validation/Test)+B(Common/Validation/Test)"
        elif model["evaluation_track"] == "board":
            b_common = b_val = b_test = "not_required"
            c_common = group_status(c_root, C_COMMON, "common", evaluation_id); c_val = group_status(c_root, C_SPLIT, "validation", evaluation_id); c_test = group_status(c_root, C_SPLIT, "test", evaluation_id)
            required = (a_val, a_test, c_common, c_val, c_test); details = "板端正式模型必测 A(Validation/Test)+C(Common/Validation/Test)"
        else: raise ValueError(f"Unknown evaluation_track: {model['evaluation_track']}")
        overall = "complete" if all(item == "complete" for item in required) else "incomplete"; all_ok &= overall == "complete"
        rows.append({**identity, "a_validation": a_val, "a_test": a_test, "b_common": b_common, "b_validation": b_val, "b_test": b_test, "c_common": c_common, "c_validation": c_val, "c_test": c_test, "overall_status": overall, "details": details})
    path = root / "evaluation_coverage.csv"; write_csv(path, rows, fields)
    if fail_on_incomplete and not all_ok: raise SystemExit(1)
    return path, all_ok


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4] / "5_model_evaluation"); parser.add_argument("--category", choices=("all", "accuracy", "complexity", "deployment", "coverage"), default="all"); parser.add_argument("--allow-incomplete", action="store_true"); args = parser.parse_args()
    if args.category in {"all", "accuracy"}: build_accuracy(args.root)
    if args.category in {"all", "complexity"}: build_complexity(args.root)
    if args.category in {"all", "deployment"}: build_deployment(args.root)
    if args.category in {"all", "coverage"}:
        path, ok = build_coverage(args.root, fail_on_incomplete=not args.allow_incomplete); print(f"{path}: {'complete' if ok else 'incomplete'}")


if __name__ == "__main__": main()
