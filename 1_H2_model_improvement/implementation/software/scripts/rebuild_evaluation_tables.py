"""Build registry-ordered per-metric evaluation CSVs and validate coverage."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from evaluation_contract import (
    LOCALIZATION_PROTOCOL, RUNTIME_PROTOCOL, PREFIX_SCOPE, FULL_SCOPE,
    content_sha256, deployment_issues, finite, policy_at, protocol_of, sha256,
    state_inventory_issues, validate_value, verify_record,
)


CSV_FIELDS = (
    "evaluation_id", "model_name", "evaluation_track", "model_role", "platform_id",
    "value", "unit", "status", "evidence_type", "protocol_id", "source_file", "notes",
    "source_sha256", "verification_status", "comparison_scope", "causality",
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
    Metric("A07", "fa", "Fa", "component/pixel/window", "01_event_level"),
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
    Metric("B05", "persistent_state_memory", "persistent_state_bytes", "byte", ""),
)

C_COMMON = ("C04a", "C06a", "C06b", "C06c", "C06d", "C06e")
C_SPLIT = (
    "C01a", "C01b", "C01c", "C01d", "C02a", "C02b", "C02c", "C02d",
    "C03a", "C03b", "C03c", "C03d", "C04b", "C04c", "C05a", "C05b", "C07",
    "C08a", "C08b", "C09a", "C09b", "C09c", "C09d",
    "C10a", "C10b", "C10c", "C10d", "C10e", "C10f",
)

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
    ("C08a", "C08_runtime_memory/C08a_host_peak_rss", "host_peak_rss_bytes", "byte", "split"),
    ("C08b", "C08_runtime_memory/C08b_device_peak_allocation", "device_peak_allocation_bytes", "byte", "split"),
    ("C09a", "C09_coverage_and_backlog/C09a_processing_coverage", "processing_coverage", "rate", "split"),
    ("C09b", "C09_coverage_and_backlog/C09b_output_coverage", "output_coverage", "rate", "split"),
    ("C09c", "C09_coverage_and_backlog/C09c_dropped_event_rate", "dropped_event_rate", "rate", "split"),
    ("C09d", "C09_coverage_and_backlog/C09d_max_backlog_events", "max_backlog_events", "count", "split"),
    ("C10a", "C10_numerical_equivalence/C10a_exact_agreement", "exact_agreement", "rate", "split"),
    ("C10b", "C10_numerical_equivalence/C10b_max_abs_error", "max_abs_error", "score", "split"),
    ("C10c", "C10_numerical_equivalence/C10c_rmse", "rmse", "score", "split"),
    ("C10d", "C10_numerical_equivalence/C10d_prediction_agreement", "prediction_agreement", "rate", "split"),
    ("C10e", "C10_numerical_equivalence/C10e_overflow_rate", "overflow_rate", "rate", "split"),
    ("C10f", "C10_numerical_equivalence/C10f_clipping_rate", "clipping_rate", "rate", "split"),
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


def checked_row(root, model, code, split, family, source, value, unit,
                status, evidence, protocol, notes=""):
    """Attach source identity and gate complete on source/protocol validation."""
    validate_value(value, code, unit)
    source = Path(source)
    scope = FULL_SCOPE if code == "C07" else PREFIX_SCOPE
    policy = policy_at(root)
    data = read_json(source) if source.is_file() else {}
    comparison = data.get("comparison", {})
    context = policy.get("model_context", {}).get(model["evaluation_id"], {})
    check = status
    if status == "complete":
        if code == "B04b" and evidence == "architecture_defined":
            errors = [] if value == 0 and data.get("paper", {}).get("architecture") == "ANN" else ["invalid_architecture_zero"]
        elif family == "deployment":
            errors = deployment_issues(root, source, model, split, code)
        else:
            errors = verify_record(root, source, model, split, family, scope)
        if code == "B05" and not errors:
            errors.extend(state_inventory_issues(data))
        if errors:
            status = "unverified"
            check = ";".join(errors)
            notes = (notes + "；" if notes else "") + "未通过源记录/协议检查：" + check
        else:
            check = "verified"
    try:
        shown = source.relative_to(root).as_posix()
    except ValueError:
        shown = source.as_posix()
    row = make_row(model, value, unit, status, evidence, protocol, shown, notes)
    legacy = policy.get("reviewed_legacy_sources", {}).get(shown, {})
    row.update(source_sha256=sha256(source) if source.is_file() else "",
               verification_status=check,
               comparison_scope=comparison.get("scope_id", legacy.get("scope_id", "unknown" if source.is_file() else scope)),
               causality=comparison.get("causality", context.get("causality", "unknown")))
    return row


def event_metrics(path: Path) -> tuple[dict[str, Any], str]:
    data = read_json(path)
    working = data["working_point"]
    event = working["overall"]["event"]
    localization = working["overall"]["localization"]
    def number(value):
        if value is None: return None
        if not finite(value): raise ValueError(f"{path}: invalid event metric {value!r}")
        return float(value)
    if any(not finite(event[k], integer=True) for k in ("tp", "fp", "fn", "tn")):
        raise ValueError(f"{path}: confusion counts must be nonnegative integers")
    per_sequence = working.get("per_sequence", {})
    per_sequence = per_sequence.values() if isinstance(per_sequence, dict) else per_sequence
    recalls = [float(row["event"]["recall"]) for row in per_sequence if row.get("event", {}).get("recall") is not None]
    macro_recall = (number(working["macro_sequence_recall"])
                    if "macro_sequence_recall" in working else sum(recalls) / len(recalls) if recalls else None)
    values = {
        "Event AP": number(working["average_precision"]), "Event IoU": number(event["iou"]),
        "Precision": number(event["precision"]), "ACC / Target-event Recall": number(event["recall"]),
        "F1": number(event["f1"]), "Pd": localization.get("pd"), "Fa": localization.get("fa"),
        "Macro sequence recall": macro_recall,
        "High-density recall": number(working["high_density"]["event"]["recall"]),
        "Ordinary-density recall": number(working["ordinary_density"]["event"]["recall"]),
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
    outputs = []
    for split in ("validation", "test"):
        for metric in A_METRICS:
            rows = []
            for model in read_registry(root):
                base = record_root(category, model)
                family = "track" if metric.level == "02_target_track_level" else "event"
                source = base / split / ("target_track_level" if family == "track" else "event_level") / "summary.json"
                value, status, evidence, protocol, note = None, "pending", "pending_evaluation", "", "待冻结协议源记录"
                if source.is_file():
                    values, protocol = track_metrics(source) if family == "track" else event_metrics(source)
                    value = values.get(metric.source_key)
                    status = "complete" if value is not None else "not_applicable"
                    evidence, note = "measured", ""
                elif (base / "paper_reference.json").is_file():
                    values, statuses = paper_accuracy(base, split)
                    value = values.get(metric.source_key)
                    status = statuses.get(metric.source_key, "pending")
                    if value is not None:
                        source = base / "paper_reference.json"
                        evidence, protocol = "paper_reported", "ICCV-2025-EV-UAV-Table-2"
                        note = "论文参照；不参与同协议完成判定"
                if metric.code in {"A06", "A07"}:
                    modern = base / split / "event_level" / "localization_v2.json"
                    if modern.is_file():
                        source = modern
                        data = read_json(source)
                        value = data["summary"].get("pd" if metric.code == "A06" else "fa")
                        protocol, family = protocol_of(data), "localization"
                        status = "complete" if value is not None else "not_applicable"
                        evidence, note = "recomputed_frozen_predictions", "统一半开窗工程口径；宽松 Pd 不代表框定位准确率"
                    elif status == "complete":
                        status, evidence = "legacy_only", "historical_evaluator"
                        note = "保留旧数值，不能作为 v2 完成；待使用冻结逐事件输出重新统计，禁止仅换协议标签"
                rows.append(checked_row(root, model, metric.code, split, family, source,
                                        value, metric.unit, status, evidence, protocol, note))
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
        if not path.exists(): return {}, "", path
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
    if sequences <= 0: raise ValueError("Complexity records require positive sequence counts")
    if "effective_acs_per_sequence" not in values and "effective_acs" in values:
        values["effective_acs_per_sequence"] = float(values["effective_acs"]) / sequences
    if values.get("effective_macs_per_sequence") is not None and values.get("effective_acs_per_sequence") is not None:
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
        splits = ("common",) if metric.code in {"B01", "B02", "B05"} else ("validation", "test")
        for split in splits:
            rows = []
            for model in registry:
                if metric.code == "B05":
                    source = record_root(category, model) / "common" / "state_memory.json"
                    data = read_json(source) if source.is_file() else {}
                    values, protocol = data.get("summary", {}), protocol_of(data)
                else:
                    values, protocol, source = load_complexity_values(category, model, split)
                value = values.get(metric.source_key)
                paper_source = source.name == "paper_reference.json"
                architecture_defined = metric.code == "B04b" and paper_source and value == 0 and model["model_role"] == "formal"
                status = "complete" if architecture_defined else ("paper_reported" if paper_source and value is not None else ("complete" if value is not None else ("pending" if model["model_role"] == "formal" else "not_reported")))
                evidence = "architecture_defined" if architecture_defined else ("paper_reported" if status == "paper_reported" else ("pending_cuda_evaluation" if status == "pending" else ("derived" if metric.code == "B04c" and status == "complete" else ("measured" if status == "complete" else "not_reported"))))
                note = "非零连接理论运算量；不证明实际跳零、加速或节能" if metric.code.startswith("B04") else ""
                if metric.code == "B04c": note += "；MAC+AC 为未加权计数，不作为跨位宽/架构成本主排名"
                if metric.code == "B05": note = "需完整持久状态清单；历史 Proposal 分量不是全部运行内存"
                if status == "paper_reported": note = "论文约数；服务器严格加载 checkpoint 后以实测 B01 替换"
                if status == "pending":
                    if metric.code == "B05":
                        evidence = "pending_state_inventory"
                        note += "；待完整状态清单与证据导出"
                    else:
                        note = (note + "；" if note else "") + "待冻结模型同协议统计导入"
                if architecture_defined: note = "普通 ANN 不执行 SNN 脉冲累加 AC；按定义为 0"
                rows.append(checked_row(root, model, metric.code, split, "complexity", source,
                                        value, metric.unit, status, evidence, protocol, note))
            path = b_metric_path(category, metric, split); write_csv(path, rows); outputs.append(path)
    return outputs


def build_deployment(root: Path) -> list[Path]:
    category = root / "C_deployment_efficiency_and_hardware_cost"
    formal = [m for m in read_registry(root) if m["model_role"] == "formal"]
    outputs = []
    for code, folder, source_key, unit, scope in C_METRICS:
        models = [m for m in formal if m["evaluation_track"] == "board" or
                  (m["evaluation_track"] == "software" and (code == "C07" or code.startswith(("C08", "C09"))))]
        if not models: continue
        for split in (("common",) if scope == "common" else ("validation", "test")):
            rows = []
            for model in models:
                source = record_root(category, model) / split / "summary.json"
                runtime_source = source.with_name("runtime.json")
                if code == "C07" and runtime_source.is_file(): source = runtime_source
                data = read_json(source) if source.is_file() else {}
                summary = data.get("summary", data.get("metrics", {}))
                value = summary.get(source_key, summary.get(code))
                status = "complete" if value is not None else "pending"
                evidence = data.get("evidence_type", "unverified" if value is not None else "pending_measurement")
                protocol = data.get("runtime_protocol_id", "") if code == "C07" else protocol_of(data)
                rows.append(checked_row(root, model, code, split, "deployment", source,
                                        value, unit, status, evidence, protocol,
                                        "" if value is not None else "待符合 v2 协议的实测；空值不代表零"))
            path = category / "01_metric_comparisons" / folder / f"{split}.csv"
            write_csv(path, rows); outputs.append(path)
    return outputs


def find_metric_file(category: Path, code: str, split: str) -> Path | None:
    candidates = list((category / "01_metric_comparisons").glob(f"**/{code}_*/{split}.csv"))
    return candidates[0] if len(candidates) == 1 else None


def row_is_complete(path: Path | None, evaluation_id: str) -> bool:
    if path is None or not path.is_file(): return False
    root = next((p for p in path.parents if (p / "model_registry.csv").is_file()), None)
    if root is None: return False
    model = next((m for m in read_registry(root) if m["evaluation_id"] == evaluation_id), None)
    if model is None: return False
    with path.open(encoding="utf-8-sig", newline="") as stream:
        matches = [r for r in csv.DictReader(stream) if r["evaluation_id"] == evaluation_id]
    if len(matches) != 1: return False
    row = matches[0]
    if row.get("status") != "complete" or row.get("verification_status") != "verified": return False
    code, split = path.parent.name.split("_", 1)[0], path.stem
    source = Path(row["source_file"])
    source = source if source.is_absolute() else root / source
    if not source.is_file() or row.get("source_sha256") != sha256(source): return False
    try:
        if code.startswith("A"):
            metric = next(m for m in A_METRICS if m.code == code)
            family = "track" if metric.level == "02_target_track_level" else "event"
            if code in {"A06", "A07"}:
                data = read_json(source); family = "localization"
                expected = data["summary"]["pd" if code == "A06" else "fa"]
            else:
                expected = (track_metrics(source) if family == "track" else event_metrics(source))[0][metric.source_key]
            unit = metric.unit
            expected_protocol = protocol_of(read_json(source))
        elif code.startswith("B"):
            metric = next(m for m in B_METRICS if m.code == code)
            family, unit = "complexity", metric.unit
            if code == "B05":
                expected = read_json(source)["summary"][metric.source_key]
                expected_protocol = protocol_of(read_json(source))
            else:
                values, expected_protocol, expected_source = load_complexity_values(root / "B_model_complexity", model, split)
                if source.absolute() != expected_source.absolute(): return False
                expected = values[metric.source_key]
        else:
            spec = next(m for m in C_METRICS if m[0] == code)
            data = read_json(source); summary = data.get("summary", data.get("metrics", {}))
            expected, unit, family = summary.get(spec[2], summary.get(code)), spec[3], "deployment"
            expected_protocol = data.get("runtime_protocol_id", "") if code == "C07" else protocol_of(data)
        if row["protocol_id"] != expected_protocol: return False
        if row["unit"] != unit or expected is None or not math.isclose(float(row["value"]), expected, rel_tol=1e-12, abs_tol=1e-12): return False
        fresh = checked_row(root, model, code, split, family, source, expected, unit,
                            "complete", row["evidence_type"], row["protocol_id"])
        return fresh["status"] == "complete" and all(row.get(k) == fresh[k] for k in ("comparison_scope", "causality"))
    except (ValueError, KeyError, TypeError, StopIteration, OSError):
        return False


def build_uncertainty(root: Path) -> Path:
    """Expose existing paired IoU intervals; never manufacture absolute CIs."""
    policy = policy_at(root)
    reference = policy.get("reviewed_uncertainty", {})
    source = root.parent / reference.get("source_file", "missing")
    data = read_json(source) if source.is_file() and content_sha256(source) == reference.get("content_sha256") else {}
    rows = []
    for split in ("validation", "test"):
        result = data.get(split, {})
        delta = result.get("delta_pro_minus_reference", {}).get("H2", {})
        interval = delta.get("95_interval", [None, None, None])
        rows.append({"evaluation_id": reference.get("evaluation_id", ""),
                     "reference_evaluation_id": reference.get("reference_evaluation_id", ""),
                     "split": split, "metric": "Event IoU", "delta": delta.get("point"),
                     "ci95_lower": interval[0], "ci95_upper": interval[2],
                     "unit": "fraction_difference", "sequence_count": len(policy.get("sequence_ids", {}).get(split, [])),
                     "bootstrap_repeats": result.get("repeats"), "method": result.get("method", ""),
                     "status": "historical_verified" if delta else "pending",
                     "source_file": reference.get("source_file", ""),
                     "source_sha256": sha256(source) if source.is_file() else "",
                     "notes": "同评分前缀的配对差值；H2 非因果、H2-pro 因果；不是同实时条件下的因果收益；并非绝对指标 CI"})
    path = root / "A_model_accuracy/01_metric_comparisons/uncertainty/paired_iou_delta.csv"
    write_csv(path, rows, tuple(rows[0]))
    return path


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
            b_common = group_status(b_root, ("B01", "B02", "B05"), "common", evaluation_id)
            b_val = group_status(b_root, ("B03", "B04a", "B04b", "B04c"), "validation", evaluation_id)
            b_test = group_status(b_root, ("B03", "B04a", "B04b", "B04c"), "test", evaluation_id)
            c_common = c_val = c_test = "not_required"; required = (a_val, a_test, b_common, b_val, b_test)
            details = "v2：软件要求 A+B；旧 Pd/Fa 与缺少 B05 清单会使准入 incomplete；C 为补充项"
        elif model["evaluation_track"] == "board":
            b_common = b_val = b_test = "not_required"
            c_common = group_status(c_root, C_COMMON, "common", evaluation_id); c_val = group_status(c_root, C_SPLIT, "validation", evaluation_id); c_test = group_status(c_root, C_SPLIT, "test", evaluation_id)
            required = (a_val, a_test, c_common, c_val, c_test); details = "v2：板端要求 A+C01–C10，并核验源记录、协议及证据哈希"
        else: raise ValueError(f"Unknown evaluation_track: {model['evaluation_track']}")
        overall = "complete" if all(item == "complete" for item in required) else "incomplete"; all_ok &= overall == "complete"
        rows.append({**identity, "a_validation": a_val, "a_test": a_test, "b_common": b_common, "b_validation": b_val, "b_test": b_test, "c_common": c_common, "c_validation": c_val, "c_test": c_test, "overall_status": overall, "details": details})
    path = root / "evaluation_coverage.csv"; write_csv(path, rows, fields)
    if fail_on_incomplete and not all_ok: raise SystemExit(1)
    return path, all_ok


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).absolute().parents[4] / "5_model_evaluation"); parser.add_argument("--category", choices=("all", "accuracy", "complexity", "deployment", "coverage"), default="all"); parser.add_argument("--allow-incomplete", action="store_true"); args = parser.parse_args()
    if args.category in {"all", "accuracy"}: build_accuracy(args.root); build_uncertainty(args.root)
    if args.category in {"all", "complexity"}: build_complexity(args.root)
    if args.category in {"all", "deployment"}: build_deployment(args.root)
    if args.category in {"all", "coverage"}:
        path, ok = build_coverage(args.root, fail_on_incomplete=not args.allow_incomplete); print(f"{path}: {'complete' if ok else 'incomplete'}")


if __name__ == "__main__": main()
