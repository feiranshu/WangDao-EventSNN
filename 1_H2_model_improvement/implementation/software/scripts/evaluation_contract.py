"""Shared, fail-closed record validation for evaluation tables and imports."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path


LOCALIZATION_PROTOCOL = "evuav-localization-half-open-v2"
RUNTIME_PROTOCOL = "full-sequence-runtime-v2"
DEPLOYMENT_PROTOCOL = "deployment-measurement-v2"
PREFIX_SCOPE = "first-event-complete-100ms-prefix-v1"
FULL_SCOPE = "full-input-sequence-v1"
SCHEMA = "evaluation-record-v2"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(path):
    """Canonical JSON digest for reviewed historical content across CRLF/LF."""
    data = json.dumps(read_json(path), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def is_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def finite(value, *, minimum=0, maximum=None, positive=False, integer=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and
            math.isfinite(value) and value >= minimum and
            (not positive or value > 0) and (maximum is None or value <= maximum) and
            (not integer or value == int(value)))


def validate_value(value, code, unit):
    if value is None:
        return
    # Dynamic baseline-subtracted measurements may be slightly negative due to
    # noise; retain the signed value and its uncertainty instead of clipping it.
    signed = code in {"C04c", "C05b"}
    maximum = 1 if unit in {"rate", "component/pixel/window"} else None
    if not finite(value, minimum=-math.inf if signed else 0, maximum=maximum,
                  positive=code in {"C06e", "C07"},
                  integer=unit in {"byte", "count"} and code != "C06c"):
        raise ValueError(f"{code}: invalid {unit} value {value!r}")


def artifact_path(base, relative):
    """Only accept record-local evidence; reject traversal and absolute paths."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Evidence path must be a nonempty relative POSIX path")
    item = Path(relative)
    if item.is_absolute() or ".." in item.parts or ":" in relative:
        raise ValueError("Evidence must stay inside its record directory")
    return Path(base) / item


def protocol_of(data):
    return str(data.get("protocol_id", data.get("metric_protocol", data.get("protocol",
               data.get("evaluation", {}).get("metric_protocol", "")))))


def policy_at(root):
    path = Path(root) / "comparison_contract.json"
    return read_json(path) if path.is_file() else {}


def verify_record(root, source, model, split, family, expected_scope=PREFIX_SCOPE):
    """Return issues. A historic digest is reviewed provenance, not new data."""
    root, source = Path(root), Path(source)
    if not source.is_file():
        return ["missing_source"]
    data = read_json(source)
    policy = policy_at(root)
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError:
        relative = ""
    old = policy.get("reviewed_legacy_sources", {}).get(relative)
    if old and content_sha256(source) == old["content_sha256"]:
        if (old["evaluation_id"] == model["evaluation_id"] and old["split"] == split
                and old["family"] == family and old["scope_id"] == expected_scope
                and old["weight_sha256"] == model.get("weight_sha256")):
            return []
        return ["legacy_identity_or_scope_mismatch"]
    errors = []
    for key, expected in (("record_schema", SCHEMA), ("evaluation_id", model["evaluation_id"]),
                          ("split", split), ("platform_id", model["platform_id"])):
        if data.get(key) != expected:
            errors.append("missing_or_mismatched_" + key)
    provenance = data.get("provenance", {})
    for key in ("weight_sha256", "config_sha256", "index_sha256"):
        if not is_hash(provenance.get(key)):
            errors.append("invalid_" + key)
    if model.get("weight_sha256") and provenance.get("weight_sha256") != model["weight_sha256"]:
        errors.append("weight_mismatch")
    if provenance.get("index_sha256") != policy.get("index_sha256"):
        errors.append("frozen_index_mismatch")
    comparison = data.get("comparison", {})
    if comparison.get("scope_id") != expected_scope:
        errors.append("incompatible_input_scope")
    if comparison.get("causality") not in {"causal", "offline_noncausal"}:
        errors.append("missing_causality")
    context = policy.get("model_context", {}).get(model["evaluation_id"], {})
    if context.get("causality") in {"causal", "offline_noncausal"} and comparison.get("causality") != context["causality"]:
        errors.append("frozen_causality_mismatch")
    if comparison.get("numeric_path") != model.get("numeric_path"):
        errors.append("numeric_path_mismatch")
    if split != "common":
        names = comparison.get("sequence_ids", [])
        expected = policy.get("sequence_ids", {}).get(split, [])
        if not expected or not isinstance(names, list) or sorted(names) != sorted(expected):
            errors.append("sequence_population_mismatch")
    if not finite(comparison.get("threshold"), maximum=1, positive=True) or comparison.get("threshold", 1) >= 1:
        errors.append("missing_threshold")
    if "threshold" in context and comparison.get("threshold") != context["threshold"]:
        errors.append("frozen_threshold_mismatch")
    artifacts = provenance.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("missing_raw_evidence")
    else:
        for item in artifacts:
            try:
                path = artifact_path(source.parent, item.get("path"))
                if not path.is_file() or not is_hash(item.get("sha256")) or sha256(path) != item["sha256"]:
                    errors.append("evidence_hash_mismatch")
            except (ValueError, AttributeError):
                errors.append("invalid_evidence_path")
    allowed = policy.get("accepted_protocols", {}).get(family, [])
    if protocol_of(data) not in allowed:
        errors.append("incompatible_metric_protocol")
    if family == "localization" and not errors:
        errors.extend(localization_summary_issues(source, data, policy))
    if family == "event" and not errors:
        try:
            if data["working_point"].get("threshold") != comparison["threshold"]:
                errors.append("event_working_point_mismatch")
            event = data["working_point"]["overall"]["event"]
            if any(not finite(event[k], integer=True) for k in ("tp", "fp", "fn", "tn")):
                errors.append("invalid_confusion_counts")
            else:
                tp, fp, fn = (event[k] for k in ("tp", "fp", "fn"))
                for key, numerator, denominator in (("iou", tp, tp+fp+fn), ("precision", tp, tp+fp),
                                                    ("recall", tp, tp+fn), ("f1", 2*tp, 2*tp+fp+fn)):
                    value = numerator / denominator if denominator else None
                    if value is None and event.get(key) is not None:
                        errors.append("undefined_event_ratio_must_be_null")
                    elif value is not None and (not finite(event.get(key), maximum=1) or not math.isclose(value, event[key], rel_tol=1e-12)):
                        errors.append("event_ratio_mismatch")
        except (KeyError, TypeError):
            errors.append("invalid_event_summary_schema")
    return sorted(set(errors))


def state_inventory_issues(data):
    inventory = data.get("state_inventory", [])
    if not isinstance(inventory, list) or not inventory:
        return ["missing_state_inventory"]
    total = 0
    for item in inventory:
        if (not isinstance(item, dict) or any(not item.get(k) for k in ("name", "dtype", "lifetime")) or
                not isinstance(item.get("shape"), list) or
                any(not finite(v, integer=True) for v in item["shape"]) or
                not finite(item.get("capacity"), integer=True) or not finite(item.get("bytes"), integer=True)):
            return ["invalid_state_inventory_item"]
        total += item["bytes"]
    if data.get("summary", {}).get("persistent_state_bytes") != total:
        return ["state_inventory_total_mismatch"]
    return []


def localization_summary_issues(source, data, policy):
    try:
        evidence = data["provenance"]["artifacts"]
        artifact = data["localization_counts_file"]
        if artifact not in {a["path"] for a in evidence}:
            return ["unhashed_localization_counts"]
        rows = read_json(artifact_path(source.parent, artifact))
        names = data["comparison"]["sequence_ids"]
        if sorted(row["sequence_id"] for row in rows) != sorted(names):
            return ["localization_sequence_population_mismatch"]
        keys = ("targets", "correct_targets", "false_components", "frame_denominator",
                "scored_localization_events", "excluded_events")
        totals = dict.fromkeys(keys, 0)
        for row in rows:
            expected = policy["sequence_files"][data["split"]][row["sequence_id"]]
            if row["source_event_sha256"] != expected["sha256"] or not is_hash(row["prediction_sha256"]):
                return ["localization_source_identity_mismatch"]
            counts = row["counts"]
            if any(not finite(counts[k], integer=True) for k in keys):
                return ["invalid_localization_counts"]
            exposure = (row["end_ms"] - row["start_ms"]) / 50
            if exposure <= 0 or not math.isclose(exposure, counts["frame_denominator"], rel_tol=0, abs_tol=1e-9) or counts["frame_denominator"] % 2:
                return ["localization_exposure_mismatch"]
            if counts["scored_localization_events"] + counts["excluded_events"] != expected["events"] or counts["correct_targets"] > counts["targets"]:
                return ["localization_count_population_mismatch"]
            for key in keys: totals[key] += counts[key]
        summary = data["summary"]
        if any(summary.get(k) != value for k, value in totals.items()):
            return ["localization_totals_mismatch"]
        derived = {"pd": totals["correct_targets"] / totals["targets"] if totals["targets"] else None,
                   "fa": totals["false_components"] / (totals["frame_denominator"] * 346 * 260)}
        for key, value in derived.items():
            if value is None:
                if summary.get(key) is not None: return ["undefined_localization_must_be_null"]
            elif not finite(summary.get(key), maximum=1) or not math.isclose(summary[key], value, rel_tol=1e-12, abs_tol=1e-15):
                return ["localization_ratio_mismatch"]
        return []
    except (KeyError, ValueError, TypeError, ZeroDivisionError, OSError):
        return ["invalid_localization_record_schema"]


def deployment_issues(root, source, model, split, code):
    data = read_json(source)
    scope = FULL_SCOPE if code == "C07" else PREFIX_SCOPE
    errors = verify_record(root, source, model, split, "deployment", scope)
    if data.get("evidence_type") != model["evaluation_track"] + "_measured":
        errors.append("missing_explicit_measurement_evidence_type")
    machine = data.get("measurement", {})
    for key in ("host_model", "backend", "precision", "software_versions"):
        if not machine.get(key): errors.append("missing_" + key)
    for key in ("threads", "batch_size"):
        if not finite(machine.get(key), positive=True, integer=True): errors.append("invalid_" + key)
    if model["evaluation_track"] == "board":
        if machine.get("transport") not in {"MMIO", "XRT"}: errors.append("missing_board_transport")
        if not is_hash(machine.get("bitstream_sha256")): errors.append("missing_bitstream_hash")
        if not finite(machine.get("clock_mhz"), positive=True): errors.append("invalid_board_clock")
    context = data.get("metric_context", {}).get(code, {})
    if context.get("definition_id") != code + "-v2" or not context.get("raw_artifact"):
        errors.append("missing_metric_context")
    known = {item.get("path") for item in data.get("provenance", {}).get("artifacts", []) if isinstance(item, dict)}
    if context.get("raw_artifact") not in known:
        errors.append("metric_evidence_not_hashed")
    required = {
        "C01": ("kernel_name", "sample_unit", "timer_resolution_ns"),
        "C02": ("timestamp_domain", "sample_unit", "includes_window_wait", "includes_queue"),
        "C03": ("offered_event_rate", "deadline_ms", "run_mode"),
        "C04": ("power_domain", "sensor", "sampling_hz", "idle_configuration"),
        "C05": ("power_domain", "sensor", "sampling_hz", "integration_interval", "idle_configuration"),
        "C06": ("report_stage", "device_part", "timing_met"),
        "C08": ("memory_domain", "sampling_method", "includes_baseline"),
        "C09": ("coverage_counts_file", "queue_observation_method"),
        "C10": ("reference_evaluation_id", "reference_weight_sha256", "tensor_stage", "denominator", "tolerance"),
    }.get(code[:3], ())
    if any(key not in context or context[key] is None or context[key] == "" for key in required):
        errors.append("missing_metric_boundary_or_population")
    if code.startswith("C02") and (context.get("sample_unit") != "event" or context.get("includes_window_wait") is not True or context.get("includes_queue") is not True):
        errors.append("C02_requires_event_arrival_to_output_latency")
    if code.startswith("C06") and (context.get("report_stage") != "post_route" or context.get("timing_met") is not True):
        errors.append("requires_post_route_timing_closure")
    for key in ("sampling_hz", "timer_resolution_ns", "deadline_ms"):
        if key in context and not finite(context[key], positive=True): errors.append("invalid_" + key)
    if code.startswith("C10"):
        if not is_hash(context.get("reference_weight_sha256")) or not finite(context.get("denominator"), positive=True, integer=True) or not finite(context.get("tolerance")):
            errors.append("invalid_numerical_reference_or_population")
    summary = data.get("summary", data.get("metrics", {}))
    for prefix in ("kernel_latency", "end_to_end_latency"):
        quantiles = [summary.get(prefix + suffix) for suffix in ("_p50_ms", "_p95_ms", "_p99_ms")]
        if all(finite(q) for q in quantiles) and quantiles != sorted(quantiles):
            errors.append("invalid_percentile_order")
    if code.startswith("C09") and not errors:
        errors.extend(coverage_trace_issues(source, data, context, policy_at(root)))
    if code == "C07":
        if data.get("runtime_protocol_id") != RUNTIME_PROTOCOL:
            errors.append("incompatible_runtime_protocol")
        if machine.get("batch_size") != 1 or machine.get("timer") != "monotonic_wall":
            errors.append("invalid_runtime_execution_mode")
        if not machine.get("synchronization") or machine.get("state_reset_per_sequence") is not True:
            errors.append("missing_synchronization_or_reset")
        warmup = machine.get("warmup", {})
        if warmup.get("input_role") not in {"synthetic", "train_dev"} or not finite(warmup.get("iterations"), positive=True, integer=True):
            errors.append("invalid_warmup")
        if machine.get("repeats") != 5:
            errors.append("runtime_requires_five_prespecified_repeats")
        if not errors:
            errors.extend(runtime_trace_issues(source, data, policy_at(root)))
    return sorted(set(errors))


def coverage_trace_issues(source, data, context, policy):
    try:
        artifact = context["coverage_counts_file"]
        if artifact not in {a["path"] for a in data["provenance"]["artifacts"]}:
            return ["unhashed_coverage_counts"]
        rows = read_json(artifact_path(source.parent, artifact))
        if sorted(row["sequence_id"] for row in rows) != sorted(data["comparison"]["sequence_ids"]):
            return ["coverage_sequence_population_mismatch"]
        fields = ("input_events", "processed_events", "tail_events", "dropped_events", "output_events", "max_backlog_events")
        for row in rows:
            if any(not finite(row[k], integer=True) for k in fields): return ["invalid_coverage_counts"]
            if row["input_events"] != policy["sequence_files"][data["split"]][row["sequence_id"]]["events"]:
                return ["coverage_input_population_mismatch"]
            if row["input_events"] != sum(row[k] for k in fields[1:4]) or row["output_events"] > row["processed_events"]:
                return ["coverage_partition_mismatch"]
        total = sum(row["input_events"] for row in rows)
        expected = {"processing_coverage": sum(r["processed_events"] for r in rows) / total,
                    "output_coverage": sum(r["output_events"] for r in rows) / total,
                    "dropped_event_rate": sum(r["dropped_events"] for r in rows) / total,
                    "max_backlog_events": max(r["max_backlog_events"] for r in rows)}
        summary = data.get("summary", data.get("metrics", {}))
        for key, value in expected.items():
            if not finite(summary.get(key)) or not math.isclose(summary[key], value, rel_tol=1e-12, abs_tol=1e-12):
                return ["coverage_summary_mismatch"]
        return []
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OSError):
        return ["invalid_coverage_trace_schema"]


def runtime_trace_issues(source, data, policy):
    """Check population, repeat coverage and the published arithmetic mean."""
    try:
        trace = artifact_path(source.parent, data["metric_context"]["C07"]["raw_artifact"])
        with trace.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        names = data["comparison"]["sequence_ids"]
        expected = {(name, repeat) for name in names for repeat in range(1, 6)}
        seen, values = set(), []
        for row in rows:
            key = (row["sequence_id"], int(row["repeat"]))
            value = float(row["runtime_ms"])
            if key in seen or key not in expected or not finite(value, positive=True):
                return ["invalid_runtime_trace_population_or_value"]
            seen.add(key); values.append(value)
            counts = [int(row[k]) for k in ("input_events", "processed_events", "tail_events", "dropped_events", "output_events")]
            if min(counts) < 0 or counts[0] != sum(counts[1:4]) or counts[4] > counts[1]:
                return ["invalid_runtime_event_coverage"]
            if counts[0] != policy["sequence_files"][data["split"]][key[0]]["events"]:
                return ["runtime_input_population_mismatch"]
        if seen != expected:
            return ["incomplete_runtime_trace"]
        summary = data.get("summary", data.get("metrics", {}))
        reported = summary.get("runtime_ms_per_sequence", summary.get("C07"))
        if not math.isclose(sum(values) / len(values), reported, rel_tol=1e-9, abs_tol=1e-9):
            return ["runtime_mean_does_not_match_trace"]
        ordered = sorted(values)
        def percentile(q):
            location = (len(ordered)-1)*q
            low, high = math.floor(location), math.ceil(location)
            return ordered[low] + (ordered[high]-ordered[low])*(location-low)
        average = sum(values)/len(values)
        statistics = {"runtime_p50_ms": percentile(.5), "runtime_p95_ms": percentile(.95),
                      "runtime_std_ms": math.sqrt(sum((v-average)**2 for v in values)/len(values))}
        if any(not finite(summary.get(key)) or not math.isclose(summary[key], value, rel_tol=1e-9, abs_tol=1e-9) for key, value in statistics.items()):
            return ["runtime_distribution_missing_or_inconsistent"]
        return []
    except (KeyError, ValueError, TypeError, OSError):
        return ["invalid_runtime_trace_schema"]
