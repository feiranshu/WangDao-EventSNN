"""Regression cases for incompatible protocols and false completeness."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

IMPLEMENTATION = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(IMPLEMENTATION / "software"))
sys.path.insert(0, str(IMPLEMENTATION / "software/scripts"))
from wangdao.localization_v2 import localization_counts, summarize_counts
from evaluation_contract import (DEPLOYMENT_PROTOCOL, FULL_SCOPE, PREFIX_SCOPE,
    RUNTIME_PROTOCOL, SCHEMA, content_sha256, sha256, state_inventory_issues, verify_record)
from rebuild_evaluation_tables import (build_deployment, load_complexity_values,
    read_json, row_is_complete)
from import_ev_spsegnet_results import preflight_records
from recompute_localization_v2 import recount


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class LocalizationV2Tests(unittest.TestCase):
    def count(self, times, labels, probabilities, end=200):
        y = np.asarray(labels)
        return localization_counts(np.asarray(times, dtype=float), np.zeros(len(y)),
                                   np.zeros(len(y)), y, y, np.asarray(probabilities),
                                   .5, start_ms=0., end_ms=float(end))

    def test_boundaries_are_counted_once_and_misses_are_not_perfect(self):
        row = self.count([0, 50, 100, 150, 199], [1, 1, 1, 1, 0], [0]*5)
        result = summarize_counts([row])
        self.assertEqual((result["targets"], result["pd"], result["frame_denominator"]), (4, 0, 4))

    def test_fractional_timestamp_and_end_exclusion(self):
        row = self.count([0, 50.25, 200], [0, 1, 1], [0, 1, 1])
        self.assertEqual((row["targets"], row["correct_targets"], row["excluded_events"]), (1, 1, 1))

    def test_repeated_false_pixel_does_not_overflow(self):
        row = self.count([25]*256, [0]*256, [1]*256, end=100)
        self.assertEqual(row["false_components"], 1)
        self.assertEqual(summarize_counts([row])["fa"], 1/(2*346*260))

    def test_empty_windows_count_and_undefined_pd_is_null(self):
        result = summarize_counts([self.count([], [], [])])
        self.assertIsNone(result["pd"])
        self.assertEqual((result["fa"], result["frame_denominator"]), (0., 4))

    def test_pooled_counts_are_not_mean_of_sequence_rates(self):
        a = self.count([25], [0], [1], end=100)
        b = self.count([25], [0], [0], end=200)
        self.assertEqual(summarize_counts([a, b])["fa"], 1/(6*346*260))

    def test_invalid_exposure_and_nonfinite_scores(self):
        with self.assertRaises(ValueError): self.count([0], [1], [1], end=99)
        with self.assertRaises(ValueError): self.count([0], [1], [float("nan")])


class RecordContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="wangdao_eval_test_", dir=IMPLEMENTATION.parents[2])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "5_model_evaluation"
        self.root.mkdir()
        self.model = dict(evaluation_id="fixture_board", model_name="Fixture", evaluation_track="board",
                          model_role="formal", numeric_path="exact_int8", weight_sha256="a"*64,
                          platform_id="fixture_device", display_order="1")
        with (self.root / "model_registry.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.model)); writer.writeheader(); writer.writerow(self.model)
        self.policy = {"index_sha256": "b"*64,
                       "accepted_protocols": {"deployment": [DEPLOYMENT_PROTOCOL],
                                              "localization": ["evuav-localization-half-open-v2"]},
                       "sequence_ids": {"test": ["test_000.npz"], "validation": ["val_000.npz"]},
                       "sequence_files": {"test": {"test_000.npz": {"events": 4, "sha256": "f"*64}}}}
        dump(self.root / "comparison_contract.json", self.policy)
        self.source = self.root / "C_deployment_efficiency_and_hardware_cost/02_model_records/formal/fixture_board/test/runtime.json"
        self.source.parent.mkdir(parents=True)
        self.trace = self.source.with_name("runtime.csv")
        self.trace.write_text("sequence_id,repeat,runtime_ms,input_events,processed_events,tail_events,dropped_events,output_events\n" +
                              "".join(f"test_000.npz,{i},{i},4,3,1,0,3\n" for i in range(1, 6)), encoding="utf-8")
        self.data = {"record_schema": SCHEMA, "evaluation_id": self.model["evaluation_id"],
                     "platform_id": self.model["platform_id"], "split": "test", "protocol_id": DEPLOYMENT_PROTOCOL,
                     "runtime_protocol_id": RUNTIME_PROTOCOL, "evidence_type": "board_measured",
                     "comparison": {"scope_id": FULL_SCOPE, "causality": "causal", "numeric_path": "exact_int8",
                                    "threshold": .35, "sequence_ids": ["test_000.npz"]},
                     "provenance": {"weight_sha256": "a"*64, "config_sha256": "c"*64, "index_sha256": "b"*64,
                                    "artifacts": [{"path": "runtime.csv", "sha256": sha256(self.trace)}]},
                     "measurement": {"host_model": "test-fixture", "backend": "fixture", "precision": "int8",
                                     "software_versions": {"fixture": "1"}, "threads": 1, "batch_size": 1,
                                     "transport": "MMIO", "bitstream_sha256": "d"*64, "clock_mhz": 100,
                                     "timer": "monotonic_wall", "synchronization": "fixture_wait",
                                     "state_reset_per_sequence": True, "repeats": 5,
                                     "warmup": {"input_role": "synthetic", "iterations": 2}},
                     "metric_context": {"C07": {"definition_id": "C07-v2", "raw_artifact": "runtime.csv"}},
                     "summary": {"runtime_ms_per_sequence": 3.0, "runtime_p50_ms": 3.0,
                                 "runtime_p95_ms": 4.8, "runtime_std_ms": 2**.5}}

    def rebuild(self, data=None):
        dump(self.source, self.data if data is None else data)
        build_deployment(self.root)
        path = self.root / "C_deployment_efficiency_and_hardware_cost/01_metric_comparisons/C07_runtime/test.csv"
        with path.open(encoding="utf-8-sig", newline="") as f: row = next(csv.DictReader(f))
        return row, path

    def test_valid_runtime_passes_and_remains_source_backed(self):
        row, path = self.rebuild()
        self.assertEqual(row["status"], "complete")
        self.assertTrue(row_is_complete(path, self.model["evaluation_id"]))
        self.data["summary"]["runtime_ms_per_sequence"] = 4
        dump(self.source, self.data)
        self.assertFalse(row_is_complete(path, self.model["evaluation_id"]))

    def test_bare_value_does_not_become_measured(self):
        row, _ = self.rebuild({"summary": {"runtime_ms_per_sequence": 1.}})
        self.assertEqual(row["status"], "unverified")
        self.assertEqual(row["evidence_type"], "unverified")
        self.assertEqual(row["protocol_id"], "")

    def test_invalid_numeric_values_are_rejected(self):
        for value in [-1, 0, True, float("nan"), float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.rebuild({"summary": {"runtime_ms_per_sequence": value}})
        for field, value in [("kernel_latency_mean_ms", -1), ("timeout_rate", 2)]:
            dump(self.source.with_name("summary.json"), {"summary": {field: value}})
            with self.assertRaises(ValueError): build_deployment(self.root)
            self.source.with_name("summary.json").unlink()

    def test_bad_protocol_population_platform_and_mean_are_unverified(self):
        mutations = [lambda d: d.update(runtime_protocol_id="full-sequence-runtime-v1"),
                     lambda d: d["comparison"].update(scope_id=PREFIX_SCOPE),
                     lambda d: d["comparison"].update(sequence_ids=["other.npz"]),
                     lambda d: d.update(platform_id="another_machine"),
                     lambda d: d["summary"].update(runtime_ms_per_sequence=2),
                     lambda d: d["measurement"].update(repeats=1),
                     lambda d: d["provenance"].update(artifacts=[])]
        for mutate in mutations:
            candidate = copy.deepcopy(self.data); mutate(candidate)
            row, path = self.rebuild(candidate)
            self.assertEqual(row["status"], "unverified")
            self.assertFalse(row_is_complete(path, self.model["evaluation_id"]))

    def test_missing_repeat_and_hash_drift_are_unverified(self):
        self.trace.write_text("\n".join(self.trace.read_text().splitlines()[:-1])+"\n", encoding="utf-8")
        row, _ = self.rebuild()
        self.assertIn("evidence_hash_mismatch", row["verification_status"])
        self.data["provenance"]["artifacts"][0]["sha256"] = sha256(self.trace)
        row, _ = self.rebuild()
        self.assertIn("incomplete_runtime_trace", row["verification_status"])

    def test_missing_ac_is_not_filled_as_zero(self):
        category = self.root / "B_model_complexity"
        source = category / "02_model_records/formal/fixture_board/test/summary.json"
        dump(source, {"protocol": "fixture", "summary": {"sequences": 1, "effective_macs_per_sequence": 12}})
        values, _, _ = load_complexity_values(category, self.model, "test")
        self.assertNotIn("effective_acs_per_sequence", values)
        self.assertNotIn("effective_ops_per_sequence", values)

    def test_state_inventory_rejects_component_only_or_wrong_totals(self):
        data = {"summary": {"persistent_state_bytes": 4}, "state_inventory": [
            {"name": "history", "shape": [4], "dtype": "uint8", "bytes": 4,
             "capacity": 4, "lifetime": "between_windows"}]}
        self.assertFalse(state_inventory_issues(data))
        data["summary"]["persistent_state_bytes"] = 8
        self.assertTrue(state_inventory_issues(data))
        self.assertTrue(state_inventory_issues({"summary": {"persistent_state_bytes": 4}}))

    def test_coverage_uses_frozen_input_count_and_pooled_denominators(self):
        source = self.source.with_name("summary.json")
        counts = source.with_name("coverage.json")
        trace = [{"sequence_id": "test_000.npz", "input_events": 4, "processed_events": 3,
                  "tail_events": 1, "dropped_events": 0, "output_events": 3, "max_backlog_events": 2}]
        dump(counts, trace)
        data = copy.deepcopy(self.data)
        data["comparison"]["scope_id"] = PREFIX_SCOPE
        data["provenance"]["artifacts"] = [{"path": counts.name, "sha256": sha256(counts)}]
        data["metric_context"] = {code: {"definition_id": code+"-v2", "raw_artifact": counts.name,
            "coverage_counts_file": counts.name, "queue_observation_method": "fixture_every_operation"}
            for code in ("C09a", "C09b", "C09c", "C09d")}
        data["summary"] = {"processing_coverage": .75, "output_coverage": .75,
                           "dropped_event_rate": 0., "max_backlog_events": 2}
        dump(source, data); build_deployment(self.root)
        path = self.root / "C_deployment_efficiency_and_hardware_cost/01_metric_comparisons/C09_coverage_and_backlog/C09a_processing_coverage/test.csv"
        self.assertTrue(row_is_complete(path, self.model["evaluation_id"]))
        # A smaller claimed input population must not manufacture 100% coverage.
        trace[0].update(input_events=3, tail_events=0); dump(counts, trace)
        data["provenance"]["artifacts"][0]["sha256"] = sha256(counts)
        data["summary"].update(processing_coverage=1., output_coverage=1.)
        dump(source, data); build_deployment(self.root)
        self.assertFalse(row_is_complete(path, self.model["evaluation_id"]))

    def test_old_server_return_fails_before_import(self):
        source = Path(self.temp.name) / "old_server"
        dump(source / "Test/event_level/summary.json", {"evaluation": {"metric_protocol": "old-full-sequence"}})
        with self.assertRaisesRegex(RuntimeError, "no files imported"):
            preflight_records(source, self.root, "a"*64)

    def test_legacy_snapshot_cannot_be_reused_after_value_edit(self):
        source = self.root / "historical.json"; dump(source, {"value": 5})
        self.policy["reviewed_legacy_sources"] = {"historical.json": {"content_sha256": content_sha256(source),
            "evaluation_id": self.model["evaluation_id"], "weight_sha256": "a"*64,
            "split": "test", "family": "complexity", "scope_id": PREFIX_SCOPE}}
        dump(self.root / "comparison_contract.json", self.policy)
        self.assertFalse(verify_record(self.root, source, self.model, "test", "complexity"))
        dump(source, {"value": 6})
        self.assertTrue(verify_record(self.root, source, self.model, "test", "complexity"))

    def test_recount_cli_uses_frozen_scores_and_preserves_legacy(self):
        events = np.zeros(4, dtype=[("t", "f8"), ("x", "i4"), ("y", "i4"), ("label", "i4"), ("name", "i4")])
        events["t"] = [0., 50.25, 150., 201.]; events["label"] = [1, 1, 0, 0]; events["name"] = events["label"]
        base = Path(self.temp.name); event_file = base / "events.npz"; scores = base / "scores.npz"
        np.savez(event_file, ev=events); np.savez(scores, probabilities=np.array([1., 1., 1.]))
        self.policy["sequence_files"]["test"]["test_000.npz"]["sha256"] = sha256(event_file)
        dump(self.root / "comparison_contract.json", self.policy)
        record = copy.deepcopy(self.data); record["comparison"]["scope_id"] = PREFIX_SCOPE
        manifest = {"record": record, "sequences": [{"sequence_id": "test_000.npz", "events_file": event_file.name,
                    "probabilities_file": scores.name, "probabilities_sha256": sha256(scores)}]}
        manifest_file = base / "manifest.json"; dump(manifest_file, manifest)
        output = base / "new/localization_v2.json"
        recount(manifest_file, self.root, output)
        result = read_json(output)
        self.assertEqual(result["summary"]["pd"], 1.)
        self.assertEqual(result["summary"]["fa"], 1/(4*346*260))
        self.assertEqual(result["summary"]["excluded_events"], 1)
        with self.assertRaises(ValueError): recount(manifest_file, self.root, output)


if __name__ == "__main__": unittest.main()
