"""External-data integrity, temporal labels, geometry, ECF and restart regression."""
from pathlib import Path
import hashlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

ROOT = Path(__file__).absolute().parents[2]
TEST_ROOT = ROOT / "results/runs/fred_external_v1/test_tmp"
TEST_ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "software"))
from wangdao.fred_adapter_v1.core import (convert_block, frame_members, label_events,
                                         letterbox, parse_annotations, validate_raw)
from wangdao.fred_adapter_v1.download import download_file
from wangdao.fred_adapter_v1.ecf import ECFCodec, RAW_DTYPE, build_codec, read_chunk
from wangdao.fred_adapter_v1.workflow import save_shard


def events(values):
    return np.array(values, dtype=RAW_DTYPE)


class AdapterTests(unittest.TestCase):
    def test_interpolation_source_time_and_half_open_box(self):
        ann = parse_annotations("0: 0,0,2,2,7,Drone\n0.01: 2,0,4,2,7,Drone", [0, 10000])
        raw = events([(1, 0, 1, 5000), (0, 0, 0, 5000), (3, 0, 1, 5000), (2, 1, 1, 10000)])
        label, ids, mask = label_events(raw, ann)
        self.assertEqual(label.tolist(), [1, 0, 0, 1])
        self.assertEqual(ids.tolist(), [1, 0, 0, 1])
        self.assertTrue(mask.all())

    def test_appearance_disappearance_does_not_create_background_truth(self):
        ann = parse_annotations("0.01: 0,0,3,3,0,Drone", [0, 10000, 20000, 30000])
        raw = events([(1, 1, 1, t) for t in [5000, 10000, 15000, 25000, 35000]])
        label, ids, valid = label_events(raw, ann)
        self.assertEqual(valid.tolist(), [False, True, False, True, False])
        self.assertEqual(label.tolist(), [0, 1, 0, 0, 0])
        self.assertEqual(ids.tolist(), [0, 1, 0, 0, 0])

    def test_gap_and_overlap_mask(self):
        text = "\n".join(f"{t}: 0,0,3,3,1,A; 1,1,4,4,2,B" for t in [0, .01, .02, .1])
        ann = parse_annotations(text, [0, 10000, 20000, 100000])
        _, _, valid = label_events(events([(1, 1, 1, 5000), (0, 0, 0, 5000), (0, 0, 1, 50000)]), ann)
        self.assertEqual(valid.tolist(), [False, True, False])

    def test_letterbox_preserves_all_fields_without_time_origin_shift(self):
        raw = events([(0, 0, 0, 4560), (1279, 719, 1, 99999999)])
        ann = parse_annotations("", [0, 100000000])
        output, mask = convert_block(raw, ann, letterbox(1280, 720))
        self.assertEqual(output["x"].tolist(), [0, 345])
        self.assertEqual(output["y"].tolist(), [32, 227])
        np.testing.assert_array_equal(output["t"], [4.56, 99999.999])
        np.testing.assert_array_equal(output["p"], raw["p"])
        self.assertEqual(len(output), len(raw))

    def test_annotation_timestamp_and_identity_checks(self):
        for text in ["0.001: 0,0,3,3,1,A", "0: 0,0,3,3,1,A; 0,0,3,3,1,A"]:
            with self.assertRaises(ValueError): parse_annotations(text, [0, 10000])
        with self.assertRaises(ValueError): frame_members(["0/Event/Frames/foo.png"])

    def test_ordering_across_decoder_chunks(self):
        with self.assertRaises(ValueError): validate_raw(events([(1, 1, 0, 10)]), 1280, 720, 11)
        with self.assertRaises(ValueError): validate_raw(events([(1280, 1, 0, 10)]), 1280, 720)

    def test_shard_resume_does_not_overwrite_and_rejects_changed_events(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            output, mask = convert_block(events([(1, 1, 0, 1000)]), parse_annotations("", [0, 10000]), letterbox(1280, 720))
            row = save_shard(Path(folder), "train_000", 0, [output], [mask])
            target = Path(folder) / row["path"]
            before = target.read_bytes()
            self.assertEqual(row, save_shard(Path(folder), "train_000", 0, [output], [mask]))
            self.assertEqual(before, target.read_bytes())
            output["t"] += 1
            with self.assertRaises(ValueError): save_shard(Path(folder), "train_000", 0, [output], [mask])


class CodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codec = ECFCodec(build_codec(ROOT / "results/runs/fred_external_v1/runtime"))

    def test_official_codec_roundtrip_across_packings_and_timestamps(self):
        rng = np.random.default_rng(123)
        for length in [1, 17, 16384]:
            for mode in ["random", "same_y", "same_y_p", "same_pixel"]:
                data = np.zeros(length, dtype=RAW_DTYPE)
                data["x"] = rng.integers(0, 1280, length)
                data["y"] = rng.integers(0, 720, length)
                data["p"] = rng.integers(0, 2, length)
                data["t"] = 123456789 + np.cumsum(rng.integers(0, 10000, length))
                if mode != "random": data["y"] = 32
                if mode in ["same_y_p", "same_pixel"]: data["p"] = 1
                if mode == "same_pixel": data["x"] = 12
                decoded = self.codec.decode(self.codec.encode(data), length)
                np.testing.assert_array_equal(data, decoded)

    def test_uncompressed_hdf5_short_final_chunk(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            data = events([(x, 1, 1, x * 1000) for x in range(19)])
            with h5py.File(Path(folder) / "test.h5", "w") as handle:
                dataset = handle.create_dataset("CD/events", data=data, chunks=(16,))
                np.testing.assert_array_equal(read_chunk(dataset, 16, self.codec), data[16:])


class DownloadTests(unittest.TestCase):
    def perform(self, status, offset, payload, content_range=None):
        full = b"abcdefgh"
        digest = hashlib.sha256(full).hexdigest()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            target = Path(folder) / "sample.zip"
            target.with_suffix(".zip.part").write_bytes(full[:offset])
            response = io.BytesIO(payload)
            response.status = status
            response.headers = {"Content-Range": content_range} if content_range else {}
            with patch("wangdao.fred_adapter_v1.download.open_url", return_value=response):
                result = download_file("https://example.test/data", target, len(full), digest)
            self.assertEqual(target.read_bytes(), full)
            self.assertEqual(result["sha256"], digest)
            before = target.stat().st_mtime_ns
            self.assertTrue(download_file("unused", target, len(full), digest)["reused"])
            self.assertEqual(before, target.stat().st_mtime_ns)

    def test_resume_with_verified_range(self): self.perform(206, 3, b"defgh", "bytes 3-7/8")
    def test_server_ignoring_range_restarts_partial(self): self.perform(200, 3, b"abcdefgh")
    def test_bad_content_range_rejected(self):
        with self.assertRaises(ValueError): self.perform(206, 3, b"defgh", "bytes 2-7/8")


if __name__ == "__main__":
    unittest.main()
