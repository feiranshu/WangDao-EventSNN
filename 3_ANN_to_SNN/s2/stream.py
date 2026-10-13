"""Protocol-A raw-event adapter using the frozen H2-pro Proposal and C1 features."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch

from core import ResearchANN, ResearchSNN, SNNRunner, class_logits


HERE = Path(__file__).absolute().parent
ROOT = HERE.parent.parent
H2 = ROOT / "1_H2_model_improvement"
STUDY = H2 / "seven_step_improvement_study" / "H2-pro"
SOFTWARE = H2 / "implementation" / "software"
sys.path[:0] = [str(STUDY), str(SOFTWARE)]
from src.stream_state import CausalPipeline, EVENT_DTYPE  # noqa: E402


@dataclass(frozen=True)
class WindowPrediction:
    window: int
    event_indices: np.ndarray
    event_time_ms: np.ndarray
    available_time_ms: np.ndarray
    candidate: np.ndarray
    probability: np.ndarray
    predicted_target: np.ndarray


def load_research_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    kind = checkpoint["model_kind"]
    if kind == "ann":
        model = ResearchANN()
    elif kind == "snn":
        model = ResearchSNN()
    else:
        raise ValueError(f"unknown research model: {kind}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return kind, model.eval()


class ProtocolAPipeline:
    """Closed-window outputs; event-time replay occurs only after publication boundary."""
    def __init__(self, model, kind: str, threshold: float = 0.35):
        if kind not in ("ann", "snn"):
            raise ValueError("kind must be ann or snn")
        self.model = model.eval()
        self.kind = kind
        self.threshold = float(threshold)
        self.front = CausalPipeline(
            SOFTWARE / "weights" / "H2_D1_seed42.npz",
            STUDY / "frozen" / "H2-pro-Causal-ANN-v1" / "teacher_exact_int8.npz",
            variant="C1-proposal-motion", threshold=0.35, initial_refiner=False)
        self.reset(0.0)

    def reset(self, t0: float) -> None:
        self.front.reset(t0)
        self.pending = np.empty(0, dtype=EVENT_DTYPE)
        self.ready: list[WindowPrediction] = []
        self.consumed = 0
        self.runner = SNNRunner(self.model, t0) if self.kind == "snn" else None

    def _score(self, features: np.ndarray, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        if not len(features):
            return np.empty(0, dtype=np.float32)
        if self.kind == "snn":
            with torch.no_grad():
                _, probability = self.runner.run_window(features, x, y, t)
            return probability
        parts = []
        with torch.no_grad():
            for start in range(0, len(features), 4096):
                block = torch.from_numpy(features[start:start+4096].astype(np.float32) / 255.0)
                parts.append(torch.softmax(class_logits(self.model(block)), dim=1)[:, 1].numpy())
        return np.concatenate(parts)

    def _consume(self, windows) -> None:
        for window in windows:
            n = len(window.event_indices)
            if n and not np.array_equal(window.event_indices, np.arange(self.consumed, self.consumed + n)):
                raise AssertionError("frozen front event index discontinuity")
            events, self.pending = self.pending[:n].copy(), self.pending[n:].copy()
            candidate = window.candidate_mask.copy()
            selected = events[candidate]
            score = self._score(window.features, selected["x"], selected["y"], selected["t"])
            probability = np.zeros(n, dtype=np.float32)
            probability[candidate] = score
            self.ready.append(WindowPrediction(
                window.window, window.event_indices.copy(), events["t"].copy(),
                np.full(n, window.right_boundary_ms, dtype=np.float64), candidate,
                probability, probability >= self.threshold))
            self.consumed += n

    def push(self, events: np.ndarray, watermark: float) -> None:
        packet = np.empty(len(events), dtype=EVENT_DTYPE)
        for key in ("x", "y", "t", "p"):
            packet[key] = events[key]
        self.front.push(events, watermark)
        self.pending = np.concatenate((self.pending, packet))
        self._consume(self.front.flush_completed_windows())

    def flush_completed_windows(self) -> list[WindowPrediction]:
        output, self.ready = self.ready, []
        return output

    def finalize(self, watermark: float | None = None) -> tuple[list[WindowPrediction], int]:
        windows, tail = self.front.finalize(watermark)
        self._consume(windows)
        if len(self.pending) != tail:
            raise AssertionError("front and adapter tail counts differ")
        return self.flush_completed_windows(), tail

    def metadata(self) -> dict:
        return {"observation_protocol": "A_closed_100ms_window",
                "state_model": self.kind, "threshold": self.threshold,
                "state_and_refiner_cost": self.runner.cost() if self.runner is not None else None}
