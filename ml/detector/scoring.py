"""Calibrated logistic models that turn features into probabilities.

Weights are fitted by `python -m ml.eval.calibrate` and stored per language model in `calibrations/`.
Runtime needs only numpy.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

CALIBRATION_DIR = Path(__file__).parent / "calibrations"


def calibration_path(model_name: str) -> Path:
    return CALIBRATION_DIR / (model_name.replace("/", "__") + ".json")


@dataclass
class LinearModel:
    features: list[str]
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    threshold: float  # probability at or above which we call it AI

    def _z(self, feats: dict[str, float]) -> np.ndarray:
        x = np.array([feats.get(f, math.nan) for f in self.features], np.float64)
        z = (x - np.array(self.mean)) / np.array(self.scale)
        return np.nan_to_num(z, nan=0.0)  # a missing feature counts as perfectly average

    def contributions(self, feats: dict[str, float]) -> dict[str, float]:
        """Each feature's push on the log-odds: positive leans AI, negative leans human."""
        return dict(zip(self.features, (self._z(feats) * np.array(self.coef)).tolist()))

    def probability(self, feats: dict[str, float]) -> float:
        logit = float(self._z(feats) @ np.array(self.coef)) + self.intercept
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))


@dataclass
class VectorProbe:
    """Logistic probe on a pooled hidden-state vector; its log-odds become the 'probe' feature."""

    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float

    def logit(self, vec: np.ndarray | None) -> float:
        if vec is None:
            return math.nan
        z = (np.asarray(vec, np.float64) - np.array(self.mean)) / np.array(self.scale)
        return float(z @ np.array(self.coef)) + self.intercept


@dataclass
class Calibration:
    model_name: str
    document: LinearModel
    window: LinearModel
    meta: dict = field(default_factory=dict)
    hidden_layer: int | None = None
    doc_probe: VectorProbe | None = None
    window_probe: VectorProbe | None = None

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        probe = lambda key: VectorProbe(**raw[key]) if raw.get(key) else None  # noqa: E731
        return cls(raw["model_name"], LinearModel(**raw["document"]), LinearModel(**raw["window"]), raw.get("meta", {}),
                   raw.get("hidden_layer"), probe("doc_probe"), probe("window_probe"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
