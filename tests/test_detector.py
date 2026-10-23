"""Detector and API behaviour with a stand-in language model, so tests run without downloading weights."""

import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml.detector.detector import Detector
from ml.detector.lm import TokenStats
from ml.detector.scoring import Calibration, LinearModel


class FakeScorer:
    """Every word is the model's top guess, so everything looks maximally predictable."""

    model_name = name = "fake/model"
    device = "cpu"

    def decode(self, token_id):
        return f"tok{token_id}"

    def score(self, text):
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        n = len(spans)
        return TokenStats(
            starts=np.array([s for s, _ in spans], np.int64), ends=np.array([e for _, e in spans], np.int64),
            logprob=np.full(n, -0.5, np.float32), rank=np.ones(n, np.int64),
            expected=np.full(n, -2.0, np.float32), variance=np.full(n, 1.0, np.float32),
            alt_ids=np.zeros((n, 5), np.int64), alt_probs=np.full((n, 5), 0.2, np.float32),
        )


def _model(features, coef):
    return LinearModel(list(features), [0.0] * len(features), [1.0] * len(features), coef, 0.0, 0.5)


@pytest.fixture
def detector():
    cal = Calibration(
        "fake/model",
        document=_model(["curvature"], [1.0]),
        window=_model(["curvature"], [1.0]),
    )
    return Detector(FakeScorer(), cal)


LONG = " ".join(f"Sentence number {i} keeps going with plain words here." for i in range(25))


def test_predictable_long_text_is_flagged(detector):
    r = detector.analyze(LONG)
    assert not r["too_short"]
    assert r["verdict"] == "likely_ai"
    assert r["ai_share"] == 1.0
    assert all(s["flagged"] for s in r["sentences"])
    assert len(r["tokens"]) == len(LONG.split())
    assert r["tokens"][0]["alts"][0] == ["tok0", 0.2]


def test_short_text_gets_no_verdict_or_flags(detector):
    r = detector.analyze("Just a short note. Nothing more.")
    assert r["too_short"] and r["verdict"] == "too_short"
    assert r["ai_share"] is None
    assert not any(s["flagged"] for s in r["sentences"])


def test_signals_are_sorted_by_influence(detector):
    pushes = [abs(s["contribution"]) for s in detector.analyze(LONG)["signals"]]
    assert pushes == sorted(pushes, reverse=True)


def test_uncalibrated_detector_still_returns_word_stats():
    r = Detector(FakeScorer(), None).analyze(LONG)
    assert r["verdict"] == "uncalibrated" and r["document_probability"] is None
    assert r["tokens"]


def test_api_detect_and_health(detector, monkeypatch):
    from api import main

    monkeypatch.setattr(main, "_load_detector", lambda: None)
    monkeypatch.setattr(main.state, "detector", detector)
    with TestClient(main.app) as client:
        assert client.get("/api/health").json()["status"] == "ready"
        r = client.post("/api/detect", json={"text": LONG})
        assert r.status_code == 200 and r.json()["verdict"] == "likely_ai"
        assert client.post("/api/detect", json={"text": "   "}).status_code == 422
        assert client.get("/").status_code == 200
