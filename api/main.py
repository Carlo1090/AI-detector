"""HTTP API and static web UI.

Models load in a background thread so the page is reachable immediately: first the detector, then the
humanizer's rewriter and meaning checker. The UI polls /api/health to see what is ready.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ml.detector.detector import MAX_CHARS, Detector

log = logging.getLogger("detect-ai")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class _State:
    detector: Detector | None = None
    error: str | None = None
    rewriter = None
    embedder = None
    humanizer_status: str = "waiting"  # waiting | loading | ready | off | error: ...
    lock = threading.Lock()


state = _State()


def _load_detector() -> None:
    try:
        state.detector = Detector.load()
        log.info("Detector ready: %s", state.detector.scorer.name)
    except Exception as exc:  # surfaced to the UI through /api/health
        log.exception("Failed to load detector")
        state.error = f"{type(exc).__name__}: {exc}"
        return
    _load_humanizer()


def _load_humanizer() -> None:
    if os.environ.get("HUMANIZE_BACKEND", "auto").lower() == "off":
        state.humanizer_status = "off"
        return
    state.humanizer_status = "loading"
    try:
        from ml.humanizer.meaning import Embedder
        from ml.humanizer.rewriter import load_rewriter

        state.embedder = Embedder()
        state.rewriter = load_rewriter()
        state.humanizer_status = "ready"
        log.info("Humanizer ready: %s", state.rewriter.name)
    except Exception as exc:
        log.exception("Failed to load humanizer")
        state.humanizer_status = f"error: {type(exc).__name__}: {exc}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load_detector, daemon=True).start()
    yield


app = FastAPI(title="DetectAI", lifespan=lifespan)


@app.middleware("http")
async def no_stale_assets(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"  # revalidate the page, CSS and JS after every update
    return response


class DetectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    sensitivity: Literal["strict", "balanced", "sensitive"] = "strict"


class HumanizeRequest(DetectRequest):
    strength: Literal["light", "balanced", "strong"] = "balanced"
    tone: Literal["natural", "academic"] = "natural"


class RewriteRequest(DetectRequest):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    tone: Literal["natural", "academic"] = "natural"


@app.get("/api/health")
def health():
    if state.error:
        return {"status": "error", "error": state.error}
    if state.detector is None:
        return {"status": "loading"}
    d = state.detector
    return {"status": "ready", "model": d.scorer.name, "device": d.scorer.device,
            "calibrated": d.calibration is not None, "humanizer": state.humanizer_status,
            "rewriter": getattr(state.rewriter, "name", None)}


def _require_detector(text: str) -> Detector:
    if state.detector is None:
        raise HTTPException(503, state.error or "The model is still loading. Try again in a moment.")
    if not text.strip():
        raise HTTPException(422, "Text is empty.")
    return state.detector


@app.post("/api/detect")
def detect(req: DetectRequest):
    detector = _require_detector(req.text)
    with state.lock:  # one job at a time; the GPU has room for one
        return detector.analyze(req.text, req.sensitivity)


@app.post("/api/humanize")
def humanize(req: HumanizeRequest):
    from ml.humanizer.humanizer import Humanizer

    detector = _require_detector(req.text)
    if detector.calibration is None:
        raise HTTPException(409, "The detector isn't calibrated, so rewrites can't be scored.")
    if req.strength != "light" and state.humanizer_status != "ready":
        raise HTTPException(503, f"The rewriting model isn't ready ({state.humanizer_status}). "
                                 "Light mode (word-level fixes only) works now.")
    with state.lock:
        return Humanizer(detector, state.rewriter, state.embedder).humanize(req.text, req.strength, req.tone)


@app.post("/api/rewrite")
def rewrite(req: RewriteRequest):
    """Suggestions for one sentence or passage of the text (the report's 'Suggest rewrites')."""
    from ml.humanizer.humanizer import Humanizer

    detector = _require_detector(req.text)
    if detector.calibration is None:
        raise HTTPException(409, "The detector isn't calibrated, so rewrites can't be scored.")
    if not req.start < req.end <= len(req.text):
        raise HTTPException(422, "The selected span is outside the text.")
    rewriter = state.rewriter if state.humanizer_status == "ready" else None
    with state.lock:
        return Humanizer(detector, rewriter, state.embedder).alternatives(req.text, req.start, req.end, req.tone)


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
