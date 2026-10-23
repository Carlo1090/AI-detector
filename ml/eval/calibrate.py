"""Fit the document and sentence-window scoring models on labelled text and save them for the detector.

    python -m ml.eval.calibrate [--model Qwen/Qwen2.5-0.5B] [--performer Qwen/Qwen2.5-0.5B-Instruct|none]
                                [--hidden-layer 16] [--per-class 900]

Training mixes student essays (incl. English learners), news and fiction, human and AI. With --hidden-layer,
a logistic probe is first trained on the language model's pooled internal state; its out-of-fold score then
joins the statistical features. Thresholds are chosen on out-of-fold predictions so that at most --doc-fpr of
human documents (and --window-fpr of human sentence windows) would be flagged. Cross-validation folds keep
all windows of a document together, so neighbouring sentences can't leak between train and test.
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from ml.detector.detector import Detector
from ml.detector.features import DOC_FEATURES, WINDOW_FEATURES
from ml.detector.lm import DEFAULT_MODEL, DEFAULT_PERFORMER, TokenScorer
from ml.detector.scoring import Calibration, LinearModel, VectorProbe, calibration_path

from .data import extract_features, training_mix


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, scale = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    scale[scale == 0] = 1.0
    return np.nan_to_num((X - mean) / scale, nan=0.0), mean, scale


# Sensitivity levels as multiples of the target false-positive rate (strict = the target itself).
SENSITIVITY = {"strict": 1, "balanced": 2.5, "sensitive": 5}


def robust_threshold(scores: np.ndarray, sources: np.ndarray, target_fpr: float, min_n: int = 40) -> float:
    """Lowest threshold that keeps the false-positive rate at or below target for *every* kind of human
    writing with enough examples, not just on average (formal and non-native writing are flagged most).
    A small group may keep up to two false positives, so one odd text can't set the threshold alone."""
    cuts = [np.quantile(scores, 1 - target_fpr)]
    for source in np.unique(sources):
        s = scores[sources == source]
        if len(s) >= min_n:
            cuts.append(np.quantile(s, 1 - max(target_fpr, 2 / len(s))))
    return max(0.5, float(max(cuts)))


def fit(rows: list[dict], features: tuple[str, ...], target_fpr: float, groups=None) -> tuple[LinearModel, dict]:
    X = np.array([[r.get(f, np.nan) for f in features] for r in rows], np.float64)
    keep = ~np.all(np.isnan(X), axis=0)  # e.g. Binoculars without a performer model
    features, X = tuple(f for f, k in zip(features, keep) if k), X[:, keep]
    y = np.array([r["label"] for r in rows])
    sources = np.array([r["source"] for r in rows])
    Z, mean, scale = _standardize(X)

    clf = LogisticRegression(C=1.0, max_iter=2000)
    oof = cross_val_predict(clf, Z, y, cv=GroupKFold(5), groups=groups if groups is not None else np.arange(len(y)),
                            method="predict_proba")[:, 1]
    threshold = robust_threshold(oof[y == 0], sources[y == 0], target_fpr)
    levels = {name: (threshold if mult == 1 else robust_threshold(oof[y == 0], sources[y == 0], target_fpr * mult))
              for name, mult in SENSITIVITY.items()}
    clf.fit(Z, y)

    human_fpr = {s: round(float((oof[(y == 0) & (sources == s)] >= threshold).mean()), 4)
                 for s in np.unique(sources[y == 0])}
    metrics = {
        "n": int(len(y)), "auroc": round(float(roc_auc_score(y, oof)), 4),
        "threshold": round(threshold, 4),
        "tpr_at_threshold": round(float((oof[y == 1] >= threshold).mean()), 4),
        "fpr_at_threshold": round(float((oof[y == 0] >= threshold).mean()), 4),
        "fpr_by_human_source": human_fpr,
        "levels": {name: {"threshold": round(t, 4), "tpr": round(float((oof[y == 1] >= t).mean()), 4),
                          "fpr": round(float((oof[y == 0] >= t).mean()), 4)} for name, t in levels.items()},
    }
    model = LinearModel(list(features), mean.tolist(), scale.tolist(), clf.coef_[0].tolist(),
                        float(clf.intercept_[0]), threshold, {k: round(v, 4) for k, v in levels.items()})
    return model, metrics


def fit_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float) -> tuple[VectorProbe, np.ndarray, float]:
    """Probe on pooled hidden states. Returns the probe, its out-of-fold log-odds, and out-of-fold AUROC."""
    Z, mean, scale = _standardize(X.astype(np.float64))
    clf = LogisticRegression(C=C, max_iter=3000)
    oof = cross_val_predict(clf, Z, y, cv=GroupKFold(5), groups=groups, method="decision_function")
    clf.fit(Z, y)
    probe = VectorProbe(mean.tolist(), scale.tolist(), clf.coef_[0].tolist(), float(clf.intercept_[0]))
    return probe, oof, float(roc_auc_score(y, oof))


def scorer_from_args(args, hidden_layer: int | None = None) -> TokenScorer:
    performer = None if args.performer.lower() == "none" else args.performer
    return TokenScorer(args.model, performer, hidden_layer=hidden_layer)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--performer", default=DEFAULT_PERFORMER)
    ap.add_argument("--hidden-layer", type=int, default=None, help="train a probe on this layer's pooled state")
    ap.add_argument("--probe-c", type=float, default=0.05)
    ap.add_argument("--per-class", type=int, default=900)
    ap.add_argument("--doc-fpr", type=float, default=0.01)
    ap.add_argument("--window-fpr", type=float, default=0.03)
    args = ap.parse_args()

    detector = Detector(scorer_from_args(args, args.hidden_layer), calibration=None)
    examples = training_mix(args.per_class)
    rows = extract_features(detector, examples, "train-mix")

    doc_rows = [{**r["document"], "label": r["label"], "source": r["group"]} for r in rows]
    window_rows = [{**w, "label": r["label"], "source": r["group"]} for r in rows for w in r["windows"]]
    window_groups = np.array([i for i, r in enumerate(rows) for _ in r["windows"]])
    meta = {"trained_on": f"train-mix ({len(examples)} texts: {', '.join(sorted({e.group for e in examples}))})",
            "date": date.today().isoformat()}

    doc_probe = window_probe = None
    if args.hidden_layer is not None:
        y_doc = np.array([r["label"] for r in rows])
        doc_probe, oof, auc = fit_probe(np.stack([r["doc_vec"] for r in rows]), y_doc, np.arange(len(rows)), args.probe_c)
        for row, logit in zip(doc_rows, oof):
            row["probe"] = float(logit)
        win_vecs = np.concatenate([r["window_vecs"] for r in rows]).astype(np.float32)
        ok = ~np.isnan(win_vecs).any(axis=1)
        y_win = np.array([r["label"] for r in window_rows])
        window_probe, oof_w, auc_w = fit_probe(win_vecs[ok], y_win[ok], window_groups[ok], args.probe_c)
        for row, logit in zip((r for r, k in zip(window_rows, ok) if k), oof_w):
            row["probe"] = float(logit)
        meta["probe"] = {"layer": args.hidden_layer, "C": args.probe_c,
                         "doc_auroc_oof": round(auc, 4), "window_auroc_oof": round(auc_w, 4)}
        print(f"probe (out-of-fold): document AUROC {auc:.4f}, window AUROC {auc_w:.4f}")

    document, doc_metrics = fit(doc_rows, DOC_FEATURES, args.doc_fpr)
    window, window_metrics = fit(window_rows, WINDOW_FEATURES, args.window_fpr, groups=window_groups)
    meta.update({"document_metrics": doc_metrics, "window_metrics": window_metrics})

    path = calibration_path(detector.scorer.name)
    Calibration(detector.scorer.name, document, window, meta, args.hidden_layer, doc_probe, window_probe).save(path)

    print(f"\nsaved {path}")
    print(f"document  (out-of-fold): {doc_metrics}")
    print(f"window    (out-of-fold): {window_metrics}")
    print("document weights:", dict(zip(document.features, np.round(document.coef, 3))))
    print("window   weights:", dict(zip(window.features, np.round(window.coef, 3))))


if __name__ == "__main__":
    main()
