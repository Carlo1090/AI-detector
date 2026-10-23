"""Compare calibration variants on cached features (no language model needed, runs in a minute).

    python -m ml.eval.ablate

Each variant fits the document model on a subset of the cached training texts (all texts ever scored for the
training mix) and is scored on the cached held-out suites at its own robust threshold. Used to find which
training sources help or hurt; the chosen setup is then saved by ml.eval.calibrate.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from ml.detector.features import DOC_FEATURES

from .calibrate import fit, fit_probe
from .data import CACHE

PREFIX = "Qwen__Qwen2.5-0.5B+Qwen__Qwen2.5-0.5B-Instruct"
SUITES = ["gb-essay-test", "students-test", "hc3-test", "raid-test", "textsight2026"]


def load(name: str, layer: int = 16) -> list[dict]:
    rows = [json.loads(l) for l in (CACHE / f"{PREFIX}__{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    vec_dir = CACHE / f"{PREFIX}__L{layer}__{name}"
    out = []
    for r in rows:
        path = vec_dir / f"{r['id']}.npz"
        if path.exists():
            r["doc_vec"] = np.load(path)["doc"].astype(np.float32)
            out.append(r)
    return out


def run(name, train, suites, features=DOC_FEATURES, probe_c=0.05):
    rows = [{**r["document"], "label": r["label"], "source": r["group"]} for r in train]
    probe = None
    if probe_c:
        y = np.array([r["label"] for r in train])
        probe, oof, _ = fit_probe(np.stack([r["doc_vec"] for r in train]), y, np.arange(len(train)), probe_c)
        for row, logit in zip(rows, oof):
            row["probe"] = float(logit)
    model, _ = fit(rows, features, 0.01)
    cells = [name, str(len(train))]
    for test in suites.values():
        feats = [{**r["document"], **({"probe": probe.logit(r["doc_vec"])} if probe else {})} for r in test]
        p = np.array([model.probability(f) for f in feats])
        y = np.array([r["label"] for r in test])
        cells.append(f"{roc_auc_score(y, p):.3f} · {100 * (p[y == 1] >= model.threshold).mean():.0f}/"
                     f"{100 * (p[y == 0] >= model.threshold).mean():.1f}")
    print("| " + " | ".join(cells) + " |", flush=True)


def main() -> None:
    pool = load("train-mix")
    suites = {s: load(s) for s in SUITES}
    fam = lambda r: r["group"].split(":")[0]  # noqa: E731
    old = {"essay", "reuter", "wp", "persuade", "persuade-ell", "daigt", "hc3-wiki_csai", "hc3-medicine",
           "hc3-finance", "hc3-open_qa"}
    print("AUROC · AI caught % / humans wrongly flagged % per suite\n")
    print("| Variant | n | " + " | ".join(SUITES) + " |")
    print("|" + "---|" * (len(SUITES) + 2))
    variants = {
        "v2-like (old sources only)": lambda r: fam(r) in old,
        "+ sonnet35 + arxiv": lambda r: fam(r) in old | {"sonnet35-orca", "sonnet35-science", "arxiv"},
        "+ sonnet35 + arxiv + gen-abstracts": lambda r: fam(r) in old | {"sonnet35-orca", "sonnet35-science",
                                                                         "arxiv", "gen-arxiv"},
        "+ RAID (everything)": lambda r: True,
        "everything except gen-abstracts": lambda r: fam(r) != "gen-arxiv",
        "everything except RAID": lambda r: not fam(r).startswith("raid"),
    }
    for name, keep in variants.items():
        run(name, [r for r in pool if keep(r)], suites)
    run("everything, probe C=0.5", pool, suites, probe_c=0.5)
    run("everything, probe C=0.005", pool, suites, probe_c=0.005)
    run("everything, no probe", pool, suites, features=tuple(f for f in DOC_FEATURES if f != "probe"), probe_c=None)


if __name__ == "__main__":
    main()
