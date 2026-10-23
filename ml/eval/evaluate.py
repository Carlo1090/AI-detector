"""Score held-out data with the saved calibration and report how the detector really does.

    python -m ml.eval.evaluate [--model ...] [--performer ...|none] [--per-class 300]

Reports AUROC, detection and false-positive rates at the detector's own threshold, detection at an
oracle 1% false-positive threshold, and a per-source breakdown. Writes data/reports/eval-<setup>.md.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date

import numpy as np
from sklearn.metrics import roc_auc_score

from ml.detector.detector import Detector
from ml.detector.lm import DEFAULT_MODEL, DEFAULT_PERFORMER
from ml.detector.scoring import Calibration, calibration_path
from ml.detector.text import count_words

from .calibrate import scorer_from_args
from .external import DEFAULT_EXTERNAL
from .data import (ROOT, arxiv_abstracts, balanced, daigt, extract_features, generated, ghostbuster_domain,
                   ghostbuster_essays, hc3, modern_claude, persuade, raid, textsight)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def evaluate(name: str, examples, rows, cal: Calibration) -> tuple[list[str], dict]:
    y = np.array([r["label"] for r in rows])
    p = np.array([cal.document.probability(r["document"]) for r in rows])
    window_flag = [np.mean([cal.window.probability(w) >= cal.window.threshold for w in r["windows"]] or [0])
                   for r in rows]
    thr = cal.document.threshold
    lines = [f"## {name}", "", f"{(y == 0).sum()} human, {(y == 1).sum()} AI texts. "
             f"Median length {int(np.median([count_words(e.text) for e in examples]))} words.", ""]
    summary = {}
    if 0 < y.sum() < len(y):
        oracle = np.quantile(p[y == 0], 0.99)
        long = np.array([count_words(e.text) >= 150 for e in examples])  # the app gives no verdict below 150 words
        summary = {"auroc": roc_auc_score(y, p), "tpr": (p[y == 1] >= thr).mean(), "fpr": (p[y == 0] >= thr).mean(),
                   "tpr_at_1fpr": (p[y == 1] > oracle).mean(),
                   "tpr150": (p[(y == 1) & long] >= thr).mean() if ((y == 1) & long).any() else np.nan,
                   "fpr150": (p[(y == 0) & long] >= thr).mean() if ((y == 0) & long).any() else np.nan}
        lines += [
            "| Metric | Value |", "|---|---:|",
            f"| AUROC | {summary['auroc']:.3f} |",
            f"| AI detected at detector threshold ({thr:.2f}) | {_pct(summary['tpr'])} |",
            f"| Humans wrongly flagged at detector threshold | {_pct(summary['fpr'])} |",
            f"| … texts of 150+ words only: AI detected / humans wrongly flagged | {_pct(summary['tpr150'])} / {_pct(summary['fpr150'])} |",
            f"| AI detected at 1% false-positive rate (oracle threshold) | {_pct(summary['tpr_at_1fpr'])} |",
            "",
        ]

    groups, labels = defaultdict(list), {}
    for r, prob, wf in zip(rows, p, window_flag):
        groups[r["group"]].append((prob, wf))
        labels[r["group"]] = r["label"]
    lines += ["| Source | Label | n | Flagged as AI (document) | Mean sentence share flagged |", "|---|---|---:|---:|---:|"]
    for group in sorted(groups, key=lambda g: (labels[g], g)):
        vals = groups[group]
        lines.append(f"| {group} | {'AI' if labels[group] else 'human'} | {len(vals)} "
                     f"| {_pct(np.mean([v[0] >= thr for v in vals]))} | {_pct(np.mean([v[1] for v in vals]))} |")
    return lines + [""], summary


def _comparison(suites, model_name: str) -> list[str]:
    """Same suites through an independent public detector at its own 0.5 cut-off (scores cached by text id)."""
    import json

    from .data import CACHE
    from .external import ExternalDetector

    path = CACHE / f"external__{model_name.replace('/', '__')}.json"
    scores = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    external = None
    lines = [f"## Comparison: {model_name}", "",
             "An independent public detector, run on the same texts at its default 0.5 cut-off.", "",
             "| Test set | AUROC | AI caught | Humans wrongly flagged |", "|---|---:|---:|---:|"]
    for title, _, examples in suites:
        for e in examples:
            if e.id not in scores:
                external = external or ExternalDetector(model_name)
                scores[e.id] = external.probability(e.text[:12_000])
        y = np.array([e.label for e in examples])
        p = np.array([scores[e.id] for e in examples])
        if 0 < y.sum() < len(y):
            lines.append(f"| {title.split(' (')[0]} | {roc_auc_score(y, p):.3f} | {_pct((p[y == 1] >= 0.5).mean())} "
                         f"| {_pct((p[y == 0] >= 0.5).mean())} |")
    path.write_text(json.dumps(scores), encoding="utf-8")
    return lines + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--performer", default=DEFAULT_PERFORMER)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--compare", nargs="?", const=DEFAULT_EXTERNAL, default=None,
                    help="also score every suite with an independent public detector (default: %(const)s)")
    args = ap.parse_args()

    performer = None if args.performer.lower() == "none" else args.performer
    cal = Calibration.load(calibration_path(f"{args.model}+{performer}" if performer else args.model))
    scorer = scorer_from_args(args, cal.hidden_layer)
    detector = Detector(scorer, cal)

    report = [f"# Detector evaluation: {scorer.name}", "",
              f"Run {date.today().isoformat()}. Calibrated on {cal.meta.get('trained_on', '?')}.", ""]
    suites = [
        ("Student-style essays (Ghostbuster test split)", "gb-essay-test", balanced(ghostbuster_essays("test"), args.per_class)),
        ("News and fiction (Ghostbuster Reuters + WritingPrompts, held-out 30%)", "gb-domains-test",
         balanced(ghostbuster_domain("reuter", "test") + ghostbuster_domain("wp", "test"), args.per_class)),
        ("Real student essays incl. English learners (PERSUADE) vs student-imitating Llama (DAIGT), held-out 30%",
         "students-test", balanced(persuade("test") + daigt("test"), args.per_class)),
        ("Formal writing: Wikipedia, medicine, finance, Q&A vs ChatGPT (HC3, held-out 30%)", "hc3-test",
         balanced(hc3("test"), args.per_class)),
        ("Newer-model AI (Claude 3.5 Sonnet, Qwen2.5 abstracts) vs human academic abstracts (arXiv), held-out 30%",
         "modern-test", balanced(modern_claude("test") + generated("test") + arxiv_abstracts("test"), args.per_class,
                                 ai_key=lambda e: e.group.split(":")[0])),
        ("RAID with adversarial attacks (GPT-4, ChatGPT, Cohere, Mistral, Llama...; held-out sources)", "raid-test",
         balanced(raid("test", group_by="attack"), args.per_class)),
        ("TextSight 2026 control set (never trained on: current models, non-native academic writers)",
         "textsight2026", textsight()),
    ]
    table = ["| Test set | AUROC | AI caught | Humans wrongly flagged | Humans wrongly flagged (150+ words) |",
             "|---|---:|---:|---:|---:|"]
    for title, key, examples in suites:
        lines, s = evaluate(title, examples, extract_features(detector, examples, key), cal)
        report += lines
        if s:
            table.append(f"| {title.split(' (')[0]} | {s['auroc']:.3f} | {_pct(s['tpr'])} | {_pct(s['fpr'])} "
                         f"| {_pct(s['fpr150'])} |")
    report[4:4] = ["## Summary", "", *table, ""]

    if args.compare:
        report += _comparison(suites, args.compare)

    out = ROOT / "data" / "reports" / f"eval-{scorer.name.replace('/', '__')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
