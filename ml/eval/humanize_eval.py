"""Benchmark the humanizer on held-out AI text: does it stop detection without changing the meaning?

    python -m ml.eval.humanize_eval [--n 60] [--strength balanced] [--words 350]

Scores every text before and after with (a) our detector, which the humanizer optimises against, and
(b) an independent detector it never sees. Only (b) says anything about detectors in general.
Writes data/reports/humanize-<setup>.md with numbers and a few before/after samples to read.
"""

from __future__ import annotations

import argparse
import random
import re
import time
from datetime import date

import numpy as np

from ml.detector.detector import Detector
from ml.humanizer.humanizer import Humanizer
from ml.humanizer.meaning import Embedder
from ml.humanizer.rewriter import load_rewriter

from .data import ROOT, daigt, ghostbuster_essays, textsight
from .external import DEFAULT_EXTERNAL, ExternalDetector


def _clip(text: str, words: int) -> str:
    """First ~`words` words, cut at a sentence end."""
    out, count = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        out.append(sentence)
        count += len(sentence.split())
        if count >= words:
            break
    return " ".join(out)


def held_out_ai(n: int, words: int, seed: int = 0):
    rng = random.Random(seed)
    pools = {
        "textsight-2026": [e for e in textsight() if e.label == 1],
        "essay-claude/gpt": [e for e in ghostbuster_essays("test") if e.label == 1],
        "daigt-student-mimic": daigt("test"),
    }
    picked = []
    for name, pool in pools.items():
        rng.shuffle(pool)
        picked += [(name, _clip(e.text, words)) for e in pool[: n // len(pools)]]
    return [(g, t) for g, t in picked if len(t.split()) >= 120]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--words", type=int, default=350)
    ap.add_argument("--strength", default="balanced", choices=["light", "balanced", "strong"])
    ap.add_argument("--tone", default="natural", choices=["natural", "academic"])
    ap.add_argument("--external", default=DEFAULT_EXTERNAL)
    args = ap.parse_args()

    detector = Detector.load()
    rewriter = load_rewriter() if args.strength != "light" else None
    humanizer = Humanizer(detector, rewriter, Embedder())
    texts = held_out_ai(args.n, args.words)
    thr = detector.calibration.document.threshold

    rows = []
    for k, (group, text) in enumerate(texts, 1):
        t0 = time.perf_counter()
        h = humanizer.humanize(text, args.strength, args.tone)
        rows.append({"group": group, "before_text": text, "after_text": h["text"],
                     "ours_before": h["before"]["document_probability"], "ours_after": h["after"]["document_probability"],
                     "share_before": h["before"]["ai_share"], "share_after": h["after"]["ai_share"],
                     "similarity": h["similarity"], "seconds": time.perf_counter() - t0,
                     "rewrites": sum(c["kind"] == "rewrite" for c in h["changes"])})
        print(f"  {k}/{len(texts)} {group}: ours {rows[-1]['ours_before']:.2f} -> {rows[-1]['ours_after']:.2f} "
              f"({rows[-1]['seconds']:.0f}s)", flush=True)

    # the independent detector loads after the humanizer is done with the GPU
    rewriter_name = getattr(rewriter, "name", "rules only")
    del humanizer, rewriter
    external = ExternalDetector(args.external)
    ext_scores = [(external.probability(r["before_text"]), external.probability(r["after_text"])) for r in rows]
    ext_thr = 0.5

    def rate(xs, t):
        return f"{100 * np.mean([x >= t for x in xs]):.0f}%"

    lines = [f"# Humanizer benchmark ({args.strength}, {args.tone})", "",
             f"Run {date.today().isoformat()} on {len(rows)} held-out AI texts (~{args.words} words each). "
             f"Detector: {detector.scorer.name}. Rewriter: {rewriter_name}.", "",
             "| Detector | Flagged before | Flagged after |", "|---|---:|---:|",
             f"| Ours (document, threshold {thr:.2f}) | {rate([r['ours_before'] for r in rows], thr)} "
             f"| {rate([r['ours_after'] for r in rows], thr)} |",
             f"| Independent: {args.external} (p ≥ 0.5) | {rate([e[0] for e in ext_scores], ext_thr)} "
             f"| {rate([e[1] for e in ext_scores], ext_thr)} |", "",
             f"Mean sentence share flagged by ours: {np.nanmean([r['share_before'] or 0 for r in rows]):.0%} → "
             f"{np.nanmean([r['share_after'] or 0 for r in rows]):.0%}. "
             f"Mean meaning similarity of rewritten passages: {np.mean([r['similarity'] for r in rows]):.2f}. "
             f"Mean time per text: {np.mean([r['seconds'] for r in rows]):.0f}s.", "",
             "| Source | n | Ours before → after | Independent before → after |", "|---|---:|---:|---:|"]
    for group in dict.fromkeys(r["group"] for r in rows):
        idx = [i for i, r in enumerate(rows) if r["group"] == group]
        lines.append(f"| {group} | {len(idx)} | {rate([rows[i]['ours_before'] for i in idx], thr)} → "
                     f"{rate([rows[i]['ours_after'] for i in idx], thr)} | "
                     f"{rate([ext_scores[i][0] for i in idx], ext_thr)} → {rate([ext_scores[i][1] for i in idx], ext_thr)} |")
    lines += ["", "## Samples", ""]
    for r in rows[:: max(1, len(rows) // 3)][:3]:
        lines += [f"**{r['group']}**", "", "> " + r["before_text"].replace("\n", "\n> "), "", "→", "",
                  "> " + r["after_text"].replace("\n", "\n> "), ""]

    out = ROOT / "data" / "reports" / f"humanize-{args.strength}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:16]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
