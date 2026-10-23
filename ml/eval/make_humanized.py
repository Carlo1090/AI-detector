"""Run AI texts through the humanizer so the detector can learn to catch "AI-paraphrased" text.

    python -m ml.eval.make_humanized [--n 150] [--strength balanced]

Turnitin added detection of text that went through AI "bypasser" tools in 2025; this is the same idea.
AI texts are taken from both the train and test sides of each dataset, and each humanized version keeps
its original's split key, so evaluation stays on texts the detector never trained on. Output:
data/generated/humanized__<rewriter>.jsonl, picked up by ml.eval.data.generated().
"""

from __future__ import annotations

import argparse
import json
import random
import re

from ml.detector.detector import Detector
from ml.humanizer.humanizer import Humanizer
from ml.humanizer.meaning import Embedder
from ml.humanizer.rewriter import load_rewriter

from .data import GENERATED, generated, ghostbuster_essays, hc3, modern_claude, raid


def _clip(text: str, words: int = 320) -> str:
    out, count = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        out.append(sentence)
        count += len(sentence.split())
        if count >= words:
            break
    return " ".join(out)


def ai_pool(rng: random.Random) -> list[tuple[str, str, str, str]]:
    """(split, id, family, text) for AI texts from several families, train and test sides alike."""
    pool = []
    for split in ("train", "test"):
        families = {
            "essay": [e for e in ghostbuster_essays(split) if e.label == 1],
            "hc3": [e for e in hc3(split) if e.label == 1],
            "sonnet35": modern_claude(split),
            "frontier": [e for e in generated(split) if e.group.startswith("frontier")],
            "raid-chat": [e for e in raid(split)
                          if e.label == 1 and (e.group.endswith("-chat") or e.group in ("raid:gpt4", "raid:chatgpt"))],
        }
        for family, items in families.items():
            rng.shuffle(items)
            pool += [(split, e.id, family, e.text) for e in items[:40] if len(e.text.split()) >= 120]
    rng.shuffle(pool)
    return pool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--strength", default="balanced", choices=["light", "balanced", "strong"])
    args = ap.parse_args()

    rng = random.Random(0)
    rewriter = load_rewriter()
    humanizer = Humanizer(Detector.load(), rewriter, Embedder())
    slug = rewriter.name.split(":")[-1].replace("/", "__")
    out = GENERATED / f"humanized__{slug}.jsonl"
    done = {json.loads(l)["source_key"] for l in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    todo = [x for x in ai_pool(rng) if x[1] not in done][: max(0, args.n - len(done))]
    print(f"{len(todo)} AI texts to humanize ({len(done)} done)", flush=True)

    with out.open("a", encoding="utf-8") as f:
        for k, (split, key, family, text) in enumerate(todo, 1):
            h = humanizer.humanize(_clip(text), args.strength, "natural")
            # The humanized copy stays on its original's side of the train/test split.
            f.write(json.dumps({"id": f"humanized-{key}", "source_key": key, "split": split, "text": h["text"],
                                "group": f"humanized:{family}"}) + "\n")
            f.flush()
            print(f"  {k}/{len(todo)} {family}: AI share {h['before']['ai_share']} -> {h['after']['ai_share']}",
                  flush=True)


if __name__ == "__main__":
    main()
