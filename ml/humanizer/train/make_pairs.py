"""Build (AI-style, human) training pairs for the humanizer.

    python -m ml.humanizer.train.make_pairs --n 2500

Take well-written human paragraphs (training splits only: essays, news, fiction, Wikipedia, expert answers,
academic abstracts, books), and have a local chat model polish each into typical "AI style" with the same
content. Fine-tuning on AI -> original-human then teaches the rewriter to undo AI polish while keeping the
information and the register. Student essays and Reddit are left out so the model doesn't learn to add
errors or slang. Output: data/pairs/pairs.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ml.eval.data import ROOT, arxiv_abstracts, ghostbuster_domain, ghostbuster_essays, hc3, raid
from ml.humanizer.meaning import Embedder, missing_spans, must_keep
from ml.humanizer.rewriter import clean_output

PAIRS = ROOT / "data" / "pairs" / "pairs.jsonl"
_NUMBERS = re.compile(r"\d+(?:[.,]\d+)*")

AI_PROMPTS = [
    "Rewrite the following passage so it reads like polished text written by an AI assistant: formal, fluent and "
    "well organised, with smooth transitions. Keep every fact and idea and roughly the same length. Output only the "
    "rewritten passage.\n\n{text}",
    "Improve the writing of this passage. Make it clear, professional and articulate. Keep all the information and "
    "roughly the same length. Reply with the improved passage only.\n\n{text}",
    "Paraphrase the following text in a sophisticated, engaging tone. Preserve every detail. Output only the "
    "paraphrase.\n\n{text}",
]


def paragraphs(text: str, lo: int = 50, hi: int = 150) -> list[str]:
    """Split into chunks of lo..hi words at paragraph, then sentence, boundaries."""
    out, buf = [], []
    for para in re.split(r"\n\s*\n", text.strip()):
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(para.split())):
            buf.append(sentence)
            if len(" ".join(buf).split()) >= hi * 0.8:
                out.append(" ".join(buf))
                buf = []
        if len(" ".join(buf).split()) >= lo:
            out.append(" ".join(buf))
            buf = []
    return [p for p in out if lo <= len(p.split()) <= hi * 1.2]


def human_sources() -> list[tuple[str, str]]:
    raw = (
        [("essay", e.text) for e in ghostbuster_essays("train") if e.label == 0]
        + [(e.group.split(":")[0], e.text) for d in ("reuter", "wp") for e in ghostbuster_domain(d, "train") if e.label == 0]
        + [(e.group.split(":")[0], e.text) for e in hc3("train", subsets=("wiki_csai", "medicine", "finance")) if e.label == 0]
        + [("arxiv", e.text) for e in arxiv_abstracts("train")]
        + [(e.group.split(":")[0], e.text) for e in raid("train")
           if e.label == 0 and e.group.split(":")[0] in ("raid-abstracts", "raid-books", "raid-wiki")]
    )
    return [(src, p) for src, text in raw for p in paragraphs(text)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_source: dict[str, list[str]] = {}
    for src, para in human_sources():
        by_source.setdefault(src, []).append(para)
    for paras in by_source.values():
        rng.shuffle(paras)
    pool, i = [], 0  # interleave sources so every kind of writing is represented
    while len(pool) < args.n and any(i < len(p) for p in by_source.values()):
        pool += [(s, p[i]) for s, p in by_source.items() if i < len(p)]
        i += 1
    pool = pool[:args.n]

    done = set()
    if PAIRS.exists():
        done = {json.loads(l)["human"] for l in PAIRS.read_text(encoding="utf-8").splitlines()}
    todo = [(s, p) for s, p in pool if p not in done]
    print(f"{len(todo)} paragraphs to rewrite ({len(done)} already done) from {sorted(by_source)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval().to("cuda")
    embedder = Embedder()  # CPU
    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    kept, rejected = 0, {}
    with PAIRS.open("a", encoding="utf-8") as f:
        for b in range(0, len(todo), args.batch):
            batch = todo[b:b + args.batch]
            prompts = [tok.apply_chat_template([{"role": "user", "content": rng.choice(AI_PROMPTS).format(text=p)}],
                                               tokenize=False, add_generation_prompt=True) for _, p in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            budget = int(max(len(p.split()) for _, p in batch) * 2.2) + 40
            with torch.inference_mode():
                out = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=budget,
                                     pad_token_id=tok.eos_token_id)
            for (src, human), text in zip(batch, tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
                ai = clean_output(text, human, keep_dashes=True)  # the AI side keeps its AI habits
                ratio = len(ai.split()) / len(human.split())
                # Same content on both sides, or the model would learn to invent facts while "restoring" text:
                # the AI side keeps every name/number of the original and adds no numbers of its own.
                missing = missing_spans(human, ai)
                names = [k for k in must_keep(human) if not any(c.isdigit() for c in k) and k[0] not in "\"“"]
                if not 0.6 <= ratio <= 1.8 or re.search(r"[　-鿿]", ai):
                    reason = "length/garbled"
                elif any(any(c.isdigit() for c in m) for m in missing):
                    reason = "dropped a number"
                elif len([m for m in missing if m in names]) > max(1, 0.2 * len(names)):
                    reason = "dropped names"
                elif not set(_NUMBERS.findall(ai)) <= set(_NUMBERS.findall(human)):
                    reason = "invented a number"
                elif embedder.similarity(human, [ai])[0] < 0.78:
                    reason = "meaning drifted"
                else:
                    f.write(json.dumps({"source": src, "ai": ai, "human": human}) + "\n")
                    kept += 1
                    continue
                rejected[reason] = rejected.get(reason, 0) + 1
            f.flush()
            print(f"  {min(b + args.batch, len(todo))}/{len(todo)} rewritten, {kept} kept, rejected {rejected}",
                  flush=True)


if __name__ == "__main__":
    main()
