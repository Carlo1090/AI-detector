"""Write AI counterparts for human texts with a local chat model, so every genre has both labels.

    python -m ml.eval.generate --model Qwen/Qwen2.5-1.5B-Instruct --source arxiv --n 400

Without genre-matched pairs a detector learns shortcuts ("academic abstracts are human"). Here each human
arXiv abstract gets an AI abstract written from the same title and target length. Output goes to
data/generated/<model>__<source>.jsonl and is picked up by ml.eval.data.generated().
"""

from __future__ import annotations

import argparse
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import GENERATED, arxiv_records

PROMPTS = {
    "arxiv": [
        "Write the abstract of a research paper titled \"{title}\". Write one paragraph of about {words} words. "
        "Output only the abstract.",
        "You are an academic author. Draft the abstract (about {words} words, one paragraph) for your paper "
        "\"{title}\". Reply with the abstract text only.",
        "Paper title: {title}\nWrite a concise scientific abstract of roughly {words} words summarising the "
        "motivation, method and results. Give only the abstract.",
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--source", default="arxiv", choices=sorted(PROMPTS))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    records = arxiv_records()
    rng.shuffle(records)
    short = args.model.split("/")[-1]
    out_path = GENERATED / f"{short}__{args.source}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["source_key"] for l in out_path.read_text(encoding="utf-8").splitlines()}
    todo = [r for r in records if r["key"] not in done][: max(0, args.n - len(done))]
    if not todo:
        print("nothing to do")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.eval().to(device)

    GENERATED.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for b in range(0, len(todo), args.batch):
            batch = todo[b:b + args.batch]
            prompts = []
            for r in batch:
                words = len(r["abstract"].split())
                ask = rng.choice(PROMPTS[args.source]).format(title=r["title"], words=words)
                prompts.append(tok.apply_chat_template([{"role": "user", "content": ask}], tokenize=False,
                                                       add_generation_prompt=True))
            enc = tok(prompts, return_tensors="pt", padding=True).to(device)
            budget = int(max(len(r["abstract"].split()) for r in batch) * 1.9) + 40
            with torch.inference_mode():
                gen = model.generate(**enc, do_sample=True, temperature=rng.choice([0.7, 0.8, 1.0]), top_p=0.95,
                                     max_new_tokens=budget, pad_token_id=tok.eos_token_id)
            texts = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for r, text in zip(batch, texts):
                text = " ".join(text.replace("**", "").split())
                for prefix in ("Abstract:", "ABSTRACT:", "Abstract"):
                    if text.startswith(prefix):
                        text = text[len(prefix):].lstrip(" :")
                if len(text.split()) < 60:
                    continue
                f.write(json.dumps({"id": f"gen-{short}-{r['key']}", "source_key": r["key"], "text": text,
                                    "group": f"gen-{args.source}:{short}", "title": r["title"]}) + "\n")
            f.flush()
            print(f"  {min(b + args.batch, len(todo))}/{len(todo)} generated", flush=True)


if __name__ == "__main__":
    main()
