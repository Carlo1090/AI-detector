"""Fine-tune the local rewriter (QLoRA) to turn AI-style text back into human writing.

    python -m ml.humanizer.train.finetune [--epochs 2] [--max-pairs 3000]

Base: Qwen2.5-1.5B-Instruct loaded in 4-bit (fits a 4 GB GPU). LoRA adapters on every attention and MLP
projection. Loss is computed only on the human answer. 5% of pairs are held out; the adapter with the lowest
held-out loss is saved to ml/humanizer/adapters/<name>/ and used by LocalRewriter automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ml.humanizer.rewriter import ADAPTER_DIR, adapter_messages

from .make_pairs import PAIRS

ACADEMIC_SOURCES = {"arxiv", "raid-abstracts"}


def encode(tok, pair: dict, max_len: int) -> dict | None:
    tone = "academic" if pair["source"] in ACADEMIC_SOURCES else "natural"
    prompt = tok.apply_chat_template(adapter_messages(pair["ai"], tone), tokenize=True, add_generation_prompt=True)
    if not isinstance(prompt, list):  # some tokenizer versions return a BatchEncoding
        prompt = prompt["input_ids"]
    answer = tok(pair["human"], add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
    ids = (prompt + answer)[:max_len]
    labels = ([-100] * len(prompt) + answer)[:max_len]
    return {"ids": ids, "labels": labels} if any(l != -100 for l in labels) else None


def batches(items: list[dict], size: int, pad_id: int, shuffle: bool, rng: random.Random):
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))  # similar lengths -> less padding
    groups = [order[i:i + size] for i in range(0, len(order), size)]
    if shuffle:
        rng.shuffle(groups)
    for g in groups:
        width = max(len(items[i]["ids"]) for i in g)
        ids = torch.full((len(g), width), pad_id)
        labels = torch.full((len(g), width), -100)
        mask = torch.zeros((len(g), width), dtype=torch.long)
        for row, i in enumerate(g):
            n = len(items[i]["ids"])
            ids[row, :n] = torch.tensor(items[i]["ids"])
            labels[row, :n] = torch.tensor(items[i]["labels"])
            mask[row, :n] = 1
        yield ids, labels, mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--name", default="qwen2.5-1.5b-humanizer")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-pairs", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    rng = random.Random(0)
    torch.manual_seed(0)
    pairs = [json.loads(l) for l in PAIRS.read_text(encoding="utf-8").splitlines()][: args.max_pairs]
    rng.shuffle(pairs)

    tok = AutoTokenizer.from_pretrained(args.base)
    data = [x for x in (encode(tok, p, args.max_len) for p in pairs) if x]
    n_eval = max(40, len(data) // 20)
    eval_set, train_set = data[:n_eval], data[n_eval:]
    print(f"{len(train_set)} training pairs, {len(eval_set)} held out", flush=True)

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=quant, device_map={"": 0},
                                                 dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    import bitsandbytes as bnb

    optim = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_set) / (args.batch * args.accum))
    total = max(1, int(steps_per_epoch * args.epochs))
    warmup = max(10, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    def evaluate() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for ids, labels, mask in batches(eval_set, args.batch, tok.pad_token_id, False, rng):
                out = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda())
                losses.append(out.loss.item())
        model.train()
        return sum(losses) / len(losses)

    out_dir = ADAPTER_DIR / args.name
    best = evaluate()
    print(f"held-out loss before training: {best:.4f}", flush=True)
    step, t0 = 0, time.time()
    model.train()
    while step < total:
        micro = 0
        for ids, labels, mask in batches(train_set, args.batch, tok.pad_token_id, True, rng):
            out = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda())
            (out.loss / args.accum).backward()
            micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                print(f"  step {step}/{total}  loss {out.loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                      f"{(time.time() - t0) / step:.1f}s/step", flush=True)
            if step % 50 == 0 or step == total:
                loss = evaluate()
                print(f"  held-out loss {loss:.4f} (best {best:.4f})", flush=True)
                if loss < best:
                    best = loss
                    model.save_pretrained(out_dir)
                    (out_dir / "training.json").write_text(json.dumps(
                        {"base": args.base, "step": step, "held_out_loss": loss, "pairs": len(train_set),
                         "epochs": args.epochs, "rank": args.rank, "lr": args.lr}, indent=2), encoding="utf-8")
                    print(f"  saved adapter -> {out_dir}", flush=True)
            if step >= total:
                break
    print(f"done in {(time.time() - t0) / 60:.0f} min; best held-out loss {best:.4f}")


if __name__ == "__main__":
    main()
