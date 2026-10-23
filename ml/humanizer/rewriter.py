"""Candidate rewrites of a passage from a language model.

Two backends with the same interface:
- LocalRewriter: a small instruct model on this machine (default Qwen2.5-1.5B-Instruct). Free and private.
- ClaudeRewriter: Claude through the Anthropic API, used when ANTHROPIC_API_KEY is set. Much stronger writing.

The detector, not the rewriter, decides which candidate wins, so the rewriter only has to produce variety.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

DEFAULT_LOCAL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_CLAUDE = "claude-opus-5-5"

BANNED = ("delve, tapestry, testament, pivotal, crucial, foster, underscore, landscape, realm, seamless, robust, "
          "leverage, navigate, showcase, vibrant, profound, multifaceted, intricate, embark, harness, myriad, plethora")

_TONES = {
    "natural": "Write clear, natural prose. Contractions are fine where a careful writer would use them.",
    "academic": "Keep an academic register suitable for a school or university essay: no contractions, no slang, "
                "no rhetorical questions.",
}

SYSTEM = """You are a skilled human editor. Rewrite the passage so it reads as if a thoughtful person wrote it, not an AI model. Keep it at the same level: an essay stays an essay, a report stays a report.

Keep:
- every idea, example, fact, name, number, citation and quotation. Drop nothing, add nothing.
- the same level of formality and vocabulary. Do not simplify it as if for a child, and do not add slang.
- roughly the same length.

Change:
- sentence structure and rhythm: mix short and long sentences, and vary how sentences begin.
- stock phrases: say the same thing in plain, specific words.
- filler and empty transitions: remove them.

Never use these words: {banned}.
Do not use em dashes, lists, headings or bold text. Do not start sentences with Moreover, Furthermore, Additionally, Overall, Ultimately or In conclusion.
{tone}
Output only the rewritten passage. No preface, no quotes, no notes."""

_EXAMPLES = [
    ("In today's fast-paced world, effective time management plays a crucial role in achieving academic success. "
     "Moreover, it fosters a sense of balance and enables students to navigate their responsibilities seamlessly.",
     "Students who manage their time well tend to do better in school. Planning ahead also keeps coursework from "
     "crowding out everything else, which makes a heavy workload easier to carry."),
    ("The Industrial Revolution was a pivotal period that transformed society in profound ways. It brought "
     "about significant advancements in technology, which in turn reshaped the economic landscape.",
     "The Industrial Revolution changed society at its roots. New machines spread quickly, and as they did, they "
     "rearranged how people worked and earned a living."),
    ("This study delves into the intricate relationship between sleep quality and academic performance among "
     "university students. The findings underscore the importance of fostering healthy sleep habits.",
     "We examined how sleep quality relates to academic performance in university students. Those who slept "
     "poorly earned lower grades, a result that gives universities good reason to take student sleep seriously."),
]

_PUSH = ("Restructure more boldly than usual: split or merge sentences and reorder clauses. Keep every idea and the "
         "same level of formality.")

# Fine-tuned humanizer adapters (ml.humanizer.train.finetune) use a short prompt with no examples.
ADAPTER_DIR = Path(__file__).parent / "adapters"
DEFAULT_ADAPTER = "qwen2.5-1.5b-humanizer"
_ADAPTER_SYSTEM = ("Rewrite the passage so it reads as if a person wrote it. Keep every idea, fact, name and number, "
                   "and keep the same register. {tone}")


def adapter_messages(passage: str, tone: str) -> list[dict]:
    return [{"role": "system", "content": _ADAPTER_SYSTEM.format(tone=_TONES.get(tone, _TONES["natural"]))},
            {"role": "user", "content": passage}]

_LEADIN = re.compile(r"^\s*(?:sure[,!.]?|certainly[,!.]?|here(?:'s| is) (?:the |a |my )?(?:rewritten|revised|new)[^:\n]*:|"
                     r"rewritten(?: passage| text)?:|revised(?: passage| text)?:)\s*", re.IGNORECASE)


def clean_output(text: str, source: str, keep_dashes: bool = False) -> str:
    text = _LEADIN.sub("", text.strip())
    text = re.split(r"\n\s*(?:note|explanation|changes made)\s*:", text, flags=re.IGNORECASE)[0].strip()
    if len(text) > 1 and text[0] in "\"“" and text[-1] in "\"”" and source[:1] not in "\"“":
        text = text[1:-1].strip()
    if "\n" not in source.strip():
        text = re.sub(r"\s*\n+\s*", " ", text)
    return text if keep_dashes else text.replace(" — ", ", ").replace("—", ", ")


def _messages(passage: str, tone: str, push: bool) -> list[dict]:
    system = SYSTEM.format(banned=BANNED, tone=_TONES.get(tone, _TONES["natural"]))
    msgs = [{"role": "system", "content": system}]
    for before, after in _EXAMPLES:
        msgs += [{"role": "user", "content": f"Rewrite this passage:\n\n{before}"},
                 {"role": "assistant", "content": after}]
    ask = f"Rewrite this passage:\n\n{passage}"
    if push:
        ask += f"\n\n{_PUSH}"
    msgs.append({"role": "user", "content": ask})
    return msgs


class LocalRewriter:
    """A local instruct model. If a fine-tuned humanizer adapter exists (ml/humanizer/adapters/<name>, or the
    folder named by HUMANIZE_ADAPTER; 'none' disables it), it is merged in and its own prompt format is used."""

    def __init__(self, model_name: str = DEFAULT_LOCAL, device: str | None = None, adapter: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if self.device == "cuda" else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).eval()

        adapter = adapter or os.environ.get("HUMANIZE_ADAPTER", DEFAULT_ADAPTER)
        path = Path(adapter) if Path(adapter).is_absolute() else ADAPTER_DIR / adapter
        self.adapter = None
        if adapter.lower() != "none" and (path / "adapter_config.json").exists():
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, path).merge_and_unload().eval()
            self.adapter = path.name
        self.name = f"local:{model_name}" + (f"+{self.adapter}" if self.adapter else "")
        self.model.to("cpu")  # parked until needed; the detector owns the GPU between rewrites

    def to(self, device: str) -> None:
        self.model.to(device)

    def rewrite(self, passages: list[str], n: int = 4, tone: str = "natural", push: bool = False,
                temperature: float = 0.9) -> list[list[str]]:
        torch = self.torch
        device = next(self.model.parameters()).device
        results = []
        for passage in passages:
            messages = adapter_messages(passage, tone) if self.adapter else _messages(passage, tone, push)
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = self.tokenizer(prompt, return_tensors="pt").to(device)
            budget = int(len(self.tokenizer(passage).input_ids) * 1.6) + 40
            with torch.inference_mode():
                out = self.model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95,
                                          repetition_penalty=1.05, max_new_tokens=budget, num_return_sequences=n,
                                          pad_token_id=self.tokenizer.eos_token_id)
            texts = self.tokenizer.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            results.append([clean_output(t, passage) for t in texts])
        return results


class ClaudeRewriter:
    """Asks Claude for n distinct rewrites per passage in one call.

    Sampling parameters aren't accepted by current Claude models; variety comes from asking for n
    structurally different versions. Server-side refusal fallbacks are on; a passage the whole chain
    declines simply gets no candidates and keeps its original wording.
    """

    def __init__(self, model: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("HUMANIZE_CLAUDE_MODEL", DEFAULT_CLAUDE)
        self.name = f"claude:{self.model}"

    def to(self, device: str) -> None:  # runs remotely; nothing to move
        pass

    def rewrite(self, passages: list[str], n: int = 4, tone: str = "natural", push: bool = False,
                temperature: float = 0.9) -> list[list[str]]:
        system = SYSTEM.format(banned=BANNED, tone=_TONES.get(tone, _TONES["natural"]))
        system += (f"\n\nReturn exactly {n} different rewrites as a JSON array of strings and nothing else. "
                   "Make them differ in sentence structure, not just word choice.")
        results = []
        for passage in passages:
            ask = f"Rewrite this passage:\n\n{passage}"
            if push:
                ask += f"\n\n{_PUSH}"
            msg = self.client.beta.messages.create(
                model=self.model, max_tokens=16000, system=system,
                output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                messages=[{"role": "user", "content": ask}])
            if msg.stop_reason == "refusal":
                results.append([])
                continue
            raw = "".join(block.text for block in msg.content if block.type == "text")
            try:
                variants = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
            except ValueError:
                variants = [raw]
            results.append([clean_output(str(v), passage) for v in variants if str(v).strip()][:n])
        return results


def load_rewriter():
    """Claude when an API key is configured (and HUMANIZE_BACKEND isn't 'local'); otherwise the local model."""
    backend = os.environ.get("HUMANIZE_BACKEND", "auto").lower()
    if backend in ("auto", "claude") and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return ClaudeRewriter()
        except ImportError:
            if backend == "claude":
                raise
    return LocalRewriter(os.environ.get("HUMANIZE_MODEL", DEFAULT_LOCAL))
