"""Independent, held-out AI detectors used only for evaluation (never for choosing rewrites)."""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_EXTERNAL = "Oxidane/tmr-ai-text-detector"


class ExternalDetector:
    """A Hugging Face sequence classifier; returns P(AI). Long texts are scored in 512-token chunks and averaged."""

    def __init__(self, model_name: str = DEFAULT_EXTERNAL, device: str | None = None):
        self.name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(self.device)
        labels = {i: str(l).lower() for i, l in self.model.config.id2label.items()}
        ai = [i for i, l in labels.items() if any(k in l for k in ("ai", "machine", "generated", "fake", "gpt"))]
        self.ai_index = ai[0] if ai else max(labels)  # default: last label is the positive class

    @torch.inference_mode()
    def probability(self, text: str) -> float:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        chunks = [ids[i:i + 500] for i in range(0, max(1, len(ids)), 500)] or [[]]
        probs = []
        tok = self.tokenizer
        for chunk in chunks:
            ids = torch.tensor([[tok.cls_token_id, *chunk, tok.sep_token_id]], device=self.device)
            logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0].float()
            probs.append(torch.softmax(logits, -1)[self.ai_index].item())
        weights = [len(c) for c in chunks]
        return sum(p * w for p, w in zip(probs, weights)) / max(1, sum(weights))
