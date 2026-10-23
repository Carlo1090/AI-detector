"""Word-by-word predictability from small causal language models.

For every token we record how likely the observer model thought it was, its rank among all possible next
tokens, and the mean/variance of log-probability under the model's own distribution. The last two feed
Fast-DetectGPT's "probability curvature" (Bao et al., 2024), which needs only this single forward pass.

With a performer model (the chat-tuned sibling of the observer) we also record what Binoculars
(Hans et al., 2024) needs: the performer's log-probability of each token and the cross-entropy between
the two models' next-token distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_PERFORMER = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class TokenStats:
    starts: np.ndarray  # char offset where the token's visible text starts
    ends: np.ndarray
    logprob: np.ndarray  # log p(actual token)
    rank: np.ndarray  # 1 = the model's top guess
    expected: np.ndarray  # E[log p] under the model's distribution (= -entropy)
    variance: np.ndarray  # Var[log p] under the model's distribution
    alt_ids: np.ndarray  # [n, k] the model's top-k guesses
    alt_probs: np.ndarray  # [n, k]
    perf_logprob: np.ndarray | None = None  # performer's log p(actual token)
    cross_entropy: np.ndarray | None = None  # -sum_v p_observer(v) log p_performer(v)
    hidden: np.ndarray | None = None  # [n, H] observer's internal state at `hidden_layer`, one row per token

    def __len__(self) -> int:
        return len(self.logprob)


def _load(name: str, device: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    model.eval().to(device)
    return model


class TokenScorer:
    def __init__(self, model_name: str = DEFAULT_MODEL, performer_name: str | None = None,
                 device: str | None = None, window: int = 1024, stride: int = 512, chunk: int = 256,
                 top_k: int = 5, hidden_layer: int | None = None):
        self.model_name = model_name
        self.performer_name = performer_name
        self.hidden_layer = hidden_layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = _load(model_name, self.device, dtype)
        self.base = model.base_model
        # The observer's output head runs in float32 so ranks and probabilities are not blurred by half precision.
        head = model.get_output_embeddings()
        self.head_weight = head.weight.detach().float()
        self.head_bias = head.bias.detach().float() if getattr(head, "bias", None) is not None else None

        self.perf_base = self.perf_head = None
        if performer_name:
            perf = _load(performer_name, self.device, dtype)
            if perf.config.vocab_size != model.config.vocab_size:
                raise ValueError(f"{performer_name} does not share {model_name}'s vocabulary")
            self.perf_base, self.perf_head = perf.base_model, perf.get_output_embeddings()

        self.hidden_size = model.config.hidden_size
        max_pos = getattr(model.config, "max_position_embeddings", None) or window
        self.window = min(window, max_pos)
        self.stride = min(stride, self.window - 1)
        self.chunk = chunk
        self.top_k = top_k
        tok = self.tokenizer
        self.start_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        self._decoded: dict[int, str] = {}

    def to(self, device: str) -> None:
        """Move the models (e.g. to CPU while a rewriter needs the GPU)."""
        self.base.to(device)
        self.head_weight = self.head_weight.to(device)
        if self.head_bias is not None:
            self.head_bias = self.head_bias.to(device)
        if self.perf_base is not None:
            self.perf_base.to(device)
            self.perf_head.to(device)
        self.device = device

    @property
    def name(self) -> str:
        """Identifies the model setup; calibrations are stored under this name."""
        return f"{self.model_name}+{self.performer_name}" if self.performer_name else self.model_name

    def decode(self, token_id: int) -> str:
        if token_id not in self._decoded:
            self._decoded[token_id] = self.tokenizer.decode([token_id])
        return self._decoded[token_id]

    @torch.inference_mode()
    def score(self, text: str) -> TokenStats:
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = [self.start_id] + list(enc["input_ids"])
        n = len(ids) - 1
        out = {
            "logprob": np.zeros(n, np.float32), "rank": np.zeros(n, np.int64),
            "expected": np.zeros(n, np.float32), "variance": np.zeros(n, np.float32),
            "alt_ids": np.zeros((n, self.top_k), np.int64), "alt_probs": np.zeros((n, self.top_k), np.float32),
        }
        if self.perf_base is not None:
            out["perf_logprob"] = np.zeros(n, np.float32)
            out["cross_entropy"] = np.zeros(n, np.float32)
        if self.hidden_layer is not None:
            out["hidden"] = np.zeros((n, self.hidden_size), np.float16)

        # Sliding window: every token after the first window is predicted with >= (window - stride) tokens of context.
        scored = 1
        while scored <= n:
            end = min(n + 1, scored + self.stride)
            begin = max(0, end - self.window)
            window_ids = torch.tensor([ids[begin:end]], device=self.device)
            obs = self.base(input_ids=window_ids, use_cache=False, output_hidden_states=self.hidden_layer is not None)
            hidden = obs.last_hidden_state[0]
            if self.hidden_layer is not None:  # state after reading each token, for the trained probe
                layer = obs.hidden_states[self.hidden_layer][0]
                out["hidden"][scored - 1:end - 1] = layer[scored - begin:end - begin].float().cpu().numpy()
            perf_hidden = None
            if self.perf_base is not None:
                perf_hidden = self.perf_base(input_ids=window_ids, use_cache=False).last_hidden_state[0]
            for c0 in range(scored, end, self.chunk):
                c1 = min(end, c0 + self.chunk)
                rows = slice(c0 - 1 - begin, c1 - 1 - begin)
                logits = hidden[rows].float() @ self.head_weight.T
                if self.head_bias is not None:
                    logits = logits + self.head_bias
                perf_logits = None if perf_hidden is None else self.perf_head(perf_hidden[rows]).float()
                self._stats(logits, perf_logits, torch.tensor(ids[c0:c1], device=self.device), out, c0 - 1)
            scored = end

        starts, ends = _visible_spans(text, enc["offset_mapping"])
        return TokenStats(starts=starts, ends=ends, **out)

    def _stats(self, logits: torch.Tensor, perf_logits: torch.Tensor | None, targets: torch.Tensor,
               out: dict, offset: int) -> None:
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        target_lp = logp.gather(1, targets[:, None])[:, 0]
        rank = (logp > target_lp[:, None]).sum(dim=1) + 1
        plogp = torch.where(p > 0, p * logp, torch.zeros_like(p))
        expected = plogp.sum(dim=1)
        variance = (plogp * logp).sum(dim=1) - expected.square()
        top_lp, top_ids = logp.topk(self.top_k, dim=1)

        sl = slice(offset, offset + len(targets))
        out["logprob"][sl] = target_lp.cpu().numpy()
        out["rank"][sl] = rank.cpu().numpy()
        out["expected"][sl] = expected.cpu().numpy()
        out["variance"][sl] = variance.clamp(min=1e-6).cpu().numpy()
        out["alt_ids"][sl] = top_ids.cpu().numpy()
        out["alt_probs"][sl] = top_lp.exp().cpu().numpy()

        if perf_logits is not None:
            perf_logp = torch.log_softmax(perf_logits, dim=-1)
            out["perf_logprob"][sl] = perf_logp.gather(1, targets[:, None])[:, 0].cpu().numpy()
            out["cross_entropy"][sl] = (-(p * perf_logp).sum(dim=1)).cpu().numpy()


def _visible_spans(text: str, offsets) -> tuple[np.ndarray, np.ndarray]:
    """Trim surrounding whitespace from each token's span and drop overlaps (split multi-byte characters)."""
    starts, ends = np.zeros(len(offsets), np.int64), np.zeros(len(offsets), np.int64)
    prev_end = 0
    for i, (s, e) in enumerate(offsets):
        s = max(s, prev_end)
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        starts[i], ends[i] = s, max(s, e)
        prev_end = max(prev_end, ends[i])
    return starts, ends
