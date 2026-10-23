"""Labelled datasets for calibration and evaluation, plus a feature cache so the language models run once per text.

Sources (downloaded on first use into data/raw/):
- Ghostbuster essays / Reuters news / WritingPrompts stories (Verma et al., 2024; CC BY 3.0): human vs
  ChatGPT/Claude (2023).
- PERSUADE 2.0 student essays (Crossley et al.; CC BY-NC-SA 4.0): human, grades 6-12, with English-learner flags.
- DAIGT v4 (Kaggle LLM-detect competition): Llama models fine-tuned to imitate those student essays. Hard AI.
- HC3 (Guo et al., 2023; CC BY-SA 4.0): Wikipedia, medical, finance and open-Q&A answers by people vs ChatGPT
  answers to the same questions: formal human writing, where detectors make false accusations.
- Claude 3.5 Sonnet answers (science conversations, Apache-2.0; SlimOrca, MIT): newer-model AI prose.
- arXiv abstracts up to 2021 (CC0): human academic writing, often by non-native English speakers.
- RAID (Dugan et al., 2024; MIT): GPT-4, ChatGPT, Cohere, Mistral, Llama and more in 8 domains, plus
  adversarial attacks (paraphrasing, synonym swaps, homoglyphs...), sampled from 30 slices of the 12 GB file.
- TextSight 2026 control set (CC BY-SA 4.0): pre-2022 human text incl. non-native academic writers vs
  genre-matched output from 2026 models. Never used for training.

Sets without an official split are split 70/30 by a hash of the text id.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ml.detector.text import normalize

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"

_HF = "https://huggingface.co/datasets"
_GHOSTBUSTER_LABELS = ["claude", "gpt", "gpt_prompt1", "gpt_prompt2", "gpt_semantic", "gpt_writing", "human"]
_TEXTSIGHT_FILES = ["ai_generated_2026-09-29", "human_literature_publicdomain",
                    "human_pmc_ccby_nonanglophone", "human_wikipedia_pre2022"]


@dataclass
class Example:
    id: str
    text: str
    label: int  # 1 = AI, 0 = human
    group: str  # finer source, e.g. "essay:human", "essay:claude", "persuade-ell", "ai:gpt-4.1"


def _fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    return dest


def _in_split(key: str, split: str) -> bool:
    bucket = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 100
    return bucket < 70 if split == "train" else bucket >= 70


def _parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _csv(path: Path) -> list[dict]:
    csv.field_size_limit(1 << 30)
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ghostbuster_essays(split: str) -> list[Example]:
    url = f"{_HF}/polsci/ghostbuster-essay-cleaned/resolve/main/data/{split}-00000-of-00001.parquet"
    out = []
    for row in _parquet(_fetch(url, RAW / "ghostbuster" / f"{split}.parquet")):
        name = _GHOSTBUSTER_LABELS[row["label"]] if isinstance(row["label"], int) else row["label"]
        out.append(Example(f"gb-{split}-{row['ID']}", row["text"], int(name != "human"), f"essay:{name}"))
    return out


def ghostbuster_domain(domain: str, split: str) -> list[Example]:
    """domain: 'reuter' (news) or 'wp' (creative writing)."""
    url = f"{_HF}/acmc/ghostbuster_{domain}/resolve/main/data/train-00000-of-00001.parquet"
    out = []
    for i, row in enumerate(_parquet(_fetch(url, RAW / f"ghostbuster_{domain}" / "train.parquet"))):
        key = f"gb{domain}-{i}"
        if row["text"] and _in_split(key, split):
            ai = bool(row["generated"])
            out.append(Example(key, row["text"], int(ai), f"{domain}:{row['model'] if ai else 'human'}"))
    return out


def persuade(split: str) -> list[Example]:
    url = f"{_HF}/nlpatunt/D_persuade_2/resolve/main/test.csv"
    out = []
    for row in _csv(_fetch(url, RAW / "persuade" / "test.csv")):
        key = f"persuade-{row['essay_id_comp']}"
        if _in_split(key, split):
            group = "persuade-ell" if row["ell_status"].strip() == "Yes" else "persuade"
            out.append(Example(key, row["full_text"], 0, group))
    return out


def daigt(split: str) -> list[Example]:
    url = f"{_HF}/ramensoft/daigt_v4/resolve/main/daigt_magic_generations.csv"
    out = []
    for i, row in enumerate(_csv(_fetch(url, RAW / "daigt_v4" / "daigt_magic_generations.csv"))):
        key = f"daigt-{i}"
        if _in_split(key, split):
            out.append(Example(key, row["text"], 1, "daigt:llama-student-mimic"))
    return out


def hc3(split: str, subsets=("wiki_csai", "medicine", "open_qa", "finance"), min_words: int = 60) -> list[Example]:
    """HC3 (Guo et al., 2023; CC BY-SA 4.0): human answers (Wikipedia, doctors, finance pros, open Q&A) and
    ChatGPT answers to the same questions. Formal human writing is where detectors make false accusations."""
    out = []
    for subset in subsets:
        url = f"{_HF}/Hello-SimpleAI/HC3/resolve/main/{subset}.jsonl"
        lines = _fetch(url, RAW / "hc3" / f"{subset}.jsonl").read_text(encoding="utf-8").splitlines()
        for q, line in enumerate(lines):
            if not line.strip():
                continue
            row = json.loads(line)
            key = f"hc3-{subset}-{q}"
            if not _in_split(key, split):  # split by question, so both answers stay on the same side
                continue
            for label, answers in ((0, row.get("human_answers") or []), (1, row.get("chatgpt_answers") or [])):
                for j, text in enumerate(answers):
                    if text and len(text.split()) >= min_words:
                        who = "chatgpt" if label else "human"
                        out.append(Example(f"{key}-{who}-{j}", text.strip(), label, f"hc3-{subset}:{who}"))
    return out


_LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|#+)\s")


def _prose(text: str, min_words: int) -> str | None:
    """Keep chat answers that read as prose: no code, few list lines, normal spacing; strip bold markers."""
    if "```" in text or len(text.split()) < min_words:
        return None
    lines = [l for l in text.splitlines() if l.strip()]
    if sum(bool(_LIST_LINE.match(l)) for l in lines) > 0.2 * len(lines):
        return None
    if len(text) / max(1, len(text.split())) > 9:  # e.g. "please remove spaces" tasks
        return None
    return text.replace("**", "").strip()


def _partial_jsonl(path: Path) -> list[dict]:
    """Records from a file that holds only the first part of a JSONL download (last line may be cut)."""
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").split("\n"):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def modern_claude(split: str, min_words: int = 120) -> list[Example]:
    """Claude 3.5 Sonnet answers (science conversations, Apache-2.0; SlimOrca instructions, MIT): newer-model
    prose, so the detector isn't only trained on 2023-era AI text. First ~12 MB of each file."""
    sources = {
        "sonnet35-science": f"{_HF}/jeffmeloy/sonnet3.5_science_conversations/resolve/main/sonnet3.5_science_conversations_sharegpt.jsonl",
        "sonnet35-orca": f"{_HF}/Gryphe/Sonnet3.5-SlimOrcaDedupCleaned-20k/resolve/main/sharegpt_claude_sonnet3.5_slimorca_subset.jsonl",
    }
    out = []
    for name, url in sources.items():
        path = RAW / name.replace("-", "_") / "part.jsonl"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(url, headers={"Range": "bytes=0-12000000"})
            with urllib.request.urlopen(req) as resp:
                path.write_bytes(resp.read())
        for i, rec in enumerate(_partial_jsonl(path)):
            turns = rec.get("conversation") or rec.get("conversations") or []
            for j, turn in enumerate(t for t in turns if t.get("from") == "gpt"):
                key = f"{name}-{i}-{j}"
                text = _prose(turn.get("value") or "", min_words)
                if text and _in_split(f"{name}-{i}", split):
                    out.append(Example(key, text, 1, f"{name}:claude-3.5-sonnet"))
    return out


def arxiv_records(min_words: int = 120) -> list[dict]:
    """arXiv abstracts up to 2021 (CC0) with titles; LaTeX-heavy ones are skipped."""
    import zlib

    path = RAW / "arxiv" / "part.jsonl.gz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{_HF}/gfissore/arxiv-abstracts-2021/resolve/main/arxiv-abstracts.jsonl.gz"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-6000000"})) as resp:
            path.write_bytes(resp.read())
    text = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(path.read_bytes()).decode("utf-8", "ignore")
    out = []
    for line in text.split("\n"):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        abstract = " ".join(rec.get("abstract", "").split())
        title = " ".join(rec.get("title", "").split())
        if len(abstract.split()) >= min_words and abstract.count("$") < 2 and "\\" not in abstract + title:
            out.append({"key": f"arxiv-{rec.get('id')}", "title": title, "abstract": abstract})
    return out


def arxiv_abstracts(split: str) -> list[Example]:
    """arXiv abstracts up to 2021 (CC0): human academic writing, much of it by non-native English speakers."""
    return [Example(r["key"], r["abstract"], 0, "arxiv:human") for r in arxiv_records() if _in_split(r["key"], split)]


_RAID_HEADER = "id,adv_source_id,source_id,model,decoding,repetition_penalty,attack,domain,title,prompt,generation\n"
_RAID_RECORD = re.compile(r"\n([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}),[0-9a-f]{8}-")


def raid(split: str, group_by: str = "model", min_words: int = 60) -> list[Example]:
    """RAID (Dugan et al., 2024; MIT): 8 domains, 11 generators incl. GPT-4/ChatGPT/Cohere, and 11 adversarial
    attacks (paraphrase, synonym swap, misspelling, homoglyphs, zero-width characters...). The full file is
    12 GB, so 30 evenly spaced 4 MB slices are used (data/raw/raid/slice_*.csv). Split by source text, so all
    versions of one source stay on the same side. group_by='attack' groups by attack type for reporting."""
    import io

    csv.field_size_limit(1 << 30)
    out = []
    for path in sorted((RAW / "raid").glob("slice_*.csv")):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if not raw.startswith("id,"):
            m = _RAID_RECORD.search(raw)
            if not m:
                continue
            raw = _RAID_HEADER + raw[m.start() + 1:]
        rows = []
        try:
            for row in csv.DictReader(io.StringIO(raw)):
                rows.append(row)
        except csv.Error:
            pass
        for row in rows[:-1]:  # the last record of a slice is cut off
            text = (row.get("generation") or "").strip()
            if len(text.split()) < min_words or not _in_split(f"raid-{row['source_id']}", split):
                continue
            ai = row["model"] != "human"
            if group_by == "attack":
                group = f"raid-attack:{row['attack']}:{'ai' if ai else 'human'}"
            else:
                group = f"raid:{row['model']}" if ai else f"raid-{row['domain']}:human"
            out.append(Example(f"raid-{row['id']}", text, int(ai), group))
    return out


GENERATED = ROOT / "data" / "generated"


def generated(split: str) -> list[Example]:
    """AI counterparts written by local models for human texts (see ml.eval.generate); each stays on the same
    train/test side as the human text it was paired with, so the pair never straddles the split."""
    out = []
    for path in sorted(GENERATED.glob("*.jsonl")) if GENERATED.exists() else []:
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if _in_split(rec["source_key"], split):
                out.append(Example(rec["id"], rec["text"], 1, rec["group"]))
    return out


def textsight() -> list[Example]:
    out = []
    for name in _TEXTSIGHT_FILES:
        url = f"{_HF}/textsightai/ai-text-detection-benchmark-2026/resolve/main/data/{name}.jsonl"
        for line in _fetch(url, RAW / "textsight2026" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            is_ai = row["label"] == "ai"
            out.append(Example(row["id"], row["text"], int(is_ai), f"ai:{row['source']}" if is_ai else row["source"]))
    return out


def training_mix(per_class: int, seed: int = 0) -> list[Example]:
    """Human and AI texts across essays, student writing, news, fiction, formal and academic writing.
    Human texts are spread evenly over sources; AI texts over source families (e.g. all six Ghostbuster GPT
    prompt variants count as one family), so newer-model data isn't drowned out by old variants."""
    pool = (ghostbuster_essays("train") + ghostbuster_domain("reuter", "train") + ghostbuster_domain("wp", "train")
            + persuade("train") + daigt("train") + hc3("train") + modern_claude("train") + arxiv_abstracts("train")
            + generated("train") + raid("train"))
    return balanced(pool, per_class, seed, ai_key=_ai_family)


_RAID_BASE = {"raid:gpt2", "raid:gpt3", "raid:mpt", "raid:mistral", "raid:cohere"}


def _ai_family(e: Example) -> str:
    """Balancing unit for AI text: each RAID chat model is its own family, older base models share one."""
    if e.group.startswith("raid:"):
        return "raid-base" if e.group in _RAID_BASE else e.group
    return e.group.split(":")[0]


def balanced(examples: list[Example], per_class: int, seed: int = 0, ai_key=None) -> list[Example]:
    """Up to per_class human and per_class AI texts, each class spread evenly across its groups."""
    rng = random.Random(seed)
    picked = []
    for label in (0, 1):
        key = ai_key if (label == 1 and ai_key) else (lambda e: e.group)
        by_group: dict[str, list[Example]] = {}
        for e in examples:
            if e.label == label:
                by_group.setdefault(key(e), []).append(e)
        for group in by_group.values():
            rng.shuffle(group)
        chosen, i = [], 0
        while len(chosen) < per_class and any(i < len(g) for g in by_group.values()):
            chosen.extend(g[i] for g in by_group.values() if i < len(g))
            i += 1
        picked += chosen[:per_class]
    return picked


def extract_features(detector, examples: list[Example], name: str) -> list[dict]:
    """Document and window features for each example, cached per model setup in data/cache/.

    With a hidden layer configured, pooled hidden-state vectors are cached alongside (one .npz per text) as
    `doc_vec` / `window_vecs`. Probe scores are never cached: they're recomputed from the vectors with the
    current calibration, so a refit never reads stale values.
    """
    import numpy as np

    scorer = detector.scorer
    slug = scorer.name.replace("/", "__")
    path = CACHE / f"{slug}__{name}.jsonl"
    vec_dir = CACHE / f"{slug}__L{scorer.hidden_layer}__{name}" if scorer.hidden_layer is not None else None
    cached = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cached[row["id"]] = row

    def has_vecs(e):
        return vec_dir is None or (vec_dir / f"{e.id}.npz").exists()

    todo = [e for e in examples if e.id not in cached or not has_vecs(e)]
    if todo:
        path.parent.mkdir(parents=True, exist_ok=True)
        if vec_dir:
            vec_dir.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for n, e in enumerate(todo, 1):
                ex = detector.extract(normalize(e.text[:12_000]).text)
                strip = lambda d: {k: v for k, v in d.items() if k != "probe"}  # noqa: E731
                row = {"id": e.id, "label": e.label, "group": e.group,
                       "document": strip(ex.document), "windows": [strip(w) for w in ex.windows]}
                if e.id not in cached:
                    f.write(json.dumps(row) + "\n")
                cached[e.id] = row
                if vec_dir and ex.doc_vec is not None:
                    dim = len(ex.doc_vec)
                    wins = np.stack([v if v is not None else np.full(dim, np.nan, np.float32) for v in ex.window_vecs]) \
                        if ex.window_vecs else np.zeros((0, dim), np.float32)
                    np.savez(vec_dir / f"{e.id}.npz", doc=ex.doc_vec.astype(np.float16), windows=wins.astype(np.float16))
                if n % 100 == 0 or n == len(todo):
                    print(f"  {name}: {n}/{len(todo)} texts scored", flush=True)

    rows = [dict(cached[e.id]) for e in examples]
    if vec_dir:
        cal = detector.calibration
        for row in rows:
            data = np.load(vec_dir / f"{row['id']}.npz")
            row["doc_vec"], row["window_vecs"] = data["doc"], data["windows"]
            if cal and cal.doc_probe:
                row["document"] = {**row["document"], "probe": cal.doc_probe.logit(row["doc_vec"])}
                row["windows"] = [{**w, "probe": cal.window_probe.logit(v) if not np.isnan(v).any() else float("nan")}
                                  for w, v in zip(row["windows"], row["window_vecs"])]
    return rows
