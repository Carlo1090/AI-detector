# DetectAI

A local web app that checks English text for AI writing **word by word**, scores every sentence the way
Turnitin's report does, and **humanizes** the flagged parts. It rewrites them, checks that the meaning
survived, and re-scores each version until it reads naturally.

Everything runs on your computer. The models download once (about 5 GB in total) and are cached after that.

## Run it (Windows)

```powershell
python -m venv .venv
# NVIDIA GPU (recommended):
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu132
# …or CPU only:  .venv\Scripts\python -m pip install torch
.venv\Scripts\python -m pip install -r requirements.txt

.venv\Scripts\python -m api
```

After setup, you can also just double-click **start.cmd**, which starts the server and opens the browser.

Open <http://127.0.0.1:8000>. Paste text, or upload or drag in a **.docx** or **.txt** file, then press
**Check text** (or Ctrl+Enter). <http://127.0.0.1:8000/#demo> runs a built-in AI-written sample, and
`#demo=humanize` runs the sample through the humanizer too.

## Detector

### What you see

| View | Shows |
|---|---|
| **Sentences** | Sentences flagged as likely AI. Each one is judged together with its neighbours, because one sentence alone is too short to judge. Hover to see why. |
| **Words** | Each word coloured by how predictable it was: the model's #1 guess, its top 10, its top 100, or a surprising word. Hover to see what the model expected instead. |
| **AI phrases** | Words, phrases and sentence shapes that AI models overuse (*delve*, *it's important to note*, "not just X, but Y", strings of em dashes…). |

The side panel shows the share of text flagged, the verdict, and a **sentence map**: a Turnitin-style colour
bar of every sentence (click a segment to jump to it). It also shows each signal's push toward "AI" or
"human", so you can see why the score came out the way it did. Texts under 150 words get no verdict.

More features:

- **Click any sentence** for its score, the reasons, and **rewrite suggestions** (Grammarly-style). Each
  suggestion shows its AI likelihood and how closely it keeps the meaning, and you can apply it in one click.
- **Download report** saves a clean printable report; choose "Save as PDF" in the print dialog.
- **Hidden-character check:** look-alike letters from other alphabets and invisible characters, both common
  ways to fool detectors, are flagged and undone before analysis.
- **How accurate is it?** shows the latest held-out evaluation next to a public detector
  (`web/accuracy.json`, rewritten by `python -m ml.eval.evaluate --compare`).
- **Recent checks** (kept only in your browser), and a light/dark/system theme toggle.

### How it scores

One pass of two small language models gives, for every token, its probability and rank, the spread of the
model's own predictions, and the base vs chat-tuned model comparison (Qwen2.5-0.5B and
Qwen2.5-0.5B-Instruct). From those:

| Signal | Meaning |
|---|---|
| Trained AI-style classifier | A logistic probe on the language model's internal state (layer 16), trained on labelled human and AI text. |
| Binoculars | Perplexity relative to cross-perplexity between the base and chat models (Hans et al., 2024). Corrects for text being predictable just because it's formal. |
| Word-choice typicality | Fast-DetectGPT probability curvature (Bao et al., 2024). |
| Predictability | Perplexity. |
| Top-10 word share | GLTR-style share of words in the model's top 10 guesses. |
| Burstiness / sentence-length variety | How much predictability and length swing between sentences. People vary more. |
| AI phrases | Weighted count of AI-typical words and phrases per 100 words ([lexicon.py](ml/detector/lexicon.py)). |

A calibrated logistic model combines them. Its thresholds are chosen so that at most ~1% of human
documents (and ~3% of human sentence windows) are flagged, separately for each kind of human writing:
student essays, English-learner essays, news, fiction, Wikipedia, medicine, finance and Q&A. Formal and
non-native writing are where detectors usually make false accusations, so averaging would hide the problem.

## Humanizer

After a check, choose a strength and tone, then press **Humanize**:

| Strength | What happens |
|---|---|
| Light | Word- and phrase-level fixes only ([rules.py](ml/humanizer/rules.py)): *utilize → use*, *plays a crucial role → plays a big part*, sentence-opening *Moreover,* dropped, contractions. Instant. |
| Balanced | Light, plus each flagged passage (up to 3 sentences) is rewritten several ways by a local model (Qwen2.5-1.5B-Instruct). Each version must keep its meaning (embedding similarity ≥ 0.80) and every name, number, citation and quotation, stay a sane length, and add no new AI phrases. The detector then picks the least AI-like version. Passages still flagged get a second round. |
| Strong | Rewrites anything borderline, more candidates, up to three rounds. |

Every change is highlighted. Click a change to switch between the new and original wording, or switch to
**Side by side** or **Clean text**. **Copy** and **Download** use your choices, and **Re-check** runs the
detector on exactly that text.

To use Claude instead of the local rewriter, set `ANTHROPIC_API_KEY` (the default model is
`claude-opus-5-5`; override it with `HUMANIZE_CLAUDE_MODEL`). This writes far better, but your text is then
sent to Anthropic's API and it costs money. `HUMANIZE_BACKEND=local|claude|off` forces a choice.

## Accuracy

See [Results](#results). The reports in `data/reports/` hold full per-source tables. Rebuild them with:

```powershell
.venv\Scripts\python -m ml.eval.calibrate --hidden-layer 16   # fit weights → ml/detector/calibrations/
.venv\Scripts\python -m ml.eval.evaluate                      # held-out detector report
.venv\Scripts\python -m ml.eval.humanize_eval                 # humanizer before/after on held-out AI text
```

The humanizer benchmark scores every text with our detector and with an **independent detector**
(`Oxidane/tmr-ai-text-detector`). The humanizer never sees the independent one, and only that one says
anything about detectors in general.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DETECT_MODEL` | `Qwen/Qwen2.5-0.5B` | Observer language model |
| `DETECT_PERFORMER` | `Qwen/Qwen2.5-0.5B-Instruct` | Chat-tuned twin for Binoculars (`none` to skip) |
| `HUMANIZE_BACKEND` | `auto` | `auto` (Claude if a key is set, else local), `local`, `claude`, `off` |
| `HUMANIZE_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Local rewriter |
| `PORT` | `8000` | Web server port |

Each model setup needs its own calibration (`python -m ml.eval.calibrate --model … --performer …`).

## Project layout

```
api/                 FastAPI server (python -m api) + static file hosting
ml/detector/
  lm.py              word-by-word statistics (+ Binoculars, + hidden states) from the language models
  text.py            sentence splitting with character offsets
  lexicon.py         AI-typical words, phrases and sentence shapes
  features.py        document + sentence-window features
  scoring.py         calibrated logistic models and the hidden-state probe
  detector.py        ties it together → JSON report
  calibrations/      fitted weights per model setup
ml/humanizer/
  rules.py           word/phrase rewrites (each edit applies on its own)
  rewriter.py        local Qwen or Claude candidate rewrites
  meaning.py         meaning checks: embeddings + must-keep names/numbers/quotes
  humanizer.py       detect → rules → rewrite → select → report
ml/eval/             datasets, calibration, evaluation, humanizer benchmark
web/                 the web page (plain HTML/CSS/JS, no build step; docx.js reads Word files in the browser)
tests/
```

## Data

- Ghostbuster essays, Reuters and WritingPrompts: Verma et al., *Ghostbuster* (2024), CC BY 3.0.
- PERSUADE 2.0 student essays (Crossley et al.), CC BY-NC-SA 4.0. Includes English-learner flags.
- DAIGT v4: Llama models fine-tuned to imitate student essays (Kaggle *LLM – Detect AI Generated Text*).
- HC3: Guo et al., *How Close is ChatGPT to Human Experts?* (2023), CC BY-SA 4.0.
- TextSight AI-detection control set (2026), CC BY-SA 4.0. Used only for testing, never for training.

All download automatically into `data/raw/` the first time you calibrate or evaluate.

## Results

_Filled in by the latest evaluation run; see `data/reports/`._
