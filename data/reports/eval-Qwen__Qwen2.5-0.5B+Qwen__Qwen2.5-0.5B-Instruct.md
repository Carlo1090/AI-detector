# Detector evaluation: Qwen/Qwen2.5-0.5B+Qwen/Qwen2.5-0.5B-Instruct

Run 2026-10-24. Calibrated on train-mix (3000 texts: arxiv:human, daigt:llama-student-mimic, essay:claude, essay:gpt, essay:gpt_prompt1, essay:gpt_prompt2, essay:gpt_semantic, essay:gpt_writing, essay:human, frontier:claude-opus-5.5, gen-arxiv:Qwen2.5-0.5B-Instruct, hc3-finance:chatgpt, hc3-finance:human, hc3-medicine:chatgpt, hc3-medicine:human, hc3-open_qa:chatgpt, hc3-open_qa:human, hc3-wiki_csai:chatgpt, hc3-wiki_csai:human, persuade, persuade-ell, raid-abstracts:human, raid-books:human, raid-recipes:human, raid-reddit:human, raid-wiki:human, raid:chatgpt, raid:cohere, raid:cohere-chat, raid:gpt2, raid:gpt3, raid:gpt4, raid:llama-chat, raid:mistral, raid:mistral-chat, raid:mpt, raid:mpt-chat, reuter:claude, reuter:gpt, reuter:gpt_prompt1, reuter:gpt_prompt2, reuter:gpt_semantic, reuter:gpt_writing, reuter:human, sonnet35-orca:claude-3.5-sonnet, sonnet35-science:claude-3.5-sonnet, wp:claude, wp:gpt, wp:gpt_prompt1, wp:gpt_prompt2, wp:gpt_semantic, wp:gpt_writing, wp:human).

## Summary

| Test set | AUROC | AI caught | Humans wrongly flagged | Humans wrongly flagged (150+ words) |
|---|---:|---:|---:|---:|
| Student-style essays | 0.999 | 94.8% | 0.0% | 0.0% |
| News and fiction | 0.999 | 89.2% | 0.0% | 0.0% |
| Real student essays incl. English learners | 0.943 | 52.0% | 0.0% | 0.0% |
| Formal writing: Wikipedia, medicine, finance, Q&A vs ChatGPT | 1.000 | 95.6% | 0.0% | 0.0% |
| Newer-model AI | 1.000 | 86.4% | 0.0% | 0.0% |
| RAID with adversarial attacks | 0.938 | 56.4% | 0.0% | 0.0% |
| TextSight 2026 control set | 0.952 | 33.9% | 0.9% | 0.2% |

## Student-style essays (Ghostbuster test split)

250 human, 250 AI texts. Median length 520 words.

| Metric | Value |
|---|---:|
| AUROC | 0.999 |
| AI detected at detector threshold (0.92) | 94.8% |
| Humans wrongly flagged at detector threshold | 0.0% |
| … texts of 150+ words only: AI detected / humans wrongly flagged | 94.8% / 0.0% |
| AI detected at 1% false-positive rate (oracle threshold) | 98.4% |

| Source | Label | n | Flagged as AI (document) | Mean sentence share flagged |
|---|---|---:|---:|---:|
| essay:human | human | 250 | 0.0% | 1.1% |
| essay:claude | AI | 42 | 73.8% | 77.0% |
| essay:gpt | AI | 42 | 100.0% | 99.3% |
| essay:gpt_prompt1 | AI | 41 | 100.0% | 98.6% |
| essay:gpt_prompt2 | AI | 41 | 100.0% | 98.4% |
| essay:gpt_semantic | AI | 42 | 97.6% | 97.3% |
| essay:gpt_writing | AI | 42 | 97.6% | 98.5% |

## News and fiction (Ghostbuster Reuters + WritingPrompts, held-out 30%)

250 human, 250 AI texts. Median length 497 words.

| Metric | Value |
|---|---:|
| AUROC | 0.999 |
| AI detected at detector threshold (0.92) | 89.2% |
| Humans wrongly flagged at detector threshold | 0.0% |
| … texts of 150+ words only: AI detected / humans wrongly flagged | 89.2% / 0.0% |
| AI detected at 1% false-positive rate (oracle threshold) | 98.8% |

| Source | Label | n | Flagged as AI (document) | Mean sentence share flagged |
|---|---|---:|---:|---:|
| reuter:human | human | 125 | 0.0% | 0.1% |
| wp:human | human | 125 | 0.0% | 0.7% |
| reuter:claude | AI | 21 | 52.4% | 69.1% |
| reuter:gpt | AI | 21 | 85.7% | 84.9% |
| reuter:gpt_prompt1 | AI | 21 | 100.0% | 98.6% |
| reuter:gpt_prompt2 | AI | 21 | 100.0% | 96.8% |
| reuter:gpt_semantic | AI | 21 | 100.0% | 93.9% |
| reuter:gpt_writing | AI | 21 | 100.0% | 97.8% |
| wp:claude | AI | 21 | 33.3% | 52.9% |
| wp:gpt | AI | 21 | 100.0% | 97.9% |
| wp:gpt_prompt1 | AI | 20 | 100.0% | 100.0% |
| wp:gpt_prompt2 | AI | 21 | 100.0% | 98.9% |
| wp:gpt_semantic | AI | 20 | 100.0% | 96.9% |
| wp:gpt_writing | AI | 21 | 100.0% | 99.5% |

## Real student essays incl. English learners (PERSUADE) vs student-imitating Llama (DAIGT), held-out 30%

250 human, 250 AI texts. Median length 336 words.

| Metric | Value |
|---|---:|
| AUROC | 0.943 |
| AI detected at detector threshold (0.92) | 52.0% |
| Humans wrongly flagged at detector threshold | 0.0% |
| … texts of 150+ words only: AI detected / humans wrongly flagged | 52.9% / 0.0% |
| AI detected at 1% false-positive rate (oracle threshold) | 72.0% |

| Source | Label | n | Flagged as AI (document) | Mean sentence share flagged |
|---|---|---:|---:|---:|
| persuade | human | 185 | 0.0% | 1.4% |
| persuade-ell | human | 65 | 0.0% | 0.5% |
| daigt:llama-student-mimic | AI | 250 | 52.0% | 51.5% |

## Formal writing: Wikipedia, medicine, finance, Q&A vs ChatGPT (HC3, held-out 30%)

250 human, 250 AI texts. Median length 168 words.

| Metric | Value |
|---|---:|
| AUROC | 1.000 |
| AI detected at detector threshold (0.92) | 95.6% |
| Humans wrongly flagged at detector threshold | 0.0% |
| … texts of 150+ words only: AI detected / humans wrongly flagged | 96.1% / 0.0% |
| AI detected at 1% false-positive rate (oracle threshold) | 100.0% |

| Source | Label | n | Flagged as AI (document) | Mean sentence share flagged |
|---|---|---:|---:|---:|
| hc3-finance:human | human | 75 | 0.0% | 0.2% |
| hc3-medicine:human | human | 76 | 0.0% | 0.0% |
| hc3-open_qa:human | human | 23 | 0.0% | 8.7% |
| hc3-wiki_csai:human | human | 76 | 0.0% | 1.7% |
| hc3-finance:chatgpt | AI | 62 | 93.5% | 89.3% |
| hc3-medicine:chatgpt | AI | 63 | 100.0% | 95.6% |
| hc3-open_qa:chatgpt | AI | 62 | 91.9% | 83.2% |
| hc3-wiki_csai:chatgpt | AI | 63 | 96.8% | 94.8% |

## Newer-model AI (Claude 3.5 Sonnet, Qwen2.5 abstracts) vs human academic abstracts (arXiv), held-out 30%

250 human, 250 AI texts. Median length 165 words.

| Metric | Value |
|---|---:|
| AUROC | 1.000 |
| AI detected at detector threshold (0.92) | 86.4% |
| Humans wrongly flagged at detector threshold | 0.0% |
| … texts of 150+ words only: AI detected / humans wrongly flagged | 91.9% / 0.0% |
| AI detected at 1% false-positive rate (oracle threshold) | 99.2% |

| Source | Label | n | Flagged as AI (document) | Mean sentence share flagged |
|---|---|---:|---:|---:|
| arxiv:human | human | 250 | 0.0% | 0.2% |
| frontier:claude-opus-5.5 | AI | 25 | 48.0% | 59.0% |
| gen-arxiv:Qwen2.5-0.5B-Instruct | AI | 75 | 93.3% | 84.8% |
| sonnet35-orca:claude-3.5-sonnet | AI | 75 | 98.7% | 94.8% |
| sonnet35-science:claude-3.5-sonnet | AI | 75 | 80.0% | 93.4% |

## RAID with adversarial attacks (GPT-4, ChatGPT, Cohere, Mistral, Llama...; held-out sources)

250 human, 250 AI texts. Median length 231 words.

| Metric | Value |
|---|---:|
| AUROC | 0.938 |
| AI detected at detector threshold (0.92) | 56.4% |
| Humans wrongly flagged at detector threshold | 0.0% |
| … texts of 150+ words only: AI detected / humans wrongly flagged | 55.1% / 0.0% |
| AI detected at 1% false-positive rate (oracle threshold) | 72.4% |

| Source | Label | n | Flagged as AI (document) | Mean sentence share flagged |
|---|---|---:|---:|---:|
| raid-attack:alternative_spelling:human | human | 41 | 0.0% | 0.1% |
| raid-attack:article_deletion:human | human | 41 | 0.0% | 0.0% |
| raid-attack:none:human | human | 42 | 0.0% | 0.1% |
| raid-attack:paraphrase:human | human | 41 | 0.0% | 1.2% |
| raid-attack:perplexity_misspelling:human | human | 41 | 0.0% | 1.5% |
| raid-attack:whitespace:human | human | 3 | 0.0% | 0.0% |
| raid-attack:zero_width_space:human | human | 41 | 0.0% | 0.1% |
| raid-attack:alternative_spelling:ai | AI | 23 | 60.9% | 65.5% |
| raid-attack:homoglyph:ai | AI | 23 | 60.9% | 59.6% |
| raid-attack:insert_paragraphs:ai | AI | 22 | 31.8% | 40.3% |
| raid-attack:none:ai | AI | 23 | 69.6% | 61.4% |
| raid-attack:number:ai | AI | 23 | 73.9% | 60.8% |
| raid-attack:paraphrase:ai | AI | 22 | 36.4% | 48.9% |
| raid-attack:perplexity_misspelling:ai | AI | 23 | 65.2% | 57.5% |
| raid-attack:synonym:ai | AI | 23 | 56.5% | 51.8% |
| raid-attack:upper_lower:ai | AI | 23 | 78.3% | 74.4% |
| raid-attack:whitespace:ai | AI | 22 | 36.4% | 53.0% |
| raid-attack:zero_width_space:ai | AI | 23 | 47.8% | 41.7% |

## TextSight 2026 control set (never trained on: current models, non-native academic writers)

644 human, 56 AI texts. Median length 196 words.

| Metric | Value |
|---|---:|
| AUROC | 0.952 |
| AI detected at detector threshold (0.92) | 33.9% |
| Humans wrongly flagged at detector threshold | 0.9% |
| … texts of 150+ words only: AI detected / humans wrongly flagged | 33.9% / 0.2% |
| AI detected at 1% false-positive rate (oracle threshold) | 37.5% |

| Source | Label | n | Flagged as AI (document) | Mean sentence share flagged |
|---|---|---:|---:|---:|
| human-pmc-esl | human | 183 | 0.5% | 1.0% |
| human-wikipedia-pre2022 | human | 155 | 0.0% | 2.1% |
| pd-literature | human | 306 | 1.6% | 1.4% |
| ai:claude-haiku-4-5 | AI | 12 | 66.7% | 74.7% |
| ai:claude-opus-5 | AI | 12 | 8.3% | 28.8% |
| ai:gpt-4.1 | AI | 12 | 50.0% | 69.8% |
| ai:gpt-oss-120b | AI | 12 | 33.3% | 39.8% |
| ai:qwen-3.8-27b | AI | 8 | 0.0% | 20.5% |

## Comparison: Oxidane/tmr-ai-text-detector

An independent public detector, run on the same texts at its default 0.5 cut-off.

| Test set | AUROC | AI caught | Humans wrongly flagged |
|---|---:|---:|---:|
| Student-style essays | 0.924 | 100.0% | 98.8% |
| News and fiction | 0.999 | 100.0% | 66.0% |
| Real student essays incl. English learners | 0.756 | 99.6% | 96.8% |
| Formal writing: Wikipedia, medicine, finance, Q&A vs ChatGPT | 0.991 | 100.0% | 60.0% |
| Newer-model AI | 0.993 | 98.4% | 10.8% |
| RAID with adversarial attacks | 0.993 | 97.2% | 4.0% |
| TextSight 2026 control set | 0.894 | 78.6% | 12.1% |
