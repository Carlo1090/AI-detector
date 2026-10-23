# Detector evaluation: Qwen/Qwen2.5-0.5B

Run 2026-10-23. Calibrated on train-mix (1800 texts: daigt:llama-student-mimic, essay:claude, essay:gpt, essay:gpt_prompt1, essay:gpt_prompt2, essay:gpt_semantic, essay:gpt_writing, essay:human, persuade, persuade-ell, reuter:claude, reuter:gpt, reuter:gpt_prompt1, reuter:gpt_prompt2, reuter:gpt_semantic, reuter:gpt_writing, reuter:human, wp:claude, wp:gpt, wp:gpt_prompt1, wp:gpt_prompt2, wp:gpt_semantic, wp:gpt_writing, wp:human).

## Summary

| Test set | AUROC | AI caught | Humans wrongly flagged |
|---|---:|---:|---:|
| Student-style essays | 0.994 | 97.7% | 5.7% |
| News and fiction | 0.961 | 71.0% | 0.0% |
| Real student essays incl. English learners | 0.762 | 50.0% | 0.3% |
| TextSight 2026 control set | 0.955 | 71.4% | 5.6% |

## Student-style essays (Ghostbuster test split)

300 human, 300 AI texts. Median length 528 words.

| Metric | Value |
|---|---:|
| AUROC | 0.994 |
| AI detected at detector threshold (0.77) | 97.7% |
| Humans wrongly flagged at detector threshold | 5.7% |
| AI detected at 1% false-positive rate (oracle threshold) | 86.3% |

| Source | Label | n | Flagged as AI (document) | Mean sentence share flagged |
|---|---|---:|---:|---:|
| essay:human | human | 300 | 5.7% | 5.9% |
| essay:claude | AI | 50 | 90.0% | 61.8% |
| essay:gpt | AI | 50 | 100.0% | 90.7% |
| essay:gpt_prompt1 | AI | 50 | 100.0% | 90.9% |
| essay:gpt_prompt2 | AI | 50 | 100.0% | 89.5% |
| essay:gpt_semantic | AI | 50 | 96.0% | 88.7% |
| essay:gpt_writing | AI | 50 | 100.0% | 90.8% |

## News and fiction (Ghostbuster Reuters + WritingPrompts, held-out 30%)

300 human, 300 AI texts. Median length 500 words.

| Metric | Value |
|---|---:|
| AUROC | 0.961 |
| AI detected at detector threshold (0.77) | 71.0% |
| Humans wrongly flagged at detector threshold | 0.0% |
| AI detected at 1% false-positive rate (oracle threshold) | 80.7% |

| Source | Label | n | Flagged as AI (document) | Mean sentence share flagged |
|---|---|---:|---:|---:|
| reuter:human | human | 150 | 0.0% | 3.0% |
| wp:human | human | 150 | 0.0% | 0.4% |
| reuter:claude | AI | 25 | 28.0% | 23.7% |
| reuter:gpt | AI | 25 | 48.0% | 51.2% |
| reuter:gpt_prompt1 | AI | 25 | 88.0% | 72.6% |
| reuter:gpt_prompt2 | AI | 25 | 96.0% | 71.6% |
| reuter:gpt_semantic | AI | 25 | 92.0% | 65.3% |
| reuter:gpt_writing | AI | 25 | 96.0% | 70.1% |
| wp:claude | AI | 25 | 8.0% | 13.3% |
| wp:gpt | AI | 25 | 80.0% | 42.4% |
| wp:gpt_prompt1 | AI | 25 | 88.0% | 43.9% |
| wp:gpt_prompt2 | AI | 25 | 76.0% | 42.7% |
| wp:gpt_semantic | AI | 25 | 80.0% | 38.0% |
| wp:gpt_writing | AI | 25 | 72.0% | 42.3% |

## Real student essays incl. English learners (PERSUADE) vs student-imitating Llama (DAIGT), held-out 30%

300 human, 300 AI texts. Median length 334 words.

| Metric | Value |
|---|---:|
| AUROC | 0.762 |
| AI detected at detector threshold (0.77) | 50.0% |
| Humans wrongly flagged at detector threshold | 0.3% |
| AI detected at 1% false-positive rate (oracle threshold) | 52.0% |

| Source | Label | n | Flagged as AI (document) | Mean sentence share flagged |
|---|---|---:|---:|---:|
| persuade | human | 235 | 0.4% | 3.3% |
| persuade-ell | human | 65 | 0.0% | 2.3% |
| daigt:llama-student-mimic | AI | 300 | 50.0% | 37.8% |

## TextSight 2026 control set (never trained on: current models, non-native academic writers)

644 human, 56 AI texts. Median length 196 words.

| Metric | Value |
|---|---:|
| AUROC | 0.955 |
| AI detected at detector threshold (0.77) | 71.4% |
| Humans wrongly flagged at detector threshold | 5.6% |
| AI detected at 1% false-positive rate (oracle threshold) | 17.9% |

| Source | Label | n | Flagged as AI (document) | Mean sentence share flagged |
|---|---|---:|---:|---:|
| human-pmc-esl | human | 183 | 10.4% | 12.7% |
| human-wikipedia-pre2022 | human | 155 | 7.7% | 10.2% |
| pd-literature | human | 306 | 1.6% | 2.0% |
| ai:claude-haiku-4-5 | AI | 12 | 75.0% | 58.8% |
| ai:claude-opus-5 | AI | 12 | 75.0% | 59.0% |
| ai:gpt-4.1 | AI | 12 | 100.0% | 73.8% |
| ai:gpt-oss-120b | AI | 12 | 75.0% | 43.4% |
| ai:qwen-3.8-27b | AI | 8 | 12.5% | 11.3% |
