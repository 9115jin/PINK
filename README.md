# PINK: Penalized INK-based Score

**When VLMs ‘Fix’ Students: Identifying and Penalizing Over-Correction in the Evaluation of Multi-line Handwritten Math OCR**
<br>Jin Seong\*, Wencke Liermann\*, Minho Kim, Jong-hun Shin, Soojong Lim · *EMNLP 2026 (Main, Oral)*

[![Paper](https://img.shields.io/badge/Paper-ACL%20Anthology-red)](https://aclanthology.org/2026.emnlp-main.476/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.22774-b31b1b.svg)](https://arxiv.org/abs/2604.22774)
[![HF Papers](https://img.shields.io/badge/%F0%9F%A4%97-Paper%20Page-yellow)](https://huggingface.co/papers/2604.22774)
[![lmms-eval](https://img.shields.io/badge/lmms--eval-fermat__pink-green)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks/fermat_pink)
[![PyPI](https://img.shields.io/pypi/v/pink-eval)](https://pypi.org/project/pink-eval/)
[![Tests](https://github.com/9115jin/PINK/actions/workflows/tests.yml/badge.svg)](https://github.com/9115jin/PINK/actions/workflows/tests.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://github.com/9115jin/PINK/blob/main/LICENSE)

## News

- **2026-11**: [Leaderboard](#fermat-leaderboard-recent-vlms-and-ocr-models) of 22 recent VLMs and OCR models on FERMAT, graded with the open Qwen3-Next-80B judge.
- **2026-11**: FERMAT-PINK is a built-in task of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) (`--tasks fermat_pink`). The paper's GPT-5 and Qwen3 oracle grades now ship with `pip install pink-eval`.
- **2026-10**: The paper was presented as an oral at EMNLP 2026 (Budapest).

Vision-language models transcribing a student's handwritten solution often **“fix” the student's mistakes** instead of copying them. The transcription looks good, and string metrics like BLEU barely notice it. But the error a teacher or tutoring system needed to see is gone. We call this **over-correction**. Across 15 VLMs, 42–66% of transcriptions contain at least one over-correction.

**PINK** is an OCR metric built to catch it:

1. An LLM judge grades a transcription with a 5-component, 100-point rubric. This measures *semantic* fidelity, so `\frac{1}{2}` and `0.5` count as the same.
2. The judge also grades the *oracle* transcription, i.e. what the student actually wrote, errors included.
3. Any rubric component on which the transcription scores **higher than the oracle** is an over-correction. It is reverted to the oracle score if the gain is ≤ 10 points and set to 0 if the gain is larger.

![PINK overview](https://raw.githubusercontent.com/9115jin/PINK/main/assets/pink_overview.png)

## Installation

```bash
pip install "pink-eval[fermat]"
# latest development version:  pip install "pink-eval[fermat] @ git+https://github.com/9115jin/PINK.git"
```

Python ≥ 3.10. `[fermat]` adds the Hugging Face `datasets` package, which you need only to download FERMAT.

## Quickstart (offline, no API key)

```bash
git clone https://github.com/9115jin/PINK.git && cd PINK && pip install -e .
python examples/quickstart.py
```

This scores Qwen2.5-VL-7B on 30 FERMAT samples whose transcriptions were already graded:

```
PINK                 = 0.7400
raw (no penalty)     = 0.9657
over-correction rate = 63.3%
```

The raw ratio is the judge score of the transcription divided by the judge score of the oracle, with no penalty. Here it is close to 1, so the transcription looks almost as good as the student's original. PINK shows that much of that score comes from fixed student errors.

## Evaluate your VLM on FERMAT

FERMAT ([Nath et al., 2025](https://aclanthology.org/2025.acl-long.720/)) contains 2,244 handwritten multi-line solutions with known student errors. It is hosted on Hugging Face under CC-BY-4.0. The dataset is gated, so first accept its terms at [huggingface.co/datasets/ai4bharat/FERMAT](https://huggingface.co/datasets/ai4bharat/FERMAT), then run `huggingface-cli login`.

```bash
# 1) Download FERMAT. Writes images/ and manifest.jsonl (problem + oracle transcription per sample).
pink fermat --out-dir data/fermat                 # add --limit 20 for a quick try

# 2) Transcribe the images with your VLM. Any OpenAI-compatible endpoint works, e.g. `vllm serve`.
pink ocr --manifest data/fermat/manifest.jsonl --model Qwen/Qwen2.5-VL-7B-Instruct \
         --base-url http://localhost:8000/v1 --out runs/qwen7b/ocr.jsonl

# 3) Grade the transcriptions with the GPT-5 judge.
export OPENAI_API_KEY=...
pink grade --judge gpt-5 --input runs/qwen7b/ocr.jsonl --out runs/qwen7b/grades.csv

# 4) Score them against the paper's GPT-5 oracle grades, which ship with the package.
pink score --model runs/qwen7b/grades.csv --oracle fermat:gpt-5
```

The bundled `fermat:gpt-5` oracle grades are the ones behind the paper's results. With `--judge gpt-5`, your PINK is directly comparable to the [FERMAT results](#fermat-results-gpt-5-judge-paper-table-1) below, and you only pay for grading your own model's transcriptions.

Dedicated OCR models usually expect their own prompt. Pass it with `--user-prompt` (and `--system-prompt ''` to send no system message); add `--image-first` if the model expects the image before the text. Leading YAML front matter, as written by olmOCR, is removed before grading.

For hybrid reasoning models (e.g. Qwen3.5 or Qwen3.8 on vLLM), add `--extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'` to `pink ocr` to transcribe without a reasoning trace. Any `<think>…</think>` block left in an output is removed before grading.

All commands run requests in parallel (`--workers`), retry failed calls, and resume where they stopped if you re-run them. Already-finished samples are skipped. You can also transcribe with your own pipeline: `pink grade` only needs a JSONL file with `sample_id`, `problem` and `solution` (your transcription).

> **Grade the model and the oracle with the same judge and prompt.** PINK compares the two scores component by component. For any judge other than the two bundled ones, grade the oracle once and reuse the result for every VLM: `pink grade --judge <judge> --input data/fermat/manifest.jsonl --out runs/oracle_grades.csv`. For this you can export the text without the images: `pink fermat --text-only`.

### Judges

| `--judge` | Model | Notes |
|---|---|---|
| `gpt-5` | `gpt-5-2025-08-07`, reasoning effort `medium` | Main judge in the paper. Uses the OpenAI API with the exact instructions of the paper's runs. Bundled oracle grades: `fermat:gpt-5`. |
| `qwen3-next-80b` | `Qwen/Qwen3-Next-80B-A3B-Thinking` (temperature 0.6, top-p 0.95) | Open-source judge. VLM-level PINK agrees with GPT-5 at r = 0.99 and Kendall τ = 0.90 (paper App. E). Serve it with [`scripts/serve_qwen3_judge.sh`](https://github.com/9115jin/PINK/blob/main/scripts/serve_qwen3_judge.sh). Bundled oracle grades: `fermat:qwen3-next-80b`. |

Any other chat model also works, e.g. `--model <name> --base-url <url>`. PINK values are comparable only when they come from the same judge.

To run the open-source judge yourself:

```bash
bash scripts/serve_qwen3_judge.sh                                          # 8x 80GB GPUs (BF16)
MODEL=Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 TP=4 bash scripts/serve_qwen3_judge.sh   # 4x 48GB GPUs (FP8)
pink grade --judge qwen3-next-80b --input runs/qwen7b/ocr.jsonl --out runs/qwen7b/grades_qwen.csv
pink score --model runs/qwen7b/grades_qwen.csv --oracle fermat:qwen3-next-80b
```

The judge reasons for several thousand tokens per sample. If grades fail with `finish_reason=length`, give the server a longer context (`MAX_LEN`, default 16384).

### With lmms-eval

FERMAT-PINK is a built-in task of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) (`fermat_pink`). Until it ships in an lmms-eval release, install lmms-eval from its `main` branch.

```bash
pip install pink-eval
export OPENAI_API_KEY=...            # default judge: GPT-5, scored against the bundled fermat:gpt-5 oracle grades
python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct --tasks fermat_pink
```

The task reports `pink`, `over_correction_rate` and `raw_score_ratio`. To use the open-source judge, set `PINK_JUDGE=qwen3-next-80b PINK_JUDGE_BASE_URL=http://localhost:8000/v1`. See the [task README](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks/fermat_pink) for all options.

## Use PINK on your own data

PINK is not tied to FERMAT. For each sample you need the **problem**, the **oracle transcription** (what is actually written) and the **model transcription**:

```jsonl
{"sample_id": "q1", "problem": "Factorise 4y^2 - 12y + 9.", "solution": "<oracle transcription>"}
```

Write one such file for the oracle and one for the model (same `sample_id`s), run `pink grade` on each, then run `pink score`. From Python:

```python
from pink import compute_pink, penalize

result = compute_pink("runs/model_grades.csv", "runs/oracle_grades.csv")
print(result.pink, result.over_correction_rate)

penalize(model_scores=[20, 20, 12], oracle_scores=[20, 5, 10])  # -> [20, 0, 10]
```

## FERMAT leaderboard: recent VLMs and OCR models

22 models evaluated with one pipeline on all 2,244 FERMAT samples: `pink ocr`, then `pink grade --judge qwen3-next-80b`, then `pink score --oracle fermat:qwen3-next-80b --missing zero`. Models marked † come from the paper. Their original transcriptions were re-graded with the same judge, so they link this table to the paper.

Scores depend on the judge. Compare a model with the rows of this table, not with the GPT-5 table further down.

| # | Model | Params | Type | PINK ↑ | 95% CI | Over-correction rate | Raw (no penalty) |
|---:|---|---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 36.0B (MoE, 3B active) | VLM | **0.928** | 0.921–0.934 | 41.8% | 1.084 |
| 2 | [Gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | 31.3B | VLM | **0.921** | 0.914–0.928 | 45.6% | 1.104 |
| 3 | Gemini 2.5 Flash † | API | VLM | **0.920** | 0.913–0.927 | 37.4% | 1.040 |
| 4 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27.8B | VLM | **0.916** | 0.909–0.923 | 39.3% | 1.057 |
| 5 | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9.7B | VLM | **0.913** | 0.906–0.920 | 37.0% | 1.035 |
| 6 | [Qwen2.5-VL-72B †](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) | 73.4B | VLM | **0.913** | 0.905–0.921 | 50.2% | 1.122 |
| 7 | [Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | 29.8B (reasoning strength low) | VLM | **0.913** | 0.906–0.919 | 35.2% | 1.017 |
| 8 | [Chandra-OCR-2](https://huggingface.co/datalab-to/chandra-ocr-2) | 5.3B | OCR | **0.912** | 0.905–0.919 | 40.0% | 1.047 |
| 9 | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 8.8B | VLM | **0.910** | 0.903–0.918 | 40.0% | 1.048 |
| 10 | GPT-4o † | API | VLM | **0.896** | 0.888–0.904 | 44.6% | 1.073 |
| 11 | [olmOCR-2-7B](https://huggingface.co/allenai/olmOCR-2-7B-1025) | 8.3B | OCR | **0.887** | 0.878–0.895 | 32.8% | 0.978 |
| 12 | [InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) | 8.5B | VLM | **0.859** | 0.849–0.869 | 32.5% | 0.951 |
| 13 | [Nanonets-OCR2-3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B) | 3.8B | OCR | **0.845** | 0.835–0.855 | 30.5% | 0.925 |
| 14 | SciEagle-Granite-9B (ETRI preview, weights not yet public) | 8.8B | VLM | **0.828** | 0.816–0.839 | 39.5% | 0.963 |
| 15 | [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 1.3B | OCR | **0.825** | 0.814–0.836 | 27.9% | 0.891 |
| 16 | [Qwen2.5-VL-7B †](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 8.3B | VLM | **0.788** | 0.774–0.802 | 32.6% | 0.883 |
| 17 | [dots.ocr](https://huggingface.co/dots-studio/dots.ocr) | 3.0B | OCR | **0.775** | 0.764–0.787 | 26.7% | 0.843 |
| 18 | [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | 959M | OCR | **0.722** | 0.709–0.735 | 24.6% | 0.781 |
| 19 | [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) | 3.3B | OCR | **0.690** | 0.678–0.703 | 24.3% | 0.749 |
| 20 | [Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2) | 3.3B | OCR | **0.675** | 0.663–0.690 | 24.5% | 0.733 |
| 21 | [d1-2B](https://huggingface.co/LiquidAI/d1-2B) | 1.9B | VLM | **0.538** | 0.521–0.555 | 26.3% | 0.611 |
| 22 | [DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) | 3.4B | OCR | **0.532** | 0.516–0.547 | 21.9% | 0.592 |

<details><summary><b>By model size</b></summary>

**≤ 4B**

| # | Model | Params | Type | PINK ↑ | 95% CI | Over-correction rate | Raw (no penalty) |
|---:|---|---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [Nanonets-OCR2-3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B) | 3.8B | OCR | **0.845** | 0.835–0.855 | 30.5% | 0.925 |
| 2 | [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 1.3B | OCR | **0.825** | 0.814–0.836 | 27.9% | 0.891 |
| 3 | [dots.ocr](https://huggingface.co/dots-studio/dots.ocr) | 3.0B | OCR | **0.775** | 0.764–0.787 | 26.7% | 0.843 |
| 4 | [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | 959M | OCR | **0.722** | 0.709–0.735 | 24.6% | 0.781 |
| 5 | [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) | 3.3B | OCR | **0.690** | 0.678–0.703 | 24.3% | 0.749 |
| 6 | [Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2) | 3.3B | OCR | **0.675** | 0.663–0.690 | 24.5% | 0.733 |
| 7 | [d1-2B](https://huggingface.co/LiquidAI/d1-2B) | 1.9B | VLM | **0.538** | 0.521–0.555 | 26.3% | 0.611 |
| 8 | [DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) | 3.4B | OCR | **0.532** | 0.516–0.547 | 21.9% | 0.592 |

**4–10B**

| # | Model | Params | Type | PINK ↑ | 95% CI | Over-correction rate | Raw (no penalty) |
|---:|---|---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9.7B | VLM | **0.913** | 0.906–0.920 | 37.0% | 1.035 |
| 2 | [Chandra-OCR-2](https://huggingface.co/datalab-to/chandra-ocr-2) | 5.3B | OCR | **0.912** | 0.905–0.919 | 40.0% | 1.047 |
| 3 | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 8.8B | VLM | **0.910** | 0.903–0.918 | 40.0% | 1.048 |
| 4 | [olmOCR-2-7B](https://huggingface.co/allenai/olmOCR-2-7B-1025) | 8.3B | OCR | **0.887** | 0.878–0.895 | 32.8% | 0.978 |
| 5 | [InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) | 8.5B | VLM | **0.859** | 0.849–0.869 | 32.5% | 0.951 |
| 6 | SciEagle-Granite-9B (ETRI preview, weights not yet public) | 8.8B | VLM | **0.828** | 0.816–0.839 | 39.5% | 0.963 |
| 7 | [Qwen2.5-VL-7B †](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 8.3B | VLM | **0.788** | 0.774–0.802 | 32.6% | 0.883 |

**10–40B**

| # | Model | Params | Type | PINK ↑ | 95% CI | Over-correction rate | Raw (no penalty) |
|---:|---|---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 36.0B (MoE, 3B active) | VLM | **0.928** | 0.921–0.934 | 41.8% | 1.084 |
| 2 | [Gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | 31.3B | VLM | **0.921** | 0.914–0.928 | 45.6% | 1.104 |
| 3 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27.8B | VLM | **0.916** | 0.909–0.923 | 39.3% | 1.057 |
| 4 | [Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | 29.8B (reasoning strength low) | VLM | **0.913** | 0.906–0.919 | 35.2% | 1.017 |

**> 40B**

| # | Model | Params | Type | PINK ↑ | 95% CI | Over-correction rate | Raw (no penalty) |
|---:|---|---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [Qwen2.5-VL-72B †](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) | 73.4B | VLM | **0.913** | 0.905–0.921 | 50.2% | 1.122 |

**API (undisclosed)**

| # | Model | Params | Type | PINK ↑ | 95% CI | Over-correction rate | Raw (no penalty) |
|---:|---|---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Gemini 2.5 Flash † | API | VLM | **0.920** | 0.913–0.927 | 37.4% | 1.040 |
| 2 | GPT-4o † | API | VLM | **0.896** | 0.888–0.904 | 44.6% | 1.073 |

</details>

<details><summary><b>General VLMs vs. dedicated OCR models</b></summary>

**General-purpose VLMs**

| # | Model | Params | PINK ↑ | 95% CI | Over-correction rate | Raw (no penalty) |
|---:|---|---:|:---:|:---:|:---:|:---:|
| 1 | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 36.0B (MoE, 3B active) | **0.928** | 0.921–0.934 | 41.8% | 1.084 |
| 2 | [Gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | 31.3B | **0.921** | 0.914–0.928 | 45.6% | 1.104 |
| 3 | Gemini 2.5 Flash † | API | **0.920** | 0.913–0.927 | 37.4% | 1.040 |
| 4 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27.8B | **0.916** | 0.909–0.923 | 39.3% | 1.057 |
| 5 | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9.7B | **0.913** | 0.906–0.920 | 37.0% | 1.035 |
| 6 | [Qwen2.5-VL-72B †](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) | 73.4B | **0.913** | 0.905–0.921 | 50.2% | 1.122 |
| 7 | [Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | 29.8B (reasoning strength low) | **0.913** | 0.906–0.919 | 35.2% | 1.017 |
| 8 | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 8.8B | **0.910** | 0.903–0.918 | 40.0% | 1.048 |
| 9 | GPT-4o † | API | **0.896** | 0.888–0.904 | 44.6% | 1.073 |
| 10 | [InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) | 8.5B | **0.859** | 0.849–0.869 | 32.5% | 0.951 |
| 11 | SciEagle-Granite-9B (ETRI preview, weights not yet public) | 8.8B | **0.828** | 0.816–0.839 | 39.5% | 0.963 |
| 12 | [Qwen2.5-VL-7B †](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 8.3B | **0.788** | 0.774–0.802 | 32.6% | 0.883 |
| 13 | [d1-2B](https://huggingface.co/LiquidAI/d1-2B) | 1.9B | **0.538** | 0.521–0.555 | 26.3% | 0.611 |

**Dedicated OCR / document-parsing models**

| # | Model | Params | PINK ↑ | 95% CI | Over-correction rate | Raw (no penalty) |
|---:|---|---:|:---:|:---:|:---:|:---:|
| 1 | [Chandra-OCR-2](https://huggingface.co/datalab-to/chandra-ocr-2) | 5.3B | **0.912** | 0.905–0.919 | 40.0% | 1.047 |
| 2 | [olmOCR-2-7B](https://huggingface.co/allenai/olmOCR-2-7B-1025) | 8.3B | **0.887** | 0.878–0.895 | 32.8% | 0.978 |
| 3 | [Nanonets-OCR2-3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B) | 3.8B | **0.845** | 0.835–0.855 | 30.5% | 0.925 |
| 4 | [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 1.3B | **0.825** | 0.814–0.836 | 27.9% | 0.891 |
| 5 | [dots.ocr](https://huggingface.co/dots-studio/dots.ocr) | 3.0B | **0.775** | 0.764–0.787 | 26.7% | 0.843 |
| 6 | [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | 959M | **0.722** | 0.709–0.735 | 24.6% | 0.781 |
| 7 | [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) | 3.3B | **0.690** | 0.678–0.703 | 24.3% | 0.749 |
| 8 | [Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2) | 3.3B | **0.675** | 0.663–0.690 | 24.5% | 0.733 |
| 9 | [DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) | 3.4B | **0.532** | 0.516–0.547 | 21.9% | 0.592 |

</details>

- **Over-correction rate** is the share of samples with at least one over-corrected rubric component. **Raw** is the score before the penalty, Σ model / Σ oracle. A raw score above 1 means the transcriptions graded *better* than what the students actually wrote.
- **95% CI** is a bootstrap over samples. Models with overlapping intervals are not reliably different.
- **What stands out**
  - Over-correction has not gone away. General-purpose VLMs over-correct 26–50% of the solutions, and all eight that score above 0.91 sit at 35–50%.
  - Most dedicated OCR models over-correct less (22–28%), but they also misread more handwriting, so they lose content instead.
  - OCR models fine-tuned from a strong VLM (Chandra-OCR-2, olmOCR-2, Nanonets-OCR2) keep roughly the over-correction level of their base model.
- **Setup**
  - Hardware and decoding: 4× L40S, vLLM 0.19.1 (transformers for models vLLM cannot serve), greedy decoding, at most 4,096 new tokens.
  - Prompts: general VLMs use the paper's OCR prompt (Fig. 18). Dedicated OCR models use the prompt from their model card.
  - Reasoning: thinking is switched off wherever the model has a switch. Muse-Glimmer-30B has none and runs at its lowest reasoning strength.
- **Clean-up**
  - Only layout markup is removed: the layout tags of Unlimited-OCR-2, the HTML of Chandra-OCR-2 (converted to Markdown as in Chandra's own pipeline), and olmOCR's front matter.
  - Empty outputs score 0. So do repetition loops too long for the judge's 16k context, which mostly affects d1-2B, Unlimited-OCR-2 and DeepSeek-OCR-2.
  - DeepSeek-OCR-2 runs without vLLM's n-gram repetition processor.
- **Not included:** NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB. Its packed 2-bit weights need vLLM nightly with Blackwell kernels.

## FERMAT results (GPT-5 judge, paper Table 1)

| # | Model | PINK ↑ | Over-correction rate |
|---:|---|:---:|:---:|
| 1 | Gemini 2.5 Flash | 0.935 | 59.3% |
| 2 | Qwen2.5-VL 72B | 0.920 | 65.7% |
| 3 | Qwen2.5-VL 32B | 0.919 | 66.2% |
| 4 | Ovis2 34B | 0.916 | 62.5% |
| 5 | InternVL3 78B | 0.911 | 58.5% |
| 6 | GPT-4o | 0.907 | 62.3% |
| 7 | InternVL3 38B | 0.905 | 58.8% |
| 8 | Ovis2 8B | 0.887 | 53.6% |
| 9 | InternVL3 8B | 0.873 | 50.8% |
| 10 | Pixtral Large (124B) | 0.850 | 56.1% |
| 11 | Qwen2.5-VL 7B | 0.832 | 54.6% |
| 12 | Llama 3.2 90B Vision | 0.814 | 50.1% |
| 13 | GPT-4o mini | 0.787 | 49.6% |
| 14 | Pixtral 12B | 0.722 | 43.9% |
| 15 | Llama 3.2 11B Vision | 0.676 | 42.1% |

Over-correction rate = share of transcriptions with at least one over-corrected rubric component (paper Fig. 14). `pink score` recomputes these numbers from the paper's grade files with `--oracle fermat:gpt-5` (max. difference 5e-5).
The printed Table 1 in the paper has the two Pixtral PINK values swapped. The values above are the correct ones, consistent with Figures 3, 6 and 15 of the paper.

## Notes

- **Prompts.** The OCR prompt (`pink.ocr.PROMPTS["default"]`) is Fig. 18. `--prompt faithful` selects the faithfulness-instructed variant from Fig. 19 (App. F).
- **Judge prompts.** The `gpt-5` preset sends `pink.rubric.GPT5_PAPER_INSTRUCTIONS`, byte-for-byte the instructions of the paper's GPT-5 runs: the rubric and output format of Fig. 16, without its "Grading Guidelines" block. The `qwen3-next-80b` preset and custom judges use the full Fig. 16 prompt (`pink.rubric.GRADING_INSTRUCTIONS`).
- **Bundled oracle grades.** `fermat:gpt-5` contains the GPT-5 oracle grades from the paper. `fermat:qwen3-next-80b` was graded with this package's `qwen3-next-80b` preset: official FP8 weights, vLLM 0.13.0, `--max-model-len 16384`, all 2,244 samples graded. Its sample-level agreement with GPT-5 is r = 0.78, the same as reported in the paper.
- **Oracle transcription.** We use FERMAT's `pert_a` (the student's erroneous solution) and remove its `\textcolor{red}{…}` error markers. Otherwise the judge would see where the error is.
- **Formula.** PINK = Σ<sub>n</sub> S<sub>penalized,n</sub> / Σ<sub>n</sub> S<sub>oracle,n</sub> (paper Eq. 4). Here S<sub>oracle,n</sub> is the sum of the oracle's component scores. A transcription that scores the same as the oracle gets exactly 1.0. Samples that the judge could not grade are left out (`--missing zero` counts them as 0 instead).

## Citation

```bibtex
@inproceedings{seong-etal-2026-vlms,
    title = "When {VLM}s `Fix' Students: Identifying and Penalizing Over-Correction in the Evaluation of Multi-line Handwritten Math {OCR}",
    author = "Seong, Jin and Liermann, Wencke and Kim, Minho and Shin, Jong-hun and Lim, Soojong",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.476/",
    doi = "10.18653/2026.emnlp-main.476",
    pages = "10932--10953"
}
```

If you evaluate on FERMAT, please also cite the dataset:

```bibtex
@inproceedings{nath-etal-2025-vision,
    title = "Can Vision-Language Models Evaluate Handwritten Math?",
    author = "Nath, Oikantik and Bathina, Hanani and Khan, Mohammed Safi Ur Rahman and Khapra, Mitesh M",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.720/",
    doi = "10.18653/v1/2025.acl-long.720",
    pages = "14784--14814"
}
```

## License

Code: [BSD-3-Clause](https://github.com/9115jin/PINK/blob/main/LICENSE), © 2026 Electronics and Telecommunications Research Institute (ETRI).
The bundled oracle grades (`src/pink/resources/oracle_grades/`) are judge scores of FERMAT samples. The example grade files in `examples/sample/` also contain judge justifications that quote FERMAT samples. FERMAT is © its authors and licensed under CC-BY-4.0.
