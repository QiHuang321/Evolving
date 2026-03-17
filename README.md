# Evolving Vision-Language Inference

A compact research codebase for AlphaEvolve-style optimization of vision-language inference on the CharXiv chart question-answering benchmark.

## Overview

This repository studies whether an LLM-guided evolutionary search can improve a fixed VLM inference pipeline beyond both a naive starter baseline and careful manual engineering. The answering models are restricted to `Qwen/Qwen3-VL-2B-Instruct` and `Qwen/Qwen3-VL-2B-Thinking`; only the surrounding inference code is optimized.

## Main Results

| Variant | Accuracy | Avg. Time / Query | Summary |
| --- | ---: | ---: | --- |
| `starting_scripts` | `0.3828` | `3.4841 s` | naive baseline |
| `manual_instruct` | `0.5391` | `0.2689 s` | best pure manual system |
| `manual_thinking` | `0.5078` | `0.2752 s` | manual thinking baseline |
| `evolved_instruct` | `0.6016` | `0.3100 s` | best accuracy and best overall |
| `evolved_thinking` | `0.5469` | `0.2714 s` | evolved thinking variant |

## Repository Layout

- `starting_scripts.py`: naive starter baseline.
- `manual_instruct.py`, `manual_thinking.py`: hand-optimized inference systems.
- `evolve_instruct.py`, `evolve_thinking.py`: AlphaEvolve-style search loops using `gpt-5.2-codex` as the mutation model.
- `evolved_instruct.py`, `evolved_thinking.py`: best evolved programs discovered during search.
- `best_accuracy.py`, `best_speed.py`, `best_overall.py`: final packaged deliverables.
- `evaluate.py`: evaluation harness for exact-match accuracy and timing.
- `charxiv/`: benchmark assets and helper code.
- `report.tex`, `report.pdf`, `neurips_2024.sty`: paper source and compiled NeurIPS-style report.
- `artifacts/evolution_logs/`: archived per-candidate evolution logs.
- `notes/assignment_prompt.md`: original project brief.

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the four primary systems:

```bash
python evaluate.py manual_instruct
python evaluate.py manual_thinking
python evaluate.py evolved_instruct
python evaluate.py evolved_thinking
```

Reproduce the packaged deliverables:

```bash
bash reproduce.sh
```

Run evolutionary search again with an OpenAI API key:

```bash
export OPENAI_API_KEY="<your-api-key>"
python evolve_instruct.py
python evolve_thinking.py
```

## Notes

- All submitted inference scripts use greedy decoding for reproducibility.
- Evolution logs are intentionally kept out of the repository root to keep the deliverable surface clean.
- The original assignment README has been preserved at `notes/assignment_prompt.md`.
