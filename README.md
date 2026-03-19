# Evolving Vision-Language Inference

AlphaEvolve-style evolutionary code optimization for VLM chart question answering on the [CharXiv](https://github.com/princeton-nlp/CharXiv) benchmark.

## Results

All numbers are on the 128-sample CharXiv descriptive validation subset with greedy decoding.

| Variant | Accuracy | Avg Time (s/q) | Description |
| --- | ---: | ---: | --- |
| `starting_scripts` | 0.3828 | 3.484 | Naive baseline |
| `manual_instruct` | 0.5391 | 0.269 | Prompt engineering + output normalization |
| `manual_thinking` | 0.5078 | 0.275 | Thinking model, suppressed CoT |
| `evolved_instruct` | 0.6094 | 0.261 | Evolutionary search (40 gen) |
| `evolved_thinking` | 0.5547 | 0.296 | Evolutionary search (Thinking) |
| **`best_accuracy`** | **0.7969** | 0.307 | Evolved + bug fixes + title & colorbar verifiers |
| **`best_overall`** | **0.7813** | 0.305 | Evolved + bug fixes + colorbar verifier |
| `best_speed` | 0.5781 | 0.196±0.014 | Per-image batched inference, no preprocessing |

### 4-Fold Stability Check

To check whether gains are concentrated on a lucky subset (same fixed
program evaluated on non-overlapping quarters — not re-derived per fold):

| Variant | Mean ± Std | Per-fold |
| --- | --- | --- |
| `best_accuracy` | 0.797 ± 0.031 | 0.8125, 0.7500, 0.8125, 0.8125 |
| `best_overall` | 0.781 ± 0.026 | 0.7813, 0.7500, 0.8125, 0.7813 |

### Accuracy Progression

Each row shows the cumulative accuracy along the `best_accuracy` branch
(deltas are relative to the previous row, not a fixed baseline):

```
Baseline              0.3828  ████████
Manual instruct       0.5391  ███████████
Evolved instruct      0.6094  ████████████
+ Post-proc fixes     0.6953  ██████████████
+ Colorbar verifier   0.7969  ████████████████
```

## Repository Structure

```
├── starting_scripts.py          # Naive baseline (teacher-provided)
├── manual_instruct.py           # Hand-optimized (Instruct model)
├── manual_thinking.py           # Hand-optimized (Thinking model)
│
├── evolve_instruct.py           # Evolution system (Instruct)
├── evolve_thinking.py           # Evolution system (Thinking)
├── evolved_instruct.py          # Best program found by evolution (Instruct)
├── evolved_thinking.py          # Best program found by evolution (Thinking)
│
├── best_accuracy.py             # Final submission: highest accuracy
├── best_overall.py              # Final submission: best accuracy/speed balance
├── best_speed.py                # Final submission: fastest inference
│
├── evaluate.py                  # Evaluation harness (exact-match, 128 samples)
├── reproduce.sh                 # Run all evaluations
├── requirements.txt             # Python dependencies
│
├── report.tex                   # NeurIPS-format paper source
├── report.pdf                   # Compiled paper
├── neurips_2024.sty             # LaTeX style file
│
├── charxiv/                     # CharXiv benchmark (data + evaluation code)
│   ├── data/                    #   JSON question/answer files
│   ├── images/                  #   Chart images (downloaded separately)
│   └── src/                     #   Evaluation utilities
│
├── artifacts/                   # (gitignored) Evolution logs
│   └── evolution_logs/
│       ├── evolve_instruct_log.jsonl
│       └── evolve_thinking_log.jsonl
│
└── notes/                       # (gitignored) Working notes
    └── assignment_prompt.md
```

## Key Techniques

### Evolution System (`evolve_instruct.py`, `evolve_thinking.py`)
- **LLM-guided mutation**: GPT-5.2-codex generates code variants
- **Dual-mode mutation**: 50% full-block rewrite, 50% focused single-function edit via `ast.parse()`
- **Cascaded evaluation**: regression tests → 1-sample crash check → 16-sample stratified pre-screen → full 128-sample eval
- **Shared model injection**: VLM loaded once, injected into candidates (eliminates ~25s/candidate model reload)
- **Behavioral diversity archive**: MAP-Elites grid indexed by per-question-type accuracy bins + NA precision/recall
- **Island model**: 2 islands × 4 programs, periodic migration

### Post-Processing Bug Fixes (+9.4 pp)
1. Negative sign stripping in candidate extraction
2. Positional tick text fallback (non-numeric tick labels)
3. Embedded number regex over-extraction
4. Caret scientific notation preservation (`10^-6`)
5. Legend count deduplication from label lists

### Targeted Verifiers (+11.8 pp cumulative from post-proc fixes)
- **Title verifier**: 3-view TTA for panel-marker detection (+1.6 pp over 0.6953 post-proc baseline)
- **Colorbar verifier**: single-probe "Does this chart have a colorbar?" (+10.2 pp; applied after title verifier)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download CharXiv images (required for evaluation)
cd charxiv/images
# Option A: if you have the CharXiv images tarball
tar xzf charxiv_images.tar.gz
# Option B: download from the CharXiv repo
#   See charxiv/images/README.md for instructions
cd ../..

# Evaluate a single script
python evaluate.py best_accuracy

# 4-fold stability check
python evaluate.py best_accuracy --cv 4

# Evaluate on a specific fold
python evaluate.py best_accuracy --fold 2/4

# Run all evaluations (accuracy + stability + speed benchmark)
bash reproduce.sh

# Re-run evolution (requires OpenAI API key)
export OPENAI_API_KEY="sk-..."
python evolve_instruct.py
python evolve_thinking.py
```

## Models

- **Answering model**: `Qwen/Qwen3-VL-2B-Instruct` / `Qwen/Qwen3-VL-2B-Thinking` (fixed, bfloat16, greedy)
- **Mutation model**: `gpt-5.2-codex` via OpenAI API (for evolution only)

## Citation

See `report.pdf` for the full writeup.
