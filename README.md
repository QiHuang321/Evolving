# Evolving VLM Inference for CharXiv

> AlphaEvolve-style code optimization for chart question answering with Qwen3-VL-2B.
>
> **Evolution-only:** 38.3% → 60.9% dev exact-match (no per-sample inspection).
> **Post-analysis dev-best:** 79.7% exact-match (colorbar verifier).
> **Best warm-start speed:** 0.196 ± 0.014 s/query.

[Paper (PDF)](./report.pdf) · [Reproduce](#reproduce) · [Results](#main-results) · [Method](#method) · [Limitations](#limitations)

## TL;DR

- We implement an **AlphaEvolve-style mutation–evaluation–selection loop** that evolves the Python `vlm_inference` function for chart QA.
- Evolution alone improves the teacher baseline from **0.3828** to **0.6094** — a clean +22.7 pp gain with no per-sample inspection.
- Manual post-processing bug fixes push accuracy to **0.6953** (+8.6 pp).
- A targeted **colorbar-existence verifier** fixes the largest remaining hallucination mode and reaches **0.7969** on the 128-sample dev subset.
- An **Instruct/Thinking question-type router** was attempted but is a **negative result** (0.75): the colorbar verifier already absorbs the Thinking model's NA-detection advantage.
- All headline numbers are on a **128-example development subset**; held-out performance may differ.

## Main Results

All numbers are on the 128-sample CharXiv descriptive validation subset with greedy decoding.

| Variant | Accuracy | Time (s/q) | What changed |
| --- | ---: | ---: | --- |
| `starting_scripts` | 0.3828 | 3.484 | teacher baseline |
| `manual_instruct` | 0.5391 | 0.269 | prompt + normalization |
| `evolved_instruct` | 0.6094 | 0.261 | **evolution only** (40 gen) |
| `best_accuracy` | **0.7969** | 0.307 | evolved v1 + post-proc fixes + verifiers |
| `best_accuracy_v3` | 0.7500 | 0.730 | Instruct/Thinking router (**negative result**) |
| `best_overall` | 0.7813 | 0.305 | accuracy/speed balance |
| `best_speed` | 0.5781 | 0.196 ± 0.014† | per-image batched, warm-start |

<sub>†Warm-start mean±std over 6 back-to-back runs; single cold-start run: 0.46–0.55 s/query.</sub>

### Two-Layer Headline

We report results in two tiers to separate evolution-driven gains from post-analysis tuning:

| Tier | Range | How |
| --- | --- | --- |
| **Evolution-only** | 0.3828 → 0.6094 | LLM mutation loop, no per-sample inspection |
| **Post-analysis dev-best** | 0.6094 → 0.7969 | manual error analysis, verifiers |

The evolution-only chain is the more conservative generalization estimate. The post-analysis gains carry inherent overfitting risk to the 128-sample dev set.

### 4-Fold Stability Check

Same fixed program evaluated on non-overlapping quarters (not re-derived per fold):

| Variant | Mean ± Std | Per-fold |
| --- | --- | --- |
| `best_accuracy` | 0.797 ± 0.031 | 0.8125, 0.7500, 0.8125, 0.8125 |
| `best_overall` | 0.781 ± 0.026 | 0.7813, 0.7500, 0.8125, 0.7813 |

Grouped-by-figure CV (via `--grouped`) is also supported — see [Reproduce](#reproduce).

## Why This Works

1. **Exact-match chart QA is bottlenecked by formatting and hallucination**, not raw perception. Post-processing fixes alone yield +8.6 pp.
2. **Evolution discovers better prompts and normalization** than manual tuning: expanded NA synonyms, "answer is" extractors, question-type gating.
3. **Narrow binary verifiers** ("does a colorbar exist?") are far more reliable than direct value extraction from uncertain visual objects — a single probe corrects 13/39 remaining errors.
4. **Verifier-augmented prompting absorbs multi-model advantages**: Instruct+colorbar-verifier scores 16/16 on continuous-legend questions, eliminating the gap that motivated a Thinking-model router (negative result, see `best_accuracy_v3.py`).

## Method

### Evolution System (`evolve_instruct.py`)

```
Population → Tournament Selection → LLM Mutation → Cascaded Evaluation → Archive → Population
                                        ↑                                    |
                                        └─── archive inspiration ────────────┘
```

- **LLM-guided mutation**: GPT-5.2-codex generates code variants (50% full-block rewrite, 50% focused single-function edit via `ast.parse()`)
- **Cascaded evaluation**: regression tests → 1-sample crash check → 16-sample stratified pre-screen → full 128-sample eval
- **Shared model injection**: VLM loaded once, injected into candidates post-`exec_module`
- **Behavioral diversity archive**: MAP-Elites grid indexed by per-question-type accuracy bins + NA precision/recall
- **Island model**: 2 islands × 4 programs, periodic migration

### Post-Processing Bug Fixes (+8.6 pp)

1. Negative sign stripping in candidate extraction
2. Positional tick text fallback (non-numeric tick labels)
3. Embedded number regex over-extraction
4. Caret scientific notation preservation (`10^-6`)
5. Legend count deduplication from label lists

### Targeted Verifiers (+10.2 pp)

- **Title verifier**: 3-view TTA for panel-marker detection
- **Colorbar verifier**: single-probe "Does this chart have a colorbar?"

### Question-Type Router (`best_accuracy_v3.py`) — Negative Result

Instruct and Thinking models have complementary error profiles on *unverified* outputs:

| Category | Instruct-only correct | Thinking-only correct |
| --- | ---: | ---: |
| Value extraction (ticks, labels) | 24 | 4 |
| Absence detection (NA) | 2 | 18 |
| **Total** | **26** | **22** |

A router was built using the Thinking model for continuous-legend questions (where absence-detection dominates). However, with the colorbar verifier already in place, Instruct scores **16/16** on continuous-legend questions, leaving no room for Thinking to help. The router scores **0.7500** — worse than `best_accuracy` (0.7969) because Thinking's weaker value-extraction hurts on the 6 questions it incorrectly answers.

**Lesson:** verifier-augmented prompting can absorb the accuracy gap that would otherwise motivate a multi-model router.

### Ablation ([details in report](./report.pdf))

- **Cascaded eval**: 31% of candidates crashed; cascade saved ~35 min of wasted GPU time
- **Diversity archive**: 13 unique behavior descriptors across 20 evaluations
- **Back-port test**: fixes applied to evolved v2 base yield 0.7578 (vs 0.7969 on v1), showing partial base-specificity

## Reproduce

```bash
# Setup
pip install -r requirements.txt
# Download CharXiv images into charxiv/images/ (see charxiv/images/README.md)

# Core evaluations
python evaluate.py best_accuracy           # single run
python evaluate.py best_accuracy --cv 4    # 4-fold stability
python evaluate.py best_accuracy --cv 4 --grouped  # grouped-by-figure folds

# Full benchmark (all variants + speed loop)
bash reproduce.sh

# Re-run evolution (requires OPENAI_API_KEY)
export OPENAI_API_KEY="sk-..."
python evolve_instruct.py
python evolve_thinking.py
```

## Limitations

- **Dev-subset tuning risk**: post-processing fixes and verifiers were developed by analysing errors on the same 128-sample set used for evaluation. The evolution-only chain (38.3%→60.9%) is the more conservative generalization estimate.
- **Back-port gap**: post-processing tuned against the v1 evolved base does not transfer perfectly to v2 (0.7969 vs 0.7578), confirming partial base-specificity.
- **Single mutation model**: we use only GPT-5.2-codex; AlphaEvolve uses multi-model ensembles.
- **Single weighted scalar**: accuracy + speed are collapsed into one score instead of Pareto multi-objective.
- **128-sample evaluation**: all reported numbers are on a small dev subset; the held-out test set may show different patterns.

## Repository Structure

<details>
<summary>File tree (click to expand)</summary>

```
├── starting_scripts.py         # Teacher baseline
├── manual_instruct.py          # Hand-optimized (Instruct)
├── manual_thinking.py          # Hand-optimized (Thinking)
├── evolve_instruct.py          # Evolution system (Instruct)
├── evolve_thinking.py          # Evolution system (Thinking)
├── evolved_instruct.py         # Best evolved program (Instruct, 0.6094)
├── evolved_thinking.py         # Best evolved program (Thinking, 0.5547)
├── best_accuracy.py            # Highest accuracy: evolved v1 + fixes + verifiers (0.7969)
├── best_accuracy_v2.py         # Back-port: evolved v2 + same fixes (0.7578)
├── best_accuracy_v3.py         # Instruct/Thinking router — negative result (0.7500)
├── best_overall.py             # Best accuracy/speed tradeoff (0.7813)
├── best_speed.py               # Fastest inference (0.196 s/q warm-start)
├── ablation_no_archive.py      # Ablation: evolution w/o diversity archive
├── ablation_no_cascade.py      # Ablation: evolution w/o cascaded eval
├── evaluate.py                 # Evaluation harness (--cv, --fold, --grouped)
├── reproduce.sh                # Run all evaluations
├── requirements.txt            # Pinned Python dependencies
├── report.tex / report.pdf     # NeurIPS-format paper
├── LICENSE                     # MIT
├── CITATION.cff                # Citation metadata
└── charxiv/                    # CharXiv benchmark data + eval code
    ├── data/                   #   JSON question/answer files
    ├── images/                 #   Chart images (download separately)
    └── src/                    #   Evaluation utilities
```

</details>

## Models

- **Answering**: `Qwen/Qwen3-VL-2B-Instruct` / `Qwen/Qwen3-VL-2B-Thinking` (fixed, bfloat16, greedy)
- **Mutation** (evolution only): `gpt-5.2-codex` via OpenAI API

## License

[MIT](LICENSE)

## Citation

```bibtex
@software{huang2026evolving,
  author  = {Huang, Qi},
  title   = {Evolving {VLM} Inference for {CharXiv}},
  year    = {2026},
  url     = {https://github.com/QiHuang321/Evolving}
}
```

See [report.pdf](./report.pdf) for the full writeup.
