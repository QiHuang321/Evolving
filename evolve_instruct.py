"""
AlphaEvolve-style evolution system for Qwen3-VL-2B-Instruct inference.

Core components (following AlphaEvolve paper):
  - LLM-based mutation: dual-mode (full-block rewrite + focused/diff-based)
  - Population management: island model + behavioral diversity archive
  - Evolution loop with tournament selection, elitism, and cascaded evaluation
  - Cascaded evaluation: 1-sample crash check → 16-sample pre-screen → full 128
  - Feature-aware archive: maintains diverse programs indexed by per-query behavior

Usage:
    python evolve_instruct.py

Outputs:
    evolved_instruct.py — best program found during evolution
    artifacts/evolution_logs/evolve_instruct_log.jsonl — full history
"""

import ast
import importlib.util
import json
import os
import random
import re
import sys
import tempfile
import textwrap
import time
import traceback
import types
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────

SEED_FILE = Path(__file__).parent / "manual_instruct.py"
OUTPUT_BEST = Path(__file__).parent / "evolved_instruct.py"
ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "evolution_logs"
EVOLVE_LOG = ARTIFACT_DIR / "evolve_instruct_log.jsonl"

# OpenAI mutation-model settings
OPENAI_MODEL = "gpt-5.2-codex"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Evolution hyper-parameters
POPULATION_SIZE = 8       # total programs kept across all islands
NUM_ISLANDS = 2           # independent sub-populations
TOURNAMENT_K = 3          # tournament selection size
MAX_GENERATIONS = 40      # total evolution iterations
EVAL_NUM_SAMPLES = 128    # must match evaluate.py default

# Markers that delimit the evolvable code region
BLOCK_START = "# EVOLVE-BLOCK-START"
BLOCK_END   = "# EVOLVE-BLOCK-END"

# Cascaded evaluation: fast pre-screen before committing to full eval
PRESCREEN_SAMPLES = 16        # quick screening sample count
PRESCREEN_THRESHOLD = 0.7     # child must reach >= this × parent accuracy
PRESCREEN_STRATIFIED = True   # stratified random sampling (not fixed first-16)

# Dual-mode mutation: full-block rewrite vs focused (diff-like) mutation
DIFF_MUTATION_PROB = 0.5      # probability of focused mutation

# Known parser regression tests — cheap phase-0 sanity checks.
# Each entry: (question_snippet, raw_model_output, expected_normalized).
# A candidate that fails any of these is immediately rejected.
REGRESSION_TESTS = [
    # Negative sign must survive stripping
    ("spatially lowest labeled tick", "-1.00", "-1.00"),
    # Caret scientific notation preserved
    ("spatially lowest labeled tick", "10^-6", "10^-6"),
    # Text tick labels are not forced through numeric extraction
    ("leftmost labeled tick", "SSM", "SSM"),
    # Legend count from comma-separated list
    ("how many discrete labels are there in the legend", "cat, dog, bird", "3"),
    # Trend synonym mapping
    ("general trend of data from left to right", "increasing", "increases"),
    # Layout canonical form
    ("layout of the subplots", "2 by 3", "2 by 3"),
    # Not Applicable synonym
    ("maximum value of the tick labels on the continuous legend",
     "There is no colorbar in this chart", "Not Applicable"),
]

# ── Data structures ─────────────────────────────────────────────────────────────

@dataclass
class Program:
    code: str                          # full file content (EVOLVE-BLOCK replaced)
    block: str                         # evolvable block only
    accuracy: Optional[float] = None
    avg_time: Optional[float] = None
    score: Optional[float] = None      # composite: 0.9*acc + 0.1*speed_bonus
    generation: int = 0
    island: int = 0
    parent_scores: list = field(default_factory=list)
    mutation_prompt: str = ""
    mutation_mode: str = "full"         # "full" (rewrite) or "focused" (diff-like)
    eval_error: str = ""
    behavior: tuple = ()               # per-query correctness vector (for diversity)

    def to_dict(self):
        d = asdict(self)
        del d["code"]
        # Compact behavior: binary string for logging
        if d.get("behavior"):
            d["behavior"] = "".join(str(b) for b in d["behavior"])
        return d


# ── Evaluation ──────────────────────────────────────────────────────────────────

CHARXIV_PATH = Path(__file__).parent / "charxiv"
sys.path.insert(0, str(CHARXIV_PATH / "src"))

# Shared model/processor: loaded once and injected into candidate modules.
_SHARED_MODEL = None
_SHARED_PROCESSOR = None


def _init_shared_model():
    """Load the VLM once so candidates don't each pay the load cost."""
    global _SHARED_MODEL, _SHARED_PROCESSOR
    if _SHARED_MODEL is not None:
        return
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32
    _SHARED_MODEL = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=dtype, device_map="auto",
    )
    _SHARED_MODEL.eval()
    _SHARED_PROCESSOR = AutoProcessor.from_pretrained(model_name, use_fast=False)
    print(f"  [shared model] Loaded {model_name} ({dtype})")


def _stratified_sample(query_keys: list, n: int) -> list:
    """Stratified random sample of n keys, proportional to question-type bins."""
    type_bins: dict[str, list] = {}
    for key in query_keys:
        qt = _CACHED_QTYPE.get(key, "other")
        type_bins.setdefault(qt, []).append(key)
    # Proportional allocation
    total = len(query_keys)
    selected = []
    for qt, keys in type_bins.items():
        k = max(1, round(n * len(keys) / total))
        selected.extend(random.sample(keys, min(k, len(keys))))
    # Trim or pad to exactly n
    if len(selected) > n:
        selected = random.sample(selected, n)
    elif len(selected) < n:
        remaining = [k for k in query_keys if k not in set(selected)]
        selected.extend(random.sample(remaining, min(n - len(selected), len(remaining))))
    return selected


def _load_queries():
    from descriptive_utils import build_descriptive_quries
    with open(CHARXIV_PATH / "data" / "descriptive_val.json") as f:
        data = json.load(f)
    queries = build_descriptive_quries(data, str(CHARXIV_PATH / "images"))
    queries = dict(list(queries.items())[:EVAL_NUM_SAMPLES])
    return queries, data


_CACHED_QUERIES = None
_CACHED_GT = None
_CACHED_QTYPE = None          # query_key -> question-type label


def _classify_question(question: str) -> str:
    """Map a question to a coarse question-type for MAP-Elites descriptors."""
    q = " ".join(question.lower().split())
    if "continuous legend" in q:
        return "colorbar"
    if "legend" in q:
        return "legend"
    if "layout of the subplots" in q or "number of subplots" in q:
        return "layout"
    if any(p in q for p in ["leftmost", "rightmost", "lowest", "highest",
                            "tick values", "tick labels", "how many"]):
        return "numeric"
    if "title" in q:
        return "title"
    if "label of the" in q or "axis" in q:
        return "axis"
    if "trend" in q or "intersect" in q:
        return "trend"
    return "other"


def _get_queries():
    global _CACHED_QUERIES, _CACHED_GT, _CACHED_QTYPE
    if _CACHED_QUERIES is None:
        _CACHED_QUERIES, _CACHED_GT = _load_queries()
        _CACHED_QTYPE = {}
        for key, q in _CACHED_QUERIES.items():
            _CACHED_QTYPE[key] = _classify_question(q["question"])
    return _CACHED_QUERIES, _CACHED_GT


def evaluate_program(program: Program, parent_accuracy: float = 0.0) -> Program:
    """Cascaded evaluation with optional pre-screening.

    If parent_accuracy > 0, the evaluation proceeds in three phases:
      1. Single-sample crash check — catches import/runtime errors instantly.
      2. Pre-screen on PRESCREEN_SAMPLES — filters obviously weak candidates.
      3. Full evaluation on all EVAL_NUM_SAMPLES — computes final score.
    If parent_accuracy == 0 (seed), skips phases 1-2 and runs full eval directly.
    Also computes a per-query behavior vector for diversity tracking.
    """
    queries, gt_data = _get_queries()
    query_keys = list(queries.keys())

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=str(Path(__file__).parent)
    ) as f:
        f.write(program.code)
        tmp_path = f.name

    try:
        spec = importlib.util.spec_from_file_location("_evolved_candidate", tmp_path)
        mod = importlib.util.module_from_spec(spec)

        # ── Inject shared model so candidates don't reload from scratch ──
        if _SHARED_MODEL is not None:
            mod._MODEL = _SHARED_MODEL
            mod._PROCESSOR = _SHARED_PROCESSOR

        spec.loader.exec_module(mod)

        # ── Phase 0: Regression tests (known parser invariants) ──────────
        if parent_accuracy > 0 and hasattr(mod, '_normalize_answer'):
            for q_snippet, raw_out, expected in REGRESSION_TESTS:
                try:
                    result = mod._normalize_answer(q_snippet, raw_out)
                    if result != expected:
                        program.accuracy = 0.0
                        program.avg_time = 999.0
                        program.score = 0.0
                        program.eval_error = (f"regression test fail: "
                            f"normalize_answer({q_snippet!r}, {raw_out!r}) "
                            f"= {result!r}, expected {expected!r}")
                        print(f"  [regression] FAIL: {program.eval_error}")
                        return program
                except Exception:
                    pass  # candidate may not expose _normalize_answer

        responses = {}  # query_key -> response string
        num_errors = 0
        start_time = time.time()

        # ── Phase 1: Crash check (1 sample) ─────────────────────────────────
        if parent_accuracy > 0:
            first_key = query_keys[0]
            first_q = queries[first_key]
            try:
                resp = mod.vlm_inference(
                    image_path=first_q["figure_path"],
                    question=first_q["question"],
                )
                if not isinstance(resp, str):
                    raise ValueError(f"non-string response: {type(resp)}")
                responses[first_key] = resp
            except Exception as e:
                program.accuracy = 0.0
                program.avg_time = 999.0
                program.score = 0.0
                program.eval_error = f"crash on first sample: {e}"
                print(f"  [safety] Crash on sample 1: {e}")
                return program

        # ── Phase 2: Pre-screen (PRESCREEN_SAMPLES) ──────────────────────────
        if parent_accuracy > 0:
            # Stratified random sampling: pick PRESCREEN_SAMPLES from each
            # question-type proportionally, avoiding fixed-subset bias.
            if PRESCREEN_STRATIFIED:
                _prescreen_keys = _stratified_sample(query_keys, PRESCREEN_SAMPLES)
            else:
                _prescreen_keys = query_keys[:PRESCREEN_SAMPLES]
            for key in _prescreen_keys:
                if key in responses:
                    continue
                try:
                    resp = mod.vlm_inference(
                        image_path=queries[key]["figure_path"],
                        question=queries[key]["question"],
                    )
                    responses[key] = resp
                except Exception:
                    responses[key] = "ERROR"
                    num_errors += 1

            pre_correct = 0
            pre_total = 0
            for key in _prescreen_keys:
                figure_id, subq_idx = key.split("_")
                gt_entry = gt_data.get(figure_id)
                if gt_entry is None:
                    continue
                gt_answer = gt_entry["answers"][int(subq_idx)]
                if str(responses.get(key, "ERROR")).strip().lower() == gt_answer.lower():
                    pre_correct += 1
                pre_total += 1

            pre_acc = pre_correct / pre_total if pre_total > 0 else 0.0
            threshold = parent_accuracy * PRESCREEN_THRESHOLD
            if pre_acc < threshold:
                elapsed = time.time() - start_time
                program.accuracy = pre_acc
                program.avg_time = elapsed / pre_total if pre_total > 0 else 999.0
                program.score = 0.0
                program.eval_error = f"prescreen fail: {pre_acc:.3f} < {threshold:.3f}"
                print(f"  [prescreen] FAIL: {pre_acc:.3f} < {threshold:.3f} "
                      f"(saved {EVAL_NUM_SAMPLES - PRESCREEN_SAMPLES} evals)")
                return program
            else:
                print(f"  [prescreen] PASS: {pre_acc:.3f} >= {threshold:.3f}")

        # ── Phase 3: Full evaluation ─────────────────────────────────────────
        for key in query_keys:
            if key in responses:
                continue
            try:
                resp = mod.vlm_inference(
                    image_path=queries[key]["figure_path"],
                    question=queries[key]["question"],
                )
                responses[key] = resp
            except Exception:
                responses[key] = "ERROR"
                num_errors += 1

        total_time = time.time() - start_time

        correct = 0
        total = 0
        behavior_bits = []
        # Per-question-type accuracy counters for MAP-Elites descriptors
        type_correct: dict[str, int] = {}
        type_total: dict[str, int] = {}
        na_tp = na_fp = na_fn = na_tn = 0  # confusion counts for NA answers
        for key in query_keys:
            figure_id, subq_idx = key.split("_")
            gt_entry = gt_data.get(figure_id)
            if gt_entry is None:
                continue
            gt_answer = gt_entry["answers"][int(subq_idx)]
            resp = str(responses.get(key, "ERROR"))
            is_correct = resp.strip().lower() == gt_answer.lower()
            if is_correct:
                correct += 1
            behavior_bits.append(1 if is_correct else 0)
            total += 1
            # Track per-type accuracy
            qt = _CACHED_QTYPE.get(key, "other")
            type_correct[qt] = type_correct.get(qt, 0) + (1 if is_correct else 0)
            type_total[qt] = type_total.get(qt, 0) + 1
            # Track NA precision/recall
            pred_na = resp.strip().lower() == "not applicable"
            gt_na = gt_answer.lower() == "not applicable"
            if pred_na and gt_na:
                na_tp += 1
            elif pred_na and not gt_na:
                na_fp += 1
            elif not pred_na and gt_na:
                na_fn += 1
            else:
                na_tn += 1

        program.accuracy = correct / total if total > 0 else 0.0
        program.avg_time = total_time / total if total > 0 else 999.0

        # Build low-dimensional behavior descriptor for MAP-Elites archive.
        # Each dimension is discretised to one of 5 bins (0..4).
        def _bin5(x):
            return min(4, int(x * 5))

        descriptor = []
        for qt in sorted(type_total.keys()):
            acc = type_correct.get(qt, 0) / type_total[qt] if type_total.get(qt) else 0
            descriptor.append(_bin5(acc))
        # NA precision bin
        na_prec = na_tp / (na_tp + na_fp) if (na_tp + na_fp) > 0 else 1.0
        descriptor.append(_bin5(na_prec))
        # NA recall bin
        na_rec = na_tp / (na_tp + na_fn) if (na_tp + na_fn) > 0 else 1.0
        descriptor.append(_bin5(na_rec))
        program.behavior = tuple(descriptor)

        baseline_time = 3.5
        speed_bonus = max(0.0, min(1.0, (baseline_time - program.avg_time) / baseline_time))
        program.score = 0.9 * program.accuracy + 0.1 * speed_bonus

    except Exception as e:
        program.accuracy = 0.0
        program.avg_time = 999.0
        program.score = 0.0
        program.eval_error = traceback.format_exc()
        print(f"  [eval error] {e}")
    finally:
        os.unlink(tmp_path)

    return program


# ── Code utilities ───────────────────────────────────────────────────────────────

def _extract_block(code: str) -> str:
    """Return only the EVOLVE-BLOCK contents (without the marker lines)."""
    lines = code.splitlines()
    inside = False
    block_lines = []
    for line in lines:
        if BLOCK_START in line:
            inside = True
            continue
        if BLOCK_END in line:
            break
        if inside:
            block_lines.append(line)
    return "\n".join(block_lines)


def _replace_block(full_code: str, new_block: str) -> str:
    """Replace the EVOLVE-BLOCK section with new_block."""
    start_marker = BLOCK_START
    end_marker = BLOCK_END
    before = full_code.split(start_marker)[0]
    after = full_code.split(end_marker)[1]
    return f"{before}{start_marker}\n{new_block}\n{end_marker}{after}"


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _extract_python_from_response(text: str) -> Optional[str]:
    """
    Extract Python code from an LLM response.
    Tries fenced code blocks first, then falls back to the full text.
    """
    # Fenced block: ```python ... ```
    pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        # Take the longest match (most likely the full block)
        return max(matches, key=len).strip()
    # No fences: assume the whole response is code
    return text.strip()


def _pick_random_function(block: str) -> Optional[str]:
    """Pick a random function name from the EVOLVE-BLOCK for targeted mutation."""
    try:
        tree = ast.parse(block)
        funcs = [node.name for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef)]
        if funcs:
            return random.choice(funcs)
    except SyntaxError:
        pass
    return None


# ── OpenAI mutation operator ─────────────────────────────────────────────────────

def _build_mutation_prompt(
    parent: Program,
    generation: int,
    other_programs: list,
    task_description: str,
) -> str:
    """
    Build the prompt sent to the mutation model.
    Inspired by AlphaEvolve: provide context, current best, and request targeted mutation.
    """
    other_examples = ""
    if other_programs:
        examples = random.sample(other_programs, min(2, len(other_programs)))
        for i, p in enumerate(examples):
            other_examples += f"\n--- Example {i+1} (score={p.score:.4f}) ---\n{p.block}\n"

    mutation_strategies = [
        "Improve the SYSTEM or USER prompt so the model outputs answers in exactly the right format for CharXiv descriptive questions (exact-match scoring).",
        "Add or improve post-processing logic. CharXiv answers are short exact strings: numbers, Yes/No, 'n by m', comma-separated legend labels, or short text.",
        "Optimise inference speed: consider batching multiple questions for the same image, reducing max_new_tokens, or using flash_attention_2.",
        "Improve handling of 'Not Applicable' cases — the model should say 'Not Applicable' when the chart doesn't support the answer, not guess.",
        "Improve numeric answer extraction — CharXiv often expects exact decimal/integer values from tick labels.",
        "Improve the prompt for specific question types: axis labels, legend names, trend descriptions, subplot layout.",
        "Add or improve answer normalisation: strip punctuation, fix casing to match expected format.",
        "Add image preprocessing (resize, contrast) if it can improve chart readability for the VLM.",
    ]
    strategy = random.choice(mutation_strategies)

    prompt = f"""You are an expert Python programmer helping to improve a Vision-Language Model (VLM) inference function for the CharXiv benchmark.

## Task
CharXiv evaluates chart understanding using **exact-match** string comparison. The model must output answers that exactly match the ground truth (case-insensitive). Common answer formats include:
- Numbers/counts: e.g. "42", "3.14", "0.5"
- Yes/No/Not Applicable
- Subplot layout: "2 by 3"
- Legend label lists: "label1, label2, label3"
- Short text: axis titles, trend descriptions

## Current program (score={parent.score:.4f}, accuracy={parent.accuracy:.4f}, avg_time={parent.avg_time:.3f}s, generation={parent.generation})
```python
{parent.block}
```

## Mutation strategy for this generation
{strategy}

## Other programs in the population for inspiration{other_examples if other_examples else " (none yet)"}

## Instructions
1. Output a **mutated version** of the EVOLVE-BLOCK Python code only.
2. The code MUST define a `vlm_inference(image_path, question)` function.
3. Keep `_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"` unchanged.
4. Use `do_sample=False` (greedy decoding) — required for reproducibility.
5. Do NOT change the model to a larger variant.
6. The code must be valid Python.
7. Wrap your output in triple backticks: ```python ... ```
8. Be creative but focused — make one or two targeted improvements.
"""
    return prompt


def _build_focused_mutation_prompt(
    parent: Program,
    generation: int,
    other_programs: list,
    target_function: str,
) -> str:
    """
    Build a focused/diff-style mutation prompt targeting a single function.
    The LLM is asked to output the complete block but change ONLY the target.
    This produces smaller, safer mutations compared to full-block rewrites.
    """
    other_examples = ""
    if other_programs:
        examples = random.sample(other_programs, min(2, len(other_programs)))
        for i, p in enumerate(examples):
            other_examples += f"\n--- Example {i+1} (score={p.score:.4f}) ---\n{p.block}\n"

    prompt = f"""You are an expert Python programmer making a SURGICAL, TARGETED improvement to a VLM inference function for the CharXiv benchmark.

## Task
CharXiv evaluates chart understanding using **exact-match** string comparison (case-insensitive).

## Current program (score={parent.score:.4f}, accuracy={parent.accuracy:.4f}, avg_time={parent.avg_time:.3f}s)
```python
{parent.block}
```

## YOUR MISSION
Make a **small, focused change** to ONLY the `{target_function}` function (or its immediate helpers).
Do NOT rewrite the entire program. Keep everything else EXACTLY as-is.
The change should be roughly 1-15 lines of modification.

## Other programs for inspiration{other_examples if other_examples else " (none)"}

## Rules
1. Output the COMPLETE EVOLVE-BLOCK code (all functions), but the ONLY changes should be in/around `{target_function}`.
2. Keep `_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"` unchanged.
3. Use `do_sample=False` (greedy decoding) — required for reproducibility.
4. Do NOT change the model to a larger variant.
5. Valid Python only.
6. Wrap output in ```python ... ```
"""
    return prompt


def mutate(
    parent: Program,
    generation: int,
    population: list,
    openai_client,
    archive_programs: Optional[list] = None,
) -> Optional[Program]:
    """Call OpenAI to produce a mutated child program.

    Dual-mode: randomly chooses between full-block rewrite and focused
    (diff-like) mutation that targets a single function.
    """
    others = [p for p in population if p is not parent and p.score is not None]
    if archive_programs:
        others = others + [p for p in archive_programs if p is not parent]

    # Choose mutation mode
    use_focused = random.random() < DIFF_MUTATION_PROB
    target_func = None
    if use_focused:
        target_func = _pick_random_function(parent.block)
        if target_func is None:
            use_focused = False

    if use_focused:
        mutation_mode = "focused"
        prompt = _build_focused_mutation_prompt(parent, generation, others, target_func)
    else:
        mutation_mode = "full"
        prompt = _build_mutation_prompt(parent, generation, others, "CharXiv descriptive QA")

    try:
        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            reasoning={"effort": "medium"},
            text={"verbosity": "medium"},
            max_output_tokens=6000,
        )
        raw = response.output_text
    except Exception as e:
        print(f"  [OpenAI error] {e}")
        return None

    new_block = _extract_python_from_response(raw)
    if new_block is None or not _is_valid_python(new_block):
        print("  [mutation] Invalid Python from mutation model, skipping.")
        return None

    seed_full_code = SEED_FILE.read_text()
    new_full_code = _replace_block(seed_full_code, new_block)

    child = Program(
        code=new_full_code,
        block=new_block,
        generation=generation,
        parent_scores=[parent.score],
        mutation_prompt=prompt,
        mutation_mode=mutation_mode,
    )
    return child


# ── Population management ────────────────────────────────────────────────────────

class Population:
    """
    Island-based population.
    Each island maintains its own top candidates.
    Periodically, the best individual from each island migrates.
    """

    def __init__(self, num_islands: int, island_size: int):
        self.num_islands = num_islands
        self.island_size = island_size
        self.islands: list[list[Program]] = [[] for _ in range(num_islands)]

    def add(self, program: Program):
        island = self.islands[program.island]
        island.append(program)
        # Keep only top island_size by score
        island.sort(key=lambda p: p.score or 0.0, reverse=True)
        if len(island) > self.island_size:
            island[:] = island[: self.island_size]

    def tournament_select(self, island_idx: int) -> Optional[Program]:
        """Tournament selection within one island."""
        island = self.islands[island_idx]
        if not island:
            return None
        candidates = random.sample(island, min(TOURNAMENT_K, len(island)))
        return max(candidates, key=lambda p: p.score or 0.0)

    def migrate(self):
        """Copy the best individual from each island to a random other island."""
        for i, island in enumerate(self.islands):
            if not island:
                continue
            best = island[0]
            target = random.choice([j for j in range(self.num_islands) if j != i])
            emigrant = Program(
                code=best.code,
                block=best.block,
                accuracy=best.accuracy,
                avg_time=best.avg_time,
                score=best.score,
                generation=best.generation,
                island=target,
                parent_scores=best.parent_scores[:],
            )
            # Only add if it improves the target island
            target_island = self.islands[target]
            if not target_island or emigrant.score > target_island[-1].score:
                target_island.append(emigrant)
                target_island.sort(key=lambda p: p.score or 0.0, reverse=True)
                if len(target_island) > self.island_size:
                    target_island[:] = target_island[: self.island_size]

    def best(self) -> Optional[Program]:
        all_programs = [p for island in self.islands for p in island]
        if not all_programs:
            return None
        return max(all_programs, key=lambda p: p.score or 0.0)

    def all_programs(self) -> list:
        return [p for island in self.islands for p in island]

    def size(self) -> int:
        return sum(len(island) for island in self.islands)


# ── Behavioral diversity archive ────────────────────────────────────────────────

class FeatureArchive:
    """MAP-Elites behavioral diversity archive with low-dimensional descriptors.

    Programs are indexed by a discretised behavior descriptor: a tuple of
    per-question-type accuracy bins (5 levels each) plus NA precision and
    recall bins.  Each unique descriptor cell retains only its highest-scoring
    member.  This is a genuine MAP-Elites grid rather than a hash-based
    dedup cache, encouraging the LLM mutation operator to explore different
    competence profiles (e.g. high-numeric-low-legend vs high-legend-low-numeric).
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._archive: dict[tuple, Program] = {}  # descriptor_tuple -> Program

    def add(self, program: Program):
        if not program.behavior or (program.score or 0) <= 0:
            return
        cell = program.behavior  # low-dim descriptor tuple used directly as key
        existing = self._archive.get(cell)
        if existing is None or (program.score or 0) > (existing.score or 0):
            self._archive[cell] = program
        # Evict lowest-scoring entry if over capacity
        if len(self._archive) > self.max_size:
            worst_key = min(self._archive, key=lambda k: self._archive[k].score or 0)
            del self._archive[worst_key]

    def sample(self, k: int = 2) -> list:
        """Sample k diverse programs from the archive."""
        programs = list(self._archive.values())
        if not programs:
            return []
        return random.sample(programs, min(k, len(programs)))

    def best(self) -> Optional[Program]:
        if not self._archive:
            return None
        return max(self._archive.values(), key=lambda p: p.score or 0)

    def size(self) -> int:
        return len(self._archive)

    def summary(self) -> str:
        if not self._archive:
            return "empty"
        scores = sorted([(p.score or 0) for p in self._archive.values()], reverse=True)
        top = ", ".join(f"{s:.4f}" for s in scores[:5])
        suffix = "..." if len(scores) > 5 else ""
        return f"{len(scores)} unique behaviors, top=[{top}{suffix}]"


# ── Logging ──────────────────────────────────────────────────────────────────────

def log_program(program: Program, generation: int):
    record = program.to_dict()
    record["generation_logged"] = generation
    record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(EVOLVE_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Main evolution loop ──────────────────────────────────────────────────────────

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if not OPENAI_API_KEY:
        print("ERROR: set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    # Pre-load the VLM once so candidate modules share it.
    _init_shared_model()
    print(f"=== AlphaEvolve for Qwen3-VL-2B-Instruct ===")
    print(f"  Population: {POPULATION_SIZE} ({NUM_ISLANDS} islands x {POPULATION_SIZE//NUM_ISLANDS})")
    print(f"  Max generations: {MAX_GENERATIONS}")
    print(f"  Eval samples: {EVAL_NUM_SAMPLES}")
    print(f"  Pre-screen: {PRESCREEN_SAMPLES} samples, threshold={PRESCREEN_THRESHOLD}")
    print(f"  Mutation: dual-mode (focused prob={DIFF_MUTATION_PROB})")
    print(f"  Mutation LLM: {OPENAI_MODEL}")
    print()

    population = Population(
        num_islands=NUM_ISLANDS,
        island_size=POPULATION_SIZE // NUM_ISLANDS,
    )
    archive = FeatureArchive(max_size=50)

    # ── Seed population with manual_instruct.py ─────────────────────────────
    print("[Gen 0] Evaluating seed program (manual_instruct.py) ...")
    seed_code = SEED_FILE.read_text()
    seed_block = _extract_block(seed_code)
    seed = Program(code=seed_code, block=seed_block, generation=0, island=0)
    seed = evaluate_program(seed)  # no prescreen for seed
    print(f"  Seed: accuracy={seed.accuracy:.4f}, avg_time={seed.avg_time:.3f}s, score={seed.score:.4f}")
    population.add(seed)
    archive.add(seed)
    log_program(seed, 0)

    # Spread seed to all islands
    for i in range(1, NUM_ISLANDS):
        clone = Program(
            code=seed.code,
            block=seed.block,
            accuracy=seed.accuracy,
            avg_time=seed.avg_time,
            score=seed.score,
            generation=0,
            island=i,
            behavior=seed.behavior,
        )
        population.add(clone)

    best_ever = seed
    print(f"\n[Gen 0] Best so far: score={best_ever.score:.4f} (accuracy={best_ever.accuracy:.4f})\n")

    # ── Evolution loop ───────────────────────────────────────────────────────
    for gen in range(1, MAX_GENERATIONS + 1):
        island_idx = (gen - 1) % NUM_ISLANDS
        print(f"[Gen {gen:03d}] Island {island_idx} — ", end="", flush=True)

        parent = population.tournament_select(island_idx)
        if parent is None:
            print("empty island, skipping.")
            continue

        print(f"parent score={parent.score:.4f} — mutating "
              f"(archive: {archive.size()} behaviors) ...", flush=True)

        child = mutate(parent, gen, population.all_programs(), openai_client,
                       archive_programs=archive.sample(2))
        if child is None:
            print(f"  [Gen {gen:03d}] Mutation failed, skipping.")
            continue

        child.island = island_idx
        print(f"  [Gen {gen:03d}] [{child.mutation_mode}] Evaluating ...", flush=True)
        child = evaluate_program(child, parent_accuracy=parent.accuracy or 0.0)

        # If prescreen failed, log and skip
        if child.eval_error and ("prescreen fail" in child.eval_error
                                 or "crash" in child.eval_error):
            log_program(child, gen)
            continue

        print(f"  [Gen {gen:03d}] accuracy={child.accuracy:.4f}, "
              f"avg_time={child.avg_time:.3f}s, score={child.score:.4f}")

        population.add(child)
        archive.add(child)
        log_program(child, gen)

        if child.score > best_ever.score:
            best_ever = child
            print(f"  *** New best! score={best_ever.score:.4f} "
                  f"(accuracy={best_ever.accuracy:.4f}, avg_time={best_ever.avg_time:.3f}s)")
            # Save incrementally
            OUTPUT_BEST.write_text(best_ever.code)
            print(f"  *** Saved to {OUTPUT_BEST}")

        # Island migration every 5 generations
        if gen % 5 == 0:
            population.migrate()
            print(f"  [Gen {gen:03d}] Island migration done.")

        # Print population summary every 10 generations
        if gen % 10 == 0:
            print(f"\n--- Population summary at gen {gen} ---")
            for i, island in enumerate(population.islands):
                scores = [f"{p.score:.4f}" for p in island]
                print(f"  Island {i}: {scores}")
            print(f"  Archive: {archive.summary()}")
            print(f"  Best ever: score={best_ever.score:.4f} "
                  f"(accuracy={best_ever.accuracy:.4f}, avg_time={best_ever.avg_time:.3f}s)\n")

    # ── Final output ─────────────────────────────────────────────────────────
    print(f"\n=== Evolution complete ===")
    print(f"  Best score : {best_ever.score:.4f}")
    print(f"  Accuracy   : {best_ever.accuracy:.4f}")
    print(f"  Avg time   : {best_ever.avg_time:.3f}s")
    print(f"  Generation : {best_ever.generation}")
    OUTPUT_BEST.write_text(best_ever.code)
    print(f"  Output     : {OUTPUT_BEST}")
    print(f"  Full log   : {EVOLVE_LOG}")


if __name__ == "__main__":
    main()
