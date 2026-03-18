"""
AlphaEvolve-style evolution system for Qwen3-VL-2B-Thinking inference.

Identical architecture to evolve_instruct.py with the following differences:
  - Seed: manual_thinking.py
  - Model: Qwen/Qwen3-VL-2B-Thinking
  - Mutation prompts reference thinking-model specifics (think blocks, budgets)

Core components:
  - LLM-based mutation: dual-mode (full-block rewrite + focused/diff-based)
  - Population management: island model + behavioral diversity archive
  - Evolution loop with tournament selection, elitism, and cascaded evaluation
  - Cascaded evaluation: 1-sample crash check → 16-sample pre-screen → full 128
  - Feature-aware archive: maintains diverse programs indexed by per-query behavior

Usage:
    python evolve_thinking.py

Outputs:
    evolved_thinking.py — best program found during evolution
    artifacts/evolution_logs/evolve_thinking_log.jsonl — full history
"""

import ast
import importlib.util
import json
import os
import random
import re
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────

SEED_FILE    = Path(__file__).parent / "manual_thinking.py"
OUTPUT_BEST  = Path(__file__).parent / "evolved_thinking.py"
ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "evolution_logs"
EVOLVE_LOG   = ARTIFACT_DIR / "evolve_thinking_log.jsonl"

OPENAI_MODEL = "gpt-5.2-codex"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

POPULATION_SIZE  = 8
NUM_ISLANDS      = 2
TOURNAMENT_K     = 3
MAX_GENERATIONS  = 40
EVAL_NUM_SAMPLES = 128

BLOCK_START = "# EVOLVE-BLOCK-START"
BLOCK_END   = "# EVOLVE-BLOCK-END"

# Cascaded evaluation settings
PRESCREEN_SAMPLES = 16
PRESCREEN_THRESHOLD = 0.7

# Dual-mode mutation
DIFF_MUTATION_PROB = 0.5

# ── Data structures ─────────────────────────────────────────────────────────────

@dataclass
class Program:
    code: str
    block: str
    accuracy: Optional[float] = None
    avg_time: Optional[float] = None
    score: Optional[float] = None
    generation: int = 0
    island: int = 0
    parent_scores: list = field(default_factory=list)
    mutation_prompt: str = ""
    mutation_mode: str = "full"
    eval_error: str = ""
    behavior: tuple = ()

    def to_dict(self):
        d = asdict(self)
        del d["code"]
        if d.get("behavior"):
            d["behavior"] = "".join(str(b) for b in d["behavior"])
        return d


# ── Evaluation ──────────────────────────────────────────────────────────────────

CHARXIV_PATH = Path(__file__).parent / "charxiv"
sys.path.insert(0, str(CHARXIV_PATH / "src"))

_CACHED_QUERIES = None
_CACHED_GT = None


def _get_queries():
    global _CACHED_QUERIES, _CACHED_GT
    if _CACHED_QUERIES is None:
        from descriptive_utils import build_descriptive_quries
        with open(CHARXIV_PATH / "data" / "descriptive_val.json") as f:
            data = json.load(f)
        queries = build_descriptive_quries(data, str(CHARXIV_PATH / "images"))
        _CACHED_QUERIES = dict(list(queries.items())[:EVAL_NUM_SAMPLES])
        _CACHED_GT = data
    return _CACHED_QUERIES, _CACHED_GT


def evaluate_program(program: Program, parent_accuracy: float = 0.0) -> Program:
    """Cascaded evaluation: crash check → pre-screen → full eval."""
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
        spec.loader.exec_module(mod)

        responses = {}
        num_errors = 0
        start_time = time.time()

        # ── Phase 1: Crash check (1 sample) ──
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

        # ── Phase 2: Pre-screen ──
        if parent_accuracy > 0:
            for key in query_keys[:PRESCREEN_SAMPLES]:
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
            for key in query_keys[:PRESCREEN_SAMPLES]:
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
                print(f"  [prescreen] FAIL: {pre_acc:.3f} < {threshold:.3f}")
                return program
            else:
                print(f"  [prescreen] PASS: {pre_acc:.3f} >= {threshold:.3f}")

        # ── Phase 3: Full evaluation ──
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

        program.accuracy = correct / total if total > 0 else 0.0
        program.avg_time = total_time / total if total > 0 else 999.0
        program.behavior = tuple(behavior_bits)

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
    before = full_code.split(BLOCK_START)[0]
    after = full_code.split(BLOCK_END)[1]
    return f"{before}{BLOCK_START}\n{new_block}\n{BLOCK_END}{after}"


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _extract_python_from_response(text: str) -> Optional[str]:
    pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return max(matches, key=len).strip()
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

def _build_mutation_prompt(parent: Program, generation: int, other_programs: list) -> str:
    other_examples = ""
    if other_programs:
        examples = random.sample(other_programs, min(2, len(other_programs)))
        for i, p in enumerate(examples):
            other_examples += f"\n--- Example {i+1} (score={p.score:.4f}) ---\n{p.block}\n"

    mutation_strategies = [
        "Improve the prompt to make the Thinking model output concise exact-match answers. Thinking models often produce verbose reasoning — suppress it better.",
        "Improve the <think></think> block injection. An empty <think></think> before the answer forces the model to skip or shorten reasoning.",
        "Improve post-processing to extract the final answer from inside or after a <think> block correctly.",
        "Improve numeric answer extraction for CharXiv tick values, counts, and measurements.",
        "Improve 'Not Applicable' detection — when does the chart not support an answer?",
        "Tune max_new_tokens for thinking model — it needs more tokens for reasoning but we want the answer to be short.",
        "Improve prompt specificity for axis labels, legend names, subplot layout, and trend questions.",
        "Add answer normalisation: strip trailing punctuation, fix casing, normalise number formatting.",
    ]
    strategy = random.choice(mutation_strategies)

    return f"""You are an expert Python programmer helping to improve a Vision-Language Model (VLM) inference function for the CharXiv benchmark.

## Task
CharXiv uses **exact-match** string comparison (case-insensitive). The model being used is **Qwen3-VL-2B-Thinking**, a thinking/reasoning model that wraps its chain-of-thought in `<think>...</think>` tags. The final answer comes after `</think>`.

Common answer formats:
- Numbers: "42", "3.14"
- Yes/No/Not Applicable
- Layout: "2 by 3"
- Legend labels: "label1, label2, label3"
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
2. MUST define `vlm_inference(image_path, question)`.
3. Keep `_MODEL_NAME = "Qwen/Qwen3-VL-2B-Thinking"` unchanged.
4. Use `do_sample=False` (greedy decoding).
5. Do NOT swap to a larger model.
6. Valid Python only.
7. Wrap output: ```python ... ```
8. Make one or two focused improvements.
"""


def _build_focused_mutation_prompt(parent: Program, generation: int,
                                    other_programs: list, target_function: str) -> str:
    """Focused/diff-style mutation prompt targeting a single function."""
    other_examples = ""
    if other_programs:
        examples = random.sample(other_programs, min(2, len(other_programs)))
        for i, p in enumerate(examples):
            other_examples += f"\n--- Example {i+1} (score={p.score:.4f}) ---\n{p.block}\n"

    return f"""You are an expert Python programmer making a SURGICAL, TARGETED improvement to a VLM inference function for the CharXiv benchmark.

## Task
CharXiv uses **exact-match** string comparison (case-insensitive). The model is **Qwen3-VL-2B-Thinking** (chain-of-thought in `<think>...</think>` tags).

## Current program (score={parent.score:.4f}, accuracy={parent.accuracy:.4f})
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
2. Keep `_MODEL_NAME = "Qwen/Qwen3-VL-2B-Thinking"` unchanged.
3. Use `do_sample=False` (greedy decoding).
4. Do NOT swap to a larger model.
5. Valid Python only.
6. Wrap output: ```python ... ```
"""


def mutate(parent: Program, generation: int, population: list, openai_client,
           archive_programs: Optional[list] = None) -> Optional[Program]:
    """Dual-mode mutation: full-block rewrite or focused (diff-like)."""
    others = [p for p in population if p is not parent and p.score is not None]
    if archive_programs:
        others = others + [p for p in archive_programs if p is not parent]

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
        prompt = _build_mutation_prompt(parent, generation, others)

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

    return Program(
        code=new_full_code,
        block=new_block,
        generation=generation,
        parent_scores=[parent.score],
        mutation_prompt=prompt,
        mutation_mode=mutation_mode,
    )


# ── Population management ────────────────────────────────────────────────────────

class Population:
    def __init__(self, num_islands: int, island_size: int):
        self.num_islands = num_islands
        self.island_size = island_size
        self.islands: list[list[Program]] = [[] for _ in range(num_islands)]

    def add(self, program: Program):
        island = self.islands[program.island]
        island.append(program)
        island.sort(key=lambda p: p.score or 0.0, reverse=True)
        if len(island) > self.island_size:
            island[:] = island[: self.island_size]

    def tournament_select(self, island_idx: int) -> Optional[Program]:
        island = self.islands[island_idx]
        if not island:
            return None
        candidates = random.sample(island, min(TOURNAMENT_K, len(island)))
        return max(candidates, key=lambda p: p.score or 0.0)

    def migrate(self):
        for i, island in enumerate(self.islands):
            if not island:
                continue
            best = island[0]
            target = random.choice([j for j in range(self.num_islands) if j != i])
            emigrant = Program(
                code=best.code, block=best.block,
                accuracy=best.accuracy, avg_time=best.avg_time,
                score=best.score, generation=best.generation,
                island=target, parent_scores=best.parent_scores[:],
            )
            t_island = self.islands[target]
            if not t_island or emigrant.score > t_island[-1].score:
                t_island.append(emigrant)
                t_island.sort(key=lambda p: p.score or 0.0, reverse=True)
                if len(t_island) > self.island_size:
                    t_island[:] = t_island[: self.island_size]

    def best(self) -> Optional[Program]:
        all_p = [p for island in self.islands for p in island]
        return max(all_p, key=lambda p: p.score or 0.0) if all_p else None

    def all_programs(self) -> list:
        return [p for island in self.islands for p in island]


# ── Behavioral diversity archive ────────────────────────────────────────────────

class FeatureArchive:
    """Lightweight MAP-Elites-style behavioral diversity archive."""

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._archive: dict[int, Program] = {}

    def add(self, program: Program):
        if not program.behavior or (program.score or 0) <= 0:
            return
        bh = hash(program.behavior)
        existing = self._archive.get(bh)
        if existing is None or (program.score or 0) > (existing.score or 0):
            self._archive[bh] = program
        if len(self._archive) > self.max_size:
            worst_key = min(self._archive, key=lambda k: self._archive[k].score or 0)
            del self._archive[worst_key]

    def sample(self, k: int = 2) -> list:
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


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if not OPENAI_API_KEY:
        print("ERROR: set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    print(f"=== AlphaEvolve for Qwen3-VL-2B-Thinking ===")
    print(f"  Population: {POPULATION_SIZE} ({NUM_ISLANDS} islands x {POPULATION_SIZE//NUM_ISLANDS})")
    print(f"  Max generations: {MAX_GENERATIONS}")
    print(f"  Pre-screen: {PRESCREEN_SAMPLES} samples, threshold={PRESCREEN_THRESHOLD}")
    print(f"  Mutation: dual-mode (focused prob={DIFF_MUTATION_PROB})")
    print()

    population = Population(num_islands=NUM_ISLANDS, island_size=POPULATION_SIZE // NUM_ISLANDS)
    archive = FeatureArchive(max_size=20)

    print("[Gen 0] Evaluating seed program (manual_thinking.py) ...")
    seed_code = SEED_FILE.read_text()
    seed_block = _extract_block(seed_code)
    seed = Program(code=seed_code, block=seed_block, generation=0, island=0)
    seed = evaluate_program(seed)
    print(f"  Seed: accuracy={seed.accuracy:.4f}, avg_time={seed.avg_time:.3f}s, score={seed.score:.4f}")
    population.add(seed)
    archive.add(seed)
    log_program(seed, 0)

    for i in range(1, NUM_ISLANDS):
        clone = Program(
            code=seed.code, block=seed.block,
            accuracy=seed.accuracy, avg_time=seed.avg_time,
            score=seed.score, generation=0, island=i,
            behavior=seed.behavior,
        )
        population.add(clone)

    best_ever = seed
    print(f"\n[Gen 0] Best: score={best_ever.score:.4f} (accuracy={best_ever.accuracy:.4f})\n")

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
            print(f"  [Gen {gen:03d}] Mutation failed.")
            continue

        child.island = island_idx
        print(f"  [Gen {gen:03d}] [{child.mutation_mode}] Evaluating ...", flush=True)
        child = evaluate_program(child, parent_accuracy=parent.accuracy or 0.0)

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
                  f"(accuracy={best_ever.accuracy:.4f})")
            OUTPUT_BEST.write_text(best_ever.code)
            print(f"  *** Saved to {OUTPUT_BEST}")

        if gen % 5 == 0:
            population.migrate()

        if gen % 10 == 0:
            print(f"\n--- Population at gen {gen} ---")
            for i, island in enumerate(population.islands):
                print(f"  Island {i}: {[f'{p.score:.4f}' for p in island]}")
            print(f"  Archive: {archive.summary()}")
            print(f"  Best: score={best_ever.score:.4f}\n")

    print(f"\n=== Evolution complete ===")
    print(f"  Best score : {best_ever.score:.4f}")
    print(f"  Accuracy   : {best_ever.accuracy:.4f}")
    print(f"  Avg time   : {best_ever.avg_time:.3f}s")
    OUTPUT_BEST.write_text(best_ever.code)
    print(f"  Output     : {OUTPUT_BEST}")
    print(f"  Log        : {EVOLVE_LOG}")


if __name__ == "__main__":
    main()
