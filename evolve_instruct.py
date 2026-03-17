"""
AlphaEvolve-style evolution system for Qwen3-VL-2B-Instruct inference.

Core components (following AlphaEvolve paper):
  - LLM-based mutation operator (OpenAI GPT-5.2-Codex)
  - Population management with island model and diversity tracking
  - Evolution loop with tournament selection and elitism

Usage:
    python evolve_instruct.py

Outputs:
    evolved_instruct.py — best program found during evolution
    artifacts/evolution_logs/evolve_instruct_log.jsonl — full history of all evaluated programs
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
    eval_error: str = ""

    def to_dict(self):
        d = asdict(self)
        # keep block only (code is reconstructable)
        del d["code"]
        return d


# ── Evaluation ──────────────────────────────────────────────────────────────────

CHARXIV_PATH = Path(__file__).parent / "charxiv"
sys.path.insert(0, str(CHARXIV_PATH / "src"))


def _load_queries():
    from descriptive_utils import build_descriptive_quries
    with open(CHARXIV_PATH / "data" / "descriptive_val.json") as f:
        data = json.load(f)
    queries = build_descriptive_quries(data, str(CHARXIV_PATH / "images"))
    queries = dict(list(queries.items())[:EVAL_NUM_SAMPLES])
    return queries, data


_CACHED_QUERIES = None
_CACHED_GT = None


def _get_queries():
    global _CACHED_QUERIES, _CACHED_GT
    if _CACHED_QUERIES is None:
        _CACHED_QUERIES, _CACHED_GT = _load_queries()
    return _CACHED_QUERIES, _CACHED_GT


def evaluate_program(program: Program) -> Program:
    """Load program code in an isolated module and evaluate on CharXiv subset."""
    queries, gt_data = _get_queries()

    # Write to temp file so imports resolve correctly
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=str(Path(__file__).parent)
    ) as f:
        f.write(program.code)
        tmp_path = f.name

    try:
        spec = importlib.util.spec_from_file_location("_evolved_candidate", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        # Isolate module state
        spec.loader.exec_module(mod)

        num_errors = 0
        start_time = time.time()

        for query_key, query in queries.items():
            try:
                response = mod.vlm_inference(
                    image_path=query["figure_path"],
                    question=query["question"],
                )
                query["response"] = response
            except Exception as e:
                query["response"] = "ERROR"
                num_errors += 1

        total_time = time.time() - start_time

        correct = 0
        total = 0
        for query_key, query in queries.items():
            if "response" not in query:
                continue
            figure_id, subq_idx = query_key.split("_")
            gt_entry = gt_data.get(figure_id)
            if gt_entry is None:
                continue
            gt_answer = gt_entry["answers"][int(subq_idx)]
            if query["response"].strip().lower() == gt_answer.lower():
                correct += 1
            total += 1

        program.accuracy = correct / total if total > 0 else 0.0
        program.avg_time = total_time / total if total > 0 else 999.0

        # Speed bonus: normalised inverse time (baseline ~3.5s → 0, 0.25s → ~1)
        # Capped so accuracy always dominates
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
        # Clean up any internal image cache or model state would persist across
        # candidates (we deliberately reuse the loaded VLM for speed).

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


def mutate(
    parent: Program,
    generation: int,
    population: list,
    openai_client,
) -> Optional[Program]:
    """Call OpenAI to produce a mutated child program."""
    others = [p for p in population if p is not parent and p.score is not None]
    prompt = _build_mutation_prompt(parent, generation, others, task_description="CharXiv descriptive QA")

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

    print(f"=== AlphaEvolve for Qwen3-VL-2B-Instruct ===")
    print(f"  Population: {POPULATION_SIZE} ({NUM_ISLANDS} islands x {POPULATION_SIZE//NUM_ISLANDS})")
    print(f"  Max generations: {MAX_GENERATIONS}")
    print(f"  Eval samples: {EVAL_NUM_SAMPLES}")
    print(f"  Mutation LLM: {OPENAI_MODEL}")
    print()

    population = Population(
        num_islands=NUM_ISLANDS,
        island_size=POPULATION_SIZE // NUM_ISLANDS,
    )

    # ── Seed population with manual_instruct.py ─────────────────────────────
    print("[Gen 0] Evaluating seed program (manual_instruct.py) ...")
    seed_code = SEED_FILE.read_text()
    seed_block = _extract_block(seed_code)
    seed = Program(code=seed_code, block=seed_block, generation=0, island=0)
    seed = evaluate_program(seed)
    print(f"  Seed: accuracy={seed.accuracy:.4f}, avg_time={seed.avg_time:.3f}s, score={seed.score:.4f}")
    population.add(seed)
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

        print(f"parent score={parent.score:.4f} — mutating ...", flush=True)

        child = mutate(parent, gen, population.all_programs(), openai_client)
        if child is None:
            print(f"  [Gen {gen:03d}] Mutation failed, skipping.")
            continue

        child.island = island_idx
        print(f"  [Gen {gen:03d}] Evaluating child ...", flush=True)
        child = evaluate_program(child)
        print(f"  [Gen {gen:03d}] accuracy={child.accuracy:.4f}, "
              f"avg_time={child.avg_time:.3f}s, score={child.score:.4f}")

        population.add(child)
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
