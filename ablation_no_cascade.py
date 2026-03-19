"""Ablation: evolution WITHOUT cascaded evaluation.

Identical to evolve_instruct.py except:
- No regression tests (phase 0)
- No crash check (phase 1)
- No pre-screen (phase 2)
- Every candidate gets full 128-sample evaluation directly.
Runs for 15 generations to get directional signal.
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

SEED_FILE = Path(__file__).parent / "manual_instruct.py"
ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "evolution_logs"
ABLATION_LOG = ARTIFACT_DIR / "ablation_no_cascade.jsonl"

OPENAI_MODEL = "gpt-5.2-codex"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

POPULATION_SIZE = 8
NUM_ISLANDS = 2
TOURNAMENT_K = 3
MAX_GENERATIONS = 15  # shortened for ablation
EVAL_NUM_SAMPLES = 128

BLOCK_START = "# EVOLVE-BLOCK-START"
BLOCK_END   = "# EVOLVE-BLOCK-END"

DIFF_MUTATION_PROB = 0.5

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

CHARXIV_PATH = Path(__file__).parent / "charxiv"
sys.path.insert(0, str(CHARXIV_PATH / "src"))

_SHARED_MODEL = None
_SHARED_PROCESSOR = None

def _init_shared_model():
    global _SHARED_MODEL, _SHARED_PROCESSOR
    if _SHARED_MODEL is not None: return
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if not torch.cuda.is_available(): dtype = torch.float32
    _SHARED_MODEL = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype, device_map="auto")
    _SHARED_MODEL.eval()
    _SHARED_PROCESSOR = AutoProcessor.from_pretrained(model_name, use_fast=False)

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

def evaluate_program(program, parent_accuracy=0.0):
    """NO cascaded evaluation — always run full 128 samples."""
    queries, gt_data = _get_queries()
    query_keys = list(queries.keys())
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=str(Path(__file__).parent)) as f:
        f.write(program.code)
        tmp_path = f.name
    try:
        spec = importlib.util.spec_from_file_location("_evolved_candidate", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if _SHARED_MODEL is not None:
            mod._MODEL = _SHARED_MODEL
            mod._PROCESSOR = _SHARED_PROCESSOR

        # NO regression tests, NO crash check, NO pre-screen
        # Go straight to full evaluation
        responses = {}
        num_errors = 0
        start_time = time.time()

        for key in query_keys:
            try:
                resp = mod.vlm_inference(image_path=queries[key]["figure_path"], question=queries[key]["question"])
                responses[key] = resp
            except Exception:
                responses[key] = "ERROR"
                num_errors += 1

        total_time = time.time() - start_time
        correct = 0; total = 0
        for key in query_keys:
            fig_id, subq_idx = key.split("_")
            gt_entry = gt_data.get(fig_id)
            if gt_entry is None: continue
            gt_answer = gt_entry["answers"][int(subq_idx)]
            if str(responses.get(key, "ERROR")).strip().lower() == gt_answer.lower(): correct += 1
            total += 1

        program.accuracy = correct / total if total > 0 else 0.0
        program.avg_time = total_time / total if total > 0 else 999.0
        baseline_time = 3.5
        speed_bonus = max(0.0, min(1.0, (baseline_time - program.avg_time) / baseline_time))
        program.score = 0.9 * program.accuracy + 0.1 * speed_bonus
    except Exception as e:
        program.accuracy = 0.0; program.avg_time = 999.0; program.score = 0.0
        program.eval_error = traceback.format_exc()
    finally:
        os.unlink(tmp_path)
    return program

def _extract_block(code):
    lines = code.splitlines(); inside = False; block_lines = []
    for line in lines:
        if BLOCK_START in line: inside = True; continue
        if BLOCK_END in line: break
        if inside: block_lines.append(line)
    return "\n".join(block_lines)

def _replace_block(full_code, new_block):
    before = full_code.split(BLOCK_START)[0]
    after = full_code.split(BLOCK_END)[1]
    return f"{before}{BLOCK_START}\n{new_block}\n{BLOCK_END}{after}"

def _is_valid_python(code):
    try: ast.parse(code); return True
    except SyntaxError: return False

def _extract_python_from_response(text):
    pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if matches: return max(matches, key=len).strip()
    return text.strip()

def _pick_random_function(block):
    try:
        tree = ast.parse(block)
        funcs = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if funcs: return random.choice(funcs)
    except SyntaxError: pass
    return None

def _build_mutation_prompt(parent, generation, other_programs, task_description):
    other_examples = ""
    if other_programs:
        examples = random.sample(other_programs, min(2, len(other_programs)))
        for i, p in enumerate(examples):
            other_examples += f"\n--- Example {i+1} (score={p.score:.4f}) ---\n{p.block}\n"
    mutation_strategies = [
        "Improve prompt for exact-match scoring.",
        "Add or improve post-processing.",
        "Optimise inference speed.",
        "Improve Not Applicable handling.",
        "Improve numeric extraction.",
        "Improve question-type prompts.",
        "Add answer normalisation.",
        "Add image preprocessing.",
    ]
    strategy = random.choice(mutation_strategies)
    return f"""You are an expert Python programmer improving a VLM inference function for CharXiv.
## Current program (score={parent.score:.4f}, accuracy={parent.accuracy:.4f})
```python
{parent.block}
```
## Strategy: {strategy}
## Other programs{other_examples if other_examples else " (none)"}
Output mutated EVOLVE-BLOCK only. Must define vlm_inference(image_path, question). Keep model name, use do_sample=False. Wrap in ```python ... ```"""

def _build_focused_mutation_prompt(parent, generation, other_programs, target_function):
    return f"""You are making a SURGICAL change to a VLM inference function.
## Current program (score={parent.score:.4f}, accuracy={parent.accuracy:.4f})
```python
{parent.block}
```
## Target: modify ONLY `{target_function}`, keep everything else identical. 1-15 lines.
Output COMPLETE EVOLVE-BLOCK. Keep model name, use do_sample=False. Wrap in ```python ... ```"""

def mutate(parent, generation, population, openai_client, archive_programs=None):
    others = [p for p in population if p is not parent and p.score is not None]
    if archive_programs:
        others = others + [p for p in archive_programs if p is not parent]
    use_focused = random.random() < DIFF_MUTATION_PROB
    target_func = None
    if use_focused:
        target_func = _pick_random_function(parent.block)
        if target_func is None: use_focused = False
    if use_focused:
        mutation_mode = "focused"
        prompt = _build_focused_mutation_prompt(parent, generation, others, target_func)
    else:
        mutation_mode = "full"
        prompt = _build_mutation_prompt(parent, generation, others, "CharXiv")
    try:
        response = openai_client.responses.create(
            model=OPENAI_MODEL, input=prompt,
            reasoning={"effort": "medium"}, text={"verbosity": "medium"}, max_output_tokens=6000,
        )
        raw = response.output_text
    except Exception as e:
        print(f"  [OpenAI error] {e}"); return None
    new_block = _extract_python_from_response(raw)
    if new_block is None or not _is_valid_python(new_block): return None
    seed_full_code = SEED_FILE.read_text()
    new_full_code = _replace_block(seed_full_code, new_block)
    return Program(code=new_full_code, block=new_block, generation=generation,
                   parent_scores=[parent.score], mutation_mode=mutation_mode)

class Population:
    def __init__(self, num_islands, island_size):
        self.num_islands = num_islands
        self.island_size = island_size
        self.islands = [[] for _ in range(num_islands)]
    def add(self, program):
        island = self.islands[program.island]
        island.append(program)
        island.sort(key=lambda p: p.score or 0.0, reverse=True)
        if len(island) > self.island_size: island[:] = island[:self.island_size]
    def tournament_select(self, island_idx):
        island = self.islands[island_idx]
        if not island: return None
        candidates = random.sample(island, min(TOURNAMENT_K, len(island)))
        return max(candidates, key=lambda p: p.score or 0.0)
    def migrate(self):
        for i, island in enumerate(self.islands):
            if not island: continue
            best = island[0]
            target = random.choice([j for j in range(self.num_islands) if j != i])
            emigrant = Program(code=best.code, block=best.block, accuracy=best.accuracy,
                avg_time=best.avg_time, score=best.score, generation=best.generation, island=target)
            self.islands[target].append(emigrant)
            self.islands[target].sort(key=lambda p: p.score or 0.0, reverse=True)
            if len(self.islands[target]) > self.island_size:
                self.islands[target][:] = self.islands[target][:self.island_size]
    def best(self):
        all_p = [p for isl in self.islands for p in isl]
        return max(all_p, key=lambda p: p.score or 0.0) if all_p else None
    def all_programs(self):
        return [p for isl in self.islands for p in isl]

# Archive kept for no-cascade ablation (archive is not being ablated here)
class FeatureArchive:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self._archive = {}
    def add(self, program):
        if not program.behavior or (program.score or 0) <= 0: return
        cell = program.behavior
        existing = self._archive.get(cell)
        if existing is None or (program.score or 0) > (existing.score or 0):
            self._archive[cell] = program
        if len(self._archive) > self.max_size:
            worst_key = min(self._archive, key=lambda k: self._archive[k].score or 0)
            del self._archive[worst_key]
    def sample(self, k=2):
        programs = list(self._archive.values())
        if not programs: return []
        return random.sample(programs, min(k, len(programs)))
    def size(self): return len(self._archive)

def log_program(program, generation):
    record = program.to_dict()
    record["generation_logged"] = generation
    record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(ABLATION_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if ABLATION_LOG.exists(): ABLATION_LOG.unlink()
    if not OPENAI_API_KEY:
        print("ERROR: set OPENAI_API_KEY"); sys.exit(1)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    _init_shared_model()
    print(f"=== ABLATION: No Cascade (15 gen) ===")
    population = Population(num_islands=NUM_ISLANDS, island_size=POPULATION_SIZE // NUM_ISLANDS)
    archive = FeatureArchive(max_size=50)
    seed_code = SEED_FILE.read_text()
    seed_block = _extract_block(seed_code)
    seed = Program(code=seed_code, block=seed_block, generation=0, island=0)
    seed = evaluate_program(seed)
    print(f"  Seed: accuracy={seed.accuracy:.4f}, score={seed.score:.4f}")
    population.add(seed)
    log_program(seed, 0)
    for i in range(1, NUM_ISLANDS):
        clone = Program(code=seed.code, block=seed.block, accuracy=seed.accuracy,
            avg_time=seed.avg_time, score=seed.score, generation=0, island=i)
        population.add(clone)
    best_ever = seed
    total_eval_time = 0.0  # track how much time spent on full evals
    for gen in range(1, MAX_GENERATIONS + 1):
        island_idx = (gen - 1) % NUM_ISLANDS
        parent = population.tournament_select(island_idx)
        if parent is None: continue
        print(f"[Gen {gen:02d}] parent={parent.score:.4f} ...", flush=True)
        child = mutate(parent, gen, population.all_programs(), openai_client,
                       archive_programs=archive.sample(2))
        if child is None: print("  mutation failed"); continue
        child.island = island_idx
        eval_start = time.time()
        child = evaluate_program(child)  # NO parent_accuracy -> no cascade
        eval_elapsed = time.time() - eval_start
        total_eval_time += eval_elapsed
        if child.eval_error:
            log_program(child, gen)
            print(f"  error ({eval_elapsed:.1f}s wasted): {child.eval_error[:60]}")
            continue
        print(f"  acc={child.accuracy:.4f}, score={child.score:.4f} ({eval_elapsed:.1f}s)")
        population.add(child)
        archive.add(child)
        log_program(child, gen)
        if child.score > best_ever.score:
            best_ever = child
            print(f"  *** New best! acc={best_ever.accuracy:.4f}")
        if gen % 5 == 0: population.migrate()
    print(f"\n=== No-Cascade ablation complete ===")
    print(f"  Best: acc={best_ever.accuracy:.4f}, score={best_ever.score:.4f}")
    print(f"  Total eval time: {total_eval_time:.1f}s")

if __name__ == "__main__":
    main()
