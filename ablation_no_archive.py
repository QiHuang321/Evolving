"""Ablation: evolution WITHOUT behavioral diversity archive.

Identical to evolve_instruct.py except FeatureArchive is disabled.
Archive members are never sampled as inspiration during mutation.
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
ABLATION_LOG = ARTIFACT_DIR / "ablation_no_archive.jsonl"

OPENAI_MODEL = "gpt-5.2-codex"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

POPULATION_SIZE = 8
NUM_ISLANDS = 2
TOURNAMENT_K = 3
MAX_GENERATIONS = 15  # shortened for ablation
EVAL_NUM_SAMPLES = 128

BLOCK_START = "# EVOLVE-BLOCK-START"
BLOCK_END   = "# EVOLVE-BLOCK-END"

PRESCREEN_SAMPLES = 16
PRESCREEN_THRESHOLD = 0.7
PRESCREEN_STRATIFIED = True
DIFF_MUTATION_PROB = 0.5

REGRESSION_TESTS = [
    ("spatially lowest labeled tick", "-1.00", "-1.00"),
    ("spatially lowest labeled tick", "10^-6", "10^-6"),
    ("leftmost labeled tick", "SSM", "SSM"),
    ("how many discrete labels are there in the legend", "cat, dog, bird", "3"),
    ("general trend of data from left to right", "increasing", "increases"),
    ("layout of the subplots", "2 by 3", "2 by 3"),
    ("maximum value of the tick labels on the continuous legend",
     "There is no colorbar in this chart", "Not Applicable"),
]

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

_CACHED_QUERIES = None
_CACHED_GT = None
_CACHED_QTYPE = None

def _classify_question(question):
    q = " ".join(question.lower().split())
    if "continuous legend" in q: return "colorbar"
    if "legend" in q: return "legend"
    if "layout of the subplots" in q or "number of subplots" in q: return "layout"
    if any(p in q for p in ["leftmost","rightmost","lowest","highest","tick values","tick labels","how many"]): return "numeric"
    if "title" in q: return "title"
    if "label of the" in q or "axis" in q: return "axis"
    if "trend" in q or "intersect" in q: return "trend"
    return "other"

def _stratified_sample(query_keys, n):
    type_bins = {}
    for key in query_keys:
        qt = _CACHED_QTYPE.get(key, "other")
        type_bins.setdefault(qt, []).append(key)
    total = len(query_keys)
    selected = []
    for qt, keys in type_bins.items():
        k = max(1, round(n * len(keys) / total))
        selected.extend(random.sample(keys, min(k, len(keys))))
    if len(selected) > n:
        selected = random.sample(selected, n)
    elif len(selected) < n:
        remaining = [k for k in query_keys if k not in set(selected)]
        selected.extend(random.sample(remaining, min(n - len(selected), len(remaining))))
    return selected

def _get_queries():
    global _CACHED_QUERIES, _CACHED_GT, _CACHED_QTYPE
    if _CACHED_QUERIES is None:
        from descriptive_utils import build_descriptive_quries
        with open(CHARXIV_PATH / "data" / "descriptive_val.json") as f:
            data = json.load(f)
        queries = build_descriptive_quries(data, str(CHARXIV_PATH / "images"))
        _CACHED_QUERIES = dict(list(queries.items())[:EVAL_NUM_SAMPLES])
        _CACHED_GT = data
        _CACHED_QTYPE = {}
        for key, q in _CACHED_QUERIES.items():
            _CACHED_QTYPE[key] = _classify_question(q["question"])
    return _CACHED_QUERIES, _CACHED_GT

def evaluate_program(program, parent_accuracy=0.0):
    """Same cascaded evaluation as main system."""
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

        if parent_accuracy > 0 and hasattr(mod, '_normalize_answer'):
            for q_snippet, raw_out, expected in REGRESSION_TESTS:
                try:
                    result = mod._normalize_answer(q_snippet, raw_out)
                    if result != expected:
                        program.accuracy = 0.0; program.avg_time = 999.0; program.score = 0.0
                        program.eval_error = f"regression test fail"
                        return program
                except Exception: pass

        responses = {}
        num_errors = 0
        start_time = time.time()

        if parent_accuracy > 0:
            first_key = query_keys[0]
            first_q = queries[first_key]
            try:
                resp = mod.vlm_inference(image_path=first_q["figure_path"], question=first_q["question"])
                if not isinstance(resp, str): raise ValueError("non-string")
                responses[first_key] = resp
            except Exception as e:
                program.accuracy = 0.0; program.avg_time = 999.0; program.score = 0.0
                program.eval_error = f"crash: {e}"; return program

        if parent_accuracy > 0:
            if PRESCREEN_STRATIFIED:
                _prescreen_keys = _stratified_sample(query_keys, PRESCREEN_SAMPLES)
            else:
                _prescreen_keys = query_keys[:PRESCREEN_SAMPLES]
            for key in _prescreen_keys:
                if key in responses: continue
                try:
                    resp = mod.vlm_inference(image_path=queries[key]["figure_path"], question=queries[key]["question"])
                    responses[key] = resp
                except Exception: responses[key] = "ERROR"; num_errors += 1
            pre_correct = 0; pre_total = 0
            for key in _prescreen_keys:
                fig_id, subq_idx = key.split("_")
                gt_entry = gt_data.get(fig_id)
                if gt_entry is None: continue
                gt_answer = gt_entry["answers"][int(subq_idx)]
                if str(responses.get(key, "ERROR")).strip().lower() == gt_answer.lower(): pre_correct += 1
                pre_total += 1
            pre_acc = pre_correct / pre_total if pre_total > 0 else 0.0
            threshold = parent_accuracy * PRESCREEN_THRESHOLD
            if pre_acc < threshold:
                program.accuracy = pre_acc; program.avg_time = 999.0; program.score = 0.0
                program.eval_error = f"prescreen fail: {pre_acc:.3f} < {threshold:.3f}"
                return program

        for key in query_keys:
            if key in responses: continue
            try:
                resp = mod.vlm_inference(image_path=queries[key]["figure_path"], question=queries[key]["question"])
                responses[key] = resp
            except Exception: responses[key] = "ERROR"; num_errors += 1

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
        "Improve the SYSTEM or USER prompt for exact-match scoring.",
        "Add or improve post-processing logic.",
        "Optimise inference speed.",
        "Improve Not Applicable handling.",
        "Improve numeric answer extraction.",
        "Improve prompt for specific question types.",
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
## Target: modify ONLY `{target_function}`, keep everything else identical. 1-15 lines change.
Output COMPLETE EVOLVE-BLOCK. Keep model name, use do_sample=False. Wrap in ```python ... ```"""

def mutate(parent, generation, population, openai_client):
    others = [p for p in population if p is not parent and p.score is not None]
    # NO archive_programs — this is the ablation
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

def log_program(program, generation):
    record = program.to_dict()
    record["generation_logged"] = generation
    record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(ABLATION_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    # Clear previous ablation log
    if ABLATION_LOG.exists(): ABLATION_LOG.unlink()
    if not OPENAI_API_KEY:
        print("ERROR: set OPENAI_API_KEY"); sys.exit(1)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    _init_shared_model()
    print(f"=== ABLATION: No Archive (15 gen) ===")
    population = Population(num_islands=NUM_ISLANDS, island_size=POPULATION_SIZE // NUM_ISLANDS)
    # NO archive
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
    for gen in range(1, MAX_GENERATIONS + 1):
        island_idx = (gen - 1) % NUM_ISLANDS
        parent = population.tournament_select(island_idx)
        if parent is None: continue
        print(f"[Gen {gen:02d}] parent={parent.score:.4f} ...", flush=True)
        child = mutate(parent, gen, population.all_programs(), openai_client)
        if child is None: print("  mutation failed"); continue
        child.island = island_idx
        child = evaluate_program(child, parent_accuracy=parent.accuracy or 0.0)
        if child.eval_error and ("prescreen" in child.eval_error or "crash" in child.eval_error):
            log_program(child, gen); continue
        print(f"  acc={child.accuracy:.4f}, score={child.score:.4f}")
        population.add(child)
        log_program(child, gen)
        if child.score > best_ever.score:
            best_ever = child
            print(f"  *** New best! acc={best_ever.accuracy:.4f}")
        if gen % 5 == 0: population.migrate()
    print(f"\n=== No-Archive ablation complete ===")
    print(f"  Best: acc={best_ever.accuracy:.4f}, score={best_ever.score:.4f}")

if __name__ == "__main__":
    main()
