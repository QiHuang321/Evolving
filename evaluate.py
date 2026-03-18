"""Evaluator for VLM inference using CharXiv benchmark.

Supports two inference modes:
  1. vlm_inference(image_path, question)          — single-query (all modules)
  2. vlm_inference_batch(image_path, questions)    — batched per-image (optional)

If the loaded module exposes vlm_inference_batch, the evaluator groups queries
by image and calls the batch path for better throughput.
"""
import importlib
import sys
import json
import time
import traceback
from collections import OrderedDict
from pathlib import Path

CHARXIV_PATH = Path(__file__).parent / "charxiv"
sys.path.insert(0, str(CHARXIV_PATH / "src"))

from descriptive_utils import build_descriptive_quries


def load_charxiv_data(num_samples=128):
    with open(CHARXIV_PATH / "data" / "descriptive_val.json") as f:
        data = json.load(f)
    queries = build_descriptive_quries(data, str(CHARXIV_PATH / "images"))
    if num_samples is not None:
        queries = dict(list(queries.items())[:num_samples])
    return queries, data


def evaluate(program):
    NUM_SAMPLES = 128

    queries, ground_truth_data = load_charxiv_data(NUM_SAMPLES)

    num_errors = 0
    start_time = time.time()

    batch_fn = getattr(program, "vlm_inference_batch", None)

    if batch_fn is not None:
        # ── Batched path: group queries by image, call batch function ──
        groups = OrderedDict()           # image_path -> [(query_key, question)]
        for query_key, query in queries.items():
            img = query["figure_path"]
            groups.setdefault(img, []).append((query_key, query["question"]))

        for img_path, items in groups.items():
            questions = [q for _, q in items]
            try:
                responses = batch_fn(image_path=img_path, questions=questions)
                if len(responses) != len(items):
                    raise ValueError(
                        f"vlm_inference_batch returned {len(responses)} answers "
                        f"for {len(items)} questions on {img_path}"
                    )
                for (qk, _), resp in zip(items, responses):
                    queries[qk]["response"] = resp
            except Exception as e:
                print(f"Batch error on {img_path}: {e}")
                traceback.print_exc()
                for qk, _ in items:
                    queries[qk]["response"] = "ERROR"
                num_errors += len(items)
    else:
        # ── Single-query fallback ──
        for query_key, query in queries.items():
            try:
                response = program.vlm_inference(
                    image_path=query["figure_path"],
                    question=query["question"],
                )
                query["response"] = response
            except Exception as e:
                print(f"Error on {query_key}: {e}")
                traceback.print_exc()
                query["response"] = "ERROR"
                num_errors += 1

    total_time = time.time() - start_time

    correct = 0
    total = 0
    for query_key, query in queries.items():
        if "response" not in query:
            continue
        figure_id, subq_idx = query_key.split("_")
        gt_entry = ground_truth_data.get(figure_id)
        if gt_entry is None:
            continue
        gt_answer = gt_entry["answers"][int(subq_idx)]
        model_response = query["response"].strip()
        if model_response.lower() == gt_answer.lower():
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "num_evaluated": total, "num_errors": num_errors, "total_time": total_time, "avg_time_per_query": total_time / total if total > 0 else 0.0}


def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <module_name>")
        sys.exit(1)

    module_name = sys.argv[1]
    program = importlib.import_module(module_name)
    results = evaluate(program)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
