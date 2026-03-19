"""Evaluator for VLM inference using CharXiv benchmark.

Supports two inference modes:
  1. vlm_inference(image_path, question)          — single-query (all modules)
  2. vlm_inference_batch(image_path, questions)    — batched per-image (optional)

If the loaded module exposes vlm_inference_batch, the evaluator groups queries
by image and calls the batch path for better throughput.

Extra modes:
  --cv 4            Run 4-fold stability check and report per-fold + mean accuracy.
  --cv 4 --grouped  Same, but folds are grouped by figure_id (no image leaks across folds).
  --fold K/N        Evaluate only on fold K of N (e.g. --fold 2/4 for the second quarter).
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


def evaluate(program, fold=None, num_folds=None, grouped=False):
    NUM_SAMPLES = 128

    queries, ground_truth_data = load_charxiv_data(NUM_SAMPLES)

    # If fold is specified, restrict to that fold's samples.
    if fold is not None and num_folds is not None:
        all_keys = list(queries.keys())
        if grouped:
            # Group by figure_id so all questions for one chart stay in the same fold.
            figure_ids = list(OrderedDict.fromkeys(k.split("_")[0] for k in all_keys))
            figs_per_fold = len(figure_ids) // num_folds
            start_fig = (fold - 1) * figs_per_fold
            end_fig = figs_per_fold * fold if fold < num_folds else len(figure_ids)
            fold_figs = set(figure_ids[start_fig:end_fig])
            fold_keys = set(k for k in all_keys if k.split("_")[0] in fold_figs)
        else:
            fold_size = len(all_keys) // num_folds
            start_idx = (fold - 1) * fold_size
            end_idx = fold_size * fold if fold < num_folds else len(all_keys)
            fold_keys = set(all_keys[start_idx:end_idx])
        queries = {k: v for k, v in queries.items() if k in fold_keys}

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
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate VLM inference on CharXiv")
    parser.add_argument("module_name", help="Python module to evaluate")
    parser.add_argument("--cv", type=int, default=0, metavar="K",
                        help="Run K-fold stability check (e.g. --cv 4)")
    parser.add_argument("--fold", type=str, default=None, metavar="K/N",
                        help="Evaluate on fold K of N (e.g. --fold 2/4)")
    parser.add_argument("--grouped", action="store_true",
                        help="Group folds by figure_id (all questions for one chart in same fold)")
    args = parser.parse_args()

    program = importlib.import_module(args.module_name)

    if args.cv > 0:
        # K-fold cross-validation
        fold_results = []
        for k in range(1, args.cv + 1):
            print(f"\n--- Fold {k}/{args.cv}{' (grouped)' if args.grouped else ''} ---")
            result = evaluate(program, fold=k, num_folds=args.cv, grouped=args.grouped)
            fold_results.append(result)
            print(f"  Accuracy: {result['accuracy']:.4f} ({result['num_evaluated']} samples)")
        accuracies = [r["accuracy"] for r in fold_results]
        mean_acc = sum(accuracies) / len(accuracies)
        import statistics
        std_acc = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        mode_label = "grouped" if args.grouped else "sequential"
        print(f"\n=== {args.cv}-fold CV ({mode_label}): {mean_acc:.4f} ± {std_acc:.4f} ===")
        print(json.dumps({"cv_folds": args.cv, "grouped": args.grouped,
                          "per_fold": accuracies,
                          "mean": mean_acc, "std": std_acc}, indent=2))
    elif args.fold:
        k, n = map(int, args.fold.split("/"))
        results = evaluate(program, fold=k, num_folds=n)
        print(json.dumps(results, indent=2))
    else:
        results = evaluate(program)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
