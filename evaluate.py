"""Evaluator for VLM inference using CharXiv benchmark.

Supports two inference modes:
  1. vlm_inference(image_path, question)          — single-query (all modules)
  2. vlm_inference_batch(image_path, questions)    — batched per-image (optional)

If the loaded module exposes vlm_inference_batch, the evaluator groups queries
by image and calls the batch path for better throughput.

Extra modes:
  --cv 4            Run 4-fold stability check and report per-fold + mean accuracy.
  --cv 4 --grouped  Same, but folds are grouped by figure_id AND stratified by
                    question type so each fold has a balanced mix.
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


def _classify_question(question: str) -> str:
    """Map a question to a coarse type for stratified splitting."""
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


def _figure_dominant_type(queries, figure_id, all_keys):
    """Return the most common question type for a figure (used for stratified grouping)."""
    from collections import Counter
    types = []
    for k in all_keys:
        if k.split("_")[0] == figure_id:
            types.append(_classify_question(queries[k]["question"]))
    if not types:
        return "other"
    return Counter(types).most_common(1)[0][0]


def _grouped_stratified_fold_keys(queries, all_keys, fold, num_folds):
    """Assign figures to folds via round-robin over question-type buckets.

    1. Classify each figure by its dominant question type.
    2. Within each type bucket, assign figures to folds via round-robin.
    3. Return the query keys belonging to the requested fold.

    This ensures:
    - All 4 questions for a given chart stay in the same fold (grouped).
    - Each fold gets an approximately balanced mix of question types (stratified).
    """
    # Get unique figure IDs in order
    figure_ids = list(OrderedDict.fromkeys(k.split("_")[0] for k in all_keys))

    # Classify each figure
    type_buckets = {}  # type -> [fig_id, ...]
    for fig_id in figure_ids:
        dtype = _figure_dominant_type(queries, fig_id, all_keys)
        type_buckets.setdefault(dtype, []).append(fig_id)

    # Round-robin assign figures to folds within each type bucket
    fold_figures = [set() for _ in range(num_folds)]
    for dtype in sorted(type_buckets.keys()):
        for i, fig_id in enumerate(type_buckets[dtype]):
            fold_figures[i % num_folds].add(fig_id)

    # Return keys for the requested fold (1-indexed)
    target_figs = fold_figures[fold - 1]
    return set(k for k in all_keys if k.split("_")[0] in target_figs)


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
            # Grouped by figure_id + stratified by question type.
            fold_keys = _grouped_stratified_fold_keys(queries, all_keys, fold, num_folds)
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
