#!/usr/bin/env python3
"""
Evaluate LLM on multi-constraint tree-LTL traces.

Usage:
    python eval_multi_constraint.py --model google/gemini-2.5-pro
    python eval_multi_constraint.py --model openai/gpt-4.1 --n_constraints 1 2 4
"""

import json
import argparse
import re
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import prompt_model


DATASET_PATH = "data/multi_constraint_dataset.json"
RESULTS_DIR = "results-multi"
SAVE_EVERY = 10


# ── Response parsing ─────────────────────────────────────────────────────────

def parse_multi_response(response_text, n_constraints):
    """
    Parse LLM response for N constraint verdicts.
    Looks for patterns like "Constraint 1: VALID" or "1: INVALID" or "1. VALID".
    Returns list of N verdicts ('VALID', 'INVALID', or 'UNPARSEABLE').
    """
    verdicts = ["UNPARSEABLE"] * n_constraints
    text = response_text.strip()

    for i in range(n_constraints):
        # Try multiple patterns
        patterns = [
            rf"Constraint\s*{i+1}\s*[:\-\.]\s*(VALID|INVALID)",
            rf"(?:^|\n)\s*{i+1}\s*[:\-\.]\s*(VALID|INVALID)",
            rf"Constraint\s*{i+1}\s*[:\-\.]\s*.*(VALID|INVALID)",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                val = match.group(1).upper()
                verdicts[i] = val
                break

    return verdicts


# ── Eval single sample ──────────────────────────────────────────────────────

def eval_single_sample(sample, model, temperature, max_tokens):
    prompt = sample["prompt"]
    n_c = sample["n_constraints"]
    labels = sample["labels"]

    try:
        response = prompt_model(
            prompt, model,
            system_prompt="",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        predictions = parse_multi_response(response, n_c)
    except Exception as e:
        response = str(e)
        predictions = ["UNPARSEABLE"] * n_c

    per_constraint = []
    for i in range(n_c):
        per_constraint.append({
            "constraint_idx": i,
            "ground_truth": labels[i],
            "prediction": predictions[i],
            "correct": predictions[i] == labels[i],
        })

    return {
        "sample_id": sample["sample_id"],
        "n_constraints": n_c,
        "per_constraint": per_constraint,
        "response": response,
    }


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_f1(results_list):
    """Compute per-constraint F1, precision, recall, and accuracy."""
    all_constraints = [c for r in results_list for c in r["per_constraint"]]

    # F1 with INVALID as positive class (detecting violations is the hard task)
    tp = sum(1 for c in all_constraints
             if c["ground_truth"] == "INVALID" and c["prediction"] == "INVALID")
    fp = sum(1 for c in all_constraints
             if c["ground_truth"] == "VALID" and c["prediction"] == "INVALID")
    fn = sum(1 for c in all_constraints
             if c["ground_truth"] == "INVALID" and c["prediction"] != "INVALID")
    tn = sum(1 for c in all_constraints
             if c["ground_truth"] == "VALID" and c["prediction"] == "VALID")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    total = len(all_constraints)
    correct = sum(1 for c in all_constraints if c["correct"])
    accuracy = correct / total if total > 0 else 0

    # Balanced accuracy
    pos = [c for c in all_constraints if c["ground_truth"] == "VALID"]
    neg = [c for c in all_constraints if c["ground_truth"] == "INVALID"]
    pos_acc = sum(1 for c in pos if c["correct"]) / len(pos) if pos else 0
    neg_acc = sum(1 for c in neg if c["correct"]) / len(neg) if neg else 0
    balanced_acc = (pos_acc + neg_acc) / 2

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "pos_accuracy": pos_acc,
        "neg_accuracy": neg_acc,
        "total_constraints": total,
        "n_valid": len(pos),
        "n_invalid": len(neg),
    }


# ── Save helper ──────────────────────────────────────────────────────────────

def save_results(all_results, output_path):
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM on multi-constraint tree-LTL traces"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=DATASET_PATH)
    parser.add_argument("--n_constraints", nargs="+", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    n_values = args.n_constraints or list(dataset["samples_by_n"].keys())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_slug = args.model.replace("/", "_").replace(".", "_")
    output_path = os.path.join(RESULTS_DIR, f"results_multi_{model_slug}.json")

    # Resume
    all_results = {
        "model": args.model,
        "temperature": args.temperature,
        "results_by_n": {},
    }
    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            prev = json.load(f)
        all_results = prev
        for n_key, n_data in prev.get("results_by_n", {}).items():
            existing[n_key] = {
                r["sample_id"]: r for r in n_data.get("samples", [])
                if r is not None
            }
        print(f"Resuming from {output_path}")

    save_lock = threading.Lock()

    for n_c in n_values:
        samples = dataset["samples_by_n"][n_c]
        if args.max_samples:
            samples = samples[:args.max_samples]

        done_map = existing.get(n_c, {})
        pending = [(i, s) for i, s in enumerate(samples)
                   if s["sample_id"] not in done_map]

        print(f"\n{'='*60}")
        print(f"n_constraints={n_c} ({len(samples)} samples, "
              f"{len(pending)} pending, {args.workers} workers)")
        print(f"{'='*60}")

        if not pending:
            print("  All done, skipping.")
            continue

        results = [None] * len(samples)
        for i, s in enumerate(samples):
            if s["sample_id"] in done_map:
                results[i] = done_map[s["sample_id"]]

        completed_count = 0
        total_pending = len(pending)

        def process(idx, sample):
            return idx, eval_single_sample(
                sample, args.model, args.temperature, args.max_tokens)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process, i, s): i for i, s in pending
            }

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed_count += 1

                n_correct = sum(1 for c in result["per_constraint"] if c["correct"])
                n_total = len(result["per_constraint"])
                print(f"  [{completed_count}/{total_pending}] "
                      f"{result['sample_id']}: {n_correct}/{n_total} correct")

                if completed_count % SAVE_EVERY == 0:
                    with save_lock:
                        done = [r for r in results if r is not None]
                        metrics = compute_f1(done)
                        all_results["results_by_n"][n_c] = {
                            "metrics": metrics,
                            "samples": done,
                        }
                        save_results(all_results, output_path)
                        print(f"  [checkpoint saved: {completed_count} samples]")

        # Final save for this n
        metrics = compute_f1(results)
        all_results["results_by_n"][n_c] = {
            "metrics": metrics,
            "samples": results,
        }
        save_results(all_results, output_path)

        m = metrics
        print(f"\n  F1: {m['f1']:.3f} | Precision: {m['precision']:.3f} | "
              f"Recall: {m['recall']:.3f}")
        print(f"  Balanced Acc: {m['balanced_accuracy']:.3f} | "
              f"Pos: {m['pos_accuracy']:.3f} | Neg: {m['neg_accuracy']:.3f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model}")
    print(f"{'='*60}")
    print(f"{'N':>4}  {'F1':>6}  {'BalAcc':>7}  {'Prec':>6}  {'Rec':>6}  "
          f"{'Pos':>6}  {'Neg':>6}")
    print(f"{'-'*4}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for n_c in n_values:
        if n_c in all_results["results_by_n"]:
            m = all_results["results_by_n"][n_c]["metrics"]
            print(f"{n_c:>4}  {m['f1']:>5.1%}  {m['balanced_accuracy']:>6.1%}  "
                  f"{m['precision']:>5.1%}  {m['recall']:>5.1%}  "
                  f"{m['pos_accuracy']:>5.1%}  {m['neg_accuracy']:>5.1%}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()