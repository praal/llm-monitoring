#!/usr/bin/env python3
"""
Evaluate an LLM on tree-LTL temporal elasticity traces.

Usage:
  python eval_tree_ltl.py --model gemini-2.5-pro
  python eval_tree_ltl.py --model gemini-2.5-pro --k_values 1 10 --max_traces 5
  python eval_tree_ltl.py --model gemini-2.5-pro --workers 8
"""

import json
import argparse
import re
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import prompt_model


DATASET_PATH = "data/tree_ltl_dataset.json"
RESULTS_DIR = "results-tree"
SAVE_EVERY = 10


# ── Prompt construction ──────────────────────────────────────────────────────

def build_prompt(nl_prompt, rendered_trace):
    trace_text = "\n".join(rendered_trace)
    return f"{nl_prompt}\n\n--- TRACE ---\n{trace_text}\n--- END TRACE ---"


# ── Response parsing ─────────────────────────────────────────────────────────

def parse_response(response_text):
    """
    Extract VALID or INVALID from the LLM response.
    Returns 'VALID', 'INVALID', or 'UNPARSEABLE'.
    """
    text = response_text.strip().upper()

    first_word = text.split()[0] if text.split() else ""
    if first_word.rstrip(".,:") == "VALID":
        return "VALID"
    if first_word.rstrip(".,:") == "INVALID":
        return "INVALID"

    has_invalid = bool(re.search(r"\bINVALID\b", text))
    has_valid = bool(re.search(r"\bVALID\b", text))

    if has_invalid and not has_valid:
        return "INVALID"
    if has_valid and not has_invalid:
        return "VALID"

    if has_valid and has_invalid:
        last_valid = max(m.start() for m in re.finditer(r"\bVALID\b", text))
        last_invalid = max(m.start() for m in re.finditer(r"\bINVALID\b", text))
        return "INVALID" if last_invalid > last_valid else "VALID"

    return "UNPARSEABLE"


# ── Single trace evaluation (called by workers) ─────────────────────────────

def eval_single_trace(trace, nl_prompt, model, temperature, max_tokens):
    """Evaluate one trace. Returns result dict."""
    prompt = build_prompt(nl_prompt, trace["rendered_trace"])
    trace_id = trace["trace_id"]
    ground_truth = trace["label"]

    try:
        response = prompt_model(
            prompt,
            model,
            system_prompt="",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        prediction = parse_response(response)
    except Exception as e:
        response = str(e)
        prediction = "UNPARSEABLE"

    return {
        "trace_id": trace_id,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "correct": prediction == ground_truth,
        "num_steps": trace["num_steps"],
        "response": response,
    }


# ── Compute summary from results list ───────────────────────────────────────

def compute_summary(k, results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    pos = [r for r in results if r["ground_truth"] == "VALID"]
    neg = [r for r in results if r["ground_truth"] == "INVALID"]
    pos_correct = sum(1 for r in pos if r["correct"])
    neg_correct = sum(1 for r in neg if r["correct"])

    return {
        "k": k,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "pos_total": len(pos),
        "pos_correct": pos_correct,
        "pos_accuracy": pos_correct / len(pos) if pos else 0,
        "neg_total": len(neg),
        "neg_correct": neg_correct,
        "neg_accuracy": neg_correct / len(neg) if neg else 0,
    }


# ── Save helper ──────────────────────────────────────────────────────────────

def save_results(all_results, output_path):
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM on tree-LTL traces"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=DATASET_PATH)
    parser.add_argument("--k_values", nargs="+", type=str, default=None)
    parser.add_argument("--max_traces", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of parallel workers")
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)

    nl_prompt = dataset["nl_prompt"]
    k_values = args.k_values or list(dataset["traces_by_k"].keys())

    # Prepare output
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_slug = args.model.replace("/", "_").replace(".", "_")
    b = dataset["tree"]["branching"]
    d = dataset["tree"]["depth"]
    output_path = os.path.join(RESULTS_DIR, f"results_b{b}_d{d}_{model_slug}.json")

    all_results = {
        "model": args.model,
        "temperature": args.temperature,
        "tree": {"branching": b, "depth": d},
        "results_by_k": {},
    }

    # Resume: load existing results if available
    existing_results = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            prev = json.load(f)
        all_results = prev
        for k_key, k_data in prev.get("results_by_k", {}).items():
            existing_results[k_key] = {
                r["trace_id"]: r for r in k_data.get("traces", [])
                if r is not None
            }
        print(f"Resuming from {output_path}")
        for k_key, done in existing_results.items():
            print(f"  k={k_key}: {len(done)} traces already done")

    save_lock = threading.Lock()

    for k in k_values:
        traces = dataset["traces_by_k"][k]
        if args.max_traces:
            traces = traces[:args.max_traces]

        # Figure out which traces still need to run
        done_map = existing_results.get(k, {})
        pending = [(i, t) for i, t in enumerate(traces)
                   if t["trace_id"] not in done_map]

        print(f"\n{'='*60}")
        print(f"k={k} ({len(traces)} traces, {len(pending)} pending, "
              f"{args.workers} workers)")
        print(f"{'='*60}")

        if not pending:
            print("  All traces already completed, skipping.")
            continue

        # Pre-fill results with existing data in order
        results = [None] * len(traces)
        for i, t in enumerate(traces):
            if t["trace_id"] in done_map:
                results[i] = done_map[t["trace_id"]]

        completed_count = 0
        total_pending = len(pending)

        def process_and_track(idx, trace):
            return idx, eval_single_trace(
                trace, nl_prompt, args.model,
                args.temperature, args.max_tokens,
            )

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_and_track, i, t): i
                for i, t in pending
            }

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed_count += 1

                status = "✓" if result["correct"] else "✗"
                print(f"  [{completed_count}/{total_pending}] "
                      f"{result['trace_id']} "
                      f"(GT={result['ground_truth']}) → "
                      f"{result['prediction']} {status}")

                # Save every SAVE_EVERY traces
                if completed_count % SAVE_EVERY == 0:
                    with save_lock:
                        done = [r for r in results if r is not None]
                        all_results["results_by_k"][k] = {
                            "summary": compute_summary(k, done),
                            "traces": done,
                        }
                        save_results(all_results, output_path)
                        print(f"  [checkpoint saved: {completed_count} traces]")

        # Final save for this k
        all_results["results_by_k"][k] = {
            "summary": compute_summary(k, results),
            "traces": results,
        }
        save_results(all_results, output_path)

        s = all_results["results_by_k"][k]["summary"]
        print(f"\n  Accuracy: {s['correct']}/{s['total']} = {s['accuracy']:.1%}")
        print(f"  Positive: {s['pos_correct']}/{s['pos_total']} = {s['pos_accuracy']:.1%}")
        print(f"  Negative: {s['neg_correct']}/{s['neg_total']} = {s['neg_accuracy']:.1%}")

    # Final summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model}")
    print(f"{'='*60}")
    print(f"{'k':>6}  {'Acc':>8}  {'Pos':>8}  {'Neg':>8}")
    print(f"{'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
    for k in k_values:
        s = all_results["results_by_k"][k]["summary"]
        print(f"{k:>6}  {s['accuracy']:>7.1%}  {s['pos_accuracy']:>7.1%}  {s['neg_accuracy']:>7.1%}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()