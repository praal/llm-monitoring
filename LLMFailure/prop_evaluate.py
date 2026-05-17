#!/usr/bin/env python3
"""
Evaluate LLM on proposition scaling traces.

Usage:
    python eval_prop_scaling.py --model google/gemini-2.5-pro
    python eval_prop_scaling.py --model openai/gpt-4.1 --n_entities 1 3 5
"""

import json
import argparse
import re
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import prompt_model


DATASET_PATH = "data/prop_scaling_dataset.json"
RESULTS_DIR = "results-prop"
SAVE_EVERY = 3


# ── Response parsing ─────────────────────────────────────────────────────────

def parse_response(response_text):
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


# ── Eval single sample ──────────────────────────────────────────────────────

def eval_single(sample, model, temperature, max_tokens):
    prompt = sample["prompt"]
    gt = sample["label"]

    try:
        response = prompt_model(
            prompt, model,
            system_prompt="",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        prediction = parse_response(response)
    except Exception as e:
        response = str(e)
        prediction = "UNPARSEABLE"

    return {
        "sample_id": sample["sample_id"],
        "seed": sample.get("seed", 0),
        "ground_truth": gt,
        "prediction": prediction,
        "correct": prediction == gt,
        "n_entities": sample["n_entities"],
        "response": response,
    }


# ── Metrics ──────────────────────────────────────────────────────────────────

import numpy as np

def compute_metrics(results):
    n = len(results)
    if n == 0:
        return {}

    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / n
    acc_sem = np.sqrt(accuracy * (1 - accuracy) / n)

    pos = [r for r in results if r["ground_truth"] == "VALID"]
    neg = [r for r in results if r["ground_truth"] == "INVALID"]
    pos_acc = sum(1 for r in pos if r["correct"]) / len(pos) if pos else 0
    neg_acc = sum(1 for r in neg if r["correct"]) / len(neg) if neg else 0
    bal_acc = (pos_acc + neg_acc) / 2

    sem_pos = np.sqrt(pos_acc * (1 - pos_acc) / len(pos)) if pos else 0
    sem_neg = np.sqrt(neg_acc * (1 - neg_acc) / len(neg)) if neg else 0
    bal_sem = np.sqrt(sem_pos**2 + sem_neg**2) / 2

    return {
        "accuracy": accuracy, "acc_sem": acc_sem,
        "bal_acc": bal_acc, "bal_sem": bal_sem,
        "pos_acc": pos_acc, "neg_acc": neg_acc,
        "n_valid": len(pos), "n_invalid": len(neg),
        "total": n,
    }


# ── Save helper ──────────────────────────────────────────────────────────────

def save_results(all_results, output_path):
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM on proposition scaling traces"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=DATASET_PATH)
    parser.add_argument("--n_entities", nargs="+", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    n_values = args.n_entities or list(dataset["samples_by_n"].keys())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_slug = args.model.replace("/", "_").replace(".", "_")
    output_path = os.path.join(RESULTS_DIR, f"results_prop_{model_slug}.json")

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

    for n_ent in n_values:
        samples = dataset["samples_by_n"][n_ent]
        if args.max_samples:
            samples = samples[:args.max_samples]

        done_map = existing.get(n_ent, {})
        pending = [(i, s) for i, s in enumerate(samples)
                   if s["sample_id"] not in done_map]

        print(f"\n{'='*60}")
        print(f"n_entities={n_ent} ({len(samples)} samples, "
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
            return idx, eval_single(
                sample, args.model, args.temperature, args.max_tokens)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process, i, s): i for i, s in pending
            }

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed_count += 1

                status = "✓" if result["correct"] else "✗"
                print(f"  [{completed_count}/{total_pending}] "
                      f"{result['sample_id']} "
                      f"(GT={result['ground_truth']}) → "
                      f"{result['prediction']} {status}")

                if completed_count % SAVE_EVERY == 0:
                    with save_lock:
                        done = [r for r in results if r is not None]
                        metrics = compute_metrics(done)
                        all_results["results_by_n"][n_ent] = {
                            "metrics": metrics,
                            "samples": done,
                        }
                        save_results(all_results, output_path)
                        print(f"  [checkpoint saved: {completed_count} samples]")

        # Final save
        metrics = compute_metrics(results)
        all_results["results_by_n"][n_ent] = {
            "metrics": metrics,
            "samples": results,
        }
        save_results(all_results, output_path)

        m = metrics
        print(f"\n  Balanced Acc: {m['bal_acc']:.1%} | "
              f"Pos: {m['pos_acc']:.1%} | Neg: {m['neg_acc']:.1%}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model}")
    print(f"{'='*60}")
    print(f"{'N':>4}  {'BalAcc':>8}  {'Pos':>7}  {'Neg':>7}  {'Acc':>7}")
    print(f"{'-'*4}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")
    for n_ent in n_values:
        if n_ent in all_results["results_by_n"]:
            m = all_results["results_by_n"][n_ent]["metrics"]
            print(f"{n_ent:>4}  {m['bal_acc']:>7.1%}  {m['pos_acc']:>6.1%}  "
                  f"{m['neg_acc']:>6.1%}  {m['accuracy']:>6.1%}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()