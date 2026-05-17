#!/usr/bin/env python3
"""
Evaluate LLM on spec-level experiment.

Usage:
    python eval_spec_level.py --model google/gemini-2.5-flash
    python eval_spec_level.py --model google/gemini-2.5-pro --patterns 3 4
    python eval_spec_level.py --model openai/gpt-4.1 --levels informal precise
    python eval_spec_level.py --model anthropic/claude-sonnet-4 --edge_only

Requires:
    - utils.py with prompt_model() function
    - data/spec_lang_dataset.json (built by build_dataset.py)
"""
from __future__ import annotations

import json
import argparse
import re
import os
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# Try to import prompt_model; provide stub if missing
try:
    from utils import prompt_model
except ImportError:
    def prompt_model(prompt, model, system_prompt="", temperature=0.2,
                     max_tokens=1000):
        raise RuntimeError(
            "utils.prompt_model not found. "
            "Please provide a utils.py with a prompt_model() function."
        )


DATASET_PATH = "data/spec_lang_dataset.json"
RESULTS_DIR = "results-spec-level"
SAVE_EVERY = 10

PATTERN_NAMES = {
    1: "Universality",
    2: "Absence",
    3: "Response",
    4: "Absence/Between",
    5: "Constrained Resp",
    6: "Tree (b2,d1)",
    7: "Tree (b2,d4)",
}


# ── Response parsing ─────────────────────────────────────────────────────────

def parse_response(response_text: str) -> str:
    """
    Parse LLM response to extract SATISFIES/VIOLATES verdict.
    Maps to VALID/INVALID for consistency with dataset labels.
    """
    text = response_text.strip().upper()
    first_word = text.split()[0] if text.split() else ""

    # Direct match on first word
    fw = first_word.rstrip(".,:;")
    if fw == "SATISFIES" or fw == "VALID":
        return "VALID"
    if fw == "VIOLATES" or fw == "INVALID":
        return "INVALID"

    # Search for keywords
    has_violates = bool(re.search(r"\b(VIOLATES|VIOLATED|VIOLATION|INVALID)\b", text))
    has_satisfies = bool(re.search(r"\b(SATISFIES|SATISFIED|VALID)\b", text))

    if has_violates and not has_satisfies:
        return "INVALID"
    if has_satisfies and not has_violates:
        return "VALID"

    # Both present — use last occurrence
    if has_satisfies and has_violates:
        last_sat = max(
            (m.start() for m in re.finditer(
                r"\b(SATISFIES|SATISFIED|VALID)\b", text)),
            default=-1
        )
        last_viol = max(
            (m.start() for m in re.finditer(
                r"\b(VIOLATES|VIOLATED|VIOLATION|INVALID)\b", text)),
            default=-1
        )
        return "INVALID" if last_viol > last_sat else "VALID"

    return "UNPARSEABLE"


# ── Eval single sample ──────────────────────────────────────────────────────

def eval_single(sample: dict, model: str, temperature: float,
                max_tokens: int, timeout: int | None = None) -> dict:
    """Evaluate a single sample and return the result dict."""
    prompt = sample["prompt"]
    gt = sample["label"]

    try:
        if timeout:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    prompt_model, prompt, model,
                    system_prompt="", temperature=temperature,
                    max_tokens=max_tokens,
                )
                response = future.result(timeout=timeout)
        else:
            response = prompt_model(
                prompt, model,
                system_prompt="",
                temperature=temperature,
                max_tokens=max_tokens,
            )
        prediction = parse_response(response)
    except (TimeoutError, concurrent.futures.TimeoutError):
        response = "TIMEOUT"
        prediction = "UNPARSEABLE"
    except Exception as e:
        response = str(e)
        prediction = "UNPARSEABLE"

    return {
        "sample_id": sample["sample_id"],
        "trace_id": sample["trace_id"],
        "pattern_id": sample["pattern_id"],
        "formula_type": sample["formula_type"],
        "spec_level": sample["spec_level"],
        "seed": sample.get("seed", 0),
        "trace_category": sample.get("trace_category", ""),
        "edge_case_tag": sample.get("edge_case_tag", "none"),
        "ground_truth": gt,
        "prediction": prediction,
        "correct": prediction == gt,
        "response": response,
    }


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy, balanced accuracy, and per-class accuracy."""
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

    # Edge case breakdown
    edge = [r for r in results if r.get("trace_category", "").startswith("edge")]
    clear = [r for r in results if not r.get("trace_category", "").startswith("edge")]
    edge_acc = sum(1 for r in edge if r["correct"]) / len(edge) if edge else 0
    clear_acc = sum(1 for r in clear if r["correct"]) / len(clear) if clear else 0

    unparseable = sum(1 for r in results if r["prediction"] == "UNPARSEABLE")

    return {
        "accuracy": accuracy,
        "acc_sem": acc_sem,
        "bal_acc": bal_acc,
        "bal_sem": bal_sem,
        "pos_acc": pos_acc,
        "neg_acc": neg_acc,
        "edge_acc": edge_acc,
        "clear_acc": clear_acc,
        "n_valid": len(pos),
        "n_invalid": len(neg),
        "n_edge": len(edge),
        "n_clear": len(clear),
        "n_unparseable": unparseable,
        "total": n,
    }


# ── Save helper ──────────────────────────────────────────────────────────────

def save_results(all_results: dict, output_path: str):
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM on spec-level experiment"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=DATASET_PATH)
    parser.add_argument("--patterns", nargs="+", type=int, default=None,
                        help="Only evaluate these pattern IDs (1=Univ, "
                             "2=Absence, 3=Response, 4=Absence/Between)")
    parser.add_argument("--levels", nargs="+", type=str, default=None,
                        help="Only evaluate these spec levels "
                             "(informal, precise, precise_ltl)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Only evaluate traces from these seeds")
    parser.add_argument("--edge_only", action="store_true",
                        help="Only evaluate edge case traces")
    parser.add_argument("--clear_only", action="store_true",
                        help="Only evaluate clear (non-edge) traces")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout per API call in seconds (0=no timeout)")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)

    timeout = args.timeout if args.timeout > 0 else None

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_slug = args.model.replace("/", "_").replace(".", "_")
    output_path = args.output or os.path.join(
        RESULTS_DIR, f"results_{model_slug}.json")

    # Filter samples
    samples = dataset["samples"]
    if args.patterns:
        samples = [s for s in samples if s["pattern_id"] in args.patterns]
    if args.levels:
        samples = [s for s in samples if s["spec_level"] in args.levels]
    if args.seeds:
        samples = [s for s in samples if s.get("seed") in args.seeds]
    if args.edge_only:
        samples = [s for s in samples
                   if s.get("trace_category", "").startswith("edge")]
    if args.clear_only:
        samples = [s for s in samples
                   if not s.get("trace_category", "").startswith("edge")]
    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples to evaluate: {len(samples)}")

    # Group by (pattern, level) for organized evaluation
    by_key = {}
    for s in samples:
        key = f"p{s['pattern_id']}_{s['spec_level']}"
        by_key.setdefault(key, []).append(s)

    # Resume from previous results if file exists
    all_results = {
        "model": args.model,
        "temperature": args.temperature,
        "dataset": args.dataset,
        "results_by_key": {},
    }
    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            prev = json.load(f)
        all_results = prev
        for key, key_data in prev.get("results_by_key", {}).items():
            existing[key] = {
                r["sample_id"]: r for r in key_data.get("samples", [])
                if r is not None
            }
        n_done = sum(len(v) for v in existing.values())
        n_unparseable = sum(
            1 for v in existing.values()
            for r in v.values() if r.get("prediction") == "UNPARSEABLE"
        )
        print(f"Resuming from {output_path} ({n_done} samples already done, "
              f"{n_unparseable} UNPARSEABLE will be retried)")

    save_lock = threading.Lock()

    for key in sorted(by_key.keys()):
        key_samples = by_key[key]
        done_map = existing.get(key, {})
        # Retry samples that are not done OR were UNPARSEABLE
        pending = [(i, s) for i, s in enumerate(key_samples)
                   if s["sample_id"] not in done_map
                   or done_map[s["sample_id"]].get("prediction") == "UNPARSEABLE"]

        pat_id = key_samples[0]["pattern_id"]
        level = key_samples[0]["spec_level"]
        pat_name = PATTERN_NAMES.get(pat_id, f"P{pat_id}")

        print(f"\n{'='*60}")
        print(f"{pat_name} / {level} "
              f"({len(key_samples)} samples, {len(pending)} pending)")
        print(f"{'='*60}")

        if not pending:
            print("  All done, skipping.")
            continue

        results = [None] * len(key_samples)
        for i, s in enumerate(key_samples):
            if s["sample_id"] in done_map:
                results[i] = done_map[s["sample_id"]]

        completed_count = 0
        total_pending = len(pending)

        def process(idx, sample):
            return idx, eval_single(
                sample, args.model, args.temperature, args.max_tokens,
                timeout=timeout)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process, i, s): i for i, s in pending
            }

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed_count += 1

                status = "✓" if result["correct"] else "✗"
                edge_tag = (f" [{result['edge_case_tag']}]"
                            if result.get("edge_case_tag", "none") != "none"
                            else "")
                print(f"  [{completed_count}/{total_pending}] "
                      f"{result['sample_id']} "
                      f"(GT={result['ground_truth']}) → "
                      f"{result['prediction']} {status}{edge_tag}")

                if completed_count % SAVE_EVERY == 0:
                    with save_lock:
                        for r in results:
                            if r is not None:
                                done_map[r["sample_id"]] = r
                        all_samples = list(done_map.values())
                        metrics = compute_metrics(all_samples)
                        all_results["results_by_key"][key] = {
                            "metrics": metrics,
                            "samples": all_samples,
                        }
                        save_results(all_results, output_path)

        # Final save for this key
        for r in results:
            if r is not None:
                done_map[r["sample_id"]] = r
        all_samples = list(done_map.values())
        metrics = compute_metrics(all_samples)
        all_results["results_by_key"][key] = {
            "metrics": metrics,
            "samples": all_samples,
        }
        save_results(all_results, output_path)

        m = metrics
        print(f"\n  Balanced Acc: {m['bal_acc']:.1%} ± {m['bal_sem']:.1%} | "
              f"Pos: {m['pos_acc']:.1%} | Neg: {m['neg_acc']:.1%} | "
              f"Edge: {m['edge_acc']:.1%} | Clear: {m['clear_acc']:.1%}")

    # ── Summary table ────────────────────────────────────────────────────

    print(f"\n{'='*80}")
    print(f"SUMMARY: {args.model}")
    print(f"{'='*80}")

    # Header
    print(f"{'Pattern':>17} | {'Level':>12} | {'BalAcc':>12} | "
          f"{'Pos':>7} | {'Neg':>7} | {'Edge':>7} | {'Clear':>7} | "
          f"{'N':>5}")
    print(f"{'-'*17}-+-{'-'*12}-+-{'-'*12}-+-"
          f"{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}")

    # Rows grouped by pattern
    for pid in sorted(PATTERN_NAMES.keys()):
        pat_name = PATTERN_NAMES[pid]
        for level in ["informal", "precise", "precise_ltl"]:
            key = f"p{pid}_{level}"
            data = all_results["results_by_key"].get(key, {})
            m = data.get("metrics", {})
            if not m:
                continue
            print(f"{pat_name:>17} | {level:>12} | "
                  f"{m['bal_acc']:>5.1%}±{m['bal_sem']:>4.1%} | "
                  f"{m['pos_acc']:>6.1%} | {m['neg_acc']:>6.1%} | "
                  f"{m['edge_acc']:>6.1%} | {m['clear_acc']:>6.1%} | "
                  f"{m['total']:>5d}")
        print()  # blank line between patterns

    # Aggregate by spec level
    print(f"\n{'--- Aggregate by Spec Level ---':>40}")
    for level in ["informal", "precise", "precise_ltl"]:
        level_results = []
        for key, data in all_results["results_by_key"].items():
            if key.endswith(f"_{level}"):
                level_results.extend(data.get("samples", []))
        if level_results:
            m = compute_metrics(level_results)
            print(f"  {level:>12}: BalAcc={m['bal_acc']:.1%}±{m['bal_sem']:.1%} | "
                  f"Edge={m['edge_acc']:.1%} | Clear={m['clear_acc']:.1%} | "
                  f"N={m['total']}")

    # Aggregate by pattern
    print(f"\n{'--- Aggregate by Pattern ---':>40}")
    for pid in sorted(PATTERN_NAMES.keys()):
        pat_name = PATTERN_NAMES[pid]
        pat_results = []
        for key, data in all_results["results_by_key"].items():
            if key.startswith(f"p{pid}_"):
                pat_results.extend(data.get("samples", []))
        if pat_results:
            m = compute_metrics(pat_results)
            print(f"  P{pid} {pat_name:>15}: BalAcc={m['bal_acc']:.1%}±{m['bal_sem']:.1%} | "
                  f"Edge={m['edge_acc']:.1%} | Clear={m['clear_acc']:.1%} | "
                  f"N={m['total']}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()