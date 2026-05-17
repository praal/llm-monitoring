"""
Temporal Elasticity (simple formula) — Evaluation Harness
=================================================

Evaluates LLM-as-a-Judge on temporal constraint monitoring traces.

Usage:
    python simple_evaluate.py --model google/gemini-2.5-flash
    python simple_evaluate.py --model google/gemini-2.5-pro --workers 5
    python simple_evaluate.py --model openai/gpt-4.1 --no-resume
    python simple_evaluate.py --model google/gemini-2.5-flash --max-traces 20
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from simple_trace_generator import (
    GAPS,
    Trace,
    format_constraint_description,
    format_trace_for_prompt,
    generate_all_traces,
)
from utils import prompt_model

# ─────────────────────────────────────────────
# Directories
# ─────────────────────────────────────────────

RESULTS_DIR = Path("results")
SEED = 42

# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

SYSTEM_PROMPT = ""

USER_PROMPT = """You evaluate whether event sequences satisfy or violate temporal constraints.
{constraint}

Here is the event sequence:

{trace}

Does this sequence satisfy or violate the constraint? Give your final answer as:
VERDICT: SATISFIED
or
VERDICT: VIOLATED"""


def build_prompt(trace: Trace) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a trace."""
    constraint = format_constraint_description("positive")
    usr = USER_PROMPT.format(constraint=constraint, trace=format_trace_for_prompt(trace))
    return SYSTEM_PROMPT, usr


# ─────────────────────────────────────────────
# Verdict parsing
# ─────────────────────────────────────────────

def parse_verdict(response: str) -> str | None:
    """
    Extract verdict from an LLM response. Handles:
      - "VERDICT: SATISFIED" / "VERDICT: VIOLATED"
      - "The constraint is satisfied/violated"
      - "The answer is SATISFIED"
      - Bare "SATISFIED" / "VIOLATED"
      - CoT with verdict at the end

    Returns 'satisfied', 'violated', or None.
    """
    if not response or not response.strip():
        return None

    text = response.strip()

    # 1) Explicit "VERDICT: X" (strongest signal)
    match = re.search(r"VERDICT\s*[:=]\s*(SATISFIED|VIOLATED)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # 2) "the constraint/sequence is satisfied/violated"
    match = re.search(
        r"(?:constraint|sequence|trace)\s+(?:is|was)\s+(satisfied|violated)",
        text, re.IGNORECASE,
    )
    if match:
        return match.group(1).lower()

    # 3) "the answer/result/conclusion is satisfied/violated"
    match = re.search(
        r"(?:answer|result|conclusion|judgment|verdict)\s+(?:is|:)\s*(satisfied|violated)",
        text, re.IGNORECASE,
    )
    if match:
        return match.group(1).lower()

    # 4) Last standalone occurrence (CoT mentions both, final answer is last)
    last_sat = -1
    last_vio = -1
    for m in re.finditer(r"\bSATISFIED\b", text, re.IGNORECASE):
        last_sat = m.start()
    for m in re.finditer(r"\bVIOLATED\b", text, re.IGNORECASE):
        last_vio = m.start()

    if last_sat == -1 and last_vio == -1:
        return None
    if last_sat > last_vio:
        return "satisfied"
    return "violated"


# ─────────────────────────────────────────────
# File I/O
# ─────────────────────────────────────────────

def results_filepath(model_name: str) -> Path:
    safe = model_name.replace("/", "_").replace(":", "_")
    return RESULTS_DIR / f"simple_{safe}.json"


def load_results(filepath: Path) -> dict[str, dict]:
    """Load existing results keyed by trace_id."""
    if filepath.exists():
        with open(filepath) as f:
            data = json.load(f)
        return {r["trace_id"]: r for r in data.get("results", [])}
    return {}


def save_results(filepath: Path, model_name: str, results: dict[str, dict]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "experiment": "Temporal Elasticity (simple formula)",
        "model": model_name,
        "num_results": len(results),
        "results": list(results.values()),
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────
# Single trace evaluation (with retries)
# ─────────────────────────────────────────────

MAX_RETRIES = 3


def evaluate_single(trace: Trace, model_name: str) -> dict:
    """Evaluate one trace with retries on empty/failed responses."""
    sys_prompt, usr_prompt = build_prompt(trace)
    response = ""

    for attempt in range(MAX_RETRIES):
        try:
            response = prompt_model(
                prompt=usr_prompt,
                model_name=model_name,
                system_prompt=sys_prompt,
                temperature=0.2,
                max_tokens=1000,
            )
        except Exception as e:
            response = ""
            if attempt == MAX_RETRIES - 1:
                print(f"  ERROR {trace.trace_id}: {e}")

        if response and response.strip():
            break
        time.sleep(1)

    predicted = parse_verdict(response)

    return {
        "trace_id": trace.trace_id,
        "gap": trace.gap,
        "a_position_label": trace.a_position_label,
        "a_position": trace.a_position,
        "trace_type": trace.trace_type,
        "trace_length": trace.trace_length,
        "ground_truth": trace.ground_truth_verdict,
        "predicted": predicted,
        "correct": predicted == trace.ground_truth_verdict,
        "raw_response": response[:2000],  # truncate to keep file size sane
    }


# ─────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────

FUTURE_TIMEOUT = 180  # seconds per trace


def run_evaluation(
    model_name: str,
    traces: list[Trace],
    resume: bool = True,
    max_traces: int | None = None,
    max_workers: int = 10,
) -> dict[str, dict]:
    """Evaluate model on traces with concurrent workers."""

    filepath = results_filepath(model_name)
    existing = load_results(filepath) if resume else {}
    if existing:
        print(f"Loaded {len(existing)} existing results.")

    pool = traces[:max_traces] if max_traces else traces
    pending = [t for t in pool if t.trace_id not in existing]
    skipped = len(pool) - len(pending)

    if not pending:
        print("Nothing to evaluate — all traces already done.")
        return existing

    print(f"Evaluating {len(pending)} traces ({skipped} skipped), "
          f"{max_workers} workers...")

    lock = threading.Lock()
    done_count = [0]
    error_count = [0]
    t0 = time.time()

    def on_result(result: dict) -> None:
        with lock:
            existing[result["trace_id"]] = result
            done_count[0] += 1
            if not result["predicted"]:
                error_count[0] += 1
            n = done_count[0]

        if n % 50 == 0 or n == len(pending):
            with lock:
                save_results(filepath, model_name, existing)
            elapsed = time.time() - t0
            rate = n / elapsed if elapsed > 0 else 0
            eta = (len(pending) - n) / rate if rate > 0 else 0
            print(f"  [{n:>5}/{len(pending)}] {n/len(pending)*100:5.1f}% | "
                  f"{rate:.1f} t/s | ETA {eta:.0f}s | "
                  f"{error_count[0]} unparsed")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_trace = {
            executor.submit(evaluate_single, t, model_name): t
            for t in pending
        }

        for future in as_completed(future_to_trace):
            trace = future_to_trace[future]
            try:
                result = future.result(timeout=FUTURE_TIMEOUT)
            except Exception as e:
                print(f"  TIMEOUT/ERROR {trace.trace_id}: {e}")
                result = {
                    "trace_id": trace.trace_id,
                    "gap": trace.gap,
                    "a_position_label": trace.a_position_label,
                    "a_position": trace.a_position,
                    "trace_type": trace.trace_type,
                    "trace_length": trace.trace_length,
                    "ground_truth": trace.ground_truth_verdict,
                    "predicted": None,
                    "correct": False,
                    "raw_response": f"FUTURE_ERROR: {e}",
                }
            on_result(result)

    save_results(filepath, model_name, existing)
    elapsed = time.time() - t0
    print(f"\nDone: {len(existing)} results saved to {filepath} "
          f"({elapsed:.0f}s, {error_count[0]} unparsed)")
    return existing


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Temporal Elasticity (simple formula): LLM Evaluation"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g. google/gemini-2.5-flash)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Concurrent API calls (default: 10)")
    parser.add_argument("--max-traces", type=int, default=None,
                        help="Limit traces (for testing)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore saved results")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating traces (seed={args.seed})...")
    traces = generate_all_traces(seed=args.seed)
    print(f"{len(traces)} traces generated.\n")

    run_evaluation(
        model_name=args.model,
        traces=traces,
        resume=not args.no_resume,
        max_traces=args.max_traces,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()