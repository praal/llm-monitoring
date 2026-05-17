#!/usr/bin/env python3
"""
Build the flat evaluation dataset from generated traces.

Takes the output of spec_level_experiment.py and produces a flat JSON file
where each sample = one trace × one spec level, with the full prompt baked in.

Usage:
    python build_dataset.py --input spec_level_traces.json --output data/spec_lang_dataset.json
"""
from __future__ import annotations

import json
import argparse
from spec_trace_generator import (
    Event, Trace, build_prompt,
    TRACE_PREAMBLE_INFORMAL, TRACE_PREAMBLE_PRECISE,
    SPEC_PROMPTS, TASK_SUFFIX,
)

FORMULA_TYPE_TO_PATTERN_ID = {
    "universality": 1,
    "absence": 2,
    "response": 3,
    "absence_between": 4,
    "constrained_response": 5,
    "tree_b2d1": 6,
    "tree_b2d4": 7,
}

PATTERN_NAMES = {
    1: "Universality",
    2: "Absence",
    3: "Response",
    4: "Absence/Between",
    5: "Constrained Resp",
    6: "Tree (b2,d1)",
    7: "Tree (b2,d4)",
}

SPEC_LEVELS = ["informal", "precise", "precise_ltl"]

GROUND_TRUTH_MAP = {
    "satisfies": "VALID",
    "violates": "INVALID",
}


def trace_from_dict(d: dict) -> Trace:
    """Reconstruct a Trace object from its JSON dict."""
    events = [
        Event(
            animal=e["animal"],
            shape=e["shape"],
            color=e["color"],
            number=e["number"],
        )
        for e in d["events"]
    ]
    return Trace(
        trace_id=d["trace_id"],
        formula_type=d["formula_type"],
        trace_category=d["trace_category"],
        edge_case_tag=d["edge_case_tag"],
        events=events,
        ground_truth=d["ground_truth"],
    )


def main():
    parser = argparse.ArgumentParser(description="Build flat evaluation dataset")
    parser.add_argument("--input", type=str, default="data/spec_level_traces.json")
    parser.add_argument("--output", type=str, default="data/spec_lang_dataset.json")
    args = parser.parse_args()

    with open(args.input) as f:
        raw = json.load(f)

    traces_data = raw["traces"]
    metadata = raw["metadata"]

    samples = []
    sample_idx = 0

    for td in traces_data:
        trace = trace_from_dict(td)
        pattern_id = FORMULA_TYPE_TO_PATTERN_ID[trace.formula_type]
        label = GROUND_TRUTH_MAP[trace.ground_truth]

        # Extract seed from trace_id (format: s{seed}_{type}_{num})
        seed = 0
        if trace.trace_id.startswith("s"):
            try:
                seed = int(trace.trace_id.split("_")[0][1:])
            except ValueError:
                pass

        for spec_level in SPEC_LEVELS:
            sample_idx += 1
            prompt = build_prompt(trace, spec_level)

            samples.append({
                "sample_id": f"sample_{sample_idx:05d}",
                "trace_id": trace.trace_id,
                "pattern_id": pattern_id,
                "formula_type": trace.formula_type,
                "spec_level": spec_level,
                "seed": seed,
                "trace_category": trace.trace_category,
                "edge_case_tag": trace.edge_case_tag,
                "label": label,
                "prompt": prompt,
            })

    dataset = {
        "metadata": {
            **metadata,
            "pattern_names": PATTERN_NAMES,
            "spec_levels": SPEC_LEVELS,
            "total_samples": len(samples),
            "traces_count": len(traces_data),
        },
        "samples": samples,
    }

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    # Summary
    print(f"Built {len(samples)} samples from {len(traces_data)} traces")
    print(f"  = {len(traces_data)} traces × {len(SPEC_LEVELS)} spec levels")
    print()

    for pid, pname in sorted(PATTERN_NAMES.items()):
        p_samples = [s for s in samples if s["pattern_id"] == pid]
        for level in SPEC_LEVELS:
            level_samples = [s for s in p_samples if s["spec_level"] == level]
            n_valid = sum(1 for s in level_samples if s["label"] == "VALID")
            n_invalid = sum(1 for s in level_samples if s["label"] == "INVALID")
            n_edge = sum(1 for s in level_samples
                         if s["trace_category"].startswith("edge"))
            print(f"  P{pid} {pname:>15} / {level:>12}: "
                  f"{len(level_samples):3d} samples "
                  f"({n_valid} valid, {n_invalid} invalid, {n_edge} edge)")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()