#!/usr/bin/env python3
"""
Verify ground truth labels of the multi-simple dataset.

For each sample, checks each constraint by scanning for the animal
then the color in temporal order.

Usage:
    python verify_multi_simple.py
    python verify_multi_simple.py --dataset data/multi_simple_dataset.json
"""

import json
import argparse


def verify_constraint(events, a_value, b_value):
    """Check F(A ∧ XF(B)): animal=a_value at step i, color=b_value at step j > i."""
    for i, e in enumerate(events):
        if e["animal"] == a_value:
            for j in range(i + 1, len(events)):
                if events[j]["color"] == b_value:
                    return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify multi-simple dataset labels"
    )
    parser.add_argument("--dataset", type=str,
                        default="data/multi_simple_dataset.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    config = dataset["config"]
    print(f"Config: n_constraints={config['n_constraints_values']}, "
          f"trace_len={config.get('trace_len', 'N/A')}, "
          f"gap={config.get('gap', 'N/A')}")

    total = 0
    correct = 0
    mismatches = []

    for n_key in sorted(dataset["samples_by_n"].keys(), key=int):
        n_c = int(n_key)
        samples = dataset["samples_by_n"][n_key]

        # Get formulas for this n
        formulas_key = f"formulas_{n_c}"
        if formulas_key not in dataset:
            print(f"  Warning: no formulas found for n={n_c}")
            continue

        # Parse a/b values from formula text
        import re
        specs = []
        for formula_nl in dataset[formulas_key]:
            m = re.search(r'animal is (?:a|an) (\w+).*color is (\w+)', formula_nl)
            if m:
                specs.append({"a_value": m.group(1), "b_value": m.group(2)})

        k_total = 0
        k_correct = 0

        for sample in samples:
            events = sample.get("events")
            if events is None:
                # Events not stored, skip
                print(f"  Warning: no events in {sample['sample_id']}, skipping")
                continue

            labels = sample["labels"]

            for i in range(n_c):
                expected = labels[i]
                result = verify_constraint(events, specs[i]["a_value"],
                                           specs[i]["b_value"])
                got = "VALID" if result else "INVALID"

                total += 1
                k_total += 1

                if got == expected:
                    correct += 1
                    k_correct += 1
                    if args.verbose:
                        print(f"  ✓ {sample['sample_id']} C{i+1}: {expected}")
                else:
                    mismatches.append({
                        "sample_id": sample["sample_id"],
                        "constraint": i + 1,
                        "expected": expected,
                        "got": got,
                    })
                    print(f"  ✗ {sample['sample_id']} C{i+1}: "
                          f"expected={expected}, got={got}")

        print(f"n={n_c}: {k_correct}/{k_total} match")

    print(f"\nTotal: {correct}/{total} match")

    if mismatches:
        print(f"\n{len(mismatches)} MISMATCHES found:")
        for m in mismatches[:20]:
            print(f"  {m['sample_id']} C{m['constraint']}: "
                  f"expected={m['expected']}, got={m['got']}")
    else:
        print("\nAll labels verified correctly!")


if __name__ == "__main__":
    main()