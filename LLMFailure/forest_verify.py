#!/usr/bin/env python3
"""
Verify ground truth labels of the multi-constraint dataset.

For each sample, checks every constraint by scanning the trace for
a valid path in temporal order.

Usage:
    python verify_multi_labels.py
    python verify_multi_labels.py --dataset data/multi_constraint_dataset.json --verbose
"""

import json
import argparse


def check_any_valid_path(trace_events, path_nodes_list, propositions=None):
    """
    Check if any path's key_values appear in the trace in temporal order.
    path_nodes_list: list of paths, each path is a list of node name strings.
    If propositions dict is provided, looks up key_value from it.
    Otherwise assumes path items are dicts with 'key_value'.
    """
    for path in path_nodes_list:
        pos = 0
        for event in trace_events:
            if pos >= len(path):
                break
            key_value = path[pos]
            event_values = {
                str(event["animal"]),
                str(event["shape"]),
                str(event["color"]),
                str(event["number"]),
            }
            if key_value in event_values:
                pos += 1
        if pos >= len(path):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify multi-constraint dataset labels"
    )
    parser.add_argument("--dataset", type=str,
                        default="data/multi_constraint_dataset.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    formulas = dataset["formulas"]
    config = dataset["config"]
    print(f"Config: b={config['b']}, d={config['d']}, "
          f"nodes/tree={config['nodes_per_tree']}")
    print(f"Formulas: {len(formulas)}")

    # Use pre-stored tree paths (key_values per path per tree)
    all_trees = []
    for i, tree_paths in enumerate(dataset["tree_paths"]):
        all_trees.append({"tree_idx": i, "path_key_values": tree_paths})

    print(f"Trees: {len(all_trees)}, paths/tree: {len(all_trees[0]['path_key_values'])}")

    # Verify each sample
    total_constraints = 0
    total_correct = 0
    total_mismatches = 0
    mismatches = []

    for n_key in sorted(dataset["samples_by_n"].keys(), key=int):
        n_c = int(n_key)
        samples = dataset["samples_by_n"][n_key]
        trees = all_trees[:n_c]

        k_total = 0
        k_correct = 0

        for sample in samples:
            events = sample["events"]
            labels = sample["labels"]

            for i in range(n_c):
                expected = labels[i]
                result = check_any_valid_path(events, trees[i]["path_key_values"])
                got = "VALID" if result else "INVALID"

                total_constraints += 1
                k_total += 1

                if got == expected:
                    total_correct += 1
                    k_correct += 1
                    if args.verbose:
                        print(f"  ✓ {sample['sample_id']} constraint {i+1}: {expected}")
                else:
                    total_mismatches += 1
                    mismatches.append({
                        "sample_id": sample["sample_id"],
                        "constraint": i + 1,
                        "expected": expected,
                        "got": got,
                    })
                    print(f"  ✗ {sample['sample_id']} constraint {i+1}: "
                          f"expected={expected}, got={got}")

        print(f"n={n_c}: {k_correct}/{k_total} match")

    print(f"\nTotal: {total_correct}/{total_constraints} match")

    if mismatches:
        print(f"\n{len(mismatches)} MISMATCHES found:")
        for m in mismatches[:20]:
            print(f"  {m['sample_id']} constraint {m['constraint']}: "
                  f"expected={m['expected']}, got={m['got']}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")
    else:
        print("\nAll labels verified correctly!")


if __name__ == "__main__":
    main()