#!/usr/bin/env python3
"""
Verify ground truth labels of the prop-tree dataset.

For each sample, reconstructs the tree with the same seed,
then checks if any path's key_values appear at the correct
entity slots in temporal order.

Usage:
    python verify_prop_tree.py
    python verify_prop_tree.py --dataset data/prop_tree_dataset.json --verbose
"""

import json
import argparse
import re


# ── Slot-aware path checking ────────────────────────────────────────────────

def extract_entities_from_rendered(rendered_step, n_entities):
    """
    Parse entity attributes from a rendered step string.
    Returns list of (animal, color, shape, number) tuples.
    """
    if n_entities == 1:
        # "Step N: Observed a/an COLOR SHAPE (number NUM) CONN a/an ANIMAL."
        m = re.search(
            r'Observed (?:a|an) (\w+) (\w+) \(number (\d+)\) .+? (?:a|an) (\w+)',
            rendered_step
        )
        if m:
            color, shape, number, animal = m.group(1), m.group(2), m.group(3), m.group(4)
            return [(animal, color, shape, number)]
        return []

    # Multi-entity: "Entity N: a/an COLOR SHAPE (number NUM) CONN a/an ANIMAL"
    entities = []
    pattern = r'Entity \d+: (?:a|an) (\w+) (\w+) \(number (\d+)\) .+? (?:a|an) (\w+)'
    for m in re.finditer(pattern, rendered_step):
        color, shape, number, animal = m.group(1), m.group(2), m.group(3), m.group(4)
        entities.append((animal, color, shape, number))
    return entities


def check_path_valid(rendered_trace, path_key_values, path_key_slots,
                     path_key_attrs, n_entities):
    """
    Check if a path's key_values appear at the correct entity slots
    in temporal order in the rendered trace.
    """
    pos = 0  # position in path
    for step_text in rendered_trace:
        if pos >= len(path_key_values):
            break
        entities = extract_entities_from_rendered(step_text, n_entities)
        slot = path_key_slots[pos] - 1  # 0-indexed
        if slot < len(entities):
            entity = entities[slot]
            # entity = (animal, color, shape, number)
            entity_values = {str(entity[0]), str(entity[1]),
                           str(entity[2]), str(entity[3])}
            if path_key_values[pos] in entity_values:
                pos += 1
    return pos >= len(path_key_values)


def verify_from_raw_events(sample, tree_paths_kv, tree_paths_slots):
    """
    Verify using raw events if available (from the dataset's internal format).
    Falls back to rendered trace parsing.
    """
    # Use rendered trace parsing since raw entity data isn't stored
    n_ent = sample["n_entities"]
    rendered = sample["rendered_trace"]

    for path_kv, path_slots in zip(tree_paths_kv, tree_paths_slots):
        if check_path_valid(rendered, path_kv, path_slots, None, n_ent):
            return "VALID"
    return "INVALID"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify prop-tree dataset labels"
    )
    parser.add_argument("--dataset", type=str,
                        default="data/prop_tree_dataset.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    config = dataset["config"]
    print(f"Config: b={config['b']}, d={config['d']}, k={config['k']}")
    print(f"Seeds: {config['seeds']}")
    print(f"N values: {config['n_entities_values']}")

    # We need to reconstruct trees for each seed to get paths + slots
    from prop_tree_trace_generator import (
        build_tree, get_all_paths, assign_propositions
    )
    import random

    total = 0
    correct = 0
    mismatches = []

    for n_key in sorted(dataset["samples_by_n"].keys(), key=int):
        n_ent = int(n_key)
        samples = dataset["samples_by_n"][n_key]

        # Group samples by seed
        by_seed = {}
        for s in samples:
            by_seed.setdefault(s["seed"], []).append(s)

        k_total = 0
        k_correct = 0

        for seed, seed_samples in sorted(by_seed.items()):
            # Reconstruct tree with this seed
            rng = random.Random(seed)
            root, sink, all_nodes = build_tree(config["b"], config["d"])
            assign_propositions(all_nodes, n_ent, rng)
            paths = get_all_paths(root, sink)

            # Extract key_values and key_slots for each path
            tree_paths_kv = [[node.key_value for node in path] for path in paths]
            tree_paths_slots = [[node.key_slot for node in path] for path in paths]

            for sample in seed_samples:
                expected = sample["label"]
                got = verify_from_raw_events(sample, tree_paths_kv, tree_paths_slots)

                total += 1
                k_total += 1

                if got == expected:
                    correct += 1
                    k_correct += 1
                    if args.verbose:
                        print(f"  ✓ {sample['sample_id']}: {expected}")
                else:
                    mismatches.append({
                        "sample_id": sample["sample_id"],
                        "expected": expected,
                        "got": got,
                        "seed": seed,
                    })
                    print(f"  ✗ {sample['sample_id']}: "
                          f"expected={expected}, got={got}")

        print(f"n={n_ent}: {k_correct}/{k_total} match")

    print(f"\nTotal: {correct}/{total} match")

    if mismatches:
        print(f"\n{len(mismatches)} MISMATCHES found:")
        for m in mismatches[:20]:
            print(f"  {m['sample_id']} (seed={m['seed']}): "
                  f"expected={m['expected']}, got={m['got']}")
    else:
        print("\nAll labels verified correctly!")


if __name__ == "__main__":
    main()