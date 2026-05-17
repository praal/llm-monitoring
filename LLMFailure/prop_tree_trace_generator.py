#!/usr/bin/env python3
"""
Proposition Scaling × Tree-LTL Generator
==========================================

Combines the tree-LTL formula (b=2, d=2) with the proposition scaling
setup (N entities per step). The formula references specific entity slots.

Fixed k=10, varying N (entities per step): 1, 2, 4, 8, 16.
Multiple seeds for different formula/slot assignments.

Usage:
    python prop_tree_gen.py
    python prop_tree_gen.py --n_entities 1 2 4 8 16 --seeds 42 43 44 45 46
"""

import random
import json
import argparse
import os
from dataclasses import dataclass, field
from typing import Optional


# ── Attribute pools ──────────────────────────────────────────────────────────

ANIMALS = [
    "deer", "hawk", "wolf", "bear", "fox", "owl", "lynx", "crane",
    "otter", "bison", "eagle", "moose", "raven", "salmon", "heron",
    "badger", "falcon", "turtle", "rabbit", "cobra", "parrot", "whale",
    "tiger", "panda", "koala", "dolphin", "jaguar", "pelican", "viper",
    "gorilla", "penguin", "leopard", "sparrow", "beetle", "mantis",
    "coyote", "osprey", "iguana", "ferret", "marmot", "toucan",
    "gazelle", "macaw", "lemur", "wombat", "jackal", "condor",
    "hamster", "lobster", "gecko", "starling", "ibis", "newt",
    "panther", "elk", "finch",
]

SHAPES = [
    "oval", "square", "triangle", "circle", "diamond", "star",
    "hexagon", "pentagon", "crescent", "cross", "arrow", "heart",
    "trapezoid", "octagon", "spiral", "rectangle", "cube", "cone",
    "prism", "sphere", "ring", "wedge", "kite", "rhombus",
    "cylinder", "pyramid", "disc", "ellipse", "arch", "dome",
    "helix", "zigzag", "chevron", "droplet", "wave", "bolt",
    "orb", "slab", "spire", "notch", "loop", "band",
    "plank", "tile", "blade", "shield", "crest", "pillar",
    "shard", "crown", "flute", "knot", "lens", "petal", "rune",
]

COLORS = [
    "red", "blue", "green", "yellow", "purple", "orange",
    "white", "black", "silver", "gold", "pink", "brown",
    "teal", "maroon", "ivory", "crimson", "coral", "azure",
    "amber", "jade", "scarlet", "indigo", "bronze", "peach",
    "violet", "khaki", "olive", "beige", "magenta", "cyan",
    "rust", "plum", "sand", "mint", "slate", "cream",
    "ruby", "onyx", "lime", "pearl", "dusk", "dawn",
    "ash", "moss", "wine", "sage", "opal", "clay",
    "fawn", "iris", "lilac", "buff", "haze", "flax", "soot",
]

NUMBER_RANGE = (1, 100)

ENTITY_CONNECTORS = ["alongside", "with", "near", "beside", "next to"]


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Proposition:
    animal: str
    shape: str
    color: str
    number: int

    def to_dict(self):
        return {"animal": self.animal, "shape": self.shape,
                "color": self.color, "number": self.number}


@dataclass
class TreeNode:
    name: str
    children: list["TreeNode"] = field(default_factory=list)
    prop: Optional[Proposition] = None
    key_attr: Optional[str] = None    # "animal", "shape", or "number"
    key_value: Optional[str] = None   # the value used in the formula
    key_slot: Optional[int] = None    # 1-indexed entity slot

    @property
    def is_leaf(self):
        return len(self.children) == 0


# ── Tree construction ────────────────────────────────────────────────────────

def build_tree(b=2, d=2):
    counter = [0]
    all_nodes = []

    def make_node(depth):
        if depth == 0:
            node = TreeNode("root")
        else:
            counter[0] += 1
            node = TreeNode(f"n{counter[0]}")
        all_nodes.append(node)
        if depth < d:
            for _ in range(b):
                child = make_node(depth + 1)
                node.children.append(child)
        return node

    root = make_node(0)
    sink = TreeNode("sink")
    all_nodes.append(sink)
    return root, sink, all_nodes


def get_all_paths(root, sink):
    paths = []
    def dfs(node, current):
        current.append(node)
        if node.is_leaf:
            paths.append(current + [sink])
        else:
            for child in node.children:
                dfs(child, current[:])
    dfs(root, [])
    return paths


# ── Proposition + slot assignment ────────────────────────────────────────────

def assign_propositions(all_nodes, n_entities, rng):
    """
    Assign propositions, key_attr, key_value, and key_slot to each node.
    key_slot is random in [1, n_entities].
    Returns noise_pool (values not used by any node).
    """
    n = len(all_nodes)
    animals = rng.sample(ANIMALS, n)
    shapes = rng.sample(SHAPES, n)
    colors = rng.sample(COLORS, n)
    numbers = rng.sample(range(NUMBER_RANGE[0], NUMBER_RANGE[1] + 1), n)

    used = {"animals": set(), "shapes": set(), "colors": set(), "numbers": set()}
    attr_choices = ["animal", "shape", "number"]

    for i, node in enumerate(all_nodes):
        node.prop = Proposition(animals[i], shapes[i], colors[i], numbers[i])
        node.key_attr = rng.choice(attr_choices)
        if node.key_attr == "animal":
            node.key_value = animals[i]
        elif node.key_attr == "shape":
            node.key_value = shapes[i]
        else:
            node.key_value = str(numbers[i])
        node.key_slot = rng.randint(1, n_entities)

        used["animals"].add(animals[i])
        used["shapes"].add(shapes[i])
        used["colors"].add(colors[i])
        used["numbers"].add(numbers[i])

    noise_pool = {
        "animals": [a for a in ANIMALS if a not in used["animals"]],
        "shapes":  [s for s in SHAPES if s not in used["shapes"]],
        "colors":  [c for c in COLORS if c not in used["colors"]],
        "numbers": [n for n in range(NUMBER_RANGE[0], NUMBER_RANGE[1] + 1)
                     if n not in used["numbers"]],
    }
    return noise_pool


# ── Formula generation ───────────────────────────────────────────────────────

def _make_desc(node, n_entities):
    if node.key_attr == "number":
        val_str = f"number {node.key_value}"
    else:
        article = "an" if node.key_value[0].lower() in "aeiou" else "a"
        val_str = f"{article} {node.key_value}"

    if n_entities == 1:
        return f"observe {val_str}"
    else:
        return f"Entity {node.key_slot}'s {node.key_attr} is {val_str}"


def generate_nl_prompt(root, sink, paths, n_entities):
    sink_desc = _make_desc(sink, n_entities)

    def describe_node(node):
        desc = _make_desc(node, n_entities)
        if node.is_leaf:
            if n_entities == 1:
                return f"{desc}, and then eventually {sink_desc}"
            else:
                return f"{desc}, and then eventually {sink_desc}"
        child_descs = [describe_node(child) for child in node.children]
        if len(child_descs) == 1:
            return f"{desc}, and then eventually {child_descs[0]}"
        else:
            options = " OR ".join(f"({d})" for d in child_descs)
            return f"{desc}, and then eventually either: {options}"

    tree_desc = describe_node(root)

    lines = []
    lines.append("You are given a trace of observed events.")
    if n_entities > 1:
        lines.append(
            f"Each step describes {n_entities} labeled entities "
            f"(Entity 1, Entity 2, ..., Entity {n_entities}). "
            f"Each entity has an animal, a color, a shape, and a number."
        )
    lines.append("")
    lines.append("The trace is VALID if it satisfies the following constraint:")
    lines.append("")
    lines.append(f"Eventually {tree_desc}.")
    lines.append("")
    lines.append("The trace is INVALID otherwise.")
    lines.append("")
    lines.append(
        "Determine whether the following trace is VALID or INVALID. "
        "Respond with VALID or INVALID."
    )
    return "\n".join(lines)


# ── Event / entity helpers ───────────────────────────────────────────────────

def make_entity_from_prop(prop):
    """Entity tuple from a Proposition."""
    return (prop.animal, prop.color, prop.shape, prop.number)


def make_noise_entities(n_entities, rng, noise_pool, forbidden=None):
    """
    Generate N entities for a noise step.
    forbidden: dict mapping (attr, slot_0indexed) -> set of values to avoid.
    """
    entities = []
    for i in range(n_entities):
        animal = rng.choice(noise_pool["animals"]) if noise_pool["animals"] else rng.choice(ANIMALS)
        color = rng.choice(noise_pool["colors"]) if noise_pool["colors"] else rng.choice(COLORS)
        shape = rng.choice(noise_pool["shapes"]) if noise_pool["shapes"] else rng.choice(SHAPES)
        number = rng.choice(noise_pool["numbers"]) if noise_pool["numbers"] else rng.randint(*NUMBER_RANGE)

        if forbidden:
            if ("animal", i) in forbidden:
                bad = forbidden[("animal", i)]
                choices = [a for a in ANIMALS if a not in bad]
                if choices:
                    animal = rng.choice(choices)
            if ("color", i) in forbidden:
                bad = forbidden[("color", i)]
                choices = [c for c in COLORS if c not in bad]
                if choices:
                    color = rng.choice(choices)
            if ("shape", i) in forbidden:
                bad = forbidden[("shape", i)]
                choices = [s for s in SHAPES if s not in bad]
                if choices:
                    shape = rng.choice(choices)
            if ("number", i) in forbidden:
                bad = forbidden[("number", i)]
                choices = [n for n in range(*NUMBER_RANGE) if n not in bad]
                if choices:
                    number = rng.choice(choices)

        entities.append((animal, color, shape, number))
    return entities


def make_signal_entities(n_entities, rng, node, noise_pool):
    """
    Generate N entities for a path event step.
    The node's key_value is placed at the correct slot.
    Other slots get noise values.
    """
    entities = make_noise_entities(n_entities, rng, noise_pool)
    slot = node.key_slot - 1  # 0-indexed

    animal, color, shape, number = entities[slot]
    # Place the node's full proposition at the correct slot
    entities[slot] = make_entity_from_prop(node.prop)
    return entities


def make_corrupted_signal_entities(n_entities, rng, node, confuser_value, noise_pool):
    """
    Generate N entities for a corrupted path event step.
    The node's key_value is replaced by a confuser value at the correct slot.
    Optionally place the real value at a WRONG slot as distractor.
    """
    entities = make_noise_entities(n_entities, rng, noise_pool)
    slot = node.key_slot - 1

    # Place corrupted proposition at the correct slot
    prop = node.prop
    animal, color, shape, number = prop.animal, prop.color, prop.shape, prop.number
    if node.key_attr == "animal":
        animal = confuser_value
    elif node.key_attr == "shape":
        shape = confuser_value
    elif node.key_attr == "number":
        number = int(confuser_value)
    entities[slot] = (animal, color, shape, number)

    # Put real value at wrong slot as distractor (30% chance)
    if n_entities > 1 and rng.random() < 0.3:
        wrong_slots = [i for i in range(n_entities) if i != slot]
        ws = rng.choice(wrong_slots)
        a, c, s, num = entities[ws]
        if node.key_attr == "animal":
            a = node.key_value
        elif node.key_attr == "shape":
            s = node.key_value
        elif node.key_attr == "number":
            num = int(node.key_value)
        entities[ws] = (a, c, s, num)

    return entities


# ── Rendering ────────────────────────────────────────────────────────────────

def render_step(step_num, entities, rng):
    if len(entities) == 1:
        animal, color, shape, number = entities[0]
        conn = rng.choice(ENTITY_CONNECTORS)
        a_art = "an" if animal[0].lower() in "aeiou" else "a"
        c_art = "an" if color[0].lower() in "aeiou" else "a"
        return (f"Step {step_num}: Observed {c_art} {color} {shape} "
                f"(number {number}) {conn} {a_art} {animal}.")

    parts = []
    for i, (animal, color, shape, number) in enumerate(entities):
        conn = rng.choice(ENTITY_CONNECTORS)
        a_art = "an" if animal[0].lower() in "aeiou" else "a"
        c_art = "an" if color[0].lower() in "aeiou" else "a"
        parts.append(
            f"Entity {i+1}: {c_art} {color} {shape} "
            f"(number {number}) {conn} {a_art} {animal}"
        )
    return f"Step {step_num}: " + ". ".join(parts) + "."


# ── Confuser map (same as original tree code) ────────────────────────────────

def build_confuser_map(root, sink, all_nodes):
    depth_map = {}
    def walk(node, d):
        depth_map[node.name] = d
        for child in node.children:
            walk(child, d + 1)
    walk(root, 0)
    depth_map[sink.name] = max(depth_map.values()) + 1

    confuser_map = {}
    for node in all_nodes:
        node_depth = depth_map[node.name]
        cross_depth = []
        for other in all_nodes:
            if other.name == node.name:
                continue
            if other.key_attr != node.key_attr:
                continue
            if other.key_value == node.key_value:
                continue
            if depth_map[other.name] != node_depth:
                cross_depth.append(other.key_value)
        confuser_map[node.name] = cross_depth
    return confuser_map


# ── Tree cut for negative corruption ────────────────────────────────────────

def random_tree_cut(root, rng, split_prob=0.6):
    cut = []
    def recurse(node):
        if node.is_leaf:
            cut.append(node)
            return
        if rng.random() > split_prob:
            cut.append(node)
        else:
            for child in node.children:
                recurse(child)
    for child in root.children:
        recurse(child)
    return cut


def assign_corruption_from_cut(cut_nodes, paths):
    corruption_map = {}
    cut_names = {node.name for node in cut_nodes}
    for path_idx, path in enumerate(paths):
        for step_idx, node in enumerate(path):
            if node.name in cut_names:
                corruption_map[path_idx] = step_idx
                break
    return corruption_map


# ── Path checking (slot-aware) ───────────────────────────────────────────────

def check_any_valid_path(trace_steps, paths):
    """
    Check if any path's key_values appear at the correct entity slots
    in temporal order.
    trace_steps: list of entity lists (one per step).
    """
    for path in paths:
        pos = 0
        for step_entities in trace_steps:
            if pos >= len(path):
                break
            node = path[pos]
            slot = node.key_slot - 1
            if slot < len(step_entities):
                entity = step_entities[slot]
                # entity = (animal, color, shape, number)
                entity_values = {
                    str(entity[0]),  # animal
                    str(entity[1]),  # color
                    str(entity[2]),  # shape
                    str(entity[3]),  # number
                }
                if node.key_value in entity_values:
                    pos += 1
        if pos >= len(path):
            return True
    return False


# ── Trace generation ─────────────────────────────────────────────────────────

def generate_trace(paths, valid_path_idx, k, n_entities, noise_pool, rng,
                   confuser_map, all_nodes, max_paths=3, max_retries=100):
    num_all_paths = len(paths)
    path_depth = len(paths[0])

    for attempt in range(max_retries):
        # Select subset of paths
        if valid_path_idx is not None:
            # Positive: valid path + (max_paths-1) distractors
            other_idxs = [i for i in range(num_all_paths) if i != valid_path_idx]
            distractor_idxs = rng.sample(other_idxs, min(max_paths - 1, len(other_idxs)))
            selected_idxs = [valid_path_idx] + distractor_idxs
            rng.shuffle(selected_idxs)
        else:
            # Negative: random max_paths paths, all corrupted
            selected_idxs = rng.sample(range(num_all_paths), min(max_paths, num_all_paths))

        selected_paths = [paths[i] for i in selected_idxs]
        num_paths = len(selected_paths)

        # Build per-path entity lists
        path_step_entities = []

        if valid_path_idx is not None:
            for si_idx, pi in enumerate(selected_idxs):
                path = paths[pi]
                steps = []
                if pi == valid_path_idx:
                    # Valid path: all nodes get correct signal
                    for si, node in enumerate(path):
                        ents = make_signal_entities(n_entities, rng, node, noise_pool)
                        steps.append(ents)
                else:
                    # Corrupted distractor
                    corrupt_idx = rng.randint(0, path_depth - 1)
                    for si, node in enumerate(path):
                        if si == corrupt_idx:
                            confusers = confuser_map.get(node.name, [])
                            if confusers:
                                cv = rng.choice(confusers)
                            else:
                                pool_key = node.key_attr + "s" if node.key_attr != "number" else "numbers"
                                cv = str(rng.choice(noise_pool[pool_key]))
                            ents = make_corrupted_signal_entities(
                                n_entities, rng, node, cv, noise_pool)
                        else:
                            ents = make_signal_entities(n_entities, rng, node, noise_pool)
                        steps.append(ents)
                path_step_entities.append(steps)
        else:
            # Negative: all selected paths corrupted via tree cut
            root = paths[0][0]
            cut_nodes = random_tree_cut(root, rng)
            corruption_map = assign_corruption_from_cut(cut_nodes, paths)

            for si_idx, pi in enumerate(selected_idxs):
                path = paths[pi]
                steps = []
                corrupt_idx = corruption_map[pi]
                for si, node in enumerate(path):
                    if si == corrupt_idx:
                        confusers = confuser_map.get(node.name, [])
                        if confusers:
                            cv = rng.choice(confusers)
                        else:
                            pool_key = node.key_attr + "s" if node.key_attr != "number" else "numbers"
                            cv = str(rng.choice(noise_pool[pool_key]))
                        ents = make_corrupted_signal_entities(
                            n_entities, rng, node, cv, noise_pool)
                    else:
                        ents = make_signal_entities(n_entities, rng, node, noise_pool)
                    steps.append(ents)
                path_step_entities.append(steps)

        # Interleave into trace
        trace_steps = []

        # Prefix noise
        n_prefix = rng.randint(1, 10) + rng.randint(1, k)
        for _ in range(n_prefix):
            trace_steps.append(make_noise_entities(n_entities, rng, noise_pool))

        for round_idx in range(path_depth):
            order = list(range(num_paths))
            rng.shuffle(order)
            for pi in order:
                trace_steps.append(path_step_entities[pi][round_idx])
            if round_idx < path_depth - 1:
                for _ in range(k):
                    trace_steps.append(make_noise_entities(n_entities, rng, noise_pool))

        # Suffix noise
        n_suffix = rng.randint(1, 10) + rng.randint(1, k)
        for _ in range(n_suffix):
            trace_steps.append(make_noise_entities(n_entities, rng, noise_pool))

        # Verify against ALL paths (not just selected)
        actually_valid = check_any_valid_path(trace_steps, paths)
        expected_valid = valid_path_idx is not None

        if actually_valid == expected_valid:
            rendered = [render_step(i + 1, ents, rng) for i, ents in enumerate(trace_steps)]
            return {
                "label": "VALID" if actually_valid else "INVALID",
                "rendered_trace": rendered,
                "num_steps": len(trace_steps),
                "n_entities": n_entities,
            }

    # Fallback: return last attempt with corrected label
    actually_valid = check_any_valid_path(trace_steps, paths)
    rendered = [render_step(i + 1, ents, rng) for i, ents in enumerate(trace_steps)]
    return {
        "label": "VALID" if actually_valid else "INVALID",
        "rendered_trace": rendered,
        "num_steps": len(trace_steps),
        "n_entities": n_entities,
        "fallback": True,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Proposition scaling × tree-LTL generator"
    )
    parser.add_argument("--n_entities", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16])
    parser.add_argument("--b", type=int, default=2)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max_paths", type=int, default=6,
                        help="Max paths per trace (reduces trace length)")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Samples per seed (half pos, half neg)")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42])
    parser.add_argument("--output", type=str,
                        default="data/prop_tree_dataset.json")
    args = parser.parse_args()

    dataset = {
        "config": {
            "b": args.b, "d": args.d, "k": args.k,
            "max_paths": args.max_paths,
            "n_entities_values": args.n_entities,
            "n_samples_per_seed": args.n_samples,
            "seeds": args.seeds,
        },
        "samples_by_n": {},
    }

    for n_ent in args.n_entities:
        all_samples = []
        total_valid = 0
        total_invalid = 0
        total_fallback = 0

        print(f"\nGenerating n_entities={n_ent} "
              f"({args.n_samples} samples × {len(args.seeds)} seeds)...")

        for seed in args.seeds:
            rng = random.Random(seed)

            root, sink, all_nodes = build_tree(args.b, args.d)
            noise_pool = assign_propositions(all_nodes, n_ent, rng)
            paths = get_all_paths(root, sink)
            confuser_map = build_confuser_map(root, sink, all_nodes)
            nl_prompt = generate_nl_prompt(root, sink, paths, n_ent)

            # Show formula for this seed
            formula_preview = nl_prompt.split("Eventually ")[1].split("\n")[0][:80]
            print(f"  seed={seed}: Eventually {formula_preview}...")

            n_pos = args.n_samples // 2
            n_neg = args.n_samples - n_pos

            for i in range(n_pos):
                valid_idx = rng.randint(0, len(paths) - 1)
                trace = generate_trace(
                    paths, valid_idx, args.k, n_ent,
                    noise_pool, rng, confuser_map, all_nodes,
                    max_paths=args.max_paths)
                trace["sample_id"] = f"ne{n_ent}_seed{seed}_pos{i}"
                trace["seed"] = seed
                trace["prompt"] = (nl_prompt + "\n\n--- TRACE ---\n"
                                   + "\n".join(trace["rendered_trace"])
                                   + "\n--- END TRACE ---")
                all_samples.append(trace)
                if trace["label"] == "VALID":
                    total_valid += 1
                else:
                    total_invalid += 1
                if trace.get("fallback"):
                    total_fallback += 1

            for i in range(n_neg):
                trace = generate_trace(
                    paths, None, args.k, n_ent,
                    noise_pool, rng, confuser_map, all_nodes,
                    max_paths=args.max_paths)
                trace["sample_id"] = f"ne{n_ent}_seed{seed}_neg{i}"
                trace["seed"] = seed
                trace["prompt"] = (nl_prompt + "\n\n--- TRACE ---\n"
                                   + "\n".join(trace["rendered_trace"])
                                   + "\n--- END TRACE ---")
                all_samples.append(trace)
                if trace["label"] == "VALID":
                    total_valid += 1
                else:
                    total_invalid += 1
                if trace.get("fallback"):
                    total_fallback += 1

        dataset["samples_by_n"][str(n_ent)] = all_samples

        # Stats
        avg_steps = sum(s["num_steps"] for s in all_samples) / len(all_samples)
        print(f"  Total: {total_valid}V/{total_invalid}I, "
              f"fallbacks={total_fallback}, "
              f"avg_steps={avg_steps:.0f}, "
              f"samples={len(all_samples)}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nDataset saved to {args.output}")


if __name__ == "__main__":
    main()