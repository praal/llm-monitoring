#!/usr/bin/env python3
"""
Tree-based LTL formula trace generator for temporal elasticity experiments.

Fixed tree: configurable b (branching) and d (depth) with a shared sink node.
8 root-to-leaf-to-sink paths, 16 tree nodes total.

Generates traces (positive and negative) for varying k (minimum step gap).
- Positive: exactly 1 path fully valid, all others corrupted at 1 step
- Negative: all paths corrupted at exactly 1 step (at leaf level to prevent
  cross-path event sharing from accidentally creating valid paths)
"""

import random
import json
import argparse
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
    key_attr: Optional[str] = None   # "animal", "shape", or "number"
    key_value: Optional[str] = None  # the value used in the formula

    @property
    def is_leaf(self):
        return len(self.children) == 0


# ── Tree construction ────────────────────────────────────────────────────────

def build_tree(b=3, d=3, extra_depth=0):
    """
    Build a full b-ary tree of depth d, plus optional extra chain nodes
    after each leaf, plus a shared sink node.

    extra_depth=1 with b=2, d=3 gives:
      depth 0: 1 root
      depth 1: 2 nodes
      depth 2: 4 nodes
      depth 3: 8 nodes (original leaves)
      depth 4: 8 extra nodes (one per original leaf)
      + 1 sink
      = 24 nodes total, 8 paths of length 6

    Returns (root, sink, all_nodes_list)
    """
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
        elif depth < d + extra_depth:
            # Extra chain: one child per node
            child = make_node(depth + 1)
            node.children.append(child)
        return node

    root = make_node(0)
    sink = TreeNode("sink")
    all_nodes.append(sink)

    return root, sink, all_nodes


def get_all_paths(root, sink):
    """Return all root-to-leaf paths, each ending with sink."""
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


# ── Proposition assignment ───────────────────────────────────────────────────

def assign_propositions(all_nodes, rng):
    """
    Assign a unique (animal, shape, color, number) to each node.
    Also assign each node a key_attr (animal, shape, or number) — the
    single attribute used to identify it in the formula.
    Returns the noise pool (attribute values not used by any tree node).
    """
    n = len(all_nodes)  # 16
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


# ── LTL formula generation ──────────────────────────────────────────────────

def generate_ltl_symbolic(root, sink):
    """Generate symbolic LTL formula from the tree using only key_values."""
    def recurse(node):
        if node.is_leaf:
            return f"{node.key_value} ∧ XF {sink.key_value}"
        child_parts = " ∨ ".join(f"({recurse(c)})" for c in node.children)
        return f"{node.key_value} ∧ XF ({child_parts})"

    return f"F ({recurse(root)})"


def generate_nl_prompt(root, sink, paths):
    """
    Generate the natural-language prompt given to the LLM being evaluated.
    Uses a nested tree-structured description mirroring the LTL formula.
    Each node is identified by a single attribute (animal, shape, or number).
    """

    def make_desc(node):
        if node.key_attr == "number":
            return f"number {node.key_value}"
        else:
            article = "an" if node.key_value[0].lower() in "aeiou" else "a"
            return f"{article} {node.key_value}"

    sink_desc = make_desc(sink)

    def describe_node(node):
        """Recursively build flat English for the tree structure."""
        desc = make_desc(node)

        if node.is_leaf:
            return (f"observe {desc}, and then eventually "
                    f"observe {sink_desc}")

        child_descs = [describe_node(child) for child in node.children]

        if len(child_descs) == 1:
            return (f"observe {desc}, and then eventually "
                    f"{child_descs[0]}")
        else:
            options = " OR ".join(f"({d})" for d in child_descs)
            return f"observe {desc}, and then eventually either: {options}"

    # Build the flat formula description
    tree_desc = describe_node(root)

    lines = []
    lines.append(
        "You are given a trace of observed events."
    )
    lines.append("")
    lines.append(
        "The trace is VALID if it satisfies the following constraint:"
    )
    lines.append("")
    lines.append(
        f"Eventually {tree_desc}."
    )
    lines.append("")
    lines.append(
        "The trace is INVALID otherwise."
    )
    lines.append("")
    lines.append(
        "Determine whether the following trace is VALID or INVALID. "
        "Respond with VALID or INVALID."
    )

    return "\n".join(lines)


# ── Event helpers ────────────────────────────────────────────────────────────

def make_event(prop):
    return prop.to_dict()


def make_noise_event(noise_pool, rng, all_tree_nodes=None, inject_prob=0.2):
    """
    Create a random noise event. With probability inject_prob, inject a
    random tree node's key_value into the event. This makes tree key_values
    appear throughout the trace, preventing the LLM from just scanning for
    rare keywords.
    """
    event = {
        "animal": rng.choice(noise_pool["animals"]),
        "shape":  rng.choice(noise_pool["shapes"]),
        "color":  rng.choice(noise_pool["colors"]),
        "number": rng.choice(noise_pool["numbers"]),
    }
    if all_tree_nodes and rng.random() < inject_prob:
        node = rng.choice(all_tree_nodes)
        if node.key_attr == "number":
            event["number"] = int(node.key_value)
        else:
            event[node.key_attr] = node.key_value
    return event


def build_confuser_map(root, sink, all_nodes):
    """
    For each node, build a list of "confuser" key_values from other tree nodes
    at DIFFERENT depths. Since the path checker is greedy and all key_values
    are unique, a value from depth D' appearing at depth D (where D != D')
    can never help complete any path — it either was already consumed at its
    correct earlier position, or appears too early to be useful.

    This guarantees no cross-path leakage while still using real tree values.

    Falls back to same-depth siblings if no cross-depth confusers exist.
    Returns: dict mapping node.name -> list of confuser key_values
    """
    # Compute depth of each node
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


def corrupt_event(event, rng, key_attr, confuser_values, noise_pool):
    """
    Return a copy with the key attribute swapped to a confuser value
    (from a sibling/cousin in the tree). Falls back to noise pool if
    no same-attr confusers exist.
    """
    corrupted = dict(event)
    if confuser_values:
        new_val = rng.choice(confuser_values)
    else:
        pool_key = key_attr + "s" if key_attr != "number" else "numbers"
        new_val = rng.choice(noise_pool[pool_key])

    if key_attr == "number":
        corrupted["number"] = int(new_val)
    else:
        corrupted[key_attr] = new_val
    return corrupted


SENTENCE_TEMPLATES = [
    "Step {step}: Observed a {color} {shape} (number {number}) alongside a {animal}.",
    "Step {step}: A {animal} appeared near a {color} {shape}, tagged {number}.",
    "Step {step}: Spotted a {color} {shape} labeled {number} next to a {animal}.",
    "Step {step}: Recorded a {animal} beside a {color} {shape} with value {number}.",
    "Step {step}: A {color} {shape} (#{number}) was seen together with a {animal}.",
    "Step {step}: Encountered a {animal} and a {color} {shape} marked {number}.",
    "Step {step}: Logged a {color} {shape} numbered {number} in the presence of a {animal}.",
    "Step {step}: A {animal} was noted next to a {color} {shape} bearing the number {number}.",
    "Step {step}: Detected a {color} {shape} ({number}) accompanied by a {animal}.",
    "Step {step}: Found a {animal} alongside a {color} {shape}, identifier {number}.",
    "Step {step}: Witnessed a {color} {shape} with number {number} near a {animal}.",
    "Step {step}: A {color} {shape} carrying label {number} appeared with a {animal}.",
]


def render_step(step_num, event, rng):
    template = rng.choice(SENTENCE_TEMPLATES)
    return template.format(
        step=step_num,
        color=event["color"],
        shape=event["shape"],
        number=event["number"],
        animal=event["animal"],
    )


# ── Tree cut generation ──────────────────────────────────────────────────────

def random_tree_cut(root, rng, split_prob=0.6):
    """
    Generate a random tree cut (antichain that every root-to-leaf path
    passes through exactly once).

    Algorithm: top-down decision at each node.
      - If leaf: must include it (no children to recurse into).
      - If internal: with probability (1 - split_prob), include this node
        in the cut (all paths through here corrupt here). Otherwise,
        recurse into children.

    split_prob controls depth distribution:
      - 0.0: always cut at root (all paths corrupt at one shared node)
      - 0.5: balanced mix of depths
      - 1.0: always cut at leaves (original behavior)

    Returns: list of TreeNode forming the cut.
    """
    cut = []

    def recurse(node):
        if node.is_leaf:
            cut.append(node)
            return
        if rng.random() > split_prob:
            # Cut here — all paths through this node corrupt at this node
            cut.append(node)
        else:
            # Split — recurse into children
            for child in node.children:
                recurse(child)

    # Always recurse past root — root cuts are too easy to detect
    for child in root.children:
        recurse(child)
    return cut


def assign_corruption_from_cut(cut_nodes, paths):
    """
    Given a tree cut, determine which path index corrupts at which step.

    Returns: dict mapping path_idx -> step_idx (position in path to corrupt)
    """
    corruption_map = {}
    cut_names = {node.name for node in cut_nodes}

    for path_idx, path in enumerate(paths):
        for step_idx, node in enumerate(path):
            if node.name in cut_names:
                corruption_map[path_idx] = step_idx
                break

    return corruption_map


# ── Trace generation ─────────────────────────────────────────────────────────

def check_any_valid_path(trace_events, paths):
    """
    Check if any path's key_values can be found in temporal order
    in the actual interleaved trace event list.
    """
    for path in paths:
        key_values = [node.key_value for node in path]
        pos = 0
        for event in trace_events:
            if pos >= len(key_values):
                break
            event_values = {
                str(event["animal"]),
                str(event["shape"]),
                str(event["color"]),
                str(event["number"]),
            }
            if key_values[pos] in event_values:
                pos += 1
        if pos >= len(key_values):
            return True
    return False


def build_corrupted_events(paths, valid_path_idx, neg_corruption_map,
                           confuser_map, noise_pool, rng):
    """
    Build per-path event lists with corruption applied.
    Always uses cross-depth tree values; falls back to noise pool if no
    same-attr cross-depth confusers exist.
    Returns (path_events, corruption_info).
    """
    path_depth = len(paths[0])
    path_events = []
    corruption_info = []

    for i, path in enumerate(paths):
        events = [make_event(node.prop) for node in path]
        if i == valid_path_idx:
            corruption_info.append(None)
        else:
            if valid_path_idx is None:
                corrupt_idx = neg_corruption_map[i]
            else:
                corrupt_idx = rng.randint(0, path_depth - 1)

            corrupted_node = path[corrupt_idx]
            confusers = confuser_map.get(corrupted_node.name, [])

            events[corrupt_idx] = corrupt_event(
                events[corrupt_idx], rng,
                corrupted_node.key_attr,
                confusers,
                noise_pool,
            )
            corruption_info.append({
                "path_idx": i,
                "corrupted_step": corrupt_idx,
                "node_name": corrupted_node.name,
                "key_attr": corrupted_node.key_attr,
            })
        path_events.append(events)

    return path_events, corruption_info


def _interleave(path_events, paths, k, noise_pool, rng, num_paths, path_depth, all_tree_nodes=None, target_injections=10):
    """Build the interleaved trace with noise padding.
    
    target_injections: expected number of noise events with tree values,
    kept constant regardless of k to prevent high-k traces from being
    flooded with tree values (which makes negative sample generation fail).
    """
    # Estimate total noise events to compute inject_prob
    n_between = (path_depth - 1) * k
    n_bookend = 2 * (5 + k // 2)  # rough estimate of prefix + suffix
    total_noise_est = max(n_between + n_bookend, 1)
    inject_prob = min(target_injections / total_noise_est, 0.3) if all_tree_nodes else 0.0

    trace_events = []
    trace_metadata = []

    # Noise at beginning
    n_prefix = rng.randint(1, 10) + rng.randint(1, k)
    for _ in range(n_prefix):
        trace_events.append(make_noise_event(noise_pool, rng, all_tree_nodes, inject_prob))
        trace_metadata.append({"type": "noise"})

    for round_idx in range(path_depth):
        order = list(range(num_paths))
        rng.shuffle(order)
        for path_idx in order:
            trace_events.append(path_events[path_idx][round_idx])
            trace_metadata.append({
                "type": "path_event",
                "path_idx": path_idx,
                "round": round_idx,
                "node": paths[path_idx][round_idx].name,
            })
        if round_idx < path_depth - 1:
            for _ in range(k):
                trace_events.append(make_noise_event(noise_pool, rng, all_tree_nodes, inject_prob))
                trace_metadata.append({"type": "noise"})

    # Noise at end
    n_suffix = rng.randint(1, 10) + rng.randint(1, k)
    for _ in range(n_suffix):
        trace_events.append(make_noise_event(noise_pool, rng, all_tree_nodes, inject_prob))
        trace_metadata.append({"type": "noise"})

    return trace_events, trace_metadata


def generate_trace(paths, valid_path_idx, k, noise_pool, rng,
                   confuser_map, all_tree_nodes=None, neg_split_prob=0.6, max_retries=100):
    """
    Generate one trace by interleaving all paths with noise.

    Noise events may contain any tree node's key_value (~20% chance),
    making tree values appear throughout the trace. The model must track
    temporal ordering, not just keyword presence.

    For negative samples, verification ensures no accidental valid path
    is created by injected noise. Retries up to max_retries times.
    """
    num_paths = len(paths)
    path_depth = len(paths[0])

    neg_corruption_map = None
    cut_info = None
    path_events = None
    corruption_info = None
    used_tree_values = False

    if valid_path_idx is None:
        root = paths[0][0]

        # Unified retry loop: try tree-value corruption first, then noise-pool.
        # Always verify because noise injection can create accidental valid paths.
        for attempt in range(max_retries):
            cut_nodes = random_tree_cut(root, rng, split_prob=neg_split_prob)
            neg_corruption_map = assign_corruption_from_cut(cut_nodes, paths)
            cut_info = [n.name for n in cut_nodes]

            # First half: try tree-value corruption. Second half: noise-pool.
            use_tree = attempt < max_retries // 2
            cm = confuser_map if use_tree else {}

            path_events, corruption_info = build_corrupted_events(
                paths, None, neg_corruption_map,
                cm, noise_pool, rng,
            )

            trace_events, trace_metadata = _interleave(
                path_events, paths, k, noise_pool, rng, num_paths, path_depth, all_tree_nodes,
            )

            if not check_any_valid_path(trace_events, paths):
                used_tree_values = use_tree
                break
        else:
            # Last resort: no tree-value injection in noise either
            cut_nodes = random_tree_cut(root, rng, split_prob=neg_split_prob)
            neg_corruption_map = assign_corruption_from_cut(cut_nodes, paths)
            cut_info = [n.name for n in cut_nodes]

            path_events, corruption_info = build_corrupted_events(
                paths, None, neg_corruption_map,
                {}, noise_pool, rng,
            )

            # No tree values in noise — guaranteed clean
            trace_events, trace_metadata = _interleave(
                path_events, paths, k, noise_pool, rng, num_paths, path_depth, None,
            )
            used_tree_values = False
    else:
        # Positive sample
        path_events, corruption_info = build_corrupted_events(
            paths, valid_path_idx, None,
            confuser_map, noise_pool, rng,
        )
        trace_events, trace_metadata = _interleave(
            path_events, paths, k, noise_pool, rng, num_paths, path_depth, all_tree_nodes,
        )

    rendered = [render_step(i + 1, e, rng) for i, e in enumerate(trace_events)]

    return {
        "trace_id": None,
        "events": trace_events,
        "rendered_trace": rendered,
        "num_steps": len(trace_events),
        "label": "VALID" if valid_path_idx is not None else "INVALID",
        "valid_path_idx": valid_path_idx,
        "corruptions": [c for c in corruption_info if c is not None],
        "tree_cut": cut_info,  # None for positive samples
        "used_tree_values": used_tree_values if valid_path_idx is None else None,
        "metadata": trace_metadata,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate tree-LTL traces for temporal elasticity experiments"
    )
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 10, 50, 100])
    parser.add_argument("--b", type=int, default=3, help="Branching factor")
    parser.add_argument("--d", type=int, default=3, help="Tree depth")
    parser.add_argument("--extra_depth", type=int, default=0,
                        help="Extra chain nodes after each leaf before sink")
    parser.add_argument("--n_pos", type=int, default=20)
    parser.add_argument("--n_neg", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="tree_ltl_dataset.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Build tree
    root, sink, all_nodes = build_tree(b=args.b, d=args.d, extra_depth=args.extra_depth)
    noise_pool = assign_propositions(all_nodes, rng)

    # Paths
    paths = get_all_paths(root, sink)

    # Formulas
    ltl_formula = generate_ltl_symbolic(root, sink)
    nl_prompt = generate_nl_prompt(root, sink, paths)

    # Dataset
    dataset = {
        "tree": {
            "branching": args.b, "depth": args.d,
            "extra_depth": args.extra_depth,
            "total_depth": args.d + args.extra_depth,
            "num_paths": len(paths), "num_nodes": len(all_nodes),
        },
        "ltl_formula": ltl_formula,
        "nl_prompt": nl_prompt,
        "paths": [
            {"path_idx": i, "nodes": [n.name for n in p]}
            for i, p in enumerate(paths)
        ],
        "propositions": {
            node.name: {**node.prop.to_dict(), "key_attr": node.key_attr, "key_value": node.key_value}
            for node in all_nodes
        },
        "noise_pool_sizes": {k: len(v) for k, v in noise_pool.items()},
        "traces_by_k": {},
    }

    # Build confuser map for sibling-based corruption
    confuser_map = build_confuser_map(root, sink, all_nodes)

    for k in args.k_values:
        traces = []
        for i in range(args.n_pos):
            valid_idx = rng.randint(0, len(paths) - 1)
            trace = generate_trace(paths, valid_idx, k, noise_pool, rng,
                                   confuser_map, all_tree_nodes=all_nodes)
            trace["trace_id"] = f"k{k}_pos_{i}"
            traces.append(trace)
        for i in range(args.n_neg):
            trace = generate_trace(paths, None, k, noise_pool, rng,
                                   confuser_map, all_tree_nodes=all_nodes)
            trace["trace_id"] = f"k{k}_neg_{i}"
            traces.append(trace)
        dataset["traces_by_k"][str(k)] = traces

    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    # Summary
    print(f"Dataset: {args.output}")
    print(f"Tree: b={args.b}, d={args.d}, extra_depth={args.extra_depth} → {len(paths)} paths, {len(all_nodes)} nodes")
    print(f"k values: {args.k_values}")
    print(f"Traces per k: {args.n_pos} pos + {args.n_neg} neg")
    print(f"Noise pool: {', '.join(f'{k}={len(v)}' for k, v in noise_pool.items())}")
    print()
    print("=" * 70)
    print("NATURAL LANGUAGE PROMPT")
    print("=" * 70)
    print(nl_prompt)
    print()
    print("=" * 70)
    print("SYMBOLIC LTL")
    print("=" * 70)
    print(ltl_formula)
    print()

    # Example trace
    ex = dataset["traces_by_k"][str(args.k_values[0])][0]
    print("=" * 70)
    print(f"EXAMPLE TRACE: {ex['trace_id']} (label={ex['label']})")
    print("=" * 70)
    for line in ex["rendered_trace"][:20]:
        print(line)
    if ex["num_steps"] > 20:
        print(f"... ({ex['num_steps']} steps total)")


if __name__ == "__main__":
    main()