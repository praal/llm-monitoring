#!/usr/bin/env python3
"""
Multi-constraint tree-LTL generator for constraint scalability experiments.

Generates traces with N simultaneous tree-LTL constraints (b=2, d=2).
Each constraint has its own proposition pool. The LLM must output a verdict
for each constraint given a single shared trace.

Usage:
    python multi_constraint_gen.py --n_constraints 1 2 4 8
    python multi_constraint_gen.py --n_constraints 4 --n_samples 30 --trace_len 500
"""

import random
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional


# ── Attribute pools (large enough for 8 trees × 8 nodes = 64 nodes) ─────────

ANIMALS = [
    "deer", "hawk", "wolf", "bear", "fox", "owl", "lynx", "crane",
    "otter", "bison", "eagle", "moose", "raven", "salmon", "heron",
    "badger", "falcon", "turtle", "rabbit", "cobra", "parrot", "whale",
    "tiger", "panda", "koala", "dolphin", "jaguar", "pelican", "viper",
    "gorilla", "penguin", "leopard", "sparrow", "beetle", "mantis",
    "coyote", "osprey", "iguana", "ferret", "marmot", "toucan",
    "gazelle", "macaw", "lemur", "wombat", "jackal", "condor",
    "hamster", "lobster", "gecko", "starling", "ibis", "newt",
    "panther", "elk", "finch", "emu", "sloth", "yak", "moth",
    "stork", "hyena", "shrimp", "mole", "toad", "wasp", "wren",
    "snail", "clam", "crab", "dove", "frog", "goat", "mink", "puma",
    "squid", "llama", "dingo", "robin", "trout", "quail", "hare",
    "chimp", "bream", "perch", "swift", "grouse", "stag", "boar",
    "shrew", "stoat", "rook", "lark", "gull", "tern",
    "skink", "asp", "colt", "foal", "ram", "ewe", "hen", "drake",
    "mare", "bull", "calf", "lamb", "pike", "carp", "bass", "cod",
    "tuna", "sole", "ray", "eel", "worm", "tick", "flea", "gnat",
    "loon", "crow", "jay", "nene", "kiwi", "rhea", "pug", "dane",
    "akita", "corgi", "boxer", "husky", "beagle", "collie", "poodle",
    "spaniel", "setter", "mastiff", "greyhound", "whippet", "samoyed",
    "vizsla", "borzoi", "saluki", "briard", "kuvasz", "puli", "mudi",
    "komondor", "barbet", "basenji", "harrier", "lurcher", "otterhound",
    "pointer", "weimaraner", "brittany", "papillon", "maltese", "havanese",
    "bolognese", "lowchen", "coton", "pekingese", "affenpinscher", "schipperke",
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
    "prong", "barb", "hoop", "coil", "fin", "spoke", "plate",
    "bead", "stud", "slot", "tab", "pin", "rod", "cap", "gear",
    "strut", "beam", "rail", "brace", "rivet", "clamp", "valve",
    "nozzle", "funnel", "chute", "grate", "mesh", "frame", "panel",
    "bracket", "socket", "dowel", "peg", "latch", "hinge", "clasp",
    "buckle", "toggle", "lever", "crank", "pulley", "winch", "reel",
    "bobbin", "spool", "drum", "anvil", "axle", "cam",
    "torus", "vane", "wick", "stem", "bulb", "pod", "hull",
    "keel", "mast", "boom", "gaff", "cleat", "thimble", "grommet",
    "ferrule", "bushing", "gasket", "washer", "shim", "spacer", "liner",
    "baffle", "diffuser", "reflector", "deflector", "absorber", "damper",
    "isolator", "coupler", "adapter", "reducer", "elbow", "flange",
    "nipple", "union", "manifold", "header", "plenum", "trunk",
    "duct", "conduit", "raceway", "trough", "hopper", "bin",
    "canister", "capsule", "cartridge", "cassette", "magazine", "clip",
    "spindle", "mandrel", "arbor", "collet", "chuck", "jig",
    "fixture", "gauge", "caliper", "micrometer", "protractor", "compass",
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
    "bone", "fog", "snow", "pine", "rose", "ink", "corn",
    "coal", "lead", "tan", "bark", "grape", "mauve",
    "taupe", "ecru", "sepia", "umber", "sienna", "ochre",
    "cerise", "puce", "wisteria", "periwinkle", "chartreuse", "vermillion",
    "cerulean", "amaranth", "cinnabar", "gamboge", "smaragdine", "xanadu",
    "zinnwaldite", "feldgrau", "glaucous", "incarnadine", "isabelline", "mikado",
    "nacarat", "palatinate", "razzmatazz", "sinopia", "skobeloff", "zaffre",
    "alabaster", "bistre", "caput", "drab", "ebony", "fulvous",
    "grullo", "heliotrope", "jasper", "kobi", "liver", "malachite",
    "nyanza", "oxblood", "phlox", "quartz", "raisin", "saffron",
    "thistle", "ube", "verdigris", "wheat", "xanthic", "yale",
    "aureolin", "bole", "coquelicot", "damson", "eggplant", "firebrick",
    "garnet", "honeydew", "iceberg", "juniper", "kumquat", "lapis",
    "melon", "nutmeg", "orchid", "papaya", "raspberry", "sapphire",
    "tangerine", "ultramarine", "vanilla", "waterspout", "xanthous", "yarrow",
    "zinnia", "apricot", "burgundy", "cardinal", "denim", "emerald",
    "flamingo", "ginger", "hazel", "lemon", "mahogany", "nectarine",
    "pewter", "quince", "rosewood", "seashell", "topaz", "umber",
    "walnut", "almond", "blush", "chamois", "dijon", "espresso",
]

NUMBER_RANGE = (1, 500)


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
    key_attr: Optional[str] = None
    key_value: Optional[str] = None

    @property
    def is_leaf(self):
        return len(self.children) == 0


# ── Tree construction ────────────────────────────────────────────────────────

def build_tree(b, d, prefix=""):
    """Build a b-ary tree of depth d with a sink node."""
    counter = [0]
    all_nodes = []

    def make_node(depth):
        if depth == 0:
            node = TreeNode(f"{prefix}root")
        else:
            counter[0] += 1
            node = TreeNode(f"{prefix}n{counter[0]}")
        all_nodes.append(node)
        if depth < d:
            for _ in range(b):
                child = make_node(depth + 1)
                node.children.append(child)
        return node

    root = make_node(0)
    sink = TreeNode(f"{prefix}sink")
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


# ── Proposition assignment with shared pools ─────────────────────────────────

def assign_all_trees(all_trees_nodes, rng, b, d):
    """
    Assign propositions to ALL trees at once. Two-phase approach:
    1. Assign globally unique key_values across all nodes, respecting pool sizes
    2. Assign non-key attributes from pools that exclude all key_values

    Returns used_key_values set.
    """
    all_nodes_flat = [node for nodes in all_trees_nodes for node in nodes]
    n_total = len(all_nodes_flat)

    # Phase 1: assign key_attr and key_value for every node
    # Build a combined pool of (attr, value) pairs, shuffle, and draw
    n_animals = min(len(set(ANIMALS)), n_total)
    n_shapes = min(len(set(SHAPES)), n_total)
    n_numbers = min(NUMBER_RANGE[1] - NUMBER_RANGE[0] + 1, n_total)

    combined_pool = []
    for a in rng.sample(list(set(ANIMALS)), n_animals):
        combined_pool.append(("animal", a))
    for s in rng.sample(list(set(SHAPES)), n_shapes):
        combined_pool.append(("shape", s))
    for n in rng.sample(range(NUMBER_RANGE[0], NUMBER_RANGE[1] + 1), n_numbers):
        combined_pool.append(("number", str(n)))

    rng.shuffle(combined_pool)

    used_key_values = set()
    for i, node in enumerate(all_nodes_flat):
        attr, val = combined_pool[i]
        node.key_attr = attr
        node.key_value = val
        used_key_values.add(val)

    # Phase 2: assign non-key attributes from clean pools
    clean_animals = [a for a in ANIMALS if a not in used_key_values]
    clean_shapes = [s for s in SHAPES if s not in used_key_values]
    clean_colors = [c for c in COLORS if c not in used_key_values]
    clean_numbers = [n for n in range(NUMBER_RANGE[0], NUMBER_RANGE[1] + 1)
                     if str(n) not in used_key_values]

    for node in all_nodes_flat:
        animal = rng.choice(clean_animals)
        shape = rng.choice(clean_shapes)
        color = rng.choice(clean_colors)
        number = rng.choice(clean_numbers)

        if node.key_attr == "animal":
            animal = node.key_value
        elif node.key_attr == "shape":
            shape = node.key_value
        elif node.key_attr == "number":
            number = int(node.key_value)

        node.prop = Proposition(animal, shape, color, number)

    return used_key_values


# ── Formula generation ───────────────────────────────────────────────────────

def generate_formula_nl(root, sink):
    """Generate flat NL formula for one tree."""
    def make_desc(node):
        if node.key_attr == "number":
            return f"number {node.key_value}"
        else:
            article = "an" if node.key_value[0].lower() in "aeiou" else "a"
            return f"{article} {node.key_value}"

    sink_desc = make_desc(sink)

    def describe_node(node):
        desc = make_desc(node)
        if node.is_leaf:
            return f"observe {desc}, and then eventually observe {sink_desc}"
        child_descs = [describe_node(child) for child in node.children]
        if len(child_descs) == 1:
            return f"observe {desc}, and then eventually {child_descs[0]}"
        else:
            options = " OR ".join(f"({d})" for d in child_descs)
            return f"observe {desc}, and then eventually either: {options}"

    return f"Eventually {describe_node(root)}."


# ── Trace event helpers ──────────────────────────────────────────────────────

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


def make_event(prop):
    return prop.to_dict()


def make_noise_event(noise_pool, rng, all_tree_nodes=None, inject_prob=0.1):
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


def render_step(step_num, event, rng):
    template = rng.choice(SENTENCE_TEMPLATES)
    return template.format(
        step=step_num,
        color=event["color"],
        shape=event["shape"],
        number=event["number"],
        animal=event["animal"],
    )


# ── Path checking ────────────────────────────────────────────────────────────

def check_any_valid_path(trace_events, paths):
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


# ── Confuser map (cross-depth) ───────────────────────────────────────────────

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


def corrupt_event(event, rng, key_attr, confuser_values, noise_pool):
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


# ── Per-tree path event generation ───────────────────────────────────────────

def generate_tree_events(paths, is_satisfied, confuser_map, noise_pool, rng,
                         max_paths=6, max_retries=50):
    """
    Generate path events for one tree, including only a subset of paths.

    max_paths: how many paths to include in the trace (keeps trace compact).
    If is_satisfied: one valid path + (max_paths-1) corrupted distractors.
    If not: max_paths corrupted paths (via tree cut).

    Returns (selected_paths, path_events) where selected_paths is the
    list of path objects actually included.
    """
    path_depth = len(paths[0])
    n_paths = len(paths)

    if is_satisfied:
        valid_idx = rng.randint(0, n_paths - 1)
        # Pick other paths as distractors
        other_idxs = [i for i in range(n_paths) if i != valid_idx]
        distractor_idxs = rng.sample(other_idxs, min(max_paths - 1, len(other_idxs)))
        selected_idxs = [valid_idx] + distractor_idxs
        rng.shuffle(selected_idxs)

        selected_paths = [paths[i] for i in selected_idxs]
        path_events = []
        for idx in selected_idxs:
            path = paths[idx]
            events = [make_event(node.prop) for node in path]
            if idx != valid_idx:
                corrupt_idx = rng.randint(0, path_depth - 1)
                node = path[corrupt_idx]
                confusers = confuser_map.get(node.name, [])
                events[corrupt_idx] = corrupt_event(
                    events[corrupt_idx], rng, node.key_attr, confusers, noise_pool)
            path_events.append(events)
        return selected_paths, path_events
    else:
        # Negative: select max_paths paths, corrupt all via tree cut
        selected_idxs = rng.sample(range(n_paths), min(max_paths, n_paths))
        selected_paths = [paths[i] for i in selected_idxs]

        root = paths[0][0]
        cut_nodes = random_tree_cut(root, rng)
        corruption_map = assign_corruption_from_cut(cut_nodes, paths)

        path_events = []
        for idx in selected_idxs:
            path = paths[idx]
            events = [make_event(node.prop) for node in path]
            corrupt_idx = corruption_map[idx]
            node = path[corrupt_idx]
            confusers = confuser_map.get(node.name, [])
            events[corrupt_idx] = corrupt_event(
                events[corrupt_idx], rng, node.key_attr, confusers, noise_pool)
            path_events.append(events)

        return selected_paths, path_events


# ── Trace assembly ───────────────────────────────────────────────────────────

def assemble_trace(tree_path_events_list, trees_paths_list, target_len,
                   noise_pool, rng, all_tree_nodes, max_retries=100):
    """
    Assemble a single trace from multiple trees' path events + noise.

    Strategy:
    - Interleave events from all trees round by round
    - Each tree has its own path_depth rounds; pad shorter trees with nothing
    - Fill remaining steps with noise to reach target_len

    Returns (trace_events, trace_metadata)
    """
    # Determine max path depth across trees
    max_depth = max(len(paths[0]) for paths in trees_paths_list)

    # Collect all path events per round
    path_event_rounds = []
    for round_idx in range(max_depth):
        round_events = []
        for tree_idx, (path_events, paths) in enumerate(
                zip(tree_path_events_list, trees_paths_list)):
            path_depth = len(paths[0])
            if round_idx < path_depth:
                for path_idx in range(len(paths)):
                    round_events.append({
                        "event": path_events[path_idx][round_idx],
                        "tree_idx": tree_idx,
                        "path_idx": path_idx,
                        "round": round_idx,
                        "node": paths[path_idx][round_idx].name,
                    })
        rng.shuffle(round_events)
        path_event_rounds.append(round_events)

    # Total path events
    total_path_events = sum(len(r) for r in path_event_rounds)
    total_noise = max(target_len - total_path_events, 10)

    # Distribute noise: some at start, between rounds, and at end
    n_gaps = max_depth + 1  # before first, between each, after last
    noise_per_gap = total_noise // n_gaps
    extra = total_noise - noise_per_gap * n_gaps

    trace_events = []
    trace_metadata = []

    # Noise at start
    n_start = noise_per_gap + (1 if extra > 0 else 0)
    extra = max(extra - 1, 0)
    for _ in range(n_start):
        trace_events.append(make_noise_event(noise_pool, rng, all_tree_nodes))
        trace_metadata.append({"type": "noise"})

    for round_idx, round_events in enumerate(path_event_rounds):
        for re in round_events:
            trace_events.append(re["event"])
            trace_metadata.append({
                "type": "path_event",
                "tree_idx": re["tree_idx"],
                "path_idx": re["path_idx"],
                "round": re["round"],
                "node": re["node"],
            })

        # Noise between rounds
        n_gap = noise_per_gap + (1 if extra > 0 else 0)
        extra = max(extra - 1, 0)
        for _ in range(n_gap):
            trace_events.append(make_noise_event(noise_pool, rng, all_tree_nodes))
            trace_metadata.append({"type": "noise"})

    return trace_events, trace_metadata


# ── Prompt construction ──────────────────────────────────────────────────────

def build_prompt(formulas_nl, rendered_trace):
    """Build the multi-constraint prompt."""
    lines = []
    lines.append("You are given a trace of observed events.")
    lines.append("")

    for i, formula in enumerate(formulas_nl, 1):
        lines.append(f"Constraint {i}: The trace is VALID for this constraint if it satisfies: {formula}")
        lines.append("")

    lines.append("For each constraint, determine whether the trace is VALID or INVALID.")
    lines.append("Respond with one line per constraint in the format:")
    for i in range(1, len(formulas_nl) + 1):
        lines.append(f"Constraint {i}: VALID or INVALID")
    lines.append("")

    trace_text = "\n".join(rendered_trace)
    lines.append(f"--- TRACE ---\n{trace_text}\n--- END TRACE ---")

    return "\n".join(lines)


# ── Sample generation with verification ──────────────────────────────────────

def generate_sample(trees_data, noise_pool, rng, all_tree_nodes, target_len,
                    max_retries=200):
    """
    Generate one sample with N constraints.
    Each constraint is independently 50/50 satisfied/violated.
    Labels are fixed upfront; only the trace assembly is retried on
    verification failure.
    """
    n_trees = len(trees_data)

    # Fix labels once
    labels = []
    for td in trees_data:
        is_satisfied = rng.random() < 0.5
        labels.append("VALID" if is_satisfied else "INVALID")

    for attempt in range(max_retries):
        tree_selected_paths = []
        tree_path_events = []
        for i, td in enumerate(trees_data):
            is_satisfied = labels[i] == "VALID"
            selected_paths, path_events = generate_tree_events(
                td["paths"], is_satisfied,
                td["confuser_map"], noise_pool, rng)
            tree_selected_paths.append(selected_paths)
            tree_path_events.append(path_events)

        # Assemble full trace using only selected paths
        trace_events, trace_metadata = assemble_trace(
            tree_path_events, tree_selected_paths, target_len,
            noise_pool, rng, all_tree_nodes)

        # Verify each constraint against ALL paths (not just selected)
        all_correct = True
        for i, td in enumerate(trees_data):
            result = check_any_valid_path(trace_events, td["paths"])
            expected = labels[i] == "VALID"
            if result != expected:
                all_correct = False
                break

        if all_correct:
            rendered = [render_step(j + 1, e, rng)
                        for j, e in enumerate(trace_events)]
            return {
                "labels": labels,
                "events": trace_events,
                "rendered_trace": rendered,
                "num_steps": len(trace_events),
                "metadata": trace_metadata,
            }

    # Last resort: fix labels to match reality
    fixed_labels = list(labels)
    tree_selected_paths = []
    tree_path_events = []
    for i, td in enumerate(trees_data):
        is_satisfied = fixed_labels[i] == "VALID"
        selected_paths, path_events = generate_tree_events(
            td["paths"], is_satisfied,
            td["confuser_map"], noise_pool, rng)
        tree_selected_paths.append(selected_paths)
        tree_path_events.append(path_events)

    trace_events, trace_metadata = assemble_trace(
        tree_path_events, tree_selected_paths, target_len,
        noise_pool, rng, all_tree_nodes)

    # Check which constraints don't match and flip them
    for i, td in enumerate(trees_data):
        result = check_any_valid_path(trace_events, td["paths"])
        expected = fixed_labels[i] == "VALID"
        if result != expected:
            fixed_labels[i] = "VALID" if result else "INVALID"

    rendered = [render_step(j + 1, e, rng)
                for j, e in enumerate(trace_events)]
    return {
        "labels": fixed_labels,
        "events": trace_events,
        "rendered_trace": rendered,
        "num_steps": len(trace_events),
        "metadata": trace_metadata,
        "fallback": True,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-constraint tree-LTL dataset"
    )
    parser.add_argument("--n_constraints", nargs="+", type=int,
                        default=[1, 5, 10, 20])
    parser.add_argument("--b", type=int, default=2)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=30,
                        help="Samples per constraint count")
    parser.add_argument("--trace_len", type=int, default=1000,
                        help="Target trace length in steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="data/multi_constraint_dataset.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    nodes_per_tree = (args.b ** (args.d + 1) - 1) // (args.b - 1) + 1
    max_n = max(args.n_constraints)
    total_nodes = max_n * nodes_per_tree

    print(f"Tree config: b={args.b}, d={args.d} → {nodes_per_tree} nodes/tree")
    print(f"Max constraints: {max_n} → {total_nodes} total nodes needed")

    # Build all trees with shared pools and globally unique key_values
    all_trees_nodes = []
    all_trees = []

    for i in range(max_n):
        root, sink, all_nodes = build_tree(args.b, args.d, prefix=f"t{i}_")
        all_trees_nodes.append(all_nodes)
        all_trees.append({
            "tree_idx": i,
            "root": root,
            "sink": sink,
            "all_nodes": all_nodes,
        })

    # Assign propositions to all trees at once
    used_key_values = assign_all_trees(all_trees_nodes, rng, args.b, args.d)
    print(f"Unique key_values: {len(used_key_values)}")

    # Now build paths, confuser maps, formulas
    all_used = {"animals": set(), "shapes": set(), "colors": set(), "numbers": set()}
    for td in all_trees:
        td["paths"] = get_all_paths(td["root"], td["sink"])
        td["confuser_map"] = build_confuser_map(td["root"], td["sink"], td["all_nodes"])
        td["formula_nl"] = generate_formula_nl(td["root"], td["sink"])

        for node in td["all_nodes"]:
            all_used["animals"].add(node.prop.animal)
            all_used["shapes"].add(node.prop.shape)
            all_used["colors"].add(node.prop.color)
            all_used["numbers"].add(node.prop.number)

    # Build global noise pool — exclude key_values only
    # (non-key attributes can safely appear in noise since the formula
    #  only checks key_values)
    noise_pool = {
        "animals": [a for a in ANIMALS if a not in used_key_values],
        "shapes":  [s for s in SHAPES if s not in used_key_values],
        "colors":  [c for c in COLORS if c not in used_key_values],
        "numbers": [n for n in range(NUMBER_RANGE[0], NUMBER_RANGE[1] + 1)
                     if str(n) not in used_key_values],
    }
    print(f"Noise pool: {', '.join(f'{k}={len(v)}' for k, v in noise_pool.items())}")

    # All tree nodes for injection
    all_tree_nodes = [node for td in all_trees for node in td["all_nodes"]]

    # Generate dataset
    dataset = {
        "config": {
            "b": args.b, "d": args.d,
            "nodes_per_tree": nodes_per_tree,
            "n_samples": args.n_samples,
            "trace_len": args.trace_len,
            "n_constraints_values": args.n_constraints,
            "seed": args.seed,
        },
        "formulas": [td["formula_nl"] for td in all_trees],
        "tree_paths": [
            [[node.key_value for node in path] for path in td["paths"]]
            for td in all_trees
        ],
        "samples_by_n": {},
    }

    for n_c in args.n_constraints:
        trees_data = all_trees[:n_c]
        formulas_nl = [td["formula_nl"] for td in trees_data]

        # Scale trace length so noise-to-path ratio is consistent across N.
        # Reference: N=5 at trace_len=1000.
        max_paths_per_tree = 6
        path_depth = args.d + 2  # root + d levels + sink
        path_events_this = n_c * max_paths_per_tree * path_depth
        ref_n = 5
        ref_path_events = ref_n * max_paths_per_tree * path_depth
        ref_trace_len = args.trace_len
        if ref_path_events < ref_trace_len:
            noise_ratio = (ref_trace_len - ref_path_events) / ref_path_events
        else:
            noise_ratio = 0.5
        effective_trace_len = min(args.trace_len,
                                  int(path_events_this * (1 + noise_ratio)))
        effective_trace_len = max(effective_trace_len, path_events_this + 20)

        print(f"\nGenerating n_constraints={n_c} ({args.n_samples} samples, "
              f"trace_len={effective_trace_len})...")

        samples = []
        for s in range(args.n_samples):
            sample = generate_sample(
                trees_data, noise_pool, rng, None, effective_trace_len)

            prompt = build_prompt(formulas_nl, sample["rendered_trace"])

            samples.append({
                "sample_id": f"nc{n_c}_s{s}",
                "n_constraints": n_c,
                "labels": sample["labels"],
                "events": sample["events"],
                "rendered_trace": sample["rendered_trace"],
                "num_steps": sample["num_steps"],
                "prompt": prompt,
                "fallback": sample.get("fallback", False),
            })

            if (s + 1) % 10 == 0:
                print(f"  {s + 1}/{args.n_samples}")

        dataset["samples_by_n"][str(n_c)] = samples

        # Stats
        n_valid = sum(l == "VALID" for s in samples for l in s["labels"])
        n_invalid = sum(l == "INVALID" for s in samples for l in s["labels"])
        n_fallback = sum(1 for s in samples if s["fallback"])
        print(f"  Labels: {n_valid} VALID, {n_invalid} INVALID")
        print(f"  Fallbacks: {n_fallback}/{args.n_samples}")

    # Save
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nDataset saved to {args.output}")


if __name__ == "__main__":
    main()