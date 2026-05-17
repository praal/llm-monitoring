#!/usr/bin/env python3
"""
Multi-Constraint Simple Formula Generator
==========================================

N constraints, each of the form F(A_i ∧ XF(B_i)).
  A_i = "the animal is X_i"
  B_i = "the color is Y_i"

Each constraint is independently satisfied or violated (50/50).
Single shared trace. LLM outputs a verdict per constraint.

Usage:
    python multi_simple_gen.py
    python multi_simple_gen.py --n_constraints 1 5 10 20 --n_samples 30
"""

import random
import json
import argparse
import os


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

SHAPES = [
    "oval", "square", "triangle", "circle", "diamond", "star",
    "hexagon", "pentagon", "crescent", "cross", "arrow", "heart",
    "trapezoid", "octagon", "spiral", "rectangle", "cube", "cone",
    "prism", "sphere", "ring", "wedge", "kite", "rhombus",
    "cylinder", "pyramid", "disc", "ellipse", "arch", "dome",
]

NUMBER_RANGE = (1, 100)


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
]


# ── Constraint spec ──────────────────────────────────────────────────────────

def generate_constraint_specs(n_constraints, rng):
    """Generate N constraint specs with unique A and B values."""
    a_values = rng.sample(ANIMALS, n_constraints)
    b_values = rng.sample(COLORS, n_constraints)

    specs = []
    for i in range(n_constraints):
        a_article = "an" if a_values[i][0].lower() in "aeiou" else "a"
        specs.append({
            "index": i + 1,
            "a_value": a_values[i],
            "b_value": b_values[i],
            "formula_nl": (f"Eventually the animal is {a_article} {a_values[i]}, "
                           f"and then eventually the color is {b_values[i]}."),
        })
    return specs


# ── Noise pools ──────────────────────────────────────────────────────────────

def build_noise_pools(specs):
    """Build noise pools excluding all constraint key values."""
    used_animals = {s["a_value"] for s in specs}
    used_colors = {s["b_value"] for s in specs}

    return {
        "animals": [a for a in ANIMALS if a not in used_animals],
        "colors": [c for c in COLORS if c not in used_colors],
        "shapes": list(SHAPES),
        "numbers": list(range(NUMBER_RANGE[0], NUMBER_RANGE[1] + 1)),
    }


# ── Event helpers ────────────────────────────────────────────────────────────

def make_noise_event(noise, rng):
    return {
        "animal": rng.choice(noise["animals"]),
        "color": rng.choice(noise["colors"]),
        "shape": rng.choice(noise["shapes"]),
        "number": rng.choice(noise["numbers"]),
    }


def render_step(step_num, event, rng):
    template = rng.choice(SENTENCE_TEMPLATES)
    return template.format(step=step_num, **event)


# ── Trace verification ──────────────────────────────────────────────────────

def verify_constraint(events, a_value, b_value):
    """Check F(A ∧ XF(B)): A appears, then B appears strictly after."""
    for i, e in enumerate(events):
        if e["animal"] == a_value:
            for j in range(i + 1, len(events)):
                if events[j]["color"] == b_value:
                    return True
    return False


# ── Trace generation ─────────────────────────────────────────────────────────

def generate_sample(specs, noise, trace_len, gap, rng, max_retries=200):
    """
    Generate one trace with N constraints, each independently 50/50 satisfied.

    Positive constraints: A placed at step_a, B placed at step_a + gap.
    Negative constraints: A's animal forbidden in entire trace. B may appear
    as distractor.
    """
    n_c = len(specs)

    # Fix labels
    labels = []
    for _ in range(n_c):
        labels.append("VALID" if rng.random() < 0.5 else "INVALID")

    for attempt in range(max_retries):
        # Plan signal positions
        signals = []

        # Constraint-specific forbidden values for negative constraints
        neg_forbidden_animals = set()

        for i, spec in enumerate(specs):
            if labels[i] == "VALID":
                step_a = rng.randint(5, trace_len - gap - 5)
                step_b = step_a + gap
                signals.append((step_a, "animal", spec["a_value"]))
                signals.append((step_b, "color", spec["b_value"]))
            else:
                neg_forbidden_animals.add(spec["a_value"])

        # Build events
        # Pre-assign signals to steps
        signal_map = {}  # step -> list of (attr, value)
        for step, attr, value in signals:
            signal_map.setdefault(step, []).append((attr, value))

        events = []
        for step in range(trace_len):
            event = make_noise_event(noise, rng)

            # Ensure no forbidden animals/colors leak in
            while event["animal"] in neg_forbidden_animals:
                event["animal"] = rng.choice(noise["animals"])

            # Apply signals
            if step in signal_map:
                for attr, value in signal_map[step]:
                    event[attr] = value

            # Distractor injection: for negative constraints, sometimes put
            # the B value in a random step (before A would have appeared)
            # or put A value in wrong attribute
            for i, spec in enumerate(specs):
                if labels[i] == "INVALID" and rng.random() < 0.02:
                    # Put B color as distractor
                    event["color"] = spec["b_value"]

            events.append(event)

        # Verify all constraints
        all_correct = True
        for i, spec in enumerate(specs):
            result = verify_constraint(events, spec["a_value"], spec["b_value"])
            expected = labels[i] == "VALID"
            if result != expected:
                all_correct = False
                break

        if all_correct:
            rendered = [render_step(j + 1, e, rng) for j, e in enumerate(events)]
            return {
                "labels": labels,
                "events": events,
                "rendered_trace": rendered,
                "num_steps": trace_len,
            }

    # Fallback: fix labels to match reality
    fixed_labels = list(labels)
    for i, spec in enumerate(specs):
        result = verify_constraint(events, spec["a_value"], spec["b_value"])
        fixed_labels[i] = "VALID" if result else "INVALID"

    rendered = [render_step(j + 1, e, rng) for j, e in enumerate(events)]
    return {
        "labels": fixed_labels,
        "events": events,
        "rendered_trace": rendered,
        "num_steps": trace_len,
        "fallback": True,
    }


# ── Prompt construction ──────────────────────────────────────────────────────

def build_prompt(specs, rendered_trace):
    lines = []
    lines.append("You are given a trace of observed events.")
    lines.append("")

    for spec in specs:
        lines.append(
            f"Constraint {spec['index']}: The trace is VALID for this "
            f"constraint if it satisfies: {spec['formula_nl']}")
        lines.append("")

    lines.append("For each constraint, determine whether the trace is VALID or INVALID.")
    lines.append("Respond with one line per constraint in the format:")
    for spec in specs:
        lines.append(f"Constraint {spec['index']}: VALID or INVALID")
    lines.append("")

    trace_text = "\n".join(rendered_trace)
    lines.append(f"--- TRACE ---\n{trace_text}\n--- END TRACE ---")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-constraint simple formula generator"
    )
    parser.add_argument("--n_constraints", nargs="+", type=int,
                        default=[1, 5, 10, 20])
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--trace_len", type=int, default=500,
                        help="Fixed trace length")
    parser.add_argument("--gap", type=int, default=50,
                        help="Gap between A_i and B_i for each constraint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="data/multi_simple_dataset.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    dataset = {
        "config": {
            "n_constraints_values": args.n_constraints,
            "n_samples": args.n_samples,
            "trace_len": args.trace_len,
            "gap": args.gap,
            "seed": args.seed,
        },
        "samples_by_n": {},
    }

    for n_c in args.n_constraints:
        trace_len = args.trace_len

        print(f"\nGenerating n_constraints={n_c} ({args.n_samples} samples, "
              f"trace_len={trace_len}, gap={args.gap})...")

        specs = generate_constraint_specs(n_c, rng)
        noise = build_noise_pools(specs)

        for s in specs[:3]:
            print(f"  C{s['index']}: {s['formula_nl']}")
        if len(specs) > 3:
            print(f"  ... and {len(specs) - 3} more")

        samples = []
        n_valid_total = 0
        n_invalid_total = 0
        n_fallback = 0

        for s_idx in range(args.n_samples):
            sample = generate_sample(specs, noise, trace_len, args.gap, rng)
            prompt = build_prompt(specs, sample["rendered_trace"])

            sample_entry = {
                "sample_id": f"nc{n_c}_s{s_idx}",
                "n_constraints": n_c,
                "labels": sample["labels"],
                "events": sample["events"],
                "rendered_trace": sample["rendered_trace"],
                "num_steps": sample["num_steps"],
                "prompt": prompt,
                "fallback": sample.get("fallback", False),
            }
            samples.append(sample_entry)

            n_valid_total += sum(1 for l in sample["labels"] if l == "VALID")
            n_invalid_total += sum(1 for l in sample["labels"] if l == "INVALID")
            if sample.get("fallback"):
                n_fallback += 1

            if (s_idx + 1) % 10 == 0:
                print(f"  {s_idx + 1}/{args.n_samples}")

        dataset["samples_by_n"][str(n_c)] = samples
        dataset[f"formulas_{n_c}"] = [s["formula_nl"] for s in specs]

        print(f"  Labels: {n_valid_total} VALID, {n_invalid_total} INVALID")
        print(f"  Fallbacks: {n_fallback}/{args.n_samples}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nDataset saved to {args.output}")


if __name__ == "__main__":
    main()