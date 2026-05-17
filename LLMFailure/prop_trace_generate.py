#!/usr/bin/env python3
"""
Proposition Scaling Experiment
===============================

Tests whether LLM accuracy degrades as observation complexity
(number of entities per step) increases, with a fixed simple formula.

Formula: F(A ∧ XF(B))
  A = "the Kth animal mentioned is X"
  B = "the Mth color mentioned is Y"

Vary N (entities per step): 1, 3, 5, 10, 20

Usage:
    python prop_scaling_gen.py
    python prop_scaling_gen.py --n_entities 1 3 5 10 20 --n_samples 30
"""

import random
import json
import argparse
import os
from dataclasses import dataclass


# ── Attribute pools ──────────────────────────────────────────────────────────

ANIMALS = [
    "deer", "hawk", "wolf", "bear", "fox", "owl", "lynx", "crane",
    "otter", "bison", "eagle", "moose", "raven", "salmon", "heron",
    "badger", "falcon", "turtle", "rabbit", "cobra", "parrot", "whale",
    "tiger", "panda", "koala", "dolphin", "jaguar", "pelican", "viper",
    "gorilla", "penguin", "leopard", "sparrow", "beetle", "mantis",
]

COLORS = [
    "red", "blue", "green", "yellow", "purple", "orange",
    "white", "black", "silver", "gold", "pink", "brown",
    "teal", "maroon", "ivory", "crimson", "coral", "azure",
    "amber", "jade", "scarlet", "indigo", "bronze", "peach",
    "violet", "khaki", "olive", "beige", "magenta", "cyan",
]

SHAPES = [
    "oval", "square", "triangle", "circle", "diamond", "star",
    "hexagon", "pentagon", "crescent", "cross", "arrow", "heart",
    "trapezoid", "octagon", "spiral", "rectangle", "cube", "cone",
    "prism", "sphere", "ring", "wedge", "kite", "rhombus",
    "cylinder", "pyramid", "disc", "ellipse", "arch", "dome",
]

NUMBER_RANGE = (1, 100)


# ── Ordinal helpers ──────────────────────────────────────────────────────────

def ordinal(n):
    """Return ordinal string: 1st, 2nd, 3rd, 4th, ..."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ── Entity rendering ─────────────────────────────────────────────────────────

ENTITY_CONNECTORS = [
    "alongside", "with", "near", "beside", "next to",
]


def render_step(step_num, entities, rng):
    """
    Render a step with N entities using explicit labels.

    Each entity is (animal, color, shape, number).
    Output like:
      Step 5: Entity 1: a red oval (number 12) alongside a deer.
      Entity 2: a blue triangle (number 7) with a wolf.
    For N=1, no label:
      Step 5: Observed a red oval (number 12) alongside a deer.
    """
    if len(entities) == 1:
        animal, color, shape, number = entities[0]
        conn = rng.choice(ENTITY_CONNECTORS)
        animal_article = "an" if animal[0].lower() in "aeiou" else "a"
        color_article = "an" if color[0].lower() in "aeiou" else "a"
        return (f"Step {step_num}: Observed {color_article} {color} {shape} "
                f"(number {number}) {conn} {animal_article} {animal}.")

    parts = []
    for i, (animal, color, shape, number) in enumerate(entities):
        conn = rng.choice(ENTITY_CONNECTORS)
        animal_article = "an" if animal[0].lower() in "aeiou" else "a"
        color_article = "an" if color[0].lower() in "aeiou" else "a"
        parts.append(
            f"Entity {i+1}: {color_article} {color} {shape} "
            f"(number {number}) {conn} {animal_article} {animal}"
        )

    return f"Step {step_num}: " + ". ".join(parts) + "."


# ── Formula generation ───────────────────────────────────────────────────────

@dataclass
class FormulaSpec:
    """Specification for F(A ∧ XF(B))."""
    # A: "the Kth animal mentioned is X"
    a_attr: str       # "animal"
    a_slot: int       # 1-indexed slot
    a_value: str      # e.g., "wolf"
    # B: "the Mth color mentioned is Y"
    b_attr: str       # "color"
    b_slot: int       # 1-indexed slot
    b_value: str      # e.g., "red"


def generate_formula_spec(n_entities, rng):
    """Generate a random formula spec for given number of entities."""
    a_slot = rng.randint(1, n_entities)
    b_slot = rng.randint(1, n_entities)
    a_value = rng.choice(ANIMALS)
    b_value = rng.choice(COLORS)

    return FormulaSpec(
        a_attr="animal", a_slot=a_slot, a_value=a_value,
        b_attr="color", b_slot=b_slot, b_value=b_value,
    )


def formula_to_nl(spec, n_entities):
    """Convert formula spec to natural language."""
    a_article = "an" if spec.a_value[0].lower() in "aeiou" else "a"
    if n_entities == 1:
        a_desc = f"the animal is {a_article} {spec.a_value}"
        b_desc = f"the color is {spec.b_value}"
    else:
        a_desc = f"Entity {spec.a_slot}'s animal is {a_article} {spec.a_value}"
        b_desc = f"Entity {spec.b_slot}'s color is {spec.b_value}"

    return (f"Eventually {a_desc}, "
            f"and then eventually {b_desc}.")


# ── Trace generation ─────────────────────────────────────────────────────────

def generate_entities(n_entities, rng, overrides=None, forbidden=None):
    """
    Generate N random entities. Each entity is (animal, color, shape, number).
    overrides: dict mapping (attr, slot_0indexed) -> value to force.
    forbidden: dict mapping (attr, slot_0indexed) -> set of values to avoid.
    """
    entities = []
    for i in range(n_entities):
        animal = rng.choice(ANIMALS)
        color = rng.choice(COLORS)
        shape = rng.choice(SHAPES)
        number = rng.randint(NUMBER_RANGE[0], NUMBER_RANGE[1])

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

        if overrides:
            if ("animal", i) in overrides:
                animal = overrides[("animal", i)]
            if ("color", i) in overrides:
                color = overrides[("color", i)]
            if ("shape", i) in overrides:
                shape = overrides[("shape", i)]
            if ("number", i) in overrides:
                number = overrides[("number", i)]

        entities.append((animal, color, shape, number))
    return entities


def inject_distractor(entities, spec, rng):
    """
    Inject the formula's key values into WRONG slots to create distractors.
    Returns modified entity list.
    """
    entities = list(entities)
    n = len(entities)
    if n <= 1:
        return entities

    # With some probability, put A's value in a wrong animal slot
    if rng.random() < 0.3:
        wrong_slots = [i for i in range(n) if i != spec.a_slot - 1]
        if wrong_slots:
            slot = rng.choice(wrong_slots)
            animal, color, shape, number = entities[slot]
            entities[slot] = (spec.a_value, color, shape, number)

    # With some probability, put B's value in a wrong color slot
    if rng.random() < 0.3:
        wrong_slots = [i for i in range(n) if i != spec.b_slot - 1]
        if wrong_slots:
            slot = rng.choice(wrong_slots)
            animal, color, shape, number = entities[slot]
            entities[slot] = (animal, spec.b_value, shape, number)

    return entities


def generate_sample(spec, n_entities, trace_len, rng):
    """
    Generate one sample (trace + label).

    Positive: A at correct slot at step_a, B at correct slot at step_b > step_a.
              Other steps may also match (fine for positives).
    Negative: Either:
      - A-corrupted: A's value NEVER appears at A's slot. Distractors put A's
        value in wrong slots.
      - B-corrupted: A appears normally at step_a, but B's value NEVER appears
        at B's slot after step_a. Distractors put B's value in wrong slots.
    """
    is_positive = rng.random() < 0.5

    step_a = rng.randint(10, trace_len // 2)
    step_b = rng.randint(step_a + 1, min(step_a + trace_len // 2, trace_len - 10))

    if is_positive:
        # Positive: place A at step_a, B at step_b
        steps = []
        for step in range(trace_len):
            overrides = {}
            if step == step_a:
                overrides = {("animal", spec.a_slot - 1): spec.a_value}
            elif step == step_b:
                overrides = {("color", spec.b_slot - 1): spec.b_value}

            entities = generate_entities(n_entities, rng, overrides=overrides)

            # Inject distractors on some noise steps
            if step not in (step_a, step_b) and rng.random() < 0.15:
                entities = inject_distractor(entities, spec, rng)

            steps.append(entities)
    else:
        # Negative: corrupt either A or B
        corrupt_target = rng.choice(["A", "B"])

        steps = []
        for step in range(trace_len):
            overrides = {}
            forbidden = {}

            if corrupt_target == "A":
                # A's value never at A's slot. B can appear freely.
                forbidden[("animal", spec.a_slot - 1)] = {spec.a_value}
                if step == step_b:
                    overrides[("color", spec.b_slot - 1)] = spec.b_value

                # Put A's value in wrong slots as distractor
                if rng.random() < 0.2 and n_entities > 1:
                    wrong_slot = rng.choice(
                        [i for i in range(n_entities) if i != spec.a_slot - 1])
                    overrides[("animal", wrong_slot)] = spec.a_value

            else:  # corrupt B
                # A appears at step_a. B's value never at B's slot in entire trace.
                if step == step_a:
                    overrides[("animal", spec.a_slot - 1)] = spec.a_value
                forbidden[("color", spec.b_slot - 1)] = {spec.b_value}

                # Put B's value in wrong slots as distractor
                if rng.random() < 0.2 and n_entities > 1:
                    wrong_slot = rng.choice(
                        [i for i in range(n_entities) if i != spec.b_slot - 1])
                    overrides[("color", wrong_slot)] = spec.b_value

            entities = generate_entities(n_entities, rng,
                                         overrides=overrides,
                                         forbidden=forbidden)
            steps.append(entities)

    # Verify
    actually_valid = verify_trace(steps, spec)
    label = "VALID" if actually_valid else "INVALID"

    # Render
    rendered = [render_step(i + 1, ents, rng) for i, ents in enumerate(steps)]

    return {
        "label": label,
        "rendered_trace": rendered,
        "num_steps": trace_len,
        "n_entities": n_entities,
    }


# ── Verification ─────────────────────────────────────────────────────────────

def verify_trace(steps, spec):
    """
    Verify F(A ∧ XF(B)):
    There exists step_a where the a_slot-th animal is a_value,
    and some step_b > step_a where the b_slot-th color is b_value.
    """
    for i, entities in enumerate(steps):
        # Check A at this step
        if len(entities) >= spec.a_slot:
            animal = entities[spec.a_slot - 1][0]  # animal is index 0 in tuple
            if animal == spec.a_value:
                # A matches, now look for B in remaining steps
                for j in range(i + 1, len(steps)):
                    if len(steps[j]) >= spec.b_slot:
                        color = steps[j][spec.b_slot - 1][1]  # color is index 1
                        if color == spec.b_value:
                            return True
    return False


# ── Prompt construction ──────────────────────────────────────────────────────

def build_prompt(formula_nl, rendered_trace, n_entities):
    lines = []
    lines.append("You are given a trace of observed events.")
    if n_entities > 1:
        lines.append(
            f"Each step describes {n_entities} labeled entities (Entity 1, Entity 2, ..., Entity {n_entities}). "
            f"Each entity has an animal, a color, a shape, and a number."
        )
    lines.append("")
    lines.append(
        f"The trace is VALID if it satisfies the following constraint:"
    )
    lines.append("")
    lines.append(formula_nl)
    lines.append("")
    lines.append("The trace is INVALID otherwise.")
    lines.append("")
    lines.append(
        "Determine whether the following trace is VALID or INVALID. "
        "Respond with VALID or INVALID."
    )
    lines.append("")

    trace_text = "\n".join(rendered_trace)
    lines.append(f"--- TRACE ---\n{trace_text}\n--- END TRACE ---")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Proposition scaling experiment"
    )
    parser.add_argument("--n_entities", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16])
    parser.add_argument("--n_samples", type=int, default=30,
                        help="Samples per seed per n_entities")
    parser.add_argument("--trace_len", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46],
                        help="Multiple seeds for different formula specs")
    parser.add_argument("--output", type=str,
                        default="data/prop_scaling_dataset.json")
    args = parser.parse_args()

    dataset = {
        "config": {
            "n_entities_values": args.n_entities,
            "n_samples_per_seed": args.n_samples,
            "trace_len": args.trace_len,
            "seeds": args.seeds,
        },
        "samples_by_n": {},
    }

    for n_ent in args.n_entities:
        total_samples = []
        total_valid = 0
        total_invalid = 0

        print(f"\nGenerating n_entities={n_ent} "
              f"({args.n_samples} samples × {len(args.seeds)} seeds)...")

        for seed in args.seeds:
            rng = random.Random(seed)
            spec = generate_formula_spec(n_ent, rng)
            formula_nl = formula_to_nl(spec, n_ent)

            print(f"  seed={seed}: {formula_nl}")

            for s in range(args.n_samples):
                sample = generate_sample(spec, n_ent, args.trace_len, rng)
                prompt = build_prompt(formula_nl, sample["rendered_trace"], n_ent)

                sample["sample_id"] = f"ne{n_ent}_seed{seed}_s{s}"
                sample["seed"] = seed
                sample["formula_nl"] = formula_nl
                sample["formula_spec"] = {
                    "a_attr": spec.a_attr, "a_slot": spec.a_slot,
                    "a_value": spec.a_value,
                    "b_attr": spec.b_attr, "b_slot": spec.b_slot,
                    "b_value": spec.b_value,
                }
                sample["prompt"] = prompt

                if sample["label"] == "VALID":
                    total_valid += 1
                else:
                    total_invalid += 1

                total_samples.append(sample)

        dataset["samples_by_n"][str(n_ent)] = total_samples
        print(f"  Total: {total_valid} VALID, {total_invalid} INVALID "
              f"({len(total_samples)} samples)")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nDataset saved to {args.output}")


if __name__ == "__main__":
    main()