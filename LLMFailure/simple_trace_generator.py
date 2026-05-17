"""
Temporal Elasticity (simple formula) — Trace Generator
==============================================

Generates structured-event traces for evaluating whether LLMs can detect
satisfaction/violation of □(A → ◇B) as the temporal gap between A and B grows.

Each timestep is a structured record: {color, shape, number, animal}
  - Proposition A: color == "red"
  - Proposition B: animal == "cat"

Independent variables:
  - gap g ∈ {1, 2, 5, 10, 20, 50, 100, 200, 500}
  - position of A: early, middle, late
  - trace type: satisfied, violated, distractor

Output: JSON file with all generated traces + ground truth metadata.
"""
from __future__ import annotations

import json
import random
import os
from dataclasses import dataclass, field, asdict
from typing import Literal


# ─────────────────────────────────────────────
# Event vocabulary (chosen to be unambiguous and LLM-friendly)
# ─────────────────────────────────────────────

# Proposition A: color == "red"
# Proposition B: animal == "cat"

COLORS = ["blue", "green", "yellow", "purple", "orange", "brown", "pink", "gray"]

SHAPES = ["triangle", "square", "circle", "pentagon", "hexagon", "star", "diamond", "oval"]

ANIMALS = ["dog", "horse", "eagle", "fish", "frog", "bear", "wolf", "deer"]

NUMBER_RANGE = (1, 99)


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class Event:
    color: str
    shape: str
    number: int
    animal: str

    @property
    def prop_A(self) -> bool:
        """Proposition A: color is red."""
        return self.color == "red"

    @property
    def prop_B(self) -> bool:
        """Proposition B: animal is cat."""
        return self.animal == "cat"

    def to_str(self) -> str:
        """Format event as a natural English sentence."""
        templates = [
            "{A_color} {color} {shape} labeled {number} with {a_animal} {animal}.",
            "There is {a_color} {color} {shape} marked {number}, accompanied by {a_animal} {animal}.",
            "Observed {a_color} {color} {shape} (number {number}) alongside {a_animal} {animal}.",
            "{A_animal} {animal} is next to {a_color} {color} {shape} bearing the number {number}.",
            "Item {number}: {a_color} {color} {shape} paired with {a_animal} {animal}.",
        ]
        # Deterministic template selection based on event content
        # (avoid hash() which is randomized across Python sessions)
        idx = (ord(self.color[0]) + ord(self.animal[0]) + self.number) % len(templates)

        def _article(word: str) -> str:
            return "an" if word[0].lower() in "aeiou" else "a"

        return templates[idx].format(
            color=self.color, shape=self.shape,
            number=self.number, animal=self.animal,
            a_color=_article(self.color),
            A_color=_article(self.color).capitalize(),
            a_animal=_article(self.animal),
            A_animal=_article(self.animal).capitalize(),
        )


@dataclass
class Trace:
    trace_id: str
    gap: int
    a_position_label: str  # "early", "middle", "late"
    a_position: int        # actual timestep index (0-based)
    trace_type: str        # "satisfied", "violated", "distractor"
    trace_length: int
    b_position: int | None  # timestep where B occurs (None if violated)
    events: list[Event] = field(default_factory=list)
    ground_truth_verdict: str = ""  # "satisfied" or "violated"
    # For distractor traces: position of the distractor B (before A)
    distractor_b_position: int | None = None

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "gap": self.gap,
            "a_position_label": self.a_position_label,
            "a_position": self.a_position,
            "trace_type": self.trace_type,
            "trace_length": self.trace_length,
            "b_position": self.b_position,
            "distractor_b_position": self.distractor_b_position,
            "ground_truth_verdict": self.ground_truth_verdict,
            "events": [
                {
                    "timestep": t,
                    "event": asdict(e),
                    "prop_A": e.prop_A,
                    "prop_B": e.prop_B,
                }
                for t, e in enumerate(self.events)
            ],
        }


# ─────────────────────────────────────────────
# Event generators
# ─────────────────────────────────────────────

def random_filler_event(rng: random.Random) -> Event:
    """Generate a filler event where A is false and B is false."""
    return Event(
        color=rng.choice(COLORS),        # excludes "red"
        shape=rng.choice(SHAPES),
        number=rng.randint(*NUMBER_RANGE),
        animal=rng.choice(ANIMALS),      # excludes "cat"
    )


def event_with_A(rng: random.Random) -> Event:
    """Generate an event where A is true (color=red), B is false."""
    return Event(
        color="red",
        shape=rng.choice(SHAPES),
        number=rng.randint(*NUMBER_RANGE),
        animal=rng.choice(ANIMALS),      # excludes "cat"
    )


def event_with_B(rng: random.Random) -> Event:
    """Generate an event where B is true (animal=cat), A is false."""
    return Event(
        color=rng.choice(COLORS),        # excludes "red"
        shape=rng.choice(SHAPES),
        number=rng.randint(*NUMBER_RANGE),
        animal="cat",
    )


# ─────────────────────────────────────────────
# Trace generation
# ─────────────────────────────────────────────

# Experiment parameters
# Maps gap -> number of traces per cell (per gap × position × type).
# Lower counts for larger gaps to manage costs.
GAP_SAMPLES = {
    1: 20,
    10: 20,
    50: 20,
    100: 20,
    300: 20,
    500: 20,
    700: 20, 
    1000: 20, 
    5000: 5,
}
GAPS = list(GAP_SAMPLES.keys())
TRACES_PER_CELL = 50  # kept for backward compat (default)
PADDING = 20  # extra timesteps after the last interesting event
MIN_TRACE_LENGTH = 100  # ensures traces aren't too short for small gaps


# A position ranges (0-indexed). Each trace picks a random position
# within the range. Trace length is based on the range max so all
# traces in a position group have identical length.
A_POSITION_RANGES = {
    "early": (0, 49),     # steps 1–50
    "late": (100, 199),   # steps 101–200
}


def compute_a_position(rng: random.Random, position_label: str) -> int:
    """Pick a random A position within the range for this label."""
    lo, hi = A_POSITION_RANGES[position_label]
    return rng.randint(lo, hi)


def compute_trace_length(gap: int, position_label: str) -> int:
    """
    Compute trace length: max_a_position + gap + PADDING.

    Uses the MAX of the position range so all traces in a
    (gap, position) cell have the same length regardless of
    where A actually lands.
    """
    max_a_pos = A_POSITION_RANGES[position_label][1]
    return max(MIN_TRACE_LENGTH, max_a_pos + gap + PADDING)


def generate_trace(
    gap: int,
    position_label: str,
    trace_type: Literal["satisfied", "violated", "distractor"],
    trace_idx: int,
    rng: random.Random,
) -> Trace:
    """Generate a single trace."""

    trace_length = compute_trace_length(gap, position_label)
    a_pos = compute_a_position(rng, position_label)

    # Sanity checks
    assert 0 <= a_pos < trace_length, f"a_pos={a_pos} out of range for N={trace_length}"

    b_pos = None
    distractor_b_pos = None

    if trace_type == "satisfied":
        b_pos = a_pos + gap
        assert b_pos < trace_length, f"b_pos={b_pos} >= N={trace_length}"
    elif trace_type == "violated":
        b_pos = None  # B never occurs after A
    elif trace_type == "distractor":
        # B occurs before A, but never after.
        # Place distractor B at a random position in [max(0, a_pos-gap-5), a_pos-1]
        # so the temporal distance between distractor-B and A is comparable to the
        # gap parameter — this makes the distractor more challenging.
        if a_pos >= 1:
            earliest = max(0, a_pos - gap - 5)
            distractor_b_pos = rng.randint(earliest, a_pos - 1)
        else:
            distractor_b_pos = None  # Can't place distractor before position 0
        b_pos = None  # No B after A

    # Build events
    events = []
    for t in range(trace_length):
        if t == a_pos:
            events.append(event_with_A(rng))
        elif t == b_pos:
            events.append(event_with_B(rng))
        elif t == distractor_b_pos:
            events.append(event_with_B(rng))
        else:
            events.append(random_filler_event(rng))

    # Ground truth: for □(A → ◇B), the constraint is satisfied iff
    # every occurrence of A is eventually followed by B.
    # We have exactly one A. So:
    #   - satisfied: B occurs after A
    #   - violated: B never occurs after A (including distractor where B is only before A)
    ground_truth = "satisfied" if trace_type == "satisfied" else "violated"

    trace_id = f"g{gap}_{position_label}_{trace_type}_{trace_idx:03d}"

    return Trace(
        trace_id=trace_id,
        gap=gap,
        a_position_label=position_label,
        a_position=a_pos,
        trace_type=trace_type,
        trace_length=trace_length,
        b_position=b_pos,
        distractor_b_position=distractor_b_pos,
        events=events,
        ground_truth_verdict=ground_truth,
    )


def generate_all_traces(seed: int = 42) -> list[Trace]:
    """Generate the full dataset of traces.
    
    Uses GAP_SAMPLES dict for per-gap sample counts. New gaps are appended 
    at the end, so traces for existing gaps are generated with the same 
    RNG state and produce identical trace IDs — existing results are reusable.
    """
    rng = random.Random(seed)
    all_traces = []

    position_labels = ["early", "late"]
    trace_types = ["satisfied", "violated", "distractor"]

    for gap in GAPS:
        n_samples = GAP_SAMPLES[gap]
        for pos_label in position_labels:
            for ttype in trace_types:
                for idx in range(n_samples):
                    trace = generate_trace(gap, pos_label, ttype, idx, rng)
                    all_traces.append(trace)

    return all_traces


# ─────────────────────────────────────────────
# Dataset statistics and validation
# ─────────────────────────────────────────────

def print_dataset_stats(traces: list[Trace]) -> None:
    """Print summary statistics of the generated dataset."""
    print(f"Total traces: {len(traces)}")
    print(f"  Gaps: {sorted(set(t.gap for t in traces))}")
    print(f"  Positions: {sorted(set(t.a_position_label for t in traces))}")
    print(f"  Types: {sorted(set(t.trace_type for t in traces))}")
    print()

    # Breakdown by gap
    print("Traces per gap:")
    for g in GAPS:
        subset = [t for t in traces if t.gap == g]
        lengths = [t.trace_length for t in subset]
        print(f"  gap={g:>3d}: {len(subset)} traces, "
              f"trace_length={min(lengths)}–{max(lengths)}")

    print()

    # Breakdown by condition
    print("Traces per (gap × position × type):")
    for g in [1, 50, 500]:  # sample a few
        for pos in ["early", "middle", "late"]:
            for ttype in ["satisfied", "violated", "distractor"]:
                subset = [t for t in traces
                          if t.gap == g and t.a_position_label == pos
                          and t.trace_type == ttype]
                if subset:
                    t0 = subset[0]
                    print(f"  gap={g:>3d}, pos={pos:>6s}, type={ttype:>10s}: "
                          f"N={t0.trace_length}, a_pos={t0.a_position}, "
                          f"b_pos={t0.b_position}, "
                          f"verdict={t0.ground_truth_verdict}")

    print()

    # Validation: check no trace has both A and B at the same timestep
    issues = 0
    for t in traces:
        for i, e in enumerate(t.events):
            if e.prop_A and e.prop_B:
                print(f"  WARNING: trace {t.trace_id} has both A and B at timestep {i}")
                issues += 1
        # Check ground truth consistency
        a_found = False
        b_after_a = False
        for i, e in enumerate(t.events):
            if e.prop_A:
                a_found = True
            if a_found and e.prop_B:
                b_after_a = True
        expected = "satisfied" if b_after_a else "violated"
        if t.ground_truth_verdict != expected:
            print(f"  WARNING: trace {t.trace_id} verdict mismatch: "
                  f"expected={expected}, got={t.ground_truth_verdict}")
            issues += 1

    if issues == 0:
        print("✓ All traces passed validation.")
    else:
        print(f"✗ {issues} issues found.")


# ─────────────────────────────────────────────
# Formatting traces for LLM prompts
# ─────────────────────────────────────────────

def format_trace_for_prompt(trace: Trace, include_labels: bool = False) -> str:
    """
    Format a trace as a string suitable for LLM prompts.

    Each timestep is presented as a natural English sentence describing
    the event (color, shape, number, animal).

    Args:
        trace: The trace to format.
        include_labels: If True, include ground-truth proposition labels
                       at each timestep (for the +Labels baseline).
    """
    lines = []
    for t, event in enumerate(trace.events):
        line = f"Step {t + 1}: {event.to_str()}"
        if include_labels:
            labels = []
            if event.prop_A:
                labels.append("A=True (color is red)")
            if event.prop_B:
                labels.append("B=True (animal is cat)")
            if labels:
                line += f"  [Labels: {', '.join(labels)}]"
        lines.append(line)
    return "\n".join(lines)


def format_constraint_description(
    framing: Literal["positive", "negative", "formal"] = "positive"
) -> str:
    """Format the constraint □(A → ◇B) in different framings."""
    if framing == "positive":
        return (
            "Constraint: Whenever the color is \"red\" at some timestep, "
            "eventually, the animal must be \"cat\" at some later timestep. "
            "This must hold for every timestep where color is \"red\"."
        )

    else:
        raise ValueError(f"Unknown framing: {framing}")


# ─────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────

def save_dataset(traces: list[Trace], filepath: str) -> None:
    """Save traces to a JSON file."""
    data = {
        "experiment": "Temporal Elasticity (simple formula)",
        "constraint": "□(A → ◇B)",
        "proposition_A": "color == red",
        "proposition_B": "animal == cat",
        "parameters": {
            "gaps": GAPS,
            "positions": ["early", "middle", "late"],
            "trace_types": ["satisfied", "violated", "distractor"],
            "traces_per_cell": TRACES_PER_CELL,
            "padding": PADDING,
            "min_trace_length": MIN_TRACE_LENGTH,
        },
        "num_traces": len(traces),
        "traces": [t.to_dict() for t in traces],
    }
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(traces)} traces to {filepath}")
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Temporal Elasticity (simple formula) — Trace Generator")
    print("=" * 60)
    print()

    traces = generate_all_traces(seed=42)
    print_dataset_stats(traces)

    output_path = "data/simple_traces.json"
    save_dataset(traces, output_path)

    # Print a sample trace for inspection
    print()
    print("=" * 60)
    print("Sample trace (satisfied, gap=100, early):")
    print("=" * 60)
    sample = next(t for t in traces
                  if t.gap == 100 and t.a_position_label == "early"
                  and t.trace_type == "satisfied")
    print(f"Trace ID: {sample.trace_id}")
    print(f"Length: {sample.trace_length}, A at: {sample.a_position}, B at: {sample.b_position}")
    print(f"Verdict: {sample.ground_truth_verdict}")
    print()
    print(format_trace_for_prompt(sample))

    print()
    print("=" * 60)
    print("Sample trace (distractor, gap=100, late):")
    print("=" * 60)
    sample_d = next(t for t in traces
                    if t.gap == 100 and t.a_position_label == "late"
                    and t.trace_type == "distractor")
    print(f"Trace ID: {sample_d.trace_id}")
    print(f"Length: {sample_d.trace_length}, A at: {sample_d.a_position}, "
          f"Distractor B at: {sample_d.distractor_b_position}")
    print(f"Verdict: {sample_d.ground_truth_verdict}")
    print()
    print(format_trace_for_prompt(sample_d))