"""
Specification Level Experiment: Trace Generator & Verifier

Generates traces for 7 temporal constraint patterns at 3 specification levels
(informal NL, precise NL, precise NL + LTL) to test whether specification
formalism affects LLM accuracy on trace checking.

Patterns:
  1. Universality / Global:       G(P)
  2. Absence / Global:            G(¬P)
  3. Response / Global:            G(P → F(S))
  4. Absence / Between Q and R:   G((Q ∧ ¬R ∧ ◇R) → (¬P U R))
  5. Constrained Response:         G(P → (¬Q U R))
  6. Tree (b=2, d=1):             F(a ∧ XF((b1 ∧ XF(d)) ∨ (b2 ∧ XF(d))))
  7. Tree (b=2, d=4):             (16-path nested formula)

Domain: Each event has 4 attributes: animal, shape, color, number.
Propositions are grounded as conditions on these attributes.
"""
from __future__ import annotations

import json
import random
import argparse
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# Domain constants
# ─────────────────────────────────────────────

ANIMALS = [
    "toucan", "crane", "pelican", "owl", "hawk", "parrot", "eagle",
    "robin", "sparrow", "falcon", "heron", "dove", "raven", "finch", "jay"
]
SHAPES = [
    "circle", "triangle", "square", "star", "diamond", "hexagon",
    "pentagon", "oval", "cross", "heart"
]
COLORS = [
    "red", "blue", "green", "yellow", "purple", "orange",
    "pink", "black", "white", "brown"
]
NUMBER_RANGE = (1, 50)


# ─────────────────────────────────────────────
# Proposition groundings
# ─────────────────────────────────────────────

# Each formula uses different attribute-based propositions to avoid
# cross-formula interference.

PROP_GROUNDINGS = {
    "universality": {
        # P = "color is red"
        "P": {"attribute": "color", "value": "red"},
    },
    "absence": {
        # P = "animal is owl"
        "P": {"attribute": "animal", "value": "owl"},
    },
    "response": {
        # P = "shape is triangle", S = "color is blue"
        "P": {"attribute": "shape", "value": "triangle"},
        "S": {"attribute": "color", "value": "blue"},
    },
    "absence_between": {
        # Q = "animal is fox", R = "shape is star", P = "color is green"
        # fox is not in default ANIMALS, add it
        "Q": {"attribute": "animal", "value": "fox"},
        "R": {"attribute": "shape", "value": "star"},
        "P": {"attribute": "color", "value": "green"},
    },
    "constrained_response": {
        # P = "shape is square", Q = "animal is fox", R = "shape is circle"
        "P": {"attribute": "shape", "value": "square"},
        "Q": {"attribute": "animal", "value": "fox"},
        "R": {"attribute": "shape", "value": "circle"},
    },
}

# Add fox to ANIMALS
if "fox" not in ANIMALS:
    ANIMALS.append("fox")


# ─────────────────────────────────────────────
# Tree definitions (patterns 6 and 7)
# ─────────────────────────────────────────────
# Trees use single-attribute propositions (animal only) for node matching.
# Each node has a unique animal label. The sink node "d" uses "deer".

# Tree b=2, d=1:
#     a(toucan)
#      /    \
# b1(crane) b2(pelican)
#     |        |
#   d(deer)  d(deer)

TREE_B2D1 = {
    "nodes": {
        "a": "toucan",
        "b1": "crane",
        "b2": "pelican",
        "d": "deer",
    },
    "paths": [
        ["a", "b1", "d"],
        ["a", "b2", "d"],
    ],
}

# Tree b=2, d=4:
#                            a(toucan)
#                         /            \
#                   b1(crane)         b2(pelican)
#                  /       \          /        \
#            c1(hawk)   c2(parrot) c3(eagle)  c4(robin)
#            /    \      /    \     /    \      /    \
#       e1(spar) e2(fal) e3(her) e4(dov) e5(rav) e6(fin) e7(jay) e8(owl)
#        / \     / \     / \     / \     / \     / \      / \     / \
#      f1  f2  f3  f4  f5  f6  f7  f8  f9 f10 f11 f12 f13 f14 f15 f16
#       |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
#      d   d   d   d   d   d   d   d   d   d   d   d   d   d   d   d
#
# f-nodes use color as distinguishing attribute since we run out of animals

TREE_B2D4 = {
    "nodes": {
        "a": {"attribute": "animal", "value": "toucan"},
        "b1": {"attribute": "animal", "value": "crane"},
        "b2": {"attribute": "animal", "value": "pelican"},
        "c1": {"attribute": "animal", "value": "hawk"},
        "c2": {"attribute": "animal", "value": "parrot"},
        "c3": {"attribute": "animal", "value": "eagle"},
        "c4": {"attribute": "animal", "value": "robin"},
        "e1": {"attribute": "animal", "value": "sparrow"},
        "e2": {"attribute": "animal", "value": "falcon"},
        "e3": {"attribute": "animal", "value": "heron"},
        "e4": {"attribute": "animal", "value": "dove"},
        "e5": {"attribute": "animal", "value": "raven"},
        "e6": {"attribute": "animal", "value": "finch"},
        "e7": {"attribute": "animal", "value": "jay"},
        "e8": {"attribute": "shape", "value": "oval"},  # reuse shape for e8
        "f1": {"attribute": "color", "value": "red"},
        "f2": {"attribute": "color", "value": "blue"},
        "f3": {"attribute": "color", "value": "green"},
        "f4": {"attribute": "color", "value": "yellow"},
        "f5": {"attribute": "color", "value": "purple"},
        "f6": {"attribute": "color", "value": "orange"},
        "f7": {"attribute": "color", "value": "pink"},
        "f8": {"attribute": "color", "value": "black"},
        "f9": {"attribute": "color", "value": "white"},
        "f10": {"attribute": "color", "value": "brown"},
        "f11": {"attribute": "number", "value": 1},
        "f12": {"attribute": "number", "value": 2},
        "f13": {"attribute": "number", "value": 3},
        "f14": {"attribute": "number", "value": 4},
        "f15": {"attribute": "number", "value": 5},
        "f16": {"attribute": "number", "value": 6},
        "d": {"attribute": "animal", "value": "deer"},
    },
    "paths": [
        ["a", "b1", "c1", "e1", "f1", "d"],
        ["a", "b1", "c1", "e1", "f2", "d"],
        ["a", "b1", "c1", "e2", "f3", "d"],
        ["a", "b1", "c1", "e2", "f4", "d"],
        ["a", "b1", "c2", "e3", "f5", "d"],
        ["a", "b1", "c2", "e3", "f6", "d"],
        ["a", "b1", "c2", "e4", "f7", "d"],
        ["a", "b1", "c2", "e4", "f8", "d"],
        ["a", "b2", "c3", "e5", "f9", "d"],
        ["a", "b2", "c3", "e5", "f10", "d"],
        ["a", "b2", "c3", "e6", "f11", "d"],
        ["a", "b2", "c3", "e6", "f12", "d"],
        ["a", "b2", "c4", "e7", "f13", "d"],
        ["a", "b2", "c4", "e7", "f14", "d"],
        ["a", "b2", "c4", "e8", "f15", "d"],
        ["a", "b2", "c4", "e8", "f16", "d"],
    ],
}


# ─────────────────────────────────────────────
# Event data structure
# ─────────────────────────────────────────────

@dataclass
class Event:
    animal: str
    shape: str
    color: str
    number: int

    def to_dict(self) -> dict:
        return {
            "animal": self.animal,
            "shape": self.shape,
            "color": self.color,
            "number": self.number,
        }

    def to_nl(self) -> str:
        """Natural language rendering of the event."""
        def _article(word: str) -> str:
            return "an" if word[0].lower() in "aeiou" else "a"

        templates = [
            "{A_color} {color} {shape} labeled {number} with {a_animal} {animal}.",
            "There is {a_color} {color} {shape} marked {number}, accompanied by {a_animal} {animal}.",
            "Observed {a_color} {color} {shape} (number {number}) alongside {a_animal} {animal}.",
            "{A_animal} {animal} is next to {a_color} {color} {shape} bearing the number {number}.",
            "Item {number}: {a_color} {color} {shape} paired with {a_animal} {animal}.",
        ]
        idx = (ord(self.color[0]) + ord(self.animal[0]) + self.number) % len(templates)
        return templates[idx].format(
            color=self.color, shape=self.shape,
            number=self.number, animal=self.animal,
            a_color=_article(self.color),
            A_color=_article(self.color).capitalize(),
            a_animal=_article(self.animal),
            A_animal=_article(self.animal).capitalize(),
        )

    def has_prop(self, prop_name: str, formula_type: str) -> bool:
        """Check if this event satisfies a named proposition."""
        grounding = PROP_GROUNDINGS[formula_type][prop_name]
        return getattr(self, grounding["attribute"]) == grounding["value"]


# ─────────────────────────────────────────────
# Trace data structure
# ─────────────────────────────────────────────

@dataclass
class Trace:
    trace_id: str
    formula_type: str
    trace_category: str  # "clear_sat", "clear_viol", "edge_sat", "edge_viol"
    edge_case_tag: str   # description of what edge case this tests, or "none"
    events: list[Event] = field(default_factory=list)
    ground_truth: str = ""  # "satisfies" or "violates"

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "formula_type": self.formula_type,
            "trace_category": self.trace_category,
            "edge_case_tag": self.edge_case_tag,
            "ground_truth": self.ground_truth,
            "trace_length": len(self.events),
            "events": [
                {"timestep": t, **e.to_dict()}
                for t, e in enumerate(self.events)
            ],
        }


# ─────────────────────────────────────────────
# Event generators
# ─────────────────────────────────────────────

def make_filler_event(rng: random.Random, formula_type: str) -> Event:
    """
    Generate a filler event that does NOT accidentally trigger any
    signal proposition for the given formula type.
    """
    groundings = PROP_GROUNDINGS[formula_type]
    forbidden = {}
    for prop_name, g in groundings.items():
        attr = g["attribute"]
        val = g["value"]
        if attr not in forbidden:
            forbidden[attr] = set()
        forbidden[attr].add(val)

    def safe_choice(pool, attr_name):
        if attr_name in forbidden:
            safe = [x for x in pool if x not in forbidden[attr_name]]
            return rng.choice(safe)
        return rng.choice(pool)

    return Event(
        animal=safe_choice(ANIMALS, "animal"),
        shape=safe_choice(SHAPES, "shape"),
        color=safe_choice(COLORS, "color"),
        number=rng.randint(*NUMBER_RANGE),
    )


def make_event_with_props(
    rng: random.Random,
    formula_type: str,
    props: list[str],
    avoid_props: list[str] | None = None,
) -> Event:
    """
    Generate an event that satisfies all named propositions in `props`,
    does NOT satisfy any in `avoid_props`, and fills remaining attributes
    with safe filler values.
    """
    groundings = PROP_GROUNDINGS[formula_type]

    # Start with a filler
    event = make_filler_event(rng, formula_type)
    event_dict = event.to_dict()

    # Set required props
    for prop_name in props:
        g = groundings[prop_name]
        event_dict[g["attribute"]] = g["value"]

    # Ensure avoid_props are NOT set
    if avoid_props:
        for prop_name in avoid_props:
            g = groundings[prop_name]
            attr = g["attribute"]
            val = g["value"]
            if event_dict[attr] == val:
                # Pick a different value
                if attr == "animal":
                    pool = [x for x in ANIMALS if x != val]
                elif attr == "shape":
                    pool = [x for x in SHAPES if x != val]
                elif attr == "color":
                    pool = [x for x in COLORS if x != val]
                else:
                    pool = [x for x in range(*NUMBER_RANGE) if x != val]
                event_dict[attr] = rng.choice(pool)

    return Event(**event_dict)


def make_filler_padding(rng: random.Random, formula_type: str, n: int) -> list[Event]:
    """Generate n filler events."""
    return [make_filler_event(rng, formula_type) for _ in range(n)]


# Target trace lengths per formula type
TARGET_LENGTHS = {
    "universality": 100,
    "absence": 100,
    "response": 100,
    "absence_between": 100,
    "constrained_response": 100,
    "tree_b2d1": 100,
    "tree_b2d4": 200,
}


def pad_trace_to_length(trace: Trace, rng: random.Random) -> Trace:
    """
    Pad a trace to the target length by inserting filler events
    at random positions throughout the trace. This spreads the
    signal events across a longer trace rather than just appending
    filler at the end.

    Special handling for universality: since G(P) requires P at every
    step, padding uses P-satisfying events instead of filler.

    Empty traces (edge case) are left as-is.
    """
    target = TARGET_LENGTHS.get(trace.formula_type, 100)
    current = len(trace.events)

    if current == 0 or current >= target:
        return trace

    n_to_add = target - current
    ft = trace.formula_type

    if ft == "universality":
        # For G(P), all events must satisfy P — pad with P events
        if trace.ground_truth == "satisfies":
            new_events = [make_event_with_props(rng, ft, ["P"])
                          for _ in range(n_to_add)]
        else:
            # Violating trace: pad with P-satisfying events (don't add more violations)
            new_events = [make_event_with_props(rng, ft, ["P"])
                          for _ in range(n_to_add)]
    elif ft in ("tree_b2d1", "tree_b2d4"):
        # For trees, use tree-specific filler
        tree_def = TREE_B2D1 if ft == "tree_b2d1" else TREE_B2D4
        new_events = [_make_tree_filler(rng, tree_def) for _ in range(n_to_add)]
    else:
        new_events = make_filler_padding(rng, ft, n_to_add)

    # Insert new events at random positions throughout the trace
    events = list(trace.events)
    for evt in new_events:
        pos = rng.randint(0, len(events))
        events.insert(pos, evt)

    trace.events = events
    return trace


# ─────────────────────────────────────────────
# Trace generators per formula
# ─────────────────────────────────────────────

def generate_universality_traces(rng: random.Random) -> list[Trace]:
    """
    G(P) where P = "color is red"
    Every step must have color=red.
    """
    ft = "universality"
    traces = []
    idx = 0

    def tid():
        nonlocal idx; idx += 1
        return f"univ_{idx:03d}"

    # --- Clear satisfying ---
    # All red, short
    events = [make_event_with_props(rng, ft, ["P"]) for _ in range(5)]
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # All red, longer
    events = [make_event_with_props(rng, ft, ["P"]) for _ in range(15)]
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # All red, length 1
    events = [make_event_with_props(rng, ft, ["P"])]
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # Generate more random clear satisfying traces
    for length in [3, 7, 10, 12, 20]:
        events = [make_event_with_props(rng, ft, ["P"]) for _ in range(length)]
        traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # --- Clear violating ---
    # Fails at start
    events = [make_filler_event(rng, ft)] + [make_event_with_props(rng, ft, ["P"]) for _ in range(4)]
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # Fails at end
    events = [make_event_with_props(rng, ft, ["P"]) for _ in range(4)] + [make_filler_event(rng, ft)]
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # Fails in middle
    events = ([make_event_with_props(rng, ft, ["P"]) for _ in range(3)]
              + [make_filler_event(rng, ft)]
              + [make_event_with_props(rng, ft, ["P"]) for _ in range(3)])
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # Multiple failures
    events = [make_filler_event(rng, ft) for _ in range(5)]
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # Generate more random clear violating traces
    for length in [5, 8, 10, 15, 20]:
        events = [make_event_with_props(rng, ft, ["P"]) for _ in range(length)]
        fail_pos = rng.randint(0, length - 1)
        events[fail_pos] = make_filler_event(rng, ft)
        traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # --- Edge cases ---
    # Empty trace: vacuously satisfies
    traces.append(Trace(tid(), ft, "edge_sat", "empty_trace", [], "satisfies"))

    # P with extra propositions (still satisfies)
    events = [make_event_with_props(rng, ft, ["P"]) for _ in range(5)]
    traces.append(Trace(tid(), ft, "edge_sat", "extra_props_present", events, "satisfies"))

    # Single step satisfies
    events = [make_event_with_props(rng, ft, ["P"])]
    traces.append(Trace(tid(), ft, "edge_sat", "single_step_sat", events, "satisfies"))

    # Single step violates
    events = [make_filler_event(rng, ft)]
    traces.append(Trace(tid(), ft, "edge_viol", "single_step_viol", events, "violates"))

    return traces


def generate_absence_traces(rng: random.Random) -> list[Trace]:
    """
    G(¬P) where P = "animal is owl"
    No step should have animal=owl.
    """
    ft = "absence"
    traces = []
    idx = 0

    def tid():
        nonlocal idx; idx += 1
        return f"abs_{idx:03d}"

    # --- Clear satisfying ---
    for length in [1, 3, 5, 7, 10, 12, 15, 20]:
        events = [make_filler_event(rng, ft) for _ in range(length)]
        traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # --- Clear violating ---
    # P at start
    events = [make_event_with_props(rng, ft, ["P"])] + make_filler_padding(rng, ft, 4)
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # P at end
    events = make_filler_padding(rng, ft, 4) + [make_event_with_props(rng, ft, ["P"])]
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # P in middle
    events = (make_filler_padding(rng, ft, 3)
              + [make_event_with_props(rng, ft, ["P"])]
              + make_filler_padding(rng, ft, 3))
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # Multiple P's
    for length in [5, 8, 10, 15]:
        events = make_filler_padding(rng, ft, length)
        n_p = rng.randint(2, min(4, length))
        positions = rng.sample(range(length), n_p)
        for pos in positions:
            events[pos] = make_event_with_props(rng, ft, ["P"])
        traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # --- Edge cases ---
    # Empty trace
    traces.append(Trace(tid(), ft, "edge_sat", "empty_trace", [], "satisfies"))

    # P co-occurs with other attributes (still violates)
    events = make_filler_padding(rng, ft, 3)
    events[1] = make_event_with_props(rng, ft, ["P"])
    traces.append(Trace(tid(), ft, "edge_viol", "P_with_other_attrs", events, "violates"))

    # Single step with P
    events = [make_event_with_props(rng, ft, ["P"])]
    traces.append(Trace(tid(), ft, "edge_viol", "single_step_P", events, "violates"))

    return traces


def generate_response_traces(rng: random.Random) -> list[Trace]:
    """
    G(P → F(S)) where P = "shape is triangle", S = "color is blue"
    Every triangle must be followed (at same step or later) by blue.
    """
    ft = "response"
    traces = []
    idx = 0

    def tid():
        nonlocal idx; idx += 1
        return f"resp_{idx:03d}"

    # --- Clear satisfying ---
    # Basic P then S
    events = (make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["P"])]
              + make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["S"])]
              + make_filler_padding(rng, ft, 2))
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # Multiple P's each followed by S
    events = ([make_event_with_props(rng, ft, ["P"])]
              + make_filler_padding(rng, ft, 1)
              + [make_event_with_props(rng, ft, ["S"])]
              + make_filler_padding(rng, ft, 1)
              + [make_event_with_props(rng, ft, ["P"])]
              + make_filler_padding(rng, ft, 1)
              + [make_event_with_props(rng, ft, ["S"])])
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # One S at end covers multiple P's
    events = ([make_event_with_props(rng, ft, ["P"])]
              + make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["P"])]
              + make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["S"])])
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # Generate more clear satisfying
    for _ in range(5):
        length = rng.randint(8, 15)
        events = make_filler_padding(rng, ft, length)
        n_p = rng.randint(1, 3)
        p_positions = sorted(rng.sample(range(length - 1), n_p))
        # Place S after last P
        s_pos = rng.randint(p_positions[-1], length - 1)
        for pos in p_positions:
            events[pos] = make_event_with_props(rng, ft, ["P"])
        events[s_pos] = make_event_with_props(rng, ft, ["S"])
        traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # --- Clear violating ---
    # P with no S ever
    events = ([make_event_with_props(rng, ft, ["P"])]
              + make_filler_padding(rng, ft, 5))
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # S before P but no S after
    events = ([make_event_with_props(rng, ft, ["S"])]
              + make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["P"])]
              + make_filler_padding(rng, ft, 3))
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # Last P unanswered
    events = ([make_event_with_props(rng, ft, ["P"])]
              + make_filler_padding(rng, ft, 1)
              + [make_event_with_props(rng, ft, ["S"])]
              + make_filler_padding(rng, ft, 1)
              + [make_event_with_props(rng, ft, ["P"])]
              + make_filler_padding(rng, ft, 2))
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # Generate more clear violating
    for _ in range(5):
        length = rng.randint(8, 15)
        events = make_filler_padding(rng, ft, length)
        # Place P near the end with no S after
        p_pos = rng.randint(length - 3, length - 1)
        events[p_pos] = make_event_with_props(rng, ft, ["P"])
        # Maybe place an S before the P (doesn't help)
        if rng.random() < 0.5 and p_pos > 1:
            s_pos = rng.randint(0, p_pos - 1)
            events[s_pos] = make_event_with_props(rng, ft, ["S"])
        traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # --- Edge cases ---
    # P and S same step (satisfies!)
    events = (make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["P", "S"])]
              + make_filler_padding(rng, ft, 2))
    traces.append(Trace(tid(), ft, "edge_sat", "P_and_S_same_step", events, "satisfies"))

    # P and S co-occur at one step, and that's the ONLY P and ONLY S in the trace
    events = (make_filler_padding(rng, ft, 4)
              + [make_event_with_props(rng, ft, ["P", "S"])]
              + make_filler_padding(rng, ft, 4))
    traces.append(Trace(tid(), ft, "edge_sat", "P_S_same_step_only_occurrence", events, "satisfies"))

    # P and S co-occur at the very last step
    events = make_filler_padding(rng, ft, 6) + [make_event_with_props(rng, ft, ["P", "S"])]
    traces.append(Trace(tid(), ft, "edge_sat", "P_S_same_step_last", events, "satisfies"))

    # Multiple P+S co-occurrences scattered through trace
    events = (make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["P", "S"])]
              + make_filler_padding(rng, ft, 3)
              + [make_event_with_props(rng, ft, ["P", "S"])]
              + make_filler_padding(rng, ft, 2))
    traces.append(Trace(tid(), ft, "edge_sat", "P_S_same_step_multiple", events, "satisfies"))

    # No P at all (vacuously satisfies)
    events = make_filler_padding(rng, ft, 8)
    traces.append(Trace(tid(), ft, "edge_sat", "no_P_vacuous", events, "satisfies"))

    # No P, but S present (vacuously satisfies)
    events = make_filler_padding(rng, ft, 5)
    events[2] = make_event_with_props(rng, ft, ["S"])
    traces.append(Trace(tid(), ft, "edge_sat", "no_P_with_S_vacuous", events, "satisfies"))

    # One S covers multiple P's (satisfies)
    events = ([make_event_with_props(rng, ft, ["P"])]
              + [make_event_with_props(rng, ft, ["P"])]
              + [make_event_with_props(rng, ft, ["S"])])
    traces.append(Trace(tid(), ft, "edge_sat", "one_S_covers_multiple_P", events, "satisfies"))

    # S before P only (violates — S doesn't retroactively cover P)
    events = ([make_event_with_props(rng, ft, ["S"])]
              + make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["P"])])
    traces.append(Trace(tid(), ft, "edge_viol", "S_before_P_only", events, "violates"))

    # P at very last step, no S (violates)
    events = make_filler_padding(rng, ft, 6) + [make_event_with_props(rng, ft, ["P"])]
    traces.append(Trace(tid(), ft, "edge_viol", "P_at_last_step", events, "violates"))

    return traces


def generate_absence_between_traces(rng: random.Random) -> list[Trace]:
    """
    G((Q ∧ ¬R ∧ ◇R) → (¬P U R))
    where Q = "animal is fox", R = "shape is star", P = "color is green"
    No green between fox and star.
    """
    ft = "absence_between"
    traces = []
    idx = 0

    def tid():
        nonlocal idx; idx += 1
        return f"absbtw_{idx:03d}"

    # --- Clear satisfying ---
    # Q then R, no P between
    events = (make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])]
              + make_filler_padding(rng, ft, 3)
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])]
              + make_filler_padding(rng, ft, 2))
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # Q then R immediately, no room for P
    events = (make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])]
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])]
              + make_filler_padding(rng, ft, 2))
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # P occurs but outside Q-R scope (before Q)
    events = ([make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])]
              + make_filler_padding(rng, ft, 1)
              + [make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])]
              + make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])])
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # P occurs after R (outside scope)
    events = ([make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])]
              + make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])]
              + make_filler_padding(rng, ft, 1)
              + [make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])])
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # No Q at all
    events = make_filler_padding(rng, ft, 8)
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # Generate more clear satisfying
    for _ in range(3):
        length = rng.randint(8, 15)
        events = make_filler_padding(rng, ft, length)
        q_pos = rng.randint(1, length - 4)
        r_pos = rng.randint(q_pos + 2, length - 1)
        events[q_pos] = make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])
        events[r_pos] = make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])
        traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # --- Clear violating ---
    # P between Q and R
    events = ([make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])]
              + make_filler_padding(rng, ft, 1)
              + [make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])]
              + make_filler_padding(rng, ft, 1)
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])])
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # P immediately after Q
    events = ([make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])]
              + [make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])]
              + make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])])
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # P just before R
    events = ([make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])]
              + make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])]
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])])
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # Generate more clear violating
    for _ in range(5):
        length = rng.randint(8, 15)
        events = make_filler_padding(rng, ft, length)
        q_pos = rng.randint(1, length - 5)
        r_pos = rng.randint(q_pos + 3, length - 1)
        p_pos = rng.randint(q_pos + 1, r_pos - 1)
        events[q_pos] = make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])
        events[r_pos] = make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])
        events[p_pos] = make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])
        traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # --- Edge cases ---

    # Q and R same step (no scope opens — satisfies)
    events = (make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["Q", "R"], avoid_props=["P"])]
              + make_filler_padding(rng, ft, 2))
    traces.append(Trace(tid(), ft, "edge_sat", "Q_R_same_step", events, "satisfies"))

    # Q, R, and P all same step (no scope opens — satisfies!)
    events = (make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["Q", "R", "P"])]
              + make_filler_padding(rng, ft, 2))
    traces.append(Trace(tid(), ft, "edge_sat", "Q_R_P_same_step", events, "satisfies"))

    # R never comes after Q (vacuously satisfies — ◇R fails)
    events = ([make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])]
              + [make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])]
              + make_filler_padding(rng, ft, 4))
    traces.append(Trace(tid(), ft, "edge_sat", "R_never_comes", events, "satisfies"))

    # P at Q step (scope opens, P must be false at Q step — violates)
    events = ([make_event_with_props(rng, ft, ["Q", "P"], avoid_props=["R"])]
              + make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])])
    traces.append(Trace(tid(), ft, "edge_viol", "P_at_Q_step", events, "violates"))

    # Multiple Q's before R, P in second scope
    events = ([make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])]
              + make_filler_padding(rng, ft, 1)
              + [make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])]
              + [make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])]
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])])
    traces.append(Trace(tid(), ft, "edge_viol", "multiple_Q_before_R", events, "violates"))

    # No Q at all, P present (satisfies — no scope ever opens)
    events = make_filler_padding(rng, ft, 3)
    events[1] = make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])
    traces.append(Trace(tid(), ft, "edge_sat", "no_Q_at_all", events, "satisfies"))

    # Nested scopes: Q R Q P R (second scope violated)
    events = ([make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])]
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])]
              + [make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])]
              + [make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])]
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])])
    traces.append(Trace(tid(), ft, "edge_viol", "nested_scope_violated", events, "violates"))

    # Nested scopes: Q R Q R (both scopes clean — satisfies)
    events = ([make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])]
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])]
              + [make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])]
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])])
    traces.append(Trace(tid(), ft, "edge_sat", "nested_scope_clean", events, "satisfies"))

    return traces


def generate_constrained_response_traces(rng: random.Random) -> list[Trace]:
    """
    G(P → (¬Q U R))
    where P = "shape is square", Q = "animal is fox", R = "shape is circle"
    Whenever square appears, fox must not appear until circle appears.
    Strong until: circle MUST eventually come.
    """
    ft = "constrained_response"
    traces = []
    idx = 0

    def tid():
        nonlocal idx; idx += 1
        return f"cresp_{idx:03d}"

    # --- Clear satisfying ---
    # Square then circle, no fox between
    events = (make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])]
              + make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])]
              + make_filler_padding(rng, ft, 2))
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # Square then circle immediately
    events = (make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])]
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])]
              + make_filler_padding(rng, ft, 2))
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # No square at all (vacuously satisfies)
    events = make_filler_padding(rng, ft, 8)
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # Fox after circle (outside constraint scope)
    events = ([make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])]
              + make_filler_padding(rng, ft, 1)
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])]
              + make_filler_padding(rng, ft, 1)
              + [make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])])
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # Multiple squares each followed by circle
    events = ([make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])]
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])]
              + make_filler_padding(rng, ft, 1)
              + [make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])]
              + make_filler_padding(rng, ft, 1)
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])])
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    for _ in range(3):
        length = rng.randint(8, 15)
        events = make_filler_padding(rng, ft, length)
        p_pos = rng.randint(1, length - 4)
        r_pos = rng.randint(p_pos + 2, length - 1)
        events[p_pos] = make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])
        events[r_pos] = make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])
        traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # --- Clear violating ---
    # Fox between square and circle
    events = ([make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])]
              + [make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])]
              + make_filler_padding(rng, ft, 1)
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])])
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # Square with no circle ever (strong until requires R)
    events = ([make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])]
              + make_filler_padding(rng, ft, 5))
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # Fox immediately after square
    events = ([make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])]
              + [make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])]
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])])
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    for _ in range(5):
        length = rng.randint(8, 15)
        events = make_filler_padding(rng, ft, length)
        p_pos = rng.randint(1, length - 5)
        r_pos = rng.randint(p_pos + 3, length - 1)
        q_pos = rng.randint(p_pos + 1, r_pos - 1)
        events[p_pos] = make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])
        events[r_pos] = make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])
        events[q_pos] = make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])
        traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # --- Edge cases ---
    # Square and circle at same step (satisfies — R holds immediately)
    events = (make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["P", "R"], avoid_props=["Q"])]
              + make_filler_padding(rng, ft, 2))
    traces.append(Trace(tid(), ft, "edge_sat", "P_R_same_step", events, "satisfies"))

    # Fox at square step (violates — ¬Q must hold from current step, Q is true)
    events = ([make_event_with_props(rng, ft, ["P", "Q"], avoid_props=["R"])]
              + make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["R"], avoid_props=["P", "Q"])])
    traces.append(Trace(tid(), ft, "edge_viol", "Q_at_P_step", events, "violates"))

    # Fox and circle at same step after square (satisfies — R holds, U satisfied)
    events = ([make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])]
              + make_filler_padding(rng, ft, 2)
              + [make_event_with_props(rng, ft, ["Q", "R"], avoid_props=["P"])])
    traces.append(Trace(tid(), ft, "edge_sat", "Q_R_same_step_after_P", events, "satisfies"))

    # Circle never comes, no fox either (violates — strong until requires R)
    events = ([make_event_with_props(rng, ft, ["P"], avoid_props=["Q", "R"])]
              + make_filler_padding(rng, ft, 5))
    traces.append(Trace(tid(), ft, "edge_viol", "R_never_comes_strong_until", events, "violates"))

    # No square at all, fox present (satisfies vacuously)
    events = make_filler_padding(rng, ft, 4)
    events[1] = make_event_with_props(rng, ft, ["Q"], avoid_props=["P", "R"])
    traces.append(Trace(tid(), ft, "edge_sat", "no_P_vacuous", events, "satisfies"))

    return traces


# ─────────────────────────────────────────────
# Tree trace generators
# ─────────────────────────────────────────────

# For trees, we use a different grounding approach: each node is identified
# by a specific attribute-value pair. We build events that match nodes
# and use filler events that don't accidentally match any node.

def _tree_node_match(event: Event, node_name: str, tree_def: dict) -> bool:
    """Check if an event matches a tree node."""
    if "nodes" in tree_def and isinstance(list(tree_def["nodes"].values())[0], dict):
        # Complex tree (b2d4) with attribute/value dicts
        spec = tree_def["nodes"][node_name]
        return getattr(event, spec["attribute"]) == spec["value"]
    else:
        # Simple tree (b2d1) with animal-only mapping
        return event.animal == tree_def["nodes"][node_name]


def _make_tree_node_event(rng: random.Random, node_name: str,
                          tree_def: dict) -> Event:
    """Create an event that matches a specific tree node."""
    # Start with random safe values
    animal = rng.choice(["fox", "deer", "toucan", "crane"])
    shape = rng.choice(["heart", "cross", "diamond", "hexagon"])
    color = rng.choice(["red", "blue", "green", "yellow", "purple",
                         "orange", "pink", "black", "white", "brown"])
    number = rng.randint(7, 45)

    if "nodes" in tree_def and isinstance(list(tree_def["nodes"].values())[0], dict):
        spec = tree_def["nodes"][node_name]
        if spec["attribute"] == "animal":
            animal = spec["value"]
        elif spec["attribute"] == "shape":
            shape = spec["value"]
        elif spec["attribute"] == "color":
            color = spec["value"]
        elif spec["attribute"] == "number":
            number = spec["value"]
    else:
        animal = tree_def["nodes"][node_name]

    return Event(animal=animal, shape=shape, color=color, number=number)


def _make_tree_filler(rng: random.Random, tree_def: dict) -> Event:
    """Create a filler event that doesn't match any tree node."""
    # Collect all values used by any node to avoid them
    used_animals = set()
    used_shapes = set()
    used_colors = set()
    used_numbers = set()

    if "nodes" in tree_def and isinstance(list(tree_def["nodes"].values())[0], dict):
        for spec in tree_def["nodes"].values():
            attr, val = spec["attribute"], spec["value"]
            if attr == "animal":
                used_animals.add(val)
            elif attr == "shape":
                used_shapes.add(val)
            elif attr == "color":
                used_colors.add(val)
            elif attr == "number":
                used_numbers.add(val)
    else:
        for val in tree_def["nodes"].values():
            used_animals.add(val)

    safe_animals = [a for a in ["gecko", "moth", "crab", "ant", "bee", "slug"]
                    if a not in used_animals]
    safe_shapes = [s for s in ["pentagon", "cross", "heart"]
                   if s not in used_shapes]
    safe_colors = [c for c in ["gray", "silver", "tan", "beige", "coral"]
                   if c not in used_colors]
    safe_numbers = [n for n in range(30, 45) if n not in used_numbers]

    return Event(
        animal=rng.choice(safe_animals),
        shape=rng.choice(safe_shapes),
        color=rng.choice(safe_colors),
        number=rng.choice(safe_numbers),
    )


def _verify_tree(events: list[Event], tree_def: dict) -> str:
    """
    Verify if a trace satisfies a tree formula.
    The trace must contain events matching one complete path
    in strictly increasing temporal order (not necessarily consecutive).
    """
    n = len(events)

    def check_path(path: list[str], start: int) -> bool:
        pos = start
        for node_name in path:
            found = False
            for i in range(pos, n):
                if _tree_node_match(events[i], node_name, tree_def):
                    pos = i + 1  # next node must be strictly after
                    found = True
                    break
            if not found:
                return False
        return True

    for path in tree_def["paths"]:
        if check_path(path, 0):
            return "satisfies"
    return "violates"


def generate_tree_b2d1_traces(rng: random.Random) -> list[Trace]:
    """
    Tree b=2, d=1: F(a ∧ XF((b1 ∧ XF(d)) ∨ (b2 ∧ XF(d))))
    2 paths: a→b1→d, a→b2→d
    """
    ft = "tree_b2d1"
    tree = TREE_B2D1
    traces = []
    idx = 0

    def tid():
        nonlocal idx; idx += 1
        return f"tb2d1_{idx:03d}"

    def filler(n=1):
        return [_make_tree_filler(rng, tree) for _ in range(n)]

    def node_evt(name):
        return _make_tree_node_event(rng, name, tree)

    # --- Clear satisfying ---
    # Path a→b1→d
    events = filler(2) + [node_evt("a")] + filler(2) + [node_evt("b1")] + filler(1) + [node_evt("d")] + filler(2)
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # Path a→b2→d
    events = filler(1) + [node_evt("a")] + filler(1) + [node_evt("b2")] + filler(2) + [node_evt("d")] + filler(1)
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # Path a→b1→d consecutive
    events = [node_evt("a"), node_evt("b1"), node_evt("d")]
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # Path a→b2→d with lots of filler
    events = filler(3) + [node_evt("a")] + filler(5) + [node_evt("b2")] + filler(4) + [node_evt("d")] + filler(2)
    traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    for _ in range(4):
        path = rng.choice(tree["paths"])
        events = []
        for node_name in path:
            events.extend(filler(rng.randint(1, 4)))
            events.append(node_evt(node_name))
        events.extend(filler(rng.randint(1, 3)))
        traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # --- Clear violating ---
    # Missing d (sink)
    events = filler(2) + [node_evt("a")] + filler(2) + [node_evt("b1")] + filler(3)
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # Missing a (root)
    events = filler(3) + [node_evt("b1")] + filler(2) + [node_evt("d")] + filler(2)
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # Wrong order: d before b1
    events = filler(1) + [node_evt("a")] + filler(1) + [node_evt("d")] + filler(1) + [node_evt("b1")] + filler(1)
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # Only filler
    events = filler(10)
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    attempts = 0
    generated_viol = 0
    while generated_viol < 4 and attempts < 40:
        attempts += 1
        # Random incomplete path
        path = rng.choice(tree["paths"])
        skip = rng.randint(0, len(path) - 1)
        events = []
        for i, node_name in enumerate(path):
            events.extend(filler(rng.randint(1, 3)))
            if i != skip:
                events.append(node_evt(node_name))
        events.extend(filler(rng.randint(1, 2)))
        if _verify_tree(events, tree) == "violates":
            traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))
            generated_viol += 1

    # --- Edge cases ---
    # b1 and b2 both present, one path completes (satisfies)
    events = [node_evt("a")] + filler(1) + [node_evt("b1")] + [node_evt("b2")] + filler(1) + [node_evt("d")]
    traces.append(Trace(tid(), ft, "edge_sat", "both_branches_present", events, "satisfies"))

    # Near-miss: substitute one node in path with wrong value
    events = filler(1) + [node_evt("a")] + filler(1) + [node_evt("b1")] + filler(1) + [_make_tree_filler(rng, tree)] + filler(1)
    traces.append(Trace(tid(), ft, "edge_viol", "near_miss_substitution", events, "violates"))

    # a appears twice, second one starts valid path (satisfies)
    events = [node_evt("a")] + filler(3) + [node_evt("a")] + filler(1) + [node_evt("b2")] + filler(1) + [node_evt("d")]
    traces.append(Trace(tid(), ft, "edge_sat", "repeated_root", events, "satisfies"))

    # Note: For b2d1 all nodes use the animal attribute, so two consecutive
    # tree nodes cannot co-occur at the same step (one animal per event).
    # Same-step edge case is structurally impossible here.

    return traces


def generate_tree_b2d4_traces(rng: random.Random) -> list[Trace]:
    """
    Tree b=2, d=4: 16-path nested formula.
    """
    ft = "tree_b2d4"
    tree = TREE_B2D4
    traces = []
    idx = 0

    def tid():
        nonlocal idx; idx += 1
        return f"tb2d4_{idx:03d}"

    def filler(n=1):
        return [_make_tree_filler(rng, tree) for _ in range(n)]

    def node_evt(name):
        return _make_tree_node_event(rng, name, tree)

    # --- Clear satisfying: pick random paths ---
    for _ in range(8):
        path = rng.choice(tree["paths"])
        events = []
        for node_name in path:
            events.extend(filler(rng.randint(1, 3)))
            events.append(node_evt(node_name))
        events.extend(filler(rng.randint(1, 3)))
        traces.append(Trace(tid(), ft, "clear_sat", "none", events, "satisfies"))

    # --- Clear violating ---
    # Missing sink
    path = rng.choice(tree["paths"])
    events = []
    for node_name in path[:-1]:  # skip d
        events.extend(filler(rng.randint(1, 2)))
        events.append(node_evt(node_name))
    events.extend(filler(3))
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # Wrong branch child (e.g., path says c1 but we put c3)
    path = list(tree["paths"][0])  # a, b1, c1, e1, f1, d
    events = []
    for node_name in path:
        events.extend(filler(rng.randint(1, 2)))
        events.append(node_evt(node_name))
    # Replace c1 with c3 (wrong subtree)
    for i, e in enumerate(events):
        if _tree_node_match(e, "c1", tree):
            events[i] = node_evt("c3")
            break
    events.extend(filler(2))
    # Only add if actually violates (substitution might accidentally form valid path)
    if _verify_tree(events, tree) == "violates":
        traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # Only filler
    events = filler(15)
    traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))

    # Random incomplete paths (skip one node) — verify actually violates
    attempts = 0
    generated_viol = 0
    while generated_viol < 5 and attempts < 50:
        attempts += 1
        path = rng.choice(tree["paths"])
        skip = rng.randint(1, len(path) - 2)  # don't skip root or sink
        events = []
        for i, node_name in enumerate(path):
            events.extend(filler(rng.randint(1, 2)))
            if i != skip:
                events.append(node_evt(node_name))
        events.extend(filler(rng.randint(1, 2)))
        # Verify it actually violates (skipping a node might still leave valid path)
        if _verify_tree(events, tree) == "violates":
            traces.append(Trace(tid(), ft, "clear_viol", "none", events, "violates"))
            generated_viol += 1

    # --- Edge cases ---
    # Two partial paths, neither complete (violates)
    p1 = tree["paths"][0]  # left-most
    p2 = tree["paths"][-1]  # right-most
    events = []
    for node_name in p1[:3]:  # a, b1, c1
        events.extend(filler(1))
        events.append(node_evt(node_name))
    events.extend(filler(2))
    for node_name in p2[3:]:  # e8, f16, d
        events.extend(filler(1))
        events.append(node_evt(node_name))
    if _verify_tree(events, tree) == "violates":
        traces.append(Trace(tid(), ft, "edge_viol", "two_partial_paths", events, "violates"))

    # Very long trace with valid path buried in noise
    path = rng.choice(tree["paths"])
    events = filler(10)
    for node_name in path:
        events.extend(filler(rng.randint(3, 6)))
        events.append(node_evt(node_name))
    events.extend(filler(10))
    traces.append(Trace(tid(), ft, "edge_sat", "path_buried_in_noise", events, "satisfies"))

    # Near-miss: one node substituted — retry until actually violates
    for _ in range(10):
        path = rng.choice(tree["paths"])
        sub_idx = rng.randint(1, len(path) - 2)
        events = []
        for i, node_name in enumerate(path):
            events.extend(filler(rng.randint(1, 2)))
            if i == sub_idx:
                events.append(_make_tree_filler(rng, tree))
            else:
                events.append(node_evt(node_name))
        events.extend(filler(2))
        if _verify_tree(events, tree) == "violates":
            traces.append(Trace(tid(), ft, "edge_viol", "near_miss_one_substitution", events, "violates"))
            break

    # Two consecutive nodes co-occur at the same step (violates — XF requires strictly later)
    # e.g., path[3]=e1 (animal=sparrow) and path[4]=f1 (color=red) use different attributes
    # so they CAN co-occur in one event, but XF means f1 must come strictly after e1
    path = tree["paths"][0]  # a, b1, c1, e1, f1, d
    events = []
    for i, node_name in enumerate(path):
        if i == 3:  # e1 and f1 merged into one event
            merged = node_evt("e1")
            # Also set f1's attribute on the same event
            f_spec = tree["nodes"]["f1"]
            setattr(merged, f_spec["attribute"], f_spec["value"])
            events.extend(filler(rng.randint(1, 2)))
            events.append(merged)
            # skip f1 (index 4) since it's merged
        elif i == 4:
            continue  # already merged with e1
        else:
            events.extend(filler(rng.randint(1, 2)))
            events.append(node_evt(node_name))
    events.extend(filler(2))
    if _verify_tree(events, tree) == "violates":
        traces.append(Trace(tid(), ft, "edge_viol", "consecutive_nodes_same_step", events, "violates"))

    # Root and first child co-occur at same step (violates — XF requires strictly later)
    # a (animal=toucan) and b1 (animal=crane) both use animal, so can't co-occur.
    # But a (animal=toucan) and c1 (animal=hawk) also can't.
    # However, a (animal=toucan) and f1 (color=red) CAN co-occur.
    # This doesn't help since they're not consecutive in the formula.
    # So for animal-vs-animal consecutive pairs, same-step is impossible.
    # For e-level and f-level (different attributes), we already test above.

    # Cross-branch mix: events from two different paths sharing a prefix,
    # but the suffix comes from the wrong branch
    # Path 0: a, b1, c1, e1, f1, d  and  Path 2: a, b1, c1, e2, f3, d
    # Take prefix a, b1, c1 then switch to e2's child f3 but with e1
    events = (filler(2)
              + [node_evt("a")] + filler(2)
              + [node_evt("b1")] + filler(2)
              + [node_evt("c1")] + filler(2)
              + [node_evt("e1")] + filler(2)  # e1 is under c1
              + [node_evt("f3")] + filler(2)  # f3 is under e2, not e1!
              + [node_evt("d")] + filler(2))
    if _verify_tree(events, tree) == "violates":
        traces.append(Trace(tid(), ft, "edge_viol", "cross_branch_wrong_leaf", events, "violates"))

    return traces


# ─────────────────────────────────────────────
# Verifier
# ─────────────────────────────────────────────

def verify_universality(events: list[Event]) -> str:
    """G(P): P must hold at every step."""
    ft = "universality"
    for e in events:
        if not e.has_prop("P", ft):
            return "violates"
    return "satisfies"


def verify_absence(events: list[Event]) -> str:
    """G(¬P): P must not hold at any step."""
    ft = "absence"
    for e in events:
        if e.has_prop("P", ft):
            return "violates"
    return "satisfies"


def verify_response(events: list[Event]) -> str:
    """G(P → F(S)): every P must be followed by an S (at same step or later)."""
    ft = "response"
    n = len(events)
    for i in range(n):
        if events[i].has_prop("P", ft):
            # Check if S occurs at step i or later
            found_s = False
            for j in range(i, n):
                if events[j].has_prop("S", ft):
                    found_s = True
                    break
            if not found_s:
                return "violates"
    return "satisfies"


def verify_absence_between(events: list[Event]) -> str:
    """
    G((Q ∧ ¬R ∧ ◇R) → (¬P U R))

    For every step i where Q holds and R does not hold:
      if R occurs at some future step j > i:
        then P must not hold at any step from i to j-1 (inclusive of i, exclusive of j)
    """
    ft = "absence_between"
    n = len(events)

    for i in range(n):
        q_holds = events[i].has_prop("Q", ft)
        r_holds = events[i].has_prop("R", ft)

        if q_holds and not r_holds:
            # Check if R eventually occurs after step i
            r_future = None
            for j in range(i + 1, n):
                if events[j].has_prop("R", ft):
                    r_future = j
                    break

            if r_future is not None:
                # ◇R is true, so check ¬P U R:
                # P must be false from step i up to (but not including) r_future
                for k in range(i, r_future):
                    if events[k].has_prop("P", ft):
                        return "violates"
    return "satisfies"


def verify_trace(trace: Trace) -> str:
    """Dispatch to the appropriate verifier."""
    if trace.formula_type == "universality":
        return verify_universality(trace.events)
    elif trace.formula_type == "absence":
        return verify_absence(trace.events)
    elif trace.formula_type == "response":
        return verify_response(trace.events)
    elif trace.formula_type == "absence_between":
        return verify_absence_between(trace.events)
    elif trace.formula_type == "constrained_response":
        return verify_constrained_response(trace.events)
    elif trace.formula_type == "tree_b2d1":
        return _verify_tree(trace.events, TREE_B2D1)
    elif trace.formula_type == "tree_b2d4":
        return _verify_tree(trace.events, TREE_B2D4)
    else:
        raise ValueError(f"Unknown formula type: {trace.formula_type}")


def verify_constrained_response(events: list[Event]) -> str:
    """
    G(P → (¬Q U R))
    For every step where P holds:
      R must eventually occur (strong until), and
      Q must not hold at any step from P up to (but not including) the step where R holds.
    """
    ft = "constrained_response"
    n = len(events)

    for i in range(n):
        if events[i].has_prop("P", ft):
            # Find first R at step i or later
            r_found = None
            for j in range(i, n):
                if events[j].has_prop("R", ft):
                    r_found = j
                    break

            if r_found is None:
                # Strong until: R must eventually come
                return "violates"

            # Check ¬Q from step i up to (but not including) r_found
            for k in range(i, r_found):
                if events[k].has_prop("Q", ft):
                    return "violates"

    return "satisfies"


# ─────────────────────────────────────────────
# Prompt generation
# ─────────────────────────────────────────────

TRACE_PREAMBLE_INFORMAL = (
    "You are given a trace of observed events. Each event describes an "
    "animal, a shape, a color, and a number.\n\n"
)

TRACE_PREAMBLE_PRECISE = (
    "You are given a trace of observed events. Each event describes an "
    "animal, a shape, a color, and a number.\n\n"
    "A trace is a finite sequence of time steps. At each time step, "
    "zero or more propositions hold simultaneously. Propositions that "
    "hold at the same step are considered to occur at the same time, "
    "not in any order relative to each other.\n\n"
)

SPEC_PROMPTS = {
    "universality": {
        "informal": (
            "Constraint: The color is always red.\n\n"
        ),
        "precise": (
            "Constraint: At every time step in the trace, the color must be red.\n\n"
        ),
        "precise_ltl": (
            "Constraint: At every time step in the trace, the color must be red. "
            "Formally: G(P), where P = \"color is red\".\n\n"
        ),
    },
    "absence": {
        "informal": (
            "Constraint: An owl never appears.\n\n"
        ),
        "precise": (
            "Constraint: At no time step in the trace does the animal \"owl\" appear.\n\n"
        ),
        "precise_ltl": (
            "Constraint: At no time step in the trace does the animal \"owl\" appear. "
            "Formally: G(¬P), where P = \"animal is owl\".\n\n"
        ),
    },
    "response": {
        "informal": (
            "Constraint: Whenever a triangle appears, a blue item should "
            "eventually appear too.\n\n"
        ),
        "precise": (
            "Constraint: It is always the case that for every occurrence of a "
            "triangle shape, the color blue must occur at the same time step "
            "or at a later time step.\n\n"
        ),
        "precise_ltl": (
            "Constraint: It is always the case that for every occurrence of a "
            "triangle shape, the color blue must occur at the same time step "
            "or at a later time step. "
            "Formally: G(P → F(S)), where P = \"shape is triangle\" and "
            "S = \"color is blue\".\n\n"
        ),
    },
    "absence_between": {
        "informal": (
            "Constraint: A green item should not occur between a fox and a star.\n\n"
        ),
        "precise": (
            "Constraint: It is always the case that if a fox appears at a "
            "time step where a star does not appear, and a star will appear "
            "at some future time step, then the color green must not appear "
            "at any time step from that point until the star appears.\n\n"
        ),
        "precise_ltl": (
            "Constraint: It is always the case that if a fox appears at a "
            "time step where a star does not appear, and a star will appear "
            "at some future time step, then the color green must not appear "
            "at any time step from that point until the star appears. "
            "Formally: G((Q ∧ ¬R ∧ ◇R) → (¬P U R)), where Q = \"animal is fox\", "
            "R = \"shape is star\", P = \"color is green\".\n\n"
        ),
    },
    "constrained_response": {
        "informal": (
            "Constraint: Whenever a square appears, a fox should not appear "
            "until a circle appears.\n\n"
        ),
        "precise": (
            "Constraint: It is always the case that whenever a square shape "
            "appears, the animal fox must not appear at any time step from "
            "that point until the shape circle appears. For every square, "
            "the shape circle must eventually appear at that time step or "
            "at a later time step.\n\n"
        ),
        "precise_ltl": (
            "Constraint: It is always the case that whenever a square shape "
            "appears, the animal fox must not appear at any time step from "
            "that point until the shape circle appears. For every square, "
            "the shape circle must eventually appear at that time step or "
            "at a later time step. "
            "Formally: G(P → (¬Q U R)), where P = \"shape is square\", "
            "Q = \"animal is fox\", R = \"shape is circle\".\n\n"
        ),
    },
    "tree_b2d1": {
        "informal": (
            "Constraint: At some point a toucan should appear, followed by "
            "either a crane or a pelican, and then a deer.\n\n"
        ),
        "precise": (
            "Constraint: At some time step, a toucan must appear, and then "
            "at some strictly later time step, either:\n"
            "  (a crane appears, and then at some strictly later time step, a deer appears)\n"
            "  OR\n"
            "  (a pelican appears, and then at some strictly later time step, a deer appears)\n"
            "The events do not need to be at consecutive time steps — "
            "other events may occur in between.\n\n"
        ),
        "precise_ltl": (
            "Constraint: At some time step, a toucan must appear, and then "
            "at some strictly later time step, either:\n"
            "  (a crane appears, and then at some strictly later time step, a deer appears)\n"
            "  OR\n"
            "  (a pelican appears, and then at some strictly later time step, a deer appears)\n"
            "The events do not need to be at consecutive time steps — "
            "other events may occur in between. "
            "Formally: F(a ∧ XF((b1 ∧ XF(d)) ∨ (b2 ∧ XF(d)))), where "
            "a = \"animal is toucan\", b1 = \"animal is crane\", "
            "b2 = \"animal is pelican\", d = \"animal is deer\".\n\n"
        ),
    },
    "tree_b2d4": {
        "informal": (
            "Constraint: At some point a toucan should appear, followed by either "
            "a crane or a pelican. If a crane, then either a hawk or a parrot. "
            "If a hawk, then either a sparrow or a falcon. If a sparrow, then "
            "either a red or blue item. If a falcon, then either a green or yellow "
            "item. If a parrot, then either a heron or a dove. If a heron, then "
            "either a purple or orange item. If a dove, then either a pink or black "
            "item. If a pelican, then either an eagle or a robin. If an eagle, then "
            "either a raven or a finch. If a raven, then either a white or brown "
            "item. If a finch, then either item number 1 or 2. If a robin, then "
            "either a jay or an oval. If a jay, then either item number 3 or 4. "
            "If an oval, then either item number 5 or 6. Everything ends with "
            "a deer.\n\n"
        ),
        "precise": (
            "Constraint: At some time step, a toucan must appear, and then "
            "at some strictly later time step, either:\n"
            "  (a crane appears, and then at some strictly later time step, either:\n"
            "    (a hawk appears, and then at some strictly later time step, either:\n"
            "      (a sparrow appears, and then at some strictly later time step, either:\n"
            "        (a red item appears, and then at some strictly later time step, a deer appears)\n"
            "        OR\n"
            "        (a blue item appears, and then at some strictly later time step, a deer appears))\n"
            "      OR\n"
            "      (a falcon appears, and then at some strictly later time step, either:\n"
            "        (a green item appears, and then at some strictly later time step, a deer appears)\n"
            "        OR\n"
            "        (a yellow item appears, and then at some strictly later time step, a deer appears)))\n"
            "    OR\n"
            "    (a parrot appears, and then at some strictly later time step, either:\n"
            "      (a heron appears, and then at some strictly later time step, either:\n"
            "        (a purple item appears, and then at some strictly later time step, a deer appears)\n"
            "        OR\n"
            "        (an orange item appears, and then at some strictly later time step, a deer appears))\n"
            "      OR\n"
            "      (a dove appears, and then at some strictly later time step, either:\n"
            "        (a pink item appears, and then at some strictly later time step, a deer appears)\n"
            "        OR\n"
            "        (a black item appears, and then at some strictly later time step, a deer appears))))\n"
            "  OR\n"
            "  (a pelican appears, and then at some strictly later time step, either:\n"
            "    (an eagle appears, and then at some strictly later time step, either:\n"
            "      (a raven appears, and then at some strictly later time step, either:\n"
            "        (a white item appears, and then at some strictly later time step, a deer appears)\n"
            "        OR\n"
            "        (a brown item appears, and then at some strictly later time step, a deer appears))\n"
            "      OR\n"
            "      (a finch appears, and then at some strictly later time step, either:\n"
            "        (item number 1 appears, and then at some strictly later time step, a deer appears)\n"
            "        OR\n"
            "        (item number 2 appears, and then at some strictly later time step, a deer appears)))\n"
            "    OR\n"
            "    (a robin appears, and then at some strictly later time step, either:\n"
            "      (a jay appears, and then at some strictly later time step, either:\n"
            "        (item number 3 appears, and then at some strictly later time step, a deer appears)\n"
            "        OR\n"
            "        (item number 4 appears, and then at some strictly later time step, a deer appears))\n"
            "      OR\n"
            "      (an oval appears, and then at some strictly later time step, either:\n"
            "        (item number 5 appears, and then at some strictly later time step, a deer appears)\n"
            "        OR\n"
            "        (item number 6 appears, and then at some strictly later time step, a deer appears))))\n"
            "The events do not need to be at consecutive time steps — "
            "other events may occur in between.\n\n"
        ),
        "precise_ltl": (
            "Constraint: At some time step, a toucan must appear, and then "
            "at some strictly later time step, either:\n"
            "  (a crane appears, and then at some strictly later time step, either:\n"
            "    (a hawk appears, and then at some strictly later time step, either:\n"
            "      (a sparrow appears, and then at some strictly later time step, either:\n"
            "        (a red item appears, and then at some strictly later time step, a deer appears)\n"
            "        OR\n"
            "        (a blue item appears, and then at some strictly later time step, a deer appears))\n"
            "      OR\n"
            "      (a falcon appears, and then at some strictly later time step, either:\n"
            "        (a green item appears, and then at some strictly later time step, a deer appears)\n"
            "        OR\n"
            "        (a yellow item appears, and then at some strictly later time step, a deer appears)))\n"
            "    OR\n"
            "    (a parrot appears, and then at some strictly later time step, either:\n"
            "      (a heron appears, and then at some strictly later time step, either:\n"
            "        (a purple item appears, and then at some strictly later time step, a deer appears)\n"
            "        OR\n"
            "        (an orange item appears, and then at some strictly later time step, a deer appears))\n"
            "      OR\n"
            "      (a dove appears, and then at some strictly later time step, either:\n"
            "        (a pink item appears, and then at some strictly later time step, a deer appears)\n"
            "        OR\n"
            "        (a black item appears, and then at some strictly later time step, a deer appears))))\n"
            "  OR\n"
            "  (a pelican appears, and then at some strictly later time step, either:\n"
            "    (an eagle appears, and then at some strictly later time step, either:\n"
            "      (a raven appears, and then at some strictly later time step, either:\n"
            "        (a white item appears, and then at some strictly later time step, a deer appears)\n"
            "        OR\n"
            "        (a brown item appears, and then at some strictly later time step, a deer appears))\n"
            "      OR\n"
            "      (a finch appears, and then at some strictly later time step, either:\n"
            "        (item number 1 appears, and then at some strictly later time step, a deer appears)\n"
            "        OR\n"
            "        (item number 2 appears, and then at some strictly later time step, a deer appears)))\n"
            "    OR\n"
            "    (a robin appears, and then at some strictly later time step, either:\n"
            "      (a jay appears, and then at some strictly later time step, either:\n"
            "        (item number 3 appears, and then at some strictly later time step, a deer appears)\n"
            "        OR\n"
            "        (item number 4 appears, and then at some strictly later time step, a deer appears))\n"
            "      OR\n"
            "      (an oval appears, and then at some strictly later time step, either:\n"
            "        (item number 5 appears, and then at some strictly later time step, a deer appears)\n"
            "        OR\n"
            "        (item number 6 appears, and then at some strictly later time step, a deer appears))))\n"
            "The events do not need to be at consecutive time steps — "
            "other events may occur in between.\n\n"
            "Formally: F(a ∧ XF(\n"
            "  (b1 ∧ XF(\n"
            "    (c1 ∧ XF(\n"
            "      (e1 ∧ XF( (f1 ∧ XF(d)) ∨ (f2 ∧ XF(d)) ))\n"
            "      ∨ (e2 ∧ XF( (f3 ∧ XF(d)) ∨ (f4 ∧ XF(d)) ))))\n"
            "    ∨ (c2 ∧ XF(\n"
            "      (e3 ∧ XF( (f5 ∧ XF(d)) ∨ (f6 ∧ XF(d)) ))\n"
            "      ∨ (e4 ∧ XF( (f7 ∧ XF(d)) ∨ (f8 ∧ XF(d)) ))))))\n"
            "  ∨ (b2 ∧ XF(\n"
            "    (c3 ∧ XF(\n"
            "      (e5 ∧ XF( (f9 ∧ XF(d)) ∨ (f10 ∧ XF(d)) ))\n"
            "      ∨ (e6 ∧ XF( (f11 ∧ XF(d)) ∨ (f12 ∧ XF(d)) ))))\n"
            "    ∨ (c4 ∧ XF(\n"
            "      (e7 ∧ XF( (f13 ∧ XF(d)) ∨ (f14 ∧ XF(d)) ))\n"
            "      ∨ (e8 ∧ XF( (f15 ∧ XF(d)) ∨ (f16 ∧ XF(d)) ))))))))\n\n"
        ),
    },
}

TASK_SUFFIX = (
    "Determine whether the following trace satisfies or violates the constraint. "
    "Respond with SATISFIES or VIOLATES followed by a brief explanation.\n\n"
)


def format_trace_for_prompt(trace: Trace) -> str:
    """Format trace events as numbered steps."""
    lines = []
    for t, event in enumerate(trace.events):
        lines.append(f"Step {t + 1}: {event.to_nl()}")
    if not lines:
        return "Trace: (empty trace — no events)\n"
    return "Trace:\n" + "\n".join(lines) + "\n"


LTL_LEGEND = (
    "The constraint is also expressed as a Linear Temporal Logic (LTL) formula. "
    "LTL operators: G (always), F (eventually), X (next time step), "
    "U (until — left side holds until right side becomes true), "
    "¬ (not), ∧ (and), ∨ (or), → (implies).\n\n"
)


def build_prompt(trace: Trace, spec_level: str) -> str:
    """
    Build the full prompt for a given trace and specification level.

    spec_level: "informal", "precise", or "precise_ltl"
    """
    if spec_level == "informal":
        preamble = TRACE_PREAMBLE_INFORMAL
    else:
        preamble = TRACE_PREAMBLE_PRECISE

    spec = SPEC_PROMPTS[trace.formula_type][spec_level]
    trace_str = format_trace_for_prompt(trace)

    if spec_level == "precise_ltl":
        return preamble + LTL_LEGEND + spec + TASK_SUFFIX + trace_str
    else:
        return preamble + spec + TASK_SUFFIX + trace_str


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Spec-level experiment trace generator")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="One or more seeds. All traces accumulate into one output file.")
    parser.add_argument("--output", type=str, default="x")
    parser.add_argument("--verify", action="store_true",
                        help="Run verifier on all traces and report mismatches")
    args = parser.parse_args()

    ALL_FORMULA_TYPES = [
        "universality", "absence", "response", "absence_between",
        "constrained_response", "tree_b2d1", "tree_b2d4",
    ]

    all_traces = []
    all_mismatches = []

    for seed in args.seeds:
        rng = random.Random(seed)

        # Generate traces for this seed
        seed_traces = []
        seed_traces.extend(generate_universality_traces(rng))
        seed_traces.extend(generate_absence_traces(rng))
        seed_traces.extend(generate_response_traces(rng))
        seed_traces.extend(generate_absence_between_traces(rng))
        seed_traces.extend(generate_constrained_response_traces(rng))
        seed_traces.extend(generate_tree_b2d1_traces(rng))
        seed_traces.extend(generate_tree_b2d4_traces(rng))

        # Prefix trace IDs with seed to avoid collisions
        for t in seed_traces:
            t.trace_id = f"s{seed}_{t.trace_id}"

        # Pad all traces to target length
        pad_rng = random.Random(seed + 10000)  # separate rng to not disturb generation
        for t in seed_traces:
            pad_trace_to_length(t, pad_rng)

        # Verify
        for trace in seed_traces:
            computed = verify_trace(trace)
            if computed != trace.ground_truth:
                all_mismatches.append({
                    "trace_id": trace.trace_id,
                    "seed": seed,
                    "formula_type": trace.formula_type,
                    "edge_case_tag": trace.edge_case_tag,
                    "expected": trace.ground_truth,
                    "computed": computed,
                })

        all_traces.extend(seed_traces)
        print(f"Seed {seed}: generated {len(seed_traces)} traces")

    # Verification report
    if args.verify or all_mismatches:
        print(f"\n=== Verification Report ===")
        print(f"Total traces: {len(all_traces)}")
        print(f"Mismatches:   {len(all_mismatches)}")
        if all_mismatches:
            print("\nMISMATCHES:")
            for m in all_mismatches:
                print(f"  {m['trace_id']} ({m['formula_type']}, "
                      f"edge={m['edge_case_tag']}): "
                      f"expected={m['expected']}, computed={m['computed']}")
        else:
            print("All traces verified correctly!")

    # Summary
    print(f"\n=== Trace Summary ===")
    print(f"Seeds: {args.seeds}")
    for ft in ALL_FORMULA_TYPES:
        ft_traces = [t for t in all_traces if t.formula_type == ft]
        n_sat = sum(1 for t in ft_traces if t.ground_truth == "satisfies")
        n_viol = sum(1 for t in ft_traces if t.ground_truth == "violates")
        n_edge = sum(1 for t in ft_traces if t.trace_category.startswith("edge"))
        print(f"  {ft:25s}: {len(ft_traces):3d} traces "
              f"({n_sat} sat, {n_viol} viol, {n_edge} edge cases)")

    # Save
    output = {
        "metadata": {
            "seeds": args.seeds,
            "total_traces": len(all_traces),
            "formula_types": ALL_FORMULA_TYPES,
            "spec_levels": ["informal", "precise", "precise_ltl"],
            "proposition_groundings": PROP_GROUNDINGS,
        },
        "traces": [t.to_dict() for t in all_traces],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(all_traces)} traces to {args.output}")

    # Also generate example prompts for inspection
    example_prompts = {}
    for ft in ALL_FORMULA_TYPES:
        ft_traces = [t for t in all_traces if t.formula_type == ft]
        if ft_traces:
            example = ft_traces[0]
            example_prompts[ft] = {
                level: build_prompt(example, level)
                for level in ["informal", "precise", "precise_ltl"]
            }

    with open("example_prompts.json", "w") as f:
        json.dump(example_prompts, f, indent=2)
    print("Saved example prompts to example_prompts.json")


if __name__ == "__main__":
    main()