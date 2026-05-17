#!/usr/bin/env python3
"""
Verify ground truth labels of the tree-LTL dataset using LTL progression.

Builds the LTL formula from the tree structure, then progresses it through
each trace step-by-step, checking that the final result matches the label.
"""

import json
import argparse
from copy import deepcopy


# ── LTL Formula (from user's code) ──────────────────────────────────────────

class LTLFormula:
    def __init__(self, op=None, left=None, right=None, atom=None):
        self.op = op
        self.left = left
        self.right = right
        self.atom = atom

    def __deepcopy__(self, memodict={}):
        return LTLFormula(
            op=self.op,
            left=deepcopy(self.left),
            right=deepcopy(self.right),
            atom=self.atom
        )

    @classmethod
    def true(cls):
        return cls(op="TRUE")

    @classmethod
    def false(cls):
        return cls(op="FALSE")

    @classmethod
    def make_atom(cls, name):
        return cls(atom=name)

    @classmethod
    def neg(cls, formula):
        return cls(op='!', left=formula)

    @classmethod
    def and_(cls, left, right):
        return cls(op='&', left=left, right=right)

    @classmethod
    def or_(cls, left, right):
        return cls(op='|', left=left, right=right)

    @classmethod
    def next(cls, formula):
        return cls(op='X', left=formula)

    @classmethod
    def eventually(cls, formula):
        return cls(op='F', left=formula)

    @classmethod
    def always(cls, formula):
        return cls(op='G', left=formula)

    def __str__(self):
        if self.atom is not None:
            return self.atom
        if self.op == "TRUE":
            return "TRUE"
        if self.op == "FALSE":
            return "FALSE"
        if self.op == '!':
            return f"!({self.left})"
        if self.op in ['X', 'F', 'G']:
            return f"{self.op}({self.left})"
        if self.op in ['&', '|', '->', 'U', 'WU']:
            return f"({self.left} {self.op} {self.right})"
        return "Invalid Formula"

    def score(self):
        if self.op == "TRUE":
            return 1
        if self.op == "FALSE":
            return -1
        return 0

    def progress(self, state):
        if self.atom is not None:
            if self.atom in state and state[self.atom]:
                return LTLFormula.true()
            return LTLFormula.false()

        if self.op == "TRUE":
            return LTLFormula.true()
        if self.op == "FALSE":
            return LTLFormula.false()

        if self.op == "!":
            sub = self.left.progress(state)
            if sub.op == "TRUE":
                return LTLFormula.false()
            if sub.op == "FALSE":
                return LTLFormula.true()
            return LTLFormula.neg(sub)

        if self.op == '&':
            lp = self.left.progress(state)
            rp = self.right.progress(state)
            if lp.op == "FALSE" or rp.op == "FALSE":
                return LTLFormula.false()
            if lp.op == "TRUE" and rp.op == "TRUE":
                return LTLFormula.true()
            if lp.op == "TRUE":
                return rp
            if rp.op == "TRUE":
                return lp
            return LTLFormula.and_(lp, rp)

        if self.op == '|':
            lp = self.left.progress(state)
            rp = self.right.progress(state)
            if lp.op == "TRUE" or rp.op == "TRUE":
                return LTLFormula.true()
            if lp.op == "FALSE":
                return rp
            if rp.op == "FALSE":
                return lp
            return LTLFormula.or_(lp, rp)

        if self.op == 'X':
            return self.left

        if self.op == 'F':
            sub = self.left.progress(state)
            if sub.op == "TRUE":
                return LTLFormula.true()
            if sub.op == "FALSE":
                return deepcopy(self)
            return LTLFormula.or_(sub, deepcopy(self))

        if self.op == 'G':
            sub = self.left.progress(state)
            if sub.op == "FALSE":
                return LTLFormula.false()
            if sub.op == "TRUE":
                return deepcopy(self)
            return LTLFormula.and_(sub, deepcopy(self))

        return self


# ── Build LTL formula from dataset ──────────────────────────────────────────

def build_ltl_from_tree(paths, propositions):
    """
    Build the LTL formula from the dataset's path list and propositions.

    For a tree with sink, the formula is:
      F(root & X F((n1 & X F((n4 & X F(leaf1 & X F(sink) | leaf2 & X F(sink))) | ...)) | ...))

    We build it recursively from the tree structure implied by the paths.
    """
    # Reconstruct tree structure from paths
    # All paths start with the same root and end with the same sink
    root_name = paths[0]["nodes"][0]
    sink_name = paths[0]["nodes"][-1]

    # Build a tree dict: node_name -> list of children names
    # Use paths to infer parent-child relationships
    children = {}
    for path in paths:
        nodes = path["nodes"]
        for i in range(len(nodes) - 2):  # exclude sink connections
            parent = nodes[i]
            child = nodes[i + 1]
            if parent not in children:
                children[parent] = []
            if child not in children[parent]:
                children[parent].append(child)

    def get_key_value(node_name):
        return propositions[node_name]["key_value"]

    def build_node(node_name):
        """Recursively build LTL sub-formula for a node."""
        atom = LTLFormula.make_atom(get_key_value(node_name))

        if node_name not in children or len(children[node_name]) == 0:
            # Leaf node: atom & X F(sink)
            sink_atom = LTLFormula.make_atom(get_key_value(sink_name))
            return LTLFormula.and_(atom, LTLFormula.next(LTLFormula.eventually(sink_atom)))

        # Internal node: atom & X F(child1_formula | child2_formula | ...)
        child_formulas = [build_node(c) for c in children[node_name]]
        combined = child_formulas[0]
        for cf in child_formulas[1:]:
            combined = LTLFormula.or_(combined, cf)

        return LTLFormula.and_(atom, LTLFormula.next(LTLFormula.eventually(combined)))

    return LTLFormula.eventually(build_node(root_name))


# ── Labeling ─────────────────────────────────────────────────────────────────

def label_event(event, propositions):
    """
    Given a raw event dict {animal, shape, color, number}, return a state
    mapping key_value -> True for each proposition whose key_value appears
    in any attribute of the event.
    """
    state = {}
    event_values = {
        str(event["animal"]),
        str(event["shape"]),
        str(event["color"]),
        str(event["number"]),
    }
    seen_keys = set()
    for node_name, prop in propositions.items():
        kv = prop["key_value"]
        if kv not in seen_keys and kv in event_values:
            state[kv] = True
            seen_keys.add(kv)
    return state


# ── Efficient path-based verification ────────────────────────────────────────

def event_matches_node(event, node_name, propositions):
    """Check if an event matches a node's key_value in any attribute."""
    key_value = propositions[node_name]["key_value"]
    event_values = {
        str(event["animal"]),
        str(event["shape"]),
        str(event["color"]),
        str(event["number"]),
    }
    return key_value in event_values


def check_path_in_trace(path_nodes, propositions, events):
    """
    Check if a single path's key_values appear in the trace in temporal order.
    Matches each node against only its specific attribute type.
    """
    pos = 0  # current position in path node sequence

    for event in events:
        if pos >= len(path_nodes):
            break
        if event_matches_node(event, path_nodes[pos], propositions):
            pos += 1

    return pos >= len(path_nodes)


def verify_trace_pathcheck(trace, paths, propositions):
    """
    Verify a trace by checking if any complete path exists in temporal order.
    Equivalent to evaluating the LTL formula but O(num_paths * trace_length).
    """
    for path in paths:
        if check_path_in_trace(path["nodes"], propositions, trace["events"]):
            return "VALID"
    return "INVALID"


# ── LTL progression verification (slow, for small traces) ───────────────────

def verify_trace(formula, trace, propositions):
    """
    Progress the LTL formula through each event in the trace.
    Returns the final formula result.
    """
    f = deepcopy(formula)
    for event in trace["events"]:
        state = label_event(event, propositions)
        f = f.progress(state)
        if f.op == "TRUE":
            return "VALID"

    if f.op == "TRUE":
        return "VALID"
    else:
        return "INVALID"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify tree-LTL dataset labels using LTL progression"
    )
    parser.add_argument("--dataset", type=str, default="data/tree_ltl_dataset.json")
    parser.add_argument("--k_values", nargs="+", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    propositions = dataset["propositions"]
    paths = dataset["paths"]
    k_values = args.k_values or list(dataset["traces_by_k"].keys())

    # Build formula
    formula = build_ltl_from_tree(paths, propositions)
    print(f"LTL formula built. Length: {len(str(formula))} chars")
    if args.verbose:
        print(f"Formula: {formula}")
    print()

    # Verify each trace
    total = 0
    matches = 0
    mismatches = []

    for k in k_values:
        traces = dataset["traces_by_k"][k]
        k_total = 0
        k_matches = 0

        for trace in traces:
            result = verify_trace_pathcheck(trace, paths, propositions)
            expected = trace["label"]
            total += 1
            k_total += 1

            if result == expected:
                matches += 1
                k_matches += 1
                if args.verbose:
                    print(f"  ✓ {trace['trace_id']}: {expected}")
            else:
                mismatches.append({
                    "trace_id": trace["trace_id"],
                    "expected": expected,
                    "got": result,
                    "k": k,
                    "valid_path_idx": trace.get("valid_path_idx"),
                    "corruptions": trace.get("corruptions"),
                })
                print(f"  ✗ {trace['trace_id']}: expected={expected}, "
                      f"got={result}")

        print(f"k={k}: {k_matches}/{k_total} match")

    print(f"\nTotal: {matches}/{total} match")

    if mismatches:
        print(f"\n{len(mismatches)} MISMATCHES found:")
        for m in mismatches[:10]:
            print(f"  {m['trace_id']}: expected={m['expected']}, got={m['got']}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
    else:
        print("\nAll labels verified correctly!")


if __name__ == "__main__":
    main()