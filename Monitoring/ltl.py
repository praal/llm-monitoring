from copy import deepcopy


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
        """Create a True constant."""
        return cls(op="TRUE")

    @classmethod
    def false(cls):
        """Create a False constant."""
        return cls(op="FALSE")
    @classmethod
    def atom(cls, name):
        """Create an atomic proposition."""
        return cls(atom=name)

    @classmethod
    def neg(cls, formula):
        """Create a negation."""
        return cls(op='!', left=formula)

    @classmethod
    def and_(cls, left, right):
        """Create a conjunction."""
        return cls(op='&', left=left, right=right)

    @classmethod
    def or_(cls, left, right):
        """Create a disjunction."""
        return cls(op='|', left=left, right=right)

    @classmethod
    def implies(cls, left, right):
        """Create an implication."""
        return cls(op='|', left=cls(op="!", left=left), right=right)

    @classmethod
    def next(cls, formula):
        """Create a 'next' formula."""
        return cls(op='X', left=formula)

    @classmethod
    def eventually(cls, formula):
        """Create an 'eventually' formula."""
        return cls(op='F', left=formula)

    @classmethod
    def always(cls, formula):
        """Create an 'always' formula."""
        return cls(op='G', left=formula)

    @classmethod
    def until(cls, left, right):
        """Create an 'until' formula."""
        return cls(op='U', left=left, right=right)

    @classmethod
    def weak_until(cls, left, right):
        """Create a 'weak until' formula."""
        return cls(op='WUَ', left=left, right=right)

    @classmethod
    def eventually_atmost(self, formula, k):
        if k == 1:
            return LTLFormula.or_(formula, LTLFormula.next(formula))
        else:
            return LTLFormula.or_(formula, LTLFormula.next(self.eventually_atmost(formula, k - 1)))

    def pretty_print(self):
        if self.atom is not None:
            return self.atom

        if self.op == "TRUE":
            return "TRUE"

        if self.op == "FALSE":
            return "FALSE"

        if self.op == '!':
            return "!" + self.left.pretty_print()

        if self.op in ['X', 'F', 'G']:
            return "(" + self.op + " " + self.left.pretty_print() + ")"

        if self.op in ['&', '|', '->', 'U', 'WU']:
            return "(" + self.left.pretty_print() + " " + self.op + " " + self.right.pretty_print()+ ")"

        return "Invalid Formula"

    def __str__(self):
        """String representation of the formula."""
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
        """
            Progress the LTL formula based on the observed state.

            Parameters:
            - state: A dictionary mapping atomic propositions to boolean values
                        (True if the proposition holds in the current state)

            Returns:
            - A new LTLFormula representing the progressed formula
        """
        if self.atom is not None:
            if self.atom in state and state[self.atom]:
                return LTLFormula.true()
            if self.atom not in state:
                return LTLFormula.false()
            if self.atom in state and state[self.atom] == False:
                return LTLFormula.false()

        if self.op == "TRUE":
            return LTLFormula.true()
        if self.op == "FALSE":
            return LTLFormula.false()

        if self.op == "!":
            sub_progression = self.left.progress(state)
            if sub_progression.op == "TRUE":
                return LTLFormula.false()
            if sub_progression.op == "FALSE":
                return LTLFormula.true()
            return LTLFormula.neg(sub_progression)

        if self.op == '&':
            left_progression = self.left.progress(state)
            right_progression = self.right.progress(state)
            if left_progression.op == "FALSE" or right_progression.op == "FALSE":
                return LTLFormula.false()
            if left_progression.op == "TRUE" and right_progression.op == "TRUE":
                return LTLFormula.true()
            if left_progression.op == "TRUE":
                return right_progression
            if right_progression.op == "TRUE":
                return left_progression
            return LTLFormula.and_(left_progression, right_progression)

        if self.op == '|':
            left_progression = self.left.progress(state)
            right_progression = self.right.progress(state)
            if left_progression.op == "TRUE" or right_progression.op == "TRUE":
                return LTLFormula.true()
            if left_progression.op == "FALSE":
                return right_progression
            if right_progression.op == "FALSE":
                return left_progression

            return LTLFormula.or_(left_progression, right_progression)

        if self.op == 'X':
            return self.left

        if self.op == 'U':
            right_progression = self.right.progress(state)
            # If right part holds now, the until is satisfied
            if right_progression.op == "TRUE":
                return LTLFormula.true()

            left_progression = self.left.progress(state)
            if left_progression.op == "FALSE":
                return right_progression

            if left_progression.op == "TRUE":
                return LTLFormula.or_(
                right_progression,
                deepcopy(self)
            )

            return LTLFormula.or_(
                right_progression,
                LTLFormula.and_(left_progression, deepcopy(self))
            )

        # Handle Eventually
        if self.op == 'F':
            sub_progression = self.left.progress(state)
            # If subformula holds now, the eventually is satisfied
            if sub_progression.op == "TRUE":
                return LTLFormula.true()
            if sub_progression.op == "FALSE":
                return deepcopy(self)
            # Otherwise, it must hold eventually in the future
            return LTLFormula.or_(
                sub_progression,
                deepcopy(self)
            )

            # Handle Always
        if self.op == 'G':
            sub_progression = self.left.progress(state)
            # If subformula doesn't hold now, the always is violated
            if sub_progression.op == "FALSE":
                return LTLFormula.false()

            if sub_progression.op == "TRUE":
                return deepcopy(self)

            return LTLFormula.and_(
                sub_progression,
                deepcopy(self)
            )

        return self

def labeling(propositions, observation):
    state = dict()
    observation = observation.lower()
    for p in propositions:
        words = p.split(" ")
        all_true = True
        for w in words:
            if w not in observation:
                all_true = False
        if all_true:
            state[p] = True
    return state
