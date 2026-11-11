"""A first, looser question for the bounded checker: the same formula with its hard arithmetic left uninterpreted.

Two LeetCode answers to one problem usually compute the same thing by another route:
``ROUND(SUM(price * units) / SUM(units), 2)`` over a join written two ways. To the solver the product of two
unknowns, a division by an unknown and the rounding inside ``ROUND`` (a real number to an integer) are the
expensive parts, and it spends its time limit on them although the two queries apply them to the same values.

:func:`abstract` rewrites a formula so that each such operation is an *uninterpreted function* of its operands:

* a product of two or more terms that are not numerals (``x * 3`` stays, it is linear);
* ``/``, integer division and modulo by a term that is not a numeral;
* real-to-integer conversion (the ``ToInt`` inside ``ROUND``).

The real operations are one meaning those functions can take, so every database on which the queries differ still
satisfies the rewritten formula: **unsatisfiable** carries over, and is usually found in a fraction of the time
because equal operands give equal results without any arithmetic. **Satisfiable** does not carry over (the solver
may have chosen a meaning under which 2 * 3 is 7), so the checker treats such a model as a candidate only: its
tables are a concrete database, the replay runs both queries on it, and a difference there is a counterexample like
any other. A model the replay does not confirm, or a timeout, leads to the exact question as before.
"""

from __future__ import annotations

try:
    import z3
except ImportError:  # pragma: no cover - the ``smt`` extra
    z3 = None


def _numeral(term) -> bool:
    return z3.is_int_value(term) or z3.is_rational_value(term)


class _Functions:
    def __init__(self):
        self.made: dict[str, object] = {}

    def get(self, name: str, *sorts):
        key = name + "|" + "|".join(sort.sexpr() for sort in sorts)
        if key not in self.made:
            self.made[key] = z3.Function(f"{name}!{len(self.made)}", *sorts)
        return self.made[key]


def _rewritten(term, kids: list, functions: _Functions):
    """``term`` over its rewritten operands ``kids``: an uninterpreted application for a hard operation, else ``None``."""

    kind = term.decl().kind()
    if kind == z3.Z3_OP_MUL:
        numerals = [k for k in kids if _numeral(k)]
        others = sorted((k for k in kids if not _numeral(k)), key=lambda k: k.get_id())  # a * b and b * a are one term
        if len(others) < 2:
            return None
        product = others[0]
        times = functions.get("times", term.sort(), term.sort(), term.sort())
        for other in others[1:]:
            product = times(product, other)
        for numeral in numerals:
            product = numeral * product
        return product
    if kind in (z3.Z3_OP_DIV, z3.Z3_OP_IDIV, z3.Z3_OP_MOD) and len(kids) == 2 and not _numeral(kids[1]):
        name = {z3.Z3_OP_DIV: "over", z3.Z3_OP_IDIV: "div", z3.Z3_OP_MOD: "mod"}[kind]
        return functions.get(name, kids[0].sort(), kids[1].sort(), term.sort())(kids[0], kids[1])
    if kind == z3.Z3_OP_TO_INT and len(kids) == 1:
        return functions.get("whole", kids[0].sort(), term.sort())(kids[0])
    return None


def rewrite(formulas: list, step) -> tuple[list, int]:
    """``formulas`` rebuilt bottom-up, with ``step(term, rewritten operands)`` replacing a term when it is not ``None``.

    Returns the new list and how many terms were replaced; with none replaced the formulas are returned as they
    were. Shared subterms are rewritten once, and the walk does not recurse (an aggregate over many rows nests deeply).
    """

    done: dict[int, object] = {}
    replaced = 0
    for formula in formulas:
        stack = [(formula, False)]
        while stack:
            term, expanded = stack.pop()
            key = term.get_id()
            if key in done:
                continue
            if not z3.is_app(term) or term.num_args() == 0:
                done[key] = term
                continue
            children = term.children()
            if not expanded:
                stack.append((term, True))
                stack.extend((child, False) for child in children if child.get_id() not in done)
                continue
            kids = [done[child.get_id()] for child in children]
            new = step(term, kids)
            if new is not None:
                replaced += 1
                done[key] = new
            elif all(kid is child or kid.eq(child) for kid, child in zip(kids, children)):
                done[key] = term
            else:
                done[key] = term.decl()(*kids)
    if not replaced:
        return list(formulas), 0
    return [done[formula.get_id()] for formula in formulas], replaced


def abstract(formulas: list) -> tuple[list, int]:
    """``formulas`` with every hard arithmetic operation uninterpreted, and how many operations that was.

    One set of functions serves the whole list, so the same operation on the same operands is the same term wherever
    it occurs. With a count of 0 the formulas are returned as they were.
    """

    functions = _Functions()
    return rewrite(formulas, lambda term, kids: _rewritten(term, kids, functions))
