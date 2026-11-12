"""Division by a row count, case by case: an exact rewrite that keeps the bounded checker's arithmetic linear.

``AVG(x)`` is a sum divided by a count, and LeetCode answers divide by ``COUNT(*)`` everywhere
(``ROUND(100 * SUM(...) / COUNT(*), 2)``). To the solver a division by a term that is not a constant is nonlinear,
the expensive kind, although under the bound a count is one of a handful of numbers: with at most three rows per
table it is 0, 1, 2 or 3.

:func:`split` finds each division (``/``, integer division, modulo) whose divisor can only take a few known values
and writes it as a choice between divisions by constants::

    x / n   becomes   if n = 1 then x / 1 else if n = 2 then x / 2 else if n = 3 then x / 3 else x / 0

Each branch is the original division under the condition that selects it, so the formula has exactly the models it
had: nothing is abstracted and nothing is assumed. The last branch is only reached when the divisor is 0 and is the
solver's own ``x / 0``, the same value the original had there.

The values a divisor can take are read from its shape and from nothing else: a numeral is itself, ``if c then a
else b`` is one of the values of ``a`` or ``b``, a sum is a sum of one value from each term, and a numeral times a
term scales its values. Anything else (a column, a ``SUM`` of a column) has no known values and is left alone.
"""

from __future__ import annotations

from fractions import Fraction

from . import bounded_abstraction

try:
    import z3
except ImportError:  # pragma: no cover - the ``smt`` extra
    z3 = None

MAX_VALUES = 16


def _numeral(term) -> Fraction | None:
    if z3.is_int_value(term):
        return Fraction(term.as_long())
    if z3.is_rational_value(term):
        return Fraction(term.numerator_as_long(), term.denominator_as_long())
    return None


def values(term, known: dict | None = None) -> frozenset | None:
    """Every value ``term`` can take, when its shape shows there are at most ``MAX_VALUES`` of them; else ``None``."""

    known = {} if known is None else known
    key = term.get_id()
    if key in known:
        return known[key]
    known[key] = None  # a term is never its own operand; this only guards the lookup below
    number = _numeral(term)
    out = None
    if number is not None:
        out = frozenset([number])
    elif z3.is_app(term):
        kind, kids = term.decl().kind(), term.children()
        if kind == z3.Z3_OP_TO_REAL:
            out = values(kids[0], known)
        elif kind == z3.Z3_OP_ITE:
            then, other = values(kids[1], known), values(kids[2], known)
            out = then | other if then is not None and other is not None else None
        elif kind in (z3.Z3_OP_ADD, z3.Z3_OP_MUL) and len(kids) <= 4 * MAX_VALUES:
            sets = [values(kid, known) for kid in kids]
            if None not in sets and (kind == z3.Z3_OP_ADD or sum(len(s) > 1 for s in sets) <= 1):
                out = frozenset([Fraction(0 if kind == z3.Z3_OP_ADD else 1)])
                for one in sets:
                    out = frozenset(a + b if kind == z3.Z3_OP_ADD else a * b for a in out for b in one)
                    if len(out) > MAX_VALUES:
                        out = None
                        break
        if out is not None and len(out) > MAX_VALUES:
            out = None
    known[key] = out
    return out


def _constant(value: Fraction, sort):
    return z3.IntVal(int(value)) if sort.kind() == z3.Z3_INT_SORT else z3.RealVal(str(value))


def split(formulas: list) -> tuple[list, int]:
    """``formulas`` with every division by a few-valued divisor written as a choice of divisions by constants."""

    known: dict = {}

    def step(term, kids):
        kind = term.decl().kind()
        if kind not in (z3.Z3_OP_DIV, z3.Z3_OP_IDIV, z3.Z3_OP_MOD) or len(kids) != 2 or _numeral(kids[1]) is not None:
            return None
        divisor = term.arg(1)  # the divisor as written: its values are read from the original term
        can = values(divisor, known)
        if can is None:
            return None
        if divisor.sort().kind() == z3.Z3_INT_SORT and any(v.denominator != 1 for v in can):
            return None
        apply = term.decl()
        out = apply(kids[0], _constant(Fraction(0), divisor.sort()))
        for value in sorted(can, reverse=True):
            if value != 0:
                constant = _constant(value, divisor.sort())
                out = z3.If(kids[1] == constant, apply(kids[0], constant), out)
        return out

    return bounded_abstraction.rewrite(formulas, step)
