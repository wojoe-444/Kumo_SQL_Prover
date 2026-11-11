"""A MySQL division of an integer by a power of ten, under ``ROUND(.., 0)``, is a product with a decimal literal.

    ROUND(pay - pay * 24 / 100)
    ROUND(pay - (24 / 100) * pay, 0)
    ROUND(pay * 0.76)

are three spellings of one amount when ``pay`` is an ``INT`` column. The prover reads ``/`` as an unknown
function, and adds, subtracts and multiplies exactly, so it proves the three equal once the divisions are written
as exact products: ``pay * 24 * 0.01`` and ``24 * 0.01 * pay``.

**The rule.** Inside the argument of ``ROUND(e)`` or ``ROUND(e, 0)``, a division ``a / 10``, ``a / 100``,
``a / 1000`` or ``a / 10000`` of an integer ``a`` is ``a * 0.1``, ``a * 0.01``, ``a * 0.001`` or ``a * 0.0001``.

**Why it is sound.** MySQL divides two exact integers in decimal arithmetic and keeps nine digits of the quotient
after the point. A quotient by ``10^k`` with ``k`` at most 4 has at most four, so nothing is cut: the division
gives the exact number ``a / 10^k``, which is also what the exact product of ``a`` with the literal gives. A NULL
``a`` makes both NULL, and the divisor is never zero.

The two spellings give the same number, but not the same *declared scale* (four digits for the division, ``k``
for the product), and MySQL uses the declared scale when it prints a value, turns it into text, divides it again,
averages it, or rounds it to more decimals than it declares. So the division is rewritten only where its value
can reach nothing but exact arithmetic and then a ``ROUND`` to zero decimals:

* the ``ROUND`` has no second argument, or the literal ``0``, and its whole argument is *exact* (the reading of
  :mod:`kumo_sql_prover.exact_rounding`: integer and ``DECIMAL`` columns of a base table of the select, plain
  number literals, NULL, ``+ - * / DIV %`` over those, ``ROUND``, and ``CASE``/``IF`` with exact arms);
* between the ``ROUND`` and the division there are only parentheses, ``+``, ``-``, ``*``, unary minus, and the
  value arms of ``CASE`` and ``IF`` (the conditions are not entered).

There, both spellings carry the same exact number up to the ``ROUND`` (the prover's exact arithmetic takes
``+``, ``-`` and ``*`` of exact numbers as never rounding), ``ROUND(x, 0)`` rounds the number itself, and the
result has no decimals in either spelling.

Refused: a divisor that is not one of the four literals (``/ 3`` is cut after nine digits, ``/ 100.0`` and
``/ 1e2`` are not integers), a dividend that is not an integer (an integer column by its declared type, an
integer literal, ``+ - *`` and unary minus over those), a division below another ``/``, a function or an
aggregate, a ``ROUND`` to one or more decimals.

Only on MySQL, and only when the caller supplied column types.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _unwrap
from .exact_rounding import _arms, _attached, _decimals, _exact, _integer, _typer

_FACTORS = {"10": "0.1", "100": "0.01", "1000": "0.001", "10000": "0.0001"}


def divide_by_power_of_ten(select: exp.Select, types: dict[str, dict[str, str]] | None, dialect: str) -> exp.Expression | None:
    """``select`` (in place) with the divisions of its own scope rewritten as the module docstring says, else ``None``."""

    if dialect != "mysql" or not types:
        return None
    rounds = [node for node in select.find_all(exp.Round) if node.find_ancestor(exp.Select) is select]
    if not rounds:
        return None
    column_type = _typer(select, types)
    changed = False
    for node in rounds:
        if not _attached(node, select) or _decimals(node) != 0 or not _exact(node.this, column_type):
            continue
        for division in _reachable_divisions(node.this):
            divisor = _unwrap(division.expression)
            if not (isinstance(divisor, exp.Literal) and not divisor.is_string and divisor.name in _FACTORS):
                continue
            dividend = _unwrap(division.this)
            if not _integer(dividend, column_type):
                continue
            left = exp.Paren(this=dividend.copy()) if isinstance(dividend, exp.Binary) and not isinstance(dividend, exp.Mul) else dividend.copy()
            division.replace(exp.Mul(this=left, expression=exp.Literal.number(_FACTORS[divisor.name])))
            changed = True
    return select if changed else None


def _reachable_divisions(node: exp.Expression) -> list[exp.Div]:
    """The ``/`` nodes whose value reaches ``node`` through parentheses, ``+``, ``-``, ``*``, unary minus and the
    value arms of ``CASE`` and ``IF`` only."""

    found: list[exp.Div] = []
    pending = [node]
    while pending:
        current = _unwrap(pending.pop())
        if type(current) is exp.Div:
            found.append(current)
        elif type(current) in (exp.Add, exp.Sub, exp.Mul):
            pending += [current.this, current.expression]
        elif isinstance(current, exp.Neg):
            pending.append(current.this)
        else:
            arms = _arms(current)
            if arms is not None:
                pending += arms
    return found
