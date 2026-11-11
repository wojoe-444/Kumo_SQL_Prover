"""A power of ten multiplied before or after a MySQL division of integers, under one ``ROUND``.

    ROUND(100 * SUM(paid = due) / COUNT(*), 2)
    ROUND(SUM(paid = due) / COUNT(*) * 100, 2)

are two spellings of one percentage. The prover reads ``/`` as an unknown function of its two operands, so it
could not see that ``(100 * a) / b`` and ``(a / b) * 100`` round to the same number. This rule brings the first
spelling to the second.

**The rule.** ``ROUND(c * a / b, d)`` (also written ``a * c / b``) is ``ROUND(a / b * c, d)`` when

* ``a`` and ``b`` are integers of an exact type (see below),
* ``c`` is the integer literal ``10``, ``100``, ``1000`` or ``10000``,
* ``d`` is absent or an integer literal from ``0`` to ``4``.

**Why it is sound.** MySQL divides two exact integers in decimal arithmetic. The quotient keeps nine digits
after the point and the digits past them are cut off (its declared scale is four digits, which is what a client
sees, but an enclosing operator receives all nine). Write ``q`` for the true quotient ``a / b`` and ``c = 10^k``:

* ``c * a`` is an exact integer, and ``c * a / b`` is ``q * 10^k`` cut after 9 digits;
* ``a / b`` is ``q`` cut after 9 digits, and its product with ``10^k`` is exact: ``q * 10^k`` cut after ``9 - k``
  digits.

Both are ``q * 10^k`` cut after at least five digits (``k`` is at most 4). ``ROUND(x, d)`` of an exact number
rounds half away from zero, so it reads only the digits of ``x`` up to the one after position ``d``: with ``d``
at most 4, it gives the same result for a number and for that number cut after five or more digits (cutting is
toward zero for both signs, as the rounding is symmetric). So both spellings are the true ``q * 10^k`` rounded
to ``d`` digits. The same holds if an engine rounded each operand of ``ROUND`` to its declared scale first: both
spellings have the declared scale 4.

A zero ``b`` makes both NULL (MySQL's ``/`` by zero is NULL), and so does a NULL ``a`` or ``b``. The product
``c * a`` is taken not to overflow, as the prover's exact arithmetic takes every product.

What would break it, and is refused: a ``DECIMAL`` or ``DOUBLE`` operand (a ``DECIMAL`` operand changes how many
digits the quotient keeps and its declared scale; a ``DOUBLE`` is binary), a literal ``100.0`` (the product gets
a declared scale of one digit), a factor that is not a power of ten (the product of a cut quotient with 3 is not
a cut of the product), more than four decimals.

**What an exact integer is here.** ``COUNT(..)``; ``SUM(e)`` of an integer ``e`` (MySQL sums integers as an
exact ``DECIMAL`` of scale 0); a comparison or a logical operator (0, 1 or NULL); an integer literal; a column
whose declared type is an integer type, read through the select the ``ROUND`` is in; ``+``, ``-``, ``*`` and
unary minus over those; a ``CASE`` or ``IF`` whose arms all are. Anything else (a function, a subquery, a column
of a derived table) is not read.

Only on MySQL, and only when the caller supplied column types.
"""

from __future__ import annotations

import re

from sqlglot import exp

from .aggregate_membership import _unwrap
from .exact_rounding import _arms, _attached, _decimals, _typer

_INTEGER_TYPE = re.compile(r"^\s*(TINYINT|SMALLINT|MEDIUMINT|INT|INTEGER|BIGINT)\b(\s*\(\s*\d+\s*\))?\s*(UNSIGNED)?\s*$", re.I)
_POWERS = ("10", "100", "1000", "10000")
_MOST_DECIMALS = 4
_TRUTH = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.NullSafeEQ, exp.Is, exp.And, exp.Or, exp.Not)


def scale_after_division(select: exp.Select, types: dict[str, dict[str, str]] | None, dialect: str) -> exp.Expression | None:
    """``select`` (in place) with ``ROUND(c * a / b, d)`` of its own scope written ``ROUND(a / b * c, d)``, else ``None``."""

    if dialect != "mysql" or not types:
        return None
    rounds = [node for node in select.find_all(exp.Round) if node.find_ancestor(exp.Select) is select]
    if not rounds:
        return None
    column_type = _typer(select, types)
    changed = False
    for node in rounds:
        if not _attached(node, select):
            continue
        decimals = _decimals(node)
        if decimals is None or decimals > _MOST_DECIMALS:
            continue
        division = _unwrap(node.this)
        if type(division) is not exp.Div:
            continue
        product = _unwrap(division.this)
        if type(product) is not exp.Mul:
            continue
        for factor, value in ((product.this, product.expression), (product.expression, product.this)):
            if _power_of_ten(factor) and _integer(value, column_type) and _integer(division.expression, column_type):
                quotient = division.copy()
                quotient.set("this", _operand(value))
                quotient.set("expression", _operand(division.expression))
                node.set("this", exp.Mul(this=quotient, expression=_unwrap(factor).copy()))
                changed = True
                break
    return select if changed else None


def _power_of_ten(node: exp.Expression) -> bool:
    node = _unwrap(node)
    return isinstance(node, exp.Literal) and not node.is_string and node.name in _POWERS


def _operand(node: exp.Expression) -> exp.Expression:
    """A copy of ``node`` that can stand as an operand of ``/``."""

    node = _unwrap(node)
    return node.copy() if isinstance(node, (exp.Column, exp.Literal, exp.Func)) and not isinstance(node, (exp.Case, exp.If)) else exp.Paren(this=node.copy())


def _integer(node: exp.Expression, column_type) -> bool:
    """Whether ``node`` is an integer of an exact MySQL type (the module docstring's list)."""

    node = _unwrap(node)
    if isinstance(node, exp.Count):
        return not any(value for key, value in node.args.items() if key not in ("this", "big_int"))
    if type(node) is exp.Sum:
        inner = node.this
        if isinstance(inner, exp.Distinct):
            if len(inner.expressions) != 1:
                return False
            inner = inner.expressions[0]
        return isinstance(inner, exp.Expression) and _integer(inner, column_type)
    if type(node) in _TRUTH:
        return True
    if isinstance(node, exp.Column):
        declared = column_type(node)
        return bool(declared and _INTEGER_TYPE.match(declared))
    if isinstance(node, exp.Literal):
        return not node.is_string and bool(re.fullmatch(r"\d+", node.name))
    if isinstance(node, exp.Neg):
        return _integer(node.this, column_type)
    if type(node) in (exp.Add, exp.Sub, exp.Mul):
        return _integer(node.this, column_type) and _integer(node.expression, column_type)
    arms = _arms(node)
    return arms is not None and all(_integer(arm, column_type) for arm in arms)
