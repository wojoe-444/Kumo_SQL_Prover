"""In MySQL ``x / 60 < 5`` over a whole-number column is ``x < 300``: a quotient by a constant, compared with a constant.

    duration / 60 >= 5 AND duration / 60 < 10        duration >= 300 AND duration < 600
    duration / 60 BETWEEN 5 AND 9.99                 duration BETWEEN 300 AND 599
    COUNT(duration / 60)                             COUNT(duration)

**Why it holds.** ``x`` is a whole number, ``c`` a positive constant and ``k`` a constant with at most four decimals.
MySQL works ``x / c`` out as an exact decimal and keeps at least four decimals of it (``div_precision_increment``
is 4 by default, and the prover assumes the default as it does for a quotient that is selected): the value compared
is the true quotient rounded or cut to ``n >= 4`` decimals. That rounding never moves a value across ``k`` when the
true quotient is ``k`` itself (``k`` has at most four decimals, so it stays ``k``) or lies more than ``0.0001`` away
from it (it moves by less than that). The true quotient differs from ``k`` by ``|x - k * c| / c``, and over the whole
numbers ``x`` the least non-zero ``|x - k * c|`` is 1 when ``k * c`` is a whole number and otherwise the distance
``d`` from ``k * c`` to the nearest whole number. So when ``1 / c`` (or ``d / c``) is more than ``0.0001``, the
comparison of the stored quotient with ``k`` is the comparison of the true quotient with ``k``, which is the
comparison of ``x`` with ``k * c``:

    x / c <  k   is   x <  ceil(k * c)          x / c >= k   is   x >= ceil(k * c)
    x / c <= k   is   x <= floor(k * c)         x / c >  k   is   x >  floor(k * c)
    x / c =  k   is   x =  k * c  and  x / c <> k  is  x <> k * c     (only when k * c is a whole number)

Both sides are NULL exactly when ``x`` is, and ``c`` is not zero, so the division raises nothing. ``COUNT(x / c)``
counts the rows where ``x`` is not NULL for the same reason. ``tools``-free check: the reading was compared with a
real MySQL 8.0 server for INT, BIGINT and SMALLINT columns (negative values and NULL among them), divisors up to
100,000 written as whole numbers and as decimals, and constants on and beside every boundary, in all six
comparisons and both operand orders; ``tests/test_quotient_comparisons.py`` replays the arithmetic of the argument
with exact fractions.

**What is read.** In MySQL only. A ``/`` whose left side is a plain column with a declared whole-number type (of a
base table, or of a derived table that outputs such a column: the resolution of ``shifted_equalities``), written
with the alias of a source of its own select or bare in a select with one source and no output of that name, and
whose right side is a positive number literal. The quotient is one side of ``<``, ``<=``, ``>``, ``>=``, ``=`` or
``<>`` against a number literal (possibly negated), or the tested value of a ``BETWEEN`` with two such literals, and
the numbers pass the test above. Or it is the whole argument of a ``COUNT`` without ``DISTINCT``.

**What is not read.** ``DIV``, ``ROUND`` or ``FLOOR`` of a quotient; a quotient compared with a column or with text;
a literal with an exponent; a constant with more than four decimals; a column that is not declared a whole number
(a decimal column keeps more decimals of the quotient, and text divides as a floating-point number); any other
dialect, where ``/`` on whole numbers is another operation.

A pre-pass of the sixteenth hardest-cases batch (``shape_fallback.SIXTEENTH_BATCH``): it fires only in the last step
of a proof.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from sqlglot import exp

from .shifted_equalities import _column_kind

NAME = "quotient_comparisons"

_FLIP = {exp.LT: exp.GT, exp.LTE: exp.GTE, exp.GT: exp.LT, exp.GTE: exp.LTE, exp.EQ: exp.EQ, exp.NEQ: exp.NEQ}
_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)\Z")
_STEP = Fraction(1, 10000)  # the quotient keeps at least four decimals


def _unparen(node: exp.Expression | None) -> exp.Expression | None:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _number(node: exp.Expression | None) -> Fraction | None:
    """The value of a number literal written with digits and at most one point, possibly negated."""

    node = _unparen(node)
    sign = 1
    if isinstance(node, exp.Neg):
        sign, node = -1, _unparen(node.this)
    if isinstance(node, exp.Literal) and not node.is_string and _NUMBER.match(str(node.this)) and len(str(node.this)) <= 30:
        return sign * Fraction(str(node.this))
    return None


def _literal(value: int) -> exp.Expression:
    number = exp.Literal.number(abs(value))
    return exp.Neg(this=number) if value < 0 else number


def safe(divisor: Fraction, constant: Fraction) -> bool:
    """Whether comparing ``x / divisor`` kept to four or more decimals with ``constant`` is comparing the true quotient."""

    if divisor <= 0 or (constant / _STEP).denominator != 1:
        return False
    product = constant * divisor
    gap = Fraction(1) if product.denominator == 1 else min(product - math.floor(product), math.ceil(product) - product)
    return gap / divisor > _STEP


def bound(kind: type, divisor: Fraction, constant: Fraction) -> int | None:
    """The whole number ``b`` with ``x / divisor <kind> constant`` being ``x <kind> b``, or None when there is none."""

    if not safe(divisor, constant):
        return None
    product = constant * divisor
    if kind in (exp.LT, exp.GTE):
        return math.ceil(product)
    if kind in (exp.LTE, exp.GT):
        return math.floor(product)
    return int(product) if product.denominator == 1 else None


def _quotient(node: exp.Expression | None, types: dict, dialect: str) -> tuple[exp.Column, Fraction] | None:
    """``(x, c)`` when ``node`` is ``x / c`` with ``x`` a whole-number column of its select and ``c`` a positive literal."""

    node = _unparen(node)
    if type(node) is not exp.Div:
        return None
    column, divisor = _unparen(node.this), _number(node.expression)
    if divisor is None or divisor <= 0 or not isinstance(column, exp.Column) or not isinstance(column.this, exp.Identifier):
        return None
    if isinstance(_unparen(node.expression), exp.Neg):
        return None
    select = column.find_ancestor(exp.Select)
    if select is None:
        return None
    if not column.table and any(isinstance(item, exp.Alias) and item.alias.lower() == column.name.lower() for item in select.expressions):
        return None  # a bare name may be an output of the select where it is read in GROUP BY, HAVING or ORDER BY
    if _column_kind(column, select, types, dialect) != "int":
        return None
    return column, divisor


def read_quotient_comparisons(tree: exp.Expression, types: dict | None, dialect: str) -> tuple[exp.Expression, bool]:
    """``tree`` with each such quotient comparison written on the column itself (module doc); whether it changed."""

    if dialect != "mysql" or not types or not any(True for _ in tree.find_all(exp.Div)):
        return tree, False
    named = {cte.alias.lower() for cte in tree.find_all(exp.CTE)}  # a WITH name may hide a table: its columns are not the declared ones
    declared = {str(t).lower(): {str(c).lower(): v for c, v in columns.items()} for t, columns in types.items() if str(t).lower() not in named}
    tree = tree.copy()
    changed = False
    for node in list(tree.find_all(exp.LT, exp.LTE, exp.GT, exp.GTE, exp.EQ, exp.NEQ, exp.Between, exp.Count)):
        new: exp.Expression | None = None
        if isinstance(node, exp.Count):
            found = _quotient(node.this, declared, dialect)
            if found is not None and not node.expressions:
                node.set("this", found[0].copy())
                changed = True
        elif isinstance(node, exp.Between):
            found = _quotient(node.this, declared, dialect)
            low, high = _number(node.args.get("low")), _number(node.args.get("high"))
            if found is not None and low is not None and high is not None and not node.args.get("symmetric"):
                first, last = bound(exp.GTE, found[1], low), bound(exp.LTE, found[1], high)
                if first is not None and last is not None:
                    new = exp.Between(this=found[0].copy(), low=_literal(first), high=_literal(last))
        else:
            for kind, value, other in ((type(node), node.this, node.expression), (_FLIP[type(node)], node.expression, node.this)):
                found, constant = _quotient(value, declared, dialect), _number(other)
                if found is None or constant is None:
                    continue
                limit = bound(kind, found[1], constant)
                if limit is not None:
                    new = kind(this=found[0].copy(), expression=_literal(limit))
                break
        if new is not None:
            node.replace(new)
            changed = True
    return tree, changed


def quotient_step(select: exp.Select, types: dict | None, dialect: str) -> exp.Select | None:
    """The reading as a rule on one select: a copy with its quotient comparisons written out, or None when there is none.

    The normalizer applies it to every select until nothing changes, so a quotient that a derived table computes
    (``SELECT secs / 60 AS minutes ..``) is read once that table has been merged into the select that compares it.
    """

    if dialect != "mysql" or not types or not any(True for _ in select.find_all(exp.Div)):
        return None
    if any(True for _ in select.root().find_all(exp.CTE)):
        return None  # a WITH name may hide a table: the declared types are not known to be its columns'
    written, changed = read_quotient_comparisons(select, types, dialect)
    return written if changed else None
