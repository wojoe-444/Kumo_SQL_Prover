"""Tests of a DATE column against a day, written with ``>=`` and ``<`` only (MySQL).

``date_windows`` writes every test of a ``DATE`` column against a constant day as ``d >= day``, ``d <= day``,
``d = day``, ``d <> day`` or ``BETWEEN``. That makes five spellings of one window read alike, and leaves two tests
that are each other's opposite looking unrelated:

    HAVING MIN(sale_date) >= '2019-01-01'                       -- every sale is in 2019 or later
    WHERE NOT EXISTS (.. WHERE s.sale_date <= '2018-12-31')     -- no sale is in 2018 or earlier

The prover compares days as the strings they are written as, so it cannot tell that no day lies between
``2018-12-31`` and ``2019-01-01``, and "sold only in spring" written with ``DATE(sale_date) < '2019-01-01'`` on one
side (which ``date_windows`` reads as ``<= '2018-12-31'``) stayed unproved against the bounds of the group
(hardest-cases batch 30, LeetCode 1084).

This pre-pass writes the upper tests with the day after:

    d <= '2018-12-31'                       is   d < '2019-01-01'
    d BETWEEN '2019-01-01' AND '2019-03-31'   is   d >= '2019-01-01' AND d < '2019-04-01'

so every boundary is named by one day, and ``d < day`` is the plain opposite of ``d >= day``.

**Why it is the same test.** The column is declared ``DATE`` and holds whole days (the assumption ``date_windows``
states and uses for ``d < day`` in the other direction), so ``d <= day`` and ``d < the next day`` hold for the same
days; a NULL makes both NULL. ``BETWEEN a AND b`` is ``d >= a AND d <= b``. ``MIN`` or ``MAX`` of such a column is one of
the column's days or NULL.

**What is read.** In MySQL, ``x <= 'YYYY-MM-DD'`` and ``x BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'`` where ``x`` is a
column declared ``DATE`` of the nearest select's sources (``date_windows`` says which), or ``MIN`` or ``MAX`` of one,
and the day after the upper day is still a day the server stores. The tests ``date_windows`` writes out are written
out first (``DATE(d) > day`` is ``d >= the next day``, a ``DATEDIFF`` against a whole number, a day worked out from a
constant), whether or not the attempt ran that rule: an attempt that reads the bounds of a group as a missing row
may be one that did not. Everything else stays as written: a ``DATETIME`` or text column (a time of day lies between
two days), a day the server would not read, a test against another column.

It runs in the last step of batch 27 only, so a pair an earlier step decides is decided as it was. An earlier step
asks for that last step only when the query tests ``MIN`` or ``MAX`` of a ``DATE`` column: that is where the opposite
of a test is needed.
"""

from __future__ import annotations

import re

from sqlglot import exp

from . import shape_fallback
from .date_windows import _Dates, _moved, _only, _unparen, _value, constant_day, write_date_windows

NAME = "half_open_days"


def _day(node: exp.Expression | None):
    """The day a string literal ``'YYYY-MM-DD'`` names, None for anything else."""

    node = _unparen(node)
    return constant_day(node) if isinstance(node, exp.Literal) and node.is_string else None


def _plan(tree: exp.Expression, dates: _Dates) -> list[tuple[exp.Expression, exp.Expression]]:
    """The tests to rewrite, each with what it becomes."""

    planned = []
    for node in tree.find_all(exp.LTE, exp.Between):
        select = node.find_ancestor(exp.Select)
        if select is None or select.args.get("with_") or select.args.get("with"):
            continue
        value = _value(node.this, select, dates)
        if value is None:
            continue
        if type(node) is exp.LTE and _only(node, "this", "expression"):
            after = _moved(_day(node.expression), 1)
            if after is None:
                continue
            new: exp.Expression = exp.LT(this=value.copy(), expression=exp.Literal.string(after.isoformat()))
        elif type(node) is exp.Between and _only(node, "this", "low", "high"):
            low, after = _day(node.args["low"]), _moved(_day(node.args["high"]), 1)
            if low is None or after is None:
                continue
            new = exp.Paren(
                this=exp.And(
                    this=exp.GTE(this=value.copy(), expression=exp.Literal.string(low.isoformat())),
                    expression=exp.LT(this=value.copy(), expression=exp.Literal.string(after.isoformat())),
                )
            )
        else:
            continue
        planned.append((node, new))
    return planned


def _tests_a_bound(tree: exp.Expression, dates: _Dates) -> bool:
    """Whether ``tree`` compares ``MIN`` or ``MAX`` of a DATE column with something: where an opposite test is needed."""

    for node in tree.find_all(exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between):
        select = node.find_ancestor(exp.Select)
        if select is None or select.args.get("with_") or select.args.get("with"):
            continue
        for side in (node.this, node.args.get("expression")):
            if side is not None and type(_value(side, select, dates)) in (exp.Min, exp.Max):
                return True
    return False


def read_half_open_days(tree: exp.Expression, schema: dict | None, types: dict | None, dialect: str | None) -> tuple[exp.Expression, bool, bool]:
    """``(tree with those tests written with the day after, whether one was, whether the tree tests MIN or MAX of a DATE
    column)``. The input is not changed."""

    if dialect != "mysql" or not types:
        return tree, False, False
    dates = _Dates(schema, types)
    if not any(re.sub(r"[(<].*", "", kind).strip().upper() == "DATE" for columns in dates.kinds.values() for kind in columns.values()):
        return tree, False, False
    bound = _tests_a_bound(tree, dates)
    written = tree
    if shape_fallback.rule_enabled("date_windows"):
        # the tests ``date_windows`` writes out (``DATE(d) > day``, ``DATEDIFF``), whether or not this attempt ran it
        written, _ = write_date_windows(tree, schema, types, dialect)
    planned = bool(_plan(written, dates))
    if not planned and written is tree:
        return tree, False, bound
    copy = written.copy() if planned else written
    for node, new in _plan(copy, dates) if planned else ():
        if node is copy:
            copy = new
        else:
            node.replace(new)
    if copy.sql(dialect="mysql") == tree.sql(dialect="mysql"):
        return tree, False, bound  # ``d < day`` went there and back
    return copy, True, bound
