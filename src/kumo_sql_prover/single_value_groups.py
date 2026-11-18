"""A group whose one distinct value is a given literal: a row with it, and no row with another.

    SELECT buyer FROM sale GROUP BY buyer HAVING COUNT(DISTINCT label) = 1 AND MIN(label) = 'S'

keeps the buyers whose labels, NULLs aside, are all ``S`` and who have at least one. The same question as two
counts, which ``counted_presence`` reads as "has a row" and "has no row":

    .. HAVING COUNT(CASE WHEN label = 'S' THEN 1 END) > 0 AND COUNT(CASE WHEN label <> 'S' THEN 1 END) = 0

**Why it holds.** Write ``V`` for the set of values of the column in the group that are not NULL.
``COUNT(DISTINCT c) = 1`` says ``V`` has exactly one value, and ``MIN(c) = 'S'`` (or ``MAX``) says the least
(greatest) value of ``V`` equals ``'S'``; together, ``V`` is one value and it equals ``'S'``. The first count is
positive exactly when some value of ``V`` equals ``'S'``, and the second is zero exactly when no value of ``V``
differs from it (``c <> 'S'`` is NULL, not TRUE, for a NULL ``c``); together, ``V`` is not empty and all of it
equals ``'S'``. The two say the same as long as "equals the literal" and "is the same value for ``DISTINCT``"
agree, which is why the literal must be a string (a text column is compared as text both ways, and a number
column is compared as a number both ways, MySQL reading the literal as a number), or a whole number beside a
column declared to hold whole numbers. A number literal beside a text column would not do: ``'5'`` and ``'5.0'``
both equal 5 and are two values for ``DISTINCT``.

**What is read.** Two top-level conjuncts of a ``HAVING``: ``COUNT(DISTINCT c) = 1`` and ``MIN(c) = literal`` or
``MAX(c) = literal`` (either side), with the same plain column ``c`` as written. The rest of the select is not
touched, and a ``HAVING`` that holds only one of the two is left alone. A floating-point column is left alone.

A narrow retry rule (``shape_fallback``). ``tests/test_single_value_groups.py`` replays the rewrite on DuckDB.
"""

from __future__ import annotations

from sqlglot import exp

from . import smt_values
from .ast_utils import conjuncts
from .own_group_membership import _unparen


def _plain(column: exp.Expression | None) -> exp.Column | None:
    column = _unparen(column)
    if isinstance(column, exp.Column) and isinstance(column.this, exp.Identifier) and not column.args.get("db"):
        return column
    return None


def _one_value(test: exp.Expression | None) -> exp.Column | None:
    """``c`` for ``COUNT(DISTINCT c) = 1``."""

    test = _unparen(test)
    if not isinstance(test, exp.EQ):
        return None
    for count, one in ((test.this, test.expression), (test.expression, test.this)):
        count, one = _unparen(count), _unparen(one)
        if isinstance(count, exp.Count) and isinstance(count.this, exp.Distinct) and len(count.this.expressions) == 1 and not count.args.get("expressions"):
            if isinstance(one, exp.Literal) and not one.is_string and one.this == "1":
                return _plain(count.this.expressions[0])
    return None


def _extreme(test: exp.Expression | None) -> tuple[exp.Column, exp.Literal] | None:
    """``(c, literal)`` for ``MIN(c) = literal`` or ``MAX(c) = literal``."""

    test = _unparen(test)
    if not isinstance(test, exp.EQ):
        return None
    for call, literal in ((test.this, test.expression), (test.expression, test.this)):
        call, literal = _unparen(call), _unparen(literal)
        if isinstance(call, (exp.Min, exp.Max)) and not call.args.get("expressions") and isinstance(literal, exp.Literal):
            column = _plain(call.this)
            if column is not None:
                return column, literal
    return None


def _declared(types: dict | None, column: exp.Column) -> list[str]:
    """The declared types of every column of that name (the column's table is not resolved here)."""

    return [str(kind) for columns in (types or {}).values() for name, kind in columns.items() if str(name).lower() == column.name.lower()]


def read_single_value_groups(select: exp.Select, types: dict | None = None) -> exp.Select | None:
    """``select`` with each such pair of ``HAVING`` tests written as two counts, or None."""

    having = select.args.get("having")
    if having is None or select.args.get("group") is None:
        return None
    parts = conjuncts(having.this)
    ones = {index: column for index, part in enumerate(parts) if (column := _one_value(part)) is not None}
    if not ones:
        return None
    replaced: dict[int, list[exp.Expression]] = {}
    for index, part in enumerate(parts):
        found = _extreme(part)
        if found is None or index in ones:
            continue
        column, literal = found
        partner = next((i for i, other in ones.items() if i not in replaced and other.sql(dialect="mysql").lower() == column.sql(dialect="mysql").lower()), None)
        if partner is None:
            continue
        kinds = [smt_values.type_class(kind) for kind in _declared(types, column)]
        if any(kind == smt_values.FLOAT64 for kind in kinds):
            continue
        if not literal.is_string and not (kinds and all(kind == smt_values.INT64 for kind in kinds) and smt_values.is_integer_text(literal.this)):
            continue

        def count(test: exp.Expression) -> exp.Count:
            return exp.Count(this=exp.Case(ifs=[exp.If(this=test, true=exp.Literal.number(1))]))

        replaced[partner] = [exp.GT(this=count(exp.EQ(this=column.copy(), expression=literal.copy())), expression=exp.Literal.number(0))]
        replaced[index] = [exp.EQ(this=count(exp.NEQ(this=column.copy(), expression=literal.copy())), expression=exp.Literal.number(0))]
    if not replaced:
        return None
    result = select.copy()
    tests = [new for index, part in enumerate(parts) for new in (replaced.get(index) or [part.copy()])]
    result.set("having", exp.Having(this=exp.and_(*tests, copy=False)))
    return result
