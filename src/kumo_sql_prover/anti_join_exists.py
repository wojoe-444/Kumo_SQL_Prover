"""An anti-join written as ``LEFT JOIN .. WHERE joined.column IS NULL`` is read as ``NOT EXISTS``.

    SELECT a.g, COUNT(DISTINCT a.x) FROM a LEFT JOIN t ON a.k = t.k AND t.f = 1 WHERE t.k IS NULL GROUP BY a.g

is

    SELECT a.g, COUNT(DISTINCT a.x) FROM a WHERE NOT EXISTS (SELECT 1 FROM t WHERE a.k = t.k AND t.f = 1) GROUP BY a.g

Take a row ``r`` of everything joined before ``t``. The ``LEFT JOIN`` keeps it once per row of ``t`` that makes the
``ON`` condition TRUE, or once, padded with NULLs, when no row of ``t`` does. The ``WHERE`` conjunct
``t.c IS NULL`` is FALSE on every matched row when ``c`` cannot be NULL there: ``c`` is declared NOT NULL, or the
``ON`` condition has a top-level conjunct that compares the bare column ``t.c`` (``=``, ``<>``, ``<``, ``<=``, ``>``,
``>=``), which is not TRUE for a NULL. On the padded row it is TRUE. So the rows that pass are exactly the rows
``r`` for which no row of ``t`` makes the ``ON`` condition TRUE, each once: ``NOT EXISTS (SELECT 1 FROM t WHERE on)``
is TRUE for exactly those rows (it is never NULL), and keeps each once. An empty ``t`` pads every row and makes
``NOT EXISTS`` TRUE for every row; duplicates of ``r`` are kept as they are on both sides.

The prover reads the join form directly when nothing is above it, but an aggregate over an outer join is first
wrapped into a derived table (``algebraic_equivalence._wrap_outer_join_aggregate``), and the anti-join inside it
never meets a ``NOT IN`` or ``NOT EXISTS`` on the other side. This rule runs before the wrap.

Conditions, all checked:

* the join is a plain ``LEFT [OUTER] JOIN`` of a base table with an ``ON`` condition (no ``USING``, ``NATURAL``,
  lateral or sampled source), the table's columns are known from the schema and its name is used once in the select;
* no join of the select uses ``USING`` or ``NATURAL`` (they read columns by name), and every join after it is an
  inner, cross or ``LEFT`` join (a ``RIGHT`` or ``FULL`` join pads the left side with NULLs, and those rows pass
  ``IS NULL`` without having met ``t`` at all);
* the test is a top-level ``WHERE`` conjunct ``t.c IS NULL`` with ``c`` a column of ``t`` that cannot be NULL on a
  matched row (above);
* the ``ON`` condition holds no subquery, aggregate or window, and every column in it is qualified with the
  joined table's name or with a source that comes before the join, so it means the same inside the subquery;
* nothing else in the select, at any depth, reads the joined table (``unmatched_outer_join._unread``, applied
  to the rewritten select): the padded columns are never seen.

Later inner and ``LEFT`` joins do not read ``t``, so each surviving row is repeated once per match either way.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import select_sources
from .unmatched_outer_join import _columns_of, _conjuncts, _plain_column, _plain_table, _unread

_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
_JOIN_ARGS = frozenset({"this", "side", "kind", "on"})


def _and_all(parts: list[exp.Expression]) -> exp.Expression:
    parts = [exp.Paren(this=p) if isinstance(p, (exp.Or, exp.Xor)) else p for p in parts]  # they bind looser than AND
    result = parts[0]
    for part in parts[1:]:
        result = exp.And(this=result, expression=part)
    return result


def _null_test(test: exp.Expression, name: str, columns: set[str]) -> str | None:
    """``c`` when ``test`` is ``name.c IS NULL`` for a column ``c`` of the joined table."""

    while isinstance(test, exp.Paren):
        test = test.this
    if type(test) is not exp.Is or type(test.expression) is not exp.Null:
        return None
    if any(value for key, value in test.args.items() if key not in ("this", "expression")):
        return None  # IS NOT NULL is parsed as IS with a negate flag
    column = _plain_column(test.this)
    if column is None or column.table.lower() != name or column.name.lower() not in columns:
        return None
    return column.name.lower()


def _compared_in_on(on: exp.Expression, name: str, column: str) -> bool:
    for part in _conjuncts(on):
        if type(part) not in _COMPARISONS:
            continue
        for side in (part.this, part.expression):
            mine = _plain_column(side)
            if mine is not None and mine.table.lower() == name and mine.name.lower() == column:
                return True
    return False


def _later_joins_keep_rows(joins: list[exp.Join]) -> bool:
    for join in joins:
        if (join.args.get("side") or "").upper() not in ("", "LEFT"):
            return False
        if (join.args.get("kind") or "").upper() not in ("", "INNER", "CROSS", "OUTER"):
            return False
    return True


def read_anti_join(select: exp.Select, schema: dict | None, not_null: dict | None) -> exp.Expression | None:
    """``select`` with one ``LEFT JOIN t .. WHERE t.c IS NULL`` anti-join written as ``NOT EXISTS``, or ``None``."""

    joins = select.args.get("joins") or []
    where = select.args.get("where")
    if not joins or where is None or not schema or select.args.get("laterals"):
        return None
    if any(j.args.get("using") or j.args.get("method") for j in joins):
        return None
    declared = {str(t).lower(): {str(c).lower() for c in columns} for t, columns in schema.items()}
    never_null = {str(t).lower(): {str(c).lower() for c in columns} for t, columns in (not_null or {}).items()}
    names = [(s.alias_or_name or "").lower() for s in select_sources(select)]
    tests = _conjuncts(where.this)
    for index, join in enumerate(joins):
        joined = join.this
        if (join.args.get("side") or "").upper() != "LEFT" or (join.args.get("kind") or "").upper() not in ("", "OUTER"):
            continue
        on = join.args.get("on")
        if on is None or any(v for key, v in join.args.items() if key not in _JOIN_ARGS):
            continue
        if not _plain_table(joined) or not _later_joins_keep_rows(joins[index + 1:]):
            continue
        name = (joined.alias_or_name or "").lower()
        columns = _columns_of(joined, declared)
        if not name or not columns or names.count(name) != 1:
            continue
        if any(isinstance(n, (exp.Query, exp.Subquery, exp.Exists, exp.AggFunc, exp.Window, exp.Star)) for n in on.walk()):
            continue
        before = set(names[: index + 1])
        if any(_plain_column(c) is None or c.table.lower() not in before | {name} for c in on.find_all(exp.Column)):
            continue
        full = ".".join(part.name for part in joined.parts).lower()
        declared_not_null = never_null.get(full) or never_null.get(joined.name.lower()) or set()
        for position, test in enumerate(tests):
            column = _null_test(test, name, columns)
            if column is None or not (column in declared_not_null or _compared_in_on(on, name, column)):
                continue
            copy = select.copy()
            new_join = copy.args["joins"][index]
            probe = exp.Select(expressions=[exp.Literal.number(1)]).from_(new_join.this.copy())
            probe.set("where", exp.Where(this=new_join.args["on"].copy()))
            absent = exp.Not(this=exp.Exists(this=probe))
            parts = [t.copy() for t in _conjuncts(copy.args["where"].this)]
            parts[position] = absent
            copy.set("where", exp.Where(this=_and_all(parts)))
            new_join.pop()
            if not copy.args.get("joins"):
                copy.set("joins", None)
            if not _unread(copy, absent, name, columns, declared):
                continue
            return copy
    return None
