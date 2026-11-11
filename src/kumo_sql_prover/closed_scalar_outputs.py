"""An output of a derived table that is built from closed scalar subqueries alone is read where it is used.

    SELECT t.k, t.v
    FROM (SELECT ps.k, SUM(ps.q) AS v, (SELECT SUM(ps.q) FROM ps) * 2 AS bound FROM ps GROUP BY ps.k) AS t
    WHERE t.v > t.bound

is

    SELECT t.k, t.v
    FROM (SELECT ps.k, SUM(ps.q) AS v FROM ps GROUP BY ps.k) AS t
    WHERE t.v > ((SELECT SUM(ps.q) FROM ps) * 2)

(TPC-H Q11 once ``group_total_windows`` has read its window threshold as a scalar subquery). From there
``closed_scalar_filters.fold_scalar_filter`` moves the filter into ``HAVING``.

Why this is exact. A closed scalar subquery (``closed_scalar_filters.closed``) has no outer reference, so it has one
value wherever it is evaluated in the query. An expression that reads columns only inside such subqueries, with
nothing volatile, is the same value on every row of the derived table, and the same value when it is written in
the select that reads the table instead. The derived table is joined with commas, ``CROSS JOIN`` or inner joins,
so every row the outer select sees comes from a row of the table: no outer join shows a NULL in place of the
value. Removing an output that nothing else reads changes no row (the table has no ``DISTINCT`` or set operation).

Conditions, all checked:

* the select joins its sources with commas, ``CROSS JOIN`` or inner ``JOIN .. ON`` only;
* the derived table has an alias without a column list and is a select without ``DISTINCT``, ``ORDER BY``, ``LIMIT``,
  ``OFFSET`` or a star, with distinct output names and another output besides this one;
* the output has an alias, holds at least one scalar subquery, every one of them closed, and outside them no
  column, aggregate, window or other nested query; nothing in it is volatile;
* every read of it is in the reading select itself: ``alias.name``, or the bare name when the derived table is the
  select's only source; no nested query of the select, and no other source, could mean the name.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import select_sources
from .closed_scalar_filters import closed
from .keyed_join_groups import declared_columns, plain_joins
from .unmatched_outer_join import _inside


def _constant(value: exp.Expression, inner: exp.Select, declared: dict[str, set[str]]) -> bool:
    """Whether ``value`` reads columns only inside closed scalar subqueries, and holds one."""

    from .counted_membership import _unstable

    scalars = [node for node in value.find_all(exp.Subquery) if not any(_inside(node.parent, other) for other in value.find_all(exp.Subquery) if other is not node)]
    if type(value) is exp.Subquery:
        scalars = [value]
    if not scalars or _unstable(value):
        return False
    if any(type(s.this) is not exp.Select or len(s.this.expressions) != 1 or s.args.get("alias") or not closed(s.this, declared) for s in scalars):
        return False
    for node in value.walk():
        if any(_inside(node, s) for s in scalars):
            continue
        if isinstance(node, (exp.Column, exp.AggFunc, exp.Window, exp.Query, exp.Exists, exp.Star)):
            return False
    return True


def inline_closed_outputs(select: exp.Select, schema: dict | None) -> exp.Expression | None:
    """``select`` with one such output of a derived table written at each of its reads, or ``None``."""

    if not schema or not plain_joins(select) or select.args.get("laterals"):
        return None
    sources = select_sources(select)
    names = [(s.alias_or_name or "").lower() for s in sources]
    if not all(names) or len(set(names)) != len(names):
        return None
    declared = declared_columns(schema)
    for position, source in enumerate(sources):
        if type(source) is not exp.Subquery or type(source.this) is not exp.Select:
            continue
        if source.args["alias"].args.get("columns") or any(v for key, v in source.args.items() if key not in ("this", "alias")):
            continue
        inner = source.this
        if any(inner.args.get(key) for key in ("distinct", "order", "limit", "offset", "qualify")) or len(inner.expressions) < 2:
            continue
        outputs = [(item.alias_or_name or "").lower() for item in inner.expressions]
        if not all(outputs) or len(set(outputs)) != len(outputs) or any(isinstance(item, exp.Star) or item.is_star for item in inner.expressions):
            continue
        alias = names[position]
        for index, item in enumerate(inner.expressions):
            if not isinstance(item, exp.Alias) or not _constant(item.this, inner, declared):
                continue
            name = outputs[index]
            reads, ok = [], True
            for column in select.find_all(exp.Column):
                if _inside(column, source) or column.name.lower() != name:
                    continue
                mine = column.table.lower() == alias if column.table else len(sources) == 1
                if column.find_ancestor(exp.Select) is not select or (column.table and column.table.lower() != alias and column.table.lower() not in names):
                    ok = False  # a nested query, or a qualifier this select does not define, might mean it
                    break
                if not mine:
                    if not column.table:
                        ok = False  # a bare name among several sources
                        break
                    continue
                if isinstance(column.this, exp.Star) or len(column.parts) > 2:
                    ok = False
                    break
                reads.append(column)
            if not ok or not reads:
                continue
            if any(isinstance(n, exp.Star) for part in select.expressions for n in part.walk() if not isinstance(n.parent, exp.Count)):
                continue
            order = select.args.get("order")
            if order is not None and any(_inside(column, order) and not column.table for column in reads) and any(
                (written.alias_or_name or "").lower() == name for written in select.expressions
            ):
                continue  # a bare ORDER BY name could mean the select's own output
            copy = select.copy()
            new_source = select_sources(copy)[position]
            for column in list(copy.find_all(exp.Column)):
                if _inside(column, new_source) or column.name.lower() != name:
                    continue
                if (column.table.lower() == alias) if column.table else True:
                    shown = column.parent if isinstance(column.parent, exp.Alias) else None
                    value = exp.Paren(this=item.this.copy())
                    if shown is None and any(column is written for written in copy.expressions):
                        column.replace(exp.alias_(value, name))  # a bare select item keeps its output name
                    else:
                        column.replace(value)
            new_source.this.expressions[index].pop()
            return copy
    return None
