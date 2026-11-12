"""A windowed select reads its derived table's own sources.

Planners put a projection under a window function (Calcite's ``ProjectJoinTransposeRule`` and
``ProjectSetOpTransposeRule`` compute the window's arguments one level down), and two plans then differ only in what
that projection names its columns::

    SELECT d.c5 + d.c12, COUNT(d.c0) OVER (PARTITION BY d.c7)
    FROM (SELECT e.id AS c0, e.sal AS c5, e.dept AS c7, b.comm AS c12 FROM e JOIN b ON e.name = b.name) AS d

    SELECT e.sal + b.comm, COUNT(e.id) OVER (PARTITION BY e.dept) FROM e JOIN b ON e.name = b.name

The derived table has one row per row its own ``FROM`` and ``WHERE`` produce, and each of its columns is an
expression of that row, so the select above can read those expressions directly: its window functions see the same
rows in the same partitions. The rewrite keeps the derived table's ``FROM``, joins and ``WHERE`` exactly as written
and replaces its select list.

The derived table may compute columns with window functions of its own (Calcite's ``ProjectToWindowRule`` stacks
one window per level). Every window function of a select reads the rows of its ``FROM`` and ``WHERE`` and no other
output, so the two levels' functions can sit in one list. Such a column is only moved when the select above reads
it outside its own window functions: SQL has no window function inside another.

Declined: a select above with anything but a select list (its own ``WHERE``, grouping, ``DISTINCT``, ``ORDER BY``,
``LIMIT``, named windows), with an aggregate outside a window, a subquery, a star, more than one source or a read
that is not of the derived table; a derived table that groups, aggregates outside a window, is ``DISTINCT`` or
limited, has a star or a subquery in its list, names a ``WINDOW`` clause, or computes a column with a function
whose value can change between two reads.

This is a shape rule (see ``shape_fallback``).
"""

from __future__ import annotations

from sqlglot import exp

_VOLATILE = (exp.Rand, exp.Randn, exp.Randstr, exp.Uuid, exp.Anonymous, exp.CurrentTimestamp, exp.CurrentDate, exp.CurrentTime)
_NESTED = (exp.Subquery, exp.Select, exp.Exists, exp.SetOperation)
_OUTER_ARGS = {"expressions", "from_", "from"}
_INNER_ARGS = {"expressions", "from_", "from", "joins", "where"}


def _star(node: exp.Expression) -> bool:
    """A star that stands for columns (``*``, ``t.*``), not the one in ``COUNT(*)``."""

    return isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count)


def _only(node: exp.Expression, allowed: set[str]) -> bool:
    return all(key in allowed for key, value in node.args.items() if value)


def inline_windowed_source(select: exp.Select) -> exp.Expression | None:
    """The rewrite of the module docstring, or ``None`` when ``select`` is not that shape."""

    if not _only(select, _OUTER_ARGS):
        return None
    from_ = select.args.get("from_") or select.args.get("from")
    source = from_.this if from_ is not None else None
    if not isinstance(source, exp.Subquery) or not source.alias or not _only(source, {"this", "alias"}):
        return None
    if source.args["alias"].args.get("columns"):
        return None
    inner = source.this
    if type(inner) is not exp.Select or not _only(inner, _INNER_ARGS):
        return None
    if (inner.args.get("from_") or inner.args.get("from")) is None:
        return None

    windowed = False
    for item in select.expressions:
        for node in item.walk():
            if isinstance(node, _NESTED) or _star(node):
                return None
            if isinstance(node, exp.Window):
                windowed = True
                if isinstance(node.args.get("alias"), exp.Identifier) or not isinstance(node.this, exp.Expression):
                    return None
            elif isinstance(node, exp.AggFunc) and not isinstance(node.parent, exp.Window):
                return None  # an aggregate of the select itself: one row per group, not per row
    if not windowed:
        return None

    values: dict[str, exp.Expression] = {}
    windowed_values: set[str] = set()  # columns the derived table computes with a window function of its own
    for item in inner.expressions:
        name = item.alias_or_name.lower()
        value = item.this if isinstance(item, exp.Alias) else item
        if not name or name in values or not isinstance(item, (exp.Alias, exp.Column)):
            return None
        if any(isinstance(node, (*_NESTED, *_VOLATILE)) or _star(node) for node in value.walk()):
            return None
        if any(isinstance(node, exp.AggFunc) and node.find_ancestor(exp.Window) is None for node in value.walk()):
            return None
        for window in value.find_all(exp.Window):
            if isinstance(window.args.get("alias"), exp.Identifier) or not isinstance(window.this, exp.Expression):
                return None
            windowed_values.add(name)
        values[name] = value

    alias = source.alias.lower()
    result = inner.copy()
    items = []
    for item in select.expressions:
        named = item
        if isinstance(item, exp.Column):
            named = exp.alias_(item.copy(), item.name)  # a bare column keeps its output name once replaced
        named = named.copy()
        holder = exp.Tuple(expressions=[named])
        for column in list(holder.find_all(exp.Column)):
            table = column.table.lower()
            if (table and table != alias) or column.args.get("db") or column.name.lower() not in values:
                return None
            if column.name.lower() in windowed_values and column.find_ancestor(exp.Window) is not None:
                return None  # a window function cannot hold another: this read stays a level above
            value = values[column.name.lower()].copy()
            column.replace(value if isinstance(value, (exp.Column, exp.Literal, exp.Paren, exp.Func, exp.Window)) else exp.Paren(this=value))
        items.append(holder.expressions[0])
    result.set("expressions", items)
    return result
