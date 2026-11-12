"""A filter on the partition keys of a windowed derived table moves below its window functions.

Calcite's ``FilterWindowTransposeRule`` and ``FilterProjectTransposeRule`` push a filter under a window when the
filter reads only columns every window partitions by::

    SELECT d.n, d.c FROM (SELECT t.name AS n, t.dept AS k, COUNT(*) OVER (PARTITION BY t.dept) AS c FROM t) AS d
    WHERE d.k = 0

    SELECT d.n, d.c FROM (SELECT t.name AS n, t.dept AS k, COUNT(*) OVER (PARTITION BY t.dept) AS c FROM t
                          WHERE t.dept = 0) AS d

Such a filter has one value per partition of every window, so it keeps or removes whole partitions. A window
function reads only the rows of the current row's partition, so the rows that stay keep their values, whatever the
function, its ``ORDER BY`` or its frame.

A filter on a derived table that merely projects a windowed one (no window function of its own) moves into it as
any filter moves through a projection; ``algebraic_equivalence._push_filter_into_derived`` does not look at a
derived table with a window function anywhere inside, so that step is done here too.

Declined: a conjunct that reads anything but the derived table's columns, reads a column that is not, text for
text, a ``PARTITION BY`` expression of every window function of the derived table, or holds a subquery, an
aggregate, a window function or a function whose value can change between two reads; a window with no
``PARTITION BY`` or one that names a ``WINDOW`` clause; a derived table that is ``DISTINCT``, grouped, aggregating,
ordered or limited or has a star; a select with joins, where the filter may sit on the padded side of an outer
join.

This is a shape rule (see ``shape_fallback``), and a pair it changed that the prover cannot prove is tried again
without it (the narrow second attempt, rule name ``window_filters``): one pair of QUITE's SQLStorm development
set is proved only when this filter stays above the window in the one query that has it.
"""

from __future__ import annotations

from sqlglot import exp

from . import shape_fallback

_INNER_ARGS = {"expressions", "from_", "from", "joins", "where"}
_VOLATILE = (exp.Rand, exp.Randn, exp.Randstr, exp.Uuid, exp.Anonymous, exp.CurrentTimestamp, exp.CurrentDate, exp.CurrentTime)
_NOT_ROW_LOCAL = (exp.Subquery, exp.Select, exp.Exists, exp.SetOperation, exp.AggFunc, exp.Window, exp.Star, *_VOLATILE)


def _star(node: exp.Expression) -> bool:
    """A star that stands for columns (``*``, ``t.*``), not the one in ``COUNT(*)``."""

    return isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count)


def _conjuncts(node: exp.Expression) -> list[exp.Expression]:
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _text(node: exp.Expression) -> str:
    while isinstance(node, exp.Paren):
        node = node.this
    return node.sql().lower()


def push_filter_below_windows(select: exp.Select) -> exp.Expression | None:
    """``select`` with the conjuncts of its ``WHERE`` that qualify moved into its derived table, or ``None``."""

    if not shape_fallback.rule_enabled("window_filters"):
        return None  # the narrow second attempt (``shape_fallback.without``)

    where = select.args.get("where")
    from_ = select.args.get("from_") or select.args.get("from")
    if where is None or from_ is None or select.args.get("joins"):
        return None
    source = from_.this
    if not isinstance(source, exp.Subquery) or not source.alias or type(source.this) is not exp.Select:
        return None
    if any(v for k, v in source.args.items() if k not in ("this", "alias")) or source.args["alias"].args.get("columns"):
        return None
    inner = source.this
    if any(v for k, v in inner.args.items() if v and k not in _INNER_ARGS) or inner.find(exp.Window) is None:
        return None
    if (inner.args.get("from_") or inner.args.get("from")) is None:
        return None
    values: dict[str, exp.Expression] = {}
    for item in inner.expressions:
        name = item.alias_or_name.lower()
        if not name or name in values or any(_star(node) for node in item.walk()):
            return None
        values[name] = item.this if isinstance(item, exp.Alias) else item
    windows = [w for w in inner.find_all(exp.Window) if w.find_ancestor(exp.Select) is inner]
    if any(a.find_ancestor(exp.Select) is inner and a.find_ancestor(exp.Window) is None for a in inner.find_all(exp.AggFunc)):
        return None
    partitions: list[set[str]] = []
    for window in windows:
        if not isinstance(window.this, exp.Expression) or isinstance(window.args.get("alias"), exp.Identifier):
            return None
        keys = window.args.get("partition_by") or []
        if not keys:
            return None
        partitions.append({_text(key) for key in keys})

    alias = source.alias.lower()
    moved, kept = [], []
    for conjunct in _conjuncts(where.this):
        columns = list(conjunct.find_all(exp.Column))
        ok = bool(columns) and not any(isinstance(node, _NOT_ROW_LOCAL) for node in conjunct.walk())
        for column in columns if ok else []:
            value = values.get(column.name.lower())
            if (column.table and column.table.lower() != alias) or column.args.get("db") or value is None:
                ok = False
            elif any(isinstance(node, _NOT_ROW_LOCAL) for node in value.walk()):
                ok = False
            elif any(_text(value) not in keys for keys in partitions):
                ok = False
            if not ok:
                break
        (moved if ok else kept).append(conjunct)
    if not moved:
        return None

    copy = select.copy()
    new_inner = (copy.args.get("from_") or copy.args.get("from")).this.this
    pushed = []
    for conjunct in moved:
        conjunct = exp.Paren(this=conjunct.copy())
        for column in list(conjunct.find_all(exp.Column)):
            value = values[column.name.lower()].copy()
            column.replace(value if isinstance(value, (exp.Column, exp.Literal, exp.Paren, exp.Func)) else exp.Paren(this=value))
        pushed.append(conjunct.this)
    condition = exp.and_(*([exp.Paren(this=new_inner.args["where"].this)] if new_inner.args.get("where") is not None else []), *pushed)
    new_inner.set("where", exp.Where(this=condition))
    copy.set("where", exp.Where(this=exp.and_(*[k.copy() for k in kept])) if kept else None)
    shape_fallback.note_fired("window_filters")
    return copy
