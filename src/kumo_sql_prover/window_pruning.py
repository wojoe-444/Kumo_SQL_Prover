"""Drop the outputs of a derived table over window functions that the query around it never reads.

``algebraic_equivalence._prune_derived`` leaves a derived table alone as soon as a window function appears anywhere
inside it, also one level further down. A plan that keeps a column the query above does not read then never meets
the plan that dropped it::

    SELECT d.n, d.c FROM (SELECT t.name AS n, t.dept AS k, SUM(t.x) OVER (PARTITION BY t.dept) AS c FROM t) AS d
    SELECT d.n, d.c FROM (SELECT t.name AS n,              SUM(t.x) OVER (PARTITION BY t.dept) AS c FROM t) AS d

A select with no ``DISTINCT``, grouping or aggregate returns one row per row of its ``FROM`` and ``WHERE``, whatever
its list holds, and a window function reads the rows, never the other outputs. Removing an output therefore changes
no row and no other output.

Declined: a derived table that is ``DISTINCT``, grouped, aggregating, ordered or limited, has a star, a ``QUALIFY``,
named windows, or a function in its list that can produce rows; a query around it that reads it with a star.
Derived tables with no window function inside are left to ``_prune_derived``.

This is a shape rule (see ``shape_fallback``).
"""

from __future__ import annotations

from sqlglot import exp

_INNER_ARGS = {"expressions", "from_", "from", "joins", "where"}
_ROW_MAKERS = (exp.UDTF, exp.Unnest, exp.Explode, exp.GenerateSeries, exp.Anonymous)


def _own(node: exp.Expression, select: exp.Select) -> bool:
    return node.find_ancestor(exp.Select) is select


def prune_windowed_derived(select: exp.Select, schema: dict | None = None) -> exp.Expression | None:
    """``select`` with the unread outputs of its windowed derived tables removed, or ``None``.

    An unread output is removed only when every column it reads is found in a source of the derived table (a base
    table's columns are the declared ones, in ``schema``): an output that reads a column nothing provides is an error
    when the query runs, and removing it would turn a query that fails into one that answers.
    """

    from .windowed_output_pruning import _reads_only_known_columns

    schema = {str(name).lower(): {str(column).lower() for column in columns} for name, columns in (schema or {}).items()}

    if any(isinstance(star, exp.Star) and not isinstance(star.parent, exp.Count) for star in select.find_all(exp.Star)):
        return None
    from_ = select.args.get("from_") or select.args.get("from")
    if from_ is None:
        return None
    sources = [from_.this] + [join.this for join in select.args.get("joins") or []]
    plans = []
    for index, source in enumerate(sources):
        if not isinstance(source, exp.Subquery) or not source.alias or type(source.this) is not exp.Select:
            continue
        if any(v for k, v in source.args.items() if k not in ("this", "alias")) or source.args["alias"].args.get("columns"):
            continue
        inner = source.this
        if any(v for k, v in inner.args.items() if v and k not in _INNER_ARGS) or inner.find(exp.Window) is None:
            continue
        names = [item.alias_or_name.lower() for item in inner.expressions]
        if "" in names or len(set(names)) != len(names):
            continue
        if any(isinstance(node, _ROW_MAKERS) or (isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count)) for item in inner.expressions for node in item.walk()):
            continue
        if any(_own(agg, inner) and agg.find_ancestor(exp.Window) is None for agg in inner.find_all(exp.AggFunc)):
            continue  # an aggregate of its own: removing it could turn one row into a row per input row
        alias = source.alias.lower()
        # every read that could be of this source: qualified with its alias, or unqualified
        read = {c.name.lower() for c in select.find_all(exp.Column) if c.find_ancestor(exp.Subquery) is not source and (not c.table or c.table.lower() == alias)}
        keep = [i for i, name in enumerate(names) if name in read or not _reads_only_known_columns(inner.expressions[i], inner, schema)] or [0]
        if len(keep) < len(names):
            plans.append((index, keep))
    if not plans:
        return None
    copy = select.copy()
    from_ = copy.args.get("from_") or copy.args.get("from")
    sources = [from_.this] + [join.this for join in copy.args.get("joins") or []]
    for index, keep in plans:
        inner = sources[index].this
        inner.set("expressions", [inner.expressions[i] for i in keep])
    return copy
