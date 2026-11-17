"""A filter on a ``DISTINCT`` derived table that lists every column of it is a ``DISTINCT`` over the filtered rows.

    SELECT d.buyer FROM (SELECT DISTINCT s.buyer FROM sales AS s WHERE s.item = 1) AS d
    WHERE NOT EXISTS (SELECT 1 FROM sales AS t WHERE t.item = 2 AND t.buyer = d.buyer)

is

    SELECT DISTINCT d.buyer FROM (SELECT s.buyer FROM sales AS s WHERE s.item = 1) AS d
    WHERE NOT EXISTS (SELECT 1 FROM sales AS t WHERE t.item = 2 AND t.buyer = d.buyer)

Why it holds. The derived table has no two equal rows, and the select lists every column of it once, so its
result is the set of derived rows that pass the ``WHERE``. The ``WHERE`` reads nothing of the source but those
columns, so without the inner ``DISTINCT`` the copies of a row pass or fail together, and the outer ``DISTINCT``
leaves one of each row that passes: the same set. The other rules already read a ``DISTINCT`` select over a plain
derived table (they merge the two), and did not read a subquery predicate outside a ``DISTINCT`` one.

The rule fires on a select that reads one derived table and nothing else, has a ``WHERE`` and no grouping,
aggregate, window, ``DISTINCT``, ``ORDER BY`` or ``LIMIT`` of its own, and lists plain columns of the derived
table: every output of it exactly once, in any order, renamed or not. The derived table is a plain ``SELECT
DISTINCT`` (no ``DISTINCT ON``, grouping, window, ``ORDER BY``, ``LIMIT`` or star). A volatile call anywhere
leaves the select as it is. A select without a ``WHERE`` is left to ``_unwrap_distinct_projection``.
"""

from __future__ import annotations

from sqlglot import exp

_OUTER = ("group", "having", "distinct", "order", "limit", "offset", "qualify", "windows", "with_", "with", "joins", "laterals", "pivots", "into", "locks", "sample")
_INNER = ("group", "having", "order", "limit", "offset", "qualify", "windows", "with_", "with", "into", "locks", "sample")


def lift_source_distinct(select: exp.Select) -> exp.Expression | None:
    from .counted_membership import _unstable

    from_ = select.args.get("from_") or select.args.get("from")
    where = select.args.get("where")
    if from_ is None or where is None or any(select.args.get(k) for k in _OUTER):
        return None
    source = from_.this
    if not isinstance(source, exp.Subquery) or not source.alias or not isinstance(source.this, exp.Select) or any(source.args.get(k) for k in ("pivots", "sample", "joins")):
        return None
    if isinstance(source.args.get("alias"), exp.TableAlias) and source.args["alias"].args.get("columns"):
        return None
    inner = source.this
    distinct = inner.args.get("distinct")
    if distinct is None or distinct.args.get("on") or any(inner.args.get(k) for k in _INNER):
        return None
    if any(isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)) for item in inner.expressions):
        return None
    if any(node.find_ancestor(exp.Select) is inner for node in inner.find_all(exp.Window, exp.AggFunc)):
        return None
    names = [item.alias_or_name.lower() for item in inner.expressions]
    if "" in names or len(set(names)) != len(names):
        return None
    alias = source.alias.lower()
    picked = []
    for item in select.expressions:
        column = item.unalias()
        if not isinstance(column, exp.Column) or isinstance(column.this, exp.Star) or (column.table and column.table.lower() != alias) or column.args.get("db"):
            return None
        picked.append(column.name.lower())
    if sorted(picked) != sorted(names):
        return None
    if any(node.find_ancestor(exp.Select) is select for node in where.find_all(exp.Window, exp.AggFunc)) or _unstable(select):
        return None
    result = select.copy()
    result.set("distinct", exp.Distinct())
    (result.args.get("from_") or result.args.get("from")).this.this.set("distinct", None)
    return result
