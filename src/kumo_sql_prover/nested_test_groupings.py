"""A ``GROUP BY`` that lists exactly its outputs, around joined tables and a nested test, is ``SELECT DISTINCT``.

    SELECT a.item_id, b.label FROM sale AS a, item AS b
    WHERE a.item_id = b.item_id AND a.item_id NOT IN (SELECT k.item_id FROM sale AS k WHERE k.paid > 5)
    GROUP BY a.item_id, b.label

has no aggregate and no ``HAVING``: the grouping only removes repeated rows, one row per distinct pair of values
(NULLs form one group as they form one distinct value). It is

    SELECT DISTINCT a.item_id, b.label FROM sale AS a, item AS b WHERE ..

``keyed_only_grouping`` reads this over an outer join and ``_group_by_to_distinct`` over a derived union; the
solver reads the rest by itself. What it cannot do by itself is hand the select to the rules that read ``DISTINCT``
over joins with a nested test (``keyed_semijoin``, ``distinct_match_joins``), so this rule covers exactly that
case: joined sources, and a nested query in the ``WHERE``.

**What is read.** A select with a plain ``GROUP BY`` (no ``ROLLUP``, ``CUBE``, grouping sets, ``ALL``) whose keys
are plain columns and whose outputs are exactly those columns as written (renamed or not), with no aggregate,
window, ``HAVING``, ``DISTINCT``, ``ORDER BY``, row limit or ``QUALIFY``; at least one join, or one derived
table that joins (which ``distinct_derived_joins`` then reads as its join); a nested query in its ``WHERE``.

A narrow retry rule (``shape_fallback``). ``tests/test_nested_test_groupings.py`` replays the rewrite on DuckDB.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import extended_grouping

_BANNED = ("having", "distinct", "qualify", "windows", "with_", "with", "limit", "offset", "order", "laterals", "pivots", "into", "locks", "sample", "cluster", "distribute", "sort", "connect", "match")


def group_as_distinct(select: exp.Select) -> exp.Select | None:
    """``select`` as ``SELECT DISTINCT`` when its grouping only removes repeated rows (module doc), else None."""

    group = select.args.get("group")
    where = select.args.get("where")
    if group is None or not group.expressions or extended_grouping(group) or any(group.args.get(key) for key in ("all", "totals")):
        return None
    from_ = select.args.get("from_") or select.args.get("from")
    joined = bool(select.args.get("joins")) or (from_ is not None and isinstance(from_.this, exp.Subquery) and isinstance(from_.this.this, exp.Select) and bool(from_.this.this.args.get("joins")))
    if where is None or where.find(exp.Select) is None or not joined or any(select.args.get(key) for key in _BANNED):
        return None
    keys = list(group.expressions)
    if any(not isinstance(key, exp.Column) or isinstance(key.this, exp.Star) for key in keys):
        return None
    if any(isinstance(node, (exp.AggFunc, exp.Window)) and node.find_ancestor(exp.Select) is select for node in select.walk()):
        return None
    outputs = [item.this if isinstance(item, exp.Alias) else item for item in select.expressions]
    if any(not isinstance(value, exp.Column) or isinstance(value.this, exp.Star) for value in outputs):
        return None
    def text(column: exp.Column) -> str:
        # a quoted name can differ from another by case alone
        return column.sql() if any(part.args.get("quoted") for part in column.find_all(exp.Identifier)) else column.sql().lower()

    if {text(key) for key in keys} != {text(value) for value in outputs}:
        return None
    result = select.copy()
    result.set("group", None)
    result.set("distinct", exp.Distinct())
    return result
