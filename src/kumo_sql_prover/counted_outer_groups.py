"""A keyed table outer-joined with a table of counts per key is the count over the outer join of the rows.

    SELECT c.k, COALESCE(g.n, 0) AS orders
    FROM c LEFT JOIN (SELECT o.ck, COUNT(*) AS n FROM o WHERE o.note <> 'x' GROUP BY o.ck) AS g ON c.k = g.ck

is

    SELECT c.k, COUNT(g.id) AS orders
    FROM c LEFT JOIN o AS g ON c.k = g.ck AND g.note <> 'x'
    GROUP BY c.k

(TPC-H Q13 and the rewrites that count each customer's orders before the join). The prover reads the second form;
the first kept its grouped table and the pair stayed unproved.

Why this is exact. ``c`` has a declared key, which is unique and not NULL, so grouping the outer join by it gives one
group per row of ``c``, as the first form gives one row per row of ``c`` (the counts table has at most one row per
value of its key, and ``WHERE`` reads ``c`` alone, so it keeps or drops a row of ``c`` with all its matches). For
a row of ``c``, the counts table holds a row for its value exactly when some row of ``o`` passes the filter with
that value, and its count is the number of them (of them with a value in ``x``, for ``COUNT(x)``); otherwise the
outer join gives NULL and ``COALESCE`` gives 0. In the second form the same rows of ``o`` are the matches of the
join, ``COUNT`` counts them through a column that is not NULL on a match (the joined column, or a declared key
column of ``o``; ``x`` itself for ``COUNT(x)``), and with no match the one NULL-extended row counts 0.

Conditions, all checked:

* the select has only a select list, ``FROM``, one join and ``WHERE``: no ``DISTINCT``, grouping, tail, window,
  aggregate or nested query of its own;
* ``FROM`` is a table the schema lists, with a declared key; the join is ``LEFT [OUTER] JOIN`` of a derived table
  with an alias and no column list, ``ON`` one equality between a column of the table and the derived table's key;
* the derived table is ``SELECT key, COUNT(*)`` or ``SELECT key, COUNT(x)`` (in either order, ``x`` a column, no
  ``DISTINCT``) over one table the schema lists, with an optional ``WHERE`` without a nested query, grouped by that
  one key column, and nothing else; its columns are bare or carry its table's name;
* every column of the select names its source; the derived table is read only in ``ON`` and as
  ``COALESCE(count, 0)`` in the select list;
* nothing volatile anywhere.

This is a pre-pass of ``normalize``: later steps lift ``COALESCE`` out of a derived table, after which the two
halves are no longer in one select.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY
from .keyed_join_groups import _and_all, declared_columns, key_of
from .unmatched_outer_join import _columns_of, _conjuncts, _inside, _plain_table, _unwrap

_OUTER_ARGS = frozenset({"expressions", "from_", "from", "joins", "where"})
_GROUPED_ARGS = frozenset({"expressions", "from_", "from", "where", "group"})


def _zero(node: exp.Expression) -> bool:
    return type(node) is exp.Literal and not node.is_string and node.this == "0"


def _counts(source: exp.Expression, declared: dict[str, set[str]]) -> dict | None:
    """What a derived table of counts per key holds: its table, key column, key output name, count output name and
    counted column (``None`` for ``COUNT(*)``), and its filter over the table's bare columns."""

    if type(source) is not exp.Subquery or not source.alias or type(source.this) is not exp.Select:
        return None
    if source.args["alias"].args.get("columns") or any(v for key, v in source.args.items() if key not in ("this", "alias")):
        return None
    inner = source.this
    if any(value for key, value in inner.args.items() if key not in _GROUPED_ARGS):
        return None
    table = inner.args.get(FROM_KEY).this if inner.args.get(FROM_KEY) is not None else None
    if type(table) is not exp.Table or not _plain_table(table):
        return None
    columns = _columns_of(table, declared)
    if not columns:
        return None
    name = (table.alias_or_name or "").lower()

    def own(column: exp.Expression) -> str | None:
        """The table column a plain column of the derived table reads, or ``None``."""

        if type(column) is not exp.Column or isinstance(column.this, exp.Star) or len(column.parts) > 2:
            return None
        if column.table and column.table.lower() != name:
            return None
        return column.name.lower() if column.name.lower() in columns else None

    group = inner.args.get("group")
    if group is None or len(group.expressions) != 1 or any(v for key, v in group.args.items() if key != "expressions"):
        return None
    key = own(group.expressions[0])
    if key is None or len(inner.expressions) != 2:
        return None
    key_name = count_name = counted = None
    for item in inner.expressions:
        value = item.this if isinstance(item, exp.Alias) else item
        if type(value) is exp.Count and isinstance(item, exp.Alias) and count_name is None:
            if any(v for k, v in value.args.items() if k not in ("this", "big_int")):
                return None
            if not isinstance(value.this, exp.Star):
                counted = own(value.this)
                if counted is None:
                    return None
            count_name = item.alias.lower()
        elif own(value) == key and key_name is None:
            key_name = (item.alias_or_name or "").lower()
        else:
            return None
    if not key_name or not count_name or key_name == count_name:
        return None
    where = inner.args.get("where")
    parts = [part.copy() for part in _conjuncts(where.this if where is not None else None)]
    for part in parts:
        for node in part.walk():
            if isinstance(node, (exp.Query, exp.Subquery, exp.AggFunc, exp.Window)):
                return None
            if type(node) is exp.Column and own(node) is None:
                return None
    return {"table": table, "key": key, "key_name": key_name, "count_name": count_name, "counted": counted, "filter": parts}


def _counted_outer_group(select: exp.Select, declared: dict[str, set[str]], keys: dict) -> exp.Select | None:
    from .counted_membership import _unstable

    if any(value for key, value in select.args.items() if key not in _OUTER_ARGS):
        return None
    base = select.args.get(FROM_KEY).this if select.args.get(FROM_KEY) is not None else None
    joins = select.args.get("joins") or []
    if type(base) is not exp.Table or not _plain_table(base) or len(joins) != 1:
        return None
    base_columns = _columns_of(base, declared)
    key = key_of(base, keys, base_columns) if base_columns else None
    if key is None:
        return None
    join = joins[0]
    if (join.args.get("side") or "").upper() != "LEFT" or (join.args.get("kind") or "").upper() not in ("", "OUTER"):
        return None
    if any(value for k, value in join.args.items() if k not in ("this", "side", "kind", "on")) or join.args.get("on") is None:
        return None
    source = join.this
    counts = _counts(source, declared)
    if counts is None or _unstable(select):
        return None
    left, right = (base.alias_or_name or "").lower(), source.alias.lower()
    if not left or left == right:
        return None
    if any(isinstance(n, (exp.AggFunc, exp.Window)) and not _inside(n, source) for n in select.walk()):
        return None
    if any(not _inside(n, source) for n in select.find_all(exp.Select) if n is not select):
        return None

    def side(column: exp.Column) -> str | None:
        if isinstance(column.this, exp.Star) or len(column.parts) != 2:
            return None
        table = column.table.lower()
        if table == left and column.name.lower() in base_columns:
            return left
        if table == right and column.name.lower() in (counts["key_name"], counts["count_name"]):
            return right
        return None

    on = _unwrap(join.args["on"])
    if type(on) is not exp.EQ or type(_unwrap(on.this)) is not exp.Column or type(_unwrap(on.expression)) is not exp.Column:
        return None
    ends = {side(_unwrap(on.this)): _unwrap(on.this), side(_unwrap(on.expression)): _unwrap(on.expression)}
    if set(ends) != {left, right} or ends[right].name.lower() != counts["key_name"]:
        return None
    read: list[str] = []
    for column in select.find_all(exp.Column):
        if _inside(column, source) or _inside(column, join.args["on"]):
            continue
        owner = side(column)
        if owner is None:
            return None
        if owner == left:
            if not _inside(column, select.args.get("where")) and column.name.lower() not in read:
                read.append(column.name.lower())
            continue
        parent = column.parent
        if column.name.lower() != counts["count_name"] or type(parent) is not exp.Coalesce or parent.this is not column:
            return None
        if len(parent.expressions) != 1 or not _zero(parent.expressions[0]) or _inside(column, select.args.get("where")):
            return None
    if any(isinstance(n, exp.Star) for item in select.expressions for n in item.walk()):
        return None

    alias = exp.to_identifier(source.alias)
    inner_key = key_of(counts["table"], keys, _columns_of(counts["table"], declared))
    counted = counts["counted"] or (inner_key[0] if inner_key else counts["key"])
    copy = select.copy()
    new_join = copy.args["joins"][0]
    new_source = new_join.this
    for column in list(copy.find_all(exp.Column)):
        if _inside(column, new_source) or column.table.lower() != right:
            continue
        if _inside(column, new_join.args["on"]):
            column.replace(exp.column(counts["key"], table=alias.copy()))
        else:
            column.parent.replace(exp.Count(this=exp.column(counted, table=alias.copy())))
    conditions = [new_join.args["on"].copy()]
    for part in counts["filter"]:
        for column in part.find_all(exp.Column):
            column.set("table", alias.copy())
        conditions.append(part)
    table = counts["table"].copy()
    table.set("alias", exp.TableAlias(this=alias.copy()))
    new_join.set("this", table)
    new_join.set("on", _and_all(conditions))
    grouped = list(key) + [name for name in read if name not in key]
    base_alias = base.args.get("alias")
    owner = (base_alias.this if base_alias is not None else base.this).copy()
    copy.set("group", exp.Group(expressions=[exp.column(name, table=owner.copy()) for name in grouped]))
    return copy


def count_outer_groups(tree: exp.Expression, schema: dict | None, keys: dict | None) -> exp.Expression:
    """``tree`` with every such outer join with a table of counts written as a count over the outer join."""

    if not schema or not keys:
        return tree
    declared = declared_columns(schema)
    for select in list(tree.find_all(exp.Select)):
        root = select
        while root is not None and root is not tree:
            root = root.parent
        if root is None:
            continue  # replaced along with an enclosing select
        new = _counted_outer_group(select, declared, keys)
        if new is None:
            continue
        if select is tree:
            tree = new
        else:
            select.replace(new)
    return tree
