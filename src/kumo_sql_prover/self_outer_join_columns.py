"""The distinct join columns of a table outer-joined with itself on one column are that column's distinct values.

    SELECT DISTINCT e.m, d.m FROM t AS e FULL JOIN t AS d ON e.m = d.m        SELECT DISTINCT e.m, e.m FROM t AS e

The rows of the join, read through the two join columns only:

* matched pairs: ``e.m = d.m`` is TRUE, so both columns hold the same non-NULL value ``v``. Every non-NULL value ``v`` of
  ``t.m`` appears, because the row that holds it matches itself: both sides are the same table, so that row is on
  the other side too. That gives ``(v, v)`` for exactly the non-NULL values of ``t.m``;
* a row of one side without a match: a row with a non-NULL ``m`` matches itself, so only rows whose ``m`` is NULL are
  unmatched. Such a row is kept (by ``LEFT``/``FULL`` for the left side, by ``RIGHT``/``FULL`` for the right side) with
  the other side padded with NULLs: ``(NULL, NULL)``, its own ``m`` being NULL as well. Under ``LEFT`` or ``RIGHT`` the
  unmatched rows of the other side are dropped, but they would only have added the same ``(NULL, NULL)``, and the
  kept side has a NULL ``m`` exactly when the dropped side does (one table).

So the set of pairs is ``{(v, v)}`` over all values ``v`` of ``t.m``, NULL included when a row holds a NULL ``m``: the
set ``SELECT DISTINCT m, m FROM t`` returns. ``DISTINCT`` makes both sides sets, so how often a pair occurs in the
join (``n * n`` times for a value held by ``n`` rows) does not matter. Over an empty table both are empty.

An inner join is not read: it has no ``(NULL, NULL)`` row. Two join columns are not read either: a row with one
of them NULL is unmatched and shows its other, non-NULL column next to a padded NULL.

Calcite's aggregate-join rules, and LLM rewriters after them, write the right-hand form for the left-hand one
(spelled ``GROUP BY e.m, d.m``, which ``keyed_only_grouping`` reads as ``DISTINCT`` first).

Conditions, all checked:

* ``SELECT DISTINCT`` (no ``DISTINCT ON``) with no ``WHERE``, ``GROUP BY``, ``HAVING``, ``QUALIFY``, ``ORDER BY``, ``LIMIT``,
  ``OFFSET`` or named window;
* the select reads exactly ``t AS e <LEFT|RIGHT|FULL> JOIN t AS d ON e.m = d.m``: two references to one table (the
  same written name, no column list, no sampling or other modifier) under two different aliases, one join, whose
  ``ON`` is that single equality between the same column of the two aliases (either way round), no ``USING``;
* every output is a plain column, ``e.m`` or ``d.m``, with or without an alias. ``d.m`` is then written ``e.m``; an
  output keeps its name (``d.m`` and ``e.m`` are both called ``m``).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY

_BANNED = ("where", "group", "having", "qualify", "order", "limit", "offset", "windows", "laterals", "with_", "with")
_TABLE_ARGS = frozenset({"this", "db", "catalog", "alias"})


def _table_name(source: exp.Expression) -> tuple[str, ...] | None:
    if not isinstance(source, exp.Table) or not isinstance(source.this, exp.Identifier):
        return None
    if any(v for k, v in source.args.items() if k not in _TABLE_ARGS):
        return None
    alias = source.args.get("alias")
    if isinstance(alias, exp.TableAlias) and alias.args.get("columns"):
        return None
    return tuple(part.sql() for part in source.parts)


def _plain_column(node: exp.Expression) -> exp.Column | None:
    if isinstance(node, exp.Column) and isinstance(node.this, exp.Identifier) and node.table and len(node.parts) == 2:
        return node
    return None


def distinct_self_outer_join_column(select: exp.Select) -> exp.Select | None:
    """``select`` over the one table when it is the distinct join columns of a self outer join (module doc), else None."""

    distinct = select.args.get("distinct")
    if distinct is None or distinct.args.get("on") or any(select.args.get(key) for key in _BANNED):
        return None
    from_ = select.args.get(FROM_KEY)
    joins = select.args.get("joins") or []
    if from_ is None or len(joins) != 1:
        return None
    join = joins[0]
    if (join.args.get("side") or "").upper() not in ("LEFT", "RIGHT", "FULL"):
        return None
    if any(v for k, v in join.args.items() if k not in ("this", "side", "kind", "on")) or (join.args.get("kind") or "OUTER").upper() != "OUTER":
        return None
    left, right = from_.this, join.this
    name = _table_name(left)
    if name is None or name != _table_name(right):
        return None
    kept, dropped = left.alias_or_name, right.alias_or_name
    if not kept or not dropped or kept.lower() == dropped.lower():
        return None
    on = join.args.get("on")
    while isinstance(on, exp.Paren):
        on = on.this
    if not isinstance(on, exp.EQ):
        return None
    first, second = _plain_column(on.this), _plain_column(on.expression)
    if first is None or second is None or first.this.sql() != second.this.sql() or {first.table, second.table} != {kept, dropped}:
        return None
    column = first.this.sql()
    for item in select.expressions:
        value = _plain_column(item.this if isinstance(item, exp.Alias) else item)
        if value is None or value.this.sql() != column or value.table not in (kept, dropped):
            return None
    copy = select.copy()
    copy.set("joins", None)
    for item in copy.expressions:
        value = item.this if isinstance(item, exp.Alias) else item
        if value.table == dropped:
            value.set("table", exp.to_identifier(kept, quoted=left.args["alias"].this.quoted if left.args.get("alias") else left.this.quoted))
    return copy
