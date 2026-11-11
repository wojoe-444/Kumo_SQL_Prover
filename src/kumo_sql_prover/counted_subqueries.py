"""A correlated ``COUNT`` subquery in a select list is a left join to the counts per key.

    SELECT o.a, (SELECT COUNT(*) FROM t AS x WHERE x.k = o.id AND x.kind = 2) AS n FROM o
    is
    SELECT o.a, COALESCE(kqc0.kqv, 0) AS n
    FROM o LEFT JOIN (SELECT x.k AS kqk0, COUNT(*) AS kqv FROM t AS x WHERE x.kind = 2 GROUP BY x.k) AS kqc0 ON o.id = kqc0.kqk0

``algebraic_equivalence._decorrelate_select_list`` reads the other aggregates this way and leaves ``COUNT`` out,
because a count over no rows is 0 and not NULL. The ``COALESCE`` is what the count needs. A rewriter that removes
subqueries (Calcite's ``SubQueryRemoveRule``, or a model that writes the join by hand) prints the join, and the two
spellings were not compared at all.

Why it holds, for one row of the outer select:

* ``o.id`` is NULL: the subquery counts the rows with ``x.k = NULL``, none, and reads 0. The join condition is not
  true for any row of the derived table, the left join pads the row with NULLs and ``COALESCE`` reads 0.
* ``o.id`` is not NULL and no row of ``t`` passes the filter with that key: 0 on both sides, as above.
* otherwise the derived table has exactly one row with ``kqk0 = o.id`` (``GROUP BY`` makes each key value one row,
  and the group of NULL keys never satisfies the equality), holding the count of exactly the rows the subquery
  counts. The join keeps the outer row once, with that count, which is never NULL.

So the left join neither drops nor repeats a row of the outer select (duplicates of the outer row stay duplicates),
and the value is the subquery's on every row. With several correlation equalities the derived table is grouped by
all of them and the same argument holds for the tuple. ``COUNT(x.c)`` counts the rows with ``x.c`` not NULL in both
forms.

Rewritten only when (the select is left as written otherwise):

* the outer select has a ``FROM``, no ``GROUP BY``, ``HAVING``, ``DISTINCT``, ``QUALIFY``, aggregate or window of
  its own (the join is added after its other joins, so those would be computed over the same rows, but the rule
  stays with the plain case), no ``*`` in its select list (a star would also list the new columns), no ``USING``
  or ``NATURAL`` join, and every one of its sources has a distinct alias;
* the subquery stands in a select-list item, outside any other subquery, and is
  ``SELECT COUNT(*) | COUNT(x.c) FROM t [AS x] [WHERE ...]``: one base table, no join, no ``DISTINCT``, ``FILTER``,
  ``GROUP BY``, ``HAVING``, ``ORDER BY``, ``LIMIT`` or nested subquery, and no function whose value changes from call
  to call;
* the subquery's table alias is not an alias of the outer select, and every column in the subquery is qualified,
  by that alias (a column of ``t``) or by an alias of the outer select (a column of the outer row);
* each conjunct of the subquery's ``WHERE`` reads columns of ``t`` only (it stays in the derived table) or is an
  equality between one column of ``t`` and one column of the outer row (a key); there is at least one key.
"""

from __future__ import annotations

from sqlglot import exp

VALUE = "kqv"
_UNSTABLE = (exp.Rand, exp.Randn, exp.Uuid, exp.CurrentTimestamp, exp.CurrentDate, exp.CurrentTime, exp.CurrentDatetime, exp.Anonymous)
_OUTER_ARGS = frozenset({"expressions", "from_", "from", "joins", "where", "order", "limit", "offset", "kind"})
_INNER_ARGS = frozenset({"expressions", "from_", "from", "where", "kind"})


def _from(select: exp.Select) -> exp.From | None:
    return select.args.get("from_") or select.args.get("from")


def _conjuncts(node: exp.Expression) -> list[exp.Expression]:
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _own(select: exp.Select, kind) -> bool:
    return any(node.find_ancestor(exp.Select) is select for node in select.find_all(kind))


def _outer_aliases(select: exp.Select) -> set[str] | None:
    source = _from(select)
    if source is None:
        return None
    aliases: list[str] = []
    for item in [source.this] + [join.this for join in select.args.get("joins") or []]:
        if not isinstance(item, (exp.Table, exp.Subquery)) or not item.alias_or_name:
            return None
        aliases.append(item.alias_or_name.lower())
    for join in select.args.get("joins") or []:
        if join.args.get("using") or join.args.get("method") or join.args.get("kind") == "NATURAL":
            return None
    return set(aliases) if len(set(aliases)) == len(aliases) else None


def _reading(subquery: exp.Subquery, outer: set[str]):
    """``(table, alias, count, local conjuncts, [(inner key column, outer column)])`` of a subquery the rule reads."""

    inner = subquery.this
    if not isinstance(inner, exp.Select) or any(value for key, value in subquery.args.items() if key != "this"):
        return None
    if any(value for key, value in inner.args.items() if key not in _INNER_ARGS) or len(inner.expressions) != 1:
        return None
    source = _from(inner)
    table = source.this if source is not None else None
    if not isinstance(table, exp.Table) or any(value for key, value in table.args.items() if key not in ("this", "alias", "db", "catalog")):
        return None
    alias_node = table.args.get("alias")
    if alias_node is not None and alias_node.args.get("columns"):
        return None
    alias = table.alias_or_name.lower()
    if not alias or alias in outer:
        return None
    count = inner.expressions[0]
    count = count.this if isinstance(count, exp.Alias) else count
    if type(count) is not exp.Count or any(value for key, value in count.args.items() if key not in ("this", "big_int")):
        return None
    if not isinstance(count.this, exp.Star) and not (isinstance(count.this, exp.Column) and count.this.table.lower() == alias):
        return None
    for node in inner.walk():
        if node is not inner and isinstance(node, (exp.Query, exp.Subquery, exp.Exists, exp.Window, *_UNSTABLE)):
            return None
        if isinstance(node, exp.AggFunc) and node is not count:
            return None
        if isinstance(node, exp.Column) and (isinstance(node.this, exp.Star) or node.args.get("db") or node.table.lower() not in outer | {alias}):
            return None
    local: list[exp.Expression] = []
    keys: list[tuple[exp.Column, exp.Column]] = []
    where = inner.args.get("where")
    for condition in _conjuncts(where.this) if where is not None else []:
        tables = {column.table.lower() for column in condition.find_all(exp.Column)}
        if tables <= {alias}:
            local.append(condition)
            continue
        if type(condition) is not exp.EQ or not isinstance(condition.this, exp.Column) or not isinstance(condition.expression, exp.Column):
            return None
        a, b = condition.this, condition.expression
        if b.table.lower() == alias:
            a, b = b, a
        if a.table.lower() != alias or b.table.lower() not in outer:
            return None
        keys.append((a, b))
    return (table, alias, count, local, keys) if keys else None


def _free_alias(select: exp.Select) -> str:
    used = {node.alias_or_name.lower() for node in select.root().walk() if isinstance(node, (exp.Table, exp.Subquery)) and node.alias_or_name}
    number = 0
    while f"kqc{number}" in used:
        number += 1
    return f"kqc{number}"


def count_subquery_as_join(select: exp.Select) -> exp.Select | None:
    """``select`` with one correlated ``COUNT`` subquery of its select list read as a left join, or None."""

    if not isinstance(select, exp.Select) or any(value for key, value in select.args.items() if key not in _OUTER_ARGS):
        return None
    candidates = [node for item in select.expressions for node in item.find_all(exp.Subquery) if node.find_ancestor(exp.Select) is select]
    if not candidates:
        return None
    if _own(select, exp.AggFunc) or _own(select, exp.Window) or _own(select, exp.Star):
        return None
    outer = _outer_aliases(select)
    if outer is None:
        return None
    for subquery in candidates:
        reading = _reading(subquery, outer)
        if reading is None:
            continue
        table, _, count, local, keys = reading
        name = _free_alias(select)
        derived = exp.Select(expressions=[exp.alias_(key.copy(), f"kqk{position}", quoted=False) for position, (key, _) in enumerate(keys)] + [exp.alias_(count.copy(), VALUE, quoted=False)])
        derived.set("from_" if select.args.get("from_") is not None else "from", exp.From(this=table.copy()))
        if local:
            condition = local[0].copy()
            for other in local[1:]:
                condition = exp.And(this=condition, expression=other.copy())
            derived.set("where", exp.Where(this=condition))
        derived.set("group", exp.Group(expressions=[key.copy() for key, _ in keys]))
        on = None
        for position, (_, column) in enumerate(keys):
            equality = exp.EQ(this=column.copy(), expression=exp.column(f"kqk{position}", table=name))
            on = equality if on is None else exp.And(this=on, expression=equality)
        index = [id(node) for node in select.find_all(exp.Subquery)].index(id(subquery))
        result = select.copy()
        list(result.find_all(exp.Subquery))[index].replace(exp.Coalesce(this=exp.column(VALUE, table=name), expressions=[exp.Literal.number(0)]))
        join = exp.Join(this=exp.Subquery(this=derived, alias=exp.TableAlias(this=exp.to_identifier(name))), side="LEFT", on=on)
        result.set("joins", list(result.args.get("joins") or []) + [join])
        return result
    return None
