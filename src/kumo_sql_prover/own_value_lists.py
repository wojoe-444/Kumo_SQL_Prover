"""The values of a table that a list drawn from the same table keeps are the list itself.

    SELECT DISTINCT viewer FROM views
    WHERE viewer IN (SELECT viewer FROM views GROUP BY viewer, day HAVING COUNT(DISTINCT article) > 1)

lists every viewer that is in the subquery's list, and each value of that list is a ``viewer`` of some row of
``views``. So the outer select adds nothing but ``DISTINCT`` and the fact that ``IN`` is never TRUE for a NULL:

    SELECT DISTINCT viewer FROM views WHERE viewer IS NOT NULL GROUP BY viewer, day HAVING COUNT(DISTINCT article) > 1

**Why it holds.** Let ``Q`` be the subquery. The outer select returns the distinct ``r.c`` over the rows ``r`` of
``T`` for which ``r.c IN (Q)`` is TRUE, that is, the values of ``Q`` that are not NULL and that some row of ``T``
holds in ``c``. Every value of ``Q`` is the ``c`` of a row of ``T`` (a row that passes ``Q``'s ``WHERE``, or the
``c`` of a group of such rows when ``c`` is a grouping column), so "some row of ``T`` holds it" is always true.
What is left is the distinct non-NULL values of ``Q``. A NULL is removed by ``c IS NOT NULL`` in ``Q``'s ``WHERE``:
``c`` is a column of the row, or a grouping column, whose NULL rows form the groups with a NULL there. Several
columns work the same way (a tuple is in a list only when every component is equal, so none is NULL), and the
outer select may list only some of the tested columns.

**What is read.** A ``SELECT DISTINCT`` of plain columns over one base table and nothing else, whose ``WHERE`` is
exactly one ``IN`` of a column (or a tuple of columns) of that table, each listed column being a tested one. The
subquery reads the same base table alone and selects the tested columns themselves, in the same order; it may
have a ``WHERE``, and a ``GROUP BY`` on plain columns that include the selected ones, with any ``HAVING``. The
normalizer's own spelling of a grouped subquery (a select from the grouped derived table with the ``HAVING`` test
in its ``WHERE``) is read too.

**Left alone.** A second condition on the outer select (the list would have to pass it too); ``NOT IN``; an outer
join, a second source, grouping, a window or a row limit on either select; a subquery over another table, with a
join or a nested query, or that reads the outer row; a selected expression that is not the plain column; a name
whose owner is not certain.

``tests/test_own_value_lists.py`` replays the rewrite on DuckDB with NULLs and repeated rows.
"""

from __future__ import annotations

from sqlglot import exp

from .own_group_membership import _from, _listed, _own_column, _plain_table, _unparen

_OUTER_BANNED = ("joins", "group", "having", "qualify", "windows", "with_", "with", "limit", "offset", "order", "laterals", "pivots", "into", "locks", "sample", "cluster", "distribute", "sort", "connect", "match", "kind", "hint")
_LIST_BANNED = ("joins", "qualify", "windows", "with_", "with", "limit", "offset", "order", "laterals", "pivots", "into", "locks", "sample", "cluster", "distribute", "sort", "connect", "match", "kind", "hint")


def _grouped_alone(query: exp.Select, schema: dict | None) -> tuple[exp.Table, set[str], list[str] | None] | None:
    """``(table, its columns, grouping columns or None)`` for a select over one base table that reads nothing else."""

    table = _from(query)
    if not _plain_table(table) or any(query.args.get(k) for k in _LIST_BANNED):
        return None
    distinct = query.args.get("distinct")
    if distinct is not None and distinct.args.get("on"):
        return None
    listed = _listed(table, schema)
    if listed is None:
        return None
    alias = table.alias_or_name.lower()
    if any(isinstance(n, (exp.Query, exp.Subquery, exp.Window)) and n is not query for n in query.walk()):
        return None
    for column in query.find_all(exp.Column):
        if isinstance(column.this, exp.Star) or _own_column(column, alias, listed) is None:
            return None  # another owner, a quoted name, or a read of the outer row
    group = query.args.get("group")
    if group is None:
        if query.args.get("having") is not None or query.find(exp.AggFunc) is not None:
            return None
        return table, listed, None
    if any(group.args.get(k) for k in ("grouping_sets", "rollup", "cube", "totals", "all")):
        return None
    keys = [_own_column(g, alias, listed) for g in group.expressions]
    if None in keys:
        return None
    return table, listed, keys


def read_own_value_list(select: exp.Select, schema: dict | None) -> exp.Select | None:
    """``select`` read as its subquery's own list, when the list is drawn from the column it is tested against."""

    distinct = select.args.get("distinct")
    where = select.args.get("where")
    if not schema or distinct is None or distinct.args.get("on") or where is None or any(select.args.get(k) for k in _OUTER_BANNED):
        return None
    row = _from(select)
    node = _unparen(where.this)
    if not _plain_table(row) or not isinstance(node, exp.In) or node.args.get("expressions") or node.args.get("unnest"):
        return None
    query = node.args.get("query")
    query = query.this if isinstance(query, exp.Subquery) and not query.alias else query
    if not isinstance(query, exp.Select):
        return None
    mine = _listed(row, schema)
    if mine is None:
        return None
    tested_node = _unparen(node.this)
    tested = [_own_column(v, row.alias_or_name.lower(), mine) for v in (tested_node.expressions if isinstance(tested_node, exp.Tuple) else [tested_node])]
    if None in tested or len(set(tested)) != len(tested) or len(query.expressions) != len(tested):
        return None
    outputs: list[tuple[str, str]] = []
    for item in select.expressions:
        name = _own_column(item.unalias(), row.alias_or_name.lower(), mine)
        if name is None or name not in tested or any(name == seen for seen, _ in outputs):
            return None
        outputs.append((name, item.alias if isinstance(item, exp.Alias) else item.unalias().name))
    source = _from(query)
    derived = source if isinstance(source, exp.Subquery) and isinstance(source.this, exp.Select) and source.alias else None
    inner = derived.this if derived is not None else query
    read = _grouped_alone(inner, schema)
    if read is None:
        return None
    table, listed, keys = read
    if table.name != row.name or bool(table.this.args.get("quoted")) != bool(row.this.args.get("quoted")):
        return None
    alias = table.alias_or_name.lower()
    result = query.copy()
    if derived is None:
        picked = [_own_column(o.unalias(), alias, listed) for o in query.expressions]
        if picked != tested or (keys is not None and any(c not in keys for c in tested)):
            return None
        present = [o.unalias() for o in query.expressions]
    else:
        # ``SELECT g.k FROM (SELECT c AS k, COUNT(..) AS a FROM T GROUP BY ..) AS g WHERE <test on g>``
        if keys is None or inner.args.get("having") is not None or inner.args.get("distinct") is not None or any(query.args.get(k) for k in (*_LIST_BANNED, "group", "having")):
            return None
        named = {o.alias.lower(): o.unalias() for o in inner.expressions if isinstance(o, exp.Alias)}
        if len(named) != len(inner.expressions):
            return None
        for column in query.find_all(exp.Column):
            if column.find_ancestor(exp.Subquery) is derived:
                continue
            if not isinstance(column.this, exp.Identifier) or column.table.lower() != derived.alias.lower() or column.name.lower() not in named:
                return None
        if any(isinstance(n, (exp.Query, exp.Subquery, exp.AggFunc, exp.Window)) and n is not query and n is not derived and n.find_ancestor(exp.Subquery) is not derived for n in query.walk()):
            return None
        picked = []
        for output in query.expressions:
            column = _unparen(output.unalias())
            behind = named.get(column.name.lower()) if isinstance(column, exp.Column) else None
            picked.append(_own_column(behind, alias, listed) if behind is not None else None)
        if picked != tested or any(c not in keys for c in tested):
            return None
        present = [o.unalias() for o in query.expressions]
    by_name = dict(zip(tested, result.expressions))
    # each output keeps the name the outer select gave it
    result.set("expressions", [by_name[name].unalias().copy() if by_name[name].unalias().name == given else exp.alias_(by_name[name].unalias().copy(), given, quoted=False) for name, given in outputs])
    filters = [exp.Not(this=exp.Is(this=p.copy(), expression=exp.Null())) for p in present]
    existing = result.args.get("where")
    result.set("where", exp.Where(this=exp.and_(*([existing.this] if existing is not None else []), *filters, copy=False)))
    result.set("distinct", exp.Distinct())
    return result
