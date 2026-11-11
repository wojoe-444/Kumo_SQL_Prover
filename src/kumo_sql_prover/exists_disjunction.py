"""``EXISTS (S WHERE p) OR EXISTS (S WHERE q)`` is ``EXISTS (S WHERE p OR q)``.

Two ``EXISTS`` tests over the same ``FROM`` (the same tables, aliases and joins, as text) that differ only in
their ``WHERE`` ask whether some row of that ``FROM`` passes ``p``, or some row passes ``q``. That is whether some
row passes ``p OR q``: ``EXISTS`` only sees rows for which the filter is TRUE, and ``p OR q`` is TRUE exactly when
one of them is. By De Morgan, ``NOT EXISTS (S WHERE p) AND NOT EXISTS (S WHERE q)`` is
``NOT EXISTS (S WHERE p OR q)``. A member without a ``WHERE`` passes every row, so the merged test has none.
The subqueries may be correlated: the identity holds for each outer row.

A subquery qualifies only when the existence of a row is all it decides: one ``SELECT`` with no ``GROUP BY``,
``HAVING``, ``QUALIFY``, ``LIMIT`` or ``OFFSET``, no aggregate in its select list (a global aggregate returns a
row whatever its filter keeps) and no window function. The select list of the first member is kept; ``EXISTS``
does not read it.
"""

from __future__ import annotations

from functools import reduce

from sqlglot import exp


def _chain(node: exp.Expression, kind: type) -> list[exp.Expression]:
    if type(node) is kind:
        return _chain(node.this, kind) + _chain(node.expression, kind)
    if isinstance(node, exp.Paren) and type(node.this) is kind:
        return _chain(node.this, kind)
    return [node]


def _existence(node: exp.Expression, negated: bool) -> exp.Select | None:
    """The subquery of an ``EXISTS`` (``NOT EXISTS`` when ``negated``) that only tests for a row, or ``None``."""

    while isinstance(node, exp.Paren):
        node = node.this
    if negated:
        if not isinstance(node, exp.Not):
            return None
        node = node.this
        while isinstance(node, exp.Paren):
            node = node.this
    if not isinstance(node, exp.Exists):
        return None
    query = node.this
    while isinstance(query, exp.Subquery) and not any(query.args.get(k) for k in ("alias", "limit", "offset", "order")):
        query = query.this
    if not isinstance(query, exp.Select):
        return None
    if any(query.args.get(key) for key in ("group", "having", "qualify", "limit", "offset", "distinct", "windows", "sample", "with", "with_", "laterals", "pivots")):
        return None
    if not (query.args.get("from_") or query.args.get("from")):
        return None
    if any(item.find(exp.AggFunc, exp.Window) for item in query.expressions):
        return None
    return query


def _source_key(query: exp.Select) -> str:
    shell = query.copy()
    shell.set("where", None)
    shell.set("expressions", [exp.Star()])
    shell.set("order", None)
    return shell.sql(dialect="bigquery", comments=False)


def _merge(parts: list[exp.Expression], negated: bool) -> list[exp.Expression] | None:
    groups: dict[str, list[int]] = {}
    queries: dict[int, exp.Select] = {}
    for index, part in enumerate(parts):
        query = _existence(part, negated)
        if query is not None:
            queries[index] = query
            groups.setdefault(_source_key(query), []).append(index)
    merged: dict[int, exp.Expression] = {}
    dropped: set[int] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        filters = [queries[i].args.get("where") for i in members]
        query = queries[members[0]].copy()
        query.set("order", None)
        if any(f is None for f in filters):
            query.set("where", None)
        else:
            conditions = [exp.Paren(this=f.this.copy()) for f in filters]
            query.set("where", exp.Where(this=reduce(lambda a, b: exp.Or(this=a, expression=b), conditions)))
        test: exp.Expression = exp.Exists(this=query)
        merged[members[0]] = exp.Not(this=test) if negated else test
        dropped.update(members[1:])
    if not merged:
        return None
    return [merged.get(i, part) for i, part in enumerate(parts) if i not in dropped]


def merge_exists_disjunction(select: exp.Select) -> exp.Expression | None:
    """``select`` with the ``EXISTS`` tests of each ``OR`` (and ``NOT EXISTS`` of each ``AND``) of its own scope that
    read the same ``FROM`` merged into one; ``None`` if nothing changed."""

    if not any(e.find_ancestor(exp.Select) is select for e in select.find_all(exp.Exists)):
        return None
    copy = select.copy()
    changed = False
    for kind, negated in ((exp.Or, False), (exp.And, True)):
        for node in list(copy.find_all(kind)):
            if type(node) is not kind or node.find_ancestor(exp.Select) is not copy:
                continue
            parent = node.parent
            while isinstance(parent, exp.Paren):
                parent = parent.parent
            if type(parent) is kind:
                continue  # only the top of a chain
            if node.parent is None:
                continue  # detached by an earlier replacement
            kept = _merge(_chain(node, kind), negated)
            if kept is None:
                continue
            node.replace(reduce(lambda a, b: kind(this=a, expression=b), [k.copy() for k in kept]))
            changed = True
    return copy if changed else None
