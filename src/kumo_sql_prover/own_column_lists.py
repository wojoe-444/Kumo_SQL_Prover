"""The values of a column that a list drawn from that column keeps, when other tests read the value only.

    SELECT DISTINCT buyer FROM sale
    WHERE buyer IN (SELECT s.buyer FROM sale AS s WHERE s.item_id IN (SELECT i.item_id FROM item AS i WHERE i.label = 'S'))
      AND buyer NOT IN (SELECT t.buyer FROM sale AS t WHERE t.item_id = 7)

lists the buyers that are in the first list and pass the second test. Every value of the first list is the
``buyer`` of a row of ``sale``, and the second test reads nothing of the outer row but ``buyer``. So it is the
first list itself, filtered:

    SELECT DISTINCT s.buyer AS buyer FROM sale AS s
    WHERE s.buyer IS NOT NULL AND s.item_id IN (SELECT i.item_id FROM item AS i WHERE i.label = 'S')
      AND s.buyer NOT IN (SELECT t.buyer FROM sale AS t WHERE t.item_id = 7)

**Why it holds.** Write ``F(u)`` for the list's ``WHERE`` on a row ``u`` of the table ``T`` and ``G(v)`` for the
other conjuncts of the outer ``WHERE`` as a test of the value ``v`` of the column. The outer select returns the
distinct ``v`` such that some row ``r`` has ``r.c = v``, the test ``v IN (list)`` is TRUE and ``G(v)`` is TRUE.
``v IN (list)`` is TRUE exactly when ``v`` is not NULL and some row ``u`` has ``u.c = v`` and ``F(u)``. Given such a
``u``, the row ``r`` can be ``u`` itself, so "some row ``r`` holds ``v``" adds nothing. What is left is the distinct
``u.c`` over the rows ``u`` with ``u.c`` not NULL, ``F(u)`` and ``G(u.c)``: the second select. No declared fact is
needed. ``own_value_lists`` reads the case with no other conjunct, and a grouped list; this rule reads a list
with nested queries in its ``WHERE`` and an outer select with further tests of the same column.

**What is read.** A ``SELECT DISTINCT`` over one base table with no join, grouping, window or tail, that lists
only one column ``c`` of it, whose ``WHERE`` has a conjunct ``c IN (list)`` where the list is a plain select of
``c`` from the same table alone (no join, grouping, aggregate, window or tail; its ``WHERE`` may hold nested
queries) that does not read the outer row. Every other conjunct reads no column of the outer row but ``c``, at
any depth, and nothing under the list's name (two nested sources of one name are first given two names). Bare columns of nested one-table queries are given their table when the caller supplied its columns;
after that every column of a nested query must name its source. ``nested_correlation.interchangeable`` must let
the column be read for itself (not a floating-point one). ``NOT IN`` as the list, a volatile call, a quoted
name: left alone.

A narrow retry rule (``shape_fallback``). ``tests/test_own_column_lists.py`` replays the rewrite on DuckDB.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import conjuncts
from .nested_correlation import interchangeable
from .own_group_membership import _from, _listed, _own_column, _plain_table, _unparen

_BANNED = ("joins", "group", "having", "qualify", "windows", "with_", "with", "limit", "offset", "order", "laterals", "pivots", "into", "locks", "sample", "cluster", "distribute", "sort", "connect", "match", "kind", "hint")


def _sources(query: exp.Select) -> list[exp.Expression]:
    first = _from(query)
    return ([first] if first is not None else []) + [join.this for join in query.args.get("joins") or []]


def read_own_column_list(select: exp.Select, schema: dict | None, types: dict | None = None) -> exp.Select | None:
    """``select`` read as the list one of its ``IN`` tests draws from its own column, or None."""

    from .counted_membership import _unstable
    from .keyed_semijoin import name_nested_sources, rename_nested_sources

    distinct = select.args.get("distinct")
    if not schema or distinct is None or distinct.args.get("on") or select.args.get("where") is None or any(select.args.get(key) for key in _BANNED):
        return None
    row = _from(select)
    if not _plain_table(row) or row.this.args.get("quoted") or not (row.alias_or_name or ""):
        return None
    name = row.alias_or_name.lower()
    # a name that two nested queries give their sources (two lists over ``sale AS s``) is made two names, so that
    # a test moved into a list cannot be answered by the list's own row
    seen = [(source.alias_or_name or "").lower() for query in select.find_all(exp.Select) if query is not select for source in _sources(query)]
    select = name_nested_sources(rename_nested_sources(select, {name} | {other for other in seen if seen.count(other) > 1}), schema)
    row, distinct = _from(select), select.args.get("distinct")
    mine = _listed(row, schema)
    if mine is None:
        return None
    nested = [query for query in select.find_all(exp.Select) if query is not select]
    if any((source.alias_or_name or "").lower() == name for query in nested for source in _sources(query)):
        return None  # a nested source still carries the outer row's name
    if any(isinstance(column.this, exp.Star) for column in select.find_all(exp.Column)) or select.find(exp.Star) is not None:
        return None
    if any(node.find_ancestor(exp.Select) is select for node in select.find_all(exp.AggFunc, exp.Window)):
        return None
    listed = {_own_column(item.unalias(), name, mine) for item in select.expressions}
    if len(listed) != 1 or None in listed:
        return None
    column = next(iter(listed))
    if not interchangeable(types, (row.name.lower(), column), (row.name.lower(), column)):
        return None
    parts = conjuncts(select.args["where"].this)
    for position, part in enumerate(parts):
        test = _unparen(part)
        if not isinstance(test, exp.In) or test.args.get("expressions") or test.args.get("unnest") or test.args.get("field"):
            continue
        query = test.args.get("query")
        query = query.this if isinstance(query, exp.Subquery) and not query.alias else query
        if not isinstance(query, exp.Select) or _own_column(test.this, name, mine) != column:
            continue
        inner_distinct = query.args.get("distinct")
        source = _from(query)
        if any(query.args.get(key) for key in _BANNED) or (inner_distinct is not None and inner_distinct.args.get("on")) or not _plain_table(source):
            continue
        if source.name.lower() != row.name.lower() or source.this.args.get("quoted") or any(source.args.get(key) or row.args.get(key) for key in ("db", "catalog")):
            continue
        theirs = source.alias_or_name.lower()
        if len(query.expressions) != 1 or _own_column(query.expressions[0].unalias(), theirs, mine) != column or not query.expressions[0].unalias().table:
            continue
        if any(node.find_ancestor(exp.Select) is query for node in query.find_all(exp.AggFunc, exp.Window)):
            continue
        if any(not read.table or read.table.lower() == name for read in query.find_all(exp.Column)):
            continue  # a bare name, or a read of the outer row
        others = [other for index, other in enumerate(parts) if index != position]
        value_only = True
        for other in others:
            for read in other.find_all(exp.Column):
                if (read.table or "").lower() == theirs:
                    value_only = False  # a source further out with the list's name: moved inside, the list's row would answer
                elif read.find_ancestor(exp.Select) is select:
                    value_only = value_only and _own_column(read, name, mine) == column
                else:
                    value_only = value_only and bool(read.table) and (read.table.lower() != name or (read.name.lower() == column and isinstance(read.this, exp.Identifier) and not read.this.args.get("quoted")))
            for deeper in other.find_all(exp.Select):
                value_only = value_only and not any((s.alias_or_name or "").lower() == theirs for s in _sources(deeper))
        if not value_only or _unstable(select):
            continue
        result = query.copy()
        result.set("distinct", exp.Distinct())
        result.set("expressions", [exp.alias_(exp.column(column, table=source.alias_or_name), item.alias_or_name, quoted=False) for item in select.expressions])
        tests = [exp.Not(this=exp.Is(this=exp.column(column, table=source.alias_or_name), expression=exp.Null()))]
        if result.args.get("where") is not None:
            tests += [inner.copy() for inner in conjuncts(result.args["where"].this)]
        for other in others:
            moved = exp.Paren(this=other.copy())  # a holder, so that a conjunct that is the column itself can be replaced
            for read in list(moved.find_all(exp.Column)):
                if not read.table or read.table.lower() == name:
                    read.replace(exp.column(column, table=source.alias_or_name))
            tests.append(moved.this)
        result.set("where", exp.Where(this=exp.and_(*tests, copy=False)))
        return result
    return None
