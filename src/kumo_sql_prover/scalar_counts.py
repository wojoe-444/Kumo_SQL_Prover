"""A filtered scalar count per row of a keyed table is a grouped count over a join.

    SELECT p.id, p.name, (SELECT COUNT(*) FROM games AS g WHERE g.first = p.id) AS wins
    FROM players AS p
    HAVING wins > 0

computes a count for every player and keeps the players whose count is positive. It is

    SELECT p.id, p.name, COUNT(*) AS wins
    FROM players AS p CROSS JOIN games AS g
    WHERE g.first = p.id
    GROUP BY p.id
    HAVING COUNT(*) > 0

which the prover compares with other grouped selects over the same two tables.

**The rule.** The select reads one base table ``t`` that has a declared key of NOT NULL columns, has no joins,
grouping, ``DISTINCT``, ``ORDER BY`` or ``LIMIT``, and returns row values of ``t`` and exactly one *scalar
item* ``n``. Its only filter is ``n > 0``, ``n >= 1`` or ``n <> 0`` (``n`` read by its output name, in
``HAVING`` or ``WHERE``, where no column of ``t`` has that name). The same is read through one derived table:
``SELECT .. FROM (SELECT .., <scalar item> AS n FROM t) AS d WHERE d.n > 0`` with plain columns of ``d`` as
outputs. The scalar item is one of

1. one nested query ``(SELECT a_1 + .. + a_m FROM f [WHERE w])`` over base tables ``f`` (inner or cross joins),
   each ``a_i`` a plain ``SUM(x)`` or ``COUNT(*)`` / ``COUNT(x)``. It is written as ``a_1 + .. + a_m`` over
   ``t CROSS JOIN f WHERE w``;
2. a sum ``q_1 + .. + q_m`` of nested queries ``(SELECT COUNT(*) FROM c [WHERE w_i])`` (or ``COUNT(x_i)``) that
   all read the same single base table ``c``. It is written as
   ``SUM(CASE WHEN w_1 THEN 1 ELSE 0 END + .. + CASE WHEN w_m THEN 1 ELSE 0 END)`` over ``t CROSS JOIN c``, with
   ``x_i IS NOT NULL`` added to ``w_i`` for ``COUNT(x_i)`` unless ``x_i`` is a literal or a column ``w_i``
   compares.

The rewritten select groups by the key of ``t`` and keeps the filter as ``HAVING`` on the written aggregate.

**Why it is sound.** The key (declared, every column NOT NULL) makes each row ``p`` of ``t`` its own group of
the rewritten select, holding the rows of ``f`` (or ``c``) joined to ``p`` that pass ``w``; the other outputs
read ``t`` only, so they are fixed in the group, and each kept ``p`` gives one row on both sides.

* Form 1, a ``p`` with at least one row of ``f`` passing ``w``: the nested query aggregates exactly the rows of
  ``p``'s group, so every ``a_i`` has the same value on both sides, and the filter decides alike. A ``p`` with
  no such row has no group in the rewritten select; in the original every ``SUM`` is NULL and every ``COUNT``
  is 0 over no rows, so the item is NULL or 0 and none of the three filters is TRUE: the row is dropped there
  too. That is why a filter is required and why only these three are read.
* Form 2, when ``c`` has rows: ``p``'s group holds every row of ``c``, each ``COUNT`` counts the rows whose
  ``w_i`` is TRUE (and whose ``x_i`` is not NULL), the indicator of row and ``i`` is 1 exactly then, and the
  sum of the indicators over the group is the sum of the counts; the ``CASE`` has an ``ELSE``, so it is never
  NULL. When ``c`` is empty every count is 0, the filter drops every row, and the rewritten select has no
  groups.

A column of the nested query is read as SQL scopes it: a bare name belongs to the nested query's own tables
when one of them has it, and to ``t`` otherwise.

**What is refused.** A table without a declared key of NOT NULL columns (two equal rows would share a group);
no filter on the item, or any other filter or comparison (``n >= 0`` keeps the rows the join loses); a second
nested query outside the item; a nested query with grouping, ``HAVING``, ``DISTINCT``, ``LIMIT``, outer joins,
derived tables, windows, ``COUNT(DISTINCT ..)``, ``MIN``/``MAX``/``AVG`` or anything but a sum of plain
aggregates; in form 2 a ``SUM`` (it is NULL where a filtered count is 0), joins, or different tables; an alias
of the nested query equal to the alias of ``t``; a column that cannot be attributed to one source.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key
from .ast_utils import FROM_KEY, select_sources
from .union_all_counts import _BRANCH_ARGS, _COMPARISONS, _all, _plain_joins, _row_value

_FILTERS = {exp.GT: "0", exp.GTE: "1", exp.NEQ: "0"}


def scalar_counts_to_grouped_join(
    select: exp.Expression,
    schema: dict[str, list[str]] | None,
    keys: dict[str, list[tuple[str, ...]]] | None,
    not_null: dict[str, frozenset[str]] | None,
) -> exp.Expression | None:
    """The grouped join the select is (see the module docstring), or ``None``."""

    if not schema or not keys or not not_null or not isinstance(select, exp.Select):
        return None
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    read = _through_derived(select, known) or _direct(select, known)
    if read is None:
        return None
    table, items, test = read
    listed = _base_table(table, known)
    alias = table.alias_or_name
    if listed is None or not alias:
        return None
    name = _table_key(table)
    declared = {c.lower() for owner, columns in not_null.items() if owner.lower() == name for c in columns}
    key = next(
        (tuple(c.lower() for c in k) for owner, listed_keys in keys.items() if owner.lower() == name for k in listed_keys if k and {c.lower() for c in k} <= declared),
        None,
    )
    if key is None:
        return None
    scalar = [position for position, (value, _, _) in enumerate(items) if value.find(exp.Select, exp.Subquery, exp.Exists) is not None]
    if len(scalar) != 1 or test[0] != items[scalar[0]][1].lower():
        return None
    outer = {alias.lower(): listed}
    outputs = []
    written = None
    for position, (value, name, aliased) in enumerate(items):
        if position == scalar[0]:
            written = _one_query(value, alias, outer, known) or _sum_of_counts(value, alias, outer, known)
            if written is None:
                return None
            value = written[0].copy()
            aliased = True
        else:
            if not _row_value(value):
                return None
            value = _qualified(value, {}, outer, {alias.lower(): alias})
            if value is None:
                return None
        outputs.append(exp.alias_(value, name, copy=False) if aliased else value)
    total, tables, where = written
    result = exp.Select(expressions=outputs)
    result.set(FROM_KEY, exp.From(this=table.copy()))
    result.set("joins", [exp.Join(this=source.copy(), kind="CROSS") for source in tables])
    if where is not None:
        result.set("where", exp.Where(this=where))
    result.set("group", exp.Group(expressions=[exp.column(column, alias) for column in key]))
    result.set("having", exp.Having(this=test[1](this=total.copy(), expression=exp.Literal.number(test[2]))))
    return result


def _filter(node: exp.Expression | None) -> tuple[str, type, str, str] | None:
    """``n > 0``, ``n >= 1`` or ``n <> 0`` as (name of n, comparison, literal, qualifier of n)."""

    if node is None:
        return None
    node = node.unnest()
    literal = _FILTERS.get(type(node))
    if literal is None:
        return None
    column, number = node.this.unnest(), node.expression.unnest()
    if not isinstance(column, exp.Column) or not isinstance(number, exp.Literal) or number.is_string or number.this != literal:
        return None
    if column.args.get("db") or column.args.get("catalog"):
        return None
    return column.name.lower(), type(node), literal, column.table.lower()


def _direct(select: exp.Select, known):
    """``SELECT .., <scalar item> AS n FROM t HAVING n > 0`` (or with the filter under ``WHERE``)."""

    present = {key for key, value in select.args.items() if value}
    if not present <= {"expressions", FROM_KEY, "from", "where", "having"} or {"where", "having"} <= present:
        return None
    sources = select_sources(select)
    if len(sources) != 1 or _base_table(sources[0], known) is None:
        return None
    clause = select.args.get("where") or select.args.get("having")
    test = _filter(clause.this if clause is not None else None)
    if test is None or test[3] or test[0] in _base_table(sources[0], known):
        return None
    items = []
    for item in select.expressions:
        if not item.alias_or_name:
            return None
        items.append((item.unalias(), item.alias_or_name, isinstance(item, exp.Alias)))
    if [name.lower() for _, name, _ in items].count(test[0]) != 1:
        return None
    return sources[0], items, test[:3]


def _through_derived(select: exp.Select, known):
    """``SELECT d.a, .. FROM (SELECT .., <scalar item> AS n FROM t) AS d WHERE d.n > 0``."""

    present = {key for key, value in select.args.items() if value}
    if not present <= {"expressions", FROM_KEY, "from", "where"}:
        return None
    sources = select_sources(select)
    if len(sources) != 1 or not isinstance(sources[0], exp.Subquery) or not isinstance(sources[0].this, exp.Select):
        return None
    derived = sources[0]
    name = derived.alias.lower()
    alias = derived.args.get("alias")
    if not name or alias.args.get("columns") or any(value for key, value in derived.args.items() if key not in ("this", "alias")):
        return None
    inner = derived.this
    if not {key for key, value in inner.args.items() if value} <= {"expressions", FROM_KEY, "from"}:
        return None
    inner_sources = select_sources(inner)
    if len(inner_sources) != 1 or _base_table(inner_sources[0], known) is None:
        return None
    by_name: dict[str, exp.Expression] = {}
    for item in inner.expressions:
        if not item.alias_or_name or item.alias_or_name.lower() in by_name:
            return None
        by_name[item.alias_or_name.lower()] = item.unalias()
    where = select.args.get("where")
    test = _filter(where.this if where is not None else None)
    if test is None or test[3] not in ("", name) or test[0] not in by_name:
        return None
    items = []
    for item in select.expressions:
        column = item.unalias()
        if not isinstance(column, exp.Column) or column.args.get("db") or (column.table and column.table.lower() != name):
            return None
        value = by_name.get(column.name.lower())
        if value is None:
            return None
        plain = isinstance(value, exp.Column) and value.name.lower() == item.alias_or_name.lower()
        items.append((value, item.alias_or_name, not plain))
    if sum(1 for item in select.expressions if item.unalias().name.lower() == test[0]) != 1:
        return None
    # the filter names the inner item, the outputs may rename it
    position = next(i for i, item in enumerate(select.expressions) if item.unalias().name.lower() == test[0])
    return inner_sources[0], items, (items[position][1].lower(), test[1], test[2])


def _qualified(node: exp.Expression, own: dict[str, list[str]], outer: dict[str, list[str]], spelling: dict[str, str]) -> exp.Expression | None:
    """A copy with each column named by its source: the nested query's own tables first, then the outer table."""

    node = exp.Paren(this=node.copy())
    for column in list(node.find_all(exp.Column)):
        if column.args.get("db") or column.args.get("catalog") or not isinstance(column.this, exp.Identifier):
            return None
        name = column.name.lower()
        if column.table:
            owner = column.table.lower()
            if name not in (own.get(owner) or outer.get(owner) or ()):
                return None
        else:
            owners = [alias for alias, listed in own.items() if name in listed]
            if not owners:
                owners = [alias for alias, listed in outer.items() if name in listed]
            if len(owners) != 1:
                return None
            owner = owners[0]
        column.replace(exp.column(name, spelling[owner]))
    return node.this


def _sources(query: exp.Select, alias: str, known) -> tuple[dict[str, list[str]], dict[str, str]] | None:
    """The base tables of a nested query by alias, none of them named like the outer table."""

    columns: dict[str, list[str]] = {}
    spelling: dict[str, str] = {}
    for source in select_sources(query):
        listed = _base_table(source, known)
        name = source.alias_or_name
        if listed is None or not name or name.lower() in columns or name.lower() == alias.lower():
            return None
        columns[name.lower()] = listed
        spelling[name.lower()] = name
    return (columns, spelling) if columns else None


def _nested(node: exp.Expression) -> exp.Select | None:
    """The select of a plain scalar nested query: one output, no grouping, nothing nested inside."""

    while isinstance(node, exp.Paren):
        node = node.this
    if not isinstance(node, exp.Subquery) or any(value for key, value in node.args.items() if key != "this"):
        return None
    query = node.this
    if not isinstance(query, exp.Select) or any(value for key, value in query.args.items() if key not in _BRANCH_ARGS):
        return None
    if len(query.expressions) != 1 or not _plain_joins(query):
        return None
    if any(inner is not query for inner in query.find_all(exp.Select, exp.Subquery, exp.Exists, exp.Window)):
        return None
    where = query.args.get("where")
    if where is not None and not _row_value(where.this):
        return None
    return query


def _plain_aggregate(node: exp.Expression, counts_only: bool) -> bool:
    if type(node) not in ((exp.Count,) if counts_only else (exp.Count, exp.Sum)):
        return False
    if any(value for key, value in node.args.items() if key not in ("this", "big_int")):
        return False
    argument = node.this
    if isinstance(node, exp.Count) and (argument is None or isinstance(argument, exp.Star)):
        return True
    return argument is not None and not isinstance(argument, exp.Distinct) and _row_value(argument)


def _terms(node: exp.Expression) -> list[exp.Expression]:
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Add):
        return _terms(node.this) + _terms(node.expression)
    return [node]


def _one_query(value: exp.Expression, alias: str, outer: dict[str, list[str]], known: dict[str, list[str]]):
    """Form 1: (the aggregates over the joined rows, the nested query's tables, its condition)."""

    query = _nested(value)
    if query is None:
        return None
    read = _sources(query, alias, known)
    if read is None:
        return None
    columns, spelling = read
    spelling = {**spelling, alias.lower(): alias}
    output = query.expressions[0].unalias()
    if not all(_plain_aggregate(term, False) for term in _terms(output)):
        return None
    total = _qualified(output, columns, outer, spelling)
    where = query.args.get("where")
    condition = None if where is None else _qualified(where.this, columns, outer, spelling)
    if total is None or (where is not None and condition is None):
        return None
    for join in query.args.get("joins") or []:
        if join.args.get("on") is not None:
            on = _qualified(join.args["on"], columns, outer, spelling)
            if on is None or not _row_value(on):
                return None
            condition = _all([part for part in (condition, on) if part is not None])
    return total, [source.copy() for source in select_sources(query)], condition


def _sum_of_counts(value: exp.Expression, alias: str, outer: dict[str, list[str]], known: dict[str, list[str]]):
    """Form 2: (the sum of indicators, the one counted table, no condition)."""

    terms = _terms(value)
    if len(terms) < 2:
        return None
    table = None
    first = None
    total = None
    for term in terms:
        query = _nested(term)
        if query is None or query.args.get("joins"):
            return None
        read = _sources(query, alias, known)
        if read is None:
            return None
        columns, spelling = read
        source = select_sources(query)[0]
        if table is None:
            table, first = _table_key(source), source
        elif _table_key(source) != table:
            return None
        # every nested query reads the table under the first one's alias
        shared = first.alias_or_name
        spelling = {name: shared for name in spelling}
        spelling[alias.lower()] = alias
        count = query.expressions[0].unalias()
        if not _plain_aggregate(count, True):
            return None
        where = query.args.get("where")
        parts = []
        if where is not None:
            condition = _qualified(where.this, columns, outer, spelling)
            if condition is None:
                return None
            parts = list(condition.flatten()) if isinstance(condition, exp.And) else [condition]
        argument = count.this
        if argument is not None and not isinstance(argument, (exp.Star, exp.Literal)):
            counted = _qualified(argument, columns, outer, spelling)
            if counted is None:
                return None
            compared = {
                side.sql() for part in parts if isinstance(part.unnest(), _COMPARISONS) for side in (part.unnest().this, part.unnest().expression) if isinstance(side, exp.Column)
            }
            if not (isinstance(counted, exp.Column) and counted.sql() in compared):
                wrapped = counted if isinstance(counted, (exp.Column, exp.Paren)) else exp.Paren(this=counted)
                parts = parts + [exp.Not(this=exp.Is(this=wrapped, expression=exp.Null()))]
        test = _all(parts)
        indicator = exp.Literal.number(1) if test is None else exp.Case(ifs=[exp.If(this=test, true=exp.Literal.number(1))], default=exp.Literal.number(0))
        total = indicator if total is None else exp.Add(this=total, expression=indicator)
    return exp.Sum(this=total), [first.copy()], None
