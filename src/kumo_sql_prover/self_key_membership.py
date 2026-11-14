"""Membership of a row's own key in a subquery that reads the same table again.

``SELECT id, name FROM students WHERE id NOT IN (SELECT s.id FROM students AS s JOIN departments AS d ON
s.department_id = d.id)`` asks, for each student, whether some student *with the same id* has a department.
When ``students.id`` is a declared key and NOT NULL, the student with the same id is the student itself, so
the subquery need not read ``students`` a second time:

    SELECT id, name FROM students WHERE NOT EXISTS (SELECT 1 FROM departments AS d WHERE students.department_id = d.id)

which is how the same question reads when written directly (``department_id NOT IN (SELECT id FROM
departments)`` normalizes to it under the same NOT NULL facts). Many hand-written solutions route an anti-join
through the table's own key this way.

``o.k [NOT] IN (SELECT [DISTINCT] i.k FROM T AS i, rest WHERE conditions [GROUP BY i.k])`` is rewritten to
``[NOT] EXISTS (SELECT 1 FROM rest WHERE conditions with i's columns read from o)`` when

* ``o`` is a base-table source ``T`` of the select that holds the predicate, whose sources are joined by inner or
  cross joins only (so ``o`` is a real row wherever the predicate is evaluated), and ``k`` is one column that is
  a declared key of ``T`` and declared NOT NULL: a row ``i`` with ``i.k = o.k`` is then ``o``, column for column,
  and neither side of the ``IN`` is ever NULL, so ``IN`` and ``EXISTS`` agree in every position;
* ``k`` is not declared a floating-point column (a NaN is not equal to itself in every engine);
* the subquery reads ``T`` once as ``i`` and at least one other base table, joined by inner or cross joins, with
  no aggregate, ``HAVING``, window, ``LIMIT``, nested subquery or reference to the query around it, and every
  column in it can be attributed to exactly one of its sources.

Without declared keys nothing here fires.
"""

from __future__ import annotations

from sqlglot import exp

from . import smt_values

_BANNED = ("having", "limit", "offset", "qualify", "windows", "with_", "with", "order", "laterals", "pivots", "sample", "kind", "into")


def _from(select: exp.Select):
    return select.args.get("from_") or select.args.get("from")


def _tables(select: exp.Select) -> dict[str, exp.Table] | None:
    """Alias -> table for a select that reads base tables joined by inner or cross joins (ON conditions only)."""

    from_ = _from(select)
    if from_ is None:
        return None
    sources = [from_.this]
    for join in select.args.get("joins") or []:
        if join.side or (join.kind or "").upper() not in ("", "INNER", "CROSS"):
            return None
        if any(join.args.get(k) for k in ("method", "using", "match_condition", "global_", "hint", "directed", "expressions", "pivots")):
            return None
        sources.append(join.this)
    out: dict[str, exp.Table] = {}
    for source in sources:
        if not isinstance(source, exp.Table) or source.args.get("db") or source.args.get("catalog") or source.args.get("pivots") or source.args.get("sample"):
            return None
        alias = source.alias_or_name.lower()
        if not alias or alias in out:
            return None
        out[alias] = source
    return out


def _is_key(table: str, column: str, keys, not_null) -> bool:
    declared = {c.lower() for c in not_null.get(table, frozenset())}
    return column in declared and any(tuple(c.lower() for c in key) == (column,) for key in keys.get(table, []))


def _rewrite(predicate: exp.In, outer: exp.Select, schema, keys, not_null) -> exp.Expression | None:
    if predicate.args.get("expressions") or predicate.args.get("unnest") or predicate.args.get("field"):
        return None
    query = predicate.args.get("query")
    body = query.this if isinstance(query, exp.Subquery) else query
    if not isinstance(body, exp.Select) or any(body.args.get(k) for k in _BANNED):
        return None
    if body.args.get("distinct") is not None and body.args["distinct"].args.get("on") is not None:
        return None
    outer_sources = _tables(outer)
    inner_sources = _tables(body)
    if outer_sources is None or inner_sources is None or len(inner_sources) < 2 or len(body.expressions) != 1:
        return None
    for node in body.walk():
        if node is not body and isinstance(node, (exp.Select, exp.Subquery, exp.SetOperation, exp.AggFunc, exp.Window, exp.Star)):
            return None

    # the left operand: a key column of one outer source
    left = predicate.this.unnest() if isinstance(predicate.this, exp.Paren) else predicate.this
    if not isinstance(left, exp.Column) or not isinstance(left.this, exp.Identifier):
        return None
    column = left.name.lower()
    if left.table:
        outer_alias = left.table.lower()
    elif len(outer_sources) == 1:
        (outer_alias,) = outer_sources
    else:
        return None
    if outer_alias not in outer_sources:
        return None
    table = outer_sources[outer_alias].name.lower()
    if column not in schema.get(table, ()) or not _is_key(table, column, keys, not_null):
        return None

    # every column of the subquery, attributed to one of its own sources
    columns = {alias: schema.get(source.name.lower()) for alias, source in inner_sources.items()}
    if any(listed is None for listed in columns.values()):
        return None
    owner: dict[int, str] = {}
    for node in body.find_all(exp.Column):
        name = node.name.lower()
        if node.table:
            alias = node.table.lower()
            if alias not in inner_sources or name not in columns[alias]:
                return None
        else:
            holders = [alias for alias, listed in columns.items() if name in listed]
            if len(holders) != 1:
                return None
            alias = holders[0]
        owner[id(node)] = alias

    # the subquery selects the same key of the same table, read once
    shown = body.expressions[0].unalias().unnest()  # ``DISTINCT(i.k)`` is a parenthesized column
    if not isinstance(shown, exp.Column) or shown.name.lower() != column:
        return None
    inner_alias = owner[id(shown)]
    if inner_sources[inner_alias].name.lower() != table:
        return None
    group = body.args.get("group")
    if group is not None:
        if len(group.expressions) != 1 or any(group.args.get(k) for k in ("grouping_sets", "cube", "rollup", "totals", "all")):
            return None
        key = group.expressions[0].unnest()
        if not isinstance(key, exp.Column) or key.name.lower() != column or owner[id(key)] != inner_alias:
            return None
    rest = {alias: source for alias, source in inner_sources.items() if alias != inner_alias}
    if outer_alias in rest or any(source.name.lower() == table for source in rest.values()):
        return None  # the outer row's name would be hidden by a source that stays; the table is read once

    conditions = []
    where = body.args.get("where")
    if where is not None:
        conditions.append(where.this)
    for join in body.args.get("joins") or []:
        if join.args.get("on") is not None:
            conditions.append(join.args["on"])
    if not conditions:
        return None
    outer_name = outer_sources[outer_alias].alias_or_name
    moved = []
    for condition in conditions:
        holder = exp.Paren(this=condition.copy())
        # ``copy`` keeps the walk order, so the nodes pair up with the originals
        for original, copied in zip(list(condition.find_all(exp.Column)), list(holder.find_all(exp.Column))):
            alias = owner[id(original)]
            name = outer_name if alias == inner_alias else inner_sources[alias].alias_or_name
            copied.set("table", exp.to_identifier(name))
        moved.append(holder)
    condition = moved[0]
    for part in moved[1:]:
        condition = exp.And(this=condition, expression=part)
    new = exp.Select(expressions=[exp.Literal.number(1)])
    names = list(rest)
    new.set("from_" if "from_" in exp.Select.arg_types else "from", exp.From(this=rest[names[0]].copy()))
    if len(names) > 1:
        new.set("joins", [exp.Join(this=rest[alias].copy(), kind="CROSS") for alias in names[1:]])
    new.set("where", exp.Where(this=condition))
    return exp.Exists(this=new)


def _left_name(predicate: exp.In) -> str:
    left = predicate.this.unnest() if isinstance(predicate.this, exp.Paren) else predicate.this
    return left.name.lower() if isinstance(left, exp.Column) else ""


def _table_of(predicate: exp.In, outer: exp.Select) -> str:
    """The base table the left operand is read from, "" when it is not a column of one outer source."""

    left = predicate.this.unnest() if isinstance(predicate.this, exp.Paren) else predicate.this
    sources = _tables(outer)
    if not isinstance(left, exp.Column) or sources is None:
        return ""
    alias = left.table.lower() if left.table else (next(iter(sources)) if len(sources) == 1 else "")
    return sources[alias].name.lower() if alias in sources else ""


def read_self_key_membership(tree: exp.Expression, schema, keys, not_null, types=None) -> exp.Expression:
    """``tree`` with every such ``IN`` read as ``EXISTS`` (in place; a pre-pass of ``normalize``)."""

    if not keys or not schema:
        return tree
    schema = {t.lower(): [c.lower() for c in cols] for t, cols in schema.items()}
    keys = {t.lower(): ks for t, ks in keys.items()}
    not_null = {t.lower(): frozenset(c.lower() for c in cols) for t, cols in (not_null or {}).items()}
    types = {t.lower(): {c.lower(): v for c, v in cols.items()} for t, cols in (types or {}).items()}
    for predicate in list(tree.find_all(exp.In)):
        outer = predicate.find_ancestor(exp.Select)
        where = outer.args.get("where") if outer is not None else None
        # only a predicate of the WHERE clause of ``outer``: every row it sees is a row of each source
        if where is None or not any(node is predicate for node in where.walk(prune=lambda n: isinstance(n, exp.Select))):
            continue
        if smt_values.type_class((types or {}).get(_table_of(predicate, outer), {}).get(_left_name(predicate))) == smt_values.FLOAT64:
            continue  # a NaN is not equal to itself in every engine
        replacement = _rewrite(predicate, outer, schema, keys, not_null)
        if replacement is not None:
            predicate.replace(replacement)
    return tree
