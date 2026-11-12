"""A membership test against a table's own groups, inside a query that already forms those groups, is ``HAVING``.

    SELECT o.k, SUM(l.q)
    FROM o, l
    WHERE o.k = l.k AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300)
    GROUP BY o.k

is

    SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k GROUP BY o.k HAVING SUM(l.q) > 300

(TPC-H Q18). The same holds when the groups are joined instead of tested:
``JOIN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300) AS g ON o.k = g.k``.

Why this is exact. Call the grouped subquery ``G`` (table ``T`` grouped by its column ``k``, kept when ``h``
holds) and the outer source that reads the same table ``l``. The conditions below make every group of the outer
query one row ``r`` of each other source together with *all* rows of ``T`` that have one value ``v`` of ``k``, each
once:

* the other sources are base tables, and the grouping columns (closed under the equalities of the ``WHERE`` and
  ``ON`` conjuncts and under "a table's declared key fixes its row") fix a key of each: one ``r`` per group;
* the same closure fixes ``l.k``: one ``v`` per group;
* no ``WHERE`` or ``ON`` conjunct and no grouping expression reads a column of ``l`` other than ``k``, and there
  is no outer join and no other subquery: whether ``(r, row of l)`` is in the join depends on the row only
  through ``v``, so the rows of ``T`` with that ``v`` are in together or out together, and no grouping column
  splits them.

So the aggregates of ``h``, taken over the outer group's ``l`` rows, are the aggregates ``G`` took over group
``v``. The tested value ``x`` is in the equality class of ``l.k``, so it is ``v``; and ``v`` is not NULL, because
``l.k`` is declared NOT NULL or sits in an equality that remains (a NULL ``v`` would pass no ``IN`` and no join,
while ``HAVING`` could keep its group). Then ``x IN (G)`` is TRUE exactly when group ``v`` passes ``h``, which is
when ``HAVING h`` over the outer group is TRUE. The joined form returns each row once more or not at all, because
``G`` has one row per ``v``: the same.

Conditions, all checked: the outer select is grouped (no ``ROLLUP`` and the like), with comma, cross or inner
joins over base tables the schema lists (plus ``G`` in the joined form), each other table with a declared key
whose columns are all declared NOT NULL;
``G`` is ``SELECT k FROM T GROUP BY k HAVING h`` (or the form the normalizer gives it inside ``IN``: a select of
the key from that grouped derived table with the test in ``WHERE``), with no ``WHERE`` of its own, one grouping
column, and ``h`` reading ``T`` only inside plain aggregates; exactly one outer source reads ``T``; in the joined
form ``G`` is read by the one equality and nowhere else; every column resolves to one source.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import extended_grouping, same_table, select_sources
from .keyed_join_groups import _and_all, _owner, declared_columns, grouped_source, key_of, plain_joins
from .unmatched_outer_join import _columns_of, _conjuncts, _inside, _plain_column, _plain_table, _unwrap

_OUTER_ARGS = frozenset({"expressions", "from_", "from", "joins", "where", "group", "having", "order", "limit", "offset"})


def _own_groups(grouped: dict) -> tuple[exp.Table, str, list[exp.Expression]] | None:
    """``(T, k, h conjuncts)`` when a grouped derived table is ``T`` grouped by its one column ``k``, unfiltered."""

    inner = grouped["select"]
    sources = select_sources(inner)
    if len(sources) != 1 or inner.args.get("where") is not None or len(grouped["keys"]) != 1:
        return None
    having = inner.args.get("having")
    return sources[0], grouped["keys"][0].name.lower(), _conjuncts(having.this if having is not None else None)


def _membership(test: exp.Expression, declared: dict[str, set[str]]):
    """``(x, T, k, h conjuncts, name of T inside)`` when ``test`` is ``x IN (the keys of T's groups that pass h)``."""

    test = _unwrap(test)
    if type(test) is not exp.In or any(v for key, v in test.args.items() if key not in ("this", "query")):
        return None
    query = test.args.get("query")
    if type(query) is not exp.Subquery or query.alias or any(v for key, v in query.args.items() if key != "this"):
        return None
    inner = query.this
    if type(inner) is not exp.Select or len(inner.expressions) != 1 or _plain_column(test.this) is None:
        return None
    if set(key for key, value in inner.args.items() if value) <= {"expressions", "from_", "from", "group", "having"}:
        # the plain spelling: wrap it as a derived table so that both spellings are read the same way
        attached = select_sources(inner)
        if len(attached) != 1 or not _plain_table(attached[0]):  # checked here: the wrapped copy cannot see a CTE of that name
            return None
        wrapped = exp.Subquery(this=inner.copy(), alias=exp.TableAlias(this=exp.to_identifier("kqo")))
        grouped = grouped_source(wrapped, declared)
        own = _own_groups(grouped) if grouped is not None else None
        if own is None or len(grouped["outputs"]) != 1:
            return None
        value = next(iter(grouped["outputs"].values()))
        if _plain_column(value) is None:
            return None
        return test.this, own[0], own[1], own[2], grouped["names"][0]
    # the normalizer's spelling: SELECT g.key FROM (SELECT k AS key, agg AS a .. FROM T GROUP BY k) AS g WHERE test(g.a)
    if set(key for key, value in inner.args.items() if value) - {"expressions", "from_", "from", "where"}:
        return None
    sources = select_sources(inner)
    if len(sources) != 1:
        return None
    grouped = grouped_source(sources[0], declared)
    own = _own_groups(grouped) if grouped is not None else None
    if own is None or own[2]:
        return None
    alias = sources[0].alias.lower()
    outputs = grouped["outputs"]

    def read(node: exp.Expression) -> exp.Expression | None:
        holder = exp.Paren(this=node.copy())
        for column in list(holder.find_all(exp.Column)):
            if _plain_column(column) is not column or column.table.lower() not in ("", alias) or column.name.lower() not in outputs:
                return None
            value = outputs[column.name.lower()].copy()
            column.replace(value if isinstance(value, (exp.Column, exp.Func)) else exp.Paren(this=value))
        if any(isinstance(n, (exp.Query, exp.Subquery, exp.Exists, exp.Window)) for n in holder.walk()):
            return None
        return holder.this

    item = inner.expressions[0]
    key = read(item.this if isinstance(item, exp.Alias) else item)
    if key is None or _plain_column(key) is None or _plain_column(key).name.lower() != own[1]:
        return None
    where = inner.args.get("where")
    tests = []
    for part in _conjuncts(where.this if where is not None else None):
        new = read(part)
        if new is None:
            return None
        tests.append(new)
    return test.this, own[0], own[1], tests, grouped["names"][0]


def filter_own_groups(select: exp.Select, schema: dict | None, keys: dict | None, not_null: dict | None, dialect: str = "bigquery") -> exp.Expression | None:
    """``select`` with one test against its own table's groups written as ``HAVING``, or ``None``."""

    group = select.args.get("group")
    if group is None or not schema or not keys or extended_grouping(group) or not group.expressions:
        return None
    if any(value for key, value in select.args.items() if key not in _OUTER_ARGS) or not plain_joins(select):
        return None
    declared = declared_columns(schema)
    never_null = declared_columns(not_null or {})
    sources = select_sources(select)
    names = [(s.alias_or_name or "").lower() for s in sources]
    if not all(names) or len(set(names)) != len(names):
        return None
    joins = select.args.get("joins") or []
    where = select.args.get("where")
    conditions: list[tuple[exp.Expression, int]] = []  # each top-level conjunct with the number of sources in scope
    for index, join in enumerate(joins):
        conditions.extend((part, index + 2) for part in _conjuncts(join.args.get("on")))
    conditions.extend((part, len(sources)) for part in _conjuncts(where.this if where is not None else None))

    # the link: a membership test in WHERE, or a joined grouped table read by one equality
    candidates = []
    for part, visible in conditions:
        if visible == len(sources) and not any(part is p for j in joins for p in _conjuncts(j.args.get("on"))):
            found = _membership(part, declared)
            if found is not None:
                candidates.append((part, None, *found))
    for position, source in enumerate(sources):
        if type(source) is not exp.Subquery:
            continue
        grouped = grouped_source(source, declared)
        own = _own_groups(grouped) if grouped is not None else None
        if own is None:
            continue
        alias = names[position]
        key_outputs = {name for name, value in grouped["outputs"].items() if _plain_column(value) is not None}
        with_grouped = [set(grouped["outputs"]) if index == position else (_columns_of(item, declared) or set()) if type(item) is exp.Table else set() for index, item in enumerate(sources)]
        scope = {id(column): visible for part, visible in conditions for column in part.find_all(exp.Column)}
        reads = [
            c for c in select.find_all(exp.Column)
            if not _inside(c, source) and (c.table.lower() == alias or (not c.table and _owner(c, names, with_grouped, scope.get(id(c), len(sources))) in (position, None) and c.name.lower() in grouped["outputs"]))
        ]
        if len(reads) != 1 or reads[0].table.lower() != alias or reads[0].name.lower() not in key_outputs:
            continue
        for part, visible in conditions:
            part = _unwrap(part)
            if type(part) is exp.EQ and (reads[0] is _unwrap(part.this) or reads[0] is _unwrap(part.expression)):
                other = part.expression if reads[0] is _unwrap(part.this) else part.this
                if _plain_column(other) is not None:
                    candidates.append((part, position, _plain_column(other), *own, grouped["names"][0]))

    for link, position, tested, table, key, tests, inside in candidates:
        if not tests:
            continue
        others = [index for index in range(len(sources)) if index != position]
        if not all(_plain_table(sources[index]) for index in others):
            continue
        same = [index for index in others if same_table(sources[index], table, dialect)]
        if len(same) != 1:
            continue
        mine = same[0]
        owned = [(_columns_of(sources[index], declared) or set()) if index != position else set() for index in range(len(sources))]
        if not all(owned[index] for index in others) or key not in owned[mine]:
            continue
        # no other subquery, window or outer reference to follow
        if any(isinstance(n, (exp.Subquery, exp.Exists, exp.Window)) and not _inside(n, link) and (position is None or not _inside(n, sources[position])) for n in select.walk()):
            continue
        table_keys = {index: key_of(sources[index], keys, owned[index], not_null) for index in others if index != mine}
        if any(value is None for value in table_keys.values()):
            continue

        def owner(column: exp.Column, visible: int) -> tuple[int, str] | None:
            index = _owner(column, names, owned, visible)
            return None if index is None or index == position else (index, column.name.lower())

        ok = True
        equal: list[tuple[tuple[int, str], tuple[int, str]]] = []
        for part, visible in conditions:
            if part is link or _unwrap(part) is link:
                continue
            for column in part.find_all(exp.Column):
                found = owner(column, visible)
                if found is None or (found[0] == mine and found[1] != key):
                    ok = False
                    break
            if not ok:
                break
            bare = _unwrap(part)
            if type(bare) is exp.EQ and _plain_column(bare.this) is not None and _plain_column(bare.expression) is not None:
                equal.append((owner(_plain_column(bare.this), visible), owner(_plain_column(bare.expression), visible)))
        if not ok:
            continue
        fixed: set[tuple[int, str]] = set()
        for expression in group.expressions:
            for column in expression.find_all(exp.Column):
                found = owner(column, len(sources))
                if found is None or (found[0] == mine and found[1] != key):
                    ok = False
                    break
            if not ok:
                break
            plain = _plain_column(expression)
            if plain is not None:
                fixed.add(owner(plain, len(sources)))
        if not ok:
            continue
        # every other column of the select must resolve too (select list, HAVING, ORDER BY), or a bare name could move
        output_names = {(item.alias_or_name or "").lower() for item in select.expressions} - {""}
        for column in select.find_all(exp.Column):
            if _inside(column, link) or (position is not None and _inside(column, sources[position])):
                continue
            if not column.table and column.name.lower() in output_names and column.find_ancestor(exp.Order) is not None:
                continue
            if owner(column, len(sources)) is None:
                ok = False
                break
        if not ok:
            continue
        changed = True
        while changed:
            changed = False
            for a, b in equal:
                if (a in fixed) != (b in fixed):
                    fixed.update((a, b))
                    changed = True
            for index, table_key in table_keys.items():
                if all((index, column) in fixed for column in table_key) and not all((index, column) in fixed for column in owned[index]):
                    fixed.update((index, column) for column in owned[index])
                    changed = True
        if (mine, key) not in fixed or not all((index, column) in fixed for index, table_key in table_keys.items() for column in table_key):
            continue
        # the tested value is the group's key value
        classes = {(mine, key)}
        changed = True
        while changed:
            changed = False
            for a, b in equal:
                if (a in classes) != (b in classes):
                    classes.update((a, b))
                    changed = True
        link_visible = next(visible for part, visible in conditions if part is link or _unwrap(part) is link)
        if owner(tested, link_visible) not in classes:
            continue
        full = ".".join(part.name for part in sources[mine].parts).lower()
        declared_not_null = key in (never_null.get(full) or never_null.get(sources[mine].name.lower()) or set())
        if not declared_not_null and not any((mine, key) in pair for pair in equal):
            continue
        # h, read over the outer source
        source = sources[mine]
        alias = source.args.get("alias")
        qualifier = alias.this if alias is not None else source.this
        having_parts = []
        for part in tests:
            new = exp.Paren(this=part.copy())
            aggregates = list(new.find_all(exp.AggFunc))
            for column in new.find_all(exp.Column):
                if column.table.lower() != inside or not any(_inside(column, a) for a in aggregates):
                    ok = False
                    break
                column.set("table", qualifier.copy())
            if not ok or not aggregates:
                ok = False
                break
            having_parts.append(new.this)
        if not ok:
            continue

        copy = select.copy()
        new_joins = copy.args.get("joins") or []
        new_where = copy.args.get("where")
        new_conditions = []
        for join in new_joins:
            new_conditions.extend(_conjuncts(join.args.get("on")))
        new_conditions.extend(_conjuncts(new_where.this if new_where is not None else None))
        index = next(i for i, (part, _) in enumerate(conditions) if part is link or _unwrap(part) is link)
        target = new_conditions[index]
        if position is not None and position == 0:
            continue  # the grouped table leads the FROM clause: leave the reordering to other rules
        # rebuild the ON and WHERE conjunct lists without the link
        cursor = 0
        for join in new_joins:
            parts = _conjuncts(join.args.get("on"))
            kept = [p for offset, p in enumerate(parts) if cursor + offset != index]
            cursor += len(parts)
            if len(kept) != len(parts):
                join.set("on", _and_all([p.copy() for p in kept]) if kept else None)
        if new_where is not None:
            parts = _conjuncts(new_where.this)
            kept = [p for offset, p in enumerate(parts) if cursor + offset != index]
            if len(kept) != len(parts):
                copy.set("where", exp.Where(this=_and_all([p.copy() for p in kept])) if kept else None)
        del target
        if position is not None:
            dropped = new_joins[position - 1]
            remaining = _conjuncts(dropped.args.get("on"))
            if any(not column.table for part in remaining for column in part.find_all(exp.Column)):
                continue  # a bare name would meet more sources in WHERE than it did in ON
            dropped.pop()
            if remaining:
                old = copy.args.get("where")
                parts = [p.copy() for p in _conjuncts(old.this if old is not None else None)] + [p.copy() for p in remaining]
                copy.set("where", exp.Where(this=_and_all(parts)))
            if not copy.args.get("joins"):
                copy.set("joins", None)
        having = copy.args.get("having")
        parts = [p.copy() for p in _conjuncts(having.this if having is not None else None)] + having_parts
        copy.set("having", exp.Having(this=_and_all(parts)))
        return copy
    return None
