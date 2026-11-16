"""A ``DISTINCT`` list that holds a key of one joined table asks which rows of that table have a partner.

    SELECT DISTINCT s.pid, p.name
    FROM product AS p JOIN sales AS s ON p.id = s.pid
    WHERE s.day < 5 AND NOT EXISTS (SELECT 1 FROM sales AS t WHERE t.pid = s.pid AND t.day > 9)

When ``product.id`` is a declared key that is NOT NULL, this is

    SELECT p.id AS pid, p.name FROM product AS p
    WHERE EXISTS (SELECT 1 FROM sales AS s WHERE p.id = s.pid AND s.day < 5
                  AND NOT EXISTS (SELECT 1 FROM sales AS t WHERE t.pid = s.pid AND t.day > 9))

Why it holds. ``s.pid`` is listed, and the join makes it equal to ``p.id``, so the list is ``p.id, p.name``: columns
of ``p`` that include its key. Every row of the join that passes the ``WHERE`` gives the tuple of its ``p`` row, two
different ``p`` rows give two different tuples (their keys differ, and a key is never NULL), and ``DISTINCT`` keeps
each tuple once. So the result has one row for each ``p`` that has at least one partner passing the conditions:
``EXISTS``, with no ``DISTINCT`` left to do.

``constraint_normalization.keyed_join_to_exists`` makes this step for a select without subqueries that lists
columns of the left table only. This rule also reads a column of the other table that a top-level equality ties to
a column of the keyed one, takes either table as the keyed one, reads a comma join, and keeps the conditions that
do not read the other table outside the ``EXISTS``.

The rule fires on a ``SELECT DISTINCT`` over exactly two base tables under an inner or cross join, with no
grouping, aggregate, window, ``ORDER BY`` or ``LIMIT``, whose outputs are plain columns as described and include
every column of a declared key of the keyed table; every key column must be declared NOT NULL and not
floating-point. Every column of the select, at any depth, must name its source in full, and no nested query may
reuse the name of either table. A volatile call leaves the select as it is. Without declared keys nothing fires.
"""

from __future__ import annotations

from sqlglot import exp

from . import smt_values

_BANNED = ("group", "having", "qualify", "windows", "with_", "with", "limit", "offset", "order", "laterals", "pivots", "into", "locks", "sample")
_JOIN_EXTRAS = ("method", "using", "match_condition", "global_", "hint", "directed", "expressions", "pivots")


def _unparen(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _plain_table(source: exp.Expression) -> bool:
    return isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier) and not any(source.args.get(k) for k in ("db", "catalog", "pivots", "sample", "joins", "version", "when"))


def _column(node: exp.Expression) -> tuple[str, str] | None:
    node = _unparen(node)
    if isinstance(node, exp.Column) and isinstance(node.this, exp.Identifier) and node.table and not node.args.get("db"):
        return node.table.lower(), node.name.lower()
    return None


def distinct_key_join_to_exists(select: exp.Select, keys: dict | None, not_null: dict | None, types: dict | None = None) -> exp.Expression | None:
    from .algebraic_equivalence import _conjuncts
    from .counted_membership import _unstable

    distinct = select.args.get("distinct")
    from_ = select.args.get("from_") or select.args.get("from")
    joins = select.args.get("joins") or []
    if not keys or not not_null or distinct is None or distinct.args.get("on") or from_ is None or len(joins) != 1 or any(select.args.get(k) for k in _BANNED):
        return None
    join = joins[0]
    if join.side or (join.kind or "").upper() not in ("", "INNER", "CROSS") or any(join.args.get(k) for k in _JOIN_EXTRAS):
        return None
    first, second = from_.this, join.this
    if not _plain_table(first) or not _plain_table(second):
        return None
    names = {first.alias_or_name.lower(): first, second.alias_or_name.lower(): second}
    if len(names) != 2:
        return None
    if any(node.find_ancestor(exp.Select) is select for node in select.find_all(exp.AggFunc, exp.Window)) or select.find(exp.Star) is not None:
        return None
    if any(isinstance(c.this, exp.Star) or not c.table or c.args.get("db") for c in select.find_all(exp.Column)):
        return None
    for nested in select.find_all(exp.Select):
        if nested is select:
            continue
        inner_from = nested.args.get("from_") or nested.args.get("from")
        sources = ([inner_from.this] if inner_from is not None else []) + [j.this for j in nested.args.get("joins") or []]
        if any((s.alias_or_name or "").lower() in names or not (s.alias_or_name or "") for s in sources):
            return None  # a nested query that reuses a name could read another row under it
    conditions = [*(_conjuncts(join.args["on"]) if join.args.get("on") is not None else []), *(_conjuncts(select.args["where"].this) if select.args.get("where") is not None else [])]
    equal: dict[tuple[str, str], tuple[str, str]] = {}
    for condition in conditions:
        condition = _unparen(condition)
        if isinstance(condition, exp.EQ):
            a, b = _column(condition.this), _column(condition.expression)
            if a is not None and b is not None and a[0] in names and b[0] in names and a[0] != b[0]:
                equal.setdefault(a, b)
                equal.setdefault(b, a)
    declared = {t.lower(): {c.lower() for c in cs} for t, cs in not_null.items()}
    table_keys = {t.lower(): [tuple(c.lower() for c in k) for k in ks if k] for t, ks in keys.items()}
    kinds = {t.lower(): {c.lower(): k for c, k in cs.items()} for t, cs in (types or {}).items()}
    for keyed in names:
        other = next(n for n in names if n != keyed)
        table = names[keyed].name.lower()
        outputs: list[tuple[str, str]] = []
        for item in select.expressions:
            found = _column(item.unalias())
            if found is not None and found[0] == other:
                found = equal.get(found)
            if found is None or found[0] != keyed:
                outputs = []
                break
            outputs.append(found)
        listed = {column for _, column in outputs}
        usable = [
            key
            for key in table_keys.get(table, [])
            if set(key) <= listed and all(c in declared.get(table, ()) and smt_values.type_class(kinds.get(table, {}).get(c)) != smt_values.FLOAT64 for c in key)
        ]
        if not outputs or not usable:
            continue
        if _unstable(select):
            return None
        outside = [c for c in conditions if not any(col.table.lower() == other for col in c.find_all(exp.Column))]
        inside = [c for c in conditions if any(col.table.lower() == other for col in c.find_all(exp.Column))]
        if not inside:
            return None  # nothing ties the two tables: a cross product, left to the other rules
        probe = exp.select("1").from_(names[other].copy())
        probe.set("where", exp.Where(this=exp.and_(*[c.copy() for c in inside], copy=False)))
        items = []
        for item, (_, column) in zip(select.expressions, outputs):
            value = exp.column(column, table=names[keyed].alias_or_name)
            items.append(value if item.alias_or_name.lower() == column and not isinstance(item, exp.Alias) else exp.alias_(value, item.alias_or_name, quoted=False))
        result = exp.select(*items).from_(names[keyed].copy())
        result.set("where", exp.Where(this=exp.and_(*[c.copy() for c in outside], exp.Exists(this=probe), copy=False)))
        return result
    return None
