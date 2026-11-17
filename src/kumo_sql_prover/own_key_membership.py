"""A row whose key is looked up in a list of keys of its own table is looked up as itself.

    SELECT p.pid FROM policy AS p
    WHERE p.pid IN (SELECT a.pid FROM policy AS a JOIN policy AS b ON a.pid <> b.pid AND a.city = b.city)

asks whether some row ``a`` of ``policy`` with the key of ``p`` has a partner ``b``. A key names one row, so ``a`` is
``p`` itself, and the test is

    EXISTS (SELECT 1 FROM policy AS b WHERE p.pid <> b.pid AND p.city = b.city)

``anti_membership_joins`` reads ``NOT IN`` this way for a select with one source and a subquery without nested
queries. This rule also reads ``IN``, a row that is one of several sources, and a subquery whose conditions hold
nested queries of their own (``.. WHERE a.pid NOT IN (SELECT c.pid FROM policy AS c JOIN ..)``), which is how the
question "rows that share one value with another row but not a second value" is often written.

**Why it holds.** ``k`` is a declared one-column key of ``T`` that is NOT NULL, ``r`` a row of ``T`` the outer select
reads, and ``Q`` the subquery ``SELECT a.k FROM T AS a, others WHERE C`` (inner joins; their ``ON`` conditions are
``WHERE`` conditions). ``r.k`` is not NULL and no value of ``Q`` is NULL, so ``r.k IN (Q)`` is TRUE or FALSE, never
NULL. It is TRUE exactly when some combination of a row ``a`` and rows of the other sources passes ``C`` with
``a.k = r.k``. The only row of ``T`` with that key is ``r``, so this is: some rows of the other sources pass ``C``
with ``r`` in place of ``a``. That is ``EXISTS (SELECT 1 FROM others WHERE C[a := r])``, and ``NOT IN`` is its
negation. ``DISTINCT`` in ``Q`` changes nothing. Every read of ``a`` in ``C`` becomes a read of ``r``, also inside
nested queries of ``C``, which see the same row either way. A conjunct of ``C[a := r]`` that reads none of the other
sources has one value for all their rows, so for ``IN`` it is written beside the ``EXISTS`` (the conjunction is
TRUE exactly when both are); under ``NOT IN`` it stays inside, where a NULL counts as "no row".

When ``Q`` has no other source (``SELECT a.k FROM T AS a WHERE C``), ``IN`` is TRUE exactly when ``C[a := r]`` is
TRUE, and as a top-level conjunct of the ``WHERE`` it is replaced by the conjuncts of ``C``; ``NOT IN`` is left
alone there (``C`` may be NULL).

**What is read.** A top-level conjunct of the ``WHERE``: ``IN`` or ``NOT IN`` of a column that is a declared
one-column NOT NULL key of a base table ``r`` of the select (the first source, or one joined by an inner or cross
join) or of a query around it (when the select sits in the ``WHERE`` of each query on the way out, so that it is
evaluated for a real row ``r``), against a subquery that selects that key of the same base table, with inner, cross
or comma joins and no grouping, aggregate, window, ``ORDER BY`` or ``LIMIT`` of its own.

**Left alone.** A name whose owner is not certain: a bare column anywhere in the subquery (except in a subquery
with the one source, whose own bare columns are its table's), a ``WITH`` inside it, a read of ``a`` from inside a
derived table. A name that would change its meaning: a query, between the read and the outer select, that has a
source called like ``r``. A key declared floating-point; a ``RIGHT`` or ``FULL`` join or ``USING`` in either select; a
row on the padded side of a ``LEFT JOIN``; a test under ``OR`` or outside the ``WHERE``.

``tests/test_own_key_membership.py`` replays the rewrite on DuckDB and shows the key is needed.
"""

from __future__ import annotations

from sqlglot import exp

from . import smt_values
from .own_group_membership import _JOIN_EXTRAS, _facts, _from, _listed, _plain_table, _real_sources, _unparen

_ALLOWED = frozenset({"expressions", "from", "from_", "joins", "where", "distinct", "hint", "comments", "kind", "operation_modifiers"})


def _names(select: exp.Select) -> list[str]:
    first = _from(select)
    sources = ([first] if first is not None else []) + [j.this for j in select.args.get("joins") or []]
    return [(s.alias_or_name or "").lower() for s in sources]


def _bound_here(column: exp.Column, query: exp.Select, alias: str) -> bool:
    """Whether ``alias`` in ``column`` means the source of ``query`` (no query in between has a source of that name)."""

    scope = column.find_ancestor(exp.Select)
    while scope is not None and scope is not query:
        if alias in _names(scope):
            return False
        scope = scope.find_ancestor(exp.Select)
    return scope is query


def _outer_row(select: exp.Select, name: str) -> exp.Table | None:
    """The base table called ``name`` that a query around ``select`` reads as a real row, when ``select`` sits in the
    ``WHERE`` of each query on the way there (a derived table does not see the sources beside it)."""

    inner = select
    while True:
        node, through_where = inner.parent, False
        while node is not None and not isinstance(node, exp.Select):
            if isinstance(node, (exp.From, exp.Join, exp.CTE, exp.With, exp.SetOperation, exp.Lateral)):
                return None
            through_where = through_where or isinstance(node, exp.Where)
            node = node.parent
        if node is None or not through_where:
            return None
        found = _real_sources(node)
        if found is None:
            return None
        real, names = found
        if name in names:
            return real.get(name)
        inner = node


def read_own_key_membership(select: exp.Select, schema: dict | None, keys: dict | None, not_null: dict | None, types: dict | None = None) -> exp.Select | None:
    """``select`` with one lookup of a row's key among the keys of its own table read as a test on the row itself."""

    from .algebraic_equivalence import _conjuncts

    where = select.args.get("where")
    if where is None or not schema or not keys or not not_null or where.find(exp.In) is None:
        return None
    found = _real_sources(select)
    if found is None:
        return None
    real, names = found
    conditions = _conjuncts(where.this)
    for index, condition in enumerate(conditions):
        node = _unparen(condition)
        negated = isinstance(node, exp.Not)
        if negated:
            node = _unparen(node.this)
        if not isinstance(node, exp.In) or node.args.get("expressions") or node.args.get("unnest") or node.args.get("query") is None:
            continue
        replacement = _read(select, node, negated, real, names, schema, keys, not_null, types)
        if replacement is None:
            continue
        result = select.copy()
        rebuilt = [c.copy() for c in conditions[:index]] + replacement + [c.copy() for c in conditions[index + 1 :]]
        result.set("where", exp.Where(this=exp.and_(*rebuilt, copy=False)))
        return result
    return None


def _read(select: exp.Select, node: exp.In, negated: bool, real: dict[str, exp.Table], names: list[str], schema: dict | None, keys: dict | None, not_null: dict | None, types: dict | None) -> list[exp.Expression] | None:
    from .algebraic_equivalence import _conjuncts

    value = _unparen(node.this)
    if not isinstance(value, exp.Column) or not isinstance(value.this, exp.Identifier) or value.args.get("db") or value.this.args.get("quoted"):
        return None
    key = value.name.lower()
    if value.table:
        row = real.get(value.table.lower()) if value.table.lower() in names else _outer_row(select, value.table.lower())
    else:
        owners = [alias for alias in names if alias in real and key in (_listed(real[alias], schema) or ())]
        row = real[owners[0]] if len(owners) == 1 and all(alias in real and _listed(real[alias], schema) is not None for alias in names) else None
    if row is None:
        return None
    declared = {str(c).lower() for c in (_facts(not_null, row.name) or ())}
    kinds = {str(k).lower(): v for k, v in (_facts(types, row.name) or {}).items()}
    if key not in declared or smt_values.type_class(kinds.get(key)) == smt_values.FLOAT64:
        return None
    if not any(len(k) == 1 and str(k[0]).lower() == key for k in (_facts(keys, row.name) or ())):
        return None
    query = node.args["query"]
    query = query.this if isinstance(query, exp.Subquery) and not query.alias else query
    if not isinstance(query, exp.Select) or any(v for k, v in query.args.items() if k not in _ALLOWED) or len(query.expressions) != 1:
        return None
    distinct = query.args.get("distinct")
    if (distinct is not None and distinct.args.get("on")) or query.find(exp.With) is not None:
        return None
    if any(isinstance(n, (exp.AggFunc, exp.Window)) and n.find_ancestor(exp.Select) is query for n in query.walk()):
        return None
    first = _from(query)
    joins = query.args.get("joins") or []
    if first is None:
        return None
    for join in joins:
        kind = join.args.get("kind")
        kind = kind.upper() if isinstance(kind, str) else ""
        if join.args.get("side") or any(join.args.get(k) for k in _JOIN_EXTRAS) or kind not in ("", "INNER", "CROSS"):
            return None
    sources = [first] + [j.this for j in joins]
    if any(not isinstance(s, (exp.Table, exp.Subquery)) for s in sources):
        return None  # a lateral source, a table function
    labels = _names(query)
    if "" in labels or len(set(labels)) != len(labels):
        return None
    picked = _unparen(query.expressions[0].unalias())
    if not isinstance(picked, exp.Column) or not isinstance(picked.this, exp.Identifier) or picked.args.get("db") or picked.name.lower() != key:
        return None
    if picked.table:
        local = picked.table.lower()
    elif len(sources) == 1:
        local = labels[0]
    else:
        return None
    if local not in labels:
        return None
    same = sources[labels.index(local)]
    if not _plain_table(same) or same.name != row.name or bool(same.this.args.get("quoted")) != bool(row.this.args.get("quoted")):
        return None
    listed = _listed(same, schema)
    if listed is None:
        return None
    others = [s for s, label in zip(sources, labels) if label != local]
    alias = row.args.get("alias")
    mine = alias.this if isinstance(alias, exp.TableAlias) else row.this
    target = mine.name.lower()
    if target in [label for label in labels if label != local]:
        return None  # a source of the subquery would take the row's name
    copy = query.copy()
    scoped = [j.args["on"] for j in copy.args.get("joins") or [] if j.args.get("on") is not None]
    if copy.args.get("where") is not None:
        scoped.append(copy.args["where"].this)
    derived = [s for s in [_from(copy)] + [j.this for j in copy.args.get("joins") or []] if not isinstance(s, exp.Table)]
    for column in [c for part in scoped for c in part.find_all(exp.Column)]:
        if isinstance(column.this, exp.Star) or column.args.get("db") or column.args.get("catalog"):
            return None
        if not column.table:
            if len(sources) == 1 and column.find_ancestor(exp.Select) is copy and column.name.lower() in listed:
                column.set("table", mine.copy())
                continue
            return None  # whose column this is depends on what the sources hold
        if column.table.lower() != local or not _bound_here(column, copy, local):
            continue
        column.set("table", mine.copy())
        scope = column.find_ancestor(exp.Select)
        while scope is not None and scope is not copy:
            if target in _names(scope):
                return None  # the row's name means another source in there
            scope = scope.find_ancestor(exp.Select)
    for source in derived:
        if any(c.table.lower() == local and _bound_here(c, copy, local) for c in source.find_all(exp.Column) if c.table):
            return None
    parts = [c for part in scoped for c in _conjuncts(part)]
    if not others:
        return None if negated or not parts else parts
    kept = [s for s in [_from(copy)] + [j.this for j in copy.args.get("joins") or []] if (s.alias_or_name or "").lower() != local]
    beside: list[exp.Expression] = []
    if not negated:
        # a condition that reads none of the other sources is the same for each of their rows: it is a test on the
        # row itself, beside the ``EXISTS`` (both must be TRUE either way; under ``NOT`` a NULL would differ)
        rest = [label for label in labels if label != local]
        reads = [any(c.table and c.table.lower() in rest and _bound_here(c, copy, c.table.lower()) for c in part.find_all(exp.Column)) for part in parts]
        beside = [part for part, inside in zip(parts, reads) if not inside]
        parts = [part for part, inside in zip(parts, reads) if inside]
    probe = exp.Select(expressions=[exp.Literal.number(1)])
    probe.set("from_" if "from_" in select.args else "from", exp.From(this=kept[0]))
    if kept[1:]:
        probe.set("joins", [exp.Join(this=source) for source in kept[1:]])
    if parts:
        probe.set("where", exp.Where(this=exp.and_(*parts, copy=False)))
    exists: exp.Expression = exp.Exists(this=probe)
    return [exp.Not(this=exists)] if negated else [exists, *beside]
