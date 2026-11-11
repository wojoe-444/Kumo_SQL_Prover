"""A keyed table joined to a table that is grouped and filtered by ``HAVING``, written as one join and one grouping.

    SELECT p.id, p.name
    FROM product AS p JOIN sales AS s ON p.id = s.product_id
    GROUP BY p.id, p.name
    HAVING MIN(s.sale_date) >= '2019-01-01' AND MAX(s.sale_date) <= '2019-03-31'

lists the products all of whose sales fall in the first quarter. It is the same query as

    SELECT p.id, p.name
    FROM product AS p JOIN (SELECT s.product_id AS k FROM sales AS s GROUP BY s.product_id
                            HAVING MIN(s.sale_date) >= '2019-01-01' AND MAX(s.sale_date) <= '2019-03-31') AS g
      ON p.id = g.k

(``GROUP BY s.product_id`` and ``GROUP BY p.id`` are the same grouping on the rows of the join, and a ``GROUP BY`` that
also lists other columns of ``p`` is too, when ``id`` is a key of ``p``). The prover reads the second form, and the
first only as a join followed by a grouping, so two writers of the same question did not meet.

**Why the two are the same.** ``p`` has a declared key, so a row of ``s`` with a non-NULL ``product_id`` meets at most
one row of ``p``, and a group of the join (by the key) is the rows of ``s`` with that key value, each paired once with
that row of ``p``. A group exists exactly when the value occurs in both tables; ``MIN``, ``MAX``, ``COUNT``, ``SUM`` over
columns of ``s`` read those rows once each, as the grouped subquery does over the rows of ``s`` with that value.
The columns of ``p`` that the select lists are the one row's own. Both queries return one row per row of ``p``
whose group passes the ``HAVING``. A NULL ``product_id`` meets no row of ``p`` and equals nothing.

**What is read.** An inner join (``JOIN .. ON`` or a comma join with the equalities in ``WHERE``) of two base
tables ``s`` and ``p`` with no row limit, where a declared key of ``p`` is covered by equalities ``s.c = p.k``
(the key columns of ``p``, each against one column of ``s``; a UNIQUE key needs no NOT NULL because the equality
rejects NULL), a ``GROUP BY`` of plain columns that is, for each key column, ``p.k`` or the ``s`` column equal to
it (and may list other columns of ``p``), a ``HAVING`` of conditions over aggregates of columns of ``s``, a select
list of columns of ``p`` or the matched ``s`` columns, and a ``WHERE`` whose parts read only ``s`` (they move into
the grouped subquery) or only ``p`` (they stay). The select list holds no aggregate (queries that output the
aggregate are left to ``eager_aggregation``).
"""

from __future__ import annotations

import itertools

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key, _unwrap
from .ast_utils import conjuncts, select_sources
from .join_distinct_aggregates import _resolve

_ALLOWED = frozenset({"expressions", "from", "from_", "joins", "where", "group", "having", "order", "distinct", "hint", "comments", "kind", "operation_modifiers"})
_counter = itertools.count()
_AGGREGATES = (exp.Count, exp.Sum, exp.Min, exp.Max)


def _alias_of(source: exp.Expression) -> str:
    return source.alias_or_name.lower()


def rewrite_keyed_group_filter(select: exp.Select, schema: dict[str, list[str]] | None, keys: dict[str, list[tuple[str, ...]]] | None) -> exp.Expression | None:
    """The select rewritten as described in the module docstring, else None."""

    if not schema or not keys or select.args.get("having") is None or select.args.get("group") is None:
        return None
    if any(value for key, value in select.args.items() if key not in _ALLOWED):
        return None
    sources = select_sources(select)
    joins = select.args.get("joins") or []
    if len(sources) != 2 or len(joins) != 1:
        return None
    join = joins[0]
    kind = join.args.get("kind")
    if join.args.get("side") or join.args.get("using") or join.args.get("method") or (kind.upper() if isinstance(kind, str) else kind) not in (None, "", "INNER", "CROSS"):
        return None
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    owners: dict[str, list[str]] = {}
    tables: dict[str, exp.Table] = {}
    for source in sources:
        columns = _base_table(source, known)
        alias = _alias_of(source) if columns is not None else ""
        if columns is None or not alias or alias in owners:
            return None
        owners[alias] = columns
        tables[alias] = source
    conditions = conjuncts(join.args["on"]) if join.args.get("on") is not None else []
    where = select.args.get("where")
    where_parts = conjuncts(where.this) if where is not None else []
    pending = conditions + where_parts
    # which table is the keyed one is decided by the equalities: try each as p
    for p_alias, s_alias in ((_alias_of(sources[0]), _alias_of(sources[1])), (_alias_of(sources[1]), _alias_of(sources[0]))):
        result = _try(select, owners, tables, keys, pending, p_alias, s_alias)
        if result is not None:
            return result
    return None


def _try(select, owners, tables, keys, pending, p_alias, s_alias):
    p_table, s_table = tables[p_alias], tables[s_alias]
    p_key = _table_key(p_table)
    equalities: dict[str, str] = {}  # p column -> s column
    s_filters: list[exp.Expression] = []
    p_filters: list[exp.Expression] = []
    for condition in pending:
        inner = _unwrap(condition)
        if isinstance(inner, exp.EQ):
            left, right = _resolve(inner.this, owners), _resolve(inner.expression, owners)
            if left is not None and right is not None and {left[0], right[0]} == {p_alias, s_alias}:
                p_side, s_side = (left, right) if left[0] == p_alias else (right, left)
                if equalities.setdefault(p_side[1], s_side[1]) != s_side[1]:
                    return None
                continue
        reads = {r[0] if r else None for r in (_resolve(c, owners) for c in condition.find_all(exp.Column))}
        if None in reads or len(reads) != 1 or any(isinstance(n, (exp.Query, exp.Subquery, exp.Window, exp.Exists, exp.AggFunc, exp.Star)) for n in condition.walk()):
            return None
        (s_filters if reads == {s_alias} else p_filters).append(condition)
    declared = [tuple(c.lower() for c in k) for k in keys.get(p_key, []) if k]
    key = next((k for k in declared if all(c in equalities for c in k)), None)
    if key is None or set(equalities) != set(key):
        return None  # every tie must be one of the key's: another one would be a condition this rewrite drops
    group = select.args["group"]
    if any(value for k, value in group.args.items() if k != "expressions"):
        return None
    grouped: set[tuple[str, str]] = set()
    for item in group.expressions:
        resolved = _resolve(item, owners)
        if resolved is None:
            return None
        grouped.add(resolved)
    for column in key:
        if (p_alias, column) not in grouped and (s_alias, equalities[column]) not in grouped:
            return None
    for alias, column in grouped:  # another grouped column must be the keyed table's own, or a matched column of s
        if alias == s_alias and column not in {equalities[c] for c in key}:
            return None
    matched_s = {equalities[c]: c for c in key}  # s column -> p key column
    # the select list: columns of p, or the matched columns of s; the having: aggregates of s and constants
    for item in select.expressions:
        for node in item.walk():
            if isinstance(node, (exp.Query, exp.Subquery, exp.Window, exp.Star, exp.Exists, exp.AggFunc)):
                return None
        for column in item.find_all(exp.Column):
            resolved = _resolve(column, owners)
            if resolved is None or (resolved[0] == s_alias and resolved[1] not in matched_s):
                return None
    having = select.args["having"]
    for node in having.walk():
        if isinstance(node, (exp.Query, exp.Subquery, exp.Window, exp.Exists)) or (isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count)):
            return None
    for column in having.find_all(exp.Column):
        resolved = _resolve(column, owners)
        if resolved is None or resolved[0] != s_alias:
            return None  # a column of p in the HAVING (even a matched one) is left alone
        if not any(isinstance(a, _AGGREGATES) for a in _ancestors(column, having)):
            if resolved[1] not in matched_s:
                return None
    for aggregate in having.find_all(exp.AggFunc):
        if not isinstance(aggregate, _AGGREGATES) or isinstance(aggregate.this, exp.Distinct):
            return None
        if any(a is not aggregate and isinstance(a, exp.AggFunc) for a in aggregate.walk()):
            return None
    if select.args.get("order") is not None:
        for column in select.args["order"].find_all(exp.Column):
            resolved = _resolve(column, owners)
            if resolved is None or (resolved[0] == s_alias and resolved[1] not in matched_s):
                return None
    number = next(_counter)
    g_alias = f"kgf{number}"
    key_columns = list(key)
    s_columns = [equalities[c] for c in key_columns]
    inner = exp.select(*[exp.alias_(exp.column(c, table=s_table.alias_or_name), f"kgk{number}_{i}", quoted=False) for i, c in enumerate(s_columns)]).from_(s_table.copy())
    inner.set("group", exp.Group(expressions=[exp.column(c, table=s_table.alias_or_name) for c in s_columns]))
    inner.set("having", exp.Having(this=having.this.copy()))
    if s_filters:
        inner.set("where", exp.Where(this=exp.and_(*[f.copy() for f in s_filters])))
    outer = select.copy()
    outer.set("having", None)
    outer.set("group", None)
    outer.set("where", None)
    outer.set("joins", None)
    # read the matched columns of s off p, keeping the output's name
    for column in list(outer.find_all(exp.Column)):
        resolved = _resolve(column, owners)
        if resolved is not None and resolved[0] == s_alias:
            replacement = exp.column(matched_s[resolved[1]], table=p_table.alias_or_name)
            if column.parent is outer and matched_s[resolved[1]] != column.name.lower():
                replacement = exp.alias_(replacement, column.name, quoted=False)
            column.replace(replacement)
    ties = [exp.EQ(this=exp.column(c, table=p_table.alias_or_name), expression=exp.column(f"kgk{number}_{i}", table=g_alias)) for i, c in enumerate(key_columns)]
    outer.set("from_" if "from_" in outer.args else "from", exp.From(this=p_table.copy()))
    outer.set("joins", [exp.Join(this=exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier(g_alias))), on=exp.and_(*ties))])
    outer.set("group", None)
    outer.set("having", None)
    outer.set("where", exp.Where(this=exp.and_(*[f.copy() for f in p_filters])) if p_filters else None)
    return outer


def _ancestors(node: exp.Expression, stop: exp.Expression):
    parent = node.parent
    while parent is not None and parent is not stop:
        yield parent
        parent = parent.parent
