"""``k NOT IN (SELECT k FROM t JOIN b ON c)`` over the same table's own key is ``NOT EXISTS (SELECT 1 FROM b WHERE c)``.

    SELECT id FROM playback WHERE id NOT IN (SELECT p.id FROM playback AS p JOIN ads AS a ON p.cid = a.cid AND a.ts >= p.start)
    SELECT id FROM playback AS p WHERE NOT EXISTS (SELECT 1 FROM ads AS a WHERE p.cid = a.cid AND a.ts >= p.start)

are the same query. The second is the form the prover reads best, because ``LEFT JOIN b ON c WHERE b.x IS NULL``
turns into it too, and so the common spellings of "the rows with no match" (a ``LEFT JOIN`` with an ``IS NULL`` test, a
``NOT EXISTS`` and a ``NOT IN`` over the key) meet.

**Why they are the same.** ``k`` is a single-column declared key of ``t``, NOT NULL, so the tested value is never NULL, the
subquery never returns a NULL, and ``k NOT IN (..)`` is TRUE or FALSE, never NULL. It is FALSE exactly when some row of
the subquery has ``k`` equal to the outer row's. A row of ``t`` with that key is the outer row itself (the key is
unique), so the subquery has such a row exactly when the outer row, in place of ``p``, finds a row of the joined tables
that passes ``c`` and the ``WHERE``. That is ``EXISTS``. The joins of the subquery are inner (or turned inner by a
``WHERE`` that rejects their padded rows), so their ``ON`` conditions are ``WHERE`` conditions.

**What is read.** One ``NOT IN`` of a column that is a NOT NULL single-column key of the select's only source (one base
table, no joins); the subquery selects (with or without ``DISTINCT``) the same key of an alias of the same table; the other sources of the
subquery are plain tables joined with ``JOIN .. ON`` or ``USING`` (one column, directly after the table), with no grouping,
limit, subquery or window anywhere in it. Every bare column of its conditions must be one the schema shows belongs to
exactly one source.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key, _unwrap
from .ast_utils import conjuncts, select_sources
from .null_rejecting_joins import left_join_to_inner

_SUBQUERY_ARGS = frozenset({"expressions", "from", "from_", "joins", "where", "distinct", "hint", "comments", "kind", "operation_modifiers"})


def rewrite_anti_membership(select: exp.Select, schema: dict[str, list[str]] | None, keys: dict[str, list[tuple[str, ...]]] | None, not_null: dict[str, frozenset[str]] | None) -> exp.Expression | None:
    """The select with one such ``NOT IN`` replaced by ``NOT EXISTS``, else None."""

    where = select.args.get("where")
    if not schema or not keys or where is None or select.args.get("joins") or not any(isinstance(n, exp.In) for n in where.walk()):
        return None
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    sources = select_sources(select)
    if len(sources) != 1 or _base_table(sources[0], known) is None or not sources[0].alias_or_name:
        return None
    table = sources[0]
    name = _table_key(table)
    declared = {c.lower() for c in (not_null or {}).get(name, ())}
    singles = {k[0].lower() for k in keys.get(name, []) if len(k) == 1 and k[0].lower() in declared}
    if not singles:
        return None
    out = select.copy()
    for node in list(out.args["where"].walk()):
        if not isinstance(node, exp.In) or not isinstance(node.parent, exp.Not) or node.args.get("expressions") or node.args.get("unnest"):
            continue
        if node.find_ancestor(exp.Select) is not out:
            continue
        probe = _probe(node, out, known, singles, name)
        if probe is not None:
            node.replace(exp.Exists(this=probe))
            return out
    return None


def _probe(node: exp.In, outer: exp.Select, known: dict[str, list[str]], singles: set[str], name: str) -> exp.Select | None:
    table = select_sources(outer)[0]
    alias = table.alias_or_name.lower()
    value = _unwrap(node.this)
    if not isinstance(value, exp.Column) or isinstance(value.this, exp.Star) or value.table.lower() not in ("", alias) or value.name.lower() not in singles:
        return None
    query = node.args.get("query")
    inner = query.this if isinstance(query, exp.Subquery) and not query.alias else None
    if not isinstance(inner, exp.Select) or any(v for k, v in inner.args.items() if k not in _SUBQUERY_ARGS) or len(inner.expressions) != 1:
        return None
    if any(isinstance(n, (exp.AggFunc, exp.Window, exp.Subquery, exp.Exists, exp.Select)) for n in inner.walk() if n is not inner):
        return None
    if isinstance(inner.args.get("distinct"), exp.Distinct) and inner.args["distinct"].args.get("on"):
        return None
    if any((j.args.get("side") or "").upper() == "LEFT" for j in inner.args.get("joins") or []):
        converted = left_join_to_inner(inner)
        inner = converted if converted is not None else inner
    inner = inner.copy()
    sources = select_sources(inner)
    joins = inner.args.get("joins") or []
    if len(sources) < 2 or len(sources) != len(joins) + 1:
        return None
    first = sources[0]
    if _base_table(first, known) is None or _table_key(first) != name or not first.alias_or_name:
        return None
    local = first.alias_or_name.lower()
    others = sources[1:]
    columns = {}
    aliases = [local]
    for source in others:
        found = _base_table(source, known)
        label = (source.alias_or_name or "").lower()
        if found is None or not label or label in aliases or label == alias:
            return None
        aliases.append(label)
        columns[label] = set(found)
    own = set(known[name])
    conditions = []
    for index, join in enumerate(joins):
        kind = join.args.get("kind")
        if join.args.get("side") or join.args.get("method") or join.args.get("global_") or (kind.upper() if isinstance(kind, str) else kind) not in (None, "", "INNER", "CROSS"):
            return None
        using = join.args.get("using")
        if using is not None:
            if index != 0 or len(using) != 1 or join.args.get("on") is not None:
                return None
            column = using[0].name.lower()
            if column not in own or column not in columns[aliases[1]]:
                return None
            conditions.append(exp.EQ(this=exp.column(column, table=local), expression=exp.column(column, table=aliases[1])))
        elif join.args.get("on") is not None:
            conditions.append(join.args["on"])
    if inner.args.get("where") is not None:
        conditions.append(inner.args["where"].this)
    picked = _unwrap(inner.expressions[0].unalias() if isinstance(inner.expressions[0], exp.Alias) else inner.expressions[0])
    if not isinstance(picked, exp.Column) or picked.name.lower() != value.name.lower() or picked.table.lower() not in ("", local):
        return None
    if picked.table == "" and any(picked.name.lower() in cols for cols in columns.values()):
        return None
    scoped = exp.and_(*conditions) if conditions else None
    if scoped is not None:
        for column in list(scoped.find_all(exp.Column)):
            if isinstance(column.this, exp.Star) or column.args.get("db") or column.args.get("catalog"):
                return None
            qualifier = column.table.lower()
            column_name = column.name.lower()
            if qualifier == local:
                column.set("table", exp.to_identifier(table.alias_or_name))
            elif qualifier in columns or qualifier == alias:
                continue
            elif qualifier:
                return None
            else:
                homes = [label for label, cols in columns.items() if column_name in cols]
                if column_name in own and not homes:
                    column.set("table", exp.to_identifier(table.alias_or_name))
                elif len(homes) == 1 and column_name not in own:
                    column.set("table", exp.to_identifier(homes[0]))
                else:
                    return None
    probe = exp.Select(expressions=[exp.Literal.number(1)])
    probe.set("from_" if "from_" in inner.args else "from", exp.From(this=others[0].copy()))
    rest = [exp.Join(this=source.copy()) for source in others[1:]]
    if rest:
        probe.set("joins", rest)
    if scoped is not None:
        probe.set("where", exp.Where(this=scoped))
    return probe
