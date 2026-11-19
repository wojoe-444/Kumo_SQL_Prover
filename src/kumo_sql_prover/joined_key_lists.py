"""A list drawn from a table the row is joined to on its key is a test of the joined row.

    SELECT s.buyer FROM sale AS s JOIN item AS i ON s.item_id = i.item_id
    WHERE s.item_id IN (SELECT x.item_id FROM item AS x WHERE x.label = 'S' OR x.label = 'I')

asks whether an item with the sale's ``item_id`` has one of two labels. With ``item_id`` a declared key of
``item`` there is one such item, and the join has already found it:

    SELECT s.buyer FROM sale AS s JOIN item AS i ON s.item_id = i.item_id
    WHERE (i.label = 'S' OR i.label = 'I')

**Why it holds.** On every row that reaches the ``WHERE``, ``s.item_id = i.item_id`` is TRUE (a top-level
conjunct of an inner join's ``ON`` or of the ``WHERE``, in a select whose joins are all inner, cross or comma
joins, so no row is padded), so both sides are not NULL. The ``IN`` is TRUE exactly when some row ``x`` of
``item`` has ``x.item_id`` equal to the tested value and passes the list's ``WHERE``. ``i`` is a row of ``item``
with that key value, and the key says no other row has it, so the ``IN`` is TRUE exactly when the list's
``WHERE`` is TRUE on ``i``. Where it is not TRUE the ``IN`` may be FALSE or NULL and the test of ``i`` FALSE or
NULL; a top-level conjunct of a ``WHERE`` only asks for TRUE. A list with no ``WHERE`` is always TRUE there and is
dropped.

**What is read.** A top-level conjunct ``X IN (SELECT [DISTINCT] x.k FROM T AS x [WHERE F])`` of the ``WHERE``
(not ``NOT IN``, not under ``OR``), where ``(k)`` is a declared one-column key of the base table ``T``; another
source ``i`` of the select reads ``T``; a top-level equality ``a.c = i.k`` ties it to a column of another
source; ``X`` is ``a.c`` or ``i.k``. The list is a plain select over ``T`` alone whose ``WHERE`` has no nested
query, aggregate or window and reads only columns that name ``x`` (bare names are given the table when the
caller supplied its columns). ``nested_correlation.interchangeable`` must let the tied columns be read one for
the other (declared types that differ, or floating-point, are left alone). A volatile call or a quoted name
leaves the select as it is.

A narrow retry rule (``shape_fallback``). ``tests/test_joined_key_lists.py`` replays the rewrite on DuckDB with
the key kept and with the key broken.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, conjuncts
from .nested_correlation import interchangeable
from .own_group_membership import _facts, _plain_table, _unparen

_JOIN_EXTRAS = ("method", "using", "match_condition", "global_", "hint", "directed", "expressions", "pivots")
_LIST_PARTS = frozenset({"expressions", "from", "from_", "where", "distinct", "comments", "kind", "hint"})
_OPAQUE = (exp.Query, exp.Subquery, exp.AggFunc, exp.Window, exp.Exists, exp.Star)


def _named(column: exp.Expression | None) -> tuple[str, str] | None:
    column = _unparen(column)
    if not isinstance(column, exp.Column) or not isinstance(column.this, exp.Identifier) or column.this.args.get("quoted"):
        return None
    if not column.table or column.args.get("db") or column.args.get("catalog"):
        return None
    return column.table.lower(), column.name.lower()


def test_joined_rows(select: exp.Select, keys: dict | None, schema: dict | None = None, types: dict | None = None) -> exp.Select | None:
    """``select`` with each list of its ``WHERE`` that is drawn from a joined keyed table read on the joined row."""

    from .counted_membership import _unstable
    from .keyed_semijoin import name_nested_sources, rename_nested_sources

    where = select.args.get("where")
    from_ = select.args.get(FROM_KEY)
    joins = select.args.get("joins") or []
    if not keys or where is None or from_ is None or not joins or where.find(exp.In) is None:
        return None
    for join in joins:
        if join.side or (join.kind or "").upper() not in ("", "INNER", "CROSS") or any(join.args.get(key) for key in _JOIN_EXTRAS):
            return None
    sources = [from_.this] + [join.this for join in joins]
    names = [(source.alias_or_name or "").lower() for source in sources]
    if not all(names) or len(set(names)) != len(names):
        return None
    select = name_nested_sources(rename_nested_sources(select, set(names)), schema)
    where, from_, joins = select.args["where"], select.args[FROM_KEY], select.args.get("joins") or []
    sources = [from_.this] + [join.this for join in joins]
    by_name = dict(zip(names, sources))
    ties = [part for join in joins if join.args.get("on") is not None for part in conjuncts(join.args["on"])] + conjuncts(where.this)
    equal: set[frozenset] = set()
    for part in ties:
        test = _unparen(part)
        if isinstance(test, exp.EQ):
            a, b = _named(test.this), _named(test.expression)
            if a is not None and b is not None and a[0] != b[0] and a[0] in by_name and b[0] in by_name:
                equal.add(frozenset((a, b)))
    if not equal:
        return None
    parts = conjuncts(where.this)
    kept, changed = [], False
    for part in parts:
        reading = _joined_test(_unparen(part), by_name, equal, keys, types)
        if reading is None:
            kept.append(part.copy())
            continue
        changed = True
        if reading is not True:
            kept.append(reading)
    if not changed or _unstable(select):
        return None
    result = select.copy()
    result.set("where", exp.Where(this=exp.and_(*kept, copy=False)) if kept else None)
    return result


def _joined_test(test: exp.Expression | None, by_name: dict[str, exp.Expression], equal: set[frozenset], keys: dict, types: dict | None):
    """The list's ``WHERE`` on the joined row, ``True`` for a list without one, None when ``test`` is not read."""

    if not isinstance(test, exp.In) or test.args.get("expressions") or test.args.get("unnest") or test.args.get("field"):
        return None
    tested = _named(test.this)
    query = test.args.get("query")
    query = query.this if isinstance(query, exp.Subquery) and not query.alias else query
    if tested is None or tested[0] not in by_name or not isinstance(query, exp.Select) or any(v for k, v in query.args.items() if k not in _LIST_PARTS):
        return None
    distinct = query.args.get("distinct")
    source = query.args.get(FROM_KEY)
    source = source.this if source is not None else None
    if (distinct is not None and distinct.args.get("on")) or not _plain_table(source) or source.this.args.get("quoted") or len(query.expressions) != 1:
        return None
    theirs = (source.alias_or_name or "").lower()
    listed = _named(query.expressions[0].unalias())
    if not theirs or theirs in by_name or listed is None or listed[0] != theirs:
        return None
    table, key = source.name.lower(), listed[1]
    if not any(tuple(str(c).lower() for c in declared) == (key,) for declared in (_facts(keys, table) or ())):
        return None
    inner = query.args.get("where")
    if inner is not None:
        if any(isinstance(node, _OPAQUE) for node in inner.this.walk()):
            return None
        if any(_named(column) is None or column.table.lower() != theirs for column in inner.find_all(exp.Column)):
            return None
    for name, joined in by_name.items():
        # ``joined`` reads the list's table, and a top-level equality ties its key to a column of another source
        if not _plain_table(joined) or joined.name.lower() != table or joined.this.args.get("quoted"):
            continue
        if any(joined.args.get(part) or source.args.get(part) for part in ("db", "catalog")):
            continue
        mine = (name, key)
        for pair in equal:
            if mine not in pair:
                continue
            (other,) = pair - {mine}
            if tested not in (mine, other):
                continue
            their_source = by_name[other[0]]
            their_table = their_source.name.lower() if isinstance(their_source, exp.Table) else None
            if not interchangeable(types, (their_table, other[1]), (table, key)) or not interchangeable(types, (table, key), (table, key)):
                continue
            if inner is None:
                return True
            moved = exp.Paren(this=inner.this.copy())
            for column in list(moved.find_all(exp.Column)):
                column.replace(exp.column(column.name, table=joined.alias_or_name))
            return moved if isinstance(_unparen(moved), (exp.Or, exp.Xor)) else moved.this
    return None
