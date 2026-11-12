"""The count guards Calcite leaves around an expanded ``NOT IN`` are dropped when they decide nothing.

Calcite's ``SubQueryRemoveRule`` writes ``x NOT IN (SELECT k FROM t WHERE p)`` as joins:

    FROM a
    CROSS JOIN (SELECT COUNT(*) AS c, COUNT(k) AS ck FROM t WHERE p) AS g
    LEFT JOIN (SELECT k, TRUE AS i FROM t WHERE p GROUP BY k) AS m ON a.x = m.k
    WHERE g.c = 0 OR m.i IS NULL AND g.ck >= g.c

``g.c = 0`` says the subquery is empty, ``m.i IS NULL`` that no row matches, and ``g.ck >= g.c`` that the subquery
holds no NULL. When ``k`` is declared NOT NULL Calcite writes ``COUNT(*)`` for both counts. The normalizer already
reads the indicator join as ``NOT EXISTS (SELECT 1 FROM t WHERE p AND t.k = a.x)``, which leaves

    FROM a CROSS JOIN (SELECT COUNT(*) AS c, COUNT(*) AS ck FROM t WHERE p) AS g
    WHERE g.c = 0 OR NOT EXISTS (SELECT 1 FROM t WHERE p AND t.k = a.x) AND g.ck >= g.c

``drop_count_guards`` rewrites that conjunct to the ``NOT EXISTS`` alone and, when nothing else reads ``g``, drops
the cross join:

* ``g.ck >= g.c`` is TRUE when both are ``COUNT(*)`` of the one select, or ``ck`` counts a column declared NOT NULL:
  the two counts are then the same number, and a count is never NULL.
* ``g.c = 0 OR NOT EXISTS (S)`` is ``NOT EXISTS (S)`` when every row ``S`` can return is a row ``c`` counts: ``S``
  reads the same table and nothing else, and its ``WHERE`` holds every conjunct of ``p``. If ``c`` is 0 no row of
  the table passes ``p``, so ``S`` is empty and ``NOT EXISTS`` is TRUE, which is what the disjunction gives; if
  ``c`` is not 0 the first disjunct is FALSE (never NULL) and the disjunction is the ``NOT EXISTS``.
* ``g`` is a select of aggregates with no ``GROUP BY`` and no ``HAVING``: exactly one row, whatever the table
  holds, so a cross join with it repeats no row and drops none.

Conditions, all checked: ``g`` is joined with ``CROSS JOIN`` or a comma, has an alias without a column list, and
its select has only ``COUNT(*)`` / ``COUNT(column)`` outputs with distinct names, one base table, an optional
``WHERE`` and nothing else; ``p`` and the probe's copy of it hold no subquery and nothing volatile (the two are
evaluated separately); the conjunct is exactly ``g.c = 0 OR (NOT EXISTS (..) [AND guard ..])`` with every other
conjunct of the right side a guard as above; every column of ``p`` belongs to the counted table, and in the probe a
conjunct counts as a copy of one of ``p`` only if all its columns belong to the probe's table (qualified with its
name, or bare and listed for that table in the schema). The join is dropped only if no other expression of the
select reads ``g`` (``unmatched_outer_join._unread``); otherwise the conjunct is simplified and the join stays.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import same_table, select_sources
from .unmatched_outer_join import _columns_of, _conjuncts, _plain_column, _plain_table, _unread, _unwrap

_COUNTED_ARGS = frozenset({"expressions", "from_", "from", "where"})
_PROBE_ARGS = frozenset({"expressions", "from_", "from", "where"})


def _and_all(parts: list[exp.Expression]) -> exp.Expression:
    parts = [exp.Paren(this=p) if isinstance(p, (exp.Or, exp.Xor)) else p for p in parts]  # they bind looser than AND
    result = parts[0]
    for part in parts[1:]:
        result = exp.And(this=result, expression=part)
    return result


def _unstable(node: exp.Expression) -> bool:
    from .counted_membership import _unstable as unstable

    return unstable(node) or any(isinstance(n, (exp.Query, exp.Subquery, exp.Exists, exp.Window, exp.AggFunc)) for n in node.walk())


def _local_text(part: exp.Expression, name: str, columns: set[str]) -> str | None:
    """``part`` with its columns unqualified and folded, when every column in it belongs to the table ``name``."""

    holder = exp.Paren(this=part.copy())
    for column in list(holder.find_all(exp.Column)):
        plain = _plain_column(column)
        if plain is None or plain.table.lower() not in ("", name) or plain.name.lower() not in columns:
            return None
        column.replace(exp.column(plain.name.lower()))
    for identifier in holder.find_all(exp.Identifier):
        identifier.set("quoted", False)
    return holder.this.sql()


def _counts(source: exp.Expression, declared: dict[str, set[str]], never_null: dict[str, set[str]]):
    """``(alias, table, filter texts, {output: counts every row})`` of a one-row derived table of counts."""

    if type(source) is not exp.Subquery or not source.alias:
        return None
    alias = source.args.get("alias")
    if alias is None or alias.args.get("columns") or any(v for key, v in source.args.items() if key not in ("this", "alias")):
        return None
    inner = source.this
    if type(inner) is not exp.Select or any(v for key, v in inner.args.items() if key not in _COUNTED_ARGS):
        return None
    sources = select_sources(inner)
    if len(sources) != 1 or not _plain_table(sources[0]):
        return None
    table = sources[0]
    name = (table.alias_or_name or "").lower()
    columns = _columns_of(table, declared)
    if not name or not columns:
        return None
    full = ".".join(part.name for part in table.parts).lower()
    not_null = never_null.get(full) or never_null.get(table.name.lower()) or set()
    outputs: dict[str, bool] = {}
    for item in inner.expressions:
        value = item.this if isinstance(item, exp.Alias) else item
        output = (item.alias or "").lower()
        if not output or output in outputs or type(value) is not exp.Count:
            return None
        if any(v for key, v in value.args.items() if key not in ("this", "big_int")):
            return None  # DISTINCT arrives as the argument; FILTER and the like as other nodes
        counted = value.this
        if isinstance(counted, exp.Star):
            outputs[output] = True
        else:
            column = _plain_column(counted)
            if column is None or column.table.lower() not in ("", name) or column.name.lower() not in columns:
                return None
            outputs[output] = column.name.lower() in not_null
    where = inner.args.get("where")
    texts = []
    for part in _conjuncts(where.this) if where is not None else []:
        text = None if _unstable(part) else _local_text(part, name, columns)
        if text is None:
            return None
        texts.append(text)
    return source.alias.lower(), table, texts, outputs


def _count_column(node: exp.Expression, alias: str, outputs: dict[str, bool]) -> str | None:
    column = _plain_column(node)
    if column is None or column.table.lower() != alias or column.name.lower() not in outputs:
        return None
    return column.name.lower()


def _is_zero(node: exp.Expression) -> bool:
    node = _unwrap(node)
    return type(node) is exp.Literal and not node.is_string and node.this == "0"


def _true_guard(part: exp.Expression, alias: str, outputs: dict[str, bool]) -> bool:
    """``g.ck >= g.c`` (or the same spelled ``<=`` / ``=``) between two outputs that both count every row."""

    part = _unwrap(part)
    if type(part) not in (exp.GTE, exp.LTE, exp.EQ):
        return False
    left, right = _count_column(part.this, alias, outputs), _count_column(part.expression, alias, outputs)
    return left is not None and right is not None and outputs[left] and outputs[right]


def _probe_within(node: exp.Expression, table: exp.Table, texts: list[str], declared: dict[str, set[str]], dialect: str) -> bool:
    """Whether ``node`` is ``NOT EXISTS (S)`` and every row of ``S`` is a row of ``table`` that passes ``texts``."""

    node = _unwrap(node)
    if type(node) is not exp.Not or type(_unwrap(node.this)) is not exp.Exists:
        return False
    probe = _unwrap(node.this).this
    if type(probe) is exp.Subquery and not probe.alias:
        probe = probe.this
    if type(probe) is not exp.Select or any(v for key, v in probe.args.items() if key not in _PROBE_ARGS):
        return False
    if any(isinstance(n, (exp.AggFunc, exp.Window)) for item in probe.expressions for n in item.walk()):
        return False  # an aggregate makes a row out of no rows
    sources = select_sources(probe)
    if len(sources) != 1 or not _plain_table(sources[0]) or not same_table(sources[0], table, dialect):
        return False
    name = (sources[0].alias_or_name or "").lower()
    columns = _columns_of(sources[0], declared)
    if not name or not columns:
        return False
    where = probe.args.get("where")
    have = set()
    for part in _conjuncts(where.this) if where is not None else []:
        if not _unstable(part):
            text = _local_text(part, name, columns)
            if text is not None:
                have.add(text)
    return all(text in have for text in texts)


def _simplified(test: exp.Expression, alias: str, table: exp.Table, texts: list[str], outputs: dict[str, bool], declared, dialect: str) -> exp.Expression | None:
    """The ``NOT EXISTS`` of ``g.c = 0 OR NOT EXISTS (..) AND guards``, or ``None``."""

    test = _unwrap(test)
    if type(test) is not exp.Or:
        return None
    for empty, rest in ((test.this, test.expression), (test.expression, test.this)):
        empty = _unwrap(empty)
        if type(empty) is not exp.EQ:
            continue
        sides = (empty.this, empty.expression)
        counted = [(_count_column(a, alias, outputs), b) for a, b in (sides, sides[::-1])]
        if not any(name is not None and outputs[name] and _is_zero(zero) for name, zero in counted):
            continue
        parts = _conjuncts(rest)
        probes = [part for part in parts if _probe_within(part, table, texts, declared, dialect)]
        if len(probes) != 1:
            continue
        if all(part is probes[0] or _true_guard(part, alias, outputs) for part in parts):
            return _unwrap(probes[0]).copy()
    return None


def drop_count_guards(select: exp.Select, schema: dict | None, not_null: dict | None, dialect: str = "bigquery") -> exp.Expression | None:
    """``select`` with the count guards of one expanded ``NOT IN`` dropped (see the module docstring), or ``None``."""

    joins = select.args.get("joins") or []
    where = select.args.get("where")
    if not joins or where is None or not schema or select.args.get("laterals"):
        return None
    if any(j.args.get("using") or j.args.get("method") for j in joins):
        return None
    declared = {str(t).lower(): {str(c).lower() for c in columns} for t, columns in schema.items()}
    never_null = {str(t).lower(): {str(c).lower() for c in columns} for t, columns in (not_null or {}).items()}
    names = [(s.alias_or_name or "").lower() for s in select_sources(select)]
    tests = _conjuncts(where.this)
    for index, join in enumerate(joins):
        if join.args.get("side") or join.args.get("on") is not None or (join.args.get("kind") or "").upper() not in ("", "CROSS"):
            continue
        if any(v for key, v in join.args.items() if key not in ("this", "kind")):
            continue
        counts = _counts(join.this, declared, never_null)
        if counts is None or names.count(counts[0]) != 1:
            continue
        alias, table, texts, outputs = counts
        replaced = {}
        for position, test in enumerate(tests):
            simple = _simplified(test, alias, table, texts, outputs, declared, dialect)
            if simple is not None:
                replaced[position] = simple
        if not replaced:
            continue
        copy = select.copy()
        parts = [replaced.get(position) or part.copy() for position, part in enumerate(_conjuncts(copy.args["where"].this))]
        copy.set("where", exp.Where(this=_and_all(parts)))
        without = copy.copy()
        without.args["joins"][index].pop()
        if not without.args.get("joins"):
            without.set("joins", None)
        if _unread(without, exp.Null(), alias, set(outputs), declared):  # nothing to skip: the join is gone
            return without
        return copy
    return None
