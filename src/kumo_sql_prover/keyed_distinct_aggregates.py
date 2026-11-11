"""``DISTINCT`` inside an aggregate removes nothing when a key keeps the argument from repeating in a group.

**The rule.** In a select that reads one base table ``t`` and groups it by plain columns ``g1 .. gn`` of ``t``
(or does not group at all, ``n = 0``), ``COUNT(DISTINCT t.c)``, ``SUM(DISTINCT t.c)`` and ``AVG(DISTINCT t.c)``
are ``COUNT(t.c)``, ``SUM(t.c)`` and ``AVG(t.c)`` when a declared key ``K`` of ``t`` lies within
``{g1 .. gn, c}`` and every column of ``K`` other than ``c`` is declared NOT NULL. MySQL's
``COUNT(DISTINCT c1, c2, ..)`` (the different tuples without a NULL) is ``COUNT(*)`` when all of ``c1, c2, ..``
are declared NOT NULL and a declared key lies within ``{g1 .. gn, c1, c2, ..}`` with its other columns NOT NULL.

    SELECT a, COUNT(DISTINCT b) FROM t WHERE p GROUP BY a      -- (a, b) a key of t, a NOT NULL
    SELECT a, COUNT(b)          FROM t WHERE p GROUP BY a

**Why it is sound.** The rows a group reads are rows of ``t`` (the ``WHERE`` only removes rows, a subset of a
keyed table is keyed), and all of them agree on ``g1 .. gn``. Take two of them whose ``c`` is not NULL and
equal. They agree on every column of ``K`` (those are among ``g1 .. gn`` and ``c``), and no column of ``K`` is
NULL in them: ``c`` is not NULL in these two rows, the others are declared NOT NULL. A key allows one row per
combination of non-NULL values, so the two rows are one row. Hence the non-NULL values of ``c`` in a group are
all different, and ``DISTINCT``, which only merges equal non-NULL values (an aggregate skips NULL arguments
with or without it), merges nothing. That holds for an empty group of an ungrouped select too (both forms
aggregate no values). The NOT NULL facts are needed: a ``UNIQUE`` key lets rows repeat when a key column is
NULL, and a ``GROUP BY`` puts the NULLs of a grouping column in one group, where ``c`` could then repeat.
For the tuple count every counted tuple is free of NULLs by declaration, so every row of the group is
counted, and by the same argument no two rows of the group hold the same tuple.

**What is read.** A select with exactly one source, a plain base table (no join, no derived table, no CTE of
that name, no ``LATERAL``, no sample); a plain ``GROUP BY`` (no ``ROLLUP``, ``CUBE`` or grouping sets, whose
super-groups mix the groups) whose every item is a column of that table, or no ``GROUP BY``; an aggregate of
the select itself (``SELECT`` list, ``HAVING`` or ``ORDER BY``; not one inside a nested select, not a window's,
none with ``FILTER``) whose ``DISTINCT`` argument is a column of the table. A column counts as the table's when
it is qualified with the table's name in this select, or bare and listed for the table in the supplied schema
(the innermost select binds a name first). ``MIN`` and ``MAX`` ignore ``DISTINCT`` anyway and are left alone.

**What is refused.** Everything else: joins (a join repeats the rows of a keyed table, and whether its other
side is keyed is another rule's concern), derived tables, grouping by an expression, an argument that is an
expression, a key with a nullable column other than the argument, and tables without a declared key.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import extended_grouping, visible_ctes

_TABLE_ARGS = frozenset({"this", "db", "catalog", "alias"})
_KINDS = (exp.Count, exp.Sum, exp.Avg)


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _single_base_table(select: exp.Select) -> exp.Table | None:
    source = select.args.get("from_") or select.args.get("from")
    if source is None or select.args.get("joins") or select.args.get("laterals"):
        return None
    table = source.this
    if type(table) is not exp.Table or not isinstance(table.this, exp.Identifier):
        return None
    if any(value for key, value in table.args.items() if key not in _TABLE_ARGS):
        return None
    alias = table.args.get("alias")
    if alias is not None and alias.args.get("columns"):
        return None
    if not table.args.get("db") and table.name.lower() in visible_ctes(table):
        return None
    return table


def _own_column(node: exp.Expression, table: exp.Table, listed: set[str]) -> str | None:
    """The lower-cased name of ``node`` when it is a plain column of ``table`` as its select names it."""

    node = _unwrap(node)
    if type(node) is not exp.Column or isinstance(node.this, exp.Star) or node.args.get("db") or node.args.get("catalog"):
        return None
    name = node.name.lower()
    if name not in listed:
        return None
    if node.table:
        return name if node.table.lower() == table.alias_or_name.lower() else None
    return name


def drop_keyed_aggregate_distinct(
    select: exp.Select,
    schema: dict[str, list[str]] | None,
    keys: dict[str, list[tuple[str, ...]]] | None,
    not_null: dict[str, frozenset[str]] | None,
) -> exp.Expression | None:
    """``select`` (changed in place) with every ``DISTINCT`` a key makes redundant removed from its aggregates, else ``None``."""

    if not keys or not schema:
        return None
    table = _single_base_table(select)
    if table is None:
        return None
    name = table.name.lower()
    declared = [tuple(c.lower() for c in key) for key in (keys.get(name) or []) if key]
    listed = {c.lower() for c in ({k.lower(): v for k, v in schema.items()}.get(name) or [])}
    if not declared or not listed:
        return None
    nonnull = {c.lower() for c in ({k.lower(): v for k, v in (not_null or {}).items()}.get(name) or ())}
    group = select.args.get("group")
    grouped: set[str] = set()
    if group is not None:
        if extended_grouping(group) or any(value for key, value in group.args.items() if key != "expressions"):
            return None
        for item in group.expressions:
            column = _own_column(item, table, listed)
            if column is None:
                return None
            grouped.add(column)

    def keyed(arguments: set[str]) -> bool:
        return any(set(key) <= grouped | arguments and all(c in nonnull or c in arguments for c in key) for key in declared)

    changed = False
    for node in list(select.find_all(*_KINDS)):
        if node.find_ancestor(exp.Select) is not select or node.find_ancestor(exp.Window) is not None or isinstance(node.parent, exp.Filter):
            continue
        if any(value for key, value in node.args.items() if key not in ("this", "big_int")):
            continue
        distinct = node.this
        if not isinstance(distinct, exp.Distinct) or distinct.args.get("on") or not distinct.expressions:
            continue
        columns = [_own_column(e, table, listed) for e in distinct.expressions]
        if None in columns:
            continue
        if len(columns) == 1:
            if keyed({columns[0]}):
                node.set("this", _unwrap(distinct.expressions[0]).copy())
                changed = True
        elif isinstance(node, exp.Count) and all(c in nonnull for c in columns) and keyed(set(columns)):
            node.set("this", exp.Star())
            changed = True
    return select if changed else None
