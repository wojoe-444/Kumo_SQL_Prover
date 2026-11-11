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

**What is read.** A select with exactly one source: a plain base table (no join, no CTE of that name, no
``LATERAL``, no sample), or a derived table that only filters one such table and lists columns of it
(``(SELECT t.a, t.b AS v FROM t WHERE p) AS d``: no ``DISTINCT``, grouping, join, ``LIMIT`` or expression, so
each of its rows is one row of ``t`` and ``d.v`` is ``t.b``; the key columns the rule needs must be among those
listed, since they are read through the select's grouping and argument); a plain ``GROUP BY`` (no ``ROLLUP``,
``CUBE`` or grouping sets, whose super-groups mix the groups) whose every item is a column of that table, or no
``GROUP BY``; an aggregate of
the select itself (``SELECT`` list, ``HAVING`` or ``ORDER BY``; not one inside a nested select, not a window's,
none with ``FILTER``) whose ``DISTINCT`` argument is a column of the table. A column counts as the table's when
it is qualified with the table's name in this select, or bare and listed for the table in the supplied schema
(the innermost select binds a name first); a bare name that a select alias gives to another expression is not
read in ``GROUP BY``, ``HAVING`` or ``ORDER BY``, which may mean the alias (the select list never reads its own
aliases). ``MIN`` and ``MAX`` ignore ``DISTINCT`` anyway and are left alone.

**What is refused.** Everything else: joins (a join repeats the rows of a keyed table, and whether its other
side is keyed is another rule's concern), any other derived table, grouping by an expression, an argument that is an
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


def _base_table(table: exp.Expression) -> bool:
    if type(table) is not exp.Table or not isinstance(table.this, exp.Identifier):
        return False
    if any(value for key, value in table.args.items() if key not in _TABLE_ARGS):
        return False
    alias = table.args.get("alias")
    if alias is not None and alias.args.get("columns"):
        return False
    return bool(table.args.get("db")) or table.name.lower() not in visible_ctes(table)


def _table_column(node: exp.Expression, name: str, listed: set[str]) -> str | None:
    """The lower-cased name of ``node`` when it is a plain column of the one source ``name`` that lists it."""

    node = _unwrap(node)
    if type(node) is not exp.Column or isinstance(node.this, exp.Star) or node.args.get("db") or node.args.get("catalog"):
        return None
    column = node.name.lower()
    if column not in listed or (node.table and node.table.lower() != name):
        return None
    return column


def _keyed_rows(select: exp.Select, schema: dict[str, list[str]]):
    """``(table name, reader)`` when the rows ``select`` reads are rows of one base table: ``reader`` maps a
    column of the select to the table column it shows, else to ``None``."""

    source = select.args.get("from_") or select.args.get("from")
    if source is None or select.args.get("joins") or select.args.get("laterals"):
        return None
    known = {k.lower(): {c.lower() for c in v} for k, v in schema.items()}
    table = source.this
    if _base_table(table):
        listed = known.get(table.name.lower()) or set()
        return table.name.lower(), lambda node: _table_column(node, table.alias_or_name.lower(), listed)
    # a derived table that only filters one base table and shows some of its columns, each row once
    if type(table) is not exp.Subquery or not table.alias or type(table.this) is not exp.Select:
        return None
    if any(value for key, value in table.args.items() if key not in ("this", "alias")) or table.args["alias"].args.get("columns"):
        return None
    inner = table.this
    if any(value for key, value in inner.args.items() if key not in ("expressions", "from_", "from", "where")):
        return None
    inner_source = inner.args.get("from_") or inner.args.get("from")
    if inner_source is None or not _base_table(inner_source.this):
        return None
    base = inner_source.this
    listed = known.get(base.name.lower()) or set()
    shown: dict[str, str] = {}
    for item in inner.expressions:
        column = _table_column(item.unalias(), base.alias_or_name.lower(), listed)
        name = item.alias_or_name.lower()
        if column is None or not name or name in shown:
            return None
        shown[name] = column

    def reader(node: exp.Expression) -> str | None:
        name = _table_column(node, table.alias.lower(), set(shown))
        return shown[name] if name is not None else None

    return base.name.lower(), reader


def drop_keyed_aggregate_distinct(
    select: exp.Select,
    schema: dict[str, list[str]] | None,
    keys: dict[str, list[tuple[str, ...]]] | None,
    not_null: dict[str, frozenset[str]] | None,
) -> exp.Expression | None:
    """``select`` (changed in place) with every ``DISTINCT`` a key makes redundant removed from its aggregates, else ``None``."""

    if not keys or not schema:
        return None
    found = _keyed_rows(select, schema)
    if found is None:
        return None
    name, read = found
    # a bare name that a select alias gives to something else may mean the alias in GROUP BY, HAVING or ORDER BY
    renamed = {
        item.alias.lower() for item in select.expressions
        if isinstance(item, exp.Alias) and item.alias and not (type(_unwrap(item.this)) is exp.Column and _unwrap(item.this).name.lower() == item.alias.lower())
    }

    def reader(node: exp.Expression) -> str | None:
        node = _unwrap(node)
        if type(node) is exp.Column and not node.table and node.name.lower() in renamed:
            top = node
            while top.parent is not None and top.parent is not select:
                top = top.parent
            if top.arg_key != "expressions":
                return None  # the select list itself never reads its own aliases
        return read(node)

    declared = [tuple(c.lower() for c in key) for key in ({k.lower(): v for k, v in keys.items()}.get(name) or []) if key]
    if not declared:
        return None
    nonnull = {c.lower() for c in ({k.lower(): v for k, v in (not_null or {}).items()}.get(name) or ())}
    group = select.args.get("group")
    grouped: set[str] = set()
    if group is not None:
        if extended_grouping(group) or any(value for key, value in group.args.items() if key != "expressions"):
            return None
        for item in group.expressions:
            column = reader(item)
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
        columns = [reader(e) for e in distinct.expressions]
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
