"""``HAVING COUNT(..) >= 1`` says nothing about a group whose every row is counted.

**The rule.** In a select with a plain ``GROUP BY`` of at least one key, a ``HAVING`` conjunct ``COUNT(*) >= 1``
is dropped, and so is ``COUNT(t.c) >= 1`` or ``COUNT(DISTINCT t.c) >= 1`` when ``t`` is a base table the select
reads through inner joins only and ``c`` is declared NOT NULL. ``> 0`` and ``<> 0`` are the same test.

    SELECT a.k, COUNT(DISTINCT b.v) FROM t AS a JOIN u AS b ON a.k = b.g GROUP BY a.k HAVING COUNT(b.v) > 0
    SELECT a.k, COUNT(DISTINCT b.v) FROM t AS a JOIN u AS b ON a.k = b.g GROUP BY a.k         -- u.v NOT NULL

**Why it is sound.** A ``GROUP BY`` with a key forms a group only from rows that exist, so every group has at
least one row and ``COUNT(*)`` is at least 1. With inner joins only, every row of the group holds a row of
``t``, in which the NOT NULL column ``c`` is not NULL; so ``COUNT(t.c)`` counts every row of the group and
``COUNT(DISTINCT t.c)`` counts at least one value. A count is never NULL, so the conjunct is TRUE for every
group and the conjunction keeps the same groups without it. (Without a ``GROUP BY`` key the one group of a
global aggregate may be empty, and an outer join may pad ``t.c`` with NULL: neither is read.)

**What is read.** The conjunct must be a top-level conjunct of ``HAVING`` comparing an unmodified ``COUNT`` with
a literal: ``>= 1``, ``> 0``, ``<> 0``, or the mirrored spellings. The count may be named by its select alias when
the schema lists the columns of every table the select reads and none has that name (as
``having_count_rules`` reads an alias). The counted column must be qualified with the name of one of the
select's sources, or the select must have that table as its only source and the schema must list the column.

**What is refused.** ``ROLLUP``, ``CUBE``, grouping sets and ``GROUP BY ()`` (a grand-total group exists over no
rows), outer joins, ``USING``/``NATURAL`` joins, a counted expression, a counted column of a derived table or of a
table without the NOT NULL declaration, counts with ``FILTER`` or ``OVER``, a higher bound (``>= 2``), and tests
under ``OR`` or ``NOT``.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import conjuncts, extended_grouping
from .decorrelation_rules import _and_all, _sources
from .having_count_rules import _count_aliases, _natural
from .semijoin_shapes import _inner_conjuncts, _plain_column, _plain_table

_FLIP = {exp.GTE: exp.LTE, exp.GT: exp.LT, exp.LTE: exp.GTE, exp.LT: exp.GT, exp.NEQ: exp.NEQ}


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _positive_count(test: exp.Expression, aliases: dict[str, exp.Expression]) -> exp.Count | None:
    """The ``COUNT`` of ``COUNT(..) >= 1``, ``> 0`` or ``<> 0`` (either order), else ``None``."""

    test = _unwrap(test)
    kind = type(test)
    if kind not in _FLIP:
        return None

    def counted(node: exp.Expression) -> exp.Count | None:
        node = _unwrap(node)
        if type(node) is exp.Column and not node.table and node.name.lower() in aliases:
            node = _unwrap(aliases[node.name.lower()])
        return node if type(node) is exp.Count else None

    count, number = counted(test.this), _natural(test.expression)
    if count is None or number is None:
        count, number, kind = counted(test.expression), _natural(test.this), _FLIP[kind]
    if count is None or number is None:
        return None
    return count if (kind, number) in ((exp.GTE, 1), (exp.GT, 0), (exp.NEQ, 0)) else None


def drop_nonempty_group_counts(select: exp.Select, schema: dict | None, not_null: dict | None) -> exp.Expression | None:
    """``select`` (changed in place) without the ``HAVING`` tests that hold for every group, else ``None``."""

    group, having = select.args.get("group"), select.args.get("having")
    if group is None or having is None or not group.expressions or extended_grouping(group):
        return None
    if any(value for key, value in group.args.items() if key != "expressions") or any(select.args.get(k) for k in ("qualify", "windows", "laterals")):
        return None
    sources = _sources(select)
    names = [(s.alias_or_name or "").lower() for s in sources]
    inner = _inner_conjuncts(select) is not None and "" not in names and len(set(names)) == len(names)
    by_name = dict(zip(names, sources))
    listed = {k.lower(): {c.lower() for c in v} for k, v in (schema or {}).items()}
    declared = {k.lower(): {c.lower() for c in v} for k, v in (not_null or {}).items()}
    aliases = _count_aliases(select, schema)
    # a select alias of the same name could be what a bare name in HAVING means
    shadowed = {item.alias.lower() for item in select.expressions if isinstance(item, exp.Alias) and item.alias}

    def always_counted(count: exp.Count) -> bool:
        if isinstance(count.parent, (exp.Filter, exp.Window)) or any(value for key, value in count.args.items() if key not in ("this", "big_int")):
            return False
        argument = count.this
        if isinstance(argument, exp.Star):
            return True
        if isinstance(argument, exp.Distinct):
            if argument.args.get("on") or len(argument.expressions) != 1:
                return False
            argument = argument.expressions[0]
        column = _plain_column(argument) if isinstance(argument, exp.Expression) else None
        if column is None or not inner:
            return False
        if column.table:
            source = by_name.get(column.table.lower())
        else:
            source = sources[0] if len(sources) == 1 and column.name.lower() not in shadowed else None
        if source is None or not _plain_table(source):
            return False
        table = source.name.lower()
        return column.name.lower() in listed.get(table, ()) and column.name.lower() in declared.get(table, ())

    kept, dropped = [], False
    for test in conjuncts(having.this):
        count = _positive_count(test, aliases)
        if count is not None and (count.find_ancestor(exp.Select) is select) and always_counted(count):
            dropped = True
        else:
            kept.append(test)
    if not dropped:
        return None
    remaining = _and_all([t.copy() for t in kept])
    select.set("having", exp.Having(this=remaining) if remaining is not None else None)
    return select
