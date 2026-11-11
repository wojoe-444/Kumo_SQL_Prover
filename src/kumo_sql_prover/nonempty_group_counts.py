"""``COUNT(..) >= 1`` says nothing about a group whose every row is counted.

**The rule.** In a select with a plain ``GROUP BY`` of at least one key, ``COUNT(*)`` is at least 1 in every
group, and so is ``COUNT(t.c)`` or ``COUNT(DISTINCT t.c)`` when ``t`` is a base table the select reads through
inner joins only and ``c`` is declared NOT NULL. A test that such a count is ``>= 1`` (``> 0``, ``<> 0``) is
dropped in the two places it is written:

    -- 1. in HAVING                                                               (u.v NOT NULL)
    SELECT a.k, COUNT(DISTINCT b.v) FROM t AS a JOIN u AS b ON a.k = b.g GROUP BY a.k HAVING COUNT(b.v) > 0
    SELECT a.k, COUNT(DISTINCT b.v) FROM t AS a JOIN u AS b ON a.k = b.g GROUP BY a.k

    -- 2. in the WHERE of a select that reads the grouped query as a derived table
    SELECT d.k FROM (SELECT k, COUNT(*) AS n FROM t GROUP BY k) AS d JOIN u ON u.g = d.k WHERE d.n >= 1
    SELECT d.k FROM (SELECT k, COUNT(*) AS n FROM t GROUP BY k) AS d JOIN u ON u.g = d.k

**Why it is sound.** A ``GROUP BY`` with a key forms a group only from rows that exist, so every group has at
least one row and ``COUNT(*)`` is at least 1. With inner joins only, every row of the group holds a row of
``t``, in which the NOT NULL column ``c`` is not NULL; so ``COUNT(t.c)`` counts every row of the group and
``COUNT(DISTINCT t.c)`` counts at least one value. A count is never NULL, so the test is TRUE for every
group and the conjunction keeps the same groups (form 1) or rows (form 2) without it. In form 2 every row of
the derived table is one of its groups, whatever its own ``HAVING`` keeps, and the outer select must read the
derived table through inner joins only: the padded row of an outer join would hold NULL for the count, and
the test would drop it. (Without a ``GROUP BY`` key the one group of a global aggregate may be empty, and an
outer join inside the grouped select may pad ``t.c`` with NULL: neither is read.)

**What is read.** The test must be a top-level conjunct (of ``HAVING``, or of the outer ``WHERE``) comparing the
count with a literal: ``>= 1``, ``> 0``, ``<> 0``, or the mirrored spellings. In ``HAVING`` the count is an
unmodified ``COUNT``, or its select alias when the schema lists the columns of every table the select reads and
none has that name (as ``having_count_rules`` reads an alias). In form 2 it is a column of the derived table,
qualified with the derived table's name (or bare when the derived table is the only source), whose output is
such a ``COUNT``. The counted column must be qualified with the name of one of the grouped select's sources, or
that select must have the table as its only source and the schema must list the column.

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
from .semijoin_shapes import _derived, _inner_conjuncts, _plain_column, _plain_table

_FLIP = {exp.GTE: exp.LTE, exp.GT: exp.LT, exp.LTE: exp.GTE, exp.LT: exp.GT, exp.NEQ: exp.NEQ}


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _positive_test(test: exp.Expression) -> exp.Expression | None:
    """The tested side of ``.. >= 1``, ``.. > 0`` or ``.. <> 0`` (either order), else ``None``."""

    test = _unwrap(test)
    kind = type(test)
    if kind not in _FLIP:
        return None
    tested, number = test.this, _natural(test.expression)
    if number is None:
        tested, number, kind = test.expression, _natural(test.this), _FLIP[kind]
    if number is None or (kind, number) not in ((exp.GTE, 1), (exp.GT, 0), (exp.NEQ, 0)):
        return None
    return _unwrap(tested)


def _keyed_groups(select: exp.Select) -> bool:
    """Whether ``select`` has a plain ``GROUP BY`` with a key: every group it forms has a row."""

    group = select.args.get("group")
    if group is None or not group.expressions or extended_grouping(group):
        return False
    if any(value for key, value in group.args.items() if key != "expressions"):
        return False
    return not any(select.args.get(k) for k in ("qualify", "windows", "laterals"))


def _always_counted(count: exp.Expression, select: exp.Select, schema: dict | None, not_null: dict | None) -> bool:
    """Whether ``count``, an aggregate of the grouped ``select``, counts at least one row of every group."""

    if type(count) is not exp.Count or count.find_ancestor(exp.Select) is not select:
        return False
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
    sources = _sources(select)
    names = [(s.alias_or_name or "").lower() for s in sources]
    if column is None or _inner_conjuncts(select) is None or "" in names or len(set(names)) != len(names):
        return False
    # a select alias of the same name could be what a bare name means
    shadowed = {item.alias.lower() for item in select.expressions if isinstance(item, exp.Alias) and item.alias}
    if column.table:
        source = dict(zip(names, sources)).get(column.table.lower())
    else:
        source = sources[0] if len(sources) == 1 and column.name.lower() not in shadowed else None
    if source is None or not _plain_table(source):
        return False
    table = source.name.lower()
    listed = {k.lower(): {c.lower() for c in v} for k, v in (schema or {}).items()}
    declared = {k.lower(): {c.lower() for c in v} for k, v in (not_null or {}).items()}
    return column.name.lower() in listed.get(table, ()) and column.name.lower() in declared.get(table, ())


def _drop_in_having(select: exp.Select, schema: dict | None, not_null: dict | None) -> exp.Select | None:
    having = select.args.get("having")
    if having is None or not _keyed_groups(select):
        return None
    aliases = _count_aliases(select, schema)
    kept, dropped = [], False
    for test in conjuncts(having.this):
        tested = _positive_test(test)
        if tested is not None and type(tested) is exp.Column and not tested.table and tested.name.lower() in aliases:
            tested = _unwrap(aliases[tested.name.lower()])
        if tested is not None and _always_counted(tested, select, schema, not_null):
            dropped = True
        else:
            kept.append(test)
    if not dropped:
        return None
    remaining = _and_all([t.copy() for t in kept])
    select.set("having", exp.Having(this=remaining) if remaining is not None else None)
    return select


def _drop_above_derived(select: exp.Select, schema: dict | None, not_null: dict | None) -> exp.Select | None:
    where = select.args.get("where")
    if where is None or select.args.get("laterals") or _inner_conjuncts(select) is None:
        return None
    sources = _sources(select)
    names = [(s.alias_or_name or "").lower() for s in sources]
    if "" in names or len(set(names)) != len(names):
        return None
    by_name = dict(zip(names, sources))
    kept, dropped = [], False
    for test in conjuncts(where.this):
        tested = _positive_test(test)
        column = _plain_column(tested) if tested is not None else None
        source = None
        if column is not None:
            source = by_name.get(column.table.lower()) if column.table else (sources[0] if len(sources) == 1 else None)
        inner = _derived(source) if source is not None else None
        counted = False
        if inner is not None and _keyed_groups(inner):
            outputs = [item for item in inner.expressions if item.alias_or_name.lower() == column.name.lower()]
            stars = any(isinstance(item.unalias(), exp.Star) or isinstance(item.unalias(), exp.Column) and isinstance(item.unalias().this, exp.Star) for item in inner.expressions)
            counted = len(outputs) == 1 and not stars and _always_counted(_unwrap(outputs[0].unalias()), inner, schema, not_null)
        if counted:
            dropped = True
        else:
            kept.append(test)
    if not dropped:
        return None
    remaining = _and_all([t.copy() for t in kept])
    select.set("where", exp.Where(this=remaining) if remaining is not None else None)
    return select


def drop_nonempty_group_counts(select: exp.Select, schema: dict | None, not_null: dict | None) -> exp.Expression | None:
    """``select`` (changed in place) without the count tests that hold for every group, else ``None``."""

    return _drop_in_having(select, schema, not_null) or _drop_above_derived(select, schema, not_null)
