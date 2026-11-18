"""An aggregate over the row's whole partition read as a join with the table's own grouped aggregate.

    SELECT id, pay, MAX(pay) OVER (PARTITION BY team) AS top FROM staff

puts each team's highest pay beside every row, as does

    SELECT id, pay, g.top FROM staff JOIN (SELECT team, MAX(pay) AS top FROM staff GROUP BY team) AS g ON staff.team = g.team

The second spelling is the one the prover reasons about: it knows the grouped table of a base table as "one row
per group, holding the group's aggregate", including what that says about a row of the group (its value is not
above the group's ``MAX``; a NULL ``MAX`` means the row's value is NULL). A window is kept as an opaque relation,
about which nothing of the kind is known, so the two spellings of one query stayed apart. This pre-pass writes
the window spelling as the join.

**The rule.** In a select whose only source is a base table ``t``, with no ``WHERE``, grouping or aggregate of
its own, every window ``AGG(c) OVER (PARTITION BY k1, .., kn)`` in the select list is replaced by a column of

    JOIN (SELECT k1 AS .., kn AS .., AGG(c) AS .. FROM t GROUP BY k1, .., kn) AS g ON t.k1 = g.k1 AND .. AND t.kn = g.kn

**Why it is sound.**

* The window has no ``ORDER BY`` and no frame, so it aggregates every row of the row's partition; the select has
  no ``WHERE``, so the partitions are made of all the rows of ``t``. The grouped table aggregates the same rows:
  the group with the row's key values.
* Every partition column is declared NOT NULL, so each row of ``t`` equals exactly one group on all the keys:
  its own (which exists, because the row itself is in it). The join therefore keeps every row exactly once and
  reads that group's aggregate. Without NOT NULL a NULL key would form a partition for the window but match no
  group in the join, so the rule is not applied.
* ``MIN``, ``MAX``, ``SUM``, ``AVG`` and ``COUNT`` over a set of rows give the same value (and type) whether they
  are written as a window or as a grouped aggregate; ``DISTINCT`` arguments are not read.
* Nothing else in the select changes meaning: the grouped table's output names are fresh (no bare name becomes
  ambiguous, and a bare ``*`` is written ``t.*``), and ``DISTINCT``, ``ORDER BY`` and ``LIMIT`` apply to the same
  rows as before.

**What is read, and what is refused.**

* The select: one plain base table with supplied columns (no ``WITH`` table of that name), no join, ``WHERE``,
  ``GROUP BY``, ``HAVING``, ``QUALIFY``, named-window clause or aggregate outside a window; windows only in its
  select list. A ``*`` with ``EXCEPT`` or ``REPLACE`` is refused.
* Every window of the select must fit, or nothing is rewritten: a plain ``MIN``, ``MAX``, ``SUM``, ``AVG`` or
  ``COUNT`` of one column of ``t`` (or ``COUNT(*)``), a non-empty ``PARTITION BY`` of plain NOT NULL columns of
  ``t``, the same set of columns in every window, no ``ORDER BY``, frame, ``FILTER`` or ``IGNORE NULLS``.
* A column is ``t``'s when it is bare and ``t`` has it, or carries ``t``'s name.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _names, _table_key
from .ast_utils import FROM_KEY, select_sources, visible_ctes

_PREFIX = "kqg"
_AGGREGATES = (exp.Min, exp.Max, exp.Sum, exp.Avg, exp.Count)
_REFUSED = ("joins", "where", "group", "having", "qualify", "windows", "laterals", "with", "with_", "connect", "pivots")


def read_own_group_windows(tree: exp.Expression, schema: dict[str, list[str]] | None, not_null: dict[str, frozenset[str]] | None) -> exp.Expression:
    """Rewrite, in place, each select described in the module docstring."""

    if not schema or not not_null or not any(True for _ in tree.find_all(exp.Window)):
        return tree
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    declared = {key.lower(): {c.lower() for c in columns} for key, columns in not_null.items()}
    taken = _names(tree)
    for select in list(tree.find_all(exp.Select)):
        _rewrite(select, known, declared, taken)
    return tree


def _fresh(taken: set[str]) -> str:
    number = 0
    while f"{_PREFIX}{number}" in taken:
        number += 1
    taken.add(f"{_PREFIX}{number}")
    return f"{_PREFIX}{number}"


def _own(node: exp.Expression, select: exp.Select) -> bool:
    return node.find_ancestor(exp.Select) is select


def _column_of(node: exp.Expression, name: str, columns: list[str]) -> str | None:
    """The column name when ``node`` is a plain column of the table (bare, or carrying the table's name)."""

    if type(node) is not exp.Column or isinstance(node.this, exp.Star) or node.args.get("db") or node.args.get("catalog"):
        return None
    if node.table and node.table.lower() != name:
        return None
    return node.name.lower() if node.name.lower() in columns else None


def _window(node: exp.Window, name: str, columns: list[str], facts: set[str]) -> tuple[tuple[str, ...], type, str | None] | None:
    """``(partition columns, aggregate, argument column or None for COUNT(*))`` for a window the rule reads."""

    if any(value for key, value in node.args.items() if key not in ("this", "partition_by", "over", "comments")):
        return None  # ORDER BY, a frame, a named window
    if (node.text("over") or "OVER").upper() != "OVER":
        return None
    aggregate = node.this
    if type(aggregate) not in _AGGREGATES:
        return None
    if any(value for key, value in aggregate.args.items() if key not in ("this", "big_int", "comments")):
        return None
    if type(aggregate) is exp.Count and isinstance(aggregate.this, exp.Star):
        argument = None
    else:
        argument = _column_of(aggregate.this, name, columns)
        if argument is None:
            return None
    keys = [_column_of(key, name, columns) for key in node.args.get("partition_by") or []]
    if not keys or None in keys or len(set(keys)) != len(keys) or not set(keys) <= facts:
        return None
    return tuple(sorted(keys)), type(aggregate), argument


def _rewrite(select: exp.Select, known: dict[str, list[str]], declared: dict[str, set[str]], taken: set[str]) -> bool:
    if any(select.args.get(key) for key in _REFUSED):
        return False
    distinct = select.args.get("distinct")
    if distinct is not None and (not isinstance(distinct, exp.Distinct) or distinct.args.get("on")):
        return False
    sources = select_sources(select)
    if len(sources) != 1:
        return False
    table = sources[0]
    columns = _base_table(table, known)
    if columns is None or not table.alias_or_name:
        return False
    if not table.args.get("db") and table.name.lower() in visible_ctes(table):
        return False
    name = table.alias_or_name.lower()
    facts = declared.get(_table_key(table), set())
    windows = [node for node in select.find_all(exp.Window) if _own(node, select)]
    if not windows:
        return False
    in_list = lambda node: any(node is item or _inside(node, item) for item in select.expressions)  # noqa: E731
    if not all(in_list(node) for node in windows):
        return False
    if any(isinstance(node, exp.AggFunc) and _own(node, select) and not isinstance(node.parent, exp.Window) for node in select.find_all(exp.AggFunc)):
        return False
    read = [_window(node, name, columns, facts) for node in windows]
    if None in read or len({keys for keys, _, _ in read}) != 1:
        return False
    for item in select.expressions:
        for star in item.find_all(exp.Star):
            if isinstance(star.parent, exp.Count):
                continue
            if any(value for value in star.args.values()) or not (star is item or (isinstance(star.parent, exp.Column) and star.parent is item)):
                return False
    # --- rewrite ---
    keys = read[0][0]
    alias = _fresh(taken)
    outputs = [exp.alias_(exp.column(key), f"{alias}k{index}") for index, key in enumerate(keys)]
    named: dict[tuple[type, str | None], str] = {}
    for node, (_, aggregate, argument) in zip(windows, read):
        if (aggregate, argument) not in named:
            named[(aggregate, argument)] = f"{alias}v{len(named)}"
            value = aggregate(this=exp.Star() if argument is None else exp.column(argument))
            outputs.append(exp.alias_(value, named[(aggregate, argument)]))
        node.replace(exp.column(named[(aggregate, argument)], table=alias))
    inner_table = table.copy()
    inner_table.set("alias", None)
    grouped = exp.Select(expressions=outputs)
    grouped.set(FROM_KEY, exp.From(this=inner_table))
    grouped.set("group", exp.Group(expressions=[exp.column(key) for key in keys]))
    condition = None
    for index, key in enumerate(keys):
        test = exp.EQ(this=exp.column(key, table=table.alias_or_name), expression=exp.column(f"{alias}k{index}", table=alias))
        condition = test if condition is None else exp.And(this=condition, expression=test)
    source = exp.Subquery(this=grouped, alias=exp.TableAlias(this=exp.to_identifier(alias)))
    select.set("joins", [exp.Join(this=source, on=condition)])
    for item in list(select.expressions):
        if isinstance(item, exp.Star):
            item.replace(exp.Column(this=exp.Star(), table=exp.to_identifier(table.alias_or_name)))
    return True


def _inside(node: exp.Expression, ancestor: exp.Expression) -> bool:
    parent = node.parent
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.parent
    return False
