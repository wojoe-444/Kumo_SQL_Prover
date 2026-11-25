"""An aggregate over the row's whole partition, in a table joined to one keyed row, read as a join with the table's
own grouped aggregate.

    SELECT o.id, o.qty, AVG(o.qty) OVER (PARTITION BY o.item) AS usual
    FROM orders AS o JOIN items AS i ON o.item = i.id WHERE i.kind = 'tool'

puts the mean quantity of the item beside every order of a tool, as does

    SELECT o.id, o.qty, g.usual
    FROM orders AS o JOIN items AS i ON o.item = i.id
    JOIN (SELECT item, AVG(qty) AS usual FROM orders GROUP BY item) AS g ON o.item = g.item WHERE i.kind = 'tool'

``own_group_windows`` reads the window of a select whose only source is the table and which has no ``WHERE``. This
pre-pass (hardest-cases batch 28) reads the select that also joins the table's rows to their one row of a keyed
table and filters on that row: the usual way to write "compare each fact with its group's aggregate, for the groups
of some kind".

**The rule.** The select has two sources joined by an inner join (or a comma): a declared table ``t`` and a keyed
source ``p``, which is a declared table or a derived table that lists columns of one declared table's rows
(``SELECT id FROM items WHERE kind = 'tool'``). Every condition of the join and of ``WHERE`` is either

* ``t.c = p.k``, a column of ``t`` equal to a column of ``p``, or
* a test that reads columns of ``p`` only.

The columns ``k`` cover a declared NOT NULL key of ``p``'s table. Every window is ``AGG(c) OVER (PARTITION BY ..)``
with the partition made of columns of ``t`` and of columns ``p.k`` (read as the ``t.c`` they are equal to), and
holding every ``t.c`` of the conditions. Each window is replaced by a column of

    JOIN (SELECT c1 AS .., AGG(c) AS .. FROM t GROUP BY c1, ..) AS g ON t.c1 = g.c1 AND ..

**Why it is sound.**

* A row of ``t`` is in the select's rows at most once: the row of ``p`` it joins holds the key values ``t.c``,
  and a key holds one row. Whether it is there depends on its ``t.c`` values alone (they pick the row of ``p``,
  and every other test reads that row only).
* So the rows of ``t`` that agree on the partition columns, which hold every ``t.c``, are all in the select's rows
  or all out. The partition of a row that is in is therefore every row of ``t`` with its partition values, each
  once: the group the grouped table aggregates.
* The join with the grouped table keeps each row exactly once. A partition column that is a ``t.c`` is not NULL on
  a row that is in (it is equal to something); every other partition column must be declared NOT NULL, since a
  NULL there forms a partition for the window and matches no group in the join.
* "Agree on the partition columns" must mean the same to ``GROUP BY`` and to the comparison ``t.c = p.k``. It does
  for whole numbers. Texts can be compared under another collation than the one they are grouped by (a ``NOCASE``
  column joined to a binary one), which the supplied types do not say, so the rule reads a pair ``t.c = p.k``
  only when both columns are declared with a whole-number type.
* ``MIN``, ``MAX``, ``SUM``, ``AVG`` and ``COUNT`` over a set of rows give the same value (and type) whether they
  are written as a window or as a grouped aggregate.

**What is refused.** Anything else: a third source, an outer join, ``USING`` or ``NATURAL``; ``GROUP BY``,
``HAVING``, ``QUALIFY``, a named-window clause, an aggregate outside a window, a window outside the select list, a
subquery anywhere in the select other than the keyed derived table; a window with ``ORDER BY``, a frame,
``FILTER``, ``DISTINCT`` or an argument that is not one column of ``t`` (or ``COUNT(*)``); windows over different
partitions; a ``*`` in the select list; a column whose source cannot be told (a bare name both sources have, a
name of neither); a condition that reads ``t`` and is not one of the equalities; a pair whose columns are not both
whole numbers; a key of ``p`` that is not declared NOT NULL.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _names
from .ast_utils import FROM_KEY
from .single_row_aggregates import Keyed, Listed, column_kind

NAME = "keyed_join_windows"

_PREFIX = "kqj"
_AGGREGATES = (exp.Min, exp.Max, exp.Sum, exp.Avg, exp.Count)
_REFUSED = ("group", "having", "qualify", "windows", "laterals", "with", "with_", "connect", "pivots", "match", "sample", "into", "kind", "hint")


def read_keyed_join_windows(tree: exp.Expression, schema: dict | None, keys: dict | None, not_null: dict | None, types: dict | None) -> tuple[exp.Expression, bool]:
    """``(tree with each select of the module docstring rewritten, whether one was)``. The input is not changed."""

    if not schema or not keys or not types or not any(True for _ in tree.find_all(exp.Window)):
        return tree, False
    copy = tree.copy()
    taken = _names(copy)
    wrote = False
    for select in list(copy.find_all(exp.Select)):
        wrote = _rewrite(select, schema, keys, not_null, types, taken) or wrote
    return (copy, True) if wrote else (tree, False)


def _fresh(taken: set[str]) -> str:
    number = 0
    while f"{_PREFIX}{number}" in taken:
        number += 1
    taken.add(f"{_PREFIX}{number}")
    return f"{_PREFIX}{number}"


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _conjuncts(node: exp.Expression | None) -> list[exp.Expression]:
    if node is None:
        return []
    node = _unwrap(node)
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _inside(node: exp.Expression, ancestor: exp.Expression) -> bool:
    while node is not None:
        if node is ancestor:
            return True
        node = node.parent
    return False


def _set_args(node: exp.Expression) -> set[str]:
    return {key for key, value in node.args.items() if value not in (None, [], False)}


class _Sides:
    """The two sources of the select: the table whose rows the windows aggregate and the keyed source."""

    def __init__(self, table: Keyed, keyed: Keyed) -> None:
        self.table, self.keyed = table, keyed
        self.shown = set(keyed.shown) if isinstance(keyed, Listed) else set(keyed.columns)

    def side(self, node: exp.Expression) -> str | None:
        """``"t"`` or ``"p"`` for a plain column of one of the two sources, else None."""

        if type(node) is not exp.Column or not isinstance(node.this, exp.Identifier) or node.args.get("db") or node.args.get("catalog"):
            return None
        name, written = node.name.lower(), (node.table or "").lower()
        if written:
            if written == self.table.alias:
                return "t" if name in self.table.columns else None
            return "p" if written == self.keyed.alias and name in self.shown else None
        in_table, in_keyed = name in self.table.columns, name in self.shown
        if in_table == in_keyed:
            return None  # a name of both sources, or of neither
        return "t" if in_table else "p"

    def pair(self, test: exp.Expression) -> tuple[str, str] | None:
        """``(column of t, column of p's table)`` for a test ``t.c = p.k`` of two whole-number columns."""

        if type(test) is not exp.EQ:
            return None
        for left, right in ((_unwrap(test.this), _unwrap(test.expression)), (_unwrap(test.expression), _unwrap(test.this))):
            if self.side(left) != "t" or self.side(right) != "p":
                continue
            qualified = right.copy()
            qualified.set("table", exp.to_identifier(self.keyed.alias))
            key = self.keyed.column(qualified, True)
            if key is None:
                return None  # a name of the list that is not a plain column of the table
            if column_kind(self.table.types.get(left.name.lower())) != "int" or column_kind(self.keyed.types.get(key)) != "int":
                return None
            return left.name.lower(), key
        return None


def _sides(select: exp.Select, schema, keys, not_null, types):
    """Each way of reading the two sources as (table, keyed source): one, or two when both are keyed tables."""

    source, joins = select.args.get(FROM_KEY), select.args.get("joins") or []
    if source is None or len(joins) != 1:
        return
    join = joins[0]
    if _set_args(join) - {"this", "on", "kind"} or (join.kind or "").upper() not in ("", "INNER", "CROSS"):
        return
    for first, second in ((source.this, join.this), (join.this, source.this)):
        table = Keyed.read(first, schema, keys, not_null, types)
        if table is None:
            continue
        keyed = Keyed.read(second, schema, keys, not_null, types) or Listed.read_list(second, schema, keys, not_null, types)
        if keyed is None or not keyed.keys or keyed.alias == table.alias:
            continue
        if isinstance(keyed, Listed) and (not isinstance(second, exp.Subquery) or keyed.inner.table.find_ancestor(exp.Subquery) is not second):
            continue  # a WITH name: the WITH queries are written in place before this pre-pass
        yield _Sides(table, keyed), join, (second if isinstance(second, exp.Subquery) else None)


def _rewrite(select: exp.Select, schema, keys, not_null, types, taken: set[str]) -> bool:
    if any(select.args.get(key) for key in _REFUSED):
        return False
    distinct = select.args.get("distinct")
    if distinct is not None and (not isinstance(distinct, exp.Distinct) or distinct.args.get("on")):
        return False
    windows = [node for node in select.find_all(exp.Window) if node.find_ancestor(exp.Select) is select]
    if not windows or not all(any(_inside(node, item) for item in select.expressions) for node in windows):
        return False
    return any(_read(select, windows, sides, join, derived, taken) for sides, join, derived in _sides(select, schema, keys, not_null, types))


def _read(select: exp.Select, windows: list[exp.Window], sides: _Sides, join: exp.Join, derived: exp.Subquery | None, taken: set[str]) -> bool:
    outside = lambda node: derived is None or not _inside(node, derived)  # noqa: E731
    if any(outside(node) for node in select.find_all((exp.Select, exp.Subquery, exp.SetOperation)) if node is not select):
        return False  # a subquery of the select: it may read either source
    if any(outside(node) and not isinstance(node.parent, exp.Window) for node in select.find_all(exp.AggFunc)):
        return False
    if any(outside(node) and not isinstance(node.parent, exp.Count) for node in select.find_all(exp.Star)):
        return False
    columns = [node for node in select.find_all(exp.Column) if outside(node)]
    if any(sides.side(node) is None for node in columns):
        return False
    # --- the conditions: t.c = p.k, or a test of p's columns only ---
    where = select.args.get("where")
    pairs: list[tuple[str, str]] = []
    for test in _conjuncts(join.args.get("on")) + _conjuncts(where.this if where is not None else None):
        pair = sides.pair(test)
        if pair is not None:
            pairs.append(pair)
        elif any(sides.side(node) != "p" for node in test.find_all(exp.Column)):
            return False
    joined = {column for column, _ in pairs}
    if not any(set(key) <= {key_column for _, key_column in pairs} for key in sides.keyed.keys):
        return False
    partner = {}
    for column, key_column in sorted(pairs):
        partner.setdefault(key_column, column)
    # --- the windows ---
    found: list[tuple[type, str | None]] = []
    partitions: set[tuple[str, ...]] = set()
    for node in windows:
        if _set_args(node) - {"this", "partition_by", "over"} or (node.text("over") or "OVER").upper() != "OVER":
            return False  # ORDER BY, a frame, a named window
        aggregate = node.this
        if type(aggregate) not in _AGGREGATES or _set_args(aggregate) - {"this", "big_int"}:
            return False
        if type(aggregate) is exp.Count and isinstance(aggregate.this, exp.Star):
            argument = None
        elif sides.side(_unwrap(aggregate.this)) == "t":
            argument = _unwrap(aggregate.this).name.lower()
        else:
            return False
        partition: list[str] = []
        for term in node.args.get("partition_by") or []:
            term = _unwrap(term)
            side = sides.side(term)
            if side == "t":
                partition.append(term.name.lower())
                continue
            qualified = term.copy() if side == "p" else None
            if qualified is None:
                return False
            qualified.set("table", exp.to_identifier(sides.keyed.alias))
            key_column = sides.keyed.column(qualified, True)
            if key_column not in partner:
                return False  # a column of p that no equality ties to a column of t
            partition.append(partner[key_column])
        if not partition or not joined <= set(partition):
            return False
        if any(column not in joined and column not in sides.table.never_null for column in partition):
            return False
        partitions.add(tuple(sorted(set(partition))))
        found.append((type(aggregate), argument))
    if len(partitions) != 1:
        return False
    # --- rewrite ---
    group = next(iter(partitions))
    alias = _fresh(taken)
    outputs = [exp.alias_(exp.column(column), f"{alias}k{index}") for index, column in enumerate(group)]
    named: dict[tuple[type, str | None], str] = {}
    for node, (aggregate, argument) in zip(windows, found):
        if (aggregate, argument) not in named:
            named[(aggregate, argument)] = f"{alias}v{len(named)}"
            outputs.append(exp.alias_(aggregate(this=exp.Star() if argument is None else exp.column(argument)), named[(aggregate, argument)]))
        node.replace(exp.column(named[(aggregate, argument)], table=alias))
    inner_table = sides.table.table.copy()
    inner_table.set("alias", None)
    grouped = exp.Select(expressions=outputs)
    grouped.set(FROM_KEY, exp.From(this=inner_table))
    grouped.set("group", exp.Group(expressions=[exp.column(column) for column in group]))
    condition = None
    for index, column in enumerate(group):
        test = exp.EQ(this=exp.column(column, table=sides.table.alias), expression=exp.column(f"{alias}k{index}", table=alias))
        condition = test if condition is None else exp.And(this=condition, expression=test)
    if not join.kind and join.args.get("on") is None:
        join.set("kind", "CROSS")  # a comma: a JOIN written after it would bind to the second source alone in MySQL
    select.append("joins", exp.Join(this=exp.Subquery(this=grouped, alias=exp.TableAlias(this=exp.to_identifier(alias))), on=condition))
    return True
