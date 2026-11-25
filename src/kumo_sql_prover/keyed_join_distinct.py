"""``SELECT DISTINCT`` over an inner join whose outputs fix every joined row removes nothing.

    SELECT DISTINCT m.staff_id AS lead_id, r.staff_id, m.label, r.age
    FROM staff AS m JOIN staff AS r ON m.staff_id = r.lead_id

lists a row per pair of joined rows. ``staff_id`` is the NOT NULL key of ``staff`` and both reads show it, so two
pairs that show the same values are the same pair: the ``DISTINCT`` has nothing to remove and the select is

    SELECT m.staff_id AS lead_id, r.staff_id, m.label, r.age
    FROM staff AS m JOIN staff AS r ON m.staff_id = r.lead_id

``keyed_rules.drop_keyed_distinct`` reads this for a select over one table. A grouped query above a ``DISTINCT``
derived table of a join was read only when it showed its group keys and aggregates; with one more output (the
lead's label above) no prover took it (hardest-cases batch 32, LeetCode 1731 "employees who report to each lead").

**Why it removes nothing.** An inner join returns each combination of one row per source that passes the tests,
once. Call a column *known* when the output row tells its value:

* a plain column that is an output is known;
* a column that the ``WHERE`` or an ``ON`` equates to a known column or to a whole-number literal (a top-level
  ``AND`` part ``a = b``) is known: the test is TRUE on every listed combination, so both are not NULL and equal.
  Texts can be equal under one collation and distinct under another, and a text can equal several numbers' spellings,
  so this step is taken only between columns declared with a whole-number type;
* when every column of a NOT NULL key of a declared table is known, the output row tells which row of that table
  the combination holds, so all its columns are known. A key column must be a whole number or a date: two stored
  values of such a type are either the same value or show as different outputs.

If that reaches every source, an output row names one combination, so no two listed rows are equal.

**What is read.** A select with a plain ``DISTINCT``, no ``GROUP BY``, ``HAVING``, aggregate, window, ``ORDER BY`` or
``LIMIT``, whose sources are all declared tables (not hidden by a ``WITH`` name) with a NOT NULL key, joined by inner
joins with ``ON``, ``CROSS JOIN`` or commas. Outputs that are not plain columns are allowed: they add nothing to
what is known and take nothing away.

**Left alone.** An outer join (a padded row has no row of the table), ``USING`` and ``NATURAL``, a derived table or
``WITH`` name among the sources (its rows can repeat), a key that is nullable, a key column of another type, a
column name two sources have when written bare, ``DISTINCT ON``.

It runs in the last step of batch 27 only, so a pair an earlier step decides is decided as it was.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, conjuncts
from .key_fixed_groupings import _JOIN_KINDS, _Sources, _set_args, _unwrap
from .single_row_aggregates import column_kind

NAME = "keyed_join_distinct"
_SELECT_ARGS = frozenset({"expressions", FROM_KEY, "joins", "where", "distinct"})
_DATE_TYPES = frozenset({"DATE"})


def _key_kind(sources: _Sources, column: tuple[str, str]) -> bool:
    """Whether two stored values of the column are the same value or show as different outputs: a whole number or a date."""

    table = sources.keyed.get(column[0])
    if table is None:
        return False
    declared = table.types.get(column[1])
    return column_kind(declared) == "int" or str(declared or "").strip().upper() in _DATE_TYPES


def read_keyed_join_distinct(select: exp.Select, schema, keys, not_null, types) -> exp.Select | None:
    """``select`` without its ``DISTINCT`` when the outputs fix every joined row (module doc), else ``None``. The input is not changed."""

    if not schema or not keys:
        return None
    distinct = select.args.get("distinct")
    if not isinstance(distinct, exp.Distinct) or _set_args(distinct):
        return None  # no DISTINCT, or DISTINCT ON
    if _set_args(select) - _SELECT_ARGS or select.args.get(FROM_KEY) is None:
        return None
    if any(isinstance(node, (exp.AggFunc, exp.Window)) and node.find_ancestor(exp.Select) is select for node in select.walk()):
        return None
    sources = _Sources()
    joins = select.args.get("joins") or []
    if not joins:
        return None  # one table: ``keyed_rules.drop_keyed_distinct``
    for source in [select.args[FROM_KEY].this] + [join.this for join in joins]:
        if type(source) is not exp.Table or not sources.add(source, schema, keys, not_null, types):
            return None
    if set(sources.columns) != set(sources.keyed) or not all(table.keys for table in sources.keyed.values()):
        return None
    for join in joins:
        if _set_args(join) - {"this", "on", "kind"} or (join.args.get("kind") or "").upper() not in _JOIN_KINDS:
            return None

    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(column: tuple[str, str]) -> tuple[str, str]:
        while parent.get(column, column) != column:
            column = parent[column]
        return column

    known: set[tuple[str, str]] = set()
    tests = [part for join in joins if join.args.get("on") is not None for part in conjuncts(join.args["on"])]
    if select.args.get("where") is not None:
        tests += conjuncts(select.args["where"].this)
    fixed: list[tuple[str, str]] = []
    for test in tests:
        test = _unwrap(test)
        if not isinstance(test, exp.EQ):
            continue
        left, right = sources.column(test.this), sources.column(test.expression)
        if left is not None and right is not None:
            if sources.whole(left) and sources.whole(right):
                parent[find(left)] = find(right)
            continue
        for column, other in ((left, test.expression), (right, test.this)):
            other = _unwrap(other)
            if column is not None and sources.whole(column) and isinstance(other, exp.Literal) and not other.is_string and other.this.isdigit():
                fixed.append(column)
    for item in select.expressions:
        column = sources.column(item.this if isinstance(item, exp.Alias) else item)
        if column is not None:
            known.add(find(column))
    known |= {find(column) for column in fixed}

    reached: set[str] = set()
    while len(reached) < len(sources.keyed):
        step = [
            name
            for name, table in sources.keyed.items()
            if name not in reached and any(all(_key_kind(sources, (name, part)) and find((name, part)) in known for part in key) for key in table.keys)
        ]
        if not step:
            return None
        for name in step:
            reached.add(name)
            known |= {find((name, column)) for column in sources.columns[name]}
    copy = select.copy()
    copy.set("distinct", None)
    return copy
