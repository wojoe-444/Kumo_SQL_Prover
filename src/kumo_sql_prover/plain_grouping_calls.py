"""``GROUPING(..)`` under a plain ``GROUP BY`` is 0.

    SELECT a, SUM(x) AS s, GROUPING(a) = 0 AS g FROM t GROUP BY a      -- g is TRUE on every row

``GROUPING(e1, .., en)`` is the bit mask of its arguments that are *not* part of the grouping set the row was
grouped by (the first argument is the high bit). A plain ``GROUP BY`` list has one grouping set, which holds
every key, so each bit is 0 and the call is the integer 0 on every row, whatever the keys' values (a NULL key
is still a key of the set). With no input rows there is no row to evaluate it on.

Calcite writes these calls when it expands ``COUNT(DISTINCT ..)`` and ``FILTER`` aggregates into a grouped
derived table with flag columns (``GROUPING(a) = 0 AS g_0``), also when only one grouping set is left. Under
``ROLLUP``, ``CUBE`` or ``GROUPING SETS`` the call is evaluated per set by ``grouping_sets.grouping_sets_to_union``;
this module reads the plain case that one leaves alone.

Conditions, all checked:

* the select has a plain ``GROUP BY`` list with at least one key: no ``ROLLUP``, ``CUBE``, ``GROUPING SETS``,
  ``WITH TOTALS``, ``GROUP BY ALL`` or ``()``;
* the call belongs to that select (its select list, ``HAVING`` or ``ORDER BY``, not a nested query's);
* every argument is one of the keys: the same text, or the same column with or without its qualifier (a name
  two keys share is not matched without a qualifier). A call with any other argument is left alone, since no
  engine accepts it and this module should not give it a meaning.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import extended_grouping
from .grouping_sets import is_grouping_call


def _is_key(argument: exp.Expression, keys: list[exp.Expression]) -> bool:
    text = argument.sql()
    if any(key.sql() == text for key in keys):
        return True
    if not isinstance(argument, exp.Column) or argument.args.get("db") or argument.args.get("catalog"):
        return False
    named = [key for key in keys if isinstance(key, exp.Column) and key.name.lower() == argument.name.lower() and not key.args.get("db")]
    if len(named) != 1:
        return False
    key = named[0]
    return not argument.table or not key.table or argument.table.lower() == key.table.lower()


def fold_plain_grouping_calls(tree: exp.Expression) -> exp.Expression:
    """Replace each ``GROUPING(keys..)`` of a select with a plain ``GROUP BY`` by 0 (module doc); edits ``tree``."""

    for select in list(tree.find_all(exp.Select)):
        group = select.args.get("group")
        if group is None or not group.expressions or extended_grouping(group) or any(group.args.get(k) for k in ("all", "totals")):
            continue
        keys = list(group.expressions)
        if any(isinstance(key, (exp.Literal, exp.Tuple)) for key in keys):
            continue  # a position or a row of keys: not read here
        for call in [n for n in select.find_all(exp.Expression) if is_grouping_call(n)]:
            if call.find_ancestor(exp.Select) is not select or call.find_ancestor(exp.Group) is not None:
                continue
            arguments = list(call.expressions)
            if arguments and all(_is_key(argument, keys) for argument in arguments):
                call.replace(exp.Literal.number(0))
    return tree
