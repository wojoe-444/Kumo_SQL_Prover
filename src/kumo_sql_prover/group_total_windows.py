"""A window total of the group sums of a grouped select is the ungrouped sum, read as a scalar subquery.

    SELECT t.k, SUM(t.x) AS v, SUM(SUM(t.x)) OVER () AS total FROM t WHERE t.a > 0 GROUP BY t.k

is

    SELECT t.k, SUM(t.x) AS v, (SELECT SUM(t.x) FROM t WHERE t.a > 0) AS total FROM t WHERE t.a > 0 GROUP BY t.k

(TPC-H Q11 and the rewrites that compute its threshold as a window over the groups in place of the second scan).
The prover keeps a windowed select whole, so the window spelling never met the subquery one.

Why this is exact. ``OVER ()`` with nothing in it is one partition holding every row of the grouped result, one
row per group, and ``SUM`` adds the group sums that are not NULL. Every input row lies in exactly one group and a
group sum is the sum of its rows' values that are not NULL (NULL when it has none), so the window is the sum of
all the input rows' values that are not NULL, and NULL when there is none: the ungrouped ``SUM`` over the same
``FROM`` and ``WHERE``. With no input rows there is no group and no output row to carry the value. The subquery is
the select's own ``FROM`` and ``WHERE``, one level down, where each name binds as it does in the select.

Conditions, all checked:

* the window is ``SUM(SUM(e)) OVER ()``: no ``PARTITION BY``, ``ORDER BY``, frame or named window, no ``DISTINCT`` or
  ``FILTER`` on either sum, and ``e`` holds no aggregate, window or nested query;
* it sits in the select list of a select with a plain ``GROUP BY`` (no ``ROLLUP``, ``CUBE``, ``GROUPING SETS``) and no
  ``HAVING`` (which would remove groups before the window sees them), ``DISTINCT`` or ``QUALIFY``;
* the select's ``FROM`` and ``WHERE`` hold no nested query other than derived tables in ``FROM``, and nothing in the
  select is volatile.

This is a pre-pass of ``normalize``, before windowed selects are set aside.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, extended_grouping


def _total(window: exp.Expression) -> exp.Expression | None:
    """``e`` for a window ``SUM(SUM(e)) OVER ()``, or ``None``."""

    if type(window) is not exp.Window or any(value for key, value in window.args.items() if key not in ("this", "over")):
        return None
    outer = window.this
    if type(outer) is not exp.Sum or any(value for key, value in outer.args.items() if key != "this"):
        return None
    inner = outer.this
    if type(inner) is not exp.Sum or any(value for key, value in inner.args.items() if key != "this"):
        return None
    value = inner.this
    if value is None or isinstance(value, exp.Distinct):
        return None
    if any(isinstance(node, (exp.AggFunc, exp.Window, exp.Query, exp.Subquery)) for node in value.walk()):
        return None
    return value


def _reads(select: exp.Select) -> bool:
    """Whether the select's ``FROM`` and ``WHERE`` can be read a second time as a scalar subquery."""

    from .counted_membership import _unstable

    if select.args.get(FROM_KEY) is None or _unstable(select):
        return False
    group = select.args.get("group")
    if group is None or not group.expressions or extended_grouping(group) or any(group.args.get(k) for k in ("all", "totals")):
        return False
    if any(select.args.get(key) for key in ("having", "distinct", "qualify", "windows", "laterals")):
        return False
    where = select.args.get("where")
    if where is not None and any(isinstance(node, (exp.Query, exp.Subquery)) for node in where.walk()):
        return False
    for join in select.args.get("joins") or []:
        on = join.args.get("on")
        if isinstance(join.this, exp.Lateral) or (on is not None and any(isinstance(node, (exp.Query, exp.Subquery)) for node in on.walk())):
            return False
    return True


def read_group_totals(tree: exp.Expression) -> exp.Expression:
    """``tree`` with every such window written as a scalar subquery over its select's ``FROM`` and ``WHERE``."""

    for window in list(tree.find_all(exp.Window)):
        value = _total(window)
        select = window.find_ancestor(exp.Select)
        if value is None or select is None or not _reads(select):
            continue
        item = window
        while item.parent is not None and item.parent is not select:
            item = item.parent
        if not any(item is written for written in select.expressions):
            continue  # only in the select list: ORDER BY may name an output
        scalar = exp.Select(expressions=[exp.Sum(this=value.copy())])
        scalar.set(FROM_KEY, select.args[FROM_KEY].copy())
        if select.args.get("joins"):
            scalar.set("joins", [join.copy() for join in select.args["joins"]])
        if select.args.get("where") is not None:
            scalar.set("where", select.args["where"].copy())
        window.replace(exp.Subquery(this=scalar))
    return tree
