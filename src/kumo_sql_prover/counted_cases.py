"""In a grouped select, ``COUNT(CASE WHEN c THEN 1 END)`` is ``SUM(CASE WHEN c THEN 1 ELSE 0 END)``.

Both count the rows of a group where ``c`` is TRUE: the ``CASE`` without ``ELSE`` is NULL on the other rows, which
``COUNT`` skips, and the ``CASE`` with ``ELSE 0`` adds nothing for them (TPC-H Q12 and its rewrites). Over *no*
rows they differ, ``COUNT`` being 0 and ``SUM`` NULL, so the rule reads only a select with a plain ``GROUP BY``:
every group it returns holds at least one row.

Conditions, all checked:

* the select has a ``GROUP BY`` with at least one expression and no ``ROLLUP``, ``CUBE`` or ``GROUPING SETS`` (their
  grand-total row exists over no rows);
* the call is a plain ``COUNT`` of this select: no ``DISTINCT``, ``FILTER`` or window;
* its argument is a searched ``CASE`` whose every ``THEN`` is a number or string literal (never NULL) and which
  has no ``ELSE`` or ``ELSE NULL``. Each ``THEN`` becomes 1 and ``ELSE 0`` is added.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import extended_grouping


def _counted_case(node: exp.Expression, select: exp.Select) -> exp.Case | None:
    if type(node) is not exp.Count or node.find_ancestor(exp.Select) is not select:
        return None
    if any(v for key, v in node.args.items() if key not in ("this", "big_int")) or isinstance(node.parent, (exp.Window, exp.Filter)):
        return None
    case = node.this
    while type(case) is exp.Paren:
        case = case.this
    if type(case) is not exp.Case or case.this is not None or not case.args.get("ifs"):
        return None
    default = case.args.get("default")
    if default is not None and type(default) is not exp.Null:
        return None
    for arm in case.args["ifs"]:
        if type(arm) is not exp.If or type(arm.args.get("true")) is not exp.Literal:
            return None
        if any(isinstance(n, (exp.AggFunc, exp.Window, exp.Query, exp.Subquery)) for n in arm.this.walk()):
            return None
    return case


def sum_counted_cases(select: exp.Select) -> exp.Expression | None:
    """``select`` with every ``COUNT(CASE WHEN .. THEN literal END)`` of its own written as a ``SUM``, or ``None``."""

    group = select.args.get("group")
    if group is None or not group.expressions or extended_grouping(group):
        return None
    if not any(_counted_case(node, select) is not None for node in select.find_all(exp.Count)):
        return None
    copy = select.copy()
    for node in list(copy.find_all(exp.Count)):
        case = _counted_case(node, copy)
        if case is None:
            continue
        new = case.copy()
        for arm in new.args["ifs"]:
            arm.set("true", exp.Literal.number(1))
        new.set("default", exp.Literal.number(0))
        node.replace(exp.Sum(this=new))
    return copy
