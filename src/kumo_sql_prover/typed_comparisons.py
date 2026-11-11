"""Two comparison spellings that the declared column types make the same.

**1. A date plus an interval, compared with a date.**

    l.shipdate < o.orderdate + INTERVAL '10' DAY
    l.shipdate < DATE_ADD(o.orderdate, INTERVAL 10 DAY)

In GoogleSQL ``date + INTERVAL n DAY`` is a DATETIME, the midnight of ``DATE_ADD(date, INTERVAL n DAY)``, and a DATE
compared with a DATETIME is read as its own midnight. Two midnights compare as their days do, so inside a
comparison whose two operands are each a date, a ``DATE_ADD`` / ``DATE_SUB`` of a date or a date ``+`` / ``-`` an
interval, the operator spelling is rewritten to the function. The provers read each spelling as an unknown
function of its own, so the two never matched. Read for whole-day units only (``DAY``, ``WEEK``, ``MONTH``,
``QUARTER``, ``YEAR``; a shorter unit leaves midnight) with a whole number of them, and a date that is a column
declared ``DATE`` or a cast to ``DATE``. Outside a comparison the two differ in type and nothing is rewritten.

**2. The same whole number added to both sides.**

    lineitem.orderkey + 1 = orders.orderkey + 1
    lineitem.orderkey = orders.orderkey

For two integer expressions ``x + n OP y + n`` is ``x OP y`` (and ``x - n OP y - n``), for every comparison: adding a
number moves both sides alike and a NULL side stays NULL. An overflow of INT64 is an error in GoogleSQL, not
another value, and a proof holds where both queries succeed. Read only when both ``x`` and ``y`` have an integer
type (floating-point addition rounds: two different values can have the same sum) and ``n`` is the same integer
literal on both sides.

Both are read in GoogleSQL only, with declared types, for columns of the comparison's own select (a table or a
derived table in its ``FROM``); a query with a CTE, a ``PIVOT`` or an alias with a column list is left alone, since
a name could then mean something the declared types do not describe.
"""

from __future__ import annotations

import re

from sqlglot import exp

_COMPARISONS = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE)
_WHOLE_DAYS = {"DAY", "WEEK", "MONTH", "QUARTER", "YEAR"}
_DIGITS = re.compile(r"[0-9]+")


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _kind(node: exp.Expression, select: exp.Select, types: dict) -> str | None:
    from .cast_rules import expression_type

    found = expression_type(node, select, types)
    return found[0] if found else None


def _whole_days(amount: exp.Expression | None, unit: exp.Expression | None) -> bool:
    return isinstance(amount, exp.Literal) and bool(_DIGITS.fullmatch(amount.name)) and unit is not None and unit.name.upper() in _WHOLE_DAYS


def _date_operand(node: exp.Expression, select: exp.Select, types: dict) -> str | None:
    """``"date"`` for a date, ``"shifted"`` for ``DATE_ADD`` / ``DATE_SUB`` of one, ``"spelled"`` for a date ``+`` / ``-`` an interval."""

    node = _unwrap(node)
    if type(node) in (exp.DateAdd, exp.DateSub):
        ok = not any(v for k, v in node.args.items() if k not in ("this", "expression", "unit")) and _whole_days(node.expression, node.args.get("unit"))
        return "shifted" if ok and _kind(node.this, select, types) == "date" else None
    if type(node) in (exp.Add, exp.Sub):
        interval = node.expression
        ok = type(interval) is exp.Interval and not any(v for k, v in interval.args.items() if k not in ("this", "unit")) and _whole_days(interval.this, interval.args.get("unit"))
        return "spelled" if ok and _kind(node.this, select, types) == "date" else None
    return "date" if _kind(node, select, types) == "date" else None


def _as_function(node: exp.Expression) -> exp.Expression:
    interval = node.expression
    function = exp.DateAdd if type(node) is exp.Add else exp.DateSub
    return function(this=node.this.copy(), expression=exp.Literal.string(interval.this.name), unit=exp.var(interval.args["unit"].name.upper()))


def _addend(node: exp.Expression) -> tuple[exp.Expression, str] | None:
    """``(x, "+n" or "-n")`` for ``x + n``, ``n + x`` and ``x - n`` with an integer literal ``n``."""

    node = _unwrap(node)
    if type(node) not in (exp.Add, exp.Sub):
        return None
    left, right = _unwrap(node.this), _unwrap(node.expression)
    number = lambda n: isinstance(n, exp.Literal) and not n.is_string and bool(_DIGITS.fullmatch(n.name))  # noqa: E731
    if number(right) and not number(left):
        return node.this, ("+" if type(node) is exp.Add else "-") + str(int(right.name))
    if type(node) is exp.Add and number(left) and not number(right):
        return node.expression, "+" + str(int(left.name))
    return None


def typed_comparison_rules(tree: exp.Expression, types: dict | None, dialect: str = "bigquery") -> exp.Expression:
    """``tree`` with the two comparison spellings of the module doc rewritten where the declared types allow."""

    if dialect != "bigquery" or not types or tree.find(exp.Add, exp.Sub) is None:
        return tree
    if tree.find(exp.With, exp.Pivot) is not None or any(alias.args.get("columns") for alias in tree.find_all(exp.TableAlias)):
        return tree
    for node in list(tree.find_all(*_COMPARISONS)):
        select = node.find_ancestor(exp.Select)
        if select is None or type(node) not in _COMPARISONS:
            continue
        sides = [_date_operand(side, select, types) for side in (node.this, node.expression)]
        if None not in sides and "spelled" in sides:
            for key, side in zip(("this", "expression"), sides):
                if side == "spelled":
                    node.set(key, _as_function(_unwrap(node.args[key])))
            continue
        left, right = _addend(node.this), _addend(node.expression)
        if left is not None and right is not None and left[1] == right[1] and _kind(left[0], select, types) == "int" and _kind(right[0], select, types) == "int":
            node.set("this", left[0].copy())
            node.set("expression", right[0].copy())
    return tree
