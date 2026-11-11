"""A sum that picks whole groups of a pre-aggregated table by a ``CASE`` is the flat sum with the same ``CASE``.

    SELECT SUM(CASE WHEN p.kind = 'promo' THEN d.total ELSE 0 END), SUM(d.total)
    FROM (SELECT l.pk, SUM(l.price) AS total FROM l GROUP BY l.pk) AS d
    JOIN p ON d.pk = p.pk

is

    SELECT SUM(CASE WHEN p.kind = 'promo' THEN l.price ELSE 0 END), SUM(l.price)
    FROM l, p
    WHERE l.pk = p.pk

(TPC-H Q14 and the rewrites that total the line items of a part before the join with ``part``).
``eager_aggregation.unnest_grouped_source`` already reads ``SUM(d.total)`` and ``SUM(d.total * w)``; a ``CASE``
is not a product, so this rule adds it.

Why this is exact. ``unnest_grouped_source`` checks that each row of the join with ``d`` stands for the rows of
the flat join that share its group and its partners in the other tables, and that nothing but the rewritten sums
reads an aggregate of ``d``. The condition ``c`` reads no aggregate of ``d``, so it has one value for all of
those flat rows. Where it is true, the row adds ``d.total``, the sum of the group's values, and the flat rows add
those values. Where it is false or NULL, the row adds 0 once and the flat rows add 0 each (a group has a row), or
with no ``ELSE`` neither side adds anything. A sum is NULL when no term is a value: that is every row having a
true ``c`` and a NULL total on one side, and every flat row having a true ``c`` and a NULL value on the other,
which are the same databases.

How it is done: each ``SUM(CASE WHEN c THEN d.s [ELSE 0] END)`` is handed to ``unnest_grouped_source`` as the
product ``SUM(d.s * mark(c))`` with an opaque ``mark``, which it rewrites to ``SUM((e) * mark(c))`` (or
``SUM(mark(c))`` when ``d.s`` is ``COUNT(*)``), and each mark is then read back as the ``CASE`` over ``e`` (or
over 1). Any other place a mark ends up declines the rewrite.

Conditions, all checked:

* the argument of the ``SUM`` is one searched ``CASE`` with one ``WHEN``, whose ``THEN`` is a qualified column and
  whose ``ELSE`` is absent, ``NULL`` or the literal 0; no ``DISTINCT``, ``FILTER`` or window;
* ``c`` holds no aggregate, window, nested query or volatile function;
* everything ``unnest_grouped_source`` requires of the select, with the product in place of the ``CASE``; this
  includes that the ``THEN`` column is a ``SUM`` or ``COUNT(*)`` of the grouped table and that ``c`` reads none
  of its aggregates.
"""

from __future__ import annotations

from sqlglot import exp

_ZERO, _NOTHING = "kumosql_cased_sum_zero", "kumosql_cased_sum_null"


def _case_of(call: exp.Expression, select: exp.Select) -> tuple[exp.Case, bool] | None:
    """The ``CASE`` a ``SUM`` of ``select`` totals and whether it has ``ELSE 0``, or ``None``."""

    from .counted_membership import _unstable

    if type(call) is not exp.Sum or call.find_ancestor(exp.Select) is not select or isinstance(call.parent, (exp.Window, exp.Filter)):
        return None
    if any(value for key, value in call.args.items() if key != "this"):
        return None
    case = call.this
    while type(case) is exp.Paren:
        case = case.this
    if type(case) is not exp.Case or case.args.get("this") is not None or len(case.args.get("ifs") or []) != 1:
        return None
    branch = case.args["ifs"][0]
    then, default = branch.args.get("true"), case.args.get("default")
    if type(then) is not exp.Column or not then.table or isinstance(then.this, exp.Star):
        return None
    zero = type(default) is exp.Literal and not default.is_string and default.this == "0"
    if not (default is None or type(default) is exp.Null or zero):
        return None
    test = branch.this
    if any(isinstance(node, (exp.AggFunc, exp.Window, exp.Query, exp.Subquery)) for node in test.walk()) or _unstable(test):
        return None
    return case, zero


def unnest_cased_sums(select: exp.Select) -> exp.Expression | None:
    """``select`` over the flat join with each such ``CASE`` moved onto the summed value, or ``None``."""

    from .eager_aggregation import unnest_grouped_source

    if not any(_case_of(call, select) for call in select.find_all(exp.Sum)):
        return None
    if any(isinstance(node, exp.Anonymous) and node.name.lower() in (_ZERO, _NOTHING) for node in select.walk()):
        return None
    marked = select.copy()
    for call in list(marked.find_all(exp.Sum)):
        found = _case_of(call, marked)
        if found is None:
            continue
        case, zero = found
        branch = case.args["ifs"][0]
        mark = exp.Anonymous(this=_ZERO if zero else _NOTHING, expressions=[branch.this.copy()])
        call.set("this", exp.Mul(this=branch.args["true"].copy(), expression=mark))
    flat = unnest_grouped_source(marked)
    if flat is None:
        return None
    for mark in [node for node in flat.find_all(exp.Anonymous) if node.name.lower() in (_ZERO, _NOTHING)]:
        parent = mark.parent
        if type(parent) is exp.Mul and parent.expression is mark and type(parent.this) is exp.Paren and type(parent.parent) is exp.Sum:
            value, spot = parent.this.this.copy(), parent
        elif type(parent) is exp.Sum and parent.this is mark:
            value, spot = exp.Literal.number(1), mark
        else:
            return None
        if any(isinstance(node, exp.Anonymous) and node.name.lower() in (_ZERO, _NOTHING) for node in value.walk()):
            return None
        case = exp.Case(ifs=[exp.If(this=mark.expressions[0].copy(), true=value)])
        if mark.name.lower() == _ZERO:
            case.set("default", exp.Literal.number(0))
        spot.replace(case)
    return flat
