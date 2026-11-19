"""A sum of ones divided by a count: the sum is the count of the rows that pass.

    SELECT SUM(CASE WHEN paid THEN 1 ELSE 0 END) / COUNT(*) FROM bill      -- COUNT(CASE WHEN paid THEN 1 END) / COUNT(*)

``conditional_count_sums`` reads ``SUM(CASE WHEN p THEN 1 ELSE 0 END)`` as ``COUNT(CASE WHEN p THEN 1 END)`` where every
group has a row. In a select with no ``GROUP BY`` the two differ on exactly one input, no rows at all: the sum is NULL
and the count is 0. Under a division by a count of the same rows that difference cannot be seen.

**Why it holds.** Let the select aggregate the rows ``R`` (its sources after ``WHERE``). If ``R`` has a row, the sum
of ones is the number of rows on which ``p`` is TRUE, which is the count: the dividend is the same number either way.
If ``R`` is empty, the divisor, a ``COUNT`` over ``R``, is 0, and in MySQL and SQLite a division by zero is NULL whatever
the dividend is (``NULL / 0`` and ``0 / 0`` are both NULL). A product of the sum with number literals is NULL when the
sum is and a number when it is one, so ``S * 100 / n`` and ``100 * S / n`` are covered by the same argument. Both
spellings are whole numbers, and MySQL gives the quotient of a DECIMAL sum or a BIGINT count by a BIGINT count the
same four more decimals.

``SUM(p)`` of a comparison ``p`` (MySQL and SQLite read TRUE as 1 and FALSE as 0) is the same sum when ``p`` is never
NULL, and otherwise it is not: with rows and ``p`` NULL on all of them the sum is NULL while the count is 0 and the
divisor is not zero. So it is read only when ``p`` compares columns declared NOT NULL of the select's one base table
with each other or with literals that are not NULL.

A column ``f`` that the select's one derived table outputs as ``CASE WHEN p THEN 1 ELSE 0 END AS f`` (or ``IF(p, 1, 0)``)
is 1 or 0 on every row and never NULL, so with a row ``SUM(f)`` is the number of rows on which ``f = 1``: it is
written ``COUNT(CASE WHEN f = 1 THEN 1 END)``. The select reads the derived table alone (a ``LEFT JOIN`` could add a
row on which ``f`` is NULL).

**What is read.** In a select with no ``GROUP BY``, no ``HAVING`` and no window, each ``/`` whose divisor is a
``COUNT`` of this select (any argument, also ``DISTINCT``; no ``FILTER``) and whose dividend is such a sum, or a
product of one such sum and number literals. Only the sum is rewritten. MySQL and SQLite only: elsewhere a division
by zero is an error or follows other rules. The sum and the count must be aggregates of this select (an aggregate
that reads only columns of an enclosing query belongs to that query); ``conditional_count_sums`` has the test.

A narrow retry rule of the tenth hardest-cases batch (``shape_fallback.TENTH_BATCH``).
``tests/test_counted_sum_ratios.py`` replays the rewrite on SQLite, which divides by zero as MySQL does.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import select_sources
from .conditional_count_sums import _condition, _own, _own_aggregate

DIALECTS = frozenset({"mysql", "sqlite"})
_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)


def _unparen(node: exp.Expression | None) -> exp.Expression | None:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _number(node: exp.Expression | None) -> bool:
    node = _unparen(node)
    return isinstance(node, exp.Literal) and not node.is_string


def _sum_in(dividend: exp.Expression | None) -> exp.Sum | None:
    """The one ``SUM`` of ``dividend`` when it is that sum or a product of it and number literals."""

    node = _unparen(dividend)
    if isinstance(node, exp.Sum):
        return node
    if isinstance(node, exp.Mul):
        left, right = _unparen(node.this), _unparen(node.expression)
        if _number(right):
            return _sum_in(left)
        if _number(left):
            return _sum_in(right)
    return None


def _if_condition(call: exp.Sum) -> exp.Expression | None:
    """``p`` for ``SUM(IF(p, 1, 0))``."""

    arm = _unparen(call.this)
    if type(arm) is exp.If and arm.args.get("true") is not None and arm.args.get("false") is not None:
        one, zero = _unparen(arm.args["true"]), _unparen(arm.args["false"])
        if isinstance(one, exp.Literal) and not one.is_string and one.name == "1" and isinstance(zero, exp.Literal) and not zero.is_string and zero.name == "0":
            return arm.this
    return None


def _never_null_comparison(call: exp.Sum, select: exp.Select, not_null: dict | None) -> exp.Expression | None:
    """``p`` for ``SUM(p)`` when ``p`` is a comparison that is never NULL on a row of the select's one base table."""

    test = _unparen(call.this)
    if not isinstance(test, _COMPARISONS):
        return None
    sources = select_sources(select)
    if len(sources) != 1 or not isinstance(sources[0], exp.Table) or sources[0].args.get("db") or sources[0].args.get("catalog"):
        return None
    table = sources[0]
    declared = {str(name).lower() for name in (not_null or {}).get(table.name.lower(), ())}
    for side in (test.this, test.expression):
        side = _unparen(side)
        if isinstance(side, exp.Literal):
            continue
        if not isinstance(side, exp.Column) or side.args.get("db") or (side.table and side.table.lower() != table.alias_or_name.lower()):
            return None
        if side.name.lower() not in declared:
            return None
    return test


def _flag_test(call: exp.Sum, select: exp.Select) -> exp.Expression | None:
    """``c = 1`` for ``SUM(c)`` when ``c`` is a 1-or-0 flag that the select's one derived table outputs."""

    column = _unparen(call.this)
    sources = select_sources(select)
    if not isinstance(column, exp.Column) or not isinstance(column.this, exp.Identifier) or column.args.get("db") or len(sources) != 1 or select.args.get("joins"):
        return None
    source = sources[0]
    if type(source) is not exp.Subquery or type(source.this) is not exp.Select or not source.alias or source.args["alias"].args.get("columns"):
        return None
    if column.table and column.table.lower() != source.alias.lower():
        return None
    items = [item for item in source.this.expressions if (item.alias_or_name or "").lower() == column.name.lower()]
    if len(items) != 1 or not isinstance(items[0], exp.Alias):
        return None
    probe = exp.Sum(this=items[0].this.copy())
    condition = _condition(probe, need_else=True)
    if (condition is None or isinstance(condition, exp.Boolean)) and _if_condition(probe) is None:
        return None
    return exp.EQ(this=column.copy(), expression=exp.Literal.number(1))


def read_counted_sum_ratios(select: exp.Select, placed: exp.Select | None = None, not_null: dict | None = None, dialect: str = "mysql") -> exp.Select | None:
    """``select`` with each sum of ones under a division by a count (module doc) written as a count, or None.

    ``placed`` is the same select where it stands in its query, when ``select`` is a copy of it."""

    if dialect not in DIALECTS or select.args.get("group") is not None or select.args.get("having") is not None:
        return None
    if not any(True for _ in select.find_all(exp.Div)) or any(True for _ in select.find_all(exp.Window)):
        return None
    placed = select if placed is None else placed
    copy = select.copy()
    changed = False
    for division in list(copy.find_all(exp.Div)):
        if not _own(division, copy):
            continue
        divisor = _unparen(division.expression)
        if type(divisor) is not exp.Count or divisor.find_ancestor(exp.Select) is not copy or isinstance(divisor.parent, (exp.Window, exp.Filter)):
            continue
        if any(True for _ in divisor.find_all(exp.Column)) and not _own_aggregate(divisor, copy, placed):
            continue
        call = _sum_in(division.this)
        if call is None or not _own(call, copy) or any(v for k, v in call.args.items() if k != "this"):
            continue
        condition = _condition(call, need_else=True)
        if condition is None or (isinstance(condition, exp.Boolean) and condition.this is True):
            condition = _if_condition(call) or _never_null_comparison(call, copy, not_null)
        if condition is None:
            condition = _flag_test(call, copy)  # reads a column of this select's own derived table, by construction
            if condition is None:
                continue
        elif any(True for _ in condition.find_all(exp.Select)) or not _own_aggregate(condition, copy, placed):
            continue
        call.replace(exp.Count(this=exp.Case(ifs=[exp.If(this=condition.copy(), true=exp.Literal.number(1))])))
        changed = True
    return copy if changed else None
