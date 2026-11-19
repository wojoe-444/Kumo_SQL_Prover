"""An average of a 1-or-0 flag under a rounding is the sum of the flag over the count of the rows: MySQL only.

    ROUND(AVG(CASE WHEN paid THEN 1 ELSE 0 END) * 100, 2)        -- ROUND(SUM(CASE WHEN paid THEN 1 ELSE 0 END) / COUNT(*) * 100, 2)
    ROUND(AVG(IF(paid, 100, 0)), 2)                              -- the same
    ROUND(100 * AVG(paid = due), 2)                              -- the same, with paid and due declared NOT NULL

**Why it holds.** ``AVG(e)`` is the sum of the values of ``e`` that are not NULL over their number. When ``e`` is never
NULL that number is the number of rows, ``COUNT(*)``, and both spellings are NULL when there is no row (the average
of nothing, and a division by zero, which is NULL in MySQL). When every value of ``e`` is ``10^k`` times a whole number
``f``, ``SUM(e)`` is ``10^k * SUM(f)`` exactly, so the value is ``10^k * SUM(f) / COUNT(*)``.

What is left is MySQL's arithmetic. ``AVG`` of whole numbers keeps the digits a division keeps: the quotient cut off
after nine decimals, though its declared scale is four. Under ``ROUND(.., r)`` every spelling of ``10^j * n / d`` over
whole numbers is the exact value rounded half up while ``j + r <= 8`` (``rounded_percentages`` has the argument), and
the average is one more spelling of it. So the rewrite is made only where the average stands directly under a
``ROUND``, with nothing but parentheses and factors that are powers of ten between them, and with ``j + r <= 8`` for all
the powers of ten together, inside the average and outside it.

Measured on MySQL 8.0.46 (the server's own digits, read back with the point moved eight places): ``AVG(e)`` and
``SUM(e) / COUNT(e)`` agree in every digit for a comparison, a ``CASE`` of 1 and 0, a ``CASE`` of 100 and 0, an INT
column and a DECIMAL column, on 1,879 tables of 1 to 3,000 rows chosen next to rounding ties, and the rounded
spellings equal the exact value rounded half up on each; grouped selects with and without ``ORDER BY`` agree too
(``tests/test_averaged_flags.py`` names the measurements and models the cut on exact arithmetic).

**What is read.** ``AVG(e)`` that is an aggregate of its own select (no ``DISTINCT``, ``FILTER`` or ``OVER``), where ``e``
is

* a ``CASE`` with no operand, or an ``IF``, whose every result is a whole-number literal that is not negative,
  with an ``ELSE`` (so it is never NULL);
* such a flag times whole-number literals that are powers of ten (``IF(p, 1, 0) * 100``);
* a comparison of columns declared NOT NULL of the select's one base table with each other or with literals: it
  is read as ``CASE WHEN p THEN 1 ELSE 0 END``. A comparison that may be NULL is left alone, because the average
  skips its NULL rows and ``COUNT(*)`` does not;
* a column that the select's one derived table outputs as such a flag (``CASE WHEN p THEN 1 ELSE 0 END AS f``).

A single-arm flag ``CASE WHEN p THEN 100 ELSE 0 END`` is written ``CASE WHEN p THEN 1 ELSE 0 END`` with the 100 outside
the quotient. Every other dialect is left alone (PostgreSQL and DuckDB average to other types), and so is an average
outside a ``ROUND``, an average of a quotient or of a column, and ``r`` above 4 (an average of whole numbers declares
four decimals, and ``ROUND`` gives no more than its argument declares).

A pre-pass of the tenth hardest-cases batch (``shape_fallback.TENTH_BATCH``), run before the sums of ones are counted.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from .conditional_count_sums import _own, _own_aggregate
from .counted_sum_ratios import _flag_test, _never_null_comparison
from .rounded_percentages import MAX_PLACES, MAX_SHIFT, _power_of_ten, _unparen, _whole_literal


def _whole_power(node: exp.Expression | None) -> int | None:
    """``j`` for a whole-number literal ``10^j`` (``100``, not ``100.0``: the flag stays a whole number)."""

    node = _unparen(node)
    if isinstance(node, exp.Literal) and not node.is_string and node.this.isdigit():
        return _power_of_ten(node)
    return None


def _flag(node: exp.Expression | None) -> tuple[exp.Expression, int] | None:
    """``(f, k)`` when ``node`` is ``f * 10^k`` for a never-NULL whole-number flag ``f`` written as ``CASE`` or ``IF``."""

    node = _unparen(node)
    if isinstance(node, exp.Mul):
        for inner, factor in ((node.this, node.expression), (node.expression, node.this)):
            k = _whole_power(factor)
            found = _flag(inner) if k is not None else None
            if found is not None:
                return found[0], found[1] + k
        return None
    if isinstance(node, exp.Case):
        arms, default = node.args.get("ifs") or [], node.args.get("default")
        if node.args.get("this") is not None or not arms or not _whole_literal(default) or not all(_whole_literal(arm.args.get("true")) for arm in arms):
            return None
        if len(arms) == 1 and _unparen(default).this == "0":
            k = _whole_power(arms[0].args.get("true"))
            if k:  # CASE WHEN p THEN 100 ELSE 0 END: the flag of p, a hundred times
                return exp.Case(ifs=[exp.If(this=arms[0].this.copy(), true=exp.Literal.number(1))], default=exp.Literal.number(0)), k
        return node.copy(), 0
    if type(node) is exp.If and not isinstance(node.parent, exp.Case):
        if _whole_literal(node.args.get("true")) and _whole_literal(node.args.get("false")):
            return _flag(exp.Case(ifs=[exp.If(this=node.this.copy(), true=node.args["true"].copy())], default=node.args["false"].copy()))
    return None


def _rounded_shift(call: exp.Avg) -> tuple[exp.Expression, int] | None:
    """``(top, j + r)`` when ``call`` stands under ``ROUND(.., r)`` behind parentheses and powers of ten only; ``top`` is
    the node to replace (the average itself)."""

    node: exp.Expression = call
    shift = 0
    while True:
        parent = node.parent
        if isinstance(parent, exp.Paren):
            node = parent
        elif isinstance(parent, exp.Mul):
            other = parent.expression if parent.this is node else parent.this
            j = _power_of_ten(other)
            if j is None:
                return None
            shift += j
            node = parent
        else:
            break
    parent = node.parent
    if not isinstance(parent, exp.Round) or parent.this is not node or any(v for key, v in parent.args.items() if key not in ("this", "decimals")):
        return None
    digits = _unparen(parent.args.get("decimals"))
    if digits is None:
        return call, shift
    if not (isinstance(digits, exp.Literal) and not digits.is_string and digits.this.isdigit()) or int(digits.this) > MAX_PLACES:
        return None
    return call, shift + int(digits.this)


def write_flag_averages(tree: exp.Expression, not_null: dict | None, dialect: str) -> tuple[exp.Expression, bool]:
    """``tree`` with each rounded average of a flag written as a sum over ``COUNT(*)`` (module doc); whether it changed."""

    if dialect != "mysql" or not any(True for _ in tree.find_all(exp.Avg)):
        return tree, False
    tree = tree.copy()
    changed = False
    for call in list(tree.find_all(exp.Avg)):
        if any(v for key, v in call.args.items() if key != "this") or isinstance(call.parent, (exp.Window, exp.Filter)):
            continue
        select = call.find_ancestor(exp.Select)
        if select is None or not _own(call, select):
            continue
        placed = _rounded_shift(call)
        if placed is None:
            continue
        argument = _unparen(call.this)
        probe = exp.Sum(this=argument.copy())
        found = _flag(argument)
        if found is None and _never_null_comparison(probe, select, not_null) is not None:
            found = exp.Case(ifs=[exp.If(this=argument.copy(), true=exp.Literal.number(1))], default=exp.Literal.number(0)), 0
        if found is None and _flag_test(probe, select) is not None:
            found = argument.copy(), 0
        if found is None:
            continue
        flag, k = found
        if placed[1] + k > MAX_SHIFT or not _own_aggregate(flag, select, select):
            continue
        quotient = sqlglot.parse_one("SELECT SUM(x) / COUNT(*)", read="mysql").expressions[0]  # the dialect's own division node
        quotient.this.set("this", flag)
        written: exp.Expression = exp.Paren(this=quotient)
        if k:
            written = exp.Paren(this=exp.Mul(this=written, expression=exp.Literal.number(10**k)))
        call.replace(written)
        changed = True
    return tree, changed
