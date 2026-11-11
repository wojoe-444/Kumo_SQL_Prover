"""A ``CASE`` arm in the select list that the select's own filter rules out.

    SELECT k, CASE AVG(x) WHEN 0 THEN NULL ELSE SUM(y) / AVG(x) END AS r FROM t GROUP BY k HAVING AVG(x) <> 0
    SELECT k,                                   SUM(y) / AVG(x)     AS r FROM t GROUP BY k HAVING AVG(x) <> 0

Every row a select returns passed its ``WHERE`` (and every group its ``HAVING``). When a top-level ``AND`` part
of that filter can only be true where ``E = n`` is false, a select-list ``CASE`` that tests ``E = n`` never takes
that arm on a returned row, and the arm can go. The prover compares the select lists of two grouped selects
without their ``HAVING`` as a premise, so a guard one query kept and the other dropped stood between them.

Four spellings of the filter are read, ``E`` an expression and ``n``, ``L``, ``K`` numbers:

* ``E <> n`` (either order) or ``NOT E = n``: true only for a non-NULL ``E`` other than ``n``;
* ``CASE E WHEN n THEN L ELSE Y END > K`` (also ``CASE WHEN E = n THEN ..``; any of ``> >= < <= = <>``, either
  order) where ``L > K`` is false and ``Y`` is NULL whenever ``E`` is (``E`` sits in ``Y`` under ``+ - * /``
  alone). The comparison is not true with the ``L`` arm, so ``E = n`` is not true; it is then true of ``Y``, so
  ``Y`` is not NULL, so ``E`` is not NULL, so ``E = n`` is false;
* ``CASE WHEN E = n THEN FALSE ELSE Y > K END`` (the same with the comparison moved into the arms, as Calcite
  writes it; the arm ``FALSE`` or ``NULL``, ``E`` under ``+ - * /`` alone on either side of the comparison);
* ``COALESCE(Y, L) > K`` where ``L > K`` is false and ``Y`` holds ``NULLIF(E, n)`` under ``+ - * /`` alone:
  the comparison is then true of ``Y``, so ``NULLIF(E, n)`` is not NULL, so ``E`` is neither NULL nor ``n``.

What is rewritten: a ``CASE`` in the select's own select list, searched or simple, whose first ``WHEN`` tests
that same ``E = n`` (same text) and returns the bare ``NULL``; that ``WHEN`` is removed, and a ``CASE`` left
without one becomes its ``ELSE``. Only a ``NULL`` arm: an arm with a value also takes part in the type of the
``CASE`` (``THEN 0.0 ELSE i`` is a decimal; ``i`` alone divides as an integer).

Left alone:

* a ``CASE`` without ``ELSE``, under a window function, inside an aggregate's argument, in a subquery, or
  anywhere but the select list; a ``WHERE`` part in a select that aggregates without ``GROUP BY`` (it returns
  a row when nothing passed the filter); an ``E`` with a bare column named like one of the select's outputs;
* ``E`` holding anything but columns, numbers, ``+ - * /``, ``CAST`` and ``SUM``, ``AVG``, ``COUNT``, ``MIN``,
  ``MAX`` or the prover's own ``KUMOSQL_`` readings of an aggregate (a subquery or another function call
  need not return the same value twice);
* a filter part under ``OR``; a ``WHERE`` part when the ``CASE`` is not evaluated per row (it holds an
  aggregate); a select with ``ROLLUP``, ``CUBE`` or grouping sets; ``QUALIFY``.
"""

from __future__ import annotations

import re
from decimal import Decimal

from sqlglot import exp

from .ast_utils import extended_grouping

_STRICT = (exp.Paren, exp.Div, exp.Mul, exp.Add, exp.Sub, exp.Neg)
_PLAIN = (
    exp.Column, exp.Identifier, exp.Literal, exp.Paren, exp.Neg, exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Cast, exp.DataType,
    exp.DataTypeParam, exp.Sum, exp.Avg, exp.Count, exp.Min, exp.Max,
)
_COMPARE = {
    exp.GT: lambda a, b: a > b, exp.GTE: lambda a, b: a >= b, exp.LT: lambda a, b: a < b,
    exp.LTE: lambda a, b: a <= b, exp.EQ: lambda a, b: a == b, exp.NEQ: lambda a, b: a != b,
}
_FLIPPED = {exp.GT: exp.LT, exp.GTE: exp.LTE, exp.LT: exp.GT, exp.LTE: exp.GTE, exp.EQ: exp.EQ, exp.NEQ: exp.NEQ}


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _conjuncts(node: exp.Expression) -> list[exp.Expression]:
    node = _unwrap(node)
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _number(node: exp.Expression) -> Decimal | None:
    node = _unwrap(node)
    if isinstance(node, exp.Literal) and not node.is_string and re.fullmatch(r"\d+(\.\d+)?", node.name or ""):
        return Decimal(node.name)
    return None


def _plain(node: exp.Expression) -> bool:
    """``E``: columns, numbers, arithmetic, casts and the five plain aggregates (or the prover's reading of one)."""

    for part in node.walk():
        if isinstance(part, exp.Anonymous):
            if not part.name.upper().startswith("KUMOSQL_"):
                return False
        elif not isinstance(part, _PLAIN):
            return False
        if isinstance(part, exp.AggFunc) and (isinstance(part.parent, (exp.Window, exp.Filter)) or isinstance(part.this, exp.Distinct)):
            return False
        if isinstance(part, exp.Column) and isinstance(part.this, exp.Star):
            return False
    return True


def _test(case: exp.Case) -> tuple[exp.Expression, Decimal] | None:
    """``(E, n)`` when the first ``WHEN`` of ``case`` tests ``E = n``."""

    ifs = case.args.get("ifs") or []
    if not ifs:
        return None
    first = ifs[0]
    if case.this is not None:  # CASE E WHEN n THEN ..
        operand, value = case.this, _number(first.this)
    else:
        condition = _unwrap(first.this)
        if not isinstance(condition, exp.EQ):
            return None
        operand, value = condition.this, _number(condition.expression)
        if value is None:
            operand, value = condition.expression, _number(condition.this)
    operand = _unwrap(operand)
    if value is None or _number(operand) is not None or not _plain(operand):
        return None
    return operand, value


def _key(operand: exp.Expression, value: Decimal) -> tuple[str, Decimal]:
    return operand.sql(), value


def _strict_in(whole: exp.Expression, operand_text: str) -> bool:
    """Whether ``whole`` holds the operand under ``+ - * /`` alone, so that it is NULL when the operand is."""

    stack = [whole]
    while stack:
        node = stack.pop()
        if node.sql() == operand_text:
            return True
        if isinstance(node, _STRICT):
            stack.extend(child for child in (node.this, node.args.get("expression")) if isinstance(child, exp.Expression))
    return False


def _strict_nullif(whole: exp.Expression) -> tuple[str, Decimal] | None:
    """``(E, n)`` of a ``NULLIF(E, n)`` that ``whole`` holds under ``+ - * /`` alone."""

    stack = [whole]
    while stack:
        node = stack.pop()
        if isinstance(node, exp.Nullif):
            operand, value = _unwrap(node.this), _number(node.expression)
            if value is not None and _number(operand) is None and _plain(operand):
                return _key(operand, value)
        elif isinstance(node, _STRICT):
            stack.extend(child for child in (node.this, node.args.get("expression")) if isinstance(child, exp.Expression))
    return None


def _ruled_out(part: exp.Expression) -> tuple[str, Decimal] | None:
    """The ``(E, n)`` for which ``E = n`` is false wherever the filter part ``part`` is true."""

    part = _unwrap(part)
    if isinstance(part, exp.Not):
        inner = _unwrap(part.this)
        if isinstance(inner, exp.EQ):
            part = exp.NEQ(this=inner.this, expression=inner.expression)
    if isinstance(part, exp.NEQ):
        for operand, other in ((part.this, part.expression), (part.expression, part.this)):
            value, operand = _number(other), _unwrap(operand)
            if value is not None and _number(operand) is None and _plain(operand):
                return _key(operand, value)
    if isinstance(part, exp.Case):
        # CASE WHEN E = n THEN FALSE ELSE Y > K END: the comparison moved into the arms
        ifs, default = part.args.get("ifs") or [], part.args.get("default")
        test = _test(part)
        if len(ifs) != 1 or default is None or test is None:
            return None
        arm, default = _unwrap(ifs[0].args["true"]), _unwrap(default)
        refused = isinstance(arm, exp.Null) or (isinstance(arm, exp.Boolean) and not arm.this)
        if not refused or type(default) not in _COMPARE:
            return None
        operand, value = test
        if any(_strict_in(_unwrap(side), operand.sql()) for side in (default.this, default.expression)):
            return _key(operand, value)
        return None
    if type(part) not in _COMPARE:
        return None
    for case, other, kind in ((part.this, part.expression, type(part)), (part.expression, part.this, _FLIPPED[type(part)])):
        case, bound = _unwrap(case), _number(other)
        if isinstance(case, exp.Coalesce) and bound is not None and len(case.expressions) == 1:
            # COALESCE(Y, L) > K with L > K false: Y is not NULL, so neither is the NULLIF(E, n) it holds
            fallback = _number(case.expressions[0])
            if fallback is not None and not _COMPARE[kind](fallback, bound):
                found = _strict_nullif(_unwrap(case.this))
                if found is not None:
                    return found
            continue
        if not isinstance(case, exp.Case) or bound is None or len(case.args.get("ifs") or []) != 1 or case.args.get("default") is None:
            continue
        test = _test(case)
        arm = _number(case.args["ifs"][0].args["true"])
        if test is None or arm is None or _COMPARE[kind](arm, bound):
            continue
        operand, value = test
        if _strict_in(_unwrap(case.args["default"]), operand.sql()):
            return _key(operand, value)
    return None


def _has_aggregate(node: exp.Expression, select: exp.Select) -> bool:
    return any(a.find_ancestor(exp.Select) is select for a in node.find_all(exp.AggFunc))


def key_under_aggregate(case: exp.Case, select: exp.Select) -> bool:
    """Whether ``case`` is an argument of one of the select's own aggregates (read per row, before grouping)."""

    node = case.parent
    while node is not None and node is not select:
        if isinstance(node, exp.AggFunc):
            return True
        node = node.parent
    return False


def drop_refuted_case_arm(select: exp.Select) -> exp.Expression | None:
    """Remove the ``WHEN E = n THEN NULL`` arms of the select list that the select's filter rules out."""

    where, having = select.args.get("where"), select.args.get("having")
    if (where is None and having is None) or select.args.get("qualify"):
        return None
    group = select.args.get("group")
    if group is not None and extended_grouping(group):
        return None
    cases = [
        case
        for item in select.expressions
        for case in item.find_all(exp.Case)
        if case.find_ancestor(exp.Select) is select and case.find_ancestor(exp.Window) is None
    ]
    if not cases:
        return None
    per_row = {key for key in (_ruled_out(part) for part in (_conjuncts(where.this) if where is not None else [])) if key}
    per_group = {key for key in (_ruled_out(part) for part in (_conjuncts(having.this) if having is not None else [])) if key}
    if group is None and any(a.find_ancestor(exp.Select) is select for a in select.find_all(exp.AggFunc) if not isinstance(a.parent, exp.Window)):
        per_row = set()  # an aggregate without GROUP BY returns a row for no input: nothing passed the WHERE
    if not per_row and not per_group:
        return None
    aliases = {item.alias.lower() for item in select.expressions if isinstance(item, exp.Alias)}
    changed = False
    for case in cases:
        if case.args.get("default") is None:
            continue
        test = _test(case)
        if test is None:
            continue
        first = case.args["ifs"][0]
        if not isinstance(_unwrap(first.args["true"]), exp.Null):
            continue
        if any(not column.table and column.name.lower() in aliases for column in test[0].find_all(exp.Column)):
            continue  # a bare name that is also an output name: some dialects read the output in the filter
        if key_under_aggregate(case, select):
            continue
        key = _key(*test)
        # a WHERE part speaks of each row: the CASE must be evaluated per row too (no aggregate of this select in it)
        if not (key in per_group or (key in per_row and not _has_aggregate(case, select))):
            continue
        rest = case.args["ifs"][1:]
        if rest:
            case.set("ifs", rest)
        else:
            value = _unwrap(case.args["default"]).copy()
            whole_item = isinstance(case.parent, exp.Alias) or (case.parent is select and case.arg_key == "expressions")
            # the item alone needs no parentheses, and with them it would not print like the same item written plainly
            case.replace(value if whole_item or isinstance(case.parent, exp.Paren) else exp.Paren(this=value))
        changed = True
    return select if changed else None
