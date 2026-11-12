"""A strict comparison of integer arithmetic with a whole number, written non-strict.

``a.x - a.y > 30`` is ``a.x - a.y >= 31`` and ``a.x - a.y < 30`` is ``a.x - a.y <= 29`` when ``a.x`` and
``a.y`` are integer columns: the difference is a whole number, and nothing lies strictly between ``n`` and
``n + 1``. Where the expression is NULL both spellings are NULL.

A report that buckets a difference of two day numbers is written both ways, ``d > 30 AND d <= 60`` and
``d BETWEEN 31 AND 60``. The solver reads a column as a real number, for which the two differ at 30.5, so the
two queries did not compare. ``cast_rules`` already gives a ``COUNT`` this spelling; this rule gives it to
arithmetic over integers.

Read only when:

* the comparison is ``>`` or ``<`` between an expression and a whole-number literal (either order);
* the expression is ``+``, ``-``, ``*`` or integer division at the top, and ``cast_rules.expression_type`` types
  it as an integer from the declared column types: integer columns and literals, their sums, differences and
  products, a ``CAST`` to an integer type, ``COUNT``, ``SUM``/``MIN``/``MAX`` of integers, ``CASE`` and
  ``COALESCE`` of integers. ``/`` is not read (a fraction in most engines), and a column whose table or type
  is not known stops the rule;
* a column of a derived table of the select counts as the arithmetic it names
  (``(SELECT a.x - a.y AS d FROM a) AS s`` and ``s.d > 30``): the derived table is a plain ``SELECT``, the
  column is qualified with its alias or it is the select's only source, and the named expression passes the
  test above inside the derived table. A row an outer join fills with NULLs has NULL in both spellings;
* any other bare column or a ``COUNT`` call compared directly is left as written. For a column the solver keeps
  ``x > 50`` and ``x <= 50`` complementary, which a rewrite to ``x >= 51`` would lose; a ``COUNT`` is
  ``cast_rules``'s and ``count_column_bounds``'s.
"""

from __future__ import annotations

import re

from sqlglot import exp

from .cast_rules import expression_type

_LIMIT = 10**15
_ARITHMETIC = (exp.Add, exp.Sub, exp.Mul, exp.IntDiv)


def _unparen(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _int_literal(node: exp.Expression) -> int | None:
    node = _unparen(node)
    if isinstance(node, exp.Neg):
        inner = _int_literal(node.this)
        return None if inner is None else -inner
    if isinstance(node, exp.Literal) and not node.is_string and re.fullmatch(r"\d+", node.name or "") and int(node.name) < _LIMIT:
        return int(node.name)
    return None


def _number(value: int) -> exp.Expression:
    return exp.Neg(this=exp.Literal.number(-value)) if value < 0 else exp.Literal.number(value)


def _integer_arithmetic(inner: exp.Expression, select: exp.Select, types: dict, dialect: str) -> bool:
    if not isinstance(inner, _ARITHMETIC) or _int_literal(inner) is not None:
        return False
    if any(isinstance(n, (exp.Subquery, exp.Window, exp.Anonymous)) for n in inner.walk()):
        return False
    kind = expression_type(inner, select, types, 0, dialect)
    return kind is not None and kind[0] == "int"


def _named_arithmetic(column: exp.Column, select: exp.Select, types: dict, dialect: str) -> bool:
    """``column`` is an output of a derived table of ``select`` that names integer arithmetic."""

    source = select.args.get("from_") or select.args.get("from")
    relations = ([source.this] if source is not None else []) + [join.this for join in select.args.get("joins") or []]
    if column.table:
        relations = [r for r in relations if r.alias_or_name.lower() == column.table.lower()]
    if len(relations) != 1:
        return False
    derived = relations[0]
    alias = derived.args.get("alias")
    if not isinstance(derived, exp.Subquery) or not isinstance(derived.this, exp.Select) or (alias is not None and alias.columns):
        return False
    inner = derived.this
    if inner.args.get("with_") or inner.args.get("with") or any(isinstance(item, exp.Star) or item.is_star for item in inner.expressions):
        return False
    named = [item for item in inner.expressions if item.alias_or_name.lower() == column.name.lower()]
    if len(named) != 1 or not isinstance(named[0], exp.Alias):
        return False
    return _integer_arithmetic(_unparen(named[0].this), inner, types, dialect)


def integer_expression_bounds(select: exp.Select, types: dict | None, dialect: str = "postgres") -> exp.Expression | None:
    """Rewrite the select's own strict comparisons of integer arithmetic with a whole number, in place."""

    if not types:
        return None
    changed = False
    for node in list(select.find_all(exp.GT, exp.LT)):
        if node.find_ancestor(exp.Select) is not select:
            continue
        left, right = node.this, node.expression
        for value, other, literal_on_right in ((_int_literal(right), left, True), (_int_literal(left), right, False)):
            if value is None:
                continue
            inner = _unparen(other)
            if isinstance(inner, exp.Column):
                if not _named_arithmetic(inner, select, types, dialect):
                    break
            elif not _integer_arithmetic(inner, select, types, dialect):
                break
            greater = isinstance(node, exp.GT) == literal_on_right  # "other > n", else "other < n"
            bound = _number(value + 1 if greater else value - 1)
            if literal_on_right:
                node.replace((exp.GTE if greater else exp.LTE)(this=other.copy(), expression=bound))
            else:
                node.replace((exp.LTE if greater else exp.GTE)(this=bound, expression=other.copy()))
            changed = True
            break
    return select if changed else None
