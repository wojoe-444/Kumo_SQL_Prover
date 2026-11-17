"""Two rows compared with ``=`` or ``<>`` are their fields compared one by one.

    (a.lat, a.lon) = (b.lat, b.lon)        is        a.lat = b.lat AND a.lon = b.lon
    (a.lat, a.lon) <> (b.lat, b.lon)       is        a.lat <> b.lat OR a.lon <> b.lon

The solver has no rows, so a pair with such a comparison was declined ("unsupported: expression Tuple").
``tuple_membership`` reads a row tested against a list of rows the same way; this is the case of one row.

Why this is exact. The standard defines ``(a, b) = (x, y)`` as ``a = x AND b = y`` in three-valued logic: FALSE when
some field differs, otherwise NULL when some field is NULL, otherwise TRUE. ``<>`` is its negation, and the
negation of ``a = x AND b = y`` in three-valued logic is ``a <> x OR b <> y``. PostgreSQL and MySQL follow the
standard. DuckDB compares rows as structs, where a NULL field equals a NULL field, so the rule is not applied for
it, nor for a dialect not checked here.

Conditions, all checked: the dialect is PostgreSQL or MySQL; both sides are written as tuples of the same length,
two or more fields; every field is a column, a literal or a signed number (nothing volatile is written twice, and
a nested row or a subquery is left alone). Ordering comparisons of rows (``<``, ``>=``) are lexicographic and are
not read.
"""

from __future__ import annotations

from sqlglot import exp

from .tuple_membership import _DIALECTS, _field


def _fields(node: exp.Expression) -> tuple[list[exp.Expression], list[exp.Expression]] | None:
    if type(node) not in (exp.EQ, exp.NEQ):
        return None
    left, right = node.this, node.expression
    while type(left) is exp.Paren and type(left.this) is exp.Tuple:
        left = left.this
    while type(right) is exp.Paren and type(right.this) is exp.Tuple:
        right = right.this
    if type(left) is not exp.Tuple or type(right) is not exp.Tuple or len(left.expressions) < 2 or len(left.expressions) != len(right.expressions):
        return None
    if not all(_field(field) for row in (left, right) for field in row.expressions):
        return None
    return list(left.expressions), list(right.expressions)


def expand_row_equality(select: exp.Select, dialect: str | None) -> exp.Expression | None:
    """``select`` with every such comparison of two rows of its own written field by field, or ``None``."""

    if (dialect or "").lower() not in _DIALECTS:
        return None
    if not any(node.find_ancestor(exp.Select) is select and _fields(node) for node in select.find_all(exp.EQ, exp.NEQ)):
        return None
    copy = select.copy()
    for node in [n for n in copy.find_all(exp.EQ, exp.NEQ) if n.find_ancestor(exp.Select) is copy]:
        found = _fields(node)
        if found is None:
            continue
        test = None
        for left, right in zip(*found):
            one = type(node)(this=left.copy(), expression=right.copy())
            test = one if test is None else (exp.And if type(node) is exp.EQ else exp.Or)(this=test, expression=one)
        node.replace(exp.Paren(this=test))
    return copy
