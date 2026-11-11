"""A row tested against a list of rows is the ``OR`` of the rows, each an ``AND`` of equalities.

    (a.name, b.name) IN (('X', 'Y'), ('Y', 'X'))

is

    (a.name = 'X' AND b.name = 'Y') OR (a.name = 'Y' AND b.name = 'X')

(TPC-H Q7 and the rewrites that fold its two nation pairs into one row ``IN``). The solver has no rows, so the
pair was declined ("unsupported: expression Tuple").

Why this is exact. The standard defines ``(a, b) = (x, y)`` as ``a = x AND b = y`` in three-valued logic (FALSE
when some field differs, otherwise NULL when some field is NULL), and ``r IN (r1, r2)`` as ``r = r1 OR r = r2``.
``NOT IN`` is the negation of that. PostgreSQL and MySQL follow it. DuckDB compares rows as structs, where a NULL
field equals a NULL field, so the rule is not applied for it, nor for a dialect not checked here.

Conditions, all checked:

* the dialect is PostgreSQL or MySQL;
* the tested row and every listed row are written as tuples of the same length, two or more fields, at least
  one listed row, and no subquery or ``UNNEST`` form of ``IN``;
* every field is a column, a literal or a signed number: nothing volatile is written twice.
"""

from __future__ import annotations

from sqlglot import exp

_DIALECTS = frozenset({"postgres", "mysql"})


def _field(node: exp.Expression) -> bool:
    if type(node) is exp.Neg:
        node = node.this
        return type(node) is exp.Literal and not node.is_string
    return type(node) in (exp.Column, exp.Literal) and not (type(node) is exp.Column and isinstance(node.this, exp.Star))


def _rows(node: exp.In) -> tuple[list[exp.Expression], list[list[exp.Expression]]] | None:
    if type(node) is not exp.In or any(value for key, value in node.args.items() if key not in ("this", "expressions")):
        return None
    tested, listed = node.this, node.expressions or []
    if type(tested) is not exp.Tuple or len(tested.expressions) < 2 or not listed:
        return None
    width = len(tested.expressions)
    if any(type(row) is not exp.Tuple or len(row.expressions) != width for row in listed):
        return None
    if not all(_field(field) for row in (tested, *listed) for field in row.expressions):
        return None
    return list(tested.expressions), [list(row.expressions) for row in listed]


def expand_row_membership(select: exp.Select, dialect: str | None) -> exp.Expression | None:
    """``select`` with every such row ``IN`` of its own written with ``AND`` and ``OR``, or ``None``."""

    if (dialect or "").lower() not in _DIALECTS:
        return None
    if not any(node.find_ancestor(exp.Select) is select and _rows(node) for node in select.find_all(exp.In)):
        return None
    copy = select.copy()
    for node in [n for n in copy.find_all(exp.In) if n.find_ancestor(exp.Select) is copy]:
        found = _rows(node)
        if found is None:
            continue
        tested, listed = found
        test = None
        for row in listed:
            both = None
            for left, right in zip(tested, row):
                equal = exp.EQ(this=left.copy(), expression=right.copy())
                both = equal if both is None else exp.And(this=both, expression=equal)
            both = exp.Paren(this=both)
            test = both if test is None else exp.Or(this=test, expression=both)
        node.replace(exp.Paren(this=test))
    return copy
