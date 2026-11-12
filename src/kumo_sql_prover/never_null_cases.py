"""``COALESCE`` around a ``CASE`` that is never NULL is the ``CASE``.

    COALESCE(CASE WHEN v.kind = 2 THEN 1 ELSE 0 END, 0)    is    CASE WHEN v.kind = 2 THEN 1 ELSE 0 END

A ``CASE`` returns the value of one of its ``THEN`` branches or of its ``ELSE``. When each of those is a literal that
is not NULL, the ``CASE`` is not NULL on any row, whatever its conditions are (a condition that is NULL is not
taken, and the ``ELSE`` answers). ``COALESCE`` returns its first argument when that is not NULL, so the other
arguments are never read. The identity is about one expression on one row: NULLs in the tested columns, duplicate
rows and empty inputs do not come into it.

It matters under an aggregate: ``conditional_count_sums`` reads ``SUM(CASE WHEN p THEN 1 ELSE 0 END)`` as a count,
and a query that wrote the sum over the padded ``CASE`` was left with a sum where the other query of the pair had
the count.

Dropped only when (the expression is left as written otherwise):

* the first argument of the ``COALESCE`` is a ``CASE`` (searched, or with an operand) that has an ``ELSE``;
* every ``THEN`` value and the ``ELSE`` value is an integer literal, or every one is a string literal
  (``CASE WHEN p THEN 1 END`` is NULL when ``p`` is not true, a column or an expression can be NULL);
* every other argument of the ``COALESCE`` is a literal of that same kind, so that dropping it cannot change the
  type of the result (``COALESCE(CASE .. THEN 1 ELSE 0 END, 0.5)`` is a decimal).
"""

from __future__ import annotations

from sqlglot import exp


def _kind(node: exp.Expression | None) -> str | None:
    if not isinstance(node, exp.Literal):
        return None
    if node.is_string:
        return "string"
    return "integer" if str(node.this).isdigit() else None


def _never_null_case(node: exp.Expression) -> str | None:
    """The kind of its values (``integer`` or ``string``) when ``node`` is a CASE the module docstring describes."""

    if type(node) is not exp.Case or any(key not in ("this", "ifs", "default") for key, value in node.args.items() if value):
        return None
    branches = node.args.get("ifs") or []
    if not branches or any(type(branch) is not exp.If for branch in branches):
        return None
    kinds = {_kind(branch.args.get("true")) for branch in branches} | {_kind(node.args.get("default"))}
    return kinds.pop() if len(kinds) == 1 and None not in kinds else None


def drop_never_null_padding(select: exp.Select) -> exp.Select | None:
    """``select`` with ``COALESCE(<never-NULL CASE>, ..)`` of its own read as the ``CASE``, or None."""

    if not isinstance(select, exp.Select):
        return None

    def padded(node: exp.Expression) -> bool:
        if type(node) is not exp.Coalesce or not node.expressions or any(value for key, value in node.args.items() if key not in ("this", "expressions")):
            return False
        kind = _never_null_case(node.this)
        return kind is not None and all(_kind(other) == kind for other in node.expressions)

    found = [index for index, node in enumerate(select.find_all(exp.Coalesce)) if node.find_ancestor(exp.Select) is select and padded(node)]
    if not found:
        return None
    result = select.copy()
    nodes = list(result.find_all(exp.Coalesce))
    for index in reversed(found):  # inner ones first: replacing an outer node would detach the inner ones
        nodes[index].replace(nodes[index].this.copy())
    return result
