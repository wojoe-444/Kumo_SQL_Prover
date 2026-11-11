"""``LEFT JOIN (SELECT <expressions>) AS t ON TRUE`` is a cross join.

A select with no ``FROM`` returns exactly one row, whatever its expressions evaluate to, so every row on the left
finds that one partner under ``ON TRUE`` and the left join never pads::

    SELECT e.id, t.c0 FROM e LEFT JOIN (SELECT (SELECT MAX(d.x) FROM d) AS c0) AS t ON TRUE
    SELECT e.id, t.c0 FROM e CROSS JOIN (SELECT (SELECT MAX(d.x) FROM d) AS c0) AS t

Calcite's ``SubQueryRemoveRule`` writes an uncorrelated scalar subquery this way (a left join on ``true`` to the
one-row aggregate that holds its value). The cross join is the form the rules for constant sources already read.

Declined: any clause on the joined select other than its list (``WHERE`` or ``HAVING`` can remove the row,
``LIMIT 0`` too), a list with an aggregate, a window or a function that can produce rows, a lateral or otherwise
decorated source, a join with ``USING``, a hint or any condition other than the literal ``TRUE``.

This is a shape rule (see ``shape_fallback``).
"""

from __future__ import annotations

from sqlglot import exp

_ROW_MAKERS = (exp.UDTF, exp.Unnest, exp.Explode, exp.GenerateSeries, exp.Anonymous, exp.AggFunc, exp.Window)


def _one_row(source: exp.Expression) -> bool:
    if not isinstance(source, exp.Subquery) or any(v for k, v in source.args.items() if k not in ("this", "alias")):
        return False
    alias = source.args.get("alias")
    if alias is None or not alias.name or alias.args.get("columns"):
        return False
    inner = source.this
    if type(inner) is not exp.Select or any(v for k, v in inner.args.items() if k != "expressions") or not inner.expressions:
        return False
    nested = lambda n: isinstance(n, (exp.Query, exp.Subquery))  # noqa: E731
    for item in inner.expressions:
        if item.is_star:
            return False
        if any(isinstance(n, _ROW_MAKERS) for n in item.walk(prune=nested)):
            return False
    return True


def cross_join_one_row(select: exp.Select) -> exp.Expression | None:
    """``select`` with each ``LEFT JOIN <one-row select> ON TRUE`` written as a cross join, or ``None``."""

    found = []
    for index, join in enumerate(select.args.get("joins") or []):
        if (join.side or "").upper() != "LEFT" or (join.kind or "").upper() not in ("", "OUTER"):
            continue
        if any(v for k, v in join.args.items() if k not in ("this", "side", "kind", "on")):
            continue
        on = join.args.get("on")
        while isinstance(on, exp.Paren):
            on = on.this
        if not isinstance(on, exp.Boolean) or on.this is not True:
            continue
        if _one_row(join.this):
            found.append(index)
    if not found:
        return None
    copy = select.copy()
    for index in found:
        join = copy.args["joins"][index]
        join.set("side", None)
        join.set("on", None)
        join.set("kind", "CROSS")
    return copy
