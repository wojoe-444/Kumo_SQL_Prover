"""Which aggregate calls of a select are a window's own function, and which are real aggregates.

``SUM(x) OVER (..)`` is a window function: it reads the rows and returns one value per row, so it does not make its
select an aggregate select. ``SUM(COUNT(*)) OVER ()`` is different: the ``COUNT(*)`` inside it is an ordinary aggregate
call that collapses the select to one row (or one row per group) before the window runs, and so is an aggregate in the
``PARTITION BY`` or ``ORDER BY`` of a window (``ROW_NUMBER() OVER (ORDER BY MAX(id))``).

Testing ``call.find_ancestor(exp.Window) is None`` to mean "not a window function" reads every aggregate under a window
as a window function, and a rule that then treats the select as row-per-row turns one row into one row per input row.
Use these helpers instead: only the call that *is* the window's function (also under ``FILTER``, ``IGNORE NULLS``
and ``RESPECT NULLS``) is skipped.
"""

from __future__ import annotations

from sqlglot import exp


def is_window_function(call: exp.Expression) -> bool:
    """Whether ``call`` is the function a window computes (``SUM`` of ``SUM(x) OVER (..)``), not a call inside one."""

    node = call
    while isinstance(node.parent, (exp.Filter, exp.IgnoreNulls, exp.RespectNulls)) and node.arg_key == "this":
        node = node.parent
    return isinstance(node.parent, exp.Window) and node.arg_key == "this"


def is_plain_aggregate(call: exp.Expression) -> bool:
    """Whether the aggregate ``call`` is an ordinary aggregate of its select (not a window's own function)."""

    return isinstance(call, exp.AggFunc) and not is_window_function(call)


def own_aggregates(select: exp.Select, node: exp.Expression | None = None) -> list[exp.AggFunc]:
    """The ordinary aggregate calls of ``select`` found under ``node`` (default: the whole select), at its own level."""

    return [
        call
        for call in (node if node is not None else select).find_all(exp.AggFunc)
        if call.find_ancestor(exp.Select) is select and not is_window_function(call)
    ]
