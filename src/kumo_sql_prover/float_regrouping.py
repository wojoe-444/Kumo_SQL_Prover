"""Refuse a proof that adds FLOAT64 values in a different grouping than the query as written.

A FLOAT64 sum is not associative, so ``SUM(f)`` and ``SUM(s)`` over ``(SELECT SUM(f) AS s FROM t GROUP BY x)``
return different values on rows ``(1, 1e16), (1, 1), (2, 1), (2, 1)``: the left adds 1e16 and the three ones in
some order, the right adds the two partial sums. ``float_sum_order`` covers the *order* of one fixed set of
additions; a normalization rule that folds a sum of partial sums into one sum (or splits one) changes the
*grouping*, which that assumption does not cover. The algebraic prover compares the normalized queries, so
without this check it proved ``float-sum-pre-aggregated`` of the numeric-traps eval (labelled not equivalent).

The test is on the number of ``SUM`` and ``AVG`` calls: a rule that regroups additions adds or removes a call
(the partial sum), and the rules that only reorder or refilter keep the count. Applied to each side, as written
against as normalized, when some call of the written pair can add FLOAT64 values:

* with declared column types, a call counts as exact only when ``sum_argument_types.is_int64`` places its
  argument; if any other call exists and a side's number of calls changed, the proof is declined;
* without declared types nothing is known about the columns, so the proof is kept and carries the assumption
  :data:`UNTYPED_ASSUMPTION` (the old behaviour of every untyped corpus; a regrouped integer sum is exact).
"""

from __future__ import annotations

from sqlglot import exp

from . import sum_argument_types
from .window_calls import is_window_function

_DEPTH = 6

UNTYPED_ASSUMPTION = (
    "sums of partial sums are read as one sum (exact for integers; a FLOAT64 sum is not associative, and no column "
    "types were given to rule it out)"
)
DECLINED = "the rewrite regroups the additions of a SUM or AVG that may add FLOAT64 values (a FLOAT64 sum is not associative)"


def _calls(tree: exp.Expression) -> list[exp.Expression]:
    return list(tree.find_all(exp.Sum, exp.Avg))


def _lowered(types) -> dict:
    return {t.lower(): {c.lower(): v for c, v in columns.items()} for t, columns in (types or {}).items()}


def _unwrap(query: exp.Expression) -> exp.Expression:
    while isinstance(query, (exp.Subquery, exp.Paren)):
        query = query.this
    return query


def _branches(query: exp.Expression) -> list[exp.Select]:
    query = _unwrap(query)
    if isinstance(query, exp.SetOperation):
        return _branches(query.this) + _branches(query.expression)
    return [query] if isinstance(query, exp.Select) else []


def _own(select: exp.Select, node: exp.Expression, types) -> list[exp.Expression]:
    """The SUM and AVG calls of ``select`` under ``node`` that may add FLOAT64 values (a window's own call is not one)."""

    return [
        call for call in node.find_all(exp.Sum, exp.Avg)
        if call.find_ancestor(exp.Select) is select and not is_window_function(call) and not sum_argument_types.is_int64(call, types)
    ]


def _sources(select: exp.Select) -> dict[str, exp.Expression]:
    """The query behind each source of ``select`` that is a derived table or a ``WITH`` name, by alias."""

    found: dict[str, exp.Expression] = {}
    from_ = select.args.get("from_") or select.args.get("from")
    holders = ([from_.this] if from_ is not None else []) + [j.this for j in select.args.get("joins") or []]
    for source in holders:
        if isinstance(source, exp.Subquery):
            found[source.alias.lower() if source.alias else f"#{len(found)}"] = source.this
        elif isinstance(source, exp.Table) and not source.args.get("db"):
            try:
                body = sum_argument_types._cte_body(source.name.lower(), select)
            except KeyError:
                body = None
            if body is not None:
                found[source.alias_or_name.lower()] = body
    return found


def _item_sums(select: exp.Select, item: exp.Expression, types, depth: int) -> bool:
    """Whether output ``item`` of ``select`` is, or reads, a SUM or AVG that may add FLOAT64 values."""

    if _own(select, item, types):
        return True
    if depth <= 0:
        return False
    return any(_reads_sum(select, column, types, depth - 1) for column in item.find_all(exp.Column) if column.find_ancestor(exp.Select) is select)


def _reads_sum(select: exp.Select, column: exp.Column, types, depth: int) -> bool:
    """Whether ``column`` of ``select`` reads an output of a derived table or CTE that is such a sum."""

    sources = _sources(select)
    candidates = [sources[column.table.lower()]] if column.table and column.table.lower() in sources else list(sources.values()) if not column.table else []
    name = column.name.lower()
    for query in candidates:
        for branch in _branches(query):
            for item in branch.expressions:
                if (item.alias_or_name or "").lower() == name and _item_sums(branch, item, types, depth):
                    return True
    return False


def sum_of_sums(tree: exp.Expression, types) -> bool:
    """Whether some SUM or AVG of ``tree`` adds the output of another SUM or AVG that may add FLOAT64 values."""

    for call in tree.find_all(exp.Sum, exp.Avg):
        select = call.find_ancestor(exp.Select)
        if select is None:
            continue
        if any(c.find_ancestor(exp.Select) is select and _reads_sum(select, c, types, _DEPTH) for c in call.find_all(exp.Column)):
            return True
    return False


def regrouped(written: list[exp.Expression], normalized: list[exp.Expression], types) -> str | None:
    """``"declined"``, ``"untyped"`` or ``None`` for the two sides as written and as normalized (module doc).

    ``written`` and ``normalized`` hold one tree per side, in the same order."""

    if all(len(_calls(a)) == len(_calls(b)) for a, b in zip(written, normalized)):
        return None
    declared = _lowered(types)
    if not any(sum_of_sums(tree, declared) for tree in written):
        return None
    return "declined" if declared else "untyped"
