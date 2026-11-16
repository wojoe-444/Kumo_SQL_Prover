"""A regrouping that carries ``ORDER BY`` / ``LIMIT`` is folded like one that does not.

Calcite writes ``COUNT(DISTINCT x)`` as two groupings (``AggregateExpandDistinctAggregatesRule``):

    SELECT t.a, COUNT(t.x) AS n
    FROM (SELECT a, x FROM s WHERE .. GROUP BY a, x) AS t
    GROUP BY t.a
    ORDER BY n DESC, t.a LIMIT 10

and the normalizer folds a regrouping back into one grouped select
(``_regroup_distinct``, ``_roll_up_aggregate``, ``_collapse_aggregate`` in ``algebraic_equivalence``):

    SELECT a, COUNT(DISTINCT x) AS n FROM s WHERE .. GROUP BY a

Those rules decline a select that sorts or cuts its rows, because they rebuild the select and would lose the tail.
A report query nearly always has one (TPC-H 16 and its rewrite differ in exactly this), so the two groupings
stayed and the pair was compared as a regrouping against a plain grouped select.

``regroup_under_tail`` does the same folding with the tail kept:

1. the tail is moved onto a select around the regrouping (``tailed_grouping_sets.isolate_tail``: the derived
   table has the regrouping's rows one for one, and each ``ORDER BY`` key is read as an output column of it, a
   key that is no output being added as a hidden column);
2. one of the folding rules rewrites the regrouping, now without a tail, as it always has;
3. the tail goes back onto the folded select: every sort key was an output column of the regrouping, so it is
   the value of the output of the same name in the folded select, and the hidden columns are dropped again.

Step 3 needs the folded select to be a plain grouped select with the same output names in the same order, which
is checked; anything else is declined. Nothing is rewritten unless a folding rule fires in step 2, so a select
those rules do not read keeps its shape.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from sqlglot import exp

from .tailed_grouping_sets import _ALIAS, _HIDDEN, _TAIL, isolate_tail

_PLAIN = frozenset({"expressions", "from_", "from", "joins", "where", "group", "having"})


def _from(select: exp.Select):
    return select.args.get("from_") or select.args.get("from")


def regroup_under_tail(select: exp.Select, rules: Iterable[Callable[[exp.Select], exp.Expression | None]]) -> exp.Expression | None:
    if not any(select.args.get(key) for key in _TAIL) or select.args.get("group") is None:
        return None
    from_ = _from(select)
    if from_ is None or select.args.get("joins") or not isinstance(from_.this, exp.Subquery) or not isinstance(from_.this.this, exp.Select):
        return None
    if from_.this.this.args.get("group") is None:
        return None
    isolated = isolate_tail(select)
    if isolated is None:
        return None
    body = _from(isolated).this.this
    names = [item.alias_or_name.lower() for item in body.expressions]
    folded = None
    for rule in rules:
        folded = rule(body)
        if folded is not None:
            break
    if not isinstance(folded, exp.Select) or any(value for key, value in folded.args.items() if key not in _PLAIN):
        return None
    if folded.find(exp.Window) is not None or [item.alias_or_name.lower() for item in folded.expressions] != names:
        return None
    values = {name: (item.this if isinstance(item, exp.Alias) else item) for name, item in zip(names, folded.expressions)}
    result = folded.copy()
    result.set("expressions", [item.copy() for name, item in zip(names, folded.expressions) if not name.startswith(_HIDDEN)])
    order = isolated.args.get("order")
    if order is not None:
        for item in order.expressions:
            key = item.this
            if isinstance(key, exp.Literal):
                continue
            if not (isinstance(key, exp.Column) and key.table.lower() == _ALIAS and key.name.lower() in values):
                return None
            item.set("this", values[key.name.lower()].copy())
    for key in _TAIL:
        result.set(key, isolated.args.get(key))
    return result
