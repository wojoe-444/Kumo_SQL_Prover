"""A count of an outer-joined table's column under a guard is the count with the guard in the join condition.

    SELECT c.k, COUNT(CASE WHEN o.id IS NOT NULL AND o.note <> 'x' THEN o.id END)
    FROM c LEFT JOIN o ON c.k = o.ck
    GROUP BY c.k

is

    SELECT c.k, COUNT(o.id)
    FROM c LEFT JOIN o ON c.k = o.ck AND o.note <> 'x'
    GROUP BY c.k

(TPC-H Q13 and the rewrites that move its filter on ``orders`` out of the join condition into the count).

Why this is exact. Take a group and a row ``r`` of the left side in it. The first form counts the rows ``o`` that
match ``r`` under the join condition, pass the guard and have a value in the counted column; the NULL-extended
row of an unmatched ``r`` has NULL in the counted column and counts nothing. The second form counts the rows ``o``
that match ``r`` under the join condition and the guard together and have a value in the counted column; when
none match, the NULL-extended row counts nothing. The two are the same number for every ``r``, so for every group,
whatever the select groups by, as long as nothing else can see which rows the join produced: every aggregate of
the select is such a count under the same guard, and nothing but those counts and the join condition reads the
outer-joined table.

A guard conjunct ``o.c IS NOT NULL`` on a column declared NOT NULL is TRUE on every row of ``o`` and is dropped
from the join condition.

Conditions, all checked:

* the select reads two sources, a table or derived table and a ``LEFT [OUTER] JOIN`` of a table the schema lists,
  with an ``ON`` condition; both have names, and every column of the select names its source;
* every aggregate of the select is ``COUNT(CASE WHEN g THEN o.x END)`` (a searched ``CASE`` with one ``WHEN``, no
  ``ELSE`` or ``ELSE NULL``, no ``DISTINCT``, ``FILTER`` or window) with ``o.x`` a column of the outer-joined table, and
  all of them have the same guard, by its text;
* outside those counts and the join condition nothing reads the outer-joined table: not ``WHERE``, ``GROUP BY``,
  ``HAVING``, ``ORDER BY`` or the select list;
* the guard holds no aggregate, window, nested query or volatile function; the select has no window or
  ``DISTINCT``, and no nested query of its own.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY
from .keyed_join_groups import _and_all, declared_columns
from .unmatched_outer_join import _columns_of, _conjuncts, _inside, _plain_table


def _guarded(call: exp.Expression, name: str, columns: set[str]) -> tuple[exp.Expression, exp.Column] | None:
    """``(guard, counted column)`` for ``COUNT(CASE WHEN guard THEN o.x END)``, or ``None``."""

    if type(call) is not exp.Count or isinstance(call.parent, (exp.Window, exp.Filter)):
        return None
    if any(value for key, value in call.args.items() if key not in ("this", "big_int")):
        return None
    case = call.this
    while type(case) is exp.Paren:
        case = case.this
    if type(case) is not exp.Case or case.args.get("this") is not None or len(case.args.get("ifs") or []) != 1:
        return None
    default = case.args.get("default")
    if default is not None and type(default) is not exp.Null:
        return None
    branch = case.args["ifs"][0]
    counted, guard = branch.args.get("true"), branch.this
    if type(counted) is not exp.Column or counted.table.lower() != name or counted.name.lower() not in columns or len(counted.parts) != 2:
        return None
    if any(isinstance(node, (exp.AggFunc, exp.Window, exp.Query, exp.Subquery)) for node in guard.walk()):
        return None
    return guard, counted


def guard_outer_counts(select: exp.Select, schema: dict | None, not_null: dict | None) -> exp.Expression | None:
    """``select`` with the one guard of its counts moved into its outer join's condition, or ``None``."""

    from .counted_membership import _unstable

    joins = select.args.get("joins") or []
    if not schema or len(joins) != 1 or select.args.get("distinct") or select.args.get("qualify") or select.args.get("windows"):
        return None
    join = joins[0]
    if (join.args.get("side") or "").upper() != "LEFT" or (join.args.get("kind") or "").upper() not in ("", "OUTER"):
        return None
    if any(value for key, value in join.args.items() if key not in ("this", "side", "kind", "on")) or join.args.get("on") is None:
        return None
    base = select.args.get(FROM_KEY).this if select.args.get(FROM_KEY) is not None else None
    table = join.this
    if type(table) is not exp.Table or not _plain_table(table) or type(base) not in (exp.Table, exp.Subquery):
        return None
    declared = declared_columns(schema)
    columns = _columns_of(table, declared)
    left, right = (base.alias_or_name or "").lower(), (table.alias_or_name or "").lower()
    if not columns or not left or not right or left == right:
        return None
    calls = [call for call in select.find_all(exp.AggFunc) if call.find_ancestor(exp.Select) is select]
    if not calls or _unstable(select):
        return None
    if any(n.find_ancestor(exp.Select) is select for n in select.find_all(exp.Window)):
        return None
    if any(not _inside(n, base) for n in select.find_all(exp.Select) if n is not select):
        return None
    guards = [_guarded(call, right, columns) for call in calls]
    if any(found is None for found in guards) or len({found[0].sql() for found in guards}) != 1:
        return None
    for column in select.find_all(exp.Column):
        if _inside(column, base):
            continue
        if isinstance(column.this, exp.Star) or len(column.parts) != 2 or column.table.lower() not in (left, right):
            return None
        if column.table.lower() == right and not _inside(column, join.args["on"]) and not any(_inside(column, call) for call in calls):
            return None
    if any(isinstance(n, exp.Star) and not isinstance(n.parent, exp.Count) for item in select.expressions for n in item.walk()):
        return None

    full = ".".join(part.name for part in table.parts).lower()
    known = {str(c).lower() for c in ((not_null or {}).get(full) or (not_null or {}).get(table.name.lower()) or ())}
    kept = []
    for part in _conjuncts(guards[0][0]):
        test = part
        while type(test) is exp.Paren:
            test = test.this
        tested = None  # the column of ``column IS NOT NULL``, in either of its two parsed forms
        if type(test) is exp.Not and type(test.this) is exp.Is and not test.this.args.get("negate"):
            tested = test.this
        elif type(test) is exp.Is and test.args.get("negate"):
            tested = test
        is_not_null = (
            tested is not None and type(tested.expression) is exp.Null and type(tested.this) is exp.Column
            and tested.this.table.lower() == right and tested.this.name.lower() in known
        )
        if not is_not_null:
            kept.append(part.copy())
    copy = select.copy()
    new_join = copy.args["joins"][0]
    for call in [c for c in copy.find_all(exp.AggFunc) if c.find_ancestor(exp.Select) is copy]:
        _, counted = _guarded(call, right, columns)
        call.set("this", counted.copy())
    if kept:
        new_join.set("on", _and_all([new_join.args["on"].copy(), *kept]))
    return copy
