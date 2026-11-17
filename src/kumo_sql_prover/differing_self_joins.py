"""A table joined to itself to find rows that agree on some columns and differ on another counts distinct values.

    SELECT DISTINCT a.viewer
    FROM views AS a JOIN views AS b ON a.viewer = b.viewer AND a.day = b.day AND a.article <> b.article

lists the viewers who saw two different articles on one day. The same question is commonly written

    SELECT DISTINCT viewer FROM views GROUP BY viewer, day HAVING COUNT(DISTINCT article) > 1

and that is the form this rule gives the join:

    SELECT DISTINCT a.viewer FROM views AS a
    WHERE a.viewer IS NOT NULL AND a.day IS NOT NULL
    GROUP BY a.viewer, a.day HAVING COUNT(DISTINCT a.article) >= 2

Why it holds, with no declared fact. A pair ``(a, b)`` passes the join exactly when both rows hold the same
values in the equated columns, none of them NULL (an equality with NULL is not TRUE), and ``a.article <> b.article``
is TRUE, which needs two articles that are not NULL and differ. Such a pair exists among the rows with one
combination of equated values exactly when those rows hold at least two different non-NULL articles, which is
``COUNT(DISTINCT article) >= 2`` over the group (``COUNT(DISTINCT ..)`` ignores NULLs). The select lists equated
columns only, read from either row (they are equal), and ``DISTINCT`` removes both the repeats the join makes and
the repeats of two groups that agree on the listed columns. ``a.x < b.x`` and ``a.x > b.x`` say the same as
``<>`` here: of two different values one is the smaller, and either row may be called ``a``. Both at once on one
pair (``a.x < b.x AND a.x > b.x``) is never TRUE, and such a select is left alone.

``duplicate_self_joins`` is the same idea for rows told apart by a declared key (``a.id <> b.id``, read as
``COUNT(*) >= 2``); it runs first and needs the key. This rule needs none, because it counts values, not rows.

The rule fires on a select that

* is ``SELECT DISTINCT`` with no ``GROUP BY``, ``HAVING``, window, aggregate, subquery, ``ORDER BY`` or ``LIMIT``;
* reads exactly two sources, both the same base table (or its distinct projection under the same column names,
  which is how the normalizer spells a table read under ``DISTINCT``), joined by an inner or cross join;
* has, in its ``WHERE`` and ``ON`` together, nothing but equalities between the same column of the two rows (at
  least one) and comparisons ``<>``, ``<`` or ``>`` between one other column of the two rows (at least one, all on
  that column);
* lists equated columns only, each at most once, read from either row.

Left alone: a column declared floating-point on either side of the reading (a NaN differs from itself in some
engines, so one row would pair with itself, and a NaN is its own group); a condition on one row alone; a third
source; an outer join.
"""

from __future__ import annotations

from sqlglot import exp

from . import smt_values
from .duplicate_self_joins import _BANNED, _column, _plain_table
from .grouped_source_filters import _inner_sources


def _projected(source: exp.Expression) -> tuple[exp.Table, set[str] | None] | None:
    """The base table behind a source and the columns it shows (None for all of them).

    A source is the table itself or its distinct projection under the same column names,
    ``(SELECT DISTINCT t.c AS c, .. FROM T AS t) AS a``, which is how the normalizer spells a table read under
    ``DISTINCT``. The reading counts values, so repeated rows change nothing.
    """

    if _plain_table(source) is not None:
        return source, None
    if not isinstance(source, exp.Subquery) or not source.alias or not isinstance(source.this, exp.Select) or source.args.get("pivots") or source.args.get("sample"):
        return None
    inner = source.this
    distinct = inner.args.get("distinct")
    table = (inner.args.get("from_") or inner.args.get("from"))
    table = table.this if table is not None else None
    if distinct is None or distinct.args.get("on") or _plain_table(table) is None or any(v for k, v in inner.args.items() if k not in ("expressions", "from", "from_", "distinct")):
        return None
    shown: set[str] = set()
    for item in inner.expressions:
        found = _column(item.unalias(), {table.alias_or_name.lower()})
        if found is None or (isinstance(item, exp.Alias) and item.alias.lower() != found[1]) or found[1] in shown:
            return None
        shown.add(found[1])
    return table, shown


def regroup_differing_self_join(select: exp.Select, types: dict[str, dict[str, str]] | None = None) -> exp.Expression | None:
    from .algebraic_equivalence import _conjuncts

    distinct = select.args.get("distinct")
    if distinct is None or distinct.args.get("on") or any(select.args.get(k) for k in _BANNED):
        return None
    sources = _inner_sources(select)
    if sources is None or len(sources) != 2:
        return None
    if any(isinstance(n, (exp.Window, exp.AggFunc, exp.Query, exp.Subquery)) and n is not select and all(n is not s and n.find_ancestor(exp.Subquery) is not s for s in sources.values()) for n in select.walk()):
        return None  # nothing but the two sources may hold a query
    (first, left), (second, right) = sources.items()
    one, other = _projected(left), _projected(right)
    if one is None or other is None or one[0].name != other[0].name or bool(one[0].this.args.get("quoted")) != bool(other[0].this.args.get("quoted")):
        return None
    table = one[0].name.lower()
    aliases = {first, second}
    conditions: list[exp.Expression] = []
    where = select.args.get("where")
    if where is not None:
        conditions.extend(_conjuncts(where.this))
    for join in select.args.get("joins") or []:
        if join.args.get("on") is not None:
            conditions.extend(_conjuncts(join.args["on"]))
    equated: list[str] = []
    apart: set[str] = set()
    orders: set[str] = set()  # which row holds the smaller value, for every ``<`` and ``>``
    for condition in conditions:
        while isinstance(condition, exp.Paren):
            condition = condition.this
        if not isinstance(condition, (exp.EQ, exp.NEQ, exp.LT, exp.GT)):
            return None
        a, b = _column(condition.this, aliases), _column(condition.expression, aliases)
        if a is None or b is None or a[0] == b[0] or a[1] != b[1]:
            return None
        if isinstance(condition, exp.EQ):
            if a[1] not in equated:
                equated.append(a[1])
        else:
            apart.add(a[1])
            if not isinstance(condition, exp.NEQ):
                orders.add(a[0] if isinstance(condition, exp.LT) else b[0])
    if not equated or len(apart) != 1 or len(orders) > 1:
        return None  # ``a.x < b.x AND a.x > b.x`` is never TRUE: no pair passes, whatever the groups hold
    (differing,) = apart
    if differing in equated:
        return None
    kinds = {k.lower(): v for k, v in {str(t).lower().split(".")[-1]: v for t, v in (types or {}).items()}.get(table, {}).items()}
    if any(smt_values.type_class(kinds.get(column)) == smt_values.FLOAT64 for column in [*equated, differing]):
        return None
    row = left.alias_or_name  # the one row that is kept
    outputs: list[exp.Expression] = []
    listed: list[str] = []
    for item in select.expressions:
        found = _column(item.unalias(), aliases)
        if found is None or found[1] not in equated or found[1] in listed:
            return None
        listed.append(found[1])
        value = exp.column(found[1], table=row)
        outputs.append(exp.alias_(value, item.alias, quoted=False) if isinstance(item, exp.Alias) else value)
    if not listed or any(shown is not None and not {*equated, differing} <= shown for _, shown in (one, other)):
        return None
    kept = exp.Table(this=one[0].this.copy(), alias=exp.TableAlias(this=exp.to_identifier(row)))
    grouped = exp.select(*outputs).from_(kept).distinct(copy=False)
    for column in equated:
        grouped = grouped.where(exp.Not(this=exp.Is(this=exp.column(column, table=row), expression=exp.Null())), copy=False)
    grouped = grouped.group_by(*[exp.column(column, table=row) for column in equated], copy=False)
    counted = exp.Count(this=exp.Distinct(expressions=[exp.column(differing, table=row)]))
    return grouped.having(exp.GTE(this=counted, expression=exp.Literal.number(2)), copy=False)
