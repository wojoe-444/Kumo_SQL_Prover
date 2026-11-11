"""A table joined to itself to find values that occur on two rows is a grouping with ``COUNT(*) >= 2``.

    SELECT DISTINCT a.email
    FROM person AS a JOIN person AS b ON a.email = b.email AND a.id <> b.id

lists the emails that two different people share. "Two different people" is spelled ``a.id <> b.id``, which means
two different rows only when ``id`` tells rows apart: a declared one-column key that is NOT NULL. Then the query is

    SELECT a.email FROM person AS a WHERE a.email IS NOT NULL GROUP BY a.email HAVING COUNT(*) >= 2

the form the other common spelling (``GROUP BY email HAVING COUNT(email) > 1``) takes. Why it holds: a pair
``(a, b)`` passes the join exactly when both rows have the same non-NULL ``email`` and different ``id``; ``id`` is
unique and never NULL, so two rows have different ``id`` exactly when they are two rows. An email is therefore
listed exactly when at least two rows carry it. ``a.id < b.id`` and ``a.id > b.id`` say the same for this purpose:
of two different rows one has the larger key, and the columns listed are equal on both.

The rule fires on a select that

* is ``SELECT DISTINCT`` with no ``GROUP BY``, ``HAVING``, window, ``ORDER BY`` or ``LIMIT``;
* reads exactly two sources, both the same base table, joined by an inner or cross join;
* has, in its ``WHERE`` and ``ON`` together, nothing but equalities between the same column of the two rows
  (``a.c = b.c``, at least one) and comparisons ``<>``, ``<`` or ``>`` between the key column of the two rows
  (at least one, all on the same key column, which is not one of the equated columns);
* lists exactly the equated columns, each once, read from either row.

The key must be a declared one-column key whose column is declared NOT NULL and not floating-point (a NaN is
different from itself in some engines, so one row would pair with itself). Without declared keys nothing fires.
"""

from __future__ import annotations

from sqlglot import exp

from . import smt_values
from .grouped_source_filters import _inner_sources

_BANNED = ("group", "having", "qualify", "windows", "with_", "with", "limit", "offset", "order", "laterals", "pivots", "into", "locks", "sample")


def _plain_table(source: exp.Expression) -> str | None:
    if isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier) and not any(source.args.get(k) for k in ("db", "catalog", "pivots", "sample", "joins", "version", "when")):
        return source.name.lower()
    return None


def _column(node: exp.Expression, aliases: set[str]) -> tuple[str, str] | None:
    """``(alias, column)`` for a plain column of one of the two rows."""

    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Column) and isinstance(node.this, exp.Identifier) and node.table and node.table.lower() in aliases and not node.args.get("db"):
        return node.table.lower(), node.name.lower()
    return None


def regroup_duplicate_self_join(select: exp.Select, keys: dict[str, list[tuple]] | None, not_null: dict[str, frozenset[str]] | None, types: dict[str, dict[str, str]] | None = None) -> exp.Expression | None:
    from .algebraic_equivalence import _conjuncts

    distinct = select.args.get("distinct")
    if not keys or not not_null or distinct is None or distinct.args.get("on") or any(select.args.get(k) for k in _BANNED):
        return None
    if select.find(exp.Window) is not None or select.find(exp.AggFunc) is not None or select.find(exp.Subquery) is not None:
        return None
    sources = _inner_sources(select)
    if sources is None or len(sources) != 2:
        return None
    (first, left), (second, right) = sources.items()
    table = _plain_table(left)
    if table is None or _plain_table(right) != table:
        return None
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
    if not equated or len(apart) != 1:
        return None
    (key,) = apart
    declared = {c.lower() for c in {k.lower(): v for k, v in not_null.items()}.get(table, frozenset())}
    table_keys = {k.lower(): v for k, v in keys.items()}.get(table, [])
    if key in equated or key not in declared or not any(tuple(c.lower() for c in k) == (key,) for k in table_keys):
        return None
    if smt_values.type_class({k.lower(): v for k, v in (types or {}).items()}.get(table, {}).get(key)) == smt_values.FLOAT64:
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
    if set(listed) != set(equated):
        return None
    grouped = exp.select(*outputs).from_(left.copy())
    for column in equated:
        grouped = grouped.where(exp.Not(this=exp.Is(this=exp.column(column, table=row), expression=exp.Null())), copy=False)
    grouped = grouped.group_by(*[exp.column(column, table=row) for column in equated], copy=False)
    grouped = grouped.having(exp.GTE(this=exp.Count(this=exp.Star()), expression=exp.Literal.number(2)), copy=False)
    return grouped
