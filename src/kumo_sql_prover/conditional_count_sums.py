"""A sum of ones over the rows that pass a condition is a count.

    SELECT k, SUM(CASE WHEN p THEN 1 ELSE 0 END) AS n FROM t GROUP BY k      -- COUNT(CASE WHEN p THEN 1 END)
    SELECT COALESCE(SUM(CASE WHEN p THEN 1 ELSE 0 END), 0) AS n FROM t         -- COUNT(CASE WHEN p THEN 1 END)
    SELECT COALESCE(SUM(CASE WHEN p THEN 1 END), 0) AS n FROM t                -- COUNT(CASE WHEN p THEN 1 END)

``CASE WHEN p THEN 1 ELSE 0 END`` is 1 on a row where ``p`` is TRUE and 0 where it is FALSE or NULL, never NULL.
Its ``SUM`` over a non-empty bag of rows is the number of rows on which ``p`` is TRUE, and ``COUNT(CASE WHEN p THEN 1
END)`` counts exactly those rows (the ``CASE`` is NULL on the others). Over no rows ``SUM`` is NULL and ``COUNT`` is 0,
so the bare ``SUM`` is read as the count only where every group has a row: under a plain ``GROUP BY`` with at least
one key. ``COALESCE(.., 0)`` around the ``SUM`` turns that NULL into the 0 ``COUNT`` returns, so that spelling is the
count everywhere, also for a global aggregate. Without the ``ELSE 0`` the ``SUM`` is NULL when no row passes (even
in a non-empty group), so only the ``COALESCE`` spelling is read.

``SUM(1)`` is the same sum with every row passing: ``COUNT(*)`` under a plain ``GROUP BY``, and ``COALESCE(SUM(1), 0)``
is ``COUNT(*)`` everywhere.

The count is the spelling the prover already uses for ``COUNT(*) FILTER (WHERE p)``, so the three meet there.

Conditions, all checked:

* the summand is the integer literal 1, or a searched ``CASE`` with one ``WHEN``, whose ``THEN`` is the integer
  literal 1 and whose ``ELSE`` is the integer literal 0 (or, inside ``COALESCE``, absent or NULL);
* the ``SUM`` has no ``DISTINCT``, no ``FILTER``, is not a window function and belongs to the select being read;
* for the bare ``SUM``: the select has a plain ``GROUP BY`` list with a key (not ``ROLLUP``, ``CUBE``, ``GROUPING SETS``
  or ``()``, whose grand-total group exists over no rows), and the sum is certainly an aggregate of that select: an
  aggregate whose columns all belong to an enclosing query is that query's aggregate, which may have no
  ``GROUP BY`` and then reads no rows as NULL. So inside a subquery expression (scalar, ``EXISTS``, ``IN``,
  ``LATERAL``) the condition must read a column qualified by one of the select's own sources, or no column at all;
* ``COALESCE`` has exactly the two arguments, the second the integer literal 0.

``SUM`` of integers and ``COUNT`` have the same type in PostgreSQL and BigQuery; elsewhere the value is the same
and, as in the rest of the prover, the type of an output is not compared.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import extended_grouping


def _literal(node: exp.Expression | None, value: str) -> bool:
    while isinstance(node, exp.Paren):
        node = node.this
    return isinstance(node, exp.Literal) and not node.is_string and node.name == value


def _condition(call: exp.Expression, need_else: bool) -> exp.Expression | None:
    """``p`` when ``call`` is ``SUM(CASE WHEN p THEN 1 ELSE 0 END)`` (the ``ELSE`` optional unless ``need_else``)."""

    if type(call) is not exp.Sum or any(v for k, v in call.args.items() if k != "this"):
        return None
    case = call.this
    while isinstance(case, exp.Paren):
        case = case.this
    if _literal(case, "1"):
        return exp.true()  # SUM(1): every row counts
    if not isinstance(case, exp.Case) or case.args.get("this") is not None or len(case.args.get("ifs") or []) != 1:
        return None
    arm, default = case.args["ifs"][0], case.args.get("default")
    if not _literal(arm.args.get("true"), "1"):
        return None
    if _literal(default, "0"):
        return arm.this
    if not need_else and (default is None or isinstance(default, exp.Null)):
        return arm.this
    return None


def _count(condition: exp.Expression) -> exp.Expression:
    if isinstance(condition, exp.Boolean) and condition.this is True and condition.parent is None:
        return exp.Count(this=exp.Star())
    return exp.Count(this=exp.Case(ifs=[exp.If(this=condition.copy(), true=exp.Literal.number(1))]))


def _own(node: exp.Expression, select: exp.Select) -> bool:
    return node.find_ancestor(exp.Select) is select and node.find_ancestor(exp.Window, exp.Filter) is None


def _sees_no_outer_row(select: exp.Select) -> bool:
    """Whether ``select`` is the query or a derived table under derived tables and set operations only."""

    node: exp.Expression = select
    while node.parent is not None:
        parent = node.parent
        if isinstance(parent, exp.SetOperation):
            node = parent
            continue
        if not isinstance(parent, exp.Subquery) or not isinstance(parent.parent, (exp.From, exp.Join)) or not isinstance(parent.parent.parent, exp.Select):
            return False  # a scalar, EXISTS or IN subquery, a LATERAL, a CTE
        node = parent.parent.parent
    return True


def _own_aggregate(condition: exp.Expression, select: exp.Select, placed: exp.Select) -> bool:
    """Whether a sum over ``condition`` written in ``select`` is certainly an aggregate of ``select`` itself.

    ``select`` is the (copied) select that holds ``condition``; ``placed`` is the same select where it stands in its query.
    """

    columns = list(condition.find_all(exp.Column))
    if not columns or _sees_no_outer_row(placed):
        return True
    source = select.args.get("from_") or select.args.get("from")
    items = ([source.this] if source is not None else []) + [join.this for join in select.args.get("joins") or []]
    names = {item.alias_or_name.lower() for item in items if isinstance(item, (exp.Table, exp.Subquery)) and item.alias_or_name}
    return any(column.table and column.table.lower() in names and column.find_ancestor(exp.Select) is select for column in columns)


def count_conditional_sums(select: exp.Select) -> exp.Select | None:
    """``select`` with its sums of ones (module doc) written as counts, or None."""

    if not any(True for _ in select.find_all(exp.Sum)):
        return None
    group = select.args.get("group")
    every_group_has_a_row = (
        group is not None
        and bool(group.expressions)
        and not extended_grouping(group)
        and not any(group.args.get(k) for k in ("all", "totals"))
    )
    copy = select.copy()
    changed = False
    for coalesce in list(copy.find_all(exp.Coalesce)):
        if not _own(coalesce, copy) or len(coalesce.expressions) != 1 or not _literal(coalesce.expressions[0], "0"):
            continue
        inner = coalesce.this
        while isinstance(inner, exp.Paren):
            inner = inner.this
        if isinstance(inner.parent, exp.Window) or not isinstance(inner, exp.Sum):
            continue
        condition = _condition(inner, need_else=False)
        if condition is not None:
            coalesce.replace(_count(condition))
            changed = True
    if every_group_has_a_row:
        for call in list(copy.find_all(exp.Sum)):
            if not _own(call, copy) or isinstance(call.parent, exp.Window):
                continue
            condition = _condition(call, need_else=True)
            if condition is not None and _own_aggregate(condition, copy, select):
                call.replace(_count(condition))
                changed = True
    return copy if changed else None
