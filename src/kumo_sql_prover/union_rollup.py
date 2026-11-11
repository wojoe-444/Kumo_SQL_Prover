"""A global aggregate over a ``UNION ALL`` of grouped selects reads each branch's totals.

Calcite's ``AggregateUnionTransposeRule`` pushes an aggregate below a ``UNION ALL``: every branch is grouped first
and the query above combines the partial results. For the empty grouping set of a ``ROLLUP`` the query above has
no ``GROUP BY`` at all, and its branches are still grouped by the full key::

    SELECT COALESCE(SUM(u.n), 0), SUM(u.s), MIN(u.k)
    FROM (SELECT t.k, COUNT(*) AS n, SUM(t.x) AS s FROM t GROUP BY t.k, t.j
          UNION ALL
          SELECT t.k, COUNT(*) AS n, SUM(t.x) AS s FROM t GROUP BY t.k, t.j) AS u

The groups of a branch partition its rows, so a combining aggregate over all groups of all branches is the same
aggregate over one total per branch::

    SELECT SUM(u.r0), SUM(u.r1), MIN(u.r2)
    FROM (SELECT COUNT(*) AS r0, SUM(t.x) AS r1, MIN(t.k) AS r2 FROM t
          UNION ALL
          SELECT COUNT(*) AS r0, SUM(t.x) AS r1, MIN(t.k) AS r2 FROM t) AS u

which is the form ``algebraic_equivalence._split_aggregates`` gives the same aggregate written above the union.
Per output:

* ``SUM`` of per-group ``SUM``, ``MIN`` of ``MIN`` and ``MAX`` of ``MAX``: NULLs are skipped at both levels, and
  the result is NULL exactly when no row has a value, with groups or with totals.
* ``MIN`` or ``MAX`` of a grouping key column: each group holds at least one row and every row of it has the
  group's key, so the extreme over groups is the extreme over rows.
* ``COALESCE(SUM(n), 0)`` of a per-group ``COUNT``: the sum of the counts is the count of all rows; with no group
  in any branch the sum is NULL, which the ``COALESCE`` reads as the 0 the totals add up to. A bare ``SUM(n)`` is
  **not** rewritten: it is NULL over no groups, where the totals give 0.

Each total is a global aggregate, so every branch contributes exactly one row and the query above still returns
its single row. Declined: any clause above the union other than the select list, a branch that is not a plain
grouped select (``HAVING``, ``DISTINCT``, a window, ``ROLLUP`` or ``GROUPING SETS``, which repeat rows across
groups), an output read with another aggregate, and ``DISTINCT`` aggregates at either level.

This is a shape rule (see ``shape_fallback``).
"""

from __future__ import annotations

from sqlglot import exp

_OUTER_ARGS = {"expressions", "from_", "from"}
_BRANCH_ARGS = {"expressions", "from_", "from", "joins", "where", "group"}
_GROUP_EXTRAS = ("rollup", "cube", "grouping_sets", "totals", "all")
_RESULT = "kumosql_r"


def _only(node: exp.Expression, allowed: set[str]) -> bool:
    return all(key in allowed for key, value in node.args.items() if value)


def _plain(agg: exp.Expression, kinds: tuple) -> bool:
    return isinstance(agg, kinds) and not agg.args.get("distinct") and not isinstance(agg.this, exp.Distinct) and not agg.expressions


def roll_up_grouped_union(select: exp.Select) -> exp.Expression | None:
    """The rewrite of the module docstring, or ``None`` when ``select`` is not that shape."""

    from .algebraic_equivalence import MAX_BRANCHES, _SPLIT_ALIAS, _is_constant, _union_all_branches

    if not _only(select, _OUTER_ARGS) or select.find(exp.Window):
        return None
    from_ = select.args.get("from_") or select.args.get("from")
    source = from_.this if from_ is not None else None
    if not isinstance(source, exp.Subquery) or not isinstance(source.this, exp.Union) or not source.alias:
        return None
    if not _only(source, {"this", "alias"}) or source.args["alias"].args.get("columns"):
        return None
    branches = _union_all_branches(source)
    if branches is None or not 2 <= len(branches) <= MAX_BRANCHES:
        return None
    names = [item.alias_or_name.lower() for item in branches[0].expressions]
    if "" in names or len(set(names)) != len(names):
        return None
    keys_of: list[set[str]] = []
    for branch in branches:
        group = branch.args.get("group")
        if not _only(branch, _BRANCH_ARGS) or group is None or not group.expressions or any(group.args.get(k) for k in _GROUP_EXTRAS):
            return None
        if len(branch.expressions) != len(names) or branch.find(exp.Window) or any(i.is_star for i in branch.expressions):
            return None
        # sqlglot lists ROLLUP, CUBE and GROUPING SETS among the keys: each puts a row in several groups
        if any(isinstance(key, exp.Literal) or key.find(exp.Rollup, exp.Cube, exp.GroupingSets, exp.AggFunc, exp.Select, exp.Subquery) for key in group.expressions):
            return None
        keys_of.append({key.sql() for key in group.expressions if isinstance(key, exp.Column)})

    alias = source.alias.lower()

    def read(column: exp.Expression) -> int | None:
        if not isinstance(column, exp.Column) or column.table and column.table.lower() != alias or column.name.lower() not in names:
            return None
        return names.index(column.name.lower())

    outer_items: list[exp.Expression] = []
    parts: list[list[exp.Expression]] = [[] for _ in branches]
    for item in select.expressions:
        expr = item.this if isinstance(item, exp.Alias) else item
        name = item.alias if isinstance(item, exp.Alias) else ""
        if _is_constant(expr):
            outer_items.append(item.copy())
            continue
        counted = (
            isinstance(expr, exp.Coalesce)
            and len(expr.expressions) == 1
            and isinstance(expr.expressions[0], exp.Literal)
            and not expr.expressions[0].is_string
            and expr.expressions[0].name == "0"
            and _plain(expr.this, (exp.Sum,))
        )
        agg = expr.this if counted else expr
        if not _plain(agg, (exp.Sum, exp.Min, exp.Max)):
            return None
        index = read(agg.this)
        if index is None:
            return None
        for n, branch in enumerate(branches):
            value = branch.expressions[index]
            value = value.this if isinstance(value, exp.Alias) else value
            if counted:
                if not _plain(value, (exp.Count,)):
                    return None
                part = value.copy()
            elif _plain(value, (type(agg),)):
                part = value.copy()
            elif isinstance(agg, (exp.Min, exp.Max)) and isinstance(value, exp.Column) and value.sql() in keys_of[n]:
                part = type(agg)(this=value.copy())
            else:
                return None
            parts[n].append(part)
        result = exp.column(f"{_RESULT}{len(parts[0]) - 1}", table=_SPLIT_ALIAS)
        combined = exp.Sum(this=result) if counted else type(agg)(this=result)
        outer_items.append(exp.alias_(combined, name) if name else combined)
    if not parts[0]:
        return None
    totals: list[exp.Select] = []
    for branch, branch_parts in zip(branches, parts):
        total = branch.copy()
        total.set("expressions", [exp.alias_(part, f"{_RESULT}{n}") for n, part in enumerate(branch_parts)])
        total.set("group", None)
        totals.append(total)
    union: exp.Expression = totals[0]
    for total in totals[1:]:
        union = exp.Union(this=union, expression=total, distinct=False)
    return exp.select(*outer_items).from_(exp.Subquery(this=union, alias=exp.TableAlias(this=exp.to_identifier(_SPLIT_ALIAS))))
