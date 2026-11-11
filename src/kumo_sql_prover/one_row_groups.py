"""Grouping a grouped derived table by all of its keys again leaves one row per group.

    SELECT d.k, MIN(d.s) AS s, MAX(d.m) AS m, COUNT(d.m) AS n FROM (SELECT k, SUM(x) AS s, MIN(y) AS m FROM t GROUP BY k) AS d GROUP BY d.k
    SELECT d.k, d.s AS s, d.m AS m, CASE WHEN NOT d.m IS NULL THEN 1 ELSE 0 END AS n FROM (SELECT k, SUM(x) AS s, MIN(y) AS m FROM t GROUP BY k) AS d

The derived table has one row per distinct tuple of its keys (NULL keys form one group). The outer select
groups those rows by columns that include every one of these keys, so two different rows never share an outer
group: each group is exactly one row. Over one row

* ``MIN(e)``, ``MAX(e)``, ``SUM(e)``, ``BIT_AND(e)``, ``BIT_OR(e)`` and ``ANY_VALUE(e)`` are ``e`` on that row (NULL
  when ``e`` is NULL, as the aggregate of no non-NULL value is), with or without ``DISTINCT``;
* ``COUNT(*)`` is 1 and ``COUNT(e)`` is 1 when ``e`` is not NULL, else 0;
* with ``FILTER (WHERE p)`` the row is aggregated only when ``p`` is TRUE: ``CASE WHEN p THEN e END`` for the
  first kind (no row, so NULL, otherwise) and ``CASE WHEN p THEN .. ELSE 0 END`` for the counts.

So the outer select is a projection of the derived table: the same rows, one per group, and none when the
derived table is empty. The projection is written into the derived table's own select list (``SELECT k, SUM(x)
AS s, MIN(y) AS m, CASE WHEN NOT MIN(y) IS NULL THEN 1 ELSE 0 END AS n FROM t GROUP BY k`` above), where each
output of the derived table is the expression that computes it.
The same holds when neither select has a ``GROUP BY`` and the derived table is a global aggregate: it has
exactly one row, and the outer aggregate over it returns one row.

The prover already read ``SUM`` of a ``SUM`` or ``COUNT``, ``MIN`` of a ``MIN`` and ``MAX`` of a ``MAX`` this way
(``algebraic_equivalence._collapse_aggregate``). Calcite's expansion of distinct aggregates writes ``MIN(s) FILTER
(WHERE g)`` over every aggregate of the derived table, and those forms were left grouped twice.

Conditions, all checked:

* the outer select reads one derived table and nothing else (no join, no nested query), has no ``WHERE``,
  ``HAVING``, ``DISTINCT``, ``QUALIFY``, ``ORDER BY``, ``LIMIT``, ``OFFSET`` or window, and a plain ``GROUP BY`` list of
  columns of the derived table;
* the derived table is a select with a plain ``GROUP BY`` (no ``ROLLUP``, ``CUBE``, ``GROUPING SETS``), no
  ``DISTINCT``, ``QUALIFY``, ``ORDER BY``, ``LIMIT``, ``OFFSET`` or window, and distinct output names; a ``HAVING``
  there only removes groups;
* every key of the derived table is one of its outputs (by its text) and the outer select groups by an output
  of each key. Grouping by fewer keys (a real roll-up) is left alone;
* or neither select has a ``GROUP BY``, the derived table aggregates and has no ``HAVING`` (which could leave it
  empty), and the outer select aggregates;
* every aggregate of the outer select is one of those listed above, over an expression without a subquery,
  window or aggregate; ``AVG`` (another type and scale), ``STRING_AGG``, ``ARRAY_AGG`` and the rest are left alone;
* an output without an alias that becomes a column or literal (``MAX(d.k)`` for a key ``k``, ``COUNT(*)``) is
  written in parentheses, ``(k)``, which the prover reads as an unnamed output like the aggregate it replaces;
* in the global case the result must still hold an aggregate (``SELECT COUNT(*) FROM (SELECT SUM(x) FROM t)`` is one
  row, ``SELECT 1 FROM t`` is not);
* integer ``SUM`` returns a wider type than its argument; the value is the same and, as elsewhere in the prover,
  the type of an output is not compared.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, extended_grouping

_BANNED = ("distinct", "qualify", "order", "limit", "offset", "windows", "with_", "with")
_VALUE = (exp.Min, exp.Max, exp.Sum, exp.BitwiseAndAgg, exp.BitwiseOrAgg, exp.AnyValue)


def _own(node: exp.Expression, select: exp.Select) -> bool:
    return node.find_ancestor(exp.Select) is select


def _plain_group(select: exp.Select) -> list[exp.Expression] | None:
    """The keys of a plain ``GROUP BY`` list ([] without one), None for any other grouping."""

    group = select.args.get("group")
    if group is None:
        return []
    if not group.expressions or extended_grouping(group) or any(group.args.get(k) for k in ("all", "totals")):
        return None
    if any(isinstance(key, (exp.Literal, exp.Tuple)) for key in group.expressions):
        return None
    return list(group.expressions)


def _over_one_row(call: exp.Expression) -> exp.Expression | None:
    """What the aggregate ``call`` (with its ``FILTER``, if any) returns over a group of one row."""

    condition = None
    if isinstance(call, exp.Filter):
        condition = call.expression.this if isinstance(call.expression, exp.Where) else None
        call = call.this
        if condition is None or condition.find(exp.AggFunc, exp.Window, exp.Query, exp.Subquery):
            return None
    if any(v for k, v in call.args.items() if k not in ("this", "big_int")):
        return None  # ORDER BY inside, a second argument, IGNORE NULLS ..
    argument = call.this
    if isinstance(argument, exp.Distinct):
        if len(argument.expressions) != 1 or argument.args.get("on"):
            return None
        argument = argument.expressions[0]
    if argument is None or (not isinstance(argument, exp.Star) and argument.find(exp.AggFunc, exp.Window, exp.Query, exp.Subquery, exp.Star)):
        return None
    if type(call) is exp.Count:
        if isinstance(argument, exp.Star):
            counted: exp.Expression = exp.Literal.number(1)
        else:
            counted = exp.Case(
                ifs=[exp.If(this=exp.Not(this=exp.Is(this=exp.Paren(this=argument.copy()), expression=exp.Null())), true=exp.Literal.number(1))],
                default=exp.Literal.number(0),
            )
        if condition is None:
            return counted
        return exp.Case(ifs=[exp.If(this=condition.copy(), true=counted)], default=exp.Literal.number(0))
    if type(call) not in _VALUE or isinstance(argument, exp.Star):
        return None
    if condition is None:
        return exp.Paren(this=argument.copy()) if not isinstance(argument, (exp.Column, exp.Literal)) else argument.copy()
    return exp.Case(ifs=[exp.If(this=condition.copy(), true=argument.copy())])


def collapse_one_row_groups(select: exp.Select) -> exp.Select | None:
    """``select`` without its ``GROUP BY`` when each of its groups is one row of a grouped derived table (module doc)."""

    if any(select.args.get(key) for key in _BANNED) or select.args.get("having") is not None or select.args.get("joins"):
        return None
    if select.args.get("where") is not None:
        return None
    from_ = select.args.get(FROM_KEY)
    source = from_.this if from_ is not None else None
    if not isinstance(source, exp.Subquery) or not source.alias or not isinstance(source.this, exp.Select):
        return None
    alias = source.args.get("alias")
    if isinstance(alias, exp.TableAlias) and alias.args.get("columns"):
        return None
    inner = source.this
    if any(inner.args.get(key) for key in _BANNED):
        return None
    if any(not _inside(q, source) for q in select.find_all(exp.Select) if q is not select):
        return None  # a nested query of the outer select could read the derived table
    if any(_own(w, select) for w in select.find_all(exp.Window)) or any(_own(w, inner) for w in inner.find_all(exp.Window)):
        return None
    outer_keys, inner_keys = _plain_group(select), _plain_group(inner)
    if outer_keys is None or inner_keys is None:
        return None
    outputs: dict[str, exp.Expression] = {}
    for item in inner.expressions:
        name = item.alias_or_name.lower()
        if not name or name in outputs or isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
            return None
        outputs[name] = item.this if isinstance(item, exp.Alias) else item
    calls = [
        a for a in select.find_all(exp.AggFunc)
        if _own(a, select) and not any(isinstance(p, exp.AggFunc) for p in _ancestors_within(a, select))
    ]
    if inner_keys:
        if not outer_keys:
            return None
        grouped = set()
        for key in outer_keys:
            if not isinstance(key, exp.Column) or (key.table and key.table.lower() != source.alias.lower()) or key.name.lower() not in outputs:
                return None
            grouped.add(outputs[key.name.lower()].sql())
        if any(key.sql() not in grouped for key in inner_keys):
            return None
    else:
        if outer_keys or inner.args.get("having") is not None or not calls:
            return None
        if not any(_own(a, inner) and a.find_ancestor(exp.Window) is None for a in inner.find_all(exp.AggFunc)):
            return None
    copy = select.copy()
    for call in [
        a for a in copy.find_all(exp.AggFunc)
        if _own(a, copy) and not any(isinstance(p, exp.AggFunc) for p in _ancestors_within(a, copy))
    ]:
        whole = call.parent if isinstance(call.parent, exp.Filter) else call
        value = _over_one_row(whole)
        if value is None:
            return None
        if not isinstance(value, (exp.Column, exp.Literal, exp.Paren)) and not isinstance(whole.parent, (exp.Alias, exp.Paren, exp.Select)):
            value = exp.Paren(this=value)
        whole.replace(value)
    if any(_own(a, copy) for a in copy.find_all(exp.AggFunc)):
        return None
    # the projection is written into the derived table's own select: each read of an output becomes its expression
    items: list[exp.Expression] = []
    for written, item in zip(select.expressions, copy.expressions):
        root = exp.Paren(this=(item.this if isinstance(item, exp.Alias) else item).copy())
        for column in list(root.find_all(exp.Column)):
            output = outputs.get(column.name.lower())
            if output is None or isinstance(column.this, exp.Star) or (column.table and column.table.lower() != source.alias.lower()):
                return None
            output = output.copy()
            operand = isinstance(column.parent, (exp.Binary, exp.Unary, exp.Between, exp.In)) and not isinstance(column.parent, exp.Paren)
            column.replace(exp.Paren(this=output) if operand and not isinstance(output, (exp.Column, exp.Literal, exp.Func)) else output)
        value = root.this
        if isinstance(item, exp.Alias):
            items.append(exp.alias_(value, item.alias))
        elif isinstance(written, exp.Column):
            items.append(value if isinstance(value, exp.Column) and value.name == written.name else exp.alias_(value, written.name))
        else:
            # an unnamed output stays unnamed: bare, a column or literal would give it that name
            items.append(exp.Paren(this=value) if isinstance(value, (exp.Column, exp.Literal)) else value)
    result = inner.copy()
    result.set("expressions", items)
    if not inner_keys and not any(_own(a, result) and a.find_ancestor(exp.Window) is None for item in items for a in item.find_all(exp.AggFunc)):
        return None  # without an aggregate the select would return a row per input row, not one
    return result


def _ancestors_within(node: exp.Expression, select: exp.Select) -> list[exp.Expression]:
    found = []
    parent = node.parent
    while parent is not None and parent is not select:
        found.append(parent)
        parent = parent.parent
    return found


def _inside(node: exp.Expression, ancestor: exp.Expression) -> bool:
    while node is not None:
        if node is ancestor:
            return True
        node = node.parent
    return False
