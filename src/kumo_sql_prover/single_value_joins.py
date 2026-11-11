"""A joined one-row derived table that holds a single value is read as the scalar subquery it stands for.

    SELECT d.k FROM d JOIN (SELECT SINGLE_VALUE(t.m) AS f FROM t WHERE t.c = 5) AS s ON d.m = s.f
    SELECT d.k FROM d WHERE d.m = (SELECT t.m FROM t WHERE t.c = 5)

Calcite plans a scalar subquery as a join to a select that returns exactly one row: NULL when the subquery has
no row, its value when it has one, an error when it has more. Its own dialect writes ``SINGLE_VALUE(x)``; its
PostgreSQL output spells the same thing out:

    CASE COUNT(*) WHEN 0 THEN NULL WHEN 1 THEN MIN(x) ELSE (SELECT NULL UNION ALL SELECT NULL) END

(with one row ``MIN(x)`` is that row's ``x``; the ``ELSE`` is a scalar subquery of two rows, which is the
error). ``one_row_joins`` reads such a join when it is written ``ON TRUE`` and the filter does not read the
value; this rule reads the single-value case wherever the value is read, and with a join condition.

The derived table returns exactly one row on every database on which it returns at all, so an inner join to
it keeps each row of the other side once, or not at all when the condition fails: the condition is a filter.
A ``LEFT JOIN .. ON TRUE`` to it is the same cross join. Each read of the one output is the value of the
scalar subquery over the same ``FROM`` and ``WHERE``.

Read only when:

* the derived table is one ``SELECT`` of one output over ``FROM``, joins and ``WHERE`` alone, and reads
  nothing from outside itself (every column in it carries a source of its own);
* the output is ``SINGLE_VALUE(x)`` or the ``CASE`` above (``MIN`` or ``MAX``; the ``ELSE`` two ``SELECT``
  of one value joined by ``UNION ALL``, with nothing else on them), ``x`` free of aggregates, windows and
  subqueries. The ``CASE`` is not read for SQLite, where a scalar subquery of two rows is its first row;
* the join is plain, ``INNER`` or ``CROSS``, with or without a condition, in a select whose joins are all of
  those kinds; or ``LEFT`` with ``ON TRUE`` in a select with no ``RIGHT`` or ``FULL`` join;
* the alias names no other source of the select or of a subquery in it, has no column list, and every read of
  it is ``alias.output`` outside ``GROUP BY``, ``ORDER BY`` and window definitions; the select has no star
  and no bare column of the output's name.
"""

from __future__ import annotations

from sqlglot import exp

from .decorrelation_rules import _free_qualifiers

_BODY_ALLOWED = frozenset({"expressions", "from_", "from", "joins", "where", "kind", "hint"})
_INNER = {"", "INNER", "CROSS"}


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _plain_value(node: exp.Expression) -> bool:
    return not any(isinstance(n, (exp.AggFunc, exp.Window, exp.Subquery, exp.Exists, exp.Star)) for n in node.walk())


def _only(node: exp.Expression, *keys: str) -> bool:
    return not any(value for key, value in node.args.items() if key not in keys and value)


def _two_rows(node: exp.Expression) -> bool:
    """``(SELECT a UNION ALL SELECT b)``: a scalar subquery that always has two rows."""

    if not isinstance(node, exp.Subquery) or not _only(node, "this"):
        return False
    union = node.this
    if type(union) is not exp.Union or union.args.get("distinct") is not False or not _only(union, "this", "expression"):
        return False
    return all(isinstance(side, exp.Select) and _only(side, "expressions", "kind") and len(side.expressions) == 1 for side in (union.this, union.expression))


def _is_number(node: exp.Expression, text: str) -> bool:
    return isinstance(node, exp.Literal) and not node.is_string and node.name == text


def single_value_argument(value: exp.Expression, dialect: str | None = None) -> exp.Expression | None:
    """``x`` when ``value`` is ``SINGLE_VALUE(x)`` or Calcite's ``CASE COUNT(*)`` spelling of it."""

    value = _unwrap(value)
    if isinstance(value, exp.Anonymous) and value.name.upper() == "SINGLE_VALUE" and len(value.expressions) == 1:
        argument = value.expressions[0]
        return argument if _plain_value(argument) else None
    if not isinstance(value, exp.Case) or (dialect or "").lower() == "sqlite":
        return None
    counted, ifs, default = value.this, value.args.get("ifs") or [], value.args.get("default")
    if not isinstance(counted, exp.Count) or not isinstance(counted.this, exp.Star) or not _only(counted, "this", "big_int") or isinstance(counted.parent, (exp.Window, exp.Filter)):
        return None
    if len(ifs) != 2 or default is None or not _two_rows(_unwrap(default)):
        return None
    none, one = ifs
    if not _is_number(none.this, "0") or not isinstance(_unwrap(none.args["true"]), exp.Null) or not _is_number(one.this, "1"):
        return None
    extreme = _unwrap(one.args["true"])
    if type(extreme) not in (exp.Min, exp.Max) or not _only(extreme, "this") or not isinstance(extreme.this, exp.Expression):
        return None
    argument = extreme.this
    if isinstance(argument, exp.Distinct) or not _plain_value(argument):
        return None
    return argument


def _body_value(body: exp.Expression, dialect: str | None) -> exp.Expression | None:
    if not isinstance(body, exp.Select) or not _only(body, *_BODY_ALLOWED) or len(body.expressions) != 1:
        return None
    if (body.args.get("from_") or body.args.get("from")) is None:
        return None
    item = body.expressions[0]
    argument = single_value_argument(item.this if isinstance(item, exp.Alias) else item, dialect)
    if argument is None or not item.alias_or_name:
        return None
    if any(not column.table for column in body.find_all(exp.Column)) or _free_qualifiers(body):
        return None  # a bare or outer name could mean another source once the body sits in the filter
    return argument


def _under(node: exp.Expression, root: exp.Expression) -> bool:
    parent = node.parent
    while parent is not None:
        if parent is root:
            return True
        parent = parent.parent
    return False


def read_single_value_join(select: exp.Select, dialect: str | None = None) -> exp.Expression | None:
    """Replace one joined single-value derived table of ``select`` by scalar subqueries where it is read."""

    joins = select.args.get("joins") or []
    if not joins or (select.args.get("from_") or select.args.get("from")) is None:
        return None
    sides = [(j.args.get("side") or "").upper() for j in joins]
    kinds = [(j.args.get("kind") or "").upper() for j in joins]
    if any(j.args.get("using") or j.args.get("method") for j in joins) or any(side in ("RIGHT", "FULL") for side in sides):
        return None
    all_inner = all(not side and kind in _INNER for side, kind in zip(sides, kinds))
    for join, side, kind in zip(joins, sides, kinds):
        item = join.this
        if not isinstance(item, exp.Subquery) or not item.alias or not _only(item, "this", "alias") or item.args["alias"].args.get("columns"):
            continue
        on = join.args.get("on")
        on_true = on is None or (isinstance(on, exp.Boolean) and bool(on.this))
        if side == "LEFT":
            if kind not in ("", "OUTER") or not on_true:
                continue
        elif side or kind not in _INNER or not (on_true or all_inner):
            continue
        argument = _body_value(item.this, dialect)
        if argument is None:
            continue
        alias, name = item.alias.lower(), item.this.expressions[0].alias_or_name.lower()
        outside = [n for n in select.walk() if n is not item and not _under(n, item)]
        if any(isinstance(n, (exp.Table, exp.Subquery, exp.Lateral, exp.Unnest, exp.CTE)) and (n.alias_or_name or "").lower() == alias for n in outside):
            continue
        if any(isinstance(n, exp.Star) and not isinstance(n.parent, exp.Count) for n in outside):
            continue
        columns = [n for n in outside if isinstance(n, exp.Column)]
        if any(not c.table and c.name.lower() == name for c in columns):
            continue
        uses = [c for c in columns if c.table.lower() == alias]
        if any(c.name.lower() != name or c.find_ancestor(exp.Group, exp.Order, exp.Window) is not None for c in uses):
            continue
        probe = item.this.copy()
        probe.set("expressions", [argument.copy()])
        for column in uses:
            value: exp.Expression = exp.Subquery(this=probe.copy())
            if column.parent is select and column.arg_key == "expressions":
                value = exp.Alias(this=value, alias=exp.to_identifier(column.name))
            column.replace(value)
        select.set("joins", [j for j in joins if j is not join] or None)
        if not on_true:
            where = select.args.get("where")
            condition = on.copy() if not isinstance(on, exp.Or) else exp.Paren(this=on.copy())
            if where is None:
                select.set("where", exp.Where(this=condition))
            else:
                kept = where.this if not isinstance(where.this, exp.Or) else exp.Paren(this=where.this)
                where.set("this", exp.And(this=kept, expression=condition))
        return select
    return None
