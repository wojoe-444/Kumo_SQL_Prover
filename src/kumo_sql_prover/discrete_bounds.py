"""A strict bound on a whole-valued column is the non-strict bound one step in.

    l.shipped < CAST('1996-01-01' AS DATE)        is        l.shipped <= CAST('1995-12-31' AS DATE)
    l.line > 4                                    is        l.line >= 5

(TPC-H Q6 and the rewrites that turn its half-open year ``d >= x AND d < x + INTERVAL '1 year'`` into
``d BETWEEN x AND <the last day>``). The solver compares dates as texts and numbers as reals, so it finds a value
strictly between the two bounds and the pair stays unproved. A column declared ``DATE`` holds whole days and one
declared an integer type holds whole numbers, so nothing lies between a bound and the next value: ``x < c`` and
``x <= c - 1`` accept the same values, and both are NULL exactly when ``x`` is.

The second half folds a constant date with more than one interval, ``DATE '1995-01-01' + INTERVAL '1 year' -
INTERVAL '1 day'``: ``algebraic_equivalence._fold_dates`` folds one interval per pass, innermost first, which is
the order the database applies them in.

Conditions, all checked:

* the dialect is PostgreSQL, DuckDB, BigQuery or MySQL, where ``DATE`` has no time of day (Oracle's has) and a
  declared type is enforced (SQLite's is not);
* the comparison is ``<`` or ``>`` between a plain column and a constant, in either order, each with or without
  parentheses around it;
* the column's declared type, followed through plain derived tables, is ``DATE`` and the constant is
  ``CAST('YYYY-MM-DD' AS DATE)`` with a day on either side of it in the calendar, or the declared type is an
  integer type and the constant is a whole number literal, with or without a sign;
* a bare column name is not also the name of a computed output of the same select, which ``HAVING`` and
  ``ORDER BY`` could mean instead.
"""

from __future__ import annotations

import datetime
import re

from sqlglot import exp

_DIALECTS = frozenset({"postgres", "duckdb", "bigquery", "mysql"})
_INTEGER = re.compile(r"^\s*(TINYINT|SMALLINT|MEDIUMINT|INT|INTEGER|BIGINT|INT2|INT4|INT8|INT64)\s*$", re.I)
_DATE = re.compile(r"^\s*DATE\s*$", re.I)
_DAY = datetime.timedelta(days=1)


def _whole(node: exp.Expression) -> int | None:
    sign = 1
    if type(node) is exp.Neg:
        sign, node = -1, node.this
    if type(node) is exp.Literal and not node.is_string and re.fullmatch(r"\d+", node.this):
        return sign * int(node.this)
    return None


def _number(value: int) -> exp.Expression:
    return exp.Literal.number(value) if value >= 0 else exp.Neg(this=exp.Literal.number(-value))


def _day(node: exp.Expression) -> datetime.date | None:
    if type(node) is not exp.Cast or any(value for key, value in node.args.items() if key not in ("this", "to")):
        return None
    to = node.args.get("to")
    if type(to) is not exp.DataType or to.this is not exp.DataType.Type.DATE or to.expressions:
        return None
    if type(node.this) is not exp.Literal or not node.this.is_string or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", node.this.this):
        return None
    try:
        return datetime.date.fromisoformat(node.this.this)
    except ValueError:
        return None


def _stepped(constant: exp.Expression, step: int, declared: str) -> exp.Expression | None:
    """``constant`` moved ``step`` values along the column's type, or ``None`` when it is not a constant of it."""

    if _INTEGER.match(declared):
        value = _whole(constant)
        return _number(value + step) if value is not None else None
    if _DATE.match(declared):
        day = _day(constant)
        if day is None or not datetime.date.min + _DAY < day < datetime.date.max - _DAY:
            return None
        return exp.Cast(this=exp.Literal.string((day + step * _DAY).isoformat()), to=exp.DataType.build("DATE"))
    return None


def _bare(node: exp.Expression) -> exp.Expression:
    while type(node) is exp.Paren:
        node = node.this
    return node


def _strict(node: exp.Expression, select: exp.Select) -> bool:
    if type(node) not in (exp.LT, exp.GT) or node.find_ancestor(exp.Select) is not select:
        return False
    return (type(_bare(node.this)) is exp.Column) != (type(_bare(node.expression)) is exp.Column)


def close_strict_bounds(select: exp.Select, types: dict | None, dialect: str | None) -> exp.Expression | None:
    """``select`` with every strict bound of its own on a ``DATE`` or integer column made non-strict, or ``None``."""

    if not types or (dialect or "").lower() not in _DIALECTS:
        return None
    if not any(_strict(node, select) for node in select.find_all(exp.LT, exp.GT)):
        return None
    from .algebraic_equivalence import _origin_type

    computed = {
        item.alias.lower()
        for item in select.expressions
        if isinstance(item, exp.Alias) and not (type(item.this) is exp.Column and item.this.name.lower() == item.alias.lower())
    }
    copy = select.copy()
    changed = False
    for node in [n for n in copy.find_all(exp.LT, exp.GT) if _strict(n, copy)]:
        column_first = type(_bare(node.this)) is exp.Column
        column, constant = (_bare(node.this), _bare(node.expression)) if column_first else (_bare(node.expression), _bare(node.this))
        if isinstance(column.this, exp.Star) or len(column.parts) > 2 or (not column.table and column.name.lower() in computed):
            continue
        declared = _origin_type(copy, column, types)
        if not declared:
            continue
        # ``x < c`` and ``c > x`` bound x from above: the constant steps down; the other two step up
        down = (type(node) is exp.LT) == column_first
        moved = _stepped(constant, -1 if down else 1, declared)
        if moved is None:
            continue
        closed = exp.LTE if type(node) is exp.LT else exp.GTE
        node.replace(closed(this=column.copy(), expression=moved) if column_first else closed(this=moved, expression=column.copy()))
        changed = True
    return copy if changed else None


def fold_chained_dates(tree: exp.Expression) -> exp.Expression:
    """``tree`` with a constant date under several intervals folded to one date."""

    from .algebraic_equivalence import _date_of, _fold_dates

    def chained(node: exp.Expression) -> bool:
        return isinstance(node, (exp.Add, exp.Sub)) and isinstance(node.expression, exp.Interval) and _date_of(node.this) is not None

    for _ in range(4):
        if not any(chained(node) for node in tree.find_all(exp.Add, exp.Sub)):
            break
        tree = _fold_dates(tree)
    return tree
