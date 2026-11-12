"""A constant timestamp plus or minus a constant interval is the timestamp it evaluates to.

    CAST('2024-10-01 12:34:56' AS TIMESTAMP) - INTERVAL '1 year'    is    CAST('2023-10-01 12:34:56' AS TIMESTAMP)
    TIMESTAMP '2024-10-01 12:34:56' - INTERVAL '30' DAY             is    CAST('2024-09-01 12:34:56' AS TIMESTAMP)

``algebraic_equivalence._fold_dates`` does this for dates. A rewriter that reduces constant expressions (Calcite's
``ReduceExpressionsRule``, which R-Bot and LLM-R2 apply) prints the computed timestamp, and the two spellings of one
instant were compared as different values.

The arithmetic is PostgreSQL's for ``timestamp without time zone``, which DuckDB shares: a value of that type is a
calendar date and a time of day with no zone, so no daylight-saving rule applies.

* ``YEAR`` and ``MONTH`` move the calendar month (a year is twelve months) and keep the day of the month and the
  time of day; a day the target month does not have becomes that month's last day
  (``2024-03-31 - 1 month = 2024-02-29``, ``2024-02-29 - 1 year = 2023-02-28``);
* ``WEEK``, ``DAY``, ``HOUR``, ``MINUTE`` and ``SECOND`` add that many times 7 days, 24 hours, 3600, 60 or 1 seconds.

Both operands are constants, so the folded value is the value on every database and for every row: NULLs,
duplicates and empty inputs do not come into it.

Folded only when (the expression is left as written otherwise):

* the dialect is PostgreSQL or DuckDB;
* the left operand is ``CAST('<literal>' AS TIMESTAMP)`` (``TIMESTAMP '<literal>'`` parses to the same) with no
  time zone, and the literal is ``YYYY-MM-DD`` or ``YYYY-MM-DD HH:MM:SS[.ffffff]`` (a ``T`` separator is accepted)
  naming a real date and time. ``TIMESTAMP(p)`` is accepted when the literal has at most ``p`` fractional digits, so
  that the cast rounds nothing and gives the value ``TIMESTAMP`` gives. ``TIMESTAMPTZ`` is declined: its day
  arithmetic depends on the session's time zone;
* the right operand is ``INTERVAL '<integer>' <unit>`` with one of the units above (singular or plural). An interval
  with several fields (``'1 year 2 months'``), a fraction or no unit is declined;
* the operator is ``+`` or ``-`` with the timestamp on the left, and the result lies in the years 1 to 9999.

The result is printed as ``CAST('YYYY-MM-DD HH:MM:SS' AS TIMESTAMP)``, with the fraction only when it is not zero.
"""

from __future__ import annotations

import calendar
import datetime
import re

from sqlglot import exp

_DIALECTS = ("postgres", "duckdb")
_LITERAL = re.compile(r"(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?)?")
_COUNT = re.compile(r"[+-]?\d{1,9}")
_SECONDS = {"WEEK": 7 * 86400, "DAY": 86400, "HOUR": 3600, "MINUTE": 60, "SECOND": 1}
_MONTHS = {"YEAR": 12, "MONTH": 1}


def _timestamp_of(node: exp.Expression) -> datetime.datetime | None:
    if type(node) is not exp.Cast or any(value for key, value in node.args.items() if key not in ("this", "to")):
        return None
    to = node.args.get("to")
    if not isinstance(to, exp.DataType) or to.this != exp.DataType.Type.TIMESTAMP or len(to.expressions) > 1:
        return None
    digits = 6
    if to.expressions:  # TIMESTAMP(p): the same value when the literal has no more than p fractional digits
        precision = to.expressions[0].this if isinstance(to.expressions[0], exp.DataTypeParam) else None
        if not isinstance(precision, exp.Literal) or precision.is_string or precision.this not in ("0", "1", "2", "3", "4", "5", "6"):
            return None
        digits = int(precision.this)
    literal = node.this
    if not isinstance(literal, exp.Literal) or not literal.is_string:
        return None
    match = _LITERAL.fullmatch(literal.this)
    if match is None:
        return None
    year, month, day, hour, minute, second, fraction = match.groups()
    if len((fraction or "").rstrip("0")) > digits:
        return None  # the cast would round the literal
    try:
        return datetime.datetime(int(year), int(month), int(day), int(hour or 0), int(minute or 0), int(second or 0), int((fraction or "0").ljust(6, "0")))  # noqa: DTZ001
    except ValueError:
        return None


def _interval_of(node: exp.Expression) -> tuple[int, str] | None:
    if type(node) is not exp.Interval or any(value for key, value in node.args.items() if key not in ("this", "unit")):
        return None
    amount, unit = node.this, node.args.get("unit")
    if not isinstance(amount, exp.Literal) or not isinstance(unit, exp.Var) or not _COUNT.fullmatch(str(amount.this).strip()):
        return None
    name = unit.name.upper()
    name = name[:-1] if name.endswith("S") else name
    if name not in _SECONDS and name not in _MONTHS:
        return None
    return int(str(amount.this).strip()), name


def shifted(moment: datetime.datetime, count: int, unit: str) -> datetime.datetime | None:
    """``moment`` plus ``count`` units as PostgreSQL computes it for a timestamp without time zone, or None out of range."""

    try:
        if unit in _SECONDS:
            result = moment + datetime.timedelta(seconds=count * _SECONDS[unit])
        else:
            months = moment.year * 12 + moment.month - 1 + count * _MONTHS[unit]
            year, month = divmod(months, 12)
            if not 1 <= year <= 9999:
                return None
            result = moment.replace(year=year, month=month + 1, day=min(moment.day, calendar.monthrange(year, month + 1)[1]))
    except (OverflowError, ValueError):
        return None
    return result if 1 <= result.year <= 9999 else None


def _literal(moment: datetime.datetime) -> exp.Expression:
    text = moment.strftime("%Y-%m-%d %H:%M:%S") if moment.year >= 1000 else f"{moment.year:04d}-" + moment.strftime("%m-%d %H:%M:%S")
    if moment.microsecond:
        text += f".{moment.microsecond:06d}".rstrip("0")
    return exp.Cast(this=exp.Literal.string(text), to=exp.DataType.build("TIMESTAMP"))


def fold_timestamp_shifts(tree: exp.Expression, dialect: str) -> exp.Expression:
    """``tree`` with every constant ``timestamp +/- interval`` replaced by its value (see the module docstring)."""

    if dialect not in _DIALECTS or not any(tree.find_all(exp.Interval)):
        return tree

    def step(node: exp.Expression) -> exp.Expression:
        if type(node) is exp.Paren and type(node.this) in (exp.Add, exp.Sub):
            inner = step(node.this)
            return inner if inner is not node.this else node  # a folded constant needs no parentheses
        if type(node) not in (exp.Add, exp.Sub):
            return node
        base = node.this
        while isinstance(base, exp.Paren):
            base = base.this
        moment, interval = _timestamp_of(base), _interval_of(node.expression)
        if moment is None or interval is None:
            return node
        count, unit = interval
        result = shifted(moment, count if type(node) is exp.Add else -count, unit)
        return _literal(result) if result is not None else node

    for _ in range(4):  # a chain ``t - i1 - i2`` folds from the inside, one level per pass
        before = tree.sql()
        tree = tree.transform(step)
        if tree.sql() == before:
            break
    return tree
