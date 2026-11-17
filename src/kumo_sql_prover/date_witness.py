"""Counterexamples for columns declared ``DATE``, ``DATETIME`` or ``TIMESTAMP``.

The SMT prover's values are numbers, strings and Booleans, so a date is the string it is written as and a column
declared ``DATE`` holds whatever string or number the solver picks. Asked whether ``d BETWEEN '2019-06-28' AND
'2019-07-27'`` and ``d > '2019-06-27' AND d <= '2019-07-27'`` differ, it answered "yes, on the row
``d = '2019-06-270'``": a text between the two days that no ``DATE`` column can hold. For a date column the two are
the same query.

A counterexample is a database, so a declared date column in it holds dates. This module says

- which strings such a column may hold (:func:`holds`): ``'YYYY-MM-DD'`` for a real day, and for a column that keeps
  the time of day also ``'YYYY-MM-DD hh:mm:ss'`` with a time that is not midnight. Midnight is written as the day
  alone, so every moment has one spelling, and those spellings sort as text the way the moments sort in time. A
  comparison of two such strings, or of one with a literal written the same way, therefore comes out for the text as it
  does for the dates;
- which values the solver is asked to choose from (:func:`domain`): the days the queries name and the days around
  them. When none of them separates the queries the answer is "not proven", never a made-up row;
- whether the queries read their date columns only in that way (:func:`readable`). A date column may be selected,
  grouped, ordered, counted, taken the ``MIN`` or ``MAX`` of, tested for ``NULL`` and compared with another date column
  or with a string literal in one of the two spellings (also inside ``CAST(.. AS DATE)``). Anything else (``LENGTH(d)``, ``d = 20190627``, ``d + 1``, a
  literal ``'2019/06/27'``, a text column on the other side) is read by an engine in a way the text does not follow,
  and then a counterexample may hold only ``NULL`` in its date columns (:data:`NODATE`).

Seconds are the finest moment written, so a difference that needs a fraction of a second is not found. A column with no
declared type is not covered: the prover's row for it is a row of a text column, which is a database.
"""

from __future__ import annotations

import datetime
import re

import sqlglot
from sqlglot import exp

from . import string_number_compare as compare

DATE = "date"  # a column that holds days
STAMP = "stamp"  # a column that keeps the time of day
NODATE = "nodate"  # a date column of queries the text reading does not cover: only NULL in a counterexample
KINDS = (DATE, STAMP, NODATE)

_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MOMENT = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}):(\d{2}):(\d{2})$")
_NEAR = 2  # days on each side of a day the queries name
_ONE_DAY = datetime.timedelta(days=1)


def kind(type_sql: str | None) -> str | None:
    """:data:`DATE` or :data:`STAMP` for a declared date or timestamp type, else None."""

    if not type_sql:
        return None
    if compare._is_temporal_type(type_sql, ("DATE",)):
        return DATE
    return STAMP if compare._is_temporal_type(type_sql, compare._STAMPS) else None


def _day(text: str) -> datetime.date | None:
    if not _DAY.match(text):
        return None
    try:
        day = datetime.date.fromisoformat(text)
    except ValueError:
        return None
    return day if 1000 <= day.year <= 9999 else None  # the years every engine's DATE holds


def _moment(text: str) -> tuple[datetime.date, tuple[int, int, int]] | None:
    found = _MOMENT.match(text)
    day = _day(found.group(1)) if found else None
    if day is None:
        return None
    hour, minute, second = (int(found.group(index)) for index in (2, 3, 4))
    return (day, (hour, minute, second)) if hour < 24 and minute < 60 and second < 60 else None


def holds(text: object, column: str) -> bool:
    """Whether a column of kind ``column`` can hold ``text``, written the one way this module writes that value."""

    if not isinstance(text, str) or column not in (DATE, STAMP):
        return False
    if _day(text) is not None:
        return True  # for a column that keeps the time: midnight
    moment = _moment(text) if column == STAMP else None
    return moment is not None and moment[1] != (0, 0, 0)


def fits(literal: str, column: str | None) -> bool:
    """Whether a string literal compared with a date column is one of the spellings the column's values are written in.

    A day alone fits any date column. A day with a time fits only a column known to keep the time (a ``DATE`` column
    compared with one is read as days by one engine and as moments by another), and not midnight, which is written as
    the day alone.
    """

    return holds(literal, STAMP if column == STAMP else DATE)


def domain(literals, column: str) -> list[str]:
    """The values a counterexample may put in a column of kind ``column``, in order: each day the queries' string
    ``literals`` name with the days around it (a few days of January 2000 when they name none) and, for a column that
    keeps the time, noon of each of those days and the moments the queries name."""

    days: set[datetime.date] = set()
    for text in literals:
        moment = _moment(text)
        day = _day(text) or (moment[0] if moment else None)
        if day is not None:
            days.update(day + offset * _ONE_DAY for offset in range(-_NEAR, _NEAR + 1) if 1001 <= day.year <= 9998)
            days.add(day)
    if not days:
        days = {datetime.date(2000, 1, 1) + offset * _ONE_DAY for offset in range(2 * _NEAR + 1)}
    found = {day.isoformat() for day in days}
    if column == STAMP:
        found.update(f"{day.isoformat()} 12:00:00" for day in days)
        found.update(text for text in literals if holds(text, STAMP))
    return sorted(found)


def _through(node: exp.Expression) -> exp.Expression:
    """``node`` without the wrappers that hand a date on as the date it is: parentheses, an alias, ``MIN`` and ``MAX``."""

    while isinstance(node, (exp.Paren, exp.Alias, exp.Min, exp.Max)) and isinstance(node.this, exp.Expression):
        node = node.this
    return node


def readable(sql: str, dialect: str, types: dict[str, dict[str, str]] | None, schema: dict | None = None) -> bool:
    """Whether ``sql`` reads the declared date and timestamp columns only in ways that come out the same for the strings
    :func:`holds` accepts as they do for the dates (the module docstring lists them).

    A column is taken for a date column by its name alone: the name of a declared date column of any table, or the alias
    a query gives one (``MIN(t.d) AS first_day``). That counts too many columns and never too few, and a column counted
    wrongly only costs a counterexample. The column on the other side of a comparison has to be a date by its declared
    type (``_Kinds.origin``), since a column of no type that only shares the name can hold text.
    """

    names = {column.lower() for columns in (types or {}).values() for column, type_sql in columns.items() if kind(type_sql)}
    if not names:
        return True
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:  # noqa: BLE001 - the prover reports the parse error itself
        return True
    kinds = compare._Kinds(tree, types, schema)

    def dated(node: exp.Expression) -> bool:
        return isinstance(node, exp.Column) and node.name.lower() in names

    def known(node: exp.Expression) -> bool:
        """Whether ``node`` is a column the declared types make a date, also through a derived table."""

        return isinstance(node, exp.Column) and kinds.origin(kinds.key(node)) == "date"

    changed = True
    while changed:  # an alias of a date column is a date column wherever the name is read
        changed = False
        for alias in tree.find_all(exp.Alias):
            if alias.alias and alias.alias.lower() not in names and dated(_through(alias)):
                names.add(alias.alias.lower())
                changed = True
    for join in tree.find_all(exp.Join):
        if any(ident.name.lower() in names for ident in join.args.get("using") or []):
            return False  # USING pairs the column with one of another table without naming either
    for node in tree.find_all(compare._SET_OPERATION):
        for branch in compare._branches(node):
            if any(dated(_through(item)) for item in branch.selects):
                return False  # the other branch's column at that position can be text
    compared: set[int] = set()
    for left, right in compare._pairs(tree):
        sides = (_through(left), _through(right))
        compared.update(id(side) for side in sides)
        for column, other in (sides, sides[::-1]):
            if not dated(column) or known(other) or isinstance(other, exp.Null):
                continue
            stamp = STAMP if kinds._temporal(kinds.key(column), compare._STAMPS) is True else DATE
            if isinstance(other, exp.Cast) and type(other) is exp.Cast and other.args["to"].is_type(exp.DataType.Type.DATE):
                other, stamp = other.this, DATE  # the prover reads CAST('2019-06-27' AS DATE) as the string
            if not (isinstance(other, exp.Literal) and other.is_string and fits(other.this, stamp)):
                return False
    for column in tree.find_all(exp.Column):
        if not dated(column):
            continue
        node: exp.Expression = column
        while True:
            parent = node.parent
            if id(node) in compared:
                while isinstance(parent, (exp.Paren, exp.Alias, exp.Min, exp.Max)):
                    parent = parent.parent
                if isinstance(parent, exp.Nullif):
                    return False  # NULLIF(d, '..') is still the date, and something else reads it
                break
            if isinstance(parent, (exp.Paren, exp.Alias, exp.Min, exp.Max, exp.Distinct)):
                node = parent
                continue
            if isinstance(parent, (exp.Select, exp.Group, exp.Ordered, exp.Is, exp.Count)):
                break
            if isinstance(parent, exp.Window) and node.arg_key == "partition_by":
                break
            return False
    return True
