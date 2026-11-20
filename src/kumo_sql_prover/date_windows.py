"""A DATE column against a day that is worked out from a constant (MySQL).

    WHERE visit_day BETWEEN '2019-06-28' AND '2019-07-27'
    WHERE visit_day > DATE_SUB('2019-07-27', INTERVAL 30 DAY) AND visit_day <= '2019-07-27'
    WHERE visit_day >= '2019-07-27' - INTERVAL 29 DAY AND visit_day < '2019-07-28'
    WHERE DATEDIFF('2019-07-27', visit_day) BETWEEN 0 AND 29
    WHERE DATEDIFF('2019-07-27', visit_day) < 30 AND DATEDIFF('2019-07-27', visit_day) >= 0

are five spellings of one window of thirty days. The prover compares dates as the strings they are written as, so it
saw ``DATE_SUB`` and ``DATEDIFF`` as functions it knows nothing about and ``> '2019-06-27'`` as a different test from
``>= '2019-06-28'``. This module writes every such test as the column against a day written out, with ``>=``, ``<=``,
``=`` or ``<>``, so that the five read alike.

**What makes that right.** The column is declared ``DATE``, so it holds whole days: ``d > day`` is ``d >= the next
day`` and ``d < day`` is ``d <= the day before``. The other side is a day the module works out the way the server
does: a string ``'YYYY-MM-DD'`` (one-digit months and days allowed, as the server reads them) that is a calendar
date, ``DATE(..)`` or ``CAST(.. AS DATE)`` of one, and ``DATE_SUB``, ``DATE_ADD``, ``SUBDATE``, ``ADDDATE`` and
``+`` or ``-`` an ``INTERVAL`` of a whole number of days on one. ``DATEDIFF(a, b)`` is the number of days from ``b`` to
``a``, so ``DATEDIFF(day, d) < n`` is ``d > day - n``. A NULL column makes every one of these spellings NULL.

**What is assumed.** The column holds calendar dates in the range the server documents for ``DATE``, 1000-01-01 to
9999-12-31. That is what the server's default SQL mode stores; a mode that admits ``'2019-07-00'`` would put a value
between two days. Every day the module works out, and the day next to it where a strict test needs one, must lie in
that range too, or the test is left alone.

**What is read.** A comparison (``=``, ``<>``, ``<``, ``<=``, ``>``, ``>=``, either way round) or a ``BETWEEN`` whose
one side is a column declared ``DATE`` (``DATE(d)`` and ``CAST(d AS DATE)`` of one are the column) and whose other
side is such a day; and a comparison or ``BETWEEN`` of ``DATEDIFF`` of such a column and such a day with whole
numbers. The column is one of a table of the nearest select, or an output of a derived table there that passes such a
column through (plainly, under ``MIN`` or ``MAX``, or with ``*``). A bare name must belong to exactly one source of
the select, and a select-list alias of the same name that means something else leaves the test alone. Months and
years (whose ends the server clamps), times of day, two columns compared, and other dialects are not read.

A pre-pass of the twelfth hardest-cases batch (``shape_fallback.TWELFTH_BATCH``). Each reading was checked on a
MySQL 8.0 server (``tests/test_date_windows.py`` records the statements).
"""

from __future__ import annotations

import datetime
import re

from sqlglot import exp

from .ast_utils import select_sources

_DAY = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})\Z")
_FIRST, _LAST = datetime.date(1000, 1, 1), datetime.date(9999, 12, 31)
_FLIPPED = {exp.GT: exp.LT, exp.LT: exp.GT, exp.GTE: exp.LTE, exp.LTE: exp.GTE, exp.EQ: exp.EQ, exp.NEQ: exp.NEQ}
_SHIFTS = {"SUBDATE": -1, "ADDDATE": 1}


def _unparen(node: exp.Expression | None) -> exp.Expression | None:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _only(node: exp.Expression, *keys: str) -> bool:
    """Whether ``node`` has no argument set besides ``keys``."""

    return not any(value for key, value in node.args.items() if key not in keys)


def _whole(node: exp.Expression | None) -> int | None:
    """A whole number written as a literal (``29``, ``-29``, and ``'29'`` as an ``INTERVAL`` is stored)."""

    node = _unparen(node)
    sign = 1
    if isinstance(node, exp.Neg):
        sign, node = -1, _unparen(node.this)
    if isinstance(node, exp.Literal) and re.fullmatch(r"-?\d{1,6}" if node.is_string else r"\d{1,6}", node.this):
        return sign * int(node.this)
    return None


def _days(node: exp.Expression | None) -> int | None:
    """The whole number of days of ``INTERVAL n DAY``."""

    node = _unparen(node)
    if type(node) is not exp.Interval or not _only(node, "this", "unit") or not _is_day(node.args.get("unit")):
        return None
    return _whole(node.this)


def _is_day(unit: exp.Expression | None) -> bool:
    return isinstance(unit, exp.Var) and unit.name.upper() == "DAY"


def _within(day: datetime.date | None) -> datetime.date | None:
    return day if day is not None and _FIRST <= day <= _LAST else None


def _moved(day: datetime.date | None, days: int | None) -> datetime.date | None:
    if day is None or days is None:
        return None
    try:
        return _within(day + datetime.timedelta(days=days))
    except OverflowError:
        return None


def constant_day(node: exp.Expression | None) -> datetime.date | None:
    """The day ``node`` is on every database, worked out as the server does, or None when it is not such a constant."""

    node = _unparen(node)
    if isinstance(node, exp.Literal):
        found = _DAY.match(node.this) if node.is_string else None
        if found is None:
            return None
        try:
            return _within(datetime.date(*(int(part) for part in found.groups())))
        except ValueError:
            return None
    if type(node) is exp.TsOrDsToDate and _only(node, "this"):
        return constant_day(node.this)
    if type(node) is exp.Cast and _only(node, "this", "to") and node.to.sql(dialect="mysql").upper() == "DATE":
        return constant_day(node.this)
    if type(node) in (exp.DateSub, exp.DateAdd) and _only(node, "this", "expression", "unit") and _is_day(node.args.get("unit")):
        days = _whole(node.expression)
        return _moved(constant_day(node.this), None if days is None else days if type(node) is exp.DateAdd else -days)
    if type(node) is exp.Anonymous and node.name.upper() in _SHIFTS and len(node.expressions) == 2:
        amount = node.expressions[1]
        days = _days(amount) if isinstance(_unparen(amount), exp.Interval) else _whole(amount)
        return _moved(constant_day(node.expressions[0]), None if days is None else _SHIFTS[node.name.upper()] * days)
    if type(node) is exp.Sub and _days(node.expression) is not None:
        return _moved(constant_day(node.this), -_days(node.expression))
    if type(node) is exp.Add:
        for day, amount in ((node.this, node.expression), (node.expression, node.this)):
            if _days(amount) is not None:
                return _moved(constant_day(day), _days(amount))
    return None


class _Dates:
    """Which columns are declared ``DATE``."""

    def __init__(self, schema: dict | None, types: dict):
        self.kinds = {str(table).lower().split(".")[-1]: {str(column).lower(): str(kind) for column, kind in columns.items()} for table, columns in types.items()}
        self.listed = {str(table).lower().split(".")[-1]: {str(column).lower() for column in columns} for table, columns in (schema or {}).items()}
        for table, columns in self.kinds.items():
            self.listed.setdefault(table, set(columns))

    def _outputs(self, source: exp.Expression, depth: int = 0) -> set[str] | None:
        """The column names ``source`` gives, or None when they are not known."""

        if isinstance(source, exp.Table):
            if source.args.get("db") or source.args.get("catalog"):
                return None
            return self.listed.get(source.name.lower())
        body = self._body(source)
        if body is None or depth > 4:
            return None
        names: set[str] = set()
        for item in body.expressions:
            if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
                inner = [s for s in select_sources(body) if not isinstance(item, exp.Column) or s.alias_or_name.lower() == item.table.lower()]
                if not inner or (isinstance(item, exp.Star) and any(item.args.values())):
                    return None
                for one in inner:
                    given = self._outputs(one, depth + 1)
                    if given is None or names & given:
                        return None
                    names |= given
                continue
            name = (item.alias_or_name or "").lower()
            if not name or name in names:
                return None
            names.add(name)
        return names

    @staticmethod
    def _body(source: exp.Expression) -> exp.Select | None:
        """The select of a derived table whose outputs keep their names."""

        if type(source) is not exp.Subquery or type(source.this) is not exp.Select:
            return None
        alias = source.args.get("alias")
        if alias is not None and alias.args.get("columns"):
            return None
        body = source.this
        return None if body.args.get("with_") or body.args.get("with") else body

    def gives_date(self, source: exp.Expression, name: str, depth: int = 0) -> bool:
        """Whether output ``name`` of ``source`` is a column declared ``DATE``, passed through unchanged."""

        if isinstance(source, exp.Table):
            if source.args.get("db") or source.args.get("catalog"):
                return False
            declared = self.kinds.get(source.name.lower(), {}).get(name)
            return declared is not None and re.sub(r"[(<].*", "", declared).strip().upper() == "DATE"
        body = self._body(source)
        if body is None or depth > 4 or self._outputs(source) is None:
            return False
        items = [item for item in body.expressions if (item.alias_or_name or "").lower() == name and not isinstance(item, exp.Star) and not (isinstance(item, exp.Column) and isinstance(item.this, exp.Star))]
        if len(items) == 1:
            value = _unparen(items[0].unalias())
            if type(value) in (exp.Min, exp.Max) and _only(value, "this") and not isinstance(value.parent, exp.Window):
                value = _unparen(value.this)
            return self.column(value, body, depth + 1) is not None
        if items:
            return False
        holders = [s for s in select_sources(body) if name in (self._outputs(s, depth + 1) or ())]  # it came through a star
        return len(holders) == 1 and self.gives_date(holders[0], name, depth + 1)

    def column(self, node: exp.Expression | None, select: exp.Select, depth: int = 0) -> exp.Column | None:
        """The column behind ``node`` when it is a ``DATE`` column of ``select``'s sources (``DATE(d)`` is ``d``)."""

        node = _unparen(node)
        while (type(node) is exp.TsOrDsToDate and _only(node, "this")) or (type(node) is exp.Cast and _only(node, "this", "to") and node.to.sql(dialect="mysql").upper() == "DATE"):
            node = _unparen(node.this)
        if not isinstance(node, exp.Column) or not isinstance(node.this, exp.Identifier) or node.args.get("db") or node.args.get("catalog"):
            return None
        sources = select_sources(select)
        names = [source.alias_or_name.lower() for source in sources if isinstance(source, (exp.Table, exp.Subquery))]
        if not sources or len(names) != len(sources) or len(set(names)) != len(names) or "" in names:
            return None
        name = node.name.lower()
        if node.table:
            if node.table.lower() not in names:
                return None
            source = sources[names.index(node.table.lower())]
        else:
            given = [self._outputs(source) for source in sources]
            if any(columns is None for columns in given):
                return None
            holders = [source for source, columns in zip(sources, given) if name in columns]
            if len(holders) != 1 or _shadowed(name, select):
                return None
            source = holders[0]
        return node if self.gives_date(source, name, depth) else None


def _shadowed(name: str, select: exp.Select) -> bool:
    """Whether a select-list alias ``name`` could be what a bare ``name`` means instead of the column."""

    for item in select.expressions:
        if isinstance(item, exp.Alias) and item.alias.lower() == name:
            value = _unparen(item.this)
            if not (isinstance(value, exp.Column) and value.name.lower() == name):
                return True
    return False


def _test(column: exp.Column, kind: type, day: datetime.date | None) -> exp.Expression | None:
    """``column kind 'day'``, a strict test written as the test on the day next to it."""

    if day is None:
        return None
    if kind is exp.GT:
        kind, day = exp.GTE, _moved(day, 1)
    elif kind is exp.LT:
        kind, day = exp.LTE, _moved(day, -1)
    if day is None:
        return None
    return kind(this=column.copy(), expression=exp.Literal.string(day.isoformat()))


def _between(column: exp.Column, low: datetime.date | None, high: datetime.date | None) -> exp.Expression | None:
    if low is None or high is None:
        return None
    return exp.Between(this=column.copy(), low=exp.Literal.string(low.isoformat()), high=exp.Literal.string(high.isoformat()))


def _difference(node: exp.Expression | None, select: exp.Select, dates: _Dates) -> tuple[exp.Column, datetime.date, int] | None:
    """``(d, day, sign)`` when ``node`` is ``DATEDIFF`` of a DATE column and a constant day: ``sign * (d - day)`` days."""

    node = _unparen(node)
    if type(node) is not exp.DateDiff or not _only(node, "this", "expression", "unit", "date_part_boundary"):
        return None
    if node.args.get("unit") is not None and not _is_day(node.args.get("unit")):
        return None
    for column, day, sign in ((node.this, node.expression, 1), (node.expression, node.this, -1)):
        found, constant = dates.column(column, select), constant_day(day)
        if found is not None and constant is not None:
            return found, constant, sign
    return None


def _rewritten(node: exp.Expression, select: exp.Select, dates: _Dates) -> exp.Expression | None:
    """The test ``node`` written as the column against days written out, or None when it is not read."""

    if type(node) in _FLIPPED and _only(node, "this", "expression"):
        for left, right, kind in ((node.this, node.expression, type(node)), (node.expression, node.this, _FLIPPED[type(node)])):
            column = dates.column(left, select)
            if column is not None and constant_day(right) is not None:
                return _test(column, kind, constant_day(right))
            difference, number = _difference(left, select, dates), _whole(right)
            if difference is not None and number is not None:
                column, day, sign = difference
                # d - day kind n says d kind day + n; day - d kind n says d (kind turned round) day - n
                return _test(column, kind if sign > 0 else _FLIPPED[kind], _moved(day, sign * number))
        return None
    if type(node) is exp.Between and _only(node, "this", "low", "high"):
        column = dates.column(node.this, select)
        if column is not None:
            return _between(column, constant_day(node.args["low"]), constant_day(node.args["high"]))
        difference, low, high = _difference(node.this, select, dates), _whole(node.args["low"]), _whole(node.args["high"])
        if difference is not None and low is not None and high is not None and low <= high:
            column, day, sign = difference
            return _between(column, _moved(day, low), _moved(day, high)) if sign > 0 else _between(column, _moved(day, -high), _moved(day, -low))
    return None


def _plan(tree: exp.Expression, dates: _Dates) -> list[tuple[exp.Expression, exp.Expression]]:
    planned = []
    for node in tree.find_all(exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between):
        select = node.find_ancestor(exp.Select)
        if select is None or select.args.get("with_") or select.args.get("with"):
            continue
        new = _rewritten(node, select, dates)
        if new is not None and new.sql(dialect="mysql") != node.sql(dialect="mysql"):
            planned.append((node, new))
    return planned


def write_date_windows(tree: exp.Expression, schema: dict | None, types: dict | None, dialect: str | None) -> tuple[exp.Expression, bool]:
    """``tree`` with each test of a DATE column against a constant day written out (module doc); whether it changed."""

    if dialect != "mysql" or not types:
        return tree, False
    dates = _Dates(schema, types)
    if not any(re.sub(r"[(<].*", "", kind).strip().upper() == "DATE" for columns in dates.kinds.values() for kind in columns.values()):
        return tree, False
    if not _plan(tree, dates):
        return tree, False
    tree = tree.copy()
    for node, new in _plan(tree, dates):
        node.replace(new)
    return tree, True
