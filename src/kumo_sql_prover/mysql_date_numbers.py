"""MySQL does arithmetic on a date as a number; DuckDB moves the date. The search's DuckDB text takes MySQL's reading.

For a DATE column ``d`` that holds 2024-01-31, MySQL's ``d + 1`` is the number 20240132: the date is read as the
digits of its year, month and day. ``d + 1 = e`` then compares numbers, so it is FALSE when ``e`` is 2024-02-01,
and ``e - d`` is 70 there, not 1. A TIME is the digits of its hours, minutes and seconds the same way. DuckDB
reads ``d + 1`` as the next day and ``e - d`` as a number of days, so the counterexample search called a pair
that writes "the day after" as ``d + 1`` equal to one that writes ``DATE_ADD(d, INTERVAL 1 DAY)``, which MySQL
tells apart on the last day of every month.

``read`` rewrites a MySQL tree, before it is printed for DuckDB, in four ways:

* **An operand of** ``+ - * / DIV %`` **that is a date or a time** is written as MySQL's number:
  ``CAST(STRFTIME(d, '%Y%m%d') AS BIGINT)``. Arithmetic with an ``INTERVAL`` is date arithmetic in both engines
  and is left alone.
* **A date or a time compared with a number that is not a constant** (``=  <>  <  <=  >  >=  <=>``, ``BETWEEN``,
  an ``IN`` list) is written as its number too, because MySQL compares the two as numbers. So is the item of an
  ``IN`` subquery that a number on the left is compared with. So is a DATE compared by one of the six operators
  with a constant that is a whole number (``d = 20240131``, ``d < DATE('2024-01-31') + 1``): MySQL's answer is the
  numbers' whether or not the constant is a date (``d < 20240132`` is every day up to the 31st). Other constants
  are left alone: MySQL reads ``d BETWEEN 20240101 AND 20240132``, ``d IN (20240131, 20240132)`` and
  ``d < 20240131.5`` through dates, and DuckDB refuses a date against a number, so such a query is one the
  search cannot run.
* **A date moved by whole days** (``d + INTERVAL 1 DAY``, ``DATE_ADD(d, INTERVAL 1 MONTH)``) is cast back to a
  date. It is a DATE in MySQL and a TIMESTAMP in DuckDB, which shows once the value is an output (2024-02-01
  against 2024-02-01 00:00:00) or is written into a string (``CONCAT(id, ',', d + INTERVAL 1 DAY)``): the search
  then reported a difference MySQL does not have.
* **An operand of** ``+`` **or** ``-`` **whose type this module cannot tell** is wrapped so that DuckDB refuses the
  query when the operand is a date, a time or a timestamp after all (``_guarded``). The search then answers
  "unknown" where it would have done DuckDB's arithmetic.

What is a date is decided from the declared column types, followed through derived tables, ``WITH`` names, set
operations, scalar subqueries and the calls that keep a date a date in MySQL (``MIN``, ``MAX``, ``COALESCE``,
``IFNULL``, ``IF``, ``CASE``, ``LEAST``, ``GREATEST``, ``LAG``, ``LEAD``, ``FIRST_VALUE``, ``LAST_VALUE``,
``DATE()``, ``CAST(.. AS DATE)``, and ``DATE_ADD`` / ``DATE_SUB`` / ``+ INTERVAL`` by days, weeks, months,
quarters or years). Each was checked on MySQL 8.0; ``tests/test_mysql_date_numbers.py`` holds the rows.
``IFNULL(d, 0)`` is a string there (``IFNULL(d, 0) + 1`` is 2025), and so is every mix of a date with something
else: those are not called dates, and the guard or DuckDB's own type check refuses them.

The module only changes what the search runs. The provers read ``d + 1`` as plain arithmetic on an unknown
number, which is MySQL's reading, and never as a day added.
"""

from __future__ import annotations

import re

from sqlglot import exp

from .float_regrouping import _branches, _is_star, _label, _Source, _unaliased
from .float_regrouping import _sources as _all_sources

DATE, TIME, NUMBER, TEXT = "date", "time", "number", "text"
_TEMPORAL = (DATE, TIME)
_DEPTH = 12
_BINARY = (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.IntDiv, exp.Mod)
_COMPARISONS = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.NullSafeEQ, exp.NullSafeNEQ)
_DAY_UNITS = {"DAY", "WEEK", "MONTH", "QUARTER", "YEAR"}
_NUMERIC_TYPES = {"INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "MEDIUMINT", "NUMERIC", "DECIMAL", "FLOAT", "DOUBLE", "REAL"}
_NUMERIC_CASTS = {
    getattr(exp.DataType.Type, name)
    for name in ("TINYINT", "SMALLINT", "INT", "BIGINT", "UTINYINT", "USMALLINT", "UINT", "UBIGINT", "DECIMAL", "DOUBLE", "FLOAT")
    if hasattr(exp.DataType.Type, name)
}
# calls whose value is a number whatever they are given
_NUMBER_CALLS = tuple(
    getattr(exp, name)
    for name in (
        "Count", "Sum", "Avg", "DateDiff", "TimestampDiff", "Year", "Month", "Day", "DayOfWeek", "DayOfMonth", "DayOfYear",
        "Week", "WeekOfYear", "Quarter", "Extract", "Abs", "Round", "Floor", "Ceil", "Sign", "Pow", "Sqrt", "Length",
        "RowNumber", "Rank", "DenseRank", "Ntile",
    )  # fmt: skip
    if hasattr(exp, name)
)
# calls whose value is one of their arguments: a date stays a date
_PICKS = (exp.Min, exp.Max, exp.AnyValue, exp.FirstValue, exp.LastValue)
_SHIFTS = (exp.Lag, exp.Lead)


def column_kind(declared: str, as_number: bool = False) -> str | None:
    """The kind of a column from its declared type: ``"date"``, ``"time"``, ``"number"`` or ``None``."""

    name = declared.split("(")[0].strip().upper()
    if name == "DATE":
        return DATE
    if name == "TIME":
        return TIME
    if name in _NUMERIC_TYPES or (as_number and name in ("BOOL", "BOOLEAN")):
        return NUMBER
    return None


def _interval(node: exp.Expression) -> bool:
    return isinstance(node, exp.Interval)


def _unit(node: exp.Expression) -> str:
    unit = node.args.get("unit")
    return (unit.name if isinstance(unit, exp.Expression) else str(unit or "")).upper()


def _sources(select: exp.Select) -> list[_Source]:
    """The sources of ``select``, with ``LATERAL (SELECT ..)`` as one this module does not follow.

    MySQL's readings of a date that comes out of a lateral derived table were never checked on a server, so its
    columns stay of unknown type here: the guard, or DuckDB's own type check, refuses arithmetic on them.
    """

    return [_Source(source.alias, "opaque", source.holder) if isinstance(source.holder, exp.Lateral) else source for source in _all_sources(select)]


class _Reader:
    """Kinds of the expressions of one tree. ``kinds`` maps a lower-case table name to its columns' kinds."""

    def __init__(self, kinds: dict[str, dict[str, str | None]]):
        self.kinds = kinds

    # --- columns -------------------------------------------------------------------------------------------

    def _output(self, source, name: str, depth: int) -> tuple[bool | None, str | None]:
        """Whether a derived table or ``WITH`` name has an output column ``name`` (``None``: cannot be told) and
        the column's kind."""

        branches = _branches(source.query)
        if branches is None or depth <= 0:
            return None, None
        first = branches[0]
        starred = any(_is_star(item) for branch in branches for item in branch.expressions)
        if source.names:
            if name not in source.names:
                return False, None
            if starred:
                return True, None
            position = source.names.index(name)
        else:
            hits = [i for i, item in enumerate(first.expressions) if not _is_star(item) and _label(item) == name]
            if len(hits) > 1:
                return True, None
            if not hits:
                if not starred:
                    # an item the engine names (``MIN(d)`` without an alias) cannot be read by a plain name
                    return False, None
                if len(branches) > 1:
                    return None, None
                found: list[tuple[bool | None, str | None]] = []
                for item in first.expressions:
                    if _is_star(item):
                        qualifier = item.table.lower() if isinstance(item, exp.Column) and item.table else ""
                        found.append(self._column(name, qualifier, first, depth - 1, outward=False))
                if any(has is None for has, _ in found):
                    return None, None
                owners = [kind for has, kind in found if has]
                return (True, owners[0]) if len(owners) == 1 else ((False, None) if not owners else (True, None))
            if starred and len(branches) > 1:
                return True, None  # a star moves the positions of the other branches
            position = hits[0]
        kinds = set()
        for branch in branches:
            if position >= len(branch.expressions):
                return True, None
            kinds.add(self.kind(_unaliased(branch.expressions[position]), branch, depth - 1))
        return True, (kinds.pop() if len(kinds) == 1 else None)

    def _column(self, name: str, qualifier: str, select: exp.Select, depth: int, outward: bool = True) -> tuple[bool | None, str | None]:
        """Whether a source of ``select`` (or, with ``outward``, of a select around it) has the column, and its kind."""

        if depth <= 0:
            return None, None
        answers: list[tuple[bool | None, str | None]] = []
        for source in _sources(select):
            if qualifier and source.alias != qualifier:
                continue
            if source.kind == "table":
                columns = self.kinds.get(source.key) or self.kinds.get(source.key.split(".")[-1])
                if columns is None or source.names:
                    answers.append((None, None))
                else:
                    answers.append((name in columns, columns.get(name)))
            elif source.kind == "query":
                answers.append(self._output(source, name, depth))
            else:
                answers.append((None, None))
        if any(has is None for has, _ in answers):
            return None, None
        owners = [kind for has, kind in answers if has]
        if len(owners) == 1:
            return True, owners[0]
        if owners:
            return True, None  # two sources have the name
        if not qualifier:
            # MySQL reads a name no source has as an item of the select's own list (HAVING, GROUP BY, ORDER BY)
            items = [item for item in select.expressions if isinstance(item, exp.Alias) and item.alias.lower() == name]
            if len(items) == 1:
                return True, self.kind(items[0].this, select, depth - 1)
            if items:
                return True, None
        if outward:
            outer = select.parent
            while outer is not None and not isinstance(outer, exp.Select):
                outer = outer.parent
            if outer is not None:
                return self._column(name, qualifier, outer, depth - 1)
        return False, None

    def stored(self, column: exp.Column, select: exp.Select | None) -> bool:
        """Whether ``column`` is read straight from a table of ``select``, not from a derived table or a ``WITH``
        query (nor from a select around it)."""

        if select is None:
            return False
        name, qualifier = column.name.lower(), column.table.lower()
        owners: list[bool] = []
        for source in _sources(select):
            if qualifier and source.alias != qualifier:
                continue
            if source.kind == "table":
                columns = self.kinds.get(source.key) or self.kinds.get(source.key.split(".")[-1])
                if columns is None or source.names:
                    owners.append(False)
                elif name in columns:
                    owners.append(True)
            elif source.kind != "query" or self._output(source, name, _DEPTH)[0] is not False:
                owners.append(False)
        return owners == [True]

    # --- expressions ---------------------------------------------------------------------------------------

    def _same(self, nodes, select, depth) -> str | None:
        """The one kind of ``nodes``, NULLs aside; ``None`` when they differ or one is not known."""

        kinds = {self.kind(node, select, depth - 1) for node in nodes if not isinstance(node, exp.Null)}
        return kinds.pop() if len(kinds) == 1 else None

    def kind(self, node: exp.Expression, select: exp.Select | None, depth: int = _DEPTH) -> str | None:
        """``"date"``, ``"time"``, ``"number"``, ``"text"`` (a string literal) or ``None`` (anything else, or not known)."""

        if depth <= 0 or node is None:
            return None
        if isinstance(node, (exp.Paren, exp.Alias)):
            return self.kind(node.this, select, depth)
        if isinstance(node, exp.Literal):
            return TEXT if node.is_string else NUMBER
        if isinstance(node, exp.Column):
            if select is None or isinstance(node.this, exp.Star):
                return None
            return self._column(node.name.lower(), node.table.lower(), select, depth)[1]
        if isinstance(node, _BINARY):
            if isinstance(node, (exp.Add, exp.Sub)) and (_interval(node.this) or _interval(node.expression)):
                moved, by = (node.expression, node.this) if _interval(node.this) else (node.this, node.expression)
                if isinstance(node, exp.Sub) and _interval(node.this):
                    return None
                return DATE if _unit(by) in _DAY_UNITS and self.kind(moved, select, depth - 1) == DATE else None
            return NUMBER
        if isinstance(node, exp.Neg):
            return NUMBER
        if isinstance(node, (exp.DateAdd, exp.DateSub)):
            return DATE if _unit(node) in _DAY_UNITS and self.kind(node.this, select, depth - 1) == DATE else None
        if isinstance(node, exp.Cast):
            target = node.to.this if isinstance(node.to, exp.DataType) else None
            if target == exp.DataType.Type.DATE:
                return DATE
            if target == exp.DataType.Type.TIME:
                return TIME
            return NUMBER if target in _NUMERIC_CASTS else None
        if isinstance(node, (exp.Date, exp.TsOrDsToDate)) and not node.expressions and len([v for v in node.args.values() if v]) == 1:
            return DATE
        if isinstance(node, exp.Window):
            call = node.this
            if isinstance(call, _SHIFTS):
                default = call.args.get("default")
                return self._same([call.this] + ([default] if default is not None else []), select, depth)
            if isinstance(call, _PICKS):
                return self.kind(call.this, select, depth - 1)
            return NUMBER if isinstance(call, _NUMBER_CALLS) else None
        if isinstance(node, _PICKS):
            return self.kind(node.this, select, depth - 1)
        if isinstance(node, _NUMBER_CALLS):
            return NUMBER
        if isinstance(node, exp.Coalesce):
            return self._same([node.this, *node.expressions], select, depth)
        if isinstance(node, (exp.Least, exp.Greatest)):
            return self._same([node.this, *node.expressions], select, depth)
        if isinstance(node, exp.If):
            return self._same([node.args.get("true"), node.args.get("false") or exp.Null()], select, depth)
        if isinstance(node, exp.Case):
            results = [branch.args.get("true") for branch in node.args.get("ifs") or []]
            return self._same(results + [node.args.get("default") or exp.Null()], select, depth)
        if isinstance(node, exp.Subquery):
            branches = _branches(node)
            if branches is None or any(len(branch.expressions) != 1 or _is_star(branch.expressions[0]) for branch in branches):
                return None
            kinds = {self.kind(_unaliased(branch.expressions[0]), branch, depth - 1) for branch in branches}
            return kinds.pop() if len(kinds) == 1 else None
        return None


def _scope(node: exp.Expression) -> exp.Select | None:
    """The select whose sources ``node`` reads."""

    parent = node.parent
    while parent is not None and not isinstance(parent, exp.Select):
        if isinstance(parent, exp.SetOperation):
            return None  # the ORDER BY of a set operation reads its output columns
        parent = parent.parent
    return parent


def _number(node: exp.Expression, kind: str) -> exp.Expression:
    """MySQL's number for a date (its year, month and day as digits) or a time (hours, minutes, seconds)."""

    if kind == DATE:
        digits: exp.Expression = exp.Anonymous(this="STRFTIME", expressions=[node.copy(), exp.Literal.string("%Y%m%d")])
    else:
        text = exp.Cast(this=node.copy(), to=exp.DataType.build("VARCHAR"))
        digits = exp.Anonymous(this="REPLACE", expressions=[text, exp.Literal.string(":"), exp.Literal.string("")])
    return exp.Cast(this=digits, to=exp.DataType.build("BIGINT"))


def _guarded(node: exp.Expression) -> exp.Expression:
    """``node`` itself, in a form DuckDB refuses to bind when ``node`` is a date, a time or a timestamp.

    ``ABS`` takes numbers only, so DuckDB's binder rejects the query for any other operand. The call sits where
    it is never evaluated, and the value is ``node``'s own. An operand that is a string is refused as well; DuckDB
    refuses to add a string column to anything by itself.
    """

    never = exp.Case(ifs=[exp.If(this=exp.false(), true=exp.Is(this=exp.Abs(this=node.copy()), expression=exp.Null()))], default=exp.false())
    return exp.Case(ifs=[exp.If(this=never, true=exp.Null())], default=node.copy())


def _varies(node: exp.Expression) -> bool:
    """Whether the expression reads a column: MySQL converts a constant on its own terms."""

    return node.find(exp.Column) is not None


_DAY_TEXT = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _day_text_moved(node: exp.Expression) -> bool:
    """``'2024-01-31' + INTERVAL 1 DAY`` and ``DATE_ADD('2024-01-31', INTERVAL 1 MONTH)``: MySQL answers with the
    string ``'2024-02-01'``, which reads as the date does."""

    if isinstance(node, (exp.DateAdd, exp.DateSub)):
        text, unit = node.this, _unit(node)
    else:
        by = node.expression if _interval(node.expression) else node.this
        text, unit = (node.this if by is node.expression else node.expression), _unit(by)
    return isinstance(text, exp.Literal) and text.is_string and bool(_DAY_TEXT.match(text.this)) and unit in _DAY_UNITS


def _whole(node: exp.Expression | None, date) -> bool:
    """Whether a constant is a whole number: whole numbers and dates (read as numbers) added and subtracted.

    MySQL compares a DATE with such a constant as the numbers they are (``d = DATE('2019-07-05') - 1`` is the 4th,
    ``d < 20190800`` is every day of July, ``d = 20190732`` is no day). Not with a decimal (``d < 20190704.5`` is no
    day), and not in ``BETWEEN`` or an ``IN`` list, where one bound that is no date empties the test."""

    if node is None:
        return False
    if isinstance(node, exp.Paren):
        return _whole(node.this, date)
    if isinstance(node, exp.Neg):
        return _whole(node.this, date)
    if isinstance(node, exp.Literal):
        return not node.is_string and node.this.isdigit()
    if isinstance(node, (exp.Add, exp.Sub)):
        return all(_whole(side, date) or date(side) for side in (node.this, node.expression))
    return False


def _depth(node: exp.Expression) -> int:
    depth = 0
    while node.parent is not None:
        node, depth = node.parent, depth + 1
    return depth


def read(tree: exp.Expression, kinds: dict[str, dict[str, str | None]]) -> bool:
    """Rewrite ``tree`` (a MySQL query) in place as the module's header says. Returns whether a date or a time was
    written as a number, which is when the search should try the last and the first day of a month."""

    reader = _Reader({table.lower(): {column.lower(): kind for column, kind in columns.items()} for table, columns in kinds.items()})
    edits: dict[int, tuple[exp.Expression, str]] = {}  # by node identity: the node and "date", "time" or "guard"

    def convert(node: exp.Expression, kind: str) -> None:
        edits[id(node)] = (node, kind)

    def known(node: exp.Expression) -> str | None:
        return reader.kind(node, _scope(node))

    for node in tree.find_all(*_BINARY):
        operands = [node.this, node.expression]
        if any(_interval(operand) for operand in operands):
            continue
        for operand in operands:
            kind = known(operand)
            if kind in _TEMPORAL:
                convert(operand, kind)
            elif kind is None and isinstance(node, (exp.Add, exp.Sub)) and not isinstance(operand, (exp.Null, exp.Boolean)):
                edits.setdefault(id(operand), (operand, "guard"))

    def kept(node: exp.Expression) -> bool:
        """Not a number column of a derived table or ``WITH`` query. MySQL may look a date up in the key it builds
        for such a table, and then matches nothing where the comparison itself is true (``a.d = x.n`` as a join
        condition against ``(SELECT MIN(d) + 1 AS n ..) AS x``, MySQL 8.0.46): the date is left as it is, and
        DuckDB refuses the comparison."""

        while isinstance(node, exp.Paren):
            node = node.this
        return not isinstance(node, exp.Column) or reader.stored(node, _scope(node))

    def against(numbers: list[exp.Expression], others: list[exp.Expression]) -> None:
        """A date or a time among ``others`` is compared as a number when a number that varies is on the other side."""

        if any(known(node) == NUMBER and _varies(node) and kept(node) for node in numbers):
            for other in others:
                kind = known(other)
                if kind in _TEMPORAL:
                    convert(other, kind)

    def against_constant(number: exp.Expression, other: exp.Expression) -> None:
        """A DATE is compared as a number with a whole number that does not vary, too (not a date and time)."""

        if not _varies(number) and known(number) == NUMBER and known(other) == DATE and _whole(number, lambda side: known(side) == DATE and not _varies(side)):
            convert(other, DATE)

    for node in tree.find_all(*_COMPARISONS):
        against([node.this], [node.expression])
        against([node.expression], [node.this])
        against_constant(node.this, node.expression)
        against_constant(node.expression, node.this)
    for node in tree.find_all(exp.Between):
        parts = [node.this, node.args.get("low"), node.args.get("high")]
        if all(part is not None for part in parts):
            for part in parts:
                against([p for p in parts if p is not part], [part])
    for node in tree.find_all(exp.In):
        values = node.expressions
        query = node.args.get("query")
        if values:
            against([node.this], values)
            if all(known(value) == NUMBER for value in values):
                against(values, [node.this])
        elif query is not None:
            branches = _branches(query)
            left = node.this.expressions if isinstance(node.this, exp.Tuple) else [node.this]
            if branches is None or len(branches) != 1 or any(_is_star(item) for item in branches[0].expressions):
                continue
            items = branches[0].expressions
            if len(items) != len(left):
                continue
            for value, item in zip(left, items):
                inner = _unaliased(item)
                kind = reader.kind(inner, branches[0])
                if kind in _TEMPORAL and known(value) == NUMBER and _varies(value) and kept(value):
                    convert(inner, kind)

    # a date moved by whole days is a date in MySQL and a timestamp in DuckDB: 2024-02-01 against 2024-02-01 00:00:00
    # once it is an output or is written into a string
    for node in tree.find_all(exp.Add, exp.Sub, exp.DateAdd, exp.DateSub):
        moved = isinstance(node, (exp.DateAdd, exp.DateSub)) or any(_interval(side) for side in (node.this, node.expression))
        cast_already = isinstance(node.parent, exp.Cast) and node.parent.args["to"].is_type(exp.DataType.Type.DATE)
        if moved and id(node) not in edits and not cast_already and (known(node) == DATE or _day_text_moved(node)):
            edits[id(node)] = (node, "day")

    wrote = False
    for node, kind in sorted(edits.values(), key=lambda edit: -_depth(edit[0])):  # the innermost first
        if kind == "guard":
            node.replace(_guarded(node))
        elif kind == "day":
            node.replace(exp.Cast(this=node.copy(), to=exp.DataType.build("DATE")))
        else:
            replacement = _number(node, kind)
            if isinstance(node.parent, exp.Select) and node.arg_key == "expressions" and isinstance(node, exp.Column):
                replacement = exp.alias_(replacement, node.name, quoted=False)  # the output column keeps its name
            node.replace(replacement)
            wrote = True
    return wrote
