"""Decline to prove queries that compare a string with a number.

The provers read a value as a number, a string or a Boolean and call values of different kinds
unequal, so ``WHERE '2' <> 2`` filters nothing out and the query is proven equal to one without the
filter. No engine reads it that way:

- MySQL compares a string with a number as numbers (``'2' = 2`` is true, ``'abc' = 0`` is true);
- DuckDB and PostgreSQL cast the string to the number's type (``'2' = 2`` is true, ``'abc' = 2`` is an error);
- BigQuery rejects the comparison as a type error, so the query cannot run.

:func:`problem` names the first such comparison so the prover returns "not proven". It looks at
equality, ordering, ``BETWEEN``, ``IN`` lists, simple ``CASE`` operands and ``NULLIF``, comparing

- a string literal with a number literal, arithmetic, a numeric ``CAST``, a ``COUNT`` or a column declared numeric
  (``x + INTERVAL ..`` and ``x - INTERVAL ..`` are a date, a timestamp or an interval, so they are not read as numbers);
- a number literal (or the other numeric forms) with a column declared as a string;
- one column compared with a string literal in one place and a number literal in another (``a = 'abc' AND a = 0``
  is not empty on MySQL when ``a`` is an integer column, since ``'abc'`` reads as 0); the same for a value that hands a
  column on (``MIN(a)``, ``COALESCE(a, b)``) or that has no known type (``YEAR(d)``);
- a condition with a string literal (``(c = 2) = 'x'``: MySQL reads the condition as 0 or 1 and the string as a number).

A negated value is a number whatever is negated (``-x = 'abc'``).

The SMT prover reads a plain comparison of that kind as an opaque predicate of its two values, true or false whatever the
engine's conversion rule is, so a pair that uses one the same way on both sides can still be proven; :func:`problem` with
``plain_ok`` reports only the other forms and the same-column case. Nothing is folded, even where the engine's reading is exact (MySQL ``'2' = 2``).

The kind of a column the schema does not declare is inferred from the query (:class:`_Kinds`): a column compared with a
string literal reads as a string and one compared with a number as a number, columns equated by a join, ``USING``, ``IN``
or a derived table's projection share a kind, and a value that mixes kinds (``COALESCE(n, 'x')``, ``CASE`` branches,
``GREATEST``) is a string and a number at once. A comparison that sets one kind against the other, a column that
gathers both kinds, or a mixed value used as a comparison or arithmetic operand is declined, since the SMT prover
cannot tell the kinds apart for a column it has no type for and calls a string and a number unequal.

The strings themselves are checked too (:meth:`_Kinds.literal_problem`, :meth:`_Kinds.untyped_text_columns`), because the
provers read a string compared with a value of no declared type, or with a date, as text:

- a date written in more than one way (``d = '2019-01-01' AND d = '20190101'`` is one day twice in MySQL; DuckDB,
  PostgreSQL and BigQuery read ``'2019-1-1'`` as that day too). Several date strings compared with one value must all be
  different days that sort as text the way they sort as dates (:func:`date_value`), for a declared date and for a column
  of no type;
- on DuckDB and PostgreSQL, strings that are numbers compared with a column of no type (``n = '1' AND n = '01'`` holds the
  integer 1; ``n >= '9' AND n < '10'`` the integer 9);
- on MySQL the same cases are read both ways by ``numeric_column_reading``: two strings of one number, or strings whose
  order as text is not their order as numbers under an ordering.
"""

from __future__ import annotations

import datetime
import re

import sqlglot
from sqlglot import exp

_INTEGER = ("TINYINT", "SMALLINT", "MEDIUMINT", "INT", "INTEGER", "BIGINT", "INT64", "BYTEINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "INT2", "INT4", "INT8")
_FLOAT = ("FLOAT", "FLOAT64", "DOUBLE", "REAL", "FLOAT4", "FLOAT8")
_DECIMAL = ("NUMERIC", "DECIMAL", "BIGNUMERIC", "BIGDECIMAL", "DEC", "NUMBER")
_STRING = ("STRING", "VARCHAR", "TEXT", "CHAR", "NVARCHAR", "NCHAR", "BPCHAR", "ENUM")  # an ENUM is read as its string when compared with one
_ARITHMETIC = (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod, exp.IntDiv)
_NUMBER_NODES = _ARITHMETIC + (exp.Count, exp.Sum, exp.Avg, exp.Abs, exp.Round, exp.Floor, exp.Ceil, exp.Length, exp.Sqrt, exp.Pow, exp.Stddev, exp.Variance)
_STRING_NODES = tuple(getattr(exp, name) for name in ("Lower", "Upper", "Concat", "ConcatWs", "Substring", "Trim", "Initcap", "Replace", "Repeat", "Reverse"))
_MIXING = (exp.Coalesce, exp.Greatest, exp.Least, exp.If, exp.Case, exp.Nullif)
_SET_OPERATION = getattr(exp, "SetOperation", exp.Union)  # sqlglot before 25 has only Union
_COMPARISONS = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.NullSafeEQ, exp.NullSafeNEQ)
_ORDERINGS = (exp.LT, exp.LTE, exp.GT, exp.GTE)
_CONDITIONS = (exp.Predicate, exp.Connector, exp.Not, exp.Boolean)  # a value that is TRUE or FALSE


def _type_kind(type_sql: str) -> str | None:
    base = re.split(r"[\s(<]", type_sql.strip().upper(), maxsplit=1)[0]
    if base in _INTEGER + _FLOAT + _DECIMAL:
        return "number"
    if base in _STRING:
        return "string"
    return None


_TEMPORAL = ("DATE", "DATETIME", "TIMESTAMP", "TIMESTAMPTZ", "TIMESTAMP_NTZ", "TIMESTAMP_LTZ")
_STAMPS = _TEMPORAL[1:]  # the types that keep a time of day


def _is_temporal_type(type_sql: str, accepted: tuple[str, ...] = _TEMPORAL) -> bool:
    return re.split(r"[\s(<]", type_sql.strip().upper(), maxsplit=1)[0] in accepted


_DATE_VALUE = re.compile(r"^\s*(\d{4})([-/.])(\d{1,2})\2(\d{1,2})(?: (\d{2}):(\d{2}):(\d{2}))?\s*$")
_DATE_LIKE = re.compile(r"^\s*(\d{1,4}[^\d\s]\d{1,2}[^\d\s]\d{1,2}|\d{6})")


def date_value(text: str) -> tuple[int, ...] | str | None:
    """The moment a string is when an engine converts it to a date: ``(year, month, day, hour, minute, second)``.

    Read here: ``YYYY-MM-DD`` with one- or two-digit month and day, ``-``, ``/`` or ``.`` between the parts, an optional
    ``hh:mm:ss`` after one space, and spaces around the whole. Every engine that accepts such a string reads it as that
    moment (one that refuses it returns no rows to differ). ``"other"`` is a string an engine may read as a date in a way
    this function does not work out (``'20190101'``, ``'19-01-01'``, a fraction, trailing text, a day that does not
    exist: MySQL reads the first two as the first of January 2019). None is a string no engine reads as a date.
    """

    found = _DATE_VALUE.match(text)
    if found is None:
        return "other" if _DATE_LIKE.match(text) and not _TIME_FORM.match(text) else None  # '10:30:00' is a time of day
    parts = [int(part) for index, part in enumerate(found.groups()) if index != 1 and part is not None]
    parts += [0] * (6 - len(parts))
    try:
        datetime.datetime(*parts)
    except ValueError:
        return "other"  # '2019-02-30', '0000-00-00': an error in one engine and version, a zero date in another
    return tuple(parts)


_TIME_FORM = re.compile(r"^([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$")
_TIME_LIKE = re.compile(r"^\s*-?\d{1,3}:\d{1,2}")
_TRUE_WORDS = ("t", "true", "y", "yes", "on", "1")
_FALSE_WORDS = ("f", "false", "n", "no", "off", "0")


def _truth(text: str) -> bool | None:
    """The Boolean DuckDB and PostgreSQL read a string as when it is compared with a Boolean column, else None."""

    word = text.strip().lower()
    return True if word in _TRUE_WORDS else False if word in _FALSE_WORDS else None


_PLAIN_INTEGER = re.compile(r"^-?(0|[1-9][0-9]*)$")
_NUMBER_TEXT = re.compile(r"^\s*[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?\s*$")


def _order_differs(numbers: dict) -> bool:
    """Whether two of the strings are ordered one way as text and the other way as the numbers (or moments) they are read as."""

    texts = sorted(numbers)
    return any(numbers[a] > numbers[b] for i, a in enumerate(texts) for b in texts[i + 1 :])


def is_integer_type(type_sql: str) -> bool:
    """Whether a declared type is one of the integer types."""

    return re.split(r"[\s(<]", type_sql.strip().upper(), maxsplit=1)[0] in _INTEGER


def _data_type_kind(data_type: object) -> str | None:
    return _type_kind(data_type.sql()) if isinstance(data_type, exp.DataType) else None


def _listed_columns(schema: dict | None) -> dict[str, set[str]]:
    """Table name to the lower-cased column names the schema lists for it (tables it lists no columns for are left out)."""

    listed: dict[str, set[str]] = {}
    for name, columns in (schema or {}).items():
        if isinstance(columns, str) or not isinstance(columns, (list, tuple, set, frozenset, dict)):
            continue
        names = {column.lower() for column in columns if isinstance(column, str)}
        if names and len(names) == len(columns):
            listed[str(name).lower().split(".")[-1]] = names
    return listed


def _column_kinds(types: dict[str, dict[str, str]] | None) -> dict[str, str]:
    """Declared kind of each column name that has one kind in every declared table."""

    kinds: dict[str, set[str | None]] = {}
    for columns in (types or {}).values():
        for name, type_sql in columns.items():
            kinds.setdefault(name.lower(), set()).add(_type_kind(type_sql))
    return {name: found.pop() for name, found in kinds.items() if len(found) == 1 and None not in found}


def _kind(node: exp.Expression, columns: dict[str, str]) -> str | None:
    """``"number"``, ``"string"`` or None (not known) for an operand."""

    while isinstance(node, (exp.Paren, exp.Alias)):
        node = node.this
    if isinstance(node, exp.Neg):
        return "number"  # -x is a number whatever x is (MySQL reads a string operand as a number first)
    if isinstance(node, exp.Literal):
        return "string" if node.is_string else "number"
    if isinstance(node, exp.Cast):
        return _data_type_kind(node.args.get("to"))
    if _interval_arithmetic(node):
        return None
    if isinstance(node, _NUMBER_NODES):
        return "number"
    if isinstance(node, exp.Column):
        return columns.get(node.name.lower())
    return None


def _interval_arithmetic(node: exp.Expression) -> bool:
    """``x + INTERVAL ..`` or ``x - INTERVAL ..``: a date, a timestamp or an interval, never a number."""

    return isinstance(node, (exp.Add, exp.Sub)) and any(isinstance(_unwrap(side), exp.Interval) for side in (node.this, node.expression))


def _pairs(tree: exp.Expression):
    """``(left, right)`` operand pairs of every comparison in the tree."""

    for node in tree.walk():
        if isinstance(node, _COMPARISONS):
            yield node.this, node.expression
        elif isinstance(node, exp.Between):
            yield node.this, node.args["low"]
            yield node.this, node.args["high"]
        elif isinstance(node, exp.In):
            if node.args.get("query") is None:
                for item in node.expressions:
                    yield node.this, item
            else:
                for branch in _branches(node.args["query"]):
                    if branch.selects and not _is_star(branch.selects[0]):
                        yield node.this, branch.selects[0]
        elif isinstance(node, exp.Case) and node.args.get("this") is not None:
            for branch in node.args.get("ifs") or []:
                yield node.args["this"], branch.this
        elif isinstance(node, exp.Nullif):
            yield node.this, node.expression


def mismatched(node: exp.Expression, types: dict[str, dict[str, str]] | None = None) -> bool:
    """Whether a plain comparison (``=``, ``<>``, ``<``, ``<=``, ``>``, ``>=``, ``<=>``) sets a string against a number."""

    return isinstance(node, _COMPARISONS) and {_kind(node.this, _column_kinds(types)), _kind(node.expression, _column_kinds(types))} == {"string", "number"}


_MYSQL_NUMBER_PREFIX = re.compile(r"[ \t\n\r\f\v]*[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?")


def mysql_number(text: str) -> float:
    """The number MySQL reads a string as when it compares the string with a number (the leading numeric part, else 0)."""

    found = _MYSQL_NUMBER_PREFIX.match(text)
    try:
        return float(found.group(0)) if found else 0.0
    except ValueError:
        return 0.0


def _is_star(node: exp.Expression) -> bool:
    node = _unwrap(node)
    return isinstance(node, exp.Star) or (isinstance(node, exp.Column) and isinstance(node.this, exp.Star))


def _branches(query: exp.Expression) -> list[exp.Select]:
    """The ``SELECT`` blocks of a query, every side of a ``UNION``, ``INTERSECT`` or ``EXCEPT`` included."""

    if isinstance(query, exp.Subquery):
        return _branches(query.this)
    if isinstance(query, _SET_OPERATION):
        return _branches(query.this) + _branches(query.expression)
    return [query] if isinstance(query, exp.Select) else []


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, (exp.Paren, exp.Alias)):
        node = node.this
    return node


_VALUE_CALLS = (exp.Min, exp.Max, exp.AnyValue, exp.FirstValue, exp.LastValue, exp.Lag, exp.Lead, exp.NthValue)
_VALUE_WRAPPERS = (exp.Window, exp.Filter, exp.IgnoreNulls, exp.RespectNulls, exp.Nullif) + _VALUE_CALLS


def _named(*names: str) -> tuple[type, ...]:
    return tuple(getattr(exp, name) for name in names if hasattr(exp, name))


# calls whose result MySQL types as a string, and as a date or a time, whatever their arguments are
_TEXT_NODES = _STRING_NODES + _named(
    "Left", "Right", "Pad", "SubstringIndex", "TimeToStr", "GroupConcat", "Chr", "Hex", "MD5", "SHA", "SHA2", "NumberToStr", "Space", "Stuff", "Elt", "Dayname", "Monthname", "Collate"
)
_TEMPORAL_NODES = _named(
    "TsOrDsToDate", "Date", "DateAdd", "DateSub", "TsOrDsAdd", "DatetimeAdd", "DatetimeSub", "TimestampAdd", "TimestampSub", "CurrentDate", "CurrentTimestamp",
    "CurrentTime", "CurrentDatetime", "UtcDate", "UtcTimestamp", "StrToDate", "StrToTime", "LastDay", "Timestamp", "Time", "UnixToTime", "DateTrunc", "TsOrDsToTimestamp",
)


def _value_parts(node: exp.Expression) -> list[exp.Expression] | None:
    """The expressions one of whose values ``node`` returns as it is (``MIN(x)``: x; ``COALESCE(x, y)``: x and y; a scalar
    subquery: what it selects); None when ``node`` computes its value some other way."""

    if isinstance(node, _VALUE_WRAPPERS) and isinstance(node.this, exp.Expression):
        return [node.this] + ([node.args["default"]] if isinstance(node, (exp.Lag, exp.Lead)) and node.args.get("default") is not None else [])
    if isinstance(node, exp.Distinct) and len(node.expressions) == 1:
        return [node.expressions[0]]
    if isinstance(node, (exp.Coalesce, exp.Greatest, exp.Least)):
        return [node.this, *node.expressions]
    if isinstance(node, exp.If) and node.args.get("true") is not None:
        return [node.args["true"]] + ([node.args["false"]] if node.args.get("false") is not None else [])
    if isinstance(node, exp.Case):
        parts = [branch.args["true"] for branch in node.args.get("ifs") or [] if branch.args.get("true") is not None]
        return parts + ([node.args["default"]] if node.args.get("default") is not None else [])
    if isinstance(node, (exp.Subquery, exp.Select, _SET_OPERATION)):
        inner = node.this if isinstance(node, exp.Subquery) else node
        branches = _branches(inner) if isinstance(inner, exp.Query) else []
        if branches and all(len(b.selects) == 1 and not _is_star(b.selects[0]) for b in branches):
            return [b.selects[0] for b in branches]
    return None


def _handed_on(node: exp.Expression, depth: int = 0) -> tuple[list[exp.Column], bool]:
    """``(columns, only)``: the columns whose value ``node`` can return as it is, type included, and whether it can return
    nothing but one of those values or NULL.

    ``MIN(n)``, ``COALESCE(n, m)``, ``CASE WHEN c THEN n END``, ``NULLIF(n, 0)``, ``LAG(n) OVER (..)`` and a scalar subquery that
    selects ``n`` hand ``n`` on: MySQL gives the result the column's type, so ``MIN(n) = 'x'`` compares a value of ``n`` with
    ``'x'`` exactly as ``n = 'x'`` does, as numbers when ``n`` is an integer column.
    """

    node = _unwrap(node)
    if isinstance(node, exp.Column):
        return [node], True
    if isinstance(node, exp.Null):
        return [], True
    parts = _value_parts(node) if depth <= 6 else None
    if parts is None:
        return [], False
    columns: list[exp.Column] = []
    only = True
    for part in parts:
        found, rest = _handed_on(part, depth + 1)
        columns.extend(found)
        only = only and rest
    return columns, only


def _carried(node: exp.Expression) -> exp.Expression:
    """The one column whose value ``node`` hands on (:func:`_handed_on`) when it can return nothing else; ``node`` otherwise."""

    node = _unwrap(node)
    columns, only = _handed_on(node)
    if only and columns and len({(c.table.lower(), c.name.lower()) for c in columns}) == 1:
        return columns[0]
    return node


def _operand_text(node: exp.Expression) -> str:
    return _unwrap(node).sql().lower()


class _Kinds:
    """The kinds (``"string"``, ``"number"``) each column of a query can hold, from the query's own comparisons.

    Columns are keyed by ``(qualifier, name)``. A comparison between two columns, a ``USING`` list, an ``IN`` subquery
    and the projection of a derived table or CTE join their keys into one class; a comparison with a literal or another
    expression adds that expression's kinds to the class. Declared kinds join in only for a class with a column the
    schema leaves undeclared, since the SMT prover reads a comparison between two declared columns of different kinds
    as an unknown function and needs no help there.
    """

    def __init__(self, tree: exp.Expression, types: dict[str, dict[str, str]] | None, schema: dict | None = None):
        self.tree = tree
        self.types = {name.lower().split(".")[-1]: {c.lower(): t for c, t in cols.items()} for name, cols in (types or {}).items()}
        self.columns = _listed_columns(schema)
        self.by_name = _column_kinds(types)
        self.parent: dict[tuple[str, str], tuple[str, str]] = {}
        self.evidence: dict[tuple[str, str], set[str]] = {}
        self.literals: dict[tuple[str, str], set[str]] = {}
        self.members: dict[tuple[str, str], set[tuple[str, str]]] = {}
        self.names: dict[str, set[str]] = {}
        self.tables: dict[str, set[str]] = {}
        self.derived: dict[str, exp.Query] = {}
        self.operands: dict[tuple[str, str], exp.Expression] = {}  # the expression behind each class that is not a column
        self.feeds: dict[tuple[str, str], set[tuple[str, str]]] = {}  # derived output -> the columns its expression can hand on
        self.dated: set[tuple[str, str]] = set()  # the classes of expressions that are a date or a time whatever the columns hold
        self.ordered: set[tuple[str, str]] = set()  # keys read by an ordering: <, BETWEEN, MIN, MAX, GREATEST, LEAST, ORDER BY
        self.reused: set[str] = set()
        self.outputs: dict[tuple[str, str], dict[tuple[int, int], exp.Expression | list]] = {}  # derived output -> what each branch puts there: an expression, or the keys of the column it is
        self._collect_sources()
        self._build()

    def _collect_sources(self) -> None:
        for node in self.tree.walk():
            if isinstance(node, exp.Table) and node.name:
                self.tables.setdefault((node.alias or node.name).lower(), set()).add(node.name.lower())
            elif isinstance(node, (exp.CTE, exp.Subquery)) and node.alias and isinstance(node.this, exp.Query):
                if node.alias.lower() in self.derived:
                    self.reused.add(node.alias.lower())  # two derived tables under one name: their outputs share keys
                self.derived[node.alias.lower()] = node.this
            elif isinstance(node, exp.Column) and node.name:
                self.names.setdefault(node.name.lower(), set()).add(node.table.lower())

    def key(self, column: exp.Column) -> tuple[str, str]:
        return (column.table.lower(), column.name.lower())

    def find(self, key: tuple[str, str]) -> tuple[str, str]:
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a: tuple[str, str], b: tuple[str, str]) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb
            self.evidence.setdefault(rb, set()).update(self.evidence.pop(ra, set()))
            self.literals.setdefault(rb, set()).update(self.literals.pop(ra, set()))
            self.members[rb] = self.members.get(rb, {rb}) | self.members.pop(ra, {ra})

    def add(self, key: tuple[str, str], kinds: set[str]) -> None:
        self.evidence.setdefault(self.find(key), set()).update(kinds)

    def declared(self, key: tuple[str, str]) -> str | None:
        qualifier, name = key
        found = self.tables.get(qualifier) if qualifier else None
        if found and all(t in self.types for t in found):
            kinds = {_type_kind(self.types[t][name]) for t in found if name in self.types[t]}
            return kinds.pop() if len(kinds) == 1 else None
        return None if qualifier in self.derived else self._declared_by_name(name)

    def temporal(self, key: tuple[str, str], accepted: tuple[str, ...] = _TEMPORAL) -> bool:
        """Whether the column is declared a date or a timestamp (read like :meth:`declared`), or ``key`` is a dated expression.

        With ``accepted`` narrowed to ``_STAMPS``: whether the column is declared a type that keeps a time of day.
        """

        return self._temporal(key, accepted) is True

    def _temporal(self, key: tuple[str, str], accepted: tuple[str, ...] = _TEMPORAL) -> bool | None:
        """:meth:`temporal`, with None for a key that says nothing by itself, so the other columns of its class decide.

        Two keys say nothing. An output of a derived table or CTE that every branch fills with a column (``t.d AS x``,
        ``MIN(t.d) AS x``, ``*``): that column is in the same class. And a name under a qualifier whose tables are known
        not to have it: ``SELECT * FROM a JOIN b`` ties the output ``d`` to ``a.d`` and ``b.d``, and only one exists.
        An output filled with any other expression is a date only when the expression is one whatever the columns hold.
        A derived table whose name is used twice, or is also a table's, is never trusted.
        """

        if key in self.dated:
            return accepted is _TEMPORAL
        qualifier, name = key
        found = self.tables.get(qualifier) if qualifier and qualifier not in self.derived else None
        if found and all(t in self.types for t in found):
            declared = [self.types[t][name] for t in found if name in self.types[t]]
            if not declared and all(t in self.columns and name not in self.columns[t] for t in found):
                return None
            return bool(declared) and all(_is_temporal_type(type_sql, accepted) for type_sql in declared)
        if qualifier in self.derived:
            filled = self.outputs.get(key)
            shared = qualifier in self.reused or self.tables.get(qualifier, set()) - {qualifier} or qualifier in self.types or qualifier in self.columns
            if not filled or shared:  # the name is also another derived table or a real table: nothing is known
                return False
            if all(isinstance(inner, list) for inner in filled.values()):
                return None
            return accepted is _TEMPORAL and all(isinstance(inner, list) or self.fixed(inner) == "date" for inner in filled.values())
        declared = []
        for table in {table for tables in self.tables.values() for table in tables if table not in self.derived}:
            if name in self.types.get(table, {}):
                declared.append(self.types[table][name])
            elif table not in self.columns or name in self.columns[table]:
                return False
        return bool(declared) and all(_is_temporal_type(type_sql, accepted) for type_sql in declared)

    def origin(self, key: tuple[str, str], depth: int = 0) -> str | None:
        """``"string"``, ``"number"`` or ``"date"`` when the column is declared so, itself or through the derived tables and
        CTEs that hand it on unchanged (``t.d AS x``, ``MIN(t.d) AS x``, ``*``); None when nothing declares it.

        Unlike a class, this follows no comparison: ``a.name = s.d`` leaves ``a.name`` a string and ``s.d`` a date.
        """

        if key in self.operands:
            return None
        declared = self.declared(key)
        if declared:
            return declared
        answer = self._temporal(key)
        if answer is True:
            return "date"
        if answer is False or key[0] not in self.derived or depth > 6:
            return None
        found = set()
        for filled in self.outputs[key].values():
            # ``*`` over a join names the column under every source; the sources known not to have it say nothing
            keys = [k for k in filled if k[0] in self.derived or self._temporal(k) is not None or self.declared(k)]
            found.update(self.origin(k, depth + 1) for k in keys or filled)
        return found.pop() if len(found) == 1 else None

    def text_and_date_problem(self) -> str | None:
        """Why a comparison of a text value with a date cannot be read, else None.

        The provers read a date as its ISO text, so ``v.s >= CAST('2019-06-28' AS DATE)`` and ``v.s >= '2019-06-28'`` look
        the same to them. For a text column they are not: MySQL converts the column's text to a date for the first
        (``'2019-6-3'`` is the third of June, and sorts after ``'2019-06-28'`` as text; MySQL 8.0.46), and DuckDB,
        PostgreSQL and BigQuery refuse it. The same holds for a text column compared with a date column, and for a text expression
        (``CONCAT``) compared with either. A string literal is not a text value here: :meth:`literal_problem` reads it.
        """

        def side(node: exp.Expression) -> str | None:
            node = _unwrap(node)
            if isinstance(node, (exp.Literal, exp.Null)):
                return None
            carried = _carried(node)
            if isinstance(carried, exp.Column):
                return self.origin(self.key(carried))
            return self.fixed(node) or (self.fixed(carried) if carried is not node else None)

        for left, right in _pairs(self.tree):
            if {side(left), side(right)} == {"string", "date"}:
                return "a text value is compared with a date (the engines convert the text to a date or refuse the comparison; the prover reads the date as text)"
        return None

    def _declared_by_name(self, name: str) -> str | None:
        """The declared kind of a column known by its name alone: the one kind the tables this query reads give it.

        A table the query does not read says nothing about the name (another table's ``s VARCHAR`` does not type
        ``t.s``). A table it does read may be the one that holds the column, in this select or in another one of the
        query, unless the schema lists that table's columns and the name is not among them. So the name has a declared
        kind only when every table read either declares it, with one kind, or is known not to have it. Types alone do
        not say a table lacks a column: a caller may declare only some columns (the text columns of a table).
        """

        read = {table for tables in self.tables.values() for table in tables if table not in self.derived}
        kinds = set()
        for table in read:
            declared = self.types.get(table, {})
            if name in declared:
                kinds.add(_type_kind(declared[name]))
            elif table not in self.columns or name in self.columns[table]:
                return None
        return kinds.pop() if len(kinds) == 1 else None

    def kinds(self, node: exp.Expression, depth: int = 0) -> set[str]:
        """Kinds ``node`` can hold; empty when nothing is known."""

        node = _unwrap(node)
        if isinstance(node, exp.Neg):
            return {"number"}
        if depth > 6:
            return set()
        if isinstance(node, exp.Literal):
            return {"string" if node.is_string else "number"}
        if isinstance(node, exp.Cast):
            kind = _data_type_kind(node.args.get("to"))
            return {kind} if kind else set()
        if _interval_arithmetic(node):
            return set()
        if isinstance(node, _NUMBER_NODES):
            return {"number"}
        if isinstance(node, _STRING_NODES):
            return {"string"}
        if isinstance(node, exp.Column):
            key = self.key(node)
            root = self.find(key)
            found = set(self.evidence.get(root, set()))
            members = self.members.get(root, {key})
            if all(self.declared(m) for m in members):
                members = {key}  # every column the schema types: declared kinds do not join what the text shows
            return found | {kind for kind in (self.declared(m) for m in members) if kind}
        if isinstance(node, (exp.Coalesce, exp.Greatest, exp.Least)):
            return set().union(*(self.kinds(a, depth + 1) for a in [node.this, *node.expressions]))
        if isinstance(node, exp.If) and node.args.get("true") is not None:
            return self.kinds(node.args["true"], depth + 1) | (self.kinds(node.args["false"], depth + 1) if node.args.get("false") is not None else set())
        if isinstance(node, exp.Case):
            branches = [branch.args["true"] for branch in node.args.get("ifs") or [] if branch.args.get("true") is not None]
            if node.args.get("default") is not None:
                branches.append(node.args["default"])
            return set().union(*(self.kinds(b, depth + 1) for b in branches))
        if isinstance(node, (exp.Nullif, exp.Max, exp.Min, exp.AnyValue)):
            return self.kinds(node.this, depth + 1)
        if isinstance(node, (exp.Subquery, exp.Select, _SET_OPERATION)):
            inner = node.this if isinstance(node, exp.Subquery) else node
            branches = _branches(inner) if isinstance(inner, exp.Query) else []
            if branches and all(len(b.selects) == 1 and not _is_star(b.selects[0]) for b in branches):
                return set().union(*(self.kinds(b.selects[0], depth + 1) for b in branches))
        return set()

    def _build(self) -> None:
        for _ in range(4):  # a kind found late flows to the expressions built on it
            self._unqualified()
            self._same_table()
            self._unqualified_derived()
            self._projections()
            self._comparisons()
            self._using()
        self._orderings()

    def _unqualified(self) -> None:
        """A column written without its table is one of the columns of that name the query qualifies elsewhere."""

        for name, qualifiers in self.names.items():
            if "" in qualifiers:
                for qualifier in qualifiers - {""}:
                    self.union(("", name), (qualifier, name))

    def _same_table(self) -> None:
        """Two names for one table read the same column: ``a.n`` and ``b.n`` of ``t AS a JOIN t AS b`` have one type.

        They can also be one row (a key joins a row to itself), so ``a.n = 'x' AND b.n = 'y'`` is a column compared with
        two strings as much as ``n = 'x' AND n = 'y'`` is.
        """

        for name, qualifiers in self.names.items():
            first: dict[str, str] = {}
            for qualifier in sorted(qualifiers - {""}):
                tables = self.tables.get(qualifier)
                if qualifier in self.derived or not tables or len(tables) != 1:
                    continue
                (table,) = tables
                if table in first:
                    self.union((first[table], name), (qualifier, name))
                else:
                    first[table] = qualifier

    def _exposed(self, select: exp.Select) -> list[tuple[str, set[str]]] | None:
        """``(alias, column names)`` of every source of ``select``; None when some source's columns are not known."""

        source = select.args.get("from_") or select.args.get("from")
        nodes = ([source.this] if source else []) + [join.this for join in select.args.get("joins") or []]
        if select.args.get("laterals"):
            return None
        found = []
        for node in nodes:
            alias = (node.alias_or_name or "").lower() if isinstance(node, (exp.Table, exp.Subquery)) else ""
            if not alias or (isinstance(node.args.get("alias"), exp.TableAlias) and node.args["alias"].args.get("columns")):
                return None
            if isinstance(node, exp.Subquery) and isinstance(node.this, exp.Query):
                names = {name.lower() for name in node.this.named_selects}
            elif isinstance(node, exp.Table) and not node.args.get("db") and node.name.lower() in self.derived and alias == node.name.lower():
                names = {name.lower() for name in self.derived[alias].named_selects}
            elif isinstance(node, exp.Table) and node.name.lower() in self.types and node.name.lower() not in self.derived:
                names = set(self.types[node.name.lower()])
            else:
                return None
            if "*" in names or "" in names:
                return None
            found.append((alias, names))
        return found

    def _unqualified_derived(self) -> None:
        """A column written without its table is the output column of a derived table or CTE when, in every select that
        reads the name that way, that one source alone can hold it.

        ``SELECT c FROM (SELECT name AS c FROM dept) AS d WHERE c > 'a' AND c < 'z'`` never writes ``d.c``, so
        :meth:`_unqualified` has nothing to tie ``c`` to and it looked like a column of no declared type. The name is
        resolved select by select; a select alias of that name, a source whose columns are not known, a name two sources
        hold or none does (an enclosing query's column), or two selects that read it from different sources leave it
        untied, as before.
        """

        holders: dict[str, set[str] | None] = {}
        for select in self.tree.find_all(exp.Select):
            bare = {
                c.name.lower()
                for c in select.find_all(exp.Column)
                if not c.table and c.name and not isinstance(c.this, exp.Star) and c.find_ancestor(exp.Select) is select
            }
            if not bare:
                continue
            exposed = self._exposed(select)
            aliases = {
                item.alias.lower()
                for item in select.expressions
                if isinstance(item, exp.Alias) and not (isinstance(item.this, exp.Column) and not item.this.table and item.this.name.lower() == item.alias.lower())
            }
            for name in bare:
                sources = [alias for alias, names in exposed if name in names] if exposed is not None else []
                if len(sources) != 1 or name in aliases or holders.get(name, set()) is None:
                    holders[name] = None
                else:
                    holders.setdefault(name, set()).add(sources[0])
        for name, sources in holders.items():
            if sources and len(sources) == 1:
                alias = next(iter(sources))
                if alias in self.derived:
                    self.union(("", name), (alias, name))

    def _sources(self, select: exp.Select) -> list[str]:
        source = select.args.get("from_") or select.args.get("from")
        nodes = ([source.this] if source else []) + [join.this for join in select.args.get("joins") or []]
        return [(n.alias or n.name).lower() for n in nodes if isinstance(n, (exp.Table, exp.Subquery)) and (n.alias or n.name)]

    def _projections(self) -> None:
        """Each output column of a derived table or CTE is the column or expression in that position of every branch."""

        for alias, query in self.derived.items():
            names = [name.lower() for name in query.named_selects]
            for branch in _branches(query):
                sources = self._sources(branch)
                for position, expr in enumerate(branch.selects):
                    inner = _unwrap(expr)
                    if _is_star(inner):
                        stars = [inner.table.lower()] if isinstance(inner, exp.Column) and inner.table else sources
                        for name, qualifiers in self.names.items():
                            if alias in qualifiers:
                                self.outputs.setdefault((alias, name), {})[(id(branch), position)] = [(source, name) for source in stars]
                                for source in stars:
                                    self.union((alias, name), (source, name))
                    elif "*" not in names and position < len(names):
                        node = (alias, names[position])
                        carried = _carried(inner)
                        self.outputs.setdefault(node, {})[(id(branch), position)] = [self.key(carried)] if isinstance(carried, exp.Column) else inner
                        if isinstance(carried, exp.Column):
                            self.union(node, self.key(carried))  # MIN(t.s) AS m is a value of t.s, as t.s AS m is
                        else:
                            self.add(node, self.kinds(inner))
                            self.feeds.setdefault(node, set()).update(self.key(column) for column in _handed_on(inner)[0])

    def _comparisons(self) -> None:
        for left, right in _pairs(self.tree):
            for operand, other in ((left, right), (right, left)):
                self._relate(_carried(operand), _carried(other))
                other = _unwrap(other)
                if isinstance(other, exp.Literal) and other.is_string:
                    for key in self.operand_keys(operand):
                        self.literals.setdefault(self.find(key), set()).add(other.this)

    def _orderings(self) -> None:
        for node in self.tree.walk():
            if isinstance(node, _ORDERINGS):
                operands = [node.this, node.expression]
            elif isinstance(node, exp.Between):
                operands = [node.this, node.args["low"], node.args["high"]]
            elif isinstance(node, (exp.Min, exp.Max, exp.Ordered)):
                operands = [node.this]
            elif isinstance(node, (exp.Greatest, exp.Least)):
                operands = [node.this, *node.expressions]
            elif isinstance(node, _named("ArgMin", "ArgMax")):
                operands = [node.expression]
            else:
                continue
            for operand in operands:
                if not isinstance(_unwrap(operand), exp.Literal):
                    self.ordered.update(self.operand_keys(operand))

    def _open(self, root: tuple[str, str]) -> set[tuple[str, str]] | None:
        """The members of a class no declared type settles as text or as a number; None for a settled class."""

        members = self.members.get(root, {root})
        return None if any(key not in self.operands and self.declared(key) for key in members) else members

    def _dated(self, members: set[tuple[str, str]], accepted: tuple[str, ...] = _TEMPORAL) -> bool:
        """Whether every column of the class is declared a date or a timestamp (or the class is a dated expression)."""

        answers = [self._temporal(key, accepted) for key in members if key not in self.operands or key in self.dated]
        answers = [answer for answer in answers if answer is not None]
        return bool(answers) and all(answers)

    def operand_keys(self, operand: exp.Expression) -> list[tuple[str, str]]:
        """The classes a string compared with ``operand`` is compared with: the column it is or hands on, else every column
        it can hand on and the expression itself (keyed by its text, so the same expression is one class)."""

        node = _carried(operand)
        if isinstance(node, exp.Column):
            keys = [self.key(node)]
        elif self.fixed(node) not in (None, "date"):
            return []  # compared in a known way, whatever the columns under it hold
        else:
            key = ("", "\x00" + _operand_text(node))
            self.operands.setdefault(key, node)
            if self.fixed(node) == "date":  # the engine converts the string to a date: literal_problem() says when that is safe
                self.dated.add(key)
            keys = [key] + [self.key(column) for column in _handed_on(node)[0]]
        seen = set(keys)
        for key in keys:  # an output of a derived table that can hand on several columns: a value of each of them
            for member in self.members.get(self.find(key), {key}):
                for fed in self.feeds.get(member, ()):
                    if fed not in seen:
                        seen.add(fed)
                        keys.append(fed)
        return keys

    def fixed(self, node: exp.Expression, depth: int = 0) -> str | None:
        """``"string"``, ``"number"``, ``"date"`` (a date or a timestamp) or ``"other"`` (a time, bytes, JSON) when MySQL's
        type for ``node`` is that whatever the untyped columns hold; None when it depends on them.

        A string is compared with such a value in a known way: as strings, as dates, or as numbers (which :func:`problem`
        reports). So two different strings compared with it are not the open case :meth:`untyped_text_columns` looks for.
        """

        node = _unwrap(node)
        if isinstance(node, exp.Literal):
            return "string" if node.is_string else "number"
        if _interval_arithmetic(node) or isinstance(node, _TEMPORAL_NODES):
            return "date"
        if isinstance(node, exp.Anonymous) and str(node.this).lower().startswith("kumo"):
            return "other"  # the prover's own placeholder (KUMOSQL_SCALAR(0)): the value it stands for was read at the entry
        if isinstance(node, (exp.Neg, exp.Boolean) + _NUMBER_NODES):
            return "number"
        if isinstance(node, _TEXT_NODES):
            return "string"
        if isinstance(node, exp.Cast):
            to = node.args.get("to")
            return _data_type_kind(to) or ("date" if isinstance(to, exp.DataType) and _is_temporal_type(to.sql()) else "other")
        if isinstance(node, exp.Column):
            return self.declared(self.key(node)) or ("date" if self.temporal(self.key(node)) else None)
        parts = _value_parts(node) if depth <= 6 else None
        if parts is None:
            return None
        found = [self.fixed(part, depth + 1) for part in parts if not isinstance(_unwrap(part), exp.Null)]
        if "string" in found and isinstance(node, (exp.Coalesce, exp.If, exp.Case, exp.Greatest, exp.Least)):
            return "string"  # MySQL types a choice between a string and anything else as a string
        return found[0] if found and None not in found and len(set(found)) == 1 else None

    def _relate(self, column: exp.Expression, other: exp.Expression) -> None:
        if not isinstance(column, exp.Column):
            return
        key = self.key(column)
        if isinstance(other, exp.Column):
            # two declared columns of different kinds are the SMT prover's opaque comparison, not an inferred one
            if not (self.declared(key) and self.declared(self.key(other))):
                self.union(key, self.key(other))
        else:
            self.add(key, self.kinds(other))
            if isinstance(other, exp.Literal) and other.is_string:
                self.literals.setdefault(self.find(key), set()).add(other.this)

    def untyped_text_columns(self) -> set[tuple[str, str]]:
        """Roots of the column classes with no declared type whose strings MySQL would read another way as numbers.

        ``n = 'x' AND n = 'y'`` is empty for a text column and not for an integer column that holds 0, since MySQL reads
        both strings as 0. ``n >= '9' AND n < '10'`` is empty for a text column (``'10'`` sorts before ``'9'``) and holds
        the integer 9. So a class is returned when two of its strings are one number, or when the class is read by an
        ordering (``<``, ``BETWEEN``, ``MIN``, ``ORDER BY``) and two of its strings are ordered one way as text and the
        other way as numbers. With neither, the strings are as different and as ordered for an integer column as for a
        text one, and a proof for the one holds for the other. A class whose columns are all declared dates is not
        returned: MySQL compares a date with a string as dates (:meth:`literal_problem`).
        """

        ordered = {self.find(key) for key in self.ordered}
        found = set()
        for root, literals in self.literals.items():
            root = self.find(root)
            members = self._open(root)
            if members is None or self._dated(members):
                continue
            numbers = {text: mysql_number(text) for text in literals}
            if len(set(numbers.values())) < len(numbers) or (root in ordered and _order_differs(numbers)):
                found.add(root)
        return found

    def literal_problem(self, dialect: str) -> str | None:
        """Why the strings one class is compared with cannot be read as text, else None.

        A column with no declared type may be a date, and a declared date is one: every engine converts the string, and
        reads more than one spelling as the same day. Several date strings compared with one value are safe together when
        :func:`date_value` reads each of them, no two are the same moment, and, under an ordering, they sort as text the
        way they sort as moments (``'2019-10-1'`` sorts before ``'2019-9-1'``): then a table of dates can be written as a
        table of texts on which every comparison comes out the same, so a proof for text is a proof for dates. For a
        class that may be a ``DATE`` column the day is the moment, since DuckDB reads ``d = '2019-01-01 10:00:00'`` for
        such a column as the day; only columns declared a type that keeps the time compare the times.

        A time is read the same way: several strings that look like times must all be written ``hh:mm:ss``.

        DuckDB and PostgreSQL convert a string compared with a number column to the column's type, so for a column with
        no declared type ``n = '1' AND n = '01'`` holds the integer 1, and ``n >= '9' AND n < '10'`` the integer 9. When
        every string of a class is a number there (one that is not would be an error for a number column), two or more
        strings are declined unless all are plain integers (``'12'``, not ``'012'`` or ``'1.5'``: different texts are then
        different numbers) and, under an ordering, sorted the same way as text. One string alone is one point, and reads
        the same either way. MySQL's reading is :meth:`untyped_text_columns`; BigQuery refuses a number compared with a
        string. The same two engines read ``'yes'``, ``'true'`` and ``'1'`` as one value for a Boolean column, so strings
        that are all such words, two of them one truth value, are declined there as well.
        """

        ordered = {self.find(key) for key in self.ordered}
        for root, literals in self.literals.items():
            root = self.find(root)
            members = self._open(root)
            if members is None:
                continue
            dates = {text: value for text in literals if (value := date_value(text)) is not None}
            if len(dates) > 1:
                # a date column drops a time of day (DuckDB), so there two strings of one day are one value
                moments = None if "other" in dates.values() else {text: value if self._dated(members, _STAMPS) else value[:3] for text, value in dates.items()}
                if moments is None or len(set(moments.values())) < len(moments) or (root in ordered and _order_differs(moments)):
                    return "one value is compared with dates written in more than one way (the engines convert each string to a date; the prover reads them as text)"
            times = [text for text in literals if _TIME_LIKE.match(text)]
            if len(times) > 1 and not all(_TIME_FORM.match(text) for text in times):
                return "one value is compared with times written in more than one way (the engines convert each string to a time; the prover reads them as text)"
            if dialect in ("mysql", "bigquery") or self._dated(members):
                continue
            if all(_NUMBER_TEXT.match(text) for text in literals):
                numbers = {text: float(text) for text in literals}
                plain = all(_PLAIN_INTEGER.match(text) for text in literals)
                if len(literals) > 1 and (not plain or (root in ordered and _order_differs(numbers))):
                    return "a column with no declared type is compared with strings that are numbers (a number column reads them as numbers; the prover reads them as text)"
            truths = [_truth(text) for text in literals]
            if None not in truths and len(set(truths)) < len(truths):
                return "a column with no declared type is compared with strings that are one truth value (a Boolean column reads 'yes' and 'true' alike; the prover reads them as text)"
        return None

    def _using(self) -> None:
        for select in self.tree.find_all(exp.Select):
            joins = select.args.get("joins") or []
            source = select.args.get("from_") or select.args.get("from")
            names = [source.this if source else None] + [j.this for j in joins]
            aliases = [(n.alias or n.name).lower() for n in names if isinstance(n, (exp.Table, exp.Subquery))]
            for join in joins:
                for ident in join.args.get("using") or []:
                    name = ident.name.lower()
                    for other in aliases[1:]:
                        self.union((aliases[0], name), (other, name))


def problem(
    sql: str, dialect: str = "bigquery", types: dict[str, dict[str, str]] | None = None, plain_ok: bool = False, schema: dict | None = None
) -> str | None:
    """A description of the first string-versus-number comparison in ``sql``, else None.

    With ``plain_ok`` a plain two-operand comparison of operands whose kinds the schema or the text settles is not reported:
    the SMT prover reads it as an opaque predicate of its operands (:meth:`kumo_sql_prover.smt_equivalence._Compiler._compare`),
    which is right for any conversion rule.
    """

    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:  # noqa: BLE001 - the prover reports the parse error itself
        return None
    columns = _column_kinds(types)
    kinds = _Kinds(tree, types, schema)
    literal_kinds: dict[str, set[str]] = {}
    for left, right in _pairs(tree):
        if any(isinstance(_unwrap(a), _CONDITIONS) and isinstance(_unwrap(b), exp.Literal) and _unwrap(b).is_string for a, b in ((left, right), (right, left))):
            # MySQL reads the condition as 0 or 1 and the string as a number ((c = 1) = 'x' is true where c <> 1); the others refuse it
            return "a condition is compared with a string (MySQL reads both as numbers; the prover does not model it)"
        sides = {_kind(left, columns), _kind(right, columns)}
        plain = plain_ok and isinstance(left.parent, _COMPARISONS)
        if sides == {"string", "number"} and not plain:
            return "a string is compared with a number (the engines convert the string; the prover does not model it)"
        found = kinds.kinds(left) | kinds.kinds(right)
        if found == {"string", "number"} and not (plain and None not in sides):
            return "a string is compared with a number (the engines convert the string; the prover does not model it)"
        for column, other in ((left, right), (right, left)):
            if not isinstance(other, exp.Literal):
                continue
            carried = column if isinstance(column, exp.Column) else _carried(column)
            # MIN(a) = 'abc' AND MIN(a) = 0 is a = 'abc' AND a = 0 for the row that holds the minimum; an expression the
            # prover has no type for (YEAR(d), a CASE over untyped columns) can be a number for both comparisons too
            if isinstance(carried, exp.Column) or kinds.fixed(carried) in (None, "date"):
                literal_kinds.setdefault(carried.sql().lower(), set()).add("string" if other.is_string else "number")
    for name, found in literal_kinds.items():
        if len(found) > 1:
            return f"column {name} is compared with both a string and a number (the engines convert the string; the prover does not model it)"
    for node in tree.walk():
        if isinstance(node, _MIXING):
            if kinds.kinds(node) == {"string", "number"} and _is_operand(node):
                return "a value of mixed kinds (a string in one branch, a number in another) is compared or used in arithmetic (the engines convert it; the prover does not model it)"
    return kinds.text_and_date_problem() or kinds.literal_problem(dialect)


def _is_operand(node: exp.Expression) -> bool:
    """Whether ``node`` is read as a value by a comparison, arithmetic or an aggregate that adds."""

    parent = node.parent
    while isinstance(parent, (exp.Paren, exp.Neg)):
        node, parent = parent, parent.parent
    if isinstance(parent, exp.Case):
        return parent.this is node
    return isinstance(parent, _COMPARISONS + _ARITHMETIC + (exp.Between, exp.In, exp.Sum, exp.Avg))
