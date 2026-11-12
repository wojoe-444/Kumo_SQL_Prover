"""Small databases directed by two queries' own text, to separate a pair that is not equivalent.

``tools/quite_bench.py`` calls ``find_witness`` after its other sources of databases found nothing. The
search reads both queries (never a label) and builds many tiny databases from what they compare:

* every constant a column is compared with (``=``, ``IN``, ``BETWEEN``, ``<``, ``LIKE``), its neighbours on
  both sides, and NULL where the column allows it. A constant compared with an expression over several
  columns (``x / y BETWEEN 2 AND 5``) goes to each of them, with its negative;
* columns compared with each other (join columns, ``IN (subquery)``) draw from one small shared set of
  values, so joins match, rows repeat and a grouped subquery has several rows per join value;
* a count compared with a number (``HAVING COUNT(*) > 4``) asks for groups larger than that number;
* columns compared within a factor (``a BETWEEN 0.9 * b AND 1.1 * b``) hold 10 and 10 times each factor;
* the constant a subquery read as one value filters on (``= (SELECT m FROM d WHERE d.day = '..')``) is in
  one row only, since a second row is an error;
* half of each table's rows are copies of another row with one or two columns drawn again: a row that
  passes every filter next to one that fails exactly one.

Rows respect the schema: NOT NULL columns, the declared lengths, precisions and keys, and foreign keys
(a child row copies the columns of a parent row). The databases are legal by construction and are still
passed through the harness's ``repair`` and ``legal``.

What counts as a witness is decided by the harness, not here: ``Replayer.differs`` replays the final
database (different bags, DuckDB's unoptimized plan agreeing, three row orders when rows can depend on
input order). This module only adds conditions, each of which rejects a database:

* **No division by zero.** PostgreSQL raises an error where DuckDB returns NULL, so a database on which a
  divisor is zero says nothing about PostgreSQL. Every divisor ``b`` is replayed as
  ``CASE WHEN b = 0 THEN error(..) ELSE b END`` (a query that raises is no evidence), and a column read
  inside a divisor only ever holds 1, -1, 2 or -2, or the powers of two inside the bounds the queries
  give it (dividing by them is exact in floating point too).
* **No tie-breaking.** A pair whose rows can depend on input order is searched only when the one source of
  that dependence is a ``LIMIT`` on the outermost query, and a database counts only if neither query
  returns more rows than its ``LIMIT`` when run without it: the ``LIMIT`` then cuts nothing. Pairs with
  ``ROW_NUMBER``, ``STRING_AGG``, a nested ``LIMIT`` and the like are not searched.
* **A bounded check.** The witness is first reduced row by row, so the unoptimized replay (a cross product
  for a comma join) stays small, and that replay runs under the per-query time limit.

Numbers are whole or have the column's scale, never more digits, so sums are exact; a comparison that is an
exact tie only after several rounded floating-point steps (``x > 1.2 * AVG(y)`` with ``x`` exactly on the
line) can still read differently in DuckDB and PostgreSQL. That exposure is the harness's own (DuckDB is its
oracle) and is not widened here beyond drawing more databases.
"""

from __future__ import annotations

import datetime
import random
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

import sqlglot
from sqlglot import exp

SEED = 20261111
_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NullSafeEQ, exp.NullSafeNEQ)
_OPERATORS = {exp.EQ: "eq", exp.NullSafeEQ: "eq", exp.NEQ: "neq", exp.NullSafeNEQ: "neq", exp.GT: "above", exp.GTE: "from", exp.LT: "below", exp.LTE: "to"}
_DIVISIONS = (exp.Div, exp.IntDiv, exp.Mod)
_DIVISORS = (1, 2, -1, -2)
_NOT_CONSTANT = (exp.Column, exp.Subquery, exp.Select, exp.AggFunc, exp.Star, exp.Window, exp.Placeholder, exp.Parameter)
_BASE_DATE = datetime.date(2000, 1, 1)


# --- values -----------------------------------------------------------------------------------


def _kind(pg_type: str) -> str:
    """``int``, ``decimal``, ``float``, ``date``, ``timestamp``, ``bool`` or ``text`` for a PostgreSQL type."""

    base = pg_type.split("(")[0].strip().upper()
    if base in {"INT", "INTEGER", "SMALLINT", "BIGINT", "INT4", "INT8", "INT2", "SERIAL", "BIGSERIAL"}:
        return "int"
    if base in {"DECIMAL", "NUMERIC"}:
        return "decimal"
    if base in {"DOUBLE PRECISION", "DOUBLE", "FLOAT", "REAL", "FLOAT8", "FLOAT4"}:
        return "float"
    if base == "DATE":
        return "date"
    if base.startswith("TIMESTAMP") or base == "DATETIME":
        return "timestamp"
    if base in {"BOOLEAN", "BOOL"}:
        return "bool"
    return "text"


def _numbers(pg_type: str) -> tuple[int, int]:
    """``(precision, scale)`` of a DECIMAL type; ``(18, 3)`` when it gives none (a value both engines hold exactly)."""

    found = re.search(r"\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)", pg_type)
    if not found:
        return 18, 3
    return int(found.group(1)), int(found.group(2) or 0)


def _length(pg_type: str) -> int | None:
    found = re.search(r"\(\s*(\d+)\s*\)", pg_type)
    return int(found.group(1)) if found and _kind(pg_type) == "text" else None


def fit(value, pg_type: str):
    """``value`` as a value the column can hold exactly, or ``None`` when it cannot."""

    kind = _kind(pg_type)
    try:
        if value is None or isinstance(value, (list, dict, bytes)):
            return None
        if kind == "bool":
            return value if isinstance(value, bool) else None
        if isinstance(value, bool):
            return None
        if kind == "int":
            number = Decimal(str(value))
            return int(number) if number == number.to_integral_value() and abs(number) < 2**31 else None
        if kind == "decimal":
            precision, scale = _numbers(pg_type)
            number = Decimal(str(value))
            exact = number.quantize(Decimal(1).scaleb(-scale))
            return exact if exact == number and abs(exact) < Decimal(10) ** (precision - scale) else None
        if kind == "float":
            number = float(value)
            return number if number == number and abs(number) < 1e15 else None
        if kind == "date":
            if isinstance(value, datetime.datetime):
                return value.date() if value.time() == datetime.time(0) else None
            if isinstance(value, datetime.date):
                return value
            return datetime.date.fromisoformat(str(value).strip()[:10]) if isinstance(value, str) else None
        if kind == "timestamp":
            if isinstance(value, datetime.datetime):
                return value.replace(tzinfo=None)
            if isinstance(value, datetime.date):
                return datetime.datetime(value.year, value.month, value.day)  # noqa: DTZ001
            return datetime.datetime.fromisoformat(str(value).strip()).replace(tzinfo=None) if isinstance(value, str) else None
        if isinstance(value, (datetime.date, datetime.datetime)):
            value = value.isoformat()
        text = value if isinstance(value, str) else str(value)
        limit = _length(pg_type)
        return text if (limit is None or len(text) <= limit) and text == text.rstrip(" ") else None
    except (ValueError, ArithmeticError, InvalidOperation, TypeError, OverflowError):
        return None


def _neighbours(value, pg_type: str) -> list:
    """The values next to ``value`` on both sides, in the column's type (unfitted)."""

    kind = _kind(pg_type)
    try:
        if kind == "int":
            number = Decimal(str(value))
            low = int(number.to_integral_value(rounding="ROUND_FLOOR"))
            high = int(number.to_integral_value(rounding="ROUND_CEILING"))
            return [low - 1, high + 1] if low == high else [low, high]
        if kind == "decimal":
            step = Decimal(1).scaleb(-_numbers(pg_type)[1])
            number = Decimal(str(value))
            return [number - step, number + step, number - 1, number + 1]
        if kind == "float":
            return [float(value) - 1, float(value) + 1]
        if kind in ("date", "timestamp"):
            base = fit(value, pg_type)
            return [] if base is None else [base - datetime.timedelta(days=1), base + datetime.timedelta(days=1)]
        if kind == "text" and isinstance(value, str):
            return [value[:-1], value + "x", value.lower() if value != value.lower() else value.upper()]
    except (ValueError, ArithmeticError, InvalidOperation, TypeError):
        pass
    return []


def _middle(low, high):
    try:
        if isinstance(low, (datetime.date, datetime.datetime)) and type(low) is type(high):
            return low + (high - low) / 2 if isinstance(low, datetime.datetime) else low + datetime.timedelta(days=(high - low).days // 2)
        if isinstance(low, (int, Decimal, float)) and isinstance(high, (int, Decimal, float)) and not isinstance(low, bool):
            middle = (Decimal(str(low)) + Decimal(str(high))) / 2
            return middle
    except (ValueError, ArithmeticError, InvalidOperation, TypeError):
        pass
    return None


def _defaults(pg_type: str) -> tuple[list, list]:
    """``(usual values, rarer values)`` of a column nothing is known about."""

    kind = _kind(pg_type)
    if kind == "int":
        return [1, 2, 3, 4], [0, -1, 5, 10, 100]
    if kind == "decimal":
        scale = _numbers(pg_type)[1]
        half = [Decimal("0.5"), Decimal("2.5")] if scale >= 1 else []
        return [Decimal(1), Decimal(2), Decimal(3), Decimal(10)], [Decimal(0), Decimal(-1), Decimal(-5), Decimal(100), *half]
    if kind == "float":
        return [1.0, 2.0, 3.0, 10.0], [0.0, -1.0, 0.5, 100.0]
    if kind == "date":
        return [_BASE_DATE, _BASE_DATE + datetime.timedelta(days=1), _BASE_DATE + datetime.timedelta(days=40)], [
            _BASE_DATE + datetime.timedelta(days=200), _BASE_DATE + datetime.timedelta(days=366), _BASE_DATE - datetime.timedelta(days=1)]
    if kind == "timestamp":
        start = datetime.datetime(2000, 1, 1)  # noqa: DTZ001
        return [start, start + datetime.timedelta(days=1), start + datetime.timedelta(days=40)], [start + datetime.timedelta(days=366)]
    if kind == "bool":
        return [True, False], []
    return ["a", "b", "c"], ["A", "ab", ""]


def _distinct(index: int, pg_type: str):
    """A value no other row index gets: for a key column the queries never read."""

    kind = _kind(pg_type)
    if kind in ("int", "decimal", "float"):
        return fit(index + 1, pg_type)
    if kind == "date":
        return _BASE_DATE + datetime.timedelta(days=index)
    if kind == "timestamp":
        return datetime.datetime(2000, 1, 1) + datetime.timedelta(days=index)  # noqa: DTZ001
    if kind == "bool":
        return bool(index % 2)
    return fit(f"u{index}", pg_type) or fit(str(index), pg_type)


# --- reading the queries ------------------------------------------------------------------------


@dataclass
class _Column:
    table: str
    name: str
    pg_type: str
    nullable: bool
    read: bool = False
    divisor: bool = False
    usual: list = field(default_factory=list)  # values a row that passes the filters is likely to hold
    rare: list = field(default_factory=list)  # neighbours: a row that fails one filter


@dataclass
class Plan:
    """What the two queries read: the tables to fill, each column's values and the groups a count asks for."""

    tables: dict  # name -> the harness's Table
    columns: dict[tuple[str, str], _Column]
    order: list[str]  # parents before children
    group: int = 0  # a count is compared with this number
    shared: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)  # column -> its class
    anchors: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)  # column -> the key column of its class
    once: dict[tuple[str, str], list] = field(default_factory=dict)  # column -> values that one row only may hold
    hints: dict[str, list[dict[str, object]]] = field(default_factory=dict)  # table -> values one branch of an OR asks for together


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, (exp.Paren, exp.Cast, exp.TryCast)):
        node = node.this
    return node


def _flatten(node: exp.Expression, kind) -> list[exp.Expression]:
    node = _unwrap(node) if isinstance(node, exp.Paren) else node
    if isinstance(node, kind):
        return _flatten(node.this, kind) + _flatten(node.expression, kind)
    return [node]


def _is_constant(node: exp.Expression) -> bool:
    return not any(isinstance(n, _NOT_CONSTANT) for n in node.walk())


class _Reader:
    def __init__(self, trees, tables, db):
        self.trees, self.tables, self.db = trees, tables, db
        self.by_name: dict[str, list[tuple[str, str]]] = {}
        for table in tables.values():
            for column in table.columns:
                self.by_name.setdefault(column, []).append((table.name, column))
        self.alias_table: dict[str, set[str]] = {}
        self.alias_columns: dict[str, set[tuple[str, str]]] = {}
        self.computed: set[str] = set()
        self.parent: dict[tuple[str, str], tuple[str, str]] = {}
        self.usual: dict[tuple[str, str], list] = {}
        self.rare: dict[tuple[str, str], list] = {}
        self.direct: dict[tuple[str, str], list] = {}
        self.read: set[tuple[str, str]] = set()
        self.divisors: set[tuple[str, str]] = set()
        self.group = 0
        self.hints: list[dict[tuple[str, str], object]] = []
        self.once: dict[tuple[str, str], list] = {}
        self.steps: dict[tuple[str, str], list] = {}  # ``a = b + step``
        self.factors: dict[tuple[str, str], list[Decimal]] = {}  # ``a <= factor * b``
        self.scaled: dict[tuple[str, str], list[Decimal]] = {}  # constants an expression over the column is compared with
        self._folded: dict[str, object] = {}

    # union-find over columns compared with each other
    def find(self, key):
        while self.parent.get(key, key) != key:
            self.parent[key] = self.parent.get(self.parent[key], self.parent[key])
            key = self.parent[key]
        return key

    def union(self, keys):
        keys = [self.find(k) for k in keys]
        for other in keys[1:]:
            if other != keys[0]:
                self.parent[other] = keys[0]

    def fold(self, node: exp.Expression):
        """The value of an expression that reads no column, computed by DuckDB; ``None`` when it has none."""

        text = node.sql(dialect="duckdb")
        if text not in self._folded:
            try:
                row = self.db.execute(f"SELECT {text}").fetchone()
                self._folded[text] = row[0] if row else None
            except Exception:  # noqa: BLE001 - an expression DuckDB cannot evaluate gives no value
                self._folded[text] = None
        return self._folded[text]

    def resolve(self, column: exp.Column, depth: int = 0) -> set[tuple[str, str]]:
        name, qualifier = column.name.lower(), column.table.lower()
        found = self.by_name.get(name)
        if found:
            if qualifier:
                named = self.alias_table.get(qualifier, set()) | ({qualifier} if qualifier in self.tables else set())
                narrowed = [key for key in found if key[0] in named]
                if narrowed:
                    return set(narrowed)
            return set(found)
        return set(self.alias_columns.get(name, ())) if depth < 4 else set()

    def columns_of(self, node: exp.Expression) -> set[tuple[str, str]]:
        found: set[tuple[str, str]] = set()
        for column in node.find_all(exp.Column):
            if not isinstance(column.this, exp.Star):
                found |= self.resolve(column)
        return found

    def add(self, keys, usual=(), rare=(), direct: bool = True):
        """Record values for columns; ``direct``: the column itself is compared with them, not an expression over it."""

        for key in keys:
            pg_type = self.tables[key[0]].columns[key[1]]
            for values, target in ((usual, self.usual), (rare, self.rare), (usual if direct else (), self.direct)):
                for value in values:
                    fitted = fit(value, pg_type)
                    if fitted is None and isinstance(value, Decimal) and _kind(pg_type) == "int":
                        fitted = fit(value.to_integral_value(rounding="ROUND_FLOOR"), pg_type)  # between two whole numbers
                    if fitted is not None:
                        target.setdefault(key, []).append(fitted)

    def constant(self, keys, value, operator: str, compound: bool):
        """Record ``column <operator> value`` (or an expression over ``keys`` compared with ``value``).

        ``operator`` is ``eq``, ``neq``, ``above`` (``>``), ``from`` (``>=``), ``below`` (``<``) or ``to`` (``<=``),
        read as if the comparison had to hold: the values that pass are the usual ones, the others the rare ones.
        """

        if value is None:
            return
        for key in keys:
            pg_type = self.tables[key[0]].columns[key[1]]
            near = _neighbours(value, pg_type)
            lower, upper = (near[0], near[1]) if len(near) >= 2 else (None, None)
            if compound:
                self.add([key], [value, *near[:2]], direct=False)
                if isinstance(value, (int, Decimal)) and not isinstance(value, bool):
                    self.scaled.setdefault(key, []).append(Decimal(str(value)))
            elif operator == "eq":
                self.add([key], [value], near)
            elif operator == "neq":
                self.add([key], near[:2], [value])
            elif operator in ("above", "from"):
                self.add([key], [upper] if operator == "above" else [value, upper], [lower] if operator == "from" else [value, lower])
            else:
                self.add([key], [lower] if operator == "below" else [value, lower], [upper] if operator == "to" else [value, upper])

    def fixed(self, part: exp.Expression):
        """``(column, value)`` for a condition that holds one column to a constant (``=``, ``IN``, ``BETWEEN``)."""

        part = _unwrap(part)
        target, value = None, None
        if isinstance(part, exp.EQ):
            for mine, other in ((part.this, part.expression), (part.expression, part.this)):
                if isinstance(_unwrap(mine), exp.Column) and _is_constant(other):
                    target, value = _unwrap(mine), self.fold(other)
                    break
        elif isinstance(part, exp.In) and part.expressions and isinstance(_unwrap(part.this), exp.Column) and _is_constant(part.expressions[0]):
            target, value = _unwrap(part.this), self.fold(part.expressions[0])
        elif isinstance(part, exp.Between) and isinstance(_unwrap(part.this), exp.Column):
            low, high = part.args.get("low"), part.args.get("high")
            if low is not None and high is not None and _is_constant(low) and _is_constant(high):
                low_value, high_value = self.fold(low), self.fold(high)
                target, value = _unwrap(part.this), _middle(low_value, high_value) if _middle(low_value, high_value) is not None else low_value
        if target is None or value is None:
            return
        for key in self.resolve(target):
            fitted = fit(value, self.tables[key[0]].columns[key[1]])
            if fitted is None and isinstance(value, Decimal):
                fitted = fit(value.to_integral_value(), self.tables[key[0]].columns[key[1]])
            if fitted is not None:
                yield key, fitted

    def like(self, keys, pattern: str):
        plain = pattern.replace("%", "").replace("_", "a")
        self.add(keys, [plain, pattern.replace("%", "x").replace("_", "a")], [plain[:-1], plain + "y", "zz"])

    def compare(self, left: exp.Expression, right: exp.Expression, operator: str):
        left_cols, right_cols = self.columns_of(left), self.columns_of(right)
        if isinstance(_unwrap(left), exp.Count) or isinstance(_unwrap(right), exp.Count) or any(
            isinstance(side, exp.Expression) and side.find(exp.Count) for side in (left, right)
        ):
            for side in (left, right):
                if _is_constant(side):
                    value = self.fold(side)
                    if isinstance(value, (int, Decimal)) and not isinstance(value, bool) and 0 < value <= 12:
                        self.group = max(self.group, int(value))
        for mine, other, columns, flipped in ((left, right, left_cols, False), (right, left, right_cols, True)):
            if columns and _is_constant(other):
                single = isinstance(_unwrap(mine), exp.Column) or len(columns) == 1
                if flipped:
                    operator = {"above": "below", "below": "above", "from": "to", "to": "from"}.get(operator, operator)
                self.constant(columns, self.fold(other), operator, compound=not single)
                return
        # ``a <= 1.1 * b``: both columns also hold 10 and 10 times the factor, so the comparison has an exact edge
        if left_cols and right_cols:
            for side in (left, right):
                for product in side.find_all(exp.Mul):
                    for factor, rest in ((product.this, product.expression), (product.expression, product.this)):
                        if _is_constant(factor) and rest.find(exp.Column) is not None:
                            value = self.fold(factor)
                            if isinstance(value, (int, Decimal)) and not isinstance(value, bool) and 0 < abs(value) <= 100 and value != 1:
                                for key in left_cols | right_cols:
                                    self.factors.setdefault(key, []).append(Decimal(str(value)))
        # ``a = b - 52``: one set of values that holds some value together with the one 52 away
        if operator == "eq":
            for mine, other in ((left, right), (right, left)):
                shifted = _unwrap(other)
                if self.one_column(mine) and isinstance(shifted, (exp.Add, exp.Sub)):
                    for column, step in ((shifted.this, shifted.expression), (shifted.expression, shifted.this)):
                        if self.one_column(column) and _is_constant(step) and not (isinstance(shifted, exp.Sub) and column is shifted.expression):
                            value = self.fold(step)
                            if isinstance(value, (int, Decimal)) and not isinstance(value, bool) and value != 0:
                                keys = sorted(self.columns_of(mine) | self.columns_of(column))
                                self.union(keys)
                                for key in keys:
                                    self.steps.setdefault(key, []).append(value)
                            return
        # two columns held equal draw from one set of values (a join); a name that stands for an expression over
        # several columns (``SUM(a) / b AS ratio``) does not tie those columns together
        if operator == "eq" and all(self.one_column(side) for side in (left, right)):
            self.union(sorted(left_cols | right_cols))

    def one_column(self, node: exp.Expression) -> bool:
        node = _unwrap(node)
        if not isinstance(node, exp.Column):
            return False
        name = node.name.lower()
        return name in self.by_name or (name not in self.computed and len(self.alias_columns.get(name, ())) == 1)

    def run(self) -> None:
        for tree in self.trees:
            for table in tree.find_all(exp.Table):
                if table.name.lower() in self.tables and table.alias:
                    self.alias_table.setdefault(table.alias.lower(), set()).add(table.name.lower())
        for _ in range(3):  # an alias of an alias (a CTE column read through a second CTE)
            for tree in self.trees:
                for alias in tree.find_all(exp.Alias):
                    name = alias.alias.lower()
                    if name and name not in self.by_name:
                        self.alias_columns.setdefault(name, set()).update(self.columns_of(alias.this))
                        if not isinstance(_unwrap(alias.this), exp.Column):
                            self.computed.add(name)  # a name for an expression, not for a column
        for tree in self.trees:
            for star in tree.find_all(exp.Star):
                select = star.find_ancestor(exp.Select)
                if isinstance(star.parent, exp.Count) or select is None or isinstance(select.parent, exp.Exists):
                    continue  # COUNT(*) and EXISTS (SELECT * ..) read no column
                qualifier = star.parent.table.lower() if isinstance(star.parent, exp.Column) else ""
                from_ = select.args.get("from_") or select.args.get("from")
                for source in ([from_.this] if from_ is not None else []) + [j.this for j in select.args.get("joins") or []]:
                    # a star over a table reads its columns; over a derived table or CTE, only what that lists
                    if isinstance(source, exp.Table) and source.name.lower() in self.tables and qualifier in ("", source.alias_or_name.lower()):
                        self.read.update((source.name.lower(), column) for column in self.tables[source.name.lower()].columns)
            self.read |= self.columns_of(tree)
            for node in tree.walk():
                if isinstance(node, _COMPARISONS):
                    self.compare(node.this, node.expression, _OPERATORS[type(node)])
                elif isinstance(node, exp.Or) and not isinstance(node.parent, exp.Or):
                    for branch in _flatten(node, exp.Or):
                        hint = {}
                        for part in _flatten(branch, exp.And):
                            for key, value in self.fixed(part):
                                hint.setdefault(key, value)
                        if len(hint) > 1:
                            self.hints.append(hint)
                elif isinstance(node, exp.Between):
                    columns = self.columns_of(node.this)
                    low, high = node.args.get("low"), node.args.get("high")
                    if columns and low is not None and high is not None and _is_constant(low) and _is_constant(high):
                        low_value, high_value = self.fold(low), self.fold(high)
                        single = isinstance(_unwrap(node.this), exp.Column) or len(columns) == 1
                        for key in columns:
                            pg_type = self.tables[key[0]].columns[key[1]]
                            inside = [v for v in (low_value, _middle(low_value, high_value), high_value) if v is not None]
                            outside = (_neighbours(low_value, pg_type)[:1] if low_value is not None else []) + (
                                _neighbours(high_value, pg_type)[1:2] if high_value is not None else [])
                            self.add([key], inside, outside, direct=single)
                            if not single:
                                self.scaled.setdefault(key, []).extend(
                                    Decimal(str(v)) for v in inside if isinstance(v, (int, Decimal)) and not isinstance(v, bool))
                    else:
                        if low is not None:
                            self.compare(node.this, low, "from")
                        if high is not None:
                            self.compare(node.this, high, "to")
                elif isinstance(node, exp.In):
                    columns = self.columns_of(node.this)
                    for item in node.expressions:
                        if columns and _is_constant(item):
                            self.constant(columns, self.fold(item), "eq", compound=False)
                    query = node.args.get("query")
                    inner = query.this if isinstance(query, exp.Subquery) else query
                    while isinstance(inner, exp.SetOperation):
                        inner = inner.this
                    if isinstance(inner, exp.Select) and len(inner.expressions) == 1 and self.one_column(node.this):
                        item = inner.expressions[0]
                        item = item.this if isinstance(item, exp.Alias) else item
                        if self.one_column(item):
                            self.union(sorted(columns | self.columns_of(item)))
                elif isinstance(node, (exp.Like, exp.ILike)):
                    pattern = node.expression
                    columns = self.columns_of(node.this)
                    if columns and isinstance(pattern, exp.Literal) and pattern.is_string:
                        self.like(columns, pattern.name)
                elif isinstance(node, exp.Subquery) and isinstance(node.parent, (exp.Binary, exp.Between, exp.Paren)) and isinstance(node.this, exp.Select):
                    # a subquery read as one value fails on two rows: one row only holds the value it filters on
                    inner = node.this
                    where = inner.args.get("where")
                    if where is not None and not inner.args.get("group") and not any(isinstance(n, exp.AggFunc) for e in inner.expressions for n in e.walk()):
                        for part in _flatten(where.this, exp.And):
                            if isinstance(_unwrap(part), exp.EQ):
                                for key, value in self.fixed(part):
                                    self.once.setdefault(key, []).append(value)
                elif isinstance(node, _DIVISIONS):
                    for column in node.expression.find_all(exp.Column):
                        if self.one_column(column):  # not a name for a sum or another expression: those are only checked
                            self.divisors |= self.resolve(column)
                elif isinstance(node, exp.Join) and node.args.get("using"):
                    for identifier in node.args["using"]:
                        keys = self.by_name.get(identifier.name.lower(), [])
                        if len(keys) > 1:
                            self.union(sorted(keys))


def _used(trees, tables) -> dict:
    names = {t.name.lower() for tree in trees for t in tree.find_all(exp.Table)}
    used = {n: tables[n] for n in names if n in tables}
    pending = list(used.values())
    while pending:  # parents of foreign keys, so a child row has a row to point at
        for _, parent, _ in pending.pop().foreign:
            if parent in tables and parent not in used:
                used[parent] = tables[parent]
                pending.append(tables[parent])
    return used


def read_queries(left: str, right: str, tables: dict, db) -> Plan | None:
    """The plan for a pair of PostgreSQL queries over ``tables``; ``None`` when they read no known table."""

    trees = [sqlglot.parse_one(sql, read="postgres") for sql in (left, right)]
    used = _used(trees, tables)
    if not used:
        return None
    reader = _Reader(trees, used, db)
    reader.run()
    # every column of a class holds the class's values
    classes: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for key in {k for keys in reader.by_name.values() for k in keys}:
        classes.setdefault(reader.find(key), []).append(key)
    columns: dict[tuple[str, str], _Column] = {}
    shared = {}
    for root, members in classes.items():
        usual = [v for m in members for v in reader.usual.get(m, [])]
        rare = [v for m in members for v in reader.rare.get(m, [])]
        for table_name, name in members:
            table = used[table_name]
            pg_type = table.columns[name]
            column = _Column(table_name, name, pg_type, name not in table.not_null, read=(table_name, name) in reader.read,
                             divisor=(table_name, name) in reader.divisors)
            default_usual, default_rare = _defaults(pg_type)
            mine = _dedupe(reader.direct.get((table_name, name), []))
            if column.divisor:
                column.usual, column.rare = _divisors(mine, pg_type)
            else:
                mine = _dedupe(v for v in (fit(u, pg_type) for u in usual) if v is not None)
                column.usual = mine or default_usual
                steps = _dedupe(step for m in members for step in reader.steps.get(m, []))
                if steps:  # each value with the one a step away, so both sides of ``a = b + step`` have rows
                    stepped = [fit(v + sign * step, pg_type) for v in column.usual[:3] for step in steps for sign in (1, -1)
                               if isinstance(v, (int, Decimal)) and not isinstance(v, bool)]
                    column.usual = _dedupe([*column.usual[:3], *(v for v in stepped if v is not None)])
                column.rare = _dedupe([*(v for v in (fit(u, pg_type) for u in rare) if v is not None), *(default_usual[:2] if mine else default_rare)])
            if len(members) > 1:
                shared[(table_name, name)] = root
            columns[(table_name, name)] = column
    for key, found in reader.factors.items():
        column = columns[key]
        if not column.divisor and _kind(column.pg_type) in ("int", "decimal", "float"):
            extra = [fit(10, column.pg_type), *(fit(10 * factor, column.pg_type) for factor in _dedupe(found))]
            extra = [v for v in extra if v is not None]
            if reader.usual.get(key) or key in shared:
                column.usual = _dedupe([*column.usual, *extra])
            else:  # nothing else is asked of the column: its sums stay near each other
                column.usual, column.rare = _dedupe(extra), _dedupe([*column.usual, *column.rare])
    # an expression compared with a constant (x / y BETWEEN 2 AND 5): x also holds the constant times each divisor
    factors = _dedupe(Decimal(str(d)) for c in columns.values() if c.divisor for d in c.usual + c.rare) or [Decimal(-1)]
    for key, constants in reader.scaled.items():
        column = columns[key]
        if not column.divisor:
            extra = [fit(constant * factor, column.pg_type) for constant in _dedupe(constants) for factor in factors]
            column.usual = _dedupe([*column.usual, *(v for v in extra if v is not None)])
    anchors: dict[tuple[str, str], tuple[str, str]] = {}
    for members in classes.values():
        keyed = sorted(m for m in members if (m[1],) in [tuple(k) for k in used[m[0]].keys])
        if keyed and len(members) > 1:
            for member in members:
                if member != keyed[0]:
                    anchors[member] = keyed[0]
    hints: dict[str, list[dict[str, object]]] = {}
    for hint in reader.hints:
        for table_name in {key[0] for key in hint}:
            mine = {name: value for (owner, name), value in hint.items() if owner == table_name}
            if mine and mine not in hints.setdefault(table_name, []):
                hints[table_name].append(mine)
    order: list[str] = []

    def place(name: str, trail: tuple[str, ...] = ()):
        if name in order or name in trail:
            return
        before = [parent for _, parent, _ in used[name].foreign if parent in used]
        before += sorted({anchor[0] for member, anchor in anchors.items() if member[0] == name})
        for other in before:
            place(other, (*trail, name))
        order.append(name)

    for name in sorted(used, key=lambda n: (not any(len(k) == 1 for k in used[n].keys), n)):
        place(name)
    once = {key: _dedupe(v for v in (fit(value, columns[key].pg_type) for value in values) if v is not None)
            for key, values in reader.once.items() if not columns[key].divisor}
    return Plan(used, columns, order, reader.group, shared, anchors, hints=hints, once=once)


def _divisors(constants: list, pg_type: str) -> tuple[list, list]:
    """``(usual, rare)`` values of a column read inside a divisor: never zero, and only powers of two.

    Dividing by a power of two is exact in floating point, so DuckDB's quotient compares with a constant as
    PostgreSQL's exact one does. When the queries bound the column (``y BETWEEN 236 AND 265``) the powers of
    two inside the bounds are the usual values (256); otherwise 1, 2, -1 and -2.
    """

    small = [v for v in (fit(d, pg_type) for d in _DIVISORS) if v is not None]
    numbers = [Decimal(str(c)) for c in constants if isinstance(c, (int, Decimal, float)) and not isinstance(c, bool)]
    if not numbers:
        return small, []
    low, high = min(numbers), max(numbers)
    powers = [sign * 2**k for k in range(0, 31) for sign in (1, -1) if low <= sign * 2**k <= high]
    inside = [v for v in (fit(d, pg_type) for d in powers) if v is not None]
    return (inside, small) if inside else (small, [])


def _dedupe(values) -> list:
    seen, kept = set(), []
    for value in values:
        mark = (type(value).__name__, value)
        if mark not in seen:
            seen.add(mark)
            kept.append(value)
    return kept


# --- databases ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Profile:
    label: str
    keyed: tuple[int, int]  # rows of a table with a one-column key
    other: tuple[int, int]  # rows of any other table; (0, 0): enough for the groups a count asks for
    shared: int  # how many of a join column's values are in use
    clean: float  # chance that a new row holds usual values only
    nulls: float  # chance of NULL in a column of a row that is not clean
    rare: float  # chance of a rare value there


_PROFILES = (
    _Profile("one row per table", (1, 1), (1, 2), 1, 0.95, 0.1, 0.3),
    _Profile("two keys", (2, 2), (3, 6), 2, 0.9, 0.1, 0.3),
    _Profile("groups", (1, 2), (0, 0), 1, 0.9, 0.05, 0.2),
    _Profile("copies", (2, 3), (6, 12), 2, 0.85, 0.15, 0.3),
    _Profile("groups of two keys", (2, 3), (0, 0), 2, 0.9, 0.05, 0.2),
    _Profile("wide", (3, 5), (8, 16), 3, 0.8, 0.2, 0.3),
    _Profile("nulls", (2, 4), (4, 10), 2, 0.4, 0.4, 0.15),
)


def database(plan: Plan, rng: random.Random, profile: int = 0) -> tuple[str, dict[str, list[tuple]]]:
    """``(label, {table: rows})``: one database drawn for ``plan``."""

    shape = _PROFILES[profile % len(_PROFILES)]
    group = (plan.group or 2) + 1
    data: dict[str, list[tuple]] = {}
    for name in plan.order:
        table = plan.tables[name]
        names = list(table.columns)
        columns = [plan.columns[(name, c)] for c in names]
        single_keys = {key[0] for key in table.keys if len(key) == 1}
        key_columns = {c for key in table.keys for c in key}
        low, high = shape.keyed if single_keys else shape.other
        # ``sources[i]``: the row that row ``i`` copies (None: a new row). For the groups a count asks for, two
        # or three new rows, each followed by enough copies of itself; otherwise the second half copies the first.
        sources: list[int | None] = []
        if high:
            count = rng.randint(low, high)
            for index in range(count):
                copies = index >= (count + 1) // 2 and index > 0 and not single_keys and shape.keyed != (1, 1)
                sources.append(rng.randrange(index) if copies else None)
        else:
            for _ in range(rng.randint(1, shape.shared + 1)):
                first = len(sources)
                sources.append(None)
                sources.extend([first] * (group - 1 + rng.randint(0, 1)))
            count = len(sources)
        once = {c: plan.once[(name, c)] for c in names if (name, c) in plan.once and c not in single_keys}
        taken: dict[str, set] = {c: set() for c in single_keys | set(once)}
        seen_keys: list[set] = [set() for _ in table.keys]
        pointing = {c for child, _, _ in table.foreign for c in child}  # set from a parent row, never drawn
        readable = [i for i, c in enumerate(columns) if c.read and c.name not in single_keys and c.name not in pointing]
        rows: list[list] = []
        kept_at: dict[int, int] = {}  # row index as planned -> its place in ``rows``

        def draw(column: _Column, index: int, nulls: float, rare: float):
            if not column.read and column.name not in single_keys:
                if column.name in key_columns:
                    return _distinct(index, column.pg_type)  # a key column nobody reads keeps the rows apart
                return None if column.nullable else column.usual[0]
            roll = rng.random()
            if column.nullable and roll < nulls:
                return None
            if column.rare and roll < nulls + rare:
                return rng.choice(column.rare)
            if column.divisor:
                return rng.choice(column.usual)  # never a value from elsewhere: it must not be zero
            anchor = plan.anchors.get((name, column.name))
            if anchor is not None and data.get(anchor[0]):
                position = list(plan.tables[anchor[0]].columns).index(anchor[1])
                return rng.choice(data[anchor[0]][:shape.shared])[position]
            pool = column.usual
            if ((name, column.name) in plan.shared or column.name in key_columns) and not _has_constants(column):
                pool = pool[:shape.shared]
            return rng.choice(pool)

        def unique(column: _Column, value, index: int):
            used_values = taken[column.name]
            if value is None or value in used_values:
                options = [v for v in [*column.usual, *column.rare] if v not in used_values]
                value = options[0] if options else None
                step = index
                while value is None or value in used_values:
                    value = _distinct(step, column.pg_type)
                    step += 1
                    if step > index + 50:
                        return None
            used_values.add(value)
            return value

        for index in range(count):
            copied = sources[index] is not None and sources[index] in kept_at
            if copied:  # a row next to another: the same but for a column or two (rarely, within a group)
                row = list(rows[kept_at[sources[index]]])
                changes = rng.choice((0, 1, 1, 2)) if high else rng.choice((0, 0, 0, 0, 0, 0, 1))
                for position in rng.sample(readable, min(len(readable), changes)):
                    row[position] = draw(columns[position], index, 0.2, 0.5)
                for position, column in enumerate(columns):
                    if not column.read and column.name in key_columns:
                        row[position] = _distinct(index, column.pg_type)
            else:
                clean = rng.random() < shape.clean
                row = [draw(column, index, 0.0 if clean else shape.nulls, 0.0 if clean else shape.rare) for column in columns]
                if clean and plan.hints.get(name) and rng.random() < 0.8:
                    for column_name, value in rng.choice(plan.hints[name]).items():
                        if not plan.columns[(name, column_name)].divisor and column_name not in pointing:
                            row[names.index(column_name)] = value
            if not copied or rng.random() < 0.2:
                clean = not copied and clean
                for child, parent, parent_columns in table.foreign:
                    parent_rows = data.get(parent)
                    if not parent_rows:
                        for c in child:  # no parent row to point at
                            row[names.index(c)] = None
                        continue
                    source = rng.choice(parent_rows)
                    parent_names = list(plan.tables[parent].columns)
                    if all(c not in table.not_null for c in child) and not clean and rng.random() < shape.nulls:
                        for c in child:
                            row[names.index(c)] = None
                    else:
                        for c, pc in zip(child, parent_columns):
                            row[names.index(c)] = source[parent_names.index(pc)]
            dropped = False
            for position, column in enumerate(columns):
                if column.name in single_keys:
                    row[position] = unique(column, row[position], index)
                    dropped = dropped or row[position] is None
                elif row[position] in once.get(column.name, ()):
                    if row[position] in taken[column.name]:  # a second row takes another value, which may repeat
                        others = [v for v in [*column.rare, *column.usual] if v not in once[column.name]]
                        row[position] = rng.choice(others) if others else unique(column, None, index)
                    else:
                        taken[column.name].add(row[position])
                elif row[position] is None and not column.nullable:
                    row[position] = column.usual[0]
            for key, seen in zip(table.keys, seen_keys):
                value = tuple(row[names.index(c)] for c in key)
                if value in seen:
                    dropped = True
                seen.add(value)
            if not dropped:
                kept_at[index] = len(rows)
                rows.append(row)
        data[name] = [tuple(row) for row in rows]
    return shape.label, data


def _has_constants(column: _Column) -> bool:
    return column.usual != _defaults(column.pg_type)[0]


# --- the guards ---------------------------------------------------------------------------------


def guard_divisions(sql: str) -> str:
    """DuckDB text in which a zero divisor raises an error, as PostgreSQL does, instead of giving NULL."""

    tree = sqlglot.parse_one(sql, read="duckdb")
    for node in list(tree.find_all(*_DIVISIONS)):
        divisor = node.expression
        if isinstance(divisor, exp.Literal) and not divisor.is_string and Decimal(divisor.name) != 0:
            continue
        checked = exp.Case(
            ifs=[exp.If(this=exp.EQ(this=exp.Paren(this=divisor.copy()), expression=exp.Literal.number(0)),
                        true=exp.Anonymous(this="error", expressions=[exp.Literal.string("division by zero")]))],
            default=exp.Paren(this=divisor.copy()),
        )
        node.set("expression", exp.Paren(this=checked))
    return tree.sql(dialect="duckdb")


def outer_limits(sql: str, determinism) -> tuple[str, int | None] | None:
    """``(the query without its outermost LIMIT, that LIMIT)`` when nothing else in it can depend on input order.

    ``None`` when rows can depend on input order for another reason (the pair is then not searched).
    ``determinism`` is the harness's own classifier.
    """

    tree = sqlglot.parse_one(sql, read="duckdb")
    limit = tree.args.get("limit")
    count = None
    if limit is not None:
        expression = limit.expression
        if not isinstance(expression, exp.Literal) or expression.is_string or not expression.name.isdigit() or tree.args.get("offset") is not None:
            return None
        count = int(expression.name)
        tree.set("limit", None)
    if determinism(tree) != "fixed":
        return None
    return tree.sql(dialect="duckdb"), count


# --- the search ---------------------------------------------------------------------------------


def find_witness(left: str, right: str, tables: dict, replayer, kind: str, seconds: float, bench, seed: int = SEED, seen: Counter | None = None):
    """``(witness, label)`` for two PostgreSQL queries, or ``(None, "")``.

    ``replayer`` is the harness's ``Replayer`` and ``bench`` the harness module (``replay_sql``, ``repair``,
    ``legal``, ``determinism``, ``QUERY_SECONDS``): the comparison is the harness's own. ``seen``, when given,
    counts what the search drew (a measuring aid).
    """

    import duckdb

    from kumo_sql_prover.duckdb_load import run_unoptimized

    if kind == "random" or seconds <= 0:
        return None, ""
    started = time.monotonic()
    try:
        queries = [guard_divisions(bench.replay_sql(sql)) for sql in (left, right)]
        limits = [outer_limits(sql, bench.determinism) for sql in queries]
        if any(found is None for found in limits) and kind != "fixed":
            return None, ""  # rows may depend on input order for a reason this search cannot rule out
        plan = read_queries(left, right, tables, replayer.db)
    except (sqlglot.errors.SqlglotError, ValueError, RecursionError):
        return None, ""
    if plan is None:
        return None, ""
    rng = random.Random(seed)

    from kumo_sql_prover.duckdb_load import insert_rows

    filled: set[str] = set(replayer.tables)  # tables that may hold rows of an earlier replay

    def load(data):
        """Replace the rows of the tables the queries read (the others are emptied once and stay empty)."""

        nonlocal filled
        replayer.db.execute("BEGIN")
        for name in filled | set(data):
            replayer.db.execute(f'DELETE FROM "{name}"')
        for name, rows in data.items():
            if rows:
                insert_rows(replayer.db, f'"{name}"', rows)
        replayer.db.execute("COMMIT")
        filled = {name for name, rows in data.items() if rows}

    def bags(data):
        """The two bags on ``data``, or ``None`` when a query fails or the data is not legal."""

        nonlocal filled
        data = bench.repair(data, replayer.tables)
        if not bench.legal(data, replayer.tables):
            return None
        try:
            load(data)
            return replayer._run(queries[0]), replayer._run(queries[1])
        except duckdb.Error:
            try:
                replayer.db.execute("ROLLBACK")
            except duckdb.Error:
                pass
            filled = set(replayer.tables)
            return None

    def differ(data) -> bool:
        found = bags(data)
        return found is not None and found[0] != found[1]

    def confirmed(data):
        nonlocal filled
        filled = set(replayer.tables)  # the harness's own loads below fill tables this search does not track
        if kind != "fixed":  # a LIMIT on the outermost query: it must cut nothing on either side
            try:
                replayer._load(bench.repair(data, replayer.tables))
                for unlimited, count in limits:
                    if count is not None and sum(replayer._run(unlimited).values()) > count:
                        return None
            except duckdb.Error:
                return None
        timer = threading.Timer(bench.QUERY_SECONDS, replayer.db.interrupt)
        timer.start()
        try:
            replayer._load(bench.repair(data, replayer.tables))
            run_unoptimized(replayer.db, *queries)  # bounded here; the harness's own check repeats it
        except duckdb.Error:
            return None
        finally:
            timer.cancel()
        return replayer.differs(queries[0], queries[1], data, kind)

    def reduce(data):
        """Drop rows, half a table at a time and then singly, while the two bags still differ."""

        deadline = time.monotonic() + max(1.5, seconds / 2)
        data = {name: list(rows) for name, rows in bench.repair(data, replayer.tables).items()}
        for name in sorted(data, key=lambda n: -len(data[n])):
            size = max(len(data[name]) // 2, 1)
            while size >= 1 and time.monotonic() < deadline:
                start = 0
                while start < len(data[name]) and time.monotonic() < deadline:
                    rows = data[name]
                    trial = {**data, name: rows[:start] + rows[start + size:]}
                    if differ(trial):
                        data = {n: list(r) for n, r in bench.repair(trial, replayer.tables).items()}
                        data.setdefault(name, [])
                    else:
                        start += size
                size = size // 2 if size > 1 else 0
        return data

    tried = failures = errors = 0
    while time.monotonic() - started < seconds and failures < 3:
        label, data = database(plan, rng, tried)
        tried += 1
        found = bags(data)
        if seen is not None:
            seen["databases"] += 1
            seen[f"{label}: " + ("failed" if found is None else "differ" if found[0] != found[1] else "same rows" if found[0] else "no rows")] += 1
        if found is None:
            errors += 1
            if errors >= 8 and errors == tried:
                return None, ""  # the queries do not run here at all
            continue
        if found[0] == found[1]:
            continue
        witness = confirmed(reduce(data))
        if witness:
            return witness, f"directed: {label} (database {tried})"
        failures += 1
    return None, ""
