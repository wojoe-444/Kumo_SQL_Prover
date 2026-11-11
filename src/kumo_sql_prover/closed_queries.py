"""A query that reads no table has one result: work it out and write the rows.

    SELECT * FROM (SELECT 1 AS a UNION ALL SELECT 2) AS t1 LEFT JOIN (SELECT 10 AS b UNION ALL SELECT 11) AS t2 ON a > b

reads only rows written in its own text, so its result is the same on every database: the two rows ``(1, NULL)``
and ``(2, NULL)``. Engines test their joins this way (inline rows on both sides, the expected rows written as a
second query), and the provers, which compare queries block by block, had nothing to match a join of constants
against a list of rows with: "UNION shapes differ", "no row-preserving mapping".

``fold_closed_query`` evaluates such a query with the small interpreter below and returns its rows as

    SELECT 1 AS a, NULL AS b UNION ALL SELECT 2, NULL

(sorted, so two queries with the same rows get the same text; ``SELECT NULL AS a, NULL AS b WHERE FALSE`` when
there are none). The provers then compare two lists of rows, which they already prove equal or refute.

The interpreter covers a fragment in which every engine agrees and nothing can fail, and declines the rest:

* **values**: INT64 (integer literals; a result outside INT64 is an error in BigQuery, so it is declined), BOOL,
  STRING (compared for equality only, passed through otherwise) and NULL; a decimal number (``1.5``, or
  ``CAST(1 AS NUMERIC)``) only to be compared (``=``, ``<``, ``IN``, ``COUNT(DISTINCT ..)``) with decimals and integers
  of at most 15 digits, which FLOAT64 and NUMERIC both hold exactly enough to compare as written;
* **expressions**: ``+ - *`` and unary minus on integers; ``= != < <= > >=`` (three-valued; both sides of one
  kind); ``AND OR NOT`` (three-valued); ``IS [NOT] NULL``, ``IS [NOT] TRUE/FALSE``; ``BETWEEN``; ``COALESCE``,
  ``IF``, ``CASE`` (both forms), ``NULLIF``; ``CAST`` and ``SAFE_CAST`` to the value's own type; ``IN`` over a
  list, a subquery or ``UNNEST`` of an array literal (three-valued: NULL when nothing matches and something
  compared was NULL; FALSE over no rows); ``EXISTS``; a scalar subquery of at most one row (more is an error);
  an element of an array written in place, ``[a, b][OFFSET(i)]`` with ``ORDINAL`` and the ``SAFE_`` forms (out of
  bounds without ``SAFE_`` is an error, so it is declined);
* **sources**: a derived table, ``UNNEST`` of an array literal of scalars, or nothing (one row);
* **joins**: comma, ``CROSS``, ``INNER``, ``LEFT``, ``RIGHT`` and ``FULL`` with an optional ``ON``, taken left to right;
* **clauses**: ``WHERE``; ``GROUP BY`` on expressions with ``COUNT(*)``, ``COUNT([DISTINCT] x)``, ``SUM``, ``MIN`` and
  ``MAX`` of integers (a select item is a grouping expression, an aggregate, or built from those); a global
  aggregate (one row, also over no rows); ``SELECT DISTINCT``; ``*`` and ``t.*``;
* **set operations**: ``UNION ALL``, ``UNION DISTINCT``, ``INTERSECT DISTINCT``, ``EXCEPT DISTINCT`` by position.

A column is resolved as SQL does: in the nearest enclosing select that has it, qualified or not; a name that
two sources of one select have is declined, and so is a bare name that is also a source's alias (in BigQuery that
reads the whole row). A derived table sees the enclosing query's outer rows but not its sibling sources.

Declined: any table or WITH table; ``ORDER BY``, ``LIMIT``, ``OFFSET``, ``HAVING``, ``QUALIFY``, windows,
``USING`` and ``NATURAL`` joins, a ``RIGHT`` or ``FULL`` join after a comma or ``CROSS JOIN`` (after a comma BigQuery
refuses it and standard SQL groups it differently; sqlglot reads the two the same), different set operators side by side without parentheses, ``GROUP BY`` extensions, ``* EXCEPT``, ``SELECT AS STRUCT``, ``INTERSECT ALL`` and
``EXCEPT ALL``; a decimal in arithmetic, in an aggregate other than ``COUNT`` or in the result (its type would have
  to be written back), exponents, longer numbers, division, functions, strings under ``<``; mixed kinds in one comparison (an integer with a decimal aside),
one ``COALESCE`` / ``IF`` / ``CASE``, one column of a set operation or one array; more than ``_MAX_ROWS`` rows at any step
or ``_MAX_STEPS`` evaluations; other dialects (the fragment means the same everywhere, but only BigQuery was checked).

The interpreter is checked against DuckDB on random queries of this fragment (``tests/test_closed_queries.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re

from sqlglot import exp

_MAX_ROWS = 512  # rows of any intermediate or final relation
_MAX_OUTPUT_ROWS = 64
_MAX_STEPS = 200_000  # expression evaluations for one query
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1
_SELECT_ARGS = {"expressions", "from_", "joins", "where", "group", "distinct"}
_INTEGER_TYPES = {exp.DataType.Type.BIGINT, exp.DataType.Type.INT}
_STRING_TYPES = {exp.DataType.Type.VARCHAR, exp.DataType.Type.TEXT}
_AGGREGATES = (exp.Count, exp.Sum, exp.Min, exp.Max)
_DIGITS = re.compile(r"[0-9]+")
_DECIMAL = re.compile(r"([0-9]+)\.([0-9]*)|\.([0-9]+)")
_DECIMAL_TYPES = {exp.DataType.Type.DECIMAL, exp.DataType.Type.BIGDECIMAL}
_EXACT_DIGITS = 15  # decimal numbers of up to this many digits are different FLOAT64 values, in the same order


class _Declined(Exception):
    """The query is outside the fragment."""


@dataclass(frozen=True)
class _Column:
    table: str | None  # the source's alias, lower-case
    name: str | None  # lower-case; None for an unnamed expression
    shown: exp.Identifier | None = None  # the name as written, for the output


@dataclass
class _Relation:
    columns: list[_Column]
    rows: list[tuple]


@dataclass
class _Scope:
    """The columns a select's expressions can read, one row of them, and the scope around the select."""

    columns: list[_Column]
    row: tuple
    aliases: frozenset[str]
    outer: "_Scope | None" = None
    group: "_Group | None" = None


@dataclass
class _Group:
    keys: dict[str, object]  # the text of each grouping expression -> its value for this group
    rows: list[tuple]


def _kind(value) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if type(value) is str:
        return "string"
    if type(value) is Fraction:
        return "decimal"
    raise _Declined


def _one_kind(values) -> None:
    if len({_kind(v) for v in values} - {"null"}) > 1:
        raise _Declined


def _decimal(digits: str, scale: int) -> Fraction:
    """The number written with ``digits`` and ``scale`` of them after the point, when it is short enough to compare exactly."""

    while scale and digits.endswith("0"):
        digits, scale = digits[:-1], scale - 1
    if len(digits.lstrip("0")) > _EXACT_DIGITS:
        raise _Declined
    return Fraction(int(digits or "0"), 10**scale)


def _one_reading(values) -> None:
    """Decline integers too long for FLOAT64 among values that are compared as decimals (all are read as the widest type)."""

    if any(type(v) is Fraction for v in values) and any(type(v) is int and abs(v) >= 10**_EXACT_DIGITS for v in values):
        raise _Declined


def _integer(value) -> int:
    if not _INT64_MIN <= value <= _INT64_MAX:
        raise _Declined  # BigQuery raises an overflow error
    return value


def _extra_args(node: exp.Expression, allowed) -> bool:
    return any(value for key, value in node.args.items() if key not in allowed)


class _Interpreter:
    def __init__(self) -> None:
        self.steps = 0

    # -- relations ----------------------------------------------------------------

    def query(self, node: exp.Expression, outer: _Scope | None) -> _Relation:
        if isinstance(node, exp.Subquery):
            if _extra_args(node, ("this",)):
                raise _Declined
            return self.query(node.this, outer)
        if isinstance(node, exp.Select):
            return self.select(node, outer)
        if isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
            return self.set_operation(node, outer)
        raise _Declined

    def set_operation(self, node: exp.SetOperation, outer: _Scope | None) -> _Relation:
        if _extra_args(node, ("this", "expression", "distinct")):
            raise _Declined
        distinct = node.args.get("distinct")
        for operand in (node.this, node.expression):
            # BigQuery wants parentheses between different set operators; without them engines disagree on the grouping
            if isinstance(operand, exp.SetOperation) and (type(operand) is not type(node) or bool(operand.args.get("distinct")) != bool(distinct)):
                raise _Declined
        left, right = self.query(node.this, outer), self.query(node.expression, outer)
        if len(left.columns) != len(right.columns):
            raise _Declined
        for position in range(len(left.columns)):
            _one_kind([row[position] for row in left.rows + right.rows])
        if isinstance(node, exp.Union):
            rows = left.rows + right.rows
            rows = list(dict.fromkeys(rows)) if distinct else rows
        elif not distinct:
            raise _Declined
        elif isinstance(node, exp.Intersect):
            there = set(right.rows)
            rows = [row for row in dict.fromkeys(left.rows) if row in there]
        else:
            there = set(right.rows)
            rows = [row for row in dict.fromkeys(left.rows) if row not in there]
        return self.sized(_Relation([_Column(None, c.name, c.shown) for c in left.columns], rows))

    def sized(self, relation: _Relation) -> _Relation:
        if len(relation.rows) > _MAX_ROWS:
            raise _Declined
        return relation

    def source(self, node: exp.Expression, outer: _Scope | None) -> tuple[_Relation, str | None]:
        """A FROM or JOIN item as a relation, and its alias."""

        if isinstance(node, exp.Subquery):
            alias = node.args.get("alias")
            if _extra_args(node, ("this", "alias")) or (alias is not None and (not isinstance(alias, exp.TableAlias) or alias.args.get("columns"))):
                raise _Declined
            name = alias.name.lower() if alias is not None and alias.name else None
            inner = self.query(node.this, outer)
            return _Relation([_Column(name, c.name, c.shown) for c in inner.columns], inner.rows), name
        if isinstance(node, exp.Unnest):
            alias = node.args.get("alias")
            arrays = node.expressions
            if node.args.get("offset") or len(arrays) != 1 or type(arrays[0]) is not exp.Array or not isinstance(alias, exp.TableAlias):
                raise _Declined
            columns = alias.args.get("columns") or []
            if alias.this is not None and columns or len(columns) > 1:
                raise _Declined
            shown = alias.this if alias.this is not None else (columns[0] if columns else None)
            if not isinstance(shown, exp.Identifier) or not shown.name:
                raise _Declined
            values = self.array(arrays[0])
            return _Relation([_Column(None, shown.name.lower(), shown)], [(value,) for value in values]), None
        raise _Declined

    def array(self, array: exp.Array) -> list:
        """The elements of an array literal of scalars; nothing in it reads a column."""

        closed = _Scope([], (), frozenset())
        values = [self.value(element, closed) for element in array.expressions]
        if len(values) > _MAX_ROWS:
            raise _Declined
        _one_kind(values)
        return values

    def select(self, select: exp.Select, outer: _Scope | None) -> _Relation:
        if _extra_args(select, _SELECT_ARGS):
            raise _Declined
        relation, aliases = _Relation([], [()]), set()
        from_ = select.args.get("from_")
        joins = select.args.get("joins") or []
        if from_ is None and joins:
            raise _Declined
        if from_ is not None:
            relation, alias = self.source(from_.this, outer)
            aliases = {alias} if alias else set()
        comma = False
        for join in joins:
            # BigQuery takes a comma and a JOIN left to right and refuses a RIGHT or FULL join after a comma; standard
            # SQL binds the JOIN first, and only for those two do the readings differ. So that shape is declined
            if comma and (join.args.get("side") or "").upper() in ("RIGHT", "FULL"):
                raise _Declined
            comma = comma or (not join.args.get("side") and join.args.get("on") is None)  # sqlglot reads a comma as CROSS JOIN
            relation, alias = self.join(relation, join, aliases, outer)
            if alias:
                if alias in aliases:
                    raise _Declined
                aliases.add(alias)
        names = frozenset(aliases)

        def scope(row: tuple, group: _Group | None = None) -> _Scope:
            return _Scope(relation.columns, row, names, outer, group)

        where = select.args.get("where")
        if where is not None:
            relation = _Relation(relation.columns, [row for row in relation.rows if self.value(where.this, scope(row)) is True])
        items = select.expressions
        group = select.args.get("group")
        aggregated = any(self.own_aggregates(item, select) for item in items)
        if group is not None or aggregated:
            if any(isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)) for item in items):
                raise _Declined
            keys = []
            if group is not None:
                if _extra_args(group, ("expressions",)) or not group.expressions:
                    raise _Declined
                keys = list(group.expressions)
                self.check_group_names(keys, items)
            groups: dict[tuple, list[tuple]] = {} if keys else {(): list(relation.rows)}
            if keys:
                keyed = [(tuple(self.value(key, scope(row)) for key in keys), row) for row in relation.rows]
                for position in range(len(keys)):
                    _one_kind([values[position] for values, _ in keyed])  # True and 1 are one dictionary key in Python
                for values, row in keyed:
                    groups.setdefault(values, []).append(row)
            texts = [key.sql() for key in keys]
            columns = [self.output_column(item) for item in items]
            rows = []
            for values, members in groups.items():
                env = scope(members[0] if members else tuple(None for _ in relation.columns), _Group(dict(zip(texts, values)), members))
                rows.append(tuple(self.value(item.unalias(), env) for item in items))
        else:
            columns, picks = [], []
            for item in items:
                if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
                    if isinstance(item, exp.Star):
                        if _extra_args(item, ()):
                            raise _Declined
                        chosen = list(range(len(relation.columns)))
                    else:
                        if _extra_args(item.this, ()) or len(item.parts) != 2:
                            raise _Declined
                        table = item.table.lower()
                        chosen = [i for i, c in enumerate(relation.columns) if c.table == table]
                    if not chosen:
                        raise _Declined
                    for index in chosen:
                        columns.append(_Column(None, relation.columns[index].name, relation.columns[index].shown))
                        picks.append(index)
                else:
                    columns.append(self.output_column(item))
                    picks.append(item.unalias())
            rows = [tuple(row[p] if isinstance(p, int) else self.value(p, scope(row)) for p in picks) for row in relation.rows]
        if not columns:
            raise _Declined
        if select.args.get("distinct") is not None:
            if _extra_args(select.args["distinct"], ()):
                raise _Declined  # DISTINCT ON
            rows = list(dict.fromkeys(rows))
        for position in range(len(columns)):
            _one_kind([row[position] for row in rows])
        return self.sized(_Relation(columns, rows))

    def output_column(self, item: exp.Expression) -> _Column:
        if isinstance(item, exp.Alias):
            alias = item.args.get("alias")
            if not isinstance(alias, exp.Identifier) or not alias.name:
                raise _Declined
            return _Column(None, alias.name.lower(), alias)
        if isinstance(item, exp.Column) and isinstance(item.this, exp.Identifier):
            return _Column(None, item.name.lower(), item.this)
        return _Column(None, None)

    def own_aggregates(self, node: exp.Expression, select: exp.Select) -> list[exp.Expression]:
        found = [n for n in node.find_all(exp.AggFunc) if n.find_ancestor(exp.Select) is select]
        if any(not isinstance(n, _AGGREGATES) or n.find_ancestor(exp.Window) is not None for n in found) or node.find(exp.Window) is not None:
            raise _Declined
        return found

    def check_group_names(self, keys: list[exp.Expression], items: list[exp.Expression]) -> None:
        """A bare name in GROUP BY that is also a select alias of something else could mean either: declined."""

        aliased = {item.alias.lower(): item.this for item in items if isinstance(item, exp.Alias) and item.alias}
        for key in keys:
            if isinstance(key, exp.Literal):
                raise _Declined  # an ordinal that was not resolved
            for column in key.find_all(exp.Column):
                if not column.table and column.name.lower() in aliased and aliased[column.name.lower()].sql() != column.sql():
                    raise _Declined

    def join(self, left: _Relation, join: exp.Join, aliases: set[str], outer: _Scope | None) -> tuple[_Relation, str | None]:
        if _extra_args(join, ("this", "on", "side", "kind")):
            raise _Declined
        side, kind = (join.args.get("side") or "").upper(), (join.args.get("kind") or "").upper()
        if side not in ("", "LEFT", "RIGHT", "FULL") or kind not in ("", "INNER", "CROSS", "OUTER") or (kind == "OUTER" and not side):
            raise _Declined
        on = join.args.get("on")
        if (kind == "CROSS" and (on is not None or side)) or ((side or kind == "INNER") and on is None):
            raise _Declined
        right, alias = self.source(join.this, outer)
        if len(left.rows) * len(right.rows) > _MAX_ROWS * 8:
            raise _Declined
        columns = left.columns + right.columns
        names = frozenset(aliases | ({alias} if alias else set()))
        rows, matched_left, matched_right = [], set(), set()
        for i, a in enumerate(left.rows):
            for j, b in enumerate(right.rows):
                row = a + b
                if on is None or self.value(on, _Scope(columns, row, names, outer)) is True:
                    rows.append(row)
                    matched_left.add(i)
                    matched_right.add(j)
        if side in ("LEFT", "FULL"):
            pad = tuple(None for _ in right.columns)
            rows += [a + pad for i, a in enumerate(left.rows) if i not in matched_left]
        if side in ("RIGHT", "FULL"):
            pad = tuple(None for _ in left.columns)
            rows += [pad + b for j, b in enumerate(right.rows) if j not in matched_right]
        return self.sized(_Relation(columns, rows)), alias

    # -- expressions --------------------------------------------------------------

    def value(self, node: exp.Expression, scope: _Scope):
        self.steps += 1
        if self.steps > _MAX_STEPS:
            raise _Declined
        if scope.group is not None and not isinstance(node, (exp.Literal, exp.Null, exp.Boolean)):
            text = node.sql()
            if text in scope.group.keys:
                return scope.group.keys[text]
            if isinstance(node, _AGGREGATES):
                return self.aggregate(node, scope)
            if isinstance(node, exp.Column):
                if self.resolves_here(node, scope):
                    raise _Declined  # a column of this select that is neither grouped nor aggregated
                return self.column(node, scope.outer)
        kind = type(node)
        if kind is exp.Paren:
            return self.value(node.this, scope)
        if kind is exp.Literal:
            if node.is_string:
                return str(node.this)
            if not _DIGITS.fullmatch(node.this):
                written = _DECIMAL.fullmatch(node.this)
                if written is None:
                    raise _Declined  # an exponent, a hexadecimal number
                whole, part = (written.group(1), written.group(2)) if written.group(1) is not None else ("", written.group(3))
                return _decimal(whole + part, len(part))
            return _integer(int(node.this))
        if kind is exp.Null:
            return None
        if kind is exp.Boolean:
            return bool(node.this)
        if kind is exp.Column:
            return self.column(node, scope)
        if kind is exp.Neg:
            operand = self.number(node.this, scope)
            return None if operand is None else _integer(-operand)
        if kind in (exp.Add, exp.Sub, exp.Mul):
            a, b = self.number(node.this, scope), self.number(node.expression, scope)
            if a is None or b is None:
                return None
            return _integer(a + b if kind is exp.Add else a - b if kind is exp.Sub else a * b)
        if kind in (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE):
            return self.compare(kind, self.value(node.this, scope), self.value(node.expression, scope))
        if kind is exp.And:
            a, b = self.truth(node.this, scope), self.truth(node.expression, scope)
            return False if a is False or b is False else None if a is None or b is None else True
        if kind is exp.Or:
            a, b = self.truth(node.this, scope), self.truth(node.expression, scope)
            return True if a is True or b is True else None if a is None or b is None else False
        if kind is exp.Not:
            a = self.truth(node.this, scope)
            return None if a is None else not a
        if kind is exp.Is:
            operand = self.value(node.this, scope)
            if isinstance(node.expression, exp.Null):
                return operand is None
            if isinstance(node.expression, exp.Boolean):
                if _kind(operand) not in ("bool", "null"):
                    raise _Declined
                return operand is bool(node.expression.this)
            raise _Declined
        if kind is exp.Between:
            if _extra_args(node, ("this", "low", "high")):
                raise _Declined
            x, low, high = (self.number(node.args[k], scope) for k in ("this", "low", "high"))
            a, b = self.compare(exp.GTE, x, low), self.compare(exp.LTE, x, high)
            return False if a is False or b is False else None if a is None or b is None else True
        if kind is exp.Coalesce:
            if _extra_args(node, ("this", "expressions")):
                raise _Declined
            values = [self.value(arg, scope) for arg in [node.this, *node.expressions]]
            _one_kind(values)
            return next((v for v in values if v is not None), None)
        if kind is exp.If:
            condition = self.truth(node.this, scope)
            yes, no = self.value(node.args["true"], scope), (self.value(node.args["false"], scope) if node.args.get("false") is not None else None)
            _one_kind([yes, no])
            return yes if condition is True else no
        if kind is exp.Case:
            subject = self.value(node.this, scope) if node.args.get("this") is not None else None
            default = self.value(node.args["default"], scope) if node.args.get("default") is not None else None
            arms, compared = [], [subject]
            for arm in node.args.get("ifs") or []:
                if type(arm) is not exp.If or arm.args.get("false") is not None:
                    raise _Declined
                if node.args.get("this") is not None:
                    candidate = self.value(arm.this, scope)
                    compared.append(candidate)
                    test = self.compare(exp.EQ, subject, candidate)
                else:
                    test = self.truth(arm.this, scope)
                arms.append((test, self.value(arm.args["true"], scope)))
            _one_reading(compared)
            _one_kind([default, *(value for _, value in arms)])
            return next((value for test, value in arms if test is True), default)
        if kind is exp.Nullif:
            a, b = self.value(node.this, scope), self.value(node.expression, scope)
            _one_kind([a, b])  # the result has the type the two share
            return None if self.compare(exp.EQ, a, b) is True else a
        if kind in (exp.Cast, exp.TryCast):
            to = node.args.get("to")
            if _extra_args(node, ("this", "to", "safe")) or not isinstance(to, exp.DataType):
                raise _Declined
            if to.this in _DECIMAL_TYPES:
                return self.to_numeric(self.value(node.this, scope), to)
            if to.expressions:
                raise _Declined
            operand = self.value(node.this, scope)
            wanted = "int" if to.this in _INTEGER_TYPES else "bool" if to.this is exp.DataType.Type.BOOLEAN else "string" if to.this in _STRING_TYPES else None
            if wanted is None or _kind(operand) not in ("null", wanted):
                raise _Declined
            return operand
        if kind is exp.In:
            return self.membership(node, scope)
        if kind is exp.Bracket:
            return self.element(node, scope)
        if kind is exp.Exists:
            return bool(self.query(node.this, scope).rows)
        if kind is exp.Subquery:
            relation = self.query(node, scope)
            if len(relation.columns) != 1 or len(relation.rows) > 1:
                raise _Declined  # more than one row is an error
            return relation.rows[0][0] if relation.rows else None
        raise _Declined

    def to_numeric(self, operand, to: exp.DataType):
        """``CAST(integer AS NUMERIC[(precision[, scale])])``: the same number, when the type holds it."""

        if _kind(operand) not in ("int", "null"):
            raise _Declined  # a FLOAT64 is rounded on the way
        sizes = []
        for parameter in to.expressions:
            size = parameter.this if isinstance(parameter, exp.DataTypeParam) and not _extra_args(parameter, ("this",)) else None
            if not isinstance(size, exp.Literal) or size.is_string or not _DIGITS.fullmatch(size.this):
                raise _Declined
            sizes.append(int(size.this))
        if len(sizes) > 2:
            raise _Declined
        precision, scale = (sizes + [0])[:2] if sizes else (_EXACT_DIGITS, 0)
        if operand is None:
            return None
        if scale > precision or abs(operand) >= 10 ** min(precision - scale, _EXACT_DIGITS):
            raise _Declined  # out of range is an error
        return Fraction(operand)

    def element(self, node: exp.Bracket, scope: _Scope):
        """``[a, b, c][OFFSET(i)]`` and the ``ORDINAL`` / ``SAFE_`` forms over an array written in place."""

        offset = node.args.get("offset")
        if _extra_args(node, ("this", "expressions", "offset", "safe")) or type(node.this) is not exp.Array or len(node.expressions) != 1 or offset not in (0, 1) or type(offset) is not int:
            raise _Declined
        values = [self.value(item, scope) for item in node.this.expressions]
        _one_kind(values)
        index = self.number(node.expressions[0], scope)
        if index is None:
            return None
        if not 0 <= index - offset < len(values):
            if node.args.get("safe"):
                return None
            raise _Declined  # out of bounds is an error
        return values[index - offset]

    def number(self, node: exp.Expression, scope: _Scope) -> int | None:
        operand = self.value(node, scope)
        if _kind(operand) not in ("int", "null"):
            raise _Declined
        return operand

    def truth(self, node: exp.Expression, scope: _Scope) -> bool | None:
        operand = self.value(node, scope)
        if _kind(operand) not in ("bool", "null"):
            raise _Declined
        return operand

    def compare(self, kind, a, b) -> bool | None:
        kinds = {_kind(a), _kind(b)} - {"null"}
        if kinds == {"int", "decimal"}:  # the integer is read as a FLOAT64 or NUMERIC: exact when it is short
            _one_reading([a, b])
        elif len(kinds) > 1 or (kinds and not kinds <= {"int", "decimal"} and kind not in (exp.EQ, exp.NEQ)):
            raise _Declined
        if a is None or b is None:
            return None
        if kind is exp.EQ:
            return a == b
        if kind is exp.NEQ:
            return a != b
        return a < b if kind is exp.LT else a <= b if kind is exp.LTE else a > b if kind is exp.GT else a >= b

    def membership(self, node: exp.In, scope: _Scope) -> bool | None:
        if _extra_args(node, ("this", "expressions", "query", "unnest")):
            raise _Declined
        given = [key for key in ("expressions", "query", "unnest") if node.args.get(key)]
        if len(given) != 1:
            raise _Declined
        needle = self.value(node.this, scope)
        if given == ["expressions"]:
            candidates = [self.value(item, scope) for item in node.expressions]
        elif given == ["query"]:
            relation = self.query(node.args["query"], scope)
            if len(relation.columns) != 1:
                raise _Declined
            candidates = [row[0] for row in relation.rows]
        else:
            unnest = node.args["unnest"]
            if not isinstance(unnest, exp.Unnest) or _extra_args(unnest, ("expressions", "explode_array")) or len(unnest.expressions) != 1 or type(unnest.expressions[0]) is not exp.Array:
                raise _Declined
            candidates = self.array(unnest.expressions[0])
        _one_reading([needle, *candidates])
        tests = [self.compare(exp.EQ, needle, candidate) for candidate in candidates]
        return True if True in tests else None if None in tests else False

    def aggregate(self, node: exp.Expression, scope: _Scope):
        group = scope.group
        if _extra_args(node, ("this", "big_int")):
            raise _Declined
        if isinstance(node.parent, (exp.Filter, exp.Window, exp.IgnoreNulls, exp.RespectNulls, exp.WithinGroup)):
            raise _Declined
        argument = node.this
        distinct = isinstance(argument, exp.Distinct)
        if distinct:
            if not isinstance(node, exp.Count) or len(argument.expressions) != 1 or _extra_args(argument, ("expressions",)):
                raise _Declined
            argument = argument.expressions[0]
        if isinstance(node, exp.Count) and isinstance(argument, exp.Star):
            if _extra_args(argument, ()):
                raise _Declined
            return len(group.rows)
        if argument is None or argument.find(exp.AggFunc, exp.Select) is not None:
            raise _Declined
        if any(not self.resolves_here(column, scope) for column in argument.find_all(exp.Column)):
            raise _Declined  # an aggregate over an enclosing query's column belongs to that query
        values = [self.value(argument, _Scope(scope.columns, row, scope.aliases, scope.outer)) for row in group.rows]
        _one_kind(values)
        present = [v for v in values if v is not None]
        if isinstance(node, exp.Count):
            return len(set(present)) if distinct else len(present)
        if any(_kind(v) != "int" for v in present):
            raise _Declined
        if not present:
            return None
        return _integer(sum(present)) if isinstance(node, exp.Sum) else min(present) if isinstance(node, exp.Min) else max(present)

    def matches(self, column: exp.Column, scope: _Scope) -> list[int]:
        parts = column.parts
        if not 1 <= len(parts) <= 2 or not all(isinstance(part, exp.Identifier) for part in parts):
            raise _Declined
        name = parts[-1].name.lower()
        table = parts[0].name.lower() if len(parts) == 2 else None
        if table is None and name in scope.aliases:
            raise _Declined  # in BigQuery a bare source alias is the whole row
        return [i for i, c in enumerate(scope.columns) if c.name == name and (table is None or c.table == table)]

    def resolves_here(self, column: exp.Column, scope: _Scope) -> bool:
        found = self.matches(column, scope)
        if len(found) > 1:
            raise _Declined
        return bool(found)

    def column(self, column: exp.Column, scope: _Scope | None):
        while scope is not None:
            found = self.matches(column, scope)
            if len(found) > 1:
                raise _Declined  # ambiguous
            if found:
                if scope.group is not None:
                    text = column.sql()
                    if text in scope.group.keys:
                        return scope.group.keys[text]
                    raise _Declined  # read from inside a subquery of a grouped select: not a grouping expression
                return scope.row[found[0]]
            scope = scope.outer
        raise _Declined


def _literal(value) -> exp.Expression:
    if value is None:
        return exp.Null()
    if type(value) is bool:
        return exp.Boolean(this=value)
    if type(value) is int:
        number = exp.Literal.number(abs(value))
        return exp.Neg(this=number) if value < 0 else number
    return exp.Literal.string(value)


def _order(row: tuple) -> tuple:
    return tuple((0, 0) if v is None else (1, int(v)) if type(v) is bool else (2, v) if type(v) is int else (3, v) for v in row)


def evaluate(tree: exp.Expression) -> tuple[list[exp.Identifier | None], list[tuple]] | None:
    """The output names and the rows of a query that reads no table, or None when it is outside the fragment."""

    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)) or tree.find(exp.Table) is not None:
        return None
    try:
        relation = _Interpreter().query(tree, None)
    except _Declined:
        return None
    if len(relation.rows) > _MAX_OUTPUT_ROWS:
        return None
    for position in range(len(relation.columns)):
        column = [row[position] for row in relation.rows]
        kinds = {_kind(v) for v in column} - {"null"}
        if len(kinds) > 1 or "decimal" in kinds:  # a decimal result would need its type (FLOAT64 or NUMERIC) written back
            return None
    return [c.shown for c in relation.columns], relation.rows


def fold_closed_query(tree: exp.Expression, dialect: str = "bigquery") -> exp.Expression:
    """``tree`` as the list of its rows when it reads no table and the interpreter covers it; ``tree`` itself otherwise."""

    if dialect != "bigquery":
        return tree
    result = evaluate(tree)
    if result is None:
        return tree
    names, rows = result

    def select(values, named: bool) -> exp.Select:
        items = []
        for name, value in zip(names, values):
            literal = _literal(value)
            items.append(exp.Alias(this=literal, alias=exp.Identifier(this=name.name, quoted=bool(name.args.get("quoted")))) if named and name is not None else literal)
        return exp.Select(expressions=items)

    if not rows:
        empty = select([None] * len(names), True)
        empty.set("where", exp.Where(this=exp.Boolean(this=False)))
        return empty
    folded: exp.Expression | None = None
    for row in sorted(rows, key=_order):
        branch = select(row, folded is None)
        folded = branch if folded is None else exp.Union(this=folded, expression=branch, distinct=False)
    return folded
