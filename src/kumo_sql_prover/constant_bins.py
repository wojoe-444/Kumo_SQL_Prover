"""A list of constant rows outer-joined to a table and aggregated per row is one aggregate query per row.

    SELECT b.bin, COUNT(s.id) AS total
    FROM (SELECT 'short' AS bin UNION ALL SELECT 'long') AS b
    LEFT JOIN (SELECT id, CASE WHEN secs < 300 THEN 'short' ELSE 'long' END AS bin FROM sessions) AS s ON s.bin = b.bin
    GROUP BY b.bin

    SELECT 'short' AS bin, COUNT(s.id) AS total FROM (SELECT id, CASE .. END AS bin FROM sessions) AS s WHERE s.bin = 'short'
    UNION ALL
    SELECT 'long' AS bin, COUNT(s.id) AS total FROM (SELECT id, CASE .. END AS bin FROM sessions) AS s WHERE s.bin = 'long'

A histogram that must show a bin with no rows is written the first way (the bins are spelled out and the rows are
outer-joined to them) or the second way (one counting query per bin, glued with ``UNION``). The prover reads the
second; this module writes the first as the second.

**Why it holds (the grouped form).** ``B`` is the list of constant rows and ``X`` the other source. ``B LEFT JOIN X
ON c`` gives, for each row ``r`` of ``B``, the rows ``(r, x)`` with ``c(r, x)`` TRUE, or the one row ``(r, NULLs)`` when
there is none. The values of the grouped column differ from row to row of ``B`` (checked, see below), so ``GROUP BY``
that column makes exactly one group per row of ``B``, and a ``UNION`` inside ``B`` removes nothing. Over the group of
``r``, an aggregate ``COUNT``, ``SUM``, ``MIN``, ``MAX`` or ``AVG`` of a plain column of ``X`` (with or without
``DISTINCT``) is that aggregate over the rows ``x`` with ``c(r, x)`` TRUE: when there are such rows they are the
group, and when there are none the group is one row whose column is NULL, which every one of these aggregates
ignores, so the value is that of the aggregate over no rows (0 for ``COUNT``, else NULL). ``COUNT(*)`` and an
aggregate of anything but a plain column would see the NULL row, so they are not read. That is the scalar query
``SELECT .. FROM X WHERE c(r, x)`` with ``r`` written in as constants: one row, the same values.

**The form counted first.** ``B LEFT JOIN (SELECT k, agg AS a .. FROM Y GROUP BY k) AS T ON b.col = T.k`` with no
``GROUP BY`` of its own: each row ``r`` of ``B`` meets the one group whose key is ``r.col``, or none. So the row
shows the aggregates over the rows of ``Y`` with ``k = r.col``, or NULLs. Over no rows ``SUM``, ``MIN``, ``MAX`` and
``AVG`` are NULL as well, and ``COUNT`` is 0 where the join shows NULL; so a ``COUNT`` output may be read only as
``COALESCE(T.a, 0)`` or ``IFNULL(T.a, 0)``. It becomes ``SELECT .. FROM Y WHERE k = r.col`` without the grouping.

**What a constant may be compared with.** Writing a constant where a column of ``B`` stood changes nothing only if
the comparison is the same comparison. A text constant must be compared with a value seen to be text (a column
declared text, or an output that is a ``CASE`` or ``IF`` of text literals), a number with whole-number columns and
number literals under ``+ - * /``; MySQL compares a text with a number as numbers, and the prover as text. A column
of ``B`` may stand only as one side of a comparison or under ``IS NULL``. Every text constant of the grouped or
joined column, and every text it may be compared with, must be plain ASCII and equal or still different once
letter case and trailing spaces are ignored: the prover compares texts exactly, and a collation that called two of
them equal would join or group them.

**What is not read.** An inner join or a join that preserves the other side; more than one join; a ``WHERE``,
``HAVING``, ``DISTINCT``, ``ORDER BY`` or ``LIMIT`` on the select; ``COUNT(*)``; a ``GROUP BY`` on anything but one
column of the constant rows; constant rows whose grouped or joined values are not seen to differ (a NULL, a
computed value, text and number mixed); a name whose owner is not certain (bare, when the other source's columns
are not known).

**Where the selects go.** In place of the select they replace: as the whole query, as a derived table or subquery,
or as one side of a ``UNION``. On the right side of a ``UNION`` that removes duplicates they are glued with ``UNION``
too, because ``A UNION b1 UNION ALL b2`` written without parentheses is ``(A UNION b1) UNION ALL b2``; the duplicates
are removed either way. Beside ``EXCEPT`` or ``INTERSECT`` nothing is written.

A pre-pass of the sixteenth hardest-cases batch (``shape_fallback.SIXTEENTH_BATCH``): it fires only in the last step
of a proof. ``tests/test_constant_bins.py`` holds the rewritten texts, a replay against DuckDB on made-up rows, the
selects left alone and the pairs proved only with the rule.
"""

from __future__ import annotations

from fractions import Fraction

from sqlglot import exp

from . import smt_values
from .ast_utils import select_sources
from .shifted_equalities import _source_kind

NAME = "constant_bins"

_COMPARISONS = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE)
_FLIPPED = {exp.EQ: exp.EQ, exp.NEQ: exp.NEQ, exp.LT: exp.GT, exp.LTE: exp.GTE, exp.GT: exp.LT, exp.GTE: exp.LTE}
_AGGREGATES = (exp.Count, exp.Sum, exp.Min, exp.Max, exp.Avg)
_ARITHMETIC = (exp.Add, exp.Sub, exp.Mul, exp.Div)
_PLAIN_SELECT = ("where", "group", "having", "order", "limit", "offset", "distinct", "with_", "with", "windows", "qualify", "joins", "laterals", "into", "locks")


class _Declined(Exception):
    """The select is not one this module reads."""


def _unparen(node: exp.Expression | None) -> exp.Expression | None:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _value(node: exp.Expression | None) -> tuple[str, object] | None:
    """``("text", s)`` or ``("number", n)`` for a literal, ``("null", None)`` for NULL; None for anything else."""

    node = _unparen(node)
    if isinstance(node, exp.Null):
        return ("null", None)
    sign = 1
    if isinstance(node, exp.Neg):
        sign, node = -1, _unparen(node.this)
    if not isinstance(node, exp.Literal):
        return None
    if node.is_string:
        return ("text", str(node.this)) if sign == 1 else None
    try:
        return ("number", sign * Fraction(str(node.this)))
    except (ValueError, ZeroDivisionError):
        return None


def _constant_kind(node: exp.Expression) -> str | None:
    """``"text"``, ``"number"`` or ``"null"`` for a constant a row may hold: a literal, NULL, or number literals under ``+ - *``."""

    found = _value(node)
    if found is not None:
        return found[0]
    node = _unparen(node)
    if isinstance(node, (exp.Add, exp.Sub, exp.Mul)) and _constant_kind(node.this) == "number" and _constant_kind(node.expression) == "number":
        return "number"
    return None


def _fold(text: str) -> str | None:
    return text.casefold().rstrip(" ") if text.isascii() else None


def _texts_safe(texts: set[str]) -> bool:
    """Whether the texts are plain ASCII and no two different ones are equal once case and trailing spaces are ignored."""

    folded = {_fold(text) for text in texts}
    return None not in folded and len(folded) == len(texts)


def _differ(values: list[tuple[str, object] | None]) -> bool:
    """Whether the values are all texts or all numbers and no two are the same (texts: also not under case or padding)."""

    if not values or any(value is None for value in values) or len({value[0] for value in values}) != 1 or values[0][0] == "null":
        return False
    if values[0][0] == "number":
        return len({value[1] for value in values}) == len(values)
    folded = [_fold(value[1]) for value in values]
    return None not in folded and len(set(folded)) == len(values)


def _branches(query: exp.Expression | None) -> list[exp.Select] | None:
    query = _unparen(query)
    while type(query) is exp.Subquery and not query.alias and not any(value for key, value in query.args.items() if key != "this"):
        query = _unparen(query.this)
    if type(query) is exp.Select:
        return [query]
    if type(query) is exp.Union and not any(query.args.get(key) for key in ("order", "limit", "offset", "with_", "with", "by_name")):
        left, right = _branches(query.this), _branches(query.expression)
        return None if left is None or right is None else left + right
    return None


def _constant_rows(source: exp.Expression) -> tuple[list[str], list[list[exp.Expression]]] | None:
    """The output names and the rows of a derived table that is a ``UNION`` or ``UNION ALL`` of selects of constants."""

    if type(source) is not exp.Subquery or not source.alias or source.args["alias"].args.get("columns"):
        return None
    if any(value for key, value in source.args.items() if key not in ("this", "alias")):
        return None
    body = source.this
    while (
        type(body) is exp.Select
        and len(body.expressions) == 1
        and type(body.expressions[0]) is exp.Star
        and not any(body.expressions[0].args.values())
        and not any(body.args.get(key) for key in _PLAIN_SELECT)
        and type((body.args.get("from_") or body.args.get("from")).this) is exp.Subquery
        and not any(value for key, value in (body.args.get("from_") or body.args.get("from")).this.args.items() if key not in ("this", "alias"))
        and not ((body.args.get("from_") or body.args.get("from")).this.args.get("alias") is not None and (body.args.get("from_") or body.args.get("from")).this.args["alias"].args.get("columns"))
    ):
        body = (body.args.get("from_") or body.args.get("from")).this.this  # SELECT * FROM (the rows) AS a
    selects = _branches(body)
    if not selects:
        return None
    rows = []
    for select in selects:
        if select.args.get("from_") or select.args.get("from") or any(select.args.get(key) for key in _PLAIN_SELECT):
            return None
        row = [item.this if isinstance(item, exp.Alias) else item for item in select.expressions]
        if any(_constant_kind(value) is None for value in row):
            return None
        rows.append(row)
    first = selects[0].expressions
    if not all(isinstance(item, exp.Alias) for item in first) or any(len(row) != len(first) for row in rows):
        return None
    names = [item.alias.lower() for item in first]
    return (names, rows) if len(set(names)) == len(names) else None


def _columns_of(source: exp.Expression, schema: dict | None) -> set[str] | None:
    """The column names a source shows, when they are known."""

    if isinstance(source, exp.Table):
        if source.args.get("db") or source.args.get("catalog"):
            return None
        for table, columns in (schema or {}).items():
            if str(table).lower() == source.name.lower():
                return {str(column).lower() for column in columns}
        return None
    if type(source) is exp.Subquery and type(source.this) is exp.Select and source.args.get("alias") is not None and not source.args["alias"].args.get("columns"):
        names: set[str] = set()
        for item in source.this.expressions:
            if type(item) is exp.Star and not any(item.args.values()):
                starred = _starred(source.this, schema)
                if starred is None:
                    return None
                names |= starred
            elif item.is_star or not item.alias_or_name:
                return None
            else:
                names.add(item.alias_or_name.lower())
        return names
    return None


def _starred(select: exp.Select, schema: dict | None) -> set[str] | None:
    """The columns a plain ``*`` of ``select`` shows, when every source is a table with known columns and no join merges any."""

    sources = select_sources(select)
    if not sources or any(join.args.get("using") or join.args.get("method") for join in select.args.get("joins") or []):
        return None
    names: set[str] = set()
    for source in sources:
        columns = _columns_of(source, schema) if isinstance(source, exp.Table) else None
        if columns is None:
            return None
        names |= columns
    return names


def _text_source(source: exp.Expression, name: str, types: dict, texts: set[str], depth: int = 0) -> bool:
    """Whether the output ``name`` of ``source`` is seen to be text; the literals it may be are added to ``texts``."""

    if isinstance(source, exp.Table):
        if source.args.get("db") or source.args.get("catalog"):
            return False
        declared = (types.get(source.name.lower()) or {}).get(name)
        return bool(declared) and smt_values.type_class(str(declared)) == smt_values.STRING
    if type(source) is not exp.Subquery or type(source.this) is not exp.Select or depth > 4 or source.args["alias"].args.get("columns"):
        return False
    body = source.this
    if body.args.get("with_") or body.args.get("with"):
        return False
    items = [item for item in body.expressions if not item.is_star and (item.alias_or_name or "").lower() == name]
    if len(items) != 1:
        return False
    for item in body.expressions:
        if item.is_star:  # a star beside the output: it must not show a column of the same name
            starred = _starred(body, types) if type(item) is exp.Star and not any(item.args.values()) else None
            if starred is None or name in starred:
                return False
    return _text_value(items[0].unalias(), body, types, texts, depth + 1)


def _text_value(value: exp.Expression, select: exp.Select, types: dict, texts: set[str], depth: int = 0) -> bool:
    """Whether ``value``, read in ``select``, is seen to be text: a text literal, a ``CASE`` or ``IF`` of them, or such a column."""

    value = _unparen(value)
    if isinstance(value, exp.Literal):
        if value.is_string:
            texts.add(str(value.this))
        return value.is_string
    if type(value) in (exp.Case, exp.If):
        results = [branch.args.get("true") for branch in value.args.get("ifs") or []] if type(value) is exp.Case else [value.args.get("true")]
        last = value.args.get("default") if type(value) is exp.Case else value.args.get("false")
        results = [_unparen(result) for result in results + ([last] if last is not None else [])]
        literal = [result for result in results if isinstance(result, exp.Literal) and result.is_string]
        if not literal or any(not isinstance(result, exp.Null) for result in results if result not in literal):
            return False
        texts.update(str(result.this) for result in literal)
        return True
    if isinstance(value, exp.Column) and isinstance(value.this, exp.Identifier) and not value.args.get("db"):
        sources = [source for source in select_sources(select) if isinstance(source, (exp.Table, exp.Subquery)) and source.alias_or_name]
        if len(sources) != len(select_sources(select)):
            return False
        if value.table:
            sources = [source for source in sources if source.alias_or_name.lower() == value.table.lower()]
        return len(sources) == 1 and _text_source(sources[0], value.name.lower(), types, texts, depth)
    return False


class _Reader:
    """One select of the form ``B LEFT JOIN X`` (or ``X RIGHT JOIN B``): who owns each name, and what may replace it."""

    def __init__(self, select: exp.Select, schema: dict | None, types: dict, dialect: str):
        self.select, self.schema, self.types, self.dialect = select, schema, types, dialect
        if any(select.args.get(key) for key in ("where", "having", "order", "limit", "offset", "distinct", "with_", "with", "windows", "qualify", "laterals", "into", "locks")):
            raise _Declined
        source = select.args.get("from_") or select.args.get("from")
        joins = select.args.get("joins") or []
        if source is None or len(joins) != 1:
            raise _Declined
        join = joins[0]
        side, kind = str(join.args.get("side") or "").upper(), str(join.args.get("kind") or "").upper()
        if kind not in ("", "OUTER") or join.args.get("method") or any(join.args.get(key) for key in ("global_", "global", "hint", "match_condition")):
            raise _Declined
        if side == "LEFT":
            self.bins, self.other = source.this, join.this
        elif side == "RIGHT":
            self.bins, self.other = join.this, source.this
        else:
            raise _Declined
        rows = _constant_rows(self.bins)
        if rows is None or not isinstance(self.other, (exp.Table, exp.Subquery)) or not self.other.alias_or_name:
            raise _Declined
        if isinstance(self.other, exp.Subquery) and any(value for key, value in self.other.args.items() if key not in ("this", "alias")):
            raise _Declined
        self.names, self.rows = rows
        self.bins_alias, self.other_alias = self.bins.alias.lower(), self.other.alias_or_name.lower()
        if self.bins_alias == self.other_alias:
            raise _Declined
        self.other_columns = _columns_of(self.other, schema)
        self.kinds = []
        for position in range(len(self.names)):
            kinds = {_constant_kind(row[position]) for row in self.rows} - {"null"}
            if len(kinds) > 1:
                raise _Declined  # text and numbers in one column
            self.kinds.append(kinds.pop() if kinds else "null")
        self.texts: set[str] = set()  # every text a text constant may be compared with, and those constants
        using = [identifier.name.lower() for identifier in join.args.get("using") or []]
        if using and join.args.get("on") is not None:
            raise _Declined
        self.using = using
        if any(name not in self.names or self.other_columns is None or name not in self.other_columns for name in using):
            raise _Declined
        self.condition = join.args.get("on")
        if self.condition is None and not using:
            raise _Declined

    def owner(self, column: exp.Column) -> str:
        """``"bins"`` or ``"other"``; declines a name whose owner is not certain."""

        if not isinstance(column.this, exp.Identifier) or column.args.get("db"):
            raise _Declined
        name, table = column.name.lower(), column.table.lower()
        if table:
            if table == self.bins_alias and name in self.names:
                return "bins"
            if table == self.other_alias and (self.other_columns is None or name in self.other_columns):
                return "other"
            raise _Declined
        if name in self.using:
            return "bins"  # the joined column of an outer join that keeps B is B's
        if self.other_columns is None:
            raise _Declined
        if (name in self.names) != (name in self.other_columns):
            return "bins" if name in self.names else "other"
        raise _Declined

    def position(self, column: exp.Column) -> int:
        return self.names.index(column.name.lower())

    def check_text(self, position: int, texts: set[str]) -> None:
        """A text constant of column ``position`` is compared with a value that may be one of ``texts``."""

        self.texts |= texts | {_value(row[position])[1] for row in self.rows if _constant_kind(row[position]) == "text"}

    def numeric(self, value: exp.Expression, select: exp.Select, source: exp.Expression | None) -> bool:
        """Whether ``value`` is whole-number columns (of ``source`` when given, else of ``select``'s sources) and number
        literals under ``+ - * /``."""

        columns = 0
        for node in value.walk():
            if isinstance(node, exp.Column):
                columns += 1
                if source is not None:
                    if self.owner(node) != "other" or _source_kind(source, node.name, self.types, self.dialect) != "int":
                        return False
                else:
                    from .shifted_equalities import _column_kind

                    if _column_kind(node, select, self.types, self.dialect) != "int":
                        return False
            elif isinstance(node, exp.Literal):
                if node.is_string:
                    return False
            elif not isinstance(node, (exp.Paren, exp.Neg, exp.Identifier, *_ARITHMETIC)):
                return False
        return columns > 0

    def check_condition(self) -> None:
        """Every column of B in the ON condition stands where a constant of its kind means the same thing."""

        for column in list(self.condition.find_all(exp.Column)):
            if self.owner(column) != "bins":
                continue
            position, kind = self.position(column), self.kinds[self.position(column)]
            parent, child = column.parent, column
            while isinstance(parent, exp.Paren):
                parent, child = parent.parent, parent
            if isinstance(parent, exp.Is) and parent.this is child and isinstance(parent.expression, exp.Null):
                continue
            if not isinstance(parent, _COMPARISONS):
                raise _Declined
            operand = _unparen(parent.expression if parent.this is child else parent.this)
            if any(self.owner(inner) == "bins" for inner in operand.find_all(exp.Column)):
                raise _Declined
            if kind == "null":
                continue
            if kind == "text":
                texts: set[str] = set()
                if not (isinstance(operand, exp.Column) and _text_source(self.other, operand.name.lower(), self.types, texts)):
                    raise _Declined
                self.check_text(position, texts)
            elif not self.numeric(operand, self.select, self.other):
                raise _Declined
        if any(self.condition.find_all(exp.Subquery, exp.Window, exp.AggFunc)):
            raise _Declined

    def constants(self, node: exp.Expression, row: list[exp.Expression]) -> exp.Expression:
        """A copy of ``node`` with each column of B replaced by its constant in ``row``."""

        node = exp.Paren(this=node.copy())  # a holder, so that a bare column can be replaced too
        for column in list(node.find_all(exp.Column)):
            if self.owner(column) == "bins":
                constant = row[self.position(column)].copy()
                constant = column.replace(constant if isinstance(constant, (exp.Literal, exp.Null)) else exp.Paren(this=constant))
                test = constant.parent
                if isinstance(test, _COMPARISONS) and test.this is constant:
                    # written with the constant last, the way the rules that read a comparison expect it
                    test.replace(_FLIPPED[type(test)](this=test.expression.copy(), expression=constant.copy()))
        return node.this


def _named(item: exp.Expression, value: exp.Expression) -> exp.Expression:
    """``value`` under the output name of ``item``."""

    if isinstance(item, exp.Alias):
        return exp.Alias(this=value, alias=item.args["alias"].copy())
    if isinstance(item, exp.Column):
        return exp.Alias(this=value, alias=exp.to_identifier(item.name))
    return value


def _union(selects: list[exp.Select], distinct: bool = False) -> exp.Expression:
    query: exp.Expression = selects[0]
    for select in selects[1:]:
        query = exp.Union(this=query, expression=select, distinct=distinct)
    return query


def _grouped(reader: _Reader) -> list[exp.Select]:
    """The grouped form (module doc): one scalar aggregate select per constant row."""

    select = reader.select
    group = select.args["group"]
    if len(group.expressions) != 1 or any(value for key, value in group.args.items() if key != "expressions"):
        raise _Declined
    key = _unparen(group.expressions[0])
    if not isinstance(key, exp.Column) or reader.owner(key) != "bins":
        raise _Declined
    label = reader.position(key)
    if not _differ([_value(row[label]) for row in reader.rows]):
        raise _Declined
    if reader.using:
        for name in reader.using:
            position = reader.names.index(name)
            kind = reader.kinds[position]
            texts: set[str] = set()
            if kind == "text" and _text_source(reader.other, name, reader.types, texts):
                reader.check_text(position, texts)
            elif kind != "number" or _source_kind(reader.other, name, reader.types, reader.dialect) != "int":
                raise _Declined
            if any(_constant_kind(row[position]) == "null" for row in reader.rows):
                raise _Declined
    else:
        reader.check_condition()
    if reader.kinds[label] == "text":
        reader.check_text(label, set())
    if not _texts_safe(reader.texts):
        raise _Declined
    aggregates = 0
    for item in select.expressions:
        if item.is_star or any(item.find_all(exp.Subquery, exp.Window, exp.Select)):
            raise _Declined
        inside: set[int] = set()
        for aggregate in item.find_all(exp.AggFunc):
            argument = _unparen(aggregate.this)
            if isinstance(argument, exp.Distinct):
                argument = _unparen(argument.expressions[0]) if len(argument.expressions) == 1 and not argument.args.get("on") else None
            extra = [key for key, value in aggregate.args.items() if value and key not in ("this", "big_int")]
            if type(aggregate) not in _AGGREGATES or extra or not isinstance(argument, exp.Column) or reader.owner(argument) != "other":
                raise _Declined
            inside.add(id(argument))
            aggregates += 1
        if any(id(column) not in inside and reader.owner(column) != "bins" for column in item.find_all(exp.Column)):
            raise _Declined
    if not aggregates:
        raise _Declined
    branches = []
    for row in reader.rows:
        if reader.using:
            tests = [exp.EQ(this=exp.column(name, table=reader.other.alias_or_name), expression=row[reader.names.index(name)].copy()) for name in reader.using]
            condition = tests[0] if len(tests) == 1 else exp.and_(*tests)
        else:
            condition = reader.constants(reader.condition, row)
        items = [_named(item, reader.constants(item.unalias(), row)) for item in select.expressions]
        branches.append(exp.Select(expressions=items).from_(reader.other.copy()).where(condition))
    return branches


def _counted_first(reader: _Reader) -> list[exp.Select]:
    """The form counted first (module doc): the grouped derived table, read once per constant row without its grouping."""

    select, other = reader.select, reader.other
    if type(other) is not exp.Subquery or type(other.this) is not exp.Select or other.args["alias"].args.get("columns"):
        raise _Declined
    inner = other.this
    if any(inner.args.get(key) for key in ("having", "order", "limit", "offset", "distinct", "with_", "with", "windows", "qualify", "laterals", "into", "locks")):
        raise _Declined
    group = inner.args.get("group")
    if group is None or len(group.expressions) != 1 or any(value for key, value in group.args.items() if key != "expressions"):
        raise _Declined
    key = _unparen(group.expressions[0])
    if isinstance(key, exp.Column) and not key.table:
        # GROUP BY a bare name: an output of that name when no source has such a column (the engines look there first)
        named = [item for item in inner.expressions if isinstance(item, exp.Alias) and item.alias.lower() == key.name.lower()]
        shown = _starred(inner, reader.schema)
        if len(named) == 1 and not (isinstance(named[0].this, exp.Column) and named[0].this.name.lower() == key.name.lower()):
            if shown is None or key.name.lower() in shown:
                raise _Declined
            key = _unparen(named[0].this)
    outputs: dict[str, tuple[str, exp.Expression]] = {}
    for item in inner.expressions:
        name, value = (item.alias_or_name or "").lower(), _unparen(item.unalias())
        if not name or name in outputs or item.is_star or any(value.find_all(exp.Subquery, exp.Window)):
            raise _Declined
        counted = value
        if isinstance(value, (exp.Coalesce, exp.Anonymous)) and _zero_default(value) is not None:
            counted = _zero_default(value)
        if value.sql() == key.sql() or (isinstance(key, exp.Column) and isinstance(value, exp.Column) and not key.table and key.name.lower() == value.name.lower() == name):
            outputs[name] = ("key", value)
        elif type(counted) is exp.Count:
            outputs[name] = ("count", value)
        elif type(value) in (exp.Sum, exp.Min, exp.Max, exp.Avg):
            outputs[name] = ("null", value)
        else:
            raise _Declined
    # the join: B's column against the grouped key, and nothing else
    if reader.using:
        if len(reader.using) != 1:
            raise _Declined
        label, joined = reader.names.index(reader.using[0]), reader.using[0]
    else:
        test = _unparen(reader.condition)
        if type(test) is not exp.EQ:
            raise _Declined
        sides = [_unparen(test.this), _unparen(test.expression)]
        if not all(isinstance(side, exp.Column) for side in sides) or sorted(reader.owner(side) for side in sides) != ["bins", "other"]:
            raise _Declined
        ours, theirs = sides if reader.owner(sides[0]) == "bins" else sides[::-1]
        label, joined = reader.position(ours), theirs.name.lower()
    if outputs.get(joined, ("", None))[0] != "key":
        raise _Declined
    if not _differ([_value(row[label]) for row in reader.rows]):
        raise _Declined
    if reader.kinds[label] == "text":
        texts: set[str] = set()
        if not _text_value(key, inner, reader.types, texts):
            raise _Declined
        reader.check_text(label, texts)
    elif not reader.numeric(key, inner, None):
        raise _Declined
    if not _texts_safe(reader.texts):
        raise _Declined
    aggregates = 0
    for item in select.expressions:
        if item.is_star or any(item.find_all(exp.Subquery, exp.Window, exp.Select, exp.AggFunc)):
            raise _Declined
        for column in item.find_all(exp.Column):
            if reader.owner(column) == "bins":
                continue
            kind = outputs.get(column.name.lower(), ("", None))[0]
            if kind == "count":
                parent, child = column.parent, column
                while isinstance(parent, exp.Paren):
                    parent, child = parent.parent, parent
                if _zero_default(parent) is not child:
                    raise _Declined  # a count of no rows is 0 where the join shows NULL
            elif kind != "null":
                raise _Declined
            aggregates += 1
    if not aggregates:
        raise _Declined

    def read(item: exp.Expression, row: list[exp.Expression]) -> exp.Expression:
        holder = exp.Paren(this=item.unalias().copy())  # the copy has the names of the select, so the owners are the same
        for column in list(holder.find_all(exp.Column)):
            if reader.owner(column) == "bins":
                constant = row[reader.position(column)].copy()
                column.replace(constant if isinstance(constant, (exp.Literal, exp.Null)) else exp.Paren(this=constant))
            else:
                value = outputs[column.name.lower()][1]
                column.replace(value.copy() if isinstance(value, exp.Func) else exp.Paren(this=value.copy()))
        return _named(item, holder.this)

    branches = []
    for row in reader.rows:
        branch = inner.copy()
        branch.set("group", None)
        branch.set("expressions", [read(item, row) for item in select.expressions])
        branches.append(branch.where(exp.EQ(this=key.copy() if isinstance(key, exp.Column) else exp.Paren(this=key.copy()), expression=row[label].copy())))
    return branches


def _zero_default(node: exp.Expression | None) -> exp.Expression | None:
    """The first argument of ``COALESCE(x, 0)`` or ``IFNULL(x, 0)``, else None."""

    if isinstance(node, exp.Coalesce):
        arguments = [node.this, *node.expressions]
    elif isinstance(node, exp.Anonymous) and str(node.this).upper() in ("IFNULL", "NVL"):
        arguments = list(node.expressions)
    else:
        return None
    if len(arguments) != 2 or _value(arguments[1]) != ("number", Fraction(0)):
        return None
    return arguments[0]


def read_constant_bins(tree: exp.Expression, schema: dict | None, types: dict | None, dialect: str) -> tuple[exp.Expression, bool]:
    """``tree`` with each such outer join to constant rows written as one aggregate select per row; whether it changed."""

    if not any(str(join.args.get("side") or "").upper() in ("LEFT", "RIGHT") for join in tree.find_all(exp.Join)):
        return tree, False
    named = {cte.alias.lower() for cte in tree.find_all(exp.CTE)}  # a WITH name may hide a table: nothing is known of it
    declared = {str(t).lower(): {str(c).lower(): v for c, v in columns.items()} for t, columns in (types or {}).items() if str(t).lower() not in named}
    known = {table: columns for table, columns in (schema or {}).items() if str(table).lower() not in named}
    tree = tree.copy()
    changed = False
    for select in reversed(list(tree.find_all(exp.Select))):
        try:
            reader = _Reader(select, known, declared, dialect)
            branches = _grouped(reader) if select.args.get("group") is not None else _counted_first(reader)
        except _Declined:
            continue
        parent = select.parent
        if parent is None:
            tree = _union(branches)
        elif type(parent) is exp.Subquery or (type(parent) is exp.Union and parent.this is select):
            select.replace(_union(branches))
        elif type(parent) is exp.Union and parent.expression is select and not any(parent.args.get(key) for key in ("by_name", "side", "kind", "on")):
            # The right side of a set operation: written without parentheses, ``A UNION b1 UNION ALL b2`` would be
            # read ``(A UNION b1) UNION ALL b2``. Under a UNION that removes duplicates the branches are glued the
            # same way, and under UNION ALL with UNION ALL: both mean the same thing however they are grouped.
            select.replace(_union(branches, distinct=bool(parent.args.get("distinct"))))
        else:
            continue
        changed = True
    return tree, changed
