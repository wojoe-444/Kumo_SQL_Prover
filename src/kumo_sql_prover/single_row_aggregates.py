"""``MIN``, ``SUM`` and ``MAX`` of a value that at most one row of a group can have are one aggregate.

A table with one row per item and shop is often turned on its side, one column per shop:

    SELECT item, SUM(CASE WHEN shop = 'north' THEN cost END) AS north, .. FROM prices GROUP BY item
    SELECT item, MAX(CASE WHEN shop = 'north' THEN cost END) AS north, .. FROM prices GROUP BY item
    SELECT item, MIN(IF(shop = 'north', cost, NULL)) AS north, .. FROM prices GROUP BY item

With ``(item, shop)`` a declared key the three say the same: a group holds the rows of one item, at most one of
them is the row of the shop ``north``, and the ``CASE`` is NULL on every other row. An aggregate over at most one
value that is not NULL is that value (or NULL when there is none), whichever of the three it is. This module
writes ``MIN`` and ``SUM`` of such an argument as ``MAX``, and the argument as ``CASE WHEN .. THEN .. END``.

**When at most one row passes.** The select reads one plain table and nothing else. The argument is a ``CASE`` with
one ``WHEN`` and no ``ELSE`` (or ``ELSE NULL``), or the ``IF`` spelling. Its condition is a conjunction; some of its
conjuncts compare a column of the table with a literal of the column's own kind (a text with a text column, a
whole number with a whole-number column: a comparison across kinds converts, and then two different stored values
can both pass). The columns compared that way, together with the plain columns of the ``GROUP BY``, cover a
declared key of the table whose columns are all NOT NULL. Two rows of one group that pass agree on the grouped
columns and on the compared ones, so on the whole key: they are one row. A key column that may be NULL is not
read, because rows with a NULL there are not kept apart by the key. Grouping expressions that are not plain
columns only make the groups smaller, and a ``WHERE`` only removes rows.

**``SUM`` is read only for exact numbers.** ``MIN`` and ``MAX`` return the value they are given. ``SUM`` adds, so
its one value has to be a number already: a column declared as a whole number or a decimal, a whole-number literal,
or sums, differences and products of those. A text would be converted (``SUM('7 apples')`` is 7 in MySQL), a
truth value is refused by some engines, and a floating-point column is left alone. And only in the dialects where
the sum of whole numbers is used like a whole number afterwards (``SUM_KEEPS_KIND``): PostgreSQL's ``SUM`` of a
BIGINT is a NUMERIC, so ``SUM(x) / 2`` keeps the half that ``MAX(x) / 2`` drops. MySQL's sum is a DECIMAL with no
decimals, which divides as its INT does; SQLite and BigQuery keep the whole number; DuckDB's ``/`` never truncates.

**Left alone.** A select over a join or a derived table, ``ROLLUP`` and the like, an aggregate with ``DISTINCT`` on
more than one value, a window, a ``FILTER``; a ``CASE`` with two ``WHEN`` or an ``ELSE`` that is not NULL; a
``GROUP BY`` name that is also the alias of another select item (the engines do not agree on which of the two is
meant).

A pre-pass of the last proof step (``shape_fallback.TWENTY_SEVENTH_BATCH``), after ``lookup_pivots``, which writes
its lookups as the ``MAX`` this module writes.
"""

from __future__ import annotations

import re

from sqlglot import exp

from . import smt_values
from .ast_utils import FROM_KEY, visible_ctes
from .unmatched_outer_join import _conjuncts, _unwrap

NAME = "single_row_aggregates"

_TEXT_TYPES = frozenset({"STRING", "VARCHAR", "TEXT", "CHAR", "NVARCHAR", "NCHAR", "ENUM"})
_EXACT = (smt_values.INT64, smt_values.NUMERIC, smt_values.BIGNUMERIC)
SUM_KEEPS_KIND = frozenset({"mysql", "sqlite", "duckdb", "bigquery"})


def _lowered(mapping: dict | None) -> dict:
    return {str(name).lower(): value for name, value in (mapping or {}).items()}


def _set_args(node: exp.Expression) -> set[str]:
    return {key for key, value in node.args.items() if value not in (None, [], False)}


def literal_kind(node: exp.Expression) -> str | None:
    """``"text"`` for a string literal, ``"int"`` for a whole-number literal (with or without a sign)."""

    node = _unwrap(node)
    if isinstance(node, exp.Neg):
        node = _unwrap(node.this)
        return "int" if isinstance(node, exp.Literal) and not node.is_string and smt_values.is_integer_text(node.this) else None
    if not isinstance(node, exp.Literal):
        return None
    if node.is_string:
        return "text"
    return "int" if smt_values.is_integer_text(node.this) else None


def column_kind(declared: object) -> str | None:
    """``"text"`` or ``"int"`` for a declared column type a literal of that kind is compared with as it is stored."""

    if not declared:
        return None
    if re.sub(r"[(<].*", "", str(declared)).strip().upper() in _TEXT_TYPES:
        return "text"
    return "int" if smt_values.type_class(str(declared)) == smt_values.INT64 else None


class Keyed:
    """One read of a plain declared table: its name in the query, its columns, types and the keys that are NOT NULL."""

    def __init__(self, table: exp.Table, name: str, columns: list, types: dict, never_null: frozenset, keys: list) -> None:
        self.table = table
        self.name = name  # the declared table, lower case
        self.alias = (table.alias or table.name).lower()  # what the query calls this read
        self.columns = {str(c).lower() for c in columns}
        self.types = {str(c).lower(): str(v) for c, v in types.items() if v}
        self.never_null = never_null
        self.keys = keys

    @classmethod
    def read(cls, table: exp.Expression | None, schema: dict | None, keys: dict | None, not_null: dict | None, types: dict | None) -> "Keyed | None":
        if type(table) is not exp.Table or not isinstance(table.this, exp.Identifier) or _set_args(table) - {"this", "alias"}:
            return None
        alias = table.args.get("alias")
        if alias is not None and (alias.args.get("columns") or not isinstance(alias.this, exp.Identifier)):
            return None
        name = table.name.lower()
        declared = _lowered(schema)
        if name not in declared or name in visible_ctes(table):
            return None  # a WITH name hides the table: its rows and keys are not the declared ones
        columns = {str(c).lower() for c in declared[name]}
        never_null = frozenset(str(c).lower() for c in (_lowered(not_null).get(name) or ()))
        own = [tuple(str(c).lower() for c in key) for key in (_lowered(keys).get(name) or [])]
        own = [key for key in own if key and set(key) <= never_null and set(key) <= columns]
        return cls(table, name, list(declared[name]), _lowered(_lowered(types).get(name)), never_null, own)

    def column(self, node: exp.Expression | None, qualified: bool = False) -> str | None:
        """The table's column ``node`` is, read from a select whose only source is this table (or, with
        ``qualified``, only when it is written with this read's name)."""

        node = _unwrap(node) if node is not None else None
        if not isinstance(node, exp.Column) or node.args.get("db") or node.args.get("catalog") or not isinstance(node.this, exp.Identifier):
            return None
        if node.table and node.table.lower() != self.alias:
            return None
        if qualified and not node.table:
            return None
        name = node.name.lower()
        return name if name in self.columns else None

    def pinned(self, condition: exp.Expression | None, qualified: bool = False) -> dict[str, exp.Expression] | None:
        """The columns a conjunct of ``condition`` compares with a literal of their own kind, with that literal.
        ``None`` when one column is compared with two literals."""

        found: dict[str, exp.Expression] = {}
        for part in _conjuncts(condition):
            if not isinstance(part, exp.EQ):
                continue
            for left, right in ((part.this, part.expression), (part.expression, part.this)):
                name = self.column(left, qualified)
                kind = literal_kind(right)
                if name is None or kind is None or column_kind(self.types.get(name)) != kind:
                    continue
                literal = _unwrap(right)
                if name in found and found[name] != literal:
                    return None
                found[name] = literal
        return found

    def one_row(self, fixed: set[str]) -> bool:
        """Whether rows that agree on the columns ``fixed`` are one row: a NOT NULL key within them."""

        return any(set(key) <= fixed for key in self.keys)

    def exact_number(self, node: exp.Expression) -> bool:
        """Whether ``node`` is an exact number as it stands: ``SUM`` of the one value is the value."""

        node = _unwrap(node)
        if isinstance(node, exp.Column):
            name = self.column(node)
            return name is not None and smt_values.type_class(self.types.get(name)) in _EXACT
        if isinstance(node, exp.Literal):
            return literal_kind(node) == "int"
        if isinstance(node, exp.Neg):
            return self.exact_number(node.this)
        if type(node) in (exp.Add, exp.Sub, exp.Mul):
            return self.exact_number(node.this) and self.exact_number(node.expression)
        return False


def guarded(argument: exp.Expression) -> tuple[exp.Expression, exp.Expression] | None:
    """``(condition, value)`` of an argument that is ``value`` on the rows that pass ``condition`` and NULL elsewhere."""

    argument = _unwrap(argument)
    if isinstance(argument, exp.Case):
        ifs = argument.args.get("ifs") or []
        default = argument.args.get("default")
        if len(ifs) != 1 or not (default is None or isinstance(_unwrap(default), exp.Null)):
            return None
        condition, value = ifs[0].this, ifs[0].args.get("true")
        if argument.this is not None:
            condition = exp.EQ(this=argument.this.copy(), expression=condition.copy())
        return (condition, value) if value is not None else None
    if type(argument) is exp.If:
        other = argument.args.get("false")
        if other is not None and not isinstance(_unwrap(other), exp.Null):
            return None
        value = argument.args.get("true")
        return (argument.this, value) if value is not None else None
    return None


def grouped_columns(select: exp.Select, table: Keyed) -> set[str] | None:
    """The plain columns of ``table`` the select groups by; ``None`` for a grouping this module does not read."""

    group = select.args.get("group")
    if group is None:
        return set()
    if _set_args(group) - {"expressions"}:
        return None  # ROLLUP, CUBE, GROUPING SETS
    aliases = {item.alias.lower(): item.this for item in select.expressions if isinstance(item, exp.Alias) and item.alias}
    found: set[str] = set()
    for term in group.expressions:
        term = _unwrap(term)
        if isinstance(term, exp.Literal) and not term.is_string and smt_values.is_integer_text(term.this):
            position = int(term.this)
            if not 1 <= position <= len(select.expressions):
                return None
            term = _unwrap(select.expressions[position - 1].unalias())
        elif isinstance(term, exp.Column) and not term.table and term.name.lower() in aliases:
            if table.column(aliases[term.name.lower()]) != term.name.lower():
                return None  # a name that is a column and another item's alias: the engines differ on which is meant
        name = table.column(term)
        if name is not None:
            found.add(name)
    return found


def read_single_row_aggregates(
    tree: exp.Expression, schema: dict | None, keys: dict | None, not_null: dict | None, types: dict | None, dialect: str = "bigquery"
) -> tuple[exp.Expression, bool]:
    """``tree`` with ``MIN`` and ``SUM`` written as ``MAX`` where at most one row of a group passes the argument's
    condition; whether it changed. ``SUM`` is read only in the dialects of ``SUM_KEEPS_KIND``."""

    if not schema or not keys or not any(isinstance(node, (exp.Min, exp.Sum, exp.Max)) for node in tree.walk()):
        return tree, False
    result = tree.copy()
    wrote = False
    sums = str(dialect or "").lower() in SUM_KEEPS_KIND
    for select in list(result.find_all(exp.Select)):
        wrote = _rewrite(select, schema, keys, not_null, types, sums) or wrote
    return (result, True) if wrote else (tree, False)


def _rewrite(select: exp.Select, schema: dict, keys: dict, not_null: dict | None, types: dict | None, sums: bool) -> bool:
    source = select.args.get(FROM_KEY)
    if source is None or select.args.get("joins") or select.args.get("laterals"):
        return False
    table = Keyed.read(source.this, schema, keys, not_null, types)
    if table is None or not table.keys:
        return False
    grouped = grouped_columns(select, table)
    if grouped is None:
        return False
    wrote = False
    for call in list(select.find_all(exp.Min, exp.Sum, exp.Max)):
        if call.find_ancestor(exp.Select) is not select or isinstance(call.parent, (exp.Window, exp.Filter)) or _set_args(call) - {"this"}:
            continue
        argument = call.this
        if isinstance(argument, exp.Distinct):
            if len(argument.expressions) != 1 or argument.args.get("on"):
                continue
            argument = argument.expressions[0]
        reading = guarded(argument)
        if reading is None:
            continue
        condition, value = reading
        pinned = table.pinned(condition)
        if pinned is None or not table.one_row(grouped | set(pinned)):
            continue
        if isinstance(call, exp.Sum) and not (sums and table.exact_number(value)):
            continue
        new = exp.Max(this=exp.Case(ifs=[exp.If(this=condition.copy(), true=value.copy())]))  # one spelling of the argument
        if new != call:
            call.replace(new)
            wrote = True
    return wrote
