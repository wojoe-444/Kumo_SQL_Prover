"""A table turned on its side with one lookup per column is the grouped select with one ``MAX`` per lookup.

``prices`` has one row per item and shop, ``(item, shop)`` is its key, and the answer wanted is one row per item
with a column per shop. Besides the grouped spelling

    SELECT item, MAX(CASE WHEN shop = 'north' THEN cost END) AS north, .. FROM prices GROUP BY item

it is written with a lookup per shop, from the list of items or from the table itself:

    SELECT i.item, n.cost AS north, ..
    FROM (SELECT DISTINCT item FROM prices) AS i LEFT JOIN prices AS n ON n.item = i.item AND n.shop = 'north' ..

    SELECT DISTINCT p.item, n.north, ..
    FROM prices AS p LEFT JOIN (SELECT item, cost AS north FROM prices WHERE shop = 'north') AS n ON n.item = p.item ..

    SELECT DISTINCT item, (SELECT cost FROM prices AS n WHERE n.shop = 'north' AND n.item = p.item) AS north, ..
    FROM prices AS p

This module writes the lookup spellings as the grouped one.

**The shape read.** A select with no ``WHERE``, ``HAVING``, ``ORDER BY``, ``LIMIT`` or window, whose sources are

* a *driver*: ``(SELECT DISTINCT g.. FROM t)`` (or the same list with ``GROUP BY``), or the table ``t`` itself;
* *lookups* of the same table, each a ``LEFT JOIN`` of ``t`` or of ``(SELECT .. FROM t WHERE ..)``, or a scalar
  subquery in the select list. A lookup's conditions are equalities and nothing else: each driver column ``g`` with
  the lookup's own ``g``, and columns of the lookup with a literal of the column's own kind (a text with a text
  column, a whole number with a whole-number column).

Every lookup equates all of the driver's columns ``g`` (for a driver that is the table: every lookup the same
columns), those columns are NOT NULL, and for each lookup the columns ``g`` with the columns it compares with a
literal cover a declared key of ``t`` whose columns are all NOT NULL.

**Why it holds.** The rows of ``t`` with one value of ``g`` are a group of the grouped spelling, and there is one
driver row per group (the ``DISTINCT`` list), or one per row of the group (the table itself). ``g`` is NOT NULL,
so a driver row finds exactly the rows of its own group. Among them at most one has the lookup's literals, by the
key. The lookup's column is the value on that row, and NULL when there is none or the value is NULL. So is
``MAX(CASE WHEN <the literals> THEN value END)`` over the group: the ``CASE`` is NULL on every other row. A lookup
never adds or drops a driver row, so a ``DISTINCT`` list as driver gives one row per group as it is. With the table
as driver every row of a group gives the same output row, so the select has to remove the copies: ``DISTINCT``, or
a ``GROUP BY`` that names every column ``g`` (what it names besides is a function of ``g``). A ``DISTINCT`` is kept
when the select list does not show every column ``g``, since two groups can then give the same row.

**What the select list may hold.** The driver's columns ``g``; columns of the lookups other than their compared
ones; scalar lookups; and literals, ``COALESCE``, ``CASE``, ``IF``, comparisons, ``AND``, ``OR``, ``NOT``,
``IS NULL``, sums, differences and products of those. Under a ``GROUP BY``, ``MAX`` and ``MIN`` of such a value are
the value (it is the same on every row of the group). Each item keeps its name.

**Left alone.** An inner join to a lookup (it drops items); a lookup condition that is not one of the equalities
above; a lookup of another table or through another lookup; a column of the driver table that is not one of ``g``;
a lookup's own compared column in the select list; a column name two sources could mean; a value column whose
declared type is not a number, text, date or time (``MAX`` may not take it) or is a single-precision ``FLOAT``
(MySQL's ``MAX`` of it is a DOUBLE, which shows the digits the FLOAT column rounds away: 0.1 becomes
0.10000000149011612 as a text); a table name a ``WITH`` hides.

A pre-pass of the last proof step (``shape_fallback.TWENTY_SEVENTH_BATCH``), before ``single_row_aggregates``.
"""

from __future__ import annotations

import re

from sqlglot import exp

from . import smt_values
from .ast_utils import FROM_KEY
from .keyed_join_groups import _and_all
from .single_row_aggregates import Keyed, _set_args, column_kind, grouped_columns, literal_kind
from .unmatched_outer_join import _conjuncts, _unwrap

NAME = "lookup_pivots"

_VALUE_NODES = (
    exp.Literal, exp.Null, exp.Boolean, exp.Paren, exp.Neg, exp.Add, exp.Sub, exp.Mul, exp.Coalesce, exp.Case, exp.If,
    exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.And, exp.Or, exp.Not, exp.Is,
)
_ORDERED_CLASSES = (smt_values.INT64, smt_values.NUMERIC, smt_values.BIGNUMERIC, smt_values.FLOAT64, smt_values.STRING, smt_values.BOOL)
_ORDERED_NAMES = frozenset({"DATE", "DATETIME", "TIME", "TIMESTAMP"})
_SINGLE_PRECISION = frozenset({"FLOAT", "FLOAT4", "REAL"})


class _Unread(Exception):
    pass


class _Driver:
    def __init__(self, alias: str, table: Keyed, outputs: dict[str, str], grouped: frozenset | None) -> None:
        self.alias = alias
        self.table = table  # the read of the table the driver lists (or is)
        self.outputs = outputs  # output name -> column of the table
        self.grouped = grouped  # the columns a DISTINCT list holds; None for the table itself


class _Lookup:
    def __init__(self, alias: str | None, table: Keyed, outputs: dict[str, exp.Expression]) -> None:
        self.alias = alias  # None for a scalar subquery
        self.table = table
        self.outputs = outputs  # output name -> expression over the table's own columns
        self.pins: dict[str, exp.Expression] = {}  # column -> the literal it is compared with
        self.matched: set[str] = set()  # the columns equated with the driver's


def _ordered(declared: object) -> bool:
    if not declared or re.sub(r"[(<].*", "", str(declared)).strip().upper() in _SINGLE_PRECISION:
        return False  # MySQL shows a FLOAT column as 0.1 and MAX of it, a DOUBLE, as 0.10000000149011612
    return (
        column_kind(declared) is not None
        or smt_values.type_class(str(declared)) in _ORDERED_CLASSES
        or re.sub(r"[(<].*", "", str(declared)).strip().upper() in _ORDERED_NAMES
    )


def _own_value(node: exp.Expression, table: Keyed, qualified: bool = False) -> bool:
    """Whether ``node`` is a value of one row of ``table``: its columns, literals and the operators this module reads."""

    for child in node.walk():
        if isinstance(child, exp.Column):
            name = table.column(child, qualified)
            if name is None or not _ordered(table.types.get(name)):
                return False
        elif not isinstance(child, (*_VALUE_NODES, exp.Identifier)):
            return False
    return True


def _pin(table: Keyed, part: exp.Expression, qualified: bool) -> tuple[str, exp.Expression] | None:
    if not isinstance(part, exp.EQ):
        return None
    for left, right in ((part.this, part.expression), (part.expression, part.this)):
        name = table.column(left, qualified)
        kind = literal_kind(right)
        if name is not None and kind is not None and column_kind(table.types.get(name)) == kind:
            return name, _unwrap(right)
    return None


def _add_pin(lookup: _Lookup, pin: tuple[str, exp.Expression]) -> bool:
    name, literal = pin
    if name in lookup.pins and lookup.pins[name] != literal:
        return False
    lookup.pins[name] = literal
    return True


def _driver_column(driver: _Driver, node: exp.Expression) -> str | None:
    """The table column a column written with the driver's name is."""

    node = _unwrap(node)
    if not isinstance(node, exp.Column) or node.args.get("db") or node.args.get("catalog") or not node.table:
        return None
    if node.table.lower() != driver.alias:
        return None
    return driver.outputs.get(node.name.lower())


def _driver(source: exp.Expression, schema: dict, keys: dict, not_null: dict | None, types: dict | None) -> _Driver | None:
    table = Keyed.read(source, schema, keys, not_null, types)
    if table is not None:
        return _Driver(table.alias, table, {name: name for name in table.columns}, None)
    if not isinstance(source, exp.Subquery) or _set_args(source) - {"this", "alias"} or not isinstance(source.this, exp.Select):
        return None
    alias = source.args.get("alias")
    if alias is None or alias.args.get("columns") or not isinstance(alias.this, exp.Identifier):
        return None
    inner = source.this
    listed = _set_args(inner)
    if listed - {"expressions", FROM_KEY, "distinct", "group"} or FROM_KEY not in listed or not listed & {"distinct", "group"}:
        return None
    if inner.args.get("distinct") is not None and inner.args["distinct"].args.get("on"):
        return None
    table = Keyed.read(inner.args[FROM_KEY].this, schema, keys, not_null, types)
    if table is None:
        return None
    outputs: dict[str, str] = {}
    for item in inner.expressions:
        name = table.column(item.unalias())
        if name is None:
            return None
        shown = (item.alias if isinstance(item, exp.Alias) else name).lower()
        if not shown or shown in outputs:
            return None
        outputs[shown] = name
    columns = frozenset(outputs.values())
    if len(columns) != len(outputs):
        return None
    if inner.args.get("group") is not None:
        group = inner.args["group"]
        if grouped_columns(inner, table) != columns or len(group.expressions) != len(columns):
            return None
    return _Driver(alias.name.lower(), table, outputs, columns)


def _joined(join: exp.Join, driver: _Driver, schema: dict, keys: dict, not_null: dict | None, types: dict | None) -> _Lookup | None:
    if _set_args(join) - {"this", "side", "kind", "on"} or (join.side or "").upper() != "LEFT" or (join.kind or "").upper() not in ("", "OUTER"):
        return None
    source, on = join.this, join.args.get("on")
    if on is None:
        return None
    table = Keyed.read(source, schema, keys, not_null, types)
    if table is not None:
        lookup = _Lookup(table.alias, table, {name: exp.column(name) for name in table.columns})
        inner_pins: list = []
    else:
        if not isinstance(source, exp.Subquery) or _set_args(source) - {"this", "alias"} or not isinstance(source.this, exp.Select):
            return None
        alias = source.args.get("alias")
        if alias is None or alias.args.get("columns") or not isinstance(alias.this, exp.Identifier):
            return None
        inner = source.this
        if _set_args(inner) - {"expressions", FROM_KEY, "where"} or FROM_KEY not in _set_args(inner):
            return None
        table = Keyed.read(inner.args[FROM_KEY].this, schema, keys, not_null, types)
        if table is None:
            return None
        outputs: dict[str, exp.Expression] = {}
        for item in inner.expressions:
            if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
                if isinstance(item, exp.Column) and (item.table or "").lower() != table.alias:
                    return None
                if _set_args(item) - {"this", "table"}:
                    return None  # EXCEPT, REPLACE
                shown = {name: exp.column(name) for name in sorted(table.columns)}
            else:
                value = item.unalias()
                name = (item.alias if isinstance(item, exp.Alias) else (_unwrap(value).name if isinstance(_unwrap(value), exp.Column) else "")).lower()
                if not name or not _own_value(value, table):
                    return None
                shown = {name: value}
            if set(shown) & set(outputs):
                return None
            outputs.update(shown)
        lookup = _Lookup(alias.name.lower(), table, outputs)
        inner_pins = _conjuncts(inner.args["where"].this) if inner.args.get("where") is not None else []
    if table.name != driver.table.name or lookup.alias == driver.alias:
        return None
    for part in inner_pins:
        pin = _pin(table, part, qualified=False)
        if pin is None or not _add_pin(lookup, pin):
            return None
    outer = _OuterNames(lookup)
    for part in _conjuncts(on):
        if not isinstance(part, exp.EQ):
            return None
        pin = outer.pin(part)
        if pin is not None:
            if not _add_pin(lookup, pin):
                return None
            continue
        for left, right in ((part.this, part.expression), (part.expression, part.this)):
            name = outer.column(left)
            if name is not None and _driver_column(driver, right) == name:
                lookup.matched.add(name)
                break
        else:
            return None
    return lookup


class _OuterNames:
    """A joined lookup as the select around it names it: ``alias.output``."""

    def __init__(self, lookup: _Lookup) -> None:
        self.lookup = lookup

    def column(self, node: exp.Expression) -> str | None:
        """The table column the lookup's output ``node`` shows as it is stored."""

        node = _unwrap(node)
        if not isinstance(node, exp.Column) or node.args.get("db") or node.args.get("catalog") or not node.table:
            return None
        if node.table.lower() != self.lookup.alias:
            return None
        value = self.lookup.outputs.get(node.name.lower())
        return self.lookup.table.column(value) if value is not None else None

    def pin(self, part: exp.EQ) -> tuple[str, exp.Expression] | None:
        for left, right in ((part.this, part.expression), (part.expression, part.this)):
            name = self.column(left)
            kind = literal_kind(right)
            if name is not None and kind is not None and column_kind(self.lookup.table.types.get(name)) == kind:
                return name, _unwrap(right)
        return None


def _scalar(node: exp.Subquery, driver: _Driver, schema: dict, keys: dict, not_null: dict | None, types: dict | None) -> _Lookup | None:
    inner = node.this
    if _set_args(node) - {"this"} or not isinstance(inner, exp.Select):
        return None
    if _set_args(inner) - {"expressions", FROM_KEY, "where"} or FROM_KEY not in _set_args(inner) or len(inner.expressions) != 1:
        return None
    table = Keyed.read(inner.args[FROM_KEY].this, schema, keys, not_null, types)
    if table is None or table.name != driver.table.name or table.alias == driver.alias:
        return None
    value = inner.expressions[0].unalias()
    if not _own_value(value, table):
        return None
    lookup = _Lookup(None, table, {"": value})
    for part in _conjuncts(inner.args["where"].this) if inner.args.get("where") is not None else []:
        if not isinstance(part, exp.EQ):
            return None
        pin = _pin(table, part, qualified=False)
        if pin is not None:
            if not _add_pin(lookup, pin):
                return None
            continue
        for left, right in ((part.this, part.expression), (part.expression, part.this)):
            name = table.column(left)
            if name is not None and _driver_column(driver, right) == name:
                lookup.matched.add(name)
                break
        else:
            return None
    return lookup


def read_lookup_pivots(tree: exp.Expression, schema: dict | None, keys: dict | None, not_null: dict | None, types: dict | None) -> tuple[exp.Expression, bool]:
    """``tree`` with every select of the shape above written as the grouped select; whether it changed."""

    if not schema or not keys or not types:
        return tree, False
    if not any(select.args.get("joins") or any(isinstance(n, exp.Subquery) for item in select.expressions for n in item.walk()) for select in tree.find_all(exp.Select)):
        return tree, False
    result = tree.copy()
    wrote = False
    for select in list(result.find_all(exp.Select)):
        if select.root() is not result:
            continue  # part of a select this pass has replaced
        new = _read(select, schema, keys, not_null, types)
        if new is None:
            continue
        if select is result:
            result = new
        else:
            select.replace(new)
        wrote = True
    return (result, True) if wrote else (tree, False)


def _read(select: exp.Select, schema: dict, keys: dict, not_null: dict | None, types: dict | None) -> exp.Select | None:
    if _set_args(select) - {"expressions", FROM_KEY, "joins", "distinct", "group"} or FROM_KEY not in _set_args(select):
        return None
    distinct, group = select.args.get("distinct"), select.args.get("group")
    if distinct is not None and distinct.args.get("on"):
        return None
    if group is not None and _set_args(group) - {"expressions"}:
        return None
    driver = _driver(select.args[FROM_KEY].this, schema, keys, not_null, types)
    if driver is None or not driver.table.keys:
        return None
    lookups: dict[str, _Lookup] = {}
    for join in select.args.get("joins") or []:
        lookup = _joined(join, driver, schema, keys, not_null, types)
        if lookup is None or lookup.alias in lookups:
            return None
        lookups[lookup.alias] = lookup
    scalars: dict[int, _Lookup] = {}
    for item in select.expressions:
        for node in item.find_all(exp.Subquery):
            if node.find_ancestor(exp.Select) is not select:
                continue
            lookup = _scalar(node, driver, schema, keys, not_null, types)
            if lookup is None:
                return None
            scalars[id(node)] = lookup
    every = [*lookups.values(), *scalars.values()]
    if not every:
        return None
    grouped = driver.grouped if driver.grouped is not None else frozenset(every[0].matched)
    table = driver.table
    if not grouped or not grouped <= table.never_null:
        return None
    for lookup in every:
        if frozenset(lookup.matched) != grouped or not lookup.table.one_row(set(grouped) | set(lookup.pins)):
            return None
    alias = driver.alias

    def found(lookup: _Lookup, value: exp.Expression) -> exp.Expression:
        own = lookup.table.column(value)
        if own is not None and (own in lookup.pins or own in lookup.matched):
            raise _Unread  # the lookup's own compared column
        test = _and_all([exp.EQ(this=exp.column(name, table=alias), expression=lookup.pins[name].copy()) for name in sorted(lookup.pins)]) if lookup.pins else None
        shown = _unwrap(value).copy()
        for column in list(shown.find_all(exp.Column)):
            if not _ordered(lookup.table.types.get(column.name.lower())):
                raise _Unread  # MAX may not take a value of this type
            if column is not shown:
                column.replace(exp.column(column.name.lower(), table=alias))
        if isinstance(shown, exp.Column):
            shown = exp.column(shown.name.lower(), table=alias)
        if test is None:
            return exp.Max(this=shown)  # the columns g alone are a key: the group is one row
        return exp.Max(this=exp.Case(ifs=[exp.If(this=test, true=shown)]))

    def resolve(column: exp.Column) -> exp.Expression:
        if column.args.get("db") or column.args.get("catalog") or not isinstance(column.this, exp.Identifier):
            raise _Unread
        name, owner = column.name.lower(), (column.table or "").lower()
        if not owner:
            owners = [key for key, lookup in lookups.items() if name in lookup.outputs] + ([alias] if name in driver.outputs else [])
            if len(owners) != 1:
                raise _Unread
            owner = owners[0]
        if owner == alias:
            stored = driver.outputs.get(name)
            if stored is None or stored not in grouped:
                raise _Unread
            return exp.column(stored, table=alias)
        lookup = lookups.get(owner)
        if lookup is None or name not in lookup.outputs:
            raise _Unread
        return found(lookup, lookup.outputs[name])

    def rebuild(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Subquery):
            lookup = scalars.get(id(node))
            if lookup is None:
                raise _Unread
            return found(lookup, lookup.outputs[""])
        if isinstance(node, exp.Column):
            return resolve(node)
        if type(node) in (exp.Max, exp.Min) and group is not None and not (_set_args(node) - {"this"}) and not isinstance(node.this, exp.Distinct):
            return rebuild(node.this)  # the same value on every row of the group
        if not isinstance(node, _VALUE_NODES):
            raise _Unread
        new = type(node)()
        for key, value in node.args.items():
            if isinstance(value, list):
                new.set(key, [rebuild(v) if isinstance(v, exp.Expression) else v for v in value])
            elif isinstance(value, exp.Expression):
                new.set(key, rebuild(value))
            else:
                new.set(key, value)
        return new

    try:
        items: list[exp.Expression] = []
        shown_plain: set[str] = set()
        for item in select.expressions:
            value = item.unalias()
            new = rebuild(value)
            if isinstance(item, exp.Alias):
                name = item.args["alias"].copy()
            elif isinstance(_unwrap(value), exp.Column):
                name = exp.to_identifier(_unwrap(value).name)
            else:
                raise _Unread  # an item without a name of its own
            if isinstance(new, exp.Column) and new.table == alias and new.name in grouped:
                shown_plain.add(new.name)
            items.append(new if isinstance(new, exp.Column) and new.name.lower() == name.name.lower() else exp.Alias(this=new, alias=name))
        if group is not None:
            named: set[str] = set()
            for term in group.expressions:
                term = _unwrap(term)
                if isinstance(term, exp.Literal) and not term.is_string and smt_values.is_integer_text(term.this):
                    position = int(term.this)
                    if not 1 <= position <= len(select.expressions):
                        raise _Unread
                    term = _unwrap(select.expressions[position - 1].unalias())
                if not isinstance(term, exp.Column):
                    rebuild(term)  # a function of the columns g, or not read
                    continue
                new = resolve(term)
                if isinstance(new, exp.Column):
                    named.add(new.name)
            if not grouped <= named:
                raise _Unread
        elif driver.grouped is None and distinct is None:
            raise _Unread  # the table as driver: a row per row of the group
    except _Unread:
        return None
    source = table.table.copy()
    source.set("alias", exp.TableAlias(this=exp.to_identifier(alias)))
    new_select = exp.Select(expressions=items)
    new_select.set(FROM_KEY, exp.From(this=source))
    new_select.set("group", exp.Group(expressions=[exp.column(name, table=alias) for name in sorted(grouped)]))
    if distinct is not None and not grouped <= shown_plain:
        new_select.set("distinct", exp.Distinct())
    return new_select
