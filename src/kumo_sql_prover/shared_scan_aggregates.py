"""Aggregates of one table that a one-row select asks in several places, read in one pass over the table.

"What share of the visits came on the day they were booked" is written with the two counts side by
side, or with one of them asked apart:

    SELECT ROUND(SUM(booked = came) / COUNT(*) * 100, 2) FROM visits

    SELECT ROUND(SUM(IF(booked = came, 1, 0)) / (SELECT COUNT(*) FROM visits) * 100, 2) FROM visits
    SELECT ROUND(COUNT(*) / (SELECT COUNT(*) FROM visits) * 100, 2)
    FROM (SELECT * FROM visits WHERE booked = came) AS i
    SELECT ROUND(x.n / COUNT(v.visit) * 100, 2)
    FROM visits AS v, (SELECT COUNT(visit) AS n FROM visits WHERE booked = came) AS x

Each of the last three is read as one select over the table with no ``WHERE``, every aggregate keeping its own
filter inside its argument:

    SELECT ROUND(SUM(IF(booked = came, 1, 0)) / COUNT(*) * 100, 2) FROM visits
    SELECT ROUND(COUNT(CASE WHEN i.booked = i.came THEN 1 END) / COUNT(*) * 100, 2) FROM visits AS i
    SELECT ROUND(CASE WHEN COUNT(*) > 0 THEN COUNT(CASE WHEN v.booked = v.came THEN v.visit END) END
                 / COUNT(v.visit) * 100, 2) FROM visits AS v

The prover compares a select of several scalar subqueries subquery by subquery (``aggregate_scalars``), which loses
what the terms have in common: ``SUM(flag)`` is NULL over no rows where ``COUNT(..)`` is 0, and the two shares are
equal there only because both divide by the same zero count. In one select the prover sees the counts together.

**Why it holds.** The select has an aggregate of its own, no ``GROUP BY`` and no ``HAVING``, so it returns exactly
one row. Its rows are the rows of the table ``T`` that pass its filter ``F`` (its ``WHERE``, and the ``WHERE`` of a
``SELECT * FROM T WHERE ..`` it reads in place of ``T``), each once: a joined one-row table multiplies them by one.

* An aggregate ``A(x)`` of the select is computed over the rows that pass ``F``. ``A(CASE WHEN F THEN x END)`` over
  every row of ``T`` reads the same ``x`` for those rows and NULL for the others, and ``COUNT``, ``SUM``, ``MIN``,
  ``MAX`` and ``AVG`` skip NULLs, with or without ``DISTINCT``. ``COUNT(*)`` counts the rows for which
  ``CASE WHEN F THEN 1 END`` is not NULL. Over no rows both give NULL, or 0 for a count.
* A scalar subquery ``(SELECT e FROM T WHERE V)`` that reads nothing of the row outside it, with ``e`` built from
  such aggregates, is the same ``e`` with each aggregate filtered by ``V`` in the same way: its value does not
  depend on the select's own filter, and the select now reads every row of ``T``.
* A column ``x.n`` of a joined one-row table, read outside every aggregate, is in MySQL and SQLite the value of
  some row of the select's input, and NULL when there is none (the two engines let an aggregate select read such
  a column; over no rows it is NULL). Every input row carries the same ``x.n``, so the value is the table's
  ``e`` when ``COUNT(*)`` of the input is positive and NULL otherwise. Other dialects refuse the query, and the rule
  leaves it alone there.

Nothing is volatile, so reading the table once gives the values that reading it once per subquery gives.

**No quotient is moved.** MySQL rounds a quotient of exact numbers, and an ``AVG``, to its declared scale when the
value is stored as an output of a derived table or a subquery, and keeps more digits while it is used where it is
computed (Kumo_SQL_Prover issue 45). So what is taken out of a subquery or a one-row table is built from ``COUNT``,
``SUM``, ``MIN`` and ``MAX`` with no division anywhere in it. The aggregates of the select itself stay where they
are and only gain the ``CASE`` inside their argument, which has the type of the argument.

**Left alone.** A select that is not the whole query, or has ``GROUP BY``, ``HAVING``, ``DISTINCT``, a window,
``ORDER BY``, ``LIMIT`` or a join with a condition other than ``TRUE``; one with no aggregate of its own (it has a row per row of the
table), or with nothing to fold; sources other than one declared table (or ``SELECT *`` from one with a filter)
and one-row aggregate tables over that same table, each a derived table or a ``WITH`` query read once; a ``WITH``
name that is the name of a declared table; a column of the table outside an aggregate; a column of a one-row table
inside an aggregate or in ``WHERE``; a subquery in ``WHERE``, inside an aggregate, under ``IN`` or ``EXISTS``, or
one that reads another table, joins, groups, or has more than one output; an aggregate with ``FILTER``,
``ORDER BY`` or several arguments when it would need a filter; any volatile call.

A pre-pass of the last proof step (``shape_fallback.TWENTY_FOURTH_BATCH``).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, is_aggregate
from .clause_alias_reads import _volatile
from .keyed_join_groups import declared_columns

NAME = "shared_scan_aggregates"

_EXACT = (exp.Count, exp.Sum, exp.Min, exp.Max)
_FILTERED = _EXACT + (exp.Avg,)
_BARE_COLUMN_DIALECTS = ("mysql", "sqlite")


def _lower(name: object) -> str:
    return str(name or "").lower()


def _set(node: exp.Expression) -> set[str]:
    return {key for key, value in node.args.items() if value}


def _plain_table(source: exp.Expression) -> bool:
    """A table named by one identifier, with at most an alias that renames no column."""

    if type(source) is not exp.Table or not isinstance(source.this, exp.Identifier) or _set(source) - {"this", "alias"}:
        return False
    alias = source.args.get("alias")
    return alias is None or (isinstance(alias, exp.TableAlias) and not _set(alias) - {"this"})


def _name(source: exp.Expression) -> exp.Identifier:
    """The identifier a source's columns are named with."""

    alias = source.args.get("alias")
    return alias.this if alias is not None else source.this


def _own_columns(node: exp.Expression, alias: str, columns: set[str]) -> bool:
    """Whether every column under ``node`` is a declared column of the one table, bare or named with ``alias``."""

    for column in node.find_all(exp.Column):
        if column.args.get("db") or column.args.get("catalog") or not isinstance(column.this, exp.Identifier):
            return False
        if _lower(column.name) not in columns or (column.table and _lower(column.table) != alias):
            return False
    return True


def _argument(call: exp.Expression) -> exp.Expression | None:
    """The one value an aggregate the rule can filter reads (the star of ``COUNT(*)`` included), or None."""

    if type(call) not in _FILTERED or _set(call) - {"this", "big_int"}:
        return None
    if isinstance(call.parent, (exp.Filter, exp.Window, exp.IgnoreNulls, exp.RespectNulls, exp.WithinGroup)):
        return None
    value = call.this
    if isinstance(value, exp.Distinct):
        if len(value.expressions) != 1 or _set(value) - {"expressions"}:
            return None
        value = value.expressions[0]
        if isinstance(value, exp.Star):
            return None
    elif isinstance(value, exp.Star) and (type(call) is not exp.Count or _set(value)):
        return None
    if value is None or isinstance(value, (exp.Order, exp.IgnoreNulls, exp.RespectNulls, exp.Tuple)):
        return None
    for node in call.walk():
        if node is not call and (is_aggregate(node) or isinstance(node, (exp.Query, exp.Subquery, exp.Window, exp.Order))):
            return None
        if isinstance(node, exp.Star) and node is not value:
            return None
    return value


def _guard(call: exp.Expression, test: exp.Expression | None) -> None:
    """``A(x)`` as ``A(CASE WHEN test THEN x END)``, in place; ``COUNT(*)`` counts the rows that pass."""

    if test is None:
        return
    value = call.this.expressions[0] if isinstance(call.this, exp.Distinct) else call.this
    kept = exp.Literal.number(1) if isinstance(value, exp.Star) else value.copy()
    value.replace(exp.Case(ifs=[exp.If(this=test.copy(), true=kept)]))


def _requalified(node: exp.Expression, name: exp.Identifier) -> exp.Expression:
    node = node.copy()
    for column in list(node.find_all(exp.Column)):
        column.set("table", name.copy())
    return node


class _Rows:
    """A one-row aggregate select over the table: its outputs by name, in order, and its filter."""

    def __init__(self, outputs: list[tuple[str, exp.Expression]], where: exp.Expression | None):
        self.outputs, self.where = outputs, where

    def named(self, name: str) -> exp.Expression | None:
        found = [value for own, value in self.outputs if own == name]
        return found[0] if len(found) == 1 else None

    def value(self, output: exp.Expression, scan: exp.Identifier) -> exp.Expression:
        """An output over every row of the table: its aggregates filtered, its columns named with the scan."""

        value = exp.Paren(this=_requalified(output, scan))
        test = _requalified(self.where, scan) if self.where is not None else None
        for call in [node for node in value.walk() if is_aggregate(node)]:
            _guard(call, test)
        return value


def _filter(select: exp.Select) -> tuple[bool, exp.Expression | None]:
    """Whether a ``WHERE`` reads one row and nothing else, and its condition."""

    where = select.args.get("where")
    if where is None:
        return True, None
    if any(is_aggregate(node) or isinstance(node, (exp.Query, exp.Subquery, exp.Window)) for node in where.walk()):
        return False, None
    return True, where.this


def _one_row(body: exp.Expression, table: str, columns: set[str]) -> _Rows | None:
    """The select of a one-row table or scalar subquery the rule reads (module doc), or None."""

    if type(body) is not exp.Select or _set(body) - {"expressions", FROM_KEY, "where"} or body.args.get(FROM_KEY) is None:
        return None
    source = body.args[FROM_KEY].this
    if not _plain_table(source) or _lower(source.name) != table:
        return None
    if _volatile(body) or any(node is not body and isinstance(node, (exp.Query, exp.Subquery, exp.Window)) for node in body.walk()):
        return None
    fine, where = _filter(body)
    if not fine or not _own_columns(body, _lower(source.alias_or_name), columns):
        return None
    outputs = []
    for item in body.expressions:
        value = item.unalias()
        if isinstance(item, exp.Star) or item.is_star:
            return None
        calls = [node for node in value.walk() if is_aggregate(node)]
        if not calls or any(isinstance(node, (exp.Div, exp.IntDiv)) for node in value.walk()):
            return None  # a quotient is rounded where it is stored (module doc)
        if any(type(call) not in _EXACT or _argument(call) is None for call in calls):
            return None
        for column in value.find_all(exp.Column):
            if not any(is_aggregate(up) for up in _ancestors(column, value)):
                return None
        outputs.append((_lower(item.alias_or_name), value))
    return _Rows(outputs, where)


def _ancestors(node: exp.Expression, stop: exp.Expression):
    while node is not stop and node.parent is not None:
        node = node.parent
        yield node


def _starred(body: exp.Expression, declared: dict) -> tuple[exp.Table, exp.Expression | None] | None:
    """``SELECT * FROM T [WHERE ..]`` over a declared table: the table and the filter, or None."""

    if type(body) is not exp.Select or _set(body) - {"expressions", FROM_KEY, "where"} or body.args.get(FROM_KEY) is None:
        return None
    if len(body.expressions) != 1 or type(body.expressions[0]) is not exp.Star or _set(body.expressions[0]):
        return None
    source = body.args[FROM_KEY].this
    if not _plain_table(source) or _lower(source.name) not in declared or _volatile(body):
        return None
    fine, where = _filter(body)
    if not fine or (where is not None and not _own_columns(where, _lower(source.alias_or_name), declared[_lower(source.name)])):
        return None
    return source, where


def read_shared_scans(tree: exp.Expression, schema: dict | None, dialect: object = None) -> tuple[exp.Expression, bool]:
    """``tree`` with its one-row aggregates over one table read in one pass (module doc); whether it changed."""

    if type(tree) is not exp.Select or not schema:
        return tree, False
    if _set(tree) - {"expressions", FROM_KEY, "where", "joins", "with_"} or tree.args.get(FROM_KEY) is None:
        return tree, False
    if not any(isinstance(node, (exp.Subquery, exp.CTE)) for node in tree.walk()):
        return tree, False
    result = _read(tree.copy(), declared_columns(schema), _lower(dialect))
    return (tree, False) if result is None else (result, True)


def _read(tree: exp.Select, declared: dict, dialect: str) -> exp.Select | None:
    # the names of the WITH clause: each has to be a one-row table read once as a source of this select
    named: dict[str, exp.Expression] = {}
    with_ = tree.args.get("with_")
    if with_ is not None:
        if _set(with_) - {"expressions"}:
            return None  # RECURSIVE
        for cte in with_.expressions:
            if type(cte) is not exp.CTE or _set(cte) - {"this", "alias"} or _set(cte.args["alias"]) - {"this"}:
                return None
            name = _lower(cte.alias)
            if not name or name in named or name in declared:
                return None  # a name that hides a declared table changes what every later read of it means
            named[name] = cte.this
    sources = [tree.args[FROM_KEY].this]
    for join in tree.args.get("joins") or []:
        if _set(join) - {"this", "kind", "on"} or _lower(join.args.get("kind")) not in ("", "cross", "inner"):
            return None
        on = join.args.get("on")
        if on is not None and not (isinstance(on, exp.Boolean) and on.this is True):
            return None  # ON TRUE is how some dialects' parsers write a join with no condition
        sources.append(join.this)
    scan: tuple[exp.Identifier, exp.Table, exp.Expression | None] | None = None
    bodies: dict[str, exp.Expression] = {}
    read: set[str] = set()
    for source in sources:
        alias = _lower(source.alias_or_name)
        if not alias or alias in bodies:
            return None
        if _plain_table(source) and _lower(source.name) in named:
            if _lower(source.name) in read:
                return None
            read.add(_lower(source.name))
            bodies[alias] = named[_lower(source.name)]
        elif _plain_table(source):
            if scan is not None or _lower(source.name) not in declared:
                return None
            scan = (_name(source), source, None)
        elif type(source) is exp.Subquery and not _set(source) - {"this", "alias"} and not _set(source.args["alias"]) - {"this"}:
            starred = _starred(source.this, declared)
            if starred is None:
                bodies[alias] = source.this
            elif scan is not None:
                return None
            else:
                table, where = starred
                scan = (_name(source), table, _requalified(where, _name(source)) if where is not None else None)
        else:
            return None
    if scan is None or set(named) - read:
        return None
    name, base, inner = scan
    table, here = _lower(base.name), _lower(name.name)
    columns = declared[table]
    if here in bodies:
        return None
    tables: dict[str, _Rows] = {}
    for alias, body in bodies.items():
        rows = _one_row(body, table, columns)
        if rows is None or any(not own for own, _ in rows.outputs) or len({own for own, _ in rows.outputs}) != len(rows.outputs):
            return None
        tables[alias] = rows
    if _volatile(tree):
        return None
    fine, outer = _filter(tree)
    if not fine or (outer is not None and not _own_columns(outer, here, columns)):
        return None
    if outer is not None and any(not column.table and any(rows.named(_lower(column.name)) is not None for rows in tables.values()) for column in outer.find_all(exp.Column)):
        return None  # a bare name that a one-row table has too

    def owner(column: exp.Column) -> str | None:
        """The source a column of the select list names: the scan's name, a one-row table's, or None."""

        if column.args.get("db") or column.args.get("catalog") or not isinstance(column.this, exp.Identifier):
            return None
        label, qualifier = _lower(column.name), _lower(column.table)
        if qualifier:
            if qualifier == here:
                return here if label in columns else None
            return qualifier if qualifier in tables and tables[qualifier].named(label) is not None else None
        owners = ([here] if label in columns else []) + [alias for alias, rows in tables.items() if rows.named(label) is not None]
        return owners[0] if len(owners) == 1 else None

    own: list[exp.Expression] = []
    subqueries: list[tuple[exp.Subquery, _Rows]] = []
    reads: list[tuple[exp.Column, str]] = []

    def visit(node: exp.Expression, inside: bool) -> bool:
        if isinstance(node, exp.Subquery):
            if inside or _set(node) - {"this"} or isinstance(node.parent, (exp.In, exp.Exists, exp.Any, exp.All)):
                return False
            rows = _one_row(node.this, table, columns)
            if rows is None or len(rows.outputs) != 1:
                return False
            subqueries.append((node, rows))
            return True
        if isinstance(node, (exp.Query, exp.Window, exp.Star)) and not (isinstance(node, exp.Star) and inside):
            return False
        if is_aggregate(node):
            if inside:
                return False
            own.append(node)
            return all(visit(child, True) for child in node.iter_expressions())
        if isinstance(node, exp.Column):
            source = owner(node)
            if source is None or (source == here) != inside:
                return False  # a table column outside an aggregate, or a one-row table's column inside one
            if source != here:
                reads.append((node, source))
            return True
        return all(visit(child, inside) for child in node.iter_expressions())

    for item in tree.expressions:
        if isinstance(item, exp.Star) or item.is_star or not visit(item, False):
            return None
    if not own or not (subqueries or tables):
        return None
    if reads and dialect not in _BARE_COLUMN_DIALECTS:
        return None
    test = outer if inner is None else inner if outer is None else exp.and_(exp.paren(inner.copy()), exp.paren(outer.copy()))
    if test is not None and any(_argument(call) is None for call in own):
        return None
    # write: every check is done, and the nodes collected above are nodes of this copy
    present = exp.GT(this=exp.Count(this=exp.Star()), expression=exp.Literal.number(0))
    _guard(present.this, test)
    for call in own:
        _guard(call, test)
    for node, rows in subqueries:
        node.replace(rows.value(rows.outputs[0][1], name))
    for column, alias in reads:
        rows = tables[alias]
        value = exp.Case(ifs=[exp.If(this=present.copy(), true=rows.value(rows.named(_lower(column.name)), name))])
        if column.parent is tree:
            value = exp.Alias(this=value, alias=exp.to_identifier(column.name))  # an output that is the bare column keeps its name
        column.replace(value)
    source = exp.Table(this=base.this.copy())
    if _lower(name.name) != _lower(base.name) or base.args.get("alias") is not None or name is not base.this:
        source.set("alias", exp.TableAlias(this=name.copy()))
    tree.set(FROM_KEY, exp.From(this=source))
    for key in ("joins", "where", "with_"):
        tree.set(key, None)
    return tree
