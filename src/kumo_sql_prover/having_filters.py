"""MySQL's ``HAVING`` in a select that groups nothing is a filter on the selected rows (hardest-cases batch 31).

    SELECT name, bonus FROM employee LEFT JOIN bonus ON .. HAVING bonus < 1000 OR bonus IS NULL

Standard SQL reads a ``HAVING`` without ``GROUP BY`` as a test on the one group of all rows, and rejects the query
above because ``bonus`` is neither grouped nor aggregated (DuckDB, PostgreSQL, SQLite and BigQuery do). MySQL runs
it: with no ``GROUP BY`` and no aggregate function the select is not a grouped one, and ``HAVING`` keeps the rows
on which its condition is TRUE, after the select list is computed. Checked on MySQL 8.0 for every spelling below.

The prover read such a select the standard way, as one group, and so reported
``SELECT a FROM t HAVING a > 1`` and ``SELECT a FROM t WHERE a > 1`` as different with a database on which MySQL
returns the same rows from both: a wrong counterexample. :func:`read` writes the ``HAVING`` as a ``WHERE`` before
either prover sees the text, and names a select it cannot read so that the pair is declined instead.

**How MySQL resolves a name in such a HAVING** (``find_item_in_list``; each line checked on MySQL 8.0):

* an unqualified name is first an output name of the select list: an alias, or the name of a column selected
  without one. ``SELECT a AS b FROM t HAVING b > 4`` tests ``a``, also when the table has a column ``b``; two
  different items of that name are an error (``Column 'b' in having clause is ambiguous``);
* failing that, the column of that name selected under another alias: ``SELECT t.a AS x FROM t HAVING a > 2``;
* a qualified name is a selected column of that table, with or without an alias (``SELECT t.a AS x .. HAVING
  t.a > 2``); ``SELECT a + 10 AS a FROM t HAVING t.a > 2`` is an error;
* a column that is not selected is an error (``Unknown column 'b' in 'having clause'``), a ``*`` selects every
  column of its tables.

So each name of the condition is replaced by the select item it names and the condition moves to ``WHERE``, where
the item's expression reads the same rows it reads in the select list. ``DISTINCT``, ``ORDER BY`` and ``LIMIT``
come after both, and a condition that only reads selected values keeps the same rows before or after duplicates
are removed.

**Left alone, and then declined** (the old reading as one group is wrong for them in MySQL, so no verdict is
given): a name that does not resolve as above or resolves twice; a subquery in the condition that reads anything
but its own tables (inside a subquery MySQL also finds columns that are not selected, and the order in which it
looks is not modeled); a window function, a user variable, ``ANY_VALUE``, a function the parser does not know (it
may be an aggregate) or a volatile call in the select; a select with no ``FROM``; a named item that holds a division (MySQL
may compare the quotient as the select list stores it, rounded to its declared scale, see issue 45), or a subquery
that could read a select alias (MySQL allows ``(SELECT alias)`` in the select list, and ``WHERE`` has no aliases);
an aggregate inside a subquery whose columns cannot be shown to be the subquery's own (an aggregate of outer
columns belongs to the outer select and makes it a grouped one).

Other dialects are not touched: there the query is an error, or (with constants only) the one group of the
standard, which is what the prover reads.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from . import volatile_calls

NAME = "having_filters"
DIALECTS = ("mysql",)
REASON = "unsupported: HAVING in a select with no GROUP BY and no aggregate (MySQL filters the selected rows with it; this one cannot be read as a WHERE)"

_UNREADABLE = (exp.Window, exp.Parameter, exp.SessionParameter, exp.Placeholder, exp.Anonymous, exp.PropertyEQ, exp.AnyValue)


def _lower(name: str | None) -> str:
    return (name or "").lower()


def _with(node: exp.Expression) -> exp.Expression | None:
    return node.args.get("with_") or node.args.get("with")


def _from(select: exp.Select) -> exp.Expression | None:
    return select.args.get("from_") or select.args.get("from")


def _own(node: exp.Expression, select: exp.Select) -> bool:
    return node.find_ancestor(exp.Select) is select


def _is_star(item: exp.Expression) -> bool:
    return isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star))


def plain(select: exp.Select) -> bool:
    """Whether ``select`` has a ``HAVING``, no ``GROUP BY`` and no aggregate function of its own: not a grouped
    select in MySQL. (The function of a window is not an aggregate of the select.)"""

    if select.args.get("having") is None or select.args.get("group") is not None:
        return False
    for call in select.find_all(exp.AggFunc):
        if isinstance(call, exp.AnyValue):
            continue  # not an aggregate function in MySQL: it does not make the select a grouped one
        if _own(call, select) and not (isinstance(call.parent, exp.Window) and call.arg_key == "this"):
            return False
    return True


# ---------------------------------------------------------------------------------------------------------------
# whose column is it


def _output_names(query: exp.Expression | None) -> list[str] | None:
    while isinstance(query, (exp.Subquery, exp.SetOperation)):
        query = query.this
    if not isinstance(query, exp.Select):
        return None
    names = []
    for item in query.expressions:
        if _is_star(item):
            return None
        names.append(_lower(item.alias_or_name))
    return names


def _ctes(node: exp.Expression) -> dict[str, exp.CTE]:
    """The WITH names visible at ``node``: the nearest definition of a name wins."""

    found: dict[str, exp.CTE] = {}
    while node is not None:
        clause = _with(node) if isinstance(node, (exp.Select, exp.SetOperation, exp.Subquery)) else None
        for cte in clause.expressions if clause is not None else ():
            found.setdefault(_lower(cte.alias), cte)
        node = node.parent
    return found


def _sources(select: exp.Select, schema: dict[str, list[str]]) -> list[tuple[str, list[str] | None]] | None:
    """``(name in the query, columns or None when they cannot be told)`` for each source of ``select``; ``None``
    for a source that is neither a table nor a derived table."""

    out: list[tuple[str, list[str] | None]] = []
    first = _from(select)
    sources = ([first.this] if first is not None else []) + [join.this for join in select.args.get("joins") or []]
    for source in sources:
        if type(source) is exp.Table and isinstance(source.this, exp.Identifier):
            cte = _ctes(select).get(_lower(source.name)) if not source.db else None
            if cte is not None:
                listed = [_lower(c.name) for c in (cte.args.get("alias").args.get("columns") or [])] if cte.args.get("alias") is not None else []
                columns = listed or _output_names(cte.this)
            else:
                columns = schema.get(_lower(source.name))
            out.append((_lower(source.alias_or_name), columns))
        elif isinstance(source, exp.Subquery) and source.alias:
            alias = source.args.get("alias")
            listed = [_lower(c.name) for c in (alias.args.get("columns") or [])]
            out.append((_lower(source.alias), listed or _output_names(source.this)))
        else:
            return None
    return out


def _inner(column: exp.Column, outer: exp.Select, schema: dict[str, list[str]]) -> bool:
    """Whether ``column``, which sits in a subquery of ``outer``, can be shown to be a column of a select between
    itself and ``outer``. False also when that cannot be told."""

    scope = column.find_ancestor(exp.Select)
    while scope is not None and scope is not outer:
        sources = _sources(scope, schema)
        if sources is None:
            return False
        if column.table:
            if any(name == _lower(column.table) for name, _ in sources):
                return True
        else:
            if any(columns is None for _, columns in sources):
                return False  # the name may be one of these columns or an outer one
            if any(_lower(column.name) in columns for _, columns in sources):
                return True
            clause = column
            while clause.parent is not None and clause.parent is not scope:
                clause = clause.parent
            if clause.arg_key in ("group", "having", "order") and _lower(column.name) in (_output_names(scope) or ()):
                return True  # an alias of that select, read in a clause that sees aliases
        scope = _enclosing(scope)
    return False


def _enclosing(select: exp.Select) -> exp.Select | None:
    """The nearest select whose sources ``select`` can read: not one it is itself a source or a WITH body of."""

    node: exp.Expression = select
    while node.parent is not None:
        parent = node.parent
        if isinstance(parent, exp.Select):
            if node.arg_key in ("from", "from_", "joins", "with", "with_"):
                node = parent  # a derived table (not LATERAL, which is not read) or a WITH body: skip this select
                continue
            return parent
        node = parent
    return None


def _nested_columns(node: exp.Expression, select: exp.Select) -> list[exp.Column]:
    return [c for c in node.find_all(exp.Column) if not _own(c, select) and not isinstance(c.this, exp.Star)]


def _self_contained(node: exp.Expression, select: exp.Select, schema: dict[str, list[str]]) -> bool:
    """Every column in the subqueries of ``node`` (a clause of ``select``) is a column of those subqueries."""

    if any(isinstance(c.this, exp.Star) and c.table and not _own(c, select) and not _inner(c, select, schema) for c in node.find_all(exp.Column)):
        return False
    return all(_inner(column, select, schema) for column in _nested_columns(node, select))


def _aggregates_stay_inside(node: exp.Expression, select: exp.Select, schema: dict[str, list[str]]) -> bool:
    """Each aggregate in a subquery of ``node`` reads a column of that subquery (or no column at all), so it is not
    an aggregate of ``select``."""

    for call in node.find_all(exp.AggFunc):
        if _own(call, select):
            continue
        columns = [c for c in call.find_all(exp.Column) if not isinstance(c.this, exp.Star)]
        if columns and not any(_inner(c, select, schema) for c in columns):
            return False
    return True


# ---------------------------------------------------------------------------------------------------------------
# the names of the condition


class _Items:
    """The select list of one select, as MySQL looks names up in it."""

    def __init__(self, select: exp.Select, schema: dict[str, list[str]]):
        self.select = select
        self.sources = _sources(select, schema)
        self.items = list(select.expressions)
        self.stars = [item for item in self.items if _is_star(item)]
        self.named: dict[str, list[exp.Expression]] = {}  # output name -> values
        self.fields: dict[str, list[exp.Column]] = {}  # column name -> the columns selected under another alias
        for item in self.items:
            if _is_star(item):
                continue
            value = (item.this if isinstance(item, exp.Alias) else item).unnest()  # `(a)` is the column a
            if isinstance(item, exp.Alias) and item.alias:
                self.named.setdefault(_lower(item.alias), []).append(value)
                if isinstance(value, exp.Column) and _lower(value.name) != _lower(item.alias):
                    self.fields.setdefault(_lower(value.name), []).append(value)
            elif isinstance(value, exp.Column):
                self.named.setdefault(_lower(value.name), []).append(value)
        self.aliases = {
            name
            for name, values in self.named.items()
            if not all(isinstance(v, exp.Column) and _lower(v.name) == name for v in values)
        }

    def _starred(self, name: str, table: str | None) -> bool:
        """``name`` (of ``table``) is a column that a ``*`` of the select list selects, and no other source has it."""

        if not self.stars or self.sources is None or any(columns is None for _, columns in self.sources):
            return False
        having = [source for source, columns in self.sources if name in columns]
        if len(having) != 1 or (table is not None and having[0] != table) or len([s for s, _ in self.sources if s == having[0]]) != 1:
            return False
        return any(isinstance(star, exp.Star) or _lower(star.table) == having[0] for star in self.stars)

    def value(self, column: exp.Column) -> exp.Expression | None:
        """What a column of the condition stands for: a node of the select list, or ``column`` itself for a column
        the select list has under that very name. ``None`` when MySQL would not find it, or find two."""

        name, table = _lower(column.name), _lower(column.table) or None
        if column.args.get("db") or column.args.get("catalog"):
            return None
        if table is None:
            if self.stars and (self.sources is None or any(columns is None or name in columns for _, columns in self.sources)):
                # a star may select a column of this name: only that column, and only when no item has the name too
                return column if name not in self.named and name not in self.fields and self._starred(name, None) else None
            for found in (self.named.get(name), self.fields.get(name)):
                if found:
                    return found[0] if len({value.sql() for value in found}) == 1 else None
            return None
        matches = {
            value.sql()
            for values in self.named.values()
            for value in values
            if isinstance(value, exp.Column) and _lower(value.name) == name and self._table_of(value) == table
        }
        if matches:
            return column if len(matches) == 1 else None
        return column if self._starred(name, table) else None

    def _table_of(self, value: exp.Column) -> str | None:
        if value.table:
            return _lower(value.table)
        if self.sources is not None and len(self.sources) == 1:
            return self.sources[0][0]  # the only source: an unqualified column is its column
        return None


def _usable(value: exp.Expression, items: _Items, schema: dict[str, list[str]]) -> bool:
    """Whether a select item may be written into ``WHERE`` in place of its name."""

    if isinstance(value, (exp.Column, exp.Literal)):
        return True
    if value.find(exp.Div) is not None:
        return False  # MySQL may test the quotient as stored, rounded to its declared scale
    for column in _nested_columns(value, items.select):
        if not column.table and _lower(column.name) in items.aliases and not _inner(column, items.select, schema):
            return False  # (SELECT alias): an alias of this select list, which WHERE does not see
    return True


def _read_select(select: exp.Select, schema: dict[str, list[str]], dialect: str) -> bool:
    """Write the ``HAVING`` of a plain select as a ``WHERE``, in place. False when it is left as it is."""

    having = select.args.get("having")
    if _from(select) is None or any(select.args.get(key) for key in ("qualify", "windows", "into", "connect", "pivots", "laterals")):
        return False
    listed = [*select.expressions, having]
    for node in select.walk():
        if node is select:
            continue
        if isinstance(node, _UNREADABLE) and (_own(node, select) or not isinstance(node, exp.Window)):
            return False  # of the select itself, or (all but a window) anywhere under it
    if any(volatile_calls.contains(node, dialect) for node in listed):
        return False
    if not all(_aggregates_stay_inside(node, select, schema) for node in listed):
        return False
    if not _self_contained(having, select, schema):
        return False
    items = _Items(select, schema)
    if items.sources is None:
        return False
    if items.stars and any(join.args.get("using") or (join.method or "").upper() == "NATURAL" for join in select.args.get("joins") or []):
        return False  # a star over USING or NATURAL selects merged columns
    condition = having.this.copy()
    holder = exp.Paren(this=condition)
    for column in [c for c in holder.find_all(exp.Column) if c.find_ancestor(exp.Select) is None]:
        if isinstance(column.this, exp.Star):
            return False
        value = items.value(column)
        if value is None or not _usable(value, items, schema):
            return False
        if value is not column:
            column.replace(value.copy() if isinstance(value, (exp.Column, exp.Literal)) else exp.Paren(this=value.copy()))
    where = select.args.get("where")
    test = holder if where is None else exp.And(this=exp.Paren(this=where.this.copy()), expression=holder)
    select.set("where", exp.Where(this=test))
    select.set("having", None)
    return True


def read(sql: str, dialect: str, schema: dict[str, list[str]] | None = None) -> tuple[str, str | None]:
    """``(sql with each such HAVING written as a WHERE, None)``, or ``(sql, REASON)`` when a select of that kind
    cannot be read. ``sql`` itself comes back when nothing applies."""

    if dialect not in DIALECTS or "having" not in sql.lower():
        return sql, None
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:  # noqa: BLE001 - the prover reports the parse error itself
        return sql, None
    selects = [select for select in tree.find_all(exp.Select) if plain(select)]
    if not selects:
        return sql, None
    known = {_lower(table): [_lower(c) for c in columns] for table, columns in (schema or {}).items()}
    for select in reversed(selects):  # the innermost first: reading an outer one copies what its condition holds
        if not _read_select(select, known, dialect):
            return sql, REASON
    if any(plain(select) for select in tree.find_all(exp.Select)):
        return sql, REASON
    return tree.sql(dialect=dialect), None
