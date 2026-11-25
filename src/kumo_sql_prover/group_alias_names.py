"""A ``GROUP BY`` name that only the select list defines is the expression it names.

    SELECT viewer_id AS id FROM (SELECT viewer_id, view_date FROM views GROUP BY viewer_id, view_date) AS p
    GROUP BY id                                           -- GROUP BY viewer_id

``p`` has no column ``id``, so the name in ``GROUP BY`` can only be the output ``viewer_id AS id``. Over a plain
table the qualifier resolves such a name; over a derived table it stayed as written, and the rules that read a
grouping by its columns (a grouping over a grouped derived table as a ``DISTINCT``, among others) did not see one.
This pre-pass (hardest-cases batch 29) writes the expression in the name's place.

**Why it is the same query.** MySQL, PostgreSQL, SQLite and DuckDB all accept an output name in ``GROUP BY``. They
differ on which wins when a column of the ``FROM`` tables has the same name, so the pre-pass reads the name only
when none has it: every source of the select is a declared table or a derived table whose output names are all
written out, and none of them lists the name. Then the name has one meaning in all four, the select item of that
alias, and grouping by the item's expression groups by the same values.

**What is read.** A bare, unquoted name in the ``GROUP BY`` of a select that is the statement itself, a branch of a
set operation, or a derived table (not ``LATERAL``) of such a select, so no outer query can give the name a third
meaning. Exactly one select item has the alias. Its expression reads at least one column, holds no aggregate,
window, subquery or volatile call, and none of its own bare names is the alias of another item. Every other select
item has an alias or is a column.

**Left alone.** A name some source has as a column; a source whose columns are not known (an undeclared table, a
table function, a derived table with ``*`` or repeated output names); a select item with no alias that is not a
column (its output name is the engine's to choose); ``ROLLUP``, ``CUBE`` and grouping sets; a
select inside a ``WHERE``, ``HAVING`` or select-list subquery; an alias two items carry; an alias of a constant (a
number written in ``GROUP BY`` would be read as a position); every other dialect.

It runs in the last step of batch 27 only, so a pair an earlier step decides is decided as it was.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import _output_names, extended_grouping, select_sources

NAME = "group_alias_names"
DIALECTS = frozenset({"mysql", "postgres", "sqlite", "duckdb"})


def read_group_alias_names(tree: exp.Expression, schema: dict[str, list[str]] | None, dialect: str | None) -> tuple[exp.Expression, bool]:
    """``(tree with each such GROUP BY name written as its expression, whether one was)``. The input is not changed."""

    if (dialect or "").lower() not in DIALECTS or not schema:
        return tree, False
    known = {name.lower().split(".")[-1]: {str(c).lower() for c in columns} for name, columns in schema.items()}
    if len(known) != len(schema):
        return tree, False  # two declared tables that differ only by their database: names are not told apart here
    if not any(_readings(select, known, dialect) for select in tree.find_all(exp.Select)):
        return tree, False
    copy = tree.copy()
    for select in list(copy.find_all(exp.Select)):
        for key, value in _readings(select, known, dialect):
            key.replace(value.copy())
    return copy, True


def _readings(select: exp.Select, known: dict[str, set[str]], dialect: str | None) -> list[tuple[exp.Column, exp.Expression]]:
    """``(GROUP BY name, the expression it stands for)`` for the names of ``select`` that are read."""

    group = select.args.get("group")
    if group is None or extended_grouping(group) or any(value for key, value in group.args.items() if key != "expressions"):
        return []
    names = [key for key in group.expressions if _bare(key)]
    if not names or not _closed(select):
        return []
    listed = _source_columns(select, known)
    if listed is None:
        return []
    aliases: dict[str, list[exp.Expression]] = {}
    for item in select.expressions:
        if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
            return []  # a star may list an output of the name
        if isinstance(item, exp.Alias):
            if not isinstance(item.args.get("alias"), exp.Identifier) or item.args["alias"].args.get("quoted"):
                return []
            aliases.setdefault(item.alias.lower(), []).append(item.this)
        elif not isinstance(item, exp.Column):
            return []  # an unnamed expression: its output name is the engine's to choose ('abc' is named abc in MySQL)
    found = []
    for key in names:
        name = key.name.lower()
        if name in listed or len(aliases.get(name, ())) != 1:
            continue
        value = aliases[name][0]
        if not _groupable(value, set(aliases) - {name}, dialect):
            continue
        found.append((key, value))
    return found


def _bare(node: exp.Expression) -> bool:
    return isinstance(node, exp.Column) and isinstance(node.this, exp.Identifier) and not node.this.args.get("quoted") and not node.table and not node.args.get("db") and not node.args.get("catalog")


def _closed(select: exp.Select) -> bool:
    """Whether no enclosing query can give a bare name of ``select`` a meaning: the statement, a branch of a set
    operation, or a derived table that is not lateral, of such a select."""

    node: exp.Expression = select
    while node.parent is not None:
        parent = node.parent
        if isinstance(parent, exp.SetOperation) and node.arg_key in ("this", "expression"):
            node = parent
        elif isinstance(parent, exp.Subquery) and node.arg_key == "this" and isinstance(parent.parent, (exp.From, exp.Join)) and not isinstance(parent.parent.parent, exp.Lateral):
            holder = parent.parent
            if isinstance(holder, exp.Join) and (holder.args.get("kind") or "").upper() == "LATERAL":
                return False
            outer = holder.parent
            if not isinstance(outer, exp.Select) or outer.args.get("laterals"):
                return False
            node = outer
        elif isinstance(parent, exp.Subquery) and node.arg_key == "this" and parent.parent is None and not any(value for key, value in parent.args.items() if key != "this"):
            node = parent
        else:
            return False
    return True


def _source_columns(select: exp.Select, known: dict[str, set[str]]) -> set[str] | None:
    """Every column name the sources of ``select`` have, None when one of them cannot be listed."""

    if select.args.get("laterals") or select.args.get("with") is not None:
        return None
    sources = select_sources(select)
    if not sources:
        return None
    listed: set[str] = set()
    for source in sources:
        if isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier) and not source.args.get("pivots") and not source.args.get("joins"):
            columns = known.get(source.name.lower())
            if columns is None:
                return None
            listed |= columns
        elif isinstance(source, exp.Subquery) and source.alias and not source.args.get("pivots") and not source.args.get("joins"):
            alias = source.args.get("alias")
            if isinstance(alias, exp.TableAlias) and alias.args.get("columns"):
                return None
            names = _output_names(source.this)
            if names is None:
                return None
            listed |= {name.lower() for name in names}
        else:
            return None
    return listed


def _groupable(value: exp.Expression, other_aliases: set[str], dialect: str | None) -> bool:
    """Whether a select item's expression can stand in ``GROUP BY`` with the meaning it has in the select list."""

    from .volatile_calls import is_volatile

    if value.find(exp.AggFunc, exp.Window, exp.Query, exp.Subquery, exp.Star, exp.Placeholder, exp.Parameter) is not None:
        return False
    columns = list(value.find_all(exp.Column))
    if not columns or any(not isinstance(column.this, exp.Identifier) for column in columns):
        return False
    if any(not column.table and column.name.lower() in other_aliases for column in columns):
        return False  # the select list reads the column; GROUP BY may read the other item
    return not any(is_volatile(node, dialect) for node in value.walk())
