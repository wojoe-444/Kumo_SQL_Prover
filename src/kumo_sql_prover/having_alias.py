"""A select alias read in ``HAVING``, written as the expression it names.

    SELECT email, COUNT(email) AS n FROM person GROUP BY email HAVING n > 1

tests the group's count, exactly as ``HAVING COUNT(email) > 1`` does. MySQL, BigQuery, DuckDB and SQLite all
let ``HAVING`` read an output by its alias. The normalizer compares ``HAVING`` conditions as written, so a
derived table spelled with the alias never matched the same table spelled with the aggregate. This pass
replaces the name by the expression before anything else reads the ``HAVING``.

**Why it is the same test.** A bare name in ``HAVING`` is looked up among the columns of the ``FROM`` sources
and among the select aliases. When no source has a column of that name and exactly one output carries it, the
alias is the only thing the name can mean, and its value for a group is the value of the aliased expression
for that group. Putting the expression in its place evaluates the same thing, provided the expression is
something ``HAVING`` computes per group the way the select list does: every column in it is inside an
aggregate (the same rows are aggregated), or is one of the ``GROUP BY`` columns (one value per group).

**What is read.** A select with a ``GROUP BY``, or whose aliased expression holds an aggregate; a bare column in
its ``HAVING``, in the select's own scope and not inside an aggregate's argument, whose name is

* the alias of exactly one output, which is not simply the column of the same name,
* not a column of any source (every source's columns must be known: a table with supplied columns, or a
  derived table that lists its outputs; an unknown source, or a star, may have such a column, and then the
  engines read the column first, so the name is left alone).

**What is refused** (the name stays as written, which costs a proof and never makes one): an expression with a
window, a subquery or a function that changes from call to call; one that reads a column outside an aggregate
that is not a plain ``GROUP BY`` column (MySQL would pick a value from the group, and nothing says the same one
twice); one that reads a bare column named like another alias (the inserted name could be read as that alias);
a name shared by two outputs; a qualified name (it can only be a source column).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import extended_grouping, select_sources
from .clause_alias_reads import _volatile, may_be_source_column

_ATOMS = (exp.Column, exp.Literal, exp.Paren, exp.Null, exp.Boolean)
_OPERATORS = (exp.Binary, exp.Unary, exp.Predicate, exp.Bracket, exp.Dot)


def read_having_aliases(tree: exp.Expression, schema: dict[str, list[str]] | None) -> exp.Expression:
    """Replace, in place, each alias read described in the module docstring."""

    if not schema:
        return tree
    for select in list(tree.find_all(exp.Select)):
        having = select.args.get("having")
        if having is None:
            continue
        names = [item.alias_or_name.lower() for item in select.expressions]
        aliases = {item.alias.lower(): item.this for item in select.expressions if isinstance(item, exp.Alias) and item.alias}
        if not aliases:
            continue
        for column in list(having.find_all(exp.Column)):
            name = column.name.lower()
            if column.table or isinstance(column.this, exp.Star) or name not in aliases or names.count(name) != 1:
                continue
            if column.find_ancestor(exp.Select) is not select or _inside_aggregate(column, having):
                continue
            value = aliases[name]
            if isinstance(value, exp.Column) and value.name.lower() == name:
                continue
            if may_be_source_column(select, name, schema) or not _per_group(select, value, set(aliases) - {name}):
                continue
            inserted = value.copy()
            plain = isinstance(inserted, _ATOMS) or (isinstance(inserted, exp.Func) and not isinstance(inserted, _OPERATORS))
            column.replace(inserted if plain or not isinstance(column.parent, _OPERATORS) else exp.paren(inserted))
    return tree


def _inside_aggregate(node: exp.Expression, top: exp.Expression) -> bool:
    """Whether an aggregate call lies between ``node`` and ``top`` (``top`` itself counts)."""

    while node is not top and node.parent is not None:
        node = node.parent
        if isinstance(node, exp.AggFunc):
            return True
    return False


def _per_group(select: exp.Select, value: exp.Expression, other_aliases: set[str]) -> bool:
    """Whether ``value`` is computed per group from aggregates and plain grouping columns only."""

    if _volatile(value) or any(isinstance(node, (exp.Window, exp.Query, exp.Subquery)) for node in value.walk()):
        return False
    group = select.args.get("group")
    if group is None and value.find(exp.AggFunc) is None:
        return False
    single = len(select_sources(select)) == 1
    grouped: list[tuple[str, str]] = []
    if group is not None:
        if extended_grouping(group) or any(part for key, part in group.args.items() if key != "expressions"):
            return False
        for item in group.expressions:
            while isinstance(item, exp.Paren):
                item = item.this
            if isinstance(item, exp.Column) and not isinstance(item.this, exp.Star):
                grouped.append((item.table.lower(), item.name.lower()))
    for column in value.find_all(exp.Column):
        if isinstance(column.this, exp.Star):
            return False
        if not column.table and column.name.lower() in other_aliases:
            return False
        if _inside_aggregate(column, value):
            continue
        table, name = column.table.lower(), column.name.lower()
        if not any(name == g_name and (table == g_table or single) for g_table, g_name in grouped):
            return False
    return True
