"""A column list on the first table of a parenthesized join, read back as the table's own names.

    SELECT x.v0 FROM r JOIN (d AS x (k0, v0) JOIN u ON x.k0 = u.b) ON r.a = x.v0
    SELECT x.v AS v0 FROM r JOIN (d AS x JOIN u ON x.k = u.b) ON r.a = x.v

``d AS x (k0, v0)`` renames the table's columns by position. ``ast_utils.expand_alias_columns`` writes the
renaming as a derived table for a table that is a ``FROM`` or ``JOIN`` item. The parser hangs the first table
of a parenthesized join one level lower, under the parentheses, with the group's joins attached to it, and that
pass does not look there. The list then survived into the normal form; a later merge of the table dropped it
while references to the new names stayed, and the prover declined the query (``x.k0 is not a column of x``).
Calcite prints every self-join this way.

Here the list is removed and every read of a new name is written with the table's own name instead. That is
the same query when every read can be found, so the pass checks, and otherwise leaves the query as written:

* the table is in the schema, is not a name a ``WITH`` in scope defines, and the list is no longer than the
  table; columns are taken in the schema's order, as the engine does, and a shorter list renames the first
  columns only. The names after renaming are distinct;
* the alias names one source in the whole select that holds the join, subqueries and derived tables included
  (a second source of that name anywhere could be the one a column means);
* every column of that select that names the alias reads one of its columns, and no column that can see the
  select's sources is written bare with a name the table has before or after renaming (a bare name could be
  this table's; one inside a derived table reads only that table's own sources);
* no star in the select's own list and no ``alias.*`` anywhere (a star would return the new names), and no
  ``USING`` or ``NATURAL`` join in the select (they name columns without a source).

A renamed column that is an output of a select keeps its new name as an alias, so the select's column names do
not change.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import visible_ctes

_TABLE_ARGS = frozenset({"this", "db", "catalog", "alias", "joins"})


def _read_back(table: exp.Table, alias: exp.TableAlias, lowered: dict[str, list[str]]) -> bool:
    renamed = [c.name.lower() for c in alias.args.get("columns") or []]
    key = ".".join(part.name for part in table.parts).lower()
    if not table.db and any(name.lower() == key for name in visible_ctes(table)):
        return False
    columns = lowered.get(key) or (lowered.get(table.name.lower()) if table.db else None)
    if not columns or len(renamed) > len(columns):
        return False
    columns = [c.lower() for c in columns]
    after = renamed + columns[len(renamed):]
    if len(set(after)) != len(after) or len(set(columns)) != len(columns):
        return False
    original = dict(zip(after, columns))
    select = table.find_ancestor(exp.Select)
    name = alias.name.lower()
    if select is None or not name:
        return False
    sources = [n for n in select.walk() if isinstance(n, (exp.Table, exp.Subquery, exp.Lateral, exp.Unnest, exp.Values)) and (n.alias_or_name or "").lower() == name]
    if len(sources) != 1 or sources[0] is not table:
        return False
    if any(isinstance(n, exp.CTE) and n.alias.lower() == name for n in select.walk()):
        return False
    if any(j.args.get("using") is not None or j.args.get("method") for j in select.find_all(exp.Join)):
        return False
    if any(isinstance(n, exp.Star) or (isinstance(n, exp.Column) and isinstance(n.this, exp.Star)) for n in select.expressions):
        return False
    taken = set(after) | set(columns)
    reads = []
    for column in select.find_all(exp.Column):
        if isinstance(column.this, exp.Star):
            if not column.table or column.table.lower() == name:
                return False
            continue
        if not column.table:
            if column.name.lower() in taken and _sees(column, select):
                return False
            continue
        if column.table.lower() != name:
            continue
        if column.name.lower() not in original or column.args.get("db") or column.args.get("catalog"):
            return False
        reads.append(column)
    for column in reads:
        new = column.name.lower()
        old = original[new]
        if old == new:
            continue
        written = exp.column(old, table=column.args["table"].copy())
        parent = column.parent
        if isinstance(parent, exp.Select) and column.arg_key == "expressions":
            written = exp.alias_(written, column.this.copy())  # the select's output keeps its name
        column.replace(written)
    alias.set("columns", None)
    return True


def _sees(node: exp.Expression, select: exp.Select) -> bool:
    """Whether ``node`` can read the sources of ``select``: it is not inside a derived table of its own scope."""

    node = node.parent
    while node is not None and node is not select:
        if isinstance(node, exp.Subquery) and isinstance(node.this, exp.Query) and isinstance(node.parent, (exp.From, exp.Join, exp.Table)):
            return False  # a derived table (not LATERAL, which is another node) reads only its own sources
        node = node.parent
    return True


def read_back_renamed_join_heads(tree: exp.Expression, schema: dict[str, list[str]] | None) -> exp.Expression:
    """Write ``(t AS x (a, b) JOIN ..)`` as ``(t AS x JOIN ..)``, with each read of ``x.a`` as ``x.<first column of t>``."""

    if not schema:
        return tree
    lowered = {key.lower(): list(columns) for key, columns in schema.items()}
    for alias in list(tree.find_all(exp.TableAlias)):
        table = alias.parent
        if not alias.args.get("columns") or not isinstance(table, exp.Table) or not isinstance(table.this, exp.Identifier):
            continue
        group = table.parent
        if not isinstance(group, exp.Subquery) or group.this is not table or not table.args.get("joins"):
            continue  # a FROM or JOIN item is expand_alias_columns' to read
        if any(value for key, value in table.args.items() if key not in _TABLE_ARGS and value):
            continue
        _read_back(table, alias, lowered)
    return tree
