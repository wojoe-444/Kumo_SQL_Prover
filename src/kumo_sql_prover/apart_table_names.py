"""A table read under its own name by a select and inside a derived table of that select gets a name at each place.

"Every manager with the number of people who report to them" is often written with the staff table at two levels,
both times under its own name:

    SELECT staff.id, staff.name, t.n
    FROM staff JOIN (SELECT boss, COUNT(*) AS n FROM staff GROUP BY boss) AS t ON t.boss = staff.id

Inside the derived table ``staff`` is the table read there, outside it is the table joined to ``t``. The rules that
merge a derived table into the select around it leave such a query alone: once the two selects are one, the one
name would mean two sources. Written with a name for each read,

    SELECT staff_1.id, staff_1.name, t.n
    FROM staff AS staff_1 JOIN (SELECT boss, COUNT(*) AS n FROM staff AS staff_2 GROUP BY boss) AS t ON t.boss = staff_1.id

it is the query those rules already read when its author named the two reads apart.

**Why it holds.** An alias changes no row: it only changes the name the columns of that source are asked by. A
column written ``staff.c`` belongs to the nearest select around it, going outwards, that reads a source called
``staff``; that is how the engines resolve the name, an inner source hiding an outer one of the same name. Each such
column is renamed with the source it belongs to, a column written without a table name is left as it is (its source
does not depend on what the sources are called), and the new names are names nothing in the query uses. Output
names are column names and do not change.

**Left alone.** A table read once; two reads of which neither is inside a derived table of the other's select (the
branches of a ``UNION``, or a read inside an ``IN`` or ``EXISTS`` subquery or a scalar subquery: the prover reads
those under one name already, and the rule would only cost a second run); reads that already have aliases; a
table read with a database name, a table function, a table with a column list, a sample or a hint; a query with a
qualified column that names no source around it, or with a database-qualified column; a select that reads two
sources of one name (the engines refuse it), a parenthesized join or a lateral source.

A pre-pass of the last proof step (``shape_fallback.TWENTY_FOURTH_BATCH``).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY

NAME = "apart_table_names"


def _lower(name: object) -> str:
    return str(name or "").lower()


def _sources(select: exp.Select) -> list[exp.Expression] | None:
    """The sources a select reads itself (tables and derived tables of its ``FROM`` and joins), or None for a
    source list the rule does not reason about (a parenthesized join, a lateral source)."""

    if select.args.get("laterals"):
        return None
    from_ = select.args.get(FROM_KEY)
    items = ([from_.this] if from_ is not None else []) + [join.this for join in select.args.get("joins") or []]
    for item in items:
        if isinstance(item, exp.Table):
            if item.args.get("joins"):
                return None
        elif not isinstance(item, (exp.Subquery, exp.Unnest, exp.Values)) or isinstance(item.this, exp.Table):
            return None
    return items


def _plain(table: exp.Table) -> bool:
    return isinstance(table.this, exp.Identifier) and not {key for key, value in table.args.items() if value} - {"this"}


def _derived(one: exp.Select, other: exp.Select) -> bool:
    """Whether one of two selects is inside a derived table of the other."""

    return one is not other and (_inside_derived(one, other) or _inside_derived(other, one))


def _inside_derived(inner: exp.Select, outer: exp.Select) -> bool:
    """Whether ``inner`` is a derived table of ``outer`` at any depth (reached through ``FROM`` and join sources
    alone, and the branches of a set operation there), or a select anywhere inside one."""

    node: exp.Expression | None = inner
    while node is not None:
        if isinstance(node, exp.Select) and node is not outer and _derived_in(node, outer):
            return True
        node = node.parent
    return False


def _derived_in(inner: exp.Select, outer: exp.Select) -> bool:
    node: exp.Expression = inner
    while node is not outer:
        parent = node.parent
        if parent is None:
            return False
        while isinstance(parent, exp.SetOperation):
            node, parent = parent, parent.parent
        holder = parent.parent if isinstance(parent, exp.Subquery) else None
        if isinstance(holder, exp.From) or (isinstance(holder, exp.Join) and holder.this is parent):
            node = holder.parent
        else:
            return False
        if not isinstance(node, exp.Select):
            return False
    return True


def name_tables_apart(tree: exp.Expression) -> tuple[exp.Expression, bool]:
    """``tree`` with an alias on every unaliased read of a table that is read unaliased too inside a derived table
    of its select, or by a select that has its own select inside a derived table; whether it changed."""

    tables = [table for table in tree.find_all(exp.Table) if isinstance(table.this, exp.Identifier)]
    reads: dict[str, int] = {}
    for table in tables:
        reads[_lower(table.name)] = reads.get(_lower(table.name), 0) + 1
    if not any(count > 1 and any(_plain(table) and _lower(table.name) == name for table in tables) for name, count in reads.items()):
        return tree, False
    result = tree.copy()
    return (result, True) if _rename(result) else (tree, False)


def _rename(tree: exp.Expression) -> bool:
    selects = list(tree.find_all(exp.Select))
    sources = {id(select): _sources(select) for select in selects}
    if any(found is None for found in sources.values()):
        return False
    owner: dict[int, exp.Select] = {}
    for select in selects:
        names = [_lower(source.alias_or_name) for source in sources[id(select)]]
        if len(set(names)) != len(names):
            return False
        for source in sources[id(select)]:
            owner[id(source)] = select
    tables = [table for table in tree.find_all(exp.Table) if isinstance(table.this, exp.Identifier)]
    reads: dict[str, int] = {}
    for table in tables:
        reads[_lower(table.name)] = reads.get(_lower(table.name), 0) + 1
    plain = [table for table in tables if _plain(table) and reads[_lower(table.name)] > 1 and id(table) in owner]
    targets = [table for table in plain if any(_derived(owner[id(table)], owner[id(other)]) for other in plain if other is not table and _lower(other.name) == _lower(table.name))]
    if not targets:
        return False
    # every qualified column, with the source it names
    bound: dict[int, list[exp.Column]] = {}
    for column in tree.find_all(exp.Column):
        if column.args.get("db") or column.args.get("catalog"):
            return False
        if not column.table:
            continue
        name, node, source = _lower(column.table), column, None
        while source is None:
            select = node.find_ancestor(exp.Select)
            if select is None:
                return False  # a name no select around the column reads
            source = next((s for s in sources[id(select)] if _lower(s.alias_or_name) == name), None)
            node = select
        bound.setdefault(id(source), []).append(column)
    taken = {_lower(node.name) for node in tree.find_all(exp.Identifier)}
    number = 0
    for table in targets:
        number += 1
        while f"{_lower(table.name)}_{number}" in taken:
            number += 1
        alias = f"{_lower(table.name)}_{number}"
        taken.add(alias)
        for column in bound.get(id(table), []):
            column.set("table", exp.to_identifier(alias))
        table.set("alias", exp.TableAlias(this=exp.to_identifier(alias)))
    return True
