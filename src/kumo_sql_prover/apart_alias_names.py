"""An alias used by a select and again inside a derived table of that select gets a name at each place.

"Each test with the value of its left and of its right name" is often written in two steps, and the lookup table
gets the same short name in both:

    SELECT c.l, c.r, c.lv, v.v AS rv
    FROM (SELECT e.l, e.r, v.v AS lv FROM tests AS e LEFT JOIN vals AS v ON e.l = v.name) AS c
    LEFT JOIN vals AS v ON c.r = v.name

Inside the derived table ``v`` is the first lookup, outside it is the second. The rules that merge a derived table
into the select around it leave such a query alone: once the two selects are one, the one name would mean two
sources. Written with a name for each,

    SELECT c.l, c.r, c.lv, v_1.v AS rv
    FROM (SELECT e.l, e.r, v_2.v AS lv FROM tests AS e LEFT JOIN vals AS v_2 ON e.l = v_2.name) AS c
    LEFT JOIN vals AS v_1 ON c.r = v_1.name

it is the query those rules already read when its author chose two names. ``apart_table_names`` does this for a
table read twice under its own name; this module does it for every other clash of two source names: two aliases,
an alias and a table's own name, the alias of a derived table and a source inside it.

**Why it holds.** An alias changes no row: it only changes the name the columns of that source are asked by. A
column written ``v.c`` belongs to the nearest select around it, going outwards, that reads a source called ``v``;
that is how the engines resolve the name, an inner source hiding an outer one of the same name. Each such column is
renamed with the source it belongs to, a column written without a source name is left as it is (its source does not
depend on what the sources are called), and the new names are names nothing in the query uses. Output names are
column names and do not change.

**Left alone.** Two sources of one name of which neither is inside a derived table of the other's select (the
branches of a ``UNION``, or a read inside an ``IN`` or ``EXISTS`` subquery or a scalar subquery: the prover reads
those under one name already, and the rule would only cost a second run); a table read with a database name, a
table function, a sample or a hint and no alias; a query with a qualified column that names no source around it, or
with a database-qualified column; a select that reads two sources of one name (the engines refuse it), a
parenthesized join or a lateral source.

A pre-pass of the last proof step (``shape_fallback.TWENTY_FIFTH_BATCH``), after ``apart_table_names``.
"""

from __future__ import annotations

from sqlglot import exp

from .apart_table_names import _derived, _lower, _plain, _sources

NAME = "apart_alias_names"


def _name(source: exp.Expression) -> str:
    return _lower(source.alias_or_name)


def _renamable(source: exp.Expression) -> bool:
    """Whether the name of ``source`` is one this rule can change: an alias, or a plain table's own name."""

    if isinstance(source, exp.Table):
        return bool(source.alias) and isinstance(source.this, exp.Identifier) or _plain(source)
    return isinstance(source, exp.Subquery) and bool(source.alias)


def name_aliases_apart(tree: exp.Expression) -> tuple[exp.Expression, bool]:
    """``tree`` with a new name for every source whose name is also the name of a source inside a derived table of
    its select, or of a source of a select that has its own select inside a derived table; whether it changed."""

    names: dict[str, int] = {}
    for source in tree.find_all(exp.Table, exp.Subquery):
        if isinstance(source.parent, (exp.From, exp.Join)) and _name(source):
            names[_name(source)] = names.get(_name(source), 0) + 1
    if not any(count > 1 for count in names.values()):
        return tree, False
    result = tree.copy()
    return (result, True) if _rename(result) else (tree, False)


def _rename(tree: exp.Expression) -> bool:
    selects = list(tree.find_all(exp.Select))
    sources = {id(select): _sources(select) for select in selects}
    if any(found is None for found in sources.values()):
        return False
    owner: dict[int, exp.Select] = {}
    named: list[exp.Expression] = []
    for select in selects:
        own = [source for source in sources[id(select)] if _name(source)]
        if len({_name(source) for source in own}) != len(own):
            return False
        for source in own:
            owner[id(source)] = select
            named.append(source)
    targets = [
        source
        for source in named
        if _renamable(source)
        and any(other is not source and _name(other) == _name(source) and _derived(owner[id(source)], owner[id(other)]) for other in named)
    ]
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
            source = next((s for s in sources[id(select)] if _name(s) == name), None)
            node = select
        bound.setdefault(id(source), []).append(column)
    taken = {_lower(node.name) for node in tree.find_all(exp.Identifier)}
    number = 0
    for source in targets:
        base = _name(source)
        number += 1
        while f"{base}_{number}" in taken:
            number += 1
        alias = f"{base}_{number}"
        taken.add(alias)
        for column in bound.get(id(source), []):
            column.set("table", exp.to_identifier(alias))
        current = source.args.get("alias")
        if isinstance(current, exp.TableAlias) and current.this is not None:
            current.set("this", exp.to_identifier(alias))  # a column list on the alias stays as it is
        else:
            source.set("alias", exp.TableAlias(this=exp.to_identifier(alias)))
    return True
