"""An outer join from a derived table of referencing columns to the table they reference is an inner join.

    SELECT t.m, p.name, COUNT(*)
    FROM (SELECT gold AS m FROM rounds UNION ALL SELECT silver FROM rounds) AS t
    LEFT JOIN members AS p ON t.m = p.member_id                       -- JOIN members AS p ON ..
    GROUP BY t.m, p.name

when ``rounds.gold`` and ``rounds.silver`` are declared NOT NULL and each is declared to reference
``members(member_id)``. ``join_rewrites`` reads this for a child that is a table; a list of winners unpivoted with
``UNION ALL`` is a derived table, and its outer join stayed outer, apart from the inner join the other query wrote.

**Why it holds.** Every row of the derived table has, in the joined column, a value that one of the referencing
columns holds in some row of its table: each operand of the union outputs such a column of its one table as it is,
and ``WHERE``, ``DISTINCT`` and ``UNION`` only drop rows. The column is NOT NULL, so the value is not NULL, and the
foreign key says a row of the referenced table has it. So every row of the derived table finds a match, the outer
join pads no row, and it returns what the inner join returns. Nothing is asked of the referenced column but the
declared foreign key (whether it is unique does not matter here: both joins repeat a row the same way).

**What is read.** In a select with no ``WITH`` left to inline, a ``LEFT JOIN`` whose right side is a table and whose
``ON`` is one equality (or ``USING`` one name) between a column of that table and a column of a derived table that
comes before it and is not itself padded by an outer join; and ``parent RIGHT JOIN derived`` when those are the
select's two sources. The derived table is a ``SELECT``, or a tree of ``UNION`` and ``UNION ALL`` over selects, in
which every select has one source, a table, no join, ``GROUP BY``, ``HAVING``, window or ``LIMIT``, no star, and
outputs at the joined position a plain column of its table that is declared NOT NULL with a declared single-column
foreign key to the joined column of the parent. The join is then written as an inner join. Everything else is
left as written.

A pre-pass of the thirteenth hardest-cases batch (``shape_fallback.THIRTEENTH_BATCH``). It needs declared foreign
keys, so it fires in the attempts that are given them. ``tests/test_referencing_union_joins.py`` replays it on DuckDB.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY
from .join_rewrites import _lookup, _lower, _same_table, _table_name

_LEAF_REFUSED = ("joins", "group", "having", "qualify", "windows", "limit", "offset", "order", "with_", "with", "laterals", "connect", "match", "into", "locks", "sample", "pivots", "cluster", "distribute", "sort", "kind", "hint", "operation_modifiers", "exclude", "prewhere", "settings", "format", "options", "for_")


def _leaves(query: exp.Expression) -> list[exp.Select] | None:
    """The selects of a tree of ``UNION`` / ``UNION ALL``, left to right, or None when it is anything else."""

    if isinstance(query, exp.Subquery):
        return None if any(v for key, v in query.args.items() if key != "this") else _leaves(query.this)
    if type(query) is exp.Union:
        if any(v for key, v in query.args.items() if key not in ("this", "expression", "distinct")):
            return None
        left, right = _leaves(query.this), _leaves(query.expression)
        return None if left is None or right is None else left + right
    return [query] if type(query) is exp.Select else None


def _references(leaf: exp.Select, position: int, parent: str, parent_column: str, not_null: dict, foreign_keys: dict) -> bool:
    """Whether ``leaf`` outputs at ``position`` a NOT NULL column of its one table that references ``parent(parent_column)``."""

    source = leaf.args.get(FROM_KEY)
    if source is None or type(source.this) is not exp.Table or any(leaf.args.get(key) for key in _LEAF_REFUSED):
        return False
    table = source.this
    if table.args.get("db") or table.args.get("catalog"):
        return False
    if any(isinstance(item, exp.Star) or item.is_star for item in leaf.expressions) or position >= len(leaf.expressions):
        return False
    column = leaf.expressions[position].unalias()
    if not isinstance(column, exp.Column) or not isinstance(column.this, exp.Identifier) or column.args.get("db") or column.args.get("catalog"):
        return False
    if column.table and column.table.lower() != (table.alias_or_name or "").lower():
        return False
    name, child = column.name.lower(), _table_name(table)
    if name not in {c.lower() for c in (_lookup(not_null, child) or ())}:
        return False
    return any(
        tuple(c.lower() for c in columns) == (name,) and _same_table(parent, referenced) and tuple(c.lower() for c in referenced_columns) == (parent_column,)
        for columns, referenced, referenced_columns in (_lookup(foreign_keys, child) or [])
    )


def _output_names(leaves: list[exp.Select]) -> list[str] | None:
    first = leaves[0]
    if any(isinstance(item, exp.Star) or item.is_star for item in first.expressions):
        return None
    names = [(item.alias_or_name or "").lower() for item in first.expressions]
    if any(len(leaf.expressions) != len(names) for leaf in leaves):
        return None
    return names


def _sides(join: exp.Join, derived_alias: str, derived_names: list[str], parent: exp.Table, parent_columns: set[str], two_sources: bool) -> tuple[str, str] | None:
    """``(derived column, parent column)`` when the join condition is one equality between the two, else None."""

    using = join.args.get("using")
    if using:
        if join.args.get("on") is not None or len(using) != 1:
            return None
        name = using[0].name.lower()
        return (name, name) if derived_names.count(name) == 1 and name in parent_columns else None
    on = join.args.get("on")
    while isinstance(on, exp.Paren):
        on = on.this
    if type(on) is not exp.EQ:
        return None
    parent_alias = (parent.alias_or_name or "").lower()
    found: dict[str, str] = {}
    for operand in (on.this, on.expression):
        while isinstance(operand, exp.Paren):
            operand = operand.this
        if not isinstance(operand, exp.Column) or not isinstance(operand.this, exp.Identifier) or operand.args.get("db") or operand.args.get("catalog"):
            return None
        name, table = operand.name.lower(), operand.table.lower()
        if table:
            side = "derived" if table == derived_alias else "parent" if table == parent_alias else None
        elif not two_sources:
            side = None  # a bare name could belong to another source
        else:
            in_derived, in_parent = name in derived_names, name in parent_columns
            side = "derived" if in_derived and not in_parent else "parent" if in_parent and not in_derived else None
        if side is None or side in found:
            return None
        found[side] = name
    if derived_names.count(found["derived"]) != 1 or found["parent"] not in parent_columns:
        return None
    return found["derived"], found["parent"]


def _padded_before(joins: list[exp.Join], upto: int) -> bool:
    """Whether a join before position ``upto`` can pad the sources to its left (``RIGHT`` or ``FULL``)."""

    return any((join.args.get("side") or "").upper() in ("RIGHT", "FULL") for join in joins[:upto])


def _candidates(select: exp.Select, schema: dict, not_null: dict, foreign_keys: dict):
    """The joins of ``select`` that the module reads as inner joins."""

    source = select.args.get(FROM_KEY)
    joins = select.args.get("joins") or []
    if source is None or not joins:
        return
    sources = [(source.this, None)] + [(join.this, join) for join in joins]
    for index, join in enumerate(joins):
        side = (join.args.get("side") or "").upper()
        if join.args.get("method") or (join.args.get("kind") or "").upper() not in ("", "OUTER"):
            continue
        if side == "LEFT":
            parent = join.this
            children = [(s, j) for s, j in sources[: index + 1] if isinstance(s, exp.Subquery)]
            if _padded_before(joins, index):
                continue
        elif side == "RIGHT" and len(joins) == 1:
            parent, children = source.this, [(join.this, None)] if isinstance(join.this, exp.Subquery) else []
        else:
            continue
        if type(parent) is not exp.Table or parent.args.get("db") or parent.args.get("catalog"):
            continue
        parent_name = _table_name(parent)
        parent_columns = {c.lower() for c in (_lookup(schema, parent_name) or ())}
        if not parent_columns:
            continue
        for child, child_join in children:
            if child_join is not None and (child_join.args.get("side") or "").upper():
                continue  # a padded row of the derived table has NULL in the joined column and no match
            alias = child.args.get("alias")
            if not isinstance(alias, exp.TableAlias) or not child.alias or alias.args.get("columns") or any(v for key, v in child.args.items() if key not in ("this", "alias")):
                continue
            leaves = _leaves(child.this)
            names = _output_names(leaves) if leaves else None
            if not names:
                continue
            sides = _sides(join, child.alias.lower(), names, parent, parent_columns, len(sources) == 2)
            if sides is None:
                continue
            position = names.index(sides[0])
            if all(_references(leaf, position, parent_name, sides[1], not_null, foreign_keys) for leaf in leaves):
                yield join
                break


def inner_referencing_joins(tree: exp.Expression, schema: dict | None, not_null: dict | None, foreign_keys: dict | None) -> tuple[exp.Expression, bool]:
    """``tree`` with each such outer join written as an inner join (module doc); whether anything changed."""

    if not schema or not not_null or not foreign_keys or tree.find(exp.With) is not None:
        return tree, False
    if not any(isinstance(node.this, exp.Subquery) for node in tree.find_all(exp.From, exp.Join)):
        return tree, False
    schema, not_null, foreign_keys = _lower(schema), _lower(not_null), _lower(foreign_keys)
    if not any(True for select in tree.find_all(exp.Select) for _ in _candidates(select, schema, not_null, foreign_keys)):
        return tree, False
    result = tree.copy()
    for select in list(result.find_all(exp.Select)):
        for join in list(_candidates(select, schema, not_null, foreign_keys)):
            join.set("side", None)
            join.set("kind", None)
    return result, True
