"""Bare column names around a value looked up by key are given their tables.

    SELECT buyer FROM sale WHERE item_id = (SELECT item_id FROM item WHERE label = 'S')

With ``label`` a declared key of ``item`` the subquery finds at most one row, and ``keyed_lookups`` reads the test
as ``EXISTS (SELECT 1 FROM item WHERE item.label = 'S' AND sale.item_id = item.item_id)``. It reads only columns
that name their table, because it moves the outer operand inside the subquery, where a bare ``item_id`` would mean
the looked-up table's column. This pass writes the names in, and changes nothing else:

    SELECT buyer FROM sale WHERE sale.item_id = (SELECT item.item_id FROM item WHERE item.label = 'S')

**Why it is the same.** SQL looks a bare name up in the nearest query that has a source with a column of that
name. The subquery has one source, a base table whose columns the caller supplied, so a bare name that table has
is that table's column. The other operand stands in the ``WHERE`` of a select whose sources are all base tables
with supplied columns, under different names; a bare name that exactly one of them has is that source's column.
Nothing is renamed and no row is touched.

**What is read.** A comparison (``= <> < <= > >=``) reached from the top of a ``WHERE`` through ``AND``, ``OR`` and
parentheses, between a scalar subquery and an operand without aggregate, window or nested query. The subquery
is one select over one base table that has a declared key: one output, a ``WHERE``, no nested query and nothing
else. The names are written only when afterwards every column on both sides names a table; a name no source has
(a read of a query further out), a name two sources have, a table that a ``WITH`` table of the statement
shadows, or a source that is not a base table leaves the test as it was.

A narrow retry rule (``shape_fallback``): ``tests/test_lookup_names.py``.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key
from .ast_utils import select_sources
from .keyed_lookups import _ALLOWED, _COMPARISONS, _OPAQUE, _tests, _unwrap


def name_lookup_columns(tree: exp.Expression, schema: dict[str, list[str]] | None, keys: dict[str, list[tuple[str, ...]]] | None) -> tuple[exp.Expression, bool]:
    """``tree`` with the names written in, in place, and whether anything was named."""

    if not schema or not keys or not any(True for _ in tree.find_all(exp.Subquery)):
        return tree, False
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    keyed = {key.lower() for key, declared in keys.items() if any(declared)}
    shadowed = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    named = False
    for select in list(tree.find_all(exp.Select)):
        where = select.args.get("where")
        if where is None:
            continue
        outer = select_sources(select)
        outer_columns = [_base_table(source, known) for source in outer]
        outer_names = [(source.alias_or_name or "").lower() for source in outer]
        if not outer or any(columns is None for columns in outer_columns) or len(set(outer_names)) != len(outer_names) or not all(outer_names):
            continue
        if any(source.name.lower() in shadowed for source in outer):
            continue
        for test in _tests(where.this):
            for mine, other in ((test.this, test.expression), (test.expression, test.this)):
                subquery = _unwrap(mine)
                if not isinstance(subquery, exp.Subquery) or any(v for k, v in subquery.args.items() if k not in ("this", "comments")):
                    continue
                query = subquery.this
                if not isinstance(query, exp.Select) or any(v for k, v in query.args.items() if k not in _ALLOWED):
                    continue
                sources = select_sources(query)
                if len(sources) != 1 or len(query.expressions) != 1 or query.args.get("where") is None:
                    continue
                table = sources[0]
                columns = _base_table(table, known)
                inner = (table.alias_or_name or "").lower()
                if columns is None or not inner or _table_key(table) not in keyed or table.name.lower() in shadowed:
                    continue
                if any(isinstance(node, _OPAQUE) for part in (query.expressions[0], query.args["where"].this, other) for node in part.walk()):
                    continue
                inside = [column for column in query.find_all(exp.Column) if not column.table]
                outside = [column for column in other.find_all(exp.Column) if not column.table]
                if not inside and not outside:
                    continue
                if any(column.name.lower() not in columns or column.this.args.get("quoted") for column in inside):
                    continue  # a read of a row further out
                owners = []
                for column in outside:
                    holders = [name for name, theirs in zip(outer_names, outer_columns) if column.name.lower() in theirs]
                    owners.append(holders[0] if len(holders) == 1 and not column.this.args.get("quoted") else None)
                if None in owners:
                    continue
                for column in inside:
                    column.set("table", exp.to_identifier(table.alias_or_name))
                by_name = dict(zip(outer_names, outer))
                for column, owner in zip(outside, owners):
                    column.set("table", exp.to_identifier(by_name[owner].alias_or_name))
                named = True
                break
    return tree, named
