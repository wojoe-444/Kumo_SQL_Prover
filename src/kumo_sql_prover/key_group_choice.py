"""Group by one declared key of a table as by any other: the grouping is the table's rows either way.

    SELECT p.name, SUM(i.paid) FROM product AS p LEFT JOIN invoice AS i ON p.id = i.product_id GROUP BY p.id
    SELECT p.name, SUM(i.paid) FROM product AS p LEFT JOIN invoice AS i ON p.id = i.product_id GROUP BY p.name

When ``id`` and ``name`` are both declared keys of ``product`` and both NOT NULL, two rows of the join agree on ``id``
exactly when they come from the same row of ``product`` (or are the unmatched rows of an outer join where ``product``
is the padded side: both columns are NULL there), and the same holds for ``name``. So the groups are the same
whichever of the two keys is named. A select item that is not aggregated is a column of that one row of ``product``
and does not depend on the choice. The rule names the key that sorts first, so that both spellings meet.

**What is read.** A select whose ``GROUP BY`` is one plain column of a base table in its ``FROM`` or ``JOIN``
sources, that column being a declared single-column key with NOT NULL, and another single-column key of the same
table that is NOT NULL and sorts before it.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key, _unwrap
from .ast_utils import select_sources


def choose_key_groups(
    tree: exp.Expression,
    schema: dict[str, list[str]] | None,
    keys: dict[str, list[tuple[str, ...]]] | None,
    not_null: dict[str, frozenset[str]] | None,
) -> exp.Expression:
    """Rewrite, in place, each ``GROUP BY`` described in the module docstring."""

    if not schema or not keys or not any(True for _ in tree.find_all(exp.Group)):
        return tree
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    for select in tree.find_all(exp.Select):
        _choose(select, known, keys, not_null or {})
    return tree


def _choose(select: exp.Select, known, keys, not_null) -> None:
    group = select.args.get("group")
    if group is None or len(group.expressions) != 1 or any(value for key, value in group.args.items() if key != "expressions"):
        return
    column = _unwrap(group.expressions[0])
    if not isinstance(column, exp.Column) or isinstance(column.this, exp.Star) or column.args.get("db") or column.args.get("catalog"):
        return
    sources = select_sources(select)
    owners: dict[str, tuple[exp.Table, list[str]]] = {}
    for source in sources:
        columns = _base_table(source, known)
        alias = source.alias_or_name.lower() if columns is not None else ""
        if columns is None or not alias or alias in owners:
            return
        owners[alias] = (source, columns)
    name = column.name.lower()
    if column.table:
        holder = owners.get(column.table.lower())
        if holder is None or name not in holder[1]:
            return
    else:
        holders = [h for h in owners.values() if name in h[1]]
        if len(holders) != 1:
            return
        holder = holders[0]
    table = _table_key(holder[0])
    declared = {c.lower() for c in not_null.get(table, ())}
    singles = sorted(k[0].lower() for k in keys.get(table, []) if len(k) == 1 and k[0].lower() in declared)
    if name not in singles or singles[0] == name:
        return
    choice = exp.column(singles[0], table=holder[0].alias_or_name)
    # an output that is the grouped column keeps its name
    group.set("expressions", [choice])
