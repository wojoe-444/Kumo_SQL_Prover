"""``COUNT(t.c)`` of a NOT NULL column of a table that no outer join pads is ``COUNT(*)``.

    SELECT pt.name, COUNT(p.id) FROM p JOIN pt ON p.kind = pt.id LEFT JOIN c ON p.id = c.post GROUP BY pt.name

counts, in each group, the joined rows on which ``p.id`` is not NULL. Every joined row holds a row of ``p`` (``p`` is
the ``FROM`` item, or inner joined, and no ``RIGHT`` or ``FULL`` join follows it, so no join pads its columns), and
``p.id`` is declared NOT NULL, so it is NULL on no joined row and the count is the number of rows in the group:
``COUNT(*)``. A group is the same set of rows either way, duplicates are counted by both, and over no rows both
are 0 (or there is no group).

Without an outer join the prover already reads the two alike. Over an outer join it reads the aggregate through a
derived table of the joined columns (``algebraic_equivalence._wrap_outer_join_aggregate``), in which the column is
an output like any other and its NOT NULL declaration is out of sight, so the rewrite is made before that.
Calcite prints ``COUNT(*)`` for a count of such a column.

Rewritten only when:

* the call is a plain ``COUNT(t.c)`` of this select: one column argument, no ``DISTINCT``, no ``FILTER``, not a
  window function;
* the select has at least one ``LEFT``, ``RIGHT`` or ``FULL`` join (the case the prover did not read);
* ``t`` names exactly one source of the select, a base table without a column list, schema or catalog, and the
  constraints declare ``c`` NOT NULL for it;
* that source cannot be padded: it is not the right side of a ``LEFT`` join or either side of a ``FULL`` join, and
  no ``RIGHT`` or ``FULL`` join comes after it (``algebraic_equivalence._null_extended``).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import select_sources

_OUTER = ("LEFT", "RIGHT", "FULL")


def count_preserved_not_null(select: exp.Select, not_null: dict[str, frozenset[str]] | None, null_extended) -> exp.Expression | None:
    """``select`` with such counts written ``COUNT(*)``, or None. ``null_extended(select, source)`` says whether a join can pad ``source``."""

    if not not_null:
        return None
    joins = select.args.get("joins") or []
    if not any((j.args.get("side") or "").upper() in _OUTER or (j.args.get("kind") or "").upper() in _OUTER for j in joins):
        return None
    declared = {table.lower(): {column.lower() for column in columns} for table, columns in not_null.items()}
    sources = select_sources(select)
    aliases = [(source.alias_or_name or "").lower() for source in sources]
    targets = []
    for count in select.find_all(exp.Count):
        if count.find_ancestor(exp.Select) is not select or isinstance(count.parent, (exp.Window, exp.Filter, exp.IgnoreNulls, exp.RespectNulls)):
            continue
        column = count.this
        if type(column) is not exp.Column or not isinstance(column.this, exp.Identifier) or count.expressions or column.args.get("db"):
            continue
        qualifier = column.table.lower()
        if not qualifier or aliases.count(qualifier) != 1:
            continue
        source = sources[aliases.index(qualifier)]
        if not isinstance(source, exp.Table) or source.db or source.catalog or (source.args.get("alias") is not None and source.args["alias"].args.get("columns")):
            continue
        if column.name.lower() not in declared.get(source.name.lower(), ()) or null_extended(select, source):
            continue
        targets.append(count)
    if not targets:
        return None
    copy = select.copy()
    by_position = {id(old): new for old, new in zip(select.walk(), copy.walk())}
    for count in targets:
        by_position[id(count)].set("this", exp.Star())
    return copy
