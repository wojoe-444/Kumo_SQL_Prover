"""Keep a renamed read of a derived ``UNION ALL`` from colliding with a select alias.

``algebraic_equivalence._canonicalize_union_source`` renames the output columns of a derived ``UNION ALL`` to
``c0, c1, ...`` and renames the enclosing select's reads of them. The enclosing select keeps its own output
aliases, and a query written with positional aliases (every Calcite plan translated by
``tools/calcite_plan_to_sql.py`` is) can already use one of those names::

    SELECT u.k AS c0, SUM(u.p) AS c1 FROM (.. UNION ALL ..) AS u GROUP BY u.k

Once the union's ``p`` is renamed ``c0`` and its ``k`` ``c1``, the bare text reads
``SELECT c1 AS c0, SUM(c0) AS c1 .. GROUP BY c1``: ``c1`` is now both a column of the source and a select alias,
and which one ``GROUP BY c1`` means depends on the dialect. The prover rightly declines such a text, so the pair
was lost to the renaming alone.

``qualify`` gives such a read its source's alias (``u.c1``), which names the column in every dialect. It touches
only a read whose *new* name is a select alias and whose *old* name was not: a read that was already ambiguous
before the renaming stays as it was written, for the prover to decline.
"""

from __future__ import annotations

from sqlglot import exp


def select_aliases(select: exp.Select) -> set[str]:
    """The output aliases the select declares, lower case."""

    return {item.alias.lower() for item in select.expressions if isinstance(item, exp.Alias) and item.alias}


def qualify(column: exp.Column, old_name: str, source_alias: str, aliases: set[str]) -> None:
    """Qualify a bare ``column`` (already renamed) with ``source_alias`` when only its new name is a select alias."""

    if column.table or not source_alias:
        return
    if column.name.lower() in aliases and old_name.lower() not in aliases:
        column.set("table", exp.to_identifier(source_alias))
