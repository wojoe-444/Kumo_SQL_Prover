"""The output name a select keeps when the ``UNION ALL`` it reads is renamed.

``algebraic_equivalence._canonicalize_union_source`` renames the output columns of a derived ``UNION ALL`` to
``c0, c1, ..`` and, with them, the enclosing select's reads of those columns. A read that is a select item by
itself

    SELECT id, SUM(n) AS n FROM (SELECT a AS id, 1 AS n FROM t UNION ALL SELECT b, 1 FROM t) AS u GROUP BY id

became ``SELECT c1, SUM(c0) AS n ..``: the select's own first output was now called ``c1``. When that select is
itself a derived table, the query around it still reads ``id``

    SELECT p.name, d.n FROM (..) AS d LEFT JOIN players AS p ON d.id = p.player_id

and the name was gone: the prover declined the query ("d.id is not a column of d"), which is safe and loses the
pair to the renaming alone (hardest-cases batch 30, LeetCode 1783).

``keep`` gives such an item its name back, ``c1 AS id``. An unaliased column's output name is the column's name
in every dialect, so the select outputs what it did before the renaming. It is done only when a column outside
the select could be reading the name (a bare column of that name, or one under the alias the select is read by),
so every other text stays as it was.
"""

from __future__ import annotations

from sqlglot import exp

NAME = "kept_union_names"


def keep(outer: exp.Select, renamed: list[tuple[exp.Column, exp.Identifier]]) -> None:
    """Alias each renamed read (``(column as renamed, its old identifier)``) that is a select item of ``outer`` by
    itself with the name it had, when a column outside ``outer`` could read that name. Changes ``outer`` in place."""

    if outer.parent is None or not renamed:
        return
    reader = outer.parent.alias.lower() if isinstance(outer.parent, exp.Subquery) and outer.parent.alias else None
    inside = {id(column) for column in outer.find_all(exp.Column)}
    outside = {
        column.name.lower()
        for column in outer.root().find_all(exp.Column)
        if id(column) not in inside and (reader is None or not column.table or column.table.lower() == reader)
    }
    taken = {item.alias.lower() for item in outer.expressions if isinstance(item, exp.Alias)}
    for column, old in renamed:
        name = old.name.lower()
        if column.parent is not outer or column.arg_key != "expressions" or name not in outside or name in taken:
            continue
        column.replace(exp.Alias(this=column.copy(), alias=old.copy()))
        taken.add(name)
