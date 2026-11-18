"""A key of a table is still a key after a join with a grouped table on all its grouping columns.

    SELECT s.id, s.name, CASE WHEN g.top > 100 THEN s.pay * 0.5 ELSE s.pay END
    FROM staff AS s JOIN (SELECT team, MAX(pay) AS top FROM staff GROUP BY team) AS g ON s.team = g.team
    GROUP BY s.id

returns the rows of the same query without its ``GROUP BY`` when ``id`` is a declared NOT NULL key of ``staff``:
every group is one row. The same holds for a ``SELECT DISTINCT`` that outputs ``s.id``.

**Why it is sound.** The grouped table ``g`` has one row for each combination of its grouping columns (NULLs
form one group). The join condition equates every grouping column of ``g`` with a column of ``s``, so the rows
of ``g`` that match one row of ``s`` all have that row's values in every grouping column: at most one row. So
the join returns each row of ``s`` at most once (exactly once for a ``LEFT JOIN`` that keeps ``s``), a ``WHERE``
only removes rows, and no two rows of the result agree on a NOT NULL key of ``s``.

* ``GROUP BY`` that lists such a key (among other columns) therefore has one row in every group. With no
  aggregate and no window in the select, each output is that row's own value (MySQL's reading of a column that
  is not grouped is "a row of the group": there is one), so the grouping is dropped.
* ``DISTINCT`` over outputs that include such a key removes nothing, and is dropped.

**What is read.** A select with exactly two sources and one join between them: a plain base table ``s`` with
supplied columns, and a derived table that is one ``GROUP BY`` of plain columns of a base table, every grouping
column output as a plain column, with no ``WHERE``, ``HAVING``, ``DISTINCT``, ``LIMIT``, window or subquery
(the reading of :mod:`kumo_sql_prover.own_group_aggregates`). For every grouping column ``k`` the ``ON`` condition
has ``s.c = g.k`` as one of its ``AND`` parts (for an inner or comma join, the ``WHERE`` counts too); other
parts only remove rows. The join is inner, or an outer join that keeps the rows of ``s`` (``s LEFT JOIN g``,
``g RIGHT JOIN s``); an outer join that keeps the rows of ``g`` is refused (its padded rows all have a NULL
key). The key is declared, all its columns are declared NOT NULL, and each is listed as ``s.column`` with the
qualifier written.

Refused: a select with ``HAVING`` or ``QUALIFY``, a ``GROUP BY`` together with ``DISTINCT``, ``ROLLUP`` and the
like, an aggregate or a window anywhere in the select's own scope, ``USING``, a third source.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _table_key, _unwrap
from .ast_utils import conjuncts, extended_grouping, plain_distinct, select_sources
from .own_group_aggregates import _grouped, _plain_table


def keyed_own_group_rows(
    select: exp.Select,
    schema: dict[str, list[str]] | None,
    keys: dict[str, list[tuple[str, ...]]] | None,
    not_null: dict[str, frozenset[str]] | None,
) -> exp.Expression | None:
    """``select`` (in place) without the ``GROUP BY`` or ``DISTINCT`` the module docstring describes, else ``None``."""

    if not schema or not keys or not not_null or not isinstance(select, exp.Select):
        return None
    group = select.args.get("group")
    distinct = select.args.get("distinct")
    if (group is None) == (distinct is None) or any(select.args.get(k) for k in ("having", "qualify")):
        return None
    if group is not None:
        if extended_grouping(group) or any(value for key, value in group.args.items() if key != "expressions"):
            return None
        listed = list(group.expressions)
        if any(node.find_ancestor(exp.Select) is select for node in select.find_all(exp.AggFunc, exp.Window)):
            return None
    else:
        if not plain_distinct(select):
            return None
        listed = [item.this if isinstance(item, exp.Alias) else item for item in select.expressions]
    table = _keyed_side(select, schema)
    if table is None:
        return None
    name, alias = table
    columns = set()
    for node in listed:
        node = _unwrap(node)
        if type(node) is exp.Column and not isinstance(node.this, exp.Star) and node.table.lower() == alias and not node.args.get("db"):
            columns.add(node.name.lower())
    declared = {c.lower() for c in next((v for k, v in not_null.items() if k.lower() == name), frozenset())}
    table_keys = next((v for k, v in keys.items() if k.lower() == name), [])
    if not any(key and {c.lower() for c in key} <= columns & declared for key in table_keys):
        return None
    select.set("group" if group is not None else "distinct", None)
    return select


def _keyed_side(select: exp.Select, schema: dict[str, list[str]]) -> tuple[str, str] | None:
    """``(table, alias)`` of the base table whose rows the select's one join returns at most once each."""

    joins = select.args.get("joins") or []
    sources = select_sources(select)
    if len(sources) != 2 or len(joins) != 1 or select.args.get("laterals"):
        return None
    join = joins[0]
    if join.method or join.args.get("using") or (join.kind or "").upper() not in ("", "INNER", "OUTER", "CROSS"):
        return None
    side = (join.side or "").upper()
    names = [source.alias_or_name.lower() for source in sources]
    if "" in names or names[0] == names[1]:
        return None
    # what every returned pair of rows satisfies: the ON condition, and for an inner join the WHERE too
    parts = list(conjuncts(join.args["on"])) if join.args.get("on") is not None else []
    if not side and select.args.get("where") is not None:
        parts += list(conjuncts(select.args["where"].this))
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    for index in (0, 1):
        table, derived = sources[index], sources[1 - index]
        if _plain_table(table, known) is None:
            continue
        grouped = _grouped(derived, known)
        if grouped is None:
            continue
        # the outer join, if any, keeps the rows of the base table
        if side not in ("", "LEFT" if index == 0 else "RIGHT"):
            continue
        _, outputs, group_keys = grouped
        if _equated_keys(parts, names[index], names[1 - index], outputs) >= set(group_keys):
            return _table_key(table), names[index]
    return None


def _equated_keys(parts: list[exp.Expression], s: str, g: str, outputs: dict[str, str]) -> set[str]:
    """The grouping columns of ``g`` that a condition among ``parts`` equates with a column of ``s``."""

    found = set()
    for part in parts:
        part = _unwrap(part)
        if not isinstance(part, exp.EQ):
            continue
        left, right = _unwrap(part.this), _unwrap(part.expression)
        if type(left) is not exp.Column or type(right) is not exp.Column or left.args.get("db") or right.args.get("db"):
            continue
        if left.table.lower() == g and right.table.lower() == s:
            left, right = right, left
        if left.table.lower() == s and right.table.lower() == g and right.name.lower() in outputs:
            found.add(outputs[right.name.lower()])
    return found
