"""Membership of a foreign-key column in its parent's key, read as "the column is not NULL".

    SELECT c.customer_id FROM customer AS c WHERE c.product_key IN (SELECT product_key FROM product)

keeps the customers whose product exists. When ``customer(product_key)`` is declared to reference
``product(product_key)``, every product key that is not NULL exists, so the query keeps exactly

    SELECT c.customer_id FROM customer AS c WHERE c.product_key IS NOT NULL

**Why the two are the same.** A declared foreign key says: for every row of the child table whose column ``f`` is
not NULL, some row of the parent table has ``pk = f``. (A NULL ``f`` references nothing; this is the reading the
prover uses everywhere, and it is the weaker one, so a proof made with it also holds when ``f`` is never NULL.)

* The left operand is the column ``f`` of a source of the select that is the child table itself. Its value on
  any row the ``WHERE`` sees is a stored ``f``, or NULL when the source is the padded side of an outer join.
* The subquery returns ``pk`` of every parent row: it has no ``WHERE``, join, grouping or row limit
  (``DISTINCT`` and an ``ORDER BY`` change nothing a membership test can see).
* So when ``f`` is not NULL the test is TRUE: the foreign key names a parent row with that ``pk``. When ``f`` is
  NULL the test is NULL or FALSE, never TRUE: NULL equals nothing.
* The test is rewritten only as a top-level ``AND`` part of a ``WHERE``, where TRUE keeps the row and anything
  else drops it, exactly as ``f IS NOT NULL`` does.

The parent column need not be a key and need not be NOT NULL for this: only the existence of a matching row is
used. When ``f`` is also declared NOT NULL, the normalizer drops the remaining test.

**What is read.** ``x IN (SELECT [DISTINCT] pk FROM parent)`` (or ``SELECT *`` when the parent has that one
column), ``x`` being a column qualified by the name of a source that is the child base table, or a bare column
that exactly one source has when every source is a base table with supplied columns; a single-column foreign
key declared from that column to ``parent(pk)``.

**What is refused.** ``NOT IN`` and any place where FALSE and NULL differ; a subquery with a ``WHERE``, a join,
``GROUP BY``, ``HAVING``, ``LIMIT`` or an expression around the column; a left operand that is an expression, a
column of a derived table, or a column of an outer query; a foreign key of several columns.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _membership_blind, _positive_where_part, _table_key, _unwrap
from .ast_utils import select_sources
from .fk_rules import _same_table


def read_fk_membership(tree: exp.Expression, schema: dict[str, list[str]] | None, foreign_keys: dict[str, list[tuple]] | None) -> exp.Expression:
    """Rewrite, in place, each membership test described in the module docstring."""

    if not schema or not foreign_keys:
        return tree
    known = {name.lower(): [c.lower() for c in columns] for name, columns in schema.items()}
    declared: list[tuple[str, str, str, str]] = []  # child table, column, parent table, column
    for child, entries in foreign_keys.items():
        for columns, parent, parent_columns in entries:
            if len(columns) == 1 and len(parent_columns) == 1:
                declared.append((child.lower(), columns[0].lower(), parent.lower(), parent_columns[0].lower()))
    if not declared:
        return tree
    for node in list(tree.find_all(exp.In)):
        select = _positive_where_part(node)
        if select is None or node.expressions or node.args.get("unnest") is not None or node.args.get("field") is not None:
            continue
        child = _child_column(node.this, select, known)
        parent = _parent_column(node.args.get("query"), known)
        if child is None or parent is None:
            continue
        if any(_same_table(child[0], c) and child[1] == f and _same_table(parent[0], p) and parent[1] == pk for c, f, p, pk in declared):
            node.replace(exp.Not(this=exp.Is(this=_unwrap(node.this).copy(), expression=exp.Null())))
    return tree


def _child_column(operand: exp.Expression, select: exp.Select, known: dict[str, list[str]]) -> tuple[str, str] | None:
    """``(table, column)`` when the operand is a column of a base-table source of ``select``."""

    column = _unwrap(operand)
    if not isinstance(column, exp.Column) or isinstance(column.this, exp.Star) or column.args.get("db") or column.args.get("catalog"):
        return None
    name = column.name.lower()
    sources = select_sources(select)
    names = [source.alias_or_name.lower() for source in sources]
    if "" in names or len(set(names)) != len(names):
        return None
    if column.table:
        owners = [source for source in sources if source.alias_or_name.lower() == column.table.lower()]
    else:
        if any(_base_table(source, known) is None for source in sources):
            return None
        owners = [source for source in sources if name in _base_table(source, known)]
    if len(owners) != 1:
        return None
    columns = _base_table(owners[0], known)
    if columns is None or name not in columns:
        return None
    return _table_key(owners[0]), name


def _parent_column(query: exp.Expression | None, known: dict[str, list[str]]) -> tuple[str, str] | None:
    """``(table, column)`` when the subquery lists one column of every row of one base table."""

    if not isinstance(query, exp.Subquery) or any(value for key, value in query.args.items() if key not in ("this", "comments")):
        return None
    inner = query.this
    if not isinstance(inner, exp.Select) or not _membership_blind(inner):
        return None
    if any(inner.args.get(key) for key in ("where", "group", "kind")) or len(inner.expressions) != 1:
        return None
    sources = select_sources(inner)
    if len(sources) != 1:
        return None
    columns = _base_table(sources[0], known)
    if columns is None or not sources[0].alias_or_name:
        return None
    item = inner.expressions[0]
    output = _unwrap(item.this if isinstance(item, exp.Alias) else item)
    if isinstance(output, exp.Star) and not any(output.args.values()):
        return (_table_key(sources[0]), columns[0]) if len(columns) == 1 else None
    if not isinstance(output, exp.Column) or isinstance(output.this, exp.Star) or output.args.get("db") or output.args.get("catalog"):
        return None
    if output.table and output.table.lower() != sources[0].alias_or_name.lower():
        return None
    if output.name.lower() not in columns:
        return None
    return _table_key(sources[0]), output.name.lower()
