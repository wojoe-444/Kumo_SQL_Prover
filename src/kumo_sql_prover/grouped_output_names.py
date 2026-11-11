"""A ``GROUP BY`` key that names an output of its select is that output's expression.

    SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY big        SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY a > 10

PostgreSQL, BigQuery, DuckDB, MySQL and SQLite accept an output name as a ``GROUP BY`` key. Where the name is also
a column of a ``FROM`` source they disagree with one another on which wins (PostgreSQL takes the input column),
so the key is read as the output only when it can be nothing else: no source of the select has a column of that
name, and the select is not inside an expression subquery, where the name could be a column of the enclosing
query. Then a dialect that accepts the query at all groups by the output's expression, and the two spellings
are the same query: same groups, same rows, same duplicates, the same single NULL group.

The prover's rules read ``GROUP BY`` keys as expressions over the sources and compare grouped derived tables by
their text, so the name is replaced by the expression before any of them runs.

Conditions, all checked:

* the key is a bare, unqualified column directly in a plain ``GROUP BY`` list (not inside ``ROLLUP``, ``CUBE`` or
  ``GROUPING SETS``, not part of a larger expression: only a whole key can name an output);
* the select names exactly one output so (compared without case), written ``<expression> AS name``; the
  expression holds no aggregate, window function or subquery, is not a star, and reads no column whose name is
  an output name of the select (so it means the same in the ``GROUP BY`` as in the select list);
* every source of the select is a table the schema lists the columns of (no derived table, ``UNNEST``, ``LATERAL``
  or table function, no alias column list), none of them has a column of that name, and the query has no ``WITH``
  (a ``WITH`` table could carry the name of a schema table and other columns);
* from the select up to the query's root there are only derived tables in ``FROM`` or ``JOIN`` and set operations:
  no scalar, ``EXISTS``, ``IN`` or quantified subquery and no ``LATERAL``, whose selects can read an outer column.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, extended_grouping

_NESTED = (exp.AggFunc, exp.Window, exp.Subquery, exp.Select, exp.Exists, exp.Star)


def _source_columns(select: exp.Select, schema: dict) -> set[str] | None:
    """The lower-cased column names of the select's sources, or None when one is not a table the schema lists."""

    from_ = select.args.get(FROM_KEY)
    if from_ is None or select.args.get("laterals"):
        return None
    lowered = {str(k).lower(): v for k, v in schema.items()}
    names: set[str] = set()
    for source in [from_.this] + [join.this for join in select.args.get("joins") or []]:
        if not isinstance(source, exp.Table) or not isinstance(source.this, exp.Identifier):
            return None
        alias = source.args.get("alias")
        if isinstance(alias, exp.TableAlias) and alias.args.get("columns"):
            return None
        if any(source.args.get(k) for k in ("pivots", "joins", "laterals")):
            return None
        columns = lowered.get(".".join(part.name for part in source.parts).lower())
        if columns is None:
            columns = lowered.get(source.name.lower())
        if not columns:
            return None
        names |= {str(column).lower() for column in columns}
    return names


def _reads_no_outer_row(select: exp.Select) -> bool:
    """Whether only derived tables and set operations lie between ``select`` and the root (module doc)."""

    node: exp.Expression = select
    while node.parent is not None:
        parent = node.parent
        if isinstance(parent, exp.SetOperation):
            pass
        elif isinstance(parent, exp.Subquery):
            if not isinstance(node, (exp.Select, exp.SetOperation, exp.Subquery)):
                return False
        elif isinstance(parent, (exp.From, exp.Join)):
            if not isinstance(node, exp.Subquery) or parent.args.get("this") is not node:
                return False
        elif isinstance(parent, exp.Select):
            if not isinstance(node, (exp.From, exp.Join)):
                return False
        else:
            return False
        node = parent
    return True


def _resolved(select: exp.Select, schema: dict) -> list[tuple[int, exp.Expression]]:
    group = select.args.get("group")
    if group is None or not group.expressions or extended_grouping(group) or any(group.args.get(k) for k in ("all", "totals")):
        return []
    outputs: dict[str, list[exp.Expression]] = {}
    for item in select.expressions:
        name = (item.alias_or_name or "").lower()
        if name:
            outputs.setdefault(name, []).append(item)
    found = []
    source_names = None
    for index, key in enumerate(group.expressions):
        if not isinstance(key, exp.Column) or key.table or not isinstance(key.this, exp.Identifier):
            continue
        items = outputs.get(key.name.lower())
        if not items or len(items) != 1 or not isinstance(items[0], exp.Alias):
            continue
        value = items[0].this
        if isinstance(value, _NESTED) or value.find(*_NESTED):
            continue
        if any(column.name.lower() in outputs for column in value.find_all(exp.Column)):
            continue
        if source_names is None:
            source_names = _source_columns(select, schema)
            if source_names is None or not _reads_no_outer_row(select):
                return []
        if key.name.lower() in source_names:
            continue
        found.append((index, value))
    return found


def group_by_named_outputs(tree: exp.Expression, schema: dict | None) -> exp.Expression:
    """``tree`` with each ``GROUP BY`` key that can only name an output of its select written as that output's
    expression (module doc)."""

    if not schema or tree.find(exp.With) is not None:
        return tree
    for select in list(tree.find_all(exp.Select)):
        for index, value in _resolved(select, schema):
            select.args["group"].expressions[index].replace(value.copy())
    return tree
