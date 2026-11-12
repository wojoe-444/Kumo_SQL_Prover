"""A grouped ``IN`` subquery whose ``HAVING`` compares an aggregate with an uncorrelated scalar subquery.

``x IN (SELECT k FROM t GROUP BY k HAVING SUM(v) > 0.95 * (SELECT MAX(m) FROM u))``

The normalizer reads the groups of an ``IN`` subquery as a derived table and its ``HAVING`` as a filter on that
table's columns (``_grouped_in_to_derived``), which is the form the prover compares. It declines a ``HAVING``
that holds a subquery, because a subquery can read the group's rows. An *uncorrelated* scalar subquery cannot:
every column in it is bound by a source inside it (``scalar_subqueries.is_uncorrelated``), so it is one value,
the same for every group and the same in ``HAVING`` as in the ``WHERE`` above the derived table. For such a
subquery the lift is the one the normalizer already makes, with the subquery carried along as a constant:

``x IN (SELECT g.k FROM (SELECT k, SUM(v) AS a FROM t GROUP BY k) AS g WHERE g.a > 0.95 * (SELECT MAX(m) FROM u))``

How it is done: each such subquery is replaced by a placeholder call for the duration of the existing rule, and
put back afterwards wherever the placeholder ended up. A ``HAVING`` the existing rule still declines (a
correlated subquery, ``EXISTS``, a window) keeps its text exactly, because the placeholders are restored either
way.

What is left alone:

* a subquery that reads a column it does not bind itself, or whose columns cannot all be resolved (a bare
  column of a table missing from the schema could belong to the group);
* a subquery with ``LIMIT``, ``OFFSET``, a window, ``LATERAL``, ``UNNEST`` or a ``WITH`` of its own, or with
  more than one output column;
* ``IN``, ``EXISTS`` and derived-table subqueries inside the ``HAVING``: only a scalar position is one value.

If the subquery returns more than one row the engine raises an error wherever it is evaluated, before and after.
A subquery that returns no row is NULL in both places.
"""

from __future__ import annotations

from typing import Callable

from sqlglot import exp

from .scalar_subqueries import is_uncorrelated

PLACEHOLDER = "kumosql_having_scalar"
_NOT_SCALAR = (exp.From, exp.Join, exp.In, exp.Exists, exp.CTE, exp.SetOperation, exp.Subquery, exp.Table, exp.TableAlias)


def _scalar_candidates(having: exp.Expression, inner: exp.Select, schema) -> list[exp.Subquery]:
    found = []
    for node in having.find_all(exp.Subquery):
        if isinstance(node.parent, _NOT_SCALAR) or node.find_ancestor(exp.Select) is not inner:
            continue  # not in a scalar position of this HAVING itself
        select = node.this
        if not isinstance(select, exp.Select) or len(select.expressions) != 1:
            continue
        if select.args.get("limit") or select.args.get("offset") or node.args.get("alias"):
            continue
        if is_uncorrelated(node, schema):
            found.append(node)
    return found


def read_scalar_having_groups(tree: exp.Expression, schema, lift: Callable[[exp.Expression], exp.Expression]) -> exp.Expression:
    """Apply ``lift`` (the grouped-``IN`` rule) with the uncorrelated scalar subqueries of each ``HAVING`` held as constants."""

    if not schema:
        return tree  # without the tables' columns no bare column can be shown to be bound inside the subquery
    held: list[exp.Subquery] = []
    for node in tree.find_all(exp.In):
        query = node.args.get("query")
        inner = query.this if isinstance(query, exp.Subquery) else query
        if not isinstance(inner, exp.Select) or not inner.args.get("group") or inner.args.get("having") is None:
            continue
        for scalar in _scalar_candidates(inner.args["having"], inner, schema):
            scalar.replace(exp.Anonymous(this=PLACEHOLDER, expressions=[exp.Literal.number(len(held))]))
            held.append(scalar)
    if not held:
        return tree
    tree = lift(tree)
    for call in list(tree.find_all(exp.Anonymous)):
        if call.name == PLACEHOLDER and len(call.expressions) == 1 and call.expressions[0].is_int:
            call.replace(held[int(call.expressions[0].name)].copy())
    return tree
