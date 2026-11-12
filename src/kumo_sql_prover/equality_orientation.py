"""``a.x = b.y`` and ``b.y = a.x`` are written one way.

Equality is symmetric: for every pair of values, NULLs included, ``x = y`` and ``y = x`` are both TRUE, both FALSE
or both NULL. The solver knows that, but a relation the prover keeps whole and matches by its text (a select with
a window function, a grouped derived table it cannot open such as one with ``COUNT(DISTINCT ..)`` over an outer
join) does not: ``LEFT JOIN posts AS p ON p.owner = u.id`` and ``.. ON u.id = p.owner`` were two relations. Calcite
prints join conditions with the left input's column first, whichever way the query wrote them.

``orient_column_equalities`` runs once, after every other rule (none of them sees the changed order), and swaps the
operands of an ``=`` between two plain columns when the right one's column name sorts first. Only the name decides:
the texts are compared after their table aliases are renamed, and a name survives that renaming while an alias does
not. Two columns of the same name (``a.id = b.id``) are left as written: ordering them by alias would turn the
equality one way in a query that names its tables ``b, e`` and the other way in one that names them ``z, a``, and
``windowed_inputs`` has by then written such an equality in the order of its sources.

Only ``=`` between two columns (a name with an optional table qualifier) is touched: no other comparison, no
expression operand, no ``IS NOT DISTINCT FROM``, no row comparison.
"""

from __future__ import annotations

from sqlglot import exp


def _key(column: exp.Column) -> str:
    return column.name.lower()


def _plain(node: exp.Expression) -> bool:
    return type(node) is exp.Column and isinstance(node.this, exp.Identifier) and not node.args.get("db") and not node.args.get("catalog")


def orient_column_equalities(tree: exp.Expression) -> exp.Expression:
    """``tree`` with every column-to-column ``=`` listing the column that sorts first on the left (in place)."""

    for node in list(tree.find_all(exp.EQ)):
        left, right = node.this, node.expression
        if type(node) is exp.EQ and _plain(left) and _plain(right) and _key(right) < _key(left):
            left_copy, right_copy = left.copy(), right.copy()
            node.set("this", right_copy)
            node.set("expression", left_copy)
    return tree
