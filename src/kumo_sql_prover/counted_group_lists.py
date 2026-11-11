"""A select over the groups of a counted union that lists something other than its keys.

    SELECT 0 AS f FROM (SELECT a FROM r GROUP BY a UNION ALL SELECT a FROM u GROUP BY a) AS t GROUP BY a HAVING COUNT(*) = 2

Calcite writes ``INTERSECT`` this way (each operand reduced to its distinct rows, the operands stacked, the rows
that occur once per operand kept) and, under a ``COUNT(*)`` that reads no column, replaces the select list by a
constant. The normalizer reads the construction back as an intersection only when the select lists exactly its
grouping keys (``collapse_counted_intersection``). This pass gives it that form:

    SELECT 0 AS f FROM (SELECT a FROM (.. UNION ALL ..) AS t GROUP BY a HAVING COUNT(*) = 2) AS t

A grouped select returns one row per group that passes ``HAVING``. When its list holds no aggregate and reads
only grouping keys, every item is a function of the group's key values, so it can be computed from the rows of
the keyed select just as well: the same rows, in the same number.

Read only when:

* the select reads one derived ``UNION`` and has ``GROUP BY`` plain columns of it (each once), ``HAVING
  COUNT(*) = n`` for a whole number ``n``, and nothing else: no join, ``WHERE``, ``DISTINCT``, window,
  ``ORDER BY``, ``LIMIT``, ``ROLLUP`` or grouping sets;
* no item of its list holds an aggregate, a window, a subquery or a star, and every column an item reads is
  one of the grouping keys, named bare or by the derived table's alias;
* the list is not already the keys themselves (that form is the existing rule's).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, extended_grouping

_OTHER_CLAUSES = ("joins", "where", "distinct", "qualify", "order", "limit", "offset", "windows", "with_", "with", "laterals", "pivots", "sample")


def _split(select: exp.Select) -> exp.Select | None:
    source = select.args.get("from_") or select.args.get("from")
    group, having = select.args.get("group"), select.args.get("having")
    if source is None or group is None or having is None or any(select.args.get(key) for key in _OTHER_CLAUSES):
        return None
    derived = source.this
    if not isinstance(derived, exp.Subquery) or not derived.alias or type(derived.this) is not exp.Union:
        return None
    alias_node = derived.args.get("alias")
    if alias_node.args.get("columns") or any(value for key, value in derived.args.items() if key not in ("this", "alias") and value):
        return None
    if extended_grouping(group) or not group.expressions or any(value for key, value in group.args.items() if key != "expressions" and value):
        return None
    test = having.this
    if not (isinstance(test, exp.EQ) and isinstance(test.left, exp.Count) and isinstance(test.left.this, exp.Star)):
        return None
    if any(value for key, value in test.left.args.items() if key not in ("this", "big_int") and value):
        return None
    if not (isinstance(test.right, exp.Literal) and test.right.is_int):
        return None
    alias = derived.alias.lower()

    def key_name(column: exp.Expression) -> str | None:
        if not isinstance(column, exp.Column) or isinstance(column.this, exp.Star):
            return None
        if column.table and column.table.lower() != alias or column.args.get("db") or column.args.get("catalog"):
            return None
        return column.name.lower()

    keys = [key_name(key) for key in group.expressions]
    if None in keys or len(set(keys)) != len(keys):
        return None
    items = select.expressions
    if any(isinstance(n, (exp.AggFunc, exp.Window, exp.Subquery, exp.Exists, exp.Star, exp.Anonymous, exp.Select)) for item in items for n in item.walk()):
        return None
    if any(key_name(column) not in keys for item in items for column in item.find_all(exp.Column)):
        return None
    listed = [key_name(item) for item in items]
    if None not in listed and len(set(listed)) == len(listed) and set(listed) == set(keys):
        return None  # already the keys: the existing rule's form
    inner = exp.Select(expressions=[key.copy() for key in group.expressions])
    inner.set(FROM_KEY, source.copy())
    inner.set("group", group.copy())
    inner.set("having", having.copy())
    outer = exp.Select(expressions=[item.copy() for item in items])
    outer.set(FROM_KEY, exp.From(this=exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier(derived.alias)))))
    for key in ("kind", "hint"):
        if select.args.get(key):
            return None
    return outer


def list_keys_of_counted_groups(tree: exp.Expression) -> exp.Expression:
    """Give each ``.. GROUP BY keys HAVING COUNT(*) = n`` over a derived union a keyed select of its own."""

    for select in list(tree.find_all(exp.Select))[::-1]:
        replacement = _split(select)
        if replacement is None:
            continue
        if select is tree:
            tree = replacement
        else:
            select.replace(replacement)
    return tree
