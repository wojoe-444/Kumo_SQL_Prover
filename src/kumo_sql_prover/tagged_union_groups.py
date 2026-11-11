"""A ``GROUP BY`` on a column that is a different constant in each ``UNION ALL`` branch groups each branch on its own.

    SELECT u.tag, SUM(u.n) AS n FROM (SELECT 1 AS tag, COUNT(*) AS n FROM a UNION ALL SELECT 2 AS tag, COUNT(*) AS n FROM b) AS u GROUP BY u.tag

    SELECT u.tag, SUM(u.n) AS n FROM (SELECT 1 AS tag, COUNT(*) AS n FROM a) AS u GROUP BY u.tag
    UNION ALL
    SELECT u.tag, SUM(u.n) AS n FROM (SELECT 2 AS tag, COUNT(*) AS n FROM b) AS u GROUP BY u.tag

Every row of branch ``i`` carries the integer literal ``c_i`` in the tag column, and the literals are pairwise
different. The select groups by that column (among others), so two rows from different branches are never in one
group: each group lies inside one branch, and the groups of the union are the groups of branch 1, then those of
branch 2, and so on. Aggregates, ``HAVING`` and the select list are evaluated per group, so each group gives the
same row in both forms; a branch without rows contributes no group in both (the ``GROUP BY`` list is not empty).

The same holds when the union is one of several inner-joined sources: a join distributes over ``UNION ALL`` (each
joined row comes from exactly one branch row, as often as before), and a ``WHERE`` filters rows one by one. So

    SELECT t.tag, COUNT(*) FROM a CROSS JOIN (SELECT 1 AS tag UNION ALL SELECT 2) AS t GROUP BY t.tag

splits the same way. Calcite's aggregate-union-transpose and the rewrites of it are written in these two forms,
and the pieces are then read by the rules for one-row sources and one-row groups.

Conditions, all checked:

* the select has a plain ``GROUP BY`` list (no ``ROLLUP``, ``CUBE``, ``GROUPING SETS``, ``ALL``, ``()``) and no
  ``DISTINCT``, ``QUALIFY``, window, ``ORDER BY``, ``LIMIT`` or ``OFFSET`` (those read all groups together);
* every join of the select is a plain inner, cross or comma join (an outer join does not distribute over the
  padded side);
* the union is a derived table with an alias and no column list, of 2 to 8 ``UNION ALL`` branches (no ``UNION``
  duplicate removal, ``ORDER BY`` or ``LIMIT`` on the union), each a select with the same number of outputs; the
  first branch names every output, as a union takes its column names from it;
* in one output position every branch has an integer literal, pairwise different. Strings are not accepted
  (equality of strings depends on the collation), nor are expressions, NULL, or negative or decimal numbers;
* the select groups by that column, written with the union's alias, or bare when the union is the only source;
* the select is a query of its own (top level, a derived table, a set-operation branch), not an operand of
  ``IN`` or a scalar position where a union could not stand.

Each copy reads its branch under the union's alias, with the first branch's column names.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, extended_grouping

_BANNED = ("distinct", "qualify", "order", "limit", "offset", "windows", "with_", "with")
MAX_BRANCHES = 8


def _branches(node: exp.Expression) -> list[exp.Select] | None:
    if isinstance(node, exp.Select):
        return [node]
    if type(node) is not exp.Union or node.args.get("distinct") is not False:
        return None
    if any(v for k, v in node.args.items() if k not in ("this", "expression", "distinct")):
        return None
    left, right = _branches(node.this), _branches(node.expression)
    return None if left is None or right is None else left + right


def _plain_inner(join: exp.Join) -> bool:
    kind = (join.args.get("kind") or "").upper()
    return kind in ("", "INNER", "CROSS") and not any(v for k, v in join.args.items() if k not in ("this", "kind", "on"))


def _integer(node: exp.Expression) -> str | None:
    value = node.this if isinstance(node, exp.Alias) else node
    if isinstance(value, exp.Literal) and not value.is_string and value.name.isdigit():
        return str(int(value.name))
    return None


def split_tagged_union_group(select: exp.Select) -> exp.Expression | None:
    """The ``UNION ALL`` of ``select`` over each branch of a tagged union it groups by (module doc), or None."""

    group = select.args.get("group")
    if group is None or not group.expressions or extended_grouping(group) or any(group.args.get(k) for k in ("all", "totals")):
        return None
    if any(select.args.get(key) for key in _BANNED) or any(w.find_ancestor(exp.Select) is select for w in select.find_all(exp.Window)):
        return None
    if not isinstance(select.parent, (exp.Subquery, exp.CTE, exp.SetOperation, type(None))):
        return None
    if isinstance(select.parent, exp.Subquery) and not isinstance(select.parent.parent, (exp.From, exp.Join, exp.SetOperation, type(None))):
        return None
    from_ = select.args.get(FROM_KEY)
    joins = list(select.args.get("joins") or [])
    if from_ is None or any(v for k, v in from_.args.items() if k != "this") or not all(_plain_inner(join) for join in joins):
        return None
    sources = [from_.this] + [join.this for join in joins]
    for position, source in enumerate(sources):
        alias = source.args.get("alias")
        if not isinstance(source, exp.Subquery) or not source.alias or (isinstance(alias, exp.TableAlias) and alias.args.get("columns")):
            continue
        if any(v for k, v in source.args.items() if k not in ("this", "alias")):
            continue
        branches = _branches(source.this) if isinstance(source.this, exp.Union) else None
        if branches is None or not 2 <= len(branches) <= MAX_BRANCHES:
            continue
        width = len(branches[0].expressions)
        if any(len(branch.expressions) != width for branch in branches):
            continue
        if any(isinstance(i, exp.Star) or (isinstance(i, exp.Column) and isinstance(i.this, exp.Star)) for b in branches for i in b.expressions):
            continue
        names = [item.alias_or_name for item in branches[0].expressions]
        if "" in names or len({name.lower() for name in names}) != width:
            continue
        tags = set()
        for index in range(width):
            values = [_integer(branch.expressions[index]) for branch in branches]
            if None not in values and len(set(values)) == len(values):
                tags.add(names[index].lower())
        grouped = any(
            isinstance(key, exp.Column)
            and key.name.lower() in tags
            and not key.args.get("db")
            and (key.table.lower() == source.alias.lower() if key.table else len(sources) == 1)
            for key in group.expressions
        )
        if not grouped:
            continue
        result: exp.Expression | None = None
        for branch in branches:
            body = branch.copy()
            body.set("expressions", [exp.alias_((item.this if isinstance(item, exp.Alias) else item).copy(), name) for item, name in zip(body.expressions, names)])
            piece = select.copy()
            piece_sources = [piece.args[FROM_KEY].this] + [join.this for join in piece.args.get("joins") or []]
            piece_sources[position].set("this", body)
            result = piece if result is None else exp.Union(this=result, expression=piece, distinct=False)
        return result
    return None
