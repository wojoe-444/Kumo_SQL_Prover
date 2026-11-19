"""Aggregates of the outer side of a left join read the matched rows only: the join is an inner join.

    SELECT COUNT(b.id) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id AND b.paid      -- .. JOIN bill AS b ON ..

**Why it holds.** A ``LEFT JOIN`` returns the rows of the inner join and, for each left row with no match, one row
whose right-side columns are all NULL. ``COUNT(x)``, ``COUNT(DISTINCT x)``, ``SUM(x)``, ``MIN(x)``, ``MAX(x)`` and
``AVG(x)`` skip a row on which ``x`` is NULL. So when every aggregate of a select reads one plain column of the right
side, the added rows change no aggregate: each is skipped by all of them, whether or not ``WHERE`` keeps it, and the
rows that come from a match are the same rows in both joins. A later join only attaches columns to a row or repeats
it; a row that descends from an added row still has NULL in every right-side column. The select has no ``GROUP BY``,
so there is one result row either way and no group can disappear with the added rows.

**What is read.** A select with no ``GROUP BY``, no ``HAVING``, no window and no ``DISTINCT``, whose joins are inner,
cross or left joins with ``ON`` (no ``USING``, ``NATURAL``, ``LATERAL``, ``RIGHT`` or ``FULL``), and whose outputs read
columns only inside aggregates. Every aggregate of the select is one of the six above over a plain column written
with the alias of one left-joined source (the same source for all of them; ``COUNT(*)`` or an aggregate of another
source declines the select). That source's ``LEFT JOIN`` becomes an inner join. A ``RIGHT JOIN`` of exactly two
sources is read the same way with the first source as the outer side.

A narrow retry rule of the tenth hardest-cases batch (``shape_fallback.TENTH_BATCH``).
``tests/test_outer_side_aggregates.py`` replays the rewrite on DuckDB.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import is_aggregate, select_sources

_SKIPPING = (exp.Count, exp.Sum, exp.Min, exp.Max, exp.Avg)


def _unparen(node: exp.Expression | None) -> exp.Expression | None:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _argument_source(call: exp.Expression) -> str | None:
    """The alias a skipping aggregate's one plain column is written with, or None."""

    if type(call) not in _SKIPPING or any(v for k, v in call.args.items() if k not in ("this", "big_int")) or isinstance(call.parent, (exp.Window, exp.Filter)):
        return None
    argument = call.this
    if isinstance(argument, exp.Distinct):
        if len(argument.expressions) != 1 or argument.args.get("on") is not None:
            return None
        argument = argument.expressions[0]
    argument = _unparen(argument)
    if isinstance(argument, exp.Column) and argument.table and not argument.args.get("db") and isinstance(argument.this, exp.Identifier):
        return argument.table.lower()
    return None


def join_matched_rows(select: exp.Select) -> exp.Select | None:
    """``select`` with the left join of its aggregated source as an inner join (module doc), or None."""

    joins = select.args.get("joins") or []
    if not joins or any(select.args.get(key) is not None for key in ("group", "having", "distinct", "qualify", "windows", "limit", "offset")):
        return None
    if any(True for _ in select.find_all(exp.Window)):
        return None
    for join in joins:
        if join.args.get("using") or join.args.get("method") or (join.args.get("kind") or "").upper() not in ("", "INNER", "CROSS", "OUTER"):
            return None
        if isinstance(join.this, exp.Lateral) or (join.args.get("side") or "").upper() not in ("", "LEFT", "RIGHT"):
            return None
    sources = select_sources(select)
    aliases = [source.alias_or_name.lower() if isinstance(source, (exp.Table, exp.Subquery)) and source.alias_or_name else None for source in sources]
    if None in aliases or len(set(aliases)) != len(aliases):
        return None
    rights = [join for join in joins if (join.args.get("side") or "").upper() == "RIGHT"]
    if rights and len(joins) != 1:
        return None
    own = [node for node in select.find_all(exp.AggFunc, exp.Anonymous) if is_aggregate(node) and node.find_ancestor(exp.Select) is select]
    if not own:
        return None
    names = {_argument_source(call) for call in own}
    if len(names) != 1 or None in names:
        return None
    name = names.pop()
    if name not in aliases:
        return None
    for item in select.expressions:  # outside the aggregates the outputs read no column of this select
        for column in item.find_all(exp.Column):
            if column.find_ancestor(exp.Select) is select and not any(is_aggregate(up) and up in own for up in _ancestors(column, select)):
                return None
    index = aliases.index(name)
    if rights:
        if index != 0 or rights[0].args.get("on") is None:
            return None
        target = joins[0]
    else:
        if index == 0:
            return None
        target = joins[index - 1]
        if (target.args.get("side") or "").upper() != "LEFT" or target.args.get("on") is None:
            return None
    result = select.copy()
    join = (result.args.get("joins") or [])[joins.index(target)]
    join.set("side", None)
    if (join.args.get("kind") or "").upper() == "OUTER":
        join.set("kind", None)
    return result


def join_matched_rows_everywhere(tree: exp.Expression) -> tuple[exp.Expression, bool]:
    """``tree`` with every such select rewritten (innermost first); whether anything changed."""

    changed = False
    tree = tree.copy()
    for select in reversed(list(tree.find_all(exp.Select))):
        new = join_matched_rows(select)
        if new is None:
            continue
        changed = True
        if select is tree:
            tree = new
        else:
            select.replace(new)
    return tree, changed


def _ancestors(node: exp.Expression, stop: exp.Expression):
    node = node.parent
    while node is not None and node is not stop:
        yield node
        node = node.parent
