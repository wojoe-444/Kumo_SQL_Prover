"""``COUNT(*)`` over a ``FULL JOIN``, against its ``LEFT JOIN UNION ALL RIGHT JOIN`` rewrite.

Engines test a full join by counting it and counting the rewrite that every textbook gives:

    SELECT COUNT(*) FROM l FULL JOIN o ON l.k = o.k

    SELECT COUNT(*) FROM (SELECT l.k, o.k AS o2 FROM l LEFT JOIN o ON l.k = o.k
                          UNION ALL
                          SELECT l.k, o.k AS o2 FROM l RIGHT JOIN o ON l.k = o.k WHERE l.k IS NULL)

The provers decline the first ("aggregate over an outer join": nothing wraps a join whose columns the aggregate
does not read) and cannot match the second, whose two branches list columns nobody reads. Three rewrites, applied
once before the rule loop, bring both to one text.

**1. A full join under an ``IS NOT NULL`` test keeps one side.** ``a FULL JOIN b ON c WHERE b.x IS NOT NULL`` is
``a RIGHT JOIN b ON c WHERE b.x IS NOT NULL``: the rows of ``a`` without a match carry NULL in every column of ``b``,
so the test drops them, and what is left is the matched pairs and the unmatched rows of ``b``. A test on ``a``
leaves ``a LEFT JOIN b``; tests on both leave the inner join. (``algebraic_equivalence._full_join_to_one_sided``
reads comparisons this way; an ``IS NOT NULL`` test was turned into ``COUNT(x)`` by another rule before anything
read it.) Read only for one join with an ``ON``, sources with different aliases, and a test that is a top-level
``WHERE`` conjunct on a column qualified by one of the two aliases.

**2. The count of a full join is the count of the two halves.** For ``SELECT COUNT(*) FROM a FULL JOIN b ON c`` with
nothing else in the select: the full join's rows are the rows of ``a LEFT JOIN b ON c`` (matched pairs and
unmatched rows of ``a``) and the unmatched rows of ``b``. When ``c`` has a top-level conjunct that compares a column
``a.k`` (``=``, ``<``, ``<=``, ``>``, ``>=``, ``!=``), ``a.k`` is not NULL on every matched pair, because a comparison with NULL
is never TRUE, and it is NULL on every unmatched row of ``b``, because the join pads it. So ``a RIGHT JOIN b ON c
WHERE a.k IS NULL`` is exactly the unmatched rows of ``b``, and the select becomes

    SELECT COUNT(*) FROM (SELECT 1 AS c0 FROM a LEFT JOIN b ON c UNION ALL SELECT 1 AS c0 FROM a RIGHT JOIN b ON c WHERE a.k IS NULL)

Declined without such a conjunct (no declared NOT NULL column is used), with ``USING``, with a ``WHERE``, a
``GROUP BY`` or anything else in the select, with any source that is not a table or an aliased derived table.

**3. ``COUNT(*)`` does not read the columns of what it counts.** In ``SELECT COUNT(*) FROM (b1 UNION ALL b2 ..)`` with
nothing else in the outer select, each branch that is a plain select (a ``FROM``, joins and a ``WHERE``; no
``DISTINCT``, grouping, ``HAVING``, ``QUALIFY``, ``ORDER BY``, ``LIMIT``; no aggregate, window, subquery, ``UNNEST`` or
unknown function in its select list, which could change how many rows it returns) returns one row per row of its
``FROM`` and ``WHERE`` whatever it lists, so its list becomes ``1 AS c0`` and the derived table loses its alias.
Read only when the body is a ``UNION ALL`` with an outer join in one of its branches, which is the shape rewrite 2
produces; a ``UNION DISTINCT`` depends on the columns and is declined.

Dropping an expression from a select list can only remove an error (a division by zero, say), and a proof holds
where both queries succeed.
"""

from __future__ import annotations

from sqlglot import exp

_COMPARISONS = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE)
_ROW_MAKERS = (exp.AggFunc, exp.Window, exp.Subquery, exp.Select, exp.Exists, exp.Unnest, exp.UDTF, exp.Explode, exp.GenerateSeries, exp.Anonymous)
_TABLE_ARGS = ("this", "db", "catalog", "alias")
FROM_KEY = "from_"


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _conjuncts(node: exp.Expression | None) -> list[exp.Expression]:
    if node is None:
        return []
    node = _unwrap(node)
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _only(node: exp.Expression, allowed) -> bool:
    return not any(value for key, value in node.args.items() if key not in allowed)


def _source_alias(source: exp.Expression) -> str:
    """The alias of a table or an aliased derived table; "" for anything else."""

    if isinstance(source, exp.Table) and _only(source, _TABLE_ARGS) and isinstance(source.this, exp.Identifier):
        return (source.alias_or_name or "").lower()
    if isinstance(source, exp.Subquery) and _only(source, ("this", "alias")) and isinstance(source.args.get("alias"), exp.TableAlias) and not source.args["alias"].args.get("columns"):
        return (source.alias or "").lower()
    return ""


def _full_join(select: exp.Select) -> tuple[exp.Join, str, str] | None:
    """The select's one join when it is ``a FULL [OUTER] JOIN b ON c``, with the two aliases."""

    from_, joins = select.args.get(FROM_KEY), select.args.get("joins") or []
    if from_ is None or len(joins) != 1:
        return None
    join = joins[0]
    if not _only(join, ("this", "on", "side", "kind")) or (join.args.get("side") or "").upper() != "FULL" or (join.args.get("kind") or "").upper() not in ("", "OUTER"):
        return None
    if join.args.get("on") is None:
        return None
    left, right = _source_alias(from_.this), _source_alias(join.this)
    if not left or not right or left == right:
        return None
    return join, left, right


def _side_of(column: exp.Expression, left: str, right: str) -> str:
    column = _unwrap(column)
    if not isinstance(column, exp.Column) or not isinstance(column.this, exp.Identifier) or len(column.parts) != 2:
        return ""
    table = column.table.lower()
    return table if table in (left, right) else ""


def _keep_tested_side(select: exp.Select) -> bool:
    """Rewrite 1, in place. True when the join changed."""

    found = _full_join(select)
    where = select.args.get("where")
    if found is None or where is None:
        return False
    join, left, right = found
    tested = set()
    for part in _conjuncts(where.this):
        if isinstance(part, exp.Not) and isinstance(_unwrap(part.this), exp.Is) and isinstance(_unwrap(part.this).expression, exp.Null):
            tested.add(_side_of(_unwrap(part.this).this, left, right))
    tested.discard("")
    if not tested:
        return False
    join.set("side", None if tested == {left, right} else "RIGHT" if tested == {right} else "LEFT")
    join.set("kind", None)
    return True


def _count_star_only(select: exp.Select, allowed) -> bool:
    if not _only(select, allowed) or len(select.expressions) != 1:
        return False
    item = select.expressions[0].unalias()
    return isinstance(item, exp.Count) and isinstance(item.this, exp.Star) and _only(item, ("this", "big_int")) and _only(item.this, ())


def _one() -> exp.Alias:
    return exp.Alias(this=exp.Literal.number(1), alias=exp.to_identifier("c0"))


def _split_count(select: exp.Select) -> bool:
    """Rewrite 2, in place. True when the select changed."""

    if not _count_star_only(select, ("expressions", FROM_KEY, "joins")):
        return False
    found = _full_join(select)
    if found is None:
        return False
    join, left, _ = found
    padded = None
    for part in _conjuncts(join.args["on"]):
        if isinstance(part, _COMPARISONS):
            for operand in (part.this, part.expression):
                if padded is None and _side_of(operand, left, left) == left:
                    padded = _unwrap(operand)
    if padded is None:
        return False
    halves = []
    for side in ("LEFT", "RIGHT"):
        half = exp.Select(expressions=[_one()])
        half.set(FROM_KEY, select.args[FROM_KEY].copy())
        copy = join.copy()
        copy.set("side", side)
        copy.set("kind", None)
        half.set("joins", [copy])
        if side == "RIGHT":
            half.set("where", exp.Where(this=exp.Is(this=padded.copy(), expression=exp.Null())))
        halves.append(half)
    select.set("joins", None)
    select.set(FROM_KEY, exp.From(this=exp.Subquery(this=exp.Union(this=halves[0], expression=halves[1], distinct=False))))
    return True


def _branches(body: exp.Expression) -> list[exp.Select] | None:
    """The selects of a ``UNION ALL`` tree, or None when the body is anything else."""

    if isinstance(body, exp.Subquery) and _only(body, ("this",)):
        return _branches(body.this)
    if isinstance(body, exp.Select):
        return [body]
    if isinstance(body, exp.Union) and _only(body, ("this", "expression")):  # distinct is False or missing
        left, right = _branches(body.this), _branches(body.expression)
        return None if left is None or right is None else left + right
    return None


def _blank_unread_columns(select: exp.Select) -> bool:
    """Rewrite 3, in place. True when the select changed."""

    if not _count_star_only(select, ("expressions", FROM_KEY)):
        return False
    source = select.args[FROM_KEY].this if select.args.get(FROM_KEY) is not None else None
    if not isinstance(source, exp.Subquery) or not _only(source, ("this", "alias")) or not isinstance(source.this, exp.Union):
        return False
    alias = source.args.get("alias")
    if alias is not None and (not isinstance(alias, exp.TableAlias) or alias.args.get("columns")):
        return False
    branches = _branches(source.this)
    if branches is None or len(branches) < 2:
        return False
    for branch in branches:
        if not _only(branch, ("expressions", FROM_KEY, "joins", "where")) or not branch.expressions:
            return False
        if any(item.find(*_ROW_MAKERS) is not None for item in branch.expressions):
            return False
    if not any((j.args.get("side") or "").upper() in ("LEFT", "RIGHT", "FULL") for branch in branches for j in branch.args.get("joins") or []):
        return False
    plain = _one().sql()
    if alias is None and all(len(b.expressions) == 1 and b.expressions[0].sql() == plain for b in branches):
        return False
    for branch in branches:
        branch.set("expressions", [_one()])
    source.set("alias", None)
    return True


def counted_join_rules(tree: exp.Expression) -> exp.Expression:
    """``tree`` with the three rewrites of the module doc applied to every select they fit."""

    if not any((j.args.get("side") or "").upper() in ("LEFT", "RIGHT", "FULL") for j in tree.find_all(exp.Join)):
        return tree
    for select in list(tree.find_all(exp.Select)):
        _keep_tested_side(select)
    for select in list(tree.find_all(exp.Select)):
        _split_count(select) or _blank_unread_columns(select)
    return tree
