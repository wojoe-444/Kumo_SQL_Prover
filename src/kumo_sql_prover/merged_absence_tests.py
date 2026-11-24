"""Two absence tests of one table, asked as one.

    SELECT p.id FROM product AS p
    WHERE NOT EXISTS (SELECT 1 FROM sales AS a WHERE a.pid = p.id AND a.day < 5)
      AND NOT EXISTS (SELECT 1 FROM sales AS b WHERE b.pid = p.id AND b.day > 9)

asks twice whether a sale of the product is missing, and is

    WHERE NOT EXISTS (SELECT 1 FROM sales AS a WHERE a.pid = p.id AND (a.day < 5 OR a.day > 9))

The provers read "no sale outside the window" as the one test (``counted_presence`` writes the bounds of a group that
way), and a query that listed the sales before the window and the sales after it in two ``NOT IN`` lists reached
them as two tests and stayed unproved (hardest-cases batch 30, LeetCode 1084).

**Why it is the same test.** ``EXISTS`` is TRUE or FALSE, never NULL: it is TRUE when some row makes its ``WHERE``
TRUE. ``W1 OR W2`` is TRUE on a row exactly when ``W1`` is TRUE or ``W2`` is TRUE on it (a NULL beside a FALSE
is not TRUE). So some row of the table makes ``W1 OR W2`` TRUE exactly when some row makes ``W1`` TRUE or some row
makes ``W2`` TRUE: ``EXISTS (W1 OR W2)`` is ``EXISTS (W1) OR EXISTS (W2)``, and the two ``NOT EXISTS`` joined by
``AND`` are its negation. The two subqueries read the same table, so the rows are the same rows, under one name
after the second one's reads are renamed. Parts that both conditions have are written once: ``(T AND a) OR (T AND
b)`` is ``T AND (a OR b)`` in three-valued logic too, and ``T OR (T AND b)`` is ``T``.

**What is read.** Top-level ``AND`` parts of one ``WHERE`` of the form ``NOT EXISTS (SELECT .. FROM t [AS a] WHERE ..)``,
each over the same single base table with no join, grouping, ``HAVING``, ``DISTINCT``, ``ORDER BY`` or row limit, whose
select list is constants, stars or plain columns (an aggregate there would make one row out of none), whose
``WHERE`` holds no nested query and names the source of every column, and where the name of the first subquery's
table is not read by the others as an outer row's.

**Left alone.** ``EXISTS`` without ``NOT`` (two rows may each pass one test and no row both), tests under ``OR``,
subqueries over different tables or with a join, a bare column, a volatile call.

It runs in the last step of batch 27 only, so a pair an earlier step decides is decided as it was.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, conjuncts

NAME = "merged_absence_tests"
_PROBE_ARGS = frozenset({"expressions", FROM_KEY, "where"})
_TABLE_ARGS = frozenset({"this", "alias", "db", "catalog"})


def _set_args(node: exp.Expression) -> set[str]:
    return {key for key, value in node.args.items() if value not in (None, [], False)}


def _unparen(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _probe(part: exp.Expression) -> tuple[exp.Select, str, str] | None:
    """``(subquery, its table, the name it reads the table by)`` for ``NOT EXISTS (plain select over one table)``."""

    part = _unparen(part)
    if not isinstance(part, exp.Not):
        return None
    test = _unparen(part.this)
    if not isinstance(test, exp.Exists):
        return None
    query = test.this.this if isinstance(test.this, exp.Subquery) else test.this
    if not isinstance(query, exp.Select) or _set_args(query) - _PROBE_ARGS or query.args.get("where") is None:
        return None
    from_ = query.args.get(FROM_KEY)
    table = from_.this if from_ is not None else None
    if not isinstance(table, exp.Table) or not isinstance(table.this, exp.Identifier) or _set_args(table) - _TABLE_ARGS:
        return None
    alias = table.args.get("alias")
    if alias is not None and (alias.args.get("columns") or not isinstance(alias.this, exp.Identifier)):
        return None
    for item in query.expressions:
        value = _unparen(item.this if isinstance(item, exp.Alias) else item)
        if not isinstance(value, (exp.Literal, exp.Star, exp.Column, exp.Null, exp.Boolean)):
            return None
    where = query.args["where"].this
    if where.find(exp.Select, exp.Subquery, exp.Exists, exp.AggFunc, exp.Window) is not None:
        return None
    for column in where.find_all(exp.Column):
        if not column.table or not isinstance(column.this, exp.Identifier) or column.args.get("db") or column.args.get("catalog"):
            return None
    key = ".".join(part.name.lower() for part in (table.args.get("catalog"), table.args.get("db"), table.this) if part is not None)
    return query, key, table.alias_or_name.lower()


def merge_absence_tests(select: exp.Select) -> exp.Select | None:
    """``select`` with the absence tests of one table in its ``WHERE`` written as one (module doc), else ``None``.
    The input is not changed."""

    from .counted_membership import _unstable

    where = select.args.get("where")
    if where is None or len(list(where.find_all(exp.Exists))) < 2:
        return None
    result = select.copy()
    parts = conjuncts(result.args["where"].this)
    groups: dict[str, list[int]] = {}
    probes: dict[int, tuple[exp.Select, str]] = {}
    for position, part in enumerate(parts):
        found = _probe(part)
        if found is not None:
            probes[position] = (found[0], found[2])
            groups.setdefault(found[1], []).append(position)
    dropped: set[int] = set()
    for positions in groups.values():
        first, name = probes[positions[0]]
        merged = [conjuncts(first.args["where"].this)]
        taken = []
        for position in positions[1:]:
            other, own = probes[position]
            condition = other.args["where"].this
            if own != name:
                if any(column.table.lower() == name for column in condition.find_all(exp.Column)):
                    continue  # the first subquery's name means an outer row here
                for column in list(condition.find_all(exp.Column)):
                    if column.table.lower() == own:
                        column.set("table", exp.to_identifier(name))
            merged.append(conjuncts(condition))
            taken.append(position)
        if not taken:
            continue
        texts = [{piece.sql() for piece in pieces} for pieces in merged]
        shared = set.intersection(*texts)
        common = [piece for piece in merged[0] if piece.sql() in shared]
        rests = [[piece for piece in pieces if piece.sql() not in shared] for pieces in merged]
        if all(rests):
            common.append(exp.Paren(this=exp.or_(*(exp.and_(*rest, copy=False) if len(rest) > 1 else rest[0] for rest in rests), copy=False)))
        if not common:
            continue
        first.set("where", exp.Where(this=exp.and_(*common, copy=False) if len(common) > 1 else common[0]))
        dropped.update(taken)
    if not dropped or _unstable(select):
        return None
    kept = [part for position, part in enumerate(parts) if position not in dropped]
    result.set("where", exp.Where(this=exp.and_(*kept, copy=False) if len(kept) > 1 else kept[0]))
    return result
