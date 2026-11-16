"""A membership test whose inner test asks about the outer row is two tests of the outer row.

    SELECT p.id FROM product AS p
    WHERE EXISTS (SELECT 1 FROM sales AS s WHERE s.pid = p.id AND s.day < 5
                  AND NOT EXISTS (SELECT 1 FROM sales AS t WHERE t.pid = s.pid AND t.day > 9))

The inner ``NOT EXISTS`` is tied to ``s.pid``, and every row ``s`` that can pass has ``s.pid = p.id``. So it asks
about ``p.id``, it is the same for every such ``s``, and it can be asked once, outside:

    SELECT p.id FROM product AS p
    WHERE EXISTS (SELECT 1 FROM sales AS s WHERE s.pid = p.id AND s.day < 5)
      AND NOT EXISTS (SELECT 1 FROM sales AS t WHERE t.pid = p.id AND t.day > 9)

Two steps, both only for a test that is a top-level conjunct of a ``WHERE`` (where a test that is not TRUE drops
the row, whether it is FALSE or NULL):

1. **Name the outer column.** The subquery's ``WHERE`` has the top-level conjunct ``s.c = p.d`` (``s`` a source
   of the subquery, ``p`` a source of the select around it), or the test is ``p.d IN (SELECT s.c ..)``. A row
   ``s`` makes the test TRUE only when ``s.c = p.d`` is TRUE, so inside the queries nested in the subquery's
   ``WHERE`` a read of ``s.c`` is replaced by ``p.d``: for the rows that matter the two are equal, and for the
   others the test is not TRUE either way.
2. **Ask once.** A top-level conjunct of the subquery's ``WHERE`` that reads no column of the subquery's own
   sources is the same for every row of it, so it moves out, next to the test: ``EXISTS (.. WHERE a AND b)`` is
   ``b AND EXISTS (.. WHERE a)`` when ``b`` does not depend on the row.

The subquery must be a plain select: no grouping, aggregate, window, ``LIMIT``, set operation or ``DISTINCT ON``
(for ``IN``, one plain output column). Every column in it, at any depth, must name its source in full, and no
query nested in it may reuse the name of one of its sources or of the outer source, so a read cannot change the
row it means. A volatile call anywhere leaves the select as it is.
"""

from __future__ import annotations

from sqlglot import exp

_BANNED = ("group", "having", "qualify", "windows", "with_", "with", "limit", "offset", "order", "laterals", "pivots", "into", "locks", "sample")


def _unparen(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _names(select: exp.Select) -> set[str] | None:
    """The names of the sources of ``select``; None when one has no name."""

    from_ = select.args.get("from_") or select.args.get("from")
    sources = ([from_.this] if from_ is not None else []) + [join.this for join in select.args.get("joins") or []]
    names = {(source.alias_or_name or "").lower() for source in sources}
    return None if "" in names else names


def _plain_column(node: exp.Expression) -> tuple[str, str] | None:
    node = _unparen(node)
    if isinstance(node, exp.Column) and isinstance(node.this, exp.Identifier) and node.table and not node.args.get("db"):
        return node.table.lower(), node.name.lower()
    return None


def _probe(test: exp.Expression) -> tuple[exp.Select, tuple[str, str] | None] | None:
    """``(subquery, outer column of an IN test)`` for ``EXISTS (select)`` or ``column IN (select)``."""

    test = _unparen(test)
    if isinstance(test, exp.Exists):
        inner = test.this.this if isinstance(test.this, exp.Subquery) else test.this
        return (inner, None) if isinstance(inner, exp.Select) else None
    if isinstance(test, exp.In) and not test.expressions and not test.args.get("unnest") and not test.args.get("field"):
        query = test.args.get("query")
        inner = query.this if isinstance(query, exp.Subquery) else query
        outer = _plain_column(test.this)
        if isinstance(inner, exp.Select) and outer is not None and len(inner.expressions) == 1:
            return inner, outer
    return None


def flatten_nested_correlation(select: exp.Select) -> exp.Expression | None:
    from .algebraic_equivalence import _conjuncts
    from .counted_membership import _unstable

    where = select.args.get("where")
    around = _names(select)
    if where is None or not around or where.find(exp.Subquery, exp.Exists) is None:
        return None
    result = select.copy()
    parts = _conjuncts(result.args["where"].this)
    hoisted: dict[int, list[exp.Expression]] = {}
    changed = False
    for position, part in enumerate(parts):
        found = _probe(part)
        if found is None:
            continue
        inner, tested = found
        own = _names(inner)
        if own is None or own & around or any(inner.args.get(k) for k in _BANNED):
            continue
        distinct = inner.args.get("distinct")
        if (distinct is not None and distinct.args.get("on")) or inner.args.get("where") is None:
            continue
        if any(node.find_ancestor(exp.Select) is inner for node in inner.find_all(exp.AggFunc, exp.Window)):
            continue
        columns = list(inner.find_all(exp.Column))
        if any(isinstance(c.this, exp.Star) or not c.table or c.args.get("db") for c in columns) or inner.find(exp.Star) is not None:
            continue
        nested = [node for node in inner.find_all(exp.Select) if node is not inner]
        if any((_names(node) is None or _names(node) & (own | around)) for node in nested):
            continue  # a nested query that reuses a name could read another row under it
        if any(node.find_ancestor(exp.Select) is inner and isinstance(node, exp.Subquery) and node.find_ancestor(exp.From, exp.Join) is not None and node.find_ancestor(exp.Where) is None for node in inner.find_all(exp.Subquery)):
            continue  # a derived table among the subquery's own sources: left alone
        inner_where = inner.args["where"]
        # 1. the outer column each own column is equal to, for every row that can make the test TRUE
        equal: dict[tuple[str, str], tuple[str, str]] = {}
        if tested is not None:
            output = _plain_column(inner.expressions[0].unalias())
            if output is None or output[0] not in own or tested[0] not in around:
                continue
            equal[output] = tested
        for conjunct in _conjuncts(inner_where.this):
            conjunct = _unparen(conjunct)
            if not isinstance(conjunct, exp.EQ):
                continue
            a, b = _plain_column(conjunct.this), _plain_column(conjunct.expression)
            for mine, theirs in ((a, b), (b, a)):
                if mine is not None and theirs is not None and mine[0] in own and theirs[0] in around:
                    equal.setdefault(mine, theirs)
        renamed = False
        for column in columns:
            key = _plain_column(column)
            scope = column.find_ancestor(exp.Select)
            if key in equal and scope is not inner and not any(a is inner.args.get("from_") or a is inner.args.get("from") for a in _ancestors(column, inner)) and not _in_join(column, inner):
                column.replace(exp.column(equal[key][1], table=equal[key][0]))
                renamed = True
        # 2. conjuncts that no longer read a row of the subquery are asked once, outside
        kept, moved = [], []
        for conjunct in _conjuncts(inner_where.this):
            reads_own = any(c.table.lower() in own for c in conjunct.find_all(exp.Column))
            if reads_own or conjunct.find(exp.Subquery, exp.Exists) is None:
                kept.append(conjunct)  # a plain condition on the outer row alone is left where it is
            else:
                moved.append(conjunct)
        if moved and kept:
            inner_where.set("this", exp.and_(*kept, copy=False))
            hoisted[position] = moved
        if (moved and kept) or renamed:
            changed = True
    if not changed or _unstable(select):
        return None
    final: list[exp.Expression] = []
    for position, part in enumerate(parts):
        final.append(part)
        final.extend(hoisted.get(position, []))
    result.set("where", exp.Where(this=exp.and_(*final, copy=False)))
    return result


def _ancestors(node: exp.Expression, stop: exp.Expression):
    node = node.parent
    while node is not None and node is not stop:
        yield node
        node = node.parent


def _in_join(column: exp.Expression, inner: exp.Select) -> bool:
    joins = inner.args.get("joins") or []
    return any(any(a is join for join in joins) for a in _ancestors(column, inner))
