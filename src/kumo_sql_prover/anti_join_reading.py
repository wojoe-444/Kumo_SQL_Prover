"""A ``LEFT JOIN`` kept only where it found nothing is ``NOT EXISTS``.

    SELECT a.buyer
    FROM a LEFT JOIN (SELECT DISTINCT buyer FROM sales WHERE item = 2) AS b ON a.buyer = b.buyer
    WHERE b.buyer IS NULL

is

    SELECT a.buyer FROM a
    WHERE NOT EXISTS (SELECT 1 FROM (SELECT DISTINCT buyer FROM sales WHERE item = 2) AS b WHERE a.buyer = b.buyer)

Why it holds. Take a row ``r`` of the left side. If some row of ``b`` makes the ``ON`` condition true, the join
gives ``r`` once per such row, and in each of them ``b.buyer`` is not NULL (the ``ON`` condition compares it, and a
comparison with NULL is never true), so ``WHERE b.buyer IS NULL`` drops them all. If no row does, the join gives
``r`` once, padded with NULLs, and the ``WHERE`` keeps it. So ``r`` is kept, once, exactly when no row of ``b``
matches: ``NOT EXISTS``. Nothing else may read ``b``, since after the rewrite it is not there to read.

The rule fires on a select whose **last** join is a plain ``LEFT [OUTER] JOIN`` with an ``ON`` condition of a
derived table that removes duplicates or groups (the other rules already read a left join of a base table or of a
derived table that only filters, and those are left to them), and whose ``WHERE`` has the top-level conjunct
``b.c IS NULL`` where the ``ON`` condition has a top-level conjunct that compares ``b.c`` itself (``=``, ``<>``,
``<``, ``<=``, ``>``, ``>=``). Every other column of the select, at any depth, must name another source: a bare
column name anywhere, a star, or a second read of ``b`` leaves the select as it is. So does a volatile call.
"""

from __future__ import annotations

from sqlglot import exp

_COMPARISONS = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE)
_JOIN_EXTRAS = ("method", "using", "match_condition", "global_", "hint", "directed", "expressions", "pivots")


def _unparen(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _own_column(node: exp.Expression, alias: str) -> str | None:
    node = _unparen(node)
    if isinstance(node, exp.Column) and isinstance(node.this, exp.Identifier) and node.table.lower() == alias and not node.args.get("db"):
        return node.name.lower()
    return None


def read_unmatched_left_join(select: exp.Select) -> exp.Expression | None:
    from .algebraic_equivalence import _conjuncts
    from .counted_membership import _unstable

    joins = select.args.get("joins") or []
    where = select.args.get("where")
    if not joins or where is None or select.args.get("laterals") or select.args.get("pivots"):
        return None
    join = joins[-1]
    side, kind = (join.side or "").upper(), (join.kind or "").upper()
    on = join.args.get("on")
    if side != "LEFT" or kind not in ("", "OUTER") or on is None or any(join.args.get(k) for k in _JOIN_EXTRAS):
        return None
    source = join.this
    if not isinstance(source, exp.Subquery) or not source.alias or not isinstance(source.this, exp.Select) or any(source.args.get(k) for k in ("pivots", "sample", "joins")):
        return None
    inner = source.this
    if inner.args.get("distinct") is None and inner.args.get("group") is None:
        return None
    alias = source.alias.lower()
    compared = set()
    for part in _conjuncts(on):
        part = _unparen(part)
        if isinstance(part, _COMPARISONS):
            compared |= {_own_column(part.this, alias), _own_column(part.expression, alias)}
    compared.discard(None)
    tests = []
    for part in _conjuncts(where.this):
        bare = _unparen(part)
        if isinstance(bare, exp.Is) and isinstance(bare.expression, exp.Null) and _own_column(bare.this, alias) in compared:
            tests.append(part)
    if not tests:
        return None
    test = tests[0]
    # nothing else may read the joined table: every other column names another source in full
    sources = [j.this for j in joins]
    if select.args.get("from_") is not None or select.args.get("from") is not None:
        sources.append(select.args.get("from_") or select.args.get("from"))
    for column in select.find_all(exp.Column):
        if any(parent is test or parent is join or any(parent is s for s in sources) for parent in _ancestors(column, select)):
            continue  # the test itself, the join, or the inside of a source, which cannot see the joined table
        if isinstance(column.this, exp.Star) or not column.table or column.table.lower() == alias:
            return None
    if any(isinstance(item, exp.Star) for item in select.expressions):
        return None
    names = [(j.this.alias_or_name or "").lower() for j in joins[:-1]]
    from_ = select.args.get("from_") or select.args.get("from")
    names.append((from_.this.alias_or_name or "").lower() if from_ is not None else "")
    if alias in names or "" in names or _unstable(select):
        return None
    result = select.copy()
    kept = result.args["joins"][:-1]
    result.set("joins", kept or None)
    probe = exp.select("1").from_(source.copy()).where(on.copy(), copy=False)
    parts = [exp.Not(this=exp.Exists(this=probe)) if part is test else part.copy() for part in _conjuncts(where.this)]
    result.set("where", exp.Where(this=exp.and_(*parts, copy=False)))
    return result


def _ancestors(node: exp.Expression, stop: exp.Expression):
    node = node.parent
    while node is not None and node is not stop:
        yield node
        node = node.parent
