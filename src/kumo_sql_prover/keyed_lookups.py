"""A filter on a value looked up by key is an ``EXISTS``.

    SELECT o.seat_id FROM cinema AS o
    WHERE o.free = 1 AND (SELECT n.free FROM cinema AS n WHERE n.seat_id = o.seat_id + 1) = 1

looks up the next seat by its key and tests the value found. The prover had no reading for a scalar subquery
that is not an aggregate. With ``seat_id`` a declared key the test is rewritten to

    .. AND EXISTS(SELECT 1 FROM cinema AS n WHERE n.seat_id = o.seat_id + 1 AND n.free = 1)

**Why it is the same.** Every column of a declared key of the looked-up table is compared for equality with a
value from outside the subquery, so the subquery finds at most one row (other filters only remove it). Its value
is that row's output, or NULL when there is no row. The comparison is therefore TRUE exactly when the row exists
and the comparison holds for it, which is when the ``EXISTS`` is TRUE. Where the comparison is not TRUE it may be
FALSE or unknown while the ``EXISTS`` is FALSE; a ``WHERE``, ``HAVING`` or ``ON`` only asks whether its
condition is TRUE, and ``AND`` / ``OR`` are TRUE for the same TRUE parts, so the rewrite is made only for a test
reached from the top of such a condition through ``AND`` and ``OR`` (never under ``NOT``, ``IS``, ``CASE`` or in
an output, where FALSE and unknown differ).

**What is read.** ``<scalar subquery> <op> <value>`` or the mirror image, ``op`` one of ``= <> < <= > >=``:

* the subquery is one select over one base table with supplied columns: one output without aggregate, window
  or nested query, a ``WHERE`` and nothing else (no join, ``GROUP BY``, ``DISTINCT``, ``LIMIT``, ``ORDER BY``);
* for some declared key of that table, each key column is compared by ``=`` with an expression whose columns
  are all qualified with another table's name (so it is a value from outside);
* the other operand has no aggregate, window or nested query, and each of its columns is qualified with a name
  other than the looked-up table's (it moves inside the subquery and must still mean the outer column).

Without a declared key covered by such equalities nothing is rewritten: the subquery could find several rows
(an error in the engines), and then no single value is tested.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key
from .ast_utils import FROM_KEY, conjuncts, select_sources

_ALLOWED = frozenset({"expressions", "from", "from_", "where", "comments", "kind", "hint"})
_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
_OPAQUE = (exp.Query, exp.Subquery, exp.AggFunc, exp.Window, exp.Exists, exp.Lambda, exp.Star)


def read_keyed_lookups(
    tree: exp.Expression, schema: dict[str, list[str]] | None, keys: dict[str, list[tuple[str, ...]]] | None
) -> exp.Expression:
    """Rewrite, in place, each test described in the module docstring."""

    if not schema or not keys or not any(True for _ in tree.find_all(exp.Subquery)):
        return tree
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    declared = {key.lower(): [tuple(c.lower() for c in k) for k in ks if k] for key, ks in keys.items()}
    for select in list(tree.find_all(exp.Select)):
        conditions = [select.args.get("where"), select.args.get("having")] + [join.args.get("on") for join in select.args.get("joins") or []]
        for condition in conditions:
            if condition is None:
                continue
            for test in _tests(condition.this if isinstance(condition, (exp.Where, exp.Having)) else condition):
                exists = _lookup(test, known, declared)
                if exists is not None:
                    test.replace(exists)
    return tree


def _tests(node: exp.Expression) -> list[exp.Expression]:
    """The comparisons reached from ``node`` through ``AND``, ``OR`` and parentheses."""

    if isinstance(node, (exp.And, exp.Or)):
        return _tests(node.this) + _tests(node.expression)
    if isinstance(node, exp.Paren):
        return _tests(node.this)
    return [node] if isinstance(node, _COMPARISONS) else []


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _outside(node: exp.Expression, inner: str) -> bool:
    """``node`` reads only columns qualified with a name other than ``inner``, and nothing opaque."""

    if any(isinstance(n, _OPAQUE) for n in node.walk()):
        return False
    return all(column.table and column.table.lower() != inner for column in node.find_all(exp.Column))


def _lookup(test: exp.Expression, known: dict[str, list[str]], declared: dict[str, list[tuple[str, ...]]]) -> exp.Exists | None:
    for mine, other, left in ((test.this, test.expression, True), (test.expression, test.this, False)):
        subquery = _unwrap(mine)
        if not isinstance(subquery, exp.Subquery) or any(v for k, v in subquery.args.items() if k not in ("this", "comments")):
            continue
        select = subquery.this
        if not isinstance(select, exp.Select) or any(v for k, v in select.args.items() if k not in _ALLOWED):
            continue
        sources = select_sources(select)
        where = select.args.get("where")
        if len(sources) != 1 or len(select.expressions) != 1 or where is None:
            continue
        table = sources[0]
        columns = _base_table(table, known)
        inner = (table.alias_or_name or "").lower()
        if columns is None or not inner:
            continue
        item = select.expressions[0]
        output = item.this if isinstance(item, exp.Alias) else item
        if any(isinstance(n, _OPAQUE) for n in output.walk()) or not _outside(other, inner):
            continue
        fixed: set[str] = set()
        for part in conjuncts(where.this):
            part = _unwrap(part)
            if not isinstance(part, exp.EQ):
                continue
            for column, value in ((part.this, part.expression), (part.expression, part.this)):
                column = _unwrap(column)
                if not isinstance(column, exp.Column) or column.name.lower() not in columns or (column.table and column.table.lower() != inner):
                    continue
                if _outside(value, inner):
                    fixed.add(column.name.lower())
        if not any(set(key) <= fixed for key in declared.get(_table_key(table), [])):
            continue
        value = output.copy()
        value = exp.Paren(this=value) if isinstance(value, (exp.Binary, exp.Unary, exp.Case)) and not isinstance(value, exp.Paren) else value
        mirror = other.copy()
        comparison = type(test)(this=value, expression=mirror) if left else type(test)(this=mirror, expression=value)
        probe = select.copy()
        probe.set("expressions", [exp.Literal.number(1)])
        kept = probe.args["where"].this
        kept = exp.Paren(this=kept) if isinstance(kept, exp.Or) else kept
        probe.set("where", exp.Where(this=exp.And(this=kept, expression=comparison)))
        return exp.Exists(this=probe)
    return None
