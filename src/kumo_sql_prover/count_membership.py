"""A correlated ``COUNT`` compared with a number, read as membership in the groups that pass the comparison.

    SELECT DISTINCT p.email FROM person AS p
    WHERE (SELECT COUNT(*) FROM person AS q WHERE q.email = p.email) > 1

keeps the rows whose address occurs more than once, and so does

    SELECT DISTINCT p.email FROM person AS p
    WHERE p.email IN (SELECT q.email FROM person AS q GROUP BY q.email HAVING COUNT(*) > 1)

The prover does not read a scalar subquery that counts (a count over no rows is 0, not NULL, so it cannot be
decorrelated like ``MIN`` or ``SUM``), and the pair went undecided. The second spelling is one it reads.

**Why the two are the same test.** Fix an outer row and let ``v`` be the values it is correlated on. Let ``M`` be
the rows of the counted table that pass the subquery's own filters and have ``x = v`` TRUE on every correlated
column (no row when a ``v`` is NULL).

* The scalar subquery has no ``GROUP BY``, so it returns exactly one row: ``COUNT(..)`` over ``M``. The test is
  ``COUNT(M) cmp n``.
* The rule only reads a comparison that is FALSE for a count of 0 (``> n``; ``>= n`` and ``= n`` with
  ``n >= 1``; ``<> 0``). So the test is TRUE only when ``M`` has a row. Then every correlated value is not NULL,
  and ``M`` is exactly the group with ``x = v`` of ``.. WHERE filters GROUP BY x``: the same rows, hence the same
  count, hence the same ``HAVING COUNT(..) cmp n``. The ``IN`` is TRUE exactly when some group that passes the
  ``HAVING`` has ``x = v`` TRUE on every column, and only the group of ``M`` can (a group with a NULL ``x`` equals
  nothing). So the ``IN`` is TRUE exactly when the test is.
* When the test is not TRUE the ``IN`` is FALSE or NULL. The rewrite is made only for a top-level ``AND`` part of
  a ``WHERE``, where a row is dropped either way.
* A subquery that is itself grouped by exactly the correlated columns (``.. WHERE q.x = p.x GROUP BY q.x``)
  returns the one group of ``M``, or no row at all when ``M`` is empty; the scalar is then NULL and no comparison
  is TRUE. Every comparison with a number is read for that spelling, 0 included.

**What is read.** ``(subquery) cmp n`` or ``n cmp (subquery)`` with ``n`` a non-negative integer literal and
``cmp`` one of ``=``, ``<>``, ``<``, ``<=``, ``>``, ``>=``, as a top-level ``AND`` part of a ``WHERE``. The
subquery is one select over one base table with supplied columns that outputs one plain ``COUNT`` (``*``, a
constant, an expression over the table's columns, with or without ``DISTINCT``) and has a ``WHERE`` whose
``AND`` parts are

* ``x = outer`` (either way round): ``x`` a column of the counted table, ``outer`` a column that is not the
  counted table's (qualified by another name, or a bare name the table does not have), at least once;
* a filter that reads only the counted table's columns and holds no subquery, window or aggregate.

The outer columns move from the subquery's ``WHERE`` to the left of the ``IN``, one level up, into the select
whose ``WHERE`` holds the comparison. A name that was not the counted table's was already looked up from that
select outwards, so it names the same column there.

**What is refused.** A comparison that a count of 0 passes (``< 2``, ``= 0``, ``>= 0``, ``<> 1``): the rows with
no match would be kept by the count and dropped by the membership. A comparison under ``NOT`` or ``OR``, in a
select list or in a ``HAVING``. A correlation that is not a plain equality, an outer column inside the ``COUNT``
or a filter, a join, ``HAVING``, ``DISTINCT``, ``LIMIT`` or ``ORDER BY`` in the subquery, an expression around the
``COUNT``.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _unwrap
from .ast_utils import FROM_KEY, conjuncts, extended_grouping, select_sources

_SELECT_ARGS = frozenset({"expressions", "from", "from_", "where", "group", "comments"})
_COMPARISONS = {
    exp.EQ: lambda count, n: count == n,
    exp.NEQ: lambda count, n: count != n,
    exp.LT: lambda count, n: count < n,
    exp.LTE: lambda count, n: count <= n,
    exp.GT: lambda count, n: count > n,
    exp.GTE: lambda count, n: count >= n,
}
_MIRRORED = {exp.EQ: exp.EQ, exp.NEQ: exp.NEQ, exp.LT: exp.GT, exp.LTE: exp.GTE, exp.GT: exp.LT, exp.GTE: exp.LTE}


def read_count_membership(tree: exp.Expression, schema: dict[str, list[str]] | None) -> exp.Expression:
    """Rewrite, in place, each comparison described in the module docstring."""

    if not schema or not any(True for _ in tree.find_all(exp.Count)):
        return tree
    known = {name.lower(): [c.lower() for c in columns] for name, columns in schema.items()}
    for node in list(tree.find_all(*_COMPARISONS)):
        if type(node) not in _COMPARISONS or not _where_part(node):
            continue
        replacement = _membership(node, known)
        if replacement is not None:
            node.replace(replacement)
    return tree


def _where_part(node: exp.Expression) -> bool:
    """Whether ``node`` is a top-level ``AND`` part of a select's ``WHERE``."""

    child, parent = node, node.parent
    while isinstance(parent, (exp.Paren, exp.And)):
        child, parent = parent, parent.parent
    return isinstance(parent, exp.Where) and parent.this is child and isinstance(parent.parent, exp.Select)


def _number(node: exp.Expression) -> int | None:
    node = _unwrap(node)
    if isinstance(node, exp.Literal) and not node.is_string and node.this.isdigit():
        return int(node.this)
    return None


def _membership(node: exp.Expression, known: dict[str, list[str]]) -> exp.Expression | None:
    left, right = _unwrap(node.this), _unwrap(node.expression)
    kind = type(node)
    if isinstance(right, exp.Subquery) and _number(left) is not None:
        left, right, kind = right, left, _MIRRORED[kind]
    number = _number(right)
    if not isinstance(left, exp.Subquery) or number is None:
        return None
    if any(value for key, value in left.args.items() if key not in ("this", "comments")):
        return None
    inner = left.this
    if not isinstance(inner, exp.Select) or any(value for key, value in inner.args.items() if key not in _SELECT_ARGS):
        return None
    sources = select_sources(inner)
    if len(sources) != 1 or len(inner.expressions) != 1:
        return None
    table = sources[0]
    columns = _base_table(table, known)
    own = table.alias_or_name.lower() if columns is not None else ""
    if not own:
        return None

    def own_column(node: exp.Expression) -> bool:
        if not isinstance(node, exp.Column) or isinstance(node.this, exp.Star) or node.args.get("db") or node.args.get("catalog"):
            return False
        return node.name.lower() in columns and (not node.table or node.table.lower() == own)

    def outer_column(node: exp.Expression) -> bool:
        if not isinstance(node, exp.Column) or isinstance(node.this, exp.Star) or node.args.get("db") or node.args.get("catalog"):
            return False
        return node.table.lower() != own if node.table else node.name.lower() not in columns

    def local(node: exp.Expression) -> bool:
        """An expression over the counted table's columns alone."""

        for walked in node.walk():
            if isinstance(walked, (exp.Query, exp.Subquery, exp.Window, exp.AggFunc, exp.Star, exp.Lambda)):
                return False
            if isinstance(walked, exp.Column) and not own_column(walked):
                return False
        return True

    # the one output: a plain COUNT over the table's own columns
    output = inner.expressions[0]
    count = _unwrap(output.this if isinstance(output, exp.Alias) else output)
    if type(count) is not exp.Count or any(value for key, value in count.args.items() if key not in ("this", "big_int")):
        return None
    counted = count.this
    if isinstance(counted, exp.Distinct):
        if counted.args.get("on") or len(counted.expressions) != 1 or not local(counted.expressions[0]):
            return None
    elif not isinstance(counted, exp.Star) and not local(counted):
        return None
    # the WHERE: correlations and local filters
    where = inner.args.get("where")
    if where is None:
        return None
    pairs: list[tuple[exp.Column, exp.Column]] = []  # (the table's column, the outer column)
    filters: list[exp.Expression] = []
    for part in conjuncts(where.this):
        if type(part) is exp.EQ:
            a, b = _unwrap(part.this), _unwrap(part.expression)
            if own_column(a) and outer_column(b):
                pairs.append((a, b))
                continue
            if own_column(b) and outer_column(a):
                pairs.append((b, a))
                continue
        if not local(part):
            return None
        filters.append(part)
    if not pairs:
        return None
    correlated = list(dict.fromkeys(column.name.lower() for column, _ in pairs))
    group = inner.args.get("group")
    if group is not None:
        if extended_grouping(group) or any(value for key, value in group.args.items() if key != "expressions"):
            return None
        grouped = [_unwrap(item) for item in group.expressions]
        if not all(own_column(item) for item in grouped) or {item.name.lower() for item in grouped} != set(correlated):
            return None
    elif _COMPARISONS[kind](0, number):
        return None  # a count of 0 passes: rows without a match are kept
    # --- rewrite ---
    tests = [exp.Paren(this=part.copy()) if isinstance(part, exp.Or) else part.copy() for part in filters]
    groups = exp.Select(expressions=[exp.column(column.name.lower(), table=table.alias_or_name) for column, _ in pairs])
    groups.set(FROM_KEY, exp.From(this=table.copy()))
    if tests:
        condition = tests[0]
        for test in tests[1:]:
            condition = exp.And(this=condition, expression=test)
        groups.set("where", exp.Where(this=condition))
    groups.set("group", exp.Group(expressions=[exp.column(name, table=table.alias_or_name) for name in correlated]))
    groups.set("having", exp.Having(this=kind(this=count.copy(), expression=exp.Literal.number(number))))
    outers = [outer.copy() for _, outer in pairs]
    operand = outers[0] if len(outers) == 1 else exp.Tuple(expressions=outers)
    return exp.In(this=operand, query=exp.Subquery(this=groups))
