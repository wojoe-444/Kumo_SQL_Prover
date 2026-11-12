"""A one-row aggregate that only asks whether a row exists is ``EXISTS``.

    SELECT e.empno FROM emp AS e WHERE NOT (SELECT MIN(TRUE) FROM dept AS d WHERE d.deptno = e.deptno) IS NULL
    SELECT e.empno FROM emp AS e WHERE (SELECT COUNT(*) FROM dept AS d WHERE d.deptno = e.deptno) > 0

Calcite writes an ``EXISTS`` the first way when it has turned the subquery into a lateral join; people write
"none" and "at least one" the second way (``0 = (SELECT COUNT(*) ..)``). The prover had no reading for a scalar
subquery of either form. The rule rewrites each test to the ``EXISTS`` it stands for:

    SELECT e.empno FROM emp AS e WHERE EXISTS(SELECT 1 FROM dept AS d WHERE d.deptno = e.deptno)

**Why it is the same.** A select with an aggregate and no ``GROUP BY`` returns exactly one row, so the scalar
subquery has a value for every outer row.

* ``MIN`` or ``MAX`` of a constant that is not NULL is that constant when the ``FROM`` and ``WHERE`` keep at
  least one row and NULL when they keep none. So ``.. IS NULL`` is TRUE exactly when no row is kept and FALSE
  otherwise, which is ``NOT EXISTS`` over the same ``FROM`` and ``WHERE``.
* ``COUNT(*)`` (or ``COUNT`` of a constant that is not NULL) is the number of rows kept and is never NULL. So
  ``= 0``, ``< 1`` and ``<= 0`` are TRUE exactly when no row is kept (``NOT EXISTS``), and ``> 0``, ``>= 1`` and
  ``<> 0`` exactly when one is (``EXISTS``). ``COUNT(c)`` and ``COUNT(DISTINCT c)`` of a column count the rows
  whose ``c`` is not NULL, so the same holds with ``c IS NOT NULL`` added to the ``WHERE``.

Neither side is ever unknown, so this holds wherever the test stands (a filter, a join condition, an output).

**What is read.** ``<scalar subquery> IS NULL`` (with or without ``NOT`` around it) for the first form, and
``<scalar subquery> <op> <0 or 1>`` with the subquery on either side for the second, where the subquery is one
select with a ``FROM``, the one output described above, and nothing but ``FROM``, joins and ``WHERE``. Refused:
``GROUP BY`` (no row or many rows), ``HAVING`` and ``LIMIT`` / ``OFFSET`` (they can drop the one row),
``DISTINCT`` on the select, ``ORDER BY``, windows, ``MIN`` / ``MAX`` of a column or of NULL (NULL although rows
exist), ``COUNT`` tested ``IS NULL`` or against another number, ``COUNT`` of an expression or of several
columns, more than one output.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY

_ALLOWED = frozenset({"expressions", "from", "from_", "where", "joins", "comments", "kind", "hint"})
# COUNT <op> n that is TRUE exactly when some row is counted (True) or none is (False)
_SOME = {(exp.GT, 0): True, (exp.GTE, 1): True, (exp.NEQ, 0): True, (exp.EQ, 0): False, (exp.LT, 1): False, (exp.LTE, 0): False}
_FLIPPED = {exp.GT: exp.LT, exp.LT: exp.GT, exp.GTE: exp.LTE, exp.LTE: exp.GTE, exp.EQ: exp.EQ, exp.NEQ: exp.NEQ}


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _constant(node: exp.Expression) -> bool:
    return isinstance(_unwrap(node), (exp.Boolean, exp.Literal))


def _one_output(node: exp.Expression) -> tuple[exp.Select, exp.Expression] | None:
    """The select of a scalar subquery of the shape read, and its one output without alias."""

    node = _unwrap(node)
    if not isinstance(node, exp.Subquery) or any(v for k, v in node.args.items() if k not in ("this", "comments")):
        return None
    select = node.this
    if not isinstance(select, exp.Select) or any(v for k, v in select.args.items() if k not in _ALLOWED):
        return None
    if select.args.get(FROM_KEY) is None or len(select.expressions) != 1:
        return None
    item = select.expressions[0]
    return select, _unwrap(item.this if isinstance(item, exp.Alias) else item)


def _probe(select: exp.Select, extra: exp.Expression | None = None) -> exp.Exists:
    probe = select.copy()
    probe.set("expressions", [exp.Literal.number(1)])
    if extra is not None:
        where = probe.args.get("where")
        if where is None:
            probe.set("where", exp.Where(this=extra))
        else:
            kept = exp.Paren(this=where.this) if isinstance(where.this, exp.Or) else where.this
            probe.set("where", exp.Where(this=exp.And(this=kept, expression=extra)))
    return exp.Exists(this=probe)


def _existence(test: exp.Expression) -> exp.Exists | None:
    """The ``EXISTS`` whose truth is "the scalar subquery of ``test`` is not NULL"."""

    if not isinstance(test, exp.Is) or not isinstance(test.expression, exp.Null):
        return None
    found = _one_output(test.this)
    if found is None:
        return None
    select, call = found
    if type(call) not in (exp.Min, exp.Max) or any(v for k, v in call.args.items() if k != "this") or not _constant(call.this):
        return None
    return _probe(select)


def _counted(test: exp.Expression) -> exp.Expression | None:
    """``EXISTS`` or ``NOT EXISTS`` for a one-row ``COUNT`` compared with 0 or 1."""

    if type(test) not in _FLIPPED:
        return None
    for count_side, number, kind in ((test.this, test.expression, type(test)), (test.expression, test.this, _FLIPPED[type(test)])):
        number = _unwrap(number)
        if not isinstance(number, exp.Literal) or number.is_string or number.this not in ("0", "1"):
            continue
        some = _SOME.get((kind, int(number.this)))
        found = _one_output(count_side)
        if some is None or found is None:
            continue
        select, call = found
        if type(call) is not exp.Count or any(v for k, v in call.args.items() if k not in ("this", "big_int")):
            continue
        counted = call.this
        if isinstance(counted, exp.Distinct):
            if len(counted.expressions) != 1 or counted.args.get("on"):
                continue
            counted = counted.expressions[0]
        counted = _unwrap(counted)
        extra = None
        if isinstance(counted, exp.Column) and not isinstance(counted.this, exp.Star):
            extra = exp.Not(this=exp.Is(this=counted.copy(), expression=exp.Null()))
        elif not isinstance(counted, exp.Star) and not _constant(counted):
            continue
        exists = _probe(select, extra)
        return exists if some else exp.Not(this=exists)
    return None


def rewrite_aggregate_presence(select: exp.Select) -> exp.Select | None:
    """A copy of ``select`` with each such test in its own scope rewritten, or None when there is none."""

    if not any(isinstance(node, exp.Subquery) and isinstance(node.parent, (exp.Is, exp.Paren, *_FLIPPED)) for node in select.find_all(exp.Subquery)):
        return None
    copy = select.copy()
    changed = False
    for test in list(copy.find_all(exp.Is, *_FLIPPED)):
        if test.find_ancestor(exp.Select) is not copy:
            continue  # a nested select is rewritten on its own turn
        if not isinstance(test, exp.Is):
            counted = _counted(test)
            if counted is not None:
                test.replace(counted)
                changed = True
            continue
        exists = _existence(test)
        if exists is None:
            continue
        outer = test.parent
        while isinstance(outer, exp.Paren):
            outer = outer.parent
        if isinstance(outer, exp.Not):
            outer.replace(exists)
        else:
            test.replace(exp.Not(this=exists))
        changed = True
    return copy if changed else None
