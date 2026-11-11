"""``(SELECT MIN(TRUE) FROM t WHERE ..) IS NOT NULL`` asks whether a row exists.

    SELECT e.empno FROM emp AS e WHERE NOT (SELECT MIN(TRUE) FROM dept AS d WHERE d.deptno = e.deptno) IS NULL

Calcite writes an ``EXISTS`` this way when it has turned the subquery into a lateral join, and the prover had no
reading for a scalar subquery of that form. The rule rewrites the test to the ``EXISTS`` it stands for:

    SELECT e.empno FROM emp AS e WHERE EXISTS(SELECT 1 FROM dept AS d WHERE d.deptno = e.deptno)

**Why it is the same.** A select with an aggregate and no ``GROUP BY`` returns exactly one row, so the scalar
subquery has a value for every outer row. ``MIN`` or ``MAX`` of a constant that is not NULL is that constant
when the ``FROM`` and ``WHERE`` keep at least one row and NULL when they keep none. So ``.. IS NULL`` is TRUE
exactly when no row is kept and FALSE otherwise, which is ``NOT EXISTS`` over the same ``FROM`` and ``WHERE``;
neither side is ever unknown. This holds wherever the test stands (a filter, a join condition, an output).

**What is read.** ``<scalar subquery> IS NULL``, with or without ``NOT`` around it, where the subquery is one
select with a ``FROM``, one output that is ``MIN`` or ``MAX`` of a literal (TRUE, FALSE, a number or a string),
and nothing but ``FROM``, joins and ``WHERE``. Refused: ``GROUP BY`` (no row or many rows), ``HAVING`` and
``LIMIT`` / ``OFFSET`` (they can drop the one row), ``DISTINCT``, ``ORDER BY``, windows, an aggregate of a
column or of NULL (NULL although rows exist), ``COUNT`` (never NULL), more than one output.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY

_ALLOWED = frozenset({"expressions", "from", "from_", "where", "joins", "comments", "kind", "hint"})


def _constant(node: exp.Expression) -> bool:
    while isinstance(node, exp.Paren):
        node = node.this
    return isinstance(node, (exp.Boolean, exp.Literal))


def _existence(test: exp.Expression) -> exp.Exists | None:
    """The ``EXISTS`` whose truth is "the scalar subquery of ``test`` is not NULL"."""

    if not isinstance(test, exp.Is) or not isinstance(test.expression, exp.Null):
        return None
    subquery = test.this
    while isinstance(subquery, exp.Paren):
        subquery = subquery.this
    if not isinstance(subquery, exp.Subquery) or any(v for k, v in subquery.args.items() if k not in ("this", "comments")):
        return None
    select = subquery.this
    if not isinstance(select, exp.Select) or any(v for k, v in select.args.items() if k not in _ALLOWED):
        return None
    if select.args.get(FROM_KEY) is None or len(select.expressions) != 1:
        return None
    item = select.expressions[0]
    call = item.this if isinstance(item, exp.Alias) else item
    while isinstance(call, exp.Paren):
        call = call.this
    if type(call) not in (exp.Min, exp.Max) or any(v for k, v in call.args.items() if k != "this") or not _constant(call.this):
        return None
    probe = select.copy()
    probe.set("expressions", [exp.Literal.number(1)])
    return exp.Exists(this=probe)


def rewrite_aggregate_presence(select: exp.Select) -> exp.Select | None:
    """A copy of ``select`` with each such test in its own scope rewritten, or None when there is none."""

    if not any(isinstance(node.expression, exp.Null) and node.find(exp.Subquery) is not None for node in select.find_all(exp.Is)):
        return None
    copy = select.copy()
    changed = False
    for test in list(copy.find_all(exp.Is)):
        if test.find_ancestor(exp.Select) is not copy:
            continue  # a nested select is rewritten on its own turn
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
