"""A subquery test beside a weaker or stronger copy of itself.

    SELECT * FROM l WHERE l.a IN (SELECT c FROM r) OR l.a IN (SELECT c FROM r WHERE l.b < r.d)
    SELECT * FROM l WHERE l.a IN (SELECT c FROM r)

The second test reads fewer rows of ``r`` than the first, so whenever it is TRUE the first is too, and the ``OR``
is the first test alone. Spark's subquery tests are written this way, and the provers match neither side to the
other ("no row-preserving mapping").

SQL's three truth values are ordered FALSE < NULL < TRUE; ``OR`` is the larger of its operands and ``AND`` the
smaller. So when a test ``b`` is never above a test ``a`` (``b <= a`` on every row of every database), ``a OR b`` is
``a`` and ``a AND b`` is ``b``, exactly, NULLs included, wherever the expression stands. This module knows ``b <= a``
in three cases, where "narrower" means the same ``FROM`` and joins (the same text) and a ``WHERE`` whose top-level
conjuncts include every conjunct of the wider one (a row that passes all of them passes the fewer):

* ``EXISTS (narrower)  <=  EXISTS (wider)``. A row of the narrower subquery is a row of the wider.
* ``x IN (SELECT e FROM narrower)  <=  x IN (SELECT e FROM wider)`` for the same ``x`` and ``e``. TRUE on the left
  is a matching row, which the wider subquery has too; NULL on the left (a NULL ``x`` or a NULL ``e`` and no match)
  leaves the right NULL or TRUE.
* ``NOT p  <=  NOT q`` when ``q <= p``: ``NOT`` turns the order around.

Each ``OR`` and each ``AND`` is read as its flat list of operands; an operand that is below another in an ``OR``, or
above another in an ``AND``, is dropped (of two that say the same, the later one).

Declined: a subquery that is not a plain select (``DISTINCT``, grouping, ``HAVING``, ``QUALIFY``, ``ORDER BY``,
``LIMIT``, a set operation, a CTE), an aggregate or window in its select list (``EXISTS (SELECT COUNT(*) ..)`` is
always TRUE), an ``IN`` whose subquery selects more or less than one expression, anything that may give another
value when read again (``RAND()``, an unknown function, ``TABLESAMPLE``), and operands that are not these tests.
"""

from __future__ import annotations

from sqlglot import exp

FROM_KEY = "from_"
_PLAIN = ("expressions", FROM_KEY, "joins", "where")
_REREAD = (exp.Rand, exp.Randn, exp.Randstr, exp.Uuid, exp.Anonymous, exp.TableSample)
_COUNTS_ROWS = (exp.AggFunc, exp.Window)


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _operands(node: exp.Expression, kind) -> list[exp.Expression]:
    node = _unwrap(node)
    if type(node) is kind:
        return _operands(node.this, kind) + _operands(node.expression, kind)
    return [node]


def _reading(node: exp.Expression):
    """``(negated, form, select)`` of a test this module reads, where ``form`` says what must be the same in two
    tests for the comparison of their subqueries to mean anything; None for any other expression."""

    negated = False
    node = _unwrap(node)
    while isinstance(node, exp.Not):
        negated, node = not negated, _unwrap(node.this)
    if isinstance(node, exp.Exists) and not any(v for k, v in node.args.items() if k != "this"):
        select, form = node.this, ("exists",)
    elif isinstance(node, exp.In) and not any(v for k, v in node.args.items() if k not in ("this", "query")) and isinstance(node.args.get("query"), exp.Subquery):
        query = node.args["query"]
        if any(v for k, v in query.args.items() if k != "this"):
            return None
        select = query.this
        if not isinstance(select, exp.Select) or len(select.expressions) != 1 or isinstance(select.expressions[0], exp.Star):
            return None
        form = ("in", node.this.sql(), select.expressions[0].unalias().sql())
    else:
        return None
    if not isinstance(select, exp.Select) or any(v for k, v in select.args.items() if k not in _PLAIN) or not select.expressions:
        return None
    if any(item.find(*_COUNTS_ROWS) is not None for item in select.expressions) or node.find(*_REREAD) is not None:
        return None
    source = (select.args[FROM_KEY].sql() if select.args.get(FROM_KEY) is not None else "", tuple(j.sql() for j in select.args.get("joins") or []))
    where = select.args.get("where")
    conjuncts = frozenset(part.sql() for part in _operands(where.this, exp.And)) if where is not None else frozenset()
    return negated, (*form, source), conjuncts


def _below(low, high) -> bool:
    """Whether the test read as ``low`` is never above the test read as ``high``."""

    if low is None or high is None or low[0] != high[0] or low[1] != high[1]:
        return False
    narrower, wider = (high[2], low[2]) if low[0] else (low[2], high[2])  # under NOT the wider subquery is the lower test
    return narrower >= wider


def absorb_subquery_tests(tree: exp.Expression) -> exp.Expression:
    """``tree`` with each ``OR`` / ``AND`` operand dropped that another operand makes redundant (see the module doc)."""

    if not any(True for _ in tree.find_all(exp.Exists, exp.In)):
        return tree
    for kind in (exp.Or, exp.And):
        tops = [n for n in tree.find_all(kind) if type(n) is kind and type(_parent(n)) is not kind]
        for top in tops:
            operands = _operands(top, kind)
            readings = [_reading(o) for o in operands]
            if sum(r is not None for r in readings) < 2:
                continue
            dropped: set[int] = set()
            for i, mine in enumerate(readings):
                for j, other in enumerate(readings):
                    if i == j or j in dropped or i in dropped or mine is None or other is None:
                        continue
                    # in an OR the lower operand goes, in an AND the higher one
                    redundant = _below(mine, other) if kind is exp.Or else _below(other, mine)
                    if redundant and not (i < j and (_below(other, mine) if kind is exp.Or else _below(mine, other))):
                        dropped.add(i)
            if not dropped:
                continue
            kept = [o for i, o in enumerate(operands) if i not in dropped]
            rebuilt = kept[0].copy()
            for operand in kept[1:]:
                rebuilt = kind(this=rebuilt, expression=_guard(operand.copy(), kind))
            rebuilt = _guard(rebuilt, kind) if len(kept) == 1 else rebuilt
            if top is tree:
                tree = rebuilt
            else:
                top.replace(rebuilt)
    return tree


def _parent(node: exp.Expression) -> exp.Expression | None:
    parent = node.parent
    while isinstance(parent, exp.Paren):
        parent = parent.parent
    return parent


def _guard(node: exp.Expression, kind) -> exp.Expression:
    """``node`` in parentheses when it is an ``OR`` or ``AND`` of the other kind, so the printed text keeps its reading."""

    return exp.Paren(this=node) if isinstance(node, (exp.Or, exp.And)) and type(node) is not kind else node
