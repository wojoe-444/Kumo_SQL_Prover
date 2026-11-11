"""A global aggregate's ``WHERE`` read as part of each aggregate's argument, for the SMT prover's block matching.

``SELECT SUM(x) FROM t WHERE p`` and ``SELECT SUM(IF(p, x, NULL)) FROM t`` return the same single row: a query
with aggregates and no ``GROUP BY`` returns one row whatever its filter keeps, and ``SUM``, ``MIN``, ``MAX``,
``AVG``, ``COUNT`` and ``COUNTIF`` skip a NULL argument, so a row the filter drops and a row whose argument is NULL
add the same nothing. The prover matches two aggregate blocks only when their filters agree row by row, so it
left this pair unknown.

:func:`pushed` gives the block an equivalent spelling with no filter: every aggregate argument is NULL on the rows
the filter drops (``COUNT(*)`` counts a value that is NULL exactly there). Two global blocks that do not match as
written are compared again in that spelling. Only global aggregates qualify: with a ``GROUP BY`` the filter also
decides which groups exist. An aggregate that does not skip NULLs (``ARRAY_AGG``, ``STRING_AGG`` with its
separators, ``ANY_VALUE``), or a block with a subquery, is left as it is.

What the rewrite changes is which rows an operation is evaluated on, not the rows returned: the error comparison
is made on the queries as compiled, not on this spelling.
"""

from __future__ import annotations

import dataclasses

from . import solver_lock

z3 = solver_lock.z3

SKIP_NULL = ("SUM", "MIN", "MAX", "AVG", "COUNT", "COUNTIF")


def pushed(block, value_sort, const_pred, val):
    """``block`` (a global ``_Agg``) with its filter folded into the aggregate arguments, or ``None``.

    ``value_sort``, ``const_pred`` and ``val`` are the SMT prover's value sort, constant predicate and value
    constructor (passed in: this module is imported by ``smt_equivalence``).
    """

    if not block.is_global or block.keys or block.subs or not block.aggs:
        return None
    if any(call.func not in SKIP_NULL for call in block.aggs):
        return None
    kept = block.cond.t
    if z3.is_true(z3.simplify(kept)):
        return block
    dropped = z3.Not(kept)
    one = value_sort.Num(1)
    calls = []
    for call in block.aggs:
        if call.arg is None:  # COUNT(*): count a value that is NULL on the rows the filter drops
            argument = val(dropped, one)
        else:
            argument = val(z3.Or(call.arg.null, dropped), call.arg.val)
        calls.append(dataclasses.replace(call, arg=argument))
    return dataclasses.replace(block, cond=const_pred(True), aggs=calls)
