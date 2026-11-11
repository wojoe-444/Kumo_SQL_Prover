"""Under a GROUP BY, a sum read as 0 when it is NULL is the sum of the column with its NULLs read as 0.

    SELECT k, COALESCE(SUM(x), 0) FROM t GROUP BY k
    is
    SELECT k, SUM(COALESCE(x, 0)) FROM t GROUP BY k

Both spellings are common for a total over a left join, and a rewrite that moves the ``COALESCE`` out of the sum left
two aggregates the prover compared as different. The second form is the one kept: the prover already compares it
with the ``CASE WHEN x IS NULL THEN 0 ELSE x END`` a rewriter prints inside a sum.

Why it holds: every group of a ``GROUP BY`` has at least one row. If some row of the group has ``x`` not NULL, both
forms add up those values (a NULL is skipped in the first and adds 0 in the second). If every ``x`` of the group is
NULL, the first reads ``SUM``'s NULL as 0 and the second adds zeros. Duplicate rows add once each in both.

The grouping is what makes it hold. Over no rows the first form is 0 and the second is NULL, so the rule leaves
alone:

* a select without ``GROUP BY`` (a global aggregate reads an empty input as one row), and a ``GROUP BY`` with
  ``ROLLUP``, ``CUBE``, ``GROUPING SETS`` or ``()`` (the grand total of an empty input is a row too);
* a sum with ``FILTER`` (the filter can leave a group no rows), a ``DISTINCT`` sum, and a sum used as a window
  function (its frame can be empty);
* a ``COALESCE`` with more than two arguments or a second argument other than the integer 0
  (``COALESCE(SUM(x), 1)`` is 1 for a group of NULLs, and ``SUM(COALESCE(x, 1))`` counts them).

Every such sum of the select is rewritten: in its select list, ``HAVING`` and ``ORDER BY``. Sums of a subquery or of
a derived table belong to that select. The rule is not a normalization: ``grouped_aggregate_spellings`` applies it to
a pair the prover did not prove as written, so that a pair proved with ``COALESCE(SUM(x), 0)`` on both sides, or
against a rewriter's ``CASE WHEN SUM(x) IS NULL ..``, is read as before.

Not reached: a grouped derived table that is inner joined. Normalization has by then lifted the ``COALESCE`` out of
it (``COALESCE(t.lifted, 0)`` over ``SUM(x)``), and the sum is no longer under its ``COALESCE``.
"""

from __future__ import annotations

from sqlglot import exp


def _summed(padded: exp.Expression) -> exp.Expression | None:
    """``x`` of ``COALESCE(SUM(x), 0)``, or None for any other expression."""

    if type(padded) is not exp.Coalesce or len(padded.expressions) != 1 or any(value for key, value in padded.args.items() if key not in ("this", "expressions")):
        return None
    total, zero = padded.this, padded.expressions[0]
    if type(total) is not exp.Sum or any(value for key, value in total.args.items() if key != "this") or isinstance(total.this, exp.Distinct):
        return None
    if not isinstance(zero, exp.Literal) or zero.is_string or zero.this != "0":
        return None
    return total.this


def pad_grouped_sums(select: exp.Select) -> exp.Select | None:
    """``select`` with its own ``COALESCE(SUM(x), 0)`` read as ``SUM(COALESCE(x, 0))``, or None."""

    if not isinstance(select, exp.Select):
        return None
    group = select.args.get("group")
    if group is None or not group.expressions or any(value for key, value in group.args.items() if key != "expressions"):
        return None
    if any(isinstance(node, (exp.Rollup, exp.Cube, exp.GroupingSets, exp.Tuple)) for node in group.walk()):
        return None  # sqlglot keeps these among the grouping expressions; ``GROUP BY ()`` is one row over no rows
    found = [index for index, node in enumerate(select.find_all(exp.Coalesce)) if node.find_ancestor(exp.Select) is select and _summed(node) is not None]
    if not found:
        return None
    result = select.copy()
    padded = list(result.find_all(exp.Coalesce))
    for index in found:
        value = _summed(padded[index])
        padded[index].replace(exp.Sum(this=exp.Coalesce(this=value.copy(), expressions=[exp.Literal.number(0)])))
    return result

