"""Under a GROUP BY, summing 1 for the rows that pass a test and 0 for the others is counting the rows that pass.

    SELECT k, SUM(CASE WHEN c THEN 1 ELSE 0 END) FROM t GROUP BY k
    is
    SELECT k, COUNT(CASE WHEN c THEN 1 END) FROM t GROUP BY k

Both spellings of a conditional count are common, and a rewrite that changes one into the other inside a grouped
derived table left two tables the prover compared as different.

Why it holds: every group of a ``GROUP BY`` has at least one row. Each row adds 1 when ``c`` is true and 0 when it is
false or NULL, so the sum is never NULL and is the number of rows with ``c`` true. ``COUNT`` of the ``CASE`` counts
the rows where the ``CASE`` is not NULL, the same rows. Duplicate rows are counted once each by both.

The grouping is what makes it hold. Over no rows the sum is NULL and the count is 0, so the rule leaves alone:

* a select without ``GROUP BY`` (a global aggregate reads an empty input as one row), and a ``GROUP BY`` with
  ``ROLLUP``, ``CUBE`` or ``GROUPING SETS`` (the grand total of an empty input is a row too);
* a sum with ``FILTER`` (the filter can leave a group no rows), a ``DISTINCT`` sum, and a sum used as a window
  function (its frame can be empty);
* a ``CASE`` with an operand, several branches, no ``ELSE``, or values other than the integers 1 and 0
  (``SUM(CASE WHEN c THEN 1 END)`` is NULL for a group where ``c`` is never true).

Every such sum of the select is rewritten: in its select list, ``HAVING`` and ``ORDER BY``. Sums of a subquery or of
a derived table belong to that select. The rule is not a normalization: ``grouped_aggregate_spellings`` applies it to
a pair the prover did not prove as written, so that a pair proved with the sum on both sides is read as before.
"""

from __future__ import annotations

from sqlglot import exp


def _is(node: exp.Expression | None, value: str) -> bool:
    return isinstance(node, exp.Literal) and not node.is_string and node.this == value


def _test_of(total: exp.Expression) -> exp.Expression | None:
    """``c`` of ``SUM(CASE WHEN c THEN 1 ELSE 0 END)``, or None for any other expression."""

    if type(total) is not exp.Sum or any(value for key, value in total.args.items() if key != "this"):
        return None
    case = total.this
    if type(case) is not exp.Case or case.args.get("this") is not None or len(case.args.get("ifs") or []) != 1:
        return None
    branch = case.args["ifs"][0]
    if not _is(branch.args.get("true"), "1") or not _is(case.args.get("default"), "0"):
        return None
    return branch.this


def count_summed_cases(select: exp.Select) -> exp.Select | None:
    """``select`` with its own ``SUM(CASE WHEN c THEN 1 ELSE 0 END)`` read as ``COUNT(CASE WHEN c THEN 1 END)``, or None."""

    if not isinstance(select, exp.Select):
        return None
    group = select.args.get("group")
    if group is None or not group.expressions or any(value for key, value in group.args.items() if key != "expressions"):
        return None
    if any(isinstance(node, (exp.Rollup, exp.Cube, exp.GroupingSets, exp.Tuple)) for node in group.walk()):
        return None  # sqlglot keeps these among the grouping expressions; ``GROUP BY ()`` is one row over no rows
    found = [
        index
        for index, node in enumerate(select.find_all(exp.Sum))
        if node.find_ancestor(exp.Select) is select and not isinstance(node.parent, (exp.Window, exp.Filter)) and _test_of(node) is not None
    ]
    if not found:
        return None
    result = select.copy()
    sums = list(result.find_all(exp.Sum))
    for index in found:
        test = _test_of(sums[index])
        sums[index].replace(exp.Count(this=exp.Case(ifs=[exp.If(this=test.copy(), true=exp.Literal.number(1))]), big_int=True))
    return result
