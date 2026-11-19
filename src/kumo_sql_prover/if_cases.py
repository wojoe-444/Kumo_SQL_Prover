"""``IF(p, a, b)`` written as ``CASE WHEN p THEN a ELSE b END``.

    SUM(IF(paid = due, 1, 0))          -- SUM(CASE WHEN paid = due THEN 1 ELSE 0 END)

**Why it holds.** ``IF(p, a, b)`` returns ``a`` when ``p`` is TRUE and ``b`` when it is FALSE or NULL, which is the
definition of the searched ``CASE`` with one ``WHEN`` and an ``ELSE``; without a third argument both return NULL where
``p`` is not TRUE. MySQL, BigQuery, DuckDB, Spark and SQLite (``IIF``) define the function that way, and each derives
the result type from the two results by the rule it uses for ``CASE``.

The solver already reads the two alike. The rules that recognise a sum of ones (``conditional_count_sums``,
``counted_presence``) read the ``CASE`` spelling only, so a grouped ``SUM(IF(p, 1, 0)) / COUNT(*)`` did not meet
``SUM(CASE WHEN p THEN 1 ELSE 0 END) / COUNT(*)``.

**What is read.** Every ``IF`` call (not the arms of a ``CASE``, which the parser also keeps as ``If`` nodes) with a
condition and a first result. Nothing else changes.

A pre-pass of the tenth hardest-cases batch (``shape_fallback.TENTH_BATCH``). ``tests/test_if_cases.py`` replays the
rewrite on DuckDB and SQLite.
"""

from __future__ import annotations

from sqlglot import exp


def write_if_calls_as_cases(tree: exp.Expression) -> tuple[exp.Expression, bool]:
    """``tree`` with each ``IF(p, a, b)`` as a searched ``CASE`` (module doc); whether anything changed."""

    if not any(type(node) is exp.If and not isinstance(node.parent, exp.Case) for node in tree.find_all(exp.If)):
        return tree, False
    tree = tree.copy()
    changed = False
    for call in reversed([node for node in tree.find_all(exp.If) if type(node) is exp.If and not isinstance(node.parent, exp.Case)]):
        condition, result, other = call.args.get("this"), call.args.get("true"), call.args.get("false")
        if condition is None or result is None or any(v for key, v in call.args.items() if key not in ("this", "true", "false")):
            continue
        case = exp.Case(ifs=[exp.If(this=condition.copy(), true=result.copy())])
        if other is not None:
            case.set("default", other.copy())
        if call is tree:
            tree = case
        else:
            call.replace(case)
        changed = True
    return tree, changed
