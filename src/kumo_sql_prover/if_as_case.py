"""``IF(test, a, b)`` read as ``CASE WHEN test THEN a ELSE b END`` before the pre-passes.

MySQL answers spell a flag ``SUM(IF(kind = 'x', 1, 0))`` where others write ``SUM(CASE WHEN kind = 'x' THEN 1 ELSE
0 END)``. The rule loop reads both, but the pre-passes that run before it (``child_counts`` and the other readers
of counted flags, pivots and conditional sums) look for the ``CASE`` spelling only, so a pair that needed one of
them stayed unproved when one answer said ``IF``. This pre-pass (hardest-cases batch 29) writes every function
call ``IF(test, a, b)`` as the ``CASE`` before they run.

**Why it is sound.** ``IF(test, a, b)`` is ``a`` when ``test`` is true and ``b`` when it is false or NULL, which is
what ``CASE WHEN test THEN a ELSE b END`` says (MySQL, BigQuery, DuckDB, and SQLite's ``IIF``). Without a third
argument both give NULL. The arms of a ``CASE`` (which the parser also holds as ``If`` nodes) are left as they are.

It runs in the last step of batch 27 only, so a pair an earlier step decides is decided as it was.
"""

from __future__ import annotations

from sqlglot import exp

NAME = "if_as_case"


def read_if_as_case(tree: exp.Expression) -> tuple[exp.Expression, bool]:
    """``(tree with every IF call written as a CASE, whether there was one)``. The input is not changed."""

    if not any(_call(node) for node in tree.find_all(exp.If)):
        return tree, False
    copy = tree.copy()
    while True:
        # one call at a time: a call inside another's argument is found again in the CASE that replaced it
        node = next((node for node in copy.find_all(exp.If) if _call(node)), None)
        if node is None:
            return copy, True
        if set(key for key, value in node.args.items() if value is not None) - {"this", "true", "false"}:
            return tree, False
        case = exp.Case(ifs=[exp.If(this=node.this.copy(), true=node.args["true"].copy())])
        if node.args.get("false") is not None:
            case.set("default", node.args["false"].copy())
        if node is copy:
            copy = case
        else:
            node.replace(case)


def _call(node: exp.Expression) -> bool:
    """Whether an ``If`` node is the function call and not an arm of a ``CASE``."""

    return type(node) is exp.If and not (isinstance(node.parent, exp.Case) and node.arg_key == "ifs") and node.args.get("true") is not None
