"""A second spelling of a grouped aggregate, tried only for a pair the prover did not prove as normalized.

``padded_sums``: under a plain ``GROUP BY``, ``COALESCE(SUM(x), 0)`` is ``SUM(COALESCE(x, 0))`` (the module states
the identity, its proof and its conditions).

It is not an entry of ``algebraic_equivalence.normalize``. A normalization changes the spelling on every pair, and
a pair that is proved today because both queries spell the total one way (or because a rewriter's
``CASE WHEN SUM(x) IS NULL THEN 0 ..`` is compared with ``COALESCE(SUM(x), 0)`` over the same sum) must stay proved.
``respell`` is applied to the two normalized texts after the solver has failed on them, and the solver is run once
more on the result: a pair proved before is proved by the first run, exactly as before. The texts are not
normalized again, and nothing is proved here: the rewrite is the identity of ``padded_sums``, applied to both texts.

(The grouped ``SUM(CASE WHEN c THEN 1 ELSE 0 END)`` read as a count is a normalization,
``conditional_count_sums``.)
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from .ast_utils import faithful_sql
from .padded_sums import pad_grouped_sums


def _respelled(sql: str, dialect: str) -> str:
    tree = sqlglot.parse_one(sql, read=dialect)
    if not any(type(node.this) is exp.Sum for node in tree.find_all(exp.Coalesce)) or not any(tree.find_all(exp.Group)):
        return sql

    def step(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Select):
            return node
        return pad_grouped_sums(node) or node

    changed = tree.transform(step)
    return faithful_sql(changed, dialect) if changed.sql() != tree.sql() else sql


def respell(left_sql: str, right_sql: str, *, dialect: str) -> tuple[str, str] | None:
    """The two queries with their grouped aggregates in the second spelling, or None when neither changes."""

    try:
        texts = _respelled(left_sql, dialect), _respelled(right_sql, dialect)
    except Exception:  # noqa: BLE001 - a text that cannot be parsed or printed faithfully is left as it is
        return None
    return texts if texts != (left_sql, right_sql) else None
