"""A second attempt without the shape rules, for a pair they do not prove.

A normalization rule is an identity: the rewritten query returns what the written one does. Sound is not the same
as harmless, though. The prover compares two normal forms, and a rule that fires on one query of a pair and not on
the other moves that side away from a form the two shared before. Each of these cost a proof that held without
the rule:

* a ``WHERE`` pushed into the derived table that wraps a ``FULL JOIN``, when the other query filters above a join
  the rule does not read (three engine test pairs);
* a ``UNION`` of provably disjoint branches written as ``UNION ALL``, after which a later step declines the
  derived table for its unnamed columns, on a pair whose two texts are the same (one DLBench pair);
* ``HAVING COUNT(k) > 0`` read as ``k IS NOT NULL``, when the other query spells the count another way.

Each was fixed where it was found, but the next one is only found by rerunning every benchmark. So the rules
listed below (the batch of 2026-11-10) are *shape rules*: the normalizer applies them on the first attempt, and
when that attempt ends without a verdict the pair is tried again with all of them off, which is the prover as
it was before them. A proof or a refutation from the second attempt stands; anything else leaves the first
answer. This is the arrangement ``algebraic_equivalence.normalize`` already uses for its keyed-``DISTINCT``
levels and for ``semijoin_rules.semijoin_reading``.

What the second attempt costs: it runs only for a pair with no verdict, and each of its steps is skipped when
both queries normalize to the same text with the rules off as with them on (nothing to try). What it guarantees:
with the rules off the normalizer takes exactly the steps it took before the batch, so a verdict that was
reachable then is reachable now, up to the solver's time limit.

The shape rules: ``single_source_columns``, ``grouped_self_membership``, ``disjoint_union``,
``tailed_grouping_sets``, ``mysql_exists_casts``, ``case_arm_rules``, ``unmatched_outer_join``,
``having_count_rules``, ``semijoin_shapes``, ``string_spelling_rules``, ``tailed_outer_aggregates``,
``wrapped_join_filters``, ``one_row_tails``, and the derived-table and ``JOIN .. ON`` sources that
``closed_sources`` lets the scalar-aggregate decorrelation read. Not in the list, because each only reads a
query that was declined before it: ``listed_output_names`` and ``printed_spelling``.

A new rule that rewrites queries the prover already read belongs in the list: guard its entry in ``normalize``
with ``shape_fallback.enabled()``.

The switch is a context variable, so it is per thread and per task and is restored when the attempt ends;
proofs the second attempt starts for parts of the pair run with the rules off too.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_ENABLED: ContextVar[bool] = ContextVar("kumo_sql_prover_shape_rules", default=True)

SKIPPED = "the shape rules change neither query (nothing to retry)"


def enabled() -> bool:
    """Whether the normalizer applies the shape rules in the current attempt."""

    return _ENABLED.get()


@contextmanager
def disabled():
    """Run the block with the shape rules off."""

    token = _ENABLED.set(False)
    try:
        yield
    finally:
        _ENABLED.reset(token)
