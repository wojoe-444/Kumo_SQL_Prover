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

What the second attempt costs: it runs only for a pair with no verdict, and it normalizes both queries again,
which is most of the time an unproven pair takes (the solver is skipped when both normalize to the same text with
the rules off as with them on). Normalization grows faster than the query: measured on undecided QUITE development
pairs, the second attempt adds 0.1 to 0.5 s to a pair of under 1,000 characters, 1 to 2 s at 2,000 to 4,000,
5 to 25 s at 12,000 and up, and on a 36,000-character pair it added about 20 s and 1.7 GB. So it is made only for a
pair whose two texts together are at most ``MAX_CHARS`` characters. The lost proofs above were all under 700.

What it guarantees, for a pair within that size: with the rules off the normalizer takes exactly the steps it
took before the batch, so a verdict that was reachable then is reachable now, up to the solver's time limit.
A larger pair gets the first attempt only.

The shape rules: ``single_source_columns``, ``grouped_self_membership``, ``disjoint_union``,
``tailed_grouping_sets``, ``mysql_exists_casts``, ``case_arm_rules``, ``unmatched_outer_join``,
``having_count_rules``, ``semijoin_shapes``, ``string_spelling_rules``, ``tailed_outer_aggregates``,
``wrapped_join_filters``, ``one_row_tails``, and the derived-table and ``JOIN .. ON`` sources that
``closed_sources`` lets the scalar-aggregate decorrelation read. Added on 2026-11-11 for the DSB rewrites of
the QUITE eval: ``renamed_join_heads``, ``spread_aggregates``, ``scalar_having_groups``,
``star_source_columns``, ``count_column_bounds``, ``extreme_group_sources``, ``setop_operand_sources``,
``empty_union_operands``, ``counted_group_lists``, ``counted_group_sources``, ``existence_joins``,
``integer_expression_bounds``, ``grouped_conditional_counts``, ``pinned_source_columns``, ``refuted_case_arms``,
``single_value_joins`` and the pairing order of ``opaque_pairing`` in the solver. Also added 2026-11-11:
``inline_rows``, ``closed_queries``, ``counted_joins``, ``count_exists_rules``, ``subquery_absorption``,
``typed_comparisons`` and ``sqlite_date_text``. Not in the list, because each only reads a query that was declined
before it: ``listed_output_names`` and ``printed_spelling``.

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
# The two query texts together; see the module docstring for the measurements behind the number.
MAX_CHARS = 4000


def worth_retrying(left_sql: str, right_sql: str) -> bool:
    """Whether a pair the first attempt left without a verdict gets the attempt without the shape rules."""

    return enabled() and len(left_sql) + len(right_sql) <= MAX_CHARS


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
