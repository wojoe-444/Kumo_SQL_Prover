"""The rules of the VeriEQL worker batch as one group of the narrow second attempt.

The batch (counted UNION ALL branches, indicator sums, scalar counts, grouped join keys, own-group joins and
windows, keyed self membership, exact rounding, membership reads of joins, correlated count tests, foreign-key
membership and the rest of the modules listed in ``docs/evals/verieql.md``) rewrites queries the prover already
read. A sound rule can still hide a proof: when it fires on one query of a pair and not on the other, the two
normal forms move apart (two proofs of the whole-corpus runs were lost that way, one of VeriEQL's LeetCode pairs
and one held-out DSB pair of QUITE, each found by switching one rule off at a time).

So every entry point of the batch goes through ``prepass`` (a rule over the whole tree) or ``step`` (a rule of one
select) below. Each records that the group fired (``shape_fallback.note_fired``), and a pair that is left without a
verdict and in which the group fired is tried again with the whole group off (``shape_fallback.without``): the whole
prover, every attempt and every narrow rule of the others, runs again as it was before the batch (the recursive call
in ``_prove_algebraic_levels``). A proof or a refutation from that run stands. The group is one name, so a pair costs
one extra run, not one per rule, and the batch can never take a proof main had, also one that main finds only with
one of its own narrow rules off (the group can hide that one too: ``test_a_pair_main_proves_without_the_rule_is_still_proved``).

What the group does not cover: a proof that needs one rule of the batch and is hidden by another rule of it is
found on the first attempt or not at all.
"""

from __future__ import annotations

from sqlglot import exp

from . import shape_fallback

NAME = "verieql_worker_rules"


def enabled() -> bool:
    """Whether the group may fire in the current attempt."""

    return shape_fallback.rule_enabled(NAME)


def prepass(rule, tree: exp.Expression, *args) -> exp.Expression:
    """``rule(tree, *args)`` (a rule over the whole tree, changing it in place), noting when it changed something."""

    if not enabled():
        return tree
    before = tree.copy()
    result = rule(tree, *args)
    if result != before:
        shape_fallback.note_fired(NAME)
    return result


def step(rule, select: exp.Expression, *args) -> exp.Expression | None:
    """``rule(select, *args)`` (a rule of one select that returns the changed select, or ``None``), noting a change."""

    if not enabled():
        return None
    result = rule(select, *args)
    if result is not None:
        shape_fallback.note_fired(NAME)
    return result
