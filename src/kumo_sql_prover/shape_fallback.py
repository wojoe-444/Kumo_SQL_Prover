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
``wrapped_join_filters``, ``one_row_tails``, ``grouped_source_filters``, ``keyed_union``, ``tailed_regroups``, ``union_join_grouping``, ``own_group_joins``, ``duplicate_self_joins`` and ``counted_presence`` (hardest-cases batch 3, 2026-11-11), ``anti_join_reading``, ``distinct_source_filters``, ``keyed_semijoin``, ``nested_correlation`` and ``presence_given_absence`` (batch 4, 2026-11-16 and 17), ``differing_self_joins``, ``own_value_lists``, ``own_group_membership`` and ``own_key_membership`` (batch 5, 2026-11-17), ``different_row_keys`` (batch 6, 2026-11-17), ``referenced_presence`` (batch 7, 2026-11-17), ``child_counts`` (batch 8, 2026-11-18), ``own_row_copies``, ``distinct_match_joins``, ``own_column_lists``, ``lookup_names``, ``joined_key_lists``, ``own_row_witness``, ``single_value_groups``, ``nested_test_groupings`` and ``distinct_derived_joins`` (batch 9, 2026-11-18), ``shifted_equalities``, ``if_cases``, ``averaged_flags``, ``rounded_percentages``, ``counted_sum_ratios``, ``aggregate_scalars``, ``outer_side_aggregates`` and ``unique_count_arguments`` (batch 10, 2026-11-18), ``boolean_sums``, ``added_sums`` and ``shared_group_names`` (batch 11, 2026-11-19), ``date_windows`` (batch 12, 2026-11-19), ``listed_cte_names``, ``referencing_union_joins``, ``top_row_subqueries`` and ``lifted_aggregates`` (batch 13, 2026-11-20), and the derived-table and ``JOIN .. ON`` sources that
``closed_sources`` lets the scalar-aggregate decorrelation read. Added on 2026-11-11 for the DSB rewrites of
the QUITE eval: ``renamed_join_heads``, ``spread_aggregates``, ``scalar_having_groups``,
``star_source_columns``, ``count_column_bounds``, ``extreme_group_sources``, ``setop_operand_sources``,
``empty_union_operands``, ``counted_group_lists``, ``counted_group_sources``, ``existence_joins``,
``integer_expression_bounds``, ``grouped_conditional_counts``, ``pinned_source_columns``, ``refuted_case_arms``,
``single_value_joins`` and the pairing order of ``opaque_pairing`` in the solver. Also added 2026-11-11:
``inline_rows``, ``closed_queries``, ``counted_joins``, ``count_exists_rules``, ``subquery_absorption``,
``typed_comparisons`` and ``sqlite_date_text``. Not in the list, because each only reads a query that was declined
before it: ``listed_output_names`` and ``printed_spelling``.

The batch of 2026-11-11, written for QUITE's TPC-H and Calcite development pairs, is in the list too:
``anti_join_exists``, ``expanded_not_in``, ``counted_groups``, ``membership_joins``, ``computed_group_outputs``,
``keyed_join_groups``, ``own_group_filters``, ``exclusive_union_filters``, ``one_row_sources``,
``closed_scalar_filters``, ``closed_scalar_outputs``, ``substring_casts``, ``discrete_bounds``,
``cased_group_sums``, ``tuple_membership``, ``counted_outer_groups``, ``guarded_outer_counts``,
``group_total_windows``, and from the Calcite pairs ``keyed_only_grouping``, ``windowed_inputs``,
``plain_grouping_calls``, ``one_row_groups``, ``key_expression_having``, ``extremum_comparisons``,
``tagged_union_groups``, ``conditional_count_sums``, ``constant_window_sums``, ``counted_known_values``,
``grouped_output_names`` and ``self_outer_join_columns``. Several of these prove a pair only together, and the
second attempt switches all of them off at once: a shape rule that fires on one query of a pair and not on the
other loses the proofs the other shape rules gave that pair. Rerun the development pairs after adding one.

From QUITE's SQLStorm development pairs, the same day: ``quoted_names``, ``joined_key_restrictions``,
``timestamp_shifts``, ``equality_orientation`` (passes over the whole query), ``preserved_counts``,
``preserved_join_sources``, ``windowed_output_pruning``, ``never_null_cases`` and ``counted_subqueries`` (rules of
one select; the last also runs as a pass before ``counted_outer_groups``, so that a ``COUNT`` subquery and a left
join to counts written by hand take one road). Three
more modules of that batch are not normalizations. ``window_inputs``, ``derived_table_proofs`` and
``grouped_aggregate_spellings`` run in ``_prove_algebraic`` on the two normalized texts, only on the attempt with
the shape rules on and only after the solver has failed on those texts, so they cannot take a verdict away.

``self_key_membership`` is the other way round, a *late rule*: it runs in a third and a fourth attempt only (the late rules on, first beside the shape rules and then without them),
and only for a pair where it would have changed a query. On the first attempt it took three pairs out of the shape that other rules
prove, and in the second it took two more.

The rules of the hardest-cases batch (``grouped_source_filters``, ``rejected_null_filters``, ``tailed_regroups``, ``union_join_grouping``,
``own_group_joins``, ``duplicate_self_joins`` and ``counted_presence``) are also in the narrow tier below: one of them rewrote the
SUM(CASE ..) > 0 side of a conditional LeetCode pair out of the shape main proves, and the whole second attempt cannot find that
out for a pair that is too large for it.

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


_LATE: ContextVar[bool] = ContextVar("kumo_sql_prover_late_rules", default=False)
_LATE_SEEN: ContextVar[list | None] = ContextVar("kumo_sql_prover_late_seen", default=None)


def late_enabled() -> bool:
    """Whether the normalizer applies the late rules: the ones that run in a third attempt only."""

    return _LATE.get()


def note_late_candidate() -> None:
    """A late rule would change a query: the third attempt is worth making."""

    seen = _LATE_SEEN.get()
    if seen is not None:
        seen.append(True)


@contextmanager
def watching_late():
    """Collect, while the block runs, whether a late rule would have changed a query. Yields the list."""

    seen: list = []
    token = _LATE_SEEN.set(seen)
    try:
        yield seen
    finally:
        _LATE_SEEN.reset(token)


@contextmanager
def late():
    """Run the block with the late rules on."""

    token = _LATE.set(True)
    try:
        yield
    finally:
        _LATE.reset(token)


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


# A narrower second attempt: one rule off, the rest of the shape rules on. The attempt above switches every shape
# rule off at once and, for cost, only for a small pair. A rule that is right for most pairs but hides the match of a
# few larger ones (reading an ``EXISTS`` inside an ``IN`` as a join helps TPC-H Q20 and breaks one DSB pair of
# 12,000 characters) is named here, and the pair is tried again with only that rule off when the rule fired.

NARROW_MAX_CHARS = 24000

_OFF: ContextVar[frozenset] = ContextVar("kumo_sql_prover_rules_off", default=frozenset())
_FIRED: ContextVar[set | None] = ContextVar("kumo_sql_prover_rules_fired", default=None)


def rule_enabled(name: str) -> bool:
    """Whether the named narrow-fallback rule may fire in the current attempt."""

    return name not in _OFF.get()


def off() -> frozenset:
    """The rules switched off for the current attempt (part of any normalization cache key)."""

    return _OFF.get()


def note_fired(name: str) -> None:
    """Record that the named rule rewrote something in the current attempt."""

    fired = _FIRED.get()
    if fired is not None:
        fired.add(name)


@contextmanager
def tracking():
    """Collect the names of the narrow-fallback rules that fire inside the block."""

    fired: set = set()
    token = _FIRED.set(fired)
    try:
        yield fired
    finally:
        _FIRED.reset(token)


@contextmanager
def without(name: str):
    """Run the block with the named narrow-fallback rule off."""

    token = _OFF.set(_OFF.get() | {name})
    try:
        yield
    finally:
        _OFF.reset(token)


# The rules of the ninth hardest-cases batch (2026-11-18) read shapes that older rules also read, and one of them cost a
# proof in an attempt made with an older rule off: with ``own_key_membership`` off, ``distinct_match_joins`` began to fire
# and took the pair out of the shape that attempt used to prove. Switching rules off one at a time cannot find that. So
# these rules fire only on a first attempt and on a narrow attempt that switches off rules of this batch alone; every
# other attempt (without the shape rules, with the late rules, with an older rule off) reads the pair exactly as it was
# read before the batch.
#
# They also cost time. A first attempt that reads a pair differently fails differently, and every failed attempt is
# followed by its retries, also inside the proofs of parts of a pair: six QUITE proofs of about 25 seconds needed more
# than 60 with the batch on from the start. So a proof runs in two stages (``_prove_algebraic_levels``). Stage one is
# the whole prover as it was before the batch, every attempt of it, with the batch off also in the proofs inside it:
# a pair that was proved before is proved by the same steps in the same time. Stage two runs only when stage one
# leaves the pair without a verdict and a rule of the batch would have fired (stage one tries each rule where it
# would fire and throws the result away, until one says yes): the attempts that have the batch on. Outside a proof
# (``normalize`` called by itself) there is no stage and the batch is on.

NINTH_BATCH = frozenset(
    {
        "own_row_copies",
        "distinct_match_joins",
        "own_column_lists",
        "lookup_names",
        "joined_key_lists",
        "own_row_witness",
        "single_value_groups",
        "nested_test_groupings",
        "distinct_derived_joins",
    }
)

# The VeriEQL worker group (``worker_rules.NAME``) is switched off as a whole for a run of the prover as it was before
# that group. The ninth batch was measured without the group, so it may fire beside the group and with the group off.
_BESIDE_NINTH = frozenset({"verieql_worker_rules"})

_NINTH_STAGE: ContextVar[bool | None] = ContextVar("kumo_sql_prover_ninth_stage", default=None)
_NINTH_SEEN: ContextVar[list | None] = ContextVar("kumo_sql_prover_ninth_seen", default=None)
_TENTH_STAGE: ContextVar[bool | None] = ContextVar("kumo_sql_prover_tenth_stage", default=None)
_TENTH_SEEN: ContextVar[list | None] = ContextVar("kumo_sql_prover_tenth_seen", default=None)


# The tenth batch (2026-11-18, shares and percentages) is one stage further out. Stage three runs only when stages one
# and two leave the pair without a verdict and a rule of the tenth batch would have fired (the earlier stages run the
# batch on a copy of the tree where it would fire, until one rule says yes): the first attempt with the ninth and the
# tenth batch on, and a narrow attempt for each rule of the tenth batch that fired, switched off. A pair that was
# proved before the batch is proved by the same steps in the same time, and an attempt with a ninth-batch rule off
# reads the pair without the tenth batch, as it did before it.

TENTH_BATCH = frozenset({"shifted_equalities", "if_cases", "averaged_flags", "rounded_percentages", "counted_sum_ratios", "aggregate_scalars", "outer_side_aggregates", "unique_count_arguments"})


def ninth_batch_enabled() -> bool:
    """Whether the rules of ``NINTH_BATCH`` may fire in the current attempt (see the comment above)."""

    return _NINTH_STAGE.get() is not False and _ENABLED.get() and not _LATE.get() and (_OFF.get() - _BESIDE_NINTH) <= NINTH_BATCH | TENTH_BATCH


def ninth_staged() -> bool:
    """Whether a staged proof is running: a proof started inside it belongs to its stage and makes no stages of its own."""

    return _NINTH_STAGE.get() is not None


def ninth_stage() -> bool:
    """Whether the ninth batch is on in the current stage (part of any normalization cache key)."""

    return _NINTH_STAGE.get() is not False


@contextmanager
def ninth_deferred():
    """Stage one: run the block with the ninth batch off, collecting whether one of its rules would have fired."""

    seen: list = []
    stage, watch = _NINTH_STAGE.set(False), _NINTH_SEEN.set(seen)
    try:
        yield seen
    finally:
        _NINTH_SEEN.reset(watch)
        _NINTH_STAGE.reset(stage)


@contextmanager
def ninth_on():
    """Stage two: run the block with the ninth batch on."""

    token = _NINTH_STAGE.set(True)
    try:
        yield
    finally:
        _NINTH_STAGE.reset(token)


def ninth_watching() -> bool:
    """Whether stage one should try a rule of the batch without effect here: an attempt in which stage two would let it
    fire first (the shape rules on, the late rules off, no rule switched off), while none has said yes yet."""

    seen = _NINTH_SEEN.get()
    return _NINTH_STAGE.get() is False and seen is not None and not seen and _ENABLED.get() and not _LATE.get() and not (_OFF.get() - _BESIDE_NINTH)


def note_ninth_candidate() -> None:
    """Record that a rule of the ninth batch would have fired in stage one."""

    seen = _NINTH_SEEN.get()
    if seen is not None and not seen:
        seen.append(True)


def tenth_batch_enabled() -> bool:
    """Whether the rules of ``TENTH_BATCH`` may fire in the current attempt: in the third stage of a proof (or outside
    any proof), on a first attempt and on a narrow attempt that switches off rules of that batch only."""

    return (
        _TENTH_STAGE.get() is not False
        and _NINTH_STAGE.get() is not False
        and _ENABLED.get()
        and not _LATE.get()
        and (_OFF.get() - _BESIDE_NINTH) <= TENTH_BATCH
    )


def stage_key() -> tuple[bool, bool, bool, bool, bool, bool, bool, bool, bool, bool]:
    """Which of the staged batches are on (part of any normalization cache key)."""

    return (
        _NINTH_STAGE.get() is not False,
        _TENTH_STAGE.get() is not False,
        _ELEVENTH_STAGE.get() is not False,
        _TWELFTH_STAGE.get() is not False,
        _THIRTEENTH_STAGE.get() is not False,
        _FIFTEENTH_STAGE.get() is True,
        _SIXTEENTH_STAGE.get() is True,
        _SEVENTEENTH_STAGE.get() is True,
        _EIGHTEENTH_STAGE.get() is True,
        _TWENTIETH_STAGE.get() is True,
    )


@contextmanager
def tenth_deferred():
    """Stages one and two: run the block with the tenth batch off, collecting whether one of its rules would have fired."""

    seen: list = []
    stage, watch = _TENTH_STAGE.set(False), _TENTH_SEEN.set(seen)
    try:
        yield seen
    finally:
        _TENTH_SEEN.reset(watch)
        _TENTH_STAGE.reset(stage)


@contextmanager
def tenth_on():
    """Stage three: run the block with the ninth and the tenth batch on."""

    ninth, tenth = _NINTH_STAGE.set(True), _TENTH_STAGE.set(True)
    try:
        yield
    finally:
        _TENTH_STAGE.reset(tenth)
        _NINTH_STAGE.reset(ninth)


def tenth_stage_running() -> bool:
    """Whether stage three of a proof is running: a proof started inside it makes one attempt and no retries."""

    return _TENTH_STAGE.get() is True


def tenth_watching() -> bool:
    """Whether an earlier stage should try the tenth batch on a copy here: an attempt in which stage three would let it
    fire first (the shape rules on, the late rules off, no rule switched off), while no rule of it has said yes yet."""

    seen = _TENTH_SEEN.get()
    return _TENTH_STAGE.get() is False and seen is not None and not seen and _ENABLED.get() and not _LATE.get() and not (_OFF.get() - _BESIDE_NINTH)


def note_tenth_candidate() -> None:
    """Record that a rule of the tenth batch would have fired in an earlier stage."""

    seen = _TENTH_SEEN.get()
    if seen is not None and not seen:
        seen.append(True)


# The eleventh batch (2026-11-19: MySQL sums of truth tests, a shared GROUP BY name, sums of additions) is a fourth
# stage: it runs only when the three stages before it leave the pair without a verdict and a rule of the batch would
# have fired (the earlier stages run the batch on a copy where it would fire, until one rule says yes). Its rules
# write spellings that older rules read, and which older rules must be on differs by pair: a ``HAVING`` over sums of
# comparisons is proved by the prover as it was before the ninth batch once the sums are sums of ones, and a rule of
# the ninth or tenth batch that fires beside it takes the pair out of that shape; a share written with a sum of a
# comparison needs the share rules of the tenth batch. So the stage makes one attempt with the eleventh batch alone
# (``eleventh_alone``), then the attempts of the third stage with all three batches on (``eleventh_on``). A pair that
# was decided before the batch is decided by the same steps in the same time.

ELEVENTH_BATCH = frozenset({"boolean_sums", "shared_group_names", "added_sums"})

_ELEVENTH_STAGE: ContextVar[bool | None] = ContextVar("kumo_sql_prover_eleventh_stage", default=None)
_ELEVENTH_SEEN: ContextVar[list | None] = ContextVar("kumo_sql_prover_eleventh_seen", default=None)


def eleventh_batch_enabled() -> bool:
    """Whether the rules of ``ELEVENTH_BATCH`` may fire in the current attempt: in the fourth stage of a proof (or
    outside any proof), on a first attempt and on a narrow attempt that switches off rules of the tenth batch only."""

    if _ELEVENTH_STAGE.get() is False or not _ENABLED.get() or _LATE.get() or not (_OFF.get() - _BESIDE_NINTH) <= TENTH_BATCH:
        return False
    return _ELEVENTH_STAGE.get() is True or (_NINTH_STAGE.get() is not False and _TENTH_STAGE.get() is not False)


@contextmanager
def eleventh_deferred():
    """Stages one to three: run the block with the eleventh batch off, collecting whether one of its rules would have fired."""

    seen: list = []
    stage, watch = _ELEVENTH_STAGE.set(False), _ELEVENTH_SEEN.set(seen)
    try:
        yield seen
    finally:
        _ELEVENTH_SEEN.reset(watch)
        _ELEVENTH_STAGE.reset(stage)


@contextmanager
def eleventh_alone():
    """Stage four, first attempt: run the block with the eleventh batch on and the ninth and the tenth off."""

    ninth, tenth, eleventh = _NINTH_STAGE.set(False), _TENTH_STAGE.set(False), _ELEVENTH_STAGE.set(True)
    try:
        yield
    finally:
        _ELEVENTH_STAGE.reset(eleventh)
        _TENTH_STAGE.reset(tenth)
        _NINTH_STAGE.reset(ninth)


def eleventh_stage_running() -> bool:
    """Whether stage four of a proof is running: a proof started inside it makes one attempt and no retries."""

    return _ELEVENTH_STAGE.get() is True


@contextmanager
def eleventh_on():
    """Stage four, the later attempts: run the block with the ninth, the tenth and the eleventh batch on."""

    ninth, tenth, eleventh = _NINTH_STAGE.set(True), _TENTH_STAGE.set(True), _ELEVENTH_STAGE.set(True)
    try:
        yield
    finally:
        _ELEVENTH_STAGE.reset(eleventh)
        _TENTH_STAGE.reset(tenth)
        _NINTH_STAGE.reset(ninth)


def eleventh_watching() -> bool:
    """Whether an earlier stage should try the eleventh batch on a copy here: an attempt in which stage four would let
    it fire first (the shape rules on, the late rules off, no rule switched off), while no rule of it has said yes yet."""

    seen = _ELEVENTH_SEEN.get()
    return _ELEVENTH_STAGE.get() is False and seen is not None and not seen and _ENABLED.get() and not _LATE.get() and not (_OFF.get() - _BESIDE_NINTH)


def note_eleventh_candidate() -> None:
    """Record that a rule of the eleventh batch would have fired in an earlier stage."""

    seen = _ELEVENTH_SEEN.get()
    if seen is not None and not seen:
        seen.append(True)


# The twelfth batch (2026-11-19: a MySQL DATE column against a day worked out from a constant) is a fifth stage, built
# like the fourth: it runs only when the four stages before it leave the pair without a verdict and its rule would have
# fired (the earlier stages try the rule on a copy until it says yes). The rule writes date tests the way the prover as
# it was already compares them, so the stage makes one attempt with the twelfth batch alone (``twelfth_alone``), then
# the attempts of the third stage with every staged batch on (``twelfth_on``). A pair that was decided before the batch
# is decided by the same steps in the same time.

TWELFTH_BATCH = frozenset({"date_windows"})

_TWELFTH_STAGE: ContextVar[bool | None] = ContextVar("kumo_sql_prover_twelfth_stage", default=None)
_TWELFTH_SEEN: ContextVar[list | None] = ContextVar("kumo_sql_prover_twelfth_seen", default=None)


def twelfth_batch_enabled() -> bool:
    """Whether the rules of ``TWELFTH_BATCH`` may fire in the current attempt: in the fifth stage of a proof (or
    outside any proof), on a first attempt and on a narrow attempt that switches off rules of the tenth batch only."""

    if _TWELFTH_STAGE.get() is False or not _ENABLED.get() or _LATE.get() or not (_OFF.get() - _BESIDE_NINTH) <= TENTH_BATCH:
        return False
    return _TWELFTH_STAGE.get() is True or (_NINTH_STAGE.get() is not False and _TENTH_STAGE.get() is not False and _ELEVENTH_STAGE.get() is not False)


@contextmanager
def twelfth_deferred():
    """Stages one to four: run the block with the twelfth batch off, collecting whether its rule would have fired."""

    seen: list = []
    stage, watch = _TWELFTH_STAGE.set(False), _TWELFTH_SEEN.set(seen)
    try:
        yield seen
    finally:
        _TWELFTH_SEEN.reset(watch)
        _TWELFTH_STAGE.reset(stage)


@contextmanager
def twelfth_alone():
    """Stage five, first attempt: run the block with the twelfth batch on and the ninth, tenth and eleventh off."""

    ninth, tenth, eleventh, twelfth = _NINTH_STAGE.set(False), _TENTH_STAGE.set(False), _ELEVENTH_STAGE.set(False), _TWELFTH_STAGE.set(True)
    try:
        yield
    finally:
        _TWELFTH_STAGE.reset(twelfth)
        _ELEVENTH_STAGE.reset(eleventh)
        _TENTH_STAGE.reset(tenth)
        _NINTH_STAGE.reset(ninth)


def twelfth_stage_running() -> bool:
    """Whether stage five of a proof is running: a proof started inside it makes one attempt and no retries."""

    return _TWELFTH_STAGE.get() is True


@contextmanager
def twelfth_on():
    """Stage five, the later attempts: run the block with every staged batch on."""

    ninth, tenth, eleventh, twelfth = _NINTH_STAGE.set(True), _TENTH_STAGE.set(True), _ELEVENTH_STAGE.set(True), _TWELFTH_STAGE.set(True)
    try:
        yield
    finally:
        _TWELFTH_STAGE.reset(twelfth)
        _ELEVENTH_STAGE.reset(eleventh)
        _TENTH_STAGE.reset(tenth)
        _NINTH_STAGE.reset(ninth)


def twelfth_watching() -> bool:
    """Whether an earlier stage should try the twelfth batch on a copy here: an attempt in which stage five would let
    it fire first (the shape rules on, the late rules off, no rule switched off), while it has not said yes yet."""

    seen = _TWELFTH_SEEN.get()
    return _TWELFTH_STAGE.get() is False and seen is not None and not seen and _ENABLED.get() and not _LATE.get() and not (_OFF.get() - _BESIDE_NINTH)


def note_twelfth_candidate() -> None:
    """Record that the rule of the twelfth batch would have fired in an earlier stage."""

    seen = _TWELFTH_SEEN.get()
    if seen is not None and not seen:
        seen.append(True)


# The thirteenth batch (2026-11-20: a column list on a WITH name, an outer join from a union of referencing columns, a
# first-row subquery read as MAX or MIN, and a select of aggregate subqueries read as one aggregate select) is a sixth
# stage, built like the fifth: it runs only when the five stages before it leave the
# pair without a verdict and one of its rules would have fired (the earlier stages try the rules on a copy until one
# says yes). One attempt with the thirteenth batch alone (``thirteenth_alone``), then the attempts of the third stage
# with every staged batch on (``thirteenth_on``). A pair that was decided before the batch is decided by the same steps
# in the same time.

THIRTEENTH_BATCH = frozenset({"listed_cte_names", "referencing_union_joins", "top_row_subqueries", "lifted_aggregates"})

_THIRTEENTH_STAGE: ContextVar[bool | None] = ContextVar("kumo_sql_prover_thirteenth_stage", default=None)
_THIRTEENTH_SEEN: ContextVar[list | None] = ContextVar("kumo_sql_prover_thirteenth_seen", default=None)


def thirteenth_batch_enabled() -> bool:
    """Whether the rules of ``THIRTEENTH_BATCH`` may fire in the current attempt: in the sixth stage of a proof (or
    outside any proof), on a first attempt and on a narrow attempt that switches off rules of the tenth batch only."""

    if _THIRTEENTH_STAGE.get() is False or not _ENABLED.get() or _LATE.get() or not (_OFF.get() - _BESIDE_NINTH) <= TENTH_BATCH:
        return False
    return _THIRTEENTH_STAGE.get() is True or (
        _NINTH_STAGE.get() is not False and _TENTH_STAGE.get() is not False and _ELEVENTH_STAGE.get() is not False and _TWELFTH_STAGE.get() is not False
    )


@contextmanager
def thirteenth_deferred():
    """Stages one to five: run the block with the thirteenth batch off, collecting whether a rule of it would have fired."""

    seen: list = []
    stage, watch = _THIRTEENTH_STAGE.set(False), _THIRTEENTH_SEEN.set(seen)
    try:
        yield seen
    finally:
        _THIRTEENTH_SEEN.reset(watch)
        _THIRTEENTH_STAGE.reset(stage)


@contextmanager
def thirteenth_alone():
    """Stage six, first attempt: run the block with the thirteenth batch on and the ninth to the twelfth off."""

    ninth, tenth, eleventh = _NINTH_STAGE.set(False), _TENTH_STAGE.set(False), _ELEVENTH_STAGE.set(False)
    twelfth, thirteenth = _TWELFTH_STAGE.set(False), _THIRTEENTH_STAGE.set(True)
    try:
        yield
    finally:
        _THIRTEENTH_STAGE.reset(thirteenth)
        _TWELFTH_STAGE.reset(twelfth)
        _ELEVENTH_STAGE.reset(eleventh)
        _TENTH_STAGE.reset(tenth)
        _NINTH_STAGE.reset(ninth)


def thirteenth_stage_running() -> bool:
    """Whether stage six of a proof is running: a proof started inside it makes one attempt and no retries."""

    return _THIRTEENTH_STAGE.get() is True


@contextmanager
def thirteenth_on():
    """Stage six, the later attempts: run the block with every staged batch on."""

    ninth, tenth, eleventh = _NINTH_STAGE.set(True), _TENTH_STAGE.set(True), _ELEVENTH_STAGE.set(True)
    twelfth, thirteenth = _TWELFTH_STAGE.set(True), _THIRTEENTH_STAGE.set(True)
    try:
        yield
    finally:
        _THIRTEENTH_STAGE.reset(thirteenth)
        _TWELFTH_STAGE.reset(twelfth)
        _ELEVENTH_STAGE.reset(eleventh)
        _TENTH_STAGE.reset(tenth)
        _NINTH_STAGE.reset(ninth)


def thirteenth_watching() -> bool:
    """Whether an earlier stage should try the thirteenth batch on a copy here: an attempt in which stage six would let
    it fire first (the shape rules on, the late rules off, no rule switched off), while no rule of it has said yes yet."""

    seen = _THIRTEENTH_SEEN.get()
    return _THIRTEENTH_STAGE.get() is False and seen is not None and not seen and _ENABLED.get() and not _LATE.get() and not (_OFF.get() - _BESIDE_NINTH)


def note_thirteenth_candidate() -> None:
    """Record that a rule of the thirteenth batch would have fired in an earlier stage."""

    seen = _THIRTEENTH_SEEN.get()
    if seen is not None and not seen:
        seen.append(True)


# The fifteenth batch (2026-11-21: counts of joined child tables written as one counted table per child) is the last
# step of a proof, after the six stages and after the retry without a shared ORDER BY .. LIMIT (``peeled_tails``). It
# is driven from outside the stages (``algebraic_equivalence._prove_equivalent_algebraic``): the whole staged proof
# runs with the batch off, collecting whether a rule of it would have fired, and only then once more with it on.
# Unlike the earlier batches its rules fire nowhere else: not in a normalization asked for outside a proof.

FIFTEENTH_BATCH = frozenset({"distinct_child_counts"})

_FIFTEENTH_STAGE: ContextVar[bool | None] = ContextVar("kumo_sql_prover_fifteenth_stage", default=None)
_FIFTEENTH_SEEN: ContextVar[list | None] = ContextVar("kumo_sql_prover_fifteenth_seen", default=None)


def fifteenth_batch_enabled() -> bool:
    """Whether the rules of ``FIFTEENTH_BATCH`` may fire in the current attempt: only in the last step of a proof."""

    return _FIFTEENTH_STAGE.get() is True and _ENABLED.get() and not _LATE.get()


def fifteenth_staged() -> bool:
    """Whether a proof is running that already decides when the fifteenth batch is on (a proof inside a proof)."""

    return _FIFTEENTH_STAGE.get() is not None


def last_steps_worth_trying(left_sql: str, right_sql: str) -> bool:
    """Whether a pair every earlier step leaves unproved gets the last steps of a proof.

    Each last step is another whole attempt on the pair, which about doubles the time of an unproved pair, and
    the time grows faster than the query (see the module docstring). So the last steps have the size limit of the
    attempt without the shape rules: a pair whose two texts together are over ``MAX_CHARS`` characters is proved as
    it was before the steps existed.
    """

    return worth_retrying(left_sql, right_sql)


@contextmanager
def fifteenth_deferred(watch: bool = True):
    """Run the block with the fifteenth batch off, collecting whether a rule of it would have fired.

    With ``watch`` false nothing is collected: the batch is off for the whole proof, and for the proofs inside it.
    """

    seen: list | None = [] if watch else None
    stage, watch = _FIFTEENTH_STAGE.set(False), _FIFTEENTH_SEEN.set(seen)
    try:
        yield seen
    finally:
        _FIFTEENTH_SEEN.reset(watch)
        _FIFTEENTH_STAGE.reset(stage)


@contextmanager
def fifteenth_on():
    """The last step: run the block with the fifteenth batch on."""

    stage, watch = _FIFTEENTH_STAGE.set(True), _FIFTEENTH_SEEN.set(None)
    try:
        yield
    finally:
        _FIFTEENTH_SEEN.reset(watch)
        _FIFTEENTH_STAGE.reset(stage)


def fifteenth_watching() -> bool:
    """Whether an earlier step should try the fifteenth batch on a copy here: while no rule of it has said yes yet."""

    seen = _FIFTEENTH_SEEN.get()
    return _FIFTEENTH_STAGE.get() is False and seen is not None and not seen and _ENABLED.get() and not _LATE.get()


def note_fifteenth_candidate() -> None:
    """Record that a rule of the fifteenth batch would have fired in an earlier step."""

    seen = _FIFTEENTH_SEEN.get()
    if seen is not None and not seen:
        seen.append(True)


# The sixteenth batch (2026-11-21: a whole number divided by a constant and compared with a constant, constant rows
# outer-joined to a table and aggregated per row, a CASE of labels compared with a label) is one more step after that
# one, driven from the same place and in the same way: every step before it runs with the batch off, collecting
# whether a rule of it would have fired, and then the staged proof runs once more with the fifteenth and the
# sixteenth batch on. So a pair an earlier step decides is decided as it was. Its rules fire nowhere else either.

SIXTEENTH_BATCH = frozenset({"quotient_comparisons", "constant_bins", "labelled_cases"})

_SIXTEENTH_STAGE: ContextVar[bool | None] = ContextVar("kumo_sql_prover_sixteenth_stage", default=None)
_SIXTEENTH_SEEN: ContextVar[list | None] = ContextVar("kumo_sql_prover_sixteenth_seen", default=None)


def sixteenth_batch_enabled() -> bool:
    """Whether the rules of ``SIXTEENTH_BATCH`` may fire in the current attempt: only in the last step of a proof."""

    return _SIXTEENTH_STAGE.get() is True and _ENABLED.get() and not _LATE.get()


@contextmanager
def sixteenth_deferred(watch: bool = True):
    """Run the block with the sixteenth batch off, collecting whether a rule of it would have fired.

    With ``watch`` false nothing is collected: the batch is off for the whole proof, and for the proofs inside it.
    """

    seen: list | None = [] if watch else None
    stage, watch = _SIXTEENTH_STAGE.set(False), _SIXTEENTH_SEEN.set(seen)
    try:
        yield seen
    finally:
        _SIXTEENTH_SEEN.reset(watch)
        _SIXTEENTH_STAGE.reset(stage)


@contextmanager
def sixteenth_on():
    """The step after the last: run the block with the fifteenth and the sixteenth batch on."""

    with fifteenth_on():
        stage, watch = _SIXTEENTH_STAGE.set(True), _SIXTEENTH_SEEN.set(None)
        try:
            yield
        finally:
            _SIXTEENTH_SEEN.reset(watch)
            _SIXTEENTH_STAGE.reset(stage)


def sixteenth_watching() -> bool:
    """Whether an earlier step should try the sixteenth batch on a copy here: while no rule of it has said yes yet."""

    seen = _SIXTEENTH_SEEN.get()
    return _SIXTEENTH_STAGE.get() is False and seen is not None and not seen and _ENABLED.get() and not _LATE.get()


def note_sixteenth_candidate() -> None:
    """Record that a rule of the sixteenth batch would have fired in an earlier step."""

    seen = _SIXTEENTH_SEEN.get()
    if seen is not None and not seen:
        seen.append(True)


# The seventeenth batch (2026-11-22: counts over a grouped table joined back to at most one of its table's own rows)
# is one more step after that one, driven in the same way: every step before it runs with the batch off, collecting
# whether its rule would have fired, and then the staged proof runs once more with the fifteenth, the sixteenth and
# the seventeenth batch on. So a pair an earlier step decides is decided as it was.

SEVENTEENTH_BATCH = frozenset({"self_lookup_counts"})

_SEVENTEENTH_STAGE: ContextVar[bool | None] = ContextVar("kumo_sql_prover_seventeenth_stage", default=None)
_SEVENTEENTH_SEEN: ContextVar[list | None] = ContextVar("kumo_sql_prover_seventeenth_seen", default=None)


def seventeenth_batch_enabled() -> bool:
    """Whether the rules of ``SEVENTEENTH_BATCH`` may fire in the current attempt: only in the last step of a proof."""

    return _SEVENTEENTH_STAGE.get() is True and _ENABLED.get() and not _LATE.get()


@contextmanager
def seventeenth_deferred(watch: bool = True):
    """Run the block with the seventeenth batch off, collecting whether a rule of it would have fired.

    With ``watch`` false nothing is collected: the batch is off for the whole proof, and for the proofs inside it.
    """

    seen: list | None = [] if watch else None
    stage, watch = _SEVENTEENTH_STAGE.set(False), _SEVENTEENTH_SEEN.set(seen)
    try:
        yield seen
    finally:
        _SEVENTEENTH_SEEN.reset(watch)
        _SEVENTEENTH_STAGE.reset(stage)


@contextmanager
def seventeenth_on():
    """The last step of all: run the block with the fifteenth, the sixteenth and the seventeenth batch on."""

    with sixteenth_on():
        stage, watch = _SEVENTEENTH_STAGE.set(True), _SEVENTEENTH_SEEN.set(None)
        try:
            yield
        finally:
            _SEVENTEENTH_SEEN.reset(watch)
            _SEVENTEENTH_STAGE.reset(stage)


def seventeenth_watching() -> bool:
    """Whether an earlier step should try the seventeenth batch on a copy here: while no rule of it has said yes yet."""

    seen = _SEVENTEENTH_SEEN.get()
    return _SEVENTEENTH_STAGE.get() is False and seen is not None and not seen and _ENABLED.get() and not _LATE.get()


def note_seventeenth_candidate() -> None:
    """Record that a rule of the seventeenth batch would have fired in an earlier step."""

    seen = _SEVENTEENTH_SEEN.get()
    if seen is not None and not seen:
        seen.append(True)


# The eighteenth batch (2026-11-22: a neighbouring row read with LAG or LEAD over a consecutive column, and windows
# whose partitions hold one row) is one more step after that one, driven in the same way: every step before it runs
# with the batch off, collecting whether one of its rules would have fired, and then the staged proof runs once more
# with the fifteenth to the eighteenth batch on. So a pair an earlier step decides is decided as it was.

EIGHTEENTH_BATCH = frozenset({"neighbour_rows", "single_row_partitions", "shared_filter_terms", "exists_outer_terms", "pinned_columns", "lookup_joins", "lookup_conditions"})

_EIGHTEENTH_STAGE: ContextVar[bool | None] = ContextVar("kumo_sql_prover_eighteenth_stage", default=None)
_EIGHTEENTH_SEEN: ContextVar[list | None] = ContextVar("kumo_sql_prover_eighteenth_seen", default=None)


def eighteenth_batch_enabled() -> bool:
    """Whether the rules of ``EIGHTEENTH_BATCH`` may fire in the current attempt: only in the last step of a proof."""

    return _EIGHTEENTH_STAGE.get() is True and _ENABLED.get() and not _LATE.get()


@contextmanager
def eighteenth_deferred(watch: bool = True):
    """Run the block with the eighteenth batch off, collecting whether a rule of it would have fired.

    With ``watch`` false nothing is collected: the batch is off for the whole proof, and for the proofs inside it.
    """

    seen: list | None = [] if watch else None
    stage, watch = _EIGHTEENTH_STAGE.set(False), _EIGHTEENTH_SEEN.set(seen)
    try:
        yield seen
    finally:
        _EIGHTEENTH_SEEN.reset(watch)
        _EIGHTEENTH_STAGE.reset(stage)


@contextmanager
def eighteenth_on():
    """The last step of all: run the block with the fifteenth, sixteenth, seventeenth and eighteenth batch on."""

    with seventeenth_on():
        stage, watch = _EIGHTEENTH_STAGE.set(True), _EIGHTEENTH_SEEN.set(None)
        try:
            yield
        finally:
            _EIGHTEENTH_SEEN.reset(watch)
            _EIGHTEENTH_STAGE.reset(stage)


def eighteenth_watching() -> bool:
    """Whether an earlier step should try the eighteenth batch on a copy here: while no rule of it has said yes yet."""

    seen = _EIGHTEENTH_SEEN.get()
    return _EIGHTEENTH_STAGE.get() is False and seen is not None and not seen and _ENABLED.get() and not _LATE.get()


def note_eighteenth_candidate() -> None:
    """Record that a rule of the eighteenth batch would have fired in an earlier step."""

    seen = _EIGHTEENTH_SEEN.get()
    if seen is not None and not seen:
        seen.append(True)


# The twentieth batch (2026-11-22: a self join whose two rows may change places, read both ways round; rows without
# a match counted with a flag over a left join; a distinct count of a key column its groups cannot repeat) is one more step after that one, driven in the same way: every step
# before it runs with the batch off, collecting whether one of its rules would have fired, and then the staged proof
# runs once more with the fifteenth to the twentieth batch on. So a pair an earlier step decides is decided as it
# was. (Batch 19 changed the counterexample search and has no rule.)

TWENTIETH_BATCH = frozenset({"grouped_key_counts", "mirrored_self_joins", "unmatched_row_counts"})

_TWENTIETH_STAGE: ContextVar[bool | None] = ContextVar("kumo_sql_prover_twentieth_stage", default=None)
_TWENTIETH_SEEN: ContextVar[list | None] = ContextVar("kumo_sql_prover_twentieth_seen", default=None)


def twentieth_batch_enabled() -> bool:
    """Whether the rules of ``TWENTIETH_BATCH`` may fire in the current attempt: only in the last step of a proof."""

    return _TWENTIETH_STAGE.get() is True and _ENABLED.get() and not _LATE.get()


@contextmanager
def twentieth_deferred(watch: bool = True):
    """Run the block with the twentieth batch off, collecting whether a rule of it would have fired.

    With ``watch`` false nothing is collected: the batch is off for the whole proof, and for the proofs inside it.
    """

    seen: list | None = [] if watch else None
    stage, watch = _TWENTIETH_STAGE.set(False), _TWENTIETH_SEEN.set(seen)
    try:
        yield seen
    finally:
        _TWENTIETH_SEEN.reset(watch)
        _TWENTIETH_STAGE.reset(stage)


@contextmanager
def twentieth_on():
    """The last step of all: run the block with the fifteenth to the eighteenth and the twentieth batch on."""

    with eighteenth_on():
        stage, watch = _TWENTIETH_STAGE.set(True), _TWENTIETH_SEEN.set(None)
        try:
            yield
        finally:
            _TWENTIETH_SEEN.reset(watch)
            _TWENTIETH_STAGE.reset(stage)


def twentieth_watching() -> bool:
    """Whether an earlier step should try the twentieth batch on a copy here: while no rule of it has said yes yet."""

    seen = _TWENTIETH_SEEN.get()
    return _TWENTIETH_STAGE.get() is False and seen is not None and not seen and _ENABLED.get() and not _LATE.get()


def note_twentieth_candidate() -> None:
    """Record that a rule of the twentieth batch would have fired in an earlier step."""

    seen = _TWENTIETH_SEEN.get()
    if seen is not None and not seen:
        seen.append(True)


@contextmanager
def without_all(names: frozenset):
    """Run the block with every named narrow-fallback rule off."""

    token = _OFF.set(_OFF.get() | frozenset(names))
    try:
        yield
    finally:
        _OFF.reset(token)
