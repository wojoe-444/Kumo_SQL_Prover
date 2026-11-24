"""The twentieth batch is one more step of a proof, after every other (``shape_fallback.TWENTIETH_BATCH``)."""

import re

import pytest

from kumo_sql_prover import algebraic_equivalence, mirrored_self_joins, shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtStatus, TableConstraints

SCHEMA = {"stays": ["guest", "room", "arrival", "departure"]}
TYPES = {"stays": {"guest": "INT", "room": "INT", "arrival": "INT", "departure": "INT"}}
BEFORE = ("<", ("column", "arrival"), ("column", "departure"))
CONSTRAINTS = {"stays": TableConstraints(not_null=frozenset({"arrival", "departure"}), checks=(BEFORE,))}
OPTIONS = {"schema": SCHEMA, "keys": {}, "not_null": {"stays": frozenset({"arrival", "departure"})}, "types": TYPES, "dialect": "mysql"}

HEAD = "SELECT DISTINCT a.guest FROM stays AS a JOIN stays AS b ON a.guest = b.guest AND a.room <> b.room"
# guests with two stays in different rooms that overlap, four ways
STARTS_INSIDE = f"{HEAD} AND b.arrival BETWEEN a.arrival AND a.departure"
OVERLAPS = f"{HEAD} AND a.arrival <= b.departure AND b.arrival <= a.departure"
EITHER_END = f"{HEAD} AND (b.arrival BETWEEN a.arrival AND a.departure OR b.departure BETWEEN a.arrival AND a.departure)"
ROOMS_IN_ORDER = "SELECT DISTINCT a.guest FROM stays AS a JOIN stays AS b ON a.guest = b.guest AND a.room < b.room WHERE a.arrival <= b.departure AND b.arrival <= a.departure"

PAIRS = [(STARTS_INSIDE, OVERLAPS), (STARTS_INSIDE, EITHER_END), (OVERLAPS, ROOMS_IN_ORDER), (EITHER_END, ROOMS_IN_ORDER)]


def prove(left: str, right: str, constraints=CONSTRAINTS, **more):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, types=TYPES, constraints=constraints, compare_names=False, dialect="mysql", exact_arithmetic=True, timeout_ms=5000, **more
    )


def switch_off(monkeypatch) -> None:
    """Take the rule out and leave every other rule and step as it is (``shape_fallback.without`` also changes the
    steps before)."""

    monkeypatch.setattr(mirrored_self_joins, "read_mirrored_self_joins", lambda tree, *args: (tree, False))


def test_the_batch_names_its_rules_and_has_its_own_entry_in_the_stage_key():
    assert shape_fallback.TWENTIETH_BATCH == frozenset({"grouped_key_counts", "mirrored_self_joins", "unmatched_row_counts"})
    assert len(shape_fallback.stage_key()) == 11 and shape_fallback.stage_key()[9] is False
    with shape_fallback.twentieth_deferred():
        assert shape_fallback.stage_key()[9] is False
    with shape_fallback.twentieth_on():
        assert shape_fallback.stage_key()[5:] == (True, True, True, True, True, False)  # the step runs with the batches before it on too
    assert shape_fallback.stage_key()[9] is False


def normal(sql: str) -> str:
    """The normal form, with the numbers of the names the normalizer makes up taken out (they count up per call)."""

    return re.sub(r"\bkqs\d+", "kqs", normalize(sql, **OPTIONS))


def test_the_rule_fires_only_in_the_last_step_of_a_proof():
    sql = STARTS_INSIDE
    assert not shape_fallback.twentieth_batch_enabled()
    with shape_fallback.tracking() as fired:
        outside = normal(sql)  # outside a proof the rule is off
    assert not fired & shape_fallback.TWENTIETH_BATCH
    with shape_fallback.twentieth_deferred() as seen:
        assert shape_fallback.twentieth_watching() and not shape_fallback.twentieth_batch_enabled()
        assert normal(sql) == outside  # an earlier step asks the rule and changes nothing
        assert seen == [True] and not shape_fallback.twentieth_watching()
    with shape_fallback.twentieth_deferred() as seen:
        normal(OVERLAPS)  # reads the same both ways round: nothing to do
        with shape_fallback.disabled():
            normal(sql)
        assert seen == []
    with shape_fallback.twentieth_deferred(watch=False) as seen:
        assert seen is None and not shape_fallback.twentieth_watching()
        assert normal(sql) == outside
    with shape_fallback.twentieth_on(), shape_fallback.tracking() as fired:
        assert shape_fallback.twentieth_batch_enabled() and shape_fallback.eighteenth_batch_enabled() and shape_fallback.seventeenth_batch_enabled()
        assert normal(sql) != outside
        with shape_fallback.without("mirrored_self_joins"), shape_fallback.tracking() as inner:
            assert normal(sql) == outside
        assert "mirrored_self_joins" not in inner
        for off in (shape_fallback.disabled, shape_fallback.late):
            with off():
                assert not shape_fallback.twentieth_batch_enabled()
    assert "mirrored_self_joins" in fired
    assert not shape_fallback.twentieth_batch_enabled() and not shape_fallback.twentieth_watching()


def runs_of(monkeypatch) -> list:
    runs: list = []
    whole = algebraic_equivalence._prove_equivalent_algebraic_whole

    def counted(*args, **kwargs):
        runs.append((shape_fallback.eighteenth_batch_enabled(), shape_fallback.twentieth_batch_enabled()))
        return whole(*args, **kwargs)

    monkeypatch.setattr(algebraic_equivalence, "_prove_equivalent_algebraic_whole", counted)
    return runs


@pytest.mark.parametrize("position", range(len(PAIRS)))
def test_a_pair_is_proved_in_one_more_run_and_only_with_the_rule(monkeypatch, position):
    left, right = PAIRS[position]
    runs = runs_of(monkeypatch)
    assert prove(left, right).proven
    # every run before the last has the batch off; the proofs inside the last run are part of it
    first = runs.index((True, True))
    assert first > 0 and not any(twentieth for _, twentieth in runs[:first]) and all(run == (True, True) for run in runs[first:])
    with monkeypatch.context() as patch:
        switch_off(patch)
        assert not prove(left, right).proven
    assert prove(left, right).proven


def test_the_proof_needs_what_the_table_declares():
    assert prove(STARTS_INSIDE, OVERLAPS).proven
    # a stay that ends before it starts overlaps nothing and still has another stay start inside no stay
    assert not prove(STARTS_INSIDE, OVERLAPS, constraints={"stays": TableConstraints(not_null=frozenset({"arrival", "departure"}))}).proven
    # a stay with no departure: another stay starts inside no such stay, and it can start inside another
    open_stays = {"stays": TableConstraints(not_null=frozenset({"arrival"}), checks=(BEFORE,))}
    assert not prove(STARTS_INSIDE, OVERLAPS, constraints=open_stays).proven
    under = prove(STARTS_INSIDE, OVERLAPS, constraints=open_stays, conditional=True, conditional_seconds=20)
    assert under.status is SmtStatus.PROVEN_CONDITIONALLY and "departure is NOT NULL" in under.reason


def test_pairs_that_differ_stay_unproved():
    # the second stay has to start inside the first, strictly after it starts: the same guest twice on one day is lost
    assert not prove(STARTS_INSIDE, f"{HEAD} AND b.arrival > a.arrival AND b.arrival <= a.departure").proven
    # without DISTINCT the number of pairs shows
    assert not prove(STARTS_INSIDE.replace("DISTINCT ", ""), OVERLAPS.replace("DISTINCT ", "")).proven
    # the room is not the same for the two stays of a pair
    assert not prove(STARTS_INSIDE.replace("a.guest FROM", "a.room FROM"), OVERLAPS.replace("a.guest FROM", "a.room FROM")).proven


def test_a_pair_the_rule_does_not_read_and_a_pair_proved_before_get_no_run_of_the_step(monkeypatch):
    runs = runs_of(monkeypatch)
    assert not prove("SELECT guest FROM stays WHERE room > 2", "SELECT guest FROM stays WHERE room > 3").proven
    assert runs and not any(twentieth for _, twentieth in runs)
    runs.clear()
    assert prove(STARTS_INSIDE, STARTS_INSIDE.replace("AS b", "AS c").replace("b.", "c.")).proven  # proved without the rule
    assert runs and not any(twentieth for _, twentieth in runs)
    runs.clear()
    assert not prove(STARTS_INSIDE, OVERLAPS.replace("a.room <> b.room", "a.room = b.room")).proven  # read, and still different
    assert any(twentieth for _, twentieth in runs)


def test_a_pair_over_the_size_limit_does_not_get_the_step(monkeypatch):
    assert shape_fallback.last_steps_worth_trying(STARTS_INSIDE, OVERLAPS)
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(STARTS_INSIDE) + len(OVERLAPS) - 1)
    runs = runs_of(monkeypatch)
    assert not prove(STARTS_INSIDE, OVERLAPS).proven
    assert runs and not any(twentieth for _, twentieth in runs)
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(STARTS_INSIDE) + len(OVERLAPS))
    assert prove(STARTS_INSIDE, OVERLAPS).proven


def test_the_step_gives_proofs_only(monkeypatch):
    # The run of the step looks for no counterexample: a pair it does not prove keeps the answer of the steps before.
    different = OVERLAPS.replace("a.room <> b.room", "a.room = b.room")
    after = prove(STARTS_INSIDE, different)
    switch_off(monkeypatch)
    before = prove(STARTS_INSIDE, different)
    assert not after.proven and after.status == before.status and after.reason == before.reason
