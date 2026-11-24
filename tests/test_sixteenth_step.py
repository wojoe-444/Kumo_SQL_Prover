"""The sixteenth batch is one more step of a proof, after every other (``shape_fallback.SIXTEENTH_BATCH``)."""

import pytest

from kumo_sql_prover import algebraic_equivalence, shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"visits": ["id", "secs", "kind"]}
TYPES = {"visits": {"id": "INT", "secs": "INT", "kind": "VARCHAR"}}
CONSTRAINTS = {"visits": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),))}
OPTIONS = {"schema": SCHEMA, "keys": {"visits": [("id",)]}, "not_null": {"visits": frozenset({"id"})}, "types": TYPES, "dialect": "mysql"}

BINS = "(SELECT 'short' AS bin UNION ALL SELECT 'long')"
LABELLED = "(SELECT id, CASE WHEN secs / 60 < 5 THEN 'short' ELSE 'long' END AS bin FROM visits)"
JOINED = f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin"
COUNTED = "SELECT 'short' AS bin, COUNT(id) AS total FROM visits WHERE secs < 300 UNION ALL SELECT 'long', COUNT(id) FROM visits WHERE secs > 299 OR secs IS NULL"
# one query for each rule of the batch, and one that none of them reads
READ = {
    "quotient_comparisons": "SELECT id FROM visits WHERE secs / 60 < 5",
    "constant_bins": f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN visits AS v ON v.kind = b.bin GROUP BY b.bin",
    "labelled_cases": "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'short' ELSE 'long' END = 'long'",
}
PLAIN = "SELECT id FROM visits WHERE secs < 300"


def prove(left: str, right: str, dialect: str = "mysql"):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect=dialect, exact_arithmetic=True, timeout_ms=5000
    )


def test_the_batch_names_three_rules_and_has_its_own_entry_in_the_stage_key():
    assert shape_fallback.SIXTEENTH_BATCH == frozenset(READ)
    assert len(shape_fallback.stage_key()) == 13 and shape_fallback.stage_key()[6] is False
    with shape_fallback.sixteenth_deferred():
        assert shape_fallback.stage_key()[6] is False
    with shape_fallback.sixteenth_on():
        assert shape_fallback.stage_key()[5:7] == (True, True)  # the step runs with the fifteenth batch on too
    assert shape_fallback.stage_key()[6] is False


@pytest.mark.parametrize("name", sorted(READ))
def test_each_rule_fires_only_in_the_last_step_of_a_proof(name):
    sql = READ[name]
    assert not shape_fallback.sixteenth_batch_enabled()
    with shape_fallback.tracking() as fired:
        outside = normalize(sql, **OPTIONS)  # outside a proof the rule is off
    assert name not in fired
    with shape_fallback.sixteenth_deferred() as seen:
        assert shape_fallback.sixteenth_watching() and not shape_fallback.sixteenth_batch_enabled()
        assert normalize(sql, **OPTIONS) == outside  # an earlier step asks the rule and changes nothing
        assert seen == [True] and not shape_fallback.sixteenth_watching()
    with shape_fallback.sixteenth_deferred() as seen:
        normalize(PLAIN, **OPTIONS)
        with shape_fallback.disabled():
            normalize(sql, **OPTIONS)
        with shape_fallback.without(name):
            normalize(sql, **OPTIONS)
        assert seen == []  # nothing to read, the shape rules off, the rule off
    with shape_fallback.sixteenth_deferred(watch=False) as seen:
        assert seen is None and not shape_fallback.sixteenth_watching()
        assert normalize(sql, **OPTIONS) == outside
    with shape_fallback.sixteenth_on(), shape_fallback.tracking() as fired:
        assert shape_fallback.sixteenth_batch_enabled()
        assert normalize(sql, **OPTIONS) != outside
        for off in (shape_fallback.disabled, shape_fallback.late):
            with off():
                assert not shape_fallback.sixteenth_batch_enabled()
    assert name in fired
    assert not shape_fallback.sixteenth_batch_enabled() and not shape_fallback.sixteenth_watching()


def runs_of(monkeypatch) -> list:
    runs: list = []
    whole = algebraic_equivalence._prove_equivalent_algebraic_whole

    def counted(*args, **kwargs):
        runs.append(shape_fallback.sixteenth_batch_enabled())
        return whole(*args, **kwargs)

    monkeypatch.setattr(algebraic_equivalence, "_prove_equivalent_algebraic_whole", counted)
    return runs


def test_the_three_rules_prove_a_pair_together_in_one_more_run(monkeypatch):
    runs = runs_of(monkeypatch)
    assert prove(JOINED, COUNTED).proven
    # every run before the last has the batch off; the proofs inside the last run are part of it
    first = runs.index(True)
    assert first > 0 and not any(runs[:first]) and all(runs[first:])


@pytest.mark.parametrize("name", sorted(READ))
def test_the_pair_needs_each_of_the_three_rules(name):
    with shape_fallback.without(name):
        assert not prove(JOINED, COUNTED).proven


def test_the_pair_is_not_proved_where_the_quotient_is_another_operation():
    assert not prove(JOINED, COUNTED, "postgres").proven


def test_a_pair_no_rule_reads_and_a_pair_proved_before_get_no_run_of_the_step(monkeypatch):
    runs = runs_of(monkeypatch)
    assert not prove(PLAIN, "SELECT id FROM visits WHERE secs < 301").proven
    assert runs and not any(runs)
    runs.clear()
    assert prove(READ["quotient_comparisons"], READ["quotient_comparisons"].replace("SELECT id", "SELECT visits.id")).proven  # proved without the rule
    assert runs and not any(runs)
    runs.clear()
    assert not prove(READ["quotient_comparisons"], "SELECT id FROM visits WHERE secs < 301").proven  # read, and still different
    assert any(runs)


def test_a_pair_over_the_size_limit_does_not_get_the_step(monkeypatch):
    assert shape_fallback.last_steps_worth_trying(JOINED, COUNTED)
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(JOINED) + len(COUNTED) - 1)
    runs = runs_of(monkeypatch)
    assert not prove(JOINED, COUNTED).proven
    assert runs and not any(runs)
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(JOINED) + len(COUNTED))
    assert prove(JOINED, COUNTED).proven


def test_the_step_gives_proofs_only():
    # The run of the step looks for no counterexample: a pair it does not prove keeps the answer of the steps before.
    different = COUNTED.replace("secs > 299", "secs > 300")
    with shape_fallback.without_all(shape_fallback.SIXTEENTH_BATCH):
        before = prove(JOINED, different)
    after = prove(JOINED, different)
    assert not after.proven and after.status == before.status and after.reason == before.reason
