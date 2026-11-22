"""The seventeenth batch is one more step of a proof, after every other (``shape_fallback.SEVENTEENTH_BATCH``)."""

from kumo_sql_prover import algebraic_equivalence, shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.self_lookup_counts import NAME
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"logins": ["player", "day", "device"]}
TYPES = {"logins": {"player": "INT", "day": "DATE", "device": "INT"}}
CONSTRAINTS = {"logins": TableConstraints(not_null=frozenset({"player", "day"}), keys=(("player", "day"),))}
OPTIONS = {"schema": SCHEMA, "keys": {"logins": [("player", "day")]}, "not_null": {"logins": frozenset({"player", "day"})}, "types": TYPES, "dialect": "mysql"}

P = "(SELECT player, MIN(day) AS first FROM logins GROUP BY player)"
COUNTED = (
    "SELECT p.first AS day, COUNT(DISTINCT p.player) AS players, SUM(CASE WHEN p.first + 1 = a.day THEN 1 ELSE 0 END) AS back "
    f"FROM {P} AS p JOIN logins AS a ON p.player = a.player GROUP BY p.first"
)
LOOKED_UP = f"SELECT t.first, COUNT(t.player) AS players, COUNT(b.player) AS back FROM {P} AS t LEFT JOIN logins AS b ON t.player = b.player AND t.first = b.day - 1 GROUP BY t.first"
PLAIN = "SELECT player FROM logins WHERE device < 3"


def prove(left: str, right: str, dialect: str = "mysql"):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect=dialect, exact_arithmetic=True, timeout_ms=5000
    )


def test_the_batch_names_one_rule_and_has_its_own_entry_in_the_stage_key():
    assert shape_fallback.SEVENTEENTH_BATCH == frozenset({NAME})
    assert len(shape_fallback.stage_key()) == 9 and shape_fallback.stage_key()[7] is False
    with shape_fallback.seventeenth_deferred():
        assert shape_fallback.stage_key()[7] is False
    with shape_fallback.seventeenth_on():
        assert shape_fallback.stage_key()[5:] == (True, True, True, False)  # the step runs with the fifteenth and the sixteenth batch on too
    assert shape_fallback.stage_key()[7] is False


def test_the_rule_fires_only_in_the_last_step_of_a_proof():
    assert not shape_fallback.seventeenth_batch_enabled()
    with shape_fallback.tracking() as fired:
        outside = normalize(LOOKED_UP, **OPTIONS)  # outside a proof the rule is off
    assert NAME not in fired
    with shape_fallback.seventeenth_deferred() as seen:
        assert shape_fallback.seventeenth_watching() and not shape_fallback.seventeenth_batch_enabled()
        assert normalize(LOOKED_UP, **OPTIONS) == outside  # an earlier step asks the rule and changes nothing
        assert seen == [True] and not shape_fallback.seventeenth_watching()
    with shape_fallback.seventeenth_deferred() as seen:
        normalize(PLAIN, **OPTIONS)
        with shape_fallback.disabled():
            normalize(LOOKED_UP, **OPTIONS)
        with shape_fallback.without(NAME):
            normalize(LOOKED_UP, **OPTIONS)
        assert seen == []  # nothing to read, the shape rules off, the rule off
    with shape_fallback.seventeenth_deferred(watch=False) as seen:
        assert seen is None and not shape_fallback.seventeenth_watching()
        assert normalize(LOOKED_UP, **OPTIONS) == outside
    with shape_fallback.seventeenth_on(), shape_fallback.tracking() as fired:
        assert shape_fallback.seventeenth_batch_enabled() and shape_fallback.sixteenth_batch_enabled()
        assert normalize(LOOKED_UP, **OPTIONS) != outside
        for off in (shape_fallback.disabled, shape_fallback.late):
            with off():
                assert not shape_fallback.seventeenth_batch_enabled()
    assert NAME in fired
    assert not shape_fallback.seventeenth_batch_enabled() and not shape_fallback.seventeenth_watching()


def runs_of(monkeypatch) -> list:
    runs: list = []
    whole = algebraic_equivalence._prove_equivalent_algebraic_whole

    def counted(*args, **kwargs):
        runs.append((shape_fallback.sixteenth_batch_enabled(), shape_fallback.seventeenth_batch_enabled()))
        return whole(*args, **kwargs)

    monkeypatch.setattr(algebraic_equivalence, "_prove_equivalent_algebraic_whole", counted)
    return runs


def test_the_rule_proves_a_pair_in_one_more_run(monkeypatch):
    runs = runs_of(monkeypatch)
    assert prove(COUNTED, LOOKED_UP).proven
    # every run before the last has the batch off; the proofs inside the last run are part of it
    first = runs.index((True, True))
    assert first > 0 and not any(seventeenth for _, seventeenth in runs[:first]) and all(run == (True, True) for run in runs[first:])
    with shape_fallback.without(NAME):
        assert not prove(COUNTED, LOOKED_UP).proven


def test_the_pair_is_not_proved_where_a_date_plus_one_is_another_operation():
    assert not prove(COUNTED, LOOKED_UP, "postgres").proven


def test_a_pair_the_rule_does_not_read_and_a_pair_proved_before_get_no_run_of_the_step(monkeypatch):
    runs = runs_of(monkeypatch)
    assert not prove(PLAIN, "SELECT player FROM logins WHERE device < 4").proven
    assert runs and not any(seventeenth for _, seventeenth in runs)
    runs.clear()
    assert prove(LOOKED_UP, LOOKED_UP.replace(" AS players", " AS started")).proven  # proved without the rule
    assert runs and not any(seventeenth for _, seventeenth in runs)
    runs.clear()
    assert not prove(LOOKED_UP, COUNTED.replace("+ 1", "+ 2")).proven  # read, and still different
    assert any(seventeenth for _, seventeenth in runs)


def test_a_pair_over_the_size_limit_does_not_get_the_step(monkeypatch):
    assert shape_fallback.last_steps_worth_trying(COUNTED, LOOKED_UP)
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(COUNTED) + len(LOOKED_UP) - 1)
    runs = runs_of(monkeypatch)
    assert not prove(COUNTED, LOOKED_UP).proven
    assert runs and not any(seventeenth for _, seventeenth in runs)
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(COUNTED) + len(LOOKED_UP))
    assert prove(COUNTED, LOOKED_UP).proven


def test_the_step_gives_proofs_only():
    # The run of the step looks for no counterexample: a pair it does not prove keeps the answer of the steps before.
    different = COUNTED.replace("+ 1", "+ 2")
    with shape_fallback.without(NAME):
        before = prove(LOOKED_UP, different)
    after = prove(LOOKED_UP, different)
    assert not after.proven and after.status == before.status and after.reason == before.reason
