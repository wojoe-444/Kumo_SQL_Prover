"""The eighteenth batch is one more step of a proof, after every other (``shape_fallback.EIGHTEENTH_BATCH``)."""

import re

import pytest

from kumo_sql_prover import algebraic_equivalence, lookup_joins, neighbour_rows, shape_fallback, single_row_partitions
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"seats": ["seat", "free"], "staff": ["person", "unit", "pay"]}
TYPES = {"seats": {"seat": "INT", "free": "BOOL"}, "staff": {"person": "INT", "unit": "INT", "pay": "INT"}}
CONSTRAINTS = {
    "seats": TableConstraints(not_null=frozenset({"seat"}), keys=(("seat",),), consecutive=("seat",), checks=(("in", ("column", "free"), ("values", (0, 1))),)),
    "staff": TableConstraints(not_null=frozenset({"person", "unit"}), keys=(("person", "unit"),)),
}
OPTIONS = {
    "schema": SCHEMA,
    "keys": {"seats": [("seat",)], "staff": [("person", "unit")]},
    "not_null": {"seats": frozenset({"seat"}), "staff": frozenset({"person", "unit"})},
    "types": TYPES,
    "consecutive": {"seats": frozenset({"seat"})},
    "dialect": "mysql",
}

NEXT_TO = "(b.seat = a.seat - 1 OR b.seat = a.seat + 1)"
WINDOWS = "(SELECT seat, free, LAG(free) OVER (ORDER BY seat) AS before, LEAD(free) OVER (ORDER BY seat) AS after FROM seats)"
# seats that are free and have a free seat next to them, eight ways
LOOKED_UP = (
    "SELECT a.seat FROM seats AS a WHERE a.free = 1 AND "
    "((SELECT b.free FROM seats AS b WHERE b.seat = a.seat - 1) = 1 OR (SELECT b.free FROM seats AS b WHERE b.seat = a.seat + 1) = 1)"
)
TESTED = f"SELECT a.seat FROM seats AS a WHERE a.free = 1 AND EXISTS(SELECT 1 FROM seats AS b WHERE {NEXT_TO} AND b.free = 1)"
WINDOWED = f"SELECT d.seat FROM {WINDOWS} AS d WHERE d.free = 1 AND (d.before = 1 OR d.after = 1)"
TWICE = f"SELECT d.seat FROM {WINDOWS} AS d WHERE (d.free = 1 AND d.after = 1) OR (d.free = 1 AND d.before = 1)"
INSIDE = f"SELECT a.seat FROM seats AS a WHERE EXISTS(SELECT 1 FROM seats AS b WHERE {NEXT_TO} AND b.free = 1 AND a.free = 1)"
SAME = f"SELECT a.seat FROM seats AS a WHERE a.free = 1 AND EXISTS(SELECT 1 FROM seats AS b WHERE {NEXT_TO} AND b.free = a.free)"
JOINED = "SELECT a.seat FROM seats AS a LEFT JOIN seats AS p ON p.seat = a.seat - 1 LEFT JOIN seats AS n ON n.seat - a.seat = 1 WHERE a.free = 1 AND (p.free = 1 OR n.free = 1)"
FLAGGED = (
    "SELECT a.seat FROM seats AS a WHERE a.free = 1 AND "
    "((SELECT b.free FROM seats AS b WHERE b.seat = a.seat - 1) IS TRUE OR COALESCE((SELECT b.free FROM seats AS b WHERE b.seat = a.seat + 1), 0) = 1)"
)
# one row for each person and unit
NUMBERED = "SELECT d.person FROM (SELECT person, pay, ROW_NUMBER() OVER (PARTITION BY person, unit ORDER BY pay) AS n FROM staff) AS d WHERE d.n = 1 AND d.pay > 2"
PLAIN = "SELECT person FROM staff WHERE pay > 2"

# a pair and the rules without which it is not proved
PAIRS = [
    (LOOKED_UP, WINDOWED, ("neighbour_rows",)),
    (TESTED, WINDOWED, ("neighbour_rows",)),
    (LOOKED_UP, TWICE, ("neighbour_rows", "shared_filter_terms")),
    (TESTED, INSIDE, ("exists_outer_terms",)),
    (LOOKED_UP, SAME, ("pinned_columns",)),
    (TESTED, JOINED, ("lookup_joins",)),
    (TESTED, FLAGGED, ("lookup_conditions",)),
    (PLAIN, NUMBERED, ("single_row_partitions",)),
]


def prove(left: str, right: str, dialect: str = "mysql", constraints=CONSTRAINTS):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, types=TYPES, constraints=constraints, compare_names=False, dialect=dialect, exact_arithmetic=True, timeout_ms=5000
    )


def switch_off(monkeypatch, name: str) -> None:
    """Take one rule out and leave every other rule and step as it is (``shape_fallback.without`` also changes the
    steps before)."""

    if name == "neighbour_rows":
        monkeypatch.setattr(neighbour_rows, "read_neighbour_rows", lambda tree, *args: (tree, False))
    elif name == "lookup_joins":
        monkeypatch.setattr(lookup_joins, "read_lookup_joins", lambda tree, *args: (tree, False))
    elif name == "single_row_partitions":
        monkeypatch.setattr(single_row_partitions, "read_single_row_partitions", lambda tree, *args: (tree, False))
    else:
        monkeypatch.setattr(algebraic_equivalence, name, lambda *args: None)


def normal(sql: str) -> str:
    """The normal form, whatever number its made-up aliases were given."""

    return re.sub(r"\bkqw\d+", "kqw", normalize(sql, **OPTIONS))


RULES = {
    "neighbour_rows": WINDOWED,
    "single_row_partitions": NUMBERED,
    "shared_filter_terms": TWICE,
    "exists_outer_terms": INSIDE,
    "pinned_columns": SAME,
    "lookup_joins": JOINED,
    "lookup_conditions": FLAGGED,
}


def test_the_batch_names_seven_rules_and_has_its_own_entry_in_the_stage_key():
    assert shape_fallback.EIGHTEENTH_BATCH == frozenset(RULES)
    assert {name for _, _, names in PAIRS for name in names} == shape_fallback.EIGHTEENTH_BATCH
    assert len(shape_fallback.stage_key()) == 11 and shape_fallback.stage_key()[8] is False
    with shape_fallback.eighteenth_deferred():
        assert shape_fallback.stage_key()[8] is False
    with shape_fallback.eighteenth_on():
        assert shape_fallback.stage_key()[5:] == (True, True, True, True, False, False)  # the step runs with the three batches before it on too
    assert shape_fallback.stage_key()[8] is False


@pytest.mark.parametrize("name", sorted(RULES))
def test_a_rule_fires_only_in_the_last_step_of_a_proof(name):
    sql = RULES[name]
    assert not shape_fallback.eighteenth_batch_enabled()
    with shape_fallback.tracking() as fired:
        outside = normal(sql)  # outside a proof the rules are off
    assert not fired & shape_fallback.EIGHTEENTH_BATCH
    with shape_fallback.eighteenth_deferred() as seen:
        assert shape_fallback.eighteenth_watching() and not shape_fallback.eighteenth_batch_enabled()
        assert normal(sql) == outside  # an earlier step asks the rules and changes nothing
        assert seen == [True] and not shape_fallback.eighteenth_watching()
    with shape_fallback.eighteenth_deferred() as seen:
        normalize("SELECT seat FROM seats WHERE free = 0", **OPTIONS)
        with shape_fallback.disabled():
            normalize(sql, **OPTIONS)
        assert seen == []  # nothing to read, the shape rules off
    with shape_fallback.eighteenth_deferred(watch=False) as seen:
        assert seen is None and not shape_fallback.eighteenth_watching()
        assert normal(sql) == outside
    with shape_fallback.eighteenth_on(), shape_fallback.tracking() as fired:
        assert shape_fallback.eighteenth_batch_enabled() and shape_fallback.seventeenth_batch_enabled() and shape_fallback.sixteenth_batch_enabled()
        assert normal(sql) != outside
        with shape_fallback.without(name):
            assert name not in _fired(sql)
        for off in (shape_fallback.disabled, shape_fallback.late):
            with off():
                assert not shape_fallback.eighteenth_batch_enabled()
    assert name in fired
    assert not shape_fallback.eighteenth_batch_enabled() and not shape_fallback.eighteenth_watching()


def _fired(sql: str) -> set:
    with shape_fallback.tracking() as fired:
        normalize(sql, **OPTIONS)
    return set(fired)


def runs_of(monkeypatch) -> list:
    runs: list = []
    whole = algebraic_equivalence._prove_equivalent_algebraic_whole

    def counted(*args, **kwargs):
        runs.append((shape_fallback.seventeenth_batch_enabled(), shape_fallback.eighteenth_batch_enabled()))
        return whole(*args, **kwargs)

    monkeypatch.setattr(algebraic_equivalence, "_prove_equivalent_algebraic_whole", counted)
    return runs


@pytest.mark.parametrize("position", range(len(PAIRS)))
def test_a_pair_is_proved_in_one_more_run_and_only_with_its_rules(monkeypatch, position):
    left, right, names = PAIRS[position]
    runs = runs_of(monkeypatch)
    assert prove(left, right).proven
    # every run before the last has the batch off; the proofs inside the last run are part of it
    first = runs.index((True, True))
    assert first > 0 and not any(eighteenth for _, eighteenth in runs[:first]) and all(run == (True, True) for run in runs[first:])
    for name in names:
        with monkeypatch.context() as patch:
            switch_off(patch, name)
            assert not prove(left, right).proven, name
    assert prove(left, right).proven


def test_a_neighbour_is_read_only_over_a_column_declared_consecutive():
    undeclared = {**CONSTRAINTS, "seats": TableConstraints(not_null=frozenset({"seat"}), keys=(("seat",),), checks=CONSTRAINTS["seats"].checks)}
    assert prove(TESTED, WINDOWED).proven
    assert not prove(TESTED, WINDOWED, constraints=undeclared).proven  # seats 1 and 3 are neighbours for LAG and not for the lookup
    assert prove(TESTED, INSIDE, constraints=undeclared).proven  # the other rules ask for no declaration of it


def test_a_window_is_read_on_its_row_only_with_a_declared_key():
    assert not prove(PLAIN, NUMBERED, constraints={"seats": CONSTRAINTS["seats"]}).proven
    assert not prove(PLAIN, NUMBERED.replace("PARTITION BY person, unit", "PARTITION BY person")).proven


def test_a_pair_the_rules_do_not_read_and_a_pair_proved_before_get_no_run_of_the_step(monkeypatch):
    runs = runs_of(monkeypatch)
    assert not prove(PLAIN, "SELECT person FROM staff WHERE pay > 3").proven
    assert runs and not any(eighteenth for _, eighteenth in runs)
    runs.clear()
    assert prove(WINDOWED, WINDOWED.replace("AS d", "AS e").replace("d.", "e.")).proven  # proved without the rules
    assert runs and not any(eighteenth for _, eighteenth in runs)
    runs.clear()
    assert not prove(TESTED, WINDOWED.replace("d.free = 1", "d.free = 0")).proven  # read, and still different
    assert any(eighteenth for _, eighteenth in runs)


def test_a_pair_over_the_size_limit_does_not_get_the_step(monkeypatch):
    assert shape_fallback.last_steps_worth_trying(TESTED, WINDOWED)
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(TESTED) + len(WINDOWED) - 1)
    runs = runs_of(monkeypatch)
    assert not prove(TESTED, WINDOWED).proven
    assert runs and not any(eighteenth for _, eighteenth in runs)
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(TESTED) + len(WINDOWED))
    assert prove(TESTED, WINDOWED).proven


def test_the_step_gives_proofs_only(monkeypatch):
    # The run of the step looks for no counterexample: a pair it does not prove keeps the answer of the steps before.
    different = WINDOWED.replace("d.free = 1", "d.free = 0")
    after = prove(TESTED, different)
    switch_off(monkeypatch, "neighbour_rows")
    before = prove(TESTED, different)
    assert not after.proven and after.status == before.status and after.reason == before.reason
