"""A distinct count of a key column its groups cannot repeat is proved equal to the plain count, in the last step
of a proof (``grouped_key_counts``, a rule of ``shape_fallback.TWENTIETH_BATCH``)."""

import pytest

from kumo_sql_prover import algebraic_equivalence, shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"staff": ["id", "name", "boss", "age"]}
TYPES = {"staff": {"id": "INT", "name": "VARCHAR(20)", "boss": "INT", "age": "INT"}}
CONSTRAINTS = {"staff": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),))}

REPORTS = "FROM staff AS m JOIN staff AS e ON e.boss = m.id"
OLDER = "FROM staff AS m JOIN staff AS e ON e.age > m.age"
# per manager, the number of people who report to them (and per person, the number of older colleagues)
PAIRS = [
    (f"SELECT m.id, m.name, COUNT(*) AS n {REPORTS} GROUP BY m.id, m.name", f"SELECT m.id, m.name, COUNT(DISTINCT e.id) AS n {REPORTS} GROUP BY m.id, m.name"),
    (f"SELECT m.id, COUNT(e.id) AS n {REPORTS} GROUP BY m.id", f"SELECT m.id, COUNT(DISTINCT e.id) AS n {REPORTS} GROUP BY m.id"),
    (f"SELECT m.id, COUNT(*) AS n {OLDER} GROUP BY m.id", f"SELECT m.id, COUNT(DISTINCT e.id) AS n {OLDER} GROUP BY m.id"),
    (f"SELECT m.id, COUNT(*) AS n {REPORTS} GROUP BY m.id HAVING COUNT(*) > 1", f"SELECT m.id, COUNT(DISTINCT e.id) AS n {REPORTS} GROUP BY m.id HAVING COUNT(DISTINCT e.id) > 1"),
]


def prove(left: str, right: str, constraints=CONSTRAINTS):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, types=TYPES, constraints=constraints, compare_names=False, dialect="mysql", exact_arithmetic=True, timeout_ms=5000
    )


def switch_off(monkeypatch) -> None:
    """Take the rule out and leave every other rule and step as it is."""

    monkeypatch.setattr(algebraic_equivalence, "grouped_key_counts", lambda *args: None)


def runs_of(monkeypatch) -> list:
    runs: list = []
    whole = algebraic_equivalence._prove_equivalent_algebraic_whole

    def counted(*args, **kwargs):
        runs.append(shape_fallback.twentieth_batch_enabled())
        return whole(*args, **kwargs)

    monkeypatch.setattr(algebraic_equivalence, "_prove_equivalent_algebraic_whole", counted)
    return runs


@pytest.mark.parametrize("position", range(len(PAIRS)))
def test_the_pairs_are_proved(position):
    assert prove(*PAIRS[position]).proven


@pytest.mark.parametrize("position", (0, 2))  # the other two are proved by the steps before (one grouping column)
def test_a_pair_is_proved_in_one_more_run_and_only_with_the_rule(monkeypatch, position):
    left, right = PAIRS[position]
    runs = runs_of(monkeypatch)
    assert prove(left, right).proven
    first = runs.index(True)
    assert first > 0 and all(runs[first:])  # every run before the last has the batch off
    with monkeypatch.context() as patch:
        switch_off(patch)
        assert not prove(left, right).proven
    assert prove(left, right).proven


def test_pairs_that_differ_stay_unproved():
    # two people of one name share a group, and a colleague older than both is in it twice
    assert not prove(f"SELECT m.name, COUNT(*) AS n {OLDER} GROUP BY m.name", f"SELECT m.name, COUNT(DISTINCT e.id) AS n {OLDER} GROUP BY m.name").proven
    # the ages of a group repeat
    assert not prove(f"SELECT m.id, COUNT(e.age) AS n {OLDER} GROUP BY m.id", f"SELECT m.id, COUNT(DISTINCT e.age) AS n {OLDER} GROUP BY m.id").proven
    # without the declared key, a number may be there twice
    bare = {"staff": TableConstraints()}
    assert not prove(*PAIRS[2], constraints=bare).proven


def test_a_pair_proved_before_gets_no_run_of_the_step(monkeypatch):
    runs = runs_of(monkeypatch)
    left = PAIRS[0][1]
    assert prove(left, left.replace("AS e", "AS r").replace("e.", "r.")).proven  # proved without the rule
    assert runs and not any(runs)
