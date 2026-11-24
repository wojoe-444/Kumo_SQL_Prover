"""The twenty-fifth batch is one more step of a proof, after every other (``shape_fallback.TWENTY_FIFTH_BATCH``)."""

import re

import pytest

from kumo_sql_prover import algebraic_equivalence, apart_alias_names, shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"marks": ["name", "v"], "tests": ["l", "op", "r"], "crew": ["id", "name", "boss", "age"]}
TYPES = {
    "marks": {"name": "VARCHAR", "v": "INT"},
    "tests": {"l": "VARCHAR", "op": "VARCHAR", "r": "VARCHAR"},
    "crew": {"id": "INT", "name": "VARCHAR", "boss": "INT", "age": "INT"},
}
CONSTRAINTS = {
    "marks": TableConstraints(not_null=frozenset({"name"}), keys=(("name",),)),
    "tests": TableConstraints(not_null=frozenset({"l", "op", "r"}), keys=(("l", "op", "r"),)),
    "crew": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
}
OPTIONS = {"schema": SCHEMA, "keys": {}, "not_null": {}, "types": TYPES, "dialect": "mysql"}

# each test with the mark of its left and of its right name
FLAT = "SELECT a.l, a.r, b.v AS lv, c.v AS rv FROM tests AS a LEFT JOIN marks AS b ON a.l = b.name LEFT JOIN marks AS c ON a.r = c.name"
NESTED = "SELECT c.l, c.r, c.lv, v.v AS rv FROM (SELECT e.l, e.r, v.v AS lv FROM tests AS e LEFT JOIN marks AS v ON e.l = v.name) AS c LEFT JOIN marks AS v ON c.r = v.name"
NESTED_APART = NESTED.replace("v.v AS lv", "w.v AS lv").replace("marks AS v ON e.l = v.name", "marks AS w ON e.l = w.name")
WITH = "WITH c AS (SELECT e.l, e.r, v.v AS lv FROM tests AS e LEFT JOIN marks AS v ON e.l = v.name) SELECT c.l, c.r, c.lv, v.v AS rv FROM c LEFT JOIN marks AS v ON c.r = v.name"
TWICE = (
    "WITH c1 AS (SELECT l, op, r, v.v AS lv FROM tests AS e LEFT JOIN marks AS v ON e.l = v.name), "
    "c2 AS (SELECT l, op, r, lv, v.v AS rv FROM c1 AS c LEFT JOIN marks AS v ON c.r = v.name) SELECT l, r, lv, rv FROM c2"
)
# an alias that is the own name of the table read inside
LED = "SELECT a.id, a.name, ROUND(AVG(b.age), 0) AS mean FROM crew AS a JOIN crew AS b ON a.id = b.boss GROUP BY a.id"
LED_SAME = "SELECT t.boss AS id, e.name, t.mean FROM (SELECT e.boss, ROUND(AVG(e.age), 0) AS mean FROM crew AS e GROUP BY e.boss) AS t JOIN crew AS e ON t.boss = e.id"

PAIRS = [(FLAT, NESTED), (FLAT, WITH), (FLAT, TWICE), (NESTED_APART, NESTED), (LED, LED_SAME)]


def prove(left: str, right: str, **more):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect="mysql", exact_arithmetic=True, timeout_ms=5000, **more
    )


def switch_off(monkeypatch) -> None:
    """Take the rule out and leave every other rule and step as it is (``shape_fallback.without`` also changes the
    steps before)."""

    monkeypatch.setattr(apart_alias_names, "name_aliases_apart", lambda tree: (tree, False))


def test_the_batch_names_its_rule_and_has_its_own_entry_in_the_stage_key():
    assert shape_fallback.TWENTY_FIFTH_BATCH == frozenset({"apart_alias_names"})
    assert len(shape_fallback.stage_key()) == 12 and shape_fallback.stage_key()[11] is False
    with shape_fallback.twenty_fifth_deferred():
        assert shape_fallback.stage_key()[11] is False
    with shape_fallback.twenty_fourth_on():
        assert shape_fallback.stage_key()[5:] == (True, True, True, True, True, True, False)  # the step before leaves this one off
    with shape_fallback.twenty_fifth_on():
        assert shape_fallback.stage_key()[5:] == (True, True, True, True, True, True, True)  # the step runs with the batches before it on too
    assert shape_fallback.stage_key()[11] is False


def normal(sql: str) -> str:
    """The normal form, with the numbers of the names the normalizer makes up taken out (they count up per call)."""

    return re.sub(r"\bkq([a-z]+)\d+", r"kq\1", normalize(sql, **OPTIONS))


def test_the_rule_fires_only_in_the_last_step_of_a_proof():
    sql = NESTED
    assert not shape_fallback.twenty_fifth_batch_enabled()
    with shape_fallback.tracking() as fired:
        outside = normal(sql)  # outside a proof the rule is off
    assert not fired & shape_fallback.TWENTY_FIFTH_BATCH
    with shape_fallback.twenty_fifth_deferred() as seen:
        assert shape_fallback.twenty_fifth_watching() and not shape_fallback.twenty_fifth_batch_enabled()
        assert normal(sql) == outside  # an earlier step asks the rule and changes nothing
        assert seen == [True] and not shape_fallback.twenty_fifth_watching()
    with shape_fallback.twenty_fifth_deferred() as seen:
        normal(FLAT)  # nothing to read
        normal(NESTED_APART)
        with shape_fallback.disabled():
            normal(sql)
        assert seen == []
    with shape_fallback.twenty_fifth_deferred(watch=False) as seen:
        assert seen is None and not shape_fallback.twenty_fifth_watching()
        assert normal(sql) == outside
    with shape_fallback.twenty_fifth_on(), shape_fallback.tracking() as fired:
        assert shape_fallback.twenty_fifth_batch_enabled() and shape_fallback.twenty_fourth_batch_enabled() and shape_fallback.twentieth_batch_enabled()
        assert normal(sql) != outside
        with shape_fallback.without("apart_alias_names"), shape_fallback.tracking() as inner:
            assert normal(sql) == outside
        assert "apart_alias_names" not in inner
        for off in (shape_fallback.disabled, shape_fallback.late):
            with off():
                assert not shape_fallback.twenty_fifth_batch_enabled()
    assert "apart_alias_names" in fired
    assert not shape_fallback.twenty_fifth_batch_enabled() and not shape_fallback.twenty_fifth_watching()


def test_a_table_read_twice_under_its_own_name_is_the_step_before(monkeypatch):
    # ``apart_table_names`` (batch 24) names those reads apart: this rule is not asked to count them again
    own = "SELECT t.boss AS id, crew.name, t.mean FROM (SELECT boss, ROUND(AVG(age), 0) AS mean FROM crew GROUP BY boss) AS t JOIN crew ON t.boss = crew.id"
    with shape_fallback.twenty_fifth_deferred() as seen, shape_fallback.twenty_fourth_deferred() as before:
        normal(own)
        assert before == [True] and seen == []
    runs = runs_of(monkeypatch)
    assert prove(LED, own).proven
    assert (True, False) in runs and not any(last for _, last in runs)


def runs_of(monkeypatch) -> list:
    runs: list = []
    whole = algebraic_equivalence._prove_equivalent_algebraic_whole

    def counted(*args, **kwargs):
        runs.append((shape_fallback.twenty_fourth_batch_enabled(), shape_fallback.twenty_fifth_batch_enabled()))
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
    assert first > 0 and not any(last for _, last in runs[:first]) and all(run == (True, True) for run in runs[first:])
    with monkeypatch.context() as patch:
        switch_off(patch)
        assert not prove(left, right).proven
    assert prove(left, right).proven


def test_pairs_that_differ_stay_unproved():
    # the second lookup is of the left name too
    assert not prove(FLAT, NESTED.replace("c.r = v.name", "c.l = v.name")).proven
    # an inner join drops the tests whose right name has no mark
    assert not prove(FLAT, NESTED.replace("AS c LEFT JOIN", "AS c JOIN")).proven
    # the mark of the first lookup shown twice
    assert not prove(FLAT, NESTED.replace("v.v AS rv", "c.lv AS rv")).proven
    # the people who report to someone else
    assert not prove(LED, LED_SAME.replace("t.boss = e.id", "t.boss = e.boss")).proven
    # with the foreign keys that make the lookups inner joins the nested spelling is still read
    keyed = {**CONSTRAINTS, "tests": TableConstraints(not_null=frozenset({"l", "op", "r"}), keys=(("l", "op", "r"),), foreign_keys=((("l",), "marks", ("name",)), (("r",), "marks", ("name",))))}
    inner = FLAT.replace("LEFT JOIN", "JOIN")
    assert prove_equivalent_algebraic(inner, NESTED, schema=SCHEMA, types=TYPES, constraints=keyed, compare_names=False, dialect="mysql", exact_arithmetic=True, timeout_ms=5000).proven
    assert not prove(inner, NESTED).proven  # and without them a test with no mark is on one side only


def test_a_pair_the_rule_does_not_read_and_a_pair_proved_before_get_no_run_of_the_step(monkeypatch):
    runs = runs_of(monkeypatch)
    assert not prove("SELECT name FROM marks WHERE v > 2", "SELECT name FROM marks WHERE v > 3").proven
    assert runs and not any(last for _, last in runs)
    runs.clear()
    assert prove(FLAT, NESTED_APART).proven  # with the two lookups named apart by hand, no step of this batch is needed
    assert runs and not any(last for _, last in runs)
    runs.clear()
    assert not prove(FLAT, NESTED.replace("c.r = v.name", "c.l = v.name")).proven  # read, and still different
    assert any(last for _, last in runs)


def test_a_pair_over_the_size_limit_does_not_get_the_step(monkeypatch):
    assert shape_fallback.last_steps_worth_trying(FLAT, NESTED)
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(FLAT) + len(NESTED) - 1)
    runs = runs_of(monkeypatch)
    assert not prove(FLAT, NESTED).proven
    assert runs and not any(last for _, last in runs)
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(FLAT) + len(NESTED))
    assert prove(FLAT, NESTED).proven


def test_the_step_gives_proofs_only(monkeypatch):
    # The run of the step looks for no counterexample: a pair it does not prove keeps the answer of the steps before.
    different = NESTED.replace("c.r = v.name", "c.l = v.name")
    after = prove(FLAT, different)
    switch_off(monkeypatch)
    before = prove(FLAT, different)
    assert not after.proven and after.status == before.status and after.reason == before.reason
