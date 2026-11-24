"""The twenty-seventh batch is one more step of a proof, after every other (``shape_fallback.TWENTY_SEVENTH_BATCH``)."""

import re

import pytest

from kumo_sql_prover import algebraic_equivalence, lookup_pivots, shape_fallback, single_row_aggregates
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"prices": ["item", "shop", "cost"]}
TYPES = {"prices": {"item": "INT", "shop": "VARCHAR", "cost": "INT"}}
KEYS = {"prices": [("item", "shop")]}
NOT_NULL = {"prices": frozenset({"item", "shop"})}
CONSTRAINTS = {"prices": TableConstraints(not_null=NOT_NULL["prices"], keys=(("item", "shop"),))}
OPTIONS = {"schema": SCHEMA, "keys": KEYS, "not_null": NOT_NULL, "types": TYPES, "dialect": "mysql"}

# one row per item with the cost in each of two shops
SUMMED = "SELECT item, SUM(CASE WHEN shop = 'north' THEN cost END) AS north, SUM(CASE WHEN shop = 'south' THEN cost END) AS south FROM prices GROUP BY item"
GREATEST = SUMMED.replace("SUM(", "MAX(")
LEAST = "SELECT item, MIN(IF(shop = 'north', cost, NULL)) AS north, MIN(IF(shop = 'south', cost, NULL)) AS south FROM prices GROUP BY 1"
JOINED = (
    "SELECT i.item, n.cost AS north, s.cost AS south FROM (SELECT DISTINCT item FROM prices) AS i "
    "LEFT JOIN prices AS n ON n.item = i.item AND n.shop = 'north' LEFT JOIN prices AS s ON s.item = i.item AND s.shop = 'south'"
)
LOOKED_UP = (
    "SELECT DISTINCT item, (SELECT cost FROM prices AS n WHERE n.shop = 'north' AND n.item = p.item) AS north, "
    "(SELECT cost FROM prices AS s WHERE s.shop = 'south' AND s.item = p.item) AS south FROM prices AS p"
)

PAIRS = [(SUMMED, GREATEST), (SUMMED, LEAST), (GREATEST, JOINED), (SUMMED, LOOKED_UP), (JOINED, LOOKED_UP)]


def prove(left: str, right: str, **more):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect="mysql", exact_arithmetic=True, timeout_ms=5000, **more
    )


def switch_off(monkeypatch) -> None:
    """Take the rules out and leave every other rule and step as it is (``shape_fallback.without`` also changes the
    steps before)."""

    monkeypatch.setattr(lookup_pivots, "read_lookup_pivots", lambda tree, *facts: (tree, False))
    monkeypatch.setattr(single_row_aggregates, "read_single_row_aggregates", lambda tree, *facts: (tree, False))


def test_the_batch_names_its_rules_and_has_its_own_entry_in_the_stage_key():
    assert shape_fallback.TWENTY_SEVENTH_BATCH == frozenset({"lookup_pivots", "single_row_aggregates"})
    assert len(shape_fallback.stage_key()) == 13 and shape_fallback.stage_key()[12] is False
    with shape_fallback.twenty_seventh_deferred():
        assert shape_fallback.stage_key()[12] is False
    with shape_fallback.twenty_fifth_on():
        assert shape_fallback.stage_key()[5:] == (True, True, True, True, True, True, True, False)  # the step before leaves this one off
    with shape_fallback.twenty_seventh_on():
        assert shape_fallback.stage_key()[5:] == (True, True, True, True, True, True, True, True)  # the step runs with the batches before it on too
    assert shape_fallback.stage_key()[12] is False


def normal(sql: str) -> str:
    """The normal form, with the numbers of the names the normalizer makes up taken out (they count up per call)."""

    return re.sub(r"\bkq([a-z]+)\d+", r"kq\1", normalize(sql, **OPTIONS))


def grouped(alias: str) -> str:
    """The grouped spelling with the table read under ``alias``, as ``lookup_pivots`` writes it."""

    return f"SELECT {alias}.item, MAX(CASE WHEN {alias}.shop = 'north' THEN {alias}.cost END) AS north, MAX(CASE WHEN {alias}.shop = 'south' THEN {alias}.cost END) AS south FROM prices AS {alias} GROUP BY {alias}.item"


@pytest.mark.parametrize(
    "sql, rule, read_as",
    [(SUMMED, "single_row_aggregates", GREATEST), (LEAST, "single_row_aggregates", GREATEST), (JOINED, "lookup_pivots", grouped("i")), (LOOKED_UP, "lookup_pivots", grouped("p"))],
)
def test_a_rule_fires_only_in_the_last_step_of_a_proof(sql, rule, read_as):
    assert not shape_fallback.twenty_seventh_batch_enabled()
    with shape_fallback.tracking() as fired:
        outside = normal(sql)  # outside a proof the rules are off
    assert not fired & shape_fallback.TWENTY_SEVENTH_BATCH
    with shape_fallback.twenty_seventh_deferred() as seen:
        assert shape_fallback.twenty_seventh_watching() and not shape_fallback.twenty_seventh_batch_enabled()
        assert normal(sql) == outside  # an earlier step asks the rules and changes nothing
        assert seen == [True] and not shape_fallback.twenty_seventh_watching()
    with shape_fallback.twenty_seventh_deferred() as seen:
        normal(GREATEST)  # nothing to read
        normal("SELECT item, SUM(cost) AS total FROM prices GROUP BY item")
        with shape_fallback.disabled():
            normal(sql)
        assert seen == []
    with shape_fallback.twenty_seventh_deferred(watch=False) as seen:
        assert seen is None and not shape_fallback.twenty_seventh_watching()
        assert normal(sql) == outside
    with shape_fallback.twenty_seventh_on(), shape_fallback.tracking() as fired:
        assert shape_fallback.twenty_seventh_batch_enabled() and shape_fallback.twenty_fifth_batch_enabled() and shape_fallback.twentieth_batch_enabled()
        assert normal(sql) == normal(read_as) != outside
        with shape_fallback.without(rule), shape_fallback.tracking() as inner:
            assert normal(sql) != normal(read_as)
        assert rule not in inner
        for off in (shape_fallback.disabled, shape_fallback.late):
            with off():
                assert not shape_fallback.twenty_seventh_batch_enabled()
    assert rule in fired
    assert not shape_fallback.twenty_seventh_batch_enabled() and not shape_fallback.twenty_seventh_watching()


def runs_of(monkeypatch) -> list:
    runs: list = []
    whole = algebraic_equivalence._prove_equivalent_algebraic_whole

    def counted(*args, **kwargs):
        runs.append((shape_fallback.twenty_fifth_batch_enabled(), shape_fallback.twenty_seventh_batch_enabled()))
        return whole(*args, **kwargs)

    monkeypatch.setattr(algebraic_equivalence, "_prove_equivalent_algebraic_whole", counted)
    return runs


@pytest.mark.parametrize("position", range(len(PAIRS)))
def test_a_pair_is_proved_in_one_more_run_and_only_with_the_rules(monkeypatch, position):
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
    assert not prove(SUMMED, GREATEST.replace("'south'", "'east'")).proven
    assert not prove(SUMMED, JOINED.replace("s.shop = 'south'", "s.shop = 'east'")).proven
    assert not prove(GREATEST, JOINED.replace("LEFT JOIN prices AS s", "JOIN prices AS s")).proven  # drops the items with no south row
    assert not prove(GREATEST, LOOKED_UP.replace("SELECT DISTINCT item", "SELECT item")).proven  # a row per row of the item
    # without the key the spellings are not the same, and nothing of the batch is read
    unkeyed = {"prices": TableConstraints(not_null=NOT_NULL["prices"])}
    for left, right in PAIRS:
        assert not prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, constraints=unkeyed, compare_names=False, dialect="mysql", exact_arithmetic=True, timeout_ms=5000).proven


def test_a_pair_the_rules_do_not_read_and_a_pair_proved_before_get_no_run_of_the_step(monkeypatch):
    runs = runs_of(monkeypatch)
    assert not prove("SELECT item FROM prices WHERE cost > 2", "SELECT item FROM prices WHERE cost > 3").proven
    assert runs and not any(last for _, last in runs)
    runs.clear()
    assert prove(SUMMED, SUMMED.replace("GROUP BY item", "GROUP BY 1")).proven  # proved before the step: no run of it
    assert runs and not any(last for _, last in runs)
    runs.clear()
    assert not prove(SUMMED, GREATEST.replace("'south'", "'east'")).proven  # read, and still different
    assert any(last for _, last in runs)


def test_a_pair_over_the_size_limit_does_not_get_the_step(monkeypatch):
    assert shape_fallback.last_steps_worth_trying(SUMMED, JOINED)
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(SUMMED) + len(JOINED) - 1)
    runs = runs_of(monkeypatch)
    assert not prove(SUMMED, JOINED).proven
    assert runs and not any(last for _, last in runs)
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(SUMMED) + len(JOINED))
    assert prove(SUMMED, JOINED).proven


def test_the_step_gives_proofs_only(monkeypatch):
    # The run of the step looks for no counterexample: a pair it does not prove keeps the answer of the steps before.
    different = GREATEST.replace("'south'", "'east'")
    after = prove(SUMMED, different)
    switch_off(monkeypatch)
    before = prove(SUMMED, different)
    assert not after.proven and after.status == before.status and after.reason == before.reason
