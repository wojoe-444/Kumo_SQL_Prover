"""Rows without a match counted with a flag are proved equal to the anti join, in the last step of a proof
(``unmatched_row_counts``, a rule of ``shape_fallback.TWENTIETH_BATCH``)."""

import pytest

from kumo_sql_prover import algebraic_equivalence, shape_fallback, unmatched_row_counts
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtStatus, TableConstraints

SCHEMA = {"visits": ["visit", "customer"], "purchases": ["id", "visit", "amount"]}
TYPES = {"visits": {"visit": "INT", "customer": "INT"}, "purchases": {"id": "INT", "visit": "INT", "amount": "INT"}}
CONSTRAINTS = {
    "visits": TableConstraints(not_null=frozenset({"visit"}), keys=(("visit",),)),
    "purchases": TableConstraints(not_null=frozenset({"id", "visit"}), keys=(("id",),)),
}
OPTIONS = {
    "schema": SCHEMA,
    "keys": {"visits": [("visit",)], "purchases": [("id",)]},
    "not_null": {"visits": frozenset({"visit"}), "purchases": frozenset({"id", "visit"})},
    "types": TYPES,
    "dialect": "mysql",
}

JOIN = "FROM visits AS v LEFT JOIN purchases AS p ON p.visit = v.visit"
# per customer, the visits without a purchase, five ways
ANTI = f"SELECT v.customer, COUNT(*) AS n {JOIN} WHERE p.id IS NULL GROUP BY v.customer"
NOT_IN = "SELECT customer, COUNT(*) AS n FROM visits WHERE visit NOT IN (SELECT visit FROM purchases) GROUP BY customer"
FLAGGED = f"SELECT v.customer, SUM(CASE WHEN p.id IS NULL THEN 1 ELSE 0 END) AS n {JOIN} GROUP BY v.customer HAVING n > 0"
LESS = f"SELECT v.customer, COUNT(v.visit) - COUNT(p.id) AS n {JOIN} GROUP BY v.customer HAVING COUNT(v.visit) - COUNT(p.id) <> 0"
OUTSIDE = f"SELECT d.customer, d.n FROM (SELECT v.customer, SUM(p.visit IS NULL) AS n {JOIN} GROUP BY v.customer) AS d WHERE d.n >= 1"

PAIRS = [(reference, counted) for reference in (ANTI, NOT_IN) for counted in (FLAGGED, LESS, OUTSIDE)]


def prove(left: str, right: str, constraints=CONSTRAINTS, **more):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, types=TYPES, constraints=constraints, compare_names=False, dialect="mysql", exact_arithmetic=True, timeout_ms=5000, **more
    )


def switch_off(monkeypatch) -> None:
    """Take the rule out and leave every other rule and step as it is."""

    monkeypatch.setattr(unmatched_row_counts, "read_unmatched_row_counts", lambda tree, *args: (tree, False))


def runs_of(monkeypatch) -> list:
    runs: list = []
    whole = algebraic_equivalence._prove_equivalent_algebraic_whole

    def counted(*args, **kwargs):
        runs.append(shape_fallback.twentieth_batch_enabled())
        return whole(*args, **kwargs)

    monkeypatch.setattr(algebraic_equivalence, "_prove_equivalent_algebraic_whole", counted)
    return runs


def test_the_rule_fires_only_in_the_last_step_of_a_proof():
    with shape_fallback.tracking() as fired:
        outside = normalize(FLAGGED, **OPTIONS)  # outside a proof the rule is off
    assert "unmatched_row_counts" not in fired
    with shape_fallback.twentieth_deferred() as seen:
        assert normalize(FLAGGED, **OPTIONS) == outside  # an earlier step asks the rule and changes nothing
        assert seen == [True]
    with shape_fallback.twentieth_deferred() as seen:
        normalize(ANTI, **OPTIONS)  # nothing to read
        with shape_fallback.disabled():
            normalize(FLAGGED, **OPTIONS)
        assert seen == []
    with shape_fallback.twentieth_on(), shape_fallback.tracking() as fired:
        assert normalize(FLAGGED, **OPTIONS) == normalize(ANTI, **OPTIONS) != outside
        with shape_fallback.without("unmatched_row_counts"), shape_fallback.tracking() as inner:
            assert normalize(FLAGGED, **OPTIONS) == outside
        assert "unmatched_row_counts" not in inner
    assert "unmatched_row_counts" in fired


@pytest.mark.parametrize("position", range(len(PAIRS)))
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
    # without the test, a customer whose visits all have a purchase is a row with a 0
    assert not prove(ANTI, FLAGGED.replace(" HAVING n > 0", "")).proven
    # a purchase without an amount is still a purchase
    assert not prove(ANTI, FLAGGED.replace("p.id IS NULL", "p.amount IS NULL")).proven
    # two visits without a purchase, not one
    assert not prove(ANTI, FLAGGED.replace("n > 0", "n > 1")).proven
    # the matched visits
    assert not prove(ANTI, FLAGGED.replace("p.id IS NULL", "p.id IS NOT NULL")).proven


def test_the_proof_against_not_in_needs_the_compared_column_declared():
    # NOT IN over a list with a NULL in it keeps no row; the join forms do not mind
    assert prove(NOT_IN, FLAGGED).proven
    open_purchases = dict(CONSTRAINTS, purchases=TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)))
    assert not prove(NOT_IN, FLAGGED, constraints=open_purchases).proven
    assert prove(ANTI, FLAGGED, constraints=open_purchases).proven
    under = prove(NOT_IN, FLAGGED, constraints=open_purchases, conditional=True, conditional_seconds=20)
    assert under.status is SmtStatus.PROVEN_CONDITIONALLY and "visit is NOT NULL" in under.reason


def test_a_pair_the_rule_does_not_read_gets_no_run_of_the_step(monkeypatch):
    runs = runs_of(monkeypatch)
    assert not prove(ANTI, ANTI.replace("p.id IS NULL", "p.amount IS NULL")).proven
    assert runs and not any(runs)
    runs.clear()
    assert not prove(ANTI, FLAGGED.replace("n > 0", "n > 1")).proven  # not read either: the test asks for more
    assert runs and not any(runs)
    runs.clear()
    assert not prove(ANTI.replace("COUNT(*)", "COUNT(*) + 1"), FLAGGED).proven  # read, and still different
    assert any(runs)
