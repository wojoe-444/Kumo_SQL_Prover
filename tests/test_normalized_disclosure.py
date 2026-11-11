"""What a proof found after algebraic normalization discloses about the queries as written."""

import pytest

pytest.importorskip("z3")

from kumo_sql_prover import float_sum_order
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import BASE_ASSUMPTIONS, ERROR_REFINES_ASSUMPTION, ERROR_SAME_ASSUMPTION

SCHEMA = {"t": ["x", "y", "f"]}
TYPES = {"t": {"x": "INT64", "y": "INT64", "f": "FLOAT64"}}
NOT_MODELED, ROW_ORDER = BASE_ASSUMPTIONS[1], BASE_ASSUMPTIONS[2]


def prove(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES)


MOVED = [
    # the rewrite adds up the groups that WHERE kept out: two INT64_MAX rows with y = 0 overflow on the right only
    ("SELECT y, SUM(x) AS s FROM t WHERE y > 0 GROUP BY y", "SELECT y, SUM(x) AS s FROM t GROUP BY y HAVING y > 0"),
    ("SELECT y, SUM(DISTINCT x) AS s FROM t WHERE y > 0 GROUP BY y", "SELECT y, SUM(DISTINCT x) AS s FROM t GROUP BY y HAVING y > 0"),
    # and the other way round: the rewrite fails on fewer databases
    ("SELECT y, SUM(x) AS s FROM t GROUP BY y HAVING y > 0", "SELECT y, SUM(x) AS s FROM t WHERE y > 0 GROUP BY y"),
]


@pytest.mark.parametrize("left, right", MOVED)
def test_a_proof_after_a_filter_moved_makes_no_claim_about_runtime_errors(left, right):
    result = prove(left, right)
    assert result.proven
    assert result.errors is None or result.errors.verdict == "unknown"
    assert NOT_MODELED in result.assumptions
    assert ERROR_SAME_ASSUMPTION not in result.assumptions and ERROR_REFINES_ASSUMPTION not in result.assumptions


KEPT = [
    # spelling only: the prover itself sees each query equal to its normal form with the same operations able to fail
    ("SELECT x + 1 AS v FROM t WHERE y > 0", "select X + 1 as v from t where 0 < y", "same"),
    ("SELECT 10 / x AS v FROM t", "SELECT 10 / x AS v FROM (SELECT x FROM t)", "same"),
    ("SELECT x FROM t WHERE y > 0", "SELECT x FROM t WHERE NOT y <= 0", "none"),
    ("SELECT y, SUM(x) AS s FROM t WHERE y > 0 GROUP BY y", "SELECT y, SUM(x) AS s FROM t WHERE 0 < y GROUP BY 1", "same"),
]


@pytest.mark.parametrize("left, right, verdict", KEPT)
def test_the_error_comparison_stands_when_normalization_moved_nothing(left, right, verdict):
    result = prove(left, right)
    assert result.proven and result.errors.verdict == verdict
    assert (NOT_MODELED in result.assumptions) is False


def test_a_sum_of_partial_sums_is_not_the_same_plan():
    # (x, f) = (1, 1e16), (1, 1), (2, 1), (2, 1): the partial sums add to 10000000000000002, no single sum does
    result = prove("SELECT SUM(f) AS v FROM t", "SELECT SUM(s) AS v FROM (SELECT SUM(f) AS s FROM t GROUP BY x)")
    if result.proven:
        assert ROW_ORDER in result.assumptions
        assert float_sum_order.SAME_PLAN_ASSUMPTION not in result.assumptions


def test_two_spellings_of_one_filter_are_not_the_same_plan():
    result = prove("SELECT SUM(f) AS s FROM t WHERE y > 0", "SELECT SUM(f) AS s FROM t WHERE 0 < y")
    assert result.proven and ROW_ORDER in result.assumptions
    assert float_sum_order.SAME_PLAN_ASSUMPTION not in result.assumptions


def test_the_same_sum_written_the_same_way_keeps_the_narrow_assumption():
    result = prove("SELECT SUM(f) AS s, y FROM t WHERE y > 0 GROUP BY y", "SELECT SUM(f) AS s, y AS y FROM t WHERE y > 0 GROUP BY y")
    assert result.proven and float_sum_order.SAME_PLAN_ASSUMPTION in result.assumptions
    assert ROW_ORDER not in result.assumptions


def test_an_integer_sum_needs_neither_assumption():
    result = prove("SELECT SUM(x) AS v FROM t", "SELECT SUM(s) AS v FROM (SELECT SUM(x) AS s FROM t GROUP BY y)")
    assert result.proven
    assert ROW_ORDER not in result.assumptions and float_sum_order.SAME_PLAN_ASSUMPTION not in result.assumptions


def test_other_dialects_carry_no_error_comparison_and_are_not_checked_again():
    left, right = "SELECT y, SUM(x) AS s FROM t WHERE y > 0 GROUP BY y", "SELECT y, SUM(x) AS s FROM t GROUP BY y HAVING y > 0"
    result = prove_equivalent_algebraic(left, right, schema=SCHEMA, dialect="postgres")
    assert result.proven and result.errors is None
