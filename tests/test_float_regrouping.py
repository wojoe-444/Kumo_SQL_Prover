"""A FLOAT64 sum is not associative: folding a sum of partial sums into one sum is not a proof (float_regrouping.py)."""

import pytest

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.float_regrouping import UNTYPED_ASSUMPTION
from kumo_sql_prover.smt_equivalence import SmtStatus

TYPES = {"t": {"x": "INT64", "f": "FLOAT64", "g": "FLOAT64", "i": "INT64", "j": "INT64"}}
SCHEMA = {"t": list(TYPES["t"])}

PARTIAL = (
    "SELECT SUM(f) AS v FROM t",
    "SELECT SUM(s) AS v FROM (SELECT SUM(f) AS s FROM t GROUP BY x)",
)
PARTIAL_INT = (
    "SELECT SUM(i) AS v FROM t",
    "SELECT SUM(s) AS v FROM (SELECT SUM(i) AS s FROM t GROUP BY x)",
)


def prove(pair, types=TYPES):
    kwargs = {"types": types} if types else {}
    return prove_equivalent_algebraic(*pair, schema=SCHEMA, **kwargs)


def test_a_sum_of_float_partial_sums_is_not_proven_equal_to_one_sum():
    assert prove(PARTIAL).status is not SmtStatus.PROVEN_EQUIVALENT


def test_a_sum_of_integer_partial_sums_still_proves():
    assert prove(PARTIAL_INT).status is SmtStatus.PROVEN_EQUIVALENT


def test_without_declared_types_the_proof_names_the_regrouping():
    result = prove(PARTIAL, types=None)
    assert result.status is SmtStatus.PROVEN_EQUIVALENT
    assert UNTYPED_ASSUMPTION in result.assumptions


def test_a_pair_that_only_reorders_a_float_sum_keeps_its_proof():
    result = prove(("SELECT SUM(f) AS v FROM t WHERE x > 0", "SELECT SUM(f) AS v FROM (SELECT f FROM t WHERE x > 0)"))
    assert result.status is SmtStatus.PROVEN_EQUIVALENT
    assert UNTYPED_ASSUMPTION not in result.assumptions


JOINED_TYPES = {"t": {"k": "INT64", "f": "FLOAT64"}, "u": {"k": "INT64", "w": "INT64"}}
JOINED = (
    "SELECT SUM(d.s * u.w) AS v FROM (SELECT k, SUM(f) AS s FROM t GROUP BY k) AS d JOIN u ON d.k = u.k",
    "SELECT SUM(t.f * u.w) AS v FROM t JOIN u ON t.k = u.k",
)


def test_a_pre_aggregated_float_sum_times_a_weight_is_not_the_flat_sum():
    result = prove_equivalent_algebraic(*JOINED, schema={"t": ["k", "f"], "u": ["k", "w"]}, types=JOINED_TYPES)
    assert result.status is not SmtStatus.PROVEN_EQUIVALENT


def test_the_same_pre_aggregation_over_integers_still_proves():
    types = {**JOINED_TYPES, "t": {"k": "INT64", "f": "INT64"}}
    result = prove_equivalent_algebraic(*JOINED, schema={"t": ["k", "f"], "u": ["k", "w"]}, types=types)
    assert result.status is SmtStatus.PROVEN_EQUIVALENT
