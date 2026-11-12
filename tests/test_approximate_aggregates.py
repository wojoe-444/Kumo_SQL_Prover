"""An approximate aggregate is not proven equal to itself (the identical-text layer used to accept it)."""

import pytest

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtStatus, prove_equivalent_smt

SCHEMA = {"t": ["x", "w"]}
CALLS = ["APPROX_TOP_COUNT(x, 2)", "APPROX_TOP_SUM(x, w, 2)", "APPROX_QUANTILES(x, 4)", "APPROX_COUNT_DISTINCT(x)"]


@pytest.mark.parametrize("call", CALLS)
@pytest.mark.parametrize("prover", [prove_equivalent_smt, prove_equivalent_algebraic])
def test_a_query_with_an_approximate_aggregate_is_not_proven_equal_to_its_copy(call, prover):
    sql = f"SELECT {call} AS a FROM t"
    assert prover(sql, sql, schema=SCHEMA).status is SmtStatus.NOT_PROVEN
