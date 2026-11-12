"""An INT64 sum over a derived table or UNION drops the row-order assumption; anything unplaced keeps it."""

from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp

from kumo_sql_prover import sum_argument_types
from kumo_sql_prover.smt_equivalence import BASE_ASSUMPTIONS, SmtStatus, prove_equivalent_smt

SCHEMA = {"t": ["x", "y", "f", "s"], "u": ["x", "f"]}
TYPES = {"t": {"x": "INT64", "y": "INT64", "f": "FLOAT64", "s": "STRING"}, "u": {"x": "FLOAT64", "f": "INT64"}}
ROW_ORDER = BASE_ASSUMPTIONS[2]


def _is_int64(sql: str, types=TYPES) -> bool:
    tree = sqlglot.parse_one(sql, read="bigquery")
    call = next(node for node in tree.find_all(exp.Sum) if node.find_ancestor(exp.Sum) is None and node.find_ancestor(exp.Subquery, exp.CTE) is None)
    return sum_argument_types.is_int64(call, types)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SUM(v) FROM (SELECT x AS v FROM t UNION ALL SELECT y AS v FROM t)",
        "SELECT SUM(v) FROM (SELECT x AS v FROM t UNION ALL SELECT y FROM t) AS d",
        "SELECT SUM(d.v) FROM (SELECT x + 1 AS v FROM t) AS d",
        "SELECT SUM(v) FROM (SELECT COALESCE(x, 0) * 2 AS v FROM t UNION ALL SELECT -y FROM t UNION ALL SELECT 7 FROM t)",
        "WITH a AS (SELECT x AS v FROM t), b AS (SELECT v FROM a UNION ALL SELECT y FROM t) SELECT SUM(v) FROM b",
        "SELECT SUM(v) FROM (SELECT s, COUNT(*) AS v FROM t GROUP BY s)",
        "SELECT SUM(v) FROM (SELECT s, SUM(x) AS v FROM t GROUP BY s)",
        "SELECT SUM(v) FROM (SELECT CAST(f AS INT64) AS v FROM t)",
        "SELECT SUM(v) FROM (SELECT IF(s = 'a', x, y) AS v FROM t)",
        "SELECT SUM(v) FROM (SELECT CASE WHEN s = 'a' THEN x WHEN s = 'b' THEN NULL ELSE 0 END AS v FROM t)",
        "SELECT SUM(v) FROM (SELECT v FROM (SELECT x AS v FROM t))",
        "SELECT SUM(v + x) FROM (SELECT y AS v, x FROM t)",
        "SELECT SUM(a.v) FROM (SELECT x AS v, s FROM t) AS a JOIN (SELECT s FROM t) AS b ON a.s = b.s",
    ],
)
def test_int64_through_derived_tables(sql):
    assert _is_int64(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SUM(v) FROM (SELECT x AS v FROM t UNION ALL SELECT f AS v FROM t)",  # one branch is FLOAT64
        "SELECT SUM(v) FROM (SELECT x AS v FROM t UNION ALL SELECT x AS v FROM u)",  # u.x is FLOAT64
        "SELECT SUM(v) FROM (SELECT f AS w, x AS v FROM t UNION ALL SELECT y, f FROM t)",  # matched by position
        "SELECT SUM(v) FROM (SELECT x / 1 AS v FROM t)",  # / is FLOAT64
        "SELECT SUM(v) FROM (SELECT x * 1.0 AS v FROM t)",
        "SELECT SUM(v) FROM (SELECT x + f AS v FROM t)",
        "SELECT SUM(v) FROM (SELECT * FROM t) AS d(v)",
        "SELECT SUM(x) FROM (SELECT * FROM t)",  # a star's columns are not traced
        "SELECT SUM(v) FROM (SELECT x AS v FROM missing)",  # undeclared table
        "SELECT SUM(x) FROM (SELECT x FROM t) AS a JOIN (SELECT x FROM u) AS b ON TRUE",  # two owners
        "SELECT SUM(x) FROM (SELECT x FROM t) AS a JOIN missing AS b ON TRUE",  # the undeclared table may own x
        "SELECT SUM(v) FROM (SELECT AVG(x) AS v FROM t)",
        "SELECT SUM(v) FROM (SELECT SAFE_CAST(s AS FLOAT64) AS v FROM t)",
        "SELECT SUM(v) FROM (SELECT IF(s = 'a', x, f) AS v FROM t)",
        "SELECT SUM(v) FROM (SELECT x AS v, f AS v FROM t)",  # the name is ambiguous
        "SELECT SUM(v) FROM (SELECT SUM(x) OVER (ORDER BY f) / 2 AS v FROM t)",
        "SELECT SUM(DISTINCT v) FROM (SELECT f AS v FROM t)",
        "WITH RECURSIVE a AS (SELECT x AS v FROM t) SELECT SUM(v) FROM a",
        "WITH a(v) AS (SELECT x FROM t) SELECT SUM(v) FROM a",
    ],
)
def test_anything_else_is_not_placed(sql):
    assert not _is_int64(sql)


def test_a_definition_sees_only_the_definitions_before_it():
    # inside ``a`` the name ``u`` is the base table (x is FLOAT64 there), not the later definition
    sql = "WITH a AS (SELECT x AS v FROM u), u AS (SELECT x FROM t) SELECT SUM(v) FROM a"
    assert not _is_int64(sql)
    assert _is_int64("WITH u AS (SELECT x FROM t), a AS (SELECT x AS v FROM u) SELECT SUM(v) FROM a")


def test_a_select_alias_outside_the_select_list_is_not_followed():
    tree = sqlglot.parse_one("SELECT s, f AS v FROM (SELECT s, f, x AS v FROM t) GROUP BY s, f ORDER BY SUM(v)", read="bigquery")
    assert not sum_argument_types.is_int64(tree.find(exp.Sum), TYPES)


def test_no_types_means_no_answer():
    assert not _is_int64("SELECT SUM(v) FROM (SELECT x AS v FROM t)", types={})


def _prove(left: str, right: str):
    return prove_equivalent_smt(left, right, schema=SCHEMA, types=TYPES)


def test_an_int64_sum_over_swapped_union_branches_needs_no_order_assumption():
    result = _prove(
        "SELECT SUM(v) AS s FROM (SELECT x AS v FROM t UNION ALL SELECT y AS v FROM t)",
        "SELECT SUM(v) AS s FROM (SELECT y AS v FROM t UNION ALL SELECT x AS v FROM t)",
    )
    assert result.status is SmtStatus.PROVEN_EQUIVALENT and ROW_ORDER not in result.assumptions


def test_a_float64_branch_keeps_the_order_assumption():
    result = _prove(
        "SELECT SUM(v) AS s FROM (SELECT x AS v FROM t UNION ALL SELECT f AS v FROM t)",
        "SELECT SUM(v) AS s FROM (SELECT f AS v FROM t UNION ALL SELECT x AS v FROM t)",
    )
    assert result.status is SmtStatus.PROVEN_EQUIVALENT and ROW_ORDER in result.assumptions


def test_without_declared_types_the_assumption_stays():
    result = prove_equivalent_smt(
        "SELECT SUM(v) AS s FROM (SELECT x AS v FROM t UNION ALL SELECT y AS v FROM t)",
        "SELECT SUM(v) AS s FROM (SELECT y AS v FROM t UNION ALL SELECT x AS v FROM t)",
        schema=SCHEMA,
    )
    assert result.status is SmtStatus.PROVEN_EQUIVALENT and ROW_ORDER in result.assumptions


DECIMALS = {"t": {"x": "INT64", "d": "DECIMAL(7, 2)", "e": "NUMERIC(10, 4)", "n": "NUMERIC", "f": "FLOAT64"}}


def _is_exact(sql: str) -> bool:
    tree = sqlglot.parse_one(sql, read="bigquery")
    call = next(node for node in tree.find_all(exp.Sum) if node.find_ancestor(exp.Sum) is None and node.find_ancestor(exp.Subquery, exp.CTE) is None)
    return sum_argument_types.is_exact(call, DECIMALS)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SUM(x) FROM t",
        "SELECT SUM(d) FROM t",
        "SELECT SUM(d + e) FROM t",
        "SELECT SUM(d * e) FROM t",  # 2 + 4 places
        "SELECT SUM(d * x) FROM t",
        "SELECT SUM(d * (1 - d)) FROM t",
        "SELECT SUM(d * 0.25) FROM t",
        "SELECT SUM(n * x) FROM t",
        "SELECT SUM(CAST(f AS DECIMAL(10, 2))) FROM t",
        "SELECT SUM(v) FROM (SELECT d AS v FROM t UNION ALL SELECT e FROM t)",
        "SELECT SUM(v) FROM (SELECT SUM(d * x) AS v FROM t GROUP BY x)",
        "SELECT SUM(CASE WHEN x > 1 THEN d ELSE 0 END) FROM t",
    ],
)
def test_exact_values_include_fixed_point(sql):
    assert _is_exact(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SUM(f) FROM t",
        "SELECT SUM(d + f) FROM t",
        "SELECT SUM(d / e) FROM t",
        "SELECT SUM(x * 0.5) FROM t",
        "SELECT SUM(0.5) FROM t",
        "SELECT SUM(d * 0.25 * e * e) FROM t",  # 2 + 2 + 4 + 4 places
        "SELECT SUM(n * n) FROM t",
        "SELECT SUM(v) FROM (SELECT d AS v FROM t UNION ALL SELECT f FROM t)",
        "SELECT SUM(d * 1e-1) FROM t",
    ],
)
def test_values_that_may_round_are_not_exact(sql):
    assert not _is_exact(sql)


def test_is_int64_still_refuses_fixed_point():
    tree = sqlglot.parse_one("SELECT SUM(d) FROM t", read="bigquery")
    assert not sum_argument_types.is_int64(tree.find(exp.Sum), DECIMALS)


def test_an_average_is_exact_when_its_values_are():
    def exact(sql):
        return sum_argument_types.is_exact(sqlglot.parse_one(sql, read="bigquery").find(exp.Avg), DECIMALS)

    assert exact("SELECT AVG(v) FROM (SELECT SUM(d * x) AS v FROM t GROUP BY x)")
    assert not exact("SELECT AVG(v) FROM (SELECT SUM(f) AS v FROM t GROUP BY x)")
    # an average is not itself a value that adds exactly
    assert not _is_exact("SELECT SUM(v) FROM (SELECT AVG(d) AS v FROM t GROUP BY x)")
