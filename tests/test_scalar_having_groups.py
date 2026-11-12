"""A grouped ``IN`` subquery whose ``HAVING`` reads an uncorrelated scalar subquery (``kumo_sql_prover.scalar_having_groups``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import _grouped_in_to_derived, normalize, prove_equivalent_algebraic
from kumo_sql_prover.scalar_having_groups import PLACEHOLDER, read_scalar_having_groups

SCHEMA = {"t": ["k", "v"], "u": ["k", "m"], "s": ["x", "y"]}
TYPES = {"t": {"k": "BIGINT", "v": "BIGINT"}, "u": {"k": "BIGINT", "m": "BIGINT"}, "s": {"x": "BIGINT", "y": "BIGINT"}}


def lifted(sql, schema=SCHEMA):
    return read_scalar_having_groups(sqlglot.parse_one(sql, read="postgres"), schema, _grouped_in_to_derived).sql(dialect="postgres")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="postgres").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name, columns in SCHEMA.items():
        db.execute(f"CREATE TABLE {name} ({', '.join(c + ' BIGINT' for c in columns)})")
        insert_rows(db, name, tables.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="postgres", write="duckdb")[0] for sql in queries])]


IN_SCALAR = "SELECT y FROM s WHERE x IN (SELECT k FROM t GROUP BY k HAVING SUM(v) > 2 * (SELECT MAX(m) FROM u))"


def test_the_scalar_subquery_moves_with_the_having():
    text = lifted(IN_SCALAR)
    assert PLACEHOLDER not in text
    assert "HAVING" not in text
    assert "> 2 * (SELECT MAX(m) FROM u)" in text
    assert "GROUP BY k) AS kqg" in text


def test_what_is_left_alone():
    for sql in (
        # the subquery reads the group's key: one value per group, not one value
        "SELECT y FROM s WHERE x IN (SELECT k FROM t GROUP BY k HAVING SUM(v) > (SELECT MAX(m) FROM u WHERE u.k = t.k))",
        # a bare column no source of the subquery has is the group's
        "SELECT y FROM s WHERE x IN (SELECT k FROM t GROUP BY k HAVING SUM(v) > (SELECT MAX(m + v) FROM u))",
        "SELECT y FROM s WHERE x IN (SELECT k FROM t GROUP BY k HAVING SUM(v) > (SELECT m FROM u ORDER BY m LIMIT 1))",
        "SELECT y FROM s WHERE x IN (SELECT k FROM t GROUP BY k HAVING EXISTS (SELECT 1 FROM u))",
        "SELECT y FROM s WHERE x IN (SELECT k FROM t GROUP BY k HAVING SUM(v) IN (SELECT m FROM u))",
        # no HAVING, or not grouped
        "SELECT y FROM s WHERE x IN (SELECT k FROM t WHERE v > (SELECT MAX(m) FROM u))",
    ):
        assert lifted(sql) == sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres"), sql
    # without a schema nothing shows that a bare column is bound inside the subquery
    assert lifted(IN_SCALAR, schema=None) == sqlglot.parse_one(IN_SCALAR, read="postgres").sql(dialect="postgres")


def test_a_table_missing_from_the_schema_is_left_alone():
    sql = "SELECT y FROM s WHERE x IN (SELECT k FROM t GROUP BY k HAVING SUM(v) > (SELECT MAX(m) FROM elsewhere))"
    assert lifted(sql) == sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres")


def test_pairs_it_proves():
    # the same test over the groups, with the group's sources joined two ways
    assert proved(
        "SELECT y FROM s WHERE x IN (SELECT t.k FROM t, u WHERE t.k = u.k GROUP BY t.k HAVING SUM(t.v) > 2 * (SELECT MAX(m) FROM u))",
        "SELECT y FROM s WHERE x IN (SELECT t.k FROM t JOIN u ON t.k = u.k GROUP BY t.k HAVING SUM(t.v) > 2 * (SELECT MAX(m) FROM u))",
    )
    # the subquery itself written two ways
    assert proved(
        "SELECT y FROM s WHERE x IN (SELECT k FROM t GROUP BY k HAVING SUM(v) > (SELECT MAX(m) FROM u, s WHERE u.k = s.x))",
        "SELECT y FROM s WHERE x IN (SELECT k FROM t GROUP BY k HAVING SUM(v) > (SELECT MAX(m) FROM u JOIN s ON u.k = s.x))",
    )
    # an unread output of the groups dropped, through a CTE
    assert proved(
        "WITH g AS (SELECT k, SUM(v) AS total FROM t GROUP BY k HAVING SUM(v) > (SELECT MAX(m) FROM u)) SELECT y FROM s WHERE x IN (SELECT k FROM g)",
        "WITH g AS (SELECT k FROM t GROUP BY k HAVING SUM(v) > (SELECT MAX(m) FROM u)) SELECT y FROM s WHERE x IN (SELECT k FROM g)",
    )


def test_near_misses_stay_unproven():
    base = "SELECT y FROM s WHERE x IN (SELECT k FROM t GROUP BY k HAVING SUM(v) {} (SELECT {} FROM u))"
    assert not proved(base.format(">", "MAX(m)"), base.format(">=", "MAX(m)"))
    assert not proved(base.format(">", "MAX(m)"), base.format(">", "MIN(m)"))
    assert not proved(base.format(">", "MAX(m)"), base.format(">", "MAX(k)"))
    assert not proved(base.format(">", "MAX(m)"), "SELECT y FROM s WHERE x IN (SELECT k FROM t GROUP BY k HAVING MAX(v) > (SELECT MAX(m) FROM u))")
    # the threshold over other rows
    assert not proved(base.format(">", "MAX(m)"), "SELECT y FROM s WHERE x IN (SELECT k FROM t GROUP BY k HAVING SUM(v) > (SELECT MAX(m) FROM u WHERE k > 0))")
    # the test dropped
    assert not proved(base.format(">", "MAX(m)"), "SELECT y FROM s WHERE x IN (SELECT k FROM t GROUP BY k)")


def test_the_lift_returns_the_same_rows_on_duckdb():
    rng = random.Random(20261111)
    queries = (
        IN_SCALAR,
        "SELECT y FROM s WHERE x IN (SELECT k FROM t WHERE v > 0 GROUP BY k HAVING COUNT(*) >= (SELECT COUNT(*) FROM u) AND SUM(v) < 3 * (SELECT MIN(m) FROM u WHERE k > 1))",
        "SELECT y FROM s WHERE x NOT IN (SELECT k FROM t GROUP BY k HAVING SUM(v) > (SELECT MAX(m) FROM u))",
    )
    value = lambda: rng.choice([None, 0, 1, 2, 3, 5])
    for _ in range(60):
        tables = {name: [tuple(value() for _ in columns) for _ in range(rng.randint(0, 6))] for name, columns in SCHEMA.items()}
        for sql in queries:
            after = lifted(sql)
            assert after != sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres")
            before_rows, after_rows = run(tables, sql, after)
            assert before_rows == after_rows, (sql, tables)


def test_off_on_the_second_attempt():
    with shape_fallback.disabled():
        assert "HAVING" in normalize(IN_SCALAR, schema=SCHEMA, dialect="postgres", types=TYPES)
    assert "HAVING" not in normalize(IN_SCALAR, schema=SCHEMA, dialect="postgres", types=TYPES)
