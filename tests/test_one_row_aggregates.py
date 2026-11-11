"""A select that aggregates without GROUP BY is one row: as an IN list, under DISTINCT, as a joined table
(``kumo_sql_prover.one_row_aggregates``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.one_row_aggregates import read_one_row_aggregates

# every column may be NULL and repeat; w is a table whose columns are not supplied
SCHEMA = {"t": ["id", "a", "b"], "u": ["id", "a", "v"]}


def rewritten(sql, schema=SCHEMA):
    before = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    text = before.sql(dialect="mysql")
    after = read_one_row_aggregates(before, schema).sql(dialect="mysql")
    return after if after != text else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, compare_names=False, dialect="mysql").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE t (id BIGINT, a BIGINT, b BIGINT)")
    db.execute("CREATE TABLE u (id BIGINT, a BIGINT, v BIGINT)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


def test_membership_in_one_row_is_a_comparison():
    assert rewritten("SELECT a FROM t GROUP BY a HAVING COUNT(b) IN (SELECT COUNT(*) FROM u)") == (
        "SELECT a FROM t GROUP BY a HAVING COUNT(b) = (SELECT COUNT(*) FROM u)"
    )
    # under NOT, in a select list, correlated: the two agree in all three truth values
    assert rewritten("SELECT id FROM t WHERE b NOT IN (SELECT MAX(v) + 1 FROM u WHERE u.a > 0)") == (
        "SELECT id FROM t WHERE NOT b = (SELECT MAX(v) + 1 FROM u WHERE u.a > 0)"
    )
    assert rewritten("SELECT id, b IN (SELECT MIN(u.v) FROM u WHERE u.a = t.a) AS f FROM t") == (
        "SELECT id, b = (SELECT MIN(u.v) FROM u WHERE u.a = t.a) AS f FROM t"
    )


def test_all_and_any_over_one_row_are_the_comparison():
    assert rewritten("SELECT id FROM t WHERE b = ALL (SELECT COUNT(*) FROM u)") == "SELECT id FROM t WHERE b = (SELECT COUNT(*) FROM u)"
    assert rewritten("SELECT id FROM t WHERE b > ANY (SELECT SUM(v) FROM u)") == "SELECT id FROM t WHERE b > (SELECT SUM(v) FROM u)"
    assert rewritten("SELECT id FROM t WHERE b <> SOME (SELECT AVG(v) FROM u)") == "SELECT id FROM t WHERE b <> (SELECT AVG(v) FROM u)"


def test_distinct_over_one_row_is_dropped():
    assert rewritten("SELECT id FROM t WHERE b = (SELECT DISTINCT COUNT(v) FROM u)") == "SELECT id FROM t WHERE b = (SELECT COUNT(v) FROM u)"
    assert rewritten("SELECT DISTINCT COUNT(*), MAX(a) FROM t") == "SELECT COUNT(*), MAX(a) FROM t"


def test_a_joined_one_row_table_becomes_scalar_subqueries():
    assert rewritten("SELECT x.a FROM (SELECT a, COUNT(b) AS n FROM t GROUP BY a) AS x, (SELECT COUNT(*) AS total FROM u) AS y WHERE x.n = y.total") == (
        "SELECT x.a FROM (SELECT a, COUNT(b) AS n FROM t GROUP BY a) AS x WHERE x.n = (SELECT COUNT(*) FROM u)"
    )
    # two outputs, a bare read, the ON moved to the WHERE, an output that keeps its name
    assert rewritten("SELECT x.a, total FROM (SELECT COUNT(*) AS total, MAX(v) AS top FROM u WHERE v > 0) AS y JOIN t AS x ON x.b = y.top WHERE x.a < total + 1") == (
        "SELECT x.a, (SELECT COUNT(*) FROM u WHERE v > 0) AS total FROM t AS x "
        "WHERE x.b = (SELECT MAX(v) FROM u WHERE v > 0) AND x.a < (SELECT COUNT(*) FROM u WHERE v > 0) + 1"
    )
    # other joins stay inner joins
    assert rewritten("SELECT t.id, y.total - t.b AS rest FROM t CROSS JOIN (SELECT SUM(v) AS total FROM u) AS y JOIN u ON u.id = t.id AND u.v < y.total") == (
        "SELECT t.id, (SELECT SUM(v) FROM u) - t.b AS rest FROM t, u WHERE u.id = t.id AND u.v < (SELECT SUM(v) FROM u)"
    )
    # never read: the row only multiplied by one
    assert rewritten("SELECT t.id FROM t, (SELECT COUNT(*) AS n FROM u) AS y") == "SELECT t.id FROM t"
    assert rewritten("SELECT t.a, COUNT(*) FROM t JOIN (SELECT AVG(v) AS m FROM u) AS y ON t.b > y.m GROUP BY t.a HAVING MAX(t.b) < 2 * MIN(y.m)") == (
        "SELECT t.a, COUNT(*) FROM t WHERE t.b > (SELECT AVG(v) FROM u) GROUP BY t.a HAVING MAX(t.b) < 2 * MIN((SELECT AVG(v) FROM u))"
    )


@pytest.mark.parametrize("sql", [
    "SELECT a FROM t GROUP BY a HAVING COUNT(b) IN (SELECT COUNT(*) FROM u GROUP BY a)",  # a row per group
    "SELECT a FROM t WHERE b IN (SELECT v FROM u)",
    "SELECT a FROM t WHERE b IN (SELECT COUNT(*) FROM u HAVING COUNT(*) > 1)",  # one row or none
    "SELECT a FROM t WHERE b IN (SELECT COUNT(*) FROM u LIMIT 0)",
    "SELECT a FROM t WHERE b IN (SELECT COUNT(*) FROM u LIMIT 1 OFFSET 1)",
    "SELECT a FROM t WHERE (a, b) IN (SELECT COUNT(*), MAX(v) FROM u)",  # a row on the left
    "SELECT a FROM t WHERE b IN (SELECT COUNT(t.b) FROM u)",  # an aggregate of the outer query's column
    "SELECT a FROM t WHERE b IN (SELECT COUNT(*) FROM w)",  # the columns of w are not supplied
    "SELECT a FROM t WHERE b IN (SELECT COUNT(*) OVER () FROM u)",  # a window: a row per row
    "SELECT a FROM t WHERE b IN (SELECT MAX(v) + u.id FROM u)",  # a column outside the aggregate
    "SELECT a FROM t WHERE b IN (SELECT GROUP_CONCAT(v) FROM u)",
    "SELECT a FROM t WHERE b IN (SELECT COUNT(*) FROM u UNION ALL SELECT COUNT(*) FROM t)",
    "SELECT a FROM t WHERE b IN (1, 2)",
    "SELECT a FROM t WHERE b = ALL (SELECT v FROM u)",
    "SELECT a FROM t WHERE b = ANY (SELECT COUNT(*) FROM u GROUP BY a)",
    "SELECT DISTINCT a, COUNT(*) FROM t GROUP BY a",
    "SELECT DISTINCT a FROM t",
    "SELECT x.a FROM t AS x LEFT JOIN (SELECT COUNT(*) AS total FROM u) AS y ON x.b = y.total",  # the row may be padded with NULLs
    "SELECT x.a FROM (SELECT COUNT(*) AS total FROM u) AS y LEFT JOIN t AS x ON x.b = y.total",  # the row is kept without a match
    "SELECT x.a FROM t AS x JOIN u ON u.id = x.id LEFT JOIN (SELECT COUNT(*) AS total FROM u) AS y ON x.b = y.total",
    "SELECT * FROM t AS x, (SELECT COUNT(*) AS total FROM u) AS y",
    "SELECT y.* FROM t AS x, (SELECT COUNT(*) AS total FROM u) AS y",
    "SELECT y.total FROM (SELECT COUNT(*) AS total FROM u) AS y",  # no other source
    "SELECT x.a FROM t AS x, (SELECT COUNT(*) AS id FROM u) AS y WHERE id > 1",  # id is x's too
    "SELECT x.a FROM w AS x, (SELECT COUNT(*) AS total FROM u) AS y WHERE total > 1",  # w may have a column total
    "SELECT x.a FROM t AS x, (SELECT COUNT(*) FROM u) AS y",  # an output without a name
    "SELECT x.a FROM t AS x, (SELECT COUNT(*) AS total, MAX(v) AS total FROM u) AS y WHERE x.b = y.total",
    "SELECT x.a FROM t AS x, (SELECT COUNT(*) AS total FROM u) AS y GROUP BY x.a, y.total",
    "SELECT x.a FROM t AS x, (SELECT COUNT(*) AS total FROM u) AS y ORDER BY y.total",
    "SELECT x.a, SUM(x.b) OVER (PARTITION BY y.total) FROM t AS x, (SELECT COUNT(*) AS total FROM u) AS y",
    "SELECT x.a FROM t AS x, (SELECT COUNT(*) AS total FROM u) AS y WHERE EXISTS (SELECT 1 FROM u WHERE u.v = y.total)",  # read from a nested query
    "SELECT x.a FROM t AS x, (SELECT COUNT(*) AS total FROM u GROUP BY a) AS y WHERE x.b = y.total",
    "SELECT x.a FROM t AS x, (SELECT v AS total FROM u) AS y WHERE x.b = y.total",
    "SELECT x.a AS total FROM t AS x, (SELECT COUNT(*) AS total FROM u) AS y GROUP BY x.a HAVING total > 1",  # total is an alias too
    "SELECT x.a FROM t AS x JOIN (SELECT COUNT(*) AS id FROM u) AS y USING (id)",
    "SELECT x.a FROM t AS x NATURAL JOIN (SELECT COUNT(*) AS id FROM u) AS y",
])
def test_other_queries_stay(sql):
    assert rewritten(sql) is None


def test_the_aggregated_table_has_to_be_known():
    sql = "SELECT a FROM t WHERE b IN (SELECT COUNT(v) FROM u)"
    assert rewritten(sql) is not None
    assert rewritten(sql, schema=None) is None
    assert rewritten(sql, schema={"t": ["id", "a", "b"]}) is None


PROOFS = [
    ("SELECT a FROM t GROUP BY a HAVING COUNT(b) IN (SELECT COUNT(*) FROM u)", "SELECT a FROM t GROUP BY a HAVING COUNT(b) = (SELECT COUNT(*) FROM u)"),
    ("SELECT id FROM t WHERE b NOT IN (SELECT MAX(v) FROM u)", "SELECT id FROM t WHERE b <> (SELECT MAX(v) FROM u)"),
    ("SELECT id FROM t WHERE b = ALL (SELECT COUNT(*) FROM u)", "SELECT id FROM t WHERE b = (SELECT COUNT(*) FROM u)"),
    ("SELECT id FROM t WHERE b > ANY (SELECT SUM(v) FROM u)", "SELECT id FROM t WHERE b > (SELECT SUM(v) FROM u)"),
    ("SELECT id FROM t WHERE b = (SELECT DISTINCT COUNT(v) FROM u)", "SELECT id FROM t WHERE b = (SELECT COUNT(v) FROM u)"),
    ("SELECT x.id FROM t AS x JOIN (SELECT MAX(v) AS top FROM u WHERE v > 0) AS y ON x.b = y.top", "SELECT id FROM t WHERE b = (SELECT MAX(v) FROM u WHERE v > 0)"),
    ("SELECT t.id FROM t, (SELECT COUNT(*) AS n FROM u) AS y", "SELECT id FROM t"),
    ("SELECT t.id, y.total - t.b FROM t CROSS JOIN (SELECT SUM(v) AS total FROM u) AS y", "SELECT id, (SELECT SUM(v) FROM u) - b FROM t"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # a count per group on the right of IN
    ("SELECT a FROM t GROUP BY a HAVING COUNT(b) IN (SELECT COUNT(*) FROM u GROUP BY a)", "SELECT a FROM t GROUP BY a HAVING COUNT(b) = (SELECT COUNT(*) FROM u)"),
    # the outer join keeps every row of t
    ("SELECT x.id FROM t AS x LEFT JOIN (SELECT MAX(v) AS top FROM u) AS y ON x.b = y.top", "SELECT id FROM t WHERE b = (SELECT MAX(v) FROM u)"),
    # a row per group joins more than once
    ("SELECT x.id FROM t AS x JOIN (SELECT MAX(v) AS top FROM u GROUP BY a) AS y ON x.b = y.top", "SELECT id FROM t WHERE b = (SELECT MAX(v) FROM u)"),
    ("SELECT x.id FROM t AS x JOIN (SELECT MAX(v) AS top FROM u) AS y ON x.b = y.top", "SELECT id FROM t WHERE b = (SELECT MIN(v) FROM u)"),
    # no aggregate: a row of t once per matching row of u
    ("SELECT x.id FROM t AS x JOIN (SELECT v AS top FROM u) AS y ON x.b = y.top", "SELECT id FROM t WHERE b IN (SELECT v FROM u)"),
    # HAVING may leave no row at all
    ("SELECT t.id FROM t, (SELECT COUNT(*) AS n FROM u HAVING COUNT(*) > 0) AS y", "SELECT id FROM t"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_the_near_misses_really_differ():
    left, right = NOT_PROVED[1]
    assert run({"t": [(1, 0, 5)]}, left, right) == [["(1,)"], []]
    left, right = NOT_PROVED[2]
    assert run({"t": [(1, 0, 5)], "u": [(1, 1, 5), (2, 2, 5)]}, left, right) == [["(1,)", "(1,)"], ["(1,)"]]
    left, right = NOT_PROVED[5]
    assert run({"t": [(1, 0, 5)]}, left, right) == [[], ["(1,)"]]


QUERIES = [
    "SELECT a FROM t GROUP BY a HAVING COUNT(b) IN (SELECT COUNT(*) FROM u)",
    "SELECT id FROM t WHERE b NOT IN (SELECT MAX(v) + 1 FROM u WHERE u.a > 1)",
    "SELECT id, b IN (SELECT MIN(u.v) FROM u WHERE u.a = t.a) AS f FROM t",
    "SELECT id, NOT b IN (SELECT MAX(u.v) FROM u WHERE u.a = t.a) AS f FROM t",
    "SELECT id FROM t WHERE b = ALL (SELECT COUNT(*) FROM u)",
    "SELECT id, b >= ANY (SELECT SUM(v) FROM u) AS f FROM t",
    "SELECT id FROM t WHERE b = (SELECT DISTINCT COUNT(v) FROM u)",
    "SELECT DISTINCT COUNT(*), MAX(a) FROM t",
    "SELECT x.a FROM (SELECT a, COUNT(b) AS n FROM t GROUP BY a) AS x, (SELECT COUNT(*) AS total FROM u) AS y WHERE x.n = y.total",
    "SELECT x.a, total FROM (SELECT COUNT(*) AS total, MAX(v) AS top FROM u WHERE v > 1) AS y JOIN t AS x ON x.b = y.top WHERE x.a < total + 1",
    "SELECT t.id, y.total - t.b AS rest FROM t CROSS JOIN (SELECT SUM(v) AS total FROM u) AS y JOIN u ON u.id = t.id AND u.v < y.total",
    "SELECT t.id FROM t, (SELECT COUNT(*) AS n FROM u) AS y",
    "SELECT t.a, COUNT(*) FROM t JOIN (SELECT AVG(v) AS m FROM u) AS y ON t.b >= y.m GROUP BY t.a HAVING MAX(t.b) < 2 * MIN(y.m)",
    "SELECT x.id, y.n, z.top FROM t AS x, (SELECT COUNT(*) AS n FROM u) AS y, (SELECT MAX(b) AS top FROM t) AS z WHERE x.b <= z.top",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the rewritten one on random tables with NULLs, repeated rows and empty tables."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 3])  # noqa: E731
    for _ in range(150):
        tables = {
            "t": [(rng.choice([1, 2, 3]), value(), value()) for _ in range(rng.choice([0, 1, 2, 4, 6]))],
            "u": [(rng.choice([1, 2, 3]), value(), value()) for _ in range(rng.choice([0, 0, 1, 2, 3, 5]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
