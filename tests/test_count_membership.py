"""A correlated COUNT compared with a number, read as membership in the groups that pass (``kumo_sql_prover.count_membership``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.count_membership import read_count_membership

# every column may be NULL and repeat
SCHEMA = {"t": ["id", "a", "b"], "u": ["id", "a", "v"]}


def rewritten(sql, schema=SCHEMA):
    before = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    text = before.sql(dialect="mysql")
    after = read_count_membership(before, schema).sql(dialect="mysql")
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


def test_a_count_above_a_number_becomes_membership():
    assert rewritten("SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a) > 1") == (
        "SELECT t.id FROM t WHERE t.a IN (SELECT u.a FROM u GROUP BY u.a HAVING COUNT(*) > 1)"
    )
    # the number first, the correlation the other way round, another part in the WHERE
    assert rewritten("SELECT t.id FROM t WHERE t.b > 0 AND 1 < (SELECT COUNT(*) FROM u AS q WHERE t.a = q.a)") == (
        "SELECT t.id FROM t WHERE t.b > 0 AND t.a IN (SELECT q.a FROM u AS q GROUP BY q.a HAVING COUNT(*) > 1)"
    )
    assert rewritten("SELECT t.id FROM t WHERE (SELECT COUNT(v) FROM u WHERE a = t.a) <> 0") == (
        "SELECT t.id FROM t WHERE t.a IN (SELECT u.a FROM u GROUP BY u.a HAVING COUNT(v) <> 0)"
    )
    # a bare outer column: u has no column b
    assert rewritten("SELECT t.id FROM t WHERE (SELECT COUNT(1) FROM u WHERE v = b) = 3") == (
        "SELECT t.id FROM t WHERE b IN (SELECT u.v FROM u GROUP BY u.v HAVING COUNT(1) = 3)"
    )
    # the same table inside and outside: a comparison of two inner columns is a filter
    assert rewritten("SELECT x.id FROM t AS x WHERE (SELECT COUNT(*) FROM t WHERE t.a = x.a AND t.b = t.id) >= 1") == (
        "SELECT x.id FROM t AS x WHERE x.a IN (SELECT t.a FROM t WHERE t.b = t.id GROUP BY t.a HAVING COUNT(*) >= 1)"
    )


def test_filters_stay_and_two_correlations_make_a_row():
    assert rewritten(
        "SELECT t.id FROM t WHERE (SELECT COUNT(DISTINCT q.v) AS n FROM u AS q WHERE q.a = t.a AND (q.v > 0 OR q.id = 1) AND q.id = t.b) >= 2"
    ) == "SELECT t.id FROM t WHERE (t.a, t.b) IN (SELECT q.a, q.id FROM u AS q WHERE (q.v > 0 OR q.id = 1) GROUP BY q.a, q.id HAVING COUNT(DISTINCT q.v) >= 2)"


def test_a_subquery_grouped_by_its_correlation_is_read_for_every_comparison():
    assert rewritten("SELECT t.id FROM t WHERE (SELECT COUNT(v) FROM u WHERE a = t.a GROUP BY a) < 2") == (
        "SELECT t.id FROM t WHERE t.a IN (SELECT u.a FROM u GROUP BY u.a HAVING COUNT(v) < 2)"
    )
    assert rewritten("SELECT t.id FROM t WHERE (SELECT COUNT(v) FROM u WHERE u.a = t.a GROUP BY u.a) = 0") == (
        "SELECT t.id FROM t WHERE t.a IN (SELECT u.a FROM u GROUP BY u.a HAVING COUNT(v) = 0)"
    )


@pytest.mark.parametrize("sql", [
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a) < 2",  # a count of 0 passes
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a) = 0",
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a) >= 0",
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a) <= 3",
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a) <> 1",
    "SELECT t.id FROM t WHERE 0 = (SELECT COUNT(*) FROM u WHERE u.a = t.a)",
    "SELECT t.id FROM t WHERE 2 > (SELECT COUNT(*) FROM u WHERE u.a = t.a)",
    "SELECT t.id FROM t WHERE NOT (SELECT COUNT(*) FROM u WHERE u.a = t.a) > 1",  # FALSE and NULL differ under NOT
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a) > 1 OR t.b = 1",
    "SELECT t.id, (SELECT COUNT(*) FROM u WHERE u.a = t.a) > 1 FROM t",
    "SELECT t.a FROM t GROUP BY t.a HAVING (SELECT COUNT(*) FROM u WHERE u.a = t.a) > 1",
    "SELECT t.id FROM t JOIN u AS s ON (SELECT COUNT(*) FROM u WHERE u.a = t.a) > 1",
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a) > t.b",  # not a number
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a) > -1",
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a) > 1.5",
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a > t.a) > 1",  # not an equality
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a OR u.v = 1) > 1",
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a + 1) > 1",
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a AND u.v > t.b) > 1",  # an outer column in a filter
    "SELECT t.id FROM t WHERE (SELECT COUNT(t.b) FROM u WHERE u.a = t.a) > 1",  # an outer column in the COUNT
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.v = 1) > 1",  # not correlated
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u) > 1",
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) + 1 FROM u WHERE u.a = t.a) > 1",  # an expression around the COUNT
    "SELECT t.id FROM t WHERE (SELECT SUM(u.v) FROM u WHERE u.a = t.a) > 1",
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a HAVING COUNT(*) > 0) > 1",
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a LIMIT 1) > 1",
    "SELECT t.id FROM t WHERE (SELECT DISTINCT COUNT(*) FROM u WHERE u.a = t.a) > 1",
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u JOIN t AS s ON s.id = u.id WHERE u.a = t.a) > 1",  # a join
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a GROUP BY u.v) > 1",  # grouped by another column
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a AND u.v = t.b GROUP BY u.a) < 2",  # by fewer columns
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a AND EXISTS (SELECT 1 FROM t AS s WHERE s.id = u.id)) > 1",
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM w WHERE w.a = t.a) > 1",  # the columns of w are not supplied
])
def test_other_comparisons_stay(sql):
    assert rewritten(sql) is None


def test_nothing_is_read_without_the_schema():
    assert rewritten("SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a) > 1", schema=None) is None


PROOFS = [
    ("SELECT DISTINCT t.a FROM t WHERE (SELECT COUNT(*) FROM t AS q WHERE q.a = t.a) > 1", "SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1"),
    ("SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a) > 1", "SELECT t.id FROM t WHERE t.a IN (SELECT a FROM u GROUP BY a HAVING COUNT(*) >= 2)"),
    ("SELECT t.id FROM t WHERE (SELECT COUNT(v) FROM u WHERE a = t.a GROUP BY a) < 2", "SELECT t.id FROM t WHERE t.a IN (SELECT a FROM u GROUP BY a HAVING COUNT(v) < 2)"),
    (
        "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a AND u.v = t.b) > 1",
        "SELECT t.id FROM t WHERE (t.a, t.b) IN (SELECT a, v FROM u GROUP BY a, v HAVING COUNT(*) > 1)",
    ),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # a row of t with no match in u has a count of 0, below 2, and is in no group
    ("SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a) < 2", "SELECT t.id FROM t WHERE t.a IN (SELECT a FROM u GROUP BY a HAVING COUNT(*) < 2)"),
    ("SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a) = 0", "SELECT t.id FROM t WHERE t.a IN (SELECT a FROM u GROUP BY a HAVING COUNT(*) = 0)"),
    # COUNT(*) counts the rows whose v is NULL
    ("SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a) > 1", "SELECT t.id FROM t WHERE t.a IN (SELECT a FROM u GROUP BY a HAVING COUNT(v) > 1)"),
    ("SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a) > 1", "SELECT t.id FROM t WHERE t.a IN (SELECT a FROM u GROUP BY a HAVING COUNT(*) > 2)"),
    # under NOT a row with a NULL a is kept by the count and dropped by the membership
    ("SELECT t.id FROM t WHERE NOT (SELECT COUNT(*) FROM u WHERE u.a = t.a) > 1", "SELECT t.id FROM t WHERE NOT t.a IN (SELECT a FROM u GROUP BY a HAVING COUNT(*) > 1)"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_the_near_misses_really_differ():
    left, right = NOT_PROVED[0]
    assert run({"t": [(1, 5, 0)]}, left, right) == [["(1,)"], []]
    left, right = NOT_PROVED[2]
    assert run({"t": [(1, 5, 0)], "u": [(1, 5, None), (2, 5, 1)]}, left, right) == [["(1,)"], []]
    left, right = NOT_PROVED[4]
    assert run({"t": [(1, None, 0)], "u": [(1, 5, 1), (2, 5, 1)]}, left, right) == [["(1,)"], []]


QUERIES = [
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a) > 1",
    "SELECT t.id FROM t WHERE t.b > 0 AND 1 <= (SELECT COUNT(*) FROM u AS q WHERE t.a = q.a)",
    "SELECT t.id, t.a FROM t WHERE (SELECT COUNT(v) FROM u WHERE a = t.a) <> 0",
    "SELECT t.id FROM t WHERE (SELECT COUNT(1) FROM u WHERE v = b) = 2",
    "SELECT x.id FROM t AS x WHERE (SELECT COUNT(*) FROM t WHERE t.a = x.a AND t.b = t.id) >= 1",
    "SELECT DISTINCT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM t WHERE t.a = x.a) > 1",
    "SELECT t.id FROM t WHERE (SELECT COUNT(DISTINCT q.v) AS n FROM u AS q WHERE q.a = t.a AND (q.v > 0 OR q.id = 1) AND q.id = t.b) >= 2",
    "SELECT t.id FROM t WHERE (SELECT COUNT(v) FROM u WHERE a = t.a GROUP BY a) < 2",
    "SELECT t.id FROM t WHERE (SELECT COUNT(v) FROM u WHERE u.a = t.a GROUP BY u.a) = 0",
    "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a = t.a AND u.v = t.b GROUP BY u.a, u.v) <> 1",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the rewritten one on random tables with NULLs and repeated rows."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 1, 1, 2, 2, 3])  # noqa: E731
    for _ in range(150):
        tables = {
            "t": [(rng.choice([1, 2, 3]), value(), value()) for _ in range(rng.choice([0, 1, 2, 4, 6]))],
            "u": [(rng.choice([1, 2, 3]), value(), value()) for _ in range(rng.choice([0, 1, 2, 4, 7]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
