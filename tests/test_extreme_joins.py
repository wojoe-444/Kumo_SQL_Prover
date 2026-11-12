"""A join with the table's own grouped MIN or MAX read as a comparison (``kumo_sql_prover.extreme_joins``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.extreme_joins import read_extreme_joins
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: a and b are NOT NULL; c and d may be NULL
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
NOT_NULL = {"t": frozenset({"a", "b"})}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a", "b"}))}
G = "(SELECT a, MIN(b) AS m FROM t GROUP BY a) AS g"
GC = "(SELECT a, MIN(c) AS m FROM t GROUP BY a) AS g"


def rewritten(sql, not_null=NOT_NULL, schema=SCHEMA):
    before = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    text = before.sql(dialect="mysql")
    after = read_extreme_joins(before, schema, not_null).sql(dialect="mysql")
    return after if after != text else None


def proved(left, right, constraints=CONSTRAINTS):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=constraints, compare_names=False, dialect="mysql").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE t (a BIGINT, b BIGINT, c BIGINT, d BIGINT)")
    db.execute("CREATE TABLE u (a BIGINT, e BIGINT)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


FIRST = "SELECT x.a, x.d FROM t AS x WHERE x.b = (SELECT MIN(kqj0.b) FROM t AS kqj0 WHERE kqj0.a = x.a)"


def test_an_inner_join_becomes_the_comparison():
    assert rewritten(f"SELECT x.a, x.d FROM t AS x JOIN {G} ON x.a = g.a AND x.b = g.m") == FIRST
    assert rewritten(f"SELECT x.a, x.d FROM {G} JOIN t AS x ON g.a = x.a WHERE g.m = x.b") == FIRST
    assert rewritten(f"SELECT x.a, x.d FROM t AS x, {G} WHERE x.a = g.a AND x.b = m") == FIRST
    assert rewritten("SELECT x.a, x.d FROM t AS x JOIN (SELECT DISTINCT a, MIN(b) AS m FROM t GROUP BY a ORDER BY m) AS g ON x.a = g.a AND x.b = g.m") == FIRST
    # MAX, a filter inside the grouped table, two keys in another order
    assert rewritten("SELECT x.d FROM t AS x JOIN (SELECT MAX(c) AS top, d AS dd, a FROM t AS y WHERE y.b > 0 OR c = 1 GROUP BY y.a, d) AS g ON g.dd = x.d AND x.a = g.a AND g.top = x.c") == (
        "SELECT x.d FROM t AS x WHERE x.c = (SELECT MAX(kqj0.c) FROM t AS kqj0 WHERE (kqj0.b > 0 OR kqj0.c = 1) AND kqj0.d = x.d AND kqj0.a = x.a)"
    )


def test_other_reads_of_the_grouped_table_name_the_row():
    assert rewritten(f"SELECT g.a, g.m, x.d FROM t AS x JOIN {G} ON x.a = g.a AND x.b = g.m WHERE g.m > 0 AND x.c = g.a ORDER BY g.a") == (
        "SELECT x.a, x.b AS m, x.d FROM t AS x WHERE x.b = (SELECT MIN(kqj0.b) FROM t AS kqj0 WHERE kqj0.a = x.a) AND x.b > 0 AND x.c = x.a ORDER BY x.a"
    )
    # other sources stay, and their ON parts join the WHERE
    assert rewritten(f"SELECT u.e, x.d FROM u JOIN t AS x ON u.a = x.a JOIN {G} ON x.a = g.a AND x.b = g.m WHERE u.e > 1") == (
        "SELECT u.e, x.d FROM u, t AS x WHERE u.a = x.a AND x.b = (SELECT MIN(kqj0.b) FROM t AS kqj0 WHERE kqj0.a = x.a) AND u.e > 1"
    )


def test_an_outer_join_that_cannot_fail():
    assert rewritten(f"SELECT x.a, x.d FROM {G} LEFT JOIN t AS x ON x.a = g.a AND x.b = g.m") == FIRST
    assert rewritten(f"SELECT x.a, x.d FROM t AS x RIGHT JOIN {G} ON x.a = g.a AND x.b = g.m") == FIRST
    assert rewritten(f"SELECT g.a, x.d FROM {G} LEFT JOIN t AS x ON x.a = g.a AND x.b = g.m WHERE x.d IS NULL") == (
        "SELECT x.a, x.d FROM t AS x WHERE x.b = (SELECT MIN(kqj0.b) FROM t AS kqj0 WHERE kqj0.a = x.a) AND x.d IS NULL"
    )
    # the table's side is kept, and the comparison in the WHERE fails for a row the grouped table did not match
    assert rewritten(f"SELECT x.a, x.d FROM t AS x LEFT JOIN {G} ON x.a = g.a WHERE x.b = g.m") == FIRST
    assert rewritten(f"SELECT x.a, x.d FROM t AS x LEFT JOIN {GC} ON x.a = g.a WHERE x.c = g.m") == FIRST.replace(".b", ".c")


@pytest.mark.parametrize("sql", [
    f"SELECT x.a, x.d FROM {GC} LEFT JOIN t AS x ON x.a = g.a AND x.c = g.m",  # a group whose c is all NULL has no match
    "SELECT x.a, x.d FROM (SELECT d, MIN(b) AS m FROM t GROUP BY d) AS g LEFT JOIN t AS x ON x.d = g.d AND x.b = g.m",  # nor has the NULL group
    f"SELECT x.a, x.d FROM {G} LEFT JOIN t AS x ON x.a = g.a AND x.b = g.m AND x.d > 0",  # the match may fail the extra test
    f"SELECT x.a, x.d FROM {G} LEFT JOIN t AS x ON x.a = g.a WHERE x.b = g.m OR x.b IS NULL",
    f"SELECT x.a, x.d FROM t AS x LEFT JOIN {G} ON x.a = g.a AND x.b = g.m",  # every row of t is kept
    f"SELECT x.a, x.d FROM t AS x FULL JOIN {G} ON x.a = g.a AND x.b = g.m",
    f"SELECT x.a, x.d FROM t AS x JOIN {G} ON x.a = g.a",  # no comparison with the minimum
    f"SELECT x.a, x.d FROM t AS x JOIN {G} ON x.b = g.m",  # no key
    f"SELECT x.a, x.d FROM t AS x JOIN {G} ON x.a = g.a AND x.b >= g.m",
    f"SELECT x.a, x.d FROM t AS x JOIN {G} ON x.a = g.a AND x.c = g.m",  # another column than the one minimized
    f"SELECT x.a, x.d FROM t AS x JOIN {G} ON x.d = g.a AND x.b = g.m",  # another column than the key
    f"SELECT x.a, x.d FROM t AS x JOIN {G} ON x.a = g.a OR x.b = g.m",
    f"SELECT x.e FROM u AS x JOIN {G} ON x.a = g.a AND x.e = g.m",  # not the same table
    "SELECT x.a FROM t AS x JOIN (SELECT a, MIN(b) AS m FROM t GROUP BY a, d) AS g ON x.a = g.a AND x.b = g.m",  # a key that is not output
    "SELECT x.a FROM t AS x JOIN (SELECT a, MIN(b) AS m FROM t GROUP BY a HAVING COUNT(*) > 1) AS g ON x.a = g.a AND x.b = g.m",
    "SELECT x.a FROM t AS x JOIN (SELECT a, MIN(b) AS m FROM t GROUP BY a LIMIT 2) AS g ON x.a = g.a AND x.b = g.m",
    "SELECT x.a FROM t AS x JOIN (SELECT a, MIN(b) AS m, COUNT(*) AS n FROM t GROUP BY a) AS g ON x.a = g.a AND x.b = g.m",  # a second aggregate
    "SELECT x.a FROM t AS x JOIN (SELECT a, d, MIN(b) AS m FROM t GROUP BY a) AS g ON x.a = g.a AND x.b = g.m",  # a column that is no key
    "SELECT x.a FROM t AS x JOIN (SELECT a, SUM(b) AS m FROM t GROUP BY a) AS g ON x.a = g.a AND x.b = g.m",
    "SELECT x.a FROM t AS x JOIN (SELECT a + 1 AS a, MIN(b) AS m FROM t GROUP BY a + 1) AS g ON x.a = g.a AND x.b = g.m",
    f"SELECT * FROM t AS x JOIN {G} ON x.a = g.a AND x.b = g.m",
    f"SELECT g.* FROM t AS x JOIN {G} ON x.a = g.a AND x.b = g.m",
    f"SELECT x.a FROM t AS x JOIN {G} USING (a) WHERE x.b = g.m",
    f"SELECT a FROM t AS x JOIN {G} ON x.a = g.a AND x.b = g.m",  # a is x's or g's
    f"SELECT x.a FROM t AS x JOIN {G} ON x.a = g.a AND x.b = g.m WHERE EXISTS (SELECT 1 FROM u WHERE u.e = g.m)",  # a nested query reads g
    f"SELECT x.a FROM t AS x JOIN {G} ON x.a = g.a AND x.b = g.m JOIN (SELECT a FROM u) AS s ON s.a = x.a WHERE m > 0",  # m may be s's
])
def test_other_joins_stay(sql):
    assert rewritten(sql) is None


def test_the_outer_join_needs_the_declared_facts():
    sql = f"SELECT x.a, x.d FROM {G} LEFT JOIN t AS x ON x.a = g.a AND x.b = g.m"
    assert rewritten(sql, not_null=None) is None
    assert rewritten(sql, not_null={"t": frozenset({"a"})}) is None
    assert rewritten(sql, not_null={"t": frozenset({"b"})}) is None
    assert rewritten(f"SELECT x.a, x.d FROM t AS x JOIN {G} ON x.a = g.a AND x.b = g.m", not_null=None) == FIRST  # the inner join needs none


PROOFS = [
    (f"SELECT x.a, x.d FROM {G} LEFT JOIN t AS x ON x.a = g.a AND x.b = g.m", f"SELECT x.a, x.d FROM t AS x JOIN {G} ON x.a = g.a AND x.b = g.m"),
    (f"SELECT g.a, x.d FROM t AS x RIGHT JOIN {G} ON x.a = g.a AND x.b = g.m", "SELECT a, d FROM t WHERE (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a)"),
    (f"SELECT x.a, x.d FROM t AS x LEFT JOIN {GC} ON x.a = g.a WHERE x.c = g.m", "SELECT s.a, s.d FROM t AS s WHERE s.c = (SELECT MIN(y.c) FROM t AS y WHERE y.a = s.a)"),
    (f"SELECT g.a, g.m, x.d FROM t AS x JOIN {G} ON x.a = g.a AND x.b = g.m", "SELECT s.a, s.b, s.d FROM t AS s WHERE s.b = (SELECT MIN(y.b) FROM t AS y WHERE y.a = s.a)"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # c may be NULL in a whole group: the outer join keeps that group with a NULL row, the inner join drops it
    (f"SELECT g.a, x.d FROM {GC} LEFT JOIN t AS x ON x.a = g.a AND x.c = g.m", f"SELECT g.a, x.d FROM t AS x JOIN {GC} ON x.a = g.a AND x.c = g.m"),
    # every row of t against the first rows
    (f"SELECT x.a, x.d FROM t AS x LEFT JOIN {G} ON x.a = g.a AND x.b = g.m", f"SELECT x.a, x.d FROM t AS x JOIN {G} ON x.a = g.a AND x.b = g.m"),
    # MIN against MAX
    (f"SELECT x.a, x.d FROM t AS x JOIN {G} ON x.a = g.a AND x.b = g.m", "SELECT x.a, x.d FROM t AS x JOIN (SELECT a, MAX(b) AS m FROM t GROUP BY a) AS g ON x.a = g.a AND x.b = g.m"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_the_nullable_outer_join_really_differs():
    left, right = NOT_PROVED[0]
    a, b = run({"t": [(1, 1, None, 5)]}, left, right)
    assert a == ["(1, None)"] and b == []


QUERIES = [
    f"SELECT x.a, x.d FROM t AS x JOIN {G} ON x.a = g.a AND x.b = g.m",
    f"SELECT g.a, g.m, x.d FROM t AS x JOIN {G} ON x.a = g.a AND x.b = g.m WHERE g.m > 0 AND x.c = g.a",
    f"SELECT u.e, x.d FROM u JOIN t AS x ON u.a = x.a JOIN {G} ON x.a = g.a AND x.b = g.m WHERE u.e > 1",
    "SELECT x.d FROM t AS x JOIN (SELECT MAX(c) AS top, d AS dd, a FROM t AS y WHERE y.b > 0 OR c = 1 GROUP BY y.a, d) AS g ON g.dd = x.d AND x.a = g.a AND g.top = x.c",
    f"SELECT x.a, x.d FROM {G} LEFT JOIN t AS x ON x.a = g.a AND x.b = g.m",
    f"SELECT g.a, x.d FROM t AS x RIGHT JOIN {G} ON x.a = g.a AND x.b = g.m WHERE x.d IS NULL",
    "SELECT g.a, x.c FROM (SELECT a, MAX(b) AS m FROM t WHERE c > 0 GROUP BY a) AS g LEFT JOIN t AS x ON x.a = g.a AND x.b = g.m",
    f"SELECT x.a, x.d FROM t AS x LEFT JOIN {GC} ON x.a = g.a WHERE x.c = g.m",
    f"SELECT x.a, x.d FROM t AS x LEFT JOIN {G} ON x.a = g.a AND x.d > 0 WHERE x.b = g.m",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the rewritten one on random tables that meet the declared facts (a and b NOT NULL)."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 2, 3])  # noqa: E731
    for _ in range(150):
        tables = {
            "t": [(rng.choice([1, 1, 2, 3]), rng.choice([0, 1, 1, 2, 3]), value(), value()) for _ in range(rng.choice([0, 1, 2, 4, 7]))],
            "u": [(value(), value()) for _ in range(rng.choice([0, 1, 3]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
