"""An aggregate over the row's whole partition read as a join with the grouped table (``kumo_sql_prover.own_group_windows``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.own_group_windows import read_own_group_windows
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: (a, b) is the key, so a and b are NOT NULL; c and d may be NULL and repeat.  u: no facts.
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
NOT_NULL = {"t": frozenset({"a", "b"})}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a", "b"}), keys=(("a", "b"),))}
G = "(SELECT a AS kqg0k0, MAX(c) AS kqg0v0 FROM t GROUP BY a) AS kqg0"


def rewritten(sql, not_null=NOT_NULL, schema=SCHEMA):
    before = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    text = before.sql(dialect="mysql")
    after = read_own_group_windows(before, schema, not_null).sql(dialect="mysql")
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


def test_a_window_over_the_whole_partition_becomes_the_join():
    assert rewritten("SELECT b, d, MAX(c) OVER (PARTITION BY a) AS top FROM t") == (
        f"SELECT b, d, kqg0.kqg0v0 AS top FROM t JOIN {G} ON t.a = kqg0.kqg0k0"
    )
    # the same window twice is one output; an alias names the row's table
    assert rewritten("SELECT x.b, CASE WHEN MAX(x.c) OVER (PARTITION BY x.a) > 5 THEN d ELSE MAX(c) OVER (PARTITION BY a) END AS d FROM t AS x ORDER BY b LIMIT 3") == (
        f"SELECT x.b, CASE WHEN kqg0.kqg0v0 > 5 THEN d ELSE kqg0.kqg0v0 END AS d FROM t AS x JOIN {G} ON x.a = kqg0.kqg0k0 ORDER BY b LIMIT 3"
    )
    # several aggregates over one partition, two partition columns in either order, COUNT(*)
    assert rewritten("SELECT DISTINCT MIN(c) OVER (PARTITION BY b, a), SUM(d) OVER (PARTITION BY a, b), COUNT(*) OVER (PARTITION BY a, b), AVG(c) OVER (PARTITION BY a, b), COUNT(d) OVER (PARTITION BY a, b) FROM t") == (
        "SELECT DISTINCT kqg0.kqg0v0, kqg0.kqg0v1, kqg0.kqg0v2, kqg0.kqg0v3, kqg0.kqg0v4 FROM t JOIN (SELECT a AS kqg0k0, b AS kqg0k1, MIN(c) AS kqg0v0, SUM(d) AS kqg0v1, COUNT(*) AS kqg0v2, "
        "AVG(c) AS kqg0v3, COUNT(d) AS kqg0v4 FROM t GROUP BY a, b) AS kqg0 ON t.a = kqg0.kqg0k0 AND t.b = kqg0.kqg0k1"
    )


def test_a_star_names_the_table():
    assert rewritten("SELECT *, MAX(c) OVER (PARTITION BY a) AS top FROM t") == f"SELECT t.*, kqg0.kqg0v0 AS top FROM t JOIN {G} ON t.a = kqg0.kqg0k0"
    assert rewritten("SELECT x.*, MAX(c) OVER (PARTITION BY a) AS top FROM t AS x") == f"SELECT x.*, kqg0.kqg0v0 AS top FROM t AS x JOIN {G} ON x.a = kqg0.kqg0k0"


def test_a_window_inside_a_derived_table():
    assert rewritten("SELECT m.b FROM (SELECT b, MAX(c) OVER (PARTITION BY a) AS top FROM t) AS m WHERE m.top > 1") == (
        f"SELECT m.b FROM (SELECT b, kqg0.kqg0v0 AS top FROM t JOIN {G} ON t.a = kqg0.kqg0k0) AS m WHERE m.top > 1"
    )


@pytest.mark.parametrize("sql", [
    "SELECT b, MAX(c) OVER (PARTITION BY d) FROM t",  # d may be NULL: its NULL partition matches no group
    "SELECT b, MAX(c) OVER (PARTITION BY a, d) FROM t",
    "SELECT b, MAX(c) OVER () FROM t",
    "SELECT b, MAX(c) OVER (PARTITION BY a ORDER BY b) FROM t",  # a running maximum
    "SELECT b, SUM(c) OVER (PARTITION BY a ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
    "SELECT b, RANK() OVER (PARTITION BY a ORDER BY c) FROM t",
    "SELECT b, ROW_NUMBER() OVER (PARTITION BY a) FROM t",
    "SELECT b, FIRST_VALUE(c) OVER (PARTITION BY a) FROM t",
    "SELECT b, MAX(c + 1) OVER (PARTITION BY a) FROM t",
    "SELECT b, COUNT(DISTINCT c) OVER (PARTITION BY a) FROM t",
    "SELECT b, MAX(c) OVER (PARTITION BY a + 1) FROM t",
    "SELECT b, MAX(c) OVER (PARTITION BY a), MIN(c) OVER (PARTITION BY b) FROM t",  # two partitions
    "SELECT b, MAX(c) OVER (PARTITION BY a), RANK() OVER (PARTITION BY a ORDER BY c) FROM t",  # one window that is not read
    "SELECT b, MAX(c) OVER (PARTITION BY a) FROM t WHERE d > 0",  # the partitions are of the filtered rows
    "SELECT a, MAX(SUM(c)) OVER (PARTITION BY a) FROM t GROUP BY a, b",
    "SELECT a, MAX(c) OVER (PARTITION BY a) FROM t GROUP BY a, c",  # the partitions are of the groups
    "SELECT MAX(c) OVER (PARTITION BY a), COUNT(*) FROM t",
    "SELECT t.b, MAX(t.c) OVER (PARTITION BY t.a) FROM t JOIN u ON u.a = t.a",  # the partitions are of the joined rows
    "SELECT b, MAX(c) OVER (PARTITION BY a) FROM (SELECT * FROM t WHERE d > 0) AS t",
    "SELECT b FROM t ORDER BY MAX(c) OVER (PARTITION BY a)",
    "SELECT b, MAX(c) OVER (PARTITION BY a) AS top FROM t QUALIFY top > 1",
    "SELECT b, MAX(e) OVER (PARTITION BY a) FROM t",  # e is not t's
    "SELECT b, MAX(u.c) OVER (PARTITION BY a) FROM t",
    "SELECT b, MAX(c) OVER w FROM t WINDOW w AS (PARTITION BY a)",  # named windows are inlined before this pass
    "SELECT b, MAX(c) FROM t GROUP BY b",
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_the_partition_columns_must_be_declared_not_null():
    sql = "SELECT b, MAX(c) OVER (PARTITION BY a) FROM t"
    assert rewritten(sql) is not None
    assert rewritten(sql, not_null=None) is None
    assert rewritten(sql, not_null={"t": frozenset({"b"})}) is None
    assert rewritten(sql, schema=None) is None


JOINED = "SELECT s.b, s.d, g.top FROM t AS s JOIN (SELECT a, MAX(c) AS top FROM t GROUP BY a) AS g ON s.a = g.a"
PROOFS = [
    ("SELECT b, d, MAX(c) OVER (PARTITION BY a) FROM t", JOINED),
    ("SELECT m.b, m.d, m.top FROM (SELECT *, MAX(c) OVER (PARTITION BY a) AS top FROM t) AS m", JOINED),
    ("WITH m AS (SELECT b, d, MAX(c) OVER (PARTITION BY a) AS top FROM t) SELECT b, d, top FROM m", JOINED),
    # what the grouped table says about a row of its group: c is at most the maximum, and a NULL maximum means a NULL c
    (
        "SELECT b, CASE WHEN MAX(c) OVER (PARTITION BY a) < 10 THEN c ELSE 2 * c END FROM t",
        "SELECT s.b, CASE WHEN g.top >= 10 THEN 2 * s.c ELSE s.c END FROM t AS s JOIN (SELECT a, MAX(c) AS top FROM t GROUP BY a) AS g ON s.a = g.a",
    ),
    ("SELECT b, COUNT(*) OVER (PARTITION BY a), SUM(d) OVER (PARTITION BY a) FROM t", "SELECT s.b, g.n, g.total FROM t AS s JOIN (SELECT a, COUNT(*) AS n, SUM(d) AS total FROM t GROUP BY a) AS g ON s.a = g.a"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # a NULL partition key: the window keeps the row, the join drops it
    ("SELECT b, MAX(c) OVER (PARTITION BY d) FROM t", "SELECT s.b, g.top FROM t AS s JOIN (SELECT d, MAX(c) AS top FROM t GROUP BY d) AS g ON s.d = g.d"),
    # a running maximum is not the group's
    ("SELECT b, MAX(c) OVER (PARTITION BY a ORDER BY b) FROM t", "SELECT s.b, g.top FROM t AS s JOIN (SELECT a, MAX(c) AS top FROM t GROUP BY a) AS g ON s.a = g.a"),
    # MIN against MAX
    ("SELECT b, MIN(c) OVER (PARTITION BY a) FROM t", "SELECT s.b, g.top FROM t AS s JOIN (SELECT a, MAX(c) AS top FROM t GROUP BY a) AS g ON s.a = g.a"),
    # the filter is applied before the window, and after the grouped table
    ("SELECT b, MAX(c) OVER (PARTITION BY a) FROM t WHERE d > 0", "SELECT s.b, g.top FROM t AS s JOIN (SELECT a, MAX(c) AS top FROM t GROUP BY a) AS g ON s.a = g.a WHERE s.d > 0"),
    # with a NULL maximum the arms differ unless both are the row's own (NULL) value
    (
        "SELECT b, CASE WHEN MAX(c) OVER (PARTITION BY a) < 10 THEN 1 ELSE 2 END FROM t",
        "SELECT s.b, CASE WHEN g.top >= 10 THEN 2 ELSE 1 END FROM t AS s JOIN (SELECT a, MAX(c) AS top FROM t GROUP BY a) AS g ON s.a = g.a",
    ),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_the_nullable_partition_really_differs():
    left, right = NOT_PROVED[0]
    a, b = run({"t": [(1, 1, 5, None)]}, left, right)
    assert a == ["(1, 5)"] and b == []


QUERIES = [
    "SELECT b, d, MAX(c) OVER (PARTITION BY a) AS top FROM t",
    "SELECT x.b, CASE WHEN MAX(x.c) OVER (PARTITION BY x.a) > 1 THEN d ELSE MAX(c) OVER (PARTITION BY a) END AS d FROM t AS x",
    "SELECT DISTINCT MIN(c) OVER (PARTITION BY b, a), SUM(d) OVER (PARTITION BY a, b), COUNT(*) OVER (PARTITION BY a, b), AVG(c) OVER (PARTITION BY a, b), COUNT(d) OVER (PARTITION BY a, b) FROM t",
    "SELECT *, MAX(c) OVER (PARTITION BY a) AS top, MIN(d) OVER (PARTITION BY a) AS low, COUNT(*) OVER (PARTITION BY a) AS n FROM t",
    "SELECT m.b FROM (SELECT b, SUM(c) OVER (PARTITION BY a) AS total, AVG(d) OVER (PARTITION BY a) AS mean FROM t) AS m WHERE m.total > 1 OR m.mean IS NULL",
    "SELECT b, COUNT(c) OVER (PARTITION BY b) FROM t",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the rewritten one on random tables that meet the declared facts: ``(a, b)`` is a key of
    ``t``; ``c`` and ``d`` may be NULL and repeat."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 2, 3])  # noqa: E731
    for _ in range(150):
        keys = rng.sample([(i, j) for i in range(3) for j in range(4)], rng.choice([0, 1, 2, 4, 7]))
        tables = {"t": [(a, b, value(), value()) for a, b in keys]}
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
