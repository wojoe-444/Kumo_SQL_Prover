"""A sum of ones over the rows that pass a condition is a count (``kumo_sql_prover.conditional_count_sums``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.conditional_count_sums import count_conditional_sums

SCHEMA = {"t": ["a", "b", "c"], "u": ["a", "b", "c"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}
ONES = "SUM(CASE WHEN b > 1 THEN 1 ELSE 0 END)"
COUNT = "COUNT(CASE WHEN b > 1 THEN 1 END)"


def counted(sql):
    out = count_conditional_sums(sqlglot.parse_one(sql, read="postgres"))
    return out.sql(dialect="postgres") if out is not None else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


def rows_on_duckdb(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name in SCHEMA:
        db.execute(f"CREATE TABLE {name} (a BIGINT, b BIGINT, c BIGINT)")
        insert_rows(db, name, rows.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def differ_on_duckdb(left, right, rows):
    a, b = rows_on_duckdb([left, right], rows)
    return a != b


def test_sums_of_ones_become_counts():
    assert counted(f"SELECT a, {ONES} AS n FROM t GROUP BY a") == f"SELECT a, {COUNT} AS n FROM t GROUP BY a"
    assert counted(f"SELECT a FROM t GROUP BY a HAVING {ONES} > 1 ORDER BY {ONES}") == f"SELECT a FROM t GROUP BY a HAVING {COUNT} > 1 ORDER BY {COUNT}"
    assert counted(f"SELECT COALESCE({ONES}, 0) AS n FROM t") == f"SELECT {COUNT} AS n FROM t"
    assert counted("SELECT COALESCE(SUM(CASE WHEN b > 1 THEN 1 END), 0) AS n FROM t") == f"SELECT {COUNT} AS n FROM t"
    assert counted("SELECT a, COALESCE(SUM(CASE WHEN b > 1 THEN 1 ELSE NULL END), 0) AS n FROM t GROUP BY a") == f"SELECT a, {COUNT} AS n FROM t GROUP BY a"
    assert counted("SELECT a, SUM(1) AS n, COALESCE(SUM(1), 0) AS m FROM t GROUP BY a") == "SELECT a, COUNT(*) AS n, COUNT(*) AS m FROM t GROUP BY a"
    assert counted("SELECT COALESCE(SUM(1), 0) AS n FROM t WHERE b > 1") == "SELECT COUNT(*) AS n FROM t WHERE b > 1"
    assert counted(f"SELECT a, COALESCE({ONES}, 0) + {ONES} AS n FROM t GROUP BY a") == f"SELECT a, {COUNT} + {COUNT} AS n FROM t GROUP BY a"


@pytest.mark.parametrize("sql", [
    f"SELECT {ONES} AS n FROM t",  # NULL over no rows
    "SELECT SUM(1) AS n FROM t",
    "SELECT a, SUM(2) AS n FROM t GROUP BY a",
    "SELECT a, SUM(1.0) AS n FROM t GROUP BY a",
    f"SELECT a, {ONES} AS n FROM t GROUP BY ROLLUP (a)",  # the grand-total group exists over no rows
    f"SELECT a, {ONES} AS n FROM t GROUP BY GROUPING SETS ((a), ())",
    "SELECT a, SUM(CASE WHEN b > 1 THEN 1 END) AS n FROM t GROUP BY a",  # NULL when no row passes
    "SELECT a, SUM(CASE WHEN b > 1 THEN 1 ELSE NULL END) AS n FROM t GROUP BY a",
    "SELECT a, SUM(CASE WHEN b > 1 THEN 2 ELSE 0 END) AS n FROM t GROUP BY a",
    "SELECT a, SUM(CASE WHEN b > 1 THEN 1 ELSE 1 END) AS n FROM t GROUP BY a",
    "SELECT a, SUM(CASE WHEN b > 1 THEN 1 WHEN b < 0 THEN 1 ELSE 0 END) AS n FROM t GROUP BY a",
    "SELECT a, SUM(CASE b WHEN 1 THEN 1 ELSE 0 END) AS n FROM t GROUP BY a",
    "SELECT a, SUM(DISTINCT CASE WHEN b > 1 THEN 1 ELSE 0 END) AS n FROM t GROUP BY a",
    "SELECT a, SUM(CASE WHEN b > 1 THEN 1 ELSE 0 END) FILTER (WHERE c > 0) AS n FROM t GROUP BY a",
    "SELECT a, SUM(CASE WHEN b > 1 THEN 1 ELSE 0 END) OVER (PARTITION BY a) AS n FROM t",
    "SELECT a, COALESCE(SUM(CASE WHEN b > 1 THEN 1 ELSE 0 END) OVER (PARTITION BY a), 0) AS n FROM t",
    "SELECT COALESCE(SUM(CASE WHEN b > 1 THEN 1 ELSE 0 END), 1) AS n FROM t",
    "SELECT COALESCE(SUM(CASE WHEN b > 1 THEN 1 ELSE 0 END), 0, 5) AS n FROM t",
    "SELECT COALESCE(SUM(CASE WHEN b > 1 THEN '1' ELSE '0' END), 0) AS n FROM t",
    "SELECT a, (SELECT SUM(CASE WHEN u.b > 1 THEN 1 ELSE 0 END) FROM u) AS n FROM t GROUP BY a",  # the inner select has no GROUP BY
])
def test_other_sums_are_left_alone(sql):
    assert counted(sql) is None


TARGETS = [
    (f"SELECT a, {ONES} AS n FROM t GROUP BY a", f"SELECT a, {COUNT} AS n FROM t GROUP BY a"),
    (f"SELECT a, {ONES} AS n FROM t GROUP BY a", "SELECT a, COUNT(*) FILTER (WHERE b > 1) AS n FROM t GROUP BY a"),
    (f"SELECT COALESCE({ONES}, 0) AS n FROM t", "SELECT COUNT(*) FILTER (WHERE b > 1) AS n FROM t"),
    ("SELECT COALESCE(SUM(CASE WHEN b > 1 THEN 1 END), 0) AS n FROM t", f"SELECT {COUNT} AS n FROM t"),
    (f"SELECT a FROM t GROUP BY a HAVING {ONES} > 1", "SELECT a FROM t GROUP BY a HAVING COUNT(*) FILTER (WHERE b > 1) > 1"),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_conditional_counts_prove(left, right):
    assert proved(left, right)


ROWS = {"t": [(1, 5, 1), (1, 6, 1), (1, 0, 1), (2, 0, 1), (2, None, 2), (None, 4, 1), (3, None, None)]}
EMPTY = {"t": []}
NEAR_MISSES = [
    # over no rows SUM is NULL, COUNT is 0
    (f"SELECT {ONES} AS n FROM t", f"SELECT {COUNT} AS n FROM t", EMPTY),
    (f"SELECT a, {ONES} AS n FROM t GROUP BY ROLLUP (a)", f"SELECT a, {COUNT} AS n FROM t GROUP BY ROLLUP (a)", EMPTY),
    # without ELSE 0 a group where no row passes sums to NULL
    ("SELECT a, SUM(CASE WHEN b > 1 THEN 1 END) AS n FROM t GROUP BY a", f"SELECT a, {COUNT} AS n FROM t GROUP BY a", ROWS),
    # other values, another condition, all rows
    ("SELECT a, SUM(CASE WHEN b > 1 THEN 2 ELSE 0 END) AS n FROM t GROUP BY a", f"SELECT a, {COUNT} AS n FROM t GROUP BY a", ROWS),
    (f"SELECT a, {ONES} AS n FROM t GROUP BY a", "SELECT a, COUNT(CASE WHEN b >= 1 THEN 1 END) AS n FROM t GROUP BY a", {"t": [(1, 1, 1)]}),
    (f"SELECT a, {ONES} AS n FROM t GROUP BY a", "SELECT a, COUNT(*) AS n FROM t GROUP BY a", ROWS),
    (f"SELECT a, {ONES} AS n FROM t GROUP BY a", "SELECT a, COUNT(b) AS n FROM t GROUP BY a", ROWS),
    (f"SELECT COALESCE({ONES}, 1) AS n FROM t", f"SELECT {COUNT} AS n FROM t", EMPTY),
    ("SELECT SUM(1) AS n FROM t", "SELECT COUNT(*) AS n FROM t", EMPTY),
    ("SELECT a, SUM(1) AS n FROM t GROUP BY a", "SELECT a, COUNT(b) AS n FROM t GROUP BY a", ROWS),
]


@pytest.mark.parametrize("left, right, rows", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right, rows):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, rows)


COUNTED_QUERIES = [
    f"SELECT a, {ONES} AS n FROM t GROUP BY a",
    f"SELECT a, c, {ONES} + 1 AS n, SUM(CASE WHEN b IS NULL THEN 1 ELSE 0 END) AS m FROM t GROUP BY a, c HAVING {ONES} >= 0",
    f"SELECT COALESCE({ONES}, 0) AS n, COALESCE(SUM(CASE WHEN a = 1 OR c = 1 THEN 1 END), 0) AS m FROM t",
    "SELECT a, COALESCE(SUM(CASE WHEN b > 1 THEN 1 ELSE NULL END), 0) AS n FROM t GROUP BY a",
    f"SELECT a, COALESCE({ONES}, 0) AS n FROM t GROUP BY ROLLUP (a)",
    "SELECT a, SUM(1) AS n, COALESCE(SUM(1), 0) AS m FROM t GROUP BY a",
    "SELECT COALESCE(SUM(1), 0) AS n FROM t WHERE b > 1",
]


def random_rows(rng):
    rows = [(rng.choice([None, 1, 2, 3]), rng.choice([None, 0, 1, 2, 5]), rng.choice([None, 1, 2])) for _ in range(rng.randint(0, 7))]
    return {"t": rows + rows[:2]}


@pytest.mark.parametrize("sql", COUNTED_QUERIES)
def test_counts_give_the_same_rows(sql):
    out = counted(sql)
    normalized = normalize(sql, schema=SCHEMA, types=TYPES, dialect="postgres")
    assert out is not None and "SUM(CASE WHEN b > 1" not in out and "SUM(1)" not in out
    rng = random.Random(len(sql))
    for rows in [{}, ROWS] + [random_rows(rng) for _ in range(14)]:
        assert not differ_on_duckdb(sql, out, rows)
        assert not differ_on_duckdb(sql, normalized, rows)


def inner_counted(sql):
    """The rule applied to the innermost grouped select of ``sql``."""

    tree = sqlglot.parse_one(sql, read="postgres")
    inner = max((s for s in tree.find_all(sqlglot.exp.Select) if s.args.get("group")), key=lambda s: s.depth)
    out = count_conditional_sums(inner)
    return out.sql(dialect="postgres") if out is not None else None


def test_a_sum_that_may_belong_to_an_enclosing_query_is_left_alone():
    # every column of the sum is the enclosing query's: the sum is that query's aggregate, NULL over no rows of t
    assert inner_counted("SELECT (SELECT SUM(CASE WHEN t.b > 1 THEN 1 ELSE 0 END) FROM u GROUP BY u.a) FROM t") is None
    assert inner_counted("SELECT a FROM t GROUP BY a HAVING EXISTS (SELECT SUM(CASE WHEN t.b > 1 THEN 1 ELSE 0 END) FROM u GROUP BY u.a)") is None
    # a bare name inside a subquery expression could be the enclosing query's column
    assert inner_counted("SELECT (SELECT SUM(CASE WHEN b > 1 THEN 1 ELSE 0 END) FROM u GROUP BY u.a) FROM t") is None
    assert inner_counted("SELECT t.a FROM t WHERE t.c IN (SELECT SUM(CASE WHEN t.b > 1 THEN 1 ELSE 0 END) FROM u AS t2 GROUP BY t2.a)") is None
    # a column of the select's own source makes it this select's aggregate, whatever else it reads
    assert (
        inner_counted("SELECT (SELECT SUM(CASE WHEN u.b > t.b THEN 1 ELSE 0 END) FROM u GROUP BY u.a) FROM t")
        == "SELECT COUNT(CASE WHEN u.b > t.b THEN 1 END) FROM u GROUP BY u.a"
    )
    # no column at all: the innermost query's aggregate
    assert inner_counted("SELECT (SELECT SUM(1) FROM u GROUP BY u.a) FROM t") == "SELECT COUNT(*) FROM u GROUP BY u.a"
    # a derived table sees no enclosing row, bare names included; so does a branch of a set operation
    assert (
        inner_counted("SELECT d.n FROM (SELECT a, SUM(CASE WHEN b > 1 THEN 1 ELSE 0 END) AS n FROM u GROUP BY a) AS d")
        == "SELECT a, COUNT(CASE WHEN b > 1 THEN 1 END) AS n FROM u GROUP BY a"
    )
    assert inner_counted("SELECT a, 0 FROM t UNION ALL SELECT a, SUM(CASE WHEN b > 1 THEN 1 ELSE 0 END) FROM u GROUP BY a") is not None
    # the COALESCE spelling is the count over any bag of rows, so it is read wherever it stands
    assert inner_counted("SELECT (SELECT COALESCE(SUM(CASE WHEN t.b > 1 THEN 1 ELSE 0 END), 0) FROM u GROUP BY u.a) FROM t") is not None
