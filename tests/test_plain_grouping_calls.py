"""GROUPING(..) under a plain GROUP BY is 0 (``kumo_sql_prover.plain_grouping_calls``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.plain_grouping_calls import fold_plain_grouping_calls

SCHEMA = {"t": ["a", "b", "c"], "u": ["a", "b", "c"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}


def folded(sql):
    return fold_plain_grouping_calls(sqlglot.parse_one(sql, read="postgres")).sql(dialect="postgres")


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


def test_grouping_of_keys_under_a_plain_group_by_is_zero():
    assert folded("SELECT a, GROUPING(a) AS g FROM t GROUP BY a") == "SELECT a, 0 AS g FROM t GROUP BY a"
    assert folded("SELECT t.a, GROUPING(a, t.b) = 0 AS g FROM t GROUP BY t.a, b") == "SELECT t.a, 0 = 0 AS g FROM t GROUP BY t.a, b"
    assert folded("SELECT a > 1 AS p, GROUPING(a > 1) AS g FROM t GROUP BY a > 1") == "SELECT a > 1 AS p, 0 AS g FROM t GROUP BY a > 1"
    assert folded("SELECT a FROM t GROUP BY a HAVING GROUPING(a) = 0 ORDER BY GROUPING(a)") == "SELECT a FROM t GROUP BY a HAVING 0 = 0 ORDER BY 0"
    # each select on its own
    assert folded("SELECT d.a, d.g FROM (SELECT a, GROUPING(a) AS g FROM t GROUP BY a) AS d") == "SELECT d.a, d.g FROM (SELECT a, 0 AS g FROM t GROUP BY a) AS d"


@pytest.mark.parametrize("sql", [
    "SELECT a, GROUPING(a) AS g FROM t GROUP BY ROLLUP (a)",
    "SELECT a, GROUPING(a) AS g FROM t GROUP BY CUBE (a)",
    "SELECT a, GROUPING(a) AS g FROM t GROUP BY GROUPING SETS ((a), ())",
    "SELECT a, GROUPING(a) AS g FROM t GROUP BY a, ROLLUP (b)",
    "SELECT a, GROUPING(b) AS g FROM t GROUP BY a",  # not a key: no engine accepts it
    "SELECT a, GROUPING(a + 1) AS g FROM t GROUP BY a",
    "SELECT t.a, GROUPING(u.a) AS g FROM t JOIN u ON t.b = u.b GROUP BY t.a",
    "SELECT t.a, u.a, GROUPING(a) AS g FROM t JOIN u ON t.b = u.b GROUP BY t.a, u.a",  # which a?
    "SELECT GROUPING(a) AS g FROM t",
    "SELECT a, (SELECT GROUPING(u.a) FROM u GROUP BY ROLLUP (u.a) LIMIT 1) AS g FROM t GROUP BY a",
])
def test_other_calls_are_left_alone(sql):
    assert folded(sql) == sql


FLAGGED = "SELECT d.a, MIN(d.s) FILTER (WHERE d.g) AS s FROM (SELECT a, SUM(b) AS s, GROUPING(a) = 0 AS g FROM t GROUP BY a) AS d GROUP BY d.a"
TARGETS = [
    ("SELECT a, SUM(b) AS s, GROUPING(a) AS g FROM t GROUP BY a", "SELECT a, SUM(b) AS s, 0 AS g FROM t GROUP BY a"),
    ("SELECT a, SUM(b) AS s FROM t GROUP BY a HAVING GROUPING(a) = 0", "SELECT a, SUM(b) AS s FROM t GROUP BY a"),
    (FLAGGED, "SELECT a, SUM(b) AS s FROM t GROUP BY a"),
    (
        "SELECT COUNT(d.p) FILTER (WHERE d.g) AS n FROM (SELECT a > 1 AS p, GROUPING(a > 1, b) = 0 AND b > 2 AS g FROM t GROUP BY a > 1, b) AS d",
        "SELECT COUNT(d.p) FILTER (WHERE d.g) AS n FROM (SELECT a > 1 AS p, b > 2 AS g FROM t GROUP BY a > 1, b) AS d",
    ),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_plain_grouping_calls_prove(left, right):
    assert proved(left, right)


ROWS = {"t": [(1, 5, 1), (1, 6, 1), (2, 7, 1), (None, 4, 1), (None, None, 2)]}
ROLLUP = "SELECT a, SUM(b) AS s, GROUPING(a) AS g FROM t GROUP BY ROLLUP (a)"
NEAR_MISSES = [
    # under ROLLUP the grand-total row has GROUPING(a) = 1
    (ROLLUP, "SELECT a, SUM(b) AS s, 0 AS g FROM t GROUP BY ROLLUP (a)"),
    (ROLLUP, "SELECT a, SUM(b) AS s, 0 AS g FROM t GROUP BY a"),
    ("SELECT a, SUM(b) AS s FROM t GROUP BY ROLLUP (a) HAVING GROUPING(a) = 0", "SELECT a, SUM(b) AS s FROM t GROUP BY ROLLUP (a)"),
    # the flag is TRUE, not FALSE
    (FLAGGED, FLAGGED.replace("GROUPING(a) = 0", "GROUPING(a) = 1")),
    ("SELECT a, SUM(b) AS s FROM t GROUP BY a HAVING GROUPING(a) = 1", "SELECT a, SUM(b) AS s FROM t GROUP BY a"),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, ROWS)


FOLDED_QUERIES = [
    "SELECT a, SUM(b) AS s, GROUPING(a) AS g FROM t GROUP BY a",
    "SELECT a, b, GROUPING(a, b) AS g, GROUPING(b) + 1 AS h FROM t GROUP BY a, b",
    "SELECT a > 1 AS p, COUNT(*) AS n, GROUPING(a > 1) = 0 AS g FROM t GROUP BY a > 1",
    "SELECT a, SUM(b) AS s FROM t GROUP BY a HAVING GROUPING(a) = 0",
    FLAGGED,
]


def random_rows(rng):
    rows = [(rng.choice([None, 1, 2, 3]), rng.choice([None, 0, 1, 2]), rng.choice([None, 1, 2])) for _ in range(rng.randint(0, 6))]
    return {"t": rows + rows[:2]}


@pytest.mark.parametrize("sql", FOLDED_QUERIES)
def test_folded_calls_give_the_same_rows(sql):
    out = folded(sql)
    normalized = normalize(sql, schema=SCHEMA, types=TYPES, dialect="postgres")
    assert "GROUPING" not in out and "GROUPING" not in normalized
    rng = random.Random(len(sql))
    for rows in [{}, ROWS] + [random_rows(rng) for _ in range(14)]:
        assert not differ_on_duckdb(sql, out, rows)
        assert not differ_on_duckdb(sql, normalized, rows)
