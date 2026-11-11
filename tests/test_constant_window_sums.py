"""A windowed SUM of an integer literal is the literal times the windowed COUNT(*) (``kumo_sql_prover.constant_window_sums``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.constant_window_sums import count_constant_window_sums

SCHEMA = {"t": ["a", "b", "c"], "u": ["a", "b", "c"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}


def spelled(sql):
    return sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres")


def counted(sql):
    return count_constant_window_sums(sqlglot.parse_one(sql, read="postgres")).sql(dialect="postgres")


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


FRAMES = [
    "PARTITION BY a",
    "PARTITION BY a ORDER BY b",
    "ORDER BY b",
    "",
    "PARTITION BY a ORDER BY b RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
    "PARTITION BY a ORDER BY b ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
    "PARTITION BY a ORDER BY b RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING",
    "ORDER BY b, a, c ROWS BETWEEN CURRENT ROW AND CURRENT ROW",
    "ORDER BY b ROWS UNBOUNDED PRECEDING",
]


@pytest.mark.parametrize("frame", FRAMES)
def test_a_constant_sum_over_a_frame_with_the_current_row_is_counted(frame):
    assert counted(f"SELECT a, SUM(100) OVER ({frame}) AS s FROM t") == spelled(f"SELECT a, 100 * COUNT(*) OVER ({frame}) AS s FROM t")
    assert counted(f"SELECT a, 1 + SUM(3) OVER ({frame}) AS s FROM t") == spelled(f"SELECT a, 1 + (3 * COUNT(*) OVER ({frame})) AS s FROM t")


@pytest.mark.parametrize("sql", [
    "SELECT a, SUM(100) OVER (ORDER BY b ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS s FROM t",  # empty on the first row
    "SELECT a, SUM(100) OVER (ORDER BY b ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS s FROM t",
    "SELECT a, SUM(100) OVER (ORDER BY b ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM t",  # an offset bound: not read
    "SELECT a, SUM(100) OVER (ORDER BY b ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW) AS s FROM t",
    "SELECT a, SUM(100) FILTER (WHERE b > 1) OVER (PARTITION BY a) AS s FROM t",
    "SELECT a, SUM(DISTINCT 100) OVER (PARTITION BY a) AS s FROM t",
    "SELECT a, SUM(1.5) OVER (PARTITION BY a) AS s FROM t",
    "SELECT a, SUM(b) OVER (PARTITION BY a) AS s FROM t",
    "SELECT a, SUM(-100) OVER (PARTITION BY a) AS s FROM t",
    "SELECT a, MAX(100) OVER (PARTITION BY a) AS s FROM t",
    "SELECT a, SUM(100) OVER w AS s FROM t WINDOW w AS (PARTITION BY a)",
    "SELECT a, SUM(100) AS s FROM t GROUP BY a",  # not a window
])
def test_other_sums_are_left_alone(sql):
    assert counted(sql) == spelled(sql)


TARGETS = [
    (f"SELECT a, SUM(100) OVER ({frame}) AS s FROM t WHERE c = 1", f"SELECT a, 100 * COUNT(*) OVER ({frame}) AS s FROM t WHERE c = 1")
    for frame in FRAMES[:6]
] + [
    (
        "SELECT SUM(7) OVER (PARTITION BY a, b ORDER BY b) AS x, SUM(b) OVER (PARTITION BY a) AS y FROM t",
        "SELECT COUNT(*) OVER (PARTITION BY a, b ORDER BY b) * 7 AS x, SUM(b) OVER (PARTITION BY a) AS y FROM t",
    ),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_constant_window_sums_prove(left, right):
    assert proved(left, right)


ROWS = {"t": [(1, 5, 1), (1, 6, 1), (1, None, 1), (2, 0, 1), (2, None, 2), (None, 4, 1)]}
NEAR_MISSES = [
    # a frame that can be empty: SUM is NULL there, the product is 0
    (
        "SELECT a, b, SUM(100) OVER (ORDER BY b, a, c ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS s FROM t",
        "SELECT a, b, 100 * COUNT(*) OVER (ORDER BY b, a, c ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS s FROM t",
    ),
    (
        "SELECT a, b, SUM(100) OVER (PARTITION BY a ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE CURRENT ROW) AS s FROM t",
        "SELECT a, b, 100 * COUNT(*) OVER (PARTITION BY a ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE CURRENT ROW) AS s FROM t",
    ),
    ("SELECT a, SUM(100) FILTER (WHERE b > 5) OVER (PARTITION BY a) AS s FROM t", "SELECT a, 100 * COUNT(*) FILTER (WHERE b > 5) OVER (PARTITION BY a) AS s FROM t"),
    # another count, another partition, another factor
    ("SELECT a, SUM(100) OVER (PARTITION BY a) AS s FROM t", "SELECT a, 100 * COUNT(b) OVER (PARTITION BY a) AS s FROM t"),
    ("SELECT a, SUM(100) OVER (PARTITION BY a) AS s FROM t", "SELECT a, 100 * COUNT(*) OVER (PARTITION BY c) AS s FROM t"),
    ("SELECT a, SUM(100) OVER (PARTITION BY a) AS s FROM t", "SELECT a, 10 * COUNT(*) OVER (PARTITION BY a) AS s FROM t"),
    ("SELECT a, SUM(100) OVER (PARTITION BY a ORDER BY b) AS s FROM t", "SELECT a, 100 * COUNT(*) OVER (PARTITION BY a) AS s FROM t"),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, ROWS)


def random_rows(rng):
    rows = [(rng.choice([None, 1, 2, 3]), rng.choice([None, 0, 1, 2, 5]), rng.choice([None, 1, 2])) for _ in range(rng.randint(0, 7))]
    return {"t": rows + rows[:2]}


@pytest.mark.parametrize("frame", [f for f in FRAMES if "ROWS BETWEEN CURRENT ROW AND CURRENT ROW" in f or "ROWS" not in f or "UNBOUNDED FOLLOWING" in f])
def test_the_product_gives_the_same_rows(frame):
    # frames whose rows do not depend on how ties are ordered, so the replay is deterministic
    sql = f"SELECT a, b, SUM(100) OVER ({frame}) AS s, 2 + SUM(3) OVER ({frame}) AS r FROM t"
    out = counted(sql)
    normalized = normalize(sql, schema=SCHEMA, types=TYPES, dialect="postgres")
    assert "SUM(" not in out and "SUM(" not in normalized
    rng = random.Random(len(sql))
    for rows in [{}, ROWS] + [random_rows(rng) for _ in range(14)]:
        assert not differ_on_duckdb(sql, out, rows)
        assert not differ_on_duckdb(sql, normalized, rows)
