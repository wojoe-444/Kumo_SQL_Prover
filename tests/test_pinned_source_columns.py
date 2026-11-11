"""A derived table's column pinned to a whole number by its own ``WHERE`` (``kumo_sql_prover.pinned_source_columns``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.pinned_source_columns import read_pinned_source_columns

SCHEMA = {"s": ["k", "y", "q", "x", "d", "c"], "o": ["k", "z"]}
TYPES = {
    "s": {"k": "BIGINT", "y": "INTEGER", "q": "INTEGER", "x": "BIGINT", "d": "DECIMAL(7,2)", "c": "CHAR(4)"},
    "o": {"k": "BIGINT", "z": "BIGINT"},
}

PINNED = "(SELECT k, y, SUM(x) AS t FROM s WHERE y = 2000 GROUP BY k, y) AS g"


def read(sql, types=TYPES):
    out = read_pinned_source_columns(sqlglot.parse_one(sql, read="postgres"), types)
    return None if out is None else out.sql(dialect="postgres")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="postgres").proven


def test_the_read_becomes_the_number():
    assert read(f"SELECT g.k, g.y, g.t FROM {PINNED}") == f"SELECT g.k, 2000 AS y, g.t FROM {PINNED}"
    assert read(f"SELECT g.y + 1 AS n FROM {PINNED} JOIN o ON o.k = g.k AND o.z > g.y WHERE g.y < o.z") == (
        f"SELECT 2000 + 1 AS n FROM {PINNED} JOIN o ON o.k = g.k AND o.z > 2000 WHERE 2000 < o.z"
    )
    # the preserved side of a LEFT JOIN, a number on the left of the comparison, an ungrouped source, a bare column
    assert read("SELECT g.q FROM (SELECT s.q, s.x FROM s WHERE 3 = s.q AND s.x > 0) AS g LEFT JOIN o ON o.k = g.x") == (
        "SELECT 3 AS q FROM (SELECT s.q, s.x FROM s WHERE 3 = s.q AND s.x > 0) AS g LEFT JOIN o ON o.k = g.x"
    )
    assert read("SELECT g.w FROM (SELECT q AS w FROM s WHERE q = 3) AS g") == "SELECT 3 AS w FROM (SELECT q AS w FROM s WHERE q = 3) AS g"
    # GROUP BY and ORDER BY keep the column: a number there is a position
    assert read(f"SELECT g.y, COUNT(*) FROM {PINNED} GROUP BY g.y ORDER BY g.y") == f"SELECT 2000 AS y, COUNT(*) FROM {PINNED} GROUP BY g.y ORDER BY g.y"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT g.y FROM (SELECT y FROM s WHERE y > 2000) AS g",  # not an equality
        "SELECT g.y FROM (SELECT y FROM s WHERE y = 2000 OR q = 1) AS g",
        "SELECT g.y FROM (SELECT y FROM s WHERE NOT y = 2000) AS g",
        "SELECT g.y FROM (SELECT y FROM s WHERE y = q) AS g",
        "SELECT g.y FROM (SELECT y FROM s WHERE y = 2000.0) AS g",
        "SELECT g.y FROM (SELECT y FROM s WHERE y = -1) AS g",
        "SELECT g.y FROM (SELECT y FROM s WHERE y + 0 = 2000) AS g",
        "SELECT g.d FROM (SELECT d FROM s WHERE d = 2000) AS g",  # 2000.00 is not written 2000
        "SELECT g.c FROM (SELECT c FROM s WHERE c = 'ab') AS g",  # a padded CHAR
        "SELECT g.n FROM (SELECT y + 1 AS n FROM s WHERE y = 2000) AS g",  # not the column itself
        "SELECT g.q FROM (SELECT q FROM s WHERE y = 2000) AS g",  # another column
        "SELECT g.y FROM (SELECT y, SUM(x) FROM s WHERE y = 2000 GROUP BY ROLLUP (y)) AS g",  # subtotal rows hold NULL
        "SELECT g.y FROM (SELECT y FROM s WHERE y = 2000 UNION ALL SELECT y FROM s) AS g",
        "SELECT g.y FROM (SELECT MAX(y) AS y FROM s WHERE y = 2000) AS g",  # a row for no input, holding NULL
        "SELECT g.y FROM (SELECT y FROM s WHERE y = 2000) AS g(z)",
        "SELECT g.y FROM o LEFT JOIN (SELECT k, y FROM s WHERE y = 2000) AS g ON g.k = o.k",  # padded with NULL
        "SELECT g.y FROM (SELECT k, y FROM s WHERE y = 2000) AS g RIGHT JOIN o ON g.k = o.k",
        "SELECT g.y FROM (SELECT k, y FROM s WHERE y = 2000) AS g FULL JOIN o ON g.k = o.k",
        "SELECT g.y, COUNT(*) FROM (SELECT k, y FROM s WHERE y = 2000) AS g GROUP BY ROLLUP (g.y)",
        "SELECT y FROM (SELECT y FROM s WHERE y = 2000) AS g",  # a bare read
        "SELECT * FROM (SELECT y FROM s WHERE y = 2000) AS g",
        "SELECT (SELECT MAX(o.z) FROM o WHERE o.k = g.y) FROM (SELECT y FROM s WHERE y = 2000) AS g",  # inside a subquery
        "SELECT COUNT(*) FROM (SELECT y FROM s WHERE y = 2000) AS g GROUP BY g.y",
        "SELECT g.y FROM (SELECT y FROM s WHERE y = 2000) AS g, (SELECT y FROM s WHERE y = 2000) AS g",
        "SELECT g.y FROM (SELECT m.y FROM missing AS m WHERE m.y = 2000) AS g",  # a table without declared types
        "SELECT g.y FROM (SELECT h.y FROM (SELECT y FROM s) AS h WHERE h.y = 2000) AS g",  # not a base table's column
    ],
)
def test_what_is_left_alone(sql):
    assert read(sql) is None, sql


def test_nothing_without_types():
    assert read(f"SELECT g.y FROM {PINNED}", types=None) is None
    assert read(f"SELECT g.y FROM {PINNED}", types={"s": {"y": "NUMERIC"}}) is None


def test_pairs_it_proves():
    grouped = "(SELECT s.k, s.y, s.q, SUM(s.x) AS t FROM s, o WHERE s.k = o.k GROUP BY s.k, s.y, s.q)"
    # the same grouped CTE read under two filters, against the literal an optimizer writes
    left = f"SELECT a.k, a.y, b.t / a.t FROM {grouped} AS a, {grouped} AS b WHERE a.y = 2000 AND a.q = 1 AND b.y = 2000 AND b.q = 2 AND a.k = b.k"
    right = (
        "SELECT a.k, 2000, b.t / a.t FROM (SELECT s.k, SUM(s.x) AS t FROM s JOIN o ON s.k = o.k WHERE s.y = 2000 AND s.q = 1 GROUP BY s.k, s.y, s.q) AS a "
        "JOIN (SELECT s.k, SUM(s.x) AS t FROM s JOIN o ON s.k = o.k WHERE s.y = 2000 AND s.q = 2 GROUP BY s.k, s.y, s.q) AS b ON a.k = b.k"
    )
    assert proved(left, right)
    assert proved(f"SELECT g.y, g.t FROM {PINNED}", f"SELECT 2000, g.t FROM {PINNED}")


def test_near_misses_stay_unproven():
    assert not proved(f"SELECT g.y, g.t FROM {PINNED}", f"SELECT 2001, g.t FROM {PINNED}")
    assert not proved(f"SELECT g.k, g.t FROM {PINNED}", f"SELECT 2000, g.t FROM {PINNED}")
    # the padded side of an outer join reads NULL, not the number
    padded = "SELECT o.k, {} FROM o LEFT JOIN (SELECT k, y FROM s WHERE y = 2000) AS g ON g.k = o.k"
    assert not proved(padded.format("g.y"), padded.format("2000"))
    # a filter that only bounds the column does not name its value
    bounded = "SELECT {} FROM (SELECT k, y FROM s WHERE y >= 2000) AS g"
    assert not proved(bounded.format("g.y"), bounded.format("2000"))
    assert not proved("SELECT g.y FROM (SELECT MAX(y) AS y FROM s WHERE y = 2000) AS g", "SELECT 2000 FROM (SELECT MAX(y) AS y FROM s WHERE y = 2000) AS g")


def test_the_engine_agrees_on_random_tables():
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(20261116)
    queries = [
        f"SELECT g.k, g.y, g.t FROM {PINNED}",
        f"SELECT g.y + 1 AS n, o.z FROM {PINNED} JOIN o ON o.k = g.k AND o.z > g.y - 2001 WHERE g.y < o.z + 2001",
        "SELECT g.q, o.z FROM (SELECT s.q, s.x FROM s WHERE 3 = s.q AND s.x > 0) AS g LEFT JOIN o ON o.k = g.x",
        f"SELECT g.y, COUNT(*) FROM {PINNED} GROUP BY g.y ORDER BY g.y",
        "SELECT DISTINCT g.q FROM (SELECT q, k FROM s WHERE q = 3 LIMIT 5) AS g",
    ]
    rows_seen = 0
    for trial in range(30):
        db = duckdb.connect()
        db.execute("CREATE TABLE s (k BIGINT, y INTEGER, q INTEGER, x BIGINT, d DECIMAL(7,2), c VARCHAR)")
        db.execute("CREATE TABLE o (k BIGINT, z BIGINT)")
        for _ in range(rng.choice([0, 1, 4, 12]) if trial % 5 else 0):
            db.execute(
                "INSERT INTO s VALUES (?, ?, ?, ?, NULL, NULL)",
                [rng.choice([None, 1, 2, 3]), rng.choice([None, 1999, 2000, 2000]), rng.choice([None, 1, 3, 3]), rng.choice([None, 0, 1, 2, 3])],
            )
        for _ in range(rng.choice([0, 2, 6])):
            db.execute("INSERT INTO o VALUES (?, ?)", [rng.choice([None, 1, 2, 3]), rng.choice([None, 0, 5])])
        for sql in queries:
            after = read(sql)
            assert after is not None, sql
            key = lambda row: tuple((v is None, v) for v in row)  # noqa: E731
            before_rows = sorted(db.execute(sql).fetchall(), key=key)
            assert before_rows == sorted(db.execute(after).fetchall(), key=key), (sql, after)
            rows_seen += len(before_rows)
    assert rows_seen > 50


def test_off_on_the_second_attempt():
    sql = f"SELECT g.y, g.t, o.z FROM {PINNED} JOIN o ON o.k = g.k"
    assert "2000 AS y" in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)
    with shape_fallback.disabled():
        assert "2000 AS y" not in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)
