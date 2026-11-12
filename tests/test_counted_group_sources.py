"""``COUNT(*)`` pushed below a chain of joins (``kumo_sql_prover.counted_group_sources``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.counted_group_sources import read_counted_group_sources

SCHEMA = {"s": ["k", "j", "x"], "o": ["k", "y", "z"], "d": ["j", "w"]}
TYPES = {table: {column: "BIGINT" for column in columns} for table, columns in SCHEMA.items()}

G = "(SELECT k, COUNT(*) AS n FROM s GROUP BY k) AS g"
G_FLAT = "(SELECT k FROM s) AS g"
H = "(SELECT k, z, COUNT(*) AS n FROM o WHERE y > 0 GROUP BY k, z) AS h"
H_FLAT = "(SELECT k, z FROM o WHERE y > 0) AS h"
TWO_LEVELS = (
    "SELECT t.z, COALESCE(SUM(t.n * e.n), 0) AS c FROM (SELECT h.z, g.j, COALESCE(SUM(g.n * h.n), 0) AS n FROM "
    "(SELECT k, j, COUNT(*) AS n FROM s GROUP BY k, j) AS g JOIN (SELECT k, z, COUNT(*) AS n FROM o WHERE y > 0 GROUP BY k, z) AS h "
    "ON g.k = h.k GROUP BY h.z, g.j) AS t JOIN (SELECT j, COUNT(*) AS n FROM d GROUP BY j) AS e ON t.j = e.j GROUP BY t.z"
)


def read(sql):
    return read_counted_group_sources(sqlglot.parse_one(sql, read="postgres")).sql(dialect="postgres")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="postgres").proven


def test_the_source_loses_its_group_by_and_its_count():
    assert read(f"SELECT o.z, SUM(g.n * o.y) FROM {G} JOIN o ON g.k = o.k GROUP BY o.z") == f"SELECT o.z, SUM(o.y) FROM {G_FLAT} JOIN o ON g.k = o.k GROUP BY o.z"
    assert read(f"SELECT h.z, SUM(g.n * h.n) AS c FROM {G}, {H} WHERE g.k = h.k GROUP BY h.z HAVING SUM(g.n * h.n) > 2") == (
        f"SELECT h.z, COUNT(*) AS c FROM {G_FLAT}, {H_FLAT} WHERE g.k = h.k GROUP BY h.z HAVING COUNT(*) > 2"
    )
    # without GROUP BY a sum over no rows is NULL: only COALESCE(.., 0) is COUNT(*)
    assert read(f"SELECT COALESCE(SUM(g.n), 0) FROM {G}") == f"SELECT COUNT(*) FROM {G_FLAT}"
    assert read(f"SELECT SUM(g.n * h.n) FROM {G}, {H} WHERE g.k = h.k") == f"SELECT SUM(h.n) FROM {G_FLAT}, {H} WHERE g.k = h.k"
    # a source that outputs nothing but its count keeps a column
    assert read("SELECT SUM(g.n * o.y) FROM (SELECT COUNT(*) AS n FROM s GROUP BY k) AS g, o GROUP BY o.z") == (
        "SELECT SUM(o.y) FROM (SELECT 1 AS n FROM s) AS g, o GROUP BY o.z"
    )


def test_levels_come_apart_from_the_inside():
    assert read(TWO_LEVELS) == (
        "SELECT t.z, COUNT(*) AS c FROM (SELECT h.z, g.j FROM (SELECT k, j FROM s) AS g JOIN (SELECT k, z FROM o WHERE y > 0) AS h "
        "ON g.k = h.k) AS t JOIN (SELECT j FROM d) AS e ON t.j = e.j GROUP BY t.z"
    )
    assert proved(TWO_LEVELS, "SELECT o.z, COUNT(*) FROM s, o, d WHERE s.k = o.k AND o.y > 0 AND s.j = d.j GROUP BY o.z")


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT o.z, COUNT(*) FROM {G} JOIN o ON g.k = o.k GROUP BY o.z",  # counts the groups, not the rows
        f"SELECT o.z, SUM(g.n * o.y), SUM(o.y) FROM {G} JOIN o ON g.k = o.k GROUP BY o.z",  # a sum the count does not weigh
        f"SELECT o.z, SUM(g.n * o.y), MAX(o.y) FROM {G} JOIN o ON g.k = o.k GROUP BY o.z",  # true, but not read
        f"SELECT o.z, SUM(g.n * o.y), AVG(g.n * o.y) FROM {G} JOIN o ON g.k = o.k GROUP BY o.z",
        f"SELECT o.z, SUM(g.n + o.y) FROM {G} JOIN o ON g.k = o.k GROUP BY o.z",
        f"SELECT o.z, SUM(g.n * g.n) FROM {G} JOIN o ON g.k = o.k GROUP BY o.z",
        f"SELECT o.z, SUM(g.n * (g.n + o.y)) FROM {G} JOIN o ON g.k = o.k GROUP BY o.z",
        f"SELECT o.z, SUM(o.y / g.n) FROM {G} JOIN o ON g.k = o.k GROUP BY o.z",
        f"SELECT o.z, SUM(DISTINCT g.n * o.y) FROM {G} JOIN o ON g.k = o.k GROUP BY o.z",
        f"SELECT o.z, SUM(g.n * o.y) FILTER (WHERE o.y > 0) FROM {G} JOIN o ON g.k = o.k GROUP BY o.z",
        f"SELECT o.z, SUM(g.n * o.y) FROM {G} JOIN o ON g.k = o.k WHERE g.n > 1 GROUP BY o.z",  # the count as a filter
        f"SELECT g.n, SUM(g.n * o.y) FROM {G} JOIN o ON g.k = o.k GROUP BY g.n",
        f"SELECT o.z, SUM(g.n * o.y) FROM {G} JOIN o ON g.k = o.k AND EXISTS (SELECT 1 FROM d WHERE d.w = g.n) GROUP BY o.z",
        f"SELECT o.z, SUM(n * o.y) FROM {G} JOIN o ON g.k = o.k GROUP BY o.z",  # a bare name
        f"SELECT SUM(g.n) FROM {G}",  # NULL over no rows, COUNT(*) would be 0
        f"SELECT g.k, g.n FROM {G}",
        f"SELECT DISTINCT SUM(g.n * o.y) FROM {G} JOIN o ON g.k = o.k GROUP BY o.z",
        f"SELECT o.z, SUM(g.n * o.y) FROM {G} JOIN o ON g.k = o.k GROUP BY o.z LIMIT 1",
        f"SELECT o.z, SUM(g.n * o.y) FROM o LEFT JOIN {G} ON g.k = o.k GROUP BY o.z",  # padded with NULL counts
        f"SELECT o.z, SUM(g.n * o.y) FROM {G} RIGHT JOIN o ON g.k = o.k GROUP BY o.z",
        f"SELECT o.z, SUM(g.n * o.y) FROM {G} FULL JOIN o ON g.k = o.k GROUP BY o.z",
        f"SELECT o.z, SUM(g.n * o.y) FROM {G} JOIN o ON g.k = o.k GROUP BY ROLLUP (o.z)",
        f"SELECT o.z, SUM(g.n * o.y), SUM(g.n * o.y) OVER () FROM {G} JOIN o ON g.k = o.k GROUP BY o.z, g.n, o.y",
        f"SELECT o.z, MY_AGG(g.n * o.y) FROM {G} JOIN o ON g.k = o.k GROUP BY o.z",
        "SELECT SUM(g.n * o.y) FROM (SELECT k, COUNT(*) AS n FROM s GROUP BY k HAVING COUNT(*) > 1) AS g, o GROUP BY o.z",
        "SELECT SUM(g.n * o.y) FROM (SELECT COUNT(*) AS n FROM s) AS g, o GROUP BY o.z",  # no GROUP BY: a row for no input
        "SELECT SUM(g.n * o.y) FROM (SELECT k, COUNT(x) AS n FROM s GROUP BY k) AS g, o GROUP BY o.z",  # skips NULLs
        "SELECT SUM(g.n * o.y) FROM (SELECT k, COUNT(DISTINCT x) AS n FROM s GROUP BY k) AS g, o GROUP BY o.z",
        "SELECT SUM(g.n * o.y) FROM (SELECT k, COUNT(*) + 1 AS n FROM s GROUP BY k) AS g, o GROUP BY o.z",
        "SELECT SUM(g.n * o.y) FROM (SELECT k, COUNT(*) AS n, COUNT(*) AS m FROM s GROUP BY k) AS g, o GROUP BY o.z",
        "SELECT SUM(g.n * o.y) FROM (SELECT k, COUNT(*) AS n, MAX(x) AS m FROM s GROUP BY k) AS g, o GROUP BY o.z",
        "SELECT SUM(g.n * o.y) FROM (SELECT k, COUNT(*) AS n FROM s GROUP BY k LIMIT 3) AS g, o GROUP BY o.z",
        "SELECT SUM(g.n * o.y) FROM (SELECT k, COUNT(*) AS n FROM s GROUP BY 1) AS g, o GROUP BY o.z",
        "SELECT SUM(g.n * o.y) FROM (SELECT k + 1 AS k, COUNT(*) AS n FROM s GROUP BY k) AS g, o GROUP BY o.z",
        "SELECT SUM(g.c * o.y) FROM (SELECT k, COUNT(*) AS n FROM s GROUP BY k) AS g(a, c), o GROUP BY o.z",
        f"SELECT SUM(g.n * o.y) FROM {G}, {G}, o GROUP BY o.z",
        # the type of the sum changes with the rewrite, and a division reads the type
        f"SELECT q.c / 2 FROM (SELECT o.z, SUM(g.n * o.y) AS c FROM {G} JOIN o ON g.k = o.k GROUP BY o.z) AS q",
        f"SELECT o.z, SUM(g.n * o.y) FROM {G} JOIN o ON g.k = o.k WHERE o.y % 2 = 0 GROUP BY o.z",
    ],
)
def test_what_is_left_alone(sql):
    assert read(sql) == sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres"), sql


def test_pairs_it_proves():
    assert proved(
        "SELECT o.z, SUM(o.y) FROM s, o WHERE s.k = o.k AND s.x > 1 GROUP BY o.z",
        "SELECT o.z, SUM(g.n * o.y) FROM (SELECT k, COUNT(*) AS n FROM s WHERE x > 1 GROUP BY k) AS g JOIN o ON g.k = o.k GROUP BY o.z",
    )
    assert proved("SELECT COUNT(*) FROM s, o WHERE s.k = o.k AND o.y > 0", f"SELECT COALESCE(SUM(g.n * h.n), 0) FROM {G}, {H} WHERE g.k = h.k")


FLAT = "SELECT o.z, COUNT(*) FROM s, o, d WHERE s.k = o.k AND o.y > 0 AND s.j = d.j GROUP BY o.z"


@pytest.mark.parametrize(
    "left, right",
    [
        (FLAT, TWO_LEVELS.replace("COALESCE(SUM(t.n * e.n), 0)", "COALESCE(SUM(t.n), 0)")),  # one table's rows not counted
        (FLAT, TWO_LEVELS.replace("COALESCE(SUM(t.n * e.n), 0)", "COUNT(*)")),
        (FLAT, TWO_LEVELS.replace("WHERE y > 0 ", "")),
        ("SELECT COUNT(*) FROM s", f"SELECT SUM(g.n) FROM {G}"),  # 0 against NULL over an empty table
    ],
)
def test_near_misses_stay_unproven(left, right):
    assert not proved(left, right)


def test_the_engine_agrees_on_random_tables():
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(20261119)
    value = lambda: rng.choice([None, *range(-1, 3)])  # noqa: E731
    queries = [
        f"SELECT o.z, SUM(g.n * o.y) FROM {G} JOIN o ON g.k = o.k GROUP BY o.z",
        f"SELECT h.z, SUM(g.n * h.n) AS c FROM {G}, {H} WHERE g.k = h.k GROUP BY h.z HAVING SUM(g.n * h.n) > 2",
        f"SELECT COALESCE(SUM(g.n), 0) FROM {G}",
        f"SELECT COALESCE(SUM(g.n * h.n), 0), SUM(g.n * h.z) FROM {G}, {H} WHERE g.k = h.k",
        f"SELECT SUM(g.n * h.n) FROM {G}, {H} WHERE g.k = h.k",
        "SELECT SUM(g.n * o.y) FROM (SELECT COUNT(*) AS n FROM s GROUP BY k) AS g, o GROUP BY o.z",
        f"SELECT g.k, SUM(g.n * o.y), SUM(o.z * g.n * 2) FROM {G} LEFT JOIN o ON g.k = o.k AND o.y > 0 GROUP BY g.k",
        TWO_LEVELS,
    ]
    readings = [read(sql) for sql in queries]
    assert all(after != sql for sql, after in zip(queries, readings))
    rows_seen = 0
    for trial in range(24):
        db = duckdb.connect()
        for table, columns in SCHEMA.items():
            db.execute(f"CREATE TABLE {table} ({', '.join(c + ' BIGINT' for c in columns)})")
            rows = [tuple(value() for _ in columns) for _ in range(rng.choice([0, 1, 4, 9]) if trial % 4 else 0 if table == "s" else 5)]
            if rows:
                db.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in columns)})", rows)
        for sql, after in zip(queries, readings):
            key = lambda row: tuple((v is None, v) for v in row)  # noqa: E731
            before_rows = sorted(db.execute(sql).fetchall(), key=key)
            assert before_rows == sorted(db.execute(after).fetchall(), key=key), (sql, after)
            rows_seen += len(before_rows)
    assert rows_seen > 100


def test_off_on_the_second_attempt():
    assert normalize(TWO_LEVELS, schema=SCHEMA, dialect="postgres", types=TYPES).count("COUNT(*)") == 1
    with shape_fallback.disabled():
        assert normalize(TWO_LEVELS, schema=SCHEMA, dialect="postgres", types=TYPES).count("COUNT(*)") > 1
