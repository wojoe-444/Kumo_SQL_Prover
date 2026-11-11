"""A pre-grouped source under a select that only takes ``MIN``/``MAX`` (``kumo_sql_prover.extreme_group_sources``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.extreme_group_sources import read_extreme_group_source

SCHEMA = {"s": ["k", "j", "x", "y"], "o": ["k", "y", "z"], "d": ["j", "w"]}
TYPES = {table: {column: "BIGINT" for column in columns} for table, columns in SCHEMA.items()}

GROUPED = "(SELECT k, MIN(x) AS m FROM s GROUP BY k) AS g"
FLAT = "(SELECT k, x AS m FROM s) AS g"


def read(sql):
    tree = sqlglot.parse_one(sql, read="postgres")
    out = read_extreme_group_source(tree)
    return None if out is None else out.sql(dialect="postgres")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="postgres").proven


def test_the_source_loses_its_group_by():
    assert read(f"SELECT MIN(g.m), MAX(o.y) FROM {GROUPED}, o WHERE g.k = o.k") == f"SELECT MIN(g.m), MAX(o.y) FROM {FLAT}, o WHERE g.k = o.k"
    assert read(f"SELECT o.z, MIN(g.m) FROM o JOIN {GROUPED} ON g.k = o.k GROUP BY o.z HAVING MIN(g.m) > 3") == (
        f"SELECT o.z, MIN(g.m) FROM o JOIN {FLAT} ON g.k = o.k GROUP BY o.z HAVING MIN(g.m) > 3"
    )
    # a select that only groups, and a key-only source
    assert read("SELECT g.k FROM (SELECT k FROM s WHERE x > 1 GROUP BY k) AS g, o WHERE g.k = o.k GROUP BY g.k") == (
        "SELECT g.k FROM (SELECT k FROM s WHERE x > 1) AS g, o WHERE g.k = o.k GROUP BY g.k"
    )
    assert read("SELECT MAX(g.b), MIN(g.a) FROM (SELECT k, j, MIN(x + y) AS a, MAX(y) AS b FROM s GROUP BY k, j) AS g") == (
        "SELECT MAX(g.b), MIN(g.a) FROM (SELECT k, j, x + y AS a, y AS b FROM s) AS g"
    )


def test_levels_come_apart_from_the_inside():
    nested = (
        "SELECT MIN(t1.m), MIN(o.z) FROM (SELECT t0.k, MIN(t0.m) AS m FROM (SELECT k, j, MIN(x) AS m FROM s GROUP BY k, j) AS t0, "
        "(SELECT j FROM d WHERE w = 1 GROUP BY j) AS t3 WHERE t0.j = t3.j GROUP BY t0.k) AS t1, o WHERE t1.k = o.k"
    )
    assert read(nested) == (
        "SELECT MIN(t1.m), MIN(o.z) FROM (SELECT t0.k, t0.m AS m FROM (SELECT k, j, x AS m FROM s) AS t0, "
        "(SELECT j FROM d WHERE w = 1) AS t3 WHERE t0.j = t3.j) AS t1, o WHERE t1.k = o.k"
    )
    assert proved(nested, "SELECT MIN(s.x), MIN(o.z) FROM s, d, o WHERE s.j = d.j AND d.w = 1 AND s.k = o.k")


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT COUNT(*) FROM {GROUPED}, o WHERE g.k = o.k",  # counts the repeats
        f"SELECT MIN(g.m), SUM(o.y) FROM {GROUPED}, o WHERE g.k = o.k",
        f"SELECT MIN(g.m), COUNT(DISTINCT o.y) FROM {GROUPED}, o WHERE g.k = o.k",  # true, but not read
        f"SELECT MIN(g.m), MY_AGG(o.y) FROM {GROUPED}, o WHERE g.k = o.k GROUP BY o.z",  # an unknown name may count
        f"SELECT g.k, g.m FROM {GROUPED}, o WHERE g.k = o.k",  # no grouping: the repeats would be returned
        f"SELECT MAX(g.m) FROM {GROUPED}",  # the greatest minimum is not the greatest value
        f"SELECT MIN(g.m + 1) FROM {GROUPED}",
        f"SELECT MIN(o.y) FROM {GROUPED}, o WHERE g.k = o.k AND g.m > 3",  # a filter on the minimum
        f"SELECT g.m FROM {GROUPED} GROUP BY g.m",
        f"SELECT MIN(o.y) FROM {GROUPED}, o WHERE g.k = o.k AND EXISTS (SELECT 1 FROM d WHERE d.w = g.m)",
        f"SELECT MIN(o.y) FROM {GROUPED}, o WHERE g.k = o.k AND o.z = (SELECT MIN(g.m) FROM d)",
        f"SELECT MIN(m) FROM {GROUPED}",  # a bare name
        f"SELECT MIN(g.m), MIN(y) FROM {GROUPED}, o WHERE g.k = o.k AND k = 1",
        f"SELECT DISTINCT MIN(g.m) FROM {GROUPED}, o WHERE g.k = o.k",
        f"SELECT MIN(g.m) FROM {GROUPED}, o WHERE g.k = o.k LIMIT 1",
        f"SELECT MIN(g.m), MIN(o.y) OVER () FROM {GROUPED}, o WHERE g.k = o.k GROUP BY o.y",
        f"SELECT MIN(g.m) FROM o LEFT JOIN {GROUPED} ON g.k = o.k",  # padded with NULLs: still true, not read
        f"SELECT MIN(g.m) FROM {GROUPED} RIGHT JOIN o ON g.k = o.k",
        f"SELECT MIN(g.m) FROM {GROUPED} FULL JOIN o ON g.k = o.k",
        f"SELECT o.z, MIN(g.m) FROM {GROUPED}, o WHERE g.k = o.k GROUP BY ROLLUP (o.z)",
        f"SELECT MIN(g.m) FROM {GROUPED}, {GROUPED}",  # two sources of one name
        "SELECT MIN(g.m) FROM (SELECT k, MIN(x) AS m FROM s GROUP BY k HAVING COUNT(*) > 1) AS g",
        "SELECT MIN(g.m) FROM (SELECT MIN(x) AS m FROM s) AS g",  # no GROUP BY: a row for no input
        "SELECT MIN(g.m) FROM (SELECT k, MIN(x) AS m FROM s GROUP BY 1) AS g",
        "SELECT MIN(g.m) FROM (SELECT k, SUM(x) AS m FROM s GROUP BY k) AS g",
        "SELECT MIN(g.m) FROM (SELECT k, MIN(x) + 1 AS m FROM s GROUP BY k) AS g",
        "SELECT MIN(g.m) FROM (SELECT k, MIN(DISTINCT x) AS m, COUNT(*) AS c FROM s GROUP BY k) AS g",
        "SELECT MIN(g.m) FROM (SELECT k, MIN(x) AS m FROM s GROUP BY k LIMIT 3) AS g",
        "SELECT MIN(g.m) FROM (SELECT DISTINCT k, MIN(x) AS m FROM s GROUP BY k) AS g",
        "SELECT MIN(g.m) FROM (SELECT k + 1 AS k, MIN(x) AS m FROM s GROUP BY k) AS g",  # not the key's text
        "SELECT MIN(g.m) FROM (SELECT k, MIN(x) AS m FROM s GROUP BY k) AS g(a, m)",
        "SELECT MIN(g.m) FROM (SELECT k, MIN(x) AS m, ROW_NUMBER() OVER (ORDER BY k) AS r FROM s GROUP BY k) AS g",
        "SELECT g.* FROM (SELECT k FROM s GROUP BY k) AS g GROUP BY g.k",
    ],
)
def test_what_is_left_alone(sql):
    assert read(sql) is None, sql


def test_pairs_it_proves():
    flat = "SELECT MIN(s.x), MIN(o.z) FROM s, o WHERE s.k = o.k AND o.y = 7"
    assert proved(flat, f"SELECT MIN(g.m), MIN(o.z) FROM {GROUPED}, o WHERE g.k = o.k AND o.y = 7")
    # both sides pre-grouped, as Calcite's aggregate-join transpose writes them
    assert proved(
        flat,
        "SELECT MIN(g.m), MIN(h.z) FROM (SELECT k, MIN(x) AS m FROM s GROUP BY k) AS g, "
        "(SELECT k, MIN(z) AS z FROM o WHERE y = 7 GROUP BY k) AS h WHERE g.k = h.k",
    )
    assert proved(
        "SELECT o.z, MAX(s.y) FROM s JOIN o ON s.k = o.k WHERE s.x > 0 GROUP BY o.z",
        "SELECT o.z, MAX(g.b) FROM (SELECT k, MAX(y) AS b FROM s WHERE x > 0 GROUP BY k) AS g JOIN o ON g.k = o.k GROUP BY o.z",
    )


def test_near_misses_stay_unproven():
    assert not proved("SELECT COUNT(*) FROM s, o WHERE s.k = o.k", f"SELECT COUNT(*) FROM {GROUPED}, o WHERE g.k = o.k")
    assert not proved("SELECT SUM(o.y) FROM s, o WHERE s.k = o.k", f"SELECT SUM(o.y) FROM {GROUPED}, o WHERE g.k = o.k")
    assert not proved("SELECT MAX(s.x) FROM s, o WHERE s.k = o.k", f"SELECT MAX(g.m) FROM {GROUPED}, o WHERE g.k = o.k")
    assert not proved("SELECT s.k, s.x FROM s, o WHERE s.k = o.k", f"SELECT g.k, g.m FROM {GROUPED}, o WHERE g.k = o.k")
    assert not proved("SELECT MIN(o.y) FROM s, o WHERE s.k = o.k AND s.x > 3", f"SELECT MIN(o.y) FROM {GROUPED}, o WHERE g.k = o.k AND g.m > 3")
    assert not proved("SELECT MIN(s.x) FROM s", "SELECT MIN(g.m) FROM (SELECT k, MIN(x) AS m FROM s GROUP BY k HAVING COUNT(*) > 1) AS g")
    assert not proved("SELECT MIN(s.x) FROM s LIMIT 0", f"SELECT MIN(g.m) FROM {GROUPED}")
    assert not proved("SELECT MIN(s.x) FROM s, o WHERE s.k = o.k", f"SELECT MIN(g.m) FROM {GROUPED}, o WHERE g.k = o.y")


def test_the_engine_agrees_on_random_tables():
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(20261114)
    value = lambda: rng.choice([None, *range(-3, 6)])  # noqa: E731
    queries = [
        f"SELECT MIN(g.m), MAX(o.y) FROM {GROUPED}, o WHERE g.k = o.k",
        f"SELECT o.z, MIN(g.m), MIN(o.y) FROM o JOIN {GROUPED} ON g.k = o.k GROUP BY o.z HAVING MIN(g.m) > -2",
        "SELECT g.k FROM (SELECT k FROM s WHERE x > 1 GROUP BY k) AS g, o WHERE g.k = o.k GROUP BY g.k",
        "SELECT MAX(g.b), MIN(g.a) FROM (SELECT k, j, MIN(x + y) AS a, MAX(y) AS b FROM s GROUP BY k, j) AS g",
        "SELECT g.k, MIN(g.m), MAX(o.z) FROM (SELECT k, MIN(x) AS m FROM s GROUP BY k, j) AS g LEFT JOIN o ON g.k = o.k AND o.y > 0 GROUP BY g.k",
        "SELECT MIN(t1.m), MIN(o.z) FROM (SELECT t0.k, MIN(t0.m) AS m FROM (SELECT k, j, MIN(x) AS m FROM s GROUP BY k, j) AS t0, "
        "(SELECT j FROM d WHERE w = 1 GROUP BY j) AS t3 WHERE t0.j = t3.j GROUP BY t0.k) AS t1, o WHERE t1.k = o.k",
        f"SELECT MIN(g.m) FROM {GROUPED} WHERE EXISTS (SELECT 1 FROM o WHERE o.k = g.k AND o.y > 1)",
    ]
    read_any = 0
    for trial in range(40):
        db = duckdb.connect()
        for table, columns in SCHEMA.items():
            db.execute(f"CREATE TABLE {table} ({', '.join(c + ' BIGINT' for c in columns)})")
            size = rng.choice([0, 1, 2, 6, 12]) if trial % 4 else 0 if table == "s" else 5
            rows = [tuple(value() for _ in columns) for _ in range(size)]
            if rows:
                db.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in columns)})", rows)
        for sql in queries:
            after = read(sql)
            assert after is not None, sql
            key = lambda row: tuple((v is None, v) for v in row)  # noqa: E731
            before_rows = sorted(db.execute(sql).fetchall(), key=key)
            assert before_rows == sorted(db.execute(after).fetchall(), key=key), (sql, after)
            read_any += bool(before_rows)
    assert read_any > 100


def test_off_on_the_second_attempt():
    # one level is ``eager_aggregation``'s; a level whose select has a subquery is this rule's alone
    sql = f"SELECT MIN(g.m) FROM {GROUPED} WHERE EXISTS (SELECT 1 FROM o WHERE o.k = g.k AND o.y > 1)"
    assert "GROUP BY" not in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES).upper()
    with shape_fallback.disabled():
        assert "GROUP BY" in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES).upper()
