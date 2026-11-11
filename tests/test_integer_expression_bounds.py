"""Strict comparisons of integer arithmetic with a whole number (``kumo_sql_prover.integer_expression_bounds``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.integer_expression_bounds import integer_expression_bounds

SCHEMA = {"a": ["id", "x", "y", "price", "day"], "b": ["id", "n"]}
TYPES = {
    "a": {"id": "BIGINT", "x": "INT", "y": "INT", "price": "DECIMAL(7,2)", "day": "DATE"},
    "b": {"id": "BIGINT", "n": "INT"},
}


def read(sql, types=TYPES):
    tree = sqlglot.parse_one(sql, read="postgres")
    out = integer_expression_bounds(tree, types, "postgres")
    return None if out is None else out.sql(dialect="postgres")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="postgres").proven


def test_strict_becomes_non_strict():
    assert read("SELECT a.id FROM a WHERE a.x - a.y > 30") == "SELECT a.id FROM a WHERE a.x - a.y >= 31"
    assert read("SELECT a.id FROM a WHERE a.x - a.y < 30") == "SELECT a.id FROM a WHERE a.x - a.y <= 29"
    assert read("SELECT a.id FROM a WHERE 30 < (a.x - a.y)") == "SELECT a.id FROM a WHERE 31 <= (a.x - a.y)"
    assert read("SELECT a.id FROM a WHERE 30 > a.x + a.y * 2") == "SELECT a.id FROM a WHERE 29 >= a.x + a.y * 2"
    assert read("SELECT a.id FROM a WHERE a.x - a.y > -1") == "SELECT a.id FROM a WHERE a.x - a.y >= 0"
    assert read("SELECT a.id FROM a WHERE a.x - a.y < 0") == "SELECT a.id FROM a WHERE a.x - a.y <= -1"
    assert read("SELECT SUM(CASE WHEN x - y > 30 AND x - y <= 60 THEN 1 ELSE 0 END) FROM a") == (
        "SELECT SUM(CASE WHEN x - y >= 31 AND x - y <= 60 THEN 1 ELSE 0 END) FROM a"
    )
    assert read("SELECT a.id FROM a JOIN b ON a.id = b.id WHERE a.x - b.n > 3") == "SELECT a.id FROM a JOIN b ON a.id = b.id WHERE a.x - b.n >= 4"
    assert read("SELECT a.id FROM a GROUP BY a.id HAVING SUM(a.x) - COUNT(*) > 3") == "SELECT a.id FROM a GROUP BY a.id HAVING SUM(a.x) - COUNT(*) >= 4"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a.id FROM a WHERE a.x > 30",  # a column: the solver keeps x > 30 and x <= 30 complementary
        "SELECT a.id FROM a WHERE a.price - a.x > 30",  # a decimal
        "SELECT a.id FROM a WHERE a.price * 2 > 30",
        "SELECT a.id FROM a WHERE a.x / a.y > 30",  # a fraction in most engines
        "SELECT a.id FROM a WHERE a.x - a.y > 30.5",
        "SELECT a.id FROM a WHERE a.x - a.y > a.id",
        "SELECT a.id FROM a WHERE a.x - a.y >= 30",
        "SELECT a.id FROM a WHERE a.x - a.y <> 30",
        "SELECT a.id FROM a WHERE a.day - 3 > 30",
        "SELECT a.id FROM a WHERE a.x - nope > 30",  # an unknown column
        "SELECT a.id FROM a, missing AS m WHERE a.x - m.z > 30",  # an unknown table
        "SELECT a.id FROM a WHERE a.x - ABS(a.y) > 30",
        "SELECT a.id FROM a WHERE a.x - (SELECT MAX(n) FROM b) > 30",
        "SELECT a.id FROM a JOIN b ON a.id = b.id WHERE x - id > 3",  # id is in both tables
        "SELECT a.id FROM a GROUP BY a.id HAVING COUNT(*) > 3",  # cast_rules' own
        "SELECT a.id FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.n - b.id > 3)",  # the subquery's own, read when it is the select
    ],
)
def test_what_is_left_alone(sql):
    assert read(sql) is None, sql


def test_nothing_without_types():
    assert read("SELECT a.id FROM a WHERE a.x - a.y > 30", None) is None
    assert read("SELECT a.id FROM a WHERE a.x - a.y > 30", {"a": {"id": "BIGINT"}}) is None


def test_pairs_it_proves():
    bucket = "SELECT a.id, SUM(CASE WHEN {} THEN 1 ELSE 0 END) FROM a GROUP BY a.id"
    assert proved(bucket.format("a.x - a.y > 30 AND a.x - a.y <= 60"), bucket.format("a.x - a.y BETWEEN 31 AND 60"))
    assert proved(bucket.format("a.x - a.y > 120"), bucket.format("a.x - a.y >= 121"))
    # the difference named in a derived table first
    assert proved(
        bucket.format("a.x - a.y > 30 AND a.x - a.y <= 60"),
        "SELECT s.id, SUM(CASE WHEN s.d BETWEEN 31 AND 60 THEN 1 ELSE 0 END) FROM (SELECT a.id, a.x - a.y AS d FROM a) AS s GROUP BY s.id",
    )
    assert proved("SELECT a.id FROM a WHERE a.x * 2 < 10", "SELECT a.id FROM a WHERE a.x * 2 <= 9")


def test_near_misses_stay_unproven():
    bucket = "SELECT a.id, SUM(CASE WHEN {} THEN 1 ELSE 0 END) FROM a GROUP BY a.id"
    assert not proved(bucket.format("a.x - a.y > 30"), bucket.format("a.x - a.y >= 30"))
    assert not proved(bucket.format("a.x - a.y > 30"), bucket.format("a.x - a.y >= 32"))
    # a decimal has values strictly between
    assert not proved(bucket.format("a.price - a.x > 30"), bucket.format("a.price - a.x >= 31"))
    assert not proved(bucket.format("a.x / 2.0 > 30"), bucket.format("a.x / 2.0 >= 31"))
    assert not proved("SELECT a.id FROM a WHERE a.x - a.y > 30", "SELECT a.id FROM a WHERE a.x - a.y >= 31 OR a.x IS NULL")


def test_the_engine_agrees_on_random_tables():
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(20261117)
    db = duckdb.connect()
    db.execute("CREATE TABLE a (id BIGINT, x INT, y INT, price DECIMAL(7,2), day DATE)")
    db.execute("CREATE TABLE b (id BIGINT, n INT)")
    value = lambda: rng.choice([None, *range(-40, 41)])  # noqa: E731
    db.executemany("INSERT INTO a VALUES (?, ?, ?, NULL, NULL)", [(rng.randint(0, 5), value(), value()) for _ in range(400)])
    db.executemany("INSERT INTO b VALUES (?, ?)", [(rng.randint(0, 5), value()) for _ in range(20)])
    for sql in (
        "SELECT a.id, a.x, a.y FROM a WHERE a.x - a.y > 30",
        "SELECT a.id, a.x, a.y FROM a WHERE a.x - a.y < -3 OR 30 < (a.x - a.y)",
        "SELECT a.id, a.x, a.y FROM a WHERE NOT 30 > a.x + a.y * 2",
        "SELECT a.id, SUM(CASE WHEN x - y > 3 AND x - y <= 60 THEN 1 ELSE 0 END), COUNT(*) FROM a GROUP BY a.id",
        "SELECT a.id, a.x, b.n FROM a JOIN b ON a.id = b.id WHERE a.x - b.n > 3",
        "SELECT a.id FROM a GROUP BY a.id HAVING SUM(a.x) - COUNT(*) > 3",
    ):
        after = read(sql)
        assert after is not None, sql
        key = lambda row: tuple((v is None, v) for v in row)  # noqa: E731
        rows = sorted(db.execute(sql).fetchall(), key=key)
        assert rows and rows == sorted(db.execute(after).fetchall(), key=key), sql


def test_off_on_the_second_attempt():
    sql = "SELECT a.id FROM a WHERE a.x - a.y > 30"
    assert ">= 31" in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)
    with shape_fallback.disabled():
        assert "> 30" in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)
