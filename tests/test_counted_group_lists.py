"""A counted intersection under a select list that is not its keys (``kumo_sql_prover.counted_group_lists``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.counted_group_lists import list_keys_of_counted_groups

SCHEMA = {"r": ["a", "b"], "u": ["a", "w"], "d": ["k", "v"]}
TYPES = {table: {column: "BIGINT" for column in columns} for table, columns in SCHEMA.items()}

STACK = "(SELECT a, b FROM r GROUP BY a, b UNION ALL SELECT a, w FROM u GROUP BY a, w) AS t"
COUNTED = f"SELECT {{}} FROM {STACK} GROUP BY a, b HAVING COUNT(*) = 2"


def listed(sql):
    return list_keys_of_counted_groups(sqlglot.parse_one(sql, read="postgres")).sql(dialect="postgres")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="postgres").proven


def test_the_groups_get_a_select_of_their_own():
    keyed = f"(SELECT a, b FROM {STACK} GROUP BY a, b HAVING COUNT(*) = 2) AS t"
    assert listed(COUNTED.format("0 AS f")) == f"SELECT 0 AS f FROM {keyed}"
    assert listed(COUNTED.format("a")) == f"SELECT a FROM {keyed}"
    assert listed(COUNTED.format("t.b + 1 AS c, a, a")) == f"SELECT t.b + 1 AS c, a, a FROM {keyed}"
    assert listed(f"SELECT COUNT(*) FROM ({COUNTED.format('0 AS f')}) AS q") == f"SELECT COUNT(*) FROM (SELECT 0 AS f FROM {keyed}) AS q"


@pytest.mark.parametrize(
    "sql",
    [
        COUNTED.format("a, b"),  # the keys themselves: the existing rule's form
        COUNTED.format("t.b, t.a"),
        COUNTED.format("a, COUNT(*)"),
        COUNTED.format("a, MAX(b)"),
        COUNTED.format("a, (SELECT MAX(k) FROM d)"),
        COUNTED.format("a, ROW_NUMBER() OVER (ORDER BY a)"),
        COUNTED.format("a, MY_FUNCTION(b)"),
        COUNTED.format("0 AS f").replace("COUNT(*) = 2", "COUNT(*) >= 2"),
        COUNTED.format("0 AS f").replace("COUNT(*) = 2", "COUNT(a) = 2"),
        COUNTED.format("0 AS f").replace("COUNT(*) = 2", "COUNT(*) = 2 AND a > 0"),
        COUNTED.format("0 AS f").replace("GROUP BY a, b HAVING", "GROUP BY ROLLUP (a, b) HAVING"),
        COUNTED.format("0 AS f").replace("GROUP BY a, b HAVING", "GROUP BY a, a + b HAVING"),
        COUNTED.format("0 AS f").replace("GROUP BY a, b HAVING", "WHERE a > 0 GROUP BY a, b HAVING"),
        COUNTED.format("0 AS f") + " LIMIT 1",
        COUNTED.format("0 AS f") + " ORDER BY a",
        COUNTED.format("DISTINCT 0 AS f"),
        f"SELECT 0 AS f FROM {STACK} GROUP BY a, b",  # no HAVING
        "SELECT 0 AS f FROM r AS t GROUP BY a, b HAVING COUNT(*) = 2",  # a table, not a union
        "SELECT 0 AS f FROM (SELECT a, b FROM r INTERSECT SELECT a, w FROM u) AS t GROUP BY a, b HAVING COUNT(*) = 2",
        f"SELECT 0 AS f FROM {STACK}, d GROUP BY a, b HAVING COUNT(*) = 2",
        f"SELECT 0 AS f FROM {STACK}(a, b) GROUP BY a, b HAVING COUNT(*) = 2",
    ],
)
def test_what_is_left_alone(sql):
    assert listed(sql) == sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres"), sql


def test_pairs_it_proves():
    two = "SELECT COUNT(*) FROM (SELECT DISTINCT a, b FROM r INTERSECT SELECT DISTINCT a, w FROM u) AS q"
    assert proved(two, f"SELECT COUNT(*) FROM ({COUNTED.format('0 AS f')}) AS q")
    # the construction over itself, as a planner writes a three-way INTERSECT
    three = "SELECT COUNT(*) FROM (SELECT DISTINCT a FROM r INTERSECT SELECT DISTINCT a FROM u INTERSECT SELECT DISTINCT k FROM d) AS q"
    inner = (
        "SELECT a FROM (SELECT a, COUNT(*) AS c FROM (SELECT a FROM r GROUP BY a) AS t GROUP BY a UNION ALL "
        "SELECT a, COUNT(*) AS c FROM (SELECT a FROM u GROUP BY a) AS t2 GROUP BY a) AS t3 GROUP BY a HAVING COUNT(*) = 2"
    )
    stacked = (
        f"SELECT COUNT(*) FROM (SELECT 0 AS f FROM (SELECT a, COUNT(*) AS c FROM ({inner}) AS t4 GROUP BY a UNION ALL "
        "SELECT k, COUNT(*) AS c FROM (SELECT k FROM d GROUP BY k) AS t5 GROUP BY k) AS t6 GROUP BY a HAVING COUNT(*) = 2) AS q"
    )
    assert proved(three, stacked)


TWO = "SELECT COUNT(*) FROM (SELECT DISTINCT a, b FROM r INTERSECT SELECT DISTINCT a, w FROM u) AS q"
BOTH = f"SELECT COUNT(*) FROM ({COUNTED.format('0 AS f')}) AS q"


@pytest.mark.parametrize(
    "left, right",
    [
        (TWO, BOTH.replace("COUNT(*) = 2", "COUNT(*) = 1")),
        (TWO, BOTH.replace("SELECT a, b FROM r GROUP BY a, b", "SELECT a, b FROM r")),  # a row twice in one operand
        (TWO, BOTH.replace("UNION ALL", "UNION")),
        ("SELECT COUNT(*) FROM (SELECT DISTINCT a, b FROM r UNION SELECT DISTINCT a, w FROM u) AS q", BOTH),
    ],
)
def test_near_misses_stay_unproven(left, right):
    assert not proved(left, right)


def test_the_engine_agrees_on_random_tables():
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(20261118)
    value = lambda: rng.choice([None, *range(0, 3)])  # noqa: E731
    queries = [
        COUNTED.format("0 AS f"),
        COUNTED.format("a"),
        COUNTED.format("t.b + 1 AS c, a, a"),
        COUNTED.format("0 AS f").replace("COUNT(*) = 2", "COUNT(*) = 1"),
        f"SELECT COALESCE(a, -1) AS z FROM (SELECT a FROM r UNION ALL SELECT a FROM u) AS t GROUP BY a HAVING COUNT(*) = 3",
    ]
    readings = [listed(sql) for sql in queries]
    assert all(after != sql for sql, after in zip(queries, readings))
    rows_seen = 0
    for trial in range(30):
        db = duckdb.connect()
        for table, columns in SCHEMA.items():
            db.execute(f"CREATE TABLE {table} ({', '.join(c + ' BIGINT' for c in columns)})")
            rows = [tuple(value() for _ in columns) for _ in range(rng.choice([0, 1, 4, 9]) if trial % 5 else 6)]
            if rows:
                db.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in columns)})", rows)
        for sql, after in zip(queries, readings):
            key = lambda row: tuple((v is None, v) for v in row)  # noqa: E731
            before_rows = sorted(db.execute(sql).fetchall(), key=key)
            assert before_rows == sorted(db.execute(after).fetchall(), key=key), (sql, after)
            rows_seen += len(before_rows)
    assert rows_seen > 100


def test_off_on_the_second_attempt():
    sql = f"SELECT COUNT(*) FROM ({COUNTED.format('0 AS f')}) AS q"
    assert "HAVING" not in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)
    with shape_fallback.disabled():
        assert "HAVING" in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)
