"""A conditional count of a value that has to be known where the condition holds (``kumo_sql_prover.counted_known_values``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.counted_known_values import count_known_values

SCHEMA = {"t": ["a", "b", "c"], "u": ["a", "b", "c"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}


def spelled(sql):
    return sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres")


def read(sql):
    out = count_known_values(sqlglot.parse_one(sql, read="postgres"))
    return None if out is None else out.sql(dialect="postgres")


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


GROUPED = "(SELECT a < 5 AS p, a > 10 AS e, a < 5 AS q FROM t GROUP BY a < 5, a > 10) AS d"
PLAIN = "(SELECT a + b < 5 AS p, 2 * a - b AS e, c FROM t) AS d"
KEYED = "(SELECT a < b AS p, -a * (b + 1) AS e, SUM(c) AS s FROM t GROUP BY a, b) AS d"

READ = [
    # where it is written
    ("SELECT COUNT(CASE WHEN a < 5 THEN a > 10 END) FROM t", "SELECT COUNT(CASE WHEN a < 5 THEN 1 END) FROM t"),
    ("SELECT c, COUNT(CASE WHEN a + 1 <> b THEN (a * b) - 3 END) AS n FROM t GROUP BY c", "SELECT c, COUNT(CASE WHEN a + 1 <> b THEN 1 END) AS n FROM t GROUP BY c"),
    ("SELECT COUNT(CASE WHEN t.a = u.a THEN t.a ELSE NULL END) FROM t, u", "SELECT COUNT(CASE WHEN t.a = u.a THEN 1 ELSE NULL END) FROM t, u"),
    ("SELECT COUNT(CASE WHEN 'x' >= a THEN a END), COUNT(CASE WHEN a < 5 THEN b END) FROM t", "SELECT COUNT(CASE WHEN 'x' >= a THEN 1 END), COUNT(CASE WHEN a < 5 THEN b END) FROM t"),
    # through the one derived table: grouped by the expressions, not grouped, grouped by their columns
    (f"SELECT COUNT(CASE WHEN p THEN e END), COUNT(CASE WHEN d.q THEN d.e END) FROM {GROUPED}", f"SELECT COUNT(CASE WHEN p THEN 1 END), COUNT(CASE WHEN d.q THEN 1 END) FROM {GROUPED}"),
    (f"SELECT c, COUNT(CASE WHEN p THEN e END) AS n FROM {PLAIN} GROUP BY c", f"SELECT c, COUNT(CASE WHEN p THEN 1 END) AS n FROM {PLAIN} GROUP BY c"),
    (f"SELECT COUNT(CASE WHEN d.p THEN d.e END) FROM {KEYED} WHERE d.s > 0", f"SELECT COUNT(CASE WHEN d.p THEN 1 END) FROM {KEYED} WHERE d.s > 0"),
]


@pytest.mark.parametrize("sql, expected", READ)
def test_known_values_are_counted_as_rows(sql, expected):
    assert read(sql) == spelled(expected)


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(CASE WHEN a < 5 THEN b > 10 END) FROM t",  # b may be NULL where a < 5
    "SELECT COUNT(CASE WHEN a < 5 THEN a + b END) FROM t",
    "SELECT COUNT(CASE WHEN t.a < 5 THEN u.a END) FROM t, u",  # another table's a
    "SELECT COUNT(CASE WHEN a < 5 OR b = 1 THEN a END) FROM t",  # TRUE with a NULL a
    "SELECT COUNT(CASE WHEN NOT a < 5 THEN a END) FROM t",  # true, not read
    "SELECT COUNT(CASE WHEN a IS NULL THEN a END) FROM t",
    "SELECT COUNT(CASE WHEN COALESCE(a, 0) < 5 THEN a END) FROM t",
    "SELECT COUNT(CASE WHEN a < 5 THEN NULLIF(a, 1) END) FROM t",
    "SELECT COUNT(CASE WHEN a < 5 THEN 10 / a END) FROM t",
    "SELECT COUNT(CASE WHEN a < 5 THEN a % 2 END) FROM t",
    "SELECT COUNT(CASE WHEN a < 5 THEN CASE WHEN a = 1 THEN 1 END END) FROM t",
    "SELECT COUNT(CASE WHEN a < 5 THEN (SELECT MAX(u.b) FROM u WHERE u.a = a) END) FROM t",
    "SELECT COUNT(CASE WHEN a < NULL THEN a END) FROM t",
    "SELECT COUNT(DISTINCT CASE WHEN a < 5 THEN a > 10 END) FROM t",
    "SELECT COUNT(CASE WHEN a < 5 THEN a > 10 ELSE b END) FROM t",
    "SELECT COUNT(CASE WHEN a < 5 THEN a > 10 WHEN b < 5 THEN b END) FROM t",
    "SELECT COUNT(CASE a WHEN 5 THEN a END) FROM t",
    "SELECT SUM(CASE WHEN a < 5 THEN a END) FROM t",
    "SELECT COUNT(CASE WHEN a < 5 THEN 1 END) FROM t",  # nothing to do
    # through a derived table
    "SELECT COUNT(CASE WHEN d.x < 5 THEN d.e END) FROM (SELECT a AS x, b > 10 AS e FROM t) AS d",
    "SELECT COUNT(CASE WHEN p THEN e END) FROM (SELECT a < 5 AS p, b > 10 AS e FROM t) AS d",
    "SELECT COUNT(CASE WHEN p THEN e END) FROM (SELECT a < 5 AS p, a > 10 AS e FROM t GROUP BY ROLLUP ((a < 5), (a > 10))) AS d",
    "SELECT COUNT(CASE WHEN p THEN e END) FROM (SELECT a < 5 AS p, a > 10 AS e FROM t GROUP BY GROUPING SETS ((a < 5), (a < 5, a > 10))) AS d",
    "SELECT COUNT(CASE WHEN p THEN e END) FROM (SELECT a < 5 AS p, MAX(a) AS e FROM t GROUP BY a < 5) AS d",
    "SELECT COUNT(CASE WHEN p THEN e END) FROM (SELECT a < 5 AS p, a > 10 AS e FROM t UNION ALL SELECT a < 5, b > 10 FROM u) AS d",
    "SELECT COUNT(CASE WHEN p THEN e END) FROM (SELECT a < 5, a > 10 FROM t) AS d (e, p)",
    "SELECT COUNT(CASE WHEN p THEN e END) FROM (SELECT a < 5 AS p, a > 10 AS e FROM t) AS d LEFT JOIN (SELECT b > 1 AS e2 FROM u) AS x ON TRUE",
    "SELECT COUNT(CASE WHEN p THEN x.e END) FROM (SELECT a < 5 AS p, a > 10 AS e FROM t) AS d",  # not this table's e
    "SELECT COUNT(CASE WHEN p THEN e END) FROM (SELECT a < 5 AS p, a > 10 AS e, b AS e FROM t) AS d",
])
def test_other_counts_are_left_alone(sql):
    assert read(sql) is None


TARGETS = [
    ("SELECT COUNT(CASE WHEN a < 5 THEN a > 10 END) FROM t", "SELECT COUNT(*) FILTER (WHERE a < 5) FROM t"),
    ("SELECT c, COUNT(a * 2) FILTER (WHERE a <> b) AS n FROM t GROUP BY c", "SELECT c, COUNT(*) FILTER (WHERE a <> b) AS n FROM t GROUP BY c"),
    (
        "SELECT COUNT(e) FILTER (WHERE q), COUNT(p) FILTER (WHERE r) FROM (SELECT a < 5 AS p, a > 10 AS e, a < 5 AS q, a > 10 AS r FROM t GROUP BY a < 5, a > 10) AS d",
        "SELECT COUNT(*) FILTER (WHERE q), COUNT(*) FILTER (WHERE r) FROM (SELECT a < 5 AS q, a > 10 AS r FROM t GROUP BY a < 5, a > 10) AS d",
    ),
    (
        "SELECT COUNT(e) FILTER (WHERE p) FROM (SELECT a < b AS p, a - b AS e FROM t) AS d",
        "SELECT COUNT(*) FILTER (WHERE p) FROM (SELECT a < b AS p FROM t) AS d",
    ),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_known_value_counts_prove(left, right):
    assert proved(left, right)


ROWS = {
    "t": [(1, None, 1), (1, 2, 1), (2, 20, 1), (None, 1, 1), (None, None, 2), (20, 3, 2), (0, 7, 2)],
    "u": [(None, 1, 1), (1, None, 1)],
}
NEAR_MISSES = [
    # the counted value reads a column the condition does not
    ("SELECT COUNT(CASE WHEN a < 5 THEN b > 10 END) FROM t", "SELECT COUNT(CASE WHEN a < 5 THEN 1 END) FROM t"),
    ("SELECT COUNT(CASE WHEN t.a < 5 THEN u.a END) FROM t, u", "SELECT COUNT(CASE WHEN t.a < 5 THEN 1 END) FROM t, u"),
    # a condition that can be TRUE with a NULL column
    ("SELECT COUNT(CASE WHEN a < 5 OR b = 1 THEN a END) FROM t", "SELECT COUNT(CASE WHEN a < 5 OR b = 1 THEN 1 END) FROM t"),
    ("SELECT COUNT(CASE WHEN a IS NULL THEN a END) FROM t", "SELECT COUNT(CASE WHEN a IS NULL THEN 1 END) FROM t"),
    ("SELECT COUNT(CASE WHEN COALESCE(a, 0) < 5 THEN a END) FROM t", "SELECT COUNT(CASE WHEN COALESCE(a, 0) < 5 THEN 1 END) FROM t"),
    # a value that can be NULL over known columns
    ("SELECT COUNT(CASE WHEN a < 5 THEN NULLIF(a, 1) END) FROM t", "SELECT COUNT(CASE WHEN a < 5 THEN 1 END) FROM t"),
    # distinct values are not rows
    ("SELECT COUNT(DISTINCT CASE WHEN a < 5 THEN a > 10 END) FROM t", "SELECT COUNT(CASE WHEN a < 5 THEN 1 END) FROM t"),
    # through a derived table: another column, and a key that a grouping set leaves out
    (
        "SELECT COUNT(e) FILTER (WHERE p) FROM (SELECT a < 5 AS p, b > 10 AS e FROM t) AS d",
        "SELECT COUNT(*) FILTER (WHERE p) FROM (SELECT a < 5 AS p, b > 10 AS e FROM t) AS d",
    ),
    (
        "SELECT COUNT(e) FILTER (WHERE p) FROM (SELECT a < 5 AS p, a > 10 AS e FROM t GROUP BY GROUPING SETS ((a < 5), (a < 5, a > 10))) AS d",
        "SELECT COUNT(*) FILTER (WHERE p) FROM (SELECT a < 5 AS p, a > 10 AS e FROM t GROUP BY GROUPING SETS ((a < 5), (a < 5, a > 10))) AS d",
    ),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, ROWS)


def random_rows(rng):
    def table():
        rows = [tuple(rng.choice([None, 0, 1, 2, 4, 5, 11, 20]) for _ in range(3)) for _ in range(rng.randint(0, 7))]
        return rows + rows[:2]

    return {"t": table(), "u": table()}


@pytest.mark.parametrize("sql", [sql for sql, _ in READ if "'x'" not in sql])
def test_the_row_count_gives_the_same_rows(sql):
    out = read(sql)
    normalized = normalize(sql, schema=SCHEMA, types=TYPES, dialect="postgres")
    assert out is not None
    rng = random.Random(len(sql))
    for rows in [{}, ROWS] + [random_rows(rng) for _ in range(14)]:
        assert not differ_on_duckdb(sql, out, rows)
        assert not differ_on_duckdb(sql, normalized, rows)
