"""The distinct join columns of a table outer-joined with itself (``kumo_sql_prover.self_outer_join_columns``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.self_outer_join_columns import distinct_self_outer_join_column

SCHEMA = {"t": ["a", "b", "c"], "u": ["a", "b", "c"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}


def spelled(sql):
    return sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres")


def read(sql):
    out = distinct_self_outer_join_column(sqlglot.parse_one(sql, read="postgres"))
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


READ = [
    ("SELECT DISTINCT e.b, d.b FROM t AS e FULL JOIN t AS d ON e.b = d.b", "SELECT DISTINCT e.b, e.b FROM t AS e"),
    ("SELECT DISTINCT d.b AS x, e.b AS y FROM t AS e FULL OUTER JOIN t AS d ON (d.b = e.b)", "SELECT DISTINCT e.b AS x, e.b AS y FROM t AS e"),
    ("SELECT DISTINCT e.b, d.b FROM t AS e LEFT JOIN t AS d ON e.b = d.b", "SELECT DISTINCT e.b, e.b FROM t AS e"),
    ("SELECT DISTINCT e.b, d.b FROM t AS e RIGHT JOIN t AS d ON e.b = d.b", "SELECT DISTINCT e.b, e.b FROM t AS e"),
    ("SELECT DISTINCT d.b FROM t AS e LEFT JOIN t AS d ON e.b = d.b", "SELECT DISTINCT e.b FROM t AS e"),
    ("SELECT DISTINCT d.b FROM t AS e RIGHT JOIN t AS d ON e.b = d.b", "SELECT DISTINCT e.b FROM t AS e"),
    ("SELECT DISTINCT t.b, d.b, d.b AS z FROM t FULL JOIN t AS d ON t.b = d.b", "SELECT DISTINCT t.b, t.b, t.b AS z FROM t"),
]


@pytest.mark.parametrize("sql, expected", READ)
def test_the_join_columns_are_read_from_one_side(sql, expected):
    assert read(sql) == spelled(expected)


@pytest.mark.parametrize("sql", [
    "SELECT DISTINCT e.b, d.b FROM t AS e JOIN t AS d ON e.b = d.b",  # no (NULL, NULL) row
    "SELECT DISTINCT e.b, d.b FROM t AS e CROSS JOIN t AS d",
    "SELECT e.b, d.b FROM t AS e FULL JOIN t AS d ON e.b = d.b",  # duplicates count
    "SELECT DISTINCT ON (e.b) e.b, d.b FROM t AS e FULL JOIN t AS d ON e.b = d.b",
    "SELECT DISTINCT e.b, d.b FROM t AS e FULL JOIN t AS d ON e.b = d.b AND e.c = d.c",  # two columns
    "SELECT DISTINCT e.b, d.b FROM t AS e FULL JOIN t AS d ON e.b = d.c",
    "SELECT DISTINCT e.b, d.b FROM t AS e FULL JOIN t AS d ON e.b = e.b",
    "SELECT DISTINCT e.b, d.b FROM t AS e FULL JOIN t AS d ON e.b >= d.b",
    "SELECT DISTINCT e.b, d.b FROM t AS e FULL JOIN t AS d ON e.b = d.b + 0",
    "SELECT DISTINCT e.b, d.b FROM t AS e FULL JOIN t AS d USING (b)",
    "SELECT DISTINCT e.b, d.b FROM t AS e FULL JOIN u AS d ON e.b = d.b",  # another table
    "SELECT DISTINCT e.b, d.b FROM t AS e FULL JOIN (SELECT * FROM t) AS d ON e.b = d.b",
    "SELECT DISTINCT e.b, d.c FROM t AS e FULL JOIN t AS d ON e.b = d.b",  # another column
    "SELECT DISTINCT e.b, d.b + 1 FROM t AS e FULL JOIN t AS d ON e.b = d.b",
    "SELECT DISTINCT e.b, b FROM t AS e FULL JOIN t AS d ON e.b = d.b",
    "SELECT DISTINCT * FROM t AS e FULL JOIN t AS d ON e.b = d.b",
    "SELECT DISTINCT e.b, d.b FROM t AS e FULL JOIN t AS d ON e.b = d.b WHERE d.b IS NULL",
    "SELECT DISTINCT e.b, d.b FROM t AS e FULL JOIN t AS d ON e.b = d.b ORDER BY 1 LIMIT 1",
    "SELECT DISTINCT e.b, d.b FROM t AS e FULL JOIN t AS d ON e.b = d.b FULL JOIN t AS f ON f.b = d.b",
    "SELECT DISTINCT e.b, d.b FROM t AS e, t AS d",
    "SELECT DISTINCT e.b, d.b FROM t AS e (b, a, c) FULL JOIN t AS d ON e.b = d.b",  # e.b is t.a
    "SELECT DISTINCT e.b, d.b FROM t AS e FULL JOIN t AS d (b, a, c) ON e.b = d.b",
    "SELECT DISTINCT e.b, COUNT(*) FROM t AS e FULL JOIN t AS d ON e.b = d.b GROUP BY e.b",
])
def test_other_joins_are_left_alone(sql):
    assert read(sql) is None


TARGETS = [
    ("SELECT DISTINCT e.b, d.b FROM t AS e FULL JOIN t AS d ON e.b = d.b", "SELECT DISTINCT b, b FROM t"),
    ("SELECT DISTINCT e.b, d.b FROM t AS e LEFT JOIN t AS d ON d.b = e.b", "SELECT DISTINCT b, b FROM t"),
    ("SELECT DISTINCT e.b, d.b FROM t AS e RIGHT JOIN t AS d ON e.b = d.b", "SELECT DISTINCT b, b FROM t"),
    ("SELECT e.b, d.b FROM t AS e FULL JOIN t AS d ON e.b = d.b GROUP BY d.b, e.b", "SELECT DISTINCT b, b FROM t"),
    ("SELECT d.b FROM t AS e RIGHT JOIN t AS d ON e.b = d.b GROUP BY d.b", "SELECT b FROM t GROUP BY b"),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_self_outer_joins_prove(left, right):
    assert proved(left, right)


ROWS = {
    "t": [(1, 5, None), (2, 5, 1), (3, None, 7), (4, 6, 2), (6, 9, 9)],
    "u": [(1, 5, 1), (2, 8, 1)],
}
NEAR_MISSES = [
    # an inner join has no row for a NULL b
    ("SELECT DISTINCT e.b, d.b FROM t AS e JOIN t AS d ON e.b = d.b", "SELECT DISTINCT b, b FROM t"),
    # a second join column: a row with a NULL c is unmatched and shows its b next to a padded NULL
    ("SELECT DISTINCT e.b, d.b FROM t AS e FULL JOIN t AS d ON e.b = d.b AND e.c = d.c", "SELECT DISTINCT b, b FROM t"),
    # two tables
    ("SELECT DISTINCT e.b, d.b FROM t AS e FULL JOIN u AS d ON e.b = d.b", "SELECT DISTINCT b, b FROM t"),
    # a column the join does not compare
    ("SELECT DISTINCT e.b, d.c FROM t AS e FULL JOIN t AS d ON e.b = d.b", "SELECT DISTINCT b, c FROM t"),
    # two different columns compared
    ("SELECT DISTINCT e.a, d.b FROM t AS e FULL JOIN t AS d ON e.a = d.b", "SELECT DISTINCT a, a FROM t"),
    # without DISTINCT a value held by two rows gives four pairs
    ("SELECT e.b, d.b FROM t AS e FULL JOIN t AS d ON e.b = d.b", "SELECT b, b FROM t"),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, ROWS)


def random_rows(rng):
    def table():
        rows = [tuple(rng.choice([None, None, 1, 2, 3]) for _ in range(3)) for _ in range(rng.randint(0, 6))]
        return rows + rows[:2]

    return {"t": table(), "u": table()}


@pytest.mark.parametrize("sql", [sql for sql, _ in READ])
def test_the_one_sided_reading_gives_the_same_rows(sql):
    out = read(sql)
    normalized = normalize(sql, schema=SCHEMA, types=TYPES, dialect="postgres")
    assert out is not None and "JOIN" not in out and "JOIN" not in normalized
    rng = random.Random(len(sql))
    for rows in [{}, ROWS, {"t": [(1, None, 1)]}, {"t": [(1, 2, 3)]}] + [random_rows(rng) for _ in range(16)]:
        assert not differ_on_duckdb(sql, out, rows)
        assert not differ_on_duckdb(sql, normalized, rows)
