"""A GROUP BY key that can only name an output of its select (``kumo_sql_prover.grouped_output_names``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.grouped_output_names import group_by_named_outputs

SCHEMA = {"t": ["a", "b", "c"], "u": ["a", "b", "c"], "v": ["k", "x"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}


def spelled(sql):
    return sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres")


def read(sql, schema=SCHEMA):
    return group_by_named_outputs(sqlglot.parse_one(sql, read="postgres"), schema).sql(dialect="postgres")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


def rows_on_duckdb(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name, columns in SCHEMA.items():
        db.execute(f"CREATE TABLE {name} ({', '.join(column + ' BIGINT' for column in columns)})")
        insert_rows(db, name, rows.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def differ_on_duckdb(left, right, rows):
    a, b = rows_on_duckdb([left, right], rows)
    return a != b


READ = [
    ("SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY big", "SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY a > 10"),
    ("SELECT a + b AS s, c, SUM(b) AS m FROM t GROUP BY s, c", "SELECT a + b AS s, c, SUM(b) AS m FROM t GROUP BY a + b, c"),
    ("SELECT x * 2 AS a, k AS b, COUNT(*) AS n FROM v GROUP BY a, b", "SELECT x * 2 AS a, k AS b, COUNT(*) AS n FROM v GROUP BY x * 2, k"),  # v has no a, b
    ("SELECT t.a + v.x AS s, COUNT(*) AS n FROM t JOIN v ON t.b = v.k GROUP BY S", "SELECT t.a + v.x AS s, COUNT(*) AS n FROM t JOIN v ON t.b = v.k GROUP BY t.a + v.x"),
    (
        "SELECT d.big, d.n FROM (SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY big) AS d WHERE d.n > 1",
        "SELECT d.big, d.n FROM (SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY a > 10) AS d WHERE d.n > 1",
    ),
    (
        "SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY big UNION ALL SELECT x > 10 AS big, COUNT(*) AS n FROM v GROUP BY big",
        "SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY a > 10 UNION ALL SELECT x > 10 AS big, COUNT(*) AS n FROM v GROUP BY x > 10",
    ),
]


@pytest.mark.parametrize("sql, expected", READ)
def test_a_key_that_names_an_output_is_its_expression(sql, expected):
    assert read(sql) == spelled(expected)


@pytest.mark.parametrize("sql", [
    "SELECT b * 0 AS b, COUNT(*) AS n FROM t GROUP BY b",  # the input column b (PostgreSQL)
    "SELECT v.x * 0 AS c, COUNT(*) AS n FROM v JOIN t ON t.a = v.k GROUP BY c",  # t.c
    "SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY t.big",
    "SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY big + 1",
    "SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY ROLLUP (big)",
    "SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY GROUPING SETS ((big), ())",
    "SELECT a > 10 AS big, b AS big, COUNT(*) AS n FROM t GROUP BY big, b",  # two outputs of that name
    "SELECT COUNT(*) AS n FROM t GROUP BY n",
    "SELECT SUM(a) OVER () AS w FROM t GROUP BY w",
    "SELECT (SELECT MAX(x) FROM v) AS m, COUNT(*) AS n FROM t GROUP BY m",
    "SELECT a + 1 AS s, s * 2 AS big FROM t GROUP BY big",  # reads another output name
    "SELECT d.a > 10 AS big, COUNT(*) AS n FROM (SELECT a FROM t) AS d GROUP BY big",  # a derived source
    "SELECT a > 10 AS big, COUNT(*) AS n FROM t, (SELECT x FROM v) AS d GROUP BY big",
    "SELECT a > 10 AS big, COUNT(*) AS n FROM t AS r (p, q, big) GROUP BY big",
    "SELECT a > 10 AS big, COUNT(*) AS n FROM unknown_table GROUP BY big",
    "WITH t AS (SELECT x AS a, k AS big FROM v) SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY big",
    # inside an expression subquery the name could be a column of the enclosing query
    "SELECT o.a FROM t AS o WHERE o.b IN (SELECT v.x * 0 AS c FROM v GROUP BY c)",
    "SELECT o.a FROM t AS o WHERE EXISTS (SELECT v.x * 0 AS c FROM v GROUP BY c HAVING COUNT(*) > o.b)",
    "SELECT o.a, (SELECT MAX(d.c) FROM (SELECT v.x * 0 AS c FROM v GROUP BY c) AS d) AS m FROM t AS o",
    "SELECT o.a FROM t AS o JOIN v ON o.a = (SELECT MAX(g.c) FROM (SELECT w.x * 0 AS c FROM v AS w GROUP BY c) AS g)",
    "SELECT o.a FROM t AS o, LATERAL (SELECT v.x * 0 AS c FROM v GROUP BY c) AS d",
])
def test_other_keys_are_left_alone(sql):
    assert read(sql) == spelled(sql)


def test_nothing_is_read_without_a_schema():
    sql = "SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY big"
    assert read(sql, None) == spelled(sql)
    assert read(sql, {}) == spelled(sql)
    assert read(sql, {"u": ["a", "big"]}) == spelled(sql)
    assert read(sql, {"t": ["a", "Big"]}) == spelled(sql)


TARGETS = [
    ("SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY big", "SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY a > 10"),
    ("SELECT a + b AS s, c, SUM(b) AS m FROM t GROUP BY s, c", "SELECT a + b AS s, c, SUM(b) AS m FROM t GROUP BY c, a + b"),
    (
        "SELECT COUNT(*) FILTER (WHERE small), COUNT(*) FILTER (WHERE big) FROM (SELECT a > 10 AS big, a < 5 AS small FROM t GROUP BY big, small) AS d",
        "SELECT COUNT(*) FILTER (WHERE q), COUNT(*) FILTER (WHERE p) FROM (SELECT a > 10 AS p, a < 5 AS q FROM t GROUP BY a > 10, a < 5) AS d",
    ),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_named_keys_prove(left, right):
    assert proved(left, right)


ROWS = {
    "t": [(1, 1, 1), (1, 2, 1), (20, 2, 1), (None, 3, 2), (11, None, 2), (11, None, 2)],
    "v": [(1, 1), (2, 2), (None, 3)],
}
NEAR_MISSES = [
    # the name is an input column: PostgreSQL (and DuckDB) group by the column, one row per b
    ("SELECT b * 0 AS b, COUNT(*) AS n FROM t GROUP BY b", "SELECT b * 0 AS b, COUNT(*) AS n FROM t GROUP BY b * 0"),
    # the output's expression is not its column
    ("SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY big", "SELECT a > 10 AS big, COUNT(*) AS n FROM t GROUP BY a"),
    ("SELECT a + b AS s, COUNT(*) AS n FROM t GROUP BY s", "SELECT a + b AS s, COUNT(*) AS n FROM t GROUP BY a, b"),
    # another output of the same select
    ("SELECT a > 10 AS big, c > 1 AS late, COUNT(*) AS n FROM t GROUP BY big, late", "SELECT a > 10 AS big, MAX(c) > 1 AS late, COUNT(*) AS n FROM t GROUP BY a > 10"),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, ROWS)


def random_rows(rng):
    def table(width):
        rows = [tuple(rng.choice([None, 0, 1, 2, 5, 11, 20]) for _ in range(width)) for _ in range(rng.randint(0, 7))]
        return rows + rows[:2]

    return {"t": table(3), "u": table(3), "v": table(2)}


@pytest.mark.parametrize("sql", [sql for sql, _ in READ])
def test_the_expression_key_gives_the_same_rows(sql):
    out = read(sql)
    normalized = normalize(sql, schema=SCHEMA, types=TYPES, dialect="postgres")
    assert out != spelled(sql)
    rng = random.Random(len(sql))
    for rows in [{}, ROWS] + [random_rows(rng) for _ in range(14)]:
        assert not differ_on_duckdb(sql, out, rows)
        assert not differ_on_duckdb(sql, normalized, rows)
