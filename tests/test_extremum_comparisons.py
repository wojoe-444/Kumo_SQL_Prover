"""A WHERE comparison with a scalar MIN or MAX subquery is an EXISTS (``kumo_sql_prover.extremum_comparisons``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.extremum_comparisons import compare_with_extremum_as_exists

SCHEMA = {"t": ["a", "b", "c"], "u": ["a", "b", "c"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}


def read(sql, schema=None):
    out = compare_with_extremum_as_exists(sqlglot.parse_one(sql, read="postgres"), schema)
    return out.sql(dialect="postgres") if out is not None else None


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


SUB = "FROM u WHERE u.c = t.c"


def test_the_four_comparisons_become_exists():
    assert read(f"SELECT t.a FROM t WHERE t.b > (SELECT MIN(u.b) {SUB})") == f"SELECT t.a FROM t WHERE EXISTS(SELECT 1 {SUB} AND t.b > u.b)"
    assert read(f"SELECT t.a FROM t WHERE t.b >= (SELECT MIN(u.b) {SUB})") == f"SELECT t.a FROM t WHERE EXISTS(SELECT 1 {SUB} AND t.b >= u.b)"
    assert read(f"SELECT t.a FROM t WHERE t.b < (SELECT MAX(u.b) {SUB})") == f"SELECT t.a FROM t WHERE EXISTS(SELECT 1 {SUB} AND t.b < u.b)"
    assert read(f"SELECT t.a FROM t WHERE t.b <= (SELECT MAX(u.b) {SUB})") == f"SELECT t.a FROM t WHERE EXISTS(SELECT 1 {SUB} AND t.b <= u.b)"
    # mirrored, DISTINCT inside, an expression on each side, other conjuncts, no WHERE inside, a join inside
    assert read(f"SELECT t.a FROM t WHERE (SELECT MIN(DISTINCT u.b) {SUB}) < t.b AND t.a = 1") == (
        f"SELECT t.a FROM t WHERE EXISTS(SELECT 1 {SUB} AND t.b > u.b) AND t.a = 1"
    )
    assert read(f"SELECT t.a FROM t WHERE t.b + 1 <= (SELECT MAX(u.b * 2) AS m {SUB} OR u.a = 1)") == (
        "SELECT t.a FROM t WHERE EXISTS(SELECT 1 FROM u WHERE (u.c = t.c OR u.a = 1) AND (t.b + 1) <= (u.b * 2))"
    )
    assert read("SELECT t.a FROM t WHERE 5 > (SELECT MIN(u.b) FROM u)") == "SELECT t.a FROM t WHERE EXISTS(SELECT 1 FROM u WHERE 5 > u.b)"
    assert read("SELECT t.a FROM t WHERE t.b > (SELECT MIN(u.b) FROM u LEFT JOIN t AS v ON v.a = u.a WHERE v.c = t.c)") == (
        "SELECT t.a FROM t WHERE EXISTS(SELECT 1 FROM u LEFT JOIN t AS v ON v.a = u.a WHERE v.c = t.c AND t.b > u.b)"
    )
    # a bare outer column is qualified from the schema of the select's only table
    assert read("SELECT a FROM t WHERE b > (SELECT MIN(u.b) FROM u WHERE u.c = t.c)", SCHEMA) == (
        "SELECT a FROM t WHERE EXISTS(SELECT 1 FROM u WHERE u.c = t.c AND t.b > u.b)"
    )


@pytest.mark.parametrize("sql", [
    f"SELECT t.a FROM t WHERE t.b < (SELECT MIN(u.b) {SUB})",  # ALL, not ANY
    f"SELECT t.a FROM t WHERE t.b > (SELECT MAX(u.b) {SUB})",
    f"SELECT t.a FROM t WHERE t.b = (SELECT MIN(u.b) {SUB})",
    f"SELECT t.a FROM t WHERE t.b <> (SELECT MIN(u.b) {SUB})",
    f"SELECT t.a FROM t WHERE NOT t.b > (SELECT MIN(u.b) {SUB})",
    f"SELECT t.a FROM t WHERE t.b > (SELECT MIN(u.b) {SUB}) OR t.a = 1",
    f"SELECT t.a, t.b > (SELECT MIN(u.b) {SUB}) AS f FROM t",
    f"SELECT t.a FROM t WHERE t.b > (SELECT SUM(u.b) {SUB})",
    f"SELECT t.a FROM t WHERE t.b > (SELECT MIN(u.b) + 1 {SUB})",
    f"SELECT t.a FROM t WHERE t.b > (SELECT MIN(u.b) {SUB} GROUP BY u.a LIMIT 1)",
    f"SELECT t.a FROM t WHERE t.b > (SELECT MIN(u.b) {SUB} HAVING COUNT(*) > 1)",
    f"SELECT t.a FROM t WHERE t.b > (SELECT MIN(u.b) FILTER (WHERE u.a > 1) {SUB})",
    f"SELECT t.a FROM t WHERE t.b > (SELECT MIN(u.b) {SUB} LIMIT 1)",
    f"SELECT t.a FROM t WHERE t.b > (SELECT MIN(u.b) {SUB} UNION ALL SELECT 1)",
    "SELECT t.a FROM t WHERE t.b > (SELECT MIN(1))",
    "SELECT t.a FROM t WHERE b > (SELECT MIN(u.b) FROM u)",  # a bare column, no schema
    "SELECT t.a FROM t WHERE t.b > (SELECT MIN(t.b) FROM u AS t)",  # the subquery has its own t
    "SELECT t.a FROM t WHERE t.b > (SELECT MIN(u.b) FROM u WHERE EXISTS (SELECT 1 FROM u AS t))",
    "SELECT t.a FROM t JOIN u ON t.a = u.a WHERE b > (SELECT MIN(v.b) FROM u AS v)",
    f"SELECT t.a, SUM(t.b) FROM t GROUP BY t.a HAVING SUM(t.b) > (SELECT MIN(u.b) FROM u)",
    f"SELECT t.a FROM t WHERE (SELECT MAX(v.b) FROM u AS v) > (SELECT MIN(u.b) {SUB})",
])
def test_other_comparisons_are_left_alone(sql):
    assert read(sql) is None


def test_a_bare_column_is_not_qualified_when_it_could_be_read_inside():
    assert read("SELECT a FROM t JOIN u ON t.a = u.a WHERE b > (SELECT MIN(v.b) FROM u AS v)", SCHEMA) is None  # two sources
    assert read("SELECT a FROM t WHERE zz > (SELECT MIN(u.b) FROM u)", SCHEMA) is None  # not a column of t
    assert read("SELECT a FROM t WHERE b > (SELECT MIN(t.b) FROM t)", SCHEMA) is None  # t is declared inside


ANY = f"SELECT t.a, t.b FROM t WHERE t.b > ANY (SELECT u.b {SUB})"
TARGETS = [
    (f"SELECT t.a, t.b FROM t WHERE t.b > (SELECT MIN(u.b) {SUB})", ANY),
    (f"SELECT a, b FROM t WHERE b > (SELECT MIN(u.b) FROM u WHERE u.c = t.c)", ANY),
    (f"SELECT t.a, t.b FROM t WHERE t.b > (SELECT MIN(u.b) {SUB})", f"SELECT t.a, t.b FROM t WHERE EXISTS (SELECT 1 {SUB} AND u.b < t.b)"),
    (f"SELECT t.a, t.b FROM t WHERE t.b <= (SELECT MAX(u.b) {SUB})", f"SELECT t.a, t.b FROM t WHERE t.b <= ANY (SELECT u.b {SUB})"),
    (f"SELECT t.a, t.b FROM t WHERE (SELECT MAX(u.b) FROM u) > t.b", "SELECT t.a, t.b FROM t WHERE EXISTS (SELECT 1 FROM u WHERE t.b < u.b)"),
    (f"SELECT t.a, t.b FROM t WHERE t.b > (SELECT MIN(u.b) {SUB})", f"SELECT t.a, t.b FROM t WHERE (SELECT MIN(u.b) {SUB}) < t.b"),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_extremum_comparisons_prove(left, right):
    assert proved(left, right)


ROWS = {
    "t": [(1, 5, 1), (1, 5, 1), (2, 0, 1), (3, 9, 2), (4, None, 1), (5, 3, 3), (6, 3, None), (7, 1, 1)],
    "u": [(1, 1, 1), (1, 7, 1), (2, None, 1), (4, None, 2), (None, 3, 9), (3, 1, 5)],
}
EXISTS_GT = f"SELECT t.a, t.b FROM t WHERE EXISTS (SELECT 1 {SUB} AND t.b > u.b)"
NEAR_MISSES = [
    # the other extremum, the other direction, strict against not strict
    (f"SELECT t.a, t.b FROM t WHERE t.b > (SELECT MAX(u.b) {SUB})", EXISTS_GT),
    (f"SELECT t.a, t.b FROM t WHERE t.b < (SELECT MIN(u.b) {SUB})", f"SELECT t.a, t.b FROM t WHERE EXISTS (SELECT 1 {SUB} AND t.b < u.b)"),
    (f"SELECT t.a, t.b FROM t WHERE t.b < (SELECT MIN(u.b) {SUB})", f"SELECT t.a, t.b FROM t WHERE NOT EXISTS (SELECT 1 {SUB} AND t.b >= u.b)"),
    (f"SELECT t.a, t.b FROM t WHERE t.b >= (SELECT MIN(u.b) FROM u)", "SELECT t.a, t.b FROM t WHERE EXISTS (SELECT 1 FROM u WHERE t.b > u.b)"),
    # negated: an empty or all-NULL subquery gives NULL, and NOT EXISTS gives TRUE
    (f"SELECT t.a, t.b FROM t WHERE NOT t.b > (SELECT MIN(u.b) {SUB})", f"SELECT t.a, t.b FROM t WHERE NOT EXISTS (SELECT 1 {SUB} AND t.b > u.b)"),
    # as an output: NULL against FALSE
    (f"SELECT t.a, t.b > (SELECT MIN(u.b) {SUB}) AS f FROM t", f"SELECT t.a, EXISTS (SELECT 1 {SUB} AND t.b > u.b) AS f FROM t"),
    # another correlation
    (f"SELECT t.a, t.b FROM t WHERE t.b > (SELECT MIN(u.b) {SUB})", "SELECT t.a, t.b FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.c AND t.b > u.b)"),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, ROWS)


READ_QUERIES = [
    f"SELECT t.a, t.b FROM t WHERE t.b > (SELECT MIN(u.b) {SUB})",
    f"SELECT t.a, t.b FROM t WHERE t.b >= (SELECT MIN(u.b) {SUB}) AND t.a > 1",
    f"SELECT t.a, t.b FROM t WHERE t.b < (SELECT MAX(u.b) {SUB})",
    f"SELECT t.a, t.b FROM t WHERE (SELECT MAX(DISTINCT u.b + u.a) {SUB} OR u.a = 1) >= t.b + 1",
    "SELECT t.a, t.b FROM t WHERE 2 > (SELECT MIN(u.b) FROM u)",
    "SELECT a, b FROM t WHERE b > (SELECT MIN(u.b) FROM u LEFT JOIN t AS v ON v.a = u.a WHERE v.c = t.c OR v.c IS NULL)",
]


def random_rows(rng):
    def table(size):
        rows = [(rng.choice([None, 1, 2, 3]), rng.choice([None, 0, 1, 2, 5]), rng.choice([None, 1, 2])) for _ in range(rng.randint(0, size))]
        return rows + rows[:2]

    return {"t": table(6), "u": table(5)}


@pytest.mark.parametrize("sql", READ_QUERIES)
def test_the_exists_reading_gives_the_same_rows(sql):
    out = read(sql, SCHEMA)
    normalized = normalize(sql, schema=SCHEMA, types=TYPES, dialect="postgres")
    assert out is not None and "EXISTS" in out and "MIN(" not in out and "MAX(" not in out
    rng = random.Random(len(sql))
    for rows in [{}, ROWS] + [random_rows(rng) for _ in range(14)]:
        assert not differ_on_duckdb(sql, out, rows)
        assert not differ_on_duckdb(sql, normalized, rows)
