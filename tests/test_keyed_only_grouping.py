"""A key-only GROUP BY over an outer join is SELECT DISTINCT (``kumo_sql_prover.keyed_only_grouping``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.keyed_only_grouping import key_only_group_to_distinct
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"t": ["a", "b", "c"], "u": ["a", "b", "c"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}
B_NOT_NULL = {"t": TableConstraints(not_null=frozenset({"b"}))}


def rewritten(sql):
    out = key_only_group_to_distinct(sqlglot.parse_one(sql, read="postgres"))
    return out.sql(dialect="postgres") if out is not None else None


def proved(left, right, constraints=None):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, types=TYPES, constraints=constraints, dialect="postgres", compare_names=False
    ).proven


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


JOIN = "FROM t LEFT JOIN u ON t.a = u.a"


def test_a_key_only_group_by_over_an_outer_join_becomes_distinct():
    assert rewritten(f"SELECT t.a, u.b {JOIN} GROUP BY t.a, u.b") == f"SELECT DISTINCT t.a, u.b {JOIN}"
    assert rewritten(f"SELECT u.b AS x, t.a {JOIN} WHERE t.c = 1 GROUP BY t.a, u.b") == f"SELECT DISTINCT u.b AS x, t.a {JOIN} WHERE t.c = 1"
    assert rewritten(f"SELECT t.a, t.a {JOIN} GROUP BY t.a") == f"SELECT DISTINCT t.a, t.a {JOIN}"
    assert rewritten("SELECT t.a + u.b FROM t FULL JOIN u ON t.a = u.a GROUP BY t.a + u.b") == "SELECT DISTINCT t.a + u.b FROM t FULL JOIN u ON t.a = u.a"
    assert rewritten("SELECT u.c FROM t RIGHT JOIN u ON t.a = u.a GROUP BY u.c") == "SELECT DISTINCT u.c FROM t RIGHT JOIN u ON t.a = u.a"


@pytest.mark.parametrize("sql", [
    "SELECT t.a, t.b FROM t JOIN u ON t.a = u.a GROUP BY t.a, t.b",  # no outer join: read as before
    "SELECT t.a, t.b FROM t GROUP BY t.a, t.b",
    f"SELECT t.a {JOIN} GROUP BY t.a, u.b",  # a key that is not an output: one row per u.b as well
    f"SELECT t.a + 1 {JOIN} GROUP BY t.a",  # computed from a key
    f"SELECT t.a, COUNT(*) {JOIN} GROUP BY t.a",
    f"SELECT t.a {JOIN} GROUP BY t.a HAVING COUNT(*) > 1",
    f"SELECT t.a {JOIN} GROUP BY t.a HAVING t.a > 1",
    f"SELECT t.a {JOIN} GROUP BY t.a ORDER BY COUNT(*)",
    f"SELECT t.a {JOIN} GROUP BY t.a ORDER BY t.a LIMIT 2",
    f"SELECT t.a {JOIN} GROUP BY t.a LIMIT 2",
    f"SELECT t.a {JOIN} GROUP BY ROLLUP (t.a)",
    f"SELECT t.a {JOIN} GROUP BY GROUPING SETS ((t.a), ())",
    f"SELECT t.a {JOIN} GROUP BY CUBE (t.a)",
    f"SELECT t.a, GROUPING(t.a) {JOIN} GROUP BY t.a",
    f"SELECT t.a, ROW_NUMBER() OVER (ORDER BY t.a) {JOIN} GROUP BY t.a",
    f"SELECT t.a, 1 {JOIN} GROUP BY t.a, 1",  # a literal key is a position
    f"SELECT * {JOIN} GROUP BY t.a",
    f"SELECT DISTINCT t.a {JOIN} GROUP BY t.a",
    f"SELECT t.a {JOIN}",
])
def test_other_shapes_are_left_alone(sql):
    assert rewritten(sql) is None


DEDUPED = "(SELECT a FROM t WHERE c = 1 GROUP BY a) AS x LEFT JOIN (SELECT a, b FROM u GROUP BY a, b) AS y ON x.a = y.a"
TARGETS = [
    (f"SELECT t.a, u.b {JOIN} GROUP BY t.a, u.b", f"SELECT DISTINCT t.a, u.b {JOIN}"),
    # duplicates removed below the join as well (aggregate-join-transpose) against none removed
    (f"SELECT x.a, y.b FROM {DEDUPED} GROUP BY x.a, y.b", f"SELECT DISTINCT t.a, u.b {JOIN} WHERE t.c = 1"),
    (f"SELECT y.b FROM {DEDUPED} GROUP BY y.b", f"SELECT DISTINCT u.b {JOIN} WHERE t.c = 1"),
    # unread columns pruned from the join's inputs
    (
        "SELECT x.a FROM (SELECT a, b, c, a + b AS k FROM t WHERE c = 1) AS x FULL JOIN (SELECT a, b, b + 5 AS k FROM u) AS y ON x.a = y.a AND x.k = y.k GROUP BY x.a",
        "SELECT DISTINCT x.a FROM (SELECT a, a + b AS k FROM t WHERE c = 1) AS x FULL JOIN (SELECT a, b + 5 AS k FROM u) AS y ON x.a = y.a AND x.k = y.k",
    ),
    (
        "SELECT x.a FROM (SELECT a FROM t GROUP BY a) AS x RIGHT JOIN (SELECT a FROM u GROUP BY a) AS y ON x.a = y.a GROUP BY x.a",
        "SELECT DISTINCT x.a FROM (SELECT DISTINCT a FROM t) AS x RIGHT JOIN (SELECT DISTINCT a FROM u) AS y ON x.a = y.a",
    ),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_key_only_groupings_prove(left, right):
    assert proved(left, right)


SELF = "SELECT y.c FROM (SELECT b FROM t WHERE a = 1 GROUP BY b) AS x LEFT JOIN (SELECT c, b FROM t GROUP BY c, b) AS y ON x.b = y.b GROUP BY y.c"
SELF_INNER = "SELECT DISTINCT t2.c FROM t AS t1 JOIN t AS t2 ON t1.b = t2.b WHERE t1.a = 1"
SELF_IN = "SELECT DISTINCT c FROM t WHERE b IN (SELECT b FROM t WHERE a = 1)"


def test_a_left_join_against_the_rows_it_came_from_needs_a_not_null_column():
    # every left row is one of the right rows, so it matches itself: but only when t.b cannot be NULL
    assert proved(SELF, SELF_INNER, B_NOT_NULL)
    assert proved(SELF, SELF_IN, B_NOT_NULL)
    assert not proved(SELF, SELF_INNER)
    assert not proved(SELF, SELF_IN)
    rows = {"t": [(1, None, 7), (2, 3, 8)]}  # the NULL b matches nothing: a padded row, c IS NULL
    assert differ_on_duckdb(SELF, SELF_INNER, rows)
    assert differ_on_duckdb(SELF, SELF_IN, rows)


ROWS = {"t": [(1, 5, 1), (1, 5, 1), (2, 7, 1), (3, 9, 2), (None, 4, 1)], "u": [(1, 1, 1), (1, 2, 1), (1, 2, 2), (4, 2, 2), (None, 3, 3)]}
NEAR_MISSES = [
    # a key left out of the outputs keeps a row per value of it
    (f"SELECT t.a {JOIN} GROUP BY t.a, u.b", f"SELECT DISTINCT t.a {JOIN}"),
    # the other join, the other side
    (f"SELECT t.a, u.b {JOIN} GROUP BY t.a, u.b", "SELECT DISTINCT t.a, u.b FROM t JOIN u ON t.a = u.a"),
    (f"SELECT t.a, u.b {JOIN} GROUP BY t.a, u.b", "SELECT DISTINCT t.a, u.b FROM t RIGHT JOIN u ON t.a = u.a"),
    (f"SELECT t.a, u.b {JOIN} GROUP BY t.a, u.b", "SELECT DISTINCT t.a, u.b FROM t FULL JOIN u ON t.a = u.a"),
    # no duplicate removal at all, or only below the join
    (f"SELECT t.a, u.b {JOIN} GROUP BY t.a, u.b", f"SELECT t.a, u.b {JOIN}"),
    (f"SELECT x.a, y.c FROM {DEDUPED.replace('a, b FROM u GROUP BY a, b', 'a, b, c FROM u GROUP BY a, b, c')} GROUP BY x.a, y.c", "SELECT x.a, y.c FROM (SELECT DISTINCT a FROM t WHERE c = 1) AS x LEFT JOIN (SELECT DISTINCT a, b, c FROM u) AS y ON x.a = y.a"),
    # a row cap
    (f"SELECT t.a, u.b {JOIN} GROUP BY t.a, u.b", f"SELECT DISTINCT t.a, u.b {JOIN} ORDER BY 1, 2 LIMIT 2"),
    # another filter
    (f"SELECT t.a, u.b {JOIN} GROUP BY t.a, u.b", f"SELECT DISTINCT t.a, u.b {JOIN} WHERE u.b IS NOT NULL"),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, ROWS)


REWRITTEN_QUERIES = [
    f"SELECT t.a, u.b {JOIN} GROUP BY t.a, u.b",
    f"SELECT u.b, t.a, u.b {JOIN} WHERE t.c = 1 GROUP BY t.a, u.b",
    "SELECT t.a + u.b FROM t FULL JOIN u ON t.a = u.a GROUP BY t.a + u.b",
    "SELECT u.c, t.b FROM t RIGHT JOIN u ON t.a = u.a AND t.b > u.b GROUP BY u.c, t.b",
    f"SELECT x.a, y.b FROM {DEDUPED} GROUP BY x.a, y.b",
]


def random_rows(rng):
    def table(size):
        rows = [(rng.choice([None, 1, 2, 3]), rng.choice([None, 0, 1, 2]), rng.choice([None, 1, 2])) for _ in range(rng.randint(0, size))]
        return rows + rows[:2]  # with duplicate rows

    return {"t": table(6), "u": table(5)}


@pytest.mark.parametrize("sql", REWRITTEN_QUERIES)
def test_the_distinct_spelling_gives_the_same_rows(sql):
    out = rewritten(sql)
    normalized = normalize(sql, schema=SCHEMA, types=TYPES, dialect="postgres")
    assert out is not None and out.startswith("SELECT DISTINCT") and not out.endswith(sql[sql.rindex("GROUP BY"):])
    rng = random.Random(len(sql))
    for rows in [{}, ROWS] + [random_rows(rng) for _ in range(14)]:
        assert not differ_on_duckdb(sql, out, rows)
        assert not differ_on_duckdb(sql, normalized, rows)
