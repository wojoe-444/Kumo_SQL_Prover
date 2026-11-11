"""The count guards of Calcite's expanded ``NOT IN`` are dropped when they decide nothing (``kumo_sql_prover.expanded_not_in``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.expanded_not_in import drop_count_guards
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"a": ["x", "g"], "t": ["k", "f", "n"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}
NOT_NULL = {"t": frozenset({"k"}), "a": frozenset({"x"})}
CONSTRAINTS = {name: TableConstraints(not_null=columns) for name, columns in NOT_NULL.items()}
COUNTS = "(SELECT COUNT(*) AS c, COUNT(*) AS ck FROM t WHERE f = 1) AS cnt"
PROBE = "NOT EXISTS(SELECT 1 FROM t AS p WHERE p.f = 1 AND p.k = a.x)"


def dropped(sql, dialect="postgres"):
    out = drop_count_guards(sqlglot.parse_one(sql, read=dialect), SCHEMA, NOT_NULL, dialect)
    return out.sql(dialect=dialect) if out is not None else None


def proved(left, right):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, dialect="postgres", compare_names=False
    ).proven


def rows_on_duckdb(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name, columns in SCHEMA.items():
        db.execute(f"CREATE TABLE {name} ({', '.join(f'{c} BIGINT' for c in columns)})")
        insert_rows(db, name, rows.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def differ_on_duckdb(left, right, rows):
    first, second = rows_on_duckdb([left, right], rows)
    return first != second


def test_the_guards_and_the_counts_are_dropped():
    assert dropped(f"SELECT a.g FROM a CROSS JOIN {COUNTS} WHERE cnt.c = 0 OR {PROBE} AND cnt.ck >= cnt.c") == f"SELECT a.g FROM a WHERE {PROBE}"
    assert dropped(f"SELECT a.g FROM a, {COUNTS} WHERE a.g > 0 AND ({PROBE} OR 0 = cnt.ck)") == f"SELECT a.g FROM a WHERE a.g > 0 AND {PROBE}"
    # COUNT(k) of a NOT NULL column counts every row
    counts = "(SELECT COUNT(*) AS c, COUNT(t.k) AS ck FROM t) AS cnt"
    probe = "NOT EXISTS(SELECT 1 FROM t WHERE t.k = a.x AND t.n > 2)"
    assert dropped(f"SELECT a.g FROM a CROSS JOIN {counts} WHERE cnt.c = 0 OR {probe} AND cnt.c <= cnt.ck") == f"SELECT a.g FROM a WHERE {probe}"
    # another read of the counts keeps the join; the conjunct is still simplified
    assert (
        dropped(f"SELECT a.g, cnt.c FROM a CROSS JOIN {COUNTS} WHERE cnt.c = 0 OR {PROBE} AND cnt.ck >= cnt.c")
        == f"SELECT a.g, cnt.c FROM a CROSS JOIN {COUNTS} WHERE {PROBE}"
    )


@pytest.mark.parametrize("sql", [
    f"SELECT a.g FROM a CROSS JOIN {COUNTS} WHERE cnt.c = 1 OR {PROBE} AND cnt.ck >= cnt.c",
    f"SELECT a.g FROM a CROSS JOIN {COUNTS} WHERE cnt.c = 0 AND {PROBE}",
    f"SELECT a.g FROM a CROSS JOIN {COUNTS} WHERE cnt.c = 0 OR {PROBE} AND cnt.ck > cnt.c",  # never TRUE: not a guard
    f"SELECT a.g FROM a CROSS JOIN {COUNTS} WHERE cnt.c = 0 OR {PROBE} AND a.g = 1",  # a real condition beside the probe
    f"SELECT a.g FROM a CROSS JOIN {COUNTS} WHERE cnt.c = 0 OR {PROBE.replace('NOT ', '')} AND cnt.ck >= cnt.c",  # EXISTS
    # the probe can return a row the count does not count
    f"SELECT a.g FROM a CROSS JOIN {COUNTS} WHERE cnt.c = 0 OR NOT EXISTS(SELECT 1 FROM t AS p WHERE p.k = a.x) AND cnt.ck >= cnt.c",
    f"SELECT a.g FROM a CROSS JOIN {COUNTS} WHERE cnt.c = 0 OR NOT EXISTS(SELECT 1 FROM t AS p WHERE p.f = 2 AND p.k = a.x) AND cnt.ck >= cnt.c",
    f"SELECT a.g FROM a CROSS JOIN {COUNTS} WHERE cnt.c = 0 OR NOT EXISTS(SELECT 1 FROM t AS p WHERE (p.f = 1 OR p.n = 1) AND p.k = a.x) AND cnt.ck >= cnt.c",
    f"SELECT a.g FROM a CROSS JOIN {COUNTS} WHERE cnt.c = 0 OR NOT EXISTS(SELECT 1 FROM a AS p WHERE p.g = 1 AND p.x = a.x) AND cnt.ck >= cnt.c",
    f"SELECT a.g FROM a CROSS JOIN {COUNTS} WHERE cnt.c = 0 OR NOT EXISTS(SELECT COUNT(*) FROM t AS p WHERE p.f = 1 AND p.k = a.x) AND cnt.ck >= cnt.c",
    # the nullable column may hold fewer values than there are rows
    f"SELECT a.g FROM a CROSS JOIN (SELECT COUNT(*) AS c, COUNT(n) AS ck FROM t WHERE f = 1) AS cnt WHERE cnt.c = 0 OR {PROBE} AND cnt.ck >= cnt.c",
    f"SELECT a.g FROM a CROSS JOIN (SELECT COUNT(n) AS c, COUNT(*) AS ck FROM t WHERE f = 1) AS cnt WHERE cnt.c = 0 OR {PROBE} AND cnt.ck >= cnt.c",
    # not a one-row source of plain counts
    f"SELECT a.g FROM a CROSS JOIN (SELECT COUNT(*) AS c, COUNT(*) AS ck FROM t WHERE f = 1 GROUP BY n) AS cnt WHERE cnt.c = 0 OR {PROBE} AND cnt.ck >= cnt.c",
    f"SELECT a.g FROM a CROSS JOIN (SELECT COUNT(*) AS c, COUNT(*) AS ck FROM t WHERE f = 1 HAVING COUNT(*) > 1) AS cnt WHERE cnt.c = 0 OR {PROBE} AND cnt.ck >= cnt.c",
    f"SELECT a.g FROM a CROSS JOIN (SELECT COUNT(DISTINCT k) AS c, COUNT(*) AS ck FROM t WHERE f = 1) AS cnt WHERE cnt.c = 0 OR {PROBE} AND cnt.ck >= cnt.c",
    f"SELECT a.g FROM a CROSS JOIN (SELECT SUM(k) AS c, COUNT(*) AS ck FROM t WHERE f = 1) AS cnt WHERE cnt.c = 0 OR {PROBE} AND cnt.ck >= cnt.c",
    f"SELECT a.g FROM a CROSS JOIN (SELECT COUNT(*) AS c, COUNT(*) AS ck FROM t WHERE f = RANDOM()) AS cnt WHERE cnt.c = 0 OR NOT EXISTS(SELECT 1 FROM t AS p WHERE p.f = RANDOM() AND p.k = a.x) AND cnt.ck >= cnt.c",
    f"SELECT a.g FROM a JOIN {COUNTS} ON cnt.c = a.g WHERE cnt.c = 0 OR {PROBE} AND cnt.ck >= cnt.c",
    f"SELECT a.g FROM a LEFT JOIN {COUNTS} ON TRUE WHERE cnt.c = 0 OR {PROBE} AND cnt.ck >= cnt.c",
    f"SELECT a.g FROM a CROSS JOIN (SELECT COUNT(*) AS c, COUNT(*) AS ck FROM z) AS cnt WHERE cnt.c = 0 OR NOT EXISTS(SELECT 1 FROM z WHERE z.k = a.x) AND cnt.ck >= cnt.c",
])
def test_other_shapes_are_left_alone(sql):
    assert dropped(sql) is None


EXPANDED = (
    "SELECT a.g FROM a CROSS JOIN (SELECT COUNT(*) AS c, COUNT(*) AS ck FROM t WHERE f = 1) AS cnt "
    "LEFT JOIN (SELECT k, TRUE AS i FROM t WHERE f = 1 GROUP BY k) AS m ON a.x = m.k "
    "WHERE cnt.c = 0 OR m.i IS NULL AND cnt.ck >= cnt.c"
)
TARGETS = [
    (EXPANDED, "SELECT a.g FROM a WHERE a.x NOT IN (SELECT k FROM t WHERE f = 1)"),
    (EXPANDED.replace("SELECT a.g FROM", "SELECT a.g, COUNT(*) FROM") + " GROUP BY a.g", "SELECT a.g, COUNT(*) FROM a WHERE a.x NOT IN (SELECT k FROM t WHERE f = 1) GROUP BY a.g"),
    (EXPANDED.replace("SELECT a.g FROM", "SELECT COUNT(DISTINCT a.g) FROM"), "SELECT COUNT(DISTINCT a.g) FROM a WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.k = a.x AND t.f = 1)"),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_expanded_not_in_proves(left, right):
    assert proved(left, right)


ROWS = {"a": [(1, 1), (1, 1), (2, 1), (3, 2), (4, 2)], "t": [(1, 1, None), (1, 1, 3), (2, 0, None), (4, 2, 4), (5, 0, 1)]}
NEAR_MISSES = [
    # the counts are of another filter: on a database where it is empty every row passes
    (EXPANDED.replace("COUNT(*) AS ck FROM t WHERE f = 1", "COUNT(*) AS ck FROM t WHERE f = 3"), "SELECT a.g FROM a WHERE a.x NOT IN (SELECT k FROM t WHERE f = 1)"),
    # ck > c is never TRUE
    (EXPANDED.replace("cnt.ck >= cnt.c", "cnt.ck > cnt.c"), "SELECT a.g FROM a WHERE a.x NOT IN (SELECT k FROM t WHERE f = 1)"),
    # the probe of another filter
    (EXPANDED, "SELECT a.g FROM a WHERE a.x NOT IN (SELECT k FROM t WHERE f = 0)"),
    # COUNT(n) counts fewer rows than COUNT(*) when n holds a NULL
    (EXPANDED.replace("COUNT(*) AS ck", "COUNT(n) AS ck"), "SELECT a.g FROM a WHERE a.x NOT IN (SELECT k FROM t WHERE f = 1)"),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, ROWS)


DROPPED_QUERIES = [
    f"SELECT a.g FROM a CROSS JOIN {COUNTS} WHERE cnt.c = 0 OR {PROBE} AND cnt.ck >= cnt.c",
    f"SELECT a.g FROM a, {COUNTS} WHERE a.g > 0 AND ({PROBE} OR 0 = cnt.ck)",
    f"SELECT a.g, cnt.c FROM a CROSS JOIN {COUNTS} WHERE cnt.c = 0 OR {PROBE} AND cnt.ck >= cnt.c",
    "SELECT a.g FROM a CROSS JOIN (SELECT COUNT(*) AS c, COUNT(t.k) AS ck FROM t) AS cnt "
    "WHERE cnt.c = 0 OR NOT EXISTS(SELECT 1 FROM t WHERE t.k = a.x AND t.n > 2) AND cnt.c <= cnt.ck",
]


def random_rows(rng):
    a = [(rng.choice([1, 2, 3]), rng.choice([None, 0, 1])) for _ in range(rng.randint(0, 5))]
    t = [(rng.choice([1, 2, 3]), rng.choice([None, 0, 1]), rng.choice([None, 1, 3])) for _ in range(rng.randint(0, 5))]
    return {"a": a + a[:2], "t": t + t[:1]}


@pytest.mark.parametrize("sql", DROPPED_QUERIES)
def test_the_rewrite_returns_the_same_rows_on_random_databases(sql):
    rewritten = dropped(sql)
    assert rewritten is not None
    rng = random.Random(len(sql))
    for _ in range(40):
        rows = random_rows(rng)
        first, second = rows_on_duckdb([sql, rewritten], rows)
        assert first == second, rows
