"""A subquery test beside a weaker or stronger copy of itself (``kumo_sql_prover.subquery_absorption``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.subquery_absorption import absorb_subquery_tests

SCHEMA = {"l": ["a", "b"], "r": ["c", "d"]}
TYPES = {name: {column: "INT64" for column in columns} for name, columns in SCHEMA.items()}
WIDE_IN, NARROW_IN = "l.a IN (SELECT c FROM r)", "l.a IN (SELECT c FROM r WHERE l.b < r.d)"
WIDE, NARROW = "EXISTS(SELECT * FROM r WHERE l.a = r.c)", "EXISTS(SELECT * FROM r WHERE l.a = r.c AND l.b < r.d)"


def rewritten(sql):
    return absorb_subquery_tests(sqlglot.parse_one(sql, read="bigquery")).sql(dialect="bigquery")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False).proven


@pytest.mark.parametrize("test, kept", [
    # OR keeps the higher test: the wider subquery, and under NOT the narrower one
    (f"{WIDE_IN} OR {NARROW_IN}", WIDE_IN),
    (f"{NARROW_IN} OR {WIDE_IN}", WIDE_IN),
    (f"{WIDE} OR {NARROW}", WIDE),
    (f"NOT {WIDE} OR NOT {NARROW}", f"NOT {NARROW}"),
    (f"NOT {NARROW_IN} OR NOT {WIDE_IN}", f"NOT {NARROW_IN}"),
    (f"{WIDE} OR {WIDE}", WIDE),
    (f"({WIDE}) OR (NOT NOT {NARROW})", WIDE),
    # AND keeps the lower one
    (f"{WIDE_IN} AND {NARROW_IN}", NARROW_IN),
    (f"{NARROW} AND {WIDE}", NARROW),
    (f"NOT {NARROW} AND NOT {WIDE}", f"NOT {WIDE}"),
    (f"NOT {WIDE_IN} AND NOT {NARROW_IN}", f"NOT {WIDE_IN}"),
    # among other operands, and with a select list that EXISTS does not read
    (f"l.b = 1 OR {NARROW} OR l.a = 2 OR {WIDE}", f"l.b = 1 OR l.a = 2 OR {WIDE}"),
    (f"l.b = 1 AND ({NARROW} OR {WIDE})", f"l.b = 1 AND ({WIDE})"),
    (f"{WIDE} OR EXISTS(SELECT r.d + 1 FROM r WHERE r.d = 3 AND (l.a = r.c))", WIDE),
    ("l.a IN (SELECT c AS x FROM r JOIN l AS m ON m.a = r.d WHERE r.d > 1) OR l.a IN (SELECT c FROM r JOIN l AS m ON m.a = r.d WHERE r.d > 1 AND m.b = 2 AND r.c > 0)", "l.a IN (SELECT c AS x FROM r JOIN l AS m ON m.a = r.d WHERE r.d > 1)"),
])
def test_the_rewrite(test, kept):
    assert rewritten(f"SELECT * FROM l WHERE {test}") == f"SELECT * FROM l WHERE {kept}"
    assert rewritten(f"SELECT l.a, {test} AS t FROM l") == f"SELECT l.a, {kept} AS t FROM l"  # exact with NULLs, so anywhere


@pytest.mark.parametrize("test", [
    f"{WIDE} OR NOT {NARROW}",  # one negated
    f"{WIDE} OR {NARROW_IN}",  # two kinds of test
    f"{WIDE_IN} OR l.b IN (SELECT c FROM r WHERE l.b < r.d)",  # another tested value
    f"{WIDE_IN} OR l.a IN (SELECT d FROM r WHERE l.b < r.d)",  # another selected column
    f"{WIDE} OR EXISTS(SELECT * FROM r AS q WHERE l.a = q.c AND l.b < q.d)",  # another source text
    f"{WIDE} OR EXISTS(SELECT * FROM r JOIN l AS m ON TRUE WHERE l.a = r.c AND l.b < r.d)",
    f"{WIDE} OR EXISTS(SELECT * FROM r WHERE l.a = r.c OR l.b < r.d)",  # wider, but not by conjuncts
    f"{WIDE} OR EXISTS(SELECT * FROM r WHERE l.a >= r.c AND l.b < r.d)",
    f"{WIDE} OR EXISTS(SELECT COUNT(*) FROM r WHERE l.a = r.c AND l.b < r.d)",  # always TRUE
    f"{WIDE} OR EXISTS(SELECT * FROM r WHERE l.a = r.c AND l.b < r.d LIMIT 1)",
    f"{WIDE} OR EXISTS(SELECT DISTINCT * FROM r WHERE l.a = r.c AND l.b < r.d)",
    f"{WIDE} OR EXISTS(SELECT r.c FROM r WHERE l.a = r.c AND l.b < r.d GROUP BY r.c)",
    "EXISTS(SELECT * FROM r WHERE RAND() < 0.5) OR EXISTS(SELECT * FROM r WHERE RAND() < 0.5 AND r.c = 1)",
    "EXISTS(SELECT * FROM r WHERE f(r.c) = 1) OR EXISTS(SELECT * FROM r WHERE f(r.c) = 1 AND r.c = 1)",
    f"{WIDE_IN} OR l.a IN (SELECT * FROM r WHERE l.b < r.d)",
    f"{WIDE_IN} OR l.a IN (SELECT c FROM r WHERE l.b < r.d UNION ALL SELECT 1)",
    "l.a IN (1, 2) OR l.a IN (1)",
    f"{WIDE} OR l.a = 1",
    f"({WIDE} AND l.b = 1) OR ({NARROW} AND l.b = 2)",  # not operands of one OR
])
def test_declined(test):
    sql = f"SELECT * FROM l WHERE {test}"
    assert rewritten(sql) == sqlglot.parse_one(sql, read="bigquery").sql(dialect="bigquery")


@pytest.mark.parametrize("left, right", [
    (f"SELECT * FROM l WHERE {WIDE_IN}", f"SELECT * FROM l WHERE {WIDE_IN} OR {NARROW_IN}"),
    (f"SELECT * FROM l WHERE NOT {NARROW}", f"SELECT * FROM l WHERE NOT {NARROW} OR NOT {WIDE}"),
    ("SELECT * FROM l WHERE NOT a IN (SELECT c FROM r WHERE NOT c IS NULL)", "SELECT * FROM l WHERE NOT a IN (SELECT c FROM r) OR NOT a IN (SELECT c FROM r WHERE NOT c IS NULL)"),
    (f"SELECT * FROM l WHERE {NARROW}", f"SELECT * FROM l WHERE {WIDE} AND {NARROW}"),
])
def test_proved(left, right):
    assert proved(left, right)


@pytest.mark.parametrize("left, right", [
    (f"SELECT * FROM l WHERE {NARROW_IN}", f"SELECT * FROM l WHERE {WIDE_IN} OR {NARROW_IN}"),
    (f"SELECT * FROM l WHERE NOT {WIDE}", f"SELECT * FROM l WHERE NOT {NARROW} OR NOT {WIDE}"),
    (f"SELECT * FROM l WHERE {WIDE}", f"SELECT * FROM l WHERE {WIDE} AND {NARROW}"),
    ("SELECT * FROM l WHERE NOT a IN (SELECT c FROM r)", "SELECT * FROM l WHERE NOT a IN (SELECT c FROM r) OR NOT a IN (SELECT c FROM r WHERE NOT c IS NULL)"),
    (f"SELECT * FROM l WHERE {WIDE}", f"SELECT * FROM l WHERE {WIDE} OR NOT {NARROW}"),
])
def test_not_proved(left, right):
    assert not proved(left, right)


def test_the_rewrite_keeps_the_rows_on_duckdb():
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    tests = [
        f"{WIDE_IN} OR {NARROW_IN}", f"{WIDE_IN} AND {NARROW_IN}", f"NOT {WIDE_IN} OR NOT {NARROW_IN}", f"NOT {NARROW_IN} AND NOT {WIDE_IN}",
        f"{NARROW} OR {WIDE}", f"NOT {NARROW} OR NOT {WIDE}", f"{NARROW} AND {WIDE} AND l.b > 0", f"NOT {NARROW} AND NOT {WIDE}",
        "NOT l.a IN (SELECT c FROM r) OR NOT l.a IN (SELECT c FROM r WHERE NOT c IS NULL)",
        "l.b IN (SELECT r.d + 1 FROM r WHERE r.c > 0) OR l.a = 1 OR l.b IN (SELECT r.d + 1 FROM r WHERE r.c > 0 AND r.d < l.a)",
    ]
    queries = [f"SELECT l.a, l.b, {test} AS t FROM l" for test in tests]
    pairs = [(sqlglot.transpile(q, read="bigquery", write="duckdb")[0], sqlglot.transpile(rewritten(q), read="bigquery", write="duckdb")[0]) for q in queries]
    assert all(before != after for before, after in pairs)
    rng = random.Random(31)
    for _ in range(80):
        db = duckdb.connect()
        for name, columns in SCHEMA.items():
            db.execute(f"CREATE TABLE {name} ({', '.join(c + ' BIGINT' for c in columns)})")
            insert_rows(db, name, [tuple(rng.choice([None, 0, 1, 2, 3]) for _ in columns) for _ in range(rng.choice([0, 1, 2, 4, 6]))])
        for before, after in pairs:
            x, y = run_unoptimized(db, before, after)
            assert sorted(map(repr, x)) == sorted(map(repr, y)), (before, after)
        db.close()
