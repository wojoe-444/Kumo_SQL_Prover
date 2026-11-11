"""A counting subquery tested against zero as EXISTS (``kumo_sql_prover.count_exists_rules``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.count_exists_rules import count_tests_to_exists

SCHEMA = {"a": ["id", "v"], "b": ["a_id", "x"]}
TYPES = {name: {column: "INT64" for column in columns} for name, columns in SCHEMA.items()}
COUNT = "(SELECT COUNT(*) FROM b WHERE b.a_id = a.id)"
EXISTS = "EXISTS(SELECT 1 FROM b WHERE b.a_id = a.id)"


def rewritten(sql):
    return count_tests_to_exists(sqlglot.parse_one(sql, read="bigquery")).sql(dialect="bigquery")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False).proven


@pytest.mark.parametrize("test, some", [
    ("{n} > 0", True), ("{n} >= 1", True), ("{n} != 0", True), ("0 < {n}", True), ("1 <= {n}", True), ("0 != {n}", True), ("({n}) > (0)", True),
    ("{n} = 0", False), ("{n} < 1", False), ("{n} <= 0", False), ("0 = {n}", False), ("1 > {n}", False), ("0 >= {n}", False),
])
def test_tests_against_zero(test, some):
    assert rewritten(f"SELECT a.id FROM a WHERE {test.format(n=COUNT)}") == f"SELECT a.id FROM a WHERE {'' if some else 'NOT '}{EXISTS}"


def test_the_rewrite():
    # anywhere in a query: the test is never NULL
    assert rewritten(f"SELECT a.id, {COUNT} > 0 AS has, NOT {COUNT} = 0 FROM a") == f"SELECT a.id, {EXISTS} AS has, NOT NOT {EXISTS} FROM a"
    # COUNT(x) counts the rows where x is not NULL
    assert rewritten("SELECT a.id FROM a WHERE (SELECT COUNT(b.x) FROM b WHERE b.a_id = a.id OR b.x = 1) > 0") == "SELECT a.id FROM a WHERE EXISTS(SELECT 1 FROM b WHERE (b.a_id = a.id OR b.x = 1) AND NOT b.x IS NULL)"
    assert rewritten("SELECT (SELECT COUNT(DISTINCT b.x + 1) FROM b) < 1") == "SELECT NOT EXISTS(SELECT 1 FROM b WHERE NOT (b.x + 1) IS NULL)"
    # joins and derived tables come along
    assert rewritten("SELECT 1 FROM a WHERE (SELECT COUNT(*) FROM b JOIN (SELECT 1 AS one) AS o ON b.x = o.one WHERE b.a_id = a.id) >= 1") == "SELECT 1 FROM a WHERE EXISTS(SELECT 1 FROM b JOIN (SELECT 1 AS one) AS o ON b.x = o.one WHERE b.a_id = a.id)"
    # grouped on the correlated key, as a WHERE conjunct
    assert rewritten("SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = a.id GROUP BY b.a_id) > 0 AND a.v = 1") == f"SELECT a.id FROM a WHERE {EXISTS} AND a.v = 1"
    assert rewritten("SELECT a.id FROM a WHERE (SELECT COUNT(b.x) FROM b WHERE a.id + 1 = b.a_id AND b.x = 2 GROUP BY b.a_id, b.x) != 0") == "SELECT a.id FROM a WHERE EXISTS(SELECT 1 FROM b WHERE a.id + 1 = b.a_id AND b.x = 2 AND NOT b.x IS NULL)"


@pytest.mark.parametrize("sql", [
    f"SELECT a.id FROM a WHERE {COUNT} > 1",
    f"SELECT a.id FROM a WHERE {COUNT} = 1",
    f"SELECT a.id FROM a WHERE {COUNT} >= 0",
    f"SELECT a.id FROM a WHERE {COUNT} > 0.5",
    f"SELECT a.id FROM a WHERE {COUNT} > a.v",
    f"SELECT a.id FROM a WHERE {COUNT} + 0 > 0",
    f"SELECT a.id FROM a WHERE {COUNT} IS NOT DISTINCT FROM 0",
    "SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = a.id LIMIT 0) > 0",  # no row: the count is NULL
    "SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = a.id HAVING COUNT(*) > 2) > 0",
    "SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = a.id ORDER BY 1) > 0",
    "SELECT a.id FROM a WHERE (SELECT DISTINCT COUNT(*) FROM b WHERE b.a_id = a.id) > 0",
    "SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b UNION ALL SELECT 0) > 0",
    "SELECT a.id FROM a WHERE (SELECT SUM(b.x) FROM b WHERE b.a_id = a.id) > 0",
    "SELECT a.id FROM a WHERE (SELECT COUNTIF(b.x > 1) FROM b WHERE b.a_id = a.id) > 0",
    "SELECT a.id FROM a WHERE (SELECT COUNT(*) + 1 FROM b WHERE b.a_id = a.id) > 0",
    "SELECT a.id FROM a WHERE (SELECT COUNT(*) OVER () FROM b WHERE b.a_id = a.id) > 0",
    "SELECT a.id FROM a WHERE (SELECT COUNT((SELECT MAX(b.x) FROM b)) FROM b) > 0",
    "SELECT a.id FROM a WHERE (SELECT COUNT(DISTINCT b.x, b.a_id) FROM b) > 0",
    "SELECT a.id FROM a WHERE (SELECT MAX(b.x) FROM b WHERE b.a_id = a.id) > 0",
    "SELECT a.id FROM a WHERE (SELECT b.x FROM b WHERE b.a_id = a.id) > 0",
    # grouped: NULL without a group, so only where NULL and FALSE do the same
    "SELECT a.id, (SELECT COUNT(*) FROM b WHERE b.a_id = a.id GROUP BY b.a_id) > 0 FROM a",
    "SELECT a.id FROM a WHERE NOT (SELECT COUNT(*) FROM b WHERE b.a_id = a.id GROUP BY b.a_id) > 0",
    "SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = a.id GROUP BY b.a_id) > 0 OR a.v = 1",
    "SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = a.id GROUP BY b.a_id) = 0",
    "SELECT a.id FROM a WHERE COALESCE((SELECT COUNT(*) FROM b WHERE b.a_id = a.id GROUP BY b.a_id) > 0, TRUE)",
    # grouped on something that can take several values: several groups are an error, not a count
    "SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = a.id GROUP BY b.x) > 0",
    "SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = a.id OR b.x = 1 GROUP BY b.a_id) > 0",
    "SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id >= a.id GROUP BY b.a_id) > 0",
    "SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = b.x GROUP BY b.a_id) > 0",
    "SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE a_id = a.id GROUP BY a_id) > 0",
    "SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = a.id GROUP BY b.a_id + 1) > 0",
    "SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = a.id GROUP BY ROLLUP (b.a_id)) > 0",
    "SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = (SELECT MAX(a.id) FROM a) GROUP BY b.a_id) > 0",
])
def test_declined(sql):
    assert rewritten(sql) == sqlglot.parse_one(sql, read="bigquery").sql(dialect="bigquery")


@pytest.mark.parametrize("left, right", [
    (f"SELECT a.id FROM a WHERE {COUNT} > 0", f"SELECT a.id FROM a WHERE {EXISTS}"),
    (f"SELECT a.id FROM a WHERE 1 <= {COUNT}", f"SELECT a.id FROM a WHERE {EXISTS}"),
    (f"SELECT a.id FROM a WHERE {COUNT} = 0", f"SELECT a.id FROM a WHERE NOT {EXISTS}"),
    ("SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = a.id GROUP BY b.a_id) > 0", f"SELECT a.id FROM a WHERE {EXISTS}"),
    ("SELECT a.id FROM a WHERE (SELECT COUNT(b.x) FROM b WHERE b.a_id = a.id) > 0", "SELECT a.id FROM a WHERE EXISTS(SELECT 1 FROM b WHERE b.a_id = a.id AND b.x IS NOT NULL)"),
    ("SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = a.id AND b.x > 2) > 0", "SELECT a.id FROM a WHERE EXISTS(SELECT b.x FROM b WHERE b.x > 2 AND a.id = b.a_id)"),
])
def test_proved(left, right):
    assert proved(left, right)


@pytest.mark.parametrize("left, right", [
    # COUNT(x) is 0 when every x of the rows is NULL
    ("SELECT a.id FROM a WHERE (SELECT COUNT(b.x) FROM b WHERE b.a_id = a.id) > 0", f"SELECT a.id FROM a WHERE {EXISTS}"),
    # a count never returns no row
    ("SELECT a.id FROM a WHERE EXISTS(SELECT COUNT(*) FROM b WHERE b.a_id = a.id)", f"SELECT a.id FROM a WHERE {EXISTS}"),
    # other numbers
    (f"SELECT a.id FROM a WHERE {COUNT} > 1", f"SELECT a.id FROM a WHERE {EXISTS}"),
    (f"SELECT a.id FROM a WHERE {COUNT} >= 0", f"SELECT a.id FROM a WHERE {EXISTS}"),
    (f"SELECT a.id FROM a WHERE {COUNT} = 0", f"SELECT a.id FROM a WHERE {EXISTS}"),
    (f"SELECT a.id FROM a WHERE {COUNT} > 0", f"SELECT a.id FROM a WHERE NOT {EXISTS}"),
    # the grouped count is NULL without a group: NOT NULL drops the row, NOT EXISTS keeps it
    ("SELECT a.id FROM a WHERE NOT (SELECT COUNT(*) FROM b WHERE b.a_id = a.id GROUP BY b.a_id) > 0", f"SELECT a.id FROM a WHERE NOT {EXISTS}"),
    ("SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = a.id GROUP BY b.a_id) = 0", f"SELECT a.id FROM a WHERE NOT {EXISTS}"),
    ("SELECT a.id, (SELECT COUNT(*) FROM b WHERE b.a_id = a.id GROUP BY b.a_id) > 0 FROM a", f"SELECT a.id, {EXISTS} FROM a"),
    # another filter
    (f"SELECT a.id FROM a WHERE {COUNT} > 0", "SELECT a.id FROM a WHERE EXISTS(SELECT 1 FROM b WHERE b.a_id = a.id AND b.x > 2)"),
])
def test_not_proved(left, right):
    assert not proved(left, right)


def test_the_rewrite_keeps_the_rows_on_duckdb():
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    queries = [
        f"SELECT a.id, a.v, {COUNT} > 0, {COUNT} = 0, 1 > {COUNT} FROM a",
        f"SELECT a.id, a.v FROM a WHERE NOT {COUNT} != 0 OR a.v = 1",
        "SELECT a.id, (SELECT COUNT(b.x) FROM b WHERE b.a_id = a.id OR b.x = a.v) > 0, (SELECT COUNT(DISTINCT b.x + a.v) FROM b) <= 0 FROM a",
        "SELECT a.id, a.v FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = a.id GROUP BY b.a_id) > 0",
        "SELECT a.id, a.v FROM a WHERE a.v > 0 AND (SELECT COUNT(b.x) FROM b WHERE b.a_id = a.id AND a.v = b.x GROUP BY b.x, b.a_id) >= 1",
        "SELECT a.id, a.v FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = 2 GROUP BY b.a_id) != 0",
        "SELECT (SELECT COUNT(*) FROM b JOIN a ON a.id = b.a_id WHERE a.v > 1) > 0, (SELECT COUNT(*) FROM a WHERE FALSE) = 0",
    ]
    pairs = [(sqlglot.transpile(q, read="bigquery", write="duckdb")[0], sqlglot.transpile(rewritten(q), read="bigquery", write="duckdb")[0]) for q in queries]
    assert all(before != after for before, after in pairs)
    rng = random.Random(23)
    for _ in range(80):
        db = duckdb.connect()
        for name, columns in SCHEMA.items():
            db.execute(f"CREATE TABLE {name} ({', '.join(c + ' BIGINT' for c in columns)})")
            insert_rows(db, name, [tuple(rng.choice([None, 0, 1, 2, 3]) for _ in columns) for _ in range(rng.choice([0, 1, 2, 4, 6]))])
        for before, after in pairs:
            x, y = run_unoptimized(db, before, after)
            assert sorted(map(repr, x)) == sorted(map(repr, y)), (before, after)
        db.close()
