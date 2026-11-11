"""A row ``IN`` a list of rows is an ``OR`` of ``AND``s of equalities (``kumo_sql_prover.tuple_membership``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.tuple_membership import expand_row_membership

SCHEMA = {"t": ["k", "a", "b", "c"]}
TYPES = {"t": {"k": "INT", "a": "INT", "b": "INT", "c": "INT"}}


def expanded(sql, dialect="postgres"):
    out = expand_row_membership(sqlglot.parse_one(sql, read=dialect), dialect)
    return out.sql(dialect=dialect) if out is not None else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


def rows_on_duckdb(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE t (k BIGINT, a BIGINT, b BIGINT, c BIGINT)")
    insert_rows(db, "t", rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def test_the_rows_become_equalities():
    assert expanded("SELECT t.k FROM t WHERE (t.a, t.b) IN ((1, 2), (2, 1))") == "SELECT t.k FROM t WHERE ((t.a = 1 AND t.b = 2) OR (t.a = 2 AND t.b = 1))"
    assert expanded("SELECT t.k FROM t WHERE NOT (a, b, c) IN ((1, -2, 'x')) AND t.k > 0") == "SELECT t.k FROM t WHERE NOT ((a = 1 AND b = -2 AND c = 'x')) AND t.k > 0"
    assert expanded("SELECT (t.a, t.b) IN ((t.c, 1)) FROM t") == "SELECT ((t.a = t.c AND t.b = 1)) FROM t"
    assert expanded("SELECT t.k FROM t WHERE (t.a, t.b) IN ((1, 2))", "mysql") == "SELECT t.k FROM t WHERE ((t.a = 1 AND t.b = 2))"


@pytest.mark.parametrize("sql", [
    "SELECT t.k FROM t WHERE t.a IN (1, 2)",  # not a row
    "SELECT t.k FROM t WHERE (t.a, t.b) IN (SELECT x.a, x.b FROM t AS x)",
    "SELECT t.k FROM t WHERE (t.a, t.b) IN ((1, 2), (3, 4, 5))",  # widths differ
    "SELECT t.k FROM t WHERE (t.a + 1, t.b) IN ((1, 2))",  # only columns and literals are read
    "SELECT t.k FROM t WHERE (t.a, t.b) IN ((1, RANDOM()))",
    "SELECT t.k FROM t WHERE (t.a, t.b) IN ((1, (SELECT MAX(x.b) FROM t AS x)))",
    "SELECT t.k FROM t WHERE (t.a, t.b) = (1, 2)",  # not IN
    "SELECT t.k FROM t WHERE EXISTS (SELECT 1 FROM t AS x WHERE (x.a, x.b) IN ((1, 2)))",  # another select's test
])
def test_tests_left_alone(sql):
    assert expanded(sql) is None


@pytest.mark.parametrize("dialect", ["duckdb", "bigquery", "oracle", None])
def test_other_dialects_are_not_read(dialect):
    assert expand_row_membership(sqlglot.parse_one("SELECT t.k FROM t WHERE (t.a, t.b) IN ((1, 2), (2, 1))"), dialect) is None


def test_proofs_and_near_misses():
    rows = "SELECT t.k FROM t WHERE (t.a, t.b) IN ((1, 2), (2, 1)) AND t.c > 0"
    spelled = "SELECT t.k FROM t WHERE ((t.a = 1 AND t.b = 2) OR (t.a = 2 AND t.b = 1)) AND t.c > 0"
    assert proved(rows, spelled) and proved(spelled, rows)
    assert proved("SELECT t.k FROM t WHERE (t.a, t.b) NOT IN ((1, 2))", "SELECT t.k FROM t WHERE t.a <> 1 OR t.b <> 2")

    # the second pair is not swapped
    other = "SELECT t.k FROM t WHERE ((t.a = 1 AND t.b = 2) OR (t.a = 2 AND t.b = 2)) AND t.c > 0"
    first, second = rows_on_duckdb([rows, other], [(1, 2, 1, 5)])
    assert first != second
    assert not proved(rows, other) and not proved(other, rows)
    # each field on its own list accepts more rows
    loose = "SELECT t.k FROM t WHERE t.a IN (1, 2) AND t.b IN (1, 2) AND t.c > 0"
    first, second = rows_on_duckdb([rows, loose], [(1, 1, 1, 5)])
    assert first != second
    assert not proved(rows, loose) and not proved(loose, rows)


def test_the_rewrite_returns_the_same_rows():
    # in a WHERE a NULL test and a FALSE one both drop the row, so DuckDB's NULL-equal rows cannot show here
    sql = "SELECT t.k, t.a FROM t WHERE (t.a, t.b) IN ((1, 2), (2, 1), (0, 0)) OR (t.c, t.a, t.k) IN ((3, 3, 1))"
    after = expanded(sql)
    assert after is not None and "IN" not in after
    rng = random.Random(3)
    value = lambda: rng.choice([None, 0, 1, 2, 3])
    for _ in range(40):
        rows = [tuple(value() for _ in range(4)) for _ in range(rng.randint(0, 7))]
        first, second = rows_on_duckdb([sql, after], rows)
        assert first == second, rows
    # NOT IN, on rows without NULLs
    sql = "SELECT t.k FROM t WHERE (t.a, t.b) NOT IN ((1, 2), (2, 1))"
    after = expanded(sql)
    for _ in range(20):
        rows = [tuple(rng.choice([0, 1, 2]) for _ in range(4)) for _ in range(rng.randint(0, 7))]
        first, second = rows_on_duckdb([sql, after], rows)
        assert first == second, rows
