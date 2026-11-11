"""MySQL ROUND over exact numbers: into the arms of a CASE, dropped over an integer (``kumo_sql_prover.exact_rounding``)."""

import random
from decimal import Decimal

import pytest
import sqlglot
from sqlglot import exp

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.exact_rounding import exact_rounding

# t: a and b are integers, c is an exact decimal, d is a binary float, s is text
SCHEMA = {"t": ["a", "b", "c", "d", "s"], "u": ["a", "e"]}
TYPES = {"t": {"a": "INT", "b": "BIGINT", "c": "DECIMAL(10, 2)", "d": "DOUBLE", "s": "VARCHAR"}, "u": {"a": "INT", "e": "INT"}}


def rewritten(sql, types=TYPES, dialect="mysql"):
    tree = sqlglot.parse_one(sql, read=dialect)
    text = tree.sql(dialect=dialect)
    for select in list(tree.find_all(exp.Select)):
        exact_rounding(select, types, dialect)
    after = tree.sql(dialect=dialect)
    return after if after != text else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="mysql", exact_arithmetic=True).proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE t (a INTEGER, b BIGINT, c DECIMAL(10, 2), d DOUBLE, s VARCHAR)")
    db.execute("CREATE TABLE u (a INTEGER, e INTEGER)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    results = run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])
    # compare numbers by value: 2 and 2.00 are one result
    return [sorted(repr(tuple(None if v is None else Decimal(str(v)).normalize() + 0 for v in row)) for row in result) for result in results]


def test_round_moves_into_exact_arms():
    assert rewritten("SELECT ROUND(CASE WHEN b > 100 THEN c * 0.5 WHEN b > 10 THEN a * 0.76 ELSE c END, 0) FROM t") == (
        "SELECT CASE WHEN b > 100 THEN ROUND(c * 0.5, 0) WHEN b > 10 THEN ROUND(a * 0.76, 0) ELSE ROUND(c, 0) END FROM t"
    )
    assert rewritten("SELECT ROUND(IF(b > 100, c - c * 0.24, c), 1) FROM t") == "SELECT IF(b > 100, ROUND(c - c * 0.24, 1), ROUND(c, 1)) FROM t"
    # a missing ELSE and a NULL arm stay NULL; a simple CASE; a nested CASE; division of exact numbers is exact arithmetic
    assert rewritten("SELECT ROUND(CASE a WHEN 1 THEN c / 3 WHEN 2 THEN NULL WHEN 3 THEN (CASE WHEN b > 0 THEN c ELSE -c END) END) FROM t") == (
        "SELECT CASE a WHEN 1 THEN ROUND(c / 3) WHEN 2 THEN NULL WHEN 3 THEN CASE WHEN b > 0 THEN ROUND(c) ELSE ROUND(-c) END END FROM t"
    )


def test_round_of_an_integer_is_the_integer():
    assert rewritten("SELECT ROUND(a), ROUND(x.b, 0), ROUND(a * 2 - b, 3) * 2, ROUND(-a), ROUND(7) FROM t AS x") == "SELECT a, x.b, (a * 2 - b) * 2, -a, 7 FROM t AS x"
    assert rewritten("SELECT ROUND(CASE WHEN b > 100 THEN a * 0.5 ELSE a END, 0) AS a FROM t") == "SELECT CASE WHEN b > 100 THEN ROUND(a * 0.5, 0) ELSE a END AS a FROM t"
    assert rewritten("SELECT CASE WHEN b > 100 THEN ROUND(a * 0.5) ELSE ROUND(a) END FROM t") == "SELECT CASE WHEN b > 100 THEN ROUND(a * 0.5) ELSE a END FROM t"
    # a column is typed through the sources of its own select
    assert rewritten("SELECT ROUND(t.a), ROUND(u.e) FROM t JOIN u ON u.a = t.a") == "SELECT t.a, u.e FROM t JOIN u ON u.a = t.a"


@pytest.mark.parametrize("sql", [
    "SELECT ROUND(c) FROM t",  # a decimal is rounded
    "SELECT ROUND(a * 0.5) FROM t",
    "SELECT ROUND(a / 2) FROM t",  # MySQL's / of integers is a decimal
    "SELECT ROUND(d) FROM t",
    "SELECT ROUND(s) FROM t",
    "SELECT ROUND(a, -1) FROM t",  # rounds to tens
    "SELECT ROUND(a, b) FROM t",
    "SELECT ROUND(CASE WHEN b > 0 THEN c ELSE d END) FROM t",  # a DOUBLE arm makes the CASE a DOUBLE
    "SELECT ROUND(CASE WHEN b > 0 THEN c ELSE 1e0 END) FROM t",
    "SELECT ROUND(CASE WHEN b > 0 THEN c ELSE s END) FROM t",
    "SELECT ROUND(CASE WHEN b > 0 THEN c ELSE ABS(c) END) FROM t",  # a function: its type is not read
    "SELECT ROUND(CASE WHEN b > 0 THEN c ELSE (SELECT MAX(e) FROM u) END) FROM t",
    "SELECT ROUND(IF(b > 0, c, d)) FROM t",
    "SELECT ROUND(COALESCE(c, a)) FROM t",
    "SELECT ROUND(z) FROM t",  # no declared type
    "SELECT ROUND(a) FROM t JOIN u ON u.e = t.b",  # a: t's or u's
    "SELECT ROUND(x.a) FROM (SELECT c AS a FROM t) AS x",  # a derived table's column: here a decimal
    "SELECT (SELECT ROUND(t.a + u.e) FROM u LIMIT 1) FROM t",  # t.a is an outer query's column
    "SELECT ROUND(CASE WHEN b > 0 THEN c ELSE a END, b) FROM t",
])
def test_other_calls_stay(sql):
    assert rewritten(sql) is None


def test_only_on_mysql_and_with_types():
    assert rewritten("SELECT ROUND(a) FROM t", dialect="bigquery") is None  # BigQuery: ROUND of an INT64 is a FLOAT64
    assert rewritten("SELECT ROUND(a) FROM t", types=None) is None
    assert rewritten("SELECT ROUND(a) FROM t", types={"t": {"a": "VARCHAR"}}) is None
    assert rewritten("SELECT ROUND(a) FROM t", types={"t": {"a": "DECIMAL(5, 1)"}}) is None
    assert rewritten("SELECT ROUND(a) FROM t", types={"t": {"a": "INTERVAL"}}) is None


PROOFS = [
    ("SELECT ROUND(CASE WHEN b > 100 THEN a * 0.5 ELSE a END, 0) FROM t", "SELECT CASE WHEN b > 100 THEN ROUND(a * 0.5) ELSE a END FROM t"),
    ("SELECT ROUND(CASE WHEN b > 100 THEN a * 0.5 ELSE a END, 0) FROM t", "SELECT CASE WHEN NOT b > 100 OR b IS NULL THEN ROUND(a, 0) ELSE ROUND(a - a * 0.5, 0) END FROM t"),
    ("SELECT ROUND(IF(b > 100, c, a), 1) FROM t", "SELECT CASE WHEN b > 100 THEN ROUND(c, 1) ELSE a END FROM t"),
    ("SELECT ROUND(a + b) FROM t", "SELECT a + b FROM t"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    ("SELECT ROUND(c) FROM t", "SELECT c FROM t"),  # a decimal is not its rounding
    ("SELECT ROUND(d) FROM t", "SELECT d FROM t"),
    ("SELECT ROUND(a / 2) FROM t", "SELECT a / 2 FROM t"),
    ("SELECT ROUND(a, -1) FROM t", "SELECT a FROM t"),
    ("SELECT ROUND(CASE WHEN b > 100 THEN a * 0.5 ELSE a END, 0) FROM t", "SELECT CASE WHEN b > 100 THEN a * 0.5 ELSE a END FROM t"),
    ("SELECT ROUND(CASE WHEN b > 100 THEN a * 0.5 ELSE a END, 0) FROM t", "SELECT CASE WHEN b >= 100 THEN ROUND(a * 0.5) ELSE a END FROM t"),
    # a NULL b takes the ELSE on the left and no arm on the right
    ("SELECT ROUND(CASE WHEN b > 100 THEN a * 0.5 ELSE a END, 0) FROM t", "SELECT CASE WHEN b <= 100 THEN ROUND(a, 0) WHEN b > 100 THEN ROUND(a - a * 0.5, 0) END FROM t"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


QUERIES = [
    "SELECT ROUND(CASE WHEN b > 2 THEN c * 0.5 WHEN b > 0 THEN a * 0.76 ELSE c END, 0) FROM t",
    "SELECT ROUND(IF(b > 1, c - c * 0.24, c), 1) FROM t",
    "SELECT ROUND(CASE a WHEN 1 THEN c * 0.3 WHEN 2 THEN NULL WHEN 3 THEN (CASE WHEN b > 0 THEN c ELSE -c END) END) FROM t",
    "SELECT ROUND(a), ROUND(x.b, 0), ROUND(a * 2 - b, 3) * 2, ROUND(-a), ROUND(7) FROM t AS x",
    "SELECT ROUND(CASE WHEN b > 1 THEN a * 0.5 ELSE a END, 0) AS a FROM t",
    "SELECT CASE WHEN b > 1 THEN ROUND(a * 0.25, 1) ELSE ROUND(a, 1) END FROM t",
    "SELECT ROUND(CASE WHEN b > 1 THEN a * 1.5 WHEN b = 1 THEN c ELSE a + b END, 1) FROM t",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_numbers(sql):
    """The written query and the rewritten one on random rows, with values that end in 5 where they are rounded (the
    halves are where a binary reading of a decimal would show)."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    integer = lambda: rng.choice([None, -3, -1, 0, 1, 1, 2, 3, 5, 7])  # noqa: E731
    exact = lambda: rng.choice([None, Decimal("0.5"), Decimal("1.5"), Decimal("2.5"), Decimal("-2.5"), Decimal("0.05"), Decimal("3.35"), Decimal("7"), Decimal("1.25")])  # noqa: E731
    for _ in range(150):
        tables = {"t": [(integer(), integer(), exact(), None, None) for _ in range(rng.choice([0, 1, 2, 4, 7]))]}
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
