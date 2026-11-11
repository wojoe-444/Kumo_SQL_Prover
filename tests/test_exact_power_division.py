"""An integer divided by a power of ten under ROUND(.., 0) is an exact product (``kumo_sql_prover.exact_power_division``)."""

import random
from decimal import ROUND_DOWN, Decimal

import pytest
import sqlglot
from sqlglot import exp

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.exact_power_division import divide_by_power_of_ten

# t: a and b are integers, c is an exact decimal, d is a binary float, s is text
SCHEMA = {"t": ["a", "b", "c", "d", "s"], "u": ["a", "e"]}
TYPES = {"t": {"a": "INT", "b": "BIGINT", "c": "DECIMAL(10, 2)", "d": "DOUBLE", "s": "VARCHAR"}, "u": {"a": "INT", "e": "INT"}}


def rewritten(sql, types=TYPES, dialect="mysql"):
    tree = sqlglot.parse_one(sql, read=dialect)
    text = tree.sql(dialect=dialect)
    for select in list(tree.find_all(exp.Select)):
        divide_by_power_of_ten(select, types, dialect)
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
    return [sorted(repr(tuple(None if v is None else Decimal(str(v)).normalize() + 0 for v in row)) for row in result) for result in results]


def test_the_division_becomes_a_product():
    assert rewritten("SELECT ROUND(a - a * 24 / 100) FROM t") == "SELECT ROUND(a - a * 24 * 0.01) FROM t"
    assert rewritten("SELECT ROUND(a - (24 / 100) * a, 0) FROM t") == "SELECT ROUND(a - (24 * 0.01) * a, 0) FROM t"
    assert rewritten("SELECT ROUND((a / 100) * 51 + (a + b) / 10 - -b / 1000 * (b / 10000)) FROM t AS x") == (
        "SELECT ROUND((a * 0.01) * 51 + (a + b) * 0.1 - -b * 0.001 * (b * 0.0001)) FROM t AS x"
    )
    # through the value arms of a CASE and an IF, not through their conditions
    assert rewritten("SELECT ROUND(CASE WHEN b / 100 > 1 THEN a / 100 WHEN b > 0 THEN c - a * (1 - 49 / 100) END) FROM t") == (
        "SELECT ROUND(CASE WHEN b / 100 > 1 THEN a * 0.01 WHEN b > 0 THEN c - a * (1 - 49 * 0.01) END) FROM t"
    )
    assert rewritten("SELECT ROUND(IF(b > 0, t.a / 10, 0), 0) FROM t JOIN u ON u.a = t.a") == "SELECT ROUND(IF(b > 0, t.a * 0.1, 0), 0) FROM t JOIN u ON u.a = t.a"


@pytest.mark.parametrize("sql", [
    "SELECT a / 100 FROM t",  # not under a ROUND: the declared scale shows
    "SELECT ROUND(a / 100, 1) FROM t",  # rounded to a decimal
    "SELECT ROUND(a / 100, 2) FROM t",
    "SELECT ROUND(a / 100, b) FROM t",
    "SELECT ROUND(a / 3) FROM t",  # cut after nine digits
    "SELECT ROUND(a / 50) FROM t",
    "SELECT ROUND(a / 100000) FROM t",
    "SELECT ROUND(a / 100.0) FROM t",  # a DECIMAL divisor
    "SELECT ROUND(a / 1e2) FROM t",  # a DOUBLE divisor
    "SELECT ROUND(a / '100') FROM t",
    "SELECT ROUND(a / b) FROM t",
    "SELECT ROUND(c / 100) FROM t",  # a DECIMAL dividend
    "SELECT ROUND(d / 100) FROM t",
    "SELECT ROUND(s / 100) FROM t",
    "SELECT ROUND(ABS(a) / 100) FROM t",  # a function: its type is not read
    "SELECT ROUND(z / 100) FROM t",  # no declared type
    "SELECT ROUND(a / 100 / 3) FROM t",  # below another division: its declared scale counts there
    "SELECT ROUND(3 / (a / 100)) FROM t",
    "SELECT ROUND(a / 100 % 7) FROM t",
    "SELECT ROUND(ABS(a / 100)) FROM t",  # below a function
    "SELECT ROUND(COALESCE(a / 100, 0)) FROM t",
    "SELECT ROUND(a / 100 + d) FROM t",  # the sum is a DOUBLE
    "SELECT ROUND(CASE WHEN b > 0 THEN a / 100 ELSE s END) FROM t",  # the CASE is text
    "SELECT ROUND(x.a / 100) FROM (SELECT c AS a FROM t) AS x",  # a derived table's column: here a decimal
    "SELECT ROUND(a / 100) FROM t JOIN u ON u.e = t.b",  # a: t's or u's
    "SELECT ROUND(SUM(a / 100)) FROM t",  # below an aggregate
    "SELECT TRUNCATE(a / 100, 0) FROM t",
])
def test_other_divisions_stay(sql):
    assert rewritten(sql) is None


def test_only_on_mysql_and_with_types():
    sql = "SELECT ROUND(a / 100) FROM t"
    assert rewritten(sql) == "SELECT ROUND(a * 0.01) FROM t"
    assert rewritten(sql, dialect="bigquery") is None  # BigQuery's / is a FLOAT64
    assert rewritten(sql, dialect="duckdb") is None
    assert rewritten(sql, types=None) is None
    assert rewritten(sql, types={"t": {"a": "DECIMAL(10, 2)"}}) is None
    assert rewritten(sql, types={"t": {"a": "VARCHAR(10)"}}) is None


def test_the_arithmetic_the_rule_rests_on():
    """An integer divided by 10, 100, 1000 or 10000 and cut after nine digits is the exact quotient; by another
    divisor, or a power of ten past the nine digits, it is not."""

    def cut(x):
        return x.quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)

    rng = random.Random(11)
    for a in list(range(-200, 201)) + [rng.randrange(-10**12, 10**12) for _ in range(2000)]:
        for k in range(1, 5):
            assert cut(Decimal(a) / Decimal(10**k)) == Decimal(a) * Decimal(1).scaleb(-k)
    assert cut(Decimal(1) / Decimal(3)) * 3 != 1
    assert cut(Decimal(1) / Decimal(10**10)) != Decimal(1).scaleb(-10)


PROOFS = [
    ("SELECT ROUND(a - a * 24 / 100) FROM t", "SELECT ROUND(a * 0.76, 0) FROM t"),
    ("SELECT ROUND(a - (24 / 100) * a) FROM t", "SELECT ROUND(a * (1 - 0.24)) FROM t"),
    ("SELECT ROUND((a / 100) * 51) FROM t", "SELECT ROUND(0.51 * a) FROM t"),
    (
        "SELECT CASE WHEN b > 10 THEN ROUND(a * (100 - 49) / 100) WHEN b > 1 THEN ROUND(a - a * (24 / 100)) ELSE a END FROM t",
        "SELECT ROUND(CASE WHEN b > 10 THEN a * 0.51 WHEN b > 1 THEN a * 0.76 ELSE a END, 0) FROM t",
    ),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    ("SELECT ROUND(a - a * 24 / 100) FROM t", "SELECT ROUND(a * 0.75, 0) FROM t"),
    ("SELECT ROUND(a * 24 / 100, 1) FROM t", "SELECT ROUND(a * 0.24, 1) FROM t"),  # not read: a ROUND to a decimal
    ("SELECT a * 24 / 100 FROM t", "SELECT a * 0.24 FROM t"),  # four declared decimals and two
    ("SELECT ROUND(a / 3) FROM t", "SELECT ROUND(a * 0.333333333) FROM t"),
    ("SELECT ROUND(c / 100) FROM t", "SELECT ROUND(c * 0.01) FROM t"),
    ("SELECT ROUND(a / 100 / 3) FROM t", "SELECT ROUND(a * 0.01 / 3) FROM t"),
    ("SELECT ROUND(a / 100) FROM t", "SELECT ROUND(a / 1000) FROM t"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


QUERIES = [
    "SELECT ROUND(a - a * 24 / 100) FROM t",
    "SELECT ROUND(a - (24 / 100) * a, 0) FROM t",
    "SELECT ROUND((a / 100) * 51 + (a + b) / 10 - -b / 1000 * (b / 10000)) FROM t AS x",
    "SELECT ROUND(CASE WHEN b / 100 > 1 THEN a / 100 WHEN b > 0 THEN c - a * (1 - 49 / 100) END) FROM t",
    "SELECT ROUND(IF(b > 0, t.a / 10, 0), 0) FROM t JOIN u ON u.a = t.a",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_numbers(sql):
    """The written query and the rewritten one on random rows with NULLs and duplicates. DuckDB divides in binary
    arithmetic, so the values are chosen to keep every result away from a rounding half."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    integer = lambda: rng.choice([None, -7, 0, 1, 3, 3, 12, 33, 101, 240, 999])  # noqa: E731
    exact = lambda: rng.choice([None, Decimal("0.25"), Decimal("1.75"), Decimal("-2.25"), Decimal("7")])  # noqa: E731
    for _ in range(150):
        tables = {
            "t": [(integer(), integer(), exact(), None, None) for _ in range(rng.choice([0, 1, 2, 4, 7]))],
            "u": [(integer(), integer()) for _ in range(rng.choice([0, 1, 3]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
