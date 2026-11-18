"""A power of ten before or after a MySQL division of integers, under one ROUND (``kumo_sql_prover.exact_scaling``)."""

import random
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal

import pytest
import sqlglot
from sqlglot import exp

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.exact_scaling import scale_after_division

# t: a and b are integers, c is an exact decimal, d is a binary float
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
TYPES = {"t": {"a": "INT", "b": "BIGINT", "c": "DECIMAL(10, 2)", "d": "DOUBLE"}, "u": {"a": "INT", "e": "INT"}}


def rewritten(sql, types=TYPES, dialect="mysql"):
    tree = sqlglot.parse_one(sql, read=dialect)
    text = tree.sql(dialect=dialect)
    for select in list(tree.find_all(exp.Select)):
        scale_after_division(select, types, dialect)
    after = tree.sql(dialect=dialect)
    return after if after != text else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="mysql", exact_arithmetic=True).proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE t (a INTEGER, b BIGINT, c DECIMAL(10, 2), d DOUBLE)")
    db.execute("CREATE TABLE u (a INTEGER, e INTEGER)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    results = run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])
    return [sorted(repr(tuple(None if v is None else Decimal(str(v)).normalize() + 0 for v in row)) for row in result) for result in results]


def test_the_power_of_ten_moves_behind_the_division():
    assert rewritten("SELECT ROUND(100 * SUM(a = b) / COUNT(*), 2) FROM t") == "SELECT ROUND(SUM(a = b) / COUNT(*) * 100, 2) FROM t"
    assert rewritten("SELECT ROUND((SUM(a) * 100) / (COUNT(DISTINCT b)), 2) AS p FROM t") == "SELECT ROUND(SUM(a) / COUNT(DISTINCT b) * 100, 2) AS p FROM t"
    assert rewritten("SELECT ROUND(10 * (a + b) / b) FROM t") == "SELECT ROUND((a + b) / b * 10) FROM t"
    assert rewritten("SELECT a, ROUND(1000 * SUM(CASE WHEN b > 1 THEN 1 ELSE 0 END) / COUNT(b), 4) FROM t GROUP BY a") == (
        "SELECT a, ROUND(SUM(CASE WHEN b > 1 THEN 1 ELSE 0 END) / COUNT(b) * 1000, 4) FROM t GROUP BY a"
    )
    assert rewritten("SELECT ROUND(10000 * SUM(IF(x.b IS NULL, x.a, -x.a)) / (SUM(x.a) - 1), 0) FROM t AS x") == (
        "SELECT ROUND(SUM(IF(x.b IS NULL, x.a, -x.a)) / (SUM(x.a) - 1) * 10000, 0) FROM t AS x"
    )


@pytest.mark.parametrize("sql", [
    "SELECT ROUND(SUM(a = b) / COUNT(*) * 100, 2) FROM t",  # already in the rule's form
    "SELECT ROUND(100.0 * SUM(a = b) / COUNT(*), 2) FROM t",  # the product has a declared scale of one digit
    "SELECT ROUND(1e2 * SUM(a = b) / COUNT(*), 2) FROM t",  # a DOUBLE
    "SELECT ROUND('100' * SUM(a = b) / COUNT(*), 2) FROM t",
    "SELECT ROUND(3 * SUM(a) / COUNT(*), 2) FROM t",  # 3 times a cut quotient is not the cut of the product
    "SELECT ROUND(50 * SUM(a) / COUNT(*), 2) FROM t",
    "SELECT ROUND(100000 * SUM(a) / COUNT(*), 2) FROM t",  # more than four digits of the nine
    "SELECT ROUND(100 * SUM(a) / COUNT(*), 5) FROM t",  # more decimals than the declared scale
    "SELECT ROUND(100 * SUM(a) / COUNT(*), -1) FROM t",
    "SELECT ROUND(100 * SUM(a) / COUNT(*), b) FROM t GROUP BY b",
    "SELECT ROUND(100 * SUM(c) / COUNT(*), 2) FROM t",  # a DECIMAL sum
    "SELECT ROUND(100 * SUM(a) / SUM(c), 2) FROM t",
    "SELECT ROUND(100 * SUM(d) / COUNT(*), 2) FROM t",  # a DOUBLE sum
    "SELECT ROUND(100 * AVG(a) / COUNT(*), 2) FROM t",  # the mean of integers is a DECIMAL
    "SELECT ROUND(100 * SUM(a / 2) / COUNT(*), 2) FROM t",
    "SELECT ROUND(100 * ABS(a) / b, 2) FROM t",  # a function: its type is not read
    "SELECT ROUND(100 * a / (SELECT COUNT(*) FROM u), 2) FROM t",
    "SELECT ROUND(100 * z / b, 2) FROM t",  # no declared type
    "SELECT ROUND(100 * x.a / x.b, 2) FROM (SELECT c AS a, b FROM t) AS x",  # a derived table's column: here a decimal
    "SELECT ROUND(100 * a / e, 2) FROM t JOIN u ON u.e = t.b",  # a: t's or u's
    "SELECT 100 * SUM(a) / COUNT(*) FROM t",  # not under a ROUND: the nine digits show
    "SELECT ROUND(100 * SUM(a) / COUNT(*) + 1, 2) FROM t",
    "SELECT ROUND(100 * SUM(a) / COUNT(*) / 2, 2) FROM t",
    "SELECT ROUND(100 * SUM(a) DIV COUNT(*), 2) FROM t",
    "SELECT ROUND(100 + SUM(a) / COUNT(*), 2) FROM t",
    "SELECT TRUNCATE(100 * SUM(a) / COUNT(*), 2) FROM t",
    "SELECT 100 * ROUND(SUM(a) / COUNT(*), 4) FROM t",
])
def test_other_expressions_stay(sql):
    assert rewritten(sql) is None


def test_only_on_mysql_and_with_types():
    sql = "SELECT ROUND(100 * a / b, 2) FROM t"
    assert rewritten(sql) == "SELECT ROUND(a / b * 100, 2) FROM t"
    assert rewritten(sql, dialect="bigquery") is None
    assert rewritten(sql, dialect="duckdb") is None
    assert rewritten(sql, types=None) is None
    assert rewritten(sql, types={"t": {"a": "DECIMAL(10, 2)", "b": "INT"}}) is None
    assert rewritten(sql, types={"t": {"a": "INT", "b": "FLOAT"}}) is None
    assert rewritten(sql, types={"t": {"a": "INT", "b": "VARCHAR(10)"}}) is None


def test_the_arithmetic_the_rule_rests_on():
    """The rule's reading of MySQL, spelled with exact decimals: a quotient of integers cut after nine digits, an
    exact product, ROUND half away from zero. Under it the two spellings agree for every power of ten and number of
    decimals the rule reads, and a factor or a number of decimals it refuses does separate them."""

    def cut(x):
        return x.quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)

    def rounded(x, d):
        return x.quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)

    def before(a, b, c, d):
        return rounded(cut(Decimal(c * a) / Decimal(b)), d)

    def after(a, b, c, d):
        return rounded(cut(Decimal(a) / Decimal(b)) * c, d)

    rng = random.Random(7)
    pairs = [(a, b) for b in range(1, 60) for a in range(-b - 3, b + 4)]
    pairs += [(rng.randrange(-10**6, 10**6), rng.randrange(1, 10**6)) for _ in range(3000)]
    pairs += [(1, 20100), (5, 100001), (1, 200001), (-1, 200001), (2, 3), (1, 7), (1, 8), (1, 16), (999999, 1000000)]
    for a, b in pairs:
        for c in (10, 100, 1000, 10000):
            for d in range(5):
                assert before(a, b, c, d) == after(a, b, c, d), (a, b, c, d)
    # what the rule refuses: a factor that is not a power of ten, and decimals past the digits both spellings keep
    assert before(1, 6, 3, 0) != after(1, 6, 3, 0)
    assert before(2, 3, 10000, 8) != after(2, 3, 10000, 8)


PROOFS = [
    ("SELECT ROUND(100 * SUM(a = b) / COUNT(*), 2) FROM t", "SELECT ROUND((SUM(a = b) / COUNT(*)) * 100, 2) FROM t"),
    ("SELECT ROUND(SUM(b = a) * 100 / COUNT(*), 2) FROM t", "SELECT ROUND(100 * (SUM(a = b) / COUNT(*)), 2) FROM t"),
    ("SELECT a, ROUND(10 * COUNT(b) / COUNT(*), 1) FROM t GROUP BY a", "SELECT a, ROUND(COUNT(b) / COUNT(*) * 10, 1) FROM t GROUP BY a"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


REFERENCE = "SELECT ROUND(100 * SUM(a = b) / COUNT(*), 2) FROM t"
NOT_PROVED = [
    # no row with both values: the left sum is NULL, the right one 0
    (REFERENCE, "SELECT ROUND(SUM(CASE WHEN a = b THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) FROM t"),
    (REFERENCE, "SELECT ROUND(AVG(a = b) * 100, 2) FROM t"),  # the mean skips the rows with a NULL
    (REFERENCE, "SELECT ROUND(SUM(a = b) / COUNT(a) * 100, 2) FROM t"),
    (REFERENCE, "SELECT ROUND(SUM(a = b) / COUNT(*) * 100, 1) FROM t"),
    (REFERENCE, "SELECT ROUND(SUM(a = b) / COUNT(*) * 10, 2) FROM t"),
    (REFERENCE, "SELECT ROUND(SUM(a = b) / COUNT(*), 2) * 100 FROM t"),  # rounded before the factor
    ("SELECT ROUND(3 * SUM(a) / COUNT(*), 2) FROM t", "SELECT ROUND(SUM(a) / COUNT(*) * 3, 2) FROM t"),
    ("SELECT ROUND(100 * SUM(c) / COUNT(*), 2) FROM t", "SELECT ROUND(SUM(c) / COUNT(*) * 100, 2) FROM t"),
    ("SELECT 100 * SUM(a) / COUNT(*) FROM t", "SELECT SUM(a) / COUNT(*) * 100 FROM t"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


# Each query's (factor, decimals) and its divisors (at most 7) keep the quotient away from a rounding half, where
# DuckDB's binary division could fall on either side.
QUERIES = [
    "SELECT ROUND(100 * SUM(CASE WHEN a = b THEN 1 WHEN a <> b THEN 0 END) / COUNT(*), 2) FROM t",
    "SELECT ROUND((SUM(a) * 100) / (COUNT(DISTINCT b)), 2) AS p FROM t",
    "SELECT a, ROUND(10 * COUNT(b) / COUNT(*), 1) FROM t GROUP BY a",
    "SELECT ROUND(1000 * (SUM(a) - COUNT(b)) / COUNT(a), 0) FROM t",
    "SELECT b, ROUND(100 * SUM(IF(a IS NULL, 1, -a)) / COUNT(*)) FROM t GROUP BY b",
    "SELECT ROUND(10000 * COUNT(DISTINCT a) / COUNT(b), 4) FROM t",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_numbers(sql):
    """The written query and the rewritten one on random rows with NULLs and duplicates, the empty table included
    (a zero divisor is NULL in both)."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, None, -2, 0, 1, 1, 2, 3, 5])  # noqa: E731
    for _ in range(150):
        tables = {"t": [(value(), value(), None, None) for _ in range(rng.choice([0, 1, 2, 3, 5, 7]))]}
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
