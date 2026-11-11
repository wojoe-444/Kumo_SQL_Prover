"""A strict bound on a DATE or integer column is the non-strict bound one step in (``kumo_sql_prover.discrete_bounds``)."""

import datetime
import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.discrete_bounds import close_strict_bounds, fold_chained_dates

SCHEMA = {"s": ["k", "n", "d", "q", "ts", "name"]}
TYPES = {"s": {"k": "INT", "n": "BIGINT", "d": "DATE", "q": "DECIMAL(15, 2)", "ts": "TIMESTAMP", "name": "VARCHAR(10)"}}


def closed(sql, dialect="postgres"):
    out = close_strict_bounds(sqlglot.parse_one(sql, read=dialect), TYPES, dialect)
    return out.sql(dialect=dialect) if out is not None else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


def rows_on_duckdb(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE s (k BIGINT, n BIGINT, d DATE, q DECIMAL(15, 2), ts TIMESTAMP, name VARCHAR)")
    for row in rows:
        db.execute("INSERT INTO s VALUES (?, ?, ?, ?, ?, ?)", list(row))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def test_the_bound_closes():
    assert closed("SELECT s.k FROM s WHERE s.n < 5") == "SELECT s.k FROM s WHERE s.n <= 4"
    assert closed("SELECT s.k FROM s WHERE s.n > 5 AND 0 < n AND 3 > s.k") == "SELECT s.k FROM s WHERE s.n >= 6 AND 1 <= n AND 2 >= s.k"
    assert closed("SELECT s.k FROM s WHERE s.n < 0 OR s.n > -1") == "SELECT s.k FROM s WHERE s.n <= -1 OR s.n >= 0"
    assert (
        closed("SELECT s.k FROM s WHERE s.d < CAST('1996-01-01' AS DATE) AND s.d > CAST('1995-02-28' AS DATE)")
        == "SELECT s.k FROM s WHERE s.d <= CAST('1995-12-31' AS DATE) AND s.d >= CAST('1995-03-01' AS DATE)"
    )
    # parentheses and quoted names
    assert closed('SELECT "k" FROM "s" WHERE "d" < (CAST(\'1996-01-01\' AS DATE)) AND ("n") > (3)') == 'SELECT "k" FROM "s" WHERE "d" <= CAST(\'1995-12-31\' AS DATE) AND "n" >= 4'
    # through a plain derived table, and in a join condition and a CASE
    assert closed("SELECT t.k FROM (SELECT s.k, s.n FROM s WHERE s.q > 0) AS t WHERE t.n < 5") == "SELECT t.k FROM (SELECT s.k, s.n FROM s WHERE s.q > 0) AS t WHERE t.n <= 4"
    assert (
        closed("SELECT CASE WHEN a.n < 2 THEN 1 END FROM s AS a JOIN s AS b ON a.k = b.k AND b.n > 7")
        == "SELECT CASE WHEN a.n <= 1 THEN 1 END FROM s AS a JOIN s AS b ON a.k = b.k AND b.n >= 8"
    )


@pytest.mark.parametrize("sql", [
    "SELECT s.k FROM s WHERE s.q < 5",  # a decimal has values between 4 and 5
    "SELECT s.k FROM s WHERE s.n < 4.5",
    "SELECT s.k FROM s WHERE s.n < '5'",
    "SELECT s.k FROM s WHERE s.n <= 5 AND s.n >= 1 AND s.n <> 3",  # nothing strict
    "SELECT s.k FROM s WHERE s.n + 1 < 5",  # not a plain column
    "SELECT s.k FROM s WHERE s.n < s.k",
    "SELECT s.k FROM s WHERE s.ts < CAST('1996-01-01' AS DATE)",  # a timestamp has times of day
    "SELECT s.k FROM s WHERE s.d < CAST('1996-01-01' AS TIMESTAMP)",
    "SELECT s.k FROM s WHERE s.d < '1996-01-01'",  # only a written DATE constant is read
    "SELECT s.k FROM s WHERE s.name < CAST('1996-01-01' AS DATE)",
    "SELECT s.k FROM s WHERE s.d < CAST('0001-01-01' AS DATE)",  # no day before it
    "SELECT u.k FROM unknown_table AS u WHERE u.n < 5",
    "SELECT t.k FROM (SELECT s.k, s.n + 0.5 AS n FROM s) AS t WHERE t.n < 5",  # computed: not whole
    "SELECT s.k, SUM(s.q) / 2 AS n FROM s GROUP BY s.k HAVING n < 5",  # could mean the output
    "SELECT s.k FROM s WHERE (SELECT COUNT(*) FROM s AS x WHERE x.q < 5) > 0",  # another select's bound, and no column here
])
def test_bounds_left_alone(sql):
    assert closed(sql) is None


@pytest.mark.parametrize("dialect", ["oracle", "sqlite", "tsql", None])
def test_other_dialects_are_not_read(dialect):
    assert close_strict_bounds(sqlglot.parse_one("SELECT s.k FROM s WHERE s.n < 5 AND s.d < CAST('1996-01-01' AS DATE)"), TYPES, dialect) is None


def test_chained_intervals_fold():
    tree = sqlglot.parse_one("SELECT s.k FROM s WHERE s.d <= DATE '1995-01-31' + INTERVAL '1 month' - INTERVAL '1 day' + INTERVAL '1 year'", read="postgres")
    assert fold_chained_dates(tree).sql(dialect="postgres") == "SELECT s.k FROM s WHERE s.d <= CAST('1996-02-27' AS DATE)"
    untouched = sqlglot.parse_one("SELECT s.k FROM s WHERE s.d <= s.d + INTERVAL '1 month' - INTERVAL '1 day'", read="postgres")
    assert fold_chained_dates(untouched).sql() == untouched.sql()


def test_proofs_and_near_misses():
    half_open = "SELECT SUM(s.q) FROM s WHERE s.d >= DATE '1995-01-01' AND s.d < DATE '1995-01-01' + INTERVAL '1 year'"
    between = "SELECT SUM(s.q) FROM s WHERE s.d BETWEEN DATE '1995-01-01' AND DATE '1995-12-31'"
    chained = "SELECT SUM(s.q) FROM s WHERE s.d BETWEEN DATE '1995-01-01' AND DATE '1995-01-01' + INTERVAL '1 year' - INTERVAL '1 day'"
    year = "SELECT SUM(s.q) FROM s WHERE EXTRACT(YEAR FROM s.d) = 1995"
    for other in (between, chained, year):
        assert proved(half_open, other) and proved(other, half_open)
    assert proved("SELECT s.k FROM s WHERE s.n > 3 AND s.n < 9", "SELECT s.k FROM s WHERE s.n BETWEEN 4 AND 8")
    assert proved("SELECT s.k FROM s WHERE s.n < 5", "SELECT s.k FROM s WHERE NOT s.n >= 5")  # the form without the rule still proves

    # one day short
    short = "SELECT SUM(s.q) FROM s WHERE s.d BETWEEN DATE '1995-01-01' AND DATE '1995-12-30'"
    first, second = rows_on_duckdb([half_open, short], [(1, 1, datetime.date(1995, 12, 31), 2, None, None)])
    assert first != second
    assert not proved(half_open, short) and not proved(short, half_open)
    # a decimal column has values between the bounds
    strict, loose = "SELECT s.k FROM s WHERE s.q < 5", "SELECT s.k FROM s WHERE s.q <= 4"
    first, second = rows_on_duckdb([strict, loose], [(1, 1, None, 4.5, None, None)])
    assert first != second
    assert not proved(strict, loose) and not proved(loose, strict)
    # a timestamp column has times of day
    strict, loose = "SELECT s.k FROM s WHERE s.ts < DATE '1996-01-01'", "SELECT s.k FROM s WHERE s.ts <= DATE '1995-12-31'"
    first, second = rows_on_duckdb([strict, loose], [(1, 1, None, None, datetime.datetime(1995, 12, 31, 12), None)])
    assert first != second
    assert not proved(strict, loose) and not proved(loose, strict)
    # off by one on an integer
    strict, loose = "SELECT s.k FROM s WHERE s.n < 5", "SELECT s.k FROM s WHERE s.n <= 5"
    first, second = rows_on_duckdb([strict, loose], [(1, 5, None, None, None, None)])
    assert first != second
    assert not proved(strict, loose) and not proved(loose, strict)


def test_the_rewrite_returns_the_same_rows():
    sql = (
        "SELECT s.k, CASE WHEN s.n > 1 THEN 1 ELSE 0 END, 2 > s.n FROM s "
        "WHERE (s.d < CAST('1995-03-01' AS DATE) OR s.d > CAST('1995-03-02' AS DATE) OR s.n < 0) AND NOT 3 < s.k"
    )
    after = closed(sql)
    assert after is not None and " < " not in after and " > " not in after
    rng = random.Random(11)
    days = [None, *(datetime.date(1995, 2, 27) + datetime.timedelta(days=i) for i in range(6))]
    for _ in range(40):
        rows = [(rng.choice([None, 1, 2, 3, 4]), rng.choice([None, -1, 0, 1, 2, 3]), rng.choice(days), None, None, None) for _ in range(rng.randint(0, 7))]
        first, second = rows_on_duckdb([sql, after], rows)
        assert first == second, rows
