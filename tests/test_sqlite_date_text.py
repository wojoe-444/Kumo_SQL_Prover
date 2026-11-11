"""SQLite's STRFTIME digits read as text or as a number (``kumo_sql_prover.sqlite_date_text``)."""

import random
import sqlite3

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.sqlite_date_text import sqlite_date_text_rules

SCHEMA = {"t": ["id", "d", "e"]}
Y, M = "STRFTIME('%Y', d)", "STRFTIME('%m', d)"


def rewritten(sql, dialect="sqlite"):
    return sqlite_date_text_rules(sqlglot.parse_one(sql, read=dialect), dialect).sql(dialect=dialect)


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, compare_names=False, dialect="sqlite").proven


@pytest.mark.parametrize("test, expected", [
    (f"CAST({Y} AS INTEGER) = 1945", f"{Y} = '1945'"),
    (f"CAST({Y} AS INTEGER) <> 45", f"{Y} <> '0045'"),
    (f"7 >= CAST({M} AS INTEGER)", f"'07' >= {M}"),
    (f"(CAST(({M}) AS INTEGER)) < 12", f"{M} < '12'"),
    (f"CAST({Y} AS INTEGER) BETWEEN 1994 AND 1995", f"{Y} BETWEEN '1994' AND '1995'"),
    (f"CAST(STRFTIME('%d', d) AS INTEGER) IN (1, 15, 31)", "STRFTIME('%d', d) IN ('01', '15', '31')"),
    (f"CAST(STRFTIME('%j', d) AS INTEGER) > 0", "STRFTIME('%j', d) > '000'"),
    (f"{Y} - STRFTIME('%Y', e) = 20", f"CAST({Y} AS INTEGER) - CAST(STRFTIME('%Y', e) AS INTEGER) = 20"),
    (f"2011 - {Y} > id", f"2011 - CAST({Y} AS INTEGER) > id"),
    (f"({M}) * 2 + STRFTIME('%H', d) / 3 = 1", f"CAST({M} AS INTEGER) * 2 + CAST(STRFTIME('%H', d) AS INTEGER) / 3 = 1"),
])
def test_the_rewrite(test, expected):
    assert rewritten(f"SELECT id FROM t WHERE {test}") == f"SELECT id FROM t WHERE {expected}"


@pytest.mark.parametrize("test", [
    f"CAST({Y} AS INTEGER) = 19450",  # too long for the width: never equal, but '19450' is not what says so
    f"CAST({M} AS INTEGER) < 100",
    f"CAST({Y} AS INTEGER) = -1",
    f"CAST({Y} AS INTEGER) = 1945.0",
    f"CAST({Y} AS INTEGER) = '1945'",
    f"CAST({Y} AS INTEGER) = id",
    f"CAST({Y} AS REAL) = 1945",
    f"CAST({Y} AS TEXT) = 1945",
    f"{Y} = 1945",  # text against an integer: SQLite orders them by type
    f"{Y} < 1997",
    "CAST(STRFTIME('%Y-%m', d) AS INTEGER) = 1945",
    "CAST(STRFTIME('%J', d) AS INTEGER) = 2459000",  # a fraction
    "CAST(STRFTIME('%s', d) AS INTEGER) = 0",  # no fixed width
    "CAST(STRFTIME('%f', d) AS INTEGER) = 1",
    "CAST(STRFTIME('%Y', d, '+1 day') AS INTEGER) = 1945",
    "CAST(SUBSTR(d, 1, 4) AS INTEGER) = 1945",  # any text, not digits
    f"CAST({Y} AS INTEGER) BETWEEN 1994 AND 19950",
    f"CAST({Y} AS INTEGER) BETWEEN 1994 AND id",
    f"CAST({Y} AS INTEGER) IN (1994, id)",
    f"CAST({Y} AS INTEGER) IN (SELECT id FROM t)",
    "STRFTIME('%Y-%m', d) - 1 = 0",
    "SUBSTR(d, 1, 4) - SUBSTR(e, 1, 4) = 0",  # '1.5' is a number too
    f"-{Y} = 0",
])
def test_declined(test):
    sql = f"SELECT id FROM t WHERE {test}"
    assert rewritten(sql) == sqlglot.parse_one(sql, read="sqlite").sql(dialect="sqlite")


def test_other_dialects_are_left_alone():
    for dialect in ("bigquery", "duckdb", "mysql", "postgres"):
        tree = sqlglot.parse_one(f"SELECT id FROM t WHERE CAST({Y} AS INTEGER) = 1945", read="sqlite")
        assert sqlite_date_text_rules(tree.copy(), dialect) == tree


@pytest.mark.parametrize("left, right", [
    (f"SELECT id FROM t WHERE CAST({Y} AS INTEGER) = 1945", f"SELECT id FROM t WHERE {Y} = '1945'"),
    (f"SELECT id FROM t WHERE CAST({M} AS INTEGER) = 2", f"SELECT id FROM t WHERE {M} = '02'"),
    (f"SELECT id FROM t WHERE {Y} - STRFTIME('%Y', e) = 20", f"SELECT id FROM t WHERE CAST({Y} AS INTEGER) - CAST(STRFTIME('%Y', e) AS INTEGER) = 20"),
    (f"SELECT 2011 - {Y} FROM t", f"SELECT 2011 - CAST({Y} AS INTEGER) FROM t"),
])
def test_proved(left, right):
    assert proved(left, right)


@pytest.mark.parametrize("left, right", [
    (f"SELECT id FROM t WHERE CAST({M} AS INTEGER) = 2", f"SELECT id FROM t WHERE {M} = '2'"),  # never '2': the month is '02'
    (f"SELECT id FROM t WHERE CAST({Y} AS INTEGER) = 1945", f"SELECT id FROM t WHERE {Y} = '1946'"),
    (f"SELECT id FROM t WHERE CAST({Y} AS INTEGER) = 1945", f"SELECT id FROM t WHERE STRFTIME('%Y', e) = '1945'"),
    (f"SELECT id FROM t WHERE CAST({Y} AS INTEGER) = 1945", f"SELECT id FROM t WHERE {M} = '1945'"),
    (f"SELECT id FROM t WHERE CAST({Y} AS INTEGER) < 1997", f"SELECT id FROM t WHERE {Y} < 1997"),  # the right side is never true
    (f"SELECT id FROM t WHERE {Y} = '1945'", f"SELECT id FROM t WHERE {Y} = 1945"),
    (f"SELECT id FROM t WHERE {Y} - STRFTIME('%Y', e) = 20", f"SELECT id FROM t WHERE CAST({Y} AS INTEGER) - CAST(STRFTIME('%m', e) AS INTEGER) = 20"),
    ("SELECT id FROM t WHERE SUBSTR(d, 1, 4) - 1 = 0", "SELECT id FROM t WHERE CAST(SUBSTR(d, 1, 4) AS INTEGER) - 1 = 0"),
])
def test_not_proved(left, right):
    assert not proved(left, right)


def test_the_rewrites_keep_the_rows_in_sqlite():
    """Run in SQLite itself, over dates that include a year before 0, year 9999, times, and text that is no date."""

    rng = random.Random(17)
    dates = [None, "1945-03-07", "1945-12-31 23:59:59", "2016-02-29", "0045-01-01", "0000-06-15", "9999-12-31", "-0100-06-01", "-4000-01-01", "1997-01-01", "1996-12-31 00:00:00", "2000-10-10 07:08:09", "not a date", "", "2459000.5"]
    tests = []
    for specifier, numbers in (("%Y", [0, 45, 1945, 1996, 1997, 2016, 9999]), ("%m", [0, 1, 2, 7, 10, 12, 99]), ("%d", [1, 7, 15, 31]), ("%H", [0, 7, 23]), ("%M", [0, 8, 59]), ("%S", [0, 9, 59]), ("%j", [1, 60, 66, 366])):
        for number in numbers:
            for operator in ("=", "<>", "<", "<=", ">", ">="):
                tests.append(f"CAST(STRFTIME('{specifier}', d) AS INTEGER) {operator} {number}")
                tests.append(f"{number} {operator} CAST(STRFTIME('{specifier}', d) AS INTEGER)")
            tests.append(f"CAST(STRFTIME('{specifier}', d) AS INTEGER) BETWEEN {number} AND {numbers[-1]}")
            tests.append(f"CAST(STRFTIME('{specifier}', d) AS INTEGER) IN ({number}, {numbers[0]})")
        for operator in ("+", "-", "*", "/", "%"):
            tests.append(f"STRFTIME('{specifier}', d) {operator} STRFTIME('{specifier}', e)")
            tests.append(f"7 {operator} STRFTIME('{specifier}', d)")
            tests.append(f"STRFTIME('{specifier}', d) {operator} id")
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE t (id INTEGER, d TEXT, e TEXT)")
    db.executemany("INSERT INTO t VALUES (?, ?, ?)", [(rng.choice([None, 0, 1, 2, 5]), a, rng.choice(dates)) for a in dates for _ in range(4)])
    changed = 0
    for test in tests:
        before = f"SELECT id, d, e, {test}, TYPEOF({test}) FROM t"
        after = rewritten(before)
        changed += before != after
        assert sorted(map(repr, db.execute(before).fetchall())) == sorted(map(repr, db.execute(after).fetchall())), test
    assert changed == len(tests)
    db.close()
