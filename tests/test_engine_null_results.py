"""A zero divisor and a failed date cast give NULL in some engines (engine_null_results.py).

The rows in ``MYSQL_ROWS`` are what MySQL 8.0 returned for the made-up table below; DuckDB and SQLite run here."""

import sqlite3

import pytest
import sqlglot

pytest.importorskip("z3")
duckdb = pytest.importorskip("duckdb")

from kumo_sql_prover import engine_null_results
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.output_properties import infer_properties
from kumo_sql_prover.smt_equivalence import TableConstraints

TYPES = {"lookups": {"word": "VARCHAR", "place": "INT", "score": "INT", "day": "DATE", "seen": "DATETIME", "ratio": "DOUBLE"}}
SCHEMA = {"lookups": list(TYPES["lookups"])}
FILLED = frozenset(SCHEMA["lookups"])
NOT_NULL = {"lookups": TableConstraints(not_null=FILLED)}
#: the divisor is declared to be 1 or more, as a VeriEQL schema declares a position
POSITIVE = {"lookups": TableConstraints(not_null=FILLED, checks=((">=", ("column", "place"), ("value", 1)),))}

NOTHING = "SELECT word FROM lookups WHERE 1 = 0"
EVERYTHING = "SELECT word FROM lookups"
ZERO = "SELECT word FROM lookups WHERE place = 0"

#: one row with a zero divisor, one without
ROWS = [("soon", 0, 5, "2024-01-31", "2024-01-31 10:00:00", 0.5), ("later", 2, 5, "2024-02-01", "2024-02-01 00:00:00", 2.0)]

REMAINDERS = {
    "never NULL": ("SELECT word FROM lookups WHERE score % place IS NULL", NOTHING),
    "always a value": ("SELECT word FROM lookups WHERE score % place IS NOT NULL", EVERYTHING),
    "counted": ("SELECT COUNT(score % place) AS n FROM lookups", "SELECT COUNT(*) AS n FROM lookups"),
    "a default that is never used": ("SELECT COALESCE(score % place, 7) AS r FROM lookups", "SELECT score % place AS r FROM lookups"),
    "the function": ("SELECT COALESCE(MOD(score, place), 7) AS r FROM lookups", "SELECT MOD(score, place) AS r FROM lookups"),
}
NULL_ON_ZERO = ("mysql", "sqlite", "duckdb")
ERROR_ON_ZERO = ("bigquery", "postgres")


def proven(pair, dialect, constraints=NOT_NULL) -> bool:
    return prove_equivalent_algebraic(
        *pair, schema=SCHEMA, types=TYPES, constraints=constraints, compare_names=False, dialect=dialect, exact_arithmetic=True, timeout_ms=4000
    ).proven


def on_duckdb(sql: str) -> list:
    con = duckdb.connect()
    con.execute("CREATE TABLE lookups (word VARCHAR, place INT, score INT, day DATE, seen TIMESTAMP, ratio DOUBLE)")
    con.executemany("INSERT INTO lookups VALUES (?, ?, ?, ?, ?, ?)", ROWS)
    return sorted(con.execute(sql).fetchall(), key=repr)


def on_sqlite(sql: str) -> list:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE lookups (word TEXT, place INT, score INT, day TEXT, seen TEXT, ratio REAL)")
    con.executemany("INSERT INTO lookups VALUES (?, ?, ?, ?, ?, ?)", ROWS)
    return sorted(con.execute(sql).fetchall(), key=repr)


@pytest.mark.parametrize("name", list(REMAINDERS))
@pytest.mark.parametrize("dialect", NULL_ON_ZERO)
def test_a_remainder_by_zero_is_null_where_it_is_not_an_error(name, dialect):
    assert not proven(REMAINDERS[name], dialect)


@pytest.mark.parametrize("name", list(REMAINDERS))
def test_duckdb_and_sqlite_tell_the_remainder_pairs_apart(name):
    left, right = REMAINDERS[name]
    assert on_duckdb(left) != on_duckdb(right)
    assert on_sqlite(left) != on_sqlite(right)


@pytest.mark.parametrize("name", list(REMAINDERS))
@pytest.mark.parametrize("dialect", ERROR_ON_ZERO)
def test_where_a_zero_modulus_is_an_error_the_remainder_is_never_null(name, dialect):
    # both queries fail on a zero divisor there, and a pair is compared where both run
    assert proven(REMAINDERS[name], dialect)


@pytest.mark.parametrize("dialect", NULL_ON_ZERO + ERROR_ON_ZERO)
def test_a_remainder_still_equals_itself(dialect):
    assert proven(("SELECT score % place AS r FROM lookups WHERE score > 1", "SELECT r FROM (SELECT score % place AS r, score FROM lookups) AS x WHERE score > 1"), dialect)


@pytest.mark.parametrize("dialect", NULL_ON_ZERO + ERROR_ON_ZERO)
@pytest.mark.parametrize("operation", ("score % 2", "MOD(score, 2)", "score / 2", "score / -100", "score % (7)"))
def test_no_engine_gives_null_for_a_divisor_written_as_a_whole_number(operation, dialect):
    assert proven((f"SELECT word FROM lookups WHERE {operation} IS NULL", NOTHING), dialect)
    assert on_duckdb(f"SELECT word FROM lookups WHERE {operation} IS NULL") == on_sqlite(f"SELECT word FROM lookups WHERE {operation} IS NULL") == []


@pytest.mark.parametrize("dialect", ("sqlite", "duckdb"))
@pytest.mark.parametrize("operation", ("score % 0", "score % 0.5", "score % ratio", "score / 0"))
def test_other_written_divisors_stay_unknown(operation, dialect):
    # SQLite: 5 % 0.5 is NULL (the divisor is cast to the integer 0)
    assert not proven((f"SELECT word FROM lookups WHERE {operation} IS NULL", NOTHING), dialect)


def test_which_divisors_are_whole_numbers_other_than_zero():
    def whole(sql: str) -> bool:
        return engine_null_results.nonzero_whole_literal(sqlglot.parse_one(sql))

    assert whole("2") and whole("-100") and whole("(7)") and whole("-(3)")
    assert not whole("0") and not whole("0.5") and not whole("2.0") and not whole("'2'") and not whole("x") and not whole("1 + 1") and not whole("1e3")


# ---- MySQL: a quotient or remainder by a number is NULL exactly when the number is zero ------------------

MYSQL_ZERO_ONLY = {
    "the quotient is NULL on a zero divisor": ("SELECT word FROM lookups WHERE score / place IS NULL", ZERO),
    "the remainder is NULL on a zero divisor": ("SELECT word FROM lookups WHERE score % place IS NULL", ZERO),
    "a quotient by a constant is never NULL": ("SELECT word FROM lookups WHERE score / 2 IS NULL", NOTHING),
    "a quotient by zero is always NULL": ("SELECT word FROM lookups WHERE score / 0 IS NULL", EVERYTHING),
    "the guarded and the plain quotient": ("SELECT score / place AS q FROM lookups", "SELECT score / NULLIF(place, 0) AS q FROM lookups"),
}
#: rows MySQL 8.0 returned for the left query of each pair over ROWS
MYSQL_ROWS = {
    "the quotient is NULL on a zero divisor": [("soon",)],
    "the remainder is NULL on a zero divisor": [("soon",)],
    "a quotient by a constant is never NULL": [],
    "a quotient by zero is always NULL": [("later",), ("soon",)],
}
MYSQL_WITH_A_POSITIVE_DIVISOR = {
    "a counted quotient": ("SELECT word, COUNT(score / place) AS n FROM lookups GROUP BY word", "SELECT word, COUNT(*) AS n FROM lookups GROUP BY word"),
    "a mean of quotients": ("SELECT word, AVG(score / place) AS q FROM lookups GROUP BY word", "SELECT word, SUM(score / place) / COUNT(*) AS q FROM lookups GROUP BY word"),
    "a rounded mean beside a share": (
        "SELECT word, ROUND(AVG(score / place), 2) AS q, ROUND(AVG(score < 3) * 100, 2) AS p FROM lookups GROUP BY word",
        "SELECT word, ROUND(SUM(score / place) / COUNT(*), 2) AS q, ROUND(SUM(IF(score < 3, 1, 0)) / COUNT(*) * 100, 2) AS p FROM lookups GROUP BY word",
    ),
    "a quotient that is never NULL": ("SELECT word FROM lookups WHERE score / place IS NULL", NOTHING),
}


@pytest.mark.parametrize("name", list(MYSQL_ZERO_ONLY))
def test_mysql_gives_null_exactly_on_a_zero_divisor(name):
    assert proven(MYSQL_ZERO_ONLY[name], "mysql")


@pytest.mark.parametrize("name", list(MYSQL_ROWS))
def test_sqlite_agrees_with_the_rows_mysql_returned(name):
    # SQLite's / by zero is NULL as MySQL's is (its % is not used: it casts to integers first)
    if "remainder" not in name:
        assert on_sqlite(MYSQL_ZERO_ONLY[name][0]) == MYSQL_ROWS[name]


@pytest.mark.parametrize("name", list(MYSQL_WITH_A_POSITIVE_DIVISOR))
def test_mysql_proves_these_only_with_a_divisor_declared_positive(name):
    assert proven(MYSQL_WITH_A_POSITIVE_DIVISOR[name], "mysql", POSITIVE)
    assert not proven(MYSQL_WITH_A_POSITIVE_DIVISOR[name], "mysql", NOT_NULL)


@pytest.mark.parametrize("dialect", ("sqlite", "duckdb", "postgres", "bigquery"))
def test_the_other_engines_keep_the_null_of_a_quotient_unknown(dialect):
    assert not proven(MYSQL_ZERO_ONLY["the quotient is NULL on a zero divisor"], dialect)
    if dialect in ("sqlite", "duckdb"):
        assert not proven(MYSQL_WITH_A_POSITIVE_DIVISOR["a counted quotient"], dialect, POSITIVE)


@pytest.mark.parametrize("divisor", ("word", "day", "seen"))
def test_a_divisor_that_is_not_a_number_stays_unknown_in_mysql(divisor):
    # MySQL reads a string as a number ('soon' is 0) and a date as its digits; neither is modeled
    assert not proven((f"SELECT word FROM lookups WHERE score / {divisor} IS NULL", NOTHING), "mysql")
    assert not proven((f"SELECT word FROM lookups WHERE score % {divisor} IS NULL", NOTHING), "mysql")


@pytest.mark.parametrize("dialect", NULL_ON_ZERO + ERROR_ON_ZERO)
def test_a_mean_is_still_its_sum_over_its_count(dialect):
    assert proven(("SELECT word, AVG(score) AS a FROM lookups GROUP BY word", "SELECT word, SUM(score) / COUNT(score) AS a FROM lookups GROUP BY word"), dialect)
    assert proven(("SELECT word, AVG(score) AS a FROM lookups GROUP BY word", "SELECT word, SUM(score) / COUNT(*) AS a FROM lookups GROUP BY word"), dialect)


# ---- DuckDB: x / 0 is infinity, x / NULLIF(0, 0) is NULL -------------------------------------------------


def test_duckdb_tells_the_guarded_quotient_from_the_plain_one():
    left, right = MYSQL_ZERO_ONLY["the guarded and the plain quotient"]
    assert on_duckdb(left) != on_duckdb(right)
    assert not proven((left, right), "duckdb")
    # with a divisor that is never zero the guard does nothing
    assert proven((left, right), "duckdb", POSITIVE)
    # SQLite's quotient is NULL on a zero divisor already
    assert on_sqlite(left) == on_sqlite(right)
    assert proven((left, right), "sqlite")


def test_duckdb_still_reads_a_mean_as_a_guarded_quotient():
    mean = "SELECT word, AVG(score) AS a FROM lookups GROUP BY word"
    guarded = "SELECT word, SUM(score) / NULLIF(COUNT(score), 0) AS a FROM lookups GROUP BY word"
    assert on_duckdb(mean) == on_duckdb(guarded)
    assert proven((mean, guarded), "duckdb")


# ---- MySQL: a cast to a date of a value that is not one is NULL ----------------------------------------

NOT_A_DATE = ("SELECT word FROM lookups WHERE CAST(word AS DATE) IS NULL", NOTHING)
DATES = (
    "CAST(day AS DATE)",
    "CAST(seen AS DATE)",
    "CAST(day AS DATETIME)",
    "CAST('2024-01-31' AS DATE)",
    "CAST('2024-01-31 10:00:00' AS DATE)",
    "CAST(CAST(seen AS DATE) AS DATETIME)",
    "CAST(COALESCE(day, seen) AS DATE)",
    "CAST(seen AS TIME)",
)
NOT_DATES = ("CAST(word AS DATE)", "CAST(score AS DATE)", "CAST(word AS DATETIME)", "CAST(CONCAT(word, '-01') AS DATE)", "CAST('soon' AS DATE)", "CAST(word AS TIME)", "CAST('10:00:00' AS TIME)")


def test_mysql_gives_null_for_a_cast_of_a_string_that_is_not_a_date():
    assert not proven(NOT_A_DATE, "mysql")
    for dialect in ("duckdb", "postgres", "bigquery"):
        assert proven(NOT_A_DATE, dialect), dialect  # the cast fails there, in both queries


@pytest.mark.parametrize("cast", DATES)
def test_mysql_casts_of_dates_are_never_null(cast):
    assert proven((f"SELECT word FROM lookups WHERE {cast} IS NULL", NOTHING), "mysql")


@pytest.mark.parametrize("cast", NOT_DATES)
def test_mysql_casts_of_other_values_may_be_null(cast):
    assert not proven((f"SELECT word FROM lookups WHERE {cast} IS NULL", NOTHING), "mysql")


def test_which_casts_may_be_null():
    declared = {"day": "DATE", "seen": "DATETIME", "word": "VARCHAR"}

    def may(sql: str) -> bool:
        return engine_null_results.mysql_cast_may_be_null(sqlglot.parse_one(sql, read="mysql"), lambda column: declared.get(column.name))

    assert not any(may(cast) for cast in DATES)
    assert all(may(cast) for cast in NOT_DATES)
    assert not may("CAST(word AS SIGNED)") and not may("CAST(word AS CHAR)") and not may("CAST(score AS DECIMAL(5, 2))")
    assert may("CAST(other AS DATE)")  # a column of unknown type
    assert may("CAST(IF(score > 1, day, word) AS DATE)") and not may("CAST(IF(score > 1, day, seen) AS DATE)")
    assert may("CAST(CASE WHEN score > 1 THEN day END AS DATE)") is False  # NULL in, NULL out
    assert may("CAST(MAX(word) AS DATE)") and not may("CAST(MIN(day) AS DATE)")


# ---- the columns reported as never NULL ------------------------------------------------------------------


def never_null(dialect: str) -> dict:
    sql = "SELECT score % place AS r, score / place AS q, SQRT(score) AS s, LN(score) AS l, CAST(word AS DATE) AS d, score + place AS a, ABS(score) AS b FROM lookups"
    result = infer_properties(sql, NOT_NULL, SCHEMA, dialect=dialect)
    return {column.name: column.non_null for column in result.columns}


def test_output_properties_follow_the_engine():
    everything = dict.fromkeys("rqsldab", True)
    assert never_null("bigquery") == everything
    assert never_null("postgres") == everything
    assert never_null("duckdb") == {**everything, "r": False, "q": False, "s": False, "l": False}
    assert never_null("sqlite") == {**everything, "r": False, "q": False, "s": False, "l": False}
    assert never_null("mysql") == {**everything, "r": False, "q": False, "s": False, "l": False, "d": False}
