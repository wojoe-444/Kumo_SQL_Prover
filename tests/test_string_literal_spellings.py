"""Strings an engine would read as numbers or as dates, compared with a value the prover reads as text (2026-11-16).

The provers read a column with no declared type as text, and a date column's strings as text too. Three ways that went wrong,
each proved before this check and each returning different rows on an engine:

- ``ORDERED``: MySQL compares an integer column with a string as numbers. ``'10'`` sorts before ``'9'`` and 10 is above 9, so
  ``n >= '9' AND n < '10'`` is empty for a text column and holds the integer 9 (MySQL 8.0.46, ``e(k INT, n INT, m INT)``).
- ``NUMBER_SPELLINGS``: DuckDB and PostgreSQL convert the string to the column's type, so ``n = '1' AND n = '01'`` holds the
  integer 1 (replayed on DuckDB below).
- ``DATE_SPELLINGS``: every engine converts a string compared with a date, and reads more than one spelling as one day.
  MySQL reads ``'20190101'``, ``'190101'``, ``'2019-1-1'`` and ``'2019/01/01'`` as the first of January 2019, for a
  declared ``DATE`` column as well (MySQL 8.0.46; DuckDB replayed below).
"""

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import numeric_column_reading, string_number_compare
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtStatus, prove_equivalent_smt
from kumo_sql_prover.string_number_compare import date_value

PROVERS = (prove_equivalent_algebraic, prove_equivalent_smt)
SCHEMA = {"e": ["k", "d", "ts", "n", "m"]}
DECLARED = {"e": {"k": "INT", "d": "DATE", "ts": "DATETIME", "n": "INT", "m": "INT"}}
TEXT = {"e": {"k": "INT", "d": "VARCHAR", "ts": "VARCHAR", "n": "VARCHAR", "m": "VARCHAR"}}
NONE = "SELECT e.k FROM e WHERE 1 = 0"
ENGINES = ("mysql", "duckdb", "postgres")


def where(condition: str) -> str:
    return "SELECT e.k FROM e WHERE " + condition


ORDERED = [
    (where("e.n >= '9' AND e.n < '10'"), NONE),
    (where("e.n BETWEEN '9' AND '10'"), NONE),
    (where("e.n = '9' AND e.n < '10'"), NONE),
    (where("e.n = '9' AND e.n > '10'"), where("e.n = '9'")),
    (where("e.n IN ('9', '10') AND e.n >= '9'"), where("e.n = '9'")),
    (where("e.n = '10' AND e.m = '9' AND e.n < e.m"), where("e.n = '10' AND e.m = '9'")),  # the order is asked of two columns
    (where("e.n = '10' AND e.m = '9' AND e.n > e.m"), NONE),
]
NUMBER_SPELLINGS = [
    (where("e.n = '1' AND e.n = '01'"), NONE),
    (where("e.n = '1' AND e.n <> '+1'"), where("e.n = '1'")),
    (where("e.n = '1' AND e.n = '1.0'"), NONE),
    (where("e.n IN ('1', '2') AND e.n NOT IN ('01')"), where("e.n IN ('1', '2')")),
]
DATE_SPELLINGS = [
    (where("e.d = '2019-01-01' AND e.d = '20190101'"), NONE),
    (where("e.d = '2019-02-01' AND e.d > '20190101'"), NONE),
    (where("e.d = '2019-01-01' AND e.d = '2019-1-1'"), NONE),
    (where("e.d = '2019-01-01' AND e.d <> '2019-01-01 00:00:00'"), where("e.d = '2019-01-01'")),
    (where("e.ts <= '2019-01-01' AND e.ts >= '2019-01-01 00:00:00'"), NONE),
    (where("DATE(e.ts) = '2019-01-01' AND DATE(e.ts) = '2019-1-1'"), NONE),
    (where("CAST(e.ts AS DATE) = '2019-01-01' AND CAST(e.ts AS DATE) <> '2019/01/01'"), where("CAST(e.ts AS DATE) = '2019-01-01'")),
    ("SELECT e.k FROM e GROUP BY e.k HAVING MIN(e.d) = '2019-01-01' AND MIN(e.d) = '20190101'", NONE),
]
# the same questions with strings that are equal and ordered as text exactly as they are as numbers or as dates
SAFE = [
    (where("e.d >= '2019-01-01' AND e.d <= '2019-03-31' AND e.d <> '2018-01-01'"), where("e.d >= '2019-01-01' AND e.d <= '2019-03-31'")),
    (where("e.d BETWEEN '2018-12-01' AND '2019-03-31'"), where("e.d >= '2018-12-01' AND e.d <= '2019-03-31'")),
    ("SELECT e.k FROM e GROUP BY e.k HAVING MIN(e.d) >= '2019-01-01' AND MIN(e.d) >= '2018-01-01'", "SELECT e.k FROM e GROUP BY e.k HAVING MIN(e.d) >= '2019-01-01'"),
    (where("e.ts >= '2019-01-01 00:00:00' AND e.ts >= '2018-01-01 00:00:00'"), where("e.ts >= '2019-01-01 00:00:00'")),
    (where("e.n >= '10' AND e.n >= '20'"), where("e.n >= '20'")),
    (where("e.n = '10' AND e.n = '20'"), NONE),
    (where("e.n > '5' AND e.n > '3'"), where("e.n > '5'")),
]


def status(prover, left, right, dialect, types=None):
    return prover(left, right, dialect=dialect, schema=SCHEMA, types=types, compare_names=False).status


@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("left,right", ORDERED + NUMBER_SPELLINGS)
def test_numbers_written_as_strings_are_not_read_as_text_for_a_column_of_no_type(prover, left, right):
    for dialect in ENGINES:
        assert status(prover, left, right, dialect) is not SmtStatus.PROVEN_EQUIVALENT, dialect
        assert status(prover, left, right, dialect, DECLARED) is not SmtStatus.PROVEN_EQUIVALENT, dialect
        assert status(prover, left, right, dialect, TEXT) is SmtStatus.PROVEN_EQUIVALENT, dialect  # a text column keeps its proofs
    # BigQuery refuses to compare a number column with a string, so there the column is not a number
    assert status(prover, left, right, "bigquery") is SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("left,right", DATE_SPELLINGS)
def test_one_day_written_two_ways_is_not_two_values(prover, left, right):
    for dialect in (*ENGINES, "bigquery"):
        assert status(prover, left, right, dialect) is not SmtStatus.PROVEN_EQUIVALENT, dialect
        assert status(prover, left, right, dialect, DECLARED) is not SmtStatus.PROVEN_EQUIVALENT, dialect


@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("left,right", SAFE)
def test_strings_that_read_the_same_way_as_text_keep_their_proofs(prover, left, right):
    for dialect in ENGINES:
        assert status(prover, left, right, dialect) is SmtStatus.PROVEN_EQUIVALENT, dialect
        assert status(prover, left, right, dialect, DECLARED) is SmtStatus.PROVEN_EQUIVALENT, dialect
        assert status(prover, left, right, dialect, TEXT) is SmtStatus.PROVEN_EQUIVALENT, dialect


@pytest.mark.parametrize("prover", PROVERS)
def test_a_declared_date_is_not_read_as_a_number(prover):
    # '2019-01-01' and '2019-03-31' are both 2019 for an integer column, and two days for a date
    left, right = where("e.d = '2019-01-01' AND e.d = '2019-03-31'"), NONE
    assert status(prover, left, right, "mysql", DECLARED) is SmtStatus.PROVEN_EQUIVALENT
    assert status(prover, left, right, "mysql") is not SmtStatus.PROVEN_EQUIVALENT  # no type: it may be an integer
    through = "SELECT e.k FROM e GROUP BY e.k HAVING MIN(e.d) >= '2019-01-01' AND MAX(e.d) <= '2019-03-31' AND MAX(e.d) <> '2019-04-01'"
    assert status(prover, through, through.replace(" AND MAX(e.d) <> '2019-04-01'", ""), "mysql", DECLARED) is SmtStatus.PROVEN_EQUIVALENT
    assert numeric_column_reading.reading(left, right, "mysql", DECLARED, SCHEMA) is None
    assert numeric_column_reading.reading(left, right, "mysql", None, SCHEMA).pair[0] == where("e.d = 2019 AND e.d = 2019")


def test_the_numeric_reading_is_asked_for_when_the_order_of_the_strings_is_not_the_order_of_the_numbers():
    crossed = numeric_column_reading.reading(where("e.n >= '9' AND e.n < '10'"), NONE, "mysql", None, SCHEMA)
    assert crossed.pair == (where("e.n >= 9 AND e.n < 10"), NONE)
    assert numeric_column_reading.reading(where("e.n >= '10' AND e.n < '20'"), NONE, "mysql", None, SCHEMA) is None  # the same order
    assert numeric_column_reading.reading(where("e.n = '9' OR e.n = '10'"), NONE, "mysql", None, SCHEMA) is None  # no ordering is asked
    by_minimum = "SELECT MIN(e.n) AS low FROM e WHERE e.n IN ('9', '10')"
    assert numeric_column_reading.reading(by_minimum, by_minimum, "mysql", None, SCHEMA).pair[0] == "SELECT MIN(e.n) AS low FROM e WHERE e.n IN (9, 10)"
    by_order = "SELECT e.n FROM e WHERE e.n IN ('9', '10') ORDER BY e.n LIMIT 1"
    assert numeric_column_reading.reading(by_order, by_order, "mysql", None, SCHEMA) is not None


@pytest.mark.parametrize(
    "text, value",
    [
        ("2019-01-01", (2019, 1, 1, 0, 0, 0)),
        ("2019-1-1", (2019, 1, 1, 0, 0, 0)),
        (" 2019-01-01 ", (2019, 1, 1, 0, 0, 0)),
        ("2019/01/01", (2019, 1, 1, 0, 0, 0)),
        ("2019-01-01 00:00:00", (2019, 1, 1, 0, 0, 0)),
        ("2019-03-31 23:59:59", (2019, 3, 31, 23, 59, 59)),
        ("20190101", "other"),
        ("190101", "other"),
        ("19-01-01", "other"),
        ("2019-01/01", "other"),
        ("2019-01-01abc", "other"),
        ("2019-01-01T00:00:00", "other"),
        ("2019-01-01 00:00:00.0", "other"),
        ("2019-01-01 10:00", "other"),
        ("2019-02-30", "other"),  # not a day: an error in one engine, a zero date in another
        ("0000-00-00", "other"),
        ("abc", None),
        ("", None),
        ("2019", None),
        ("12345", None),
        ("x2019-01-01", None),
        ("10:30:00", None),  # a time of day
    ],
)
def test_the_moment_a_date_string_is(text, value):
    assert date_value(text) == value


@pytest.mark.parametrize(
    "condition, declined",
    [
        ("e.d = '2019-01-01' AND e.d <> '2019-02-01'", False),
        ("e.d = '2019-01-01 00:00:00' AND e.d <> '2019-02-01 00:00:00'", False),
        ("e.d = '20190101'", False),  # one string: nothing to confuse it with
        ("e.d > '2019-1-1'", False),
        ("e.d = '2019-01-01' AND e.n = '2019-1-1'", False),  # two values
        ("e.d = 'abc' AND e.d <> '2019-1-1'", False),  # 'abc' is no date in any engine
        ("e.d = '2019-01-01' AND e.d <> '2019-1-1'", True),
        ("e.d IN ('2019-01-01', '2019-01-02 00:00:00')", False),  # two days
        ("e.d IN ('2019-01-01', '2019-01-01 00:00:00')", True),  # one moment
        ("e.d BETWEEN '2019-01-01' AND '2019-1-31'", False),  # two days, in the order of their texts
        ("e.d IN ('2019-9-1', '2019-10-1')", False),  # no ordering is asked
        ("e.d BETWEEN '2019-9-1' AND '2019-10-1'", True),  # '2019-10-1' sorts before '2019-9-1'
        ("e.d >= '2019-07-27' AND e.d <= '2019-6-28'", True),
        ("e.d = '2019-01-01' AND e.d <> '20190102'", True),  # a spelling that is not worked out
        ("e.d = e.ts AND e.d = '2019-01-01' AND e.ts = '20190101'", True),  # one value under two names
        ("COALESCE(e.d, e.ts) = '2019-01-01' AND e.d <> '20190101'", True),  # a value that hands the column on
        ("e.d = '2019-01-01 10:00:00' AND e.d <> '2019-01-01 11:00:00'", True),  # one day twice for a DATE column in DuckDB
        ("e.d = '2019-01-01 10:00:00' AND e.d <> '2019-01-02 10:00:00'", False),
    ],
)
def test_which_date_strings_are_declined(condition, declined):
    for dialect in (*ENGINES, "bigquery"):
        for types in (None, DECLARED):
            found = string_number_compare.problem(where(condition), dialect, types, plain_ok=True, schema=SCHEMA)
            assert (found is not None and "dates written in more than one way" in found) is declined, (dialect, types)
    assert string_number_compare.problem(where(condition), "mysql", TEXT, plain_ok=True, schema=SCHEMA) is None  # text is compared as text


@pytest.mark.parametrize(
    "condition, declined",
    [
        ("e.n = '1' AND e.n <> '2'", False),  # plain integers: different strings, different numbers
        ("e.n >= '10' AND e.n < '20'", False),  # and in the same order
        ("e.n = '01'", False),  # one string
        ("e.n = '1' AND e.n = 'x'", False),  # 'x' is an error for a number column: the column is text
        ("e.n = '1' AND e.n = '01'", True),
        ("e.n = '1' AND e.n = ' 1'", True),
        ("e.n = '1.5' AND e.n = '2'", True),  # DuckDB rounds '1.5' for an integer column
        ("e.n >= '9' AND e.n < '10'", True),
        ("e.n > '1.5'", False),  # one string is one point, whatever number it is
        ("e.n > '1.5' AND e.n < '3'", True),  # an ordering over a string that is not a plain integer
    ],
)
def test_which_numeric_strings_are_declined_where_the_engine_converts_them(condition, declined):
    for dialect in ("duckdb", "postgres"):
        found = string_number_compare.problem(where(condition), dialect, None, plain_ok=True, schema=SCHEMA)
        assert (found is not None and "strings that are numbers" in found) is declined, dialect
        assert string_number_compare.problem(where(condition), dialect, TEXT, plain_ok=True, schema=SCHEMA) is None
    assert string_number_compare.problem(where(condition), "bigquery", None, plain_ok=True, schema=SCHEMA) is None
    reported = string_number_compare.problem(where(condition), "mysql", None, plain_ok=True, schema=SCHEMA)
    assert reported is None or "strings that are numbers" not in reported  # MySQL: the numeric reading decides


@pytest.mark.parametrize(
    "condition, engines",
    [
        ("e.n = '10:00:00' AND e.n <> '10:00'", (*ENGINES, "bigquery")),  # one time of day written two ways
        ("e.n >= '8:00:00' AND e.n < '10:00:00'", (*ENGINES, "bigquery")),
        ("e.n = 'yes' AND e.n <> 'true'", ("duckdb", "postgres")),  # one truth value for a Boolean column
        ("e.n IN ('0', 'f') AND e.n <> 'no'", ("duckdb", "postgres")),
    ],
)
def test_times_and_truth_values_written_two_ways_are_declined(condition, engines):
    for dialect in engines:
        assert string_number_compare.problem(where(condition), dialect, None, plain_ok=True, schema=SCHEMA) is not None, dialect
        assert string_number_compare.problem(where(condition), dialect, TEXT, plain_ok=True, schema=SCHEMA) is None, dialect


@pytest.mark.parametrize("condition", ["e.n >= '08:00:00' AND e.n < '10:00:00'", "e.n = '10:00'", "e.n = 'yes' AND e.n <> 'no'", "e.n IN ('yes', 'true', 'maybe')"])
def test_times_and_truth_values_written_one_way_are_read_as_before(condition):
    for dialect in ("duckdb", "postgres", "bigquery"):
        assert string_number_compare.problem(where(condition), dialect, None, plain_ok=True, schema=SCHEMA) is None, dialect


def test_two_times_on_one_day_are_two_values_only_for_a_column_that_keeps_the_time():
    condition = where("e.ts >= '2019-01-01 00:00:00' AND e.ts <= '2019-01-01 23:59:59'")
    for dialect in (*ENGINES, "bigquery"):
        assert string_number_compare.problem(condition, dialect, DECLARED, plain_ok=True, schema=SCHEMA) is None  # DATETIME
        assert string_number_compare.problem(condition, dialect, None, plain_ok=True, schema=SCHEMA) is not None  # it may be a DATE
        assert string_number_compare.problem(condition.replace("e.ts", "e.d"), dialect, DECLARED, plain_ok=True, schema=SCHEMA) is not None


def test_duckdb_returns_the_rows_the_text_reading_denies():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE e (k INT, d DATE, ts TIMESTAMP, n INT, m INT)")
    con.execute("INSERT INTO e VALUES (1, DATE '2019-01-01', TIMESTAMP '2019-01-01 00:00:00', 1, 9), (2, DATE '2019-02-01', TIMESTAMP '2019-02-01 10:00:00', 9, 10), (3, NULL, NULL, NULL, NULL)")
    differing = 0
    for left, right in NUMBER_SPELLINGS + [ORDERED[0], ORDERED[1], DATE_SPELLINGS[2], DATE_SPELLINGS[3], DATE_SPELLINGS[4]]:
        differing += sorted(con.execute(left).fetchall()) != sorted(con.execute(right).fetchall())
    assert differing == len(NUMBER_SPELLINGS) + 5
    for left, right in SAFE:
        assert sorted(con.execute(left).fetchall()) == sorted(con.execute(right).fetchall())
    assert con.execute("SELECT e.k FROM e WHERE e.d = '2019-01-01 10:00:00' AND e.d = '2019-01-01 11:00:00'").fetchall() == [(1,)]
