"""A string compared with a value that is not known to be text (``kumo_sql_prover.string_number_compare``, 2026-11-16).

MySQL compares a string with a number as numbers, and reads ``'x'`` and ``'y'`` both as 0. The provers read a value they have no
type for as text, where two different strings are two different values. For a bare untyped column the check of issue #613
covered this; a value built on the column was not checked, and each left query below was proved equal to its right query.
MySQL 8.0.46 returns different rows for every pair of ``HANDED_ON``, ``UNTYPED_CALLS``, ``TIED``, ``BOTH_KINDS`` and
``CONDITIONS`` on ``t(k INT, s INT, c INT)`` holding ``(1, 0, 1), (2, 0, 0), (3, NULL, 0), (4, 5, 5)``, and the same rows for
every pair of ``TEXT``.
"""

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import numeric_column_reading, string_number_compare
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtStatus, prove_equivalent_smt

PROVERS = (prove_equivalent_algebraic, prove_equivalent_smt)
SCHEMA = {"t": ["k", "s", "c"]}
TEXT_COLUMNS = {"t": {"k": "INT", "s": "VARCHAR", "c": "VARCHAR"}}
NONE = "SELECT t.k FROM t WHERE 1 = 0"


def two_strings(value: str) -> tuple[str, str]:
    """``value = 'x' AND value <> 'y'`` against ``value = 'x'``: one query when ``value`` is text, two when it is the number 0."""

    return f"SELECT t.k FROM t WHERE {value} = 'x' AND {value} <> 'y'", f"SELECT t.k FROM t WHERE {value} = 'x'"


# the value is a column's own value, with the column's type
HANDED_ON = [
    "COALESCE(t.s)",
    "IFNULL(t.s, NULL)",
    "COALESCE(NULL, t.s)",
    "COALESCE(t.s, t.c)",
    "CASE WHEN t.k = 1 THEN t.s END",
    "CASE t.k WHEN 1 THEN t.s ELSE t.c END",
    "IF(t.k = 1, t.s, NULL)",
    "IF(t.k = 1, t.s, t.c)",
    "NULLIF(t.s, '1q')",
    "(SELECT MIN(u.s) FROM t AS u)",
]
# a number whatever the column is
UNTYPED_CALLS = ["TRUNCATE(t.s, 0)", "YEAR(t.s)", "DAY(t.s)", "t.s | 0", "LN(t.s + 1)", "UNIX_TIMESTAMP(t.s)", "BIT_COUNT(t.s)", "-t.s", "-(-t.s)", "COALESCE(t.s, 0)", "IF(t.k = 1, t.s, 7)"]
# a string or a date whatever the column is: compared as strings or as dates, so the proof stands
TEXT = [
    two_strings("LEFT(t.s, 1)"),
    two_strings("LOWER(t.s)"),
    two_strings("CAST(t.s AS CHAR)"),
    two_strings("COALESCE(t.s, 'q')"),
    two_strings("CASE WHEN t.k = 1 THEN 'a' ELSE 'b' END"),
    two_strings("CASE WHEN t.k = 1 THEN 'a' ELSE t.s END"),
    ("SELECT t.k FROM t WHERE DATE(t.s) = '2020-01-01' AND DATE(t.s) <> '2020-01-02'", "SELECT t.k FROM t WHERE DATE(t.s) = '2020-01-01'"),
    (
        "SELECT t.k FROM t WHERE DATE_FORMAT(t.s, '%Y-%m') = '2020-01' AND DATE_FORMAT(t.s, '%Y-%m') <> '2020-02'",
        "SELECT t.k FROM t WHERE DATE_FORMAT(t.s, '%Y-%m') = '2020-01'",
    ),
]
# one string on the value and one on the column under it
TIED = [
    ("SELECT t.k FROM t WHERE COALESCE(t.s, t.c) = 'x' AND t.s <> 'y'", "SELECT t.k FROM t WHERE t.s = 'x'"),
    ("SELECT d.k FROM (SELECT t.k, COALESCE(t.s, t.c) AS f, t.s AS s FROM t) AS d WHERE d.f = 'x' AND d.s <> 'y'", "SELECT t.k FROM t WHERE t.s = 'x'"),
    (
        "SELECT d.k FROM (SELECT t.k, MIN(t.s) AS m FROM t GROUP BY t.k) AS d WHERE d.m = 'x' AND d.m <> 'y'",
        "SELECT d.k FROM (SELECT t.k, MIN(t.s) AS m FROM t GROUP BY t.k) AS d WHERE d.m = 'x'",
    ),
]
# a string and a number: 'abc' reads as 0, so the value 0 meets both
BOTH_KINDS = [
    ("SELECT t.k FROM t GROUP BY t.k HAVING MIN(t.s) = 'abc' AND MIN(t.s) = 0", NONE),
    ("SELECT t.k FROM t WHERE TRUNCATE(t.s, 0) = 'abc' AND TRUNCATE(t.s, 0) = 0", NONE),
    ("SELECT t.k FROM t WHERE COALESCE(t.s, t.c) = 'abc' AND COALESCE(t.s, t.c) = 0", NONE),
]
# a condition is 0 or 1 on MySQL: (t.c = 2) = 'x' holds where t.c <> 2
CONDITIONS = [
    ("SELECT t.k FROM t WHERE (t.c = 2) = 'x'", NONE),
    ("SELECT t.k FROM t WHERE (t.c IN (5, 6)) = 'x'", NONE),
    ("SELECT t.k FROM t WHERE FALSE = 'x'", NONE),
    two_strings("(t.c = 2)"),
    two_strings("(t.c BETWEEN 5 AND 6)"),
    two_strings("EXISTS (SELECT 1 FROM t AS u WHERE u.k = 5)"),
]


@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("value", HANDED_ON)
def test_a_value_handed_on_from_an_untyped_column_is_checked_like_the_column(prover, value):
    left, right = two_strings(value)
    assert not prover(left, right, dialect="mysql", schema=SCHEMA).proven
    # declared as text, every column the value can be is text, and the strings are compared as strings
    if prover is prove_equivalent_algebraic or "SELECT" not in value:  # the SMT prover alone does not read a scalar subquery
        assert prover(left, right, dialect="mysql", schema=SCHEMA, types=TEXT_COLUMNS).status is SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("value", UNTYPED_CALLS)
def test_a_value_the_prover_has_no_type_for_is_checked(prover, value):
    left, right = two_strings(value)
    assert not prover(left, right, dialect="mysql", schema=SCHEMA).proven
    assert not prover(left, right, dialect="mysql", schema=SCHEMA, types=TEXT_COLUMNS).proven  # the call returns a number for a text column too


@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("left,right", TEXT)
def test_a_value_that_is_text_or_a_date_keeps_its_proof(prover, left, right):
    assert prover(left, right, dialect="mysql", schema=SCHEMA).status is SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("left,right", TIED)
def test_a_string_on_the_value_and_one_on_its_column_are_two_strings_on_the_column(prover, left, right):
    assert not prover(left, right, dialect="mysql", schema=SCHEMA, compare_names=False).proven
    assert prover(left, right, dialect="mysql", schema=SCHEMA, types=TEXT_COLUMNS, compare_names=False).status is SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("dialect", ["mysql", "duckdb", "postgres", "bigquery"])
@pytest.mark.parametrize("left,right", BOTH_KINDS + CONDITIONS)
def test_a_value_compared_with_a_string_and_a_number_and_a_condition_compared_with_a_string_are_declined(prover, dialect, left, right):
    if dialect == "bigquery":
        left, right = (sql.replace("TRUNCATE(", "TRUNC(") for sql in (left, right))
    assert not prover(left, right, dialect=dialect, schema=SCHEMA, compare_names=False).proven


def test_the_reading_replaces_the_strings_compared_with_a_value():
    left, right = two_strings("COALESCE(t.s, t.c)")
    found = numeric_column_reading.reading(left, right, "mysql")
    assert found.pair == ("SELECT t.k FROM t WHERE COALESCE(t.s, t.c) = 0 AND COALESCE(t.s, t.c) <> 0", right)
    # the string on the value counts for each column it can be, and the strings on both are replaced
    tied = numeric_column_reading.reading(*TIED[0], "mysql")
    assert tied.pair == ("SELECT t.k FROM t WHERE COALESCE(t.s, t.c) = 0 AND t.s <> 0", TIED[0][1])
    assert numeric_column_reading.reading(*two_strings("LEFT(t.s, 1)"), "mysql") is None
    assert numeric_column_reading.reading(*two_strings("COALESCE(t.s, t.c)"), "mysql", TEXT_COLUMNS) is None


def _value(sql: str) -> sqlglot.exp.Expression:
    return sqlglot.parse_one(f"SELECT {sql} FROM t", read="mysql").selects[0]


def test_what_a_value_hands_on():
    def handed(sql):
        columns, only = string_number_compare._handed_on(_value(sql))
        return sorted(c.sql() for c in columns), only

    assert handed("MIN(t.s)") == (["t.s"], True)
    assert handed("MAX(DISTINCT t.s)") == (["t.s"], True)
    assert handed("LAG(t.s, 1, t.c) OVER (ORDER BY t.k)") == (["t.c", "t.s"], True)
    assert handed("COALESCE(t.s, NULL)") == (["t.s"], True)
    assert handed("COALESCE(t.s, 0)") == (["t.s"], False)
    assert handed("CASE WHEN t.c = 1 THEN t.s ELSE t.c END") == (["t.c", "t.s"], True)
    assert handed("(SELECT u.s FROM t AS u UNION ALL SELECT u.c FROM t AS u)") == (["u.c", "u.s"], True)
    assert handed("t.s + 0") == ([], False)
    assert handed("UPPER(t.s)") == ([], False)
    # one column and nothing else: the value is read as that column
    assert string_number_compare._carried(_value("NULLIF(MIN(t.s), 5)")).sql() == "t.s"
    assert string_number_compare._carried(_value("COALESCE(t.s, t.s)")).sql() == "t.s"
    assert string_number_compare._carried(_value("COALESCE(t.s, t.c)")).sql() == "COALESCE(t.s, t.c)"


def test_the_type_mysql_gives_a_value():
    def fixed(sql, types=None):
        tree = sqlglot.parse_one(f"SELECT {sql} FROM t", read="mysql")
        return string_number_compare._Kinds(tree, types, SCHEMA).fixed(tree.selects[0])

    assert fixed("LEFT(t.s, 1)") == "string"
    assert fixed("-t.s") == "number"
    assert fixed("DATE(t.s)") == fixed("t.s + INTERVAL 1 DAY") == fixed("CAST(t.s AS DATE)") == "other"
    assert fixed("KUMOSQL_SCALAR(0)") == "other"  # the prover's placeholder for a scalar subquery it has set aside
    assert fixed("CASE WHEN t.k = 1 THEN 'a' ELSE t.s END") == "string"  # a choice between a string and anything is a string
    assert fixed("CASE WHEN t.k = 1 THEN 1 ELSE t.s END") is None
    assert fixed("COALESCE(t.s, t.c)") is None
    assert fixed("COALESCE(t.s, t.c)", TEXT_COLUMNS) == "string"
    assert fixed("TRUNCATE(t.s, 0)") is None  # a number, but the prover does not know the call: checked as an open value
    assert fixed("MIN(t.s)", TEXT_COLUMNS) == "string"
