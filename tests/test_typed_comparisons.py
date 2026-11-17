"""Comparison spellings that the declared types make the same (``kumo_sql_prover.typed_comparisons``)."""

import datetime
import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.typed_comparisons import typed_comparison_rules

TYPES = {"o": {"k": "INT64", "d": "DATE", "f": "FLOAT64", "s": "STRING", "t": "TIMESTAMP"}, "l": {"k": "INT64", "d": "DATE", "f": "FLOAT64", "n": "NUMERIC"}}
SCHEMA = {name: list(columns) for name, columns in TYPES.items()}
JOIN = "SELECT o.k, l.k AS k2 FROM o JOIN l ON "


def rewritten(test, types=TYPES, dialect="bigquery", query=JOIN + "{}"):
    sql = query.format(test)
    return typed_comparison_rules(sqlglot.parse_one(sql, read=dialect), types, dialect).sql(dialect=dialect).removeprefix(JOIN)


def proved(left, right):
    return prove_equivalent_algebraic(JOIN + left, JOIN + right, schema=SCHEMA, types=TYPES, compare_names=False).proven


@pytest.mark.parametrize("test, expected", [
    ("l.d < o.d + INTERVAL '10' DAY", "l.d < DATE_ADD(o.d, INTERVAL '10' DAY)"),
    ("o.d + INTERVAL 2 DAY <= l.d", "DATE_ADD(o.d, INTERVAL '2' DAY) <= l.d"),
    ("o.d - INTERVAL 1 MONTH = l.d + INTERVAL 1 YEAR", "DATE_SUB(o.d, INTERVAL '1' MONTH) = DATE_ADD(l.d, INTERVAL '1' YEAR)"),
    ("(o.d + INTERVAL 3 WEEK) > DATE_SUB(l.d, INTERVAL 1 QUARTER)", "DATE_ADD(o.d, INTERVAL '3' WEEK) > DATE_SUB(l.d, INTERVAL '1' QUARTER)"),
    ("o.d + INTERVAL 1 DAY != CAST('2020-01-01' AS DATE)", "DATE_ADD(o.d, INTERVAL '1' DAY) <> CAST('2020-01-01' AS DATE)"),
    ("l.k + 1 = o.k + 1", "l.k = o.k"),
    ("1 + l.k < o.k * 2 + 1", "l.k < o.k * 2"),
    ("l.k - 7 >= (o.k - 7)", "l.k >= o.k"),
    ("l.k + 01 != o.k + 1", "l.k <> o.k"),
])
def test_the_rewrite(test, expected):
    assert rewritten(test) == expected


@pytest.mark.parametrize("test", [
    "l.d < o.d + INTERVAL 10 HOUR",  # leaves midnight
    "l.d < o.d + INTERVAL '1-2' YEAR TO MONTH",
    "l.d < o.d + INTERVAL o.k DAY",
    "l.d < o.d + INTERVAL -1 DAY",
    "o.t < o.d + INTERVAL 1 DAY",  # a TIMESTAMP is not a midnight
    "o.s < o.d + INTERVAL 1 DAY",
    "o.t + INTERVAL 1 DAY > l.d",
    "l.d < (o.d + INTERVAL 1 DAY) + INTERVAL 1 DAY",  # nested: not read
    "l.d < COALESCE(o.d + INTERVAL 1 DAY, l.d)",  # not an operand of the comparison
    "l.d IS NOT DISTINCT FROM o.d + INTERVAL 1 DAY",
    "l.f + 1 = o.f + 1",  # floating-point addition rounds
    "l.n + 1 = o.n + 1",
    "l.k + 1 = o.f + 1",
    "l.k + 1 = o.k + 2",
    "l.k + 1 = o.k - 1",
    "l.k + 1 = 1 - o.k",
    "l.k + 1 = o.k",
    "l.k * 2 = o.k * 2",
    "l.k + 1.0 = o.k + 1.0",
    "l.k + o.k = o.k + o.k",
    "x.k + 1 = y.k + 1",  # no such source
])
def test_declined(test):
    assert rewritten(test) == sqlglot.parse_one(JOIN + test, read="bigquery").sql(dialect="bigquery").removeprefix(JOIN)


def test_declined_without_types_or_in_another_dialect_or_where_names_may_mislead():
    for test in ("l.k + 1 = o.k + 1", "l.d < o.d + INTERVAL '10' DAY"):
        plain = sqlglot.parse_one(JOIN + test, read="bigquery").sql(dialect="bigquery").removeprefix(JOIN)
        assert rewritten(test, types=None) == plain
        assert rewritten(test, types={"o": TYPES["o"]}) == plain  # l is not declared
        assert rewritten(test, dialect="sqlite") == sqlglot.parse_one(JOIN + test, read="sqlite").sql(dialect="sqlite").removeprefix(JOIN)
        for query in ("WITH o AS (SELECT 1.5 AS k, 'x' AS d) " + JOIN + "{}", "SELECT o.k, l.k AS k2 FROM o JOIN (SELECT 1.5, 'x') AS l(k, d) ON {}"):
            sql = query.format(test)
            assert typed_comparison_rules(sqlglot.parse_one(sql, read="bigquery"), TYPES).sql(dialect="bigquery") == sqlglot.parse_one(sql, read="bigquery").sql(dialect="bigquery")
    # a derived table's own columns are followed; a column of an enclosing query is not
    assert typed_comparison_rules(sqlglot.parse_one("SELECT 1 FROM (SELECT k + 0 AS j FROM o) AS p JOIN l ON p.j + 1 = l.k + 1", read="bigquery"), TYPES).sql(dialect="bigquery").endswith("ON p.j = l.k")
    assert typed_comparison_rules(sqlglot.parse_one("SELECT 1 FROM (SELECT f AS j FROM o) AS p JOIN l ON p.j + 1 = l.k + 1", read="bigquery"), TYPES).sql(dialect="bigquery").endswith("ON p.j + 1 = l.k + 1")
    assert typed_comparison_rules(sqlglot.parse_one("SELECT 1 FROM o WHERE EXISTS(SELECT 1 FROM l WHERE l.k + 1 = o.k + 1)", read="bigquery"), TYPES).sql(dialect="bigquery").endswith("l.k + 1 = o.k + 1)")


@pytest.mark.parametrize("left, right", [
    ("l.k = o.k AND l.d < o.d + INTERVAL '10' DAY", "l.k = o.k AND l.d < DATE_ADD(o.d, INTERVAL 10 DAY)"),
    ("o.d + INTERVAL '2' DAY <= l.d AND l.d < o.d + INTERVAL '7' DAY", "DATE_ADD(o.d, INTERVAL 2 DAY) <= l.d AND l.d < DATE_ADD(o.d, INTERVAL 7 DAY)"),
    ("l.d = o.d - INTERVAL 1 MONTH", "l.d = DATE_SUB(o.d, INTERVAL 1 MONTH)"),
    ("l.k + 1 = o.k + 1", "l.k = o.k"),
    ("l.k - 3 < o.k - 3", "l.k < o.k"),
])
def test_proved(left, right):
    assert proved(left, right)


@pytest.mark.parametrize("left, right", [
    ("l.d < o.d + INTERVAL '10' DAY", "l.d < DATE_ADD(o.d, INTERVAL 11 DAY)"),
    ("l.d < o.d + INTERVAL '10' DAY", "l.d < DATE_SUB(o.d, INTERVAL 10 DAY)"),
    ("l.d < o.d + INTERVAL '10' DAY", "l.d < DATE_ADD(o.d, INTERVAL 10 MONTH)"),
    ("l.d < o.d + INTERVAL '10' DAY", "l.d < o.d"),
    ("l.d < o.d + INTERVAL '10' DAY", "l.d <= DATE_ADD(o.d, INTERVAL 10 DAY)"),
    ("l.k + 1 = o.k + 2", "l.k = o.k"),
    ("l.k + 1 = o.k - 1", "l.k = o.k"),
    ("l.k + 1 < o.k + 1", "l.k <= o.k"),
    ("l.f + 1 = o.f + 1", "l.f = o.f"),  # 1e16 + 1 = (1e16 + 2) + 1 in floating point
])
def test_not_proved(left, right):
    assert not proved(left, right)


def test_the_two_readings_agree_on_duckdb():
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import run_unoptimized

    rng = random.Random(5)
    days = [None, datetime.date(2020, 1, 31), datetime.date(2020, 2, 29), datetime.date(2020, 3, 1), datetime.date(2019, 12, 31), datetime.date(2021, 2, 28), datetime.date(2020, 2, 10)]
    for _ in range(40):
        db = duckdb.connect()
        for name in ("o", "l"):
            db.execute(f"CREATE TABLE {name} (k BIGINT, d DATE)")
            for _ in range(rng.choice([0, 1, 3, 6])):
                db.execute(f"INSERT INTO {name} VALUES (?, ?)", [rng.choice([None, 0, 1, 2, 3]), rng.choice(days)])
        # 1: an interval sum is a midnight (a TIMESTAMP here, a DATETIME in GoogleSQL); the comparison reads the date as its midnight
        for operator in ("<", "<=", "=", "!=", ">", ">="):
            for shift in ("+ INTERVAL 10 DAY", "- INTERVAL 1 MONTH", "+ INTERVAL 1 YEAR", "+ INTERVAL 2 WEEK"):
                spelled = f"SELECT o.k, l.k, o.d, l.d FROM o JOIN l ON l.d {operator} o.d {shift}"
                as_date = f"SELECT o.k, l.k, o.d, l.d FROM o JOIN l ON l.d {operator} CAST(o.d {shift} AS DATE)"
                x, y = run_unoptimized(db, spelled, as_date)
                assert sorted(map(repr, x)) == sorted(map(repr, y)), spelled
            # 2: the same number on both sides
            x, y = run_unoptimized(db, f"SELECT o.k, l.k FROM o JOIN l ON l.k + 5 {operator} o.k * 2 + 5", f"SELECT o.k, l.k FROM o JOIN l ON l.k {operator} o.k * 2")
            assert sorted(map(repr, x)) == sorted(map(repr, y)), operator
        db.close()


@pytest.mark.parametrize("test, expected", [
    ("l.d BETWEEN o.d AND o.d + INTERVAL '30' DAY", "l.d BETWEEN o.d AND DATE_ADD(o.d, INTERVAL '30' DAY)"),
    ("l.d BETWEEN o.d - INTERVAL 1 DAY AND o.d + INTERVAL 1 DAY", "l.d BETWEEN DATE_SUB(o.d, INTERVAL '1' DAY) AND DATE_ADD(o.d, INTERVAL '1' DAY)"),
    ("l.k BETWEEN o.k AND o.k + 3", "l.k BETWEEN o.k AND o.k + 3"),  # not dates: left alone
    ("l.f BETWEEN o.f AND o.f + INTERVAL '3' DAY", "l.f BETWEEN o.f AND o.f + INTERVAL '3' DAY"),  # not a date
])
def test_the_bounds_of_a_between_are_read_like_comparisons(test, expected):
    assert rewritten(test) == expected


def test_a_between_with_a_shifted_bound_meets_its_two_comparisons():
    assert proved("l.d BETWEEN o.d AND o.d + INTERVAL '30' DAY", "l.d >= o.d AND l.d <= DATE_ADD(o.d, INTERVAL '30' DAY)")
    assert proved("l.d BETWEEN o.d AND o.d + INTERVAL '30' DAY", "l.d >= o.d AND l.d <= o.d + INTERVAL '30' DAY")
    assert not proved("l.d BETWEEN o.d AND o.d + INTERVAL '30' DAY", "l.d >= o.d AND l.d <= o.d + INTERVAL '31' DAY")
