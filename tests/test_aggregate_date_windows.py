"""``MIN`` or ``MAX`` of a DATE column against a constant day is read like the column (``date_windows``, MySQL).

"Every move of the item falls in the first quarter" is ``HAVING MIN(moved_on) >= '2024-01-01' AND MAX(moved_on) <=
'2024-03-31'``, and as often ``MAX(moved_on) < '2024-04-01'`` or ``'2024-1-1'``. The rule read those spellings on a
column and on the output of a derived table; here it reads them on the aggregate where it is computed. Each reading
is replayed on DuckDB (with the day cast to a date, which is how MySQL compares a date with such a string) over
moves around both ends of the window, an item without a dated move and an item without moves.
"""

from __future__ import annotations

import datetime

import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.date_windows import write_date_windows

SCHEMA = {"items": ["item_id", "label"], "moves": ["item_id", "moved_on", "units"]}
TYPES = {"items": {"item_id": "INT", "label": "VARCHAR"}, "moves": {"item_id": "INT", "moved_on": "DATE", "units": "INT"}}
HEAD = "SELECT item_id FROM moves GROUP BY item_id HAVING "

# (the test as written, the test as read)
READ = [
    ("MAX(moved_on) < '2024-04-01'", "MAX(moved_on) <= '2024-03-31'"),
    ("MIN(moved_on) > '2023-12-31'", "MIN(moved_on) >= '2024-01-01'"),
    ("MIN(moved_on) >= '2024-1-1'", "MIN(moved_on) >= '2024-01-01'"),
    ("'2024-01-01' <= MIN(moved_on)", "MIN(moved_on) >= '2024-01-01'"),
    ("'2024-04-01' > MAX(moves.moved_on)", "MAX(moves.moved_on) <= '2024-03-31'"),
    ("MAX(DATE(moved_on)) <= DATE('2024-03-31')", "MAX(moved_on) <= '2024-03-31'"),
    ("MIN(moved_on) >= DATE_SUB('2024-01-31', INTERVAL 30 DAY)", "MIN(moved_on) >= '2024-01-01'"),
    ("MIN(moved_on) = CAST('2024-02-29' AS DATE)", "MIN(moved_on) = '2024-02-29'"),
    ("MAX(moved_on) BETWEEN '2024-1-1' AND DATE('2024-03-31')", "MAX(moved_on) BETWEEN '2024-01-01' AND '2024-03-31'"),
    ("(MIN(moved_on)) <> '2024-2-1'", "MIN(moved_on) <> '2024-02-01'"),
]
LEFT_ALONE = [
    "MAX(units) < '2024-04-01'",  # not a DATE column
    "MIN(DISTINCT moved_on) > '2023-12-31'",
    "COUNT(moved_on) < '2024-04-01'",
    "MAX(moved_on) < '2024-04'",  # not a day
    "MAX(moved_on) < MIN(moved_on)",
    "DATEDIFF('2024-04-01', MAX(moved_on)) > 0",
    "MAX(moved_on + INTERVAL 1 DAY) < '2024-04-01'",
]


def reading(test: str, dialect: str = "mysql") -> str | None:
    tree, changed = write_date_windows(sqlglot.parse_one(HEAD + test, read=dialect), SCHEMA, TYPES, dialect)
    return tree.sql(dialect=dialect)[len(HEAD) :] if changed else None


@pytest.mark.parametrize("written, read", READ)
def test_an_aggregate_of_a_date_column_is_read_like_the_column(written, read):
    assert reading(written) == read


@pytest.mark.parametrize("written", LEFT_ALONE)
def test_a_test_that_is_not_one_of_a_date_and_a_day_is_left_alone(written):
    assert reading(written) is None


def test_other_dialects_and_window_aggregates_are_not_read():
    assert reading("MAX(moved_on) < '2024-04-01'", "postgres") is None
    tree = sqlglot.parse_one("SELECT item_id, MAX(moved_on) OVER (PARTITION BY item_id) < '2024-04-01' AS early FROM moves", read="mysql")
    assert not write_date_windows(tree, SCHEMA, TYPES, "mysql")[1]


def test_a_select_alias_of_the_column_name_that_means_something_else_leaves_the_test_alone():
    tree = sqlglot.parse_one("SELECT item_id, MAX(units) AS moved_on FROM moves GROUP BY item_id HAVING MAX(moved_on) < '2024-04-01'", read="mysql")
    assert not write_date_windows(tree, SCHEMA, TYPES, "mysql")[1]


def test_with_the_switch_off_the_aggregate_is_not_read():
    with shape_fallback.without("aggregate_date_windows"):
        assert reading("MAX(moved_on) < '2024-04-01'") is None
        tree = sqlglot.parse_one("SELECT item_id FROM moves WHERE moved_on < '2024-04-01'", read="mysql")
        assert write_date_windows(tree, SCHEMA, TYPES, "mysql")[1]  # the column is read as before


def _days():
    edges = [datetime.date(2024, 1, 1), datetime.date(2024, 3, 31), datetime.date(2024, 2, 29), datetime.date(2024, 2, 1), datetime.date(2024, 1, 31)]
    return sorted({day + datetime.timedelta(days=shift) for day in edges for shift in (-1, 0, 1)})


def _as_dates(test: str) -> str:
    """The test for DuckDB: every day string cast to a date, which is how MySQL compares a date with it."""

    tree = sqlglot.parse_one(HEAD + test, read="mysql")
    for literal in list(tree.find_all(sqlglot.exp.Literal)):
        if literal.is_string and literal.this.count("-") == 2:
            year, month, day = (int(part) for part in literal.this.split("-"))
            literal.replace(sqlglot.exp.cast(sqlglot.exp.Literal.string(datetime.date(year, month, day).isoformat()), "DATE"))
    return tree.sql(dialect="duckdb")


@pytest.mark.parametrize("written, read", [pair for pair in READ if "DATE_SUB" not in pair[0]])
def test_a_reading_returns_the_rows_of_the_test_as_written(written, read):
    duckdb = pytest.importorskip("duckdb")
    db = duckdb.connect(":memory:")
    db.execute("CREATE TABLE moves (item_id INT, moved_on DATE, units INT)")
    days = _days()
    rows = []
    item = 0
    for first in days:  # one or two dated moves per item, every pair of days around the ends of the window
        for second in [None, *days[:: max(1, len(days) // 5)]]:
            item += 1
            rows.append((item, first, 1))
            if second is not None:
                rows.append((item, second, 1))
            rows.append((item, None, 1))  # and a move without a day
    rows.append((item + 1, None, 1))  # an item without a dated move
    db.executemany("INSERT INTO moves VALUES (?, ?, ?)", rows)
    as_written = sorted(db.execute(_as_dates(written)).fetchall())
    as_read = sorted(db.execute(_as_dates(read)).fetchall())
    assert as_written == as_read
    assert 0 < len(as_written) < item + 1  # the test keeps some items and drops others


# proofs: the grouped test in three spellings, and windows one day off
QUARTER = "SELECT i.item_id, i.label FROM items AS i JOIN moves AS m ON i.item_id = m.item_id GROUP BY i.item_id, i.label HAVING "
PLAIN = QUARTER + "MIN(m.moved_on) >= '2024-01-01' AND MAX(m.moved_on) <= '2024-03-31'"
SPELLINGS = [
    QUARTER + "MIN(m.moved_on) >= '2024-01-01' AND MAX(m.moved_on) < '2024-04-01'",
    QUARTER + "MIN(m.moved_on) > '2023-12-31' AND MAX(m.moved_on) <= '2024-3-31'",
    QUARTER + "'2024-01-01' <= MIN(m.moved_on) AND '2024-04-01' > MAX(m.moved_on)",
    QUARTER + "MIN(DATE(m.moved_on)) >= DATE('2024-01-01') AND MAX(m.moved_on) <= DATE_SUB('2024-04-01', INTERVAL 1 DAY)",
]
ONE_DAY_OFF = [
    QUARTER + "MIN(m.moved_on) >= '2024-01-01' AND MAX(m.moved_on) < '2024-03-31'",
    QUARTER + "MIN(m.moved_on) > '2024-01-01' AND MAX(m.moved_on) <= '2024-03-31'",
    QUARTER + "MIN(m.moved_on) >= '2024-01-01' AND MIN(m.moved_on) < '2024-04-01'",
]


def prove(left: str, right: str):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="mysql", exact_arithmetic=True, timeout_ms=20000)


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_the_spellings_of_one_window_are_proved_equal(spelling):
    assert prove(PLAIN, spelling).proven
    with shape_fallback.without("aggregate_date_windows"):
        assert not prove(PLAIN, spelling).proven


def test_two_days_that_sort_differently_as_text_are_still_declined():
    # '2024-1-1' sorts after '2024-03-31' as text: the check on the query as written declines the pair before any rule
    relaxed = QUARTER + "MIN(m.moved_on) >= '2024-1-1' AND MAX(m.moved_on) <= '2024-03-31'"
    assert "written in more than one way" in prove(PLAIN, relaxed).reason


@pytest.mark.parametrize("other", ONE_DAY_OFF)
def test_a_window_one_day_off_is_not_proved(other):
    assert not prove(PLAIN, other).proven
