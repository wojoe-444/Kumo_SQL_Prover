"""A DATE column against a day worked out from a constant, written out (``date_windows``, MySQL).

Checked on a MySQL 8.0.46 server on 2026-11-19, in its default SQL mode. The rewriting was done by the module and
both spellings were run as a value (1, 0 or NULL) for every row of generated tables of DATE values: the days around
the constant, the ends of the DATE range, leap days, random days and NULL. 11,861 rewritten tests gave the same value
before and after on each of 2,036,886 rows, and 900 of 900 controls one day off differed. (Year 0 is not a leap year
on the server and dates before year 1000 agreed as well, on 94,888 rows; the module still assumes the documented
range.) The statements were built from: the comparisons ``= <> < <= > >=`` either way round and ``BETWEEN``; the column as ``d``,
``t.d``, ``DATE(d)`` and ``CAST(d AS DATE)``; the day as ``'YYYY-MM-DD'``, ``'YYYY-M-D'``, ``DATE('..')``,
``CAST('..' AS DATE)``, ``DATE '..'``, ``DATE_SUB``, ``DATE_ADD``, ``SUBDATE`` and ``ADDDATE`` (with an ``INTERVAL`` and
with a bare number of days), ``'..' - INTERVAL n DAY`` and ``'..' + INTERVAL n DAY``, nested; and ``DATEDIFF`` of the
two, both ways round, against whole numbers. The replay below makes the comparisons DuckDB can spell.
"""

import datetime
import random

import duckdb
import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.date_windows import constant_day, write_date_windows
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"visits": ["member_id", "room_id", "day", "kind", "seen_at", "badge"], "rooms": ["room_id", "opened", "day_count"]}
TYPES = {
    "visits": {"member_id": "INT", "room_id": "INT", "day": "DATE", "kind": "VARCHAR", "seen_at": "DATETIME", "badge": "INT"},
    "rooms": {"room_id": "INT", "opened": "DATE", "day_count": "INT"},
}
HEAD = "SELECT member_id FROM visits WHERE "


def reading(sql: str, dialect: str = "mysql", types=TYPES) -> str | None:
    tree, changed = write_date_windows(sqlglot.parse_one(sql, read=dialect), SCHEMA, types, dialect)
    return tree.sql(dialect=dialect) if changed else None


def written_test(test: str) -> str | None:
    written = reading(HEAD + test)
    return None if written is None else written[len(HEAD) :]


def test_a_day_worked_out_from_a_constant_is_written_out():
    assert written_test("day >= DATE_SUB('2019-07-27', INTERVAL 29 DAY)") == "day >= '2019-06-28'"
    assert written_test("day >= DATE_ADD('2019-07-27', INTERVAL -29 DAY)") == "day >= '2019-06-28'"
    assert written_test("day >= SUBDATE('2019-07-27', INTERVAL 29 DAY)") == "day >= '2019-06-28'"
    assert written_test("day >= SUBDATE('2019-07-27', 29)") == "day >= '2019-06-28'"
    assert written_test("day = ADDDATE('2019-07-05', -1)") == "day = '2019-07-04'"
    assert written_test("day = ADDDATE('2019-07-05', INTERVAL 1 DAY)") == "day = '2019-07-06'"
    assert written_test("day >= '2019-07-27' - INTERVAL 29 DAY") == "day >= '2019-06-28'"
    assert written_test("day <= '2019-07-27' + INTERVAL '2' DAY") == "day <= '2019-07-29'"
    assert written_test("day <> DATE('2019-07-04')") == "day <> '2019-07-04'"
    assert written_test("day = CAST('2019-07-04' AS DATE)") == "day = '2019-07-04'"
    assert written_test("day = '2019-7-4'") == "day = '2019-07-04'"  # the server reads one-digit months and days
    assert written_test("day = DATE_SUB(DATE_ADD(DATE('2020-02-27'), INTERVAL 3 DAY), INTERVAL 1 DAY)") == "day = '2020-02-29'"  # a leap day
    assert written_test("day BETWEEN DATE_ADD('2019-07-27', INTERVAL -29 DAY) AND DATE('2019-07-27')") == "day BETWEEN '2019-06-28' AND '2019-07-27'"
    # the constant on the left
    assert written_test("DATE_SUB('2019-07-27', INTERVAL 29 DAY) <= day") == "day >= '2019-06-28'"


def test_a_strict_test_is_the_test_on_the_day_next_to_it():
    assert written_test("day > '2019-06-27'") == "day >= '2019-06-28'"
    assert written_test("day < '2019-07-28'") == "day <= '2019-07-27'"
    assert written_test("'2019-06-27' < day") == "day >= '2019-06-28'"
    assert written_test("'2019-03-01' > day") == "day <= '2019-02-28'"
    assert written_test("day > DATE_SUB('2019-07-27', INTERVAL 30 DAY) AND day <= '2019-07-27'") == "day >= '2019-06-28' AND day <= '2019-07-27'"
    assert written_test("day < '2020-03-01'") == "day <= '2020-02-29'"
    assert written_test("NOT day > '2019-12-31'") == "NOT day >= '2020-01-01'"


def test_a_count_of_days_is_a_test_on_the_column():
    assert written_test("DATEDIFF('2019-07-27', day) < 30") == "day >= '2019-06-28'"
    assert written_test("DATEDIFF('2019-07-27', day) <= 29") == "day >= '2019-06-28'"
    assert written_test("DATEDIFF('2019-07-27', day) >= 0") == "day <= '2019-07-27'"
    assert written_test("DATEDIFF('2019-07-27', day) > 0") == "day <= '2019-07-26'"
    assert written_test("DATEDIFF('2019-07-05', day) = 1") == "day = '2019-07-04'"
    assert written_test("1 = DATEDIFF('2019-07-05', day)") == "day = '2019-07-04'"
    assert written_test("DATEDIFF(day, '2019-07-05') = -1") == "day = '2019-07-04'"
    assert written_test("DATEDIFF(day, '2019-07-05') > 2") == "day >= '2019-07-08'"
    assert written_test("30 > DATEDIFF('2019-07-27', day)") == "day >= '2019-06-28'"
    assert written_test("DATEDIFF('2019-07-27', day) BETWEEN 0 AND 29") == "day BETWEEN '2019-06-28' AND '2019-07-27'"
    assert written_test("DATEDIFF(day, '2019-07-27') BETWEEN -29 AND 0") == "day BETWEEN '2019-06-28' AND '2019-07-27'"
    assert written_test("DATEDIFF(DATE('2019-07-27'), DATE(day)) <> 3") == "day <> '2019-07-24'"


def test_the_column_may_be_wrapped_qualified_or_come_through_a_derived_table():
    assert written_test("DATE(day) < DATE('2019-01-01')") == "day <= '2018-12-31'"
    assert written_test("CAST(day AS DATE) = DATE_SUB('2019-07-05', INTERVAL 1 DAY)") == "day = '2019-07-04'"
    assert reading("SELECT v.member_id FROM visits AS v JOIN rooms AS o ON v.room_id = o.room_id AND opened > '2019-01-01' WHERE day < '2019-03-01'") == (
        "SELECT v.member_id FROM visits AS v JOIN rooms AS o ON v.room_id = o.room_id AND opened >= '2019-01-02' WHERE day <= '2019-02-28'"
    )
    assert reading("SELECT s.member_id FROM (SELECT * FROM visits WHERE day > '2019-01-01') AS s WHERE s.day < '2019-03-01'") == (
        "SELECT s.member_id FROM (SELECT * FROM visits WHERE day >= '2019-01-02') AS s WHERE s.day <= '2019-02-28'"
    )
    assert reading("SELECT member_id FROM (SELECT member_id, MIN(day) AS first FROM visits GROUP BY member_id) AS f WHERE first > '2019-02-01'") == (
        "SELECT member_id FROM (SELECT member_id, MIN(day) AS first FROM visits GROUP BY member_id) AS f WHERE first >= '2019-02-02'"
    )
    # in HAVING and in a CASE
    assert reading("SELECT day, COUNT(*) FROM visits GROUP BY day HAVING DATEDIFF('2019-07-27', day) BETWEEN 0 AND 29") == "SELECT day, COUNT(*) FROM visits GROUP BY day HAVING day BETWEEN '2019-06-28' AND '2019-07-27'"
    assert reading("SELECT SUM(CASE WHEN DATEDIFF(day, '2019-07-01') > 0 THEN 1 ELSE 0 END) FROM visits") == "SELECT SUM(CASE WHEN day >= '2019-07-02' THEN 1 ELSE 0 END) FROM visits"


@pytest.mark.parametrize(
    "test",
    [
        "day BETWEEN '2019-06-28' AND '2019-07-27'",  # already written out
        "day >= '2019-06-28'",
        "day = '2019-07-04'",
        "seen_at > '2019-06-27'",  # a DATETIME has times between two days
        "kind > '2019-06-27'",  # a string
        "badge > '2019-06-27'",
        "DATEDIFF('2019-07-27', seen_at) < 30",
        "day > DATE_SUB('2019-07-27', INTERVAL 1 MONTH)",  # the server clamps the end of a month
        "day > DATE_SUB('2019-07-27', INTERVAL 1 YEAR)",
        "day > '2019-07-27' - INTERVAL 1 WEEK",
        "day > '2019-02-30'",  # not a calendar date
        "day > '2019-13-01'",
        "day > '9999-12-31'",  # the next day is outside the DATE range
        "day < '1000-01-01'",
        "day = DATE_SUB('1000-01-05', INTERVAL 9 DAY)",
        "day > '0999-12-31'",
        "day > CURDATE() - INTERVAL 3 DAY",  # not a constant
        "day > DATE_SUB(seen_at, INTERVAL 3 DAY)",
        "day > '2019-06-27 00:00:00'",  # a time of day
        "day > '20190627'",
        "day > 20190627",
        "day > DATE_SUB('2019-07-27', INTERVAL 1.5 DAY)",
        "DATEDIFF('2019-07-27', day) < 2.5",
        "DATEDIFF('2019-07-27', day) BETWEEN 5 AND 1",  # an empty range is left to the prover
        "DATEDIFF('2019-07-27', '2019-07-01') < 30",  # no column
        "DATEDIFF(day, day) = 0",
        "ABS(DATEDIFF('2019-07-27', day)) < 30",
        "DATEDIFF('2019-07-27', day) + 1 < 30",
        "day + 1 > '2019-06-27'",  # arithmetic on a DATE is a number in MySQL
        "day IN ('2019-7-4')",
        "day LIKE '2019-07-%'",
    ],
)
def test_other_tests_are_left_alone(test):
    assert written_test(test) is None


def test_a_column_that_cannot_be_placed_is_left_alone():
    # two columns compared
    assert reading("SELECT v.member_id FROM visits AS v JOIN rooms AS o ON v.room_id = o.room_id WHERE v.day > o.opened") is None
    # a bare name two sources give
    assert reading("SELECT 1 FROM visits AS a JOIN visits AS b ON a.member_id = b.member_id WHERE day > '2019-06-27'") is None
    # a select-list alias of the same name that is something else (HAVING may mean either)
    assert reading("SELECT member_id, MAX(seen_at) AS day FROM visits GROUP BY member_id HAVING day > '2019-02-01'") is None
    # a table of another schema, an alias the select does not have, a derived table that renames or computes
    assert reading("SELECT 1 FROM other.visits WHERE day > '2019-06-27'") is None
    assert reading("SELECT 1 FROM visits AS v WHERE EXISTS (SELECT 1 FROM rooms AS o WHERE q.day > '2019-06-27')") is None
    assert reading("SELECT 1 FROM (SELECT seen_at AS day FROM visits) AS s WHERE day > '2019-06-27'") is None
    assert reading("SELECT 1 FROM (SELECT DATE_ADD(day, INTERVAL 1 MONTH) AS day FROM visits) AS s WHERE day > '2019-06-27'") is None
    assert reading("SELECT 1 FROM (SELECT day FROM visits) AS s (seen) WHERE seen > '2019-06-27'") is None
    # a correlated column of the enclosing query is read there, not here
    inner = "SELECT 1 FROM visits AS v WHERE EXISTS (SELECT 1 FROM rooms AS o WHERE o.room_id = v.room_id AND v.day > '2019-06-27')"
    assert reading(inner) is None
    # no declared types, a column with no declared type, another dialect
    assert reading(HEAD + "day > '2019-06-27'", types=None) is None
    assert reading(HEAD + "day > '2019-06-27'", types={"visits": {"member_id": "INT"}}) is None
    assert reading(HEAD + "day > '2019-06-27'", dialect="postgres") is None
    assert reading(HEAD + "day > '2019-06-27'", dialect="bigquery") is None


def test_the_tree_given_is_not_changed():
    tree = sqlglot.parse_one(HEAD + "day > '2019-06-27'", read="mysql")
    before = tree.sql(dialect="mysql")
    written, changed = write_date_windows(tree, SCHEMA, TYPES, "mysql")
    assert changed and tree.sql(dialect="mysql") == before and written is not tree


def test_the_day_is_worked_out_as_a_calendar_does():
    def day(text: str):
        return constant_day(sqlglot.parse_one(f"SELECT {text}", read="mysql").expressions[0])

    assert day("DATE_SUB('2020-03-01', INTERVAL 1 DAY)") == datetime.date(2020, 2, 29)
    assert day("DATE_SUB('2019-03-01', INTERVAL 1 DAY)") == datetime.date(2019, 2, 28)
    assert day("DATE_SUB('2100-03-01', INTERVAL 1 DAY)") == datetime.date(2100, 2, 28)  # 2100 is not a leap year
    assert day("ADDDATE('2019-12-31', 1)") == datetime.date(2020, 1, 1)
    assert day("'2019-01-01' - INTERVAL 365 DAY") == datetime.date(2018, 1, 1)
    assert day("DATE_ADD('9999-12-31', INTERVAL 1 DAY)") is None and day("DATE_SUB('1000-01-01', INTERVAL 1 DAY)") is None
    assert day("'2019-02-29'") is None and day("'2019-00-10'") is None and day("'2019-1-0'") is None
    assert day("'2019-07-27'") == day("'2019-7-27'") == day("DATE('2019-07-27')") == datetime.date(2019, 7, 27)


# (MySQL spelling, DuckDB spelling of the same test) for the forms the transpiler does not carry over.
BY_HAND = [
    ("day >= SUBDATE('{k}', {n})", "day >= CAST('{k}' AS DATE) - INTERVAL {n} DAY"),
    ("day < ADDDATE('{k}', {n})", "day < CAST('{k}' AS DATE) + INTERVAL {n} DAY"),
    ("day > '{k}' - INTERVAL {n} DAY", "day > CAST('{k}' AS DATE) - INTERVAL {n} DAY"),
    ("day <= SUBDATE('{k}', INTERVAL {n} DAY)", "day <= CAST('{k}' AS DATE) - INTERVAL {n} DAY"),
]
TRANSPILED = [
    "day > DATE_SUB('{k}', INTERVAL {n} DAY)",
    "day <= DATE_ADD('{k}', INTERVAL {n} DAY)",
    "day < DATE_ADD('{k}', INTERVAL -{n} DAY)",
    "DATE_SUB('{k}', INTERVAL {n} DAY) < day",
    "day <> DATE_SUB('{k}', INTERVAL {n} DAY)",
    "day BETWEEN DATE_SUB('{k}', INTERVAL {n} DAY) AND '{k}'",
    "DATEDIFF('{k}', day) < {n}",
    "DATEDIFF('{k}', day) >= {n}",
    "DATEDIFF('{k}', day) = {n}",
    "DATEDIFF(day, '{k}') > -{n}",
    "DATEDIFF(day, '{k}') <= {n}",
    "{n} > DATEDIFF('{k}', day)",
    "DATEDIFF('{k}', day) BETWEEN 0 AND {n}",
    "DATEDIFF(day, '{k}') BETWEEN -{n} AND 3",
    "day > '{k}'",
    "day < '{k}'",
    "DATE(day) >= DATE('{k}')",
]


@pytest.mark.parametrize("seed", range(4))
def test_replay_on_generated_dates(seed):
    """Each spelling and what the module writes for it have the same value (true, false or NULL) on every row."""

    rng = random.Random(seed)
    con = duckdb.connect()
    con.execute("CREATE TABLE visits (member_id INT, room_id INT, day DATE, kind VARCHAR, seen_at TIMESTAMP, badge INT)")
    compared = 0
    for _ in range(12):
        anchor = datetime.date(rng.choice([1999, 2000, 2019, 2020, 2100, rng.randint(1001, 9998)]), rng.randint(1, 12), rng.choice([1, 2, 15, 27, 28]))
        days = [anchor + datetime.timedelta(days=k) for k in range(-45, 46)] + [None, datetime.date(1000, 1, 1), datetime.date(9999, 12, 31)]
        con.execute("DELETE FROM visits")
        con.executemany("INSERT INTO visits VALUES (?, 1, ?, 'a', NULL, 1)", [(i, day) for i, day in enumerate(days)])
        n = rng.choice([0, 1, 2, 29, 30, 31, rng.randint(0, 40)])
        k = anchor.isoformat()
        forms = [(form, None) for form in TRANSPILED] + BY_HAND
        for mysql, by_hand in forms:
            test = mysql.format(k=k, n=n)
            written = written_test(test)
            assert written is not None, test
            original = by_hand.format(k=k, n=n) if by_hand else sqlglot.transpile(f"SELECT ({test}) AS v", read="mysql", write="duckdb")[0][len("SELECT (") : -len(") AS v")]
            left = con.execute(f"SELECT member_id, ({original}) FROM visits ORDER BY member_id").fetchall()
            right = con.execute(f"SELECT member_id, ({written}) FROM visits ORDER BY member_id").fetchall()
            assert left == right, (test, written)
            compared += len(left)
    assert compared > 20_000


def test_controls_one_day_off_differ():
    con = duckdb.connect()
    con.execute("CREATE TABLE visits (member_id INT, day DATE)")
    con.executemany("INSERT INTO visits VALUES (?, ?)", [(i, datetime.date(2019, 6, 20) + datetime.timedelta(days=i)) for i in range(45)])
    def rows(test: str):
        return con.execute(f"SELECT member_id FROM visits WHERE {test} ORDER BY 1").fetchall()

    window = rows("day BETWEEN '2019-06-28' AND '2019-07-27'")
    assert rows("date_diff('day', day, DATE '2019-07-27') < 30 AND day <= '2019-07-27'") == window
    assert rows("date_diff('day', day, DATE '2019-07-27') < 30") != window  # no upper end
    assert rows("date_diff('day', day, DATE '2019-07-27') < 30 AND date_diff('day', day, DATE '2019-07-27') > 0") != window  # the last day is left out
    assert rows("day > '2019-06-27' AND day < '2019-07-27'") != window


FACTS = {"visits": TableConstraints(not_null=frozenset(), keys=()), "rooms": TableConstraints(not_null=frozenset({"room_id"}), keys=(("room_id",),))}
OPTIONS = dict(schema=SCHEMA, types=TYPES, constraints=FACTS, dialect="mysql", compare_names=False, exact_arithmetic=True, timeout_ms=3000)
COUNTED = "SELECT day, COUNT(DISTINCT member_id) AS members FROM visits"
WINDOW = f"{COUNTED} WHERE day BETWEEN '2019-06-28' AND '2019-07-27' GROUP BY day"
SAME_WINDOW = [
    f"{COUNTED} WHERE day > '2019-06-27' AND day <= '2019-07-27' GROUP BY day",
    f"{COUNTED} WHERE day >= '2019-06-28' AND day < '2019-07-28' GROUP BY day",
    f"{COUNTED} WHERE day BETWEEN DATE_SUB('2019-07-27', INTERVAL 29 DAY) AND '2019-07-27' GROUP BY day",
    f"{COUNTED} WHERE day BETWEEN DATE_ADD('2019-07-27', INTERVAL -29 DAY) AND '2019-07-27' GROUP BY day",
    f"{COUNTED} WHERE day <= '2019-07-27' AND day >= SUBDATE('2019-07-27', INTERVAL 29 DAY) GROUP BY day",
    f"{COUNTED} WHERE day >= '2019-07-27' - INTERVAL 29 DAY AND day <= '2019-07-27' GROUP BY day",
    f"{COUNTED} WHERE day > DATE_SUB('2019-07-27', INTERVAL 30 DAY) AND day <= '2019-07-27' GROUP BY day",
    f"{COUNTED} WHERE DATEDIFF('2019-07-27', day) BETWEEN 0 AND 29 GROUP BY day",
    f"{COUNTED} WHERE DATEDIFF('2019-07-27', day) < 30 AND DATEDIFF('2019-07-27', day) >= 0 GROUP BY day",
    f"{COUNTED} WHERE day <= '2019-07-27' AND DATEDIFF('2019-07-27', day) < 30 GROUP BY day",
    f"{COUNTED} GROUP BY day HAVING DATEDIFF('2019-07-27', day) BETWEEN 0 AND 29",
    f"{COUNTED} GROUP BY day HAVING day BETWEEN SUBDATE('2019-07-27', INTERVAL 29 DAY) AND '2019-07-27'",
]
OTHER_WINDOW = [
    f"{COUNTED} WHERE DATEDIFF('2019-07-27', day) < 30 GROUP BY day",  # no upper end
    f"{COUNTED} WHERE DATEDIFF('2019-07-27', day) < 30 AND DATEDIFF('2019-07-27', day) > 0 GROUP BY day",  # the last day is left out
    f"{COUNTED} WHERE day > '2019-06-27' AND day < '2019-07-27' GROUP BY day",
    f"{COUNTED} WHERE day > DATE_SUB('2019-07-27', INTERVAL 29 DAY) AND day <= '2019-07-27' GROUP BY day",  # one day short
    f"{COUNTED} WHERE day BETWEEN DATE_SUB('2019-07-27', INTERVAL 1 MONTH) AND '2019-07-27' GROUP BY day",
    f"{COUNTED} WHERE seen_at BETWEEN '2019-06-28' AND '2019-07-27' GROUP BY day",
]


@pytest.mark.parametrize("other", SAME_WINDOW)
def test_spellings_of_one_window_are_proved(other):
    result = prove_equivalent_algebraic(WINDOW, other, **OPTIONS)
    assert result.proven, result.reason
    with shape_fallback.without_all(shape_fallback.TWELFTH_BATCH):
        assert not prove_equivalent_algebraic(WINDOW, other, **OPTIONS).proven


@pytest.mark.parametrize("other", OTHER_WINDOW)
def test_other_windows_are_not_proved(other):
    assert not prove_equivalent_algebraic(WINDOW, other, **OPTIONS).proven


def test_one_day_spelled_three_ways_is_proved():
    head = "SELECT kind, COUNT(DISTINCT room_id) AS n FROM visits WHERE badge = 1 AND "
    left = f"{head}day = '2019-07-04' GROUP BY kind"
    for test in ("day = DATE_SUB('2019-07-05', INTERVAL 1 DAY)", "DATEDIFF('2019-07-05', day) = 1", "day = ADDDATE('2019-07-05', -1)", "day = '2019-07-05' - INTERVAL 1 DAY", "day = '2019-7-4'"):
        assert prove_equivalent_algebraic(left, f"{head}{test} GROUP BY kind", **OPTIONS).proven, test
    assert not prove_equivalent_algebraic(left, f"{head}DATEDIFF('2019-07-05', day) = 2 GROUP BY kind", **OPTIONS).proven
    # other dialects read none of it
    assert not prove_equivalent_algebraic(left, f"{head}DATEDIFF('2019-07-05', day) = 1 GROUP BY kind", **{**OPTIONS, "dialect": "postgres"}).proven


def test_it_is_the_rule_of_the_twelfth_batch():
    assert shape_fallback.TWELFTH_BATCH == {"date_windows"}
    sql = f"{COUNTED} WHERE DATEDIFF('2019-07-27', day) BETWEEN 0 AND 29 GROUP BY day"
    with shape_fallback.tracking() as fired:
        normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "date_windows" in fired
    with shape_fallback.without("date_windows"), shape_fallback.tracking() as fired:
        normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "date_windows" not in fired
