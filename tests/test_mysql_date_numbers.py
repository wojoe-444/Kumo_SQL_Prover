"""MySQL does arithmetic on a date as a number (2024-01-31 is 20240131), and the counterexample search's DuckDB
text takes that reading when it is told the column types (``mysql_date_numbers``).

Every expected row below is what a MySQL 8.0 server returns for the query as written on the same made-up table;
the test checks that DuckDB returns it for the translated text. The table has a stay that ends on the first day
of a month and one that ends on the first day of a year, where "the day after" and "the number plus one" part.
"""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from kumo_sql_prover import counterexample as cx  # noqa: E402
from kumo_sql_prover import mysql_date_numbers as dates  # noqa: E402

TYPES = {"room": "INT", "arrive": "DATE", "depart": "DATE", "bell": "TIME", "code": "INT", "note": "VARCHAR"}
KINDS = {"stays": {name: dates.column_kind(declared) for name, declared in TYPES.items()}}
COLUMNS = set(TYPES)
TABLES = {"stays": set(TYPES)}
ROWS = """(1, '2024-01-31', '2024-02-01', '10:59:59', 20240201, 'a'), (2, '2024-01-15', '2024-01-16', '10:00:00', 20240116, 'b'),
          (3, NULL, '2024-01-16', NULL, NULL, NULL), (4, '2024-12-31', '2025-01-01', '00:00:00', 1, 'c')"""


def database():
    db = duckdb.connect(":memory:")
    db.execute("CREATE TABLE stays (room BIGINT, arrive DATE, depart DATE, bell TIME, code BIGINT, note VARCHAR)")
    db.execute(f"INSERT INTO stays VALUES {ROWS}")
    return db


def text(sql: str, notes: set | None = None) -> str:
    return cx.to_duckdb(sql, "mysql", COLUMNS, TABLES, KINDS, notes)


def rows(sql: str) -> list:
    """The rows DuckDB returns for the search's text, conditions as MySQL's 1 and 0, in a fixed order."""

    out = database().execute(text(sql)).fetchall()
    return sorted(([int(value) if isinstance(value, bool) else value for value in row] for row in out), key=repr)


N = None
# (the MySQL query, the rows MySQL 8.0 returns for it)
READINGS = [
    ("SELECT room, arrive + 1, arrive - 1, depart - arrive FROM stays", [[1, 20240132, 20240130, 70], [2, 20240116, 20240114, 1], [3, N, N, N], [4, 20241232, 20241230, 8870]]),
    (
        "SELECT room, arrive + 1 = depart, depart - 1 = arrive, depart - arrive = 1, arrive + INTERVAL 1 DAY = depart, DATEDIFF(depart, arrive) = 1 FROM stays",
        [[1, 0, 0, 0, 1, 1], [2, 1, 1, 1, 1, 1], [3, N, N, N, N, N], [4, 0, 0, 0, 1, 1]],
    ),
    # a date against a number that is not a constant is compared as a number
    (
        "SELECT room, arrive + 1 = code, arrive = code - 1, depart = code, code = depart, code < depart, code + 0 = depart FROM stays",
        [[1, 0, 0, 1, 1, 0, 1], [2, 1, 1, 1, 1, 0, 1], [3, N, N, N, N, N, N], [4, 0, 0, 0, 0, 1, 0]],
    ),
    # a date against a whole number that is a constant is compared as a number too, whether or not the number is a date
    (
        "SELECT room, arrive = 20240131, arrive <> 20240131, arrive < 20240132, arrive >= 20240132, arrive = DATE('2024-02-01') - 70, "
        "arrive < DATE('2024-01-31') + 1, depart > DATE('2024-01-31') + 1, arrive > 5, 20240131 = arrive, arrive = (20240130 + 1), arrive = -(-20240131) FROM stays",
        [[1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], [2, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0], [3, N, N, N, N, N, N, 0, N, N, N, N], [4, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0]],
    ),
    # calls that keep a date a date
    (
        "SELECT room, IFNULL(arrive, depart) + 1, IF(room = 1, arrive, depart) + 1, LEAST(arrive, depart) + 1, GREATEST(arrive, depart) + 1, "
        "COALESCE(arrive, depart) + 1, (CASE WHEN room = 1 THEN arrive ELSE depart END) + 1 FROM stays",
        [
            [1, 20240132, 20240132, 20240132, 20240202, 20240132, 20240132],
            [2, 20240116, 20240117, 20240116, 20240117, 20240116, 20240117],
            [3, 20240117, 20240117, N, N, 20240117, 20240117],
            [4, 20241232, 20250102, 20241232, 20250102, 20241232, 20250102],
        ],
    ),
    (
        "SELECT room, (SELECT MIN(arrive) FROM stays AS x WHERE x.room = stays.room) + 1, DATE_ADD(arrive, INTERVAL 1 DAY) + 1, "
        "DATE_SUB(arrive, INTERVAL 1 MONTH) + 1, (arrive + INTERVAL 1 DAY) + 1 FROM stays",
        [[1, 20240132, 20240202, 20231232, 20240202], [2, 20240116, 20240117, 20231216, 20240117], [3, N, N, N, N], [4, 20241232, 20250102, 20241131, 20250102]],
    ),
    (
        "SELECT room, LEAD(arrive) OVER (ORDER BY room) + 1, LAG(arrive) OVER (ORDER BY room) + 1, LAG(arrive, 1, arrive) OVER (ORDER BY room) + 1, "
        "FIRST_VALUE(arrive) OVER (ORDER BY room DESC) + 1, MIN(arrive) OVER () + 1, MAX(depart) OVER (PARTITION BY room) - arrive FROM stays",
        [
            [1, 20240116, N, 20240132, 20241232, 20240116, 70],
            [2, N, 20240132, 20240132, 20241232, 20240116, 1],
            [3, 20241232, 20240116, 20240116, 20241232, 20240116, N],
            [4, N, N, N, 20241232, 20240116, 8870],
        ],
    ),
    ("SELECT room, MIN(arrive) + 1, MAX(arrive) - MIN(arrive) FROM stays GROUP BY room", [[1, 20240132, 0], [2, 20240116, 0], [3, N, N], [4, 20241232, 0]]),
    # through a derived table, a set operation, a WITH name and a star
    (
        "SELECT x.room, x.began + 1, x.began + 1 = s.depart, s.depart - x.began FROM (SELECT room, MIN(arrive) AS began FROM stays GROUP BY room) AS x "
        "JOIN stays AS s ON s.room = x.room",
        [[1, 20240132, 0, 70], [2, 20240116, 1, 1], [3, N, N, N], [4, 20241232, 0, 8870]],
    ),
    (
        "SELECT u.room, u.moment + 1 FROM (SELECT room, arrive AS moment FROM stays UNION ALL SELECT room, depart FROM stays) AS u",
        [[1, 20240132], [1, 20240202], [2, 20240116], [2, 20240117], [3, 20240117], [3, N], [4, 20241232], [4, 20250102]],
    ),
    ("WITH c AS (SELECT room, MIN(arrive) AS began FROM stays GROUP BY room) SELECT c.room, c.began + 1 FROM c", [[1, 20240132], [2, 20240116], [3, N], [4, 20241232]]),
    ("SELECT x.room FROM (SELECT *, arrive AS began FROM stays) AS x WHERE x.began + 1 = x.depart", [[2]]),
    # a column list renames by position: x.room is the arrival
    ("SELECT x.r FROM (SELECT room, arrive, depart FROM stays) AS x (r, room, d) WHERE x.room + 1 = x.d", [[2]]),
    # a time is its hours, minutes and seconds
    ("SELECT room, bell + 1, bell - bell, bell + 1 = bell FROM stays", [[1, 105960, 0, 0], [2, 100001, 0, 0], [3, N, N, N], [4, 1, 0, 0]]),
    # lists, ranges and IN subqueries
    (
        "SELECT room, arrive + 1 = depart + 0, (arrive + 1) IN (depart, arrive), arrive IN (depart - 1, 5), depart > arrive + 1, depart >= arrive + 1, "
        "arrive + 1 BETWEEN arrive AND depart FROM stays",
        [[1, 0, 0, 0, 1, 1, 1], [2, 1, 1, 1, 0, 1, 1], [3, N, N, N, N, N, N], [4, 0, 0, 0, 1, 1, 1]],
    ),
    (
        "SELECT room, code IN (SELECT depart FROM stays), (room, code) IN (SELECT room, depart FROM stays), (room, arrive + 1) IN (SELECT room, depart FROM stays) FROM stays",
        [[1, 1, 1, 0], [2, 1, 1, 1], [3, N, N, N], [4, 0, 0, 0]],
    ),
]


@pytest.mark.parametrize("sql,expected", READINGS)
def test_duckdb_returns_what_mysql_returns(sql, expected):
    assert rows(sql) == sorted(expected, key=repr)


# MySQL answers these; the text is one DuckDB refuses, so the search says "unknown" instead of reading them its own way
REFUSED = [
    "SELECT IFNULL(arrive, 0) + 1 FROM stays",  # a string in MySQL: 2025
    "SELECT NULLIF(arrive, depart) + 1 FROM stays",  # a call this module does not know: the guard
    "SELECT x.v + 1 FROM stays, LATERAL (SELECT arrive AS v) AS x",  # a source this module does not follow: the guard
    "SELECT depart - NULLIF(arrive, depart) FROM stays",
    "SELECT depart IN (SELECT arrive + 1 FROM stays) FROM stays",  # no row matches in MySQL
    "SELECT arrive BETWEEN 20240101 AND 20240132 FROM stays",  # MySQL reads the constants as dates
    "SELECT arrive IN (20240131, 20240132) FROM stays",  # no row in MySQL: one value that is no date empties the list
    "SELECT arrive < 20240131.5 FROM stays",  # MySQL reads the decimal as the 31st
    "SELECT COALESCE(arrive + 1, depart) FROM stays",
    # a number column of a derived table: in the select list MySQL compares numbers (room 2 matches), as a join
    # condition it looks the date up in the key it builds for the derived table and matches no room
    "SELECT a.room FROM stays AS a JOIN (SELECT room, MIN(arrive) + 1 AS n FROM stays GROUP BY room) AS x ON a.room = x.room AND a.depart = x.n",
    "SELECT DATEDIFF(arrive + 1, depart) FROM stays",
]


@pytest.mark.parametrize("sql", REFUSED)
def test_what_is_not_read_is_refused_by_duckdb(sql):
    with pytest.raises(duckdb.Error):
        database().execute(text(sql)).fetchall()


UNTOUCHED = [
    "SELECT room + 1, code - room, code * 2 / 3, COUNT(*) + 1, SUM(code) / COUNT(*) - 1 FROM stays GROUP BY room, code",
    "SELECT room FROM stays WHERE DATEDIFF(depart, arrive) = 1 AND arrive < depart AND DATE_ADD(arrive, INTERVAL 1 HOUR) > arrive",
    "SELECT room, ROUND(AVG(code), 2) + IFNULL(MAX(code), 0), (CASE WHEN room > 1 THEN 1 ELSE 0 END) + 1 FROM stays GROUP BY room",
    "SELECT x.total + 1 FROM (SELECT SUM(code) AS total FROM stays) AS x",
    "SELECT room FROM stays WHERE arrive = '2024-01-31' OR depart BETWEEN '2024-01-01' AND '2024-12-31'",
]


@pytest.mark.parametrize("sql", UNTOUCHED)
def test_arithmetic_on_numbers_and_date_arithmetic_stay_as_written(sql):
    notes: set = set()
    assert text(sql, notes) == cx.to_duckdb(sql, "mysql", COLUMNS, TABLES)
    assert not notes


# what MySQL 8.0 returns, as text: a date moved by whole days is a date (a string, when it started as one)
MOVED = (
    "SELECT room, arrive + INTERVAL 1 DAY, DATE_ADD(arrive, INTERVAL 1 MONTH), CONCAT(room, ',', arrive + INTERVAL 1 DAY), "
    "DATE_ADD('2024-01-31', INTERVAL 1 DAY), CONCAT(room, ',', DATE_ADD(arrive, INTERVAL 1 HOUR)), CONCAT('x', '2024-01-31' + INTERVAL 1 MONTH) FROM stays",
    [
        ["1", "2024-02-01", "2024-02-29", "1,2024-02-01", "2024-02-01", "1,2024-01-31 01:00:00", "x2024-02-29"],
        ["2", "2024-01-16", "2024-02-15", "2,2024-01-16", "2024-02-01", "2,2024-01-15 01:00:00", "x2024-02-29"],
        ["3", N, N, N, "2024-02-01", N, "x2024-02-29"],
        ["4", "2025-01-01", "2025-01-31", "4,2025-01-01", "2024-02-01", "4,2024-12-31 01:00:00", "x2024-02-29"],
    ],
)


def test_a_date_moved_by_whole_days_stays_a_date():
    sql, expected = MOVED
    got = database().execute(text(sql)).fetchall()
    assert sorted(([None if value is None else str(value) for value in row] for row in got), key=repr) == sorted(expected, key=repr)
    # without the column types DuckDB answers with a timestamp, which is another value once it is an output
    plain = database().execute(cx.to_duckdb(sql, "mysql", COLUMNS, TABLES)).fetchall()
    assert "2024-02-01 00:00:00" in {str(row[1]) for row in plain}
    notes: set = set()
    assert "CAST(arrive + INTERVAL '1' DAY AS DATE) = depart" in text("SELECT room FROM stays WHERE arrive + INTERVAL 1 DAY = depart", notes) and not notes


def test_a_name_sqlglot_takes_for_a_keyword_is_quoted():
    import sqlglot

    sql = "WITH install AS (SELECT room, MIN(arrive) AS install FROM stays GROUP BY room) SELECT install FROM install"
    written = text(sql)
    assert written.count('"install"') == 4
    sqlglot.parse_one(written, read="duckdb")  # the search reads its own text again to check arbitrary picks
    assert cx.guard_arbitrary_picks(written) == written
    assert len(database().execute(written).fetchall()) == 4


def test_the_reading_is_mysqls_only_and_needs_the_column_types():
    sql = "SELECT room FROM stays WHERE arrive + 1 = depart"
    plain = cx.to_duckdb(sql, "mysql", COLUMNS, TABLES)
    assert "STRFTIME" not in plain and "STRFTIME" not in cx.to_duckdb(sql, "postgres", COLUMNS, TABLES, KINDS)
    notes: set = set()
    assert text(sql, notes).count("STRFTIME") == 2 and notes == {cx.DATE_NUMBERS}
    # a guard alone is not a date read as a number
    notes = set()
    assert "ABS(" in text("SELECT NULLIF(code, room) + 1 FROM stays", notes) and not notes


def test_the_guard_keeps_a_number_as_it_is():
    out = database().execute(text("SELECT room, NULLIF(code, room) + 1, 2 - NULLIF(code, 1) FROM stays")).fetchall()
    assert sorted(out, key=repr) == [(1, 20240202, -20240199), (2, 20240117, -20240114), (3, None, None), (4, 2, None)]


def test_column_kinds():
    assert [dates.column_kind(name) for name in ("DATE", "date", "TIME", "INT", "DECIMAL(10, 2)", "VARCHAR(20)", "DATETIME", "BOOL", "ENUM")] == [
        "date", "date", "time", "number", "number", None, None, None, None,
    ]  # fmt: skip
    assert dates.column_kind("BOOL", as_number=True) == "number"


SPEC = cx.Spec(
    tables={
        "stays": cx.Table(
            "stays",
            [cx.Column("room", "INT", not_null=True), cx.Column("arrive", "DATE"), cx.Column("depart", "DATE"), cx.Column("code", "INT")],
            primary_key=("room",),
        )
    }
)
NUMBER_PLUS_ONE = "SELECT room FROM stays WHERE arrive + 1 = depart"
DAY_AFTER = "SELECT room FROM stays WHERE DATE_ADD(arrive, INTERVAL 1 DAY) = depart"
DIFFERENCE_OF_ONE = "SELECT room FROM stays WHERE depart - arrive = 1"
ONE_DAY_APART = "SELECT room FROM stays WHERE DATEDIFF(depart, arrive) = 1"


def test_the_search_tells_the_number_plus_one_from_the_day_after():
    found = cx.Searcher(SPEC, NUMBER_PLUS_ONE, DAY_AFTER).search(200, seed=1)
    assert found is not None
    # the database holds a stay over the end of a month, which only the day after counts
    assert found.left_rows == [] and len(found.right_rows) == 1
    assert cx.Searcher(SPEC, DIFFERENCE_OF_ONE, ONE_DAY_APART).search(200, seed=1) is not None


def test_the_search_finds_nothing_between_two_spellings_of_one_reading():
    assert cx.Searcher(SPEC, NUMBER_PLUS_ONE, DIFFERENCE_OF_ONE).search(200, seed=1) is None
    assert cx.Searcher(SPEC, DAY_AFTER, ONE_DAY_APART).search(200, seed=1) is None


def test_month_ends_are_tried_only_when_a_date_is_read_as_a_number():
    assert "2020-01-31" in cx.Searcher(SPEC, NUMBER_PLUS_ONE, DAY_AFTER).constants.dates
    assert "2020-01-31" not in cx.Searcher(SPEC, DAY_AFTER, ONE_DAY_APART).constants.dates
