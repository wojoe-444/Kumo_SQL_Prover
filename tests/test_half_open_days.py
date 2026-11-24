"""Tests of a DATE column against a day, written with ``>=`` and ``<`` only (``half_open_days``, MySQL).

The rewriting is replayed on DuckDB over generated days (the days around each constant, month ends, leap days and
NULL): each test gives the same value before and after on every row, and a test one day off does not.
``date_windows`` was checked on a MySQL server for the same step the other way round (``d < day`` as ``d <= the day
before``); this module was not run on a server.
"""

import datetime

import duckdb
import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.half_open_days import read_half_open_days
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"shops": ["shop_id", "city", "region"], "calls": ["call_id", "shop_id", "seen", "logged", "note"]}
TYPES = {
    "shops": {"shop_id": "INT", "city": "VARCHAR", "region": "VARCHAR"},
    "calls": {"call_id": "INT", "shop_id": "INT", "seen": "DATE", "logged": "DATETIME", "note": "VARCHAR"},
}
HEAD = "SELECT call_id FROM calls WHERE "
GROUPED = "SELECT shop_id FROM calls GROUP BY shop_id HAVING "


def _read(sql: str, dialect: str = "mysql", types=TYPES):
    tree = sqlglot.parse_one(sql, read=dialect)
    before = tree.sql(dialect=dialect)
    new, wrote, bound = read_half_open_days(tree, SCHEMA, types, dialect)
    assert tree.sql(dialect=dialect) == before  # the tree given is not changed
    assert wrote or new is tree
    return (new.sql(dialect=dialect) if wrote else None), bound


def written(test: str, head: str = HEAD) -> str | None:
    text, _ = _read(head + test)
    return None if text is None else text[len(head) :]


READ = [
    ("seen <= '2019-03-31'", "seen < '2019-04-01'"),
    ("seen <= '2018-12-31'", "seen < '2019-01-01'"),
    ("seen <= '2019-02-28'", "seen < '2019-03-01'"),
    ("seen <= '2020-02-28'", "seen < '2020-02-29'"),  # a leap year
    ("'2019-03-31' >= seen", "seen < '2019-04-01'"),
    ("calls.seen <= '2019-3-31'", "calls.seen < '2019-04-01'"),
    ("seen BETWEEN '2019-01-01' AND '2019-03-31'", "(seen >= '2019-01-01' AND seen < '2019-04-01')"),
    ("NOT seen BETWEEN '2019-01-01' AND '2019-03-31'", "NOT (seen >= '2019-01-01' AND seen < '2019-04-01')"),
    ("seen >= '2019-01-01' AND seen <= '2019-03-31'", "seen >= '2019-01-01' AND seen < '2019-04-01'"),
    # the tests date_windows writes out
    ("DATE(seen) < '2019-01-01'", "seen < '2019-01-01'"),
    ("DATE(seen) > '2019-03-31'", "seen >= '2019-04-01'"),
    ("seen > '2019-03-31'", "seen >= '2019-04-01'"),
    ("seen <= DATE('2019-03-31')", "seen < '2019-04-01'"),
    ("seen < DATE('2019-04-01')", "seen < '2019-04-01'"),
    ("DATEDIFF('2019-03-31', seen) >= 0", "seen < '2019-04-01'"),
    ("seen <= DATE_SUB('2019-04-02', INTERVAL 2 DAY)", "seen < '2019-04-01'"),
]

# a test whose two sides do not name whole days, or that is already in the form
LEFT_ALONE = [
    "seen < '2019-04-01'",
    "seen >= '2019-01-01'",
    "seen >= '2019-01-01' AND seen < '2019-04-01'",
    "seen = '2019-03-31'",
    "seen <> '2019-03-31'",
    "logged <= '2019-03-31'",  # a DATETIME: noon of the 31st is after it and before April
    "logged BETWEEN '2019-01-01' AND '2019-03-31'",
    "DATE(logged) <= '2019-03-31'",
    "note <= '2019-03-31'",  # text
    "seen <= '2019-03-31 12:00:00'",
    "seen <= '9999-12-31'",  # there is no day after
    "seen <= note",
    "seen <= logged",
    "call_id <= 20190331",
    "seen <= '2019-02-30'",  # not a day
]


@pytest.mark.parametrize("test, expected", READ)
def test_an_upper_test_is_written_with_the_day_after(test, expected):
    assert written(test) == expected


@pytest.mark.parametrize("test", LEFT_ALONE)
def test_other_tests_are_left_alone(test):
    assert written(test) is None


def test_the_bounds_of_a_group_are_read_and_asked_for():
    assert _read(GROUPED + "MAX(seen) <= '2019-03-31'") == (GROUPED + "MAX(seen) < '2019-04-01'", True)
    assert _read(GROUPED + "MIN(seen) >= '2019-01-01' AND MAX(seen) <= '2019-03-31'") == (GROUPED + "MIN(seen) >= '2019-01-01' AND MAX(seen) < '2019-04-01'", True)
    assert _read(GROUPED + "MIN(seen) BETWEEN '2019-01-01' AND '2019-03-31'") == (GROUPED + "(MIN(seen) >= '2019-01-01' AND MIN(seen) < '2019-04-01')", True)
    assert _read(GROUPED + "MIN(seen) >= '2019-01-01'") == (None, True)  # nothing to write, and the last step is worth running
    assert _read(GROUPED + "MAX(logged) <= '2019-03-31'") == (None, False)
    assert _read(GROUPED + "COUNT(seen) <= 3") == (None, False)
    assert _read(HEAD + "seen <= '2019-03-31'")[1] is False


def test_a_derived_table_passes_its_date_column_on():
    sql = "SELECT d.shop_id FROM (SELECT shop_id, seen FROM calls) AS d WHERE d.seen <= '2019-03-31'"
    assert _read(sql)[0] == sql.replace("<= '2019-03-31'", "< '2019-04-01'")
    computed = "SELECT d.shop_id FROM (SELECT shop_id, logged AS seen FROM calls) AS d WHERE d.seen <= '2019-03-31'"
    assert _read(computed)[0] is None


def test_only_mysql_with_declared_types_is_read():
    for dialect in ("postgres", "sqlite", "duckdb", None):
        tree = sqlglot.parse_one(HEAD + "seen <= '2019-03-31'")
        assert read_half_open_days(tree, SCHEMA, TYPES, dialect) == (tree, False, False)
    tree = sqlglot.parse_one(HEAD + "seen <= '2019-03-31'", read="mysql")
    assert read_half_open_days(tree, SCHEMA, None, "mysql") == (tree, False, False)
    no_dates = {"shops": TYPES["shops"], "calls": {**TYPES["calls"], "seen": "DATETIME"}}
    assert read_half_open_days(tree, SCHEMA, no_dates, "mysql") == (tree, False, False)


def test_without_date_windows_only_the_plain_tests_are_read():
    with shape_fallback.without("date_windows"):
        assert written("seen <= '2019-03-31'") == "seen < '2019-04-01'"
        assert written("DATE(seen) > '2019-03-31'") is None


def _days() -> list:
    days = {None, datetime.date(1000, 1, 1), datetime.date(9999, 12, 30), datetime.date(9999, 12, 31)}
    for constant in ("2018-12-31", "2019-01-01", "2019-02-28", "2019-03-31", "2019-04-01", "2019-04-02", "2020-02-28", "2020-02-29"):
        day = datetime.date.fromisoformat(constant)
        days.update(day + datetime.timedelta(days=offset) for offset in range(-3, 4))
    return sorted(days, key=lambda day: (day is not None, day))


def _values(database, test: str) -> list:
    return database.execute(f"SELECT call_id, ({test}) FROM calls ORDER BY call_id").fetchall()


def _database():
    database = duckdb.connect()
    database.execute("CREATE TABLE calls (call_id INT, shop_id INT, seen DATE, logged TIMESTAMP, note VARCHAR)")
    database.executemany("INSERT INTO calls VALUES (?, 1, ?, NULL, NULL)", list(enumerate(_days())))
    return database


# DuckDB spells these as MySQL does
REPLAYED = [test for test, _ in READ if "DATEDIFF" not in test and "DATE_SUB" not in test and "DATE(" not in test and "2019-3-31" not in test]


@pytest.mark.parametrize("test", REPLAYED)
def test_replay_on_generated_days(test):
    database = _database()
    after = written(test)
    values = _values(database, test)
    assert _values(database, after) == values
    assert {value for _, value in values} == {True, False, None}  # the rows tell the two sides of the boundary apart


@pytest.mark.parametrize("test, expected", [(test, expected) for test, expected in READ if test in REPLAYED])
def test_controls_one_day_off_differ(test, expected):
    database = _database()
    day = sqlglot.parse_one(expected).find_all(sqlglot.exp.Literal)
    last = list(day)[-1].this
    off = (datetime.date.fromisoformat(last) + datetime.timedelta(days=1)).isoformat()
    assert _values(database, expected.replace(last, off)) != _values(database, test)


OPTIONS = dict(
    schema=SCHEMA,
    types=TYPES,
    constraints={
        "shops": TableConstraints(not_null=frozenset({"shop_id", "city", "region"}), keys=(("shop_id",),)),
        "calls": TableConstraints(not_null=frozenset({"call_id", "shop_id", "seen", "logged"}), keys=(("call_id",),)),
    },
    compare_names=False,
    dialect="mysql",
    exact_arithmetic=True,
    timeout_ms=3000,
)
# shops whose every call was in the first week of 2024: the bounds of the group, and "no call outside the week"
BOUNDS = "SELECT s.shop_id, s.city FROM shops AS s JOIN (SELECT shop_id FROM calls GROUP BY shop_id HAVING MIN({0}) >= '2024-01-01' AND MAX({0}) <= '2024-01-07') AS v ON s.shop_id = v.shop_id"
NO_CALL_OUTSIDE = (
    "WITH inside AS (SELECT s.shop_id, s.city FROM shops AS s JOIN calls AS c ON s.shop_id = c.shop_id WHERE c.{0} BETWEEN '2024-01-01' AND '2024-01-07'),"
    " outside AS (SELECT s.shop_id FROM shops AS s JOIN calls AS c ON s.shop_id = c.shop_id WHERE {1})"
    " SELECT DISTINCT inside.shop_id, inside.city FROM inside WHERE inside.shop_id NOT IN (SELECT outside.shop_id FROM outside)"
)
OUTSIDE = [
    "DATE(c.seen) < '2024-01-01' OR DATE(c.seen) > '2024-01-07'",
    "c.seen <= '2023-12-31' OR c.seen >= '2024-01-08'",
]


@pytest.mark.parametrize("outside", OUTSIDE)
def test_a_pair_is_proved_and_only_with_the_rule(outside):
    left, right = BOUNDS.format("seen"), NO_CALL_OUTSIDE.format("seen", outside)
    with shape_fallback.tracking() as fired:
        assert prove_equivalent_algebraic(left, right, **OPTIONS).proven
    assert "half_open_days" in fired
    with shape_fallback.without("half_open_days"):
        assert not prove_equivalent_algebraic(left, right, **OPTIONS).proven


@pytest.mark.parametrize(
    "column, outside",
    [
        ("seen", "DATE(c.seen) < '2024-01-01' OR DATE(c.seen) > '2024-01-08'"),  # a day more
        ("seen", "DATE(c.seen) <= '2024-01-01' OR DATE(c.seen) > '2024-01-07'"),  # the first day is outside
        # a DATETIME: a call at noon of the 7th is after the group's upper bound and not outside the week by its day
        ("logged", "DATE(c.logged) < '2024-01-01' OR DATE(c.logged) > '2024-01-07'"),
        ("logged", "c.logged <= '2023-12-31' OR c.logged >= '2024-01-08'"),
    ],
)
def test_pairs_that_differ_stay_unproved(column, outside):
    assert not prove_equivalent_algebraic(BOUNDS.format(column), NO_CALL_OUTSIDE.format(column, outside), **OPTIONS).proven


def test_it_is_a_rule_of_the_twenty_seventh_step():
    assert "half_open_days" in shape_fallback.TWENTY_SEVENTH_BATCH
