"""``x = y - 1``, ``y - x = 1`` and ``x + 1 = y`` written one way for whole numbers (``shifted_equalities``)."""

import itertools
import sqlite3

import pytest
import sqlglot

from kumo_sql_prover.shifted_equalities import write_shifted_equalities

TYPES = {"visit": {"owner": "INT", "day": "DATE", "room": "INT", "floor": "BIGINT", "note": "VARCHAR", "fee": "DOUBLE", "cost": "DECIMAL(8, 2)"}}
ROOMS = "SELECT a.owner FROM visit AS a JOIN visit AS b ON {} WHERE a.owner = b.owner"
WRITTEN = "a.room + 1 = b.floor"
SPELLINGS = [
    "a.room + 1 = b.floor",
    "b.floor = a.room + 1",
    "1 + a.room = b.floor",
    "b.floor = (1 + a.room)",
    "a.room = b.floor - 1",
    "b.floor - 1 = a.room",
    "b.floor - a.room = 1",
    "1 = b.floor - a.room",
    "a.room - b.floor = -1",
    "a.room - -1 = b.floor",
    "b.floor + -1 = a.room",
    "(b.floor) - (1) = (a.room)",
]


def reading(test: str, types=TYPES, dialect: str = "mysql", frame: str = ROOMS) -> str | None:
    tree, changed = write_shifted_equalities(sqlglot.parse_one(frame.format(test), read=dialect), types, dialect)
    return tree.sql(dialect=dialect) if changed else None


@pytest.mark.parametrize("test", SPELLINGS[1:])
def test_every_spelling_is_written_the_same_way(test):
    assert reading(test) == ROOMS.format(WRITTEN)


def test_what_is_written_already_stays_and_larger_steps_keep_their_size():
    assert reading(WRITTEN) is None
    assert reading("a.room = b.floor - 30") == ROOMS.format("a.room + 30 = b.floor")
    assert reading("a.room = b.floor + 30") == ROOMS.format("b.floor + 30 = a.room")
    assert reading("a.room - b.floor = 0") is None
    assert reading("a.room + 0 = b.floor") is None


def test_dates_are_read_in_mysql_only_and_never_beside_a_whole_number():
    assert reading("a.day = b.day - 1") == ROOMS.format("a.day + 1 = b.day")
    assert reading("b.day - a.day = 1") == ROOMS.format("a.day + 1 = b.day")
    for dialect in ("sqlite", "duckdb", "postgres", "bigquery"):
        assert reading("a.day = b.day - 1", dialect=dialect) is None, dialect
        assert reading("a.room = b.floor - 1", dialect=dialect) is not None, dialect
    assert reading("a.day = b.room - 1") is None
    assert reading("b.room - a.day = 1") is None


def test_the_type_is_read_through_a_derived_table():
    frame = "SELECT 1 FROM (SELECT owner, MIN(day) AS first, MAX(room) AS top, COUNT(*) AS n, MIN(day + 1) AS later, MIN(note) AS word FROM visit GROUP BY owner) AS f JOIN visit AS v ON {}"
    assert reading("f.first = v.day - 1", frame=frame) == frame.format("f.first + 1 = v.day")
    assert reading("v.room - f.top = 2", frame=frame) == frame.format("f.top + 2 = v.room")
    assert reading("f.n = v.room - 1", frame=frame) is None  # a count is a whole number, but only columns are read
    assert reading("f.later = v.day - 1", frame=frame) is None
    assert reading("f.word = v.room - 1", frame=frame) is None
    nested = "SELECT 1 FROM (SELECT g.owner, g.first FROM (SELECT owner, MIN(day) AS first FROM visit GROUP BY owner) AS g) AS f JOIN visit AS v ON {}"
    assert reading("f.first = v.day - 1", frame=nested) == nested.format("f.first + 1 = v.day")
    renamed = "SELECT 1 FROM (SELECT owner, MIN(day) FROM visit GROUP BY owner) AS f(owner, first) JOIN visit AS v ON {}"
    assert reading("f.first = v.day - 1", frame=renamed, dialect="postgres") is None


LEFT_ALONE = [
    "a.note = b.note - 1",  # text: MySQL compares floating-point numbers
    "a.fee = b.fee - 1",  # floating point: a sum rounds
    "a.cost = b.cost - 1",
    "a.room = b.fee - 1",
    "a.room = b.note - 1",
    "a.room = b.floor - 1.0",
    "a.room = b.floor - '1'",
    "a.room = b.floor * 2",
    "a.room = b.floor - a.owner",  # three columns
    "a.room - 1 = b.floor - 2",
    "a.room >= b.floor - 1",
    "a.room <=> b.floor - 1",
    "a.room = COALESCE(b.floor, 0) - 1",
    "a.room = DATE_ADD(b.day, INTERVAL 1 DAY)",
    "DATEDIFF(b.day, a.day) = 1",
    "a.day = b.day - INTERVAL 1 DAY",
    "room = b.floor - 1",  # no alias on a column
    "a.room = c.floor - 1",  # not a source of this select
]


@pytest.mark.parametrize("test", LEFT_ALONE)
def test_other_comparisons_are_left_alone(test):
    sqlglot.parse_one(ROOMS.format(test), read="mysql")
    assert reading(test) is None


def test_it_needs_declared_types_and_reads_each_select_by_its_own_sources():
    assert reading("a.room = b.floor - 1", types=None) is None
    assert reading("a.room = b.floor - 1", types={"visit": {"room": "INT"}}) is None
    inner = "SELECT 1 FROM visit AS a WHERE EXISTS(SELECT 1 FROM visit AS b WHERE {})"
    assert reading("a.room = b.floor - 1", frame=inner) is None  # a is the outer select's source: not read
    assert reading("b.room = b.floor - 1", frame=inner) == inner.format("b.room + 1 = b.floor")


def test_the_spellings_agree_on_whole_numbers_and_nulls_in_sqlite():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE visit (owner INT, day INT, room INT, floor BIGINT, note VARCHAR, fee DOUBLE, cost DECIMAL)")
    values = [None, -3, -1, 0, 1, 2, 3, 7]
    con.executemany("INSERT INTO visit (owner, room, floor) VALUES (1, ?, ?)", list(itertools.product(values, values)))
    for test in SPELLINGS:
        # the truth value itself, NULL included, on every pair of values
        query = "SELECT a.rowid, {} FROM visit AS a JOIN visit AS b ON a.rowid = b.rowid ORDER BY 1"
        assert con.execute(query.format(test)).fetchall() == con.execute(query.format(WRITTEN)).fetchall(), test
    assert con.execute("SELECT COUNT(*) FROM visit AS a JOIN visit AS b ON a.rowid = b.rowid AND a.room + 1 = b.floor").fetchone()[0] == 4


def test_a_mysql_date_in_arithmetic_is_the_number_yyyymmdd():
    """Worked by hand from MySQL's rule (a DATE in a numeric context is YYYYMMDD): the three spellings say the same
    thing about two whole numbers, and none of them says "the day after" across the end of a month."""

    def number(date: str) -> int:
        return int(date.replace("-", ""))

    for first, day in (("2024-03-01", "2024-03-02"), ("2024-01-31", "2024-02-01"), ("2024-01-31", "2024-01-31"), ("2023-12-31", "2024-01-01")):
        x, y = number(first), number(day)
        assert (x == y - 1) == (y - x == 1) == (x + 1 == y) == (y - 1 == x)
    assert number("2024-01-31") + 1 != number("2024-02-01")  # the day after, and not equal
    # so nothing here reads the sum as a date function
    assert reading("DATEDIFF(b.day, a.day) = 1") is None and reading("b.day = DATE_ADD(a.day, INTERVAL 1 DAY)") is None


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402

SCHEMA = {"visit": ["owner", "day", "room", "floor", "note", "fee", "cost"]}
OPTIONS = dict(schema=SCHEMA, types=TYPES, compare_names=False, dialect="mysql")
FIRST = "SELECT v.owner FROM (SELECT owner, MIN(day) AS first FROM visit GROUP BY owner) AS f JOIN visit AS v ON v.owner = f.owner AND {}"


def test_the_spellings_prove_equal_and_a_date_function_does_not():
    plus = FIRST.format("f.first + 1 = v.day")
    for other in ("f.first = v.day - 1", "v.day - f.first = 1", "v.day - 1 = f.first"):
        assert prove_equivalent_algebraic(plus, FIRST.format(other), **OPTIONS).proven, other
    for other in ("DATEDIFF(v.day, f.first) = 1", "v.day = DATE_ADD(f.first, INTERVAL 1 DAY)", "f.first + 2 = v.day", "v.day + 1 = f.first"):
        assert not prove_equivalent_algebraic(plus, FIRST.format(other), **OPTIONS).proven, other
    text = dict(OPTIONS, types={"visit": {**TYPES["visit"], "day": "VARCHAR"}})
    assert not prove_equivalent_algebraic(plus, FIRST.format("v.day - f.first = 1"), **text).proven


def test_it_is_a_rule_of_the_tenth_batch():
    assert "shifted_equalities" in shape_fallback.TENTH_BATCH
    sql = FIRST.format("f.first = v.day - 1")
    with shape_fallback.tracking() as fired:
        normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "shifted_equalities" in fired
    with shape_fallback.without("shifted_equalities"), shape_fallback.tracking() as fired:
        normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "shifted_equalities" not in fired
