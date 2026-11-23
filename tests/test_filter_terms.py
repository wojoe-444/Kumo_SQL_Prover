"""Three readings of a ``WHERE`` that keep its rows (``filter_terms``): shared terms of an ``OR``, terms of an
``EXISTS`` that do not read its rows, and a column pinned to one whole number."""

import random

import pytest
from sqlglot import exp, parse_one

duckdb = pytest.importorskip("duckdb")

from kumo_sql_prover.filter_terms import NAMES, exists_outer_terms, pinned_columns, shared_filter_terms

TYPES = {"seats": {"seat": "INT", "free": "BOOL", "price": "DECIMAL(6,2)", "note": "VARCHAR"}, "rooms": {"room": "INT", "size": "INT"}}
A = "SELECT a.seat FROM seats AS a WHERE"


def apply(rule, sql: str) -> str | None:
    tree = parse_one(sql, read="mysql")
    text = tree.sql(dialect="mysql")
    new = rule(tree)
    assert tree.sql(dialect="mysql") == text  # the select given is not changed
    if new is None:
        return None
    assert rule(new) is None  # once is enough: the rule has nothing more to write
    return new.sql(dialect="mysql")


def pinned(select, types=TYPES):
    return pinned_columns(select, types)


SHARED = [
    (f"{A} (a.free = 1 AND a.seat > 2) OR (a.free = 1 AND a.price < 5)", f"{A} a.free = 1 AND (a.seat > 2 OR a.price < 5)"),
    (
        f"{A} a.free = 1 AND a.seat > 2 OR a.price < 5 AND a.free = 1 OR a.free = 1 AND a.note = 'x'",
        f"{A} a.free = 1 AND (a.seat > 2 OR a.price < 5 OR a.note = 'x')",
    ),
    (f"{A} a.free = 1 OR (a.free = 1 AND a.seat > 2)", f"{A} a.free = 1"),
    (
        f"{A} a.seat > 0 AND ((a.free = 1 AND a.price > 2 AND a.note = 'x') OR (a.note = 'x' AND a.price < 1 AND a.free = 1))",
        f"{A} a.seat > 0 AND (a.free = 1 AND a.note = 'x' AND (a.price > 2 OR a.price < 1))",
    ),
    (
        f"{A} (a.free = 1 AND ((a.seat > 2 AND a.price > 1) OR (a.seat > 2 AND a.price < 0))) OR (a.free = 1 AND a.seat < 0)",
        f"{A} a.free = 1 AND ((a.seat > 2 AND (a.price > 1 OR a.price < 0)) OR a.seat < 0)",
    ),
]

OUTER = [
    (
        f"{A} EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 AND a.free = 1 AND b.free = 1)",
        f"{A} (a.free = 1 AND EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 AND b.free = 1))",
    ),
    (  # under OR; every term moves and the subquery only asks whether the table has a row
        f"{A} a.seat > 3 OR EXISTS(SELECT 1 FROM seats AS b WHERE a.free = 1 AND a.price > 2)",
        f"{A} a.seat > 3 OR (a.free = 1 AND a.price > 2 AND EXISTS(SELECT 1 FROM seats AS b))",
    ),
    (  # a join inside, an OR as the term, a term that reads both stays
        f"{A} a.price > 1 AND (EXISTS(SELECT b.seat FROM seats AS b JOIN rooms AS r ON r.room = b.seat WHERE (a.free = 1 OR a.seat = 4) AND r.size > a.seat) OR a.seat = 9)",
        f"{A} a.price > 1 AND (((a.free = 1 OR a.seat = 4) AND EXISTS(SELECT b.seat FROM seats AS b JOIN rooms AS r ON r.room = b.seat WHERE r.size > a.seat)) OR a.seat = 9)",
    ),
    (  # a source without an alias goes by its table's name
        f"{A} EXISTS(SELECT 1 FROM seats WHERE seats.seat = a.seat + 1 AND a.free = 1)",
        f"{A} (a.free = 1 AND EXISTS(SELECT 1 FROM seats WHERE seats.seat = a.seat + 1))",
    ),
]

PINNED = [
    (
        f"{A} a.free = 1 AND EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat - 1 AND b.free = a.free)",
        f"{A} a.free = 1 AND EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat - 1 AND b.free = 1)",
    ),
    ("SELECT seat FROM seats WHERE free = 1 AND seat > free AND 3 >= free", "SELECT seat FROM seats WHERE free = 1 AND seat > 1 AND 3 >= 1"),
    (  # the literal on the left, a joined table, uses under OR and NOT; a decimal compared with it keeps the column
        "SELECT a.seat FROM seats AS a JOIN rooms AS r ON r.room = a.seat WHERE 2 = r.size AND a.seat < r.size AND (a.free = r.size OR NOT a.price <> r.size)",
        "SELECT a.seat FROM seats AS a JOIN rooms AS r ON r.room = a.seat WHERE 2 = r.size AND a.seat < 2 AND (a.free = 2 OR NOT a.price <> r.size)",
    ),
    (  # against a scalar subquery that selects a whole-number column, and against a sum
        f"{A} a.free = 1 AND ((SELECT b.free FROM seats AS b WHERE b.seat = a.seat - 1) = a.free OR a.seat + 2 > a.free)",
        f"{A} a.free = 1 AND ((SELECT b.free FROM seats AS b WHERE b.seat = a.seat - 1) = 1 OR a.seat + 2 > 1)",
    ),
    (  # only an operand of a comparison is written: the IN keeps the column
        f"{A} (a.seat = 4) AND a.seat IN (SELECT b.seat FROM seats AS b WHERE b.seat <= a.seat)",
        f"{A} (a.seat = 4) AND a.seat IN (SELECT b.seat FROM seats AS b WHERE b.seat <= 4)",
    ),
]


@pytest.mark.parametrize("sql, expected", SHARED)
def test_a_term_every_branch_holds_is_written_once(sql, expected):
    assert apply(shared_filter_terms, sql) == expected


@pytest.mark.parametrize("sql, expected", OUTER)
def test_a_term_that_does_not_read_the_subquery_moves_in_front(sql, expected):
    assert apply(exists_outer_terms, sql) == expected


@pytest.mark.parametrize("sql, expected", PINNED)
def test_a_pinned_column_is_written_as_its_number(sql, expected):
    assert apply(pinned, sql) == expected


def _connection(seed: int):
    rng = random.Random(seed)
    connection = duckdb.connect()
    connection.execute("CREATE TABLE seats (seat INTEGER, free INTEGER, price DECIMAL(6,2), note VARCHAR)")
    connection.execute("CREATE TABLE rooms (room INTEGER, size INTEGER)")
    for seat in rng.sample(range(0, 10), rng.randrange(0, 8)):
        connection.execute("INSERT INTO seats VALUES (?, ?, ?, ?)", [seat, rng.choice((0, 1, 1, 2, None)), rng.choice((0.5, 1, 2, 4.5, None)), rng.choice(("x", "y", None))])
    for room in rng.sample(range(0, 10), rng.randrange(0, 6)):
        connection.execute("INSERT INTO rooms VALUES (?, ?)", [room, rng.choice((1, 2, 2, 5, None))])
    return connection


def _rows(connection, sql: str) -> list:
    return sorted(connection.execute(parse_one(sql, read="mysql").sql(dialect="duckdb")).fetchall(), key=repr)


@pytest.mark.parametrize("sql, expected", SHARED)
def test_shared_terms_keep_the_value_of_the_condition(sql, expected):
    """Not only the rows the filter keeps: the rows on which it is FALSE are the same too."""

    def negated(text: str) -> str:
        tree = parse_one(text, read="mysql")
        tree.args["where"].set("this", exp.Not(this=exp.Paren(this=tree.args["where"].this)))
        return tree.sql(dialect="mysql")

    for seed in range(60):
        connection = _connection(seed)
        assert _rows(connection, sql) == _rows(connection, expected), seed
        assert _rows(connection, negated(sql)) == _rows(connection, negated(expected)), seed


@pytest.mark.parametrize("sql, expected", OUTER + PINNED)
def test_the_reading_returns_the_same_rows(sql, expected):
    for seed in range(60):
        connection = _connection(seed)
        assert _rows(connection, sql) == _rows(connection, expected), seed


SHARED_LEFT_ALONE = [
    f"{A} a.free = 1 AND (a.seat > 2 OR a.price < 5)",
    f"{A} (a.free = 1 AND a.seat > 2) OR (a.free = 0 AND a.price < 5)",
    f"{A} (a.free = 1 AND a.seat > 2) OR (a.FREE = 1 AND a.price < 5)",  # another spelling is another text
    f"{A} (RAND() > 0.5 AND a.seat > 2) OR (RAND() > 0.5 AND a.price < 5)",
    f"{A} (some_function(a.seat) AND a.seat > 2) OR (some_function(a.seat) AND a.price < 5)",
    f"{A} NOT ((a.free = 1 AND a.seat > 2) OR (a.free = 1 AND a.price < 5))",  # only the AND and OR on top are read
    "SELECT a.seat FROM seats AS a",
    "SELECT a.seat, (a.free = 1 AND a.seat > 2) OR (a.free = 1 AND a.price < 5) FROM seats AS a",
]

OUTER_LEFT_ALONE = [
    f"{A} NOT EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 AND a.free = 1)",
    f"{A} EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 AND a.free = 1) IS TRUE",
    f"{A} CASE WHEN EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 AND a.free = 1) THEN 1 ELSE 0 END = 0",
    f"{A} EXISTS(SELECT COUNT(*) FROM seats AS b WHERE b.seat = a.seat + 1 AND a.free = 1)",
    f"{A} EXISTS(SELECT b.note FROM seats AS b WHERE b.seat = a.seat + 1 AND a.free = 1 GROUP BY b.note)",
    f"{A} EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 AND a.free = 1 LIMIT 0)",
    f"{A} EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 AND free = 1)",  # a bare name is the subquery's own first
    f"{A} EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 AND b.free = 1)",
    f"{A} EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 OR a.free = 1)",
    f"{A} EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 AND a.free = (SELECT MAX(size) FROM rooms))",
    f"{A} EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 AND RAND() * a.free > 0.5)",
    f"{A} EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 AND 1 = 1)",
    f"{A} EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = 2 UNION ALL SELECT 1 FROM rooms AS r WHERE a.free = 1)",
    f"{A} EXISTS(SELECT 1 FROM seats AS a WHERE a.free = 1)",  # the name is the subquery's own source
    "SELECT a.seat, EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 AND a.free = 1) FROM seats AS a",
]

PINNED_LEFT_ALONE = [
    f"{A} a.free = 1 OR a.seat > a.free",  # a pin under OR pins nothing
    f"{A} NOT a.free = 1 AND a.seat > a.free",
    f"{A} a.free = 1 AND a.seat + a.free > 2",  # not an operand of a comparison
    f"{A} a.free = 1 AND a.free IN (1, 2) AND a.free BETWEEN 0 AND 3 AND a.free IS NOT NULL",
    f"{A} a.price = 1 AND a.seat > a.price",  # a decimal: 1.00 is not written 1
    f"{A} a.free = 1 AND a.note > a.free",  # against a text the engine converts, and not a column as it does a number
    f"{A} a.free = 1 AND a.price > a.free AND LENGTH(a.note) > a.free AND a.seat / 2 > a.free",
    f"{A} a.free = 1 AND a.free = (SELECT MAX(r.size) FROM rooms AS r) AND a.free = (SELECT r.size FROM rooms AS r UNION SELECT 1)",
    f"{A} a.free = 1 AND a.free = (SELECT d.size FROM (SELECT note AS size FROM seats) AS d) AND a.free = (SELECT size FROM rooms, seats)",
    f"WITH seats AS (SELECT '1' AS seat, '1' AS free) {A} a.free = 1 AND a.seat > a.free",
    f"{A} a.note = 1 AND a.seat > a.note",  # a text that equals 1 need not be '1'
    f"{A} a.free = '1' AND a.seat > a.free",
    f"{A} a.free = 1.0 AND a.seat > a.free",
    f"{A} a.free = -1 AND a.seat > a.free",
    f"{A} a.free = a.seat AND a.seat > a.free",
    f"{A} a.free <=> 1 AND a.seat > a.free",
    f"{A} a.free = 1 AND EXISTS(SELECT 1 FROM rooms AS a WHERE a.size = a.free)",  # another source has taken the name
    "SELECT seat FROM seats WHERE free = 1 AND EXISTS(SELECT 1 FROM rooms WHERE size = free)",  # a bare name inside a subquery
    "SELECT a.seat FROM seats AS a, seats AS b WHERE free = 1 AND a.seat > b.free",  # a bare pin with two sources
    "SELECT d.seat FROM (SELECT seat, free FROM seats) AS d WHERE d.free = 1 AND d.seat > d.free",  # not a declared table
    f"{A} a.free = 1",
    "SELECT a.seat FROM seats AS a JOIN rooms AS r ON r.size = 1 AND a.seat > r.size",  # ON is not the WHERE
    "SELECT a.seat FROM seats AS a GROUP BY a.seat HAVING a.seat = 1 AND MAX(a.free) > a.seat",
]


@pytest.mark.parametrize("sql", SHARED_LEFT_ALONE)
def test_other_conditions_keep_their_terms(sql):
    assert apply(shared_filter_terms, sql) is None


@pytest.mark.parametrize("sql", OUTER_LEFT_ALONE)
def test_other_subqueries_keep_their_terms(sql):
    assert apply(exists_outer_terms, sql) is None


@pytest.mark.parametrize("sql", PINNED_LEFT_ALONE)
def test_other_columns_are_not_pinned(sql):
    assert apply(pinned, sql) is None


def test_nothing_is_pinned_without_declared_types():
    sql = PINNED[0][0]
    assert apply(lambda select: pinned_columns(select, None), sql) is None
    assert apply(lambda select: pinned_columns(select, {}), sql) is None
    assert apply(lambda select: pinned_columns(select, {"seats": {"seat": "INT"}}), sql) is None
    assert apply(lambda select: pinned_columns(select, {"SEATS": {"FREE": "tinyint(1)"}}), sql) is not None


def test_each_condition_is_needed():
    """Where a rule declines, writing anyway would change the rows: the engine says so."""

    def differs(original: str, reading: str) -> bool:
        return any(_rows(connection, original) != _rows(connection, reading) for connection in map(_connection, range(60)))

    inside = "EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 AND a.free = 1)"
    moved = "(a.free = 1 AND EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1))"
    assert not differs(f"{A} {inside}", f"{A} {moved}")
    # under NOT the NULL of the moved term shows
    assert differs(f"{A} NOT {inside}", f"{A} NOT {moved}")
    # an aggregate returns a row over no rows
    counted = "EXISTS(SELECT COUNT(*) FROM seats AS b WHERE b.seat = a.seat + 1 AND a.free = 1)"
    assert differs(f"{A} {counted}", f"{A} (a.free = 1 AND EXISTS(SELECT COUNT(*) FROM seats AS b WHERE b.seat = a.seat + 1))")
    # a bare name is the subquery's own column
    own = "EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 AND free = 1)"
    assert differs(f"{A} {own}", f"{A} (free = 1 AND EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1))")
    # a pin under OR pins nothing
    assert differs(f"{A} a.free = 1 OR a.seat > a.free", f"{A} a.free = 1 OR a.seat > 1")
    # a name another source has taken is that source's column
    taken = "EXISTS(SELECT 1 FROM rooms AS a WHERE a.size = a.room)"
    assert differs(f"SELECT a.seat FROM seats AS a WHERE a.seat = 2 AND {taken}", "SELECT a.seat FROM seats AS a WHERE a.seat = 2 AND EXISTS(SELECT 1 FROM rooms AS a WHERE a.size = 2)")


def test_a_number_written_out_is_not_compared_with_a_text_as_its_column_is():
    """Why a pinned column is written as its number only against whole numbers: SQLite compares the column with a
    text as numbers, and the number written out as texts."""

    import sqlite3

    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE seats (seat INTEGER, free INTEGER, note TEXT)")
    connection.execute("INSERT INTO seats VALUES (1, 1, '05')")
    assert connection.execute("SELECT seat FROM seats WHERE free = 1 AND free < note").fetchall() == [(1,)]
    assert connection.execute("SELECT seat FROM seats WHERE free = 1 AND 1 < note").fetchall() == []
    assert apply(pinned, "SELECT seat FROM seats WHERE free = 1 AND free < note") is None


def test_the_names():
    assert NAMES == ("shared_filter_terms", "exists_outer_terms", "pinned_columns")
