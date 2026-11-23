"""``LAG`` and ``LEAD`` over a column of consecutive whole numbers, read as a keyed lookup (``neighbour_rows``)."""

import itertools
import random

import pytest
from sqlglot import parse_one

duckdb = pytest.importorskip("duckdb")

from kumo_sql_prover.neighbour_rows import NAME, read_neighbour_rows

SCHEMA = {"seats": ["seat", "free", "price"], "rooms": ["room", "size"]}
CONSECUTIVE = {"seats": frozenset({"seat"})}
D = "(SELECT seat, free, LAG(free) OVER (ORDER BY seat) AS before, LEAD(free) OVER (ORDER BY seat) AS after FROM seats)"


def read(sql: str, consecutive=CONSECUTIVE, schema=SCHEMA) -> str | None:
    tree = parse_one(sql, read="mysql")
    text = tree.sql(dialect="mysql")
    new, wrote = read_neighbour_rows(tree, schema, consecutive)
    assert tree.sql(dialect="mysql") == text  # the tree given is not changed
    assert wrote or new is tree
    return new.sql(dialect="mysql") if wrote else None


BEFORE = "(SELECT kqn0.free FROM seats AS kqn0 WHERE kqn0.seat = d.seat - 1)"
AFTER = "(SELECT kqn0.free FROM seats AS kqn0 WHERE kqn0.seat = d.seat + 1)"

TEXTS = [
    (
        f"SELECT d.seat FROM {D} AS d WHERE d.free = 1 AND (d.before = 1 OR d.after = 1)",
        f"SELECT d.seat FROM seats AS d WHERE d.free = 1 AND ({BEFORE} = 1 OR {AFTER} = 1)",
    ),
    (  # bare names are the derived table's columns
        f"SELECT seat FROM {D} AS d WHERE free = 1 AND (before = 1 OR after = 1) ORDER BY seat",
        f"SELECT d.seat FROM seats AS d WHERE d.free = 1 AND ({BEFORE} = 1 OR {AFTER} = 1) ORDER BY d.seat",
    ),
    (  # a star, an offset, and DESC swaps before and after
        "SELECT d.seat, d.two FROM (SELECT *, LAG(price, 2) OVER (ORDER BY seat DESC) AS two FROM seats) AS d",
        "SELECT d.seat, (SELECT kqn0.price FROM seats AS kqn0 WHERE kqn0.seat = d.seat + 2) AS two FROM seats AS d",
    ),
    (  # a default written NULL, the table's own alias, a neighbour inside an expression
        "SELECT d.seat FROM (SELECT seat, LEAD(free, 1, NULL) OVER (ORDER BY s.seat ASC) = 1 AS next_free FROM seats AS s) AS d WHERE d.next_free",
        f"SELECT d.seat FROM seats AS d WHERE ({AFTER} = 1)",
    ),
    (
        "SELECT d.seat, d.gain FROM (SELECT seat, price - LAG(price) OVER (ORDER BY seat) AS gain FROM seats) AS d WHERE d.gain > 0",
        "SELECT d.seat, d.price - (SELECT kqn0.price FROM seats AS kqn0 WHERE kqn0.seat = d.seat - 1) AS gain FROM seats AS d "
        "WHERE (d.price - (SELECT kqn0.price FROM seats AS kqn0 WHERE kqn0.seat = d.seat - 1)) > 0",
    ),
    (  # ORDER BY without LIMIT changes no rows; DISTINCT removes none beside the consecutive column
        "SELECT DISTINCT d.free FROM (SELECT DISTINCT seat, free, LEAD(free) OVER (ORDER BY seat) AS after FROM seats ORDER BY seat) AS d WHERE d.after = d.free",
        f"SELECT DISTINCT d.free FROM seats AS d WHERE {AFTER} = d.free",
    ),
    (
        "SELECT COUNT(*) FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d WHERE d.before = 1",
        f"SELECT COUNT(*) FROM seats AS d WHERE {BEFORE} = 1",
    ),
    (
        "SELECT d.seat AS s, d.before + 1 AS b FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d ORDER BY s",
        f"SELECT d.seat AS s, {BEFORE} + 1 AS b FROM seats AS d ORDER BY s",
    ),
    (  # inside a subquery, with a column of the enclosing query left as it is
        "SELECT x.seat FROM seats AS x WHERE x.seat IN (SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d WHERE d.before = x.free)",
        f"SELECT x.seat FROM seats AS x WHERE x.seat IN (SELECT d.seat FROM seats AS d WHERE {BEFORE} = x.free)",
    ),
    (  # a column of the derived table under another name
        "SELECT d.taken FROM (SELECT free AS taken, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d WHERE d.before = 1",
        f"SELECT d.free AS taken FROM seats AS d WHERE {BEFORE} = 1",
    ),
]


@pytest.mark.parametrize("sql, expected", TEXTS)
def test_the_neighbours_are_written_as_lookups(sql, expected):
    assert read(sql) == expected


def test_every_lookup_of_one_select_has_the_same_name_and_a_taken_name_is_skipped():
    text = read(f"SELECT d.seat, kqn0.room FROM {D} AS d, rooms AS kqn0 WHERE d.before = 1")
    assert text is None  # a second source: left alone
    text = read(f"SELECT kqn0.seat FROM {D} AS kqn0 WHERE kqn0.before = 1 OR kqn0.after = 1")
    assert text == (
        "SELECT kqn0.seat FROM seats AS kqn0 WHERE (SELECT kqn1.free FROM seats AS kqn1 WHERE kqn1.seat = kqn0.seat - 1) = 1 "
        "OR (SELECT kqn1.free FROM seats AS kqn1 WHERE kqn1.seat = kqn0.seat + 1) = 1"
    )


def _tables():
    """Small tables whose seats are consecutive from some first number, with every mix of 0, 1 and NULL."""

    rng = random.Random(18)
    for rows in range(0, 6):
        for _ in range(6):
            first = rng.choice((1, 4, -2, 0))
            yield [(first + n, rng.choice((0, 1, 1, None)), rng.choice((3, 5, 5, 8, None))) for n in range(rows)]


def _rows(connection, sql: str) -> list:
    return sorted(connection.execute(parse_one(sql, read="mysql").sql(dialect="duckdb")).fetchall(), key=repr)


def _connection(rows: list):
    connection = duckdb.connect()
    connection.execute("CREATE TABLE seats (seat INTEGER, free INTEGER, price INTEGER)")
    if rows:
        connection.executemany("INSERT INTO seats VALUES (?, ?, ?)", rows)
    return connection


@pytest.mark.parametrize("position", range(len(TEXTS)))
def test_the_reading_returns_the_same_rows(position):
    sql, expected = TEXTS[position]
    for rows in _tables():
        connection = _connection(rows)
        assert _rows(connection, sql) == _rows(connection, expected), rows


LEFT_ALONE = [
    # the window
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (PARTITION BY price ORDER BY seat) AS before FROM seats) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER () AS before FROM seats) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY price) AS before FROM seats) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat, price) AS before FROM seats) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat + 1) AS before FROM seats) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS before FROM seats) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER w AS before FROM seats WINDOW w AS (ORDER BY seat)) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, LAG(free, 1, 0) OVER (ORDER BY seat) AS before FROM seats) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, LAG(free, 0) OVER (ORDER BY seat) AS before FROM seats) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, LAG(free, price) OVER (ORDER BY seat) AS before FROM seats) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, LAG(free, 1.5) OVER (ORDER BY seat) AS before FROM seats) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before, SUM(price) OVER (ORDER BY seat) AS total FROM seats) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, ROW_NUMBER() OVER (ORDER BY seat) AS n FROM seats) AS d WHERE d.n = 1",
    "SELECT d.seat FROM (SELECT seat, LAG((SELECT MAX(size) FROM rooms)) OVER (ORDER BY seat) AS before FROM seats) AS d WHERE d.before = 1",
    # the derived table
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats WHERE free = 1) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats LIMIT 3) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats ORDER BY seat LIMIT 3) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats GROUP BY seat, free) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats, rooms) AS d WHERE d.before = 1",
    "SELECT d.free FROM (SELECT DISTINCT free, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM (SELECT * FROM seats) AS s) AS d WHERE d.before = 1",
    "SELECT d.seat FROM (SELECT seat, seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d WHERE d.before = 1",
    "SELECT d.room FROM (SELECT room, LAG(size) OVER (ORDER BY room) AS before FROM rooms) AS d WHERE d.before = 1",  # not declared consecutive
    "SELECT d.seat FROM (SELECT seat, free FROM seats) AS d WHERE d.free = 1",  # no neighbour to read
    # the select that reads it
    "SELECT d.seat, r.room FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d JOIN rooms AS r ON r.room = d.seat",
    "SELECT d.before, COUNT(*) FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d GROUP BY d.before",
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d ORDER BY d.seat LIMIT 2",
    "SELECT * FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d",
    "SELECT d.* FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d",
    "SELECT d.seat, SUM(d.before) OVER () FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d",
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d WHERE d.before = (SELECT MAX(size) FROM rooms)",
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d WHERE d.price = 1",  # not a column of the derived table
    "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d (a, b) WHERE d.b = 1",
    # a WITH clause could hide the declared table
    "WITH seats AS (SELECT 1 AS seat, 1 AS free) SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d WHERE d.before = 1",
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_selects_are_left_alone(sql):
    assert read(sql) is None


def test_a_bare_name_that_is_not_an_output_is_left_alone():
    # price here is a column of the enclosing query: the table's own price must not capture it
    sql = "SELECT x.room FROM rooms AS x WHERE EXISTS (SELECT 1 FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d WHERE d.before = size)"
    assert read(sql) is None


def test_nothing_is_read_without_the_declaration():
    sql = TEXTS[0][0]
    assert read(sql, consecutive=None) is None
    assert read(sql, consecutive={}) is None
    assert read(sql, consecutive={"seats": frozenset()}) is None
    assert read(sql, consecutive={"seats": frozenset({"price"})}) is None
    assert read(sql, schema=None) is None
    assert read(sql, consecutive={"SEATS": frozenset({"SEAT"})}) is not None  # names are compared without case


def test_each_condition_is_needed():
    """Dropping a condition gives a reading that returns other rows: the engine says so."""

    def differs(original: str, reading: str, tables) -> bool:
        for rows in tables:
            connection = _connection(rows)
            if _rows(connection, original) != _rows(connection, reading):
                return True
        return False

    lookup = f"SELECT d.seat FROM seats AS d WHERE {BEFORE} = 1"
    # consecutive: with a gap the row before is not the row with the number one less
    original = "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d WHERE d.before = 1"
    assert not differs(original, lookup, _tables())
    assert differs(original, lookup, [[(1, 1, 3), (3, 0, 3)]])
    # a filter inside: the row before is the one before among the rows kept
    filtered = "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats WHERE price = 5) AS d WHERE d.before = 1"
    assert differs(filtered, f"SELECT d.seat FROM seats AS d WHERE d.price = 5 AND {BEFORE} = 1", [[(1, 1, 5), (2, 0, 3), (3, 0, 5)]])
    # PARTITION BY: the row before is the one before in the partition
    parted = "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (PARTITION BY price ORDER BY seat) AS before FROM seats) AS d WHERE d.before = 1"
    assert differs(parted, lookup, [[(1, 1, 5), (2, 0, 3), (3, 0, 5)]])
    # a default value: no row before is the default, a row before with a NULL is NULL
    defaulted = "SELECT d.seat FROM (SELECT seat, LAG(free, 1, 1) OVER (ORDER BY seat) AS before FROM seats) AS d WHERE d.before = 1"
    assert differs(defaulted, lookup, [[(1, 0, 5), (2, 0, 3)]])
    # DISTINCT without the consecutive column removes rows
    distinct = "SELECT d.free FROM (SELECT DISTINCT free, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d"
    assert differs(distinct, "SELECT d.free FROM seats AS d", [[(1, 1, 5), (2, 1, 3), (3, 1, 5)]])
    # ORDER BY another column is another order
    other = "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY price) AS before FROM seats) AS d WHERE d.before = 1"
    assert differs(other, lookup, [[(1, 1, 3), (2, 0, 8), (3, 0, 5)]])


def test_the_rule_never_raises_on_odd_shapes():
    shapes = [
        "SELECT 1",
        "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d UNION ALL SELECT seat FROM seats",
        "SELECT (SELECT MAX(d.before) FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d)",
        "SELECT LAG(free) OVER (ORDER BY seat) FROM seats",
        "SELECT d.seat FROM (SELECT seat, LAG(free) OVER (ORDER BY seat) AS before FROM seats) AS d WHERE d.before IN (SELECT e.before FROM (SELECT seat, LEAD(free) OVER (ORDER BY seat) AS before FROM seats) AS e)",
    ]
    for sql, repeat in itertools.product(shapes, range(2)):
        read(sql)
    assert NAME == "neighbour_rows"
