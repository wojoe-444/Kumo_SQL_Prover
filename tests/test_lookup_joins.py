"""A left join that looks up one row by its key, for the filter alone, is read as scalar lookups (``lookup_joins``)."""

import random

import pytest
from sqlglot import parse_one

duckdb = pytest.importorskip("duckdb")

from kumo_sql_prover.lookup_joins import NAME, lookup_joins, read_lookup_joins

TYPES = {"seats": {"seat": "INT", "free": "INT", "price": "DECIMAL(6,2)", "note": "VARCHAR"}, "rooms": {"room": "INT", "size": "INT"}, "tags": {"tag": "VARCHAR", "seat": "INT"}}
KEYS = {"seats": [("seat",)], "rooms": [("room",)], "tags": [("tag",)]}
NOT_NULL = {"seats": frozenset({"seat"}), "rooms": frozenset({"room"}), "tags": frozenset({"tag"})}
A = "SELECT a.seat FROM seats AS a"


def lookup(column: str, value: str, table: str = "seats", key: str = "seat") -> str:
    return f"(SELECT kql0.{column} FROM {table} AS kql0 WHERE kql0.{key} = {value})"


def read(sql: str, keys=KEYS, not_null=NOT_NULL, types=TYPES) -> str | None:
    tree = parse_one(sql, read="mysql")
    text = tree.sql(dialect="mysql")
    new, wrote = read_lookup_joins(tree, keys, not_null, types)
    assert tree.sql(dialect="mysql") == text  # the tree given is not changed
    assert wrote or new is tree
    if not wrote:
        return None
    assert read_lookup_joins(new, keys, not_null, types) == (new, False)  # once is enough
    return new.sql(dialect="mysql")


TEXTS = [
    (
        f"{A} LEFT JOIN seats AS p ON p.seat = a.seat - 1 LEFT JOIN seats AS n ON n.seat - a.seat = 1 WHERE a.free = 1 AND (p.free = 1 OR n.free = 1)",
        f"{A} WHERE a.free = 1 AND ({lookup('free', 'a.seat - 1')} = 1 OR {lookup('free', 'a.seat + 1')} = 1)",
    ),
    (  # the equality any way round, in parentheses, with LEFT OUTER
        f"{A} LEFT OUTER JOIN seats AS p ON a.seat = (p.seat + 1) LEFT JOIN seats AS n ON a.seat + 1 = n.seat WHERE p.free IS TRUE OR COALESCE(n.price, 0) > 2 ORDER BY a.seat",
        f"{A} WHERE {lookup('free', 'a.seat - 1')} IS TRUE OR COALESCE({lookup('price', 'a.seat + 1')}, 0) > 2 ORDER BY a.seat",
    ),
    (  # a name of the select list in ORDER BY; DISTINCT; the table by its own name
        "SELECT DISTINCT seats.seat FROM seats LEFT JOIN seats AS p ON seats.seat - p.seat = 2 WHERE seats.free = 1 AND p.free = 0 ORDER BY seat",
        f"SELECT DISTINCT seats.seat FROM seats WHERE seats.free = 1 AND {lookup('free', 'seats.seat - 2')} = 0 ORDER BY seat",
    ),
    (  # another table by its key; two columns of the found row; a plain join beside it stays
        f"SELECT a.seat, r.size FROM seats AS a JOIN rooms AS r ON r.room = a.free LEFT JOIN rooms AS q ON q.room = a.seat + r.size WHERE q.size > r.size OR q.room IS NULL",
        f"SELECT a.seat, r.size FROM seats AS a JOIN rooms AS r ON r.room = a.free WHERE {lookup('size', 'a.seat + r.size', 'rooms', 'room')} > r.size OR {lookup('room', 'a.seat + r.size', 'rooms', 'room')} IS NULL",
    ),
    (  # one join is a lookup and the other is not
        f"{A} LEFT JOIN seats AS p ON p.seat = a.seat - 1 LEFT JOIN seats AS n ON n.free = a.seat WHERE p.free = 1 OR n.free = 1",
        f"{A} LEFT JOIN seats AS n ON n.free = a.seat WHERE {lookup('free', 'a.seat - 1')} = 1 OR n.free = 1",
    ),
    (  # inside a derived table and inside a subquery
        f"SELECT d.seat FROM ({A} LEFT JOIN seats AS p ON p.seat = a.seat - 1 WHERE p.free = 1) AS d WHERE d.seat IN ({A} LEFT JOIN seats AS n ON n.seat = a.seat + 1 WHERE n.free IS NULL)",
        f"SELECT d.seat FROM ({A} WHERE {lookup('free', 'a.seat - 1')} = 1) AS d WHERE d.seat IN ({A} WHERE {lookup('free', 'a.seat + 1')} IS NULL)",
    ),
    (  # a name kql0 already in the query
        "SELECT kql0.seat FROM seats AS kql0 LEFT JOIN seats AS p ON p.seat = kql0.seat - 1 WHERE p.free = 1",
        "SELECT kql0.seat FROM seats AS kql0 WHERE (SELECT kql1.free FROM seats AS kql1 WHERE kql1.seat = kql0.seat - 1) = 1",
    ),
]


@pytest.mark.parametrize("sql, expected", TEXTS)
def test_the_join_is_read_as_lookups(sql, expected):
    assert read(sql) == expected


def _connection(seed: int):
    rng = random.Random(seed)
    connection = duckdb.connect()
    connection.execute("CREATE TABLE seats (seat INTEGER PRIMARY KEY, free INTEGER, price DECIMAL(6,2), note VARCHAR)")
    connection.execute("CREATE TABLE rooms (room INTEGER PRIMARY KEY, size INTEGER)")
    for seat in rng.sample(range(0, 9), rng.randrange(0, 8)):
        connection.execute("INSERT INTO seats VALUES (?, ?, ?, ?)", [seat, rng.choice((0, 1, 1, 2, None)), rng.choice((0.5, 1, 2, 4.5, None)), rng.choice(("x", "y", None))])
    for room in rng.sample(range(0, 9), rng.randrange(0, 7)):
        connection.execute("INSERT INTO rooms VALUES (?, ?)", [room, rng.choice((1, 2, 2, 5, None))])
    return connection


def _rows(connection, sql: str) -> list:
    return sorted(connection.execute(parse_one(sql, read="mysql").sql(dialect="duckdb")).fetchall(), key=repr)


@pytest.mark.parametrize("sql, expected", TEXTS)
def test_the_reading_returns_the_same_rows(sql, expected):
    for seed in range(60):
        connection = _connection(seed)
        assert _rows(connection, sql) == _rows(connection, expected), seed


LEFT_ALONE = [
    # the join
    f"{A} JOIN seats AS p ON p.seat = a.seat - 1 WHERE p.free = 1",
    f"{A} RIGHT JOIN seats AS p ON p.seat = a.seat - 1 WHERE p.free = 1",
    f"{A} LEFT JOIN seats AS p ON p.seat = a.seat - 1 FULL JOIN rooms AS r ON r.room = a.seat WHERE p.free = 1",
    f"{A} LEFT JOIN seats AS p USING (seat) WHERE p.free = 1",
    f"{A} NATURAL LEFT JOIN rooms AS p WHERE p.size = 1",
    f"{A} LEFT JOIN (SELECT * FROM seats) AS p ON p.seat = a.seat - 1 WHERE p.free = 1",
    # the ON
    f"{A} LEFT JOIN seats AS p ON p.free = a.seat - 1 WHERE p.free = 1",  # not the key
    f"{A} LEFT JOIN seats AS p ON p.seat = a.seat - 1 AND p.free = 1 WHERE p.price > 1",  # more than the one equality
    f"{A} LEFT JOIN seats AS p ON p.seat < a.seat WHERE p.free = 1",
    f"{A} LEFT JOIN seats AS p ON ABS(p.seat - a.seat) = 1 WHERE p.free = 1",
    f"{A} LEFT JOIN seats AS p ON p.seat * 2 = a.seat WHERE p.free = 1",
    f"{A} LEFT JOIN seats AS p ON p.seat + p.seat = a.seat WHERE p.free = 1",  # the key twice
    f"{A} LEFT JOIN seats AS p ON p.seat - p.seat = a.seat WHERE p.free = 1",  # and not at all
    f"{A} LEFT JOIN seats AS p ON p.seat = a.price WHERE p.free = 1",  # a decimal is not a whole number
    f"{A} LEFT JOIN seats AS p ON p.seat = a.note WHERE p.free = 1",
    f"{A} LEFT JOIN seats AS p ON p.seat = a.seat - 1.5 WHERE p.free = 1",
    f"{A} LEFT JOIN seats AS p ON p.seat = 4 WHERE p.free = 1",  # no column of the row to look up for
    f"{A} LEFT JOIN seats AS p ON p.seat = 4 - a.seat WHERE p.free = 1",
    f"{A} LEFT JOIN tags AS g ON g.tag = a.note WHERE g.seat = 1",  # a text key
    f"{A} LEFT JOIN seats AS p ON p.seat = n.seat - 1 LEFT JOIN seats AS n ON n.free = a.seat WHERE p.free = 1",  # a source that comes after
    f"{A} LEFT JOIN seats AS p ON p.seat = (SELECT MAX(room) FROM rooms AS r) WHERE p.free = 1",
    # where the found row is read
    f"SELECT a.seat, p.free FROM seats AS a LEFT JOIN seats AS p ON p.seat = a.seat - 1 WHERE p.free = 1",
    f"SELECT * FROM seats AS a LEFT JOIN seats AS p ON p.seat = a.seat - 1 WHERE p.free = 1",
    f"SELECT a.seat, p.* FROM seats AS a LEFT JOIN seats AS p ON p.seat = a.seat - 1 WHERE p.free = 1",
    f"{A} LEFT JOIN seats AS p ON p.seat = a.seat - 1 WHERE p.free = 1 ORDER BY p.price",
    f"SELECT COUNT(*) FROM seats AS a LEFT JOIN seats AS p ON p.seat = a.seat - 1 WHERE a.free = 1 GROUP BY p.free",
    f"{A} LEFT JOIN seats AS p ON p.seat = a.seat - 1 LEFT JOIN seats AS q ON q.seat = p.free WHERE a.free = 1",
    f"{A} LEFT JOIN seats AS p ON p.seat = a.seat - 1 WHERE EXISTS(SELECT 1 FROM rooms AS r WHERE r.size = p.free)",
    f"{A} LEFT JOIN seats AS p ON p.seat = a.seat - 1 WHERE a.free = 1",  # never read
    f"{A} LEFT JOIN seats AS p ON p.seat = a.seat - 1",
    # names
    f"SELECT seat FROM rooms AS r LEFT JOIN seats AS p ON p.seat = r.room - 1 WHERE p.free = 1",  # a bare name
    f"{A} LEFT JOIN seats AS p ON p.seat = a.seat - 1 WHERE p.free = 1 AND free = 1",
    f"{A} LEFT JOIN seats AS p ON p.seat = a.seat - 1 WHERE p.free = 1 GROUP BY seat",
    f"WITH seats AS (SELECT 1 AS seat, 1 AS free UNION ALL SELECT 1, 0) {A} LEFT JOIN seats AS p ON p.seat = a.seat - 1 WHERE p.free = 1",
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_joins_are_left_alone(sql):
    assert read(sql) is None


def test_nothing_is_read_without_a_declared_key_that_is_never_null_and_a_whole_number():
    sql = TEXTS[0][0]
    assert read(sql, keys=None) is None
    assert read(sql, keys={}) is None
    assert read(sql, keys={"seats": [("seat", "free")]}) is None
    assert read(sql, not_null=None) is None
    assert read(sql, not_null={"seats": frozenset({"free"})}) is None
    assert read(sql, types=None) is None
    assert read(sql, types={"seats": {"seat": "VARCHAR", "free": "INT"}}) is None
    assert read(sql, types={"seats": {"seat": "DECIMAL(4,1)", "free": "INT"}}) is None
    assert read(sql, keys={"SEATS": [("SEAT",)]}, not_null={"Seats": frozenset({"Seat"})}, types={"SEATS": {"SEAT": "bigint"}}) is not None  # names without case


def test_each_condition_is_needed():
    """Where the rule declines, reading the join as lookups would change the rows or fail: the engine says so."""

    def outcome(sql: str, setup: list[str]):
        connection = duckdb.connect()
        for statement in setup:
            connection.execute(statement)
        try:
            return _rows(connection, sql)
        except duckdb.Error:
            return "error"

    seats = ["CREATE TABLE seats (seat INTEGER, free INTEGER, price DECIMAL(6,2), note VARCHAR)"]
    # not a key: two rows are found, the join repeats the row and the lookup has no one value
    twice = seats + ["INSERT INTO seats VALUES (1, 1, 1, 'x'), (2, 1, 2, 'y'), (3, 1, 3, 'z')"]
    joined = f"{A} LEFT JOIN seats AS p ON p.free = a.seat WHERE p.price > 0"
    assert outcome(joined, twice) == [(1,), (1,), (1,)]
    assert outcome(f"{A} WHERE (SELECT p.price FROM seats AS p WHERE p.free = a.seat) > 0", twice) == "error"
    # an inner join drops the row that finds nothing, a lookup keeps it for IS NULL
    rows = seats + ["INSERT INTO seats VALUES (1, 1, 1, 'x'), (2, 0, 2, 'y')"]
    assert outcome(f"{A} JOIN seats AS p ON p.seat = a.seat - 1 WHERE p.free IS NULL", rows) == []
    assert outcome(f"{A} WHERE (SELECT p.free FROM seats AS p WHERE p.seat = a.seat - 1) IS NULL", rows) == [(1,)]
    # more in the ON than the key: the lookup must keep it
    extra = f"{A} LEFT JOIN seats AS p ON p.seat = a.seat - 1 AND p.free = 1 WHERE p.price IS NULL"
    assert outcome(extra, rows) == [(1,)]
    assert outcome(f"{A} WHERE (SELECT p.price FROM seats AS p WHERE p.seat = a.seat - 1) IS NULL", rows) == [(1,)]
    assert outcome(extra.replace("p.free = 1", "p.free = 0"), rows) == [(1,), (2,)]
    # a text compared with a number: MySQL and SQLite find '1' and '01' alike (SQLite shown)
    import sqlite3

    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE tags (tag TEXT PRIMARY KEY, seat INTEGER)")
    connection.execute("CREATE TABLE seats (seat INTEGER, free INTEGER)")
    connection.execute("INSERT INTO tags VALUES ('1', 5), ('01', 6)")
    connection.execute("INSERT INTO seats VALUES (1, 1)")
    assert connection.execute("SELECT COUNT(*) FROM seats AS a LEFT JOIN tags AS g ON g.tag + 0 = a.seat").fetchall() == [(2,)]


def test_one_select_is_read_by_the_function_for_one_select():
    select = parse_one(TEXTS[0][0], read="mysql")
    assert lookup_joins(select, KEYS, NOT_NULL, TYPES).sql(dialect="mysql") == TEXTS[0][1]
    assert lookup_joins(parse_one(LEFT_ALONE[0], read="mysql"), KEYS, NOT_NULL, TYPES) is None


def test_the_name():
    assert NAME == "lookup_joins"
