"""A test on a value looked up by a key is read as ``EXISTS`` with the test inside (``lookup_conditions``)."""

import random
import sqlite3

import pytest
from sqlglot import parse_one

duckdb = pytest.importorskip("duckdb")

from kumo_sql_prover.lookup_conditions import NAME, lookup_conditions

KEYS = {"seats": [("seat",)], "rooms": [("room",)]}
NOT_NULL = {"seats": frozenset({"seat"}), "rooms": frozenset({"room"})}
A = "SELECT a.seat FROM seats AS a WHERE"
L = "(SELECT b.free FROM seats AS b WHERE b.seat = a.seat - 1)"
E = "EXISTS(SELECT 1 FROM seats AS b WHERE b.seat = a.seat - 1 AND"


def read(sql: str, dialect: str = "mysql", keys=KEYS, not_null=NOT_NULL) -> str | None:
    tree = parse_one(sql, read="mysql")
    text = tree.sql(dialect="mysql")
    new = lookup_conditions(tree, keys, not_null, dialect)
    assert tree.sql(dialect="mysql") == text  # the select given is not changed
    if new is None:
        return None
    assert lookup_conditions(new, keys, not_null, dialect) is None  # once is enough
    return new.sql(dialect="mysql")


# the test, and the test inside the EXISTS
TESTS = [
    (f"{L} IS TRUE", "b.free IS TRUE"),
    (f"{L} IS FALSE", "b.free IS FALSE"),
    (f"{L} IS NOT NULL", "NOT b.free IS NULL"),
    (f"{L} <=> 1", "b.free <=> 1"),
    (f"COALESCE({L}, 0) = 1", "COALESCE(b.free, 0) = 1"),
    (f"IFNULL({L}, 0) > 0", "COALESCE(b.free, 0) > 0"),
    (f"COALESCE({L}, FALSE)", "COALESCE(b.free, FALSE)"),
    (f"COALESCE({L}, NULL, 0) <> 0", "COALESCE(b.free, NULL, 0) <> 0"),
    (f"NOT {L} = 0", "NOT b.free = 0"),
    (f"{L} IN (1, 2)", "b.free IN (1, 2)"),
    (f"{L} + 1 = 2", "b.free + 1 = 2"),
    (f"{L} BETWEEN 1 AND 2", "b.free BETWEEN 1 AND 2"),
    (f"({L} = 1) IS TRUE", "(b.free = 1) IS TRUE"),
]
NUMBER_ALONE = [(L, "b.free")]


@pytest.mark.parametrize("test, inside", TESTS + NUMBER_ALONE)
def test_the_test_moves_inside_an_exists(test, inside):
    assert read(f"{A} {test}") == f"{A} {E} {inside})"
    # under AND and OR, beside other terms
    assert read(f"{A} a.free = 1 AND (a.price > 2 OR {test})") == f"{A} a.free = 1 AND (a.price > 2 OR {E} {inside}))"


@pytest.mark.parametrize("test, inside", TESTS)
@pytest.mark.parametrize("dialect", ("postgres", "bigquery", None))
def test_other_dialects_read_the_same_tests(dialect, test, inside):
    assert read(f"{A} {test}", dialect) == f"{A} {E} {inside})"


@pytest.mark.parametrize("dialect", ("postgres", "bigquery", "snowflake", None))
def test_a_number_alone_is_a_test_only_where_truth_is_a_number(dialect):
    for test, _ in NUMBER_ALONE:
        assert read(f"{A} {test}", dialect) is None
    # and a fallback number does not make a missing row FALSE there
    assert read(f"{A} COALESCE({L}, 0)", dialect) is None
    assert read(f"{A} COALESCE({L}, 0)", "sqlite") == f"{A} {E} COALESCE(b.free, 0))"


def test_other_shapes_of_the_lookup():
    assert read(f"{A} (SELECT b.free FROM seats AS b WHERE a.seat + 1 = b.seat) IS TRUE") == f"{A} EXISTS(SELECT 1 FROM seats AS b WHERE a.seat + 1 = b.seat AND b.free IS TRUE)"
    assert read(f"{A} (SELECT seats.note FROM seats WHERE (seats.seat = a.seat)) IS NOT NULL") == f"{A} EXISTS(SELECT 1 FROM seats WHERE (seats.seat = a.seat) AND NOT seats.note IS NULL)"
    two = read(f"{A} {L} IS TRUE OR (SELECT r.size FROM rooms AS r WHERE r.room = a.seat) <=> 2")
    assert two == f"{A} {E} b.free IS TRUE) OR EXISTS(SELECT 1 FROM rooms AS r WHERE r.room = a.seat AND r.size <=> 2)"


def _connection(seed: int):
    rng = random.Random(seed)
    connection = duckdb.connect()
    connection.execute("CREATE TABLE seats (seat INTEGER PRIMARY KEY, free INTEGER, price DECIMAL(6,2), note VARCHAR)")
    for seat in rng.sample(range(0, 9), rng.randrange(0, 8)):
        connection.execute("INSERT INTO seats VALUES (?, ?, ?, ?)", [seat, rng.choice((0, 1, 1, 2, None)), rng.choice((0.5, 1, 2, 4.5, None)), rng.choice(("x", "y", None))])
    return connection


def _rows(connection, sql: str) -> list:
    return sorted(connection.execute(parse_one(sql, read="mysql").sql(dialect="duckdb")).fetchall(), key=repr)


@pytest.mark.parametrize("test, inside", TESTS)
def test_the_reading_returns_the_same_rows(test, inside):
    for form in ("{}", "a.free = 1 AND (a.price > 2 OR {})"):
        sql = f"{A} " + form.format(test)
        for seed in range(50):
            connection = _connection(seed)
            assert _rows(connection, sql) == _rows(connection, read(sql)), seed


def test_a_number_alone_returns_the_same_rows_in_sqlite():
    """DuckDB wants a Boolean there; SQLite reads a number as MySQL does."""

    rng = random.Random(7)
    for _ in range(60):
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE seats (seat INTEGER PRIMARY KEY, free INTEGER, price REAL)")
        for seat in rng.sample(range(0, 9), rng.randrange(0, 8)):
            connection.execute("INSERT INTO seats VALUES (?, ?, ?)", [seat, rng.choice((0, 1, 1, 2, None)), rng.choice((0.5, 2, 4.5, None))])
        for test in (L, f"COALESCE({L}, 0)", f"NOT {L}"):
            sql = f"{A} a.price > 2 OR {test}"
            new = read(sql, "sqlite")
            assert new is not None
            assert sorted(connection.execute(sql).fetchall()) == sorted(connection.execute(new).fetchall())


LEFT_ALONE = [
    # TRUE for a row that is not there, or not worked out
    f"{A} {L} IS NULL",
    f"{A} NOT ({L} IS TRUE)",
    f"{A} {L} IS NOT TRUE",
    f"{A} COALESCE({L}, 1) = 1",
    f"{A} COALESCE({L}, TRUE)",
    f"{A} COALESCE({L}, a.free) = 1",
    f"{A} NOT ({L} = 0 AND a.price > 2)",
    f"{A} NOT ({L} = 0 OR a.price > 2)",  # never TRUE for a missing row, but the test is read up to its own AND or OR only
    f"{A} {L} <=> NULL",
    f"{A} {L} <=> a.free",
    f"{A} CASE WHEN {L} = 1 THEN 1 ELSE 0 END = 1",
    f"{A} IF({L}, 1, 0) = 1",
    f"{A} COALESCE({L}, 0.5) > 0",
    f"{A} COALESCE({L}, 'n') = 'y'",
    # read already
    f"{A} {L} = 1",
    f"{A} 0 < {L}",
    f"{A} {L} <> a.free",
    # where the test stands
    f"SELECT a.seat, {L} IS TRUE FROM seats AS a",
    f"SELECT a.seat FROM seats AS a GROUP BY a.seat HAVING MAX(a.free) = 1 AND (SELECT b.free FROM seats AS b WHERE b.seat = MAX(a.seat)) IS TRUE",
    f"{A} a.seat IN (SELECT b.seat FROM seats AS b WHERE b.seat = a.seat - 1)",
    f"{A} EXISTS(SELECT b.free FROM seats AS b WHERE b.seat = a.seat - 1)",
    # the test
    f"{A} COALESCE({L}, 0) = (SELECT MAX(r.room) FROM rooms AS r)",
    f"{A} COALESCE({L}, 0) = free",  # a bare name would be the subquery's own
    f"SELECT b.seat FROM seats AS b WHERE COALESCE((SELECT b.free FROM seats AS b WHERE b.seat = 3), 0) = b.free",  # the lookup's own name outside
    f"{A} COALESCE({L}, 0) > RAND()",
    # the lookup
    f"{A} (SELECT b.free FROM seats AS b WHERE b.free = a.seat - 1) IS TRUE",  # not the key
    f"{A} (SELECT b.free FROM seats AS b WHERE b.seat = a.seat - 1 AND b.free = 1) IS TRUE",
    f"{A} (SELECT b.free FROM seats AS b WHERE b.seat = b.free - 1) IS TRUE",
    f"{A} (SELECT b.free FROM seats AS b WHERE b.seat = seat - 1) IS TRUE",
    f"{A} (SELECT b.free FROM seats AS b WHERE b.seat < a.seat) IS TRUE",
    f"{A} (SELECT b.free + 1 FROM seats AS b WHERE b.seat = a.seat - 1) IS TRUE",
    f"{A} (SELECT free FROM seats AS b WHERE b.seat = a.seat - 1) IS TRUE",
    f"{A} (SELECT MAX(b.free) FROM seats AS b WHERE b.seat = a.seat - 1) IS TRUE",
    f"{A} (SELECT b.free FROM seats AS b WHERE b.seat = a.seat - 1 LIMIT 1) IS TRUE",
    f"{A} (SELECT DISTINCT b.free FROM seats AS b WHERE b.seat = a.seat - 1) IS TRUE",
    f"{A} (SELECT b.free FROM seats AS b JOIN rooms AS r ON r.room = b.seat WHERE b.seat = a.seat - 1) IS TRUE",
    f"{A} (SELECT b.free FROM (SELECT * FROM seats) AS b WHERE b.seat = a.seat - 1) IS TRUE",
    f"{A} (SELECT b.free FROM seats AS b WHERE b.seat = (SELECT MAX(r.room) FROM rooms AS r)) IS TRUE",
    f"{A} (SELECT b.free FROM seats AS b) IS TRUE",
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_tests_are_left_alone(sql):
    assert read(sql) is None


def test_nothing_is_read_without_a_declared_key_that_is_never_null():
    sql = f"{A} {L} IS TRUE"
    assert read(sql, keys=None) is None
    assert read(sql, keys={}) is None
    assert read(sql, keys={"seats": [("seat", "free")]}) is None
    assert read(sql, not_null=None) is None
    assert read(sql, not_null={"seats": frozenset({"free"})}) is None
    assert read(sql, keys={"SEATS": [("SEAT",)]}, not_null={"Seats": frozenset({"Seat"})}) is not None  # names without case


def test_each_condition_is_needed():
    """Where the rule declines, writing the EXISTS would change the rows: the engine says so."""

    def differs(test: str, inside: str, prefix: str = "") -> bool:
        original, reading = f"{A} {prefix}{test}", f"{A} {prefix}{E} {inside})"
        return any(_rows(connection, original) != _rows(connection, reading) for connection in map(_connection, range(50)))

    assert not differs(f"{L} IS TRUE", "b.free IS TRUE")
    # a test that is TRUE when no row is found
    assert differs(f"{L} IS NULL", "b.free IS NULL")
    assert differs(f"COALESCE({L}, 1) = 1", "COALESCE(b.free, 1) = 1")
    assert differs(f"NOT ({L} IS TRUE)", "NOT (b.free IS TRUE)")
    # under NOT, the NULL of a missing row is not the FALSE of the EXISTS
    assert differs(f"({L} + 1 = 2)", "b.free + 1 = 2", prefix="NOT ")
    # not the key: more than one row is found and the lookup has no one value
    connection = duckdb.connect()
    connection.execute("CREATE TABLE seats (seat INTEGER, free INTEGER)")
    connection.execute("INSERT INTO seats VALUES (1, 1), (2, 1), (3, 0)")
    with pytest.raises(duckdb.Error):
        connection.execute("SELECT a.seat FROM seats AS a WHERE (SELECT b.seat FROM seats AS b WHERE b.free = a.free) IS NOT NULL").fetchall()


def test_the_name():
    assert NAME == "lookup_conditions"
