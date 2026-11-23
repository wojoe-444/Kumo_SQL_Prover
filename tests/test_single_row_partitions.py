"""A window over partitions that hold one row each is read on that row (``single_row_partitions``)."""

import random

import pytest
from sqlglot import parse_one

duckdb = pytest.importorskip("duckdb")

from kumo_sql_prover.single_row_partitions import NAME, read_single_row_partitions

SCHEMA = {"staff": ["person", "unit", "flag", "pay"], "seats": ["seat", "free"], "notes": ["person", "body"]}
KEYS = {"staff": [("person", "unit")], "seats": [("seat",)], "notes": [("person",)]}
NOT_NULL = {"staff": frozenset({"person", "unit"}), "seats": frozenset({"seat"}), "notes": frozenset()}
P = "PARTITION BY person, unit"


def read(sql: str, keys=KEYS, not_null=NOT_NULL, schema=SCHEMA) -> str | None:
    tree = parse_one(sql, read="mysql")
    text = tree.sql(dialect="mysql")
    new, wrote = read_single_row_partitions(tree, schema, keys, not_null)
    assert tree.sql(dialect="mysql") == text  # the tree given is not changed
    assert wrote or new is tree
    return new.sql(dialect="mysql") if wrote else None


TEXTS = [
    (f"SELECT person, ROW_NUMBER() OVER ({P} ORDER BY flag) AS n FROM staff", "SELECT person, 1 AS n FROM staff"),
    (  # the key's columns in any order, by the table's alias, with a filter
        "SELECT person, RANK() OVER (PARTITION BY unit, person ORDER BY pay DESC) AS n, DENSE_RANK() OVER (PARTITION BY s.person, s.unit) AS m FROM staff AS s WHERE pay > 3",
        "SELECT person, 1 AS n, 1 AS m FROM staff AS s WHERE pay > 3",
    ),
    (  # more partition columns only split further
        f"SELECT person, COUNT(*) OVER ({P}) AS c, COUNT(flag) OVER ({P}) AS f, COUNT(DISTINCT pay + 1) OVER ({P}, flag) AS g FROM staff",
        "SELECT person, 1 AS c, CASE WHEN NOT flag IS NULL THEN 1 ELSE 0 END AS f, CASE WHEN NOT (pay + 1) IS NULL THEN 1 ELSE 0 END AS g FROM staff",
    ),
    (
        f"SELECT person, MIN(pay) OVER ({P}) AS a, MAX(pay + 1) OVER ({P}) AS b, FIRST_VALUE(flag) OVER ({P} ORDER BY pay) AS c, "
        f"LAST_VALUE(flag) OVER ({P} ORDER BY pay) AS d, SUM(1) OVER ({P}) AS e FROM staff",
        "SELECT person, pay AS a, (pay + 1) AS b, flag AS c, flag AS d, 1 AS e FROM staff",
    ),
    (
        f"SELECT d.person FROM (SELECT person, unit, ROW_NUMBER() OVER ({P} ORDER BY flag) AS n FROM staff) AS d WHERE d.n = 1",
        "SELECT d.person FROM (SELECT person, unit, 1 AS n FROM staff) AS d WHERE d.n = 1",
    ),
    (
        f"SELECT person, 2 * COUNT(1) OVER ({P}) - MIN(pay - 1) OVER (PARTITION BY unit, person) AS v FROM staff",
        "SELECT person, 2 * 1 - (pay - 1) AS v FROM staff",
    ),
    ("SELECT seat, COUNT(free) OVER (PARTITION BY seat) AS c FROM seats", "SELECT seat, CASE WHEN NOT free IS NULL THEN 1 ELSE 0 END AS c FROM seats"),
    (  # one window is read, the other is not
        f"SELECT person, ROW_NUMBER() OVER ({P}) AS n, RANK() OVER (PARTITION BY unit ORDER BY pay) AS m FROM staff",
        "SELECT person, 1 AS n, RANK() OVER (PARTITION BY unit ORDER BY pay) AS m FROM staff",
    ),
]


@pytest.mark.parametrize("sql, expected", TEXTS)
def test_the_window_is_read_on_its_row(sql, expected):
    assert read(sql) == expected


def _connection(seed: int):
    rng = random.Random(seed)
    connection = duckdb.connect()
    connection.execute("CREATE TABLE staff (person INTEGER, unit INTEGER, flag VARCHAR, pay INTEGER)")
    connection.execute("CREATE TABLE seats (seat INTEGER, free INTEGER)")
    keys = rng.sample([(p, u) for p in range(1, 5) for u in range(1, 4)], rng.randrange(0, 9))
    for person, unit in keys:
        connection.execute("INSERT INTO staff VALUES (?, ?, ?, ?)", [person, unit, rng.choice(("Y", "N", None)), rng.choice((2, 4, 4, 7, None))])
    for seat in rng.sample(range(1, 9), rng.randrange(0, 6)):
        connection.execute("INSERT INTO seats VALUES (?, ?)", [seat, rng.choice((0, 1, None))])
    return connection


def _rows(connection, sql: str) -> list:
    return sorted(connection.execute(parse_one(sql, read="mysql").sql(dialect="duckdb")).fetchall(), key=repr)


@pytest.mark.parametrize("position", range(len(TEXTS)))
def test_the_reading_returns_the_same_rows(position):
    sql, expected = TEXTS[position]
    for seed in range(40):
        connection = _connection(seed)
        assert _rows(connection, sql) == _rows(connection, expected), seed


LEFT_ALONE = [
    # the partitions may hold more than one row
    "SELECT person, ROW_NUMBER() OVER (PARTITION BY person ORDER BY flag) AS n FROM staff",
    "SELECT person, ROW_NUMBER() OVER (PARTITION BY unit) AS n FROM staff",
    "SELECT person, ROW_NUMBER() OVER (ORDER BY person, unit) AS n FROM staff",
    "SELECT person, ROW_NUMBER() OVER (PARTITION BY person + 0, unit) AS n FROM staff",
    "SELECT person, COUNT(*) OVER (PARTITION BY person) AS n FROM notes",  # a key that may be NULL
    "SELECT person, COUNT(*) OVER (PARTITION BY free) AS n FROM seats",
    # the select
    f"SELECT s.person, ROW_NUMBER() OVER (PARTITION BY s.person, s.unit) AS n FROM staff AS s JOIN notes AS t ON t.person = s.person",
    f"SELECT person, unit, ROW_NUMBER() OVER ({P}) AS n FROM staff GROUP BY person, unit",
    f"SELECT person, ROW_NUMBER() OVER ({P}) AS n FROM (SELECT * FROM staff) AS s",
    f"SELECT person, ROW_NUMBER() OVER ({P}) AS n FROM staff, seats",
    f"SELECT MAX(pay), COUNT(*) OVER ({P}) AS n FROM staff",
    # the window
    f"SELECT person, SUM(pay) OVER ({P}) AS n FROM staff",
    f"SELECT person, SUM('3') OVER ({P}) AS n FROM staff",
    f"SELECT person, AVG(pay) OVER ({P}) AS n FROM staff",
    f"SELECT person, LAG(pay) OVER ({P} ORDER BY pay) AS n FROM staff",
    f"SELECT person, NTILE(2) OVER ({P} ORDER BY pay) AS n FROM staff",
    f"SELECT person, COUNT(NULL) OVER ({P}) AS n FROM staff",
    f"SELECT person, MIN(DISTINCT pay) OVER ({P}) AS n FROM staff",
    f"SELECT person, COUNT(*) OVER ({P} ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS n FROM staff",
    f"SELECT person, COUNT(*) OVER w AS n FROM staff WINDOW w AS ({P})",
    f"SELECT person, MAX((SELECT MAX(free) FROM seats)) OVER ({P}) AS n FROM staff",
    # a WITH clause could hide the declared table
    f"WITH staff AS (SELECT 1 AS person, 1 AS unit UNION ALL SELECT 1, 1) SELECT person, ROW_NUMBER() OVER ({P}) AS n FROM staff",
    "SELECT person FROM staff WHERE pay > 3",
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_windows_are_left_alone(sql):
    assert read(sql) is None


def test_nothing_is_read_without_a_declared_key_whose_columns_are_never_null():
    sql = TEXTS[0][0]
    assert read(sql, keys=None) is None
    assert read(sql, keys={}) is None
    assert read(sql, keys={"staff": []}) is None
    assert read(sql, keys={"staff": [()]}) is None
    assert read(sql, not_null=None) is None
    assert read(sql, not_null={"staff": frozenset({"person"})}) is None
    assert read(sql, schema=None) is None
    assert read(sql, keys={"STAFF": [("PERSON", "UNIT")]}, not_null={"Staff": frozenset({"Person", "Unit"})}) is not None  # names without case


def test_each_condition_is_needed():
    """Where the rule declines, reading the window on its row would change the rows: the engine says so."""

    def differs(original: str, reading: str, setup: list[str]) -> bool:
        connection = duckdb.connect()
        for statement in setup:
            connection.execute(statement)
        return _rows(connection, original) != _rows(connection, reading)

    staff = ["CREATE TABLE staff (person INTEGER, unit INTEGER, flag VARCHAR, pay INTEGER)", "INSERT INTO staff VALUES (1, 1, 'Y', 2), (1, 2, 'N', 4)"]
    # only part of the key: two rows share a partition
    assert differs("SELECT person, ROW_NUMBER() OVER (PARTITION BY person ORDER BY unit) AS n FROM staff", "SELECT person, 1 AS n FROM staff", staff)
    assert differs("SELECT person, COUNT(*) OVER (PARTITION BY person) AS n FROM staff", "SELECT person, 1 AS n FROM staff", staff)
    # a unique column that may be NULL repeats NULL
    notes = ["CREATE TABLE notes (person INTEGER, body VARCHAR)", "INSERT INTO notes VALUES (NULL, 'a'), (NULL, 'b'), (3, 'c')"]
    assert differs("SELECT body, COUNT(*) OVER (PARTITION BY person) AS n FROM notes", "SELECT body, 1 AS n FROM notes", notes)
    # a frame need not hold the current row
    framed = "SELECT person, COUNT(*) OVER (PARTITION BY person, unit ORDER BY pay ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS n FROM staff"
    assert differs(framed, "SELECT person, 1 AS n FROM staff", staff)
    # a join can repeat the row
    joined = "SELECT s.person, COUNT(*) OVER (PARTITION BY s.person, s.unit) AS n FROM staff AS s JOIN staff AS t ON t.person = s.person"
    assert differs(joined, "SELECT s.person, 1 AS n FROM staff AS s JOIN staff AS t ON t.person = s.person", staff)
    # COUNT of a column counts no NULL
    nulls = ["CREATE TABLE staff (person INTEGER, unit INTEGER, flag VARCHAR, pay INTEGER)", "INSERT INTO staff VALUES (1, 1, NULL, NULL)"]
    assert differs("SELECT person, COUNT(flag) OVER (PARTITION BY person, unit) AS n FROM staff", "SELECT person, 1 AS n FROM staff", nulls)


def test_the_name():
    assert NAME == "single_row_partitions"
