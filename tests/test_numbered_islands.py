"""Runs of consecutive key values found with a row number, read as "a neighbour exists" (``numbered_islands``).

Each reading is run as written and as read on DuckDB and on SQLite over made-up tables of seats drawn at random
(gaps, runs of every length, flags of three values and NULL), and the rows must be the same. The queries the rule
leaves alone come, where a reading by hand would change rows, with a table that shows it.
"""

from __future__ import annotations

import random
import sqlite3

import duckdb
import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.numbered_islands import read_numbered_islands
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"seats": ["seat", "free", "zone", "price"], "loose": ["seat", "free"], "parts": ["seat", "free"], "pairs": ["hall", "seat", "free"]}
TYPES = {
    "seats": {"seat": "INT", "free": "INT", "zone": "INT", "price": "DECIMAL(8,2)"},
    "loose": {"seat": "INT", "free": "INT"},  # no key
    "parts": {"seat": "DECIMAL(8,2)", "free": "INT"},  # a key that is not a whole number
    "pairs": {"hall": "INT", "seat": "INT", "free": "INT"},  # a key of two columns
}
KEYS = {"seats": [("seat",)], "loose": [], "parts": [("seat",)], "pairs": [("hall", "seat")]}
NOT_NULL = {"seats": frozenset({"seat"}), "loose": frozenset({"seat"}), "parts": frozenset({"seat"}), "pairs": frozenset({"hall", "seat"})}
CONSTRAINTS = {name: TableConstraints(not_null=NOT_NULL[name], keys=tuple(KEYS[name])) for name in SCHEMA}

NUMBERED = "SELECT seat, ROW_NUMBER() OVER (ORDER BY seat) AS rn FROM seats WHERE free = 1"
RUNS = "SELECT seat, seat - ROW_NUMBER() OVER (ORDER BY seat) AS run FROM seats WHERE free = 1"

READ = [
    # the window form, with the difference worked out at either level
    f"SELECT b.seat FROM (SELECT a.seat, COUNT(*) OVER (PARTITION BY a.seat - a.rn) AS n FROM ({NUMBERED}) AS a) AS b WHERE b.n >= 2",
    f"SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY run) AS n FROM ({RUNS}) AS a) AS b WHERE n > 1 ORDER BY seat",
    f"SELECT seat FROM (SELECT *, COUNT(1) OVER (PARTITION BY run) AS n FROM ({RUNS}) AS a) AS b WHERE 1 < n",
    f"SELECT seat FROM (SELECT seat, COUNT(seat) OVER (PARTITION BY run) AS n FROM ({RUNS}) AS a) AS b WHERE n <> 1",
    # other columns of the row, another filter in the select that tests the count, an aggregate over the result
    "SELECT b.seat, b.zone FROM (SELECT a.seat, a.zone, COUNT(*) OVER (PARTITION BY a.d) AS n FROM (SELECT s.*, (s.seat - ROW_NUMBER() OVER (ORDER BY s.seat ASC)) AS d FROM seats AS s WHERE s.free <> 0 AND s.zone IN (1, 2)) AS a) AS b WHERE b.n >= 2 AND b.zone = 1",
    f"SELECT COUNT(*) AS c, MIN(seat) AS low FROM (SELECT seat, COUNT(*) OVER (PARTITION BY run) AS n FROM ({RUNS}) AS a) AS b WHERE n >= 2",
    # RANK and DENSE_RANK number rows that never tie as ROW_NUMBER does; a column the filter fixes orders nothing
    "SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY d) AS n FROM (SELECT seat, free, seat - RANK() OVER (ORDER BY free, seat) AS d FROM seats WHERE free = 1) AS a) AS b WHERE n >= 2",
    "SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY d) AS n FROM (SELECT seat, seat - DENSE_RANK() OVER (ORDER BY seat, zone DESC) AS d FROM seats WHERE free = 1 OR zone IS NULL) AS a) AS b WHERE n >= 2",
    # no filter at all
    "SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY d) AS n FROM (SELECT seat, seat - ROW_NUMBER() OVER (ORDER BY seat) AS d FROM seats) AS a) AS b WHERE n >= 2",
    # the membership form: the runs two rows or more share, listed from a copy of the numbered select
    f"SELECT seat FROM ({RUNS}) AS a WHERE run IN (SELECT run FROM ({RUNS}) AS c GROUP BY run HAVING COUNT(*) >= 2)",
    f"SELECT a.seat FROM ({RUNS}) AS a WHERE a.run IN (SELECT c.run FROM ({RUNS}) AS c GROUP BY c.run HAVING COUNT(c.run) > 1) ORDER BY a.seat",
    f"SELECT seat FROM ({RUNS}) AS a WHERE run IN (SELECT run FROM (SELECT run, COUNT(*) AS n FROM ({RUNS}) AS c GROUP BY run) AS g WHERE n > 1)",
    # the join form
    f"SELECT a.seat FROM ({RUNS}) AS a JOIN (SELECT run FROM ({RUNS}) AS c GROUP BY run HAVING COUNT(*) >= 2) AS g ON a.run = g.run",
    f"SELECT seat FROM (SELECT run AS r FROM ({RUNS}) AS c GROUP BY run HAVING COUNT(seat) >= 2) AS g INNER JOIN ({RUNS}) AS a ON g.r = a.run ORDER BY seat",
]

LEFT_ALONE = [
    # numbered in no promised order, backwards, or within partitions
    "SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY d) AS n FROM (SELECT seat, seat - ROW_NUMBER() OVER () AS d FROM seats WHERE free = 1) AS a) AS b WHERE n >= 2",
    "SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY d) AS n FROM (SELECT seat, seat - ROW_NUMBER() OVER (ORDER BY seat DESC) AS d FROM seats WHERE free = 1) AS a) AS b WHERE n >= 2",
    "SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY d) AS n FROM (SELECT seat, seat - ROW_NUMBER() OVER (PARTITION BY zone ORDER BY seat) AS d FROM seats WHERE free = 1) AS a) AS b WHERE n >= 2",
    # ordered by something that is not the key first
    "SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY d) AS n FROM (SELECT seat, seat - ROW_NUMBER() OVER (ORDER BY zone, seat) AS d FROM seats WHERE free = 1) AS a) AS b WHERE n >= 2",
    "SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY d) AS n FROM (SELECT seat, zone - ROW_NUMBER() OVER (ORDER BY seat) AS d FROM seats WHERE free = 1) AS a) AS b WHERE n >= 2",
    # no key, a key that is not a whole number, a key of two columns
    "SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY d) AS n FROM (SELECT seat, seat - ROW_NUMBER() OVER (ORDER BY seat) AS d FROM loose WHERE free = 1) AS a) AS b WHERE n >= 2",
    "SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY d) AS n FROM (SELECT seat, seat - ROW_NUMBER() OVER (ORDER BY seat) AS d FROM parts WHERE free = 1) AS a) AS b WHERE n >= 2",
    "SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY d) AS n FROM (SELECT seat, seat - ROW_NUMBER() OVER (ORDER BY seat) AS d FROM pairs WHERE free = 1) AS a) AS b WHERE n >= 2",
    # a run of three is more than a neighbour; a run of one is not read either
    f"SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY run) AS n FROM ({RUNS}) AS a) AS b WHERE n >= 3",
    f"SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY run) AS n FROM ({RUNS}) AS a) AS b WHERE n = 1",
    f"SELECT seat FROM ({RUNS}) AS a WHERE run IN (SELECT run FROM ({RUNS}) AS c GROUP BY run HAVING COUNT(*) >= 3)",
    # the count, the run or the row number read somewhere else
    f"SELECT seat, n FROM (SELECT seat, COUNT(*) OVER (PARTITION BY run) AS n FROM ({RUNS}) AS a) AS b WHERE n >= 2",
    f"SELECT seat, run FROM (SELECT seat, run, COUNT(*) OVER (PARTITION BY run) AS n FROM ({RUNS}) AS a) AS b WHERE n >= 2",
    f"SELECT * FROM (SELECT seat, COUNT(*) OVER (PARTITION BY run) AS n FROM ({RUNS}) AS a) AS b WHERE n >= 2",
    f"SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY run) AS n FROM ({RUNS}) AS a) AS b WHERE n >= 2 OR seat = 1",
    # a running count, a count of a column that may be NULL, a count of other rows
    f"SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY run ORDER BY seat) AS n FROM ({RUNS}) AS a) AS b WHERE n >= 2",
    "SELECT seat FROM (SELECT seat, COUNT(zone) OVER (PARTITION BY d) AS n FROM (SELECT seat, zone, seat - ROW_NUMBER() OVER (ORDER BY seat) AS d FROM seats WHERE free = 1) AS a) AS b WHERE n >= 2",
    f"SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY run) AS n FROM ({RUNS}) AS a WHERE seat > 3) AS b WHERE n >= 2",
    "SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY d, zone) AS n FROM (SELECT seat, zone, seat - ROW_NUMBER() OVER (ORDER BY seat) AS d FROM seats WHERE free = 1) AS a) AS b WHERE n >= 2",
    # the numbered select is more than a filtered table
    "SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY d) AS n FROM (SELECT seat, seat - ROW_NUMBER() OVER (ORDER BY seat) AS d FROM seats WHERE free = 1 LIMIT 3) AS a) AS b WHERE n >= 2",
    "SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY d) AS n FROM (SELECT DISTINCT seat, seat - ROW_NUMBER() OVER (ORDER BY seat) AS d FROM seats WHERE free = 1) AS a) AS b WHERE n >= 2",
    "SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY d) AS n FROM (SELECT seat, seat - ROW_NUMBER() OVER (ORDER BY seat) AS d FROM seats WHERE free = (SELECT MAX(free) FROM seats)) AS a) AS b WHERE n >= 2",
    "SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY d) AS n FROM (SELECT seat, seat - ROW_NUMBER() OVER (ORDER BY seat) AS d FROM seats WHERE ABS(free) = 1) AS a) AS b WHERE n >= 2",
    # the list of long runs is read from another select than the rows are
    f"SELECT seat FROM ({RUNS}) AS a WHERE run IN (SELECT run FROM (SELECT seat, seat - ROW_NUMBER() OVER (ORDER BY seat) AS run FROM seats WHERE free = 2) AS c GROUP BY run HAVING COUNT(*) >= 2)",
    f"SELECT seat FROM ({RUNS}) AS a WHERE run NOT IN (SELECT run FROM ({RUNS}) AS c GROUP BY run HAVING COUNT(*) >= 2)",
    f"SELECT a.seat FROM ({RUNS}) AS a LEFT JOIN (SELECT run FROM ({RUNS}) AS c GROUP BY run HAVING COUNT(*) >= 2) AS g ON a.run = g.run",
    f"SELECT a.seat, g.run FROM ({RUNS}) AS a JOIN (SELECT run FROM ({RUNS}) AS c GROUP BY run HAVING COUNT(*) >= 2) AS g ON a.run = g.run",
]


def read(sql: str, dialect: str = "mysql"):
    tree = sqlglot.parse_one(sql, read=dialect)
    new, wrote = read_numbered_islands(tree, SCHEMA, KEYS, NOT_NULL, TYPES)
    return new.sql(dialect=dialect), wrote


def seats(seed: int) -> list[tuple]:
    rng = random.Random(seed)
    numbers = rng.sample(range(-3, 14), rng.randint(0, 14))
    return [(n, rng.choice([1, 1, 1, 0, 2, None]), rng.choice([1, 2, None]), rng.choice([1.5, 2.25])) for n in numbers]


def rows(engine: str, sql: str, data: list[tuple]) -> list:
    con = sqlite3.connect(":memory:") if engine == "sqlite" else duckdb.connect()
    con.execute("CREATE TABLE seats (seat INT, free INT, zone INT, price DECIMAL(8,2))")
    if data:
        con.executemany("INSERT INTO seats VALUES (?, ?, ?, ?)", data)
    return sorted(con.execute(sql).fetchall(), key=repr)


@pytest.mark.parametrize("sql", READ)
def test_the_reading_returns_the_rows_of_the_query_as_written(sql):
    new, wrote = read(sql, "duckdb")
    assert wrote and "ROW_NUMBER" not in new and "RANK" not in new and "EXISTS" in new
    for seed in range(60):
        data = seats(seed)
        for engine in ("duckdb", "sqlite"):
            assert rows(engine, new, data) == rows(engine, sql, data), (seed, engine)


def test_the_tables_drawn_hold_runs_of_every_kind():
    lengths = set()
    for seed in range(60):
        free = sorted(n for n, flag, *_ in seats(seed) if flag == 1)
        run = 1
        for a, b in zip(free, free[1:] + [None]):
            if b == a + 1:
                run += 1
            else:
                lengths.add(run)
                run = 1
    assert {1, 2, 3} <= lengths


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_queries_are_left_alone(sql):
    new, wrote = read(sql)
    assert not wrote and new == sqlglot.parse_one(sql, read="mysql").sql(dialect="mysql")


def test_the_reading_is_one_filtered_read_of_the_table():
    new, wrote = read(READ[0])
    assert wrote
    assert new == (
        "SELECT b.seat FROM (SELECT kqi0.seat FROM seats AS kqi0 WHERE (kqi0.free = 1) AND EXISTS(SELECT 1 FROM seats AS kqi1 "
        "WHERE (kqi1.free = 1) AND (kqi1.seat = kqi0.seat - 1 OR kqi1.seat = kqi0.seat + 1))) AS b"
    )


def test_a_name_already_in_the_query_is_not_taken():
    new, wrote = read(READ[0].replace("AS a)", "AS kqi0)").replace("a.", "kqi0."))
    assert wrote and "seats AS kqi1 WHERE" in new and "seats AS kqi2 WHERE" in new


def test_why_a_key_must_be_a_whole_number():
    # 1, 1.5 and 3 are numbered 1, 2 and 3: the first and the third share the difference 0 and are no neighbours
    con = duckdb.connect()
    con.execute("CREATE TABLE parts (seat DECIMAL(8,2), free INT)")
    con.executemany("INSERT INTO parts VALUES (?, 1)", [(1,), (1.5,), (3,)])
    sql = "SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY d) AS n FROM (SELECT seat, seat - ROW_NUMBER() OVER (ORDER BY seat) AS d FROM parts WHERE free = 1) AS a) AS b WHERE n >= 2"
    by_hand = "SELECT x.seat FROM parts AS x WHERE x.free = 1 AND EXISTS (SELECT 1 FROM parts AS y WHERE y.free = 1 AND (y.seat = x.seat - 1 OR y.seat = x.seat + 1))"
    assert len(con.execute(sql).fetchall()) == 2 and con.execute(by_hand).fetchall() == []


def test_why_a_run_of_three_is_not_read():
    data = [(1, 1, 1, 1.5), (2, 1, 1, 1.5), (5, 1, 1, 1.5), (6, 1, 1, 1.5), (7, 1, 1, 1.5)]
    three = f"SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY run) AS n FROM ({RUNS}) AS a) AS b WHERE n >= 3"
    two = three.replace(">= 3", ">= 2")
    assert rows("duckdb", three, data) != rows("duckdb", two, data)


REFERENCE = (
    "SELECT s.seat FROM seats AS s WHERE s.free = 1 AND ((SELECT n.free FROM seats AS n WHERE n.seat = s.seat - 1) = 1 "
    "OR (SELECT n.free FROM seats AS n WHERE n.seat = s.seat + 1) = 1)"
)
PROVED = [READ[0], READ[1], READ[3], READ[6], READ[9], READ[11], READ[12]]


def prove(left: str, right: str, dialect: str = "mysql"):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, dialect=dialect, compare_names=False)


@pytest.mark.parametrize("sql", PROVED)
def test_the_numbered_spelling_is_proved_equal_to_the_lookup_spelling(sql):
    assert prove(REFERENCE, sql).proven
    with shape_fallback.without("numbered_islands"):
        assert not prove(REFERENCE, sql).proven


def test_two_numbered_spellings_are_proved_equal():
    assert prove(READ[0], READ[12]).proven
    assert prove(READ[1], READ[9], "duckdb").proven


def test_a_longer_run_is_not_proved_equal_to_a_neighbour():
    three = f"SELECT seat FROM (SELECT seat, COUNT(*) OVER (PARTITION BY run) AS n FROM ({RUNS}) AS a) AS b WHERE n >= 3"
    assert not prove(REFERENCE, three).proven
    assert not prove(READ[1], three).proven


def test_without_a_key_nothing_is_proved():
    loose = {**CONSTRAINTS, "seats": TableConstraints(not_null=NOT_NULL["seats"], keys=())}
    result = prove_equivalent_algebraic(REFERENCE, READ[0], schema=SCHEMA, types=TYPES, constraints=loose, dialect="mysql", compare_names=False)
    assert not result.proven
    nullable = {**CONSTRAINTS, "seats": TableConstraints(not_null=frozenset(), keys=(("seat",),))}
    result = prove_equivalent_algebraic(REFERENCE, READ[0], schema=SCHEMA, types=TYPES, constraints=nullable, dialect="mysql", compare_names=False)
    assert not result.proven
