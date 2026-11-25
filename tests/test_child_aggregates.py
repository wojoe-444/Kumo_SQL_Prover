"""A parent row's ``SUM``, ``AVG``, ``MIN`` or ``MAX`` over its child rows (``child_aggregates``).

Each reading is run beside the query as written on SQLite and on DuckDB over made-up rows; each query the rule
leaves alone has a hand-made reading that returns other rows.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot
from sqlglot import exp

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.child_aggregates import read_child_aggregates, read_grouped_left_join, read_subquery_spellings, split_joined_aggregates
from kumo_sql_prover.smt_equivalence import SmtStatus, TableConstraints

TABLES = {
    # ann leads two staff labelled bo; the first bo leads cy and di (di's age is not known), cy leads ed (age not known), nobody has lead 9
    "staff": ("staff_id INT, label VARCHAR, lead_id INT, age INT", [(1, "ann", None, 50), (2, "bo", 1, 30), (3, "bo", 1, 44), (4, "cy", 2, 41), (5, "di", 2, None), (6, "ed", 4, None), (7, "flo", 9, 30)]),
    "shifts": ("staff_id INT, day DATE, hours INT, rate INT", [(2, "2024-03-01", 4, 10), (2, "2024-03-02", 5, 10), (3, "2024-03-01", 1, 12), (4, "2024-03-01", 6, None), (None, "2024-03-03", 8, 9)]),
    # two badge codes that are the same number
    "badges": ("code VARCHAR, hours INT", [("2", 1), ("02", 2), ("3", 5)]),
    # two notes without an id
    "notes": ("note_id INT, staff_id INT, body VARCHAR", [(None, 2, "late"), (None, 2, "late"), (7, 3, "early")]),
}
SCHEMA = {name: [part.split()[0] for part in columns.split(", ")] for name, (columns, _) in TABLES.items()}
TYPES = {name: dict(part.split() for part in columns.split(", ")) for name, (columns, _) in TABLES.items()}
KEYS = {"staff": [("staff_id",)], "shifts": [("staff_id", "day")], "badges": [("code",)], "notes": [("note_id",)]}
NOT_NULL = {"staff": frozenset({"staff_id"}), "shifts": frozenset({"day"}), "badges": frozenset({"code"}), "notes": frozenset()}
CONSTRAINTS = {name: TableConstraints(not_null=NOT_NULL[name], keys=tuple(KEYS[name])) for name in TABLES}
REPORTS = "FROM staff AS m LEFT JOIN staff AS r ON r.lead_id = m.staff_id"
WORKED = "FROM staff AS m LEFT JOIN shifts AS s ON s.staff_id = m.staff_id"
PER_LEAD = "(SELECT lead_id, AVG(age) AS mean_age, MAX(age) AS oldest, COUNT(*) AS reports FROM staff GROUP BY lead_id) AS g"

READ = [
    # grouped after the join
    f"SELECT m.staff_id, AVG(r.age) AS mean_age {REPORTS} GROUP BY m.staff_id",
    f"SELECT m.staff_id, m.label, MIN(r.age) AS youngest, MAX(r.age) AS oldest, SUM(r.age) AS total {REPORTS} GROUP BY m.staff_id, m.label",
    f"SELECT m.label, COUNT(r.age) AS known, COUNT(r.staff_id) AS reports, ROUND(AVG(r.age), 0) AS mean_age {REPORTS} GROUP BY m.staff_id",
    f"SELECT m.staff_id, SUM(s.hours * s.rate) AS pay, COUNT(DISTINCT s.hours) AS lengths {WORKED} GROUP BY m.staff_id",
    f"SELECT m.staff_id, SUM(s.hours) AS long_hours {WORKED} AND s.hours > 4 GROUP BY m.staff_id",
    f"SELECT m.staff_id, COALESCE(SUM(s.hours), 0) + m.age AS score {WORKED} WHERE m.age > 35 GROUP BY m.staff_id",
    f"SELECT d.staff_id FROM (SELECT m.staff_id, MAX(r.age) AS oldest {REPORTS} GROUP BY m.staff_id) AS d WHERE d.oldest > 40",
    # aggregated first
    f"SELECT m.staff_id, g.mean_age, g.oldest FROM staff AS m LEFT JOIN {PER_LEAD} ON g.lead_id = m.staff_id",
    f"SELECT m.staff_id, g.oldest FROM staff AS m LEFT JOIN {PER_LEAD} ON m.staff_id = g.lead_id WHERE m.age > 35",
    f"SELECT m.staff_id, COALESCE(g.reports, 0) AS reports, g.oldest - g.mean_age AS spread FROM staff AS m LEFT JOIN {PER_LEAD} ON g.lead_id = m.staff_id",
    f"SELECT m.staff_id FROM staff AS m LEFT JOIN {PER_LEAD} ON g.lead_id = m.staff_id",
    "SELECT m.staff_id, w.total, w.longest FROM staff AS m LEFT JOIN (SELECT s.staff_id, s.day, SUM(s.hours) AS total, MAX(s.hours) AS longest FROM shifts AS s WHERE s.hours > 3 GROUP BY s.staff_id, s.day) AS w ON w.staff_id = m.staff_id AND w.day = '2024-03-01'",
    # a function around the aggregate of a per-row subquery
    "SELECT m.staff_id, (SELECT ROUND(AVG(r.age), 0) FROM staff AS r WHERE r.lead_id = m.staff_id) AS mean_age FROM staff AS m",
    "SELECT m.staff_id, (SELECT COALESCE(SUM(s.hours), 0) + COUNT(*) FROM shifts AS s WHERE s.staff_id = m.staff_id) AS score FROM staff AS m",
    "SELECT m.staff_id, 1 + (SELECT ABS(MIN(r.age) - MAX(r.age)) FROM staff AS r WHERE r.lead_id = m.staff_id) AS spread FROM staff AS m",
    # a per-row subquery grouped by the column its WHERE fixes
    "SELECT m.staff_id, (SELECT AVG(r.age) FROM staff AS r WHERE r.lead_id = m.staff_id GROUP BY r.lead_id) AS mean_age FROM staff AS m",
    "SELECT m.staff_id, (SELECT COUNT(*) FROM staff AS r WHERE r.lead_id = m.staff_id GROUP BY r.lead_id) AS reports FROM staff AS m",
    "SELECT m.staff_id, (SELECT ROUND(AVG(r.age) + 1, 0) FROM staff AS r WHERE r.lead_id = m.staff_id GROUP BY r.lead_id) AS mean_age FROM staff AS m",
    "SELECT m.staff_id, (SELECT SUM(s.rate) FROM shifts AS s WHERE s.staff_id = m.staff_id AND s.hours = m.lead_id GROUP BY s.hours, s.staff_id) AS total FROM staff AS m",
]

LEFT_ALONE = [
    # the padded row is counted, or is not NULL under the aggregate
    f"SELECT m.staff_id, COUNT(*) AS n, MAX(r.age) AS oldest {REPORTS} GROUP BY m.staff_id",
    f"SELECT m.staff_id, SUM(COALESCE(r.age, 1)) AS total {REPORTS} GROUP BY m.staff_id",
    f"SELECT m.staff_id, SUM(CASE WHEN r.age > 40 THEN 1 ELSE 0 END) AS older {REPORTS} GROUP BY m.staff_id",
    f"SELECT m.staff_id, MAX(r.age / 2) AS half {REPORTS} GROUP BY m.staff_id",
    f"SELECT m.staff_id, SUM(1) AS n {REPORTS} GROUP BY m.staff_id",
    # an aggregate that reads the parent
    f"SELECT m.staff_id, SUM(m.age) AS total {REPORTS} GROUP BY m.staff_id",
    f"SELECT m.staff_id, SUM(r.age + m.age) AS total {REPORTS} GROUP BY m.staff_id",
    # the group is not one parent row
    f"SELECT m.label, MAX(r.age) AS oldest {REPORTS} GROUP BY m.label",
    f"SELECT r.lead_id, MAX(r.age) AS oldest {REPORTS} GROUP BY r.lead_id",
    "SELECT n.note_id, SUM(s.hours) AS total FROM notes AS n LEFT JOIN shifts AS s ON s.staff_id = n.staff_id GROUP BY n.note_id",
    "SELECT s.staff_id, MAX(r.age) AS oldest FROM shifts AS s LEFT JOIN staff AS r ON r.lead_id = s.staff_id GROUP BY s.staff_id",
    # a text set equal to a number
    "SELECT m.staff_id, SUM(b.hours) AS total FROM staff AS m LEFT JOIN badges AS b ON b.code = m.staff_id GROUP BY m.staff_id",
    # other joins, tests and clauses
    "SELECT m.staff_id, MAX(r.age) AS oldest FROM staff AS m JOIN staff AS r ON r.lead_id = m.staff_id GROUP BY m.staff_id",
    "SELECT m.staff_id, MAX(r.age) AS oldest FROM staff AS r RIGHT JOIN staff AS m ON r.lead_id = m.staff_id GROUP BY m.staff_id",
    f"SELECT m.staff_id, MAX(r.age) AS oldest {REPORTS} AND m.age > 35 GROUP BY m.staff_id",
    f"SELECT m.staff_id, MAX(r.age) AS oldest {REPORTS} OR r.age > 40 GROUP BY m.staff_id",
    "SELECT m.staff_id, MAX(r.age) AS oldest FROM staff AS m LEFT JOIN staff AS r ON r.lead_id < m.staff_id GROUP BY m.staff_id",
    f"SELECT m.staff_id, MAX(r.age) AS oldest {REPORTS} WHERE r.age > 35 GROUP BY m.staff_id",
    f"SELECT m.staff_id, MAX(r.age) AS oldest {REPORTS} GROUP BY m.staff_id HAVING MAX(r.age) > 35",
    f"SELECT m.staff_id, MAX(r.age) AS oldest {REPORTS} GROUP BY m.staff_id ORDER BY m.staff_id LIMIT 2",
    f"SELECT DISTINCT m.label, MAX(r.age) AS oldest {REPORTS} GROUP BY m.staff_id",
    f"SELECT m.staff_id, MAX(r.age) AS oldest, r.label {REPORTS} GROUP BY m.staff_id",
    f"SELECT m.staff_id, MAX(r.age) AS oldest {REPORTS} LEFT JOIN shifts AS s ON s.staff_id = m.staff_id GROUP BY m.staff_id",
    "WITH staff AS (SELECT staff_id, staff_id AS lead_id, 1 AS age FROM notes) SELECT m.staff_id, MAX(r.age) AS oldest FROM staff AS m LEFT JOIN staff AS r ON r.lead_id = m.staff_id GROUP BY m.staff_id",
    # an aggregated-first table that is not found by all its group columns, is read by them, or holds one aggregate
    "SELECT m.staff_id, w.total, w.longest FROM staff AS m LEFT JOIN (SELECT s.staff_id, s.day, SUM(s.hours) AS total, MAX(s.hours) AS longest FROM shifts AS s GROUP BY s.staff_id, s.day) AS w ON w.staff_id = m.staff_id",
    f"SELECT m.staff_id, g.lead_id, g.oldest FROM staff AS m LEFT JOIN {PER_LEAD} ON g.lead_id = m.staff_id",
    f"SELECT m.staff_id, g.oldest FROM staff AS m LEFT JOIN {PER_LEAD} ON g.lead_id = m.staff_id AND g.reports > 1",
    f"SELECT m.staff_id, g.oldest FROM staff AS m JOIN {PER_LEAD} ON g.lead_id = m.staff_id",
    f"SELECT m.staff_id, oldest FROM staff AS m LEFT JOIN {PER_LEAD} ON g.lead_id = m.staff_id",
    f"SELECT * FROM staff AS m LEFT JOIN {PER_LEAD} ON g.lead_id = m.staff_id",
    "SELECT m.staff_id, g.oldest FROM staff AS m LEFT JOIN (SELECT lead_id, MAX(age) AS oldest FROM staff GROUP BY lead_id) AS g ON g.lead_id = m.staff_id",
    "SELECT m.staff_id, g.oldest, g.n FROM staff AS m LEFT JOIN (SELECT lead_id, MAX(age) AS oldest, COUNT(*) AS n FROM staff GROUP BY lead_id HAVING COUNT(*) > 1) AS g ON g.lead_id = m.staff_id",
    # per-row subqueries outside the two spellings
    "SELECT m.staff_id, (SELECT AVG(r.age) FROM staff AS r WHERE r.lead_id = m.staff_id) AS mean_age FROM staff AS m",
    "SELECT m.staff_id, (SELECT SUM(r.age) / COUNT(*) FROM staff AS r WHERE r.lead_id = m.staff_id) AS mean_age FROM staff AS m",
    "SELECT m.staff_id, (SELECT ROUND(AVG(r.age), 0) FROM staff AS r WHERE r.lead_id = m.staff_id LIMIT 1) AS mean_age FROM staff AS m",
    "SELECT m.staff_id, (SELECT ROUND(AVG(r.age), 0) + m.age FROM staff AS r WHERE r.lead_id = m.staff_id) AS mean_age FROM staff AS m",
    "SELECT m.staff_id, (SELECT CONCAT(MAX(r.label), 'x') FROM staff AS r WHERE r.lead_id = m.staff_id) AS mean_age FROM staff AS m",
    "SELECT m.staff_id, (SELECT AVG(r.age) FROM staff AS r WHERE r.lead_id = m.staff_id GROUP BY r.label) AS mean_age FROM staff AS m",
    "SELECT m.staff_id, (SELECT AVG(r.age) FROM staff AS r WHERE r.lead_id < m.staff_id GROUP BY r.lead_id) AS mean_age FROM staff AS m",
    "SELECT m.staff_id, (SELECT AVG(r.age) FROM staff AS r WHERE r.lead_id = r.staff_id GROUP BY r.lead_id) AS mean_age FROM staff AS m",
    "SELECT m.staff_id, (SELECT COUNT(r.age) FROM staff AS r WHERE r.lead_id = m.staff_id GROUP BY r.lead_id) AS known FROM staff AS m",
    "SELECT m.staff_id, (SELECT AVG(r.age) FROM staff AS r WHERE r.lead_id = m.staff_id GROUP BY r.lead_id HAVING COUNT(*) > 1) AS mean_age FROM staff AS m",
    "SELECT m.staff_id, (SELECT COALESCE(SUM(r.age), 0) FROM staff AS r WHERE r.lead_id = m.staff_id GROUP BY r.lead_id) AS total FROM staff AS m",
]


def _database(dialect: str):
    database = pytest.importorskip("duckdb").connect() if dialect == "duckdb" else sqlite3.connect(":memory:")
    for name, (columns, rows) in TABLES.items():
        database.execute(f"CREATE TABLE {name} ({columns})")
        database.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(rows[0]))})", rows)
    return database


def _rows(database, sql: str, dialect: str) -> list:
    rows = database.execute(sqlglot.transpile(sql, read="mysql", write=dialect)[0]).fetchall()
    return sorted([tuple(float(v) if isinstance(v, (int, float)) or type(v).__name__ == "Decimal" else v for v in row) for row in rows], key=repr)


def _read(sql: str):
    """The statement with every select the rule reads written its way (to a fixed point), None when it reads none."""

    tree, fired = sqlglot.parse_one(sql, read="mysql"), False
    for _ in range(20):
        for select in list(tree.find_all(exp.Select)):
            reading = read_child_aggregates(select, SCHEMA, KEYS, NOT_NULL, TYPES)
            if reading is not None:
                if select is tree:
                    tree = reading
                else:
                    select.replace(reading)
                fired = True
                break
        else:
            break
    return tree if fired else None


@pytest.mark.parametrize("written", READ)
@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_a_reading_returns_the_rows_of_the_query_as_written(written, dialect):
    read = _read(written)
    assert read is not None and read.sql(dialect="mysql") != sqlglot.parse_one(written, read="mysql").sql(dialect="mysql")
    database = _database(dialect)
    try:
        as_written = _rows(database, written, dialect)
    except Exception as error:  # DuckDB wants every shown parent column in the GROUP BY; SQLite and MySQL run it
        if dialect == "duckdb" and "must appear in the GROUP BY clause" in str(error):
            pytest.skip("DuckDB rejects a column the GROUP BY does not list")
        raise
    assert as_written == _rows(database, read.sql(dialect="mysql"), dialect)


def test_the_readings_are_written_one_table_per_aggregate():
    read = _read(READ[1]).sql(dialect="mysql")
    assert read.count("LEFT JOIN (SELECT r.lead_id AS kqk0") == 3 and "GROUP BY r.lead_id" in read and "GROUP BY m." not in read
    assert "COALESCE(kqa0.kqv, 0) AS known" in _read(READ[2]).sql(dialect="mysql")
    split = _read(READ[7]).sql(dialect="mysql")
    assert split.count("LEFT JOIN (SELECT lead_id AS kqk0") == 2 and "COUNT(*)" not in split  # the count nobody reads is not joined
    assert "LEFT JOIN" not in _read(READ[10]).sql(dialect="mysql")  # nothing of the table is read
    assert _read(READ[12]).sql(dialect="mysql") == "SELECT m.staff_id, (ROUND((SELECT AVG(r.age) FROM staff AS r WHERE r.lead_id = m.staff_id), 0)) AS mean_age FROM staff AS m"
    assert _read(READ[16]).sql(dialect="mysql") == "SELECT m.staff_id, NULLIF((SELECT COUNT(*) FROM staff AS r WHERE r.lead_id = m.staff_id), 0) AS reports FROM staff AS m"


def test_the_made_up_rows_tell_the_cases_apart():
    database = _database("sqlite")
    # a lead with reports, a lead whose reports have no known age, staff who lead nobody
    assert _rows(database, READ[0], "sqlite") == sorted([(1.0, 37.0), (2.0, 41.0), (3.0, None), (4.0, None), (5.0, None), (6.0, None), (7.0, None)], key=repr)
    assert ("cy", 0.0, 1.0, None) in _rows(database, READ[2], "sqlite")  # one report, of no known age
    assert (4.0, None, 1.0) in _rows(database, READ[3], "sqlite")  # hours without a rate
    assert (3.0, None) in _rows(database, READ[16], "sqlite") and (1.0, 2.0) in _rows(database, READ[16], "sqlite")
    assert (3.0, 12.0) in _rows(database, READ[18], "sqlite")


@pytest.mark.parametrize("written", LEFT_ALONE)
def test_a_query_outside_the_readings_stays_as_written(written):
    assert _read(written) is None


def test_nothing_is_read_without_the_declared_facts():
    tree = sqlglot.parse_one(READ[0], read="mysql")
    assert read_grouped_left_join(tree, SCHEMA, KEYS, NOT_NULL, TYPES) is not None
    assert read_grouped_left_join(tree, SCHEMA, None, NOT_NULL, TYPES) is None
    assert read_grouped_left_join(tree, SCHEMA, KEYS, None, TYPES) is None  # the key may be NULL
    assert read_grouped_left_join(tree, None, KEYS, NOT_NULL, TYPES) is None
    assert read_grouped_left_join(tree, SCHEMA, KEYS, NOT_NULL, None) is not None  # both linked columns undeclared
    half = {"staff": {"staff_id": "INT"}}
    assert read_grouped_left_join(tree, SCHEMA, KEYS, NOT_NULL, half) is None  # one linked column declared, the other not


def test_the_select_it_is_given_is_not_changed():
    for written, rule in ((READ[0], lambda s: read_grouped_left_join(s, SCHEMA, KEYS, NOT_NULL, TYPES)), (READ[7], split_joined_aggregates), (READ[12], read_subquery_spellings), (READ[15], read_subquery_spellings)):
        tree = sqlglot.parse_one(written, read="mysql")
        before = tree.sql()
        read = rule(tree)
        assert read is not None and read is not tree and tree.sql() == before


# (a query left alone, what the reading would be if it were read anyway) with different rows on SQLite
BY_HAND = [
    # COUNT(*) counts the padded row
    (LEFT_ALONE[0], "SELECT m.staff_id, COALESCE(g.n, 0) AS n, g.oldest FROM staff AS m LEFT JOIN (SELECT lead_id, COUNT(*) AS n, MAX(age) AS oldest FROM staff GROUP BY lead_id) AS g ON g.lead_id = m.staff_id"),
    # COALESCE and CASE have a value on the padded row
    (LEFT_ALONE[1], "SELECT m.staff_id, g.total FROM staff AS m LEFT JOIN (SELECT lead_id, SUM(COALESCE(age, 1)) AS total FROM staff GROUP BY lead_id) AS g ON g.lead_id = m.staff_id"),
    (LEFT_ALONE[2], "SELECT m.staff_id, g.older FROM staff AS m LEFT JOIN (SELECT lead_id, SUM(CASE WHEN age > 40 THEN 1 ELSE 0 END) AS older FROM staff GROUP BY lead_id) AS g ON g.lead_id = m.staff_id"),
    # two staff with one label are one group
    (LEFT_ALONE[7], "SELECT m.label, g.oldest FROM staff AS m LEFT JOIN (SELECT lead_id, MAX(age) AS oldest FROM staff GROUP BY lead_id) AS g ON g.lead_id = m.staff_id"),
    # two notes without an id are one group
    (LEFT_ALONE[9], "SELECT n.note_id, g.total FROM notes AS n LEFT JOIN (SELECT staff_id, SUM(hours) AS total FROM shifts GROUP BY staff_id) AS g ON g.staff_id = n.staff_id"),
    # '2' and '02' both equal the number 2 and are two groups
    (LEFT_ALONE[11], "SELECT m.staff_id, g.total FROM staff AS m LEFT JOIN (SELECT code, SUM(hours) AS total FROM badges GROUP BY code) AS g ON g.code = m.staff_id"),
    # a test on the parent in the ON pads the row; in a WHERE it would drop it
    (LEFT_ALONE[14], "SELECT m.staff_id, g.oldest FROM staff AS m LEFT JOIN (SELECT lead_id, MAX(age) AS oldest FROM staff GROUP BY lead_id) AS g ON g.lead_id = m.staff_id WHERE m.age > 35"),
    # a table found by one of its two group columns has several rows for a parent row
    (LEFT_ALONE[24], "SELECT m.staff_id, a.total, b.longest FROM staff AS m LEFT JOIN (SELECT s.staff_id, s.day, SUM(s.hours) AS total FROM shifts AS s GROUP BY s.staff_id, s.day) AS a ON a.staff_id = m.staff_id LEFT JOIN (SELECT s.staff_id, s.day, MAX(s.hours) AS longest FROM shifts AS s GROUP BY s.staff_id, s.day) AS b ON b.staff_id = m.staff_id"),
    # COUNT of a column can be 0 for a group; NULLIF would hide it
    (LEFT_ALONE[40], "SELECT m.staff_id, NULLIF((SELECT COUNT(r.age) FROM staff AS r WHERE r.lead_id = m.staff_id), 0) AS known FROM staff AS m"),
    # COALESCE has a value for no rows; with the GROUP BY there is no row
    (LEFT_ALONE[42], "SELECT m.staff_id, (SELECT COALESCE(SUM(r.age), 0) FROM staff AS r WHERE r.lead_id = m.staff_id) AS total FROM staff AS m"),
]


@pytest.mark.parametrize("written, by_hand", BY_HAND)
def test_outside_the_readings_the_reading_would_return_other_rows(written, by_hand):
    database = _database("sqlite")
    assert _rows(database, written, "sqlite") != _rows(database, by_hand, "sqlite")
    assert _read(written) is None


OPTIONS = dict(schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect="mysql")
SUBQUERIES = "SELECT m.staff_id, m.label, (SELECT MAX(r.age) FROM staff AS r WHERE r.lead_id = m.staff_id) AS oldest, (SELECT AVG(r.age) FROM staff AS r WHERE r.lead_id = m.staff_id) AS mean_age FROM staff AS m"
GROUPED = f"SELECT m.staff_id, m.label, MAX(r.age) AS oldest, AVG(r.age) AS mean_age {REPORTS} GROUP BY m.staff_id"
JOINED = f"SELECT m.staff_id, m.label, g.oldest, g.mean_age FROM staff AS m LEFT JOIN {PER_LEAD} ON g.lead_id = m.staff_id"
PAIRS = [
    (SUBQUERIES, GROUPED),
    (SUBQUERIES, JOINED),
    (GROUPED, JOINED),
    (
        "SELECT m.staff_id, (SELECT COUNT(*) FROM staff AS r WHERE r.lead_id = m.staff_id) AS reports, (SELECT ROUND(AVG(r.age), 0) FROM staff AS r WHERE r.lead_id = m.staff_id GROUP BY r.lead_id) AS mean_age FROM staff AS m",
        f"SELECT m.staff_id, COUNT(r.staff_id) AS reports, ROUND(AVG(r.age), 0) AS mean_age {REPORTS} GROUP BY m.staff_id",
    ),
]


@pytest.mark.parametrize("left, right", PAIRS)
def test_the_three_spellings_are_proved_equal_and_only_with_the_rule(left, right):
    database = _database("sqlite")
    assert _rows(database, left, "sqlite") == _rows(database, right, "sqlite")
    assert prove_equivalent_algebraic(left, right, **OPTIONS).status is SmtStatus.PROVEN_EQUIVALENT
    with shape_fallback.without("child_aggregates"):
        assert prove_equivalent_algebraic(left, right, **OPTIONS).status is not SmtStatus.PROVEN_EQUIVALENT


NOT_EQUAL = [
    # COUNT(*) after the join is 1 for a lead of nobody
    (f"SELECT m.staff_id, COUNT(*) AS n, MAX(r.age) AS oldest {REPORTS} GROUP BY m.staff_id", "SELECT m.staff_id, (SELECT COUNT(*) FROM staff AS r WHERE r.lead_id = m.staff_id) AS n, (SELECT MAX(r.age) FROM staff AS r WHERE r.lead_id = m.staff_id) AS oldest FROM staff AS m"),
    # a text set equal to a number: '2' and '02' are one sum per row and two groups
    (LEFT_ALONE[11], BY_HAND[5][1]),
    ("SELECT m.staff_id, (SELECT SUM(b.hours) FROM badges AS b WHERE b.code = m.staff_id) AS total FROM staff AS m", BY_HAND[5][1]),
    # grouped by a label two staff share
    (LEFT_ALONE[7], "SELECT m.label, (SELECT MAX(r.age) FROM staff AS r WHERE r.lead_id = m.staff_id) AS oldest FROM staff AS m"),
    # the sum of a value that is not NULL on the padded row
    (LEFT_ALONE[1], "SELECT m.staff_id, (SELECT SUM(COALESCE(r.age, 1)) FROM staff AS r WHERE r.lead_id = m.staff_id) AS total FROM staff AS m"),
]


@pytest.mark.parametrize("left, right", NOT_EQUAL)
def test_pairs_that_differ_are_not_proved(left, right):
    database = _database("sqlite")
    assert _rows(database, left, "sqlite") != _rows(database, right, "sqlite")
    assert prove_equivalent_algebraic(left, right, **OPTIONS).status is not SmtStatus.PROVEN_EQUIVALENT


def test_a_key_that_may_be_null_gives_no_proof():
    left = "SELECT n.note_id, SUM(s.hours) AS total FROM notes AS n LEFT JOIN shifts AS s ON s.staff_id = n.staff_id GROUP BY n.note_id"
    right = "SELECT n.note_id, (SELECT SUM(s.hours) FROM shifts AS s WHERE s.staff_id = n.staff_id) AS total FROM notes AS n"
    database = _database("sqlite")
    assert _rows(database, left, "sqlite") != _rows(database, right, "sqlite")
    assert prove_equivalent_algebraic(left, right, **OPTIONS).status is not SmtStatus.PROVEN_EQUIVALENT
    solid = dict(OPTIONS, constraints=dict(CONSTRAINTS, notes=TableConstraints(not_null=frozenset({"note_id"}), keys=(("note_id",),))))
    assert prove_equivalent_algebraic(left, right, **solid).status is SmtStatus.PROVEN_EQUIVALENT

