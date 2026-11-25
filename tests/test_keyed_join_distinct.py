"""``SELECT DISTINCT`` over an inner join whose outputs fix every joined row removes nothing (``keyed_join_distinct``).

Each reading is run with and without its ``DISTINCT`` on SQLite and on DuckDB over made-up rows.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot
from sqlglot import exp

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.keyed_join_distinct import read_keyed_join_distinct
from kumo_sql_prover.smt_equivalence import TableConstraints

TABLES = {
    "staff": ("staff_id INT, label VARCHAR, lead_id INT, age INT", [(1, "ann", None, 50), (2, "bo", 1, 30), (3, "bo", 1, 30), (4, "cy", 2, 41), (5, "a1", 2, None), (6, "ann", 9, 30)]),
    "shifts": ("staff_id INT, day DATE, hours INT", [(2, "2024-03-01", 4), (2, "2024-03-02", 4), (3, "2024-03-01", 4), (4, "2024-03-01", 6)]),
    "badges": ("code VARCHAR, staff_id INT", [("a1", 2), ("A1", 2), ("b7", 3)]),
    "notes": ("note_id INT, staff_id INT, body VARCHAR", [(None, 2, "late"), (None, 2, "late"), (7, 3, "early")]),
}
SCHEMA = {name: [part.split()[0] for part in columns.split(", ")] for name, (columns, _) in TABLES.items()}
TYPES = {name: dict(part.split() for part in columns.split(", ")) for name, (columns, _) in TABLES.items()}
KEYS = {"staff": [("staff_id",)], "shifts": [("staff_id", "day")], "badges": [("code",)], "notes": [("note_id",)]}
NOT_NULL = {"staff": frozenset({"staff_id"}), "shifts": frozenset({"staff_id", "day"}), "badges": frozenset({"code"}), "notes": frozenset()}
CONSTRAINTS = {name: TableConstraints(not_null=NOT_NULL[name], keys=tuple(KEYS[name])) for name in TABLES}
LEADS = "FROM staff AS m JOIN staff AS r ON m.staff_id = r.lead_id"

READ = [
    # a key of each read is shown
    f"SELECT DISTINCT m.staff_id AS lead, r.staff_id, m.label, r.age {LEADS}",
    # the report's key is shown, and the join's test tells the lead's
    f"SELECT DISTINCT r.staff_id, m.label {LEADS}",
    f"SELECT DISTINCT r.staff_id, r.age + 1 AS next_age, m.label {LEADS}",
    "SELECT DISTINCT r.staff_id, m.label FROM staff AS m, staff AS r WHERE m.staff_id = r.lead_id AND r.age > 20",
    "SELECT DISTINCT r.staff_id, m.label FROM staff AS m CROSS JOIN staff AS r WHERE m.staff_id = r.lead_id",
    # a key a whole-number literal fixes
    "SELECT DISTINCT r.staff_id, m.label FROM staff AS m JOIN staff AS r ON r.age >= m.age WHERE m.staff_id = 2",
    "SELECT DISTINCT r.staff_id, m.label FROM staff AS m, staff AS r WHERE 6 = m.staff_id",
    # a key of two columns, one of them a date
    "SELECT DISTINCT s.staff_id, s.day, m.label FROM shifts AS s JOIN staff AS m ON m.staff_id = s.staff_id",
    "SELECT DISTINCT s.day, r.staff_id, m.label FROM shifts AS s JOIN staff AS r ON r.staff_id = s.staff_id JOIN staff AS m ON m.staff_id = r.lead_id",
    # three reads, each reached through the one before
    "SELECT DISTINCT s.staff_id AS who, s.day, s.hours FROM shifts AS s JOIN staff AS r ON r.staff_id = s.staff_id JOIN staff AS m ON r.lead_id = m.staff_id WHERE m.age > 0",
    # inside a derived table and a WITH
    f"SELECT d.lead, COUNT(*) AS n FROM (SELECT DISTINCT m.staff_id AS lead, r.staff_id {LEADS}) AS d GROUP BY d.lead",
    f"WITH d AS (SELECT DISTINCT m.staff_id AS lead, r.staff_id {LEADS}) SELECT lead FROM d",
]

LEFT_ALONE = [
    # a read whose row the outputs do not tell
    f"SELECT DISTINCT m.staff_id, m.label {LEADS}",
    f"SELECT DISTINCT r.lead_id, r.age {LEADS}",
    f"SELECT DISTINCT m.label, r.label {LEADS}",
    "SELECT DISTINCT r.staff_id, m.label FROM staff AS m, staff AS r",
    "SELECT DISTINCT r.staff_id, m.label FROM staff AS m JOIN staff AS r ON m.staff_id >= r.lead_id",
    "SELECT DISTINCT r.staff_id, m.label FROM staff AS m JOIN staff AS r ON m.staff_id = r.lead_id OR m.age = 50",
    "SELECT DISTINCT r.staff_id, m.label FROM staff AS m, staff AS r WHERE m.staff_id = 2 OR r.age = 30",
    "SELECT DISTINCT r.staff_id, m.label FROM staff AS m, staff AS r WHERE m.staff_id = '2'",
    "SELECT DISTINCT r.staff_id, m.label FROM staff AS m, staff AS r WHERE m.staff_id = r.age + 0",
    "SELECT DISTINCT r.staff_id + 0 AS staff_id, m.staff_id AS lead FROM staff AS m, staff AS r",
    # half of a key of two columns
    "SELECT DISTINCT s.staff_id, m.label FROM shifts AS s JOIN staff AS m ON m.staff_id = s.staff_id",
    "SELECT DISTINCT s.day, m.staff_id FROM shifts AS s JOIN staff AS m ON m.staff_id = s.hours",
    # a padded row holds no row of the table
    "SELECT DISTINCT r.staff_id, m.label FROM staff AS r LEFT JOIN staff AS m ON m.staff_id = r.lead_id",
    "SELECT DISTINCT r.staff_id, m.staff_id AS lead FROM staff AS r LEFT JOIN staff AS m ON m.staff_id = r.lead_id",
    "SELECT DISTINCT r.staff_id, m.staff_id AS lead FROM staff AS r RIGHT JOIN staff AS m ON m.staff_id = r.lead_id",
    "SELECT DISTINCT r.staff_id, m.staff_id AS lead FROM staff AS r FULL JOIN staff AS m ON m.staff_id = r.lead_id",
    # USING and NATURAL
    "SELECT DISTINCT s.day, m.label FROM shifts AS s JOIN staff AS m USING (staff_id)",
    "SELECT DISTINCT s.day, m.label FROM shifts AS s NATURAL JOIN staff AS m",
    # a key that is a text or may be NULL, and a text equated with a number
    "SELECT DISTINCT b.code, m.staff_id FROM badges AS b JOIN staff AS m ON m.staff_id = b.staff_id",
    "SELECT DISTINCT n.note_id, m.staff_id FROM notes AS n JOIN staff AS m ON m.staff_id = n.staff_id",
    "SELECT DISTINCT r.staff_id, b.staff_id AS own FROM staff AS r JOIN badges AS b ON b.code = r.label",
    # sources whose rows can repeat or that the declared columns do not place
    "SELECT DISTINCT r.staff_id, d.staff_id AS own FROM staff AS r JOIN (SELECT staff_id FROM shifts) AS d ON d.staff_id = r.staff_id",
    "WITH staff AS (SELECT staff_id, lead_id, label FROM notes) SELECT DISTINCT m.staff_id AS lead, r.staff_id FROM staff AS m JOIN staff AS r ON m.staff_id = r.lead_id",
    "SELECT DISTINCT r.staff_id, e.staff_id AS own FROM staff AS r JOIN elsewhere AS e ON e.staff_id = r.staff_id",
    "SELECT DISTINCT staff_id, label FROM staff AS m JOIN shifts AS s ON m.staff_id = s.hours",
    # not a plain DISTINCT of joined rows
    f"SELECT m.staff_id AS lead, r.staff_id {LEADS}",
    "SELECT DISTINCT staff_id, label FROM staff",
    f"SELECT DISTINCT m.staff_id AS lead, r.staff_id {LEADS} ORDER BY 1 LIMIT 2",
    f"SELECT DISTINCT m.staff_id AS lead, r.staff_id {LEADS} GROUP BY m.staff_id, r.staff_id",
    f"SELECT DISTINCT m.staff_id AS lead, COUNT(*) AS n {LEADS}",
    f"SELECT DISTINCT m.staff_id AS lead, r.staff_id, ROW_NUMBER() OVER (ORDER BY r.staff_id) AS k {LEADS}",
    f"SELECT DISTINCT ON (m.staff_id) m.staff_id AS lead, r.staff_id {LEADS}",
]


def _database(dialect: str):
    database = pytest.importorskip("duckdb").connect() if dialect == "duckdb" else sqlite3.connect(":memory:")
    for name, (columns, rows) in TABLES.items():
        database.execute(f"CREATE TABLE {name} ({columns})")
        database.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(rows[0]))})", rows)
    return database


def _rows(database, sql: str, dialect: str) -> list:
    return sorted(database.execute(sqlglot.transpile(sql, read="mysql", write=dialect)[0]).fetchall(), key=repr)


def _read(sql: str, read: str = "mysql"):
    """The statement with the first select the rule reads written without its DISTINCT, None when it reads none."""

    tree = sqlglot.parse_one(sql, read=read)
    for select in list(tree.find_all(exp.Select)):
        plain = read_keyed_join_distinct(select, SCHEMA, KEYS, NOT_NULL, TYPES)
        if plain is not None:
            if select is tree:
                return plain
            select.replace(plain)
            return tree
    return None


@pytest.mark.parametrize("written", READ)
@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_a_reading_returns_the_rows_of_the_query_as_written(written, dialect):
    read = _read(written)
    assert read is not None and "DISTINCT" not in read.sql(dialect="mysql")
    assert read.sql(dialect="mysql") == sqlglot.parse_one(written.replace("SELECT DISTINCT", "SELECT"), read="mysql").sql(dialect="mysql")
    database = _database(dialect)
    assert _rows(database, written, dialect) == _rows(database, read.sql(dialect="mysql"), dialect)


def test_the_made_up_rows_tell_the_cases_apart():
    database = _database("sqlite")
    assert len(_rows(database, READ[0], "sqlite")) == 4  # ann leads two, bo leads two; nobody has lead 9
    assert len(_rows(database, READ[7], "sqlite")) == 4
    assert len(_rows(database, READ[8], "sqlite")) == 4
    assert _rows(database, READ[10], "sqlite") == [(1, 2), (2, 2)]


@pytest.mark.parametrize("written", LEFT_ALONE)
def test_a_query_outside_the_reading_stays_as_written(written):
    assert _read(written, "postgres" if "DISTINCT ON" in written else "mysql") is None


def test_nothing_is_read_without_declared_keys():
    tree = sqlglot.parse_one(READ[0], read="mysql")
    assert read_keyed_join_distinct(tree, SCHEMA, None, NOT_NULL, TYPES) is None
    assert read_keyed_join_distinct(tree, SCHEMA, KEYS, None, TYPES) is None  # the key may be NULL
    assert read_keyed_join_distinct(tree, SCHEMA, KEYS, NOT_NULL, None) is None  # the key may be a text
    assert read_keyed_join_distinct(tree, None, KEYS, NOT_NULL, TYPES) is None


def test_the_select_it_is_given_is_not_changed():
    tree = sqlglot.parse_one(READ[0], read="mysql")
    before = tree.sql()
    read = read_keyed_join_distinct(tree, SCHEMA, KEYS, NOT_NULL, TYPES)
    assert read is not None and read is not tree and tree.sql() == before


# (a query left alone, rows with its DISTINCT, rows without) on SQLite
BY_HAND = [
    (f"SELECT DISTINCT m.staff_id, m.label {LEADS}", 2, 4),  # a lead of two is listed twice
    (f"SELECT DISTINCT m.label, r.label {LEADS}", 3, 4),  # two reports with one label
    ("SELECT DISTINCT s.staff_id, m.label FROM shifts AS s JOIN staff AS m ON m.staff_id = s.staff_id", 3, 4),  # two days of one person
    ("SELECT DISTINCT n.note_id, m.staff_id FROM notes AS n JOIN staff AS m ON m.staff_id = n.staff_id", 2, 3),  # two notes without an id
    # one label, two badge codes that differ only in case
    ("SELECT DISTINCT r.staff_id, b.staff_id AS own FROM staff AS r JOIN badges AS b ON b.code = r.label COLLATE NOCASE", 1, 2),
    ("SELECT DISTINCT r.staff_id, m.staff_id AS lead FROM staff AS r JOIN (SELECT staff_id FROM shifts) AS m ON m.staff_id = r.staff_id", 3, 4),
]


@pytest.mark.parametrize("written, with_distinct, without", BY_HAND)
def test_outside_the_reading_the_distinct_removes_rows(written, with_distinct, without):
    database = _database("sqlite")
    assert len(database.execute(written).fetchall()) == with_distinct
    assert len(database.execute(written.replace("SELECT DISTINCT", "SELECT")).fetchall()) == without
    assert _read(written) is None


# reports per lead with the lead's label: a grouped query over the join, and over a DISTINCT list of the joined rows
OPTIONS = dict(schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect="mysql")
JOINED = f"SELECT m.staff_id, m.label, COUNT(*) AS reports, MAX(r.age) AS oldest {LEADS} GROUP BY m.staff_id"
LISTED = (
    "WITH pairs AS (SELECT DISTINCT m.staff_id AS lead, r.staff_id AS report, m.label AS lead_label, r.age AS report_age {})"
    " SELECT lead, lead_label, COUNT({}) AS reports, MAX(report_age) AS oldest FROM pairs GROUP BY lead"
)


def test_a_pair_is_proved_and_only_with_the_rule():
    assert prove_equivalent_algebraic(JOINED, LISTED.format(LEADS, "*"), **OPTIONS).proven
    assert prove_equivalent_algebraic(JOINED, LISTED.format(LEADS, "DISTINCT report"), **OPTIONS).proven
    with shape_fallback.without("keyed_join_distinct"):
        assert not prove_equivalent_algebraic(JOINED, LISTED.format(LEADS, "*"), **OPTIONS).proven


def test_pairs_that_differ_stay_unproved():
    # the list drops the report's key: two reports of one age are one row
    loose = LISTED.replace("r.staff_id AS report, ", "").format(LEADS, "*")
    assert not prove_equivalent_algebraic(JOINED, loose, **OPTIONS).proven
    # a lead with no reports is listed by the outer join
    assert not prove_equivalent_algebraic(JOINED, LISTED.format("FROM staff AS m LEFT JOIN staff AS r ON m.staff_id = r.lead_id", "*"), **OPTIONS).proven
    # without the declared key the list may repeat
    bare = {**CONSTRAINTS, "staff": TableConstraints()}
    assert not prove_equivalent_algebraic(JOINED, LISTED.format(LEADS, "*"), **{**OPTIONS, "constraints": bare}).proven
