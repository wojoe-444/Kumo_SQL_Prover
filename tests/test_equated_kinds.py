"""A text set equal to a number is not one shared value: the algebraic prover declines such a pair (issue 71).

``'2'`` and ``'02'`` are two texts and both equal the number 2 in MySQL, SQLite and DuckDB. A per-row subquery over
``b.code = m.staff_id`` adds both up; a join to the table grouped by ``code`` returns two rows. The pair was proved.
"""

from __future__ import annotations

import sqlite3

import pytest

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtStatus, TableConstraints, prove_equivalent_smt
from kumo_sql_prover.string_number_compare import EQUATED, equated_kinds_problem

SCHEMA = {"staff": ["staff_id", "label", "badge"], "badges": ["code", "hours", "day"], "days": ["day", "note"]}
TYPES = {
    "staff": {"staff_id": "INT", "label": "VARCHAR", "badge": "VARCHAR"},
    "badges": {"code": "VARCHAR", "hours": "INT", "day": "VARCHAR"},
    "days": {"day": "DATE", "note": "VARCHAR"},
}
CONSTRAINTS = {
    "staff": TableConstraints(not_null=frozenset({"staff_id"}), keys=(("staff_id",),)),
    "badges": TableConstraints(not_null=frozenset({"code"}), keys=(("code",),)),
}
ROWS = {
    "staff": ("staff_id INT, label VARCHAR, badge VARCHAR", [(2, "bo", "2"), (3, "cy", "3")]),
    "badges": ("code VARCHAR, hours INT, day VARCHAR", [("2", 1, "2024-03-01"), ("02", 2, "2024-3-1"), ("3", 5, "2024-03-02")]),
    "days": ("day DATE, note VARCHAR", [("2024-03-01", "a")]),
}
PER_CODE = "(SELECT code, SUM(hours) AS total FROM badges GROUP BY code) AS g"
PER_ROW = "SELECT m.staff_id, (SELECT SUM(b.hours) FROM badges AS b WHERE b.code = m.{column}) AS total FROM staff AS m"
JOINED = "SELECT m.staff_id, g.total FROM staff AS m LEFT JOIN " + PER_CODE + " ON g.code = m.{column}"
DIALECTS = ["mysql", "sqlite", "duckdb", "postgres"]


def _rows(sql: str) -> list:
    database = sqlite3.connect(":memory:")
    for name, (columns, rows) in ROWS.items():
        database.execute(f"CREATE TABLE {name} ({columns})")
        database.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(rows[0]))})", rows)
    return sorted(database.execute(sql).fetchall(), key=repr)


def test_two_texts_equal_one_number_on_sqlite():
    assert _rows(PER_ROW.format(column="staff_id")) == [(2, 3), (3, 5)]
    assert _rows(JOINED.format(column="staff_id")) == [(2, 1), (2, 2), (3, 5)]
    # between two texts the comparison is of one value, and the two spellings agree
    assert _rows(PER_ROW.format(column="badge")) == _rows(JOINED.format(column="badge")) == [(2, 1), (3, 5)]


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_text_equated_with_a_number_is_not_proved(dialect):
    options = dict(schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, dialect=dialect)
    result = prove_equivalent_algebraic(PER_ROW.format(column="staff_id"), JOINED.format(column="staff_id"), **options)
    assert result.status is SmtStatus.NOT_PROVEN and EQUATED in result.reason
    assert prove_equivalent_smt(PER_ROW.format(column="staff_id"), JOINED.format(column="staff_id"), **options).status is not SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize("dialect", DIALECTS)
def test_two_texts_equated_are_still_proved(dialect):
    options = dict(schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, dialect=dialect)
    assert prove_equivalent_algebraic(PER_ROW.format(column="badge"), JOINED.format(column="badge"), **options).status is SmtStatus.PROVEN_EQUIVALENT


FOUND = [
    "SELECT m.staff_id FROM staff AS m JOIN badges AS b ON b.code = m.staff_id",
    "SELECT m.staff_id FROM staff AS m JOIN badges AS b ON m.staff_id <=> b.code",
    "SELECT m.staff_id FROM staff AS m, badges AS b WHERE (b.code) = (m.staff_id + 1)",
    "SELECT m.staff_id FROM staff AS m WHERE m.staff_id IN (SELECT code FROM badges)",
    "SELECT m.staff_id FROM staff AS m WHERE EXISTS (SELECT 1 FROM badges AS b WHERE b.code = m.staff_id)",
    "SELECT b.code FROM badges AS b JOIN days AS d ON d.day = b.day",  # a text and a date
    "SELECT b.code FROM badges AS b JOIN days AS d USING (day)",
    "SELECT g.code FROM (SELECT code FROM badges) AS g JOIN staff AS m ON g.code = m.staff_id",
    "WITH g AS (SELECT code FROM badges) SELECT g.code FROM g JOIN staff AS m ON g.code = m.staff_id",
    "SELECT m.staff_id FROM staff AS m WHERE EXISTS (SELECT 1 FROM badges WHERE code = staff_id)",  # bare names of two tables
    "SELECT m.staff_id FROM staff AS m, badges AS b WHERE code = staff_id",
]
NOT_FOUND = [
    "SELECT m.staff_id FROM staff AS m JOIN badges AS b ON b.code = m.badge",
    "SELECT m.staff_id FROM staff AS m JOIN badges AS b ON b.hours = m.staff_id",
    "SELECT m.staff_id FROM staff AS m WHERE m.staff_id = '2'",  # a literal is one value: string_number_compare.problem reads it
    "SELECT b.code FROM badges AS b WHERE b.code = 2",
    "SELECT m.staff_id FROM staff AS m JOIN badges AS b ON b.code < m.staff_id",  # no rule reads an ordering as one value
    "SELECT m.staff_id FROM staff AS m JOIN other AS o ON o.code = m.staff_id",  # no declared type
    "SELECT m.staff_id FROM staff AS m JOIN staff AS r USING (staff_id)",
    # two values of one row: a test on that row, whatever the engine converts
    "SELECT b.code FROM badges AS b WHERE b.hours + 1 = b.code",
    "SELECT code FROM badges WHERE hours = code",
    "SELECT b.code FROM badges AS b GROUP BY b.code HAVING b.code = COUNT(*)",
    "SELECT d.note FROM days AS d JOIN days AS e ON d.day = e.day",
]


@pytest.mark.parametrize("sql", FOUND)
def test_the_equalities_it_names(sql):
    assert equated_kinds_problem(sql, "mysql", TYPES) == EQUATED


@pytest.mark.parametrize("sql", NOT_FOUND)
def test_the_comparisons_it_leaves(sql):
    assert equated_kinds_problem(sql, "mysql", TYPES) is None


def test_nothing_is_named_without_declared_types_or_for_text_that_does_not_parse():
    assert equated_kinds_problem(FOUND[0], "mysql", None) is None
    assert equated_kinds_problem(FOUND[0], "mysql", {}) is None
    assert equated_kinds_problem("SELECT FROM WHERE", "mysql", TYPES) is None


def test_a_name_that_is_a_text_in_one_table_and_a_date_in_another_is_read_by_its_table():
    assert equated_kinds_problem("SELECT 1 FROM badges AS b JOIN days AS d ON b.day = d.day", "mysql", TYPES) == EQUATED
    assert equated_kinds_problem("SELECT 1 FROM badges AS b JOIN badges AS c ON b.day = c.day", "mysql", TYPES) is None
    assert equated_kinds_problem("SELECT 1 FROM badges AS b JOIN staff AS m ON day = label", "mysql", TYPES) is None  # bare, two kinds: not known
