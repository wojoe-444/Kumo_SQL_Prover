"""A subquery whose derived table is called like the outer source its bare column reads.

    SELECT name FROM unit WHERE EXISTS (SELECT 1 FROM (SELECT 2 * no AS f FROM unit) AS unit WHERE no = unit.f)

The derived table has one column, ``f``, so the bare ``no`` of the subquery's ``WHERE`` is the outer row's.
Merging the derived table into the subquery would put the inner ``unit`` (which has a ``no``) in its place and
capture the column, so the prover declined the query (KumoSQL #370: "a subquery reuses the name"). Since batch 25
``apart_alias_names`` gives the derived table and the source inside it a name each before the merge, and the query is
read as it is written. These tests hold the reading to the rows SQLite returns: each query is proved equal to the
spelling with the same rows and never to the one with other rows, whichever of the two the column belongs to.
"""

from __future__ import annotations

import sqlite3

import pytest

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic

SCHEMA = {"staff": ["id", "name", "pay", "no"], "unit": ["no", "name"]}
DDL = ("CREATE TABLE staff (id INT, name TEXT, pay INT, no INT)", "CREATE TABLE unit (no INT, name TEXT)")
ROWS = {
    "unit": [(0, "a"), (1, "b"), (2, "a"), (4, "c"), (None, "d")],
    "staff": [(1, "e", 2, 1), (2, "f", 4, 4), (3, "g", None, 2), (4, "h", 1, None), (5, "i", 0, 0)],
}
SPELLED = "SELECT o.name FROM unit AS o WHERE EXISTS (SELECT 1 FROM {source} AS t WHERE {test})"

# (name, as written, the same rows, other rows, whether the prover reads it)
CASES = [
    (
        "the bare column is the outer row's",
        "SELECT name FROM unit WHERE EXISTS (SELECT 1 FROM (SELECT 2 * no AS f FROM unit) AS unit WHERE no = unit.f)",
        SPELLED.format(source="unit", test="o.no = 2 * t.no"),
        SPELLED.format(source="unit", test="t.no = 2 * t.no"),
        True,
    ),
    (
        "the derived table has the column, so it is the inner row's",
        "SELECT name FROM unit WHERE EXISTS (SELECT 1 FROM (SELECT 2 * no AS f, no FROM unit) AS unit WHERE no = unit.f)",
        SPELLED.format(source="unit", test="t.no = 2 * t.no"),
        SPELLED.format(source="unit", test="o.no = 2 * t.no"),
        True,
    ),
    (
        "a second outer column beside it",
        "SELECT name FROM unit WHERE EXISTS (SELECT 1 FROM (SELECT 2 * no AS f FROM unit) AS unit WHERE no = unit.f AND name = 'a')",
        SPELLED.format(source="unit", test="o.no = 2 * t.no AND o.name = 'a'"),
        SPELLED.format(source="unit", test="o.no = 2 * t.no AND t.name = 'a'"),
        True,
    ),
    (
        "two derived tables of the name, one in the other",
        "SELECT name FROM unit WHERE EXISTS (SELECT 1 FROM (SELECT f FROM (SELECT 2 * no AS f FROM unit) AS unit) AS unit WHERE no = unit.f)",
        SPELLED.format(source="unit", test="o.no = 2 * t.no"),
        SPELLED.format(source="unit", test="t.no = 2 * t.no"),
        True,
    ),
    (
        "another table under the outer name, with a star",
        "SELECT name FROM unit WHERE EXISTS (SELECT 1 FROM (SELECT * FROM staff) AS unit WHERE no = unit.pay)",
        SPELLED.format(source="staff", test="t.no = t.pay"),
        SPELLED.format(source="staff", test="o.no = t.pay"),
        True,
    ),
    (
        "another table's column under the outer name",
        "SELECT name FROM unit WHERE EXISTS (SELECT 1 FROM (SELECT pay AS f FROM staff) AS unit WHERE no = unit.f)",
        SPELLED.format(source="staff", test="o.no = t.pay"),
        SPELLED.format(source="staff", test="t.no = t.pay"),
        False,  # still declined: a subquery reuses the name
    ),
]
IDS = [case[0] for case in CASES]


def _rows(sql: str) -> list:
    con = sqlite3.connect(":memory:")
    for statement in DDL:
        con.execute(statement)
    for table, rows in ROWS.items():
        con.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * len(rows[0]))})", rows)
    return sorted(con.execute(sql).fetchall())


def _proven(left: str, right: str, dialect: str) -> bool:
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, dialect=dialect, compare_names=False).proven


@pytest.mark.parametrize("name, written, same, other, read", CASES, ids=IDS)
def test_sqlite_returns_the_rows_of_one_spelling_and_not_of_the_other(name, written, same, other, read):
    assert _rows(written) == _rows(same) != _rows(other)


@pytest.mark.parametrize("name, written, same, other, read", CASES, ids=IDS)
@pytest.mark.parametrize("dialect", ["mysql", "sqlite"])
def test_the_query_is_proved_equal_to_the_spelling_with_its_rows_only(name, written, same, other, read, dialect):
    assert _proven(written, same, dialect) is read
    assert not _proven(written, other, dialect)


def test_without_the_names_apart_the_query_is_declined_as_before():
    _, written, same, _, _ = CASES[0]
    with shape_fallback.without("apart_alias_names"):
        result = prove_equivalent_algebraic(written, same, schema=SCHEMA, dialect="mysql", compare_names=False)
    assert not result.proven and "reuses the name" in result.reason
