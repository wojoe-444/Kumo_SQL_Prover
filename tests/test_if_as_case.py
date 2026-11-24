"""``IF(test, a, b)`` read as a ``CASE`` before the pre-passes (``if_as_case``).

Each reading is run as written and as read on DuckDB and on SQLite over made-up rows with a NULL in the tested
column, so the third argument is taken for a false test and for one that is NULL.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.if_as_case import read_if_as_case
from kumo_sql_prover.smt_equivalence import TableConstraints

TABLES = {
    "orders": ("id INT, shop INT, day INT", [(1, 1, 1), (2, 1, 2), (3, 2, 2), (4, None, 3)]),
    "order_lines": ("order_id INT, sku INT, qty INT, note VARCHAR", [(1, 10, 5, "a"), (1, 11, 1, None), (2, 10, None, "b"), (3, 12, 2, "c"), (3, 13, 0, "d")]),
}
SCHEMA = {name: [part.split()[0] for part in columns.split(", ")] for name, (columns, _) in TABLES.items()}
TYPES = {name: dict(part.split() for part in columns.split(", ")) for name, (columns, _) in TABLES.items()}
CONSTRAINTS = {
    "orders": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
    "order_lines": TableConstraints(not_null=frozenset({"order_id", "sku"}), keys=(("order_id", "sku"),)),
}

# (as written in MySQL, the number of CASE expressions in the reading)
READ = [
    ("SELECT order_id, SUM(IF(qty > 1, 1, 0)) AS big FROM order_lines GROUP BY order_id", 1),
    ("SELECT order_id, COUNT(IF(qty > 1, 1, NULL)) AS big, SUM(IF(note IS NULL, qty, 0)) AS unnoted FROM order_lines GROUP BY order_id", 2),
    ("SELECT sku, IF(qty > 1, 'many', IF(qty = 1, 'one', 'none')) AS size FROM order_lines", 2),
    ("SELECT sku FROM order_lines WHERE IF(qty IS NULL, 0, qty) < 2 ORDER BY IF(note = 'd', 0, 1), sku", 2),
    # an IF inside a CASE arm, and a CASE inside an IF
    ("SELECT sku, CASE WHEN note IS NULL THEN IF(qty > 1, 1, 0) ELSE 2 END AS x, IF(CASE WHEN qty > 1 THEN TRUE ELSE FALSE END, 1, 0) AS y FROM order_lines", 4),
    ("SELECT o.id, (SELECT SUM(IF(l.qty > 1, 1, 0)) FROM order_lines AS l WHERE l.order_id = o.id) AS big FROM orders AS o", 1),
]

LEFT_ALONE = [
    "SELECT order_id, SUM(CASE WHEN qty > 1 THEN 1 ELSE 0 END) AS big FROM order_lines GROUP BY order_id",
    "SELECT sku, CASE WHEN qty > 1 THEN 'many' WHEN qty = 1 THEN 'one' ELSE 'none' END AS size FROM order_lines",
    "SELECT sku, IFNULL(qty, 0) AS qty, NULLIF(qty, 0) AS q FROM order_lines",
]


def _rows(database, sql: str, dialect: str) -> list:
    return sorted(database.execute(sqlglot.transpile(sql, read="mysql", write=dialect)[0]).fetchall(), key=repr)


def _databases():
    duckdb = pytest.importorskip("duckdb")
    for dialect, database in (("duckdb", duckdb.connect()), ("sqlite", sqlite3.connect(":memory:"))):
        for name, (columns, rows) in TABLES.items():
            database.execute(f"CREATE TABLE {name} ({columns})")
            database.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(rows[0]))})", rows)
        yield dialect, database


def _read(sql: str) -> tuple:
    return read_if_as_case(sqlglot.parse_one(sql, read="mysql"))


@pytest.mark.parametrize("written, cases", READ)
def test_a_reading_returns_the_rows_of_the_query_as_written(written, cases):
    tree, wrote = _read(written)
    read = tree.sql(dialect="mysql")
    assert wrote and "IF(" not in read.replace("NULLIF(", "") and len(list(tree.find_all(sqlglot.exp.Case))) == cases, read
    for dialect, database in _databases():
        assert _rows(database, written, dialect) == _rows(database, read, dialect), dialect


@pytest.mark.parametrize("written", LEFT_ALONE)
def test_a_query_without_an_if_call_stays_as_written(written):
    given = sqlglot.parse_one(written, read="mysql")
    tree, wrote = read_if_as_case(given)
    assert not wrote and tree is given


def test_the_tree_it_is_given_is_not_changed():
    tree = sqlglot.parse_one(READ[0][0], read="mysql")
    before = tree.sql()
    read, wrote = read_if_as_case(tree)
    assert wrote and read is not tree and tree.sql() == before


def test_a_null_test_takes_the_third_argument():
    database = next(db for dialect, db in _databases() if dialect == "duckdb")
    read = _read("SELECT sku, IF(qty > 1, 'many', 'few') AS size FROM order_lines")[0].sql(dialect="mysql")
    assert (10, "few") in [row for row in _rows(database, read, "duckdb") if row[0] == 10]  # order 2: qty is NULL


def _proved(left: str, right: str) -> bool:
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect="mysql").proven


FLAT = "SELECT o.id, o.day, COUNT(l.sku) AS n, SUM({}) AS big FROM orders AS o LEFT JOIN order_lines AS l ON l.order_id = o.id GROUP BY o.id, o.day"
COUNTED = "SELECT o.id, o.day, IFNULL(c.n, 0) AS n, IFNULL(c.big, 0) AS big FROM orders AS o LEFT JOIN (SELECT order_id, COUNT(sku) AS n, SUM({}) AS big FROM order_lines GROUP BY order_id) AS c ON c.order_id = o.id"
PAIRS = [
    (COUNTED.format("CASE WHEN qty > 1 THEN 1 ELSE 0 END"), FLAT.format("IF(l.qty > 1, 1, 0)")),
    (COUNTED.format("IF(qty > 1, 1, 0)"), FLAT.format("CASE WHEN l.qty > 1 THEN 1 ELSE 0 END")),
]


def test_a_pair_is_proved_and_only_with_the_rule():
    # IF on the flat side: the reading of counts over a joined child table looks for the CASE spelling
    left, right = PAIRS[0]
    assert _proved(left, right)
    with shape_fallback.without("if_as_case"):
        assert not _proved(left, right)


def test_the_pair_with_the_spellings_swapped_is_proved():
    # IF inside the counted table: an earlier batch reads it there, with the rule or without
    assert _proved(*PAIRS[1])


def test_pairs_that_differ_stay_unproved():
    # the arms the other way round, and a flag that is NULL where the other is 0
    assert not _proved(PAIRS[0][0], FLAT.format("IF(l.qty > 1, 0, 1)"))
    assert not _proved(PAIRS[0][0], FLAT.format("IF(l.qty > 1, 1, NULL)").replace("SUM(", "MAX("))
