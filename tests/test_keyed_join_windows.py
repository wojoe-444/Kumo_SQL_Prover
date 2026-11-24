"""A whole-partition aggregate beside a join to one keyed row (``keyed_join_windows``).

Each reading is run as written and as read on DuckDB and on SQLite over made-up tables: an item with three orders,
items with one and with none, an order with no item and one whose item does not exist, orders with no shop. The
rows must be the same. Queries the rule leaves alone come, where a reading by hand would change rows, with those
rows.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.keyed_join_windows import read_keyed_join_windows
from kumo_sql_prover.smt_equivalence import TableConstraints

TABLES = {
    "orders": (
        "id INT, item INT, qty INT, price INT, shop INT, day INT, note VARCHAR",
        [(1, 10, 5, 100, 1, 1, "saw"), (2, 10, 1, 50, 1, 2, "saw"), (3, 10, 9, 70, 2, 2, "kite"), (4, 11, 4, 20, None, 1, "kite"), (5, 12, 3, 30, 2, 3, "file"), (6, None, 7, 10, 1, 3, None), (7, 99, 2, 5, 1, 1, "none"), (8, 12, 6, 15, None, 3, "file")],
    ),
    "items": ("id INT, kind VARCHAR, size INT, name VARCHAR", [(10, "tool", 1, "saw"), (11, "toy", 2, "kite"), (12, "tool", 3, "file"), (13, "tool", 1, "axe")]),
    "shops": ("id INT, city VARCHAR", [(1, "x"), (2, "y")]),
}
SCHEMA = {name: [part.split()[0] for part in columns.split(", ")] for name, (columns, _) in TABLES.items()}
TYPES = {name: dict(part.split() for part in columns.split(", ")) for name, (columns, _) in TABLES.items()}
KEYS = {"orders": [("id",)], "items": [("id",), ("name",)], "shops": [("id",)]}
NOT_NULL = {"orders": frozenset({"id", "day"}), "items": frozenset({"id", "name"}), "shops": frozenset()}
CONSTRAINTS = {name: TableConstraints(not_null=NOT_NULL[name], keys=tuple(KEYS[name])) for name in TABLES}

JOINED = "FROM orders AS o JOIN items AS i ON o.item = i.id"
USUAL = "AVG(o.qty) OVER (PARTITION BY o.item) AS usual"

READ = [
    f"SELECT o.id, o.qty, {USUAL} {JOINED} WHERE i.kind = 'tool'",
    f"SELECT o.id, {USUAL} {JOINED}",
    f"SELECT o.id, {USUAL}, i.name {JOINED} WHERE i.kind IN ('tool', 'toy') AND (i.size > 1 OR i.name = 'saw')",
    # the keyed source is a list of the keyed table's rows
    "SELECT qty, 2 * MAX(qty) OVER (PARTITION BY item) AS top FROM orders JOIN (SELECT id AS k FROM items WHERE kind = 'tool') AS f ON item = f.k",
    "SELECT o.id, COUNT(*) OVER (PARTITION BY f.id) AS n FROM (SELECT DISTINCT id, size FROM items) AS f, orders AS o WHERE o.item = f.id AND f.size < 3",
    # a comma, the partition named by the key, two aggregates, the keyed table first
    "SELECT o.qty, COUNT(*) OVER (PARTITION BY i.id) AS n, MAX(o.qty) OVER (PARTITION BY o.item) AS top FROM items AS i, orders AS o WHERE i.id = o.item AND i.name <> 'saw'",
    "SELECT o.id, SUM(o.qty) OVER (PARTITION BY o.item, i.id) AS s, MIN(o.price) OVER (PARTITION BY i.id, o.item) AS low FROM orders AS o CROSS JOIN items AS i WHERE o.item = i.id",
    # a finer partition by a column that is never NULL
    f"SELECT o.id, SUM(o.qty) OVER (PARTITION BY o.item, o.day) AS s {JOINED} WHERE i.kind = 'tool'",
    # DISTINCT, ORDER BY and LIMIT apply to the same rows
    f"SELECT DISTINCT o.item, MIN(o.qty) OVER (PARTITION BY o.item) AS low {JOINED} WHERE i.size < 3 ORDER BY o.item LIMIT 5",
    # bare names that only one source has
    "SELECT qty, AVG(qty) OVER (PARTITION BY item) AS usual, COUNT(note) OVER (PARTITION BY item) AS notes FROM orders JOIN items ON item = items.id WHERE kind = 'tool'",
    # inside a derived table
    f"SELECT SUM(x.price) AS total FROM (SELECT o.price, o.qty, 0.5 * {USUAL} {JOINED} WHERE i.kind = 'tool') AS x WHERE x.qty < x.usual",
]

LEFT_ALONE = [
    # a test of the orders: the partition is some of the item's orders
    f"SELECT o.id, {USUAL} {JOINED} WHERE i.kind = 'tool' AND o.qty > 1",
    f"SELECT o.id, {USUAL} FROM orders AS o JOIN items AS i ON o.item = i.id AND o.day < i.size",
    # the join is not by a key: an order appears once per item of its shop's number
    "SELECT o.id, COUNT(*) OVER (PARTITION BY o.shop) AS n FROM orders AS o JOIN items AS i ON o.shop = i.size",
    # a partition column that may be NULL and is not the joined one
    f"SELECT o.id, COUNT(*) OVER (PARTITION BY o.item, o.shop) AS n {JOINED}",
    # the partition does not hold the joined column, or is by another column of the keyed table
    f"SELECT o.id, COUNT(*) OVER (PARTITION BY o.day) AS n {JOINED}",
    f"SELECT o.id, COUNT(*) OVER (PARTITION BY i.kind) AS n {JOINED}",
    f"SELECT o.id, COUNT(*) OVER () AS n {JOINED}",
    # a text key: the comparison and the grouping may use different collations
    "SELECT o.id, COUNT(*) OVER (PARTITION BY o.note) AS n FROM orders AS o JOIN items AS i ON o.note = i.name",
    # a key that is not declared NOT NULL
    "SELECT o.id, COUNT(*) OVER (PARTITION BY o.shop) AS n FROM orders AS o JOIN shops AS s ON o.shop = s.id",
    # joins that are not inner joins of two sources
    f"SELECT o.id, {USUAL} FROM orders AS o LEFT JOIN items AS i ON o.item = i.id",
    f"SELECT o.id, {USUAL} FROM items AS i RIGHT JOIN orders AS o ON o.item = i.id",
    f"SELECT o.id, {USUAL} {JOINED} JOIN shops AS s ON s.id = o.shop",
    "SELECT o.id, COUNT(*) OVER (PARTITION BY o.id) AS n FROM orders AS o JOIN orders AS q USING (id)",
    f"SELECT o.id, {USUAL} FROM orders AS o",
    # windows that are not a plain aggregate of the whole partition
    f"SELECT o.id, AVG(o.qty) OVER (PARTITION BY o.item ORDER BY o.id) AS usual {JOINED}",
    f"SELECT o.id, SUM(o.qty) OVER (PARTITION BY o.item ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s {JOINED}",
    f"SELECT o.id, COUNT(DISTINCT o.qty) OVER (PARTITION BY o.item) AS n {JOINED}",
    f"SELECT o.id, AVG(o.qty + 1) OVER (PARTITION BY o.item) AS usual {JOINED}",
    f"SELECT o.id, MAX(i.size) OVER (PARTITION BY o.item) AS s {JOINED}",
    f"SELECT o.id, ROW_NUMBER() OVER (PARTITION BY o.item) AS rn {JOINED}",
    f"SELECT o.id, SUM(o.qty) FILTER (WHERE o.qty > 1) OVER (PARTITION BY o.item) AS s {JOINED}",
    f"SELECT o.id, {USUAL}, COUNT(*) OVER (PARTITION BY o.item, o.day) AS n {JOINED}",
    f"SELECT o.id, AVG(o.qty) OVER w AS usual {JOINED} WINDOW w AS (PARTITION BY o.item)",
    # the rest of the select
    f"SELECT o.item, SUM(SUM(o.qty)) OVER (PARTITION BY o.item) AS s {JOINED} GROUP BY o.item",
    f"SELECT o.id {JOINED} ORDER BY AVG(o.qty) OVER (PARTITION BY o.item)",
    f"SELECT o.*, {USUAL} {JOINED}",
    f"SELECT *, {USUAL} {JOINED}",
    f"SELECT o.id, {USUAL}, (SELECT MAX(q.qty) FROM orders AS q WHERE q.item = o.item) AS top {JOINED}",
    f"SELECT o.id, {USUAL} {JOINED} WHERE i.size IN (SELECT s.id FROM shops AS s)",
    # a name both sources have, and a name of neither
    "SELECT id, AVG(qty) OVER (PARTITION BY item) AS usual FROM orders JOIN items ON item = items.id",
    f"SELECT o.id, {USUAL} {JOINED} WHERE x.kind = 'tool'",
    # a keyed source that is not a list of the keyed table's rows
    "SELECT o.id, COUNT(*) OVER (PARTITION BY o.item) AS n FROM orders AS o JOIN (SELECT id + 0 AS id FROM items) AS f ON o.item = f.id",
    "SELECT o.id, COUNT(*) OVER (PARTITION BY o.item) AS n FROM orders AS o JOIN (SELECT size AS id FROM items) AS f ON o.item = f.id",
    "SELECT o.id, COUNT(*) OVER (PARTITION BY o.item) AS n FROM orders AS o JOIN (SELECT i.id FROM items AS i JOIN shops AS s ON s.id = i.size) AS f ON o.item = f.id",
    "SELECT o.id, COUNT(*) OVER (PARTITION BY o.item) AS n FROM orders AS o JOIN (SELECT id FROM items LIMIT 2) AS f ON o.item = f.id",
]

# (as written, read by hand the way the rule would): the rows differ, so the refusal is needed
BY_HAND = [
    (
        f"SELECT o.id, COUNT(*) OVER (PARTITION BY o.item) AS n {JOINED} WHERE o.qty > 1",
        f"SELECT o.id, g.n {JOINED} JOIN (SELECT item, COUNT(*) AS n FROM orders GROUP BY item) AS g ON o.item = g.item WHERE o.qty > 1",
    ),
    (
        "SELECT o.id, COUNT(*) OVER (PARTITION BY o.shop) AS n FROM orders AS o JOIN items AS i ON o.shop = i.size",
        "SELECT o.id, g.n FROM orders AS o JOIN items AS i ON o.shop = i.size JOIN (SELECT shop, COUNT(*) AS n FROM orders GROUP BY shop) AS g ON o.shop = g.shop",
    ),
    (
        f"SELECT o.id, COUNT(*) OVER (PARTITION BY o.item, o.shop) AS n {JOINED}",
        f"SELECT o.id, g.n {JOINED} JOIN (SELECT item, shop, COUNT(*) AS n FROM orders GROUP BY item, shop) AS g ON o.item = g.item AND o.shop = g.shop",
    ),
    (
        f"SELECT o.id, COUNT(*) OVER (PARTITION BY o.day) AS n {JOINED}",
        f"SELECT o.id, g.n {JOINED} JOIN (SELECT day, COUNT(*) AS n FROM orders GROUP BY day) AS g ON o.day = g.day",
    ),
]


def _rows(database, sql: str, dialect: str) -> list:
    return sorted(database.execute(sqlglot.transpile(sql, read="postgres", write=dialect)[0]).fetchall(), key=repr)


def _databases():
    duckdb = pytest.importorskip("duckdb")
    for dialect, database in (("duckdb", duckdb.connect()), ("sqlite", sqlite3.connect(":memory:"))):
        for name, (columns, rows) in TABLES.items():
            database.execute(f"CREATE TABLE {name} ({columns})")
            database.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(rows[0]))})", rows)
        yield dialect, database


def _read(sql: str, **changed) -> tuple[str, bool]:
    facts = {"schema": SCHEMA, "keys": KEYS, "not_null": NOT_NULL, "types": TYPES, **changed}
    tree, wrote = read_keyed_join_windows(sqlglot.parse_one(sql, read="postgres"), facts["schema"], facts["keys"], facts["not_null"], facts["types"])
    return tree.sql(dialect="postgres"), wrote


@pytest.mark.parametrize("written", READ)
def test_a_reading_returns_the_rows_of_the_query_as_written(written):
    read, wrote = _read(written)
    assert wrote and " OVER " not in read, read
    for dialect, database in _databases():
        rows = _rows(database, written, dialect)
        assert rows == _rows(database, read, dialect), dialect
        assert rows and rows != [(None,)], "the made-up rows do not reach this query"


@pytest.mark.parametrize("written", LEFT_ALONE)
def test_a_query_that_is_not_read_stays_as_written(written):
    read, wrote = _read(written)
    assert not wrote and read == sqlglot.parse_one(written, read="postgres").sql(dialect="postgres")


@pytest.mark.parametrize("written, by_hand", BY_HAND)
def test_reading_a_refused_query_by_hand_changes_rows(written, by_hand):
    assert not _read(written)[1]
    database = next(db for dialect, db in _databases() if dialect == "duckdb")
    assert _rows(database, written, "duckdb") != _rows(database, by_hand, "duckdb")


def test_the_reading_needs_the_declared_facts():
    written = READ[0]
    assert _read(written)[1]
    assert not _read(written, types=None)[1]
    assert not _read(written, keys={"orders": [("id",)]})[1]
    assert not _read(written, not_null={"orders": frozenset({"id"})})[1]
    assert not _read(written, types={**TYPES, "items": {**TYPES["items"], "id": "VARCHAR"}})[1]
    assert not _read(written, schema={"orders": SCHEMA["orders"]})[1]
    # a WITH name hides the declared table: its rows are not the declared ones
    assert not _read(f"WITH items AS (SELECT item AS id, 'tool' AS kind FROM orders) {written}")[1]


def test_the_tree_it_is_given_is_not_changed():
    tree = sqlglot.parse_one(READ[0], read="postgres")
    before = tree.sql()
    read, wrote = read_keyed_join_windows(tree, SCHEMA, KEYS, NOT_NULL, TYPES)
    assert wrote and read is not tree and tree.sql() == before


def _proved(left: str, right: str) -> bool:
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect="postgres").proven


BELOW_USUAL = "SELECT SUM(o.price) AS total FROM orders AS o, items AS i WHERE i.id = o.item AND i.kind = 'tool' AND o.qty < (SELECT 0.5 * AVG(q.qty) FROM orders AS q WHERE q.item = i.id)"
PAIRS = [
    (
        f"SELECT o.id, {USUAL} {JOINED} WHERE i.kind = 'tool'",
        f"SELECT o.id, g.usual {JOINED} JOIN (SELECT item, AVG(qty) AS usual FROM orders GROUP BY item) AS g ON g.item = o.item WHERE i.kind = 'tool'",
    ),
    (
        BELOW_USUAL,
        "WITH tools AS (SELECT id FROM items WHERE kind = 'tool'), "
        "x AS (SELECT o.price, o.qty, 0.5 * AVG(o.qty) OVER (PARTITION BY o.item) AS t FROM orders AS o JOIN tools ON o.item = tools.id) "
        "SELECT SUM(price) AS total FROM x WHERE qty < t",
    ),
]


@pytest.mark.parametrize("left, right", PAIRS)
def test_a_pair_is_proved_and_only_with_the_rule(left, right):
    assert _proved(left, right)
    with shape_fallback.without("keyed_join_windows"):
        assert not _proved(left, right)


def test_pairs_that_differ_stay_unproved():
    # the mean of the orders above one, not of all the item's orders
    assert not _proved(BELOW_USUAL, PAIRS[1][1].replace("ON o.item = tools.id", "ON o.item = tools.id WHERE o.qty > 1"))
    # the mean of the day's orders
    assert not _proved(BELOW_USUAL, PAIRS[1][1].replace("PARTITION BY o.item", "PARTITION BY o.item, o.day"))
