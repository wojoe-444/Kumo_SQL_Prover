"""A grouped derived table's join to a keyed row that the select around it joins again (``rejoined_parents``).

Each reading is run as written and as read on DuckDB and on SQLite over made-up tables: an item with three orders,
items with one and with none, an order with no item and one whose item does not exist. The rows must be the same.
Queries the rule leaves alone come, where a reading by hand would change rows, with those rows.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.rejoined_parents import drop_rejoined_parent
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

TOOLS = "i2.kind = 'tool'"
G = f"(SELECT i2.id AS item, AVG(o2.qty) AS usual, COUNT(*) AS n FROM items AS i2, orders AS o2 WHERE o2.item = i2.id AND {TOOLS} GROUP BY i2.id) AS g"
OUTER = "SELECT o.id, g.usual, g.n FROM orders AS o, items AS i,"

READ = [
    f"{OUTER} {G} WHERE o.item = i.id AND i.kind = 'tool' AND i.id = g.item",
    # the key tied to the group's key through the order's column
    f"{OUTER} {G} WHERE o.item = i.id AND i.kind = 'tool' AND g.item = o.item",
    # ON clauses, the grouped table first
    f"SELECT o.id, g.usual FROM {G} JOIN items AS i ON i.id = g.item AND i.kind = 'tool' JOIN orders AS o ON o.item = i.id",
    # the select around repeats more than the filter
    f"{OUTER} {G} WHERE o.item = i.id AND i.kind = 'tool' AND i.size < 3 AND i.id = g.item AND o.qty < g.usual",
    # no filter inside
    f"{OUTER} (SELECT o2.item AS item, AVG(o2.qty) AS usual, COUNT(i2.id) AS n FROM orders AS o2 JOIN items AS i2 ON i2.id = o2.item GROUP BY o2.item) AS g WHERE o.item = i.id AND i.id = g.item",
    # grouped by both spellings of the key, a test of the orders kept, the key read inside an aggregate and beside one
    f"{OUTER} (SELECT o2.item AS item, 0.5 * AVG(o2.qty) + i2.id AS usual, MAX(i2.id) AS n FROM orders AS o2 JOIN items AS i2 ON i2.id = o2.item AND o2.qty > 1 WHERE {TOOLS} GROUP BY o2.item, i2.id) AS g WHERE o.item = i.id AND i.kind = 'tool' AND i.id = g.item",
    # two filter tests, and one that is not a comparison
    f"{OUTER} (SELECT i2.id AS item, AVG(o2.qty) AS usual, COUNT(*) AS n FROM items AS i2, orders AS o2 WHERE o2.item = i2.id AND i2.kind IN ('tool', 'toy') AND (i2.size > 1 OR i2.name = 'saw') GROUP BY i2.id) AS g "
    "WHERE (i.size > 1 OR i.name = 'saw') AND o.item = i.id AND i.kind IN ('tool', 'toy') AND i.id = g.item",
    # the select around is itself grouped
    f"SELECT i.kind, SUM(g.n) AS orders, COUNT(*) AS items FROM items AS i JOIN {G} ON i.id = g.item WHERE i.kind = 'tool' GROUP BY i.kind",
    # inside a derived table
    f"SELECT t.id FROM ({OUTER} {G} WHERE o.item = i.id AND i.kind = 'tool' AND i.id = g.item) AS t WHERE t.n > 1",
]

LEFT_ALONE = [
    # the select around does not repeat the filter, or repeats it under an OR
    f"{OUTER} {G} WHERE o.item = i.id AND i.id = g.item",
    f"{OUTER} {G} WHERE o.item = i.id AND (i.kind = 'tool' OR o.qty > 3) AND i.id = g.item",
    f"{OUTER} {G} WHERE o.item = i.id AND i.kind = 'toy' AND i.id = g.item",
    # the group's key is not tied to the key of the second read
    f"{OUTER} {G} WHERE o.item = i.id AND i.kind = 'tool' AND i.size = g.item",
    f"{OUTER} {G} WHERE o.item = i.id AND i.kind = 'tool' AND (i.id = g.item OR o.qty > 3)",
    f"{OUTER} {G} WHERE o.item = i.id AND i.kind = 'tool' AND i.id <= g.item",
    f"SELECT o.id, g.usual, g.n FROM orders AS o, {G} WHERE o.item = g.item",
    # tied through a text column
    f"{OUTER} {G} WHERE o.note = i.name AND i.kind = 'tool' AND CAST(g.item AS VARCHAR) = o.note AND i.id = o.shop",
    # an outer join around
    f"SELECT o.id, g.usual, g.n FROM orders AS o JOIN items AS i ON o.item = i.id LEFT JOIN {G} ON i.id = g.item WHERE i.kind = 'tool'",
    # inside the grouped table: the join is not by the key, or reads more
    f"{OUTER} (SELECT i2.size AS item, AVG(o2.qty) AS usual, COUNT(*) AS n FROM items AS i2, orders AS o2 WHERE o2.shop = i2.size AND {TOOLS} GROUP BY i2.size) AS g WHERE o.item = i.id AND i.kind = 'tool' AND i.size = g.item",
    f"{OUTER} (SELECT i2.id AS item, AVG(o2.qty) AS usual, COUNT(*) AS n FROM items AS i2, orders AS o2 WHERE o2.item = i2.id AND o2.day = i2.size AND {TOOLS} GROUP BY i2.id) AS g WHERE o.item = i.id AND i.kind = 'tool' AND i.id = g.item",
    f"{OUTER} (SELECT i2.id AS item, AVG(o2.qty) AS usual, COUNT(*) AS n FROM items AS i2, orders AS o2 WHERE o2.item = i2.id AND o2.qty > i2.size AND {TOOLS} GROUP BY i2.id) AS g WHERE o.item = i.id AND i.kind = 'tool' AND i.id = g.item",
    f"{OUTER} (SELECT i2.id AS item, AVG(o2.qty) AS usual, COUNT(*) AS n FROM items AS i2 LEFT JOIN orders AS o2 ON o2.item = i2.id WHERE {TOOLS} GROUP BY i2.id) AS g WHERE o.item = i.id AND i.kind = 'tool' AND i.id = g.item",
    f"{OUTER} (SELECT i2.id AS item, AVG(o2.qty) AS usual, MAX(i2.size) AS n FROM items AS i2, orders AS o2 WHERE o2.item = i2.id AND {TOOLS} GROUP BY i2.id) AS g WHERE o.item = i.id AND i.kind = 'tool' AND i.id = g.item",
    f"{OUTER} (SELECT i2.id AS item, AVG(o2.qty) AS usual, COUNT(*) AS n FROM items AS i2, orders AS o2 WHERE o2.item = i2.id AND {TOOLS} GROUP BY i2.id, i2.size) AS g WHERE o.item = i.id AND i.kind = 'tool' AND i.id = g.item",
    f"{OUTER} (SELECT i2.id AS item, AVG(o2.qty) AS usual, COUNT(*) AS n FROM items AS i2, orders AS o2 WHERE o2.item = i2.id AND {TOOLS} GROUP BY i2.id, o2.day) AS g WHERE o.item = i.id AND i.kind = 'tool' AND i.id = g.item",
    f"{OUTER} (SELECT i2.id AS item, AVG(o2.qty) AS usual, COUNT(*) AS n FROM items AS i2, orders AS o2 WHERE o2.item = i2.id AND {TOOLS} GROUP BY i2.id HAVING COUNT(*) > 1) AS g WHERE o.item = i.id AND i.kind = 'tool' AND i.id = g.item",
    f"{OUTER} (SELECT DISTINCT i2.id AS item, AVG(o2.qty) AS usual, COUNT(*) AS n FROM items AS i2, orders AS o2 WHERE o2.item = i2.id AND {TOOLS} GROUP BY i2.id) AS g WHERE o.item = i.id AND i.kind = 'tool' AND i.id = g.item",
    f"{OUTER} (SELECT i2.id AS item, AVG(o2.qty) AS usual, COUNT(*) AS n FROM items AS i2, orders AS o2, shops AS s2 WHERE o2.item = i2.id AND {TOOLS} GROUP BY i2.id) AS g WHERE o.item = i.id AND i.kind = 'tool' AND i.id = g.item",
    f"{OUTER} (SELECT i2.id AS item, AVG(o2.qty) AS usual, COUNT(*) AS n FROM items AS i2, orders AS o2 WHERE o2.item = i2.id AND i2.size IN (SELECT s.id FROM shops AS s) GROUP BY i2.id) AS g WHERE o.item = i.id AND i.size IN (SELECT s.id FROM shops AS s) AND i.id = g.item",
    f"{OUTER} (SELECT i2.id AS item, AVG(o2.qty) AS usual, SUM(COUNT(*)) OVER () AS n FROM items AS i2, orders AS o2 WHERE o2.item = i2.id AND {TOOLS} GROUP BY i2.id) AS g WHERE o.item = i.id AND i.kind = 'tool' AND i.id = g.item",
    # a column inside without its source's name
    f"{OUTER} (SELECT i2.id AS item, AVG(qty) AS usual, COUNT(*) AS n FROM items AS i2, orders AS o2 WHERE o2.item = i2.id AND {TOOLS} GROUP BY i2.id) AS g WHERE o.item = i.id AND i.kind = 'tool' AND i.id = g.item",
    # a text key, and a key that is not declared NOT NULL
    f"{OUTER} (SELECT i2.name AS item, AVG(o2.qty) AS usual, COUNT(*) AS n FROM items AS i2, orders AS o2 WHERE o2.note = i2.name AND {TOOLS} GROUP BY i2.name) AS g WHERE o.item = i.id AND i.kind = 'tool' AND i.name = g.item",
    "SELECT o.id, g.n FROM orders AS o, shops AS s, (SELECT s2.id AS shop, COUNT(*) AS n FROM shops AS s2, orders AS o2 WHERE o2.shop = s2.id GROUP BY s2.id) AS g WHERE o.shop = s.id AND s.id = g.shop",
    # the second read is of another table
    f"SELECT o.id, g.usual, g.n FROM orders AS o, shops AS i, {G} WHERE o.shop = i.id AND i.id = g.item",
]

# (as written, read by hand the way the rule would): the rows differ, so the refusal is needed
BARE = "(SELECT o2.item AS item, AVG(o2.qty) AS usual, COUNT(*) AS n FROM orders AS o2 GROUP BY o2.item) AS g"
BY_HAND = [(written, written.replace(G, BARE)) for written in (LEFT_ALONE[0], LEFT_ALONE[1], LEFT_ALONE[6])] + [
    (
        LEFT_ALONE[11],
        f"{OUTER} (SELECT o2.item AS item, AVG(o2.qty) AS usual, COUNT(*) AS n FROM orders AS o2 GROUP BY o2.item) AS g WHERE o.item = i.id AND i.kind = 'tool' AND i.id = g.item",
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


def _read(sql: str, **changed) -> tuple[str, int]:
    """The query with the rule applied to every select until nothing changes, and how often it was applied."""

    facts = {"schema": SCHEMA, "keys": KEYS, "not_null": NOT_NULL, "types": TYPES, **changed}
    tree, applied = sqlglot.parse_one(sql, read="postgres"), 0
    while True:
        for select in tree.find_all(sqlglot.exp.Select):
            new = drop_rejoined_parent(select, facts["schema"], facts["keys"], facts["not_null"], facts["types"])
            if new is not None:
                if select is tree:
                    tree = new
                else:
                    select.replace(new)
                applied += 1
                break
        else:
            return tree.sql(dialect="postgres"), applied


@pytest.mark.parametrize("written", READ)
def test_a_reading_returns_the_rows_of_the_query_as_written(written):
    read, applied = _read(written)
    assert applied == 1 and "i2" not in read, read
    for dialect, database in _databases():
        rows = _rows(database, written, dialect)
        assert rows == _rows(database, read, dialect), dialect
        assert rows, "the made-up rows do not reach this query"


@pytest.mark.parametrize("written", LEFT_ALONE)
def test_a_query_that_is_not_read_stays_as_written(written):
    assert _read(written)[1] == 0


@pytest.mark.parametrize("written, by_hand", BY_HAND)
def test_reading_a_refused_query_by_hand_changes_rows(written, by_hand):
    assert by_hand != written and _read(written)[1] == 0
    database = next(db for dialect, db in _databases() if dialect == "duckdb")
    assert _rows(database, written, "duckdb") != _rows(database, by_hand, "duckdb")


def test_the_reading_needs_the_declared_facts():
    written = READ[0]
    assert _read(written)[1] == 1
    assert _read(written, types=None)[1] == 0
    assert _read(written, keys={"orders": [("id",)]})[1] == 0
    assert _read(written, not_null={"orders": frozenset({"id"})})[1] == 0
    assert _read(written, types={**TYPES, "orders": {**TYPES["orders"], "item": "VARCHAR"}})[1] == 0
    # a WITH name hides the declared table: the two reads are not of the declared rows
    assert _read(f"WITH items AS (SELECT item AS id, 'tool' AS kind FROM orders) {written}")[1] == 0


def test_the_rule_copies_the_select_it_is_given():
    select = sqlglot.parse_one(READ[0], read="postgres")
    before = select.sql()
    assert drop_rejoined_parent(select, SCHEMA, KEYS, NOT_NULL, TYPES) is not None
    assert select.sql() == before


def _proved(left: str, right: str) -> bool:
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect="postgres").proven


BELOW_USUAL = "SELECT SUM(o.price) AS total FROM orders AS o, items AS i WHERE i.id = o.item AND i.kind = 'tool' AND o.qty < (SELECT 0.5 * AVG(q.qty) FROM orders AS q WHERE q.item = i.id)"
STAGED = (
    "WITH tools AS (SELECT id FROM items WHERE kind = 'tool'), "
    "theirs AS (SELECT tools.id, o.qty, o.price FROM tools JOIN orders AS o ON tools.id = o.item), "
    "usual AS (SELECT id, 0.5 * AVG(qty) AS t FROM theirs GROUP BY id) "
    "SELECT SUM(theirs.price) AS total FROM theirs JOIN usual ON theirs.id = usual.id WHERE theirs.qty < usual.t"
)


def test_a_pair_is_proved_and_only_with_the_rule():
    assert _proved(BELOW_USUAL, STAGED)
    with shape_fallback.without("rejoined_parents"):
        assert not _proved(BELOW_USUAL, STAGED)


def test_pairs_that_differ_stay_unproved():
    # the mean of the orders above one, not of all the item's orders
    assert not _proved(BELOW_USUAL, STAGED.replace("FROM theirs GROUP BY id", "FROM theirs WHERE qty > 1 GROUP BY id"))
    # the totals of every item, with the means of the tools: an order of a toy has no mean, but an order of a tool
    # read against the means of the small tools only is another query
    small = STAGED.replace("usual AS (SELECT id, 0.5 * AVG(qty) AS t FROM theirs GROUP BY id)", "usual AS (SELECT i.id, 0.5 * AVG(o.qty) AS t FROM items AS i JOIN orders AS o ON o.item = i.id WHERE i.kind = 'tool' AND i.size < 3 GROUP BY i.id)")
    assert small != STAGED and not _proved(BELOW_USUAL, small)
