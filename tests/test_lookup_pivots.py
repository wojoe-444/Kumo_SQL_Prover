"""A table turned on its side with one lookup per column, read as the grouped select (``lookup_pivots``).

Each reading is run as written and as read on SQLite, and on DuckDB where DuckDB takes the spelling (it refuses a
``GROUP BY`` that leaves a selected column out), over a made-up table with one row per item and shop: an item in
every shop, an item in one, a cost that is NULL, two items with the same costs. The rows must be the same.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot
from sqlglot import exp

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.lookup_pivots import read_lookup_pivots
from kumo_sql_prover.smt_equivalence import TableConstraints

COLUMNS = "item INT, shop VARCHAR, cost INT, note VARCHAR, seen DATE"
ROWS = [
    (1, "north", 10, "a", "2024-01-02"),
    (1, "south", 12, "b", "2024-01-03"),
    (1, "east", 10, "c", None),
    (2, "south", 7, "d", "2024-02-01"),
    (3, "north", None, None, "2024-03-01"),
    (3, "east", 10, "e", "2024-03-02"),
    (4, "south", 7, "f", "2024-02-01"),
    (5, "east", 1, "g", "2024-05-05"),
]
SCHEMA = {"prices": ["item", "shop", "cost", "note", "seen"], "stock": ["item", "shop", "n"]}
TYPES = {"prices": {"item": "INT", "shop": "VARCHAR", "cost": "INT", "note": "VARCHAR", "seen": "DATE"}, "stock": {"item": "INT", "shop": "VARCHAR", "n": "INT"}}
KEYS = {"prices": [("item", "shop")], "stock": [("item", "shop")]}
NOT_NULL = {"prices": frozenset({"item", "shop"}), "stock": frozenset({"item", "shop"})}
CONSTRAINTS = {name: TableConstraints(not_null=NOT_NULL[name], keys=tuple(KEYS[name])) for name in SCHEMA}

NORTH = "MAX(CASE WHEN {a}.shop = 'north' THEN {a}.cost END)"
SOUTH = "MAX(CASE WHEN {a}.shop = 'south' THEN {a}.cost END)"

# (as written, as read, whether DuckDB takes the spelling as written)
READ = [
    (
        "SELECT i.item, n.cost AS north, s.cost AS south FROM (SELECT DISTINCT item FROM prices) AS i "
        "LEFT JOIN prices AS n ON n.item = i.item AND n.shop = 'north' LEFT JOIN prices AS s ON i.item = s.item AND 'south' = s.shop",
        f"SELECT i.item, {NORTH} AS north, {SOUTH} AS south FROM prices AS i GROUP BY i.item".format(a="i"),
        True,
    ),
    (
        "SELECT DISTINCT p.item, n.north, south FROM prices AS p LEFT JOIN (SELECT item, cost AS north FROM prices WHERE shop = 'north') AS n ON n.item = p.item "
        "LEFT OUTER JOIN (SELECT item, cost AS south FROM prices WHERE shop = 'south') AS s ON s.item = p.item",
        f"SELECT p.item, {NORTH} AS north, {SOUTH} AS south FROM prices AS p GROUP BY p.item".format(a="p"),
        True,
    ),
    (
        "SELECT DISTINCT item, (SELECT cost FROM prices AS n WHERE n.shop = 'north' AND n.item = p.item) AS north, "
        "(SELECT s.cost FROM prices AS s WHERE p.item = s.item AND s.shop = 'south') AS south FROM prices AS p",
        f"SELECT p.item, {NORTH} AS north, {SOUTH} AS south FROM prices AS p GROUP BY p.item".format(a="p"),
        True,
    ),
    (  # the table as driver with GROUP BY: the lookups' columns are the same on every row of the group
        "SELECT p.item, n.cost AS north, COALESCE(s.cost, 0) AS south FROM prices AS p LEFT JOIN prices AS n ON n.item = p.item AND n.shop = 'north' "
        "LEFT JOIN prices AS s ON s.item = p.item AND s.shop = 'south' GROUP BY p.item",
        f"SELECT p.item, {NORTH} AS north, COALESCE({SOUTH}, 0) AS south FROM prices AS p GROUP BY p.item".format(a="p"),
        False,
    ),
    (
        "SELECT a.item, b.cost AS north FROM prices AS a LEFT JOIN prices AS b ON a.item = b.item AND b.shop = 'north' GROUP BY 1, 2",
        f"SELECT a.item, {NORTH} AS north FROM prices AS a GROUP BY a.item".format(a="a"),
        True,
    ),
    (
        "SELECT a.item, MAX(b.cost) AS north, MIN(c.cost) AS south FROM prices AS a LEFT JOIN prices AS b ON a.item = b.item AND b.shop = 'north' "
        "LEFT JOIN prices AS c ON a.item = c.item AND c.shop = 'south' GROUP BY a.item",
        f"SELECT a.item, {NORTH} AS north, {SOUTH} AS south FROM prices AS a GROUP BY a.item".format(a="a"),
        True,
    ),
    (  # the list of items written with GROUP BY, and with both
        "SELECT i.item, n.cost AS north FROM (SELECT item FROM prices GROUP BY item) AS i LEFT JOIN prices AS n ON n.item = i.item AND n.shop = 'north'",
        f"SELECT i.item, {NORTH} AS north FROM prices AS i GROUP BY i.item".format(a="i"),
        True,
    ),
    (
        "SELECT i.item, n.cost AS north FROM (SELECT DISTINCT item FROM prices GROUP BY item) AS i LEFT JOIN prices AS n ON n.item = i.item AND n.shop = 'north'",
        f"SELECT i.item, {NORTH} AS north FROM prices AS i GROUP BY i.item".format(a="i"),
        True,
    ),
    (  # a list under another name, a lookup that shows a value of its own, a text and a date
        "SELECT i.what, n.twice, n.note AS said, n.seen FROM (SELECT DISTINCT (item) AS what FROM prices) AS i "
        "LEFT JOIN (SELECT item AS it, cost * 2 AS twice, note, seen FROM prices WHERE shop = 'north') AS n ON n.it = i.what",
        "SELECT i.item AS what, MAX(CASE WHEN i.shop = 'north' THEN i.cost * 2 END) AS twice, MAX(CASE WHEN i.shop = 'north' THEN i.note END) AS said, "
        "MAX(CASE WHEN i.shop = 'north' THEN i.seen END) AS seen FROM prices AS i GROUP BY i.item",
        True,
    ),
    (  # the select list does not show the item: two items with the same costs are one row, and DISTINCT stays
        "SELECT DISTINCT n.cost AS north, s.cost AS south FROM prices AS a LEFT JOIN prices AS n ON a.item = n.item AND n.shop = 'north' "
        "LEFT JOIN prices AS s ON a.item = s.item AND s.shop = 'south'",
        f"SELECT DISTINCT {NORTH} AS north, {SOUTH} AS south FROM prices AS a GROUP BY a.item".format(a="a"),
        True,
    ),
    (  # without DISTINCT, from the list: one row per item, the same costs twice
        "SELECT s.cost AS south FROM (SELECT DISTINCT item FROM prices) AS a LEFT JOIN prices AS s ON a.item = s.item AND s.shop = 'south'",
        f"SELECT {SOUTH} AS south FROM prices AS a GROUP BY a.item".format(a="a"),
        True,
    ),
    (  # the whole key in the lookup's conditions
        "SELECT DISTINCT p.shop, p.item, o.cost AS was FROM prices AS p LEFT JOIN prices AS o ON o.item = p.item AND o.shop = p.shop",
        "SELECT p.shop, p.item, MAX(p.cost) AS was FROM prices AS p GROUP BY p.item, p.shop",
        True,
    ),
    (  # inside a derived table
        "SELECT t.item FROM (SELECT DISTINCT p.item, n.cost AS north FROM prices AS p LEFT JOIN prices AS n ON n.item = p.item AND n.shop = 'north') AS t WHERE t.north > 5",
        f"SELECT t.item FROM (SELECT p.item, {NORTH} AS north FROM prices AS p GROUP BY p.item) AS t WHERE t.north > 5".format(a="p"),
        True,
    ),
]
# left as written
LEFT_ALONE = [
    "SELECT a.item, n.cost AS north FROM prices AS a LEFT JOIN prices AS n ON a.item = n.item AND n.shop = 'north'",  # a row per row of the item
    "SELECT DISTINCT a.item, n.cost AS north FROM prices AS a JOIN prices AS n ON a.item = n.item AND n.shop = 'north'",  # drops items
    "SELECT DISTINCT a.item, n.cost AS north FROM prices AS a LEFT JOIN prices AS n ON a.item = n.item AND n.shop = 5",  # a text against a number
    "SELECT DISTINCT a.item, n.cost AS north FROM prices AS a LEFT JOIN prices AS n ON a.item = n.item AND n.shop <> 'north'",
    "SELECT DISTINCT a.item, n.cost AS north FROM prices AS a LEFT JOIN prices AS n ON a.item = n.item AND n.cost = 10",  # (item, cost) is no key
    "SELECT DISTINCT a.item, n.cost AS north FROM prices AS a LEFT JOIN prices AS n ON a.item = n.item AND n.shop = 'north' AND a.cost > 3",
    "SELECT DISTINCT a.item, a.cost, n.cost AS north FROM prices AS a LEFT JOIN prices AS n ON a.item = n.item AND n.shop = 'north'",  # a column of the driver's row
    "SELECT DISTINCT a.item, n.shop FROM prices AS a LEFT JOIN prices AS n ON a.item = n.item AND n.shop = 'north'",  # the lookup's compared column
    "SELECT DISTINCT a.item, n.cost AS north FROM prices AS a LEFT JOIN prices AS n ON a.item = n.item AND n.shop = 'north' WHERE a.cost > 3",
    "SELECT DISTINCT a.item, n.cost AS north FROM prices AS a LEFT JOIN prices AS n ON a.item = n.item AND n.shop = 'north' ORDER BY 1",
    "SELECT DISTINCT a.item, n.n AS north FROM prices AS a LEFT JOIN stock AS n ON a.item = n.item AND n.shop = 'north'",  # another table
    "SELECT DISTINCT a.item, s.cost AS south FROM prices AS a LEFT JOIN prices AS n ON a.item = n.item AND n.shop = 'north' LEFT JOIN prices AS s ON s.item = n.item AND s.shop = 'south'",
    "SELECT DISTINCT a.shop, n.cost FROM prices AS a LEFT JOIN prices AS n ON a.shop = n.shop AND n.item = 1 LEFT JOIN prices AS s ON a.item = s.item AND s.shop = 'south'",  # two lists
    "SELECT a.item, SUM(n.cost) AS north FROM prices AS a LEFT JOIN prices AS n ON a.item = n.item AND n.shop = 'north' GROUP BY a.item",  # once per row of the item
    "SELECT a.item, n.cost AS north FROM prices AS a LEFT JOIN prices AS n ON a.item = n.item AND n.shop = 'north' GROUP BY n.cost",
    "SELECT DISTINCT item, cost AS north FROM prices AS a LEFT JOIN prices AS n ON a.item = n.item AND n.shop = 'north'",  # names two sources have
    "SELECT DISTINCT item, (SELECT cost FROM prices AS n WHERE n.shop = 'north' AND n.item = item) AS north FROM prices AS p",  # the inner item twice
    "SELECT DISTINCT item, (SELECT MAX(cost) FROM prices AS n WHERE n.shop = 'north' AND n.item = p.item) AS north FROM prices AS p",
    "SELECT i.item, n.cost AS north FROM (SELECT DISTINCT item FROM prices WHERE cost > 7) AS i LEFT JOIN prices AS n ON n.item = i.item AND n.shop = 'north'",
    "SELECT i.item, n.cost AS north FROM (SELECT DISTINCT item, shop FROM prices) AS i LEFT JOIN prices AS n ON n.item = i.item AND n.shop = 'north'",
    "WITH prices AS (SELECT 1 AS item, 'north' AS shop, 2 AS cost UNION ALL SELECT 1, 'north', 3) SELECT DISTINCT a.item, n.cost AS north FROM prices AS a LEFT JOIN prices AS n ON a.item = n.item AND n.shop = 'north'",
]


def read(sql: str, keys=KEYS, not_null=NOT_NULL, types=TYPES) -> tuple[str, bool]:
    tree, wrote = read_lookup_pivots(sqlglot.parse_one(sql, read="mysql"), SCHEMA, keys, not_null, types)
    return tree.sql(dialect="mysql"), wrote


def _sorted(rows) -> list:
    return sorted(([str(value) if value is not None else None for value in row] for row in rows), key=repr)


def _duckdb_rows(sql: str, rows=ROWS) -> list:
    duckdb = pytest.importorskip("duckdb")
    db = duckdb.connect(":memory:")
    db.execute(f"CREATE TABLE prices ({COLUMNS})")
    db.executemany("INSERT INTO prices VALUES (?, ?, ?, ?, ?)", rows)
    return _sorted(db.execute(sqlglot.transpile(sql, read="mysql", write="duckdb")[0]).fetchall())


def _sqlite_rows(sql: str, rows=ROWS) -> list:
    db = sqlite3.connect(":memory:")
    db.execute(f"CREATE TABLE prices ({COLUMNS})")
    db.executemany("INSERT INTO prices VALUES (?, ?, ?, ?, ?)", rows)
    return _sorted(db.execute(sqlglot.transpile(sql, read="mysql", write="sqlite")[0]).fetchall())


def prove(left: str, right: str, constraints=CONSTRAINTS):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, constraints=constraints, compare_names=False, dialect="mysql", exact_arithmetic=True, timeout_ms=5000)


@pytest.mark.parametrize("written, expected, duckdb_takes_it", READ)
def test_a_reading_returns_the_rows_of_the_query_as_written(written, expected, duckdb_takes_it):
    reading, wrote = read(written)
    assert wrote
    assert sqlglot.parse_one(reading, read="mysql") == sqlglot.parse_one(expected, read="mysql")
    assert not list(sqlglot.parse_one(reading, read="mysql").find_all(exp.Join))
    assert _sqlite_rows(written) == _sqlite_rows(reading) and _sqlite_rows(written)
    assert _duckdb_rows(reading) == _sqlite_rows(reading)
    if duckdb_takes_it:
        assert _duckdb_rows(written) == _duckdb_rows(reading)


@pytest.mark.parametrize("written", LEFT_ALONE)
def test_a_query_that_is_not_read_stays_as_written(written):
    tree = sqlglot.parse_one(written, read="mysql")
    same, wrote = read_lookup_pivots(tree, SCHEMA, KEYS, NOT_NULL, TYPES)
    assert not wrote and same is tree


def test_the_readings_refused_would_change_rows():
    north = f"SELECT a.item, {NORTH} AS north FROM prices AS a GROUP BY a.item".format(a="a")
    # the table as driver with no DISTINCT: a row per row of the item
    assert _duckdb_rows(LEFT_ALONE[0]) != _duckdb_rows(north)
    # an inner join: only the items with a north row
    assert _duckdb_rows(LEFT_ALONE[1]) != _duckdb_rows(north)
    # SUM over the driver's rows adds the one cost once per row of the item
    assert _duckdb_rows(LEFT_ALONE[13]) != _duckdb_rows(north)
    # an item column that may be NULL: the lookup finds nothing for it, the group has its rows
    rows = [(None, "north", 3, "a", None), (None, "south", 4, "b", None), (1, "north", 5, "c", None)]
    written = READ[0][0]
    assert _duckdb_rows(written, rows) != _duckdb_rows(READ[0][1], rows)
    # without the key two rows of one item and shop are two lookup rows
    rows = [(1, "north", 3, "a", None), (1, "north", 4, "b", None)]
    assert _duckdb_rows(written, rows) != _duckdb_rows(READ[0][1], rows)


def test_a_key_is_needed_and_its_columns_must_be_not_null():
    written = READ[0][0]
    assert read(written)[1]
    assert not read(written, keys={"prices": []})[1]
    assert not read(written, keys={"prices": [("item", "shop", "cost")]})[1]
    assert not read(written, not_null={"prices": frozenset({"shop"})})[1]
    assert not read(written, not_null={"prices": frozenset({"item"})})[1]
    assert not read(written, not_null=None)[1]
    assert not read(written, types=None)[1]
    untyped = {"prices": {"item": "INT", "shop": "VARCHAR", "cost": "GEOGRAPHY"}}
    assert not read(written, types=untyped)[1]  # MAX may not take the value


GROUPED = "SELECT item, MAX(CASE WHEN shop = 'north' THEN cost END) AS north, MAX(CASE WHEN shop = 'south' THEN cost END) AS south FROM prices GROUP BY item"
SUMMED = GROUPED.replace("MAX(", "SUM(")
LOOKUPS = [READ[0][0], READ[1][0], READ[2][0], READ[5][0]]


@pytest.mark.parametrize("position", range(len(LOOKUPS)))
def test_a_lookup_spelling_is_proved_equal_to_the_grouped_one_and_only_with_the_rule(position):
    written = LOOKUPS[position]
    assert _sqlite_rows(written) == _sqlite_rows(GROUPED) == _sqlite_rows(SUMMED)
    assert prove(written, GROUPED).proven
    assert prove(SUMMED, written).proven  # with the other rule of the batch
    with shape_fallback.without("lookup_pivots"):
        assert not prove(written, GROUPED).proven
    assert not prove(written, GROUPED, {name: TableConstraints(not_null=NOT_NULL[name]) for name in SCHEMA}).proven
    assert not prove(written, GROUPED, {name: TableConstraints(not_null=frozenset({"shop"}), keys=tuple(KEYS[name])) for name in SCHEMA}).proven


def test_two_lookup_spellings_are_proved_equal():
    assert prove(LOOKUPS[0], LOOKUPS[1]).proven
    assert prove(LOOKUPS[1], LOOKUPS[2]).proven


def test_spellings_that_differ_stay_unproved():
    for written in LOOKUPS:
        assert not prove(written, GROUPED.replace("'south'", "'east'")).proven
        assert not prove(written, GROUPED.replace("THEN cost END) AS north", "THEN cost + 1 END) AS north")).proven
    assert not prove(LEFT_ALONE[0], GROUPED.replace(", MAX(CASE WHEN shop = 'south' THEN cost END) AS south", "")).proven
    assert not prove(LEFT_ALONE[1], GROUPED.replace(", MAX(CASE WHEN shop = 'south' THEN cost END) AS south", "")).proven


def test_a_single_precision_value_is_not_read():
    # MySQL shows a FLOAT column as 0.1 and MAX of it, a DOUBLE, as 0.10000000149011612 once it is a text
    written = READ[0][0]
    for declared in ("FLOAT", "float", "REAL", "FLOAT4"):
        assert not read(written, types={**TYPES, "prices": {**TYPES["prices"], "cost": declared}})[1]
    for declared in ("DOUBLE", "DECIMAL(8, 2)", "BIGINT"):
        assert read(written, types={**TYPES, "prices": {**TYPES["prices"], "cost": declared}})[1]
