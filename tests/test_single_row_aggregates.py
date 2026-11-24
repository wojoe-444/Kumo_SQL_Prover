"""``MIN``, ``SUM`` and ``MAX`` of a value only one row of a group can have (``single_row_aggregates``).

Each reading is run as written and as read on DuckDB and on SQLite over a made-up table with one row per item and
shop: an item in every shop, an item in one, a cost that is NULL, two items with the same cost. The rows must be the
same. The queries the rule leaves alone come with a table on which the reading would be wrong where one exists.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.single_row_aggregates import Keyed, guarded, read_single_row_aggregates
from kumo_sql_prover.smt_equivalence import TableConstraints

COLUMNS = "item INT, shop VARCHAR, cost INT, note VARCHAR, weight DOUBLE"
ROWS = [
    (1, "north", 10, "a", 1.5),
    (1, "south", 12, "b", 2.5),
    (1, "east", 10, "c", 0.5),
    (2, "south", 7, "7 apples", 4.0),
    (3, "north", None, None, None),
    (3, "east", 10, "d", 3.0),
    (4, "east", 12, "e", 2.0),
]
SCHEMA = {"prices": ["item", "shop", "cost", "note", "weight"]}
TYPES = {"prices": {"item": "INT", "shop": "VARCHAR", "cost": "INT", "note": "VARCHAR", "weight": "DOUBLE"}}
KEYS = {"prices": [("item", "shop")]}
NOT_NULL = {"prices": frozenset({"item", "shop"})}
CONSTRAINTS = {"prices": TableConstraints(not_null=NOT_NULL["prices"], keys=(("item", "shop"),))}

# (as written, the aggregates of the reading in order)
READ = [
    ("SELECT item, SUM(CASE WHEN shop = 'north' THEN cost END) AS north FROM prices GROUP BY item", ["MAX"]),
    ("SELECT item, MIN(CASE WHEN shop = 'north' THEN cost END) AS north, MAX(CASE WHEN shop = 'south' THEN cost END) AS south FROM prices GROUP BY item", ["MAX", "MAX"]),
    ("SELECT item, SUM(IF(shop = 'east', cost, NULL)) AS east, MIN(IF('south' = shop, note, NULL)) AS said FROM prices GROUP BY 1", ["MAX", "MAX"]),
    ("SELECT item, SUM(CASE shop WHEN 'north' THEN cost ELSE NULL END) AS north FROM prices GROUP BY item", ["MAX"]),
    ("SELECT p.item, SUM(CASE WHEN p.shop = 'north' AND p.cost > 5 THEN p.cost * 2 - 1 END) AS north FROM prices AS p GROUP BY p.item", ["MAX"]),
    ("SELECT item, SUM(DISTINCT CASE WHEN shop = 'south' THEN cost END) AS south FROM prices WHERE cost > 7 GROUP BY item", ["MAX"]),
    # the whole key in the condition: no grouping is needed
    ("SELECT MIN(CASE WHEN item = 1 AND shop = 'south' THEN note END) AS said FROM prices", ["MAX"]),
    # the whole key in the GROUP BY: every group is one row
    ("SELECT item, shop, SUM(CASE WHEN cost > 9 THEN cost END) AS dear FROM prices GROUP BY item, shop", ["MAX"]),
    # a grouping expression besides the key column only makes the groups smaller
    ("SELECT item, cost > 9 AS dear, MIN(CASE WHEN shop = 'east' THEN cost END) AS east FROM prices GROUP BY item, cost > 9", ["MAX"]),
    # in HAVING too
    ("SELECT item, MIN(CASE WHEN shop = 'south' THEN note END) AS said FROM prices GROUP BY item HAVING SUM(CASE WHEN shop = 'south' THEN cost END) > 7", ["MAX", "MAX"]),
]
# left as written
LEFT_ALONE = [
    "SELECT shop, SUM(CASE WHEN cost = 10 THEN cost END) AS tens FROM prices GROUP BY shop",  # (shop, cost) is no key
    "SELECT item, SUM(CASE WHEN shop = 'north' THEN cost ELSE 0 END) AS north FROM prices GROUP BY item",  # 0 on the other rows
    "SELECT item, MIN(CASE WHEN shop = 'north' THEN cost WHEN shop = 'south' THEN cost END) AS either FROM prices GROUP BY item",
    "SELECT item, MIN(CASE WHEN shop <> 'north' THEN cost END) AS others FROM prices GROUP BY item",
    "SELECT item, MIN(CASE WHEN shop = 7 THEN cost END) AS odd FROM prices GROUP BY item",  # a text column against a number converts
    "SELECT shop, MIN(CASE WHEN item = '1' THEN cost END) AS odd FROM prices GROUP BY shop",  # a number column against a text converts
    "SELECT item, SUM(CASE WHEN shop = 'south' THEN note END) AS n FROM prices GROUP BY item",  # SUM converts a text
    "SELECT item, SUM(CASE WHEN shop = 'south' THEN weight END) AS w FROM prices GROUP BY item",  # a floating-point column
    "SELECT item, SUM(CASE WHEN shop = 'south' THEN cost / 2 END) AS half FROM prices GROUP BY item",
    "SELECT a.item, MIN(CASE WHEN a.shop = 'north' THEN b.cost END) AS north FROM prices AS a JOIN prices AS b ON a.item = b.item GROUP BY a.item",
    "SELECT item, MIN(CASE WHEN shop = 'north' THEN cost END) OVER (PARTITION BY item) AS north FROM prices",
    "SELECT item, MIN(CASE WHEN shop = 'north' THEN cost END) AS north FROM prices GROUP BY ROLLUP (item)",
    "SELECT MIN(CASE WHEN shop = 'north' THEN cost END) AS north FROM prices",  # no grouping: every item's north row
    "SELECT item, MIN(CASE WHEN shop = 'north' THEN cost END) AS north FROM (SELECT * FROM prices) AS p GROUP BY item",
    "WITH prices AS (SELECT 1 AS item, 'north' AS shop, 2 AS cost UNION ALL SELECT 1, 'north', 3) SELECT item, MIN(CASE WHEN shop = 'north' THEN cost END) AS north FROM prices GROUP BY item",
    "SELECT shop AS item, MIN(CASE WHEN shop = 'north' THEN cost END) AS north FROM prices GROUP BY item",  # a name that is a column and an alias
]


def read(sql: str, keys=KEYS, not_null=NOT_NULL, types=TYPES) -> tuple[str, bool]:
    tree, wrote = read_single_row_aggregates(sqlglot.parse_one(sql, read="mysql"), SCHEMA, keys, not_null, types)
    return tree.sql(dialect="mysql"), wrote


def _sorted(rows) -> list:
    return sorted((list(row) for row in rows), key=repr)


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


@pytest.mark.parametrize("written, aggregates", READ)
def test_a_reading_returns_the_rows_of_the_query_as_written(written, aggregates):
    reading, wrote = read(written)
    assert wrote and reading != written
    calls = [type(node).__name__.upper() for node in sqlglot.parse_one(reading, read="mysql").find_all(sqlglot.exp.AggFunc)]
    assert sorted(calls) == sorted(aggregates)
    for rows in (_duckdb_rows, _sqlite_rows):
        assert rows(written) == rows(reading) and rows(written)


@pytest.mark.parametrize("written", LEFT_ALONE)
def test_a_query_that_is_not_read_stays_as_written(written):
    tree = sqlglot.parse_one(written, read="mysql")
    same, wrote = read_single_row_aggregates(tree, SCHEMA, KEYS, NOT_NULL, TYPES)
    assert not wrote and same is tree


def test_the_readings_refused_would_change_rows():
    # two rows of one shop cost 10: their sum is 20, their greatest 10
    by_shop = LEFT_ALONE[0]
    assert _duckdb_rows(by_shop) != _duckdb_rows(by_shop.replace("SUM(", "MAX("))
    # MySQL and SQLite read the text '7 apples' as the number 7 under SUM; MAX returns the text
    text = LEFT_ALONE[6]
    assert _sqlite_rows(text) != _sqlite_rows(text.replace("SUM(", "MAX("))
    # with no grouping every item has a row of the shop
    whole = "SELECT SUM(CASE WHEN shop = 'east' THEN cost END) AS east FROM prices"
    assert not read(whole)[1] and _duckdb_rows(whole) != _duckdb_rows(whole.replace("SUM(", "MAX("))


def test_a_key_is_needed_and_its_columns_must_be_not_null():
    written = READ[0][0]
    assert read(written)[1]
    assert not read(written, keys={"prices": []})[1]
    assert not read(written, keys={"prices": [("item", "shop", "cost")]})[1]  # the key needs a column nothing fixes
    assert not read(written, not_null={"prices": frozenset({"item"})})[1]
    assert not read(written, not_null=None)[1]
    assert read(written, types=None)[0].count("MAX(") == 0  # without types the comparison's kinds are not known
    # two rows with a NULL item are kept apart by no key: their sum is not their greatest
    rows = [(None, "north", 3, "a", 1.0), (None, "north", 4, "b", 1.0)]
    assert _duckdb_rows(written, rows) != _duckdb_rows(written.replace("SUM(", "MAX("), rows)


def test_the_parts_of_an_argument():
    north = sqlglot.parse_one("CASE WHEN shop = 'north' THEN cost END")
    condition, value = guarded(north)
    assert condition.sql() == "shop = 'north'" and value.sql() == "cost"
    condition, value = guarded(sqlglot.parse_one("CASE shop WHEN 'north' THEN cost ELSE NULL END"))
    assert condition.sql() == "shop = 'north'" and value.sql() == "cost"
    assert guarded(sqlglot.parse_one("IF(shop = 'north', cost, NULL)", read="mysql")) is not None
    assert guarded(sqlglot.parse_one("IF(shop = 'north', cost, 0)", read="mysql")) is None
    assert guarded(sqlglot.parse_one("CASE WHEN shop = 'north' THEN cost ELSE 0 END")) is None
    assert guarded(sqlglot.parse_one("cost")) is None
    table = Keyed.read(sqlglot.parse_one("SELECT 1 FROM prices AS p", read="mysql").find(sqlglot.exp.Table), SCHEMA, KEYS, NOT_NULL, TYPES)
    assert table.alias == "p" and table.keys == [("item", "shop")]
    assert table.pinned(sqlglot.parse_one("p.shop = 'north' AND 3 = item AND cost = 'x' AND note > 'a'")) .keys() == {"shop", "item"}
    assert table.pinned(sqlglot.parse_one("shop = 'north' AND shop = 'south'")) is None
    assert table.one_row({"item", "shop", "cost"}) and not table.one_row({"item", "cost"})
    assert Keyed.read(sqlglot.parse_one("SELECT 1 FROM other.prices", read="mysql").find(sqlglot.exp.Table), SCHEMA, KEYS, NOT_NULL, TYPES) is None


SUMS = "SELECT item, SUM(CASE WHEN shop = 'north' THEN cost END) AS north, SUM(CASE WHEN shop = 'south' THEN cost END) AS south FROM prices GROUP BY item"
GREATEST = "SELECT item, MAX(CASE WHEN shop = 'north' THEN cost END) AS north, MAX(IF(shop = 'south', cost, NULL)) AS south FROM prices GROUP BY 1"
LEAST = "SELECT p.item, MIN(CASE p.shop WHEN 'north' THEN p.cost END) AS north, MIN(CASE WHEN 'south' = p.shop THEN p.cost ELSE NULL END) AS south FROM prices AS p GROUP BY p.item"


@pytest.mark.parametrize("left, right", [(SUMS, GREATEST), (SUMS, LEAST), (GREATEST, LEAST)])
def test_the_three_spellings_are_proved_equal_and_only_with_the_rule(left, right):
    assert _duckdb_rows(left) == _duckdb_rows(right) and _sqlite_rows(left) == _sqlite_rows(right)
    assert prove(left, right).proven
    with shape_fallback.without("single_row_aggregates"):
        assert not prove(left, right).proven
    # without the key, or with a key column that may be NULL, the spellings are not the same
    assert not prove(left, right, {"prices": TableConstraints(not_null=NOT_NULL["prices"])}).proven
    assert not prove(left, right, {"prices": TableConstraints(not_null=frozenset({"shop"}), keys=(("item", "shop"),))}).proven


def test_spellings_that_differ_stay_unproved():
    assert not prove(SUMS, GREATEST.replace("'south'", "'east'")).proven
    assert not prove(SUMS, GREATEST.replace("THEN cost END", "THEN cost + 1 END")).proven
    by_shop = "SELECT shop, SUM(CASE WHEN cost = 10 THEN cost END) AS tens FROM prices GROUP BY shop"
    assert not prove(by_shop, by_shop.replace("SUM(", "MAX(")).proven
    whole = "SELECT SUM(CASE WHEN shop = 'east' THEN cost END) AS east FROM prices"
    assert not prove(whole, whole.replace("SUM(", "MAX(")).proven


def test_a_sum_is_read_only_where_it_is_used_like_its_one_value():
    # PostgreSQL's SUM of a BIGINT is a NUMERIC: SUM(x) / 2 keeps the half that MAX(x) / 2 drops
    summed = "SELECT item, SUM(CASE WHEN shop = 'north' THEN cost END) / 2 AS north FROM prices GROUP BY item"
    least = "SELECT item, MIN(CASE WHEN shop = 'north' THEN cost END) / 2 AS north FROM prices GROUP BY item"
    for dialect in ("postgres", "redshift", "tsql", "spark", ""):
        tree = sqlglot.parse_one(summed, read="mysql")
        assert read_single_row_aggregates(tree, SCHEMA, KEYS, NOT_NULL, TYPES, dialect) == (tree, False)
        assert read_single_row_aggregates(sqlglot.parse_one(least, read="mysql"), SCHEMA, KEYS, NOT_NULL, TYPES, dialect)[1]
    for dialect in ("mysql", "sqlite", "duckdb", "bigquery", "MySQL"):
        assert read_single_row_aggregates(sqlglot.parse_one(summed, read="mysql"), SCHEMA, KEYS, NOT_NULL, TYPES, dialect)[1]
    greatest = summed.replace("SUM(", "MAX(")
    assert prove(summed, greatest).proven
    assert not prove_equivalent_algebraic(summed, greatest, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect="postgres", exact_arithmetic=True, timeout_ms=5000).proven
