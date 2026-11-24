"""A grouping with no aggregate whose outputs the group fixes, read as ``SELECT DISTINCT`` (``key_fixed_groupings``).

Each reading is run as written and as read on SQLite over made-up rows (SQLite, like MySQL, accepts a select item
outside the ``GROUP BY``), and on DuckDB where DuckDB accepts the query as written.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.key_fixed_groupings import read_key_fixed_grouping
from kumo_sql_prover.smt_equivalence import TableConstraints

TABLES = {
    "shops": ("shop_id INT, city VARCHAR, region INT", [(1, "north", 7), (2, "north", 7), (3, "south", 8), (4, None, 8)]),
    "visits": ("visit_id INT, shop_id INT, day INT, guest VARCHAR", [(10, 1, 2, "ann"), (11, 1, 3, "bo"), (12, 2, 2, "ann"), (13, 3, 5, "N1"), (14, None, 2, "cy"), (15, None, 4, None), (16, 2, 2, "n1")]),
    "stock": ("shop_id INT, item INT, qty INT", [(1, 1, 5), (1, 2, 5), (2, 1, 0), (3, 2, 9)]),
    "guests": ("guest_id INT, name VARCHAR", [(1, "ann"), (None, "bo"), (None, "cy")]),
    "towns": ("code VARCHAR, name VARCHAR", [("n1", "north one"), ("N1", "north two"), ("s1", "south")]),
}
SCHEMA = {name: [part.split()[0] for part in columns.split(", ")] for name, (columns, _) in TABLES.items()}
TYPES = {name: dict(part.split() for part in columns.split(", ")) for name, (columns, _) in TABLES.items()}
KEYS = {"shops": [("shop_id",)], "visits": [("visit_id",)], "stock": [("shop_id", "item")], "guests": [("guest_id",)], "towns": [("code",)]}
NOT_NULL = {"shops": frozenset({"shop_id"}), "visits": frozenset({"visit_id"}), "stock": frozenset({"shop_id", "item"}), "guests": frozenset(), "towns": frozenset({"code"})}
CONSTRAINTS = {name: TableConstraints(not_null=NOT_NULL[name], keys=tuple(KEYS[name])) for name in TABLES}
JOINED = "FROM shops AS s JOIN visits AS v ON s.shop_id = v.shop_id"
LIST = "(SELECT shop_id FROM visits WHERE day = 3 AND shop_id > 0)"

# (as written, as read)
READ = [
    (f"SELECT s.shop_id, s.city {JOINED} GROUP BY s.shop_id", f"SELECT DISTINCT s.shop_id, s.city {JOINED}"),
    # the key the join equates with the table's
    (f"SELECT s.shop_id, s.city {JOINED} GROUP BY v.shop_id", f"SELECT DISTINCT s.shop_id, s.city {JOINED}"),
    (f"SELECT s.city, s.shop_id, s.region {JOINED} GROUP BY 2", f"SELECT DISTINCT s.city, s.shop_id, s.region {JOINED}"),
    (
        "SELECT s.shop_id, s.city FROM shops AS s, visits AS v WHERE s.shop_id = v.shop_id AND v.day > 2 GROUP BY v.shop_id",
        "SELECT DISTINCT s.shop_id, s.city FROM shops AS s, visits AS v WHERE s.shop_id = v.shop_id AND v.day > 2",
    ),
    # under a DISTINCT the keys need not be read off the outputs
    (f"SELECT DISTINCT s.city {JOINED} GROUP BY v.shop_id", f"SELECT DISTINCT s.city {JOINED}"),
    (f"SELECT s.shop_id, s.city {JOINED} WHERE s.shop_id NOT IN {LIST} GROUP BY s.shop_id", f"SELECT DISTINCT s.shop_id, s.city {JOINED} WHERE NOT s.shop_id IN {LIST}"),
    ("SELECT shop_id, city, region FROM shops GROUP BY shop_id", "SELECT DISTINCT shop_id, city, region FROM shops"),
    # a key the outputs fix through the table's key
    (f"SELECT s.shop_id {JOINED} GROUP BY v.shop_id, s.city", f"SELECT DISTINCT s.shop_id {JOINED}"),
    # outputs that are the keys: the NULL group is the NULL row
    ("SELECT v.shop_id, v.day FROM visits AS v GROUP BY v.shop_id, v.day", "SELECT DISTINCT v.shop_id, v.day FROM visits AS v"),
    ("SELECT v.day, v.shop_id FROM visits AS v GROUP BY 2, 1", "SELECT DISTINCT v.day, v.shop_id FROM visits AS v"),
    (
        "SELECT k.shop_id, k.item, k.qty FROM stock AS k JOIN visits AS v ON v.shop_id = k.shop_id GROUP BY k.shop_id, k.item",
        "SELECT DISTINCT k.shop_id, k.item, k.qty FROM stock AS k JOIN visits AS v ON v.shop_id = k.shop_id",
    ),
    # a bare name that is an output, and one that is a column and an output the rows equate with it
    (f"SELECT s.shop_id AS id, s.city AS town {JOINED} GROUP BY id", f"SELECT DISTINCT s.shop_id AS id, s.city AS town {JOINED}"),
    (f"SELECT v.shop_id AS visit_id, s.city {JOINED} AND v.visit_id = v.shop_id GROUP BY visit_id", f"SELECT DISTINCT v.shop_id AS visit_id, s.city {JOINED} AND v.visit_id = v.shop_id"),
    (
        "SELECT p.shop_id, s.shop_id AS own, s.region FROM (SELECT shop_id, day FROM visits WHERE day > 1) AS p JOIN shops AS s ON s.shop_id = p.shop_id GROUP BY s.shop_id, p.shop_id",
        "SELECT DISTINCT p.shop_id, s.shop_id AS own, s.region FROM (SELECT shop_id, day FROM visits WHERE day > 1) AS p JOIN shops AS s ON s.shop_id = p.shop_id",
    ),
]
ON_DUCKDB = {8, 9}  # the readings DuckDB accepts as written: every output is a key

LEFT_ALONE = [
    # not a plain grouping without aggregates
    f"SELECT s.shop_id, COUNT(*) AS n {JOINED} GROUP BY s.shop_id",
    f"SELECT s.shop_id, s.city {JOINED} GROUP BY s.shop_id HAVING COUNT(*) > 1",
    f"SELECT s.shop_id, s.city {JOINED} GROUP BY s.shop_id HAVING s.shop_id > 1",
    f"SELECT s.shop_id, s.city {JOINED} GROUP BY s.shop_id ORDER BY s.city LIMIT 1",
    f"SELECT s.shop_id, ROW_NUMBER() OVER (ORDER BY s.shop_id) AS r {JOINED} GROUP BY s.shop_id",
    f"SELECT s.shop_id, s.city {JOINED} GROUP BY s.shop_id WITH ROLLUP",
    # the join's test does not hold on every grouped row
    "SELECT s.shop_id, s.city FROM visits AS v LEFT JOIN shops AS s ON s.shop_id = v.shop_id GROUP BY v.shop_id",
    "SELECT s.shop_id, s.city FROM shops AS s JOIN visits AS v USING (shop_id) GROUP BY v.shop_id",
    "SELECT s.shop_id, s.city FROM shops AS s NATURAL JOIN visits AS v GROUP BY v.shop_id",
    "SELECT s.shop_id, s.city FROM shops AS s JOIN visits AS v ON s.shop_id >= v.shop_id GROUP BY v.shop_id",
    "SELECT s.shop_id, s.city FROM shops AS s JOIN visits AS v ON s.shop_id = v.shop_id OR v.day = 2 GROUP BY v.shop_id",
    # an output the group does not fix
    f"SELECT s.shop_id, v.day {JOINED} GROUP BY s.shop_id",
    f"SELECT s.shop_id, s.city {JOINED} GROUP BY v.day",
    f"SELECT s.shop_id + 0 AS shop_id, s.city {JOINED} GROUP BY s.shop_id",
    f"SELECT s.shop_id, s.city {JOINED} GROUP BY s.shop_id + 0",
    # a key the outputs do not fix: two shops of one city are two rows
    f"SELECT s.city {JOINED} GROUP BY s.shop_id",
    f"SELECT s.region, s.city {JOINED} GROUP BY v.shop_id",
    # a key that may be NULL, a key or an equated column that is a text
    "SELECT g.guest_id, g.name FROM guests AS g GROUP BY g.guest_id",
    "SELECT t.code, t.name FROM towns AS t GROUP BY t.code",
    "SELECT v.guest, t.code FROM visits AS v JOIN towns AS t ON t.code = v.guest GROUP BY v.guest",
    # a bare name that could mean two columns the rows do not equate
    "SELECT s.region AS shop_id, s.city FROM shops AS s GROUP BY shop_id",
    "SELECT s.shop_id, s.city FROM shops AS s JOIN visits AS v ON s.shop_id >= v.shop_id GROUP BY shop_id",
    f"SELECT s.shop_id AS shop_id, s.city AS town {JOINED} GROUP BY shop_id",
    # a derived table's column is not known to be a whole number: its key is not read off the table's
    "SELECT p.shop_id, s.region FROM (SELECT shop_id, day FROM visits WHERE day > 1) AS p JOIN shops AS s ON s.shop_id = p.shop_id GROUP BY s.shop_id, p.shop_id",
    # sources the declared columns do not place
    "SELECT s.shop_id, s.city FROM shops AS s JOIN elsewhere AS e ON e.shop_id = s.shop_id GROUP BY s.shop_id",
    "SELECT s.shop_id, s.city FROM shops AS s JOIN (SELECT * FROM visits) AS v ON s.shop_id = v.shop_id GROUP BY s.shop_id",
    f"SELECT s.shop_id, s.city, nowhere {JOINED} GROUP BY s.shop_id",
    f"SELECT s.*  {JOINED} GROUP BY s.shop_id",
]


def _rows(database, sql: str, dialect: str) -> list:
    return sorted(database.execute(sqlglot.transpile(sql, read="mysql", write=dialect)[0]).fetchall(), key=repr)


def _database(dialect: str):
    database = pytest.importorskip("duckdb").connect() if dialect == "duckdb" else sqlite3.connect(":memory:")
    for name, (columns, rows) in TABLES.items():
        database.execute(f"CREATE TABLE {name} ({columns})")
        database.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(rows[0]))})", rows)
    return database


def _read(sql: str, dialect: str = "mysql"):
    return read_key_fixed_grouping(sqlglot.parse_one(sql, read="mysql"), SCHEMA, KEYS, NOT_NULL, TYPES, dialect)


@pytest.mark.parametrize("index", range(len(READ)))
def test_a_reading_returns_the_rows_of_the_query_as_written(index):
    written, expected = READ[index]
    read = _read(written)
    assert read is not None and read.sql(dialect="mysql") == expected
    for dialect in ["sqlite"] + (["duckdb"] if index in ON_DUCKDB else []):
        database = _database(dialect)
        assert _rows(database, written, dialect) == _rows(database, expected, dialect), dialect


def test_the_made_up_rows_tell_the_cases_apart():
    database = _database("sqlite")
    assert _rows(database, READ[0][0], "sqlite") == [(1, "north"), (2, "north"), (3, "south")]
    assert _rows(database, READ[4][0], "sqlite") == [("north",), ("south",)]
    assert _rows(database, READ[5][0], "sqlite") == [(2, "north"), (3, "south")]
    assert (None, 2) in _rows(database, READ[8][0], "sqlite")
    assert len(_rows(database, f"SELECT s.shop_id {JOINED}", "sqlite")) == 5  # the join repeats a shop


@pytest.mark.parametrize("written", LEFT_ALONE)
def test_a_query_outside_the_reading_stays_as_written(written):
    assert _read(written) is None


@pytest.mark.parametrize("dialect", ["bigquery", "duckdb", "snowflake", None])
def test_only_dialects_that_accept_an_output_outside_the_grouping_are_read(dialect):
    assert _read(READ[0][0], dialect) is None
    assert _read(READ[0][0], "sqlite") is not None and _read(READ[0][0], "postgres") is not None


def test_nothing_is_read_without_declared_keys():
    tree = sqlglot.parse_one(READ[0][0], read="mysql")
    assert read_key_fixed_grouping(tree, SCHEMA, None, NOT_NULL, TYPES, "mysql") is None
    assert read_key_fixed_grouping(tree, SCHEMA, KEYS, None, TYPES, "mysql") is None  # the key may be NULL
    assert read_key_fixed_grouping(tree, SCHEMA, KEYS, NOT_NULL, None, "mysql") is None  # the key may be a text
    assert read_key_fixed_grouping(tree, None, KEYS, NOT_NULL, TYPES, "mysql") is None


def test_the_select_it_is_given_is_not_changed():
    tree = sqlglot.parse_one(READ[0][0], read="mysql")
    before = tree.sql()
    read = read_key_fixed_grouping(tree, SCHEMA, KEYS, NOT_NULL, TYPES, "mysql")
    assert read is not None and read is not tree and tree.sql() == before


# (a query left alone, the DISTINCT a careless reading would write, rows of each on SQLite)
BY_HAND = [
    (f"SELECT s.city {JOINED} GROUP BY s.shop_id", f"SELECT DISTINCT s.city {JOINED}", 3, 2),  # two shops of one city
    ("SELECT g.guest_id FROM guests AS g GROUP BY g.guest_id, g.name", "SELECT DISTINCT g.guest_id FROM guests AS g", 3, 2),  # two guests without an id
    ("SELECT t.name FROM visits AS v JOIN towns AS t ON t.code = v.guest COLLATE NOCASE GROUP BY v.guest COLLATE NOCASE", "SELECT DISTINCT t.name FROM visits AS v JOIN towns AS t ON t.code = v.guest COLLATE NOCASE", 1, 2),
]


@pytest.mark.parametrize("written, by_hand, grouped, distinct", BY_HAND)
def test_outside_the_reading_the_distinct_by_hand_changes_rows(written, by_hand, grouped, distinct):
    database = _database("sqlite")
    assert len(database.execute(written).fetchall()) == grouped
    assert len(database.execute(by_hand).fetchall()) == distinct
    if "COLLATE" not in written:
        assert _read(written) is None


# shops all of whose visits fall in one week: the bounds of a group against two lists, the second spelling grouped
OPTIONS = dict(
    schema={**SCHEMA, "calls": ["call_id", "shop_id", "seen"]},
    types={**TYPES, "calls": {"call_id": "INT", "shop_id": "INT", "seen": "DATE"}},
    constraints={**CONSTRAINTS, "calls": TableConstraints(not_null=frozenset({"call_id", "shop_id", "seen"}), keys=(("call_id",),))},
    compare_names=False,
    dialect="mysql",
)
BOUNDS = "SELECT s.shop_id, s.city FROM shops AS s JOIN (SELECT shop_id FROM calls GROUP BY shop_id HAVING MIN(seen) >= '2024-01-01' AND MAX(seen) <= '2024-01-07') AS v ON s.shop_id = v.shop_id"
LISTS = (
    "SELECT s.shop_id, s.city FROM shops AS s JOIN calls AS v ON s.shop_id = v.shop_id WHERE s.shop_id IN (SELECT shop_id FROM calls WHERE seen BETWEEN '2024-01-01' AND '2024-01-{}')"
    " AND s.shop_id NOT IN (SELECT shop_id FROM calls WHERE seen NOT BETWEEN '2024-01-01' AND '2024-01-07') GROUP BY {}"
)


@pytest.mark.parametrize("key", ["s.shop_id", "v.shop_id", "1"])
def test_a_pair_is_proved_and_only_with_the_rule(key):
    assert prove_equivalent_algebraic(BOUNDS, LISTS.format("07", key), **OPTIONS).proven
    with shape_fallback.without("key_fixed_groupings"):
        assert not prove_equivalent_algebraic(BOUNDS, LISTS.format("07", key), **OPTIONS).proven


def test_pairs_that_differ_stay_unproved():
    assert not prove_equivalent_algebraic(BOUNDS, LISTS.format("06", "s.shop_id"), **OPTIONS).proven
    assert not prove_equivalent_algebraic(BOUNDS, LISTS.format("07", "v.seen"), **OPTIONS).proven
    assert not prove_equivalent_algebraic(BOUNDS.replace("s.city", "s.region"), LISTS.format("07", "s.shop_id"), **OPTIONS).proven
