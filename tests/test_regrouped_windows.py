"""An aggregate over the groups of a partition read as a join with the table grouped again (``regrouped_windows``).

Each reading is run as written and as read on DuckDB and on SQLite over made-up tables: a shop with three items,
shops with one, orders with no shop (their groups form a partition of their own for the window). The rows must be
the same. Queries the rule leaves alone come, where a reading by hand would change rows, with those rows.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.regrouped_windows import read_regrouped_windows
from kumo_sql_prover.smt_equivalence import TableConstraints

TABLES = {
    "orders": (
        "id INT, shop INT, item INT, price INT, day INT",
        [(1, 1, 10, 100, 1), (2, 1, 10, 50, 2), (3, 1, 11, 7, 2), (4, 1, 12, 20, 1), (5, 2, 10, 30, 3), (6, None, 10, 9, 3), (7, None, 11, 2, 1), (8, 3, 12, 15, 3), (9, 3, 13, 400, 3), (10, 9, 13, 1, 2)],
    ),
    "shops": ("id INT, name VARCHAR, city INT", [(1, "a", 1), (2, "b", 1), (3, "c", 2), (4, "d", 2)]),
}
SCHEMA = {name: [part.split()[0] for part in columns.split(", ")] for name, (columns, _) in TABLES.items()}
TYPES = {name: dict(part.split() for part in columns.split(", ")) for name, (columns, _) in TABLES.items()}
CONSTRAINTS = {name: TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)) for name in TABLES}

GROUPED = "FROM orders GROUP BY shop, item"
T = f"(SELECT shop, item, SUM(price) AS total, AVG(SUM(price)) OVER (PARTITION BY shop) AS mean {GROUPED}) AS t"

READ = [
    f"SELECT s.name, t.item, t.total FROM shops AS s, {T} WHERE s.id = t.shop AND t.total <= 0.5 * t.mean",
    f"SELECT s.name, t.item, t.mean FROM {T} JOIN shops AS s ON t.shop = s.id",
    # an unnamed item keeps its name, and the value is read in every clause
    f"SELECT t.item, t.mean FROM {T} JOIN shops AS s ON t.shop = s.id WHERE t.mean > 10 OR t.item = 13 ORDER BY t.mean, t.item",
    # several windows of one partition, COUNT(*), a grouping column as the argument, HAVING
    "SELECT t.shop, t.item, t.n, t.groups, t.top FROM (SELECT shop, item, COUNT(*) AS n, COUNT(*) OVER (PARTITION BY shop) AS groups, MAX(item) OVER (PARTITION BY shop) AS top "
    "FROM orders GROUP BY shop, item HAVING SUM(price) > 5) AS t JOIN shops AS s ON t.shop = s.id",
    # a partition of two columns, tied to two other columns
    "SELECT o.id, t.most FROM orders AS o JOIN (SELECT shop, item, day, MAX(COUNT(*)) OVER (PARTITION BY shop, item) AS most FROM orders GROUP BY shop, item, day) AS t ON o.shop = t.shop AND o.item = t.item AND o.day = t.day",
    # the partition column compared with a constant
    f"SELECT t.item, t.total, t.mean FROM {T} WHERE t.shop = 1",
    # a body with a join and a filter
    "SELECT t.name, t.item, t.share FROM (SELECT s.name, o.shop, o.item, MIN(SUM(o.price)) OVER (PARTITION BY o.shop) AS share FROM orders AS o JOIN shops AS s ON s.id = o.shop WHERE o.price > 1 GROUP BY s.name, o.shop, o.item) AS t, shops AS c WHERE c.id = t.shop",
    # the select around is grouped, and the window's value is aggregated
    f"SELECT s.city, MAX(t.mean) AS best, COUNT(*) AS n FROM {T}, shops AS s WHERE s.id = t.shop GROUP BY s.city",
    # inside a derived table
    f"SELECT z.item FROM (SELECT t.item, t.mean FROM shops AS s, {T} WHERE s.id = t.shop) AS z WHERE z.mean > 20",
]

LEFT_ALONE = [
    # nothing rejects a NULL partition value: its groups form a partition for the window and match no group in a join
    f"SELECT t.item, t.total, t.mean FROM {T}",
    f"SELECT t.item, t.total, t.mean FROM {T} WHERE t.item = 10",
    f"SELECT t.item, t.mean FROM {T} JOIN shops AS s ON t.shop = s.id OR t.item = s.id",
    f"SELECT t.item, t.mean FROM {T} JOIN shops AS s ON t.shop >= s.id",
    f"SELECT t.item, t.mean FROM {T} JOIN shops AS s ON s.id = t.item WHERE t.shop IS NOT DISTINCT FROM s.city",
    # an outer join around
    f"SELECT s.name, t.item, t.mean FROM shops AS s LEFT JOIN {T} ON s.id = t.shop",
    f"SELECT s.name, t.item, t.mean FROM {T} JOIN shops AS s USING (item)",
    # the columns of the derived table are listed by a star, or the window's name may be meant by a bare name
    f"SELECT * FROM shops AS s, {T} WHERE s.id = t.shop",
    f"SELECT t.* FROM shops AS s, {T} WHERE s.id = t.shop",
    f"SELECT s.name, mean FROM shops AS s, {T} WHERE s.id = t.shop",
    # a second source of the name in a subquery
    f"SELECT s.name, t.mean FROM shops AS s, {T} WHERE s.id = t.shop AND EXISTS (SELECT 1 FROM orders AS t WHERE t.shop = s.id)",
    # windows the rule does not read
    "SELECT t.item, t.r FROM (SELECT shop, item, RANK() OVER (PARTITION BY shop ORDER BY SUM(price)) AS r FROM orders GROUP BY shop, item) AS t JOIN shops AS s ON t.shop = s.id",
    "SELECT t.item, t.m FROM (SELECT shop, item, AVG(SUM(price)) OVER (PARTITION BY shop ORDER BY item) AS m FROM orders GROUP BY shop, item) AS t JOIN shops AS s ON t.shop = s.id",
    "SELECT t.item, t.m FROM (SELECT shop, item, AVG(SUM(price)) OVER () AS m FROM orders GROUP BY shop, item) AS t JOIN shops AS s ON t.shop = s.id",
    "SELECT t.item, t.m FROM (SELECT shop, item, 2 * AVG(SUM(price)) OVER (PARTITION BY shop) AS m FROM orders GROUP BY shop, item) AS t JOIN shops AS s ON t.shop = s.id",
    "SELECT t.item, t.m FROM (SELECT shop, item, COUNT(DISTINCT item) OVER (PARTITION BY shop) AS m FROM orders GROUP BY shop, item) AS t JOIN shops AS s ON t.shop = s.id",
    "SELECT t.item, t.m FROM (SELECT shop, item, AVG(SUM(price)) OVER (PARTITION BY shop) AS m, SUM(SUM(price)) OVER (PARTITION BY shop, item) AS n FROM orders GROUP BY shop, item) AS t JOIN shops AS s ON t.shop = s.id",
    "SELECT t.item, t.m FROM (SELECT shop, item, AVG(SUM(price)) OVER w AS m FROM orders GROUP BY shop, item WINDOW w AS (PARTITION BY shop)) AS t JOIN shops AS s ON t.shop = s.id",
    # the partition is not a grouping column listed under a name
    "SELECT t.item, t.m FROM (SELECT item, AVG(SUM(price)) OVER (PARTITION BY shop) AS m FROM orders GROUP BY shop, item) AS t JOIN shops AS s ON t.item = s.id",
    "SELECT t.item, t.m FROM (SELECT shop + 0 AS shop, item, AVG(SUM(price)) OVER (PARTITION BY shop) AS m FROM orders GROUP BY shop, item) AS t JOIN shops AS s ON t.shop = s.id",
    # bodies that are not a plain grouped select
    "SELECT t.id, t.m FROM (SELECT id, shop, AVG(price) OVER (PARTITION BY shop) AS m FROM orders) AS t JOIN shops AS s ON t.shop = s.id",
    "SELECT t.item, t.m FROM (SELECT DISTINCT shop, item, AVG(SUM(price)) OVER (PARTITION BY shop) AS m FROM orders GROUP BY shop, item) AS t JOIN shops AS s ON t.shop = s.id",
    "SELECT t.item, t.m FROM (SELECT shop, item, AVG(SUM(price)) OVER (PARTITION BY shop) AS m FROM orders GROUP BY shop, item ORDER BY item LIMIT 3) AS t JOIN shops AS s ON t.shop = s.id",
    "SELECT t.item, t.m FROM (SELECT shop, item, AVG(SUM(price)) OVER (PARTITION BY shop) AS m FROM orders GROUP BY ROLLUP (shop, item)) AS t JOIN shops AS s ON t.shop = s.id",
    "SELECT t.m FROM (SELECT AVG(SUM(price)) OVER (PARTITION BY shop) AS m FROM orders GROUP BY shop, item) AS t",
]

# (as written, read by hand the way the rule would): the rows differ, so the refusal is needed
MEANS = "(SELECT x.shop, AVG(x.total) AS mean FROM (SELECT shop, SUM(price) AS total FROM orders GROUP BY shop, item) AS x GROUP BY x.shop) AS m"
PLAIN = f"(SELECT shop, item, SUM(price) AS total {GROUPED}) AS t"
BY_HAND = [
    (LEFT_ALONE[0], f"SELECT t.item, t.total, m.mean FROM {PLAIN}, {MEANS} WHERE t.shop = m.shop"),
    (LEFT_ALONE[1], f"SELECT t.item, t.total, m.mean FROM {PLAIN}, {MEANS} WHERE t.item = 10 AND t.shop = m.shop"),
    (LEFT_ALONE[5], f"SELECT s.name, t.item, m.mean FROM shops AS s LEFT JOIN {PLAIN} ON s.id = t.shop, {MEANS} WHERE t.shop = m.shop"),
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


def _read(sql: str) -> tuple[str, bool]:
    tree, wrote = read_regrouped_windows(sqlglot.parse_one(sql, read="postgres"))
    return tree.sql(dialect="postgres"), wrote


@pytest.mark.parametrize("written", READ)
def test_a_reading_returns_the_rows_of_the_query_as_written(written):
    read, wrote = _read(written)
    assert wrote and " OVER " not in read, read
    for dialect, database in _databases():
        rows = _rows(database, written, dialect)
        assert rows == _rows(database, read, dialect), dialect
        assert rows, "the made-up rows do not reach this query"
        columns = [[d[0].lower() for d in database.execute(sqlglot.transpile(sql, read="postgres", write=dialect)[0]).description] for sql in (written, read)]
        assert columns[0] == columns[1], dialect


@pytest.mark.parametrize("written", LEFT_ALONE)
def test_a_query_that_is_not_read_stays_as_written(written):
    read, wrote = _read(written)
    assert not wrote and read == sqlglot.parse_one(written, read="postgres").sql(dialect="postgres")


@pytest.mark.parametrize("written, by_hand", BY_HAND)
def test_reading_a_refused_query_by_hand_changes_rows(written, by_hand):
    assert not _read(written)[1]
    database = next(db for dialect, db in _databases() if dialect == "duckdb")
    assert _rows(database, written, "duckdb") != _rows(database, by_hand, "duckdb")


def test_the_tree_it_is_given_is_not_changed():
    tree = sqlglot.parse_one(READ[0], read="postgres")
    before = tree.sql()
    read, wrote = read_regrouped_windows(tree)
    assert wrote and read is not tree and tree.sql() == before


def _proved(left: str, right: str) -> bool:
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect="postgres").proven


WINDOWED = f"SELECT s.name, t.item, t.total FROM shops AS s, {T} WHERE s.id = t.shop AND t.total <= 0.5 * t.mean"
JOINED = f"SELECT s.name, t.item, t.total FROM shops AS s, {PLAIN}, {MEANS} WHERE s.id = t.shop AND t.shop = m.shop AND t.total <= 0.5 * m.mean"


def test_a_pair_is_proved_and_only_with_the_rule():
    assert _proved(WINDOWED, JOINED)
    with shape_fallback.without("regrouped_windows"):
        assert not _proved(WINDOWED, JOINED)


def test_pairs_that_differ_stay_unproved():
    # the mean of the shop's orders, not of its items' totals
    assert not _proved(WINDOWED, JOINED.replace("AVG(x.total) AS mean FROM (SELECT shop, SUM(price) AS total FROM orders GROUP BY shop, item) AS x", "AVG(x.price) AS mean FROM orders AS x"))
    # the mean over every shop
    assert not _proved(WINDOWED.replace("PARTITION BY shop", "PARTITION BY item"), JOINED)
