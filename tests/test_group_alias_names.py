"""A ``GROUP BY`` name that only the select list defines, written as its expression (``group_alias_names``).

Each reading is run as written and as read on DuckDB and on SQLite over made-up rows.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.group_alias_names import read_group_alias_names
from kumo_sql_prover.smt_equivalence import TableConstraints

TABLES = {
    "views": ("viewer INT, day INT, article INT", [(1, 1, 10), (1, 1, 11), (1, 2, 10), (2, 1, 10), (3, 2, 12), (3, 2, 13), (None, 2, 14)]),
    "shops": ("id INT, city VARCHAR", [(1, "north"), (2, "north"), (3, "south"), (4, None)]),
}
SCHEMA = {name: [part.split()[0] for part in columns.split(", ")] for name, (columns, _) in TABLES.items()}
TYPES = {name: dict(part.split() for part in columns.split(", ")) for name, (columns, _) in TABLES.items()}
CONSTRAINTS = {"views": TableConstraints(), "shops": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),))}
PER_DAY = "(SELECT viewer, day FROM views GROUP BY viewer, day) AS p"

# (as written, the GROUP BY of the reading)
READ = [
    (f"SELECT viewer AS id FROM {PER_DAY} GROUP BY id", "GROUP BY viewer"),
    (f"SELECT viewer + 1 AS nxt, COUNT(*) AS n FROM {PER_DAY} GROUP BY nxt", "GROUP BY viewer + 1"),
    ("SELECT viewer AS v, day AS d, COUNT(*) AS n FROM (SELECT viewer, day, article FROM views) AS p GROUP BY v, d", "GROUP BY viewer, day"),
    ("SELECT p.viewer AS v, day, COUNT(*) AS n FROM (SELECT viewer, day, article FROM views) AS p GROUP BY v, day", "GROUP BY p.viewer, day"),
    ("SELECT viewer AS id, COUNT(*) AS n FROM views GROUP BY id", "GROUP BY viewer"),
    (f"SELECT s.city AS town, COUNT(*) AS n FROM {PER_DAY} JOIN shops AS s ON s.id = p.viewer GROUP BY town", "GROUP BY s.city"),
    (f"SELECT id FROM (SELECT viewer AS id FROM {PER_DAY} GROUP BY id) AS q WHERE id > 1", "GROUP BY viewer"),
    (f"SELECT viewer AS id FROM {PER_DAY} GROUP BY id UNION ALL SELECT id FROM shops", "GROUP BY viewer"),
    ("SELECT CASE WHEN article > 10 THEN 'big' ELSE 'small' END AS size, COUNT(*) AS n FROM (SELECT article FROM views) AS p GROUP BY size", "GROUP BY CASE WHEN article > 10 THEN 'big' ELSE 'small' END"),
    (f"SELECT viewer AS id, COUNT(*) AS n FROM {PER_DAY} GROUP BY id HAVING COUNT(*) > 1", "GROUP BY viewer"),
]

LEFT_ALONE = [
    # the name is a column of a source: the engines read the column, and they do not all say so
    f"SELECT MAX(day) AS viewer, COUNT(*) AS n FROM {PER_DAY} GROUP BY viewer",
    f"SELECT s.id AS viewer, COUNT(*) AS n FROM {PER_DAY} JOIN shops AS s ON s.id = p.viewer GROUP BY viewer",
    # a source whose columns are not written out
    "SELECT viewer AS id FROM (SELECT * FROM views) AS p GROUP BY id",
    "SELECT a AS id FROM elsewhere GROUP BY id",
    f"SELECT viewer AS id FROM {PER_DAY}, (SELECT * FROM shops) AS s GROUP BY id",
    # the name is not one item's alias
    f"SELECT viewer AS id, day AS id FROM {PER_DAY} GROUP BY id",
    f"SELECT viewer AS v FROM {PER_DAY} GROUP BY id",
    f"SELECT viewer AS id FROM {PER_DAY} GROUP BY p.id",
    f"SELECT viewer AS `id` FROM {PER_DAY} GROUP BY id",
    # what the name stands for cannot be a grouping
    f"SELECT 1 AS one, COUNT(*) AS n FROM {PER_DAY} GROUP BY one",
    f"SELECT 'x' AS label, COUNT(*) AS n FROM {PER_DAY} GROUP BY label",
    f"SELECT COUNT(*) AS n FROM {PER_DAY} GROUP BY n",
    f"SELECT (SELECT MAX(id) FROM shops) + viewer AS m FROM {PER_DAY} GROUP BY m",
    f"SELECT viewer + RAND() AS r FROM {PER_DAY} GROUP BY r",
    f"SELECT ROW_NUMBER() OVER (ORDER BY viewer) AS r FROM {PER_DAY} GROUP BY r",
    # the expression reads a bare name that is another item's alias
    f"SELECT viewer AS day, day + 1 AS nxt FROM {PER_DAY} GROUP BY nxt",
    # another output without a name of its own
    f"SELECT viewer AS id, viewer + 1 FROM {PER_DAY} GROUP BY id",
    f"SELECT viewer AS id, COUNT(*) FROM {PER_DAY} GROUP BY id",
    f"SELECT viewer AS id, p.* FROM {PER_DAY} GROUP BY id",
    # more than one grouping
    f"SELECT viewer AS id FROM {PER_DAY} GROUP BY id WITH ROLLUP",
    # an enclosing query could give the name a meaning
    f"SELECT id FROM shops WHERE id IN (SELECT viewer AS v FROM {PER_DAY} GROUP BY v)",
    f"SELECT id, (SELECT MAX(viewer) AS v FROM {PER_DAY} GROUP BY v LIMIT 1) AS top FROM shops",
    f"SELECT s.id, q.v FROM shops AS s, LATERAL (SELECT viewer AS v FROM {PER_DAY} WHERE p.viewer = s.id GROUP BY v) AS q",
    f"WITH q AS (SELECT viewer AS v FROM {PER_DAY} GROUP BY v) SELECT v FROM q",
    # no GROUP BY name at all
    f"SELECT viewer AS id FROM {PER_DAY} GROUP BY viewer",
    f"SELECT viewer AS id FROM {PER_DAY} GROUP BY 1",
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


def _read(sql: str, dialect: str = "mysql") -> tuple:
    return read_group_alias_names(sqlglot.parse_one(sql, read="mysql"), SCHEMA, dialect)


@pytest.mark.parametrize("written, grouping", READ)
def test_a_reading_returns_the_rows_of_the_query_as_written(written, grouping):
    tree, wrote = _read(written)
    read = tree.sql(dialect="mysql")
    assert wrote and grouping in read and read != written, read
    for dialect, database in _databases():
        assert _rows(database, written, dialect) == _rows(database, read, dialect), dialect


@pytest.mark.parametrize("written", LEFT_ALONE)
def test_a_query_outside_the_reading_stays_as_written(written):
    given = sqlglot.parse_one(written, read="mysql")
    tree, wrote = read_group_alias_names(given, SCHEMA, "mysql")
    assert not wrote and tree is given


def test_only_the_four_dialects_and_only_with_declared_columns():
    written = READ[0][0]
    assert all(_read(written, dialect)[1] for dialect in ("mysql", "postgres", "sqlite", "duckdb"))
    assert not any(_read(written, dialect)[1] for dialect in ("bigquery", "tsql", "oracle", "snowflake", ""))
    assert not read_group_alias_names(sqlglot.parse_one(written, read="mysql"), None, "mysql")[1]
    assert not read_group_alias_names(sqlglot.parse_one(READ[4][0], read="mysql"), {"shops": SCHEMA["shops"]}, "mysql")[1]


def test_the_tree_it_is_given_is_not_changed():
    tree = sqlglot.parse_one(READ[0][0], read="mysql")
    before = tree.sql()
    read, wrote = read_group_alias_names(tree, SCHEMA, "mysql")
    assert wrote and read is not tree and tree.sql() == before


def test_a_name_a_source_also_has_is_the_column_so_the_alias_by_hand_changes_rows():
    # LEFT_ALONE[0] with a bare column in place of MAX: SQLite groups by the source's viewer, not by the output named viewer
    written = f"SELECT day AS viewer, COUNT(*) AS n FROM {PER_DAY} GROUP BY viewer"
    by_hand = f"SELECT day AS viewer, COUNT(*) AS n FROM {PER_DAY} GROUP BY day"
    assert not _read(written)[1]
    database = sqlite3.connect(":memory:")
    for name, (columns, rows) in TABLES.items():
        database.execute(f"CREATE TABLE {name} ({columns})")
        database.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(rows[0]))})", rows)
    assert _rows(database, written, "sqlite") != _rows(database, by_hand, "sqlite")


def _proved(left: str, right: str) -> bool:
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect="mysql").proven


FLAT = "SELECT DISTINCT viewer AS id FROM views GROUP BY day, viewer HAVING COUNT(DISTINCT article) > 1"
STAGED = "SELECT viewer AS id FROM (SELECT viewer FROM views GROUP BY viewer, day HAVING COUNT(DISTINCT article) > {}) AS p GROUP BY id"


def test_a_pair_is_proved_and_only_with_the_rule():
    assert _proved(FLAT, STAGED.format(1))
    # key_fixed_groupings (batch 30) reads the staged spelling too, as SELECT DISTINCT of its output
    with shape_fallback.without_all(frozenset({"group_alias_names", "key_fixed_groupings"})):
        assert not _proved(FLAT, STAGED.format(1))
    for name in ("group_alias_names", "key_fixed_groupings"):
        with shape_fallback.without(name):
            assert _proved(FLAT, STAGED.format(1)), name


def test_pairs_that_differ_stay_unproved():
    assert not _proved(FLAT, STAGED.format(2))
    assert not _proved(FLAT, STAGED.format(1).replace("GROUP BY viewer, day", "GROUP BY viewer"))
