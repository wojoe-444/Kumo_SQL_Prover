"""A column a one-source select names as ``x`` and as ``c.x``, written bare (``one_source_names``).

Each reading is run as written and as read on DuckDB and on SQLite over made-up rows.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.one_source_names import read_one_source_names
from kumo_sql_prover.smt_equivalence import TableConstraints

TABLES = {
    "views": ("viewer INT, day INT, article INT", [(1, 1, 10), (1, 1, 11), (1, 2, 10), (2, 1, 10), (3, 2, 12), (3, 2, 13), (None, 2, 14)]),
    "shops": ("id INT, city VARCHAR", [(1, "north"), (2, "north"), (3, "south"), (4, None)]),
}
SCHEMA = {name: [part.split()[0] for part in columns.split(", ")] for name, (columns, _) in TABLES.items()}
TYPES = {name: dict(part.split() for part in columns.split(", ")) for name, (columns, _) in TABLES.items()}
CONSTRAINTS = {"views": TableConstraints(), "shops": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),))}

# (as written, as read)
READ = [
    ("SELECT DISTINCT viewer FROM views AS v GROUP BY v.viewer", "SELECT DISTINCT viewer FROM views AS v GROUP BY viewer"),
    ("SELECT v.viewer, COUNT(*) AS n FROM views AS v WHERE viewer > 1 GROUP BY v.viewer", "SELECT viewer, COUNT(*) AS n FROM views AS v WHERE viewer > 1 GROUP BY viewer"),
    ("SELECT views.day, article FROM views WHERE views.article > 10 AND day = 2", "SELECT day, article FROM views WHERE article > 10 AND day = 2"),
    # only the name written both ways: v.article stays
    ("SELECT v.viewer, v.article FROM views AS v WHERE viewer = 3", "SELECT viewer, v.article FROM views AS v WHERE viewer = 3"),
    ("SELECT p.viewer, viewer + 1 AS nxt FROM (SELECT viewer, day FROM views) AS p ORDER BY p.viewer", "SELECT viewer, viewer + 1 AS nxt FROM (SELECT viewer, day FROM views) AS p ORDER BY viewer"),
    # an alias of the column itself: either meaning of the name is the column
    ("SELECT v.viewer AS viewer, MAX(day) AS last FROM views AS v GROUP BY viewer HAVING MAX(v.day) > 1", "SELECT viewer AS viewer, MAX(day) AS last FROM views AS v GROUP BY viewer HAVING MAX(day) > 1"),
    # a read inside a nested query is that query's own
    (
        "SELECT viewer FROM views AS v WHERE v.viewer IN (SELECT s.id FROM shops AS s WHERE s.id = v.viewer)",
        "SELECT viewer FROM views AS v WHERE viewer IN (SELECT s.id FROM shops AS s WHERE s.id = v.viewer)",
    ),
    ("SELECT id FROM (SELECT v.viewer AS id, viewer + day AS total FROM views AS v) AS q", "SELECT id FROM (SELECT viewer AS id, viewer + day AS total FROM views AS v) AS q"),
    (
        "SELECT viewer, day, ROW_NUMBER() OVER (PARTITION BY v.viewer ORDER BY v.day, article) AS r, article FROM views AS v",
        "SELECT viewer, day, ROW_NUMBER() OVER (PARTITION BY viewer ORDER BY day, article) AS r, article FROM views AS v",
    ),
]

LEFT_ALONE = [
    "SELECT viewer, day FROM views AS v WHERE article > 10",
    "SELECT v.viewer, v.day FROM views AS v WHERE v.article > 10",
    "SELECT v.viewer, day FROM views AS v",  # no name is written both ways
    # more than one source
    "SELECT v.day, day FROM views AS v JOIN shops AS s ON s.id = v.viewer",
    "SELECT v.day, day FROM views AS v, shops AS s",
    "SELECT v.day, day FROM views AS v, LATERAL (SELECT id FROM shops WHERE id = v.viewer) AS s",
    # GROUP BY, HAVING or ORDER BY could read the bare name as the other output
    "SELECT day AS viewer, MAX(viewer) AS m FROM views AS v GROUP BY v.viewer",
    "SELECT day AS viewer, article FROM views AS v WHERE viewer IS NOT NULL ORDER BY v.viewer DESC, article LIMIT 1",
    "SELECT viewer + 1 AS viewer FROM views AS v ORDER BY v.viewer",
    # a column named like its source
    "SELECT v.v, v FROM (SELECT viewer AS v FROM views) AS v",
    # not a plain column of the source
    "SELECT v.*, viewer FROM views AS v",
    "SELECT main.views.viewer, viewer FROM main.views",
    # the qualified read is a nested query's
    "SELECT viewer FROM views AS v WHERE EXISTS (SELECT 1 FROM shops AS s WHERE s.id = v.viewer)",
    "SELECT viewer, (SELECT MAX(s.id) FROM shops AS s WHERE s.id < v.viewer) AS below FROM views AS v",
    # another table's name
    "SELECT id FROM shops AS s WHERE EXISTS (SELECT 1 FROM views AS v WHERE viewer = s.id AND s.city = 'north')",
]


def _rows(database, sql: str, dialect: str) -> list:
    return sorted(database.execute(sqlglot.transpile(sql, read="mysql", write=dialect)[0]).fetchall(), key=repr)


def _database(dialect: str):
    database = pytest.importorskip("duckdb").connect() if dialect == "duckdb" else sqlite3.connect(":memory:")
    for name, (columns, rows) in TABLES.items():
        database.execute(f"CREATE TABLE {name} ({columns})")
        database.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(rows[0]))})", rows)
    return database


@pytest.mark.parametrize("written, expected", READ)
def test_a_reading_returns_the_rows_of_the_query_as_written(written, expected):
    tree, wrote = read_one_source_names(sqlglot.parse_one(written, read="mysql"))
    read = tree.sql(dialect="mysql")
    assert wrote and read == expected
    for dialect in ("duckdb", "sqlite"):
        database = _database(dialect)
        assert _rows(database, written, dialect) == _rows(database, read, dialect), dialect


@pytest.mark.parametrize("written", LEFT_ALONE)
def test_a_query_outside_the_reading_stays_as_written(written):
    given = sqlglot.parse_one(written, read="mysql")
    tree, wrote = read_one_source_names(given)
    assert not wrote and tree is given


def test_the_tree_it_is_given_is_not_changed():
    tree = sqlglot.parse_one(READ[0][0], read="mysql")
    before = tree.sql()
    read, wrote = read_one_source_names(tree)
    assert wrote and read is not tree and tree.sql() == before


@pytest.mark.parametrize("dialect", ["duckdb", "sqlite"])
def test_a_name_that_is_also_another_output_changes_rows_when_written_bare_by_hand(dialect):
    written = LEFT_ALONE[7]
    by_hand = written.replace("ORDER BY v.viewer", "ORDER BY viewer")
    assert by_hand != written
    database = _database(dialect)
    assert _rows(database, written, dialect) == [(2, 12)]
    assert _rows(database, by_hand, dialect) == [(2, 10)]  # ORDER BY reads the output named viewer, which is the day


def _proved(left: str, right: str) -> bool:
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect="mysql").proven


# shops.id is a key: the rules that read a grouping by a key compare the select list with the grouping as written
SOLD = "(SELECT COUNT(DISTINCT viewer) FROM views)"
BARE = "SELECT id FROM shops GROUP BY id HAVING COUNT(DISTINCT city) = " + SOLD
MIXED = "SELECT DISTINCT id FROM shops AS s GROUP BY s.id HAVING COUNT(DISTINCT {}) {} " + SOLD


def test_a_pair_is_proved_and_only_with_the_rule():
    assert _proved(BARE, MIXED.format("city", "="))
    with shape_fallback.without("one_source_names"):
        assert not _proved(BARE, MIXED.format("city", "="))


def test_pairs_that_differ_stay_unproved():
    assert not _proved(BARE, MIXED.format("city", ">"))
    assert not _proved(BARE, MIXED.format("id", "="))
