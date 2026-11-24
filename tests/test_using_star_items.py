"""``USING`` read as ``ON``: the select items beside a star stay, and ``t.*`` is left as it is written.

A select of ``*`` and more items over a ``USING`` or ``NATURAL`` join lost every item but the star when the join was
read as ``ON``: ``SELECT *, a.x + 1 AS y FROM a JOIN b USING (k)`` was proved equal to a query of three columns.
Each reading below is run as written and as read on DuckDB and on SQLite over made-up tables (a key with two
matches, keys without a match on either side, a NULL key on both sides); the rows must be the same.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import _using_to_on, prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import TableConstraints

TABLES = {
    "a": ("k INT, x INT", [(1, 10), (2, 20), (None, 30), (4, 40)]),
    "b": ("k INT, z INT", [(1, 100), (1, 101), (3, 300), (None, 400)]),
    "c": ("k INT, w INT", [(1, 7), (3, 8), (4, 9)]),
}
SCHEMA = {"a": ["k", "x"], "b": ["k", "z"], "c": ["k", "w"]}
TYPES = {table: {column: "INT" for column in columns} for table, columns in SCHEMA.items()}

# (as written, the same query with the join spelled out)
READ = [
    ("SELECT *, a.x + 1 AS y FROM a JOIN b USING (k)", "SELECT a.k, a.x, b.z, a.x + 1 AS y FROM a JOIN b ON a.k = b.k"),
    ("SELECT a.x + 1 AS y, * FROM a JOIN b USING (k)", "SELECT a.x + 1 AS y, a.k, a.x, b.z FROM a JOIN b ON a.k = b.k"),
    ("SELECT x - 1 AS before, *, z + k AS after FROM a LEFT JOIN b USING (k)", "SELECT a.x - 1 AS before, a.k, a.x, b.z, b.z + a.k AS after FROM a LEFT JOIN b ON a.k = b.k"),
    ("SELECT *, 1 AS one FROM a NATURAL JOIN b", "SELECT a.k, a.x, b.z, 1 AS one FROM a JOIN b ON a.k = b.k"),
    ("SELECT *, w * 2 AS twice FROM a JOIN b USING (k) JOIN c USING (k)", "SELECT a.k, a.x, b.z, c.w, c.w * 2 AS twice FROM a JOIN b ON a.k = b.k JOIN c ON a.k = c.k"),
    # ``t.*`` is the columns of one source: the merged column is not part of it
    ("SELECT a.*, b.z FROM a LEFT JOIN b USING (k)", "SELECT a.k, a.x, b.z FROM a LEFT JOIN b ON a.k = b.k"),
    ("SELECT b.*, a.x FROM a LEFT JOIN b USING (k)", "SELECT b.k, b.z, a.x FROM a LEFT JOIN b ON a.k = b.k"),
    ("SELECT a.*, b.*, k FROM a LEFT JOIN b USING (k)", "SELECT a.k, a.x, b.k, b.z, a.k FROM a LEFT JOIN b ON a.k = b.k"),
    ("SELECT a.x, k FROM a RIGHT JOIN b USING (k)", "SELECT a.x, b.k FROM a RIGHT JOIN b ON a.k = b.k"),
]
# (as written, a query it must not be proved equal to)
DIFFERENT = [
    ("SELECT *, a.x + 1 AS y FROM a JOIN b USING (k)", "SELECT a.k, a.x, b.z FROM a JOIN b ON a.k = b.k"),  # the item beside the star
    ("SELECT *, a.x + 1 AS y FROM a JOIN b USING (k)", "SELECT a.k, a.x, b.z, a.x + 2 AS y FROM a JOIN b ON a.k = b.k"),
    ("SELECT a.x + 1 AS y, * FROM a JOIN b USING (k)", "SELECT a.k, a.x, b.z, a.x + 1 AS y FROM a JOIN b ON a.k = b.k"),  # the order of the columns
    ("SELECT *, 1 AS one FROM a NATURAL JOIN b", "SELECT a.k, a.x, b.z FROM a JOIN b ON a.k = b.k"),
    ("SELECT b.*, a.x FROM a LEFT JOIN b USING (k)", "SELECT a.k, b.z, a.x FROM a LEFT JOIN b ON a.k = b.k"),  # b.k is NULL on a row without a match
    ("SELECT a.x, k FROM a RIGHT JOIN b USING (k)", "SELECT a.x, a.k FROM a RIGHT JOIN b ON a.k = b.k"),  # the merged column of a RIGHT JOIN is the right one
]
# left as written: two stars, a star with a modifier, ``t.*`` beside a RIGHT or FULL join (see the test below)
LEFT_ALONE = [
    "SELECT a.*, k FROM a RIGHT JOIN b USING (k)",
    "SELECT a.*, b.z FROM a FULL JOIN b USING (k)",
    "SELECT b.*, c.w FROM a JOIN b USING (k) RIGHT JOIN c ON b.k = c.k",
    "SELECT *, * FROM a JOIN b USING (k)",
    "SELECT * EXCLUDE (x), 1 AS one FROM a JOIN b USING (k)",
    "SELECT a.* EXCLUDE (x), z FROM a JOIN b USING (k)",
]


def read(sql: str) -> str:
    return _using_to_on(sqlglot.parse_one(sql, read="duckdb"), SCHEMA).sql(dialect="duckdb")


def _sorted(rows) -> list:
    return sorted((list(row) for row in rows), key=repr)


def _duckdb_rows(sql: str) -> list:
    duckdb = pytest.importorskip("duckdb")
    db = duckdb.connect(":memory:")
    for name, (columns, rows) in TABLES.items():
        db.execute(f"CREATE TABLE {name} ({columns})")
        db.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' for _ in rows[0])})", rows)
    return _sorted(db.execute(sql).fetchall())


def _sqlite_rows(sql: str) -> list:
    db = sqlite3.connect(":memory:")
    for name, (columns, rows) in TABLES.items():
        db.execute(f"CREATE TABLE {name} ({columns})")
        db.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' for _ in rows[0])})", rows)
    return _sorted(db.execute(sqlglot.transpile(sql, read="duckdb", write="sqlite")[0]).fetchall())


def prove(left: str, right: str, **more):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="mysql", **more)


@pytest.mark.parametrize("written, spelled_out", READ)
def test_a_reading_returns_the_rows_of_the_query_as_written(written, spelled_out):
    reading = read(written)
    assert "USING" not in reading and "NATURAL" not in reading
    for rows in (_duckdb_rows, _sqlite_rows):
        assert rows(written) == rows(reading) == rows(spelled_out) and rows(written)


@pytest.mark.parametrize("written, spelled_out", READ)
def test_a_reading_is_proved_equal_to_the_join_spelled_out(written, spelled_out):
    assert prove(written, spelled_out).proven


@pytest.mark.parametrize("written, other", DIFFERENT)
def test_a_query_that_differs_is_not_proved(written, other):
    assert not prove(written, other).proven
    assert _duckdb_rows(written) != _duckdb_rows(other)


@pytest.mark.parametrize("written", LEFT_ALONE)
def test_a_spelling_that_is_not_read_keeps_its_join(written):
    tree = sqlglot.parse_one(written, read="duckdb")
    assert _using_to_on(tree.copy(), SCHEMA) == tree


def test_the_engines_disagree_on_a_table_star_of_the_left_source_beside_a_right_join():
    # SQLite lists the merged column where DuckDB lists the column as stored (NULL on a row without a match)
    written = "SELECT a.*, k FROM a RIGHT JOIN b USING (k)"
    assert [3, None, 3] in _sqlite_rows(written) and [3, None, 3] not in _duckdb_rows(written)
    assert [None, None, 3] in _duckdb_rows(written)


def test_the_items_beside_a_star_are_all_kept():
    reading = sqlglot.parse_one(read("SELECT x - 1 AS before, *, z + k AS after FROM a LEFT JOIN b USING (k)"), read="duckdb")
    assert [item.alias_or_name for item in reading.expressions] == ["before", "k", "x", "z", "after"]


# ``USING`` is read before the pre-passes: a rule that leaves a USING join alone reads the join spelled with ON
KEYED = {"a": TableConstraints(not_null=frozenset({"k", "x"}), keys=(("k",),)), "b": TableConstraints(not_null=frozenset({"k"}), keys=(("k",),))}
ADDED = "SELECT SUM(a.x + IFNULL(b.z, 0)) AS total FROM a LEFT JOIN b ON a.k = b.k"
ADDED_USING = "SELECT SUM(a.x) + COALESCE(SUM(b.z), 0) AS total FROM a LEFT JOIN b USING (k)"


def test_a_using_join_is_read_by_the_rules_that_run_before_the_normal_form():
    assert prove(ADDED, ADDED_USING.replace("USING (k)", "ON a.k = b.k"), constraints=KEYED, exact_arithmetic=True).proven
    assert prove(ADDED, ADDED_USING, constraints=KEYED, exact_arithmetic=True).proven
    with shape_fallback.without("early_using"):
        assert not prove(ADDED, ADDED_USING, constraints=KEYED, exact_arithmetic=True).proven
    # a sum that leaves a row out is still different
    assert not prove(ADDED, ADDED_USING.replace("SUM(a.x) +", "SUM(a.x + 1) +"), constraints=KEYED, exact_arithmetic=True).proven
