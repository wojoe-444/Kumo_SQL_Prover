"""``apart_table_names``: a table read under its own name by a select and inside a derived table of it gets a name per read.

Each reading below is run as written and as rewritten on DuckDB and on SQLite over a made-up staff table (a boss who
is nobody's row, a NULL boss, a NULL age); the rows must be the same. The spellings left alone are ones with nothing
to name apart, or ones the rule does not reason about.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover.apart_table_names import name_tables_apart

TABLES = {
    "staff": ("id INT, name TEXT, boss INT, age INT", [(1, "ada", None, 50), (2, "bob", 1, 40), (3, "cy", 1, None), (4, "di", 2, 30), (5, "ed", 9, 20), (6, "bob", 2, 30)]),
    "team": ("id INT, name TEXT", [(1, "red"), (2, "blue"), (7, "grey")]),
}

# (as written, as read)
READ = [
    # a derived table over the table it is joined to
    (
        "SELECT staff.id, staff.name, t.n FROM staff JOIN (SELECT boss, COUNT(*) AS n FROM staff GROUP BY boss) AS t ON t.boss = staff.id",
        "SELECT staff_1.id, staff_1.name, t.n FROM staff AS staff_1 JOIN (SELECT boss, COUNT(*) AS n FROM staff AS staff_2 GROUP BY boss) AS t ON t.boss = staff_1.id",
    ),
    # a column without a table name is left as it is; a qualified one follows its own read
    (
        "SELECT id, name FROM staff JOIN (SELECT boss FROM staff WHERE staff.age > 25 GROUP BY boss) AS b ON b.boss = id WHERE staff.age < 60 ORDER BY staff.id",
        "SELECT id, name FROM staff AS staff_1 JOIN (SELECT boss FROM staff AS staff_2 WHERE staff_2.age > 25 GROUP BY boss) AS b ON b.boss = id "
        "WHERE staff_1.age < 60 ORDER BY staff_1.id",
    ),
    # a read with an alias keeps it; a read in a subquery inside the derived table is one of the reads to tell apart
    (
        "SELECT staff.id, t.n FROM staff, (SELECT s.boss, COUNT(*) AS n FROM staff AS s WHERE EXISTS (SELECT 1 FROM staff WHERE staff.id = s.boss) "
        "GROUP BY s.boss) AS t WHERE t.boss = staff.id",
        "SELECT staff_1.id, t.n FROM staff AS staff_1, (SELECT s.boss, COUNT(*) AS n FROM staff AS s WHERE EXISTS(SELECT 1 FROM staff AS staff_2 "
        "WHERE staff_2.id = s.boss) GROUP BY s.boss) AS t WHERE t.boss = staff_1.id",
    ),
    # the nearest read of the name, going outwards: inside the derived table that is its own read
    (
        "SELECT t.id FROM (SELECT id FROM staff WHERE EXISTS (SELECT 1 FROM team WHERE team.id = staff.id)) AS t JOIN staff ON staff.boss = t.id",
        "SELECT t.id FROM (SELECT id FROM staff AS staff_2 WHERE EXISTS(SELECT 1 FROM team WHERE team.id = staff_2.id)) AS t JOIN staff AS staff_1 ON staff_1.boss = t.id",
    ),
    # a star of the table, GROUP BY and HAVING
    (
        "SELECT staff.* FROM staff JOIN (SELECT staff.boss FROM staff GROUP BY staff.boss HAVING COUNT(staff.id) > 1) AS b ON b.boss = staff.boss",
        "SELECT staff_1.* FROM staff AS staff_1 JOIN (SELECT staff_2.boss FROM staff AS staff_2 GROUP BY staff_2.boss HAVING COUNT(staff_2.id) > 1) AS b "
        "ON b.boss = staff_1.boss",
    ),
    # a derived table of a derived table, and a new name that is already taken is skipped
    (
        "SELECT staff_1.id FROM staff, (SELECT id FROM (SELECT id, boss FROM staff) AS u WHERE boss IS NOT NULL) AS staff_1 WHERE staff.id = staff_1.id",
        "SELECT staff_1.id FROM staff AS staff_2, (SELECT id FROM (SELECT id, boss FROM staff AS staff_3) AS u WHERE NOT boss IS NULL) AS staff_1 WHERE staff_2.id = staff_1.id",
    ),
    # only the table that is read at two levels
    (
        "SELECT team.name, staff.name FROM team JOIN staff ON staff.id = team.id JOIN (SELECT MIN(age) AS low FROM staff) AS m ON staff.age > m.low",
        "SELECT team.name, staff_1.name FROM team JOIN staff AS staff_1 ON staff_1.id = team.id JOIN (SELECT MIN(age) AS low FROM staff AS staff_2) AS m "
        "ON staff_1.age > m.low",
    ),
    # a branch of a UNION inside the derived table
    (
        "SELECT staff.id FROM staff JOIN (SELECT id FROM team UNION ALL SELECT boss FROM staff WHERE boss IS NOT NULL) AS u ON u.id = staff.id",
        "SELECT staff_1.id FROM staff AS staff_1 JOIN (SELECT id FROM team UNION ALL SELECT boss FROM staff AS staff_2 WHERE NOT boss IS NULL) AS u ON u.id = staff_1.id",
    ),
]

LEFT_ALONE = [
    ("SELECT staff.id FROM staff WHERE staff.age > 30", "one read"),
    ("SELECT a.id, t.n FROM staff AS a JOIN (SELECT boss, COUNT(*) AS n FROM staff AS b GROUP BY boss) AS t ON a.id = t.boss", "both reads have aliases"),
    ("SELECT staff.id, t.n FROM staff JOIN (SELECT boss, COUNT(*) AS n FROM staff AS b GROUP BY boss) AS t ON staff.id = t.boss", "the inner read has an alias: the names are apart already"),
    ("SELECT id, name FROM staff WHERE id IN (SELECT boss FROM staff WHERE staff.age > 25)", "a read in an IN subquery: no derived table between the two"),
    ("SELECT staff.id, (SELECT COUNT(*) FROM staff WHERE boss = 1) AS n FROM staff", "a read in a scalar subquery"),
    ("SELECT id FROM staff WHERE EXISTS (SELECT 1 FROM team WHERE team.id = staff.id AND EXISTS (SELECT 1 FROM staff WHERE staff.boss = team.id))", "reads in EXISTS subqueries"),
    ("SELECT staff.id FROM staff UNION SELECT staff.boss FROM staff", "two branches of a UNION: neither is inside the other"),
    ("SELECT x.id FROM (SELECT id FROM staff) AS x JOIN (SELECT boss FROM staff) AS y ON x.id = y.boss", "two derived tables side by side"),
    ("SELECT staff.id FROM staff JOIN (SELECT boss FROM staff WHERE other.id = 1) AS t ON t.boss = staff.id", "a name no select around the column reads"),
    ("SELECT staff.id FROM staff JOIN (SELECT boss FROM staff WHERE main.staff.id = 1) AS t ON t.boss = staff.id", "a column named with a database"),
    ("SELECT staff.id FROM staff, staff, (SELECT boss FROM staff) AS t WHERE staff.id = t.boss", "two sources of one name in a select"),
    ("SELECT staff.id FROM (staff JOIN team ON staff.id = team.id) JOIN (SELECT boss FROM staff) AS t ON t.boss = staff.id", "a parenthesized join"),
    ("SELECT staff.id FROM main.staff JOIN (SELECT boss FROM main.staff) AS t ON t.boss = staff.id", "tables read with a database name"),
]


def rewrite(sql: str) -> str | None:
    tree = sqlglot.parse_one(sql, read="duckdb")
    before = tree.sql(dialect="duckdb")
    new, wrote = name_tables_apart(tree)
    assert tree.sql(dialect="duckdb") == before  # the tree given is not changed
    assert wrote == (new is not tree)
    return new.sql(dialect="duckdb") if wrote else None


def normal(sql: str) -> str:
    return sqlglot.parse_one(sql, read="duckdb").sql(dialect="duckdb")


@pytest.mark.parametrize("written, read", READ, ids=[written[:70] for written, _ in READ])
def test_the_reading(written, read):
    assert rewrite(written) == normal(read)


@pytest.mark.parametrize("written, why", LEFT_ALONE, ids=[why for _, why in LEFT_ALONE])
def test_left_alone(written, why):
    assert rewrite(written) is None, why


def test_a_rewritten_query_is_left_alone():
    for _, read in READ:
        assert rewrite(read) is None


def test_no_new_name_is_a_name_of_the_query():
    for written, _ in READ:
        tree = sqlglot.parse_one(written, read="duckdb")
        before = {node.name.lower() for node in tree.find_all(sqlglot.exp.Identifier)}
        new, _ = name_tables_apart(tree)
        aliases = [table.alias.lower() for table in new.find_all(sqlglot.exp.Table) if table.alias and table.alias.lower() not in before]
        assert aliases and len(set(aliases)) == len(aliases)
        # and no column still asks for the table by a name that two reads had
        assert not [column for column in new.find_all(sqlglot.exp.Column) if column.table.lower() == "staff"]


def _rounded(rows) -> list:
    return sorted(([value if value is None or isinstance(value, str) else round(float(value), 6) for value in row] for row in rows), key=repr)


def _duckdb_rows(sql: str) -> list:
    duckdb = pytest.importorskip("duckdb")
    db = duckdb.connect(":memory:")
    for name, (columns, rows) in TABLES.items():
        db.execute(f"CREATE TABLE {name} ({columns})")
        db.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' for _ in rows[0])})", rows)
    return _rounded(db.execute(sql).fetchall())


def _sqlite_rows(sql: str) -> list:
    db = sqlite3.connect(":memory:")
    for name, (columns, rows) in TABLES.items():
        db.execute(f"CREATE TABLE {name} ({columns})")
        db.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' for _ in rows[0])})", rows)
    return _rounded(db.execute(sqlglot.transpile(sql, read="duckdb", write="sqlite")[0]).fetchall())


@pytest.mark.parametrize("written, read", READ, ids=[written[:70] for written, _ in READ])
def test_the_engines_return_the_same_rows_for_both_spellings(written, read):
    for rows in (_duckdb_rows, _sqlite_rows):
        assert rows(written) == rows(rewrite(written)) == rows(read), rows.__name__


def test_the_rows_are_not_all_the_same():
    # the made-up table tells the reads apart: the managers are not everyone, and the star query returns rows
    assert _duckdb_rows(READ[0][0]) == [[1.0, "ada", 2.0], [2.0, "bob", 2.0]]
    assert len(_duckdb_rows(READ[4][0])) == 4
    assert all(_duckdb_rows(written) for written, _ in READ)
