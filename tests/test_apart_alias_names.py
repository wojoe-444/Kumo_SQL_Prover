"""``apart_alias_names``: a source name used by a select and again inside the derived table it starts from gets a name at each place.

Each reading below is run as written and as rewritten on DuckDB and on SQLite over made-up tables (a boss who is
nobody's row, a NULL boss, a NULL age); the rows must be the same. The spellings left alone are ones with nothing
to name apart, or ones the rule does not reason about.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover.apart_alias_names import name_aliases_apart

TABLES = {
    "staff": ("id INT, name TEXT, boss INT, age INT", [(1, "ada", None, 50), (2, "bob", 1, 40), (3, "cy", 1, None), (4, "di", 2, 30), (5, "ed", 9, 20), (6, "bob", 2, 30)]),
    "team": ("id INT, name TEXT", [(1, "red"), (2, "blue"), (7, "grey")]),
}

# (as written, as read)
READ = [
    # the same short name for a lookup inside a derived table and for the one joined to it
    (
        "SELECT c.id, c.boss_name, v.name FROM (SELECT e.id, v.name AS boss_name FROM staff AS e LEFT JOIN staff AS v ON e.boss = v.id) AS c LEFT JOIN team AS v ON c.id = v.id",
        "SELECT c.id, c.boss_name, v_1.name FROM (SELECT e.id, v_2.name AS boss_name FROM staff AS e LEFT JOIN staff AS v_2 ON e.boss = v_2.id) AS c "
        "LEFT JOIN team AS v_1 ON c.id = v_1.id",
    ),
    # an alias that is the own name of a table read inside the derived table
    (
        "SELECT staff.name, t.n FROM (SELECT boss, COUNT(*) AS n FROM staff GROUP BY boss) AS t JOIN team AS staff ON t.boss = staff.id",
        "SELECT staff_1.name, t.n FROM (SELECT boss, COUNT(*) AS n FROM staff AS staff_2 GROUP BY boss) AS t JOIN team AS staff_1 ON t.boss = staff_1.id",
    ),
    # a derived table called like the source inside it
    (
        "SELECT t.id FROM (SELECT t.id, t.age FROM staff AS t WHERE t.age > 25) AS t WHERE t.age < 45",
        "SELECT t_1.id FROM (SELECT t_2.id, t_2.age FROM staff AS t_2 WHERE t_2.age > 25) AS t_1 WHERE t_1.age < 45",
    ),
    # a star, GROUP BY, HAVING and ORDER BY follow their own source
    (
        "SELECT v.*, c.n FROM (SELECT v.boss, COUNT(v.id) AS n FROM staff AS v GROUP BY v.boss HAVING COUNT(v.id) > 0) AS c JOIN team AS v ON c.boss = v.id ORDER BY v.id",
        "SELECT v_1.*, c.n FROM (SELECT v_2.boss, COUNT(v_2.id) AS n FROM staff AS v_2 GROUP BY v_2.boss HAVING COUNT(v_2.id) > 0) AS c JOIN team AS v_1 "
        "ON c.boss = v_1.id ORDER BY v_1.id",
    ),
    # the derived table a derived table starts from
    (
        "SELECT a.id, b.low FROM (SELECT x.boss, x.low FROM (SELECT a.boss, MIN(a.age) AS low FROM staff AS a GROUP BY a.boss) AS x WHERE x.low > 20) AS b JOIN staff AS a ON b.boss = a.id",
        "SELECT a_1.id, b.low FROM (SELECT x.boss, x.low FROM (SELECT a_2.boss, MIN(a_2.age) AS low FROM staff AS a_2 GROUP BY a_2.boss) AS x "
        "WHERE x.low > 20) AS b JOIN staff AS a_1 ON b.boss = a_1.id",
    ),
    # a column without a source name is left as it is; a qualified one follows its own source
    (
        "SELECT id, name FROM (SELECT boss FROM staff AS s WHERE s.age > 25 GROUP BY boss) AS b JOIN staff AS s ON b.boss = id WHERE s.age < 60",
        "SELECT id, name FROM (SELECT boss FROM staff AS s_2 WHERE s_2.age > 25 GROUP BY boss) AS b JOIN staff AS s_1 ON b.boss = id WHERE s_1.age < 60",
    ),
    # a new name that is already taken is skipped
    (
        "SELECT s_1.id FROM (SELECT s.id FROM team AS s) AS s_1, staff AS s WHERE s.id = s_1.id",
        "SELECT s_1.id FROM (SELECT s_3.id FROM team AS s_3) AS s_1, staff AS s_2 WHERE s_2.id = s_1.id",
    ),
    # a subquery of the outer select that uses the name too keeps it: its columns are its own
    (
        "SELECT v.id FROM (SELECT v.boss FROM staff AS v) AS t JOIN team AS v ON t.boss = v.id WHERE v.id IN (SELECT v.id FROM staff AS v WHERE v.age > 25)",
        "SELECT v_1.id FROM (SELECT v_2.boss FROM staff AS v_2) AS t JOIN team AS v_1 ON t.boss = v_1.id WHERE v_1.id IN (SELECT v.id FROM staff AS v WHERE v.age > 25)",
    ),
]

# read the same way, on DuckDB only (SQLite has no column list on an alias)
COLUMN_LIST = (
    "SELECT t.a FROM (SELECT t.id, t.age FROM staff AS t) AS t (a, b) WHERE t.b > 25",
    "SELECT t_1.a FROM (SELECT t_2.id, t_2.age FROM staff AS t_2) AS t_1(a, b) WHERE t_1.b > 25",
)

INSIDE = "(SELECT v.boss FROM staff AS v) AS t"

LEFT_ALONE = [
    ("SELECT s.id FROM staff AS s WHERE s.age > 30", "one source"),
    ("SELECT a.id, t.n FROM (SELECT b.boss, COUNT(*) AS n FROM staff AS b GROUP BY b.boss) AS t JOIN staff AS a ON a.id = t.boss", "the names are apart already"),
    ("SELECT v.id, c.n FROM team AS v JOIN (SELECT v.boss, COUNT(v.id) AS n FROM staff AS v GROUP BY v.boss) AS c ON c.boss = v.id", "a derived table that is joined in"),
    ("SELECT s.id, u.id FROM (SELECT s.id FROM team AS s UNION ALL SELECT s.boss FROM staff AS s) AS u JOIN staff AS s ON u.id = s.id", "the branches of a UNION inside the derived table"),
    (
        "SELECT s.id, t.n FROM (SELECT b.boss, COUNT(*) AS n FROM staff AS b WHERE EXISTS (SELECT 1 FROM team AS s WHERE s.id = b.boss) GROUP BY b.boss) AS t, staff AS s WHERE t.boss = s.id",
        "the name inside a subquery of the derived table",
    ),
    ("SELECT s.id, s.name FROM staff AS s WHERE s.id IN (SELECT s.boss FROM staff AS s WHERE s.age > 25)", "the same alias in an IN subquery: no derived table between the two"),
    ("SELECT s.id, (SELECT COUNT(*) FROM staff AS s WHERE s.boss = 1) AS n FROM staff AS s", "the same alias in a scalar subquery"),
    ("SELECT s.id FROM staff AS s UNION SELECT s.boss FROM staff AS s", "two branches of a UNION: neither is inside the other"),
    ("SELECT x.id FROM (SELECT s.id FROM staff AS s) AS x JOIN (SELECT s.boss FROM staff AS s) AS y ON x.id = y.boss", "two derived tables side by side"),
    (f"SELECT v.id FROM {INSIDE.replace(') AS t', ' WHERE other.id = 1) AS t')} JOIN team AS v ON t.boss = v.id", "a name no select around the column reads"),
    (f"SELECT v.id FROM {INSIDE.replace(') AS t', ' WHERE main.v.id = 1) AS t')} JOIN team AS v ON t.boss = v.id", "a column named with a database"),
    (f"SELECT v.id FROM {INSIDE}, team AS v, staff AS v WHERE v.id = t.boss", "two sources of one name in a select"),
    (f"SELECT v.id FROM {INSIDE} JOIN (team AS v JOIN staff AS w ON v.id = w.id) ON t.boss = v.id", "a parenthesized join"),
    (f"SELECT v FROM {INSIDE} JOIN team AS v ON t.boss = v.id", "a bare name that may be the source's whole row"),
    (f"SELECT COUNT(v) FROM {INSIDE} JOIN team AS v ON t.boss = v.id", "a bare name inside a call"),
]


def rewrite(sql: str) -> str | None:
    tree = sqlglot.parse_one(sql, read="duckdb")
    before = tree.sql(dialect="duckdb")
    new, wrote = name_aliases_apart(tree)
    assert tree.sql(dialect="duckdb") == before  # the tree given is not changed
    assert wrote == (new is not tree)
    return new.sql(dialect="duckdb") if wrote else None


def normal(sql: str) -> str:
    return sqlglot.parse_one(sql, read="duckdb").sql(dialect="duckdb")


@pytest.mark.parametrize("written, read", [*READ, COLUMN_LIST], ids=[written[:70] for written, _ in [*READ, COLUMN_LIST]])
def test_the_reading(written, read):
    assert rewrite(written) == normal(read)


@pytest.mark.parametrize("written, why", LEFT_ALONE, ids=[why for _, why in LEFT_ALONE])
def test_left_alone(written, why):
    assert rewrite(written) is None, why


def test_a_rewritten_query_is_left_alone():
    for _, read in [*READ, COLUMN_LIST]:
        assert rewrite(read) is None


def test_no_new_name_is_a_name_of_the_query():
    for written, _ in [*READ, COLUMN_LIST]:
        tree = sqlglot.parse_one(written, read="duckdb")
        before = {node.name.lower() for node in tree.find_all(sqlglot.exp.Identifier)}
        new, _ = name_aliases_apart(tree)
        sources = [*new.find_all(sqlglot.exp.Table), *new.find_all(sqlglot.exp.Subquery)]
        aliases = [source.alias.lower() for source in sources if source.alias and source.alias.lower() not in before]
        assert len(aliases) >= 2 and len(set(aliases)) == len(aliases)
        # and every qualified column asks for a source by a name exactly one source of the query has
        names = [source.alias_or_name.lower() for source in sources if isinstance(source.parent, (sqlglot.exp.From, sqlglot.exp.Join))]
        for column in new.find_all(sqlglot.exp.Column):
            assert not column.table or names.count(column.table.lower()) == 1, column.sql()


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


def test_duckdb_returns_the_same_rows_with_a_column_list():
    written, read = COLUMN_LIST
    assert _duckdb_rows(written) == _duckdb_rows(rewrite(written)) == _duckdb_rows(read) == [[1.0], [2.0], [4.0], [6.0]]


def test_the_rows_are_not_all_the_same():
    # the made-up tables tell the sources apart: a lookup of the boss differs from a lookup of the team
    assert _duckdb_rows(READ[0][0]) == [[1.0, None, "red"], [2.0, "ada", "blue"], [3.0, "ada", None], [4.0, "bob", None], [5.0, None, None], [6.0, "bob", None]]
    assert all(_duckdb_rows(written) for written, _ in READ)
