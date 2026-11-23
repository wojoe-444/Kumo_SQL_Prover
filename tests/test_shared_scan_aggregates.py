"""``shared_scan_aggregates``: the aggregates a one-row select asks of one table in a scalar subquery, a joined
one-row table or a filtered copy of the table are read in one pass over the table.

Each reading below is run as written and as rewritten on SQLite, and on DuckDB when the spelling is one DuckDB takes
(it refuses a column of a joined one-row table beside an aggregate), over a made-up table in three states: with rows
(a NULL in every column the filters read, repeated values), with rows none of which passes the filters, and empty.
The refused spellings are ones the rewrite would get wrong or does not reason about, each with the reason beside it.
MySQL's reading of a one-row table's column over no rows is the same as SQLite's (NULL); the gate has no MySQL
server, so that reading is replayed on one when the rule changes.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover.shared_scan_aggregates import read_shared_scans

SCHEMA = {"visits": ["visit", "guest", "booked", "came"], "guests": ["guest", "town"]}
COLUMNS = {"visits": "visit INT, guest INT, booked INT, came INT", "guests": "guest INT, town TEXT"}
GUESTS = [(1, "york"), (2, "bath")]
# three states of the table
FILLED = [(1, 1, 3, 3), (2, 1, 4, 6), (3, 2, 5, 5), (4, 2, 7, None), (5, None, 8, 8), (6, 3, None, 2), (7, 3, 9, 9), (8, 3, 9, 9), (9, 2, 6, 4)]
NONE_PASS = [(1, 0, 3, None), (2, 0, None, 6), (3, 0, 5, 4), (4, None, 7, 6)]
STATES = {"filled": FILLED, "none passes": NONE_PASS, "empty": []}

# (as written, as read)
READ = [
    # a scalar subquery over the table the select reads
    (
        "SELECT ROUND(SUM(CASE WHEN booked = came THEN 1 ELSE 0 END) * 100.0 / (SELECT COUNT(*) FROM visits), 2) AS share FROM visits",
        "SELECT ROUND(SUM(CASE WHEN booked = came THEN 1 ELSE 0 END) * 100.0 / (COUNT(*)), 2) AS share FROM visits",
    ),
    (
        "SELECT (SELECT COUNT(visit) FROM visits WHERE booked = came) * 100.0 / COUNT(v.visit) AS share FROM visits AS v",
        "SELECT (COUNT(CASE WHEN v.booked = v.came THEN v.visit END)) * 100.0 / COUNT(v.visit) AS share FROM visits AS v",
    ),
    # the select's own filter goes into its own aggregates, each subquery's filter into that subquery's
    (
        "SELECT COUNT(*) * 100.0 / (SELECT COUNT(*) FROM visits AS w WHERE w.guest > 1) AS share, "
        "MAX(came) - (SELECT MIN(b.booked) + 1 FROM visits AS b) AS span FROM visits WHERE booked = came",
        "SELECT COUNT(CASE WHEN booked = came THEN 1 END) * 100.0 / (COUNT(CASE WHEN visits.guest > 1 THEN 1 END)) AS share, "
        "MAX(CASE WHEN booked = came THEN came END) - (MIN(visits.booked) + 1) AS span FROM visits",
    ),
    (
        "SELECT COUNT(DISTINCT guest) - (SELECT COUNT(DISTINCT guest) FROM visits WHERE came IS NULL) AS n, AVG(came - booked) AS wait "
        "FROM visits WHERE came >= booked",
        "SELECT COUNT(DISTINCT CASE WHEN came >= booked THEN guest END) - (COUNT(DISTINCT CASE WHEN visits.came IS NULL THEN visits.guest END)) AS n, "
        "AVG(CASE WHEN came >= booked THEN came - booked END) AS wait FROM visits",
    ),
    (
        "SELECT (SELECT SUM(came) FROM visits WHERE guest = 1) - (SELECT SUM(DISTINCT came) FROM visits WHERE guest = 2) + COUNT(1) AS n FROM visits",
        "SELECT (SUM(CASE WHEN visits.guest = 1 THEN visits.came END)) - (SUM(DISTINCT CASE WHEN visits.guest = 2 THEN visits.came END)) + COUNT(1) AS n FROM visits",
    ),
    # an aggregate the rule cannot filter stays as it is when the select has no filter
    (
        "SELECT GROUP_CONCAT(came) || (SELECT COUNT(*) FROM visits) AS list FROM visits",
        "SELECT GROUP_CONCAT(came) || (COUNT(*)) AS list FROM visits",
    ),
    # a filtered copy of the table in place of the table
    (
        "SELECT ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM visits), 2) AS share FROM (SELECT * FROM visits WHERE booked = came) AS same",
        "SELECT ROUND(COUNT(CASE WHEN same.booked = same.came THEN 1 END) * 100.0 / (COUNT(*)), 2) AS share FROM visits AS same",
    ),
    (
        "SELECT SUM(same.came) + (SELECT MAX(visit) FROM visits) AS n FROM (SELECT * FROM visits AS w WHERE w.booked = came) AS same WHERE same.guest > 1",
        "SELECT SUM(CASE WHEN (same.booked = same.came) AND (same.guest > 1) THEN same.came END) + (MAX(same.visit)) AS n FROM visits AS same",
    ),
    # a joined one-row table: its column is NULL when the select has no row to read it from
    (
        "SELECT ROUND(x.n * 100.0 / COUNT(v.visit), 2) AS share FROM visits AS v, (SELECT COUNT(visit) AS n FROM visits WHERE booked = came) AS x",
        "SELECT ROUND(CASE WHEN COUNT(*) > 0 THEN (COUNT(CASE WHEN v.booked = v.came THEN v.visit END)) END * 100.0 / COUNT(v.visit), 2) AS share FROM visits AS v",
    ),
    (
        "WITH same AS (SELECT COUNT(visit) AS n FROM visits WHERE booked = came) SELECT ROUND(same.n * 100.0 / COUNT(visit), 2) AS share FROM visits, same",
        "SELECT ROUND(CASE WHEN COUNT(*) > 0 THEN (COUNT(CASE WHEN visits.booked = visits.came THEN visits.visit END)) END * 100.0 / COUNT(visit), 2) AS share FROM visits",
    ),
    (
        "SELECT n, top, COUNT(*) AS found FROM visits AS v CROSS JOIN (SELECT COUNT(*) AS n, MAX(came) + 1 AS top FROM visits AS w WHERE w.guest > 1) AS x "
        "WHERE v.came IS NOT NULL",
        "SELECT CASE WHEN COUNT(CASE WHEN NOT v.came IS NULL THEN 1 END) > 0 THEN (COUNT(CASE WHEN v.guest > 1 THEN 1 END)) END AS n, "
        "CASE WHEN COUNT(CASE WHEN NOT v.came IS NULL THEN 1 END) > 0 THEN (MAX(CASE WHEN v.guest > 1 THEN v.came END) + 1) END AS top, "
        "COUNT(CASE WHEN NOT v.came IS NULL THEN 1 END) AS found FROM visits AS v",
    ),
    (
        "SELECT x.n + y.n + COUNT(*) AS total FROM visits JOIN (SELECT COUNT(*) AS n FROM visits WHERE guest = 1) AS x "
        "JOIN (SELECT SUM(came) AS n FROM visits WHERE guest = 2) AS y",
        "SELECT CASE WHEN COUNT(*) > 0 THEN (COUNT(CASE WHEN visits.guest = 1 THEN 1 END)) END + "
        "CASE WHEN COUNT(*) > 0 THEN (SUM(CASE WHEN visits.guest = 2 THEN visits.came END)) END + COUNT(*) AS total FROM visits",
    ),
    # a one-row table nothing reads multiplies the rows by one
    ("SELECT COUNT(*) AS n FROM visits, (SELECT COUNT(*) AS n FROM visits WHERE guest = 1) AS x", "SELECT COUNT(*) AS n FROM visits"),
]

REFUSED = [
    ("SELECT COUNT(*) / (SELECT COUNT(*) / 2 FROM visits) FROM visits", "a quotient is not taken out of a subquery"),
    ("SELECT COUNT(*) / (SELECT AVG(came) FROM visits) FROM visits", "nor an average"),
    ("SELECT x.n, COUNT(*) FROM visits, (SELECT COUNT(*) * 1.0 / MAX(came) AS n FROM visits) AS x", "nor a quotient out of a one-row table"),
    ("SELECT visit, (SELECT COUNT(*) FROM visits) FROM visits", "no aggregate of its own: a row per row of the table"),
    ("SELECT (SELECT COUNT(*) FROM visits) AS n FROM visits", "no aggregate of its own, again"),
    ("SELECT guest, COUNT(*) / (SELECT COUNT(*) FROM visits) FROM visits", "a column of the table outside an aggregate"),
    ("SELECT COUNT(*) / (SELECT COUNT(*) FROM guests) FROM visits", "the subquery reads another table"),
    ("SELECT COUNT(*) / (SELECT COUNT(*) FROM visits AS w WHERE w.guest = v.guest) FROM visits AS v", "the subquery reads the row outside it"),
    ("SELECT COUNT(*) / (SELECT COUNT(*) FROM visits AS w WHERE w.guest = town) FROM visits AS v", "a name that is not a column of the table"),
    ("SELECT COUNT(*) / (SELECT COUNT(*) FROM visits GROUP BY guest) FROM visits", "a grouped subquery has a row per group"),
    ("SELECT COUNT(*) / (SELECT COUNT(*) FROM visits) FROM visits GROUP BY guest", "a grouped select"),
    ("SELECT COUNT(*) / (SELECT COUNT(*) FROM visits) FROM visits HAVING COUNT(*) > 1", "HAVING can remove the one row"),
    ("SELECT COUNT(*) / (SELECT COUNT(*) FROM visits) FROM visits LIMIT 1", "LIMIT"),
    ("SELECT DISTINCT COUNT(*) / (SELECT COUNT(*) FROM visits) FROM visits", "DISTINCT"),
    ("SELECT COUNT(*) FROM visits WHERE came > (SELECT MIN(came) FROM visits)", "a subquery in WHERE filters rows"),
    ("SELECT SUM((SELECT COUNT(*) FROM visits)) FROM visits", "a subquery inside an aggregate"),
    ("SELECT COUNT(*) + (SELECT COUNT(*) FROM visits JOIN guests ON 1 = 1) FROM visits", "a subquery over a join"),
    ("SELECT COUNT(*) FROM visits WHERE guest IN (SELECT MAX(guest) FROM visits)", "a membership test"),
    ("SELECT COUNT(*) + EXISTS (SELECT COUNT(*) FROM visits) FROM visits", "EXISTS"),
    ("SELECT COUNT(x.n) FROM visits, (SELECT COUNT(*) AS n FROM visits) AS x", "a one-row table's column inside an aggregate"),
    ("SELECT COUNT(*) FROM visits, (SELECT COUNT(*) AS n FROM visits) AS x WHERE x.n > 2", "a one-row table's column in WHERE"),
    ("SELECT guest, COUNT(*) FROM visits, (SELECT COUNT(*) AS guest FROM visits) AS x", "a bare name two sources have"),
    ("SELECT COUNT(*) FROM visits, (SELECT COUNT(*) AS guest FROM visits) AS x WHERE guest > 1", "a bare name two sources have, in WHERE"),
    ("SELECT x.n, COUNT(*) FROM visits JOIN (SELECT COUNT(*) AS n FROM visits) AS x ON x.n > 2", "a join with a condition"),
    ("SELECT x.n, COUNT(*) FROM visits LEFT JOIN (SELECT COUNT(*) AS n FROM visits) AS x", "an outer join"),
    ("SELECT x.n, COUNT(*) FROM visits, (SELECT COUNT(*) AS n FROM guests) AS x", "a one-row table over another table"),
    ("SELECT x.n, COUNT(*) FROM visits, (SELECT guest, COUNT(*) AS n FROM visits) AS x", "a one-row table with a column of some row"),
    ("SELECT x.n, COUNT(*) FROM visits, (SELECT COUNT(*) AS n FROM visits GROUP BY guest) AS x", "a grouped table has a row per group"),
    ("SELECT x.n, y.n FROM (SELECT COUNT(*) AS n FROM visits) AS x, (SELECT COUNT(*) AS n FROM visits) AS y", "no table is read by the select itself"),
    ("WITH visits AS (SELECT 1 AS n) SELECT n, COUNT(*) FROM visits", "a WITH name hides the table"),
    ("WITH guests AS (SELECT COUNT(*) AS n FROM visits) SELECT guests.n, COUNT(*) FROM visits, guests", "a WITH name hides another declared table"),
    ("WITH x AS (SELECT COUNT(*) AS n FROM visits) SELECT COUNT(*) FROM visits", "a WITH query nothing reads"),
    ("WITH x AS (SELECT COUNT(*) AS n FROM visits) SELECT x.n + y.n, COUNT(*) FROM visits, x, x AS y", "a WITH query read twice"),
    ("SELECT COUNT(*) FILTER (WHERE came > 1) + (SELECT COUNT(*) FROM visits) FROM visits WHERE guest > 1", "an aggregate with FILTER"),
    ("SELECT GROUP_CONCAT(came) || (SELECT COUNT(*) FROM visits) FROM visits WHERE guest > 1", "an aggregate the rule cannot filter, under a filter"),
    ("SELECT COUNT(*) + (SELECT COUNT(*) FROM visits WHERE RANDOM() > 0) FROM visits", "a volatile call"),
    ("SELECT COUNT(*) OVER () + (SELECT COUNT(*) FROM visits) FROM visits", "a window"),
    ("SELECT COUNT(*), * FROM visits, (SELECT COUNT(*) AS n FROM visits) AS x", "a star"),
    ("SELECT COUNT(*) + (SELECT COUNT(*) FROM visits) FROM (SELECT visit, came FROM visits WHERE came > 1) AS v", "a copy that lists its columns"),
    ("SELECT COUNT(*) + (SELECT COUNT(*) FROM visits) FROM (SELECT * FROM visits WHERE came > (SELECT 1)) AS v", "a copy filtered by a subquery"),
    ("SELECT COUNT(*) + (SELECT COUNT(*) FROM visits) FROM (SELECT * FROM visits LIMIT 2) AS v", "a copy with LIMIT"),
    ("SELECT COUNT(*) + (SELECT COUNT(*) FROM visits) FROM visits UNION ALL SELECT 1", "not the whole query"),
    ("SELECT COUNT(*) FROM visits", "nothing to fold"),
]


def rewrite(sql: str, dialect: str = "sqlite") -> str | None:
    tree = sqlglot.parse_one(sql, read="sqlite")
    before = tree.sql(dialect="sqlite")
    new, wrote = read_shared_scans(tree, SCHEMA, dialect)
    assert tree.sql(dialect="sqlite") == before  # the tree given is not changed
    assert wrote == (new is not tree)
    return new.sql(dialect="sqlite") if wrote else None


def normal(sql: str) -> str:
    return sqlglot.parse_one(sql, read="sqlite").sql(dialect="sqlite")


@pytest.mark.parametrize("written, read", READ, ids=[written[:70] for written, _ in READ])
def test_the_reading(written, read):
    assert rewrite(written) == normal(read)


@pytest.mark.parametrize("written, why", REFUSED, ids=[why for _, why in REFUSED])
def test_refused(written, why):
    assert rewrite(written) is None, why


def test_a_rewritten_query_is_left_alone():
    for _, read in READ:
        assert rewrite(read) is None


def reads_a_joined_column(written: str) -> bool:
    return "x.n" in written or "same.n" in written or written.startswith("SELECT n, top")


def test_a_joined_column_is_read_only_where_the_dialect_lets_a_select_read_it():
    joined = [written for written, _ in READ if reads_a_joined_column(written)]
    assert len(joined) == 4
    for written in joined:
        assert rewrite(written, "mysql") == rewrite(written, "sqlite") is not None
        for dialect in ("duckdb", "postgres", "bigquery", ""):
            assert rewrite(written, dialect) is None
    for written, _ in READ:
        if written not in joined:
            assert rewrite(written, "duckdb") == rewrite(written, "sqlite")


def test_no_schema_no_reading():
    tree = sqlglot.parse_one(READ[0][0], read="sqlite")
    assert read_shared_scans(tree, None, "mysql") == (tree, False)
    assert read_shared_scans(tree, {"guests": ["guest"]}, "mysql") == (tree, False)  # the table is not declared


def _rounded(rows) -> list:
    return sorted(([value if value is None or isinstance(value, str) else round(float(value), 6) for value in row] for row in rows), key=repr)


def _sqlite_rows(sql: str, visits: list) -> list:
    db = sqlite3.connect(":memory:")
    for name, rows in (("visits", visits), ("guests", GUESTS)):
        db.execute(f"CREATE TABLE {name} ({COLUMNS[name]})")
        if rows:
            db.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' for _ in rows[0])})", rows)
    return _rounded(db.execute(sql).fetchall())


def _duckdb_rows(sql: str, visits: list) -> list:
    duckdb = pytest.importorskip("duckdb")
    db = duckdb.connect(":memory:")
    for name, rows in (("visits", visits), ("guests", GUESTS)):
        db.execute(f"CREATE TABLE {name} ({COLUMNS[name]})")
        if rows:
            db.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' for _ in rows[0])})", rows)
    return _rounded(db.execute(sqlglot.transpile(sql, read="sqlite", write="duckdb")[0]).fetchall())


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("written, read", READ, ids=[written[:70] for written, _ in READ])
def test_sqlite_returns_the_same_row_for_both_spellings(written, read, state):
    rows = _sqlite_rows(written, STATES[state])
    assert len(rows) == 1
    assert rows == _sqlite_rows(rewrite(written), STATES[state]) == _sqlite_rows(read, STATES[state])


DUCKDB = [(written, read) for written, read in READ if not reads_a_joined_column(written) and "GROUP_CONCAT" not in written]


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("written, read", DUCKDB, ids=[written[:70] for written, _ in DUCKDB])
def test_duckdb_returns_the_same_row_for_both_spellings(written, read, state):
    rows = _duckdb_rows(written, STATES[state])
    assert len(rows) == 1
    assert rows == _duckdb_rows(rewrite(written), STATES[state]) == _duckdb_rows(read, STATES[state])


def test_a_joined_column_is_null_over_no_rows_and_the_table_value_otherwise():
    written = "SELECT x.n AS n, COUNT(*) AS found FROM visits AS v, (SELECT COUNT(*) AS n FROM visits) AS x WHERE v.guest = 3"
    read = rewrite(written)
    assert read == normal(
        "SELECT CASE WHEN COUNT(CASE WHEN v.guest = 3 THEN 1 END) > 0 THEN (COUNT(*)) END AS n, COUNT(CASE WHEN v.guest = 3 THEN 1 END) AS found FROM visits AS v"
    )
    assert _sqlite_rows(written, FILLED) == _sqlite_rows(read, FILLED) == [[9.0, 3.0]]
    assert _sqlite_rows(written, NONE_PASS) == _sqlite_rows(read, NONE_PASS) == [[None, 0.0]]  # four rows in the table, none read
    assert _sqlite_rows(written, []) == _sqlite_rows(read, []) == [[None, 0.0]]
