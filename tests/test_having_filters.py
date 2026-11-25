"""MySQL's HAVING in a select that groups nothing, read as a WHERE (hardest-cases batch 31).

Every spelling in ``READ`` was run on MySQL 8.0 as written and as read, over 61 databases (60 random ones and the
one below): the rows were the same each time. The rows MySQL returned from the query as written on the database
below are recorded here, and the tests run the query as read on SQLite against them, so a change to the reading
that changes rows fails without a MySQL server. ``DECLINED`` holds what is not read, with what MySQL does with it.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover import having_filters
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtStatus, prove_equivalent_smt

SCHEMA = {"t": ["id", "a", "b"], "u": ["id", "c"], "w": ["k", "a"]}
TYPES = {"t": {"id": "INT", "a": "INT", "b": "INT"}, "u": {"id": "INT", "c": "INT"}, "w": {"k": "INT", "a": "INT"}}
TABLES = {'t': [(1, 1, 5), (2, 2, None), (3, 3, 1), (4, 3, 7), (5, None, 2), (6, 0, 0), (7, 4, 3), (8, 5, None)], 'u': [(1, 3), (2, 4), (7, None), (3, 3)], 'w': [(1, 2), (2, 3), (3, None)]}

# (the query, the rows MySQL 8.0 returns from it on TABLES)
READ = [
    ('SELECT a FROM t HAVING a > 1', [(2,), (3,), (3,), (4,), (5,)]),
    ('SELECT a AS x FROM t HAVING x > 1', [(2,), (3,), (3,), (4,), (5,)]),
    ('SELECT id, a AS b FROM t HAVING b > 2', [(3, 3), (4, 3), (7, 4), (8, 5)]),
    ('SELECT id, a AS b, b AS a FROM t HAVING a > 2', [(1, 1, 5), (4, 3, 7), (7, 4, 3)]),
    ('SELECT t.a FROM t HAVING t.a > 1', [(2,), (3,), (3,), (4,), (5,)]),
    ('SELECT t.a AS x FROM t HAVING t.a > 2', [(3,), (3,), (4,), (5,)]),
    ('SELECT t.a AS x FROM t HAVING a > 2', [(3,), (3,), (4,), (5,)]),
    ('SELECT a + 1 AS x, a FROM t HAVING a + 1 > 3', [(4, 3), (4, 3), (5, 4), (6, 5)]),
    ('SELECT a + 1 AS x, a FROM t HAVING x > 3 OR a IS NULL', [(4, 3), (4, 3), (5, 4), (6, 5), (None, None)]),
    ('SELECT DISTINCT a AS x FROM t HAVING x > 1', [(2,), (3,), (4,), (5,)]),
    ('SELECT DISTINCT a, b FROM t HAVING a >= b', [(0, 0), (3, 1), (4, 3)]),
    ('SELECT a FROM t HAVING a NOT IN (SELECT c FROM u WHERE c IS NOT NULL)', [(0,), (1,), (2,), (5,)]),
    ('SELECT a FROM t HAVING a NOT IN (SELECT c FROM u)', []),
    ('SELECT id FROM t HAVING id IN (SELECT id FROM u WHERE c > 2)', [(1,), (2,), (3,)]),
    ('SELECT id FROM t HAVING id IN (SELECT u.id FROM u JOIN w ON u.c = w.a)', [(1,), (3,)]),
    ('SELECT id FROM t HAVING id NOT IN (SELECT x.id FROM (SELECT id, c FROM u WHERE c > 1) AS x)', [(4,), (5,), (6,), (7,), (8,)]),
    ('SELECT id FROM t HAVING EXISTS (SELECT 1 FROM u WHERE c > 3)', [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,)]),
    ('SELECT id, a FROM t HAVING a = (SELECT MAX(c) FROM u)', [(7, 4)]),
    ('SELECT id, a FROM t HAVING a > (SELECT COUNT(*) FROM u)', [(8, 5)]),
    ('SELECT t.a, u.c FROM t LEFT JOIN u ON t.id = u.id HAVING c < 3 OR c IS NULL', [(0, None), (3, None), (4, None), (5, None), (None, None)]),
    ('SELECT t.a, u.c FROM t LEFT JOIN u ON t.id = u.id HAVING u.c < 3 OR u.c IS NULL', [(0, None), (3, None), (4, None), (5, None), (None, None)]),
    ('SELECT a FROM t HAVING a > 1 ORDER BY a DESC LIMIT 2', [(4,), (5,)]),
    ('SELECT * FROM t HAVING b > 2', [(1, 1, 5), (4, 3, 7), (7, 4, 3)]),
    ('SELECT t.* FROM t JOIN u ON t.id = u.id HAVING b > 2', [(1, 1, 5), (7, 4, 3)]),
    ('SELECT * FROM t JOIN u ON t.a = u.c HAVING c > 1 AND b IS NOT NULL', [(3, 3, 1, 1, 3), (3, 3, 1, 3, 3), (4, 3, 7, 1, 3), (4, 3, 7, 3, 3), (7, 4, 3, 2, 4)]),
    ('SELECT *, a + b AS s FROM t HAVING s > 3', [(1, 1, 5, 6), (3, 3, 1, 4), (4, 3, 7, 10), (7, 4, 3, 7)]),
    ('SELECT a FROM t HAVING a', [(1,), (2,), (3,), (3,), (4,), (5,)]),
    ('SELECT a, b FROM t WHERE b > 1 HAVING a > 1', [(3, 7), (4, 3)]),
    ('SELECT a, b FROM t WHERE b > 1 OR a = 1 HAVING a > 1 OR b IS NULL', [(3, 7), (4, 3)]),
    ('SELECT id, (SELECT COUNT(*) FROM u WHERE u.c = p.a) AS n FROM t p HAVING n > 0', [(3, 2), (4, 2), (7, 1)]),
    ('SELECT id, (SELECT COUNT(*) FROM u WHERE u.c = p.a) + (SELECT COUNT(*) FROM w WHERE w.a = p.b) AS n FROM t p HAVING n > 1', [(3, 2), (4, 2), (7, 2)]),
    ('SELECT id, COALESCE((SELECT SUM(CASE WHEN c = id THEN 1 ELSE 0 END) FROM u), 0) AS n FROM t HAVING n > 0', [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]),
    ('SELECT x.a FROM (SELECT a, b FROM t HAVING b IS NULL OR b > 2) x', [(1,), (2,), (3,), (4,), (5,)]),
    ('SELECT id, a + 10 AS a FROM t HAVING a > 12', [(3, 13), (4, 13), (7, 14), (8, 15)]),
    ('SELECT id, CASE WHEN a > b THEN a ELSE b END AS m FROM t HAVING m >= 2 AND id <> 3', [(1, 5), (4, 7), (5, 2), (7, 4)]),
    ('SELECT 1 FROM t HAVING 1 = 1', [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)]),
    ('SELECT 1 AS one FROM t HAVING one = 1', [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)]),
    ('SELECT a FROM t HAVING 1 = 0', []),
    ('SELECT a, a FROM t HAVING a > 1', [(2, 2), (3, 3), (3, 3), (4, 4), (5, 5)]),
    ('SELECT a AS x, a AS y FROM t HAVING t.a > 1', [(2, 2), (3, 3), (3, 3), (4, 4), (5, 5)]),
    ('SELECT id FROM t HAVING id IN (SELECT id FROM u) UNION SELECT id FROM u HAVING id > 2', [(1,), (2,), (3,), (7,)]),
    ('SELECT a FROM t HAVING a > 1 UNION ALL SELECT c FROM u HAVING c IS NULL', [(2,), (3,), (3,), (4,), (5,), (None,)]),
    ('WITH v AS (SELECT id, a FROM t HAVING a IS NOT NULL) SELECT v.id FROM v JOIN u ON v.id = u.id HAVING id > 1', [(2,), (3,), (7,)]),
    ('WITH v AS (SELECT c FROM u WHERE c > 1) SELECT a FROM t HAVING a IN (SELECT c FROM v)', [(3,), (3,), (4,)]),
    ('SELECT id FROM t HAVING id IN (SELECT id FROM u HAVING id > 1)', [(2,), (3,), (7,)]),
    ('SELECT t.id, u.c FROM t JOIN u USING (id) HAVING id > 1 AND c IS NOT NULL', [(2, 4), (3, 3)]),
    ("SELECT UPPER(CONCAT(a, 'x')) AS s, b FROM t HAVING s = '2X' OR b = 1", [('2X', None), ('3X', 1)]),
    ('SELECT a * b AS p, a - b AS d FROM t HAVING p > d', [(12, 1), (21, -4), (3, 2), (5, -4)]),
    ('SELECT `a` AS `X` FROM t HAVING x > 1', [(2,), (3,), (3,), (4,), (5,)]),
    ('SELECT A FROM t HAVING a > 1', [(2,), (3,), (3,), (4,), (5,)]),
    ('SELECT DISTINCT(a) AS x, (b) FROM t HAVING a > 1 AND b > 1', [(3, 7), (4, 3)]),
    ('SELECT (a) FROM t HAVING a > 2', [(3,), (3,), (4,), (5,)]),
]

DECLINED = [
    "SELECT a FROM t HAVING b > 1",  # not selected: an error
    "SELECT id, a AS b, b FROM t HAVING b > 2",  # two items of that name: an error
    "SELECT a + 1 AS x FROM t HAVING a > 1",  # an error
    "SELECT id, a + 10 AS a FROM t HAVING t.a > 2",  # an error
    "SELECT t.id, u.id FROM t JOIN u ON t.id = u.id HAVING id > 1",  # ambiguous: an error
    "SELECT a FROM t HAVING EXISTS (SELECT 1 FROM u WHERE u.c = t.a)",  # a correlated subquery
    "SELECT a FROM t HAVING EXISTS (SELECT 1 FROM u WHERE u.c = a)",
    "SELECT a FROM t HAVING EXISTS (SELECT 1 FROM u WHERE u.c = b)",
    "SELECT a FROM t HAVING a IN (SELECT c FROM u WHERE u.id = t.id)",
    "SELECT a, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM t HAVING a > 1",  # a window
    "SELECT a, SUM(a) OVER () AS s FROM t HAVING a > 1",
    "SELECT a / b AS q FROM t HAVING q > 0.5",  # a division behind a name
    "SELECT a, (SELECT AVG(c) / 2 FROM u) AS h FROM t HAVING a > h",
    "SELECT a, RAND() AS r FROM t HAVING a > 1",  # a volatile call
    "SELECT a, @v := a AS s FROM t HAVING s > 1",  # a user variable
    "SELECT a, MYSTERY(a) AS m FROM t HAVING m > 1",  # may be an aggregate
    "SELECT a AS x, (SELECT x) AS y FROM t HAVING y > 1",  # (SELECT alias)
    "SELECT 1 AS one HAVING one = 1",  # no FROM
    "SELECT *, a + 1 AS b FROM t HAVING b > 2",  # a star column and an item of one name
    "SELECT * FROM t JOIN u USING (id) HAVING id > 1",  # a star over USING
    "SELECT * FROM t JOIN u ON t.id = u.id HAVING id > 1",  # a star with two columns of that name
    "SELECT * FROM t AS t1 JOIN t AS t2 ON t1.id = t2.id HAVING t1.a > 1 AND b > 1",
    "SELECT a FROM t HAVING a > (SELECT MAX(t.b) FROM u)",  # an aggregate of outer columns: the select is a grouped one
    "SELECT nosuch.* FROM t HAVING a > 1",
    "SELECT ANY_VALUE(a) AS v FROM t HAVING v > 1",  # not an aggregate in MySQL, and not read
    "SELECT a, JSON_ARRAYAGG(b) AS j FROM t HAVING a > 1",  # an aggregate the parser does not know
    "SELECT id FROM t WHERE EXISTS (SELECT u.c AS z FROM u HAVING z = t.a)",  # an outer column in the condition
]

UNTOUCHED = [  # grouped selects: read as before
    "SELECT COUNT(*) AS c FROM t HAVING c > 1",
    "SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) > 1",
    "SELECT MAX(a) FROM t HAVING MAX(a) > 1",
    "SELECT a FROM t WHERE a > 1",
    "SELECT 1 FROM t HAVING COUNT(*) > 1",
]


def _database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    for table, columns in SCHEMA.items():
        connection.execute(f"CREATE TABLE {table} ({', '.join(f'{c} INTEGER' for c in columns)})")
        connection.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * len(columns))})", TABLES[table])
    return connection


def read(sql: str, dialect: str = "mysql") -> tuple[str, str | None]:
    return having_filters.read(sql, dialect, SCHEMA)


@pytest.mark.parametrize("sql, mysql_rows", READ)
def test_the_query_as_read_returns_the_rows_mysql_returns_from_the_query_as_written(sql, mysql_rows):
    new, unread = read(sql)
    assert unread is None and new != sql
    tree = sqlglot.parse_one(new, read="mysql")
    assert not any(having_filters.plain(select) for select in tree.find_all(sqlglot.exp.Select))
    rows = _database().execute(sqlglot.transpile(new, read="mysql", write="sqlite")[0]).fetchall()
    assert sorted(rows, key=repr) == sorted(mysql_rows, key=repr)
    assert read(new) == (new, None)  # nothing is left to read


def test_the_recorded_rows_tell_the_readings_apart():
    # the database is not one on which everything returns the same rows: most spellings keep some rows and drop some
    kept = [len(rows) for _, rows in READ]
    assert sum(1 for count in kept if count) >= len(READ) - 2 and len({tuple(rows) for _, rows in READ}) > 30


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT a AS x FROM t HAVING x > 1", "SELECT a AS x FROM t WHERE (a > 1)"),
        # the alias, also when the table has a column of that name
        ("SELECT id, a AS b FROM t HAVING b > 2", "SELECT id, a AS b FROM t WHERE (a > 2)"),
        ("SELECT id, a + 10 AS a FROM t HAVING a > 12", "SELECT id, a + 10 AS a FROM t WHERE ((a + 10) > 12)"),
        # a column selected under another alias, by its own name
        ("SELECT t.a AS x FROM t HAVING a > 2", "SELECT t.a AS x FROM t WHERE (t.a > 2)"),
        ("SELECT a, b FROM t WHERE b > 1 HAVING a > 1", "SELECT a, b FROM t WHERE (b > 1) AND (a > 1)"),
        ("SELECT * FROM t HAVING b > 2", "SELECT * FROM t WHERE (b > 2)"),
        (
            "SELECT t.a, u.c FROM t LEFT JOIN u ON t.id = u.id HAVING c < 3 OR c IS NULL",
            "SELECT t.a, u.c FROM t LEFT JOIN u ON t.id = u.id WHERE (u.c < 3 OR u.c IS NULL)",
        ),
        (
            "SELECT id FROM t HAVING id IN (SELECT id FROM u HAVING id > 1)",
            "SELECT id FROM t WHERE (id IN (SELECT id FROM u WHERE (id > 1)))",
        ),
    ],
)
def test_the_text_of_a_reading(sql, expected):
    assert read(sql) == (expected, None)


@pytest.mark.parametrize("sql", DECLINED)
def test_what_cannot_be_read_is_declined(sql):
    assert read(sql) == (sql, having_filters.REASON)
    for prove in (prove_equivalent_algebraic, prove_equivalent_smt):
        for pair in ((sql, "SELECT a FROM t"), ("SELECT a FROM t", sql), (sql, sql)):
            result = prove(*pair, schema=SCHEMA, types=TYPES, dialect="mysql")
            assert result.status is SmtStatus.NOT_PROVEN and result.reason == having_filters.REASON, (prove.__name__, pair)


@pytest.mark.parametrize("sql", UNTOUCHED)
def test_a_grouped_select_is_not_touched(sql):
    assert read(sql) == (sql, None)


@pytest.mark.parametrize("dialect", ["duckdb", "sqlite", "postgres", "bigquery", "tsql"])
def test_other_dialects_are_not_touched(dialect):
    # there the query is an error, or the one group of the standard
    for sql, _ in READ[:6]:
        assert read(sql, dialect) == (sql, None)
    connection = _database()
    with pytest.raises(sqlite3.OperationalError, match="HAVING clause on a non-aggregate query"):
        connection.execute("SELECT a FROM t HAVING a > 1")


PROVED = [
    ("SELECT a FROM t HAVING a > 1", "SELECT a FROM t WHERE a > 1"),
    ("SELECT a AS x FROM t HAVING x > 1", "SELECT a AS x FROM t WHERE a > 1"),
    ("SELECT id, a AS b FROM t HAVING b > 2", "SELECT id, a AS b FROM t WHERE a > 2"),
    ("SELECT DISTINCT a AS x FROM t HAVING x > 1", "SELECT DISTINCT a AS x FROM t WHERE a > 1"),
    (
        "SELECT t.a, u.c FROM t LEFT JOIN u ON t.id = u.id HAVING c < 3 OR c IS NULL",
        "SELECT t.a, u.c FROM t LEFT JOIN u ON t.id = u.id WHERE u.c < 3 OR u.c IS NULL",
    ),
    ("SELECT a FROM t HAVING a NOT IN (SELECT c FROM u)", "SELECT a FROM t WHERE a NOT IN (SELECT c FROM u)"),
    ("SELECT x.a FROM (SELECT a, b FROM t HAVING b IS NULL OR b > 2) x", "SELECT a FROM t WHERE b IS NULL OR b > 2"),
]


@pytest.mark.parametrize("left, right", PROVED)
def test_the_having_and_the_where_spelling_are_proved_equal_in_mysql(left, right):
    for prove in (prove_equivalent_algebraic, prove_equivalent_smt):
        assert prove(left, right, schema=SCHEMA, types=TYPES, dialect="mysql").proven, prove.__name__
        assert prove(right, left, schema=SCHEMA, types=TYPES, dialect="mysql").proven, prove.__name__


def test_the_wrong_counterexample_is_gone_and_a_true_difference_is_still_found():
    # Before, the select was read as one group of all rows and the pair below was reported as different, with a
    # database on which MySQL returns the same rows from both queries.
    same = prove_equivalent_algebraic("SELECT a FROM t HAVING a > 1", "SELECT a FROM t WHERE a > 1", schema=SCHEMA, types=TYPES, dialect="mysql")
    assert same.status is SmtStatus.PROVEN_EQUIVALENT
    # the alias wins over the column of that name: `b` here is `a`
    for prove in (prove_equivalent_algebraic, prove_equivalent_smt):
        differ = prove("SELECT id, a AS b FROM t HAVING b > 2", "SELECT id, a AS b FROM t WHERE b > 2", schema=SCHEMA, types=TYPES, dialect="mysql")
        assert not differ.proven
        other = prove("SELECT a FROM t HAVING a > 1", "SELECT a FROM t", schema=SCHEMA, types=TYPES, dialect="mysql")
        assert not other.proven
    connection = _database()
    assert sorted(connection.execute("SELECT id, a AS b FROM t WHERE b > 2").fetchall()) != sorted(dict(READ)["SELECT id, a AS b FROM t HAVING b > 2"])


def test_in_another_dialect_the_pair_is_never_proved_through_this_reading():
    # DuckDB, PostgreSQL and BigQuery reject the query; whatever the prover says there, it is not a proof built on
    # MySQL's reading
    for dialect in ("duckdb", "postgres", "bigquery"):
        result = prove_equivalent_algebraic("SELECT id, a AS b FROM t HAVING b > 2", "SELECT id, a AS b FROM t WHERE a > 2", schema=SCHEMA, types=TYPES, dialect=dialect)
        assert not result.proven, dialect
