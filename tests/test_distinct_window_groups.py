"""``distinct_window_groups``: a ``DISTINCT`` over the aggregates of a row's whole partition is the grouped aggregate.

Each reading below is run as written and as rewritten on DuckDB and on SQLite over made-up tables that hold a NULL
partition value, a NULL argument, repeated rows and a partition of one row; the rows must be the same. The refused
spellings are ones the rewrite would get wrong, each with the reason beside it.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover.distinct_window_groups import read_distinct_window_groups

TABLES = {
    "projects": ("team INT, person INT", [(1, 1), (1, 2), (1, 3), (2, 1), (2, 4), (3, 5), (None, 2), (None, 6), (4, 7), (4, 7)]),
    "staff": ("person INT, years INT, town TEXT", [(1, 3, "york"), (2, 5, "york"), (3, None, "bath"), (4, 3, None), (5, 10, "bath"), (6, 1, None), (7, 2, "york")]),
}
JOINED = "FROM projects AS p JOIN staff AS e ON p.person = e.person"

# (as written, as read)
READ = [
    # the direct form
    (
        f"SELECT DISTINCT p.team, ROUND(AVG(e.years) OVER (PARTITION BY p.team), 2) AS mean {JOINED}",
        f"SELECT p.team, ROUND(AVG(e.years), 2) AS mean {JOINED} GROUP BY p.team",
    ),
    (
        f"SELECT DISTINCT team, COUNT(*) OVER (PARTITION BY p.team) AS n, 'all' AS label {JOINED} WHERE e.years > 1 ORDER BY team",
        f"SELECT team, COUNT(*) AS n, 'all' AS label {JOINED} WHERE e.years > 1 GROUP BY p.team ORDER BY team",
    ),
    (
        f"SELECT DISTINCT e.town, p.team, MAX(e.years) OVER (PARTITION BY p.team, e.town) - MIN(e.years) OVER (PARTITION BY e.town, p.team) AS spread {JOINED}",
        f"SELECT e.town, p.team, MAX(e.years) - MIN(e.years) AS spread {JOINED} GROUP BY p.team, e.town",
    ),
    # a partition expression that is not shown: the DISTINCT stays, two teams with one mean are one row
    (
        f"SELECT DISTINCT SUM(e.years) OVER (PARTITION BY p.team) AS total {JOINED}",
        f"SELECT DISTINCT SUM(e.years) AS total {JOINED} GROUP BY p.team",
    ),
    (
        f"SELECT DISTINCT e.town, COUNT(e.years) OVER (PARTITION BY p.team, e.town) AS n {JOINED}",
        f"SELECT DISTINCT e.town, COUNT(e.years) AS n {JOINED} GROUP BY p.team, e.town",
    ),
    # an expression as the partition, shown as it is
    (
        "SELECT DISTINCT years % 2 AS odd, COUNT(*) OVER (PARTITION BY years % 2) AS n FROM staff",
        "SELECT years % 2 AS odd, COUNT(*) AS n FROM staff GROUP BY years % 2",
    ),
    # one level up: a derived table and a WITH query read once
    (
        "SELECT DISTINCT town FROM (SELECT town, COUNT(*) OVER (PARTITION BY town) AS n FROM staff) AS t WHERE n > 1",
        "SELECT DISTINCT town FROM (SELECT town, COUNT(*) AS n FROM staff GROUP BY town) AS t WHERE n > 1",
    ),
    (
        f"WITH means AS (SELECT p.team, AVG(e.years) OVER (PARTITION BY p.team) AS mean {JOINED}) SELECT DISTINCT team, ROUND(mean, 1) AS mean FROM means ORDER BY team",
        f"WITH means AS (SELECT p.team, AVG(e.years) AS mean {JOINED} GROUP BY p.team) SELECT DISTINCT team, ROUND(mean, 1) AS mean FROM means ORDER BY team",
    ),
    (
        "SELECT DISTINCT * FROM (SELECT town, MAX(years) OVER (PARTITION BY town) AS top, 1 AS one FROM staff ORDER BY town) AS t WHERE top + one < 9",
        "SELECT DISTINCT * FROM (SELECT town, MAX(years) AS top, 1 AS one FROM staff GROUP BY town) AS t WHERE top + one < 9",
    ),
    # columns of the derived table nobody reads are left out: a star, and outputs that differ inside a partition
    (
        "SELECT DISTINCT town FROM (SELECT *, COUNT(years) OVER (PARTITION BY town) AS n FROM staff) AS t WHERE n >= 2",
        "SELECT DISTINCT town FROM (SELECT town, COUNT(years) AS n FROM staff GROUP BY town) AS t WHERE n >= 2",
    ),
    (
        "SELECT DISTINCT town AS place FROM (SELECT person, years, town, MAX(years) OVER (PARTITION BY town) AS top FROM staff) AS t WHERE top > 3 ORDER BY place",
        "SELECT DISTINCT town AS place FROM (SELECT town, MAX(years) AS top FROM staff GROUP BY town) AS t WHERE top > 3 ORDER BY place",
    ),
    # a test on the row's number in its partition: some row of n passes it exactly when the count does
    (
        "WITH numbered AS (SELECT town, ROW_NUMBER() OVER (PARTITION BY town) AS rn FROM staff) SELECT DISTINCT town FROM numbered WHERE rn > 1",
        "WITH numbered AS (SELECT town, COUNT(*) AS rn FROM staff GROUP BY town) SELECT DISTINCT town FROM numbered WHERE rn > 1",
    ),
    (
        "SELECT DISTINCT town FROM (SELECT town, ROW_NUMBER() OVER (PARTITION BY town ORDER BY years DESC) AS rn FROM staff) AS t WHERE t.rn = 2",
        "SELECT DISTINCT town FROM (SELECT town, COUNT(*) AS rn FROM staff GROUP BY town) AS t WHERE t.rn >= 2",
    ),
    (
        "SELECT DISTINCT town FROM (SELECT town, ROW_NUMBER() OVER (PARTITION BY town ORDER BY person) AS rn, SUM(years) OVER (PARTITION BY town) AS total FROM staff) AS t WHERE 2 <= rn OR total > 9",
        "SELECT DISTINCT town FROM (SELECT town, COUNT(*) AS rn, SUM(years) AS total FROM staff GROUP BY town) AS t WHERE rn >= 2 OR total > 9",
    ),
    (
        "SELECT DISTINCT town FROM (SELECT town, ROW_NUMBER() OVER (PARTITION BY town ORDER BY person) AS rn FROM staff) AS t WHERE (rn > 0) AND town IS NOT NULL",
        None,  # the reader tests the partition column
    ),
    (
        "SELECT DISTINCT t.n FROM (SELECT person, town, COUNT(*) OVER (PARTITION BY town) AS n FROM staff WHERE years > 2) AS t",
        "SELECT DISTINCT t.n FROM (SELECT COUNT(*) AS n FROM staff WHERE years > 2 GROUP BY town) AS t",  # the columns nobody reads are left out
    ),
]

# (as written, why the rewrite would be wrong)
REFUSED = [
    (f"SELECT p.team, AVG(e.years) OVER (PARTITION BY p.team) AS mean {JOINED}", "no DISTINCT: one row per input row"),
    (f"SELECT DISTINCT p.team, e.person, AVG(e.years) OVER (PARTITION BY p.team) AS mean {JOINED}", "person differs inside a partition"),
    (f"SELECT DISTINCT p.team, e.years - AVG(e.years) OVER (PARTITION BY p.team) AS above {JOINED}", "a row's own value beside the window"),
    ("SELECT DISTINCT UPPER(town), COUNT(*) OVER (PARTITION BY town) FROM staff", "two spellings a collation calls equal would show two rows"),
    ("SELECT DISTINCT town, SUM(years) OVER (PARTITION BY town ORDER BY person) FROM staff", "a running sum"),
    ("SELECT DISTINCT town, SUM(years) OVER (PARTITION BY town ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM staff", "a frame"),
    ("SELECT DISTINCT town, SUM(years) OVER () FROM staff", "no partition: an empty table gives no row, a grouped total one"),
    ("SELECT DISTINCT town, SUM(years) OVER (PARTITION BY 1) FROM staff", "GROUP BY 1 would name the first output"),
    ("SELECT DISTINCT town, SUM(years) OVER (PARTITION BY town), COUNT(*) OVER (PARTITION BY person) FROM staff", "two different partitions"),
    ("SELECT DISTINCT town, ROW_NUMBER() OVER (PARTITION BY town) FROM staff", "not an aggregate of the partition"),
    ("SELECT DISTINCT town, FIRST_VALUE(years) OVER (PARTITION BY town) FROM staff", "an arbitrary row's value"),
    ("SELECT DISTINCT town, COUNT(DISTINCT years) OVER (PARTITION BY town) FROM staff", "a DISTINCT argument is not read"),
    ("SELECT DISTINCT town, SUM(years) FILTER (WHERE years > 1) OVER (PARTITION BY town) FROM staff", "a FILTER is not read"),
    ("SELECT DISTINCT town, SUM(years) OVER w FROM staff WINDOW w AS (PARTITION BY town)", "a named window is not read"),
    ("SELECT DISTINCT town, SUM(years) OVER (PARTITION BY town) + RAND() FROM staff", "a different value for every row"),
    ("SELECT DISTINCT town, SUM(years) OVER (PARTITION BY town) FROM staff LIMIT 2", "the limit counts rows before the DISTINCT is the same, but which rows is not"),
    ("SELECT DISTINCT town, SUM(years) OVER (PARTITION BY town) FROM staff GROUP BY town, years", "windows over groups"),
    ("SELECT DISTINCT town, SUM(years) OVER (PARTITION BY town), (SELECT MAX(team) FROM projects) FROM staff", "a subquery in the outputs is not read"),
    ("SELECT DISTINCT *, SUM(years) OVER (PARTITION BY town) FROM staff", "a star shows every column"),
    ("SELECT DISTINCT years AS town, SUM(years) OVER (PARTITION BY town) FROM staff", "the output alias redefines the partition's name"),
    ("SELECT DISTINCT ON (town) town, SUM(years) OVER (PARTITION BY town) FROM staff", "DISTINCT ON keeps a row per town"),
    ("SELECT DISTINCT town, SUM(years) OVER (PARTITION BY town) AS total FROM staff ORDER BY person", "ordered by a column that is not shown"),
    (f"SELECT DISTINCT person, COUNT(*) OVER (PARTITION BY p.person) AS n FROM projects AS p JOIN staff AS e USING (person)", "USING merges two columns into the bare name"),
    # one level up
    ("SELECT town FROM (SELECT town, COUNT(*) OVER (PARTITION BY town) AS n FROM staff) AS t WHERE n > 1", "the reader has no DISTINCT"),
    ("SELECT DISTINCT town FROM (SELECT town, COUNT(*) OVER (PARTITION BY town) AS n FROM staff) AS t WHERE UPPER(town) = 'YORK'", "the reader tests the partition column"),
    ("SELECT DISTINCT LENGTH(town) FROM (SELECT town, COUNT(*) OVER (PARTITION BY town) AS n FROM staff) AS t", "the reader computes with the partition column"),
    ("SELECT DISTINCT town, COUNT(*) FROM (SELECT town, COUNT(*) OVER (PARTITION BY town) AS n FROM staff) AS t GROUP BY town", "the reader counts the copies"),
    ("SELECT DISTINCT t.town FROM (SELECT town, COUNT(*) OVER (PARTITION BY town) AS n FROM staff) AS t JOIN projects AS p ON p.team = t.n", "a join is not read"),
    ("SELECT DISTINCT town FROM (SELECT town, COUNT(*) OVER (PARTITION BY town) AS n FROM staff LIMIT 3) AS t", "a limit inside"),
    ("SELECT DISTINCT a FROM (SELECT town, COUNT(*) OVER (PARTITION BY town) AS n FROM staff) AS t (a, b)", "a column list renames the outputs"),
    (
        "WITH c AS (SELECT town, COUNT(*) OVER (PARTITION BY town) AS n FROM staff) SELECT DISTINCT town FROM c WHERE n > (SELECT COUNT(*) FROM c) - 5",
        "the WITH query is read twice",
    ),
    ("WITH c (a, n) AS (SELECT town, COUNT(*) OVER (PARTITION BY town) FROM staff) SELECT DISTINCT a FROM c", "a column list on the WITH query"),
    ("SELECT DISTINCT town FROM (SELECT town, ROW_NUMBER() OVER (PARTITION BY town ORDER BY person) AS rn FROM staff) AS t WHERE NOT rn > 1", "under NOT: that no row passes is not what a count says"),
    ("SELECT DISTINCT town FROM (SELECT town, ROW_NUMBER() OVER (PARTITION BY town ORDER BY person) AS rn FROM staff) AS t WHERE rn > 1 AND rn < 3", "two tests on a row are not two tests on the count"),
    ("SELECT DISTINCT town FROM (SELECT town, ROW_NUMBER() OVER (PARTITION BY town ORDER BY person) AS rn FROM staff) AS t WHERE rn < 3", "a test from above is not read"),
    ("SELECT DISTINCT town FROM (SELECT town, ROW_NUMBER() OVER (PARTITION BY town ORDER BY person) AS rn FROM staff) AS t WHERE rn = 0", "no row has the number 0"),
    ("SELECT DISTINCT town FROM (SELECT town, ROW_NUMBER() OVER (PARTITION BY town ORDER BY person) AS rn FROM staff) AS t WHERE rn + 1 > 2", "the number inside a sum is not read"),
    ("SELECT DISTINCT town FROM (SELECT town, ROW_NUMBER() OVER (PARTITION BY town ORDER BY person) AS rn FROM staff) AS t WHERE CASE WHEN rn > 1 THEN FALSE ELSE TRUE END", "inside a CASE"),
    ("SELECT DISTINCT town, rn FROM (SELECT town, ROW_NUMBER() OVER (PARTITION BY town ORDER BY person) AS rn FROM staff) AS t WHERE rn > 1", "the number is shown"),
    ("SELECT DISTINCT town FROM (SELECT town, ROW_NUMBER() OVER (PARTITION BY town ORDER BY person) AS rn FROM staff) AS t WHERE rn > 1 ORDER BY rn", "ordered by the number"),
    ("SELECT DISTINCT town FROM (SELECT town, RANK() OVER (PARTITION BY town ORDER BY years) AS rn FROM staff) AS t WHERE rn > 1", "a rank has ties and gaps"),
    ("SELECT DISTINCT town FROM (SELECT town, ROW_NUMBER() OVER (ORDER BY years) AS rn FROM staff) AS t WHERE rn > 1", "numbered over the whole table"),
    ("SELECT DISTINCT town FROM (SELECT town, ROW_NUMBER() OVER (PARTITION BY town) AS rn, COUNT(*) OVER (PARTITION BY person) AS n FROM staff) AS t WHERE rn > 1 OR n > 1", "two different partitions are read"),
    ("SELECT DISTINCT town, person FROM (SELECT *, COUNT(*) OVER (PARTITION BY town) AS n FROM staff) AS t WHERE n > 1", "person differs inside a partition"),
    ("SELECT DISTINCT town FROM (SELECT *, COUNT(*) OVER (PARTITION BY town) AS n FROM staff) AS t WHERE n > 1 OR years > 3", "a row's own value in the test"),
    ("SELECT DISTINCT * FROM (SELECT *, COUNT(*) OVER (PARTITION BY town) AS n FROM staff) AS t", "every column of every row is shown"),
    ("SELECT DISTINCT town FROM (SELECT *, COUNT(*) OVER (PARTITION BY town) AS n FROM staff) AS t WHERE n > 1 AND town <> 'bath'", "the reader tests the partition column"),
    ("SELECT DISTINCT person FROM (SELECT *, COUNT(*) OVER (PARTITION BY e.person) AS n FROM staff AS e JOIN projects AS p ON p.team = e.years) AS t WHERE n > 1", "the star of a join can show a name twice"),
    ("SELECT DISTINCT town FROM (SELECT DISTINCT ON (town) town, COUNT(*) OVER (PARTITION BY town) AS n FROM staff) AS t WHERE n > 1", "DISTINCT ON inside is not read"),
]


def rewrite(sql: str, dialect: str = "duckdb") -> str | None:
    tree = sqlglot.parse_one(sql, read=dialect)
    before = tree.sql(dialect=dialect)
    new, wrote = read_distinct_window_groups(tree)
    assert tree.sql(dialect=dialect) == before  # the tree given is not changed
    assert wrote == (new is not tree)
    return new.sql(dialect=dialect) if wrote else None


def normal(sql: str) -> str:
    return sqlglot.parse_one(sql, read="duckdb").sql(dialect="duckdb")


@pytest.mark.parametrize("written, read", READ, ids=[written[:70] for written, _ in READ])
def test_the_reading(written, read):
    assert rewrite(written) == (None if read is None else normal(read))


@pytest.mark.parametrize("written, why", REFUSED, ids=[why for _, why in REFUSED])
def test_refused(written, why):
    assert rewrite(written) is None, why


def test_a_rewritten_query_is_left_alone():
    for written, read in READ:
        if read is not None:
            assert rewrite(read) is None


def test_several_selects_of_one_statement_are_read():
    union = (
        "SELECT DISTINCT town, COUNT(*) OVER (PARTITION BY town) AS n FROM staff WHERE years > 2 "
        "UNION ALL SELECT DISTINCT town, COUNT(*) OVER (PARTITION BY town) AS n FROM staff WHERE years <= 2"
    )
    assert rewrite(union) == normal(
        "SELECT town, COUNT(*) AS n FROM staff WHERE years > 2 GROUP BY town UNION ALL SELECT town, COUNT(*) AS n FROM staff WHERE years <= 2 GROUP BY town"
    )


def test_a_statement_without_a_window_or_without_distinct_is_returned_as_it_is():
    for sql in ("SELECT DISTINCT town FROM staff", "SELECT town, COUNT(*) OVER (PARTITION BY town) FROM staff"):
        tree = sqlglot.parse_one(sql, read="duckdb")
        assert read_distinct_window_groups(tree) == (tree, False)


def _rounded(rows) -> list:
    return sorted(([value if value is None or isinstance(value, str) else round(float(value), 6) for value in row] for row in rows), key=repr)


def _duckdb_rows(sql: str) -> list:
    duckdb = pytest.importorskip("duckdb")
    db = duckdb.connect(":memory:")
    for name, (columns, rows) in TABLES.items():
        db.execute(f"CREATE TABLE {name} ({columns})")
        if rows:
            db.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' for _ in rows[0])})", rows)
    return _rounded(db.execute(sql).fetchall())


def _sqlite_rows(sql: str) -> list:
    db = sqlite3.connect(":memory:")
    for name, (columns, rows) in TABLES.items():
        db.execute(f"CREATE TABLE {name} ({columns})")
        db.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' for _ in rows[0])})", rows)
    return _rounded(db.execute(sqlglot.transpile(sql, read="duckdb", write="sqlite")[0]).fetchall())


RUN = [(written, read) for written, read in READ if read is not None]


@pytest.mark.parametrize("written, read", RUN, ids=[written[:70] for written, _ in RUN])
def test_the_engines_return_the_same_rows_for_both_spellings(written, read):
    for rows in (_duckdb_rows, _sqlite_rows):
        assert rows(written) == rows(rewrite(written)), rows.__name__
        assert rows(written) == rows(read), rows.__name__


@pytest.mark.parametrize("written, read", RUN, ids=[written[:70] for written, _ in RUN])
def test_empty_tables_give_no_row_both_ways(written, read, monkeypatch):
    for name in TABLES:
        monkeypatch.setitem(TABLES, name, (TABLES[name][0], []))
    assert _duckdb_rows(written) == _duckdb_rows(rewrite(written)) == []
