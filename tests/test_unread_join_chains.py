"""An outer join nothing reads, in a chain of joins or beside windows over groups (``unread_join_chains``).

Each reading is run as written and as read on DuckDB and on SQLite over made-up tables in which the dropped join
both repeats rows (a post with three votes) and finds none (a post with no vote). The rows must be the same. The
queries the rule leaves alone come with the rows the reading would change where there are such rows.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import TableConstraints
from kumo_sql_prover.unread_join_chains import drop_unread_chained_join

TABLES = {
    "posts": ("id INT, owner INT, kind INT, day INT, title VARCHAR", [(1, 10, 1, 5, "a"), (2, 10, 1, 7, "b"), (3, 11, 2, 7, "c"), (4, None, 2, 9, "d"), (5, 12, 1, 9, "e")]),
    "people": ("id INT, name VARCHAR", [(10, "ann"), (11, "bo"), (13, "cy")]),
    "votes": ("id INT, post INT, kind INT", [(1, 1, 2), (2, 1, 2), (3, 1, 3), (4, 3, 2), (5, 9, 2)]),
    "notes": ("id INT, post INT, body VARCHAR", [(1, 1, "x"), (2, 2, "y"), (3, 2, "z")]),
}
SCHEMA = {"posts": ["id", "owner", "kind", "day", "title"], "people": ["id", "name"], "votes": ["id", "post", "kind"], "notes": ["id", "post", "body"]}
TYPES = {name: dict(part.split() for part in columns.split(", ")) for name, (columns, _) in TABLES.items()}
CONSTRAINTS = {name: TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)) for name in TABLES}

JOINED = "FROM posts AS p JOIN people AS u ON p.owner = u.id"
VOTES = "LEFT JOIN votes AS v ON v.post = p.id AND v.kind = 2"

# (as written, the number of joins the reading drops)
READ = [
    (f"SELECT p.id, p.title, u.name {JOINED} {VOTES} GROUP BY p.id, p.title, u.name", 1),
    (f"SELECT DISTINCT p.id, u.name {JOINED} {VOTES}", 1),
    (f"SELECT DISTINCT p.id, u.name {JOINED} LEFT OUTER JOIN votes AS v ON v.post = p.id LEFT JOIN notes AS n ON n.post = p.id", 2),
    # the unread join in the middle: the join after it does not read it
    ("SELECT DISTINCT p.id, n.body FROM posts AS p LEFT JOIN votes AS v ON v.post = p.id LEFT JOIN notes AS n ON n.post = p.id", 1),
    ("SELECT DISTINCT p.id, u.name FROM posts AS p LEFT JOIN votes AS v ON v.post = p.id AND v.kind IN (2, 3) JOIN people AS u ON p.owner = u.id", 1),
    ("SELECT DISTINCT p.id, u.name FROM posts AS p LEFT JOIN votes AS v ON v.post = p.id, people AS u WHERE p.owner = u.id", 1),
    # aggregates that cannot see repeats
    (f"SELECT u.name, MIN(p.day) AS first, MAX(p.day) AS last, COUNT(DISTINCT p.id) AS posts {JOINED} {VOTES} GROUP BY u.name", 1),
    (f"SELECT u.name, SUM(DISTINCT p.day) AS days {JOINED} {VOTES} GROUP BY u.name HAVING MAX(p.day) > 5", 1),
    # windows over groups: one join is enough
    ("SELECT p.id, ROW_NUMBER() OVER (PARTITION BY p.kind ORDER BY p.day DESC, p.id) AS rn FROM posts AS p LEFT JOIN votes AS v ON v.post = p.id GROUP BY p.id, p.kind, p.day", 1),
    (f"SELECT p.id, u.name, RANK() OVER (ORDER BY p.day) AS r, COUNT(*) OVER (PARTITION BY p.kind) AS same {JOINED} {VOTES} GROUP BY p.id, u.name, p.day, p.kind", 1),
    (f"SELECT u.name, MAX(p.day) AS last, SUM(MAX(p.day)) OVER () AS total {JOINED} {VOTES} GROUP BY u.name", 1),
    # a derived table as the unread source, and columns inside another derived table that have no qualifier
    (f"SELECT DISTINCT p.id, u.name {JOINED} LEFT JOIN (SELECT post, COUNT(*) AS n FROM votes GROUP BY post) AS c ON c.post = p.id", 1),
    ("SELECT DISTINCT p.id, u.name FROM (SELECT id, owner FROM posts WHERE day > 5) AS p JOIN people AS u ON p.owner = u.id LEFT JOIN votes AS v ON v.post = p.id", 1),
    # bare names that are not columns of the joined source, and the columns of another source
    (f"SELECT DISTINCT p.id, name {JOINED} {VOTES} WHERE title <> 'a'", 1),
    (f"SELECT DISTINCT p.*, name {JOINED} LEFT JOIN (SELECT post, COUNT(*) AS n FROM votes GROUP BY post) AS c ON c.post = p.id ORDER BY title", 1),
    # any test in the ON
    (f"SELECT DISTINCT p.id, u.name {JOINED} LEFT JOIN votes AS v ON post IN (SELECT post FROM notes)", 1),
    (f"SELECT DISTINCT p.id, u.name {JOINED} LEFT JOIN votes AS v ON v.post * 2 > p.id + v.kind OR CAST(v.post AS VARCHAR) = p.title", 1),
    (f"SELECT DISTINCT p.id, u.name {JOINED} LEFT JOIN votes AS v ON v.post IN (SELECT n.post FROM notes AS n WHERE n.id < 3)", 1),
    # inside a derived table
    (f"SELECT t.id FROM (SELECT p.id, u.name {JOINED} {VOTES} GROUP BY p.id, u.name) AS t WHERE t.name <> 'bo'", 1),
]
# left as written; dropping the join by hand changes the rows of the first six over the made-up tables
LEFT_ALONE = [
    f"SELECT p.id, u.name {JOINED} {VOTES}",  # repeats are returned
    f"SELECT u.name, COUNT(*) AS n {JOINED} {VOTES} GROUP BY u.name",
    f"SELECT u.name, SUM(p.day) AS days {JOINED} {VOTES} GROUP BY u.name",
    f"SELECT u.name, COUNT(p.id) AS n, MAX(p.day) AS last {JOINED} {VOTES} GROUP BY u.name",
    f"SELECT DISTINCT p.id, COUNT(*) OVER (PARTITION BY p.id) AS n {JOINED} {VOTES}",  # a window of a DISTINCT select reads the repeats
    f"SELECT u.name, MAX(p.day) AS last, SUM(COUNT(*)) OVER () AS total {JOINED} {VOTES} GROUP BY u.name",
    f"SELECT DISTINCT p.id, u.name {JOINED} {VOTES} WHERE v.id IS NULL",  # the joined source is read
    f"SELECT DISTINCT p.id, v.kind {JOINED} {VOTES}",
    f"SELECT u.name, MAX(v.id) AS last {JOINED} {VOTES} GROUP BY u.name",
    "SELECT DISTINCT p.id, u.name FROM posts AS p LEFT JOIN votes AS v ON v.post = p.id JOIN people AS u ON u.id = v.id + 9",  # a later join reads it
    f"SELECT DISTINCT p.id, u.name {JOINED} JOIN votes AS v ON v.post = p.id",  # an inner join removes rows
    f"SELECT DISTINCT p.id, u.name {JOINED} RIGHT JOIN votes AS v ON v.post = p.id",
    f"SELECT DISTINCT p.id, u.name {JOINED} FULL JOIN votes AS v ON v.post = p.id",
    f"SELECT DISTINCT p.id, u.name, kind {JOINED} {VOTES}",  # a bare name the joined source has
    f"SELECT DISTINCT p.id, u.name {JOINED} {VOTES} ORDER BY post",
    f"SELECT DISTINCT p.id, u.name {JOINED} LEFT JOIN (SELECT post, COUNT(*) AS n FROM votes GROUP BY post) AS c ON c.post = p.id WHERE n IS NULL",
    f"SELECT DISTINCT p.id, name {JOINED} LEFT JOIN (SELECT * FROM votes) AS c ON c.post = p.id",  # its columns are not listed
    f"SELECT DISTINCT p.id, name {JOINED} LEFT JOIN ballots AS v ON v.post = p.id",  # a table the schema does not have
    f"WITH votes AS (SELECT 1 AS post, 2 AS name) SELECT DISTINCT p.id, name {JOINED} LEFT JOIN votes AS v ON v.post = p.id",  # the WITH hides the table
    f"SELECT DISTINCT * {JOINED} {VOTES}",
    f"SELECT DISTINCT p.id, v.* {JOINED} {VOTES}",
    f"SELECT DISTINCT p.id, u.name {JOINED} LEFT JOIN votes AS v USING (id)",
    "SELECT DISTINCT p.id, u.name FROM posts AS p LEFT JOIN votes AS v ON v.post = p.id JOIN people AS u USING (id)",
    "SELECT DISTINCT p.id, u.name FROM posts AS p LEFT JOIN votes AS v ON v.post = p.id NATURAL JOIN people AS u",
    f"SELECT DISTINCT p.id, u.name {JOINED} LEFT JOIN votes AS p2 ON p2.post = p.id LEFT JOIN notes AS p2 ON p2.post = p.id",  # one name, two sources
    f"SELECT p.id, u.name {JOINED} {VOTES} GROUP BY ROLLUP (p.id, u.name)",
    f"SELECT DISTINCT ON (p.id) p.id, u.name {JOINED} {VOTES}",
    f"SELECT DISTINCT p.id, some_count(u.name) AS n {JOINED} {VOTES}",  # a function the prover does not know
    "SELECT DISTINCT p.id FROM posts AS p LEFT JOIN votes AS v ON v.post = p.id",  # one join, no window: the older rule reads it
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


def _read(sql: str, schema=SCHEMA) -> tuple[str, int]:
    """The query with the rule applied to every select until nothing changes, and the number of joins dropped."""

    tree, dropped = sqlglot.parse_one(sql, read="postgres"), 0
    while True:
        for select in tree.find_all(sqlglot.exp.Select):
            new = drop_unread_chained_join(select, schema)
            if new is not None:
                if select is tree:
                    tree = new
                else:
                    select.replace(new)
                dropped += 1
                break
        else:
            return tree.sql(dialect="postgres"), dropped


@pytest.mark.parametrize("written, joins", READ)
def test_a_reading_returns_the_rows_of_the_query_as_written(written, joins):
    read, dropped = _read(written)
    assert dropped == joins, read
    for dialect, database in _databases():
        assert _rows(database, written, dialect) == _rows(database, read, dialect), dialect


@pytest.mark.parametrize("written", LEFT_ALONE)
def test_a_query_that_is_not_read_stays_as_written(written):
    assert _read(written)[1] == 0


def test_a_bare_name_is_read_only_with_the_columns_of_the_joined_source():
    written = f"SELECT DISTINCT p.id, name {JOINED} {VOTES} WHERE title <> 'a'"
    assert _read(written)[1] == 1
    assert _read(written, schema=None)[1] == 0
    assert _read(written, schema={"posts": SCHEMA["posts"], "people": SCHEMA["people"]})[1] == 0
    assert _read(written.replace("p.id, name", "p.id, u.name").replace("title", "p.title"), schema=None)[1] == 1


def test_the_rule_copies_the_select_it_is_given():
    select = sqlglot.parse_one(READ[0][0], read="postgres")
    before = select.sql()
    assert drop_unread_chained_join(select) is not None
    assert select.sql() == before


def test_dropping_the_join_of_a_select_that_sees_repeats_changes_rows():
    database = next(db for dialect, db in _databases() if dialect == "duckdb")
    for written in LEFT_ALONE[:6]:
        dropped = written.replace(f" {VOTES}", "")
        assert dropped != written
        assert _rows(database, written, "duckdb") != _rows(database, dropped, "duckdb"), written


def _proved(left: str, right: str) -> bool:
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect="postgres").proven


RANKED = "ROW_NUMBER() OVER (PARTITION BY p.kind ORDER BY p.day DESC) AS rn"
PAIRS = [
    (f"SELECT DISTINCT p.id, p.title, u.name {JOINED} {VOTES}", f"SELECT p.id, p.title, u.name {JOINED}"),
    (
        f"WITH r AS (SELECT p.id, p.title, u.name, COUNT(v.id) AS votes, {RANKED} {JOINED} {VOTES} GROUP BY p.id, p.title, p.day, p.kind, u.name) SELECT id, title, name FROM r WHERE rn <= 2",
        f"WITH r AS (SELECT p.id, p.title, u.name, {RANKED} {JOINED}) SELECT id, title, name FROM r WHERE rn <= 2",
    ),
]


@pytest.mark.parametrize("left, right", PAIRS)
def test_a_pair_is_proved_and_only_with_the_rule(left, right):
    assert _proved(left, right)
    with shape_fallback.without("unread_join_chains"):
        assert not _proved(left, right)


def test_pairs_that_differ_stay_unproved():
    # the count is read, so the join stays; an inner join to votes drops posts without a vote
    assert not _proved(
        f"WITH r AS (SELECT p.id, COUNT(v.id) AS votes, {RANKED} {JOINED} {VOTES} GROUP BY p.id, p.day, p.kind) SELECT id, votes FROM r WHERE rn <= 2",
        f"WITH r AS (SELECT p.id, 0 AS votes, {RANKED} {JOINED}) SELECT id, votes FROM r WHERE rn <= 2",
    )
    assert not _proved(f"SELECT DISTINCT p.id, u.name {JOINED} JOIN votes AS v ON v.post = p.id", f"SELECT p.id, u.name {JOINED}")
    assert not _proved(f"SELECT p.id, u.name {JOINED} {VOTES}", f"SELECT p.id, u.name {JOINED}")
