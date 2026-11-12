"""A quoted name that needs no quotes is the unquoted name (``kumo_sql_prover.quoted_names``)."""

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.quoted_names import unquote_plain_names

SCHEMA = {"p": ["id", "kind", "day"], "u": ["id", "name"]}
TYPES = {"p": {"id": "INT", "kind": "INT", "day": "INT"}, "u": {"id": "INT", "name": "TEXT"}}


def unquoted(sql, dialect="postgres"):
    return unquote_plain_names(sqlglot.parse_one(sql, read=dialect), dialect).sql(dialect=dialect)


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


def test_plain_names_lose_their_quotes():
    assert unquoted('SELECT "t0"."id" AS "post_id", "t0"."kind" FROM "p" AS "t0" WHERE "t0"."day" > 3') == "SELECT t0.id AS post_id, t0.kind FROM p AS t0 WHERE t0.day > 3"
    assert unquoted('SELECT "d"."a_1" FROM (SELECT "p"."id" AS "a_1" FROM "p") AS "d"', "duckdb") == "SELECT d.a_1 FROM (SELECT p.id AS a_1 FROM p) AS d"


@pytest.mark.parametrize("sql", [
    'SELECT "p"."Id" FROM "p"'.replace('"p"', "p"),  # an upper-case letter: unquoted it would be folded
    'SELECT p."my col" FROM p',
    'SELECT p."1st" FROM p',
    'SELECT p."a.b" FROM p',
    'SELECT p."order", p."select", p."user", p."limit" FROM p AS p',  # reserved words
    'SELECT "table"."end" FROM p AS "table" GROUP BY "table"."end"',
    'SELECT p.id AS "from" FROM p',
    "SELECT 'id' AS a FROM p",  # a string, not a name
])
def test_other_names_keep_their_quotes(sql):
    assert unquoted(sql) == sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres")


def test_other_dialects_are_left_alone():
    # BigQuery and MySQL quote with backticks and compare names by their own rules; Snowflake folds to upper case
    for dialect, sql in (("bigquery", "SELECT `t0`.`id` FROM `p` AS `t0`"), ("mysql", "SELECT `t0`.`id` FROM `p` AS `t0`"), ("snowflake", 'SELECT "t0"."id" FROM "p" AS "t0"')):
        assert unquoted(sql, dialect) == sqlglot.parse_one(sql, read=dialect).sql(dialect=dialect)


RANKED = "SELECT s.id, s.r FROM (SELECT p.id, RANK() OVER (PARTITION BY p.kind ORDER BY p.day DESC) AS r FROM p) AS s WHERE s.r <= 2"
TARGETS = [
    # a select with a window is matched by its text: the quotes made it another text
    (RANKED, 'SELECT "s"."id", "s"."r" FROM (SELECT "p"."id", RANK() OVER (PARTITION BY "p"."kind" ORDER BY "p"."day" DESC) AS "r" FROM "p") AS "s" WHERE "s"."r" <= 2'),
    (RANKED, 'SELECT "t1"."id", "t1"."w0" FROM (SELECT "t0"."id", RANK() OVER (PARTITION BY "t0"."kind" ORDER BY "t0"."day" DESC) AS "w0" FROM "p" AS "t0") AS "t1" WHERE "t1"."w0" <= 2'),
    ("SELECT p.id, u.name FROM p JOIN u ON p.kind = u.id", 'SELECT "p"."id", "u"."name" FROM "p" INNER JOIN "u" ON "p"."kind" = "u"."id"'),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_quoted_spellings_prove(left, right):
    assert proved(left, right)


NEAR_MISSES = [
    # single quotes make a string, double quotes a name
    ("SELECT p.id, 'kind' AS k FROM p", 'SELECT p.id, "kind" AS k FROM p'),
    # another column, quoted or not
    (RANKED, RANKED.replace("PARTITION BY p.kind", 'PARTITION BY "p"."day"')),
    (RANKED, RANKED.replace("p.day DESC", '"p"."day"')),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    duckdb = pytest.importorskip("duckdb")
    assert not proved(left, right)
    db = duckdb.connect()
    db.execute("CREATE TABLE p (id BIGINT, kind BIGINT, day BIGINT)")
    db.executemany("INSERT INTO p VALUES (?, ?, ?)", [(1, 1, 10), (2, 1, 20), (3, 1, 30), (4, 2, 5), (5, 2, 7), (6, None, 1)])
    a, b = (sorted(map(repr, db.execute(sqlglot.transpile(q, read="postgres", write="duckdb")[0]).fetchall())) for q in (left, right))
    assert a != b


def test_unquoted_queries_give_the_same_rows():
    duckdb = pytest.importorskip("duckdb")
    db = duckdb.connect()
    db.execute("CREATE TABLE p (id BIGINT, kind BIGINT, day BIGINT)")
    db.execute("CREATE TABLE u (id BIGINT, name VARCHAR)")
    db.executemany("INSERT INTO p VALUES (?, ?, ?)", [(1, 1, 10), (2, 1, 20), (3, 1, 30), (4, 2, 5), (5, 2, 7), (6, None, 1)])
    db.executemany("INSERT INTO u VALUES (?, ?)", [(1, "a"), (2, "b"), (2, "c")])
    for _, sql in TARGETS:
        out = unquoted(sql)
        assert '"' not in out
        assert sorted(map(repr, db.execute(sql).fetchall())) == sorted(map(repr, db.execute(out).fetchall()))
