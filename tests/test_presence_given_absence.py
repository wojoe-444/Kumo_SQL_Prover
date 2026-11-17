"""A presence test need not repeat what an absence test beside it already says (``presence_given_absence``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.presence_given_absence import drop_implied_presence_filters

SCHEMA = {"product": ["id", "price"], "sales": ["pid", "day", "shop"]}
DAY = {"sales": frozenset({"day"})}
HEAD = "SELECT p.id FROM product AS p WHERE "
NONE_OUTSIDE = "NOT EXISTS (SELECT 1 FROM sales AS t WHERE t.pid = p.id AND (t.day < 5 OR t.day > 9))"
INSIDE = "p.id IN (SELECT s.pid FROM sales AS s WHERE s.day >= 5 AND s.day <= 9)"
LISTED = HEAD + INSIDE + " AND " + NONE_OUTSIDE
ASKED = HEAD + "EXISTS (SELECT 1 FROM sales AS s WHERE s.pid = p.id AND s.day BETWEEN 5 AND 9) AND " + NONE_OUTSIDE
OUT = "NOT EXISTS(SELECT 1 FROM sales AS t WHERE t.pid = p.id AND (t.day < 5 OR t.day > 9))"


def dropped(sql: str, not_null=DAY, schema=SCHEMA) -> str | None:
    result = drop_implied_presence_filters(sqlglot.parse_one(sql, read="mysql"), not_null, schema)
    return None if result is None else result.sql(dialect="mysql")


def test_the_presence_test_keeps_only_what_the_absence_test_does_not_say():
    assert dropped(LISTED) == HEAD + "p.id IN (SELECT s.pid FROM sales AS s) AND " + OUT
    assert dropped(ASKED) == HEAD + "EXISTS(SELECT 1 FROM sales AS s WHERE s.pid = p.id) AND " + OUT
    # the order of the two tests, and a condition the absence test says nothing about
    assert dropped(HEAD + NONE_OUTSIDE + " AND " + INSIDE) == HEAD + OUT + " AND p.id IN (SELECT s.pid FROM sales AS s)"
    kept = dropped(LISTED.replace("s.day <= 9)", "s.day <= 9 AND s.shop = 1)"))
    assert kept == HEAD + "p.id IN (SELECT s.pid FROM sales AS s WHERE s.shop = 1) AND " + OUT
    # a narrower bound stays: only the bound the absence test gives is dropped
    assert dropped(LISTED.replace("s.day >= 5", "s.day >= 6")) == HEAD + "p.id IN (SELECT s.pid FROM sales AS s WHERE s.day >= 6) AND " + OUT
    # half of the range is all the absence test says
    half = HEAD + INSIDE + " AND NOT EXISTS (SELECT 1 FROM sales AS t WHERE t.pid = p.id AND t.day < 5)"
    assert dropped(half) == HEAD + "p.id IN (SELECT s.pid FROM sales AS s WHERE s.day <= 9) AND NOT EXISTS(SELECT 1 FROM sales AS t WHERE t.pid = p.id AND t.day < 5)"


def test_other_spellings_of_the_same_conditions():
    between = LISTED.replace("(t.day < 5 OR t.day > 9)", "NOT t.day BETWEEN 5 AND 9")
    assert dropped(between) == HEAD + "p.id IN (SELECT s.pid FROM sales AS s) AND NOT EXISTS(SELECT 1 FROM sales AS t WHERE t.pid = p.id AND NOT t.day BETWEEN 5 AND 9)"
    turned = LISTED.replace("s.day >= 5 AND s.day <= 9", "9 >= s.day AND 5 <= s.day")
    assert dropped(turned) == HEAD + "p.id IN (SELECT s.pid FROM sales AS s) AND " + OUT
    listed = HEAD + "p.id IN (SELECT s.pid FROM sales AS s WHERE s.day <> 1 AND s.day <> 2) AND NOT EXISTS (SELECT 1 FROM sales AS t WHERE p.id = t.pid AND t.day IN (1, 2))"
    assert dropped(listed) == HEAD + "p.id IN (SELECT s.pid FROM sales AS s) AND NOT EXISTS(SELECT 1 FROM sales AS t WHERE p.id = t.pid AND t.day IN (1, 2))"
    # bare names: the subquery's own table first, and the tested expression stands outside it
    bare = "SELECT id FROM product WHERE id IN (SELECT pid FROM sales WHERE day BETWEEN 5 AND 9) AND NOT EXISTS (SELECT 1 FROM sales WHERE NOT day BETWEEN 5 AND 9 AND pid = product.id)"
    assert dropped(bare) == "SELECT id FROM product WHERE id IN (SELECT pid FROM sales) AND NOT EXISTS(SELECT 1 FROM sales WHERE NOT day BETWEEN 5 AND 9 AND pid = product.id)"


@pytest.mark.parametrize(
    "sql",
    [
        LISTED.replace("t.pid = p.id", "t.pid = p.price"),  # the absence test asks about other rows
        LISTED.replace("FROM sales AS t", "FROM returns AS t"),  # another table
        LISTED.replace("t.pid = p.id AND ", ""),  # no tie: it asks about every sale, and is left to the other rules
        LISTED.replace("t.pid = p.id", "t.pid = p.id AND t.shop = p.price"),  # a tie the presence test does not have
        LISTED.replace("(t.day < 5 OR t.day > 9)", "t.day < 5 AND t.shop = 1"),  # two conditions: a row need only fail one
        LISTED.replace("(t.day < 5 OR t.day > 9)", "(t.day < 5 OR t.day > p.price)"),  # a comparison with the outer row
        LISTED.replace("(t.day < 5 OR t.day > 9)", "(t.shop < 5 OR t.shop > 9)").replace("s.day", "s.shop"),  # shop may be NULL
        LISTED.replace(INSIDE, "p.id NOT IN (SELECT s.pid FROM sales AS s WHERE s.day >= 5 AND s.day <= 9)"),
        LISTED.replace(INSIDE + " AND", "(" + INSIDE + " OR p.price = 1) AND"),  # the presence test under OR
        HEAD + INSIDE + " OR " + NONE_OUTSIDE,
        LISTED.replace("s.day <= 9)", "s.day <= 9 LIMIT 1)"),
        LISTED.replace("SELECT s.pid FROM sales AS s", "SELECT MAX(s.pid) FROM sales AS s"),
        LISTED.replace("FROM sales AS s", "FROM sales AS s JOIN product AS q ON q.id = s.pid"),
        LISTED.replace("SELECT 1 FROM sales AS t", "SELECT 1 FROM sales AS t LIMIT 0"),
        LISTED.replace("SELECT 1 FROM sales AS t WHERE", "SELECT COUNT(*) FROM sales AS t WHERE"),  # one row always: never "none"
        LISTED.replace("s.day >= 5 AND s.day <= 9", "s.day >= 6 AND s.day <= 8"),  # not what the absence test says
        LISTED.replace("s.day >= 5 AND s.day <= 9", "(s.day >= 5 OR s.shop = 1) AND s.day < 10"),
        LISTED.replace("t.day > 9", "t.day > RAND()"),
        HEAD + INSIDE,  # nothing says there is none outside
    ],
)
def test_other_selects_are_left_alone(sql):
    assert dropped(sql) is None


def test_it_needs_the_column_declared_not_null_and_an_owner_for_every_name():
    assert dropped(LISTED, None) is None
    assert dropped(LISTED, {"sales": frozenset({"pid"})}) is None
    assert dropped(LISTED, {"product": frozenset({"day"})}) is None
    bare = LISTED.replace("t.day", "day")
    assert dropped(bare, DAY, None) is None  # without the column list a bare name has no certain owner
    assert dropped(bare) is not None
    # an outer name that the outer select's sources do not settle
    two = "SELECT p.id FROM product AS p, product AS q WHERE id IN (SELECT s.pid FROM sales AS s WHERE s.day >= 5) AND NOT EXISTS (SELECT 1 FROM sales AS t WHERE t.pid = id AND t.day < 5)"
    assert dropped(two) is None


THROUGH = (
    "SELECT c.id FROM (SELECT p.id, p.price AS cost FROM product AS p WHERE EXISTS (SELECT 1 FROM sales AS s WHERE p.id = s.pid AND s.day BETWEEN 5 AND 9)) AS c "
    "WHERE NOT EXISTS (SELECT 1 FROM sales AS t WHERE (t.day < 5 OR t.day > 9) AND c.id = t.pid)"
)


def test_an_absence_test_asked_of_a_derived_table_that_only_filters():
    assert dropped(THROUGH) == (
        "SELECT c.id FROM (SELECT p.id, p.price AS cost FROM product AS p WHERE EXISTS(SELECT 1 FROM sales AS s WHERE p.id = s.pid)) AS c "
        "WHERE NOT EXISTS(SELECT 1 FROM sales AS t WHERE (t.day < 5 OR t.day > 9) AND c.id = t.pid)"
    )
    # the tie goes through the output's name to the column behind it
    renamed = THROUGH.replace("SELECT p.id, p.price AS cost", "SELECT p.id AS k, p.price AS cost").replace("c.id", "c.k")
    assert dropped(renamed) is not None
    for kept in (
        THROUGH.replace("c.id = t.pid", "c.cost = t.pid"),  # another column of the row
        THROUGH.replace("FROM product AS p WHERE", "FROM product AS p JOIN sales AS x ON x.pid = p.id WHERE"),
        THROUGH.replace("SELECT p.id, p.price AS cost", "SELECT DISTINCT p.id, p.price AS cost"),
        THROUGH.replace("SELECT p.id, p.price AS cost", "SELECT p.id + 0 AS id, p.price AS cost"),
        THROUGH.replace(") AS c WHERE", ") AS c JOIN product AS q ON q.id = c.id WHERE"),  # a join reads the derived table before the filter
        THROUGH.replace(") AS c WHERE", ") AS c LEFT JOIN product AS q ON q.id = c.id WHERE"),
        THROUGH.replace("FROM sales AS t", "FROM sales AS p"),  # the test's own table has the name of the table behind c
        THROUGH.replace("FROM sales AS t", "FROM sales AS c").replace("t.day", "c.day").replace("t.pid", "c.pid"),
        THROUGH.replace("s.day BETWEEN 5 AND 9))", "s.day BETWEEN 5 AND 9) LIMIT 3)"),
    ):
        assert dropped(kept) is None, kept


def _tables(con, rng, null_days: bool):
    con.execute("CREATE TABLE product (id INT, price INT)")
    con.execute("CREATE TABLE sales (pid INT, day INT, shop INT)")
    for _ in range(rng.randint(0, 6)):
        con.execute("INSERT INTO product VALUES (?, ?)", [rng.choice([None, 0, 1, 2, 3]), rng.choice([None, 2, 5, 5])])
    for _ in range(rng.randint(0, 9)):
        day = rng.choice([None, 1, 4, 5, 7, 9, 12] if null_days else [1, 4, 5, 7, 9, 12])
        con.execute("INSERT INTO sales VALUES (?, ?, ?)", [rng.choice([None, 0, 1, 1, 2, 3]), day, rng.choice([None, 0, 1, 2])])


SPELLINGS = [
    LISTED,
    ASKED,
    LISTED.replace("s.day <= 9)", "s.day <= 9 AND s.shop = 1)"),
    LISTED.replace("(t.day < 5 OR t.day > 9)", "NOT t.day BETWEEN 5 AND 9"),
    HEAD + INSIDE + " AND NOT EXISTS (SELECT 1 FROM sales AS t WHERE t.pid = p.id AND t.day < 5)",
    HEAD + "p.id IN (SELECT s.pid FROM sales AS s WHERE s.day <> 1 AND s.day <> 4) AND NOT EXISTS (SELECT 1 FROM sales AS t WHERE p.id = t.pid AND t.day IN (1, 4))",
    HEAD + "p.price + 1 IN (SELECT s.pid FROM sales AS s WHERE s.day > 4) AND NOT EXISTS (SELECT 1 FROM sales AS t WHERE t.pid = p.price + 1 AND t.day <= 4)",
    "SELECT id FROM product WHERE id IN (SELECT pid FROM sales WHERE day BETWEEN 5 AND 9) AND NOT EXISTS (SELECT 1 FROM sales WHERE NOT day BETWEEN 5 AND 9 AND pid = product.id)",
    THROUGH,
    THROUGH.replace("SELECT c.id FROM", "SELECT c.id, COUNT(*) OVER () FROM"),
    THROUGH.replace("SELECT c.id FROM", "SELECT COUNT(*), MAX(c.cost) FROM"),
]


@pytest.mark.parametrize("seed", range(3))
def test_the_shorter_form_returns_the_rows_duckdb_returns(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    for sql in SPELLINGS:
        short = dropped(sql)
        assert short is not None, sql
        for _ in range(8):
            con = duckdb.connect()
            _tables(con, rng, null_days=False)
            assert sorted(con.execute(sql).fetchall(), key=repr) == sorted(con.execute(short).fetchall(), key=repr), sql


def test_a_day_that_may_be_null_is_why_the_declaration_is_needed():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE product (id INT, price INT)")
    con.execute("CREATE TABLE sales (pid INT, day INT, shop INT)")
    con.execute("INSERT INTO product VALUES (1, 2)")
    con.execute("INSERT INTO sales VALUES (1, NULL, 0)")  # a sale on no known day: neither inside the range nor outside it
    short = HEAD + "p.id IN (SELECT s.pid FROM sales AS s) AND " + NONE_OUTSIDE
    assert con.execute(LISTED).fetchall() == [] and con.execute(short).fetchall() == [(1,)]
    assert dropped(LISTED, {"sales": frozenset({"pid", "shop"})}) is None


def test_the_prover_reads_both_forms_as_one_query():
    from kumo_sql_prover.smt_equivalence import TableConstraints

    def facts(*sales_columns):
        return {"sales": TableConstraints(not_null=frozenset(sales_columns)), "product": TableConstraints(not_null=frozenset({"id"}))}

    def proven(left, right, constraints):
        return prove_equivalent_algebraic(left, right, dialect="mysql", schema=SCHEMA, constraints=constraints).status.value == "proven_equivalent"

    short = HEAD + "EXISTS (SELECT 1 FROM sales AS s WHERE s.pid = p.id) AND " + NONE_OUTSIDE
    assert proven(ASKED, short, facts("day", "pid"))
    assert proven(LISTED, short, facts("day", "pid"))
    # with a day that may be NULL the two differ (the test above), and with another range they differ too
    assert not proven(ASKED, short, facts("pid"))
    assert not proven(ASKED.replace("BETWEEN 5 AND 9", "BETWEEN 6 AND 9"), short, facts("day", "pid"))
