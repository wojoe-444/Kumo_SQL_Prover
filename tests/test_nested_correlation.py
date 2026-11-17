"""A membership test whose inner test asks about the outer row is two tests of the outer row (``nested_correlation``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.nested_correlation import flatten_nested_correlation, interchangeable

HEAD = "SELECT p.id FROM product AS p WHERE "
LATE = "NOT EXISTS (SELECT 1 FROM sales AS t WHERE t.pid = s.pid AND t.day > 9)"
NESTED = HEAD + "EXISTS (SELECT 1 FROM sales AS s WHERE s.pid = p.id AND s.day < 5 AND " + LATE + ")"
LISTED = HEAD + "p.id IN (SELECT s.pid FROM sales AS s WHERE s.day < 5 AND " + LATE + ")"
ASKED_ONCE = "NOT EXISTS(SELECT 1 FROM sales AS t WHERE t.pid = p.id AND t.day > 9)"
FLAT = HEAD + "EXISTS(SELECT 1 FROM sales AS s WHERE s.pid = p.id AND s.day < 5) AND " + ASKED_ONCE
INTS = {"product": {"id": "INT", "price": "INT"}, "sales": {"pid": "INT", "day": "INT", "shop": "INT"}}


def flattened(sql: str, types=None) -> str | None:
    result = flatten_nested_correlation(sqlglot.parse_one(sql, read="mysql"), types)
    return None if result is None else result.sql(dialect="mysql")


def test_the_inner_test_is_asked_once_about_the_outer_row():
    assert flattened(NESTED) == FLAT
    assert flattened(NESTED, INTS) == FLAT
    assert flattened(NESTED.replace("s.pid = p.id AND", "p.id = s.pid AND")) == FLAT.replace("s.pid = p.id", "p.id = s.pid")
    assert flattened(LISTED) == HEAD + "p.id IN (SELECT s.pid FROM sales AS s WHERE s.day < 5) AND " + ASKED_ONCE
    # the other conditions of the outer select keep their places
    assert flattened(NESTED.replace("WHERE EXISTS", "WHERE p.price > 3 AND EXISTS")) == FLAT.replace("WHERE EXISTS", "WHERE p.price > 3 AND EXISTS")
    # two tests of the outer row, each with its own inner test
    second = " AND EXISTS (SELECT 1 FROM sales AS u WHERE u.pid = p.id AND u.day = 7 AND EXISTS (SELECT 1 FROM sales AS v WHERE v.pid = u.pid AND v.day = 8))"
    assert flattened(NESTED + second) == (
        FLAT + " AND EXISTS(SELECT 1 FROM sales AS u WHERE u.pid = p.id AND u.day = 7) AND EXISTS(SELECT 1 FROM sales AS v WHERE v.pid = p.id AND v.day = 8)"
    )


def test_a_membership_test_of_the_tied_column_is_asked_once_too():
    absent = NESTED.replace(LATE, "s.pid NOT IN (SELECT t.pid FROM sales AS t WHERE t.day > 9)")
    assert flattened(absent) == HEAD + "EXISTS(SELECT 1 FROM sales AS s WHERE s.pid = p.id AND s.day < 5) AND NOT p.id IN (SELECT t.pid FROM sales AS t WHERE t.day > 9)"
    # a plain condition on the tied column stays as it is written
    plain = NESTED.replace("s.day < 5", "s.pid < 5")
    assert flattened(plain) == FLAT.replace("s.day < 5", "s.pid < 5")


def test_an_inner_test_that_still_reads_the_row_stays_inside_and_names_the_outer_column():
    still = NESTED.replace("t.pid = s.pid", "t.pid = s.pid AND t.day > s.day")
    assert flattened(still) == (
        HEAD + "EXISTS(SELECT 1 FROM sales AS s WHERE s.pid = p.id AND s.day < 5 AND NOT EXISTS(SELECT 1 FROM sales AS t WHERE t.pid = p.id AND t.day > s.day AND t.day > 9))"
    )


@pytest.mark.parametrize(
    "sql",
    [
        NESTED.replace("WHERE EXISTS", "WHERE NOT EXISTS"),  # no row of s passing is not "the inner test fails"
        NESTED.replace("WHERE EXISTS", "WHERE p.price > 3 OR EXISTS"),  # not a top-level conjunct
        LISTED.replace("p.id IN", "p.id NOT IN"),
        NESTED.replace("s.pid = p.id AND ", "s.pid >= p.id AND "),  # nothing says s.pid is p.id
        NESTED.replace("s.pid = p.id AND s.day < 5 AND", "s.pid = p.id AND s.day < 5 OR"),
        NESTED.replace("t.pid = s.pid", "t.pid = s.shop"),  # tied to a column the outer row does not fix
        NESTED.replace("SELECT 1 FROM sales AS s", "SELECT COUNT(*) FROM sales AS s"),  # one row whatever passes
        NESTED.replace("FROM sales AS t", "FROM sales AS s"),  # the nested query reuses a name
        NESTED.replace("FROM sales AS t", "FROM sales AS p").replace("t.", "p."),
        NESTED.replace("t.day > 9", "day > 9"),  # a column that does not name its table
        NESTED.replace("s.day < 5", "s.day < RAND()"),
        NESTED.replace("s.day < 5 AND", "s.day < 5 GROUP BY s.pid HAVING").replace("s.pid = p.id AND", "s.pid = p.id AND 1 = 1 AND"),
        NESTED.replace("FROM sales AS s WHERE", "FROM (SELECT * FROM sales) AS s WHERE"),
        LISTED.replace("SELECT s.pid FROM", "SELECT s.pid + 1 FROM"),
        HEAD + "EXISTS (SELECT 1 FROM sales AS s WHERE s.pid = p.id AND s.day < 5)",  # nothing nested
    ],
)
def test_other_selects_are_left_alone(sql):
    assert flattened(sql) is None


@pytest.mark.parametrize(
    "product_id, sales_pid, same",
    [
        ("INT", "BIGINT", True),
        ("VARCHAR", "TEXT", True),
        ("DATE", "DATE", True),
        ("DECIMAL(10, 2)", "DECIMAL(10,2)", True),
        ("INT", "DOUBLE", False),  # 1 = 1.0, and 1 / 2 is not 1.0 / 2 everywhere
        ("DOUBLE", "DOUBLE", False),  # 0.0 = -0.0
        ("INT", "VARCHAR", False),  # MySQL: 1 = '1abc'
        ("DECIMAL(10, 2)", "DECIMAL(12, 4)", False),  # 1.00 = 1.0000, and their texts differ
        ("DATE", "DATETIME", False),
        ("INT", None, False),  # one side is not declared
        ("GEOMETRY", "GEOMETRY", False),  # a type the prover does not know
    ],
)
def test_one_column_is_read_for_another_only_when_equal_means_the_same_value(product_id, sales_pid, same):
    types = {"product": {"id": product_id, "price": "INT"}, "sales": {"day": "INT"}}
    if sales_pid is not None:
        types["sales"]["pid"] = sales_pid
    assert interchangeable(types, ("product", "id"), ("sales", "pid")) is same
    assert interchangeable(types, ("sales", "pid"), ("product", "id")) is same
    assert (flattened(NESTED, types) is not None) is same
    assert (flattened(LISTED, types) is not None) is same


def test_without_declared_types_an_equality_is_read_as_the_prover_reads_it_elsewhere():
    assert interchangeable(None, ("product", "id"), ("sales", "pid"))
    assert interchangeable({}, ("product", "id"), (None, "pid"))
    assert not interchangeable(INTS, ("product", "id"), (None, "pid"))  # a derived source has no declared type


def test_columns_the_types_say_nothing_about_are_read_as_with_no_types():
    # A harness may know only which columns hold dates. The two id columns are then as undeclared as with no types.
    dates = {"sales": {"day": "DATE"}}
    assert interchangeable(dates, ("product", "id"), ("sales", "pid"))
    assert flattened(NESTED, dates) is not None and flattened(LISTED, dates) is not None
    assert flattened(NESTED, dates) == flattened(NESTED, None)
    # A derived source may hand on a declared type, so it is not read as undeclared once anything is declared.
    assert not interchangeable(dates, ("product", "id"), (None, "pid"))
    assert not interchangeable(dates, (None, "id"), (None, "pid"))
    # One declared and the other not stays refused.
    assert not interchangeable({"sales": {"pid": "INT"}}, ("product", "id"), ("sales", "pid"))


SPELLINGS = [
    NESTED,
    LISTED,
    NESTED.replace("s.pid = p.id AND", "p.id = s.pid AND"),
    NESTED.replace("t.pid = s.pid", "t.pid = s.pid AND t.day > s.day"),
    NESTED.replace("WHERE EXISTS", "WHERE p.price > 3 AND EXISTS"),
    NESTED.replace(LATE, "s.pid NOT IN (SELECT t.pid FROM sales AS t WHERE t.day > 9 AND t.shop = s.pid)"),
    NESTED.replace(LATE, "s.day < (SELECT MAX(t.day) FROM sales AS t WHERE t.pid = s.pid)"),
    LISTED.replace(LATE, "s.pid IN (SELECT t.shop FROM sales AS t WHERE t.day > 2)"),
    NESTED.replace(LATE, "(" + LATE + " OR EXISTS (SELECT 1 FROM sales AS u WHERE u.shop = s.pid))"),
]


@pytest.mark.parametrize("seed", range(3))
def test_the_flat_form_returns_the_rows_duckdb_returns(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    for sql in SPELLINGS:
        flat = flattened(sql, INTS)
        assert flat is not None, sql
        for _ in range(6):
            con = duckdb.connect()
            con.execute("CREATE TABLE product (id INT, price INT)")
            con.execute("CREATE TABLE sales (pid INT, day INT, shop INT)")
            for _ in range(rng.randint(0, 6)):
                con.execute("INSERT INTO product VALUES (?, ?)", [rng.choice([None, 0, 1, 2, 3]), rng.choice([None, 2, 5, 5])])
            for _ in range(rng.randint(0, 9)):
                con.execute("INSERT INTO sales VALUES (?, ?, ?)", [rng.choice([None, 0, 1, 1, 2, 3]), rng.choice([None, 1, 4, 7, 12]), rng.choice([None, 0, 1, 2])])
            assert sorted(con.execute(sql).fetchall(), key=repr) == sorted(con.execute(flat).fetchall(), key=repr), sql


def test_the_prover_reads_both_forms_as_one_query():
    schema = {"product": ["id", "price"], "sales": ["pid", "day", "shop"]}
    flat = FLAT.replace("EXISTS(", "EXISTS (")
    assert prove_equivalent_algebraic(NESTED, flat, dialect="mysql", schema=schema, types=INTS).status.value == "proven_equivalent"
    assert prove_equivalent_algebraic(LISTED, flat, dialect="mysql", schema=schema, types=INTS).status.value == "proven_equivalent"
    # the inner test dropped, or asked about another day: not the same query
    assert prove_equivalent_algebraic(NESTED, flat.replace("t.day > 9", "t.day > 8"), dialect="mysql", schema=schema, types=INTS).status.value != "proven_equivalent"
    assert prove_equivalent_algebraic(NESTED, flat.split(" AND NOT EXISTS")[0], dialect="mysql", schema=schema, types=INTS).status.value != "proven_equivalent"
    mixed = {"product": {"id": "INT", "price": "INT"}, "sales": {"pid": "VARCHAR", "day": "INT", "shop": "INT"}}
    assert prove_equivalent_algebraic(NESTED, flat, dialect="mysql", schema=schema, types=mixed).status.value != "proven_equivalent"
