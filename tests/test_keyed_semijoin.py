"""A ``DISTINCT`` list that holds a key of one joined table asks which rows of that table have a partner (``keyed_semijoin``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.keyed_semijoin import distinct_key_join_to_exists

KEYS = {"product": [("id",)]}
NOT_NULL = {"product": frozenset({"id"})}
JOINED = "SELECT DISTINCT p.id, p.name FROM product AS p JOIN sales AS s ON p.id = s.pid WHERE s.day < 5"
EXISTS = "SELECT p.id, p.name FROM product AS p WHERE EXISTS(SELECT 1 FROM sales AS s WHERE p.id = s.pid AND s.day < 5)"


def rewritten(sql: str, keys=KEYS, not_null=NOT_NULL, types=None) -> str | None:
    result = distinct_key_join_to_exists(sqlglot.parse_one(sql, read="mysql"), keys, not_null, types)
    return None if result is None else result.sql(dialect="mysql")


def test_a_distinct_list_with_the_key_is_the_keyed_table_filtered_by_exists():
    assert rewritten(JOINED) == EXISTS
    # the key read from the other table, which the join makes equal to it; the output keeps its name
    assert rewritten(JOINED.replace("DISTINCT p.id", "DISTINCT s.pid")) == EXISTS.replace("SELECT p.id,", "SELECT p.id AS pid,")
    # a comma join, the keyed table second, a condition on the keyed table alone stays outside
    comma = "SELECT DISTINCT p.name AS n, p.id FROM sales AS s, product AS p WHERE p.id = s.pid AND p.price > 3"
    assert rewritten(comma) == "SELECT p.name AS n, p.id FROM product AS p WHERE p.price > 3 AND EXISTS(SELECT 1 FROM sales AS s WHERE p.id = s.pid)"
    # a subquery that reads the other table goes inside with it
    nested = JOINED + " AND NOT EXISTS (SELECT 1 FROM sales AS t WHERE t.pid = s.pid AND t.day > 9)"
    assert rewritten(nested) == EXISTS[:-1] + " AND NOT EXISTS(SELECT 1 FROM sales AS t WHERE t.pid = s.pid AND t.day > 9))"
    # no condition between the two: every product, when some sale passes
    assert rewritten(JOINED.replace("JOIN sales AS s ON p.id = s.pid", "CROSS JOIN sales AS s")) == "SELECT p.id, p.name FROM product AS p WHERE EXISTS(SELECT 1 FROM sales AS s WHERE s.day < 5)"
    # a two-column key, both listed
    two = "SELECT DISTINCT p.id, p.shop FROM product AS p JOIN sales AS s ON p.id = s.pid AND p.shop = s.shop"
    assert rewritten(two, {"product": [("id", "shop")]}, {"product": frozenset({"id", "shop"})}) == (
        "SELECT p.id, p.shop FROM product AS p WHERE EXISTS(SELECT 1 FROM sales AS s WHERE p.id = s.pid AND p.shop = s.shop)"
    )


@pytest.mark.parametrize(
    "sql, keys, not_null",
    [
        (JOINED, None, NOT_NULL),
        (JOINED, KEYS, None),
        (JOINED, KEYS, {"product": frozenset({"name"})}),  # two rows whose key is NULL are two rows and one tuple
        (JOINED, {"product": [("id", "shop")]}, {"product": frozenset({"id", "shop"})}),  # the list holds half the key
        (JOINED, {"sales": [("pid",)]}, {"sales": frozenset({"pid"})}),  # the other table's key: p.name is not a column of it
        (JOINED.replace("SELECT DISTINCT", "SELECT"), KEYS, NOT_NULL),  # every partner is a row
        (JOINED.replace("DISTINCT p.id, p.name", "DISTINCT p.name"), KEYS, NOT_NULL),  # two products of one name are one row
        (JOINED.replace("DISTINCT p.id, p.name", "DISTINCT p.id, s.day"), KEYS, NOT_NULL),  # a column of the partner
        (JOINED.replace("DISTINCT p.id, p.name", "DISTINCT p.id, UPPER(p.name) AS name"), KEYS, NOT_NULL),
        (JOINED.replace("JOIN sales", "LEFT JOIN sales"), KEYS, NOT_NULL),
        (JOINED + " ORDER BY p.id LIMIT 3", KEYS, NOT_NULL),
        (JOINED.replace("p.id, p.name FROM", "p.id, COUNT(*) AS n FROM") + " GROUP BY p.id", KEYS, NOT_NULL),
        (JOINED.replace("JOIN sales AS s ON p.id = s.pid", "JOIN sales AS s ON p.id = s.pid JOIN shop AS h ON h.id = s.shop"), KEYS, NOT_NULL),  # three tables
        (JOINED.replace("s.day < 5", "day < 5"), KEYS, NOT_NULL),  # a column that does not name its table
        (JOINED.replace("s.day < 5", "s.day < RAND()"), KEYS, NOT_NULL),
        (JOINED.replace("FROM product AS p JOIN sales AS s", "FROM (SELECT * FROM product) AS p JOIN sales AS s"), KEYS, NOT_NULL),
    ],
)
def test_other_selects_are_left_alone(sql, keys, not_null):
    assert rewritten(sql, keys, not_null) is None


def test_a_floating_point_key_is_left_alone():
    assert rewritten(JOINED, types={"product": {"id": "DOUBLE"}}) is None
    assert rewritten(JOINED, types={"product": {"id": "INT"}}) == EXISTS


SPELLINGS = [
    JOINED,
    JOINED.replace("JOIN sales AS s ON p.id = s.pid", "CROSS JOIN sales AS s"),
    JOINED.replace("DISTINCT p.id", "DISTINCT s.pid"),
    "SELECT DISTINCT p.name AS n, p.id FROM sales AS s, product AS p WHERE p.id = s.pid AND p.price > 3",
    JOINED + " AND NOT EXISTS (SELECT 1 FROM sales AS t WHERE t.pid = s.pid AND t.day > 9)",
    JOINED + " AND s.pid NOT IN (SELECT t.pid FROM sales AS t WHERE t.day > 9)",
]


@pytest.mark.parametrize("seed", range(3))
def test_the_exists_form_returns_the_rows_duckdb_returns_when_the_key_holds(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    differs_without_the_key = 0
    for sql in SPELLINGS:
        filtered = rewritten(sql)
        assert filtered is not None, sql
        for keyed in (True, True, False, False):
            con = duckdb.connect()
            con.execute("CREATE TABLE product (id INT, name VARCHAR, price INT)")
            con.execute("CREATE TABLE sales (pid INT, day INT)")
            for row in range(rng.randint(0, 8)):
                key = row if keyed else rng.choice([None, 1, 1, 1, 2])  # without the key two rows can be one tuple
                con.execute("INSERT INTO product VALUES (?, ?, ?)", [key, rng.choice([None, "a", "a", "a", "b"]), rng.choice([None, 2, 5, 5])])
            for _ in range(rng.randint(0, 9)):
                con.execute("INSERT INTO sales VALUES (?, ?)", [rng.choice([None, 0, 1, 1, 2, 3]), rng.choice([None, 1, 4, 7, 12])])
            same = sorted(con.execute(sql).fetchall(), key=repr) == sorted(con.execute(filtered).fetchall(), key=repr)
            if keyed:
                assert same, sql
            else:
                differs_without_the_key += not same
    assert differs_without_the_key > 0  # the key is what makes it true


SCHEMA = {"product": ["id", "name", "price"], "sales": ["pid", "day", "shop"]}


def test_a_nested_one_table_query_written_with_bare_names_is_read():
    from kumo_sql_prover.keyed_semijoin import name_nested_sources

    bare = JOINED + " AND p.id NOT IN (SELECT pid FROM sales WHERE day > 9)"
    assert rewritten(bare) is None  # without the column list a bare name has no certain owner
    result = distinct_key_join_to_exists(sqlglot.parse_one(bare, read="mysql"), KEYS, NOT_NULL, None, SCHEMA)
    assert result.sql(dialect="mysql") == (
        "SELECT p.id, p.name FROM product AS p WHERE NOT p.id IN (SELECT sales.pid FROM sales WHERE sales.day > 9) "
        "AND EXISTS(SELECT 1 FROM sales AS s WHERE p.id = s.pid AND s.day < 5)"
    )

    def named(sql: str) -> str:
        return name_nested_sources(sqlglot.parse_one(sql, read="mysql"), SCHEMA).sql(dialect="mysql")

    head = "SELECT p.id FROM product AS p WHERE "
    # the nested table's alias is the name its columns get
    assert named(head + "EXISTS (SELECT 1 FROM sales AS t WHERE pid = p.id AND day > 9)") == head + "EXISTS(SELECT 1 FROM sales AS t WHERE t.pid = p.id AND t.day > 9)"
    # a name the nested table does not have reads an outer row and stays bare; so does a bare name of the select itself
    assert named(head + "EXISTS (SELECT 1 FROM sales WHERE pid = id)") == head + "EXISTS(SELECT 1 FROM sales WHERE sales.pid = id)"
    assert named("SELECT id FROM product AS p WHERE price > 3") == "SELECT id FROM product AS p WHERE price > 3"
    # two sources, a GROUP BY that may read an output name, a table the caller did not list, a WITH table of that name
    for kept in (
        head + "EXISTS(SELECT 1 FROM sales AS t JOIN product AS q ON q.id = t.pid WHERE day > 9)",
        head + "p.id IN (SELECT pid FROM returns WHERE day > 9)",
        "WITH sales AS (SELECT 1 AS pid) SELECT p.id FROM product AS p WHERE p.id IN (SELECT pid FROM sales)",
    ):
        assert named(kept) == kept
    grouped = named(head + "p.id IN (SELECT pid FROM sales WHERE day > 9 GROUP BY pid)")
    assert grouped == head + "p.id IN (SELECT sales.pid FROM sales WHERE sales.day > 9 GROUP BY pid)"
    # a deeper query's names are its own
    deep = named(head + "p.id IN (SELECT pid FROM sales WHERE shop IN (SELECT id FROM product WHERE price > 3))")
    assert deep == head + "p.id IN (SELECT sales.pid FROM sales WHERE sales.shop IN (SELECT product.id FROM product WHERE product.price > 3))"


def test_the_named_form_returns_the_rows_duckdb_returns():
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(7)
    bare = JOINED + " AND p.id NOT IN (SELECT pid FROM sales WHERE day > 9 AND pid IS NOT NULL)"
    result = distinct_key_join_to_exists(sqlglot.parse_one(bare, read="mysql"), KEYS, NOT_NULL, None, SCHEMA).sql(dialect="mysql")
    for _ in range(25):
        con = duckdb.connect()
        con.execute("CREATE TABLE product (id INT PRIMARY KEY, name VARCHAR, price INT)")
        con.execute("CREATE TABLE sales (pid INT, day INT, shop INT)")
        for key in rng.sample(range(6), rng.randint(0, 5)):
            con.execute("INSERT INTO product VALUES (?, ?, ?)", [key, rng.choice([None, "a", "b"]), rng.choice([None, 2, 5])])
        for _ in range(rng.randint(0, 9)):
            con.execute("INSERT INTO sales VALUES (?, ?, ?)", [rng.choice([None, 0, 1, 1, 2, 3]), rng.choice([None, 1, 4, 7, 12]), rng.choice([None, 0, 1, 2])])
        assert sorted(con.execute(bare).fetchall(), key=repr) == sorted(con.execute(result).fetchall(), key=repr)


def test_a_nested_query_that_reuses_a_source_name_gets_a_fresh_one():
    from kumo_sql_prover.keyed_semijoin import rename_nested_sources

    # two different rows are read under the name s: the nested one is renamed, and its test stays outside the probe
    reused = JOINED + " AND EXISTS (SELECT 1 FROM sales AS s WHERE s.day = 1)"
    assert rewritten(reused) == (
        "SELECT p.id, p.name FROM product AS p WHERE EXISTS(SELECT 1 FROM sales AS kns0_s WHERE kns0_s.day = 1) "
        "AND EXISTS(SELECT 1 FROM sales AS s WHERE p.id = s.pid AND s.day < 5)"
    )

    def apart(sql: str, *taken: str) -> str:
        return rename_nested_sources(sqlglot.parse_one(sql, read="mysql"), set(taken)).sql(dialect="mysql")

    head = "SELECT p.id FROM product AS p JOIN sales ON p.id = sales.pid WHERE "
    plain = head + "p.id NOT IN (SELECT sales.pid FROM sales WHERE sales.day > 9)"
    assert apart(plain, "p", "sales") == head + "NOT p.id IN (SELECT kns0_sales.pid FROM sales AS kns0_sales WHERE kns0_sales.day > 9)"
    # a deeper query that has its own source of that name keeps its reads; a read of the outer p is not touched
    deep = head + "EXISTS (SELECT 1 FROM sales WHERE sales.pid = p.id AND sales.shop IN (SELECT sales.shop FROM sales WHERE sales.day = 1))"
    assert apart(deep, "p", "sales") == head + (
        "EXISTS(SELECT 1 FROM sales AS kns0_sales WHERE kns0_sales.pid = p.id AND kns0_sales.shop IN "
        "(SELECT kns1_sales.shop FROM sales AS kns1_sales WHERE kns1_sales.day = 1))"
    )
    # no clash, or a derived table below the clash: as written
    assert apart(plain, "p", "s") == sqlglot.parse_one(plain, read="mysql").sql(dialect="mysql")
    derived = head + "p.id IN (SELECT sales.pid FROM sales JOIN (SELECT 1 AS one) AS d ON TRUE)"
    assert apart(derived, "p", "sales") == sqlglot.parse_one(derived, read="mysql").sql(dialect="mysql")


def test_the_renamed_form_returns_the_rows_duckdb_returns():
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(11)
    spellings = [
        "SELECT DISTINCT product.id, product.name FROM product JOIN sales ON product.id = sales.pid "
        "WHERE product.id NOT IN (SELECT sales.pid FROM sales WHERE sales.day > 9 AND sales.pid IS NOT NULL)",
        "SELECT DISTINCT sales.pid, product.name FROM sales JOIN product ON product.id = sales.pid WHERE sales.day < 8 "
        "AND NOT EXISTS (SELECT 1 FROM sales WHERE sales.pid = product.id AND sales.day > 9)",
        "SELECT DISTINCT p.id, p.name FROM product AS p JOIN sales AS s ON p.id = s.pid "
        "WHERE p.id NOT IN (SELECT p.id FROM product AS p JOIN sales AS s ON p.id = s.pid WHERE s.day > 9)",
    ]
    for sql in spellings:
        result = distinct_key_join_to_exists(sqlglot.parse_one(sql, read="mysql"), KEYS, NOT_NULL, None, SCHEMA)
        assert result is not None, sql
        for _ in range(20):
            con = duckdb.connect()
            con.execute("CREATE TABLE product (id INT PRIMARY KEY, name VARCHAR, price INT)")
            con.execute("CREATE TABLE sales (pid INT, day INT, shop INT)")
            for key in rng.sample(range(6), rng.randint(0, 5)):
                con.execute("INSERT INTO product VALUES (?, ?, ?)", [key, rng.choice([None, "a", "b"]), rng.choice([None, 2, 5])])
            for _ in range(rng.randint(0, 9)):
                con.execute("INSERT INTO sales VALUES (?, ?, ?)", [rng.choice([None, 0, 1, 1, 2, 3]), rng.choice([None, 1, 4, 7, 12]), rng.choice([None, 0, 1, 2])])
            assert sorted(con.execute(sql).fetchall(), key=repr) == sorted(con.execute(result.sql(dialect="duckdb")).fetchall(), key=repr), sql
