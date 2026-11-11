"""A ``LEFT JOIN`` kept only where it found nothing is ``NOT EXISTS`` (``anti_join_reading``), and a filter on a
``DISTINCT`` derived table that lists all its columns is a ``DISTINCT`` over the filtered rows (``distinct_source_filters``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.anti_join_reading import read_unmatched_left_join
from kumo_sql_prover.distinct_source_filters import lift_source_distinct

RIGHT = "(SELECT DISTINCT t.buyer FROM sales AS t WHERE t.item = 2)"
JOINED = f"SELECT a.buyer, a.day FROM sales AS a LEFT JOIN {RIGHT} AS b ON a.buyer = b.buyer WHERE b.buyer IS NULL AND a.item = 1"
READ = f"SELECT a.buyer, a.day FROM sales AS a WHERE NOT EXISTS(SELECT 1 FROM {RIGHT} AS b WHERE a.buyer = b.buyer) AND a.item = 1"


def unmatched(sql: str) -> str | None:
    result = read_unmatched_left_join(sqlglot.parse_one(sql, read="mysql"))
    return None if result is None else result.sql(dialect="mysql")


def lifted(sql: str) -> str | None:
    result = lift_source_distinct(sqlglot.parse_one(sql, read="mysql"))
    return None if result is None else result.sql(dialect="mysql")


def test_a_left_join_that_found_nothing_is_not_exists():
    assert unmatched(JOINED) == READ
    # a grouped right side, a comparison other than equality, an earlier join that stays
    grouped = RIGHT.replace("SELECT DISTINCT t.buyer FROM sales AS t WHERE t.item = 2", "SELECT t.buyer, MAX(t.day) AS top FROM sales AS t GROUP BY t.buyer")
    sql = f"SELECT a.buyer FROM sales AS a JOIN items AS i ON a.item = i.item LEFT JOIN {grouped} AS b ON a.buyer = b.buyer AND a.day < b.top WHERE (b.top IS NULL)"
    assert unmatched(sql) == f"SELECT a.buyer FROM sales AS a JOIN items AS i ON a.item = i.item WHERE NOT EXISTS(SELECT 1 FROM {grouped} AS b WHERE a.buyer = b.buyer AND a.day < b.top)"
    # the left side may be a derived table with bare names of its own
    sql = f"SELECT a.buyer FROM (SELECT DISTINCT buyer FROM sales WHERE item = 1) AS a LEFT JOIN {RIGHT} AS b ON a.buyer = b.buyer WHERE b.buyer IS NULL"
    assert unmatched(sql) == f"SELECT a.buyer FROM (SELECT DISTINCT buyer FROM sales WHERE item = 1) AS a WHERE NOT EXISTS(SELECT 1 FROM {RIGHT} AS b WHERE a.buyer = b.buyer)"


@pytest.mark.parametrize(
    "sql",
    [
        JOINED.replace("WHERE b.buyer IS NULL AND", "WHERE"),
        JOINED.replace("b.buyer IS NULL", "b.buyer IS NOT NULL"),
        JOINED.replace("b.buyer IS NULL", "(b.buyer IS NULL OR a.day = 3)"),  # not asked of every row
        JOINED.replace("LEFT JOIN", "JOIN"),
        JOINED.replace("LEFT JOIN", "RIGHT JOIN"),
        JOINED.replace("ON a.buyer = b.buyer", "ON a.buyer <=> b.buyer"),  # a match may have a NULL buyer
        JOINED.replace("ON a.buyer = b.buyer", "ON COALESCE(b.buyer, 0) = a.buyer"),
        JOINED.replace("ON a.buyer = b.buyer", "ON a.buyer = b.buyer OR a.day = 1"),
        JOINED.replace("SELECT a.buyer, a.day FROM", "SELECT a.buyer, b.buyer AS other FROM"),  # the padded column is read
        JOINED.replace("SELECT a.buyer, a.day FROM", "SELECT * FROM"),
        JOINED.replace("AND a.item = 1", "AND item = 1"),  # whose item
        JOINED.replace("AND a.item = 1", "AND a.item IN (SELECT item FROM items WHERE name = b.buyer)"),
        JOINED.replace("AND a.item = 1", "AND RAND() < 0.5"),
        JOINED.replace(f"{RIGHT} AS b", "sales AS b"),  # a base table: other rules read it
        JOINED.replace("SELECT DISTINCT t.buyer", "SELECT t.buyer"),  # a derived table that only filters: other rules read it
        JOINED.replace(" WHERE b.buyer IS NULL", " JOIN items AS i ON i.item = a.item WHERE b.buyer IS NULL"),  # not the last join
        f"SELECT a.buyer FROM sales AS a LEFT JOIN {RIGHT.replace('t.buyer FROM', 't.buyer, t.day FROM')} AS b ON a.buyer = b.buyer WHERE b.day IS NULL",  # a match may have a NULL day
    ],
)
def test_other_left_joins_are_left_alone(sql):
    assert unmatched(sql) is None


INSIDE = "SELECT d.buyer, d.day AS at FROM (SELECT DISTINCT s.day, s.buyer FROM sales AS s WHERE s.item = 1) AS d WHERE NOT EXISTS(SELECT 1 FROM sales AS t WHERE t.item = 2 AND t.buyer = d.buyer)"
OUTSIDE = "SELECT DISTINCT d.buyer, d.day AS at FROM (SELECT s.day, s.buyer FROM sales AS s WHERE s.item = 1) AS d WHERE NOT EXISTS(SELECT 1 FROM sales AS t WHERE t.item = 2 AND t.buyer = d.buyer)"


def test_the_distinct_of_a_filtered_derived_table_moves_outside():
    assert lifted(INSIDE) == OUTSIDE
    assert lifted("SELECT buyer FROM (SELECT DISTINCT buyer FROM sales) AS d WHERE buyer > 3") == "SELECT DISTINCT buyer FROM (SELECT buyer FROM sales) AS d WHERE buyer > 3"


@pytest.mark.parametrize(
    "sql",
    [
        INSIDE.replace("SELECT d.buyer, d.day AS at FROM", "SELECT d.buyer FROM"),  # two days of one buyer are two rows
        INSIDE.replace("SELECT d.buyer, d.day AS at FROM", "SELECT d.buyer, d.day, d.day AS again FROM"),
        INSIDE.replace("SELECT d.buyer, d.day AS at FROM", "SELECT d.buyer, d.day + 1 AS at FROM"),
        INSIDE.replace("SELECT DISTINCT s.day, s.buyer", "SELECT s.day, s.buyer"),
        INSIDE.replace("WHERE s.item = 1)", "WHERE s.item = 1 LIMIT 5)"),
        INSIDE.replace("WHERE s.item = 1)", "WHERE s.item = 1 GROUP BY s.day, s.buyer)"),
        INSIDE.replace("SELECT DISTINCT s.day, s.buyer", "SELECT DISTINCT s.day, ROW_NUMBER() OVER (ORDER BY s.id) AS buyer"),
        INSIDE.split(" WHERE NOT EXISTS")[0],  # no filter: another rule
        INSIDE + " LIMIT 3",
        INSIDE.replace("AS d WHERE", "AS d JOIN items AS i ON i.item = d.day WHERE"),
        INSIDE.replace("AS d WHERE", "AS d WHERE RAND() < 0.5 AND"),
        INSIDE.replace("SELECT d.buyer, d.day AS at FROM", "SELECT DISTINCT d.buyer, d.day AS at FROM"),
    ],
)
def test_other_selects_keep_their_distinct(sql):
    assert lifted(sql) is None


SPELLINGS = [
    JOINED,
    f"SELECT a.buyer FROM sales AS a JOIN items AS i ON a.item = i.item LEFT JOIN (SELECT t.buyer, MAX(t.day) AS top FROM sales AS t GROUP BY t.buyer) AS b ON a.buyer = b.buyer AND a.day < b.top WHERE b.top IS NULL",
    f"SELECT a.buyer FROM (SELECT DISTINCT buyer FROM sales WHERE item = 1) AS a LEFT JOIN {RIGHT} AS b ON a.buyer = b.buyer WHERE b.buyer IS NULL",
    INSIDE,
]


@pytest.mark.parametrize("seed", range(3))
def test_both_readings_return_the_rows_duckdb_returns(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    for sql in SPELLINGS:
        read = unmatched(sql) or lifted(sql)
        assert read is not None, sql
        for _ in range(6):
            con = duckdb.connect()
            con.execute("CREATE TABLE sales (id INT, buyer INT, item INT, day INT)")
            con.execute("CREATE TABLE items (item INT, name VARCHAR)")
            for row in range(rng.randint(0, 10)):
                con.execute("INSERT INTO sales VALUES (?, ?, ?, ?)", [row, rng.choice([None, 1, 2, 3]), rng.choice([None, 1, 2]), rng.choice([None, 1, 2, 3])])
            for item in (1, 1, 2):
                if rng.random() < 0.8:
                    con.execute("INSERT INTO items VALUES (?, ?)", [item, "x"])
            assert sorted(con.execute(sql).fetchall(), key=repr) == sorted(con.execute(read).fetchall(), key=repr), sql


z3 = pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic  # noqa: E402

SCHEMA = {"sales": ["id", "buyer", "item", "day"], "items": ["item", "name"]}


def prove(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, dialect="mysql", compare_names=False)


def test_an_anti_join_between_two_distinct_lists_is_the_filtered_distinct():
    anti = f"SELECT a.buyer FROM (SELECT DISTINCT s.buyer FROM sales AS s WHERE s.item = 1) AS a LEFT JOIN {RIGHT} AS b ON a.buyer = b.buyer WHERE b.buyer IS NULL"
    plain = "SELECT DISTINCT s.buyer FROM sales AS s WHERE s.item = 1 AND NOT EXISTS (SELECT 1 FROM sales AS t WHERE t.item = 2 AND t.buyer = s.buyer)"
    assert prove(anti, plain).proven
    assert not prove(anti, plain.replace("SELECT DISTINCT", "SELECT")).proven
    assert not prove(anti, plain.replace("t.item = 2", "t.item = 3")).proven
    assert not prove(anti.replace("b.buyer IS NULL", "b.buyer IS NOT NULL"), plain).proven
    # NOT IN is another question when a buyer can be NULL
    assert not prove(anti, "SELECT DISTINCT s.buyer FROM sales AS s WHERE s.item = 1 AND s.buyer NOT IN (SELECT t.buyer FROM sales AS t WHERE t.item = 2)").proven
