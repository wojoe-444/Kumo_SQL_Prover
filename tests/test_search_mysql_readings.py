"""The counterexample search runs MySQL answers on DuckDB. These are the spellings MySQL accepts and DuckDB
rejects that ``counterexample.to_duckdb`` reads when it is told the schema's columns: a ``GROUP BY`` on a
select-list alias, an aggregate beside ungrouped columns without ``GROUP BY``, a grouped column of another
source, a column under its own item's alias, a ``HAVING`` that filters rows, and conditions used as numbers.

Every expected row below is what a MySQL 8.0 server returns for the query as written (``ONLY_FULL_GROUP_BY``
off) on the same two made-up tables; the test checks that DuckDB returns it for the translated text.
"""

from __future__ import annotations

import pytest
import sqlglot

duckdb = pytest.importorskip("duckdb")

from kumo_sql_prover import counterexample as cx  # noqa: E402

COLUMNS = {"shop", "town", "sales", "person", "pay", "bonus"}
TABLES = {"shops": {"shop", "town", "sales"}, "staff": {"person", "shop", "pay", "bonus"}}


def database():
    db = duckdb.connect(":memory:")
    db.execute("CREATE TABLE shops (shop BIGINT, town VARCHAR, sales BIGINT)")
    db.execute("INSERT INTO shops VALUES (1, 'york', 30), (2, 'york', 45), (3, 'bath', 45), (4, NULL, 7)")
    db.execute("CREATE TABLE staff (person BIGINT, shop BIGINT, pay BIGINT, bonus BIGINT)")
    db.execute("INSERT INTO staff VALUES (1, 1, 20, 150), (2, 1, 8, NULL), (3, 2, 20, 40), (4, 3, 12, 0), (5, NULL, 30, 99)")
    return db


def rows(sql: str) -> list:
    """The rows DuckDB returns for the search's text of a MySQL query, numbers as floats, in a fixed order."""

    out = database().execute(cx.to_duckdb(sql, "mysql", COLUMNS, TABLES)).fetchall()
    return sorted(([value if value is None or isinstance(value, str) else round(float(value), 4) for value in row] for row in out), key=repr)


# (the MySQL query, the rows MySQL returns for it, what the search's DuckDB text must hold)
READINGS = [
    # GROUP BY on a select-list alias: grouped by the item's expression
    ("SELECT town AS place, COUNT(*) AS n FROM shops GROUP BY place", [["bath", 1.0], ["york", 2.0], [None, 1.0]], "GROUP BY town"),
    ("SELECT sales DIV 10 AS band, COUNT(*) AS n, MAX(shop) FROM shops GROUP BY band", [[0.0, 1.0, 4.0], [3.0, 1.0, 1.0], [4.0, 2.0, 3.0]], "GROUP BY sales // 10"),
    ("SELECT s.town AS place, COUNT(*) FROM shops AS s GROUP BY place", [["bath", 1.0], ["york", 2.0], [None, 1.0]], "GROUP BY s.town"),
    ("SELECT town AS place, SUM(sales) AS total FROM shops GROUP BY place HAVING total > 10 ORDER BY place", [["bath", 45.0], ["york", 75.0]], "GROUP BY town HAVING total > 10"),
    # the alias is also a column of the one source: MySQL groups by the column, and so does the text
    ("SELECT town AS shop, COUNT(*) FROM shops GROUP BY shop", [["bath", 1.0], ["york", 1.0], ["york", 1.0], [None, 1.0]], "GROUP BY shop"),
    # an aggregate and no GROUP BY: one group, the other columns arbitrary within it
    ("SELECT t.total / COUNT(s.shop) AS mean FROM (SELECT SUM(sales) AS total FROM shops) AS t, shops AS s", [[31.75]], "ANY_VALUE(t.total)"),
    ("SELECT t.total, COUNT(s.shop) FROM (SELECT SUM(sales) AS total FROM shops) AS t, shops AS s WHERE s.shop > 99", [[None, 0.0]], "ANY_VALUE(t.total)"),
    # grouped by the column of one source, read from the other
    ("SELECT a.shop, COUNT(*) FROM shops AS a JOIN staff AS b ON a.shop = b.shop GROUP BY b.shop", [[1.0, 2.0], [2.0, 1.0], [3.0, 1.0]], "ANY_VALUE(a.shop)"),
    # a column under its own item's alias is the column
    ("SELECT shop, CASE WHEN COUNT(*) > 1 THEN 'many' ELSE town END AS town FROM shops GROUP BY shop", [[1.0, "york"], [2.0, "york"], [3.0, "bath"], [4.0, None]], "ELSE ANY_VALUE(town) END"),
    ("SELECT town, CASE WHEN COUNT(*) > 1 THEN 'many' ELSE town END AS town2 FROM shops GROUP BY town", [["bath", "bath"], ["york", "many"], [None, None]], "ELSE town END"),
    # HAVING without GROUP BY or an aggregate filters rows
    ("SELECT person, bonus FROM staff HAVING bonus < 100 OR bonus IS NULL", [[2.0, None], [3.0, 40.0], [4.0, 0.0], [5.0, 99.0]], "WHERE bonus < 100 OR bonus IS NULL"),
    ("SELECT person, IFNULL(bonus, 0) AS extra FROM staff WHERE pay > 10 HAVING extra < 100", [[3.0, 40.0], [4.0, 0.0], [5.0, 99.0]], "WHERE (pay > 10) AND ((COALESCE(bonus, 0)) < 100)"),
    ("SELECT DISTINCT shop FROM staff HAVING shop > 1", [[2.0], [3.0]], "WHERE shop > 1"),
    # conditions are the numbers 1 and 0
    ("SELECT AVG(pay > 10), AVG(bonus IS NULL) FROM staff", [[0.8, 0.2]], "AVG(CAST(pay > 10 AS BIGINT))"),
    ("SELECT SUM((pay > 10) * bonus), SUM(pay > 10) FROM staff", [[289.0, 4.0]], "CAST((pay > 10) AS BIGINT) * bonus"),
    ("SELECT person FROM staff WHERE (pay > 10) + (bonus > 50) = 2", [[1.0], [5.0]], "CAST((pay > 10) AS BIGINT) + CAST((bonus > 50) AS BIGINT)"),
    ("SELECT person, 1 - (shop IS NULL) FROM staff WHERE person > 3", [[4.0, 1.0], [5.0, 0.0]], "1 - CAST((shop IS NULL) AS BIGINT)"),
]


@pytest.mark.parametrize("sql,expected,holds", READINGS, ids=[str(n) for n in range(len(READINGS))])
def test_duckdb_returns_what_mysql_returns(sql, expected, holds):
    text = cx.to_duckdb(sql, "mysql", COLUMNS, TABLES)
    assert holds in text, text
    assert rows(sql) == expected


def test_a_group_by_name_is_the_alias_only_when_mysql_reads_it_so():
    """MySQL looks the name up in the FROM clause first; the alias is meant when no source has such a column, or
    when several have (the lookup is ambiguous and MySQL falls back to the select list)."""

    def text(sql: str) -> str:
        return cx.to_duckdb(sql, "mysql", COLUMNS, TABLES)

    # two sources have a column named shop: the alias
    assert "GROUP BY a.person" in text("SELECT a.person AS shop, COUNT(*) FROM staff AS a JOIN shops AS b ON a.shop = b.shop GROUP BY shop")
    # one source has it: that column is grouped, the item is an ungrouped column
    assert text("SELECT person AS shop, COUNT(*) FROM staff GROUP BY shop") == "SELECT ANY_VALUE(person) AS shop, COUNT(*) FROM staff GROUP BY shop"
    # a column of the schema, but of no source here: the alias
    assert "GROUP BY person" in text("SELECT person AS town, COUNT(*) FROM staff GROUP BY town")
    # a derived table could have a column of that name: left as written
    derived = text("SELECT t.person AS shop, COUNT(*) FROM (SELECT * FROM staff) AS t GROUP BY shop")
    assert derived.endswith("GROUP BY shop") and "ANY_VALUE(t.person)" in derived
    # a name defined twice in the query (here by a derived table too) could be a column of a source
    twice = text("SELECT t.pay AS place, COUNT(*) FROM (SELECT pay, town AS place FROM staff JOIN shops USING (shop)) AS t GROUP BY place")
    assert twice.endswith("GROUP BY place")
    # an item with an aggregate is never a grouping key
    assert text("SELECT COUNT(*) AS n, town FROM shops GROUP BY n").endswith("GROUP BY n")


def test_a_caller_that_does_not_name_the_columns_gets_the_reading_it_had():
    def legacy(sql: str) -> str:
        return cx.relax_grouping(sqlglot.parse_one(sql, read="mysql")).sql(dialect="duckdb")

    assert legacy("SELECT town AS place, COUNT(*) AS n FROM shops GROUP BY place") == "SELECT ANY_VALUE(town) AS place, COUNT(*) AS n FROM shops GROUP BY place"
    assert legacy("SELECT s.town AS place, COUNT(*) FROM shops AS s GROUP BY place") == "SELECT s.town AS place, COUNT(*) FROM shops AS s GROUP BY s.town"
    assert "ANY_VALUE" not in legacy("SELECT t.total + COUNT(s.shop) FROM t, shops AS s")
    assert "ANY_VALUE" not in legacy("SELECT a.shop, COUNT(*) FROM shops AS a JOIN staff AS b ON a.shop = b.shop GROUP BY b.shop")
    assert cx.to_duckdb("SELECT person, bonus FROM staff HAVING bonus < 100", "mysql") == "SELECT person, bonus FROM staff HAVING bonus < 100"


# a HAVING that stays where it is: the query is handed to DuckDB as written
HAVING_KEPT = [
    "SELECT person, ROW_NUMBER() OVER (ORDER BY pay) AS place FROM staff HAVING place = 1",  # HAVING runs after the window, WHERE before
    "SELECT person, IFNULL(bonus, 0) AS bonus FROM staff HAVING bonus < 100",  # a column of the schema and the alias of something else
    "SELECT shop, COUNT(*) AS n FROM staff GROUP BY shop HAVING n > 1",  # a grouped query's HAVING
    "SELECT COUNT(*) AS n FROM staff HAVING n > 1",  # an aggregate: one group
]


@pytest.mark.parametrize("sql", HAVING_KEPT)
def test_a_having_that_is_not_a_plain_row_filter_stays(sql):
    assert " HAVING " in cx.to_duckdb(sql, "mysql", COLUMNS, TABLES)


def test_a_sum_of_a_condition_and_a_condition_in_a_filter_are_not_cast():
    text = cx.to_duckdb("SELECT SUM(pay > 10), COUNT(*) FROM staff WHERE pay > 10 AND (bonus > 5 OR shop IS NULL)", "mysql", COLUMNS, TABLES)
    assert "CAST" not in text
    # and other dialects keep their conditions as they are
    assert "CAST" not in cx.to_duckdb("SELECT AVG(pay > 10) FROM staff", "postgres", COLUMNS, TABLES)


def test_a_flag_column_stored_as_a_number_runs_arithmetic():
    """MySQL's BOOL is a small whole number. A harness that says so (``Column.as_number``) gets a whole-number column
    holding 1 and 0, which DuckDB reads as a condition where one is asked for; without it the column is a BOOLEAN."""

    def spec(as_number: bool) -> cx.Spec:
        return cx.Spec({"SEATS": cx.Table("SEATS", [cx.Column("SEAT", "INT", not_null=True), cx.Column("FREE", "BOOL", as_number=as_number)], primary_key=("SEAT",))})

    sums = "SELECT a.SEAT FROM SEATS AS a, SEATS AS b WHERE a.SEAT + 1 = b.SEAT AND a.FREE + b.FREE = 2"
    plain = "SELECT a.SEAT FROM SEATS AS a, SEATS AS b WHERE a.SEAT + 1 = b.SEAT AND a.FREE AND b.FREE = 1"
    assert cx.Searcher(spec(True), sums, plain).runs()
    assert not cx.Searcher(spec(False), sums, plain).runs()
    assert cx._duck_type(cx.Column("FREE", "BOOL", as_number=True)) == "TINYINT" and cx._duck_type(cx.Column("FREE", "BOOL")) == "BOOLEAN"
    assert cx._domain(cx.Column("FREE", "BOOL", as_number=True), cx.Constants()) == [1, 0]
    assert cx.Searcher(spec(True), sums, plain).search(60) is None
    assert cx.Searcher(spec(True), sums, "SELECT a.SEAT FROM SEATS AS a WHERE a.FREE").search(60) is not None
