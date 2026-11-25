"""``x IN (..)`` inside the subquery of another ``IN``, written as a join in that subquery (``nested_membership_joins``).

Each reading is run as written and as read on DuckDB and on SQLite over made-up rows with NULLs in the listed and
the tested columns and a repeated firm, so the lists differ in how often a value is listed and agree on which. The
key columns carry one name in every table, as real schemas name them: that is what the bare names of the nested
spelling rely on, and what the reading has to keep apart.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.nested_membership_joins import read_nested_membership_joins
from kumo_sql_prover.smt_equivalence import TableConstraints

TABLES = {
    "sellers": ("seller_id INT, name VARCHAR", [(1, "ann"), (2, "bo"), (3, "cy"), (4, "di")]),
    "firms": ("firm_id INT, colour VARCHAR", [(10, "red"), (10, "red"), (11, "blue"), (12, None), (13, "red")]),
    "sales": ("sale_id INT, seller_id INT, firm_id INT", [(100, 1, 10), (101, 1, 11), (102, 2, 11), (103, 3, 13), (104, None, 10), (105, 4, None), (106, 2, 12)]),
}
SCHEMA = {name: [part.split()[0] for part in columns.split(", ")] for name, (columns, _) in TABLES.items()}
TYPES = {name: dict(part.split() for part in columns.split(", ")) for name, (columns, _) in TABLES.items()}
CONSTRAINTS = {
    "sellers": TableConstraints(not_null=frozenset({"seller_id"}), keys=(("seller_id",),)),
    "firms": TableConstraints(),
    "sales": TableConstraints(not_null=frozenset({"sale_id"}), keys=(("sale_id",),)),
}
RED = "(SELECT firm_id FROM firms WHERE colour = 'red')"
RED_JOIN = "(SELECT sales.seller_id FROM sales, firms AS kqm1 WHERE sales.firm_id = kqm1.firm_id AND kqm1.colour = 'red')"

# (as written, the subquery as read)
READ = [
    (f"SELECT name FROM sellers AS s WHERE s.seller_id IN (SELECT seller_id FROM sales WHERE firm_id IN {RED})", RED_JOIN),
    # the red list holds a NULL seller, so NOT IN keeps nobody; the blue list does not
    (f"SELECT name FROM sellers AS s WHERE NOT s.seller_id IN (SELECT seller_id FROM sales WHERE firm_id IN {RED})", RED_JOIN),
    (
        "SELECT name FROM sellers WHERE seller_id NOT IN (SELECT seller_id FROM sales WHERE firm_id IN (SELECT firm_id FROM firms WHERE colour = 'blue'))",
        "(SELECT sales.seller_id FROM sales, firms AS kqm1 WHERE sales.firm_id = kqm1.firm_id AND kqm1.colour = 'blue')",
    ),
    (
        "SELECT name FROM sellers WHERE seller_id IN (SELECT DISTINCT seller_id FROM sales WHERE firm_id IN (SELECT DISTINCT firm_id FROM firms WHERE colour <> 'red'))",
        "(SELECT DISTINCT sales.seller_id FROM sales, firms AS kqm1 WHERE sales.firm_id = kqm1.firm_id AND kqm1.colour <> 'red')",
    ),
    (
        "SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM sales WHERE (sale_id > 102 OR seller_id = 1) AND firm_id IN (SELECT firm_id FROM firms WHERE colour = 'red' OR colour IS NULL))",
        "(SELECT sales.seller_id FROM sales, firms AS kqm1 WHERE (sales.sale_id > 102 OR sales.seller_id = 1) AND sales.firm_id = kqm1.firm_id AND (kqm1.colour = 'red' OR kqm1.colour IS NULL))",
    ),
    (
        "SELECT name FROM sellers AS s WHERE s.seller_id IN (SELECT x.seller_id FROM sales AS x WHERE x.firm_id IN (SELECT f.firm_id FROM firms AS f WHERE f.colour = 'red') AND x.sale_id < 104)",
        "(SELECT x.seller_id FROM sales AS x, firms AS kqm1 WHERE x.firm_id = kqm1.firm_id AND kqm1.colour = 'red' AND x.sale_id < 104)",
    ),
    # the table of the subquery again inside it
    (
        "SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM sales WHERE firm_id IN (SELECT firm_id FROM sales WHERE seller_id = 1))",
        "(SELECT sales.seller_id FROM sales, sales AS kqm1 WHERE sales.firm_id = kqm1.firm_id AND kqm1.seller_id = 1)",
    ),
    # a test in the select list is NULL for a seller the list does not hold when the list holds a NULL
    (f"SELECT name, seller_id IN (SELECT seller_id FROM sales WHERE firm_id IN {RED}) AS sells_red FROM sellers", RED_JOIN + " AS sells_red"),
    ("SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM sales WHERE firm_id IN (SELECT firm_id FROM firms))", "(SELECT sales.seller_id FROM sales, firms AS kqm1 WHERE sales.firm_id = kqm1.firm_id)"),
    # an outer row named under its own alias stays as written
    (
        f"SELECT name FROM sellers AS s WHERE s.seller_id IN (SELECT seller_id FROM sales WHERE firm_id IN {RED} AND sales.sale_id > s.seller_id + 99)",
        RED_JOIN[:-1] + " AND sales.sale_id > s.seller_id + 99)",
    ),
    # three levels: the innermost pair is read
    (
        "SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM sales WHERE firm_id IN (SELECT firm_id FROM firms WHERE firm_id IN (SELECT firm_id FROM sales WHERE seller_id = 2)))",
        "(SELECT firms.firm_id FROM firms, sales AS kqm1 WHERE firms.firm_id = kqm1.firm_id AND kqm1.seller_id = 2)",
    ),
]

LEFT_ALONE = [
    # the inner test is not a plain positive part of the WHERE
    f"SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM sales WHERE firm_id NOT IN {RED})",
    f"SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM sales WHERE sale_id > 103 OR firm_id IN {RED})",
    f"SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM sales WHERE (firm_id IN {RED}) IS NOT TRUE)",
    "SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM sales WHERE firm_id IN (10, 13))",
    # the subquery is more than one table filtered
    f"SELECT name FROM sellers WHERE seller_id IN (SELECT x.seller_id FROM sales AS x JOIN sellers AS y ON y.seller_id = x.seller_id WHERE x.firm_id IN {RED})",
    f"SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM sales WHERE firm_id IN {RED} GROUP BY seller_id)",
    f"SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM sales WHERE firm_id IN {RED} LIMIT 2)",
    f"SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id + 0 FROM sales WHERE firm_id IN {RED})",
    f"SELECT name FROM sellers WHERE seller_id IN (SELECT * FROM (SELECT seller_id, firm_id FROM sales) AS q WHERE firm_id IN {RED})",
    f"SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM sales WHERE firm_id IN {RED} AND sale_id IN (SELECT sale_id FROM sales))",
    # the inner select is more than one table filtered, or reads an outer row
    "SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM sales WHERE firm_id IN (SELECT MAX(firm_id) FROM firms))",
    "SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM sales WHERE firm_id IN (SELECT firm_id FROM firms ORDER BY firm_id LIMIT 1))",
    "SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM sales AS x WHERE firm_id IN (SELECT firm_id FROM firms WHERE firm_id = x.firm_id))",
    "SELECT name FROM sellers AS s WHERE seller_id IN (SELECT seller_id FROM sales WHERE firm_id IN (SELECT firm_id FROM firms WHERE s.seller_id > 2))",
    "SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM sales WHERE firm_id IN (SELECT firm_id FROM firms WHERE seller_id > 2))",
    # a name the declared columns do not place
    "SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM sales WHERE firm_id IN (SELECT firm_id FROM elsewhere))",
    f"SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM sales WHERE name = 'ann' AND firm_id IN {RED})",
    # not the subquery of an IN: how often a row is listed shows
    f"SELECT seller_id FROM sales WHERE firm_id IN {RED}",
    f"SELECT name FROM sellers AS s WHERE EXISTS (SELECT 1 FROM sales WHERE seller_id = s.seller_id AND firm_id IN {RED})",
    f"SELECT name FROM sellers AS s WHERE s.seller_id = (SELECT MAX(seller_id) FROM sales WHERE firm_id IN {RED})",
]


def _rows(database, sql: str, dialect: str) -> list:
    return sorted(database.execute(sqlglot.transpile(sql, read="mysql", write=dialect)[0]).fetchall(), key=repr)


def _database(dialect: str):
    database = pytest.importorskip("duckdb").connect() if dialect == "duckdb" else sqlite3.connect(":memory:")
    for name, (columns, rows) in TABLES.items():
        database.execute(f"CREATE TABLE {name} ({columns})")
        database.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(rows[0]))})", rows)
    return database


def _read(sql: str) -> tuple:
    return read_nested_membership_joins(sqlglot.parse_one(sql, read="mysql"), SCHEMA)


@pytest.mark.parametrize("written, subquery", READ)
def test_a_reading_returns_the_rows_of_the_query_as_written(written, subquery):
    tree, wrote = _read(written)
    read = tree.sql(dialect="mysql")
    assert wrote and subquery in read, read
    for dialect in ("duckdb", "sqlite"):
        database = _database(dialect)
        assert _rows(database, written, dialect) == _rows(database, read, dialect), dialect


def test_the_made_up_rows_tell_the_cases_apart():
    database = _database("sqlite")
    assert _rows(database, READ[0][0], "sqlite") == [("ann",), ("cy",)]
    assert _rows(database, READ[1][0], "sqlite") == []  # a NULL seller sold to a red firm
    assert _rows(database, READ[2][0], "sqlite") == [("cy",), ("di",)]
    assert (None,) in {row[1:] for row in _rows(database, READ[7][0], "sqlite")}


@pytest.mark.parametrize("written", LEFT_ALONE)
def test_a_query_outside_the_reading_stays_as_written(written):
    given = sqlglot.parse_one(written, read="mysql")
    tree, wrote = read_nested_membership_joins(given, SCHEMA)
    assert not wrote and tree is given


def test_nothing_is_read_without_declared_columns():
    tree = sqlglot.parse_one(READ[0][0], read="mysql")
    assert not read_nested_membership_joins(tree, None)[1]
    assert not read_nested_membership_joins(tree, {"sellers": SCHEMA["sellers"], "sales": SCHEMA["sales"]})[1]


def test_the_new_alias_is_not_one_the_statement_uses():
    written = "SELECT name FROM sellers AS kqm1 WHERE kqm1.seller_id IN (SELECT seller_id FROM sales WHERE firm_id IN (SELECT firm_id FROM firms))"
    tree, wrote = _read(written)
    assert wrote and "firms AS kqm1x" in tree.sql(dialect="mysql")
    for dialect in ("duckdb", "sqlite"):
        database = _database(dialect)
        assert _rows(database, written, dialect) == _rows(database, tree.sql(dialect="mysql"), dialect)


def test_the_tree_it_is_given_is_not_changed():
    tree = sqlglot.parse_one(READ[0][0], read="mysql")
    before = tree.sql()
    read, wrote = read_nested_membership_joins(tree, SCHEMA)
    assert wrote and read is not tree and tree.sql() == before


@pytest.mark.parametrize("dialect", ["duckdb", "sqlite"])
def test_outside_an_in_the_join_by_hand_lists_a_row_more_often(dialect):
    written = LEFT_ALONE[17]
    by_hand = "SELECT sales.seller_id FROM sales, firms AS f WHERE sales.firm_id = f.firm_id AND f.colour = 'red'"
    database = _database(dialect)
    assert len(_rows(database, written, dialect)) == 3 and len(_rows(database, by_hand, dialect)) == 5  # firm 10 is listed twice


def _proved(left: str, right: str) -> bool:
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect="mysql").proven


JOINED = "SELECT s.name FROM sellers AS s WHERE s.seller_id {} (SELECT x.seller_id FROM sales AS x JOIN firms AS f ON x.firm_id = f.firm_id WHERE f.colour = 'red')"
NESTED = "SELECT s.name FROM sellers AS s WHERE s.seller_id {} (SELECT seller_id FROM sales WHERE firm_id IN (SELECT firm_id FROM firms WHERE colour = '{}'))"


@pytest.mark.parametrize("test", ["IN", "NOT IN"])
def test_a_pair_is_proved_and_only_with_the_rule(test):
    assert _proved(JOINED.format(test), NESTED.format(test, "red"))
    with shape_fallback.without("nested_membership_joins"):
        assert not _proved(JOINED.format(test), NESTED.format(test, "red"))


def test_pairs_that_differ_stay_unproved():
    assert not _proved(JOINED.format("IN"), NESTED.format("IN", "blue"))
    assert not _proved(JOINED.format("IN"), NESTED.format("NOT IN", "red"))
