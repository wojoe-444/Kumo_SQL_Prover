"""A column list on a ``WITH`` name is read by position, or the pair is declined (Kumo_SQL_Prover issue 67).

The solver layer looked a ``WITH`` name up by the output names of its query and did not look at the list. With

    WITH t (a, b) AS (SELECT b, a FROM x) SELECT a FROM t

``t.a`` is the first output, which is ``x.b``; read by the query's own names it was ``x.a``. So the query was proved
equal to ``SELECT a FROM x``, and against ``SELECT b FROM x`` (which it is) the solver returned a counterexample.
Each reading below is first shown on DuckDB and SQLite.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover import SmtStatus, prove_equivalent_smt
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.listed_cte_names import listed_body, renames_an_output

SCHEMA = {"x": ["a", "b", "c"]}
TYPES = {"x": {"a": "INT", "b": "INT", "c": "INT"}}
ROWS = [(1, 10, 100), (2, 20, 200), (3, None, 300), (None, 40, 400)]

SWAPPED = "WITH t (a, b) AS (SELECT b, a FROM x) SELECT a FROM t"
# (a query with a list, what it is, what it is not)
CASES = [
    (SWAPPED, "SELECT b FROM x", "SELECT a FROM x"),
    ("WITH t (a, b) AS (SELECT b, a FROM x) SELECT t.b FROM t WHERE t.a > 15", "SELECT a FROM x WHERE b > 15", "SELECT b FROM x WHERE a > 15"),
    ("WITH t (a, k) AS (SELECT b, a FROM x) SELECT a FROM t", "SELECT b FROM x", "SELECT a FROM x"),
    ("WITH t (a, b) AS (SELECT x.b, x.a FROM x) SELECT a FROM t", "SELECT b FROM x", "SELECT a FROM x"),
    ("WITH t (b, a) AS (SELECT a, c FROM x) SELECT a FROM t", "SELECT c FROM x", "SELECT a FROM x"),
    ("WITH t (a, b) AS (SELECT b, a FROM x UNION ALL SELECT c, c FROM x) SELECT a FROM t", "SELECT b FROM x UNION ALL SELECT c FROM x", "SELECT a FROM x UNION ALL SELECT c FROM x"),
    ("WITH t (a, b) AS (SELECT * FROM (SELECT b, a FROM x) AS s) SELECT a FROM t", "SELECT b FROM x", "SELECT a FROM x"),
    # a list of new names, and a list that repeats the names of its query
    ("WITH t (p, q) AS (SELECT b, a FROM x) SELECT p FROM t", "SELECT b FROM x", "SELECT a FROM x"),
    ("WITH t (b, a) AS (SELECT b, a FROM x) SELECT a FROM t", "SELECT a FROM x", "SELECT b FROM x"),
    ("WITH t (n) AS (SELECT a + b FROM x) SELECT n FROM t", "SELECT a + b FROM x", "SELECT a FROM x"),
]
READ = {CASES[7][0], CASES[8][0], CASES[9][0]}  # the others are declined


def _rows(db, sql: str) -> list:
    return sorted((tuple(row) for row in db.execute(sql).fetchall()), key=repr)


@pytest.mark.parametrize("listed, same, other", CASES)
def test_the_engines_read_the_list_by_position(listed, same, other):
    duckdb = pytest.importorskip("duckdb")
    for db in (duckdb.connect(":memory:"), sqlite3.connect(":memory:")):
        db.execute("CREATE TABLE x (a INT, b INT, c INT)")
        db.executemany("INSERT INTO x VALUES (?, ?, ?)", ROWS)
        assert _rows(db, listed) == _rows(db, same) != _rows(db, other)


@pytest.mark.parametrize("dialect", ["mysql", "duckdb", "postgres", "bigquery"])
@pytest.mark.parametrize("listed, same, other", CASES)
def test_a_list_is_read_by_position_or_the_pair_is_declined(listed, same, other, dialect):
    for prove in (prove_equivalent_algebraic, prove_equivalent_smt):
        wrong = prove(listed, other, schema=SCHEMA, types=TYPES, compare_names=False, dialect=dialect)
        assert wrong.status is not SmtStatus.PROVEN_EQUIVALENT  # the false proof
        right = prove(listed, same, schema=SCHEMA, types=TYPES, compare_names=False, dialect=dialect)
        assert right.status is not SmtStatus.NOT_EQUIVALENT and right.counterexample is None  # the wrong counterexample
        if listed in READ:
            assert right.proven
        else:
            assert "column list on a WITH name" in right.reason and "column list on a WITH name" in wrong.reason


def _entry(sql: str) -> sqlglot.exp.CTE:
    return sqlglot.parse_one(sql, read="duckdb").find(sqlglot.exp.CTE)


def test_the_query_of_a_listed_entry():
    assert listed_body(_entry(SWAPPED)) is None
    assert listed_body(_entry("WITH t (a, b) AS (SELECT * FROM x) SELECT a FROM t")) is None  # a star: the outputs are not known here
    assert listed_body(_entry("WITH t (a) AS (SELECT a, b FROM x) SELECT a FROM t")) is None  # not one name per output
    same = _entry("WITH t (b, a) AS (SELECT b, a FROM x ORDER BY a LIMIT 2) SELECT a FROM t")
    assert listed_body(same) is same.this  # the names the query gives: nothing to rename, whatever else the query has
    named = listed_body(_entry("WITH t (p, q) AS (SELECT b, a + 1 FROM x) SELECT p FROM t"))
    assert named.sql(dialect="duckdb") == "SELECT b AS p, a + 1 AS q FROM x"
    assert listed_body(_entry("WITH t (p, q) AS (SELECT b, a FROM x ORDER BY a) SELECT p FROM t")) is None  # ORDER BY names outputs


def test_whether_a_list_renames_an_output():
    parse = lambda sql: sqlglot.parse_one(sql, read="duckdb")  # noqa: E731
    assert renames_an_output(parse(SWAPPED))
    assert renames_an_output(parse("WITH t (p) AS (SELECT a FROM x) SELECT p FROM t"))
    assert not renames_an_output(parse("WITH t (a, b) AS (SELECT a, b FROM x) SELECT a FROM t"))
    assert not renames_an_output(parse("WITH t AS (SELECT b AS a FROM x) SELECT a FROM t"))


def test_a_list_on_a_derived_table_was_and_is_read_by_position():
    listed = "SELECT a FROM (SELECT b, a FROM x) AS t (a, b)"
    assert prove_equivalent_algebraic(listed, "SELECT b FROM x", schema=SCHEMA, types=TYPES, compare_names=False, dialect="duckdb").proven
    assert not prove_equivalent_algebraic(listed, "SELECT a FROM x", schema=SCHEMA, types=TYPES, compare_names=False, dialect="duckdb").proven
