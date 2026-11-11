"""Inline rows written as UNNEST of STRUCT literals are read as a derived table (``kumo_sql_prover.inline_rows``)."""

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.inline_rows import inline_struct_rows

SCHEMA = {"t": ["k", "v"]}
TYPES = {"t": {"k": "INT64", "v": "INT64"}}


def rewritten(sql, dialect="bigquery"):
    return inline_struct_rows(sqlglot.parse_one(sql, read=dialect), dialect).sql(dialect=dialect)


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False).proven


def test_the_rows_become_a_derived_table():
    assert rewritten("SELECT * FROM UNNEST([STRUCT(1 AS a, 2 AS b), STRUCT(3 AS a, NULL AS b)]) AS r") == (
        "SELECT * FROM (SELECT 1 AS a, 2 AS b UNION ALL SELECT 3 AS a, NULL AS b) AS r"
    )
    # any join position, with or without an alias, casts and negative numbers kept as written
    assert rewritten(
        "SELECT r.a, t.k FROM t LEFT JOIN UNNEST([STRUCT(CAST(-1 AS INT64) AS a), STRUCT(0 AS a)]) AS r ON t.k = r.a "
        "FULL JOIN UNNEST([STRUCT('x' AS s)]) AS q ON TRUE, UNNEST([STRUCT(TRUE AS f)])"
    ) == (
        "SELECT r.a, t.k FROM t LEFT JOIN (SELECT CAST(-1 AS INT64) AS a UNION ALL SELECT 0 AS a) AS r ON t.k = r.a "
        "FULL JOIN (SELECT 'x' AS s) AS q ON TRUE CROSS JOIN (SELECT TRUE AS f)"
    )


@pytest.mark.parametrize("sql", [
    "SELECT * FROM UNNEST([STRUCT(1 AS a), STRUCT(2 AS a)]) AS r WITH OFFSET AS o",  # the position is a column too
    "SELECT * FROM t, UNNEST([STRUCT(t.k AS a)]) AS r",  # a field that reads the row beside it
    "SELECT * FROM UNNEST([STRUCT(ABS(1) AS a)]) AS r",  # a function
    "SELECT * FROM UNNEST([STRUCT((SELECT 1) AS a)]) AS r",
    "SELECT * FROM UNNEST([STRUCT(1 AS a), STRUCT(2 AS b)]) AS r",  # field names differ
    "SELECT * FROM UNNEST([STRUCT(1 AS a, 2 AS b), STRUCT(2 AS b, 1 AS a)]) AS r",  # field order differs
    "SELECT * FROM UNNEST([STRUCT(1 AS a, 2 AS a)]) AS r",  # a name twice
    "SELECT * FROM UNNEST([STRUCT(1, 2)]) AS r",  # unnamed fields
    "SELECT * FROM UNNEST([STRUCT(1 AS a), STRUCT(2.5 AS a)]) AS r",  # two kinds in one field
    "SELECT * FROM UNNEST([STRUCT(1 AS a), STRUCT('x' AS a)]) AS r",
    "SELECT * FROM UNNEST([STRUCT(1 AS a), STRUCT(CAST(2 AS NUMERIC) AS a)]) AS r",
    "SELECT r FROM UNNEST([STRUCT(1 AS a)]) AS r",  # the alias read as the whole struct
    "SELECT COUNT(r) FROM UNNEST([STRUCT(1 AS a)]) AS r",
    "SELECT r.a FROM UNNEST([STRUCT(1 AS a)]) AS r WHERE r IS NOT NULL",
    "SELECT r.a.x FROM UNNEST([STRUCT(1 AS a)]) AS r",  # a longer path
    "SELECT r.r FROM UNNEST([STRUCT(1 AS r)]) AS r",  # a field named like the alias
    "SELECT * FROM UNNEST([1, 2]) AS r",  # not structs
    "SELECT * FROM UNNEST(ARRAY<STRUCT<a INT64>>[]) AS r",  # no rows to write
    "SELECT * FROM UNNEST(arr) AS r",
    "SELECT * FROM t WHERE STRUCT(t.k AS a) IN UNNEST([STRUCT(1 AS a)])",  # not a source
])
def test_declined(sql):
    assert rewritten(sql) == sqlglot.parse_one(sql, read="bigquery").sql(dialect="bigquery")


def test_other_dialects_are_left_alone():
    sql = "SELECT * FROM UNNEST([STRUCT(1 AS a)]) AS r"
    assert inline_struct_rows(sqlglot.parse_one(sql, read="bigquery"), "duckdb").sql(dialect="bigquery") == sql


ROWS = "UNNEST([STRUCT(1 AS x, 10 AS y), STRUCT(2 AS x, NULL AS y)]) AS r"
TABLE = "(SELECT 1 AS x, 10 AS y UNION ALL SELECT 2 AS x, NULL AS y) AS r"


@pytest.mark.parametrize("left, right", [
    (f"SELECT t.k, r.y FROM t JOIN {ROWS} ON t.k = r.x", f"SELECT t.k, r.y FROM t JOIN {TABLE} ON t.k = r.x"),
    (f"SELECT t.k, r.y FROM t LEFT JOIN {ROWS} ON t.k = r.x", f"SELECT t.k, r.y FROM t LEFT JOIN {TABLE} ON t.k = r.x"),
    (f"SELECT t.k, y FROM t, {ROWS} WHERE t.k = x", f"SELECT t.k, r.y FROM t JOIN {TABLE} ON t.k = r.x"),
    # a field that is NULL in every row never passes a comparison
    ("SELECT t.k FROM t, UNNEST([STRUCT(0 AS a, NULL AS b)]) AS r WHERE t.v < r.b AND t.k = r.a", "SELECT t.k FROM t WHERE FALSE"),
])
def test_proved(left, right):
    assert proved(left, right)


@pytest.mark.parametrize("left, right", [
    # the same value on two rows doubles every match
    ("SELECT t.k FROM t JOIN UNNEST([STRUCT(1 AS x), STRUCT(1 AS x)]) AS r ON t.k = r.x", "SELECT t.k FROM t WHERE t.k = 1"),
    # the NULL row matches nothing, and a LEFT JOIN keeps the unmatched rows of t
    (f"SELECT t.k, r.y FROM t LEFT JOIN {ROWS} ON t.k = r.x", f"SELECT t.k, r.y FROM t JOIN {TABLE} ON t.k = r.x"),
    (f"SELECT t.k, r.y FROM t JOIN {ROWS} ON t.k = r.y", f"SELECT t.k, r.y FROM t JOIN {TABLE} ON t.k = r.x"),
    # the rows side of a RIGHT JOIN is kept when t is empty
    (f"SELECT r.x FROM t RIGHT JOIN {ROWS} ON t.k = r.x", f"SELECT r.x FROM t JOIN {TABLE} ON t.k = r.x"),
    # a NULL in the rows is a row
    ("SELECT COUNT(*) FROM t, UNNEST([STRUCT(1 AS x), STRUCT(NULL AS x)]) AS r", "SELECT COUNT(*) FROM t"),
    # the first row only
    (f"SELECT r.x FROM {ROWS} LIMIT 1", "SELECT 1"),
])
def test_not_proved(left, right):
    assert not proved(left, right)
