"""Two rows compared with ``=`` or ``<>`` are their fields compared one by one (``row_equality``)."""

import itertools

import pytest
import sqlglot

from kumo_sql_prover.row_equality import expand_row_equality

BASE = "SELECT a.pid FROM place AS a JOIN place AS b ON (a.lat, a.lon) = (b.lat, b.lon) WHERE a.pid <> b.pid"


def rewritten(sql: str, dialect: str | None = "mysql") -> str | None:
    result = expand_row_equality(sqlglot.parse_one(sql, read="mysql"), dialect)
    return None if result is None else result.sql(dialect="mysql")


def test_rows_are_compared_field_by_field():
    assert rewritten(BASE) == "SELECT a.pid FROM place AS a JOIN place AS b ON (a.lat = b.lat AND a.lon = b.lon) WHERE a.pid <> b.pid"
    assert rewritten(BASE, "postgres") is not None
    assert rewritten(BASE.replace(") = (", ") <> (")) == "SELECT a.pid FROM place AS a JOIN place AS b ON (a.lat <> b.lat OR a.lon <> b.lon) WHERE a.pid <> b.pid"
    assert rewritten("SELECT 1 FROM t WHERE (a, b, c) = (1, 'x', -2) AND NOT (a, b) = (c, d)") == "SELECT 1 FROM t WHERE (a = 1 AND b = 'x' AND c = -2) AND NOT (a = c AND b = d)"
    # only the select's own comparisons: a nested select is read when the rule reaches it
    nested = "SELECT 1 FROM t WHERE EXISTS (SELECT 1 FROM u WHERE (u.a, u.b) = (t.a, t.b))"
    assert rewritten(nested) is None
    inner = sqlglot.parse_one(nested, read="mysql").find(sqlglot.exp.Exists).this
    assert expand_row_equality(inner, "mysql").sql(dialect="mysql") == "SELECT 1 FROM u WHERE (u.a = t.a AND u.b = t.b)"


@pytest.mark.parametrize(
    "sql, dialect",
    [
        (BASE, "duckdb"),  # rows are structs there: a NULL field equals a NULL field
        (BASE, "bigquery"),
        (BASE, None),
        (BASE.replace(") = (", ") < ("), "mysql"),  # lexicographic
        (BASE.replace("(a.lat, a.lon) = (b.lat, b.lon)", "(a.lat, a.lon) = (b.lat, b.lon, b.pid)"), "mysql"),
        (BASE.replace("(a.lat, a.lon) = (b.lat, b.lon)", "(a.lat, RAND()) = (b.lat, b.lon)"), "mysql"),  # written twice would be drawn twice
        (BASE.replace("(a.lat, a.lon) = (b.lat, b.lon)", "(a.lat, a.lon + 1) = (b.lat, b.lon)"), "mysql"),
        (BASE.replace("(a.lat, a.lon) = (b.lat, b.lon)", "(a.lat, a.lon) = (SELECT c.lat, c.lon FROM place AS c LIMIT 1)"), "mysql"),
        (BASE.replace("(a.lat, a.lon) = (b.lat, b.lon)", "a.lat = b.lat"), "mysql"),
    ],
)
def test_other_comparisons_are_left_alone(sql, dialect):
    assert rewritten(sql, dialect) is None


@pytest.mark.parametrize("operator", ["=", "<>"])
def test_three_valued_logic_agrees_on_sqlite(operator):
    """SQLite follows the standard for row values: every combination of NULL, equal and different fields."""

    import sqlite3

    con = sqlite3.connect(":memory:")
    values = [None, 1, 2]
    for a, b, c, d in itertools.product(values, repeat=4):
        row = con.execute(f"SELECT (?1, ?2) {operator} (?3, ?4)", (a, b, c, d)).fetchone()[0]
        joined = "(?1 = ?3 AND ?2 = ?4)" if operator == "=" else "(?1 <> ?3 OR ?2 <> ?4)"
        assert con.execute(f"SELECT {joined}", (a, b, c, d)).fetchone()[0] == row, (a, b, c, d)


def test_duckdb_is_why_the_dialect_is_checked():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    assert con.execute("SELECT (1, NULL) = (1, NULL)").fetchone()[0] is True
    assert con.execute("SELECT 1 = 1 AND NULL = NULL").fetchone()[0] is None


def test_the_prover_reads_a_row_comparison():
    pytest.importorskip("z3")
    from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic

    schema = {"place": ["pid", "lat", "lon"]}
    fields = BASE.replace("(a.lat, a.lon) = (b.lat, b.lon)", "a.lon = b.lon AND a.lat = b.lat")
    assert prove_equivalent_algebraic(BASE, fields, schema=schema, compare_names=False, dialect="mysql").proven
    assert not prove_equivalent_algebraic(BASE, fields.replace("a.lon = b.lon AND ", ""), schema=schema, compare_names=False, dialect="mysql").proven
    assert not prove_equivalent_algebraic(BASE, fields, schema=schema, compare_names=False, dialect="duckdb").proven
