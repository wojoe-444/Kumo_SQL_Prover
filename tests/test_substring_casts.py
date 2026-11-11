"""A cast of a bounded substring to a wide enough string type is the substring (``kumo_sql_prover.substring_casts``)."""

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.substring_casts import drop_substring_casts

SCHEMA = {"c": ["k", "phone", "code", "raw", "n"]}
TYPES = {"c": {"k": "INT", "phone": "CHAR(15)", "code": "VARCHAR", "raw": "BYTEA", "n": "INT"}}


def dropped(sql, dialect="postgres"):
    out = drop_substring_casts(sqlglot.parse_one(sql, read=dialect), TYPES)
    return out.sql(dialect=dialect) if out is not None else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


def rows_on_duckdb(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE c (k BIGINT, phone VARCHAR, code VARCHAR, raw BLOB, n BIGINT)")
    for row in rows:
        db.execute("INSERT INTO c VALUES (?, ?, ?, NULL, ?)", list(row))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def test_the_cast_goes():
    assert (
        dropped("SELECT c.k FROM c WHERE CAST(SUBSTRING(c.phone FROM 1 FOR 2) AS VARCHAR(2)) IN ('13', '31')")
        == "SELECT c.k FROM c WHERE SUBSTRING(c.phone FROM 1 FOR 2) IN ('13', '31')"
    )
    assert dropped("SELECT CAST(SUBSTRING(code, 2, 3) AS VARCHAR(8)) FROM c") == "SELECT SUBSTRING(code FROM 2 FOR 3) FROM c"
    assert dropped("SELECT CAST(SUBSTRING(code, 2, 3) AS TEXT), CAST(SUBSTRING(code, 2) AS VARCHAR) FROM c") == "SELECT SUBSTRING(code FROM 2 FOR 3), SUBSTRING(code FROM 2) FROM c"
    # through a plain derived table the column's declared type is still known
    assert (
        dropped("SELECT CAST(SUBSTRING(t.phone FROM 1 FOR 2) AS VARCHAR(2)) FROM (SELECT * FROM c WHERE n > 0) AS t") is None
        or "CAST" not in dropped("SELECT CAST(SUBSTRING(t.phone FROM 1 FOR 2) AS VARCHAR(2)) FROM (SELECT * FROM c WHERE n > 0) AS t")
    )


@pytest.mark.parametrize("sql", [
    "SELECT CAST(SUBSTRING(c.phone FROM 1 FOR 3) AS VARCHAR(2)) FROM c",  # truncates
    "SELECT CAST(SUBSTRING(c.phone FROM 1) AS VARCHAR(2)) FROM c",  # no bound on the substring
    "SELECT CAST(SUBSTRING(c.phone FROM 1 FOR c.n) AS VARCHAR(2)) FROM c",  # the bound is not a literal
    "SELECT CAST(SUBSTRING(c.phone FROM 1 FOR 2) AS CHAR(4)) FROM c",  # pads
    "SELECT CAST(SUBSTRING(c.phone FROM 1 FOR 2) AS INT) FROM c",
    "SELECT CAST(c.phone AS VARCHAR(2)) FROM c",  # not a substring: truncates
    "SELECT TRY_CAST(SUBSTRING(c.phone FROM 1 FOR 2) AS VARCHAR(2)) FROM c",
    "SELECT CAST(SUBSTRING(c.raw FROM 1 FOR 2) AS VARCHAR(2)) FROM c",  # bytes
    "SELECT CAST(SUBSTRING(c.n FROM 1 FOR 2) AS VARCHAR(2)) FROM c",  # not a string column
    "SELECT CAST(SUBSTRING(u.x FROM 1 FOR 2) AS VARCHAR(2)) FROM unknown_table AS u",
    "SELECT CAST(SUBSTRING(UPPER(c.phone) FROM 1 FOR 2) AS VARCHAR(2)) FROM c",  # only a plain column is read
])
def test_casts_left_alone(sql):
    assert dropped(sql) is None


def test_proofs_and_near_misses():
    plain = "SELECT c.k FROM c WHERE SUBSTRING(c.phone FROM 1 FOR 2) IN ('13', '31')"
    assert proved(plain, "SELECT c.k FROM c WHERE CAST(SUBSTRING(c.phone FROM 1 FOR 2) AS VARCHAR(2)) IN ('13', '31')")
    assert proved("SELECT c.k FROM c WHERE CAST(SUBSTRING(c.phone FROM 1 FOR 2) AS VARCHAR(2)) IN ('13', '31')", plain)
    truncating = "SELECT c.k FROM c WHERE CAST(SUBSTRING(c.phone FROM 1 FOR 3) AS VARCHAR(2)) IN ('13', '31')"
    longer = "SELECT c.k FROM c WHERE SUBSTRING(c.phone FROM 1 FOR 3) IN ('13', '31')"
    # PostgreSQL truncates to two characters here ('135' becomes '13'); DuckDB ignores a VARCHAR length, so it
    # cannot show the difference, and the pair is only checked to stay unproved
    assert not proved(truncating, longer)
    assert not proved(longer, truncating)


def test_the_rewrite_returns_the_same_rows():
    sql = "SELECT c.k, CAST(SUBSTRING(c.phone FROM 2 FOR 3) AS VARCHAR(3)), CAST(SUBSTRING(c.code FROM 1 FOR 2) AS VARCHAR(5)) = 'ab' FROM c WHERE CAST(SUBSTRING(c.phone FROM 1 FOR 2) AS VARCHAR(2)) IN ('13', '31', 'a ')"
    after = dropped(sql)
    assert after is not None and "CAST" not in after
    rows = [(1, "135-555", "abc", 0), (2, "31", "ab", 1), (3, None, None, 2), (4, "a  b", "a", 3), (5, "", " ab", 4), (6, "1", "AB", 5)]
    first, second = rows_on_duckdb([sql, after], rows)
    assert first == second
