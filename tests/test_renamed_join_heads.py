"""A column list on the first table of a parenthesized join (``kumo_sql_prover.renamed_join_heads``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.renamed_join_heads import read_back_renamed_join_heads

SCHEMA = {"r": ["a", "b", "x"], "d": ["k", "v"], "u": ["b", "w"]}
TYPES = {table: {column: "BIGINT" for column in columns} for table, columns in SCHEMA.items()}


def read(sql, schema=SCHEMA):
    return read_back_renamed_join_heads(sqlglot.parse_one(sql, read="postgres"), schema).sql(dialect="postgres")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="postgres").proven


def test_the_new_names_are_read_back():
    assert read("SELECT x.v0, r.x FROM r JOIN (d AS x(k0, v0) JOIN u ON x.k0 = u.b) ON r.a = x.v0") == (
        "SELECT x.v AS v0, r.x FROM r JOIN (d AS x JOIN u ON x.k = u.b) ON r.a = x.v"
    )
    # names swapped: each read takes the column in that position
    assert read("SELECT x.v FROM r JOIN (d AS x(v, k) JOIN u ON x.v = u.b) ON r.a = x.k") == (
        "SELECT x.k AS v FROM r JOIN (d AS x JOIN u ON x.k = u.b) ON r.a = x.v"
    )
    # a short list renames the first columns; a correlated subquery and a derived table's own bare columns are fine
    assert read(
        "SELECT r.x FROM r JOIN (d AS x(k0) JOIN (SELECT b FROM u WHERE w = 1) AS t ON x.k0 = t.b) ON r.a = x.v "
        "WHERE EXISTS (SELECT 1 FROM u AS u2 WHERE u2.w = x.k0) GROUP BY r.x, x.k0 ORDER BY x.k0"
    ) == (
        "SELECT r.x FROM r JOIN (d AS x JOIN (SELECT b FROM u WHERE w = 1) AS t ON x.k = t.b) ON r.a = x.v "
        "WHERE EXISTS(SELECT 1 FROM u AS u2 WHERE u2.w = x.k) GROUP BY r.x, x.k ORDER BY x.k"
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT x.k0 FROM d AS x(k0, v0)",  # a FROM item: expand_alias_columns reads it
        "SELECT x.k0 FROM r JOIN d AS x(k0, v0) ON r.a = x.k0",
        "SELECT k0 FROM r JOIN (d AS x(k0, v0) JOIN u ON x.k0 = u.b) ON r.a = x.v0",  # a bare new name
        "SELECT r.x FROM r JOIN (d AS x(k0, v0) JOIN u ON x.k0 = u.b) ON r.a = x.v0 WHERE v > 1",  # a bare old name
        "SELECT * FROM r JOIN (d AS x(k0, v0) JOIN u ON x.k0 = u.b) ON r.a = x.v0",
        "SELECT x.* FROM r JOIN (d AS x(k0, v0) JOIN u ON x.k0 = u.b) ON r.a = x.v0",
        "SELECT r.x FROM r JOIN (d AS x(k0, v0) JOIN u ON x.k0 = u.b) ON r.a = x.nope",  # not a column after renaming
        "SELECT r.x FROM r JOIN (d AS x(k0, v0) JOIN u ON x.k0 = u.b) ON r.a = x.k",  # the old name is gone
        "SELECT r.x FROM r JOIN (d AS x(k0, v0, extra) JOIN u ON x.k0 = u.b) ON r.a = x.v0",  # a longer list
        "SELECT r.x FROM r JOIN (d AS x(v) JOIN u ON x.v = u.b) ON r.a = x.v",  # two columns named v after renaming
        "SELECT r.x FROM r JOIN (missing AS x(k0, v0) JOIN u ON x.k0 = u.b) ON r.a = x.v0",
        "SELECT r.x FROM r JOIN (d AS x(k0, v0) JOIN u USING (b)) ON r.a = x.v0",
        "SELECT r.x FROM r JOIN (d AS x(k0, v0) JOIN u ON x.k0 = u.b) ON r.a = x.v0 WHERE EXISTS (SELECT 1 FROM u AS x WHERE x.w = 1)",
        "WITH d AS (SELECT a AS p, b AS q FROM r) SELECT r.x FROM r JOIN (d AS x(k0, v0) JOIN u ON x.k0 = u.b) ON r.a = x.v0",
    ],
)
def test_what_is_left_alone(sql):
    assert read(sql) == sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres"), sql


def test_no_schema_no_change():
    sql = "SELECT x.v0 FROM r JOIN (d AS x(k0, v0) JOIN u ON x.k0 = u.b) ON r.a = x.v0"
    assert read(sql, None) == sql


def test_pairs_it_proves():
    renamed = "SELECT r.x, x.v0 FROM r JOIN (d AS x(k0, v0) JOIN (SELECT u.b FROM u GROUP BY u.b) AS t ON x.k0 = t.b) ON r.a = x.v0"
    assert proved(renamed, "SELECT r.x, d.v FROM r, d WHERE r.a = d.v AND EXISTS (SELECT 1 FROM u WHERE u.b = d.k)")
    swapped = "SELECT r.x FROM r JOIN (d AS x(v, k) JOIN u ON x.v = u.b) ON r.a = x.k WHERE x.v > 3"
    assert proved(swapped, "SELECT r.x FROM r, d, u WHERE d.k = u.b AND r.a = d.v AND d.k > 3")


def test_near_misses_stay_unproven():
    renamed = "SELECT r.x, x.v0 FROM r JOIN (d AS x(k0, v0) JOIN (SELECT u.b FROM u GROUP BY u.b) AS t ON x.k0 = t.b) ON r.a = x.v0"
    # the columns the other way round
    assert not proved(renamed, "SELECT r.x, d.k FROM r, d WHERE r.a = d.k AND EXISTS (SELECT 1 FROM u WHERE u.b = d.v)")
    assert not proved(renamed, "SELECT r.x, d.v FROM r, d WHERE EXISTS (SELECT 1 FROM u WHERE u.b = d.k)")
    swapped = "SELECT r.x FROM r JOIN (d AS x(v, k) JOIN u ON x.v = u.b) ON r.a = x.k WHERE x.v > 3"
    # reading the new names as if they were the table's own
    assert not proved(swapped, "SELECT r.x FROM r, d, u WHERE d.v = u.b AND r.a = d.k AND d.v > 3")
    assert not proved(swapped, "SELECT r.x FROM r, d, u WHERE d.k = u.b AND r.a = d.v AND d.v > 3")


def test_the_engine_agrees_on_random_tables():
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(20261116)
    value = lambda: rng.choice([None, *range(0, 4)])  # noqa: E731
    queries = [
        "SELECT x.v0, r.x FROM r JOIN (d AS x(k0, v0) JOIN u ON x.k0 = u.b) ON r.a = x.v0",
        "SELECT x.v FROM r JOIN (d AS x(v, k) JOIN u ON x.v = u.b) ON r.a = x.k",
        "SELECT r.x, COUNT(*) FROM r JOIN (d AS x(k0) JOIN (SELECT b FROM u WHERE w = 1) AS t ON x.k0 = t.b) ON r.a = x.v "
        "WHERE EXISTS (SELECT 1 FROM u AS u2 WHERE u2.w = x.k0) GROUP BY r.x, x.k0",
        "SELECT r.x, x.v FROM r LEFT JOIN (d AS x(v, k) LEFT JOIN u ON x.v = u.b) ON r.a = x.k",
    ]
    rows_seen = 0
    for trial in range(30):
        db = duckdb.connect()
        for table, columns in SCHEMA.items():
            db.execute(f"CREATE TABLE {table} ({', '.join(c + ' BIGINT' for c in columns)})")
            rows = [tuple(value() for _ in columns) for _ in range(rng.choice([0, 1, 3, 7]) if trial % 5 else 4)]
            if rows:
                db.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in columns)})", rows)
        for sql in queries:
            after = read(sql)
            assert after != sql and "(k" not in after and "(v" not in after, sql
            key = lambda row: tuple((v is None, v) for v in row)  # noqa: E731
            before_rows = sorted(db.execute(sql).fetchall(), key=key)
            assert before_rows == sorted(db.execute(after).fetchall(), key=key), (sql, after)
            assert [d[0] for d in db.execute(sql).description] == [d[0] for d in db.execute(after).description]
            rows_seen += len(before_rows)
    assert rows_seen > 100


def test_off_on_the_second_attempt():
    sql = "SELECT r.x FROM r JOIN (d AS x(v, k) JOIN u ON x.v = u.b) ON r.a = x.k"
    assert "x.v = " not in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES).replace("r.a = x.v", "")
    with shape_fallback.disabled():
        try:
            unread = normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)
        except Exception:  # declined without the pass
            return
        assert "x(v, k)" in unread or "x.v = u.b" in unread
