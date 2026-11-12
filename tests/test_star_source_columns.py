"""A derived ``SELECT *`` over repeated names lists the columns read from it (``kumo_sql_prover.star_source_columns``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.star_source_columns import list_read_star_columns

SCHEMA = {"r": ["a", "b", "x"], "d": ["k", "v"], "u": ["b", "w"]}
TYPES = {table: {column: "BIGINT" for column in columns} for table, columns in SCHEMA.items()}

TWICE = "(SELECT * FROM r, d AS d1, d AS d2 WHERE r.a = d1.k AND r.b = d2.k AND d1.v < d2.v) AS t"


def listed(sql):
    return list_read_star_columns(sqlglot.parse_one(sql, read="postgres"), SCHEMA).sql(dialect="postgres")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="postgres").proven


def test_the_star_becomes_the_columns_read():
    assert listed(f"SELECT MIN(t.a), MIN(t.x) FROM {TWICE}, u WHERE t.b = u.b") == (
        "SELECT MIN(t.a), MIN(t.x) FROM (SELECT r.a, r.b, r.x FROM r, d AS d1, d AS d2 WHERE r.a = d1.k AND r.b = d2.k AND d1.v < d2.v) AS t, u WHERE t.b = u.b"
    )
    # level by level from the outside, through a star over a star
    assert listed(f"SELECT t1.w, t1.a FROM (SELECT * FROM {TWICE}, u WHERE t.b = u.w) AS t1 WHERE t1.x > 0") == (
        "SELECT t1.w, t1.a FROM (SELECT t.a, t.x, u.w FROM (SELECT r.a, r.b, r.x FROM r, d AS d1, d AS d2 "
        "WHERE r.a = d1.k AND r.b = d2.k AND d1.v < d2.v) AS t, u WHERE t.b = u.w) AS t1 WHERE t1.x > 0"
    )
    # a column read only by a correlated subquery counts
    assert "SELECT r.a, r.x FROM r" in listed(f"SELECT t.a FROM {TWICE} WHERE EXISTS (SELECT 1 FROM u WHERE u.w = t.x)")
    # a written list with a name twice loses its unread items
    assert listed("SELECT t.a, t.v FROM (SELECT r.a, d1.k, d1.v, d2.k, r.x + 1 AS v2 FROM r, d AS d1, d AS d2) AS t") == (
        "SELECT t.a, t.v FROM (SELECT r.a, d1.v FROM r, d AS d1, d AS d2) AS t"
    )


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT * FROM {TWICE}",  # a star reads every column (an error in a derived table, fine at the top)
        f"SELECT t.* FROM {TWICE}",
        f"SELECT a FROM {TWICE}",  # a bare column
        f"SELECT t.a FROM {TWICE} ORDER BY x",
        f"SELECT t.k FROM {TWICE}",  # there twice: PostgreSQL's error stays
        f"SELECT t.nope FROM {TWICE}",
        f"SELECT COUNT(*) FROM {TWICE}",  # nothing read
        f"SELECT t.a FROM {TWICE}, {TWICE}",
        f"SELECT t.a FROM {TWICE} WHERE EXISTS (SELECT 1 FROM u AS t WHERE t.w = 1)",  # the alias means another table there
        "SELECT t.a FROM (SELECT DISTINCT * FROM r, d AS d1, d AS d2) AS t",  # fewer columns merge more rows
        "SELECT t.a FROM (SELECT * FROM r, d AS d1, d AS d2 LIMIT 3) AS t",
        "SELECT t.a FROM (SELECT * FROM r, d AS d1, d AS d2 ORDER BY 1) AS t",
        "SELECT t.a FROM (SELECT * FROM r, d AS d1 UNION ALL SELECT * FROM r, d AS d2) AS t",
        "SELECT t.a FROM (SELECT * FROM r, d, d) AS t",  # two sources of one name
        "SELECT t.a FROM (SELECT * FROM r, missing AS m1, missing AS m2) AS t",  # a table the schema does not have
        "SELECT t.a FROM (SELECT * FROM r JOIN d AS d1 USING (a) JOIN d AS d2 USING (a)) AS t",
        "SELECT t.a FROM (SELECT * FROM r NATURAL JOIN d AS d1, d AS d2) AS t",
        "SELECT t.a FROM (SELECT * FROM r AS r(a, b, x), d AS d1(a, v), d AS d2) AS t",  # the columns are renamed
        "SELECT t.a FROM (SELECT * EXCEPT (x) FROM r, d AS d1, d AS d2) AS t",
        "SELECT t.a FROM (SELECT d1.*, * FROM r, d AS d1, d AS d2) AS t",
        "SELECT t.p FROM (SELECT * FROM r, d AS d1, d AS d2) AS t(p, q, s, e, f, g, h)",
        "SELECT t.a FROM (SELECT r.a, COUNT(*) AS c, SUM(d1.v) AS c FROM r, d AS d1) AS t",  # an aggregate decides the row count
        "SELECT t.a FROM (SELECT r.a, d1.k, d2.k FROM r, d AS d1, d AS d2 GROUP BY r.a, d1.k, d2.k) AS t",
        "SELECT t.a FROM (SELECT DISTINCT r.a, d1.k, d2.k FROM r, d AS d1, d AS d2) AS t",
        "SELECT t.a FROM (SELECT r.a, r.b, r.x FROM r) AS t",  # nothing repeated: the list stays whole
    ],
)
def test_what_is_left_alone(sql):
    tree = sqlglot.parse_one(sql, read="bigquery" if "EXCEPT (" in sql else "postgres")
    assert list_read_star_columns(tree.copy(), SCHEMA).sql() == tree.sql(), sql


def test_no_schema_no_change():
    sql = f"SELECT t.a FROM {TWICE}"
    assert list_read_star_columns(sqlglot.parse_one(sql, read="postgres"), None).sql(dialect="postgres") == sql


def test_pairs_it_proves():
    flat = "SELECT MIN(r.a), MIN(r.x) FROM r, d AS d1, d AS d2, u WHERE r.a = d1.k AND r.b = d2.k AND d1.v < d2.v AND r.b = u.b AND u.w = 3"
    assert proved(flat, f"SELECT MIN(t.a), MIN(t.x) FROM {TWICE}, (SELECT * FROM u WHERE w = 3) AS t2 WHERE t.b = t2.b")
    # Calcite's shape: one star per join level
    levels = (
        "SELECT MIN(t3.a), MIN(t3.x) FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM r, d WHERE r.a = d.k) AS t0, d AS d0 "
        "WHERE t0.b = d0.k AND t0.v < d0.v) AS t1, (SELECT * FROM u WHERE w = 3) AS t2 WHERE t1.b = t2.b) AS t3"
    )
    assert proved(flat, levels)


def test_near_misses_stay_unproven():
    flat = "SELECT MIN(r.a), MIN(r.x) FROM r, d AS d1, d AS d2 WHERE r.a = d1.k AND r.b = d2.k AND d1.v < d2.v"
    assert proved(flat, f"SELECT MIN(t.a), MIN(t.x) FROM {TWICE}")
    assert not proved(flat, f"SELECT MIN(t.a), MIN(t.b) FROM {TWICE}")
    assert not proved(flat, f"SELECT MIN(t.a), MIN(t.x) FROM {TWICE.replace('d1.v < d2.v', 'd1.v <= d2.v')}")
    assert not proved(flat, f"SELECT MIN(t.a), MIN(t.x) FROM {TWICE.replace('r.b = d2.k', 'r.x = d2.k')}")
    # the unread columns still decide how many rows there are
    count = "SELECT COUNT(t.a) FROM {}"
    assert not proved(count.format(TWICE), count.format("(SELECT r.a FROM r, d AS d1 WHERE r.a = d1.k) AS t"))
    assert not proved(count.format("(SELECT DISTINCT * FROM r, d AS d1, d AS d2) AS t"), count.format("(SELECT DISTINCT r.a FROM r, d AS d1, d AS d2) AS t"))


def test_the_engine_agrees_on_random_tables():
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(20261115)
    value = lambda: rng.choice([None, *range(0, 4)])  # noqa: E731
    queries = [
        f"SELECT MIN(t.a), MIN(t.x), COUNT(*) FROM {TWICE}, u WHERE t.b = u.b",
        f"SELECT t1.w, t1.a FROM (SELECT * FROM {TWICE}, u WHERE t.b = u.w) AS t1 WHERE t1.x > 0",
        f"SELECT t.a FROM {TWICE} WHERE EXISTS (SELECT 1 FROM u WHERE u.w = t.x)",
        "SELECT t.a, t.v FROM (SELECT r.a, d1.k, d1.v, d2.k, r.x + 1 AS v2 FROM r, d AS d1, d AS d2) AS t",
        "SELECT t.x, SUM(t.b) FROM (SELECT * FROM r LEFT JOIN d AS d1 ON r.a = d1.k LEFT JOIN d AS d2 ON r.b = d2.k) AS t GROUP BY t.x",
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
            after = listed(sql)
            assert after != sql, sql
            key = lambda row: tuple((v is None, v) for v in row)  # noqa: E731
            before_rows = sorted(db.execute(sql).fetchall(), key=key)
            assert before_rows == sorted(db.execute(after).fetchall(), key=key), (sql, after)
            rows_seen += len(before_rows)
    assert rows_seen > 100


def test_off_on_the_second_attempt():
    sql = f"SELECT MIN(t1.a) FROM (SELECT * FROM {TWICE}, u WHERE t.b = u.b) AS t1"
    assert "*" not in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)
    with shape_fallback.disabled():
        assert "*" in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)
