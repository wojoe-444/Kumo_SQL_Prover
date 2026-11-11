"""A joined single-value derived table read as its scalar subquery (``kumo_sql_prover.single_value_joins``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.single_value_joins import read_single_value_join

SCHEMA = {"a": ["k", "d"], "dd": ["sk", "d", "m"]}
TYPES = {table: {column: "BIGINT" for column in columns} for table, columns in SCHEMA.items()}

CASE = "CASE COUNT(*) WHEN 0 THEN NULL WHEN 1 THEN MIN(t.m) ELSE (SELECT CAST(NULL AS INT) UNION ALL SELECT CAST(NULL AS INT)) END"
ONE = f"(SELECT {CASE} AS f FROM dd AS t WHERE t.d = 2) AS s"
NAMED = "(SELECT SINGLE_VALUE(t.m) AS f FROM dd AS t WHERE t.d = 2) AS s"
SCALAR = "(SELECT t.m FROM dd AS t WHERE t.d = 2)"


def read(sql, dialect="postgres"):
    out = read_single_value_join(sqlglot.parse_one(sql, read="postgres"), dialect)
    return None if out is None else out.sql(dialect="postgres")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="postgres").proven


def test_the_join_becomes_a_filter():
    assert read(f"SELECT x.sk FROM dd AS x JOIN {NAMED} ON x.m = s.f") == f"SELECT x.sk FROM dd AS x WHERE x.m = {SCALAR}"
    assert read(f"SELECT x.sk FROM dd AS x JOIN {ONE} ON x.m = s.f") == f"SELECT x.sk FROM dd AS x WHERE x.m = {SCALAR}"
    assert read(f"SELECT x.sk FROM dd AS x JOIN {ONE.replace('MIN', 'MAX')} ON x.m = s.f") == f"SELECT x.sk FROM dd AS x WHERE x.m = {SCALAR}"
    # the condition joins the filter that is there; an OR keeps its parentheses
    assert read(f"SELECT x.sk FROM dd AS x JOIN {NAMED} ON x.m = s.f OR x.d = 1 WHERE x.sk > 0") == f"SELECT x.sk FROM dd AS x WHERE x.sk > 0 AND (x.m = {SCALAR} OR x.d = 1)"
    # with no condition, with ON TRUE, as a comma join, as LEFT JOIN .. ON TRUE; a lone select-list read keeps its name
    for join in (f"CROSS JOIN {NAMED}", f"JOIN {ONE} ON TRUE", f", {ONE}", f"LEFT JOIN {NAMED} ON TRUE", f"LEFT JOIN {ONE} ON TRUE"):
        assert read(f"SELECT x.sk, s.f FROM dd AS x {join} WHERE x.m = s.f") == f"SELECT x.sk, {SCALAR} AS f FROM dd AS x WHERE x.m = {SCALAR}", join
    # a LEFT join elsewhere does not stop a join without a condition
    assert read(f"SELECT x.sk FROM dd AS x LEFT JOIN a ON a.d = x.d CROSS JOIN {NAMED} WHERE x.m = s.f") is not None
    # reads in HAVING and inside an aggregate; the table unread
    assert read(f"SELECT x.d, SUM(x.m - s.f) FROM dd AS x JOIN {ONE} ON x.m >= s.f GROUP BY x.d HAVING MAX(s.f) > 0") is not None
    assert read(f"SELECT x.sk FROM dd AS x CROSS JOIN {NAMED}") == "SELECT x.sk FROM dd AS x"


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT x.sk FROM dd AS x LEFT JOIN {NAMED} ON x.m = s.f",  # the unmatched rows stay
        f"SELECT x.sk FROM dd AS x RIGHT JOIN {NAMED} ON x.m = s.f",
        f"SELECT x.sk FROM dd AS x FULL JOIN {NAMED} ON TRUE",
        f"SELECT x.sk FROM dd AS x JOIN {NAMED} ON x.m = s.f LEFT JOIN a ON a.d = x.d",  # a condition, and an outer join after it
        f"SELECT x.sk FROM dd AS x JOIN {NAMED} ON x.m = s.f RIGHT JOIN a ON a.d = x.d",
        f"SELECT x.sk FROM {NAMED} JOIN dd AS x ON x.m = s.f",  # the FROM item
        f"SELECT x.sk FROM dd AS x JOIN {NAMED} USING (f)",
        f"SELECT x.sk FROM dd AS x JOIN LATERAL {NAMED} ON x.m = s.f",
        f"SELECT * FROM dd AS x JOIN {NAMED} ON x.m = s.f",
        f"SELECT x.sk, s.* FROM dd AS x JOIN {NAMED} ON x.m = s.f",
        f"SELECT x.sk FROM dd AS x JOIN {NAMED} ON x.m = f",  # a bare read
        f"SELECT x.sk FROM dd AS x JOIN {NAMED} ON x.m = s.f GROUP BY x.sk, s.f",
        f"SELECT x.sk FROM dd AS x JOIN {NAMED} ON x.m = s.f ORDER BY s.f",
        f"SELECT x.sk, SUM(x.m) OVER (PARTITION BY s.f) FROM dd AS x JOIN {NAMED} ON x.m = s.f",
        f"SELECT x.sk FROM dd AS x JOIN {NAMED} ON x.m = s.f WHERE EXISTS (SELECT 1 FROM a AS s WHERE s.d = x.d)",  # the alias again
        f"SELECT x.sk FROM dd AS x JOIN {NAMED} ON x.m = s.g",  # not an output
        "SELECT x.sk FROM dd AS x JOIN (SELECT SINGLE_VALUE(t.m) AS f FROM dd AS t WHERE t.d = 2) AS s(g) ON x.m = s.g",
        "SELECT x.sk FROM dd AS x JOIN (SELECT SINGLE_VALUE(m) AS f FROM dd AS t WHERE t.d = 2) AS s ON x.m = s.f",  # a bare column inside
        "SELECT x.sk FROM dd AS x JOIN (SELECT SINGLE_VALUE(t.m) AS f, COUNT(*) AS n FROM dd AS t) AS s ON x.m = s.f",
        "SELECT x.sk FROM dd AS x JOIN (SELECT SINGLE_VALUE(t.m) AS f FROM dd AS t GROUP BY t.d) AS s ON x.m = s.f",
        "SELECT x.sk FROM dd AS x JOIN (SELECT SINGLE_VALUE(t.m) AS f FROM dd AS t HAVING COUNT(*) > 0) AS s ON x.m = s.f",
        "SELECT x.sk FROM dd AS x JOIN (SELECT SINGLE_VALUE(t.m) AS f FROM dd AS t LIMIT 0) AS s ON x.m = s.f",
        "SELECT x.sk FROM dd AS x JOIN (SELECT DISTINCT SINGLE_VALUE(t.m) AS f FROM dd AS t) AS s ON x.m = s.f",
        "SELECT x.sk FROM dd AS x JOIN (SELECT SINGLE_VALUE(SUM(t.m)) AS f FROM dd AS t) AS s ON x.m = s.f",
        "SELECT x.sk FROM dd AS x JOIN (SELECT SINGLE_VALUE((SELECT MAX(a.k) FROM a)) AS f FROM dd AS t) AS s ON x.m = s.f",
        "SELECT x.sk FROM dd AS x JOIN (SELECT MIN(t.m) AS f FROM dd AS t WHERE t.d = 2) AS s ON x.m = s.f",  # the least of many rows
        "SELECT x.sk FROM dd AS x JOIN (SELECT t.m AS f FROM dd AS t WHERE t.d = 2) AS s ON x.m = s.f",  # a row per row
        "SELECT x.sk FROM dd AS x JOIN (SELECT SINGLE_VALUE(t.m) AS f) AS s ON x.m = s.f",  # no FROM
        # an outer reference inside the derived table would meet the select's own sources once moved
        "SELECT a.k FROM a WHERE EXISTS (SELECT 1 FROM dd AS x JOIN (SELECT SINGLE_VALUE(t.m + a.k) AS f FROM dd AS t) AS s ON x.m = s.f)",
    ],
)
def test_what_is_left_alone(sql):
    tree = sqlglot.parse_one(sql, read="postgres")
    for select in tree.find_all(sqlglot.exp.Select):
        assert read_single_value_join(select, "postgres") is None, sql


@pytest.mark.parametrize(
    "value",
    [
        "CASE COUNT(*) WHEN 0 THEN NULL WHEN 1 THEN MIN(t.m) ELSE NULL END",  # many rows are NULL, not an error
        "CASE COUNT(*) WHEN 0 THEN NULL WHEN 1 THEN MIN(t.m) END",
        "CASE COUNT(*) WHEN 0 THEN NULL WHEN 1 THEN MIN(t.m) ELSE (SELECT NULL UNION SELECT NULL) END",  # one row
        "CASE COUNT(*) WHEN 0 THEN NULL WHEN 1 THEN MIN(t.m) ELSE (SELECT NULL UNION ALL SELECT NULL LIMIT 1) END",
        "CASE COUNT(*) WHEN 0 THEN NULL WHEN 1 THEN MIN(t.m) ELSE (SELECT NULL UNION ALL SELECT NULL WHERE 1 = 0) END",
        "CASE COUNT(*) WHEN 0 THEN NULL WHEN 1 THEN MIN(t.m) ELSE (SELECT NULL UNION ALL SELECT a.k FROM a) END",
        "CASE COUNT(*) WHEN 0 THEN NULL WHEN 1 THEN MIN(t.m) ELSE (SELECT NULL) END",
        "CASE COUNT(*) WHEN 0 THEN 0 WHEN 1 THEN MIN(t.m) ELSE (SELECT NULL UNION ALL SELECT NULL) END",
        "CASE COUNT(*) WHEN 0 THEN NULL WHEN 2 THEN MIN(t.m) ELSE (SELECT NULL UNION ALL SELECT NULL) END",
        "CASE COUNT(*) WHEN 0 THEN NULL WHEN 1 THEN SUM(t.m) ELSE (SELECT NULL UNION ALL SELECT NULL) END",
        "CASE COUNT(*) WHEN 0 THEN NULL WHEN 1 THEN MIN(t.m) + 1 ELSE (SELECT NULL UNION ALL SELECT NULL) END",
        "CASE COUNT(*) WHEN 0 THEN NULL WHEN 1 THEN MIN(DISTINCT t.m) ELSE (SELECT NULL UNION ALL SELECT NULL) END",
        "CASE COUNT(*) WHEN 0 THEN NULL WHEN 1 THEN MIN(t.m) WHEN 2 THEN 5 ELSE (SELECT NULL UNION ALL SELECT NULL) END",
        "CASE COUNT(t.m) WHEN 0 THEN NULL WHEN 1 THEN MIN(t.m) ELSE (SELECT NULL UNION ALL SELECT NULL) END",  # NULLs not counted
        "CASE COUNT(DISTINCT t.m) WHEN 0 THEN NULL WHEN 1 THEN MIN(t.m) ELSE (SELECT NULL UNION ALL SELECT NULL) END",
        "CASE COUNT(*) FILTER (WHERE t.m > 0) WHEN 0 THEN NULL WHEN 1 THEN MIN(t.m) ELSE (SELECT NULL UNION ALL SELECT NULL) END",
        "CASE WHEN COUNT(*) > 1 THEN (SELECT NULL UNION ALL SELECT NULL) ELSE MIN(t.m) END",  # another spelling: not read
    ],
)
def test_a_case_that_is_not_the_single_value(value):
    assert read(f"SELECT x.sk FROM dd AS x JOIN (SELECT {value} AS f FROM dd AS t WHERE t.d = 2) AS s ON x.m = s.f") is None, value


def test_the_case_is_not_read_for_sqlite():
    # there a scalar subquery of two rows is its first row, so the CASE is NULL on many rows and not an error
    assert read(f"SELECT x.sk FROM dd AS x JOIN {ONE} ON x.m = s.f", dialect="sqlite") is None
    assert read(f"SELECT x.sk FROM dd AS x JOIN {NAMED} ON x.m = s.f", dialect="sqlite") is not None


def test_pairs_it_proves():
    scalar = "SELECT a.k FROM a WHERE a.d IN (SELECT d FROM dd WHERE m = (SELECT m FROM dd WHERE d = 2))"
    grouped = "SELECT a.k FROM a INNER JOIN (SELECT x.d FROM dd AS x {} GROUP BY x.d) AS g ON a.d = g.d"
    assert proved(scalar, grouped.format(f"INNER JOIN {ONE} ON x.m = s.f"))
    assert proved(scalar, grouped.format(f"LEFT JOIN {ONE} ON TRUE WHERE x.m = s.f"))
    assert proved(scalar, grouped.format(f"LEFT JOIN {NAMED} ON TRUE WHERE x.m = s.f"))
    assert proved("SELECT x.sk FROM dd AS x WHERE x.m > (SELECT t.m FROM dd AS t WHERE t.d = 2)", f"SELECT x.sk FROM dd AS x JOIN {ONE} ON x.m > s.f")


def test_near_misses_stay_unproven():
    scalar = "SELECT x.sk FROM dd AS x WHERE x.m = (SELECT t.m FROM dd AS t WHERE t.d = 2)"
    assert not proved(scalar, f"SELECT x.sk FROM dd AS x LEFT JOIN {ONE} ON x.m = s.f")
    assert not proved(scalar, f"SELECT x.sk FROM dd AS x JOIN {ONE.replace('t.d = 2', 't.d = 3')} ON x.m = s.f")
    assert not proved(scalar, f"SELECT x.sk FROM dd AS x JOIN {ONE} ON x.d = s.f")
    assert not proved(scalar, f"SELECT x.sk FROM dd AS x JOIN {ONE.replace('MIN(t.m)', 'MIN(t.sk)')} ON x.m = s.f")
    assert not proved(scalar, f"SELECT x.sk FROM dd AS x JOIN {ONE} ON x.m = s.f OR s.f IS NULL")
    assert not proved(scalar, "SELECT x.sk FROM dd AS x JOIN (SELECT t.m AS f FROM dd AS t WHERE t.d = 2) AS s ON x.m = s.f JOIN a ON TRUE")


def test_the_engine_agrees_on_random_tables():
    # Where the derived table has more than one row both spellings are an error in PostgreSQL, and DuckDB plans
    # the two-row subquery eagerly; so the tables hold at most one row of each ``d`` and the error is switched off.
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(20261118)
    queries = [
        f"SELECT x.sk FROM dd AS x JOIN {ONE} ON x.m = s.f",
        f"SELECT x.sk, s.f FROM dd AS x LEFT JOIN {ONE} ON TRUE",
        f"SELECT x.sk, s.f + 1 AS g FROM dd AS x, {ONE} WHERE x.m <= s.f OR s.f IS NULL",
        f"SELECT a.k, COUNT(*), MAX(s.f) FROM a JOIN dd AS x ON a.d = x.d JOIN {ONE.replace('MIN', 'MAX')} ON x.m >= s.f GROUP BY a.k HAVING MAX(s.f) > 0",
        f"SELECT a.k FROM a WHERE a.d IN (SELECT x.d FROM dd AS x JOIN {ONE} ON x.m = s.f)",
    ]
    rows_seen = 0
    for trial in range(40):
        db = duckdb.connect()
        db.execute("SET scalar_subquery_error_on_multiple_rows=false")
        db.execute("CREATE TABLE a (k BIGINT, d BIGINT)")
        db.execute("CREATE TABLE dd (sk BIGINT, d BIGINT, m BIGINT)")
        if trial % 7:
            days = rng.sample([1, 2, 3, 4, 5], rng.choice([0, 2, 4, 5])) + [None] * rng.choice([0, 2])
            for sk, day in enumerate(days):
                db.execute("INSERT INTO dd VALUES (?, ?, ?)", [sk, day, rng.choice([None, 0, 1, 1, 2])])
            for _ in range(rng.choice([0, 3, 8])):
                db.execute("INSERT INTO a VALUES (?, ?)", [rng.choice([None, 1, 2]), rng.choice([None, 1, 2, 3, 4])])
        for sql in queries:
            tree = sqlglot.parse_one(sql, read="postgres")
            changed = [read_single_value_join(select, "postgres") is not None for select in list(tree.find_all(sqlglot.exp.Select))]
            assert any(changed), sql
            after = tree.sql(dialect="postgres")
            assert "COUNT(*) WHEN" not in after
            key = lambda row: tuple((v is None, v) for v in row)  # noqa: E731
            before_rows = sorted(db.execute(sql).fetchall(), key=key)
            assert before_rows == sorted(db.execute(after).fetchall(), key=key), (sql, after)
            rows_seen += len(before_rows)
    assert rows_seen > 100


def test_off_on_the_second_attempt():
    sql = f"SELECT x.sk FROM dd AS x JOIN {ONE} ON x.m = s.f"
    assert "COUNT(*)" not in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)
    with shape_fallback.disabled():
        assert "COUNT(*)" in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)
