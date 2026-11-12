"""A derived table's ``COUNT`` column compared strictly with a whole number (``kumo_sql_prover.count_column_bounds``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.count_column_bounds import count_column_bounds

SCHEMA = {"t": ["k", "v"], "s": ["x", "y"]}
TYPES = {"t": {"k": "BIGINT", "v": "BIGINT"}, "s": {"x": "BIGINT", "y": "BIGINT"}}
G = "(SELECT k, COUNT(*) AS c, SUM(v) AS total, COUNT(v) + 1 AS more, COUNT(DISTINCT v) AS d FROM t GROUP BY k) AS g"


def rewritten(sql):
    out = count_column_bounds(sqlglot.parse_one(sql, read="postgres"))
    return out.sql(dialect="postgres") if out is not None else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="postgres").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name, columns in SCHEMA.items():
        db.execute(f"CREATE TABLE {name} ({', '.join(c + ' BIGINT' for c in columns)})")
        insert_rows(db, name, tables.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="postgres", write="duckdb")[0] for sql in queries])]


def test_strict_bounds_on_a_count_column():
    assert rewritten(f"SELECT g.k FROM {G} WHERE g.c > 4") == f"SELECT g.k FROM {G} WHERE g.c >= 5"
    assert rewritten(f"SELECT g.k FROM {G} WHERE g.c < 4") == f"SELECT g.k FROM {G} WHERE g.c <= 3"
    assert rewritten(f"SELECT g.k FROM {G} WHERE 4 < g.c") == f"SELECT g.k FROM {G} WHERE 5 <= g.c"
    assert rewritten(f"SELECT g.k FROM {G} WHERE 4 > g.c") == f"SELECT g.k FROM {G} WHERE 3 >= g.c"
    assert rewritten(f"SELECT g.k FROM {G} WHERE g.d > 0 AND g.total > 0") == f"SELECT g.k FROM {G} WHERE g.d >= 1 AND g.total > 0"
    assert rewritten(f"SELECT g.k FROM {G} WHERE c > 4") == f"SELECT g.k FROM {G} WHERE c >= 5"  # the select's only source
    assert rewritten(f"SELECT g.k FROM {G} WHERE g.c < 0") == f"SELECT g.k FROM {G} WHERE g.c <= -1"
    assert rewritten(f"SELECT s.y FROM s LEFT JOIN {G} ON g.k = s.x AND g.c > 1") == f"SELECT s.y FROM s LEFT JOIN {G} ON g.k = s.x AND g.c >= 2"


def test_what_is_left_alone():
    for sql in (
        f"SELECT g.k FROM {G} WHERE g.total > 4",  # a sum is read as a real
        f"SELECT g.k FROM {G} WHERE g.more > 4",  # not the COUNT call itself
        f"SELECT g.k FROM {G} WHERE g.c >= 4 AND g.c <= 9 AND g.c = 5 AND g.c <> 6",
        f"SELECT g.k FROM {G} WHERE g.c > 4.5",  # not a whole number
        f"SELECT g.k FROM {G} WHERE g.c > g.d",
        f"SELECT g.k FROM {G}, s WHERE c > 4",  # a bare name with two sources
        "SELECT g.k FROM (SELECT k, COUNT(*) OVER (PARTITION BY k) AS c FROM t) AS g WHERE g.c > 4",
        "SELECT g.k FROM (SELECT k, COUNT(*) FILTER (WHERE v > 0) AS c FROM t GROUP BY k) AS g WHERE g.c > 4",
        "SELECT g.k FROM (SELECT k, COUNT(*) AS c FROM t GROUP BY k UNION ALL SELECT k, v FROM t) AS g WHERE g.c > 4",
        "SELECT g.k FROM (SELECT k, COUNT(*) AS n FROM t GROUP BY k) AS g(k, c) WHERE g.c > 4",
        "SELECT g.k FROM (SELECT k, COUNT(*) AS c, v AS c FROM t GROUP BY k, v) AS g WHERE g.k > 4",
        "SELECT t.k FROM t WHERE t.v > 4",  # a base table's column
        # the inner select's own comparison is the inner select's to rewrite
        f"SELECT s.y FROM s WHERE EXISTS (SELECT 1 FROM {G} WHERE g.c >= 1 AND s.x > 4)",
    ):
        assert rewritten(sql) is None, sql


def test_pairs_it_proves():
    grouped = "SELECT y FROM s WHERE x IN (SELECT k FROM t GROUP BY k HAVING COUNT(*) {})"
    through_cte = "WITH g AS (SELECT k, COUNT(*) AS cnt FROM t GROUP BY k HAVING COUNT(*) {}) SELECT y FROM s WHERE x IN (SELECT k FROM g)"
    assert proved(grouped.format("> 4"), grouped.format(">= 5"))
    assert proved(grouped.format("> 4"), through_cte.format("> 4"))
    assert proved(grouped.format("< 3"), through_cte.format("<= 2"))
    assert proved(
        "SELECT y FROM s WHERE x IN (SELECT g.k FROM (SELECT k, COUNT(*) AS c FROM t GROUP BY k) AS g WHERE g.c > 4)",
        "SELECT y FROM s WHERE x IN (SELECT g.k FROM (SELECT k, COUNT(*) AS c FROM t GROUP BY k) AS g WHERE g.c >= 5)",
    )


def test_near_misses_stay_unproven():
    grouped = "SELECT y FROM s WHERE x IN (SELECT k FROM t GROUP BY k HAVING COUNT(*) {})"
    through_cte = "WITH g AS (SELECT k, COUNT(*) AS cnt FROM t GROUP BY k HAVING COUNT(*) {}) SELECT y FROM s WHERE x IN (SELECT k FROM g)"
    assert not proved(grouped.format("> 4"), grouped.format(">= 4"))
    assert not proved(grouped.format("> 4"), through_cte.format(">= 6"))
    assert not proved(grouped.format("< 3"), through_cte.format("<= 3"))
    derived = "SELECT y FROM s WHERE x IN (SELECT g.k FROM (SELECT k, {} AS c FROM t GROUP BY k) AS g WHERE g.c {})"
    assert not proved(derived.format("COUNT(*)", "> 4"), derived.format("COUNT(*)", ">= 4"))
    assert not proved(derived.format("SUM(v)", "> 4"), derived.format("SUM(v)", ">= 5"))  # 4.5 is a sum in other types
    assert not proved(derived.format("COUNT(*)", "> 4"), derived.format("COUNT(v)", ">= 5"))


def test_the_rewrite_returns_the_same_rows_on_duckdb():
    rng = random.Random(20261112)
    queries = (
        f"SELECT g.k, g.c FROM {G} WHERE g.c > 2 OR g.d < 2",
        f"SELECT s.y, g.c FROM s LEFT JOIN {G} ON g.k = s.x WHERE NOT g.c > 1",
        f"SELECT s.y FROM s WHERE s.x IN (SELECT g.k FROM {G} WHERE 1 < g.c) OR s.y NOT IN (SELECT g.k FROM {G} WHERE g.d < 1)",
    )
    value = lambda: rng.choice([None, 0, 1, 2, 3])
    for _ in range(60):
        tables = {name: [tuple(value() for _ in columns) for _ in range(rng.randint(0, 8))] for name, columns in SCHEMA.items()}
        for sql in queries:
            tree = sqlglot.parse_one(sql, read="postgres")
            for select in list(tree.find_all(sqlglot.exp.Select)):
                count_column_bounds(select)
            after = tree.sql(dialect="postgres")
            assert after != sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres")
            before_rows, after_rows = run(tables, sql, after)
            assert before_rows == after_rows, (sql, tables)
