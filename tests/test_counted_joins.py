"""COUNT(*) over a FULL JOIN against its LEFT JOIN UNION ALL RIGHT JOIN rewrite (``kumo_sql_prover.counted_joins``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.counted_joins import counted_join_rules

SCHEMA = {"a": ["k", "v"], "b": ["k", "w"]}
TYPES = {name: {column: "INT64" for column in columns} for name, columns in SCHEMA.items()}
HALVES = "SELECT a.k, b.k AS k2 FROM a LEFT JOIN b ON {on} UNION ALL SELECT a.k, b.k AS k2 FROM a RIGHT JOIN b ON {on} WHERE a.k IS NULL"


def rewritten(sql):
    return counted_join_rules(sqlglot.parse_one(sql, read="bigquery")).sql(dialect="bigquery")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False).proven


def test_the_three_rewrites():
    # 1: an IS NOT NULL test keeps the tested side
    assert rewritten("SELECT a.v FROM a FULL JOIN b ON a.k = b.k WHERE NOT b.w IS NULL") == "SELECT a.v FROM a RIGHT JOIN b ON a.k = b.k WHERE NOT b.w IS NULL"
    assert rewritten("SELECT b.w FROM a FULL OUTER JOIN b ON a.k = b.k WHERE a.v > 1 AND NOT a.v IS NULL") == "SELECT b.w FROM a LEFT JOIN b ON a.k = b.k WHERE a.v > 1 AND NOT a.v IS NULL"
    assert rewritten("SELECT 1 FROM a FULL JOIN b ON a.k = b.k WHERE NOT a.v IS NULL AND NOT b.w IS NULL") == "SELECT 1 FROM a JOIN b ON a.k = b.k WHERE NOT a.v IS NULL AND NOT b.w IS NULL"
    # 2: the count of a full join is the count of its halves
    halves = "SELECT COUNT(*) FROM (SELECT 1 AS c0 FROM a LEFT JOIN b ON a.k = b.k AND b.w = 2 UNION ALL SELECT 1 AS c0 FROM a RIGHT JOIN b ON a.k = b.k AND b.w = 2 WHERE a.k IS NULL)"
    assert rewritten("SELECT COUNT(*) FROM a FULL JOIN b ON a.k = b.k AND b.w = 2") == halves
    # 3: COUNT(*) does not read the columns of the halves
    assert rewritten(f"SELECT COUNT(*) FROM ({HALVES.format(on='a.k = b.k AND b.w = 2')}) AS j") == halves


@pytest.mark.parametrize("sql", [
    "SELECT a.v FROM a FULL JOIN b ON a.k = b.k WHERE b.w IS NULL",  # the test keeps the padded rows
    "SELECT a.v FROM a FULL JOIN b ON a.k = b.k WHERE NOT b.w IS NULL OR a.v = 1",  # not a conjunct
    "SELECT a.v FROM a FULL JOIN b ON a.k = b.k WHERE NOT w IS NULL",  # whose column?
    "SELECT a.v FROM a FULL JOIN b ON a.k = b.k WHERE NOT (a.v + b.w) IS NULL",  # not a column
    "SELECT a.v FROM a FULL JOIN b USING (k) WHERE NOT b.w IS NULL",
    "SELECT x.v FROM a AS x FULL JOIN a AS x ON TRUE WHERE NOT x.v IS NULL",
    "SELECT a.v FROM a FULL JOIN b ON a.k = b.k JOIN b AS c ON TRUE WHERE NOT b.w IS NULL",  # two joins
    "SELECT COUNT(*) FROM a FULL JOIN b ON b.k = 1",  # no comparison on a column of a
    "SELECT COUNT(*) FROM a FULL JOIN b ON a.k IS NOT DISTINCT FROM b.k",  # TRUE for two NULLs
    "SELECT COUNT(*) FROM a FULL JOIN b ON a.k = b.k OR b.w = 1",  # not a conjunct
    "SELECT COUNT(*) FROM a FULL JOIN b ON COALESCE(a.k, 0) = b.k",  # a.k can be NULL on a match
    "SELECT COUNT(*) FROM a FULL JOIN b USING (k)",
    "SELECT COUNT(*) FROM a FULL JOIN b ON a.k = b.k WHERE a.v = 1",
    "SELECT COUNT(*) FROM a FULL JOIN b ON a.k = b.k GROUP BY a.v",
    "SELECT COUNT(*), MAX(a.v) FROM a FULL JOIN b ON a.k = b.k",
    "SELECT COUNT(a.v) FROM a FULL JOIN b ON a.k = b.k",
    "SELECT COUNT(DISTINCT a.v) FROM a FULL JOIN b ON a.k = b.k",
    "SELECT COUNT(*) FROM a LEFT JOIN b ON a.k = b.k",
    "SELECT COUNT(*) FROM a FULL JOIN b ON a.k = b.k LIMIT 0",
    # 3 declined: the number of rows of a branch depends on its select list or its clauses
    "SELECT COUNT(*) FROM (SELECT a.k FROM a LEFT JOIN b ON a.k = b.k UNION DISTINCT SELECT b.k FROM b)",
    "SELECT COUNT(*) FROM (SELECT DISTINCT a.k FROM a LEFT JOIN b ON a.k = b.k UNION ALL SELECT b.k FROM b)",
    "SELECT COUNT(*) FROM (SELECT MAX(a.k) FROM a LEFT JOIN b ON a.k = b.k UNION ALL SELECT b.k FROM b)",
    "SELECT COUNT(*) FROM (SELECT a.k FROM a LEFT JOIN b ON a.k = b.k GROUP BY a.k UNION ALL SELECT b.k FROM b)",
    "SELECT COUNT(*) FROM (SELECT a.k FROM a LEFT JOIN b ON a.k = b.k UNION ALL SELECT b.k FROM b LIMIT 1)",
    "SELECT COUNT(*) FROM (SELECT a.k FROM a LEFT JOIN b ON a.k = b.k UNION ALL (SELECT b.k FROM b ORDER BY b.k LIMIT 1))",
    "SELECT COUNT(*) FROM (SELECT a.k, ROW_NUMBER() OVER () FROM a LEFT JOIN b ON a.k = b.k UNION ALL SELECT b.k, 1 FROM b)",
    "SELECT COUNT(*) FROM (SELECT (SELECT 1) FROM a LEFT JOIN b ON a.k = b.k UNION ALL SELECT b.k FROM b)",
    "SELECT COUNT(*) FROM (SELECT a.k FROM a LEFT JOIN b ON a.k = b.k UNION ALL SELECT b.k FROM b) AS j WHERE j.k = 1",  # the outer select reads a column
    "SELECT COUNT(j.k) FROM (SELECT a.k FROM a LEFT JOIN b ON a.k = b.k UNION ALL SELECT b.k FROM b) AS j",
    "SELECT COUNT(*) FROM (SELECT a.k FROM a JOIN b ON a.k = b.k UNION ALL SELECT b.k FROM b)",  # no outer join: not this module's shape
    "SELECT COUNT(*) FROM (SELECT a.k FROM a LEFT JOIN b ON a.k = b.k)",
])
def test_declined(sql):
    assert rewritten(sql) == sqlglot.parse_one(sql, read="bigquery").sql(dialect="bigquery")


@pytest.mark.parametrize("on", ["a.k = b.k", "a.k = b.w", "a.k = b.k AND b.w = 2", "a.k = b.k AND a.v = 2", "a.k < b.k", "a.k != 3"])
def test_the_count_and_its_rewrite_are_proved(on):
    assert proved(f"SELECT COUNT(*) FROM a FULL JOIN b ON {on}", f"SELECT COUNT(*) FROM ({HALVES.format(on=on)})")


def test_the_tested_side_is_proved():
    assert proved("SELECT COUNT(*) FROM a FULL JOIN b ON a.k = b.k WHERE NOT b.k IS NULL", "SELECT COUNT(*) FROM a RIGHT OUTER JOIN b ON a.k = b.k WHERE NOT b.k IS NULL")
    assert proved("SELECT a.v, b.w FROM a FULL JOIN b ON a.k = b.k WHERE NOT a.v IS NULL", "SELECT a.v, b.w FROM a LEFT JOIN b ON a.k = b.k WHERE NOT a.v IS NULL")


@pytest.mark.parametrize("left, right", [
    # the unmatched rows of b are missing
    ("SELECT COUNT(*) FROM a FULL JOIN b ON a.k = b.k", "SELECT COUNT(*) FROM a LEFT JOIN b ON a.k = b.k"),
    # the second half without its filter counts the matched pairs twice
    ("SELECT COUNT(*) FROM a FULL JOIN b ON a.k = b.k", "SELECT COUNT(*) FROM (SELECT a.k FROM a LEFT JOIN b ON a.k = b.k UNION ALL SELECT a.k FROM a RIGHT JOIN b ON a.k = b.k)"),
    # a.v can be NULL on a matched pair, so the filter keeps more than the unmatched rows
    ("SELECT COUNT(*) FROM a FULL JOIN b ON a.k = b.k", "SELECT COUNT(*) FROM (SELECT a.k FROM a LEFT JOIN b ON a.k = b.k UNION ALL SELECT a.k FROM a RIGHT JOIN b ON a.k = b.k WHERE a.v IS NULL)"),
    # the join condition says nothing about a.k, which can be NULL on a matched pair
    ("SELECT COUNT(*) FROM a FULL JOIN b ON a.v = 3", "SELECT COUNT(*) FROM (SELECT a.k FROM a LEFT JOIN b ON a.v = 3 UNION ALL SELECT a.k FROM a RIGHT JOIN b ON a.v = 3 WHERE a.k IS NULL)"),
    # another join condition in one half
    ("SELECT COUNT(*) FROM a FULL JOIN b ON a.k = b.k", "SELECT COUNT(*) FROM (SELECT a.k FROM a LEFT JOIN b ON a.k = b.k UNION ALL SELECT a.k FROM a RIGHT JOIN b ON a.k = b.w WHERE a.k IS NULL)"),
    # UNION DISTINCT collapses the rows
    ("SELECT COUNT(*) FROM a FULL JOIN b ON a.k = b.k", "SELECT COUNT(*) FROM (SELECT a.k FROM a LEFT JOIN b ON a.k = b.k UNION DISTINCT SELECT a.k FROM a RIGHT JOIN b ON a.k = b.k WHERE a.k IS NULL)"),
    # a test of the other side keeps the other half
    ("SELECT COUNT(*) FROM a FULL JOIN b ON a.k = b.k WHERE NOT b.k IS NULL", "SELECT COUNT(*) FROM a LEFT JOIN b ON a.k = b.k"),
    ("SELECT a.v FROM a FULL JOIN b ON a.k = b.k WHERE b.w IS NULL", "SELECT a.v FROM a LEFT JOIN b ON a.k = b.k WHERE b.w IS NULL"),
])
def test_not_proved(left, right):
    assert not proved(left, right)


def test_the_rewrites_keep_the_rows_on_duckdb():
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    queries = [
        "SELECT a.v, b.w FROM a FULL JOIN b ON a.k = b.k WHERE NOT b.w IS NULL",
        "SELECT a.v, b.w FROM a FULL JOIN b ON a.k < b.k WHERE NOT a.v IS NULL AND b.k > 0",
        "SELECT a.k FROM a FULL JOIN b ON a.v = b.w WHERE NOT a.v IS NULL AND NOT b.k IS NULL",
        "SELECT COUNT(*) FROM a FULL JOIN b ON a.k = b.k",
        "SELECT COUNT(*) FROM a FULL JOIN b ON a.k = b.k AND b.w = 1",
        "SELECT COUNT(*) FROM a FULL JOIN b ON b.w > a.v AND a.k != 2",
        "SELECT COUNT(*) FROM a FULL JOIN (SELECT k, w FROM b WHERE w > 0) AS c ON 1 = a.v",
        f"SELECT COUNT(*) FROM ({HALVES.format(on='a.k = b.k')}) AS j",
        "SELECT COUNT(*) FROM (SELECT a.k / b.w, b.* FROM a LEFT JOIN b ON a.k = b.k UNION ALL SELECT 1, 2, 3 FROM b WHERE b.w IS NULL)",
    ]
    pairs = [(sqlglot.transpile(q, read="bigquery", write="duckdb")[0], sqlglot.transpile(rewritten(q), read="bigquery", write="duckdb")[0]) for q in queries]
    assert all(before != after for before, after in pairs)
    rng = random.Random(11)
    for _ in range(60):
        db = duckdb.connect()
        for name, columns in SCHEMA.items():
            db.execute(f"CREATE TABLE {name} ({', '.join(c + ' BIGINT' for c in columns)})")
            insert_rows(db, name, [tuple(rng.choice([None, 0, 1, 2, 3]) for _ in columns) for _ in range(rng.choice([0, 1, 2, 4, 6]))])
        for before, after in pairs:
            x, y = run_unoptimized(db, before, after)
            assert sorted(map(repr, x)) == sorted(map(repr, y)), (before, after)
        db.close()
