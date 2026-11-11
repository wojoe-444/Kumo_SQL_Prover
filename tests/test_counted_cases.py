"""In a grouped select ``COUNT(CASE WHEN c THEN 1 END)`` is ``SUM(CASE WHEN c THEN 1 ELSE 0 END)`` (``kumo_sql_prover.counted_cases``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.counted_cases import sum_counted_cases

SCHEMA = {"l": ["g", "q", "p"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}


def summed(sql, dialect="postgres"):
    out = sum_counted_cases(sqlglot.parse_one(sql, read=dialect))
    return out.sql(dialect=dialect) if out is not None else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


def rows_on_duckdb(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name, columns in SCHEMA.items():
        db.execute(f"CREATE TABLE {name} ({', '.join(f'{c} BIGINT' for c in columns)})")
        insert_rows(db, name, rows.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def test_the_count_becomes_a_sum():
    assert (
        summed("SELECT l.g, COUNT(CASE WHEN l.q = 1 THEN 1 END) FROM l GROUP BY l.g")
        == "SELECT l.g, SUM(CASE WHEN l.q = 1 THEN 1 ELSE 0 END) FROM l GROUP BY l.g"
    )
    assert (
        summed("SELECT g, COUNT(CASE WHEN q = 1 THEN 'x' WHEN p > 2 THEN 7 ELSE NULL END) + 1 FROM l GROUP BY g HAVING COUNT(CASE WHEN p IS NULL THEN 0 END) > 0")
        == "SELECT g, SUM(CASE WHEN q = 1 THEN 1 WHEN p > 2 THEN 1 ELSE 0 END) + 1 FROM l GROUP BY g HAVING SUM(CASE WHEN p IS NULL THEN 1 ELSE 0 END) > 0"
    )


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(CASE WHEN l.q = 1 THEN 1 END) FROM l",  # no rows: COUNT is 0, SUM is NULL
    "SELECT l.g, COUNT(CASE WHEN l.q = 1 THEN 1 END) FROM l GROUP BY ROLLUP (l.g)",  # the grand total over no rows
    "SELECT l.g, COUNT(CASE WHEN l.q = 1 THEN l.p END) FROM l GROUP BY l.g",  # the THEN may be NULL
    "SELECT l.g, COUNT(CASE WHEN l.q = 1 THEN 1 ELSE 2 END) FROM l GROUP BY l.g",  # counts every row
    "SELECT l.g, COUNT(CASE l.q WHEN 1 THEN 1 END) FROM l GROUP BY l.g",  # a simple CASE: left alone
    "SELECT l.g, COUNT(DISTINCT CASE WHEN l.q = 1 THEN 1 END) FROM l GROUP BY l.g",
    "SELECT l.g, COUNT(CASE WHEN l.q = 1 THEN 1 END) FILTER (WHERE l.p > 0) FROM l GROUP BY l.g",
    "SELECT l.g, COUNT(CASE WHEN l.q = 1 THEN 1 END) OVER (PARTITION BY l.g) FROM l GROUP BY l.g, l.q",
    "SELECT l.g, COUNT(l.q) FROM l GROUP BY l.g",
    "SELECT l.g, (SELECT COUNT(CASE WHEN x.q = 1 THEN 1 END) FROM l AS x) FROM l GROUP BY l.g",  # another select's count
])
def test_counts_left_alone(sql):
    assert summed(sql) is None


def test_proofs_and_near_misses():
    count = "SELECT l.g, COUNT(CASE WHEN l.q IN (1, 2) THEN 1 END) FROM l GROUP BY l.g"
    total = "SELECT l.g, SUM(CASE WHEN l.q = 1 OR l.q = 2 THEN 1 ELSE 0 END) FROM l GROUP BY l.g"
    assert proved(count, total) and proved(total, count)
    # without GROUP BY the two differ over no rows
    one, two = "SELECT COUNT(CASE WHEN l.q = 1 THEN 1 END) FROM l", "SELECT SUM(CASE WHEN l.q = 1 THEN 1 ELSE 0 END) FROM l"
    first, second = rows_on_duckdb([one, two], {})
    assert first != second
    assert not proved(one, two) and not proved(two, one)
    # another condition
    other = "SELECT l.g, SUM(CASE WHEN l.q = 1 THEN 1 ELSE 0 END) FROM l GROUP BY l.g"
    first, second = rows_on_duckdb([count, other], {"l": [(1, 2, 0)]})
    assert first != second
    assert not proved(count, other) and not proved(other, count)


def test_the_rewrite_returns_the_same_rows():
    sql = "SELECT g, COUNT(CASE WHEN q = 1 THEN 5 WHEN p > 2 THEN 7 ELSE NULL END) + 1 FROM l GROUP BY g HAVING COUNT(CASE WHEN p IS NULL THEN 0 END) >= 0"
    after = summed(sql)
    assert after is not None
    rng = random.Random(7)
    value = lambda: rng.choice([None, 0, 1, 1, 2, 3])
    for _ in range(40):
        rows = {"l": [tuple(value() for _ in range(3)) for _ in range(rng.randint(0, 6))]}
        first, second = rows_on_duckdb([sql, after], rows)
        assert first == second, rows
