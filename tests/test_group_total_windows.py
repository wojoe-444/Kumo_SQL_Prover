"""``SUM(SUM(e)) OVER ()`` of a grouped select is the ungrouped sum (``kumo_sql_prover.group_total_windows``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.group_total_windows import read_group_totals

SCHEMA = {"t": ["k", "x", "a"], "u": ["k", "w"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}


def read(sql, dialect="postgres"):
    before = sqlglot.parse_one(sql, read=dialect)
    text = before.sql(dialect=dialect)
    after = read_group_totals(before).sql(dialect=dialect)
    return None if after == text else after


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


def test_the_window_becomes_a_scalar_subquery():
    assert (
        read("SELECT t.k, SUM(t.x) AS v, SUM(SUM(t.x)) OVER () AS total FROM t WHERE t.a > 0 GROUP BY t.k")
        == "SELECT t.k, SUM(t.x) AS v, (SELECT SUM(t.x) FROM t WHERE t.a > 0) AS total FROM t WHERE t.a > 0 GROUP BY t.k"
    )
    assert (
        read("SELECT d.k FROM (SELECT t.k, SUM(x * w) AS v, SUM(SUM(x * w)) OVER () * 2 AS bound FROM t JOIN u ON t.k = u.k GROUP BY t.k) AS d WHERE d.v > d.bound")
        == "SELECT d.k FROM (SELECT t.k, SUM(x * w) AS v, (SELECT SUM(x * w) FROM t JOIN u ON t.k = u.k) * 2 AS bound FROM t JOIN u ON t.k = u.k GROUP BY t.k) AS d WHERE d.v > d.bound"
    )


@pytest.mark.parametrize("sql", [
    "SELECT t.k, SUM(SUM(t.x)) OVER (PARTITION BY t.k) FROM t GROUP BY t.k",
    "SELECT t.k, SUM(SUM(t.x)) OVER (ORDER BY t.k) FROM t GROUP BY t.k",  # a running total
    "SELECT t.k, SUM(SUM(t.x)) OVER (ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t GROUP BY t.k",
    "SELECT t.k, SUM(SUM(t.x)) OVER w FROM t GROUP BY t.k WINDOW w AS ()",
    "SELECT t.k, SUM(SUM(t.x)) OVER () FROM t GROUP BY t.k HAVING SUM(t.x) > 0",  # the window sees the kept groups only
    "SELECT t.k, SUM(COUNT(t.x)) OVER () FROM t GROUP BY t.k",
    "SELECT t.k, MAX(SUM(t.x)) OVER () FROM t GROUP BY t.k",
    "SELECT t.k, SUM(SUM(DISTINCT t.x)) OVER () FROM t GROUP BY t.k",  # a value in two groups is summed twice
    "SELECT t.k, SUM(t.x) OVER () FROM t",  # not grouped
    "SELECT SUM(SUM(t.x)) OVER () FROM t",  # no GROUP BY: one row, left to the other rules
    "SELECT t.k, SUM(SUM(t.x)) OVER () FROM t GROUP BY ROLLUP (t.k)",  # the grand total row is summed too
    "SELECT DISTINCT SUM(SUM(t.x)) OVER () FROM t GROUP BY t.k",
    "SELECT t.k, SUM(SUM(t.x)) OVER () FROM t WHERE t.a IN (SELECT u.w FROM u) GROUP BY t.k",
    "SELECT t.k, SUM(t.x) FROM t GROUP BY t.k ORDER BY SUM(SUM(t.x)) OVER ()",
    "SELECT t.k, SUM(SUM(t.x)) OVER () FROM t WHERE RANDOM() < 0.5 GROUP BY t.k",
])
def test_windows_left_alone(sql):
    assert read(sql) is None


def test_proofs_and_near_misses():
    window = "SELECT d.k, d.v FROM (SELECT t.k, SUM(t.x) AS v, SUM(SUM(t.x)) OVER () * 2 AS bound FROM t WHERE t.a > 0 GROUP BY t.k) AS d WHERE d.v > d.bound"
    scalar = "SELECT t.k, SUM(t.x) FROM t WHERE t.a > 0 GROUP BY t.k HAVING SUM(t.x) > (SELECT SUM(t.x) * 2 FROM t WHERE t.a > 0)"
    assert proved(window, scalar) and proved(scalar, window)

    # the second scan without the filter is another total
    unfiltered = "SELECT t.k, SUM(t.x) FROM t WHERE t.a > 0 GROUP BY t.k HAVING SUM(t.x) > (SELECT SUM(t.x) * 2 FROM t)"
    first, second = rows_on_duckdb([window, unfiltered], {"t": [(1, -3, 1), (2, 5, 0)]})
    assert first != second
    assert not proved(window, unfiltered) and not proved(unfiltered, window)
    # with HAVING before the window, the total is of the kept groups
    kept = "SELECT t.k, SUM(SUM(t.x)) OVER () FROM t GROUP BY t.k HAVING SUM(t.x) > 0"
    whole = "SELECT t.k, (SELECT SUM(t.x) FROM t) FROM t GROUP BY t.k HAVING SUM(t.x) > 0"
    first, second = rows_on_duckdb([kept, whole], {"t": [(1, 5, 0), (2, -3, 0)]})
    assert first != second
    assert not proved(kept, whole) and not proved(whole, kept)


@pytest.mark.parametrize("sql", [
    "SELECT t.k, SUM(t.x) AS v, SUM(SUM(t.x)) OVER () AS total FROM t WHERE t.a > 0 GROUP BY t.k",
    "SELECT t.k, t.a, SUM(SUM(t.x * u.w)) OVER () - SUM(t.x) FROM t JOIN u ON t.k = u.k GROUP BY t.k, t.a",
])
def test_the_rewrite_returns_the_same_rows(sql):
    after = read(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 2, 3])
    for _ in range(40):
        rows = {
            "t": [(value(), value(), value()) for _ in range(rng.randint(0, 6))],
            "u": [(value(), value()) for _ in range(rng.randint(0, 4))],
        }
        first, second = rows_on_duckdb([sql, after], rows)
        assert first == second, rows
