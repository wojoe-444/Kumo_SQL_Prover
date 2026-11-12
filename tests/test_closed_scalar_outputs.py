"""A derived table's output built from closed scalar subqueries is read where it is used (``kumo_sql_prover.closed_scalar_outputs``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.closed_scalar_outputs import inline_closed_outputs

SCHEMA = {"t": ["k", "x", "a"], "u": ["k", "w"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}

BOUND = "(SELECT SUM(u.w) FROM u) * 2"


def inlined(sql, dialect="postgres"):
    out = inline_closed_outputs(sqlglot.parse_one(sql, read=dialect), SCHEMA)
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


def test_the_output_is_read_where_it_is_used():
    assert (
        inlined(f"SELECT d.k, d.v FROM (SELECT t.k, SUM(t.x) AS v, {BOUND} AS bound FROM t GROUP BY t.k) AS d WHERE d.v > d.bound")
        == f"SELECT d.k, d.v FROM (SELECT t.k, SUM(t.x) AS v FROM t GROUP BY t.k) AS d WHERE d.v > ({BOUND})"
    )
    # bare reads of an only source, a read in the select list keeps its name, an inner join
    assert (
        inlined(f"SELECT k, bound FROM (SELECT t.k, {BOUND} AS bound FROM t) AS d WHERE x0 > bound".replace("x0", "k"))
        == f"SELECT k, ({BOUND}) AS bound FROM (SELECT t.k FROM t) AS d WHERE k > ({BOUND})"
    )
    assert (
        inlined(f"SELECT d.k, u.w + d.b AS s FROM u JOIN (SELECT t.k, (SELECT MAX(u.w) FROM u) AS b FROM t) AS d ON d.k = u.k")
        == "SELECT d.k, u.w + ((SELECT MAX(u.w) FROM u)) AS s FROM u JOIN (SELECT t.k FROM t) AS d ON d.k = u.k"
    )


@pytest.mark.parametrize("sql", [
    f"SELECT d.k FROM u LEFT JOIN (SELECT t.k, {BOUND} AS bound FROM t) AS d ON d.k = u.k WHERE u.w > d.bound",  # NULL on an unmatched row
    f"SELECT d.k FROM (SELECT t.k, {BOUND} + t.x AS bound FROM t) AS d WHERE d.k > d.bound",  # reads a column of its own row
    "SELECT d.k FROM (SELECT t.k, (SELECT SUM(u.w) FROM u WHERE u.k = t.k) AS bound FROM t) AS d WHERE d.k > d.bound",  # correlated
    "SELECT d.k FROM (SELECT t.k, 5 AS bound FROM t) AS d WHERE d.k > d.bound",  # no subquery: other rules read constants
    f"SELECT d.k FROM (SELECT t.k, {BOUND} * SUM(t.x) AS bound FROM t GROUP BY t.k) AS d WHERE d.k > d.bound",
    f"SELECT d.k FROM (SELECT DISTINCT t.k, {BOUND} AS bound FROM t) AS d WHERE d.k > d.bound",
    f"SELECT d.k FROM (SELECT t.k, {BOUND} AS bound FROM t LIMIT 3) AS d WHERE d.k > d.bound",
    f"SELECT d.bound FROM (SELECT {BOUND} AS bound FROM t) AS d",  # the only output
    f"SELECT d.k FROM (SELECT t.k, {BOUND} AS bound FROM t) AS d WHERE d.k > (SELECT MAX(u.w) FROM u WHERE u.w < d.bound)",  # read from a nested query
    f"SELECT d.k FROM (SELECT t.k, {BOUND} AS bound FROM t) AS d, u WHERE d.k > bound",  # a bare name among two sources
    f"SELECT d.k FROM (SELECT t.k, {BOUND} AS bound FROM t) AS d",  # never read
    f"SELECT d.k FROM (SELECT t.k, (SELECT SUM(u.w) FROM u) * RANDOM() AS bound FROM t) AS d WHERE d.k > d.bound",
    f"SELECT * FROM (SELECT t.k, {BOUND} AS bound FROM t) AS d WHERE d.k > d.bound",
])
def test_outputs_left_alone(sql):
    assert inlined(sql) is None


def test_proofs_and_near_misses():
    inside = f"SELECT d.k, d.v FROM (SELECT t.k, SUM(t.x) AS v, {BOUND} AS bound FROM t GROUP BY t.k) AS d WHERE d.v > d.bound"
    having = "SELECT t.k, SUM(t.x) FROM t GROUP BY t.k HAVING SUM(t.x) > (SELECT SUM(u.w) FROM u) * 2"
    assert proved(inside, having) and proved(having, inside)
    other = "SELECT t.k, SUM(t.x) FROM t GROUP BY t.k HAVING SUM(t.x) > (SELECT SUM(u.w) FROM u) * 3"
    first, second = rows_on_duckdb([inside, other], {"t": [(1, 5, 0)], "u": [(0, 2)]})
    assert first != second
    assert not proved(inside, other) and not proved(other, inside)
    # under an outer join the value is NULL on the unmatched rows
    outer = f"SELECT u.k FROM u LEFT JOIN (SELECT t.k, {BOUND} AS bound FROM t) AS d ON d.k = u.k WHERE d.bound IS NULL"
    plain = f"SELECT u.k FROM u LEFT JOIN (SELECT t.k FROM t) AS d ON d.k = u.k WHERE ({BOUND}) IS NULL"
    first, second = rows_on_duckdb([outer, plain], {"t": [], "u": [(0, 2)]})
    assert first != second
    assert not proved(outer, plain) and not proved(plain, outer)


@pytest.mark.parametrize("sql", [
    f"SELECT d.k, d.v, d.bound FROM (SELECT t.k, SUM(t.x) AS v, {BOUND} AS bound FROM t GROUP BY t.k) AS d WHERE d.v > d.bound OR d.bound IS NULL",
    "SELECT d.k, u.w + d.b AS s FROM u JOIN (SELECT t.k, (SELECT MAX(u.w) FROM u) AS b FROM t) AS d ON d.k = u.k AND d.b > u.k",
])
def test_the_rewrite_returns_the_same_rows(sql):
    after = inlined(sql)
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
