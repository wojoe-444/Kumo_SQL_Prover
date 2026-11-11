"""``LEFT JOIN (SELECT <expressions>) AS t ON TRUE`` is a cross join (``kumo_sql_prover.one_row_joins``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.one_row_joins import cross_join_one_row

SCHEMA = {"t": ["a", "b", "c"], "u": ["a", "b", "c"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}

SCALAR = "(SELECT MAX(u.b) FROM u WHERE u.a < 2)"
JOINED = f"SELECT t.a, o.v FROM t LEFT JOIN (SELECT {SCALAR} AS v) AS o ON TRUE"
DIRECT = f"SELECT t.a, {SCALAR} AS v FROM t"


def crossed(sql):
    out = cross_join_one_row(sqlglot.parse_one(sql, read="postgres"))
    return out.sql(dialect="postgres") if out is not None else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


def results(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name in SCHEMA:
        db.execute(f"CREATE TABLE {name} (a BIGINT, b BIGINT, c BIGINT)")
        insert_rows(db, name, rows.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def databases(count, seed):
    rng = random.Random(seed)
    value = lambda: rng.choice([None, 0, 1, 2, 3])
    yield {}
    yield {"t": [(1, 2, None)]}
    yield {"u": [(1, 5, 1)]}
    for _ in range(count):
        yield {name: [(value(), value(), value()) for _ in range(rng.randint(0, 5))] for name in SCHEMA}


def test_a_left_join_to_a_one_row_select_is_a_cross_join():
    out = crossed(JOINED)
    assert out is not None and "CROSS JOIN" in out and "LEFT JOIN" not in out and " ON " not in out
    assert proved(JOINED, DIRECT)
    assert proved("SELECT t.a, o.v, o.w FROM t LEFT JOIN (SELECT 5 AS v, 'x' AS w) AS o ON TRUE", "SELECT t.a, 5, 'x' FROM t")
    assert proved(f"SELECT t.a FROM t LEFT JOIN (SELECT {SCALAR} AS v) AS o ON (TRUE) WHERE o.v < t.b", f"SELECT t.a FROM t WHERE {SCALAR} < t.b")


def test_two_one_row_joins_are_both_rewritten():
    sql = f"SELECT t.a FROM t LEFT JOIN (SELECT {SCALAR} AS v) AS o ON TRUE LEFT JOIN (SELECT 3 AS w) AS p ON TRUE WHERE o.v < p.w OR t.b < 1"
    out = crossed(sql)
    assert out is not None and out.count("CROSS JOIN") == 2 and "LEFT JOIN" not in out
    assert proved(sql, f"SELECT t.a FROM t WHERE {SCALAR} < 3 OR t.b < 1")


def test_the_rewrite_returns_the_same_rows_on_duckdb():
    queries = [JOINED, crossed(JOINED), DIRECT]
    for rows in databases(40, seed=20261112):
        before, after, direct = results(queries, rows)
        assert before == after == direct, rows


NEAR_MISSES = [
    "SELECT t.a, o.v FROM t LEFT JOIN (SELECT 5 AS v WHERE 1 = 0) AS o ON TRUE",  # the row can be filtered away
    "SELECT t.a, o.v FROM t LEFT JOIN (SELECT 5 AS v LIMIT 0) AS o ON TRUE",
    "SELECT t.a, o.v FROM t LEFT JOIN (SELECT 5 AS v HAVING 1 = 0) AS o ON TRUE",
    "SELECT t.a, o.v FROM t LEFT JOIN (SELECT 5 AS v FROM u) AS o ON TRUE",  # any number of rows
    "SELECT t.a, o.v FROM t LEFT JOIN (SELECT u.b AS v FROM u LIMIT 1) AS o ON TRUE",  # none over an empty table
    "SELECT t.a, o.v FROM t LEFT JOIN (SELECT 5 AS v) AS o ON t.b > 3",  # a real condition pads
    "SELECT t.a, o.v FROM t LEFT JOIN (SELECT 5 AS v) AS o ON FALSE",
    "SELECT t.a, o.v FROM t LEFT JOIN (SELECT 5 AS v) AS o ON NULL",
    "SELECT o.v FROM t RIGHT JOIN (SELECT 5 AS v) AS o ON TRUE",  # keeps the row over an empty t
    "SELECT o.v FROM t FULL JOIN (SELECT 5 AS v) AS o ON TRUE",
    "SELECT t.a, o.v FROM t LEFT JOIN (SELECT UNNEST(ARRAY[1, 2]) AS v) AS o ON TRUE",  # a function that makes rows
    "SELECT t.a, o.v FROM t LEFT JOIN (SELECT GENERATE_SERIES(1, 0) AS v) AS o ON TRUE",
    "SELECT t.a, o.v FROM t LEFT JOIN (SELECT some_function(1) AS v) AS o ON TRUE",
    "SELECT t.a, o.v FROM t LEFT JOIN (SELECT 5 AS v UNION ALL SELECT 6) AS o ON TRUE",
    "SELECT t.a, o.v FROM t LEFT JOIN (SELECT COUNT(*) AS v) AS o ON TRUE",
    "SELECT t.a, o.v FROM t LEFT JOIN LATERAL (SELECT t.b AS v) AS o ON TRUE",
    "SELECT t.a, o.v FROM t LEFT JOIN u AS o(v, w, x) ON TRUE",
    "SELECT t.a, o.v FROM t CROSS JOIN (SELECT 5 AS v) AS o",  # already the cross join
    "SELECT t.a, o.v FROM t INNER JOIN (SELECT 5 AS v) AS o ON TRUE",
]


@pytest.mark.parametrize("sql", NEAR_MISSES)
def test_other_shapes_are_left_alone(sql):
    assert crossed(sql) is None


@pytest.mark.parametrize(
    "left, right",
    [
        # an empty joined select pads with NULL where the constant would read 5
        ("SELECT t.a, o.v FROM t LEFT JOIN (SELECT 5 AS v WHERE 1 = 0) AS o ON TRUE", "SELECT t.a, 5 FROM t"),
        ("SELECT t.a, o.v FROM t LEFT JOIN (SELECT 5 AS v) AS o ON t.b > 3", "SELECT t.a, 5 FROM t"),
        ("SELECT t.a, o.v FROM t LEFT JOIN (SELECT 5 AS v FROM u) AS o ON TRUE", "SELECT t.a, 5 FROM t"),
        ("SELECT o.v FROM t RIGHT JOIN (SELECT 5 AS v) AS o ON TRUE", "SELECT 5 FROM t"),
    ],
)
def test_pairs_that_differ_are_not_proved(left, right):
    assert not proved(left, right)
    differing = False
    for rows in databases(30, seed=7):
        a, b = results([left, right], rows)
        differing = differing or a != b
    assert differing  # the pair really differs on DuckDB, so a proof would be false


def test_the_rule_is_off_under_the_shape_fallback():
    assert proved(JOINED, DIRECT)
    with shape_fallback.disabled():
        assert not proved(JOINED, DIRECT)
