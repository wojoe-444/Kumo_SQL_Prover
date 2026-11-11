"""A filter on a value looked up by key is an ``EXISTS`` (``kumo_sql_prover.keyed_lookups``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.keyed_lookups import read_keyed_lookups
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: a is a key; b and c may be NULL and repeat. p: (a, b) is a key. u: no key.
SCHEMA = {"t": ["a", "b", "c"], "p": ["a", "b", "c"], "u": ["a", "c"]}
KEYS = {"t": [("a",)], "p": [("a", "b")]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}
CONSTRAINTS = {
    "t": TableConstraints(not_null=frozenset({"a"}), keys=(("a",),)),
    "p": TableConstraints(not_null=frozenset({"a", "b"}), keys=(("a", "b"),)),
}


def rewritten(sql, keys=KEYS):
    before = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    text = before.sql(dialect="mysql")
    after = read_keyed_lookups(before, SCHEMA, keys).sql(dialect="mysql")
    return after if after != text else None


def proved(left, right, constraints=CONSTRAINTS):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=constraints, types=TYPES, compare_names=False, dialect="mysql", exact_arithmetic=True).proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name in ("t", "p"):
        db.execute(f"CREATE TABLE {name} (a BIGINT, b BIGINT, c BIGINT)")
    db.execute("CREATE TABLE u (a BIGINT, c BIGINT)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


def test_a_tested_lookup_is_exists():
    assert (
        rewritten("SELECT x.a FROM t AS x WHERE (SELECT y.b FROM t AS y WHERE y.a = x.a + 1) = 1")
        == "SELECT x.a FROM t AS x WHERE EXISTS(SELECT 1 FROM t AS y WHERE y.a = x.a + 1 AND y.b = 1)"
    )
    # the mirror image keeps the operand order; a bare inner column is the looked-up table's
    assert (
        rewritten("SELECT x.a FROM t AS x WHERE x.c < (SELECT b FROM t AS y WHERE x.a - 1 = a)")
        == "SELECT x.a FROM t AS x WHERE EXISTS(SELECT 1 FROM t AS y WHERE x.a - 1 = a AND x.c < b)"
    )


def test_and_or_are_read_through_and_other_filters_stay():
    assert rewritten(
        "SELECT x.a FROM t AS x WHERE x.b = 1 AND ((SELECT y.b FROM t AS y WHERE y.a = x.a - 1) = 1 OR (SELECT y.b FROM t AS y WHERE y.a = x.a + 1 AND y.c > 0) >= x.c)"
    ) == (
        "SELECT x.a FROM t AS x WHERE x.b = 1 AND (EXISTS(SELECT 1 FROM t AS y WHERE y.a = x.a - 1 AND y.b = 1)"
        " OR EXISTS(SELECT 1 FROM t AS y WHERE y.a = x.a + 1 AND y.c > 0 AND y.b >= x.c))"
    )


def test_a_composite_key_needs_every_column():
    assert (
        rewritten("SELECT x.a FROM t AS x WHERE (SELECT y.c FROM p AS y WHERE y.a = x.a AND y.b = x.b) <> 3")
        == "SELECT x.a FROM t AS x WHERE EXISTS(SELECT 1 FROM p AS y WHERE y.a = x.a AND y.b = x.b AND y.c <> 3)"
    )
    assert rewritten("SELECT x.a FROM t AS x WHERE (SELECT y.c FROM p AS y WHERE y.a = x.a) <> 3") is None


def test_an_output_expression_and_a_join_condition():
    assert (
        rewritten("SELECT x.a, z.c FROM t AS x JOIN u AS z ON (SELECT y.b + y.c FROM t AS y WHERE y.a = z.a) = x.b")
        == "SELECT x.a, z.c FROM t AS x JOIN u AS z ON EXISTS(SELECT 1 FROM t AS y WHERE y.a = z.a AND (y.b + y.c) = x.b)"
    )


@pytest.mark.parametrize(
    "sql",
    [
        # FALSE and unknown differ under NOT, in an output, under IS and in CASE
        "SELECT x.a FROM t AS x WHERE NOT (SELECT y.b FROM t AS y WHERE y.a = x.a + 1) = 1",
        "SELECT x.a, (SELECT y.b FROM t AS y WHERE y.a = x.a + 1) = 1 AS f FROM t AS x",
        "SELECT x.a FROM t AS x WHERE ((SELECT y.b FROM t AS y WHERE y.a = x.a + 1) = 1) IS FALSE",
        "SELECT x.a FROM t AS x WHERE CASE WHEN (SELECT y.b FROM t AS y WHERE y.a = x.a + 1) = 1 THEN 0 ELSE 1 END = 1",
        # not a key, or not all of it, or not an equality, or not fixed from outside
        "SELECT x.a FROM t AS x WHERE (SELECT y.b FROM t AS y WHERE y.c = x.a) = 1",
        "SELECT x.a FROM t AS x WHERE (SELECT y.c FROM u AS y WHERE y.a = x.a) = 1",
        "SELECT x.a FROM t AS x WHERE (SELECT y.b FROM t AS y WHERE y.a > x.a) = 1",
        "SELECT x.a FROM t AS x WHERE (SELECT y.b FROM t AS y WHERE y.a = y.c) = 1",
        "SELECT x.a FROM t AS x WHERE (SELECT y.b FROM t AS y WHERE y.a = x.a OR y.c = 1) = 1",
        # shapes with their own meaning
        "SELECT x.a FROM t AS x WHERE (SELECT MAX(y.b) FROM t AS y WHERE y.a = x.a) = 1",
        "SELECT x.a FROM t AS x WHERE (SELECT y.b FROM t AS y JOIN u ON u.a = y.a WHERE y.a = x.a) = 1",
        "SELECT x.a FROM t AS x WHERE (SELECT DISTINCT y.b FROM t AS y WHERE y.a = x.a) = 1",
        "SELECT x.a FROM t AS x WHERE (SELECT y.b FROM t AS y WHERE y.a = x.a LIMIT 1) = 1",
        "SELECT x.a FROM t AS x WHERE (SELECT y.b FROM t AS y WHERE y.a = x.a GROUP BY y.b) = 1",
        # the other operand must still mean the outer column inside the subquery
        "SELECT x.a FROM t AS x WHERE (SELECT y.b FROM t AS y WHERE y.a = x.a) = c",
        "SELECT y.a FROM u AS y WHERE (SELECT y.b FROM t AS y WHERE y.a = 3) = y.c",
        "SELECT x.a FROM t AS x WHERE (SELECT y.b FROM t AS y WHERE y.a = x.a) = (SELECT MAX(u.c) FROM u)",
        "SELECT x.a FROM t AS x WHERE (SELECT y.b FROM t AS y WHERE y.a = x.a) IN (1, 2)",
        "SELECT x.a FROM t AS x WHERE (SELECT y.b FROM t AS y WHERE y.a = x.a) IS NULL",
    ],
)
def test_other_tests_stay(sql):
    assert rewritten(sql) is None


def test_without_a_declared_key_nothing_is_read():
    sql = "SELECT x.a FROM t AS x WHERE (SELECT y.b FROM t AS y WHERE y.a = x.a + 1) = 1"
    assert rewritten(sql, keys={}) is None
    assert rewritten(sql, keys={"t": [("a", "b")]}) is None


NEXT_FREE = "SELECT x.a FROM t AS x WHERE x.b = 1 AND ((SELECT y.b FROM t AS y WHERE y.a = x.a - 1) = 1 OR (SELECT y.b FROM t AS y WHERE y.a = x.a + 1) = 1)"

PROOFS = [
    (NEXT_FREE, "SELECT DISTINCT x.a FROM t AS x, t AS y WHERE x.b = 1 AND y.b = 1 AND (x.a = y.a + 1 OR x.a = y.a - 1)"),
    (NEXT_FREE, "SELECT x.a FROM t AS x WHERE x.b = 1 AND (EXISTS (SELECT 1 FROM t AS y WHERE y.b = 1 AND y.a = x.a - 1) OR EXISTS (SELECT 1 FROM t AS y WHERE y.b = 1 AND y.a = x.a + 1))"),
    (
        "SELECT x.a, x.c FROM u AS x WHERE (SELECT y.c FROM t AS y WHERE y.a = x.a) > x.c",
        "SELECT x.a, x.c FROM u AS x JOIN t AS y ON y.a = x.a WHERE y.c > x.c",
    ),
]

NOT_PROVED = [
    # only one neighbour
    (NEXT_FREE, "SELECT x.a FROM t AS x WHERE x.b = 1 AND EXISTS (SELECT 1 FROM t AS y WHERE y.b = 1 AND y.a = x.a + 1)"),
    # the looked-up value is not tested
    (NEXT_FREE, "SELECT x.a FROM t AS x WHERE x.b = 1 AND (EXISTS (SELECT 1 FROM t AS y WHERE y.a = x.a - 1) OR EXISTS (SELECT 1 FROM t AS y WHERE y.a = x.a + 1))"),
    # a missing row is unknown, which NOT keeps out and NOT EXISTS lets in
    (
        "SELECT x.a FROM u AS x WHERE NOT (SELECT y.c FROM t AS y WHERE y.a = x.a) = 1",
        "SELECT x.a FROM u AS x WHERE NOT EXISTS (SELECT 1 FROM t AS y WHERE y.a = x.a AND y.c = 1)",
    ),
    # a left join keeps the rows without a match
    (
        "SELECT x.a, x.c FROM u AS x WHERE (SELECT y.c FROM t AS y WHERE y.a = x.a) > x.c",
        "SELECT x.a, x.c FROM u AS x LEFT JOIN t AS y ON y.a = x.a AND y.c > x.c",
    ),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)


def test_without_the_key_nothing_is_proved():
    left, right = PROOFS[1]
    assert not proved(left, right, constraints={})


DIFFERENTIAL = [
    NEXT_FREE,
    "SELECT x.a, x.c FROM u AS x WHERE (SELECT y.c FROM t AS y WHERE y.a = x.a) > x.c OR x.c IS NULL",
    "SELECT x.a FROM t AS x WHERE x.c <= (SELECT y.b + 1 FROM t AS y WHERE x.a - 1 = y.a AND y.c IS NOT NULL)",
    "SELECT x.a, x.c FROM u AS x WHERE (SELECT y.c FROM p AS y WHERE y.a = x.a AND y.b = x.c) <> 2",
    "SELECT x.a, z.c FROM t AS x JOIN u AS z ON (SELECT y.b FROM t AS y WHERE y.a = z.a) = x.b OR z.c = x.c",
    "SELECT x.a, COUNT(*) FROM u AS x GROUP BY x.a HAVING (SELECT y.b FROM t AS y WHERE y.a = x.a) = 1 AND COUNT(*) > 0",
]


@pytest.mark.parametrize("sql", DIFFERENTIAL)
def test_the_rewritten_query_returns_the_same_rows(sql):
    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 3])
    for _ in range(150):
        keyed = {a: (a, value(), value()) for a in (rng.randint(0, 5) for _ in range(rng.randint(0, 5)))}
        pairs = {(a, b): (a, b, value()) for a, b in ((rng.randint(0, 3), rng.randint(0, 3)) for _ in range(rng.randint(0, 6)))}
        tables = {"t": list(keyed.values()), "p": list(pairs.values()), "u": [(value(), value()) for _ in range(rng.randint(0, 5))]}
        before_rows, after_rows = run(tables, sql, after)
        assert before_rows == after_rows, tables
