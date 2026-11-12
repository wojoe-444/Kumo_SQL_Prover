"""``ABS(x) = n`` says ``x`` is ``n`` or ``-n`` (``kumo_sql_prover.absolute_values``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.absolute_values import read_absolute_values
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"t": ["a", "b", "s"]}
TYPES = {"t": {"a": "INT", "b": "INT", "s": "VARCHAR"}}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a"}), keys=(("a",),))}


def rewritten(sql, types=TYPES):
    before = sqlglot.parse_one(sql, read="mysql")
    text = before.sql(dialect="mysql")
    after = read_absolute_values(before, types).sql(dialect="mysql")
    return after if after != text else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=CONSTRAINTS, types=TYPES, compare_names=False, dialect="mysql", exact_arithmetic=True).proven


def test_abs_equal_to_a_number():
    assert rewritten("SELECT x.a FROM t AS x JOIN t AS y ON ABS(x.a - y.a) = 1") == "SELECT x.a FROM t AS x JOIN t AS y ON ((x.a - y.a) = 1 OR (x.a - y.a) = -1)"
    assert rewritten("SELECT x.a FROM t AS x WHERE 2 = ABS(x.b) AND x.a > 0") == "SELECT x.a FROM t AS x WHERE (x.b = 2 OR x.b = -2) AND x.a > 0"
    assert rewritten("SELECT ABS(x.b * 2 + 1) = 3 AS f FROM t AS x") == "SELECT ((x.b * 2 + 1) = 3 OR (x.b * 2 + 1) = -3) AS f FROM t AS x"
    assert rewritten("SELECT x.a FROM t AS x WHERE ABS(x.b) = 0") is None  # zero is the solver's own reading
    assert rewritten("SELECT x.a FROM t AS x WHERE NOT ABS(x.b) = 1.5", types=None) == "SELECT x.a FROM t AS x WHERE NOT (x.b = 1.5 OR x.b = -1.5)"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT x.a FROM t AS x WHERE ABS(x.b) < 2",
        "SELECT x.a FROM t AS x WHERE ABS(x.b) <> 2",
        "SELECT x.a FROM t AS x WHERE ABS(x.b) = x.a",
        "SELECT x.a FROM t AS x WHERE ABS(x.b) = -2",
        "SELECT x.a FROM t AS x WHERE ABS(x.b) = '2'",
        "SELECT x.a FROM t AS x WHERE ABS(x.s) = 2",  # a string is converted by the engine
        "SELECT x.a FROM t AS x WHERE ABS(x.b / 2) = 1",
        "SELECT x.a FROM t AS x WHERE ABS(ROUND(x.b)) = 1",
        "SELECT x.a FROM t AS x WHERE ABS((SELECT MAX(y.b) FROM t AS y)) = 1",
        "SELECT x.a FROM t AS x WHERE ABS(x.b) + 1 = 2",
        "SELECT x.a FROM t AS x WHERE x.b = 2",
    ],
)
def test_other_tests_stay(sql):
    assert rewritten(sql) is None


def test_a_negative_number_is_never_matched():
    # ABS(x) = -2 is never TRUE; x = -2 OR x = 2 would be
    assert not proved("SELECT x.a FROM t AS x WHERE ABS(x.b) = -2", "SELECT x.a FROM t AS x WHERE x.b = -2 OR x.b = 2")


PROOFS = [
    (
        "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON ABS(x.a - y.a) = 1 WHERE x.b = 1 AND y.b = 1",
        "SELECT DISTINCT x.a FROM t AS x, t AS y WHERE x.b = 1 AND y.b = 1 AND (x.a = y.a + 1 OR x.a = y.a - 1)",
    ),
    ("SELECT x.a FROM t AS x WHERE ABS(x.b) = 3", "SELECT x.a FROM t AS x WHERE x.b IN (3, -3)"),
]

NOT_PROVED = [
    ("SELECT x.a FROM t AS x WHERE ABS(x.b) = 3", "SELECT x.a FROM t AS x WHERE x.b = 3"),
    ("SELECT x.a FROM t AS x WHERE ABS(x.b) = 3", "SELECT x.a FROM t AS x WHERE x.b = 3 OR x.b = -3 OR x.b IS NULL"),
    (
        "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON ABS(x.a - y.a) = 1",
        "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a + 1",
    ),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT x.a, y.a FROM t AS x JOIN t AS y ON ABS(x.a - y.a) = 1",
        "SELECT x.a, ABS(x.b) = 2 AS f, NOT ABS(x.b - x.a) = 0 AS g FROM t AS x",
        "SELECT x.a FROM t AS x WHERE NOT ABS(x.b * 2 - 1) = 3 OR x.b IS NULL",
    ],
)
def test_the_rewritten_query_returns_the_same_rows(sql):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    for _ in range(150):
        db = duckdb.connect()
        db.execute("CREATE TABLE t (a BIGINT, b BIGINT, s VARCHAR)")
        insert_rows(db, "t", [(rng.randint(-3, 3), rng.choice([None, -3, -2, -1, 0, 1, 2, 3]), None) for _ in range(rng.randint(0, 6))])
        before_rows, after_rows = [sorted(map(repr, r)) for r in run_unoptimized(db, *[sqlglot.transpile(q, read="mysql", write="duckdb")[0] for q in (sql, after)])]
        assert before_rows == after_rows
