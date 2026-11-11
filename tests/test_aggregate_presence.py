"""A one-row aggregate that only asks whether a row exists is ``EXISTS`` (``kumo_sql_prover.aggregate_presence``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.aggregate_presence import rewrite_aggregate_presence
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.smt_equivalence import TableConstraints

# t and u hold any rows: every column may be NULL and rows may repeat
SCHEMA = {"t": ["a", "b"], "u": ["a", "c"]}


def rewritten(sql):
    after = rewrite_aggregate_presence(canonical_negation(sqlglot.parse_one(sql, read="mysql")))
    return None if after is None else after.sql(dialect="mysql")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, compare_names=False, dialect="mysql").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE t (a BIGINT, b BIGINT)")
    db.execute("CREATE TABLE u (a BIGINT, c BIGINT)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


def test_not_null_is_exists():
    assert (
        rewritten("SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(TRUE) FROM u WHERE u.a = x.a) IS NULL")
        == "SELECT x.a FROM t AS x WHERE EXISTS(SELECT 1 FROM u WHERE u.a = x.a)"
    )
    assert (
        rewritten("SELECT x.a FROM t AS x WHERE (SELECT MAX(1) AS m FROM u WHERE u.a = x.a AND u.c > 3) IS NOT NULL AND x.b > 0")
        == "SELECT x.a FROM t AS x WHERE EXISTS(SELECT 1 FROM u WHERE u.a = x.a AND u.c > 3) AND x.b > 0"
    )


def test_null_is_not_exists():
    assert (
        rewritten("SELECT x.a FROM t AS x WHERE (SELECT MIN('y') FROM u WHERE u.a = x.a) IS NULL")
        == "SELECT x.a FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM u WHERE u.a = x.a)"
    )


@pytest.mark.parametrize(
    "test, expected",
    [
        ("(SELECT COUNT(*) FROM u WHERE u.a = x.a) = 0", "NOT EXISTS(SELECT 1 FROM u WHERE u.a = x.a)"),
        ("0 = (SELECT COUNT(*) FROM u WHERE u.a = x.a)", "NOT EXISTS(SELECT 1 FROM u WHERE u.a = x.a)"),
        ("(SELECT COUNT(1) FROM u WHERE u.a = x.a) < 1", "NOT EXISTS(SELECT 1 FROM u WHERE u.a = x.a)"),
        ("0 >= (SELECT COUNT(*) FROM u WHERE u.a = x.a)", "NOT EXISTS(SELECT 1 FROM u WHERE u.a = x.a)"),
        ("(SELECT COUNT(*) FROM u WHERE u.a = x.a) > 0", "EXISTS(SELECT 1 FROM u WHERE u.a = x.a)"),
        ("(SELECT COUNT(*) FROM u WHERE u.a = x.a) >= 1", "EXISTS(SELECT 1 FROM u WHERE u.a = x.a)"),
        ("1 <= (SELECT COUNT(*) FROM u WHERE u.a = x.a)", "EXISTS(SELECT 1 FROM u WHERE u.a = x.a)"),
        ("(SELECT COUNT(*) FROM u WHERE u.a = x.a) <> 0", "EXISTS(SELECT 1 FROM u WHERE u.a = x.a)"),
        ("0 < (SELECT COUNT(*) FROM u)", "EXISTS(SELECT 1 FROM u)"),
        # COUNT of a column counts the rows where it is not NULL
        ("(SELECT COUNT(u.c) FROM u WHERE u.a = x.a) = 0", "NOT EXISTS(SELECT 1 FROM u WHERE u.a = x.a AND NOT u.c IS NULL)"),
        ("(SELECT COUNT(DISTINCT u.c) FROM u WHERE u.a = x.a OR u.c = 1) > 0", "EXISTS(SELECT 1 FROM u WHERE (u.a = x.a OR u.c = 1) AND NOT u.c IS NULL)"),
        ("(SELECT COUNT(c) FROM u) > 0", "EXISTS(SELECT 1 FROM u WHERE NOT c IS NULL)"),
    ],
)
def test_a_count_compared_with_zero_is_an_existence_test(test, expected):
    assert rewritten(f"SELECT x.a FROM t AS x WHERE {test}") == f"SELECT x.a FROM t AS x WHERE {expected}"


def test_joins_inside_the_subquery_are_kept():
    assert (
        rewritten("SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(TRUE) FROM u JOIN t AS y ON y.a = u.a WHERE u.c = x.b) IS NULL")
        == "SELECT x.a FROM t AS x WHERE EXISTS(SELECT 1 FROM u JOIN t AS y ON y.a = u.a WHERE u.c = x.b)"
    )


def test_an_output_and_a_join_condition_are_read_too():
    assert rewritten("SELECT (SELECT MIN(TRUE) FROM u WHERE u.a = x.a) IS NULL AS none FROM t AS x") == "SELECT NOT EXISTS(SELECT 1 FROM u WHERE u.a = x.a) AS none FROM t AS x"


def test_only_the_selects_own_tests_are_rewritten():
    # the nested select has its own turn in the rule loop
    assert rewritten("SELECT d.a FROM (SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(TRUE) FROM u WHERE u.a = x.a) IS NULL) AS d") is None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(u.c) FROM u WHERE u.a = x.a) IS NULL",  # NULL although rows exist
        "SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(NULL) FROM u WHERE u.a = x.a) IS NULL",
        "SELECT x.a FROM t AS x WHERE NOT (SELECT COUNT(1) FROM u WHERE u.a = x.a) IS NULL",  # never NULL
        "SELECT x.a FROM t AS x WHERE NOT (SELECT SUM(1) FROM u WHERE u.a = x.a) IS NULL",
        "SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(TRUE) FROM u WHERE u.a = x.a GROUP BY u.c) IS NULL",  # no row, or several
        "SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(TRUE) FROM u WHERE u.a = x.a HAVING COUNT(*) > 1) IS NULL",
        "SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(TRUE) FROM u WHERE u.a = x.a LIMIT 0) IS NULL",
        "SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(TRUE) FROM u WHERE u.a = x.a LIMIT 1 OFFSET 1) IS NULL",
        "SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(DISTINCT TRUE) FROM u WHERE u.a = x.a) IS NULL",
        "SELECT x.a FROM t AS x WHERE NOT (SELECT DISTINCT MIN(TRUE) FROM u WHERE u.a = x.a) IS NULL",
        "SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(TRUE) OVER () FROM u WHERE u.a = x.a) IS NULL",
        "SELECT x.a FROM t AS x WHERE NOT (SELECT TRUE FROM u WHERE u.a = x.a) IS NULL",  # no aggregate: a row per row of u
        "SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(TRUE)) IS NULL",  # no FROM
        "SELECT x.a FROM t AS x WHERE (SELECT MIN(TRUE) FROM u WHERE u.a = x.a) = TRUE",  # unknown, not FALSE, without a row
        "SELECT x.a FROM t AS x WHERE (SELECT MIN(TRUE) FROM u WHERE u.a = x.a)",
        "SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(TRUE) + MAX(u.c) FROM u WHERE u.a = x.a) IS NULL",
        "SELECT x.a FROM t AS x WHERE NOT x.b IS NULL",
        # counts that ask for more than "any row"
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM u WHERE u.a = x.a) = 1",
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM u WHERE u.a = x.a) > 1",
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM u WHERE u.a = x.a) <= 1",
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM u WHERE u.a = x.a) >= 0",
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM u WHERE u.a = x.a) < 0",
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM u WHERE u.a = x.a) = '0'",
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM u WHERE u.a = x.a) = x.b",
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM u WHERE u.a = x.a GROUP BY u.c) = 0",
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM u WHERE u.a = x.a HAVING MAX(u.c) > 1) = 0",
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM u WHERE u.a = x.a LIMIT 0) = 0",
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(u.c + 1) FROM u WHERE u.a = x.a) = 0",
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(DISTINCT u.a, u.c) FROM u WHERE u.a = x.a) = 0",
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(*) + 1 FROM u WHERE u.a = x.a) > 0",
        "SELECT x.a FROM t AS x WHERE (SELECT SUM(1) FROM u WHERE u.a = x.a) > 0",
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM u WHERE u.a = x.a) + 1 > 0",
        "SELECT x.a FROM t AS x WHERE x.b > 0",
    ],
)
def test_other_tests_stay(sql):
    assert rewritten(sql) is None


PROOFS = [
    (
        "SELECT x.a, x.b FROM t AS x WHERE NOT (SELECT MIN(TRUE) FROM u WHERE x.a = 2 * u.a + 30) IS NULL",
        "SELECT x.a, x.b FROM t AS x WHERE EXISTS (SELECT 1 FROM u WHERE x.a = 2 * u.a + 30)",
    ),
    (
        "SELECT x.a, x.b FROM t AS x WHERE (SELECT MAX(1) FROM u WHERE u.a = x.a AND u.c > 5) IS NULL",
        "SELECT x.a, x.b FROM t AS x WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.a = x.a AND u.c > 5)",
    ),
    (
        # the derived table's computed column, as Calcite pulls it out
        "SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(TRUE) FROM (SELECT u.a, u.c + 30 AS f FROM u) AS d WHERE x.b >= d.f) IS NULL",
        "SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(TRUE) FROM u WHERE x.b >= u.c + 30) IS NULL",
    ),
]

PROOFS += [
    (
        "SELECT x.a, x.b FROM t AS x WHERE 0 = (SELECT COUNT(*) FROM u JOIN t AS y ON y.a = u.c WHERE u.a = x.a AND y.b = 3)",
        "SELECT x.a, x.b FROM t AS x WHERE NOT EXISTS (SELECT 1 FROM u JOIN t AS y ON y.a = u.c WHERE u.a = x.a AND y.b = 3)",
    ),
    (
        # the anti-join spelling of "none"
        "SELECT x.a, x.b FROM t AS x WHERE (SELECT COUNT(*) FROM u WHERE u.a = x.a AND u.c = 3) = 0",
        "SELECT x.a, x.b FROM t AS x LEFT JOIN (SELECT u.a FROM u WHERE u.c = 3) AS d ON d.a = x.a WHERE d.a IS NULL",
    ),
    (
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(u.c) FROM u WHERE u.a = x.a) > 0",
        "SELECT x.a FROM t AS x WHERE EXISTS (SELECT 1 FROM u WHERE u.a = x.a AND u.c IS NOT NULL)",
    ),
]

NOT_PROVED = [
    # NOT IN is empty when the subquery holds a NULL; the count is not
    (
        "SELECT x.a FROM t AS x WHERE 0 = (SELECT COUNT(*) FROM u WHERE u.a = x.a)",
        "SELECT x.a FROM t AS x WHERE x.a NOT IN (SELECT u.a FROM u)",
    ),
    # COUNT(c) skips NULLs, COUNT(*) does not
    (
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(u.c) FROM u WHERE u.a = x.a) > 0",
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM u WHERE u.a = x.a) > 0",
    ),
    # at least one is not exactly one
    (
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM u WHERE u.a = x.a) = 1",
        "SELECT x.a FROM t AS x WHERE EXISTS (SELECT 1 FROM u WHERE u.a = x.a)",
    ),
    (
        "SELECT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM u WHERE u.a = x.a) = 0",
        "SELECT x.a FROM t AS x WHERE EXISTS (SELECT 1 FROM u WHERE u.a = x.a)",
    ),
    # a NULL value is NULL although the row exists
    (
        "SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(u.c) FROM u WHERE u.a = x.a) IS NULL",
        "SELECT x.a FROM t AS x WHERE EXISTS (SELECT 1 FROM u WHERE u.a = x.a)",
    ),
    # the other polarity
    (
        "SELECT x.a FROM t AS x WHERE (SELECT MIN(TRUE) FROM u WHERE u.a = x.a) IS NULL",
        "SELECT x.a FROM t AS x WHERE EXISTS (SELECT 1 FROM u WHERE u.a = x.a)",
    ),
    # another filter
    (
        "SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(TRUE) FROM u WHERE u.a = x.a) IS NULL",
        "SELECT x.a FROM t AS x WHERE EXISTS (SELECT 1 FROM u WHERE u.a = x.a + 1)",
    ),
    # two laterals one level down that differ only inside
    (
        "SELECT d.a FROM (SELECT x.a, l.m FROM t AS x, LATERAL (SELECT MIN(TRUE) AS m FROM u WHERE u.c = x.b) AS l) AS d WHERE NOT d.m IS NULL",
        "SELECT d.a FROM (SELECT x.a, l.m FROM t AS x, LATERAL (SELECT MIN(TRUE) AS m FROM u WHERE u.c = x.b + 1) AS l) AS d WHERE NOT d.m IS NULL",
    ),
    # a join multiplies rows where the test does not
    (
        "SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(TRUE) FROM u WHERE u.a = x.a) IS NULL",
        "SELECT x.a FROM t AS x JOIN u ON u.a = x.a",
    ),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)


def test_a_lateral_output_keeps_its_name():
    # the aggregate read as a bare output of a derived table is still named m for the select around it
    inner = "SELECT x.a, l.m FROM t AS x, LATERAL (SELECT MAX(u.c) AS m FROM u WHERE u.a = x.a) AS l"
    assert " AS m FROM t AS x" in normalize(inner, schema=SCHEMA, dialect="mysql")
    text = normalize(f"SELECT d.a FROM ({inner}) AS d WHERE NOT d.m IS NULL", schema=SCHEMA, dialect="mysql")
    assert "MAX(" in text or "d.m" not in text  # the test still reads the aggregate
    # with a key on u.a the lateral one level down is the lateral itself (the shape Calcite writes)
    keyed = {"u": TableConstraints(not_null=frozenset({"a"}), keys=(("a",),))}
    left = "SELECT x.b FROM t AS x, LATERAL (SELECT MIN(TRUE) AS m FROM u WHERE x.a = u.a) AS l WHERE NOT l.m IS NULL"
    right = "SELECT d.b FROM (SELECT x.b, l.m FROM t AS x, LATERAL (SELECT MIN(TRUE) AS m FROM u WHERE x.a = u.a) AS l) AS d WHERE NOT d.m IS NULL"
    other = right.replace("x.a = u.a", "x.a = u.a + 1")
    assert prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=keyed, compare_names=False, dialect="mysql").proven
    assert not prove_equivalent_algebraic(left, other, schema=SCHEMA, constraints=keyed, compare_names=False, dialect="mysql").proven


DIFFERENTIAL = [
    "SELECT x.a, x.b FROM t AS x WHERE NOT (SELECT MIN(TRUE) FROM u WHERE u.a = x.a) IS NULL",
    "SELECT x.a, x.b FROM t AS x WHERE (SELECT MAX(7) FROM u WHERE u.a = x.a AND u.c > 1) IS NULL",
    "SELECT x.a, (SELECT MIN('k') FROM u WHERE u.c = x.b) IS NULL AS none FROM t AS x",
    "SELECT x.a FROM t AS x WHERE NOT (SELECT MIN(FALSE) FROM u LEFT JOIN t AS y ON y.a = u.c WHERE u.a = x.a) IS NULL OR x.b IS NULL",
    "SELECT x.a, y.a FROM t AS x LEFT JOIN u AS y ON y.a = x.a AND (SELECT MIN(TRUE) FROM u AS z WHERE z.c = y.c + 1) IS NULL",
    "SELECT x.a, x.b FROM t AS x WHERE 0 = (SELECT COUNT(*) FROM u WHERE u.a = x.a)",
    "SELECT x.a, x.b FROM t AS x WHERE (SELECT COUNT(u.c) FROM u WHERE u.a = x.a OR u.c = x.b) >= 1 OR x.b IS NULL",
    "SELECT x.a, (SELECT COUNT(DISTINCT u.c) FROM u WHERE u.a = x.a) < 1 AS none, 1 <= (SELECT COUNT(1) FROM u WHERE u.c > x.b) AS some FROM t AS x",
    "SELECT x.a, y.c FROM t AS x LEFT JOIN u AS y ON y.a = x.a AND (SELECT COUNT(*) FROM t AS z WHERE z.b = y.c) <> 0",
]


@pytest.mark.parametrize("sql", DIFFERENTIAL)
def test_the_rewritten_query_returns_the_same_rows(sql):
    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 3])
    for _ in range(150):
        tables = {
            "t": [(value(), value()) for _ in range(rng.randint(0, 5))],
            "u": [(value(), value()) for _ in range(rng.randint(0, 5))],
        }
        before_rows, after_rows = run(tables, sql, after)
        assert before_rows == after_rows, tables
