"""``(k, v) IN (SELECT k, MIN(v) .. GROUP BY k)`` read as a comparison with the group's aggregate (``kumo_sql_prover.aggregate_membership``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.aggregate_membership import read_aggregate_membership
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation

SCHEMA = {"t": ["a", "b", "c"], "u": ["a", "d"]}


def rewritten(sql, dialect="mysql", schema=SCHEMA):
    before = canonical_negation(sqlglot.parse_one(sql, read=dialect))
    text = before.sql(dialect=dialect)
    after = read_aggregate_membership(before, schema, dialect).sql(dialect=dialect)
    return after if after != text else None


def proved(left, right, **options):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, compare_names=False, dialect="mysql", **options).proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE t (a BIGINT, b BIGINT, c BIGINT)")
    db.execute("CREATE TABLE u (a BIGINT, d BIGINT)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


def test_the_grouped_membership_becomes_a_comparison():
    assert rewritten("SELECT a, c FROM t WHERE (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a)") == (
        "SELECT a, c FROM t WHERE t.b = (SELECT MIN(kqm0.b) FROM t AS kqm0 WHERE kqm0.a = t.a)"
    )
    # the subquery's own filter comes along, a second aggregate gets its own comparison, the key may come last
    assert rewritten("SELECT x.c FROM t AS x WHERE x.c > 0 AND (x.b, c + 1, x.a) IN (SELECT MAX(b), SUM(y.c), a FROM t AS y WHERE y.c > 1 OR b = 2 GROUP BY y.a)") == (
        "SELECT x.c FROM t AS x WHERE x.c > 0 AND (x.b = (SELECT MAX(kqm0.b) FROM t AS kqm0 WHERE (kqm0.c > 1 OR kqm0.b = 2) AND kqm0.a = x.a) "
        "AND (x.c + 1) = (SELECT SUM(kqm1.c) FROM t AS kqm1 WHERE (kqm1.c > 1 OR kqm1.b = 2) AND kqm1.a = x.a))"
    )
    # two keys, and the left side read from the one source of a join that has the column
    assert rewritten("SELECT d FROM u JOIN t ON u.a = t.a WHERE (d, b, c) IN (SELECT a, b, AVG(c) FROM t GROUP BY a, b)") == (
        "SELECT d FROM u JOIN t ON u.a = t.a WHERE t.c = (SELECT AVG(kqm0.c) FROM t AS kqm0 WHERE kqm0.a = u.d AND kqm0.b = t.b)"
    )


def test_what_membership_cannot_see_is_read_through():
    plain = "SELECT a, c FROM t WHERE t.b = (SELECT MIN(kqm0.b) FROM t AS kqm0 WHERE kqm0.a = t.a)"
    assert rewritten("SELECT a, c FROM t WHERE (a, b) IN (SELECT DISTINCT a, MIN(b) FROM t GROUP BY a)") == plain
    assert rewritten("SELECT a, c FROM t WHERE (a, b) IN (SELECT a, MIN(b) AS m FROM t GROUP BY a ORDER BY a DESC)") == plain
    assert rewritten("SELECT a, c FROM t WHERE (a, b) IN (SELECT * FROM (SELECT a, MIN(b) AS m FROM t GROUP BY a) AS g)") == plain
    assert rewritten("WITH g AS (SELECT a, MIN(b) AS m FROM t GROUP BY a) SELECT a, c FROM t WHERE (b, a) IN (SELECT m, g.a FROM g)") is None  # WITH is inlined before the rule runs
    assert rewritten("SELECT a, c FROM t WHERE (b, a) IN (SELECT m, g.a FROM (SELECT a, MIN(b) AS m FROM t GROUP BY a) AS g)") == plain


def test_an_output_alias_that_grouping_does_not_read_is_fine():
    assert rewritten("SELECT a, c FROM t WHERE (a, b) IN (SELECT a, MIN(b) AS b FROM t GROUP BY a)") == (
        "SELECT a, c FROM t WHERE t.b = (SELECT MIN(kqm0.b) FROM t AS kqm0 WHERE kqm0.a = t.a)"
    )


def test_a_subquery_of_one_row_compares_each_value():
    assert rewritten("SELECT a FROM t WHERE b IN (SELECT MAX(b) FROM t)") == "SELECT a FROM t WHERE t.b = (SELECT MAX(kqm0.b) FROM t AS kqm0)"
    assert rewritten("SELECT a FROM t WHERE (b, c) IN (SELECT MIN(b), COUNT(*) + 1 FROM t WHERE a > 0)") == (
        "SELECT a FROM t WHERE (t.b = (SELECT MIN(kqm0.b) FROM t AS kqm0 WHERE kqm0.a > 0) AND t.c = (SELECT COUNT(*) + 1 FROM t AS kqm1 WHERE kqm1.a > 0))"
    )


def test_a_fresh_name_avoids_the_names_in_the_statement():
    assert rewritten("SELECT kqm0.a FROM t AS kqm0 WHERE (kqm0.a, kqm0.b) IN (SELECT a, MIN(b) FROM t GROUP BY a)") == (
        "SELECT kqm0.a FROM t AS kqm0 WHERE kqm0.b = (SELECT MIN(kqm1.b) FROM t AS kqm1 WHERE kqm1.a = kqm0.a)"
    )


@pytest.mark.parametrize("sql", [
    "SELECT a FROM t WHERE (a, b) IN (SELECT a, COUNT(b) FROM t GROUP BY a)",  # COUNT is 0, not NULL, for a key with no group
    "SELECT a FROM t WHERE (a, b) IN (SELECT a, COALESCE(MIN(b), 0) FROM t GROUP BY a)",  # nor is an expression around an aggregate
    "SELECT a FROM t WHERE (a, b) IN (SELECT a, MIN(b) + 1 FROM t GROUP BY a)",
    "SELECT a FROM t WHERE b IN (SELECT MIN(b) FROM t GROUP BY a)",  # the key is not output: any group may match
    "SELECT a FROM t WHERE (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a, c)",  # c is not output
    "SELECT a FROM t WHERE (a, b) NOT IN (SELECT a, MIN(b) FROM t GROUP BY a)",  # FALSE and NULL differ under NOT
    "SELECT a FROM t WHERE NOT ((a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a))",
    "SELECT a FROM t WHERE c = 1 OR (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a)",  # not an AND part
    "SELECT a, (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a) FROM t",  # a value, not a filter
    "SELECT a FROM t GROUP BY a HAVING (a, MAX(b)) IN (SELECT a, MIN(b) FROM t GROUP BY a)",
    "SELECT a FROM t WHERE (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a HAVING COUNT(*) > 1)",
    "SELECT a FROM t WHERE (a, b) IN (SELECT a AS b, MIN(b) FROM t GROUP BY b)",  # GROUP BY b: the column or the alias
    "SELECT a FROM t WHERE (a, b) IN (SELECT g.a, g.m FROM (SELECT a, MIN(b) AS m FROM t GROUP BY a) AS g WHERE g.m > 0)",  # the wrapper filters
    "SELECT a FROM t WHERE b IN (SELECT g.m FROM (SELECT a, MIN(b) AS m FROM t GROUP BY a) AS g)",  # the wrapper drops the key
    "SELECT a FROM t WHERE (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a ORDER BY a LIMIT 2)",
    "SELECT a FROM t WHERE (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a LIMIT 1)",
    "SELECT a FROM t WHERE (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a WITH ROLLUP)",
    "SELECT a FROM t WHERE (a, b) IN (SELECT t.a, MIN(d) FROM t JOIN u ON t.a = u.a GROUP BY t.a)",  # a join inside
    "SELECT a FROM t WHERE (a, b) IN (SELECT a, MIN(b) FROM (SELECT * FROM t) AS s GROUP BY a)",  # not a base table
    "SELECT a FROM t WHERE (a, b) IN (SELECT a, MIN(b) FROM w GROUP BY a)",  # columns not supplied
    "SELECT a FROM t WHERE (a, b) IN (SELECT a, MIN(b + t.c) FROM u GROUP BY a)",  # reads the enclosing query
    "SELECT a FROM t WHERE (a, b) IN (SELECT a + 1, MIN(b) FROM t GROUP BY a + 1)",  # the key is not a column
    "SELECT a FROM t WHERE (a, b) IN (SELECT a, MIN(DISTINCT b) FROM t GROUP BY a)",
    "SELECT a FROM t WHERE (a, (SELECT MAX(d) FROM u)) IN (SELECT a, MIN(b) FROM t GROUP BY a)",  # a subquery on the left
    "SELECT t.a FROM t JOIN u ON t.a = u.a WHERE (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a)",  # a is t's or u's
    "SELECT a FROM t, (SELECT d FROM u) AS s WHERE (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a)",  # a source whose columns are not supplied
    "SELECT a FROM t WHERE (a, b) IN (SELECT a, MIN(b) FROM t)",  # a bare column next to an aggregate, no GROUP BY
    "SELECT a FROM t WHERE (a, b) IN ((1, 2), (3, 4))",
])
def test_other_tests_stay(sql):
    assert rewritten(sql) is None


PROOFS = [
    ("WITH g AS (SELECT a, MIN(b) AS m FROM t GROUP BY a) SELECT a, c FROM t WHERE (a, b) IN (SELECT a, m FROM g)",
     "SELECT x.a, x.c FROM t AS x JOIN (SELECT a, MIN(b) AS m FROM t GROUP BY a) AS g ON x.a = g.a AND x.b = g.m"),
    ("SELECT a, c FROM t WHERE (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a)",
     "SELECT x.a, x.c FROM t AS x JOIN (SELECT a, MIN(b) AS m FROM t GROUP BY a) AS g ON x.a = g.a AND x.b = g.m"),
    ("SELECT a, c FROM t WHERE (a, b) IN (SELECT a, MAX(b) FROM t WHERE c > 0 GROUP BY a)",
     "SELECT x.a, x.c FROM t AS x WHERE x.b = (SELECT MAX(y.b) FROM t AS y WHERE y.c > 0 AND y.a = x.a)"),
    ("SELECT a FROM t WHERE b IN (SELECT MAX(b) FROM t)", "SELECT a FROM t WHERE b = (SELECT MAX(b) FROM t)"),
    ("SELECT u.d FROM u WHERE (u.a, u.d) IN (SELECT a, MIN(b) FROM t GROUP BY a)",
     "SELECT u.d FROM u JOIN (SELECT a, MIN(b) AS m FROM t GROUP BY a) AS g ON u.a = g.a AND u.d = g.m"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # the first row of each group against the last
    ("SELECT a, c FROM t WHERE (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a)", "SELECT a, c FROM t WHERE (a, b) IN (SELECT a, MAX(b) FROM t GROUP BY a)"),
    # the group's rows against every row with that value in any group
    ("SELECT a, c FROM t WHERE (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a)", "SELECT a, c FROM t WHERE b IN (SELECT MIN(b) FROM t GROUP BY a)"),
    # a join repeats nothing here, but a filter inside the subquery is not a filter outside
    ("SELECT a, c FROM t WHERE (a, b) IN (SELECT a, MIN(b) FROM t WHERE c > 0 GROUP BY a)", "SELECT a, c FROM t WHERE c > 0 AND (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a)"),
    # NOT IN keeps the rows of the NULL group's complement differently from a negated comparison
    ("SELECT a, c FROM t WHERE (a, b) NOT IN (SELECT a, MIN(b) FROM t GROUP BY a)", "SELECT a, c FROM t WHERE b <> (SELECT MIN(y.b) FROM t AS y WHERE y.a = t.a)"),
    # COUNT of a key with no group is 0
    ("SELECT u.d FROM u WHERE (u.a, u.d) IN (SELECT a, COUNT(b) FROM t GROUP BY a)", "SELECT u.d FROM u WHERE u.d = (SELECT COUNT(y.b) FROM t AS y WHERE y.a = u.a)"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_the_count_pair_really_differs():
    left, right = NOT_PROVED[-1]
    a, b = run({"t": [(1, 1, 1)], "u": [(2, 0)]}, left, right)
    assert a == [] and b == ["(0,)"]


QUERIES = [
    "SELECT a, c FROM t WHERE (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a)",
    "SELECT a, c FROM t WHERE (a, b) IN (SELECT DISTINCT a, MAX(b) FROM t GROUP BY a ORDER BY a)",
    "SELECT a, c FROM t WHERE (b, a) IN (SELECT m, g.a FROM (SELECT a, SUM(b) AS m FROM t WHERE c > 0 GROUP BY a) AS g)",
    "SELECT a, c FROM t WHERE (a, b) IN (SELECT * FROM (SELECT DISTINCT a, MIN(b) FROM t GROUP BY a) AS g)",
    "SELECT a, c FROM t WHERE c > 0 AND (a, b) IN (SELECT a, MAX(b) FROM t WHERE c > 1 OR b = 2 GROUP BY a)",
    "SELECT x.c FROM t AS x WHERE (x.b, c + 1, x.a) IN (SELECT MAX(b), SUM(y.c), a FROM t AS y GROUP BY y.a)",
    "SELECT d FROM u JOIN t ON u.a = t.a WHERE (d, b, c) IN (SELECT a, b, AVG(c) FROM t GROUP BY a, b)",
    "SELECT u.d FROM u WHERE (u.a, u.d) IN (SELECT a, SUM(b) FROM t GROUP BY a)",
    "SELECT a FROM t WHERE b IN (SELECT MAX(b) FROM t)",
    "SELECT a FROM t WHERE (b, c) IN (SELECT MIN(b), COUNT(*) + 1 FROM t WHERE a > 0)",
    "SELECT u.d FROM u WHERE (u.a, u.d) IN (SELECT COUNT(a), COUNT(DISTINCT b) FROM t)",
    "SELECT a FROM t WHERE (a, a, b) IN (SELECT a, a, MIN(b) FROM t GROUP BY a)",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the rewritten one on random small tables with NULLs, repeats and empty tables."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 2, 3])  # noqa: E731
    for _ in range(150):
        tables = {
            "t": [(value(), value(), value()) for _ in range(rng.choice([0, 1, 2, 4, 6]))],
            "u": [(value(), value()) for _ in range(rng.choice([0, 1, 3, 5]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
