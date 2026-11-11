"""A lower bound on a correlated ``COUNT(*)`` is a membership test in the groups that pass it (``kumo_sql_prover.correlated_count_tests``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.correlated_count_tests import read_correlated_count_tests
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: (a, b) is a key and both are NOT NULL; c and d may be NULL and repeat. u has no key: its rows repeat and hold NULLs.
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a", "b"}), keys=(("a", "b"),))}


def rewritten(sql, types=None, schema=SCHEMA, dialect="mysql"):
    tree = sqlglot.parse_one(sql, read="mysql")
    text = tree.sql(dialect="mysql")
    after = read_correlated_count_tests(tree, schema, types, dialect).sql(dialect="mysql")
    return after if after != text else None


def proved(left, right, constraints=CONSTRAINTS):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=constraints, compare_names=False, dialect="mysql").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE t (a BIGINT, b BIGINT, c BIGINT, d BIGINT)")
    db.execute("CREATE TABLE u (a BIGINT, e BIGINT)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


def test_one_tie_becomes_a_grouped_membership_test():
    assert rewritten("SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c) > 1") == (
        "SELECT o.a FROM t AS o WHERE o.c IN (SELECT x.c FROM t AS x GROUP BY x.c HAVING COUNT(*) > 1)"
    )
    # the literal first, the tie the other way round, a count of the tied column, a GROUP BY of the tied column
    assert rewritten("SELECT o.a FROM t AS o WHERE 1 < (SELECT COUNT(x.c) FROM t AS x WHERE o.c = x.c GROUP BY x.c)") == (
        "SELECT o.a FROM t AS o WHERE o.c IN (SELECT x.c FROM t AS x GROUP BY x.c HAVING COUNT(*) > 1)"
    )
    # nested: the outer row is the row of the select whose WHERE holds the test
    assert rewritten("SELECT o.a FROM t AS o WHERE o.a IN (SELECT y.a FROM t AS y WHERE (SELECT COUNT(*) FROM u WHERE u.a = y.c) > 0)") == (
        "SELECT o.a FROM t AS o WHERE o.a IN (SELECT y.a FROM t AS y WHERE y.c IN (SELECT u.a FROM u GROUP BY u.a HAVING COUNT(*) > 0))"
    )


def test_several_ties_become_a_row_membership_test():
    # the conditions on x alone stay in the grouped select; the bare e is u's (the innermost select binds it)
    assert rewritten("SELECT o.a FROM t AS o WHERE (SELECT COUNT(1) FROM u WHERE e = o.c AND u.a = o.d AND u.e > 0) >= 2") == (
        "SELECT o.a FROM t AS o WHERE (o.c, o.d) IN (SELECT e, u.a FROM u WHERE u.e > 0 GROUP BY e, u.a HAVING COUNT(*) >= 2)"
    )
    assert rewritten("SELECT t.a FROM t JOIN u ON u.a = t.a WHERE (SELECT COUNT(*) FROM t AS x WHERE x.d = u.e AND x.c = t.c AND x.a <> x.b) = 3") == (
        "SELECT t.a FROM t JOIN u ON u.a = t.a WHERE (u.e, t.c) IN (SELECT x.d, x.c FROM t AS x WHERE x.a <> x.b GROUP BY x.d, x.c HAVING COUNT(*) = 3)"
    )


def test_a_row_equality_is_read_where_it_is_a_conjunction():
    sql = "SELECT o.a FROM t AS o WHERE 1 = (SELECT COUNT(*) FROM t AS x WHERE (o.c, o.d) = (x.c, x.d)) AND o.b > 0"
    after = "SELECT o.a FROM t AS o WHERE (o.c, o.d) IN (SELECT x.c, x.d FROM t AS x GROUP BY x.c, x.d HAVING COUNT(*) = 1) AND o.b > 0"
    assert rewritten(sql) == after
    assert rewritten(sql, dialect="bigquery") == after
    # DuckDB finds (1, NULL) = (1, NULL): a row with a NULL key counts its partners there
    assert rewritten(sql, dialect="duckdb") is None
    assert rewritten(sql, dialect=None) is None
    assert rewritten("SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE (x.c, x.d) = (o.c, 1)) > 1") is None
    assert rewritten("SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE (x.c, x.d) = (o.c)) > 1") is None


@pytest.mark.parametrize("sql", [
    # bounds that zero passes: the rows without a partner pass them and are in no group
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c) >= 0",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c) = 0",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c) < 2",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c) <= 2",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c) <> 1",
    "SELECT o.a FROM t AS o WHERE 0 <= (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c)",
    "SELECT o.a FROM t AS o WHERE 3 > (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c)",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c) > -1",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c) > o.b",
    # not a top-level conjunct of WHERE
    "SELECT o.a FROM t AS o WHERE NOT (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c) > 1",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c) > 1 OR o.b = 1",
    "SELECT o.a, (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c) > 1 FROM t AS o",
    "SELECT o.a FROM t AS o GROUP BY o.a, o.c HAVING (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c) > 1",
    "SELECT o.a FROM t AS o JOIN u ON (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c) > 1",
    # the correlation is not a plain column equality
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c > o.c) > 1",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c + 1) > 1",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c AND x.b <> o.b) > 1",  # the other rows of the group
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c OR x.d = o.d) > 1",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = 1) > 1",  # no tie at all
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = c) > 1",  # the bare c is x's own
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = z.c) > 1",  # bound further out, or nowhere
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS o WHERE o.c = o.d) > 1",  # the inner source hides the outer one
    # another count, another aggregate
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(x.d) FROM t AS x WHERE x.c = o.c) > 1",  # skips the NULLs of d
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(DISTINCT x.c) FROM t AS x WHERE x.c = o.c) > 0",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(NULL) FROM t AS x WHERE x.c = o.c) > 0",
    "SELECT o.a FROM t AS o WHERE (SELECT SUM(x.b) FROM t AS x WHERE x.c = o.c) > 1",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) + 1 FROM t AS x WHERE x.c = o.c) > 1",
    # more than a filtered base table
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x JOIN u ON u.a = x.a WHERE x.c = o.c) > 1",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM (SELECT c FROM t) AS x WHERE x.c = o.c) > 1",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c GROUP BY x.d) > 1",  # one count per d
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c HAVING COUNT(*) > 0) > 1",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c LIMIT 1) > 1",
    "SELECT o.a FROM t AS o WHERE (SELECT DISTINCT COUNT(*) FROM t AS x WHERE x.c = o.c) > 1",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c AND x.d IN (SELECT e FROM u)) > 1",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c AND RAND() > 0.5) > 1",
    "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM w AS x WHERE x.c = o.c) > 1",  # a table the schema does not list
    "SELECT o.a FROM t AS o WHERE o.c IN (SELECT x.c FROM t AS x GROUP BY x.c HAVING COUNT(*) > 1)",  # already the membership test
])
def test_other_tests_stay(sql):
    assert rewritten(sql) is None


def test_tied_columns_need_one_type():
    sql = "SELECT o.a FROM t AS o WHERE (SELECT COUNT(*) FROM u AS x WHERE x.e = o.c) > 1"
    assert rewritten(sql) is not None  # neither declared
    assert rewritten(sql, types={"t": {"c": "INT"}, "u": {"e": "INT"}}) is not None
    assert rewritten(sql, types={"t": {"c": "INT"}, "u": {"e": "VARCHAR"}}) is None  # 1 = '1' and 1 = '01': one o.c, two groups
    assert rewritten(sql, types={"t": {"c": "INT"}}) is None
    assert rewritten(sql, types={"t": {"c": "DOUBLE"}, "u": {"e": "DOUBLE"}}) is None
    assert rewritten(sql, schema=None) is None


REFERENCE = "SELECT o.a, o.b FROM t AS o WHERE o.c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1) AND (o.c, o.d) IN (SELECT c, d FROM t GROUP BY c, d HAVING COUNT(*) = 1)"

PROOFS = [
    (
        "SELECT o.a, o.b FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c) > 1 AND (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c AND x.d = o.d) = 1",
        REFERENCE,
    ),
    (
        "SELECT o.a, o.b FROM t AS o WHERE 1 < (SELECT COUNT(1) FROM t AS x WHERE o.c = x.c) AND 1 = (SELECT COUNT(x.c) FROM t AS x WHERE (x.c, x.d) = (o.c, o.d))",
        REFERENCE,
    ),
    (
        "SELECT SUM(o.b) FROM t AS o WHERE (SELECT COUNT(*) FROM u WHERE u.a = o.c AND u.e > 0) >= 2",
        "SELECT SUM(o.b) FROM t AS o WHERE o.c IN (SELECT a FROM u WHERE e > 0 GROUP BY a HAVING COUNT(*) >= 2)",
    ),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # a count of 0: the rows with a NULL c, which are in no group, pass it
    (
        "SELECT o.a, o.b FROM t AS o WHERE (SELECT COUNT(*) FROM u WHERE u.a = o.c) = 0",
        "SELECT o.a, o.b FROM t AS o WHERE o.c IN (SELECT a FROM u GROUP BY a HAVING COUNT(*) = 0)",
    ),
    # an upper bound: so do the rows without a partner
    (
        "SELECT o.a, o.b FROM t AS o WHERE (SELECT COUNT(*) FROM u WHERE u.a = o.c) < 2",
        "SELECT o.a, o.b FROM t AS o WHERE o.c IN (SELECT a FROM u GROUP BY a HAVING COUNT(*) < 2)",
    ),
    # COUNT(d) skips the NULLs of d, COUNT(*) does not
    (
        "SELECT o.a, o.b FROM t AS o WHERE (SELECT COUNT(x.d) FROM t AS x WHERE x.c = o.c) > 1",
        "SELECT o.a, o.b FROM t AS o WHERE o.c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1)",
    ),
    # the other rows of the group: one fewer than the group's count
    (
        "SELECT o.a, o.b FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c AND x.b <> o.b) >= 1",
        "SELECT o.a, o.b FROM t AS o WHERE o.c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) >= 1)",
    ),
    # another bound
    (
        "SELECT o.a, o.b FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c) > 1",
        "SELECT o.a, o.b FROM t AS o WHERE o.c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) >= 1)",
    ),
    # another tie
    (
        "SELECT o.a, o.b FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.d) > 1",
        "SELECT o.a, o.b FROM t AS o WHERE o.c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1)",
    ),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_the_refused_bounds_do_differ():
    """The near misses are real: one NULL key, or one row without a partner, tells the two spellings apart."""

    tables = {"t": [(1, 1, None, 5), (1, 2, 7, 5)], "u": [(7, 1), (7, 2), (7, 3)]}
    for left, right in NOT_PROVED[:2]:
        a, b = run(tables, left, right)
        assert a != b, (left, right)
    a, b = run({"t": [(1, 1, 7, None), (1, 2, 7, None), (1, 3, 7, 1)]}, *NOT_PROVED[2])
    assert a != b
    for left, right in NOT_PROVED[3:5]:
        a, b = run({"t": [(1, 1, 7, None)]}, left, right)
        assert a != b, (left, right)


CASES = [
    "SELECT o.a, o.b FROM t AS o WHERE (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c) > 1",
    "SELECT o.a, o.b FROM t AS o WHERE 1 = (SELECT COUNT(*) FROM t AS x WHERE x.c = o.c AND x.d = o.d) AND o.b > 0",
    "SELECT o.a, o.b FROM t AS o WHERE (SELECT COUNT(1) FROM u WHERE e = o.c AND u.a = o.d AND u.e > 0) >= 2",
    "SELECT o.a, o.b FROM t AS o WHERE 1 < (SELECT COUNT(x.c) FROM t AS x WHERE o.c = x.c GROUP BY x.c)",
    "SELECT o.a, o.b FROM t AS o WHERE (SELECT COUNT(x.d) FROM t AS x WHERE o.c = x.c AND x.d = o.d GROUP BY x.d, x.c) = 2",
    "SELECT t.a, t.b, u.e FROM t JOIN u ON u.a = t.a WHERE (SELECT COUNT(*) FROM t AS x WHERE x.d = u.e AND x.c = t.c AND x.a <> x.b) >= 1",
    "SELECT o.a, o.b FROM t AS o WHERE o.d IN (SELECT y.e FROM u AS y WHERE (SELECT COUNT(*) FROM u WHERE u.a = y.a) > 1)",
    "SELECT o.a, o.b FROM t AS o WHERE (SELECT COUNT(*) FROM u AS x WHERE x.e = o.c AND x.a IS NULL) > 0 AND (SELECT COUNT(*) FROM t AS x WHERE x.d = o.d) = 3",
    "SELECT COUNT(*), SUM(o.b) FROM t AS o WHERE (SELECT COUNT(*) FROM u AS x WHERE x.a = o.a AND x.e = o.d) = 1",
]


@pytest.mark.parametrize("sql", CASES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The count test and the membership test on random tables whose tied columns hold NULLs and repeat.

    The ties are written as column equalities: DuckDB, which runs the comparison, finds two NULLs of a row
    equality equal, and the rule reads that spelling only for the dialects that do not.
    """

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, None, 1, 1, 2, 3])  # noqa: E731
    for _ in range(150):
        keys = {(rng.choice([1, 1, 2, 3]), rng.choice([0, 1, 2, 3, 4])) for _ in range(rng.choice([0, 1, 2, 4, 7]))}
        tables = {
            "t": [(a, b, value(), value()) for a, b in sorted(keys)],
            "u": [(value(), value()) for _ in range(rng.choice([0, 1, 3, 6]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
