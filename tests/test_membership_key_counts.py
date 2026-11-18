"""In the HAVING of a grouped membership subquery, COUNT of a compared group key is COUNT(*) (``kumo_sql_prover.membership_key_counts``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.membership_key_counts import read_membership_key_counts
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: (a, b) is a key and both are NOT NULL; c and d may be NULL and repeat. u has no key: its rows repeat and hold NULLs.
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a", "b"}), keys=(("a", "b"),))}


def rewritten(sql, schema=SCHEMA):
    tree = sqlglot.parse_one(sql, read="mysql")
    text = tree.sql(dialect="mysql")
    after = read_membership_key_counts(tree, schema).sql(dialect="mysql")
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


def random_tables(rng):
    value = lambda: rng.choice([None, None, 1, 1, 2, 3])  # noqa: E731
    keys = {(rng.choice([1, 1, 2, 3]), rng.choice([0, 1, 2, 3, 4])) for _ in range(rng.choice([0, 1, 2, 4, 7]))}
    return {
        "t": [(a, b, value(), value()) for a, b in sorted(keys)],
        "u": [(value(), value()) for _ in range(rng.choice([0, 1, 3, 6]))],
    }



def test_a_count_of_a_compared_key_becomes_a_row_count():
    assert rewritten("SELECT a FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1)") == (
        "SELECT a FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1)"
    )
    # any HAVING condition, even one that zero passes: the NULL group never matches whichever way it is decided
    assert rewritten("SELECT a FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(c) = 0)") == (
        "SELECT a FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) = 0)"
    )
    # x.c and the bare c are one column of the only source by the schema
    assert rewritten("SELECT o.a FROM t AS o WHERE (o.c, o.d) IN (SELECT x.c, d FROM t AS x GROUP BY c, x.d HAVING COUNT(x.d) = 1 AND COUNT(c) < 3) AND o.b > 0") == (
        "SELECT o.a FROM t AS o WHERE (o.c, o.d) IN (SELECT x.c, d FROM t AS x GROUP BY c, x.d HAVING COUNT(*) = 1 AND COUNT(*) < 3) AND o.b > 0"
    )
    assert rewritten("SELECT o.a FROM t AS o WHERE o.c IN (SELECT x.c FROM t AS x GROUP BY c HAVING COUNT(x.c) > 1)", schema=None) is None
    # a join in the subquery: only the count of the compared key changes
    assert rewritten("SELECT a FROM t WHERE c IN (SELECT DISTINCT t.c AS k FROM t JOIN u ON u.a = t.a GROUP BY t.c HAVING COUNT(t.c) + COUNT(u.e) > 1)") == (
        "SELECT a FROM t WHERE c IN (SELECT DISTINCT t.c AS k FROM t JOIN u ON u.a = t.a GROUP BY t.c HAVING COUNT(*) + COUNT(u.e) > 1)"
    )


@pytest.mark.parametrize("sql", [
    # NULL and FALSE differ here
    "SELECT a FROM t WHERE c NOT IN (SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1)",
    "SELECT a FROM t WHERE NOT c IN (SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1)",
    "SELECT a FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1) OR b = 1",
    "SELECT a FROM t WHERE (c IN (SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1)) IS NULL",
    "SELECT a, c IN (SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1) FROM t",
    "SELECT a FROM t GROUP BY a, c HAVING c IN (SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1)",
    # the counted column is not both grouped and returned
    "SELECT a FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(d) > 1)",
    "SELECT a FROM t WHERE c IN (SELECT c FROM t GROUP BY c, d HAVING COUNT(d) > 1)",  # the groups with a NULL d may match
    "SELECT a FROM t WHERE c IN (SELECT MAX(c) FROM t GROUP BY c HAVING COUNT(c) > 1)",
    "SELECT a FROM t WHERE c IN (SELECT c + 0 FROM t GROUP BY c HAVING COUNT(c) > 1)",
    "SELECT a FROM t WHERE c IN (SELECT x.c FROM t AS x JOIN t AS y ON x.a = y.a GROUP BY x.c HAVING COUNT(y.c) > 1)",
    "SELECT a FROM t WHERE c IN (SELECT d AS c FROM t GROUP BY c HAVING COUNT(c) > 1)",  # c: the column or the alias
    # another count
    "SELECT a FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(DISTINCT c) > 0)",
    "SELECT a FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(c + 0) > 1)",
    "SELECT a FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING SUM(c) > 1)",
    "SELECT a FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING (SELECT COUNT(c) FROM u) > 1)",
    "SELECT a FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1)",
    # more than a grouped select
    "SELECT a FROM t WHERE c IN (SELECT c FROM t GROUP BY c WITH ROLLUP HAVING COUNT(c) > 1)",  # the total row: a NULL key over non-NULL rows
    "SELECT a FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1 LIMIT 3)",
    "SELECT a FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1 ORDER BY c)",
    # not a membership test
    "SELECT a FROM t WHERE EXISTS (SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1)",
    "SELECT a FROM t WHERE c = (SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1)",
    "SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1",
])
def test_other_counts_stay(sql):
    assert rewritten(sql) is None


PROOFS = [
    ("SELECT a, b FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1)", "SELECT a, b FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1)"),
    (
        "SELECT SUM(b) FROM t WHERE (c, d) IN (SELECT c, d FROM t GROUP BY c, d HAVING COUNT(d) = 1) AND c IN (SELECT c FROM t GROUP BY c HAVING COUNT(c) >= 2)",
        "SELECT SUM(b) FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1) AND (c, d) IN (SELECT c, d FROM t GROUP BY c, d HAVING COUNT(*) = 1)",
    ),
    # through the join rule: the join with the groups is the membership test first
    (
        "SELECT x.a, x.b FROM t AS x JOIN (SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1) AS g ON g.c = x.c",
        "SELECT a, b FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1)",
    ),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # NOT IN: the NULL group, returned by one side only, makes every test NULL
    ("SELECT a, b FROM t WHERE c NOT IN (SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1)", "SELECT a, b FROM t WHERE c NOT IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1)"),
    # COUNT(d) in the groups of c: the NULLs of d sit in groups that match
    ("SELECT a, b FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(d) > 1)", "SELECT a, b FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1)"),
    # the grouped select on its own returns the NULL group on one side
    ("SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1", "SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1"),
    # in the select list NULL and FALSE differ
    ("SELECT a, b, c IN (SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1) FROM t", "SELECT a, b, c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1) FROM t"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_the_refused_counts_do_differ():
    tables = {"t": [(1, 1, None, 1), (1, 2, None, 2), (2, 1, 7, None), (2, 2, 7, None), (3, 1, 8, 1)]}
    for left, right in NOT_PROVED:
        a, b = run(tables, left, right)
        assert a != b, (left, right)


CASES = [
    "SELECT a, b FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(c) > 1)",
    "SELECT a, b FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(c) = 0)",
    "SELECT a, b FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(c) < 2 OR MAX(d) IS NULL)",
    "SELECT o.a, o.b FROM t AS o WHERE (o.c, o.d) IN (SELECT x.c, d FROM t AS x GROUP BY c, x.d HAVING COUNT(x.d) = 1 AND COUNT(c) < 3) AND o.b > 0",
    "SELECT a, b FROM t WHERE c IN (SELECT DISTINCT t.c AS k FROM t JOIN u ON u.a = t.a GROUP BY t.c HAVING COUNT(t.c) + COUNT(u.e) > 1)",
    "SELECT a, e FROM u WHERE (e, a) IN (SELECT t.c, u.a FROM t LEFT JOIN u ON u.e = t.d GROUP BY u.a, t.c HAVING COUNT(u.a) <> COUNT(t.c) OR COUNT(u.a) = 2)",
    "SELECT COUNT(*), SUM(b) FROM t WHERE d IN (SELECT d FROM t WHERE b > 0 GROUP BY d HAVING NOT COUNT(d) > 1) AND c IN (SELECT e FROM u GROUP BY e HAVING COUNT(e) >= 1)",
]


@pytest.mark.parametrize("sql", CASES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """COUNT(key) and COUNT(*) in the subquery's HAVING on random tables whose keys hold NULLs and repeat."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    for _ in range(150):
        tables = random_tables(rng)
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
