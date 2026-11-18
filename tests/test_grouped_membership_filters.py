"""A test on group keys that holds a subquery runs before the grouping (``kumo_sql_prover.grouped_membership_filters``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.grouped_membership_filters import grouped_membership_filters
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: (a, b) is a key and both are NOT NULL; c and d may be NULL and repeat. u has no key: its rows repeat and hold NULLs.
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a", "b"}), keys=(("a", "b"),))}


def rewritten(sql, schema=SCHEMA):
    tree = sqlglot.parse_one(sql, read="mysql")
    text = tree.sql(dialect="mysql")
    grouped_membership_filters(tree, schema)
    after = tree.sql(dialect="mysql")
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


def test_a_having_test_on_keys_moves_to_where():
    assert rewritten("SELECT a, COUNT(*) FROM t GROUP BY a HAVING a IN (SELECT e FROM u)") == "SELECT a, COUNT(*) FROM t WHERE a IN (SELECT e FROM u) GROUP BY a"
    # only the conjunct with a subquery moves; the others are left to the rules that had them
    assert rewritten("SELECT x.a, x.d, SUM(x.c) FROM t AS x WHERE x.b > 0 GROUP BY x.a, x.d HAVING x.d NOT IN (SELECT e FROM u) AND SUM(x.c) > 1 AND x.a > 2") == (
        "SELECT x.a, x.d, SUM(x.c) FROM t AS x WHERE x.b > 0 AND NOT x.d IN (SELECT e FROM u) GROUP BY x.a, x.d HAVING SUM(x.c) > 1 AND x.a > 2"
    )
    # a subquery that reads the key of the group it is asked for
    assert rewritten("SELECT x.a, COUNT(*) FROM t AS x GROUP BY x.a HAVING EXISTS (SELECT 1 FROM u AS v WHERE v.a = x.a AND v.e > 0)") == (
        "SELECT x.a, COUNT(*) FROM t AS x WHERE EXISTS(SELECT 1 FROM u AS v WHERE v.a = x.a AND v.e > 0) GROUP BY x.a"
    )
    assert rewritten("SELECT x.a, COUNT(*) FROM t AS x GROUP BY x.a HAVING x.a = (SELECT MAX(e) FROM u) OR x.a IN (SELECT a FROM u)") == (
        "SELECT x.a, COUNT(*) FROM t AS x WHERE (x.a = (SELECT MAX(e) FROM u) OR x.a IN (SELECT a FROM u)) GROUP BY x.a"
    )
    # a bare name inside the subquery is the subquery's own column (v.a = v.a here): nothing of the group is read
    assert rewritten("SELECT x.a, COUNT(*) FROM t AS x GROUP BY x.a HAVING EXISTS (SELECT 1 FROM u AS v WHERE v.a = a)") == (
        "SELECT x.a, COUNT(*) FROM t AS x WHERE EXISTS(SELECT 1 FROM u AS v WHERE v.a = a) GROUP BY x.a"
    )


def test_a_test_above_a_grouped_derived_table_moves_into_it():
    assert rewritten("SELECT d.k, d.n FROM (SELECT a AS k, COUNT(*) AS n FROM t GROUP BY a) AS d WHERE d.k IN (SELECT e FROM u)") == (
        "SELECT d.k, d.n FROM (SELECT a AS k, COUNT(*) AS n FROM t WHERE a IN (SELECT e FROM u) GROUP BY a) AS d"
    )
    # the output x.a is the GROUP BY item a: one column of the one source
    assert rewritten("SELECT k, n FROM (SELECT x.a AS k, COUNT(*) AS n FROM t AS x GROUP BY a) AS d WHERE k IN (SELECT e FROM u) AND n > 1") == (
        "SELECT k, n FROM (SELECT x.a AS k, COUNT(*) AS n FROM t AS x WHERE x.a IN (SELECT e FROM u) GROUP BY a) AS d WHERE n > 1"
    )
    # read from inside the subquery, the key names its source
    assert rewritten("SELECT d.k, d.n FROM (SELECT a AS k, COUNT(*) AS n FROM t GROUP BY a) AS d WHERE EXISTS (SELECT 1 FROM u AS v WHERE v.e = d.k)") == (
        "SELECT d.k, d.n FROM (SELECT a AS k, COUNT(*) AS n FROM t WHERE EXISTS(SELECT 1 FROM u AS v WHERE v.e = t.a) GROUP BY a) AS d"
    )
    # two keys, a HAVING of the derived table's own, a negated test
    assert rewritten("SELECT d.k, d.n FROM (SELECT x.a AS k, x.d, COUNT(*) AS n FROM t AS x GROUP BY x.a, x.d HAVING COUNT(*) > 1) AS d WHERE NOT EXISTS (SELECT 1 FROM u AS v WHERE v.e = d.k AND v.a = d.d)") == (
        "SELECT d.k, d.n FROM (SELECT x.a AS k, x.d, COUNT(*) AS n FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM u AS v WHERE v.e = x.a AND v.a = x.d) GROUP BY x.a, x.d HAVING COUNT(*) > 1) AS d"
    )
    # a key that is an expression, read outside the subquery
    assert rewritten("SELECT d.k FROM (SELECT a + 1 AS k, COUNT(*) AS n FROM t GROUP BY a + 1) AS d WHERE d.k IN (SELECT e FROM u)") == (
        "SELECT d.k FROM (SELECT a + 1 AS k, COUNT(*) AS n FROM t WHERE (a + 1) IN (SELECT e FROM u) GROUP BY a + 1) AS d"
    )


@pytest.mark.parametrize("sql", [
    "SELECT a, COUNT(*) FROM t GROUP BY a HAVING a > 1",  # no subquery: another rule's
    "SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) IN (SELECT e FROM u)",  # an aggregate of the group
    "SELECT a, COUNT(*) FROM t GROUP BY a HAVING d IN (SELECT e FROM u)",  # not a key
    "SELECT COUNT(*) FROM t HAVING 1 IN (SELECT e FROM u)",  # no key: the global aggregate keeps its row over no rows
    "SELECT a, COUNT(*) FROM t GROUP BY a WITH ROLLUP HAVING a IN (SELECT e FROM u)",
    "SELECT a, COUNT(*) FROM t GROUP BY a HAVING a IN (SELECT e FROM u WHERE u.a = MAX(t.b))",
    "SELECT x.a, COUNT(*) FROM t AS x GROUP BY x.a HAVING EXISTS (SELECT 1 FROM u AS v WHERE v.a = x.d)",  # the subquery reads a column that is not a key
    "SELECT a + 1 AS a, COUNT(*) FROM t GROUP BY t.a HAVING a IN (SELECT e FROM u)",  # a is the alias here
    "SELECT a, COUNT(*) FROM t GROUP BY a HAVING a IN (SELECT e FROM u WHERE RAND() > 0.5)",
    "SELECT a, COUNT(*) OVER () FROM t GROUP BY a HAVING a IN (SELECT e FROM u)",
    "SELECT d.k, d.n FROM (SELECT a AS k, COUNT(*) AS n FROM t GROUP BY a) AS d WHERE d.n IN (SELECT e FROM u)",  # not a key output
    "SELECT d.k, d.n FROM (SELECT a AS k, COUNT(*) AS n FROM t GROUP BY a) AS d WHERE d.k > 1",
    "SELECT d.k, d.n FROM (SELECT a AS k, COUNT(*) AS n FROM t GROUP BY a LIMIT 2) AS d WHERE d.k IN (SELECT e FROM u)",  # which groups the LIMIT keeps
    "SELECT d.k, d.n FROM (SELECT a AS k, COUNT(*) AS n FROM t) AS d WHERE d.k IN (SELECT e FROM u)",  # not grouped
    "SELECT d.k, d.n FROM (SELECT DISTINCT a AS k, COUNT(*) AS n FROM t GROUP BY a) AS d WHERE d.k IN (SELECT e FROM u)",
    "SELECT d.k, d.n FROM (SELECT a AS k, COUNT(*) AS n FROM t GROUP BY a, d) AS d WHERE EXISTS (SELECT 1 FROM u AS t WHERE t.e = d.k)",  # the subquery declares the key's source name
    "SELECT d.k, d.n FROM (SELECT MAX(a) AS k, COUNT(*) AS n FROM t GROUP BY d) AS d WHERE d.k IN (SELECT e FROM u)",
    "SELECT d.k, d.n FROM (SELECT a AS k, COUNT(*) AS n FROM t GROUP BY a) AS d JOIN u ON u.a = d.k WHERE d.k IN (SELECT e FROM u)",  # a join above
    "SELECT d.k FROM (SELECT a + 1 AS k, COUNT(*) AS n FROM t GROUP BY a + 1) AS d WHERE EXISTS (SELECT 1 FROM u AS v WHERE v.e = d.k)",  # an expression key read inside
    "SELECT d.k FROM (SELECT a AS k, COUNT(*) AS n FROM t GROUP BY a) AS d WHERE EXISTS (SELECT 1 FROM u AS v WHERE v.e = d.k AND v.a = z.a)",  # a name bound further out
    "SELECT d.k FROM (SELECT a AS k, c AS a, COUNT(*) AS n FROM t GROUP BY a) AS d WHERE d.k IN (SELECT e FROM u)",  # GROUP BY a: the column or the alias
    "SELECT d.k, d.n FROM (SELECT a AS k, COUNT(*) OVER () AS n FROM t GROUP BY a) AS d WHERE d.k IN (SELECT e FROM u)",  # the window counts the groups
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_a_bare_name_in_a_subquery_needs_the_schema():
    sql = "SELECT a, COUNT(*) FROM t GROUP BY a HAVING a IN (SELECT e FROM u)"
    assert rewritten(sql) is not None
    assert rewritten(sql, schema=None) is None  # e may be a column of t
    assert rewritten("SELECT a, COUNT(*) FROM t GROUP BY a HAVING a IN (SELECT v.e FROM u AS v)", schema=None) is not None


PROOFS = [
    ("SELECT a, COUNT(*) FROM t GROUP BY a HAVING a IN (SELECT e FROM u)", "SELECT a, COUNT(*) FROM t WHERE a IN (SELECT e FROM u) GROUP BY a"),
    ("SELECT x.d, SUM(x.c) FROM t AS x GROUP BY x.d HAVING EXISTS (SELECT 1 FROM u AS v WHERE v.a = x.d AND v.e > 0)", "SELECT d, SUM(c) FROM t WHERE d IN (SELECT a FROM u WHERE e > 0) GROUP BY d"),
    ("SELECT d.k, d.n FROM (SELECT a AS k, COUNT(*) AS n FROM t GROUP BY a) AS d WHERE d.k IN (SELECT e FROM u)", "SELECT a, COUNT(b) FROM t WHERE a IN (SELECT e FROM u) GROUP BY a"),
    ("SELECT d.k, d.n FROM (SELECT c AS k, COUNT(*) AS n FROM t GROUP BY c) AS d WHERE EXISTS (SELECT 1 FROM u AS v WHERE v.e = d.k)", "SELECT c, COUNT(*) FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.e = t.c) GROUP BY c"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # a test on a column that is not a key: HAVING reads one value of the group, WHERE filters its rows
    ("SELECT d.k, d.n FROM (SELECT a AS k, MAX(d) AS m, COUNT(*) AS n FROM t GROUP BY a) AS d WHERE d.m IN (SELECT e FROM u)", "SELECT a, COUNT(*) FROM t WHERE d IN (SELECT e FROM u) GROUP BY a"),
    # the global aggregate returns its row even when WHERE removes every row
    ("SELECT COUNT(*) FROM t HAVING 1 IN (SELECT e FROM u)", "SELECT COUNT(*) FROM t WHERE 1 IN (SELECT e FROM u)"),
    # NOT IN is not NOT EXISTS: u.e may be NULL
    ("SELECT a, COUNT(*) FROM t GROUP BY a HAVING a NOT IN (SELECT e FROM u)", "SELECT a, COUNT(*) FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.e = t.a) GROUP BY a"),
    # another membership
    ("SELECT a, COUNT(*) FROM t GROUP BY a HAVING a IN (SELECT e FROM u)", "SELECT a, COUNT(*) FROM t WHERE a IN (SELECT a FROM u) GROUP BY a"),
    # the test on the group count is not a test on the rows
    ("SELECT d.k, d.n FROM (SELECT a AS k, COUNT(*) AS n FROM t GROUP BY a) AS d WHERE d.n IN (SELECT e FROM u)", "SELECT a, COUNT(*) FROM t WHERE b IN (SELECT e FROM u) GROUP BY a"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


CASES = [
    "SELECT a, COUNT(*) FROM t GROUP BY a HAVING a IN (SELECT e FROM u)",
    "SELECT x.a, x.d, SUM(x.c) FROM t AS x WHERE x.b > 0 GROUP BY x.a, x.d HAVING x.d NOT IN (SELECT e FROM u) AND SUM(x.c) > 1 AND x.a > 0",
    "SELECT x.d, COUNT(*), MAX(x.c) FROM t AS x GROUP BY x.d HAVING EXISTS (SELECT 1 FROM u AS v WHERE v.a = x.d AND v.e > 0)",
    "SELECT x.c, COUNT(*) FROM t AS x GROUP BY x.c HAVING x.c = (SELECT MAX(e) FROM u) OR x.c IN (SELECT a FROM u)",
    "SELECT x.c, x.d FROM t AS x GROUP BY x.c, x.d HAVING NOT EXISTS (SELECT 1 FROM u AS v WHERE v.a = x.c OR v.e = x.d)",
    "SELECT d.k, d.n FROM (SELECT c AS k, COUNT(*) AS n FROM t GROUP BY c) AS d WHERE d.k IN (SELECT e FROM u)",
    "SELECT k, n FROM (SELECT x.d AS k, COUNT(*) AS n FROM t AS x GROUP BY d) AS d WHERE k NOT IN (SELECT e FROM u) AND n > 1",
    "SELECT d.k, d.n FROM (SELECT c AS k, COUNT(*) AS n FROM t GROUP BY c) AS d WHERE EXISTS (SELECT 1 FROM u AS v WHERE v.e = d.k)",
    "SELECT d.k, d.n FROM (SELECT x.c AS k, x.d, COUNT(*) AS n FROM t AS x GROUP BY x.c, x.d HAVING COUNT(*) > 1) AS d WHERE NOT EXISTS (SELECT 1 FROM u AS v WHERE v.e = d.k AND v.a = d.d)",
    "SELECT d.k FROM (SELECT c + 1 AS k, COUNT(*) AS n FROM t GROUP BY c + 1) AS d WHERE d.k IN (SELECT e FROM u)",
    "SELECT d.k, d.m FROM (SELECT c AS k, MAX(d) AS m FROM t GROUP BY c) AS d WHERE d.k > (SELECT MIN(e) FROM u) OR d.k IS NULL AND EXISTS (SELECT 1 FROM u)",
]


@pytest.mark.parametrize("sql", CASES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The test after the grouping and before it on random tables: NULL keys, NULLs in the subquery, empty tables, repeats."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 2, 3])  # noqa: E731
    for _ in range(150):
        keys = {(rng.choice([1, 1, 2, 3]), rng.choice([0, 1, 2, 3, 4])) for _ in range(rng.choice([0, 1, 2, 4, 7]))}
        tables = {
            "t": [(a, b, value(), value()) for a, b in sorted(keys)],
            "u": [(value(), value()) for _ in range(rng.choice([0, 1, 3, 6]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
