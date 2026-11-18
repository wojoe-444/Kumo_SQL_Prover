"""``COUNT(..) >= 1`` holds for every group whose rows are all counted (``kumo_sql_prover.nonempty_group_counts``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.nonempty_group_counts import drop_nonempty_group_counts
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: a and b are NOT NULL (and (a, b) is a key); c and d may be NULL. u declares nothing.
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
NOT_NULL = {"t": frozenset({"a", "b"})}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a", "b"}), keys=(("a", "b"),))}


def rewritten(sql, not_null=NOT_NULL, schema=SCHEMA):
    tree = sqlglot.parse_one(sql, read="mysql")
    text = tree.sql(dialect="mysql")
    for select in list(tree.find_all(sqlglot.exp.Select)):
        drop_nonempty_group_counts(select, schema, not_null)
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


def test_a_having_test_every_group_passes_is_dropped():
    assert rewritten("SELECT a, SUM(c) FROM t GROUP BY a HAVING COUNT(*) >= 1") == "SELECT a, SUM(c) FROM t GROUP BY a"
    assert rewritten("SELECT a, SUM(c) FROM t GROUP BY a HAVING COUNT(b) > 0 AND SUM(c) > 1") == "SELECT a, SUM(c) FROM t GROUP BY a HAVING SUM(c) > 1"
    assert rewritten("SELECT a FROM t GROUP BY a HAVING 0 < COUNT(a) AND 1 <= COUNT(*)") == "SELECT a FROM t GROUP BY a"
    # inner joins: every row of a group still holds a row of t
    assert rewritten("SELECT x.a, COUNT(DISTINCT u.e) AS n FROM t AS x JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(DISTINCT x.b) <> 0") == (
        "SELECT x.a, COUNT(DISTINCT u.e) AS n FROM t AS x JOIN u ON u.a = x.c GROUP BY x.a"
    )
    # through the select alias, which no table column shares
    assert rewritten("SELECT x.a, COUNT(x.b) AS n FROM t AS x JOIN u ON u.a = x.c GROUP BY x.a HAVING n >= 1") == "SELECT x.a, COUNT(x.b) AS n FROM t AS x JOIN u ON u.a = x.c GROUP BY x.a"
    # COUNT(*) counts the padded rows of an outer join too
    assert rewritten("SELECT x.a, COUNT(*) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(*) > 0") == "SELECT x.a, COUNT(*) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a"


def test_a_test_on_the_count_of_a_grouped_derived_table_is_dropped():
    assert rewritten("SELECT d.a FROM (SELECT a, COUNT(*) AS n FROM t GROUP BY a) AS d WHERE d.n >= 1") == "SELECT d.a FROM (SELECT a, COUNT(*) AS n FROM t GROUP BY a) AS d"
    assert rewritten("SELECT d.a, v.e FROM (SELECT a, COUNT(b) AS n FROM t GROUP BY a) AS d JOIN u AS v ON v.a = d.a WHERE d.n > 0 AND v.e > 1") == (
        "SELECT d.a, v.e FROM (SELECT a, COUNT(b) AS n FROM t GROUP BY a) AS d JOIN u AS v ON v.a = d.a WHERE v.e > 1"
    )
    # the derived table's own HAVING only removes groups
    assert rewritten("SELECT a FROM (SELECT a, COUNT(DISTINCT b) AS n FROM t GROUP BY a HAVING SUM(c) > 0) AS d WHERE n <> 0") == (
        "SELECT a FROM (SELECT a, COUNT(DISTINCT b) AS n FROM t GROUP BY a HAVING SUM(c) > 0) AS d"
    )


@pytest.mark.parametrize("sql", [
    "SELECT a, SUM(c) FROM t GROUP BY a HAVING COUNT(*) >= 2",  # a real bound
    "SELECT a, SUM(c) FROM t GROUP BY a HAVING COUNT(*) = 1",
    "SELECT a, SUM(c) FROM t GROUP BY a HAVING COUNT(*) >= 0",  # true as well, but not this rule's test
    "SELECT a, SUM(c) FROM t GROUP BY a HAVING COUNT(c) >= 1",  # c may be NULL in every row of a group
    "SELECT a, SUM(c) FROM t GROUP BY a HAVING COUNT(DISTINCT d) > 0",
    "SELECT a, SUM(c) FROM t GROUP BY a HAVING COUNT(b + 0) >= 1",  # an expression
    "SELECT SUM(c) FROM t HAVING COUNT(*) >= 1",  # no GROUP BY: the one group may be empty
    "SELECT a, SUM(c) FROM t GROUP BY a WITH ROLLUP HAVING COUNT(b) >= 1",
    "SELECT a, SUM(c) FROM t GROUP BY a HAVING COUNT(*) >= 1 OR SUM(c) > 1",
    "SELECT a, SUM(c) FROM t GROUP BY a HAVING NOT COUNT(*) >= 1",
    "SELECT x.a, COUNT(x.b) FROM u LEFT JOIN t AS x ON u.a = x.a GROUP BY x.a HAVING COUNT(x.b) >= 1",  # x.b is NULL in a padded row
    "SELECT u.a FROM u GROUP BY u.a HAVING COUNT(u.e) >= 1",  # nothing declared for u
    "SELECT g.a FROM (SELECT a, b FROM t) AS g GROUP BY g.a HAVING COUNT(g.b) >= 1",  # a derived table's column
    "SELECT a, c AS b FROM t GROUP BY a, c HAVING COUNT(b) >= 1",  # b may be the alias of c here
    "SELECT d.a FROM (SELECT a, COUNT(*) AS n FROM t GROUP BY a) AS d WHERE d.n >= 2",
    "SELECT d.a FROM (SELECT a, COUNT(c) AS n FROM t GROUP BY a) AS d WHERE d.n >= 1",
    "SELECT d.a FROM (SELECT COUNT(*) AS n, MAX(a) AS a FROM t) AS d WHERE d.n >= 1",  # a global aggregate: n is 0 over no rows
    "SELECT d.a FROM u LEFT JOIN (SELECT a, COUNT(*) AS n FROM t GROUP BY a) AS d ON d.a = u.a WHERE d.n >= 1",  # n is NULL in a padded row
    "SELECT d.a FROM (SELECT a, SUM(b) AS n FROM t GROUP BY a) AS d WHERE d.n >= 1",
    "SELECT d.a FROM (SELECT a, COUNT(*) + 0 AS n FROM t GROUP BY a) AS d WHERE d.n >= 1",
    "SELECT d.a FROM (SELECT a, COUNT(*) AS n FROM t GROUP BY a) AS d WHERE d.n >= 1 OR d.a = 2",
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_the_not_null_declaration_is_needed():
    sql = "SELECT a, SUM(c) FROM t GROUP BY a HAVING COUNT(b) >= 1"
    assert rewritten(sql) is not None
    assert rewritten(sql, not_null={"t": frozenset({"a"})}) is None
    assert rewritten(sql, not_null=None) is None
    assert rewritten(sql, schema=None) is None
    assert rewritten("SELECT a, SUM(c) FROM t GROUP BY a HAVING COUNT(*) >= 1", not_null=None, schema=None) is not None  # COUNT(*) needs no fact


PROOFS = [
    ("SELECT a, SUM(c) FROM t GROUP BY a HAVING COUNT(b) >= 1", "SELECT a, SUM(c) FROM t GROUP BY a"),
    ("SELECT x.a, MAX(u.e) FROM t AS x JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(DISTINCT x.b) > 0 AND COUNT(*) > 0", "SELECT x.a, MAX(u.e) FROM t AS x JOIN u ON u.a = x.c GROUP BY x.a"),
    ("SELECT d.a FROM (SELECT a, COUNT(*) AS n FROM t GROUP BY a) AS d WHERE d.n >= 1", "SELECT DISTINCT a FROM t"),
    ("SELECT d.a, d.n FROM (SELECT a, COUNT(b) AS n FROM t GROUP BY a) AS d JOIN (SELECT c, COUNT(*) AS m FROM t GROUP BY c) AS g ON g.c = d.a WHERE g.m > 0 AND d.n > 0",
     "SELECT a, COUNT(*) FROM t WHERE a IN (SELECT c FROM t) GROUP BY a"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # c may be NULL in every row of a group
    ("SELECT a, SUM(d) FROM t GROUP BY a HAVING COUNT(c) >= 1", "SELECT a, SUM(d) FROM t GROUP BY a"),
    # a real bound
    ("SELECT a, SUM(c) FROM t GROUP BY a HAVING COUNT(b) >= 2", "SELECT a, SUM(c) FROM t GROUP BY a"),
    # the global aggregate over no rows
    ("SELECT SUM(c) FROM t HAVING COUNT(*) >= 1", "SELECT SUM(c) FROM t"),
    ("SELECT d.n FROM (SELECT COUNT(*) AS n FROM t) AS d WHERE d.n >= 1", "SELECT COUNT(*) FROM t"),
    # the padded side of an outer join
    ("SELECT u.a, COUNT(x.b) FROM u LEFT JOIN t AS x ON u.a = x.a GROUP BY u.a HAVING COUNT(x.b) >= 1", "SELECT u.a, COUNT(x.b) FROM u LEFT JOIN t AS x ON u.a = x.a GROUP BY u.a"),
    ("SELECT u.a, d.n FROM u LEFT JOIN (SELECT a, COUNT(*) AS n FROM t GROUP BY a) AS d ON d.a = u.a WHERE d.n >= 1", "SELECT u.a, d.n FROM u LEFT JOIN (SELECT a, COUNT(*) AS n FROM t GROUP BY a) AS d ON d.a = u.a"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_without_not_null_nothing_is_proved():
    left, right = PROOFS[0]
    assert not proved(left, right, constraints={"t": TableConstraints(not_null=frozenset({"a"}))})


CASES = [
    "SELECT a, SUM(c) FROM t GROUP BY a HAVING COUNT(*) >= 1",
    "SELECT a, SUM(c) FROM t GROUP BY a HAVING COUNT(b) > 0 AND SUM(c) > 1",
    "SELECT a FROM t WHERE c > 0 GROUP BY a HAVING 0 < COUNT(a) AND 1 <= COUNT(*)",
    "SELECT x.a, COUNT(DISTINCT u.e) AS n FROM t AS x JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(DISTINCT x.b) <> 0",
    "SELECT x.a, COUNT(x.b) AS n FROM t AS x JOIN u ON u.a = x.c GROUP BY x.a HAVING n >= 1",
    "SELECT x.d, COUNT(*) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.d HAVING COUNT(*) > 0",
    "SELECT d.a FROM (SELECT a, COUNT(*) AS n FROM t GROUP BY a) AS d WHERE d.n >= 1",
    "SELECT d.a, v.e FROM (SELECT a, COUNT(b) AS n FROM t WHERE d > 0 GROUP BY a) AS d JOIN u AS v ON v.a = d.a WHERE d.n > 0 AND v.e > 1",
    "SELECT a FROM (SELECT a, COUNT(DISTINCT b) AS n FROM t GROUP BY a HAVING SUM(c) > 0) AS d WHERE n <> 0",
    "SELECT d.c, d.n FROM (SELECT c, COUNT(a) AS n FROM t GROUP BY c) AS d WHERE d.n >= 1",
]


@pytest.mark.parametrize("sql", CASES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """With and without the test on random tables: a and b are never NULL, c and d hold NULLs, u holds NULLs and repeats."""

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
