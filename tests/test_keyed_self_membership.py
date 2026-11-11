"""A membership test that finds the row's own table again through its key (``kumo_sql_prover.keyed_self_membership``)."""

import random

import pytest
import sqlglot
from sqlglot import exp

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.keyed_self_membership import keyed_self_membership
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: a is the key (NOT NULL); b, c and d may be NULL and repeat.  w: (k1, k2) is the key.  u: no facts at all.
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"], "w": ["k1", "k2", "v"]}
KEYS = {"t": [("a",)], "w": [("k1", "k2")]}
NOT_NULL = {"t": frozenset({"a"}), "w": frozenset({"k1", "k2"})}
CONSTRAINTS = {
    "t": TableConstraints(not_null=frozenset({"a"}), keys=(("a",),)),
    "w": TableConstraints(not_null=frozenset({"k1", "k2"}), keys=(("k1", "k2"),)),
}


def rewritten(sql, keys=KEYS, not_null=NOT_NULL, schema=SCHEMA):
    tree = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    text = tree.sql(dialect="mysql")
    for select in list(tree.find_all(exp.Select)):
        keyed_self_membership(select, schema, keys, not_null)
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
    db.execute("CREATE TABLE w (k1 BIGINT, k2 BIGINT, v BIGINT)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


def test_a_membership_test_on_the_key_reads_the_row_itself():
    assert rewritten("SELECT x.a, x.d FROM t AS x WHERE NOT x.a IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.b WHERE u.e > p.c)") == (
        "SELECT x.a, x.d FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM u WHERE u.a = x.b AND u.e > x.c)"
    )
    # DISTINCT and ORDER BY in the subquery, a comma join, the table second, bare names
    assert rewritten("SELECT d FROM t WHERE a IN (SELECT DISTINCT p.a FROM u, t AS p WHERE e = b OR c = 1 ORDER BY p.a)") == (
        "SELECT d FROM t WHERE EXISTS(SELECT 1 FROM u WHERE (e = t.b OR t.c = 1))"
    )
    # the subquery's table hides the outer one of the same name: its columns are the outer row's afterwards
    assert rewritten("SELECT t.d FROM t WHERE NOT t.a IN (SELECT t.a FROM t JOIN u ON u.a = t.b)") == (
        "SELECT t.d FROM t WHERE NOT EXISTS(SELECT 1 FROM u WHERE u.a = t.b)"
    )
    # two more sources, a nested query that reads the row
    assert rewritten(
        "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.b JOIN w ON w.v = u.e WHERE NOT EXISTS(SELECT 1 FROM u AS z WHERE z.e = p.c))"
    ) == "SELECT x.d FROM t AS x WHERE EXISTS(SELECT 1 FROM u, w WHERE u.a = x.b AND w.v = u.e AND NOT EXISTS(SELECT 1 FROM u AS z WHERE z.e = x.c))"


def test_exists_with_the_key_equality():
    assert rewritten("SELECT x.d FROM t AS x WHERE NOT EXISTS(SELECT p.b FROM t AS p JOIN u ON u.a = p.b AND p.a = x.a WHERE u.e >= p.c)") == (
        "SELECT x.d FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM u WHERE u.a = x.b AND u.e >= x.c)"
    )
    assert rewritten("SELECT x.d FROM t AS x WHERE EXISTS(SELECT * FROM u, t AS p WHERE x.a = p.a AND u.a = p.b)") == (
        "SELECT x.d FROM t AS x WHERE EXISTS(SELECT 1 FROM u WHERE u.a = x.b)"
    )


def test_a_key_of_two_columns():
    assert rewritten("SELECT w.v FROM w WHERE (w.k1, w.k2) IN (SELECT p.k1, p.k2 FROM w AS p JOIN u ON u.a = p.v)") == (
        "SELECT w.v FROM w WHERE EXISTS(SELECT 1 FROM u WHERE u.a = w.v)"
    )
    assert rewritten("SELECT x.v FROM w AS x WHERE NOT EXISTS(SELECT 1 FROM w AS p, u WHERE p.k1 = x.k1 AND u.a = p.v AND x.k2 = p.k2)") == (
        "SELECT x.v FROM w AS x WHERE NOT EXISTS(SELECT 1 FROM u WHERE u.a = x.v)"
    )
    # half the key names many rows
    assert rewritten("SELECT w.v FROM w WHERE w.k1 IN (SELECT p.k1 FROM w AS p JOIN u ON u.a = p.v)") is None
    assert rewritten("SELECT x.v FROM w AS x WHERE EXISTS(SELECT 1 FROM w AS p, u WHERE p.k1 = x.k1 AND u.a = p.v)") is None
    assert rewritten("SELECT w.v FROM w WHERE (w.k1, w.k2) IN (SELECT p.k2, p.k1 FROM w AS p JOIN u ON u.a = p.v)") is None  # crossed


def test_the_row_is_kept_by_the_joins_of_its_select():
    inner = "(SELECT p.a FROM t AS p JOIN u AS v ON v.e = p.d)"
    assert rewritten(f"SELECT x.a, u.e FROM t AS x LEFT JOIN u ON u.a = x.a WHERE NOT x.a IN {inner} OR u.e > 1") == (
        "SELECT x.a, u.e FROM t AS x LEFT JOIN u ON u.a = x.a WHERE NOT EXISTS(SELECT 1 FROM u AS v WHERE v.e = x.d) OR u.e > 1"
    )
    assert rewritten(f"SELECT x.a FROM u RIGHT JOIN t AS x ON u.a = x.a JOIN w ON w.v = x.b WHERE x.a IN {inner}") == (
        "SELECT x.a FROM u RIGHT JOIN t AS x ON u.a = x.a JOIN w ON w.v = x.b WHERE EXISTS(SELECT 1 FROM u AS v WHERE v.e = x.d)"
    )
    # a row an outer join may have padded with NULLs is not a row of the table
    assert rewritten(f"SELECT u.e FROM u LEFT JOIN t AS x ON u.a = x.a WHERE NOT x.a IN {inner}") is None
    assert rewritten(f"SELECT u.e FROM t AS x RIGHT JOIN u ON u.a = x.a WHERE NOT x.a IN {inner}") is None
    assert rewritten(f"SELECT u.e FROM t AS x FULL JOIN u ON u.a = x.a WHERE NOT x.a IN {inner}") is None
    assert rewritten(f"SELECT u.e FROM u FULL JOIN t AS x ON u.a = x.a WHERE NOT x.a IN {inner}") is None
    assert rewritten(f"SELECT u.e FROM t AS x JOIN w ON w.v = x.a RIGHT JOIN u ON u.a = x.a WHERE NOT x.a IN {inner}") is None


def test_one_source_leaves_the_condition():
    # a positive AND part of the WHERE: NULL and FALSE both drop the row
    assert rewritten("SELECT x.d FROM t AS x WHERE x.b > 0 AND x.a IN (SELECT p.a FROM t AS p WHERE p.c > 1 AND p.d = 2)") == (
        "SELECT x.d FROM t AS x WHERE x.b > 0 AND (x.c > 1 AND x.d = 2)"
    )
    # a condition that cannot be NULL, anywhere
    assert rewritten("SELECT x.d FROM t AS x WHERE NOT x.a IN (SELECT p.a FROM t AS p WHERE EXISTS(SELECT 1 FROM u WHERE u.a = p.b))") == (
        "SELECT x.d FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM u WHERE u.a = x.b)"
    )
    assert rewritten("SELECT x.d FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM t AS p WHERE p.a = x.a AND p.c IS NULL) OR x.b = 1") == (
        "SELECT x.d FROM t AS x WHERE NOT x.c IS NULL OR x.b = 1"
    )
    assert rewritten("SELECT x.d FROM t AS x WHERE NOT x.a IN (SELECT p.a FROM t AS p)") == "SELECT x.d FROM t AS x WHERE NOT TRUE"
    # NOT (c > 1) is NULL for a NULL c, where NOT IN is TRUE
    assert rewritten("SELECT x.d FROM t AS x WHERE NOT x.a IN (SELECT p.a FROM t AS p WHERE p.c > 1)") is None
    assert rewritten("SELECT x.d FROM t AS x WHERE x.b = 1 OR x.a IN (SELECT p.a FROM t AS p WHERE p.c > 1)") is None
    assert rewritten("SELECT x.d FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM t AS p WHERE p.a = x.a AND p.c > 1)") is None


@pytest.mark.parametrize("sql", [
    "SELECT x.d FROM t AS x WHERE x.b IN (SELECT p.b FROM t AS p JOIN u ON u.a = p.c)",  # b is no key
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.b FROM t AS p JOIN u ON u.a = p.c)",  # another column
    "SELECT x.d FROM t AS x WHERE x.b IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.c)",
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a + 0 FROM t AS p JOIN u ON u.a = p.c)",
    "SELECT x.d FROM t AS x WHERE x.a + 0 IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.c)",
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT u.a FROM u JOIN t AS p ON u.a = p.c)",  # another table's column
    "SELECT u.e FROM u WHERE u.a IN (SELECT p.a FROM u AS p JOIN t ON t.a = p.e)",  # u has no key
    "SELECT x.d FROM t AS x WHERE x.a IN (1, 2)",
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a FROM u LEFT JOIN t AS p ON u.a = p.c)",  # a NULL among the outputs
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a FROM t AS p LEFT JOIN u ON u.a = p.c WHERE u.e = 1)",  # left as written; the step loop makes it inner first
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a FROM t AS p JOIN u USING (a))",
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a FROM t AS p NATURAL JOIN u)",
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.c GROUP BY p.a)",
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.c GROUP BY p.a HAVING COUNT(*) > 1)",
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.c LIMIT 1)",
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT MAX(p.a) FROM t AS p JOIN u ON u.a = p.c)",
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.c UNION SELECT 1)",
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a FROM t AS p JOIN (SELECT a FROM u) AS s ON s.a = p.c)",  # a derived table beside it
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a FROM (SELECT * FROM t WHERE c > 0) AS p JOIN u ON u.a = p.c)",
    # names that cannot be settled
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.b WHERE EXISTS(SELECT 1 FROM u AS z WHERE z.e = c))",  # c: p's, unless z has it
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.b WHERE EXISTS(SELECT 1 FROM u AS p WHERE p.e = 1))",  # a nested p
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.b WHERE EXISTS(SELECT 1 FROM u AS x WHERE x.e = p.c))",  # a nested x
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a FROM t AS p JOIN u AS x ON x.a = p.b)",  # a source named like x
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.b WHERE EXISTS(SELECT p.* FROM u AS z))",
    "SELECT d FROM t JOIN u ON u.e = t.b WHERE a IN (SELECT p.a FROM t AS p JOIN u AS v ON v.a = p.b)",  # a: t's or u's
    "SELECT d FROM t JOIN (SELECT e FROM u) AS s ON s.e = t.b WHERE a IN (SELECT p.a FROM t AS p JOIN u AS v ON v.a = p.b)",  # a derived table may have it
    # EXISTS
    "SELECT x.d FROM t AS x WHERE EXISTS(SELECT 1 FROM t AS p JOIN u ON u.a = p.b)",  # no key equality
    "SELECT x.d FROM t AS x WHERE EXISTS(SELECT 1 FROM t AS p JOIN u ON u.a = p.b WHERE p.b = x.b)",  # not the key
    "SELECT x.d FROM t AS x WHERE EXISTS(SELECT 1 FROM t AS p JOIN u ON u.a = p.b WHERE p.a = x.b)",
    "SELECT x.d FROM t AS x WHERE EXISTS(SELECT 1 FROM t AS p JOIN u ON u.a = p.b WHERE p.a = x.a OR u.e = 1)",
    "SELECT x.d FROM t AS x WHERE EXISTS(SELECT 1 FROM t AS p JOIN u ON u.a = p.b WHERE p.a >= x.a)",
    "SELECT x.d FROM t AS x WHERE EXISTS(SELECT 1 FROM t AS p JOIN u ON u.a = p.b WHERE p.a = x.a LIMIT 0)",
    "SELECT x.d FROM t AS x WHERE EXISTS(SELECT COUNT(*) FROM t AS p JOIN u ON u.a = p.b WHERE p.a = x.a)",  # always one row
    "SELECT x.d FROM t AS x WHERE EXISTS(SELECT 1 FROM t AS p LEFT JOIN u ON u.a = p.b AND p.a = x.a)",
    "SELECT x.d FROM t AS x WHERE EXISTS(SELECT 1 FROM t AS p JOIN u ON u.a = p.b WHERE p.a = u.a)",  # not the outer row
    "SELECT x.d FROM t AS x WHERE EXISTS(SELECT 1 FROM t JOIN u ON u.a = t.b WHERE a = x.a AND x.a = u.a)",  # bare a: t's or u's
    # the test is not in the WHERE of the select that holds the row
    "SELECT x.d, x.a IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.b) FROM t AS x",
    "SELECT x.d FROM t AS x GROUP BY x.d, x.a HAVING x.a IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.b)",
    "SELECT x.d FROM t AS x JOIN u ON x.a IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.b)",
    "SELECT x.d FROM t AS x WHERE EXISTS(SELECT 1 FROM u AS z WHERE z.e = 1 AND x.a IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.b))",
])
def test_other_tests_stay(sql):
    assert rewritten(sql) is None


def test_the_declared_facts_are_needed():
    sql = "SELECT x.d FROM t AS x WHERE NOT x.a IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.b)"
    assert rewritten(sql) is not None
    assert rewritten(sql, keys=None) is None
    assert rewritten(sql, keys={"u": [("a",)]}) is None
    assert rewritten(sql, not_null=None) is None  # a UNIQUE column that may be NULL: NOT IN of a NULL is not TRUE
    assert rewritten(sql, not_null={"t": frozenset({"b"})}) is None
    assert rewritten(sql, schema=None) is None
    exists = "SELECT x.d FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM t AS p JOIN u ON u.a = p.b WHERE p.a = x.a)"
    assert rewritten(exists) is not None
    assert rewritten(exists, keys={"t": [("a", "b")]}) is None
    assert rewritten(exists, not_null=None) is None


ANTI = "SELECT x.a, x.d FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM u WHERE u.a = x.b AND u.e > x.c)"
PROOFS = [
    ("SELECT a, d FROM t WHERE a NOT IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.b WHERE u.e > p.c)", ANTI),
    ("SELECT a, d FROM t WHERE a NOT IN (SELECT DISTINCT t.a FROM u, t WHERE u.a = b AND e > c)", ANTI),
    # a left join whose padded rows the WHERE rejects, USING, a WITH table: all read by the time the rule runs
    ("SELECT a, d FROM t WHERE a NOT IN (SELECT p.a FROM t AS p LEFT JOIN u ON u.a = p.b WHERE u.e > p.c)", ANTI),
    ("WITH hit AS (SELECT DISTINCT t.a FROM t JOIN u ON u.a = t.b AND u.e > t.c) SELECT a, d FROM t WHERE a NOT IN (SELECT * FROM hit)", ANTI),
    ("SELECT y.a, y.d FROM t AS y WHERE NOT EXISTS(SELECT 1 FROM t AS p JOIN u ON u.a = p.b WHERE p.a = y.a AND u.e > p.c)", ANTI),
    ("SELECT y.a, y.d FROM t AS y LEFT JOIN u ON u.a = y.b AND u.e > y.c WHERE u.a IS NULL", "SELECT a, d FROM t WHERE a NOT IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.b WHERE u.e > p.c)"),
    ("SELECT a FROM t WHERE a IN (SELECT p.a FROM t AS p WHERE p.c > 1)", "SELECT a FROM t WHERE c > 1"),
    ("SELECT v FROM w WHERE (k1, k2) IN (SELECT p.k1, p.k2 FROM w AS p JOIN u ON u.a = p.v)", "SELECT v FROM w WHERE EXISTS(SELECT 1 FROM u WHERE u.a = w.v)"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NO_KEY = {"t": TableConstraints(not_null=frozenset({"a"}))}
NOT_PROVED = [
    # without the key two rows may share a, and the other row's b may have the match
    ("SELECT a, d FROM t WHERE a NOT IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.b WHERE u.e > p.c)", ANTI, NO_KEY),
    # NOT IN over a column that may be NULL is not NOT EXISTS
    ("SELECT x.a FROM t AS x WHERE x.b NOT IN (SELECT u.a FROM u)", "SELECT x.a FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM u WHERE u.a = x.b)", CONSTRAINTS),
    # the outer join leaves a NULL among the subquery's outputs
    ("SELECT x.a FROM t AS x WHERE x.a NOT IN (SELECT p.a FROM u LEFT JOIN t AS p ON u.a = p.b)", "SELECT x.a FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM u WHERE u.a = x.b)", CONSTRAINTS),
    # NULL c: NOT IN is TRUE, NOT (c > 1) is NULL
    ("SELECT x.a FROM t AS x WHERE x.a NOT IN (SELECT p.a FROM t AS p WHERE p.c > 1)", "SELECT x.a FROM t AS x WHERE NOT x.c > 1", CONSTRAINTS),
    # a padded row
    (
        "SELECT u.e FROM u LEFT JOIN t AS x ON u.a = x.b WHERE x.a NOT IN (SELECT p.a FROM t AS p JOIN u AS v ON v.e = p.d)",
        "SELECT u.e FROM u LEFT JOIN t AS x ON u.a = x.b WHERE NOT EXISTS(SELECT 1 FROM u AS v WHERE v.e = x.d)",
        CONSTRAINTS,
    ),
    # half the key
    ("SELECT v FROM w WHERE k1 IN (SELECT p.k1 FROM w AS p JOIN u ON u.a = p.v)", "SELECT v FROM w WHERE EXISTS(SELECT 1 FROM u WHERE u.a = w.v)", CONSTRAINTS),
]


@pytest.mark.parametrize("left, right, constraints", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right, constraints):
    assert not proved(left, right, constraints)
    assert not proved(right, left, constraints)


def test_the_look_alikes_really_differ():
    # two rows share a (no key): the second row's b has the match
    left, right, _ = NOT_PROVED[0]
    a, b = run({"t": [(1, 5, 0, 0), (1, 6, 0, 0)], "u": [(5, 1)]}, left, right)
    assert a == [] and b == ["(1, 0)"]
    left, right, _ = NOT_PROVED[3]
    a, b = run({"t": [(1, 0, None, 0)]}, left, right)
    assert a == ["(1,)"] and b == []
    left, right, _ = NOT_PROVED[4]
    a, b = run({"u": [(7, 1)], "t": [(1, 0, 0, 1)]}, left, right)
    assert a == [] and b == ["(1,)"]


QUERIES = [
    "SELECT x.a, x.d FROM t AS x WHERE NOT x.a IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.b WHERE u.e > p.c)",
    "SELECT d FROM t WHERE a IN (SELECT DISTINCT p.a FROM u, t AS p WHERE e = b OR c = 1 ORDER BY p.a)",
    "SELECT t.d FROM t WHERE NOT t.a IN (SELECT t.a FROM t JOIN u ON u.a = t.b)",
    "SELECT x.d FROM t AS x WHERE x.a IN (SELECT p.a FROM t AS p JOIN u ON u.a = p.b JOIN w ON w.v = u.e WHERE NOT EXISTS(SELECT 1 FROM u AS z WHERE z.e = p.c))",
    "SELECT x.d FROM t AS x WHERE NOT EXISTS(SELECT p.b FROM t AS p JOIN u ON u.a = p.b AND p.a = x.a WHERE u.e >= p.c)",
    "SELECT x.d FROM t AS x WHERE x.c = 1 OR EXISTS(SELECT * FROM u, t AS p WHERE x.a = p.a AND u.a = p.b)",
    "SELECT w.v FROM w WHERE NOT (w.k1, w.k2) IN (SELECT p.k1, p.k2 FROM w AS p JOIN u ON u.a = p.v)",
    "SELECT x.v FROM w AS x WHERE NOT EXISTS(SELECT 1 FROM w AS p, u WHERE p.k1 = x.k1 AND u.a = p.v AND x.k2 = p.k2)",
    "SELECT x.a, u.e FROM t AS x LEFT JOIN u ON u.a = x.a WHERE NOT x.a IN (SELECT p.a FROM t AS p JOIN u AS v ON v.e = p.d) OR u.e > 1",
    "SELECT x.a FROM u RIGHT JOIN t AS x ON u.a = x.a JOIN w ON w.v = x.b WHERE NOT x.a IN (SELECT p.a FROM t AS p JOIN u AS v ON v.e = p.d)",
    "SELECT x.d FROM t AS x WHERE x.b > 0 AND x.a IN (SELECT p.a FROM t AS p WHERE p.c > 1 AND p.d = 2)",
    "SELECT x.d FROM t AS x WHERE NOT x.a IN (SELECT p.a FROM t AS p WHERE EXISTS(SELECT 1 FROM u WHERE u.a = p.b))",
    "SELECT x.d FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM t AS p WHERE p.a = x.a AND p.c IS NULL) OR x.b = 1",
    "SELECT x.d FROM t AS x WHERE NOT x.a IN (SELECT p.a FROM t AS p)",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the rewritten one on random tables that meet the declared facts: ``t.a`` and ``w.(k1, k2)``
    are keys; every other column may be NULL and repeat."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 2, 3])  # noqa: E731
    for _ in range(150):
        pairs = rng.sample([(i, j) for i in range(3) for j in range(3)], rng.choice([0, 1, 3, 5]))
        tables = {
            "t": [(a, value(), value(), value()) for a in rng.sample(range(8), rng.choice([0, 1, 2, 4, 7]))],
            "u": [(value(), value()) for _ in range(rng.choice([0, 1, 3, 5]))],
            "w": [(k1, k2, value()) for k1, k2 in pairs],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
