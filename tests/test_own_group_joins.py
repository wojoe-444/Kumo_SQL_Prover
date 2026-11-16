"""The join of a table with its own grouped aggregate (``kumo_sql_prover.own_group_joins``)."""

import random

import pytest
import sqlglot
from sqlglot import exp

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.own_group_joins import inner_own_group_join, own_group_join
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: (a, b) is the key, so a and b are NOT NULL; c and d may be NULL and repeat.  u: no facts.
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
NOT_NULL = {"t": frozenset({"a", "b"})}
KEYS = {"t": [("a", "b")]}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a", "b"}), keys=(("a", "b"),))}
G = "(SELECT a, MAX(c) AS top FROM t GROUP BY a) AS g"
GD = "(SELECT d, MAX(c) AS top FROM t GROUP BY d) AS g"


def rewritten(sql, not_null=NOT_NULL, schema=SCHEMA):
    tree = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    text = tree.sql(dialect="mysql")
    for select in list(tree.find_all(exp.Select)):
        inner_own_group_join(select, schema, not_null)
    after = tree.sql(dialect="mysql")
    return after if after != text else None


def recognized(sql, schema=SCHEMA):
    return own_group_join(sqlglot.parse_one(sql, read="mysql"), schema)


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


def test_the_join_is_recognized():
    assert recognized(f"SELECT s.b, g.top FROM t AS s JOIN {G} ON s.a = g.a")
    assert recognized(f"SELECT s.b, g.top FROM {G} JOIN t AS s ON s.a = g.a")
    assert recognized(f"SELECT s.b, g.top FROM t AS s, {G} WHERE s.a = g.a AND s.c = g.top")
    assert recognized("SELECT s.b FROM t AS s JOIN (SELECT a, b, CASE WHEN MAX(c) > 1 THEN 0.5 ELSE 1 END AS rate, COUNT(*) AS n FROM t AS y GROUP BY y.a, b) AS g ON s.a = g.a")
    assert not recognized(f"SELECT s.e, g.top FROM u AS s JOIN {G} ON s.a = g.a")  # another table
    assert not recognized("SELECT s.b, g.top FROM t AS s JOIN (SELECT a, MAX(e) AS top FROM u GROUP BY a) AS g ON s.a = g.a")
    assert not recognized("SELECT s.b, g.top FROM t AS s JOIN (SELECT a, MAX(c) AS top FROM t WHERE d > 0 GROUP BY a) AS g ON s.a = g.a")
    assert not recognized("SELECT s.b, g.top FROM t AS s JOIN (SELECT a, MAX(c) AS top FROM t GROUP BY a HAVING COUNT(*) > 1) AS g ON s.a = g.a")
    assert not recognized("SELECT s.b, g.top FROM t AS s JOIN (SELECT a, MAX(c) AS top FROM t GROUP BY a + 1) AS g ON s.a = g.a")
    assert not recognized("SELECT s.b, g.top FROM t AS s JOIN (SELECT MAX(c) AS top FROM t) AS g ON s.c = g.top")
    assert not recognized(f"SELECT s.b, g.top FROM t AS s JOIN {G} ON s.a = g.a JOIN u ON u.a = s.a")  # a third source
    assert not recognized("SELECT s.b FROM t AS s JOIN t AS g ON s.a = g.a")
    assert not recognized(f"SELECT s.b, g.top FROM t AS s JOIN {G} ON s.a = g.a", schema=None)


def test_the_aggregate_is_not_pulled_up_out_of_the_join():
    sql = f"SELECT s.b, s.d, g.top FROM t AS s JOIN {G} ON s.a = g.a"
    form = normalize(sql, schema=SCHEMA, dialect="mysql", not_null=NOT_NULL, keys=KEYS)
    assert "GROUP BY a) AS g" in form and "GROUP BY s." not in form
    # a grouped table over another table is pulled up as before
    other = normalize("SELECT s.b, g.top FROM t AS s JOIN (SELECT a, MAX(e) AS top FROM u GROUP BY a) AS g ON s.a = g.a", schema=SCHEMA, dialect="mysql", not_null=NOT_NULL, keys=KEYS)
    assert "GROUP BY s.a, s.b" in other


def test_an_outer_join_that_cannot_fail_is_inner():
    assert rewritten(f"SELECT s.b, g.top FROM t AS s LEFT JOIN {G} ON s.a = g.a") == f"SELECT s.b, g.top FROM t AS s JOIN {G} ON s.a = g.a"
    assert rewritten(f"SELECT s.b, g.top FROM t AS s LEFT OUTER JOIN {G} ON g.a = s.a WHERE g.top > 1") == f"SELECT s.b, g.top FROM t AS s JOIN {G} ON g.a = s.a WHERE g.top > 1"
    assert rewritten(f"SELECT s.b, g.top FROM {G} RIGHT JOIN t AS s ON s.a = g.a") == f"SELECT s.b, g.top FROM {G} JOIN t AS s ON s.a = g.a"
    # two grouping columns, renamed outputs, any other output, other joins around it
    wide = "(SELECT b AS bb, y.a, CASE WHEN MAX(c) > 1 THEN 0.5 ELSE 1 END AS rate FROM t AS y GROUP BY y.a, b) AS g"
    assert rewritten(f"SELECT s.d, g.rate, u.e FROM u JOIN t AS s ON u.a = s.a LEFT JOIN {wide} ON s.a = g.a AND g.bb = s.b LEFT JOIN u AS v ON v.a = s.d") == (
        f"SELECT s.d, g.rate, u.e FROM u JOIN t AS s ON u.a = s.a JOIN {wide} ON s.a = g.a AND g.bb = s.b LEFT JOIN u AS v ON v.a = s.d"
    )


@pytest.mark.parametrize("sql", [
    f"SELECT s.b, g.top FROM t AS s LEFT JOIN {GD} ON s.d = g.d",  # a NULL d has no group to equal
    f"SELECT s.b, g.top FROM t AS s LEFT JOIN {G} ON s.a = g.a AND g.top > 1",  # the match may fail the extra test
    f"SELECT s.b, g.top FROM t AS s LEFT JOIN {G} ON s.a = g.a AND s.d > 1",
    f"SELECT s.b, g.top FROM t AS s LEFT JOIN {G} ON s.b = g.a",  # another column than the key
    f"SELECT s.b, g.top FROM t AS s LEFT JOIN {G} ON s.a >= g.a",
    f"SELECT s.b, g.top FROM t AS s LEFT JOIN {G} ON s.a = g.a OR s.b = g.a",
    f"SELECT s.b, g.top FROM t AS s LEFT JOIN {G} ON s.a = g.top",
    f"SELECT s.e, g.top FROM u AS s LEFT JOIN {G} ON s.a = g.a",  # a row of another table need not have a group
    "SELECT s.b, g.top FROM t AS s LEFT JOIN (SELECT a, MAX(c) AS top FROM t WHERE d > 0 GROUP BY a) AS g ON s.a = g.a",  # the row's group may be filtered away
    "SELECT s.b, g.top FROM t AS s LEFT JOIN (SELECT a, MAX(c) AS top FROM t GROUP BY a HAVING MAX(c) > 0) AS g ON s.a = g.a",
    "SELECT s.b, g.top FROM t AS s LEFT JOIN (SELECT a, MAX(c) AS top FROM t GROUP BY a LIMIT 1) AS g ON s.a = g.a",
    "SELECT s.b, g.top FROM t AS s LEFT JOIN (SELECT a, MAX(c) AS top FROM t GROUP BY a, b) AS g ON s.a = g.a",  # a grouping column that is not tied
    "SELECT s.b, g.top FROM t AS s LEFT JOIN (SELECT a + 0 AS a, MAX(c) AS top FROM t GROUP BY a) AS g ON s.a = g.a",
    "SELECT s.b, g.top FROM t AS s LEFT JOIN (SELECT MAX(c) AS top, a FROM u AS t GROUP BY a) AS g ON s.a = g.a",
    f"SELECT s.b, g.top FROM t AS s LEFT JOIN {G} USING (a)",
    f"SELECT s.b, g.top FROM t AS s LEFT JOIN {G} ON a = g.a",  # the row's side is not named
    f"SELECT s.b, g.top FROM u LEFT JOIN t AS s ON u.a = s.a LEFT JOIN {G} ON s.a = g.a",  # s itself may be padded
    f"SELECT s.b, g.top FROM u RIGHT JOIN t AS s ON u.a = s.a LEFT JOIN {G} ON s.a = g.a",
    f"SELECT s.b, g.top FROM t AS s LEFT JOIN {G} ON s.a = g.a FULL JOIN u ON u.a = s.a",
    f"SELECT s.b, g.top FROM t AS s RIGHT JOIN {G} ON s.a = g.a",  # the grouped side is kept: already every group has a row
    f"SELECT s.b, g.top FROM {G} LEFT JOIN t AS s ON s.a = g.a",
    f"SELECT s.b, g.top FROM t AS s FULL JOIN {G} ON s.a = g.a",
    f"SELECT s.b, g.top FROM t AS s JOIN {G} ON s.a = g.a",
])
def test_other_joins_stay(sql):
    assert rewritten(sql) is None


def test_the_key_must_be_declared_not_null():
    sql = f"SELECT s.b, g.top FROM t AS s LEFT JOIN {G} ON s.a = g.a"
    assert rewritten(sql, not_null=None) is None
    assert rewritten(sql, not_null={"t": frozenset({"b"})}) is None
    assert rewritten(sql, schema=None) is None


INNER = f"SELECT s.b, s.d, g.top FROM t AS s JOIN {G} ON s.a = g.a"
PROOFS = [
    (f"SELECT s.b, s.d, g.top FROM t AS s LEFT JOIN {G} ON s.a = g.a", INNER),
    (f"SELECT s.b, s.d, g.top FROM {G} RIGHT JOIN t AS s ON s.a = g.a", INNER),
    # qualified on one side, bare on the other, another order of the arms: one normal form either way
    (
        f"SELECT s.b, CASE WHEN g.top < 10 THEN s.c ELSE 2 * s.c END AS c FROM t AS s JOIN {G} ON s.a = g.a",
        "SELECT b, CASE WHEN top >= 10 THEN 2 * c ELSE c END AS c FROM t JOIN (SELECT a AS k, MAX(c) AS top FROM t GROUP BY a) AS h ON t.a = h.k",
    ),
    (
        f"SELECT s.b, CASE WHEN g.top < 10 THEN s.c ELSE 2 * s.c END AS c FROM t AS s LEFT JOIN {G} ON s.a = g.a",
        "SELECT s.b, s.c * r.rate FROM t AS s JOIN (SELECT a, CASE WHEN MAX(c) < 10 THEN 1 ELSE 2 END AS rate FROM t GROUP BY a) AS r ON s.a = r.a",
    ),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # a NULL d has no group: the outer join keeps the row with a NULL, the inner join drops it
    (f"SELECT s.b, g.top FROM t AS s LEFT JOIN {GD} ON s.d = g.d", f"SELECT s.b, g.top FROM t AS s JOIN {GD} ON s.d = g.d"),
    # the row's group may be filtered away
    (
        "SELECT s.b, g.top FROM t AS s LEFT JOIN (SELECT a, MAX(c) AS top FROM t WHERE d > 0 GROUP BY a) AS g ON s.a = g.a",
        "SELECT s.b, g.top FROM t AS s JOIN (SELECT a, MAX(c) AS top FROM t WHERE d > 0 GROUP BY a) AS g ON s.a = g.a",
    ),
    (f"SELECT s.b, g.top FROM t AS s LEFT JOIN {G} ON s.a = g.a AND g.top > 1", f"SELECT s.b, g.top FROM t AS s JOIN {G} ON s.a = g.a AND g.top > 1"),
    # with a NULL maximum the constant arms differ
    (
        f"SELECT s.b, CASE WHEN g.top < 10 THEN 1 ELSE 2 END FROM t AS s JOIN {G} ON s.a = g.a",
        f"SELECT s.b, CASE WHEN g.top >= 10 THEN 2 ELSE 1 END FROM t AS s JOIN {G} ON s.a = g.a",
    ),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_the_nullable_key_really_differs():
    left, right = NOT_PROVED[0]
    a, b = run({"t": [(1, 1, 5, None)]}, left, right)
    assert a == ["(1, None)"] and b == []


QUERIES = [
    f"SELECT s.b, s.d, g.top FROM t AS s LEFT JOIN {G} ON s.a = g.a",
    f"SELECT s.b, g.top FROM t AS s LEFT OUTER JOIN {G} ON g.a = s.a WHERE g.top > 1 OR g.top IS NULL",
    f"SELECT s.b, g.top FROM {G} RIGHT JOIN t AS s ON s.a = g.a",
    "SELECT s.d, g.rate, u.e FROM u JOIN t AS s ON u.a = s.a LEFT JOIN (SELECT b AS bb, y.a, CASE WHEN MAX(c) > 1 THEN 0.5 ELSE 1 END AS rate FROM t AS y GROUP BY y.a, b) AS g "
    "ON s.a = g.a AND g.bb = s.b LEFT JOIN u AS v ON v.a = s.d",
    "SELECT s.b, g.n, g.total FROM t AS s LEFT JOIN (SELECT a, COUNT(d) AS n, SUM(c) AS total FROM t GROUP BY a) AS g ON s.a = g.a",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the rewritten one on random tables that meet the declared facts: ``(a, b)`` is a key of
    ``t``; ``c`` and ``d`` may be NULL and repeat; ``u`` holds anything."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 2, 3])  # noqa: E731
    for _ in range(150):
        keys = rng.sample([(i, j) for i in range(3) for j in range(4)], rng.choice([0, 1, 2, 4, 7]))
        tables = {
            "t": [(a, b, value(), value()) for a, b in keys],
            "u": [(value(), value()) for _ in range(rng.choice([0, 1, 3]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
