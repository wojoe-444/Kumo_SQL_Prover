"""A table's key is a key of its join with a grouped table (``kumo_sql_prover.own_group_keys``)."""

import random

import pytest
import sqlglot
from sqlglot import exp

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.own_group_keys import keyed_own_group_rows
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: (a, b) is the key; c and d are nullable
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
KEYS = {"t": [("a", "b")]}
NOT_NULL = {"t": frozenset({"a", "b"})}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a", "b"}), keys=(("a", "b"),))}
G = "(SELECT a AS k, MAX(c) AS top FROM t GROUP BY a) AS g"
BY_C = "(SELECT c AS k, COUNT(*) AS n FROM t GROUP BY c) AS g"
BY_TWO = "(SELECT a AS k, b AS k2, MAX(c) AS top FROM t GROUP BY a, b) AS g"


def rewritten(sql, keys=KEYS, not_null=NOT_NULL):
    tree = sqlglot.parse_one(sql, read="mysql")
    text = tree.sql(dialect="mysql")
    for select in list(tree.find_all(exp.Select)):
        keyed_own_group_rows(select, SCHEMA, keys, not_null)
    after = tree.sql(dialect="mysql")
    return after if after != text else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=CONSTRAINTS, compare_names=False, dialect="mysql").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE t (a BIGINT, b BIGINT, c BIGINT, d BIGINT)")
    db.execute("CREATE TABLE u (a BIGINT, e BIGINT)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


def test_a_grouping_that_lists_the_key_is_dropped():
    assert rewritten(f"SELECT s.a, s.b, CASE WHEN g.top > 1 THEN s.c * 2 ELSE s.c END AS c FROM t AS s JOIN {G} ON s.a = g.k GROUP BY s.a, s.b") == (
        f"SELECT s.a, s.b, CASE WHEN g.top > 1 THEN s.c * 2 ELSE s.c END AS c FROM t AS s JOIN {G} ON s.a = g.k"
    )
    # more grouping columns, more ON conditions, a WHERE, an ORDER BY; the grouped table first
    assert rewritten(f"SELECT s.c, g.top FROM {G} INNER JOIN t AS s ON g.k = s.a AND g.top > s.d WHERE s.d > 0 GROUP BY s.b, s.c, g.top, s.a ORDER BY s.c") == (
        f"SELECT s.c, g.top FROM {G} INNER JOIN t AS s ON g.k = s.a AND g.top > s.d WHERE s.d > 0 ORDER BY s.c"
    )
    # an outer join that keeps the table's rows, and a grouped table over a nullable column
    assert rewritten(f"SELECT s.a, s.b, g.n FROM t AS s LEFT JOIN {BY_C} ON s.c = g.k GROUP BY s.a, s.b") == f"SELECT s.a, s.b, g.n FROM t AS s LEFT JOIN {BY_C} ON s.c = g.k"
    assert rewritten(f"SELECT s.a, s.b, g.n FROM {BY_C} RIGHT JOIN t AS s ON s.c = g.k GROUP BY s.a, s.b") == f"SELECT s.a, s.b, g.n FROM {BY_C} RIGHT JOIN t AS s ON s.c = g.k"
    # a comma join: the WHERE holds the equality
    assert rewritten(f"SELECT s.a, s.b, g.top FROM t AS s, {G} WHERE s.c > 0 AND g.k = s.a GROUP BY s.a, s.b, g.top") == (
        f"SELECT s.a, s.b, g.top FROM t AS s, {G} WHERE s.c > 0 AND g.k = s.a"
    )
    # every grouping column of the grouped table is equated
    assert rewritten(f"SELECT s.a, s.b, g.top FROM t AS s JOIN {BY_TWO} ON s.a = g.k AND s.d = g.k2 GROUP BY s.a, s.b") == (
        f"SELECT s.a, s.b, g.top FROM t AS s JOIN {BY_TWO} ON s.a = g.k AND s.d = g.k2"
    )


def test_a_distinct_that_outputs_the_key_is_dropped():
    assert rewritten(f"SELECT DISTINCT s.b, g.top, s.a AS x FROM t AS s JOIN {G} ON s.a = g.k") == f"SELECT s.b, g.top, s.a AS x FROM t AS s JOIN {G} ON s.a = g.k"
    assert rewritten(f"SELECT DISTINCT s.a, s.b, s.c + g.n FROM t AS s LEFT JOIN {BY_C} ON g.k = s.c WHERE s.d IS NULL") == (
        f"SELECT s.a, s.b, s.c + g.n FROM t AS s LEFT JOIN {BY_C} ON g.k = s.c WHERE s.d IS NULL"
    )


@pytest.mark.parametrize("sql", [
    f"SELECT s.a, g.top FROM t AS s JOIN {G} ON s.a = g.k GROUP BY s.a, g.top",  # a alone is not a key
    f"SELECT DISTINCT s.a, s.c FROM t AS s JOIN {G} ON s.a = g.k",
    f"SELECT s.a, s.b FROM t AS s JOIN {G} ON s.a = g.k GROUP BY a, b",  # the qualifier is not written
    f"SELECT s.a, s.b FROM t AS s JOIN {G} ON s.a = g.k GROUP BY g.k, s.b",  # g.k is not a column of the key
    f"SELECT s.a, s.b, COUNT(*) FROM t AS s JOIN {G} ON s.a = g.k GROUP BY s.a, s.b",  # an aggregate
    f"SELECT s.a, s.b FROM t AS s JOIN {G} ON s.a = g.k GROUP BY s.a, s.b ORDER BY MAX(s.c)",
    f"SELECT s.a, s.b, ROW_NUMBER() OVER (ORDER BY s.a) FROM t AS s JOIN {G} ON s.a = g.k GROUP BY s.a, s.b",
    f"SELECT s.a, s.b FROM t AS s JOIN {G} ON s.a = g.k GROUP BY s.a, s.b HAVING s.a > 1",
    f"SELECT DISTINCT s.a, s.b FROM t AS s JOIN {G} ON s.a = g.k GROUP BY s.a, s.b",  # both at once
    f"SELECT s.a, s.b FROM t AS s JOIN {G} ON s.a = g.k GROUP BY s.a, s.b WITH ROLLUP",
    f"SELECT s.a, s.b FROM t AS s JOIN {G} ON s.a > g.k GROUP BY s.a, s.b",  # not an equality: many groups match
    f"SELECT s.a, s.b FROM t AS s JOIN {G} ON s.a = g.top GROUP BY s.a, s.b",  # not the grouping column
    f"SELECT s.a, s.b FROM t AS s JOIN {G} ON s.a = g.k OR s.b = g.k GROUP BY s.a, s.b",
    f"SELECT s.a, s.b FROM t AS s JOIN {BY_TWO} ON s.a = g.k GROUP BY s.a, s.b",  # one of two grouping columns
    f"SELECT s.a, s.b FROM t AS s LEFT JOIN {G} ON TRUE WHERE s.a = g.k GROUP BY s.a, s.b",  # an outer join: only its ON is read
    f"SELECT s.a, s.b FROM t AS s RIGHT JOIN {G} ON s.a = g.k GROUP BY s.a, s.b",  # keeps the groups: padded rows share a NULL key
    f"SELECT s.a, s.b FROM {G} LEFT JOIN t AS s ON s.a = g.k GROUP BY s.a, s.b",
    "SELECT s.a, s.b FROM t AS s JOIN (SELECT a AS k, c AS top FROM t) AS g ON s.a = g.k GROUP BY s.a, s.b",  # not grouped
    "SELECT s.a, s.b FROM t AS s JOIN (SELECT a AS k, MAX(c) AS top FROM t WHERE d > 0 GROUP BY a) AS g ON s.a = g.k GROUP BY s.a, s.b",  # not read
    "SELECT s.a, s.b FROM t AS s JOIN (SELECT a + 1 AS k, MAX(c) AS top FROM t GROUP BY a) AS g ON s.a = g.k GROUP BY s.a, s.b",
    f"SELECT s.a, s.b FROM t AS s JOIN {G} ON s.a = g.k JOIN u ON u.a = s.a GROUP BY s.a, s.b",  # a third source
    f"SELECT s.a, s.b FROM t AS s JOIN {G} USING (a) GROUP BY s.a, s.b",
    "SELECT s.a, s.b FROM t AS s GROUP BY s.a, s.b",  # one source: another rule's
    "SELECT s.a, s.b FROM t AS s JOIN u AS g ON s.a = g.a GROUP BY s.a, s.b",  # a base table: rows repeat
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_the_key_must_be_declared_and_not_null():
    sql = f"SELECT s.a, s.b FROM t AS s JOIN {G} ON s.a = g.k GROUP BY s.a, s.b"
    assert rewritten(sql) is not None
    assert rewritten(sql, keys={}) is None
    assert rewritten(sql, keys={"t": [("a", "b", "c")]}) is None
    assert rewritten(sql, not_null={"t": frozenset({"a"})}) is None
    assert rewritten(sql, not_null={}) is None
    assert rewritten(sql, keys={"u": [("a", "b")]}) is None


PROOFS = [
    (
        f"SELECT s.a, s.b, CASE WHEN g.top > 1 THEN s.c * 2 ELSE s.c END FROM t AS s JOIN {G} ON s.a = g.k GROUP BY s.a, s.b",
        f"SELECT s.a, s.b, CASE WHEN g.top > 1 THEN s.c * 2 ELSE s.c END FROM t AS s JOIN {G} ON s.a = g.k",
    ),
    (f"SELECT DISTINCT s.a, s.b, g.top FROM t AS s JOIN {G} ON s.a = g.k", f"SELECT s.a, s.b, g.top FROM t AS s JOIN {G} ON g.k = s.a"),
    (f"SELECT s.b, s.a, g.top FROM t AS s, {G} WHERE s.a = g.k GROUP BY s.a, s.b, g.top", f"SELECT s.b, s.a, g.top FROM t AS s JOIN {G} ON g.k = s.a"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # a alone is not a key: the grouping merges rows
    (f"SELECT s.a, g.top FROM t AS s JOIN {G} ON s.a = g.k GROUP BY s.a, g.top", f"SELECT s.a, g.top FROM t AS s JOIN {G} ON s.a = g.k"),
    (f"SELECT DISTINCT s.a, s.c FROM t AS s JOIN {G} ON s.a = g.k", f"SELECT s.a, s.c FROM t AS s JOIN {G} ON s.a = g.k"),
    # several groups match one row
    (f"SELECT DISTINCT s.a, s.b FROM t AS s JOIN {BY_TWO} ON s.a = g.k", f"SELECT s.a, s.b FROM t AS s JOIN {BY_TWO} ON s.a = g.k"),
    (f"SELECT DISTINCT s.a, s.b FROM t AS s JOIN {G} ON s.a >= g.k", f"SELECT s.a, s.b FROM t AS s JOIN {G} ON s.a >= g.k"),
    # the groups without a row of s all show a NULL key
    (f"SELECT DISTINCT s.a, s.b FROM t AS s RIGHT JOIN {BY_C} ON s.d = g.k", f"SELECT s.a, s.b FROM t AS s RIGHT JOIN {BY_C} ON s.d = g.k"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


QUERIES = [
    f"SELECT s.a, s.b, s.c, g.top FROM t AS s JOIN {G} ON s.a = g.k GROUP BY s.a, s.b, s.c, g.top",
    f"SELECT DISTINCT s.b, g.top, s.a AS x FROM t AS s JOIN {G} ON s.a = g.k",
    f"SELECT s.b, s.a, CASE WHEN g.top > s.c THEN 1 ELSE 0 END FROM t AS s LEFT JOIN {G} ON g.k = s.a AND g.top > 1 WHERE s.d IS NULL OR s.d > 0 GROUP BY s.b, s.a, s.c, g.top",
    f"SELECT DISTINCT s.c, s.a, s.b, g.n FROM {BY_C} RIGHT JOIN t AS s ON s.c = g.k",
    f"SELECT s.a, s.b, g.top FROM t AS s, {G} WHERE g.k = s.a AND s.c > 0 GROUP BY s.a, s.b, g.top",
    f"SELECT s.a, s.b, g.top FROM t AS s JOIN {BY_TWO} ON s.a = g.k AND s.d = g.k2 GROUP BY s.a, s.b, g.top",
    f"SELECT DISTINCT s.a, s.b, s.c + g.n FROM t AS s LEFT JOIN {BY_C} ON g.k = s.d",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the one without its GROUP BY or DISTINCT, as bags, on random tables with a key on
    (a, b), NULLs and repeated values in the other columns."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 2, 3])  # noqa: E731
    for _ in range(150):
        keys = rng.sample([(i, j) for i in range(3) for j in range(4)], rng.choice([0, 1, 2, 4, 7]))
        tables = {"t": [(a, b, value(), value()) for a, b in keys]}
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
