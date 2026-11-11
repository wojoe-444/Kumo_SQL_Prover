"""``DISTINCT`` in an aggregate whose argument a key keeps from repeating in a group (``kumo_sql_prover.keyed_distinct_aggregates``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.keyed_distinct_aggregates import drop_keyed_aggregate_distinct
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: (a, b) is a key and both are NOT NULL; c and d may be NULL and repeat. u has no key.
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
KEYS = {"t": [("a", "b")]}
NOT_NULL = {"t": frozenset({"a", "b"})}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a", "b"}), keys=(("a", "b"),))}


def rewritten(sql, keys=KEYS, not_null=NOT_NULL, schema=SCHEMA):
    tree = sqlglot.parse_one(sql, read="mysql")
    text = tree.sql(dialect="mysql")
    for select in list(tree.find_all(sqlglot.exp.Select)):
        drop_keyed_aggregate_distinct(select, schema, keys, not_null)
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


def test_the_rest_of_the_key_is_grouped():
    assert rewritten("SELECT a, COUNT(DISTINCT b) FROM t GROUP BY a") == "SELECT a, COUNT(b) FROM t GROUP BY a"
    assert rewritten("SELECT x.b, COUNT(DISTINCT x.a) AS n FROM t AS x WHERE x.c > 0 GROUP BY x.b") == "SELECT x.b, COUNT(x.a) AS n FROM t AS x WHERE x.c > 0 GROUP BY x.b"
    assert rewritten("SELECT a, SUM(DISTINCT b), AVG(DISTINCT (b)) FROM t GROUP BY a") == "SELECT a, SUM(b), AVG(b) FROM t GROUP BY a"
    # more grouping columns only split the groups further
    assert rewritten("SELECT a, d, COUNT(DISTINCT b) FROM t GROUP BY a, d") == "SELECT a, d, COUNT(b) FROM t GROUP BY a, d"
    # HAVING and ORDER BY read the same group
    assert rewritten("SELECT a FROM t GROUP BY a HAVING COUNT(DISTINCT b) > 1 ORDER BY COUNT(DISTINCT b)") == "SELECT a FROM t GROUP BY a HAVING COUNT(b) > 1 ORDER BY COUNT(b)"
    # a subquery in WHERE only removes rows of t
    assert rewritten("SELECT a, COUNT(DISTINCT b) FROM t WHERE a IN (SELECT e FROM u) GROUP BY a") == "SELECT a, COUNT(b) FROM t WHERE a IN (SELECT e FROM u) GROUP BY a"


def test_the_whole_key_is_grouped_or_counted():
    # each group is one row: any column's values are distinct in it, NULL or not
    assert rewritten("SELECT a, b, COUNT(DISTINCT c) FROM t GROUP BY a, b") == "SELECT a, b, COUNT(c) FROM t GROUP BY a, b"
    assert rewritten("SELECT COUNT(DISTINCT a) FROM t", keys={"t": [("a",)]}) == "SELECT COUNT(a) FROM t"
    # a nullable single-column key: the counted rows are the ones where it is not NULL
    assert rewritten("SELECT COUNT(DISTINCT c) FROM t", keys={"t": [("c",)]}) == "SELECT COUNT(c) FROM t"
    assert rewritten("SELECT a, COUNT(DISTINCT c) FROM t GROUP BY a", keys={"t": [("a", "c")]}) == "SELECT a, COUNT(c) FROM t GROUP BY a"


def test_the_tuple_count_is_the_row_count():
    assert rewritten("SELECT a, COUNT(DISTINCT a, b) FROM t GROUP BY a") == "SELECT a, COUNT(*) FROM t GROUP BY a"
    assert rewritten("SELECT COUNT(DISTINCT b, a) FROM t") == "SELECT COUNT(*) FROM t"
    assert rewritten("SELECT COUNT(DISTINCT a, b, c) FROM t") is None  # c may be NULL: such a row is not counted
    assert rewritten("SELECT d, COUNT(DISTINCT a, c) FROM t GROUP BY d") is None


@pytest.mark.parametrize("sql", [
    "SELECT a, COUNT(DISTINCT c) FROM t GROUP BY a",  # (a, c) is not a key
    "SELECT d, COUNT(DISTINCT b) FROM t GROUP BY d",  # (d, b) is not a key
    "SELECT COUNT(DISTINCT b) FROM t",  # b alone is not a key
    "SELECT a, COUNT(b) FROM t GROUP BY a",  # nothing to drop
    "SELECT a, MIN(DISTINCT b) FROM t GROUP BY a",  # MIN never saw the repeats
    "SELECT a, COUNT(DISTINCT b + 1) FROM t GROUP BY a",  # not a column
    "SELECT a + 1, COUNT(DISTINCT b) FROM t GROUP BY a + 1",  # grouped by an expression: two values of a share a group
    "SELECT a, COUNT(DISTINCT b) FROM t GROUP BY a WITH ROLLUP",  # the total row mixes the groups
    "SELECT x.a, COUNT(DISTINCT x.b) FROM t AS x JOIN u ON u.a = x.a GROUP BY x.a",  # a join repeats the rows of t
    "SELECT x.a, COUNT(DISTINCT y.b) FROM t AS x, t AS y GROUP BY x.a",
    "SELECT a, COUNT(DISTINCT b) FROM (SELECT a, b FROM t UNION ALL SELECT a, b FROM t) AS t GROUP BY a",  # not the base table
    "WITH t AS (SELECT a, e AS b FROM u) SELECT a, COUNT(DISTINCT b) FROM t GROUP BY a",  # a CTE with the table's name
    "SELECT a, COUNT(DISTINCT e) FROM u GROUP BY a",  # no key declared
    "SELECT a, COUNT(DISTINCT b) OVER (PARTITION BY a) FROM t",  # a window reads another set of rows
    "SELECT a, (SELECT COUNT(DISTINCT y.e) FROM u AS y WHERE y.a = x.a) FROM t AS x GROUP BY a",  # the subquery's own aggregate
    "SELECT a, COUNT(DISTINCT u.b) FROM t GROUP BY a",  # not this table's column
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_the_key_and_its_not_null_columns_are_needed():
    sql = "SELECT a, COUNT(DISTINCT b) FROM t GROUP BY a"
    assert rewritten(sql, keys=None) is None
    assert rewritten(sql, keys={"t": [("a", "b", "c")]}) is None
    assert rewritten(sql, keys={"t": [("b",)]}) is not None
    # a nullable grouping column of the key: its NULL group may repeat b
    assert rewritten(sql, not_null={"t": frozenset({"b"})}) is None
    assert rewritten(sql, not_null=None) is None
    # the argument itself may be nullable: the rows where it is NULL are not counted
    assert rewritten(sql, not_null={"t": frozenset({"a"})}) is not None
    assert rewritten(sql, schema=None) is None


PROOFS = [
    ("SELECT a, COUNT(DISTINCT b) FROM t GROUP BY a", "SELECT a, COUNT(*) FROM t GROUP BY a"),
    ("SELECT b, SUM(DISTINCT a) FROM t WHERE c > 0 GROUP BY b", "SELECT b, SUM(a) FROM t WHERE c > 0 GROUP BY b"),
    ("SELECT a, COUNT(DISTINCT a, b) FROM t GROUP BY a", "SELECT a, COUNT(b) FROM t GROUP BY a"),
    ("SELECT a, COUNT(DISTINCT b) FROM t WHERE a IN (SELECT e FROM u) GROUP BY a", "SELECT a, COUNT(a) FROM t WHERE a IN (SELECT e FROM u) GROUP BY a"),
    ("SELECT a FROM t GROUP BY a HAVING COUNT(DISTINCT b) > 2", "SELECT a FROM t GROUP BY a HAVING COUNT(*) > 2"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # c repeats within a group and may be NULL
    ("SELECT a, COUNT(DISTINCT c) FROM t GROUP BY a", "SELECT a, COUNT(c) FROM t GROUP BY a"),
    ("SELECT a, COUNT(DISTINCT c) FROM t GROUP BY a", "SELECT a, COUNT(*) FROM t GROUP BY a"),
    # b repeats across the groups of d
    ("SELECT d, COUNT(DISTINCT b) FROM t GROUP BY d", "SELECT d, COUNT(b) FROM t GROUP BY d"),
    ("SELECT COUNT(DISTINCT b) FROM t", "SELECT COUNT(b) FROM t"),
    # a join repeats the rows of t
    ("SELECT x.a, COUNT(DISTINCT x.b) FROM t AS x JOIN u ON u.a = x.a GROUP BY x.a", "SELECT x.a, COUNT(x.b) FROM t AS x JOIN u ON u.a = x.a GROUP BY x.a"),
    # u has no key
    ("SELECT a, COUNT(DISTINCT e) FROM u GROUP BY a", "SELECT a, COUNT(e) FROM u GROUP BY a"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_without_the_key_nothing_is_proved():
    left, right = PROOFS[0]
    assert not proved(left, right, constraints={"t": TableConstraints(not_null=frozenset({"a", "b"}))})
    # the key without NOT NULL on the grouped column: the NULL group may repeat b
    assert not proved(left, right, constraints={"t": TableConstraints(not_null=frozenset({"b"}), keys=(("a", "b"),))})


KEYED = [
    "SELECT a, COUNT(DISTINCT b) FROM t GROUP BY a",
    "SELECT x.b, COUNT(DISTINCT x.a) AS n, SUM(DISTINCT x.a), AVG(DISTINCT x.a) FROM t AS x WHERE x.c > 0 OR x.d IS NULL GROUP BY x.b",
    "SELECT a, d, COUNT(DISTINCT b), SUM(DISTINCT b) FROM t GROUP BY a, d",
    "SELECT a, b, COUNT(DISTINCT c), SUM(DISTINCT d) FROM t GROUP BY a, b",
    "SELECT a FROM t GROUP BY a HAVING COUNT(DISTINCT b) > 1",
    "SELECT a, COUNT(DISTINCT a, b), COUNT(DISTINCT b, a) FROM t GROUP BY a",
    "SELECT COUNT(DISTINCT b, a), COUNT(DISTINCT c) FROM t",
    "SELECT a, COUNT(DISTINCT b) FROM t WHERE a IN (SELECT e FROM u) GROUP BY a",
    "SELECT d, (SELECT COUNT(DISTINCT y.b) FROM t AS y WHERE y.a = x.d GROUP BY y.a) FROM t AS x",
]


@pytest.mark.parametrize("sql", KEYED)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the rewritten one on random tables in which (a, b) is a key; c and d hold NULLs and repeats."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 2, 3])  # noqa: E731
    for _ in range(150):
        keys = {(rng.choice([1, 1, 2, 3]), rng.choice([0, 1, 2, 3, 4])) for _ in range(rng.choice([0, 1, 2, 4, 7, 10]))}
        tables = {
            "t": [(a, b, value(), value()) for a, b in sorted(keys)],
            "u": [(value(), value()) for _ in range(rng.choice([0, 1, 3, 5]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)


def test_a_nullable_key_column_is_the_counterexample():
    """Why the other key columns must be NOT NULL: UNIQUE lets (NULL, 1) repeat, and GROUP BY puts both in one group."""

    tables = {"t": [(None, 1, 0, 0), (None, 1, 0, 0)]}
    a, b = run(tables, "SELECT a, COUNT(DISTINCT b) FROM t GROUP BY a", "SELECT a, COUNT(b) FROM t GROUP BY a")
    assert a != b
