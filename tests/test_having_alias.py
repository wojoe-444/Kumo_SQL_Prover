"""A select alias read in HAVING, written as the expression it names (``kumo_sql_prover.having_alias``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.having_alias import read_having_aliases

# every column may be NULL and repeat; w is a table whose columns are not supplied
SCHEMA = {"t": ["id", "a", "b"], "u": ["id", "v"]}


def rewritten(sql, schema=SCHEMA):
    before = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    text = before.sql(dialect="mysql")
    after = read_having_aliases(before, schema).sql(dialect="mysql")
    return after if after != text else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, compare_names=False, dialect="mysql").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE t (id BIGINT, a BIGINT, b BIGINT)")
    db.execute("CREATE TABLE u (id BIGINT, v BIGINT)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


def test_an_alias_of_an_aggregate_becomes_the_aggregate():
    assert rewritten("SELECT a, COUNT(b) AS n FROM t GROUP BY a HAVING n > 1") == "SELECT a, COUNT(b) AS n FROM t GROUP BY a HAVING COUNT(b) > 1"
    assert rewritten("SELECT COUNT(*) AS n FROM t HAVING n > 1") == "SELECT COUNT(*) AS n FROM t HAVING COUNT(*) > 1"
    # an expression around the aggregate keeps its parentheses; a plain HAVING test is left as it is
    assert rewritten("SELECT a, SUM(b) + 1 AS s FROM t GROUP BY a HAVING s * 2 > 4 AND a > 0") == (
        "SELECT a, SUM(b) + 1 AS s FROM t GROUP BY a HAVING (SUM(b) + 1) * 2 > 4 AND a > 0"
    )
    # in a derived table, two aliases, one under OR
    assert rewritten("SELECT x.a FROM (SELECT a, COUNT(DISTINCT b) AS n, MAX(b) AS top FROM t GROUP BY a HAVING n >= 2 OR top IS NULL) AS x") == (
        "SELECT x.a FROM (SELECT a, COUNT(DISTINCT b) AS n, MAX(b) AS top FROM t GROUP BY a HAVING COUNT(DISTINCT b) >= 2 OR MAX(b) IS NULL) AS x"
    )


def test_an_alias_of_grouping_columns_becomes_the_columns():
    assert rewritten("SELECT a AS g, COUNT(*) AS n FROM t GROUP BY a HAVING g > 0 AND n > 1") == (
        "SELECT a AS g, COUNT(*) AS n FROM t GROUP BY a HAVING a > 0 AND COUNT(*) > 1"
    )
    assert rewritten("SELECT t.a, u.v, COUNT(*) AS n, t.a + u.v AS s FROM t JOIN u ON t.id = u.id GROUP BY t.a, u.v HAVING s > 2 AND n > 0") == (
        "SELECT t.a, u.v, COUNT(*) AS n, t.a + u.v AS s FROM t JOIN u ON t.id = u.id GROUP BY t.a, u.v HAVING (t.a + u.v) > 2 AND COUNT(*) > 0"
    )


@pytest.mark.parametrize("sql", [
    "SELECT a, COUNT(*) AS b FROM t GROUP BY a HAVING b > 1",  # b is a column of t too
    "SELECT a, COUNT(*) AS n FROM w GROUP BY a HAVING n > 1",  # w may have a column n
    "SELECT a, COUNT(*) AS n FROM (SELECT * FROM t) AS d GROUP BY a HAVING n > 1",  # so may a star
    "SELECT t.a, COUNT(*) AS v FROM t JOIN u ON t.id = u.id GROUP BY t.a HAVING v > 1",  # a column of the second source
    "SELECT a, COUNT(*) AS n, SUM(b) AS n FROM t GROUP BY a HAVING n > 1",  # two outputs share the name
    "SELECT a, b AS x FROM t GROUP BY a HAVING x > 1",  # b is not one value per group
    "SELECT a, b + MAX(id) AS x FROM t GROUP BY a HAVING x > 1",
    "SELECT t.a, COUNT(*) AS n, u.id + 1 AS s FROM t JOIN u ON t.id = u.id GROUP BY t.a, t.id HAVING s > 2",  # u.id is not grouped, t.id is
    "SELECT a, COUNT(*) AS n FROM t GROUP BY a HAVING SUM(n) > 1",  # inside an aggregate
    "SELECT a, COUNT(*) AS n FROM t GROUP BY a HAVING t.n > 1",  # qualified: only a column
    "SELECT a, COUNT(*) OVER () AS n FROM t GROUP BY a HAVING n > 1",  # a window
    "SELECT a, (SELECT COUNT(*) FROM u) AS n FROM t GROUP BY a HAVING n > 1",  # a subquery
    "SELECT a, RAND() AS r FROM t GROUP BY a HAVING r > 0.5",  # a new value at each call
    "SELECT a AS x FROM t HAVING x > 1",  # no grouping and no aggregate
    "SELECT a, COUNT(*) AS n FROM t GROUP BY a HAVING EXISTS (SELECT 1 FROM u WHERE u.v = n)",  # read from a nested query
    "SELECT a, COUNT(*) AS n FROM t GROUP BY a WITH ROLLUP HAVING n > 1",
    "SELECT a, COUNT(b) AS n, SUM(n) AS s FROM t GROUP BY a HAVING s > 1",  # the expression reads a name that is an alias
    "SELECT a, COUNT(b) AS n FROM t GROUP BY a HAVING COUNT(b) > 1",  # nothing to do
    "SELECT a AS a, COUNT(b) AS n FROM t GROUP BY a HAVING a > 1",
])
def test_other_names_stay(sql):
    assert rewritten(sql) is None


def test_the_columns_of_every_source_have_to_be_known():
    sql = "SELECT a, COUNT(b) AS n FROM t GROUP BY a HAVING n > 1"
    assert rewritten(sql) is not None
    assert rewritten(sql, schema=None) is None
    assert rewritten(sql, schema={"u": ["id", "v"]}) is None
    assert rewritten(sql, schema={"t": ["id", "a", "b", "n"]}) is None


PROOFS = [
    ("SELECT x.a FROM (SELECT a, COUNT(b) AS n FROM t GROUP BY a HAVING n > 1) AS x", "SELECT a FROM t GROUP BY a HAVING COUNT(b) > 1"),
    (
        "SELECT x.a FROM (SELECT a, SUM(b) AS s, COUNT(*) AS n FROM t GROUP BY a HAVING s > 10 AND n >= 2) AS x",
        "SELECT a FROM t GROUP BY a HAVING SUM(b) > 10 AND COUNT(*) >= 2",
    ),
    ("SELECT a AS g, COUNT(*) AS n FROM t GROUP BY a HAVING g > 0 AND n > 1", "SELECT a, COUNT(*) FROM t WHERE a > 0 GROUP BY a HAVING COUNT(*) > 1"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # COUNT(*) counts the rows whose b is NULL, COUNT(b) does not
    ("SELECT x.a FROM (SELECT a, COUNT(*) AS n FROM t GROUP BY a HAVING n > 1) AS x", "SELECT a FROM t GROUP BY a HAVING COUNT(b) > 1"),
    # another threshold
    ("SELECT x.a FROM (SELECT a, COUNT(b) AS n FROM t GROUP BY a HAVING n > 1) AS x", "SELECT a FROM t GROUP BY a HAVING COUNT(b) > 2"),
    # b is a column of t: the name is not read as the alias
    ("SELECT x.a FROM (SELECT a, COUNT(*) AS b FROM t GROUP BY a HAVING b > 1) AS x", "SELECT a FROM t GROUP BY a HAVING COUNT(*) > 1"),
    # the alias of another aggregate
    ("SELECT x.a FROM (SELECT a, MIN(b) AS n, MAX(b) AS m FROM t GROUP BY a HAVING n > 1) AS x", "SELECT a FROM t GROUP BY a HAVING MAX(b) > 1"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_the_near_misses_really_differ():
    left, right = NOT_PROVED[0]
    assert run({"t": [(1, 7, None), (2, 7, None)]}, left, right) == [["(7,)"], []]
    left, right = NOT_PROVED[3]
    assert run({"t": [(1, 7, 1), (2, 7, 5)]}, left, right) == [[], ["(7,)"]]


QUERIES = [
    "SELECT a, COUNT(b) AS n FROM t GROUP BY a HAVING n > 1",
    "SELECT COUNT(*) AS n FROM t HAVING n > 1",
    "SELECT a, SUM(b) + 1 AS s FROM t GROUP BY a HAVING s * 2 > 4 AND a > 0",
    "SELECT x.a FROM (SELECT a, COUNT(DISTINCT b) AS n, MAX(b) AS top FROM t GROUP BY a HAVING n >= 2 OR top IS NULL) AS x",
    "SELECT a AS g, COUNT(*) AS n FROM t GROUP BY a HAVING g > 0 AND n > 1",
    "SELECT t.a, u.v, COUNT(*) AS n, t.a + u.v AS s FROM t JOIN u ON t.id = u.id GROUP BY t.a, u.v HAVING s > 2 AND n > 0",
    "SELECT a, MIN(b) AS low, MAX(b) AS high FROM t GROUP BY a HAVING low <> high OR NOT high > 1",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the rewritten one on random tables with NULLs and repeated rows."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 3])  # noqa: E731
    for _ in range(150):
        tables = {
            "t": [(rng.choice([1, 2, 3]), value(), value()) for _ in range(rng.choice([0, 1, 2, 4, 7]))],
            "u": [(rng.choice([1, 2, 3]), value()) for _ in range(rng.choice([0, 1, 3]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
