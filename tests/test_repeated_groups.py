"""Two different rows that agree on columns, read as the groups of more than one row (``kumo_sql_prover.repeated_groups``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.repeated_groups import read_repeated_groups
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: id is a NOT NULL key, code a key that may be NULL; a and b may be NULL and repeat. u has no key.
SCHEMA = {"t": ["id", "code", "a", "b"], "u": ["id", "a"]}
NOT_NULL = {"t": frozenset({"id"})}
KEYS = {"t": [("id",), ("code",)]}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"id"}), keys=(("id",), ("code",)))}


def rewritten(sql, not_null=NOT_NULL, keys=KEYS, types=None):
    before = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    text = before.sql(dialect="mysql")
    after = read_repeated_groups(before, SCHEMA, not_null, keys, types).sql(dialect="mysql")
    return after if after != text else None


def proved(left, right, constraints=CONSTRAINTS):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=constraints, compare_names=False, dialect="mysql").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE t (id BIGINT, code BIGINT, a BIGINT, b BIGINT)")
    db.execute("CREATE TABLE u (id BIGINT, a BIGINT)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


GROUPS = "SELECT x.a FROM t AS x WHERE NOT x.a IS NULL GROUP BY x.a HAVING COUNT(*) > 1"


def test_a_self_join_on_different_keys_becomes_the_groups():
    assert rewritten("SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id") == GROUPS
    assert rewritten("SELECT DISTINCT x.a FROM t AS x, t AS y WHERE x.id != y.id AND y.a = x.a") == GROUPS
    assert rewritten("SELECT DISTINCT x.a FROM t AS x CROSS JOIN t AS y WHERE (x.a = y.a) AND y.id <> x.id") == GROUPS
    # either order of the keys, and the column read from the other side
    assert rewritten("SELECT DISTINCT y.a FROM t AS x INNER JOIN t AS y ON x.a = y.a WHERE x.id < y.id") == GROUPS
    assert rewritten("SELECT DISTINCT (y.a) AS dup FROM t AS x, t AS y WHERE y.id > x.id AND x.a = y.a") == GROUPS.replace("SELECT x.a", "SELECT (x.a) AS dup")
    # GROUP BY of the outputs in place of DISTINCT, two compared columns
    assert rewritten("SELECT x.a, y.b FROM t AS x JOIN t AS y ON x.a = y.a AND y.b = x.b WHERE x.id < y.id GROUP BY x.a, x.b") == (
        "SELECT x.a, x.b FROM t AS x WHERE NOT x.a IS NULL AND NOT x.b IS NULL GROUP BY x.a, x.b HAVING COUNT(*) > 1"
    )
    # the table under its own name on one side
    assert rewritten("SELECT DISTINCT t.a FROM t JOIN t AS y ON t.a = y.a AND t.id <> y.id") == GROUPS.replace("x.", "t.").replace(" AS x", "")
    # in a subquery
    assert rewritten("SELECT u.id FROM u WHERE u.a IN (SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id)") == f"SELECT u.id FROM u WHERE u.a IN ({GROUPS})"


def test_a_compared_column_declared_not_null_needs_no_test():
    declared = {"t": frozenset({"id", "a"})}
    assert rewritten("SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id", not_null=declared) == GROUPS.replace(" WHERE NOT x.a IS NULL", "")


def test_an_outer_join_whose_padded_rows_the_where_rejects():
    assert rewritten("SELECT DISTINCT x.a FROM t AS x LEFT JOIN t AS y ON x.a = y.a WHERE x.id <> y.id") == GROUPS
    assert rewritten("SELECT DISTINCT x.a FROM t AS x RIGHT JOIN t AS y ON x.id <> y.id WHERE x.a = y.a") == GROUPS
    assert rewritten("SELECT DISTINCT x.a FROM t AS x FULL OUTER JOIN t AS y ON x.a = y.a WHERE x.id > y.id") == GROUPS


@pytest.mark.parametrize("sql", [
    "SELECT DISTINCT x.a FROM t AS x LEFT JOIN t AS y ON x.a = y.a AND x.id <> y.id",  # every row of x is kept
    "SELECT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id",  # a row per pair
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <= y.id",  # a row pairs with itself
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id >= y.id",
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a",
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.id <> y.id",  # nothing compared
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.code <> y.code",  # a key that may be NULL
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.b <> y.b",  # not a key
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.code",  # two different columns
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.b AND x.id <> y.id",
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id AND x.id < y.id",  # two key tests
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id AND x.id = y.id",
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id WHERE x.b > 0",  # a filter on one side
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a OR x.id <> y.id",
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id + 1",
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> x.id",  # one side only
    "SELECT DISTINCT x.id FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id",  # the rows, not the groups
    "SELECT DISTINCT x.a, x.b FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id",  # an output that is not compared
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.b = y.b AND x.id <> y.id",  # a compared column that is not output
    "SELECT DISTINCT x.a + 1 FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id",
    "SELECT DISTINCT a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id",  # a is either side's
    "SELECT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id GROUP BY x.a HAVING COUNT(*) > 2",
    "SELECT x.a, COUNT(*) FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id GROUP BY x.a",
    "SELECT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.b = y.b AND x.id <> y.id GROUP BY x.a",  # grouped by fewer columns
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id LIMIT 1",
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id ORDER BY x.a",
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y USING (a) WHERE x.id <> y.id",
    "SELECT DISTINCT x.a FROM t AS x NATURAL JOIN t AS y WHERE x.id <> y.id",
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id JOIN u ON u.a = x.a",  # a third source
    "SELECT DISTINCT x.a FROM u AS x JOIN u AS y ON x.a = y.a AND x.id <> y.id",  # no declared key
    "SELECT DISTINCT x.a FROM t AS x JOIN u AS y ON x.a = y.a AND x.id <> y.id",  # two tables
    "SELECT DISTINCT x.a FROM t AS x JOIN (SELECT * FROM t) AS y ON x.a = y.a AND x.id <> y.id",
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_the_key_has_to_be_declared_unique_and_not_null():
    sql = "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id"
    assert rewritten(sql) == GROUPS
    assert rewritten(sql, not_null=None) is None
    assert rewritten(sql, not_null={"t": frozenset({"a"})}) is None
    assert rewritten(sql, keys=None) is None
    assert rewritten(sql, keys={"t": [("code",)]}) is None
    assert rewritten(sql, keys={"t": [("id", "code")]}) is None  # unique only together with another column
    assert rewritten(sql, types={"t": {"id": "DOUBLE"}}) is None  # NaN is not equal to itself
    assert rewritten(sql, types={"t": {"id": "INT", "a": "FLOAT64"}}) is None  # and is one group all the same
    assert rewritten(sql, types={"t": {"id": "INT", "a": "VARCHAR", "b": "DOUBLE"}}) == GROUPS


PROOFS = [
    ("SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id", "SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1"),
    ("SELECT DISTINCT y.a FROM t AS x, t AS y WHERE x.a = y.a AND x.id < y.id", "SELECT a FROM t WHERE a IS NOT NULL GROUP BY a HAVING COUNT(*) >= 2"),
    (
        "SELECT x.a, x.b FROM t AS x JOIN t AS y ON x.a = y.a AND x.b = y.b AND x.id > y.id GROUP BY x.a, x.b",
        "SELECT a, b FROM t WHERE a IS NOT NULL AND b IS NOT NULL GROUP BY a, b HAVING COUNT(*) > 1",
    ),
    ("SELECT DISTINCT x.a FROM t AS x LEFT JOIN t AS y ON x.a = y.a WHERE x.id <> y.id", "SELECT a FROM t GROUP BY a HAVING COUNT(a) >= 2"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # two rows with a NULL a: COUNT(*) keeps the NULL group, the join compares nothing equal
    ("SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id", "SELECT a FROM t GROUP BY a HAVING COUNT(*) > 1"),
    # two rows with a NULL code are different rows the comparison does not tell apart
    ("SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.code <> y.code", "SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1"),
    # b is not a key: two rows may share it
    ("SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.b <> y.b", "SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1"),
    # a row pairs with itself
    ("SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <= y.id", "SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1"),
    # no DISTINCT: a row per pair
    ("SELECT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id", "SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1"),
    # no key on u: one row repeated is two rows the comparison does not tell apart
    ("SELECT DISTINCT x.a FROM u AS x JOIN u AS y ON x.a = y.a AND x.id <> y.id", "SELECT a FROM u GROUP BY a HAVING COUNT(a) > 1"),
    # the outer join keeps every row of x
    ("SELECT DISTINCT x.a FROM t AS x LEFT JOIN t AS y ON x.a = y.a AND x.id <> y.id", "SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1"),
    # more than two rows
    ("SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id", "SELECT a FROM t GROUP BY a HAVING COUNT(a) > 2"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_the_near_misses_really_differ():
    rows = {"t": [(1, None, None, 1), (2, None, None, 1)]}
    left, right = NOT_PROVED[0]
    assert run(rows, left, right) == [[], ["(None,)"]]
    rows = {"t": [(1, None, 5, 1), (2, None, 5, 1)]}
    for left, right in NOT_PROVED[1:3]:
        assert run(rows, left, right) == [[], ["(5,)"]]
    rows = {"t": [(1, 1, 5, 1)]}
    left, right = NOT_PROVED[3]
    assert run(rows, left, right) == [["(5,)"], []]
    left, right = NOT_PROVED[5]
    assert run({"u": [(1, 5), (1, 5)]}, left, right) == [[], ["(5,)"]]


QUERIES = [
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id",
    "SELECT DISTINCT y.a FROM t AS x, t AS y WHERE x.a = y.a AND x.id < y.id",
    "SELECT DISTINCT x.a AS dup FROM t AS x CROSS JOIN t AS y WHERE y.id < x.id AND y.a = x.a",
    "SELECT x.a FROM t AS x JOIN t AS y ON x.a = y.a WHERE x.id != y.id GROUP BY x.a",
    "SELECT DISTINCT x.b, x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.b = y.b AND x.id > y.id",
    "SELECT x.a, x.b FROM t AS x JOIN t AS y ON x.a = y.a AND x.b = y.b AND x.id <> y.id GROUP BY x.a, x.b",
    "SELECT DISTINCT x.a FROM t AS x LEFT JOIN t AS y ON x.a = y.a WHERE x.id <> y.id",
    "SELECT DISTINCT x.a FROM t AS x RIGHT JOIN t AS y ON x.id <> y.id WHERE x.a = y.a",
    "SELECT DISTINCT x.a FROM t AS x FULL OUTER JOIN t AS y ON x.a = y.a WHERE x.id > y.id",
    "SELECT u.id FROM u WHERE u.a IN (SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id)",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the rewritten one on random tables that meet the declared facts: id unique and not NULL;
    a and b NULL or repeated."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, None, 1, 1, 2, 3])  # noqa: E731
    for _ in range(150):
        size = rng.choice([0, 1, 2, 3, 5, 8])
        ids = rng.sample(range(1, 20), size)
        tables = {
            "t": [(ids[i], None, value(), rng.choice([None, 1, 1, 2])) for i in range(size)],
            "u": [(rng.choice([1, 2]), value()) for _ in range(rng.choice([0, 1, 3]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
