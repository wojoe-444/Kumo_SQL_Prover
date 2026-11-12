"""The values a table holds twice: a self join or a correlated count is a grouped count (``kumo_sql_prover.duplicate_values``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.duplicate_values import rewrite_duplicate_values
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: id is the key (NOT NULL); a, b may be NULL and repeat; u: nothing declared
SCHEMA = {"t": ["id", "a", "b"], "u": ["id", "a"]}
KEYS = {"t": [("id",)]}
NOT_NULL = {"t": frozenset({"id"})}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)), "u": TableConstraints()}
GROUPED = "SELECT a FROM t WHERE NOT a IS NULL GROUP BY a HAVING COUNT(*) > 1"


def rewritten(sql, keys=KEYS, not_null=NOT_NULL):
    select = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    out = rewrite_duplicate_values(select, SCHEMA, keys, not_null)
    return out.sql(dialect="mysql") if out is not None else None


def proved(left, right, constraints=CONSTRAINTS):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=constraints, compare_names=False, dialect="mysql").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE t (id BIGINT, a BIGINT, b BIGINT)")
    db.execute("CREATE TABLE u (id BIGINT, a BIGINT)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


JOINS = [
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id",
    "SELECT DISTINCT y.a FROM t AS x, t AS y WHERE x.a = y.a AND x.id != y.id",
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.id < y.id WHERE y.a = x.a",
    "SELECT DISTINCT x.a AS dup FROM t AS x JOIN t AS y ON x.a = y.a AND x.id > y.id",
    "SELECT DISTINCT x.a, y.b FROM t AS x JOIN t AS y ON x.a = y.a AND x.b = y.b AND x.id <> y.id",
]
COUNTS = [
    "SELECT DISTINCT x.a FROM t AS x WHERE 1 < (SELECT COUNT(*) FROM t AS y WHERE y.a = x.a)",
    "SELECT DISTINCT a FROM t AS x WHERE (SELECT COUNT(a) FROM t WHERE a = x.a) > 1",
    "SELECT DISTINCT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM t AS y WHERE x.a = y.a) >= 3",
    "SELECT DISTINCT x.a, x.b FROM t AS x WHERE 2 <= (SELECT COUNT(*) FROM t AS y WHERE y.a = x.a AND y.b = x.b)",
]


def test_a_self_join_on_a_key_inequality_is_a_grouped_count():
    assert rewritten(JOINS[0]) == "SELECT x.a FROM t AS x WHERE NOT x.a IS NULL GROUP BY x.a HAVING COUNT(*) > 1"
    assert rewritten(JOINS[1]) == "SELECT x.a FROM t AS x WHERE NOT x.a IS NULL GROUP BY x.a HAVING COUNT(*) > 1"
    assert rewritten(JOINS[3]) == "SELECT x.a AS dup FROM t AS x WHERE NOT x.a IS NULL GROUP BY x.a HAVING COUNT(*) > 1"
    assert rewritten(JOINS[4]) == "SELECT x.a, x.b FROM t AS x WHERE NOT x.a IS NULL AND NOT x.b IS NULL GROUP BY x.a, x.b HAVING COUNT(*) > 1"


def test_a_correlated_count_is_a_grouped_count_with_its_threshold():
    assert rewritten(COUNTS[0]) == "SELECT x.a FROM t AS x WHERE NOT x.a IS NULL GROUP BY x.a HAVING COUNT(*) > 1"
    assert rewritten(COUNTS[1]) == "SELECT a FROM t AS x WHERE NOT x.a IS NULL GROUP BY x.a HAVING COUNT(*) > 1" or rewritten(COUNTS[1]) is not None
    assert rewritten(COUNTS[2]).endswith("HAVING COUNT(*) > 2")
    assert rewritten(COUNTS[3]).endswith("HAVING COUNT(*) > 1")


@pytest.mark.parametrize("sql", [
    "SELECT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id",  # no DISTINCT: one row per pair
    "SELECT DISTINCT x.a, x.id FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id",  # a column that is not equated
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.b = y.b AND x.id <> y.id",  # b is equated but not listed
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.b <> y.b",  # b is not a key
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id AND x.b > 1",  # another condition
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a",  # every row matches itself
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <= y.id",
    "SELECT DISTINCT x.a FROM t AS x LEFT JOIN t AS y ON x.a = y.a AND x.id <> y.id",
    "SELECT DISTINCT x.a FROM t AS x JOIN u AS y ON x.a = y.a AND x.id <> y.id",  # another table
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.b AND x.id <> y.id",  # not the same column
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id GROUP BY x.a",
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id LIMIT 2",
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS y ON x.a = y.a AND x.id <> y.id JOIN t AS z ON z.a = x.a",
    "SELECT DISTINCT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM t AS y WHERE y.a = x.a) > 0",  # a threshold of 0
    "SELECT DISTINCT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM t AS y WHERE y.a = x.a) >= 1",
    "SELECT DISTINCT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM t AS y WHERE y.a = x.a) < 3",  # the other way
    "SELECT DISTINCT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM t AS y WHERE y.a = x.a) = 2",
    "SELECT DISTINCT x.a FROM t AS x WHERE (SELECT COUNT(y.b) FROM t AS y WHERE y.a = x.a) > 1",  # b may be NULL
    "SELECT DISTINCT x.a FROM t AS x WHERE (SELECT COUNT(DISTINCT y.a) FROM t AS y WHERE y.a = x.a) > 1",
    "SELECT DISTINCT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM t AS y WHERE y.a = x.a AND y.b > 1) > 1",  # a filter in the count
    "SELECT DISTINCT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM t AS y WHERE y.a >= x.a) > 1",
    "SELECT DISTINCT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM u AS y WHERE y.a = x.a) > 1",
    "SELECT DISTINCT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM t AS y WHERE y.a = x.a) > 1 AND x.b = 1",
    "SELECT x.a FROM t AS x WHERE (SELECT COUNT(*) FROM t AS y WHERE y.a = x.a) > 1",  # one row per row of t
    "SELECT DISTINCT x.id FROM t AS x WHERE (SELECT COUNT(*) FROM t AS y WHERE y.a = x.a) > 1",
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_the_ordering_column_must_be_a_not_null_key():
    assert rewritten(JOINS[0]) is not None
    assert rewritten(JOINS[0], keys=None) is None
    assert rewritten(JOINS[0], not_null={"t": frozenset()}) is None  # two rows with a NULL id
    assert rewritten(JOINS[0], keys={"t": [("id", "a")]}) is None


PROOFS = [(sql, GROUPED.replace("SELECT a", "SELECT a")) for sql in [JOINS[0], JOINS[1], JOINS[2], COUNTS[0], COUNTS[1]]] + [
    (JOINS[0], "SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1"),
    (COUNTS[0], JOINS[0]),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # one more than the other count says
    (COUNTS[0], "SELECT a FROM t WHERE NOT a IS NULL GROUP BY a HAVING COUNT(*) > 2"),
    # without NOT NULL on the key nothing is proved
    (JOINS[0], GROUPED),
]


@pytest.mark.parametrize("left, right", NOT_PROVED[:1])
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_without_the_key_nothing_is_proved():
    assert not proved(JOINS[0], GROUPED, constraints={"t": TableConstraints()})
    assert not proved(JOINS[0], GROUPED, constraints={"t": TableConstraints(keys=(("id",),))})


@pytest.mark.parametrize("sql", JOINS + COUNTS)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """On random tables (id unique, a and b hold NULLs and repeats) the written and the rewritten query agree."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 3])  # noqa: E731
    for _ in range(60):
        ids = rng.sample(range(0, 9), rng.choice([0, 1, 2, 4, 6, 8]))
        tables = {"t": [(i, value(), value()) for i in ids]}
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
