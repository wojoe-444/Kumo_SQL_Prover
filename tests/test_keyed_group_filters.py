"""A keyed table joined to a grouped, HAVING-filtered table is one join with the grouped subquery (``kumo_sql_prover.keyed_group_filters``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

import kumo_sql_prover.keyed_group_filters as module
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.smt_equivalence import TableConstraints

# p: k is a key (so NOT NULL); n and o may be NULL. s: k may be NULL, d and v may be NULL. r: (a, b) is a key.
SCHEMA = {"p": ["k", "n", "o"], "s": ["k", "d", "v"], "r": ["a", "b", "c"]}
KEYS = {"p": [("k",)], "r": [("a", "b")]}
CONSTRAINTS = {"p": TableConstraints(not_null=frozenset({"k"}), keys=(("k",),)), "r": TableConstraints(not_null=frozenset({"a", "b"}), keys=(("a", "b"),))}
HAVING = "HAVING MIN(s.d) >= 2 AND MAX(s.d) <= 5"
HUB = "SELECT p.k, p.n FROM p JOIN (SELECT k FROM s GROUP BY k HAVING MIN(d) >= 2 AND MAX(d) <= 5) AS g ON p.k = g.k"


def rewritten(sql, keys=KEYS):
    select = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    out = module.rewrite_keyed_group_filter(select, SCHEMA, keys)
    return out.sql(dialect="mysql") if out is not None else None


def proved(left, right, constraints=CONSTRAINTS):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=constraints, compare_names=False, dialect="mysql").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE p (k BIGINT, n BIGINT, o BIGINT)")
    db.execute("CREATE TABLE s (k BIGINT, d BIGINT, v BIGINT)")
    db.execute("CREATE TABLE r (a BIGINT, b BIGINT, c BIGINT)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


def test_the_flat_join_is_one_join_with_the_grouped_subquery():
    after = rewritten(f"SELECT p.k, p.n FROM p JOIN s ON p.k = s.k GROUP BY p.k, p.n {HAVING}")
    assert after is not None and "JOIN (SELECT s.k AS kgk" in after and "GROUP BY s.k HAVING MIN(s.d) >= 2 AND MAX(s.d) <= 5" in after
    assert after.startswith("SELECT p.k, p.n FROM p JOIN (")
    # grouped on the other side's column, written as a comma join, the matched column listed from s
    after = rewritten(f"SELECT s.k, p.n FROM s, p WHERE s.k = p.k GROUP BY s.k {HAVING}")
    assert after is not None and after.startswith("SELECT p.k, p.n FROM p JOIN (")
    # the output keeps the name of the column it replaces
    after = rewritten(f"SELECT s.v, p.n FROM s JOIN p ON s.v = p.k GROUP BY s.v {HAVING}")
    assert after is not None and after.startswith("SELECT p.k AS v, p.n FROM p JOIN (")
    # a filter on s moves into the subquery, one on p stays
    after = rewritten(f"SELECT p.k FROM p JOIN s ON s.k = p.k AND s.v > 0 WHERE p.o = 1 GROUP BY p.k {HAVING}")
    assert after is not None and "FROM s WHERE s.v > 0 GROUP BY s.k" in after and after.endswith("WHERE p.o = 1")


@pytest.mark.parametrize("sql", [
    f"SELECT p.k, p.n FROM p JOIN s ON p.k = s.k GROUP BY p.k, p.n {HAVING.replace('s.d) >= 2', 'p.o) >= 2')}",  # an aggregate over the keyed table
    f"SELECT p.k, COUNT(*) FROM p JOIN s ON p.k = s.k GROUP BY p.k {HAVING}",  # the aggregate is output
    f"SELECT p.k, p.n FROM p JOIN s ON p.k = s.k GROUP BY p.n {HAVING}",  # the key is not grouped
    f"SELECT p.k, p.n FROM p JOIN s ON p.k = s.k GROUP BY p.k, s.d {HAVING}",  # a column of s splits the groups
    f"SELECT p.k, s.d FROM p JOIN s ON p.k = s.k GROUP BY p.k {HAVING}",  # a column of s that is not matched
    f"SELECT p.k FROM p JOIN s ON p.n = s.k GROUP BY p.k {HAVING}",  # not tied by the key
    f"SELECT p.k FROM p JOIN s ON p.k = s.k AND p.o = s.v GROUP BY p.k {HAVING}",  # a second tie that is not the key's
    f"SELECT p.k FROM p JOIN s ON p.k = s.k WHERE p.o = s.v GROUP BY p.k {HAVING}",
    f"SELECT p.k FROM p LEFT JOIN s ON p.k = s.k GROUP BY p.k {HAVING}",
    f"SELECT p.k FROM p RIGHT JOIN s ON p.k = s.k GROUP BY p.k {HAVING}",
    f"SELECT p.k FROM p JOIN s ON p.k <= s.k GROUP BY p.k {HAVING}",
    f"SELECT p.k FROM p JOIN s ON p.k = s.k GROUP BY p.k {HAVING} LIMIT 2",
    "SELECT p.k FROM p JOIN s ON p.k = s.k GROUP BY p.k",  # no HAVING: nothing to move
    f"SELECT p.k FROM p JOIN s ON p.k = s.k GROUP BY p.k HAVING MIN(s.d) >= p.o",
    f"SELECT p.k FROM p JOIN s ON p.k = s.k GROUP BY p.k HAVING COUNT(DISTINCT s.d) > 1",
    f"SELECT p.k FROM p JOIN s ON p.k = s.k GROUP BY p.k HAVING MIN(s.d) >= (SELECT MIN(o) FROM p)",
    f"SELECT p.k, ROW_NUMBER() OVER (ORDER BY p.k) FROM p JOIN s ON p.k = s.k GROUP BY p.k {HAVING}",
    f"SELECT * FROM p JOIN s ON p.k = s.k GROUP BY p.k {HAVING}",
    f"SELECT p.k FROM p JOIN s ON p.k = s.k JOIN r ON r.a = p.k GROUP BY p.k {HAVING}",
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_the_key_is_needed():
    sql = f"SELECT p.k, p.n FROM p JOIN s ON p.k = s.k GROUP BY p.k, p.n {HAVING}"
    assert rewritten(sql) is not None
    assert rewritten(sql, keys=None) is None
    assert rewritten(sql, keys={"p": [("k", "n")]}) is None  # the key needs a tie on n too
    assert rewritten(sql, keys={"p": [("n",)]}) is None
    # a key of several columns needs each tied and grouped
    two = "SELECT r.a, r.c FROM r JOIN s ON r.a = s.k AND r.b = s.v GROUP BY r.a, r.b, r.c HAVING MIN(s.d) > 1"
    assert rewritten(two) is not None
    assert rewritten(two.replace("AND r.b = s.v", "")) is None
    assert rewritten(two.replace("GROUP BY r.a, r.b, r.c", "GROUP BY r.a, r.c")) is None


PROOFS = [
    (HUB, f"SELECT p.k, p.n FROM p JOIN s ON p.k = s.k GROUP BY p.k, p.n {HAVING}"),
    (HUB, f"SELECT s.k, p.n FROM s JOIN p ON s.k = p.k GROUP BY s.k {HAVING}"),
    (HUB, f"SELECT p.k, p.n FROM p, s WHERE p.k = s.k GROUP BY p.k {HAVING}"),
    (HUB, f"SELECT DISTINCT p.k, p.n FROM p JOIN s ON p.k = s.k GROUP BY p.k {HAVING}"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # grouped on a column that is not the key: two products may share a name
    (HUB, f"SELECT p.k, p.n FROM p JOIN s ON p.k = s.k GROUP BY p.n {HAVING}"),
    # a left join keeps products with no sales when the HAVING reads nothing from them
    (HUB, "SELECT p.k, p.n FROM p LEFT JOIN s ON p.k = s.k GROUP BY p.k, p.n HAVING COUNT(*) >= 0"),
    # the other bound
    (HUB, f"SELECT p.k, p.n FROM p JOIN s ON p.k = s.k GROUP BY p.k, p.n {HAVING.replace('MAX(s.d) <= 5', 'MAX(s.d) < 5')}"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_without_the_key_nothing_is_proved():
    left, right = PROOFS[0]
    assert not proved(left, right, constraints={"p": TableConstraints(not_null=frozenset({"k"}))})


FLAT = [
    f"SELECT p.k, p.n FROM p JOIN s ON p.k = s.k GROUP BY p.k, p.n {HAVING}",
    f"SELECT s.k, p.n FROM s, p WHERE s.k = p.k GROUP BY s.k, p.n {HAVING}",
    f"SELECT p.k FROM p JOIN s ON s.k = p.k AND s.v > 0 WHERE p.o = 1 GROUP BY p.k {HAVING}",
    "SELECT p.k, p.o FROM p JOIN s ON p.k = s.k GROUP BY p.k, p.o HAVING COUNT(*) >= 2",
    "SELECT p.k FROM p JOIN s ON p.k = s.k WHERE s.d > 1 GROUP BY p.k HAVING SUM(s.v) > 2 OR MAX(s.d) = 3",
    "SELECT DISTINCT p.n FROM p JOIN s ON p.k = s.k GROUP BY p.k, p.n HAVING COUNT(s.v) > 0",
]


@pytest.mark.parametrize("sql", FLAT)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """On random tables (p's key is unique, s holds NULLs and repeats) the written and the rewritten query agree."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 2, 2, 3, 5, 6])  # noqa: E731
    for _ in range(200):
        ids = rng.sample(range(0, 5), rng.choice([0, 1, 2, 3, 5]))
        tables = {
            "p": [(i, value(), value()) for i in ids],
            "s": [(rng.choice([None, 0, 1, 2, 3, 4, 7]), value(), value()) for _ in range(rng.choice([0, 1, 3, 6, 9]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
