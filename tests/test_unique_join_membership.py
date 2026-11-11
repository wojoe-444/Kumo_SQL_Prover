"""An inner join with a grouped derived table on all of its group keys is a membership test (``kumo_sql_prover.unique_join_membership``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.unique_join_membership import read_unique_joins
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: (a, b) is a key and both are NOT NULL; c and d may be NULL and repeat. u has no key: its rows repeat and hold NULLs.
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a", "b"}), keys=(("a", "b"),))}


def rewritten(sql, schema=SCHEMA):
    tree = sqlglot.parse_one(sql, read="mysql")
    text = tree.sql(dialect="mysql")
    after = read_unique_joins(tree, schema).sql(dialect="mysql")
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


def random_tables(rng):
    value = lambda: rng.choice([None, None, 1, 1, 2, 3])  # noqa: E731
    keys = {(rng.choice([1, 1, 2, 3]), rng.choice([0, 1, 2, 3, 4])) for _ in range(rng.choice([0, 1, 2, 4, 7]))}
    return {
        "t": [(a, b, value(), value()) for a, b in sorted(keys)],
        "u": [(value(), value()) for _ in range(rng.choice([0, 1, 3, 6]))],
    }


G = "(SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1) AS g"
N = "(SELECT c AS k, COUNT(*) AS n FROM t GROUP BY c) AS g"


def test_a_join_on_every_key_becomes_a_membership_test():
    assert rewritten(f"SELECT t.a, t.b FROM t JOIN {G} ON t.c = g.c") == "SELECT t.a, t.b FROM t WHERE t.c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1)"
    # two keys, tied in WHERE through a comma join; the compared columns follow the derived table's select list
    assert rewritten("SELECT x.a, e FROM t AS x, u, (SELECT d, c FROM t GROUP BY c, d) AS g WHERE x.c = g.c AND u.e = g.d") == (
        "SELECT x.a, e FROM t AS x, u WHERE (u.e, x.c) IN (SELECT d, c FROM t GROUP BY c, d)"
    )
    # an aggregate output tied as well: still at most one row of the derived table per row of t
    assert rewritten("SELECT t.a, t.b FROM t JOIN (SELECT c AS k, MAX(d) AS m FROM t GROUP BY c) AS g ON t.c = g.k AND t.d = g.m") == (
        "SELECT t.a, t.b FROM t WHERE (t.c, t.d) IN (SELECT c AS k, MAX(d) AS m FROM t GROUP BY c)"
    )


def test_one_join_at_a_time_and_the_other_conditions_stay():
    sql = (
        "SELECT SUM(x.b) FROM t AS x JOIN (SELECT c, d FROM t GROUP BY c, d HAVING COUNT(*) = 1) AS g ON x.c = g.c AND g.d = x.d "
        "JOIN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1) AS h ON h.c = x.c WHERE x.b > 0"
    )
    assert rewritten(sql) == (
        "SELECT SUM(x.b) FROM t AS x WHERE x.b > 0 AND x.c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1) "
        "AND (x.c, x.d) IN (SELECT c, d FROM t GROUP BY c, d HAVING COUNT(*) = 1)"
    )


def test_conditions_on_the_derived_table_alone_move_into_the_test():
    assert rewritten(f"SELECT t.a, t.b FROM t JOIN {N} ON t.c = g.k WHERE g.n > 1 AND t.b > 0") == (
        f"SELECT t.a, t.b FROM t WHERE t.b > 0 AND t.c IN (SELECT g.k FROM {N} WHERE g.n > 1)"
    )
    # the derived table first: the next source takes its place and its ON conditions become filters
    assert rewritten(f"SELECT t.a, t.b FROM {N} JOIN t ON t.c = g.k AND t.b > 0 JOIN u ON u.a = t.a WHERE g.n > 1") == (
        f"SELECT t.a, t.b FROM t JOIN u ON u.a = t.a WHERE t.b > 0 AND t.c IN (SELECT g.k FROM {N} WHERE g.n > 1)"
    )
    # an output that is not read is not compared
    assert rewritten("SELECT t.a FROM t INNER JOIN (SELECT c, d, COUNT(*) AS n FROM t GROUP BY c, d) AS g ON t.c = g.c AND t.d = g.d") == (
        "SELECT t.a FROM t WHERE (t.c, t.d) IN (SELECT g.c, g.d FROM (SELECT c, d, COUNT(*) AS n FROM t GROUP BY c, d) AS g)"
    )


@pytest.mark.parametrize("sql", [
    f"SELECT t.a, t.b FROM t LEFT JOIN {G} ON t.c = g.c",  # keeps the rows without a partner
    "SELECT t.a, t.b FROM t JOIN (SELECT c FROM t GROUP BY c) AS g ON t.c = g.c LEFT JOIN u ON u.a = t.a",  # an ON that is not a filter
    f"SELECT t.a, g.c FROM t JOIN {G} ON t.c = g.c",  # the derived table is read
    f"SELECT t.a, g.n FROM t JOIN {N} ON t.c = g.k",
    f"SELECT t.a FROM t JOIN {G} ON t.c = g.c GROUP BY t.a, g.c",
    f"SELECT t.a FROM t JOIN {G} ON t.c = g.c ORDER BY g.c",
    f"SELECT * FROM t JOIN {G} ON t.c = g.c",
    "SELECT t.a FROM t JOIN (SELECT c, d FROM t GROUP BY c, d) AS g ON t.c = g.c",  # one key of two: several partners
    "SELECT t.a FROM t JOIN (SELECT c, COUNT(*) AS n FROM t GROUP BY c, d) AS g ON t.c = g.c",  # a key that is not an output
    f"SELECT t.a FROM t JOIN {N} ON t.b = g.n",  # tied on the count only: groups with the same count
    f"SELECT t.a FROM t JOIN {G} ON t.c > g.c",
    f"SELECT t.a FROM t JOIN {G} ON t.c = g.c + 0",
    f"SELECT t.a FROM t JOIN {G} ON t.c = g.c OR t.d = g.c",
    "SELECT a FROM u JOIN (SELECT c AS e FROM t GROUP BY c) AS g ON u.a = g.e WHERE e > 0",  # the bare e may be the derived table's
    "SELECT t.a FROM t JOIN (SELECT DISTINCT c FROM t) AS g ON t.c = g.c",  # not grouped
    "SELECT t.a FROM t JOIN (SELECT c FROM t GROUP BY c LIMIT 2) AS g ON t.c = g.c",
    "SELECT t.a FROM t JOIN (SELECT c FROM t GROUP BY c WITH ROLLUP) AS g ON t.c = g.c",
    f"SELECT t.a FROM t JOIN {G} ON t.c = g.c WHERE EXISTS (SELECT 1 FROM u WHERE u.a = g.c)",  # read by a nested select
    f"SELECT u.a FROM u JOIN {G} ON u.e = g.c WHERE EXISTS (SELECT 1 FROM t AS x JOIN t AS y ON x.a = y.a WHERE c = 1)",  # a bare c the nested select may not bind
    f"SELECT t.a FROM t JOIN {G} ON t.c = g.c WHERE g.c IN (SELECT a FROM u)",
    f"SELECT t.a FROM t JOIN {G} USING (c)",
    f"SELECT t.a FROM t NATURAL JOIN {G}",
    f"SELECT t.a FROM t, {G}",  # no tie
    f"SELECT g.c FROM {G}",
    "SELECT t.a FROM t JOIN u ON t.a = u.a",
])
def test_other_joins_stay(sql):
    assert rewritten(sql) is None


PROOFS = [
    (f"SELECT t.a, t.b FROM t JOIN {G} ON t.c = g.c", "SELECT a, b FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1)"),
    (
        "SELECT SUM(x.b) FROM t AS x JOIN (SELECT c, d FROM t GROUP BY c, d HAVING COUNT(*) = 1) AS g ON x.c = g.c AND g.d = x.d "
        "JOIN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1) AS h ON h.c = x.c",
        "SELECT SUM(b) FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1) AND (c, d) IN (SELECT c, d FROM t GROUP BY c, d HAVING COUNT(*) = 1)",
    ),
    (f"SELECT t.a, t.b FROM t JOIN {N} ON t.c = g.k WHERE g.n > 1", "SELECT a, b FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1)"),
    (f"SELECT t.a, t.b FROM {N}, t WHERE g.n = 2 AND g.k = t.c", "SELECT a, b FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) = 2)"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # one key of two: a row of t joins once per d of its c
    ("SELECT t.a, t.b FROM t JOIN (SELECT c, d FROM t GROUP BY c, d) AS g ON t.c = g.c", "SELECT a, b FROM t WHERE c IN (SELECT c FROM t GROUP BY c, d)"),
    # tied on the count: once per group with that count
    (f"SELECT t.a, t.b FROM t JOIN {N} ON t.b = g.n", "SELECT a, b FROM t WHERE b IN (SELECT COUNT(*) FROM t GROUP BY c)"),
    # the outer join keeps every row of t
    (f"SELECT t.a, t.b FROM t LEFT JOIN {G} ON t.c = g.c", "SELECT a, b FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1)"),
    # another bound
    (f"SELECT t.a, t.b FROM t JOIN {N} ON t.c = g.k WHERE g.n > 1", "SELECT a, b FROM t WHERE c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) >= 1)"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_the_refused_joins_do_differ():
    tables = {"t": [(1, 1, 7, 1), (1, 2, 7, 2), (2, 1, 8, None), (3, 1, 9, None)]}
    for left, right in NOT_PROVED:
        a, b = run(tables, left, right)
        assert a != b, (left, right)


CASES = [
    f"SELECT t.a, t.b FROM t JOIN {G} ON t.c = g.c",
    "SELECT x.a, x.b, u.e FROM t AS x, u, (SELECT d, c FROM t GROUP BY c, d) AS g WHERE x.c = g.c AND u.e = g.d",
    "SELECT t.a, t.b FROM t JOIN (SELECT c AS k, MAX(d) AS m FROM t GROUP BY c) AS g ON t.c = g.k AND t.d = g.m",
    "SELECT SUM(x.b), COUNT(*) FROM t AS x JOIN (SELECT c, d FROM t GROUP BY c, d HAVING COUNT(*) = 1) AS g ON x.c = g.c AND g.d = x.d "
    "JOIN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1) AS h ON h.c = x.c WHERE x.b > 0",
    f"SELECT t.a, t.b FROM t JOIN {N} ON t.c = g.k WHERE g.n > 1 AND t.b > 0",
    f"SELECT t.a, t.b, u.e FROM {N} JOIN t ON t.c = g.k AND t.b > 0 JOIN u ON u.a = t.a WHERE g.n > 1",
    "SELECT t.a, COUNT(*) FROM t INNER JOIN (SELECT c, d, COUNT(*) AS n FROM t GROUP BY c, d) AS g ON t.c = g.c AND t.d = g.d GROUP BY t.a",
    "SELECT u.a, u.e FROM u JOIN (SELECT a AS k, e, SUM(e) AS s FROM u GROUP BY a, e) AS g ON u.a = g.k AND u.e = g.e AND u.e = g.k WHERE g.s > 1 OR g.e = 1",
]


@pytest.mark.parametrize("sql", CASES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The join and the membership test on random tables whose tied columns hold NULLs and repeat."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    for _ in range(150):
        tables = random_tables(rng)
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
