"""A ``LEFT JOIN`` whose padded rows no kept group needs is an inner join (``kumo_sql_prover.counted_outer_joins``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.counted_outer_joins import counted_at_least_once, counted_outer_join_to_inner
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: (a, b) is a key and both are NOT NULL; c and d may be NULL and repeat. u has no key: its rows repeat and hold NULLs.
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a", "b"}), keys=(("a", "b"),))}


def rewritten(sql, schema=SCHEMA):
    tree = sqlglot.parse_one(sql, read="mysql")
    text = tree.sql(dialect="mysql")
    counted_outer_join_to_inner(tree, schema)
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


def test_a_lower_bound_on_a_count_of_the_padded_side_makes_the_join_inner():
    assert rewritten("SELECT x.a, COUNT(DISTINCT u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(DISTINCT u.e) > 0") == (
        "SELECT x.a, COUNT(DISTINCT u.e) FROM t AS x JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(DISTINCT u.e) > 0"
    )
    # another column of the padded side is counted: it is NULL in the padded rows too
    assert rewritten("SELECT x.a, COUNT(DISTINCT u.e) AS n FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(u.a) >= 1") == (
        "SELECT x.a, COUNT(DISTINCT u.e) AS n FROM t AS x JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(u.a) >= 1"
    )
    # through the select alias, which no table column shares
    assert rewritten("SELECT x.a, COUNT(u.e) AS n FROM t AS x LEFT OUTER JOIN u ON u.a = x.c GROUP BY x.a HAVING n <> 0 AND x.a > 1") == (
        "SELECT x.a, COUNT(u.e) AS n FROM t AS x JOIN u ON u.a = x.c GROUP BY x.a HAVING n <> 0 AND x.a > 1"
    )
    # every NULL-skipping aggregate of the padded side, an equality as the bound, a WHERE on the left side
    assert rewritten("SELECT x.a, SUM(u.e), MAX(u.a), MIN(u.e), AVG(u.e) FROM t AS x LEFT JOIN u ON u.a = x.c WHERE x.d > 0 GROUP BY x.a HAVING 2 = COUNT(u.e)") == (
        "SELECT x.a, SUM(u.e), MAX(u.a), MIN(u.e), AVG(u.e) FROM t AS x JOIN u ON u.a = x.c WHERE x.d > 0 GROUP BY x.a HAVING 2 = COUNT(u.e)"
    )
    # inner joins before it, group keys from both sides
    assert rewritten("SELECT u.a, x.d, COUNT(u.e) FROM t AS x JOIN t AS y ON y.a = x.a LEFT JOIN u ON u.a = x.c GROUP BY u.a, x.d HAVING COUNT(u.e) > 2") == (
        "SELECT u.a, x.d, COUNT(u.e) FROM t AS x JOIN t AS y ON y.a = x.a JOIN u ON u.a = x.c GROUP BY u.a, x.d HAVING COUNT(u.e) > 2"
    )
    # a derived table on the padded side
    assert rewritten("SELECT x.a FROM t AS x LEFT JOIN (SELECT a, MAX(e) AS m FROM u GROUP BY a) AS g ON g.a = x.c GROUP BY x.a HAVING COUNT(g.m) > 0") == (
        "SELECT x.a FROM t AS x JOIN (SELECT a, MAX(e) AS m FROM u GROUP BY a) AS g ON g.a = x.c GROUP BY x.a HAVING COUNT(g.m) > 0"
    )


@pytest.mark.parametrize("sql", [
    "SELECT x.a, COUNT(DISTINCT u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a",  # no bound: the all-padded groups stay, with 0
    "SELECT x.a, COUNT(DISTINCT u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(DISTINCT u.e) >= 0",
    "SELECT x.a, COUNT(DISTINCT u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(DISTINCT u.e) < 2",  # an upper bound
    "SELECT x.a, COUNT(DISTINCT u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(DISTINCT u.e) <> 1",
    "SELECT x.a, COUNT(DISTINCT u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(u.e) > 0 OR x.a = 1",
    "SELECT x.a, COUNT(DISTINCT u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING NOT COUNT(u.e) > 0",
    "SELECT x.a, COUNT(*) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(u.e) > 0",  # COUNT(*) counts the padded rows
    "SELECT x.a, COUNT(u.e), COUNT(x.c) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(u.e) > 0",  # so does a left-side count
    "SELECT x.a, COUNT(u.e), SUM(x.b) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(u.e) > 0",
    "SELECT x.a, COUNT(COALESCE(u.e, 0)) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(u.e) > 0",  # not NULL in a padded row
    "SELECT x.a, COUNT(u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(x.c) > 0",  # the bound is on the left side
    "SELECT x.a, x.d, COUNT(u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(u.e) > 0",  # d may come from a padded row
    "SELECT COUNT(u.e) FROM t AS x LEFT JOIN u ON u.a = x.c HAVING COUNT(u.e) > 0",  # no GROUP BY
    "SELECT x.a, COUNT(u.e) FROM u RIGHT JOIN t AS x ON u.a = x.c GROUP BY x.a HAVING COUNT(u.e) > 0",
    "SELECT x.a, COUNT(u.e) FROM t AS x FULL JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(u.e) > 0",
    "SELECT x.a, COUNT(u.e) FROM t AS x LEFT JOIN u ON u.a = x.c JOIN t AS y ON y.a = x.a GROUP BY x.a HAVING COUNT(u.e) > 0",  # not the last join
    "SELECT x.a, COUNT(u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(u.e) > 0 LIMIT 2",
    "SELECT x.a, COUNT(u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a WITH ROLLUP HAVING COUNT(u.e) > 0",
    "SELECT x.a, COUNT(u.e) OVER () FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a, u.e HAVING COUNT(u.e) > 0",
    "SELECT x.a, COUNT(u.e) AS c FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING c > 0",  # c is a column of t: not surely the alias
    "SELECT x.a, (SELECT MAX(e) FROM u), COUNT(u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(u.e) > 0",
    "SELECT x.a + 1, COUNT(u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a + 1 HAVING COUNT(u.e) > 0",
    "SELECT x.a, COUNT(u.e) FROM t AS x LEFT JOIN u USING (a) GROUP BY x.a HAVING COUNT(u.e) > 0",
    "SELECT x.a, COUNT(u.e) FROM t AS x JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(u.e) > 0",  # already inner
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_the_alias_is_read_only_when_no_column_can_have_its_name():
    sql = "SELECT x.a, COUNT(DISTINCT u.e) AS n FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING n > 0"
    assert rewritten(sql) is not None
    assert rewritten(sql, schema=None) is None
    assert rewritten(sql, schema={"t": ["a", "b", "c", "d"]}) is None  # the columns of u are not listed


def test_which_tests_bound_a_count_from_below():
    def bounded(test):
        having = sqlglot.parse_one(f"SELECT 1 FROM t GROUP BY a HAVING {test}", read="mysql").args["having"].this
        return counted_at_least_once(having, {}) is not None

    for test in ("COUNT(c) > 0", "COUNT(c) >= 1", "COUNT(c) <> 0", "COUNT(c) = 1", "COUNT(c) = 7", "COUNT(c) > 3", "0 < COUNT(c)", "1 <= COUNT(c)", "0 <> COUNT(c)", "(COUNT(DISTINCT c)) >= (2)"):
        assert bounded(test), test
    for test in ("COUNT(c) >= 0", "COUNT(c) = 0", "COUNT(c) <> 1", "COUNT(c) < 5", "COUNT(c) <= 5", "0 > COUNT(c)", "1 > COUNT(c)", "COUNT(c) + 1 > 1", "SUM(c) > 0", "COUNT(c) > d", "COUNT(c) > '0'"):
        assert not bounded(test), test


PROOFS = [
    ("SELECT x.a, COUNT(DISTINCT u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(DISTINCT u.e) > 0",
     "SELECT x.a, COUNT(DISTINCT u.e) FROM t AS x JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(DISTINCT u.e) > 0"),
    ("SELECT x.a, COUNT(u.e) AS n FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING n >= 1", "SELECT x.a, COUNT(u.e) FROM t AS x JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(u.e) > 0"),
    ("SELECT x.a, SUM(u.e), MAX(u.a) FROM t AS x LEFT JOIN u ON u.a = x.c WHERE x.d > 0 GROUP BY x.a HAVING COUNT(u.e) = 2",
     "SELECT x.a, SUM(u.e), MAX(u.a) FROM t AS x JOIN u ON u.a = x.c AND x.d > 0 GROUP BY x.a HAVING COUNT(u.e) = 2"),
    # y.b is NOT NULL: on the inner join the bound holds for every group
    ("SELECT x.a, COUNT(DISTINCT y.b) FROM t AS x LEFT JOIN t AS y ON y.a = x.c GROUP BY x.a HAVING COUNT(y.b) > 0", "SELECT x.a, COUNT(DISTINCT y.b) FROM t AS x JOIN t AS y ON y.a = x.c GROUP BY x.a"),
    # and with the left side read only to be matched, the grouped self-join is a membership test
    ("SELECT x.c, COUNT(DISTINCT y.b) FROM t AS x LEFT JOIN t AS y ON y.a = x.c GROUP BY x.c HAVING COUNT(y.b) > 0", "SELECT a, COUNT(*) FROM t WHERE a IN (SELECT c FROM t) GROUP BY a"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


def inner(sql):
    return sql.replace("LEFT JOIN", "JOIN")


NOT_PROVED = [
    # a group of unmatched rows is a row with 0 on the left join and no row on the inner join
    "SELECT x.a, COUNT(DISTINCT u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a",
    "SELECT x.a, COUNT(DISTINCT u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(DISTINCT u.e) >= 0",
    "SELECT x.a, COUNT(u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(u.e) < 2",
    "SELECT x.a, COUNT(u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(x.b) > 0",
    # a kept group counts its padded rows
    "SELECT x.a, COUNT(*) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(u.e) > 0",
    "SELECT x.a, COUNT(u.e), COUNT(x.b) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(u.e) > 0",
    # without GROUP BY and with a bound on the left side: one row against none when nothing matches
    "SELECT COUNT(u.e) FROM t AS x LEFT JOIN u ON u.a = x.c HAVING COUNT(x.a) > 0",
]


@pytest.mark.parametrize("sql", NOT_PROVED)
def test_pairs_that_are_not_proved(sql):
    assert not proved(sql, inner(sql))
    assert not proved(inner(sql), sql)


CASES = [
    "SELECT x.a, COUNT(DISTINCT u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(DISTINCT u.e) > 0",
    "SELECT x.a, COUNT(DISTINCT u.e) AS n FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.a HAVING COUNT(u.a) >= 1",
    "SELECT x.a, COUNT(u.e) AS n FROM t AS x LEFT OUTER JOIN u ON u.a = x.c GROUP BY x.a HAVING n <> 0 AND x.a > 1",
    "SELECT x.a, SUM(u.e), MAX(u.a), MIN(u.e), AVG(u.e) FROM t AS x LEFT JOIN u ON u.a = x.c WHERE x.d > 0 GROUP BY x.a HAVING 2 = COUNT(u.e)",
    "SELECT u.a, x.d, COUNT(u.e) FROM t AS x JOIN t AS y ON y.a = x.a LEFT JOIN u ON u.a = x.c GROUP BY u.a, x.d HAVING COUNT(u.e) > 1",
    "SELECT u.e, COUNT(u.a), SUM(DISTINCT u.a) FROM t AS x LEFT JOIN u ON u.a = x.c OR u.e = x.d GROUP BY u.e HAVING COUNT(DISTINCT u.a) > 0",
    "SELECT x.a FROM t AS x LEFT JOIN (SELECT a, MAX(e) AS m FROM u GROUP BY a) AS g ON g.a = x.c GROUP BY x.a HAVING COUNT(g.m) > 0",
    "SELECT DISTINCT x.d, COUNT(u.e) FROM t AS x LEFT JOIN u ON u.a = x.c GROUP BY x.d, x.a HAVING COUNT(u.e) > 0 AND MAX(u.e) < 3",
]


@pytest.mark.parametrize("sql", CASES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The left join and the inner join on random tables: unmatched rows, NULL join columns, NULL counted values, repeats."""

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


def test_the_refused_shapes_do_differ():
    """Why COUNT(*) and an unbounded HAVING are refused: one matched and one unmatched row in a group, and a group of unmatched rows."""

    tables = {"t": [(1, 0, 5, None), (1, 1, 9, None), (2, 0, 9, None)], "u": [(5, 7)]}
    for sql in NOT_PROVED[:6]:
        a, b = run(tables, sql, inner(sql))
        assert a != b, sql
