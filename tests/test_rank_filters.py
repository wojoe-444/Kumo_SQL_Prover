"""``RANK() .. = 1`` and ``v = MIN(v) OVER ..`` read as a comparison with the partition's extreme (``kumo_sql_prover.rank_filters``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.rank_filters import read_rank_filters
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: a and b are NOT NULL and (a, b) is a key; c and d may be NULL and repeat
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
NOT_NULL = {"t": frozenset({"a", "b"})}
KEYS = {"t": [("a", "b")]}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a", "b"}), keys=(("a", "b"),))}


def rewritten(sql, not_null=NOT_NULL, keys=KEYS, schema=SCHEMA):
    before = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    text = before.sql(dialect="mysql")
    after = read_rank_filters(before, schema, not_null, keys).sql(dialect="mysql")
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


FIRST = "SELECT a, c FROM (SELECT a, c FROM t WHERE t.b = (SELECT MIN(kqr0.b) FROM t AS kqr0 WHERE kqr0.a = t.a)) AS x"
LAST = FIRST.replace("MIN", "MAX")


def test_a_rank_of_one_keeps_the_rows_with_the_smallest_value():
    assert rewritten("SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 1") == FIRST
    assert rewritten("SELECT a, c FROM (SELECT a, c, DENSE_RANK() OVER (PARTITION BY a ORDER BY b ASC) AS r FROM t) AS x WHERE x.r = 1") == FIRST
    assert rewritten("SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY t.a ORDER BY t.b DESC) AS r FROM t) AS x WHERE 1 = r") == LAST
    assert rewritten("SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r <= 1") == FIRST
    assert rewritten("SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r < 2") == FIRST
    # the key (a, b) makes row 1 the only first row
    assert rewritten("SELECT a, c FROM (SELECT a, c, ROW_NUMBER() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 1") == FIRST


def test_the_rest_of_both_filters_is_kept():
    assert rewritten(
        "SELECT x.c FROM (SELECT *, RANK() OVER (PARTITION BY a ORDER BY b DESC) AS r FROM t AS s WHERE c > 0 OR d = 1) AS x "
        "JOIN u ON u.a = x.a WHERE u.e > 2 AND x.r = 1 AND x.d IS NULL"
    ) == (
        "SELECT x.c FROM (SELECT * FROM t AS s WHERE (c > 0 OR d = 1) AND s.b = (SELECT MAX(kqr0.b) FROM t AS kqr0 "
        "WHERE (kqr0.c > 0 OR kqr0.d = 1) AND kqr0.a = s.a)) AS x JOIN u ON u.a = x.a WHERE u.e > 2 AND x.d IS NULL"
    )
    # no partition: the whole table's first rows
    assert rewritten("SELECT a FROM (SELECT a, RANK() OVER (ORDER BY b) AS r FROM t) AS x WHERE r = 1") == (
        "SELECT a FROM (SELECT a FROM t WHERE t.b = (SELECT MIN(kqr0.b) FROM t AS kqr0)) AS x"
    )


def test_a_value_equal_to_its_windowed_extreme():
    by_c = "SELECT a, c FROM (SELECT a, c FROM t WHERE t.c = (SELECT MIN(kqr0.c) FROM t AS kqr0 WHERE kqr0.a = t.a)) AS x"
    assert rewritten("SELECT a, c FROM (SELECT a, c, MIN(c) OVER (PARTITION BY a) AS m FROM t) AS x WHERE c = m") == by_c
    assert rewritten("SELECT a, c FROM (SELECT a, c, MIN(c) OVER (PARTITION BY a ORDER BY c) AS m FROM t) AS x WHERE x.m = x.c") == by_c
    assert rewritten("SELECT a, c FROM (SELECT a, c, MAX(c) OVER (PARTITION BY a ORDER BY c DESC) AS m FROM t) AS x WHERE c = m") == by_c.replace("MIN", "MAX")
    # through a renamed column, and FIRST_VALUE of a NOT NULL column
    assert rewritten("SELECT a, w FROM (SELECT a, b AS w, FIRST_VALUE(b) OVER (PARTITION BY a ORDER BY b) AS m FROM t) AS x WHERE w = m") == (
        "SELECT a, w FROM (SELECT a, b AS w FROM t WHERE t.b = (SELECT MIN(kqr0.b) FROM t AS kqr0 WHERE kqr0.a = t.a)) AS x"
    )


@pytest.mark.parametrize("sql", [
    "SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY c) AS r FROM t) AS x WHERE r = 1",  # c may be NULL: a NULL sorts first on some engines
    "SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY c ORDER BY b) AS r FROM t) AS x WHERE r = 1",  # the NULL partition
    "SELECT a, c FROM (SELECT a, c, ROW_NUMBER() OVER (PARTITION BY a ORDER BY b) AS r FROM u AS t) AS x WHERE r = 1",  # not t's columns
    "SELECT a, c FROM (SELECT a, c, ROW_NUMBER() OVER (ORDER BY b) AS r FROM t) AS x WHERE r = 1",  # b alone is no key: ties are picked arbitrarily
    "SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 2",
    "SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r <= 2",
    "SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 1 OR c = 2",
    "SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE NOT r = 1",
    "SELECT a, r FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 1",  # the rank is output
    "SELECT * FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 1",
    "SELECT a FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 1 ORDER BY r",
    "SELECT a FROM (SELECT a, RANK() OVER (PARTITION BY a ORDER BY b) AS r, COUNT(*) OVER (PARTITION BY a) AS n FROM t) AS x WHERE r = 1",  # another window
    "SELECT a FROM (SELECT a, RANK() OVER (PARTITION BY a ORDER BY b, c) AS r FROM t) AS x WHERE r = 1",  # two sort keys
    "SELECT a FROM (SELECT a, RANK() OVER (PARTITION BY a ORDER BY b + 1) AS r FROM t) AS x WHERE r = 1",
    "SELECT a FROM (SELECT a, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t GROUP BY a, b) AS x WHERE r = 1",  # ranks groups, not rows
    "SELECT a FROM (SELECT DISTINCT a, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 1",
    "SELECT a FROM (SELECT a, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t LIMIT 3) AS x WHERE r = 1",
    "SELECT a FROM (SELECT t.a, RANK() OVER (PARTITION BY t.a ORDER BY t.b) AS r FROM t JOIN u ON t.a = u.a) AS x WHERE r = 1",  # a join inside
    "SELECT u.e FROM u LEFT JOIN (SELECT a, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x ON u.a = x.a WHERE x.r = 1",  # below an outer join
    "SELECT a FROM (SELECT a, RANK() OVER (PARTITION BY a ORDER BY b ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS r FROM t) AS x WHERE r = 1",
    "SELECT a FROM (SELECT a, c, MIN(c) OVER (PARTITION BY a ORDER BY c DESC) AS m FROM t) AS x WHERE c = m",  # a running minimum from the top
    "SELECT a FROM (SELECT a, c, MIN(c) OVER (PARTITION BY a ORDER BY b) AS m FROM t) AS x WHERE c = m",  # a running minimum in another order
    "SELECT a FROM (SELECT a, c, MIN(c) OVER (PARTITION BY a) AS m FROM t) AS x WHERE d = m",  # another column
    "SELECT a FROM (SELECT a, c + 1 AS c, MIN(c) OVER (PARTITION BY a) AS m FROM t) AS x WHERE c = m",  # c outside is not t.c
    "SELECT a FROM (SELECT a, c, MIN(c) OVER (PARTITION BY a) AS m FROM t) AS x WHERE c >= m",
    "SELECT a FROM (SELECT a, c, FIRST_VALUE(c) OVER (PARTITION BY a ORDER BY c) AS m FROM t) AS x WHERE c = m",  # c may be NULL
    "SELECT a FROM (SELECT a, c, FIRST_VALUE(b) OVER (PARTITION BY a ORDER BY c) AS m FROM t) AS x WHERE b = m",  # ordered by another column
    "SELECT a FROM (SELECT a, c, SUM(c) OVER (PARTITION BY a) AS m FROM t) AS x WHERE c = m",
    "SELECT a FROM (SELECT a, RANK() OVER (PARTITION BY a ORDER BY b) AS c FROM t) AS x WHERE c = 1",  # the name is one of t's columns
    "SELECT x.a FROM (SELECT a, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x, u WHERE r = 1",  # r may be u's
    "SELECT a FROM (SELECT a, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 1 AND EXISTS (SELECT 1 FROM u WHERE u.e = x.r)",
])
def test_other_windows_stay(sql):
    assert rewritten(sql) is None


def test_facts_are_needed():
    ranked = "SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 1"
    numbered = ranked.replace("RANK()", "ROW_NUMBER()")
    assert rewritten(ranked, not_null=None) is None
    assert rewritten(ranked, not_null={"t": frozenset({"a"})}) is None
    assert rewritten(ranked, not_null={"t": frozenset({"b"})}) is None
    assert rewritten(numbered, keys=None) is None
    assert rewritten(numbered, keys={"t": [("a", "b", "c")]}) is None
    assert rewritten(numbered, keys={"t": [("b",)]}) is not None
    assert rewritten(ranked, schema=None) is None


PROOFS = [
    ("SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 1",
     "SELECT t.a, t.c FROM t JOIN (SELECT a, MIN(b) AS m FROM t GROUP BY a) AS g ON t.a = g.a AND t.b = g.m"),
    ("SELECT a, c FROM (SELECT a, c, DENSE_RANK() OVER (PARTITION BY a ORDER BY b DESC) AS r FROM t) AS x WHERE r = 1",
     "SELECT a, c FROM t WHERE (a, b) IN (SELECT a, MAX(b) FROM t GROUP BY a)"),
    ("SELECT a, c FROM (SELECT a, c, ROW_NUMBER() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 1",
     "SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 1"),
    ("SELECT a, c FROM (SELECT a, c, MIN(c) OVER (PARTITION BY a) AS m FROM t) AS x WHERE c = m",
     "SELECT s.a, s.c FROM t AS s WHERE s.c = (SELECT MIN(y.c) FROM t AS y WHERE y.a = s.a)"),
    ("WITH x AS (SELECT *, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t WHERE c > 0) SELECT a, d FROM x WHERE r = 1",
     "SELECT s.a, s.d FROM t AS s WHERE s.c > 0 AND s.b = (SELECT MIN(y.b) FROM t AS y WHERE y.c > 0 AND y.a = s.a)"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # first against last
    ("SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 1",
     "SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b DESC) AS r FROM t) AS x WHERE r = 1"),
    # the window sees the filtered rows; a filter applied after it does not change the ranks
    ("SELECT a, d FROM (SELECT a, d, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t WHERE c > 0) AS x WHERE r = 1",
     "SELECT a, d FROM (SELECT a, c, d, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 1 AND c > 0"),
    # partitioned against not
    ("SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 1",
     "SELECT a, c FROM (SELECT a, c, RANK() OVER (ORDER BY b) AS r FROM t) AS x WHERE r = 1"),
    # the second row is not the first
    ("SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 2",
     "SELECT a, c FROM t WHERE (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a)"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_without_the_declared_facts_nothing_is_proved():
    left, right = PROOFS[0]
    assert not proved(left, right, constraints=None)
    numbered = PROOFS[2][0]
    grouped = PROOFS[0][1]
    assert not proved(numbered, grouped, constraints={"t": TableConstraints(not_null=frozenset({"a", "b"}))})


QUERIES = [
    "SELECT a, c FROM (SELECT a, c, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS x WHERE r = 1",
    "SELECT a, c FROM (SELECT a, c, DENSE_RANK() OVER (PARTITION BY a ORDER BY b DESC) AS r FROM t) AS x WHERE r <= 1",
    "SELECT a, c FROM (SELECT a, c, ROW_NUMBER() OVER (PARTITION BY a ORDER BY b DESC) AS r FROM t) AS x WHERE r < 2",
    "SELECT a FROM (SELECT a, RANK() OVER (ORDER BY b) AS r FROM t WHERE c > 0 OR d = 1) AS x WHERE r = 1",
    "SELECT x.c, u.e FROM (SELECT *, RANK() OVER (PARTITION BY a ORDER BY b DESC) AS r FROM t AS s WHERE c > 0) AS x JOIN u ON u.a = x.a WHERE u.e > 0 AND x.r = 1",
    "SELECT a, c FROM (SELECT a, c, MIN(c) OVER (PARTITION BY a) AS m FROM t) AS x WHERE c = m",
    "SELECT a, c FROM (SELECT a, c, MIN(c) OVER (PARTITION BY a ORDER BY c) AS m FROM t) AS x WHERE x.m = x.c",
    "SELECT a, c FROM (SELECT a, c, MAX(c) OVER (PARTITION BY a ORDER BY c DESC) AS m FROM t WHERE d > 0) AS x WHERE c = m",
    "SELECT a, c FROM (SELECT a, c, MAX(c) OVER () AS m FROM t) AS x WHERE c = m",
    "SELECT a, w FROM (SELECT a, b AS w, FIRST_VALUE(b) OVER (PARTITION BY a ORDER BY b DESC) AS m FROM t) AS x WHERE w = m",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the rewritten one on random tables that meet the declared facts (a, b NOT NULL, key (a, b))."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 2, 3])  # noqa: E731
    for _ in range(150):
        keys = {(rng.choice([1, 1, 2, 3]), rng.choice([0, 1, 2, 3, 4])) for _ in range(rng.choice([0, 1, 2, 4, 7]))}
        tables = {
            "t": [(a, b, value(), value()) for a, b in sorted(keys)],
            "u": [(value(), value()) for _ in range(rng.choice([0, 1, 3]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
