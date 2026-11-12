"""Under a key, the rows that hold their group's MIN or MAX are one row per group (``kumo_sql_prover.extreme_rows``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.extreme_rows import read_extreme_rows
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: (a, b) is a key (so a and b are NOT NULL); c and d may be NULL and repeat
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
KEYS = {"t": [("a", "b")]}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a", "b"}), keys=(("a", "b"),))}
FIRST = "x.b = (SELECT MIN(y.b) FROM t AS y WHERE y.a = x.a)"
FIRST_C = "x.c = (SELECT MIN(y.c) FROM t AS y WHERE y.d = x.d)"


def rewritten(sql, keys=KEYS, schema=SCHEMA):
    before = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    text = before.sql(dialect="mysql")
    after = read_extreme_rows(before, schema, keys).sql(dialect="mysql")
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


def test_distinct_over_one_row_per_group_is_dropped():
    assert rewritten(f"SELECT DISTINCT x.a, x.d FROM t AS x WHERE {FIRST}") == f"SELECT x.a, x.d FROM t AS x WHERE {FIRST}"
    assert rewritten(f"SELECT DISTINCT x.d, x.a AS k FROM t AS x WHERE x.c > 0 AND {FIRST}") == f"SELECT x.d, x.a AS k FROM t AS x WHERE x.c > 0 AND {FIRST}"
    # the subquery's own filter only changes which row is first
    filtered = "x.b = (SELECT MAX(y.b) FROM t AS y WHERE y.c > 0 AND y.a = x.a)"
    assert rewritten(f"SELECT DISTINCT a, d FROM t AS x WHERE {filtered}") == f"SELECT a, d FROM t AS x WHERE {filtered}"


def test_grouping_one_row_per_group_is_dropped():
    assert rewritten(f"SELECT x.a, x.d FROM t AS x WHERE {FIRST} GROUP BY x.a") == f"SELECT x.a, x.d FROM t AS x WHERE {FIRST}"
    assert rewritten(f"SELECT DISTINCT a, d FROM t AS x WHERE {FIRST} GROUP BY a") == f"SELECT a, d FROM t AS x WHERE {FIRST}"


def test_the_key_and_the_value_alone_are_the_grouped_query():
    grouped = "SELECT x.a, MIN(x.b) AS b FROM t AS x WHERE NOT x.a IS NULL GROUP BY x.a HAVING NOT MIN(x.b) IS NULL"
    assert rewritten(f"SELECT x.a, x.b FROM t AS x WHERE {FIRST}") == grouped
    assert rewritten(f"SELECT x.a, x.b FROM t AS x WHERE (SELECT MIN(b) FROM t AS y WHERE x.a = y.a) = b") == grouped
    assert rewritten(f"SELECT DISTINCT x.a, x.b FROM t AS x WHERE {FIRST}") == grouped
    assert rewritten(f"SELECT b + 1 AS n, a FROM t AS x WHERE {FIRST} ORDER BY x.b DESC") == (
        "SELECT MIN(x.b) + 1 AS n, a FROM t AS x WHERE NOT x.a IS NULL GROUP BY x.a HAVING NOT MIN(x.b) IS NULL ORDER BY MIN(x.b) DESC"
    )
    # the subquery's filter becomes the grouped query's WHERE
    assert rewritten("SELECT x.a, x.b FROM t AS x WHERE x.b = (SELECT MAX(y.b) FROM t AS y WHERE (y.c > 0 OR d = 1) AND y.a = x.a)") == (
        "SELECT x.a, MAX(x.b) AS b FROM t AS x WHERE (x.c > 0 OR x.d = 1) AND NOT x.a IS NULL GROUP BY x.a HAVING NOT MAX(x.b) IS NULL"
    )
    # without a key, DISTINCT is what makes it one row per group; it stays when the partition is not all listed
    assert rewritten(f"SELECT DISTINCT x.d, x.c FROM t AS x WHERE {FIRST_C}") == (
        "SELECT x.d, MIN(x.c) AS c FROM t AS x WHERE NOT x.d IS NULL GROUP BY x.d HAVING NOT MIN(x.c) IS NULL"
    )
    assert rewritten(f"SELECT DISTINCT x.c FROM t AS x WHERE {FIRST_C}") == (
        "SELECT DISTINCT MIN(x.c) AS c FROM t AS x WHERE NOT x.d IS NULL GROUP BY x.d HAVING NOT MIN(x.c) IS NULL"
    )


@pytest.mark.parametrize("sql", [
    f"SELECT x.a, x.d FROM t AS x WHERE {FIRST}",  # nothing to drop, and d is not the key or the value
    f"SELECT DISTINCT x.d FROM t AS x WHERE {FIRST}",  # the partition column is not listed: two groups may share d
    f"SELECT DISTINCT x.d, x.a + 1 FROM t AS x WHERE {FIRST}",  # not listed as a column
    f"SELECT DISTINCT x.d, x.c FROM t AS x WHERE {FIRST_C} AND x.a > 0",  # no key within (d, c): a filter on the row
    f"SELECT x.d, x.c FROM t AS x WHERE {FIRST_C}",  # no key within (d, c) and no DISTINCT: the first rows may repeat
    f"SELECT x.a, x.b FROM t AS x WHERE {FIRST} AND x.c > 0",  # the first row must pass c > 0: not a grouped query
    f"SELECT x.a, x.d FROM t AS x WHERE {FIRST} GROUP BY x.d",  # not the partition
    f"SELECT x.a, COUNT(*) FROM t AS x WHERE {FIRST} GROUP BY x.a",  # an aggregate reads the group
    f"SELECT x.a, x.b FROM t AS x WHERE {FIRST} LIMIT 1",
    f"SELECT x.a, x.b FROM t AS x WHERE {FIRST} OR x.c = 1",
    f"SELECT x.a, x.b FROM t AS x WHERE NOT {FIRST}",
    "SELECT x.a, x.b FROM t AS x WHERE x.b >= (SELECT MIN(y.b) FROM t AS y WHERE y.a = x.a)",
    "SELECT x.a, x.b FROM t AS x WHERE x.b = (SELECT MIN(y.b) FROM t AS y)",  # no partition: an empty table has no row but one group
    "SELECT x.a, x.b FROM t AS x WHERE x.b = (SELECT MIN(y.c) FROM t AS y WHERE y.a = x.a)",  # another column's minimum
    "SELECT x.a, x.b FROM t AS x WHERE x.b = (SELECT MIN(y.b) FROM t AS y WHERE y.a = x.d)",  # compared with another column
    "SELECT x.a, x.b FROM t AS x WHERE x.b = (SELECT MIN(y.b) FROM t AS y WHERE y.a = x.a AND y.c < x.c)",  # a filter that reads the outer row
    "SELECT x.a, x.b FROM t AS x WHERE x.b = (SELECT MIN(y.e) FROM u AS y WHERE y.a = x.a)",  # another table
    "SELECT x.a, x.b FROM t AS x WHERE x.b = (SELECT COUNT(y.b) FROM t AS y WHERE y.a = x.a)",
    "SELECT x.a, x.b FROM t AS x WHERE x.b = (SELECT MIN(y.b) FROM t AS y WHERE y.a = x.a GROUP BY y.a)",
    "SELECT t.a, t.b FROM t WHERE t.b = (SELECT MIN(t.b) FROM t WHERE t.a = t.a)",  # one name for both
    f"SELECT x.a, x.b FROM t AS x JOIN u ON u.a = x.a WHERE {FIRST}",
    f"SELECT x.a, x.b, (SELECT MAX(e) FROM u) FROM t AS x WHERE {FIRST}",
    f"SELECT x.a, x.b, COUNT(*) OVER () FROM t AS x WHERE {FIRST}",
    f"SELECT * FROM t AS x WHERE {FIRST}",
    f"SELECT x.a, x.b FROM t AS x WHERE {FIRST} AND x.c = (SELECT MIN(y.c) FROM t AS y WHERE y.a = x.a)",  # two comparisons
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_the_key_is_needed():
    for sql in (f"SELECT DISTINCT x.a, x.d FROM t AS x WHERE {FIRST}", f"SELECT x.a, x.d FROM t AS x WHERE {FIRST} GROUP BY x.a", f"SELECT x.a, x.b FROM t AS x WHERE {FIRST}"):
        assert rewritten(sql, keys=None) is None
        assert rewritten(sql, keys={"t": [("a", "b", "c")]}) is None
        assert rewritten(sql, keys={"t": [("b",)]}) is not None


PROOFS = [
    (f"SELECT DISTINCT x.a, x.d FROM t AS x WHERE {FIRST}", f"SELECT x.a, x.d FROM t AS x WHERE {FIRST}"),
    (f"SELECT x.a, x.d FROM t AS x WHERE {FIRST} GROUP BY x.a", "SELECT a, d FROM t WHERE (a, b) IN (SELECT a, MIN(b) FROM t GROUP BY a)"),
    (f"SELECT x.a, x.b FROM t AS x WHERE {FIRST}", "SELECT a, MIN(b) FROM t GROUP BY a"),
    ("SELECT a, b FROM (SELECT a, b, ROW_NUMBER() OVER (PARTITION BY a ORDER BY b DESC) AS r FROM t) AS s WHERE r = 1", "SELECT a, MAX(b) FROM t GROUP BY a"),
    ("SELECT DISTINCT x.a, x.d FROM t AS x JOIN (SELECT a, MIN(b) AS m FROM t GROUP BY a) AS g ON x.a = g.a AND x.b = g.m",
     "SELECT a, d FROM (SELECT a, d, RANK() OVER (PARTITION BY a ORDER BY b) AS r FROM t) AS s WHERE r = 1"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # without the key the first rows may repeat
    (f"SELECT DISTINCT x.d, x.a FROM t AS x WHERE {FIRST_C}", f"SELECT x.d, x.a FROM t AS x WHERE {FIRST_C}"),
    (f"SELECT x.d, x.c FROM t AS x WHERE {FIRST_C}", "SELECT d, MIN(c) FROM t GROUP BY d"),
    # c and d may be NULL: the grouped query keeps the NULL group
    (f"SELECT DISTINCT x.d, x.c FROM t AS x WHERE {FIRST_C}", "SELECT d, MIN(c) FROM t GROUP BY d"),
    # the first row against the last
    (f"SELECT x.a, x.b FROM t AS x WHERE {FIRST}", "SELECT a, MAX(b) FROM t GROUP BY a"),
    # the first row must pass the filter; the grouped query filters before choosing
    (f"SELECT x.a, x.b FROM t AS x WHERE {FIRST} AND x.c > 0", "SELECT a, MIN(b) FROM t WHERE c > 0 GROUP BY a"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_without_the_key_nothing_is_proved():
    left, right = PROOFS[2]
    assert not proved(left, right, constraints={"t": TableConstraints(not_null=frozenset({"a", "b"}))})


KEYED = [
    f"SELECT DISTINCT x.a, x.d FROM t AS x WHERE {FIRST}",
    f"SELECT DISTINCT x.d, x.a AS k FROM t AS x WHERE x.c > 0 AND {FIRST}",
    "SELECT DISTINCT a, d FROM t AS x WHERE x.b = (SELECT MAX(y.b) FROM t AS y WHERE y.c > 0 AND y.a = x.a)",
    f"SELECT x.a, x.d FROM t AS x WHERE {FIRST} GROUP BY x.a",
    f"SELECT x.a, x.b FROM t AS x WHERE {FIRST}",
    f"SELECT b + 1 AS n, a FROM t AS x WHERE {FIRST}",
    "SELECT x.a, x.b FROM t AS x WHERE x.b = (SELECT MAX(y.b) FROM t AS y WHERE (y.c > 0 OR d = 1) AND y.a = x.a)",
]
UNKEYED = [
    f"SELECT DISTINCT x.d, x.c FROM t AS x WHERE {FIRST_C}",
    f"SELECT DISTINCT x.c FROM t AS x WHERE {FIRST_C}",
    "SELECT DISTINCT x.c + x.d FROM t AS x WHERE x.c = (SELECT MAX(y.c) FROM t AS y WHERE y.a > 1 AND y.d = x.d)",
]


@pytest.mark.parametrize("sql", KEYED + UNKEYED)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the rewritten one on random tables in which (a, b) is a key; c and d hold NULLs and repeats."""

    after = rewritten(sql)
    assert after is not None
    # DuckDB has no MySQL-style column outside the GROUP BY; ANY_VALUE picks from the group's one row
    sql = sql.replace("SELECT x.a, x.d FROM t AS x WHERE x.b = (SELECT MIN(y.b) FROM t AS y WHERE y.a = x.a) GROUP BY", "SELECT x.a, ANY_VALUE(x.d) FROM t AS x WHERE x.b = (SELECT MIN(y.b) FROM t AS y WHERE y.a = x.a) GROUP BY")
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 2, 3])  # noqa: E731
    for _ in range(150):
        keys = {(rng.choice([1, 1, 2, 3]), rng.choice([0, 1, 2, 3, 4])) for _ in range(rng.choice([0, 1, 2, 4, 7]))}
        tables = {"t": [(a, b, value(), value()) for a, b in sorted(keys)]}
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
