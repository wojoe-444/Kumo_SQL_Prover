"""A source joined only to be matched is an ``IN`` test under a select blind to repeats (``kumo_sql_prover.membership_joins``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.membership_joins import join_to_membership, positivity_test, repeat_blind
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: (a, b) is a key and both are NOT NULL; c and d may be NULL and repeat. u has no key: its rows repeat and hold NULLs.
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a", "b"}), keys=(("a", "b"),))}
TYPES = {"t": {"a": "INT", "b": "INT", "c": "INT", "d": "INT"}, "u": {"a": "VARCHAR", "e": "INT"}}


def rewritten(sql, types=None):
    tree = sqlglot.parse_one(sql, read="mysql")
    after = join_to_membership(tree, SCHEMA, types)
    return None if after is None else after.sql(dialect="mysql")


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


def test_a_grouped_self_join_counts_distinct_values():
    assert rewritten("SELECT x.c, COUNT(DISTINCT y.b) FROM t AS x JOIN t AS y ON x.c = y.a GROUP BY x.c") == (
        "SELECT y.a AS c, COUNT(DISTINCT y.b) FROM t AS y WHERE y.a IN (SELECT x.c FROM t AS x) GROUP BY y.a"
    )
    # comma join, the link in WHERE, grouped by the other side of the link
    assert rewritten("SELECT y.a, COUNT(DISTINCT y.b) AS n FROM t AS x, t AS y WHERE x.c = y.a GROUP BY y.a") == (
        "SELECT y.a, COUNT(DISTINCT y.b) AS n FROM t AS y WHERE y.a IN (SELECT x.c FROM t AS x) GROUP BY y.a"
    )
    # MIN and MAX read the set of values; a COUNT only asked whether it is positive does too
    assert rewritten("SELECT x.a, MAX(x.c), MIN(x.d) FROM t AS x JOIN u ON x.b = u.a GROUP BY x.a HAVING COUNT(x.c) > 0 ORDER BY x.a") == (
        "SELECT x.a, MAX(x.c), MIN(x.d) FROM t AS x WHERE x.b IN (SELECT u.a FROM u) GROUP BY x.a HAVING COUNT(x.c) > 0 ORDER BY x.a"
    )
    # no GROUP BY: one group, over the same set of rows
    assert rewritten("SELECT COUNT(DISTINCT x.b) FROM t AS x JOIN u ON u.a = x.a") == "SELECT COUNT(DISTINCT x.b) FROM t AS x WHERE x.a IN (SELECT u.a FROM u)"


def test_select_distinct_reads_the_set_of_rows():
    assert rewritten("SELECT DISTINCT x.a, x.d FROM t AS x JOIN u ON u.e = x.d") == "SELECT DISTINCT x.a, x.d FROM t AS x WHERE x.d IN (SELECT u.e FROM u)"
    # the matched column read in the select list is the column it equals, under its output name
    assert rewritten("SELECT DISTINCT u.e, x.a FROM t AS x JOIN u ON u.e = x.d") == "SELECT DISTINCT x.d AS e, x.a FROM t AS x WHERE x.d IN (SELECT u.e FROM u)"
    # conditions on the matched source alone move into the test
    assert rewritten("SELECT DISTINCT x.a, x.d FROM t AS x JOIN u ON u.e = x.d AND u.a > 1 WHERE x.c > 0") == (
        "SELECT DISTINCT x.a, x.d FROM t AS x WHERE x.c > 0 AND x.d IN (SELECT u.e FROM u WHERE u.a > 1)"
    )
    # a derived table is matched the same way
    assert rewritten("SELECT DISTINCT x.a FROM t AS x JOIN (SELECT a, MAX(e) AS m FROM u GROUP BY a) AS g ON g.m = x.b") == (
        "SELECT DISTINCT x.a FROM t AS x WHERE x.b IN (SELECT g.m FROM (SELECT a, MAX(e) AS m FROM u GROUP BY a) AS g)"
    )
    # the last source that qualifies goes; the other joins stay joins
    assert rewritten("SELECT x.a, COUNT(DISTINCT x.b) FROM t AS x JOIN u AS v ON v.a = x.a JOIN u AS w ON w.e = x.c GROUP BY x.a") == (
        "SELECT x.a, COUNT(DISTINCT x.b) FROM t AS x, u AS v WHERE v.a = x.a AND x.c IN (SELECT w.e FROM u AS w) GROUP BY x.a"
    )
    # a subquery that reads the other source does not stop the matched one from going
    assert rewritten("SELECT DISTINCT x.a FROM t AS x JOIN u ON u.a = x.a WHERE EXISTS (SELECT 1 FROM t AS z WHERE z.a = u.e)") == (
        "SELECT DISTINCT u.a AS a FROM u WHERE EXISTS(SELECT 1 FROM t AS z WHERE z.a = u.e) AND u.a IN (SELECT x.a FROM t AS x)"
    )


@pytest.mark.parametrize("sql", [
    "SELECT x.a, COUNT(x.b) FROM t AS x JOIN u ON u.a = x.a GROUP BY x.a",  # COUNT sees the repeats
    "SELECT x.a, SUM(x.b) FROM t AS x JOIN u ON u.a = x.a GROUP BY x.a",
    "SELECT x.a, COUNT(DISTINCT x.b) FROM t AS x JOIN u ON u.a = x.a GROUP BY x.a HAVING COUNT(*) > 1",  # a bound that counts repeats
    "SELECT x.a, COUNT(DISTINCT x.b) FROM t AS x JOIN u ON u.a = x.a GROUP BY x.a HAVING COUNT(x.b) >= 2",
    "SELECT x.a, COUNT(DISTINCT x.b), COUNT(x.b) > 0, COUNT(x.b) + 0 FROM t AS x JOIN u ON u.a = x.a GROUP BY x.a",
    "SELECT x.a FROM t AS x JOIN u ON u.a = x.a",  # no DISTINCT: each match is a row
    "SELECT DISTINCT x.a FROM t AS x LEFT JOIN u ON u.a = x.a",  # unmatched rows stay
    "SELECT DISTINCT x.a FROM u LEFT JOIN t AS x ON u.a = x.a",
    "SELECT DISTINCT x.a FROM t AS x JOIN u ON u.a = x.a AND u.e = x.b",  # two ties to the other source
    "SELECT DISTINCT x.a FROM t AS x JOIN u ON u.a > x.a",  # not an equality
    "SELECT DISTINCT x.a FROM t AS x JOIN u ON u.a = x.a + 1",  # not a column
    "SELECT DISTINCT x.a FROM t AS x JOIN u ON u.a = x.a OR u.e = 1",
    "SELECT DISTINCT x.a FROM t AS x JOIN u ON u.a = x.a LIMIT 3",
    "SELECT DISTINCT x.a FROM t AS x JOIN u USING (a)",
    "SELECT DISTINCT x.a FROM t AS x NATURAL JOIN u",
    "SELECT DISTINCT * FROM t AS x JOIN u ON u.a = x.a",
    "SELECT DISTINCT x.a, RAND() FROM t AS x JOIN u ON u.a = x.a",
    "SELECT DISTINCT x.a, ROW_NUMBER() OVER (ORDER BY x.a) FROM t AS x JOIN u ON u.a = x.a",
    "SELECT x.a, COUNT(DISTINCT x.b) FROM t AS x JOIN u ON u.a = x.a GROUP BY x.a WITH ROLLUP",
    "SELECT x.a, x.d, COUNT(DISTINCT x.b) FROM t AS x JOIN u ON u.a = x.a GROUP BY x.a",  # d is picked from an arbitrary row
    "SELECT x.a + 1, COUNT(DISTINCT x.b) FROM t AS x JOIN u ON u.a = x.a GROUP BY x.a + 1",  # grouped by an expression
    "SELECT DISTINCT a FROM t AS x JOIN u ON u.e = x.d",  # a bare column the rule cannot attribute
    "SELECT DISTINCT x.a, x.b, u.e FROM t AS x JOIN u ON u.a = x.a",  # both sources are read
    "SELECT DISTINCT x.a FROM t AS x",  # nothing joined
    "SELECT DISTINCT x.a FROM t AS x JOIN u ON u.a = x.a WHERE u.e IN (SELECT z.b FROM t AS z WHERE z.a = x.b)",  # x read from a subquery, u tested with one
    "SELECT x.a, COUNT(DISTINCT x.b) FROM t AS x JOIN u ON u.a = x.a GROUP BY x.a HAVING x.a IN (SELECT e FROM u)",
    "SELECT x.a, GROUP_CONCAT(DISTINCT x.b) FROM t AS x JOIN u ON u.a = x.a GROUP BY x.a",
    "SELECT DISTINCT x.a FROM t AS x JOIN t AS x ON x.a = x.b",  # one name for two sources
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_a_matched_column_is_read_as_its_partner_only_within_one_type():
    # x.a is INT and u.a is VARCHAR: MySQL finds 1 = '1' and '01', which are different outputs
    assert rewritten("SELECT DISTINCT x.a, u.e FROM t AS x JOIN u ON u.a = x.a", TYPES) is None
    assert rewritten("SELECT DISTINCT x.a, u.e FROM t AS x JOIN u ON u.a = x.a") == "SELECT DISTINCT u.a AS a, u.e FROM u WHERE u.a IN (SELECT x.a FROM t AS x)"
    assert rewritten("SELECT DISTINCT x.d, u.a FROM t AS x JOIN u ON u.e = x.d", TYPES) == "SELECT DISTINCT u.e AS d, u.a FROM u WHERE u.e IN (SELECT x.d FROM t AS x)"
    floating = {"t": dict(TYPES["t"], d="DOUBLE"), "u": dict(TYPES["u"], e="DOUBLE")}
    assert rewritten("SELECT DISTINCT x.d, u.a FROM t AS x JOIN u ON u.e = x.d", floating) is None  # 0.0 = -0.0
    # the comparison itself is the same comparison in both forms, whatever the types: nothing else reads u
    assert rewritten("SELECT DISTINCT x.b FROM t AS x JOIN u ON u.a = x.a", TYPES) == "SELECT DISTINCT x.b FROM t AS x WHERE x.a IN (SELECT u.a FROM u)"


def test_which_selects_are_blind_to_repeats():
    def blind(sql):
        return repeat_blind(sqlglot.parse_one(sql, read="mysql"))

    assert blind("SELECT DISTINCT x.a, x.b + 1 FROM t AS x")
    assert blind("SELECT x.a, MIN(x.b), COUNT(DISTINCT x.c) FROM t AS x GROUP BY x.a HAVING 0 < COUNT(x.d)")
    assert blind("SELECT DISTINCT x.a, MAX(x.b) FROM t AS x GROUP BY x.a")
    assert not blind("SELECT x.a FROM t AS x")
    assert not blind("SELECT x.a FROM t AS x GROUP BY x.a HAVING COUNT(*) > 1")
    assert not blind("SELECT DISTINCT x.a, COUNT(*) FROM t AS x GROUP BY x.a")  # DISTINCT dedups the output, not what COUNT reads
    assert not blind("SELECT x.a, x.d, MAX(x.b) FROM t AS x GROUP BY x.a")  # d comes from an arbitrary row
    assert not blind("SELECT x.a, MAX(x.b), (SELECT MAX(e) FROM u WHERE u.a = x.d) FROM t AS x GROUP BY x.a")
    count = lambda sql: sqlglot.parse_one(sql, read="mysql").find(sqlglot.exp.Count)  # noqa: E731
    for test in ("COUNT(c) > 0", "COUNT(c) >= 1", "COUNT(c) <> 0", "COUNT(c) = 0", "COUNT(c) < 1", "COUNT(c) <= 0", "0 < COUNT(c)", "1 <= COUNT(c)", "(COUNT(c)) > (0)"):
        assert positivity_test(count(f"SELECT 1 FROM t HAVING {test}")), test
    for test in ("COUNT(c) > 1", "COUNT(c) >= 2", "COUNT(c) = 1", "COUNT(c) <> 1", "COUNT(c) < 2", "COUNT(c) >= 0", "COUNT(c) < 0", "0 > COUNT(c)", "COUNT(c) + 1 > 0", "COUNT(c) > '0'", "COUNT(c) > d"):
        assert not positivity_test(count(f"SELECT 1 FROM t HAVING {test}")), test


PROOFS = [
    ("SELECT x.c, COUNT(DISTINCT y.b) FROM t AS x JOIN t AS y ON x.c = y.a GROUP BY x.c", "SELECT a, COUNT(*) FROM t WHERE a IN (SELECT c FROM t) GROUP BY a"),
    ("SELECT y.a, COUNT(DISTINCT y.b) FROM t AS x, t AS y WHERE x.c = y.a GROUP BY y.a", "SELECT a, COUNT(b) FROM t WHERE EXISTS (SELECT 1 FROM t AS z WHERE z.c = t.a) GROUP BY a"),
    ("SELECT DISTINCT x.a, x.d FROM t AS x JOIN u ON u.e = x.d", "SELECT DISTINCT a, d FROM t WHERE d IN (SELECT e FROM u)"),
    ("SELECT x.a, MAX(x.c) FROM t AS x JOIN u ON x.b = u.a WHERE u.e > 1 GROUP BY x.a", "SELECT a, MAX(c) FROM t WHERE b IN (SELECT a FROM u WHERE e > 1) GROUP BY a"),
    ("SELECT DISTINCT u.e, x.a, x.b FROM t AS x JOIN u ON u.e = x.d", "SELECT d, a, b FROM t WHERE d IN (SELECT e FROM u)"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # a COUNT, a SUM or a plain select sees each match
    ("SELECT x.a, COUNT(x.b) FROM t AS x JOIN u ON u.a = x.a GROUP BY x.a", "SELECT a, COUNT(b) FROM t WHERE a IN (SELECT a FROM u) GROUP BY a"),
    ("SELECT x.a, SUM(x.c) FROM t AS x JOIN u ON u.a = x.a GROUP BY x.a", "SELECT a, SUM(c) FROM t WHERE a IN (SELECT a FROM u) GROUP BY a"),
    ("SELECT x.a FROM t AS x JOIN u ON u.a = x.a", "SELECT a FROM t WHERE a IN (SELECT a FROM u)"),
    ("SELECT x.a, COUNT(DISTINCT x.b) FROM t AS x JOIN u ON u.a = x.a GROUP BY x.a HAVING COUNT(*) > 1", "SELECT a, COUNT(DISTINCT b) FROM t WHERE a IN (SELECT a FROM u) GROUP BY a HAVING COUNT(*) > 1"),
    # the left join keeps the unmatched rows
    ("SELECT DISTINCT x.a FROM t AS x LEFT JOIN u ON u.a = x.a", "SELECT DISTINCT a FROM t WHERE a IN (SELECT a FROM u)"),
    ("SELECT x.a, COUNT(DISTINCT u.e) FROM t AS x LEFT JOIN u ON u.a = x.a GROUP BY x.a", "SELECT a, 1 FROM t WHERE a IN (SELECT a FROM u) GROUP BY a"),
    # matched on another column
    ("SELECT DISTINCT x.a, x.d FROM t AS x JOIN u ON u.e = x.d", "SELECT DISTINCT a, d FROM t WHERE d IN (SELECT a FROM u)"),
    # the distinct values of c in a group are not its rows (c repeats and may be NULL)
    ("SELECT x.a, COUNT(DISTINCT x.c) FROM t AS x JOIN u ON u.a = x.a GROUP BY x.a", "SELECT a, COUNT(*) FROM t WHERE a IN (SELECT a FROM u) GROUP BY a"),
    # NOT IN is another test
    ("SELECT DISTINCT x.a, x.d FROM t AS x JOIN u ON u.e <> x.d", "SELECT DISTINCT a, d FROM t WHERE d NOT IN (SELECT e FROM u)"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_the_key_decides_whether_the_distinct_count_is_the_row_count():
    left, right = PROOFS[0]
    assert not proved(left, right, constraints={"t": TableConstraints(not_null=frozenset({"a", "b"}))})


CASES = [
    "SELECT x.c, COUNT(DISTINCT y.b) FROM t AS x JOIN t AS y ON x.c = y.a GROUP BY x.c",
    "SELECT y.a, COUNT(DISTINCT y.b) AS n, SUM(DISTINCT y.c), MIN(y.d) FROM t AS x, t AS y WHERE x.c = y.a GROUP BY y.a",
    "SELECT x.a, MAX(x.c), MIN(x.d) FROM t AS x JOIN u ON x.b = u.a GROUP BY x.a HAVING COUNT(x.c) > 0",
    "SELECT x.d, AVG(DISTINCT x.c) FROM t AS x JOIN u ON x.d = u.e GROUP BY x.d HAVING COUNT(x.c) = 0 OR MAX(x.b) > 1",
    "SELECT COUNT(DISTINCT x.b), MAX(x.c) FROM t AS x JOIN u ON u.a = x.a",
    "SELECT DISTINCT x.a, x.d FROM t AS x JOIN u ON u.e = x.d",
    "SELECT DISTINCT u.e, x.a FROM t AS x JOIN u ON u.e = x.d",
    "SELECT DISTINCT x.a, x.d FROM t AS x JOIN u ON u.e = x.d AND u.a > 1 WHERE x.c > 0",
    "SELECT DISTINCT x.a FROM t AS x JOIN (SELECT a, MAX(e) AS m FROM u GROUP BY a) AS g ON g.m = x.b",
    "SELECT x.a, COUNT(DISTINCT x.b) FROM t AS x JOIN u AS v ON v.a = x.a JOIN u AS w ON w.e = x.c GROUP BY x.a",
    "SELECT DISTINCT x.a FROM t AS x JOIN u ON u.a = x.a WHERE EXISTS (SELECT 1 FROM t AS z WHERE z.a = u.e)",
    "SELECT DISTINCT x.c, u.e FROM t AS x CROSS JOIN u WHERE u.a = x.c AND x.d IS NULL",
    "SELECT u.e, COUNT(DISTINCT u.a) FROM u JOIN t AS x ON x.d = u.e GROUP BY u.e",
]


@pytest.mark.parametrize("sql", CASES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The join and the membership test on random tables: (a, b) is a key of t, everything else holds NULLs and repeats."""

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
