"""A join that only asks for a match, under COUNT(DISTINCT), is a membership test (``kumo_sql_prover.join_distinct_aggregates``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.join_distinct_aggregates import count_distinct_under_key, read_join_distinct_aggregates
from kumo_sql_prover.smt_equivalence import TableConstraints

# f: (u, v) is a key (so u and v are NOT NULL); w may be NULL. p and q: nothing declared.
SCHEMA = {"f": ["u", "v", "w"], "p": ["x", "y"], "q": ["x", "z"]}
KEYS = {"f": [("u", "v")]}
NOT_NULL = {"f": frozenset({"u", "v"})}
CONSTRAINTS = {"f": TableConstraints(not_null=frozenset({"u", "v"}), keys=(("u", "v"),))}


def rewritten(sql):
    before = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    text = before.sql(dialect="mysql")
    after = read_join_distinct_aggregates(before, SCHEMA).sql(dialect="mysql")
    return after if after != text else None


def proved(left, right, constraints=CONSTRAINTS):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=constraints, compare_names=False, dialect="mysql").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE f (u BIGINT, v BIGINT, w BIGINT)")
    db.execute("CREATE TABLE p (x BIGINT, y BIGINT)")
    db.execute("CREATE TABLE q (x BIGINT, z BIGINT)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


def test_a_join_that_only_asks_for_a_match_is_a_membership_test():
    assert rewritten("SELECT p.x, COUNT(DISTINCT q.z) AS n FROM p JOIN q ON p.x = q.x GROUP BY p.x") == (
        "SELECT q.x, COUNT(DISTINCT q.z) AS n FROM q WHERE q.x IN (SELECT p.x FROM p) GROUP BY q.x"
    )
    # grouped on the other side's column, the join written the other way round
    assert rewritten("SELECT q.x AS k, MIN(q.z) FROM p INNER JOIN q ON q.x = p.x GROUP BY q.x") == (
        "SELECT q.x AS k, MIN(q.z) FROM q WHERE q.x IN (SELECT p.x FROM p) GROUP BY q.x"
    )
    # the aggregates decide which table is kept
    assert rewritten("SELECT q.x, MAX(p.y) FROM p JOIN q ON p.x = q.x GROUP BY q.x") == (
        "SELECT p.x, MAX(p.y) FROM p WHERE p.x IN (SELECT q.x FROM q) GROUP BY p.x"
    )


def test_the_output_keeps_its_name_and_a_filter_on_the_kept_table_stays():
    assert rewritten("SELECT p.y, COUNT(DISTINCT q.z) FROM q JOIN p ON q.z = p.y GROUP BY p.y") == (
        "SELECT q.z AS y, COUNT(DISTINCT q.z) FROM q WHERE q.z IN (SELECT p.y FROM p) GROUP BY q.z"
    )
    after = rewritten("SELECT a.u AS k, COUNT(DISTINCT b.w) FROM f AS a JOIN f AS b ON a.u = b.v WHERE b.w > 1 GROUP BY a.u")
    assert after == "SELECT b.v AS k, COUNT(DISTINCT b.w) FROM f AS b WHERE b.w > 1 AND b.v IN (SELECT a.u FROM f AS a) GROUP BY b.v"
    after = rewritten("SELECT a.u, COUNT(DISTINCT b.w) FROM f AS a JOIN f AS b ON a.u = b.v GROUP BY a.u")
    assert after == "SELECT b.v AS u, COUNT(DISTINCT b.w) FROM f AS b WHERE b.v IN (SELECT a.u FROM f AS a) GROUP BY b.v"


def test_a_left_join_needs_a_having_that_counts_a_value_of_the_null_side():
    sql = "SELECT a.u, COUNT(DISTINCT b.w) AS n FROM f AS a LEFT JOIN f AS b ON a.u = b.v GROUP BY a.u HAVING n > 0"
    assert rewritten(sql) == (
        "SELECT b.v AS u, COUNT(DISTINCT b.w) AS n FROM f AS b WHERE b.v IN (SELECT a.u FROM f AS a) GROUP BY b.v HAVING COUNT(DISTINCT b.w) > 0"
    )
    assert rewritten(sql.replace("n > 0", "COUNT(DISTINCT b.w) >= 1")) is not None
    assert rewritten(sql.replace("n > 0", "n <> 0")) is not None
    # no HAVING, or one that does not need a counted value: the unmatched rows of a make groups of their own
    assert rewritten(sql.replace(" HAVING n > 0", "")) is None
    assert rewritten(sql.replace("n > 0", "n >= 0")) is None
    assert rewritten(sql.replace("n > 0", "n > 1")) is None
    assert rewritten(sql.replace("COUNT(DISTINCT b.w) AS n", "COUNT(DISTINCT a.w) AS n")) is None  # counts the preserved side
    assert rewritten(sql.replace("n > 0", "n > 0 OR a.u = 1")) is None


@pytest.mark.parametrize("sql", [
    "SELECT p.x, COUNT(DISTINCT q.z) FROM p JOIN q ON p.x = q.x WHERE p.y = 1 GROUP BY p.x",  # a filter on the table that is not kept
    "SELECT p.x, COUNT(DISTINCT q.z) FROM p JOIN q ON p.x = q.x WHERE q.z = (SELECT MAX(z) FROM q) GROUP BY p.x",
    "SELECT p.x, COUNT(q.z) FROM p JOIN q ON p.x = q.x GROUP BY p.x",  # the repeats count
    "SELECT p.x, COUNT(*) FROM p JOIN q ON p.x = q.x GROUP BY p.x",
    "SELECT p.x, SUM(DISTINCT q.z) FROM p JOIN q ON p.x = q.x GROUP BY p.x",
    "SELECT p.x, COUNT(DISTINCT q.z), COUNT(DISTINCT p.y) FROM p JOIN q ON p.x = q.x GROUP BY p.x",  # aggregates over both tables
    "SELECT p.x, p.y, COUNT(DISTINCT q.z) FROM p JOIN q ON p.x = q.x GROUP BY p.x, p.y",
    "SELECT p.x, p.y, COUNT(DISTINCT q.z) FROM p JOIN q ON p.x = q.x GROUP BY p.x",  # a column of the other table outside an aggregate
    "SELECT p.x, COUNT(DISTINCT q.z) FROM p LEFT JOIN q ON p.x = q.x GROUP BY p.x",
    "SELECT p.x, COUNT(DISTINCT q.z) FROM p RIGHT JOIN q ON p.x = q.x GROUP BY p.x",
    "SELECT p.x, COUNT(DISTINCT q.z) FROM p JOIN q ON p.x < q.x GROUP BY p.x",
    "SELECT p.x, COUNT(DISTINCT q.z) FROM p JOIN q ON p.x = q.x AND p.y = q.z GROUP BY p.x",
    "SELECT p.x, COUNT(DISTINCT q.z) FROM p JOIN q ON p.x = q.x GROUP BY p.x LIMIT 3",
    "SELECT p.x, COUNT(DISTINCT q.z) FROM p JOIN q ON p.x = q.x GROUP BY p.x ORDER BY COUNT(DISTINCT q.z) LIMIT 3",
    "SELECT p.x, COUNT(DISTINCT q.z) FROM p JOIN q ON p.x = q.x",  # no grouping
    "SELECT p.x, COUNT(DISTINCT q.z) FROM p JOIN q ON p.x = q.x GROUP BY p.x HAVING MAX(p.y) > 1",
    "SELECT p.x, COUNT(DISTINCT q.z) FROM p JOIN q ON p.x = q.x JOIN p AS r ON r.x = q.x GROUP BY p.x",
    "SELECT p.x, COUNT(DISTINCT q.z) OVER () FROM p JOIN q ON p.x = q.x GROUP BY p.x",
    "SELECT p.x, COUNT(DISTINCT q.z + p.y) FROM p JOIN q ON p.x = q.x GROUP BY p.x",
    "SELECT * FROM p JOIN q ON p.x = q.x GROUP BY p.x",
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


KEYED_COUNTS = [
    ("SELECT u, COUNT(DISTINCT v) FROM f GROUP BY u", "SELECT u, COUNT(v) FROM f GROUP BY u"),
    ("SELECT u, COUNT(DISTINCT v) FROM f WHERE w > 0 GROUP BY u", "SELECT u, COUNT(v) FROM f WHERE w > 0 GROUP BY u"),
    ("SELECT x.v, COUNT(DISTINCT x.u) FROM f AS x GROUP BY x.v", "SELECT x.v, COUNT(x.u) FROM f AS x GROUP BY x.v"),
]


def counted(sql, keys=KEYS, not_null=NOT_NULL):
    select = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    out = count_distinct_under_key(select, keys, not_null)
    return out.sql(dialect="mysql") if out is not None else None


@pytest.mark.parametrize("sql, expected", KEYED_COUNTS)
def test_distinct_count_under_a_key_counts_the_rows(sql, expected):
    assert counted(sql) == expected


@pytest.mark.parametrize("sql", [
    "SELECT w, COUNT(DISTINCT v) FROM f GROUP BY w",  # (u, v) is not inside (w, v)
    "SELECT u, COUNT(DISTINCT w) FROM f GROUP BY u",
    "SELECT COUNT(DISTINCT v) FROM f",  # no group: v alone is not a key
    "SELECT u, COUNT(DISTINCT v) FROM f GROUP BY u, w + 1",
    "SELECT a.u, COUNT(DISTINCT a.v) FROM f AS a JOIN f AS b ON a.u = b.u GROUP BY a.u",  # a join repeats the rows
    "SELECT u, COUNT(DISTINCT v, w) FROM f GROUP BY u",
    "SELECT u, COUNT(DISTINCT v) OVER (PARTITION BY u) FROM f",
])
def test_distinct_count_without_the_key_stays(sql):
    assert counted(sql) is None


def test_every_key_column_must_be_not_null():
    sql = "SELECT u, COUNT(DISTINCT v) FROM f GROUP BY u"
    assert counted(sql, not_null={"f": frozenset({"u"})}) is None  # UNIQUE (u, v) lets two rows hold (1, NULL)
    assert counted(sql, not_null=None) is None
    assert counted(sql, keys=None) is None
    assert counted(sql, keys={"f": [("u", "v", "w")]}, not_null={"f": frozenset({"u", "v", "w"})}) is None


PROOFS = [
    # the second-degree followers of the LeetCode problem, in four spellings
    ("SELECT u AS k, COUNT(u) AS n FROM f WHERE u IN (SELECT v FROM f) GROUP BY u",
     "SELECT a.v AS k, COUNT(DISTINCT b.v) AS n FROM f AS a JOIN f AS b ON a.v = b.u GROUP BY a.v"),
    ("SELECT u AS k, COUNT(u) AS n FROM f WHERE u IN (SELECT v FROM f) GROUP BY u",
     "SELECT a.v AS k, COUNT(DISTINCT b.v) AS n FROM f AS a LEFT JOIN f AS b ON a.v = b.u GROUP BY a.v HAVING n > 0"),
    ("SELECT u AS k, COUNT(u) AS n FROM f WHERE u IN (SELECT v FROM f) GROUP BY u",
     "SELECT a.v AS k, COUNT(DISTINCT b.v) AS n FROM f AS a LEFT JOIN f AS b ON a.v = b.u WHERE b.v IS NOT NULL GROUP BY a.v"),
    ("SELECT u AS k, COUNT(u) AS n FROM f WHERE u IN (SELECT v FROM f) GROUP BY u",
     "SELECT u AS k, COUNT(DISTINCT v) AS n FROM f WHERE u IN (SELECT v FROM f) GROUP BY u"),
    ("SELECT x, MAX(z) FROM q WHERE x IN (SELECT x FROM p) GROUP BY x", "SELECT p.x, MAX(q.z) FROM p JOIN q ON p.x = q.x GROUP BY p.x"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # COUNT(q.z) counts the repeats the join makes
    ("SELECT x, COUNT(z) FROM q WHERE x IN (SELECT x FROM p) GROUP BY x", "SELECT p.x, COUNT(q.z) FROM p JOIN q ON p.x = q.x GROUP BY p.x"),
    # a left join keeps the unmatched rows of p with a count of 0
    ("SELECT x, COUNT(DISTINCT z) FROM q WHERE x IN (SELECT x FROM p) GROUP BY x", "SELECT p.x, COUNT(DISTINCT q.z) FROM p LEFT JOIN q ON p.x = q.x GROUP BY p.x"),
    # a filter on the other table
    ("SELECT x, COUNT(DISTINCT z) FROM q WHERE x IN (SELECT x FROM p) GROUP BY x", "SELECT p.x, COUNT(DISTINCT q.z) FROM p JOIN q ON p.x = q.x WHERE p.y = 1 GROUP BY p.x"),
    # without NOT NULL on w the distinct count and the row count differ
    ("SELECT u, COUNT(DISTINCT w) FROM f GROUP BY u", "SELECT u, COUNT(w) FROM f GROUP BY u"),
    # a key that does not cover the group and the value
    ("SELECT w, COUNT(DISTINCT v) FROM f GROUP BY w", "SELECT w, COUNT(v) FROM f GROUP BY w"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_a_nullable_key_proves_nothing():
    left, right = PROOFS[3]
    assert not proved(left, right, constraints={"f": TableConstraints(not_null=frozenset({"u"}), keys=(("u", "v"),))})


JOINS = [
    "SELECT p.x, COUNT(DISTINCT q.z) AS n FROM p JOIN q ON p.x = q.x GROUP BY p.x",
    "SELECT q.x AS k, MIN(q.z) FROM p INNER JOIN q ON q.x = p.x GROUP BY q.x",
    "SELECT q.x, MAX(p.y) FROM p JOIN q ON p.x = q.x GROUP BY q.x",
    "SELECT p.x, COUNT(DISTINCT p.y), MIN(p.y) FROM p JOIN q ON p.x = q.x WHERE p.y > 0 GROUP BY p.x",
    "SELECT p.x, COUNT(DISTINCT q.z) AS n FROM p LEFT JOIN q ON p.x = q.x GROUP BY p.x HAVING n > 0",
    "SELECT p.x FROM p JOIN q ON p.x = q.x GROUP BY p.x",
]


@pytest.mark.parametrize("sql", JOINS)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """On random tables with NULLs and repeats the written and the rewritten query return the same rows."""

    after = rewritten(sql)
    if after is None:
        pytest.skip("not rewritten")
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 2, 3])  # noqa: E731
    for _ in range(200):
        tables = {
            "p": [(value(), value()) for _ in range(rng.choice([0, 1, 2, 4, 6]))],
            "q": [(value(), value()) for _ in range(rng.choice([0, 1, 2, 4, 6]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
