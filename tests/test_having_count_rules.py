"""Implied ``COUNT`` bounds dropped from ``HAVING``, and counted keys tested for NULL (``kumo_sql_prover.having_count_rules``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.having_count_rules import rewrite_count_tests

SCHEMA = {"t": ["a", "b", "c"]}


def rewritten(sql, dialect="mysql", schema=None):
    out = rewrite_count_tests(canonical_negation(sqlglot.parse_one(sql, read=dialect)), schema)
    return out.sql(dialect=dialect) if out is not None else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, compare_names=False, dialect="mysql").proven


def run(rows, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE t (a BIGINT, b BIGINT, c BIGINT)")
    insert_rows(db, "t", rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


def differ_on_duckdb(left, right, rows):
    a, b = run(rows, left, right)
    return a != b


def test_implied_bounds_are_dropped():
    assert rewritten("SELECT a FROM t GROUP BY a, c HAVING COUNT(*) > 1 AND COUNT(DISTINCT b) > 1") == (
        "SELECT a FROM t GROUP BY a, c HAVING COUNT(DISTINCT b) > 1"
    )
    assert rewritten("SELECT a FROM t GROUP BY a, c HAVING COUNT(b) >= 3 AND 2 <= COUNT(*) AND c > 0") == (
        "SELECT a FROM t GROUP BY a, c HAVING COUNT(b) >= 3 AND c > 0"
    )
    assert rewritten("SELECT a FROM t GROUP BY a, c HAVING COUNT(b) >= 2 AND COUNT(DISTINCT b) = 2") == (
        "SELECT a FROM t GROUP BY a, c HAVING COUNT(DISTINCT b) = 2"
    )
    assert rewritten("SELECT a FROM t GROUP BY a, c HAVING COUNT(*) >= 2 AND COUNT(*) > 4") == "SELECT a FROM t GROUP BY a, c HAVING COUNT(*) > 4"
    assert rewritten("SELECT a FROM t GROUP BY a, c HAVING COUNT(*) >= 2 AND COUNT(*) >= 2") == "SELECT a FROM t GROUP BY a, c HAVING COUNT(*) >= 2"
    assert rewritten("SELECT MAX(a) FROM t HAVING COUNT(*) > 0 AND COUNT(b) = 3") == "SELECT MAX(a) FROM t HAVING COUNT(b) = 3"


def test_counted_keys_are_tested_for_null():
    assert rewritten("SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1") == "SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1 AND NOT a IS NULL"
    assert rewritten("SELECT a, b FROM t GROUP BY a, b HAVING COUNT(DISTINCT b) = 1 AND SUM(c) > 2") == (
        "SELECT a, b FROM t GROUP BY a, b HAVING COUNT(DISTINCT b) = 1 AND SUM(c) > 2 AND NOT b IS NULL"
    )
    # once is enough, and a test already in WHERE counts
    assert rewritten("SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1 AND NOT a IS NULL") is None
    assert rewritten("SELECT a FROM t WHERE a IS NOT NULL GROUP BY a HAVING COUNT(a) > 1") is None


@pytest.mark.parametrize("sql", [
    "SELECT a FROM t GROUP BY a HAVING COUNT(*) > 1",  # COUNT(*) counts the NULL group's rows
    "SELECT a FROM t GROUP BY a HAVING COUNT(b) > 1",  # b is no key
    "SELECT a FROM t GROUP BY a HAVING COUNT(a) >= 0",  # true for the NULL group
    "SELECT a FROM t GROUP BY a HAVING COUNT(a) < 2",
    "SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1 OR SUM(b) > 3",  # not a conjunct
    "SELECT a FROM t GROUP BY a WITH ROLLUP HAVING COUNT(a) > 1",  # the total row has a NULL key and counts every a
    "SELECT b AS a FROM t GROUP BY t.a, b HAVING COUNT(a) > 1",  # the name is the select alias in HAVING
    "SELECT a FROM t GROUP BY a HAVING COUNT(*) >= 3 AND COUNT(DISTINCT b) >= 2",  # 2 different values need only 2 rows
    "SELECT a FROM t GROUP BY a HAVING COUNT(b) >= 2 AND COUNT(DISTINCT c) >= 2",  # another column
    "SELECT a FROM t GROUP BY a HAVING COUNT(DISTINCT b) >= 2 AND COUNT(b) >= 2 OR c > 1",
    "SELECT a FROM t GROUP BY a HAVING COUNT(*) = 2 AND COUNT(b) >= 2",  # an equality is never dropped
    "SELECT a FROM t GROUP BY a HAVING COUNT(*) <= 2 AND COUNT(b) <= 2",  # upper bounds run the other way
    "SELECT a FROM t GROUP BY a HAVING COUNT(*) FILTER (WHERE b > 1) >= 1 AND COUNT(b) >= 2",
])
def test_other_tests_stay(sql):
    assert rewritten(sql, "postgres" if "FILTER" in sql else "mysql") is None


def test_string_literals_in_a_counted_expression_keep_their_case():
    """``COUNT(DISTINCT NULLIF(c, 'A'))`` bounds ``COUNT(NULLIF(c, 'A'))`` from below, not ``COUNT(NULLIF(c, 'a'))``."""

    same = "SELECT a FROM t GROUP BY a HAVING COUNT(NULLIF(c, 'A')) >= 1 AND COUNT(DISTINCT NULLIF(c, 'A')) >= 1"
    other = "SELECT a FROM t GROUP BY a HAVING COUNT(NULLIF(c, 'a')) >= 1 AND COUNT(DISTINCT NULLIF(c, 'A')) >= 1"
    assert rewritten(same) == "SELECT a FROM t GROUP BY a HAVING COUNT(DISTINCT NULLIF(c, 'A')) >= 1"
    assert rewritten(other) is None


def test_other_spellings_of_a_counted_key_test_are_read():
    """The test is added to one query at a time, so each spelling of ``COUNT(a) > 1`` must get it."""

    added = " AND NOT a IS NULL"
    for sql in (
        "SELECT a FROM t GROUP BY a HAVING COUNT(t.a) > 1",  # one source: t.a is a
        "SELECT a FROM t GROUP BY a HAVING NOT COUNT(a) <= 1",
        "SELECT a FROM t GROUP BY a HAVING NOT COUNT(a) < 2",
        "SELECT a FROM t GROUP BY a HAVING COUNT(a) BETWEEN 1 AND 3",
        "SELECT a FROM t AS x GROUP BY a HAVING 1 < COUNT(x.a)",
    ):
        assert rewritten(sql) == sql + added, sql
    assert rewritten("SELECT t.a FROM t GROUP BY t.a HAVING COUNT(a) > 1") == "SELECT t.a FROM t GROUP BY t.a HAVING COUNT(a) > 1 AND NOT t.a IS NULL"
    # a select alias of the COUNT, when no table read has a column of that name
    aliased = "SELECT a, COUNT(a) AS n FROM t GROUP BY a HAVING n > 1"
    assert rewritten(aliased, schema=SCHEMA) == aliased + added
    assert rewritten("SELECT a, COUNT(a) AS n FROM t GROUP BY a HAVING n > 1 AND COUNT(*) >= 2", schema=SCHEMA) == aliased + added


@pytest.mark.parametrize("sql, schema", [
    ("SELECT a, COUNT(a) AS n FROM t GROUP BY a HAVING n > 1", None),  # n may be a column of t
    ("SELECT a, COUNT(a) AS c FROM t GROUP BY a HAVING c > 1", SCHEMA),  # c is a column of t
    ("SELECT a, COUNT(a) AS n FROM t JOIN u ON t.a = u.a GROUP BY a HAVING n > 1", SCHEMA),  # u's columns are not listed
    ("SELECT a, COUNT(a) AS n FROM (SELECT a FROM t) AS d GROUP BY a HAVING n > 1", SCHEMA),  # nor a derived table's
    ("SELECT a, COUNT(b) AS n FROM t GROUP BY a HAVING n > 1", SCHEMA),  # the alias counts another column
    ("SELECT t.a FROM t JOIN t AS x ON t.b = x.b GROUP BY t.a HAVING COUNT(a) > 1", None),  # two sources: a bare a is not known to be t.a
    ("SELECT t.a FROM t JOIN t AS x ON t.b = x.b GROUP BY t.a HAVING COUNT(x.a) > 1", None),
    ("SELECT a FROM t GROUP BY a HAVING NOT COUNT(a) >= 2", None),  # an upper bound
    ("SELECT a FROM t GROUP BY a HAVING NOT COUNT(a) = 0", None),  # not read
    ("SELECT a FROM t GROUP BY a HAVING COUNT(a) BETWEEN 0 AND 3", None),  # true for the NULL group
    ("SELECT a FROM t GROUP BY a HAVING COUNT(a) NOT BETWEEN 1 AND 3", None),
])
def test_other_spellings_that_stay(sql, schema):
    assert rewritten(sql, schema=schema) is None


PROOFS = [
    ("SELECT DISTINCT a FROM t GROUP BY c, a HAVING COUNT(DISTINCT b) > 1",
     "SELECT DISTINCT a FROM t GROUP BY c, a HAVING COUNT(*) > 1 AND COUNT(DISTINCT b) > 1"),
    ("SELECT a, SUM(c) FROM t GROUP BY a HAVING COUNT(b) >= 2", "SELECT a, SUM(c) FROM t GROUP BY a HAVING COUNT(b) >= 2 AND COUNT(*) >= 1"),
    ("SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1", "SELECT a FROM t WHERE a IS NOT NULL GROUP BY a HAVING COUNT(*) > 1"),
    ("SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1", "SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1 AND a IS NOT NULL"),
    ("SELECT a, MIN(b) FROM t GROUP BY a HAVING COUNT(DISTINCT a) = 1", "SELECT a, MIN(b) FROM t WHERE a IS NOT NULL GROUP BY a"),
    # one test spelled two ways: both sides get the key test, or neither (each was proved before the rule existed)
    ("SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1", "SELECT t.a FROM t GROUP BY t.a HAVING COUNT(a) > 1"),
    ("SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1", "SELECT a FROM t GROUP BY a HAVING COUNT(t.a) > 1"),
    ("SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1", "SELECT a FROM t GROUP BY a HAVING NOT COUNT(a) <= 1"),
    ("SELECT a, COUNT(a) AS n FROM t GROUP BY a HAVING COUNT(a) > 1", "SELECT a, COUNT(a) AS n FROM t GROUP BY a HAVING n > 1"),
    ("SELECT a FROM t GROUP BY a HAVING COUNT(a) = 1", "SELECT a FROM t GROUP BY a HAVING COUNT(a) BETWEEN 1 AND 1"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_target_pairs_are_proved(left, right):
    assert proved(left, right)


# Each pair differs on the rows given (checked below), so none may be proved.
NEAR_MISSES = [
    # COUNT(*) >= 3 is not implied by two different values
    ("SELECT a FROM t GROUP BY a HAVING COUNT(DISTINCT b) >= 2", "SELECT a FROM t GROUP BY a HAVING COUNT(*) >= 3 AND COUNT(DISTINCT b) >= 2", [(1, 1, 0), (1, 2, 0)]),
    # a bound on COUNT(*) says nothing about COUNT(b): the rows may hold NULLs
    ("SELECT a FROM t GROUP BY a HAVING COUNT(*) >= 2", "SELECT a FROM t GROUP BY a HAVING COUNT(*) >= 2 AND COUNT(b) >= 2", [(1, 1, 0), (1, None, 0)]),
    # nor does COUNT(b) bound COUNT(DISTINCT b)
    ("SELECT a FROM t GROUP BY a HAVING COUNT(b) >= 2", "SELECT a FROM t GROUP BY a HAVING COUNT(b) >= 2 AND COUNT(DISTINCT b) >= 2", [(1, 5, 0), (1, 5, 0)]),
    # COUNT(*) keeps the group of NULL keys that COUNT(a) drops
    ("SELECT a FROM t GROUP BY a HAVING COUNT(*) > 1", "SELECT a FROM t GROUP BY a HAVING COUNT(a) > 1", [(None, 1, 0), (None, 2, 0)]),
    ("SELECT a FROM t GROUP BY a HAVING COUNT(*) > 1", "SELECT a FROM t WHERE a IS NOT NULL GROUP BY a HAVING COUNT(*) > 1", [(None, 1, 0), (None, 2, 0)]),
    # a counted column that is no key may be NULL in a group that still passes on other rows
    ("SELECT a FROM t GROUP BY a HAVING COUNT(b) >= 1", "SELECT a FROM t WHERE b IS NOT NULL GROUP BY a HAVING COUNT(*) >= 2", [(1, 1, 0), (1, None, 0)]),
]


@pytest.mark.parametrize("left, right, rows", NEAR_MISSES)
def test_near_misses_differ_and_stay_unproved(left, right, rows):
    assert differ_on_duckdb(left, right, rows)
    assert not proved(left, right)


REWRITES = [
    "SELECT a, c FROM t GROUP BY a, c HAVING COUNT(*) > 1 AND COUNT(DISTINCT b) > 1",
    "SELECT a FROM t GROUP BY a HAVING COUNT(b) >= 2 AND COUNT(*) >= 2 AND COUNT(DISTINCT b) = 2",
    "SELECT MAX(a) FROM t HAVING COUNT(*) >= 1 AND COUNT(b) > 1",
    "SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(a) > 1",
    "SELECT a, b, SUM(c) FROM t GROUP BY a, b HAVING COUNT(DISTINCT b) >= 1 AND COUNT(a) = 2",
    "SELECT a FROM t WHERE c > 1 GROUP BY a HAVING 1 <= COUNT(a) AND COUNT(*) > 0",
    "SELECT a, COUNT(*) FROM t GROUP BY a HAVING NOT COUNT(t.a) <= 1 AND COUNT(*) >= 2",
    "SELECT t.a, SUM(b) FROM t GROUP BY t.a HAVING COUNT(DISTINCT a) BETWEEN 1 AND 2",
    "SELECT a, COUNT(a) AS n FROM t GROUP BY a HAVING n >= 2 AND NOT COUNT(*) < 2",
]


def test_rewrites_keep_every_result():
    """Original and rewritten select return the same bag on random small tables with NULLs and duplicate rows."""

    rng = random.Random(20261110)
    pairs = [(sql, rewritten(sql, schema=SCHEMA)) for sql in REWRITES]
    assert all(after is not None and after != before for before, after in pairs)
    for _ in range(60):
        rows = [tuple(rng.choice([None, 1, 2]) for _ in range(3)) for _ in range(rng.randint(0, 7))]
        rows += rows[:2]  # exact duplicate rows
        results = run(rows, *[sql for pair in pairs for sql in pair])
        for index, (before, after) in enumerate(pairs):
            assert results[2 * index] == results[2 * index + 1], (before, after, rows)
