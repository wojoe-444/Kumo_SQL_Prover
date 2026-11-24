"""A sum of FLOAT64 partial sums stays unproved however the partial sum is handed on: through ``SELECT *``, a
column list, an unnamed item, a set operation's positions or a ``WITH`` name read twice (float_regrouping.py).

Each of the first spellings was proved equal to the flat sum before: the check looked a column up by its name in
the derived table's select list and found no sum there."""

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import float_regrouping
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic

SCHEMA = {"t": ["x", "f", "i"]}
PARTIAL = "(SELECT x, SUM(f) AS s FROM t GROUP BY x)"
WITH = f"WITH d AS {PARTIAL} "

# the flat sum on the left, the same additions in another grouping on the right
SPELLINGS = {
    "star": ("SELECT SUM(f) FROM t", f"SELECT SUM(s) FROM (SELECT * FROM {PARTIAL} AS e) AS d"),
    "qualified star": ("SELECT SUM(f) FROM t", f"SELECT SUM(s) FROM (SELECT e.* FROM {PARTIAL} AS e) AS d"),
    "two stars": ("SELECT SUM(f) FROM t", f"SELECT SUM(s) FROM (SELECT * FROM (SELECT * FROM {PARTIAL} AS e) AS c) AS d"),
    "star over a WITH name": ("SELECT SUM(f) FROM t", f"WITH e AS {PARTIAL}, d AS (SELECT * FROM e) SELECT SUM(s) FROM d"),
    "column list": ("SELECT SUM(f) FROM t", f"SELECT SUM(v) FROM {PARTIAL} AS d (k, v)"),
    "column list on an unnamed sum": ("SELECT SUM(f) FROM t", "SELECT SUM(v) FROM (SELECT SUM(f) FROM t GROUP BY x) AS d (v)"),
    "column list on a WITH name": ("SELECT SUM(f) FROM t", "WITH d (v) AS (SELECT SUM(f) AS s FROM t GROUP BY x) SELECT SUM(v) FROM d"),
    "column list that swaps two names": ("SELECT SUM(i) FROM t AS u (x, i, f)", "SELECT SUM(s) FROM (SELECT SUM(i) AS s FROM t AS u (x, i, f) GROUP BY x) AS d"),
    "a WITH name read twice": (WITH + "SELECT SUM(f) AS v FROM t UNION ALL SELECT MAX(s) FROM d", WITH + "SELECT SUM(s) AS v FROM d UNION ALL SELECT MAX(s) FROM d"),
}
# the same spellings over the INT64 column, whose additions are exact in any grouping
EXACT = {
    "star": ("SELECT SUM(i) FROM t", "SELECT SUM(s) FROM (SELECT * FROM (SELECT x, SUM(i) AS s FROM t GROUP BY x) AS e) AS d"),
    "column list": ("SELECT SUM(i) FROM t", "SELECT SUM(v) FROM (SELECT x, SUM(i) AS s FROM t GROUP BY x) AS d (k, v)"),
    "column list that swaps two names": ("SELECT SUM(f) FROM t AS u (x, i, f)", "SELECT SUM(s) FROM (SELECT SUM(f) AS s FROM t AS u (x, i, f) GROUP BY x) AS d"),
}


def types(dialect: str) -> dict:
    return {"t": {"x": "INT64", "f": "DOUBLE PRECISION" if dialect == "postgres" else "FLOAT64", "i": "INT64"}}


def prove(pair, dialect):
    return prove_equivalent_algebraic(*pair, schema=SCHEMA, types=types(dialect), compare_names=False, dialect=dialect, exact_arithmetic=True, timeout_ms=3000)


@pytest.mark.parametrize("dialect", ["bigquery", "postgres", "duckdb"])
@pytest.mark.parametrize("name", list(SPELLINGS))
def test_a_regrouped_float_sum_is_not_proved(name, dialect):
    assert not prove(SPELLINGS[name], dialect).proven


@pytest.mark.parametrize("dialect", ["bigquery", "postgres"])
@pytest.mark.parametrize("name", list(EXACT))
def test_the_same_spelling_over_whole_numbers_is_proved(name, dialect):
    assert prove(EXACT[name], dialect).proven


def reads(sql: str, order=None, read="postgres") -> bool:
    return float_regrouping.sum_of_sums(sqlglot.parse_one(sql, read=read), types("bigquery"), order)


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT SUM(s) FROM {PARTIAL} AS d",
        f"SELECT SUM(s) FROM (SELECT * FROM {PARTIAL} AS e) AS d",
        f"SELECT SUM(s) FROM (SELECT e.*, 1 AS one FROM {PARTIAL} AS e) AS d",
        f"SELECT SUM(v) FROM {PARTIAL} AS d (k, v)",
        f"SELECT SUM(v) FROM (SELECT * FROM {PARTIAL} AS e) AS d (k, v)",  # positions are not known past a star
        "SELECT SUM(d.sum) FROM (SELECT SUM(f) FROM t GROUP BY x) AS d",  # the name the engine gives the item
        "WITH d (k, v) AS " + PARTIAL + " SELECT SUM(v) FROM d",
        "WITH d (k, v) AS " + PARTIAL + " SELECT SUM(w) FROM d AS e (j, w)",  # two column lists: not followed
        "WITH RECURSIVE d AS " + PARTIAL + " SELECT SUM(s) FROM d",  # not followed
        f"SELECT SUM(s) FROM t AS k, LATERAL (SELECT SUM(f) AS s FROM t WHERE t.x = k.x) AS d",  # not followed
        # a set operation gives its columns by position: the second branch's sum is read under the first branch's name
        "SELECT SUM(v) FROM (SELECT i AS v FROM t UNION ALL SELECT SUM(f) AS z FROM t GROUP BY x) AS d",
        "SELECT SUM(v) FROM (SELECT x AS k, i AS v FROM t UNION ALL SELECT x, SUM(f) FROM t GROUP BY x) AS d",
        # the sum inside the argument itself, and inside an item of the derived table
        "SELECT SUM((SELECT SUM(f) FROM t WHERE t.x = k.x)) FROM (SELECT DISTINCT x FROM t) AS k",
        "SELECT SUM(s) FROM (SELECT (SELECT SUM(f) FROM t WHERE t.x = k.x) AS s FROM (SELECT DISTINCT x FROM t) AS k) AS d",
        # read from the select around it
        f"SELECT (SELECT SUM(d.s + u.i) FROM t AS u) FROM {PARTIAL} AS d",
        # a literal column of a UNION whose other branch is a FLOAT64 column is not a column of literals
        "SELECT SUM(s) FROM (SELECT SUM(a) AS s FROM (SELECT 1 AS a FROM t UNION ALL SELECT f AS z FROM t) AS u GROUP BY a) AS d",
        "SELECT SUM(s) FROM (SELECT SUM(a) AS s FROM (SELECT 1 AS b, f AS a FROM t) AS u (a, b) GROUP BY b) AS d",
    ],
)
def test_the_partial_sum_is_found(sql):
    assert reads(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SUM(f) FROM t",
        "SELECT SUM(f) FROM (SELECT * FROM t) AS d",
        "SELECT SUM(f) FROM (SELECT x, f FROM t) AS d (k, f)",
        f"SELECT SUM(x) FROM {PARTIAL} AS d",  # the grouping column, not the sum
        f"SELECT SUM(k) FROM {PARTIAL} AS d (k, v)",
        "SELECT SUM(s) FROM (SELECT * FROM (SELECT x, SUM(i) AS s FROM t GROUP BY x) AS e) AS d",  # INT64 partial sums
        "SELECT SUM(v) FROM (SELECT i AS v FROM t UNION ALL SELECT SUM(i) FROM t GROUP BY x) AS d",
        "SELECT SUM(s) FROM (SELECT SUM(a) AS s FROM (SELECT 1 AS a FROM t UNION ALL SELECT 2 FROM t) AS u GROUP BY a) AS d",
        f"SELECT SUM(u.f) FROM t AS u JOIN {PARTIAL} AS d ON d.x = u.x",  # the base table's column
    ],
)
def test_a_sum_that_reads_no_partial_sum_is_left_alone(sql):
    assert not reads(sql)


def test_a_column_list_on_a_base_table_is_typed_only_with_the_declared_order():
    partial = "SELECT SUM(s) FROM (SELECT SUM(a) AS s FROM t AS u (k, b, a) GROUP BY k) AS d"  # a is t.i, INT64
    assert not reads(partial, order={"t": ["x", "f", "i"]})
    assert reads(partial)
    swapped = "SELECT SUM(s) FROM (SELECT SUM(i) AS s FROM t AS u (x, i, f) GROUP BY x) AS d"  # i is t.f, FLOAT64
    assert reads(swapped, order={"t": ["x", "f", "i"]})
    assert reads(swapped)


def profile(sql: str) -> tuple[int, int]:
    return float_regrouping._profile(sqlglot.parse_one(sql, read="bigquery"), float_regrouping._Reader(types("bigquery"), None))


def test_a_with_definition_counts_once_for_each_place_that_reads_it():
    assert profile(WITH + "SELECT SUM(s) FROM d") == (2, 1)
    assert profile(WITH + "SELECT SUM(s) FROM d UNION ALL SELECT MAX(s) FROM d") == (3, 1)
    assert profile(WITH + "SELECT SUM(f) FROM t") == (1, 0)  # never read: no copy
    assert profile(WITH + ", e AS (SELECT SUM(s) AS v FROM d) SELECT a.v, b.v FROM e AS a, e AS b") == (4, 2)
    assert profile(f"SELECT SUM(s) FROM {PARTIAL} AS d") == (2, 1)
