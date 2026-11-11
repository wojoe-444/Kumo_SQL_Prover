"""Grouping a grouped derived table by all of its keys again is a projection (``kumo_sql_prover.one_row_groups``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.one_row_groups import collapse_one_row_groups

SCHEMA = {"t": ["a", "b", "c"], "u": ["a", "b", "c"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}
GROUPED = "(SELECT a, SUM(b) AS s, MIN(c) AS m FROM t GROUP BY a) AS d"
PAIRS = "(SELECT a, b, SUM(c) AS s FROM t GROUP BY a, b) AS d"


def collapsed(sql):
    out = collapse_one_row_groups(sqlglot.parse_one(sql, read="postgres"))
    return out.sql(dialect="postgres") if out is not None else None


def proved(left, right, names=False):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=names).proven


def rows_on_duckdb(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name in SCHEMA:
        db.execute(f"CREATE TABLE {name} (a BIGINT, b BIGINT, c BIGINT)")
        insert_rows(db, name, rows.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def differ_on_duckdb(left, right, rows):
    a, b = rows_on_duckdb([left, right], rows)
    return a != b


def test_a_regrouping_by_every_key_becomes_a_projection():
    assert collapsed(f"SELECT d.a, MIN(d.s) AS s, MAX(d.m) AS m FROM {GROUPED} GROUP BY d.a") == "SELECT a, SUM(b) AS s, MIN(c) AS m FROM t GROUP BY a"
    assert collapsed(f"SELECT d.a, MIN(d.s), SUM(DISTINCT d.m) FROM {GROUPED} GROUP BY d.a") == "SELECT a, SUM(b), MIN(c) FROM t GROUP BY a"
    assert collapsed(f"SELECT d.a AS k, SUM(d.s + d.m) AS v FROM {GROUPED} GROUP BY d.a") == "SELECT a AS k, (SUM(b) + MIN(c)) AS v FROM t GROUP BY a"
    assert collapsed(f"SELECT d.a, COUNT(*) AS n, COUNT(d.m) AS k FROM {GROUPED} GROUP BY d.a") == (
        "SELECT a, 1 AS n, CASE WHEN NOT (MIN(c)) IS NULL THEN 1 ELSE 0 END AS k FROM t GROUP BY a"
    )
    assert collapsed(f"SELECT d.a, MIN(d.s) FILTER (WHERE d.m > 2) AS s, COUNT(*) FILTER (WHERE d.m > 2) AS n FROM {GROUPED} GROUP BY d.a") == (
        "SELECT a, CASE WHEN MIN(c) > 2 THEN SUM(b) END AS s, CASE WHEN MIN(c) > 2 THEN 1 ELSE 0 END AS n FROM t GROUP BY a"
    )
    # grouped by more than the keys, in any order; the derived table's HAVING stays
    assert collapsed(f"SELECT d.b, d.a, MAX(d.s) AS s FROM {PAIRS} GROUP BY d.b, d.a") == "SELECT b, a, SUM(c) AS s FROM t GROUP BY a, b"
    assert collapsed(f"SELECT d.a, d.s, MAX(d.m) AS m FROM {GROUPED} GROUP BY d.a, d.s") == "SELECT a, SUM(b) AS s, MIN(c) AS m FROM t GROUP BY a"
    assert collapsed("SELECT d.a, MIN(d.s) AS s FROM (SELECT a, SUM(b) AS s FROM t GROUP BY a HAVING COUNT(*) > 1) AS d GROUP BY d.a") == (
        "SELECT a, SUM(b) AS s FROM t GROUP BY a HAVING COUNT(*) > 1"
    )
    # an unnamed output stays unnamed
    assert collapsed(f"SELECT d.a, MAX(d.a), COUNT(*) FROM {GROUPED} GROUP BY d.a") == "SELECT a, (a), (1) FROM t GROUP BY a"
    # a constant key tells no rows apart: at most one row below, so one row per outer group
    assert collapsed("SELECT d.tag, SUM(d.n) AS n FROM (SELECT 1 AS tag, COUNT(*) AS n FROM t GROUP BY TRUE) AS d GROUP BY d.tag") == (
        "SELECT 1 AS tag, COUNT(*) AS n FROM t GROUP BY TRUE"
    )
    # one row over one row
    assert collapsed("SELECT MIN(d.s) AS s, COUNT(*) AS n FROM (SELECT SUM(b) AS s FROM t) AS d") == "SELECT SUM(b) AS s, 1 AS n FROM t"


@pytest.mark.parametrize("sql", [
    f"SELECT d.a, MIN(d.s) AS s FROM {PAIRS} GROUP BY d.a",  # fewer keys: a real roll-up
    f"SELECT MIN(d.s) AS s FROM {GROUPED}",
    f"SELECT d.s, COUNT(*) AS n FROM {GROUPED} GROUP BY d.s",  # an aggregate is not a key
    f"SELECT d.a, AVG(d.s) AS s FROM {GROUPED} GROUP BY d.a",
    f"SELECT d.a, STRING_AGG(CAST(d.s AS TEXT), ',') AS s FROM {GROUPED} GROUP BY d.a",
    f"SELECT d.a, MIN(d.s) AS s FROM {GROUPED} GROUP BY d.a HAVING MIN(d.s) > 1",
    f"SELECT d.a, MIN(d.s) AS s FROM {GROUPED} WHERE d.m > 1 GROUP BY d.a",
    f"SELECT d.a, MIN(d.s) AS s FROM {GROUPED} GROUP BY d.a ORDER BY 1 LIMIT 2",
    f"SELECT DISTINCT d.a, MIN(d.s) AS s FROM {GROUPED} GROUP BY d.a",
    f"SELECT d.a, MIN(d.s) AS s FROM {GROUPED} JOIN u ON u.a = d.a GROUP BY d.a",
    f"SELECT d.a, MIN(d.s) AS s FROM {GROUPED} GROUP BY ROLLUP (d.a)",
    f"SELECT d.a, MIN(d.s) AS s, (SELECT MAX(u.b) FROM u WHERE u.a = d.a) AS x FROM {GROUPED} GROUP BY d.a",
    f"SELECT d.a, SUM(MIN(d.s)) OVER () AS s FROM {GROUPED} GROUP BY d.a",
    "SELECT d.a, MIN(d.s) AS s FROM (SELECT a, SUM(b) AS s FROM t GROUP BY ROLLUP (a)) AS d GROUP BY d.a",
    "SELECT d.a, MIN(d.s) AS s FROM (SELECT a, SUM(b) AS s FROM t GROUP BY a ORDER BY 2 LIMIT 3) AS d GROUP BY d.a",
    "SELECT d.a, MIN(d.s) AS s FROM (SELECT DISTINCT a, SUM(b) AS s FROM t GROUP BY a, c) AS d GROUP BY d.a",
    "SELECT d.a, MIN(d.s) AS s FROM (SELECT a, SUM(b) AS s FROM t GROUP BY a, c) AS d GROUP BY d.a",  # a key that is not an output
    "SELECT d.a, MIN(d.b) AS s FROM (SELECT a, b FROM t) AS d GROUP BY d.a",  # not grouped
    "SELECT COUNT(*) AS n FROM (SELECT SUM(b) AS s FROM t) AS d",  # SELECT 1 FROM t would be a row per row
    "SELECT MIN(d.s) AS s FROM (SELECT SUM(b) AS s FROM t HAVING COUNT(*) > 1) AS d",  # the one row may be gone
    "SELECT MIN(d.s) AS s FROM (SELECT b AS s FROM t) AS d",
    "SELECT SUM(d.n) AS n FROM (SELECT COUNT(*) AS n FROM t GROUP BY TRUE) AS d",  # no row below over no input, one row above
])
def test_other_shapes_are_left_alone(sql):
    assert collapsed(sql) is None


TARGETS = [
    (f"SELECT d.a, MIN(d.s) AS s, MAX(d.m) AS m FROM {GROUPED} GROUP BY d.a", "SELECT a, SUM(b) AS s, MIN(c) AS m FROM t GROUP BY a"),
    (f"SELECT d.a, MAX(d.s) AS s, SUM(d.m) AS m, COUNT(*) AS n FROM {GROUPED} GROUP BY d.a", "SELECT a, SUM(b) AS s, MIN(c) AS m, 1 AS n FROM t GROUP BY a"),
    (f"SELECT d.a, COUNT(d.m) AS n FROM {GROUPED} GROUP BY d.a", "SELECT a, CASE WHEN COUNT(c) > 0 THEN 1 ELSE 0 END AS n FROM t GROUP BY a"),
    (
        "SELECT d.a, MIN(d.s) FILTER (WHERE d.g) AS s, MIN(d.m) FILTER (WHERE d.g) AS m, SUM(d.a) FILTER (WHERE d.g) AS k FROM (SELECT a, SUM(b) AS s, MIN(c) AS m, GROUPING(a) = 0 AS g FROM t GROUP BY a) AS d GROUP BY d.a",
        "SELECT a, SUM(b) AS s, MIN(c) AS m, SUM(DISTINCT a) AS k FROM t GROUP BY a",
    ),
    ("SELECT MIN(d.s) AS s, MAX(d.m) AS m FROM (SELECT SUM(b) AS s, MIN(c) AS m FROM t) AS d", "SELECT SUM(b) AS s, MIN(c) AS m FROM t"),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_one_row_groups_prove(left, right):
    assert proved(left, right)


def test_an_unnamed_output_is_not_given_a_column_name():
    left = f"SELECT d.a, MIN(d.s), MAX(d.m) FROM {GROUPED} GROUP BY d.a"
    assert proved(left, "SELECT a, SUM(b), MIN(c) FROM t GROUP BY a", names=True)
    assert not proved(left, "SELECT a, SUM(b) AS s, MIN(c) AS m FROM t GROUP BY a", names=True)
    # (MAX of a key is read as the key, named, by an older rule before this one runs: not tested here)


ROWS = {"t": [(1, 5, 1), (1, 6, None), (2, 7, 3), (2, None, 4), (None, 4, 1), (None, None, None), (3, None, None)]}
NEAR_MISSES = [
    # fewer keys than the derived table has: several rows per group
    (f"SELECT d.a, MIN(d.s) AS s FROM {PAIRS} GROUP BY d.a", "SELECT a, SUM(c) AS s FROM t GROUP BY a"),
    (f"SELECT d.a, MIN(d.s) AS s FROM {PAIRS} GROUP BY d.a", f"SELECT d.a, d.s FROM {PAIRS}"),
    # COUNT of the one value is 1 or 0, not the value; AVG and SUM of it are not its count
    (f"SELECT d.a, COUNT(d.s) AS s FROM {GROUPED} GROUP BY d.a", "SELECT a, SUM(b) AS s FROM t GROUP BY a"),
    (f"SELECT d.a, COUNT(d.m) AS n FROM {GROUPED} GROUP BY d.a", "SELECT a, 1 AS n FROM t GROUP BY a"),
    (f"SELECT d.a, COUNT(*) AS n FROM {GROUPED} GROUP BY d.a", "SELECT a, COUNT(*) AS n FROM t GROUP BY a"),
    # a FILTER that fails leaves NULL (or 0), not the value
    (f"SELECT d.a, MIN(d.s) FILTER (WHERE d.m > 2) AS s FROM {GROUPED} GROUP BY d.a", "SELECT a, SUM(b) AS s FROM t GROUP BY a"),
    # not grouped below: MIN really chooses
    ("SELECT d.a, MIN(d.b) AS s FROM (SELECT a, b FROM t) AS d GROUP BY d.a", "SELECT a, b AS s FROM t"),
    # a global aggregate over a grouped table is one row, not one per group
    (f"SELECT MIN(d.s) AS s FROM {GROUPED}", f"SELECT d.s FROM {GROUPED}"),
    # grouped by a constant, the derived table is empty over no input, and a global aggregate above still returns a row
    ("SELECT SUM(d.n) AS n FROM (SELECT COUNT(*) AS n FROM u GROUP BY TRUE) AS d", "SELECT COUNT(*) AS n FROM u GROUP BY TRUE"),
    # the derived table's one row removed by HAVING: the outer aggregate still returns a row
    ("SELECT MIN(d.s) AS s FROM (SELECT SUM(b) AS s FROM t HAVING COUNT(*) > 99) AS d", "SELECT SUM(b) AS s FROM t HAVING COUNT(*) > 99"),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, ROWS)


COLLAPSED_QUERIES = [
    f"SELECT d.a, MIN(d.s) AS s, MAX(d.m) AS m FROM {GROUPED} GROUP BY d.a",
    f"SELECT d.a, COUNT(*) AS n, COUNT(d.m) AS k, COUNT(DISTINCT d.s) AS j FROM {GROUPED} GROUP BY d.a",
    f"SELECT d.a, MIN(d.s) FILTER (WHERE d.m > 2) AS s, COUNT(*) FILTER (WHERE d.m > 2) AS n, COUNT(d.s) FILTER (WHERE d.m > 0) AS k FROM {GROUPED} GROUP BY d.a",
    f"SELECT d.b, d.a, MAX(d.s) + SUM(d.a - d.b) AS s FROM {PAIRS} GROUP BY d.b, d.a",
    f"SELECT d.a, MAX(d.a), COUNT(*), SUM(DISTINCT d.m) FROM {GROUPED} GROUP BY d.a",
    "SELECT d.a, MIN(d.s) AS s FROM (SELECT a, SUM(b) AS s FROM t GROUP BY a HAVING COUNT(*) > 1) AS d GROUP BY d.a",
    "SELECT MIN(d.s) AS s, COUNT(*) AS n, COUNT(d.s) AS k FROM (SELECT SUM(b) AS s FROM t) AS d",
    "SELECT d.p, BIT_OR(d.s) AS s FROM (SELECT a > 1 AS p, SUM(b) AS s FROM t GROUP BY a > 1) AS d GROUP BY d.p",
    "SELECT d.tag, SUM(d.n) AS n, MAX(d.m) AS m FROM (SELECT 1 AS tag, COUNT(*) AS n, MIN(b) AS m FROM t GROUP BY TRUE) AS d GROUP BY d.tag",
]


def random_rows(rng):
    rows = [(rng.choice([None, 1, 2, 3]), rng.choice([None, 0, 1, 2]), rng.choice([None, 1, 2, 3])) for _ in range(rng.randint(0, 7))]
    return {"t": rows + rows[:2]}


@pytest.mark.parametrize("sql", COLLAPSED_QUERIES)
def test_the_projection_gives_the_same_rows(sql):
    out = collapsed(sql)
    normalized = normalize(sql, schema=SCHEMA, types=TYPES, dialect="postgres")
    assert out is not None and " AS d" not in out
    rng = random.Random(len(sql))
    for rows in [{}, ROWS] + [random_rows(rng) for _ in range(14)]:
        assert not differ_on_duckdb(sql, out, rows)
        assert not differ_on_duckdb(sql, normalized, rows)
