"""A HAVING condition over grouping expressions alone moves to WHERE (``kumo_sql_prover.key_expression_having``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.key_expression_having import key_expression_having_to_where

SCHEMA = {"t": ["a", "b", "c"], "u": ["a", "b", "c"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}


def moved(sql):
    out = key_expression_having_to_where(sqlglot.parse_one(sql, read="postgres"))
    return out.sql(dialect="postgres") if out is not None else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


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


def test_conjuncts_over_grouping_expressions_move_to_where():
    assert moved("SELECT 2 * a AS d, COUNT(*) AS n FROM t GROUP BY 2 * a HAVING NOT 2 * a IS NULL") == (
        "SELECT 2 * a AS d, COUNT(*) AS n FROM t WHERE NOT 2 * a IS NULL GROUP BY 2 * a"
    )
    assert moved("SELECT 2 * a AS d, COUNT(*) AS n FROM t WHERE c = 1 GROUP BY 2 * a HAVING 2 * a > 2 AND COUNT(*) > 1") == (
        "SELECT 2 * a AS d, COUNT(*) AS n FROM t WHERE c = 1 AND 2 * a > 2 GROUP BY 2 * a HAVING COUNT(*) > 1"
    )
    assert moved("SELECT a + b AS d, c FROM t GROUP BY a + b, c HAVING a + b > c OR c IS NULL") == (
        "SELECT a + b AS d, c FROM t WHERE (a + b > c OR c IS NULL) GROUP BY a + b, c"
    )
    assert moved("SELECT a > 1 AS p FROM t GROUP BY a > 1 HAVING a > 1") == "SELECT a > 1 AS p FROM t WHERE a > 1 GROUP BY a > 1"


@pytest.mark.parametrize("sql", [
    "SELECT 2 * a AS d FROM t GROUP BY 2 * a HAVING a > 1",  # a alone is not a key
    "SELECT 2 * a AS d FROM t GROUP BY 2 * a HAVING 2 * a + b > 1",
    "SELECT 2 * a AS d FROM t GROUP BY 2 * a HAVING a * 2 > 1",  # another text
    "SELECT 2 * a AS d FROM t GROUP BY 2 * a HAVING SUM(2 * a) > 1",
    "SELECT 2 * a AS d FROM t GROUP BY 2 * a HAVING 2 * a > (SELECT MAX(u.a) FROM u)",
    "SELECT 2 * a AS d FROM t GROUP BY 2 * a HAVING COUNT(*) > 1",
    "SELECT a FROM t GROUP BY a HAVING a > 1",  # a key column: the older rule's
    "SELECT 2 * a AS d FROM t GROUP BY ROLLUP (2 * a) HAVING 2 * a > 1",
    "SELECT 2 * a AS d FROM t GROUP BY GROUPING SETS ((2 * a), ()) HAVING 2 * a > 1",
    "SELECT 2 * a AS d FROM t GROUP BY CUBE (2 * a) HAVING 2 * a > 1",
    "SELECT 2 * a AS d, ROW_NUMBER() OVER (ORDER BY 2 * a) AS r FROM t GROUP BY 2 * a HAVING 2 * a > 1",
    "SELECT 2 * a AS d, b AS a FROM t GROUP BY 2 * a, b HAVING 2 * a > 1",  # an alias named like the column
    "SELECT RANDOM() * a AS d FROM t GROUP BY RANDOM() * a HAVING RANDOM() * a > 1",
    "SELECT 2 * a AS d FROM t GROUP BY 2 * a",
    "SELECT 2 * a AS d FROM t HAVING 2 * 3 > 1",
])
def test_other_conjuncts_stay(sql):
    assert moved(sql) is None


GROUPED = "SELECT 2 * a AS d, COUNT(*) AS n FROM t GROUP BY 2 * a"
TARGETS = [
    (f"{GROUPED} HAVING 2 * a > 2 AND COUNT(*) > 1", "SELECT 2 * a AS d, COUNT(*) AS n FROM t WHERE 2 * a > 2 GROUP BY 2 * a HAVING COUNT(*) > 1"),
    (f"{GROUPED} HAVING 2 * a IS NOT NULL", "SELECT 2 * a AS d, COUNT(*) AS n FROM t WHERE a IS NOT NULL GROUP BY 2 * a"),
    # Calcite's semi-join as a join with a grouped derived table that drops its NULL key
    (
        "SELECT u.b FROM u WHERE EXISTS (SELECT 1 FROM (SELECT 2 * a AS f FROM t GROUP BY 2 * a) AS d WHERE u.a = d.f)",
        "SELECT u.b FROM u JOIN (SELECT 2 * a AS f, TRUE AS g FROM t GROUP BY 2 * a HAVING 2 * a IS NOT NULL) AS d ON u.a = d.f",
    ),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_key_expression_conditions_prove(left, right):
    assert proved(left, right)


ROWS = {"t": [(1, 5, 1), (1, 6, 1), (2, 7, 1), (3, 0, 2), (None, 4, 1), (None, None, 2)], "u": [(2, 1, 1), (4, 2, 2), (None, 3, 3)]}
NEAR_MISSES = [
    # the condition, or none
    (f"{GROUPED} HAVING 2 * a > 2", GROUPED),
    (f"{GROUPED} HAVING 2 * a > 2", "SELECT 2 * a AS d, COUNT(*) AS n FROM t WHERE 2 * a >= 2 GROUP BY 2 * a"),
    # a condition on an aggregate is not a condition on the rows
    ("SELECT 2 * a AS d, COUNT(*) AS n FROM t GROUP BY 2 * a HAVING MAX(b) > 5", "SELECT 2 * a AS d, COUNT(*) AS n FROM t WHERE b > 5 GROUP BY 2 * a"),
    ("SELECT 2 * a AS d, COUNT(*) AS n FROM t GROUP BY 2 * a HAVING COUNT(b) > 0", "SELECT 2 * a AS d, COUNT(*) AS n FROM t WHERE b IS NOT NULL GROUP BY 2 * a"),
    # under ROLLUP the grand-total row has a NULL key: HAVING drops it, WHERE does not
    (
        "SELECT 2 * a AS d, COUNT(*) AS n FROM t GROUP BY ROLLUP (2 * a) HAVING 2 * a IS NOT NULL",
        "SELECT 2 * a AS d, COUNT(*) AS n FROM t WHERE 2 * a IS NOT NULL GROUP BY ROLLUP (2 * a)",
    ),
    # a global aggregate keeps its row when WHERE removes every input row, HAVING removes the row
    ("SELECT COUNT(*) AS n FROM t HAVING 1 > 2", "SELECT COUNT(*) AS n FROM t WHERE 1 > 2"),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, ROWS)


MOVED_QUERIES = [
    f"{GROUPED} HAVING 2 * a > 2 AND COUNT(*) > 1",
    f"{GROUPED} HAVING 2 * a IS NOT NULL",
    f"{GROUPED} HAVING 2 * a IS NULL OR 2 * a = 2",
    "SELECT a + b AS d, c, SUM(a) AS s FROM t WHERE c > 0 GROUP BY a + b, c HAVING a + b > c",
    "SELECT a > 1 AS p, MAX(b) AS m FROM t GROUP BY a > 1 HAVING a > 1 AND MAX(b) > 0",
]


def random_rows(rng):
    rows = [(rng.choice([None, 1, 2, 3]), rng.choice([None, 0, 1, 2]), rng.choice([None, 1, 2, 3])) for _ in range(rng.randint(0, 7))]
    return {"t": rows + rows[:2]}


@pytest.mark.parametrize("sql", MOVED_QUERIES)
def test_the_moved_condition_gives_the_same_rows(sql):
    out = moved(sql)
    normalized = normalize(sql, schema=SCHEMA, types=TYPES, dialect="postgres")
    assert out is not None
    rng = random.Random(len(sql))
    for rows in [{}, ROWS] + [random_rows(rng) for _ in range(14)]:
        assert not differ_on_duckdb(sql, out, rows)
        assert not differ_on_duckdb(sql, normalized, rows)
