"""``COUNT(*)`` over a finer grouping counts the extra key when it is never NULL (``kumo_sql_prover.counted_groups``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.counted_groups import count_extra_key
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"a": ["g", "x", "n"], "b": ["g", "y"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}
NOT_NULL = {"a": frozenset({"x"})}
CONSTRAINTS = {"a": TableConstraints(not_null=frozenset({"x"}))}


def counted(sql, dialect="postgres"):
    out = count_extra_key(sqlglot.parse_one(sql, read=dialect), SCHEMA, NOT_NULL)
    return out.sql(dialect=dialect) if out is not None else None


def proved(left, right):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, dialect="postgres", compare_names=False
    ).proven


def rows_on_duckdb(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name, columns in SCHEMA.items():
        db.execute(f"CREATE TABLE {name} ({', '.join(f'{c} BIGINT' for c in columns)})")
        insert_rows(db, name, rows.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def differ_on_duckdb(left, right, rows):
    first, second = rows_on_duckdb([left, right], rows)
    return first != second


def test_the_count_reads_the_extra_key():
    assert (
        counted("SELECT d.g, COUNT(*) AS c FROM (SELECT a.g, a.x FROM a GROUP BY a.g, a.x) AS d GROUP BY d.g")
        == "SELECT d.g, COUNT(d.x) AS c FROM (SELECT a.g, a.x FROM a GROUP BY a.g, a.x) AS d GROUP BY d.g"
    )
    # the key is added to the outputs when the derived table does not list it
    assert (
        counted("SELECT g, COUNT(*) FROM (SELECT a.g FROM a JOIN b ON a.g = b.g WHERE b.y > 0 GROUP BY a.g, a.x) AS d GROUP BY g")
        == "SELECT g, COUNT(d.kqx) FROM (SELECT a.g, a.x AS kqx FROM a JOIN b ON a.g = b.g WHERE b.y > 0 GROUP BY a.g, a.x) AS d GROUP BY g"
    )
    # no outer GROUP BY: the count of distinct keys overall
    assert counted("SELECT COUNT(*) + 1 FROM (SELECT x FROM a GROUP BY x) AS d") == "SELECT COUNT(d.x) + 1 FROM (SELECT x FROM a GROUP BY x) AS d"
    assert counted('SELECT COUNT(*) FROM (SELECT x FROM a GROUP BY x) AS "D"', "postgres") == 'SELECT COUNT("D".x) FROM (SELECT x FROM a GROUP BY x) AS "D"'


@pytest.mark.parametrize("sql", [
    "SELECT d.g, COUNT(*) FROM (SELECT a.g, a.n FROM a GROUP BY a.g, a.n) AS d GROUP BY d.g",  # n may be NULL
    "SELECT d.x, COUNT(*) FROM (SELECT a.g, a.x FROM a GROUP BY a.g, a.x) AS d GROUP BY d.x",  # the extra key g may be NULL
    "SELECT d.g, COUNT(*) FROM (SELECT b.g, a.x FROM b LEFT JOIN a ON a.g = b.g GROUP BY b.g, a.x) AS d GROUP BY d.g",  # padded
    "SELECT d.g, COUNT(*) FROM (SELECT b.g, a.x FROM a RIGHT JOIN b ON a.g = b.g GROUP BY b.g, a.x) AS d GROUP BY d.g",
    "SELECT COUNT(*) FROM (SELECT a.g, a.x FROM a GROUP BY a.g, a.x) AS d",  # two extra keys
    "SELECT d.g, COUNT(*) FROM (SELECT a.g, a.x FROM a GROUP BY a.g, a.x) AS d GROUP BY d.g, d.x",  # no extra key
    "SELECT d.g, COUNT(*) FROM (SELECT a.g, a.x FROM a) AS d GROUP BY d.g",  # not grouped: repeats count
    "SELECT d.g, COUNT(*) FROM (SELECT DISTINCT a.g, a.x FROM a) AS d GROUP BY d.g",
    "SELECT d.g, COUNT(*) FROM (SELECT a.g, a.x FROM a GROUP BY a.g, a.x HAVING COUNT(*) > 1) AS d GROUP BY d.g",
    "SELECT d.g, COUNT(*) FROM (SELECT a.g, a.x FROM a GROUP BY a.g, a.x LIMIT 3) AS d GROUP BY d.g",
    "SELECT d.g, COUNT(*) FROM (SELECT a.g, a.x, COUNT(*) AS c FROM a GROUP BY a.g, a.x) AS d GROUP BY d.g",
    "SELECT d.g, COUNT(*) FROM (SELECT a.g, a.x FROM a GROUP BY a.g, a.x) AS d WHERE d.x > 1 GROUP BY d.g",
    "SELECT d.g, COUNT(*) FROM (SELECT a.g, a.x FROM a GROUP BY a.g, a.x) AS d GROUP BY d.g HAVING COUNT(*) > 1",
    "SELECT d.g, COUNT(*) FROM (SELECT a.g, a.x FROM a GROUP BY a.g, a.x) AS d GROUP BY d.g ORDER BY 2 LIMIT 1",
    "SELECT d.g, COUNT(*) FROM (SELECT a.g, a.x FROM a GROUP BY a.g, a.x) AS d JOIN b ON b.g = d.g GROUP BY d.g",
    "SELECT d.g, COUNT(d.x) FROM (SELECT a.g, a.x FROM a GROUP BY a.g, a.x) AS d GROUP BY d.g",  # nothing to respell
    "SELECT d.g, COUNT(*) OVER () FROM (SELECT a.g, a.x FROM a GROUP BY a.g, a.x) AS d",
    "SELECT d.g, COUNT(*) FROM (SELECT a.g, a.x + 1 AS x FROM a GROUP BY a.g, a.x + 1) AS d GROUP BY d.g",
    "SELECT d.g, COUNT(*) FROM (SELECT z.g, z.x FROM z GROUP BY z.g, z.x) AS d GROUP BY d.g",  # no schema for z
    "SELECT d.g, COUNT(*) FROM (SELECT s.g, s.x FROM (SELECT g, x FROM a) AS s GROUP BY s.g, s.x) AS d GROUP BY d.g",
    "SELECT d.g, COUNT(*) FROM (SELECT a.g, a.x FROM a GROUP BY ROLLUP (a.g, a.x)) AS d GROUP BY d.g",
])
def test_other_shapes_are_left_alone(sql):
    assert counted(sql) is None


TARGETS = [
    ("SELECT d.g, COUNT(*) FROM (SELECT a.g, a.x FROM a GROUP BY a.g, a.x) AS d GROUP BY d.g", "SELECT a.g, COUNT(DISTINCT a.x) FROM a GROUP BY a.g"),
    ("SELECT g, COUNT(*) FROM (SELECT a.g FROM a JOIN b ON a.g = b.g WHERE b.y > 0 GROUP BY a.g, a.x) AS d GROUP BY g",
     "SELECT a.g, COUNT(DISTINCT a.x) FROM a, b WHERE a.g = b.g AND b.y > 0 GROUP BY a.g"),
    ("SELECT COUNT(*) FROM (SELECT x FROM a WHERE n = 1 GROUP BY x) AS d", "SELECT COUNT(DISTINCT x) FROM a WHERE n = 1"),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_counts_of_distinct_keys_prove(left, right):
    assert proved(left, right)


ROWS = {"a": [(1, 1, None), (1, 1, 4), (1, 2, None), (2, 3, 5), (None, 3, 5), (None, 4, None)], "b": [(1, 1), (1, 2), (3, 1)]}
NEAR_MISSES = [
    # the inner row whose n is NULL is counted by COUNT(*) and not by COUNT(DISTINCT n)
    ("SELECT d.g, COUNT(*) FROM (SELECT a.g, a.n FROM a GROUP BY a.g, a.n) AS d GROUP BY d.g", "SELECT a.g, COUNT(DISTINCT a.n) FROM a GROUP BY a.g"),
    # g is NULL on some rows: the NULL group of g is a row of the derived table
    ("SELECT d.x, COUNT(*) FROM (SELECT a.g, a.x FROM a GROUP BY a.g, a.x) AS d GROUP BY d.x", "SELECT a.x, COUNT(DISTINCT a.g) FROM a GROUP BY a.x"),
    # an outer join pads x with NULL
    ("SELECT d.g, COUNT(*) FROM (SELECT b.g, a.x FROM b LEFT JOIN a ON a.g = b.g GROUP BY b.g, a.x) AS d GROUP BY d.g",
     "SELECT b.g, COUNT(DISTINCT a.x) FROM b LEFT JOIN a ON a.g = b.g GROUP BY b.g"),
    # without the inner grouping every source row is counted
    ("SELECT d.g, COUNT(*) FROM (SELECT a.g, a.x FROM a) AS d GROUP BY d.g", "SELECT a.g, COUNT(DISTINCT a.x) FROM a GROUP BY a.g"),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, ROWS)


COUNTED_QUERIES = [
    ("SELECT d.g, COUNT(*) AS c FROM (SELECT a.g, a.x FROM a GROUP BY a.g, a.x) AS d GROUP BY d.g", "SELECT a.g, COUNT(DISTINCT a.x) FROM a GROUP BY a.g"),
    ("SELECT g, COUNT(*) FROM (SELECT a.g FROM a JOIN b ON a.g = b.g WHERE b.y > 0 GROUP BY a.g, a.x) AS d GROUP BY g",
     "SELECT a.g, COUNT(DISTINCT a.x) FROM a JOIN b ON a.g = b.g WHERE b.y > 0 GROUP BY a.g"),
    ("SELECT COUNT(*) + 1 FROM (SELECT x FROM a WHERE n = 1 GROUP BY x) AS d", "SELECT COUNT(DISTINCT x) + 1 FROM a WHERE n = 1"),
]


def random_rows(rng):
    a = [(rng.choice([None, 1, 2]), rng.choice([1, 2, 3]), rng.choice([None, 1])) for _ in range(rng.randint(0, 6))]
    b = [(rng.choice([None, 1, 2]), rng.choice([None, 0, 1])) for _ in range(rng.randint(0, 4))]
    return {"a": a + a[:2], "b": b + b[:1]}


@pytest.mark.parametrize("sql, meaning", COUNTED_QUERIES)
def test_the_rewrite_returns_the_same_rows_on_random_databases(sql, meaning):
    rewritten = counted(sql)
    assert rewritten is not None
    rng = random.Random(len(sql))
    for _ in range(40):
        rows = random_rows(rng)
        first, second, third = rows_on_duckdb([sql, rewritten, meaning], rows)
        assert first == second == third, rows
