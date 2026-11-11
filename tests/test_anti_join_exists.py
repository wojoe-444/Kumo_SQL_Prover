"""``LEFT JOIN t ON .. WHERE t.c IS NULL`` is read as ``NOT EXISTS`` (``kumo_sql_prover.anti_join_exists``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.anti_join_exists import read_anti_join
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"a": ["k", "g", "x"], "t": ["k", "f", "n"], "b": ["k", "y"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}
NOT_NULL = {"t": frozenset({"f"})}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"f"}))}


def read(sql, dialect="postgres"):
    out = read_anti_join(sqlglot.parse_one(sql, read=dialect), SCHEMA, NOT_NULL)
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


def test_the_anti_join_becomes_not_exists():
    assert (
        read("SELECT a.g, COUNT(DISTINCT a.x) FROM a LEFT JOIN t ON a.k = t.k AND t.f = 1 WHERE t.k IS NULL GROUP BY a.g")
        == "SELECT a.g, COUNT(DISTINCT a.x) FROM a WHERE NOT EXISTS(SELECT 1 FROM t WHERE a.k = t.k AND t.f = 1) GROUP BY a.g"
    )
    # a declared NOT NULL column works without being compared in ON; other conjuncts and joins stay
    assert (
        read("SELECT a.g FROM a JOIN b ON a.k = b.k LEFT JOIN t AS s ON a.k > s.n WHERE b.y = 2 AND s.f IS NULL AND a.x = 1")
        == "SELECT a.g FROM a JOIN b ON a.k = b.k WHERE b.y = 2 AND NOT EXISTS(SELECT 1 FROM t AS s WHERE a.k > s.n) AND a.x = 1"
    )
    # a later inner or LEFT join that does not read the joined table
    assert (
        read("SELECT a.g, b.y FROM a LEFT JOIN t ON a.k = t.k LEFT JOIN b ON a.k = b.k WHERE t.k IS NULL")
        == "SELECT a.g, b.y FROM a LEFT JOIN b ON a.k = b.k WHERE NOT EXISTS(SELECT 1 FROM t WHERE a.k = t.k)"
    )


@pytest.mark.parametrize("sql", [
    "SELECT a.g FROM a LEFT JOIN t ON a.k = t.k WHERE t.n IS NULL",  # n may be NULL on a matched row
    "SELECT a.g FROM a LEFT JOIN t ON a.k = t.k OR t.n = 1 WHERE t.n IS NULL",  # the comparison is under OR
    "SELECT a.g FROM a LEFT JOIN t ON a.k IS NOT DISTINCT FROM t.n WHERE t.n IS NULL",  # NULL matches NULL
    "SELECT a.g FROM a LEFT JOIN t ON a.k = t.k WHERE t.k IS NULL OR a.x = 1",  # not a conjunct
    "SELECT a.g FROM a LEFT JOIN t ON a.k = t.k WHERE t.k IS NOT NULL",
    "SELECT a.g, t.f FROM a LEFT JOIN t ON a.k = t.k WHERE t.k IS NULL",  # the padded column is read
    "SELECT a.g FROM a LEFT JOIN t ON a.k = t.k WHERE t.k IS NULL ORDER BY t.n",
    "SELECT a.g FROM a LEFT JOIN t ON a.k = t.k WHERE t.k IS NULL AND COALESCE(t.n, 0) = 0",
    "SELECT a.g FROM a LEFT JOIN t ON a.k = t.k JOIN b ON b.k = t.n WHERE t.k IS NULL",  # a later join reads it
    "SELECT a.g FROM a LEFT JOIN t ON a.k = t.k RIGHT JOIN b ON a.k = b.k WHERE t.k IS NULL",  # pads the left side
    "SELECT a.g FROM a LEFT JOIN t ON a.k = t.k FULL JOIN b ON a.k = b.k WHERE t.k IS NULL",
    "SELECT a.g FROM a JOIN t ON a.k = t.k WHERE t.k IS NULL",  # not an outer join
    "SELECT a.g FROM a RIGHT JOIN t ON a.k = t.k WHERE t.k IS NULL",
    "SELECT a.g FROM a LEFT JOIN t USING (k) WHERE t.k IS NULL",
    "SELECT a.g FROM a LEFT JOIN (SELECT k FROM t) AS t ON a.k = t.k WHERE t.k IS NULL",  # not a base table
    "SELECT a.g FROM a LEFT JOIN t ON a.k = t.k AND EXISTS (SELECT 1 FROM b WHERE b.k = t.n) WHERE t.k IS NULL",
    "SELECT a.g FROM a LEFT JOIN t ON k = t.k WHERE t.k IS NULL",  # a bare column in ON
    "SELECT * FROM a LEFT JOIN t ON a.k = t.k WHERE t.k IS NULL",
    "SELECT a.g FROM a LEFT JOIN t ON a.k = t.k LEFT JOIN t AS t ON a.k = t.k WHERE t.k IS NULL",
    "SELECT a.g FROM a LEFT JOIN z ON a.k = z.k WHERE z.k IS NULL",  # no schema for the joined table
    "SELECT a.g FROM a LEFT JOIN t ON a.k = t.k GROUP BY a.g HAVING MAX(t.k) IS NULL",
])
def test_other_shapes_are_left_alone(sql):
    assert read(sql) is None


PROBE = "NOT EXISTS (SELECT 1 FROM t WHERE t.k = a.k AND t.f = 1)"
TARGETS = [
    # under an aggregate, which is where the join form used to be wrapped before it was read
    ("SELECT a.g, COUNT(DISTINCT a.x) FROM a LEFT JOIN t ON a.k = t.k AND t.f = 1 WHERE t.k IS NULL GROUP BY a.g",
     f"SELECT a.g, COUNT(DISTINCT a.x) FROM a WHERE {PROBE} GROUP BY a.g"),
    ("SELECT a.g, SUM(a.x), COUNT(*) FROM a LEFT JOIN (SELECT k FROM t WHERE f = 1) AS d ON a.k = d.k WHERE d.k IS NULL GROUP BY a.g",
     f"SELECT a.g, SUM(a.x), COUNT(*) FROM a WHERE {PROBE} GROUP BY a.g"),
    ("SELECT COUNT(*) FROM a JOIN b ON a.k = b.k LEFT JOIN t ON a.k = t.k AND t.f = 1 WHERE t.f IS NULL AND b.y > 0",
     f"SELECT COUNT(*) FROM a, b WHERE a.k = b.k AND b.y > 0 AND {PROBE}"),
    ("SELECT a.g, b.y FROM a LEFT JOIN t ON a.k = t.k AND t.f = 1 JOIN b ON a.k = b.k WHERE t.k IS NULL",
     f"SELECT a.g, b.y FROM a JOIN b ON a.k = b.k WHERE {PROBE}"),
    # without an aggregate it proved before and still does
    ("SELECT a.g FROM a LEFT JOIN t ON a.k = t.k AND t.f = 1 WHERE t.k IS NULL", f"SELECT a.g FROM a WHERE {PROBE}"),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_anti_joins_prove(left, right):
    assert proved(left, right)


ROWS = {
    "a": [(1, 1, 5), (1, 1, 5), (2, 1, 6), (3, 2, 7), (None, 2, 8), (4, 3, 9)],
    "t": [(1, 1, None), (1, 1, 3), (2, 0, None), (4, 1, 4), (None, 1, None)],
    "b": [(1, 1), (3, 2), (3, 3), (9, 4)],
}
GROUPED = "SELECT a.g, COUNT(*) FROM a WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.k = a.k) GROUP BY a.g"
NEAR_MISSES = [
    # n is NULL on matched rows too, so those pass the test
    ("SELECT a.g, COUNT(*) FROM a LEFT JOIN t ON a.k = t.k WHERE t.n IS NULL GROUP BY a.g", GROUPED),
    # IS NOT NULL keeps the matched rows, once per match
    ("SELECT a.g, COUNT(*) FROM a LEFT JOIN t ON a.k = t.k WHERE t.k IS NOT NULL GROUP BY a.g", GROUPED),
    # the filter is part of the ON condition, not of the probe on the other side
    ("SELECT a.g, COUNT(*) FROM a LEFT JOIN t ON a.k = t.k AND t.f = 1 WHERE t.k IS NULL GROUP BY a.g", GROUPED),
    # NOT IN is not NOT EXISTS when the probed column holds a NULL
    ("SELECT a.g, COUNT(*) FROM a LEFT JOIN t ON a.k = t.k WHERE t.k IS NULL GROUP BY a.g",
     "SELECT a.g, COUNT(*) FROM a WHERE a.k NOT IN (SELECT t.k FROM t) GROUP BY a.g"),
    # a RIGHT JOIN after it pads the left side: those rows pass IS NULL without having met t, but the probe
    # written for them finds the rows of t with f = 1
    ("SELECT b.y, COUNT(*) FROM a LEFT JOIN t ON a.k = t.k OR t.f = 1 RIGHT JOIN b ON a.k = b.k WHERE t.f IS NULL GROUP BY b.y",
     "SELECT b.y, COUNT(*) FROM a RIGHT JOIN b ON a.k = b.k WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.k = a.k OR t.f = 1) GROUP BY b.y"),
    # an inner join is no anti-join
    ("SELECT a.g, COUNT(*) FROM a JOIN t ON a.k = t.k WHERE t.n IS NULL GROUP BY a.g", GROUPED),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, ROWS)


READ_QUERIES = [
    "SELECT a.g, COUNT(DISTINCT a.x) FROM a LEFT JOIN t ON a.k = t.k AND t.f = 1 WHERE t.k IS NULL GROUP BY a.g",
    "SELECT a.g, a.x FROM a JOIN b ON a.k = b.k LEFT JOIN t AS s ON a.k > s.n WHERE b.y >= 2 AND s.f IS NULL",
    "SELECT a.g, b.y FROM a LEFT JOIN t ON a.k = t.k LEFT JOIN b ON a.k = b.k WHERE t.k IS NULL",
    "SELECT a.g, b.y FROM a LEFT JOIN b ON a.k = b.k LEFT JOIN t ON b.y = t.n AND a.x <> t.k WHERE t.n IS NULL",
    "SELECT a.g, SUM(b.y) FROM a, b LEFT JOIN t ON b.k < t.k WHERE t.k IS NULL GROUP BY a.g",
]


def random_rows(rng):
    def table(width, size):
        rows = [tuple(rng.choice([None, 1, 2, 3]) for _ in range(width)) for _ in range(rng.randint(0, size))]
        return rows + rows[:2]  # with duplicate rows

    never_null = [(k, rng.choice([0, 1]), n) for k, _, n in table(3, 5)]
    return {"a": table(3, 5), "t": never_null, "b": table(2, 4)}


@pytest.mark.parametrize("sql", READ_QUERIES)
def test_the_rewrite_returns_the_same_rows_on_random_databases(sql):
    rewritten = read(sql)
    assert rewritten is not None
    rng = random.Random(len(sql))
    for _ in range(40):
        rows = random_rows(rng)
        first, second = rows_on_duckdb([sql, rewritten], rows)
        assert first == second, rows
