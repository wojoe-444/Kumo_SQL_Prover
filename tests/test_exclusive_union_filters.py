"""A ``UNION ALL`` of one query under exclusive filters is one select (``kumo_sql_prover.exclusive_union_filters``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.exclusive_union_filters import merge_exclusive_unions

SCHEMA = {"l": ["x", "pk", "q", "b"], "p": ["k", "b", "s"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}
TYPES["p"]["s"] = "STRING"


def merged(sql, dialect="postgres"):
    before = sqlglot.parse_one(sql, read=dialect)
    text = before.sql(dialect=dialect)
    after = merge_exclusive_unions(before).sql(dialect=dialect)
    return None if after == text else after


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


def rows_on_duckdb(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name, columns in SCHEMA.items():
        db.execute(f"CREATE TABLE {name} ({', '.join(c + (' VARCHAR' if TYPES[name][c] == 'STRING' else ' BIGINT') for c in columns)})")
        insert_rows(db, name, rows.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def differ_on_duckdb(left, right, rows):
    first, second = rows_on_duckdb([left, right], rows)
    return first != second


def test_the_branches_become_one_select():
    assert (
        merged("SELECT l.x FROM l JOIN p ON p.k = l.pk WHERE p.b = 1 AND l.q < 5 UNION ALL SELECT l.x FROM l JOIN p ON p.k = l.pk WHERE p.b = 2 AND l.q > 9")
        == "SELECT l.x FROM l JOIN p ON p.k = l.pk WHERE (p.b = 1 AND l.q < 5) OR (p.b = 2 AND l.q > 9)"
    )
    # three branches, nested either way, inside a derived table, the literal on either side
    assert (
        merged("SELECT SUM(r) FROM (SELECT x AS r FROM l WHERE b = 1 UNION ALL (SELECT x AS r FROM l WHERE 2 = b UNION ALL SELECT x AS r FROM l WHERE b = 3 AND q > 0)) AS u")
        == "SELECT SUM(r) FROM (SELECT x AS r FROM l WHERE (b = 1) OR (2 = b) OR (b = 3 AND q > 0)) AS u"
    )
    assert (
        merged("SELECT p.k FROM p WHERE p.s = 'Brand#21' UNION ALL SELECT p.k FROM p WHERE p.s = 'Brand#42'")
        == "SELECT p.k FROM p WHERE (p.s = 'Brand#21') OR (p.s = 'Brand#42')"
    )


@pytest.mark.parametrize("sql", [
    # the filters do not exclude each other
    "SELECT l.x FROM l WHERE l.b = 1 UNION ALL SELECT l.x FROM l WHERE l.b = 1",
    "SELECT l.x FROM l WHERE l.b = 1 UNION ALL SELECT l.x FROM l WHERE l.q = 2",
    "SELECT l.x FROM l WHERE l.b = 1 UNION ALL SELECT l.x FROM l WHERE l.b <> 1",  # not read: only column = literal
    "SELECT l.x FROM l WHERE l.b = 1 UNION ALL SELECT l.x FROM l WHERE l.b = 2 OR l.q = 3",
    "SELECT l.x FROM l WHERE l.b = 1 UNION ALL SELECT l.x FROM l WHERE l.b = 2 UNION ALL SELECT l.x FROM l WHERE l.b = 2 AND l.q = 1",
    "SELECT l.x FROM l WHERE l.b = 1 UNION ALL SELECT l.x FROM l WHERE l.b = '2'",  # a number and a string
    "SELECT p.k FROM p WHERE p.s = 'a' UNION ALL SELECT p.k FROM p WHERE p.s = 'A '",  # equal under some collations
    "SELECT l.x FROM l WHERE l.b = 1 UNION ALL SELECT l.x FROM l",
    # not UNION ALL, or more on the union
    "SELECT l.x FROM l WHERE l.b = 1 UNION SELECT l.x FROM l WHERE l.b = 2",
    "SELECT l.x FROM l WHERE l.b = 1 INTERSECT ALL SELECT l.x FROM l WHERE l.b = 2",
    "SELECT l.x FROM l WHERE l.b = 1 EXCEPT ALL SELECT l.x FROM l WHERE l.b = 2",
    "SELECT l.x FROM l WHERE l.b = 1 UNION ALL SELECT l.x FROM l WHERE l.b = 2 ORDER BY 1 LIMIT 3",
    "SELECT l.x FROM l WHERE l.b = 1 UNION ALL (SELECT l.x FROM l WHERE l.b = 2 UNION SELECT l.x FROM l WHERE l.b = 3)",
    # the branches are not the same query
    "SELECT l.x FROM l WHERE l.b = 1 UNION ALL SELECT l.q FROM l WHERE l.b = 2",
    "SELECT l.x FROM l WHERE l.b = 1 UNION ALL SELECT l.x FROM l JOIN p ON p.k = l.pk WHERE l.b = 2",
    "SELECT l.x FROM l JOIN p ON p.k = l.pk WHERE l.b = 1 UNION ALL SELECT l.x FROM l JOIN p ON p.k = l.q WHERE l.b = 2",
    "SELECT l.x FROM l AS l WHERE l.b = 1 UNION ALL SELECT l.x FROM p AS l WHERE l.b = 2",
    # a branch does more than filter
    "SELECT COUNT(*) FROM l WHERE l.b = 1 UNION ALL SELECT COUNT(*) FROM l WHERE l.b = 2",  # two rows, not one
    "SELECT DISTINCT l.x FROM l WHERE l.b = 1 UNION ALL SELECT DISTINCT l.x FROM l WHERE l.b = 2",
    "SELECT l.x FROM l WHERE l.b = 1 GROUP BY l.x UNION ALL SELECT l.x FROM l WHERE l.b = 2 GROUP BY l.x",
    "(SELECT l.x FROM l WHERE l.b = 1 LIMIT 1) UNION ALL (SELECT l.x FROM l WHERE l.b = 2 LIMIT 1)",
    "SELECT ROW_NUMBER() OVER () FROM l WHERE l.b = 1 UNION ALL SELECT ROW_NUMBER() OVER () FROM l WHERE l.b = 2",
    "SELECT RANDOM() FROM l WHERE l.b = 1 UNION ALL SELECT RANDOM() FROM l WHERE l.b = 2",
    "SELECT GENERATE_SERIES(1, l.x) FROM l WHERE l.b = 1 UNION ALL SELECT GENERATE_SERIES(1, l.x) FROM l WHERE l.b = 2",
])
def test_shapes_left_alone(sql):
    assert merged(sql) is None


TARGETS = [
    (
        "SELECT SUM(l.x) FROM l, p WHERE (p.k = l.pk AND p.b = 1 AND l.q < 5) OR (p.k = l.pk AND p.b = 2 AND l.q > 9)",
        "SELECT SUM(r) FROM (SELECT l.x AS r FROM l JOIN p ON p.k = l.pk WHERE p.b = 1 AND l.q < 5 "
        "UNION ALL SELECT l.x AS r FROM l JOIN p ON p.k = l.pk WHERE p.b = 2 AND l.q > 9) AS u",
    ),
    (
        "SELECT l.x, l.q FROM l WHERE l.b IN (1, 2, 3) AND l.q > 0",
        "SELECT l.x, l.q FROM l WHERE l.b = 1 AND l.q > 0 UNION ALL SELECT l.x, l.q FROM l WHERE l.b = 2 AND l.q > 0 UNION ALL SELECT l.x, l.q FROM l WHERE l.b = 3 AND l.q > 0",
    ),
]

NEAR_MISSES = [
    (  # overlapping filters: a row in both branches comes back twice
        "SELECT l.x FROM l WHERE l.b = 1 OR l.q = 2",
        "SELECT l.x FROM l WHERE l.b = 1 UNION ALL SELECT l.x FROM l WHERE l.q = 2",
        {"l": [(7, 0, 2, 1)]},
    ),
    (  # the same filter twice
        "SELECT l.x FROM l WHERE l.b = 1",
        "SELECT l.x FROM l WHERE l.b = 1 UNION ALL SELECT l.x FROM l WHERE l.b = 1",
        {"l": [(7, 0, 2, 1)]},
    ),
    (  # a branch is missing from the single select
        "SELECT l.x FROM l WHERE l.b = 1",
        "SELECT l.x FROM l WHERE l.b = 1 UNION ALL SELECT l.x FROM l WHERE l.b = 2",
        {"l": [(7, 0, 2, 2)]},
    ),
    (  # one aggregate row per branch is not one aggregate row
        "SELECT COUNT(*) FROM l WHERE l.b = 1 OR l.b = 2",
        "SELECT COUNT(*) FROM l WHERE l.b = 1 UNION ALL SELECT COUNT(*) FROM l WHERE l.b = 2",
        {"l": [(7, 0, 2, 2)]},
    ),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_targets_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


@pytest.mark.parametrize("left, right, rows", NEAR_MISSES)
def test_near_misses_are_not_proved(left, right, rows):
    assert differ_on_duckdb(left, right, rows)
    assert not proved(left, right)
    assert not proved(right, left)


REPLAYED = [
    "SELECT l.x FROM l JOIN p ON p.k = l.pk WHERE p.b = 1 AND l.q < 2 UNION ALL SELECT l.x FROM l JOIN p ON p.k = l.pk WHERE p.b = 2 AND l.q > 0",
    "SELECT SUM(r), COUNT(*) FROM (SELECT x AS r FROM l WHERE b = 1 UNION ALL (SELECT x AS r FROM l WHERE 2 = b UNION ALL SELECT x AS r FROM l WHERE b = 3 AND q > 0)) AS u",
    "SELECT l.x, p.s FROM l, p WHERE l.b = 0 AND p.k IS NULL UNION ALL SELECT l.x, p.s FROM l, p WHERE l.b = 1 AND (p.k = l.pk OR l.q IS NULL) UNION ALL SELECT l.x, p.s FROM l, p WHERE NOT l.q = 1 AND l.b = 3",
]


@pytest.mark.parametrize("sql", REPLAYED)
def test_the_rewrite_returns_the_same_rows(sql):
    after = merged(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 3])
    for _ in range(40):
        rows = {
            "l": [tuple(value() for _ in range(4)) for _ in range(rng.randint(0, 5))],
            "p": [(value(), value(), rng.choice([None, "a", "b"])) for _ in range(rng.randint(0, 4))],
        }
        first, second = rows_on_duckdb([sql, after], rows)
        assert first == second, rows
