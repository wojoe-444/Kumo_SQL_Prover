"""The guard of a count over an outer-joined table moves into the join condition (``kumo_sql_prover.guarded_outer_counts``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.guarded_outer_counts import guard_outer_counts
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"c": ["k", "name"], "o": ["id", "ck", "note", "x"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}
NOT_NULL = {"c": frozenset({"k"}), "o": frozenset({"id"})}
CONSTRAINTS = {
    "c": TableConstraints(not_null=frozenset({"k"}), keys=(("k",),)),
    "o": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
}


def guarded(sql, dialect="postgres"):
    out = guard_outer_counts(sqlglot.parse_one(sql, read=dialect), SCHEMA, NOT_NULL)
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


def test_the_guard_moves_into_the_join():
    assert (
        guarded("SELECT c.k, COUNT(CASE WHEN o.id IS NOT NULL AND o.note <> 3 THEN o.id END) FROM c LEFT JOIN o ON c.k = o.ck GROUP BY c.k")
        == "SELECT c.k, COUNT(o.id) FROM c LEFT JOIN o ON c.k = o.ck AND o.note <> 3 GROUP BY c.k"
    )
    # a nullable column keeps its IS NOT NULL; an OR is parenthesized; two counts under one guard; grouped by another column
    assert (
        guarded("SELECT c.name, COUNT(CASE WHEN o.x IS NOT NULL OR c.name = 1 THEN o.id END), COUNT(CASE WHEN o.x IS NOT NULL OR c.name = 1 THEN o.x ELSE NULL END) + 1 "
                "FROM c LEFT OUTER JOIN o ON c.k = o.ck WHERE c.k > 0 GROUP BY c.name HAVING COUNT(CASE WHEN o.x IS NOT NULL OR c.name = 1 THEN o.id END) < 9")
        == "SELECT c.name, COUNT(o.id), COUNT(o.x) + 1 FROM c LEFT OUTER JOIN o ON c.k = o.ck AND (o.x IS NOT NULL OR c.name = 1) "
        "WHERE c.k > 0 GROUP BY c.name HAVING COUNT(o.id) < 9"
    )
    # a guard that is only a NOT NULL fact of the table goes
    assert guarded("SELECT COUNT(CASE WHEN o.id IS NOT NULL THEN o.x END) FROM c LEFT JOIN o ON c.k = o.ck") == "SELECT COUNT(o.x) FROM c LEFT JOIN o ON c.k = o.ck"


@pytest.mark.parametrize("sql", [
    "SELECT c.k, COUNT(CASE WHEN o.note <> 3 THEN o.id END), COUNT(*) FROM c LEFT JOIN o ON c.k = o.ck GROUP BY c.k",  # COUNT(*) sees the joined rows
    "SELECT c.k, COUNT(CASE WHEN o.note <> 3 THEN o.id END), COUNT(CASE WHEN o.note <> 4 THEN o.id END) FROM c LEFT JOIN o ON c.k = o.ck GROUP BY c.k",  # two guards
    "SELECT c.k, COUNT(CASE WHEN o.note <> 3 THEN o.id END), SUM(c.name) FROM c LEFT JOIN o ON c.k = o.ck GROUP BY c.k",
    "SELECT c.k, COUNT(CASE WHEN o.note <> 3 THEN c.k END) FROM c LEFT JOIN o ON c.k = o.ck GROUP BY c.k",  # the left column counts a padded row
    "SELECT c.k, COUNT(CASE WHEN o.note <> 3 THEN 1 END) FROM c LEFT JOIN o ON c.k = o.ck GROUP BY c.k",
    "SELECT c.k, COUNT(CASE WHEN o.note <> 3 THEN o.id ELSE 0 END) FROM c LEFT JOIN o ON c.k = o.ck GROUP BY c.k",
    "SELECT c.k, COUNT(DISTINCT CASE WHEN o.note <> 3 THEN o.id END) FROM c LEFT JOIN o ON c.k = o.ck GROUP BY c.k",
    "SELECT c.k, o.note, COUNT(CASE WHEN o.note <> 3 THEN o.id END) FROM c LEFT JOIN o ON c.k = o.ck GROUP BY c.k, o.note",  # grouped by the joined table
    "SELECT c.k, COUNT(CASE WHEN o.note <> 3 THEN o.id END) FROM c LEFT JOIN o ON c.k = o.ck WHERE o.x IS NULL GROUP BY c.k",  # WHERE reads it
    "SELECT c.k, COUNT(CASE WHEN o.note <> 3 THEN o.id END) FROM c JOIN o ON c.k = o.ck GROUP BY c.k",  # an inner join loses the customers without a passing order
    "SELECT c.k, COUNT(CASE WHEN o.note <> 3 THEN o.id END) FROM c RIGHT JOIN o ON c.k = o.ck GROUP BY c.k",
    "SELECT c.k, COUNT(CASE WHEN o.note <> 3 THEN o.id END) FROM c LEFT JOIN o ON c.k = o.ck LEFT JOIN o AS p ON p.ck = c.k GROUP BY c.k",
    "SELECT c.k, COUNT(CASE WHEN o.note <> 3 THEN o.id END) OVER () FROM c LEFT JOIN o ON c.k = o.ck",
    "SELECT c.k, COUNT(CASE WHEN o.note IN (SELECT z.note FROM o AS z) THEN o.id END) FROM c LEFT JOIN o ON c.k = o.ck GROUP BY c.k",
    "SELECT c.k, COUNT(CASE WHEN RANDOM() < 0.5 THEN o.id END) FROM c LEFT JOIN o ON c.k = o.ck GROUP BY c.k",
    "SELECT k, COUNT(CASE WHEN note <> 3 THEN id END) FROM c LEFT JOIN o ON k = ck GROUP BY k",  # bare names
    "SELECT c.k, COUNT(o.id) FROM c LEFT JOIN o ON c.k = o.ck GROUP BY c.k",  # no guard
])
def test_shapes_left_alone(sql):
    assert guarded(sql) is None


def test_proofs_and_near_misses():
    in_join = "SELECT c.k, COUNT(o.id) FROM c LEFT JOIN o ON c.k = o.ck AND o.note <> 3 GROUP BY c.k"
    in_count = "SELECT c.k, COUNT(CASE WHEN o.id IS NOT NULL AND o.note <> 3 THEN o.id END) FROM c LEFT JOIN o ON c.k = o.ck GROUP BY c.k"
    assert proved(in_join, in_count) and proved(in_count, in_join)

    # the guard in WHERE drops the customers without a passing order
    in_where = "SELECT c.k, COUNT(o.id) FROM c LEFT JOIN o ON c.k = o.ck WHERE o.note <> 3 GROUP BY c.k"
    first, second = rows_on_duckdb([in_count, in_where], {"c": [(1, 0)], "o": []})
    assert first != second
    assert not proved(in_count, in_where) and not proved(in_where, in_count)
    # another guard
    other = "SELECT c.k, COUNT(CASE WHEN o.note <> 4 THEN o.id END) FROM c LEFT JOIN o ON c.k = o.ck GROUP BY c.k"
    first, second = rows_on_duckdb([in_join, other], {"c": [(1, 0)], "o": [(7, 1, 3, 0)]})
    assert first != second
    assert not proved(in_join, other) and not proved(other, in_join)
    # COUNT(*) under the moved guard counts the padded row
    star = "SELECT c.k, COUNT(*) FROM c LEFT JOIN o ON c.k = o.ck AND o.note <> 3 GROUP BY c.k"
    first, second = rows_on_duckdb([in_count, star], {"c": [(1, 0)], "o": []})
    assert first != second
    assert not proved(in_count, star) and not proved(star, in_count)


@pytest.mark.parametrize("sql", [
    "SELECT c.k, COUNT(CASE WHEN o.id IS NOT NULL AND o.note <> 3 THEN o.id END) FROM c LEFT JOIN o ON c.k = o.ck GROUP BY c.k",
    "SELECT c.name, COUNT(CASE WHEN o.x IS NOT NULL OR c.name = 1 THEN o.id END), COUNT(CASE WHEN o.x IS NOT NULL OR c.name = 1 THEN o.x ELSE NULL END) + 1 "
    "FROM c LEFT OUTER JOIN o ON c.k = o.ck WHERE c.k > 0 GROUP BY c.name HAVING COUNT(CASE WHEN o.x IS NOT NULL OR c.name = 1 THEN o.id END) < 9",
    "SELECT COUNT(CASE WHEN o.note = c.name THEN o.x END) FROM c LEFT JOIN o ON c.k < o.ck",
])
def test_the_rewrite_returns_the_same_rows(sql):
    after = guarded(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 2, 3])
    for _ in range(40):
        rows = {
            "c": [(k, value()) for k in rng.sample(range(5), rng.randint(0, 4))],
            "o": [(i, value(), value(), value()) for i in rng.sample(range(8), rng.randint(0, 7))],
        }
        first, second = rows_on_duckdb([sql, after], rows)
        assert first == second, rows
