"""A made-up three-table schema, rows and a DuckDB runner shared by the tests of the 2026-11-11 rule modules."""

import random

import pytest
import sqlglot

SCHEMA = {"p": ["id", "owner", "kind", "day"], "u": ["id", "name", "rep"], "c": ["id", "post", "score"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}
ROWS = {
    "p": [(1, 1, 1, 10), (2, 1, 1, 20), (3, None, 1, 30), (4, 2, 2, 40), (5, 9, 1, 50), (6, 2, 1, 20), (7, 1, 2, 5), (8, 3, 1, 60)],
    "u": [(1, 7, 5), (2, 8, 6), (3, 9, 0), (5, 4, 1)],
    "c": [(1, 1, 3), (2, 1, 0), (3, 4, 5), (4, None, 1), (5, 2, None), (6, 4, 5)],
}


def proved(left, right):
    from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic

    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


def rows_on_duckdb(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name, columns in SCHEMA.items():
        db.execute(f"CREATE TABLE {name} ({', '.join(f'{column} BIGINT' for column in columns)})")
        insert_rows(db, name, rows.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def differ_on_duckdb(left, right, rows=None):
    a, b = rows_on_duckdb([left, right], ROWS if rows is None else rows)
    return a != b


def random_databases(seed, count=14):
    rng = random.Random(seed)

    def table(width, size):
        rows = [tuple(rng.choice([None, 1, 2, 3]) for _ in range(width)) for _ in range(rng.randint(0, size))]
        return rows + rows[:2]  # with duplicate rows

    return [{}, ROWS] + [{"p": table(4, 7), "u": table(3, 4), "c": table(3, 6)} for _ in range(count)]
