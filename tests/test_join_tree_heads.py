"""A parenthesized join tree whose first source is a derived table is read like one whose first source is a table.

``FROM ((SELECT ..) AS a JOIN b ON p)`` is ``FROM (SELECT ..) AS a JOIN b ON p``: the parentheses around a whole
join tree change nothing. Joined in later, the tree is flattened under the conditions ``flatten_join_tree`` already
has (inner and cross joins only). Every proved pair is also run on random DuckDB databases (NULLs, duplicates,
empty tables); a pair that is not equivalent must stay unproved.
"""

import random
from collections import Counter

import pytest
import sqlglot

pytest.importorskip("z3")
duckdb = pytest.importorskip("duckdb")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.duckdb_load import insert_rows
from kumo_sql_prover.outer_join_flatten import flatten_join_tree

SCHEMA = {"a": ["k", "x"], "b": ["k", "y"], "c": ["k", "z"]}
FIRST = "(SELECT k, MIN(x) AS low FROM a GROUP BY k) AS m"

# (as written, the same query without the parentheses)
SAME = [
    (f"SELECT m.k, b.y FROM ({FIRST} JOIN b ON m.k = b.k AND m.low = b.y)", f"SELECT m.k, b.y FROM {FIRST} JOIN b ON m.k = b.k AND m.low = b.y"),
    (f"SELECT m.k, b.y FROM ({FIRST} LEFT JOIN b ON m.k = b.k) WHERE m.low > 0", f"SELECT m.k, b.y FROM {FIRST} LEFT JOIN b ON m.k = b.k WHERE m.low > 0"),
    (f"SELECT m.low, b.y, c.z FROM ({FIRST} JOIN b ON m.k = b.k) JOIN c ON c.k = m.k", f"SELECT m.low, b.y, c.z FROM {FIRST} JOIN b ON m.k = b.k JOIN c ON c.k = m.k"),
    (f"SELECT m.low, c.z FROM c JOIN ({FIRST} JOIN b ON m.k = b.k) ON c.k = m.k", f"SELECT m.low, c.z FROM c JOIN {FIRST} ON c.k = m.k JOIN b ON m.k = b.k"),
    (
        "SELECT u.k, b.y FROM ((SELECT k FROM a UNION ALL SELECT k FROM c) AS u JOIN b ON u.k = b.k)",
        "SELECT u.k, b.y FROM (SELECT k FROM a UNION ALL SELECT k FROM c) AS u JOIN b ON u.k = b.k",
    ),
]
# (as written, a query it differs from)
DIFFERENT = [
    (f"SELECT m.k, b.y FROM ({FIRST} LEFT JOIN b ON m.k = b.k)", f"SELECT m.k, b.y FROM {FIRST} JOIN b ON m.k = b.k"),
    (f"SELECT m.k, b.y FROM ({FIRST} JOIN b ON m.k = b.k AND m.low = b.y)", f"SELECT m.k, b.y FROM {FIRST} JOIN b ON m.k = b.k"),
    (f"SELECT m.low, c.z FROM c LEFT JOIN ({FIRST} JOIN b ON m.k = b.k) ON c.k = m.k", f"SELECT m.low, c.z FROM c LEFT JOIN {FIRST} ON c.k = m.k JOIN b ON m.k = b.k"),
]


def _differ(left: str, right: str, trials: int = 150) -> bool:
    rng = random.Random(7)
    con = duckdb.connect()
    for table, columns in SCHEMA.items():
        con.execute(f"CREATE TABLE {table} ({', '.join(c + ' INTEGER' for c in columns)})")
    for _ in range(trials):
        for table, columns in SCHEMA.items():
            con.execute(f"DELETE FROM {table}")
            insert_rows(con, table, [[rng.choice([None, 0, 1, 2, 12]) for _ in columns] for _ in range(rng.randint(0, 4))])
        if Counter(con.execute(left).fetchall()) != Counter(con.execute(right).fetchall()):
            return True
    return False


def _proven(left: str, right: str) -> bool:
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, compare_names=False).proven


@pytest.mark.parametrize("written, plain", SAME)
def test_a_join_tree_that_starts_with_a_derived_table_is_read(written, plain):
    assert not _differ(written, plain)
    assert _proven(written, plain)


@pytest.mark.parametrize("written, other", DIFFERENT)
def test_a_query_that_differs_is_not_proved(written, other):
    assert _differ(written, other)
    assert not _proven(written, other)


def test_the_tree_in_from_loses_only_its_parentheses():
    written, plain = SAME[1]
    read = flatten_join_tree(sqlglot.parse_one(written))
    assert read is not None and read.sql() == sqlglot.parse_one(plain).sql()


def test_what_is_not_a_join_tree_is_left_alone():
    for sql in (
        f"SELECT m.k FROM {FIRST} JOIN b ON m.k = b.k",  # no parentheses
        "SELECT j.k FROM (SELECT a.k FROM a JOIN b ON a.k = b.k) AS j",  # a derived table
        f"SELECT m.low, c.z FROM c LEFT JOIN ({FIRST} JOIN b ON m.k = b.k) ON c.k = m.k",  # joined in by an outer join
    ):
        assert flatten_join_tree(sqlglot.parse_one(sql)) is None
