"""A keyed table outer-joined with counts per key is the count over the outer join (``kumo_sql_prover.counted_outer_groups``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.counted_outer_groups import count_outer_groups
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"c": ["k", "name", "fk"], "o": ["id", "ck", "note", "x"], "h": ["ck", "x"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}
KEYS = {"c": [("k",)], "o": [("id",)]}
CONSTRAINTS = {
    "c": TableConstraints(not_null=frozenset({"k"}), keys=(("k",),)),
    "o": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
}


def counted(sql, dialect="postgres"):
    before = sqlglot.parse_one(sql, read=dialect)
    text = before.sql(dialect=dialect)
    after = count_outer_groups(before, SCHEMA, KEYS).sql(dialect=dialect)
    return None if after == text else after


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


COUNTS = "(SELECT o.ck, COUNT(*) AS n FROM o WHERE o.note <> 3 GROUP BY o.ck) AS g"


def test_the_counts_table_becomes_a_count():
    assert (
        counted(f"SELECT c.k, COALESCE(g.n, 0) AS orders FROM c LEFT JOIN {COUNTS} ON c.k = g.ck")
        == "SELECT c.k, COUNT(g.id) AS orders FROM c LEFT JOIN o AS g ON c.k = g.ck AND g.note <> 3 GROUP BY c.k"
    )
    # bare columns inside, the count first, COUNT(x), a filter on the keyed table, another column joined, inside a derived table
    assert (
        counted("SELECT d.m, COUNT(*) FROM (SELECT a.name, COALESCE(t.m, 0) + 1 AS m FROM c AS a LEFT OUTER JOIN "
                "(SELECT COUNT(x) AS m, ck AS j FROM o GROUP BY ck) AS t ON t.j = a.fk WHERE a.name > 0) AS d GROUP BY d.m")
        == "SELECT d.m, COUNT(*) FROM (SELECT a.name, COUNT(t.x) + 1 AS m FROM c AS a LEFT OUTER JOIN o AS t ON t.ck = a.fk WHERE a.name > 0 GROUP BY a.k, a.name) AS d GROUP BY d.m"
    )
    # a table without a declared key counts the joined column
    assert (
        counted("SELECT c.k, COALESCE(g.n, 0) FROM c LEFT JOIN (SELECT h.ck, COUNT(*) AS n FROM h GROUP BY h.ck) AS g ON c.k = g.ck")
        == "SELECT c.k, COUNT(g.ck) FROM c LEFT JOIN h AS g ON c.k = g.ck GROUP BY c.k"
    )


@pytest.mark.parametrize("sql", [
    f"SELECT c.k, g.n FROM c LEFT JOIN {COUNTS} ON c.k = g.ck",  # the bare count is NULL without a match, not 0
    f"SELECT c.k, COALESCE(g.n, 1) FROM c LEFT JOIN {COUNTS} ON c.k = g.ck",
    f"SELECT c.k, COALESCE(g.n, 0), g.ck FROM c LEFT JOIN {COUNTS} ON c.k = g.ck",  # the key is read
    f"SELECT c.k, COALESCE(g.n, 0) FROM c JOIN {COUNTS} ON c.k = g.ck",  # an inner join drops the unmatched
    f"SELECT c.k, COALESCE(g.n, 0) FROM c RIGHT JOIN {COUNTS} ON c.k = g.ck",
    f"SELECT c.k, COALESCE(g.n, 0) FROM c LEFT JOIN {COUNTS} ON c.k = g.ck AND c.name > 0",
    f"SELECT c.k, COALESCE(g.n, 0) FROM c LEFT JOIN {COUNTS} ON c.k < g.ck",
    f"SELECT c.k, COALESCE(g.n, 0) FROM c LEFT JOIN {COUNTS} ON c.k = g.n",  # joined on the count
    f"SELECT c.k, COALESCE(g.n, 0) FROM c LEFT JOIN {COUNTS} ON c.k = g.ck WHERE COALESCE(g.n, 0) > 1",  # a filter on the count
    f"SELECT DISTINCT COALESCE(g.n, 0) FROM c LEFT JOIN {COUNTS} ON c.k = g.ck",
    f"SELECT k, COALESCE(n, 0) FROM c LEFT JOIN {COUNTS} ON k = g.ck",  # bare names outside
    f"SELECT h.ck, COALESCE(g.n, 0) FROM h LEFT JOIN {COUNTS} ON h.ck = g.ck",  # no key: a row of h is not a group
    f"SELECT c.k, COALESCE(g.n, 0) FROM c LEFT JOIN {COUNTS} ON c.k = g.ck LEFT JOIN h ON h.ck = c.k",
    "SELECT c.k, COALESCE(g.n, 0) FROM c LEFT JOIN (SELECT o.ck, SUM(o.x) AS n FROM o GROUP BY o.ck) AS g ON c.k = g.ck",  # a sum of no rows is NULL
    "SELECT c.k, COALESCE(g.n, 0) FROM c LEFT JOIN (SELECT o.ck, COUNT(DISTINCT o.x) AS n FROM o GROUP BY o.ck) AS g ON c.k = g.ck",
    "SELECT c.k, COALESCE(g.n, 0) FROM c LEFT JOIN (SELECT o.ck, COUNT(*) AS n FROM o GROUP BY o.ck, o.note) AS g ON c.k = g.ck",  # several rows per key
    "SELECT c.k, COALESCE(g.n, 0) FROM c LEFT JOIN (SELECT o.ck, COUNT(*) AS n FROM o GROUP BY o.ck HAVING COUNT(*) > 1) AS g ON c.k = g.ck",
    "SELECT c.k, COALESCE(g.n, 0) FROM c LEFT JOIN (SELECT o.ck, COUNT(*) AS n FROM o WHERE o.x IN (SELECT h.x FROM h) GROUP BY o.ck) AS g ON c.k = g.ck",
    "SELECT c.k, COALESCE(g.n, 0) FROM c LEFT JOIN (SELECT o.ck + 1 AS ck, COUNT(*) AS n FROM o GROUP BY o.ck + 1) AS g ON c.k = g.ck",
    "SELECT c.k, COALESCE(g.n, 0) FROM c LEFT JOIN (SELECT o.ck, COUNT(*) AS n FROM o WHERE RANDOM() < 0.5 GROUP BY o.ck) AS g ON c.k = g.ck",
])
def test_shapes_left_alone(sql):
    assert counted(sql) is None


def test_proofs_and_near_misses():
    joined = "SELECT c.k, COUNT(o.id) FROM c LEFT JOIN o ON c.k = o.ck AND o.note <> 3 GROUP BY c.k"
    before = f"SELECT c.k, COALESCE(g.n, 0) FROM c LEFT JOIN {COUNTS} ON c.k = g.ck"
    assert proved(joined, before) and proved(before, joined)
    histogram = "SELECT t.n, COUNT(*) FROM (SELECT c.k, COUNT(o.id) AS n FROM c LEFT JOIN o ON c.k = o.ck AND o.note <> 3 GROUP BY c.k) AS t GROUP BY t.n"
    assert proved(histogram, f"SELECT t.n, COUNT(*) FROM (SELECT c.k, COALESCE(g.n, 0) AS n FROM c LEFT JOIN {COUNTS} ON c.k = g.ck) AS t GROUP BY t.n")

    # the filter after the join drops the customers without a passing order
    late = "SELECT c.k, COUNT(o.id) FROM c LEFT JOIN o ON c.k = o.ck WHERE o.note <> 3 GROUP BY c.k"
    first, second = rows_on_duckdb([before, late], {"c": [(1, 0, 0)], "o": []})
    assert first != second
    assert not proved(before, late) and not proved(late, before)
    # the bare count is NULL for a customer without orders
    bare = f"SELECT c.k, g.n FROM c LEFT JOIN {COUNTS} ON c.k = g.ck"
    first, second = rows_on_duckdb([joined, bare], {"c": [(1, 0, 0)], "o": []})
    assert first != second
    assert not proved(joined, bare) and not proved(bare, joined)
    # COUNT(x) is not COUNT(*)
    of_x = "SELECT c.k, COALESCE(g.n, 0) FROM c LEFT JOIN (SELECT o.ck, COUNT(o.x) AS n FROM o WHERE o.note <> 3 GROUP BY o.ck) AS g ON c.k = g.ck"
    first, second = rows_on_duckdb([joined, of_x], {"c": [(1, 0, 0)], "o": [(7, 1, 0, None)]})
    assert first != second
    assert not proved(joined, of_x) and not proved(of_x, joined)


@pytest.mark.parametrize("sql", [
    f"SELECT c.k, c.name, COALESCE(g.n, 0) AS orders FROM c LEFT JOIN {COUNTS} ON c.k = g.ck WHERE c.name > 0 OR c.fk IS NULL",
    "SELECT a.name, COALESCE(t.m, 0) + 1 AS m FROM c AS a LEFT OUTER JOIN (SELECT COUNT(x) AS m, ck AS j FROM o GROUP BY ck) AS t ON t.j = a.fk",
    "SELECT c.k, COALESCE(g.n, 0) * 2, COALESCE(g.n, 0) FROM c LEFT JOIN (SELECT h.ck, COUNT(*) AS n FROM h WHERE h.x IS NULL OR h.x > 1 GROUP BY h.ck) AS g ON c.fk = g.ck",
])
def test_the_rewrite_returns_the_same_rows(sql):
    after = counted(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 2, 3])
    for _ in range(40):
        rows = {
            "c": [(k, value(), value()) for k in rng.sample(range(5), rng.randint(0, 4))],
            "o": [(i, value(), value(), value()) for i in rng.sample(range(8), rng.randint(0, 7))],
            "h": [(value(), value()) for _ in range(rng.randint(0, 5))],
        }
        first, second = rows_on_duckdb([sql, after], rows)
        assert first == second, rows
