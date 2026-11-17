"""A key-only ``GROUP BY`` over a join that reads a derived ``UNION`` is ``SELECT DISTINCT`` (``union_join_grouping``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.union_join_grouping import group_over_union_join_to_distinct

UNION = "(SELECT s1.x AS id, s1.w AS w FROM s1 UNION ALL SELECT s2.y AS id, s2.w AS w FROM s2) AS t"
GROUPED = f"SELECT c.k, c.a FROM {UNION}, c WHERE c.k = t.id AND t.w > 3 GROUP BY c.k, c.a"


def rewritten(sql: str) -> str | None:
    result = group_over_union_join_to_distinct(sqlglot.parse_one(sql, read="duckdb"))
    return None if result is None else result.sql(dialect="duckdb")


def test_a_key_only_grouping_over_the_join_is_a_distinct():
    assert rewritten(GROUPED) == f"SELECT DISTINCT c.k, c.a FROM {UNION}, c WHERE c.k = t.id AND t.w > 3"
    # keys in another order, an alias on an output, an explicit inner join, a UNION that removes duplicates itself
    assert rewritten(f"SELECT c.a AS b, c.k FROM c JOIN {UNION} ON c.k = t.id GROUP BY c.k, c.a") == f"SELECT DISTINCT c.a AS b, c.k FROM c JOIN {UNION} ON c.k = t.id"
    assert rewritten(GROUPED.replace("UNION ALL", "UNION")) is not None
    # a key of the union itself
    assert rewritten(f"SELECT t.id FROM {UNION}, c WHERE c.k = t.id GROUP BY t.id") == f"SELECT DISTINCT t.id FROM {UNION}, c WHERE c.k = t.id"


def test_a_key_that_is_not_output_still_splits_the_groups():
    some = GROUPED.replace("SELECT c.k, c.a", "SELECT c.k AS key")
    assert rewritten(some) == f"SELECT kqg.kq_g0 AS key FROM (SELECT DISTINCT c.k AS kq_g0, c.a AS kq_g1 FROM {UNION}, c WHERE c.k = t.id AND t.w > 3) AS kqg"


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT t.id FROM {UNION} GROUP BY t.id",  # the union alone: the older rule reads it
        "SELECT c.k, c.a FROM s1, c WHERE c.k = s1.x GROUP BY c.k, c.a",  # no derived union among the sources
        GROUPED.replace("SELECT c.k, c.a", "SELECT c.k, c.a, COUNT(*) AS n"),  # an aggregate
        GROUPED + " HAVING COUNT(*) > 1",
        GROUPED.replace("SELECT c.k, c.a", "SELECT c.k").replace("GROUP BY c.k, c.a", "GROUP BY c.k, c.a + 1"),  # some keys output, a key that is no column
        GROUPED.replace("SELECT c.k, c.a", "SELECT c.k, c.a + 1 AS a"),  # an output that is not a key
        GROUPED.replace("GROUP BY c.k, c.a", "GROUP BY ROLLUP (c.k, c.a)"),
        GROUPED.replace("GROUP BY c.k, c.a", "GROUP BY 1, 2"),  # positions
        GROUPED + " ORDER BY c.k LIMIT 3",
        GROUPED.replace("SELECT c.k, c.a", "SELECT DISTINCT c.k, c.a"),
        f"SELECT c.k, c.a FROM c LEFT JOIN {UNION} ON c.k = t.id GROUP BY c.k, c.a",  # an outer join
        GROUPED.replace("SELECT c.k, c.a", "SELECT c.k, c.a, ROW_NUMBER() OVER (ORDER BY c.k) AS r"),
    ],
)
def test_other_selects_keep_their_grouping(sql):
    assert rewritten(sql) is None


@pytest.mark.parametrize("seed", range(3))
def test_both_spellings_return_the_rows_duckdb_returns(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    queries = [
        GROUPED,
        f"SELECT c.a, t.id FROM c JOIN {UNION} ON c.k = t.id OR c.a IS NULL GROUP BY t.id, c.a",
        f"SELECT t.w FROM {UNION} CROSS JOIN c GROUP BY t.w",
        f"SELECT c.k, t.w FROM {UNION.replace('UNION ALL', 'UNION')}, c WHERE c.a = t.w GROUP BY c.k, t.w",
        f"SELECT c.a FROM {UNION}, c WHERE c.k = t.id GROUP BY c.k, c.a",
        f"SELECT t.w AS v, t.w AS again FROM {UNION} CROSS JOIN c GROUP BY t.w, c.a, c.k",
    ]
    for sql in queries:
        distinct = rewritten(sql)
        assert distinct is not None
        for _ in range(6):
            con = duckdb.connect()
            con.execute("CREATE TABLE s1 (x INT, w INT)")
            con.execute("CREATE TABLE s2 (y INT, w INT)")
            con.execute("CREATE TABLE c (k INT, a INT)")
            value = lambda: rng.choice([None, 1, 2, 4, 5])  # noqa: E731
            for table in ("s1", "s2", "c"):
                for _ in range(rng.randint(0, 6)):
                    con.execute(f"INSERT INTO {table} VALUES (?, ?)", [value(), value()])
            rows = lambda text: sorted(con.execute(text).fetchall(), key=repr)  # noqa: E731
            assert rows(sql) == rows(distinct), sql


z3 = pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic  # noqa: E402

SCHEMA = {"s1": ["x", "w"], "s2": ["y", "w"], "c": ["k", "a"], "f": ["k", "v"]}
TYPES = {"s1": {"x": "INT64", "w": "INT64"}, "s2": {"y": "INT64", "w": "INT64"}, "c": {"k": "INT64", "a": "INT64"}, "f": {"k": "INT64", "v": "NUMERIC"}}
FIRST = "SELECT c.k, c.a FROM s1, c WHERE c.k = s1.x AND s1.w > 3"
SECOND = "SELECT c.k, c.a FROM s2, c WHERE c.k = s2.y AND s2.w > 3"


def proven(left: str, right: str) -> bool:
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="bigquery").proven


def test_the_grouped_join_meets_the_union_of_joins():
    assert proven(GROUPED, f"{FIRST} UNION DISTINCT {SECOND}")
    # as a derived table of a query that aggregates over it, the way TPC-DS 54 reads its customers
    outer = "SELECT m.k, SUM(f.v) AS r FROM ({body}) AS m, f WHERE m.k = f.k GROUP BY m.k"
    assert proven(outer.format(body=GROUPED), outer.format(body=f"{FIRST} UNION DISTINCT {SECOND}"))


def test_a_union_that_keeps_duplicates_is_not_the_grouped_join():
    assert not proven(GROUPED, f"{FIRST} UNION ALL {SECOND}")
    assert not proven(GROUPED, f"{FIRST} UNION DISTINCT {SECOND.replace('s2.w > 3', 's2.w > 4')}")
