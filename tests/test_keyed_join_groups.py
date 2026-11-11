"""Keyed tables joined to a pre-aggregated table are the aggregate over the flat join (``kumo_sql_prover.keyed_join_groups``)."""

import random

import pytest
import sqlglot
from sqlglot import exp

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.keyed_join_groups import pull_up_keyed_join
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"c": ["ck", "name"], "o": ["k", "ck", "price"], "l": ["k", "n", "q"], "h": ["x", "y"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}
KEYS = {"c": [("ck",)], "o": [("k",)], "l": [("k", "n")]}
CONSTRAINTS = {
    "c": TableConstraints(not_null=frozenset({"ck"}), keys=(("ck",),)),
    "o": TableConstraints(not_null=frozenset({"k"}), keys=(("k",),)),
    "l": TableConstraints(not_null=frozenset({"k", "n"}), keys=(("k", "n"),)),
}


def pulled(sql, dialect="postgres"):
    out = pull_up_keyed_join(sqlglot.parse_one(sql, read=dialect), SCHEMA, KEYS)
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


def test_the_join_becomes_a_grouped_join():
    assert (
        pulled(
            "SELECT c.name, o.k, g.total FROM c JOIN o ON c.ck = o.ck "
            "JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k HAVING SUM(l.q) > 300) AS g ON o.k = g.k "
            "ORDER BY o.price DESC LIMIT 100"
        )
        == "SELECT c.name, o.k, SUM(l.q) AS total FROM c, o, l WHERE c.ck = o.ck AND o.k = l.k "
        "GROUP BY c.ck, c.name, o.k, o.ck, o.price, l.k HAVING SUM(l.q) > 300 ORDER BY o.price DESC LIMIT 100"
    )
    # bare names are resolved, a filter on an aggregate goes to HAVING, the grouped table's own filter to WHERE
    assert (
        pulled(
            "SELECT name, o.k, total + 1 FROM c, o, (SELECT k AS kk, SUM(q) AS total FROM l WHERE n > 1 GROUP BY k) AS g "
            "WHERE c.ck = o.ck AND o.k = kk AND (total > 5 OR price > 3) ORDER BY total, 2"
        )
        == "SELECT c.name, o.k, SUM(l.q) + 1 FROM c, o, l WHERE c.ck = o.ck AND o.k = l.k AND l.n > 1 "
        "GROUP BY c.ck, c.name, o.k, o.ck, o.price, l.k HAVING (SUM(l.q) > 5 OR o.price > 3) ORDER BY SUM(l.q), 2"
    )
    # a source of the grouped table that shares a name with the select gets a fresh alias
    out = pulled("SELECT o.k, g.total FROM o JOIN (SELECT o.ck, COUNT(*) AS total FROM o GROUP BY o.ck) AS g ON o.ck = g.ck")
    assert out.startswith("SELECT o.k, COUNT(*) AS total FROM o, o AS kqp") and "GROUP BY o.k, o.ck, kqp" in out
    # an output name in ORDER BY keeps meaning the output
    assert (
        pulled("SELECT o.k, g.total FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k ORDER BY total")
        == "SELECT o.k, SUM(l.q) AS total FROM o, l WHERE o.k = l.k GROUP BY o.k, l.k ORDER BY total"
    )


@pytest.mark.parametrize("sql", [
    # a source without a declared key: two equal rows would share a group
    "SELECT h.x, g.total FROM h JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON h.x = g.k",
    "SELECT o.k, g.total FROM o JOIN h ON h.x = o.k JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k",
    # outer joins pad rows
    "SELECT o.k, g.total FROM o LEFT JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k",
    "SELECT o.k, g.total FROM (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g RIGHT JOIN o ON o.k = g.k",
    "SELECT o.k, g.total FROM o FULL JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k",
    "SELECT o.k, g.total FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g USING (k)",
    # the select does more than join
    "SELECT DISTINCT o.ck, g.total FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k",
    "SELECT o.ck, SUM(g.total) FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k GROUP BY o.ck",
    "SELECT MAX(g.total) FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k",
    "SELECT o.k, ROW_NUMBER() OVER (ORDER BY g.total) FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k",
    "SELECT o.k, g.total FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k WHERE EXISTS (SELECT 1 FROM c WHERE c.ck = o.ck)",
    "SELECT * FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k",
    # two derived tables, or none
    "SELECT o.k, g.total FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k JOIN (SELECT l.k, COUNT(*) AS n FROM l GROUP BY l.k) AS f ON o.k = f.k",
    "SELECT o.k, c.name FROM o JOIN c ON o.ck = c.ck",
    # the derived table is not a plain grouped select
    "SELECT o.k, g.q FROM o JOIN (SELECT l.k, l.q FROM l) AS g ON o.k = g.k",
    "SELECT o.k, g.total FROM o JOIN (SELECT SUM(l.q) AS total FROM l) AS g ON o.k = g.total",
    "SELECT o.k, g.total FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k LIMIT 5) AS g ON o.k = g.k",
    "SELECT o.k, g.total FROM o JOIN (SELECT DISTINCT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k",
    "SELECT o.k, g.total FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY ROLLUP (l.k)) AS g ON o.k = g.k",
    "SELECT o.k, g.total FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k HAVING MAX(l.q) > (SELECT MAX(x) FROM h)) AS g ON o.k = g.k",
    "SELECT o.k, g.total FROM o JOIN (SELECT l.k, SUM(l.q) OVER () AS total FROM l GROUP BY l.k, l.q) AS g ON o.k = g.k",
    "SELECT o.k, g.total FROM o JOIN (SELECT l.k, l.k + SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k",  # a key outside the aggregate
    "SELECT o.k, g.total FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l LEFT JOIN h ON h.x = l.k GROUP BY l.k) AS g ON o.k = g.k",
    "SELECT o.k, g.total FROM o JOIN (SELECT u.k, SUM(u.q) AS total FROM unknown_table AS u GROUP BY u.k) AS g ON o.k = g.k",
    "SELECT o.k, g.total FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g(a, total) ON o.k = g.a",
    # a name that does not resolve to one source
    "SELECT k, g.total FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k",
    "SELECT o.k, g.total FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k WHERE g.missing = 1",
    "SELECT o.k, g.total FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k LIMIT (SELECT 1)",
])
def test_shapes_left_alone(sql):
    assert pulled(sql) is None


def test_inside_a_membership_test_the_join_stays():
    # the prover reads the query of an IN or EXISTS without grouping, so the pre-aggregated table stays there
    tree = sqlglot.parse_one(
        "SELECT c.name FROM c WHERE c.ck IN (SELECT o.ck FROM o JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k WHERE o.price > g.total)",
        read="postgres",
    )
    inner = tree.find(exp.In).args["query"].this
    assert pull_up_keyed_join(inner, SCHEMA, KEYS) is None
    assert pull_up_keyed_join(inner.copy(), SCHEMA, KEYS) is not None  # the same select on its own is read


TARGETS = [
    (  # the pre-aggregated join against the grouped join, two keyed tables, HAVING and a tail
        "SELECT c.name, c.ck, o.k, o.price, SUM(l.q) FROM c, o, l WHERE c.ck = o.ck AND o.k = l.k "
        "GROUP BY c.name, c.ck, o.k, o.price HAVING SUM(l.q) > 300 ORDER BY o.price DESC LIMIT 10",
        "SELECT c.name, c.ck, o.k, o.price, g.total FROM c JOIN o ON c.ck = o.ck "
        "JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k HAVING SUM(l.q) > 300) AS g ON o.k = g.k ORDER BY o.price DESC LIMIT 10",
    ),
    (  # the filter on the total written outside
        "SELECT c.name, o.k, SUM(l.q), COUNT(*) FROM c, o, l WHERE c.ck = o.ck AND o.k = l.k GROUP BY c.ck, c.name, o.k HAVING SUM(l.q) > 300",
        "SELECT c.name, o.k, g.total, g.n FROM c, o, (SELECT l.k, SUM(l.q) AS total, COUNT(*) AS n FROM l GROUP BY l.k) AS g "
        "WHERE c.ck = o.ck AND o.k = g.k AND g.total > 300",
    ),
]

NEAR_MISSES = [
    (  # another bound
        "SELECT c.name, o.k, SUM(l.q) FROM c, o, l WHERE c.ck = o.ck AND o.k = l.k GROUP BY c.ck, c.name, o.k HAVING SUM(l.q) > 300",
        "SELECT c.name, o.k, g.total FROM c JOIN o ON c.ck = o.ck JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k HAVING SUM(l.q) > 200) AS g ON o.k = g.k",
        {"c": [(1, 7)], "o": [(5, 1, 9)], "l": [(5, 1, 250)]},
    ),
    (  # the flat join grouped by customer only: several orders share a group
        "SELECT c.name, SUM(l.q) FROM c, o, l WHERE c.ck = o.ck AND o.k = l.k GROUP BY c.ck, c.name",
        "SELECT c.name, g.total FROM c JOIN o ON c.ck = o.ck JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k",
        {"c": [(1, 7)], "o": [(5, 1, 9), (6, 1, 9)], "l": [(5, 1, 2), (6, 1, 3)]},
    ),
    (  # a table without a key: equal rows are one group in the flat join and two rows in the other
        "SELECT h.x, SUM(l.q) FROM h, l WHERE h.x = l.k GROUP BY h.x, h.y",
        "SELECT h.x, g.total FROM h JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON h.x = g.k",
        {"h": [(5, 1), (5, 1)], "l": [(5, 1, 2)]},
    ),
    (  # a LEFT JOIN keeps the orders without lines
        "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k GROUP BY o.k",
        "SELECT o.k, g.total FROM o LEFT JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k) AS g ON o.k = g.k",
        {"o": [(5, 1, 9)], "l": []},
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
    "SELECT c.name, o.k, g.total FROM c JOIN o ON c.ck = o.ck JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k HAVING SUM(l.q) > 2) AS g ON o.k = g.k",
    "SELECT name, o.k, total + 1 FROM c, o, (SELECT k AS kk, SUM(q) AS total FROM l WHERE n > 1 GROUP BY k) AS g WHERE c.ck = o.ck AND o.k = kk AND (total > 3 OR price > 1)",
    "SELECT o.k, g.total, g.n FROM o JOIN (SELECT o.ck, COUNT(*) AS total, MAX(l.q) AS n FROM o JOIN l ON l.k = o.k GROUP BY o.ck) AS g ON o.ck = g.ck",
    "SELECT o.k, o.price, g.a, g.b FROM o, (SELECT l.q AS a, COUNT(DISTINCT l.k) AS b FROM l GROUP BY l.q HAVING MIN(l.n) < 3) AS g WHERE g.b <> o.price",
    "SELECT c.ck, g.x, g.t FROM c, (SELECT h.x, SUM(h.y) AS t FROM h GROUP BY h.x) AS g WHERE c.name = g.x OR g.t IS NULL",
]


@pytest.mark.parametrize("sql", REPLAYED)
def test_the_rewrite_returns_the_same_rows(sql):
    after = pulled(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 3])
    for _ in range(40):
        rows = {
            "c": [(key, value()) for key in rng.sample(range(4), rng.randint(0, 3))],
            "o": [(key, value(), value()) for key in rng.sample(range(4), rng.randint(0, 4))],
            "l": [(k, n, value()) for k, n in rng.sample([(a, b) for a in range(4) for b in range(3)], rng.randint(0, 6))],
            "h": [(value(), value()) for _ in range(rng.randint(0, 4))],
        }
        first, second = rows_on_duckdb([sql, after], rows)
        assert first == second, rows
