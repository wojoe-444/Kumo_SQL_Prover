"""A membership test against a table's own groups is ``HAVING`` where the query forms those groups (``kumo_sql_prover.own_group_filters``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.own_group_filters import filter_own_groups
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"c": ["ck", "name"], "o": ["k", "ck", "price"], "l": ["k", "n", "q"], "h": ["x", "y"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}
KEYS = {"c": [("ck",)], "o": [("k",)], "l": [("k", "n")]}
NOT_NULL = {"c": frozenset({"ck"}), "o": frozenset({"k"}), "l": frozenset({"k", "n"})}
CONSTRAINTS = {
    "c": TableConstraints(not_null=frozenset({"ck"}), keys=(("ck",),)),
    "o": TableConstraints(not_null=frozenset({"k"}), keys=(("k",),)),
    "l": TableConstraints(not_null=frozenset({"k", "n"}), keys=(("k", "n"),)),
}


def filtered(sql, dialect="postgres", not_null=NOT_NULL):
    out = filter_own_groups(sqlglot.parse_one(sql, read=dialect), SCHEMA, KEYS, not_null, dialect)
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


def test_the_membership_test_becomes_having():
    assert (
        filtered("SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300) GROUP BY o.k")
        == "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k GROUP BY o.k HAVING SUM(l.q) > 300"
    )
    # the spelling the normalizer gives the subquery, two keyed tables, a tail, an existing HAVING
    assert (
        filtered(
            "SELECT c.name, o.k, SUM(l.q) FROM c, o, l WHERE o.k IN (SELECT g.kk FROM (SELECT k AS kk, SUM(q) AS a FROM l GROUP BY k) AS g WHERE g.a > 300) "
            "AND c.ck = o.ck AND o.k = l.k GROUP BY c.name, c.ck, o.k, o.price HAVING COUNT(*) > 1 ORDER BY o.price LIMIT 3"
        )
        == "SELECT c.name, o.k, SUM(l.q) FROM c, o, l WHERE c.ck = o.ck AND o.k = l.k GROUP BY c.name, c.ck, o.k, o.price "
        "HAVING COUNT(*) > 1 AND SUM(l.q) > 300 ORDER BY o.price LIMIT 3"
    )
    # grouping by the order key alone fixes the customer through the join
    assert (
        filtered("SELECT o.k, SUM(l.q) FROM c JOIN o ON c.ck = o.ck JOIN l ON l.k = o.k WHERE l.k IN (SELECT k FROM l GROUP BY k HAVING COUNT(*) > 2) GROUP BY o.k")
        == "SELECT o.k, SUM(l.q) FROM c JOIN o ON c.ck = o.ck JOIN l ON l.k = o.k GROUP BY o.k HAVING COUNT(*) > 2"
    )
    # the table alone, its key declared NOT NULL
    assert (
        filtered("SELECT x.k, MAX(x.q) FROM l AS x WHERE x.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 3) GROUP BY x.k")
        == "SELECT x.k, MAX(x.q) FROM l AS x GROUP BY x.k HAVING SUM(x.q) > 3"
    )


def test_the_joined_groups_become_having():
    assert (
        filtered(
            "SELECT c.name, o.k, SUM(l.q) FROM c CROSS JOIN o CROSS JOIN l JOIN (SELECT l0.k AS k0 FROM l AS l0 GROUP BY l0.k HAVING SUM(l0.q) > 300) AS t ON o.k = t.k0 "
            "WHERE c.ck = o.ck AND o.k = l.k GROUP BY c.name, c.ck, o.k"
        )
        == "SELECT c.name, o.k, SUM(l.q) FROM c CROSS JOIN o CROSS JOIN l WHERE c.ck = o.ck AND o.k = l.k GROUP BY c.name, c.ck, o.k HAVING SUM(l.q) > 300"
    )
    # a bare name in an earlier ON is not a read of the joined groups
    assert (
        filtered("SELECT o.k, SUM(q) FROM o JOIN l ON o.k = k AND n > 0 OR FALSE JOIN (SELECT k FROM l GROUP BY k HAVING SUM(q) > 3) AS f ON f.k = o.k GROUP BY o.k")
        is None  # the ON reads l.n: the group is not all of the order's lines
    )
    assert (
        filtered("SELECT o.k, SUM(q) FROM o JOIN l ON o.k = l.k JOIN (SELECT k FROM l GROUP BY k HAVING SUM(q) > 3) AS f ON f.k = o.k GROUP BY o.k")
        == "SELECT o.k, SUM(q) FROM o JOIN l ON o.k = l.k GROUP BY o.k HAVING SUM(l.q) > 3"
    )


@pytest.mark.parametrize("sql", [
    # the group is not one order's lines: grouped by customer, several orders share a group
    "SELECT o.ck, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300) GROUP BY o.ck",
    # a filter on the lines: the group holds only some of the order's lines
    "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND l.n > 2 AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300) GROUP BY o.k",
    "SELECT o.k, SUM(l.q) FROM o JOIN l ON o.k = l.k AND l.q > 0 WHERE o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300) GROUP BY o.k",
    # a grouping column splits the order's lines
    "SELECT o.k, l.n, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300) GROUP BY o.k, l.n",
    # a joined table without a key repeats the lines
    "SELECT o.k, SUM(l.q) FROM o, l, h WHERE o.k = l.k AND h.x = o.k AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300) GROUP BY o.k",
    # a keyed table whose row the group does not fix
    "SELECT o.k, SUM(l.q) FROM o, l, c WHERE o.k = l.k AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300) GROUP BY o.k",
    # the tested value is not the group's key
    "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.ck IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300) GROUP BY o.k, o.ck",
    "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.k IN (SELECT l.n FROM l GROUP BY l.n HAVING SUM(l.q) > 300) GROUP BY o.k",
    # the subquery filters, joins, or groups by more
    "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.k IN (SELECT l.k FROM l WHERE l.n > 1 GROUP BY l.k HAVING SUM(l.q) > 300) GROUP BY o.k",
    "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.k IN (SELECT l.k FROM l JOIN h ON h.x = l.k GROUP BY l.k HAVING SUM(l.q) > 300) GROUP BY o.k",
    "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.k IN (SELECT l.k FROM l GROUP BY l.k, l.n HAVING SUM(l.q) > 300) GROUP BY o.k",
    "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING l.k > 3) GROUP BY o.k",
    "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.k IN (SELECT l.k FROM l GROUP BY l.k) GROUP BY o.k",  # nothing to move
    "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300 LIMIT 5) GROUP BY o.k",
    # another table's groups
    "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.k IN (SELECT h.x FROM h GROUP BY h.x HAVING SUM(h.y) > 300) GROUP BY o.k",
    # the table read twice outside
    "SELECT o.k, SUM(l.q) FROM o, l, l AS l2 WHERE o.k = l.k AND l2.k = o.k AND l2.n = 1 AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300) GROUP BY o.k",
    # not IN, under OR, NOT IN
    "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.k NOT IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300) GROUP BY o.k",
    "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND (o.price > 3 OR o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300)) GROUP BY o.k",
    # outer joins, another subquery, no grouping
    "SELECT o.k, SUM(l.q) FROM o LEFT JOIN l ON o.k = l.k WHERE o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300) GROUP BY o.k",
    "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300) AND EXISTS (SELECT 1 FROM h WHERE h.x = l.q) GROUP BY o.k",
    "SELECT o.k FROM o, l WHERE o.k = l.k AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300)",
    "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300) GROUP BY ROLLUP (o.k)",
    # the joined groups are read elsewhere too
    "SELECT o.k, t.k0, SUM(l.q) FROM o, l, (SELECT l0.k AS k0 FROM l AS l0 GROUP BY l0.k HAVING SUM(l0.q) > 300) AS t WHERE o.k = l.k AND o.k = t.k0 GROUP BY o.k, t.k0",
    "SELECT o.k, SUM(l.q) FROM o, l, (SELECT l0.k AS k0, SUM(l0.q) AS s FROM l AS l0 GROUP BY l0.k) AS t WHERE o.k = l.k AND o.k = t.k0 AND t.s > 300 GROUP BY o.k",
])
def test_shapes_left_alone(sql):
    assert filtered(sql) is None


def test_a_null_key_needs_an_equality_or_a_declaration():
    sql = "SELECT h.x, SUM(h.y) FROM h WHERE h.x IN (SELECT x FROM h GROUP BY x HAVING COUNT(*) > 1) GROUP BY h.x"
    assert filtered(sql) is None  # h.x may be NULL: the group of NULLs passes HAVING and fails IN
    assert differ_on_duckdb(sql, "SELECT h.x, SUM(h.y) FROM h GROUP BY h.x HAVING COUNT(*) > 1", {"h": [(None, 1), (None, 2)]})


TARGETS = [
    (
        "SELECT c.name, c.ck, o.k, o.price, SUM(l.q) FROM c, o, l WHERE o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300) "
        "AND c.ck = o.ck AND o.k = l.k GROUP BY c.name, c.ck, o.k, o.price ORDER BY o.price DESC LIMIT 10",
        "SELECT c.name, c.ck, o.k, o.price, SUM(l.q) FROM c, o, l WHERE c.ck = o.ck AND o.k = l.k "
        "GROUP BY c.name, c.ck, o.k, o.price HAVING SUM(l.q) > 300 ORDER BY o.price DESC LIMIT 10",
    ),
    (  # against the join with the pre-aggregated, pre-filtered table (TPC-H Q18's usual rewrite)
        "SELECT c.name, c.ck, o.k, o.price, SUM(l.q) FROM c, o, l WHERE o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300) "
        "AND c.ck = o.ck AND o.k = l.k GROUP BY c.name, c.ck, o.k, o.price ORDER BY o.price DESC LIMIT 10",
        "SELECT c.name, c.ck, o.k, o.price, g.total FROM c JOIN o ON c.ck = o.ck "
        "JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k HAVING SUM(l.q) > 300) AS g ON o.k = g.k ORDER BY o.price DESC LIMIT 10",
    ),
    (  # the groups joined instead of tested
        "SELECT c.name, c.ck, o.k, SUM(l.q) FROM c, o, l WHERE o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 300) "
        "AND c.ck = o.ck AND o.k = l.k GROUP BY c.name, c.ck, o.k",
        "SELECT c.name, c.ck, o.k, SUM(l.q) FROM c CROSS JOIN o CROSS JOIN l JOIN (SELECT l0.k AS k0 FROM l AS l0 GROUP BY l0.k HAVING SUM(l0.q) > 300) AS t ON o.k = t.k0 "
        "WHERE c.ck = o.ck AND o.k = l.k GROUP BY c.name, c.ck, o.k",
    ),
]

NEAR_MISSES = [
    (  # grouped by customer: IN filters orders, HAVING filters the customer's total
        "SELECT o.ck, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 3) GROUP BY o.ck",
        "SELECT o.ck, SUM(l.q) FROM o, l WHERE o.k = l.k GROUP BY o.ck HAVING SUM(l.q) > 3",
        {"o": [(5, 1, 9), (6, 1, 9)], "l": [(5, 1, 2), (6, 1, 2)]},
    ),
    (  # a filter on the lines: the outer total is smaller than the subquery's
        "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND l.n > 1 AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 3) GROUP BY o.k",
        "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND l.n > 1 GROUP BY o.k HAVING SUM(l.q) > 3",
        {"o": [(5, 1, 9)], "l": [(5, 1, 2), (5, 2, 2)]},
    ),
    (  # a table without a key repeats the lines
        "SELECT o.k, SUM(l.q) FROM o, l, h WHERE o.k = l.k AND h.x = o.k AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 3) GROUP BY o.k",
        "SELECT o.k, SUM(l.q) FROM o, l, h WHERE o.k = l.k AND h.x = o.k GROUP BY o.k HAVING SUM(l.q) > 3",
        {"o": [(5, 1, 9)], "l": [(5, 1, 2)], "h": [(5, 0), (5, 0)]},
    ),
    (  # another bound
        "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 3) GROUP BY o.k",
        "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k GROUP BY o.k HAVING SUM(l.q) > 4",
        {"o": [(5, 1, 9)], "l": [(5, 1, 4)]},
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
    "SELECT o.k, SUM(l.q) FROM o, l WHERE o.k = l.k AND o.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 3) GROUP BY o.k",
    "SELECT c.name, o.k, MAX(l.q), COUNT(*) FROM c, o, l WHERE o.k IN (SELECT g.kk FROM (SELECT k AS kk, SUM(q) AS a, COUNT(*) AS b FROM l GROUP BY k) AS g WHERE g.a > 2 OR g.b = 1) AND c.ck = o.ck AND o.k = l.k GROUP BY c.name, c.ck, o.k, o.price HAVING COUNT(*) < 3",
    "SELECT o.k, SUM(l.q) FROM c JOIN o ON c.ck = o.ck JOIN l ON l.k = o.k WHERE l.k IN (SELECT k FROM l GROUP BY k HAVING COUNT(q) >= 2) GROUP BY o.k",
    "SELECT x.k, MAX(x.q) FROM l AS x WHERE x.k IN (SELECT l.k FROM l GROUP BY l.k HAVING SUM(l.q) > 3 AND MIN(l.n) = 0) GROUP BY x.k",
    "SELECT c.name, o.k, SUM(l.q) FROM c CROSS JOIN o CROSS JOIN l JOIN (SELECT l0.k AS k0 FROM l AS l0 GROUP BY l0.k HAVING SUM(l0.q) > 2) AS t ON o.k = t.k0 WHERE c.ck = o.ck AND o.k = l.k AND o.k > 0 GROUP BY c.name, c.ck, o.k",
    "SELECT o.price, o.k, SUM(q) FROM o JOIN l ON o.k = l.k JOIN (SELECT k FROM l GROUP BY k HAVING AVG(q) > 1) AS f ON f.k = o.k GROUP BY o.k, o.price",
]


@pytest.mark.parametrize("sql", REPLAYED)
def test_the_rewrite_returns_the_same_rows(sql):
    after = filtered(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 3])
    for _ in range(40):
        rows = {
            "c": [(key, value()) for key in rng.sample(range(4), rng.randint(0, 3))],
            "o": [(key, value(), value()) for key in rng.sample(range(4), rng.randint(0, 4))],
            "l": [(k, n, value()) for k, n in rng.sample([(a, b) for a in range(4) for b in range(3)], rng.randint(0, 8))],
            "h": [(value(), value()) for _ in range(rng.randint(0, 4))],
        }
        first, second = rows_on_duckdb([sql, after], rows)
        assert first == second, rows
