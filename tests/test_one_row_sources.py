"""A joined one-row table of aggregates is a scalar subquery where it is read (``kumo_sql_prover.one_row_sources``),
and a filter with a closed scalar subquery on a grouped derived table is ``HAVING`` (``closed_scalar_filters``)."""

import random

import pytest
import sqlglot
from sqlglot import exp

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.closed_scalar_filters import fold_scalar_filter, lift_scalar_arithmetic
from kumo_sql_prover.one_row_sources import read_one_row_source

SCHEMA = {"ps": ["k", "sk", "q"], "s": ["sk", "n"], "h": ["x", "y"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}


def read(sql, dialect="postgres"):
    out = read_one_row_source(sqlglot.parse_one(sql, read=dialect), SCHEMA)
    return out.sql(dialect=dialect) if out is not None else None


def folded(sql, dialect="postgres"):
    out = fold_scalar_filter(sqlglot.parse_one(sql, read=dialect), SCHEMA)
    return out.sql(dialect=dialect) if out is not None else None


def lifted(sql, dialect="postgres"):
    tree = sqlglot.parse_one(sql, read=dialect)
    for select in list(tree.find_all(exp.Select)):
        out = lift_scalar_arithmetic(select)
        if out is not None:
            if select is tree:
                return out.sql(dialect=dialect)
            select.replace(out)
            return tree.sql(dialect=dialect)
    return None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


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


GROUPED = "(SELECT ps.k, SUM(ps.q) AS total FROM ps GROUP BY ps.k) AS v"


def test_the_one_row_table_becomes_scalar_subqueries():
    assert (
        read(f"SELECT v.k, v.total FROM {GROUPED}, (SELECT SUM(ps.q) * 2 AS bound FROM ps) AS t WHERE v.total > t.bound")
        == f"SELECT v.k, v.total FROM {GROUPED} WHERE v.total > (SELECT SUM(ps.q) * 2 FROM ps)"
    )
    # an inner join's ON becomes a filter; a bare name only the one-row table has is a read of it
    assert (
        read(f"SELECT v.k, bound FROM {GROUPED} JOIN (SELECT SUM(q) AS bound, COUNT(*) AS n FROM ps WHERE sk > 1) AS t ON v.total > t.bound AND t.n > 0")
        == f"SELECT v.k, (SELECT SUM(q) FROM ps WHERE sk > 1) FROM {GROUPED} WHERE v.total > (SELECT SUM(q) FROM ps WHERE sk > 1) AND (SELECT COUNT(*) FROM ps WHERE sk > 1) > 0"
    )
    # LEFT JOIN .. ON TRUE always finds the one row
    assert (
        read("SELECT h.x FROM h LEFT JOIN (SELECT MAX(q) AS m FROM ps) AS t ON TRUE WHERE h.y < t.m")
        == "SELECT h.x FROM h WHERE h.y < (SELECT MAX(q) FROM ps)"
    )
    # the one-row table first
    assert (
        read("SELECT h.x FROM (SELECT MAX(q) AS m FROM ps) AS t JOIN h ON h.y < t.m")
        == "SELECT h.x FROM h WHERE h.y < (SELECT MAX(q) FROM ps)"
    )
    # nobody reads it: the join just goes
    assert read("SELECT h.x FROM h, (SELECT MAX(q) AS m FROM ps) AS t") == "SELECT h.x FROM h"


@pytest.mark.parametrize("sql", [
    # not always one row
    "SELECT h.x FROM h, (SELECT MAX(q) AS m FROM ps GROUP BY k) AS t WHERE h.y < t.m",
    "SELECT h.x FROM h, (SELECT MAX(q) AS m FROM ps HAVING COUNT(*) > 1) AS t WHERE h.y < t.m",
    "SELECT h.x FROM h, (SELECT q AS m FROM ps) AS t WHERE h.y < t.m",
    "SELECT h.x FROM h, (SELECT MAX(q) AS m FROM ps LIMIT 0) AS t WHERE h.y < t.m",
    "SELECT h.x FROM h, (SELECT DISTINCT MAX(q) AS m FROM ps) AS t WHERE h.y < t.m",
    "SELECT h.x FROM h, (SELECT MAX(q) OVER () AS m FROM ps) AS t WHERE h.y < t.m",
    "SELECT h.x FROM h, (SELECT 1 AS m) AS t WHERE h.y < t.m",  # no aggregate: left to other rules
    "SELECT h.x FROM h, (SELECT COUNT(*) AS c, COUNT(q) AS ck FROM ps) AS t WHERE t.c = 0 OR t.ck >= t.c",  # Calcite's NOT IN guard: another rule's
    "SELECT h.x FROM h, (SELECT MAX(q) AS m, 1 AS one FROM ps) AS t WHERE h.y < t.m",
    "SELECT h.x FROM h, (SELECT MAX(q) + RANDOM() AS m FROM ps) AS t WHERE h.y < t.m AND h.x < t.m",
    # joined in a way that can pad or drop
    "SELECT h.x FROM h LEFT JOIN (SELECT MAX(q) AS m FROM ps) AS t ON h.y < t.m",
    "SELECT h.x FROM h RIGHT JOIN (SELECT MAX(q) AS m FROM ps) AS t ON TRUE WHERE h.y < t.m",
    "SELECT h.x FROM h FULL JOIN (SELECT MAX(q) AS m FROM ps) AS t ON h.y < t.m",
    "SELECT h.x FROM h LEFT JOIN s ON s.sk = h.x, (SELECT MAX(q) AS m FROM ps) AS t WHERE h.y < t.m",
    "SELECT h.x FROM h JOIN (SELECT MAX(q) AS y FROM ps) AS t USING (y)",
    # a read that cannot be followed
    "SELECT * FROM h, (SELECT MAX(q) AS m FROM ps) AS t",
    "SELECT h.x, t.* FROM h, (SELECT MAX(q) AS m FROM ps) AS t",
    "SELECT h.x FROM h, (SELECT MAX(q) AS y FROM ps) AS t WHERE y > 1",  # h has y too
    "SELECT h.x FROM h, (SELECT MAX(q) AS m FROM ps) AS t(z) WHERE h.y < t.z",
    "SELECT u.a FROM unknown_table AS u, (SELECT MAX(q) AS m FROM ps) AS t WHERE u.a < m",  # m could be u's
    "SELECT h.x, t.m FROM h, (SELECT MAX(q) AS m FROM ps) AS t GROUP BY h.x, t.m",
    "SELECT h.x, t.m AS m FROM h, (SELECT MAX(q) AS m FROM ps) AS t ORDER BY m",
    # alone in FROM
    "SELECT t.m FROM (SELECT MAX(q) AS m FROM ps) AS t",
])
def test_shapes_left_alone(sql):
    assert read(sql) is None


def test_the_filter_becomes_having():
    assert (
        folded(f"SELECT v.k, v.total FROM {GROUPED} WHERE v.total > (SELECT SUM(ps.q) * 2 FROM ps)")
        == "SELECT ps.k, SUM(ps.q) AS total FROM ps GROUP BY ps.k HAVING SUM(ps.q) > (SELECT SUM(ps.q) * 2 FROM ps)"
    )
    # bare names inside the subquery resolve to its own sources, derived tables included
    assert (
        folded(f"SELECT v.k FROM {GROUPED} WHERE v.total > (SELECT MAX(t) FROM (SELECT sk, SUM(q) AS t FROM ps GROUP BY sk) AS d JOIN s ON s.sk = d.sk)")
        == "SELECT ps.k FROM ps GROUP BY ps.k HAVING SUM(ps.q) > (SELECT MAX(t) FROM (SELECT sk, SUM(q) AS t FROM ps GROUP BY sk) AS d JOIN s ON s.sk = d.sk)"
    )


@pytest.mark.parametrize("sql", [
    f"SELECT v.k FROM {GROUPED} WHERE v.total > (SELECT SUM(ps.q) FROM ps WHERE ps.k = v.k)",  # correlated
    f"SELECT v.k FROM {GROUPED} WHERE v.total > (SELECT SUM(ps.q) FROM ps WHERE ps.sk = total)",  # a bare outer name
    f"SELECT v.k FROM {GROUPED} WHERE v.total > (SELECT SUM(u.a) FROM unknown_table AS u)",  # columns not known
    f"SELECT v.k FROM {GROUPED} WHERE v.total > (SELECT SUM(z) FROM ps)",  # z is nobody's here: it reaches outside
    f"SELECT v.k FROM {GROUPED} WHERE v.total IN (SELECT SUM(ps.q) FROM ps)",
    f"SELECT v.k FROM {GROUPED} WHERE EXISTS (SELECT SUM(ps.q) FROM ps)",
    f"SELECT v.k FROM {GROUPED} WHERE v.total > (SELECT SUM(ps.q) FROM ps WHERE ps.sk IN (SELECT sk FROM s))",  # nested query
    f"SELECT v.k FROM {GROUPED} WHERE v.total > (SELECT SUM(d.q) FROM (SELECT * FROM ps) AS d)",
    f"SELECT v.k FROM {GROUPED} WHERE v.total > (SELECT SUM(q) FROM ps, LATERAL (SELECT 1) AS z)",
    f"SELECT v.k FROM {GROUPED} WHERE v.total > 5",  # no subquery: the other rule's job
    f"SELECT v.k FROM {GROUPED} JOIN h ON h.x = v.k WHERE v.total > (SELECT SUM(ps.q) FROM ps)",
    f"SELECT v.k FROM {GROUPED} WHERE v.total > :p AND v.total < (SELECT SUM(ps.q) FROM ps)",
    "SELECT v.k FROM (SELECT ps.k, ps.q AS total FROM ps) AS v WHERE v.total > (SELECT SUM(ps.q) FROM ps)",  # not grouped
])
def test_filters_left_alone(sql):
    assert folded(sql) is None


def test_arithmetic_leaves_the_scalar_subquery():
    assert lifted("SELECT h.x FROM h WHERE h.y > (SELECT SUM(q) * 2 FROM ps)") == "SELECT h.x FROM h WHERE h.y > ((SELECT SUM(q) FROM ps) * 2)"
    assert (
        lifted("SELECT h.x, (SELECT (1 - COUNT(*)) / 4 AS c FROM ps WHERE sk = 1) FROM h")
        == "SELECT h.x, ((1 - (SELECT COUNT(*) FROM ps WHERE sk = 1)) / 4) FROM h"
    )
    for sql in [
        "SELECT h.x FROM h WHERE h.y > (SELECT SUM(q) FROM ps)",  # nothing to move
        "SELECT h.x FROM h WHERE h.y > (SELECT SUM(q) * COUNT(*) FROM ps)",  # two aggregates
        "SELECT h.x FROM h WHERE h.y > (SELECT SUM(q) * 2 FROM ps GROUP BY k)",  # not one row
        "SELECT h.x FROM h WHERE h.y > (SELECT SUM(q) * 2 FROM ps HAVING COUNT(*) > 1)",
        "SELECT h.x FROM h WHERE h.y > (SELECT SUM(q) * 2 FROM ps LIMIT 1)",
        "SELECT h.x FROM h WHERE h.y IN (SELECT SUM(q) * 2 FROM ps)",
        "SELECT h.x FROM h WHERE h.y > (SELECT SUM(q) / 0 FROM ps)",
        "SELECT h.x FROM h WHERE h.y > (SELECT COALESCE(SUM(q), 0) FROM ps)",
        "SELECT h.x FROM h WHERE h.y > (SELECT q * 2 FROM ps)",
        "SELECT h.x FROM h, (SELECT SUM(q) * 2 AS m FROM ps) AS t",
    ]:
        assert lifted(sql) is None, sql


TARGETS = [
    (  # TPC-H Q11's shape: the threshold in HAVING against a cross-joined one-row total
        "SELECT ps.k, SUM(ps.q) AS total FROM ps, s WHERE ps.sk = s.sk AND s.n = 7 GROUP BY ps.k "
        "HAVING SUM(ps.q) > (SELECT SUM(ps.q) * 2 FROM ps, s WHERE ps.sk = s.sk AND s.n = 7)",
        "SELECT v.k, v.total FROM (SELECT ps.k, SUM(ps.q) AS total FROM ps JOIN s ON ps.sk = s.sk WHERE s.n = 7 GROUP BY ps.k) AS v, "
        "(SELECT SUM(ps.q) * 2 AS bound FROM ps JOIN s ON ps.sk = s.sk WHERE s.n = 7) AS t WHERE v.total > t.bound",
    ),
    (  # the factor outside, an inner join on the comparison
        "SELECT ps.k, SUM(ps.q) AS total FROM ps GROUP BY ps.k HAVING SUM(ps.q) > (SELECT SUM(ps.q) * 2 FROM ps)",
        "SELECT v.k, v.total FROM (SELECT ps.k, SUM(ps.q) AS total FROM ps GROUP BY ps.k) AS v "
        "JOIN (SELECT SUM(ps.q) AS bound FROM ps) AS t ON v.total > t.bound * 2",
    ),
    (  # LEFT JOIN ON TRUE
        "SELECT h.x FROM h WHERE h.y < (SELECT SUM(q) FROM ps)",
        "SELECT h.x FROM h LEFT JOIN (SELECT SUM(q) AS m FROM ps) AS t ON TRUE WHERE h.y < t.m",
    ),
]

NEAR_MISSES = [
    (  # another filter in the total
        "SELECT ps.k, SUM(ps.q) AS total FROM ps, s WHERE ps.sk = s.sk AND s.n = 7 GROUP BY ps.k "
        "HAVING SUM(ps.q) > (SELECT SUM(ps.q) * 2 FROM ps, s WHERE ps.sk = s.sk AND s.n = 7)",
        "SELECT v.k, v.total FROM (SELECT ps.k, SUM(ps.q) AS total FROM ps JOIN s ON ps.sk = s.sk WHERE s.n = 7 GROUP BY ps.k) AS v, "
        "(SELECT SUM(ps.q) * 2 AS bound FROM ps JOIN s ON ps.sk = s.sk WHERE s.n = 8) AS t WHERE v.total > t.bound",
        {"ps": [(1, 1, 5), (2, 2, -9)], "s": [(1, 7), (2, 8)]},
    ),
    (  # another factor
        "SELECT ps.k, SUM(ps.q) AS total FROM ps GROUP BY ps.k HAVING SUM(ps.q) > (SELECT SUM(ps.q) * 2 FROM ps)",
        "SELECT v.k, v.total FROM (SELECT ps.k, SUM(ps.q) AS total FROM ps GROUP BY ps.k) AS v, (SELECT SUM(ps.q) * 3 AS bound FROM ps) AS t WHERE v.total > t.bound",
        {"ps": [(1, 1, 3), (2, 1, -2)]},
    ),
    (  # a grouped table is not one row: each group's own total, not the grand total
        "SELECT h.x FROM h WHERE h.y < (SELECT MAX(q) FROM ps)",
        "SELECT h.x FROM h, (SELECT MAX(q) AS m FROM ps GROUP BY k) AS t WHERE h.y < t.m",
        {"h": [(1, 0)], "ps": [(1, 1, 5), (2, 1, 6)]},
    ),
    (  # LEFT JOIN on the comparison keeps the rows that fail it
        "SELECT h.x FROM h WHERE h.y < (SELECT MAX(q) FROM ps)",
        "SELECT h.x FROM h LEFT JOIN (SELECT MAX(q) AS m FROM ps) AS t ON h.y < t.m",
        {"h": [(1, 9)], "ps": [(1, 1, 5)]},
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
    (read, f"SELECT v.k, v.total FROM {GROUPED}, (SELECT SUM(ps.q) * 2 AS bound FROM ps) AS t WHERE v.total > t.bound"),
    (read, f"SELECT v.k, bound FROM {GROUPED} JOIN (SELECT SUM(q) AS bound, COUNT(*) AS n FROM ps WHERE sk > 1) AS t ON v.total > t.bound OR t.n = 0"),
    (read, "SELECT h.x, t.m + 1 FROM h LEFT JOIN (SELECT MAX(q) AS m, MIN(q) AS w FROM ps) AS t ON TRUE WHERE h.y < t.m OR t.w IS NULL"),
    (read, "SELECT h.x FROM (SELECT MAX(q) AS m FROM ps) AS t JOIN h ON h.y < t.m JOIN s ON s.sk = h.x OR s.n > t.m"),
    (folded, f"SELECT v.k, v.total FROM {GROUPED} WHERE v.total > (SELECT SUM(ps.q) * 2 FROM ps) OR v.k = (SELECT MIN(sk) FROM s)"),
    (lifted, "SELECT h.x, (SELECT (1 - COUNT(*)) * 4 AS c FROM ps WHERE sk = 1) FROM h WHERE h.y > (SELECT -SUM(q) + 2 FROM ps)"),
]


@pytest.mark.parametrize("rule, sql", REPLAYED)
def test_the_rewrite_returns_the_same_rows(rule, sql):
    after = rule(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 3])
    for _ in range(40):
        rows = {name: [tuple(value() for _ in columns) for _ in range(rng.randint(0, 4))] for name, columns in SCHEMA.items()}
        first, second = rows_on_duckdb([sql, after], rows)
        assert first == second, rows
