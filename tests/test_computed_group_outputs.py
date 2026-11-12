"""Inside a membership test, arithmetic on an aggregate leaves the grouped derived table (``kumo_sql_prover.computed_group_outputs``)."""

import random

import pytest
import sqlglot
from sqlglot import exp

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.computed_group_outputs import lift_computed_outputs

SCHEMA = {"s": ["k", "name"], "ps": ["sk", "pk", "n"], "l": ["pk", "sk", "q"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}


def rewritten(sql, dialect="postgres"):
    """``sql`` with the rule applied until it no longer fires, or ``None`` if it never does."""

    tree = sqlglot.parse_one(sql, read=dialect)
    fired = False
    for _ in range(8):
        for select in list(tree.find_all(exp.Select)):
            out = lift_computed_outputs(select)
            if out is not None:
                if select is tree:
                    tree = out
                else:
                    select.replace(out)
                fired = True
                break
        else:
            break
    return tree.sql(dialect=dialect) if fired else None


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


def test_the_arithmetic_moves_to_the_read():
    assert (
        rewritten("SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, 2 * SUM(l.q) AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > d.v)")
        == "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, SUM(l.q) AS kql0 FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > (2 * d.kql0))"
    )
    # two aggregates, one of them twice; a join with ON; a read in the ON
    assert (
        rewritten("SELECT s.name FROM s WHERE EXISTS (SELECT 1 FROM ps JOIN (SELECT l.pk, SUM(l.q) - (COUNT(*) + SUM(l.q)) / 2 AS v FROM l GROUP BY l.pk) AS d ON ps.pk = d.pk AND ps.n > d.v WHERE ps.sk = s.k)")
        == "SELECT s.name FROM s WHERE EXISTS(SELECT 1 FROM ps JOIN (SELECT l.pk, SUM(l.q) AS kql0, COUNT(*) AS kql1 FROM l GROUP BY l.pk) AS d ON ps.pk = d.pk AND ps.n > (d.kql0 - (d.kql1 + d.kql0) / 2) WHERE ps.sk = s.k)"
    )
    # an output nobody reads is still replaced by its aggregate
    assert (
        rewritten("SELECT s.name FROM s WHERE s.k IN (SELECT d.pk FROM (SELECT l.pk, -SUM(l.q) AS v FROM l GROUP BY l.pk) AS d)")
        == "SELECT s.name FROM s WHERE s.k IN (SELECT d.pk FROM (SELECT l.pk, SUM(l.q) AS kql0 FROM l GROUP BY l.pk) AS d)"
    )


@pytest.mark.parametrize("sql", [
    # not the query of a membership test
    "SELECT ps.sk FROM ps, (SELECT l.pk, 2 * SUM(l.q) AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > d.v",
    # an outer join pads d.v with NULL, and COUNT(*) + 1 of nothing is not NULL
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps LEFT JOIN (SELECT l.pk, COUNT(*) + 1 AS v FROM l GROUP BY l.pk) AS d ON ps.pk = d.pk WHERE d.v IS NULL)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM (SELECT l.pk, COUNT(*) + 1 AS v FROM l GROUP BY l.pk) AS d RIGHT JOIN ps ON ps.pk = d.pk WHERE d.v IS NULL)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps FULL JOIN (SELECT l.pk, COUNT(*) + 1 AS v FROM l GROUP BY l.pk) AS d ON ps.pk = d.pk)",
    # a bare aggregate or a grouping column: nothing to move
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, SUM(l.q) AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, l.pk + 1 AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
    # more than arithmetic on literals and aggregates
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, COALESCE(SUM(l.q), 0) AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, l.pk * SUM(l.q) AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, SUM(l.q) / 0 AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, 1 / SUM(l.q) AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, SUM(l.q) / COUNT(*) AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, 2 * SUM(l.q) OVER () AS v FROM l) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, 2 * SUM(l.q) FILTER (WHERE l.sk > 0) AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, 2 * SUM((SELECT MAX(n) FROM ps)) AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
    # the derived table is not a plain grouped select
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT 2 * SUM(l.q) AS v FROM l) AS d WHERE ps.n > d.v)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, 2 * SUM(l.q) AS v FROM l GROUP BY l.pk HAVING COUNT(*) > 1) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, 2 * SUM(l.q) AS v FROM l GROUP BY l.pk ORDER BY 2 LIMIT 3) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT DISTINCT l.pk, 2 * SUM(l.q) AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, 2 * SUM(l.q) AS v FROM l GROUP BY ROLLUP (l.pk)) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, 2 * SUM(l.q) AS v FROM l GROUP BY l.pk) AS d(a, v) WHERE ps.pk = d.a AND ps.n > d.v)",
    # a read that is not ``d.v``, or one that cannot be followed
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, 2 * SUM(l.q) AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > v)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT * FROM (SELECT 2 * SUM(l.q) AS v FROM l GROUP BY l.pk) AS d)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT d.* FROM (SELECT 2 * SUM(l.q) AS v FROM l GROUP BY l.pk) AS d)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, 2 * SUM(l.q) AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND EXISTS (SELECT 1 FROM l AS d WHERE d.q = ps.n))",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps JOIN (SELECT l.pk, 2 * SUM(l.q) AS v FROM l GROUP BY l.pk) AS d USING (pk))",
])
def test_shapes_left_alone(sql):
    assert rewritten(sql) is None


TARGETS = [
    (
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, 2 * SUM(l.q) AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps JOIN (SELECT l.pk, SUM(l.q) AS t FROM l GROUP BY l.pk) AS e ON ps.pk = e.pk WHERE ps.n > 2 * e.t)",
    ),
    (
        "SELECT s.name FROM s WHERE EXISTS (SELECT 1 FROM ps JOIN (SELECT l.pk, l.sk, 2 * SUM(l.q) AS v FROM l GROUP BY l.pk, l.sk) AS d ON ps.pk = d.pk AND ps.sk = d.sk WHERE ps.sk = s.k AND ps.n > d.v)",
        "SELECT s.name FROM s WHERE EXISTS (SELECT 1 FROM ps JOIN (SELECT l.pk, l.sk, SUM(l.q) AS t FROM l GROUP BY l.pk, l.sk) AS e ON ps.pk = e.pk AND ps.sk = e.sk WHERE ps.sk = s.k AND ps.n > 2 * e.t)",
    ),
    (  # what decorrelating the scalar subquery leaves, against the join written by hand
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.n > (SELECT 2 * SUM(l.q) FROM l WHERE l.pk = ps.pk AND l.sk = ps.sk))",
        "SELECT s.name FROM s WHERE EXISTS (SELECT 1 FROM ps JOIN (SELECT l.pk, l.sk, SUM(l.q) AS t FROM l GROUP BY l.pk, l.sk) AS e ON ps.pk = e.pk AND ps.sk = e.sk WHERE ps.sk = s.k AND ps.n > 2 * e.t)",
    ),
]

NEAR_MISSES = [
    (  # another factor
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, 2 * SUM(l.q) AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps JOIN (SELECT l.pk, SUM(l.q) AS t FROM l GROUP BY l.pk) AS e ON ps.pk = e.pk WHERE ps.n > 3 * e.t)",
        {"s": [(1, 10)], "ps": [(1, 5, 5)], "l": [(5, 1, 2)]},
    ),
    (  # another aggregate
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, 2 * SUM(l.q) AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps JOIN (SELECT l.pk, MAX(l.q) AS t FROM l GROUP BY l.pk) AS e ON ps.pk = e.pk WHERE ps.n > 2 * e.t)",
        {"s": [(1, 10)], "ps": [(1, 5, 5)], "l": [(5, 1, 2), (5, 2, 2)]},
    ),
    (  # under a LEFT JOIN the padded row reads NULL, not 1
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps LEFT JOIN (SELECT l.pk, COUNT(*) + 1 AS v FROM l GROUP BY l.pk) AS d ON ps.pk = d.pk WHERE d.v IS NULL)",
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps LEFT JOIN (SELECT l.pk, COUNT(*) AS t FROM l GROUP BY l.pk) AS e ON ps.pk = e.pk WHERE COALESCE(e.t, 0) + 1 IS NULL)",
        {"s": [(1, 10)], "ps": [(1, 5, 5)], "l": []},
    ),
    (  # the grouping differs
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, 2 * SUM(l.q) AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps JOIN (SELECT l.pk, SUM(l.q) AS t FROM l GROUP BY l.pk, l.sk) AS e ON ps.pk = e.pk WHERE ps.n > 2 * e.t)",
        {"s": [(1, 10)], "ps": [(1, 5, 5)], "l": [(5, 1, 2), (5, 2, 2)]},
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
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, 2 * SUM(l.q) AS v FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.n > d.v)",
    "SELECT s.name FROM s WHERE s.k NOT IN (SELECT ps.sk FROM ps JOIN (SELECT l.pk, SUM(l.q) - (COUNT(*) + SUM(l.q)) / 2 AS v FROM l GROUP BY l.pk) AS d ON ps.pk = d.pk AND ps.n > d.v)",
    "SELECT s.name FROM s WHERE EXISTS (SELECT 1 FROM ps, (SELECT l.pk, -COUNT(l.q) * 3 AS v, MAX(l.q) + 1 AS w FROM l GROUP BY l.pk) AS d WHERE ps.pk = d.pk AND ps.sk = s.k AND ps.n + d.v < d.w - 4)",
    "SELECT s.name, s.k IN (SELECT d.v FROM (SELECT l.pk, MIN(l.q) + 1 AS v FROM l GROUP BY l.pk) AS d) FROM s",
]


@pytest.mark.parametrize("sql", REPLAYED)
def test_the_rewrite_returns_the_same_rows(sql):
    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 3])
    for _ in range(40):
        rows = {name: [tuple(value() for _ in columns) for _ in range(rng.randint(0, 4))] for name, columns in SCHEMA.items()}
        first, second = rows_on_duckdb([sql, after], rows)
        assert first == second, rows
