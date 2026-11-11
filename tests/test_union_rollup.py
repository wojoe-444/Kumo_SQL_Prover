"""A global aggregate over a UNION ALL of grouped selects reads each branch's totals (``kumo_sql_prover.union_rollup``),
and a renamed read of a derived UNION ALL keeps clear of the select's aliases (``kumo_sql_prover.union_source_names``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.union_rollup import roll_up_grouped_union

SCHEMA = {"t": ["a", "b", "c"], "u": ["a", "b", "c"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}

GROUPED = "SELECT t.a AS k, COUNT(*) AS n, COUNT(t.c) AS m, SUM(t.b) AS s, MIN(t.b) AS lo, MAX(t.b) AS hi FROM t GROUP BY t.a, t.c"
OTHER = "SELECT u.a AS k, COUNT(*) AS n, COUNT(u.c) AS m, SUM(u.b) AS s, MIN(u.b) AS lo, MAX(u.b) AS hi FROM u WHERE u.b > 1 GROUP BY u.a, u.c"
UNION = f"({GROUPED} UNION ALL {OTHER}) AS g"
TOTALS = (
    "SELECT SUM(x.n) AS n, SUM(x.m) AS m, SUM(x.s) AS s, MIN(x.lo) AS lo, MAX(x.hi) AS hi, MIN(x.k) AS k FROM ("
    "SELECT COUNT(*) AS n, COUNT(t.c) AS m, SUM(t.b) AS s, MIN(t.b) AS lo, MAX(t.b) AS hi, MIN(t.a) AS k FROM t UNION ALL "
    "SELECT COUNT(*) AS n, COUNT(u.c) AS m, SUM(u.b) AS s, MIN(u.b) AS lo, MAX(u.b) AS hi, MIN(u.a) AS k FROM u WHERE u.b > 1) AS x"
)
ROLLED = f"SELECT COALESCE(SUM(g.n), 0) AS n, COALESCE(SUM(g.m), 0) AS m, SUM(g.s) AS s, MIN(g.lo) AS lo, MAX(g.hi) AS hi, MIN(g.k) AS k FROM {UNION}"


def rolled(sql):
    out = roll_up_grouped_union(sqlglot.parse_one(sql, read="postgres"))
    return out.sql(dialect="postgres") if out is not None else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


def results(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name in SCHEMA:
        db.execute(f"CREATE TABLE {name} (a BIGINT, b BIGINT, c BIGINT)")
        insert_rows(db, name, rows.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def databases(count, seed):
    rng = random.Random(seed)
    value = lambda: rng.choice([None, 0, 1, 2, 3])
    yield {}
    yield {"t": [(1, 2, None)]}
    yield {"u": [(None, 5, 1), (None, None, 1)]}
    for _ in range(count):
        yield {name: [(value(), value(), value()) for _ in range(rng.randint(0, 6))] for name in SCHEMA}


def test_the_totals_of_grouped_branches_are_read():
    out = rolled(ROLLED)
    assert out is not None and "GROUP BY" not in out and "COALESCE" not in out
    assert out.count("COUNT(*)") == 2 and "MIN(t.a)" in out and "MIN(u.a)" in out and "WHERE u.b > 1" in out
    assert proved(ROLLED, TOTALS)


def test_the_rewrite_returns_the_same_rows_on_duckdb():
    out = rolled(ROLLED)
    for rows in databases(60, seed=20261111):
        before, after, totals = results([ROLLED, out, TOTALS], rows)
        assert before == after == totals, rows


@pytest.mark.parametrize("sql", [
    f"SELECT SUM(g.n) AS n FROM {UNION}",  # NULL over no groups, where the totals add up to 0
    f"SELECT COALESCE(SUM(g.s), 0) AS s FROM {UNION}",  # a sum is not a count: its total may be NULL
    f"SELECT COALESCE(SUM(g.n), 1) AS n FROM {UNION}",
    f"SELECT MIN(g.n) AS n FROM {UNION}",  # the smallest group, not a total
    f"SELECT SUM(g.lo) AS s FROM {UNION}",
    f"SELECT MAX(g.lo) AS s FROM {UNION}",
    f"SELECT SUM(DISTINCT g.s) AS s FROM {UNION}",
    f"SELECT COUNT(g.s) AS s FROM {UNION}",
    f"SELECT SUM(g.s) AS s FROM {UNION} WHERE g.k = 1",
    f"SELECT SUM(g.s) AS s FROM {UNION} GROUP BY g.k",
    f"SELECT SUM(g.s) AS s FROM {UNION} HAVING COUNT(*) > 1",
    f"SELECT SUM(g.s) AS s FROM {UNION} LIMIT 1",
    f"SELECT SUM(g.s) + 1 AS s FROM {UNION}",
    f"SELECT g.k FROM {UNION}",  # no aggregate: a row per group
    f"SELECT 1 AS one FROM {UNION}",
    f"SELECT SUM(g.s) AS s FROM ({GROUPED}) AS g",  # one branch: algebraic_equivalence._roll_up_aggregate
    f"SELECT SUM(g.s) AS s FROM ({GROUPED} UNION {OTHER}) AS g",  # UNION removes equal groups
    f"SELECT SUM(g.s) AS s FROM ({GROUPED} HAVING COUNT(*) > 1 UNION ALL {OTHER}) AS g",
    f"SELECT SUM(g.s) AS s FROM ({GROUPED.replace('GROUP BY t.a, t.c', 'GROUP BY ROLLUP (t.a, t.c)')} UNION ALL {OTHER}) AS g",  # a row is in several groups
    f"SELECT SUM(g.s) AS s FROM ({GROUPED.replace('SELECT', 'SELECT DISTINCT')} UNION ALL {OTHER}) AS g",
    f"SELECT SUM(g.s) AS s FROM ({GROUPED} UNION ALL SELECT u.a, 1, 1, u.b, u.b, u.b FROM u) AS g",  # an ungrouped branch
    f"SELECT SUM(g.s) AS s FROM ({GROUPED.replace('SUM(t.b)', 'SUM(DISTINCT t.b)')} UNION ALL {OTHER}) AS g",
    f"SELECT COALESCE(SUM(g.n), 0) AS n FROM ({GROUPED.replace('COUNT(*)', 'COUNT(DISTINCT t.b)')} UNION ALL {OTHER}) AS g",
    f"SELECT MIN(g.k) AS k FROM ({GROUPED.replace('t.a AS k', 't.b AS k')} UNION ALL {OTHER}) AS g",  # not a grouping key
])
def test_other_shapes_are_left_alone(sql):
    assert rolled(sql) is None


def test_a_bare_sum_of_counts_is_not_the_count():
    summed = f"SELECT SUM(g.n) AS n FROM {UNION}"
    counted = "SELECT SUM(x.n) AS n FROM (SELECT COUNT(*) AS n FROM t UNION ALL SELECT COUNT(*) AS n FROM u WHERE u.b > 1) AS x"
    assert not proved(summed, counted)
    assert results([summed, counted], {})[0] != results([summed, counted], {})[1]  # NULL and 0 over empty tables


def test_the_smallest_group_is_not_the_total():
    assert not proved(f"SELECT MIN(g.s) AS s FROM {UNION}", "SELECT MIN(x.s) AS s FROM (SELECT SUM(t.b) AS s FROM t UNION ALL SELECT SUM(u.b) AS s FROM u WHERE u.b > 1) AS x")


def test_the_rule_is_a_shape_rule():
    with shape_fallback.disabled():
        text = normalize(ROLLED, schema=SCHEMA, types=TYPES, dialect="postgres")
    assert "COALESCE" in text and "GROUP BY" in text


# ---- union_source_names: positional aliases above a derived UNION ALL ----

PLAN_ABOVE = (
    "SELECT u.c0 AS c0, u.c1 AS c1, SUM(u.c2) AS c2 FROM (SELECT t.a AS c0, t.c AS c1, t.b AS c2 FROM t UNION ALL "
    "SELECT u.a AS c0, u.c AS c1, u.b AS c2 FROM u) AS u GROUP BY u.c0, u.c1"
)
PLAN_BELOW = (
    "SELECT w.c0 AS c0, w.c1 AS c1, SUM(w.c2) AS c2 FROM (SELECT t.a AS c0, t.c AS c1, SUM(t.b) AS c2 FROM t GROUP BY t.a, t.c UNION ALL "
    "SELECT u.a AS c0, u.c AS c1, SUM(u.b) AS c2 FROM u GROUP BY u.a, u.c) AS w GROUP BY w.c0, w.c1"
)


def test_positional_aliases_above_a_union_do_not_hide_its_columns():
    assert proved(PLAN_ABOVE, PLAN_BELOW)
    for rows in databases(40, seed=7):
        above, below = results([PLAN_ABOVE, PLAN_BELOW], rows)
        assert above == below, rows


def test_a_different_grouping_under_the_same_names_is_not_proved():
    other_key = PLAN_BELOW.replace("GROUP BY w.c0, w.c1", "GROUP BY w.c1, w.c0").replace("SELECT w.c0 AS c0, w.c1 AS c1", "SELECT w.c1 AS c0, w.c0 AS c1")
    assert not proved(PLAN_ABOVE, other_key)
    fewer = "SELECT w.c0 AS c0, MIN(w.c1) AS c1, SUM(w.c2) AS c2" + PLAN_BELOW[PLAN_BELOW.index(" FROM"):].replace("GROUP BY w.c0, w.c1", "GROUP BY w.c0")
    assert not proved(PLAN_ABOVE, fewer)


def test_a_read_that_was_ambiguous_as_written_stays_unqualified():
    from kumo_sql_prover import union_source_names
    from sqlglot import exp

    column = exp.column("c1")
    union_source_names.qualify(column, "c0", "u", {"c0", "c1"})  # the old name was a select alias too
    assert column.sql() == "c1"
    union_source_names.qualify(column, "k", "u", {"c1"})
    assert column.sql() == "u.c1"
    union_source_names.qualify(exp.column("c2"), "k", "u", {"c1"})  # no clash: left as it is
