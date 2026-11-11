"""A distinct-aggregate expansion written with projection layers and CASE arms collapses to one grouped select
(``kumo_sql_prover.expansion_layers``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.expansion_layers import collapse_layered_expansion

SCHEMA = {"t": ["k", "x", "y", "f"]}
TYPES = {"t": {"k": "INT", "x": "INT", "y": "INT", "f": "BOOLEAN"}}


def expansion(sets, outputs, keys=("k",), tag="GROUPING(t.k, t.x, t.y, t.f)", flags=(), inner_extra="", layer=None, where=""):
    """The plan shape: an aggregate over ``sets``, a projection layer with one flag per entry of ``flags``, the outer aggregate."""

    col = lambda key: f"t.{key}" if f"t.{key}" in sets else ("CAST(NULL AS BOOLEAN)" if key == "f" else "CAST(NULL AS BIGINT)")  # only keys are listed
    grouped = f"SELECT {col('k')} AS k, {col('x')} AS x, {col('y')} AS y, {col('f')} AS f, SUM(t.y) AS s, COUNT(t.y) AS n, MIN(t.y) AS lo, {tag} AS tag FROM t{where} GROUP BY {sets}{inner_extra}"
    flag_items = "".join(f", {text} AS g{n}" for n, text in enumerate(flags))
    middle = layer or f"SELECT g.k AS k, g.x AS x, g.y AS y, g.f AS f, g.s AS s, g.n AS n, g.lo AS lo{flag_items} FROM ({grouped}) AS g"
    head = "".join(f"p.{key} AS {key}, " for key in keys)
    by = f" GROUP BY {', '.join('p.' + key for key in keys)}" if keys else ""
    return f"SELECT {head}{outputs} FROM ({middle}) AS p{by}"


COUNT_AND_SUM = expansion(
    "GROUPING SETS ((t.k, t.x), (t.k))",
    "COUNT(CASE WHEN p.g0 THEN p.x END) AS a, MIN(CASE WHEN p.g1 THEN p.s END) AS b",
    flags=("(g.tag = 0)", "(g.tag = 1)"),
    tag="GROUPING(t.k, t.x)",
)
COUNT_AND_SUM_DIRECT = "SELECT t.k AS k, COUNT(DISTINCT t.x) AS a, SUM(t.y) AS b FROM t GROUP BY t.k"


def collapsed(sql):
    out = collapse_layered_expansion(sqlglot.parse_one(sql, read="mysql"))
    return out.sql(dialect="mysql") if out is not None else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="mysql", compare_names=False).proven


def results(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE t (k BIGINT, x BIGINT, y BIGINT, f BOOLEAN)")
    if rows:
        db.executemany("INSERT INTO t VALUES (?, ?, ?, ?)", rows)
    # the queries are MySQL text, as the Calcite fixtures are: sqlglot spells COUNT(DISTINCT a, b) for DuckDB
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="mysql", write="duckdb")[0] for q in queries))]


def databases(count, seed):
    rng = random.Random(seed)
    yield []
    yield [(None, None, None, None)]
    yield [(1, None, 2, True), (1, None, 3, False), (1, 4, None, None)]
    for _ in range(count):
        yield [(rng.choice([None, 1, 2]), rng.choice([None, 1, 2, 3]), rng.choice([None, 1, 2, 5]), rng.choice([None, True, False])) for _ in range(rng.randint(0, 7))]


READ = [
    (COUNT_AND_SUM, COUNT_AND_SUM_DIRECT),
    (  # no outer key: one row even over no rows, through the empty grouping set
        expansion("GROUPING SETS ((t.x), ())", "MIN(CASE WHEN p.g1 THEN p.s END) AS b, COUNT(CASE WHEN p.g0 THEN p.x END) AS a", keys=(), tag="GROUPING(t.x)", flags=("(g.tag = 0)", "(g.tag = 1)")),
        "SELECT SUM(t.y) AS b, COUNT(DISTINCT t.x) AS a FROM t",
    ),
    (  # two arguments, written as a row count over rows where both are present
        expansion("GROUPING SETS ((t.k, t.x, t.y), (t.k))", "COUNT(CASE WHEN p.g0 AND p.y IS NOT NULL AND p.x IS NOT NULL THEN 1 END) AS a, MAX(CASE WHEN p.g1 THEN p.lo END) AS b", tag="GROUPING(t.k, t.x, t.y)", flags=("(g.tag = 0)", "(g.tag = 3)")),
        "SELECT t.k AS k, COUNT(DISTINCT t.y, t.x) AS a, MIN(t.y) AS b FROM t GROUP BY t.k",
    ),
    (  # a filtered distinct count: the filter is a grouped boolean key
        expansion("GROUPING SETS ((t.k, t.x, t.f), (t.k))", "COUNT(CASE WHEN p.g0 THEN p.x END) AS a, MIN(CASE WHEN p.g1 THEN p.n END) AS b", tag="GROUPING(t.k, t.x, t.f)", flags=("((g.tag = 0) AND (g.f IS TRUE))", "(g.tag = 3)")),
        "SELECT t.k AS k, COUNT(DISTINCT CASE WHEN t.f THEN t.x END) AS a, COUNT(t.y) AS b FROM t GROUP BY t.k",
    ),
    (  # SUM, MIN and MAX of a distinct argument
        expansion("GROUPING SETS ((t.k, t.x), (t.k, t.y), (t.k))", "SUM(CASE WHEN p.g0 THEN p.x END) AS a, MIN(CASE WHEN p.g1 THEN p.y END) AS b, MAX(CASE WHEN p.g0 THEN p.x END) AS c", tag="GROUPING(t.k, t.x, t.y)", flags=("(g.tag = 1)", "(g.tag = 2)")),
        "SELECT t.k AS k, SUM(DISTINCT t.x) AS a, MIN(t.y) AS b, MAX(t.x) AS c FROM t GROUP BY t.k",
    ),
    (  # no grouping sets: the groups are rolled up and counted
        expansion("t.k, t.x, t.y", "SUM(p.s) AS a, MIN(p.lo) AS b, COUNT(CASE WHEN p.x IS NOT NULL AND p.y IS NOT NULL THEN 1 END) AS c", tag="0"),
        "SELECT t.k AS k, SUM(t.y) AS a, MIN(t.y) AS b, COUNT(DISTINCT t.x, t.y) AS c FROM t GROUP BY t.k",
    ),
    (  # FILTER spelling above the layers, and a source filter that stays where it is
        expansion("GROUPING SETS ((t.k, t.x), (t.k))", "COUNT(p.x) FILTER (WHERE p.g0) AS a, COALESCE(MIN(p.s) FILTER (WHERE p.g1), 0) AS b", tag="GROUPING(t.k, t.x)", flags=("(g.tag = 0)", "(g.tag = 1)"), where=" WHERE t.y > 1"),
        "SELECT t.k AS k, COUNT(DISTINCT t.x) AS a, COALESCE(SUM(t.y), 0) AS b FROM t WHERE t.y > 1 GROUP BY t.k",
    ),
]


@pytest.mark.parametrize("plan, direct", READ)
def test_an_expansion_is_the_aggregate_it_expands(plan, direct):
    out = collapsed(plan)
    assert out is not None and "GROUPING" not in out and "CASE WHEN p." not in out
    for rows in databases(50, seed=len(plan)):
        before, after, written = results([plan, out, direct], rows)
        assert before == after == written, rows
    assert proved(plan, direct)


NOT_READ = [
    # the set read is distinct on more than the argument: its rows are (x, y) pairs, not values of x
    expansion("GROUPING SETS ((t.k, t.x, t.y), (t.k))", "COUNT(CASE WHEN p.g0 THEN p.x END) AS a", tag="GROUPING(t.k, t.x, t.y)", flags=("(g.tag = 0)",)),
    expansion("GROUPING SETS ((t.k, t.x, t.y), (t.k))", "SUM(CASE WHEN p.g0 THEN p.x END) AS a", tag="GROUPING(t.k, t.x, t.y)", flags=("(g.tag = 0)",)),
    # a row count with no argument counts the NULL tuple too
    expansion("GROUPING SETS ((t.k, t.x), (t.k))", "COUNT(CASE WHEN p.g0 THEN 1 END) AS a", tag="GROUPING(t.k, t.x)", flags=("(g.tag = 0)",)),
    # one of two arguments may be NULL
    expansion("GROUPING SETS ((t.k, t.x, t.y), (t.k))", "COUNT(CASE WHEN p.g0 AND p.x IS NOT NULL THEN 1 END) AS a", tag="GROUPING(t.k, t.x, t.y)", flags=("(g.tag = 0)",)),
    # an outer key missing from a set: that set's rows carry a padded NULL key
    expansion("GROUPING SETS ((t.k, t.x), (t.x))", "COUNT(CASE WHEN p.g0 THEN p.x END) AS a", tag="GROUPING(t.k, t.x)", flags=("(g.tag = 0)",)),
    # the flag selects two sets, or none
    expansion("GROUPING SETS ((t.k, t.x), (t.k))", "MIN(CASE WHEN p.g0 THEN p.s END) AS a", tag="GROUPING(t.k, t.x)", flags=("(g.tag = g.tag)",)),
    expansion("GROUPING SETS ((t.k, t.x), (t.k))", "MIN(CASE WHEN p.g0 THEN p.s END) AS a", tag="GROUPING(t.k, t.x)", flags=("(g.tag = 5)",)),
    expansion("GROUPING SETS ((t.k, t.x), (t.k))", "MIN(p.s) AS a", tag="GROUPING(t.k, t.x)"),
    # the sum of a count is NULL over no groups; the smallest group sum is not a sum
    expansion("t.k, t.x", "SUM(p.n) AS a", tag="0"),
    expansion("t.k, t.x", "MIN(p.s) AS a", tag="0"),
    expansion("t.k, t.x", "COUNT(p.s) AS a", tag="0"),
    # an inner aggregate read under a row filter
    expansion("GROUPING SETS ((t.k, t.f), (t.k))", "MIN(CASE WHEN p.g0 THEN p.s END) AS a", tag="GROUPING(t.k, t.f)", flags=("((g.tag = 0) AND (g.f IS TRUE))",)),
    # ELSE arms, DISTINCT, other aggregates and clauses
    expansion("GROUPING SETS ((t.k, t.x), (t.k))", "COUNT(CASE WHEN p.g0 THEN p.x ELSE 0 END) AS a", tag="GROUPING(t.k, t.x)", flags=("(g.tag = 0)",)),
    expansion("GROUPING SETS ((t.k, t.x), (t.k))", "COUNT(DISTINCT CASE WHEN p.g0 THEN p.x END) AS a", tag="GROUPING(t.k, t.x)", flags=("(g.tag = 0)",)),
    expansion("GROUPING SETS ((t.k, t.x), (t.k))", "AVG(CASE WHEN p.g0 THEN p.x END) AS a", tag="GROUPING(t.k, t.x)", flags=("(g.tag = 0)",)),
    expansion("GROUPING SETS ((t.k, t.x), (t.k))", "MIN(CASE WHEN p.g1 THEN p.s END) AS a", tag="GROUPING(t.k, t.x)", flags=("(g.tag = 0)", "(g.tag = 1)"), inner_extra=" HAVING COUNT(*) > 1"),
    expansion("GROUPING SETS ((t.k, t.x), (t.k))", "MIN(CASE WHEN p.g1 THEN p.s END) AS a", tag="GROUPING(t.k, t.x)", flags=("(g.tag = 0)", "(g.tag = 1)")) + " HAVING COUNT(*) > 1",
    expansion("GROUPING SETS ((t.k, t.x), (t.k))", "MIN(CASE WHEN p.g1 THEN p.s END) AS a", tag="GROUPING(t.k, t.x)", flags=("(g.tag = 0)", "(g.tag = 1)")).replace(" AS p GROUP BY", " AS p WHERE p.x > 0 GROUP BY"),
    expansion("GROUPING SETS ((t.k, t.x), (t.k), (t.k))", "MIN(CASE WHEN p.g1 THEN p.s END) AS a", tag="GROUPING(t.k, t.x)", flags=("(g.tag = 0)", "(g.tag = 1)")),  # a repeated set
    expansion("ROLLUP (t.k, t.x)", "MIN(CASE WHEN p.g1 THEN p.s END) AS a", tag="GROUPING(t.k, t.x)", flags=("(g.tag = 0)", "(g.tag = 1)")),  # spelled out by expand_grouping_sets first
    # a layer that computes something: not a projection of the grouped rows
    expansion("GROUPING SETS ((t.k, t.x), (t.k))", "COUNT(CASE WHEN p.g0 THEN p.x END) AS a", layer="SELECT g.k AS k, g.x + 1 AS x, (g.tag = 0) AS g0 FROM (SELECT t.k AS k, t.x AS x, GROUPING(t.k, t.x) AS tag FROM t GROUP BY GROUPING SETS ((t.k, t.x), (t.k))) AS g"),
    expansion("GROUPING SETS ((t.k, t.x), (t.k))", "COUNT(CASE WHEN p.g0 THEN p.x END) AS a", layer="SELECT g.k AS k, g.x AS x, (g.tag = 0) AS g0 FROM (SELECT t.k AS k, t.x AS x, GROUPING(t.k, t.x) AS tag FROM t GROUP BY GROUPING SETS ((t.k, t.x), (t.k))) AS g WHERE g.x > 0"),
    "SELECT COUNT(DISTINCT t.x) AS a FROM t",
    "SELECT p.k, SUM(p.y) AS a FROM (SELECT t.k AS k, t.y AS y FROM t) AS p GROUP BY p.k",  # no grouped select below
]


@pytest.mark.parametrize("sql", NOT_READ)
def test_other_shapes_are_left_alone(sql):
    assert collapsed(sql) is None


@pytest.mark.parametrize("plan, direct, rows", [
    # rows distinct on (x, y) counted as values of x
    (NOT_READ[0], "SELECT t.k AS k, COUNT(DISTINCT t.x) AS a FROM t GROUP BY t.k", [(1, 1, 1, None), (1, 1, 2, None)]),
    # the NULL value is a group of its own
    (NOT_READ[2], "SELECT t.k AS k, COUNT(DISTINCT t.x) AS a FROM t GROUP BY t.k", [(1, None, 1, None), (1, 2, 1, None)]),
    # SUM of per-group counts over no groups at all
    (expansion("t.x", "SUM(p.n) AS a", keys=(), tag="0"), "SELECT COUNT(t.y) AS a FROM t", []),
    # the smallest group sum
    (NOT_READ[9], "SELECT t.k AS k, MIN(t.y) AS a FROM t GROUP BY t.k", [(1, 1, 1, None), (1, 1, 2, None)]),
])
def test_what_is_not_the_same_aggregate_is_not_proved(plan, direct, rows):
    first, second = results([plan, direct], rows)
    assert first != second  # the two really differ on this database
    assert not proved(plan, direct)


def test_a_distinct_count_of_other_arguments_is_not_proved():
    both = "SELECT t.k AS k, COUNT(DISTINCT t.x, t.y) AS a FROM t GROUP BY t.k"
    assert proved(both, both)
    assert not proved(both, both.replace("t.x, t.y", "t.x"))
    assert not proved(both, both.replace("t.x, t.y", "t.x, t.f"))
    rows = [(1, 1, 1, None), (1, 1, 2, None), (1, None, 3, None)]
    assert results([both], rows) == [["(1, 2)"]]  # two distinct pairs without a NULL
    assert results([both.replace("t.x, t.y", "t.x")], rows) == [["(1, 1)"]]


def test_the_rule_is_a_shape_rule():
    with shape_fallback.disabled():
        text = normalize(COUNT_AND_SUM, schema=SCHEMA, types=TYPES, dialect="mysql")
    assert "UNION ALL" in text  # the grouping sets, one grouped select each
    assert "UNION ALL" not in normalize(COUNT_AND_SUM, schema=SCHEMA, types=TYPES, dialect="mysql")
