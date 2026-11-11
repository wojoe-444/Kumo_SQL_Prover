"""The levels around window functions: a filter on partition keys moves below them (``window_filters``), a windowed
select reads its derived table's sources (``window_sources``), unread outputs go (``window_pruning``), and
``window_levels`` settles all three before the windows are isolated."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.window_filters import push_filter_below_windows
from kumo_sql_prover.window_levels import settle_window_levels
from kumo_sql_prover.window_pruning import prune_windowed_derived
from kumo_sql_prover.window_sources import inline_windowed_source

SCHEMA = {"t": ["a", "b", "c", "k"], "u": ["a", "b", "c", "k"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}

COUNTED = "SELECT t.k AS n, t.a AS d, t.b AS m, COUNT(*) OVER (PARTITION BY t.a) AS c FROM t"


def applied(rule, sql):
    tree = sqlglot.parse_one(sql, read="postgres")
    out = rule(tree, SCHEMA) if rule in (prune_windowed_derived, settle_window_levels) else rule(tree)
    return out.sql(dialect="postgres") if out is not None else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


def results(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name in SCHEMA:
        db.execute(f"CREATE TABLE {name} (a BIGINT, b BIGINT, c BIGINT, k BIGINT)")
        insert_rows(db, name, rows.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def databases(count, seed):
    rng = random.Random(seed)
    value = lambda: rng.choice([None, 0, 1, 2])
    yield {}
    yield {"t": [(1, 2, None, 1)]}
    for _ in range(count):
        yield {name: [(value(), value(), value(), value()) for _ in range(rng.randint(0, 7))] for name in SCHEMA}


def differs(left, right, seed=5):
    return any(a != b for rows in databases(40, seed) for a, b in [results([left, right], rows)])


# ------------------------------------------------------------------ filters below windows


def test_a_filter_on_the_partition_key_moves_below_the_window():
    above = f"SELECT d.n, d.c FROM ({COUNTED}) AS d WHERE d.d = 0"
    out = applied(push_filter_below_windows, above)
    assert out is not None and out.endswith("AS d") and "WHERE t.a = 0" in out
    below = "SELECT t.k, COUNT(*) OVER (PARTITION BY t.a) FROM t WHERE t.a = 0"
    assert proved(above, below)
    assert proved(f"SELECT d.n, d.c FROM ({COUNTED}) AS d WHERE d.d > 0", "SELECT t.k, COUNT(*) OVER (PARTITION BY t.a) FROM t WHERE t.a > 0")


def test_only_the_conjuncts_on_partition_keys_move():
    above = f"SELECT d.n, d.c FROM ({COUNTED}) AS d WHERE d.d > 0 AND d.m = 1 AND d.c > 1"
    out = applied(push_filter_below_windows, above)
    assert out is not None and "WHERE t.a > 0" in out and "d.m = 1" in out and "d.c > 1" in out
    assert proved(above, "SELECT d.n, d.c FROM (SELECT t.k AS n, t.b AS m, COUNT(*) OVER (PARTITION BY t.a) AS c FROM t WHERE t.a > 0) AS d WHERE d.m = 1 AND d.c > 1")


def test_every_window_must_partition_by_the_filtered_column():
    two = "SELECT t.k AS n, t.a AS d, t.b AS m, COUNT(*) OVER (PARTITION BY t.a, t.b) AS c, SUM(t.c) OVER (PARTITION BY t.b, t.a ORDER BY t.k) AS s FROM t"
    assert applied(push_filter_below_windows, f"SELECT d.n FROM ({two}) AS d WHERE d.d = 0 AND d.m > 1") is not None
    one = "SELECT t.k AS n, t.a AS d, COUNT(*) OVER (PARTITION BY t.a) AS c, SUM(t.c) OVER (PARTITION BY t.b) AS s FROM t"
    assert applied(push_filter_below_windows, f"SELECT d.n, d.s FROM ({one}) AS d WHERE d.d = 0") is None


FILTER_NEAR_MISSES = [
    f"SELECT d.n, d.c FROM ({COUNTED}) AS d WHERE d.m = 0",  # not a partition key: the count would change
    f"SELECT d.n, d.c FROM ({COUNTED}) AS d WHERE d.c > 1",  # the window's own value
    f"SELECT d.n, d.c FROM ({COUNTED}) AS d WHERE d.d = 0 OR d.m = 1",
    f"SELECT d.n, d.c FROM ({COUNTED}) AS d WHERE d.d = (SELECT MAX(u.a) FROM u)",
    f"SELECT d.n, d.c FROM ({COUNTED}) AS d WHERE d.d + RANDOM() > 0",
    "SELECT d.n, d.c FROM (SELECT t.k AS n, t.a AS d, COUNT(*) OVER () AS c FROM t) AS d WHERE d.d = 0",  # one partition
    "SELECT d.n, d.c FROM (SELECT t.k AS n, t.a + 1 AS d, COUNT(*) OVER (PARTITION BY t.a) AS c FROM t) AS d WHERE d.d = 0",
    "SELECT d.n, d.c FROM (SELECT DISTINCT t.k AS n, t.a AS d, COUNT(*) OVER (PARTITION BY t.a) AS c FROM t) AS d WHERE d.d = 0",
    "SELECT d.n, d.c FROM (SELECT t.k AS n, t.a AS d, COUNT(*) OVER (PARTITION BY t.a) AS c FROM t LIMIT 3) AS d WHERE d.d = 0",
    "SELECT d.n, d.c FROM (SELECT t.k AS n, t.a AS d, COUNT(*) OVER w AS c FROM t WINDOW w AS (PARTITION BY t.a)) AS d WHERE d.d = 0",
    f"SELECT d.n, d.c FROM u LEFT JOIN ({COUNTED}) AS d ON u.k = d.n WHERE d.d = 0",
    f"SELECT d.n, d.c FROM ({COUNTED}) AS d JOIN u ON u.k = d.n WHERE d.d = 0",
    f"SELECT d.n, d.c FROM ({COUNTED}) AS d WHERE u.a = 0",
    "SELECT d.n FROM (SELECT t.k AS n, t.a AS d FROM t) AS d WHERE d.d = 0",  # no window: left to the general rule
]


@pytest.mark.parametrize("sql", FILTER_NEAR_MISSES)
def test_other_filters_stay_above(sql):
    assert applied(push_filter_below_windows, sql) is None


@pytest.mark.parametrize(
    "left, right",
    [
        (f"SELECT d.n, d.c FROM ({COUNTED}) AS d WHERE d.m = 0", "SELECT t.k, COUNT(*) OVER (PARTITION BY t.a) FROM t WHERE t.b = 0"),
        (
            "SELECT d.n, d.c FROM (SELECT t.k AS n, t.a AS d, COUNT(*) OVER () AS c FROM t) AS d WHERE d.d = 0",
            "SELECT t.k, COUNT(*) OVER () FROM t WHERE t.a = 0",
        ),
        (
            "SELECT d.n, d.s FROM (SELECT t.k AS n, t.a AS d, SUM(t.c) OVER (PARTITION BY t.b) AS s FROM t) AS d WHERE d.d = 0",
            "SELECT t.k, SUM(t.c) OVER (PARTITION BY t.b) FROM t WHERE t.a = 0",
        ),
        (
            "SELECT d.n, d.r FROM (SELECT t.k AS n, t.a AS d, RANK() OVER (PARTITION BY t.b ORDER BY t.c) AS r FROM t) AS d WHERE d.d > 0",
            "SELECT t.k, RANK() OVER (PARTITION BY t.b ORDER BY t.c) FROM t WHERE t.a > 0",
        ),
    ],
)
def test_a_filter_off_the_partition_key_is_not_proved_to_move(left, right):
    assert not proved(left, right)
    assert differs(left, right)  # the pair really differs on DuckDB, so a proof would be false


# ------------------------------------------------------------------ a windowed select over a projection


def test_a_windowed_select_reads_its_derived_tables_sources():
    nested = "SELECT d.x + d.y, COUNT(d.z) OVER (PARTITION BY d.w) FROM (SELECT t.a AS x, u.b AS y, t.c AS z, t.b AS w FROM t JOIN u ON t.k = u.k WHERE u.a > 0) AS d"
    out = applied(inline_windowed_source, nested)
    assert out is not None and " AS d" not in out and "PARTITION BY t.b" in out and "JOIN u ON t.k = u.k WHERE u.a > 0" in out
    flat = "SELECT t.a + u.b, COUNT(t.c) OVER (PARTITION BY t.b) FROM t JOIN u ON t.k = u.k WHERE u.a > 0"
    assert proved(nested, flat)
    other_names = "SELECT e.q + e.p, COUNT(e.r) OVER (PARTITION BY e.s) FROM (SELECT u.b AS p, t.a AS q, t.c AS r, t.b AS s FROM t JOIN u ON t.k = u.k WHERE u.a > 0) AS e"
    assert proved(nested, other_names)
    computed = "SELECT d.n, SUM(d.x) OVER (PARTITION BY d.w ORDER BY d.o) FROM (SELECT t.k AS n, t.a + 1 AS x, t.b AS w, t.c AS o FROM t) AS d"
    assert proved(computed, "SELECT t.k, SUM(t.a + 1) OVER (PARTITION BY t.b ORDER BY t.c) FROM t")


def test_two_levels_of_window_functions_share_one_list():
    stacked = "SELECT d.n, d.c, SUM(d.s) OVER (PARTITION BY d.m ORDER BY d.o) FROM (SELECT t.k AS n, t.b AS s, t.a AS m, t.c AS o, COUNT(*) OVER (PARTITION BY t.a) AS c FROM t) AS d"
    out = applied(inline_windowed_source, stacked)
    assert out is not None and " AS d" not in out and out.count(" OVER ") == 2
    assert proved(stacked, "SELECT t.k, COUNT(*) OVER (PARTITION BY t.a), SUM(t.b) OVER (PARTITION BY t.a ORDER BY t.c) FROM t")


SOURCE_NEAR_MISSES = [
    "SELECT d.n, SUM(d.c) OVER (PARTITION BY d.m) FROM (SELECT t.k AS n, t.a AS m, COUNT(*) OVER (PARTITION BY t.a) AS c FROM t) AS d",  # a window inside a window
    "SELECT d.n, SUM(d.s) OVER (PARTITION BY d.c) FROM (SELECT t.k AS n, t.b AS s, COUNT(*) OVER (PARTITION BY t.a) AS c FROM t) AS d",
    "SELECT d.n, SUM(d.s) OVER (PARTITION BY d.m) FROM (SELECT t.k AS n, t.b AS s, t.a AS m FROM t) AS d WHERE d.s > 0",  # its own WHERE
    "SELECT DISTINCT d.n, SUM(d.s) OVER (PARTITION BY d.m) FROM (SELECT t.k AS n, t.b AS s, t.a AS m FROM t) AS d",
    "SELECT d.n, SUM(d.s) OVER (PARTITION BY d.m) FROM (SELECT t.k AS n, t.b AS s, t.a AS m FROM t) AS d ORDER BY d.n LIMIT 2",
    "SELECT d.n, SUM(d.s) OVER (PARTITION BY d.m) FROM (SELECT DISTINCT t.k AS n, t.b AS s, t.a AS m FROM t) AS d",
    "SELECT d.n, SUM(d.s) OVER (PARTITION BY d.m) FROM (SELECT t.k AS n, t.b AS s, t.a AS m FROM t LIMIT 3) AS d",
    "SELECT d.n, SUM(d.s) OVER (PARTITION BY d.m) FROM (SELECT t.k AS n, SUM(t.b) AS s, t.a AS m FROM t GROUP BY t.k, t.a) AS d",
    "SELECT d.n, SUM(d.s) OVER (PARTITION BY d.m) FROM (SELECT t.k AS n, RANDOM() AS s, t.a AS m FROM t) AS d",
    "SELECT d.n, SUM(d.s) OVER (PARTITION BY d.m) FROM (SELECT t.k AS n, (SELECT MAX(u.b) FROM u) AS s, t.a AS m FROM t) AS d",
    "SELECT d.n, SUM(d.s) OVER (PARTITION BY d.m) FROM (SELECT t.k AS n, t.b AS s, t.a AS m FROM t) AS d JOIN u ON u.k = d.n",
    "SELECT d.n, SUM(d.s) OVER (PARTITION BY d.m), u.a FROM (SELECT t.k AS n, t.b AS s, t.a AS m FROM t) AS d, u",
    "SELECT d.n, MAX(d.s) FROM (SELECT t.k AS n, t.b AS s, t.a AS m FROM t) AS d GROUP BY d.n",
    "SELECT SUM(SUM(d.s)) OVER () FROM (SELECT t.k AS n, t.b AS s, t.a AS m FROM t) AS d",  # an aggregate of its own
    "SELECT d.n, d.s FROM (SELECT t.k AS n, t.b AS s, t.a AS m FROM t) AS d",  # no window: not this rule's
    "SELECT d.n, SUM(d.s) OVER (PARTITION BY d.m) FROM (SELECT t.k AS n, t.b AS s, t.a AS m FROM t UNION ALL SELECT u.k, u.b, u.a FROM u) AS d",
]


@pytest.mark.parametrize("sql", SOURCE_NEAR_MISSES)
def test_other_windowed_selects_keep_their_derived_table(sql):
    assert applied(inline_windowed_source, sql) is None


# ------------------------------------------------------------------ unread outputs


def test_an_unread_output_of_a_windowed_derived_table_is_dropped():
    wide = "SELECT d.n, d.c FROM (SELECT t.k AS n, t.b AS m, SUM(t.c) OVER (PARTITION BY t.a) AS c, RANK() OVER (ORDER BY t.b) AS r FROM t) AS d"
    out = applied(prune_windowed_derived, wide)
    assert out is not None and "RANK" not in out and "t.b" not in out and "SUM(t.c) OVER (PARTITION BY t.a)" in out
    assert proved(wide, "SELECT t.k, SUM(t.c) OVER (PARTITION BY t.a) FROM t")
    below = "SELECT e.n FROM (SELECT d.n, d.m FROM (SELECT t.k AS n, t.b AS m, SUM(t.c) OVER (PARTITION BY t.a) AS c FROM t) AS d) AS e"
    assert applied(prune_windowed_derived, below) is not None  # a window one level further down


PRUNE_NEAR_MISSES = [
    "SELECT d.n FROM (SELECT DISTINCT t.k AS n, t.b AS m, SUM(t.c) OVER (PARTITION BY t.a) AS c FROM t) AS d",  # m decides which rows collapse
    "SELECT d.n FROM (SELECT t.k AS n, t.b AS m, SUM(t.c) OVER (PARTITION BY t.a) AS c FROM t ORDER BY t.b LIMIT 2) AS d",
    "SELECT d.n FROM (SELECT t.k AS n, SUM(COUNT(*)) OVER () AS c FROM t GROUP BY t.k) AS d",
    "SELECT d.n FROM (SELECT MAX(t.k) AS n, SUM(COUNT(*)) OVER () AS c FROM t) AS d",
    "SELECT d.n FROM (SELECT t.k AS n, UNNEST(ARRAY[1, 2]) AS m, SUM(t.c) OVER () AS c FROM t) AS d",  # m makes rows
    "SELECT d.* FROM (SELECT t.k AS n, t.b AS m, SUM(t.c) OVER (PARTITION BY t.a) AS c FROM t) AS d",
    "SELECT d.n, d.m, d.c FROM (SELECT t.k AS n, t.b AS m, SUM(t.c) OVER (PARTITION BY t.a) AS c FROM t) AS d",  # all read
    "SELECT d.n FROM (SELECT t.k AS n, t.b AS m FROM t) AS d",  # no window: left to the general rule
    # an output that reads a column nothing provides is an error when the query runs: dropping it would turn a
    # query that fails into one that answers
    "SELECT d.n, d.c FROM (SELECT t.k AS n, t.nope AS m, SUM(t.c) OVER (PARTITION BY t.a) AS c FROM t) AS d",
    "SELECT d.n, d.c FROM (SELECT t.k AS n, nope AS m, SUM(t.c) OVER (PARTITION BY t.a) AS c FROM t) AS d",
    "SELECT d.n, d.c FROM (SELECT x.k AS n, x.b AS m, SUM(x.c) OVER (PARTITION BY x.a) AS c FROM nowhere AS x) AS d",  # no schema for the table
]


@pytest.mark.parametrize("sql", PRUNE_NEAR_MISSES)
def test_other_derived_tables_keep_their_outputs(sql):
    assert applied(prune_windowed_derived, sql) is None


def test_pruning_does_not_merge_a_distinct_that_reads_the_column():
    left = "SELECT d.n FROM (SELECT DISTINCT t.a AS n, t.b AS m, COUNT(*) OVER (PARTITION BY t.a) AS c FROM t) AS d"
    right = "SELECT d.n FROM (SELECT DISTINCT t.a AS n, COUNT(*) OVER (PARTITION BY t.a) AS c FROM t) AS d"
    assert not proved(left, right)
    assert differs(left, right)


# ------------------------------------------------------------------ all three together


def test_the_rules_return_the_same_rows_on_duckdb():
    cases = [
        (push_filter_below_windows, f"SELECT d.n, d.c FROM ({COUNTED}) AS d WHERE d.d > 0 AND d.m = 1"),
        (push_filter_below_windows, "SELECT d.n, d.r FROM (SELECT t.k AS n, t.a AS d, RANK() OVER (PARTITION BY t.a ORDER BY t.b NULLS LAST) AS r FROM t WHERE t.c > 0) AS d WHERE d.d IS NULL"),
        (inline_windowed_source, "SELECT d.x + d.y, COUNT(d.z) OVER (PARTITION BY d.w) FROM (SELECT t.a AS x, u.b AS y, t.c AS z, t.b AS w FROM t JOIN u ON t.k = u.k) AS d"),
        (inline_windowed_source, "SELECT d.n, d.c, SUM(d.s) OVER (PARTITION BY d.m) FROM (SELECT t.k AS n, t.b AS s, t.a AS m, COUNT(*) OVER (PARTITION BY t.c) AS c FROM t) AS d"),
        (prune_windowed_derived, "SELECT d.n, d.c FROM (SELECT t.k AS n, t.b AS m, SUM(t.c) OVER (PARTITION BY t.a) AS c, RANK() OVER (ORDER BY t.b) AS r FROM t) AS d"),
        (settle_window_levels, f"SELECT e.n, e.c FROM (SELECT d.n, d.d, d.c, d.m FROM ({COUNTED}) AS d) AS e WHERE e.d = 1"),
    ]
    for rule, sql in cases:
        out = applied(rule, sql)
        assert out is not None and out != sql, sql
        for rows in databases(40, seed=20261113):
            before, after = results([sql, out], rows)
            assert before == after, (sql, rows)


def _random_query(rng):
    names, items = [], []
    for column in rng.sample(SCHEMA["t"], rng.randint(1, 4)):
        names.append((f"p{len(names)}", column))
        items.append(f"t.{column} AS {names[-1][0]}")
    for _ in range(rng.randint(1, 2)):
        keys = rng.sample(["a", "b", "c"], rng.randint(0, 2))
        call = rng.choice(["SUM(t.c)", "COUNT(t.b)", "COUNT(*)", "MIN(t.k)", "RANK()", "DENSE_RANK()"])
        order = f" ORDER BY t.{rng.choice(SCHEMA['t'])} NULLS FIRST, t.k DESC NULLS LAST" if "RANK" in call or rng.random() < 0.3 else ""
        partition = "PARTITION BY " + ", ".join(f"t.{key}" for key in keys) if keys else ""
        names.append((f"w{len(names)}", None))
        items.append(f"{call} OVER ({partition}{order}) AS {names[-1][0]}")
    body = f"SELECT {', '.join(items)} FROM t{rng.choice(['', ' WHERE t.k > 0'])}"
    plain = [name for name, column in names if column]
    kind = rng.choice("FSP")
    if kind == "F":
        tests = [rng.choice([f"d.{name} = 1", f"d.{name} > 0", f"d.{name} IS NULL"]) for name, _ in rng.sample(names, rng.randint(1, min(2, len(names))))]
        return f"SELECT {', '.join('d.' + name for name, _ in names)} FROM ({body}) AS d WHERE {' AND '.join(tests)}"
    if kind == "S":
        keys = rng.sample(plain, rng.randint(0, min(2, len(plain))))
        partition = "PARTITION BY " + ", ".join(f"d.{key}" for key in keys) if keys else ""
        return f"SELECT {', '.join('d.' + name for name, _ in names)}, COUNT(d.{rng.choice(plain)}) OVER ({partition}) FROM ({body}) AS d"
    return f"SELECT {', '.join('d.' + name for name, _ in rng.sample(names, rng.randint(1, len(names))))} FROM ({body}) AS d"


def test_random_queries_keep_their_rows_through_the_pass():
    rng = random.Random(20261114)
    rewritten = 0
    for _ in range(120):
        sql = _random_query(rng)
        out = applied(settle_window_levels, sql)
        if out is None or out == sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres"):
            continue
        rewritten += 1
        for rows in databases(4, seed=rng.randint(0, 10**6)):
            before, after = results([sql, out], rows)
            assert before == after, (sql, out, rows)
    assert rewritten > 40


def test_the_rules_are_off_under_the_shape_fallback():
    above = f"SELECT d.n, d.c FROM ({COUNTED}) AS d WHERE d.d = 0"
    below = "SELECT t.k, COUNT(*) OVER (PARTITION BY t.a) FROM t WHERE t.a = 0"
    assert proved(above, below)
    with shape_fallback.disabled():
        assert not proved(above, below)
