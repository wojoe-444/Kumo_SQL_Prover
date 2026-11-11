"""Bounded equivalence: SQL the encoder learned after the first version, each construct against the replay engine.

Every construct has (a) pairs that must come out bounded-equivalent, (b) pairs that must come out different with a replayed
counterexample, and (c) the encoding compared with DuckDB on random small databases with NULLs, duplicates and empty tables
(the query DuckDB runs is the one the replay runs, so an encoding that disagrees with the replay fails here).
"""

import random

import pytest

pytest.importorskip("z3")
pytest.importorskip("duckdb")

from kumo_sql_prover import bounded_equivalence as be  # noqa: E402
from kumo_sql_prover.bounded_equivalence import BColumn, BoundedSchema, BoundedStatus, BTable, check_bounded  # noqa: E402


def schema() -> BoundedSchema:
    t = BTable("t", [BColumn("a", "INT"), BColumn("b", "INT")])
    u = BTable("u", [BColumn("a", "INT", True), BColumn("c", "VARCHAR")])
    e = BTable("e", [BColumn("id", "INT"), BColumn("d", "DATE")])
    return BoundedSchema({"t": t, "u": u, "e": e})


def check(left, right, rows=2, **kwargs):
    return check_bounded(left, right, schema(), rows=rows, dialect="mysql", **kwargs)


def same(left, right, rows=2):
    result = check(left, right, rows)
    assert result.status is BoundedStatus.BOUNDED_EQUIVALENT and result.bound == rows, (left, right, result.reason, result.counterexample)


def different(left, right, rows=2):
    result = check(left, right, rows)
    assert result.status is BoundedStatus.DIFFERENT and result.counterexample is not None, (left, right, result.reason)
    return result.counterexample


def unsupported(left, right, word=""):
    result = check(left, right)
    assert result.status is BoundedStatus.UNKNOWN and "unsupported" in result.reason and word in result.reason, (left, right, result.reason)


DATES = [None, "2020-01-31", "2020-02-01", "2020-02-29", "2019-12-31", "2021-01-01"]


def concrete(rng):
    return {
        "t": [(rng.choice([None, 1, 2, 3]), rng.choice([None, 0, 1, 3])) for _ in range(rng.randint(0, 3))],
        "u": [(rng.choice([1, 2, 3]), rng.choice([None, "a", "ab", "b"])) for _ in range(rng.randint(0, 3))],
        "e": [(rng.randint(1, 3), rng.choice(DATES)) for _ in range(rng.randint(0, 3))],
    }


def norm(rows):
    def cell(value):
        if hasattr(value, "hour") and hasattr(value, "date") and not (value.hour or value.minute or value.second):
            return value.date()  # DuckDB gives a TIMESTAMP for DATE + INTERVAL
        if isinstance(value, bool):
            return int(value)
        if hasattr(value, "is_finite") or isinstance(value, float):
            return round(float(value), 6)
        return value

    return sorted((tuple(cell(v) for v in row) for row in rows), key=repr)


def agrees_with_replay(sql, trials=6, seed=1):
    """The encoding's rows for ``sql`` equal the rows of the query the replay runs, on random databases."""

    s = schema()
    replay = be.DuckDBReplay(s, sql, sql, "mysql")
    rng = random.Random(seed)
    compared = 0
    for trial in range(trials):
        data = concrete(rng) if trial else {"t": [], "u": [], "e": []}  # the empty database first
        mine = be.evaluate(sql, s, data, "mysql")
        theirs, _ = replay._run(data)
        assert norm(mine) == norm(theirs), (sql, data, mine, theirs)
        compared += 1
    return compared


# --- parentheses in FROM ---------------------------------------------------------------------------


def test_a_parenthesized_join_keeps_its_joins():
    same("select x.a from ((select a from t) x join (select a from u) y on x.a = y.a)", "select t.a from t join u on t.a = u.a")
    same("select t.a from t join (u join e on u.a = e.id) on t.a = u.a", "select t.a from t join u on t.a = u.a join e on u.a = e.id")
    same("select t.a from (t left join u on t.a = u.a)", "select t.a from t left join u on t.a = u.a")
    # the join inside the parentheses used to be dropped, which made these two "equivalent"
    different("select x.a from ((select a from t) x join (select a from u) y on true)", "select a from t")
    # an inner join inside a left join is not the same as two left joins
    different(
        "select t.a, e.id from t left join (u join e on u.a = e.id) on t.a = u.a",
        "select t.a, e.id from t left join u on t.a = u.a left join e on u.a = e.id",
    )


@pytest.mark.parametrize("sql", [
    "select t.a, u.c, e.id from t left join (u join e on u.a = e.id) on t.a = u.a",
    "select x.a, y.c from ((select a from t) x join (select a, c from u) y on x.a = y.a)",
    "select t.b from (t join u on t.a = u.a)",
])
def test_parenthesized_joins_match_the_replay(sql):
    assert agrees_with_replay(sql)


def test_a_clause_on_a_parenthesized_query_is_not_dropped():
    # ``(SELECT ..) LIMIT 1`` used to compile as the inner query alone
    unsupported("(select a from t order by a) limit 1", "select a from t", "limit")
    same("(select a from t) order by a", "select a from t")
    unsupported("select t.a from (t join u on t.a = u.a) as j", "select t.a from t join u on t.a = u.a", "alias")


# --- parsing and scoping ---------------------------------------------------------------------------


def test_is_binds_after_a_comparison():
    # sqlglot reads ``a = 2 IS TRUE`` as ``a = (2 IS TRUE)``; the engines read ``(a = 2) IS TRUE``
    same("select a from t where a = 2 is true", "select a from t where a = 2")
    same("select a from t where a < b is null", "select a from t where a is null or b is null")
    cx = different("select a from t where a = 2 is true", "select a from t where a = 1")
    assert cx["t"][0][0] in (1, 2)
    for sql in ("select a, b from t where a = 2 is true or b < 1 is false", "select a from t where a >= b is null"):
        assert agrees_with_replay(sql)
    # ``a = 2 IS NOT TRUE`` goes to the replay as ``a = NOT 2 IS TRUE``, which means something else there
    unsupported("select a from t where a = 2 is not true", "select a from t where a <> 2 or a is null", "IS NOT")


def test_an_unknown_function_is_a_predicate_only_where_a_condition_is_expected():
    same("select a from t where p(a) and not p(a)", "select a from t where false")
    same("select case when p(a) then 1 else 0 end from t", "select if(p(a), 1, 0) from t")
    same("select t.a from t join u on p(t.a, u.a)", "select t.a from t, u where p(t.a, u.a)")
    # ``YEARWEEK(a) = 201901`` used to compare a truth value with a number, which is never true
    unsupported("select a from t where yearweek(a) = 201901", "select a from t where false", "function YEARWEEK")
    unsupported("select unix_timestamp(c) from u", "select unix_timestamp(c) from u", "function UNIX_TIMESTAMP")
    unsupported("select a from t where f(a) + 1 > 0", "select a from t", "function F")


def test_a_window_inside_a_subquery_does_not_make_the_outer_query_windowed():
    inner = "select a, lead(a) over (order by a) as nxt from t"
    same(f"select min(nxt - a) from ({inner}) x", f"select min(d) from (select nxt - a as d from ({inner}) y) z")
    same(f"with w as ({inner}) select count(*), max(nxt) from w", "select count(*), case when count(a) > 1 or count(a) < count(*) then max(a) end from t")
    different(f"with w as ({inner}) select count(nxt) from w", "select count(a) from t")
    assert agrees_with_replay(f"with w as ({inner}) select count(nxt), max(a) from w")


def test_single_value_is_the_value_of_the_only_row():
    # Calcite's SINGLE_VALUE: the row's value, NULL without rows; several rows are a runtime error (not modeled, the
    # first row is taken, as for the scalar subquery it replaces)
    same("select t.a, (select b from t x where x.a = 1) from t", "select t.a, s.v from t left join (select single_value(b) as v from t where a = 1) s on true")
    same("select single_value(a) from t where a > 5 and a < 5", "select max(a) from t where false")
    unsupported("select a from t where single_value(a, b) = 1", "select a from t", "function SINGLE_VALUE")
    # DuckDB has no SINGLE_VALUE, so the replay cannot confirm a difference: unknown, never a verdict
    result = check("select single_value(a) from t", "select count(a) from t")
    assert result.status is BoundedStatus.UNKNOWN


# --- names that a select item and several FROM columns share -----------------------------------------


def test_a_group_by_name_two_tables_share_is_the_select_item():
    # MySQL: ``GROUP BY a`` with ``a`` in both tables groups by the select item named ``a`` (the replay rewrites it so)
    joined = "from t join u on t.b = u.a"
    same(f"select t.a, count(*) {joined} group by a", f"select t.a, count(*) {joined} group by t.a")
    same(f"select t.b as a, count(*) {joined} group by a", f"select t.b, count(*) {joined} group by t.b")
    different(f"select t.a, count(*) {joined} group by a", f"select u.a, count(*) {joined} group by u.a")
    # no select item of that name, two of them, or a use the replay engine rejects: still ambiguous
    unsupported(f"select t.a as k, count(*) {joined} group by a", f"select t.a, count(*) {joined} group by t.a", "ambiguous")
    unsupported(f"select t.a, u.a, count(*) {joined} group by a", f"select t.a, u.a, count(*) {joined} group by t.a, u.a", "ambiguous")
    unsupported(f"select t.a, count(*) {joined} group by t.a having a > 1", f"select t.a, count(*) {joined} group by t.a having t.a > 1", "ambiguous")


def test_order_by_reads_a_select_item_before_a_from_column():
    joined = "from t join u on t.b = u.a"
    same(f"select t.a, u.c {joined} order by a limit 1", f"select t.a, u.c {joined} order by t.a limit 1")
    same(f"select t.*, u.c {joined} order by a limit 1", f"select t.a, t.b, u.c {joined} order by t.a limit 1")
    unsupported(f"select t.a, u.a {joined} order by a limit 1", f"select t.a, u.a {joined} order by t.a limit 1", "ambiguous")
    # an alias after a star used to be paired with the wrong output column
    same("select *, -a as x from t where a is not null order by x limit 1", "select a, b, -a from t where a is not null order by a desc limit 1")
    different("select *, -a as x from t where a is not null and b is not null order by x limit 1", "select a, b, -a from t where a is not null and b is not null order by b limit 1")


@pytest.mark.parametrize("sql", [
    "select t.a, count(*), max(u.c) from t join u on t.b = u.a group by a",
    "select t.b as a, count(u.c) from t left join u on t.b = u.a group by a",
    "select u.a, t.b from t join u on t.b = u.a order by a, b limit 2",
    "select *, -a as x from t order by x, b limit 1",
    "select t.*, u.c from t left join u on t.a = u.a order by a, b, c limit 2",
])
def test_shared_names_match_the_replay(sql):
    assert agrees_with_replay(sql)


# --- FILTER, FETCH, explicit window frames ---------------------------------------------------------


def test_an_aggregate_filter_restricts_the_rows_aggregated():
    same("select count(*) filter (where a > 1), sum(b) filter (where a > 1) from t", "select count(case when a > 1 then 1 end), sum(case when a > 1 then b end) from t")
    same("select a, count(distinct b) filter (where b > 0) from t group by a", "select a, count(distinct case when b > 0 then b end) from t group by a")
    same("select a, sum(b) filter (where b > 0) over (partition by a) from t", "select a, sum(case when b > 0 then b end) over (partition by a) from t")
    # a NULL condition filters the row out; an empty filtered group sums to NULL, not 0
    different("select sum(b) filter (where a > 1) from t", "select coalesce(sum(b) filter (where a > 1), 0) from t", rows=1)
    different("select count(*) filter (where a > 1) from t", "select count(*) filter (where not (a <= 1)) + count(*) filter (where a is null) from t", rows=1)
    unsupported("select row_number() filter (where a > 1) over (order by a) from t", "select 1 from t", "FILTER")


def test_fetch_first_is_limit():
    same("select a from t order by a fetch next 1 rows only", "select a from t order by a limit 1")
    same("select a from t order by a offset 1 rows fetch next 2 rows only", "select a from t order by a limit 2 offset 1")
    different("select a from t order by a fetch first 1 rows only", "select a from t order by a limit 2")
    unsupported("select a from t order by a fetch first 1 rows with ties", "select a from t order by a limit 1", "WITH TIES")


def test_the_two_frames_that_do_not_depend_on_ties():
    whole = "range between unbounded preceding and unbounded following"
    same(f"select a, sum(b) over (partition by a {whole}) from t", "select a, sum(b) over (partition by a) from t")
    same(f"select a, count(b) over (partition by a order by b rows between unbounded preceding and unbounded following) from t", "select a, count(b) over (partition by a) from t")
    same("select a, sum(b) over (partition by a order by b range between unbounded preceding and current row) from t", "select a, sum(b) over (partition by a order by b) from t")
    # with ORDER BY the default frame ends at the current row's peers, the whole-partition frame does not
    different(f"select sum(b) over (order by b {whole}) from t", "select sum(b) over (order by b) from t")
    unsupported("select sum(b) over (order by b rows between unbounded preceding and current row) from t", "select sum(b) over (order by b) from t", "window frame")
    unsupported("select sum(b) over (order by b rows between 1 preceding and current row) from t", "select sum(b) over (order by b) from t", "window frame")


@pytest.mark.parametrize("sql", [
    "select a, count(*) filter (where b > 0), sum(b) filter (where b <> 1), min(b) filter (where a is null), avg(b) filter (where b > 0) from t group by a",
    "select count(*) filter (where a > 1), count(distinct b) filter (where a > 1), max(a) filter (where b is null) from t",
    "select a, b, sum(b) filter (where b > 0) over (partition by a), count(*) filter (where a > 1) over () from t",
    "select a, b from t order by a, b offset 1 rows fetch next 2 rows only",
    "select a, b, sum(b) over (partition by a order by b range between unbounded preceding and current row) from t",
    "select a, b, count(*) over (order by b rows between unbounded preceding and unbounded following), max(b) over (partition by a order by b range between unbounded preceding and unbounded following) from t",
    "select a, b, first_value(b) over (partition by a order by b, a range between unbounded preceding and unbounded following) from t where b is not null",
])
def test_filter_fetch_and_frames_match_the_replay(sql):
    assert agrees_with_replay(sql)
