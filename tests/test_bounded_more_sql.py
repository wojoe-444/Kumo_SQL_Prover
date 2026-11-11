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
