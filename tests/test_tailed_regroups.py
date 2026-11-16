"""A regrouping with ``ORDER BY`` / ``LIMIT`` folds like one without (``tailed_regroups``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.algebraic_equivalence import _collapse_aggregate, _regroup_distinct, _roll_up_aggregate
from kumo_sql_prover.tailed_regroups import regroup_under_tail

RULES = (_collapse_aggregate, _roll_up_aggregate, _regroup_distinct)
TWICE = "SELECT t.a, COUNT(t.x) AS n FROM (SELECT a, x FROM s WHERE b > 1 GROUP BY a, x) AS t GROUP BY t.a"
ONCE = "SELECT a AS a, COUNT(DISTINCT x) AS n FROM s WHERE b > 1 GROUP BY a"


def folded(sql: str, rules=RULES) -> str | None:
    result = regroup_under_tail(sqlglot.parse_one(sql, read="duckdb"), rules)
    return None if result is None else result.sql(dialect="duckdb")


def test_the_tail_is_kept_on_the_folded_select():
    # a sort key that is an output, by name or by its expression, is the folded output of that name
    assert folded(TWICE + " ORDER BY n DESC, t.a LIMIT 10") == ONCE + " ORDER BY COUNT(DISTINCT x) DESC, a LIMIT 10"
    assert folded(TWICE + " ORDER BY COUNT(t.x) DESC, a LIMIT 10 OFFSET 2") == ONCE + " ORDER BY COUNT(DISTINCT x) DESC, a LIMIT 10 OFFSET 2"
    # a sort key that is no output is folded with the select and does not become one
    assert folded(TWICE + " ORDER BY MAX(t.x) NULLS FIRST, t.a") == ONCE + " ORDER BY MAX(x) NULLS FIRST, a"
    # a constant key and a LIMIT alone
    assert folded(TWICE + " ORDER BY 'x', t.a LIMIT 3") == ONCE + " ORDER BY 'x', a LIMIT 3"
    assert folded(TWICE + " LIMIT 3") == ONCE + " LIMIT 3"
    # a partial aggregate rolled up, with its tail
    rolled = "SELECT t.a, SUM(t.p) AS total FROM (SELECT a, x, SUM(c) AS p FROM s GROUP BY a, x) AS t GROUP BY t.a ORDER BY total DESC, t.a LIMIT 5"
    assert folded(rolled) == "SELECT a AS a, SUM(c) AS total FROM s GROUP BY a ORDER BY SUM(c) DESC, a LIMIT 5"


@pytest.mark.parametrize(
    "sql",
    [
        TWICE,  # no tail: the folding rules read it themselves
        "SELECT t.a, t.x FROM (SELECT a, x FROM s GROUP BY a, x) AS t ORDER BY t.a LIMIT 3",  # not a regrouping
        "SELECT t.a, COUNT(t.x) AS n FROM (SELECT a, x FROM s) AS t GROUP BY t.a ORDER BY t.a LIMIT 3",  # the source is not grouped
        "SELECT t.a, COUNT(t.x) AS n FROM (SELECT a, x FROM s GROUP BY a, x) AS t JOIN u ON u.a = t.a GROUP BY t.a ORDER BY t.a LIMIT 3",
        "SELECT a, COUNT(x) AS n FROM s GROUP BY a ORDER BY a LIMIT 3",  # one grouping
        TWICE.replace("COUNT(t.x) AS n", "MEDIAN(t.x) AS n") + " ORDER BY t.a LIMIT 3",  # no folding rule reads it
        TWICE.replace("COUNT(t.x) AS n", "COUNT(t.x) AS n, ROW_NUMBER() OVER (ORDER BY t.a) AS r") + " ORDER BY t.a LIMIT 3",
        TWICE.replace("SELECT t.a,", "SELECT DISTINCT t.a,") + " ORDER BY t.a LIMIT 3",
        TWICE.replace("COUNT(t.x) AS n", "COUNT(t.x)") + " ORDER BY t.a LIMIT 3",  # an output without a name
        TWICE + " ORDER BY (SELECT MAX(b) FROM s) LIMIT 3",  # a subquery as a sort key
    ],
)
def test_other_selects_are_left_alone(sql):
    assert folded(sql) is None


def test_nothing_is_rewritten_when_no_folding_rule_fires():
    assert folded(TWICE + " ORDER BY t.a LIMIT 3", rules=()) is None
    assert folded(TWICE + " ORDER BY t.a LIMIT 3", rules=(lambda select: None,)) is None
    # a rule that changes the output names is not trusted with the tail
    renamed = lambda select: sqlglot.parse_one("SELECT a AS other, COUNT(DISTINCT x) AS n FROM s GROUP BY a")  # noqa: E731
    assert folded(TWICE + " ORDER BY t.a LIMIT 3", rules=(renamed,)) is None


AGGREGATES = ["COUNT(t.x) AS n", "SUM(t.x) AS n", "MAX(t.x) AS n", "COUNT(t.x) AS n, MIN(t.x) AS m"]
TAILS = ["ORDER BY n DESC NULLS LAST, t.a NULLS FIRST LIMIT {k}", "ORDER BY t.a DESC NULLS LAST LIMIT {k} OFFSET 1", "ORDER BY MAX(t.x) NULLS FIRST, t.a NULLS LAST", "ORDER BY 2, 1 NULLS FIRST LIMIT {k}"]


@pytest.mark.parametrize("seed", range(3))
def test_the_folded_select_returns_the_rows_duckdb_returns_in_order(seed):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.algebraic_equivalence import _resolve_ordinals

    rng = random.Random(seed)
    fired = 0
    for _ in range(12):
        sql = f"SELECT t.a, {rng.choice(AGGREGATES)} FROM (SELECT a, x FROM s WHERE b {rng.choice(['> 0', 'IS NOT NULL', '< 3'])} GROUP BY a, x) AS t GROUP BY t.a " + rng.choice(TAILS).format(k=rng.randint(1, 4))
        result = regroup_under_tail(_resolve_ordinals(sqlglot.parse_one(sql, read="duckdb"), group_by=True), RULES)
        assert result is not None, sql
        fired += 1
        for _ in range(4):
            con = duckdb.connect()
            con.execute("CREATE TABLE s (a INT, x INT, b INT)")
            value = lambda: rng.choice([None, 0, 1, 2, 3])  # noqa: E731
            for _ in range(rng.randint(0, 12)):
                con.execute("INSERT INTO s VALUES (?, ?, ?)", [value(), value(), value()])
            # every tail sorts by the one grouping key last, so the order is total and the rows compare as lists
            assert con.execute(sql).fetchall() == con.execute(result.sql(dialect="duckdb")).fetchall(), (sql, result.sql(dialect="duckdb"))
    assert fired == 12


z3 = pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic  # noqa: E402

SCHEMA = {"s": ["a", "x", "b"]}
TAIL = " ORDER BY n DESC, a LIMIT 10"


def proven(left: str, right: str) -> bool:
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, dialect="duckdb").proven


def test_a_distinct_count_written_as_two_groupings_meets_the_plain_one_under_a_tail():
    plain = "SELECT a, COUNT(DISTINCT x) AS n FROM s WHERE b > 1 GROUP BY a"
    assert proven(TWICE, plain)  # as before
    assert proven(TWICE + " ORDER BY n DESC, t.a LIMIT 10", plain + TAIL)
    assert proven(TWICE + " ORDER BY COUNT(t.x) DESC, t.a LIMIT 10", plain + TAIL)


def test_a_different_tail_or_body_is_not_proved():
    plain = "SELECT a, COUNT(DISTINCT x) AS n FROM s WHERE b > 1 GROUP BY a"
    assert not proven(TWICE + " ORDER BY n DESC, t.a LIMIT 10", plain + " ORDER BY n, a LIMIT 10")
    assert not proven(TWICE + " ORDER BY n DESC, t.a LIMIT 10", plain + " ORDER BY n DESC, a LIMIT 9")
    assert not proven(TWICE + " ORDER BY n DESC, t.a LIMIT 10", plain.replace("COUNT(DISTINCT x)", "COUNT(x)") + TAIL)
