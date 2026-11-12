"""An aggregate inside a window (``SUM(COUNT(*)) OVER ()``) is a real aggregate of its select.

The window pruning rules read every aggregate under a window as the window's own function, so they dropped the
column holding the select's only aggregate: one row of ``SELECT 1 AS x, SUM(COUNT(*)) OVER () FROM events`` became
one row per event.
"""

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtStatus
from kumo_sql_prover.unread_windows import drop_unread_windows
from kumo_sql_prover.window_calls import is_window_function, own_aggregates
from kumo_sql_prover.window_pruning import prune_windowed_derived

SCHEMA = {"events": ["id", "k"]}
RIGHT = "SELECT 1 AS x FROM events"

AGGREGATE_UNDER_A_WINDOW = [
    "SELECT x FROM (SELECT 1 AS x, SUM(COUNT(*)) OVER () AS w FROM events) AS d",
    "SELECT x FROM (SELECT 1 AS x, COUNT(*) OVER (PARTITION BY COUNT(*)) AS w FROM events) AS d",
    "SELECT x FROM (SELECT 1 AS x, SUM(id) OVER (PARTITION BY COUNT(*)) AS w FROM events) AS d",
    "SELECT x FROM (SELECT 1 AS x, ROW_NUMBER() OVER (ORDER BY MAX(id)) AS w FROM events) AS d",
    "SELECT x FROM (SELECT 1 AS x, SUM(COUNT(*)) OVER (PARTITION BY MAX(id)) AS w FROM events) AS d",
]


def parsed(sql):
    return sqlglot.parse_one(sql, read="bigquery")


@pytest.mark.parametrize("left", AGGREGATE_UNDER_A_WINDOW)
def test_a_select_aggregating_under_a_window_is_not_a_row_per_row_select(left):
    assert prove_equivalent_algebraic(left, RIGHT, schema=SCHEMA).status is not SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize("left", AGGREGATE_UNDER_A_WINDOW)
def test_neither_pruning_rule_drops_the_window_column(left):
    tree = parsed(left)
    assert prune_windowed_derived(tree) is None
    assert drop_unread_windows(tree.copy()).sql(dialect="bigquery") == tree.sql(dialect="bigquery")


def test_a_plain_window_column_is_still_pruned_and_proves():
    left = "SELECT x FROM (SELECT 1 AS x, SUM(id) OVER (PARTITION BY k) AS w FROM events) AS d"
    assert prove_equivalent_algebraic(left, RIGHT, schema=SCHEMA).status is SmtStatus.PROVEN_EQUIVALENT
    assert prune_windowed_derived(parsed(left), SCHEMA) is not None


def test_only_the_function_of_a_window_is_a_window_function():
    select = parsed("SELECT SUM(COUNT(*)) OVER (PARTITION BY MAX(id) ORDER BY MIN(k)), AVG(id) OVER () FROM events")
    calls = {c.sql(): is_window_function(c) for c in select.find_all(sqlglot.exp.AggFunc)}
    assert calls == {
        "SUM(COUNT(*))": True,
        "COUNT(*)": False,
        "MAX(id)": False,
        "MIN(k)": False,
        "AVG(id)": True,
    }
    assert sorted(c.sql() for c in own_aggregates(select)) == ["COUNT(*)", "MAX(id)", "MIN(k)"]


@pytest.mark.parametrize("left", AGGREGATE_UNDER_A_WINDOW)
def test_the_other_window_rules_keep_the_aggregate_under_a_window(left):
    """``windowed_output_pruning``, ``windowed_inputs`` and ``one_row_groups`` read the same way as the two above."""

    from kumo_sql_prover.one_row_groups import collapse_one_row_groups
    from kumo_sql_prover.windowed_inputs import spell_windowed_input
    from kumo_sql_prover.windowed_output_pruning import prune_windowed_outputs

    tree = parsed(left)
    assert prune_windowed_outputs(tree.copy(), SCHEMA) is None
    assert collapse_one_row_groups(tree.copy()) is None
    assert spell_windowed_input(tree.find(sqlglot.exp.Subquery).this.copy()) is None
