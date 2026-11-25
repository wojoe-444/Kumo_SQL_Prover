"""A column list on a ``WITH`` name, read in every attempt of the last proof step (``late_cte_lists``).

``listed_cte_names`` writes the list as aliases, in a stage that makes one attempt. A pair that needs the list read
and a reading of a later attempt was not reached. The rewriting itself is ``listed_cte_names``'s, tested in
``tests/test_listed_cte_names.py`` and ``tests/test_listed_cte_reading.py``; this file checks where it runs.
"""

import pytest

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"shops": ["shop_id", "city"], "calls": ["call_id", "shop_id", "seen"]}
TYPES = {"shops": {"shop_id": "INT", "city": "VARCHAR"}, "calls": {"call_id": "INT", "shop_id": "INT", "seen": "DATE"}}
OPTIONS = dict(
    schema=SCHEMA,
    types=TYPES,
    constraints={
        "shops": TableConstraints(not_null=frozenset({"shop_id", "city"}), keys=(("shop_id",),)),
        "calls": TableConstraints(not_null=frozenset({"call_id", "shop_id", "seen"}), keys=(("call_id",),)),
    },
    compare_names=False,
    dialect="mysql",
    exact_arithmetic=True,
    timeout_ms=3000,
)
LISTED = "WITH t (k, c) AS (SELECT shop_id, call_id FROM calls) SELECT s.city FROM shops AS s WHERE s.shop_id NOT IN (SELECT k FROM t)"
READ = "SELECT s.city FROM shops AS s WHERE NOT s.shop_id IN (SELECT t.shop_id AS k FROM calls AS t)"


def _normalized(sql: str) -> str:
    return normalize(sql, schema=SCHEMA, dialect="mysql")


def test_the_last_step_reads_the_list_when_the_thirteenth_batch_is_off():
    # a rule of the thirteenth batch switched off leaves the batch on; an older rule switched off turns it off, which
    # is what an attempt of the last step with an older rule off looks like
    with shape_fallback.twenty_seventh_on(), shape_fallback.without("distinct_derived_joins"):
        assert not shape_fallback.thirteenth_batch_enabled()
        assert _normalized(LISTED) == READ
        with shape_fallback.without("late_cte_lists"):
            assert "t(k, c)" in _normalized(LISTED)
        with shape_fallback.without("listed_cte_names"):
            assert "t(k, c)" in _normalized(LISTED)


def test_outside_the_last_step_the_list_is_read_as_before():
    with shape_fallback.without("distinct_derived_joins"):
        assert not shape_fallback.thirteenth_batch_enabled() and not shape_fallback.twenty_seventh_batch_enabled()
        assert "t(k, c)" in _normalized(LISTED)
    assert _normalized(LISTED) == READ  # outside a proof the thirteenth batch reads it
    with shape_fallback.without("late_cte_lists"):
        assert _normalized(LISTED) == READ


# shops whose every call was in the first week of 2024
BOUNDS = "SELECT s.shop_id, s.city FROM shops AS s JOIN (SELECT shop_id FROM calls GROUP BY shop_id HAVING MIN(seen) >= '2024-01-01' AND MAX(seen) <= '2024-01-07') AS v ON s.shop_id = v.shop_id"
NOT_OUTSIDE = (
    "WITH outside{0} AS (SELECT a.shop_id, b.call_id FROM shops AS a JOIN calls AS b ON a.shop_id = b.shop_id WHERE b.seen < '2024-01-01' OR b.seen > '{2}')"
    " SELECT a.shop_id, a.city FROM shops AS a JOIN calls AS b ON a.shop_id = b.shop_id WHERE a.shop_id NOT IN (SELECT {1} FROM outside) GROUP BY a.shop_id"
)


@pytest.mark.parametrize("names, read", [("(shop_id, call_id)", "shop_id"), ("(sid, cid)", "sid")])
def test_a_pair_is_proved_and_only_with_the_list_read_in_the_last_step(names, read):
    left, right = BOUNDS, NOT_OUTSIDE.format(names, read, "2024-01-07")
    assert prove_equivalent_algebraic(left, right, **OPTIONS).proven
    with shape_fallback.without("late_cte_lists"):
        assert not prove_equivalent_algebraic(left, right, **OPTIONS).proven


def test_the_pair_without_a_list_does_not_need_it():
    with shape_fallback.without("late_cte_lists"):
        assert prove_equivalent_algebraic(BOUNDS, NOT_OUTSIDE.format("", "shop_id", "2024-01-07"), **OPTIONS).proven


@pytest.mark.parametrize(
    "names, read, last",
    [
        ("(call_id, shop_id)", "shop_id", "2024-01-07"),  # the list calls the call's id shop_id
        ("(shop_id, call_id)", "call_id", "2024-01-07"),
        ("(shop_id, call_id)", "shop_id", "2024-01-08"),  # a day more
    ],
)
def test_pairs_that_differ_stay_unproved(names, read, last):
    assert not prove_equivalent_algebraic(BOUNDS, NOT_OUTSIDE.format(names, read, last), **OPTIONS).proven
