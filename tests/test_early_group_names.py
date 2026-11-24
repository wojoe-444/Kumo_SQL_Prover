"""A ``GROUP BY`` name two joined tables share is read at every stage of a proof (MySQL).

``shared_group_names`` reads the name as the select-list column MySQL means. It was read in the fourth stage only,
which makes one attempt, so a pair that needs the reading and then one of the later attempts of the first stage
stayed unproved: the grouped join below against the same test in a derived table is proved with the name written
out, and was not with the bare name.
"""

from __future__ import annotations

import pytest

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"items": ["item_id", "label"], "moves": ["item_id", "moved_on", "units"]}
TYPES = {"items": {"item_id": "INT", "label": "VARCHAR"}, "moves": {"item_id": "INT", "moved_on": "DATE", "units": "INT"}}
KEYED = {"items": TableConstraints(not_null=frozenset({"item_id"}), keys=(("item_id",),)), "moves": TableConstraints()}
JOINED = "items AS i JOIN moves AS m ON i.item_id = m.item_id"
WINDOW = "MIN(m.moved_on) >= '2024-01-01' AND MAX(m.moved_on) <= '2024-03-31'"
DERIVED = (
    "SELECT i.item_id FROM items AS i JOIN (SELECT item_id FROM moves GROUP BY item_id "
    "HAVING MIN(moved_on) >= '2024-01-01' AND MAX(moved_on) <= '2024-03-31') AS m ON i.item_id = m.item_id"
)
BARE = f"SELECT i.item_id FROM {JOINED} GROUP BY item_id HAVING {WINDOW}"
WRITTEN_OUT = f"SELECT i.item_id FROM {JOINED} GROUP BY i.item_id HAVING {WINDOW}"


def prove(left: str, right: str, dialect: str = "mysql"):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, types=TYPES, constraints=KEYED, compare_names=False, dialect=dialect, exact_arithmetic=True, timeout_ms=20000
    )


def test_the_bare_name_is_proved_where_the_written_out_name_is():
    assert prove(WRITTEN_OUT, DERIVED).proven
    assert prove(BARE, DERIVED).proven
    assert prove(BARE, WRITTEN_OUT).proven


def test_without_the_early_reading_the_pair_is_left_unproved():
    with shape_fallback.without("early_group_names"):
        assert not prove(BARE, DERIVED).proven
        assert prove(BARE, WRITTEN_OUT).proven  # the fourth stage reads the name as before


@pytest.mark.parametrize(
    "other",
    [
        DERIVED.replace("<= '2024-03-31'", "<= '2024-03-30'"),  # another window
        DERIVED.replace("MIN(moved_on)", "MAX(moved_on)"),  # a move on or after the first day is enough
        f"SELECT m.item_id FROM items AS i RIGHT JOIN moves AS m ON i.item_id = m.item_id GROUP BY m.item_id HAVING {WINDOW}",  # moves of no item
    ],
)
def test_a_query_that_differs_is_not_proved(other):
    assert not prove(BARE, other).proven


def test_the_name_is_read_on_mysql_only():
    # PostgreSQL and SQLite look the name up in another order: the query is not read there
    assert not prove(BARE, DERIVED, dialect="postgres").proven
