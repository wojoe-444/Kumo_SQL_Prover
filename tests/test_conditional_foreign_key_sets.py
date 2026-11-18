"""The condition search tries the foreign keys without the unique keys when every candidate together proves nothing.

"Items whose sales all fall in the quarter" equals "items with no sale outside the quarter" when every item has a
sale (a foreign key from the items to the sales). The candidates also hold "an item is sold once" (a unique key on
the sales' item column), and with it declared the first query is rewritten out of the shape the proof needs. The
sets tried before held the foreign keys only beside every unique key, so the pair was left unknown.
"""

import pytest

z3 = pytest.importorskip("z3")

from kumo_sql_prover import conditional_equivalence  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import SmtStatus  # noqa: E402

SCHEMA = {"item": ["item_id", "label"], "sale": ["item_id", "sold_on", "buyer"]}
TYPES = {"sale": {"sold_on": "DATE"}}
ALL_INSIDE = "SELECT p.item_id, p.label FROM item p JOIN (SELECT item_id FROM sale GROUP BY item_id HAVING MIN(sold_on) >= '2019-01-01' AND MAX(sold_on) <= '2019-03-31') s ON p.item_id = s.item_id"
NONE_OUTSIDE = "SELECT item_id, label FROM item WHERE item_id NOT IN (SELECT item_id FROM sale WHERE sold_on < '2019-01-01' OR sold_on > '2019-03-31')"
EVERY_ITEM_SOLD = "item(item_id) references sale(item_id)"


def decide(left: str, right: str):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="mysql", conditional=True, conditional_seconds=120)


def test_the_pair_is_proved_under_the_foreign_key_and_no_unique_key():
    result = decide(ALL_INSIDE, NONE_OUTSIDE)
    assert result.status is SmtStatus.PROVEN_CONDITIONALLY, result.reason
    named = {c.text for c in result.conditions}
    assert EVERY_ITEM_SOLD in named
    assert named == {EVERY_ITEM_SOLD, "item.item_id is NOT NULL", "sale.item_id is NOT NULL", "sale.sold_on is NOT NULL"}
    # each one is needed: an item with no sale, an item with no id, a sale of no item, a sale with no date
    for dropped in result.conditions:
        rest = conditional_equivalence.with_conditions(None, [c for c in result.conditions if c is not dropped])
        assert not prove_equivalent_algebraic(ALL_INSIDE, NONE_OUTSIDE, schema=SCHEMA, types=TYPES, constraints=rest, compare_names=False, dialect="mysql").proven, dropped.text


def test_with_duplicates_removed_on_one_side_the_items_key_is_named_too():
    result = decide(ALL_INSIDE, NONE_OUTSIDE.replace("SELECT item_id, label", "SELECT DISTINCT item_id, label"))
    assert result.status is SmtStatus.PROVEN_CONDITIONALLY, result.reason
    named = {c.text for c in result.conditions}
    assert EVERY_ITEM_SOLD in named and "(item_id) is unique in item" in named
    assert "(item_id) is unique in sale" not in named


def test_a_pair_the_other_sets_prove_keeps_its_answer():
    # proved by the NOT NULL facts alone: the foreign-key sets are never reached
    listed = "SELECT item_id FROM item WHERE item_id NOT IN (SELECT item_id FROM sale)"
    asked = "SELECT item_id FROM item AS p WHERE NOT EXISTS (SELECT 1 FROM sale AS s WHERE s.item_id = p.item_id)"
    result = decide(listed, asked)
    assert result.status is SmtStatus.PROVEN_CONDITIONALLY, result.reason
    assert all(c.kind == "not_null" for c in result.conditions)


def test_a_pair_that_differs_under_every_candidate_stays_unproved():
    result = decide(ALL_INSIDE, NONE_OUTSIDE.replace("sold_on > '2019-03-31'", "sold_on > '2019-04-30'"))
    assert result.status not in (SmtStatus.PROVEN_EQUIVALENT, SmtStatus.PROVEN_CONDITIONALLY)
