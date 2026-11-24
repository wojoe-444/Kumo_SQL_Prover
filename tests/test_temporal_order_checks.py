"""The VeriEQL harness passes "one date before another" only to a pair that just compares the dates
(``verieql_bench.declared_checks`` with ``bench_order_only.compared_only``)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

veri = pytest.importorskip("verieql_bench")

# hand-written cases through the harness's reader of declared checks; the family's floors stay in tests/test_verieql_benchmarks.py
pytestmark = pytest.mark.canary

SCHEMA = {"STAYS": {"GUEST": "INT", "ROOM": "INT", "ARRIVAL": "DATE", "DEPARTURE": "DATE", "PAID": "INT", "DUE": "INT", "BOOKED": "DATETIME"}}
BEFORE = {"lt": [{"value": "STAYS__ARRIVAL"}, {"value": "STAYS__DEPARTURE"}]}
PAID_UP = {"lte": [{"value": "STAYS__PAID"}, {"value": "STAYS__DUE"}]}
PASSED = ("<", ("column", "arrival"), ("column", "departure"))

HEAD = "SELECT DISTINCT A.GUEST FROM STAYS AS A JOIN STAYS AS B ON A.GUEST = B.GUEST AND A.ROOM <> B.ROOM"
STARTS_INSIDE = f"{HEAD} AND B.ARRIVAL BETWEEN A.ARRIVAL AND A.DEPARTURE"
OVERLAPS = f"{HEAD} AND A.ARRIVAL <= B.DEPARTURE AND B.ARRIVAL <= A.DEPARTURE"
NIGHTS = f"{HEAD} AND A.ARRIVAL <= B.DEPARTURE AND B.ARRIVAL <= A.DEPARTURE AND DATEDIFF(A.DEPARTURE, A.ARRIVAL) > 0"
ALL = "SELECT GUEST FROM STAYS"
IN_ORDER = "SELECT GUEST FROM STAYS WHERE ARRIVAL < DEPARTURE OR ARRIVAL IS NULL OR DEPARTURE IS NULL"


def case(*constraints, pair=(ALL, ALL)) -> dict:
    return {"index": 3, "suite": "leetcode", "schema": SCHEMA, "constraint": list(constraints), "pair": list(pair)}


def checks(pair, *constraints) -> dict:
    made = case(*constraints, pair=pair or (ALL, ALL))
    return veri.declared_checks(made, veri.build_spec(made), pair)


def test_an_ordering_of_dates_is_passed_to_a_pair_that_only_compares_them():
    assert checks((STARTS_INSIDE, OVERLAPS), BEFORE) == {"stays": (PASSED,)}
    assert checks((ALL, IN_ORDER), BEFORE) == {"stays": (PASSED,)}
    assert checks((ALL, ALL), BEFORE) == {"stays": (PASSED,)}
    # no pair given, or a pair that does anything else with a date
    assert checks(None, BEFORE) == {}
    assert checks((STARTS_INSIDE, NIGHTS), BEFORE) == {}
    assert checks((NIGHTS, OVERLAPS), BEFORE) == {}
    assert checks((ALL, "SELECT GUEST FROM STAYS WHERE ARRIVAL < '2024-03-01'"), BEFORE) == {}
    assert checks((ALL, "SELECT GUEST FROM STAYS WHERE DEPARTURE = ARRIVAL + 1"), BEFORE) == {}


def test_other_checks_do_not_depend_on_the_pair():
    numbers = ("<=", ("column", "paid"), ("column", "due"))
    assert checks(None, PAID_UP) == checks((ALL, NIGHTS), PAID_UP) == {"stays": (numbers,)}
    assert checks((STARTS_INSIDE, NIGHTS), BEFORE, PAID_UP) == {"stays": (numbers,)}
    assert checks((STARTS_INSIDE, OVERLAPS), BEFORE, PAID_UP) == {"stays": (PASSED, numbers)}


def test_what_is_not_an_ordering_of_two_dates_of_one_type():
    pair = (STARTS_INSIDE, OVERLAPS)
    assert checks(pair, {"lt": [{"value": "STAYS__BOOKED"}, {"value": "STAYS__ARRIVAL"}]}) == {}  # a date and time against a date
    assert checks(pair, {"lt": [{"value": "STAYS__ARRIVAL"}, {"value": "STAYS__DUE"}]}) == {}  # a date against a number
    assert checks(pair, {"neq": [{"value": "STAYS__ARRIVAL"}, {"value": "STAYS__DEPARTURE"}]}) == {}
    assert checks(pair, {"lt": [{"value": "STAYS__ARRIVAL"}, {"date": "2024-03-01"}]}) == {}  # against a constant
    for op, sign in (("lte", "<="), ("gt", ">"), ("gte", ">=")):
        assert checks(pair, {op: [{"value": "STAYS__ARRIVAL"}, {"value": "STAYS__DEPARTURE"}]}) == {"stays": ((sign, ("column", "arrival"), ("column", "departure")),)}


def test_a_pair_is_proved_with_the_ordering_and_not_without():
    with_check, without = case(BEFORE, pair=(ALL, IN_ORDER)), case(pair=(ALL, IN_ORDER))
    assert veri.prove(with_check, veri.build_spec(with_check), 3000).proven
    assert not veri.prove(without, veri.build_spec(without), 3000).proven
    # the same filter beside a use of the dates that is not a comparison: the ordering is not passed, nothing is proved
    nights = "SELECT GUEST FROM STAYS WHERE DATEDIFF(DEPARTURE, ARRIVAL) > -1000"
    other = case(BEFORE, pair=(nights, nights.replace("FROM STAYS WHERE", "FROM STAYS WHERE (ARRIVAL < DEPARTURE OR ARRIVAL IS NULL OR DEPARTURE IS NULL) AND")))
    assert not veri.prove(other, veri.build_spec(other), 3000).proven


def test_overlapping_stays_are_proved_with_the_ordering_and_not_without():
    # the proof also needs a departure that is never NULL (the conditional harness finds that condition by itself)
    filled = {"not_null": {"value": "STAYS__DEPARTURE"}}
    with_check, without = case(BEFORE, filled, pair=(STARTS_INSIDE, OVERLAPS)), case(filled, pair=(STARTS_INSIDE, OVERLAPS))
    assert veri.prove(with_check, veri.build_spec(with_check), 3000).proven
    assert not veri.prove(without, veri.build_spec(without), 800).proven
