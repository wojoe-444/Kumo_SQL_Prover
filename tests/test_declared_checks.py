"""The VeriEQL harness passes a case's declared value checks to the prover (``verieql_bench.declared_checks``)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from kumo_sql_prover import counterexample as cx  # noqa: E402
from kumo_sql_prover import value_checks  # noqa: E402

veri = pytest.importorskip("verieql_bench")

SCHEMA = {
    "MOVES": {"MACHINE": "INT", "STEP": "INT", "KIND": "ENUM,START,END", "AT": "NUMERIC", "DAY": "DATE"},
    "PAY": {"EMP": "INT", "AMOUNT": "INT", "POS": "INT", "FLOOR": "INT", "CEILING": "INT", "NOTE": "VARCHAR"},
}
KEYS = [
    {"primary": [{"value": "MOVES__MACHINE"}, {"value": "MOVES__STEP"}, {"value": "MOVES__KIND"}]},
    {"primary": [{"value": "PAY__EMP"}]},
]


def case(*constraints, pair=("SELECT 1", "SELECT 1")) -> dict:
    return {"index": 3, "suite": "leetcode", "schema": SCHEMA, "constraint": [*KEYS, *constraints], "pair": list(pair)}


def checks(*constraints) -> dict:
    made = case(*constraints)
    return veri.declared_checks(made, veri.build_spec(made))


KIND = {"in": [{"value": "MOVES__KIND"}, [{"literal": "START"}, {"literal": "END"}]]}
POSITIVE = {"gt": [{"value": "PAY__AMOUNT"}, 0]}
RANGE = {"between": [{"value": "PAY__POS"}, 1, 500]}
ORDERED = {"lt": [{"value": "PAY__FLOOR"}, {"value": "PAY__CEILING"}]}


def test_the_checks_the_harness_passes():
    assert checks(KIND) == {"moves": (("in", ("column", "kind"), ("values", ("START", "END"))),)}
    assert checks(RANGE) == {"pay": ((">=", ("column", "pos"), ("value", 1)), ("<=", ("column", "pos"), ("value", 500)))}
    assert checks(ORDERED) == {"pay": (("<", ("column", "floor"), ("column", "ceiling")),)}
    assert checks({"neq": [{"value": "PAY__NOTE"}, {"literal": "none"}]}) == {"pay": (("<>", ("column", "note"), ("value", "none")),)}
    assert checks({"gte": [{"value": "MOVES__AT"}, 0]}) == {"moves": ((">=", ("column", "at"), ("value", 0)),)}
    assert checks() == {}


def test_a_whole_number_above_a_bound_is_at_least_the_next_one():
    """The solver's numbers are not whole, so "above 0" alone would not give it "at least 1"."""

    assert checks(POSITIVE) == {"pay": ((">", ("column", "amount"), ("value", 0)), (">=", ("column", "amount"), ("value", 1)))}
    assert checks({"lt": [{"value": "PAY__POS"}, 10]}) == {"pay": (("<", ("column", "pos"), ("value", 10)), ("<=", ("column", "pos"), ("value", 9)))}
    # a NUMERIC column may hold 0.5: no such reading
    assert checks({"gt": [{"value": "MOVES__AT"}, 0]}) == {"moves": ((">", ("column", "at"), ("value", 0)),)}


@pytest.mark.parametrize(
    "constraint",
    [
        {"gt": [{"value": "MOVES__DAY"}, {"date": "2020-01-01"}]},  # a day
        {"gt": [{"value": "MOVES__DAY"}, 0]},  # a number against a DATE column
        {"gt": [{"value": "PAY__NOTE"}, {"literal": "m"}]},  # an ordering of texts
        {"gt": [{"value": "PAY__AMOUNT"}, {"literal": "0"}]},  # a text against a number column
        {"in": [{"value": "PAY__AMOUNT"}, [1, {"literal": "two"}]]},  # one value of another kind
        {"in": [{"value": "PAY__AMOUNT"}, []]},
        {"between": [{"value": "PAY__NOTE"}, {"literal": "a"}, {"literal": "m"}]},
        {"between": [{"value": "MOVES__DAY"}, {"date": "2020-01-01"}, {"date": "2020-12-31"}]},
        {"lt": [{"value": "PAY__EMP"}, {"value": "MOVES__MACHINE"}]},  # two tables
        {"lt": [{"value": "PAY__NOTE"}, {"value": "PAY__AMOUNT"}]},  # a text column against a number column
        {"imply": [{"gt": [{"value": "PAY__AMOUNT"}, 0]}, {"gt": [{"value": "PAY__POS"}, 0]}]},
        {"gt": [{"value": "PAY__AMOUNT"}, True]},
        {"gt": [0, {"value": "PAY__AMOUNT"}]},  # the constant first: not a form the benchmark writes
    ],
)
def test_what_the_harness_leaves_out(constraint):
    """A check that is not passed is a fact the prover does not use: always safe."""

    assert checks(constraint) == {}


def test_every_passed_check_is_one_the_prover_reads():
    passed = checks(KIND, POSITIVE, RANGE, ORDERED, {"neq": [{"value": "PAY__NOTE"}, {"literal": "none"}]})
    assert all(value_checks.columns(check) for table in passed.values() for check in table)


def test_the_generated_databases_meet_every_passed_check():
    """The harness looks for counterexamples on generated databases. Those meet the declared checks as the harness
    reads them; the facts the prover is given must hold on every one of them, or a proof would be checked against
    databases it does not speak about."""

    made = case(KIND, POSITIVE, RANGE, ORDERED)
    spec = veri.build_spec(made)
    passed = veri.declared_checks(made, spec)
    generator = cx._Generator(spec, cx.Constants(), random.Random(5))
    rows_seen = 0
    for _ in range(80):
        database = generator.database({"MOVES", "PAY"}, 4)
        for name, rows in database.items():
            names = [column.name.lower() for column in spec.tables[name].columns]
            for row in rows:
                rows_seen += 1
                assert not any(value_checks.violated(check, dict(zip(names, row))) for check in passed[name.lower()])
    assert rows_seen > 200


BY_KIND = (
    "SELECT MACHINE, SUM(CASE WHEN KIND = 'START' THEN -AT ELSE AT END) AS SPENT FROM MOVES GROUP BY MACHINE",
    "SELECT MACHINE, SUM(CASE WHEN KIND = 'END' THEN AT ELSE -AT END) AS SPENT FROM MOVES GROUP BY MACHINE",
)
BY_AMOUNT = ("SELECT EMP FROM PAY WHERE AMOUNT >= 1 OR AMOUNT IS NULL", "SELECT EMP FROM PAY")


def test_a_pair_is_proved_with_its_declared_checks_and_not_without():
    for constraint, pair in ((KIND, BY_KIND), (POSITIVE, BY_AMOUNT)):
        with_check, without = case(constraint, pair=pair), case(pair=pair)
        assert veri.prove(with_check, veri.build_spec(with_check), 3000).proven, pair
        assert not veri.prove(without, veri.build_spec(without), 3000).proven, pair


def test_the_conditional_harness_passes_the_declared_checks():
    bench = pytest.importorskip("conditional_bench")
    assert bench.decide_verieql(case(KIND, pair=BY_KIND)).kind == "equivalent"
    assert bench.decide_verieql(case(pair=BY_KIND)).kind != "equivalent"
