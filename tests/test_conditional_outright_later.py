"""A proof under conditions does not hide an outright proof of a later step (``prove_equivalent_algebraic``).

The staged proof ends at the first step that proves. With ``conditional=True`` every step also searches for
conditions, so a step that finds "equivalent whenever .." ended the proof before a later step could prove the pair
outright: the pivot below was reported as equal only when ``item_id`` alone is unique, although its declared key
(``item_id``, ``depot``) already makes it equal, which the plain proof says. Found on the conditional VeriEQL sample of
2026-11-25: 11 of its 3,000 cases had a condition the prover itself did not need. Now a proof under conditions is
followed by the plain staged proof, and an outright proof wins.

The tables are made up.
"""

import sqlite3

import pytest

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import _outright_proof_may_follow, prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtEquivalenceResult, SmtStatus, TableConstraints

SCHEMA = {"stock": ["item_id", "depot", "cost"]}
TYPES = {"stock": {"item_id": "INT", "depot": "VARCHAR", "cost": "INT"}}
NOT_NULL = frozenset({"item_id", "depot"})
KEYED = {"stock": TableConstraints(not_null=NOT_NULL, keys=(("item_id", "depot"),))}
UNKEYED = {"stock": TableConstraints(not_null=NOT_NULL)}
ADDED = (
    "SELECT item_id, SUM(CASE WHEN depot = 'north' THEN cost ELSE NULL END) AS north, "
    "SUM(CASE WHEN depot = 'south' THEN cost ELSE NULL END) AS south FROM stock GROUP BY 1"
)
LARGEST = (
    "SELECT item_id, MAX(CASE WHEN depot = 'north' THEN cost END) AS north, "
    "MAX(CASE WHEN depot = 'south' THEN cost END) AS south FROM stock GROUP BY item_id"
)


def prove(constraints, **options):
    return prove_equivalent_algebraic(ADDED, LARGEST, schema=SCHEMA, types=TYPES, constraints=constraints, dialect="mysql", compare_names=False, **options)


def test_the_plain_proof_is_outright_under_the_declared_key():
    assert prove(KEYED).status is SmtStatus.PROVEN_EQUIVALENT


def test_the_search_for_conditions_gives_the_same_outright_proof():
    result = prove(KEYED, conditional=True)
    assert result.status is SmtStatus.PROVEN_EQUIVALENT
    assert not result.conditions


def test_without_the_later_step_the_proof_under_conditions_is_kept():
    with shape_fallback.without("single_row_aggregates"):
        result = prove(KEYED, conditional=True)
    assert result.status is SmtStatus.PROVEN_CONDITIONALLY
    assert [(c.kind, c.table, tuple(c.columns)) for c in result.conditions] == [("unique", "stock", ("item_id",))]


def test_without_the_key_the_pair_is_not_equal_and_stays_conditional():
    database = sqlite3.connect(":memory:")
    database.execute("CREATE TABLE stock (item_id INT NOT NULL, depot TEXT NOT NULL, cost INT)")
    database.executemany("INSERT INTO stock VALUES (?, ?, ?)", [(1, "north", 5), (1, "north", 7), (1, "south", 2)])
    assert database.execute(ADDED).fetchall() == [(1, 12, 2)]
    assert database.execute(LARGEST).fetchall() == [(1, 7, 2)]
    assert not prove(UNKEYED).proven
    result = prove(UNKEYED, conditional=True)
    assert result.status is SmtStatus.PROVEN_CONDITIONALLY
    assert [(c.kind, c.table, tuple(c.columns)) for c in result.conditions] == [("unique", "stock", ("item_id",))]


class _Fact:
    def __init__(self, kind):
        self.kind = kind


def test_only_a_proof_under_conditions_of_the_search_is_followed():
    from kumo_sql_prover import statement_values

    found = SmtEquivalenceResult(SmtStatus.PROVEN_CONDITIONALLY, "equivalent whenever ..", conditions=(_Fact("unique"),))
    assert _outright_proof_may_follow(found, ADDED, LARGEST, {"conditional": True})
    assert not _outright_proof_may_follow(found, ADDED, LARGEST, {})  # nobody asked for conditions
    assert not _outright_proof_may_follow(SmtEquivalenceResult(SmtStatus.NOT_PROVEN, "no"), ADDED, LARGEST, {"conditional": True})
    assert not _outright_proof_may_follow(SmtEquivalenceResult(SmtStatus.PROVEN_EQUIVALENT, "yes"), ADDED, LARGEST, {"conditional": True})
    clock = SmtEquivalenceResult(SmtStatus.PROVEN_CONDITIONALLY, "one clock", conditions=tuple(_Fact(kind) for kind in statement_values.KINDS))
    assert not _outright_proof_may_follow(clock, ADDED, LARGEST, {"conditional": True})  # the plain proof says the same


def test_a_pair_over_the_size_limit_of_the_last_steps_is_not_proved_again():
    found = SmtEquivalenceResult(SmtStatus.PROVEN_CONDITIONALLY, "equivalent whenever ..", conditions=(_Fact("unique"),))
    long = ADDED + " /* " + "x" * shape_fallback.MAX_CHARS + " */"
    assert not shape_fallback.last_steps_worth_trying(long, LARGEST)
    assert not _outright_proof_may_follow(found, long, LARGEST, {"conditional": True})
