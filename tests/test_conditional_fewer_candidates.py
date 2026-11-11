"""The condition search also tries fewer candidates than all of them (``conditional_equivalence.add_conditions``).

More facts are not always more proofs. A candidate key on ``team`` lets a rule drop the ``GROUP BY team`` of one
side, which takes that side out of the shape the proof needs; a candidate key on ``email`` makes a query that looks
for duplicate emails empty while the other side is not recognised as empty. The search used to give such a pair up
because it does not prove under every candidate together. It now also tries the NOT NULL candidates alone, and
everything but one unique key.
"""

import pytest

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.conditional_equivalence import add_conditions, candidate_conditions, with_conditions
from kumo_sql_prover.smt_equivalence import SmtEquivalenceResult, SmtStatus

SCHEMA = {"staff": ["team", "emp", "pay"]}
GROUPS = "(SELECT team, MAX(pay) AS best FROM staff GROUP BY team) AS m"
INNER = f"SELECT s.team, s.emp, CASE WHEN m.best > 100 THEN pay * 2 ELSE pay END AS pay FROM staff AS s INNER JOIN {GROUPS} ON s.team = m.team"
LEFT = f"SELECT s.team, s.emp, CASE WHEN m.best > 100 THEN s.pay * 2 ELSE s.pay END AS pay FROM staff AS s LEFT JOIN {GROUPS} ON s.team = m.team"


def prove(left, right, **options):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, dialect="mysql", compare_names=False, **options)


def test_the_pair_is_lost_under_every_candidate_and_proved_under_the_not_null_ones():
    candidates = candidate_conditions(INNER, LEFT, schema=SCHEMA, dialect="mysql")
    assert {c.kind for c in candidates} == {"unique", "not_null"}
    assert not prove(INNER, LEFT, constraints=with_conditions(None, candidates)).proven  # the case the search has to survive
    assert prove(INNER, LEFT, constraints=with_conditions(None, [c for c in candidates if c.kind == "not_null"])).proven


def test_the_verdict_names_the_one_not_null_fact():
    # this test and the next are in the slow tier (tests/order.json): each search runs the prover several times
    result = prove(INNER, LEFT, conditional=True)
    assert result.status.value == "proven_conditionally"
    assert [(c.kind, c.table, tuple(c.columns)) for c in result.conditions] == [("not_null", "staff", ("team",))]


def test_a_pair_that_differs_under_the_not_null_facts_stays_unproved():
    assert prove(INNER, LEFT.replace("m.best > 100", "m.best > 101"), conditional=True).status.value != "proven_conditionally"
    # a filter inside the groups: a row's own group can be missing, whatever is NOT NULL
    filtered = LEFT.replace("FROM staff GROUP BY team", "FROM staff WHERE pay > 0 GROUP BY team")
    assert prove(INNER, filtered, conditional=True).status.value != "proven_conditionally"


PEOPLE = {"person": ["id", "email"]}
COUNTED = "SELECT email FROM person GROUP BY email HAVING COUNT(email) > 1"
PAIRED = "SELECT DISTINCT a.email FROM person AS a JOIN person AS b ON a.email = b.email WHERE a.id > b.id"


def test_an_ordered_pair_of_rows_of_one_table_makes_its_column_a_key_candidate():
    found = {(c.kind, tuple(c.columns)) for c in candidate_conditions(COUNTED, PAIRED, schema=PEOPLE, dialect="mysql")}
    assert ("unique", ("id",)) in found  # a.id > b.id asks for two different rows, as a.id <> b.id does
    # between two tables, or two columns, it is only a comparison
    other = "SELECT a.email FROM person AS a JOIN visits AS v ON a.id > v.id"
    assert ("unique", ("id",)) not in {(c.kind, tuple(c.columns)) for c in candidate_conditions(other, other, schema={**PEOPLE, "visits": ["id"]}, dialect="mysql")}


def test_a_key_candidate_that_spoils_the_proof_is_left_out():
    # a prover stand-in: the pair is proved under the id key, and lost as soon as email is a key too
    asked: list[frozenset] = []

    def prover(constraints, pair=None):
        if pair is not None:
            return SmtEquivalenceResult(SmtStatus.NOT_PROVEN, "not shown empty")
        table = constraints.get("person")
        keys = set(table.keys) if table else set()
        asked.append(frozenset(keys))
        fine = table is not None and ("id",) in keys and ("email",) not in keys and "id" in table.not_null
        return SmtEquivalenceResult(SmtStatus.PROVEN_EQUIVALENT if fine else SmtStatus.NOT_PROVEN, "stand-in")

    start = SmtEquivalenceResult(SmtStatus.NOT_PROVEN, "not proven")
    result = add_conditions(COUNTED, PAIRED, start, prover, schema=PEOPLE, dialect="mysql")
    assert result.status is SmtStatus.PROVEN_CONDITIONALLY
    assert sorted((c.kind, tuple(c.columns)) for c in result.conditions) == [("not_null", ("id",)), ("unique", ("id",))]
    assert frozenset({("id",), ("email",)}) in asked  # everything together was tried first

    # a prover that any key spoils: the NOT NULL facts alone are tried
    def shy(constraints, pair=None):
        table = constraints.get("person")
        fine = pair is None and table is not None and not table.keys and "email" in table.not_null
        return SmtEquivalenceResult(SmtStatus.PROVEN_EQUIVALENT if fine else SmtStatus.NOT_PROVEN, "stand-in")

    result = add_conditions(COUNTED, PAIRED, start, shy, schema=PEOPLE, dialect="mysql")
    assert result.status is SmtStatus.PROVEN_CONDITIONALLY
    assert [(c.kind, tuple(c.columns)) for c in result.conditions] == [("not_null", ("email",))]

    # a prover that proves nothing under any set leaves the answer as it was
    nothing = lambda constraints, pair=None: SmtEquivalenceResult(SmtStatus.NOT_PROVEN, "stand-in")  # noqa: E731
    assert add_conditions(COUNTED, PAIRED, start, nothing, schema=PEOPLE, dialect="mysql") is start
