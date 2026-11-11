"""The condition search also tries the NOT NULL candidates alone (``conditional_equivalence.add_conditions``).

More facts are not always more proofs: a candidate key on ``team`` lets a rule drop the ``GROUP BY team`` of one
side, which takes that side out of the shape the proof needs. The pair below is settled by one NOT NULL fact, and
the search used to give it up because it does not prove under every candidate together.
"""

import pytest

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.conditional_equivalence import candidate_conditions, with_conditions

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
    result = prove(INNER, LEFT, conditional=True)
    assert result.status.value == "proven_conditionally"
    assert [(c.kind, c.table, tuple(c.columns)) for c in result.conditions] == [("not_null", "staff", ("team",))]


def test_a_pair_that_differs_under_the_not_null_facts_stays_unproved():
    assert prove(INNER, LEFT.replace("m.best > 100", "m.best > 101"), conditional=True).status.value != "proven_conditionally"
    # a filter inside the groups: a row's own group can be missing, whatever is NOT NULL
    filtered = LEFT.replace("FROM staff GROUP BY team", "FROM staff WHERE pay > 0 GROUP BY team")
    assert prove(INNER, filtered, conditional=True).status.value != "proven_conditionally"
