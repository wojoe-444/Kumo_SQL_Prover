"""Derived tables that differ only in a filter must not use up the budget for proving two spellings equal.

A decorrelated report reads one grouped derived table once per quarter. With each quarter's filter moved inside, the
query holds a dozen bodies that differ in a literal; they are different relations, and proving each of them unequal to
the others used up the proofs ``_unify_opaque`` may spend before the spelling on the other side was tried.
"""

import pytest

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic

QUARTERS = (1, 2, 3)
KINDS = (("sf", "store"), ("wf", "web"))


def _body_left(table: str, name: str) -> str:
    return (
        f"select a, q, y, sum(x) as {name} from {table}, d where {table}.k = d.k and {table}.p between 1 and 5 "
        "and a in ('u', 'v') group by a, q, y"
    )


def _body_right(table: str, name: str, n: int) -> str:
    return (
        f"SELECT {table}{n}.a, d{n}.q, d{n}.y, SUM({table}{n}.x) AS {name} FROM {table} AS {table}{n}, d AS d{n} "
        f"WHERE {table}{n}.k = d{n}.k AND ({table}{n}.p >= 1 AND {table}{n}.p <= 5) AND {table}{n}.a IN ('u', 'v') "
        f"GROUP BY d{n}.y, d{n}.q, {table}{n}.a"
    )


def _pair() -> tuple[str, str]:
    ctes = ", ".join(f"{name} as ({_body_left(table, name)})" for table, name in KINDS)
    sources, conditions, outputs = [], [], []
    for table, name in KINDS:
        for q in QUARTERS:
            sources.append(f"{name} {name[0]}{q}")
            conditions.append(f"{name[0]}{q}.q = {q} and {name[0]}{q}.y = 2001")
            if q > 1:
                conditions.append(f"s1.a = {name[0]}{q}.a")
        outputs += [f"{name[0]}{q}.{name}" for q in QUARTERS]
    conditions.append("s1.a = w1.a")
    left = f"with {ctes} select s1.a, {', '.join(outputs)} from {', '.join(sources)} where {' and '.join(conditions)}"
    derived, right_conditions, right_outputs = [], [], []
    alias = {}
    for table, name in KINDS:
        for q in QUARTERS:
            n = len(alias)
            alias[f"{name[0]}{q}"] = f"t{n}"
            derived.append(f"({_body_right(table, name, n)}) AS t{n}")
    for table, name in KINDS:
        for q in QUARTERS:
            a = alias[f"{name[0]}{q}"]
            right_conditions.append(f"{a}.q = {q} AND {a}.y = 2001")
            if q > 1:
                right_conditions.append(f"{alias['s1']}.a = {a}.a")
        right_outputs += [f"{alias[f'{name[0]}{q}']}.{name}" for q in QUARTERS]
    right_conditions.append(f"{alias['s1']}.a = {alias['w1']}.a")
    right = f"SELECT {alias['s1']}.a, {', '.join(right_outputs)} FROM {', '.join(derived)} WHERE {' AND '.join(right_conditions)}"
    return left, right


SCHEMA = {"sf": ["k", "a", "p", "x"], "wf": ["k", "a", "p", "x"], "d": ["k", "q", "y"]}


@pytest.fixture(autouse=True)
def _first_attempt_only(monkeypatch):
    # a large pair gets no second attempt without the shape rules; so does this one, to read the first attempt alone
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", 0)


def test_a_dozen_bodies_that_differ_in_a_literal_still_meet_their_other_spelling():
    left, right = _pair()
    result = prove_equivalent_algebraic(left, right, schema=SCHEMA, dialect="postgres", compare_names=False)
    assert result.proven, result.reason


def test_quarters_that_are_swapped_are_still_not_proved_equal():
    left, right = _pair()
    swapped = right.replace("t0.q = 1", "t0.q = 4", 1)
    assert not prove_equivalent_algebraic(left, swapped, schema=SCHEMA, dialect="postgres", compare_names=False).proven
