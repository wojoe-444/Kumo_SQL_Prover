"""The bounded checker's first, looser question: hard arithmetic left uninterpreted."""

from __future__ import annotations

import random

import pytest

z3 = pytest.importorskip("z3")
pytest.importorskip("duckdb")

from kumo_sql_prover import bounded_abstraction as ba  # noqa: E402
from kumo_sql_prover.bounded_equivalence import BColumn, BoundedSchema, BoundedStatus, BTable, check_bounded  # noqa: E402

X, Y, Z = z3.Reals("x y z")
I, J = z3.Ints("i j")


def _unsat(*formulas) -> bool:
    solver = z3.Solver()
    solver.set("timeout", 5000)
    solver.add(*formulas)
    return solver.check() == z3.unsat


@pytest.mark.parametrize(
    ("formula", "count"),
    [
        (X * Y > 1, 1),
        (X * Y * Z > 1, 2),  # (x * y) * z: two products
        (X / Y > 1, 1),
        (I / J > 1, 1),
        (I % J == 1, 1),
        (z3.ToInt(X) == I, 1),
        (z3.And(X * Y > 1, z3.ToInt(X / Y) == I), 3),
    ],
)
def test_hard_operations_become_uninterpreted(formula, count):
    (loose,), replaced = ba.abstract([formula])
    assert replaced == count
    assert not loose.eq(formula)


@pytest.mark.parametrize("formula", [X * 3 > Y, X / 2 > Y, I % 7 == 1, I / 2 == J, X + Y > Z, z3.If(X > Y, X, Y) == Z, z3.ToReal(I) > X])
def test_linear_arithmetic_is_left_alone(formula):
    out, replaced = ba.abstract([formula])
    assert replaced == 0 and out[0].eq(formula)


def test_equal_operands_give_equal_results_without_arithmetic():
    # the same product, quotient and rounding on both sides: no difference whatever the functions mean
    for left, right in [(X * Y, Y * X), (X / Y, X / Y), (z3.ToReal(z3.ToInt(X * Y / Z)), z3.ToReal(z3.ToInt(Y * X / Z)))]:
        loose, replaced = ba.abstract([left != right])
        assert replaced and _unsat(*loose)


def test_one_set_of_functions_serves_every_formula():
    loose, _ = ba.abstract([X * Y == 6, Y * X == 7])
    assert _unsat(*loose)


def test_a_fact_of_arithmetic_is_lost_and_that_is_the_point():
    # distributivity holds of numbers, not of an unknown function: the looser question has a model here
    exact = X * (Y + 1) != X * Y + X
    assert _unsat(exact)
    loose, _ = ba.abstract([exact])
    assert not _unsat(*loose)


def test_unsatisfiable_carries_over_to_the_exact_formula():
    # whenever the looser formula has no model, neither has the exact one (the real operations are one meaning)
    rng = random.Random(11)
    terms = [X, Y, Z, z3.ToReal(I), z3.RealVal(2), z3.RealVal(0)]

    def term(depth: int):
        if depth == 0 or rng.random() < 0.3:
            return rng.choice(terms)
        a, b = term(depth - 1), term(depth - 1)
        return rng.choice([a + b, a - b, a * b, a / b, z3.ToReal(z3.ToInt(a)), z3.If(a > b, a, b)])

    carried = 0
    for _ in range(150):
        a, b = term(3), term(3)
        formula = rng.choice([a != a, a * b != b * a, z3.And(a == b, a + 1 == b), a > b, z3.And(a > b, b > a), a / b != a / b])
        loose, _ = ba.abstract([formula])
        if _unsat(*loose):
            carried += 1
            solver = z3.Solver()
            solver.set("timeout", 5000)
            solver.add(formula)
            assert solver.check() != z3.sat, formula
    assert carried > 20


def test_deep_terms_do_not_exhaust_the_stack():
    total = X
    for _ in range(3000):
        total = z3.If(total > 0, total * Y, total)
    loose, replaced = ba.abstract([total > 1])
    assert replaced == 3000 and len(loose) == 1


# --- in the checker ---------------------------------------------------------------------------------


def schema() -> BoundedSchema:
    prices = BTable("prices", [BColumn("pid", "INT64", True), BColumn("price", "INT64"), BColumn("units", "INT64")])
    return BoundedSchema({"prices": prices})


def check(left: str, right: str, **kwargs):
    return check_bounded(left, right, schema(), rows=3, dialect="mysql", **kwargs)


EQUAL = [
    (
        "select pid, round(sum(price * units) / sum(units), 2) as p from prices group by pid",
        "select pid, round(sum(units * price) / sum(units), 2) as p from (select * from prices) s group by pid",
    ),
    ("select pid, round(avg(price), 0) from prices group by pid", "select pid, round(sum(price) / count(price), 0) from prices group by pid"),
    # equal only by arithmetic: the looser question finds a model, the replay rejects it, the exact question decides
    ("select price * (units + 1) from prices", "select price * units + price from prices"),
    ("select price * 2 from prices", "select price + price from prices"),
]

DIFFERENT = [
    ("select price * units from prices", "select price + units from prices"),
    ("select pid, round(sum(price * units) / sum(units), 2) from prices group by pid", "select pid, round(avg(price), 2) from prices group by pid"),
    ("select round(price / 3, 0) from prices", "select round(price / 3, 1) from prices"),
    ("select price / units from prices", "select units / price from prices"),
]


@pytest.mark.parametrize(("left", "right"), EQUAL)
def test_equal_pairs_are_bounded_equivalent_with_and_without_the_looser_question(left, right):
    for abstract in (True, False):
        assert check(left, right, abstract_arithmetic=abstract).status is BoundedStatus.BOUNDED_EQUIVALENT


@pytest.mark.parametrize(("left", "right"), DIFFERENT)
def test_a_difference_is_reported_only_with_a_database_the_replay_confirms(left, right):
    for abstract in (True, False):
        result = check(left, right, abstract_arithmetic=abstract)
        assert result.status is BoundedStatus.DIFFERENT and result.counterexample is not None
