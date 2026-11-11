"""Division by a row count written as a choice of divisions by constants (an exact rewrite)."""

from __future__ import annotations

from fractions import Fraction

import pytest

z3 = pytest.importorskip("z3")
pytest.importorskip("duckdb")

from kumo_sql_prover import bounded_count_division as bc  # noqa: E402
from kumo_sql_prover.bounded_equivalence import (  # noqa: E402
    BColumn,
    BoundedSchema,
    BoundedStatus,
    BTable,
    check_bounded,
    prepared_solver,
)

X, Y = z3.Reals("x y")
I, J = z3.Ints("i j")
P, Q, R = z3.Bools("p q r")
COUNT = z3.Sum(z3.If(P, 1, 0), z3.If(Q, 1, 0), z3.If(R, 1, 0))


def _unsat(*formulas) -> bool:
    solver = z3.Solver()
    solver.set("timeout", 10000)
    solver.add(*formulas)
    return solver.check() == z3.unsat


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        (z3.IntVal(4), {4}),
        (z3.RealVal("1/3"), {Fraction(1, 3)}),
        (COUNT, {0, 1, 2, 3}),
        (z3.ToReal(COUNT), {0, 1, 2, 3}),
        (z3.If(COUNT == 0, z3.RealVal(1), z3.ToReal(COUNT)), {0, 1, 2, 3}),  # read from the shape, not the condition
        (2 * COUNT + 1, {1, 3, 5, 7}),
        (z3.If(P, 5, 7) - z3.If(Q, 1, 0), None),  # a subtraction is not one of the shapes read
        (I, None),
        (COUNT + I, None),
        (z3.If(P, I, 1), None),
        (z3.Sum(*[z3.If(z3.Bool(f"b{k}"), 1, 0) for k in range(bc.MAX_VALUES)]), None),  # 17 values: too many
    ],
)
def test_the_values_a_term_can_take_are_read_from_its_shape(term, expected):
    got = bc.values(term)
    assert got == (None if expected is None else frozenset(Fraction(v) for v in expected))


@pytest.mark.parametrize(
    "formula",
    [
        X / z3.ToReal(COUNT) > Y,
        X / z3.If(COUNT == 0, z3.RealVal(1), z3.ToReal(COUNT)) > Y,
        I / COUNT == J,
        I % COUNT == J,
        z3.And(X / z3.ToReal(COUNT) > 1, (X * 100) / z3.ToReal(COUNT) < Y),
        (X / z3.ToReal(COUNT)) / z3.ToReal(2 * COUNT + 1) > Y,  # a division inside a division
    ],
)
def test_the_rewrite_has_exactly_the_models_of_the_original(formula):
    (out,), replaced = bc.split([formula])
    assert replaced >= 1 and not out.eq(formula)
    assert _unsat(out != formula)


@pytest.mark.parametrize("formula", [X / Y > 1, X / 3 > Y, I / J == 2, I % 5 == J, X / z3.ToReal(COUNT + I) > 1, X * z3.ToReal(COUNT) > 1])
def test_other_divisions_are_left_alone(formula):
    out, replaced = bc.split([formula])
    assert replaced == 0 and out[0].eq(formula)


def test_no_division_by_an_unknown_is_left_when_the_divisor_is_a_count():
    (out,), _ = bc.split([X / z3.If(COUNT == 0, z3.RealVal(1), z3.ToReal(COUNT)) > Y])

    def divisions(term):
        found = [term] if term.decl().kind() == z3.Z3_OP_DIV and not z3.is_rational_value(term.arg(1)) else []
        return found + [d for child in term.children() for d in divisions(child)]

    assert divisions(out) == []


def test_an_integer_divisor_keeps_integer_division():
    (out,), _ = bc.split([I / COUNT == J])
    # 7 / 2 is 3 in integers: a real division slipped in would make this satisfiable with j = 3.5 or unsatisfiable
    assert _unsat(out, I == 7, P, Q, z3.Not(R), J != 3)
    assert not _unsat(out, I == 7, P, Q, z3.Not(R), J == 3)


# --- in the checker ---------------------------------------------------------------------------------


def schema() -> BoundedSchema:
    queries = BTable("queries", [BColumn("name", "STRING"), BColumn("rating", "INT64"), BColumn("position", "INT64")])
    return BoundedSchema({"queries": queries})


def check(left: str, right: str, **kwargs):
    return check_bounded(left, right, schema(), rows=3, dialect="mysql", timeout_ms=10000, **kwargs)


EQUAL = [
    (
        "select name, round(100 * sum(case when rating < 3 then 1 else 0 end) / count(1), 2) from queries group by name",
        "select name, round((sum(case when rating < 3 then 1 else 0 end) / count(1)) * 100, 2) from queries group by name",
    ),
    ("select name, avg(rating) from queries group by name", "select name, sum(rating) / count(rating) from queries group by name"),
    (
        "select name, round(avg(rating / position), 2) from queries group by name",
        "select name, round(sum(rating / position) / count(rating / position), 2) from queries group by 1",
    ),
]

DIFFERENT = [
    # COUNT(*) also counts the rows whose rating is NULL
    ("select name, avg(rating) from queries group by name", "select name, sum(rating) / count(*) from queries group by name"),
    ("select name, sum(rating) / count(*) from queries group by name", "select name, sum(rating) / (count(*) + 1) from queries group by name"),
    ("select name, sum(rating) % count(*) from queries group by name", "select name, sum(rating) % 2 from queries group by name"),
]


@pytest.mark.parametrize(("left", "right"), EQUAL)
def test_equal_answers_that_divide_by_a_count_are_bounded_equivalent(left, right):
    assert check(left, right).status is BoundedStatus.BOUNDED_EQUIVALENT


@pytest.mark.parametrize(("left", "right"), DIFFERENT)
def test_a_different_count_is_a_replayed_counterexample(left, right):
    result = check(left, right)
    assert result.status is BoundedStatus.DIFFERENT and result.counterexample is not None


# --- the solver the checker asks ----------------------------------------------------------------------


def test_the_prepared_solver_answers_and_gives_models():
    solver = prepared_solver(5000)
    solver.add(X == Y + 1, z3.If(P, X, Y) > 3, Y < 10)
    assert solver.check() == z3.sat
    model = solver.model()
    assert z3.is_true(model.eval(z3.And(X == Y + 1, z3.If(P, X, Y) > 3, Y < 10), model_completion=True))
    solver = prepared_solver(5000)
    solver.add(X == Y + 1, X < Y)
    assert solver.check() == z3.unsat


def test_the_prepared_solver_stops_at_its_limit():
    solver = prepared_solver(50)
    a, b, c = z3.Ints("a b c")
    solver.add(a * a * a + b * b * b + c * c * c == 42, a > 1000, b < -1000)
    assert solver.check() == z3.unknown
