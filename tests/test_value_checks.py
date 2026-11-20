"""Declared value checks of a table as facts for the solver (``value_checks``, ``TableConstraints.checks``)."""

from fractions import Fraction

import duckdb
import pytest
import z3

from kumo_sql_prover import value_checks
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.conditional_equivalence import Condition, with_conditions
from kumo_sql_prover.smt_equivalence import SmtStatus, TableConstraints, _Val, _value_sort

SCHEMA = {"moves": ["machine", "step", "kind", "at"], "pay": ["emp", "amount", "pos", "floor", "ceiling"]}
TYPES = {
    "moves": {"machine": "INT", "step": "INT", "kind": "VARCHAR", "at": "INT"},
    "pay": {"emp": "INT", "amount": "INT", "pos": "INT", "floor": "INT", "ceiling": "INT"},
}
KIND = ("in", ("column", "kind"), ("values", ("start", "end")))
POSITIVE = (">", ("column", "amount"), ("value", 0))
FROM_ONE = (">=", ("column", "pos"), ("value", 1))
TO_500 = ("<=", ("column", "pos"), ("value", 500))
ORDERED = ("<", ("column", "floor"), ("column", "ceiling"))


def facts(checks: bool = True, kind_not_null: bool = True, pay_not_null: bool = True) -> dict:
    return {
        "moves": TableConstraints(
            not_null=frozenset({"machine", "step"} | ({"kind"} if kind_not_null else set())),
            keys=(("machine", "step"),),
            checks=(KIND,) if checks else (),
        ),
        "pay": TableConstraints(
            not_null=frozenset({"emp"} | ({"amount", "pos", "floor", "ceiling"} if pay_not_null else set())),
            keys=(("emp",),),
            checks=(POSITIVE, FROM_ONE, TO_500, ORDERED) if checks else (),
        ),
    }


def proved(left: str, right: str, **given) -> bool:
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, constraints=facts(**given), dialect="mysql", compare_names=False, exact_arithmetic=True, timeout_ms=5000).proven


# (left, right): equal on every database the checks admit, and different on one they rule out.
BY_CHECKS = [
    (
        "SELECT machine, SUM(CASE WHEN kind = 'start' THEN -at ELSE at END) AS t FROM moves GROUP BY machine",
        "SELECT machine, SUM(CASE WHEN kind = 'end' THEN at ELSE -at END) AS t FROM moves GROUP BY machine",
    ),
    ("SELECT machine FROM moves WHERE kind <> 'start'", "SELECT machine FROM moves WHERE kind = 'end'"),
    ("SELECT machine FROM moves WHERE kind IN ('start', 'end')", "SELECT machine FROM moves"),
    ("SELECT machine, IF(kind = 'start', 1, 2) AS n FROM moves", "SELECT machine, CASE kind WHEN 'start' THEN 1 WHEN 'end' THEN 2 END AS n FROM moves"),
    ("SELECT emp FROM pay WHERE amount > 0", "SELECT emp FROM pay"),
    ("SELECT emp, ABS(amount) AS a FROM pay", "SELECT emp, amount AS a FROM pay"),
    ("SELECT emp FROM pay WHERE pos BETWEEN 1 AND 500", "SELECT emp FROM pay"),
    ("SELECT emp FROM pay WHERE pos < 501", "SELECT emp FROM pay WHERE pos > 0"),
    ("SELECT emp FROM pay WHERE floor < ceiling", "SELECT emp FROM pay"),
    ("SELECT emp, CASE WHEN ceiling > floor THEN 'up' ELSE 'flat' END AS way FROM pay", "SELECT emp, 'up' AS way FROM pay"),
]


@pytest.mark.parametrize("left, right", BY_CHECKS)
def test_a_pair_equal_under_the_checks_is_proved_only_with_them(left, right):
    assert proved(left, right)
    assert not proved(left, right, checks=False)


@pytest.mark.parametrize(
    "left, right",
    [
        ("SELECT machine FROM moves WHERE kind <> 'start'", "SELECT machine FROM moves WHERE kind = 'stop'"),  # not a listed value
        ("SELECT machine FROM moves WHERE kind = 'start'", "SELECT machine FROM moves"),
        ("SELECT emp FROM pay WHERE amount > 1", "SELECT emp FROM pay"),  # the check says more than zero, not more than one
        ("SELECT emp FROM pay WHERE pos BETWEEN 2 AND 500", "SELECT emp FROM pay"),
        ("SELECT emp FROM pay WHERE pos BETWEEN 1 AND 499", "SELECT emp FROM pay"),
        ("SELECT emp FROM pay WHERE floor + 1 < ceiling", "SELECT emp FROM pay"),
        ("SELECT emp FROM pay WHERE emp > 0", "SELECT emp FROM pay"),  # no check on that column
        ("SELECT emp FROM pay WHERE amount >= 1", "SELECT emp FROM pay"),  # the solver's numbers are not whole: the caller passes ">= 1" for a whole-number column
    ],
)
def test_a_pair_the_checks_do_not_decide_is_not_proved(left, right):
    assert not proved(left, right)


def test_a_check_says_nothing_about_a_null():
    """A CHECK passes on NULL, so each of these needs the column declared NOT NULL as well."""

    for left, right in (BY_CHECKS[0], BY_CHECKS[2], BY_CHECKS[3]):
        assert not proved(left, right, kind_not_null=False), left
    for left, right in (BY_CHECKS[4], BY_CHECKS[6], BY_CHECKS[8]):
        assert not proved(left, right, pay_not_null=False), left
    # true with or without a NULL
    assert proved("SELECT machine FROM moves WHERE kind <> 'start'", "SELECT machine FROM moves WHERE kind = 'end'", kind_not_null=False)
    assert proved("SELECT emp, ABS(amount) AS a FROM pay", "SELECT emp, amount AS a FROM pay", pay_not_null=False)


def test_the_grammar():
    assert value_checks.columns(KIND) == ("kind",) and value_checks.columns(ORDERED) == ("floor", "ceiling")
    assert value_checks.columns(("=", ("column", "Kind"), ("value", "start"))) == ("kind",)
    assert value_checks.columns(("<", ("value", 0), ("column", "amount"))) == ("amount",)
    assert value_checks.columns(("<=", ("column", "amount"), ("value", Fraction(1, 2)))) == ("amount",)
    for unread in (
        None,
        "amount > 0",
        ("like", ("column", "kind"), ("value", "s%")),  # an operator the module does not read
        ("in", ("column", "kind"), ("values", ())),
        ("in", ("column", "kind"), ("values", ("start", None))),
        ("in", ("value", "start"), ("values", ("start",))),
        (">", ("value", 1), ("value", 0)),  # no column
        (">", ("column", "kind"), ("value", "a")),  # an ordering of texts
        ("=", ("column", "amount"), ("value", True)),
        ("=", ("column", "amount"), ("value", 1.5)),  # a float is not an exact value
        (">", ("column", "amount")),
        (">", ("column", 3), ("value", 0)),
    ):
        assert value_checks.columns(unread) == ()
        assert value_checks.facts(unread, {}, _value_sort()) == [] and not value_checks.violated(unread, {"kind": "x", "amount": -1})


def holds(check, values: dict) -> bool:
    """Whether the solver allows a row with these values (None for NULL) under the check's facts."""

    V = _value_sort()
    cols, solver = {}, z3.Solver()
    for name, value in values.items():
        cols[name] = _Val(z3.Bool(f"{name}#null"), z3.Const(name, V))
        solver.add(cols[name].null == z3.BoolVal(value is None))
        if isinstance(value, str):
            solver.add(cols[name].val == V.Str(z3.StringVal(value)))
        elif value is not None:
            solver.add(cols[name].val == V.Num(z3.RealVal(value)))
    solver.add(*value_checks.facts(check, cols, V))
    return solver.check() == z3.sat


def test_the_facts_are_the_checks():
    assert holds(KIND, {"kind": "start"}) and holds(KIND, {"kind": "end"}) and not holds(KIND, {"kind": "stop"})
    assert holds(KIND, {"kind": None}) and not holds(KIND, {"kind": 3})
    assert holds(POSITIVE, {"amount": 1}) and not holds(POSITIVE, {"amount": 0}) and not holds(POSITIVE, {"amount": -4}) and holds(POSITIVE, {"amount": None})
    assert not holds(POSITIVE, {"amount": "many"})  # an ordering is of numbers
    assert holds(TO_500, {"pos": 500}) and not holds(TO_500, {"pos": 501}) and holds(FROM_ONE, {"pos": 1}) and not holds(FROM_ONE, {"pos": 0})
    assert holds(ORDERED, {"floor": 1, "ceiling": 2}) and not holds(ORDERED, {"floor": 2, "ceiling": 2})
    assert holds(ORDERED, {"floor": None, "ceiling": 0}) and holds(ORDERED, {"floor": 9, "ceiling": None})
    assert holds(("<>", ("column", "floor"), ("column", "ceiling")), {"floor": 1, "ceiling": 2})
    assert not holds(("<>", ("column", "floor"), ("column", "ceiling")), {"floor": 2, "ceiling": 2})
    assert holds(("=", ("column", "kind"), ("value", "start")), {"kind": "start"}) and not holds(("=", ("column", "kind"), ("value", "start")), {"kind": "end"})
    assert holds(("in", ("column", "pos"), ("values", (0, 1))), {"pos": 1}) and not holds(("in", ("column", "pos"), ("values", (0, 1))), {"pos": 2})
    # a column the occurrence does not read gives no fact
    assert value_checks.facts(ORDERED, {"floor": _Val(z3.Bool("f#null"), z3.Const("f", _value_sort()))}, _value_sort()) == []


def test_a_row_that_breaks_a_check_is_named():
    assert value_checks.violated(KIND, {"kind": "stop"}) and not value_checks.violated(KIND, {"kind": "end"}) and not value_checks.violated(KIND, {"kind": None})
    assert not value_checks.violated(KIND, {"machine": 1})  # the column is not in the row
    assert value_checks.violated(POSITIVE, {"AMOUNT": 0}) and not value_checks.violated(POSITIVE, {"amount": 3}) and not value_checks.violated(POSITIVE, {"amount": None})
    assert value_checks.violated(ORDERED, {"floor": 2, "ceiling": 2}) and not value_checks.violated(ORDERED, {"floor": 1, "ceiling": 2})
    assert not value_checks.violated(ORDERED, {"floor": None, "ceiling": 2})
    assert not value_checks.violated(POSITIVE, {"amount": "many"})  # values that cannot be compared are not judged


def test_checks_agree_with_a_database():
    """Each check written as a SQL CHECK: the rows the database accepts are the rows the module does not call a violation."""

    con = duckdb.connect()
    con.execute("CREATE TABLE moves (kind VARCHAR CHECK (kind IN ('start', 'end')))")
    con.execute("CREATE TABLE pay (amount INT CHECK (amount > 0), pos INT CHECK (pos >= 1 AND pos <= 500), floor INT, ceiling INT, CHECK (floor < ceiling))")
    rows = [("moves", {"kind": k}) for k in ("start", "end", "stop", "", None)]
    rows += [("pay", {"amount": a, "pos": 1, "floor": 0, "ceiling": 1}) for a in (-1, 0, 1, None)]
    rows += [("pay", {"amount": 1, "pos": p, "floor": 0, "ceiling": 1}) for p in (0, 1, 500, 501, None)]
    rows += [("pay", {"amount": 1, "pos": 1, "floor": f, "ceiling": c}) for f, c in ((0, 1), (1, 1), (2, 1), (None, 1), (1, None))]
    checks = {"moves": (KIND,), "pay": (POSITIVE, FROM_ONE, TO_500, ORDERED)}
    refused = 0
    for table, row in rows:
        try:
            con.execute(f"INSERT INTO {table} ({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})", list(row.values()))
            accepted = True
        except duckdb.ConstraintException:
            accepted = False
            refused += 1
        assert accepted == (not any(value_checks.violated(check, row) for check in checks[table])), (table, row)
    assert refused == 8


def test_a_counterexample_must_meet_the_checks():
    """Without the check the two differ on a row with a negative amount; with it no such database is offered."""

    left, right = "SELECT emp FROM pay WHERE amount > 0", "SELECT emp FROM pay"
    options = dict(schema=SCHEMA, types=TYPES, dialect="mysql", compare_names=False, exact_arithmetic=True, timeout_ms=5000)
    assert prove_equivalent_algebraic(left, right, constraints=facts(checks=False), **options).status is SmtStatus.NOT_EQUIVALENT
    other = "SELECT emp FROM pay WHERE amount > 1"
    result = prove_equivalent_algebraic(other, right, constraints=facts(), **options)
    assert result.status is SmtStatus.NOT_EQUIVALENT and result.counterexample is not None
    rows = result.counterexample.tables["pay"]
    assert rows and not any(value_checks.violated(check, row) for check in (POSITIVE, FROM_ONE, TO_500, ORDERED) for row in rows), rows
    assert any(row["amount"] == 1 for row in rows)  # the one legal value on which the two differ


def test_conditions_keep_the_declared_checks():
    merged = with_conditions(facts(kind_not_null=False), [Condition(kind="not_null", table="moves", columns=("kind",))])
    assert merged["moves"].checks == (KIND,) and "kind" in merged["moves"].not_null
    assert merged["pay"].checks == (POSITIVE, FROM_ONE, TO_500, ORDERED)


def test_no_checks_is_the_default():
    assert TableConstraints().checks == ()
