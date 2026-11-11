"""``x * 0.5`` is ``x / 2`` in FLOAT64; no other factor is."""

from __future__ import annotations

import pytest
import sqlglot

from kumo_sql_prover import float_halving, smt_values
from kumo_sql_prover.smt_equivalence import SmtStatus, prove_equivalent_smt

SCHEMA = {"t": ["x", "f", "n", "s"]}
TYPES = {"t": {"x": "INT64", "f": "FLOAT64", "n": "NUMERIC", "s": "STRING"}}


def _prove(left: str, right: str):
    return prove_equivalent_smt(left, right, schema=SCHEMA, types=TYPES)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("SELECT f * 0.5 AS a FROM t", "SELECT f / 2 AS a FROM t"),
        ("SELECT 0.5 * f AS a FROM t", "SELECT f / 2 AS a FROM t"),
        ("SELECT x * 0.5 AS a FROM t", "SELECT x / 2 AS a FROM t"),
        ("SELECT f * 0.25 AS a FROM t", "SELECT f / 4 AS a FROM t"),
        ("SELECT f * 5e-1 AS a FROM t", "SELECT f / 2 AS a FROM t"),
        ("SELECT x FROM t WHERE f * 0.125 > 3", "SELECT x FROM t WHERE f / 8 > 3"),
        ("SELECT s, SUM(f * 0.5) AS a FROM t GROUP BY s", "SELECT s, SUM(f / 2) AS a FROM t GROUP BY s"),
    ],
)
def test_a_product_by_a_reciprocal_power_of_two_is_the_division(left, right):
    assert _prove(left, right).status is SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("SELECT f * 0.1 AS a FROM t", "SELECT f / 10 AS a FROM t"),  # 0.1 is not a double: last-bit differences
        ("SELECT f * 0.2 AS a FROM t", "SELECT f / 5 AS a FROM t"),
        ("SELECT f * 0.5 AS a FROM t", "SELECT f / 4 AS a FROM t"),
        ("SELECT n * 0.5 AS a FROM t", "SELECT n / 2 AS a FROM t"),  # NUMERIC arithmetic rounds otherwise
        ("SELECT x * 0.5 AS a FROM t", "SELECT DIV(x, 2) AS a FROM t"),  # integer division truncates
        ("SELECT f * 0.5 * 2 AS a FROM t", "SELECT f AS a FROM t"),  # the smallest subnormal halves to zero
        ("SELECT x * 0.5 + x * 0.5 AS a FROM t", "SELECT x AS a FROM t"),  # FLOAT64 cannot hold 2**53 + 1
        ("SELECT SUM(f * 0.5) AS a FROM t", "SELECT SUM(f) / 2 AS a FROM t"),
    ],
)
def test_other_arithmetic_stays_unproven(left, right):
    assert _prove(left, right).status is not SmtStatus.PROVEN_EQUIVALENT


def _rewrite(sql: str, kind: str | None):
    return float_halving.as_division(sqlglot.parse_one(sql, read="bigquery"), lambda operand: kind)


@pytest.mark.parametrize(
    ("sql", "kind", "expected"),
    [
        ("f * 0.5", smt_values.FLOAT64, "f / 2"),
        ("0.5 * f", smt_values.FLOAT64, "f / 2"),
        ("f * 0.0625", smt_values.INT64, "f / 16"),
        ("f * 2.5e-1", smt_values.FLOAT64, "f / 4"),
        ("(f + g) * 0.5", smt_values.FLOAT64, "(f + g) / 2"),
    ],
)
def test_as_division_spells_the_quotient(sql, kind, expected):
    assert _rewrite(sql, kind).sql(dialect="bigquery") == expected


@pytest.mark.parametrize(
    ("sql", "kind"),
    [
        ("f * 0.5", None),  # the operand's type is not visible
        ("f * 0.5", smt_values.type_class("NUMERIC")),
        ("f * 0.3", smt_values.FLOAT64),
        ("f * 0.75", smt_values.FLOAT64),
        ("f * 1.0", smt_values.FLOAT64),
        ("f * 2.0", smt_values.FLOAT64),
        ("f * 0.0", smt_values.FLOAT64),
        ("f * 1", smt_values.INT64),
        ("f * '0.5'", smt_values.FLOAT64),
        ("f * g", smt_values.FLOAT64),
        ("f * 1e-400", smt_values.FLOAT64),  # underflows to zero
        ("f * 5e-324", smt_values.FLOAT64),  # a power of two too small to name a divisor for
    ],
)
def test_as_division_leaves_everything_else(sql, kind):
    assert _rewrite(sql, kind) is None


def test_every_accepted_factor_is_an_exact_reciprocal():
    for k in range(1, 61):
        literal = sqlglot.exp.Literal.number(repr(2.0**-k))
        divisor = float_halving._reciprocal_power(literal)
        assert divisor == 2**k and float(literal.this) * divisor == 1.0
