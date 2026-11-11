"""``x * 0.5`` is ``x / 2`` in FLOAT64, bit for bit.

BigQuery computes both in FLOAT64 when ``x`` is INT64 or FLOAT64: ``/`` converts both operands to FLOAT64, and a
decimal literal beside an INT64 or FLOAT64 operand is a FLOAT64, which converts an INT64 ``x`` the same way. A
power of two and its reciprocal are both exact doubles, so the exact product ``x * 2**-k`` and the exact quotient
``x / 2**k`` are the same real number, and IEEE 754 rounds one real number to one double (the same subnormal
when the result underflows, the same infinity or NaN when ``x`` is one). Neither can fail: the divisor is not zero
and halving cannot overflow.

The SMT prover reads a FLOAT64 product and a quotient as two unrelated functions, so it left the pair unknown.
:func:`as_division` gives a product by such a literal its spelling as a division, which the prover then compiles
as it compiles any division. It answers ``None`` for anything else: a factor that is not a power of two (``x * 0.1``
and ``x / 10`` differ in the last bit), a factor of 1 or more (``x * 2`` is left to the product), an integer literal
(``x * 1`` stays INT64), and an operand whose type is not visibly INT64 or FLOAT64 (a NUMERIC operand makes the
literal NUMERIC, with other rounding).
"""

from __future__ import annotations

import math

from sqlglot import exp

from . import smt_values

_FLOATING = (smt_values.INT64, smt_values.FLOAT64)


def _reciprocal_power(literal: exp.Expression) -> int | None:
    """``2**k`` (k from 1 to 60) when ``literal`` is a decimal or exponent literal whose double is ``2**-k``."""

    if not isinstance(literal, exp.Literal) or literal.is_string or smt_values.is_integer_text(literal.this):
        return None
    try:
        value = float(literal.this)
    except ValueError:
        return None
    if not 0 < value < 1:
        return None
    mantissa, exponent = math.frexp(value)  # value == mantissa * 2**exponent with 0.5 <= mantissa < 1
    if mantissa != 0.5 or not -59 <= exponent <= 0:
        return None
    return 2 ** (1 - exponent)


def as_division(product: exp.Mul, class_of) -> exp.Div | None:
    """``operand / 2**k`` for ``operand * 2**-k`` (either order) over an INT64 or FLOAT64 operand, else ``None``.

    ``class_of`` gives the visible type of an operand (``smt_values`` class names, ``None`` when it is not visible).
    """

    for factor, operand in ((product.expression, product.this), (product.this, product.expression)):
        divisor = _reciprocal_power(factor)
        if divisor is None:
            continue
        if class_of(operand) not in _FLOATING:
            return None
        return exp.Div(this=operand.copy(), expression=exp.Literal.number(str(divisor)))
    return None
