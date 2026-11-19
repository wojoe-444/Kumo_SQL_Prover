"""The number of decimals MySQL declares for an expression, as a value two queries can be compared on.

MySQL gives every exact-number expression a declared scale and rounds a value to it when it is returned or stored
(Kumo_SQL_Prover issue 47). The rules, measured on MySQL 8.0.46 with ``CREATE TABLE .. AS SELECT`` over made-up tables:

* a whole-number column or literal has scale 0, a decimal literal has the digits after its point (``1.0`` has 1), a
  ``DECIMAL(p, s)`` column has ``s``;
* a product adds the scales of its factors, a sum or difference takes the larger one;
* a quotient has the scale of its dividend plus four (``div_precision_increment``): ``paid / due`` has 4,
  ``paid * 1.0 / due`` has 5, ``100.0 * paid / due`` has 5, ``paid / due / due`` has 8;
* ``SUM``, ``MIN``, ``MAX`` keep the scale, ``AVG`` adds four, ``COUNT`` has 0;
* ``ROUND(v, r)`` and ``TRUNCATE(v, r)`` give ``min(r, scale of v)``, ``ROUND(v)``, ``FLOOR`` and ``CEIL`` give 0;
* ``CASE``, ``IF``, ``COALESCE``, ``IFNULL``, ``GREATEST`` and ``LEAST`` take the largest scale of their arms;
* an operand that is a floating-point number makes the whole expression floating point.

So ``SELECT paid / due`` returns ``0.6667`` and ``SELECT paid * 1.0 / due`` returns ``0.66667`` on the same row: the
value is the same quotient, cut at a different declared scale. The provers read ``1.0`` and ``1`` as the same number, so
they proved the two queries equal.

A scale here is a value, not just a number, because a column of a table the schema does not type has an unknown scale:
a *set of terms*, each one ``(constant, columns, cap)`` for ``min(cap, constant + the scales of those columns)``, and the
scale is the largest term. Two scales are the same when their sets are equal. ``FLOAT`` stands for floating point and
``None`` for a scale this module cannot work out; a pair whose scales are not known to be equal is not a pair it can
call equal.
"""

from __future__ import annotations

import re

FLOAT = "float"
_MAX_SCALE = 30  # MySQL declares at most 30 decimals
DIVISION_INCREMENT = 4


def constant(digits: int):
    return frozenset({(min(digits, _MAX_SCALE), (), None)})


def column(name: str):
    return frozenset({(0, (name,), None)})


def _clean(terms):
    """Terms in their simplest form: a cap at or below the constant is the cap; a constant term the others outgrow goes."""

    fixed = set()
    for digits, columns, cap in terms:
        if cap is not None and (not columns or cap <= digits):
            digits, columns, cap = min(cap, digits), (), None
        if not columns:
            digits = min(digits, _MAX_SCALE)
        fixed.add((digits, columns, cap))
    bound = lambda t: t[0] if t[2] is None else min(t[2], t[0])  # noqa: E731 - the least the term can be
    return frozenset(t for t in fixed if t[1] or not any(u != t and bound(u) >= t[0] for u in fixed))


def _both(first, second):
    if first is None or second is None:
        return None
    return FLOAT if FLOAT in (first, second) else True


def plus(first, second):
    """The scale of a product: the scales add (a floating-point factor makes the product floating point)."""

    if _both(first, second) is not True:
        return _both(first, second)
    if any(cap is not None for _, _, cap in first) or any(cap is not None for _, _, cap in second):
        return None
    return _clean({(a + c, tuple(sorted(b + d)), None) for a, b, _ in first for c, d, _ in second})


def add_digits(scale, digits: int):
    """The scale of a quotient: its dividend's plus ``digits``."""

    if scale is None or scale == FLOAT:
        return scale
    if any(cap is not None for _, _, cap in scale):
        return None
    return _clean({(a + digits, b, None) for a, b, _ in scale})


def largest(scales):
    """The scale of ``CASE``, ``IF``, ``COALESCE``, a sum: the largest of its operands'."""

    scales = list(scales)
    if not scales or any(s is None for s in scales):
        return None
    if any(s == FLOAT for s in scales):
        return FLOAT
    return _clean(set().union(*scales))


def rounded(scale, digits: int):
    """The scale of ``ROUND(v, digits)``: no more than ``v`` declares, no more than ``digits`` (none at all below zero)."""

    if scale is None or scale == FLOAT:
        return scale
    digits = max(digits, 0)
    return _clean({(a, b, digits if cap is None else min(cap, digits)) for a, b, cap in scale})


def of_type(declared: str | None):
    """The scale of a column declared ``declared`` (``INT``, ``DECIMAL(10, 2)``, ``DOUBLE``), or None when it is unknown."""

    if not declared:
        return None
    text = declared.strip().upper()
    name = re.split(r"[\s(]", text, maxsplit=1)[0]
    if name in {"DECIMAL", "NUMERIC", "DEC", "FIXED"}:
        found = re.search(r"\(\s*\d+\s*,\s*(\d+)\s*\)", text)
        return constant(int(found.group(1)) if found else 0)
    if name in {"FLOAT", "DOUBLE", "REAL"}:
        return FLOAT
    if name in {"INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "MEDIUMINT", "BIT", "YEAR", "BOOL", "BOOLEAN", "SIGNED", "UNSIGNED"}:
        return constant(0)
    return None


def of_literal(text: str):
    """The scale of a number literal: the digits after its point, or floating point when it has an exponent."""

    if "e" in text.lower():
        return FLOAT
    return constant(len(text.split(".", 1)[1]) if "." in text else 0)


def tag(scale, fallback: str) -> str:
    """A short text two equal scales share and two different ones do not; ``fallback`` stands in for an unknown one."""

    if scale is None:
        return "?" + fallback
    if scale == FLOAT:
        return FLOAT
    return "|".join(f"{a}+{'*'.join(b)}" + (f"<{cap}" if cap is not None else "") for a, b, cap in sorted(scale, key=repr))
