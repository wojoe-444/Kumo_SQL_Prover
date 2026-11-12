"""``ABS(x) = n`` for a number ``n`` says ``x`` is ``n`` or ``-n``.

    SELECT DISTINCT a.seat_id FROM cinema AS a JOIN cinema AS b ON ABS(a.seat_id - b.seat_id) = 1

is the usual spelling of "the neighbouring row". The prover reads ``ABS`` as a function it knows nothing about,
so it could not connect this with ``a.seat_id = b.seat_id + 1 OR a.seat_id = b.seat_id - 1``. The pre-pass writes

    .. ON (a.seat_id - b.seat_id = 1 OR a.seat_id - b.seat_id = -1)

**Why it is the same.** For a number ``x``, ``ABS(x)`` is ``x`` or ``-x``, whichever is not negative, so for a
literal ``n > 0`` it equals ``n`` exactly when ``x = n`` or ``x = -n``. (``n = 0`` is left to the solver, which reads ``ABS(x - y) = 0`` as ``x = y`` itself.)
When ``x`` is NULL both sides are unknown (``NULL = n OR NULL = -n``). The identity holds in every position, so
no context is checked.

**What is read.** ``ABS(x) = n`` or ``n = ABS(x)`` where ``n`` is a numeric literal that is not negative and
``x`` is built from columns, numeric literals, ``+``, ``-``, ``*`` and parentheses. With column types supplied,
a column of ``x`` whose type is not numeric stops the rewrite (the engines would convert a string). Other
comparisons (``<``, ``>``) and other operands are left as they are.
"""

from __future__ import annotations

from sqlglot import exp

_NUMERIC = ("INT", "BIGINT", "SMALLINT", "TINYINT", "INTEGER", "DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "REAL", "BOOL", "NUMBER", "MEDIUMINT")


def read_absolute_values(tree: exp.Expression, types: dict[str, dict[str, str]] | None = None) -> exp.Expression:
    """Rewrite, in place, each ``ABS(x) = n`` described in the module docstring."""

    if not any(True for _ in tree.find_all(exp.Abs)):
        return tree
    by_column: dict[str, set[str]] = {}
    for columns in (types or {}).values():
        for name, kind in columns.items():
            by_column.setdefault(name.lower(), set()).add(str(kind).upper().split("(")[0].split(",")[0].strip())
    for test in list(tree.find_all(exp.EQ)):
        for call, number in ((test.this, test.expression), (test.expression, test.this)):
            call, number = _unwrap(call), _unwrap(number)
            if not isinstance(call, exp.Abs) or not isinstance(number, exp.Literal) or number.is_string:
                continue
            operand = call.this
            if not _arithmetic(operand, by_column):
                continue
            def inner() -> exp.Expression:
                piece = operand.copy()
                return exp.Paren(this=piece) if isinstance(piece, exp.Binary) else piece

            try:
                zero = float(number.this) == 0
            except ValueError:
                continue
            if zero:
                continue  # ABS(x - y) = 0 is read by the solver itself (smt_equivalence), which needs the ABS to be there
            else:
                replacement = exp.Paren(
                    this=exp.Or(
                        this=exp.EQ(this=inner(), expression=number.copy()),
                        expression=exp.EQ(this=inner(), expression=exp.Neg(this=number.copy())),
                    )
                )
            test.replace(replacement)
            break
    return tree


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _arithmetic(node: exp.Expression, by_column: dict[str, set[str]]) -> bool:
    for part in node.walk():
        if isinstance(part, exp.Column):
            if isinstance(part.this, exp.Star):
                return False
            kinds = by_column.get(part.name.lower())
            if kinds and not all(kind in _NUMERIC for kind in kinds):
                return False
        elif isinstance(part, exp.Literal):
            if part.is_string:
                return False
        elif not isinstance(part, (exp.Add, exp.Sub, exp.Mul, exp.Neg, exp.Paren, exp.Identifier)):
            return False
    return True
