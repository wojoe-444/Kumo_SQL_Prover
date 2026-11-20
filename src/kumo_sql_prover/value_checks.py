"""Declared value checks of a table (``TableConstraints.checks``), as facts for the solver.

A table may declare more than keys and NOT NULL columns: ``CHECK (kind IN ('start', 'end'))``, ``CHECK (amount > 0)``,
``CHECK (position BETWEEN 1 AND 500)``, ``CHECK (login < logout)``, or an ``ENUM`` column. Two queries that differ only
on rows such a check rules out return the same rows on every database the declaration admits:

    SUM(CASE WHEN kind = 'start' THEN -at ELSE at END)        -- kind is 'start' or 'end', never NULL
    SUM(CASE WHEN kind = 'end' THEN at ELSE -at END)

**What a check says.** As SQL defines ``CHECK``: on every row the condition is TRUE or NULL, never FALSE. So a check
says nothing about a row on which one of its columns is NULL, and each fact below is guarded by "the columns are not
NULL". Whether the column can be NULL is the business of ``TableConstraints.not_null``.

**The grammar.** Each check is a tuple ``(operator, left, right)`` over one row of the table:

* ``("in", ("column", name), ("values", (v1, v2, ..)))``: the column is one of the listed values;
* ``(op, left, right)`` with ``op`` one of ``= <> < <= > >=`` and each side ``("column", name)`` or ``("value", v)``.

A value is an ``int``, a ``fractions.Fraction`` or a ``str``. A text is compared as the solver compares texts: exactly.
**Assumed** for a text value: the column holds the listed text as it is written (what an ``ENUM`` column stores), not
another text that a collation calls equal to it. An ordering (``< <= > >=``) is read as an ordering of numbers, so the
caller passes one only for columns that hold numbers; with a text on either side it is skipped, and so is every check
this module cannot read. A skipped check is a fact not used, which is always safe.

``facts`` gives the solver's assertions for one occurrence of the table; ``violated`` tells whether a row of a
made-up database breaks a check (such a database is not a legal counterexample).
"""

from __future__ import annotations

from fractions import Fraction

try:  # pragma: no cover
    import z3
except ImportError:  # pragma: no cover
    z3 = None  # only ``facts`` needs the solver, and only the solver layer calls it

ORDERINGS = ("<", "<=", ">", ">=")
OPERATORS = ("=", "<>", *ORDERINGS)


def _number(value) -> Fraction | None:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        return None
    return Fraction(value)


def _well_formed(check) -> bool:
    if not isinstance(check, tuple) or len(check) != 3:
        return False
    op, left, right = check
    sides = (left, right)
    if not all(isinstance(side, tuple) and len(side) == 2 for side in sides):
        return False
    if op == "in":
        return left[0] == "column" and right[0] == "values" and bool(right[1]) and all(isinstance(v, str) or _number(v) is not None for v in right[1])
    if op not in OPERATORS or not any(side[0] == "column" for side in sides):
        return False
    for kind, payload in sides:
        if kind == "column":
            if not isinstance(payload, str):
                return False
        elif kind != "value" or not (isinstance(payload, str) or _number(payload) is not None):
            return False
        elif isinstance(payload, str) and op in ORDERINGS:
            return False  # an ordering of texts is not read
    return True


def columns(check) -> tuple[str, ...]:
    """The columns a well-formed check reads (lower-case), or ``()`` when the check is not read."""

    if not _well_formed(check):
        return ()
    return tuple(payload.lower() for kind, payload in check[1:] if kind == "column")


def facts(check, cols: dict, sort) -> list:
    """The assertions ``check`` gives for one occurrence whose column values are ``cols`` (name -> value with
    ``.null`` and ``.val``): none unless every column of the check is among them."""

    names = columns(check)
    if not names or any(name not in cols for name in names):
        return []
    op, left, right = check
    known = z3.And(*[z3.Not(cols[name].null) for name in names])

    def literal(value):
        number = _number(value)
        if number is None:
            return sort.Str(z3.StringVal(value))
        return sort.Num(z3.RealVal(f"{number.numerator}/{number.denominator}"))

    if op == "in":
        value = cols[left[1].lower()].val
        return [z3.Implies(known, z3.Or(*[value == literal(v) for v in right[1]]))]
    terms = [cols[payload.lower()].val if kind == "column" else literal(payload) for kind, payload in (left, right)]
    if op == "=":
        return [z3.Implies(known, terms[0] == terms[1])]
    if op == "<>":
        return [z3.Implies(known, terms[0] != terms[1])]
    a, b = sort.num(terms[0]), sort.num(terms[1])
    ordered = {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
    return [z3.Implies(known, z3.And(sort.is_Num(terms[0]), sort.is_Num(terms[1]), ordered))]


def violated(check, row: dict) -> bool:
    """Whether ``row`` (column name -> Python value, None for NULL) makes ``check`` FALSE. A check that is not read, a
    missing column, a NULL and values that cannot be compared never count as a violation."""

    names = columns(check)
    if not names:
        return False
    lowered = {str(key).lower(): value for key, value in row.items()}
    if any(name not in lowered or lowered[name] is None for name in names):
        return False
    op, left, right = check

    def value(side):
        kind, payload = side
        return lowered[payload.lower()] if kind == "column" else payload

    def comparable(a, b) -> bool:
        return (isinstance(a, str) and isinstance(b, str)) or (_plain_number(a) and _plain_number(b))

    if op == "in":
        held = value(left)
        options = [v for v in right[1] if comparable(held, v)]
        return not any(held == v for v in options) and len(options) == len(right[1])
    a, b = value(left), value(right)
    if not comparable(a, b) or (op in ORDERINGS and isinstance(a, str)):
        return False
    return not {"=": a == b, "<>": a != b, "<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]


def _plain_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float, Fraction))
