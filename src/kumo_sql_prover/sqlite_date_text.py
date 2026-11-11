"""SQLite: the digits ``STRFTIME`` returns, read as text or as a number.

    SELECT title FROM movie WHERE CAST(STRFTIME('%Y', release_date) AS INTEGER) = 1945
    SELECT title FROM movie WHERE STRFTIME('%Y', release_date) = '1945'

``STRFTIME('%Y', x)`` is NULL or the year as text: four digits, ``0000`` to ``9999`` (a date before year 0 gives a
minus sign and digits, ``-100``). ``%m``, ``%d``, ``%H``, ``%M`` and ``%S`` give two digits and ``%j`` three. Translations
between dialects move freely between the text and the number, and the provers read ``CAST`` and ``STRFTIME`` as
unknown functions, so the two spellings never matched. For ``D`` one of these calls with a single specifier:

**1. A comparison of the number is a comparison of the text.** ``CAST(D AS INTEGER) OP n`` is ``D OP 'n'`` with ``n``
written at the specifier's width (``7`` as ``'07'`` for ``%m``), for ``= != < <= > >=``, ``BETWEEN`` and ``IN``, when
``0 <= n`` and ``n`` fits the width. Digit strings of one width sort as their numbers do (SQLite compares text byte
by byte), only the string at that width has the number ``n``, and a year before 0 starts with ``-``, which sorts
below every digit as its number sorts below every ``n >= 0``. Both sides are NULL together.

**2. Arithmetic reads the text as its number.** In ``D + x``, ``D - x``, ``D * x``, ``D / x`` and ``D % x`` (either
side) SQLite converts the text to the integer it spells, which is what ``CAST(D AS INTEGER)`` is; the bare call is
written with the cast.

SQLite only. Declined: a format with more than one specifier or any other text (``'%Y-%m'``), another specifier
(``%J`` and ``%f`` have a fraction, ``%s`` has no fixed width), a ``STRFTIME`` with modifiers, a number that is
negative or too long for the width (the comparison is then decided by the width, not by the text), a cast to
another type, and a bare call compared with a number (``STRFTIME('%Y', d) < 1997`` compares text with an integer,
which SQLite orders by type: never true).
"""

from __future__ import annotations

import re

from sqlglot import exp

_WIDTHS = {"%Y": 4, "%m": 2, "%d": 2, "%H": 2, "%M": 2, "%S": 2, "%j": 3}
_INTEGER_TYPES = {exp.DataType.Type.INT, exp.DataType.Type.BIGINT}
_COMPARISONS = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE)
_ARITHMETIC = (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod)
_DIGITS = re.compile(r"[0-9]+")


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _width(node: exp.Expression) -> int | None:
    """The width of the digits ``node`` returns when it is ``STRFTIME`` with one of the specifiers above."""

    if type(node) is not exp.TimeToStr or any(v for k, v in node.args.items() if k not in ("this", "format")):
        return None
    form = node.args.get("format")
    return _WIDTHS.get(form.this) if isinstance(form, exp.Literal) and form.is_string else None


def _number_of(node: exp.Expression) -> exp.Expression | None:
    """``D`` when ``node`` is ``CAST(D AS INTEGER)``."""

    node = _unwrap(node)
    if type(node) is not exp.Cast or any(v for k, v in node.args.items() if k not in ("this", "to")):
        return None
    to = node.args.get("to")
    if not isinstance(to, exp.DataType) or to.this not in _INTEGER_TYPES or to.expressions:
        return None
    inner = _unwrap(node.this)
    return inner if _width(inner) is not None else None


def _as_text(node: exp.Expression, width: int) -> exp.Literal | None:
    """The integer literal ``node`` written at ``width``, or None when it is not one or does not fit."""

    node = _unwrap(node)
    if not isinstance(node, exp.Literal) or node.is_string or not _DIGITS.fullmatch(node.this):
        return None
    number = int(node.this)
    return exp.Literal.string(str(number).zfill(width)) if number < 10**width else None


def sqlite_date_text_rules(tree: exp.Expression, dialect: str) -> exp.Expression:
    """``tree`` with the two rewrites of the module doc applied; another dialect's tree is returned as it is."""

    if dialect != "sqlite" or tree.find(exp.TimeToStr) is None:
        return tree
    for node in list(tree.find_all(*_COMPARISONS, exp.Between, exp.In)):
        if isinstance(node, _COMPARISONS):
            for mine, other in (("this", "expression"), ("expression", "this")):
                date = _number_of(node.args[mine])
                text = _as_text(node.args[other], _width(date)) if date is not None else None
                if text is not None:
                    node.set(mine, date.copy())
                    node.set(other, text)
                    break
        elif isinstance(node, exp.Between):
            date = _number_of(node.this)
            if date is None or any(v for k, v in node.args.items() if k not in ("this", "low", "high")):
                continue
            low, high = _as_text(node.args["low"], _width(date)), _as_text(node.args["high"], _width(date))
            if low is not None and high is not None:
                node.set("this", date.copy())
                node.set("low", low)
                node.set("high", high)
        else:
            date = _number_of(node.this)
            if date is None or not node.expressions or any(v for k, v in node.args.items() if k not in ("this", "expressions")):
                continue
            texts = [_as_text(item, _width(date)) for item in node.expressions]
            if None not in texts:
                node.set("this", date.copy())
                node.set("expressions", texts)
    for node in list(tree.find_all(*_ARITHMETIC)):
        for key in ("this", "expression"):
            operand = _unwrap(node.args[key])
            if _width(operand) is not None:
                node.set(key, exp.Cast(this=operand.copy(), to=exp.DataType.build("INTEGER", dialect="sqlite")))
    return tree
