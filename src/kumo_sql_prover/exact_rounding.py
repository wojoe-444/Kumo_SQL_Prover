"""MySQL ``ROUND`` over exact numbers: moved into the arms of a ``CASE``, and dropped over an integer.

    ROUND(CASE WHEN top > 100 THEN pay * 0.5 ELSE pay END, 0)
    CASE WHEN top > 100 THEN ROUND(pay * 0.5, 0) ELSE pay END

are two spellings of one value when ``pay`` is an ``INT`` column. The prover compares the arms of the two
``CASE`` expressions, and met ``ROUND(pay, 0)`` on one side and ``pay`` on the other. This rule brings both to the
second form.

**Rule 1: ROUND moves into the arms.** ``ROUND(CASE WHEN c1 THEN a1 .. ELSE an END, d)`` is
``CASE WHEN c1 THEN ROUND(a1, d) .. ELSE ROUND(an, d) END`` (and the same for ``IF(c, a, b)``) when every arm is
an *exact* number or NULL.

The conditions pick the same arm on both sides, so the question is whether rounding the arm's value gives what
rounding the ``CASE``'s value gives. MySQL gives a ``CASE`` one result type for all its arms, and converts the
chosen arm to it. When all arms are exact (integers and ``DECIMAL``), that type is an exact type with at least
the scale of every arm, so the conversion changes no value, and ``ROUND`` of a ``DECIMAL`` is defined on the
value (half away from zero). With a ``DOUBLE`` among the arms it would not hold: the ``CASE`` would be a
``DOUBLE``, a decimal ``2.5`` in another arm would be rounded as a binary number, and MySQL rounds those to the
nearest even digit on most platforms. A string arm brings the same conversion. So an arm must be *exact*:

* an integer or ``DECIMAL``/``NUMERIC`` column of a base table of the select (declared type, supplied by the
  caller), a number literal written without an exponent, NULL;
* ``+``, ``-``, ``*``, ``/``, ``DIV``, ``%`` and unary minus over exact operands (MySQL computes all of them in
  exact arithmetic when both operands are exact; ``/`` then rounds at its own fixed scale, which is part of the
  arm's value on both sides);
* ``ROUND`` of an exact number with a literal number of decimals, and a nested ``CASE`` or ``IF`` with exact arms.

A missing ``ELSE`` is a NULL arm, and ``ROUND(NULL, d)`` is NULL.

The ``CASE`` may also stand inside exact arithmetic: ``ROUND(pay - CASE WHEN c THEN pay * 0.5 ELSE 0 END, d)`` is
``CASE WHEN c THEN ROUND(pay - pay * 0.5, d) ELSE ROUND(pay - 0, d) END``. Between the ``ROUND`` and the ``CASE``
there are only parentheses, ``+``, ``-``, ``*`` and unary minus, and the whole argument is exact: these operators
are computed exactly on exact numbers, so the argument's value is the same expression with the chosen arm in
the ``CASE``'s place, and they give NULL for a NULL operand, which is what a missing ``ELSE`` gives. The other
operands are columns, literals and arithmetic, so writing them once per arm changes nothing. (At most three
``CASE``/``IF`` in one argument are read this way.) A ``/`` may be on the way too when every arm of the ``CASE``
is an integer (an integer column or literal, ``+ - *`` over those) or NULL: the ``CASE`` is then an integer of
the arm's value, and a division sees the same operand, value and type, as it sees with the arm in its place
(with a ``DECIMAL`` arm the ``CASE`` could carry more declared decimals than the arm, which a division shows).

**Rule 2: ROUND of a whole number is the number.** ``ROUND(e)`` and ``ROUND(e, d)`` with ``d`` a literal ``0``
or more are ``e`` when ``e`` is an exact whole number: an integer column, a plain number literal with a whole
value (``7``, ``1.0``), a zero literal times or divided by an exact number (``pay * 0 * 0.01`` is zero, or
NULL), or ``+``, ``-``, ``*`` and unary minus over those. In what is left, a zero divided by a number literal
other than zero is written without the division (``pay * 0 / 100`` is ``pay * 0``: zero, or NULL with ``pay``). MySQL returns the same number (an
integer for an integer, a ``DECIMAL`` without decimals for ``pay * 1.0``; the prover compares numbers by value,
as it does for ``pay * 1.0`` and ``pay``), and NULL for NULL.

**What is read.** Only on MySQL (BigQuery's ``ROUND`` of an ``INT64`` is a ``FLOAT64``). The ``ROUND`` stands in
the select list, the ``WHERE`` or an ``ON`` condition of its select (in ``GROUP BY``, ``HAVING`` and
``ORDER BY`` a bare name may be an output of the select, with another type). The number of decimals is absent
or a non-negative integer literal. A column is typed through the select the ``ROUND`` is in: its qualifier
names a base-table source of that select, or it is bare, every source of the select lists its columns (base
tables with declared types, derived tables whose outputs all have names, no ``USING``) and exactly one, a base
table, has the name; and the caller declared the column's type. A column of a derived table, of an outer query
or of unknown type is not exact, and the ``ROUND`` is left as written.
"""

from __future__ import annotations

import re
from decimal import Decimal

from sqlglot import exp

from .aggregate_membership import _table_key, _unwrap
from .ast_utils import select_sources, visible_ctes

_INTEGER = re.compile(r"^\s*(TINYINT|SMALLINT|MEDIUMINT|INT|INTEGER|BIGINT|INT64)\b(\s*\(\s*\d+\s*\))?\s*(UNSIGNED)?\s*$", re.I)
_DECIMAL = re.compile(r"^\s*(DECIMAL|NUMERIC|DEC|FIXED)\b", re.I)
_PLAIN_NUMBER = re.compile(r"^(\d+\.?\d*|\.\d+)$")
_MOST_CHOICES = 3


def exact_rounding(select: exp.Select, types: dict[str, dict[str, str]] | None, dialect: str) -> exp.Expression | None:
    """``select`` (in place) with the ``ROUND`` calls of its own scope rewritten as the module docstring says, else ``None``."""

    if dialect != "mysql" or not types:
        return None
    rounds = _own_rounds(select)
    if not rounds:
        return None
    column_type = _typer(select, types)
    changed = False
    for node in rounds:
        if not _attached(node, select):
            continue  # inside a ROUND that was rebuilt
        decimals = _decimals(node)
        if decimals is None:
            continue
        replacement = _rounded(node.this, node, decimals, column_type, top=True)
        if replacement is None:
            continue
        node.replace(replacement)
        changed = True
    return select if changed else None


def _own_rounds(select: exp.Select) -> list[exp.Round]:
    """The ``ROUND`` calls of ``select``'s own scope in its select list, ``WHERE`` and ``ON`` conditions.

    A bare name there is a column of a source. In ``GROUP BY``, ``HAVING`` and ``ORDER BY`` it may be an output of
    the select instead, whose type is not the column's, so those clauses are not read."""

    places = list(select.expressions)
    if select.args.get("where") is not None:
        places.append(select.args["where"])
    places += [join.args["on"] for join in select.args.get("joins") or [] if join.args.get("on") is not None]
    return [node for place in places for node in place.find_all(exp.Round) if node.find_ancestor(exp.Select) is select]


def _attached(node: exp.Expression, root: exp.Expression) -> bool:
    while node is not None:
        if node is root:
            return True
        node = node.parent
    return False


def _decimals(node: exp.Round) -> int | None:
    """The literal number of decimals (0 when absent), or None when the call is not one the rule reads."""

    if any(value for key, value in node.args.items() if key not in ("this", "decimals")):
        return None
    decimals = node.args.get("decimals")
    if decimals is None:
        return 0
    decimals = _unwrap(decimals)
    if isinstance(decimals, exp.Literal) and not decimals.is_string and re.fullmatch(r"\d+", decimals.name):
        return int(decimals.name)
    return None


def _typer(select: exp.Select, types: dict[str, dict[str, str]]):
    """A function from a column read in ``select``'s own scope to its declared type, or None.

    A qualified column is typed through its qualifier. A bare column is typed when every source of the select
    lists its columns (a base table with declared types, or a derived table whose outputs all have names) and
    exactly one of them has the name: that one is a base table, and the type is the declared one."""

    sources = select_sources(select)
    tables: dict[str, dict[str, str]] = {}
    listed: dict[str, set[str]] = {}
    complete = True
    for source in sources:
        name = source.alias_or_name.lower()
        if not name or name in listed:
            complete = False
            continue
        listed[name] = set()
        tables[name] = {}
        if isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier):
            alias = source.args.get("alias")
            declared = types.get(_table_key(source))
            if (alias is not None and alias.args.get("columns")) or (not source.args.get("db") and source.name.lower() in visible_ctes(source)) or not declared:
                complete = False
                continue
            tables[name] = {c.lower(): t for c, t in declared.items()}
            listed[name] = set(tables[name])
        elif isinstance(source, exp.Subquery) and isinstance(source.this, exp.Select) and not (source.args.get("alias") is not None and source.args["alias"].args.get("columns")):
            outputs = [item.alias_or_name.lower() for item in source.this.expressions]
            if any(isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)) for item in source.this.expressions) or "" in outputs:
                complete = False
                continue
            listed[name] = set(outputs)
        else:
            complete = False
    if any(join.method or join.args.get("using") for join in select.args.get("joins") or []):
        complete = False  # a merged column belongs to no single source

    def column_type(column: exp.Column) -> str | None:
        if isinstance(column.this, exp.Star) or column.args.get("db") or column.args.get("catalog"):
            return None
        name = column.name.lower()
        if column.table:
            return (tables.get(column.table.lower()) or {}).get(name)
        if not complete:
            return None
        owners = [source for source, columns in listed.items() if name in columns]
        return tables[owners[0]].get(name) if len(owners) == 1 else None

    return column_type


def _integer(node: exp.Expression, column_type) -> bool:
    node = _unwrap(node)
    if isinstance(node, exp.Column):
        declared = column_type(node)
        return bool(declared and _INTEGER.match(declared))
    if isinstance(node, exp.Literal):
        return not node.is_string and bool(re.fullmatch(r"\d+", node.name))
    if isinstance(node, exp.Neg):
        return _integer(node.this, column_type)
    if type(node) in (exp.Add, exp.Sub, exp.Mul):
        return _integer(node.this, column_type) and _integer(node.expression, column_type)
    return False


def _zero(node: exp.Expression, column_type) -> bool:
    """Whether ``node`` is zero (or NULL) whatever its columns hold: a zero literal, its product with an exact number,
    its quotient by one."""

    node = _unwrap(node)
    if isinstance(node, exp.Literal):
        return not node.is_string and bool(_PLAIN_NUMBER.match(node.name)) and Decimal(node.name) == 0
    if isinstance(node, exp.Neg):
        return _zero(node.this, column_type)
    if type(node) is exp.Mul:
        return (_zero(node.this, column_type) and _exact(node.expression, column_type)) or (_zero(node.expression, column_type) and _exact(node.this, column_type))
    if type(node) is exp.Div:
        return _zero(node.this, column_type) and _exact(node.expression, column_type)
    return False


def _whole(node: exp.Expression, column_type) -> bool:
    """Whether ``node`` is an exact number with a whole value: ``_integer``, with whole decimal literals and zero
    products too."""

    node = _unwrap(node)
    if _zero(node, column_type):
        return True
    if isinstance(node, exp.Literal):
        return not node.is_string and bool(_PLAIN_NUMBER.match(node.name)) and Decimal(node.name) == Decimal(node.name).to_integral_value()
    if isinstance(node, exp.Neg):
        return _whole(node.this, column_type)
    if type(node) in (exp.Add, exp.Sub, exp.Mul):
        return _whole(node.this, column_type) and _whole(node.expression, column_type)
    return _integer(node, column_type)


def _exact(node: exp.Expression, column_type) -> bool:
    node = _unwrap(node)
    if isinstance(node, exp.Null):
        return True
    if isinstance(node, exp.Column):
        declared = column_type(node)
        return bool(declared and (_INTEGER.match(declared) or _DECIMAL.match(declared)))
    if isinstance(node, exp.Literal):
        return not node.is_string and bool(_PLAIN_NUMBER.match(node.name))
    if isinstance(node, exp.Neg):
        return _exact(node.this, column_type)
    if type(node) in (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.IntDiv, exp.Mod):
        return _exact(node.this, column_type) and _exact(node.expression, column_type)
    if isinstance(node, exp.Round):
        return _decimals(node) is not None and _exact(node.this, column_type)
    arms = _arms(node)
    return arms is not None and all(_exact(arm, column_type) for arm in arms)


def _arms(node: exp.Expression) -> list[exp.Expression] | None:
    """The value arms of a ``CASE`` or ``IF`` (a missing ``ELSE`` is left out: it is NULL)."""

    if isinstance(node, exp.Case):
        if any(not isinstance(branch, exp.If) or branch.args.get("false") is not None for branch in node.args.get("ifs") or []):
            return None
        arms = [branch.args["true"] for branch in node.args["ifs"]]
        return arms + ([node.args["default"]] if node.args.get("default") is not None else [])
    if type(node) is exp.If and not isinstance(node.parent, exp.Case):
        return [node.args["true"]] + ([node.args["false"]] if node.args.get("false") is not None else [])
    return None


def _choice(node: exp.Expression, divided: bool = False) -> tuple[exp.Expression, bool] | None:
    """The first ``CASE`` or ``IF`` whose value reaches ``node`` through parentheses, ``+``, ``-``, ``*``, ``/`` and
    unary minus only, and whether a ``/`` is on the way."""

    node = _unwrap(node)
    if _arms(node) is not None:
        return node, divided
    if type(node) in (exp.Add, exp.Sub, exp.Mul):
        return _choice(node.this, divided) or _choice(node.expression, divided)
    if type(node) is exp.Div:
        return _choice(node.this, True) or _choice(node.expression, True)
    if isinstance(node, exp.Neg):
        return _choice(node.this, divided)
    return None


def _without_zero_quotients(node: exp.Expression, column_type) -> exp.Expression:
    """``node`` (new, changed in place) with each ``z / 100`` whose dividend is zero written ``z``.

    Zero (or NULL) divided by a literal that is not zero is zero (or NULL) again. Left as a quotient, it would be a
    division the prover cannot read once the ``ROUND`` around it is gone."""

    root = exp.Paren(this=node)
    for _ in range(32):
        for division in root.find_all(exp.Div):
            divisor = _unwrap(division.expression)
            literal = isinstance(divisor, exp.Literal) and not divisor.is_string and bool(_PLAIN_NUMBER.match(divisor.name)) and Decimal(divisor.name) != 0
            if literal and _zero(division.this, column_type):
                dividend = _unwrap(division.this).copy()
                division.replace(exp.Paren(this=dividend) if isinstance(dividend, (exp.Binary, exp.Neg)) else dividend)
                break
        else:
            break
    return root.this


def _rounded(value: exp.Expression, call: exp.Round, decimals: int, column_type, top: bool = False) -> exp.Expression | None:
    """``ROUND(value, decimals)`` in the rule's form; at the top, None when that is the call as written.

    ``value`` is read where it stands in the select (a column is typed through its select); the result is new."""

    inner = _unwrap(value)
    if _whole(inner, column_type):
        return _without_zero_quotients(exp.Paren(this=inner.copy()) if isinstance(inner, exp.Binary) else inner.copy(), column_type)
    if isinstance(inner, exp.Null):
        return inner.copy()
    arms = _arms(inner)
    if arms is not None and all(_exact(arm, column_type) for arm in arms):
        rounded = [_rounded(arm, call, decimals, column_type) for arm in arms]
        result = inner.copy()
        for arm, replacement in zip(_arms(result), rounded):
            arm.replace(replacement)
        return result
    found = _choice(inner)
    choice = found[0] if found is not None else None
    if (
        choice is not None
        and choice is not inner
        and _exact(inner, column_type)
        and sum(1 for node in inner.find_all(exp.Case, exp.If) if _arms(node) is not None) <= _MOST_CHOICES
        and (not found[1] or all(_integer(arm, column_type) or isinstance(_unwrap(arm), exp.Null) for arm in _arms(choice)))
    ):
        # f(CASE WHEN c THEN a .. END) is CASE WHEN c THEN f(a) .. END for exact +, - and *: round each
        rounded = []
        for index in range(len(_arms(choice))):
            whole = exp.Paren(this=inner.copy())
            there = _choice(whole.this)[0]
            arm = _arms(there)[index].copy()
            there.replace(arm if isinstance(arm, (exp.Column, exp.Literal, exp.Null, exp.Paren)) else exp.Paren(this=arm))
            rounded.append(_rounded(whole.this, call, decimals, column_type))
        result = choice.copy()
        for arm, replacement in zip(_arms(result), rounded):
            arm.replace(replacement)
        return result
    if top:
        return None
    again = call.copy()
    again.set("this", value.copy())
    return again
