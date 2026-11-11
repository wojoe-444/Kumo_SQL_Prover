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

**Rule 2: ROUND of a whole number is the number.** ``ROUND(e)`` and ``ROUND(e, d)`` with ``d`` a literal ``0``
or more are ``e`` when ``e`` is an exact whole number: an integer column, a plain number literal with a whole
value (``7``, ``1.0``), or ``+``, ``-``, ``*`` and unary minus over those. MySQL returns the same number (an
integer for an integer, a ``DECIMAL`` without decimals for ``pay * 1.0``; the prover compares numbers by value,
as it does for ``pay * 1.0`` and ``pay``), and NULL for NULL.

**What is read.** Only on MySQL (BigQuery's ``ROUND`` of an ``INT64`` is a ``FLOAT64``). The number of decimals
is absent or a non-negative integer literal. A column is typed through the select the ``ROUND`` is in: its
qualifier names a base-table source of that select (or it is bare and the select has that one base table as its
only source), and the caller declared the column's type. A column of a derived table, of an outer query or of
unknown type is not exact, and the ``ROUND`` is left as written.
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


def exact_rounding(select: exp.Select, types: dict[str, dict[str, str]] | None, dialect: str) -> exp.Expression | None:
    """``select`` (in place) with the ``ROUND`` calls of its own scope rewritten as the module docstring says, else ``None``."""

    if dialect != "mysql" or not types:
        return None
    rounds = [node for node in select.find_all(exp.Round) if node.find_ancestor(exp.Select) is select]
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
    """A function from a column of ``select``'s own scope to its declared type, or None."""

    sources = select_sources(select)
    tables: dict[str, dict[str, str]] = {}
    plain = True
    for source in sources:
        name = source.alias_or_name.lower()
        if isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier) and name and name not in tables:
            alias = source.args.get("alias")
            if (alias is not None and alias.args.get("columns")) or (not source.args.get("db") and source.name.lower() in visible_ctes(source)):
                plain = False
                tables[name] = {}
                continue
            tables[name] = {c.lower(): t for c, t in (types.get(_table_key(source)) or {}).items()}
        else:
            plain = False
            if name:
                tables[name] = {}

    def column_type(column: exp.Column) -> str | None:
        if isinstance(column.this, exp.Star) or column.args.get("db") or column.args.get("catalog"):
            return None
        if column.find_ancestor(exp.Select) is not select:
            return None
        if column.table:
            return (tables.get(column.table.lower()) or {}).get(column.name.lower())
        if plain and len(sources) == 1:
            return next(iter(tables.values()), {}).get(column.name.lower())
        return None

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


def _whole(node: exp.Expression, column_type) -> bool:
    """Whether ``node`` is an exact number with a whole value: ``_integer``, with whole decimal literals too."""

    node = _unwrap(node)
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


def _rounded(value: exp.Expression, call: exp.Round, decimals: int, column_type, top: bool = False) -> exp.Expression | None:
    """``ROUND(value, decimals)`` in the rule's form; at the top, None when that is the call as written.

    ``value`` is read where it stands in the select (a column is typed through its select); the result is new."""

    inner = _unwrap(value)
    if _whole(inner, column_type):
        return exp.Paren(this=inner.copy()) if isinstance(inner, exp.Binary) else inner.copy()
    if isinstance(inner, exp.Null):
        return inner.copy()
    arms = _arms(inner)
    if arms is not None and all(_exact(arm, column_type) for arm in arms):
        rounded = [_rounded(arm, call, decimals, column_type) for arm in arms]
        result = inner.copy()
        for arm, replacement in zip(_arms(result), rounded):
            arm.replace(replacement)
        return result
    if top:
        return None
    again = call.copy()
    again.set("this", value.copy())
    return again
