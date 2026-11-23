"""Operations that give NULL for operands that are not NULL, engine by engine.

The solver reads ``x + y`` as NULL exactly when an operand is. Three operations do not follow that rule in every
engine, and reading them as if they did proves pairs that an engine tells apart:

* ``x % y`` and ``MOD(x, y)`` with a zero divisor. BigQuery and PostgreSQL raise an error, which a proof may assume
  away (a pair is compared where both queries run). MySQL, SQLite and DuckDB return NULL, so over NOT NULL columns
  ``COUNT(x % y)`` is not ``COUNT(*)`` and ``WHERE x % y IS NULL`` is not an empty filter.
* ``x / y`` with a zero divisor: NULL in MySQL and SQLite (the solver already kept that possible in every dialect).
  In MySQL a zero divisor is the only way: for a divisor that is a number, the quotient and the remainder are NULL
  exactly when it is zero (:func:`zero_divisor_only`). SQLite's ``%`` casts its operands to integers first
  (``5 % 0.5`` is NULL) and DuckDB's ``/`` over DOUBLE gives infinity, so there the NULL stays an unknown.
* ``CAST(x AS DATE)`` (and the other date and time types) of a value that is not a date: an error in BigQuery,
  PostgreSQL and DuckDB, NULL with a warning in MySQL (``CAST('soon' AS DATE)``, ``CAST(5 AS DATE)``).
  :func:`mysql_cast_may_be_null` says when the operand is not known to be a date.

``output_properties`` uses the same facts for the columns it reports as never NULL, and there also for the
functions with a limited domain (``SQRT(-1)``, ``LN(0)``, ``LOG(0)`` and ``MOD(5, 0)`` are NULL in MySQL and
SQLite, ``POWER(-8, 0.5)`` in SQLite).

Issue 64. Checked on MySQL 8.0.46, SQLite 3 and DuckDB 1.5.6; ``tests/test_engine_null_results.py`` holds the pairs.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlglot import exp

from . import date_witness

#: Engines where a zero divisor, a value outside a function's domain and a failed cast are errors, never NULL.
FAILING = frozenset({"bigquery", "postgres"})

#: Functions that return NULL outside their domain where that is not an error.
LIMITED_DOMAIN = frozenset({"SQRT", "MOD", "POWER", "POW", "LN", "LOG", "LOG2", "LOG10"})

_TEMPORAL = (
    exp.DataType.Type.DATE,
    exp.DataType.Type.DATETIME,
    exp.DataType.Type.TIMESTAMP,
    exp.DataType.Type.TIMESTAMPTZ,
    exp.DataType.Type.TIMESTAMPLTZ,
    exp.DataType.Type.TIME,
)
_MOMENTS = _TEMPORAL[:-1]
_PICKS = (exp.Min, exp.Max, exp.AnyValue)
_NOW = (exp.CurrentDate, exp.CurrentTimestamp, exp.CurrentDatetime)


def fails_instead(dialect: str | None) -> bool:
    """Whether ``dialect`` raises an error where another engine returns NULL (a zero divisor, a failed cast)."""

    return dialect in FAILING


def zero_divisor_only(dialect: str | None) -> bool:
    """Whether a quotient or remainder by a number is NULL exactly when the number is zero (operands not NULL)."""

    return dialect == "mysql"


def nonzero_whole_literal(node: exp.Expression | None) -> bool:
    """Whether ``node`` is a whole number other than zero written out (``2``, ``-100``, ``(7)``).

    A quotient or a remainder by one is NULL in no engine when the dividend is not NULL. A decimal is left out:
    SQLite's ``%`` casts ``0.5`` to the integer 0 first."""

    while isinstance(node, (exp.Paren, exp.Neg)):
        node = node.this
    return isinstance(node, exp.Literal) and not node.is_string and node.this.isdigit() and int(node.this) != 0


def temporal_target(to: exp.DataType | None) -> bool:
    return to is not None and to.is_type(*_TEMPORAL)


def _is_date(node: exp.Expression | None, declared: Callable[[exp.Column], str | None]) -> bool:
    """Whether ``node`` is a date or a date and time whenever it is not NULL."""

    if node is None:
        return False
    if isinstance(node, exp.Paren):
        return _is_date(node.this, declared)
    if isinstance(node, exp.Column):
        return date_witness.kind(declared(node)) is not None
    if isinstance(node, exp.Cast):
        return not isinstance(node, exp.TryCast) and node.args["to"].is_type(*_MOMENTS)
    if isinstance(node, exp.Literal):
        return node.is_string and (date_witness.holds(node.this, date_witness.DATE) or date_witness.holds(node.this, date_witness.STAMP))
    if isinstance(node, _NOW):
        return True
    if isinstance(node, _PICKS):
        return isinstance(node.this, exp.Expression) and not node.args.get("expressions") and _is_date(node.this, declared)
    if isinstance(node, exp.Coalesce):
        return all(_is_date(branch, declared) for branch in [node.this, *node.expressions])
    if isinstance(node, exp.If):
        return node.args.get("false") is not None and _is_date(node.args.get("true"), declared) and _is_date(node.args.get("false"), declared)
    if isinstance(node, exp.Case):
        results = [branch.args.get("true") for branch in node.args.get("ifs") or []]
        if node.args.get("default") is not None:
            results.append(node.args["default"])
        return bool(results) and all(_is_date(result, declared) for result in results)
    return False


def mysql_cast_may_be_null(cast: exp.Cast, declared: Callable[[exp.Column], str | None]) -> bool:
    """Whether MySQL can return NULL for ``cast`` when its operand is not NULL.

    Only a cast to a date or time type can, and only of a value that is not one already: a column declared as a date
    or a date and time, a cast to one, a string written as a day or as a day and a time, the current date, or a
    ``MIN``, ``MAX``, ``COALESCE``, ``IF`` or ``CASE`` over those. ``declared`` gives the declared type of a column,
    None when it is not known (the cast may then be NULL)."""

    to = cast.args.get("to")
    if not temporal_target(to):
        return False
    if to.is_type(exp.DataType.Type.TIME):
        operand = cast.this.this if isinstance(cast.this, exp.Paren) else cast.this
        return not (isinstance(operand, exp.Column) and date_witness.kind(declared(operand)) is not None)
    return not _is_date(cast.this, declared)
