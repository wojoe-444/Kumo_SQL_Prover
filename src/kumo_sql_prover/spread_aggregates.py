"""``STDDEV`` and ``VARIANCE`` read as a function of three sums.

The sample standard deviation of the non-NULL values ``x`` of a group is

    sqrt((SUM(x * x) - SUM(x) * SUM(x) / COUNT(x)) / (COUNT(x) - 1))

and NULL when fewer than two values are present; the population form divides by ``COUNT(x)`` and the variances
leave out the root. Each is therefore one fixed function of ``SUM(x * x)``, ``SUM(x)`` and ``COUNT(x)``, which
all skip the rows where ``x`` is NULL, as the aggregate does. The prover has no model of these aggregates and
declined every query that holds one (``unsupported: aggregate STDDEV_SAMP(..)``). This module writes

    STDDEV_SAMP(x)   as   KUMOSQL_STDDEV_SAMP(SUM(x * x), SUM(x), COUNT(x))

and likewise ``STDDEV_POP``, ``VAR_SAMP`` and ``VAR_POP``. The outer call is a name the solver knows nothing
about: it only assumes that equal arguments give equal results. So two queries agree on the aggregate when they
agree on the three sums, which is what the prover can already show (the same rows of the same group, a join
written another way, a filter moved). Nothing is assumed about the value: not that a deviation is zero for a
constant, not that it is positive, not how ``STDDEV_SAMP`` relates to ``VAR_SAMP``.

One reading is assumed and recorded (``ASSUMPTION``): the engine computes the aggregate from the exact sums.
PostgreSQL does for integer and ``NUMERIC`` arguments; for floating-point arguments every engine rounds, and
two plans may round differently, as with ``SUM`` and ``AVG`` (already a base assumption of the prover).

Left alone:

* a window call (``STDDEV(x) OVER (..)``) and a call with ``FILTER``, ``DISTINCT``, ``ORDER BY`` or more than
  one argument. ``DISTINCT x`` is not ``DISTINCT x * x``: 2 and -2 are two values with one square;
* an argument that is not built from columns, numbers, ``+ - * /``, ``CAST``, ``COALESCE`` and ``CASE`` alone
  (a subquery or a function call is written three times here, and need not read the same value each time);
* bare ``STDDEV`` and ``VARIANCE`` outside PostgreSQL, BigQuery and DuckDB, where the name may mean the
  population form (MySQL).
"""

from __future__ import annotations

from sqlglot import exp

ASSUMPTION = "STDDEV and VARIANCE are computed from the exact sums of their argument (no floating-point rounding)"
PREFIX = "KUMOSQL_"

_SAMPLE_BY_DEFAULT = {"postgres", "bigquery", "duckdb"}
_PLAIN = (
    exp.Column, exp.Identifier, exp.Literal, exp.Paren, exp.Neg, exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Cast, exp.DataType,
    exp.DataTypeParam, exp.Coalesce, exp.Case, exp.If, exp.Null, exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.And, exp.Or,
    exp.Not, exp.Is, exp.Boolean,
)


def _name(node: exp.Expression, dialect: str) -> str | None:
    kind = type(node)
    if kind is exp.StddevSamp:
        return "STDDEV_SAMP"
    if kind is exp.StddevPop:
        return "STDDEV_POP"
    if kind is exp.VariancePop:
        return "VAR_POP"
    if kind is exp.Variance:  # VAR_SAMP and VARIANCE are one class in sqlglot
        return "VAR_SAMP" if dialect in _SAMPLE_BY_DEFAULT else None
    if kind is exp.Stddev:
        return "STDDEV_SAMP" if dialect in _SAMPLE_BY_DEFAULT else None
    return None


def _plain_argument(arg: exp.Expression | None) -> bool:
    if not isinstance(arg, exp.Expression) or isinstance(arg, exp.Star):
        return False
    if any(isinstance(c.this, exp.Star) for c in arg.find_all(exp.Column)):
        return False
    return all(isinstance(node, _PLAIN) for node in arg.walk())


def rewrite_spread_aggregates(tree: exp.Expression, dialect: str, assumptions: set[str] | None = None) -> exp.Expression:
    """Write each plain ``STDDEV``/``VARIANCE`` call of ``tree`` as an uninterpreted function of three sums."""

    for node in list(tree.find_all(exp.Stddev, exp.StddevSamp, exp.StddevPop, exp.Variance, exp.VariancePop)):
        name = _name(node, dialect)
        if name is None or isinstance(node.parent, (exp.Window, exp.Filter, exp.WithinGroup, exp.Order)):
            continue
        if any(value for key, value in node.args.items() if key != "this") or not _plain_argument(node.this):
            continue
        arg = node.this
        squares = exp.Sum(this=exp.Mul(this=exp.Paren(this=arg.copy()), expression=exp.Paren(this=arg.copy())))
        call = exp.Anonymous(this=PREFIX + name, expressions=[squares, exp.Sum(this=arg.copy()), exp.Count(this=arg.copy())])
        if node is tree:
            tree = call
        else:
            node.replace(call)
        if assumptions is not None:
            assumptions.add(ASSUMPTION)
    return tree
