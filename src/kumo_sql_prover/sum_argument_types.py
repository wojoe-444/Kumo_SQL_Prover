"""Whether a ``SUM`` argument is certainly INT64, read from the query text and the declared column types.

The SMT prover drops "SUM and AVG are treated as independent of row order" from a proof when every ``SUM`` adds an
exact type (``float_sum_order``). It sees the type of a base-table column, a literal, a cast and arithmetic over
those, but a column of a derived table it keeps whole (a ``UNION ALL``, say) has no visible type, so
``SELECT SUM(v) FROM (SELECT x AS v FROM t UNION ALL SELECT y AS v FROM t)`` kept the assumption although both
branches are INT64 and an INT64 sum is exact in any order.

:func:`is_int64` follows such a column to where it is defined: through a derived table or CTE to the select item of
that name, and through a set operation to the item at the same position of every branch. It answers ``True`` only
when every path ends in a column declared INT64, an integer literal, ``COUNT``, a cast to INT64, or ``+``, ``-``,
``*``, ``SUM``, ``MIN``, ``MAX``, ``COALESCE`` and ``IF``/``CASE`` over such values. Anything it cannot place (a star,
an undeclared table, a bare column two sources could own, a column of an outer query) is ``False``: the assumption
then stays, which is the answer the prover gave before.

:func:`is_exact` answers the wider question the float regrouping check asks: whether a ``SUM`` adds only values
that can be added in any grouping without rounding. That is INT64 as above, and with ``decimals`` also a fixed-point
type (``NUMERIC``, ``DECIMAL``): adding or subtracting two of them is exact, and so is a product when the decimal
places of its factors add up to at most nine (the finest scale a BigQuery ``NUMERIC`` keeps; a product with more
places is rounded back). A decimal literal counts only next to a declared decimal value, because beside an INT64 it
is a FLOAT64 in BigQuery. A division is never exact.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from sqlglot import exp

from . import smt_values

_DEPTH = 12
NUMERIC_PLACES = 9  # a product of fixed-point values is exact up to this many decimal places
_UNBOUNDED = 10**6  # a declared decimal type without a scale: any number of places


class _Context(NamedTuple):
    types: dict
    decimals: bool


class _Number(NamedTuple):
    """A value that adds exactly: ``places`` decimal places at most, ``decimal`` when a declared decimal type is in it."""

    places: int
    decimal: bool


_INTEGER = _Number(0, False)


def _places(declared: str) -> int:
    match = re.search(r"\(\s*\d+\s*,\s*(\d+)\s*\)", declared)
    return int(match.group(1)) if match else _UNBOUNDED


def _unwrap(query: exp.Expression) -> exp.Expression:
    while isinstance(query, exp.Subquery) and not query.args.get("alias") or isinstance(query, exp.Paren):
        query = query.this
    return query.this if isinstance(query, exp.Subquery) else query


def _cte_body(name: str, node: exp.Expression) -> exp.Expression | None:
    """The nearest ``WITH`` definition called ``name`` around ``node``."""

    ancestor, inside = node, None
    while ancestor is not None:
        if isinstance(ancestor, exp.CTE):
            inside = ancestor  # a definition sees only the definitions before it
        with_ = ancestor.args.get("with_") or ancestor.args.get("with")
        if with_ is not None:
            if with_.args.get("recursive"):
                raise KeyError(name)
            visible = with_.expressions
            if any(cte is inside for cte in visible):
                visible = visible[: next(i for i, cte in enumerate(visible) if cte is inside)]
            for cte in visible:
                if cte.alias.lower() == name:
                    if cte.args["alias"].columns:
                        raise KeyError(name)
                    return cte.this
        ancestor = ancestor.parent
    return None


def _join(numbers) -> _Number | None:
    """The common kind of several values (set operation branches, CASE results): ``None`` when one is not exact."""

    numbers = list(numbers)
    if not numbers or any(n is None for n in numbers):
        return None
    return _Number(max(n.places for n in numbers), any(n.decimal for n in numbers))


def _output(query: exp.Expression, name: str, ctx: _Context, depth: int) -> _Number | None:
    """Whether output column ``name`` of ``query`` is certainly exact (an INT64, or a fixed-point value with ``decimals``)."""

    query = _unwrap(query)
    if isinstance(query, exp.SetOperation):
        first = query
        while isinstance(first, exp.SetOperation):
            first = _unwrap(first.this)
        if not isinstance(first, exp.Select) or query.args.get("by_name"):
            return None
        names = [item.alias_or_name.lower() for item in first.expressions]
        if names.count(name) != 1 or any(isinstance(item, exp.Star) for item in first.expressions):
            return None
        return _position(query, names.index(name), ctx, depth)
    if not isinstance(query, exp.Select):
        return None
    items = [item for item in query.expressions if item.alias_or_name.lower() == name]
    if len(items) != 1 or any(isinstance(item, exp.Star) or item.find(exp.Star) is not None and not item.find(exp.Count) for item in query.expressions):
        return None
    item = items[0]
    return _value(item.this if isinstance(item, exp.Alias) else item, query, ctx, depth)


def _position(query: exp.Expression, index: int, ctx: _Context, depth: int) -> _Number | None:
    query = _unwrap(query)
    if isinstance(query, exp.SetOperation):
        return _join([_position(query.this, index, ctx, depth), _position(query.expression, index, ctx, depth)])
    if not isinstance(query, exp.Select) or index >= len(query.expressions):
        return None
    if any(isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)) for item in query.expressions):
        return None
    item = query.expressions[index]
    return _value(item.this if isinstance(item, exp.Alias) else item, query, ctx, depth)


def _declared(declared: str, ctx: _Context) -> _Number | None:
    kind = smt_values.type_class(declared)
    if kind == smt_values.INT64:
        return _INTEGER
    if ctx.decimals and kind in (smt_values.NUMERIC, smt_values.BIGNUMERIC):
        return _Number(_places(declared), True)
    return None


def _column(column: exp.Column, select: exp.Select, ctx: _Context, depth: int) -> _Number | None:
    if isinstance(column.this, exp.Star) or column.args.get("db") or column.args.get("catalog"):
        return None
    from_ = select.args.get("from_") or select.args.get("from")
    sources = ([from_.this] if from_ is not None else []) + [join.this for join in select.args.get("joins") or []]
    name, qualifier = column.name.lower(), column.table.lower()
    owners = []
    for source in sources:
        if not isinstance(source, (exp.Table, exp.Subquery)):
            return None
        if qualifier and source.alias_or_name.lower() != qualifier:
            continue
        if source.args.get("pivots") or source.args.get("sample"):
            return None
        if isinstance(source, exp.Table):
            body = None if source.args.get("db") else _cte_body(source.name.lower(), select)
            if body is None:
                key = ".".join(part.name for part in source.parts).lower()
                declared = ctx.types.get(key)
                if declared is None:
                    return None  # an undeclared table may own the column
                if name in declared:
                    owners.append(_declared(declared[name], ctx))
                continue
        else:
            if source.args["alias"].columns if source.args.get("alias") is not None else False:
                return None
            body = source.this
        outputs = _unwrap(body)
        while isinstance(outputs, exp.SetOperation):
            outputs = _unwrap(outputs.this)
        if not isinstance(outputs, exp.Select) or any(isinstance(i, exp.Star) or (isinstance(i, exp.Column) and isinstance(i.this, exp.Star)) for i in outputs.expressions):
            return None
        if name in [item.alias_or_name.lower() for item in outputs.expressions]:
            owners.append(_output(body, name, ctx, depth - 1))
    return owners[0] if len(owners) == 1 else None


def _decimal_literal(node: exp.Expression) -> _Number | None:
    """A decimal literal such as ``0.25`` (not an exponent form): its places, flagged as a literal by ``decimal=False``."""

    if isinstance(node, exp.Literal) and not node.is_string and re.fullmatch(r"\d*\.\d+", node.this):
        return _Number(len(node.this.split(".")[1]), False)
    return None


def _arithmetic(node: exp.Expression, select: exp.Select, ctx: _Context, depth: int) -> _Number | None:
    left, right = node.this, node.expression
    first, second = _value(left, select, ctx, depth), _value(right, select, ctx, depth)
    if ctx.decimals:
        # A decimal literal is a FLOAT64 unless the other operand is a declared decimal value (BigQuery coerces it).
        if first is None and second is not None and second.decimal:
            first = _decimal_literal(_strip(left))
        elif second is None and first is not None and first.decimal:
            second = _decimal_literal(_strip(right))
    if first is None or second is None:
        return None
    decimal = first.decimal or second.decimal
    if type(node) is not exp.Mul:
        return _Number(max(first.places, second.places), decimal)
    if first.places == 0 or second.places == 0:
        return _Number(max(first.places, second.places), decimal)  # a whole number as a factor adds no places
    places = first.places + second.places
    return _Number(places, decimal) if places <= NUMERIC_PLACES else None  # beyond that a product is rounded


def _strip(node: exp.Expression) -> exp.Expression:
    while isinstance(node, (exp.Paren, exp.Neg)):
        node = node.this
    return node


def _value(node: exp.Expression, select: exp.Select, ctx: _Context, depth: int) -> _Number | None:
    if depth <= 0:
        return None
    if isinstance(node, exp.Paren):
        return _value(node.this, select, ctx, depth)
    if isinstance(node, exp.Literal):
        return _INTEGER if not node.is_string and smt_values.is_integer_text(node.this) else None
    if isinstance(node, exp.Column):
        return _column(node, select, ctx, depth)
    if isinstance(node, exp.Neg):
        return _value(node.this, select, ctx, depth)
    if type(node) in (exp.Add, exp.Sub, exp.Mul):
        return _arithmetic(node, select, ctx, depth)
    if isinstance(node, (exp.Count, exp.CountIf)):
        return _INTEGER
    if isinstance(node, (exp.Sum, exp.Min, exp.Max)) and isinstance(node.this, exp.Expression):
        inner = node.this.expressions[0] if isinstance(node.this, exp.Distinct) and len(node.this.expressions) == 1 else node.this
        return None if isinstance(inner, exp.Distinct) else _value(inner, select, ctx, depth)
    if isinstance(node, exp.Cast) and not isinstance(node, exp.TryCast):
        target = node.args["to"].sql(dialect="bigquery")
        return _declared(target, ctx)
    if isinstance(node, exp.Coalesce):
        return _join(_value(part, select, ctx, depth) for part in [node.this, *node.expressions])
    if isinstance(node, exp.If):
        branches = [node.args.get("true"), node.args.get("false")]
        if branches[0] is None or isinstance(branches[0], exp.Null):
            return None
        return _join(_value(b, select, ctx, depth) for b in branches if b is not None and not isinstance(b, exp.Null))
    if isinstance(node, exp.Case):
        branches = [branch.args.get("true") for branch in node.args.get("ifs") or []] + ([node.args["default"]] if node.args.get("default") is not None else [])
        return _join(_value(b, select, ctx, depth) for b in branches if not isinstance(b, exp.Null))
    return None


def _argument(call: exp.Expression, types: dict[str, dict[str, str]], decimals: bool) -> _Number | None:
    if not types or not isinstance(call.this, exp.Expression) or isinstance(call.this, exp.Distinct):
        return None
    select = call.find_ancestor(exp.Select)
    if select is None or not any(call is node for item in select.expressions for node in item.walk()):
        return None  # in ORDER BY, HAVING or QUALIFY a name can be a select alias instead
    try:
        return _value(call.this, select, _Context(types, decimals), _DEPTH)
    except (AttributeError, KeyError, RecursionError):
        return None


def is_int64(call: exp.Expression, types: dict[str, dict[str, str]]) -> bool:
    """Whether the argument of aggregate ``call`` (a ``SUM``) is certainly INT64. ``types`` is keyed in lower case."""

    return _argument(call, types, False) is not None


def is_exact(call: exp.Expression, types: dict[str, dict[str, str]]) -> bool:
    """Whether aggregate ``call`` adds only values that add without rounding, in any grouping: INT64 as for
    :func:`is_int64`, and fixed-point values (see the module doc). ``types`` is keyed in lower case."""

    return _argument(call, types, isinstance(call, exp.Sum)) is not None
