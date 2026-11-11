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
"""

from __future__ import annotations

from sqlglot import exp

from . import smt_values

_DEPTH = 12


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


def _output(query: exp.Expression, name: str, types, depth: int) -> bool:
    """Whether output column ``name`` of ``query`` is certainly INT64."""

    query = _unwrap(query)
    if isinstance(query, exp.SetOperation):
        first = query
        while isinstance(first, exp.SetOperation):
            first = _unwrap(first.this)
        if not isinstance(first, exp.Select) or query.args.get("by_name"):
            return False
        names = [item.alias_or_name.lower() for item in first.expressions]
        if names.count(name) != 1 or any(isinstance(item, exp.Star) for item in first.expressions):
            return False
        return _position(query, names.index(name), types, depth)
    if not isinstance(query, exp.Select):
        return False
    items = [item for item in query.expressions if item.alias_or_name.lower() == name]
    if len(items) != 1 or any(isinstance(item, exp.Star) or item.find(exp.Star) is not None and not item.find(exp.Count) for item in query.expressions):
        return False
    item = items[0]
    return _value(item.this if isinstance(item, exp.Alias) else item, query, types, depth)


def _position(query: exp.Expression, index: int, types, depth: int) -> bool:
    query = _unwrap(query)
    if isinstance(query, exp.SetOperation):
        return _position(query.this, index, types, depth) and _position(query.expression, index, types, depth)
    if not isinstance(query, exp.Select) or index >= len(query.expressions):
        return False
    if any(isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)) for item in query.expressions):
        return False
    item = query.expressions[index]
    return _value(item.this if isinstance(item, exp.Alias) else item, query, types, depth)


def _column(column: exp.Column, select: exp.Select, types, depth: int) -> bool:
    if isinstance(column.this, exp.Star) or column.args.get("db") or column.args.get("catalog"):
        return False
    from_ = select.args.get("from_") or select.args.get("from")
    sources = ([from_.this] if from_ is not None else []) + [join.this for join in select.args.get("joins") or []]
    name, qualifier = column.name.lower(), column.table.lower()
    owners = []
    for source in sources:
        if not isinstance(source, (exp.Table, exp.Subquery)):
            return False
        if qualifier and source.alias_or_name.lower() != qualifier:
            continue
        if source.args.get("pivots") or source.args.get("sample"):
            return False
        if isinstance(source, exp.Table):
            body = None if source.args.get("db") else _cte_body(source.name.lower(), select)
            if body is None:
                key = ".".join(part.name for part in source.parts).lower()
                declared = types.get(key)
                if declared is None:
                    return False  # an undeclared table may own the column
                if name in declared:
                    owners.append(smt_values.type_class(declared[name]) == smt_values.INT64)
                continue
        else:
            if source.args["alias"].columns if source.args.get("alias") is not None else False:
                return False
            body = source.this
        outputs = _unwrap(body)
        while isinstance(outputs, exp.SetOperation):
            outputs = _unwrap(outputs.this)
        if not isinstance(outputs, exp.Select) or any(isinstance(i, exp.Star) or (isinstance(i, exp.Column) and isinstance(i.this, exp.Star)) for i in outputs.expressions):
            return False
        if name in [item.alias_or_name.lower() for item in outputs.expressions]:
            owners.append(_output(body, name, types, depth - 1))
    return len(owners) == 1 and owners[0]


def _value(node: exp.Expression, select: exp.Select, types, depth: int) -> bool:
    if depth <= 0:
        return False
    if isinstance(node, exp.Paren):
        return _value(node.this, select, types, depth)
    if isinstance(node, exp.Literal):
        return not node.is_string and smt_values.is_integer_text(node.this)
    if isinstance(node, exp.Column):
        return _column(node, select, types, depth)
    if isinstance(node, exp.Neg):
        return _value(node.this, select, types, depth)
    if type(node) in (exp.Add, exp.Sub, exp.Mul):
        return _value(node.this, select, types, depth) and _value(node.expression, select, types, depth)
    if isinstance(node, (exp.Count, exp.CountIf)):
        return True
    if isinstance(node, (exp.Sum, exp.Min, exp.Max)) and isinstance(node.this, exp.Expression):
        inner = node.this.expressions[0] if isinstance(node.this, exp.Distinct) and len(node.this.expressions) == 1 else node.this
        return not isinstance(inner, exp.Distinct) and _value(inner, select, types, depth)
    if isinstance(node, exp.Cast) and not isinstance(node, exp.TryCast):
        return smt_values.type_class(node.args["to"].sql(dialect="bigquery")) == smt_values.INT64
    if isinstance(node, exp.Coalesce):
        return all(_value(part, select, types, depth) for part in [node.this, *node.expressions])
    if isinstance(node, exp.If):
        branches = [node.args.get("true"), node.args.get("false")]
        return all(isinstance(b, exp.Null) or (b is not None and _value(b, select, types, depth)) for b in branches) and branches[0] is not None and not isinstance(branches[0], exp.Null)
    if isinstance(node, exp.Case):
        branches = [branch.args.get("true") for branch in node.args.get("ifs") or []] + ([node.args["default"]] if node.args.get("default") is not None else [])
        typed = [b for b in branches if not isinstance(b, exp.Null)]
        return bool(typed) and all(_value(b, select, types, depth) for b in typed)
    return False


def is_int64(call: exp.Expression, types: dict[str, dict[str, str]]) -> bool:
    """Whether the argument of aggregate ``call`` (a ``SUM``) is certainly INT64. ``types`` is keyed in lower case."""

    if not types or not isinstance(call.this, exp.Expression) or isinstance(call.this, exp.Distinct):
        return False
    select = call.find_ancestor(exp.Select)
    if select is None or not any(call is node for item in select.expressions for node in item.walk()):
        return False  # in ORDER BY, HAVING or QUALIFY a name can be a select alias instead
    try:
        return _value(call.this, select, types, _DEPTH)
    except (AttributeError, KeyError, RecursionError):
        return False
