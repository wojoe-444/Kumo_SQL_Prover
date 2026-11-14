"""The numeric type of a column of a derived relation that the SMT prover keeps whole.

A grouped derived table is not compiled: the prover names it by its text and reads its columns as
unknowns. Their types were unknown too, and BigQuery reads a decimal literal by the type of the value
next to it: ``1.3`` is the exact NUMERIC 1.3 beside a NUMERIC and the nearest FLOAT64 beside an INT64
or a FLOAT64. So in

    SELECT .. FROM item, (SELECT cs_item_sk AS k, AVG(cs_ext_discount_amt) AS a FROM catalog_sales GROUP BY 1) AS d
    WHERE i_item_sk = d.k AND i_current_price > 1.3 * d.a

the literal had an unknown reading, one unknown for each query of a pair, and two queries that
differ in anything else (an alias, ``BETWEEN`` against two comparisons) were never proved equal. It
is the shape every decorrelated ``x > (SELECT 1.3 * AVG(y) ..)`` takes, so TPC-H and DSB rewrites
ended there. ``2 * d.a`` was unaffected: an integer literal reads the same in every type.

This module reads the type where it is certain: an output that is a column of a base table with a
declared type, or ``SUM``, ``MIN``, ``MAX``, ``AVG`` or ``COUNT`` of one. BigQuery's rules for these
are fixed: ``SUM``, ``MIN`` and ``MAX`` keep the type of their argument, ``AVG`` of an INT64 or a
FLOAT64 is a FLOAT64 and of a NUMERIC or BIGNUMERIC keeps it, ``COUNT`` is an INT64. A column of a
nested derived table has the type of the output it names, read the same way (the average of per-group
sums in ``SELECT k, AVG(s) FROM (SELECT k, c, SUM(x) AS s FROM t GROUP BY k, c) GROUP BY k``).
Anything else (an expression, a name two sources could supply, a set operation, a window) has no
answer here and stays an unknown, as before.
"""

from __future__ import annotations

from sqlglot import exp

from . import smt_values

_NUMERIC = (smt_values.INT64, smt_values.NUMERIC, smt_values.BIGNUMERIC, smt_values.FLOAT64)
_MAX_DEPTH = 8  # derived tables followed inwards


def _sources(select: exp.Select) -> dict[str, str | exp.Select] | None:
    """Alias to base table, or to the select of a derived table, for every source of ``select``. ``None`` when a
    source is anything else, or when a ``WITH`` could give a table name another meaning."""

    from_ = select.args.get("from") or select.args.get("from_")
    if from_ is None or select.args.get("with_") or select.args.get("with"):
        return None
    out: dict[str, str | exp.Select] = {}
    for source in [from_.this, *(join.this for join in select.args.get("joins") or [])]:
        if isinstance(source, exp.Table) and not (source.args.get("db") or source.args.get("catalog")):
            found: str | exp.Select = source.name.lower()
        elif isinstance(source, exp.Subquery) and isinstance(source.this, exp.Select) and source.alias:
            found = source.this
        else:
            return None
        if source.args.get("pivots") or source.args.get("sample"):
            return None
        alias = source.alias_or_name.lower()
        if not alias or alias in out:
            return None
        out[alias] = found
    return out


def _columns(source: str | exp.Select, schema: dict[str, list[str]]) -> list[str] | None:
    if isinstance(source, str):
        return schema.get(source)
    names = [item.alias_or_name.lower() for item in source.expressions]
    if any(isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)) for item in source.expressions):
        return None
    return names if "" not in names and len(set(names)) == len(names) else None


def _column_class(column: exp.Column, sources: dict[str, str | exp.Select], schema, types, depth: int) -> str | None:
    name = column.name.lower()
    if column.table:
        alias = column.table.lower()
        if alias not in sources:
            return None
    else:
        # A bare name is this select's own column only when every source lists its columns and exactly one has it.
        listed = {alias: _columns(source, schema) for alias, source in sources.items()}
        if any(columns is None for columns in listed.values()):
            return None
        holders = [alias for alias, columns in listed.items() if name in columns]
        if len(holders) != 1:
            return None
        alias = holders[0]
    source = sources[alias]
    if isinstance(source, str):
        cls = smt_values.type_class(types.get(source, {}).get(name))
        return cls if cls in _NUMERIC else None
    names = _columns(source, schema)
    if names is None or name not in names or depth >= _MAX_DEPTH:
        return None
    return _output_classes(source, schema, types, depth + 1)[names.index(name)]


def _class(e: exp.Expression, sources, schema, types, depth: int) -> str | None:
    if isinstance(e, exp.Paren):
        return _class(e.this, sources, schema, types, depth)
    if isinstance(e, exp.Column):
        return None if isinstance(e.this, exp.Star) else _column_class(e, sources, schema, types, depth)
    if isinstance(e, (exp.Count, exp.CountIf)):
        return None if isinstance(e.parent, exp.Window) else smt_values.INT64
    if isinstance(e, (exp.Sum, exp.Min, exp.Max, exp.Avg)):
        if not isinstance(e.this, exp.Expression) or isinstance(e.parent, exp.Window):
            return None
        inner = e.this.expressions[0] if isinstance(e.this, exp.Distinct) and len(e.this.expressions) == 1 else e.this
        if not isinstance(inner.unnest() if isinstance(inner, exp.Paren) else inner, exp.Column):
            return None
        cls = _class(inner, sources, schema, types, depth)
        if isinstance(e, exp.Avg) and cls in (smt_values.INT64, smt_values.FLOAT64):
            return smt_values.FLOAT64
        return cls
    return None


def _output_classes(select: exp.Select, schema, types, depth: int) -> list[str | None]:
    outputs = [(item.this if isinstance(item, exp.Alias) else item) for item in select.expressions]
    sources = _sources(select)
    if sources is None:
        return [None] * len(outputs)
    return [_class(value, sources, schema, types, depth) for value in outputs]


def output_classes(select: exp.Select, schema: dict[str, list[str]] | None, types: dict[str, dict[str, str]] | None) -> list[str | None]:
    """For each output of ``select`` (a derived relation's body, its CTEs expanded): INT64, NUMERIC, BIGNUMERIC or
    FLOAT64 when the declared types fix it, ``None`` otherwise. ``schema`` and ``types`` are keyed by lower-case
    table and column."""

    if not types:
        return [None] * len(select.expressions)
    return _output_classes(select, schema or {}, types, 0)
