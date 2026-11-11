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
FLOAT64 is a FLOAT64 and of a NUMERIC or BIGNUMERIC keeps it, ``COUNT`` is an INT64. Anything else
(an expression, a column of a nested derived table, a name two tables could supply, a set operation)
has no answer here and stays an unknown, as before.
"""

from __future__ import annotations

from sqlglot import exp

from . import smt_values

_NUMERIC = (smt_values.INT64, smt_values.NUMERIC, smt_values.BIGNUMERIC, smt_values.FLOAT64)


def _tables(select: exp.Select) -> dict[str, str] | None:
    """Alias to base table for every source of ``select``, or ``None`` when one of them is not a plain table."""

    sources = []
    from_ = select.args.get("from") or select.args.get("from_")
    if from_ is None:
        return None
    sources.append(from_.this)
    sources.extend(join.this for join in select.args.get("joins") or [])
    out: dict[str, str] = {}
    for source in sources:
        if not isinstance(source, exp.Table) or source.args.get("db") or source.args.get("catalog"):
            return None
        alias = source.alias_or_name.lower()
        if alias in out:
            return None
        out[alias] = source.name.lower()
    return out


def _column_class(column: exp.Column, tables: dict[str, str], schema: dict[str, list[str]], types: dict[str, dict[str, str]]) -> str | None:
    name = column.name.lower()
    if column.table:
        table = tables.get(column.table.lower())
        if table is None:
            return None
    else:
        # A bare name is this select's own column only when every source lists its columns and exactly one has it.
        if any(t not in schema for t in tables.values()):
            return None
        holders = [t for t in tables.values() if name in schema[t]]
        if len(holders) != 1 or list(tables.values()).count(holders[0]) != 1:
            return None
        table = holders[0]
    cls = smt_values.type_class(types.get(table, {}).get(name))
    return cls if cls in _NUMERIC else None


def _class(e: exp.Expression, tables, schema, types) -> str | None:
    if isinstance(e, exp.Paren):
        return _class(e.this, tables, schema, types)
    if isinstance(e, exp.Column):
        return None if isinstance(e.this, exp.Star) else _column_class(e, tables, schema, types)
    if isinstance(e, (exp.Count, exp.CountIf)):
        return smt_values.INT64
    if isinstance(e, (exp.Sum, exp.Min, exp.Max, exp.Avg)):
        if e.args.get("distinct") or not isinstance(e.this, exp.Expression) or isinstance(e.parent, exp.Window):
            return None
        inner = e.this.expressions[0] if isinstance(e.this, exp.Distinct) and len(e.this.expressions) == 1 else e.this
        if not isinstance(inner.unnest() if isinstance(inner, exp.Paren) else inner, exp.Column):
            return None
        cls = _class(inner, tables, schema, types)
        if isinstance(e, exp.Avg) and cls in (smt_values.INT64, smt_values.FLOAT64):
            return smt_values.FLOAT64
        return cls
    return None


def output_classes(select: exp.Select, schema: dict[str, list[str]] | None, types: dict[str, dict[str, str]] | None) -> list[str | None]:
    """For each output of ``select`` (a derived relation's body): INT64, NUMERIC, BIGNUMERIC or FLOAT64 when the
    declared types fix it, ``None`` otherwise. ``schema`` and ``types`` are keyed by lower-case table and column."""

    outputs = [(item.this if isinstance(item, exp.Alias) else item) for item in select.expressions]
    tables = _tables(select) if types else None
    if tables is None:
        return [None] * len(outputs)
    return [_class(value, tables, schema or {}, types or {}) for value in outputs]
