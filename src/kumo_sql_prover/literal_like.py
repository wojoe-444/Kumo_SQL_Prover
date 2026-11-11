"""``x LIKE 'abc'`` is ``x = 'abc'`` when the pattern has no wildcard.

A ``LIKE`` pattern without ``%``, ``_`` or a backslash matches exactly one string, itself, so the test is an
equality: NULL when ``x`` is NULL on both sides, TRUE for the same values otherwise. The prover reads ``LIKE`` as
an opaque test, so it could not see that ``sname LIKE 'abc'`` and ``sname = 'abc'`` keep the same rows.

The two are the same test only where ``LIKE`` and ``=`` compare strings the same way, so the rule is limited to:

* **BigQuery** and **DuckDB**: both compare byte for byte (and BigQuery applies a column's collation to both);
* **PostgreSQL**, when the left side is a column declared ``VARCHAR`` or ``TEXT``. For ``CHAR(n)`` the two treat
  the padding differently, so an undeclared column is left alone.

It is never applied to MySQL (``=`` ignores trailing spaces under a PAD SPACE collation and ``LIKE`` does not) or
SQLite (``LIKE`` ignores ASCII case and ``=`` does not), to ``ILIKE``, to a pattern with an ``ESCAPE`` clause, or to
a pattern that is not a string literal. ``NOT LIKE`` follows, as the negation of the same test.
"""

from __future__ import annotations

import re

from sqlglot import exp

_ANY_STRING = ("bigquery", "duckdb")
_DECLARED_STRING = ("postgres",)
_VARYING = re.compile(r"^\s*(varchar|text|string|character varying)\b", re.I)
_WILDCARDS = set("%_\\")


def _own(node: exp.Expression, select: exp.Select) -> bool:
    return node.find_ancestor(exp.Select) is select


def _declared_varying(column: exp.Expression, select: exp.Select, types: dict[str, dict[str, str]] | None) -> bool:
    """Whether ``column`` is a column of a base table of ``select`` whose declared type is VARCHAR or TEXT."""

    if not types or not isinstance(column, exp.Column) or isinstance(column.this, exp.Star):
        return False
    from_ = select.args.get("from_") or select.args.get("from")
    sources = ([from_.this] if from_ is not None else []) + [join.this for join in select.args.get("joins") or []]
    ctes = {cte.alias.lower() for cte in select.root().find_all(exp.CTE) if cte.alias}
    owners = []
    for source in sources:
        if not isinstance(source, exp.Table) or source.args.get("db") or source.name.lower() in ctes:
            if column.table and source.alias_or_name.lower() == column.table.lower():
                return False  # a derived table or CTE: its column types are not declared
            if not column.table:
                return False  # a bare column may come from it
            continue
        declared = types.get(source.name.lower(), {})
        if column.table and source.alias_or_name.lower() != column.table.lower():
            continue
        if column.name.lower() in declared:
            owners.append(declared[column.name.lower()])
    return len(owners) == 1 and bool(_VARYING.match(owners[0]))


def literal_like_to_equality(select: exp.Select, types: dict[str, dict[str, str]] | None, dialect: str = "bigquery") -> exp.Expression | None:
    """``select`` with each wildcard-free ``LIKE`` of its own scope written as ``=``; ``None`` if nothing changed."""

    if dialect not in _ANY_STRING + _DECLARED_STRING:
        return None
    found = [
        node for node in select.find_all(exp.Like)
        if type(node) is exp.Like and _own(node, select) and node.args.get("escape") is None
        and isinstance(node.expression, exp.Literal) and node.expression.is_string and not set(node.expression.this) & _WILDCARDS
        and not isinstance(node.parent, exp.Escape)
        and (dialect in _ANY_STRING or _declared_varying(node.this, select, types))
    ]
    if not found:
        return None
    copy = select.copy()
    changed = False
    for node in list(copy.find_all(exp.Like)):
        if type(node) is not exp.Like or not _own(node, copy) or node.args.get("escape") is not None or isinstance(node.parent, exp.Escape):
            continue
        pattern = node.expression
        if not (isinstance(pattern, exp.Literal) and pattern.is_string) or set(pattern.this) & _WILDCARDS:
            continue
        if dialect not in _ANY_STRING and not _declared_varying(node.this, copy, types):
            continue
        node.replace(exp.EQ(this=node.this.copy(), expression=pattern.copy()))
        changed = True
    return copy if changed else None
