"""Inline rows written as ``UNNEST`` of an array of ``STRUCT`` literals are a derived table.

sqlglot writes another engine's ``(VALUES (1, 2), (1, NULL)) AS t(a, b)`` in GoogleSQL as

    UNNEST([STRUCT(1 AS a, 2 AS b), STRUCT(1 AS a, NULL AS b)]) AS t

which is how GoogleSQL spells inline rows. The provers model ``UNNEST`` only as a cross join with an unknown
array, so every pair that joins inline rows (most of Trino's outer-join tests) was declined: "UNNEST outside a
cross join", "field t.b of an UNNEST element", "cannot resolve column a without a schema".

The array is a literal that reads no column, so the ``UNNEST`` is a relation with one row per element whatever
row of another source it is joined with, in any join position (a comma, ``JOIN .. ON``, either side of an outer
join). An element is a struct, and ``UNNEST`` of an array of structs is a table whose columns are the struct's
fields: ``SELECT *`` returns them, ``t.a`` and a bare ``a`` read them. That is the relation

    (SELECT 1 AS a, 2 AS b UNION ALL SELECT 1 AS a, NULL AS b) AS t

and the rewrite puts it there. The rows come out in no particular order either way (no ``WITH OFFSET``).

So that the array and the ``UNION ALL`` cannot disagree about a column's type, every value of one field must be
of one kind (integer literals and ``CAST(.. AS INT64)``, or decimal literals, or strings, or booleans, or one other
cast type), NULLs aside: neither form then converts anything, and an all-NULL field is INT64 in both.

Declined (the ``UNNEST`` stays as written): another dialect; ``WITH OFFSET``; more than one array; an array
that is not a literal, is empty, or has more than ``_MAX_ROWS`` elements; an element that is not a ``STRUCT`` with
every field named; structs whose field names differ (in spelling, order or number) or repeat; a field value that
is anything but a literal, NULL, a boolean, a negative number or a cast of one of those to a plain type (so no
column, no subquery, no function); a field with values of more than one kind; an alias with a column list; an
alias that is also the name of one of the fields; a query that reads the alias as a value (a bare ``t``, which is
the whole struct) or through a longer path (``t.a.b``); an ``UNNEST`` that is not a ``FROM`` or ``JOIN`` source
(``x IN UNNEST(..)``, a select-list ``UNNEST``).
"""

from __future__ import annotations

import re

from sqlglot import exp

_MAX_ROWS = 64
_INTEGER_TYPES = {exp.DataType.Type.BIGINT, exp.DataType.Type.INT}
_DIGITS = re.compile(r"[0-9]+")


def _literal(node: exp.Expression) -> bool:
    return isinstance(node, (exp.Literal, exp.Null, exp.Boolean)) or (
        isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal) and not node.this.is_string
    )


def _kind(node: exp.Expression) -> str | None:
    """The kind of a field value: ``null``, ``int``, ``decimal``, ``string``, ``bool`` or ``cast:<type>``; None if not allowed."""

    if isinstance(node, exp.Cast) and type(node) is exp.Cast:
        to = node.args.get("to")
        if not isinstance(to, exp.DataType) or to.expressions or any(v for k, v in node.args.items() if k not in ("this", "to")) or not _literal(node.this):
            return None
        return "int" if to.this in _INTEGER_TYPES and _kind(node.this) in ("int", "null") else f"cast:{to.sql(dialect='bigquery').upper()}"
    if not _literal(node):
        return None
    if isinstance(node, exp.Null):
        return "null"
    if isinstance(node, exp.Boolean):
        return "bool"
    number = node.this if isinstance(node, exp.Neg) else node
    if number.is_string:
        return "string"
    return "int" if _DIGITS.fullmatch(number.this) else "decimal"


def _fields(unnest: exp.Unnest) -> list[list[tuple[exp.Identifier, exp.Expression]]] | None:
    """The rows as ``[(field name, value)]`` lists, or None when the array is not inline rows as described above."""

    arrays = unnest.expressions
    if len(arrays) != 1 or type(arrays[0]) is not exp.Array or unnest.args.get("offset"):
        return None
    items = arrays[0].expressions
    if not items or len(items) > _MAX_ROWS:
        return None
    rows = []
    for item in items:
        if type(item) is not exp.Struct or not item.expressions:
            return None
        row = []
        for field in item.expressions:
            if not isinstance(field, exp.PropertyEQ) or not isinstance(field.this, exp.Identifier) or not field.this.name:
                return None
            row.append((field.this, field.expression))
        rows.append(row)
    names = [name.name.lower() for name, _ in rows[0]]
    if len(set(names)) != len(names) or any([name.name.lower() for name, _ in row] != names for row in rows):
        return None
    for position in range(len(names)):
        kinds = {_kind(row[position][1]) for row in rows} - {"null"}
        if None in kinds or len(kinds) > 1:
            return None
    return rows


def _alias(unnest: exp.Unnest) -> tuple[bool, str]:
    """(whether the alias is a form the rewrite reads, the alias or "")."""

    alias = unnest.args.get("alias")
    if alias is None:
        return True, ""
    if not isinstance(alias, exp.TableAlias):
        return False, ""
    columns = alias.args.get("columns") or []
    if alias.this is not None and columns or len(columns) > 1:
        return False, ""
    name = alias.name or (columns[0].name if columns else "")
    return bool(name), name


def _read_as_value(tree: exp.Expression, alias: str, names: set[str]) -> bool:
    """Whether some column in ``tree`` reads ``alias`` as the struct itself or through a path the rewrite would change."""

    lowered = alias.lower()
    for column in tree.find_all(exp.Column):
        parts = [part.name.lower() for part in column.parts]
        if len(parts) == 1 and parts[0] == lowered:
            return True
        if len(parts) > 2 and lowered in parts:
            return True
        if len(parts) == 2 and parts[0] == lowered and not isinstance(column.this, exp.Star) and parts[1] not in names:
            return True
    return any(isinstance(dot, exp.Dot) and any(isinstance(c, exp.Column) and lowered in [p.name.lower() for p in c.parts] for c in dot.find_all(exp.Column)) for dot in tree.find_all(exp.Dot))


def inline_struct_rows(tree: exp.Expression, dialect: str = "bigquery") -> exp.Expression:
    """``tree`` with each ``UNNEST([STRUCT(..), ..])`` source written as a derived table of its rows (see the module doc)."""

    if dialect != "bigquery" or not any(isinstance(n, exp.Unnest) for n in tree.find_all(exp.Unnest)):
        return tree
    targets = []
    for unnest in tree.find_all(exp.Unnest):
        if not isinstance(unnest.parent, (exp.From, exp.Join)) or unnest.arg_key != "this":
            continue
        rows = _fields(unnest)
        ok, alias = _alias(unnest)
        if rows is None or not ok:
            continue
        names = {name.name.lower() for name, _ in rows[0]}
        if alias and (alias.lower() in names or _read_as_value(tree, alias, names)):
            continue
        targets.append((unnest, rows, alias))
    if not targets:
        return tree
    for unnest, rows, alias in targets:
        body: exp.Expression | None = None
        for row in rows:
            select = exp.Select(expressions=[exp.Alias(this=value.copy(), alias=exp.Identifier(this=name.name, quoted=bool(name.args.get("quoted")))) for name, value in row])
            body = select if body is None else exp.Union(this=body, expression=select, distinct=False)
        unnest.replace(exp.Subquery(this=body, alias=exp.TableAlias(this=exp.to_identifier(alias)) if alias else None))
    return tree
