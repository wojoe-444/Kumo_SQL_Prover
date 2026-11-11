"""A derived table's column that its own ``WHERE`` pins to a whole number is read as that number.

    SELECT g.y, g.s FROM (SELECT y, SUM(x) AS s FROM t WHERE y = 2000 GROUP BY y) AS g JOIN u ON ..
    SELECT 2000 AS y, g.s FROM (SELECT y, SUM(x) AS s FROM t WHERE y = 2000 GROUP BY y) AS g JOIN u ON ..

Every row the derived table returns was built from rows that passed ``y = 2000``, and its output ``y`` is that
column itself, so ``g.y`` is 2000 on every row. An optimizer that inlines a CTE read under ``y = 2000`` writes
the literal (Calcite: ``CAST(2000 AS INTEGER) AS d_year``) where the query read the column; with the filter
pushed into the grouped derived table on both sides, this rule gives both the literal.

Read only when all of these hold:

* the derived table is one ``SELECT`` (no set operation) with a ``WHERE`` whose top-level ``AND`` part is
  ``column = n`` or ``n = column``, ``n`` a whole number written in digits, and one of its outputs is that same
  column, written the same way. No ``ROLLUP``, ``CUBE`` or grouping sets (their subtotal rows hold NULL there),
  and no aggregate without ``GROUP BY`` (such a select returns a row for no input);
* the column is declared an integer type in the types given to the prover and belongs to a base table the
  derived table reads. For any other type the comparison can hold for a value that is not written ``n``
  (``2000.00``, a padded ``CHAR``, a case-insensitive collation), and an integer literal divides differently
  from a ``NUMERIC``; without types nothing is read;
* the derived table is this select's ``FROM`` item or the right side of a plain, ``INNER`` or ``CROSS`` join,
  under an alias no other source shares and without an alias column list, and the select has no ``RIGHT`` or
  ``FULL`` join (which would pad it with NULLs) and no extended grouping (whose subtotal rows read NULL);
* the read is ``alias.column`` in the select's own select list, ``WHERE``, ``HAVING`` or a join's ``ON``.
  Reads in ``GROUP BY`` and ``ORDER BY`` stay, because a number there is a position; reads inside a subquery
  stay too.

A select-list item that is the column alone keeps its name (``2000 AS y``).
"""

from __future__ import annotations

import re

from sqlglot import exp

from .ast_utils import extended_grouping, select_sources, visible_ctes

_INTEGER = {"INT", "INTEGER", "BIGINT", "INT64", "INT32", "SMALLINT", "INT16", "TINYINT", "INT8"}
_JOIN_KINDS = {"", "INNER", "CROSS"}
_LIMIT = 2**31


def _conjuncts(node: exp.Expression) -> list[exp.Expression]:
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _whole_number(node: exp.Expression) -> exp.Literal | None:
    if isinstance(node, exp.Literal) and not node.is_string and re.fullmatch(r"\d+", node.name or "") and int(node.name) < _LIMIT:
        return node
    return None


def _integer_column(column: exp.Column, inner: exp.Select, types: dict) -> bool:
    """Whether ``column`` is an integer column of a base table among ``inner``'s own sources."""

    if isinstance(column.this, exp.Star) or column.args.get("db") or column.args.get("catalog"):
        return False
    sources = select_sources(inner)
    names = [(s.alias_or_name or "").lower() for s in sources]
    if not sources or "" in names or len(set(names)) != len(names):
        return False
    if any(isinstance(s, (exp.Lateral, exp.Unnest)) for s in sources):
        return False
    if column.table:
        owners = [s for s, name in zip(sources, names) if name == column.table.lower()]
    else:
        owners = sources if len(sources) == 1 else []
    if len(owners) != 1 or not isinstance(owners[0], exp.Table):
        return False
    table = owners[0]
    alias = table.args.get("alias")
    if (alias is not None and alias.args.get("columns")) or not isinstance(table.this, exp.Identifier):
        return False
    if not table.args.get("db") and not table.args.get("catalog") and table.name.lower() in {n.lower() for n in visible_ctes(table)}:
        return False  # a WITH table of that name, not the base table
    key = ".".join(part.name for part in table.parts).lower()
    declared = types.get(key) or types.get(table.name.lower()) or {}
    return str(declared.get(column.name.lower(), "")).upper() in _INTEGER


def _pinned(source: exp.Expression, types: dict) -> dict[str, exp.Literal]:
    """``{output name: the whole number it always holds}`` for a derived table this rule reads."""

    if not isinstance(source, exp.Subquery) or not isinstance(source.this, exp.Select):
        return {}
    alias = source.args.get("alias")
    if alias is None or not source.alias or alias.args.get("columns"):
        return {}
    if any(value for key, value in source.args.items() if key not in ("this", "alias") and value):
        return {}
    inner = source.this
    where, group = inner.args.get("where"), inner.args.get("group")
    if where is None or any(inner.args.get(key) for key in ("with_", "with", "laterals", "pivots", "qualify")):
        return {}
    if group is not None and extended_grouping(group):
        return {}
    if group is None and any(a.find_ancestor(exp.Select) is inner and not isinstance(a.parent, exp.Window) for a in inner.find_all(exp.AggFunc)):
        return {}
    names = [(item.alias_or_name or "").lower() for item in inner.expressions]
    values: dict[str, exp.Literal] = {}
    for part in _conjuncts(where.this):
        if not isinstance(part, exp.EQ):
            continue
        for column, other in ((part.this, part.expression), (part.expression, part.this)):
            number = _whole_number(other)
            if number is None or not isinstance(column, exp.Column) or not _integer_column(column, inner, types):
                continue
            text = column.sql()
            for item, name in zip(inner.expressions, names):
                value = item.this if isinstance(item, exp.Alias) else item
                if isinstance(value, exp.Column) and value.sql() == text and name and names.count(name) == 1:
                    values.setdefault(name, number)
    return values


def read_pinned_source_columns(select: exp.Select, types: dict | None) -> exp.Expression | None:
    """Replace the select's own reads of a pinned derived-table column by its number."""

    if not types:
        return None
    from_ = select.args.get("from_") or select.args.get("from")
    joins = select.args.get("joins") or []
    if from_ is None:
        return None
    if any((j.args.get("side") or "").upper() in ("RIGHT", "FULL") or j.args.get("method") or j.args.get("using") for j in joins):
        return None
    group = select.args.get("group")
    if group is not None and extended_grouping(group):
        return None
    sources = select_sources(select)
    aliases = [(s.alias_or_name or "").lower() for s in sources]
    candidates = [from_.this] + [j.this for j in joins if not (j.args.get("side") or "") and (j.args.get("kind") or "").upper() in _JOIN_KINDS]
    pinned: dict[str, dict[str, exp.Literal]] = {}
    for source in candidates:
        alias = (source.alias_or_name or "").lower()
        if not alias or aliases.count(alias) != 1:
            continue
        found = _pinned(source, types)
        if found:
            pinned[alias] = found
    if not pinned:
        return None
    places = list(select.expressions)
    for key in ("where", "having"):
        if select.args.get(key) is not None:
            places.append(select.args[key])
    places.extend(j.args["on"] for j in joins if j.args.get("on") is not None)
    changed = False
    for place in places:
        for column in list(place.find_all(exp.Column)):
            if isinstance(column.this, exp.Star) or column.find_ancestor(exp.Select) is not select:
                continue
            if any(isinstance(p, (exp.Order, exp.Group)) for p in _ancestors(column, select)):
                continue  # ORDER BY inside OVER or an aggregate: a number there is not this value
            number = pinned.get(column.table.lower(), {}).get(column.name.lower())
            if number is None:
                continue
            literal = exp.Literal.number(number.name)
            if any(column is item for item in select.expressions):
                literal = exp.Alias(this=literal, alias=column.this.copy())
            column.replace(literal)
            changed = True
    return select if changed else None


def _ancestors(node: exp.Expression, top: exp.Expression):
    node = node.parent
    while node is not None and node is not top:
        yield node
        node = node.parent
