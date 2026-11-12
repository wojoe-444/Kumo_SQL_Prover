"""A ``UNION`` with an operand that can never hold a row is its other operand.

    SELECT a FROM r UNION ALL SELECT b FROM u WHERE FALSE      is   SELECT a FROM r
    SELECT a FROM r UNION     SELECT b FROM u WHERE FALSE      is   SELECT DISTINCT a FROM r

A filter on a constant tag over a ``UNION ALL`` (``WHERE source = 'store'`` over branches that each select a
literal ``source``) is pushed into every branch by the normalizer and folds to ``FALSE`` in all but one. The
branches that are left empty stayed in the query, so one table read through a tagged union and the same table
read directly met the prover in two forms. Inside a larger set operation (``.. EXCEPT ..``) the prover has no
reading of the union form at all.

An operand with no rows adds nothing to a union. What it does decide is kept:

* the output names are the first operand's, so when the first operand is the empty one the other operand's
  items take its names (only a plain select is renamed, and only when its own clauses cannot read the old
  names: no ``GROUP BY``, ``HAVING``, ``ORDER BY``, ``QUALIFY``; anything else is left as written);
* a duplicate-removing ``UNION`` still removes the duplicates of the operand that remains: a select gets
  ``DISTINCT``, a ``UNION ALL`` becomes ``UNION``, an operand that already returns distinct rows stays.

A union also gives its columns one type, and an empty operand takes part in that: ``SELECT x UNION ALL
SELECT CAST(y AS FLOAT64) .. WHERE FALSE`` returns ``x`` as a float, which rounds a large integer. The prover
checks the types of a union's branches when it reads the union, so an operand is dropped here only when that
check has nothing to find: at every position every select of both operands lists the same column of the same
table (read directly or through derived tables that pass it on by name). Any other pair of items (two columns, a cast, a
literal, an expression) leaves the union to the prover.

Left alone: operands that do not select the same table columns position by position; a ``UNION`` with its own ``ORDER BY``, ``LIMIT``, ``WITH`` or ``BY NAME``; both operands empty
or neither (``empty_rules`` reads a relation that is empty as a whole); a remaining operand that carries a
``LIMIT``, ``OFFSET``, ``ORDER BY`` or ``DISTINCT ON`` under a duplicate-removing ``UNION`` (``DISTINCT``
would be applied before the limit), or an ``ORDER BY``, ``LIMIT`` or ``OFFSET`` under any ``UNION`` (outside
the union the ``ORDER BY`` would order the whole result); ``INTERSECT ALL`` and ``EXCEPT ALL`` operands under it; a star in the
empty first operand (its names are not known).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import distinct_on, select_sources
from .empty_rules import is_empty

_SETOP_EXTRAS = ("order", "limit", "offset", "with_", "with", "by_name", "side", "kind", "on", "match")
_ROW_CLAUSES = ("limit", "offset", "order", "qualify", "windows", "with_", "with", "into", "locks", "sample", "pivots")
_NAME_READERS = ("group", "having", "order", "qualify", "windows", "distinct", "limit", "offset", "with_", "with")


def _operand(node: exp.Expression) -> exp.Expression | None:
    """The query inside parentheses that carry nothing of their own; ``None`` for anything else."""

    while isinstance(node, (exp.Subquery, exp.Paren)):
        if node.alias or any(value for key, value in node.args.items() if key != "this" and value):
            return None
        node = node.this
    return node if isinstance(node, (exp.Select, exp.SetOperation)) else None


def _first_select(node: exp.Expression) -> exp.Select | None:
    while True:
        node = _operand(node) if isinstance(node, (exp.Subquery, exp.Paren)) else node
        if isinstance(node, exp.SetOperation):
            node = node.this
        else:
            return node if isinstance(node, exp.Select) else None


def _names(node: exp.Expression) -> list[str] | None:
    first = _first_select(node)
    if first is None:
        return None
    names = []
    for item in first.expressions:
        if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
            return None
        names.append(item.alias_or_name or "")
    return names


def _table_column(select: exp.Select, item: exp.Expression, depth: int = 0) -> tuple | None:
    """The table column an output item of ``select`` passes on unchanged, as ``(table, db, catalog, column)``."""

    value = item.this if isinstance(item, exp.Alias) else item
    while isinstance(value, exp.Paren):
        value = value.this
    if not isinstance(value, exp.Column) or isinstance(value.this, exp.Star) or depth > 20:
        return None
    sources = select_sources(select)
    if value.table:
        named = [s for s in sources if (s.alias_or_name or "").lower() == value.table.lower()]
    else:
        named = sources  # a bare column of a select with one source is that source's
    if len(named) != 1 or (not value.table and len(sources) != 1):
        return None
    source = named[0]
    alias = source.args.get("alias")
    if alias is not None and alias.args.get("columns"):
        return None  # ``t AS x(a, b)`` renames the columns
    if isinstance(source, exp.Table):
        if not isinstance(source.this, exp.Identifier) or any(v for k, v in source.args.items() if k not in ("this", "db", "catalog", "alias") and v):
            return None
        return (source.name, source.text("db"), source.text("catalog"), value.name.lower())
    if not isinstance(source, exp.Subquery) or not isinstance(source.this, exp.Select):
        return None
    if any(v for k, v in source.args.items() if k not in ("this", "alias") and v):
        return None
    inner = source.this
    passed = [i for i in inner.expressions if (i.alias_or_name or "").lower() == value.name.lower()]
    if len(passed) != 1 or any(isinstance(i, exp.Star) or (isinstance(i, exp.Column) and isinstance(i.this, exp.Star)) for i in inner.expressions):
        return None
    return _table_column(inner, passed[0], depth + 1)


def _leaves(node: exp.Expression) -> list[exp.Select] | None:
    """The selects a tree of set operations is made of; ``None`` when something else is among them."""

    node = _operand(node) if isinstance(node, (exp.Subquery, exp.Paren)) else node
    if isinstance(node, exp.Select):
        return [node]
    if not isinstance(node, exp.SetOperation):
        return None
    left, right = _leaves(node.this), _leaves(node.expression)
    return None if left is None or right is None else left + right


def _same_columns(left: exp.Expression, right: exp.Expression) -> bool:
    """Whether every select of both operands lists the same table columns position by position, so that no
    operand converts another's values."""

    leaves = [_leaves(left), _leaves(right)]
    if None in leaves:
        return False
    selects = leaves[0] + leaves[1]
    first = [_table_column(selects[0], item) for item in selects[0].expressions]
    if not first or None in first:
        return False
    return all([_table_column(select, item) for item in select.expressions] == first for select in selects[1:])


def _renamed(select: exp.Select, names: list[str]) -> exp.Select | None:
    """``select`` with its items named ``names``; ``None`` when a clause of its own could read the old names."""

    if any(select.args.get(key) for key in _NAME_READERS) or len(select.expressions) != len(names) or "" in names:
        return None
    out = select.copy()
    items = []
    for item, name in zip(out.expressions, names):
        if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
            return None
        value = item.this if isinstance(item, exp.Alias) else item
        items.append(item if (item.alias_or_name or "") == name else exp.alias_(value.copy(), name, quoted=False))
    out.set("expressions", items)
    return out


def _as_set(query: exp.Expression) -> exp.Expression | None:
    """``query`` with its duplicate rows removed, written without a new level."""

    if isinstance(query, exp.Select):
        if query.args.get("distinct") is not None:
            return None if distinct_on(query) else query
        if any(query.args.get(key) for key in _ROW_CLAUSES) or any(isinstance(n, exp.Window) for n in query.walk()):
            return None
        query.set("distinct", exp.Distinct())
        return query
    if any(query.args.get(key) for key in _SETOP_EXTRAS):
        return None
    if query.args.get("distinct") is True:
        return query  # UNION, INTERSECT and EXCEPT return distinct rows
    if type(query) is exp.Union and query.args.get("distinct") is False:
        query.set("distinct", True)
        return query
    return None


def drop_empty_union_operand(node: exp.Expression) -> exp.Expression | None:
    """The non-empty operand of a ``UNION`` whose other operand can never hold a row, else ``None``."""

    if type(node) is not exp.Union or any(node.args.get(key) for key in _SETOP_EXTRAS):
        return None
    distinct = node.args.get("distinct")
    if distinct is not True and distinct is not False:
        return None
    left_empty, right_empty = is_empty(node.this), is_empty(node.expression)
    if left_empty == right_empty:
        return None
    kept = _operand(node.expression if left_empty else node.this)
    gone = _operand(node.this if left_empty else node.expression)
    if kept is None or gone is None or not _same_columns(kept, gone):
        return None
    if isinstance(kept, exp.Select) and any(kept.args.get(key) for key in ("order", "limit", "offset")):
        return None  # outside its parentheses the ORDER BY would order the whole result
    kept = kept.copy()
    if left_empty:
        wanted, have = _names(node.this), _names(kept)
        if wanted is None or have is None or len(wanted) != len(have):
            return None
        if [n.lower() for n in wanted] != [n.lower() for n in have]:
            kept = _renamed(kept, wanted) if isinstance(kept, exp.Select) else None
            if kept is None:
                return None
    if distinct:
        kept = _as_set(kept)
        if kept is None:
            return None
    return exp.Subquery(this=kept) if isinstance(node.parent, exp.SetOperation) else kept
