"""A presence test that a declared foreign key answers is dropped.

    SELECT p.item_id FROM item AS p WHERE p.item_id IN (SELECT s.item_id FROM sale AS s) AND ..

asks, for each row of ``item``, whether ``sale`` holds a row with its ``item_id``. When ``item(item_id)`` is declared
to reference ``sale(item_id)`` and ``item.item_id`` is NOT NULL, the answer is yes for every row, and the test can
go. "Items whose sales all fall in the quarter" and "items with no sale outside the quarter" differ only on an
item with no sale; under that foreign key there is none, and this rule is what lets the two normal forms meet.

**Why it holds.** A foreign key ``C(c) -> P(k)`` says every row of ``C`` whose ``c`` is not NULL has a row of ``P``
with ``k = c``. With ``c`` NOT NULL that is every row of ``C``. So for a real row ``x`` of ``C``,
``x.c IN (SELECT k FROM P)`` finds a match and is TRUE (a match makes ``IN`` TRUE whatever NULLs the list holds),
and ``EXISTS (SELECT .. FROM P WHERE P.k = x.c)`` is TRUE. A TRUE conjunct of a ``WHERE`` changes nothing.

**What is read.** A conjunct of the ``WHERE`` of a select, in one of two spellings:

- ``x.c IN (SELECT k FROM P)``: the query reads one base table, lists the one column ``k`` of it (``DISTINCT``
  or not), and has nothing else (no ``WHERE``, join, grouping, ``LIMIT`` or set operation);
- ``EXISTS (SELECT .. FROM P WHERE P.k = x.c)``: the same, with that one equality as the whole ``WHERE`` and any
  select list without an aggregate or a window.

``x`` is a base table of the select that is a real row wherever the ``WHERE`` is tested: the ``FROM`` source or an
inner or cross joined one, in a select with no ``RIGHT`` or ``FULL`` join. ``x.c`` is written with the source's
name, or bare when the select reads that one table and the supplied column lists give it the column.

**Left alone.** ``NOT IN`` and ``NOT EXISTS`` (they would be FALSE: other rules decide what an always-empty select
is); a test under ``OR`` or in an ``ON``; a foreign key over several columns; a column the foreign key names that
is not declared NOT NULL (a NULL has no parent and the test is not TRUE for it); a column declared floating-point
on either side; a quoted name; a test whose columns cannot be placed for certain (a bare name the column lists
do not give to the table read by the query, or give to both tables of a correlated test).

``tests/test_referenced_presence.py`` replays the rewrite on DuckDB and shows each declared fact is needed.
"""

from __future__ import annotations

from sqlglot import exp

from . import smt_values
from .ast_utils import conjuncts, is_function_table
from .exists_constant_rules import _real_sources
from .fk_rules import _same_table
from .own_group_membership import _facts, _plain_table, _unparen

_BLOCKING = ("joins", "group", "having", "qualify", "limit", "offset", "laterals", "with", "order", "windows", "connect", "match", "into", "locks", "sample")


def _columns(schema: dict | None, table: exp.Table) -> set[str] | None:
    listed = _facts(schema, table.name)
    return {str(c).lower() for c in listed} if listed is not None else None


def _plain_column(node: exp.Expression | None) -> exp.Column | None:
    node = _unparen(node)
    if not isinstance(node, exp.Column) or not isinstance(node.this, exp.Identifier) or node.this.args.get("quoted") or node.args.get("db"):
        return None
    return node


def _one_table(body: exp.Expression | None) -> exp.Table | None:
    """The one base table a plain select reads (no join, grouping, tail or nested query), None otherwise."""

    if not isinstance(body, exp.Select) or any(body.args.get(k) for k in _BLOCKING):
        return None
    distinct = body.args.get("distinct")
    if distinct is not None and distinct.args.get("on"):
        return None
    from_ = body.args.get("from_") or body.args.get("from")
    parent = from_.this if from_ is not None else None
    if not _plain_table(parent) or is_function_table(parent) or parent.this.args.get("quoted"):
        return None
    return parent


def _of(column: exp.Column | None, table: exp.Table, listed: set[str] | None) -> str | None:
    """The name of ``column`` when it is certainly a column of ``table`` as seen inside the query that reads it."""

    if column is None:
        return None
    name = column.name.lower()
    if column.table:
        return name if column.table.lower() == table.alias_or_name.lower() else None
    return name if listed is not None and name in listed else None


def _tested(node: exp.Expression, schema: dict | None) -> tuple[exp.Column, exp.Table, str] | None:
    """``(the outer column, the table the test reads, that table's column)`` for a presence test, None otherwise."""

    node = _unparen(node)
    if isinstance(node, exp.In):
        query = node.args.get("query")
        if query is None or node.args.get("expressions") or node.args.get("field") or node.args.get("unnest"):
            return None
        body = query.this if isinstance(query, exp.Subquery) else query
        parent = _one_table(body)
        outer = _plain_column(node.this)
        if parent is None or outer is None or body.args.get("where") is not None or len(body.expressions) != 1:
            return None
        item = body.expressions[0]
        key = _of(_plain_column(item.this if isinstance(item, exp.Alias) else item), parent, _columns(schema, parent))
        return (outer, parent, key) if key is not None else None
    if isinstance(node, exp.Exists):
        body = node.this
        parent = _one_table(body)
        if parent is None or body.args.get("where") is None or body.find(exp.AggFunc, exp.Window, exp.Subquery, exp.Exists):
            return None
        test = _unparen(body.args["where"].this)
        if not isinstance(test, exp.EQ):
            return None
        listed = _columns(schema, parent)
        inner_name = parent.alias_or_name.lower()
        for mine, other in ((test.this, test.expression), (test.expression, test.this)):
            key, outer = _of(_plain_column(mine), parent, listed), _plain_column(other)
            if key is None or outer is None or not outer.table or outer.table.lower() == inner_name:
                continue
            # the select list may read the table's own columns only (an outer name there changes nothing, but keep it plain)
            if any(c.table and c.table.lower() != inner_name for item in body.expressions for c in item.find_all(exp.Column)):
                return None
            return outer, parent, key
    return None


def drop_referenced_presence(select: exp.Select, schema: dict | None, not_null: dict | None, foreign_keys: dict | None, types: dict | None = None) -> exp.Select | None:
    """``select`` without the ``WHERE`` conjuncts that test for a row a declared foreign key says is there."""

    where = select.args.get("where")
    if not foreign_keys or not not_null or where is None:
        return None
    real = [s for s in _real_sources(select) or [] if _plain_table(s) and not is_function_table(s) and not s.this.args.get("quoted")]
    if not real:
        return None
    everything = 1 + len(select.args.get("joins") or [])
    by_name: dict[str, exp.Table | None] = {}
    for source in real:
        name = source.alias_or_name.lower()
        by_name[name] = None if name in by_name else source
    parts = conjuncts(where.this)
    kept = []
    for part in parts:
        if not _answered(part, real, by_name, everything, schema, not_null, foreign_keys, types):
            kept.append(part)
    if len(kept) == len(parts):
        return None
    result = select.copy()
    result.set("where", exp.Where(this=exp.and_(*[p.copy() for p in kept])) if kept else None)
    return result


def _answered(part: exp.Expression, real: list, by_name: dict, everything: int, schema, not_null, foreign_keys, types) -> bool:
    found = _tested(part, schema)
    if found is None:
        return False
    outer, parent, key = found
    if outer.table:
        child = by_name.get(outer.table.lower())
    else:
        # a bare name: the select reads this one table, and the lists give it the column
        child = real[0] if everything == 1 and len(real) == 1 and outer.name.lower() in (_columns(schema, real[0]) or ()) else None
    if child is None:
        return False
    column = outer.name.lower()
    if column not in {str(c).lower() for c in (_facts(not_null, child.name) or ())}:
        return False
    for name, kinds in ((child.name, column), (parent.name, key)):
        declared = {str(k).lower(): v for k, v in (_facts(types, name) or {}).items()}
        if smt_values.type_class(declared.get(kinds)) == smt_values.FLOAT64:
            return False
    for cols, fk_parent, parent_cols in _facts(foreign_keys, child.name) or ():
        if len(cols) == 1 and len(parent_cols) == 1 and str(cols[0]).lower() == column and str(parent_cols[0]).lower() == key and _same_table(parent.name, str(fk_parent)):
            return True
    return False
