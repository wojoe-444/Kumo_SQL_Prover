"""Two rows of one table told apart by a key are told apart by the table's first key.

    SELECT a.pid FROM policy AS a JOIN policy AS b ON a.city = b.city AND a.pid <> b.pid

asks for another row with the same city, and says "another row" with the key ``pid``. When the table has a second
key, say ``code``, the same question can be written ``a.code <> b.code``, and other rules of this package write it
with whichever key they find first (``own_group_membership`` reads ``city IN (SELECT city .. HAVING COUNT(*) > 1)``
as a row with the same city and another key value). Two spellings of one question then differ only in the key
they name. This rule names one key for all of them: the declared one-column NOT NULL key whose name sorts first.

**Why it holds.** ``k`` and ``k0`` are declared one-column keys of ``T``, both NOT NULL, and ``x`` and ``y`` are
rows of ``T``. A key holds a different value on every row and no NULL, so ``x.k <> y.k`` is TRUE exactly when ``x``
and ``y`` are different rows, FALSE when they are the same row, and never NULL; the same goes for ``k0``. When
either row is the NULL padding of an outer join, both comparisons are NULL. So ``x.k <> y.k`` and
``x.k0 <> y.k0`` have the same value wherever they are written.

**What is read.** A ``<>`` between the same column of two different sources, in the ``WHERE`` or an ``ON`` of the
select, where each source is a base table read by the select or by a query around it (the nearest source of that
name, when the select sits in the ``WHERE`` of each query on the way out), both are the same table, and the column
is a declared one-column NOT NULL key that is not the first such key by name.

**Left alone.** ``=`` (an equality of keys is what join and key rules read), ``<`` and ``>``; a column of a derived
table or of a name whose owner is not certain (a bare column, a quoted name); a key or a first key declared
floating-point (a NaN differs from itself in some engines); a select with ``USING``.

``tests/test_different_row_keys.py`` replays the rewrite on DuckDB and shows both keys are needed.
"""

from __future__ import annotations

from sqlglot import exp

from . import smt_values
from .own_group_membership import _facts, _from, _plain_table, _unparen


def _sources(select: exp.Select) -> dict[str, exp.Expression] | None:
    first = _from(select)
    joins = select.args.get("joins") or []
    if first is None or any(j.args.get("using") for j in joins):
        return None
    found: dict[str, exp.Expression] = {}
    for source in [first] + [j.this for j in joins]:
        name = (source.alias_or_name or "").lower()
        if not name or name in found:
            return None
        found[name] = source
    return found


def _owner(select: exp.Select, name: str) -> exp.Table | None:
    """The base table the nearest source called ``name`` reads, seen from a condition of ``select``."""

    inner = select
    while True:
        sources = _sources(inner)
        if sources is None:
            return None
        if name in sources:
            return sources[name] if _plain_table(sources[name]) else None
        node, through_where = inner.parent, False
        while node is not None and not isinstance(node, exp.Select):
            if isinstance(node, (exp.From, exp.Join, exp.CTE, exp.With, exp.SetOperation, exp.Lateral)):
                return None
            through_where = through_where or isinstance(node, exp.Where)
            node = node.parent
        if node is None or not through_where:
            return None
        inner = node


def _first_key(table: exp.Table, keys: dict | None, not_null: dict | None, types: dict | None) -> tuple[str, set[str]] | None:
    """``(the first one-column NOT NULL key by name, all such keys)`` of a base table."""

    declared = {str(c).lower() for c in (_facts(not_null, table.name) or ())}
    kinds = {str(k).lower(): v for k, v in (_facts(types, table.name) or {}).items()}
    single = {str(k[0]).lower() for k in (_facts(keys, table.name) or ()) if len(k) == 1}
    usable = {k for k in single if k in declared and smt_values.type_class(kinds.get(k)) != smt_values.FLOAT64}
    return (min(usable), usable) if usable else None


def name_different_rows(select: exp.Select, keys: dict | None, not_null: dict | None, types: dict | None = None) -> exp.Select | None:
    """``select`` with each ``x.k <> y.k`` between two rows of one table written with the table's first key."""

    if not keys or not not_null:
        return None
    parts = [j.args["on"] for j in select.args.get("joins") or [] if j.args.get("on") is not None]
    if select.args.get("where") is not None:
        parts.append(select.args["where"].this)
    changes: list[tuple[exp.NEQ, str]] = []
    for part in parts:
        for node in part.find_all(exp.NEQ):
            if node.find_ancestor(exp.Select) is not select:
                continue  # a nested query's own condition is read when that query is
            left, right = _unparen(node.this), _unparen(node.expression)
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                continue
            if any(not c.table or c.args.get("db") or not isinstance(c.this, exp.Identifier) or c.this.args.get("quoted") for c in (left, right)):
                continue
            key = left.name.lower()
            if right.name.lower() != key or left.table.lower() == right.table.lower():
                continue
            one, other = _owner(select, left.table.lower()), _owner(select, right.table.lower())
            if one is None or other is None or one.name != other.name or bool(one.this.args.get("quoted")) != bool(other.this.args.get("quoted")):
                continue
            first = _first_key(one, keys, not_null, types)
            if first is None or key not in first[1] or first[0] == key:
                continue
            changes.append((node, first[0]))
    if not changes:
        return None
    marks = {id(node): name for node, name in changes}
    result = select.copy()
    # the copy has the same shape: walk both trees in step to find the copied nodes
    for original, copied in zip(select.walk(), result.walk()):
        name = marks.get(id(original))
        if name is not None:
            for side in (copied.this, copied.expression):
                _unparen(side).set("this", exp.to_identifier(name))
    return result
