"""``COUNT(DISTINCT j.k)`` in a group that cannot hold a row of ``j`` twice is ``COUNT(j.k)``.

    -- "per manager, the number of people who report to them"
    SELECT m.id, m.name, COUNT(DISTINCT e.id) AS reports
    FROM staff AS m JOIN staff AS e ON e.boss = m.id GROUP BY m.id, m.name          -- COUNT(e.id)

**Why it holds.** ``COUNT(DISTINCT x)`` counts the different values of ``x`` that are not NULL and ``COUNT(x)`` the
rows on which ``x`` is not NULL. They agree on a group in which no two rows hold the same value of ``x`` that is
not NULL.

The select joins declared tables. A row of the join is a combination of one row of each table (or, on the padded
side of a left join, of no row of it), and a join returns each combination at most once. Take a source ``s`` other
than ``j`` that has a declared key whose columns are all declared NOT NULL and all among the plain columns of the
``GROUP BY``. Two rows of one group agree on those columns. If the values are not NULL, both rows hold the one row
of ``s`` with that key; if they are NULL, both hold no row of ``s`` (only padding makes such a column NULL). When that is so
for every source but ``j``, two different rows of a group can differ only in their row of ``j``, and ``(k)`` is a
declared key of ``j``: two different rows of ``j`` do not share a value of ``k`` that is not NULL (a unique key
admits several NULLs, and neither count reads a NULL). So the group does not repeat a value of ``j.k`` that is
not NULL. Further expressions in the ``GROUP BY`` only split groups, and ``WHERE`` and
``HAVING`` only remove rows and groups.

The same equality is used on both sides of the argument: a key tells its values apart the way ``GROUP BY`` and
``DISTINCT`` do under the column's collation. Floating-point key columns are left alone all the same.

**What is read.** A select with a plain ``GROUP BY`` (no ``ROLLUP``, ``CUBE`` or grouping sets, whose totals span
several groups) over declared tables under different names, joined by inner, cross or left joins with ``ON`` (no
``USING``, ``NATURAL`` or ``LATERAL``), and not inside or over a ``WITH`` clause that could have taken a table's
name. A grouping column is a column written with its source's name, a bare name that exactly one source has, a
bare name no source has that the select list gives to such a column, or a position of such a column. Each
``COUNT(DISTINCT j.k)`` of the select itself (not of a subquery, not a window function) with such a ``j`` and ``k``
loses its ``DISTINCT``; nothing else changes.

A rule of the twentieth hardest-cases batch. ``tests/test_grouped_key_counts.py`` runs the reading on SQLite over
made-up tables and holds tables on which the refused shapes count differently.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY
from .filter_terms import _lower, _unwrapped

NAME = "grouped_key_counts"

_FLOATS = ("FLOAT", "DOUBLE", "REAL", "FLOAT64", "FLOAT4", "FLOAT8", "DOUBLE PRECISION")
_REFUSED_CLAUSES = ("laterals", "pivots", "sample", "into")


def _of(table: exp.Table, facts: dict | None):
    return next((value for name, value in (facts or {}).items() if _lower(name) == _lower(table.name)), None)


def _under_with(select: exp.Select) -> bool:
    if select.find(exp.With) is not None:
        return True
    node: exp.Expression | None = select
    while node is not None:
        if node.args.get("with_") or node.args.get("with"):
            return True
        node = node.parent
    return False


def grouped_key_counts(select: exp.Select, keys: dict | None, not_null: dict | None, types: dict | None) -> exp.Select | None:
    """The select with ``DISTINCT`` taken out of each count of a key column its groups cannot repeat (module doc),
    or None."""

    group, from_ = select.args.get("group"), select.args.get(FROM_KEY)
    if group is None or from_ is None or not keys or not types or not group.expressions:
        return None
    if any(value for key, value in group.args.items() if key != "expressions") or any(select.args.get(key) for key in _REFUSED_CLAUSES):
        return None
    counts = []
    for count in select.find_all(exp.Count):
        distinct = count.this
        if not isinstance(distinct, exp.Distinct) or distinct.args.get("on") is not None or len(distinct.expressions) != 1 or count.expressions:
            continue
        if count.find_ancestor(exp.Window, exp.Select) is not select or not isinstance(_unwrapped(distinct.expressions[0]), exp.Column):
            continue
        counts.append(count)
    if not counts:
        return None
    joins = select.args.get("joins") or []
    sources = [from_.this] + [join.this for join in joins]
    for join in joins:
        if join.args.get("using") or join.args.get("method") or _lower(join.args.get("side")) not in ("", "left") or _lower(join.args.get("kind")) not in ("", "inner", "outer", "cross"):
            return None
        if join.args.get("on") is None and (_lower(join.args.get("side")) or _lower(join.args.get("kind")) == "outer"):
            return None
    columns: dict[str, dict[str, str]] = {}  # source name -> its declared columns (lower case) and their types
    tables: dict[str, exp.Table] = {}
    for source in sources:
        if not isinstance(source, exp.Table) or source.args.get("db") or source.args.get("catalog") or source.args.get("pivots") or source.args.get("sample"):
            return None
        declared = _of(source, types)
        name = _lower(source.alias_or_name)
        if not isinstance(declared, dict) or not name or name in columns or not isinstance(source.this, exp.Identifier):
            return None
        alias = source.args.get("alias")
        if alias is not None and alias.args.get("columns"):
            return None
        columns[name] = {_lower(column): str(kind or "") for column, kind in declared.items()}
        tables[name] = source
    if _under_with(select):
        return None

    def owner(column: exp.Column) -> tuple[str, str] | None:
        if column.args.get("db") or column.args.get("catalog"):
            return None
        name = _lower(column.name)
        if column.table:
            table = _lower(column.table)
            return (table, name) if table in columns and name in columns[table] else None
        owners = [table for table in columns if name in columns[table]]
        return (owners[0], name) if len(owners) == 1 else None

    named: dict[str, exp.Expression] = {}
    twice: set[str] = set()
    for item in select.expressions:
        if isinstance(item, exp.Alias):
            name = _lower(item.alias)
            if name in named:
                twice.add(name)
            named[name] = item.this
    grouped: set[tuple[str, str]] = set()
    for entry in group.expressions:
        entry = _unwrapped(entry)
        if isinstance(entry, exp.Literal):
            if entry.is_string or not entry.name.isdigit() or not 1 <= int(entry.name) <= len(select.expressions):
                return None
            item = select.expressions[int(entry.name) - 1]
            entry = _unwrapped(item.this if isinstance(item, exp.Alias) else item)
        elif isinstance(entry, exp.Column) and not entry.table and not any(_lower(entry.name) in declared for declared in columns.values()):
            name = _lower(entry.name)  # a name no source has: the select list's
            if name in twice or name not in named:
                return None
            entry = _unwrapped(named[name])
        if isinstance(entry, exp.Column):
            found = owner(entry)
            if found is not None:
                grouped.add(found)

    def pinned(name: str) -> bool:
        never_null = {_lower(column) for column in _of(tables[name], not_null) or ()}
        return any(key and all(_lower(column) in never_null and (name, _lower(column)) in grouped for column in key) for key in _of(tables[name], keys) or ())

    new = None
    for count in counts:
        found = owner(_unwrapped(count.this.expressions[0]))
        if found is None:
            continue
        name, column = found
        if not any(len(key) == 1 and _lower(key[0]) == column for key in _of(tables[name], keys) or ()):
            continue
        kind = columns[name][column].upper().split("(")[0].strip()
        if not kind or kind in _FLOATS or not all(pinned(other) for other in columns if other != name):
            continue
        if new is None:
            marks = [id(node) for node in select.find_all(exp.Count)]
            new = select.copy()
            copies = dict(zip(marks, new.find_all(exp.Count)))
        copy = copies[id(count)]
        copy.set("this", _unwrapped(copy.this.expressions[0]))
    return new
