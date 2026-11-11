"""A pre-grouped source under a select that only takes minimums and maximums reads as its rows.

    SELECT MIN(g.m), MAX(o.y) FROM (SELECT k, MIN(x) AS m FROM s GROUP BY k) AS g, o WHERE g.k = o.k
    SELECT MIN(g.m), MAX(o.y) FROM (SELECT k,     x  AS m FROM s           ) AS g, o WHERE g.k = o.k

Calcite's ``AggregateJoinTransposeRule`` pushes ``MIN`` and ``MAX`` below a join this way, one level per join,
and wraps each other table in ``SELECT key FROM t GROUP BY key``. ``eager_aggregation`` undoes the single
level; this rule undoes any level whose select has subqueries, further joins or a grouping of its own, so the
levels come apart one at a time from the inside.

Why the two selects above return the same rows. Call the grouped source ``g`` and the same source without its
``GROUP BY`` (each ``MIN(x)`` replaced by ``x``) ``g'``. Every row of ``g`` stands for the one or more rows of
``g'`` that have its key, and each of those rows carries the same key values. The outer select reads ``g``
only through its key columns, except inside ``MIN(g.m)``:

* ``WHERE``, ``ON``, ``GROUP BY`` and correlated subqueries see only key columns, so a joined row built on a
  row of ``g`` passes them exactly when the rows built on the matching rows of ``g'`` do. The outer select's
  groups are therefore the same groups, each non-empty on both sides, with every joined row repeated once or
  more.
* ``MIN`` and ``MAX`` do not see repeats, so every other aggregate of the outer select is unchanged.
* ``MIN(g.m)`` over a group is the least ``MIN(x)`` of the keys in it, which is the least ``x`` over all their
  rows (a key whose ``x`` are all NULL contributes NULL on both sides, and NULLs are skipped).

So the rule needs, and checks:

* the outer select groups or aggregates, and every aggregate of its own is ``MIN`` or ``MAX`` of one argument
  (a ``COUNT`` or ``SUM`` would count the repeats; a select with no grouping would return them). A function
  name the parser does not know, or ``WITHIN GROUP``, outside the select's sources stops the rule: it may be
  an aggregate;
* no ``DISTINCT``, window, ``LIMIT``, ``OFFSET``, ``QUALIFY``, ``ROLLUP`` or grouping sets on the outer
  select, and no ``RIGHT`` or ``FULL`` join in it; the grouped source is its ``FROM`` item or the right side of
  a plain, ``INNER`` or ``CROSS`` join, so no outer join can pad it with NULLs;
* the grouped source is one ``SELECT .. GROUP BY`` of plain keys with no ``HAVING``, ``DISTINCT``, window,
  ``ORDER BY`` or ``LIMIT``; each output is a grouping key (same text) or ``MIN(e)`` / ``MAX(e)`` with a
  name, where ``e`` holds no aggregate, subquery or window. A select without ``GROUP BY`` is not read: it
  returns a row for no input;
* every column of the outer select (subqueries included) that reads an aggregate output is qualified by the
  source's alias and is the whole argument of the same function, ``MIN(g.m)`` for ``MIN(x) AS m``, in the
  outer select itself;
* no bare column anywhere in the outer select has the name of one of the source's outputs, and no star reads
  it (nothing then reads the source in a way this module did not look at).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, extended_grouping, select_sources

_EXTREMES = (exp.Min, exp.Max)
_OUTER_BANNED = ("distinct", "limit", "offset", "qualify", "windows", "with_", "with", "laterals", "pivots", "connect", "sample")
_INNER_ALLOWED = frozenset({"expressions", "from_", "from", "joins", "where", "group", "kind", "hint"})
_JOIN_KINDS = {"", "INNER", "CROSS"}


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _own(node: exp.Expression, select: exp.Select) -> bool:
    return node.find_ancestor(exp.Select) is select


def _plain_extreme(call: exp.Expression) -> bool:
    """``MIN(e)`` or ``MAX(e)``: one argument, no modifier, not a window."""

    if type(call) not in _EXTREMES or isinstance(call.parent, (exp.Window, exp.Filter)):
        return False
    if any(value for key, value in call.args.items() if key != "this"):
        return False
    return isinstance(call.this, exp.Expression) and not isinstance(call.this, (exp.Distinct, exp.Star))


def _grouped_outputs(inner: exp.Select) -> tuple[dict[str, type], set[str]] | None:
    """``({aggregate output: MIN or MAX}, {every output name})`` of a grouped source this rule reads, else None."""

    if any(value for key, value in inner.args.items() if key not in _INNER_ALLOWED and value):
        return None
    group = inner.args.get("group")
    if group is None or extended_grouping(group) or not group.expressions:
        return None
    if any(isinstance(n, exp.Window) or (isinstance(n, exp.Star) and not isinstance(n.parent, exp.Count)) for n in inner.walk()):
        return None
    keys = {_unwrap(k).sql() for k in group.expressions}
    if any(isinstance(n, (exp.AggFunc, exp.Subquery, exp.Literal)) for k in group.expressions for n in [_unwrap(k)]):
        return None  # GROUP BY 1 is an ordinal, not a key
    aggregates: dict[str, type] = {}
    names: set[str] = set()
    for item in inner.expressions:
        name = (item.alias_or_name or "").lower()
        value = _unwrap(item.this if isinstance(item, exp.Alias) else item)
        if not name or name in names:
            return None
        names.add(name)
        if value.sql() in keys:
            continue
        if not _plain_extreme(value) or not isinstance(item, exp.Alias):
            return None
        if any(isinstance(n, (exp.AggFunc, exp.Subquery, exp.Exists, exp.Window)) for n in value.this.walk()):
            return None
        aggregates[name] = type(value)
    return aggregates, names


def read_extreme_group_source(select: exp.Select) -> exp.Expression | None:
    """Drop the ``GROUP BY`` of the pre-grouped sources of ``select`` when ``select`` cannot tell the difference.

    The sources' own sources come first: a level is read while the select above it still aggregates, so a
    chain of pre-grouped levels comes apart in one call, from the inside.
    """

    changed = False
    for source in select_sources(select):
        if isinstance(source, exp.Subquery) and isinstance(source.this, exp.Select):
            changed = read_extreme_group_source(source.this) is not None or changed
    changed = _ungroup_sources(select) or changed
    return select if changed else None


def _ungroup_sources(select: exp.Select) -> bool:
    if any(select.args.get(key) for key in _OUTER_BANNED):
        return False
    group = select.args.get("group")
    if group is not None and extended_grouping(group):
        return False
    own_aggregates = [n for n in select.find_all(exp.AggFunc) if _own(n, select)]
    if not own_aggregates and group is None:
        return False
    if not all(_plain_extreme(call) for call in own_aggregates):
        return False
    if any(isinstance(n, exp.Window) for n in select.walk()):
        return False
    joins = select.args.get("joins") or []
    if any((j.args.get("side") or "").upper() in ("RIGHT", "FULL") or j.args.get("method") or j.args.get("using") for j in joins):
        return False
    from_ = select.args.get("from_") or select.args.get("from")
    if from_ is None:
        return False
    candidates = [from_.this] + [
        j.this for j in joins if not (j.args.get("side") or "") and (j.args.get("kind") or "").upper() in _JOIN_KINDS
    ]
    sources = select_sources(select)
    if any(isinstance(s, (exp.Lateral, exp.Unnest)) for s in sources):
        return False  # a lateral source reads its siblings
    source_ids = {id(s) for s in sources}
    aliases = [(s.alias_or_name or "").lower() for s in sources]
    if any(isinstance(n, exp.Star) and not isinstance(n.parent, exp.Count) and _own(n, select) for n in select.find_all(exp.Star)):
        return False
    if any(not _inside_source(n, select, source_ids) for n in select.find_all(exp.Anonymous, exp.WithinGroup)):
        return False  # a function this module does not know may be an aggregate that counts its rows
    columns = [c for c in select.find_all(exp.Column) if not isinstance(c.this, exp.Star)]
    changed = False
    for source in candidates:
        if not isinstance(source, exp.Subquery) or not isinstance(source.this, exp.Select):
            continue
        alias_node = source.args.get("alias")
        alias = (source.alias or "").lower()
        if not alias or aliases.count(alias) != 1 or (alias_node is not None and alias_node.args.get("columns")):
            continue
        if any(value for key, value in source.args.items() if key not in ("this", "alias") and value):
            continue
        inner = source.this
        read = _grouped_outputs(inner)
        if read is None:
            continue
        aggregates, names = read
        ok = True
        for column in columns:
            if _inside_source(column, select, source_ids):
                continue  # a derived table reads its own sources, never its siblings
            name, table = column.name.lower(), column.table.lower()
            if not table:
                if name in names:
                    ok = False  # a bare name that may be this source's output
                    break
                continue
            if table != alias or not _binds_to(column, select, alias):
                continue
            if name not in names:
                ok = False  # not an output of the source: leave the query to the prover's own errors
                break
            if name in aggregates:
                parent = column.parent
                while isinstance(parent, exp.Paren):
                    parent = parent.parent
                if type(parent) is not aggregates[name] or not _plain_extreme(parent) or not _own(parent, select) or _unwrap(parent.this) is not column:
                    ok = False
                    break
        if not ok:
            continue
        for item in inner.expressions:
            value = _unwrap(item.this if isinstance(item, exp.Alias) else item)
            if isinstance(item, exp.Alias) and type(value) in _EXTREMES and item.alias.lower() in aggregates:
                item.set("this", value.this.copy())
        inner.set("group", None)
        changed = True
    return changed


def _inside_source(node: exp.Expression, select: exp.Select, source_ids: set[int]) -> bool:
    """Whether ``node`` sits inside one of ``select``'s own FROM or JOIN items."""

    node = node.parent
    while node is not None and node is not select:
        if id(node) in source_ids:
            return True
        node = node.parent
    return False


def _selects_above(node: exp.Expression):
    node = node.parent
    while node is not None:
        if isinstance(node, exp.Select):
            yield node
        node = node.parent


def _binds_to(column: exp.Column, select: exp.Select, alias: str) -> bool:
    """Whether ``alias`` on ``column`` names a source of ``select`` and not one of a select in between."""

    for scope in _selects_above(column):
        if scope is select:
            return True
        if any((s.alias_or_name or "").lower() == alias for s in select_sources(scope)):
            return False
    return False
