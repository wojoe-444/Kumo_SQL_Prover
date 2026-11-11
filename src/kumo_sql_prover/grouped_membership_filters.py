"""A test on group keys that holds a subquery filters the rows before the grouping as well as the groups after it.

Two placements of one identity. ``having_rules`` and ``algebraic_equivalence._push_filter_into_derived`` already
move a condition on group keys that holds no subquery; ``semijoin_shapes`` step 3 moves one with closed
subqueries into a ``SELECT DISTINCT``. Here the condition holds subqueries and the keys are those of a
``GROUP BY``:

**1. From ``HAVING`` to ``WHERE``.**

    SELECT k, COUNT(*) FROM t GROUP BY k HAVING k IN (SELECT c FROM u)
    SELECT k, COUNT(*) FROM t WHERE k IN (SELECT c FROM u) GROUP BY k

**2. From above a grouped derived table into it.**

    SELECT d.k, d.n FROM (SELECT g AS k, COUNT(*) AS n FROM t GROUP BY g) AS d WHERE EXISTS (SELECT 1 FROM u WHERE u.c = d.k)
    SELECT d.k, d.n FROM (SELECT g AS k, COUNT(*) AS n FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.c = t.g) GROUP BY g) AS d

**Why it is sound.** The condition ``P`` reads, of the grouped query, only group keys; everything else in it is
a subquery over its own tables. So ``P`` has one value for all rows of a group, the value it has for the
group: the keys are equal across the group as the grouping compares them (all NULL in the NULL group), and
the subqueries, having no other tie to the row, are the same queries for each of them. Keeping the groups for
which ``P`` is TRUE therefore keeps exactly the groups whose rows all pass ``P``, and dropping the rows that
fail ``P`` before grouping removes whole groups and leaves the others, with all their rows and so all their
aggregates, untouched. A group of the derived table that its own ``HAVING`` drops is dropped either way. At
least one key is needed: a global aggregate returns its row even when ``WHERE`` removes every input row.

Moving ``P`` changes the scope its names are read in, which the rule checks rather than assumes:

* A key is read through its output name above the derived table and as the key expression inside it. Inside a
  subquery of ``P`` the key expression must be a column qualified with a name no select of ``P`` declares;
  a bare key column of a derived table with one named source is qualified with that source's name first.
* Every other column inside a subquery of ``P`` must be bound inside ``P`` (qualified by an alias a select of
  ``P`` declares, or a bare column of the one table its select reads, by the supplied schema). A name bound
  further out is refused: one level further in, the derived table's own sources could capture it.

**What is read.** A top-level conjunct of ``HAVING`` (form 1) or of the ``WHERE`` of a select whose only source is
a derived table (form 2) that holds at least one subquery; a plain ``GROUP BY`` with at least one key (no
``ROLLUP``, ``CUBE``, grouping sets); in form 1 the keys read are ``GROUP BY`` columns spelled as in the
``GROUP BY`` and not shadowed by a select alias; in form 2 the derived table has no ``DISTINCT``, ``LIMIT``,
``ORDER BY`` or window, and the outputs read are non-aggregate outputs that are ``GROUP BY`` items.

**What is refused.** A conjunct with an aggregate or window of the grouped select, with a volatile function, or
with a read of anything but keys; a select with window functions or ``QUALIFY``; conjuncts without a subquery
(the existing rules own those, and the split keeps this rule from changing what they did).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import conjuncts, distinct_on, extended_grouping, plain_distinct
from .decorrelation_rules import _and_all, _bound_by, _bound_within, _sources
from .semijoin_shapes import _VOLATILE, _derived, _own, _plain_column, _plain_table


def _plain_keys(select: exp.Select) -> exp.Group | None:
    group = select.args.get("group")
    if group is None or not group.expressions or extended_grouping(group):
        return None
    if any(value for key, value in group.args.items() if key != "expressions"):
        return None
    if any(select.args.get(k) for k in ("qualify", "windows", "laterals")) or any(_own(w, select) for w in select.find_all(exp.Window)):
        return None
    return group


def _bare_bound(column: exp.Column, schema: dict | None) -> bool:
    """Whether a bare column inside a subquery is a listed column of the one table its select reads."""

    inner = column.find_ancestor(exp.Select)
    if inner is None:
        return False
    tables = _sources(inner)
    if len(tables) != 1 or not _plain_table(tables[0]):
        return False
    known = {k.lower(): [c.lower() for c in v] for k, v in (schema or {}).items()}.get(tables[0].name.lower())
    return known is not None and column.name.lower() in known


def _movable(part: exp.Expression, select: exp.Select, schema: dict | None, key_of) -> list[tuple[exp.Column, object]] | None:
    """The key reads of ``part`` as ``(column, key)`` pairs when every name in it is a key or bound inside it."""

    nested = [n for n in part.find_all(exp.Select)]
    if not nested:
        return None
    if any(isinstance(n, (exp.AggFunc, exp.Window)) and _own(n, select) for n in part.walk()):
        return None
    if any(isinstance(n, _VOLATILE) for n in part.walk()):
        return None
    reads = []
    for column in part.find_all(exp.Column):
        if isinstance(column.this, exp.Star) or column.args.get("db") or column.args.get("catalog"):
            return None
        own = _own(column, select)
        if not own and column.table and _bound_within(column, part):
            continue
        if not own and not column.table and _bare_bound(column, schema):
            continue
        key = key_of(column, own)
        if key is None:
            return None
        reads.append((column, key))
    return reads


def having_membership_to_where(select: exp.Select, schema: dict | None) -> exp.Select | None:
    """Form 1 (in place)."""

    having = select.args.get("having")
    group = _plain_keys(select)
    if having is None or group is None:
        return None
    keys = {k.sql().lower() for k in group.expressions if _plain_column(k) is not None}
    shadowed = {
        item.alias.lower() for item in select.expressions
        if isinstance(item, exp.Alias) and not (isinstance(item.this, exp.Column) and item.this.name.lower() == item.alias.lower())
    }
    sources = _sources(select)
    names = {(s.alias_or_name or "").lower() for s in sources}

    def key_of(column: exp.Column, own: bool):
        if column.sql().lower() not in keys or column.name.lower() in shadowed:
            return None
        if not own and (not column.table or column.table.lower() not in names):
            return None  # read from a subquery, a key must name its source: a bare name binds further in
        return True

    moved, kept = [], []
    for part in conjuncts(having.this):
        reads = _movable(part, select, schema, key_of)
        (moved if reads is not None else kept).append(part)
    if not moved:
        return None
    where = select.args.get("where")
    condition = _and_all((conjuncts(where.this) if where is not None else []) + [m.copy() for m in moved])
    remaining = _and_all([k.copy() for k in kept])
    select.set("where", exp.Where(this=condition))
    select.set("having", exp.Having(this=remaining) if remaining is not None else None)
    return select


def _source_column(value: exp.Expression, sources: list[exp.Expression], listed: dict[str, set[str]]) -> exp.Column | None:
    """``value`` as a column qualified with the name of the source it belongs to, when that is certain."""

    column = _plain_column(value)
    if column is None:
        return None
    names = [(s.alias_or_name or "").lower() for s in sources]
    if column.table:
        return column if names.count(column.table.lower()) == 1 else None
    if len(sources) != 1 or not names[0]:
        return None
    source = sources[0]
    if _plain_table(source):
        if column.name.lower() not in listed.get(source.name.lower(), ()):
            return None
    else:
        body = _derived(source)
        if body is None or column.name.lower() not in {i.alias_or_name.lower() for i in body.expressions}:
            return None
    return exp.column(column.name, table=source.alias_or_name)


def _key_outputs(inner: exp.Select, listed: dict[str, set[str]]) -> dict[str, exp.Expression] | None:
    """Output name -> key expression for the non-aggregate outputs of a plainly grouped derived select.

    An output is a key when it is spelled as a ``GROUP BY`` item, or when both are columns of one source (``t.g``
    and a bare ``g`` of the only source ``t``; a bare ``GROUP BY`` name that a select alias also carries is not
    matched this way, since dialects differ on which of the two it means)."""

    group = _plain_keys(inner)
    if group is None or plain_distinct(inner) or distinct_on(inner):
        return None
    if any(inner.args.get(k) for k in ("limit", "offset", "order", "with_", "with")):
        return None
    if any(isinstance(e, exp.Star) or isinstance(e.unalias(), exp.Column) and isinstance(e.unalias().this, exp.Star) for e in inner.expressions):
        return None
    sources = _sources(inner)
    aliased = {item.alias.lower() for item in inner.expressions if isinstance(item, exp.Alias) and item.alias}
    keys = {e.sql().lower() for e in group.expressions}
    for item in group.expressions:
        bare = _plain_column(item)
        if bare is not None and not bare.table and bare.name.lower() in aliased:
            continue
        column = _source_column(item, sources, listed)
        if column is not None:
            keys.add(column.sql().lower())
    outputs: dict[str, exp.Expression] = {}
    names: set[str] = set()
    for item in inner.expressions:
        name = item.alias_or_name.lower()
        value = item.unalias()
        if not name or name in names:
            return None
        names.add(name)
        if any(isinstance(n, (exp.Subquery, exp.Select, exp.Window, exp.AggFunc, *_VOLATILE)) for n in value.walk()):
            continue
        column = _source_column(value, sources, listed)
        if value.sql().lower() in keys or (column is not None and column.sql().lower() in keys):
            outputs[name] = value
    return outputs


def push_membership_into_grouped(select: exp.Select, schema: dict | None) -> exp.Select | None:
    """Form 2 (in place)."""

    where = select.args.get("where")
    sources = _sources(select)
    if where is None or len(sources) != 1 or select.args.get("laterals"):
        return None
    inner = _derived(sources[0])
    if inner is None:
        return None
    listed = {k.lower(): {c.lower() for c in v} for k, v in (schema or {}).items()}
    outputs = _key_outputs(inner, listed)
    if not outputs:
        return None
    alias = sources[0].alias.lower()
    inner_sources = _sources(inner)

    def key_of(column: exp.Column, own: bool):
        if column.table.lower() not in (("", alias) if own else (alias,)) or column.name.lower() not in outputs:
            return None
        value = outputs[column.name.lower()]
        return value if own else _source_column(value, inner_sources, listed)

    moved, kept, pushed = [], [], []
    for part in conjuncts(where.this):
        reads = _movable(part, select, schema, key_of)
        if reads is not None:
            declared = set().union(*[_bound_by(n) for n in part.find_all(exp.Select)])
            if any(not _own(column, select) and key.table.lower() in declared for column, key in reads):
                reads = None  # a subquery of the conjunct declares the name the key would be qualified with
            elif alias in declared:
                reads = None
        if reads is None:
            kept.append(part)
            continue
        holder = exp.Paren(this=part.copy())
        position = {id(c): i for i, c in enumerate(part.find_all(exp.Column))}
        copies = list(holder.this.find_all(exp.Column))
        for column, key in reads:
            target = copies[position[id(column)]]
            target.replace(exp.Paren(this=key.copy()) if not isinstance(key, exp.Column) else key.copy())
        moved.append(part)
        pushed.append(holder.this)
    if not moved:
        return None
    existing = inner.args.get("where")
    inner.set("where", exp.Where(this=_and_all((conjuncts(existing.this) if existing is not None else []) + pushed)))
    remaining = _and_all([k.copy() for k in kept])
    select.set("where", exp.Where(this=remaining) if remaining is not None else None)
    return select


def grouped_membership_filters(select: exp.Select, schema: dict | None) -> exp.Expression | None:
    """``select`` after one of the two moves, else ``None``."""

    return having_membership_to_where(select, schema) or push_membership_into_grouped(select, schema)
