"""``DISTINCT`` or ``GROUP BY`` over the keys of a grouped derived table removes nothing.

**The rule.** A derived table ``d`` that is a plain ``GROUP BY`` select has one row per combination of its group
keys. In a select that reads only ``d``:

* ``SELECT DISTINCT`` whose select list holds, as plain columns of ``d``, an output for every group key of ``d``
  returns no repeated row, so the ``DISTINCT`` is dropped;
* a ``GROUP BY`` (without aggregates and without ``HAVING``) whose items are plain columns of ``d`` holding an
  output for every group key of ``d`` puts every row of ``d`` in its own group, so the ``GROUP BY`` is dropped.

    SELECT DISTINCT d.k, d.n FROM (SELECT g AS k, COUNT(*) AS n FROM t GROUP BY g) AS d WHERE d.n > 1
    SELECT          d.k, d.n FROM (SELECT g AS k, COUNT(*) AS n FROM t GROUP BY g) AS d WHERE d.n > 1

**Why it is sound.** ``GROUP BY`` forms one group per combination of key values, comparing two NULLs as equal,
so two different rows of ``d`` differ in at least one key output under that comparison. ``DISTINCT`` and an
outer ``GROUP BY`` compare rows the same way (NULLs equal). The outer ``WHERE`` only removes rows. With every
key of ``d`` among the compared columns, no two rows that reach the ``DISTINCT`` are equal, and no two rows
share an outer group; a ``DISTINCT`` over rows that are all different, and a grouping whose groups are single
rows read without aggregates, both return their input. In the single-row group a column that is not an outer
``GROUP BY`` item (MySQL allows reading one) has that row's value, which is what the select without the
``GROUP BY`` returns.

**What is read.** The derived table: a select with a plain ``GROUP BY`` of at least one key and no ``ROLLUP``,
``CUBE``, grouping sets, window, ``LIMIT`` or ``ORDER BY``, whose every ``GROUP BY`` item is one of its outputs
(spelled as the item, or the same column of the same source: ``t.g`` and a bare ``g`` of the only source
``t``). A ``HAVING`` or a ``DISTINCT`` in it only removes rows. The outer select: one source, only a select list,
``WHERE``, ``DISTINCT`` or ``GROUP BY``, and ``ORDER BY``; no aggregate or window function of its own.

**What is refused.** A join (the other side may repeat the rows of ``d``), a derived table with a group key it
does not output (two of its rows may then look alike), an outer select list or ``GROUP BY`` that misses a key or
holds it only inside an expression, ``DISTINCT ON``, an outer ``HAVING`` or ``LIMIT``, and aggregates over the
single-row groups (other rules read those).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import distinct_on, extended_grouping, plain_distinct
from .decorrelation_rules import _sources
from .semijoin_shapes import _derived, _only, _own, _plain_column, _plain_table

_OUTER = frozenset({"expressions", "from_", "from", "where", "distinct", "group", "order"})


def _source_text(value: exp.Expression, sources: list[exp.Expression], listed: dict[str, set[str]]) -> str | None:
    """``value`` spelled as ``source.column`` when it is certainly a column of one source of its select."""

    column = _plain_column(value)
    if column is None:
        return None
    names = [(s.alias_or_name or "").lower() for s in sources]
    if column.table:
        return f"{column.table.lower()}.{column.name.lower()}" if names.count(column.table.lower()) == 1 else None
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
    return f"{names[0]}.{column.name.lower()}"


def renaming_aliases(select: exp.Select) -> set[str]:
    """The select aliases that give a name to something other than the column of that name."""

    return {
        item.alias.lower() for item in select.expressions
        if isinstance(item, exp.Alias) and item.alias and not (_plain_column(item.this) is not None and item.this.name.lower() == item.alias.lower())
    }


def unique_key_outputs(source: exp.Expression, schema: dict | None) -> list[set[str]] | None:
    """For a plainly grouped derived table, the output names that carry each of its group keys (one set per
    ``GROUP BY`` item, none empty): its rows are unique on any choice of one name per set. Else ``None``."""

    inner = _derived(source)
    if inner is None:
        return None
    group = inner.args.get("group")
    if group is None or not group.expressions or extended_grouping(group) or distinct_on(inner):
        return None
    if any(value for key, value in group.args.items() if key != "expressions"):
        return None
    if any(inner.args.get(k) for k in ("limit", "offset", "order", "qualify", "windows", "laterals", "with_", "with")):
        return None
    if any(_own(w, inner) for w in inner.find_all(exp.Window)):
        return None
    if any(isinstance(e, exp.Star) or isinstance(e.unalias(), exp.Column) and isinstance(e.unalias().this, exp.Star) for e in inner.expressions):
        return None
    listed = {k.lower(): {c.lower() for c in v} for k, v in (schema or {}).items()}
    sources = _sources(inner)
    names = [item.alias_or_name.lower() for item in inner.expressions]
    if "" in names or len(set(names)) != len(names):
        return None
    renamed = renaming_aliases(inner)
    carried: list[set[str]] = []
    for item in group.expressions:
        bare = _plain_column(item)
        if bare is not None and not bare.table and bare.name.lower() in renamed:
            return None  # a bare GROUP BY name that another output's alias also carries: dialects differ on which it means
        spellings = {item.sql().lower()}
        text = _source_text(item, sources, listed)
        if text is not None:
            spellings.add(text)
        found = set()
        for output in inner.expressions:
            value = output.unalias()
            if any(isinstance(n, (exp.Subquery, exp.Select, exp.Window, exp.AggFunc)) for n in value.walk()):
                continue
            if value.sql().lower() in spellings or _source_text(value, sources, listed) in spellings:
                found.add(output.alias_or_name.lower())
        if not found:
            return None
        carried.append(found)
    return carried


def drop_dedup_over_grouped(select: exp.Select, schema: dict | None) -> exp.Expression | None:
    """``select`` (changed in place) without a ``DISTINCT`` or ``GROUP BY`` that its grouped source makes redundant, else ``None``."""

    sources = _sources(select)
    distinct, group = plain_distinct(select), select.args.get("group")
    if len(sources) != 1 or not _only(select, _OUTER) or (not distinct and group is None):
        return None
    if select.args.get("distinct") is not None and not distinct:
        return None
    carried = unique_key_outputs(sources[0], schema)
    if carried is None:
        return None
    if any(isinstance(n, (exp.AggFunc, exp.Window)) and _own(n, select) for n in select.walk()):
        return None
    alias = sources[0].alias.lower()

    def covered(items: list[exp.Expression]) -> bool:
        held = set()
        for item in items:
            column = _plain_column(item.unalias())
            if column is not None and column.table.lower() in ("", alias):
                held.add(column.name.lower())
        return all(names & held for names in carried)

    if group is not None:
        if extended_grouping(group) or any(value for key, value in group.args.items() if key != "expressions") or not covered(group.expressions):
            return None
        select.set("group", None)
        return select
    if not covered(select.expressions):
        return None
    select.set("distinct", None)
    return select
