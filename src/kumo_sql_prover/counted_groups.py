"""``COUNT(*)`` over a finer grouping counts the extra key when that key is never NULL.

    SELECT d.g, COUNT(*) FROM (SELECT a.g, a.x FROM a GROUP BY a.g, a.x) AS d GROUP BY d.g

has one inner row per distinct ``(g, x)``, so the outer count is the number of distinct ``x`` in the group: the
usual way to write ``COUNT(DISTINCT a.x) .. GROUP BY a.g`` without ``DISTINCT``. ``distinct_rules.regroup_distinct``
already folds ``COUNT(d.x)`` over such a derived table into ``COUNT(DISTINCT a.x)``; it declines ``COUNT(*)``, and
rightly so in general, because ``COUNT(*)`` also counts the inner row whose ``x`` is NULL and ``COUNT(DISTINCT x)``
does not. When ``x`` cannot be NULL the two counts are the same number on every group (and both are 0 over no rows
without a ``GROUP BY``), so ``count_extra_key`` writes ``COUNT(*)`` as ``COUNT(d.x)``, adding ``x`` to the derived
table's outputs if it is not one, and leaves the rest to that rule.

``COUNT(*) = COUNT(d.x)`` needs only that no row of the derived table has a NULL ``x``. The rule still fires only
in the shape the other rule finishes, so that it does not respell counts elsewhere:

* the outer select reads the derived table alone, with no ``WHERE``, join, ``HAVING``, ``DISTINCT``, window or tail,
  and its ``GROUP BY`` (if any) lists outputs of the derived table that are grouping columns of it;
* the derived table groups by plain columns, has no ``HAVING``, ``DISTINCT``, window or tail, and its outputs are
  grouping columns under distinct names;
* exactly one grouping column ``x`` of the derived table is not among the outer keys;
* ``x`` is never NULL: it is a column, declared NOT NULL, of a base table that the derived table reads through
  inner and cross joins only (an outer join can pad it with NULLs).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import extended_grouping, select_sources
from .unmatched_outer_join import _columns_of, _plain_column, _plain_table

_OUTER_ARGS = frozenset({"expressions", "from_", "from", "group"})
_INNER_ARGS = frozenset({"expressions", "from_", "from", "joins", "where", "group"})


def _never_null(column: exp.Column, inner: exp.Select, declared: dict[str, set[str]], never_null: dict[str, set[str]]) -> bool:
    """Whether ``column`` is a NOT NULL column of a base table that ``inner`` reads through inner joins only."""

    joins = inner.args.get("joins") or []
    if any(j.args.get("side") or j.args.get("using") or j.args.get("method") for j in joins):
        return False
    if any((j.args.get("kind") or "").upper() not in ("", "INNER", "CROSS") for j in joins):
        return False
    sources = select_sources(inner)
    if not sources or not all(_plain_table(s) or type(s) is exp.Subquery for s in sources):
        return False
    named = [s for s in sources if (s.alias_or_name or "").lower() == column.table.lower()] if column.table else sources
    if len(named) != 1 or (not column.table and len(sources) != 1) or not _plain_table(named[0]):
        return False
    table = named[0]
    if column.name.lower() not in (_columns_of(table, declared) or ()):
        return False
    full = ".".join(part.name for part in table.parts).lower()
    return column.name.lower() in (never_null.get(full) or never_null.get(table.name.lower()) or ())


def _star_counts(select: exp.Select) -> list[exp.Count]:
    """The plain ``COUNT(*)`` calls of ``select`` itself (not windowed, not filtered, not of a nested select)."""

    return [
        node for node in select.find_all(exp.Count)
        if isinstance(node.this, exp.Star) and node.find_ancestor(exp.Select) is select
        and not any(v for key, v in node.args.items() if key not in ("this", "big_int"))
        and not isinstance(node.parent, (exp.Window, exp.Filter))
    ]


def count_extra_key(select: exp.Select, schema: dict | None, not_null: dict | None) -> exp.Expression | None:
    """``select`` with its ``COUNT(*)`` written as a count of the derived table's one extra grouping key, or ``None``."""

    if not schema or any(value for key, value in select.args.items() if key not in _OUTER_ARGS):
        return None
    from_ = select.args.get("from_") or select.args.get("from")
    source = from_.this if from_ is not None else None
    if type(source) is not exp.Subquery or not source.alias or type(source.this) is not exp.Select:
        return None
    if source.args["alias"].args.get("columns") or any(v for key, v in source.args.items() if key not in ("this", "alias")):
        return None
    if not _star_counts(select):
        return None
    inner = source.this
    if any(value for key, value in inner.args.items() if key not in _INNER_ARGS):
        return None
    group = inner.args.get("group")
    if group is None or extended_grouping(group) or not group.expressions:
        return None
    keys = list(group.expressions)
    if any(_plain_column(k) is None for k in keys):
        return None
    key_sql = [k.sql() for k in keys]
    outputs: dict[str, str] = {}
    for item in inner.expressions:
        value = item.this if isinstance(item, exp.Alias) else item
        name = (item.alias_or_name or "").lower()
        if not name or name in outputs or type(value) is not exp.Column or value.sql() not in key_sql:
            return None
        outputs[name] = value.sql()
    alias = source.alias.lower()
    outer_group = select.args.get("group")
    kept: set[str] = set()
    if outer_group is not None:
        if extended_grouping(outer_group):
            return None
        for key in outer_group.expressions:
            column = _plain_column(key)
            if column is None or column.table.lower() not in ("", alias) or column.name.lower() not in outputs:
                return None
            kept.add(outputs[column.name.lower()])
    extra = [k for k in keys if k.sql() not in kept]
    if len(extra) != 1 or len(set(key_sql)) != len(key_sql):
        return None
    declared = {str(t).lower(): {str(c).lower() for c in columns} for t, columns in schema.items()}
    never_null = {str(t).lower(): {str(c).lower() for c in columns} for t, columns in (not_null or {}).items()}
    if not _never_null(extra[0], inner, declared, never_null):
        return None
    copy = select.copy()
    new_source = (copy.args.get("from_") or copy.args.get("from")).this
    name = next((n for n, sql in outputs.items() if sql == extra[0].sql()), None)
    if name is None:
        name = "kqx"
        while name in outputs:
            name += "x"
        new_source.this.append("expressions", exp.alias_(extra[0].copy(), name))
    for node in _star_counts(copy):
        node.set("this", exp.Column(this=exp.to_identifier(name), table=source.args["alias"].this.copy()))
    return copy
