"""A joined one-row table of aggregates is a scalar subquery at each place it is read.

    SELECT v.k, v.total
    FROM (SELECT ps.k, SUM(ps.q) AS total FROM ps GROUP BY ps.k) AS v,
         (SELECT SUM(ps.q) * 2 AS bound FROM ps) AS t
    WHERE v.total > t.bound

is

    SELECT v.k, v.total
    FROM (SELECT ps.k, SUM(ps.q) AS total FROM ps GROUP BY ps.k) AS v
    WHERE v.total > (SELECT SUM(ps.q) * 2 FROM ps)

(TPC-H Q11 and the rewrites that compute its threshold once, in a CTE that is cross joined).

Why this is exact. A select whose outputs are aggregates, with no ``GROUP BY`` and no ``HAVING``, returns exactly
one row whatever its sources hold. A cross join with one row repeats nothing and drops nothing, an inner join
``ON c`` with it is a filter ``c``, and a ``LEFT JOIN .. ON TRUE`` with it always finds its one row. So the join can
go, and each read of an output is the value a scalar subquery computing that output returns: the derived table is
not correlated (it is not ``LATERAL``) and holds nothing volatile, so every evaluation gives the same value.

Conditions, all checked:

* the select's joins are commas, ``CROSS JOIN`` or inner ``JOIN .. ON`` (no ``USING``, ``NATURAL``, ``LATERAL``),
  apart from the one-row table itself, which may also be joined with ``LEFT JOIN .. ON TRUE``; there is at least one
  other source, and the one-row table is not the first source of a select whose next join has an ``ON``-less
  outer shape (the sources are otherwise left as written);
* the one-row table has an alias without a column list and is a select with only a select list, sources, joins
  and ``WHERE``; every output has a name, holds an aggregate of that select, and reads columns only inside
  aggregates; no window, no nested aggregate, nothing volatile;
* every read of it is a qualified column ``alias.output``, or a bare name that only it can own: every other source
  is a base table the schema lists or a derived table with named outputs and no star, and none has that name;
* no read sits in ``GROUP BY``, and no other source or nested query uses the alias;
* its outputs are not all plain ``COUNT`` calls: that is the guard table of Calcite's ``NOT IN`` expansion, which
  ``expanded_not_in.drop_count_guards`` reads while it is still a table.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, select_sources
from .keyed_join_groups import _and_all, declared_columns
from .unmatched_outer_join import _columns_of, _conjuncts, _inside, _plain_column, _plain_table, _unwrap

_ROW_ARGS = frozenset({"expressions", "from_", "from", "joins", "where"})


def _one_row(source: exp.Expression) -> dict[str, exp.Expression] | None:
    """``{output: expression}`` when ``source`` is a derived table of aggregates that always has exactly one row."""

    from .counted_membership import _unstable

    if type(source) is not exp.Subquery or not source.alias or type(source.this) is not exp.Select:
        return None
    if source.args["alias"].args.get("columns") or any(v for key, v in source.args.items() if key not in ("this", "alias")):
        return None
    inner = source.this
    if any(value for key, value in inner.args.items() if key not in _ROW_ARGS) or _unstable(inner):
        return None
    outputs: dict[str, exp.Expression] = {}
    for item in inner.expressions:
        name = (item.alias_or_name or "").lower()
        value = item.this if isinstance(item, exp.Alias) else item
        if not name or name in outputs or not isinstance(item, exp.Alias):
            return None
        aggregates = [a for a in value.find_all(exp.AggFunc) if a.find_ancestor(exp.Select) is inner]
        if not aggregates or any(isinstance(n, exp.Window) for n in value.walk()):
            return None
        if any(isinstance(a.parent, (exp.Filter, exp.WithinGroup)) or any(isinstance(n, exp.AggFunc) and n is not a for n in a.walk()) for a in aggregates):
            return None
        for column in value.find_all(exp.Column):
            if column.find_ancestor(exp.Select) is inner and not any(_inside(column, a) for a in aggregates):
                return None
        outputs[name] = value
    return outputs or None


def _output_names(source: exp.Expression, declared: dict[str, set[str]]) -> set[str] | None:
    """The column names a source offers, or ``None`` when they are not known."""

    if type(source) is exp.Table:
        return _columns_of(source, declared) if _plain_table(source) else None
    if type(source) is exp.Subquery and type(source.this) is exp.Select and not source.args["alias"].args.get("columns") if source.args.get("alias") else False:
        names = set()
        for item in source.this.expressions:
            name = (item.alias_or_name or "").lower()
            if not name or isinstance(item, exp.Star) or item.is_star:
                return None
            names.add(name)
        return names
    return None


def read_one_row_source(select: exp.Select, schema: dict | None) -> exp.Expression | None:
    """``select`` with one joined one-row table of aggregates read as scalar subqueries, or ``None``."""

    joins = select.args.get("joins") or []
    if not joins or select.args.get("laterals"):
        return None
    sources = select_sources(select)
    names = [(s.alias_or_name or "").lower() for s in sources]
    if not all(names) or len(set(names)) != len(names):
        return None
    declared = declared_columns(schema or {})
    for position, source in enumerate(sources):
        outputs = _one_row(source)
        if outputs is None:
            continue
        if all(type(value) is exp.Count for value in outputs.values()):
            continue  # a table of plain counts is Calcite's NOT IN guard: ``expanded_not_in`` reads it as a table
        alias = names[position]
        # the joins: plain everywhere, and the one-row table may also be LEFT JOINed ON TRUE
        ok = True
        for index, join in enumerate(joins):
            if join.args.get("using") or join.args.get("method") or isinstance(join.this, exp.Lateral):
                ok = False
            elif any(v for key, v in join.args.items() if key not in ("this", "kind", "on", "side")):
                ok = False
            elif join.args.get("side"):
                on = join.args.get("on")
                ok = ok and index + 1 == position and (join.args.get("side") or "").upper() == "LEFT" and on is not None and type(_unwrap(on)) is exp.Boolean and _unwrap(on).this is True
            elif (join.args.get("kind") or "").upper() not in ("", "INNER", "CROSS"):
                ok = False
        if not ok:
            continue
        # no nested query or other source may use the alias
        if any((s.alias_or_name or "").lower() == alias for nested in select.find_all(exp.Select) if nested is not select and not _inside(nested, source) for s in select_sources(nested)):
            continue
        others = [_output_names(item, declared) for index, item in enumerate(sources) if index != position]
        group = select.args.get("group")
        reads = []
        for column in select.find_all(exp.Column):
            if _inside(column, source):
                continue
            if column.table:
                if column.table.lower() != alias:
                    continue
                if _plain_column(column) is not column or len(column.parts) != 2 or column.name.lower() not in outputs:
                    ok = False
                    break
            else:
                if column.name.lower() not in outputs:
                    continue
                nearest = column.find_ancestor(exp.Select)
                if nearest is not select or any(owned is None or column.name.lower() in owned for owned in others):
                    ok = False
                    break
                if column.find_ancestor(exp.Order) is not None and any((item.alias_or_name or "").lower() == column.name.lower() for item in select.expressions):
                    ok = False  # could mean the output of that name
                    break
            if group is not None and _inside(column, group):
                ok = False
                break
            reads.append(column)
        if not ok or any(isinstance(n, exp.Star) and not isinstance(n.parent, exp.Count) for item in select.expressions for n in item.walk()):
            continue

        copy = select.copy()
        new_sources = select_sources(copy)
        new_source = new_sources[position]
        new_inner = new_source.this
        new_joins = copy.args.get("joins") or []
        for column in list(copy.find_all(exp.Column)):
            if _inside(column, new_source):
                continue
            if (column.table.lower() == alias) or (not column.table and column.name.lower() in outputs and column.find_ancestor(exp.Select) is copy):
                scalar = new_inner.copy()
                scalar.set("expressions", [outputs[column.name.lower()].copy()])
                column.replace(exp.Subquery(this=scalar))
        moved: list[exp.Expression] = []
        if position == 0:
            first = new_joins[0]
            if first.args.get("side"):
                continue
            moved = _conjuncts(first.args.get("on"))
            copy.set(FROM_KEY, exp.From(this=first.this.copy()))
            first.pop()
        else:
            join = new_joins[position - 1]
            if not join.args.get("side"):
                moved = _conjuncts(join.args.get("on"))
            join.pop()
        if any(not column.table for part in moved for column in part.find_all(exp.Column) if column.find_ancestor(exp.Select) is copy or column.find_ancestor(exp.Select) is None):
            continue  # a bare name would meet more sources in WHERE than it did in ON
        if not copy.args.get("joins"):
            copy.set("joins", None)
        if moved:
            where = copy.args.get("where")
            parts = [p.copy() for p in _conjuncts(where.this if where is not None else None)] + [p.copy() for p in moved]
            copy.set("where", exp.Where(this=_and_all(parts)))
        return copy
    return None
