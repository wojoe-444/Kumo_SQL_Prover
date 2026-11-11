"""Collapse a distinct-aggregate expansion written as a plan: projection layers and ``CASE`` arms.

Calcite's ``AggregateExpandDistinctAggregatesRule`` computes ``COUNT(DISTINCT x)`` next to other aggregates by
grouping twice. The inner aggregate groups by ``GROUPING SETS``, one set per distinct argument plus the plain
keys, and tags each row with ``GROUPING(..)``; the outer aggregate regroups by the plain keys and lets each of its
aggregates read the rows of one set::

    SELECT p.k, COUNT(CASE WHEN p.g0 THEN p.x END), MIN(CASE WHEN p.g1 THEN p.s END)
    FROM (SELECT g.k, g.x, g.s, g.tag = 0 AS g0, g.tag = 1 AS g1
          FROM (SELECT t.k, t.x, SUM(t.y) AS s, GROUPING(t.k, t.x) AS tag
                FROM t GROUP BY GROUPING SETS ((t.k, t.x), (t.k))) AS g) AS p
    GROUP BY p.k

is ``SELECT t.k, COUNT(DISTINCT t.x), SUM(t.y) FROM t GROUP BY t.k``. ``grouping_expansion`` reads this when the
outer aggregates are written ``agg(x) FILTER (WHERE tag = n)`` directly above the grouped select. A plan translated
to SQL (``tools/calcite_plan_to_sql.py``) spells the filter as a ``CASE`` arm and computes the flags in projection
layers between the two aggregates; this module reads that spelling, and a few more of the rule's outputs.

Why the collapse is exact. Let ``K`` be the outer keys and require ``K`` to be part of every grouping set. Every set
then groups the same rows by at least ``K``, so the outer groups are the ``K``-groups of the source, whichever
sets the outer aggregates read, and no outer key is ever a padded NULL. Inside one ``K``-group, the rows of the set
``C`` an aggregate reads (the one set its flag selects) are the distinct value tuples of ``C``'s other keys:

* ``C = K``: exactly one row. ``MIN``, ``MAX`` or ``SUM`` of an inner aggregate is that aggregate.
* ``COUNT(x)``, ``COUNT(CASE WHEN x IS NOT NULL AND y IS NOT NULL THEN 1 END)`` with ``C = K + {x, y}``: the number
  of distinct non-NULL tuples, ``COUNT(DISTINCT x, y)``.
* ``SUM(x)``, ``MIN(x)``, ``MAX(x)`` with ``C = K + {x}``: ``SUM(DISTINCT x)``, ``MIN(x)``, ``MAX(x)``.
* a flag that also asks ``f IS TRUE`` of grouped boolean keys ``f``, with ``C = K + {x} + {f..}``: the tuples
  with ``f`` true are the distinct ``x`` of the rows where ``f`` holds, ``COUNT(DISTINCT CASE WHEN f THEN x END)``.
* ``SUM`` of an inner ``SUM``, ``MIN`` of ``MIN``, ``MAX`` of ``MAX`` over any set: the aggregate over the
  ``K``-group (NULLs are skipped at both levels). ``SUM`` of an inner ``COUNT`` is **not** read: over no rows it is
  NULL where the count is 0.

Declined, never approximated: an aggregate whose flag selects no set or several, a set that is not exactly the
keys the aggregate reads (its rows would be distinct on more than the argument), a repeated set, an outer key
missing from some set, a layer that is more than a projection of plain expressions, ``HAVING``/``DISTINCT``/row
limits at any level, and every aggregate or expression not listed above.

This is a shape rule (see ``shape_fallback``).
"""

from __future__ import annotations

from sqlglot import exp

from .grouping_sets import is_grouping_call

MAX_LAYERS = 6
_PROJECTION_NODES = (exp.Column, exp.Identifier, exp.Literal, exp.Boolean, exp.Null, exp.Paren, exp.EQ, exp.And, exp.Is)
_LEVEL_EXTRAS = ("joins", "where", "having", "distinct", "qualify", "limit", "offset", "order", "windows", "with_", "with", "laterals", "pivots")
_INNER_EXTRAS = ("having", "distinct", "qualify", "limit", "offset", "order", "windows", "with_", "with", "pivots")
_OUTER_AGGS = (exp.Count, exp.Sum, exp.Min, exp.Max)


class _Decline(Exception):
    pass


def _key(node: exp.Expression) -> str:
    return node.sql().lower()


def _source(select: exp.Select) -> exp.Subquery | None:
    from_ = select.args.get("from_") or select.args.get("from")
    source = from_.this if from_ is not None else None
    if not isinstance(source, exp.Subquery) or not isinstance(source.this, exp.Select) or not source.alias:
        return None
    if any(value for key, value in source.args.items() if key not in ("this", "alias")) or source.args["alias"].args.get("columns"):
        return None
    return source


def _outputs(select: exp.Select) -> dict[str, exp.Expression]:
    names = [item.alias_or_name.lower() for item in select.expressions]
    if not all(names) or len(set(names)) != len(names) or any(item.is_star for item in select.expressions):
        raise _Decline
    return {name: item.this if isinstance(item, exp.Alias) else item for name, item in zip(names, select.expressions)}


def _substitute(node: exp.Expression, values: dict[str, exp.Expression], alias: str) -> exp.Expression:
    """``node`` with each column of the derived table ``alias`` replaced by the expression that computes it."""

    def value_of(column: exp.Column) -> exp.Expression:
        if column.table and column.table.lower() != alias or column.name.lower() not in values or column.args.get("db"):
            raise _Decline
        return values[column.name.lower()].copy()

    if isinstance(node, exp.Column):
        return value_of(node)
    copy = node.copy()
    for column in list(copy.find_all(exp.Column)):
        column.replace(value_of(column))
    return copy


def _plain(agg: exp.Expression) -> bool:
    return not agg.args.get("distinct") and not isinstance(agg.this, exp.Distinct) and not agg.expressions and agg.this is not None


def _atoms(node: exp.Expression) -> list[exp.Expression]:
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.And):
        return _atoms(node.left) + _atoms(node.right)
    return [node]


def _unparen(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def collapse_layered_expansion(select: exp.Select) -> exp.Expression | None:
    """The single grouped select ``select`` computes (see the module docstring), or ``None``."""

    try:
        return _collapse(select)
    except (_Decline, KeyError, ValueError):
        return None


def _collapse(select: exp.Select) -> exp.Expression | None:
    if any(select.args.get(k) for k in _LEVEL_EXTRAS) or select.find(exp.Window):
        return None
    if not any(isinstance(node, exp.AggFunc) for node in select.find_all(exp.AggFunc)):
        return None
    source = _source(select)
    if source is None:
        return None
    # ---- the projection layers between the two aggregates ----
    layers: list[tuple[str, exp.Select]] = []
    inner = source.this
    alias = source.alias.lower()
    while inner.args.get("group") is None:
        if len(layers) >= MAX_LAYERS or any(inner.args.get(k) for k in _LEVEL_EXTRAS):
            return None
        for item in inner.expressions:
            value = item.this if isinstance(item, exp.Alias) else item
            if any(not isinstance(node, _PROJECTION_NODES) for node in value.walk()):
                return None
        below = _source(inner)
        if below is None:
            return None
        layers.append((alias, inner))
        inner, alias = below.this, below.alias.lower()
    # ---- the grouped select ----
    if any(inner.args.get(k) for k in _INNER_EXTRAS) or inner.find(exp.Window):
        return None
    if any(isinstance(node, (exp.Subquery, exp.Select, exp.Exists)) for item in inner.expressions for node in item.walk()):
        return None
    group = inner.args["group"]
    if any(group.args.get(k) for k in ("cube", "rollup", "totals", "all")):
        return None
    listed = list(group.args.get("grouping_sets") or []) + [e for e in group.expressions if isinstance(e, exp.GroupingSets)]
    if listed:
        if len(listed) != 1 or any(not isinstance(e, exp.GroupingSets) for e in group.expressions):
            return None
        members = []
        for item in listed[0].expressions:
            item = _unparen(item)
            members.append(list(item.expressions) if isinstance(item, exp.Tuple) else [item])
    else:
        members = [list(group.expressions)]
    if not members or any(not isinstance(k, exp.Column) for one in members for k in one):
        return None
    sets = [frozenset(_key(k) for k in one) for one in members]
    if len(set(sets)) != len(sets):
        return None  # a repeated set repeats its groups
    all_keys = frozenset().union(*sets)

    values = _outputs(inner)  # output name -> the grouped select's own expression
    for layer_alias, layer in reversed(layers):
        values = {name: _substitute(value, values, alias) for name, value in _outputs(layer).items()}
        alias = layer_alias

    def grouped(node: exp.Expression) -> exp.Expression:
        """An expression of ``select`` in the grouped select's terms."""

        return _substitute(node, values, alias)

    def is_key(node: exp.Expression) -> bool:
        return isinstance(node, exp.Column) and _key(node) in all_keys

    def flag(node: exp.Expression, present: frozenset[str]):
        """The constant a flag expression has on the rows of the set ``present`` (raises for anything else)."""

        node = _unparen(node)
        if isinstance(node, exp.Boolean):
            return bool(node.this)
        if isinstance(node, exp.Literal) and not node.is_string:
            return int(node.this)
        if is_grouping_call(node):
            mask = 0
            arguments = [node.this] if node.args.get("this") is not None else []
            for argument in arguments + list(node.expressions):
                if not is_key(argument):
                    raise _Decline
                mask = mask * 2 + (_key(argument) not in present)
            return mask
        if isinstance(node, exp.EQ):
            left, right = flag(node.left, present), flag(node.right, present)
            if isinstance(left, bool) != isinstance(right, bool):
                raise _Decline
            return left == right
        raise _Decline

    outer_keys = [grouped(k) for k in select.args["group"].expressions] if select.args.get("group") else []
    if select.args.get("group") is not None and any(select.args["group"].args.get(k) for k in ("cube", "rollup", "totals", "all", "grouping_sets")):
        return None
    if any(not is_key(k) for k in outer_keys):
        return None
    outer_set = frozenset(_key(k) for k in outer_keys)
    if any(not outer_set <= one for one in sets):
        return None

    def guarded(value: exp.Expression, filters: list[exp.Expression]) -> exp.Expression:
        if not filters:
            return value.copy()
        condition = filters[0].copy()
        for more in filters[1:]:
            condition = exp.And(this=condition, expression=more.copy())
        return exp.Case(ifs=[exp.If(this=condition, true=value.copy())])

    def read(agg: exp.Expression, extra: exp.Expression | None) -> exp.Expression:
        """The aggregate over the source's rows that one outer aggregate computes."""

        if not isinstance(agg, _OUTER_AGGS) or not _plain(agg):
            raise _Decline
        argument, condition = agg.this, extra
        if isinstance(argument, exp.Case):
            ifs = argument.args.get("ifs") or []
            default = argument.args.get("default")
            if argument.args.get("this") is not None or len(ifs) != 1 or (default is not None and not isinstance(default, exp.Null)) or condition is not None:
                raise _Decline
            condition, argument = ifs[0].this, ifs[0].args["true"]
        value = _unparen(grouped(argument))
        flags, true_keys, present_keys = [], [], []
        for atom in _atoms(grouped(condition)) if condition is not None else []:
            if is_key(atom):
                true_keys.append(atom)
            elif isinstance(atom, exp.Is) and is_key(_unparen(atom.this)) and isinstance(atom.expression, exp.Boolean) and atom.expression.this:
                true_keys.append(_unparen(atom.this))
            elif isinstance(atom, exp.Not) and isinstance(_unparen(atom.this), exp.Is) and is_key(_unparen(_unparen(atom.this).this)) and isinstance(_unparen(atom.this).expression, exp.Null):
                present_keys.append(_unparen(_unparen(atom.this).this))
            else:
                flags.append(atom)
        selected = [one for one in sets if all(flag(atom, one) is True for atom in flags)]
        if len(selected) != 1:
            raise _Decline
        chosen = selected[0]
        filter_set = frozenset(_key(k) for k in true_keys)

        def distinct_arguments(arguments: list[exp.Expression]) -> list[exp.Expression]:
            unique, seen = [], set()
            for argument in arguments:
                if _key(argument) not in seen:
                    seen.add(_key(argument))
                    unique.append(argument)
            if not unique or chosen != outer_set | seen | filter_set:
                raise _Decline  # the set's rows are distinct on other keys than the ones read
            return [guarded(argument, true_keys) for argument in unique]

        if isinstance(agg, exp.Count):
            if is_key(value):
                return exp.Count(this=exp.Distinct(expressions=distinct_arguments([value] + present_keys)))
            if isinstance(value, exp.Literal) and not value.is_string:
                return exp.Count(this=exp.Distinct(expressions=distinct_arguments(present_keys)))
            raise _Decline
        if present_keys:
            raise _Decline
        if is_key(value):
            (argument,) = distinct_arguments([value])
            return exp.Sum(this=exp.Distinct(expressions=[argument])) if isinstance(agg, exp.Sum) else type(agg)(this=argument)
        if true_keys or not isinstance(value, (exp.Count, exp.Sum, exp.Min, exp.Max, exp.Avg)):
            raise _Decline
        if chosen == outer_set:
            return value.copy()  # one row of this set in each outer group
        if type(value) is type(agg) and not isinstance(value, exp.Count) and _plain(value):
            return value.copy()  # SUM of SUM, MIN of MIN, MAX of MAX over the groups
        raise _Decline

    rewritten = 0

    def collapse_item(node: exp.Expression) -> exp.Expression:
        nonlocal rewritten
        if isinstance(node, exp.Filter):
            predicate = node.expression.this if isinstance(node.expression, exp.Where) else node.expression
            rewritten += 1
            return read(node.this, predicate)
        if isinstance(node, exp.AggFunc):
            rewritten += 1
            return read(node, None)
        if isinstance(node, exp.Column):
            value = grouped(node)
            if not is_key(value) or _key(value) not in outer_set:
                raise _Decline
            return value
        if isinstance(node, (exp.Select, exp.Subquery, exp.Window, exp.Anonymous)) or is_grouping_call(node):
            raise _Decline  # a function sqlglot does not know may be an aggregate
        copy = node.copy()
        for mine, theirs in zip(list(copy.iter_expressions()), list(node.iter_expressions())):
            mine.replace(collapse_item(theirs))
        return copy

    outputs = []
    for item in select.expressions:
        value = collapse_item(item.this if isinstance(item, exp.Alias) else item)
        name = item.alias_or_name
        if not name:
            return None
        outputs.append(exp.alias_(value, name))
    if not rewritten:
        return None
    result = inner.copy()
    result.set("expressions", outputs)
    result.set("group", exp.Group(expressions=[k.copy() for k in outer_keys]) if outer_keys else None)
    return result
