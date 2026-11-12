"""Inside a membership test, arithmetic on an aggregate moves out of a grouped derived table to where it is read.

    WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, 0.5 * SUM(l.q) AS v FROM l GROUP BY l.pk) AS d
                  WHERE ps.pk = d.pk AND ps.n > d.v)

and

    WHERE s.k IN (SELECT ps.sk FROM ps, (SELECT l.pk, SUM(l.q) AS q FROM l GROUP BY l.pk) AS d
                  WHERE ps.pk = d.pk AND ps.n > 0.5 * d.q)

test the same thing: the first is what decorrelating ``ps.n > (SELECT 0.5 * SUM(..) ..)`` leaves, the second is how
a rewriter writes the join by hand (TPC-H Q20). The prover matches a grouped derived table inside an ``IN`` or
``EXISTS`` by what it computes, so the two tables (one returning half the sum, one the sum) never met.
``lift_computed_outputs`` writes the first as the second: the derived table returns the bare aggregate and each
read of ``d.v`` becomes ``(0.5 * d.<aggregate>)``.

Why this is exact. A derived table's output is a function of its row; replacing the output ``f(agg1, ..)`` by the
outputs ``agg1, ..`` and every read ``d.v`` by ``f(d.agg1, ..)`` evaluates the same function on the same values, as
long as ``f`` is deterministic and each read sees a real row of the derived table. An outer join can pad the row
with NULLs, and then ``d.v`` is NULL while ``f`` of NULLs need not be; so the select must join with commas and
inner joins only. ``f`` is limited to ``+``, ``-``, ``*``, unary minus and parentheses over number literals and
aggregates, and ``/`` by a non-zero number literal: no function call whose result type could depend on where it is
evaluated and nothing that can raise on one side only.

Conditions, all checked:

* the select is the whole query of an ``IN`` or ``EXISTS`` (``distinct_rules._membership_query``); elsewhere the
  prover already reads both spellings, and the rule should not respell what it reads;
* every join of the select is a comma, ``CROSS JOIN`` or inner ``JOIN .. ON`` (no ``USING``, no ``NATURAL``);
* the derived table has an alias without a column list, used by one source of the select and by no source of a
  nested query, and is a plain grouped select (select list, sources, joins, ``WHERE``, ``GROUP BY`` without
  ``ROLLUP`` and the like; no ``HAVING``, ``DISTINCT``, window or tail) with uniquely named outputs and no star;
* the output is arithmetic as above with at least one aggregate, and its aggregates are plain calls of the derived
  table itself (not windowed, not filtered, holding no subquery);
* the select has no star, and every column of it named like the output is qualified with the derived table's alias
  (a bare one could be a different column).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import extended_grouping, select_sources
from .distinct_rules import _membership_query
from .unmatched_outer_join import _inside

_DERIVED_ARGS = frozenset({"expressions", "from_", "from", "joins", "where", "group"})
_ARITHMETIC = (exp.Mul, exp.Add, exp.Sub)


def _number(node: exp.Expression) -> bool:
    return type(node) is exp.Literal and not node.is_string


def _nonzero(node: exp.Expression) -> bool:
    if not _number(node):
        return False
    try:
        return float(node.this) != 0.0
    except ValueError:
        return False


def _aggregates(node: exp.Expression, inner: exp.Select, found: list[exp.Expression]) -> bool:
    """Whether ``node`` is arithmetic over number literals and plain aggregates of ``inner``; collects the aggregates."""

    if type(node) is exp.Paren or type(node) is exp.Neg:
        return _aggregates(node.this, inner, found)
    if _number(node):
        return True
    if type(node) in _ARITHMETIC:
        return _aggregates(node.this, inner, found) and _aggregates(node.expression, inner, found)
    if type(node) is exp.Div:
        if any(v for key, v in node.args.items() if key not in ("this", "expression", "typed", "safe")) or node.args.get("safe"):
            return False
        return _nonzero(node.expression) and _aggregates(node.this, inner, found)
    if isinstance(node, exp.AggFunc):
        if isinstance(node.parent, (exp.Window, exp.Filter, exp.WithinGroup)) or node.find_ancestor(exp.Select) is not inner:
            return False
        if any(isinstance(n, (exp.Query, exp.Subquery, exp.Window)) or (isinstance(n, exp.AggFunc) and n is not node) for n in node.walk()):
            return False
        found.append(node)
        return True
    return False


def _plain_joins(select: exp.Select) -> bool:
    for join in select.args.get("joins") or []:
        if join.args.get("side") or join.args.get("using") or join.args.get("method"):
            return False
        if (join.args.get("kind") or "").upper() not in ("", "INNER", "CROSS"):
            return False
        if any(v for key, v in join.args.items() if key not in ("this", "kind", "on")):
            return False
    return True


def lift_computed_outputs(select: exp.Select) -> exp.Expression | None:
    """``select`` with one computed aggregate output of a grouped derived table moved to its reads, or ``None``."""

    if not _membership_query(select) or not _plain_joins(select) or select.args.get("laterals"):
        return None
    if any(isinstance(n, exp.Star) for item in select.expressions for n in item.walk() if not isinstance(n.parent, exp.Count)):
        return None
    sources = select_sources(select)
    names = [(s.alias_or_name or "").lower() for s in sources]
    for index, source in enumerate(sources):
        if type(source) is not exp.Subquery or not source.alias or type(source.this) is not exp.Select:
            continue
        if source.args["alias"].args.get("columns") or any(v for key, v in source.args.items() if key not in ("this", "alias")):
            continue
        alias = source.alias.lower()
        inner = source.this
        if names.count(alias) != 1 or any(value for key, value in inner.args.items() if key not in _DERIVED_ARGS):
            continue
        group = inner.args.get("group")
        if group is None or extended_grouping(group) or not group.expressions:
            continue
        outputs = [(item.alias_or_name or "").lower() for item in inner.expressions]
        if not all(outputs) or len(set(outputs)) != len(outputs) or any(isinstance(item, exp.Star) or item.is_star for item in inner.expressions):
            continue
        # another source called the same in a nested query would make ``alias.v`` ambiguous to follow
        shadowed = False
        for nested in select.find_all(exp.Select):
            if nested is select or _inside(nested, inner):
                continue
            if any((s.alias_or_name or "").lower() == alias for s in select_sources(nested)):
                shadowed = True
                break
        if shadowed:
            continue
        for position, item in enumerate(inner.expressions):
            if type(item) is not exp.Alias:
                continue
            value = item.this
            while type(value) is exp.Paren:
                value = value.this
            found: list[exp.Expression] = []
            if isinstance(value, exp.AggFunc) or not _aggregates(value, inner, found) or not found:
                continue
            name = outputs[position]
            reads = []
            bare = False
            for column in select.find_all(exp.Column):
                if _inside(column, inner) or column.name.lower() != name:
                    continue
                if len(column.parts) != 2 or column.table.lower() != alias:
                    bare = True
                    break
                reads.append(column)
            if bare:
                continue
            copy = select.copy()
            new_source = select_sources(copy)[index]
            new_inner = new_source.this
            new_item = new_inner.expressions[position]
            new_value = new_item.this
            new_found: list[exp.Expression] = []
            _aggregates(new_value, new_inner, new_found)
            taken = set(outputs)
            qualifier = new_source.args["alias"].this
            added = []
            texts: dict[str, str] = {}
            for aggregate in new_found:
                text = aggregate.sql()
                fresh = texts.get(text)
                if fresh is None:
                    number = 0
                    while f"kql{number}" in taken:
                        number += 1
                    fresh = f"kql{number}"
                    taken.add(fresh)
                    texts[text] = fresh
                    added.append(exp.alias_(aggregate.copy(), fresh))
                aggregate.replace(exp.Column(this=exp.to_identifier(fresh), table=qualifier.copy()))
            formula = exp.Paren(this=new_value.copy())
            new_reads = [
                column for column in copy.find_all(exp.Column)
                if not _inside(column, new_inner) and column.name.lower() == name and column.table.lower() == alias
            ]
            if len(new_reads) != len(reads):
                continue
            for column in new_reads:
                column.replace(formula.copy())
            new_item.pop()
            for extra in added:
                new_inner.append("expressions", extra)
            return copy
    return None
