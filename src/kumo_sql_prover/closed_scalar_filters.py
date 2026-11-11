"""A filter on a grouped derived table's outputs is ``HAVING`` even when it compares with a closed scalar subquery.

    SELECT v.k, v.total
    FROM (SELECT ps.k, SUM(ps.q) AS total FROM ps GROUP BY ps.k) AS v
    WHERE v.total > (SELECT SUM(ps.q) * 2 FROM ps)

is

    SELECT ps.k, SUM(ps.q) AS total FROM ps GROUP BY ps.k HAVING SUM(ps.q) > (SELECT SUM(ps.q) * 2 FROM ps)

(TPC-H Q11). ``algebraic_equivalence._fold_filter_into_grouping`` moves such a filter into ``HAVING`` but declines
any subquery in the select, because moving a subquery from the outer select into the grouped one changes what
its outer references can see. A *closed* scalar subquery has no outer reference, so it is the same value in
either place; ``fold_scalar_filter`` sets the closed ones aside, lets that rule do its work, and puts them back.

A scalar subquery is closed when every column in it resolves to one of its own sources, at every level:

* its sources are base tables the schema lists or derived tables with an alias, named outputs and no star, under
  distinct names, joined without ``LATERAL``;
* each column is qualified with one of those names and is a column of that source, or is bare and exactly one of
  the sources has it (so the name cannot reach an enclosing query, before or after the move);
* each derived table is closed in the same way, and there is no other nested query (no subquery in a select
  list or a condition, no ``EXISTS``, no set operation).

Conditions on the rest are those of ``_fold_filter_into_grouping``; the scalar subqueries must sit in the
select's ``WHERE`` as values (not the query of an ``IN``).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import select_sources
from .keyed_join_groups import declared_columns
from .unmatched_outer_join import _columns_of, _inside, _plain_column, _plain_table


def closed(select: exp.Expression, declared: dict[str, set[str]]) -> bool:
    """Whether every column of ``select`` resolves to one of its own sources, at every level."""

    if type(select) is not exp.Select or select.args.get("laterals") or select.args.get("with_") or select.args.get("with"):
        return False
    sources = select_sources(select)
    names = [(s.alias_or_name or "").lower() for s in sources]
    if not sources or not all(names) or len(set(names)) != len(names):
        return False
    owned: list[set[str]] = []
    for source in sources:
        if type(source) is exp.Table:
            columns = _columns_of(source, declared) if _plain_table(source) else None
            if not columns:
                return False
            owned.append(columns)
        elif type(source) is exp.Subquery:
            alias = source.args.get("alias")
            if alias is None or alias.args.get("columns") or any(v for key, v in source.args.items() if key not in ("this", "alias")):
                return False
            if not closed(source.this, declared):
                return False
            outputs = set()
            for item in source.this.expressions:
                name = (item.alias_or_name or "").lower()
                if not name or isinstance(item, exp.Star) or item.is_star:
                    return False
                outputs.add(name)
            owned.append(outputs)
        else:
            return False
    for join in select.args.get("joins") or []:
        if join.args.get("using") or join.args.get("method") or isinstance(join.this, exp.Lateral):
            return False
    for node in select.walk():
        if node is select or any(_inside(node, source) for source in sources if type(source) is exp.Subquery):
            continue
        if isinstance(node, (exp.Query, exp.Subquery, exp.Exists)):
            return False
        if isinstance(node, exp.Column):
            if _plain_column(node) is not node or len(node.parts) > 2:
                return False
            if node.table:
                if node.table.lower() not in names or node.name.lower() not in owned[names.index(node.table.lower())]:
                    return False
            elif sum(node.name.lower() in columns for columns in owned) != 1:
                # an output name in ORDER BY or GROUP BY is not a column of a source; such selects are simply not read
                return False
    return True


def fold_scalar_filter(select: exp.Select, schema: dict | None) -> exp.Expression | None:
    """``select`` (a filter on a grouped derived table, comparing with closed scalar subqueries) as ``HAVING``, or ``None``."""

    where = select.args.get("where")
    if where is None or not schema or select.args.get("joins"):
        return None
    from_ = select.args.get("from_") or select.args.get("from")
    if from_ is None or type(from_.this) is not exp.Subquery:
        return None
    scalars = [
        node for node in where.find_all(exp.Subquery)
        if node.find_ancestor(exp.Select) is select and not node.alias
    ]
    if not scalars:
        return None
    declared = declared_columns(schema)
    for node in scalars:
        if node.arg_key == "query" or isinstance(node.parent, (exp.Exists, exp.Any, exp.All)) or any(v for key, v in node.args.items() if key != "this"):
            return None
        if not closed(node.this, declared):
            return None
    from .algebraic_equivalence import _fold_filter_into_grouping

    if select.find(exp.Placeholder) is not None:
        return None  # the query has placeholders of its own
    copy = select.copy()
    mapping: dict[str, exp.Expression] = {}
    position = 0
    for node in list(copy.args["where"].find_all(exp.Subquery)):
        if node.find_ancestor(exp.Select) is copy and not node.alias:
            name = f"kqs{position}"
            position += 1
            mapping[name] = node.copy()
            node.replace(exp.Placeholder(this=name))
    folded = _fold_filter_into_grouping(copy)
    if folded is None:
        return None
    seen: list[str] = []
    for node in list(folded.find_all(exp.Placeholder)):
        if node.name in mapping:
            seen.append(node.name)
            node.replace(mapping[node.name].copy())
    if sorted(seen) != sorted(mapping):
        return None
    return folded


_GLOBAL_ARGS = frozenset({"expressions", "from_", "from", "joins", "where"})


def lift_scalar_arithmetic(select: exp.Select) -> exp.Expression | None:
    """``(SELECT SUM(x) * 2 FROM t)`` read in ``select`` as ``((SELECT SUM(x) FROM t) * 2)``, or ``None``.

    A select of one aggregate expression with no ``GROUP BY`` returns exactly one row, so the scalar subquery is
    that expression's value, and arithmetic on the aggregate with number literals (``+ - *``, ``/`` by a non-zero
    literal) can be done on the subquery's value instead. Both spellings then read the same subquery. Only an
    expression with a single aggregate is moved; the subquery has only a select list, sources, joins and
    ``WHERE``, and sits where a value is read (not the query of an ``IN``, ``EXISTS``, ``ANY`` or ``ALL``).
    """

    from .computed_group_outputs import _aggregates

    for node in select.find_all(exp.Subquery):
        if node.find_ancestor(exp.Select) is not select or node.alias or any(v for key, v in node.args.items() if key != "this"):
            continue
        if node.arg_key == "query" or isinstance(node.parent, (exp.Exists, exp.Any, exp.All, exp.From, exp.Join, exp.SetOperation, exp.Subquery)):
            continue
        inner = node.this
        if type(inner) is not exp.Select or len(inner.expressions) != 1 or any(value for key, value in inner.args.items() if key not in _GLOBAL_ARGS):
            continue
        item = inner.expressions[0]
        value = item.this if isinstance(item, exp.Alias) else item
        while type(value) is exp.Paren:
            value = value.this
        found: list[exp.Expression] = []
        if isinstance(value, exp.AggFunc) or not _aggregates(value, inner, found) or len(found) != 1:
            continue
        copy = select.copy()
        target = next(n for n in copy.find_all(exp.Subquery) if n.find_ancestor(exp.Select) is copy and n.sql() == node.sql() and not n.alias and n.arg_key == node.arg_key)
        new_inner = target.this
        new_item = new_inner.expressions[0]
        new_value = new_item.this if isinstance(new_item, exp.Alias) else new_item
        formula = new_value.copy()
        holder = exp.Paren(this=formula)
        aggregate = next(n for n in holder.find_all(exp.AggFunc))
        scalar = new_inner.copy()
        scalar.set("expressions", [aggregate.copy()])
        aggregate.replace(exp.Subquery(this=scalar))
        target.replace(holder)
        return copy
    return None
