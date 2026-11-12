"""A pre-counted source under a select that weighs every sum by the count reads as its rows.

    SELECT o.z, SUM(g.n * o.y) FROM (SELECT k, COUNT(*) AS n FROM s GROUP BY k) AS g JOIN o ON g.k = o.k GROUP BY o.z
    SELECT o.z, SUM(      o.y) FROM (SELECT k                FROM s           ) AS g JOIN o ON g.k = o.k GROUP BY o.z

Calcite's ``AggregateJoinTransposeRule`` pushes ``COUNT(*)`` below a join this way: each side is grouped by its
join keys and counted, and the count above the join becomes ``COALESCE(SUM(a.n * b.n), 0)``, one level per
join. The normalizer undoes a single level (``eager_aggregation``, ``sum_of_grouped_counts``); this pass undoes
a chain of levels from the inside, before the other rules rewrite a level into a form the next one cannot read.

Why the two selects above return the same rows. Call the grouped source ``g`` and the same source without its
``GROUP BY`` and its count ``g'``. A row of ``g`` with count ``n`` stands for ``n`` rows of ``g'`` with the same
key values, and ``n >= 1``. The outer select reads ``g`` only through key columns, except for ``g.n`` as a
factor of the product inside each of its sums:

* ``WHERE``, ``ON``, ``GROUP BY`` and subqueries see only key columns, so the outer groups are the same groups,
  with each joined row built on a row of ``g`` repeated ``n`` times;
* ``SUM(g.n * e)`` adds ``n * e`` for that joined row, and ``SUM(e)`` over its ``n`` repeats adds the same
  (``e`` reads nothing that differs between the repeats; a NULL ``e`` is skipped either way);
* with no other factor, ``SUM(g.n)`` over a group is the number of repeated rows in it, ``COUNT(*)``. A group
  of a ``GROUP BY`` is never empty; without ``GROUP BY`` the sum over no rows is NULL where ``COUNT(*)`` is 0,
  so there only ``COALESCE(SUM(g.n), 0)`` is read (as ``COUNT(*)``), and a bare ``SUM(g.n)`` leaves the source
  as written.

One thing does change: the type. ``SUM`` of a whole-number product is a wider type than ``COUNT(*)`` or the
``SUM`` of one factor (in PostgreSQL ``NUMERIC`` against ``BIGINT``), with the same value. The value only reads
differently where the type picks the operation, which is division (``7 / 2`` is 3 for ``BIGINT`` and 3.5 for
``NUMERIC``), so a query that divides or takes a remainder anywhere is left as written.

So the pass needs, and checks:

* every aggregate of the outer select's own is a plain ``SUM`` whose argument is a product with ``g.n`` as
  exactly one factor (any other aggregate, ``COUNT``, ``AVG``, a ``SUM`` without the factor, would see the
  repeats), and ``g.n`` is read nowhere else, subqueries included;
* no ``DISTINCT``, window, ``LIMIT``, ``OFFSET``, ``QUALIFY``, ``ROLLUP`` or grouping sets on the outer select,
  no ``RIGHT`` or ``FULL`` join in it, no lateral source, no star, no function name the parser does not know;
  the grouped source is its ``FROM`` item or the right side of a plain, ``INNER`` or ``CROSS`` join;
* the grouped source is one ``SELECT .. GROUP BY`` of plain keys with no ``HAVING``, ``DISTINCT``, window,
  ``ORDER BY`` or ``LIMIT``; each output is a grouping key (same text) or the one ``COUNT(*) AS n``;
* every other column of the outer select that reads the source is qualified by its alias and names a key
  output, and no bare column of the outer select has the name of one of the source's outputs.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import extended_grouping, select_sources
from .extreme_group_sources import _INNER_ALLOWED, _JOIN_KINDS, _OUTER_BANNED, _binds_to, _inside_source, _own, _unwrap

_COUNT_FLAGS = ("this", "big_int")


def _counted_outputs(inner: exp.Select) -> tuple[str, set[str]] | None:
    """``(count output, every output name)`` of a grouped source this pass reads, else None."""

    if any(value for key, value in inner.args.items() if key not in _INNER_ALLOWED and value):
        return None
    group = inner.args.get("group")
    if group is None or extended_grouping(group) or not group.expressions:
        return None
    if any(isinstance(n, exp.Window) or (isinstance(n, exp.Star) and not isinstance(n.parent, exp.Count)) for n in inner.walk()):
        return None
    if any(isinstance(n, (exp.AggFunc, exp.Subquery, exp.Literal)) for k in group.expressions for n in [_unwrap(k)]):
        return None  # GROUP BY 1 is an ordinal, not a key
    keys = {_unwrap(k).sql() for k in group.expressions}
    count = None
    names: set[str] = set()
    for item in inner.expressions:
        name = (item.alias_or_name or "").lower()
        value = _unwrap(item.this if isinstance(item, exp.Alias) else item)
        if not name or name in names:
            return None
        names.add(name)
        if value.sql() in keys:
            continue
        if count is not None or not isinstance(item, exp.Alias) or type(value) is not exp.Count or not isinstance(value.this, exp.Star):
            return None
        if any(v for k, v in value.args.items() if k not in _COUNT_FLAGS and v) or isinstance(value.parent, (exp.Window, exp.Filter)):
            return None
        count = name
    return None if count is None else (count, names)


def _factors(node: exp.Expression) -> list[exp.Expression]:
    """The factors of a product, through parentheses."""

    node = _unwrap(node)
    if isinstance(node, exp.Mul):
        return _factors(node.this) + _factors(node.expression)
    return [node]


def _plain_sum(call: exp.Expression) -> bool:
    if type(call) is not exp.Sum or isinstance(call.parent, (exp.Window, exp.Filter)):
        return False
    if any(value for key, value in call.args.items() if key != "this" and value):
        return False
    return isinstance(call.this, exp.Expression) and not isinstance(call.this, (exp.Distinct, exp.Star))


def _zero_default(node: exp.Expression) -> exp.Expression | None:
    """The ``COALESCE(node, 0)`` directly around ``node``, if that is what its parent is."""

    parent = node.parent
    while isinstance(parent, exp.Paren):
        node, parent = parent, parent.parent
    if not isinstance(parent, exp.Coalesce) or parent.this is not node or len(parent.expressions) != 1:
        return None
    default = _unwrap(parent.expressions[0])
    return parent if isinstance(default, exp.Literal) and not default.is_string and default.name == "0" else None


def _ungroup(select: exp.Select) -> bool:
    if any(select.args.get(key) for key in _OUTER_BANNED):
        return False
    if not any(isinstance(s, exp.Subquery) and isinstance(s.this, exp.Select) and s.this.args.get("group") for s in select_sources(select)):
        return False  # no grouped source: nothing to read (checked first, the rest walks the select)
    group = select.args.get("group")
    if group is not None and extended_grouping(group):
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
        return False
    source_ids = {id(s) for s in sources}
    aliases = [(s.alias_or_name or "").lower() for s in sources]
    if any(isinstance(n, exp.Star) and not isinstance(n.parent, exp.Count) and _own(n, select) for n in select.find_all(exp.Star)):
        return False
    if any(not _inside_source(n, select, source_ids) for n in select.find_all(exp.Anonymous, exp.WithinGroup)):
        return False
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
        read = _counted_outputs(inner)
        if read is None:
            continue
        count, names = read
        own_aggregates = [n for n in select.find_all(exp.AggFunc) if _own(n, select)]
        if not own_aggregates or not all(_plain_sum(call) for call in own_aggregates):
            continue
        weights: dict[int, exp.Column] = {}  # the count factor of each own SUM
        ok = True
        for call in own_aggregates:
            found = [f for f in _factors(call.this) if isinstance(f, exp.Column) and f.table.lower() == alias and f.name.lower() == count]
            if len(found) != 1:
                ok = False
                break
            weights[id(found[0])] = found[0]
            if len(_factors(call.this)) == 1 and group is None and _zero_default(call) is None:
                ok = False  # SUM(g.n) over no rows is NULL, COUNT(*) is 0
                break
        if not ok:
            continue
        for column in select.find_all(exp.Column):
            if isinstance(column.this, exp.Star) or _inside_source(column, select, source_ids):
                continue
            name, table = column.name.lower(), column.table.lower()
            if not table:
                if name in names:
                    ok = False  # a bare name that may be this source's output
                    break
                continue
            if table != alias or not _binds_to(column, select, alias):
                continue
            if name not in names or (name == count and id(column) not in weights):
                ok = False  # not an output, or the count read outside a sum's product
                break
        if not ok:
            continue
        for call in own_aggregates:
            factors = _factors(call.this)
            rest = [f for f in factors if id(f) not in weights]
            if rest:
                product = rest[0].copy()
                for factor in rest[1:]:
                    product = exp.Mul(this=product, expression=factor.copy())
                call.set("this", product)
            else:
                (_zero_default(call) or call).replace(exp.Count(this=exp.Star()))
        kept = [item for item in inner.expressions if (item.alias_or_name or "").lower() != count]
        inner.set("expressions", kept or [exp.alias_(exp.Literal.number(1), count)])
        inner.set("group", None)
        changed = True
    return changed


def read_counted_group_sources(tree: exp.Expression) -> exp.Expression:
    """Ungroup, from the inside, every pre-counted source whose count only weighs the sums of the select above it."""

    if not any(isinstance(n.this, exp.Star) for n in tree.find_all(exp.Count)):
        return tree
    if any(True for _ in tree.find_all(exp.Div, exp.IntDiv, exp.Mod)):
        return tree  # SUM of a whole number and COUNT(*) differ in type: a division above them would not be the same one
    for select in list(tree.find_all(exp.Select))[::-1]:  # the deepest selects first
        _ungroup(select)
    return tree
