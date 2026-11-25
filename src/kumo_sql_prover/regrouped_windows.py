"""An aggregate over the groups of a partition, in a grouped derived table, read as a join with the table grouped again.

"Items that sell less than a tenth of their shop's mean" needs each (shop, item) total beside the mean of the
shop's totals. One spelling asks for the mean with a window over the groups:

    SELECT s.name, t.item, t.total
    FROM shops AS s,
         (SELECT shop, item, SUM(price) AS total, AVG(SUM(price)) OVER (PARTITION BY shop) AS mean
          FROM orders GROUP BY shop, item) AS t
    WHERE s.id = t.shop AND t.total <= 0.1 * t.mean

The other groups the grouped table again and joins the two:

    SELECT s.name, t.item, t.total
    FROM shops AS s,
         (SELECT shop, item, SUM(price) AS total FROM orders GROUP BY shop, item) AS t,
         (SELECT x.shop, AVG(x.total) AS mean FROM (SELECT shop, SUM(price) AS total FROM orders GROUP BY shop, item) AS x
          GROUP BY x.shop) AS m
    WHERE s.id = t.shop AND t.shop = m.shop AND t.total <= 0.1 * m.mean

A window is kept as an opaque relation, so the two stayed apart. This pre-pass (hardest-cases batch 28) writes the
first spelling as the second, which the prover reads.

**The rule.** ``t`` is a derived table of a select ``S`` whose joins are all inner joins (or commas). Its body has
``GROUP BY`` and one or more select-list items ``AGG(e) OVER (PARTITION BY p1, .., pn) AS w``, where every ``p`` is
a grouping term the body also lists as an item of its own, and ``e`` is any expression over the groups. The window
items are taken out of ``t``; ``S`` gets one more source

    (SELECT x.k1, .., AGG(x.a) AS v FROM (SELECT p1 AS k1, .., e AS a FROM <the body's FROM, WHERE, GROUP BY, HAVING>) AS x
     GROUP BY x.k1, ..) AS m

with ``t.p1 = m.k1 AND ..`` added to its ``WHERE``, and every ``t.w`` in ``S`` is written ``m.v``.

**Why it is sound.**

* A window of a grouped select is evaluated over its groups, after ``HAVING``. Without ``ORDER BY`` or a frame it
  aggregates ``e`` over every group that has the row's partition values. ``x`` holds the same groups (the same
  ``FROM``, ``WHERE``, ``GROUP BY`` and ``HAVING``), and ``m`` aggregates ``e`` over the ones that share the values.
* A row of ``t`` whose partition values are not NULL equals exactly one row of ``m`` on them: the one made of its
  own partition. The join keeps it once, with the window's value.
* A row of ``t`` with a NULL partition value matches no row of ``m``. ``S`` must therefore already reject it: for
  every partition column there is a condition of ``S`` (in ``WHERE`` or in an inner join's ``ON``) that is an
  equality with ``t.p`` on one side, which is not true of NULL.
* ``MIN``, ``MAX``, ``SUM``, ``AVG`` and ``COUNT`` over a set of rows give the same value (and type) whether they
  are written as a window or as a grouped aggregate.

**What is refused.** Anything else: an outer join, ``USING`` or ``NATURAL`` in ``S``; a ``*`` in ``S`` (the columns
of ``t`` change); a bare column in ``S`` spelled like a window's name; a second source named like ``t``; a body with
``DISTINCT``, ``ORDER BY``, ``LIMIT``, ``QUALIFY``, ``ROLLUP`` or a set operation; a window that is not a whole select
item, has ``ORDER BY``, a frame, a name, ``FILTER``, ``DISTINCT`` or no ``PARTITION BY``; windows over different
partitions; a partition term that is not a grouping term listed under a name; a partition column ``S`` does not
compare with ``=``.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _names
from .ast_utils import FROM_KEY, extended_grouping

NAME = "regrouped_windows"

_PREFIX = "kqr"
_AGGREGATES = (exp.Min, exp.Max, exp.Sum, exp.Avg, exp.Count)
_BODY = {"expressions", FROM_KEY, "joins", "where", "group", "having"}
_INNER = ("", "INNER", "CROSS")


def read_regrouped_windows(tree: exp.Expression) -> tuple[exp.Expression, bool]:
    """``(tree with each derived table of the module docstring rewritten, whether one was)``. The input is not changed."""

    if not any(True for _ in tree.find_all(exp.Window)):
        return tree, False
    copy = tree.copy()
    taken = _names(copy)
    wrote = False
    for select in list(copy.find_all(exp.Select)):
        while _rewrite(select, taken):
            wrote = True
    return (copy, True) if wrote else (tree, False)


def _fresh(taken: set[str]) -> str:
    number = 0
    while f"{_PREFIX}{number}" in taken:
        number += 1
    taken.add(f"{_PREFIX}{number}")
    return f"{_PREFIX}{number}"


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _conjuncts(node: exp.Expression | None) -> list[exp.Expression]:
    if node is None:
        return []
    node = _unwrap(node)
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _set_args(node: exp.Expression) -> set[str]:
    return {key for key, value in node.args.items() if value not in (None, [], False)}


def _inside(node: exp.Expression, ancestor: exp.Expression) -> bool:
    while node is not None:
        if node is ancestor:
            return True
        node = node.parent
    return False


class _Body:
    """The window items of a grouped body: ``(item, aggregate, argument or None)``, and the partition's output names."""

    def __init__(self, windows: list[tuple[exp.Expression, type, exp.Expression | None]], partition: list[tuple[str, exp.Expression]]) -> None:
        self.windows, self.partition = windows, partition


def _body(body: exp.Expression) -> _Body | None:
    if not isinstance(body, exp.Select) or _set_args(body) - _BODY or body.args.get(FROM_KEY) is None:
        return None
    group = body.args.get("group")
    if group is None or not group.expressions or extended_grouping(group) or _set_args(group) - {"expressions"}:
        return None
    own = [node for node in body.find_all(exp.Window) if node.find_ancestor(exp.Select) is body]
    if not own:
        return None
    listed: dict[str, exp.Expression] = {}  # output name -> the grouping term the item is
    names: list[str] = []
    for item in body.expressions:
        name = item.alias_or_name.lower()
        if not name or isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
            return None
        names.append(name)
        value = _unwrap(item.unalias())
        if any(value == _unwrap(term) for term in group.expressions):
            listed[name] = value
    if len(set(names)) != len(names):
        return None
    windows: list[tuple[exp.Expression, type, exp.Expression | None]] = []
    partitions: set[tuple[str, ...]] = set()
    for item in body.expressions:
        node = item.unalias()
        if not any(_inside(window, item) for window in own):
            continue
        if not isinstance(item, exp.Alias) or type(node) is not exp.Window:
            return None  # the window is a part of an item, or has no name
        if _set_args(node) - {"this", "partition_by", "over"} or (node.text("over") or "OVER").upper() != "OVER":
            return None  # ORDER BY, a frame, a named window
        aggregate = node.this
        if type(aggregate) not in _AGGREGATES or _set_args(aggregate) - {"this", "big_int"}:
            return None
        argument = None if type(aggregate) is exp.Count and isinstance(aggregate.this, exp.Star) else aggregate.this
        if argument is not None and any(isinstance(inner, (exp.Window, exp.Distinct, exp.Star)) and not isinstance(inner.parent, exp.Count) for inner in argument.walk()):
            return None
        if argument is not None and isinstance(argument, exp.Distinct):
            return None
        found: list[str] = []
        for term in node.args.get("partition_by") or []:
            name = next((name for name, value in listed.items() if value == _unwrap(term)), None)
            if name is None:
                return None
            found.append(name)
        if not found:
            return None
        partitions.add(tuple(sorted(set(found))))
        windows.append((item, type(aggregate), argument))
    if len(partitions) != 1 or len(windows) != len(own) or len(windows) == len(body.expressions):
        return None
    return _Body(windows, [(name, listed[name]) for name in next(iter(partitions))])


def _rewrite(select: exp.Select, taken: set[str]) -> bool:
    source = select.args.get(FROM_KEY)
    joins = list(select.args.get("joins") or [])
    if source is None or select.args.get("laterals") or select.args.get("pivots"):
        return False
    if any(_set_args(join) - {"this", "on", "kind"} or (join.kind or "").upper() not in _INNER for join in joins):
        return False
    sources = [source.this] + [join.this for join in joins]
    where = select.args.get("where")
    conditions = _conjuncts(where.this if where is not None else None)
    for join in joins:
        conditions += _conjuncts(join.args.get("on"))
    for derived in sources:
        if type(derived) is not exp.Subquery or not derived.alias or derived.args["alias"].args.get("columns") or _set_args(derived) - {"this", "alias"}:
            continue
        alias = derived.alias.lower()
        read = _body(derived.this)
        if read is None:
            continue
        outside = lambda node: not _inside(node, derived)  # noqa: E731
        if sum(1 for node in select.find_all((exp.Table, exp.Subquery)) if outside(node) or node is derived if (node.alias_or_name or "").lower() == alias) != 1:
            continue  # another source of the name, here or in a subquery
        if any(outside(node) and not isinstance(node.parent, exp.Count) for node in select.find_all(exp.Star)):
            continue
        named = {item.alias.lower() for item, _, _ in read.windows}
        columns = [node for node in select.find_all(exp.Column) if outside(node)]
        if any(not node.table and node.name.lower() in named for node in columns):
            continue  # a bare name that may be the window's
        rejected = set()
        for test in conditions:
            if type(test) is not exp.EQ:
                continue
            for side in (_unwrap(test.this), _unwrap(test.expression)):
                if type(side) is exp.Column and (side.table or "").lower() == alias and not side.args.get("db"):
                    rejected.add(side.name.lower())
        if not {name for name, _ in read.partition} <= rejected:
            continue  # a NULL partition value would leave the join and stay in the window
        _write(select, derived, read, taken)
        return True
    return False


def _write(select: exp.Select, derived: exp.Subquery, read: _Body, taken: set[str]) -> None:
    alias, body = derived.alias, derived.this
    fresh = _fresh(taken)
    inner_name = f"{fresh}x"
    inner = body.copy()
    inner_items = [exp.alias_(term.copy(), f"{fresh}k{index}") for index, (_, term) in enumerate(read.partition)]
    outer_items = [exp.alias_(exp.column(f"{fresh}k{index}", table=inner_name), f"{fresh}k{index}") for index in range(len(read.partition))]
    values: dict[str, str] = {}  # the window's name in t -> its name in the new source
    for index, (item, aggregate, argument) in enumerate(read.windows):
        if argument is None:
            call = aggregate(this=exp.Star())
        else:
            inner_items.append(exp.alias_(argument.copy(), f"{fresh}a{index}"))
            call = aggregate(this=exp.column(f"{fresh}a{index}", table=inner_name))
        outer_items.append(exp.alias_(call, f"{fresh}v{index}"))
        values[item.alias.lower()] = f"{fresh}v{index}"
    inner.set("expressions", inner_items)
    regrouped = exp.Select(expressions=outer_items)
    regrouped.set(FROM_KEY, exp.From(this=exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier(inner_name)))))
    regrouped.set("group", exp.Group(expressions=[exp.column(f"{fresh}k{index}", table=inner_name) for index in range(len(read.partition))]))
    for node in list(select.find_all(exp.Column)):
        if not _inside(node, derived) and (node.table or "").lower() == alias.lower() and node.name.lower() in values:
            value = exp.column(values[node.name.lower()], table=fresh)
            listed = isinstance(node.parent, exp.Select) and node.arg_key == "expressions"
            node.replace(exp.alias_(value, node.this.copy()) if listed else value)  # a select item keeps its name
    body.set("expressions", [item for item in body.expressions if not any(item is window for window, _, _ in read.windows)])
    condition = select.args["where"].this if select.args.get("where") is not None else None
    if condition is not None and isinstance(_unwrap(condition), exp.Or):
        condition = exp.Paren(this=condition)
    for index, (name, _) in enumerate(read.partition):
        test = exp.EQ(this=exp.column(name, table=alias), expression=exp.column(f"{fresh}k{index}", table=fresh))
        condition = test if condition is None else exp.And(this=condition, expression=test)
    select.set("where", exp.Where(this=condition))
    select.append("joins", exp.Join(this=exp.Subquery(this=regrouped, alias=exp.TableAlias(this=exp.to_identifier(fresh)))))
