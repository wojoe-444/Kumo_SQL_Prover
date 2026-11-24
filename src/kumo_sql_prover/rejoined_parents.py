"""A grouped derived table drops its join to a keyed row that the select around it joins again.

"Orders of tools below a fifth of the item's mean quantity" is often written in stages, and the stage that
computes the means joins the item table and repeats its filter:

    SELECT SUM(o.price) FROM orders AS o, items AS i,
         (SELECT i2.id AS item, AVG(o2.qty) AS usual FROM items AS i2, orders AS o2
          WHERE o2.item = i2.id AND i2.kind = 'tool' GROUP BY i2.id) AS g
    WHERE o.item = i.id AND i.kind = 'tool' AND i.id = g.item AND o.qty < 0.2 * g.usual

The select around ``g`` only keeps a group whose key is the id of an item ``i`` that passes ``i.kind = 'tool'``. For
such a key the join and the filter inside ``g`` change nothing, so ``g`` is read as the grouped table alone:

    (SELECT o2.item AS item, AVG(o2.qty) AS usual FROM orders AS o2 GROUP BY o2.item) AS g

which is the spelling the prover reaches from a correlated subquery (``.. < (SELECT 0.2 * AVG(qty) FROM orders
WHERE item = i.id)``) and from a window (``keyed_join_windows``). Hardest-cases batch 28.

**The rule.** ``g`` is a derived table of a select ``S`` whose joins are all inner joins (or commas). Its body has
two sources, a declared table ``p`` with a declared NOT NULL key and a declared table ``t``, inner joined, and

* its conditions (``ON`` and ``WHERE``) are: one ``t.c = p.k`` per column ``k`` of the key, tests that read ``p``
  only (the filter), and tests that read ``t`` only (kept);
* it groups by the key: for each key column, ``p.k`` or the ``t.c`` it is equal to, and nothing else;
* its select list reads ``p`` through key columns only.

In ``S`` there is another read ``p'`` of the same declared table, and the conditions of ``S`` (its ``WHERE`` and the
``ON`` of its joins, which every row it keeps passes) say that each ``p'.k`` is equal to the output of ``g`` that
shows ``p.k``, directly or through other whole-number columns, and repeat every filter test of ``g`` on ``p'``.

Then ``p``, the equalities and the filter are removed from ``g``, and ``p.k`` is written as ``t.c``.

**Why it is sound.** Take a combination of rows that ``S`` keeps, and let ``v`` be the key values of its row of
``g``. Its row of ``p'`` has the key ``v`` and passes the filter. A key holds one row, so that row is the only row of
the declared table with the key ``v``, and it is the one the old ``g`` joins to the rows of ``t`` with ``t.c = v``:
each of those rows once, all of them (the filter passes). The old group for ``v`` is therefore made of exactly the
rows of ``t`` with ``t.c = v`` that pass the kept tests, which is the new group for ``v``; the two have the same
aggregates, and one exists exactly when the other does. Groups with other keys are in no combination ``S`` keeps,
before or after: a new group whose key matches no filtered row (or is NULL) fails the same conditions of ``S``.

"Equal" must mean the same to the comparisons and to ``GROUP BY``, which it does for whole numbers but not for
texts under different collations, so every column the rule reads as equal to another must be declared with a
whole-number type (as in ``keyed_join_windows``).

**What is refused.** Anything else: an outer join, ``USING`` or ``NATURAL`` in ``S`` or in ``g``; a column of ``g``
that does not carry the name of one of its two sources; ``HAVING``, ``DISTINCT``, ``ORDER BY``, ``LIMIT``, a window
or a subquery in ``g``; a test in ``g`` that reads both sources and is not one of the equalities, or a second
equality on a key column; a select-list item or grouping term of ``g`` that reads another column of ``p``; a filter
test with a subquery or a function the prover does not know; a filter test ``S`` does not repeat as it is written;
a ``WITH`` name hiding the declared table.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY
from .single_row_aggregates import Keyed, column_kind

NAME = "rejoined_parents"

_INNER = ("", "INNER", "CROSS")
_BODY = {"expressions", FROM_KEY, "joins", "where", "group"}


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


def _plain(node: exp.Expression) -> tuple[str, str] | None:
    """``(source name, column name)``, lower case, of a column written with its source's name."""

    node = _unwrap(node)
    if type(node) is not exp.Column or not isinstance(node.this, exp.Identifier) or not node.table or node.args.get("db") or node.args.get("catalog"):
        return None
    return node.table.lower(), node.name.lower()


def _inner_joins(select: exp.Select) -> list[exp.Join] | None:
    joins = list(select.args.get("joins") or [])
    if any(_set_args(join) - {"this", "on", "kind"} or (join.kind or "").upper() not in _INNER for join in joins):
        return None
    return joins


def _conditions(select: exp.Select, joins: list[exp.Join]) -> list[exp.Expression]:
    where = select.args.get("where")
    found = _conjuncts(where.this if where is not None else None)
    for join in joins:
        found += _conjuncts(join.args.get("on"))
    return found


class _Grouped:
    """What the rule reads of the body of ``g``: see ``_grouped``."""

    def __init__(self, parent: Keyed, table: Keyed, partner: dict[str, str], filters: list[exp.Expression], kept: list[exp.Expression], shown: dict[str, str]) -> None:
        self.parent, self.table, self.partner, self.filters, self.kept, self.shown = parent, table, partner, filters, kept, shown


def _grouped(body: exp.Expression, schema, keys, not_null, types) -> _Grouped | None:
    if not isinstance(body, exp.Select) or _set_args(body) - _BODY or not body.args.get("group"):
        return None
    joins = _inner_joins(body)
    source = body.args.get(FROM_KEY)
    if joins is None or len(joins) != 1 or source is None:
        return None
    if any(isinstance(node, (exp.Window, exp.Subquery, exp.Select, exp.SetOperation, exp.Star, exp.Anonymous)) and not isinstance(node.parent, exp.Count) for node in body.walk() if node is not body):
        return None
    for first, second in ((source.this, joins[0].this), (joins[0].this, source.this)):
        parent, table = Keyed.read(first, schema, keys, not_null, types), Keyed.read(second, schema, keys, not_null, types)
        if parent is None or table is None or not parent.keys or parent.alias == table.alias:
            continue
        read = _body(body, joins, parent, table)
        if read is not None:
            return read
    return None


def _body(body: exp.Select, joins: list[exp.Join], parent: Keyed, table: Keyed) -> _Grouped | None:
    names = {parent.alias: parent, table.alias: table}
    for node in body.find_all(exp.Column):
        plain = _plain(node)
        if plain is None or plain[0] not in names or plain[1] not in names[plain[0]].columns:
            return None
    partner: dict[str, str] = {}
    filters: list[exp.Expression] = []
    kept: list[exp.Expression] = []
    for test in _conditions(body, joins):
        sides = {node.table.lower() for node in test.find_all(exp.Column)}
        if sides == {parent.alias}:
            filters.append(test)
        elif sides <= {table.alias}:
            kept.append(test)
        else:
            if type(test) is not exp.EQ:
                return None
            left, right = _plain(test.this), _plain(test.expression)
            if left is None or right is None:
                return None
            if left[0] == parent.alias:
                left, right = right, left
            if left[0] != table.alias or right[0] != parent.alias or right[1] in partner:
                return None
            if column_kind(table.types.get(left[1])) != "int" or column_kind(parent.types.get(right[1])) != "int":
                return None
            partner[right[1]] = left[1]
    if not any(set(key) == set(partner) for key in parent.keys):
        return None
    group = body.args["group"]
    if _set_args(group) - {"expressions"}:
        return None
    covered: set[str] = set()
    for term in group.expressions:
        plain = _plain(term)
        if plain is None:
            return None
        if plain[0] == parent.alias and plain[1] in partner:
            covered.add(plain[1])
        elif plain[0] == table.alias and plain[1] in partner.values():
            covered.update(key for key, column in partner.items() if column == plain[1])
        else:
            return None
    if covered != set(partner):
        return None
    shown: dict[str, str] = {}  # output name of g -> key column of the parent it shows
    for item in body.expressions:
        value = _unwrap(item.unalias())
        plain = _plain(value) if isinstance(value, exp.Column) else None
        name = item.alias_or_name.lower()
        if any(node.table.lower() == parent.alias and node.name.lower() not in partner for node in item.find_all(exp.Column)):
            return None  # another column of the parent
        if plain is None or not name:
            continue
        if plain[0] == parent.alias:
            shown[name] = plain[1]
        elif plain[1] in partner.values():
            shown.update({name: key for key, column in partner.items() if column == plain[1]})
    return _Grouped(parent, table, partner, filters, kept, shown)


def _renamed(test: exp.Expression, old: str, new: str) -> exp.Expression:
    copy = test.copy()
    for node in [copy] if isinstance(copy, exp.Column) else list(copy.find_all(exp.Column)):
        if node.table.lower() == old:
            node.set("table", exp.to_identifier(new))
    return copy


def drop_rejoined_parent(select: exp.Select, schema: dict | None, keys: dict | None, not_null: dict | None, types: dict | None) -> exp.Select | None:
    """``select`` with one derived table read as the module docstring says, or None. The input is not changed."""

    if not schema or not keys or not types or select.args.get(FROM_KEY) is None or select.args.get("laterals") or select.args.get("pivots"):
        return None
    joins = _inner_joins(select)
    if not joins:
        return None
    sources = [select.args[FROM_KEY].this] + [join.this for join in joins]
    derived = [(index, source) for index, source in enumerate(sources) if type(source) is exp.Subquery and source.alias and not source.args["alias"].args.get("columns") and not _set_args(source) - {"this", "alias"}]
    if not derived:
        return None
    reads = {}
    for source in sources:
        read = Keyed.read(source, schema, keys, not_null, types) if isinstance(source, exp.Table) else None
        if read is not None:
            if read.alias in reads:
                return None
            reads[read.alias] = read
    conditions = _conditions(select, joins)
    for index, source in derived:
        alias = source.alias.lower()
        if alias in reads or sum(1 for other in sources if (other.alias_or_name or "").lower() == alias) != 1:
            continue
        grouped = _grouped(source.this, schema, keys, not_null, types)
        if grouped is None:
            continue
        # --- columns the conditions of the select say are equal, through whole-number columns only ---
        whole = lambda plain: (plain[0] == alias and plain[1] in grouped.shown) or (  # noqa: E731
            plain[0] in reads and column_kind(reads[plain[0]].types.get(plain[1])) == "int"
        )
        linked: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for test in conditions:
            if type(test) is not exp.EQ:
                continue
            left, right = _plain(test.this), _plain(test.expression)
            if left is None or right is None or not whole(left) or not whole(right):
                continue
            linked.setdefault(left, set()).add(right)
            linked.setdefault(right, set()).add(left)
        for again in reads.values():
            if again.name != grouped.parent.name:
                continue
            if not all(_reaches(linked, (again.alias, key), alias, grouped.shown, key) for key in grouped.partner):
                continue
            wanted = [_renamed(test, grouped.parent.alias, again.alias) for test in grouped.filters]
            if not all(any(test == other for other in conditions) for test in wanted):
                continue
            return _written(select, index, grouped)
    return None


def _reaches(linked: dict, start: tuple[str, str], alias: str, shown: dict[str, str], key: str) -> bool:
    """Whether the equalities tie ``start`` to an output of the derived table that shows the key column ``key``."""

    seen, todo = {start}, [start]
    while todo:
        for other in linked.get(todo.pop(), ()):
            if other[0] == alias and shown.get(other[1]) == key:
                return True
            if other not in seen:
                seen.add(other)
                todo.append(other)
    return False


def _written(select: exp.Select, index: int, grouped: _Grouped) -> exp.Select:
    copy = select.copy()
    source = ([copy.args[FROM_KEY].this] + [join.this for join in copy.args["joins"]])[index]
    body = source.this
    parent, table = grouped.parent.alias, grouped.table.alias
    kept_table = next(item for item in [body.args[FROM_KEY].this, body.args["joins"][0].this] if (item.alias or item.name).lower() == table)
    body.set(FROM_KEY, exp.From(this=kept_table.copy()))
    body.set("joins", None)
    condition = None
    for test in grouped.kept:
        condition = test.copy() if condition is None else exp.And(this=condition, expression=test.copy())
    body.set("where", exp.Where(this=condition) if condition is not None else None)
    for node in list(body.find_all(exp.Column)):  # what is left names the parent by key columns only
        if node.table.lower() == parent:
            node.replace(exp.column(grouped.partner[node.name.lower()], table=table))
    terms: list[exp.Expression] = []
    for term in body.args["group"].expressions:
        if not any(term == other for other in terms):
            terms.append(term)
    body.args["group"].set("expressions", terms)
    return copy
