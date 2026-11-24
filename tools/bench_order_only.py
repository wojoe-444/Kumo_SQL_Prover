"""Whether two queries use the dates or times of a schema for nothing but their order.

A benchmark schema may say that one date or time column of a table is before another (``login < logout``). The
prover's value checks state an ordering as an ordering of numbers (``kumo_sql_prover.value_checks``), and a date is
not a number: MySQL reads ``login + 1``, ``login = '10:00:00'``, ``CONCAT(login, '')`` and ``login LIKE '10%'`` by
rules of the date and time types that the prover's numbers do not follow. So a harness passes such a check only for a
pair of queries that does nothing with a value of that type except

* compare it with another value of the same type (``= <> < <= > >= <=>``, ``BETWEEN``, ``IN`` a list or a subquery),
* test it for NULL, count it, group by it, order by it, partition by it,
* take one of several (``MIN``, ``MAX``, ``GREATEST``, ``LEAST``, ``COALESCE``, ``ANY_VALUE``, ``LAG`` and ``LEAD``
  without a default, ``FIRST_VALUE``, ``LAST_VALUE``),
* select it, under its own name or another.

Every one of those gives the same answer for any values that are in the same order: a query that uses dates only
so cannot tell them from numbers in that order, which is all the check then says.

``compared_only`` goes by names. A value of the type is a column the schema declares with it, or a name a query gives
to such a value (``MIN(login) AS first_login``); every column a query writes with one of those names has to stand in
one of the places above. Anything the function does not know is a refusal: a name the schema declares with two types,
a name given to something else as well, a column list on a table name (``WITH c (a, b) AS ..``), a set operation
whose branches do not agree on where the values are, ``SELECT *`` in one, a window frame over an ordering by such a
value, an unnamed computed value in a derived table. A refusal means the check is not passed, which is a fact not
used and always safe.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

TEMPORAL_TYPES = ("DATE", "TIME", "DATETIME", "TIMESTAMP")

_TESTS = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.NullSafeEQ)
_ONE_OF = (exp.Min, exp.Max, exp.Greatest, exp.Least, exp.Coalesce, exp.AnyValue, exp.FirstValue, exp.LastValue)
_NEIGHBOURS = (exp.Lag, exp.Lead)
_SET_OPERATIONS = (exp.Union, exp.Intersect, exp.Except)


def _unwrapped(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _arguments(node: exp.Expression) -> list[exp.Expression]:
    out = []
    for value in node.args.values():
        if isinstance(value, exp.Expression):
            out.append(value)
        elif isinstance(value, list):
            out.extend(item for item in value if isinstance(item, exp.Expression))
    return out


def _first_select(node: exp.Expression | None) -> exp.Select | None:
    """The first select of a set operation's side (every set operation inside it is compared on its own)."""

    while isinstance(node, (exp.Paren, exp.Subquery) + _SET_OPERATIONS):
        node = node.this
    return node if isinstance(node, exp.Select) else None


class _Reader:
    def __init__(self, names: set[str]):
        self.names = names

    def value(self, node: exp.Expression | None) -> bool:
        """Whether ``node`` is a value of the type: a column with one of the names, or one of several such values."""

        if node is None:
            return False
        node = _unwrapped(node)
        if isinstance(node, exp.Column):
            return node.name.lower() in self.names
        if isinstance(node, _ONE_OF):
            arguments = _arguments(node)
            return bool(arguments) and all(self.value(argument) for argument in arguments)
        if isinstance(node, _NEIGHBOURS):
            return node.args.get("default") is None and self.value(node.this)
        if isinstance(node, exp.Window):
            return self.value(node.this)
        if isinstance(node, exp.Subquery):
            inner = node.this
            if isinstance(inner, exp.Select) and len(inner.expressions) == 1:
                item = inner.expressions[0]
                return self.value(item.this if isinstance(item, exp.Alias) else item)
        return False

    def placed(self, node: exp.Expression, top: exp.Expression) -> bool:
        """Whether the value ``node`` stands where only its order matters (module doc)."""

        child, parent = node, node.parent
        while isinstance(parent, exp.Paren):
            child, parent = parent, parent.parent
        if parent is None:
            return False
        if isinstance(parent, _TESTS):
            return self.value(parent.left) and self.value(parent.right)
        if isinstance(parent, exp.Between):
            return all(self.value(parent.args.get(key)) for key in ("this", "low", "high"))
        if isinstance(parent, exp.In):
            if parent.args.get("unnest") is not None or parent.args.get("field") is not None:
                return False
            if not self.value(parent.this):
                return False
            query = parent.args.get("query")
            if query is not None:
                return self.value(query)
            return bool(parent.expressions) and all(self.value(item) for item in parent.expressions)
        if isinstance(parent, exp.Is):
            return child is parent.this and isinstance(parent.expression, exp.Null)
        if isinstance(parent, exp.Count):
            return True
        if isinstance(parent, exp.Distinct):
            return isinstance(parent.parent, exp.Count) and parent.args.get("on") is None
        if isinstance(parent, exp.Group):
            return True
        if isinstance(parent, exp.Ordered):
            window = parent.find_ancestor(exp.Window, exp.Select)
            return not (isinstance(window, exp.Window) and window.args.get("spec") is not None)
        if isinstance(parent, exp.Window):
            if child.arg_key == "partition_by":
                return True
            return child is parent.this and self.value(parent) and self.placed(parent, top)
        if isinstance(parent, _ONE_OF + _NEIGHBOURS):
            return self.value(parent) and self.placed(parent, top)
        if isinstance(parent, exp.Alias):
            return child is parent.this and isinstance(parent.parent, exp.Select) and parent.arg_key == "expressions" and self._selected(parent.parent, top, named=True)
        if isinstance(parent, exp.Select) and child.arg_key == "expressions":
            return self._selected(parent, top, named=isinstance(_unwrapped(node), exp.Column))
        return False

    def _selected(self, select: exp.Select, top: exp.Expression, named: bool) -> bool:
        """Whether a value in the select list of ``select`` goes on only to places that are read here too: out of the
        query, into a named column of a derived table, or to where a scalar subquery stands."""

        node: exp.Expression = select
        while isinstance(node.parent, _SET_OPERATIONS):
            node = node.parent
        parent = node.parent
        if parent is None:
            return node is top
        if isinstance(parent, exp.Exists):
            return True
        if isinstance(parent, exp.CTE):
            return named
        if isinstance(parent, exp.Subquery):
            holder = parent.parent
            while isinstance(holder, exp.Paren):
                holder = holder.parent
            if isinstance(holder, (exp.From, exp.Join)):
                return named
            if isinstance(holder, exp.Exists):
                return True
            if isinstance(holder, _SET_OPERATIONS) or holder is None:
                return False  # a branch in parentheses: not read
            return node is select and len(select.expressions) == 1 and self.placed(parent, top)
        return False


def compared_only(queries, kinds: dict[tuple[str, str], str], kind: str) -> bool:
    """Whether ``queries`` (texts in MySQL's dialect) use the values of type ``kind`` only for their order (module
    doc). ``kinds`` gives the declared type (upper case, without a length) of each ``(table, column)``."""

    by_name: dict[str, set[str]] = {}
    for (_table, column), declared in kinds.items():
        by_name.setdefault(column.lower(), set()).add(declared)
    names = {name for name, declared in by_name.items() if declared == {kind}}
    if any(kind in declared and len(declared) > 1 for declared in by_name.values()):
        return False  # a name with two types: which one a query means is not read here
    try:
        trees = [sqlglot.parse_one(sql, read="mysql") for sql in queries]
    except Exception:
        return False
    reader = _Reader(names)
    changed = True
    while changed:  # the names the queries give to such values
        changed = False
        for tree in trees:
            for alias in tree.find_all(exp.Alias):
                name = alias.alias.lower()
                if name and name not in names and reader.value(alias.this):
                    names.add(name)
                    changed = True
    for tree in trees:
        if any(alias.args.get("columns") for alias in tree.find_all(exp.TableAlias)):
            return False
        for alias in tree.find_all(exp.Alias):
            if alias.alias.lower() in names and not reader.value(alias.this):
                return False  # the name is also given to something else
        for operation in tree.find_all(*_SET_OPERATIONS):
            lists = []
            for side in (operation.this, operation.expression):
                first = _first_select(side)
                if first is None or any(isinstance(part, exp.Star) and not isinstance(part.parent, exp.Count) for item in first.expressions for part in item.walk()):
                    return False
                lists.append([reader.value(item.this if isinstance(item, exp.Alias) else item) for item in first.expressions])
            if lists[0] != lists[1]:
                return False
        for column in tree.find_all(exp.Column):
            if column.name.lower() in names and not reader.placed(column, tree):
                return False
        for select in tree.find_all(exp.Select):
            for item in select.expressions:
                inside = _unwrapped(item)
                if not isinstance(inside, (exp.Alias, exp.Column)) and reader.value(inside) and not reader.placed(inside, tree):
                    return False
    return True
