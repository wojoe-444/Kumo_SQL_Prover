"""A ``LIMIT`` on the driving rows of a query is the ``LIMIT`` of the query when each row stays one row.

"The ten newest posts, each with its number of comments" is written with the cut last, or with the cut first and the
counting done for the ten posts only:

    SELECT p.title, COALESCE(c.n, 0) AS comments
    FROM posts AS p LEFT JOIN (SELECT post, COUNT(*) AS n FROM comments GROUP BY post) AS c ON p.id = c.post
    ORDER BY p.made DESC LIMIT 10

    WITH top AS (SELECT id, title, made FROM posts ORDER BY made DESC LIMIT 10)
    SELECT t.title, COALESCE(c.n, 0) AS comments
    FROM top AS t
    LEFT JOIN (SELECT post, COUNT(*) AS n FROM comments WHERE post IN (SELECT id FROM top) GROUP BY post) AS c ON t.id = c.post
    ORDER BY t.made DESC

``lift`` writes the second like the first: the tail of the driving table moves to the end of the query. The prover
then finds the same tail on both queries and compares what is left (``peeled_tails``). Three readings, each a way of
writing the same rows:

**1. The tail of the driving table is the tail of the query.** The query is ``SELECT items FROM D AS d`` plus joins,
with no ``WHERE``, grouping, ``DISTINCT``, aggregate, window or nested query of its own and no ``LIMIT`` or
``OFFSET``; ``D`` is a derived table, or a ``WITH`` query read exactly once, that ends in ``ORDER BY keys LIMIT n``
(``OFFSET m``); every join is a ``LEFT JOIN`` on one equality between a column of the joined table and a column of
``d`` or of an earlier joined table, and the joined table has at most one row per value of that column: a plain table
whose column is a declared NOT NULL one-column key, a select of one such table, or a select grouped by exactly that
column. Then the query returns exactly one row for each row of ``D``, and that row carries the ``D`` row's columns.
So: pick the rows of ``D`` its tail may keep and extend each, or extend every row of ``D`` without the tail and keep
the rows the same ordering and the same cut may keep. The two are the same sets of possible results, because the
order keys are columns of ``D`` that each output row carries unchanged. Each order key must be an output column of
``D``; it is written as that column of ``d``. The query's own ``ORDER BY`` (it had no ``LIMIT``) only arranged the
rows and is replaced. When the query has its own ``LIMIT n2`` under the very same ordering and neither has an
``OFFSET``, the cut is the smaller of the two.

**2. A row number tested against a constant is a ``LIMIT``.** ``FROM (SELECT .., ROW_NUMBER() OVER (ORDER BY keys)
AS rn FROM ..) AS d WHERE rn <= n``, where nothing else reads ``rn``, keeps the first ``n`` rows by ``keys``, tied
rows numbered either way: the rows ``ORDER BY keys LIMIT n`` may keep. No ``PARTITION BY``, no ``DISTINCT`` beside
the number, and no key that the select list would give another meaning as an ``ORDER BY`` name.

**3. A counted table restricted to the driving rows is the counted table.** The driving table is a ``WITH`` query
``T``. A joined table ``SELECT c.fk, aggregates FROM child AS c .. GROUP BY c.fk``, joined on ``d.id = fk``, that
keeps only the child rows of ``T``'s own rows, by ``WHERE c.fk IN (SELECT id FROM T)`` or by ``JOIN T AS t ON c.fk =
t.id``, has for every row of ``d`` the same group as without the restriction: ``d.id`` is one of ``T``'s ids. The
groups it loses belong to no row of ``d``. The ``JOIN`` form needs ``T.id`` to be unique (``T`` selects from one
plain table and ``id`` is its declared NOT NULL one-column key), since two rows of ``T`` with one id would count
each child row twice. The restriction is removed. **Assumed**, and said in the proof (``TIE_ASSUMPTION``): a ``WITH``
query that cuts tied rows is the same rows wherever the query reads it.

Nothing else is read. An inner join to the driving rows is not (it can drop one of the first ``n`` rows, and the
query written the other way then shows the next row instead), nor a ``WHERE`` above the cut, nor a join that may
find two rows.

A proof found on a lifted query always carries ``TIE_ASSUMPTION``. ``tests/test_lifted_tails.py`` replays each
reading against DuckDB and holds the shapes that must be left alone.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from .ast_utils import FROM_KEY, LossySql, canonical_negation, distinct_on, faithful_sql
from .keyed_join_groups import declared_columns

NAME = "lifted_tails"
_ROOT_ARGS = frozenset({"expressions", FROM_KEY, "joins", "with", "with_", "order", "limit", "where"})
_REFUSED = (exp.AggFunc, exp.Window, exp.Query, exp.Subquery)


class _Declined(Exception):
    pass


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _only(node: exp.Expression, allowed) -> bool:
    return not any(value for key, value in node.args.items() if key not in allowed)


def _with_of(node: exp.Expression) -> exp.With | None:
    return node.args.get("with_") or node.args.get("with")


def _whole_number(node: exp.Expression | None) -> int | None:
    node = _unwrap(node) if node is not None else None
    if type(node) is exp.Literal and not node.is_string and node.this.isdigit():
        return int(node.this)
    return None


class _Query:
    """The query being read: its ``WITH`` queries, the declared tables and keys, and how a source is resolved."""

    def __init__(self, tree: exp.Select, schema: dict, keys: dict | None, not_null: dict | None) -> None:
        self.tree = tree
        self.declared = declared_columns(schema)
        self.keys = {str(table).lower(): [tuple(str(c).lower() for c in key) for key in listed] for table, listed in (keys or {}).items()}
        self.not_null = {str(table).lower(): {str(c).lower() for c in columns} for table, columns in (not_null or {}).items()}
        self.ctes: dict[str, exp.CTE] = {}
        clause = _with_of(tree)
        if any(isinstance(node, exp.With) and node is not clause for node in tree.walk()):
            raise _Declined  # a WITH inside the query could give one of the names another meaning
        if clause is not None:
            if clause.args.get("recursive"):
                raise _Declined
            for cte in clause.expressions:
                name = cte.alias_or_name.lower()
                if name in self.ctes or name in self.declared or cte.args["alias"].args.get("columns") or type(cte.this) is not exp.Select:
                    raise _Declined
                self.ctes[name] = cte

    def reads(self, name: str) -> int:
        return sum(1 for table in self.tree.find_all(exp.Table) if not table.args.get("db") and table.name.lower() == name)

    def source(self, node: exp.Expression) -> tuple[str, str, object] | None:
        """``(alias, kind, what)``: ``("base", table name)``, ``("with", name)`` or ``("derived", select)``."""

        if type(node) is exp.Subquery:
            if not node.alias or node.args["alias"].args.get("columns") or type(node.this) is not exp.Select or not _only(node, ("this", "alias")):
                return None
            return node.alias.lower(), "derived", node.this
        if type(node) is not exp.Table or not isinstance(node.this, exp.Identifier) or not _only(node, ("this", "alias")):
            return None
        alias = node.args.get("alias")
        if alias is not None and alias.args.get("columns"):
            return None
        name = node.name.lower()
        if name in self.ctes:
            return node.alias_or_name.lower(), "with", name
        if name in self.declared:
            return node.alias_or_name.lower(), "base", name
        return None

    def body(self, kind: str, what) -> exp.Select | None:
        return what if kind == "derived" else self.ctes[what].this if kind == "with" else None

    def one_column_key(self, table: str, column: str) -> bool:
        return column in self.not_null.get(table, ()) and (column,) in self.keys.get(table, ())

    def outputs(self, select: exp.Select) -> list[tuple[str | None, exp.Expression | None]] | None:
        """``(name, expression)`` for each output column of ``select``; ``None`` when a star hides them."""

        listed = []
        for item in select.expressions:
            if type(item) is exp.Star and len(select.expressions) == 1 and not select.args.get("joins"):
                only = self.source(select.args[FROM_KEY].this) if select.args.get(FROM_KEY) is not None else None
                if only is None or only[1] != "base" or not _only(item, ()):
                    return None
                return [(name, None) for name in sorted(self.declared[only[2]])]  # each a column of the one table
            if isinstance(item, exp.Star) or (type(item) is exp.Column and isinstance(item.this, exp.Star)):
                return None
            if type(item) is exp.Alias:
                listed.append((item.alias.lower(), item.this))
            elif type(item) is exp.Column:
                listed.append((item.name.lower(), item))
            else:
                listed.append((None, item))
        return listed

    def single_table(self, select: exp.Select) -> tuple[str, str] | None:
        """``(alias, table)`` when ``select`` reads one plain table and keeps or drops whole rows of it."""

        if not _only(select, ("expressions", FROM_KEY, "where", "order", "limit", "offset")) or select.args.get(FROM_KEY) is None or distinct_on(select):
            return None
        only = self.source(select.args[FROM_KEY].this)
        if only is None or only[1] != "base":
            return None
        if any(isinstance(node, (exp.AggFunc, exp.Window)) for item in select.expressions for node in item.walk()):
            return None
        return only[0], only[2]

    def unique_output(self, select: exp.Select, column: str) -> bool:
        """Whether ``select`` has at most one row per value of its output ``column``."""

        outputs = self.outputs(select)
        if outputs is None or [name for name, _ in outputs].count(column) != 1:
            return False
        expression = next(expr for name, expr in outputs if name == column)
        group = select.args.get("group")
        if group is not None:
            if not _only(group, ("expressions",)) or len(group.expressions) != 1 or type(_unwrap(group.expressions[0])) is not exp.Column:
                return False
            grouped = _unwrap(group.expressions[0])
            if expression is None or type(expression) is not exp.Column or select.args.get("distinct") and distinct_on(select):
                return False
            if any(type(item) is exp.Alias and item.alias.lower() == grouped.name.lower() and item.this.sql() != grouped.sql() for item in select.expressions) and not grouped.table:
                return False  # GROUP BY name could mean the select alias
            return expression.sql().lower() == grouped.sql().lower()
        only = self.single_table(select)
        if only is None:
            return False
        if expression is None:
            return self.one_column_key(only[1], column)
        if type(expression) is not exp.Column or (expression.table and expression.table.lower() != only[0]):
            return False
        return self.one_column_key(only[1], expression.name.lower())


def _qualified(node: exp.Expression) -> tuple[str, str] | None:
    node = _unwrap(node)
    if type(node) is not exp.Column or not node.table or not isinstance(node.this, exp.Identifier) or node.args.get("db"):
        return None
    return node.table.lower(), node.name.lower()


def _equality(node: exp.Expression | None) -> tuple[tuple[str, str], tuple[str, str]] | None:
    node = _unwrap(node) if node is not None else None
    if type(node) is not exp.EQ:
        return None
    ends = _qualified(node.this), _qualified(node.expression)
    return None if None in ends else ends


def _numbered_rows(query: _Query, root: exp.Select, alias: str, body: exp.Select) -> bool:
    """Reading 2: ``WHERE rn <= n`` over ``ROW_NUMBER() OVER (ORDER BY keys) AS rn`` becomes the body's tail."""

    where = root.args.get("where")
    test = _unwrap(where.this) if where is not None else None
    if type(test) not in (exp.LTE, exp.LT):
        return False
    column, count = _unwrap(test.this), _whole_number(test.expression)
    if type(column) is not exp.Column or (column.table and column.table.lower() != alias) or count is None:
        return False
    if not column.table and root.args.get("joins"):
        return False  # the name could be a joined table's column
    count -= type(test) is exp.LT
    name = column.name.lower()
    if count < 1 or not _only(body, ("expressions", FROM_KEY, "joins", "where", "group", "having")):
        return False
    numbered = [item for item in body.expressions if type(item) is exp.Alias and item.alias.lower() == name]
    if len(numbered) != 1 or any(isinstance(item, exp.Star) or (type(item) is exp.Column and isinstance(item.this, exp.Star)) for item in body.expressions):
        return False
    window = numbered[0].this
    if type(window) is not exp.Window or type(window.this) is not exp.RowNumber or not _only(window, ("this", "order", "over")) or window.args.get("order") is None:
        return False
    if (window.args.get("over") or "OVER").upper() != "OVER":
        return False
    # nothing else reads the number
    for node in root.walk():
        if type(node) is exp.Column and node.name.lower() == name and node is not column and node.find_ancestor(exp.Select) is root:
            return False
    if any(isinstance(item, exp.Star) or (type(item) is exp.Column and isinstance(item.this, exp.Star)) for item in root.expressions):
        return False
    order = window.args["order"]
    aliases = {item.alias.lower(): item.this.sql() for item in body.expressions if type(item) is exp.Alias and item is not numbered[0]}
    for key in order.expressions:
        if type(key) is not exp.Ordered or type(_unwrap(key.this)) is exp.Literal:
            return False  # an ORDER BY number would be a position
        for node in key.this.walk():
            if type(node) is exp.Column and not node.table and aliases.get(node.name.lower(), node.sql()) != node.sql():
                return False  # as an ORDER BY name it would mean the select alias
    numbered[0].pop()
    body.set("order", order.copy())
    body.set("limit", exp.Limit(expression=exp.Literal.number(count)))
    root.set("where", None)
    return True


def _unrestrict(query: _Query, root: exp.Select, alias: str, name: str) -> bool:
    """Reading 3: drop the restriction of joined counted tables to the rows of the driving ``WITH`` query."""

    changed = False
    for join in root.args.get("joins") or []:
        joined = query.source(join.this)
        ends = _equality(join.args.get("on"))
        if joined is None or joined[1] == "base" or ends is None or not _only(join, ("this", "side", "kind", "on")):
            continue
        if joined[1] == "with" and query.reads(joined[2]) != 1:
            continue
        if (join.args.get("side") or "").upper() not in ("", "LEFT") or (join.args.get("kind") or "").upper() not in ("", "INNER", "OUTER"):
            continue  # a RIGHT or FULL join would show the groups of other rows
        mine, other = (ends if ends[0][0] == joined[0] else ends[::-1])
        if mine[0] != joined[0] or other[0] != alias:
            continue
        counted = query.body(joined[1], joined[2])
        if _restriction_dropped(query, counted, mine[1], name, other[1]):
            changed = True
    return changed


def _restriction_dropped(query: _Query, counted: exp.Select, output: str, name: str, driving: str) -> bool:
    """Whether ``counted`` (grouped by its output ``output``) was restricted to the ``driving`` column of ``name``,
    and now is not."""

    if not _only(counted, ("expressions", FROM_KEY, "joins", "where", "group")) or counted.args.get(FROM_KEY) is None:
        return False
    group = counted.args.get("group")
    if group is None or not _only(group, ("expressions",)) or len(group.expressions) != 1:
        return False
    sources = [counted.args[FROM_KEY].this] + [join.this for join in counted.args.get("joins") or []]
    read = [query.source(source) for source in sources]
    if None in read or len({source[0] for source in read}) != len(read):
        return False
    children = [source for source in read if source[1] == "base"]
    if len(children) != 1:
        return False
    child, child_columns = children[0][0], query.declared[children[0][2]]
    outputs = query.outputs(query.ctes[name].this) or []
    names = [column for column, _ in outputs]
    if names.count(driving) != 1:
        return False

    def of_child(node: exp.Expression) -> str | None:
        node = _unwrap(node)
        if type(node) is not exp.Column or not isinstance(node.this, exp.Identifier):
            return None
        if node.table:
            return node.name.lower() if node.table.lower() == child and node.name.lower() in child_columns else None
        if len(read) > 1 and node.name.lower() in names:
            return None  # could be a column of the WITH query
        return node.name.lower() if node.name.lower() in child_columns else None

    restricting = None
    if len(read) == 2:
        # JOIN form: child JOIN name ON child.fk = name.driving, and the WITH query has one row per driving value
        other = next(source for source in read if source is not children[0])
        join = counted.args["joins"][0]
        if other[1] != "with" or other[2] != name or not _only(join, ("this", "kind", "on")) or (join.args.get("kind") or "").upper() not in ("", "INNER"):
            return False
        test = _unwrap(join.args["on"]) if join.args.get("on") is not None else None
        if type(test) is not exp.EQ:
            return False
        sides = [_unwrap(test.this), _unwrap(test.expression)]
        theirs = [side for side in sides if type(side) is exp.Column and side.table and side.table.lower() == other[0] and side.name.lower() == driving]
        if len(theirs) != 1:
            return False
        linked = of_child(next(side for side in sides if side is not theirs[0]))
        if linked is None or not query.unique_output(query.ctes[name].this, driving) or query.ctes[name].this.args.get("group") is not None:
            return False
        # every other mention of the WITH query is its driving column, which the join makes equal to the linked one
        for node in counted.walk():
            if type(node) is exp.Column and node.table and node.table.lower() == other[0] and node.find_ancestor(exp.Join) is not join:
                if node.name.lower() != driving:
                    return False
        restricting = ("join", join, other[0], linked)
    elif len(read) == 1:
        where = counted.args.get("where")
        parts = _conjuncts(where.this) if where is not None else []
        for part in parts:
            if type(part) is not exp.In or part.args.get("expressions") or _only(part, ("this", "query")) is False:
                continue
            listed = part.args.get("query")
            listed = listed.this if type(listed) is exp.Subquery and _only(listed, ("this",)) else listed
            if type(listed) is not exp.Select or not _only(listed, ("expressions", FROM_KEY)) or len(listed.expressions) != 1 or listed.args.get(FROM_KEY) is None:
                continue
            inner = query.source(listed.args[FROM_KEY].this)
            item = _unwrap(listed.expressions[0])
            if inner is None or inner[1] != "with" or inner[2] != name or type(item) is not exp.Column or item.name.lower() != driving:
                continue
            if item.table and item.table.lower() != inner[0]:
                continue
            linked = of_child(part.this)
            if linked is not None:
                restricting = ("in", part, parts, linked)
                break
    if restricting is None:
        return False
    linked = restricting[3]
    grouped = _unwrap(group.expressions[0])
    by_driving = restricting[0] == "join" and type(grouped) is exp.Column and bool(grouped.table) and grouped.table.lower() == restricting[2]
    if not by_driving and of_child(grouped) != linked:
        return False
    listed = query.outputs(counted)
    if listed is None or [column for column, _ in listed].count(output) != 1:
        return False
    shown = next(expr for column, expr in listed if column == output)
    if shown is None or shown.sql().lower() != grouped.sql().lower():
        return False  # the joined column is not the grouped one
    if any(type(item) is exp.Alias and not grouped.table and item.alias.lower() == grouped.name.lower() and item.this.sql() != grouped.sql() for item in counted.expressions):
        return False
    # every column left must be the child's
    skip = restricting[1]
    for node in counted.walk():
        if type(node) is not exp.Column or any(parent is skip for parent in _ancestors(node)):
            continue
        if restricting[0] == "join" and node.table and node.table.lower() == restricting[2]:
            continue  # the driving column, rewritten below
        if of_child(node) is None:
            return False
    if any(isinstance(node, (exp.Query, exp.Subquery, exp.Window)) and not any(parent is skip for parent in _ancestors(node)) and node is not skip for node in counted.walk() if node is not counted):
        return False
    if restricting[0] == "join":
        _, join, theirs, linked = restricting
        for node in list(counted.walk()):
            if type(node) is exp.Column and node.table and node.table.lower() == theirs and not any(parent is join for parent in _ancestors(node)):
                node.replace(exp.column(linked, table=child))
        if query.source(counted.args[FROM_KEY].this)[1] == "with":
            counted.args[FROM_KEY].set("this", join.this.copy())
        counted.set("joins", None)
    else:
        _, part, parts, _ = restricting
        rest = [other for other in parts if other is not part]
        counted.set("where", exp.Where(this=_and_all([other.copy() for other in rest])) if rest else None)
    return True


def _ancestors(node: exp.Expression):
    node = node.parent
    while node is not None:
        yield node
        node = node.parent


def _conjuncts(node: exp.Expression) -> list[exp.Expression]:
    node = _unwrap(node)
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _and_all(parts: list[exp.Expression]) -> exp.Expression:
    parts = [exp.Paren(this=part) if isinstance(part, (exp.Or, exp.Xor)) else part for part in parts]
    result = parts[0]
    for part in parts[1:]:
        result = exp.And(this=result, expression=part)
    return result


def _mapped_order(query: _Query, body: exp.Select, alias: str) -> list[exp.Ordered] | None:
    """The ``ORDER BY`` of ``body`` written over its output columns as columns of ``alias``."""

    outputs = query.outputs(body)
    if outputs is None or any(name is None for name, _ in outputs) or len({name for name, _ in outputs}) != len(outputs):
        return None
    mapped = []
    for item in body.args["order"].expressions:
        if type(item) is not exp.Ordered or not _only(item, ("this", "desc", "nulls_first")):
            return None
        key = _unwrap(item.this)
        position = None
        if type(key) is exp.Literal:
            number = _whole_number(key)
            if number is None or not 1 <= number <= len(outputs) or any(expr is None for _, expr in outputs):
                return None  # no position, or a star: this module does not know the order of a table's columns
            position = number - 1
        else:
            named = [i for i, (name, _) in enumerate(outputs) if type(key) is exp.Column and not key.table and key.name.lower() == name]
            if len(named) == 1:
                position = named[0]  # an ORDER BY name is the output column of that name
            elif not named:
                same = [i for i, (_, expr) in enumerate(outputs) if expr is not None and expr.sql().lower() == key.sql().lower()]
                if len(same) == 1:
                    position = same[0]
                elif not same and type(key) is exp.Column and all(expr is None for _, expr in outputs):
                    position = next((i for i, (name, _) in enumerate(outputs) if name == key.name.lower()), None)  # SELECT * of one table
        if position is None:
            return None
        # the names as the query writes them (some dialects tell ``T`` from ``t``)
        shown = body.expressions[position] if len(body.expressions) == len(outputs) and outputs[position][1] is not None else key
        new = item.copy()
        new.set("this", exp.column(shown.alias_or_name, table=alias))
        mapped.append(new)
    return mapped


def _lift(query: _Query, root: exp.Select) -> bool:
    """Readings 2, 3 and then 1 on the query's own select."""

    if not _only(root, _ROOT_ARGS) or root.args.get(FROM_KEY) is None:
        return False
    driver = query.source(root.args[FROM_KEY].this)
    if driver is None or driver[1] == "base":
        return False
    alias, kind, what = driver
    if kind == "with" and query.reads(what) == 1 or kind == "derived":
        _numbered_rows(query, root, alias, query.body(kind, what))
    if root.args.get("where") is not None:
        return False  # a filter above the cut
    changed = kind == "with" and _unrestrict(query, root, alias, what)
    if kind == "with" and query.reads(what) != 1:
        return changed
    body = query.body(kind, what)
    limit, offset = body.args.get("limit"), body.args.get("offset")
    if body.args.get("order") is None or type(limit) is not exp.Limit or not _only(limit, ("expression",)) or _whole_number(limit.expression) is None or distinct_on(body):
        return changed
    if offset is not None and (type(offset) is not exp.Offset or not _only(offset, ("expression",)) or _whole_number(offset.expression) is None):
        return changed
    # every item is worked out from one row: the driving row and what its lookups find
    if any(isinstance(node, _REFUSED) for item in root.expressions for node in item.walk()):
        return changed
    seen = {alias}
    for join in root.args.get("joins") or []:
        joined = query.source(join.this)
        ends = _equality(join.args.get("on"))
        if joined is None or ends is None or not _only(join, ("this", "side", "kind", "on")) or joined[0] in seen:
            return changed
        if (join.args.get("side") or "").upper() != "LEFT" or (join.args.get("kind") or "").upper() not in ("", "OUTER"):
            return changed
        mine, other = (ends if ends[0][0] == joined[0] else ends[::-1])
        if mine[0] != joined[0] or other[0] not in seen:
            return changed
        if joined[1] == "base":
            if not query.one_column_key(joined[2], mine[1]):
                return changed
        elif not query.unique_output(query.body(joined[1], joined[2]), mine[1]):
            return changed
        seen.add(joined[0])
    mapped = _mapped_order(query, body, root.args[FROM_KEY].this.alias_or_name)
    if mapped is None:
        return changed
    count = _whole_number(limit.expression)
    if root.args.get("limit") is not None:
        # the query cuts too: only under the very same ordering, and without an OFFSET on either
        own, order = root.args["limit"], root.args.get("order")
        if type(own) is not exp.Limit or not _only(own, ("expression",)) or _whole_number(own.expression) is None or offset is not None or order is None:
            return changed
        outputs = {item.alias.lower(): item.this for item in root.expressions if type(item) is exp.Alias}
        keys = []
        for item in order.expressions:
            if type(item) is not exp.Ordered or not _only(item, ("this", "desc", "nulls_first")):
                return changed
            key = _unwrap(item.this)
            if type(key) is exp.Literal:
                number = _whole_number(key)
                if number is None or not 1 <= number <= len(root.expressions):
                    return changed
                key = root.expressions[number - 1]
                key = _unwrap(key.this if type(key) is exp.Alias else key)
            elif type(key) is exp.Column and not key.table:
                if key.name.lower() not in outputs:
                    return changed
                key = _unwrap(outputs[key.name.lower()])
            column = _qualified(key)
            if column is None or column[0] != alias:
                return changed
            keys.append((column[1], bool(item.args.get("desc")), item.args.get("nulls_first")))
        if keys != [(item.this.name.lower(), bool(item.args.get("desc")), item.args.get("nulls_first")) for item in mapped]:
            return changed
        count = min(count, _whole_number(own.expression))
    # a key that the query shows is named by its position, the way the solver layer finds an output column
    single = not root.args.get("joins")
    for item in mapped:
        shown = [
            index
            for index, listed in enumerate(root.expressions)
            for column in [_unwrap(listed.this if type(listed) is exp.Alias else listed)]
            if type(column) is exp.Column and column.name.lower() == item.this.name.lower() and (column.table.lower() == alias if column.table else single)
        ]
        if shown:
            item.set("this", exp.Literal.number(shown[0] + 1))
    body.set("order", None)
    body.set("limit", None)
    body.set("offset", None)
    root.set("order", exp.Order(expressions=mapped))
    root.set("limit", exp.Limit(expression=exp.Literal.number(count)))
    if offset is not None:
        root.set("offset", offset.copy())
    return True


def lift(sql: str, dialect: str, schema: dict | None, keys: dict | None, not_null: dict | None) -> str | None:
    """``sql`` with the tail of its driving table lifted to the end of the query (see the module docstring), or
    ``None`` when nothing is read."""

    if not schema:
        return None
    try:
        from .counted_membership import _unstable
        from .quoted_names import unquote_plain_names

        tree = unquote_plain_names(canonical_negation(sqlglot.parse_one(sql, read=dialect)), dialect)
        if type(tree) is not exp.Select or _unstable(tree):
            return None
        if any(identifier.args.get("quoted") for identifier in tree.find_all(exp.Identifier)):
            return None  # names are compared without their case here
        if not _lift(_Query(tree, schema, keys, not_null), tree):
            return None
        return faithful_sql(tree, dialect)
    except (_Declined, sqlglot.errors.SqlglotError, LossySql, RecursionError, TypeError, KeyError, AttributeError):
        return None
