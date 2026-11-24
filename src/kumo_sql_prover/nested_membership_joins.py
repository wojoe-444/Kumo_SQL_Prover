"""A membership test inside the subquery of another one, read as a join in that subquery.

    SELECT s.name FROM salesperson AS s
    WHERE s.sales_id NOT IN (SELECT sales_id FROM orders
                             WHERE com_id IN (SELECT com_id FROM company WHERE name = 'RED'))

lists, for the outer test, the ``sales_id`` of the orders whose company is RED. The same list comes from

    SELECT orders.sales_id FROM orders, company AS c WHERE orders.com_id = c.com_id AND c.name = 'RED'

and that is the spelling the provers read: they take one membership test apart, not one inside another. This
pre-pass (hardest-cases batch 29) writes the inner test as the join.

**Why the outer test cannot tell.** ``y IN (B)`` and ``y NOT IN (B)`` depend only on which values ``B`` lists
(a NULL among them included), not on how often. A row ``b`` of ``B``'s table passes ``x IN (SELECT k FROM p WHERE
c)`` exactly when some row of ``p`` has ``c`` TRUE and ``x = k`` TRUE; a test that is FALSE or NULL drops the row
either way, since it is a top-level ``AND`` part of the ``WHERE``. In the join, ``b`` appears once for each such row
of ``p`` and not at all when there is none. So the join lists the same values, some of them more often.

**What is read.** ``B`` is the subquery of an ``IN`` (negated or not): one select item, a column of its one source,
a declared table; a ``WHERE``; optionally ``DISTINCT``; nothing else. One top-level ``AND`` part of its ``WHERE`` is
``x IN (N)`` with ``x`` a column of that table. ``N`` selects one column of its one source, a declared table, with
an optional ``WHERE`` and ``DISTINCT``, reads nothing but its own table (no outer name, no nested query), and is
the only nested query in ``B``. The names are settled before the sources share a scope: every bare column of ``B``
must be a declared column of ``B``'s table and gets its name, and every column of ``N`` is written under a new
alias that occurs nowhere in the statement.

**Left alone.** ``NOT IN`` as the inner test (FALSE and NULL differ there), an inner test under ``OR``, a ``B`` with a
join, grouping, ``HAVING``, ``LIMIT`` or a window, an ``N`` that reads an outer column, a table without declared
columns, a bare name the declared columns do not place.

It runs in the last step of batch 27 only, so a pair an earlier step decides is decided as it was.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import conjuncts, select_sources

NAME = "nested_membership_joins"
_PLAIN_ARGS = frozenset({"expressions", "from", "from_", "where", "distinct"})


def read_nested_membership_joins(tree: exp.Expression, schema: dict[str, list[str]] | None) -> tuple[exp.Expression, bool]:
    """``(tree with each such inner test written as a join, whether one was)``. The input is not changed."""

    if not schema:
        return tree, False
    known = {name.lower().split(".")[-1]: {str(c).lower() for c in columns} for name, columns in schema.items()}
    if len(known) != len(schema) or not any(_found(node, known) for node in tree.find_all(exp.In)):
        return tree, False
    copy = tree.copy()
    text = copy.sql().lower()
    count = 0
    while True:  # one at a time: a test two levels down is read after the one above it was
        found = next((hit for hit in (_found(node, known) for node in copy.find_all(exp.In)) if hit), None)
        if found is None or count > 20:
            return copy, count > 0
        count += 1
        alias = f"kqm{count}"
        while alias in text:
            alias += "x"
        _write(*found, alias)


def _plain(select: exp.Expression | None) -> exp.Table | None:
    """The one declared-looking table a plain one-column select reads, None when the select is anything more."""

    if not isinstance(select, exp.Select) or any(value for key, value in select.args.items() if key not in _PLAIN_ARGS):
        return None
    distinct = select.args.get("distinct")
    if distinct is not None and distinct.args.get("on"):
        return None
    if len(select.expressions) != 1:
        return None
    sources = select_sources(select)
    if len(sources) != 1:
        return None
    table = sources[0]
    if not isinstance(table, exp.Table) or not isinstance(table.this, exp.Identifier) or table.this.args.get("quoted") or table.args.get("db") or table.args.get("pivots") or table.args.get("joins") or not table.alias_or_name:
        return None
    alias = table.args.get("alias")
    if alias is not None and alias.args.get("columns"):
        return None
    return table


def _own(node: exp.Expression, table: exp.Table, columns: set[str]) -> bool:
    """Whether ``node`` is a plain column of ``table``: bare and declared, or under the table's name and declared."""

    if not isinstance(node, exp.Column) or not isinstance(node.this, exp.Identifier) or node.this.args.get("quoted") or node.args.get("db") or node.args.get("catalog"):
        return False
    if node.name.lower() not in columns:
        return False
    return not node.table or node.table.lower() == table.alias_or_name.lower()


def _query(node: exp.In) -> exp.Select | None:
    query = node.args.get("query")
    if node.expressions or node.args.get("unnest") is not None or node.args.get("field") is not None:
        return None
    if isinstance(query, exp.Subquery) and not any(value for key, value in query.args.items() if key != "this"):
        query = query.this
    return query if isinstance(query, exp.Select) else None


def _found(outer: exp.In, known: dict[str, set[str]]):
    """``(body, its table, the inner test, the inner select, its table)`` when ``outer`` holds a test that is read."""

    body = _query(outer)
    table = _plain(body)
    if table is None or body.args.get("where") is None:
        return None
    columns = known.get(table.name.lower())
    if columns is None:
        return None
    item = body.expressions[0]
    if not _own(item.this if isinstance(item, exp.Alias) else item, table, columns):
        return None
    nested = [node for node in body.find_all(exp.Select) if node is not body]
    if len(nested) != 1 or body.find(exp.Exists, exp.Window, exp.AggFunc) is not None:
        return None
    inner_select = nested[0]
    tests = [part for part in conjuncts(body.args["where"].this) if isinstance(part, exp.In) and _query(part) is inner_select]
    if len(tests) != 1:
        return None
    test = tests[0]
    inner_table = _plain(inner_select)
    if inner_table is None or not _own(test.this, table, columns):
        return None
    inner_columns = known.get(inner_table.name.lower())
    if inner_columns is None:
        return None
    inner_item = inner_select.expressions[0]
    if not _own(inner_item.this if isinstance(inner_item, exp.Alias) else inner_item, inner_table, inner_columns):
        return None
    inside = {id(node) for node in inner_select.find_all(exp.Column)}
    for column in body.find_all(exp.Column):
        if id(column) in inside:
            if not _own(column, inner_table, inner_columns):
                return None  # an outer name, a star, or a name the declared columns do not place
        elif isinstance(column.this, exp.Star) or (not column.table and not _own(column, table, columns)):
            return None
    if inner_select.find(exp.Star) is not None or body.find(exp.Star) is not None:
        return None
    return body, table, test, inner_select, inner_table


def _write(body: exp.Select, table: exp.Table, test: exp.In, inner_select: exp.Select, inner_table: exp.Table, alias: str) -> None:
    inside = {id(node) for node in inner_select.find_all(exp.Column)}
    name = table.alias_or_name
    for column in list(body.find_all(exp.Column)):
        if id(column) in inside:
            column.set("table", exp.to_identifier(alias))
        elif not column.table:
            column.set("table", exp.to_identifier(name))
    inner_item = inner_select.expressions[0]
    key = inner_item.this if isinstance(inner_item, exp.Alias) else inner_item
    parts = [exp.EQ(this=test.this.copy(), expression=key.copy())]
    if inner_select.args.get("where") is not None:
        parts += [exp.paren(part.copy()) if isinstance(part, exp.Or) else part.copy() for part in conjuncts(inner_select.args["where"].this)]
    joined = inner_table.copy()
    joined.set("alias", exp.TableAlias(this=exp.to_identifier(alias)))
    kept = []
    for part in conjuncts(body.args["where"].this):
        if part is test:
            kept += parts
        else:
            kept.append(exp.paren(part.copy()) if isinstance(part, exp.Or) else part.copy())
    body.set("where", exp.Where(this=exp.and_(*kept)))
    body.append("joins", exp.Join(this=joined))
