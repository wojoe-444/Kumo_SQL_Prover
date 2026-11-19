"""A one-row select that combines several aggregates, with each aggregate as its own scalar subquery.

    SELECT ROUND(COUNT(b.id) / COUNT(DISTINCT o.id), 2) AS share FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id

is

    SELECT ROUND((SELECT COUNT(b.id) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id)
               / (SELECT COUNT(DISTINCT o.id) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id), 2) AS share

The same number is often asked with the two counts already apart
(``SELECT ROUND((SELECT COUNT(*) FROM bill WHERE paid) / (SELECT COUNT(*) FROM bill), 2)``), and the prover compares a
select with no ``FROM`` subquery by subquery. Written this way each count is proved equal to its partner by itself,
over whatever rows each side counts, and the arithmetic around them is compared as written.

**Why it holds.** A select with aggregates, no ``GROUP BY`` and no ``HAVING`` returns exactly one row whatever its
sources hold. Each aggregate in it is computed over the rows of the sources that pass ``WHERE``, which is what the
select of that one aggregate over the same sources and the same ``WHERE`` returns, in its one row. The outputs read
columns only inside the aggregates, so each output is a function of those values (and of the scalar subqueries
already written in it, which do not depend on a row). Nothing is volatile, so reading the sources once per
aggregate gives the values one reading gives.

**No quotient crosses a derived table.** MySQL computes a quotient of exact numbers (and an ``AVG``) with more digits
than its declared scale and keeps them while the value is used directly, but rounds the value to that scale when it
is stored as an output of a derived table that has to be materialised: ``SELECT t.q * 100 FROM (SELECT SUM(f) / COUNT(*)
AS q FROM bill) t`` is ``33.3300`` on a table where ``SELECT SUM(f) / COUNT(*) * 100 FROM bill`` is ``33.3333`` (MySQL
8.0.46, Kumo_SQL_Prover issue 45). So an output of a one-row table is read out of it only when it is built from
``COUNT``, ``SUM``, ``MIN`` and ``MAX`` with no division anywhere in it, values that are stored as they are; and a
select is split only when its aggregates are of that kind, so that nothing but such a value is moved into a scalar
subquery.

**What is read.** Only the query itself (not a derived table or a subquery), when it is a ``SELECT`` with a ``FROM``,
no ``GROUP BY``, ``HAVING``, ``DISTINCT``, window, ``ORDER BY``, ``LIMIT`` or set operation, in which

* every column of every source resolves inside the select (``closed_scalar_filters.closed``, with the scalar
  subqueries of the outputs set aside), and ``WHERE`` holds no subquery;
* every scalar subquery in the outputs is closed in the same sense;
* each output reads columns only inside aggregates of this select, and no aggregate holds another aggregate or a
  subquery;
* some output combines at least two such terms (aggregates or scalar subqueries). A select of one aggregate per
  output, the shape the prover already compares block by block, is left alone.

A pre-pass of the tenth hardest-cases batch (``shape_fallback.TENTH_BATCH``): it fires on the first attempt and on
a retry that switches off rules of that batch only. ``tests/test_aggregate_scalars.py`` replays the rewrite on DuckDB.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, is_aggregate, select_sources
from .clause_alias_reads import _volatile
from .closed_scalar_filters import closed
from .keyed_join_groups import declared_columns

_CLAUSES = ("group", "having", "distinct", "qualify", "windows", "order", "limit", "offset", "with_", "with", "laterals", "connect", "match", "into", "locks", "sample", "pivots")


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, (exp.Alias, exp.Paren)):
        node = node.this
    return node


_EXACT = (exp.Count, exp.Sum, exp.Min, exp.Max)


def _exact(call: exp.Expression) -> bool:
    """Whether an aggregate is a ``COUNT``, ``SUM``, ``MIN`` or ``MAX`` with no division in it (module doc)."""

    return type(call) in _EXACT and not any(isinstance(node, (exp.Div, exp.IntDiv, exp.Avg)) for node in call.walk())


def split_one_row_select(tree: exp.Expression, schema: dict | None) -> tuple[exp.Expression, bool]:
    """``tree`` with its aggregates as scalar subqueries when it is such a one-row select (module doc); whether it changed."""

    if type(tree) is not exp.Select or not schema or any(tree.args.get(key) for key in _CLAUSES):
        return tree, False
    read = _read_one_row_sources(tree, declared_columns(schema))
    if read is not None:
        return read, True
    source = tree.args.get(FROM_KEY)
    if source is None or not tree.expressions or any(isinstance(item, exp.Star) or item.is_star for item in tree.expressions):
        return tree, False
    if _volatile(tree) or any(True for _ in tree.find_all(exp.Window)):
        return tree, False
    declared = declared_columns(schema)
    own = [node for node in tree.find_all(exp.AggFunc, exp.Anonymous) if is_aggregate(node) and node.find_ancestor(exp.Select) is tree]
    nested = [node for node in tree.find_all(exp.Subquery) if node.find_ancestor(exp.Select) is tree and not isinstance(node.parent, (exp.From, exp.Join))]
    if not own:
        return tree, False
    outputs = set()
    for item in tree.expressions:
        outputs.update(id(node) for node in item.walk())
    if any(id(node) not in outputs for node in own + nested):
        return tree, False  # an aggregate or a subquery outside the select list (WHERE, ON)
    for call in own:
        if not _exact(call):
            return tree, False
        if any(node is not call and (is_aggregate(node) or isinstance(node, (exp.Query, exp.Subquery))) for node in call.walk()):
            return tree, False
        if any(id(up) in outputs and is_aggregate(up) for up in _ancestors(call, tree)):
            return tree, False
    for sub in nested:
        if any(v for key, v in sub.args.items() if key != "this") or isinstance(sub.parent, (exp.In, exp.Exists, exp.Any, exp.All)) or not closed(sub.this, declared):
            return tree, False
        if any(is_aggregate(up) for up in _ancestors(sub, tree)):
            return tree, False
    for item in tree.expressions:
        for column in item.find_all(exp.Column):
            if column.find_ancestor(exp.Select) is tree and not any(is_aggregate(up) for up in _ancestors(column, tree)):
                return tree, False
    if not any(_terms(item, tree) >= 2 for item in tree.expressions):
        return tree, False
    body = tree.copy()
    for sub in [node for node in body.find_all(exp.Subquery) if node.find_ancestor(exp.Select) is body and not isinstance(node.parent, (exp.From, exp.Join))]:
        sub.replace(exp.Literal.number(0))
    if not closed(body, declared):
        return tree, False
    result = tree.copy()
    calls = [node for node in result.find_all(exp.AggFunc, exp.Anonymous) if is_aggregate(node) and node.find_ancestor(exp.Select) is result]
    frame = exp.Select()
    for key in (FROM_KEY, "joins", "where"):
        value = result.args.get(key)
        if value is not None:
            frame.set(key, [v.copy() for v in value] if isinstance(value, list) else value.copy())
    for call in calls:
        one = frame.copy()
        one.set("expressions", [call.copy()])
        call.replace(exp.Subquery(this=one))
    for key in (FROM_KEY, "joins", "where"):
        result.set(key, None)
    return result, True


def _one_row_body(source: exp.Expression, declared: dict) -> exp.Select | None:
    """The select of a derived table that returns exactly one row (aggregates, no grouping), or None."""

    if type(source) is not exp.Subquery or not source.alias or any(v for key, v in source.args.items() if key not in ("this", "alias")):
        return None
    if source.args["alias"].args.get("columns"):
        return None
    body = source.this
    if type(body) is not exp.Select or any(body.args.get(key) for key in _CLAUSES) or body.args.get(FROM_KEY) is None:
        return None
    if _volatile(body) or any(True for _ in body.find_all(exp.Window)) or not closed(body, declared):
        return None
    names = [(item.alias_or_name or "").lower() for item in body.expressions]
    if not all(names) or len(set(names)) != len(names) or any(isinstance(item, exp.Star) or item.is_star for item in body.expressions):
        return None
    for item in body.expressions:
        calls = [node for node in item.walk() if is_aggregate(node) and node.find_ancestor(exp.Select) is body]
        if not calls or not all(_exact(call) for call in calls) or any(isinstance(node, (exp.Div, exp.IntDiv, exp.Query, exp.Subquery)) for node in item.walk()):
            return None  # a quotient is rounded where a derived table stores it: it is not read out of one
        for column in item.find_all(exp.Column):
            if not any(is_aggregate(up) for up in _ancestors(column, body)):
                return None
    return body


def _read_one_row_sources(tree: exp.Select, declared: dict) -> exp.Select | None:
    """A select whose sources are all one-row derived tables, cross joined, with each read of an output as the scalar
    subquery that computes it: the cross join of one-row tables is one row, and no source is left."""

    if tree.args.get("where") is not None:
        return None
    joins = tree.args.get("joins") or []
    for join in joins:
        if join.args.get("on") is not None or join.args.get("using") or join.args.get("side") or join.args.get("method") or (join.args.get("kind") or "").upper() not in ("", "CROSS", "INNER"):
            return None
    sources = select_sources(tree)
    bodies = {}
    for source in sources:
        body = _one_row_body(source, declared)
        if body is None or source.alias.lower() in bodies:
            return None
        bodies[source.alias.lower()] = body
    if not bodies or any(is_aggregate(node) and node.find_ancestor(exp.Select) is tree for node in tree.find_all(exp.AggFunc, exp.Anonymous)):
        return None
    if _volatile(tree) or any(True for _ in tree.find_all(exp.Window)) or any(isinstance(item, exp.Star) or item.is_star for item in tree.expressions):
        return None
    outputs = {alias: {(item.alias_or_name or "").lower(): item.unalias() for item in body.expressions} for alias, body in bodies.items()}
    for sub in tree.find_all(exp.Subquery):
        if sub.find_ancestor(exp.Select) is tree and not isinstance(sub.parent, (exp.From, exp.Join)) and not closed(sub.this, declared):
            return None  # a nested subquery of the outputs that may read a column of one of these tables
    result = tree.copy()
    for column in list(result.find_all(exp.Column)):
        if column.find_ancestor(exp.Select) is not result:
            continue  # inside a derived table, or inside a nested subquery that is closed
        if column.args.get("db") or not isinstance(column.this, exp.Identifier):
            return None
        name = column.name.lower()
        owners = [alias for alias, named in outputs.items() if name in named] if not column.table else ([column.table.lower()] if name in outputs.get(column.table.lower(), {}) else [])
        if len(owners) != 1:
            return None
        body = bodies[owners[0]]
        one = exp.Select()
        for key in (FROM_KEY, "joins", "where"):
            value = body.args.get(key)
            if value is not None:
                one.set(key, [v.copy() for v in value] if isinstance(value, list) else value.copy())
        one.set("expressions", [outputs[owners[0]][name].copy()])
        scalar = exp.Subquery(this=one)
        if column.parent is result:  # an output that is the bare column keeps its name
            scalar = exp.Alias(this=scalar, alias=exp.to_identifier(column.name))
        column.replace(scalar)
    for key in (FROM_KEY, "joins"):
        result.set(key, None)
    if not any(_terms(item, result) >= 2 for item in result.expressions):
        return None  # nothing is combined: the prover already reads a projection of a one-row table
    return result


def _terms(item: exp.Expression, select: exp.Select) -> int:
    """How many aggregates and scalar subqueries of ``select`` an output combines (0 when it is one of them by itself)."""

    top = _unwrap(item)
    if is_aggregate(top) or isinstance(top, exp.Subquery):
        return 0
    count = 0
    for node in top.walk():
        if node.find_ancestor(exp.Select) is not select:
            continue
        if is_aggregate(node) or (isinstance(node, exp.Subquery) and not isinstance(node.parent, (exp.From, exp.Join))):
            count += 1
    return count


def _ancestors(node: exp.Expression, stop: exp.Expression):
    node = node.parent
    while node is not None and node is not stop:
        yield node
        node = node.parent
