"""A scalar subquery that sorts by its one output and keeps the first row is ``MAX`` or ``MIN`` of that output: MySQL only.

    SELECT (SELECT n FROM nums GROUP BY n HAVING COUNT(*) = 1 ORDER BY n DESC LIMIT 1) AS n

is

    SELECT (SELECT MAX(top_rows.n)
            FROM (SELECT n FROM nums GROUP BY n HAVING COUNT(*) = 1) AS top_rows) AS n

and, when the subquery has no ``GROUP BY``, ``HAVING`` or aggregate, ``(SELECT amount FROM pay WHERE .. ORDER BY amount
DESC LIMIT 1)`` is ``(SELECT MAX(amount) FROM pay WHERE ..)``. "The largest value" is asked both ways, and the prover
knew nothing about a ``LIMIT`` inside a subquery of an expression ("unsupported: expression Subquery").

**Why it holds.** A subquery used as a value is NULL when it returns no row, and ``MAX`` over no rows is NULL. When
it returns rows, MySQL sorts NULL below every value: last under ``DESC``, first under ``ASC``. So the first row under
``DESC`` holds the largest value that is not NULL when there is one, and NULL when every row is NULL, which is what
``MAX`` returns. Under ``ASC`` the first row is NULL as soon as one row is NULL, while ``MIN`` skips it; the two agree
only when no row can be NULL, and the module reads ``ASC`` only for a column declared NOT NULL of the subquery's one
table.

Rows that tie for the first place must hold one and the same value, or which of them ``LIMIT 1`` keeps would matter
(two texts that a collation calls equal, ``0.0`` and ``-0.0``). So the output must have a declared whole-number or
``DATE`` type (``cast_rules.expression_type``), for which equal means identical.

Checked on MySQL 8.0.46 (``tests/test_top_row_subqueries.py`` lists the statements and replays them on SQLite, which
sorts NULL the same way): over tables with NULLs, ties and no rows, the first row under ``DESC`` equals ``MAX`` for a
plain, a grouped, a filtered and a correlated subquery; under ``ASC`` it differs from ``MIN`` when a row is NULL; an
unqualified ``ORDER BY`` name that is both the name of the output and a column of the table sorts by the output.

**What is read.** In MySQL, a subquery in the place of a value (not a derived table, not the operand of ``IN``,
``EXISTS``, ``ANY``, ``ALL`` or a set operation) that is a ``SELECT`` with a ``FROM``, exactly one output, ``LIMIT 1``
with no offset but zero, and an ``ORDER BY`` of one key that is its output: the same expression, an unqualified name
that is the output's name (MySQL looks an ``ORDER BY`` name up among the outputs first), or the ordinal ``1`` (which an
earlier pass has already written out). A key that is the output's expression and holds an unqualified name equal to
the output's alias (``-n AS n .. ORDER BY -n``) is not read: the server may read that name as the output. No window, no volatile call, no ``WITH``. A subquery with ``GROUP BY``,
``HAVING``, ``DISTINCT`` or an aggregate keeps them inside a derived table, and is read only when every column in it
resolves inside it (``closed_scalar_filters.closed``); a plain one may read columns of the query around it. Everything
else is left as written, including every other dialect (PostgreSQL sorts NULL above every value).

A pre-pass of the thirteenth hardest-cases batch (``shape_fallback.THIRTEENTH_BATCH``).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, _shape, is_aggregate
from .cast_rules import expression_type
from .clause_alias_reads import _volatile
from .closed_scalar_filters import closed
from .keyed_join_groups import declared_columns

DIALECTS = frozenset({"mysql"})

_REFUSED = ("qualify", "windows", "with_", "with", "laterals", "connect", "match", "into", "locks", "sample", "pivots", "cluster", "distribute", "sort", "kind", "hint", "operation_modifiers", "exclude", "prewhere", "settings", "format", "options", "for_")
_OPERAND_OF = (exp.From, exp.Join, exp.In, exp.Exists, exp.Any, exp.All, exp.SetOperation, exp.Subquery, exp.CTE, exp.Lateral, exp.Values, exp.Table, exp.With)
_ROUNDS = 16


def _number(node: exp.Expression | None, value: int) -> bool:
    return isinstance(node, exp.Literal) and not node.is_string and node.name == str(value)


def _first_row_only(select: exp.Select) -> bool:
    """``LIMIT 1`` with no offset, or with the offset zero."""

    limit, offset = select.args.get("limit"), select.args.get("offset")
    if type(limit) is not exp.Limit or not _number(limit.expression, 1) or any(v for key, v in limit.args.items() if key != "expression"):
        return False
    return offset is None or (type(offset) is exp.Offset and _number(offset.expression, 0) and not any(v for key, v in offset.args.items() if key != "expression"))


def _sorted_by_output(select: exp.Select) -> exp.Ordered | None:
    """The one ``ORDER BY`` key of ``select`` when it is the select's one output, else None."""

    order = select.args.get("order")
    if type(order) is not exp.Order or len(order.expressions) != 1 or any(v for key, v in order.args.items() if key != "expressions"):
        return None
    ordered = order.expressions[0]
    if type(ordered) is not exp.Ordered or any(v for key, v in ordered.args.items() if key not in ("this", "desc", "nulls_first")):
        return None
    item = select.expressions[0]
    key, output = ordered.this, item.unalias()
    if _shape(key) == _shape(output):
        name = (item.alias_or_name or "").lower()
        if isinstance(item, exp.Alias) and not isinstance(output, exp.Column) and any(not column.table and column.name.lower() == name for column in key.find_all(exp.Column)):
            return None  # ``-n AS n .. ORDER BY -n``: the name inside the key may be read as the output
        return ordered
    bare = isinstance(key, exp.Column) and isinstance(key.this, exp.Identifier) and not key.table and not key.args.get("db") and not key.args.get("catalog")
    if bare and (item.alias_or_name or "").lower() == key.name.lower() != "":
        return ordered  # an unqualified ORDER BY name is looked up among the outputs first
    return None


def _never_null(select: exp.Select, output: exp.Expression, not_null: dict | None) -> bool:
    """Whether ``output`` is a NOT NULL column of the one table ``select`` reads."""

    source = select.args.get(FROM_KEY)
    if source is None or select.args.get("joins") or type(source.this) is not exp.Table or not isinstance(output, exp.Column):
        return False
    table = source.this
    if table.args.get("db") or table.args.get("catalog") or not isinstance(output.this, exp.Identifier):
        return False
    if output.table and output.table.lower() != (table.alias_or_name or "").lower():
        return False
    return output.name.lower() in {c.lower() for c in (not_null or {}).get(table.name.lower(), ())}


def _plan(sub: exp.Subquery, declared: dict, types: dict, not_null: dict | None, dialect: str) -> exp.Select | None:
    """The select that ``sub`` is rewritten to hold, or None when it is not read."""

    if any(v for key, v in sub.args.items() if key != "this") or sub.parent is None or isinstance(sub.parent, _OPERAND_OF):
        return None
    select = sub.this
    if type(select) is not exp.Select or select.args.get(FROM_KEY) is None or any(select.args.get(key) for key in _REFUSED):
        return None
    if len(select.expressions) != 1 or not _first_row_only(select):
        return None
    item = select.expressions[0]
    output = item.unalias()
    if isinstance(output, exp.Star) or item.is_star or _volatile(select) or any(True for _ in select.find_all(exp.Window)):
        return None
    ordered = _sorted_by_output(select)
    if ordered is None:
        return None
    kind = expression_type(output, select, types, dialect=dialect)
    if kind is None or kind[0] not in ("int", "date"):
        return None
    if ordered.args.get("desc"):
        if ordered.args.get("nulls_first"):
            return None
        call = exp.Max
    else:
        if not _never_null(select, output, not_null):
            return None
        call = exp.Min
    grouped = bool(select.args.get("group") or select.args.get("having")) or any(
        is_aggregate(node) and node.find_ancestor(exp.Select) is select for node in select.find_all(exp.AggFunc, exp.Anonymous)
    )
    body = select.copy()
    for key in ("order", "limit", "offset"):
        body.set(key, None)
    if not grouped:
        body.set("distinct", None)  # the largest of the values is the largest of the distinct values
        body.set("expressions", [call(this=output.copy())])
        return body
    if not closed(body, declared):
        return None  # a column of the query around it, which a derived table cannot read
    if isinstance(item, exp.Alias) and isinstance(item.args.get("alias"), exp.Identifier):
        name = item.args["alias"].copy()  # HAVING may read the output by this name, so it is kept
    elif isinstance(output, exp.Column) and isinstance(output.this, exp.Identifier):
        name = output.this.copy()
    else:
        name = exp.to_identifier("top_value")
        body.set("expressions", [exp.alias_(output.copy(), "top_value")])
    return exp.Select(expressions=[call(this=exp.Column(this=name, table=exp.to_identifier("top_rows")))]).from_(exp.Subquery(this=body, alias=exp.TableAlias(this=exp.to_identifier("top_rows"))))


def write_top_rows(tree: exp.Expression, schema: dict | None, types: dict | None, not_null: dict | None, dialect: str) -> tuple[exp.Expression, bool]:
    """``tree`` with each first-row subquery written as ``MAX`` or ``MIN`` (module doc); whether anything changed."""

    if dialect not in DIALECTS or not schema or not types or not any(True for _ in tree.find_all(exp.Limit)):
        return tree, False
    declared = declared_columns(schema)
    if not any(_plan(sub, declared, types, not_null, dialect) is not None for sub in tree.find_all(exp.Subquery)):
        return tree, False
    result = tree.copy()  # the caller's tree is not written
    for _ in range(_ROUNDS):
        for sub in result.find_all(exp.Subquery):
            plan = _plan(sub, declared, types, not_null, dialect)
            if plan is not None:
                sub.set("this", plan)
                break
        else:
            break
    return result, True
