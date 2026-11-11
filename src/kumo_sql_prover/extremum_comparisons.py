"""A ``WHERE`` comparison with a scalar ``MIN`` or ``MAX`` subquery is an ``EXISTS``.

    SELECT * FROM e WHERE e.no > (SELECT MIN(d.no) FROM d WHERE d.name = e.job)
    SELECT * FROM e WHERE EXISTS (SELECT 1 FROM d WHERE d.name = e.job AND e.no > d.no)

Let V be the values ``d.no`` takes on the subquery's rows. ``MIN`` ignores NULLs and is NULL when V holds no
non-NULL value, so ``x > MIN`` is then NULL and the row is dropped; no row of the subquery makes ``x > y`` TRUE
either (a NULL ``y`` gives NULL), so ``EXISTS`` is FALSE and the row is dropped. Otherwise ``MIN`` is the least
non-NULL value ``m``, which some row attains. If ``x > m`` is TRUE that row makes ``x > y`` TRUE. If some row makes
``x > y`` TRUE, its ``y`` is not NULL and ``y >= m``, so ``x > m``. A NULL ``x`` is never TRUE in either form. The same
argument gives

    x >  (SELECT MIN(y) ..)   EXISTS (.. AND x >  y)        x <  (SELECT MAX(y) ..)   EXISTS (.. AND x <  y)
    x >= (SELECT MIN(y) ..)   EXISTS (.. AND x >= y)        x <= (SELECT MAX(y) ..)   EXISTS (.. AND x <= y)

and the mirrored spellings (``(SELECT MIN(y) ..) < x``). A ``WHERE`` conjunct keeps a row only when it is TRUE, so
the two forms keep the same rows, each as often as before (the subquery is evaluated per row and adds none).
This is how the prover already reads ``x > ANY (SELECT y ..)``, which LLM rewriters like to spell with ``MIN``.

Not equivalent, and left alone: ``x < MIN``, ``x > MAX`` (those are ``ALL``, TRUE or NULL-sensitive over an empty or
NULL-holding V), ``x = MIN``, ``x <> MIN``, and any use outside a top-level ``WHERE`` conjunct (in a select list
FALSE and NULL differ; under ``NOT`` or ``OR`` the conjunct is not required to be TRUE).

Conditions, all checked:

* the comparison is a top-level ``AND`` conjunct of the select's ``WHERE``;
* the subquery is one select with a ``FROM``, whose only output is ``MIN(y)`` or ``MAX(y)`` (``DISTINCT`` inside
  changes neither), with no ``GROUP BY``, ``HAVING``, ``DISTINCT``, ``QUALIFY``, window, ``ORDER BY``, ``LIMIT``,
  ``OFFSET``, ``FILTER`` or set operation. Its sources, joins and ``WHERE`` are kept; the comparison is added to
  the ``WHERE``, which is applied to the same joined rows the aggregate read;
* ``y`` holds no aggregate, window or subquery, and ``x`` none either;
* every column of ``x`` is qualified by a source of the outer select, and the subquery (at any depth) declares no
  source of that name: moved inside, ``x`` must still read the outer row. A bare column is qualified first, but
  only when the outer select reads exactly one table and the schema lists the column for it. A constant ``x``
  moves freely.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY

# comparison with the subquery on the right -> (aggregate that makes it an EXISTS, comparison kept)
_RIGHT = {exp.GT: exp.Min, exp.GTE: exp.Min, exp.LT: exp.Max, exp.LTE: exp.Max}
_MIRROR = {exp.GT: exp.LT, exp.GTE: exp.LTE, exp.LT: exp.GT, exp.LTE: exp.GTE}
_BODY_ARGS = frozenset({"expressions", FROM_KEY, "joins", "where"})
_NESTED = (exp.AggFunc, exp.Window, exp.Subquery, exp.Select, exp.Exists)


def _conjuncts(node: exp.Expression) -> list[exp.Expression]:
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _unparen(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _extremum(node: exp.Expression) -> tuple[exp.Select, type, exp.Expression] | None:
    """``(body, MIN or MAX, y)`` for a scalar subquery that only computes ``MIN(y)`` or ``MAX(y)``."""

    node = _unparen(node)
    if not isinstance(node, exp.Subquery) or any(v for k, v in node.args.items() if k != "this"):
        return None
    body = node.this
    if not isinstance(body, exp.Select) or any(v for k, v in body.args.items() if k not in _BODY_ARGS):
        return None
    if body.args.get(FROM_KEY) is None or len(body.expressions) != 1:
        return None
    call = body.expressions[0]
    call = call.this if isinstance(call, exp.Alias) else call
    if type(call) not in (exp.Min, exp.Max) or any(v for k, v in call.args.items() if k != "this"):
        return None
    value = call.this
    if isinstance(value, exp.Distinct):
        if len(value.expressions) != 1 or value.args.get("on"):
            return None
        value = value.expressions[0]
    if value is None or isinstance(value, exp.Star) or value.find(*_NESTED, exp.Star):
        return None
    if any(isinstance(n, exp.Window) for n in body.walk()):
        return None
    return body, type(call), value


def _sources(select: exp.Select) -> list[exp.Expression]:
    from_ = select.args.get(FROM_KEY)
    return ([from_.this] if from_ is not None else []) + [join.this for join in select.args.get("joins") or []]


def _lone_table(select: exp.Select, schema: dict | None) -> tuple[str, set[str]] | None:
    """``(alias, columns)`` when ``select`` reads exactly one table whose columns ``schema`` lists."""

    sources = _sources(select)
    if len(sources) != 1 or not isinstance(sources[0], exp.Table) or not isinstance(sources[0].this, exp.Identifier):
        return None
    alias = sources[0].args.get("alias")
    if isinstance(alias, exp.TableAlias) and alias.args.get("columns"):
        return None
    lowered = {str(k).lower(): v for k, v in (schema or {}).items()}
    columns = lowered.get(".".join(part.name for part in sources[0].parts).lower())
    if not columns or not sources[0].alias_or_name:
        return None
    return sources[0].alias_or_name, {str(c).lower() for c in columns}


def _as_exists(conjunct: exp.Expression, outer: set[str], lone: tuple[str, set[str]] | None) -> exp.Expression | None:
    node = _unparen(conjunct)
    if type(node) not in _RIGHT:
        return None
    for value, subquery, kind in ((node.this, node.expression, type(node)), (node.expression, node.this, _MIRROR[type(node)])):
        found = _extremum(subquery)
        if found is None or found[1] is not _RIGHT[kind]:
            continue
        body, _, inner_value = found
        if value.find(*_NESTED):
            return None
        inside = {
            (n.alias_or_name or "").lower() for n in body.walk() if isinstance(n, (exp.Table, exp.Subquery, exp.CTE, exp.Unnest, exp.Lateral))
        }
        value = exp.Paren(this=value.copy())
        for column in value.find_all(exp.Column):
            if not column.table and lone is not None and column.name.lower() in lone[1] and isinstance(column.this, exp.Identifier):
                column.set("table", exp.to_identifier(lone[0]))  # the outer select's only table has this column
            if not column.table or column.table.lower() not in outer or column.table.lower() in inside:
                return None
        value = value.this
        test = kind(this=value.copy(), expression=exp.Paren(this=inner_value.copy()) if isinstance(inner_value, exp.Binary) else inner_value.copy())
        if isinstance(test.this, exp.Binary):
            test.set("this", exp.Paren(this=test.this))
        result = body.copy()
        result.set("expressions", [exp.Literal.number(1)])
        where = result.args.get("where")
        if where is None:
            result.set("where", exp.Where(this=test))
        else:
            kept = where.this
            kept = exp.Paren(this=kept) if isinstance(kept, (exp.Or, exp.Xor)) else kept
            result.set("where", exp.Where(this=exp.And(this=kept, expression=test)))
        return exp.Exists(this=result)
    return None


def compare_with_extremum_as_exists(select: exp.Select, schema: dict | None = None) -> exp.Select | None:
    """``select`` with each ``WHERE`` conjunct ``x > (SELECT MIN(y) ..)`` (module doc) written as ``EXISTS``, or None."""

    where = select.args.get("where")
    if where is None or not where.find(exp.Subquery):
        return None
    outer = {(source.alias_or_name or "").lower() for source in _sources(select)} - {""}
    lone = _lone_table(select, schema)
    parts, changed = [], False
    for conjunct in _conjuncts(where.this):
        replacement = _as_exists(conjunct, outer, lone)
        changed = changed or replacement is not None
        parts.append(replacement if replacement is not None else conjunct.copy())
    if not changed:
        return None
    parts = [exp.Paren(this=p) if isinstance(p, (exp.Or, exp.Xor)) else p for p in parts]
    condition = parts[0]
    for part in parts[1:]:
        condition = exp.And(this=condition, expression=part)
    copy = select.copy()
    copy.set("where", exp.Where(this=condition))
    return copy
