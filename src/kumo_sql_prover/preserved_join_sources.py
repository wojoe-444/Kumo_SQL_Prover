"""An inner join written as a derived table on the preserved side of ``LEFT JOIN``s is read as the join itself.

    SELECT d.title, COUNT(c.id)
    FROM (SELECT p.id, p.title, u.name FROM p, u WHERE p.owner = u.id AND p.kind = 1) AS d
    LEFT JOIN c ON d.id = c.post
    GROUP BY d.id, d.title, d.name

is

    SELECT p.title, COUNT(c.id)
    FROM p CROSS JOIN u LEFT JOIN c ON p.id = c.post
    WHERE p.owner = u.id AND p.kind = 1
    GROUP BY p.id, p.title, u.name

Calcite prints every join input as a derived table, so the rewrites of R-Bot, LLM-R2 and LearnedRewrite have this
shape wherever the query joins before an outer join. A plain select of inner joins is flattened by the prover
itself; under an outer join it was not, and an aggregate or a window over the join was compared as another relation.

Why it holds. Write ``X`` for the rows of the derived table's sources joined by its inner joins, ``w`` for its
``WHERE``, and ``J`` for the joins of the select that reads it. The derived table has one row per row of
``sigma_w(X)`` and each output is an expression over that row. Every join of ``J`` is an inner or a ``LEFT`` join
with the derived table on its left, so no join pads the derived table's columns with NULL, and ``J`` extends each
left row independently of the other left rows: it yields the matching right rows, or (``LEFT``) one padded row.
So the rows of ``J(sigma_w(X))`` are the rows of ``J(X)`` whose ``X`` part satisfies ``w``, which is ``sigma_w(J(X))``
since ``w`` reads ``X`` columns only and those are not padded. TRUE, FALSE and NULL of ``w`` fall the same way on
both sides, duplicates of a row of ``X`` are kept or dropped together, and with no rows nothing changes. Every
output ``d.x`` is replaced by the expression that computes it, on the same row, so the select's ``ON``, ``WHERE``,
``GROUP BY``, ``HAVING``, windows, ``ORDER BY`` and ``LIMIT`` see the same values on the same bag of rows.

The comma joins of the derived table are written ``CROSS JOIN``, so that the printed text is the left-deep join
in every dialect (a comma binds looser than ``JOIN``).

Conditions checked (None otherwise):

* the derived table is the select's ``FROM`` item, aliased, without a column list, ``LATERAL`` or sample;
* the select has at least one join, and each is a plain inner, ``CROSS`` or ``LEFT`` join (no ``RIGHT``, ``FULL``,
  ``SEMI``, ``ANTI``, ``NATURAL``, ``ASOF``, ``USING``, lateral or unnest source);
* the derived table's body is one ``SELECT`` over two or more sources (a filter on one table is handled elsewhere)
  joined by commas, ``CROSS`` or inner ``JOIN .. ON`` only, with nothing but a select list, ``FROM``, joins and
  ``WHERE``: no ``DISTINCT``, grouping, ``HAVING``, window, ``QUALIFY``, ``ORDER BY``, ``LIMIT``, ``OFFSET`` or ``WITH``;
* its outputs have distinct names and are expressions without an aggregate, window, subquery, star, or a function
  that can differ between two evaluations or return several rows;
* its sources are aliased tables or derived tables, every column in the body names one of them by its alias, and
  none of those aliases is used by another source of the select;
* neither the select nor the body has a subquery outside its sources (its columns could be captured by the newly
  visible names), the select has no star other than ``COUNT(*)``, and every column of the select names one of the
  select's sources by its alias, the ones reading the derived table naming one of its outputs.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import select_sources

_BODY_ARGS = frozenset({"expressions", "from_", "from", "joins", "where", "kind"})
_UNSTABLE = (exp.Rand, exp.Randn, exp.Uuid, exp.CurrentTimestamp, exp.CurrentDate, exp.CurrentTime, exp.CurrentDatetime, exp.Anonymous, exp.UDTF, exp.Unnest, exp.Explode)
_NESTED = (exp.Query, exp.Subquery, exp.Exists)


def _from_key(select: exp.Select) -> str:
    return "from_" if select.args.get("from_") is not None else "from"


def _plain_join(join: exp.Join, allow_left: bool) -> bool:
    side = (join.args.get("side") or "").upper()
    kind = (join.args.get("kind") or "").upper()
    if any(value for key, value in join.args.items() if key not in ("this", "on", "side", "kind")):
        return False  # USING, NATURAL, ASOF, GLOBAL, a match condition, a pivot
    if side not in ("", "LEFT") or (side == "LEFT" and not allow_left):
        return False
    if kind not in ("", "INNER", "CROSS", "OUTER") or (kind == "OUTER" and side != "LEFT"):
        return False
    if side == "LEFT" and join.args.get("on") is None:
        return False
    return isinstance(join.this, (exp.Table, exp.Subquery))


def _plain_source(source: exp.Expression) -> bool:
    if isinstance(source, exp.Table):
        return bool(source.alias_or_name) and not any(source.args.get(key) for key in ("joins", "pivots", "laterals", "sample", "when", "version"))
    if isinstance(source, exp.Subquery):
        alias = source.args.get("alias")
        return bool(source.alias) and not alias.args.get("columns") and not any(value for key, value in source.args.items() if key not in ("this", "alias"))
    return False


def _nested_outside_sources(select: exp.Select, sources: list[exp.Expression]) -> bool:
    """Whether ``select`` holds a subquery that is not one of its ``FROM`` / ``JOIN`` items."""

    for node in select.walk():
        if node is select or not isinstance(node, _NESTED):
            continue
        if not any(node is source or _inside(node, source) for source in sources):
            return True
    return False


def _inside(node: exp.Expression, root: exp.Expression) -> bool:
    parent = node.parent
    while parent is not None:
        if parent is root:
            return True
        parent = parent.parent
    return False


def merge_preserved_join_source(select: exp.Select) -> exp.Expression | None:
    """``select`` reading the inner joins of its derived ``FROM`` item directly, or None (see the module docstring)."""

    joins = select.args.get("joins") or []
    from_ = select.args.get(_from_key(select))
    if from_ is None or not joins:
        return None
    source = from_.this
    if not isinstance(source, exp.Subquery) or not _plain_source(source) or not isinstance(source.this, exp.Select):
        return None
    if not all(_plain_join(join, allow_left=True) for join in joins) or not any((j.args.get("side") or "").upper() == "LEFT" for j in joins):
        return None
    sources = select_sources(select)
    if not all(_plain_source(s) for s in sources):
        return None
    body = source.this
    if any(value for key, value in body.args.items() if key not in _BODY_ARGS):
        return None
    body_joins = body.args.get("joins") or []
    body_sources = select_sources(body)
    if len(body_sources) < 2 or not all(_plain_source(s) for s in body_sources) or not all(_plain_join(j, allow_left=False) for j in body_joins):
        return None
    if _nested_outside_sources(body, body_sources) or _nested_outside_sources(select, sources):
        return None
    alias = source.alias.lower()
    body_aliases = [s.alias_or_name.lower() for s in body_sources]
    other_aliases = [s.alias_or_name.lower() for s in sources if s is not source]
    if len(set(body_aliases)) != len(body_aliases) or len(set(other_aliases)) != len(other_aliases) or alias in other_aliases:
        return None
    if set(body_aliases) & set(other_aliases):
        return None
    # the body: every column names one of its own sources, every output is a plain expression under its own name
    outputs: dict[str, exp.Expression] = {}
    for item in body.expressions:
        value = item.this if isinstance(item, exp.Alias) else item
        name = (item.alias_or_name or "").lower()
        if not name or name in outputs or isinstance(value, exp.Star):
            return None
        if any(isinstance(n, (exp.AggFunc, exp.Window, exp.Star, *_NESTED, *_UNSTABLE)) for n in value.walk()):
            return None
        outputs[name] = value
    for column in body.find_all(exp.Column):
        if any(_inside(column, s) for s in body_sources if isinstance(s, exp.Subquery)):
            continue
        if not isinstance(column.this, exp.Identifier) or column.args.get("db") or column.table.lower() not in body_aliases:
            return None
    # the select: every column names one of its sources; a read of the derived table names one of its outputs
    visible = set(other_aliases) | {alias}
    reads: list[exp.Column] = []
    for node in select.walk():
        if isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count):
            return None
        if not isinstance(node, exp.Column) or any(_inside(node, s) for s in sources if isinstance(s, exp.Subquery)):
            continue
        if not isinstance(node.this, exp.Identifier) or node.args.get("db") or node.table.lower() not in visible:
            return None
        if node.table.lower() == alias:
            if node.name.lower() not in outputs:
                return None
            reads.append(node)
    copy = select.copy()
    new_source = copy.args[_from_key(copy)].this
    new_body = new_source.this
    by_position = {id(old): new for old, new in zip(select.walk(), copy.walk())}
    for column in reads:
        target = by_position[id(column)]
        value = outputs[column.name.lower()].copy()
        named = target.parent is copy and target.arg_key == "expressions"  # a bare select item keeps its output name
        if not isinstance(value, (exp.Column, exp.Literal, exp.Paren, exp.Func)) or isinstance(value, (exp.Binary, exp.Unary, exp.Case)):
            value = exp.Paren(this=value)
        if named and not (isinstance(value, exp.Column) and value.name.lower() == column.name.lower()):
            value = exp.alias_(value.unnest() if isinstance(value, exp.Paren) else value, column.name, quoted=bool(column.this.args.get("quoted")))
        target.replace(value)
    lifted = [new_body.args[_from_key(new_body)].this] + list(new_body.args.get("joins") or [])
    new_joins: list[exp.Join] = []
    for join in lifted[1:]:
        join = join.copy()
        if join.args.get("on") is None and not join.args.get("kind"):
            join.set("kind", "CROSS")
        new_joins.append(join)
    copy.set(_from_key(copy), exp.From(this=lifted[0].copy()))
    copy.set("joins", new_joins + [j.copy() for j in copy.args.get("joins") or []])
    body_where = new_body.args.get("where")
    if body_where is not None:
        mine = copy.args.get("where")
        parts = [body_where.this.copy()] + ([mine.this.copy()] if mine is not None else [])
        parts = [exp.Paren(this=p) if isinstance(p, (exp.Or, exp.Xor)) else p for p in parts]
        condition = parts[0] if len(parts) == 1 else exp.And(this=parts[0], expression=parts[1])
        copy.set("where", exp.Where(this=condition))
    return copy
