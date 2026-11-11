"""Filters around a grouped derived table that is joined to other sources.

A grouped (or ``DISTINCT``) derived table is one relation the SMT prover keeps whole, so a filter has to
sit on the same side of it in both queries of a pair before the two relations can be matched. An optimizer
moves such filters freely, and decorrelated TPC-DS and DSB queries differ from their rewrites in exactly
this. Two rules, both for a select whose sources are joined by inner or cross joins only:

``push_filter_into_grouped_source``
    ``FROM (SELECT k, SUM(x) AS s FROM t GROUP BY k) AS d JOIN u ON .. WHERE d.k BETWEEN 1 AND 5`` filters
    ``t`` instead: ``.. FROM (SELECT k, SUM(x) AS s FROM t WHERE k BETWEEN 1 AND 5 GROUP BY k) AS d JOIN u ON ..``.
    A condition on values the derived table groups by keeps or drops whole groups, so it selects the same
    rows before the grouping. ``algebraic_equivalence._push_filter_into_derived`` does this for a select
    that reads the derived table alone; this is the same step when other sources are joined to it.

``drop_rejected_null_filter``
    ``FROM (SELECT k, SUM(x) AS s FROM t WHERE k IS NOT NULL GROUP BY k) AS d JOIN u ON d.k = u.k`` does not
    need its ``k IS NOT NULL``: the rows it drops form the group whose key is NULL, and the join keeps no
    row of that group. The same holds one level further in, when the derived table groups a second derived
    table by the same value, which is where an optimizer leaves the inferred filter in a decorrelated
    ``> (SELECT AVG(..) FROM (grouped) ..)``.

A condition the derived table already filters on (by its text, a ``BETWEEN`` also read as its two comparisons)
is true of every row it returns, so it is dropped outside and not added inside a second time. An optimizer's
transitive-predicate rule leaves such copies above the join.

Both rules read only a derived table that computes an aggregate (``SUM``, ``AVG``, ..): a ``GROUP BY`` or
``DISTINCT`` with none is a duplicate removal that other rules already read.

Neither rule fires through an outer join, a ``LIMIT``, a window, ``DISTINCT ON`` or an extended grouping,
and a condition that is moved holds nothing but columns, literals, comparisons, ``BETWEEN``, ``IN`` lists,
``IS [NOT] NULL`` and ``AND``/``OR``/``NOT``: nothing that can fail or that is evaluated a different number
of times inside the derived table.
"""

from __future__ import annotations

from sqlglot import exp

_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
_PLAIN = (
    exp.Column, exp.Identifier, exp.Literal, exp.Boolean, exp.Null, exp.Paren, exp.Not, exp.And, exp.Or,
    exp.Between, exp.In, exp.Is, exp.Neg, *_COMPARISONS,
)


def _from(select: exp.Select):
    return select.args.get("from_") or select.args.get("from")


def _inner_sources(select: exp.Select) -> dict[str, exp.Expression] | None:
    """Alias -> source for a select whose sources are tables and derived tables joined by inner or cross joins,
    each under its own alias. ``None`` for anything else."""

    from_ = _from(select)
    if from_ is None:
        return None
    sources = [from_.this]
    for join in select.args.get("joins") or []:
        if join.side or (join.kind or "").upper() not in ("", "INNER", "CROSS"):
            return None
        if any(join.args.get(k) for k in ("method", "using", "match_condition", "global_", "hint", "directed", "expressions", "pivots")):
            return None
        sources.append(join.this)
    out: dict[str, exp.Expression] = {}
    for source in sources:
        if not isinstance(source, (exp.Table, exp.Subquery)) or source.args.get("pivots") or source.args.get("sample"):
            return None
        alias = source.alias_or_name.lower()
        if not alias or alias in out:
            return None
        out[alias] = source
    return out


def _conjuncts(node: exp.Expression) -> list[exp.Expression]:
    from .algebraic_equivalence import _conjuncts as conjuncts

    return conjuncts(node)


def _and_all(parts: list[exp.Expression]) -> exp.Expression:
    from .algebraic_equivalence import _and_all as and_all  # parenthesizes an OR: sqlglot prints the tree as it is

    return and_all(parts)


def _aggregates(body: exp.Select) -> bool:
    """Whether ``body`` computes an aggregate of its own. A ``GROUP BY`` or ``DISTINCT`` with none is a duplicate
    removal, which the semi-join and ``DISTINCT`` rules read in their own way: moving a filter into it first took a
    proof away (QED ``testJoinPushTransitivePredicatesRule2``), so both rules leave it alone."""

    return any(node.find_ancestor(exp.Select) is body for node in body.find_all(exp.AggFunc))


def _text(condition: exp.Expression) -> str:
    """``condition`` as text that does not depend on quoting, letter case of names or outer parentheses."""

    copy = exp.Paren(this=condition.copy())
    for identifier in copy.find_all(exp.Identifier):
        identifier.set("quoted", False)
        identifier.set("this", identifier.name.lower())
    inner = copy.this
    while isinstance(inner, exp.Paren):
        inner = inner.this
    return inner.sql()


def _held(body: exp.Select) -> set[str]:
    """The conditions ``body`` already filters its rows by, a ``BETWEEN`` also as its two comparisons."""

    where = body.args.get("where")
    found: set[str] = set()
    for part in _conjuncts(where.this) if where is not None else []:
        part = part.unnest() if isinstance(part, exp.Paren) else part
        found.add(_text(part))
        if isinstance(part, exp.Between) and not part.args.get("symmetric"):
            found.add(_text(exp.GTE(this=part.this.copy(), expression=part.args["low"].copy())))
            found.add(_text(exp.LTE(this=part.this.copy(), expression=part.args["high"].copy())))
    return found


def _holders(select: exp.Select) -> list[tuple[exp.Expression, str]]:
    """The nodes of ``select`` that hold a condition every result row satisfies: its WHERE and each join's ON."""

    holders = []
    where = select.args.get("where")
    if where is not None:
        holders.append((where, "this"))
    for join in select.args.get("joins") or []:
        if join.args.get("on") is not None:
            holders.append((join, "on"))
    return holders


def push_filter_into_grouped_source(select: exp.Select) -> exp.Expression | None:
    from .algebraic_equivalence import _derived_key_outputs

    if not select.args.get("joins"):
        return None  # the single-source case is ``_push_filter_into_derived``
    sources = _inner_sources(select)
    if sources is None:
        return None
    keyed = {}
    for alias, source in sources.items():
        if isinstance(source, exp.Subquery) and isinstance(source.this, exp.Select) and _from(source.this) is not None and _aggregates(source.this):
            outputs = _derived_key_outputs(source.this)
            if outputs:
                keyed[alias] = outputs

    def target(part: exp.Expression) -> str | None:
        columns = list(part.find_all(exp.Column))
        if not columns or not all(isinstance(node, _PLAIN) for node in part.walk()):
            return None
        tables = {c.table.lower() for c in columns}
        if len(tables) != 1:
            return None
        (alias,) = tables
        if alias not in keyed or not all(c.name.lower() in keyed[alias] for c in columns):
            return None
        return alias

    held = {alias: _held(sources[alias].this) for alias in keyed}

    def moved_form(part: exp.Expression, alias: str) -> exp.Expression:
        wrapper = exp.Paren(this=part.copy())
        for column in list(wrapper.find_all(exp.Column)):
            value = keyed[alias][column.name.lower()].copy()
            column.replace(value if isinstance(value, (exp.Column, exp.Literal, exp.Paren)) else exp.Paren(this=value))
        return wrapper.this

    if not keyed or not any(target(part) for holder, key in _holders(select) for part in _conjuncts(holder.args[key])):
        return None
    copy = select.copy()
    moved: dict[str, list[exp.Expression]] = {}
    for holder, key in _holders(copy):
        kept = []
        for part in _conjuncts(holder.args[key]):
            alias = target(part)
            if alias is None:
                kept.append(part)
                continue
            inside = moved_form(part, alias)
            if _text(inside) not in held[alias]:  # a condition the derived table already filters on is only dropped
                held[alias].add(_text(inside))
                moved.setdefault(alias, []).append(inside)
        if isinstance(holder, exp.Where):
            if kept:
                holder.set("this", _and_all(kept))
            else:
                copy.set("where", None)
        elif kept:
            holder.set("on", _and_all(kept))
        else:  # a join left with no condition is a cross join
            holder.set("on", None)
            holder.set("kind", "CROSS")
    for alias, parts in moved.items():
        inner = _inner_sources(copy)[alias].this
        existing = inner.args.get("where")
        inner.set("where", exp.Where(this=_and_all(([existing.this] if existing is not None else []) + parts)))
    return copy


def _is_not_null(part: exp.Expression) -> exp.Expression | None:
    """The operand of ``x IS NOT NULL`` (also written ``NOT x IS NULL``)."""

    part = part.unnest() if isinstance(part, exp.Paren) else part
    if isinstance(part, exp.Not):
        inner = part.this.unnest() if isinstance(part.this, exp.Paren) else part.this
        if isinstance(inner, exp.Is) and isinstance(inner.expression, exp.Null):
            return inner.this
    return None


def _rejected(select: exp.Select, sources: dict[str, exp.Expression]) -> set[tuple[str, str]]:
    """``(alias, column)`` of the source columns no result row of ``select`` has as NULL: an operand of a
    comparison, or the operand of ``IS NOT NULL``, in a condition every row satisfies."""

    found: set[tuple[str, str]] = set()
    for holder, key in _holders(select):
        for part in _conjuncts(holder.args[key]):
            operands = [part.this, part.expression] if isinstance(part, _COMPARISONS) else [_is_not_null(part)]
            for operand in operands:
                operand = operand.unnest() if isinstance(operand, exp.Paren) else operand
                if isinstance(operand, exp.Column) and operand.table and operand.table.lower() in sources:
                    found.add((operand.table.lower(), operand.name.lower()))
    return found


def _plain_outputs(body: exp.Select) -> dict[str, exp.Expression] | None:
    """Output name -> value of a select that only filters and projects its rows."""

    if any(body.args.get(k) for k in ("group", "having", "distinct", "limit", "offset", "qualify", "windows", "with_", "with", "order")):
        return None
    if any(isinstance(e, exp.Star) or (isinstance(e, exp.Column) and isinstance(e.this, exp.Star)) for e in body.expressions):
        return None
    if any(isinstance(n, (exp.AggFunc, exp.Window, exp.Subquery)) for item in body.expressions for n in item.walk()):
        return None
    outputs = {}
    for item in body.expressions:
        name = item.alias_or_name.lower()
        if not name or name in outputs:
            return None
        outputs[name] = item.this if isinstance(item, exp.Alias) else item
    return outputs


def _strip(body: exp.Expression, name: str, depth: int = 0) -> bool:
    """Remove from ``body`` (in place) the filters that only drop rows whose output ``name`` is NULL."""

    from .algebraic_equivalence import _derived_key_outputs

    if not isinstance(body, exp.Select) or depth > 4:
        return False
    outputs = _derived_key_outputs(body) or _plain_outputs(body)
    if not outputs or name not in outputs:
        return False
    value = outputs[name]
    value = value.unnest() if isinstance(value, exp.Paren) else value
    changed = False
    where = body.args.get("where")
    if where is not None:
        text = value.sql()
        kept = [part for part in _conjuncts(where.this) if (operand := _is_not_null(part)) is None or operand.sql() != text]
        if len(kept) != len(_conjuncts(where.this)):
            body.set("where", exp.Where(this=_and_all(kept)) if kept else None)
            changed = True
    # The rows of a source whose column is this value and NULL reach only output rows where the value is NULL.
    sources = _inner_sources(body)
    if sources is not None and isinstance(value, exp.Column):
        alias = value.table.lower() if value.table else (next(iter(sources)) if len(sources) == 1 else None)
        source = sources.get(alias) if alias else None
        if isinstance(source, exp.Subquery):
            changed = _strip(source.this, value.name.lower(), depth + 1) or changed
    return changed


def drop_rejected_null_filter(select: exp.Select) -> exp.Expression | None:
    sources = _inner_sources(select)
    if sources is None or not any(isinstance(s, exp.Subquery) for s in sources.values()):
        return None
    targets = [
        (alias, name)
        for alias, name in sorted(_rejected(select, sources))
        if isinstance(sources[alias], exp.Subquery) and isinstance(sources[alias].this, exp.Select) and _aggregates(sources[alias].this)
    ]
    if not targets:
        return None
    copy = select.copy()
    copied = _inner_sources(copy)
    changed = False
    for alias, name in targets:
        changed = _strip(copied[alias].this, name) or changed
    return copy if changed else None
