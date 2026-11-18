"""A second copy of the outer row inside an ``EXISTS`` is the outer row itself.

    SELECT p.patron_id, p.patron_name FROM patron AS p
    WHERE EXISTS (SELECT 1 FROM patron AS k, loan AS o
                  WHERE k.patron_id = o.patron_id AND o.title = 'B'
                    AND k.patron_id = p.patron_id AND k.patron_name = p.patron_name)

asks whether some row ``k`` of ``patron`` that equals the outer row ``p`` on two columns has a loan of ``B``, and
reads nothing of ``k`` but those two columns. It is

    SELECT p.patron_id, p.patron_name FROM patron AS p
    WHERE EXISTS (SELECT 1 FROM loan AS o
                  WHERE p.patron_id = o.patron_id AND o.title = 'B' AND p.patron_name IS NOT NULL)

**Why it holds.** Let the subquery read ``k`` only through columns ``c1 .. cn`` (n >= 1), each tied to the outer row
by a top-level conjunct ``k.ci = p.ci``, and write ``F(k.c1 .. k.cn, rest)`` for its other conditions.

* If a row ``k`` and rows of the other sources pass, every ``k.ci = p.ci`` is TRUE, so each ``p.ci`` is not NULL and
  holds the value ``F`` read from ``k``: the same other rows pass ``F(p.c1 .. p.cn, rest)``.
* If every ``p.ci`` is not NULL and other rows pass ``F(p.c1 .. p.cn, rest)``, then ``p`` is a real row of the table
  (a row padded by an outer join has only NULLs), so ``k`` can be that very row: ``k.ci = p.ci`` is TRUE for a
  value that is not NULL, and ``F`` reads the same values.

So the test is ``EXISTS (rest WHERE F(p.c1 .. p.cn, rest) AND p.c1 IS NOT NULL AND ..)``, with no key and no other
declared fact, and ``NOT EXISTS`` follows. An ``IS NOT NULL`` is left out when the column is declared NOT NULL and
``p`` is not on the padded side of an outer join, or when another top-level conjunct compares the column with
something (a comparison with NULL is never TRUE).

The copy appears in the prover's own readings: with a key and every column of a table declared NOT NULL, a join to
a table of counts filtered on those counts is read as ``EXISTS`` tests around a copy of the row, and then no longer
met the plain spelling (``id IN (..) AND id NOT IN (..)``) that it met with fewer declared facts.

**What is read.** An ``EXISTS`` anywhere in the ``WHERE`` of a select, whose subquery is a plain select (no
grouping, aggregate, window, ``DISTINCT``, tail or set operation) over two or more sources joined by commas or
inner joins; one of them is a base table that an outer source of the select also reads, under another name. Every
column in the subquery, at any depth, names its source; the copy is read only through plain columns that a
top-level equality ties to the same column of the outer row; no nested query names a source like the copy or
like the outer row. ``nested_correlation.interchangeable`` must let the column be read for itself (not a
floating-point one). A volatile call leaves the select as it is.

A narrow retry rule (``shape_fallback``). ``tests/test_own_row_copies.py`` replays the rewrite on DuckDB.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, conjuncts
from .exists_constant_rules import _real_sources
from .fk_rules import _same_table
from .nested_correlation import interchangeable
from .own_group_membership import _facts, _plain_table, _unparen

_BLOCKING = ("group", "having", "qualify", "limit", "offset", "laterals", "with_", "with", "order", "windows", "distinct", "connect", "match", "into", "locks", "sample", "pivots")
_JOIN_EXTRAS = ("method", "using", "match_condition", "global_", "hint", "directed", "expressions", "pivots")
_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)


def _from(select: exp.Select) -> exp.Expression | None:
    from_ = select.args.get(FROM_KEY)
    return from_.this if from_ is not None else None


def _named(column: exp.Expression | None) -> tuple[str, str] | None:
    """``(source, column)`` in lower case for a plain column that names its source."""

    column = _unparen(column)
    if not isinstance(column, exp.Column) or not isinstance(column.this, exp.Identifier) or column.this.args.get("quoted"):
        return None
    if not column.table or column.args.get("db") or column.args.get("catalog"):
        return None
    return column.table.lower(), column.name.lower()


def _base(source: exp.Expression | None) -> bool:
    return source is not None and _plain_table(source) and isinstance(source.this, exp.Identifier) and not source.this.args.get("quoted")


def _reading(body: exp.Select, outer: dict[str, exp.Table], real: set[str], not_null: dict | None, types: dict | None) -> exp.Select | None:
    """``body`` (a copy the caller owns) without a copy of one outer row, or None."""

    if any(body.args.get(key) for key in _BLOCKING):
        return None
    joins = body.args.get("joins") or []
    first = _from(body)
    if first is None or not joins:
        return None
    for join in joins:
        if join.side or (join.kind or "").upper() not in ("", "INNER", "CROSS") or any(join.args.get(key) for key in _JOIN_EXTRAS):
            return None
    sources = [first] + [join.this for join in joins]
    names = [(source.alias_or_name or "").lower() for source in sources]
    if any(not name for name in names) or len(set(names)) != len(names) or any(not isinstance(s, (exp.Table, exp.Subquery)) for s in sources):
        return None
    if any(isinstance(node, (exp.AggFunc, exp.Window)) and node.find_ancestor(exp.Select) is body for node in body.find_all(exp.AggFunc, exp.Window)):
        return None
    columns = list(body.find_all(exp.Column))
    if any(isinstance(column.this, exp.Star) or not column.table for column in columns):
        return None  # a bare name could be the copy's
    parts = [part for join in joins if join.args.get("on") is not None for part in conjuncts(join.args["on"])]
    if body.args.get("where") is not None:
        parts += conjuncts(body.args["where"].this)
    declared_elsewhere = {
        (node.alias_or_name or "").lower() for node in body.find_all(exp.Table, exp.Subquery) if not any(node is source for source in sources) and (node.alias_or_name or "")
    }
    for position, copy in enumerate(sources):
        mine = names[position]
        if not _base(copy) or mine in outer or mine in declared_elsewhere:
            continue
        for theirs, row in outer.items():
            if theirs in names or theirs in declared_elsewhere or not _same_table(copy.name, row.name):
                continue
            if any(copy.args.get(key) or row.args.get(key) for key in ("db", "catalog")):
                continue
            ties, kept = set(), []
            for part in parts:
                test = _unparen(part)
                if isinstance(test, exp.EQ):
                    ends = {_named(test.this), _named(test.expression)}
                    tied = next((name for name in (e[1] for e in ends if e is not None) if ends == {(mine, name), (theirs, name)}), None)
                    if tied is not None:
                        ties.add(tied)
                        continue
                kept.append(part)
            if not ties:
                continue
            reads = [column for column in columns if column.table.lower() == mine]
            tying = {id(node) for part in parts if not any(part is k for k in kept) for node in part.find_all(exp.Column)}
            others = [column for column in reads if id(column) not in tying]
            if any(_named(column) is None or column.name.lower() not in ties or any(column is part for part in parts) for column in others):
                continue  # the copy is read for something the outer row is not tied on
            table = row.name.lower()
            if not all(interchangeable(types, (table, name), (table, name)) for name in ties):
                continue
            for column in others:
                column.replace(exp.column(column.name, table=row.alias_or_name))
            never_null = {str(c).lower() for c in (_facts(not_null, row.name) or ())}
            compared = set()
            for part in kept:
                test = _unparen(part)
                if isinstance(test, _COMPARISONS):
                    compared |= {end[1] for end in (_named(test.this), _named(test.expression)) if end is not None and end[0] == theirs}
            guards = [
                exp.Not(this=exp.Is(this=exp.column(name, table=row.alias_or_name), expression=exp.Null()))
                for name in sorted(ties)
                if name not in compared and not (theirs in real and name in never_null)
            ]
            rest = [source for index, source in enumerate(sources) if index != position]
            for source in rest:
                source.pop()
            body.set(FROM_KEY, exp.From(this=rest[0]))
            body.set("joins", [exp.Join(this=source) for source in rest[1:]] or None)
            tests = [part.copy() for part in kept] + guards
            body.set("where", exp.Where(this=exp.and_(*tests, copy=False)) if tests else None)
            return body
    return None


def drop_own_row_copies(select: exp.Select, not_null: dict | None, types: dict | None = None) -> exp.Select | None:
    """``select`` with each ``EXISTS`` of its ``WHERE`` that holds a copy of an outer row read without the copy."""

    from .counted_membership import _unstable

    where = select.args.get("where")
    if where is None or not any(test.find_ancestor(exp.Select) is select for test in where.find_all(exp.Exists)):
        return None
    result = select.copy()
    sources = [_from(result)] + [join.this for join in result.args.get("joins") or []]
    if any(source is None or not (source.alias_or_name or "") for source in sources):
        return None
    names = [source.alias_or_name.lower() for source in sources]
    outer = {name: source for name, source in zip(names, sources) if _base(source) and names.count(name) == 1}
    if not outer:
        return None
    real = {(source.alias_or_name or "").lower() for source in _real_sources(result) or []}
    changed = False
    for test in list(result.args["where"].find_all(exp.Exists)):
        if test.find_ancestor(exp.Select) is not result or not isinstance(test.this, exp.Select):
            continue
        body = _reading(test.this.copy(), outer, real, not_null, types)
        if body is not None:
            test.set("this", body)
            changed = True
    if not changed or _unstable(select):
        return None
    return result
