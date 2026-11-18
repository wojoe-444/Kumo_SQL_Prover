"""A row that passes ``NOT EXISTS`` over its own tables is not itself a row the test looks for.

    SELECT DISTINCT s.buyer FROM sale AS s JOIN item AS i ON s.item_id = i.item_id
    WHERE (i.label = 'S' OR i.label = 'I')
      AND NOT EXISTS (SELECT 1 FROM sale AS k JOIN item AS j ON k.item_id = j.item_id
                      WHERE (j.label = 'S' OR j.label = 'I') AND j.label = 'I' AND k.buyer = s.buyer)

keeps a sale of a buyer who has no sale of an ``I``. The sale itself is one of that buyer's sales, so its own
label is not ``I``, and the select may say so:

    .. WHERE (i.label = 'S' OR i.label = 'I') AND NOT i.label = 'I' AND NOT EXISTS (..)

``counted_presence`` writes the first form for ``.. WHERE label IN ('S', 'I') GROUP BY buyer HAVING
SUM(CASE WHEN label = 'I' THEN 1 ELSE 0 END) = 0``, and with the added test the solver can see that the row's
label is ``S``, which is how the list spelling of the question says it.

**Why it holds.** Let the subquery read sources ``k1 .. kn`` and let each ``ki`` read the same base table as a
different source ``si`` of the select. The select's joins are inner, cross or comma joins, so on a row that
reaches the ``WHERE`` each ``si`` is a real row of its table, and ``k1 = s1, .., kn = sn`` is one of the
combinations the subquery ranges over. Write each condition of the subquery with ``si`` for ``ki``. A condition
that is then, as text, a condition of the select itself (a conjunct of an ``ON`` or of the ``WHERE``) is TRUE on
the row. A tie ``ki.c = si.c`` becomes ``si.c = si.c``, TRUE because ``c`` is declared NOT NULL. If the remaining
conditions ``R1 .. Rm`` were all TRUE on the row, the row would be a row the subquery returns and ``NOT EXISTS``
would be FALSE. So on every row the ``WHERE`` keeps, ``R1 AND .. AND Rm`` is not TRUE; every column they read is
declared NOT NULL and they are built from comparisons with literals, so it is FALSE, and ``NOT (R1 AND .. AND
Rm)`` is TRUE. Adding a conjunct that is TRUE on every kept row and can only remove rows that the ``NOT EXISTS``
removes anyway changes nothing.

**What is read.** A top-level conjunct ``NOT EXISTS (plain select)`` of the ``WHERE`` of a select whose sources
are base tables under inner, cross or comma joins, each under its own name. The subquery: base tables under
inner, cross or comma joins, no grouping, aggregate, window, ``DISTINCT``, row limit or nested query; each of its
tables is read by exactly one source of the subquery and one of the select. Every column of the subquery names
its source. At least one condition must remain, and the remaining ones are built with ``AND``, ``OR``, ``NOT``
from comparisons (``= <> < <= > >=``, ``BETWEEN``, ``IN`` of literals) between a column declared NOT NULL and a
literal. The conjunct is not added twice. A volatile call or a quoted name leaves the select as it is.

A narrow retry rule (``shape_fallback``). ``tests/test_own_row_witness.py`` replays the rewrite on DuckDB.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, conjuncts
from .own_group_membership import _facts, _plain_table, _unparen

_BLOCKING = ("group", "having", "qualify", "limit", "offset", "laterals", "with_", "with", "order", "windows", "distinct", "connect", "match", "into", "locks", "sample", "pivots")
_JOIN_EXTRAS = ("method", "using", "match_condition", "global_", "hint", "directed", "expressions", "pivots")
_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)


def _shape(select: exp.Select) -> tuple[dict[str, exp.Table], list[exp.Expression]] | None:
    """``(sources by name, top-level conditions)`` of a select over base tables under inner, cross or comma joins."""

    from_ = select.args.get(FROM_KEY)
    if from_ is None:
        return None
    joins = select.args.get("joins") or []
    for join in joins:
        if join.side or (join.kind or "").upper() not in ("", "INNER", "CROSS") or any(join.args.get(key) for key in _JOIN_EXTRAS):
            return None
    sources = [from_.this] + [join.this for join in joins]
    if any(not _plain_table(source) or not isinstance(source.this, exp.Identifier) or source.this.args.get("quoted") or source.args.get("db") for source in sources):
        return None
    names = [(source.alias_or_name or "").lower() for source in sources]
    if not all(names) or len(set(names)) != len(names):
        return None
    parts = [part for join in joins if join.args.get("on") is not None for part in conjuncts(join.args["on"])]
    if select.args.get("where") is not None:
        parts += conjuncts(select.args["where"].this)
    return dict(zip(names, sources)), parts


def _text(node: exp.Expression) -> str:
    copy = exp.Paren(this=_unparen(node).copy())
    for identifier in copy.find_all(exp.Identifier):
        if not identifier.args.get("quoted"):
            identifier.set("this", identifier.name.lower())
    return copy.this.sql(dialect="mysql")


def _decided(node: exp.Expression | None, sources: dict[str, exp.Table], not_null: dict | None) -> bool:
    """Whether ``node`` is built from comparisons of NOT NULL columns of ``sources`` with literals: never NULL."""

    node = _unparen(node)
    if isinstance(node, (exp.And, exp.Or)):
        return _decided(node.this, sources, not_null) and _decided(node.expression, sources, not_null)
    if isinstance(node, exp.Not):
        return _decided(node.this, sources, not_null)

    def column(end: exp.Expression | None) -> bool:
        end = _unparen(end)
        if not isinstance(end, exp.Column) or not isinstance(end.this, exp.Identifier) or end.this.args.get("quoted") or not end.table:
            return False
        source = sources.get(end.table.lower())
        return source is not None and end.name.lower() in {str(c).lower() for c in (_facts(not_null, source.name) or ())}

    def literal(end: exp.Expression | None) -> bool:
        end = _unparen(end)
        end = end.this if isinstance(end, exp.Neg) else end
        return isinstance(end, exp.Literal)

    if isinstance(node, _COMPARISONS):
        return (column(node.this) and literal(node.expression)) or (literal(node.this) and column(node.expression))
    if isinstance(node, exp.Between) and not node.args.get("symmetric"):
        return column(node.this) and literal(node.args.get("low")) and literal(node.args.get("high"))
    if isinstance(node, exp.In) and node.expressions and not any(node.args.get(key) for key in ("query", "unnest", "field")):
        return column(node.this) and all(literal(item) for item in node.expressions)
    return False


def add_own_row_tests(select: exp.Select, not_null: dict | None) -> exp.Select | None:
    """``select`` with, for each ``NOT EXISTS`` over its own tables, the test the row itself must fail; or None."""

    from .counted_membership import _unstable

    where = select.args.get("where")
    if where is None or not not_null or where.find(exp.Exists) is None:
        return None
    outer = _shape(select)
    if outer is None:
        return None
    sources, conditions = outer
    known = {_text(part) for part in conditions}
    added: list[exp.Expression] = []
    for part in conjuncts(where.this):
        test = _unparen(part)
        if not isinstance(test, exp.Not) or not isinstance(_unparen(test.this), exp.Exists):
            continue
        body = _unparen(test.this).this
        if not isinstance(body, exp.Select) or any(body.args.get(key) for key in _BLOCKING):
            continue
        inner = _shape(body)
        if inner is None:
            continue
        theirs, tests = inner
        if set(theirs) & set(sources) or any(isinstance(node, (exp.Query, exp.Subquery, exp.AggFunc, exp.Window, exp.Star)) for node in body.walk() if node is not body):
            continue
        if any(not column.table or column.args.get("db") or not isinstance(column.this, exp.Identifier) or column.this.args.get("quoted") for column in body.find_all(exp.Column)):
            continue
        tables = [source.name.lower() for source in theirs.values()]
        mine = [source.name.lower() for source in sources.values()]
        if len(set(tables)) != len(tables) or any(mine.count(table) != 1 for table in tables):
            continue  # which row stands for which is certain only when each table is read once on each side
        stands = {name: next(o for o, source in sources.items() if source.name.lower() == theirs[name].name.lower()) for name in theirs}
        remaining = []
        for condition in tests:
            moved = exp.Paren(this=condition.copy())
            for column in list(moved.find_all(exp.Column)):
                if column.table.lower() in stands:
                    column.replace(exp.column(column.name, table=sources[stands[column.table.lower()]].alias_or_name))
            moved = _unparen(moved)
            if _text(moved) in known:
                continue  # a condition of the select itself: TRUE on the row
            if isinstance(moved, exp.EQ) and _text(moved.this) == _text(moved.expression) and _decided(exp.EQ(this=moved.this.copy(), expression=exp.Literal.number(0)), sources, not_null):
                continue  # a tie to the row's own NOT NULL column
            remaining.append(moved)
        if not remaining or not all(_decided(condition, sources, not_null) for condition in remaining):
            continue
        fails = exp.Not(this=exp.Paren(this=exp.and_(*remaining, copy=False)) if len(remaining) > 1 or isinstance(remaining[0], (exp.And, exp.Or)) else remaining[0])
        if _text(fails) not in known and _text(fails) not in {_text(done) for done in added}:
            added.append(fails)
    if not added or _unstable(select):
        return None
    result = select.copy()
    result.set("where", exp.Where(this=exp.and_(result.args["where"].this, *added, copy=False)))
    return result
