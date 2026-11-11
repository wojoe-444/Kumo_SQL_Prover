"""A lower bound on a correlated ``COUNT(*)`` is a membership test in the groups that pass the bound.

**The rule.** As a top-level conjunct of ``WHERE``,

    (SELECT COUNT(*) FROM s AS x WHERE x.c1 = o.k1 AND x.c2 = o.k2 AND p(x)) > 1

is

    (o.k1, o.k2) IN (SELECT x.c1, x.c2 FROM s AS x WHERE p(x) GROUP BY x.c1, x.c2 HAVING COUNT(*) > 1)

for any comparison of the count with a literal that puts a lower bound of 1 or more on it: ``> m`` (``m >= 0``),
``>= m`` and ``= m`` (``m >= 1``), with the literal on either side. One tie is written ``o.k IN (SELECT x.c ..)``.

**Why it is sound.** Fix a row ``o`` of the outer select.

* Some ``o.ki`` is NULL. No row of ``x`` passes ``x.ci = o.ki``, so the count is 0 and the bound fails. On the
  other side a row comparison with a NULL component is never TRUE, so the ``IN`` is not TRUE either.
* No ``o.ki`` is NULL. The rows of ``x`` that pass ``p`` and all the equalities are the rows whose ``(c1, c2)``
  equals ``(k1, k2)``, and those are exactly one group of the grouped subquery (or none): ``GROUP BY`` puts
  together the rows whose keys are equal, and a group with a NULL key holds none of them. The correlated count
  ``N`` is that group's ``COUNT(*)``, or 0 when there is no such group. A bound that needs ``N >= 1`` therefore
  holds exactly when the group exists and its count passes the same comparison, which is exactly when the
  subquery returns the row ``(k1, k2)``, the only row it could return that equals ``(o.k1, o.k2)``.

Where the ``IN`` is not TRUE it is FALSE or NULL, which a ``WHERE`` conjunct treats alike; nothing but a
top-level conjunct is rewritten. A bound that zero satisfies (``= 0``, ``< 2``, ``<> 1``) is not read: the rows
with no partner pass it, and they are not in any group.

``x.c = o.k`` compares as ``o.k IN (SELECT x.c ..)`` does, but ``GROUP BY x.c`` must also put together exactly
the rows the equality matches with one ``o.k``. Values that compare equal are one value only within one type
(MySQL finds ``1 = '1'`` and ``1 = '01'``, two groups), so both columns of each tie must be declared with one
type that is not a floating type, or neither declared (the prover's standing same-type assumption, as in
``semijoin_shapes`` step 4).

**What is read.** The subquery: a select with only a select list, ``FROM``, ``WHERE`` and optionally a
``GROUP BY``; one source, a plain base table whose columns the schema lists; one output, ``COUNT(*)``, ``COUNT``
of a non-NULL literal, or ``COUNT`` of one of the tied columns (not NULL in every counted row). Its ``WHERE`` is
a conjunction of ties (``x.c = o.k``, either order, or one row equality ``(o.k1, o.k2) = (x.c1, x.c2)``) and of
conditions that read ``x`` alone and hold no subquery. The row equality is read only for the dialects where it
is the conjunction of its column equalities (MySQL, BigQuery, PostgreSQL; DuckDB finds two NULLs in a row
comparison equal, so there a NULL key would count its partners). A column is ``x``'s when it is qualified with ``x``'s name,
or bare and listed for the table (the innermost select binds a bare name first); it is the outer row's when it is
qualified with the name of a source of the select whose ``WHERE`` holds the test, and ``x`` has another name.
A ``GROUP BY`` in the subquery may list tied columns only: it then returns NULL instead of 0 when nothing
matches and never more than one row, and the bound fails on NULL as it does on 0.

**What is refused.** Everything else: a count in ``SELECT``, ``HAVING``, ``ON`` or under ``OR``/``NOT``; ``COUNT(DISTINCT
..)``, ``COUNT`` of another column, other aggregates; a subquery with a join, a derived table, ``DISTINCT``,
``HAVING``, ``LIMIT`` or a window; a tie that is not a plain column equality or whose outer side is an expression,
a bare name or a name bound further out; a subquery without a tie; a bound that is not a lower bound of at
least 1; ties between columns of different or floating declared types.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, conjuncts
from .decorrelation_rules import _and_all, _sources
from .having_count_rules import _natural
from .semijoin_shapes import _FLOATING, _VOLATILE, _declared_type, _derived, _only, _plain_column, _plain_table

_SUBQUERY = frozenset({"expressions", "from_", "from", "where", "group"})
_MIRROR = {exp.GT: exp.LT, exp.LT: exp.GT, exp.GTE: exp.LTE, exp.LTE: exp.GTE, exp.EQ: exp.EQ}


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _scalar_select(node: exp.Expression) -> exp.Select | None:
    node = _unwrap(node)
    if type(node) is not exp.Subquery or node.alias or any(value for key, value in node.args.items() if key != "this"):
        return None
    return node.this if type(node.this) is exp.Select else None


def _membership(test: exp.Expression, outer: exp.Select, known: dict[str, set[str]], types: dict | None, row_ties: bool) -> exp.Expression | None:
    """The ``IN`` test that ``test`` is, else ``None``."""

    kind = type(test)
    if kind not in _MIRROR:
        return None
    body, number = _scalar_select(test.this), _natural(test.expression)
    if body is None or number is None:
        body, number, kind = _scalar_select(test.expression), _natural(test.this), _MIRROR[kind]
    if body is None or number is None:
        return None
    if not ((kind is exp.GT and number >= 0) or (kind in (exp.GTE, exp.EQ) and number >= 1)):
        return None
    sources = _sources(body)
    if not _only(body, _SUBQUERY) or len(sources) != 1 or not _plain_table(sources[0]) or len(body.expressions) != 1:
        return None
    source = sources[0]
    listed = known.get(source.name.lower())
    name = source.alias_or_name.lower()
    outer_sources = _sources(outer)
    outer_names = [(s.alias_or_name or "").lower() for s in outer_sources]
    if not listed or "" in outer_names or len(set(outer_names)) != len(outer_names) or name in outer_names:
        return None
    if not all(_plain_table(s) or _derived(s) is not None for s in outer_sources):
        return None
    by_name = dict(zip(outer_names, outer_sources))

    def inner(node: exp.Expression) -> exp.Column | None:
        column = _plain_column(node)
        if column is None or column.name.lower() not in listed:
            return None
        return column if column.table.lower() in ("", name) else None

    def outside(node: exp.Expression) -> exp.Column | None:
        column = _plain_column(node)
        return column if column is not None and column.table.lower() in by_name else None

    ties: list[tuple[exp.Column, exp.Column]] = []
    local: list[exp.Expression] = []
    where = body.args.get("where")
    for part in conjuncts(where.this) if where is not None else []:
        pairs = None
        if type(part) is exp.EQ:
            left, right = _unwrap(part.this), _unwrap(part.expression)
            if type(left) is exp.Tuple or type(right) is exp.Tuple:
                if row_ties and type(left) is type(right) and len(left.expressions) == len(right.expressions) and left.expressions:
                    pairs = list(zip(left.expressions, right.expressions))
            else:
                pairs = [(left, right)]
        found = []
        for a, b in pairs or []:
            if inner(a) is not None and outside(b) is not None:
                found.append((inner(a), outside(b)))
            elif inner(b) is not None and outside(a) is not None:
                found.append((inner(b), outside(a)))
        if pairs and len(found) == len(pairs):
            ties += found
            continue
        # not a tie: it must read x alone
        if any(isinstance(n, (exp.Select, exp.Subquery, exp.AggFunc, exp.Window, *_VOLATILE)) for n in part.walk()):
            return None
        if any(inner(c) is None for c in part.find_all(exp.Column)):
            return None
        local.append(part)
    if not ties:
        return None
    tied = {c.name.lower() for c, _ in ties}
    count = _unwrap(body.expressions[0].unalias())
    if type(count) is not exp.Count or any(value for key, value in count.args.items() if key not in ("this", "big_int")):
        return None
    counted = _unwrap(count.this) if isinstance(count.this, exp.Expression) else None
    if not isinstance(counted, exp.Star) and not (isinstance(counted, exp.Literal)) and not (counted is not None and inner(counted) is not None and inner(counted).name.lower() in tied):
        return None
    group = body.args.get("group")
    if group is not None:
        if any(value for key, value in group.args.items() if key != "expressions"):
            return None
        if any(inner(item) is None or inner(item).name.lower() not in tied for item in group.expressions):
            return None
    for mine, other in ties:
        kinds = [_declared_type(mine, source, types or {}), _declared_type(other, by_name[other.table.lower()], types or {})]
        if kinds[0] != kinds[1] or any(f in k for k in kinds if k for f in _FLOATING):
            return None
    grouped = exp.Select(expressions=[mine.copy() for mine, _ in ties])
    grouped.set(FROM_KEY, exp.From(this=source.copy()))
    condition = _and_all([part.copy() for part in local])
    if condition is not None:
        grouped.set("where", exp.Where(this=condition))
    grouped.set("group", exp.Group(expressions=[mine.copy() for mine, _ in ties]))
    grouped.set("having", exp.Having(this=kind(this=exp.Count(this=exp.Star()), expression=exp.Literal.number(number))))
    left = ties[0][1].copy() if len(ties) == 1 else exp.Tuple(expressions=[other.copy() for _, other in ties])
    return exp.In(this=left, query=exp.Subquery(this=grouped))


_ROW_TIES = frozenset({"mysql", "bigquery", "postgres"})  # (a, b) = (c, d) is a = c AND b = d there, NULLs included


def read_correlated_count_tests(tree: exp.Expression, schema: dict[str, list[str]] | None, types: dict | None = None, dialect: str | None = None) -> exp.Expression:
    """``tree`` (changed in place) with each bounded correlated count in a ``WHERE`` written as a grouped membership test."""

    if not schema:
        return tree
    known = {k.lower(): {c.lower() for c in v} for k, v in schema.items()}
    kinds = {k.lower(): {c.lower(): t for c, t in v.items()} for k, v in (types or {}).items()}
    for select in list(tree.find_all(exp.Select)):
        where = select.args.get("where")
        if where is None or select.args.get("laterals"):
            continue
        parts = conjuncts(where.this)
        rewritten = [_membership(part, select, known, kinds, dialect in _ROW_TIES) for part in parts]
        if not any(r is not None for r in rewritten):
            continue
        where.set("this", _and_all([new if new is not None else part.copy() for part, new in zip(parts, rewritten)]))
    return tree
