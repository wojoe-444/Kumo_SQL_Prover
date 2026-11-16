"""A ``LEFT JOIN`` with a derived table that keeps only the unmatched rows is ``NOT EXISTS``.

    SELECT c.id FROM clients AS c
    LEFT JOIN (SELECT p.id FROM clients AS p JOIN calls AS k ON k.client = p.id WHERE k.day > p.joined) AS d
      ON c.id = d.id
    WHERE d.id IS NULL

is

    SELECT c.id FROM clients AS c
    WHERE NOT EXISTS (SELECT 1 FROM (SELECT p.id FROM clients AS p JOIN calls AS k ..) AS d WHERE c.id = d.id)

The second form is the one the other rules read: the derived table under ``EXISTS`` is merged into the subquery,
and a subquery that finds the row's own table again through its key is read as a test on the row itself.

**Why it is sound.** Take a row ``r`` of everything the select joins before ``d``. The ``LEFT JOIN`` keeps
``r`` once per row of ``d`` that makes the ``ON`` condition TRUE, or once, padded with NULLs, when no row does.
The ``ON`` condition has a top-level ``AND`` part that compares ``d.c`` (``=``, ``<>``, ``<``, ``<=``, ``>``,
``>=``), which is not TRUE for a NULL ``d.c``; so in every matched row ``d.c`` is not NULL, and the ``WHERE``
part ``d.c IS NULL`` keeps exactly the padded rows: ``r`` once when no row of ``d`` matches it, and never
otherwise. That is ``NOT EXISTS (SELECT 1 FROM d WHERE on)`` over the rows ``r``. The other ``WHERE`` parts are
the same on both sides, since nothing else reads ``d``.

A derived table is not correlated, so it means the same inside the subquery. ``DISTINCT`` on it is dropped
there (whether a matching row exists does not depend on how often it occurs), unless it has a ``LIMIT``.

**What is read.** The *last* join of the select is a plain ``LEFT [OUTER] JOIN`` of an aliased derived table
with an ``ON`` condition. No join of the select is ``RIGHT``, ``FULL``, ``NATURAL`` or has ``USING``. The
``WHERE`` has, as top-level ``AND`` parts, one or more tests ``d.c IS NULL`` with the qualifier written, and for
at least one of them the ``ON`` condition compares ``d.c`` as said. Every such test is dropped (on a padded row
each is TRUE).

The derived table is one select whose outputs all have names (no star, no unnamed expression), and the columns
of ``d`` that the ``ON`` condition and the tests read are among them. Nothing else may read ``d``: outside the
join, no column is qualified with ``d`` (at any depth), no bare column has the name of one of ``d``'s outputs,
and the select list has no bare ``*``.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _unwrap
from .ast_utils import FROM_KEY, conjuncts, plain_distinct, select_sources

_COMPARISONS = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE)
_JOIN_ARGS = frozenset({"this", "on", "side", "kind", "comments"})


def derived_anti_join(select: exp.Select) -> exp.Expression | None:
    """``select`` (in place) with the join described in the module docstring written as ``NOT EXISTS``, else ``None``."""

    joins = select.args.get("joins") or []
    where = select.args.get("where")
    if not joins or where is None or select.args.get("laterals"):
        return None
    for join in joins:
        if join.method or join.args.get("using") or (join.side or "").upper() in ("RIGHT", "FULL"):
            return None
    join = joins[-1]
    on = join.args.get("on")
    if (join.side or "").upper() != "LEFT" or (join.kind or "").upper() not in ("", "OUTER") or on is None:
        return None
    if any(value for key, value in join.args.items() if key not in _JOIN_ARGS):
        return None
    derived = join.this
    if not isinstance(derived, exp.Subquery) or not derived.alias or not isinstance(derived.this, exp.Query):
        return None
    if derived.args["alias"].args.get("columns") or any(value for key, value in derived.args.items() if key not in ("this", "alias", "comments")):
        return None
    d = derived.alias.lower()
    if [source.alias_or_name.lower() for source in select_sources(select)].count(d) != 1:
        return None
    rejected = _rejected_columns(on, d)
    parts = conjuncts(where.this)
    tests = [part for part in parts if _null_test(part, d) is not None]
    if not any(_null_test(part, d) in rejected for part in tests):
        return None
    if any(isinstance(item, exp.Star) for item in select.expressions):
        return None
    outputs = _outputs(derived.this)
    if outputs is None:
        return None
    for node in select.walk():
        if not isinstance(node, exp.Column):
            continue
        if _within(node, on, select) or any(_within(node, test, select) for test in tests):
            if node.table.lower() == d and node.name.lower() not in outputs:
                return None  # not a column the derived table names
        elif _within(node, join, select):
            continue  # inside the derived table
        elif node.table.lower() == d or (not node.table and node.name.lower() in outputs):
            return None
    # nothing is refused from here on
    inner = derived.copy()
    query = inner.this
    if isinstance(query, exp.Select) and plain_distinct(query) and not query.args.get("limit") and not query.args.get("offset"):
        query.set("distinct", None)
    subquery = exp.Select(expressions=[exp.Literal.number(1)])
    subquery.set(FROM_KEY, exp.From(this=inner))
    subquery.set("where", exp.Where(this=on.copy()))
    kept = [exp.Paren(this=part.copy()) if isinstance(part, exp.Or) else part.copy() for part in parts if not any(part is test for test in tests)]
    condition = None
    for part in kept + [exp.Not(this=exp.Exists(this=subquery))]:
        condition = part if condition is None else exp.And(this=condition, expression=part)
    select.set("joins", joins[:-1] or None)
    select.set("where", exp.Where(this=condition))
    return select


def _within(node: exp.Expression, ancestor: exp.Expression, stop: exp.Expression) -> bool:
    while node is not None and node is not stop:
        if node is ancestor:
            return True
        node = node.parent
    return False


def _column_of(node: exp.Expression, d: str) -> str | None:
    node = _unwrap(node)
    if type(node) is exp.Column and not isinstance(node.this, exp.Star) and node.table.lower() == d and not node.args.get("db"):
        return node.name.lower()
    return None


def _null_test(part: exp.Expression, d: str) -> str | None:
    """``c`` when ``part`` is ``d.c IS NULL``."""

    part = _unwrap(part)
    if isinstance(part, exp.Is) and isinstance(_unwrap(part.expression), exp.Null):
        return _column_of(part.this, d)
    return None


def _rejected_columns(on: exp.Expression, d: str) -> set[str]:
    """The columns of ``d`` that a top-level ``AND`` part of ``on`` compares: NULL there, the part is not TRUE."""

    found = set()
    for part in conjuncts(on):
        if type(part) in _COMPARISONS:
            for side in (part.this, part.expression):
                name = _column_of(side, d)
                if name is not None:
                    found.add(name)
    return found


def _outputs(query: exp.Expression) -> set[str] | None:
    """The output names of a derived table's select, or None when they cannot be listed."""

    if not isinstance(query, exp.Select):
        return None
    names = set()
    for item in query.expressions:
        if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)) or not item.alias_or_name:
            return None
        names.add(item.alias_or_name.lower())
    return names
