"""In the ``HAVING`` of a grouped membership subquery, ``COUNT`` of a compared group key is ``COUNT(*)``.

**The rule.** As a top-level conjunct of ``WHERE``,

    x IN (SELECT k FROM s GROUP BY k HAVING COUNT(k) > 1)

is

    x IN (SELECT k FROM s GROUP BY k HAVING COUNT(*) > 1)

and likewise with several compared columns, ``(x, y) IN (SELECT k1, k2 .. GROUP BY k1, k2 HAVING COUNT(k2) = 1)``,
for any ``HAVING`` condition: every ``COUNT(k)`` of the subquery's own ``HAVING`` whose argument is a ``GROUP BY``
column that the subquery also returns is replaced.

**Why it is sound.** The two subqueries differ only in which groups pass ``HAVING``. In a group whose ``k`` is
not NULL every row has that ``k``, so ``COUNT(k)`` counts every row and equals ``COUNT(*)``: the ``HAVING``
decides the same. In the group whose ``k`` is NULL the two counts differ (0 and the number of rows), so that
group may be returned by one subquery only; but its row holds NULL in the position of ``k``, and a row
comparison with a NULL component is never TRUE, so the row cannot make the ``IN`` TRUE for any ``x``. It can
only turn a FALSE into a NULL or back, which a ``WHERE`` conjunct treats alike. Hence the rule reads only an
``IN`` that is a top-level conjunct of a ``WHERE`` (not under ``NOT`` or ``OR``, not in a select list, where
NULL and FALSE differ).

**What is read.** The ``IN`` subquery: a select with only a select list, ``FROM``, joins, ``WHERE``, a plain
``GROUP BY``, ``HAVING`` and ``DISTINCT``. The counted column, a ``GROUP BY`` item and an output must be the same
column: spelled alike, or the same column of the same source (``s.k`` and a bare ``k`` of the only source
``s``, a base table whose columns the schema lists). The count is the subquery's own (not a nested select's or a
window's), without ``DISTINCT`` or ``FILTER``.

**What is refused.** ``NOT IN`` and any other place than a ``WHERE`` conjunct; ``COUNT`` of a column that is not
both grouped and returned (its NULLs are counted out of rows that can match); a subquery with ``LIMIT``,
``ORDER BY``, ``ROLLUP`` or grouping sets (the total rows hold NULL keys of non-NULL groups); a bare name that a
select alias of the subquery gives to something else, in ``GROUP BY`` or as the argument.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import conjuncts, extended_grouping
from .decorrelation_rules import _sources
from .grouped_source_dedup import _source_text, renaming_aliases
from .semijoin_shapes import _only, _own, _plain_column

_BODY = frozenset({"expressions", "from_", "from", "joins", "where", "group", "having", "distinct"})


def _same(node: exp.Expression, sources: list[exp.Expression], listed: dict[str, set[str]], renamed: set[str]) -> set[str]:
    """The spellings under which the plain column ``node`` is compared (empty when it is not one, or ambiguous)."""

    column = _plain_column(node)
    if column is None or (not column.table and column.name.lower() in renamed):
        return set()
    spellings = {column.sql().lower()}
    text = _source_text(column, sources, listed)
    if text is not None:
        spellings.add(text)
    return spellings


def _count_keys(body: exp.Select, listed: dict[str, set[str]]) -> bool:
    group, having = body.args.get("group"), body.args.get("having")
    if group is None or having is None or not _only(body, _BODY) or extended_grouping(group):
        return False
    if any(value for key, value in group.args.items() if key != "expressions"):
        return False
    if body.args.get("distinct") is not None and body.args["distinct"].args.get("on"):
        return False
    sources = _sources(body)
    renamed = renaming_aliases(body)
    grouped = [_same(item, sources, listed, renamed) for item in group.expressions]
    returned = [_same(item.unalias(), sources, listed, set()) for item in body.expressions]
    changed = False
    for count in list(having.find_all(exp.Count)):
        if not _own(count, body) or count.find_ancestor(exp.Window) is not None or isinstance(count.parent, exp.Filter):
            continue
        if any(value for key, value in count.args.items() if key not in ("this", "big_int")) or not isinstance(count.this, exp.Expression):
            continue
        mine = _same(count.this, sources, listed, renamed)
        if mine and any(mine & g for g in grouped) and any(mine & r for r in returned):
            count.set("this", exp.Star())
            changed = True
    return changed


def read_membership_key_counts(tree: exp.Expression, schema: dict[str, list[str]] | None) -> exp.Expression:
    """``tree`` (changed in place) with ``COUNT(key)`` read as ``COUNT(*)`` in the ``HAVING`` of grouped ``IN`` subqueries of a ``WHERE``."""

    listed = {k.lower(): {c.lower() for c in v} for k, v in (schema or {}).items()}
    for select in list(tree.find_all(exp.Select)):
        where = select.args.get("where")
        if where is None:
            continue
        for part in conjuncts(where.this):
            query = part.args.get("query") if type(part) is exp.In else None
            if type(query) is not exp.Subquery or query.alias or type(query.this) is not exp.Select:
                continue
            if any(value for key, value in query.args.items() if key != "this"):
                continue
            _count_keys(query.this, listed)
    return tree
