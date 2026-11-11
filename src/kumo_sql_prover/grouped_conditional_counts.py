"""A conditional count of a grouped query, written as a sum of ones and zeros.

``COUNT(CASE WHEN c THEN 1 END)`` counts the rows of a group where ``c`` is TRUE: the ``CASE`` is NULL for
every other row, and ``COUNT`` skips NULLs. ``SUM(CASE WHEN c THEN 1 ELSE 0 END)`` adds a one for the same
rows and a zero for the rest. The two differ only over no rows at all, where the count is 0 and the sum is
NULL, and a group of a ``GROUP BY`` always has a row. Reports that bucket rows are written both ways, and
the two spellings did not compare. This rule gives the count the sum's spelling.

Read only when:

* the select has a ``GROUP BY`` with keys and no ``ROLLUP``, ``CUBE``, ``GROUPING SETS`` or empty grouping
  set: each of those can make a grand-total row over no input rows, where 0 and NULL differ. A query with
  aggregates and no ``GROUP BY`` is left alone for the same reason;
* the ``COUNT`` is the select's own, has one argument, no ``DISTINCT``, no ``FILTER`` and no ``OVER`` (a
  filter or a window frame can be empty);
* its argument is a searched ``CASE`` whose every arm returns a literal that is not NULL, with no ``ELSE``
  or ``ELSE NULL``. With several arms the ``CASE`` is not NULL exactly when some arm's condition is TRUE,
  which is when the sum's ``CASE`` returns its one;
* some column of the argument is the select's own: qualified with the name of one of its sources, or
  unqualified and listed by name in a derived table that is one of its sources (the innermost name wins).
  A column inside a subquery of the argument does not count. An aggregate that reads only columns of an
  enclosing query belongs to that query, which may have no ``GROUP BY``.

The count of the rows where a column is, or is not, NULL has a shorter spelling still, and is given it:
``SUM(CASE WHEN x IS NOT NULL THEN 1 ELSE 0 END)`` is ``COUNT(x)`` and ``SUM(CASE WHEN x IS NULL THEN 1 ELSE 0
END)`` is ``COUNT(*) - COUNT(x)``, again only inside a group, which has a row. This is read under the same
conditions, for a sum with no ``DISTINCT``, ``FILTER`` or ``OVER`` whose one arm tests one of the select's own
columns and returns the number 1, with ``ELSE 0``.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import extended_grouping


def _own_names(select: exp.Select) -> tuple[set[str], set[str]]:
    """Names of the select's sources, and the column names its derived tables list."""

    source = select.args.get("from_") or select.args.get("from")
    relations = ([source.this] if source is not None else []) + [join.this for join in select.args.get("joins") or []]
    listed: set[str] = set()
    for relation in relations:
        if isinstance(relation, exp.Subquery) and isinstance(relation.this, exp.Select):
            listed.update(item.alias_or_name.lower() for item in relation.this.expressions if isinstance(item, (exp.Alias, exp.Column)) and item.alias_or_name)
    return {relation.alias_or_name.lower() for relation in relations if relation.alias_or_name}, listed


def _is_own(node: exp.Expression, own: set[str], listed: set[str]) -> bool:
    if not isinstance(node, exp.Column):
        return False
    return node.table.lower() in own if node.table else node.name.lower() in listed


def _null_test(node: exp.Expression) -> tuple[exp.Expression, bool] | None:
    """``(x, True)`` for ``x IS NULL`` and ``(x, False)`` for ``x IS NOT NULL``, in either spelling of the tree."""

    negated = False
    while isinstance(node, (exp.Paren, exp.Not)):
        negated ^= isinstance(node, exp.Not)
        node = node.this
    if not isinstance(node, exp.Is) or not isinstance(node.expression, exp.Null):
        return None
    return node.this, not (negated ^ bool(node.args.get("negate")))


def _is_number(node: exp.Expression | None, text: str) -> bool:
    return isinstance(node, exp.Literal) and not node.is_string and node.name == text


def _null_counts(select: exp.Select, own: set[str], listed: set[str]) -> bool:
    changed = False
    for total in list(select.find_all(exp.Sum)):
        if total.find_ancestor(exp.Select) is not select or isinstance(total.parent, (exp.Window, exp.Filter)):
            continue
        case = total.this
        if not isinstance(case, exp.Case) or case.this is not None:
            continue
        arms = case.args.get("ifs") or []
        if len(arms) != 1 or not _is_number(arms[0].args.get("true"), "1") or not _is_number(case.args.get("default"), "0"):
            continue
        test = _null_test(arms[0].this)
        if test is None or not _is_own(test[0], own, listed):
            continue
        column, is_null = test
        if not is_null:
            total.replace(exp.Count(this=column.copy()))
        else:
            missing = exp.Sub(this=exp.Count(this=exp.Star()), expression=exp.Count(this=column.copy()))
            total.replace(missing if isinstance(total.parent, (exp.Alias, exp.Select)) else exp.Paren(this=missing))
        changed = True
    return changed


def grouped_conditional_counts(select: exp.Select) -> exp.Expression | None:
    """Rewrite the grouped select's own ``COUNT(CASE WHEN c THEN 1 END)`` as a sum of ones and zeros, in place."""

    group = select.args.get("group")
    if group is None or not group.expressions or extended_grouping(group):
        return None
    if any(group.args.get(key) for key in ("all", "rollup", "cube", "grouping_sets", "totals")):
        return None
    own, listed = _own_names(select)
    changed = False
    for count in list(select.find_all(exp.Count)):
        if count.find_ancestor(exp.Select) is not select or isinstance(count.parent, (exp.Window, exp.Filter)):
            continue
        case = count.this
        if count.expressions or not isinstance(case, exp.Case) or case.this is not None:
            continue
        arms = case.args.get("ifs") or []
        default = case.args.get("default")
        if not arms or (default is not None and not isinstance(default, exp.Null)):
            continue
        if any(not isinstance(arm, exp.If) or not isinstance(arm.args.get("true"), exp.Literal) for arm in arms):
            continue
        if not any(_is_own(node, own, listed) and node.find_ancestor(exp.Select) is select for node in case.walk()):
            continue
        ones = exp.Case(
            ifs=[exp.If(this=arm.this.copy(), true=exp.Literal.number(1)) for arm in arms],
            default=exp.Literal.number(0),
        )
        count.replace(exp.Sum(this=ones))
        changed = True
    changed = _null_counts(select, own, listed) or changed
    return select if changed else None
