"""A conditional count of a grouped query, written one way.

``COUNT(CASE WHEN c THEN 1 END)`` counts the rows of a group where ``c`` is TRUE: the ``CASE`` is NULL for
every other row, and ``COUNT`` skips NULLs. ``SUM(CASE WHEN c THEN 1 ELSE 0 END)`` adds a one for the same
rows and a zero for the rest. The two differ only over no rows at all, where the count is 0 and the sum is
NULL, and a group of a ``GROUP BY`` always has a row. Reports that bucket rows are written both ways, and
the two spellings did not compare. This rule gives the sum the count's spelling, which is the one the prover
already uses for ``COUNT(*) FILTER (WHERE c)``.

Three readings, in this order:

* **The literal of a count.** ``COUNT(CASE WHEN c THEN 'y' ELSE NULL END)`` is ``COUNT(CASE WHEN c THEN 1
  END)``: a searched ``CASE`` with one arm that returns a literal (never NULL) and no ``ELSE`` or ``ELSE
  NULL`` is not NULL exactly when ``c`` is TRUE, whatever the literal. True of any count, grouped or not.
* **The sum of ones.** ``SUM(CASE WHEN c THEN 1 ELSE 0 END)`` is ``COUNT(CASE WHEN c THEN 1 END)``, read only
  when the select has a ``GROUP BY`` with keys and no ``ROLLUP``, ``CUBE``, ``GROUPING SETS`` or empty
  grouping set (each of those can make a grand-total row over no input rows, where 0 and NULL differ; a query
  with aggregates and no ``GROUP BY`` is left alone for the same reason), and only when some column of the
  argument is the select's own: qualified with the name of one of its sources, or unqualified and listed by
  name in a derived table that is one of its sources. A column inside a subquery of the argument does not
  count. An aggregate that reads only columns of an enclosing query belongs to that query, which may have no
  ``GROUP BY``.
* **The count of a NULL test.** ``COUNT(CASE WHEN x IS NOT NULL THEN 1 END)`` is ``COUNT(x)`` and
  ``COUNT(CASE WHEN x IS NULL THEN 1 END)`` is ``COUNT(*) - COUNT(x)``, for a column ``x`` of the select's own
  (``COUNT(*)`` belongs to the select it is written in, so ``x`` must too). Over no rows all of them are 0, so
  no ``GROUP BY`` is needed here; the sums of the same tests get there through the reading above, under its
  conditions.

Every aggregate read is the select's own, has one argument and no ``DISTINCT``, ``FILTER`` or ``OVER`` (a
filter or a window frame can be empty). ``SUM`` of integers and ``COUNT`` have the same type in PostgreSQL and
BigQuery; elsewhere the value is the same.
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


def _plain_call(call: exp.Expression, select: exp.Select) -> bool:
    """One of the select's own aggregate calls, with nothing on it but its argument."""

    if call.find_ancestor(exp.Select) is not select or isinstance(call.parent, (exp.Window, exp.Filter)):
        return False
    return not any(value for key, value in call.args.items() if key not in ("this", "big_int") and value)


def _one_arm(case: exp.Expression | None) -> exp.If | None:
    """The arm of a searched ``CASE`` with one ``WHEN``."""

    if not isinstance(case, exp.Case) or case.this is not None:
        return None
    arms = case.args.get("ifs") or []
    return arms[0] if len(arms) == 1 and isinstance(arms[0], exp.If) else None


def _counted(condition: exp.Expression) -> exp.Count:
    return exp.Count(this=exp.Case(ifs=[exp.If(this=condition.copy(), true=exp.Literal.number(1))]))


def grouped_conditional_counts(select: exp.Select) -> exp.Expression | None:
    """Rewrite the select's own conditional counts to ``COUNT(CASE WHEN c THEN 1 END)`` or shorter, in place."""

    group = select.args.get("group")
    grouped = (
        group is not None
        and bool(group.expressions)
        and not extended_grouping(group)
        and not any(group.args.get(key) for key in ("all", "rollup", "cube", "grouping_sets", "totals"))
    )
    own, listed = _own_names(select)
    changed = False
    if grouped:
        for total in list(select.find_all(exp.Sum)):
            arm = _one_arm(total.this) if _plain_call(total, select) else None
            if arm is None or not _is_number(arm.args.get("true"), "1") or not _is_number(total.this.args.get("default"), "0"):
                continue
            if not any(_is_own(node, own, listed) and node.find_ancestor(exp.Select) is select for node in total.this.walk()):
                continue
            total.replace(_counted(arm.this))
            changed = True
    for count in list(select.find_all(exp.Count)):
        arm = _one_arm(count.this) if _plain_call(count, select) else None
        if arm is None:
            continue
        default = count.this.args.get("default")
        if not isinstance(arm.args.get("true"), exp.Literal) or (default is not None and not isinstance(default, exp.Null)):
            continue
        test = _null_test(arm.this)
        if test is not None and _is_own(test[0], own, listed):
            column, is_null = test
            if not is_null:
                count.replace(exp.Count(this=column.copy()))
            else:
                missing = exp.Sub(this=exp.Count(this=exp.Star()), expression=exp.Count(this=column.copy()))
                count.replace(missing if isinstance(count.parent, (exp.Alias, exp.Select)) else exp.Paren(this=missing))
            changed = True
        elif default is not None or not _is_number(arm.args.get("true"), "1"):
            count.replace(_counted(arm.this))
            changed = True
    return select if changed else None
