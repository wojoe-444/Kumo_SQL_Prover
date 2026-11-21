"""A ``CASE`` of labels compared with one label, where only TRUE matters, is the condition that picks that label.

    WHERE CASE WHEN secs < 300 THEN 'short' WHEN secs < 900 THEN 'medium' ELSE 'long' END = 'medium'
    WHERE secs < 900 AND secs >= 300

Rows are classified with a ``CASE`` and then one class is asked for. The solver could decide such a filter, but it
reads numbers as reals, so against ``secs BETWEEN 300 AND 899`` over a whole-number column it finds 299.5 and the
pair stays unproved. Written out as plain comparisons, the bounds are closed by ``discrete_bounds`` like any other.

**Why it holds.** A searched ``CASE`` gives the result of the first ``WHEN`` whose condition is TRUE, else the
``ELSE`` result, else NULL. Every result is a literal (or NULL), and the texts among the results and the label are
such that two are equal or stay different under any collation that ignores letter case and trailing spaces (the
test of ``constant_bins``). So ``CASE .. END = label`` is TRUE exactly when the arm that is picked holds that label:
for some arm ``i`` with the label, ``c_i`` is TRUE and no earlier condition is TRUE; or no condition is TRUE and the
``ELSE`` result is the label. "``c`` is not TRUE" is ``NOT c OR c IS NULL``, and it is just ``NOT c`` when ``c``
cannot be NULL here: ``c`` is made of comparisons of columns and literals under ``AND``, ``OR`` and ``NOT`` (NULL
only when one of its columns is), and all those columns are in comparisons that the same term requires to be TRUE.
When every condition is of that kind, they read the same columns and each is TRUE only when none of those
columns is NULL (no ``OR`` that one side alone can satisfy), "no condition is TRUE" is written ``(a column IS NULL)
OR (every NOT c)``. ``NOT`` is pushed onto the comparisons (``NOT x <= 5`` is ``x > 5``; De Morgan's laws
hold with NULL). The comparison is replaced only where FALSE and NULL mean the same thing: as a whole condition, or
a term of the top-level ``AND``, of a ``WHERE``, a ``HAVING`` or a join's ``ON``.

**What is not read.** A ``CASE x WHEN ..`` with an operand; a result that is not a literal or NULL; texts and
numbers mixed; a label no arm holds; a condition with a subquery, an aggregate, a window or a function call (the
condition is written more than once); a condition that divides by anything but a constant that is not zero (a
``CASE`` stops at the first TRUE condition, so ``WHEN y = 0 .. WHEN x / y > 2`` never divides by zero, and the
reading would); a comparison whose value is used (in a select list, under ``NOT`` or ``OR``).

A narrow rule of the sixteenth hardest-cases batch (``shape_fallback.SIXTEENTH_BATCH``): it fires only in the last
step of a proof. ``tests/test_labelled_cases.py`` holds the rewritten texts, a replay against DuckDB with NULLs, the
filters left alone and the pairs proved only with the rule.
"""

from __future__ import annotations

from sqlglot import exp

from .constant_bins import _texts_safe, _unparen, _value

NAME = "labelled_cases"

_OPPOSITE = {exp.EQ: exp.NEQ, exp.NEQ: exp.EQ, exp.LT: exp.GTE, exp.GTE: exp.LT, exp.LTE: exp.GT, exp.GT: exp.LTE}
_PLAIN = (exp.And, exp.Or, exp.Not, exp.Paren, exp.Between, exp.Is, exp.In, exp.Column, exp.Identifier, exp.Literal, exp.Null, exp.Neg, exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Boolean, *_OPPOSITE)


def _strict(node: exp.Expression) -> bool:
    """Whether a value is NULL only when one of its columns is: columns and literals under ``+ - *`` and division by a constant."""

    node = _unparen(node)
    if isinstance(node, exp.Column):
        return True
    if isinstance(node, exp.Literal):
        return True
    if isinstance(node, exp.Neg):
        return _strict(node.this)
    if isinstance(node, (exp.Add, exp.Sub, exp.Mul)):
        return _strict(node.this) and _strict(node.expression)
    if isinstance(node, exp.Div):
        divisor = _value(node.expression)
        return _strict(node.this) and divisor is not None and divisor[0] == "number" and divisor[1] != 0
    return False


def _key(column: exp.Column) -> tuple:
    """A column as it is written: two spellings with the same key are surely the same column (a quoted name keeps its case)."""

    parts = []
    for part in (column.args.get("catalog"), column.args.get("db"), column.args.get("table"), column.this):
        if part is None:
            parts.append("")
        elif isinstance(part, exp.Identifier) and not part.quoted:
            parts.append(part.name.lower())
        else:
            parts.append(part.sql())
    return tuple(parts)


def _columns(node: exp.Expression) -> frozenset:
    return frozenset(_key(column) for column in node.find_all(exp.Column))


def _comparison(node: exp.Expression) -> bool:
    if type(node) in _OPPOSITE:
        return _strict(node.this) and _strict(node.expression)
    if type(node) is exp.Between:
        return _strict(node.this) and _strict(node.args["low"]) and _strict(node.args["high"])
    return False


def _total(node: exp.Expression) -> bool:
    """Whether a condition is NULL only when one of its columns is."""

    node = _unparen(node)
    if isinstance(node, (exp.And, exp.Or)):
        return _total(node.this) and _total(node.expression)
    if isinstance(node, exp.Not):
        return _total(node.this)
    return _comparison(node)


def _forced(node: exp.Expression) -> frozenset:
    """Columns that are not NULL whenever the condition is TRUE."""

    node = _unparen(node)
    if isinstance(node, exp.And):
        return _forced(node.this) | _forced(node.expression)
    if isinstance(node, exp.Or):
        return _forced(node.this) & _forced(node.expression)
    if isinstance(node, exp.Not):
        inner = _unparen(node.this)
        return _columns(inner) if _comparison(inner) else frozenset()
    return _columns(node) if _comparison(node) else frozenset()


def negate(node: exp.Expression) -> exp.Expression:
    """``NOT node`` with the negation pushed onto the comparisons (exact with NULL)."""

    node = _unparen(node)
    if isinstance(node, exp.And):
        return exp.Paren(this=exp.Or(this=negate(node.this), expression=negate(node.expression)))
    if isinstance(node, exp.Or):
        return exp.Paren(this=exp.And(this=negate(node.this), expression=negate(node.expression)))
    if isinstance(node, exp.Not):
        inner = _unparen(node.this)
        return inner.copy() if isinstance(inner, (exp.Column, exp.Literal, exp.Boolean)) else exp.Paren(this=inner.copy())
    if type(node) in _OPPOSITE:
        return _OPPOSITE[type(node)](this=node.this.copy(), expression=node.expression.copy())
    if type(node) is exp.Between and not any(value for key, value in node.args.items() if key not in ("this", "low", "high")):
        return exp.Paren(this=exp.Or(this=exp.LT(this=node.this.copy(), expression=node.args["low"].copy()), expression=exp.GT(this=node.this.copy(), expression=node.args["high"].copy())))
    return exp.Not(this=exp.Paren(this=node.copy()))


def _all(terms: list[exp.Expression]) -> exp.Expression:
    result = terms[0]
    for term in terms[1:]:
        result = exp.And(this=result, expression=term)
    return result


def _reading(test: exp.Expression) -> exp.Expression | None:
    """The condition under which ``CASE .. END = label`` is TRUE, or None when ``test`` is not such a comparison."""

    if type(test) is not exp.EQ:
        return None
    left, right = _unparen(test.this), _unparen(test.expression)
    case, label = (left, right) if type(left) is exp.Case else (right, left)
    wanted = _value(label)
    if type(case) is not exp.Case or case.args.get("this") is not None or wanted is None or wanted[0] == "null":
        return None
    arms = [(arm.this, _value(arm.args.get("true"))) for arm in case.args.get("ifs") or []]
    default = case.args.get("default")
    last = _value(default) if default is not None else ("null", None)
    values = [value for _, value in arms] + [last]
    if not arms or any(value is None or value[0] not in (wanted[0], "null") for value in values):
        return None
    if wanted[0] == "text" and not _texts_safe({value[1] for value in values if value[0] == "text"} | {wanted[1]}):
        return None
    conditions = [condition for condition, _ in arms]
    for condition in conditions:
        for node in condition.walk():
            if not isinstance(node, _PLAIN):
                return None
            if isinstance(node, exp.Div):
                # A CASE stops at the first TRUE condition and the reading does not, so a later condition must not
                # be able to fail where an earlier one shields it: only division by a constant that is not zero.
                divisor = _value(node.expression)
                if divisor is None or divisor[0] != "number" or divisor[1] == 0:
                    return None
    if not any(value == wanted for value in values):
        return None
    # every condition reads the same columns, is NULL only when one of them is, and is TRUE only when none of them is
    read = _columns(conditions[0])
    same = bool(read) and all(_columns(condition) == read == _forced(condition) and _total(condition) for condition in conditions)

    def not_true(condition: exp.Expression, known: frozenset) -> exp.Expression:
        if _total(condition) and _columns(condition) <= known:
            return negate(condition)
        return exp.Paren(this=exp.Or(this=negate(condition), expression=exp.Is(this=exp.Paren(this=condition.copy()), expression=exp.Null())))

    terms = []
    for position, (condition, value) in enumerate(arms):
        if value == wanted:
            known = _forced(condition)
            terms.append(_all([exp.Paren(this=condition.copy())] + [not_true(earlier, known) for earlier in conditions[:position]]))
    if last == wanted:
        if same:
            # the columns as the conditions write them (their quoting kept), one test for each
            written: dict[tuple, exp.Column] = {}
            for column in conditions[0].find_all(exp.Column):
                written.setdefault(_key(column), column)
            missing = [exp.Is(this=written[key].copy(), expression=exp.Null()) for key in sorted(written)]
            empty = missing[0]
            for test_ in missing[1:]:
                empty = exp.Or(this=empty, expression=test_)
            terms.append(exp.Or(this=empty, expression=exp.Paren(this=_all([negate(condition) for condition in conditions]))))
        else:
            terms.append(_all([not_true(condition, frozenset()) for condition in conditions]))
    result = exp.Paren(this=terms[0]) if len(terms) > 1 else terms[0]
    for term in terms[1:]:
        result = exp.Or(this=result, expression=exp.Paren(this=term))
    return exp.Paren(this=result)


def _terms(condition: exp.Expression):
    """The terms of the top-level ``AND`` of a condition, without their parentheses."""

    node = _unparen(condition)
    if isinstance(node, exp.And):
        yield from _terms(node.this)
        yield from _terms(node.expression)
    else:
        yield node


def read_labelled_cases(select: exp.Select) -> exp.Select | None:
    """A copy of ``select`` with each such filter term written out (module doc), or None when there is none."""

    if not any(True for _ in select.find_all(exp.Case)):
        return None
    copy = select.copy()
    holders = [copy.args.get("where"), copy.args.get("having")]
    conditions = [holder.this for holder in holders if holder is not None] + [join.args["on"] for join in copy.args.get("joins") or [] if join.args.get("on") is not None]
    changed = False
    for condition in conditions:
        for term in list(_terms(condition)):
            reading = _reading(term)
            if reading is not None:
                term.replace(reading)
                changed = True
    return copy if changed else None
