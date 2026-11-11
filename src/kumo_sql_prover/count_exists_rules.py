"""A counting subquery tested against zero is an ``EXISTS``.

    SELECT a.id FROM a WHERE (SELECT COUNT(*) FROM b WHERE b.a_id = a.id) > 0
    SELECT a.id FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.a_id = a.id)

A select with ``COUNT(*)`` and no ``GROUP BY`` returns exactly one row, and its count is a number from 0 up, never
NULL. So ``(SELECT COUNT(*) FROM .. WHERE w) > 0`` is TRUE when some row passes ``w`` and FALSE otherwise, which is
what ``EXISTS (SELECT 1 FROM .. WHERE w)`` is; neither is ever NULL, so one replaces the other anywhere in a query.
The provers do not read a scalar subquery inside a comparison ("unsupported: expression Subquery"), and they do
read ``EXISTS``.

* **Some row**: ``n > 0``, ``n >= 1``, ``n != 0`` (and the mirrored ``0 < n``, ``1 <= n``, ``0 != n``) become ``EXISTS``.
* **No row**: ``n = 0``, ``n < 1``, ``n <= 0`` (and mirrored) become ``NOT EXISTS``.
* ``COUNT(x)`` and ``COUNT(DISTINCT x)`` count the rows where ``x`` is not NULL, so the test moves into the
  subquery: ``EXISTS (SELECT 1 FROM .. WHERE w AND x IS NOT NULL)``. Without it the two differ on a group whose
  ``x`` are all NULL.
* **Grouped on the correlated key**: ``(SELECT COUNT(*) FROM b WHERE b.k = a.id GROUP BY b.k) > 0`` has one group
  when a row passes and none otherwise (every grouping column is equated, by a top-level ``WHERE`` conjunct, with
  an expression that reads no source of the subquery, so all passing rows share one value of it). With no group
  the scalar subquery is NULL, not FALSE, so this form is read only where NULL and FALSE do the same: as a
  top-level conjunct of a ``WHERE``, and only for the "some row" tests.

Declined: any other number or operator (``n > 1``, ``n = 1``); a subquery with ``HAVING``, ``DISTINCT``, ``QUALIFY``,
``ORDER BY``, ``LIMIT`` or ``OFFSET`` (``LIMIT 0`` returns no row and makes the count NULL), a set operation, a second
select item, a ``FILTER``, a window, an aggregate other than ``COUNT`` or an aggregate, window or subquery inside
``COUNT(x)``; a grouped subquery anywhere but a top-level ``WHERE`` conjunct, in a "no row" test, or with a grouping
column that is not equated as above (several groups are an error, not a count).
"""

from __future__ import annotations

import re

from sqlglot import exp

FROM_KEY = "from_"
_PLAIN = ("expressions", FROM_KEY, "joins", "where")
_DIGITS = re.compile(r"[0-9]+")
# operator, with the count on the left -> the number that makes it "some row" / "no row"
_SOME = {exp.GT: 0, exp.GTE: 1, exp.NEQ: 0}
_NONE = {exp.EQ: 0, exp.LT: 1, exp.LTE: 0}
_MIRROR = {exp.GT: exp.LT, exp.LT: exp.GT, exp.GTE: exp.LTE, exp.LTE: exp.GTE, exp.EQ: exp.EQ, exp.NEQ: exp.NEQ}


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _conjuncts(node: exp.Expression | None) -> list[exp.Expression]:
    if node is None:
        return []
    node = _unwrap(node)
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _counting(node: exp.Expression) -> tuple[exp.Select, exp.Expression | None] | None:
    """The select of a scalar counting subquery and the counted expression (None for ``COUNT(*)``)."""

    while isinstance(node, exp.Subquery) and isinstance(node.this, exp.Subquery) and not any(v for k, v in node.args.items() if k != "this"):
        node = node.this  # ((SELECT ..))
    if not isinstance(node, exp.Subquery) or any(v for k, v in node.args.items() if k != "this") or not isinstance(node.this, exp.Select):
        return None
    select = node.this
    if any(v for k, v in select.args.items() if k not in (*_PLAIN, "group")) or len(select.expressions) != 1:
        return None
    count = select.expressions[0].unalias()
    if type(count) is not exp.Count or any(v for k, v in count.args.items() if k not in ("this", "big_int")):
        return None
    counted = count.this
    if isinstance(counted, exp.Star):
        return (select, None) if not any(counted.args.values()) else None
    if isinstance(counted, exp.Distinct):
        if len(counted.expressions) != 1 or any(v for k, v in counted.args.items() if k != "expressions"):
            return None
        counted = counted.expressions[0]
    if counted is None or counted.find(exp.AggFunc, exp.Window, exp.Select, exp.Star) is not None:
        return None
    return select, counted


def _one_group(select: exp.Select) -> bool:
    """Whether every grouping column is equated with something the subquery's own sources do not provide."""

    group = select.args["group"]
    if any(v for k, v in group.args.items() if k != "expressions") or not group.expressions:
        return False
    from .ast_utils import select_sources

    own = {(s.alias_or_name or "").lower() for s in select_sources(select)}
    if "" in own:
        return False
    fixed = set()
    for part in _conjuncts(select.args["where"].this if select.args.get("where") is not None else None):
        if not isinstance(part, exp.EQ):
            continue
        for mine, other in ((part.this, part.expression), (part.expression, part.this)):
            mine, other = _unwrap(mine), _unwrap(other)
            if not (isinstance(mine, exp.Column) and mine.table and mine.table.lower() in own):
                continue
            if other.find(exp.Select, exp.AggFunc, exp.Window) is not None:
                continue
            columns = list(other.find_all(exp.Column))
            # the other side reads only columns qualified by something that is not a source of this subquery
            if all(c.table and c.table.lower() not in own for c in columns):
                fixed.add(mine.sql())
    return all(isinstance(key, exp.Column) and key.table and key.sql() in fixed for key in group.expressions)


def _exists(select: exp.Select, counted: exp.Expression | None) -> exp.Exists:
    body = exp.Select(expressions=[exp.Literal.number(1)])
    for key in (FROM_KEY, "joins"):
        value = select.args.get(key)
        if value:
            body.set(key, [j.copy() for j in value] if isinstance(value, list) else value.copy())
    parts = _conjuncts(select.args["where"].this if select.args.get("where") is not None else None)
    parts = [p.copy() for p in parts]
    if counted is not None:
        parts.append(exp.Not(this=exp.Is(this=exp.Paren(this=counted.copy()) if not isinstance(counted, (exp.Column, exp.Paren)) else counted.copy(), expression=exp.Null())))
    if parts:
        condition = parts[0] if len(parts) == 1 else None
        if condition is None:
            condition = exp.Paren(this=parts[0]) if isinstance(parts[0], exp.Or) else parts[0]
            for part in parts[1:]:
                condition = exp.And(this=condition, expression=exp.Paren(this=part) if isinstance(part, exp.Or) else part)
        body.set("where", exp.Where(this=condition))
    return exp.Exists(this=body)


def _number(node: exp.Expression) -> int | None:
    node = _unwrap(node)
    return int(node.this) if isinstance(node, exp.Literal) and not node.is_string and _DIGITS.fullmatch(node.this) else None


def _reading(node: exp.Expression) -> tuple[bool, exp.Select, exp.Expression | None] | None:
    """(whether the test asks for some row, the counting select, the counted expression) for a test this module reads."""

    kind = type(node)
    if kind not in _MIRROR:
        return None
    left, right = _unwrap(node.this), _unwrap(node.expression)
    if _counting(left) is None:
        left, right, kind = right, left, _MIRROR[kind]
    found, number = _counting(left), _number(right)
    if found is None or number is None:
        return None
    if _SOME.get(kind) == number:
        return True, *found
    if _NONE.get(kind) == number:
        return False, *found
    return None


def count_tests_to_exists(tree: exp.Expression) -> exp.Expression:
    """``tree`` with each counting subquery tested against zero written as ``EXISTS`` or ``NOT EXISTS`` (see the module doc)."""

    if not any(isinstance(n.this, exp.Select) for n in tree.find_all(exp.Subquery)):
        return tree
    targets = []
    for node in tree.find_all(*_MIRROR):
        reading = _reading(node)
        if reading is None:
            continue
        some, select, counted = reading
        if select.args.get("group") is not None:
            holder = node.find_ancestor(exp.Where)  # the nearest WHERE: NULL and FALSE both drop the row there
            if not some or holder is None or not any(part is node for part in _conjuncts(holder.this)) or not _one_group(select):
                continue
        targets.append((node, some, select, counted))
    for node, some, select, counted in targets:
        exists = _exists(select, counted)
        replacement = exists if some else exp.Not(this=exists)
        if node is tree:
            tree = replacement
        else:
            node.replace(replacement)
    return tree
