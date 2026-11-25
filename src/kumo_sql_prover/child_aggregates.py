"""A parent row's ``SUM``, ``AVG``, ``MIN`` or ``MAX`` over its child rows, read in one spelling.

"Each lead with the average age of the staff who report to them" is written in three ways:

    -- a subquery per row
    SELECT m.staff_id, (SELECT AVG(r.age) FROM staff AS r WHERE r.lead_id = m.staff_id) AS mean_age FROM staff AS m

    -- grouped after the join
    SELECT m.staff_id, AVG(r.age) AS mean_age
    FROM staff AS m LEFT JOIN staff AS r ON r.lead_id = m.staff_id GROUP BY m.staff_id

    -- aggregated first, joined afterwards
    SELECT m.staff_id, g.mean_age
    FROM staff AS m LEFT JOIN (SELECT lead_id, AVG(age) AS mean_age, MAX(age) AS oldest FROM staff GROUP BY lead_id) AS g
      ON g.lead_id = m.staff_id

``algebraic_equivalence._decorrelate_select_list`` already reads the first as a ``LEFT JOIN`` to a table with one row
per ``lead_id`` and one aggregate, and counts alone are read by ``child_counts``. The second met neither, and the third
only when its table held exactly the one aggregate. This module writes the second and the third the way the first is
read, one table per aggregate (hardest-cases batch 32):

    SELECT m.staff_id, kqa0.kqv AS mean_age
    FROM staff AS m LEFT JOIN (SELECT r.lead_id AS kqk0, AVG(r.age) AS kqv FROM staff AS r GROUP BY r.lead_id) AS kqa0
      ON m.staff_id = kqa0.kqk0

**Grouped after the join** (:func:`read_grouped_left_join`). The select reads one declared table (the parent), ``LEFT``
joins one declared table (the child) and groups by columns of the parent that hold a NOT NULL key of it. No two parent
rows share a key value, so a group is one parent row with the child rows the ``ON`` finds for it, or, when it finds
none, that parent row once with NULL in every child column. Fix a parent row ``p`` and call ``C(p)`` the child rows the
``ON`` accepts for it. The ``ON`` is equalities between a child column and a parent column plus tests on the child's own
columns, so ``C(p)`` is the child rows that pass the tests and whose linked columns equal ``p``'s: exactly one group of
the child grouped by its linked columns after the tests (and no group when ``C(p)`` is empty, since a group's key that
equals a value is not NULL and a NULL in ``p``'s linked column equals nothing in both forms).

* ``SUM``, ``AVG``, ``MIN`` and ``MAX`` of an expression built from child columns with ``+``, ``-`` and ``*`` are the
  aggregate over ``C(p)`` when it is not empty. When it is empty the group is the padded row, on which the expression is
  NULL (those operators give NULL for a NULL operand), so the aggregate is NULL: what the ``LEFT JOIN`` to the grouped
  child shows for a parent row without a group.
* ``COUNT`` of such an expression is the count over ``C(p)``, and 0 for the padded row: ``COALESCE(count, 0)``.
* Everything else in the select list reads parent columns, which are the same on every row of the group.

The equality that links the two tables is asked once per joined pair in the flat form and once per group in the other.
That is the same test only when equal linked values group together, which holds between two columns of one type class
(two whole numbers, two texts under the column's collation) and fails between a text and a number (``'1'`` and ``'01'``
both equal 1 and are two groups). So both linked columns must be declared with the same type class, or both undeclared.

**Aggregated first** (:func:`split_joined_aggregates`). A ``LEFT JOIN`` to a grouped derived table on an equality for
every one of its group columns finds at most one of its rows. A table with the same source, tests and grouping and only
one of the aggregates has a row for the same groups, so joining one such table per aggregate that is read shows the same
values, and an aggregate nothing reads needs no join. The table's group columns may only be read by the ``ON``.

**Two spellings of the per-row subquery** (:func:`read_subquery_spellings`), so that it reaches the reading above:

* ``(SELECT ROUND(AVG(r.age)) FROM .. WHERE ..)`` is ``ROUND((SELECT AVG(r.age) FROM .. WHERE ..))``. A select with an
  aggregate and no ``GROUP BY`` returns exactly one row, so the function is applied to the same value either way. Read
  for ``ROUND``, ``ABS``, ``FLOOR``, ``CEIL``, ``COALESCE``/``IFNULL`` and ``NULLIF`` around ``SUM``, ``AVG``, ``MIN``,
  ``MAX`` and ``COUNT``. MySQL does not round a value that passes through a scalar subquery (unlike one a derived table
  stores, see ``derived_rounding``); measured on MySQL 8.0.
* ``(SELECT AVG(r.age) FROM staff AS r WHERE r.lead_id = m.staff_id GROUP BY r.lead_id)`` groups by a column the
  ``WHERE`` sets equal to an outer value, so there is at most one group: no row, read as NULL, when no row passes, which
  is what ``SUM``, ``AVG``, ``MIN`` and ``MAX`` without the ``GROUP BY`` give for no rows, alone or under ``ROUND``,
  ``ABS``, ``FLOOR``, ``CEIL`` and arithmetic (each is NULL for a NULL operand). ``COUNT(*)`` would give 0 there, so it
  is written ``NULLIF(COUNT(*), 0)`` (with a group it is at least 1). ``COALESCE`` around the aggregate, or ``COUNT`` of
  a column, has a value for no rows and is left alone.

**Parents that have a child row** (:func:`read_matched_parents`). Per-row subqueries under ``WHERE EXISTS (a child
row)`` are read as ``LEFT JOIN``s to grouped tables plus that ``EXISTS``. Each kept parent row has child rows, so each
grouped table over the same child, link and tests has exactly one row for it; the inner join of parent and child
grouped by the parent's NOT NULL key has one group per such parent row, made of the same child rows. The select is
written that way (``COUNT(*)`` of such a group is at least 1, so ``COALESCE(count, 0)`` and ``NULLIF(count, 0)`` around
it are the count). ``p.x IN (SELECT c.k FROM child AS c ..)`` as a part of the ``WHERE`` is the same filter with one
more link. Left alone unless every grouped table and the filter read the same child table with the same link and tests.

**Left alone.** An inner, right or full join; more than two sources; a parent without a NOT NULL key in the
``GROUP BY``; a ``GROUP BY`` on anything but parent columns; ``HAVING``, ``DISTINCT``, a window, ``ORDER BY``, ``LIMIT``
or a subquery in the grouped select; ``COUNT(*)`` there (it counts the padded row); an aggregate over a parent column
or over ``COALESCE``, ``CASE``, a division or a function (it need not be NULL on the padded row); a ``WHERE`` that reads
the child; an ``ON`` with anything but column equalities and child tests; linked columns of different type classes.

It runs in the last step of batch 27 only, so a pair an earlier step decides is decided as it was.
"""

from __future__ import annotations

import re

from sqlglot import exp

from . import smt_values
from .ast_utils import FROM_KEY, conjuncts
from .key_fixed_groupings import _Sources, _set_args, _unwrap

NAME = "child_aggregates"
_GROUPED_ARGS = frozenset({"expressions", FROM_KEY, "joins", "where", "group"})
_DERIVED_ARGS = frozenset({"expressions", FROM_KEY, "joins", "where", "group"})
_SUBQUERY_ARGS = frozenset({"expressions", FROM_KEY, "joins", "where", "group"})
_AGGREGATES = (exp.Sum, exp.Avg, exp.Min, exp.Max, exp.Count)
_STRICT = (exp.Add, exp.Sub, exp.Mul, exp.Neg, exp.Paren, exp.Literal, exp.Column, exp.Identifier)
_VALUE_FUNCTIONS = (exp.Round, exp.Abs, exp.Floor, exp.Ceil, exp.Coalesce, exp.Nullif)
_VALUE_NODES = (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Neg, exp.Paren, exp.Literal)
_STRICT_VALUE = (exp.Round, exp.Abs, exp.Floor, exp.Ceil, exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Neg, exp.Paren, exp.Literal)
_NESTED = (exp.Subquery, exp.Select, exp.Exists, exp.Window, exp.SetOperation)
_PREFIX = "kqa"
_KEY, _VALUE = "kqk", "kqv"


def _and_all(parts: list[exp.Expression]) -> exp.Expression:
    result = parts[0]
    for part in parts[1:]:
        result = exp.And(this=result, expression=part)
    return result


def _own_aggregates(node: exp.Expression) -> list[exp.Expression]:
    """The outermost aggregate calls under ``node`` (``node`` itself when it is one)."""

    if isinstance(node, exp.AggFunc):
        return [node]
    found: list[exp.Expression] = []
    for child in node.iter_expressions():
        found += _own_aggregates(child)
    return found


def _plain_aggregate(call: exp.Expression) -> exp.Expression | None:
    """The argument of a ``SUM``/``AVG``/``MIN``/``MAX``/``COUNT`` call with one argument (``DISTINCT`` allowed), else None."""

    if type(call) not in _AGGREGATES or _set_args(call) - {"this", "big_int"}:
        return None
    argument = call.this
    if isinstance(argument, exp.Distinct):
        if _set_args(argument) - {"expressions"} or len(argument.expressions) != 1:
            return None
        argument = argument.expressions[0]
    if argument is None or any(isinstance(n, exp.AggFunc) for n in argument.walk()):
        return None
    return argument


def _fresh(taken: set[str], count: int) -> list[str]:
    names, number = [], 0
    while len(names) < count:
        name = f"{_PREFIX}{number}"
        number += 1
        if name not in taken:
            names.append(name)
    return names


def _same_class(sources: _Sources, left: tuple[str, str], right: tuple[str, str]) -> bool:
    """Whether equal values of the two columns group together: one declared type class, or neither declared."""

    declared = [sources.keyed[source].types.get(column) for source, column in (left, right)]
    if not any(declared):
        return True
    classes = [smt_values.type_class(str(d)) if d else None for d in declared]
    if classes[0] is None or classes[0] != classes[1]:
        return False
    names = [re.sub(r"[(<].*", "", str(d)).strip().upper() for d in declared]
    return classes[0] != smt_values.OTHER or names[0] == names[1]  # a date and a time are both "other"


def read_grouped_left_join(select: exp.Select, schema, keys, not_null, types) -> exp.Select | None:
    """A parent ``LEFT JOIN`` child grouped by the parent's key, written aggregated first (module doc), else ``None``."""

    if not schema or not keys or _set_args(select) - _GROUPED_ARGS or select.args.get(FROM_KEY) is None:
        return None
    group, joins = select.args.get("group"), select.args.get("joins") or []
    if group is None or _set_args(group) - {"expressions"} or len(joins) != 1:
        return None
    join = joins[0]
    if _set_args(join) - {"this", "on", "side", "kind"} or join.args.get("on") is None:
        return None
    if (join.args.get("side") or "").upper() != "LEFT" or (join.args.get("kind") or "").upper() not in ("", "OUTER"):
        return None
    if any(isinstance(node, _NESTED + (exp.Star,)) for node in select.walk() if node is not select):
        return None
    sources = _Sources()
    for source in (select.args[FROM_KEY].this, join.this):
        if type(source) is not exp.Table or not sources.add(source, schema, keys, not_null, types):
            return None
    if len(sources.keyed) != 2:
        return None
    parent, child = list(sources.keyed)
    if any(name.lower().startswith(_PREFIX) for name in sources.columns):
        return None

    def owner(node: exp.Expression) -> str | None:
        """``parent`` or ``child`` when every column under ``node`` belongs to that source and there is one, else None."""

        owners = set()
        for column in node.find_all(exp.Column):
            found = sources.column(column)
            if found is None:
                return None
            owners.add(found[0])
        return owners.pop() if len(owners) == 1 else None

    grouped: set[str] = set()
    for expression in group.expressions:
        found = sources.column(expression)
        if found is None or found[0] != parent:
            return None
        grouped.add(found[1])
    if not any(set(key) <= grouped for key in sources.keyed[parent].keys):
        return None  # Keyed.read keeps only keys whose columns are all NOT NULL

    links: list[tuple[exp.Expression, exp.Expression]] = []  # (child column, parent column)
    tests: list[exp.Expression] = []
    for part in conjuncts(join.args["on"]):
        part = _unwrap(part)
        if any(isinstance(n, exp.AggFunc) for n in part.walk()):
            return None
        if isinstance(part, exp.EQ):
            left, right = sources.column(part.this), sources.column(part.expression)
            if left is not None and right is not None and {left[0], right[0]} == {parent, child}:
                if not _same_class(sources, left, right):
                    return None
                links.append((part.this, part.expression) if left[0] == child else (part.expression, part.this))
                continue
        if owner(part) != child:
            return None
        tests.append(part)
    if not links:
        return None
    where = select.args.get("where")
    if where is not None:
        for part in conjuncts(where.this):
            if any(isinstance(n, exp.AggFunc) for n in part.walk()) or (owner(part) != parent and any(part.find_all(exp.Column))):
                return None

    calls: dict[str, exp.Expression] = {}
    for item in select.expressions:
        inside: set[int] = set()
        for call in _own_aggregates(item):
            argument = _plain_aggregate(call)
            if argument is None or owner(argument) != child or any(not isinstance(n, _STRICT) for n in argument.walk()):
                return None
            calls.setdefault(call.sql(), call)
            inside |= {id(n) for n in call.walk()}
        for column in item.find_all(exp.Column):
            if id(column) not in inside and (sources.column(column) or ("", ""))[0] != parent:
                return None
    if not calls:
        return None

    names = dict(zip(calls, _fresh(set(sources.columns), len(calls))))
    copy = select.copy()
    for item in copy.expressions:
        for call in _own_aggregates(item):
            value: exp.Expression = exp.column(_VALUE, table=names[call.sql()])
            if isinstance(call, exp.Count):
                value = exp.Coalesce(this=value, expressions=[exp.Literal.number(0)])
            if call is item:
                copy.set("expressions", [value if e is item else e for e in copy.expressions])
            else:
                call.replace(value)
    new_joins = []
    for text, call in calls.items():
        name = names[text]
        derived = exp.Select(expressions=[exp.alias_(c.copy(), f"{_KEY}{i}") for i, (c, _) in enumerate(links)] + [exp.alias_(call.copy(), _VALUE)])
        derived.set(FROM_KEY, exp.From(this=join.this.copy()))
        if tests:
            derived.set("where", exp.Where(this=_and_all([t.copy() for t in tests])))
        derived.set("group", exp.Group(expressions=[c.copy() for c, _ in links]))
        on = _and_all([exp.EQ(this=p.copy(), expression=exp.column(f"{_KEY}{i}", table=name)) for i, (_, p) in enumerate(links)])
        new_joins.append(exp.Join(this=exp.Subquery(this=derived, alias=exp.TableAlias(this=exp.to_identifier(name))), side="LEFT", on=on))
    copy.set("joins", new_joins)
    copy.set("group", None)
    return copy


def _output_name(item: exp.Expression) -> str | None:
    if isinstance(item, exp.Alias):
        return item.alias.lower() if isinstance(item.args.get("alias"), exp.Identifier) else None
    item = _unwrap(item)
    return item.name.lower() if isinstance(item, exp.Column) and isinstance(item.this, exp.Identifier) else None


def split_joined_aggregates(select: exp.Select) -> exp.Select | None:
    """A ``LEFT JOIN`` to a grouped derived table with several aggregates, as one join per aggregate read (module doc)."""

    joins = select.args.get("joins") or []
    if select.args.get(FROM_KEY) is None or not joins:
        return None
    taken = {source.alias_or_name.lower() for source in [select.args[FROM_KEY].this] + [j.this for j in joins]}
    if any(name.startswith(_PREFIX) for name in taken):
        return None
    for position, join in enumerate(joins):
        source = join.this
        if _set_args(join) - {"this", "on", "side", "kind"} or join.args.get("on") is None:
            continue
        if (join.args.get("side") or "").upper() != "LEFT" or (join.args.get("kind") or "").upper() not in ("", "OUTER"):
            continue
        if type(source) is not exp.Subquery or _set_args(source) - {"this", "alias"} or type(source.this) is not exp.Select:
            continue
        alias = source.args.get("alias")
        if alias is None or alias.args.get("columns") or not isinstance(alias.this, exp.Identifier):
            continue
        name, inner = alias.name.lower(), source.this
        group = inner.args.get("group")
        if _set_args(inner) - _DERIVED_ARGS or group is None or _set_args(group) - {"expressions"} or not group.expressions:
            continue
        if any(isinstance(node, _NESTED + (exp.Star,)) and not (isinstance(node, exp.Star) and isinstance(node.parent, exp.Count)) for node in inner.walk() if node is not inner):
            continue
        group_texts = [_unwrap(g).sql() for g in group.expressions]
        if any(not isinstance(_unwrap(g), exp.Column) for g in group.expressions):
            continue
        shown: dict[str, int] = {}  # output name of a group column -> its position in GROUP BY
        values: dict[str, exp.Expression] = {}  # output name of an aggregate item -> the item's expression
        readable = True
        for item in inner.expressions:
            output = _output_name(item)
            body = item.this if isinstance(item, exp.Alias) else item
            if output is None or output in shown or output in values:
                readable = False
                break
            if _unwrap(body).sql() in group_texts:
                shown[output] = group_texts.index(_unwrap(body).sql())
            elif _own_aggregates(body) and not any(isinstance(c, exp.Column) and not any(isinstance(a, exp.AggFunc) for a in _ancestors(c, body)) for c in body.find_all(exp.Column)):
                values[output] = body
            else:
                readable = False
                break
        if not readable or len(values) < 2 or set(shown.values()) != set(range(len(group_texts))):
            continue

        # the ON equates every group column with a value that does not read the table, and nothing else
        links: list[tuple[int, exp.Expression]] = []
        for part in conjuncts(join.args["on"]):
            part = _unwrap(part)
            if not isinstance(part, exp.EQ):
                links = []
                break
            for mine, other in ((part.this, part.expression), (part.expression, part.this)):
                mine = _unwrap(mine)
                if isinstance(mine, exp.Column) and mine.table.lower() == name and mine.name.lower() in shown and not _reads(other, name, set(shown) | set(values)):
                    links.append((shown[mine.name.lower()], other))
                    break
            else:
                links = []
                break
        if {index for index, _ in links} != set(range(len(group_texts))):
            continue

        # how the rest of the select reads the table: only its aggregates, by the table's name
        read: list[exp.Column] = []
        fine = True
        for node in select.walk():
            if not isinstance(node, exp.Column) or _within(node, join):
                continue
            if isinstance(node.this, exp.Star):
                fine = False
                break
            if node.table.lower() == name:
                if node.name.lower() not in values or node.find_ancestor(exp.Select) is not select:
                    fine = False
                    break
                read.append(node)
            elif not node.table and node.name.lower() in set(shown) | set(values):
                fine = False
                break
        if not fine or any(isinstance(node, exp.Star) and not isinstance(node.parent, (exp.Count, exp.Column)) for node in select.expressions):
            continue

        wanted = [output for output in values if any(node.name.lower() == output for node in read)]
        fresh = dict(zip(wanted, _fresh(taken, len(wanted))))
        marks = {id(node): node.name.lower() for node in read}
        copy = select.copy()
        copies = [node for node in copy.walk() if isinstance(node, exp.Column)]
        originals = [node for node in select.walk() if isinstance(node, exp.Column)]
        for original, node in zip(originals, copies):
            output = marks.get(id(original))
            if output is not None:
                node.replace(exp.column(_VALUE, table=fresh[output]))
        split = []
        for output in wanted:
            derived = inner.copy()
            derived.set("expressions", [exp.alias_(group.expressions[i].copy(), f"{_KEY}{i}") for i in range(len(group_texts))] + [exp.alias_(values[output].copy(), _VALUE)])
            on = _and_all([exp.EQ(this=other.copy(), expression=exp.column(f"{_KEY}{index}", table=fresh[output])) for index, other in links])
            split.append(exp.Join(this=exp.Subquery(this=derived, alias=exp.TableAlias(this=exp.to_identifier(fresh[output]))), side="LEFT", on=on))
        new_joins = list(copy.args["joins"])
        new_joins[position : position + 1] = split
        copy.set("joins", new_joins or None)
        return copy
    return None


def _ancestors(node: exp.Expression, top: exp.Expression) -> list[exp.Expression]:
    found = []
    while node is not top and node.parent is not None:
        node = node.parent
        found.append(node)
    return found


def _within(node: exp.Expression, top: exp.Expression) -> bool:
    while node is not None:
        if node is top:
            return True
        node = node.parent
    return False


def _reads(node: exp.Expression, name: str, outputs: set[str]) -> bool:
    """Whether ``node`` may read the derived table ``name``: by its name, or by a bare column it shows."""

    if any(isinstance(n, _NESTED + (exp.AggFunc,)) for n in node.walk()):
        return True
    return any(c.table.lower() == name or (not c.table and c.name.lower() in outputs) for c in node.find_all(exp.Column))


def read_subquery_spellings(select: exp.Select) -> exp.Select | None:
    """A per-row aggregate subquery in the select list with a function around its aggregate, or grouped by the column
    its ``WHERE`` fixes, written as the plain aggregate subquery (module doc), else ``None``."""

    if select.args.get(FROM_KEY) is None:
        return None
    for item in select.expressions:
        for subquery in item.find_all(exp.Subquery):
            inner = subquery.this
            if subquery.find_ancestor(exp.Select) is not select or type(inner) is not exp.Select or _set_args(subquery) - {"this"}:
                continue
            if isinstance(subquery.parent, (exp.In, exp.Any, exp.All, exp.Exists, exp.Unnest, exp.Lateral)):
                continue  # a list of rows, not one value
            if _set_args(inner) - _SUBQUERY_ARGS or len(inner.expressions) != 1 or inner.args.get(FROM_KEY) is None:
                continue
            if any(isinstance(node, _NESTED + (exp.Star,)) and not (isinstance(node, exp.Star) and isinstance(node.parent, exp.Count)) for node in inner.walk() if node is not inner):
                continue
            value = inner.expressions[0]
            value = value.this if isinstance(value, exp.Alias) else value
            calls = _own_aggregates(value)
            if not calls or any(type(call) not in _AGGREGATES for call in calls):
                continue
            outside = [node for node in value.walk() if not any(_within(node, call) for call in calls)]
            if any(not isinstance(node, _VALUE_FUNCTIONS + _VALUE_NODES) for node in outside):
                continue  # a column, or a function this module was not checked for, beside the aggregates
            group = inner.args.get("group")
            if group is not None:
                rewritten = _ungrouped(subquery, inner, value, group)
            elif any(isinstance(node, _VALUE_FUNCTIONS) for node in outside):
                rewritten = _lifted(subquery, inner, value, calls)
            else:
                continue
            if rewritten is None:
                continue
            marks = [node for node in select.walk()]
            copy = select.copy()
            for original, node in zip(marks, copy.walk()):
                if original is subquery:
                    node.replace(rewritten)
                    return copy
    return None


def _lifted(subquery: exp.Subquery, inner: exp.Select, value: exp.Expression, calls: list[exp.Expression]) -> exp.Expression:
    """``value`` with each aggregate call replaced by the subquery that computes just that call."""

    def alone(call: exp.Expression) -> exp.Subquery:
        one = inner.copy()
        one.set("expressions", [call.copy()])
        return exp.Subquery(this=one)

    if value in calls:
        return subquery.copy()
    originals = list(value.walk())
    copy = value.copy()
    for original, node in zip(originals, list(copy.walk())):
        if any(original is call for call in calls):
            node.replace(alone(original))
    return exp.Paren(this=copy)


def _ungrouped(subquery: exp.Subquery, inner: exp.Select, value: exp.Expression, group: exp.Group) -> exp.Expression | None:
    """The subquery without a ``GROUP BY`` on columns its ``WHERE`` sets equal to outer values."""

    if _set_args(group) - {"expressions"} or inner.args.get("joins") or inner.args.get("where") is None:
        return None
    source = inner.args[FROM_KEY].this
    if type(source) is not exp.Table or not source.alias_or_name:
        return None
    name = source.alias_or_name.lower()
    fixed: set[str] = set()
    for part in conjuncts(inner.args["where"].this):
        part = _unwrap(part)
        if not isinstance(part, exp.EQ):
            continue
        for mine, other in ((part.this, part.expression), (part.expression, part.this)):
            mine, other = _unwrap(mine), _unwrap(other)
            # an outer value is a column written with another source's name; a bare name could be the table's own
            if isinstance(mine, exp.Column) and mine.table.lower() == name and isinstance(other, exp.Column) and other.table and other.table.lower() != name:
                fixed.add(mine.name.lower())
    for expression in group.expressions:
        expression = _unwrap(expression)
        if not isinstance(expression, exp.Column) or expression.table.lower() != name or expression.name.lower() not in fixed:
            return None
    calls = _own_aggregates(value)
    if type(value) is exp.Count and isinstance(value.this, exp.Star) and not _set_args(value) - {"this", "big_int"}:
        wrap = True
    elif all(type(call) in (exp.Sum, exp.Avg, exp.Min, exp.Max) for call in calls) and all(isinstance(node, _STRICT_VALUE) for node in value.walk() if not any(_within(node, call) for call in calls)):
        wrap = False  # every aggregate is NULL for no rows, and each of these operators is NULL for a NULL operand
    else:
        return None
    one = inner.copy()
    one.set("group", None)
    plain = exp.Subquery(this=one)
    return exp.Nullif(this=plain, expression=exp.Literal.number(0)) if wrap else plain


def _per_group_table(join: exp.Join, parent: str, sources: _Sources, schema, keys, not_null, types):
    """``(alias, child table, child alias, links, tests, value)`` for a ``LEFT JOIN`` to one grouped table of the written
    form: ``(SELECT c.k AS kqk0, <value> AS kqv FROM child AS c [WHERE tests] GROUP BY c.k) AS alias ON p.x = alias.kqk0``."""

    source = join.this
    if _set_args(join) - {"this", "on", "side", "kind"} or join.args.get("on") is None:
        return None
    if (join.args.get("side") or "").upper() != "LEFT" or (join.args.get("kind") or "").upper() not in ("", "OUTER"):
        return None
    if type(source) is not exp.Subquery or _set_args(source) - {"this", "alias"} or type(source.this) is not exp.Select:
        return None
    alias, inner = source.args.get("alias"), source.this
    if alias is None or alias.args.get("columns") or not isinstance(alias.this, exp.Identifier):
        return None
    name, group = alias.name.lower(), inner.args.get("group")
    if _set_args(inner) - {"expressions", FROM_KEY, "where", "group"} or group is None or _set_args(group) - {"expressions"}:
        return None
    if any(isinstance(node, _NESTED) or (isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count)) for node in inner.walk() if node is not inner):
        return None
    child = _Sources()
    if type(inner.args[FROM_KEY].this) is not exp.Table or not child.add(inner.args[FROM_KEY].this, schema, keys, not_null, types) or not child.keyed:
        return None
    child_alias = next(iter(child.keyed))
    items = inner.expressions
    if len(items) != len(group.expressions) + 1 or any(not isinstance(item, exp.Alias) for item in items):
        return None
    shown: list[str] = []
    for position, (item, grouped) in enumerate(zip(items, group.expressions)):
        column, again = child.column(item.this), child.column(grouped)
        if item.alias.lower() != f"{_KEY}{position}" or column is None or column != again:
            return None
        shown.append(column[1])
    value = items[-1]
    if value.alias.lower() != _VALUE or not _own_aggregates(value.this):
        return None
    if any(child.column(c) is None or not any(isinstance(a, exp.AggFunc) for a in _ancestors(c, value.this)) for c in value.this.find_all(exp.Column)):
        return None
    tests = []
    if inner.args.get("where") is not None:
        for part in conjuncts(inner.args["where"].this):
            if any(child.column(c) is None for c in part.find_all(exp.Column)) or any(isinstance(n, exp.AggFunc) for n in part.walk()):
                return None
            tests.append(part)
    links: dict[int, tuple[str, str]] = {}
    for part in conjuncts(join.args["on"]):
        part = _unwrap(part)
        if not isinstance(part, exp.EQ):
            return None
        for mine, other in ((part.this, part.expression), (part.expression, part.this)):
            mine, owner = _unwrap(mine), sources.column(other)
            if isinstance(mine, exp.Column) and mine.table.lower() == name and mine.name.lower().startswith(_KEY) and mine.name[len(_KEY) :].isdigit() and owner is not None and owner[0] == parent:
                position = int(mine.name[len(_KEY) :])
                if position >= len(shown) or position in links:
                    return None
                declared = [sources.keyed[parent].types.get(owner[1]), child.keyed[child_alias].types.get(shown[position])]
                if any(declared) and not _same_declared(declared):
                    return None
                links[position] = owner
                break
        else:
            return None
    if set(links) != set(range(len(shown))):
        return None
    return name, child.keyed[child_alias], child, frozenset((shown[i], links[i][1]) for i in links), tests, value.this


def _same_declared(declared: list) -> bool:
    classes = [smt_values.type_class(str(d)) if d else None for d in declared]
    if classes[0] is None or classes[0] != classes[1]:
        return False
    names = [re.sub(r"[(<].*", "", str(d)).strip().upper() for d in declared]
    return classes[0] != smt_values.OTHER or names[0] == names[1]


def _renamed(node: exp.Expression, reader: _Sources, name: str) -> exp.Expression:
    """A copy of ``node``, whose columns are all of the one table ``reader`` holds, with each written as ``name.column``."""

    copy = node.copy()
    for column in [copy] if isinstance(copy, exp.Column) else list(copy.find_all(exp.Column)):
        fresh = exp.column(column.name, table=name)
        if column is copy:
            return fresh
        column.replace(fresh)
    return copy


def read_matched_parents(select: exp.Select, schema, keys, not_null, types) -> exp.Select | None:
    """Parent rows that have a child row, each with aggregates over its child rows, written as one grouped inner join.

        SELECT m.staff_id, COALESCE(a.kqv, 0), b.kqv
        FROM staff AS m LEFT JOIN (SELECT r.lead_id AS kqk0, COUNT(*) AS kqv FROM staff AS r GROUP BY r.lead_id) AS a ON m.staff_id = a.kqk0
                        LEFT JOIN (SELECT s.lead_id AS kqk0, AVG(s.age) AS kqv FROM staff AS s GROUP BY s.lead_id) AS b ON m.staff_id = b.kqk0
        WHERE EXISTS (SELECT 1 FROM staff AS t WHERE t.lead_id = m.staff_id)

    is what per-row subqueries under a "has a report" filter are read as. Every kept parent row has child rows, so each
    grouped table (same child, same link, same tests as the ``EXISTS``) has exactly one row for it, holding the
    aggregate over those child rows. The inner join of the parent with the child, grouped by the parent's NOT NULL key,
    has one group per such parent row, made of the same child rows:

        SELECT m.staff_id, COUNT(*), AVG(r.age) FROM staff AS m JOIN staff AS r ON r.lead_id = m.staff_id GROUP BY m.staff_id
    """

    if not schema or not keys or _set_args(select) - {"expressions", FROM_KEY, "joins", "where"} or select.args.get(FROM_KEY) is None:
        return None
    joins, where = select.args.get("joins") or [], select.args.get("where")
    if not joins or where is None:
        return None
    sources = _Sources()
    if type(select.args[FROM_KEY].this) is not exp.Table or not sources.add(select.args[FROM_KEY].this, schema, keys, not_null, types) or not sources.keyed:
        return None
    parent = next(iter(sources.keyed))
    if not sources.keyed[parent].keys:
        return None
    tables = [_per_group_table(join, parent, sources, schema, keys, not_null, types) for join in joins]
    if any(table is None for table in tables) or len({table[0] for table in tables}) != len(tables) or parent in {table[0] for table in tables}:
        return None
    first = tables[0]
    row = _fresh({parent} | {table[0] for table in tables}, 1)[0]

    def canonical(reader: _Sources, tests: list[exp.Expression]) -> frozenset[str]:
        return frozenset(_renamed(test, reader, row).sql() for test in tests)

    link, wanted = first[3], canonical(first[2], first[4])
    if any(table[1].name != first[1].name or table[3] != link or canonical(table[2], table[4]) != wanted for table in tables):
        return None

    # the WHERE: one EXISTS over the same child, link and tests, and tests on the parent alone
    kept, found = [], False
    for part in conjuncts(where.this):
        part = _unwrap(part)
        if not found and _same_rows(part, parent, sources, link, wanted, first[1].name, row, schema, keys, not_null, types):
            found = True
            continue
        if any(isinstance(node, _NESTED + (exp.AggFunc,)) for node in part.walk()):
            return None
        if any((sources.column(column) or ("", ""))[0] != parent for column in part.find_all(exp.Column)):
            return None
        kept.append(part)
    if not found:
        return None

    values = {table[0]: _renamed(table[5], table[2], row) for table in tables}
    copy = select.copy()
    for item in copy.expressions:
        if any(isinstance(node, _NESTED + (exp.AggFunc, exp.Star)) for node in item.walk()):
            return None
        for column in list(item.find_all(exp.Column)):
            if column.table.lower() in values:
                if column.name.lower() != _VALUE:
                    return None
                value, holder = values[column.table.lower()].copy(), column.parent
                if isinstance(holder, exp.Coalesce) and holder.this is column and isinstance(value, exp.Count) and len(holder.expressions) == 1:
                    column = holder  # a count of a parent row that has child rows is never NULL
                    holder = column.parent
                counted = isinstance(value, exp.Count) and isinstance(value.this, exp.Star)
                if counted and isinstance(holder, exp.Nullif) and holder.this is column and isinstance(holder.expression, exp.Literal) and holder.expression.sql() == "0":
                    column = holder  # and the count of its child rows is at least 1
                if column is item:
                    return None
                column.replace(value if isinstance(column.parent, (exp.Alias, exp.Select)) else exp.Paren(this=value) if not isinstance(value, exp.Func) else value)
            elif (sources.column(column) or ("", ""))[0] != parent:
                return None
    child_source = exp.Table(this=exp.to_identifier(first[1].table.name), alias=exp.TableAlias(this=exp.to_identifier(row)))
    on = [exp.EQ(this=exp.column(child_column, table=row), expression=exp.column(parent_column, table=parent)) for child_column, parent_column in sorted(link)]
    on += [_renamed(test, first[2], row) for test in first[4]]
    copy.set("joins", [exp.Join(this=child_source, on=_and_all(on))])
    copy.set("where", exp.Where(this=_and_all([part.copy() for part in kept])) if kept else None)
    key = sorted(sources.keyed[parent].keys)[0]
    copy.set("group", exp.Group(expressions=[exp.column(column, table=parent) for column in key]))
    return copy


def _same_rows(test: exp.Expression, parent: str, sources: _Sources, link: frozenset, wanted: frozenset, table: str, row: str, schema, keys, not_null, types) -> bool:
    """Whether ``EXISTS (SELECT .. FROM child AS c WHERE ..)`` asks for a child row with the given link and tests.

    ``p.x IN (SELECT c.k FROM child AS c WHERE ..)`` as a part of the ``WHERE`` is the same question with one more
    link: it keeps the row only when it is TRUE, which is when a listed ``c.k`` equals ``p.x``."""

    listed = None
    if isinstance(test, exp.Exists):
        inner = test.this
    elif isinstance(test, exp.In) and not test.expressions and isinstance(test.args.get("query"), exp.Subquery) and not _set_args(test) - {"this", "query"}:
        inner, listed = test.args["query"].this, sources.column(test.this)
        if listed is None or listed[0] != parent or _set_args(test.args["query"]) - {"this"}:
            return False
    else:
        return False
    if type(inner) is not exp.Select or _set_args(inner) - {"expressions", FROM_KEY, "where"} or inner.args.get(FROM_KEY) is None:
        return False
    if len(inner.expressions) != 1 or any(isinstance(node, _NESTED + (exp.AggFunc,)) for node in inner.walk() if node is not inner):
        return False
    child = _Sources()
    if type(inner.args[FROM_KEY].this) is not exp.Table or not child.add(inner.args[FROM_KEY].this, schema, keys, not_null, types) or not child.keyed:
        return False
    child_alias = next(iter(child.keyed))
    if child.keyed[child_alias].name != table or child_alias == parent:
        return False
    links, tests = set(), []
    if listed is not None:
        shown = _unwrap(inner.expressions[0].this if isinstance(inner.expressions[0], exp.Alias) else inner.expressions[0])
        if not isinstance(shown, exp.Column) or shown.table.lower() not in ("", child_alias) or shown.name.lower() not in child.columns[child_alias]:
            return False
        links.add((shown.name.lower(), listed[1]))
    for part in conjuncts(inner.args["where"].this) if inner.args.get("where") is not None else []:
        part = _unwrap(part)
        if isinstance(part, exp.EQ):
            pair = None
            for mine, other in ((part.this, part.expression), (part.expression, part.this)):
                mine, other = _unwrap(mine), _unwrap(other)
                # the child's column is written with the child's name here; a bare name inside the subquery is the child's too
                if isinstance(mine, exp.Column) and isinstance(other, exp.Column) and other.table.lower() == parent and mine.table.lower() in ("", child_alias) and mine.name.lower() in child.columns[child_alias]:
                    owner = sources.column(other)
                    if owner is not None:
                        pair = (mine.name.lower(), owner[1])
                        break
            if pair is not None:
                links.add(pair)
                continue
        if any(column.table.lower() not in ("", child_alias) or column.name.lower() not in child.columns[child_alias] for column in part.find_all(exp.Column)):
            return False
        tests.append(part)
    return frozenset(links) == link and frozenset(_renamed(test, child, row).sql() for test in tests) == wanted


def read_child_aggregates(select: exp.Select, schema, keys, not_null, types) -> exp.Select | None:
    """``select`` with one of the readings of the module doc applied, else ``None``. The input is not changed."""

    return (
        read_subquery_spellings(select)
        or read_grouped_left_join(select, schema, keys, not_null, types)
        or split_joined_aggregates(select)
        or read_matched_parents(select, schema, keys, not_null, types)
    )
