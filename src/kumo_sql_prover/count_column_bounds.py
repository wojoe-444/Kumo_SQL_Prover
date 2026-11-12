"""A strict comparison of a derived table's ``COUNT`` column with a whole number, written non-strict.

``g.c > 4`` is ``g.c >= 5`` and ``g.c < 4`` is ``g.c <= 3`` when ``g.c`` is ``COUNT(..) AS c`` of the derived
table ``g``: a count is a whole number, so nothing lies strictly between ``n`` and ``n + 1``. Where the column
is NULL (``g`` on the padded side of an outer join) both spellings are NULL.

``cast_rules`` already makes this change for a ``COUNT(..)`` call compared in place (``HAVING COUNT(*) > 4``).
The normalizer also reads a grouped ``IN`` subquery's ``HAVING`` as a filter over a derived table
(``_grouped_in_to_derived``), and which of the two happens first depends on how the subquery was written: a
``HAVING`` lifted before ``cast_rules`` saw it stays ``g.c > 4``, one lifted after is ``g.c >= 5``. The solver
reads a derived table's column as a real number, for which the two differ, so two queries that group the same
rows did not compare. This rule gives the column the spelling the call gets.

Read only when:

* the comparison is ``>`` or ``<`` between a column and a whole-number literal (either order);
* the column names a derived table in this select's own ``FROM`` or joins, by its alias, or bare when the
  select has that one source;
* that derived table is a plain ``SELECT`` (no set operation) whose output of that name is a ``COUNT`` call
  itself (optionally ``DISTINCT``), with no ``FILTER`` or ``OVER`` around it. ``COUNT(..) + 1``, ``SUM``,
  ``AVG`` and every other number are left alone: the solver's reals keep ``x > 50 OR x <= 50`` complementary.
"""

from __future__ import annotations

import re

from sqlglot import exp

from .ast_utils import select_sources

_LIMIT = 10**15


def _int_literal(node: exp.Expression) -> int | None:
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Neg):
        inner = _int_literal(node.this)
        return None if inner is None else -inner
    if isinstance(node, exp.Literal) and not node.is_string and re.fullmatch(r"\d+", node.name or "") and int(node.name) < _LIMIT:
        return int(node.name)
    return None


def _number(value: int) -> exp.Expression:
    return exp.Neg(this=exp.Literal.number(-value)) if value < 0 else exp.Literal.number(value)


def _count_outputs(source: exp.Expression) -> set[str]:
    """Lower-cased names of the derived table's outputs that are a ``COUNT`` call itself."""

    if not isinstance(source, exp.Subquery) or not isinstance(source.this, exp.Select):
        return set()
    alias = source.args.get("alias")
    if alias is not None and alias.args.get("columns"):
        return set()  # ``AS g(a, b)`` renames the outputs
    inner = source.this
    names: dict[str, list[exp.Expression]] = {}
    for item in inner.expressions:
        name = (item.alias_or_name or "").lower()
        if name:
            names.setdefault(name, []).append(item.this if isinstance(item, exp.Alias) else item)
    found = set()
    for name, values in names.items():
        if len(values) != 1:
            continue  # two outputs of one name: the read is ambiguous
        value = values[0]
        while isinstance(value, exp.Paren):
            value = value.this
        if isinstance(value, exp.Count) and not any(isinstance(n, (exp.Window, exp.Filter)) for n in value.walk()):
            found.add(name)
    return found


def count_column_bounds(select: exp.Select) -> exp.Expression | None:
    """Rewrite the select's own strict comparisons of a derived ``COUNT`` column with a whole number."""

    sources = select_sources(select)
    counts = {}
    for source in sources:
        alias = (source.alias_or_name or "").lower()
        if alias:
            counts[alias] = _count_outputs(source)
    if not any(counts.values()):
        return None
    lone = next(iter(counts)) if len(sources) == 1 and len(counts) == 1 else None
    changed = False
    for node in list(select.find_all(exp.GT, exp.LT)):
        if node.find_ancestor(exp.Select) is not select:
            continue
        left, right = node.this, node.expression
        for value, other, literal_on_right in ((_int_literal(right), left, True), (_int_literal(left), right, False)):
            if value is None:
                continue
            while isinstance(other, exp.Paren):
                other = other.this
            if not isinstance(other, exp.Column) or isinstance(other.this, exp.Star):
                break
            table = other.table.lower() or lone
            if table is None or other.name.lower() not in counts.get(table, ()):
                break
            greater = isinstance(node, exp.GT) == literal_on_right  # "other > n", else "other < n"
            bound = _number(value + 1 if greater else value - 1)
            if literal_on_right:
                node.replace((exp.GTE if greater else exp.LTE)(this=other.copy(), expression=bound))
            else:
                node.replace((exp.LTE if greater else exp.GTE)(this=bound, expression=other.copy()))
            changed = True
            break
    return select if changed else None
