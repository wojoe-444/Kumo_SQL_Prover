"""``COUNT(DISTINCT x)`` over rows that cannot repeat a value of ``x`` is ``COUNT(x)``.

    SELECT COUNT(DISTINCT f.owner)
    FROM (SELECT owner, MIN(day) AS first FROM visit GROUP BY owner) AS f
    JOIN visit AS v ON v.owner = f.owner AND v.day = f.first + 1          -- COUNT(f.owner)

**Why it holds.** ``COUNT(DISTINCT x)`` counts the different values of ``x`` that are not NULL and ``COUNT(x)`` the
rows on which ``x`` is not NULL. They agree when no two rows of the select's joined sources have the same value of
``x`` that is not NULL. The select joins two sources, ``P`` and ``J``, with an inner join:

* ``P`` has a column ``u`` on which its rows do not repeat: ``P`` is a derived table grouped by exactly that one plain
  column (one row per value), or a base table with a declared one-column key ``(u)`` (a unique key admits several
  NULLs, and neither count reads a NULL).
* A row of ``P`` meets at most one row of ``J``: ``J`` is a base table with a declared key, and the join condition holds
  a top-level conjunct ``J.c = e`` for every column ``c`` of the key, where ``e`` reads no column of ``J``. For one row of
  ``P`` each ``e`` is one value, a matching row of ``J`` has that value in every key column (none of them NULL, because a
  NULL equals nothing), and the key admits one such row. "Has that value" needs ``=`` to tell the key's values apart
  the way the key does, which a comparison between a text column and a number does not (``'5'`` and ``'5.0'`` both
  equal 5, and so do two large whole numbers beside a floating-point one): so ``e`` is a plain column that holds
  the same base column as ``c`` (the group column of ``P`` taken from the same table), or ``e`` is built from plain
  columns, whole-number literals and ``+ - *`` only and either ``c`` is declared ``DATE`` (a date equals one date, one
  number and one string) or ``c`` and every column of ``e`` are whole numbers (declared so, or the ``MIN`` or ``MAX``
  of such a column in a derived table).

So every row of the join comes from a different row of ``P``, and the rows do not repeat a value of ``u`` that is not
NULL. The counted ``x`` is ``P.u`` itself, or a column ``J.c`` with the conjunct ``J.c = P.u`` in the join condition where
both hold the same base column or both are declared whole numbers: on every row of the inner join the two are equal
and not NULL, so they repeat together. ``WHERE`` only removes rows.

**What is read.** A select over exactly two sources joined by one inner ``JOIN .. ON`` (no ``USING``, ``NATURAL``,
``LATERAL``), either order, with no ``GROUP BY`` or window. Each ``COUNT(DISTINCT x)`` of the select with such an ``x``
loses its ``DISTINCT``; nothing else changes. Floating-point columns are left alone.

A narrow retry rule of the tenth hardest-cases batch (``shape_fallback.TENTH_BATCH``).
``tests/test_unique_count_arguments.py`` replays the rewrite on DuckDB.
"""

from __future__ import annotations

import re

from sqlglot import exp

from . import smt_values
from .ast_utils import conjuncts, select_sources
from .shifted_equalities import _source_kind

_ARITHMETIC = (exp.Column, exp.Identifier, exp.Literal, exp.Add, exp.Sub, exp.Mul, exp.Paren, exp.Neg)
_PLAIN_CLAUSES = ("having", "distinct", "qualify", "windows", "limit", "offset", "laterals")


def _unparen(node: exp.Expression | None) -> exp.Expression | None:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _plain(node: exp.Expression | None) -> exp.Column | None:
    node = _unparen(node)
    if isinstance(node, exp.Column) and isinstance(node.this, exp.Identifier) and not node.args.get("db"):
        return node
    return None


def _base(source: exp.Expression, column: str) -> tuple[str, str] | None:
    """``(table, column)`` of the base column a source's output ``column`` holds unchanged, or None."""

    column = column.lower()
    if isinstance(source, exp.Table):
        if source.args.get("db") or source.args.get("catalog"):
            return None
        return source.name.lower(), column
    if type(source) is exp.Subquery and type(source.this) is exp.Select and not (source.args.get("alias") and source.args["alias"].args.get("columns")):
        body = source.this
        inner = select_sources(body)
        if len(inner) != 1 or body.args.get("joins"):
            return None
        for item in body.expressions:
            if (item.alias_or_name or "").lower() != column:
                continue
            value = _plain(item.unalias())
            if value is None or (value.table and value.table.lower() != (inner[0].alias_or_name or "").lower()):
                return None
            return _base(inner[0], value.name)
    return None


def _grouped_column(source: exp.Expression) -> str | None:
    """The output name of the one plain column a derived table is grouped by, or None."""

    if type(source) is not exp.Subquery or type(source.this) is not exp.Select or (source.args.get("alias") and source.args["alias"].args.get("columns")):
        return None
    body = source.this
    group = body.args.get("group")
    if group is None or len(group.expressions) != 1 or any(group.args.get(key) for key in ("rollup", "cube", "grouping_sets", "all", "totals")):
        return None
    if any(body.args.get(key) for key in ("limit", "offset", "qualify")) or len(select_sources(body)) != 1 or body.args.get("joins"):
        return None
    key = _plain(group.expressions[0])
    if key is None:
        return None
    names = [name for item in body.expressions if (name := (item.alias_or_name or "").lower()) and (value := _plain(item.unalias())) is not None and value.name.lower() == key.name.lower() and (not value.table or not key.table or value.table.lower() == key.table.lower())]
    return names[0] if len(names) == 1 else None


def _integer_or_date(types: dict | None, base: tuple[str, str] | None, dates: bool) -> bool:
    if base is None:
        return False
    declared = ((types or {}).get(base[0]) or {}).get(base[1])
    if not declared:
        return False
    return smt_values.type_class(str(declared)) == smt_values.INT64 or (dates and re.sub(r"[(<].*", "", str(declared)).strip().upper() == "DATE")


def drop_needless_count_distinct(select: exp.Select, keys: dict | None, types: dict | None) -> exp.Select | None:
    """``select`` with ``COUNT(DISTINCT x)`` as ``COUNT(x)`` where its rows cannot repeat ``x`` (module doc), or None."""

    joins = select.args.get("joins") or []
    if len(joins) != 1 or select.args.get("group") is not None or any(select.args.get(key) for key in _PLAIN_CLAUSES):
        return None
    counts = [node for node in select.find_all(exp.Count) if isinstance(node.this, exp.Distinct) and node.find_ancestor(exp.Select) is select and not isinstance(node.parent, (exp.Window, exp.Filter))]
    if not counts or any(True for _ in select.find_all(exp.Window)):
        return None
    join = joins[0]
    if join.args.get("on") is None or join.args.get("using") or join.args.get("method") or join.args.get("side") or (join.args.get("kind") or "").upper() not in ("", "INNER") or isinstance(join.this, exp.Lateral):
        return None
    sources = select_sources(select)
    names = [(source.alias_or_name or "").lower() if isinstance(source, (exp.Table, exp.Subquery)) else "" for source in sources]
    if len(sources) != 2 or not all(names) or names[0] == names[1]:
        return None
    tests = []
    for test in conjuncts(join.args["on"]):
        test = _unparen(test)
        if isinstance(test, exp.EQ):
            tests.append((test.this, test.expression))
            tests.append((test.expression, test.this))
    types = {str(t).lower(): {str(c).lower(): v for c, v in columns.items()} for t, columns in (types or {}).items()}
    for p, j in ((0, 1), (1, 0)):
        P, J = sources[p], sources[j]
        unique = _grouped_column(P)
        if unique is None and isinstance(P, exp.Table):
            single = [key for key in (keys or {}).get(P.name.lower(), []) if len(key) == 1]
            unique = str(single[0][0]).lower() if single else None
        if unique is None or not isinstance(J, exp.Table) or J.args.get("db") or J.args.get("catalog"):
            continue
        pinned = None
        for key in (keys or {}).get(J.name.lower(), []):
            if key and all(_pins(tests, names[j], str(column).lower(), sources, names, types) for column in key):
                pinned = key
                break
        if pinned is None:
            continue
        base = _base(P, unique)
        if base is not None and smt_values.type_class(str(((types.get(base[0]) or {}).get(base[1])) or "")) == smt_values.FLOAT64:
            continue
        same = {(names[p], unique)}
        for left, right in tests:
            left, right = _plain(left), _plain(right)
            if left is None or right is None or not left.table or not right.table:
                continue
            if left.table.lower() == names[j] and right.table.lower() == names[p] and right.name.lower() == unique:
                other = _base(J, left.name)
                if other is not None and (other == base or (_integer_or_date(types, other, False) and _integer_or_date(types, base, False))):
                    same.add((names[j], left.name.lower()))
        result = select.copy()
        changed = False
        for count in [node for node in result.find_all(exp.Count) if isinstance(node.this, exp.Distinct) and node.find_ancestor(exp.Select) is result and not isinstance(node.parent, (exp.Window, exp.Filter))]:
            distinct = count.this
            if len(distinct.expressions) != 1 or distinct.args.get("on") is not None:
                continue
            argument = _plain(distinct.expressions[0])
            if argument is None or not argument.table or (argument.table.lower(), argument.name.lower()) not in same:
                continue
            count.set("this", argument.copy())
            changed = True
        if changed:
            return result
    return None


def drop_needless_count_distinct_everywhere(tree: exp.Expression, keys: dict | None, types: dict | None) -> tuple[exp.Expression, bool]:
    """``tree`` with every such select rewritten (innermost first); whether anything changed."""

    if not keys or not any(isinstance(node.this, exp.Distinct) for node in tree.find_all(exp.Count)):
        return tree, False
    changed = False
    tree = tree.copy()
    for select in reversed(list(tree.find_all(exp.Select))):
        new = drop_needless_count_distinct(select, keys, types)
        if new is None:
            continue
        changed = True
        if select is tree:
            tree = new
        else:
            select.replace(new)
    return tree, changed


def _pins(tests, alias: str, column: str, sources, names, types) -> bool:
    """Whether some conjunct ``alias.column = e`` fixes the key column to one value per row of the other source."""

    source = sources[names.index(alias)]
    base = _base(source, column)
    for left, right in tests:
        left = _plain(left)
        if left is None or not left.table or left.table.lower() != alias or left.name.lower() != column:
            continue
        right = _unparen(right)
        reads = list(right.find_all(exp.Column))
        if any(True for _ in right.find_all(exp.Select, exp.Subquery)) or not reads or any(not c.table or c.table.lower() == alias or c.table.lower() not in names for c in reads):
            continue
        plain = _plain(right)
        if plain is not None and base is not None and _base(sources[names.index(plain.table.lower())], plain.name) == base:
            return True
        if not all(isinstance(node, _ARITHMETIC) and not (isinstance(node, exp.Literal) and (node.is_string or not smt_values.is_integer_text(node.this))) for node in right.walk()):
            continue  # one value per row of the other source: columns, whole numbers, + - * and nothing else
        if _integer_or_date(types, base, True) and not _integer_or_date(types, base, False):
            return True  # a date equals one date, one number and one string
        if _integer_or_date(types, base, False) and all(_source_kind(sources[names.index(c.table.lower())], c.name, types, "") == "int" for c in reads):
            return True  # whole numbers on both sides (a number read from text could equal two large whole numbers)
    return False
