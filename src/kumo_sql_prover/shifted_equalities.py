"""``x = y - 1``, ``y - x = 1`` and ``x + 1 = y`` are one comparison for whole numbers, written one way.

    f.first = v.day - 1          v.day - f.first = 1          v.day - 1 = f.first          -- f.first + 1 = v.day

**Why it holds.** For whole numbers ``y - n = x``, ``y - x = n`` and ``x + n = y`` say the same thing, and each is
NULL exactly when ``x`` or ``y`` is. Whole-number arithmetic is exact apart from overflow, which is an error and not
another value, and a proof holds where both queries succeed (the convention of ``typed_comparisons``). It is not
true of floating-point numbers, where a sum rounds, nor of text, which MySQL turns into a floating-point number.

In MySQL a ``DATE`` in arithmetic is the whole number ``YYYYMMDD`` (``DATE '2024-03-01' + 1`` is ``20240302``), and a
``DATE`` column compared with a number that is not a constant is compared as that number. So the three spellings
are the same statement about two whole numbers there too. (It is not the day after: at the end of a month
``d + 1`` is not a date, and ``d + 1 = e`` is not ``DATEDIFF(e, d) = 1``. Nothing here reads either as the other.)

**What is read.** An ``=`` whose sides are, in either order, ``x + n`` (or ``n + x``, ``x - n``) and ``y``, or ``y - x``
and ``n``, where ``n`` is a whole-number literal and ``x`` and ``y`` are plain columns written with the alias of a source
of the comparison's own select. Each column must have a known type, and the same kind for both: a whole-number
type in any dialect, or ``DATE`` in MySQL. The type is the declared one for a column of a base table, and for a
column of a derived table the type of the plain column, ``MIN`` or ``MAX`` of a plain column that the derived table
outputs under that name (the least or greatest of some whole numbers is one of them). The comparison is written
``x + n = y`` with ``n`` positive; with ``n = 0`` nothing is rewritten.

A pre-pass of the tenth hardest-cases batch (``shape_fallback.TENTH_BATCH``). ``tests/test_shifted_equalities.py``
replays the spellings on SQLite (whole numbers) and checks the MySQL date reading against values worked by hand.
"""

from __future__ import annotations

import re

from sqlglot import exp

from . import smt_values
from .ast_utils import select_sources


def _unparen(node: exp.Expression | None) -> exp.Expression | None:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _plain(node: exp.Expression | None) -> exp.Column | None:
    node = _unparen(node)
    if isinstance(node, exp.Column) and isinstance(node.this, exp.Identifier) and node.table and not node.args.get("db"):
        return node
    return None


def _whole(node: exp.Expression | None) -> int | None:
    node = _unparen(node)
    sign = 1
    if isinstance(node, exp.Neg):
        sign, node = -1, _unparen(node.this)
    if isinstance(node, exp.Literal) and not node.is_string and smt_values.is_integer_text(node.this):
        return sign * int(node.this)
    return None


def _source_kind(source: exp.Expression, name: str, types: dict, dialect: str, depth: int = 0) -> str | None:
    """``"int"`` or ``"date"`` for the output ``name`` of a source, or None when it is not known to be one."""

    name = name.lower()
    if isinstance(source, exp.Table):
        if source.args.get("db") or source.args.get("catalog"):
            return None
        declared = (types.get(source.name.lower()) or {}).get(name)
        if not declared:
            return None
        if smt_values.type_class(str(declared)) == smt_values.INT64:
            return "int"
        if dialect == "mysql" and re.sub(r"[(<].*", "", str(declared)).strip().upper() == "DATE":
            return "date"
        return None
    if type(source) is not exp.Subquery or type(source.this) is not exp.Select or depth > 4:
        return None
    if source.args.get("alias") is not None and source.args["alias"].args.get("columns"):
        return None
    body = source.this
    if body.args.get("with_") or body.args.get("with"):
        return None
    items = [item for item in body.expressions if (item.alias_or_name or "").lower() == name]
    if len(items) != 1:
        return None
    value = _unparen(items[0].unalias())
    if type(value) in (exp.Min, exp.Max) and not any(v for key, v in value.args.items() if key != "this") and not isinstance(value.parent, exp.Window):
        value = _unparen(value.this)
    return _column_kind(value, body, types, dialect, depth + 1) if isinstance(value, exp.Column) else None


def _column_kind(column: exp.Column, select: exp.Select, types: dict, dialect: str, depth: int = 0) -> str | None:
    if not isinstance(column.this, exp.Identifier) or column.args.get("db"):
        return None
    sources = [s for s in select_sources(select) if isinstance(s, (exp.Table, exp.Subquery)) and s.alias_or_name]
    names = [s.alias_or_name.lower() for s in sources]
    if len(set(names)) != len(names) or len(sources) != len(select_sources(select)):
        return None
    if column.table:
        if column.table.lower() not in names:
            return None
        return _source_kind(sources[names.index(column.table.lower())], column.name, types, dialect, depth)
    if len(sources) == 1:
        return _source_kind(sources[0], column.name, types, dialect, depth)
    return None


def _shift(test: exp.EQ) -> tuple[exp.Column, int, exp.Column] | None:
    """``(x, n, y)`` when ``test`` says ``x + n = y``."""

    for left, right in ((test.this, test.expression), (test.expression, test.this)):
        left, right = _unparen(left), _unparen(right)
        if isinstance(left, (exp.Add, exp.Sub)):
            a, b = _unparen(left.this), _unparen(left.expression)
            y = _plain(right)
            if y is not None:
                if _plain(a) is not None and _whole(b) is not None:
                    return _plain(a), _whole(b) if isinstance(left, exp.Add) else -_whole(b), y
                if isinstance(left, exp.Add) and _plain(b) is not None and _whole(a) is not None:
                    return _plain(b), _whole(a), y
            if isinstance(left, exp.Sub) and _plain(a) is not None and _plain(b) is not None and _whole(right) is not None:
                return _plain(b), _whole(right), _plain(a)  # a - b = n says b + n = a
    return None


def write_shifted_equalities(tree: exp.Expression, types: dict | None, dialect: str) -> tuple[exp.Expression, bool]:
    """``tree`` with each such comparison written ``x + n = y`` with ``n`` positive (module doc); whether it changed."""

    if not types or not any(True for _ in tree.find_all(exp.EQ)):
        return tree, False
    declared = {str(t).lower(): {str(c).lower(): v for c, v in columns.items()} for t, columns in types.items()}
    tree = tree.copy()
    changed = False
    for test in list(tree.find_all(exp.EQ)):
        found = _shift(test)
        if found is None:
            continue
        x, n, y = found
        if n == 0:
            continue
        if n < 0:
            x, n, y = y, -n, x
        select = test.find_ancestor(exp.Select)
        if select is None or select.args.get("with_") or select.args.get("with"):
            continue
        kinds = {_column_kind(x, select, declared, dialect), _column_kind(y, select, declared, dialect)}
        if len(kinds) != 1 or None in kinds:
            continue
        new = exp.EQ(this=exp.Add(this=x.copy(), expression=exp.Literal.number(n)), expression=y.copy())
        if new.sql(dialect="mysql") == test.sql(dialect="mysql"):
            continue
        test.replace(new)
        changed = True
    return tree, changed
