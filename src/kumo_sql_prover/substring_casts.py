"""A cast of a bounded substring to a string type that is wide enough is the substring.

    CAST(SUBSTRING(c.phone FROM 1 FOR 2) AS VARCHAR(2)) IN ('13', '31')

is

    SUBSTRING(c.phone FROM 1 FOR 2) IN ('13', '31')

(Calcite writes the cast when it rewrites TPC-H Q22; the original has none). ``SUBSTRING(x FROM a FOR n)`` of a
string returns a string of at most ``n`` characters. ``VARCHAR(m)`` with ``m >= n`` keeps every one of them and
pads nothing, and ``VARCHAR``, ``TEXT`` and ``STRING`` without a length keep any string, so the cast returns its
argument. A NULL stays NULL.

Conditions, all checked:

* a plain ``CAST`` (not ``TRY_CAST``, no ``FORMAT``) to ``VARCHAR``, ``TEXT`` or ``STRING``, with no length or one
  literal length ``m``; ``CHAR(m)`` is not read: it pads, and a padded value compares differently in places;
* its argument is ``SUBSTRING`` with a literal length ``n`` and ``0 <= n <= m`` (any ``n`` when the type has no
  length), on a column whose declared type is ``CHAR``, ``VARCHAR``, ``TEXT`` or ``STRING``: a substring of bytes
  is bytes, and that cast is not a no-op.
"""

from __future__ import annotations

import re

from sqlglot import exp

_TEXT = {exp.DataType.Type.VARCHAR, exp.DataType.Type.TEXT}
_STRING_TYPE = re.compile(r"^\s*(CHAR|CHARACTER|BPCHAR|VARCHAR|CHARACTER VARYING|TEXT|STRING)\b", re.I)


def _length(node: exp.Expression | None) -> int | None:
    if type(node) is exp.DataTypeParam:
        node = node.this
    if type(node) is exp.Literal and not node.is_string and node.this.isdigit():
        return int(node.this)
    return None


def _wide_enough(cast: exp.Cast) -> bool:
    to = cast.args.get("to")
    if type(cast) is not exp.Cast or type(to) is not exp.DataType or to.this not in _TEXT or to.args.get("nested"):
        return False
    if any(value for key, value in cast.args.items() if key not in ("this", "to")):
        return False
    value = cast.this
    if type(value) is not exp.Substring or any(v for key, v in value.args.items() if key not in ("this", "start", "length")):
        return False
    bound = _length(value.args.get("length"))
    params = to.expressions or []
    if not params:
        return bound is not None or value.args.get("length") is None
    if len(params) != 1 or bound is None:
        return False
    width = _length(params[0])
    return width is not None and bound <= width


def drop_substring_casts(select: exp.Select, types: dict | None) -> exp.Expression | None:
    """``select`` with every such cast of its own replaced by its argument, or ``None``."""

    if not types:
        return None
    casts = [c for c in select.find_all(exp.Cast) if c.find_ancestor(exp.Select) is select and _wide_enough(c)]
    if not casts:
        return None
    from .algebraic_equivalence import _origin_type

    copy = select.copy()
    changed = False
    for cast in [c for c in copy.find_all(exp.Cast) if c.find_ancestor(exp.Select) is copy and _wide_enough(c)]:
        column = cast.this.this
        if type(column) is not exp.Column:
            continue
        declared = _origin_type(copy, column, types)
        if declared and _STRING_TYPE.match(declared):
            cast.replace(cast.this.copy())
            changed = True
    return copy if changed else None
