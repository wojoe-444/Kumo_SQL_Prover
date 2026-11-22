"""A window whose partitions hold one row each: its function is read on that row.

    SELECT person, unit FROM (SELECT person, unit, ROW_NUMBER() OVER (PARTITION BY person ORDER BY flag) AS n FROM staff) AS d WHERE d.n = 1
    SELECT person, unit FROM (SELECT person, unit, 1 AS n FROM staff) AS d WHERE d.n = 1

When ``person`` is a declared key of ``staff`` and never NULL, no two rows of the table share a partition, so each
partition is exactly the row the window is computed for. Over one row, with the default frame (which always holds
the current row):

* ``ROW_NUMBER()``, ``RANK()`` and ``DENSE_RANK()`` are 1;
* ``COUNT(*)`` and ``COUNT`` of a constant that is not NULL are 1; ``COUNT(e)`` is 1 when ``e`` is not NULL on the
  row and 0 otherwise;
* ``MIN(e)``, ``MAX(e)``, ``FIRST_VALUE(e)`` and ``LAST_VALUE(e)`` are ``e`` on the row (NULL when ``e`` is NULL, as
  the aggregate of no non-NULL value is), and ``SUM`` of a number written out is that number.

**When the partitions are single rows.** The window's select reads one declared table and nothing else (a
``WHERE`` only removes rows), it has no ``GROUP BY``, ``HAVING`` or aggregate of its own, and the ``PARTITION BY``
lists every column of one declared key of that table whose columns are all declared NOT NULL (a unique column that
may be NULL can repeat NULL, and those rows would share a partition). More expressions in the ``PARTITION BY`` only
split further.

**Left alone.** A window with a frame, a name or ``FILTER``; ``DISTINCT`` in the call other than in ``COUNT``; ``SUM``
of anything but a written number (MySQL sums a text as a number, so the sum is not the value); any other function
(``LAG``, ``NTILE``, ``AVG``, whose results have another type or need a typed NULL); a select with a join, a derived
table, ``GROUP BY``, ``HAVING`` or an aggregate that is not a window's own function; a query that still has a
``WITH`` clause (a name there could hide the declared table).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY
from .window_calls import own_aggregates

NAME = "single_row_partitions"

_ONE = (exp.RowNumber, exp.Rank, exp.DenseRank)
_SAME = (exp.Min, exp.Max, exp.FirstValue, exp.LastValue)


def _lower(name: str | None) -> str:
    return (name or "").lower()


def _declared(mapping: dict | None, table: str):
    mapping = mapping or {}
    if table in mapping:
        return mapping[table]
    return next((value for key, value in mapping.items() if _lower(key) == table), None)


def _table(select: exp.Select, schema: dict | None) -> tuple[exp.Table, frozenset] | None:
    """The one declared table the select reads, and its columns; None for any other FROM."""

    if set(k for k, v in select.args.items() if v) & {"joins", "group", "having", "qualify", "windows", "laterals", "pivots"}:
        return None
    from_ = select.args.get(FROM_KEY)
    table = from_.this if from_ is not None else None
    if not isinstance(table, exp.Table) or set(k for k, v in table.args.items() if v) - {"this", "db", "catalog", "alias"}:
        return None
    columns = _declared(schema, _lower(table.name))
    if not columns:
        return None
    return table, frozenset(_lower(c) for c in columns)


def _partition_columns(window: exp.Window, table: exp.Table, columns: frozenset) -> set[str]:
    found = set()
    for part in window.args.get("partition_by") or []:
        if isinstance(part, exp.Column) and not part.args.get("db") and not part.args.get("catalog") and _lower(part.name) in columns:
            if not part.table or _lower(part.table) == _lower(table.alias_or_name):
                found.add(_lower(part.name))
    return found


def _value(window: exp.Window) -> exp.Expression | None:
    """What the window's function is on a partition of one row, or None when the rule does not read it."""

    if set(k for k, v in window.args.items() if v) - {"this", "partition_by", "order", "over"} or window.args.get("over") not in (None, "OVER"):
        return None  # a frame, a window name, OVER a named window
    call = window.this
    if isinstance(call, _ONE):
        return exp.Literal.number(1) if not any(v for v in call.args.values()) else None
    if isinstance(call, exp.Count):
        if set(k for k, v in call.args.items() if v) - {"this", "big_int"}:
            return None
        counted = call.this
        if isinstance(counted, exp.Distinct):
            if len(counted.expressions) != 1 or counted.args.get("on"):
                return None
            counted = counted.expressions[0]
        if isinstance(counted, exp.Star) or (isinstance(counted, exp.Literal)) or isinstance(counted, exp.Boolean):
            return exp.Literal.number(1)
        if counted is None or isinstance(counted, exp.Null) or not _row_value(counted):
            return None
        return exp.Case(ifs=[exp.If(this=exp.Not(this=exp.Is(this=_wrapped(counted), expression=exp.Null())), true=exp.Literal.number(1))], default=exp.Literal.number(0))
    if isinstance(call, exp.Sum):
        # the sum of one written number is that number; a column could be a text, which SUM reads as a number
        number = call.this
        if set(k for k, v in call.args.items() if v) - {"this"} or not isinstance(number, exp.Literal) or number.is_string:
            return None
        return number.copy()
    if isinstance(call, _SAME):
        if set(k for k, v in call.args.items() if v) - {"this"}:
            return None
        if call.this is None or isinstance(call.this, exp.Distinct) or not _row_value(call.this):
            return None
        return _wrapped(call.this)
    return None


def _row_value(node: exp.Expression) -> bool:
    return not any(isinstance(part, (exp.AggFunc, exp.Window, exp.Subquery, exp.Select, exp.Star)) for part in node.walk())


def _wrapped(node: exp.Expression) -> exp.Expression:
    node = node.copy()
    return node if isinstance(node, (exp.Column, exp.Literal, exp.Paren, exp.Func, exp.Case, exp.Null, exp.Boolean)) else exp.Paren(this=node)


def read_single_row_partitions(tree: exp.Expression, schema: dict | None, keys: dict | None, not_null: dict | None) -> tuple[exp.Expression, bool]:
    """``tree`` with every window over single-row partitions replaced by its value on the row (module doc), and
    whether anything was written. The tree given is not changed."""

    if not schema or not keys or not any(True for _ in tree.find_all(exp.Window)) or any(True for _ in tree.find_all(exp.With)):
        return tree, False
    new = tree.copy()
    wrote = False
    for window in list(new.find_all(exp.Window)):
        if window.find_ancestor(exp.Window) is not None or any(True for inner in window.find_all(exp.Window) if inner is not window):
            continue
        select = window.find_ancestor(exp.Select)
        if select is None:
            continue
        read = _table(select, schema)
        if read is None:
            continue
        table, columns = read
        if own_aggregates(select):
            continue
        name = _lower(table.name)
        never_null = frozenset(_lower(c) for c in _declared(not_null, name) or ())
        partition = _partition_columns(window, table, columns)
        if not any(key and all(_lower(c) in partition and _lower(c) in never_null for c in key) for key in _declared(keys, name) or ()):
            continue
        value = _value(window)
        if value is None:
            continue
        if not isinstance(window.parent, (exp.Alias, exp.Paren)) and not isinstance(value, (exp.Column, exp.Literal, exp.Paren, exp.Func, exp.Case)):
            value = exp.Paren(this=value)
        window.replace(value)
        wrote = True
    return (new, True) if wrote else (tree, False)
