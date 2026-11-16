"""Two different rows of a keyed table that agree on some columns, read as the groups holding more than one row.

    SELECT DISTINCT a.email FROM person AS a JOIN person AS b ON a.email = b.email AND a.id <> b.id

lists the addresses that two people share, and so does

    SELECT a.email FROM person AS a WHERE a.email IS NOT NULL GROUP BY a.email HAVING COUNT(*) > 1

The prover compares a join with a join and a grouping with a grouping, so the pair stayed undecided. This rule
writes the first spelling as the second.

**Why the two are the same.** Let ``g`` be the compared columns and ``k`` a column that is declared NOT NULL and
declared a key by itself.

* The join keeps a pair of rows ``(r, s)`` when ``r.g = s.g`` is TRUE on every compared column, which makes
  every one of them not NULL in both rows, and ``r.k <> s.k`` is TRUE. No two rows of the table share a ``k``
  and no ``k`` is NULL, so ``r.k <> s.k`` is TRUE exactly when ``r`` and ``s`` are two different rows (and FALSE
  for a row paired with itself). The pairs kept are therefore the pairs of different rows with equal, not NULL
  ``g``.
* The select outputs exactly the compared columns, and ``DISTINCT`` (or a ``GROUP BY`` of the outputs) keeps
  each value once. So a value ``v`` comes out once when at least two rows hold ``g = v``, and not otherwise.
* ``WHERE g IS NOT NULL GROUP BY g HAVING COUNT(*) > 1`` outputs each not NULL value once when at least two
  rows hold it. The group of NULLs is removed by the ``WHERE``, as the join's comparison removes it: this is
  where ``COUNT(*) > 1`` without the test (which keeps a NULL group of two rows) differs, and it is never
  written here.
* ``r.k < s.k`` or ``r.k > s.k`` in place of ``<>`` changes nothing: two different rows have different keys,
  one of the two orders has the smaller key first, and both rows carry the same ``g``.
* Nothing is kept from an empty table by either query.

**What is read.** A select with exactly two sources, both the same base table with supplied columns under two
different names, joined by a comma or an inner or cross join, whose ``ON`` and ``WHERE`` parts are, together:

* one or more ``a.c = b.c`` (the same column of both sides, each qualified), and
* exactly one ``a.k <> b.k``, ``a.k < b.k`` or ``a.k > b.k`` (either way round), ``k`` being declared NOT NULL
  and a declared key on its own.

No column read by a comparison may be declared a floating-point type: NaN is not equal to itself in a comparison
and is one value in a grouping, so both halves of the argument fail for it.

The outputs are exactly the compared columns, each read from either side, with ``DISTINCT`` or with a
``GROUP BY`` that lists exactly those columns. An outer join is read as well when the ``WHERE`` holds at least
one of the comparisons: each compares a column of one side with a column of the other, so it is not TRUE on a
row padded with NULLs, and the outer join then keeps what the inner join keeps.

**What is refused.** Any other condition (a filter on either side, ``<=``, an ``OR``), a key of several
columns, a key that is not declared NOT NULL, outputs other than the compared columns, a bare column (it is
ambiguous between the two sides), no ``DISTINCT`` or grouping (the join returns a row per pair), ``HAVING``,
windows, ``ORDER BY``, ``LIMIT``, ``USING`` and ``NATURAL``.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key, _unwrap
from .ast_utils import FROM_KEY, conjuncts, extended_grouping, plain_distinct, select_sources

_SELECT_ARGS = frozenset({"expressions", "from", "from_", "joins", "where", "distinct", "group", "comments"})
_JOIN_ARGS = frozenset({"this", "on", "side", "kind", "comments"})
_OUTER = ("LEFT", "RIGHT", "FULL")
_DIFFERENT = (exp.NEQ, exp.LT, exp.GT)
_FLOATS = ("FLOAT", "DOUBLE", "REAL")


def read_repeated_groups(
    tree: exp.Expression,
    schema: dict[str, list[str]] | None,
    not_null: dict[str, frozenset[str]] | None,
    keys: dict[str, list[tuple[str, ...]]] | None,
    types: dict[str, dict[str, str]] | None = None,
) -> exp.Expression:
    """Rewrite, in place, each select described in the module docstring."""

    if not schema or not keys or not not_null:
        return tree
    known = {name.lower(): [c.lower() for c in columns] for name, columns in schema.items()}
    declared = {name.lower(): {c.lower() for c in columns} for name, columns in not_null.items()}
    single = {name.lower(): {key[0].lower() for key in table_keys if len(key) == 1} for name, table_keys in keys.items()}
    floats = {
        name.lower(): {c.lower() for c, kind in columns.items() if str(kind).upper().startswith(_FLOATS)}
        for name, columns in (types or {}).items()
    }
    for select in list(tree.find_all(exp.Select))[::-1]:
        _rewrite(select, known, declared, single, floats)
    return tree


def _rewrite(select: exp.Select, known, declared, single, floats) -> bool:
    if any(value for key, value in select.args.items() if key not in _SELECT_ARGS):
        return False
    group = select.args.get("group")
    distinct = select.args.get("distinct")
    if distinct is not None and not (plain_distinct(select) and not distinct.expressions):
        return False
    if distinct is None and group is None:
        return False
    sources = select_sources(select)
    joins = select.args.get("joins") or []
    if len(sources) != 2 or len(joins) != 1:
        return False
    first, second = sources
    columns = _base_table(first, known)
    if columns is None or _base_table(second, known) is None or _table_key(first) != _table_key(second):
        return False
    a, b = first.alias_or_name.lower(), second.alias_or_name.lower()
    if not a or not b or a == b:
        return False
    join = joins[0]
    if any(value for key, value in join.args.items() if key not in _JOIN_ARGS):
        return False
    side, kind = (join.side or "").upper(), (join.kind or "").upper()
    if side not in ("",) + _OUTER or kind not in ("", "INNER", "CROSS", "OUTER") or (kind == "OUTER" and not side):
        return False
    where = select.args.get("where")
    where_parts = conjuncts(where.this) if where is not None else []
    on_parts = conjuncts(join.args["on"]) if join.args.get("on") is not None else []
    if side and not where_parts:
        return False  # an outer join with nothing to reject its padded rows

    def column_of(node: exp.Expression) -> tuple[str, str] | None:
        """``(side, column)`` for a column of the table qualified by one of its two names."""

        node = _unwrap(node)
        if not isinstance(node, exp.Column) or isinstance(node.this, exp.Star) or node.args.get("db") or node.args.get("catalog"):
            return None
        qualifier, name = node.table.lower(), node.name.lower()
        if qualifier not in (a, b) or name not in columns:
            return None
        return qualifier, name

    def both_sides(part: exp.Expression) -> str | None:
        """The column a comparison reads on both sides, one read from each."""

        left, right = column_of(part.this), column_of(part.expression)
        if left is None or right is None or left[0] == right[0] or left[1] != right[1]:
            return None
        return left[1]

    compared: list[str] = []
    different: list[str] = []
    for part in where_parts + on_parts:
        part = _unwrap(part)
        if type(part) is exp.EQ:
            name = both_sides(part)
            if name is None:
                return False
            if name not in compared:
                compared.append(name)
        elif type(part) in _DIFFERENT:
            name = both_sides(part)
            if name is None:
                return False
            different.append(name)
        else:
            return False
    if not compared or len(different) != 1:
        return False
    table = _table_key(first)
    key = different[0]
    if key in compared or key not in single.get(table, ()) or key not in declared.get(table, ()):
        return False
    if any(name in floats.get(table, ()) for name in compared + [key]):
        return False  # NaN: not equal to itself in a comparison, one value in a grouping
    # the outputs, and the grouping when there is one, are exactly the compared columns
    outputs = []
    for item in select.expressions:
        read = column_of(item.this if isinstance(item, exp.Alias) else item)
        if read is None:
            return False
        outputs.append(read[1])
    if set(outputs) != set(compared):
        return False
    if group is not None:
        if extended_grouping(group) or any(value for name, value in group.args.items() if name != "expressions"):
            return False
        grouped = [column_of(item) for item in group.expressions]
        if None in grouped or {name for _, name in grouped} != set(compared):
            return False
    # --- rewrite ---
    name = first.alias_or_name
    for item in select.expressions:
        column = _unwrap(item.this if isinstance(item, exp.Alias) else item)
        column.set("table", exp.to_identifier(name))
    tests = [exp.Not(this=exp.Is(this=exp.column(c, table=name), expression=exp.Null())) for c in compared if c not in declared.get(table, ())]
    select.set(FROM_KEY, exp.From(this=first.copy()))
    select.set("joins", None)
    select.set("distinct", None)
    select.set("where", exp.Where(this=_all(tests)) if tests else None)
    select.set("group", exp.Group(expressions=[exp.column(c, table=name) for c in compared]))
    select.set("having", exp.Having(this=exp.GT(this=exp.Count(this=exp.Star()), expression=exp.Literal.number(1))))
    return True


def _all(tests: list[exp.Expression]) -> exp.Expression:
    condition = tests[0]
    for test in tests[1:]:
        condition = exp.And(this=condition, expression=test)
    return condition
