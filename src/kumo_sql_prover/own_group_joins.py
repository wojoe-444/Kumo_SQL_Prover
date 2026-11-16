"""A ``LEFT JOIN`` from a table to its own groups never pads: it is an inner join.

    SELECT s.employee_id, s.salary, m.top
    FROM salaries AS s
    LEFT JOIN (SELECT company_id, MAX(salary) AS top FROM salaries GROUP BY company_id) AS m
      ON s.company_id = m.company_id

looks every row up in the groups of its own table. A row with a ``company_id`` is in the group of that
``company_id``, so the lookup always finds exactly that group and the ``LEFT`` keeps nothing an inner join would
drop. People write it either way, and the prover read the two as different shapes.

The join is made inner when all of this holds:

* the right side is a derived table over **one base table** ``T`` with a plain ``GROUP BY`` on columns of ``T``
  and nothing that removes a group or a row: no ``WHERE``, ``HAVING``, ``DISTINCT``, ``LIMIT``, window or
  ``QUALIFY`` (a filter could remove the very group the row belongs to);
* the ``ON`` is exactly one equality per grouping column, each between that column as the derived table outputs
  it and **the same column** of a source ``L`` that reads the same base table ``T`` (an extra condition, or a
  comparison with another column, could reject the row's own group);
* ``L`` is a real row wherever the join is evaluated: it is the first source or reached by inner or cross joins
  only, before this join;
* every grouping column is declared NOT NULL on ``T``. A row with a NULL there is in the NULL group, but
  ``NULL = NULL`` is not true, so the ``LEFT JOIN`` pads it and an inner join drops it. A column declared
  floating-point does not count (a NaN has the same problem in some engines).

Without NOT NULL facts nothing fires.
"""

from __future__ import annotations

from sqlglot import exp

from . import smt_values

_BODY_BANNED = ("where", "having", "distinct", "limit", "offset", "qualify", "windows", "with_", "with", "order", "joins", "laterals", "pivots")


def _from(select: exp.Select):
    return select.args.get("from_") or select.args.get("from")


def _base(source: exp.Expression) -> str | None:
    if isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier) and not (source.args.get("db") or source.args.get("catalog") or source.args.get("pivots") or source.args.get("sample") or source.args.get("joins")):
        return source.name.lower()
    return None


def _groups(source: exp.Expression) -> tuple[str, dict[str, str]] | None:
    """``(base table, output name -> grouping column)`` for a derived table that only groups one base table; every
    grouping column must be an output."""

    if not isinstance(source, exp.Subquery) or not source.alias or not isinstance(source.this, exp.Select):
        return None
    if source.args.get("alias") is not None and source.args["alias"].args.get("columns"):
        return None
    body = source.this
    if any(body.args.get(key) for key in _BODY_BANNED) or body.find(exp.Window) is not None:
        return None
    from_ = _from(body)
    table = _base(from_.this) if from_ is not None else None
    group = body.args.get("group")
    if table is None or group is None or any(group.args.get(k) for k in ("rollup", "cube", "grouping_sets", "totals", "all")):
        return None
    alias = from_.this.alias_or_name.lower()
    keys = []
    for key in group.expressions:
        if not isinstance(key, exp.Column) or not isinstance(key.this, exp.Identifier) or (key.table and key.table.lower() != alias):
            return None
        keys.append(key.name.lower())
    if not keys or len(set(keys)) != len(keys):
        return None
    outputs: dict[str, str] = {}
    names = set()
    for item in body.expressions:
        name = item.alias_or_name.lower()
        if not name or name in names:
            return None
        names.add(name)
        value = item.unalias()
        if isinstance(value, exp.Column) and isinstance(value.this, exp.Identifier) and (not value.table or value.table.lower() == alias) and value.name.lower() in keys:
            outputs[name] = value.name.lower()
    if set(outputs.values()) != set(keys):
        return None
    return table, outputs


def inner_join_own_groups(select: exp.Select, not_null: dict[str, frozenset[str]] | None, types: dict[str, dict[str, str]] | None = None) -> exp.Expression | None:
    joins = select.args.get("joins") or []
    from_ = _from(select)
    if not not_null or from_ is None or not joins:
        return None
    declared = {t.lower(): {c.lower() for c in cols} for t, cols in not_null.items()}
    for table, columns in (types or {}).items():  # a NaN is in its group but not equal to itself in every engine
        declared[table.lower()] = declared.get(table.lower(), set()) - {c.lower() for c, t in columns.items() if smt_values.type_class(t) == smt_values.FLOAT64}
    real: dict[str, str] = {}  # alias -> base table, for sources that are real rows so far
    first = _base(from_.this)
    if first is not None:
        real[from_.this.alias_or_name.lower()] = first
    for index, join in enumerate(joins):
        side, kind = (join.side or "").upper(), (join.kind or "").upper()
        plain = not any(join.args.get(k) for k in ("method", "using", "match_condition", "global_", "hint", "directed", "expressions", "pivots"))
        if side == "LEFT" and kind in ("", "OUTER") and plain and join.args.get("on") is not None:
            found = _groups(join.this)
            if found is not None and _ties_own_groups(join, found, real, declared):
                copy = select.copy()
                target = copy.args["joins"][index]
                target.set("side", None)
                target.set("kind", None)
                return copy
        if side or kind not in ("", "INNER", "CROSS") or not plain:
            return None  # sources after an outer join may be padded rows
        base = _base(join.this)
        if base is not None:
            alias = join.this.alias_or_name.lower()
            if alias in real:
                return None
            real[alias] = base
    return None


def _ties_own_groups(join: exp.Join, found: tuple[str, dict[str, str]], real: dict[str, str], declared: dict[str, set[str]]) -> bool:
    from .algebraic_equivalence import _conjuncts

    table, outputs = found
    alias = join.this.alias.lower()
    if alias in real or not set(outputs.values()) <= declared.get(table, set()):
        return False
    tied: dict[str, str] = {}  # grouping column -> the alias of the row that supplies it
    for part in _conjuncts(join.args["on"]):
        part = part.unnest() if isinstance(part, exp.Paren) else part
        if not isinstance(part, exp.EQ):
            return False
        sides = [part.this.unnest() if isinstance(part.this, exp.Paren) else part.this, part.expression.unnest() if isinstance(part.expression, exp.Paren) else part.expression]
        if not all(isinstance(s, exp.Column) and s.table for s in sides):
            return False
        group_side = next((s for s in sides if s.table.lower() == alias), None)
        row_side = next((s for s in sides if s.table.lower() != alias), None)
        if group_side is None or row_side is None:
            return False
        column = outputs.get(group_side.name.lower())
        owner = row_side.table.lower()
        if column is None or real.get(owner) != table or row_side.name.lower() != column or column in tied:
            return False
        tied[column] = owner
    return set(tied) == set(outputs.values()) and len(set(tied.values())) == 1
