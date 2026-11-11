"""A ``SUM`` of the counts of grouped ``UNION ALL`` branches is the ``SUM`` over the branches' rows.

    SELECT p.id, SUM(w.n) AS wins
    FROM players AS p
    JOIN (SELECT g.first AS who, COUNT(*) AS n FROM games AS g GROUP BY g.first
          UNION ALL
          SELECT g.second, COUNT(g.second) FROM games AS g GROUP BY g.second) AS w ON p.id = w.who
    GROUP BY p.id

counts per column first and adds the counts up afterwards. It is

    SELECT p.id, SUM(w.n) AS wins
    FROM players AS p
    JOIN (SELECT g.first AS who, 1 AS n FROM games AS g
          UNION ALL
          SELECT g.second, CASE WHEN g.second IS NULL THEN 0 ELSE 1 END FROM games AS g) AS w ON p.id = w.who
    GROUP BY p.id

where each branch returns its rows ungrouped, carrying what the row adds to its group's count. The rewritten
select is the shape :mod:`kumo_sql_prover.union_all_counts` reads.

**The rule.** In a select whose joins are inner or cross joins, let ``u`` be a derived ``UNION ALL`` with at
least one *grouped branch*: a select with a plain ``GROUP BY`` over base tables (inner or cross joins, an
optional ``WHERE``), whose every output is either ``COUNT(*)`` / ``COUNT(x)`` or fixed by the group (a group
key, or a column of a table whose declared NOT NULL key is among the key columns). Call a column of ``u`` a
*count column* when some grouped branch computes it with ``COUNT``; every grouped branch must then compute it
with ``COUNT``. If the select reads the count columns only as ``SUM(u.n)`` (the whole argument), and has no
other aggregate, each grouped branch loses its ``GROUP BY`` and writes ``1`` for ``COUNT(*)`` and
``CASE WHEN x IS NULL THEN 0 ELSE 1 END`` for ``COUNT(x)``.

**Why it is sound.** A plain ``GROUP BY`` makes one row per non-empty group, so each row ``b`` of a grouped
branch stands for the rows ``b'`` of the ungrouped branch that fall in its group: at least one, all with the
same values in the columns that are not counts (they are fixed by the group), and with ``n`` of ``b`` equal to
the sum of ``n`` over them (``COUNT(*)`` counts them, ``COUNT(x)`` those whose ``x`` is not NULL). The select
reads only the other columns outside ``SUM(u.n)``, so its ``WHERE``, ``ON`` and group keys treat ``b`` and each
``b'`` alike: a kept ``b`` becomes its ``b'`` rows in the same group of the select, a dropped ``b`` drops them
all, and the select has the same groups (a group that held ``b`` holds some ``b'``). ``SUM(u.n)`` adds, for
each ``b``, either ``n`` or the same total over its ``b'`` rows; counts and the written values are never NULL, so
no row is skipped, and the sum is NULL only for a global aggregate over no rows, on both sides. The other
branches are not touched.

**What is refused.** A grouped branch with ``HAVING``, ``DISTINCT``, ``LIMIT``, grouping sets, windows, a nested
query, ``COUNT(DISTINCT ..)``, another aggregate, or an output the group does not fix; a select that reads a
count column anywhere but as the whole argument of a ``SUM`` (a filter on the counts, ``MAX`` of them, the
count itself as an output), that has another aggregate (``COUNT(*)`` would count rows, not groups), outer
joins, ``USING``, a star, windows or nested queries, or a column that cannot be attributed to one source;
``UNION`` instead of ``UNION ALL``.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key
from .ast_utils import FROM_KEY, extended_grouping, select_sources
from .union_all_counts import _BRANCH_ARGS, _SELECT_ARGS, _fixed, _inside, _integer_value, _plain_joins, _row_value, _union_all_branches

_GROUPED_ARGS = _BRANCH_ARGS | {"group"}


def ungroup_counted_branches(
    select: exp.Expression,
    schema: dict[str, list[str]] | None,
    keys: dict[str, list[tuple[str, ...]]] | None,
    not_null: dict[str, frozenset[str]] | None,
) -> exp.Expression | None:
    """The select with its counted branches ungrouped (see the module docstring), or ``None``."""

    if not schema or not isinstance(select, exp.Select):
        return None
    if any(value for key, value in select.args.items() if key not in _SELECT_ARGS) or not _plain_joins(select):
        return None
    if extended_grouping(select.args.get("group")):
        return None
    distinct = select.args.get("distinct")
    if distinct is not None and distinct.args.get("on"):
        return None
    sources = select_sources(select)
    unions = [s for s in sources if isinstance(s, exp.Subquery)]
    if len(unions) != 1:
        return None
    union = unions[0]
    alias = union.args.get("alias")
    if alias is None or not union.alias or alias.args.get("columns") or any(value for key, value in union.args.items() if key not in ("this", "alias")):
        return None
    branches = _union_all_branches(union.this)
    if branches is None or len(branches) < 2:
        return None
    for node in select.find_all(exp.Select, exp.Subquery, exp.Exists, exp.Window):
        if node is not select and node is not union and not _inside(node, union):
            return None
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    declared = {name.lower(): {c.lower() for c in listed} for name, listed in (not_null or {}).items()}
    declared_keys = {name.lower(): listed for name, listed in (keys or {}).items()}
    width = len(branches[0].expressions)
    names = [item.alias_or_name.lower() for item in branches[0].expressions]
    if any(len(b.expressions) != width for b in branches) or len(set(names)) != width or not all(names):
        return None
    counted: set[int] = set()
    grouped: list[tuple[exp.Select, set[int]]] = []
    for branch in branches:
        if branch.args.get("group") is None:
            continue
        positions = _counted_positions(branch, known, declared_keys, declared)
        if positions is None:
            return None
        counted |= positions
        grouped.append((branch, positions))
    if not grouped or not counted or any(positions != counted for _, positions in grouped):
        return None
    if any(isinstance(item.unalias(), (exp.Count, exp.Sum)) and not isinstance(item, exp.Alias) for item in branches[0].expressions):
        return None  # the select could not name the count
    # how the select reads the union: count columns as SUM(u.n) only, and no other aggregate
    columns: dict[str, list[str]] = {}
    for source in sources:
        name = source.alias_or_name.lower()
        if not name or name in columns:
            return None
        listed = names if source is union else _base_table(source, known)
        if listed is None:
            return None
        columns[name] = listed
    me = union.alias.lower()
    count_names = {names[position] for position in counted}
    outputs = {item.alias.lower() for item in select.expressions if isinstance(item, exp.Alias)}
    def count_column(column: exp.Column, argument: bool = False) -> bool | None:
        """Whether a column of the select is a count column of the union; ``None`` when that cannot be told."""

        name = column.name.lower()
        if column.args.get("db") or column.args.get("catalog"):
            return None
        if column.table:
            if column.table.lower() not in columns:
                return None
            return column.table.lower() == me and name in count_names
        if name not in count_names:
            return False  # whichever source or output it names, it is no count
        if any(name in listed for alias_, listed in columns.items() if alias_ != me):
            return None
        return True if argument or name not in outputs else None  # an aggregate's argument never reads an output by name

    sums = 0
    for node in select.find_all(exp.AggFunc):
        if _inside(node, union):
            continue
        argument = node.this
        if type(node) is not exp.Sum or isinstance(node.parent, exp.Filter) or any(value for key, value in node.args.items() if key != "this"):
            return None
        if not isinstance(argument, exp.Column) or count_column(argument, True) is not True:
            return None
        sums += 1
    if not sums:
        return None
    for column in select.find_all(exp.Column):
        if _inside(column, union) or isinstance(column.parent, exp.Sum):
            continue
        if count_column(column) is not False:
            return None
    copy = select.copy()
    new_union = next(s for s in select_sources(copy) if isinstance(s, exp.Subquery))
    for branch in _union_all_branches(new_union.this):
        if branch.args.get("group") is None:
            continue
        branch.set("group", None)
        for item in branch.expressions:
            call = item.unalias()
            if not isinstance(call, (exp.Count, exp.Sum)):
                continue
            argument = call.this
            if isinstance(call, exp.Sum):
                value: exp.Expression = _wrapped(argument)
            elif argument is None or isinstance(argument, exp.Star):
                value = exp.Literal.number(1)
            else:
                value = exp.Case(ifs=[exp.If(this=exp.Is(this=_wrapped(argument), expression=exp.Null()), true=exp.Literal.number(0))], default=exp.Literal.number(1))
            call.replace(value)
    return copy


def _wrapped(node: exp.Expression) -> exp.Expression:
    node = node.copy()
    return node if isinstance(node, (exp.Column, exp.Literal, exp.Paren)) else exp.Paren(this=node)


def _counted_positions(branch: exp.Select, known, declared_keys, declared) -> set[int] | None:
    """The positions a grouped branch computes with ``COUNT``; ``None`` when the branch is not read."""

    group = branch.args.get("group")
    if any(value for key, value in branch.args.items() if key not in _GROUPED_ARGS) or not _plain_joins(branch):
        return None
    if not group.expressions or extended_grouping(group):
        return None
    if any(node is not branch for node in branch.find_all(exp.Select, exp.Subquery, exp.Exists, exp.Window)):
        return None
    base: dict[str, str] = {}
    for source in select_sources(branch):
        if _base_table(source, known) is None or not source.alias_or_name or source.alias_or_name.lower() in base:
            return None
        base[source.alias_or_name.lower()] = _table_key(source)
    if any(key.find(exp.AggFunc) is not None or not _row_value(key) for key in group.expressions):
        return None
    positions = set()
    for position, item in enumerate(branch.expressions):
        value = item.unalias()
        if isinstance(value, exp.Count):
            if isinstance(value.this, exp.Distinct) or any(v for k, v in value.args.items() if k not in ("this", "big_int")):
                return None
            if value.this is not None and not isinstance(value.this, exp.Star) and not _row_value(value.this):
                return None
            positions.add(position)
        elif type(value) is exp.Sum:
            if any(v for k, v in value.args.items() if k != "this") or not _row_value(value.this) or not _integer_value(value.this):
                return None
            positions.add(position)
        elif value.find(exp.AggFunc) is not None or not _row_value(value) or not _fixed([value], list(group.expressions), declared_keys, declared, base, set()):
            return None
    if branch.args.get("where") is not None and branch.args["where"].find(exp.AggFunc) is not None:
        return None
    return positions
