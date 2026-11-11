"""A keyed table joined to a grouped derived table is the grouping of the join.

    SELECT p.id, p.name, g.wins
    FROM (SELECT w.who, COUNT(*) AS wins FROM (..) AS w GROUP BY w.who) AS g
    JOIN players AS p ON g.who = p.id

with ``id`` a declared key of ``players`` is

    SELECT p.id, p.name, COUNT(*) AS wins
    FROM (..) AS w CROSS JOIN players AS p
    WHERE p.id = w.who
    GROUP BY p.id

``eager_aggregation.pull_up_aggregate`` does this when the grouped table reads base tables; this rule reads a
grouped table over any sources (a derived ``UNION ALL``, say), keeps them as they are, and is tried after it.

**The rule.** A select without grouping, aggregates, ``DISTINCT`` or row limits has two sources, a base table
``t`` and a derived table ``g``, in an inner or cross join. ``g`` is a select with a plain ``GROUP BY`` whose
every output is a group key or built from aggregates alone, and every group key is an output. The ``WHERE`` and
``ON`` hold, as top-level ``AND`` parts, an equality ``g.k = t.c`` for every key output of ``g``, and the
columns ``t.c`` so tied contain a declared key of ``t``. Then ``t`` joins the sources of ``g`` under ``g``'s
``WHERE`` and those equalities, the grouping is by the tied columns of ``t``, ``g.k`` is read as ``t.c``, an
aggregate output of ``g`` as its aggregate, and any other condition stays in the ``WHERE``, or becomes a
``HAVING`` when it reads an aggregate output.

**Why it is sound.** ``g`` has one row per group, a group being the rows of its input with one value of the
keys. An equality is TRUE only for equal values that are not NULL, so a row of ``g`` joins the rows of ``t``
whose tied columns hold its key values; they contain a declared key, so that is at most one row of ``t``, and
a row of ``t`` likewise joins at most the one group with its values. The original select therefore returns one
row per pair (group, row of ``t``) that agree. In the rewritten select every input row of such a group joins
that same row of ``t`` and no other, so grouping the joined rows by the tied columns of ``t`` gives one group
per such pair, holding exactly the input rows of the original group, once each: every aggregate (also
``DISTINCT`` ones and ``AVG``) has the value it had. A group with a NULL key, or with no agreeing row of ``t``,
is in neither result. The remaining conditions read only the group's row and its row of ``t``: they hold for
every row of a rewritten group or for none, so filtering the rows before grouping (or the groups after, for a
condition on an aggregate) keeps the same groups. Columns of ``t`` are fixed in a group by its key.

**What is refused.** Outer joins, ``USING``, more than two sources, a select with its own grouping, aggregates,
windows, ``DISTINCT``, ordering or a row limit; a grouped table with ``HAVING``, ``DISTINCT``, a row limit,
grouping sets, windows, an output that is neither a key nor made of aggregates, a hidden group key, or a nested
query outside its ``FROM``; a key that is not tied, tied columns that contain no declared key; a column that
is not qualified by one of the two sources; and a column of the grouped table's input that cannot be attributed
to one of its sources once ``t`` stands next to them.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key
from .ast_utils import FROM_KEY, conjuncts, extended_grouping, select_sources
from .union_all_counts import _BRANCH_ARGS, _all, _plain_joins

_GROUPED_ARGS = _BRANCH_ARGS | {"group"}


def _output_names(query: exp.Expression) -> list[str] | None:
    while isinstance(query, (exp.Subquery, exp.SetOperation)):
        query = query.this
    if not isinstance(query, exp.Select):
        return None
    names = [item.alias_or_name.lower() for item in query.expressions]
    return names if all(names) and not any(isinstance(item, exp.Star) or item.find(exp.Star) is not None and item.alias_or_name == "*" for item in query.expressions) else None


def join_before_grouping(
    select: exp.Expression,
    schema: dict[str, list[str]] | None,
    keys: dict[str, list[tuple[str, ...]]] | None,
) -> exp.Expression | None:
    """The select rewritten as the module docstring describes, or ``None`` when it is not that shape."""

    if not schema or not keys or not isinstance(select, exp.Select):
        return None
    if any(value for key, value in select.args.items() if key not in _BRANCH_ARGS) or not _plain_joins(select):
        return None
    sources = select_sources(select)
    if len(sources) != 2:
        return None
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    tables = [s for s in sources if _base_table(s, known) is not None]
    derived = [s for s in sources if isinstance(s, exp.Subquery)]
    if len(tables) != 1 or len(derived) != 1:
        return None
    table, source = tables[0], derived[0]
    alias = source.args.get("alias")
    if alias is None or not source.alias or alias.args.get("columns") or any(value for key, value in source.args.items() if key not in ("this", "alias")):
        return None
    inner = source.this
    if not isinstance(inner, exp.Select) or any(value for key, value in inner.args.items() if key not in _GROUPED_ARGS):
        return None
    group = inner.args.get("group")
    if group is None or not group.expressions or extended_grouping(group):
        return None
    for node in select.find_all(exp.Select, exp.Subquery, exp.Exists, exp.Window, exp.AggFunc):
        if node is select or node is source or _within(node, source):
            continue
        return None
    t_alias, g_alias = table.alias_or_name.lower(), source.alias.lower()
    if t_alias == g_alias:
        return None
    inner = _qualified_input(inner, known, t_alias)
    if inner is None:
        return None
    group_sql = [key.sql() for key in inner.args["group"].expressions]
    key_outputs: dict[str, exp.Expression] = {}
    aggregate_outputs: dict[str, exp.Expression] = {}
    for item in inner.expressions:
        name, value = item.alias_or_name.lower(), item.unalias()
        if not name or name in key_outputs or name in aggregate_outputs or isinstance(value, exp.Star):
            return None
        if value.sql() in group_sql:
            key_outputs[name] = value
        elif value.find(exp.AggFunc) is not None and all(c.find_ancestor(exp.AggFunc) is not None for c in value.find_all(exp.Column)):
            aggregate_outputs[name] = value
        else:
            return None
    if {value.sql() for value in key_outputs.values()} != set(group_sql):
        return None
    own = [c for c in select.find_all(exp.Column) if not _within(c, source)]
    for column in own:
        if not column.table or column.args.get("db") or column.table.lower() not in (t_alias, g_alias):
            return None
        if column.table.lower() == g_alias and column.name.lower() not in key_outputs and column.name.lower() not in aggregate_outputs:
            return None
        if column.table.lower() == t_alias and column.name.lower() not in known[_table_key(table)]:
            return None
    pool: list[exp.Expression] = []
    for join in select.args.get("joins") or []:
        if join.args.get("on") is not None:
            pool += conjuncts(join.args["on"])
    if select.args.get("where") is not None:
        pool += conjuncts(select.args["where"].this)
    tie: dict[str, exp.Column] = {}
    tied: list[exp.Column] = []
    rest: list[exp.Expression] = []
    for part in pool:
        used = False
        if isinstance(part, exp.EQ):
            for mine, other in ((part.this, part.expression), (part.expression, part.this)):
                if (
                    isinstance(mine, exp.Column)
                    and isinstance(other, exp.Column)
                    and mine.table.lower() == g_alias
                    and mine.name.lower() in key_outputs
                    and other.table.lower() == t_alias
                ):
                    tie.setdefault(mine.name.lower(), other)
                    tied.append((other, key_outputs[mine.name.lower()]))
                    used = True
                    break
        if not used:
            rest.append(part)
    if set(tie) != set(key_outputs):
        return None
    tied_names = {column.name.lower() for column, _ in tied}
    declared = keys.get(_table_key(table)) or next((listed for name, listed in keys.items() if name.lower() == _table_key(table)), [])
    if not any(key and {c.lower() for c in key} <= tied_names for key in declared):
        return None

    def converted(node: exp.Expression) -> exp.Expression:
        node = exp.Paren(this=node.copy())
        for column in list(node.find_all(exp.Column)):
            if column.table.lower() != g_alias:
                continue
            name = column.name.lower()
            value = tie[name].copy() if name in tie else aggregate_outputs[name].copy()
            column.replace(value if isinstance(value, (exp.Column, exp.Func)) else exp.Paren(this=value))
        return node.this

    def reads_aggregate(node: exp.Expression) -> bool:
        return any(c.table.lower() == g_alias and c.name.lower() in aggregate_outputs for c in node.find_all(exp.Column))

    where_parts = conjuncts(inner.args["where"].this) if inner.args.get("where") is not None else []
    where_parts = [part.copy() for part in where_parts]
    where_parts += [exp.EQ(this=column.copy(), expression=value.copy() if isinstance(value, exp.Column) else exp.Paren(this=value.copy())) for column, value in tied]
    where_parts += [converted(part) for part in rest if not reads_aggregate(part)]
    having_parts = [converted(part) for part in rest if reads_aggregate(part)]
    outputs = []
    for item in select.expressions:
        if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
            return None
        value = converted(item.unalias())
        name = item.alias_or_name
        renamed = isinstance(item, exp.Column) and not (isinstance(value, exp.Column) and value.name.lower() == name.lower())
        outputs.append(exp.alias_(value, name, copy=False) if name and (isinstance(item, exp.Alias) or renamed) else value)
    group_by, seen = [], set()
    for column, _ in tied:
        if column.sql().lower() not in seen:
            seen.add(column.sql().lower())
            group_by.append(column.copy())
    result = exp.Select(expressions=outputs)
    result.set(FROM_KEY, inner.args[FROM_KEY].copy())
    result.set("joins", [join.copy() for join in inner.args.get("joins") or []] + [exp.Join(this=table.copy(), kind="CROSS")])
    result.set("where", exp.Where(this=_all(where_parts)))
    result.set("group", exp.Group(expressions=group_by))
    if having_parts:
        result.set("having", exp.Having(this=_all(having_parts)))
    return result


def _within(node: exp.Expression, ancestor: exp.Expression) -> bool:
    parent = node.parent
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.parent
    return False


def _qualified_input(inner: exp.Select, known: dict[str, list[str]], taken: str) -> exp.Select | None:
    """A copy of the grouped select with each of its columns named by its source; ``None`` when one cannot be.

    The table ``taken`` is about to stand next to the sources, so a bare column could become one of its columns.
    """

    inner = inner.copy()
    if not _plain_inner_joins(inner):
        return None
    sources = select_sources(inner)
    columns: dict[str, list[str] | None] = {}
    for item in sources:
        name = item.alias_or_name.lower()
        if not name or name == taken or name in columns:
            return None
        if isinstance(item, exp.Subquery):
            if item.args.get("alias") is None or item.args["alias"].args.get("columns"):
                return None
            columns[name] = _output_names(item.this)
        elif isinstance(item, exp.Table):
            columns[name] = _base_table(item, known)
        else:
            return None
        if columns[name] is None:
            return None
    spelling = {item.alias_or_name.lower(): item.alias_or_name for item in sources}
    for node in inner.find_all(exp.Select, exp.Subquery, exp.Exists, exp.Window):
        if node is inner or any(node is item or _within(node, item) for item in sources):
            continue
        return None
    for column in list(inner.find_all(exp.Column)):
        if any(_within(column, item) for item in sources if isinstance(item, exp.Subquery)):
            continue
        name = column.name.lower()
        if column.args.get("db") or isinstance(column.this, exp.Star):
            return None
        if column.table:
            if name not in (columns.get(column.table.lower()) or ()):
                return None
            continue
        owners = [alias for alias, listed in columns.items() if name in listed]
        if len(owners) != 1:
            return None
        column.replace(exp.column(name, spelling[owners[0]]))
    return inner


def _plain_inner_joins(select: exp.Select) -> bool:
    return _plain_joins(select)
