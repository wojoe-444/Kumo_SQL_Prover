"""A join that only asks whether the other table has a match, read as a membership test.

    SELECT a.follower AS follower, COUNT(DISTINCT b.follower) AS num
    FROM follow AS a JOIN follow AS b ON a.follower = b.followee
    GROUP BY a.follower

lists each person who follows someone (a row of ``a``) and has followers (the rows of ``b`` that name them),
with the number of distinct followers. The join repeats each row of ``b`` once for every row of ``a`` that
matches, and nothing the select reads cares about the repeats, so it is

    SELECT b.followee AS follower, COUNT(DISTINCT b.follower) AS num
    FROM follow AS b WHERE b.followee IN (SELECT a.follower FROM follow AS a)
    GROUP BY b.followee

**Why the two are the same.** Group on the join column (``a.x = b.y``). A group exists for the value ``v`` when
some row of ``a`` has ``x = v`` and some row of ``b`` has ``y = v``; a NULL on either side equals nothing and meets
no row. That is the set of values ``b.y`` takes that also appear in ``a.x``, which the ``IN`` keeps (a NULL ``b.y`` gives NULL,
not TRUE, and is dropped as the join drops it). The rows of ``b`` in the group are the rows with ``y = v``, each
repeated as often as ``v`` occurs in ``a.x``; ``COUNT(DISTINCT e)``, ``MIN(e)`` and ``MAX(e)`` read a set of values
and give the same answer on a repeated row.

**What is read.** A select over two base tables joined by an inner ``JOIN .. ON a.x = b.y`` (``ON`` is that one
equality of two plain columns, one from each table), no ``WHERE``, no limit, ``GROUP BY`` one of the two join
columns, a select list and ``HAVING`` that read only that column and ``COUNT(DISTINCT e)``, ``MIN(e)`` or ``MAX(e)``
of columns of the table the aggregates share (the other table, ``a``, is read through the join column only).
The join column is replaced by ``b.y`` everywhere, which is equal to ``a.x`` on every row the join keeps.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key, _unwrap
from .ast_utils import conjuncts, select_sources

_ALLOWED = frozenset({"expressions", "from", "from_", "joins", "where", "group", "having", "order", "distinct", "hint", "comments", "kind", "operation_modifiers"})
_INNER = {None, "", "INNER"}


def read_join_distinct_aggregates(tree: exp.Expression, schema: dict[str, list[str]] | None) -> exp.Expression:
    """Rewrite, in place, each select described in the module docstring."""

    if not schema or not any(True for _ in tree.find_all(exp.Join)):
        return tree
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    for select in list(tree.find_all(exp.Select))[::-1]:
        _rewrite(select, known)
    return tree


def _positive_count(conjunct: exp.Expression, alias: str, owners: dict[str, list[str]]) -> bool:
    """``COUNT(DISTINCT e) > 0`` (or ``>= 1``, ``<> 0``) over a column of the table ``alias``."""

    conjunct = _unwrap(conjunct)
    if isinstance(conjunct, exp.Not):
        inner = _unwrap(conjunct.this)
        if isinstance(inner, exp.EQ):
            conjunct = exp.NEQ(this=inner.this, expression=inner.expression)
    if not isinstance(conjunct, (exp.GT, exp.GTE, exp.NEQ)):
        return False
    count, number = _unwrap(conjunct.this), _unwrap(conjunct.expression)
    if not isinstance(number, exp.Literal) or number.is_string or _aggregate_kind(count) != "count":
        return False
    wanted = {exp.GT: "0", exp.GTE: "1", exp.NEQ: "0"}[type(conjunct)]
    columns = list(count.find_all(exp.Column))
    resolved = [_resolve(c, owners) for c in columns]
    return number.name == wanted and bool(columns) and None not in resolved and {r[0] for r in resolved} == {alias}


def _is_inner(join: exp.Join, select: exp.Select | None = None, owners: dict[str, list[str]] | None = None) -> bool:
    """An inner join, or a left join whose ``HAVING`` needs a counted value from the null-extended side."""

    if join.args.get("using") or join.args.get("method"):
        return False
    side = join.args.get("side")
    if side:
        having = select.args.get("having") if select is not None else None
        if (side.upper() if isinstance(side, str) else side) != "LEFT" or having is None or join.args.get("on") is None:
            return False
        alias = join.this.alias_or_name.lower()
        return any(_positive_count(part, alias, owners) for part in conjuncts(having.this))
    kind = join.args.get("kind")
    return (kind.upper() if isinstance(kind, str) else kind) in _INNER and join.args.get("on") is not None


def _resolve(column: exp.Expression, owners: dict[str, list[str]]) -> tuple[str, str] | None:
    """``(source alias, column name)`` of a plain column of one of the two sources."""

    column = _unwrap(column)
    if not isinstance(column, exp.Column) or isinstance(column.this, exp.Star) or column.args.get("db") or column.args.get("catalog"):
        return None
    name = column.name.lower()
    if column.table:
        alias = column.table.lower()
        return (alias, name) if alias in owners and name in owners[alias] else None
    holders = [alias for alias, columns in owners.items() if name in columns]
    return (holders[0], name) if len(holders) == 1 else None


def _expand_having_aliases(select: exp.Select, owners: dict[str, list[str]]) -> None:
    """``HAVING num > 0`` for a select item ``COUNT(..) AS num`` reads the aggregate (a column of that name wins)."""

    having = select.args.get("having")
    if having is None:
        return
    items = {item.alias.lower(): item.this for item in select.expressions if isinstance(item, exp.Alias) and item.alias}
    taken = {c for columns in owners.values() for c in columns}
    for column in list(having.find_all(exp.Column)):
        name = column.name.lower()
        if not column.table and name in items and name not in taken and _aggregate_kind(_unwrap(items[name])) is not None:
            column.replace(items[name].copy())


def _aggregate_kind(node: exp.Expression) -> str | None:
    if isinstance(node, exp.Count) and isinstance(node.this, exp.Distinct) and len(node.this.expressions) == 1 and not node.this.args.get("on"):
        return "count"
    if isinstance(node, (exp.Min, exp.Max)) and not isinstance(node.this, exp.Distinct) and not any(v for k, v in node.args.items() if k != "this"):
        return "extreme"
    return None


def _rewrite(select: exp.Select, known: dict[str, list[str]]) -> bool:
    having = select.args.get("having")
    saved = having.copy() if having is not None else None
    if _rewrite_select(select, known):
        return True
    if saved is not None:
        select.set("having", saved)  # the alias read is put back when nothing else changes
    return False


def _rewrite_select(select: exp.Select, known: dict[str, list[str]]) -> bool:
    if any(value for key, value in select.args.items() if key not in _ALLOWED):
        return False
    sources = select_sources(select)
    joins = select.args.get("joins") or []
    if len(sources) != 2 or len(joins) != 1:
        return False
    owners: dict[str, list[str]] = {}
    for source in sources:
        columns = _base_table(source, known)
        alias = source.alias_or_name.lower() if columns is not None else ""
        if columns is None or not alias or alias in owners:
            return False
        owners[alias] = columns
    _expand_having_aliases(select, owners)
    if not _is_inner(joins[0], select, owners):
        return False
    on = _unwrap(joins[0].args["on"])
    if not isinstance(on, exp.EQ):
        return False
    left, right = _resolve(on.this, owners), _resolve(on.expression, owners)
    if left is None or right is None or left[0] == right[0]:
        return False
    group = select.args.get("group")
    if group is None or len(group.expressions) != 1 or any(value for key, value in group.args.items() if key != "expressions"):
        return False
    grouped = _resolve(group.expressions[0], owners)
    if grouped not in (left, right):
        return False
    # the aggregates must all read the same table: that is the one kept, the other only answers "is there a match"
    sides = set()
    reads: list[exp.Column] = []
    for node in select.walk():
        if isinstance(node, (exp.Query, exp.Subquery, exp.Window, exp.Star, exp.Exists)) and node is not select:
            return False
        if isinstance(node, exp.Join) or any(isinstance(p, exp.Join) for p in _ancestors(node, select)):
            continue
        if _aggregate_kind(node) is not None:
            columns = [c for c in node.find_all(exp.Column)]
            resolved = [_resolve(c, owners) for c in columns]
            if not columns or None in resolved or len({r[0] for r in resolved}) != 1:
                return False
            sides.add(resolved[0][0])
    if len(sides) > 1:
        return False
    where = select.args.get("where")
    for column in select.find_all(exp.Column):
        if any(isinstance(p, exp.Join) for p in _ancestors(column, select)):
            continue
        if where is not None and (column.parent is where or any(p is where for p in _ancestors(column, select))):
            continue  # a filter on the kept table alone: checked below
        resolved = _resolve(column, owners)
        if resolved is None:
            return False
        inside = any(_aggregate_kind(p) is not None for p in _ancestors(column, select))
        if not inside and resolved not in (left, right):
            return False  # a bare read of a column that is not the join column
        reads.append(column)
    # which table is kept: the one the aggregates read; without aggregates, the one that is not grouped's partner is free
    keep = next(iter(sides)) if sides else right[0]
    kept_join, other_join = (left, right) if left[0] == keep else (right, left)
    for node in select.walk():
        if isinstance(node, (exp.Count, exp.Min, exp.Max)) and _aggregate_kind(node) is None:
            return False
        if isinstance(node, exp.AggFunc) and _aggregate_kind(node) is None:
            return False
    if where is not None:
        filtered = [_resolve(column, owners) for column in where.find_all(exp.Column)]
        if None in filtered or any(r[0] != keep for r in filtered) or any(
            isinstance(node, (exp.Query, exp.Subquery, exp.Window, exp.Exists, exp.AggFunc, exp.Star)) for node in where.walk()
        ):
            return False
    keep_source = next(s for s in sources if s.alias_or_name.lower() == keep)
    other_source = next(s for s in sources if s.alias_or_name.lower() == other_join[0])
    # replace the join column by the kept side's, outside the aggregates
    for column in reads:
        resolved = _resolve(column, owners)
        if any(_aggregate_kind(p) is not None for p in _ancestors(column, select)):
            continue
        if resolved != kept_join:
            replacement = exp.column(kept_join[1], table=keep_source.alias_or_name)
            if column.parent is select and kept_join[1] != column.name.lower():
                replacement = exp.alias_(replacement, column.name, quoted=False)  # the output keeps its name
            column.replace(replacement)
    membership = exp.In(
        this=exp.column(kept_join[1], table=keep_source.alias_or_name),
        query=exp.Subquery(this=exp.select(exp.column(other_join[1], table=other_source.alias_or_name)).from_(other_source.copy())),
    )
    select.set("joins", None)
    select.set("from_" if "from_" in select.args else "from", exp.From(this=keep_source.copy()))
    select.set("where", exp.Where(this=exp.and_(where.this.copy() if where is not None else None, membership)))
    return True


def _ancestors(node: exp.Expression, stop: exp.Expression):
    parent = node.parent
    while parent is not None and parent is not stop:
        yield parent
        parent = parent.parent


def count_distinct_under_key(select: exp.Select, keys: dict[str, list[tuple[str, ...]]] | None, not_null: dict[str, frozenset[str]] | None) -> exp.Expression | None:
    """``COUNT(DISTINCT c)`` of one table's grouped rows is ``COUNT(c)`` when the group columns and ``c`` hold a key.

    Within one group the group columns are fixed, so two rows with the same ``c`` agree on every column of a key
    that lies inside the group columns and ``c``, and are the same row. Each row then has its own ``c``, and
    counting the values once equals counting the rows that have one. Every column of the key must be declared NOT
    NULL: a ``UNIQUE`` key lets two rows hold ``(1, NULL)``.
    """

    group = select.args.get("group")
    sources = select_sources(select)
    if group is None or len(sources) != 1 or select.args.get("joins") or not isinstance(sources[0], exp.Table) or not keys:
        return None
    table = sources[0]
    name = _table_key(table)
    declared = {c.lower() for c in (not_null or {}).get(name, ())}
    usable = [set(k) for k in keys.get(name, []) if k and all(c.lower() in declared for c in k)]
    if not usable:
        return None
    alias = table.alias_or_name.lower()
    grouped: set[str] = set()
    for item in group.expressions:
        item = _unwrap(item)
        if not isinstance(item, exp.Column) or isinstance(item.this, exp.Star) or (item.table and item.table.lower() != alias):
            return None
        grouped.add(item.name.lower())
    usable = [{c.lower() for c in k} for k in keys.get(name, []) if k and all(c.lower() in declared for c in k)]
    changed = False
    for count in list(select.find_all(exp.Count)):
        if count.find_ancestor(exp.Select) is not select or count.find_ancestor(exp.Window):
            continue
        argument = count.this
        if not isinstance(argument, exp.Distinct) or len(argument.expressions) != 1 or argument.args.get("on"):
            continue
        column = _unwrap(argument.expressions[0])
        if not isinstance(column, exp.Column) or isinstance(column.this, exp.Star) or (column.table and column.table.lower() != alias):
            continue
        if any(key <= grouped | {column.name.lower()} for key in usable):
            count.set("this", column.copy())
            changed = True
    return select if changed else None


def rewrite_join_distinct_select(select: exp.Select, schema: dict[str, list[str]] | None) -> exp.Expression | None:
    """The same rewrite for one select met during normalization, after a left join became an inner join."""

    if not schema or not select.args.get("joins"):
        return None
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    copy = select.copy()
    return copy if _rewrite(copy, known) else None
