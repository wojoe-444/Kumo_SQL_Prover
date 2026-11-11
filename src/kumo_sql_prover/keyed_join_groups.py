"""Keyed tables joined to one pre-aggregated table are the aggregate over the flat join, with its tail.

    SELECT c.name, o.k, g.total
    FROM c JOIN o ON c.ck = o.ck
    JOIN (SELECT l.k, SUM(l.q) AS total FROM l GROUP BY l.k HAVING SUM(l.q) > 300) AS g ON o.k = g.k
    ORDER BY o.price DESC LIMIT 100

is

    SELECT c.name, o.k, SUM(l.q) AS total
    FROM c, o, l
    WHERE c.ck = o.ck AND o.k = l.k
    GROUP BY c.ck, c.name, o.k, o.ck, o.price, l.k
    HAVING SUM(l.q) > 300
    ORDER BY o.price DESC LIMIT 100

(TPC-H Q18 and its rewrites). ``eager_aggregation.pull_up_aggregate`` does this for one keyed table, a grouped
table without ``HAVING`` and no tail; ``pull_up_keyed_join`` reads what that rule declines.

Why this is exact. Call the base tables ``T1 .. Tn`` and the grouped table ``G``. A row of the join is one row
``r`` of each ``Ti`` with one row of ``G``, that is, with one group ``κ`` of ``G``'s sources that passes ``G``'s
``HAVING``. The flat join groups by a declared key of every ``Ti`` and by ``G``'s grouping columns, so its groups
are the pairs ``(r, κ)``: distinct rows of a keyed table differ on the key, so no two ``r`` share a group. The
rows of group ``(r, κ)`` are ``r`` with each row of ``G``'s sources that passes ``G``'s ``WHERE`` and has grouping
values ``κ``, each once; the join conditions that do not read an aggregate are constant on a group, so they keep
or drop it whole, as they keep or drop the joined row. An aggregate over ``G``'s sources therefore sees the very
rows ``G`` aggregated, and a condition that reads an aggregate decides in ``HAVING`` what it decided as a filter.
One joined row, one group, same values: the bag of rows before ``ORDER BY`` and ``LIMIT`` is the same.

Conditions, all checked:

* the select has only a select list, sources, joins, ``WHERE``, ``ORDER BY``, ``LIMIT`` and ``OFFSET``: no
  grouping, ``DISTINCT``, aggregate, window or subquery of its own;
* its joins are commas, ``CROSS JOIN`` or inner ``JOIN .. ON`` (no ``USING``, ``NATURAL`` or ``LATERAL``);
* its sources are base tables the schema lists, each with a declared key whose columns are all declared NOT NULL
  (a unique key with a NULL in it does not tell two rows apart), under distinct names, and exactly one
  derived table with an alias and no column list;
* the derived table is a plain grouped select over base tables the schema lists, joined the same way, grouping
  by plain columns, with optional ``WHERE`` and ``HAVING`` and no subquery, window, ``DISTINCT`` or tail; its
  outputs have distinct names and each is a grouping column or an expression whose columns all sit inside
  aggregates;
* every column resolves: qualified with a source's name, or bare and owned by exactly one of the sources in scope
  (in an ``ON``, the sources joined so far); a bare name in ``ORDER BY`` that is an output name is left to mean
  the output;
* a source of the derived table whose name the select already uses gets a fresh alias;
* something is carried over: the grouped table has a ``HAVING``, a condition reads one of its aggregates, or the
  select has an ``ORDER BY``, ``LIMIT`` or ``OFFSET``. A plain pre-aggregated join is left as it is, because the
  other query of a pair may keep the pre-aggregated table or a CTE of it, and reading one side flat would move it
  away from the form the two shared (twelve DSB development pairs, too long for the shape fallback, lost their
  proof on main to this);
* the select is not inside an ``IN`` or ``EXISTS``: the prover reads a membership query without grouping, so there
  the pre-aggregated table is the better spelling.
"""

from __future__ import annotations

import itertools

from sqlglot import exp

from .ast_utils import FROM_KEY, extended_grouping, select_sources
from .unmatched_outer_join import _columns_of, _conjuncts, _inside, _plain_column, _plain_table

_OUTER_ARGS = frozenset({"expressions", "from_", "from", "joins", "where", "order", "limit", "offset"})
_GROUPED_ARGS = frozenset({"expressions", "from_", "from", "joins", "where", "group", "having"})
_FRESH = itertools.count()


def _and_all(parts: list[exp.Expression]) -> exp.Expression:
    parts = [exp.Paren(this=p) if isinstance(p, (exp.Or, exp.Xor)) else p for p in parts]  # they bind looser than AND
    result = parts[0]
    for part in parts[1:]:
        result = exp.And(this=result, expression=part)
    return result


def plain_joins(select: exp.Select) -> bool:
    """Every join of ``select`` is a comma, ``CROSS JOIN`` or inner ``JOIN .. ON``."""

    for join in select.args.get("joins") or []:
        if join.args.get("side") or join.args.get("using") or join.args.get("method"):
            return False
        if (join.args.get("kind") or "").upper() not in ("", "INNER", "CROSS"):
            return False
        if any(v for key, v in join.args.items() if key not in ("this", "kind", "on")):
            return False
        if isinstance(join.this, exp.Lateral):
            return False
    return True


def declared_columns(schema: dict) -> dict[str, set[str]]:
    return {str(t).lower(): {str(c).lower() for c in columns} for t, columns in schema.items()}


def key_of(table: exp.Table, keys: dict, columns: set[str], not_null: dict | None) -> tuple[str, ...] | None:
    """The first declared key of ``table`` whose columns the schema lists and declares NOT NULL.

    A unique key tells rows apart only where its columns have values: two rows ``(1, NULL)`` satisfy
    ``UNIQUE (a, b)``, and a ``GROUP BY a, b`` makes one group of them. So a key with a column that may be NULL does
    not identify a row, and is not used.
    """

    full = ".".join(part.name for part in table.parts).lower()
    never_null = {str(c).lower() for c in ((not_null or {}).get(full) or (not_null or {}).get(table.name.lower()) or ())}
    for key in keys.get(full) or keys.get(table.name.lower()) or []:
        key = tuple(str(c).lower() for c in key)
        if key and set(key) <= columns and set(key) <= never_null:
            return key
    return None


def qualify_local(select: exp.Select, declared: dict[str, set[str]]) -> bool:
    """Qualify every column of a subquery-free select over base tables with its source's name, in place."""

    sources = select_sources(select)
    names = [(s.alias_or_name or "").lower() for s in sources]
    columns = [_columns_of(s, declared) for s in sources]
    for column in list(select.find_all(exp.Column)):
        plain = _plain_column(column)
        if plain is None or plain is not column or len(column.parts) > 2:
            return False
        if column.table:
            if column.table.lower() not in names or column.name.lower() not in columns[names.index(column.table.lower())]:
                return False
            continue
        owners = [index for index, owned in enumerate(columns) if column.name.lower() in owned]
        if len(owners) != 1:
            return False
        source = sources[owners[0]]
        alias = source.args.get("alias")
        column.set("table", (alias.this if alias is not None else source.this).copy())
    return True


def grouped_source(source: exp.Expression, declared: dict[str, set[str]]) -> dict | None:
    """What a plain grouped derived table computes: a qualified copy, its output expressions and grouping columns."""

    if type(source) is not exp.Subquery or not source.alias or type(source.this) is not exp.Select:
        return None
    if source.args["alias"].args.get("columns") or any(v for key, v in source.args.items() if key not in ("this", "alias")):
        return None
    inner = source.this
    if any(value for key, value in inner.args.items() if key not in _GROUPED_ARGS) or not plain_joins(inner):
        return None
    group = inner.args.get("group")
    if group is None or extended_grouping(group) or not group.expressions:
        return None
    if any(isinstance(n, (exp.Query, exp.Subquery, exp.Exists, exp.Window, exp.Star)) and n is not inner and not isinstance(n.parent, exp.Count) for n in inner.walk()):
        return None
    sources = select_sources(inner)
    names = [(s.alias_or_name or "").lower() for s in sources]
    if not sources or not all(_plain_table(s) and _columns_of(s, declared) for s in sources) or not all(names) or len(set(names)) != len(names):
        return None
    inner = inner.copy()
    if not qualify_local(inner, declared):
        return None
    keys = []
    for key in inner.args["group"].expressions:
        column = _plain_column(key)
        if column is None:
            return None
        keys.append(column)
    key_sql = {k.sql().lower() for k in keys}
    outputs: dict[str, exp.Expression] = {}
    for item in inner.expressions:
        name = (item.alias_or_name or "").lower()
        value = item.this if isinstance(item, exp.Alias) else item
        if not name or name in outputs:
            return None
        aggregates = [n for n in value.find_all(exp.AggFunc)]
        if any(isinstance(a.parent, (exp.Filter, exp.WithinGroup)) or any(isinstance(n, exp.AggFunc) and n is not a for n in a.walk()) for a in aggregates):
            return None
        plain = _plain_column(value)
        if plain is not None:
            if plain.sql().lower() not in key_sql:
                return None
        elif not aggregates or any(not any(_inside(column, a) for a in aggregates) for column in value.find_all(exp.Column)):
            return None
        outputs[name] = value
    having = inner.args.get("having")
    if having is not None:
        aggregates = list(having.find_all(exp.AggFunc))
        if any(isinstance(a.parent, (exp.Filter, exp.WithinGroup)) for a in aggregates):
            return None
        for column in having.find_all(exp.Column):
            if not any(_inside(column, a) for a in aggregates) and column.sql().lower() not in key_sql:
                return None
    return {"select": inner, "names": names, "outputs": outputs, "keys": keys}


def rename_sources(inner: exp.Select, names: list[str], used: set[str]) -> None:
    """Give each source of the qualified select ``inner`` whose name is in ``used`` a fresh alias, in place."""

    renamed: dict[str, str] = {}
    for source, name in zip(select_sources(inner), names):
        if name in used:
            renamed[name] = f"kqp{next(_FRESH)}"
            source.set("alias", exp.TableAlias(this=exp.to_identifier(renamed[name])))
    if renamed:
        for column in inner.find_all(exp.Column):
            if column.table.lower() in renamed:
                column.set("table", exp.to_identifier(renamed[column.table.lower()]))


def _owner(column: exp.Column, names: list[str], owned: list[set[str]], visible: int) -> int | None:
    """The index of the source ``column`` reads, among the first ``visible`` sources when it is bare."""

    if _plain_column(column) is not column or len(column.parts) > 2:
        return None
    if column.table:
        if column.table.lower() not in names:
            return None
        index = names.index(column.table.lower())
        return index if column.name.lower() in owned[index] else None
    owners = [index for index in range(visible) if column.name.lower() in owned[index]]
    return owners[0] if len(owners) == 1 else None


def pull_up_keyed_join(select: exp.Select, schema: dict | None, keys: dict | None, not_null: dict | None = None) -> exp.Expression | None:
    """``select`` (keyed tables joined to one grouped derived table) as the aggregate over the flat join, or ``None``."""

    if not schema or not keys or not select.args.get("joins"):
        return None
    if select.find_ancestor(exp.In, exp.Exists) is not None:
        return None  # inside IN / EXISTS the grouped relation stays a derived table: the prover reads no grouping there
    if any(value for key, value in select.args.items() if key not in _OUTER_ARGS) or not plain_joins(select):
        return None
    sources = select_sources(select)
    derived = [index for index, s in enumerate(sources) if type(s) is exp.Subquery]
    if len(derived) != 1 or not all(_plain_table(s) for index, s in enumerate(sources) if index != derived[0]):
        return None
    position = derived[0]
    source = sources[position]
    names = [(s.alias_or_name or "").lower() for s in sources]
    if not all(names) or len(set(names)) != len(names):
        return None
    for node in select.walk():
        if isinstance(node, (exp.Subquery, exp.Exists, exp.Window, exp.AggFunc, exp.Query, exp.Star)) and node is not select and node is not source and not _inside(node, source):
            return None
    declared = declared_columns(schema)
    owned: list[set[str]] = []
    table_keys: dict[int, tuple[str, ...]] = {}
    for index, item in enumerate(sources):
        if index == position:
            owned.append(set())
            continue
        columns = _columns_of(item, declared)
        key = key_of(item, keys, columns, not_null) if columns else None
        if key is None:
            return None
        owned.append(columns)
        table_keys[index] = key
    grouped = grouped_source(source, declared)
    if grouped is None:
        return None
    owned[position] = set(grouped["outputs"])
    output_names = {(item.alias_or_name or "").lower() for item in select.expressions} - {""}

    copy = select.copy()
    new_sources = select_sources(copy)
    new_source = new_sources[position]
    inner = grouped["select"]
    rename_sources(inner, grouped["names"], set(names))
    outputs = grouped["outputs"]
    read: dict[int, list[str]] = {index: [] for index in table_keys}

    def resolve(root: exp.Expression, visible: int, order: bool = False) -> exp.Expression | None:
        """``root`` with the derived table's outputs replaced by what they compute and bare columns qualified."""

        holder = exp.Paren(this=root)
        for column in list(holder.find_all(exp.Column)):
            if order and not column.table and column.name.lower() in output_names:
                continue  # the output of that name, before and after
            index = _owner(column, names, owned, visible)
            if index is None:
                return None
            if index == position:
                value = outputs[column.name.lower()].copy()
                column.replace(value if isinstance(value, (exp.Column, exp.Func)) else exp.Paren(this=value))
                continue
            item = new_sources[index]
            alias = item.args.get("alias")
            column.set("table", (alias.this if alias is not None else item.this).copy())
            if column.name.lower() not in read[index]:
                read[index].append(column.name.lower())
        return holder.this

    where_parts: list[exp.Expression] = []
    having_parts: list[exp.Expression] = []

    def place(condition: exp.Expression, visible: int) -> bool:
        for part in _conjuncts(condition):
            new = resolve(part.copy(), visible)
            if new is None:
                return False
            (having_parts if new.find(exp.AggFunc) is not None else where_parts).append(new)
        return True

    for index, join in enumerate(copy.args.get("joins") or []):
        on = join.args.get("on")
        if on is not None and not place(on, index + 2):
            return None
    where = copy.args.get("where")
    if where is not None and not place(where.this, len(sources)):
        return None
    items = []
    for item in copy.expressions:
        value = item.this if isinstance(item, exp.Alias) else item
        direct = type(value) is exp.Column and _owner(value, names, owned, len(sources)) == position
        new = resolve(value.copy(), len(sources))
        if new is None:
            return None
        if isinstance(item, exp.Alias):
            items.append(exp.alias_(new, item.args["alias"].copy()))
        elif direct:
            items.append(exp.alias_(new, value.this.copy()))
        else:
            items.append(new)
    order = copy.args.get("order")
    if order is not None:
        for ordered in order.expressions:
            new = resolve(ordered.this.copy(), len(sources), order=True)
            if new is None:
                return None
            ordered.set("this", new)
    for key in ("limit", "offset"):
        if copy.args.get(key) is not None and copy.args[key].find(exp.Column) is not None:
            return None

    inner_where = inner.args.get("where")
    where_parts.extend(part.copy() for part in _conjuncts(inner_where.this if inner_where is not None else None))
    for join in inner.args.get("joins") or []:
        where_parts.extend(part.copy() for part in _conjuncts(join.args.get("on")))
    inner_having = inner.args.get("having")
    having_parts.extend(part.copy() for part in _conjuncts(inner_having.this if inner_having is not None else None))
    if not having_parts and not any(copy.args.get(key) is not None for key in ("order", "limit", "offset")):
        # Nothing to carry over: a plain pre-aggregated join is left as the other query of a pair may write it.
        # DSB's ``wss JOIN store JOIN date_dim`` (keys on both) was read flat on one query and not on the CTE form
        # of the other, and a pair too long for the shape fallback lost its proof on main.
        return None

    group_by: list[exp.Expression] = []
    seen: set[str] = set()
    for index in sorted(table_keys):
        item = new_sources[index]
        alias = item.args.get("alias")
        qualifier = alias.this if alias is not None else item.this
        for name in [*table_keys[index], *read[index]]:
            column = exp.Column(this=exp.to_identifier(name), table=qualifier.copy())
            if column.sql().lower() not in seen:
                seen.add(column.sql().lower())
                group_by.append(column)
    for key in grouped["keys"]:
        if key.sql().lower() not in seen:
            seen.add(key.sql().lower())
            group_by.append(key.copy())

    flat = [s.copy() for index, s in enumerate(new_sources) if index != position] + [s.copy() for s in select_sources(inner)]
    result = exp.Select(expressions=items)
    result.set(FROM_KEY, exp.From(this=flat[0]))
    result.set("joins", [exp.Join(this=item) for item in flat[1:]])
    if where_parts:
        result.set("where", exp.Where(this=_and_all(where_parts)))
    result.set("group", exp.Group(expressions=group_by))
    if having_parts:
        result.set("having", exp.Having(this=_and_all(having_parts)))
    for key in ("order", "limit", "offset"):
        if copy.args.get(key) is not None:
            result.set(key, copy.args[key].copy())
    return result
