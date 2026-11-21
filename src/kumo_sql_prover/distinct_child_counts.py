"""Counts of the distinct keys of joined child tables are one count per child.

"Each note with its number of replies and its number of votes" is written with both child tables joined and the
duplicates that makes counted away, or with each child counted on its own:

    SELECT n.id, n.title, COUNT(DISTINCT r.id) AS replies, COUNT(DISTINCT v.id) AS votes
    FROM notes AS n LEFT JOIN replies AS r ON n.id = r.note LEFT JOIN votes AS v ON n.id = v.note
    WHERE n.kind = 1
    GROUP BY n.id, n.title

    SELECT n.id, n.title, COALESCE(a.cnt, 0) AS replies, COALESCE(b.cnt, 0) AS votes
    FROM notes AS n
    LEFT JOIN (SELECT note, COUNT(*) AS cnt FROM replies GROUP BY note) AS a ON n.id = a.note
    LEFT JOIN (SELECT note, COUNT(*) AS cnt FROM votes GROUP BY note) AS b ON n.id = b.note
    WHERE n.kind = 1

The first is read as the second. ``child_counts`` reads one child table, where the joined rows of a parent row are
its child rows and a plain ``COUNT`` counts them; with two child tables the joined rows are the pairs, and only a
count of distinct keys still counts the rows of one child. It also lets the select keep a ``WHERE`` over the parent
columns, which ``child_counts`` does not read; with one child table a plain ``COUNT(c.k)`` is read as well.

Why the two are equal. A **parent row** is a row of the driving table with the one row each lookup finds for it (a
lookup is a join on a declared NOT NULL one-column key of the joined table, so it finds at most one row; a row an
inner lookup finds nothing for is gone in both forms, and a ``LEFT`` lookup pads). ``WHERE`` reads parent columns
only, so it keeps or drops a parent row with all of its joined rows. Each child is ``LEFT`` joined on one equality
between one of its columns and a parent column (and tests on its own columns), so the joined rows of a parent row
are the combinations of its matching rows of each child, a child with no match standing as one padded row: at least
one row, and every matching child row appears in some of them. The ``GROUP BY`` names parent columns only, among
them a declared NOT NULL key of the driving table, so each parent row is exactly one group. ``COUNT(DISTINCT c.k)``
over that group, ``k`` a declared NOT NULL one-column key of the child, is the number of its matching child rows:
each has its own value of ``k``, never NULL, and the padded row counts nothing. With one child table the joined rows
of a parent row are its matching child rows or one padded row, so ``COUNT(c.k)`` without ``DISTINCT`` is that number
too; with two it would count pairs, and is not read. In the second form the derived table
has one row per value of the child's linked column with a matching row, holding their number; the left join on the
parent column finds that row or pads (also for a NULL parent value, which no group equals), and ``COALESCE(.., 0)``
is the same number. That join finds at most one row, so the select has one row per parent row, as the grouping does.

**What is read** (anything else is left alone): a select with a select list, ``FROM``, joins, an optional ``WHERE``
and a ``GROUP BY`` of plain columns, and nothing else (no ``DISTINCT``, ``HAVING``, window, tail, grouping sets);
plain tables the schema lists, each under its own name; joins with ``ON`` only; every aggregate is
``COUNT(DISTINCT c.k)`` as above (or ``COUNT(c.k)`` with one child), with no ``FILTER``; every column names its table
or belongs to one table only; every other column in the select list, ``WHERE`` and ``GROUP BY`` is a parent column;
no nested query, no volatile call. A child whose ``ON`` reads another child (a grandchild) is not read. The linked
columns of a child and its parent must not be declared with types of different families: ``=`` between a number and
a text converts one of them, and two texts it calls equal to one number are two groups of the counted table. A
``GROUP BY`` name that is also a select alias of something else is not read (dialects differ on which it means), and
inside a larger query every count needs its alias, since the rewritten column would have another name.

A rule of the fifteenth hardest-cases batch (``shape_fallback.FIFTEENTH_BATCH``): it fires only in the last step of
a proof, on a pair every earlier step left unproved. ``tests/test_distinct_child_counts.py`` replays the reading on
DuckDB and shows each condition is needed.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY
from .keyed_join_groups import _and_all, declared_columns, key_of
from .unmatched_outer_join import _columns_of, _conjuncts, _plain_table, _unwrap

_REFUSED = (exp.Query, exp.Subquery, exp.Window, exp.Star, exp.Filter, exp.WithinGroup, exp.Grouping, exp.IgnoreNulls, exp.RespectNulls)

NAME = "distinct_child_counts"
_ARGS = frozenset({"expressions", "from_", "from", "joins", "where", "group"})
_INNER, _COUNTS, _VALUE = "kqs", "kqd", "kqn0"


def _owner(node: exp.Expression, sources: dict[str, set[str]]) -> tuple[str, str] | None:
    """``(source, column)`` for a plain column that names its source or belongs to exactly one."""

    if type(node) is not exp.Column or not isinstance(node.this, exp.Identifier) or node.this.args.get("quoted"):
        return None
    if node.args.get("db") or node.args.get("catalog"):
        return None
    name = node.name.lower()
    if node.table:
        if node.args["table"].args.get("quoted"):
            return None
        alias = node.table.lower()
        return (alias, name) if name in sources.get(alias, ()) else None
    owners = [alias for alias, columns in sources.items() if name in columns]
    return (owners[0], name) if len(owners) == 1 else None


def _one_column_key(table: exp.Table, name: str, keys: dict, not_null: dict | None) -> bool:
    """Whether ``name`` alone is a declared key of ``table`` and declared NOT NULL."""

    full = ".".join(part.name for part in table.parts).lower()
    never_null = {str(c).lower() for c in ((not_null or {}).get(full) or (not_null or {}).get(table.name.lower()) or ())}
    declared = keys.get(full) or keys.get(table.name.lower()) or []
    return name in never_null and any(tuple(str(c).lower() for c in key) == (name,) for key in declared)


def _counted(call: exp.Expression, sources: dict[str, set[str]]) -> tuple[str, str, bool] | None:
    """``(source, column, distinct)`` for ``COUNT([DISTINCT] source.column)``, else ``None``."""

    if type(call) is not exp.Count or any(value for key, value in call.args.items() if key not in ("this", "big_int")):
        return None
    inner, distinct = _unwrap(call.this), False
    if type(inner) is exp.Distinct:
        if inner.args.get("on") or len(inner.expressions) != 1:
            return None
        inner, distinct = _unwrap(inner.expressions[0]), True
    owner = _owner(inner, sources)
    return None if owner is None else (*owner, distinct)


def _type_family(types: dict | None, table: exp.Table, column: str) -> str | None:
    full = ".".join(part.name for part in table.parts).lower()
    lowered = {str(name).lower(): value for name, value in (types or {}).items()}
    declared = lowered.get(full) or lowered.get(table.name.lower()) or {}
    for name, value in declared.items():
        if str(name).lower() == column and value:
            from .set_operation_types import _family

            return _family(str(value))
    return None


def _equality(part: exp.Expression, sources: dict[str, set[str]]):
    part = _unwrap(part)
    if type(part) is not exp.EQ:
        return None
    ends = _owner(_unwrap(part.this), sources), _owner(_unwrap(part.expression), sources)
    return None if None in ends else ends


def _distinct_child_counts(select: exp.Select, declared: dict[str, set[str]], keys: dict, not_null: dict | None, types: dict | None) -> exp.Select | None:
    if type(select) is not exp.Select or any(value for key, value in select.args.items() if key not in _ARGS):
        return None
    group, joins = select.args.get("group"), select.args.get("joins") or []
    if group is None or not group.expressions or any(value for key, value in group.args.items() if key != "expressions") or not joins:
        return None
    base = select.args.get(FROM_KEY).this if select.args.get(FROM_KEY) is not None else None
    if base is None or not _plain_table(base):
        return None
    sources: dict[str, set[str]] = {}
    tables: dict[str, exp.Table] = {}
    for table in [base] + [join.this for join in joins]:
        if not _plain_table(table):
            return None
        columns, alias = _columns_of(table, declared), (table.alias_or_name or "").lower()
        if not columns or not alias or alias in sources or alias.startswith(_COUNTS) or alias == _INNER:
            return None
        sources[alias], tables[alias] = columns, table
    left = (base.alias_or_name or "").lower()
    base_key = key_of(base, keys, sources[left], not_null)
    if base_key is None:
        return None
    if any(isinstance(node, _REFUSED) for node in select.walk() if node is not select):
        return None

    # every aggregate counts the distinct values of a key of one joined table: those tables are the children
    calls: list[tuple[exp.Count, str]] = []
    plain = False
    for call in select.find_all(exp.AggFunc):
        counted = _counted(call, sources)
        if counted is None or counted[0] == left or not _one_column_key(tables[counted[0]], counted[1], keys, not_null):
            return None
        calls.append((call, counted[0]))
        plain = plain or not counted[2]
    children = {alias for _, alias in calls}
    if not children or (plain and len(children) > 1):
        return None  # with two children a plain COUNT counts pairs of their rows
    listed = {id(node) for item in select.expressions for node in item.walk()}
    if any(id(call) not in listed for call, _ in calls):
        return None  # an aggregate outside the select list
    if select.parent is not None and any(type(item) is not exp.Alias for item in select.expressions if item.find(exp.AggFunc) is not None):
        return None  # inside a larger query the count is known by its name

    parents, links = {left}, {}
    for join in joins:
        if any(value for key, value in join.args.items() if key not in ("this", "side", "kind", "on")) or join.args.get("on") is None:
            return None
        side, kind = (join.args.get("side") or "").upper(), (join.args.get("kind") or "").upper()
        alias = (join.this.alias_or_name or "").lower()
        parts = _conjuncts(join.args["on"])
        if alias in children:
            if side != "LEFT" or kind not in ("", "OUTER"):
                return None
            link, tests = None, []
            for part in parts:
                ends = _equality(part, sources)
                if link is None and ends is not None and {ends[0][0] == alias, ends[1][0] == alias} == {True, False}:
                    own, other = ends if ends[0][0] == alias else ends[::-1]
                    if other[0] not in parents:
                        return None
                    link = (own[1], other)
                    continue
                if any(isinstance(node, exp.AggFunc) for node in part.walk()):
                    return None
                columns = list(part.find_all(exp.Column))
                if not columns or any((_owner(column, sources) or ("",))[0] != alias for column in columns):
                    return None
                tests.append(part)
            if link is None or link[0] == _VALUE:
                return None
            families = {_type_family(types, tables[alias], link[0]), _type_family(types, tables[link[1][0]], link[1][1])} - {None}
            if len(families) > 1:
                return None  # ``=`` across types is not the equality the counted table groups by
            links[alias] = (link[0], link[1], tests)
            continue
        # a lookup: at most one row per parent row
        if not ((side == "" and kind in ("", "INNER")) or (side == "LEFT" and kind in ("", "OUTER"))) or len(parts) != 1:
            return None
        ends = _equality(parts[0], sources)
        if ends is None or {ends[0][0] == alias, ends[1][0] == alias} != {True, False}:
            return None
        own, other = ends if ends[0][0] == alias else ends[::-1]
        if other[0] not in parents or not _one_column_key(tables[alias], own[1], keys, not_null):
            return None
        parents.add(alias)
    if set(links) != children:
        return None

    def of_parents(node: exp.Expression | None) -> bool:
        if node is None:
            return True
        return all((_owner(column, sources) or ("",))[0] in parents for column in node.find_all(exp.Column))

    where = select.args.get("where")
    if not of_parents(where) or (where is not None and any(isinstance(node, exp.AggFunc) for node in where.walk())):
        return None
    grouped = [_owner(_unwrap(item), sources) for item in group.expressions]
    if any(owner is None or owner[0] not in parents for owner in grouped):
        return None
    if not set(base_key) <= {name for alias, name in grouped if alias == left}:
        return None
    for item in group.expressions:
        item = _unwrap(item)
        if not item.table and any(type(listed_item) is exp.Alias and listed_item.alias.lower() == item.name.lower() and _owner(_unwrap(listed_item.this), sources) != _owner(item, sources) for listed_item in select.expressions):
            return None  # the name is also a select alias of something else
    inside = {id(column) for call, _ in calls for column in call.find_all(exp.Column)}
    for item in select.expressions:
        for column in item.find_all(exp.Column):
            if id(column) not in inside and (_owner(column, sources) or ("",))[0] not in parents:
                return None

    copy = select.copy()
    names = {alias: f"{_COUNTS}{index}" for index, alias in enumerate(sorted(links))}
    fresh = list(copy.find_all(exp.AggFunc))
    if len(fresh) != len(calls):
        return None
    for (_, alias), new in zip(calls, fresh):
        new.replace(exp.Coalesce(this=exp.column(_VALUE, table=names[alias]), expressions=[exp.Literal.number(0)]))
    new_joins = []
    for join in copy.args["joins"]:
        alias = (join.this.alias_or_name or "").lower()
        if alias not in links:
            new_joins.append(join)
            continue
        key, other, tests = links[alias]
        table = tables[alias].copy()
        table.set("alias", exp.TableAlias(this=exp.to_identifier(_INNER)))
        inner = exp.Select(expressions=[exp.alias_(exp.column(key, table=_INNER), key), exp.alias_(exp.Count(this=exp.Star()), _VALUE)])
        inner.set(FROM_KEY, exp.From(this=table))
        if tests:
            kept = []
            for test in tests:
                test = test.copy()
                for column in test.find_all(exp.Column):
                    column.set("table", exp.to_identifier(_INNER))
                kept.append(test)
            inner.set("where", exp.Where(this=_and_all(kept)))
        inner.set("group", exp.Group(expressions=[exp.column(key, table=_INNER)]))
        derived = exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier(names[alias])))
        on = exp.EQ(this=exp.column(other[1], table=other[0]), expression=exp.column(key, table=names[alias]))
        new_joins.append(exp.Join(this=derived, on=on, side="LEFT"))
    copy.set("joins", new_joins)
    copy.set("group", None)
    return copy


def read_distinct_child_counts(
    tree: exp.Expression, schema: dict | None, keys: dict | None, not_null: dict | None = None, types: dict | None = None
) -> tuple[exp.Expression, bool]:
    """A copy of ``tree`` with every select described above written with one counted table per child, and whether
    any was (``tree`` itself when none was). ``tree`` is not changed."""

    if not schema or not keys:
        return tree, False
    from .counted_membership import _unstable

    declared = declared_columns(schema)

    def written(select: exp.Select) -> exp.Select | None:
        if not (select.args.get("joins") and select.args.get("group")) or _unstable(select):
            return None
        return _distinct_child_counts(select, declared, keys, not_null, types)

    if not any(written(select) is not None for select in tree.find_all(exp.Select)):
        return tree, False
    tree = tree.copy()
    # the selects read hold no nested query, so no two of them overlap
    for select in list(tree.find_all(exp.Select)):
        new = written(select)
        if new is None:
            continue
        if select is tree:
            tree = new
        else:
            select.replace(new)
    return tree, True
