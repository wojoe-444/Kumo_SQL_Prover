"""Counts of a parent row's child rows, read in one spelling.

"Each bill with its owner's name, the number of the owner's friends and the number of them who are members" is
written in three ways, and no two of them met:

    -- grouped after the joins
    SELECT i.bill_id, c.member_name, COUNT(f.owner_id), COUNT(m.mail)
    FROM bill AS i JOIN member AS c ON c.member_id = i.owner_id
    LEFT JOIN friend AS f ON c.member_id = f.owner_id LEFT JOIN member AS m ON f.friend_mail = m.mail
    GROUP BY i.bill_id, c.member_name

    -- counted per owner first, joined afterwards
    SELECT i.bill_id, c.member_name, COALESCE(g.n, 0), COALESCE(t.n, 0)
    FROM bill AS i JOIN member AS c ON c.member_id = i.owner_id
    LEFT JOIN (SELECT owner_id, COUNT(*) AS n FROM friend GROUP BY owner_id) AS g ON i.owner_id = g.owner_id
    LEFT JOIN (SELECT owner_id, COUNT(*) AS n FROM friend WHERE friend_mail IN (SELECT mail FROM member)
               GROUP BY owner_id) AS t ON i.owner_id = t.owner_id

    -- each member with its counts first, the bills joined to that
    SELECT i.bill_id, t.member_name, t.friends, t.trusted
    FROM bill AS i JOIN (SELECT a.member_id, a.member_name, COUNT(b.owner_id) AS friends,
                                SUM(CASE WHEN b.friend_mail IN (SELECT mail FROM member) THEN 1 ELSE 0 END) AS trusted
                         FROM member AS a LEFT JOIN friend AS b ON a.member_id = b.owner_id
                         GROUP BY a.member_id, a.member_name) AS t ON i.owner_id = t.member_id

Each is read as the same thing: **parent rows** (a driving table, and tables looked up from it by a declared
one-column key, at most one row each), **one child table** linked to a parent column by an equality, and outputs
that are either values of the parent row or **counts of the child rows of that parent row that pass some tests**.
The select is then written counted first, with one count per set of tests:

    SELECT i.bill_id, c.member_name, COALESCE(kqf.kqn0, 0), COALESCE(kqf.kqn1, 0)
    FROM bill AS i JOIN member AS c ON c.member_id = i.owner_id
    LEFT JOIN (SELECT kqs.owner_id AS owner_id, COUNT(*) AS kqn0,
                      COUNT(CASE WHEN kqs.friend_mail IN (SELECT kqm.mail FROM member AS kqm) THEN 1 END) AS kqn1
               FROM friend AS kqs GROUP BY kqs.owner_id) AS kqf ON i.owner_id = kqf.owner_id

**Why the three are that.** Fix a parent row ``r``: one row of the driving table with the one row each lookup
finds for it (a row an inner lookup finds nothing for is gone in every form). Write ``v`` for the value of the
linked parent column on ``r``, and ``n(T)`` for the number of child rows whose key equals ``v`` and pass the tests
``T``.

* *Grouped after the joins.* The joined rows of ``r`` are one per child row whose key equals ``v`` (and that passes
  the tests written in ``ON``), or one row padded with NULLs when there is none. A ``GROUP BY`` that holds a NOT
  NULL key of the driving table makes one group of exactly these rows: no two driving rows share the key, and the
  other grouped columns are columns of ``r``. A further ``LEFT JOIN m ON m.key = f.c`` on a declared one-column key
  finds at most one row per joined row, so it neither adds nor removes a row; ``COUNT(m.x)`` for a column that is
  not NULL on a match counts the rows with a match, which are the rows where ``f.c IN (SELECT key FROM m)`` is
  TRUE. Every aggregate read here counts the joined rows on which a test is TRUE that is not TRUE on the padded
  row (a comparison, ``IN`` or ``IS NOT NULL`` on a child column): ``n(T)``, and 0 for the padded row.
* *Counted first.* The derived table has one row per key value with a child row that passes its ``WHERE``, holding
  the count; the left join on ``v`` finds that row or pads, and ``COALESCE(count, 0)`` is ``n(T)`` either way. The
  join finds at most one row, so the select returns one row per parent row, as the grouping by the key does.
* *Parent with counts first.* The inner derived table is the first form over one table ``a`` grouped by a column
  the outer join uses, so it has one row per row of ``a`` when that column is a declared one-column key (two rows
  of ``a`` with one key value would fall into one group). The outer join on that key finds at most one of them; an
  inner join keeps the outer rows that find one, and reads its counts. A ``LEFT`` join does the same when a
  declared foreign key from a NOT NULL column says every outer row finds one and the inner table drops no row of
  ``a`` (without that the padded row would read NULL where the other forms read 0, and the select is left alone).
* *The written form.* One count per set of tests over all the child rows of a key value, joined by ``LEFT JOIN``
  with ``COALESCE(count, 0)``: ``n(T)`` for a key value with child rows, 0 for one with none, and 0 for a NULL
  key value, which no child group equals.

A grouped-after-joins select needs the NOT NULL key of the driving table in its ``GROUP BY``; the other two do not
group the parent rows, so they are read without it.

The written form links the child to one agreed column: among the parent columns that the inner lookups make
equal to the linked one, the first by table and column name. Two columns an inner join equates are equal on every
row that survives it, so the child rows found are the same. A ``LEFT`` lookup whose column is NOT NULL and
declared to reference the looked-up key finds exactly one row, and is written as an inner join. Nothing is merged
across any other ``LEFT`` lookup, across two parents that read the same table, or between columns declared with
different types.

**What is read** (everything else is left alone): plain tables the schema lists, each with its own alias; joins
with ``ON`` only; lookups ``INNER`` or ``LEFT`` on one equality with a declared one-column key; the child ``LEFT``
joined on one equality with a parent column, plus tests on its own columns; counts ``COUNT(col)``,
``COUNT(CASE WHEN tests THEN value END)``, ``SUM(CASE WHEN tests THEN 1 ELSE 0 END)``,
``SUM(CASE WHEN col IS NULL THEN 0 ELSE 1 END)``, ``COUNT(*)`` inside a counted-first table, each possibly under
``COALESCE(.., 0)``; tests that are conjunctions of ``col IS NOT NULL``, ``col <op> literal`` and
``col IN (SELECT column FROM table)``. A table ``LEFT`` joined to the child on a declared one-column key, or a
list ``(SELECT DISTINCT column FROM table)``, is only asked for a match: ``COUNT`` of its joined column counts the
child rows whose value is ``IN`` that column. A counted-first table may read the child through a derived table that
shows the child's rows with 0/1 flag columns (``CASE WHEN tests THEN 1 ELSE 0 END AS flag``) and add a flag up
with ``SUM(flag)``. ``MIN``/``MAX``/``ANY_VALUE`` of a parent column is that column. No ``WHERE`` beside the
joins, no ``DISTINCT``, ``HAVING``, window, tail, volatile call or quoted name.

A narrow retry rule (``shape_fallback``): a pair it does not help is tried again without it.
``tests/test_child_counts.py`` replays each spelling and its reading on DuckDB and shows each declared fact is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlglot import exp

from .ast_utils import FROM_KEY
from .fk_rules import _same_table
from .keyed_join_groups import _and_all, declared_columns
from .unmatched_outer_join import _columns_of, _conjuncts, _plain_table, _unwrap

_FLAT_ARGS = frozenset({"expressions", "from_", "from", "joins", "group"})
_JOINED_ARGS = frozenset({"expressions", "from_", "from", "joins"})
_COUNTS_ARGS = frozenset({"expressions", "from_", "from", "joins", "where", "group"})
_MEMBER_ARGS = frozenset({"expressions", "from_", "from", "distinct"})
_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
_INNER, _CHILD, _MEMBER = "kqs", "kqf", "kqm"


def _full(table: exp.Table) -> str:
    return ".".join(part.name for part in table.parts).lower()


def _facts_of(facts: dict | None, table: exp.Table):
    full = _full(table)
    for name, value in (facts or {}).items():
        if _same_table(full, str(name)):
            return value
    return None


def _literal(node: exp.Expression | None, text: str) -> bool:
    node = _unwrap(node) if node is not None else None
    return type(node) is exp.Literal and not node.is_string and node.this == text


def _constant(node: exp.Expression) -> bool:
    """A literal that is never NULL."""

    node = _unwrap(node)
    if type(node) is exp.Neg:
        node = _unwrap(node.this)
    return type(node) is exp.Literal


def _is_null(node: exp.Expression) -> exp.Expression | None:
    """``x`` of ``x IS NULL``."""

    node = _unwrap(node)
    return _unwrap(node.this) if type(node) is exp.Is and type(_unwrap(node.expression)) is exp.Null else None


def _is_not_null(node: exp.Expression) -> exp.Expression | None:
    node = _unwrap(node)
    return _is_null(node.this) if type(node) is exp.Not else None


@dataclass
class _Parent:
    table: exp.Table
    alias: str
    columns: set[str]
    kind: str  # "from", "inner" or "left"
    on: tuple[tuple[str, str], tuple[str, str]] | None = None  # (this table's key, the earlier parent's column)


@dataclass
class _Model:
    parents: list[_Parent]
    child: exp.Table
    child_key: str
    links: list[tuple[str, str]]  # the parent columns the child key is compared with (all must be equal)
    outputs: list[tuple[str | None, str, object]]  # (alias, "parent", expression) or (alias, "count", guards)
    guards: dict[tuple, exp.Expression | None] = field(default_factory=dict)  # comparison guards: key to expression
    held: set[str] = field(default_factory=set)  # the driving table's columns a grouped select groups by


class _Scope:
    """The sources of one select and the columns each gives."""

    def __init__(self) -> None:
        self.sources: dict[str, set[str]] = {}

    def add(self, alias: str, columns: set[str]) -> bool:
        if not alias or alias in self.sources:
            return False
        self.sources[alias] = columns
        return True

    def owner(self, node: exp.Expression | None) -> tuple[str, str] | None:
        node = _unwrap(node) if node is not None else None
        if type(node) is not exp.Column or not isinstance(node.this, exp.Identifier) or node.this.args.get("quoted"):
            return None
        if node.args.get("db") or node.args.get("catalog"):
            return None
        name = node.name.lower()
        if node.table:
            if node.args["table"].args.get("quoted"):
                return None
            alias = node.table.lower()
            return (alias, name) if name in self.sources.get(alias, ()) else None
        owners = [alias for alias, columns in self.sources.items() if name in columns]
        return (owners[0], name) if len(owners) == 1 else None


def _join(join: exp.Join) -> tuple[str, exp.Expression, list[exp.Expression]] | None:
    if any(value for key, value in join.args.items() if key not in ("this", "side", "kind", "on")) or join.args.get("on") is None:
        return None
    side, kind = (join.args.get("side") or "").upper(), (join.args.get("kind") or "").upper()
    if side == "" and kind in ("", "INNER"):
        return "inner", join.this, _conjuncts(join.args["on"])
    if side == "LEFT" and kind in ("", "OUTER"):
        return "left", join.this, _conjuncts(join.args["on"])
    return None


class _Reader:
    def __init__(self, schema: dict, keys: dict, not_null: dict | None, foreign_keys: dict | None, types: dict | None) -> None:
        self.declared = declared_columns(schema)
        self.keys, self.not_null, self.foreign_keys, self.types = keys, not_null or {}, foreign_keys or {}, types or {}

    # --- declared facts -------------------------------------------------------------------------------------------

    def columns(self, source: exp.Expression) -> set[str] | None:
        return _columns_of(source, self.declared) if _plain_table(source) else None

    def _distinct_list(self, source: exp.Expression) -> tuple[exp.Table, str, str] | None:
        """``(SELECT DISTINCT c FROM t) AS x`` or ``(SELECT c FROM t GROUP BY c) AS x``: the table, ``c`` and the output name."""

        if type(source) is not exp.Subquery or not source.alias or type(source.this) is not exp.Select:
            return None
        if source.args["alias"].args.get("columns") or any(v for key, v in source.args.items() if key not in ("this", "alias")):
            return None
        inner = source.this
        if any(value for key, value in inner.args.items() if key not in ("expressions", "from_", "from", "distinct", "group")):
            return None
        table = inner.args.get(FROM_KEY).this if inner.args.get(FROM_KEY) is not None else None
        listed = self.columns(table) if table is not None else None
        if not listed or len(inner.expressions) != 1:
            return None
        scope = _Scope()
        scope.add((table.alias_or_name or "").lower(), listed)
        item = inner.expressions[0]
        owner = scope.owner(item.this if isinstance(item, exp.Alias) else item)
        distinct, group = inner.args.get("distinct"), inner.args.get("group")
        if owner is None or (distinct is None) == (group is None):
            return None
        if distinct is not None and distinct.args.get("on") is not None:
            return None
        if group is not None and (len(group.expressions) != 1 or any(v for k, v in group.args.items() if k != "expressions") or scope.owner(group.expressions[0]) != owner):
            return None
        return table, owner[1], (item.alias_or_name or "").lower()

    def source_columns(self, source: exp.Expression) -> set[str] | None:
        listed = self.columns(source)
        if listed:
            return listed
        values = self._distinct_list(source)
        return {values[2]} if values is not None else None

    def one_column_keys(self, table: exp.Table) -> set[str]:
        return {str(key[0]).lower() for key in _facts_of(self.keys, table) or [] if len(key) == 1}

    def never_null(self, table: exp.Table) -> set[str]:
        return {str(c).lower() for c in _facts_of(self.not_null, table) or ()}

    def references(self, child: exp.Table, column: str, parent: exp.Table, key: str) -> bool:
        for foreign in _facts_of(self.foreign_keys, child) or []:
            columns, target, target_columns = foreign[0], foreign[1], foreign[2]
            if tuple(str(c).lower() for c in columns) == (column,) and tuple(str(c).lower() for c in target_columns) == (key,) and _same_table(_full(parent), str(target)):
                return True
        return False

    def type_of(self, table: exp.Table, column: str) -> str | None:
        declared = _facts_of(self.types, table) or {}
        for name, value in declared.items():
            if str(name).lower() == column:
                return str(value).upper()
        return None

    # --- tests and counts -----------------------------------------------------------------------------------------

    def _non_null(self, node: exp.Expression, ctx: dict) -> frozenset | None:
        """The tests under which ``node`` has a value on a joined row (and never on the padded row)."""

        node = _unwrap(node)
        owner = ctx["scope"].owner(node)
        if owner is not None:
            alias, name = owner
            if alias == ctx["child"]:
                if name in ctx.get("flags", ()):
                    return None
                return frozenset() if name == ctx["key"] or name in self.never_null(ctx["table"]) else frozenset({("notnull", name)})
            presence = ctx["presence"].get(alias)
            if presence is not None and name in presence["solid"]:
                return frozenset({("member", presence["column"], _full(presence["table"]), presence["key"])})
            return None
        if type(node) is exp.Case and node.args.get("this") is None and len(node.args.get("ifs") or []) == 1:
            default = node.args.get("default")
            if default is not None and type(_unwrap(default)) is not exp.Null:
                return None
            branch = node.args["ifs"][0]
            tests, value = self._tests(branch.this, ctx), _unwrap(branch.args["true"])
            if tests is None:
                return None
            if _constant(value):
                return tests
            inner = self._non_null(value, ctx) if ctx["scope"].owner(value) is not None else None
            return None if inner is None else tests | inner
        return None

    def _tests(self, node: exp.Expression, ctx: dict) -> frozenset | None:
        """A conjunction of tests on the child row, none of which is TRUE on the padded row."""

        found: set[tuple] = set()
        for part in _conjuncts(node):
            target = _is_not_null(part)
            if target is not None:
                tests = self._non_null(target, ctx) if ctx["scope"].owner(target) is not None else None
                if tests is None:
                    return None
                found |= tests
                continue
            if type(part) is exp.In:
                test = self._membership(part, ctx)
                if test is None:
                    return None
                found.add(test)
                continue
            if type(part) in _COMPARISONS:
                test = self._comparison(part, ctx)
                if test is None:
                    return None
                found.add(test)
                continue
            return None
        return frozenset(found)

    def _child_column(self, node: exp.Expression, ctx: dict) -> str | None:
        owner = ctx["scope"].owner(node)
        return owner[1] if owner is not None and owner[0] == ctx["child"] and owner[1] not in ctx.get("flags", ()) else None

    def _flag(self, node: exp.Expression, ctx: dict) -> frozenset | None:
        """The tests under which a 0/1 flag is 1: ``CASE WHEN tests THEN 1 ELSE 0 END`` or ``CASE WHEN x IS NULL THEN 0 ELSE 1 END``."""

        case = _unwrap(node)
        if type(case) is not exp.Case or case.args.get("this") is not None or len(case.args.get("ifs") or []) != 1:
            return None
        branch, default = case.args["ifs"][0], case.args.get("default")
        if _literal(branch.args["true"], "1") and _literal(default, "0"):
            return self._tests(branch.this, ctx)
        if _literal(branch.args["true"], "0") and _literal(default, "1") and _is_null(branch.this) is not None:
            target = _is_null(branch.this)
            return self._non_null(target, ctx) if ctx["scope"].owner(target) is not None else None
        return None

    def _membership(self, node: exp.In, ctx: dict) -> tuple | None:
        if any(value for key, value in node.args.items() if key not in ("this", "query")):
            return None
        column = self._child_column(node.this, ctx)
        query = node.args.get("query")
        query = query.this if type(query) is exp.Subquery and not any(v for k, v in query.args.items() if k != "this") else query
        if column is None or type(query) is not exp.Select or any(v for k, v in query.args.items() if k not in _MEMBER_ARGS):
            return None
        distinct = query.args.get("distinct")
        if distinct is not None and distinct.args.get("on") is not None:
            return None
        source = query.args.get(FROM_KEY).this if query.args.get(FROM_KEY) is not None else None
        listed = self.columns(source) if source is not None else None
        if not listed or len(query.expressions) != 1:
            return None
        scope = _Scope()
        scope.add((source.alias_or_name or "").lower(), listed)
        item = query.expressions[0]
        owner = scope.owner(item.this if isinstance(item, exp.Alias) else item)
        return ("member", column, _full(source), owner[1]) if owner is not None else None

    def _comparison(self, node: exp.Expression, ctx: dict) -> tuple | None:
        left, right = _unwrap(node.this), _unwrap(node.expression)
        if self._child_column(left, ctx) is not None and _constant(right):
            column = left
        elif self._child_column(right, ctx) is not None and _constant(left):
            column = right
        else:
            return None
        copy = node.copy()
        target = _unwrap(copy.this) if column is left else _unwrap(copy.expression)
        target.replace(exp.column(self._child_column(column, ctx), table=_INNER))
        key = ("test", copy.sql())
        ctx["guards"][key] = copy
        return key

    def _count(self, node: exp.Expression, ctx: dict) -> frozenset | None:
        """The tests a count counts the child rows by, or ``None`` when ``node`` is not such a count."""

        node = _unwrap(node)
        if type(node) is exp.Coalesce and len(node.expressions) == 1 and _literal(node.expressions[0], "0"):
            node = _unwrap(node.this)
        if type(node) is exp.Count:
            if any(value for key, value in node.args.items() if key not in ("this", "big_int")):
                return None
            if isinstance(node.this, exp.Star):
                return ctx["base"] if ctx["star"] else None
            tests = self._non_null(node.this, ctx)
            return None if tests is None else ctx["base"] | tests
        if type(node) is exp.Sum and not any(value for key, value in node.args.items() if key != "this"):
            owner = ctx["scope"].owner(node.this)
            if owner is not None and owner[0] == ctx["child"] and owner[1] in ctx.get("flags", ()):
                return ctx["base"] | ctx["flags"][owner[1]]  # a 0/1 flag of the child row, added up
            tests = self._flag(node.this, ctx)
            return None if tests is None else ctx["base"] | tests
        return None

    # --- sources ----------------------------------------------------------------------------------------------------

    def _equality(self, node: exp.Expression, scope: _Scope) -> tuple[tuple[str, str], tuple[str, str]] | None:
        node = _unwrap(node)
        if type(node) is not exp.EQ:
            return None
        left, right = scope.owner(node.this), scope.owner(node.expression)
        return (left, right) if left is not None and right is not None else None

    def _lookup(self, kind: str, table: exp.Table, parts: list[exp.Expression], scope: _Scope, parents: list[_Parent]) -> _Parent | None:
        alias = (table.alias_or_name or "").lower()
        ends = self._equality(parts[0], scope) if len(parts) == 1 and _plain_table(table) else None
        if ends is None:
            return None
        own, other = ends if ends[0][0] == alias else ends[::-1]
        if own[0] != alias or other[0] not in {p.alias for p in parents} or own[1] not in self.one_column_keys(table):
            return None
        return _Parent(table, alias, scope.sources[alias], kind, (own, other))

    def _presence(self, kind: str, source: exp.Expression, parts: list[exp.Expression], scope: _Scope, child: str) -> dict | None:
        """A source left joined to the child on a column that holds each value once: at most one row per child row."""

        alias = (source.alias_or_name or "").lower()
        ends = self._equality(parts[0], scope) if kind == "left" and len(parts) == 1 else None
        if ends is None:
            return None
        own, other = ends if ends[0][0] == alias else ends[::-1]
        if own[0] != alias or other[0] != child:
            return None
        values = self._distinct_list(source)
        if values is not None:
            return {"table": values[0], "key": values[1], "column": other[1], "solid": {values[2]}} if own[1] == values[2] else None
        if not _plain_table(source) or own[1] not in self.one_column_keys(source):
            return None
        return {"table": source, "key": own[1], "column": other[1], "solid": {own[1]} | self.never_null(source)}

    @staticmethod
    def _own_columns(node: exp.Expression):
        """The columns of ``node`` that are not inside a nested query."""

        stack = [node]
        while stack:
            current = stack.pop()
            if isinstance(current, exp.Column):
                yield current
                continue
            for child in current.iter_expressions():
                if not isinstance(child, (exp.Select, exp.Subquery, exp.SetOperation)):
                    stack.append(child)

    def _parent_value(self, node: exp.Expression, scope: _Scope, parents: set[str]) -> exp.Expression | None:
        """``node`` over parent columns only, each written with its source."""

        node = _unwrap(node)
        if type(node) in (exp.Min, exp.Max, exp.AnyValue) and not any(v for k, v in node.args.items() if k != "this") and scope.owner(node.this) is not None:
            node = _unwrap(node.this)
        if any(isinstance(n, (exp.AggFunc, exp.Window, exp.Query, exp.Subquery, exp.Star)) for n in node.walk()):
            return None
        copy = node.copy()
        for column in [copy] if isinstance(copy, exp.Column) else list(copy.find_all(exp.Column)):
            owner = scope.owner(column)
            if owner is None or owner[0] not in parents:
                return None
            fresh = exp.column(owner[1], table=owner[0])
            if column is copy:
                copy = fresh
            else:
                column.replace(fresh)
        return copy

    def _flat(self, select: exp.Expression) -> _Model | None:
        """A select grouped after its joins: lookups, the child, tables only asked for a match."""

        if type(select) is not exp.Select or any(value for key, value in select.args.items() if key not in _FLAT_ARGS):
            return None
        group, joins = select.args.get("group"), select.args.get("joins") or []
        base = select.args.get(FROM_KEY).this if select.args.get(FROM_KEY) is not None else None
        if group is None or not group.expressions or any(v for k, v in group.args.items() if k != "expressions") or not joins:
            return None
        scope = _Scope()
        if base is None or not self.columns(base):
            return None
        for source in [base] + [join.this for join in joins]:
            columns = self.source_columns(source)
            if not columns or not scope.add((source.alias_or_name or "").lower(), columns):
                return None
        counted: set[str] = set()  # the sources an aggregate other than MIN/MAX of a column reads
        for item in select.expressions:
            for call in item.find_all(exp.AggFunc):
                if type(call) in (exp.Min, exp.Max, exp.AnyValue) and scope.owner(call.this) is not None:
                    continue
                for column in self._own_columns(call):
                    owner = scope.owner(column)
                    if owner is None:
                        return None
                    counted.add(owner[0])
        parents = [_Parent(base, (base.alias_or_name or "").lower(), scope.sources[(base.alias_or_name or "").lower()], "from")]
        ctx: dict | None = None
        for join in joins:
            read = _join(join)
            if read is None:
                return None
            kind, table, parts = read
            alias = (table.alias_or_name or "").lower()
            if ctx is None and alias not in counted:
                lookup = self._lookup(kind, table, parts, scope, parents)
                if lookup is None:
                    return None
                parents.append(lookup)
            elif ctx is None:
                if kind != "left" or not _plain_table(table):
                    return None
                ctx = {"scope": scope, "child": alias, "table": table, "key": None, "presence": {}, "base": frozenset(), "star": False, "guards": {}}
                link, others = None, []
                for part in parts:
                    ends = self._equality(part, scope)
                    if link is None and ends is not None and {ends[0][0] == alias, ends[1][0] == alias} == {True, False}:
                        own, other = ends if ends[0][0] == alias else ends[::-1]
                        if other[0] in {p.alias for p in parents}:
                            link = (own, other)
                            continue
                    others.append(part)
                if link is None:
                    return None
                ctx["key"] = link[0][1]
                base_tests = self._tests(_and_all(others), ctx) if others else frozenset()
                if base_tests is None:
                    return None
                ctx["base"], ctx["link"] = base_tests, link[1]
            else:
                presence = self._presence(kind, table, parts, scope, ctx["child"])
                if presence is None:
                    return None
                ctx["presence"][alias] = presence
        if ctx is None:
            return None
        names = {p.alias for p in parents}
        grouped = [scope.owner(g) for g in group.expressions]
        if any(g is None or g[0] not in names for g in grouped):
            return None
        outputs = self._outputs(select, scope, names, lambda value: self._count(value, ctx))
        if outputs is None:
            return None
        held = {name for alias, name in grouped if alias == parents[0].alias}
        return _Model(parents, ctx["table"], ctx["key"], [ctx["link"]], outputs, ctx["guards"], held)

    def _one_group_per_row(self, model: _Model) -> bool:
        """The grouping holds a declared key of the driving table whose columns are all NOT NULL."""

        driving = model.parents[0]
        never_null = self.never_null(driving.table)
        declared = ([str(c).lower() for c in key] for key in _facts_of(self.keys, driving.table) or [])
        return any(key and set(key) <= model.held and set(key) <= never_null for key in declared)

    def _outputs(self, select: exp.Select, scope: _Scope, parents: set[str], count) -> list | None:
        outputs = []
        for item in select.expressions:
            alias = item.alias if isinstance(item, exp.Alias) else None
            value = item.this if isinstance(item, exp.Alias) else item
            tests = count(value)
            if tests is not None:
                outputs.append((alias, "count", tests))
                continue
            parent = self._parent_value(value, scope, parents)
            if parent is None:
                return None
            outputs.append((alias or (value.name if isinstance(_unwrap(value), exp.Column) else None), "parent", parent))
        return outputs

    def _counts_table(self, source: exp.Expression) -> dict | None:
        """A derived table of counts per key of one child table: its child, key, key output and count outputs."""

        if type(source) is not exp.Subquery or not source.alias or type(source.this) is not exp.Select:
            return None
        if source.args["alias"].args.get("columns") or any(v for key, v in source.args.items() if key not in ("this", "alias")):
            return None
        inner = source.this
        if any(value for key, value in inner.args.items() if key not in _COUNTS_ARGS):
            return None
        group = inner.args.get("group")
        table = inner.args.get(FROM_KEY).this if inner.args.get(FROM_KEY) is not None else None
        if table is None or group is None or len(group.expressions) != 1 or any(v for k, v in group.args.items() if k != "expressions"):
            return None
        joins = inner.args.get("joins") or []
        columns, flags, guards = self.columns(table), {}, {}
        if not columns:
            view = self._child_view(table) if not joins else None
            if view is None:
                return None
            table, columns, flags, guards = view["table"], view["columns"] | set(view["flags"]), view["flags"], view["guards"]
        scope, alias = _Scope(), (inner.args.get(FROM_KEY).this.alias_or_name or "").lower()
        if not scope.add(alias, columns):
            return None
        for join in joins:
            listed = self.source_columns(join.this)
            if not listed or not scope.add((join.this.alias_or_name or "").lower(), listed):
                return None
        key = scope.owner(group.expressions[0])
        if key is None or key[0] != alias or key[1] in flags:
            return None
        ctx = {"scope": scope, "child": alias, "table": table, "key": key[1], "presence": {}, "base": frozenset(), "star": True, "guards": guards, "flags": flags}
        for join in joins:
            read = _join(join)
            presence = self._presence(read[0], read[1], read[2], scope, alias) if read is not None else None
            if presence is None:
                return None
            ctx["presence"][(join.this.alias_or_name or "").lower()] = presence
        where = inner.args.get("where")
        if where is not None:
            base = self._tests(where.this, ctx)
            if base is None:
                return None
            ctx["base"] = base
        key_name, counts = None, {}
        for item in inner.expressions:
            value = item.this if isinstance(item, exp.Alias) else item
            if scope.owner(value) == key and key_name is None:
                key_name = (item.alias_or_name or "").lower()
                continue
            tests = self._count(value, ctx) if isinstance(item, exp.Alias) else None
            if tests is None or not item.alias or item.alias.lower() in counts:
                return None
            counts[item.alias.lower()] = tests
        if not key_name or not counts or key_name in counts:
            return None
        return {"table": table, "key": key[1], "key_name": key_name, "counts": counts, "guards": ctx["guards"]}

    def _child_view(self, source: exp.Expression) -> dict | None:
        """A derived table that is the child's rows with 0/1 flag columns beside its own: its table, columns and flags."""

        if type(source) is not exp.Subquery or not source.alias or type(source.this) is not exp.Select:
            return None
        if source.args["alias"].args.get("columns") or any(v for key, v in source.args.items() if key not in ("this", "alias")):
            return None
        inner = source.this
        if any(value for key, value in inner.args.items() if key not in _JOINED_ARGS):
            return None
        table = inner.args.get(FROM_KEY).this if inner.args.get(FROM_KEY) is not None else None
        columns = self.columns(table) if table is not None else None
        scope, alias = _Scope(), (table.alias_or_name or "").lower() if table is not None else ""
        if not columns or not scope.add(alias, columns):
            return None
        joins = inner.args.get("joins") or []
        for join in joins:
            listed = self.source_columns(join.this)
            if not listed or not scope.add((join.this.alias_or_name or "").lower(), listed):
                return None
        ctx = {"scope": scope, "child": alias, "table": table, "key": None, "presence": {}, "base": frozenset(), "star": False, "guards": {}}
        for join in joins:
            read = _join(join)
            presence = self._presence(read[0], read[1], read[2], scope, alias) if read is not None else None
            if presence is None:
                return None
            ctx["presence"][(join.this.alias_or_name or "").lower()] = presence
        shown: set[str] = set()
        flags: dict[str, frozenset] = {}
        for item in inner.expressions:
            value = item.this if isinstance(item, exp.Alias) else item
            if isinstance(value, exp.Star) and not joins:
                shown |= columns
                continue
            if isinstance(value, exp.Column) and isinstance(value.this, exp.Star) and value.table.lower() == alias:
                shown |= columns
                continue
            owner = scope.owner(value)
            if owner is not None and owner[0] == alias and (not isinstance(item, exp.Alias) or item.alias.lower() == owner[1]):
                shown.add(owner[1])
                continue
            tests = self._flag(value, ctx) if isinstance(item, exp.Alias) else None
            if tests is None or item.alias.lower() in flags:
                return None
            flags[item.alias.lower()] = tests
        if set(flags) & shown:
            return None
        return {"table": table, "columns": shown, "flags": flags, "guards": ctx["guards"]}

    def _joined(self, select: exp.Expression) -> _Model | None:
        """A select that joins its parents to tables of counts, or to one table of parents with their counts."""

        if type(select) is not exp.Select or any(value for key, value in select.args.items() if key not in _JOINED_ARGS):
            return None
        joins = select.args.get("joins") or []
        base = select.args.get(FROM_KEY).this if select.args.get(FROM_KEY) is not None else None
        columns = self.columns(base) if base is not None else None
        scope = _Scope()
        if not columns or not joins or not scope.add((base.alias_or_name or "").lower(), columns):
            return None
        parents = [_Parent(base, (base.alias_or_name or "").lower(), columns, "from")]
        counts: dict[str, dict] = {}
        grouped: dict | None = None
        links: list[tuple[str, str]] = []
        for join in joins:
            read = _join(join)
            if read is None:
                return None
            kind, source, parts = read
            alias = (source.alias_or_name or "").lower()
            listed = self.columns(source)
            if listed:
                if counts or grouped is not None or not scope.add(alias, listed):
                    return None
                lookup = self._lookup(kind, source, parts, scope, parents)
                if lookup is None:
                    return None
                parents.append(lookup)
                continue
            if grouped is not None:
                return None
            info = self._counts_table(source)
            if info is not None:
                if kind != "left" or not scope.add(alias, {info["key_name"]} | set(info["counts"])):
                    return None
                ends = self._equality(parts[0], scope) if len(parts) == 1 else None
                if ends is None:
                    return None
                own, other = ends if ends[0][0] == alias else ends[::-1]
                if own != (alias, info["key_name"]) or other[0] not in {p.alias for p in parents}:
                    return None
                counts[alias] = info
                links.append(other)
                continue
            if counts or type(source) is not exp.Subquery or not source.alias or source.args["alias"].args.get("columns"):
                return None
            if any(v for key, v in source.args.items() if key not in ("this", "alias")):
                return None
            model = self._flat(source.this)
            if model is None:
                return None
            names = {(name or "").lower(): (what, payload) for name, what, payload in model.outputs}
            if "" in names or len(names) != len(model.outputs) or not scope.add(alias, set(names)):
                return None
            ends = self._equality(parts[0], scope) if len(parts) == 1 else None
            if ends is None:
                return None
            own, other = ends if ends[0][0] == alias else ends[::-1]
            if own[0] != alias or other[0] not in {p.alias for p in parents}:
                return None
            what, payload = names[own[1]]
            first = model.parents[0]
            if what != "parent" or type(payload) is not exp.Column or payload.table.lower() != first.alias or payload.name.lower() not in self.one_column_keys(first.table):
                return None
            if payload.name.lower() not in model.held:
                return None  # grouped by the joined key: each of its values is one row of the table
            if kind == "left":
                # only where a foreign key says every outer row finds its row: the padded row would read NULL, not 0
                holder = next(p for p in parents if p.alias == other[0])
                real = self._real(parents)
                if holder.alias not in real or other[1] not in self.never_null(holder.table) or not self.references(holder.table, other[1], first.table, payload.name.lower()):
                    return None
                if len(model.parents) != 1:
                    return None  # a lookup inside could drop the row the foreign key promises
            grouped = {"alias": alias, "model": model, "names": names, "key": payload.name.lower(), "other": other}
        if not counts and grouped is None:
            return None
        names = {p.alias for p in parents}
        if grouped is not None:
            return self._over_grouped(select, scope, parents, grouped)
        first = next(iter(counts.values()))
        guards: dict = {}
        for info in counts.values():
            if _full(info["table"]) != _full(first["table"]) or info["key"] != first["key"]:
                return None
            guards.update(info["guards"])

        def count(value: exp.Expression) -> frozenset | None:
            value = _unwrap(value)
            column = None
            if type(value) is exp.Coalesce and len(value.expressions) == 1 and _literal(value.expressions[0], "0"):
                column = value.this
            elif type(value) is exp.Case and value.args.get("this") is None and len(value.args.get("ifs") or []) == 1:
                branch, default = value.args["ifs"][0], value.args.get("default")
                target = _is_null(branch.this)
                if target is not None and _literal(branch.args["true"], "0") and default is not None and scope.owner(default) is not None and scope.owner(default) == scope.owner(target):
                    column = target
            owner = scope.owner(column) if column is not None else None
            return counts[owner[0]]["counts"].get(owner[1]) if owner is not None and owner[0] in counts else None

        outputs = self._outputs(select, scope, names, count)
        if outputs is None:
            return None
        return _Model(parents, first["table"], first["key"], links, outputs, guards)

    def _real(self, parents: list[_Parent]) -> set[str]:
        """The parents that hold a row on every joined row: the driving table and the inner lookups from such a table."""

        real = {parents[0].alias}
        for parent in parents[1:]:
            if parent.kind == "inner" and parent.on[1][0] in real:
                real.add(parent.alias)
        return real

    def _over_grouped(self, select: exp.Select, scope: _Scope, parents: list[_Parent], grouped: dict) -> _Model | None:
        model, alias = grouped["model"], grouped["alias"]
        taken = {p.alias for p in parents} | {alias}
        renames = {}
        for parent in model.parents:
            fresh, index = parent.alias, 0
            while fresh in taken:
                fresh, index = f"kqp{index}", index + 1
            taken.add(fresh)
            renames[parent.alias] = fresh

        def moved(node: exp.Expression) -> exp.Expression:
            copy = node.copy()
            for column in [copy] if isinstance(copy, exp.Column) else list(copy.find_all(exp.Column)):
                column.set("table", exp.to_identifier(renames[column.table.lower()]))
            return copy

        inner = []
        for parent in model.parents:
            table = parent.table.copy()
            table.set("alias", exp.TableAlias(this=exp.to_identifier(renames[parent.alias])))
            on = None if parent.on is None else ((renames[parent.on[0][0]], parent.on[0][1]), (renames[parent.on[1][0]], parent.on[1][1]))
            inner.append(_Parent(table, renames[parent.alias], parent.columns, parent.kind, on))
        inner[0].kind, inner[0].on = "inner", ((inner[0].alias, grouped["key"]), grouped["other"])
        names = {p.alias for p in parents}

        outputs = []
        for item in select.expressions:
            name = item.alias if isinstance(item, exp.Alias) else None
            value = _unwrap(item.this if isinstance(item, exp.Alias) else item)
            if type(value) is exp.Coalesce and len(value.expressions) == 1 and _literal(value.expressions[0], "0"):
                owner = scope.owner(value.this)
                if owner is None or owner[0] != alias or grouped["names"][owner[1]][0] != "count":
                    return None
                outputs.append((name, "count", grouped["names"][owner[1]][1]))
                continue
            owner = scope.owner(value)
            if owner is not None and owner[0] == alias:
                what, payload = grouped["names"][owner[1]]
                outputs.append((name or value.name, what, moved(payload) if what == "parent" else payload))
                continue
            parent = self._parent_value(value, scope, names)
            if parent is None:
                return None
            outputs.append((name or (value.name if isinstance(value, exp.Column) else None), "parent", parent))
        links = [(renames[a], c) for a, c in model.links]
        return _Model(parents + inner, model.child, model.child_key, links, outputs, model.guards)

    # --- the one spelling -------------------------------------------------------------------------------------------

    def _write(self, model: _Model) -> exp.Select | None:
        parents = model.parents
        driving = parents[0]
        tables = [_full(p.table) for p in parents]
        counted = sorted({tuple(sorted(tests)) for _, what, tests in model.outputs if what == "count"})
        if len(set(tables)) != len(tables) or not counted:
            return None
        by_alias = {p.alias: p for p in parents}
        real = {driving.alias}
        for parent in parents[1:]:
            own, other = parent.on
            holder = by_alias[other[0]]
            if parent.kind == "left" and holder.alias in real and other[1] in self.never_null(holder.table) and self.references(holder.table, other[1], parent.table, own[1]):
                parent.kind = "inner"  # a declared foreign key from a NOT NULL column: every row finds its one row
            if parent.kind == "inner" and holder.alias in real:
                real.add(parent.alias)
        # the parent columns the inner lookups make equal to the linked one
        equal = {model.links[0]}
        changed = True
        while changed:
            changed = False
            for parent in parents[1:]:
                if parent.kind != "inner":
                    continue
                own, other = parent.on
                if (own in equal) != (other in equal):
                    equal |= {own, other}
                    changed = True
        if any(link not in equal for link in model.links):
            return None
        kinds = {self.type_of(by_alias[alias].table, name) for alias, name in equal} | {self.type_of(model.child, model.child_key)}
        if len(kinds) > 1:
            return None
        link = min(equal, key=lambda column: (_full(by_alias[column[0]].table), column[1]))

        child = _CHILD
        if child in by_alias:
            return None
        listed = [exp.alias_(exp.column(model.child_key, table=_INNER), model.child_key)]
        for index, tests in enumerate(counted):
            if tests:
                flag = exp.Case(ifs=[exp.If(this=_and_all([self._test(t, model) for t in tests]), true=exp.Literal.number(1))])
                listed.append(exp.alias_(exp.Count(this=flag), f"kqn{index}"))
            else:
                listed.append(exp.alias_(exp.Count(this=exp.Star()), f"kqn{index}"))
        source = model.child.copy()
        source.set("alias", exp.TableAlias(this=exp.to_identifier(_INNER)))
        inner = exp.Select(expressions=listed)
        inner.set(FROM_KEY, exp.From(this=source))
        inner.set("group", exp.Group(expressions=[exp.column(model.child_key, table=_INNER)]))
        joined = exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier(child)))

        items = []
        for alias, what, payload in model.outputs:
            if what == "parent":
                value = payload.copy()
            else:
                name = f"kqn{counted.index(tuple(sorted(payload)))}"
                value = exp.Coalesce(this=exp.column(name, table=child), expressions=[exp.Literal.number(0)])
            items.append(exp.alias_(value, alias) if alias else value)
        result = exp.Select(expressions=items)
        result.set(FROM_KEY, exp.From(this=driving.table.copy()))
        joins = []
        for parent in parents[1:]:
            own, other = parent.on
            on = exp.EQ(this=exp.column(own[1], table=own[0]), expression=exp.column(other[1], table=other[0]))
            joins.append(exp.Join(this=parent.table.copy(), on=on, **({"side": "LEFT"} if parent.kind == "left" else {})))
        on = exp.EQ(this=exp.column(link[1], table=link[0]), expression=exp.column(model.child_key, table=child))
        joins.append(exp.Join(this=joined, on=on, side="LEFT"))
        result.set("joins", joins)
        return result

    def _test(self, test: tuple, model: _Model) -> exp.Expression:
        if test[0] == "notnull":
            return exp.Not(this=exp.Is(this=exp.column(test[1], table=_INNER), expression=exp.Null()))
        if test[0] == "member":
            listed = exp.to_table(test[2])
            listed.set("alias", exp.TableAlias(this=exp.to_identifier(_MEMBER)))
            query = exp.Select(expressions=[exp.column(test[3], table=_MEMBER)])
            query.set(FROM_KEY, exp.From(this=listed))
            return exp.In(this=exp.column(test[1], table=_INNER), query=exp.Subquery(this=query))
        return model.guards[test].copy()

    def read(self, select: exp.Select) -> exp.Select | None:
        model = self._joined(select)
        if model is None:
            model = self._flat(select)
            if model is not None and not self._one_group_per_row(model):
                return None
        return self._write(model) if model is not None else None


def read_child_counts(
    tree: exp.Expression,
    schema: dict | None,
    keys: dict | None,
    not_null: dict | None = None,
    foreign_keys: dict | None = None,
    types: dict | None = None,
) -> tuple[exp.Expression, bool]:
    """``tree`` with every select described above written in the one spelling, and whether any was rewritten."""

    if not schema or not keys:
        return tree, False
    from .counted_membership import _unstable

    reader, fired = _Reader(schema, keys, not_null, foreign_keys, types), False
    for select in list(tree.find_all(exp.Select)):
        root = select
        while root is not None and root is not tree:
            root = root.parent
        if root is None or _unstable(select):
            continue  # replaced along with an enclosing select, or not the same value twice
        target = select
        if select is tree and select.args.get("order") is not None and not select.args.get("limit") and not select.args.get("offset"):
            target = select.copy()  # the order of the whole result is not part of what is compared
            target.set("order", None)
        new = reader.read(target)
        if new is None or new.sql() == target.sql():
            continue  # not read, or already in the one spelling
        fired = True
        if select is tree:
            tree = new
        else:
            select.replace(new)
    return tree, fired
