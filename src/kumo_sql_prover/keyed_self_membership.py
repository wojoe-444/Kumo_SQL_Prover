"""A membership test that finds the row's own table again through its key reads the row itself.

    SELECT s.id FROM sessions AS s
    WHERE s.id NOT IN (SELECT p.id FROM sessions AS p JOIN ads AS a ON a.customer = p.customer WHERE a.at >= p.start)

asks, for each session, whether the session is among the sessions that have an ad. With ``id`` a key of
``sessions`` that question is about the row ``s`` itself:

    SELECT s.id FROM sessions AS s
    WHERE NOT EXISTS (SELECT 1 FROM ads AS a WHERE a.customer = s.customer AND a.at >= s.start)

The same holds for ``EXISTS (SELECT .. FROM sessions AS p JOIN ads AS a ON .. WHERE p.id = s.id AND ..)``.

**The rule.** Let ``x`` be a source of a select that is a base table ``t``, and ``K`` a set of columns of ``t``
that contains a declared key of ``t`` and whose columns are all declared NOT NULL. In the select's ``WHERE``,

    (x.k1, .., x.kn) IN (SELECT p.k1, .., p.kn FROM t AS p, r1, .., rm WHERE C)        -- K = {k1 .. kn}
    EXISTS (SELECT .. FROM t AS p, r1, .., rm WHERE p.k1 = x.k1 AND .. AND p.kn = x.kn AND C)

are both replaced by

    EXISTS (SELECT 1 FROM r1, .., rm WHERE C[p := x])

where ``C`` collects the subquery's ``WHERE`` and the ``ON`` conditions of its (inner) joins, and ``C[p := x]``
reads every column of ``p`` from ``x`` instead. With no other source (``m = 0``) the replacement is ``C[p := x]``
itself, see below.

**Why it is sound.**

* ``x`` is a row of ``t``: it is a base-table source that no outer join of its select can NULL-pad (it is not
  the right side of a ``LEFT JOIN``, not left of a ``RIGHT JOIN`` and not under a ``FULL JOIN``), and the test is
  in the ``WHERE``, which is evaluated on joined rows.
* ``p`` ranges over every row of ``t``: it is a base-table source of the subquery and all the subquery's joins
  are inner or cross joins, so the subquery's rows are exactly the combinations ``(p, r1 .. rm)`` that make
  ``C`` TRUE.
* ``p.k = x.k`` on every column of ``K`` holds for exactly one row ``p``, namely ``x`` itself: the columns are NOT
  NULL, so the row ``x`` satisfies it, and ``K`` contains a key, so no other row does. Hence "some combination
  ``(p, r..)`` with ``p.k = x.k`` makes ``C`` TRUE" is "some combination ``(r..)`` makes ``C[p := x]`` TRUE".
* The ``IN`` is never NULL: its left operands are NOT NULL columns of a real row and the subquery's outputs are
  NOT NULL columns of real rows (inner joins only), so it is TRUE when a row matches and FALSE otherwise. An
  ``EXISTS`` is never NULL either. Both sides being two-valued, the replacement is good under ``NOT``, ``OR``
  and anywhere else in the ``WHERE``.
* Duplicates do not matter: ``IN`` and ``EXISTS`` only ask whether a row exists, so the subquery's ``DISTINCT``
  and ``ORDER BY`` (without a row limit) are dropped.

With no other source the subquery has at most the one row ``p = x``, and the test is "``C[p := x]`` is TRUE".
That is written as the bare condition, which is the same thing when the condition cannot be NULL (it is built
from ``EXISTS``, ``IS NULL``, boolean constants, ``AND``, ``OR`` and ``NOT`` only, or there is no condition at
all), or when the test is a top-level ``AND`` part of the ``WHERE`` in positive position, where NULL and FALSE
both drop the row. Otherwise the test is kept.

**What is read, and what is refused.**

* ``x``: a plain base table with supplied columns among the sources of the select whose ``WHERE`` holds the
  test (the nearest enclosing select), not NULL-padded as said above. The select's joins are inner, cross or
  outer joins without ``USING`` or ``NATURAL``, and its sources have distinct names. A left operand is
  ``x.col``, or a bare ``col`` when every source of the select is a base table with supplied columns and ``x``
  is the only one that has it.
* The subquery: one select with nothing but outputs, ``FROM``, inner or cross joins with optional ``ON``,
  ``WHERE``, ``DISTINCT`` and ``ORDER BY``. Grouping, ``HAVING``, aggregates, windows, ``LIMIT``, ``USING``,
  outer joins, set operations and ``WITH`` are refused. Every source is a plain base table with supplied
  columns and the names are distinct. (The rule runs in the step loop, after ``USING`` is written as ``ON``,
  ``WITH`` tables are inlined, bare columns are qualified and an outer join whose padded rows the ``WHERE``
  rejects has become an inner join.)
* ``p``: a source of the subquery that is the same table as ``x``. For ``IN`` the outputs are plain columns of
  ``p``, the left operands the same columns of ``x``, position by position. For ``EXISTS`` the equalities
  ``p.k = x.k`` are top-level ``AND`` parts of the subquery's ``WHERE`` or ``ON`` with ``x`` named by its
  qualifier, and ``x``'s name is not a source of the subquery.
* Names. Every column of ``p`` inside the subquery is found and requalified with ``x``'s name: ``p.col``, or a
  bare ``col`` in the subquery's own scope that only ``p`` has. The rewrite is refused when a column could be
  ``p``'s and this cannot be settled (a bare name of ``t`` inside a nested query, ``p.*``), when a nested query
  declares a source with ``p``'s name (its ``p.col`` would not be ours) or when anything inside the subquery
  other than ``p`` is named like ``x`` (it would capture the new ``x.col``).
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key, _unwrap
from .ast_utils import FROM_KEY, conjuncts, select_sources, visible_ctes

_SUBQUERY_ARGS = frozenset({"expressions", "from", "from_", "joins", "where", "distinct", "order", "comments", "kind", "hint"})
_INNER = ("", "INNER", "CROSS")


def keyed_self_membership(
    select: exp.Select,
    schema: dict[str, list[str]] | None,
    keys: dict[str, list[tuple[str, ...]]] | None,
    not_null: dict[str, frozenset[str]] | None,
) -> exp.Expression | None:
    """``select`` (in place) with each test described in the module docstring rewritten, else ``None``."""

    where = select.args.get("where")
    if not schema or not keys or where is None:
        return None
    if not any(isinstance(node, (exp.In, exp.Exists)) for node in where.walk()):
        return None
    known = {name.lower(): [c.lower() for c in columns] for name, columns in schema.items()}
    declared_keys = {name.lower(): [tuple(c.lower() for c in key) for key in table_keys if key] for name, table_keys in keys.items()}
    declared = {name.lower(): {c.lower() for c in columns} for name, columns in (not_null or {}).items()}
    outer = _Scope.read(select, known)
    if outer is None or not outer.kept:
        return None
    changed = False
    for node in list(where.find_all(exp.In, exp.Exists)):
        if node.find_ancestor(exp.Select) is not select or not _still_inside(node, where):
            continue
        plan = _plan(node, outer, known, declared_keys, declared)
        if plan is None:
            continue
        replacement = _replacement(node, where, plan, outer.written[plan.x])
        if replacement is None:
            continue
        node.replace(replacement)
        changed = True
    return select if changed else None


def _still_inside(node: exp.Expression, root: exp.Expression) -> bool:
    while node is not None:
        if node is root:
            return True
        node = node.parent
    return False


def _name(source: exp.Expression) -> str:
    return source.alias_or_name.lower()


def _plain_table(source: exp.Expression, known: dict[str, list[str]]) -> list[str] | None:
    """The supplied columns of a base-table source that is not a ``WITH`` table, else None."""

    columns = _base_table(source, known)
    if columns is None or not _name(source):
        return None
    if not source.args.get("db") and source.name.lower() in visible_ctes(source):
        return None
    return columns


class _Scope:
    """The sources of the select that holds the test: which are base tables, and which no outer join can pad."""

    def __init__(self, names: list[str], columns: dict[str, list[str] | None], tables: dict[str, str], kept: set[str], written: dict[str, str]):
        self.names, self.columns, self.tables, self.kept, self.written = names, columns, tables, kept, written

    @classmethod
    def read(cls, select: exp.Select, known: dict[str, list[str]]) -> "_Scope | None":
        sources = select_sources(select)
        joins = select.args.get("joins") or []
        if not sources or len(sources) != len(joins) + 1:
            return None
        names = [_name(source) for source in sources]
        if "" in names or len(set(names)) != len(names):
            return None
        padded: set[int] = set()
        for index, join in enumerate(joins):
            side, kind = (join.side or "").upper(), (join.kind or "").upper()
            if join.method or join.args.get("using") or kind not in ("", "INNER", "CROSS", "OUTER") or side not in ("", "LEFT", "RIGHT", "FULL"):
                return None
            if side in ("LEFT", "FULL"):
                padded.add(index + 1)
            if side in ("RIGHT", "FULL"):
                padded.update(range(index + 1))
        columns = {name: _plain_table(source, known) for name, source in zip(names, sources)}
        tables = {name: _table_key(source) for name, source in zip(names, sources) if columns[name] is not None}
        kept = {name for index, name in enumerate(names) if index not in padded and columns[name] is not None}
        return cls(names, columns, tables, kept, {name: source.alias_or_name for name, source in zip(names, sources)})

    def owner(self, column: exp.Expression) -> tuple[str, str] | None:
        """``(source name, column name)`` for a plain column of this select that names one of its base tables."""

        column = _unwrap(column)
        if type(column) is not exp.Column or isinstance(column.this, exp.Star) or column.args.get("db") or column.args.get("catalog"):
            return None
        name = column.name.lower()
        if column.table:
            source = column.table.lower()
            if self.columns.get(source) is None or name not in self.columns[source]:
                return None
            return source, name
        if any(columns is None for columns in self.columns.values()):
            return None
        owners = [source for source in self.names if name in self.columns[source]]
        return (owners[0], name) if len(owners) == 1 else None


class _Plan:
    """One test to rewrite: the subquery, its source ``p``, the columns of its sources, the name of ``x`` and the
    ``p.k = x.k`` parts that go away."""

    def __init__(self, query: exp.Select, p: exp.Table, columns: dict[str, list[str]], x: str, drop: list[exp.Expression]):
        self.query, self.p, self.columns, self.x, self.drop = query, p, columns, x, drop


def _plain_query(query: exp.Expression | None, known: dict[str, list[str]]) -> tuple[exp.Select, list[exp.Table], dict[str, list[str]]] | None:
    """The subquery's select, its sources and their columns by name, when it is a plain inner join of base tables."""

    if not isinstance(query, exp.Select):
        return None
    if any(value for key, value in query.args.items() if key not in _SUBQUERY_ARGS):
        return None
    distinct = query.args.get("distinct")
    if distinct is not None and (not isinstance(distinct, exp.Distinct) or distinct.args.get("on") or distinct.expressions):
        return None
    sources = select_sources(query)
    joins = query.args.get("joins") or []
    if not sources or len(sources) != len(joins) + 1:
        return None
    for join in joins:
        if join.side or join.method or (join.kind or "").upper() not in _INNER:
            return None
        if any(value for key, value in join.args.items() if key not in ("this", "on", "kind", "comments")):
            return None
    columns: dict[str, list[str]] = {}
    for source in sources:
        cols = _plain_table(source, known)
        if cols is None or _name(source) in columns:
            return None
        columns[_name(source)] = cols
    for node in query.walk():
        if isinstance(node, (exp.With, exp.CTE, exp.Window, exp.Lambda)):
            return None
        if isinstance(node, exp.AggFunc) and node.find_ancestor(exp.Select) is query:
            return None  # an aggregate of the subquery itself: one row whatever the tables hold
    return query, sources, columns


def _own_column(column: exp.Expression, query: exp.Select, p_name: str, columns: dict[str, list[str]]) -> str | None:
    """The column name when ``column`` is a plain column of ``p`` in the subquery's own scope, else None."""

    column = _unwrap(column)
    if type(column) is not exp.Column or isinstance(column.this, exp.Star) or column.args.get("db") or column.args.get("catalog"):
        return None
    if column.find_ancestor(exp.Select) is not query:
        return None
    name = column.name.lower()
    if column.table:
        return name if column.table.lower() == p_name and name in columns[p_name] else None
    owners = [source for source, cols in columns.items() if name in cols]
    return name if owners == [p_name] else None


def _plan(node: exp.Expression, outer: _Scope, known, declared_keys, declared) -> _Plan | None:
    if isinstance(node, exp.In):
        subquery = node.args.get("query")
        if subquery is None or node.expressions or node.args.get("unnest") or node.args.get("field"):
            return None
        if not isinstance(subquery, exp.Subquery) or any(value for key, value in subquery.args.items() if key not in ("this", "comments")):
            return None
        read = _plain_query(subquery.this, known)
    else:
        read = _plain_query(node.this, known)
    if read is None:
        return None
    query, sources, columns = read
    if isinstance(node, exp.In):
        left = _unwrap(node.this)
        lefts = list(left.expressions) if isinstance(left, exp.Tuple) else [left]
        owners = [outer.owner(operand) for operand in lefts]
        if None in owners or len({owner[0] for owner in owners}) != 1 or len(lefts) != len(query.expressions):
            return None
        x = owners[0][0]
        outputs = [item.this if isinstance(item, exp.Alias) else item for item in query.expressions]
        for p in sources:
            compared = [_own_column(output, query, _name(p), columns) for output in outputs]
            if None in compared or compared != [owner[1] for owner in owners]:
                continue
            if _settles(x, outer, _table_key(p), set(compared), declared_keys, declared):
                return _Plan(query, p, columns, x, [])
        return None
    parts = _condition_parts(query)
    for p in sources:
        compared: set[str] = set()
        x = None
        drop = []
        for part in parts:
            pair = _key_equality(part, query, _name(p), columns, outer)
            if pair is None or (x is not None and pair[0] != x):
                continue
            x = pair[0]
            compared.add(pair[1])
            drop.append(part)
        if x is not None and _settles(x, outer, _table_key(p), compared, declared_keys, declared):
            return _Plan(query, p, columns, x, drop)
    return None


def _settles(x: str, outer: _Scope, table: str, compared: set[str], declared_keys, declared) -> bool:
    """Whether equality on ``compared`` with the kept row ``x`` names that row of ``table`` and no other."""

    if x not in outer.kept or outer.tables.get(x) != table:
        return False
    if not compared <= declared.get(table, set()):
        return False
    return any(set(key) <= compared for key in declared_keys.get(table, []))


def _condition_parts(query: exp.Select) -> list[exp.Expression]:
    where = query.args.get("where")
    parts = [part for join in query.args.get("joins") or [] if join.args.get("on") is not None for part in conjuncts(join.args["on"])]
    return parts + (conjuncts(where.this) if where is not None else [])


def _key_equality(part: exp.Expression, query: exp.Select, p_name: str, columns: dict[str, list[str]], outer: _Scope) -> tuple[str, str] | None:
    """``(x, k)`` when ``part`` is ``p.k = x.k`` with ``x`` a source, named by its qualifier, of the select around the
    subquery."""

    if not isinstance(part, exp.EQ):
        return None
    for ours, theirs in ((part.this, part.expression), (part.expression, part.this)):
        name = _own_column(ours, query, p_name, columns)
        theirs = _unwrap(theirs)
        if name is None or type(theirs) is not exp.Column or not theirs.table or theirs.args.get("db") or theirs.args.get("catalog"):
            continue
        x = theirs.table.lower()
        if x in columns or x not in outer.names or theirs.name.lower() != name:
            continue
        return x, name
    return None


def _columns_of_p(plan: _Plan, parts: list[exp.Expression]) -> list[exp.Column] | None:
    """Every column of ``p`` in the conditions that stay, or None when a name cannot be settled."""

    query, p_name = plan.query, _name(plan.p)
    table_columns = set(plan.columns[p_name])
    if plan.x in plan.columns and plan.x != p_name:
        return None  # another source of the subquery is named like x
    found = []
    for part in parts:
        for node in part.walk():
            if isinstance(node, (exp.Table, exp.Subquery, exp.CTE)) and node.alias_or_name.lower() in (p_name, plan.x):
                return None  # a nested source named like p (its p.col is not ours) or like x (it would capture x.col)
            if not isinstance(node, exp.Column):
                continue
            if node.args.get("db") or node.args.get("catalog"):
                return None
            if isinstance(node.this, exp.Star):
                if not node.table or node.table.lower() == p_name:
                    return None
                continue
            name = node.name.lower()
            if node.table:
                if node.table.lower() == p_name:
                    if name not in table_columns:
                        return None
                    found.append(node)
            elif node.find_ancestor(exp.Select) is query:
                owners = [source for source, cols in plan.columns.items() if name in cols]
                if p_name in owners:
                    if len(owners) != 1:
                        return None
                    found.append(node)
            elif name in table_columns:
                return None  # a bare name of the table inside a nested query: whose it is depends on that query's sources
    return found


def _two_valued(node: exp.Expression) -> bool:
    node = _unwrap(node)
    if isinstance(node, (exp.And, exp.Or)):
        return _two_valued(node.this) and _two_valued(node.expression)
    if isinstance(node, exp.Not):
        return _two_valued(node.this)
    if isinstance(node, exp.Is):
        return isinstance(node.expression, exp.Null)
    return isinstance(node, (exp.Exists, exp.Boolean))


def _replacement(node: exp.Expression, where: exp.Where, plan: _Plan, x_written: str) -> exp.Expression | None:
    query, p = plan.query, plan.p
    parts = [part for part in _condition_parts(query) if not any(part is dropped for dropped in plan.drop)]
    found = _columns_of_p(plan, parts)
    if found is None:
        return None
    rest = [source for source in select_sources(query) if source is not p]
    if not rest:
        positive = any(part is node for part in conjuncts(where.this))
        if not positive and not all(_two_valued(part) for part in parts):
            return None
    # nothing is refused from here on
    for column in found:
        column.set("table", exp.to_identifier(x_written))
    parts = [exp.Paren(this=part.copy()) if isinstance(part, exp.Or) else part.copy() for part in parts]
    condition = None
    for part in parts:
        condition = part if condition is None else exp.And(this=condition, expression=part)
    if not rest:
        if condition is None:
            return exp.true()
        return exp.Paren(this=condition) if len(parts) > 1 else condition
    replacement = exp.Select(expressions=[exp.Literal.number(1)])
    replacement.set(FROM_KEY, exp.From(this=rest[0].copy()))
    if len(rest) > 1:
        replacement.set("joins", [exp.Join(this=source.copy()) for source in rest[1:]])
    if condition is not None:
        replacement.set("where", exp.Where(this=condition))
    return exp.Exists(this=replacement)
