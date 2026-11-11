"""A grouped count over a ``UNION ALL`` of reads of one source is a sum of indicators over that source.

    SELECT p.id, p.name, COUNT(*) AS wins
    FROM players AS p
    JOIN (SELECT g.first AS who FROM games AS g UNION ALL SELECT g.second FROM games AS g) AS w ON p.id = w.who
    GROUP BY p.id

unpivots two columns of ``games`` and counts, per player, the rows of the union that name the player. It is

    SELECT p.id, p.name,
           SUM(CASE WHEN p.id = g.first THEN 1 ELSE 0 END + CASE WHEN p.id = g.second THEN 1 ELSE 0 END) AS wins
    FROM players AS p CROSS JOIN games AS g
    GROUP BY p.id
    HAVING SUM(CASE WHEN p.id = g.first THEN 1 ELSE 0 END + CASE WHEN p.id = g.second THEN 1 ELSE 0 END) > 0

which reads ``games`` once and counts, for every pair of a player and a game, how many branches of the union
would have produced a row for it. The prover compares grouped selects source by source, so it relates the second
form to any other spelling over ``players`` and ``games`` and has nothing to match the union with.

**The rule.** Let the select have a plain ``GROUP BY`` and, among its sources (base tables and one derived
table ``u``, joined by inner or cross joins only), let ``u`` be a ``UNION ALL`` of ``n`` branches

    SELECT e_i1, .., e_im FROM f [WHERE / ON w_i]

over the same base tables ``f`` (the same tables the same number of times; the aliases may differ). An output
column of ``u`` is *uniform* when every branch computes it with the same expression and *varying* otherwise.
Call the top-level ``AND`` parts of the select's ``WHERE`` and ``ON`` clauses that read a varying column the
*linking* parts, and write ``c_i`` for: the linking parts with branch ``i``'s expressions in place of ``u``'s
columns, and ``w_i``. Then the select is rewritten to read ``f`` in place of ``u``, without the linking parts,
with

* ``COUNT(*)`` (and ``COUNT`` of a value that cannot be NULL) as ``SUM(ind_1 + .. + ind_n)``, where ``ind_i`` is
  ``CASE WHEN c_i THEN 1 ELSE 0 END``;
* ``COUNT(x)`` as the same sum with ``x_i IS NOT NULL`` added to each ``c_i`` (``x_i``: ``x`` as branch ``i``
  computes it);
* ``SUM(x)``, for an ``x`` every branch computes from integer literals alone (``1``, a ``CASE`` over them), as
  ``SUM(CASE WHEN c_1 THEN x_1 ELSE 0 END + ..)``, where a ``CASE WHEN y IS NULL THEN a ELSE b END`` inside
  ``x_i`` is written as ``b`` when ``c_i`` being TRUE leaves ``y`` no way to be NULL (see below);
* a uniform column, anywhere, as its expression; a varying column outside the aggregates and the linking parts
  as the expression a linking part ``u.v = e`` equates it with;
* and ``HAVING SUM(ind_1 + .. + ind_n) > 0`` added.

Three readings widen what counts as uniform or as needing no test. ``x_i IS NOT NULL`` is left out of ``c_i``
when ``x_i`` is a literal, a column declared NOT NULL, or an operand of a comparison among the ``AND`` parts of
``c_i`` (a comparison is TRUE only when neither operand is NULL). An output that every branch's own conditions
equate with one and the same column (branch ``i`` returns ``g.first`` under ``ON p.id = g.first``, branch ``j``
returns ``g.second`` under ``ON p.id = g.second``) is read as that column, which it equals on every row the
branch returns. A bare column name that several sources have is read as any of them when the remaining
equalities of the select equate them all. A ``HAVING`` part that says the same as the added one (``s > 0``,
``s >= 1`` or ``s <> 0`` on the count, written out or by its output name) is dropped: the indicator sum is a
non-negative integer, so the three tests are the added test.

**Why it is sound.** Inner and cross joins make every ``ON`` part a filter of the product of the sources, and
``UNION ALL`` keeps every row of every branch, so the rows the original select groups are the triples (a row
``x`` of the other sources, a branch ``i``, a row ``r`` of ``f``) on which ``c_i`` and the remaining parts are
TRUE. The rewritten select groups the pairs ``(x, r)`` on which the remaining parts are TRUE. Group keys read no
varying column once ``u.v`` is replaced by the ``e`` it equals on every kept triple (``=`` is TRUE only for two
equal values that are not NULL, and the prover reads ``=`` as identity everywhere), so a triple and its pair
have the same key: the original group of a key is the set of triples over the pairs of the rewritten group.

* ``COUNT(*)`` counts those triples: for each pair the number of ``i`` with ``c_i`` TRUE, which is
  ``ind_1 + .. + ind_n``, summed over the group's pairs. ``COUNT(x)`` counts the triples whose ``x_i`` is not
  NULL, and ``SUM(x)`` adds ``x_i`` over the triples; each ``x_i`` is then built from integer literals, never
  NULL, so no row is skipped and the added zeros change nothing. The ``CASE`` has an ``ELSE``, a rewritten group
  has a row, so the rewritten sums are never NULL, as the original counts and sums are not (an original group
  has a triple).
* A rewritten group whose pairs carry no triple does not exist in the original: its indicator sum is 0 and the
  added ``HAVING`` removes it. Every other group has a triple and a positive sum. With no rows in ``f`` or in
  the other sources both selects have no groups.
* Everything outside the aggregates must be fixed by the group key in the rewritten select, which has more rows
  per group than the original: it is a key expression, or a column of a base table whose declared key (all of
  its columns declared NOT NULL) is among the key columns. Anything else (MySQL's loose ``GROUP BY`` picking a
  row) is refused.
* ``HAVING``, ``ORDER BY`` and ``DISTINCT`` of the select act on the grouped rows, which are the same.

**What is refused.** ``UNION`` (it removes duplicates, so a row counts once however many branches give it), a
branch with ``DISTINCT``, grouping, aggregates, windows, ``LIMIT``, a nested query or a function whose value is
not fixed by the row, branches over different tables, outer joins in the select or in a branch (a NULL-padded
row is not a filter of the product), ``USING``, a star, a second derived table, ``LIMIT``, grouping sets, any
other aggregate (``MIN``, ``AVG``, ``COUNT(DISTINCT ..)``, a ``SUM`` of a column), a varying column read outside
an aggregate with no equality to replace it, a column that cannot be attributed to one source, and a
select-list alias read where a source column of the same name exists.
"""

from __future__ import annotations

from collections import Counter

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key
from .ast_utils import FROM_KEY, conjuncts, extended_grouping, select_sources

_PREFIX = "kqu"
_JOIN_KINDS = {"", "INNER", "CROSS"}
_BRANCH_ARGS = frozenset({"expressions", FROM_KEY, "from", "joins", "where"})
_SELECT_ARGS = _BRANCH_ARGS | {"group", "having", "order", "distinct"}
_NOT_ROW_VALUES = (exp.Select, exp.Subquery, exp.Exists, exp.Window, exp.AggFunc, exp.Rand, exp.Anonymous, exp.Star)
_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)


def count_over_union_all(
    select: exp.Expression,
    schema: dict[str, list[str]] | None,
    keys: dict[str, list[tuple[str, ...]]] | None,
    not_null: dict[str, frozenset[str]] | None,
) -> exp.Expression | None:
    """The select rewritten as the module docstring describes, or ``None`` when it is not that shape."""

    if not schema or not isinstance(select, exp.Select):
        return None
    group = select.args.get("group")
    if group is None or not group.expressions or extended_grouping(group):
        return None
    if any(value for key, value in select.args.items() if key not in _SELECT_ARGS):
        return None
    distinct = select.args.get("distinct")
    if distinct is not None and distinct.args.get("on"):
        return None
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    site = _Site(select, known)
    if not site.ok:
        return None
    return site.rewritten({name.lower(): listed for name, listed in (keys or {}).items()}, not_null or {})


def _plain_joins(select: exp.Select) -> bool:
    for join in select.args.get("joins") or []:
        if any(value for key, value in join.args.items() if key not in ("this", "on", "kind")):
            return False
        if (join.args.get("kind") or "").upper() not in _JOIN_KINDS:
            return False
    return True


def _union_all_branches(query: exp.Expression) -> list[exp.Select] | None:
    """The selects of a ``UNION ALL`` tree, left to right; ``None`` for anything else."""

    while isinstance(query, exp.Subquery) and not any(value for key, value in query.args.items() if key != "this"):
        query = query.this
    if isinstance(query, exp.Select):
        return [query]
    if type(query) is not exp.Union or query.args.get("distinct") is not False:
        return None
    if any(value for key, value in query.args.items() if key not in ("this", "expression", "distinct")):
        return None
    left, right = _union_all_branches(query.this), _union_all_branches(query.expression)
    return None if left is None or right is None else left + right


def _row_value(node: exp.Expression) -> bool:
    return not any(isinstance(n, _NOT_ROW_VALUES) for n in node.walk())


class _Branch:
    """One branch: its base tables, and its outputs and conditions with every column qualified."""

    def __init__(self, select: exp.Select, known: dict[str, list[str]]):
        self.tables: list[tuple[str, str]] = []  # (table name, alias) in FROM order
        self.sources: list[exp.Table] = []
        self.outputs: list[exp.Expression] = []
        self.names: list[str] = []
        self.conditions: list[exp.Expression] = []
        self.ok = self._read(select, known)

    def _read(self, select: exp.Select, known: dict[str, list[str]]) -> bool:
        if any(value for key, value in select.args.items() if key not in _BRANCH_ARGS) or not _plain_joins(select):
            return False
        columns: dict[str, list[str]] = {}
        spelling: dict[str, str] = {}
        sources = select_sources(select)
        if not sources:
            return False
        for source in sources:
            listed = _base_table(source, known)
            alias = source.alias_or_name.lower() if isinstance(source, exp.Table) else ""
            if listed is None or not alias or alias in columns:
                return False
            columns[alias] = listed
            spelling[alias] = source.alias_or_name
            self.tables.append((_table_key(source), alias))
            self.sources.append(source)
        parts = [part for join in select.args.get("joins") or [] if join.args.get("on") is not None for part in conjuncts(join.args["on"])]
        if select.args.get("where") is not None:
            parts += conjuncts(select.args["where"].this)
        for item in select.expressions:
            value = _qualified(item.unalias(), columns, spelling)
            if value is None:
                return False
            self.outputs.append(value)
            self.names.append(item.alias_or_name.lower())
        for part in parts:
            value = _qualified(part, columns, spelling)
            if value is None:
                return False
            self.conditions.append(value)
        return True

    def renamed(self, aliases: dict[str, list[str]]) -> None:
        """Name the k-th read of each table as ``aliases`` says, so that the branches share one set of sources."""

        seen: Counter[str] = Counter()
        mapping = {}
        for name, alias in self.tables:
            mapping[alias] = aliases[name][seen[name]]
            seen[name] += 1
        for node in self.outputs + self.conditions:
            for column in node.find_all(exp.Column):
                column.set("table", exp.to_identifier(mapping[column.table.lower()]))


    def equal_to(self, column: exp.Column) -> set[tuple[str, str]]:
        """The columns (source, name) this branch's conditions make equal to ``column`` on each of its rows."""

        found = {(column.table, column.name)}
        grew = True
        while grew:
            grew = False
            for part in self.conditions:
                if isinstance(part, exp.EQ) and isinstance(part.this, exp.Column) and isinstance(part.expression, exp.Column):
                    pair = {(part.this.table, part.this.name), (part.expression.table, part.expression.name)}
                    if pair & found and not pair <= found:
                        found |= pair
                        grew = True
        return found


def _qualified(node: exp.Expression, columns: dict[str, list[str]], spelling: dict[str, str]) -> exp.Expression | None:
    """A copy of a row expression with each column named by its source's alias; ``None`` when one cannot be."""

    if not _row_value(node):
        return None
    node = exp.Paren(this=node.copy())  # a parent, so that a bare column can be replaced
    for column in list(node.find_all(exp.Column)):
        owner = _owner(column, columns)
        if owner is None:
            return None
        column.replace(exp.column(column.name.lower(), spelling[owner]))
    return node.this


def _owner(column: exp.Column, columns: dict[str, list[str]]) -> str | None:
    if column.args.get("db") or column.args.get("catalog") or not isinstance(column.this, exp.Identifier):
        return None
    name = column.name.lower()
    if column.table:
        alias = column.table.lower()
        return alias if name in columns.get(alias, ()) else None
    owners = [alias for alias, listed in columns.items() if name in listed]
    return owners[0] if len(owners) == 1 else None


def _integer_value(node: exp.Expression) -> bool:
    """Built from integer literals alone (through ``CASE .. ELSE``, ``+``, ``-``, ``*``): an integer, never NULL."""

    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Literal):
        return not node.is_string and node.this.isdigit()
    if isinstance(node, exp.Neg):
        return _integer_value(node.this)
    if isinstance(node, (exp.Add, exp.Sub, exp.Mul)):
        return _integer_value(node.this) and _integer_value(node.expression)
    if isinstance(node, exp.Case) and node.args.get("this") is None and node.args.get("default") is not None:
        return all(_integer_value(arm.args["true"]) for arm in node.args["ifs"]) and _integer_value(node.args["default"])
    if isinstance(node, exp.If) and not isinstance(node.parent, exp.Case) and node.args.get("false") is not None:
        return _integer_value(node.args["true"]) and _integer_value(node.args["false"])
    return False


def _all(parts: list[exp.Expression]) -> exp.Expression | None:
    result = None
    for part in parts:
        part = part.copy()
        if isinstance(part, (exp.Or, exp.Xor)):
            part = exp.Paren(this=part)
        result = part if result is None else exp.And(this=result, expression=part)
    return result


def _indicator_sum(conditions: list[list[exp.Expression]], values: list[exp.Expression]) -> exp.Expression:
    """``CASE WHEN c_1 THEN v_1 ELSE 0 END + ..``, one term per branch."""

    total = None
    for parts, value in zip(conditions, values):
        test = _all(parts)
        term = value.copy() if test is None else exp.Case(ifs=[exp.If(this=test, true=value.copy())], default=exp.Literal.number(0))
        total = term if total is None else exp.Add(this=total, expression=term)
    return total


class _Site:
    """A grouped select over one ``UNION ALL`` source and base tables."""

    def __init__(self, select: exp.Select, known: dict[str, list[str]]):
        self.select = select
        self.ok = self._read(known)

    def _read(self, known: dict[str, list[str]]) -> bool:
        select = self.select
        if not _plain_joins(select):
            return False
        sources = select_sources(select)
        unions = [s for s in sources if isinstance(s, exp.Subquery)]
        if len(unions) != 1:
            return False
        self.union = unions[0]
        if any(value for key, value in self.union.args.items() if key not in ("this", "alias")):
            return False
        alias = self.union.args.get("alias")
        if alias is None or not self.union.alias or alias.args.get("columns"):
            return False
        selects = _union_all_branches(self.union.this)
        if selects is None or len(selects) < 2:
            return False
        # no other nested query and no window: every expression of the select is a value of one of its rows
        for node in select.find_all(exp.Select, exp.Subquery, exp.Exists, exp.Window):
            if node is not select and node is not self.union and not _inside(node, self.union):
                return False
        self.branches = [_Branch(s, known) for s in selects]
        if not all(b.ok for b in self.branches):
            return False
        first = self.branches[0]
        width = len(first.outputs)
        if any(len(b.outputs) != width for b in self.branches) or len(set(first.names)) != width or not all(first.names):
            return False
        tables = Counter(name for name, _ in first.tables)
        if any(Counter(name for name, _ in b.tables) != tables for b in self.branches):
            return False
        self.columns: dict[str, list[str]] = {}  # the select's sources: alias -> column names
        self.tables: dict[str, str] = {}  # alias -> base table name, for the base tables
        self.spelling: dict[str, str] = {}  # alias -> the alias as the select writes it
        for source in sources:
            name = source.alias_or_name.lower()
            if not name or name in self.columns:
                return False
            self.spelling[name] = source.alias_or_name
            if source is self.union:
                self.columns[name] = list(first.names)
                continue
            listed = _base_table(source, known)
            if listed is None:
                return False
            self.columns[name] = listed
            self.tables[name] = _table_key(source)
        self.alias = self.union.alias.lower()
        # the union's tables join the select's sources, under names no other source has
        taken = set(self.columns) - {self.alias}
        aliases: dict[str, list[str]] = {}
        self.inner: list[tuple[exp.Table, str, str]] = []  # (table, its name, its alias in the rewritten select)
        for (name, own), table in zip(first.tables, first.sources):
            own = table.alias_or_name
            if own.lower() in taken:
                number = 0
                while f"{_PREFIX}{number}" in taken:
                    number += 1
                own = f"{_PREFIX}{number}"
            taken.add(own.lower())
            aliases.setdefault(name, []).append(own)
            self.inner.append((table, name, own))
        for branch in self.branches:
            branch.renamed(aliases)
        self.position = {name: i for i, name in enumerate(first.names)}
        for i in range(width):
            # a column every branch's own conditions equate with one and the same column is that column
            values = [b.outputs[i] for b in self.branches]
            if all(isinstance(value, exp.Column) for value in values) and len({value.sql() for value in values}) > 1:
                shared = set.intersection(*[b.equal_to(value) for b, value in zip(self.branches, values)])
                if shared:
                    table, name = min(shared)
                    for b in self.branches:
                        b.outputs[i] = exp.column(name, table)
        self.uniform = {
            name: first.outputs[i] for name, i in self.position.items() if all(b.outputs[i].sql() == first.outputs[i].sql() for b in self.branches)
        }
        return True

    def _resolved(self, node: exp.Expression) -> exp.Expression | None:
        """A copy with every column named by its source, ``None`` when a column is not one source's."""

        return _qualified(node, self.columns, self.spelling)

    def _varying(self, node: exp.Expression) -> bool:
        return any(c.table.lower() == self.alias and c.name.lower() not in self.uniform for c in node.find_all(exp.Column))

    def _in_branch(self, node: exp.Expression, branch: _Branch | None) -> exp.Expression | None:
        """A resolved expression with the union's columns replaced: as a branch computes them, or the uniform ones."""

        node = exp.Paren(this=node.copy())
        for column in list(node.find_all(exp.Column)):
            if column.table.lower() != self.alias:
                continue
            name = column.name.lower()
            if branch is not None:
                value = branch.outputs[self.position[name]]
            elif name in self.uniform:
                value = self.uniform[name]
            else:
                return None
            column.replace(_operand(value))
        return node.this

    def rewritten(self, keys: dict[str, list[tuple[str, ...]]], not_null: dict[str, frozenset[str]]) -> exp.Expression | None:
        select = self.select
        names = {item.alias.lower() for item in select.expressions if isinstance(item, exp.Alias)}
        pool: list[exp.Expression] = []
        for join in select.args.get("joins") or []:
            if join.args.get("on") is not None:
                pool += conjuncts(join.args["on"])
        if select.args.get("where") is not None:
            pool += conjuncts(select.args["where"].this)
        linking, kept = [], []
        for part in pool:
            part = self._resolved(part)
            if part is None:
                return None
            if self._varying(part):
                linking.append(part)
            else:
                kept.append(self._in_branch(part, None))
        # what a varying column equals on every row the select keeps
        equal: dict[str, exp.Expression] = {}
        for part in linking:
            if not isinstance(part, exp.EQ):
                continue
            for mine, other in ((part.this, part.expression), (part.expression, part.this)):
                if isinstance(mine, exp.Column) and mine.table.lower() == self.alias and not self._varying(other):
                    equal.setdefault(mine.name.lower(), self._in_branch(other, None))
        same = _equated(linking + kept)
        conditions = [[self._in_branch(part, branch) for part in linking] + [c.copy() for c in branch.conditions] for branch in self.branches]
        compared = [
            {side.sql() for part in parts if isinstance(part, _COMPARISONS) for side in (part.this, part.expression) if isinstance(side, exp.Column)}
            for parts in conditions
        ]
        ones = [exp.Literal.number(1) for _ in self.branches]
        declared = {name.lower(): {c.lower() for c in listed} for name, listed in not_null.items()}
        base = dict(self.tables)
        base.update({alias.lower(): name for _, name, alias in self.inner})

        def never_null(value: exp.Expression, branch: int) -> bool:
            value = value.unnest()
            if isinstance(value, exp.Literal):
                return True
            if not isinstance(value, exp.Column):
                return False
            # a NOT NULL column, or one a comparison of this branch's condition rejects when NULL
            return value.name.lower() in declared.get(base.get(value.table.lower(), ""), ()) or value.sql() in compared[branch]

        def aggregate(node: exp.Expression) -> exp.Expression | None:
            if type(node) not in (exp.Count, exp.Sum) or isinstance(node.parent, exp.Filter):
                return None
            if any(value for key, value in node.args.items() if key not in ("this", "big_int")) or isinstance(node.this, exp.Distinct):
                return None
            argument = node.this
            if isinstance(node, exp.Count) and (argument is None or isinstance(argument, exp.Star)):
                return exp.Sum(this=_indicator_sum(conditions, ones))
            argument = None if argument is None else self._resolved(argument)
            if argument is None:
                return None
            values = [self._in_branch(argument, branch) for branch in self.branches]
            if isinstance(node, exp.Count):
                tests = [
                    parts if never_null(value, i) else parts + [exp.Not(this=exp.Is(this=_operand(value), expression=exp.Null()))]
                    for i, (parts, value) in enumerate(zip(conditions, values))
                ]
                return exp.Sum(this=_indicator_sum(tests, ones))
            if not all(_integer_value(value) for value in values):
                return None
            return exp.Sum(this=_indicator_sum(conditions, [settled(value, i) for i, value in enumerate(values)]))

        def settled(value: exp.Expression, branch: int) -> exp.Expression:
            """``value`` with each ``CASE WHEN y IS NULL THEN a ELSE b END`` read as ``b`` where ``y`` is never NULL."""

            value = exp.Paren(this=value.copy())
            for case in list(value.find_all(exp.Case)):
                ifs = case.args.get("ifs") or []
                if case.this is not None or len(ifs) != 1 or case.args.get("default") is None:
                    continue
                test = ifs[0].this.unnest()
                if isinstance(test, exp.Is) and isinstance(test.expression, exp.Null) and never_null(test.this, branch):
                    case.replace(case.args["default"].copy())
            value = value.this
            while isinstance(value, exp.Paren) and isinstance(value.this, (exp.Literal, exp.Column)):
                value = value.this
            return value

        def outside(node: exp.Expression, allowed: set[str] = frozenset()) -> exp.Expression | None:
            """A copy whose columns outside the aggregates are read from the rewritten select's sources."""

            node = exp.Paren(this=node.copy())
            for column in list(node.find_all(exp.Column)):
                if column.find_ancestor(exp.AggFunc) is not None:
                    continue
                name = column.name.lower()
                if not column.table and name in allowed:
                    if any(name in listed for listed in self.columns.values()):
                        return None  # the output of that name or the source column: the dialects disagree
                    continue
                owner = _owner(column, self.columns)
                if owner is None and not column.table:
                    owner = _equated_owner(name, self.columns, same)
                if owner is None:
                    return None
                if owner != self.alias:
                    column.replace(exp.column(name, self.spelling[owner]))
                elif name in self.uniform:
                    column.replace(_operand(self.uniform[name]))
                elif name in equal:
                    column.replace(_operand(equal[name]))
                else:
                    return None
            return node.this

        def finished(node: exp.Expression) -> exp.Expression | None:
            """``outside`` with the aggregates rewritten as well."""

            node = exp.Paren(this=node)
            for call in [c for c in node.find_all(exp.AggFunc) if c.find_ancestor(exp.AggFunc) is None]:
                replacement = aggregate(call)
                if replacement is None:
                    return None
                call.replace(replacement)
            return node.this

        group_keys = []
        for key in select.args["group"].expressions:
            value = outside(key)
            if value is None or any(value.find_all(exp.AggFunc)):
                return None
            group_keys.append(value)
        items, read = [], []
        for item in select.expressions:
            value = outside(item.unalias())
            if value is None:
                return None
            read.append(value)
            items.append(value)
        having = None
        if select.args.get("having") is not None:
            having = outside(select.args["having"].this, names)
            read.append(having)
        ordered = []
        for term in select.args["order"].expressions if select.args.get("order") is not None else []:
            ordered.append(outside(term.this, names))
            read.append(ordered[-1])
        if any(node is None for node in read) or not _fixed(read, group_keys, keys, declared, base, names):
            return None
        # every check has passed except the aggregates' own
        copy = select.copy()
        outputs = []
        for item, value in zip(select.expressions, items):
            value = finished(value)
            if value is None:
                return None
            name = item.alias_or_name
            renamed = isinstance(item, exp.Column) and not (isinstance(value, exp.Column) and value.name.lower() == name.lower())
            outputs.append(exp.alias_(value, name, copy=False) if name and (isinstance(item, exp.Alias) or renamed) else value)
        test = exp.GT(this=exp.Sum(this=_indicator_sum(conditions, ones)), expression=exp.Literal.number(0))
        if having is not None:
            having = finished(having)
            if having is None:
                return None
            test = _with_positive(having, test, outputs)
        for position, value in enumerate(ordered):
            value = finished(value)
            if value is None:
                return None
            copy.args["order"].expressions[position].set("this", value)
        tables = []
        for source in select_sources(select):
            if source is not self.union:
                tables.append(source.copy())
                continue
            for table, _, alias in self.inner:
                table = table.copy()
                if table.alias_or_name != alias:
                    table.set("alias", exp.TableAlias(this=exp.to_identifier(alias)))
                tables.append(table)
        copy.set("expressions", outputs)
        copy.set(FROM_KEY, exp.From(this=tables[0]))
        copy.set("joins", [exp.Join(this=table, kind="CROSS") for table in tables[1:]] or None)
        where = _all(kept)
        copy.set("where", exp.Where(this=where) if where is not None else None)
        copy.set("group", exp.Group(expressions=group_keys))
        copy.set("having", exp.Having(this=test))
        return copy


def _fixed(read: list[exp.Expression], group_keys: list[exp.Expression], keys, declared, base, names: set[str]) -> bool:
    """Whether everything read outside an aggregate is fixed by the group key (see the module docstring)."""

    key_sql = {key.unnest().sql() for key in group_keys}
    grouped: dict[str, set[str]] = {}
    for key in group_keys:
        key = key.unnest()
        if isinstance(key, exp.Column):
            grouped.setdefault(key.table.lower(), set()).add(key.name.lower())
    determined = set()
    for alias, have in grouped.items():
        table = base.get(alias, "")
        for key in keys.get(table, []):
            wanted = {c.lower() for c in key}
            if wanted and wanted <= have and wanted <= declared.get(table, set()):
                determined.add(alias)

    def fixed(node: exp.Expression) -> bool:
        if isinstance(node, exp.AggFunc) or node.sql() in key_sql:
            return True
        if isinstance(node, exp.Column):
            return node.table.lower() in determined if node.table else node.name.lower() in names
        return all(fixed(child) for child in node.iter_expressions())

    return all(fixed(node) for node in read)


def _with_positive(having: exp.Expression, test: exp.GT, outputs: list[exp.Expression]) -> exp.Expression:
    """``having AND test``, without the parts of ``having`` that say what ``test`` says.

    ``test`` is ``s > 0`` for a sum ``s`` of indicators, an integer that is never negative, so ``s >= 1``,
    ``s <> 0`` and ``s > 0`` again (``s`` written out, or read by the name of the output that is exactly ``s``)
    add nothing to it.
    """

    total = test.this.sql()
    named = {item.alias.lower() for item in outputs if isinstance(item, exp.Alias) and item.this.sql() == total}

    def is_total(node: exp.Expression) -> bool:
        node = node.unnest()
        return node.sql() == total or (isinstance(node, exp.Column) and not node.table and node.name.lower() in named)

    def number(node: exp.Expression) -> str | None:
        node = node.unnest()
        return node.this if isinstance(node, exp.Literal) and not node.is_string else None

    def implied(part: exp.Expression) -> bool:
        if not isinstance(part, (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NEQ)):
            return False
        if is_total(part.this):
            return (type(part), number(part.expression)) in ((exp.GT, "0"), (exp.GTE, "1"), (exp.NEQ, "0"))
        if is_total(part.expression):
            return (type(part), number(part.this)) in ((exp.LT, "0"), (exp.LTE, "1"), (exp.NEQ, "0"))
        return False

    kept = [part for part in conjuncts(having) if not implied(part)]
    rest = _all(kept)
    return test if rest is None else exp.And(this=exp.Paren(this=rest) if isinstance(rest, exp.Or) else rest, expression=test)


def _equated(parts: list[exp.Expression]) -> set[frozenset[str]]:
    """The pairs of columns (as printed, lower case) that a top-level ``=`` of the conditions equates."""

    pairs = set()
    for part in parts:
        if isinstance(part, exp.EQ) and isinstance(part.this, exp.Column) and isinstance(part.expression, exp.Column):
            pairs.add(frozenset({part.this.sql().lower(), part.expression.sql().lower()}))
    return pairs


def _equated_owner(name: str, columns: dict[str, list[str]], same: set[frozenset[str]]) -> str | None:
    """The source to read a bare column from when several have it: any, if the conditions make them all equal.

    ``GROUP BY id`` over ``a JOIN b ON a.id = b.id`` names the same value whichever source it is read from, on
    every row the select keeps.
    """

    owners = [alias for alias, listed in columns.items() if name in listed]
    if len(owners) < 2:
        return None
    first = owners[0]
    if all(frozenset({f"{first}.{name}", f"{other}.{name}"}) in same for other in owners[1:]):
        return first
    return None


def _inside(node: exp.Expression, ancestor: exp.Expression) -> bool:
    parent = node.parent
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.parent
    return False


def _operand(node: exp.Expression) -> exp.Expression:
    node = node.copy()
    return node if isinstance(node, (exp.Column, exp.Literal, exp.Paren)) else exp.Paren(this=node)
