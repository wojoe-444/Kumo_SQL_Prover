"""A row asked whether its own value is shared: a grouped count of its table is a test for another row.

    SELECT p.pid FROM policy AS p
    WHERE p.city IN (SELECT city FROM policy GROUP BY city HAVING COUNT(*) > 1)

keeps a policy when its city holds more than one policy. The policy is one of them, so this asks whether
*another* policy has the same city, and with ``pid`` a declared key that is NOT NULL it is

    SELECT p.pid FROM policy AS p
    WHERE EXISTS (SELECT 1 FROM policy AS q WHERE q.city = p.city AND q.pid <> p.pid)

People write the question both ways (and a third way, ``p.pid IN (SELECT a.pid FROM policy AS a JOIN policy AS b
ON a.pid <> b.pid AND a.city = b.city)``, which the prover already reads as the ``EXISTS``). The grouped form is
opaque to the prover: the count is a free value. This rule gives it the ``EXISTS`` form.

**Why it holds.** Let ``r`` be a row of table ``T`` that the outer select reads, and ``G`` the subquery
``SELECT k FROM T GROUP BY k HAVING <count> <op> <n>`` over the same table with no ``WHERE``. When ``r.k`` is not
NULL, ``r`` is a member of the group of ``G`` with key ``r.k``, and that is the only group ``r.k`` can equal.

* *Counting rows* (``COUNT(*)``, ``COUNT(1)``, ``COUNT`` of a column declared NOT NULL or of a grouping column,
  ``COUNT(DISTINCT key)``): the group has more than one row exactly when a row other than ``r`` has the same
  ``k``. Two rows of ``T`` are different rows exactly when their key differs (the key is unique and never NULL),
  so "another row" is ``q.k = r.k AND q.key <> r.key``.
* *Counting values* (``COUNT(DISTINCT x)`` with ``x`` declared NOT NULL): ``r.x`` is one of the group's values, so
  the group has more than one exactly when some row with the same ``k`` has ``q.x <> r.x``. No key is needed.

With ``E`` for that ``EXISTS``:

=============================================  ======================================
``r.k IN (G: more than one)``                  ``E``
``r.k IN (G: exactly one)``                    ``r.k IS NOT NULL AND NOT E``
``r.k NOT IN (G: more than one)``              ``NOT E``  (``k`` declared NOT NULL)
``r.k NOT IN (G: exactly one)``                ``E``      (``k`` declared NOT NULL)
=============================================  ======================================

The test is a top-level conjunct of the ``WHERE``, where a test that is not TRUE drops the row, so the first two
lines need only agree on when they are TRUE. A NULL ``r.k`` makes ``IN`` not TRUE and ``E`` FALSE (first line); the
second line says so itself. The group of ``r`` always has at least one row, which is why "not more than one" is
"exactly one" in the last two lines; there ``k`` must be declared NOT NULL, because ``NOT IN`` is NULL for a NULL
``r.k`` and for a list that holds a NULL (the NULL group), and ``NOT E`` would be TRUE.

"More than one" is ``> 1``, ``>= 2`` or ``<> 1``; "exactly one" is ``= 1``, ``< 2`` or ``<= 1`` (a group never
counts zero rows, and never zero values of a NOT NULL column). Several columns work the same way:
``(r.a, r.b) IN (SELECT a, b FROM T GROUP BY a, b HAVING ..)``.

**What is read.** One conjunct of the ``WHERE`` of a select: ``IN`` or ``NOT IN`` of a column (or a tuple of
columns) of one source ``r``, against a subquery that selects exactly its grouping columns, the same columns of the
same base table in the same order, with one comparison of one count with 1 or 2 in its ``HAVING`` and nothing else
(the normalizer's own spelling of such a subquery, a select from the grouped derived table with the test in
``WHERE``, is read too). ``r`` is the first source or one joined by an inner or cross join, a plain base table the
supplied column lists know.

**Left alone.** A subquery with a ``WHERE``, a join, more grouping columns than it selects, another threshold or a
second ``HAVING`` conjunct; a tested column that is not the grouping column itself; a row count without a declared
one-column key that is NOT NULL; a column declared floating-point (a NaN is its own group but equals nothing in
some engines); a select with a ``RIGHT`` or ``FULL`` join or ``USING``; a source on the padded side of a ``LEFT JOIN``; a test
under ``OR``, in a ``HAVING`` or in a select list; a column whose owner is not certain.

``tests/test_own_group_membership.py`` replays every line of the table on DuckDB with NULLs and duplicates, and
shows the NOT NULL and key conditions are needed.
"""

from __future__ import annotations

from sqlglot import exp

from . import smt_values

_GROUPED_BANNED = ("joins", "where", "qualify", "windows", "with_", "with", "limit", "offset", "order", "laterals", "pivots", "into", "locks", "sample", "cluster", "distribute", "sort", "connect", "match", "kind", "hint")
_LIST_BANNED = ("joins", "group", "having", "qualify", "windows", "with_", "with", "limit", "offset", "order", "laterals", "pivots", "into", "locks", "sample", "cluster", "distribute", "sort", "connect", "match", "kind", "hint")
_JOIN_EXTRAS = ("method", "using", "match_condition", "global_", "hint", "directed", "expressions", "pivots")
_MORE, _ONE = "more than one", "exactly one"


def _unparen(node: exp.Expression | None) -> exp.Expression | None:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _plain_table(source: exp.Expression | None) -> bool:
    return isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier) and not any(source.args.get(k) for k in ("db", "catalog", "pivots", "sample", "joins", "version", "when"))


def _from(select: exp.Select) -> exp.Expression | None:
    from_ = select.args.get("from_") or select.args.get("from")
    return from_.this if from_ is not None else None


def _listed(table: exp.Table, schema: dict | None) -> set[str] | None:
    """The columns the supplied lists give ``table`` (lower case), None when they do not know it."""

    for name, columns in (schema or {}).items():
        if str(name).lower().split(".")[-1] == table.name.lower():
            return {str(c).lower() for c in columns}
    return None


def _facts(facts: dict | None, table: str):
    for name, value in (facts or {}).items():
        if str(name).lower().split(".")[-1] == table.lower():
            return value
    return None


def _own_column(node: exp.Expression | None, alias: str, listed: set[str]) -> str | None:
    """The name of a plain column of the one source ``alias`` (bare, or qualified by it), None otherwise."""

    node = _unparen(node)
    if not isinstance(node, exp.Column) or not isinstance(node.this, exp.Identifier) or node.args.get("db") or node.this.args.get("quoted"):
        return None
    if node.table and node.table.lower() != alias:
        return None
    return node.name.lower() if node.name.lower() in listed else None


def _threshold(node: exp.Expression | None) -> tuple[exp.Expression, str] | None:
    """``(the counted side, "more than one" | "exactly one")`` for a comparison of something with 1 or 2."""

    node = _unparen(node)
    kinds = {exp.GT: ">", exp.GTE: ">=", exp.LT: "<", exp.LTE: "<=", exp.EQ: "=", exp.NEQ: "<>"}
    if type(node) not in kinds:
        return None
    op = kinds[type(node)]
    left, right = _unparen(node.this), _unparen(node.expression)
    if isinstance(left, exp.Literal):
        left, right = right, left
        op = {">": "<", ">=": "<=", "<": ">", "<=": ">=", "=": "=", "<>": "<>"}[op]
    if not isinstance(right, exp.Literal) or right.is_string or right.this not in ("1", "2"):
        return None
    meaning = {(">", "1"): _MORE, (">=", "2"): _MORE, ("<>", "1"): _MORE, ("=", "1"): _ONE, ("<", "2"): _ONE, ("<=", "1"): _ONE}.get((op, right.this))
    return (left, meaning) if meaning else None


class _Groups:
    """``SELECT k.. FROM T GROUP BY k.. HAVING <count> <op> <n>``: the table, the selected columns and what is counted."""

    def __init__(self, query: exp.Expression | None, schema: dict | None):
        self.ok = False
        query = query.this if isinstance(query, exp.Subquery) and not query.alias else query
        if not isinstance(query, exp.Select):
            return
        distinct = query.args.get("distinct")
        if distinct is not None and distinct.args.get("on"):
            return
        source = _from(query)
        if isinstance(source, exp.Subquery) and isinstance(source.this, exp.Select) and source.alias:
            read = self._through_derived(query, source)
        else:
            read = self._direct(query)
        if read is None:
            return
        grouped, outputs, counted, self.meaning = read
        table = _from(grouped)
        if not _plain_table(table) or any(grouped.args.get(k) for k in _GROUPED_BANNED):
            return
        listed = _listed(table, schema)
        group = grouped.args.get("group")
        if listed is None or group is None or any(group.args.get(k) for k in ("grouping_sets", "rollup", "cube", "totals", "all")):
            return
        alias = table.alias_or_name.lower()
        keys = [_own_column(g, alias, listed) for g in group.expressions]
        self.columns = [_own_column(o, alias, listed) for o in outputs]
        if None in keys or None in self.columns or len(set(keys)) != len(keys) or len(self.columns) != len(keys) or set(self.columns) != set(keys):
            return
        self.table = table
        self.count = self._count(counted, alias, listed)
        self.ok = self.count is not None

    @staticmethod
    def _direct(query: exp.Select):
        having = query.args.get("having")
        found = _threshold(having.this) if having is not None else None
        if found is None:
            return None
        return query, [o.unalias() for o in query.expressions], found[0], found[1]

    @staticmethod
    def _through_derived(query: exp.Select, source: exp.Subquery):
        """The normalizer's spelling: ``SELECT g.k FROM (SELECT k AS k0, COUNT(..) AS c0 FROM T GROUP BY k) AS g WHERE g.c0 <op> <n>``."""

        grouped = source.this
        where = query.args.get("where")
        found = _threshold(where.this) if where is not None else None
        if found is None or any(query.args.get(k) for k in _LIST_BANNED) or grouped.args.get("having") is not None or grouped.args.get("distinct") is not None:
            return None
        named = {o.alias.lower(): o.unalias() for o in grouped.expressions if isinstance(o, exp.Alias)}
        if len(named) != len(grouped.expressions):
            return None

        def inner(node: exp.Expression | None) -> exp.Expression | None:
            node = _unparen(node)
            if not isinstance(node, exp.Column) or not isinstance(node.this, exp.Identifier) or node.args.get("db"):
                return None
            if node.table and node.table.lower() != source.alias.lower():
                return None
            return named.get(node.name.lower())

        outputs = [inner(o.unalias()) for o in query.expressions]
        counted = inner(found[0])
        if counted is None or None in outputs:
            return None
        return grouped, outputs, counted, found[1]

    def _count(self, counted: exp.Expression, alias: str, listed: set[str]) -> tuple[str, str | None] | None:
        """``("rows", None)``, ``("rows", c)`` when that needs ``c`` to be NOT NULL or the key, or ``("values", x)``."""

        counted = _unparen(counted)
        if not isinstance(counted, exp.Count) or counted.args.get("expressions"):
            return None
        argument = _unparen(counted.this)
        if isinstance(argument, (exp.Star, exp.Literal)):
            return ("rows", None)
        if isinstance(argument, exp.Distinct):
            if len(argument.expressions) != 1:
                return None
            column = _own_column(argument.expressions[0], alias, listed)
            return ("values", column) if column is not None and column not in self.columns else None
        column = _own_column(argument, alias, listed)
        if column is None:
            return None
        return ("rows", None) if column in self.columns else ("rows", column)


def _real_sources(select: exp.Select) -> tuple[dict[str, exp.Table], list[str]] | None:
    """The base tables that hold a real row wherever the ``WHERE`` is evaluated (by alias) and every source's alias; None when the select is not read."""

    first = _from(select)
    if first is None:
        return None
    joins = select.args.get("joins") or []
    sources = [(first, True)]
    for join in joins:
        side = (join.args.get("side") or "").upper()
        kind = join.args.get("kind")
        kind = kind.upper() if isinstance(kind, str) else ""
        if side in ("RIGHT", "FULL") or any(join.args.get(k) for k in _JOIN_EXTRAS) or kind not in ("", "INNER", "CROSS"):
            return None
        sources.append((join.this, side == ""))
    names = [s.alias_or_name.lower() for s, _ in sources]
    if "" in names or len(set(names)) != len(names):
        return None
    return {s.alias_or_name.lower(): s for s, real in sources if real and _plain_table(s)}, names


def _fresh(select: exp.Select) -> str:
    root = select
    while root.parent is not None:
        root = root.parent
    used = {i.name.lower() for i in root.find_all(exp.Identifier)}
    index = 0
    while f"ogm{index}" in used:
        index += 1
    return f"ogm{index}"


def read_own_group_membership(select: exp.Select, schema: dict | None, keys: dict | None, not_null: dict | None, types: dict | None = None) -> exp.Select | None:
    """``select`` with one membership test against its own table's grouped count read as a test for another row."""

    from .algebraic_equivalence import _conjuncts

    where = select.args.get("where")
    if where is None or not schema or where.find(exp.In) is None:
        return None
    found = _real_sources(select)
    if found is None:
        return None
    real, names = found
    conditions = _conjuncts(where.this)
    for index, condition in enumerate(conditions):
        node = _unparen(condition)
        negated = isinstance(node, exp.Not)
        if negated:
            node = _unparen(node.this)
        if not isinstance(node, exp.In) or node.args.get("expressions") or node.args.get("unnest") or node.args.get("query") is None:
            continue
        replacement = _read(select, node, negated, real, names, schema, keys, not_null, types)
        if replacement is None:
            continue
        result = select.copy()
        rebuilt = [c.copy() for c in conditions[:index]] + replacement + [c.copy() for c in conditions[index + 1 :]]
        result.set("where", exp.Where(this=exp.and_(*rebuilt, copy=False)))
        return result
    return None


def _read(select: exp.Select, node: exp.In, negated: bool, real: dict[str, exp.Table], names: list[str], schema: dict | None, keys: dict | None, not_null: dict | None, types: dict | None) -> list[exp.Expression] | None:
    groups = _Groups(node.args["query"], schema)
    if not groups.ok:
        return None
    tested = _unparen(node.this)
    values = list(tested.expressions) if isinstance(tested, exp.Tuple) else [tested]
    if len(values) != len(groups.columns):
        return None
    row: exp.Table | None = None
    for value, column in zip(values, groups.columns):
        value = _unparen(value)
        if not isinstance(value, exp.Column) or not isinstance(value.this, exp.Identifier) or value.args.get("db") or value.this.args.get("quoted") or value.name.lower() != column:
            return None
        if value.table:
            owner = real.get(value.table.lower())
        else:
            # a bare name: certain only when every source is a known table and exactly one has the column
            owners = [alias for alias in names if alias in real and column in (_listed(real[alias], schema) or ())]
            owner = real[owners[0]] if len(owners) == 1 and all(alias in real and _listed(real[alias], schema) is not None for alias in names) else None
        if owner is None or (row is not None and owner is not row):
            return None
        row = owner
    if row is None or row.name != groups.table.name or bool(row.this.args.get("quoted")) != bool(groups.table.this.args.get("quoted")):
        return None
    listed = _listed(row, schema)
    if listed is None or any(c not in listed for c in groups.columns):
        return None
    declared = {str(c).lower() for c in (_facts(not_null, row.name) or ())}
    kinds = {str(k).lower(): v for k, v in (_facts(types, row.name) or {}).items()}

    def floating(column: str) -> bool:
        return smt_values.type_class(kinds.get(column)) == smt_values.FLOAT64

    if any(floating(c) for c in groups.columns):
        return None
    what, column = groups.count
    if what == "values":
        if column not in declared or floating(column):
            return None
        apart = column
    else:
        singles = sorted(k[0].lower() for k in (_facts(keys, row.name) or ()) if len(k) == 1 and str(k[0]).lower() in declared and not floating(str(k[0]).lower()))
        if not singles or (column is not None and column not in declared):
            return None
        apart = singles[0]
        if apart in groups.columns:
            return None  # every group of a key has one row
    if negated and any(c not in declared for c in groups.columns):
        return None
    other = _fresh(select)
    alias = row.args.get("alias")
    mine = (alias.this if isinstance(alias, exp.TableAlias) else row.this).copy()

    def column_of(table: exp.Identifier | str, name: str) -> exp.Column:
        return exp.Column(this=exp.to_identifier(name), table=table.copy() if isinstance(table, exp.Identifier) else exp.to_identifier(table))

    ties = [exp.EQ(this=column_of(other, c), expression=column_of(mine, c)) for c in groups.columns]
    ties.append(exp.NEQ(this=column_of(other, apart), expression=column_of(mine, apart)))
    source = exp.Table(this=row.this.copy(), alias=exp.TableAlias(this=exp.to_identifier(other)))
    probe = exp.Select(expressions=[exp.Literal.number(1)])
    probe.set("from_" if "from_" in select.args else "from", exp.From(this=source))
    probe.set("where", exp.Where(this=exp.and_(*ties, copy=False)))
    exists: exp.Expression = exp.Exists(this=probe)
    another = (groups.meaning == _MORE) != negated
    if another:
        return [exists]
    present = [] if negated else [exp.Not(this=exp.Is(this=column_of(mine, c), expression=exp.Null())) for c in groups.columns if c not in declared]
    return present + [exp.Not(this=exists)]
