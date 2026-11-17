"""A presence test need not repeat what an absence test beside it already says.

    SELECT p.id FROM product AS p
    WHERE p.id IN (SELECT s.pid FROM sales AS s WHERE s.day >= 5 AND s.day <= 9)
      AND NOT EXISTS (SELECT 1 FROM sales AS t WHERE t.pid = p.id AND (t.day < 5 OR t.day > 9))

The ``NOT EXISTS`` keeps a product only when none of its sales has ``day < 5 OR day > 9``. With ``day`` declared
NOT NULL that condition is TRUE or FALSE for every row, so every sale of a kept product has ``day >= 5`` and
``day <= 9``. The ``IN`` test then asks its rows for something they all have, and is

    p.id IN (SELECT s.pid FROM sales AS s)

"Has a sale inside the range and none outside" and "has a sale and none outside" are one question, and people
write both.

**What is read.** Two top-level conjuncts of one ``WHERE`` (there a test that is not TRUE drops the row, FALSE
or NULL alike, and when the absence test is FALSE the row is dropped whatever the presence test says):

* the absence test ``NOT EXISTS (SELECT .. FROM t [AS a] WHERE ..)`` over one base table, whose ``WHERE`` is made
  of ties ``a.c = e`` (``e`` reads no column of ``a``) and of one condition ``D`` built with ``AND``, ``OR`` and
  ``NOT`` from comparisons (``=``, ``<>``, ``<``, ``<=``, ``>``, ``>=``, ``BETWEEN``, ``IN`` of a list) of a
  column of ``a`` declared NOT NULL with a literal. Such a ``D`` is never NULL.
* the presence test ``EXISTS (SELECT .. FROM t [AS b] WHERE ..)`` or ``e IN (SELECT b.c FROM t [AS b] WHERE ..)``
  over the same base table, with at least the absence test's ties (the same columns tied to the same
  expressions).

Every row the presence test can count satisfies the ties, so it has ``NOT D``. ``NOT D`` is written as a
conjunction of comparisons (the negation is pushed inward, which is exact in three-valued logic too), and each
conjunct of the presence test's ``WHERE`` that is made only of those is dropped.

**Through a derived table that only filters.** ``SELECT c.id FROM (SELECT p.id FROM product AS p WHERE EXISTS (..))
AS c WHERE NOT EXISTS (.. t.pid = c.id ..)`` asks the two tests of the same row of ``product``: ``c.id`` is ``p.id``
of the row behind it. It is read the same way when the select has that one source and no join, and the derived
table lists plain columns of one base table with no join, grouping, aggregate, window, ``DISTINCT``, ``ORDER BY``
or ``LIMIT``. A row the shorter presence test lets into the derived table fails the absence test outside it.

**Left alone.** A subquery with a join, grouping, an aggregate, a window, ``LIMIT``, a set operation or a
volatile call; an absence test with any other conjunct (a nested query, a comparison of two columns, a column
not declared NOT NULL); a column whose owner is not certain (a bare name needs the supplied column list); an
outer select with two sources whose names differ only by case; the tests under ``OR``, under ``NOT``, in a
``HAVING`` or in a select list.

``tests/test_presence_given_absence.py`` replays the rewrite on DuckDB with NULL keys and shows why the NOT NULL
declaration is needed.
"""

from __future__ import annotations

from sqlglot import exp

_BANNED = ("joins", "group", "having", "qualify", "windows", "with_", "with", "limit", "offset", "laterals", "pivots", "into", "locks", "sample")
_FLIP = {"=": "<>", "<>": "=", "<": ">=", ">=": "<", ">": "<=", "<=": ">"}
_MIRROR = {"=": "=", "<>": "<>", "<": ">", ">": "<", "<=": ">=", ">=": "<="}
_OPS = {exp.EQ: "=", exp.NEQ: "<>", exp.LT: "<", exp.LTE: "<=", exp.GT: ">", exp.GTE: ">="}


def _unparen(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _name(column: exp.Column) -> str:
    """The column's name: lower case unless it was quoted (a quoted name can differ from another by case alone)."""

    return column.name if column.this.args.get("quoted") else column.name.lower()


class _Outer:
    """The select around the tests: how an expression of it is written so that two spellings of one read agree."""

    def __init__(self, select: exp.Select, schema: dict | None):
        from_ = select.args.get("from_") or select.args.get("from")
        sources = ([from_.this] if from_ is not None else []) + [j.this for j in select.args.get("joins") or []]
        self.names = [s.alias_or_name.lower() for s in sources]
        self.single: tuple[str, set[str]] | None = None
        if len(sources) == 1 and isinstance(sources[0], exp.Table) and isinstance(sources[0].this, exp.Identifier) and not sources[0].args.get("db"):
            for name, columns in (schema or {}).items():
                if str(name).lower().split(".")[-1] == sources[0].name.lower():
                    self.single = (sources[0].alias_or_name, {str(c).lower() for c in columns})

    def text(self, node: exp.Expression) -> str | None:
        """``node`` with unquoted names in lower case, literals as written and every column naming its source.

        A bare column is given the select's only source when that source is a table known to have it; otherwise
        None (the owner is not certain).
        """

        copy = exp.Paren(this=node.copy())
        for column in list(copy.find_all(exp.Column)):
            if not isinstance(column.this, exp.Identifier) or column.args.get("db"):
                return None
            if not column.table:
                if self.single is None or column.name.lower() not in self.single[1]:
                    return None
                column.set("table", exp.to_identifier(self.single[0]))
        for identifier in copy.find_all(exp.Identifier):
            if not identifier.args.get("quoted"):
                identifier.set("this", identifier.name.lower())
        return copy.this.sql(dialect="mysql")


class _Probe:
    """One subquery over a single base table: its own columns, ties to the outside and other conjuncts."""

    def __init__(self, query: exp.Select, schema: dict | None, outer: _Outer, nested: bool = False):
        from .algebraic_equivalence import _conjuncts

        self.ok = False
        self.query = query
        self.outer = outer
        source = (query.args.get("from_") or query.args.get("from"))
        source = source.this if source is not None else None
        if any(query.args.get(k) for k in _BANNED) or not isinstance(source, exp.Table) or not isinstance(source.this, exp.Identifier):
            return
        if any(source.args.get(k) for k in ("db", "catalog", "pivots", "sample", "joins", "version", "when")):
            return
        distinct = query.args.get("distinct")
        if distinct is not None and distinct.args.get("on"):
            return
        if any(isinstance(n, (exp.AggFunc, exp.Window)) and n.find_ancestor(exp.Select) is query for n in query.walk()):
            return  # an aggregate makes one row of none
        if not nested and any(isinstance(n, exp.Query) and n is not query for n in query.walk()):
            return  # a nested query is not read
        self.table = source.name.lower()
        self.alias = source.alias_or_name.lower()
        listed = None
        for name, columns in (schema or {}).items():
            if str(name).lower().split(".")[-1] == self.table:
                listed = {str(c).lower() for c in columns}
        self.listed = listed
        where = query.args.get("where")
        self.conjuncts = _conjuncts(where.this) if where is not None else []
        self.ok = True

    def own(self, column: exp.Column) -> bool | None:
        """Whether ``column`` is a column of the probe's table; None when that is not certain."""

        if not isinstance(column.this, exp.Identifier) or column.args.get("db"):
            return None
        if column.table:
            return column.table.lower() == self.alias
        if self.listed is None:
            return None
        return column.name.lower() in self.listed

    def reads(self, node: exp.Expression) -> bool | None:
        """Whether ``node`` reads a column of the probe's table; None when a column's owner is not certain."""

        found = False
        for column in node.find_all(exp.Column):
            owned = self.own(column)
            if owned is None:
                return None
            found = found or owned
        return found

    def tie(self, condition: exp.Expression) -> tuple[str, str] | None:
        """``(own column, the outside expression's text)`` for ``a.c = e``."""

        condition = _unparen(condition)
        if not isinstance(condition, exp.EQ):
            return None
        for mine, other in ((condition.this, condition.expression), (condition.expression, condition.this)):
            mine, other = _unparen(mine), _unparen(other)
            if isinstance(mine, exp.Column) and self.own(mine) is True and self.reads(other) is False and other.find(exp.Column) is not None:
                text = self.outer.text(other)
                if text is not None:
                    return _name(mine), text
        return None


def _normal(node: exp.Expression, probe: _Probe, negate: bool, not_null: frozenset | None):
    """``node`` (or its negation) as nested ``("and" | "or", frozenset)`` over ``("atom", column, op, literal)``.

    None when it is not built from comparisons of the probe's own columns with literals. With ``not_null`` given,
    every column must be in it (the condition is then never NULL).
    """

    node = _unparen(node)
    if isinstance(node, exp.Not):
        return _normal(node.this, probe, not negate, not_null)
    if isinstance(node, (exp.And, exp.Or)):
        parts = [_normal(node.this, probe, negate, not_null), _normal(node.expression, probe, negate, not_null)]
        if None in parts:
            return None
        return _join("and" if isinstance(node, exp.And) != negate else "or", parts)

    def atom(column: exp.Expression, op: str, literal: exp.Expression):
        column, literal = _unparen(column), _unparen(literal)
        if isinstance(literal, exp.Neg) and isinstance(literal.this, exp.Literal) and not literal.this.is_string:
            text = "-" + literal.this.sql()
        elif isinstance(literal, exp.Literal):
            text = literal.sql()
        else:
            return None
        if not isinstance(column, exp.Column) or probe.own(column) is not True:
            return None
        if not_null is not None and column.name.lower() not in not_null:
            return None
        return ("atom", _name(column), _FLIP[op] if negate else op, text)

    if type(node) in _OPS:
        op = _OPS[type(node)]
        if isinstance(_unparen(node.this), exp.Column):
            return atom(node.this, op, node.expression)
        return atom(node.expression, _MIRROR[op], node.this)
    if isinstance(node, exp.Between) and not node.args.get("symmetric"):
        parts = [atom(node.this, ">=", node.args["low"]), atom(node.this, "<=", node.args["high"])]
        if None in parts:
            return None
        return _join("or" if negate else "and", parts)
    if isinstance(node, exp.In) and node.expressions and not any(node.args.get(k) for k in ("query", "unnest", "field")):
        parts = [atom(node.this, "=", item) for item in node.expressions]
        if None in parts:
            return None
        return _join("and" if negate else "or", parts)
    return None


def _join(kind: str, parts: list):
    flat: set = set()
    for part in parts:
        flat |= part[1] if part[0] == kind else {part}
    return next(iter(flat)) if len(flat) == 1 else (kind, frozenset(flat))


def _parts(normal) -> frozenset:
    return normal[1] if normal[0] == "and" else frozenset({normal})


def _absence(test: exp.Expression, schema: dict | None, not_null: dict, outer: _Outer) -> tuple[frozenset, frozenset, str] | None:
    """``(ties, what every tied row satisfies, table)`` for a ``NOT EXISTS`` of the shape in the module docstring."""

    test = _unparen(test)
    if not isinstance(test, exp.Not) or not isinstance(_unparen(test.this), exp.Exists):
        return None
    inner = _unparen(test.this).this
    inner = inner.this if isinstance(inner, exp.Subquery) else inner
    if not isinstance(inner, exp.Select):
        return None
    probe = _Probe(inner, schema, outer)
    if not probe.ok:
        return None
    declared = frozenset()
    for name, columns in not_null.items():
        if str(name).lower().split(".")[-1] == probe.table:
            declared = frozenset(str(c).lower() for c in columns)
    ties, satisfied = set(), set()
    for condition in probe.conjuncts:
        tie = probe.tie(condition)
        if tie is not None:
            ties.add(tie)
            continue
        if probe.reads(condition) is not True:
            return None  # a condition on the outer row alone, or a column whose owner is not certain
        negated = _normal(condition, probe, True, declared)
        if negated is None or satisfied:
            return None  # one condition only: of two, a tied row need only fail one
        satisfied = set(_parts(negated))
    if not ties or not satisfied:
        return None
    return frozenset(ties), frozenset(satisfied), probe.table


def _shortened(conditions: list[exp.Expression], absences: list, outer: _Outer, schema: dict | None) -> list[exp.Expression] | None:
    """``conditions`` with each presence test shortened by what ``absences`` give its rows; None when nothing changes."""

    changed = False
    rebuilt = []
    for condition in conditions:
        rebuilt.append(condition)
        test = _unparen(condition)
        extra: tuple[str, str] | None = None
        if isinstance(test, exp.Exists):
            inner = test.this
        elif isinstance(test, exp.In) and test.args.get("query") is not None and not test.expressions:
            inner = test.args["query"]
        else:
            continue
        inner = inner.this if isinstance(inner, exp.Subquery) else inner
        if not isinstance(inner, exp.Select):
            continue
        probe = _Probe(inner, schema, outer)
        if not probe.ok:
            continue
        if isinstance(test, exp.In):
            item = _unparen(inner.expressions[0].unalias()) if len(inner.expressions) == 1 else None
            tested = _unparen(test.this)
            text = outer.text(tested)  # the tested expression stands outside the subquery: its names are the outer select's
            if not isinstance(item, exp.Column) or probe.own(item) is not True or text is None or tested.find(exp.Column) is None:
                continue
            extra = (_name(item), text)
        ties = {tie for tie in (probe.tie(c) for c in probe.conjuncts) if tie is not None} | ({extra} if extra else set())
        given: set = set()
        for needed, satisfied, table in absences:
            if table == probe.table and needed <= ties:
                given |= satisfied
        if not given:
            continue
        kept = []
        for part in probe.conjuncts:
            normal = None if probe.tie(part) is not None else _normal(part, probe, False, None)
            if normal is not None and _parts(normal) <= given:
                continue
            kept.append(part)
        if len(kept) == len(probe.conjuncts):
            continue
        replacement = inner.copy()
        replacement.set("where", exp.Where(this=exp.and_(*[k.copy() for k in kept], copy=False)) if kept else None)
        new_test = test.copy()
        key = "this" if isinstance(new_test, exp.Exists) else "query"
        new_test.set(key, exp.Subquery(this=replacement) if isinstance(new_test.args[key], exp.Subquery) else replacement)
        rebuilt[-1] = new_test
        changed = True
    return rebuilt if changed else None


def _filtering_source(select: exp.Select, schema: dict | None) -> tuple[exp.Select, str, dict[str, exp.Column]] | None:
    """``(inner select, its name, its outputs)`` when ``select`` reads one derived table that only filters one table.

    The derived table lists plain columns of its one base table (no star, no repeated name) and has no join, grouping,
    aggregate, window, ``DISTINCT`` or ``LIMIT``; its outputs map each name to the column of the table behind it.
    """

    from_ = select.args.get("from_") or select.args.get("from")
    source = from_.this if from_ is not None else None
    if select.args.get("joins") or select.args.get("laterals") or not isinstance(source, exp.Subquery) or not source.alias or not isinstance(source.this, exp.Select):
        return None
    alias = source.args.get("alias")
    if not isinstance(alias, exp.TableAlias) or alias.args.get("columns") or any(source.args.get(k) for k in ("pivots", "sample", "joins")):
        return None
    inner = source.this
    probe = _Probe(inner, schema, _Outer(select, schema), nested=True)
    if not probe.ok or inner.args.get("distinct") or inner.args.get("order"):
        return None
    outputs: dict[str, exp.Column] = {}
    for item in inner.expressions:
        value = _unparen(item.unalias())
        if not isinstance(value, exp.Column) or probe.own(value) is not True or not item.alias_or_name or item.alias_or_name.lower() in outputs:
            return None
        outputs[item.alias_or_name.lower()] = exp.column(value.name, table=inner.args["from_" if inner.args.get("from_") else "from"].this.alias_or_name)
    return inner, source.alias.lower(), outputs


def _through_filtering_source(select: exp.Select, conditions: list[exp.Expression], not_null: dict, schema: dict | None) -> exp.Expression | None:
    """The same for an absence test asked of a derived table that only filters: the presence test is inside it.

    ``SELECT c.id FROM (SELECT p.id FROM product AS p WHERE EXISTS (..)) AS c WHERE NOT EXISTS (.. t.pid = c.id ..)``
    keeps the rows of ``product`` that pass both tests, and ``c.id`` is ``p.id`` of the row behind it. A row the
    shorter presence test lets in fails the absence test, and nothing but the ``WHERE`` reads the derived table.
    """

    found = _filtering_source(select, schema)
    if found is None:
        return None
    inner, name, outputs = found
    behind = inner.args.get("from_") or inner.args.get("from")
    inside = _Outer(inner, schema)
    absences = []
    for condition in conditions:
        moved = condition.copy()
        if any(not column.table for column in moved.find_all(exp.Column)):
            continue  # a bare name could be the derived table's or the tested table's
        probes = [s.alias_or_name.lower() for query in moved.find_all(exp.Select) for s in [(query.args.get("from_") or query.args.get("from"))] if s is not None for s in [s.this]]
        if name in probes or behind.this.alias_or_name.lower() in probes:
            continue  # the test's own table would be read for the row behind the derived table
        holder = exp.Paren(this=moved)
        for column in list(holder.find_all(exp.Column)):
            if column.table.lower() == name:
                target = outputs.get(column.name.lower())
                if target is None:
                    holder = None
                    break
                column.replace(target.copy())
        if holder is None:
            continue
        absence = _absence(holder.this, schema, not_null, inside)
        if absence is not None:
            absences.append(absence)
    where = inner.args.get("where")
    if not absences or where is None:
        return None
    from .algebraic_equivalence import _conjuncts

    shortened = _shortened(_conjuncts(where.this), absences, inside, schema)
    if shortened is None:
        return None
    result = select.copy()
    target = (result.args.get("from_") or result.args.get("from")).this.this
    target.set("where", exp.Where(this=exp.and_(*[c.copy() for c in shortened], copy=False)))
    return result


def drop_implied_presence_filters(select: exp.Select, not_null: dict | None, schema: dict | None = None) -> exp.Expression | None:
    """Drop from a presence test the conditions an absence test of the same ``WHERE`` gives every row it can count."""

    from .algebraic_equivalence import _conjuncts
    from .counted_membership import _unstable

    where = select.args.get("where")
    if where is None or not not_null or _unstable(select):
        return None
    conditions = _conjuncts(where.this)
    outer = _Outer(select, schema)
    if len(set(outer.names)) != len(outer.names):
        return None
    absences = [found for found in (_absence(c, schema, not_null, outer) for c in conditions) if found is not None]
    shortened = _shortened(conditions, absences, outer, schema) if absences else None
    if shortened is None:
        return _through_filtering_source(select, conditions, not_null, schema)
    result = select.copy()
    result.set("where", exp.Where(this=exp.and_(*[c.copy() for c in shortened], copy=False)))
    return result
