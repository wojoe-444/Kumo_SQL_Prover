"""A join that only asks whether a matching row exists, read as ``EXISTS`` or ``NOT EXISTS``.

Three spellings of an existence test that planners and people write as joins:

**The anti-join.** ``a LEFT JOIN d ON a.x = d.k WHERE d.k IS NULL``

    SELECT a.g, COUNT(*) FROM a LEFT JOIN d ON a.x = d.k WHERE d.k IS NULL GROUP BY a.g
    SELECT a.g, COUNT(*) FROM a WHERE NOT EXISTS (SELECT 1 FROM d WHERE a.x = d.k) GROUP BY a.g

A row of ``a`` that matches rows of ``d`` is joined to each of them, and in every such row ``d.k`` is not NULL,
because ``a.x = d.k`` was true. The ``WHERE`` therefore drops all of them, however many there were. A row of
``a`` with no match appears once, with NULL in every column of ``d``, and passes. So the join returns exactly
the rows of ``a`` for which no row of ``d`` satisfies the ``ON`` condition, once each: the ``NOT EXISTS``.
The tested column must be one that cannot be NULL in a matched row: a column the ``ON`` condition compares
(a top-level ``=``, ``<>``, ``<``, ``<=``, ``>``, ``>=`` with that column alone on one side), or an output of
the derived table ``d`` that is a constant (``TRUE AS f``, the marker Calcite adds).

**The semi-join.** ``a JOIN (SELECT DISTINCT k FROM t ..) AS d ON a.x = d.k``

    SELECT a.g FROM a JOIN (SELECT k, TRUE AS f FROM t GROUP BY k) AS d ON a.x = d.k
    SELECT a.g FROM a WHERE EXISTS (SELECT 1 FROM (SELECT k, TRUE AS f FROM t GROUP BY k) AS d WHERE a.x = d.k)

The rows of ``d`` differ in their non-constant outputs (``DISTINCT``, or a ``GROUP BY`` whose every key is an
output and whose every output is a key or a constant). When the ``ON`` condition equates every one of those
outputs with a value from the other side, at most one row of ``d`` can match a given row, so the join keeps or
drops rows of ``a`` and never repeats one.

**The marker join.** ``a LEFT JOIN (SELECT k, TRUE AS f FROM t GROUP BY k) AS d ON a.x = d.k WHERE d.f IS NOT NULL OR p``

With ``d`` duplicate-free in the same sense and every distinguishing output equated, each row of ``a`` appears
exactly once, with its one match or with NULLs. A column of ``d`` that cannot be NULL in a matched row is then
NULL exactly when there is no match, so ``d.f IS NOT NULL`` is the ``EXISTS`` test and ``d.f IS NULL`` its
negation, wherever in the select they stand.

All three are read only when:

* the join is a plain ``LEFT`` join (anti, marker) or a plain inner join with ``ON`` (semi), no ``USING`` or
  ``NATURAL`` anywhere in the select, no lateral source, and no ``RIGHT`` or ``FULL`` join after it (which
  would pad the rows this join produced);
* ``d`` is a table or a derived table with an alias no other source of the select has, and no subquery of the
  select names a source the same;
* the ``ON`` condition names the source of every column, and holds no subquery, aggregate, window,
  ``RANDOM()`` or function name the parser does not know;
* apart from the ``ON`` condition, the select reads ``d`` only in the top-level ``d.col IS NULL`` tests of its
  ``WHERE`` (anti; once one of them shows that no matched row survives, the others are true of every row
  left), only in ``IS [NOT] NULL`` tests of never-NULL columns in the select itself (marker), or not at all
  (semi). A bare column that could be one of ``d``'s, or a star, stops the rule.

A derived table that returns a row for no input (an aggregate without ``GROUP BY``) is not read as
duplicate-free by this module: it has no ``DISTINCT`` or ``GROUP BY`` of plain keys.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import extended_grouping, select_sources

_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
_BODY_ALLOWED = frozenset({"expressions", "from_", "from", "joins", "where", "group", "distinct", "kind", "hint"})


def _unparen(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _conjuncts(node: exp.Expression) -> list[exp.Expression]:
    node = _unparen(node)
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _constant(node: exp.Expression) -> bool:
    """A literal that is never NULL."""

    node = _unparen(node)
    return isinstance(node, (exp.Literal, exp.Boolean))


def _item_value(item: exp.Expression) -> exp.Expression:
    return _unparen(item.this if isinstance(item, exp.Alias) else item)


def _output_names(source: exp.Expression, schema: dict[str, list[str]]) -> list[str] | None:
    """The column names of a source, lower-cased; ``None`` when they are not known."""

    if isinstance(source, exp.Table):
        parts = [p.name for p in (source.args.get("catalog"), source.args.get("db"), source.this) if p is not None]
        columns = schema.get(".".join(parts).lower())
        return None if columns is None else list(columns)
    body = source.this
    while isinstance(body, exp.SetOperation):
        body = body.this
    while isinstance(body, exp.Subquery):
        body = body.this
    if not isinstance(body, exp.Select):
        return None
    names = []
    for item in body.expressions:
        name = (item.alias_or_name or "").lower()
        if not name or isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
            return None
        names.append(name)
    return names


def _constant_outputs(source: exp.Expression) -> set[str]:
    """Names of a derived table's outputs that are a literal in every row."""

    if not isinstance(source, exp.Subquery) or not isinstance(source.this, exp.Select):
        return set()
    names = [(item.alias_or_name or "").lower() for item in source.this.expressions]
    return {name for name, item in zip(names, source.this.expressions) if name and names.count(name) == 1 and _constant(_item_value(item))}


def _distinguishing_outputs(source: exp.Expression) -> set[str] | None:
    """The outputs on which no two rows of the derived table agree, or ``None`` when rows may repeat."""

    if not isinstance(source, exp.Subquery) or not isinstance(source.this, exp.Select):
        return None
    inner = source.this
    if any(value for key, value in inner.args.items() if key not in _BODY_ALLOWED and value):
        return None
    distinct = inner.args.get("distinct")
    if distinct is not None and (not isinstance(distinct, exp.Distinct) or distinct.args.get("on")):
        return None
    group = inner.args.get("group")
    if group is None and distinct is None:
        return None
    for node in inner.walk():
        if isinstance(node, (exp.AggFunc, exp.Window, exp.Anonymous, exp.WithinGroup)) and node.find_ancestor(exp.Select) is inner:
            return None  # an aggregate makes the row count its own matter; an unknown name may be one
    names = [(item.alias_or_name or "").lower() for item in inner.expressions]
    if "" in names or len(set(names)) != len(names):
        return None
    if any(isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)) for item in inner.expressions):
        return None
    values = [_item_value(item) for item in inner.expressions]
    if any(isinstance(n, (exp.Subquery, exp.Exists)) for value in values for n in value.walk()):
        return None
    if group is not None:
        if extended_grouping(group) or not group.expressions:
            return None
        keys = [_unparen(key) for key in group.expressions]
        if any(isinstance(key, (exp.Literal, exp.Boolean, exp.Subquery)) for key in keys):
            return None  # GROUP BY 1 is an ordinal
        key_texts = {key.sql() for key in keys}
        written = {value.sql() for value in values}
        if not key_texts <= written:
            return None  # a key that is not returned lets the returned rows repeat
        if any(not _constant(value) and value.sql() not in key_texts for value in values):
            return None
    return {name for name, value in zip(names, values) if not _constant(value)}


def _own_star(select: exp.Select) -> bool:
    return any(
        isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)) for item in select.expressions
    )


def _inside(node: exp.Expression, top: exp.Expression, ids: set[int]) -> bool:
    node = node.parent
    while node is not None and node is not top:
        if id(node) in ids:
            return True
        node = node.parent
    return False


def _within(node: exp.Expression, ancestor: exp.Expression) -> bool:
    while node is not None:
        if node is ancestor:
            return True
        node = node.parent
    return False


def read_existence_joins(select: exp.Select, schema: dict[str, list[str]] | None = None) -> exp.Expression | None:
    """Rewrite one anti-join or semi-join of ``select`` as a ``NOT EXISTS`` or ``EXISTS`` test, in place."""

    joins = select.args.get("joins") or []
    if not joins or _own_star(select) or select.args.get("laterals"):
        return None
    if any(j.args.get("using") or j.args.get("method") for j in joins):
        return None
    sources = select_sources(select)
    if any(not isinstance(s, (exp.Table, exp.Subquery)) for s in sources):
        return None  # LATERAL, UNNEST and table functions read their siblings
    lowered = {key.lower(): [c.lower() for c in columns] for key, columns in (schema or {}).items()}
    source_ids = {id(s) for s in sources}
    aliases = [(s.alias_or_name or "").lower() for s in sources]
    outside = [n for n in select.walk() if isinstance(n, (exp.Column, exp.Table, exp.Subquery)) and id(n) not in source_ids and not _inside(n, select, source_ids)]
    columns = [n for n in outside if isinstance(n, exp.Column) and not isinstance(n.this, exp.Star)]
    if any(isinstance(n, exp.Column) and isinstance(n.this, exp.Star) for n in outside):
        return None
    nested_names = {(n.alias_or_name or "").lower() for n in outside if isinstance(n, (exp.Table, exp.Subquery))}
    bare = {c.name.lower() for c in columns if not c.table}
    where = select.args.get("where")
    for index, join in enumerate(joins):
        source, on = join.this, join.args.get("on")
        side, kind = (join.args.get("side") or "").upper(), (join.args.get("kind") or "").upper()
        anti = side == "LEFT" and kind in ("", "OUTER")
        semi = not side and kind in ("", "INNER")
        if on is None or not (anti or semi):
            continue
        if any((later.args.get("side") or "").upper() in ("RIGHT", "FULL") for later in joins[index + 1 :]):
            continue
        alias = (source.alias_or_name or "").lower()
        if not alias or aliases.count(alias) != 1 or alias in nested_names:
            continue
        alias_node = source.args.get("alias")
        if alias_node is not None and alias_node.args.get("columns"):
            continue
        extra = ("this", "alias") if isinstance(source, exp.Subquery) else ("this", "db", "catalog", "alias")
        if any(value for key, value in source.args.items() if key not in extra and value):
            continue
        if any(isinstance(n, (exp.Subquery, exp.Exists, exp.AggFunc, exp.Window, exp.Select, exp.Rand, exp.Anonymous)) for n in on.walk()):
            continue  # a call that need not return the same value twice is not one test per pair of rows
        if any(not c.table for c in on.find_all(exp.Column)):
            continue
        names = _output_names(source, lowered)
        if bare and (names is None or bare & set(names)):
            continue  # a bare column that may be one of this source's
        parts = _conjuncts(on)
        reads = [c for c in columns if c.table.lower() == alias and not _within(c, join)]
        if anti:
            never_null = _constant_outputs(source)
            for part in parts:
                if isinstance(part, _COMPARISONS):
                    for operand in (_unparen(part.this), _unparen(part.expression)):
                        if isinstance(operand, exp.Column) and operand.table.lower() == alias:
                            never_null.add(operand.name.lower())
            tests = []
            for part in _conjuncts(where.this) if where is not None else []:
                tested = _unparen(part.this) if _is_null_test(part) and not part.args.get("negate") else None
                if isinstance(tested, exp.Column) and tested.table.lower() == alias and tested.find_ancestor(exp.Select) is select:
                    tests.append((part, tested))
            if tests and {id(c) for c in reads} == {id(tested) for _, tested in tests} and any(tested.name.lower() in never_null for _, tested in tests):
                kept = [part for part in _top_parts(where.this) if not any(_unparen(part) is test for test, _ in tests)]
                probe = exp.Not(this=_probe(source, on))
            else:
                # a marker join: at most one row of the source matches, and the select only asks whether one did
                needed = _distinguishing_outputs(source)
                if needed is None or not reads or not needed <= _equated(parts, alias):
                    continue
                marks = []
                for column in reads:
                    test = column.parent
                    while isinstance(test, exp.Paren):
                        test = test.parent
                    if not _is_null_test(test) or _unparen(test.this) is not column:
                        break
                    if column.name.lower() not in never_null or test.find_ancestor(exp.Select) is not select:
                        break
                    marks.append(test)
                if len(marks) != len(reads):
                    continue
                for test in marks:
                    found = bool(test.args.get("negate"))  # IS NOT NULL: a row matched
                    outer = test.parent
                    while isinstance(outer, exp.Paren):
                        outer = outer.parent
                    if isinstance(outer, exp.Not):
                        test, found = outer, not found
                    test.replace(_probe(source, on) if found else exp.Not(this=_probe(source, on)))
                select.set("joins", [j for j in joins if j is not join] or None)
                return select
        else:
            if reads:
                continue
            needed = _distinguishing_outputs(source)
            if needed is None:
                continue
            equated = _equated(parts, alias)
            if not needed <= equated:
                continue
            kept = _top_parts(where.this) if where is not None else []
            probe = _probe(source, on)
        condition = probe
        for part in reversed(kept):
            member = part.copy()
            if isinstance(member, (exp.Or, exp.Xor)):
                member = exp.Paren(this=member)
            condition = exp.And(this=member, expression=condition)
        select.set("where", exp.Where(this=condition))
        select.set("joins", [j for j in joins if j is not join] or None)
        return select
    return None


def _is_null_test(node: exp.Expression) -> bool:
    """``x IS NULL`` or ``x IS NOT NULL`` (the parser marks the second with ``negate``)."""

    return isinstance(node, exp.Is) and isinstance(node.expression, exp.Null)


def _equated(parts: list[exp.Expression], alias: str) -> set[str]:
    """The columns of ``alias`` that a top-level ``=`` of the ``ON`` condition equates with a value of the other side."""

    equated = set()
    for part in parts:
        if not isinstance(part, exp.EQ):
            continue
        left, right = _unparen(part.this), _unparen(part.expression)
        for mine, other in ((left, right), (right, left)):
            if isinstance(mine, exp.Column) and mine.table.lower() == alias and not any(c.table.lower() == alias for c in other.find_all(exp.Column)):
                equated.add(mine.name.lower())
    return equated


def _top_parts(node: exp.Expression) -> list[exp.Expression]:
    """The members of an ``AND`` chain as written (parentheses around a member kept)."""

    inner = _unparen(node)
    if isinstance(inner, exp.And):
        return _top_parts(inner.this) + _top_parts(inner.expression)
    return [node]


def _probe(source: exp.Expression, on: exp.Expression) -> exp.Exists:
    body = exp.Select(expressions=[exp.Literal.number(1)]).from_(source.copy())
    body.set("where", exp.Where(this=on.copy()))
    return exp.Exists(this=body)


def read_existence_joins_everywhere(tree: exp.Expression, schema: dict[str, list[str]] | None = None) -> exp.Expression:
    """``read_existence_joins`` on every select of ``tree``, innermost first, until none is left to read."""

    for select in list(tree.find_all(exp.Select))[::-1]:
        for _ in range(len(select.args.get("joins") or [])):
            if read_existence_joins(select, schema) is None:
                break
    return tree
