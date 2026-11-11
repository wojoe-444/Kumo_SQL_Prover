"""A grouped table restricted to the keys of the relation it is joined to on that key needs no restriction.

    FROM (B) AS tp LEFT JOIN (SELECT v.post, COUNT(*) AS n FROM v WHERE v.post IN (SELECT id FROM (B) AS x) GROUP BY v.post) AS d
         ON tp.id = d.post
    is
    FROM (B) AS tp LEFT JOIN (SELECT v.post, COUNT(*) AS n FROM v GROUP BY v.post) AS d ON tp.id = d.post

A rewriter adds the ``IN`` so that the grouped table is computed for the wanted keys only (a "magic set"). ``B`` is
usually a common table expression named in both places, which the prover has inlined by the time this runs.

Why it holds. Write ``K`` for the values of ``id`` in ``B``.

* The ``IN`` tests a ``GROUP BY`` key of the grouped table, so it keeps or removes whole groups: a group whose key
  is in ``K`` keeps every row and so every aggregate value, a group whose key is not in ``K`` (or is NULL, for which
  ``IN`` is never true) disappears. A NULL among ``K`` changes nothing: ``IN`` is then NULL instead of false for a
  key outside ``K``, and ``WHERE`` drops the row either way.
* The join condition is a conjunction that contains ``tp.id = d.post``. A row of ``d`` joins a row of ``tp`` only
  when its key equals that row's ``id``, a value of ``K``. So the groups the ``IN`` removes are exactly groups that
  join no row, and an inner or left join does not keep an unmatched row of ``d``.

The joined rows are therefore the same with and without the ``IN``, duplicates and NULL padding included.

``B`` is read twice, so the argument needs the two readings to give the same rows. That is so for a deterministic
``B``. When ``B`` computes a window function it is assumed
(``smt_equivalence.WINDOW_SOURCE_ASSUMPTION``, recorded on the proof): the prover makes the same assumption
whenever it compares two window computations by their text.

Dropped only when (the query is left as written otherwise):

* the grouped table is an aliased derived table without a column list, joined with ``JOIN``/``INNER JOIN`` or
  ``LEFT JOIN`` and an ``ON`` (a right or full join keeps unmatched groups; no ``USING``, ``NATURAL`` or lateral);
* its body is a select with a plain ``GROUP BY`` (no ``ROLLUP``, ``CUBE`` or ``GROUPING SETS``), with no
  ``DISTINCT``, window, ``ORDER BY``, ``LIMIT`` or ``QUALIFY`` of its own;
* the ``IN`` is a conjunct of the body's ``WHERE`` (not negated), its left side is a column that is a ``GROUP BY``
  key of the body and an output of it under some name ``k``, and its subquery is ``SELECT [DISTINCT] c FROM (B)
  [AS x]`` with nothing else, where ``c`` is a column and an output name of ``B``;
* a conjunct of the join's ``ON`` is ``p.c = d.k`` (either way round), where ``p`` is another source of the same
  select, an aliased derived table without a column list whose body has the same text as ``B``, and ``d`` the
  grouped table's alias, which no other source of the select has;
* ``B`` has no function whose value changes from call to call, no ``LIMIT``, ``OFFSET`` or ``FETCH``, and no ``*``
  in its select list;
* the select is the query itself or a derived table under derived tables only. Inside a scalar, ``EXISTS`` or ``IN``
  subquery a column of ``B`` could name a row of the enclosing query, and the same text read in the grouped table's
  ``WHERE`` could then name another row.
"""

from __future__ import annotations

from sqlglot import exp

_UNSTABLE = (exp.Rand, exp.Randn, exp.Uuid, exp.CurrentTimestamp, exp.CurrentDate, exp.CurrentTime, exp.CurrentDatetime, exp.Anonymous, exp.Limit, exp.Offset, exp.Fetch)
_BODY_ARGS = frozenset({"expressions", "from_", "from", "joins", "where", "group", "having", "kind"})
_PROBE_ARGS = frozenset({"expressions", "from_", "from", "distinct", "kind"})


def _from(select: exp.Select) -> exp.From | None:
    return select.args.get("from_") or select.args.get("from")


def _conjuncts(node: exp.Expression) -> list[exp.Expression]:
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _derived(source: exp.Expression) -> exp.Select | None:
    """The body of an aliased derived table without a column list or a tail of its own."""

    if not isinstance(source, exp.Subquery) or not source.alias or source.args["alias"].args.get("columns"):
        return None
    if any(value for key, value in source.args.items() if key not in ("this", "alias")):
        return None
    return source.this if isinstance(source.this, exp.Select) else None


def _output_names(select: exp.Select) -> list[str] | None:
    names = []
    for item in select.expressions:
        if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)) or not item.alias_or_name:
            return None
        names.append(item.alias_or_name.lower())
    return names


def _probe(condition: exp.Expression):
    """``(key column, c, B)`` of ``key IN (SELECT [DISTINCT] c FROM (B) [AS x])``."""

    if type(condition) is not exp.In or not isinstance(condition.this, exp.Column) or any(value for key, value in condition.args.items() if key not in ("this", "query")):
        return None
    query = condition.args.get("query")
    if not isinstance(query, exp.Subquery) or any(value for key, value in query.args.items() if key != "this"):
        return None
    probe = query.this
    if not isinstance(probe, exp.Select) or any(value for key, value in probe.args.items() if key not in _PROBE_ARGS) or len(probe.expressions) != 1:
        return None
    distinct = probe.args.get("distinct")
    if distinct is not None and distinct.args.get("on") is not None:
        return None
    source = _from(probe).this if _from(probe) is not None else None
    if not isinstance(source, exp.Subquery) or not isinstance(source.this, exp.Select) or any(value for key, value in source.args.items() if key not in ("this", "alias")):
        return None
    alias = source.args.get("alias")
    if alias is not None and alias.args.get("columns"):
        return None
    item = probe.expressions[0]
    item = item.this if isinstance(item, exp.Alias) else item
    if not isinstance(item, exp.Column) or isinstance(item.this, exp.Star) or (item.table and item.table.lower() != source.alias.lower()):
        return None
    body = source.this
    names = _output_names(body)
    if names is None or names.count(item.name.lower()) != 1 or any(isinstance(node, _UNSTABLE) for node in body.walk()):
        return None
    return condition.this, item.name.lower(), body


def _sees_no_outer_row(select: exp.Select) -> bool:
    """Whether ``select`` is the query or a derived table under derived tables only, so that no enclosing row is in scope."""

    node = select
    while node.parent is not None:
        derived = node.parent
        if not isinstance(derived, exp.Subquery) or any(value for key, value in derived.args.items() if key not in ("this", "alias")):
            return False
        holder = derived.parent
        if not isinstance(holder, (exp.From, exp.Join)) or not isinstance(holder.parent, exp.Select):
            return False
        node = holder.parent
    return True


def _redundant(select: exp.Select) -> tuple[exp.Expression, exp.Select] | None:
    """A redundant ``IN`` conjunct of a grouped table joined in ``select``, with the ``B`` it reads."""

    source = _from(select)
    joins = select.args.get("joins") or []
    if source is None or not joins or not _sees_no_outer_row(select):
        return None
    items = [source.this] + [join.this for join in joins]
    aliases = [item.alias_or_name.lower() for item in items if isinstance(item, (exp.Table, exp.Subquery))]
    for join in joins:
        body = _derived(join.this)
        if body is None or join.args.get("on") is None or join.args.get("using") or join.args.get("method"):
            continue
        if (join.side or "").upper() not in ("", "LEFT") or (join.kind or "").upper() not in ("", "INNER", "OUTER"):
            continue
        name = join.this.alias.lower()
        if aliases.count(name) != 1 or any(value for key, value in body.args.items() if key not in _BODY_ARGS):
            continue
        group, where = body.args.get("group"), body.args.get("where")
        if group is None or not group.expressions or any(value for key, value in group.args.items() if key != "expressions") or where is None:
            continue
        if any(window.find_ancestor(exp.Select) is body for window in body.find_all(exp.Window)):
            continue
        if any(isinstance(node, (exp.Rollup, exp.Cube, exp.GroupingSets, exp.Tuple)) for node in group.walk()):
            continue  # sqlglot keeps these among the grouping expressions
        for condition in _conjuncts(where.this):
            probe = _probe(condition)
            if probe is None:
                continue
            key, wanted, relation = probe
            if not any(isinstance(g, exp.Column) and g.name.lower() == key.name.lower() and g.table.lower() == key.table.lower() for g in group.expressions):
                continue
            outputs = [
                item.alias_or_name.lower()
                for item in body.expressions
                if isinstance(item.unalias(), exp.Column) and item.unalias().name.lower() == key.name.lower() and item.unalias().table.lower() == key.table.lower()
            ]
            names = _output_names(body)
            if names is None or not outputs:
                continue
            outputs = [output for output in outputs if names.count(output) == 1]
            text = relation.sql()
            for equality in _conjuncts(join.args["on"]):
                if type(equality) is not exp.EQ or not isinstance(equality.this, exp.Column) or not isinstance(equality.expression, exp.Column):
                    continue
                for mine, other in ((equality.this, equality.expression), (equality.expression, equality.this)):
                    if mine.table.lower() != name or mine.name.lower() not in outputs or other.name.lower() != wanted or other.table.lower() == name:
                        continue
                    for item in items:
                        if item is join.this or not isinstance(item, exp.Subquery) or item.alias.lower() != other.table.lower() or aliases.count(item.alias.lower()) != 1:
                            continue
                        partner = _derived(item)
                        if partner is not None and partner.sql() == text:
                            return condition, relation
    return None


def drop_joined_key_restrictions(tree: exp.Expression, assumptions: set[str] | None = None) -> exp.Expression:
    """``tree`` without the ``IN`` restrictions the module docstring shows redundant."""

    if not any(isinstance(node.args.get("query"), exp.Subquery) for node in tree.find_all(exp.In)):
        return tree
    for _ in range(8):
        found = None
        for select in tree.find_all(exp.Select):
            found = _redundant(select)
            if found is not None:
                break
        if found is None:
            break
        condition, relation = found
        if assumptions is not None and relation.find(exp.Window) is not None:
            from .smt_equivalence import WINDOW_SOURCE_ASSUMPTION

            assumptions.add(WINDOW_SOURCE_ASSUMPTION)
        where = condition.find_ancestor(exp.Where)
        rest = [c for c in _conjuncts(where.this) if c is not condition]
        if not rest:
            where.pop()
            continue
        kept = rest[0]
        for other in rest[1:]:
            kept = exp.And(this=kept, expression=other)
        where.set("this", kept)
    return tree
