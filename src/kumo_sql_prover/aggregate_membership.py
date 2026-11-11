"""``(k, v) IN (SELECT k, MIN(v) FROM t GROUP BY k)`` read as a comparison with the group's own aggregate.

    SELECT player_id, device_id FROM activity
    WHERE (player_id, event_date) IN (SELECT player_id, MIN(event_date) FROM activity GROUP BY player_id)

keeps the rows of each player's first day. The SMT prover does not read an ``IN`` whose subquery computes an
aggregate, so the pair went undecided whatever the other query looked like. The same test, written the way the
prover already reads (and decorrelates), is

    WHERE activity.event_date = (SELECT MIN(f.event_date) FROM activity AS f WHERE f.player_id = activity.player_id)

**Why the two are the same test.** The grouped subquery returns one row per group, and its groups are told
apart by the keys ``k1 .. kn``. When every key is output, a row ``(x1 .. xn, y)`` on the left can only equal the
row of the group whose keys are ``x1 .. xn``:

* that group exists exactly when some row of ``t`` (passing the subquery's ``WHERE``) has ``k = x`` on every key,
  which is when the scalar subquery ``SELECT AGG(v) FROM t WHERE .. AND k = x`` reads a row at all;
* its aggregate is the scalar subquery's value, because both aggregate the same rows;
* with no such group the ``IN`` is not TRUE, and the scalar subquery aggregates no rows, so ``MIN``, ``MAX``,
  ``SUM`` and ``AVG`` are NULL there and the comparison is not TRUE either. ``COUNT`` is 0 over no rows, which a
  left value of 0 would equal, so a grouped subquery that outputs a ``COUNT`` (or any expression around an
  aggregate) is left alone;
* a NULL key never equals anything: the ``IN`` is not TRUE for the NULL group, and ``k = x`` keeps no row.

The rewrite is made only where "not TRUE" is all that matters: a top-level ``AND`` part of a ``WHERE``. Under
``NOT``, in a select list or in an ``OR`` the difference between FALSE and NULL shows, and the test is kept.

**Without ``GROUP BY``** the subquery returns exactly one row, so ``(x, y) IN (SELECT MIN(a), COUNT(*) FROM t)``
is ``x = (SELECT MIN(a) FROM t) AND y = (SELECT COUNT(*) FROM t)`` for any aggregate expressions.

**What is read.** The subquery is one select over one base table whose columns the caller supplied, with an
optional ``WHERE``, a ``GROUP BY`` of plain columns and nothing else (no ``HAVING``, ``DISTINCT``, window, join,
``LIMIT`` or nested query). Each output is a group key or a plain ``MIN``, ``MAX``, ``SUM`` or ``AVG``; every key
is output; every column in it is the table's own (a bare name the table has, or one qualified by the table). The
left operands hold no subquery, aggregate or window.

**Names.** The left operands move one level down, into the new scalar subqueries, so each of their columns has to
name its source first: a bare one gets the qualifier of the single source of the enclosing select that has it
(every source of that select must then be a base table with supplied columns), through
``_checked_qualification`` like every other pass that writes qualifiers. The subquery's table is given a fresh
alias that occurs nowhere in the statement, so it cannot capture any of them.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, conjuncts, extended_grouping, select_sources

_SELECT_ARGS = frozenset({"expressions", "from", "from_", "where", "group", "comments", "kind", "hint"})
_TABLE_ARGS = frozenset({"this", "alias", "db", "catalog", "comments"})
_NULL_ON_NO_ROWS = (exp.Min, exp.Max, exp.Sum, exp.Avg)
_PREFIX = "kqm"


def read_aggregate_membership(tree: exp.Expression, schema: dict[str, list[str]] | None, dialect: str = "bigquery") -> exp.Expression:
    """Rewrite, in place, each membership test described in the module docstring."""

    if not schema or not any(True for _ in tree.find_all(exp.In)):
        return tree
    from .algebraic_equivalence import _checked_qualification

    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    sites = [node for node in tree.find_all(exp.In) if _site(node, known) is not None]
    if not sites:
        return tree
    tree = _checked_qualification(tree, lambda t: _qualify_operands(t, known), schema, dialect, "aggregate_membership_qualification")
    taken = _names(tree)
    for node in list(tree.find_all(exp.In)):
        site = _site(node, known)
        if site is None or not _qualified(node):
            continue
        replacement = _comparisons(node, site, taken)
        if node is tree:
            tree = replacement
        else:
            node.replace(replacement)
    return tree


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _table_key(table: exp.Table) -> str:
    return ".".join(p.name for p in (table.args.get("catalog"), table.args.get("db"), table.this) if p is not None).lower()


def _base_table(source: exp.Expression, known: dict[str, list[str]]) -> list[str] | None:
    """The supplied columns of a plain base-table source, else None."""

    if not isinstance(source, exp.Table) or not isinstance(source.this, exp.Identifier):
        return None
    if any(value for key, value in source.args.items() if key not in _TABLE_ARGS):
        return None
    alias = source.args.get("alias")
    if alias is not None and alias.args.get("columns"):
        return None
    return known.get(_table_key(source))


def _lefts(node: exp.In) -> list[exp.Expression]:
    left = _unwrap(node.this)
    return list(left.expressions) if isinstance(left, exp.Tuple) else [left]


def _positive_where_part(node: exp.In) -> exp.Select | None:
    """The select whose ``WHERE`` has ``node`` as a top-level ``AND`` part, else None."""

    child, parent = node, node.parent
    while isinstance(parent, (exp.Paren, exp.And)):
        child, parent = parent, parent.parent
    if not isinstance(parent, exp.Where) or parent.this is not child or not isinstance(parent.parent, exp.Select):
        return None
    return parent.parent if any(part is node for part in conjuncts(parent.this)) else None


def _site(node: exp.In, known: dict[str, list[str]]):
    """``(inner select, table, key positions, aggregate positions)`` when ``node`` is a test the module reads."""

    if node.args.get("expressions") or node.args.get("unnest") or node.args.get("field") or _positive_where_part(node) is None:
        return None
    query = node.args.get("query")
    inner = query.this if isinstance(query, exp.Subquery) else query
    if isinstance(query, exp.Subquery) and any(value for key, value in query.args.items() if key not in ("this", "comments")):
        return None
    if not isinstance(inner, exp.Select) or any(value for key, value in inner.args.items() if key not in _SELECT_ARGS):
        return None
    sources = select_sources(inner)
    if len(sources) != 1:
        return None
    table = sources[0]
    columns = _base_table(table, known)
    if columns is None or not table.alias_or_name:
        return None
    own = table.alias_or_name.lower()
    for walked in inner.walk():
        if walked is not inner and isinstance(walked, (exp.Query, exp.Subquery, exp.Window, exp.Exists, exp.Lambda)):
            return None
        if isinstance(walked, exp.Column):
            if isinstance(walked.this, exp.Star) or walked.args.get("db") or walked.args.get("catalog"):
                return None
            if walked.name.lower() not in columns or (walked.table and walked.table.lower() != own):
                return None
        elif isinstance(walked, exp.Star) and not isinstance(walked.parent, exp.Count):
            return None
    lefts = _lefts(node)
    if len(lefts) != len(inner.expressions):
        return None
    for left in lefts:
        if any(isinstance(n, (exp.Query, exp.Subquery, exp.AggFunc, exp.Window, exp.Exists, exp.Star)) for n in left.walk()):
            return None
    group = inner.args.get("group")
    outputs = [_unwrap(item.this if isinstance(item, exp.Alias) else item) for item in inner.expressions]
    if group is None:
        # one row whatever the table holds: every output must be computed from aggregates alone
        for output in outputs:
            if not any(True for _ in output.find_all(exp.AggFunc)):
                return None
            if any(isinstance(n, exp.Column) and n.find_ancestor(exp.AggFunc) is None for n in output.walk()):
                return None
            if any(call.find_ancestor(exp.AggFunc) is not None for call in output.find_all(exp.AggFunc)):
                return None
        return inner, table, {}, list(range(len(outputs)))
    if extended_grouping(group) or any(value for key, value in group.args.items() if key != "expressions"):
        return None
    keys = []
    for key in group.expressions:
        key = _unwrap(key)
        if not isinstance(key, exp.Column):
            return None
        keys.append(key.name.lower())
    if not keys or len(set(keys)) != len(keys):
        return None
    key_positions: dict[int, str] = {}
    aggregate_positions: list[int] = []
    for position, output in enumerate(outputs):
        if isinstance(output, exp.Column) and output.name.lower() in keys:
            key_positions[position] = output.name.lower()
        elif type(output) in _NULL_ON_NO_ROWS and not any(value for key, value in output.args.items() if key != "this") and not any(
            isinstance(n, exp.AggFunc) for n in output.this.walk()
        ) and not isinstance(output.this, exp.Distinct):
            aggregate_positions.append(position)
        else:
            return None
    if not aggregate_positions or set(key_positions.values()) != set(keys):
        return None
    return inner, table, key_positions, aggregate_positions


def _owners(select: exp.Select, known: dict[str, list[str]]) -> dict[str, list[str]] | None:
    """Column name -> the sources of ``select`` that have it; None unless every source is a base table with supplied columns."""

    owners: dict[str, list[str]] = {}
    seen: set[str] = set()
    for source in select_sources(select):
        columns = _base_table(source, known)
        name = source.alias_or_name.lower() if columns is not None else ""
        if not name or name in seen:
            return None
        seen.add(name)
        for column in columns:
            owners.setdefault(column, []).append(source.alias_or_name)
    return owners


def _qualify_operands(tree: exp.Expression, known: dict[str, list[str]]) -> exp.Expression:
    """Name the source of each bare column in the left operands of the tests the module reads."""

    for node in tree.find_all(exp.In):
        if _site(node, known) is None:
            continue
        select = _positive_where_part(node)
        bare = [c for left in _lefts(node) for c in left.find_all(exp.Column) if not c.table]
        if not bare:
            continue
        owners = _owners(select, known)
        if owners is None or any(len(owners.get(c.name.lower(), ())) != 1 for c in bare):
            continue  # left as written: the rewrite then declines this test
        for column in bare:
            column.set("table", exp.to_identifier(owners[column.name.lower()][0]))
    return tree


def _qualified(node: exp.In) -> bool:
    return all(c.table for left in _lefts(node) for c in left.find_all(exp.Column))


def _names(tree: exp.Expression) -> set[str]:
    return {identifier.name.lower() for identifier in tree.find_all(exp.Identifier)}


def _fresh(taken: set[str]) -> str:
    number = 0
    while f"{_PREFIX}{number}" in taken:
        number += 1
    taken.add(f"{_PREFIX}{number}")
    return f"{_PREFIX}{number}"


def _comparisons(node: exp.In, site, taken: set[str]) -> exp.Expression:
    inner, table, key_positions, aggregate_positions = site
    lefts = _lefts(node)
    outputs = [_unwrap(item.this if isinstance(item, exp.Alias) else item) for item in inner.expressions]
    parts = []
    for position in aggregate_positions:
        alias = _fresh(taken)
        source = table.copy()
        source.set("alias", exp.TableAlias(this=exp.to_identifier(alias)))

        def own(piece: exp.Expression) -> exp.Expression:
            piece = piece.copy()
            for column in piece.find_all(exp.Column):
                column.set("table", exp.to_identifier(alias))
            return piece

        scalar = exp.Select(expressions=[own(outputs[position])])
        scalar.set(FROM_KEY, exp.From(this=source))
        tests = [own(part) for part in conjuncts(inner.args["where"].this)] if inner.args.get("where") is not None else []
        tests = [exp.Paren(this=test) if isinstance(test, exp.Or) else test for test in tests]
        for key_position, key in key_positions.items():
            tests.append(exp.EQ(this=exp.column(key, table=alias), expression=_operand(lefts[key_position])))
        if tests:
            condition = tests[0]
            for test in tests[1:]:
                condition = exp.And(this=condition, expression=test)
            scalar.set("where", exp.Where(this=condition))
        parts.append(exp.EQ(this=_operand(lefts[position]), expression=exp.Subquery(this=scalar)))
    result = parts[0]
    for part in parts[1:]:
        result = exp.And(this=result, expression=part)
    return exp.Paren(this=result) if len(parts) > 1 else result


def _operand(node: exp.Expression) -> exp.Expression:
    node = node.copy()
    return node if isinstance(node, (exp.Column, exp.Literal, exp.Paren)) else exp.Paren(this=node)
