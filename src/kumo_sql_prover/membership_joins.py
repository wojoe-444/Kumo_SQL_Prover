"""Inside a membership test, a nested ``IN (SELECT ..)`` is read as a join (a nested ``EXISTS`` is left to the semi-join reading).

    WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p WHERE p.f = 1))

and

    WHERE s.k IN (SELECT ps.sk FROM ps JOIN p ON ps.pk = p.k WHERE p.f = 1)

test the same thing (TPC-H Q20 and its rewrites). The prover compares a top-level ``IN`` with a join, but it reads
the query inside an ``IN`` or ``EXISTS`` as it is written, so the two inner queries never met.
``join_nested_membership`` writes the first as the second:

    SELECT ps.sk FROM ps, p WHERE ps.pk = p.k AND p.f = 1

Why this is exact. Call the inner query ``S`` (it is the whole query of an ``IN`` or ``EXISTS``) and its conjunct
``e IN (N)``. A row of ``S``'s sources passes that conjunct only when it is TRUE, which is when some row of ``N``
has an output equal to ``e`` (a NULL on either side is never TRUE, and an empty ``N`` gives FALSE). In the joined
form the row survives once per row of ``N``'s sources that passes ``N``'s filter with ``e = output`` TRUE: at least
once in exactly the same cases. The joined form can repeat a row of ``S``; it cannot add or lose a distinct one,
and ``S``'s outputs do not read the joined sources, so the *set* of rows ``S`` returns is unchanged. ``IN``,
``NOT IN`` and ``EXISTS`` read only that set (TRUE, FALSE and NULL of ``x IN (..)`` depend on which values occur,
never on how often).

Conditions, all checked:

* ``S`` is the whole query of an ``IN`` or ``EXISTS`` (``distinct_rules._membership_query``) and has only a select
  list, sources, joins, ``WHERE`` and an optional plain ``DISTINCT``: no grouping, ``HAVING``, window, ``ORDER BY``,
  ``LIMIT`` or ``OFFSET``, and no aggregate or star in its select list (those read more than the set of rows);
* the test is a top-level ``WHERE`` conjunct ``e IN (N)`` (not ``NOT IN``, not under ``OR``) with one value;
* ``N`` is a plain select of one expression from base tables the schema lists, joined by commas or inner joins
  with ``ON``: no grouping, ``HAVING``, window, tail, set operation or nested query (a plain ``DISTINCT`` is dropped:
  the test ignores repeats);
* names keep their meaning. Every column of ``N`` is qualified with one of ``N``'s sources or, when ``N`` has one
  source, is a bare name: one the table has is qualified with it, any other is a reference to an enclosing query
  and binds as before, because the table does not have it. A source of ``N`` whose name is already used in
  ``S`` (as a source, or as a qualifier anywhere outside ``N``) is renamed to a fresh alias. No bare column of ``S``
  outside ``N`` may have a name that a source of ``N`` has: it would bind to the joined table.
"""

from __future__ import annotations

import itertools

from sqlglot import exp

from .ast_utils import select_sources
from .distinct_rules import _membership_query
from .unmatched_outer_join import _columns_of, _conjuncts, _inside, _plain_column, _plain_table

_OUTER_ARGS = frozenset({"expressions", "from_", "from", "joins", "where", "distinct"})
_INNER_ARGS = frozenset({"expressions", "from_", "from", "joins", "where", "distinct"})
_FRESH = itertools.count()


def _and_all(parts: list[exp.Expression]) -> exp.Expression:
    parts = [exp.Paren(this=p) if isinstance(p, (exp.Or, exp.Xor)) else p for p in parts]  # they bind looser than AND
    result = parts[0]
    for part in parts[1:]:
        result = exp.And(this=result, expression=part)
    return result


def _plain_joins(select: exp.Select) -> bool:
    """Every join of ``select`` is a comma, ``CROSS JOIN`` or inner ``JOIN .. ON``."""

    for join in select.args.get("joins") or []:
        if join.args.get("side") or join.args.get("using") or join.args.get("method"):
            return False
        if (join.args.get("kind") or "").upper() not in ("", "INNER", "CROSS"):
            return False
        if any(v for key, v in join.args.items() if key not in ("this", "kind", "on")):
            return False
    return True


def _nested(test: exp.Expression) -> tuple[exp.Select, exp.Expression | None] | None:
    """``(N, e)`` when ``test`` is ``e IN (N)`` with one value.

    A nested ``EXISTS`` is not read: the prover's semi-join reading already spells it, and a pair of DSB queries
    (about 6,000 to 11,000 characters, too long for the shape fallback) lost a proof on main because this rule
    rewrote an ``EXISTS`` inside an ``EXISTS`` on one query only.
    """

    if type(test) is not exp.In or any(v for key, v in test.args.items() if key not in ("this", "query")):
        return None
    query = test.args.get("query")
    if type(query) is not exp.Subquery or query.alias or any(v for key, v in query.args.items() if key != "this"):
        return None
    inner = query.this
    if type(inner) is not exp.Select or isinstance(test.this, (exp.Tuple, exp.Struct)) or len(inner.expressions) != 1:
        return None
    return inner, test.this


def join_nested_membership(select: exp.Select, schema: dict | None) -> exp.Expression | None:
    """``select`` (the query of a membership test) with one nested ``e IN (SELECT ..)`` written as a join, or ``None``."""

    where = select.args.get("where")
    if where is None or not schema or not _membership_query(select):
        return None
    if any(value for key, value in select.args.items() if key not in _OUTER_ARGS):
        return None
    distinct = select.args.get("distinct")
    if distinct is not None and distinct.args.get("on"):
        return None
    if any(isinstance(n, (exp.AggFunc, exp.Window, exp.Star)) for item in select.expressions for n in item.walk()):
        return None
    if any(j.args.get("using") or j.args.get("method") for j in select.args.get("joins") or []):
        return None
    declared = {str(t).lower(): {str(c).lower() for c in columns} for t, columns in schema.items()}
    tests = _conjuncts(where.this)
    for position, test in enumerate(tests):
        found = _nested(test)
        if found is None:
            continue
        inner, left = found
        if any(value for key, value in inner.args.items() if key not in _INNER_ARGS):
            continue
        inner_distinct = inner.args.get("distinct")
        if (inner_distinct is not None and inner_distinct.args.get("on")) or not _plain_joins(inner):
            continue
        if any(isinstance(n, (exp.Query, exp.Subquery, exp.Exists, exp.AggFunc, exp.Window, exp.Star)) for n in inner.walk() if n is not inner):
            continue
        sources = select_sources(inner)
        if not sources or not all(_plain_table(s) for s in sources):
            continue
        names = [(s.alias_or_name or "").lower() for s in sources]
        columns = [_columns_of(s, declared) for s in sources]
        if not all(names) or len(set(names)) != len(names) or not all(columns):
            continue
        owned = set().union(*columns)
        # what the enclosing select already calls things, outside the nested query
        used = {(s.alias_or_name or "").lower() for s in select_sources(select)}
        captured = False
        for column in select.find_all(exp.Column):
            if _inside(column, inner):
                continue
            if column.table:
                used.update(part.name.lower() for part in column.parts[:-1])
            elif column.name.lower() in owned:
                captured = True
                break
        if captured:
            continue
        copy = select.copy()
        new_test = _conjuncts(copy.args["where"].this)[position]
        new_inner = _nested(new_test)[0]
        renamed: dict[str, str] = {}
        for source, name in zip(select_sources(new_inner), names):
            if name in used:
                renamed[name] = f"kqm{next(_FRESH)}"
                source.set("alias", exp.TableAlias(this=exp.to_identifier(renamed[name])))
        resolved = True
        for column in list(new_inner.find_all(exp.Column)):
            plain = _plain_column(column)
            if plain is None:
                resolved = False
                break
            table = plain.table.lower()
            if table in names:
                if table in renamed:
                    column.set("table", exp.to_identifier(renamed[table]))
            elif table:
                continue  # a source of an enclosing query, by name
            elif len(names) != 1:
                resolved = False
                break
            elif plain.name.lower() in columns[0]:
                column.set("table", exp.to_identifier(renamed.get(names[0], names[0])))
        if not resolved:
            continue
        parts = [t for index, t in enumerate(_conjuncts(copy.args["where"].this)) if index != position]
        if left is not None:
            value = new_inner.expressions[0]
            value = value.this if isinstance(value, exp.Alias) else value
            left = new_test.this
            if isinstance(left, (exp.Binary, exp.Connector, exp.Predicate, exp.Not)):
                left = exp.Paren(this=left)
            if isinstance(value, (exp.Binary, exp.Connector, exp.Predicate, exp.Not)):
                value = exp.Paren(this=value)
            parts.append(exp.EQ(this=left.copy(), expression=value.copy()))
        inner_where = new_inner.args.get("where")
        if inner_where is not None:
            parts.extend(part.copy() for part in _conjuncts(inner_where.this))
        added = []
        inner_from = new_inner.args.get("from_") or new_inner.args.get("from")
        added.append(exp.Join(this=inner_from.this.copy()))
        for join in new_inner.args.get("joins") or []:
            on = join.args.get("on")
            added.append(exp.Join(this=join.this.copy()))
            if on is not None:
                parts.extend(part.copy() for part in _conjuncts(on))
        copy.set("where", exp.Where(this=_and_all([p.copy() for p in parts])))
        copy.set("joins", list(copy.args.get("joins") or []) + added)
        return copy
    return None
