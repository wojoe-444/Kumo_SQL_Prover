"""An inner join with a grouped derived table on all of its group keys is a membership test.

**The rule.** In a select whose joins are all plain inner joins (``JOIN .. ON``, ``CROSS JOIN``, a comma),

    SELECT .. FROM y JOIN (SELECT k1, k2, COUNT(*) AS n FROM s GROUP BY k1, k2 HAVING h) AS d
                       ON y.a = d.k1 AND y.b = d.k2
    WHERE d.n > 1 AND rest

is

    SELECT .. FROM y WHERE (y.a, y.b) IN (SELECT d.k1, d.k2 FROM (SELECT k1, k2, COUNT(*) AS n .. HAVING h) AS d WHERE d.n > 1)
                       AND rest

when ``d`` is read by nothing but equalities of one of its columns with a column of another source, covering
every group key of ``d``, and conditions on ``d`` alone. When there are no such conditions and the equalities
read every output of ``d`` exactly once, the membership test is written over the derived table's own select:
``(y.a, y.b) IN (SELECT k1, k2 FROM s GROUP BY k1, k2 HAVING h)``.

**Why it is sound.** A plain ``GROUP BY`` select has one row per combination of its group keys, with NULLs
compared as equal, so two different rows of ``d`` differ in a key output. Fix a row ``r`` of the other
sources. A row of ``d`` joins with ``r`` only if every tied column of it equals ``r``'s column, which needs both
not NULL, so the rows of ``d`` that join with ``r`` agree on all key outputs and are at most one row. The join
therefore returns ``r`` once when such a row passes the conditions on ``d`` alone, and not at all otherwise;
nothing else of ``d`` is read, so nothing else of the row matters. The ``IN`` is TRUE for ``r`` exactly when the
subquery returns a row equal to ``r``'s columns in every tied position: the same row of ``d``, passing the same
conditions. Where the ``IN`` is not TRUE it is FALSE or NULL, and it stands as a ``WHERE`` conjunct. With all
joins inner, every ``ON`` conjunct filters the joined rows as a ``WHERE`` conjunct does, so the conditions of a
removed join move to ``WHERE``. The select returns the same multiset of rows before its own grouping,
aggregates, ``DISTINCT`` and ordering, whatever those are.

No column is read from another place than before, so no type condition is needed: ``y.a = d.k`` and ``y.a IN
(SELECT d.k ..)`` are the same comparison.

The rule runs once before the other rules (so that the groups of the membership test are read like those of
any ``IN`` subquery) and again among the shape rules, for joins that other steps expose: a wrapper select merged
away, an outer join that a ``WHERE`` on the derived table's count makes inner.

**What is read.** The derived table: ``grouped_source_dedup.unique_key_outputs`` (a plain ``GROUP BY`` of at
least one key, each key among its outputs, no ``ROLLUP``, window, ``LIMIT`` or ``ORDER BY``, not ``LATERAL``).
The select: sources with different names; every condition that reads ``d`` is ``d.k = y.c`` with ``y`` another
source of the select (either order) or reads ``d`` alone and holds no subquery, aggregate, window or volatile
function. A column is ``d``'s when it is qualified with ``d``'s name.

**What is refused.** Outer joins anywhere in the select (their ``ON`` is not a filter), ``USING`` and
``NATURAL``; a read of ``d`` in the select list, ``GROUP BY``, ``HAVING``, ``ORDER BY`` or a nested subquery;
``SELECT *``; a bare column with the name of an output of ``d``, in the select itself or in a nested subquery
that does not bind all of its own names (it may be ``d``'s);
equalities that leave a group key of ``d`` untied (two rows of ``d`` could then join with one row); a derived
table that is not grouped (other rules read ``SELECT DISTINCT`` ones).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, conjuncts
from .decorrelation_rules import _and_all, _closed, _sources
from .grouped_source_dedup import unique_key_outputs
from .semijoin_shapes import _VOLATILE, _derived, _inner_conjuncts, _link, _own, _under

_CLAUSES = frozenset({"expressions", "from_", "from", "joins", "where", "group", "having", "order", "distinct", "limit", "offset"})


def _rewrite(select: exp.Select, schema: dict | None) -> bool:
    if any(value for key, value in select.args.items() if key not in _CLAUSES):
        return False
    sources = _sources(select)
    names = [(s.alias_or_name or "").lower() for s in sources]
    parts = _inner_conjuncts(select)
    if parts is None or len(sources) < 2 or "" in names or len(set(names)) != len(names):
        return False
    if any(isinstance(n, exp.Star) and _own(n, select) and not isinstance(n.parent, exp.Count) for n in select.find_all(exp.Star)):
        return False
    outside = [c for c in select.find_all(exp.Column) if not any(_under(c, source) for source in sources)]
    bare: set[str] = set()  # the bare names that a source of this select may bind
    closed: dict[int, bool] = {}
    for column in outside:
        if column.table:
            continue
        if not _own(column, select):
            nested = column.find_ancestor(exp.Select)
            while nested.find_ancestor(exp.Select) is not select:
                nested = nested.find_ancestor(exp.Select)
            if id(nested) not in closed:
                closed[id(nested)] = _closed(nested, schema)
            if closed[id(nested)]:
                continue  # bound inside the nested select
        bare.add(column.name.lower())
    for index in range(len(sources) - 1, -1, -1):
        source, name = sources[index], names[index]
        carried = unique_key_outputs(source, schema)
        if carried is None:
            continue
        body = _derived(source)
        outputs = [item.alias_or_name.lower() for item in body.expressions]
        if bare & set(outputs):
            continue  # a bare name that may be a column of d
        reads = [c for c in outside if c.table.lower() == name]
        touching = [i for i, part in enumerate(parts) if any(_under(c, part) for c in reads)]
        if not touching or any(not _own(c, select) or not any(_under(c, parts[i]) for i in touching) for c in reads):
            continue
        ties: list[tuple[exp.Column, exp.Column]] = []
        local: list[int] = []
        for i in touching:
            part = parts[i]
            sides = _link(part, set(names))
            if sides is not None:
                mine, other = sides if sides[0].table.lower() == name else sides[::-1]
                if mine.name.lower() not in outputs:
                    break
                ties.append((mine, other))
                continue
            if any(isinstance(n, (exp.Select, exp.Subquery, exp.AggFunc, exp.Window, *_VOLATILE)) for n in part.walk()):
                break
            if any(c.table.lower() != name or c.name.lower() not in outputs for c in part.find_all(exp.Column)):
                break
            local.append(i)
        else:
            tied = [mine.name.lower() for mine, _ in ties]
            if not ties or not all(names_ & set(tied) for names_ in carried):
                continue
            if not local and sorted(tied) == sorted(outputs):
                order = sorted(range(len(ties)), key=lambda j: outputs.index(tied[j]))
                lefts = [ties[j][1].copy() for j in order]
                query = body.copy()
            else:
                lefts = [other.copy() for _, other in ties]
                query = exp.Select(expressions=[mine.copy() for mine, _ in ties])
                query.set(FROM_KEY, exp.From(this=source.copy()))
                condition = _and_all([parts[i].copy() for i in local])
                if condition is not None:
                    query.set("where", exp.Where(this=condition))
            membership = exp.In(this=lefts[0] if len(lefts) == 1 else exp.Tuple(expressions=lefts), query=exp.Subquery(this=query))
            used = {id(parts[i]) for i in touching}
            where = select.args.get("where")
            kept = [p.copy() for p in (conjuncts(where.this) if where is not None else []) if id(p) not in used]
            joins = list(select.args.get("joins") or [])
            if index == 0:  # d is the first source: the next one takes its place, its ON conditions become filters
                gone = joins.pop(0)
                select.set(FROM_KEY, exp.From(this=gone.this.copy()))
            else:
                gone = joins.pop(index - 1)
            kept += [p.copy() for p in conjuncts(gone.args["on"]) if id(p) not in used] if gone.args.get("on") is not None else []
            for join in joins:
                on = join.args.get("on")
                if on is not None:
                    join.set("on", _and_all([p.copy() for p in conjuncts(on) if id(p) not in used]))
            select.set("joins", joins or None)
            select.set("where", exp.Where(this=_and_all(kept + [membership])))
            return True
    return False


def unique_join_to_membership(select: exp.Select, schema: dict | None) -> exp.Expression | None:
    """``select`` (changed in place) with one such join written as a membership test, else ``None``."""

    return select if _rewrite(select, schema) else None


def read_unique_joins(tree: exp.Expression, schema: dict | None) -> exp.Expression:
    """``tree`` (changed in place) with each inner join to a grouped derived table on all of its keys written as a membership test."""

    for select in list(tree.find_all(exp.Select)):
        for _ in range(8):
            if not _rewrite(select, schema):
                break
    return tree
