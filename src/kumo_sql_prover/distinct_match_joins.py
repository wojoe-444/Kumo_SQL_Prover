"""Under ``DISTINCT`` or in an ``IN`` list, a joined table that no output reads is only asked for a match.

    SELECT DISTINCT p.patron_id, p.patron_name
    FROM patron AS p JOIN loan AS o1 ON p.patron_id = o1.patron_id AND o1.title = 'A'
                     JOIN loan AS o2 ON p.patron_id = o2.patron_id AND o2.title = 'B'
    WHERE p.patron_id NOT IN (SELECT t.patron_id FROM loan AS t WHERE t.title = 'C')

lists each ``(patron_id, patron_name)`` of a patron row that has a partner in ``o1`` and one in ``o2``. It is

    SELECT DISTINCT p.patron_id, p.patron_name FROM patron AS p
    WHERE p.patron_id NOT IN (SELECT t.patron_id FROM loan AS t WHERE t.title = 'C')
      AND EXISTS (SELECT 1 FROM loan AS o1 WHERE p.patron_id = o1.patron_id AND o1.title = 'A')
      AND EXISTS (SELECT 1 FROM loan AS o2 WHERE p.patron_id = o2.patron_id AND o2.title = 'B')

**Why it holds.** Call the sources an output reads *kept* and the others *asked*. With inner joins only, the
select keeps the combinations of rows on which every ``ON`` and ``WHERE`` conjunct is TRUE, lists the kept rows'
columns, and ``DISTINCT`` leaves each listed tuple once. A combination of kept rows contributes its tuple exactly
when some rows of the asked sources complete it, however many do. The conjuncts that read no asked source do not
depend on the choice; the others say which asked rows complete it. Asked sources that no conjunct reads together
are chosen independently, so "some rows complete it" is one ``EXISTS`` per group of asked sources that conjuncts
connect. No key or other declared fact is needed: ``DISTINCT`` already removes what the joins multiply.

Two asked sources tied only through a kept column are separated first: ``a.x = p.z AND a.x = b.y`` is
``a.x = p.z AND p.z = b.y`` when ``nested_correlation.interchangeable`` lets one column be read for the other.

The list of an ``IN`` or ``NOT IN`` is read the same way, with or without ``DISTINCT``: the test asks which
values the list holds (and whether it holds a NULL), not how often, and a kept row lists its value exactly when
asked rows complete it.

    x NOT IN (SELECT s.buyer FROM item AS i JOIN sale AS s ON s.item_id = i.item_id WHERE i.label = 'I')
    x NOT IN (SELECT s.buyer FROM sale AS s WHERE EXISTS (SELECT 1 FROM item AS i WHERE s.item_id = i.item_id AND i.label = 'I'))

``keyed_semijoin`` makes the ``DISTINCT`` step for two base tables when the list holds a key (it runs first), and
other rules read a ``DISTINCT`` select without nested queries, which this rule leaves to them.

**What is read.** A ``SELECT DISTINCT`` (no ``ON``) that holds at least one nested query, or the select of an
``IN`` list, over two or more base tables under inner, cross or comma joins with ``ON`` only, each under its own
name, with no grouping, aggregate, window, ``ORDER BY`` or row limit; outputs are plain columns; every column, at any depth, names its source (the bare columns of a nested one-table
query are given that table when the caller supplied its columns); no nested query reuses a source's name; every asked source is read by some conjunct. A volatile call leaves the
select as it is.

A narrow retry rule (``shape_fallback``). ``tests/test_distinct_match_joins.py`` replays the rewrite on DuckDB.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, conjuncts
from .nested_correlation import interchangeable
from .own_group_membership import _plain_table, _unparen
from .own_row_copies import _named

_BANNED = ("group", "having", "qualify", "windows", "with_", "with", "limit", "offset", "order", "laterals", "pivots", "into", "locks", "sample", "connect", "match")
_JOIN_EXTRAS = ("method", "using", "match_condition", "global_", "hint", "directed", "expressions", "pivots")


def ask_unlisted_joins(select: exp.Select, types: dict | None = None, schema: dict | None = None) -> exp.Select | None:
    """``select`` with the joined tables no output reads moved into ``EXISTS`` tests, or None."""

    from .counted_membership import _unstable
    from .keyed_semijoin import name_nested_sources, rename_nested_sources

    distinct = select.args.get("distinct")
    from_ = select.args.get(FROM_KEY)
    joins = select.args.get("joins") or []
    holder = select.parent
    listing = isinstance(holder, exp.Subquery) and not holder.alias and isinstance(holder.parent, exp.In) and holder.arg_key == "query"
    if (distinct is None and not listing) or (distinct is not None and distinct.args.get("on")) or from_ is None or not joins or any(select.args.get(key) for key in _BANNED):
        return None
    for join in joins:
        if join.side or (join.kind or "").upper() not in ("", "INNER", "CROSS") or any(join.args.get(key) for key in _JOIN_EXTRAS):
            return None
    sources = [from_.this] + [join.this for join in joins]
    if any(not _plain_table(source) or not isinstance(source.this, exp.Identifier) or source.this.args.get("quoted") for source in sources):
        return None
    names = [(source.alias_or_name or "").lower() for source in sources]
    if len(set(names)) != len(names):
        return None
    # a nested query's own source gets its own name, and the bare columns of a nested one-table query name that table
    select = name_nested_sources(rename_nested_sources(select, set(names)), schema)
    distinct, from_, joins = select.args.get("distinct"), select.args.get(FROM_KEY), select.args.get("joins") or []
    sources = [from_.this] + [join.this for join in joins]
    nested = [node for node in select.find_all(exp.Select) if node is not select]
    if not nested and not listing:
        return None  # left to the rules that read a select without nested queries
    if any(node.find_ancestor(exp.Select) is select for node in select.find_all(exp.AggFunc, exp.Window)) or select.find(exp.Star) is not None:
        return None
    if any(not column.table or column.args.get("db") for column in select.find_all(exp.Column)):
        return None
    for query in nested:
        inner_from = query.args.get(FROM_KEY)
        inner = ([inner_from.this] if inner_from is not None else []) + [join.this for join in query.args.get("joins") or []]
        if any((source.alias_or_name or "").lower() in names or not (source.alias_or_name or "") for source in inner):
            return None  # a nested query that reuses a name could read another row under it
    listed = []
    for item in select.expressions:
        found = _named(item.unalias())
        if found is None or found[0] not in names:
            return None
        listed.append(found[0])
    kept = [name for name in names if name in listed]
    asked = [name for name in names if name not in listed]
    if not asked:
        return None
    if _unstable(select):
        return None
    by_name = dict(zip(names, sources))
    parts = [part.copy() for join in joins if join.args.get("on") is not None for part in conjuncts(join.args["on"])]
    if select.args.get("where") is not None:
        parts += [part.copy() for part in conjuncts(select.args["where"].this)]

    def readers(part: exp.Expression) -> set[str]:
        return {column.table.lower() for column in part.find_all(exp.Column) if column.table.lower() in asked}

    # a column of an asked source that a top-level equality ties to a kept column
    tied: dict[tuple[str, str], tuple[str, str]] = {}
    for part in parts:
        test = _unparen(part)
        if isinstance(test, exp.EQ):
            a, b = _named(test.this), _named(test.expression)
            if a is None or b is None:
                continue
            for mine, theirs in ((a, b), (b, a)):
                if mine[0] in asked and theirs[0] in kept and interchangeable(types, (by_name[mine[0]].name.lower(), mine[1]), (by_name[theirs[0]].name.lower(), theirs[1])):
                    tied.setdefault(mine, theirs)
    for index, part in enumerate(parts):
        test = _unparen(part)
        if not isinstance(test, exp.EQ) or len(readers(part)) != 2:
            continue
        a, b = _named(test.this), _named(test.expression)
        if a is None or b is None or a[0] == b[0]:
            continue
        for mine, other in ((a, b), (b, a)):
            through = tied.get(mine)
            if through is not None and interchangeable(types, (by_name[through[0]].name.lower(), through[1]), (by_name[other[0]].name.lower(), other[1])):
                # mine = through holds beside it, so this equality is one between the kept column and the other
                parts[index] = exp.EQ(this=exp.column(through[1], table=by_name[through[0]].alias_or_name), expression=exp.column(other[1], table=by_name[other[0]].alias_or_name))
                break

    group = {name: name for name in asked}  # asked sources that some conjunct reads together

    def find(name: str) -> str:
        while group[name] != name:
            name = group[name]
        return name

    for part in parts:
        read = sorted(readers(part))
        for other in read[1:]:
            group[find(other)] = find(read[0])
    outside = [part for part in parts if not readers(part)]
    tests = []
    for root in sorted({find(name) for name in asked}, key=asked.index):
        members = [name for name in asked if find(name) == root]
        inside = [part for part in parts if readers(part) & set(members)]
        if not inside:
            return None  # nothing ties it to the rest: a cross product, left to the other rules
        probe = exp.select("1").from_(by_name[members[0]].copy())
        for name in members[1:]:
            probe.append("joins", exp.Join(this=by_name[name].copy()))
        probe.set("where", exp.Where(this=exp.and_(*inside, copy=False)))
        tests.append(exp.Exists(this=probe))
    result = exp.Select(expressions=[item.copy() for item in select.expressions])
    if distinct is not None:
        result.set("distinct", distinct.copy())
    result.set(FROM_KEY, exp.From(this=by_name[kept[0]].copy()))
    if kept[1:]:
        result.set("joins", [exp.Join(this=by_name[name].copy()) for name in kept[1:]])
    result.set("where", exp.Where(this=exp.and_(*outside, *tests, copy=False)))
    return result
