"""A table joined only to be matched is a membership test, in a select that cannot see repeated rows.

**The rule.** In a select whose result does not depend on how often an input row repeats (below), an inner-joined
source ``x`` that the select reads only through one equality ``x.c = y.d`` with a column of another source is
replaced by the test ``y.d IN (SELECT x.c FROM .. AS x)``; every other place that names ``x.c`` reads ``y.d``:

    SELECT a.k, COUNT(DISTINCT b.v) FROM t AS a JOIN t AS b ON a.k = b.g GROUP BY a.k
    SELECT b.g AS k, COUNT(DISTINCT b.v) FROM t AS b WHERE b.g IN (SELECT a.k FROM t AS a) GROUP BY b.g

Conditions that read ``x`` alone (``x.e > 0``) move into the subquery's ``WHERE``; one that holds a subquery of
its own stays in place as a condition on ``y.d`` when ``x.c`` is the only column of ``x`` it reads. A derived table ``x`` that
returns only the matched column, with no condition of its own in the select, is used as the subquery itself.

**Why it is sound.** Write ``R`` for the rows the other sources and the other conditions produce, and for a
row ``r`` of ``R`` let ``n(r)`` be the number of rows of ``x`` that pass ``x``'s own conditions and have
``x.c = y.d(r)``. The join holds ``r`` exactly ``n(r)`` times, each time with ``x.c`` equal to ``y.d(r)``;
nothing else of ``x`` is read, so with ``x.c`` read as ``y.d`` the ``n(r)`` copies are identical rows.
The membership test holds ``r`` once when ``n(r) >= 1`` and not at all otherwise: ``y.d IN (..)`` is TRUE
exactly when some row of the subquery equals ``y.d`` (so both are not NULL), and where it is not TRUE it is
FALSE or NULL, which a ``WHERE`` conjunct treats alike. A NULL ``y.d`` or ``x.c`` matches nothing in either
form. So the two forms feed the select the same *set* of different rows, with different repeat counts, and the
select is one that reads only the set:

* a plain ``SELECT DISTINCT`` without aggregates, or
* a select with a plain ``GROUP BY`` and/or aggregates in which every aggregate is ``MIN``, ``MAX``,
  ``COUNT(DISTINCT ..)``, ``SUM(DISTINCT ..)``, ``AVG(DISTINCT ..)`` or a ``COUNT`` that is only compared with
  0 or 1 to ask whether it is positive (``COUNT(v) > 0``, ``>= 1``, ``<> 0``, ``= 0``, ``< 1``, ``<= 0``: TRUE
  or FALSE by whether a counted row exists, however often). The groups are the same in both forms (a group
  exists when a row with its key does), and each such aggregate is a function of the set of values in the
  group. Columns outside aggregates must be ``GROUP BY`` columns, so that no value is picked from an arbitrary
  row of a group; this is checked on the rewritten select, where ``x.c`` and ``y.d`` are one column
  (``SELECT y.d .. GROUP BY x.c`` reads its group key). A column of a grouped derived table is as good as a
  ``GROUP BY`` column when every group key of that derived table is one: the derived table has one row per
  combination of its keys, so all rows of a group hold the same row of it
  (``grouped_source_dedup.unique_key_outputs``).

Reading ``x.c`` as ``y.d`` replaces a value by one that compares equal to it. Equal values are one value only
within one type (``1 = 1.0``, MySQL converts a string compared with a number), so when ``x.c`` is read anywhere
but in the equality, the rule needs both columns declared with one type that is not a floating type, or neither
declared (the prover's standing same-type assumption, which ``semijoin_shapes`` step 4 relies on too; like the
rest of the prover it takes two strings that compare equal to be the same string).

**What is read.** A select with only a select list, ``FROM``, inner joins (``JOIN .. ON``, ``CROSS JOIN``,
comma), ``WHERE``, ``GROUP BY``, ``HAVING``, ``ORDER BY`` and ``DISTINCT``; two or more sources with different
names, each a table or a derived table; every column of the select qualified with one of those names. The
source ``x`` that goes is the last one that qualifies. Its link must be the one condition that reads ``x``
together with another source and must be a plain ``x.c = y.d``.

**What is refused.** Outer joins (the padded rows are not a membership test), ``USING`` and ``NATURAL``,
``LIMIT``/``OFFSET`` (which rows come first may depend on repeats), window functions, ``ROLLUP``/``CUBE``/grouping
sets, volatile functions, a ``COUNT``, ``SUM`` or ``AVG`` that sees repeats, a ``DISTINCT ON``, a column of
``x`` other than ``x.c`` read anywhere outside ``x``'s own conditions, any read of ``x`` from a nested subquery,
a bare column the rule cannot attribute, a second condition tying ``x`` to another source, and a link whose
other side is an expression.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, extended_grouping, plain_distinct
from .decorrelation_rules import _and_all, _closed, _sources
from .grouped_source_dedup import unique_key_outputs
from .semijoin_shapes import _FLOATING, _VOLATILE, _declared_type, _derived, _inner_conjuncts, _link, _only, _own, _plain_column, _plain_table, _under

_CLAUSES = frozenset({"expressions", "from_", "from", "joins", "where", "group", "having", "order", "distinct"})
_BLIND = (exp.Min, exp.Max)
_DISTINCT_BLIND = (exp.Count, exp.Sum, exp.Avg)


def _literal_number(node: exp.Expression) -> int | None:
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Literal) and not node.is_string and node.this in ("0", "1"):
        return int(node.this)
    return None


def positivity_test(count: exp.Expression) -> bool:
    """Whether ``count`` (a ``COUNT``) is read only by a comparison that asks if it is zero or not."""

    parent = count.parent
    while isinstance(parent, exp.Paren):
        count, parent = parent, parent.parent
    if type(parent) not in (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ, exp.NEQ):
        return False
    other = parent.expression if parent.this is count else parent.this
    number = _literal_number(other)
    if number is None:
        return False
    kind = type(parent)
    if parent.this is not count:  # literal first: mirror the comparison
        kind = {exp.GT: exp.LT, exp.LT: exp.GT, exp.GTE: exp.LTE, exp.LTE: exp.GTE}.get(kind, kind)
    # with a count n >= 0: n > 0, n <> 0, n = 0, n <= 0 split at zero; so do n >= 1 and n < 1
    return (number == 0 and kind in (exp.GT, exp.NEQ, exp.EQ, exp.LTE)) or (number == 1 and kind in (exp.GTE, exp.LT))


def _set_reading(select: exp.Select) -> bool:
    """Whether the DISTINCT, grouping and aggregates of ``select`` read only the set of its input rows."""

    distinct = select.args.get("distinct")
    if distinct is not None and not plain_distinct(select):
        return False
    group = select.args.get("group")
    if group is not None and (extended_grouping(group) or any(value for key, value in group.args.items() if key != "expressions")):
        return False
    aggregates = [n for n in select.find_all(exp.AggFunc) if _own(n, select)]
    for node in select.walk():
        if isinstance(node, (exp.Window, *_VOLATILE)) and _own(node, select):
            return False
    if group is None and not aggregates:
        return distinct is not None
    for node in aggregates:
        if isinstance(node.parent, exp.Filter) or any(value for key, value in node.args.items() if key not in ("this", "big_int")):
            return False
        if isinstance(node, _BLIND):
            continue
        if isinstance(node, _DISTINCT_BLIND) and isinstance(node.this, exp.Distinct) and not node.this.args.get("on"):
            continue
        if type(node) is exp.Count and positivity_test(node):
            continue
        return False
    return True


def _reads_only_keys(select: exp.Select, schema: dict | None = None) -> bool:
    """Whether a grouped or aggregating ``select`` reads, outside its aggregates, only ``GROUP BY`` columns, or
    columns of a grouped derived table all of whose group keys are ``GROUP BY`` columns (one row of it per group)."""

    group = select.args.get("group")
    if group is None and not any(_own(n, select) for n in select.find_all(exp.AggFunc)):
        return True  # a plain SELECT DISTINCT: every output row is computed from one input row
    keys = set()
    for item in group.expressions if group is not None else []:
        column = _plain_column(item)
        if column is None or not column.table:
            return False
        keys.add((column.table.lower(), column.name.lower()))
    fixed = set()  # sources that contribute one row to every group
    for source in _sources(select):
        name = (source.alias_or_name or "").lower()
        carried = unique_key_outputs(source, schema)
        if name and carried is not None and all(any((name, output) in keys for output in outputs) for outputs in carried):
            fixed.add(name)
    for clause in ("expressions", "having", "order"):
        value = select.args.get(clause)
        for part in value if isinstance(value, list) else [value] if value is not None else []:
            if part.find(exp.Select) is not None:
                return False  # a subquery may read any column of the group's rows
            for column in part.find_all(exp.Column):
                if column.find_ancestor(exp.AggFunc) is None and (column.table.lower(), column.name.lower()) not in keys and column.table.lower() not in fixed:
                    return False
    return True


def repeat_blind(select: exp.Select) -> bool:
    """Whether ``select`` returns the same rows whenever its input holds the same set of different rows."""

    return _set_reading(select) and _reads_only_keys(select)


def join_to_membership(select: exp.Select, schema: dict | None, types: dict | None = None) -> exp.Expression | None:
    """``select`` with one matched-only source turned into an ``IN`` test (a new select), else ``None``."""

    if not _only(select, _CLAUSES) or not _set_reading(select):
        return None
    sources = _sources(select)
    names = [(s.alias_or_name or "").lower() for s in sources]
    parts = _inner_conjuncts(select)
    if parts is None or len(sources) < 2 or "" in names or len(set(names)) != len(names):
        return None
    if not all(_plain_table(s) or _derived(s) is not None for s in sources):
        return None
    source_nodes = list(sources)
    reads: dict[str, list[exp.Column]] = {name: [] for name in names}  # the select's own reads of each source
    nested: set[str] = set()  # sources a nested subquery reads (or may read)
    for column in select.find_all(exp.Column):
        if any(_under(column, source) for source in source_nodes):
            continue
        if isinstance(column.this, exp.Star) or column.args.get("db") or column.args.get("catalog"):
            return None
        table = column.table.lower()
        if _own(column, select):
            if table not in reads:
                return None
            reads[table].append(column)
        elif table in reads:
            nested.add(table)
        elif not table:
            inner = column.find_ancestor(exp.Select)
            while inner.find_ancestor(exp.Select) is not select:
                inner = inner.find_ancestor(exp.Select)
            if not _closed(inner, schema):
                return None  # a bare name in a subquery that may be a column of any source
    if any(isinstance(n, exp.Star) and _own(n, select) and not isinstance(n.parent, exp.Count) for n in select.find_all(exp.Star)):
        return None
    for index in range(len(sources) - 1, -1, -1):
        name = names[index]
        if name in nested:
            continue
        touching = [i for i, part in enumerate(parts) if any(_under(c, part) for c in reads[name])]
        links = [i for i in touching if any(c.table.lower() != name for c in parts[i].find_all(exp.Column) if _own(c, select))]
        if len(links) != 1:
            continue
        sides = _link(parts[links[0]], set(names))
        if sides is None:
            continue
        mine, other = sides if sides[0].table.lower() == name else sides[::-1]
        if mine.table.lower() != name:
            continue
        local = [i for i in touching if i not in links]
        # a condition on x with a subquery stays where it is, as a condition on y.d, if x.c is all it reads of x
        staying = [i for i in local if any(isinstance(n, (exp.Select, exp.Subquery)) for n in parts[i].walk())]
        local = [i for i in local if i not in staying]
        if any(isinstance(n, (exp.AggFunc, exp.Window, *_VOLATILE)) for i in local for n in parts[i].walk()):
            continue
        if any(isinstance(n, _VOLATILE) for i in staying for n in parts[i].walk()):
            continue
        elsewhere = [c for c in reads[name] if not any(_under(c, parts[i]) for i in [links[0]] + local)]
        if any(c.name.lower() != mine.name.lower() for c in elsewhere):
            continue
        if elsewhere:
            by_name = dict(zip(names, sources))
            kinds = [_declared_type(c, by_name[c.table.lower()], types or {}) for c in (mine, other)]
            if kinds[0] != kinds[1] or any(f in k for k in kinds if k for f in _FLOATING):
                continue
        inner = _derived(sources[index])
        if inner is not None and not local and len(inner.expressions) == 1 and inner.expressions[0].alias_or_name.lower() == mine.name.lower() and not inner.find(exp.Star):
            body = inner.copy()  # a derived table that returns just the matched column is the subquery itself
        else:
            body = exp.Select(expressions=[mine.copy()])
            body.set(FROM_KEY, exp.From(this=sources[index].copy()))
            condition = _and_all([parts[i].copy() for i in local])
            if condition is not None:
                body.set("where", exp.Where(this=condition))
        membership = exp.In(this=other.copy(), query=exp.Subquery(this=body))
        # the other reads of x.c (select list, GROUP BY, HAVING, ORDER BY, conditions that stay) are re-pointed in a copy, found by position
        marked = select.copy()
        position = {id(c): i for i, c in enumerate(select.find_all(exp.Column))}
        copies = list(marked.find_all(exp.Column))
        staying_copies = {i: copies[position[id(next(c for c in reads[name] if _under(c, parts[i])))]] for i in staying}
        for i, column in staying_copies.items():  # the conjunct's root in the copy, found from one of its columns
            depth, node = 0, next(c for c in reads[name] if _under(c, parts[i]))
            while node is not parts[i]:
                depth, node = depth + 1, node.parent
            for _ in range(depth):
                column = column.parent
            staying_copies[i] = column
        for column in elsewhere:
            target = copies[position[id(column)]]
            item = target.parent is marked and target.arg_key == "expressions"
            target.replace(exp.alias_(other.copy(), target.name) if item else other.copy())
        kept = [staying_copies[i] if i in staying_copies else parts[i].copy() for i in range(len(parts)) if i not in [links[0]] + local]
        result = exp.Select(expressions=[item.copy() for item in marked.expressions])
        for clause in ("distinct", "group", "having", "order"):
            if marked.args.get(clause) is not None:
                result.set(clause, marked.args[clause].copy())
        remaining = [s.copy() for i, s in enumerate(sources) if i != index]
        result.set(FROM_KEY, exp.From(this=remaining[0]))
        if remaining[1:]:
            result.set("joins", [exp.Join(this=source) for source in remaining[1:]])
        result.set("where", exp.Where(this=_and_all(kept + [membership])))
        if not _reads_only_keys(result, schema):
            continue  # with x.c read as y.d, a column outside the aggregates is still not a GROUP BY column
        return result
    return None
