"""A ``LEFT JOIN`` whose padded rows no group can be kept for is an inner join.

**The rule.** In a grouped select whose last join is ``.. LEFT JOIN p ON c``, the join is made inner when

* a conjunct of ``HAVING`` puts a lower bound of 1 or more on ``COUNT(p.x)`` or ``COUNT(DISTINCT p.x)`` for a
  column ``x`` of the padded source ``p`` (``> 0``, ``>= 1``, ``<> 0``, ``= 3``, the count read directly or
  through its select alias), and
* every aggregate of the select is ``COUNT``, ``SUM``, ``AVG``, ``MIN`` or ``MAX`` (with or without
  ``DISTINCT``) of a column of ``p``, and
* outside the aggregates the select list, ``HAVING`` and ``ORDER BY`` read only ``GROUP BY`` columns.

    SELECT a.k, COUNT(DISTINCT b.v) FROM t AS a LEFT JOIN u AS b ON a.k = b.g GROUP BY a.k HAVING COUNT(b.v) > 0
    SELECT a.k, COUNT(DISTINCT b.v) FROM t AS a      JOIN u AS b ON a.k = b.g GROUP BY a.k HAVING COUNT(b.v) > 0

**Why it is sound.** The left join's rows are the inner join's rows plus, for each left row without a match,
one row with NULL in every column of ``p`` (a padded row). ``WHERE`` is applied row by row and treats the inner
join's rows the same in both forms. Fix a group key and compare the two groups: the inner form's group is the
left form's group without its padded rows. In a padded row ``p.x`` is NULL, and each of the five aggregates
skips NULL arguments, so every aggregate has the same value over both groups whenever the inner form has the
group at all. The bounded count is one of these aggregates: it is at least 1 only if some row of the group has a
non-NULL ``p.x``, and that row is a row of the inner join. So a group the left form keeps exists in the inner
form with the same key and the same aggregates, hence the same ``HAVING`` value and the same output row; a
group the left form drops is either absent from the inner form (it held only padded rows) or dropped there by
the same ``HAVING``; and every group of the inner form is a group of the left form. Group keys may be columns of
either side: padded rows may share a group with matched rows whose key is NULL, and the argument is per group
either way.

**What is read.** A select with a plain ``GROUP BY`` of columns, whose joins before the last are plain inner
joins and whose last join is a ``LEFT [OUTER] JOIN .. ON`` of a named table or derived table; columns qualified
with the source names. The select alias of a count is read in ``HAVING`` only when the schema lists every
source's columns and none has that name (as ``having_count_rules`` reads it).

**What is refused.** ``COUNT(*)`` or an aggregate of a left-side column or of an expression (the padded rows
count), an aggregate with ``FILTER`` or other modifiers, window functions, ``LIMIT``, ``ROLLUP``/``CUBE``/grouping
sets, a select without ``GROUP BY``, a subquery in the select list, ``HAVING`` or ``ORDER BY``, a column outside
the aggregates that is not a ``GROUP BY`` column, a bound that is only an upper bound or a ``<> n`` other than
``<> 0``, a bound under ``OR`` or ``NOT``, ``RIGHT`` and ``FULL`` joins, and a ``LEFT JOIN`` that is not the last
join (later joins could read the padded rows).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import conjuncts, extended_grouping
from .decorrelation_rules import _sources
from .having_count_rules import _count_aliases, _natural
from .semijoin_shapes import _derived, _only, _own, _plain_column, _plain_table

_CLAUSES = frozenset({"expressions", "from_", "from", "joins", "where", "group", "having", "order", "distinct"})
_NULL_SKIPPING = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)
_FLIP = {exp.GTE: exp.LTE, exp.GT: exp.LT, exp.LTE: exp.GTE, exp.LT: exp.GT, exp.EQ: exp.EQ, exp.NEQ: exp.NEQ}


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def counted_at_least_once(test: exp.Expression, aliases: dict[str, exp.Expression]) -> exp.Count | None:
    """The ``COUNT`` that ``test`` bounds from below by 1 or more (``> 0``, ``>= 1``, ``= 2``, ``<> 0``), else ``None``."""

    test = _unwrap(test)
    kind = type(test)
    if kind not in _FLIP:
        return None

    def counted(node: exp.Expression) -> exp.Count | None:
        node = _unwrap(node)
        if type(node) is exp.Column and not node.table and node.name.lower() in aliases:
            node = _unwrap(aliases[node.name.lower()])
        return node if type(node) is exp.Count else None

    count, number = counted(test.this), _natural(test.expression)
    if count is None or number is None:
        count, number, kind = counted(test.expression), _natural(test.this), _FLIP[kind]
    if count is None or number is None:
        return None
    bounded = (kind is exp.GT and number >= 0) or (kind in (exp.GTE, exp.EQ) and number >= 1) or (kind is exp.NEQ and number == 0)
    return count if bounded else None


def _argument(node: exp.Expression) -> exp.Column | None:
    """The one plain column an unmodified aggregate reads, with or without ``DISTINCT``."""

    if isinstance(node.parent, exp.Filter) or any(value for key, value in node.args.items() if key not in ("this", "big_int")):
        return None
    argument = node.this
    if isinstance(argument, exp.Distinct):
        if argument.args.get("on") or len(argument.expressions) != 1:
            return None
        argument = argument.expressions[0]
    return _plain_column(argument) if isinstance(argument, exp.Expression) else None


def counted_outer_join_to_inner(select: exp.Select, schema: dict | None) -> exp.Expression | None:
    """``select`` (changed in place) with its trailing ``LEFT JOIN`` made inner, else ``None``."""

    joins = select.args.get("joins") or []
    group, having = select.args.get("group"), select.args.get("having")
    if not joins or group is None or having is None or not _only(select, _CLAUSES):
        return None
    if not group.expressions or extended_grouping(group) or any(value for key, value in group.args.items() if key != "expressions"):
        return None
    if select.args.get("distinct") is not None and select.args["distinct"].args.get("on"):
        return None
    last = joins[-1]
    if (last.side or "").upper() != "LEFT" or (last.kind or "").upper() not in ("", "OUTER") or last.args.get("on") is None:
        return None
    if any(value for key, value in last.args.items() if key not in ("this", "on", "kind", "side")):
        return None
    for join in joins[:-1]:
        if join.side or (join.kind or "").upper() not in ("", "INNER", "CROSS") or any(value for key, value in join.args.items() if key not in ("this", "on", "kind")):
            return None
    sources = _sources(select)
    names = [(s.alias_or_name or "").lower() for s in sources]
    if "" in names or len(set(names)) != len(names) or not all(_plain_table(s) or _derived(s) is not None for s in sources):
        return None
    padded = names[-1]
    keys = set()
    for item in group.expressions:
        column = _plain_column(item)
        if column is None or column.table.lower() not in names:
            return None
        keys.add((column.table.lower(), column.name.lower()))
    aliases = _count_aliases(select, schema)
    found = False
    for clause in ("expressions", "having", "order"):
        value = select.args.get(clause)
        for part in value if isinstance(value, list) else [value] if value is not None else []:
            for node in part.walk():
                if isinstance(node, (exp.Select, exp.Subquery, exp.Window, exp.Star)):
                    return None
                if isinstance(node, exp.AggFunc):
                    argument = _argument(node) if isinstance(node, _NULL_SKIPPING) else None
                    if argument is None or argument.table.lower() != padded:
                        return None
                elif type(node) is exp.Column and node.find_ancestor(exp.AggFunc) is None:
                    if clause == "having" and not node.table and node.name.lower() in aliases:
                        continue  # the select alias of a COUNT, whose own aggregate is checked in the select list
                    if (node.table.lower(), node.name.lower()) not in keys:
                        return None
    for test in conjuncts(having.this):
        if counted_at_least_once(test, aliases) is not None:
            found = True
    if not found or not all(_own(n, select) for n in having.find_all(exp.AggFunc)):
        return None
    last.set("side", None)
    last.set("kind", None)
    return select
