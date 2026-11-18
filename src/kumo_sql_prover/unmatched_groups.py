"""A ``LEFT JOIN`` grouped by the kept table's key and filtered to a zero count, read as ``NOT EXISTS``.

    SELECT c.name FROM customers AS c LEFT JOIN orders AS o ON c.id = o.customer_id
    GROUP BY c.id HAVING COUNT(o.id) = 0

lists the customers without an order, as does

    SELECT c.name FROM customers AS c WHERE NOT EXISTS (SELECT 1 FROM orders AS o WHERE c.id = o.customer_id)

The prover reads an aggregate over an outer join as one opaque relation, so the first spelling matched nothing
but itself. This rule writes it as the second, which the prover compares with every other anti-join.

**Why the two are the same.** Let ``a`` be the kept table, ``b`` the joined one and ``k`` a declared key of ``a``
whose columns are declared NOT NULL and all appear in the ``GROUP BY``.

* The outer join returns, for each row ``r`` of ``a``, one row per row of ``b`` on which the ``ON`` condition is
  TRUE, or a single row padded with NULLs when there is none. A ``WHERE`` that reads only ``a`` keeps or drops
  all rows of one ``r`` together.
* No two rows of ``a`` share ``k`` and no ``k`` is NULL, so grouping by columns of ``a`` that include ``k`` puts
  the rows of each ``r`` in a group of their own: one group per kept row of ``a``.
* The counted column ``b.c`` is not NULL on every matched row: it is declared NOT NULL, or the ``ON`` condition
  has a top-level ``AND`` part ``b.c = ..``, which is only TRUE for a ``b.c`` that is not NULL. On the padded
  row it is NULL. So ``COUNT(b.c)`` is the number of matches of ``r``, and ``COUNT(b.c) = 0`` holds exactly when
  no row of ``b`` makes the ``ON`` condition TRUE for ``r``: ``NOT EXISTS (SELECT 1 FROM b WHERE on)``.
* Each kept group is one row of ``a`` and outputs columns of ``a`` (one value in the group, the row's own), so
  the result has one row per unmatched ``r`` with the same values.
* ``NOT EXISTS`` over an empty ``b`` keeps every row of ``a``, as the outer join pads every row; an empty ``a``
  gives no row either way.

**What is read.** A select with exactly two sources, base tables with supplied columns under different names,
joined by one ``LEFT JOIN`` (the kept table first) or ``RIGHT JOIN`` (the kept table second) with an ``ON``
condition that reads only the two tables, each column qualified or a bare name only one of them has; an
optional ``WHERE`` that reads only the kept table; a ``GROUP BY`` of plain columns of the kept table covering a
declared NOT NULL key; a ``HAVING`` that is exactly ``COUNT(b.c) = 0``, ``< 1`` or ``<= 0`` (either way round);
outputs that are plain columns of the kept table.

**What is refused.** ``COUNT(*)`` (the padded row counts), a counted column that may be NULL on a match, a
grouping that does not cover a declared NOT NULL key (two rows of the kept table could share a group), any
other ``HAVING`` part, an aggregate or expression in the outputs, ``DISTINCT``, ``ORDER BY``, ``LIMIT``, a
nested query or a volatile function anywhere in the select, a bare ``GROUP BY`` name that is also the alias of
another output, an inner or full join, ``USING``, a third source, a query that still holds a ``WITH``.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key, _unwrap
from .ast_utils import FROM_KEY, conjuncts, extended_grouping, select_sources
from .clause_alias_reads import _volatile

_SELECT_ARGS = frozenset({"expressions", "from", "from_", "joins", "where", "group", "having", "comments"})
_JOIN_ARGS = frozenset({"this", "on", "side", "kind", "comments"})


def read_unmatched_groups(
    tree: exp.Expression,
    schema: dict[str, list[str]] | None,
    not_null: dict[str, frozenset[str]] | None,
    keys: dict[str, list[tuple[str, ...]]] | None,
) -> exp.Expression:
    """Rewrite, in place, each select described in the module docstring."""

    if not schema or not keys or not not_null or tree.find(exp.With) is not None:
        return tree
    known = {name.lower(): [c.lower() for c in columns] for name, columns in schema.items()}
    declared = {name.lower(): {c.lower() for c in columns} for name, columns in not_null.items()}
    unique = {name.lower(): [tuple(c.lower() for c in key) for key in table_keys if key] for name, table_keys in keys.items()}
    for select in list(tree.find_all(exp.Select))[::-1]:
        _rewrite(select, known, declared, unique)
    return tree


def _zero_count(test: exp.Expression) -> exp.Count | None:
    """The ``COUNT`` that ``test`` requires to be 0."""

    test = _unwrap(test)
    left, right = (_unwrap(test.this), _unwrap(test.expression)) if isinstance(test, exp.Binary) else (None, None)

    def number(node) -> int | None:
        return int(node.this) if isinstance(node, exp.Literal) and not node.is_string and node.this.isdigit() else None

    if type(left) is exp.Count and number(right) is not None:
        if (type(test), number(right)) in ((exp.EQ, 0), (exp.LT, 1), (exp.LTE, 0)):
            return left
    if type(right) is exp.Count and number(left) is not None:
        if (type(test), number(left)) in ((exp.EQ, 0), (exp.GT, 1), (exp.GTE, 0)):
            return right
    return None


def _rewrite(select: exp.Select, known, declared, unique) -> bool:
    if any(value for key, value in select.args.items() if key not in _SELECT_ARGS):
        return False
    group, having = select.args.get("group"), select.args.get("having")
    sources = select_sources(select)
    joins = select.args.get("joins") or []
    if group is None or having is None or len(sources) != 2 or len(joins) != 1:
        return False
    join = joins[0]
    side, kind = (join.side or "").upper(), (join.kind or "").upper()
    if any(value for key, value in join.args.items() if key not in _JOIN_ARGS) or side not in ("LEFT", "RIGHT") or kind not in ("", "OUTER"):
        return False
    on = join.args.get("on")
    if on is None:
        return False
    kept, joined = (sources[0], sources[1]) if side == "LEFT" else (sources[1], sources[0])
    kept_columns, joined_columns = _base_table(kept, known), _base_table(joined, known)
    if kept_columns is None or joined_columns is None:
        return False
    a, b = kept.alias_or_name.lower(), joined.alias_or_name.lower()
    if not a or not b or a == b:
        return False
    # nothing nested, no star, and every column placed on one of the two tables
    if _volatile(select):
        return False
    for node in select.walk():
        if node is not select and isinstance(node, (exp.Query, exp.Subquery, exp.Window, exp.Lambda)):
            return False
        if isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count):
            return False

    def owner(node: exp.Expression) -> tuple[str, str] | None:
        """``("a" or "b", column)`` for a column of the kept or of the joined table."""

        if not isinstance(node, exp.Column) or isinstance(node.this, exp.Star) or node.args.get("db") or node.args.get("catalog"):
            return None
        name = node.name.lower()
        if node.table:
            if node.table.lower() == a and name in kept_columns:
                return "a", name
            if node.table.lower() == b and name in joined_columns:
                return "b", name
            return None
        if (name in kept_columns) == (name in joined_columns):
            return None  # either table's, or neither's
        return ("a", name) if name in kept_columns else ("b", name)

    placed = {id(node): owner(node) for node in select.find_all(exp.Column)}
    if None in placed.values():
        return False
    where = select.args.get("where")
    if where is not None and any(placed[id(node)][0] != "a" for node in where.find_all(exp.Column)):
        return False
    if where is not None and where.find(exp.AggFunc) is not None or on.find(exp.AggFunc) is not None:
        return False
    # the grouping: plain columns of the kept table that cover a declared NOT NULL key
    if extended_grouping(group) or any(value for key, value in group.args.items() if key != "expressions"):
        return False
    grouped = [placed.get(id(_unwrap(item))) for item in group.expressions]
    if None in grouped or any(which != "a" for which, _ in grouped):
        return False
    for item in group.expressions:
        if _unwrap(item).table:
            continue
        for output in select.expressions:  # a bare name that an output's alias could capture
            value = _unwrap(output.this if isinstance(output, exp.Alias) else output)
            if output.alias_or_name.lower() == _unwrap(item).name.lower() and placed.get(id(value)) != placed[id(_unwrap(item))]:
                return False
    names = {name for _, name in grouped}
    table = _table_key(kept)
    facts = declared.get(table, set())
    if not any(set(key) <= names and set(key) <= facts for key in unique.get(table, ())):
        return False
    # a joined column that is not NULL on every matched row
    joined_facts = set(declared.get(_table_key(joined), set()))
    for part in conjuncts(on):
        if type(part) is exp.EQ:
            for operand in (_unwrap(part.this), _unwrap(part.expression)):
                if isinstance(operand, exp.Column) and placed[id(operand)][0] == "b":
                    joined_facts.add(placed[id(operand)][1])

    def match_count(node: exp.Expression) -> bool:
        if type(node) is not exp.Count or any(value for key, value in node.args.items() if key not in ("this", "big_int")):
            return False
        counted = _unwrap(node.this)
        return isinstance(counted, exp.Column) and placed[id(counted)][0] == "b" and placed[id(counted)][1] in joined_facts

    tested = _zero_count(having.this)
    if tested is None or not match_count(tested):
        return False
    for item in select.expressions:
        value = _unwrap(item.this if isinstance(item, exp.Alias) else item)
        if not isinstance(value, exp.Column) or placed[id(value)][0] != "a":
            return False
    # --- rewrite ---
    inner = exp.Select(expressions=[exp.Literal.number(1)])
    inner.set(FROM_KEY, exp.From(this=joined.copy()))
    inner.set("where", exp.Where(this=on.copy()))
    test: exp.Expression = exp.Not(this=exp.Exists(this=inner))
    if where is not None:
        kept_filter = where.this.copy()
        test = exp.And(this=exp.Paren(this=kept_filter) if isinstance(kept_filter, exp.Or) else kept_filter, expression=test)
    select.set(FROM_KEY, exp.From(this=kept.copy()))
    select.set("joins", None)
    select.set("where", exp.Where(this=test))
    select.set("group", None)
    select.set("having", None)
    return True
