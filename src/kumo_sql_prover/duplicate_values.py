"""The values that occur more than once in a table, written as a join with itself or as a correlated count.

    SELECT DISTINCT a.email FROM person AS a JOIN person AS b ON a.email = b.email AND a.id <> b.id
    SELECT DISTINCT p.email FROM person AS p WHERE 1 < (SELECT COUNT(*) FROM person AS q WHERE q.email = p.email)

both list each email that two or more rows hold, as does

    SELECT email FROM person WHERE email IS NOT NULL GROUP BY email HAVING COUNT(*) > 1

which is the form this module writes (the prover reads grouped counts well, and the other two spellings
only through a join or a scalar subquery).

**Why they are the same.**

* *The self join.* ``a.id <> b.id`` on a declared key ``id`` says that ``a`` and ``b`` are different rows, and
  ``a.id < b.id`` that ``a`` is the earlier of two different rows. A value ``v`` of the equated columns (none
  NULL: the equality rejects a NULL) is output when two different rows hold it, which is when
  ``COUNT(*) > 1`` over the rows that hold it. ``DISTINCT`` removes the repeats the join makes, so the output is one
  row per such value. The select list may read the equated columns from either side.
* *The correlated count.* ``COUNT(*)`` of the rows that equal the outer row on the columns ``C`` is the size of the group
  of the outer row's value; ``COUNT(c)`` for a column of ``C`` is the same, because the equality is only true
  for a non-NULL ``c``. A NULL value in ``C`` makes the subquery count 0, and the outer row is dropped by any
  threshold of 1 or more; a threshold of 0 or less is not read. ``DISTINCT`` makes one row per value.

**What is read.** A ``SELECT DISTINCT`` of plain columns that are exactly the equated columns ``C`` (as a set),
over one table (the count) or two aliases of one base table (the join), with nothing else in the ``WHERE``, no
grouping and no limit; the join is inner and has the one inequality on a single-column declared key (NOT NULL);
the count compares ``COUNT(*)`` (or ``COUNT(c)``) with an integer, ``> n`` or ``>= n`` (also written the other way round),
with ``n`` of 1 or more after ``>=`` is turned into ``> n - 1``, and the subquery's ``WHERE`` is only the equalities.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key, _unwrap
from .ast_utils import conjuncts, plain_distinct, select_sources

_SELECT_ARGS = frozenset({"expressions", "from", "from_", "joins", "where", "distinct", "hint", "comments", "kind", "operation_modifiers"})
_INEQUALITIES = {exp.NEQ: "neq", exp.LT: "lt", exp.GT: "gt"}


def rewrite_duplicate_values(select: exp.Select, schema: dict[str, list[str]] | None, keys: dict[str, list[tuple[str, ...]]] | None, not_null: dict[str, frozenset[str]] | None) -> exp.Expression | None:
    """The select rewritten as described in the module docstring, else None."""

    if not schema or select.args.get("group") is not None or not plain_distinct(select):
        return None
    if any(value for key, value in select.args.items() if key not in _SELECT_ARGS):
        return None
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    return _self_join(select, known, keys or {}, not_null or {}) or _counted(select, known)


def _column(node: exp.Expression) -> exp.Column | None:
    node = _unwrap(node)
    return node if isinstance(node, exp.Column) and not isinstance(node.this, exp.Star) and not node.args.get("db") and not node.args.get("catalog") else None


def _names(select: exp.Select, table_aliases: set[str]) -> list[str] | None:
    """The output column names of a select of plain columns, else None."""

    out = []
    for item in select.expressions:
        column = _column(item.this if isinstance(item, exp.Alias) else item)
        if column is None or (column.table and column.table.lower() not in table_aliases):
            return None
        out.append(column.name.lower())
    return out


def _build(table: exp.Table, select: exp.Select, columns: list[str], threshold: int) -> exp.Select:
    """``SELECT c AS name, .. FROM t WHERE c IS NOT NULL .. GROUP BY c .. HAVING COUNT(*) > threshold``."""

    alias = table.alias_or_name
    out = select.copy()
    items = []
    for item in select.expressions:
        label = item.alias_or_name
        column = _column(item.this if isinstance(item, exp.Alias) else item)
        replacement = exp.column(column.name, table=alias)
        items.append(exp.alias_(replacement, label, quoted=False) if label.lower() != column.name.lower() or isinstance(item, exp.Alias) else replacement)
    out.set("expressions", items)
    out.set("distinct", None)
    out.set("joins", None)
    out.set("from_" if "from_" in out.args else "from", exp.From(this=table.copy()))
    out.set("where", exp.Where(this=exp.and_(*[exp.Not(this=exp.Is(this=exp.column(c, table=alias), expression=exp.Null())) for c in columns])))
    out.set("group", exp.Group(expressions=[exp.column(c, table=alias) for c in columns]))
    out.set("having", exp.Having(this=exp.GT(this=exp.Count(this=exp.Star()), expression=exp.Literal.number(threshold))))
    return out


def _self_join(select: exp.Select, known, keys, not_null) -> exp.Expression | None:
    sources = select_sources(select)
    joins = select.args.get("joins") or []
    if len(sources) != 2 or len(joins) != 1:
        return None
    join = joins[0]
    kind = join.args.get("kind")
    if join.args.get("side") or join.args.get("using") or join.args.get("method") or (kind.upper() if isinstance(kind, str) else kind) not in (None, "", "INNER", "CROSS"):
        return None
    a, b = sources
    columns = _base_table(a, known)
    if columns is None or _base_table(b, known) is None or _table_key(a) != _table_key(b) or a.alias_or_name.lower() == b.alias_or_name.lower():
        return None
    aliases = {a.alias_or_name.lower(), b.alias_or_name.lower()}
    if "" in aliases:
        return None
    parts = (conjuncts(join.args["on"]) if join.args.get("on") is not None else []) + (conjuncts(select.args["where"].this) if select.args.get("where") is not None else [])
    equated: list[str] = []
    ordering = None
    declared = {c.lower() for c in not_null.get(_table_key(a), ())}
    singles = {k[0].lower() for k in keys.get(_table_key(a), []) if len(k) == 1 and k[0].lower() in declared}
    for part in parts:
        node = _unwrap(part)
        if isinstance(node, exp.EQ):
            left, right = _column(node.this), _column(node.expression)
            if left is not None and right is not None and {left.table.lower(), right.table.lower()} == aliases and left.name.lower() == right.name.lower() and left.name.lower() in columns:
                if left.name.lower() in equated:
                    return None
                equated.append(left.name.lower())
                continue
            return None
        kind_of = _INEQUALITIES.get(type(node)) if not isinstance(node, exp.Not) else None
        if isinstance(node, exp.Not) and isinstance(_unwrap(node.this), exp.EQ):
            node, kind_of = _unwrap(node.this), "neq"
        if kind_of is None:
            return None
        left, right = _column(node.this), _column(node.expression)
        if left is None or right is None or {left.table.lower(), right.table.lower()} != aliases or left.name.lower() != right.name.lower() or left.name.lower() not in singles or ordering is not None:
            return None
        ordering = left.name.lower()
    if not equated or ordering is None:
        return None
    outputs = _names(select, aliases)
    if outputs is None or set(outputs) != set(equated):
        return None
    return _build(a, select, equated, 1)


def _counted(select: exp.Select, known) -> exp.Expression | None:
    sources = select_sources(select)
    if len(sources) != 1 or select.args.get("joins") or select.args.get("where") is None:
        return None
    outer = sources[0]
    columns = _base_table(outer, known)
    if columns is None or not outer.alias_or_name:
        return None
    alias = outer.alias_or_name.lower()
    where = _unwrap(select.args["where"].this)
    flipped = {exp.GT: exp.LT, exp.GTE: exp.LTE, exp.LT: exp.GT, exp.LTE: exp.GTE}
    if not isinstance(where, (exp.GT, exp.GTE, exp.LT, exp.LTE)):
        return None
    left, right = _unwrap(where.this), _unwrap(where.expression)
    comparison = type(where)
    if isinstance(left, exp.Literal):  # 1 < (SELECT ..) is (SELECT ..) > 1
        left, right, comparison = right, left, flipped[comparison]
    number = right
    if not isinstance(number, exp.Literal) or number.is_string or not number.name.isdigit():
        return None
    if comparison not in (exp.GT, exp.GTE):
        return None
    threshold = int(number.name) - (1 if comparison is exp.GTE else 0)
    if threshold < 1:
        return None
    if not isinstance(left, exp.Subquery) or not isinstance(left.this, exp.Select):
        return None
    inner = left.this
    if any(value for key, value in inner.args.items() if key not in ("expressions", "from", "from_", "where")) or len(inner.expressions) != 1:
        return None
    inner_sources = select_sources(inner)
    if len(inner_sources) != 1 or _base_table(inner_sources[0], known) is None or _table_key(inner_sources[0]) != _table_key(outer):
        return None
    inner_alias = inner_sources[0].alias_or_name.lower()
    if not inner_alias or inner_alias == alias or inner.args.get("where") is None:
        return None
    equated: list[str] = []
    for part in conjuncts(inner.args["where"].this):
        node = _unwrap(part)
        if not isinstance(node, exp.EQ):
            return None
        l, r = _column(node.this), _column(node.expression)
        if l is None or r is None:
            return None
        for inner_side, outer_side in ((l, r), (r, l)):
            # a bare column of the subquery's WHERE binds to the subquery's own table first
            if inner_side.table.lower() in (inner_alias, "") and outer_side.table.lower() == alias and inner_side.name.lower() == outer_side.name.lower() and inner_side.name.lower() in columns:
                equated.append(inner_side.name.lower())
                break
        else:
            return None
    if not equated or len(set(equated)) != len(equated):
        return None
    count = _unwrap(inner.expressions[0])
    if not isinstance(count, exp.Count) or isinstance(count.this, exp.Distinct):
        return None
    if not isinstance(count.this, exp.Star):
        argument = _column(count.this)
        if argument is None or (argument.table and argument.table.lower() != inner_alias) or argument.name.lower() not in equated:
            return None  # counting a column the equality does not make non-NULL
    outputs = _names(select, {alias})
    if outputs is None or set(outputs) != set(equated):
        return None
    return _build(outer, select, equated, threshold)
