"""A select that aggregates without ``GROUP BY`` returns exactly one row: three spellings that rest on it.

    SELECT c.id FROM (SELECT id, COUNT(DISTINCT k) AS n FROM customer GROUP BY id) AS c,
                     (SELECT COUNT(*) AS total FROM product) AS p
    WHERE c.n = p.total

compares each customer's count with the number of products, as does

    SELECT c.id FROM (SELECT id, COUNT(DISTINCT k) AS n FROM customer GROUP BY id) AS c
    WHERE c.n = (SELECT COUNT(*) FROM product)

and as does ``.. HAVING COUNT(DISTINCT k) IN (SELECT COUNT(*) FROM product)``. The prover reads the scalar
subquery; it did not read the membership test at all and it kept the joined one-row table as a join.

**The one fact.** A select with no ``GROUP BY`` whose outputs are built from aggregates only (``COUNT``, ``SUM``,
``MIN``, ``MAX``, ``AVG``, constants and operators around them) returns exactly one row, whatever its ``FROM``
and ``WHERE`` keep, even nothing. Call it a one-row select. It is read as one only when it has no ``HAVING``,
``LIMIT``, ``OFFSET``, ``ORDER BY``, ``QUALIFY``, window, nested query in its outputs, or a function that changes
from call to call, when no column in its outputs stands outside an aggregate, and when every source's columns
are known and each aggregate that reads columns reads one of them (an aggregate over outer columns alone belongs
to the query outside, and the select then returns a row per input row).

**What is rewritten, and why each is the same query.**

1. ``x IN (one-row select with one output)`` is ``x = (that select)``, anywhere. A membership test in a
   one-element list ``{r}`` is by definition ``x = r``: TRUE when equal, NULL when ``x`` or ``r`` is NULL, FALSE
   otherwise. The two expressions agree in all three truth values, so the rewrite also holds under ``NOT``
   (``NOT IN`` becomes ``NOT (x = ..)``), in a select list and in ``HAVING``.
   ``x = ALL (one-row select)`` and ``x = ANY (..)`` (``SOME``), with any of the six comparisons, are the
   comparison with that select for the same reason: over one value "every" and "some" are that value's own
   comparison, NULL included.
2. ``SELECT DISTINCT`` in a one-row select is dropped: one row has no duplicates.
3. A one-row select used as a derived table in a select whose joins are all inner or cross joins is removed
   from the ``FROM``, and every read of one of its outputs becomes the scalar subquery computing that output.
   Each row of the other sources was paired with the single row, so the rows and their multiplicity are
   unchanged; with inner joins only, an ``ON`` condition filters exactly like a ``WHERE`` condition, so all
   ``ON`` parts move to the ``WHERE``. The value read is the same: the scalar subquery aggregates the same rows
   of the same tables. (A derived table cannot read the select around it, so the subquery means the same
   wherever it is put.)

**What is refused in 3.** An outer, ``USING`` or ``NATURAL`` join anywhere in the select; a select with no other
source; a star; a derived table with a column list, or outputs that are unnamed or share a name; a read of its
outputs in ``GROUP BY``, ``ORDER BY`` or a window; a bare name that another source may also have (every other
source's columns must be known, from the supplied schema or from a derived table's listed outputs) or that is
also an alias of the select; a nested query of the select that names the derived table or one of its outputs.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _unwrap
from .ast_utils import FROM_KEY, conjuncts, select_sources
from .clause_alias_reads import _volatile

_ONE_ROW_ARGS = frozenset({"expressions", "from", "from_", "joins", "where", "distinct", "comments"})
_AGGREGATES = (exp.Count, exp.Sum, exp.Min, exp.Max, exp.Avg)
_JOIN_ARGS = frozenset({"this", "on", "kind", "comments"})
_COMPARISONS = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE)


def read_one_row_aggregates(tree: exp.Expression, schema: dict[str, list[str]] | None) -> exp.Expression:
    """Rewrite, in place, the three spellings described in the module docstring."""

    if not any(True for _ in tree.find_all(exp.AggFunc)):
        return tree
    known = {name.lower(): [c.lower() for c in columns] for name, columns in (schema or {}).items()}
    for node in list(tree.find_all(exp.In)):
        query = node.args.get("query")
        if query is None or node.expressions or node.args.get("unnest") is not None or node.args.get("field") is not None:
            continue
        inner = query.this if isinstance(query, exp.Subquery) else query
        if isinstance(query, exp.Subquery) and any(value for key, value in query.args.items() if key not in ("this", "comments")):
            continue
        if isinstance(_unwrap(node.this), exp.Tuple) or not _one_row(inner, known) or len(inner.expressions) != 1:
            continue
        node.replace(exp.EQ(this=node.this.copy(), expression=exp.Subquery(this=inner.copy())))
    for node in list(tree.find_all(exp.All, exp.Any)):
        inner = node.this.this if isinstance(node.this, exp.Subquery) else node.this
        if isinstance(node.this, exp.Subquery) and any(value for key, value in node.this.args.items() if key not in ("this", "comments")):
            continue
        parent = node.parent
        if type(parent) not in _COMPARISONS or parent.expression is not node or isinstance(_unwrap(parent.this), exp.Tuple):
            continue
        if _one_row(inner, known) and len(inner.expressions) == 1:
            node.replace(exp.Subquery(this=inner.copy()))
    for select in list(tree.find_all(exp.Select)):
        if select.args.get("distinct") is not None and _one_row(select, known):
            select.set("distinct", None)
    for select in list(tree.find_all(exp.Select))[::-1]:
        for _ in range(4):  # one derived table at a time
            if not _inline_source(select, known):
                break
    return tree


def _one_row(select: exp.Expression, known: dict[str, list[str]]) -> bool:
    """Whether ``select`` aggregates without ``GROUP BY`` and so returns exactly one row (module docstring)."""

    if not isinstance(select, exp.Select) or any(value for key, value in select.args.items() if key not in _ONE_ROW_ARGS):
        return False
    distinct = select.args.get("distinct")
    if distinct is not None and (distinct.args.get("on") or distinct.expressions):
        return False
    if not select.expressions:
        return False
    # the names its own sources answer to: an aggregate over outer columns alone belongs to the query outside
    qualifiers: set[str] = set()
    columns: set[str] = set()
    for source in select_sources(select):
        offered = _output_names(source, known)
        if offered is None or not source.alias_or_name:
            return False
        qualifiers.add(source.alias_or_name.lower())
        columns |= set(offered)
    aggregated = False
    for item in select.expressions:
        value = item.this if isinstance(item, exp.Alias) else item
        if _volatile(value):
            return False
        for node in value.walk():
            if isinstance(node, (exp.Window, exp.Query, exp.Subquery, exp.Lambda)):
                return False
            if isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count):
                return False
            if isinstance(node, exp.AggFunc):
                if not isinstance(node, _AGGREGATES):
                    return False
                read = list(node.find_all(exp.Column))
                if read and not any((c.table.lower() in qualifiers) if c.table else (c.name.lower() in columns) for c in read):
                    return False
                aggregated = True
            elif isinstance(node, exp.Column) and not _aggregated(node, value):
                return False
    return aggregated


def _aggregated(node: exp.Expression, top: exp.Expression) -> bool:
    """Whether an aggregate call lies between ``node`` and ``top`` (``top`` itself counts)."""

    while node is not top and node.parent is not None:
        node = node.parent
        if isinstance(node, exp.AggFunc):
            return True
    return False


def _output_names(source: exp.Expression, known: dict[str, list[str]]) -> list[str] | None:
    """The column names a source of a select offers, or None when they are not known."""

    columns = _base_table(source, known)
    if columns is not None:
        return columns
    if isinstance(source, exp.Subquery) and isinstance(source.this, exp.Select):
        alias = source.args.get("alias")
        if alias is not None and alias.args.get("columns"):
            return None
        names = []
        for item in source.this.expressions:
            if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)) or not item.alias_or_name:
                return None
            names.append(item.alias_or_name.lower())
        return names
    return None


def _inside(node: exp.Expression, ancestor: exp.Expression) -> bool:
    parent = node.parent
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.parent
    return False


def _inline_source(select: exp.Select, known: dict[str, list[str]]) -> bool:
    sources = select_sources(select)
    joins = select.args.get("joins") or []
    if len(sources) < 2 or select.args.get("laterals"):
        return False
    for join in joins:
        if any(value for key, value in join.args.items() if key not in _JOIN_ARGS) or (join.kind or "").upper() not in ("", "INNER", "CROSS"):
            return False
    names = [source.alias_or_name.lower() for source in sources]
    for source in sources:
        if not isinstance(source, exp.Subquery) or not _one_row(source.this, known):
            continue
        alias = source.args.get("alias")
        if any(value for key, value in source.args.items() if key not in ("this", "alias", "comments")) or (alias is not None and alias.args.get("columns")):
            continue
        own = source.alias.lower()
        if own and names.count(own) != 1:
            continue
        outputs: dict[str, exp.Expression] = {}
        for item in source.this.expressions:
            if not isinstance(item, exp.Alias) or not item.alias or item.alias.lower() in outputs:
                outputs = {}
                break
            outputs[item.alias.lower()] = item.this
        if not outputs:
            continue
        others: set[str] | None = set()
        for other in sources:
            if other is source:
                continue
            columns = _output_names(other, known)
            if columns is None:
                others = None
                break
            others |= set(columns)
        reads = _reads(select, sources, source, own, outputs, others)
        if reads is None:
            continue
        # --- rewrite ---
        for column in reads:
            scalar = source.this.copy()
            scalar.set("distinct", None)
            scalar.set("expressions", [outputs[column.name.lower()].copy()])
            replacement: exp.Expression = exp.Subquery(this=scalar)
            if isinstance(column.parent, exp.Select) and column.arg_key == "expressions":
                replacement = exp.alias_(replacement, column.name)
            column.replace(replacement)
        where = select.args.get("where")
        parts = [part for join in joins if join.args.get("on") is not None for part in conjuncts(join.args["on"])]
        parts += conjuncts(where.this) if where is not None else []
        parts = [exp.Paren(this=part.copy()) if isinstance(part, exp.Or) else part.copy() for part in parts]
        remaining = [other.copy() for other in sources if other is not source]
        select.set(FROM_KEY, exp.From(this=remaining[0]))
        select.set("joins", [exp.Join(this=other) for other in remaining[1:]] or None)
        if parts:
            condition = parts[0]
            for part in parts[1:]:
                condition = exp.And(this=condition, expression=part)
            select.set("where", exp.Where(this=condition))
        else:
            select.set("where", None)
        return True
    return False


def _reads(select: exp.Select, sources: list, source: exp.Subquery, own: str, outputs: dict, others: set[str] | None) -> list[exp.Column] | None:
    """The columns of ``select`` that read ``source``, or None when some read cannot be placed or moved."""

    reads: list[exp.Column] = []
    aliases = {item.alias.lower() for item in select.expressions if isinstance(item, exp.Alias) and item.alias}
    for node in select.walk():
        if isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count):
            if not any(_inside(node, other) for other in sources):
                return None
        if not isinstance(node, exp.Column) or any(node is other or _inside(node, other) for other in sources):
            continue
        qualifier, name = node.table.lower(), node.name.lower()
        if isinstance(node.this, exp.Star):
            return None
        mine = (bool(own) and qualifier == own) if qualifier else name in outputs
        if not mine:
            continue
        if node.find_ancestor(exp.Select) is not select:
            return None  # read from a nested query
        if name not in outputs or node.args.get("db") or node.args.get("catalog"):
            return None
        if not qualifier and (others is None or name in others or name in aliases):
            return None  # another source, or an alias of this select, may answer to the name
        if node.find_ancestor(exp.Group, exp.Order, exp.Window) is not None:
            return None
        reads.append(node)
    return reads
