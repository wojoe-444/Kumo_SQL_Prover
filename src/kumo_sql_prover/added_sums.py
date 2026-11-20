"""A sum of an addition is the addition of the sums, when no term is ever NULL.

    SELECT SUM(b.apples + IFNULL(c.apples, 0)) FROM boxes AS b LEFT JOIN chests AS c ON b.chest_id = c.chest_id
    SELECT SUM(b.apples) + SUM(IFNULL(c.apples, 0)) FROM boxes AS b LEFT JOIN chests AS c ON b.chest_id = c.chest_id
    SELECT SUM(b.apples) + IFNULL(SUM(c.apples), 0) FROM boxes AS b LEFT JOIN chests AS c ON b.chest_id = c.chest_id

are three spellings of one total (with ``b.apples`` declared NOT NULL). The prover compared the aggregates one by
one and found no partner for ``SUM(b.apples + ..)``. This module writes the first and the third as the second.

**Splitting a sum.** ``SUM(t1 + t2)`` adds ``t1 + t2`` over the rows on which that is not NULL. When neither term is
ever NULL, that is every row, and adding all the ``t1`` and all the ``t2`` gives the same total; over no rows both
spellings are NULL (``NULL + NULL``). When a term can be NULL the spellings differ: a row with ``t1`` NULL is skipped
whole by the first and still adds its ``t2`` in the second. So every term must be never NULL:

* a column declared NOT NULL of a table of this select that no outer join pads, or
* ``COALESCE(c, n)`` or ``IFNULL(c, n)`` of a column of this select with a whole-number literal ``n``.

The additions are regrouped, which is exact for whole numbers and fixed-point numbers and not for floating point, so
every column must have a declared type that is a whole number or fixed-point (``float_regrouping`` would refuse the
proof otherwise; this module does not fire at all). Every column must belong to this select: a sum of an enclosing
query's column is that query's aggregate, and splitting a sum that mixes the two would move a part of it there.

**A total read as 0, next to a sum that is NULL over no rows.** Under a ``GROUP BY``, ``COALESCE(SUM(c), 0)`` is
``SUM(COALESCE(c, 0))`` (``padded_sums``). Without a ``GROUP BY`` they differ over no rows: 0 against NULL. But as a
term of an addition whose other term is ``SUM(t)`` for a never-NULL ``t`` of the same select, the difference
cannot show: over no rows ``SUM(t)`` is NULL and so is the addition, whichever spelling stands beside it; over
some rows the two spellings are equal, as under a ``GROUP BY``. So in such an addition the module writes
``COALESCE(SUM(c), 0)`` as ``SUM(COALESCE(c, 0))``.

**What is read.** ``SUM`` with no ``DISTINCT`` and no ``FILTER``, outside a window, whose argument is an addition
of two or more terms as above (parentheses allowed); and, in a select without ``GROUP BY``, ``COALESCE(SUM(c), 0)``
or ``IFNULL(SUM(c), 0)`` (the literal 0 only) that is a term of an addition with a ``SUM`` of a never-NULL term. The
select reads tables and derived tables joined by inner, cross and left joins. A subtraction, a product, a term
that is a literal or a ``CASE`` (other rules read sums of those) and a column that cannot be placed leave the sum
alone.

A pre-pass of the eleventh hardest-cases batch (``shape_fallback.ELEVENTH_BATCH``). ``tests/test_added_sums.py``
replays both rewrites on generated tables with NULLs and with no rows.
"""

from __future__ import annotations

from sqlglot import exp

from . import smt_values
from .counted_presence import _Facts, _sources

_EXACT = (smt_values.INT64, smt_values.NUMERIC, smt_values.BIGNUMERIC)


def _unparen(node: exp.Expression | None) -> exp.Expression | None:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _terms(node: exp.Expression | None) -> list[exp.Expression]:
    node = _unparen(node)
    if isinstance(node, exp.Add):
        return _terms(node.this) + _terms(node.expression)
    return [node] if node is not None else []


def _zero_padded(node: exp.Expression | None) -> tuple[exp.Expression, exp.Literal] | None:
    """``(x, n)`` for ``COALESCE(x, n)`` or ``IFNULL(x, n)`` with a whole-number literal ``n``."""

    node = _unparen(node)
    if type(node) is not exp.Coalesce or len(node.expressions) != 1 or any(v for k, v in node.args.items() if k not in ("this", "expressions", "is_nvl", "is_null")):
        return None
    filler = _unparen(node.expressions[0])
    if not isinstance(filler, exp.Literal) or filler.is_string or not filler.this.isdigit():
        return None
    return node.this, filler


class _Reader:
    def __init__(self, select: exp.Select, facts: _Facts):
        self.select, self.facts = select, facts
        found = _sources(select)
        self.sources, self.real = found if found is not None else ({}, set())

    def _exact_column(self, node: exp.Expression | None, never_null: bool) -> bool:
        """A column of one of the select's tables with an exact declared type (and, if asked, never NULL here)."""

        owner = self.facts.owner(node, self.sources) if self.sources else None
        if owner is None:
            return False
        name, field = owner
        source = self.sources[name]
        if not isinstance(source, exp.Table):
            return False
        table = source.name.lower()
        if smt_values.type_class(self.facts.kinds.get(table, {}).get(field)) not in _EXACT or self.facts.kinds.get(table, {}).get(field) is None:
            return False
        return not never_null or (name in self.real and field in self.facts.declared.get(table, ()))

    def never_null(self, term: exp.Expression | None) -> bool:
        padded = _zero_padded(term)
        if padded is not None:
            return self._exact_column(padded[0], never_null=False)
        return self._exact_column(term, never_null=True)

    def plain_sum(self, node: exp.Expression | None) -> exp.Sum | None:
        node = _unparen(node)
        if type(node) is not exp.Sum or node.args.get("distinct") or isinstance(node.this, exp.Distinct) or any(v for k, v in node.args.items() if k != "this"):
            return None
        if node.find_ancestor(exp.Select) is not self.select or node.find_ancestor(exp.Window, exp.Filter) is not None:
            return None
        return node


def split_added_sums(tree: exp.Expression, schema: dict | None, not_null: dict | None, types: dict | None) -> tuple[exp.Expression, bool]:
    """``tree`` with sums of never-NULL additions split and zero-padded totals moved inside (module doc); whether it changed."""

    if not types or tree.find(exp.Sum) is None:
        return tree, False
    facts = _Facts(not_null or {}, types, schema)

    def plan(select: exp.Select) -> tuple[list[exp.Sum], list[exp.Coalesce]]:
        """The sums of ``select`` to split, and the padded totals to move inside."""

        reader = _Reader(select, facts)
        if not reader.sources:
            return [], []
        sums = [node for node in select.find_all(exp.Sum) if reader.plain_sum(node) is node]
        splits = [call for call in sums if len(_terms(call.this)) >= 2 and all(reader.never_null(term) for term in _terms(call.this))]
        moves: list[exp.Coalesce] = []
        if select.args.get("group") is None:
            for call in sums:
                top: exp.Expression = call
                while isinstance(top.parent, exp.Paren):
                    top = top.parent
                padded = top.parent
                if _zero_padded(padded) is None or _unparen(padded.this) is not call or _zero_padded(padded)[1].this != "0" or len(_terms(call.this)) != 1:
                    continue
                if not reader._exact_column(call.this, never_null=False):
                    continue
                outer: exp.Expression = padded
                while isinstance(outer.parent, (exp.Paren, exp.Add)):
                    outer = outer.parent
                if not isinstance(_unparen(outer), exp.Add):
                    continue
                anchors = [reader.plain_sum(term) for term in _terms(outer)]
                if any(anchor is not None and anchor is not call and all(reader.never_null(term) for term in _terms(anchor.this)) for anchor in anchors):
                    moves.append(padded)
        return splits, moves

    if not any(any(plan(select)) for select in tree.find_all(exp.Select)):
        return tree, False
    tree = tree.copy()
    changed = False
    for select in list(tree.find_all(exp.Select)):
        splits, moves = plan(select)
        for padded in moves:
            call = _unparen(padded.this)
            inner = exp.Coalesce(this=call.this.copy(), expressions=[exp.Literal.number(0)])
            padded.replace(exp.Sum(this=inner))
            changed = True
        for call in splits:
            if call.find_ancestor(exp.Select) is not select:
                continue
            added: exp.Expression | None = None
            for term in _terms(call.this):
                part = exp.Sum(this=term.copy())
                added = part if added is None else exp.Add(this=added, expression=part)
            call.replace(exp.Paren(this=added))
            changed = True
    return tree, changed
