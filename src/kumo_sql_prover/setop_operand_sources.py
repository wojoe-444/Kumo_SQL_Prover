"""An operand of ``INTERSECT`` or ``EXCEPT`` that joins a filtered derived table reads the tables themselves.

    SELECT DISTINCT c.n, d.x FROM s JOIN c ON s.ck = c.k JOIN (SELECT k, x FROM t WHERE x > 3) AS d ON s.dk = d.k
    SELECT DISTINCT c.n, t.x FROM s, c, t WHERE s.ck = c.k AND s.dk = t.k AND t.x > 3

The normalizer reads a duplicate-removing ``INTERSECT`` or ``EXCEPT`` as an ``EXISTS`` test of its second
operand (``set_operation_to_exists``), but only when both operands select from tables alone. An operand that
filters one of its tables in a derived table first (a ``WITH`` step per table is how a rewrite usually spells
it) kept its set operation, so the two spellings of one query met the prover in two forms and did not compare.

A derived table that only projects, filters and inner-joins its own sources returns one row for each
combination of their rows that passes its conditions, so the select around it sees the same rows, in the same
number, when it joins those sources itself and takes the conditions into its ``WHERE``. That is the fold the
normalizer already makes for a select that groups (``_merge_spj_source``); this rule asks for the same fold,
with the same checks, in an operand that does not group.

Read only when:

* the select is an operand of ``INTERSECT`` or ``EXCEPT`` (through parentheses, not through an alias);
* it does not group, aggregate or use a window, and has no ``ORDER BY``, ``LIMIT``, ``OFFSET``, ``QUALIFY``,
  ``WITH`` or ``DISTINCT ON`` (which keeps the first row in an order). A plain ``DISTINCT`` stays on the
  select: the rows below it are unchanged;
* every condition of ``_merge_spj_source`` holds: inner joins only on both levels, a derived table whose
  outputs are plain expressions with distinct names, no subquery reading the derived table by its alias.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import distinct_on, select_sources


def _set_test_above(select: exp.Select) -> exp.Expression | None:
    """The ``INTERSECT`` or ``EXCEPT`` that ``select`` is an operand of (through parentheses, not an alias)."""

    node = select.parent
    while isinstance(node, (exp.Subquery, exp.Paren)) and not node.alias and not any(
        value for key, value in node.args.items() if key != "this" and value
    ):
        node = node.parent
    return node if isinstance(node, (exp.Intersect, exp.Except)) else None


_NOT_A_SET = ("limit", "offset", "order", "qualify", "windows", "with_", "with", "pivots", "sample", "into", "locks")


def _read_as_set(select: exp.Select, above: exp.Expression) -> exp.Expression | None:
    """``DISTINCT`` on an operand of a duplicate-removing set operation the normalizer keeps."""

    if above.args.get("distinct") is not True or select.args.get("distinct") is not None:
        return None
    if any(select.args.get(key) for key in _NOT_A_SET) or any(isinstance(n, exp.Window) for n in select.walk()):
        return None  # DISTINCT before a LIMIT or beside a window is another query
    if any(isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)) for item in select.expressions):
        return None
    from .setop_rules import merge_same_source, set_operation_to_exists

    if merge_same_source(above) is not None or set_operation_to_exists(above) is not None:
        return None  # read as a filter or an EXISTS test on this pass: both operands as they are
    out = select.copy()
    out.set("distinct", exp.Distinct())
    return out


def merge_operand_sources(select: exp.Select) -> exp.Expression | None:
    """Fold the filtering derived tables of an ``INTERSECT`` / ``EXCEPT`` operand into it, then read it as a set."""

    above = _set_test_above(select)
    if above is None or distinct_on(select):
        return None
    if select.args.get("group") or select.args.get("having") or any(
        node.find_ancestor(exp.Select) is select for node in select.find_all(exp.AggFunc)
    ):
        return None  # a grouping select is the existing rule's
    if any(isinstance(source, exp.Subquery) for source in select_sources(select)):
        from .algebraic_equivalence import _merge_spj_source

        body = select.copy()
        distinct = body.args.get("distinct")
        body.set("distinct", None)
        merged = _merge_spj_source(body, plain=True)
        if merged is not None:
            merged.set("distinct", distinct)
            return merged
    return _read_as_set(select, above)
