"""A ``GROUP BY`` over an outer join that only outputs its keys is ``SELECT DISTINCT``.

    SELECT a.k, b.name FROM a LEFT JOIN b ON a.k = b.k GROUP BY a.k, b.name
    SELECT DISTINCT a.k, b.name FROM a LEFT JOIN b ON a.k = b.k

A select with a plain ``GROUP BY``, no aggregate and no ``HAVING`` returns one row per distinct tuple of its keys,
with NULL keys in one group. ``SELECT DISTINCT`` returns one row per distinct tuple of its outputs, with NULLs
equal. When the outputs are exactly the keys the two are the same rows: the same tuples, each once, whatever the
joins below produce (padded rows, duplicates), and no row at all over no joined rows (the select has at least one
key, so it is not the one-group ``GROUP BY ()``).

Why only over an outer join: the prover reads an aggregate over an outer join through a derived table of the
join's columns (``algebraic_equivalence._wrap_outer_join_aggregate``), which it keeps whole and compares by its
text. A key-only ``GROUP BY`` was wrapped the same way, so it never met the ``SELECT DISTINCT`` spelling of the
same query, nor a spelling with other duplicates below the join (Calcite's aggregate-join-transpose writes
``GROUP BY`` on both inputs of the join and again above it). ``SELECT DISTINCT`` over an outer join the prover
reads directly, branch by branch. Selects without an outer join already meet their ``DISTINCT`` spelling.

Conditions, all checked:

* the select's own joins include a ``LEFT``, ``RIGHT`` or ``FULL`` join;
* a plain ``GROUP BY`` list with at least one key: no ``ROLLUP``, ``CUBE``, ``GROUPING SETS``, ``GROUP BY ALL`` or
  ``()``, and no key that is a literal (a position, or a constant);
* no ``HAVING``, ``DISTINCT``, ``QUALIFY``, ``ORDER BY``, ``LIMIT``, ``OFFSET``, named window, and no aggregate, window
  function or ``GROUPING()`` call of the select itself;
* every output is, by its text, one of the keys, and every key is an output. ``SELECT a + 1 .. GROUP BY a`` (an
  output computed from a key can repeat) and ``SELECT a .. GROUP BY a, b`` (one row per ``b`` as well) are left
  alone; so are stars.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import extended_grouping
from .grouping_sets import is_grouping_call

_OUTER = ("LEFT", "RIGHT", "FULL")
_BANNED = ("having", "distinct", "qualify", "order", "limit", "offset", "windows", "with_", "with")


def key_only_group_to_distinct(select: exp.Select) -> exp.Select | None:
    """``select`` as ``SELECT DISTINCT`` when it is a key-only ``GROUP BY`` over an outer join (module doc), else None."""

    group = select.args.get("group")
    if group is None or not group.expressions or extended_grouping(group) or any(group.args.get(k) for k in ("all", "totals")):
        return None
    if any(select.args.get(key) for key in _BANNED):
        return None
    joins = select.args.get("joins") or []
    if not any((join.args.get("side") or "").upper() in _OUTER for join in joins):
        return None
    keys = list(group.expressions)
    if any(isinstance(key, (exp.Literal, exp.Null, exp.Boolean, exp.Tuple, exp.Paren)) or not key.find(exp.Column) for key in keys):
        return None
    if any(
        (isinstance(node, (exp.AggFunc, exp.Window)) or is_grouping_call(node)) and node.find_ancestor(exp.Select) is select
        for node in select.walk()
    ):
        return None
    key_texts = {key.sql() for key in keys}
    output_texts = set()
    for item in select.expressions:
        value = item.this if isinstance(item, exp.Alias) else item
        if isinstance(value, exp.Star) or (isinstance(value, exp.Column) and isinstance(value.this, exp.Star)):
            return None
        output_texts.add(value.sql())
    if output_texts != key_texts:
        return None
    copy = select.copy()
    copy.set("group", None)
    copy.set("distinct", exp.Distinct())
    return copy
