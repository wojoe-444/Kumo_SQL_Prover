"""A windowed ``SUM`` of an integer literal is that literal times the windowed row count.

    SUM(100) OVER (PARTITION BY d ORDER BY s)        100 * COUNT(*) OVER (PARTITION BY d ORDER BY s)

``SUM(c)`` over a frame of ``n`` rows adds ``c`` once per row: ``c * n``, and ``COUNT(*)`` over the same frame is ``n``.
The two differ only over an empty frame, where ``SUM`` is NULL and ``c * COUNT(*)`` is ``c * 0``. So the rewrite is
made only for a frame that always holds the current row: no frame clause (the default frame, with or without
``ORDER BY``, ends at the current row or spans the partition), or a frame that starts at ``UNBOUNDED PRECEDING`` or
``CURRENT ROW`` and ends at ``CURRENT ROW`` or ``UNBOUNDED FOLLOWING``, without ``EXCLUDE``. The partition, the order and
the frame are copied unchanged, so both functions read the same rows (ties in a ``ROWS`` frame included: both
are computed over the one frame the engine chose).

The prover compares windowed selects by their text (``algebraic_equivalence._isolate_windows``), so this runs
before the windows are isolated: Calcite's rule that reduces ``SUM`` of a constant writes the right-hand form.

Conditions, all checked:

* the function is ``SUM`` of an integer literal, with no ``DISTINCT`` and no ``FILTER`` (a filter could leave no row);
* the window is written inline (``OVER (..)``, not a named window) with only ``PARTITION BY``, ``ORDER BY`` and a
  frame as above; a frame bounded by an offset (``1 PRECEDING``, ``2 FOLLOWING``) or with ``EXCLUDE`` is left alone;
* integer ``SUM`` and ``COUNT`` have the same type in PostgreSQL and BigQuery; elsewhere the value is the same and,
  as in the rest of the prover, the type of an output is not compared.
"""

from __future__ import annotations

from sqlglot import exp

_WINDOW_ARGS = frozenset({"this", "partition_by", "order", "spec", "over"})


def _holds_current_row(spec: exp.Expression | None) -> bool:
    if spec is None:
        return True
    if not isinstance(spec, exp.WindowSpec) or spec.args.get("exclude"):
        return False
    start, end = spec.args.get("start"), spec.args.get("end")
    start_side, end_side = (spec.args.get("start_side") or "").upper(), (spec.args.get("end_side") or "").upper()
    start_ok = (start == "UNBOUNDED" and start_side == "PRECEDING") or start == "CURRENT ROW"
    end_ok = end is None or end == "CURRENT ROW" or (end == "UNBOUNDED" and end_side == "FOLLOWING")
    return bool(start_ok and end_ok)


def count_constant_window_sums(tree: exp.Expression) -> exp.Expression:
    """Rewrite each ``SUM(<integer literal>) OVER (..)`` of ``tree`` as the literal times ``COUNT(*) OVER (..)``."""

    for window in list(tree.find_all(exp.Window)):
        call = window.this
        if type(call) is not exp.Sum or any(v for k, v in call.args.items() if k != "this"):
            continue
        value = call.this
        if not isinstance(value, exp.Literal) or value.is_string or not value.name.isdigit():
            continue
        if any(v for k, v in window.args.items() if k not in _WINDOW_ARGS) or (window.args.get("over") or "OVER").upper() != "OVER":
            continue
        if not _holds_current_row(window.args.get("spec")):
            continue
        counted = window.copy()
        counted.set("this", exp.Count(this=exp.Star()))
        product = exp.Mul(this=exp.Literal.number(value.name), expression=counted)
        replacement = exp.Paren(this=product) if isinstance(window.parent, (exp.Binary, exp.Unary)) and not isinstance(window.parent, exp.Paren) else product
        if window is tree:
            tree = replacement
        else:
            window.replace(replacement)
    return tree
