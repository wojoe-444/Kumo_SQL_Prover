"""Settle the levels around window functions before the normalizer isolates them.

``algebraic_equivalence._isolate_windows`` puts each select's window functions in a derived table that the prover
then keeps whole, so two queries agree only when those derived tables read alike. Plans differ in what sits around
a window function: an extra projection under it, a column nobody reads beside it, a filter above it that could sit
below. The three rules that remove those differences run here, once, before the windows are isolated, and again in
the main rule loop for the shapes other rewrites uncover:

* ``window_filters.push_filter_below_windows``
* ``window_sources.inline_windowed_source``
* ``window_pruning.prune_windowed_derived``

This is a shape pass (see ``shape_fallback``): the caller skips it when shape rules are off.
"""

from __future__ import annotations

from sqlglot import exp

from .window_filters import push_filter_below_windows
from .window_pruning import prune_windowed_derived
from .window_sources import inline_windowed_source

_RULES = (push_filter_below_windows, inline_windowed_source, prune_windowed_derived)
_MAX_STEPS = 24


def settle_window_levels(tree: exp.Expression) -> exp.Expression:
    """``tree`` with the rules above applied until none does (at most ``_MAX_STEPS`` rewrites)."""

    if tree.find(exp.Window) is None:
        return tree
    for _ in range(_MAX_STEPS):
        for select in tree.find_all(exp.Select):
            rewritten = next((out for out in (rule(select) for rule in _RULES) if out is not None), None)
            if rewritten is None:
                continue
            if select is tree:
                tree = rewritten
            else:
                select.replace(rewritten)
            break
        else:
            break
    return tree
