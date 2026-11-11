"""A ``UNION`` of tagged branches that each output a NOT NULL key is a ``UNION ALL``.

``SELECT product_id, 'store1' AS store, store1 AS price FROM products UNION SELECT product_id, 'store2', store2
FROM products`` turns one row per product into one row per product and store. When ``product_id`` is a declared
key of ``products`` and NOT NULL, no branch returns the same row twice (two of its rows have different keys),
and the tag column keeps the branches apart, so the ``UNION`` has nothing to remove and equals ``UNION ALL``.

``disjoint_union`` reads the same thing for branches of at most one row. This module adds the branches a key
keeps free of repeats. A branch qualifies when it

* reads one base table, with no join, ``GROUP BY``, ``HAVING``, window, ``LIMIT`` or set-returning function;
* lists, as plain columns, every column of one declared key of that table, each declared NOT NULL (a key column
  the ``WHERE`` clause equates to a literal counts as listed, as in ``keyed_rules.drop_keyed_distinct``).

A ``SELECT DISTINCT`` branch is left alone, as ``disjoint_union`` leaves it. The branches must be told apart by
a column of different literals (``disjoint_union._told_apart``, which refuses strings a collation could call
equal). Without declared keys nothing here fires, so only proofs that are given keys (or look for them, as the
conditional verdict does) can change.
"""

from __future__ import annotations

from sqlglot import exp

from .disjoint_union import _BANNED, _UNION_ARGS, _branches, _never_repeats, _one_row_per_input, _told_apart
from .keyed_rules import _fixed_columns, _single_table


def _keyed(select: exp.Select, keys: dict[str, list[tuple[str, ...]]], not_null: dict[str, frozenset[str]] | None) -> bool:
    """Whether ``select`` returns no row twice because it lists a NOT NULL key of the one table it reads."""

    if any(select.args.get(key) for key in (*_BANNED, "distinct", "group", "having", "limit", "offset", "windows")):
        return False
    if not _one_row_per_input(select) or any(isinstance(node, (exp.Window, exp.AggFunc)) for item in select.expressions for node in item.walk()):
        return False
    table = _single_table(select)
    if table is None or table.args.get("db") or table.args.get("catalog"):
        return False
    name, alias = table.name.lower(), table.alias_or_name.lower()
    listed = set()
    for item in select.expressions:
        value = item.unalias()
        if isinstance(value, exp.Column) and isinstance(value.this, exp.Identifier) and (not value.table or value.table.lower() == alias):
            listed.add(value.name.lower())
    listed |= _fixed_columns(select, name, alias)
    declared = {c.lower() for c in (not_null or {}).get(name, frozenset())}
    by_table = {t.lower(): ks for t, ks in keys.items()}
    return any(key and set(key) <= listed and set(key) <= declared for key in (tuple(c.lower() for c in k) for k in by_table.get(name, [])))


def separate_keyed_unions(tree: exp.Expression, keys: dict[str, list[tuple[str, ...]]] | None, not_null: dict[str, frozenset[str]] | None) -> exp.Expression:
    """``tree`` with every such ``UNION`` marked ``ALL`` (in place; a pre-pass of ``normalize``)."""

    if not keys:
        return tree
    lowered = {t.lower(): frozenset(c.lower() for c in cols) for t, cols in (not_null or {}).items()}
    for union in list(tree.find_all(exp.Union)):
        if type(union) is not exp.Union or not union.args.get("distinct"):
            continue
        if any(value for key, value in union.args.items() if key not in _UNION_ARGS):
            continue
        branches = _branches(union)
        if branches is None or len(branches) < 2:
            continue
        if all(_never_repeats(branch) or _keyed(branch, keys, lowered) for branch in branches) and _told_apart(branches):
            union.set("distinct", False)
    return tree
