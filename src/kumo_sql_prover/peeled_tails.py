"""A pair whose queries end in the same ``ORDER BY .. LIMIT`` is proved on the queries without it.

    SELECT p.title, (SELECT COUNT(*) FROM comments AS c WHERE c.post = p.id) AS n
    FROM posts AS p ORDER BY p.created DESC LIMIT 10

    SELECT p.title, COUNT(c.id) AS n
    FROM posts AS p LEFT JOIN comments AS c ON c.post = p.id
    GROUP BY p.id, p.title ORDER BY p.created DESC LIMIT 10

Without their last line the prover proves these two equal. With it the pair stayed unproved: most rewrite rules read a
select only when it has no ``ORDER BY``, ``LIMIT`` or ``OFFSET`` of its own (a rule that regroups a select would have
to say what its tail then means), so the normal forms of the two sides never met. The solver layer already knows what
a shared tail means (``smt_equivalence._prove_with_limit``), but it is handed the two normal forms, tails and all.

This module takes the tail off before the rules run. ``peel`` gives the two queries without their tails, each
``ORDER BY`` key that is not an output column added as one more output column (``smt_equivalence._split_limit``, the
reading the solver layer uses), and the prover is run again on those. The pair is proved when they are equal and the
two tails are the same.

Why this is sound. Let ``L`` and ``R`` be the queries without their tails, with the extra key columns. Both tails
order by the same output positions of ``L`` and ``R`` (the same direction and the same place for NULLs at each) and
keep the same rows by position (the same ``LIMIT`` and ``OFFSET``, constants). If ``L`` and ``R`` return the same bag
of rows on every database, then the rows either tail may keep, and the order it may give them, are a function of that
bag alone, so the two queries have the same possible results; the extra columns are then dropped on both sides alike.
Rows tied on the ordering may be cut either way by either query: the proof says so
(``smt_equivalence.TIE_ASSUMPTION``) unless the ordering covers every output column, exactly as the solver layer says
it. Nothing is concluded when the two tails differ, when only one query has one, or when a tail is not read
(``LIMIT`` without ``ORDER BY``, a ``LIMIT`` that is not a constant, an ``ORDER BY`` on a select alias inside an
expression, a key that is not an output of a ``DISTINCT`` select or of a set operation, tails stacked in parentheses),
or when an ``ORDER BY`` key of either query is a number that is not a position in its select list: every engine
rejects ``SELECT a FROM s ORDER BY 3``, the prover declines it even against itself, and taking the tail off would
take the reason off with it (``_unread_position``).

An ``ORDER BY`` without ``LIMIT`` or ``OFFSET`` at the top of a query does not change the bag it returns, which is
what the prover compares (``canonical_rules.drop_order_without_limit`` drops it, but late: after the rules that
decline a select with a tail). When neither query cuts rows, ``peel`` gives the queries without that ``ORDER BY``. The
one exception is kept: the ``ORDER BY`` of ``SELECT DISTINCT ON`` picks the row each key keeps.

This is the last step of a proof (hardest-cases batch 15): it runs only on a pair every earlier step left unproved, so
a pair that was decided before is decided by the same steps, with the same answer. A counterexample for the queries
without their tails is not a counterexample for the pair (the rows before the cut may differ where the first rows do
not), so only a proof is taken from the second run. ``tests/test_peeled_tails.py`` holds the pairs, the tails that must
not be peeled and a check against DuckDB.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from .ast_utils import LossySql, canonical_negation, distinct_on, faithful_sql, merge_wrapper_tails, select_sources

NAME = "peeled_tails"
_QUERIES = (exp.Select, exp.Union, exp.Intersect, exp.Except)


@dataclass(frozen=True)
class Peeled:
    """The two queries without their tails, and whether a proof of them leaves tied rows to be cut alike."""

    left: str
    right: str
    ties: bool


def _without_bare_order(sql: str, dialect: str) -> str | None:
    """``sql`` without a top-level ``ORDER BY`` that cuts no rows, or ``None`` when it has none to drop."""

    tree = canonical_negation(sqlglot.parse_one(sql, read=dialect))
    if not isinstance(tree, _QUERIES) or tree.args.get("order") is None:
        return None
    if tree.args.get("limit") is not None or tree.args.get("offset") is not None:
        return None
    if isinstance(tree, exp.Select) and distinct_on(tree):
        return None  # the ORDER BY picks the row DISTINCT ON keeps per key
    tree.set("order", None)
    return faithful_sql(tree, dialect)


def _least_columns(select: exp.Select, depth: int = 0) -> int:
    """A number of output columns that ``select`` is sure to have: one for each item, and for a ``*`` the least
    number its sources are seen to show (a derived table shows what its own select has; any other source shows at
    least one column)."""

    sources = select_sources(select)
    merged = any(
        join.args.get("using") or join.args.get("method") or str(join.args.get("kind") or "").upper() in ("SEMI", "ANTI") for join in select.args.get("joins") or []
    )  # USING and NATURAL show a joined column once, and a semijoin shows one side only

    def shown(source: exp.Expression) -> int:
        first = source.this if isinstance(source, exp.Subquery) else None
        while isinstance(first, (exp.Union, exp.Intersect, exp.Except, exp.Subquery)):
            first = first.this
        return _least_columns(first, depth + 1) if isinstance(first, exp.Select) and depth < 8 else 1

    total = 0
    for item in select.expressions:
        star = item if isinstance(item, exp.Star) else item.this if isinstance(item, exp.Column) and isinstance(item.this, exp.Star) else None
        if star is None or any(value for value in star.args.values()):
            total += 1  # an item, or a star with columns taken out of it
        elif isinstance(item, exp.Star):
            total += 1 if merged or not sources else sum(shown(source) for source in sources)
        else:
            named = [source for source in sources if source.alias_or_name.lower() == item.table.lower()]
            total += shown(named[0]) if len(named) == 1 and not merged else 1
    return total


def _unread_position(sql: str, dialect: str) -> bool:
    """Whether a top-level ``ORDER BY`` key is a number that is not seen to be a position in the select list.

    ``ORDER BY 3`` under a select of one column is an error in every engine, and the prover declines such a query,
    even against itself. ``_split_limit`` reads a number that is out of range as one more constant column, which is
    right for the solver layer because the query was declined before it gets there; here it would take the
    declined key off with the tail, and the pair would be proved on what is left. So a number key must be a whole
    number from 1 to the number of columns the select list is sure to have (``_least_columns``). A ``*`` stands
    for at least one column, so ``SELECT * .. ORDER BY 1`` is always a position; a larger number beside a star of
    a plain table may be one too, but a table's columns are not counted here.
    """

    root = merge_wrapper_tails(canonical_negation(sqlglot.parse_one(sql, read=dialect)))
    if not isinstance(root, _QUERIES) or root.args.get("order") is None:
        return False
    first = root
    while isinstance(first, (exp.Union, exp.Intersect, exp.Except, exp.Subquery)):
        first = first.this
    for item in root.args["order"].expressions:
        key = item.this if isinstance(item, exp.Ordered) else item
        if not isinstance(key, exp.Literal) or key.is_string:
            continue
        if not isinstance(first, exp.Select) or not key.this.isdigit() or not 1 <= int(key.this) <= _least_columns(first):
            return True
    return False


def peel(left_sql: str, right_sql: str, dialect: str) -> Peeled | None:
    """The pair without its tails when both tails say the same thing, else ``None`` (see the module docstring)."""

    from .smt_equivalence import _split_limit

    try:
        if _unread_position(left_sql, dialect) or _unread_position(right_sql, dialect):
            return None  # an ORDER BY number that names no output column: the query is not one the prover reads
        left_core, left_spec = _split_limit(left_sql, dialect)
        right_core, right_spec = _split_limit(right_sql, dialect)
        if left_core is None or right_core is None:
            return None  # a tail that is not read
        if left_spec is None and right_spec is None:
            # Neither query is read as cutting rows. ``_split_limit`` also says so for a query it cannot model, so the
            # ORDER BY is dropped only where the query is seen to have no LIMIT or OFFSET.
            left_bare, right_bare = _without_bare_order(left_sql, dialect), _without_bare_order(right_sql, dialect)
            if left_bare is None and right_bare is None:
                return None
            return Peeled(left_bare or left_sql, right_bare or right_sql, False)
    except (sqlglot.errors.SqlglotError, LossySql, RecursionError, TypeError):
        return None
    if left_spec is None or right_spec is None:
        return None  # only one query cuts rows
    if left_spec[:3] != right_spec[:3] or left_spec[4] != right_spec[4]:
        return None  # another LIMIT, OFFSET or ordering, or an order key that would stand in for an output
    return Peeled(left_core, right_core, not left_spec[3])
