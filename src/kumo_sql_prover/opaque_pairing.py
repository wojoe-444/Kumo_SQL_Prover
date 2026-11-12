"""Which derived relations of two queries to compare first.

The SMT prover keeps a grouped derived table whole and identifies it by its text; two differently written
bodies get one identity when a nested proof shows them equal (``smt_equivalence._unify_opaque``). That step has
a budget of nested proofs and compares each body with the ones before it, in the order the bodies were read:
the first query's, then the second's. A query that reads one grouped CTE three times with three filters

    FROM ss AS ss1, ss AS ss2, ss AS ss3 WHERE ss1.q = 1 AND ss2.q = 2 AND ss3.q = 3 ..

has three bodies that differ in one literal, and comparing them with each other (three proofs that fail) leaves
nothing of the budget for the comparisons that matter, each body with its counterpart in the other query.

This module orders a first round of comparisons, made before that step and with a budget of its own: only a body
that occurs in one query against a body that occurs in the other. For each body of the second query the
candidates of the first are tried in this order:

1. one that no body has been paired with yet, before one that has;
2. one with the same literals (as a multiset), before one with others;
3. otherwise in the order the first query reads them.

Nothing here decides that two bodies are equal: every pairing is a nested proof by the prover itself, the same
call the existing step makes, and a comparison that is skipped or fails only leaves two bodies apart. The round
is part of the first attempt only (``shape_fallback.enabled()``); the existing step then runs as before, skipping
the pairs this round already tried.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

import sqlglot
from sqlglot import exp

CROSS_PROOFS = 6


def _profile(sql: str) -> tuple[frozenset, tuple]:
    """``(tables read, literals as a sorted multiset)`` of a body's text."""

    tree = sqlglot.parse_one(sql, read="bigquery")
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables = frozenset(t.name.lower() for t in tree.find_all(exp.Table)) - ctes
    literals = Counter((bool(node.is_string), node.name) for node in tree.find_all(exp.Literal))
    return tables, tuple(sorted(literals.items()))


def pair_across(
    bodies: dict[str, tuple[str, int, str]],
    left_keys: set[str],
    right_keys: set[str],
    equal: Callable[[str, str], bool],
    budget: int = CROSS_PROOFS,
) -> tuple[dict[str, str], set[tuple[str, str]]]:
    """``({body of the second query: equal body of the first}, {pairs compared})``.

    ``bodies`` maps a body's key to ``(text, arity, text in the proof's dialect)`` in reading order; ``equal`` runs
    the nested proof for two keys.
    """

    left_only = [key for key in bodies if key in left_keys and key not in right_keys]
    right_only = [key for key in bodies if key in right_keys and key not in left_keys]
    mapping: dict[str, str] = {}
    tried: set[tuple[str, str]] = set()
    if not left_only or not right_only:
        return mapping, tried
    profiles = {key: _profile(bodies[key][0]) for key in left_only + right_only}
    taken: set[str] = set()
    for key in right_only:
        tables, literals = profiles[key]
        candidates = [rep for rep in left_only if bodies[rep][1] == bodies[key][1] and profiles[rep][0] == tables]
        candidates.sort(key=lambda rep: (rep in taken, profiles[rep][1] != literals))  # stable: reading order last
        for rep in candidates:
            if budget <= 0:
                return mapping, tried
            budget -= 1
            tried.add((rep, key))
            if equal(rep, key):
                mapping[key] = rep
                taken.add(rep)
                break
    return mapping, tried
