"""What a proof found on normalized queries may say about the queries as written.

The algebraic prover normalizes both queries and hands the two results to the SMT prover. Every normalization rule
keeps the rows a query returns on a database where nothing fails, and the proof stands on that. Two things the SMT
prover then says about the pair are read off the *normalized* text, though, and a rule that keeps the rows can
change both:

* **Which operations can fail.** ``HAVING y > 0`` moved into ``WHERE`` no longer adds up the groups the filter
  drops, so a ``SUM`` that overflowed there no longer can. The SMT prover compared two ``WHERE`` queries and said
  "both queries can raise the same runtime errors", which is false of the pair as written: one of them fails on a
  database where the other returns rows. The error comparison is kept only when, on each side, the SMT prover itself
  proves the written query equal to its normalized form *with the same error exposure* (verdict ``none`` or
  ``same``); a side it cannot vouch for that way leaves the verdict ``unknown`` and the plain "runtime errors are
  not modeled" label. Only BigQuery proofs carry an error comparison, and only they pay for this check.
* **Which FLOAT64 sums share a plan.** ``float_sum_order`` narrows "SUM and AVG are treated as independent of row
  order" to "an identical SUM over the same rows returns the same value" when every sum has an identical partner on
  the other side. A sum of partial sums merged into one ``SUM`` has such a partner only after normalization; as
  written, the two queries add the values in different groupings. The narrowed label is kept only when the queries
  as written pair their ``SUM`` and ``AVG`` calls the same way; otherwise the order assumption is listed again.

Nothing here changes a verdict: a proof stays a proof. It changes what the proof discloses.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

import sqlglot

from . import float_sum_order, smt_errors
from .smt_equivalence import (
    BASE_ASSUMPTIONS,
    ERROR_REFINES_ASSUMPTION,
    ERROR_SAME_ASSUMPTION,
    SmtEquivalenceResult,
    SmtStatus,
    prove_equivalent_smt,
)

NOT_MODELED, ROW_ORDER = BASE_ASSUMPTIONS[1], BASE_ASSUMPTIONS[2]
UNKNOWN_DETAIL = (
    "the queries were compared after algebraic normalization, which can change where an operation fails; "
    "the error comparison of the queries as written is not known"
)
# The side check compiles a query twice more; past this size that costs more than the proof did.
_SIDE_CHECK_CHARACTERS = 6000
_SIDE_CHECK_TIMEOUT_MS = 2000
_KEEPS_EXPOSURE = (smt_errors.NONE, smt_errors.SAME)


def _same_exposure(written: str, normalized: str, options: dict) -> bool:
    """Whether the SMT prover proves ``written`` equal to ``normalized`` with the same operations able to fail."""

    if written == normalized:
        return True
    if len(written) > _SIDE_CHECK_CHARACTERS or len(normalized) > _SIDE_CHECK_CHARACTERS:
        return False
    options = {**options, "compare_names": False, "timeout_ms": min(options.get("timeout_ms") or 5000, _SIDE_CHECK_TIMEOUT_MS)}
    try:
        side = prove_equivalent_smt(written, normalized, **options)
    except Exception:  # noqa: BLE001 - not vouching keeps the verdict unknown
        return False
    report = getattr(side, "errors", None)
    return side.status is SmtStatus.PROVEN_EQUIVALENT and report is not None and report.verdict in _KEEPS_EXPOSURE


def _without_error_claim(assumptions: tuple) -> tuple:
    """``assumptions`` with a claim about runtime errors replaced by "not modeled" (added when no claim was listed)."""

    claims = (ERROR_SAME_ASSUMPTION, ERROR_REFINES_ASSUMPTION)
    if NOT_MODELED in assumptions:
        return tuple(a for a in assumptions if a not in claims)
    if any(a in claims for a in assumptions):
        return tuple(dict.fromkeys(NOT_MODELED if a in claims else a for a in assumptions))
    head = 1 if assumptions and assumptions[0] == BASE_ASSUMPTIONS[0] else 0
    return assumptions[:head] + (NOT_MODELED,) + assumptions[head:]


def _paired_sums(written: tuple[str, str], dialect: str) -> bool:
    """Whether every SUM and AVG of one written query has an identical partner (same plan text) in the other."""

    try:
        left, right = (float_sum_order.call_keys(sqlglot.parse_one(sql, read=dialect)) for sql in written)
    except (sqlglot.errors.SqlglotError, RecursionError):
        return False
    return None not in left and None not in right and Counter(left) == Counter(right)


def settle(result: SmtEquivalenceResult, written: tuple[str, str], normalized: tuple[str, str], options: dict) -> SmtEquivalenceResult:
    """``result`` (the SMT prover's answer for ``normalized``) with its disclosures made true of ``written``."""

    if result.status is not SmtStatus.PROVEN_EQUIVALENT or tuple(written) == tuple(normalized):
        return result
    assumptions = tuple(result.assumptions)
    changes: dict = {}
    if float_sum_order.SAME_PLAN_ASSUMPTION in assumptions and not _paired_sums(written, options.get("dialect", "bigquery")):
        assumptions = tuple(dict.fromkeys(ROW_ORDER if a == float_sum_order.SAME_PLAN_ASSUMPTION else a for a in assumptions))
    report = getattr(result, "errors", None)
    if report is not None and report.verdict != smt_errors.UNKNOWN:
        if not all(_same_exposure(w, n, options) for w, n in zip(written, normalized)):
            assumptions = _without_error_claim(assumptions)
            changes["errors"] = dataclasses.replace(report, verdict=smt_errors.UNKNOWN, detail=UNKNOWN_DETAIL)
    if assumptions != tuple(result.assumptions):
        changes["assumptions"] = assumptions
    return dataclasses.replace(result, **changes) if changes else result
