"""Kumo SQL Prover: equivalence proofs for BigQuery SQL.

Importing the package first teaches sqlglot's BigQuery parser the syntax it lacks (``bigquery_syntax``), as every
parse the prover makes relies on it. The names below are the public entry points; everything else is a module of
this package (``kumo_sql_prover.equivalence``, ``kumo_sql_prover.smt_equivalence`` and so on).
"""

from . import bigquery_syntax as _bigquery_syntax  # noqa: F401

from .lift_subqueries import LiftDiagnostic, LiftResult, count_inline_subqueries, lift_subqueries
from .equivalence import EquivalenceResult, EquivalenceStatus, build_bag_verifier_sql, prove_equivalent
from .smt_equivalence import Counterexample, SmtEquivalenceResult, SmtStatus, prove_equivalent_smt
from .result_equivalence import ResultEquivalence, ResultEquivalenceStatus, assert_result_equivalent, check_result_equivalence, generate_synthetic_dataset

__version__ = "0.1.0"

__all__ = [
    "LiftDiagnostic",
    "LiftResult",
    "count_inline_subqueries",
    "lift_subqueries",
    "EquivalenceResult",
    "EquivalenceStatus",
    "build_bag_verifier_sql",
    "prove_equivalent",
    "Counterexample",
    "SmtEquivalenceResult",
    "SmtStatus",
    "prove_equivalent_smt",
    "ResultEquivalence",
    "ResultEquivalenceStatus",
    "assert_result_equivalent",
    "check_result_equivalence",
    "generate_synthetic_dataset",
]
