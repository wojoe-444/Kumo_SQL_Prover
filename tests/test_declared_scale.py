"""MySQL returns a quotient at the number of decimals its expression declares, and a decimal literal adds one (issue 47).

On MySQL 8.0.46, over a made-up ``bill(id, paid, due)`` with ``paid`` and ``due`` declared ``INT`` and one row ``(1, 2, 3)``:
``paid / due`` is ``0.6667``, ``paid * 1.0 / due`` is ``0.66667`` and ``ROUND(100.0 * paid / due, 5)`` is ``66.66667`` where
``ROUND(100 * paid / due, 5)`` is ``66.6667``. The provers read ``1.0`` and ``1`` as the same number and proved each pair
equal, so in the ``mysql`` dialect they must not. Every table here is made up.
"""

from __future__ import annotations

import pytest
import sqlglot

from kumo_sql_prover import declared_scale as S
from kumo_sql_prover import derived_rounding as D
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtEquivalenceResult, SmtStatus, TableConstraints, prove_equivalent_smt
from kumo_sql_prover.uexpr import prove_bag_equivalent

pytest.importorskip("z3")

SCHEMA = {"bill": ["id", "paid", "due", "owed"]}
TYPES = {"bill": {"id": "INT", "paid": "INT", "due": "INT", "owed": "DECIMAL(10,2)"}}
CONSTRAINTS = {"bill": TableConstraints(not_null=frozenset({"id", "paid", "due", "owed"}), keys=(("id",),))}
PROVERS = [prove_equivalent_algebraic, prove_equivalent_smt, prove_bag_equivalent]
by_prover = pytest.mark.parametrize("prover", PROVERS, ids=lambda p: p.__name__)


def prove(prover, left, right, dialect="mysql", types=TYPES):
    return prover(left, right, schema=SCHEMA, types=types, constraints=CONSTRAINTS, compare_names=False, dialect=dialect, timeout_ms=4000)


def proven(result) -> bool:
    return result.status in (SmtStatus.PROVEN_EQUIVALENT, SmtStatus.PROVEN_CONDITIONALLY)


# --- the pairs MySQL answers differently ------------------------------------------------------------------------------------
DIFFERENT = {
    "a factor of 1.0": ("SELECT paid * 1.0 / due AS q FROM bill", "SELECT paid / due AS q FROM bill"),
    "a leading 100.0": ("SELECT 100.0 * paid / due AS q FROM bill", "SELECT 100 * paid / due AS q FROM bill"),
    "a factor that comes last": ("SELECT paid / due * 1.0 AS q FROM bill", "SELECT paid / due AS q FROM bill"),
    "a factor outside the quotient": ("SELECT 100.0 * (paid / due) AS q FROM bill", "SELECT paid * 100 / due AS q FROM bill"),
    "rounded to five places": ("SELECT ROUND(100.0 * paid / due, 5) AS q FROM bill", "SELECT ROUND(100 * paid / due, 5) AS q FROM bill"),
    "rounded to eight places": ("SELECT ROUND(paid * 1.0 / due, 8) AS q FROM bill", "SELECT ROUND(paid / due, 8) AS q FROM bill"),
    "truncated to six places": ("SELECT TRUNCATE(paid * 1.0 / due, 6) AS q FROM bill", "SELECT TRUNCATE(paid / due, 6) AS q FROM bill"),
    "a sum over a count": ("SELECT SUM(paid) * 1.0 / COUNT(*) AS q FROM bill", "SELECT SUM(paid) / COUNT(*) AS q FROM bill"),
    "an average": ("SELECT AVG(paid * 1.0) AS q FROM bill", "SELECT AVG(paid) AS q FROM bill"),
    "a sum of quotients": ("SELECT SUM(100.0 * paid / due) AS q FROM bill", "SELECT SUM(100 * paid / due) AS q FROM bill"),
    "a case arm": (
        "SELECT CASE WHEN due > 0 THEN paid * 1.0 / due ELSE 0 END AS q FROM bill",
        "SELECT CASE WHEN due > 0 THEN paid / due ELSE 0 END AS q FROM bill",
    ),
    "a grouped quotient": (
        "SELECT due, SUM(paid) * 1.0 / COUNT(*) AS q FROM bill GROUP BY due",
        "SELECT due, SUM(paid) / COUNT(*) AS q FROM bill GROUP BY due",
    ),
    "a quotient a derived table stores, returned plainly": (
        "SELECT t.q FROM (SELECT SUM(paid) * 1.0 / COUNT(*) AS q FROM bill) AS t",
        "SELECT t.q FROM (SELECT SUM(paid) / COUNT(*) AS q FROM bill) AS t",
    ),
    "a quotient a derived table stores, computed on": (
        "SELECT t.q * 100 AS p FROM (SELECT SUM(paid) * 1.0 / COUNT(*) AS q FROM bill) AS t",
        "SELECT t.q * 100 AS p FROM (SELECT SUM(paid) / COUNT(*) AS q FROM bill) AS t",
    ),
    "a quotient a CTE stores": (
        "WITH t AS (SELECT paid * 1.0 / due AS q FROM bill GROUP BY paid, due) SELECT ROUND(q, 3) AS r FROM t",
        "WITH t AS (SELECT paid / due AS q FROM bill GROUP BY paid, due) SELECT ROUND(q, 3) AS r FROM t",
    ),
    "a star over a derived table": (
        "SELECT * FROM (SELECT SUM(paid) * 1.0 / COUNT(*) AS q FROM bill) AS t",
        "SELECT * FROM (SELECT SUM(paid) / COUNT(*) AS q FROM bill) AS t",
    ),
    "a union branch": (
        "SELECT paid * 1.0 / due AS q FROM bill UNION ALL SELECT 1",
        "SELECT paid / due AS q FROM bill UNION ALL SELECT 1",
    ),
    "a second column": (
        "SELECT id, 100.0 * paid / due AS q FROM bill",
        "SELECT id, 100 * paid / due AS q FROM bill",
    ),
    "a decimal literal in the divisor's dividend of a decimal column": (
        "SELECT owed * 1.0 / due AS q FROM bill",
        "SELECT owed / due AS q FROM bill",
    ),
}
kinds = pytest.mark.parametrize("name", sorted(DIFFERENT))


@by_prover
@kinds
def test_a_decimal_literal_changes_the_declared_scale_in_mysql(prover, name):
    left, right = DIFFERENT[name]
    result = prove(prover, left, right)
    assert not proven(result), result.reason


@pytest.mark.parametrize("dialect", ["duckdb", "bigquery", "postgres", "sqlite"])
@by_prover
@kinds
def test_other_dialects_are_not_declined_for_it(prover, name, dialect):
    """DuckDB and BigQuery divide into a floating-point number: ``1.0`` changes nothing there (the check is for MySQL only)."""

    left, right = DIFFERENT[name]
    result = prove(prover, left, right, dialect=dialect)
    assert "issue 47" not in result.reason


def test_duckdb_still_proves_the_issue_pair():
    left, right = DIFFERENT["a factor of 1.0"]
    assert prove(prove_equivalent_algebraic, left, right, dialect="duckdb").proven


def test_the_algebraic_prover_says_why():
    left, right = DIFFERENT["a factor of 1.0"]
    assert "issue 47" in prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="mysql", compare_names=False).reason


def test_the_issue_pairs_without_a_schema_type():
    """With no types the column scales are unknown, but the same on both sides: only the literal tells the pairs apart."""

    for left, right in DIFFERENT.values():
        if "owed" in left:
            continue
        assert not proven(prove(prove_equivalent_algebraic, left, right, types=None))


# --- what still proves -----------------------------------------------------------------------------------------------------
STILL_PROVEN = {
    "rounded to two places the spellings agree": (
        "SELECT ROUND(100.0 * paid / due, 2) AS q FROM bill",
        "SELECT ROUND(100 * paid / due, 2) AS q FROM bill",
    ),
    "rounded to four places the spellings agree": (
        "SELECT ROUND(paid * 1.0 / due, 4) AS q FROM bill",
        "SELECT ROUND(paid / due, 4) AS q FROM bill",
    ),
    "rounded to a whole number": (
        "SELECT ROUND(100.0 * paid / due) AS q FROM bill",
        "SELECT ROUND(100 * paid / due) AS q FROM bill",
    ),
    "the same literal on both sides": (
        "SELECT 100.0 * paid / due AS q FROM bill",
        "SELECT paid * 100.0 / due AS q FROM bill",
    ),
    "a whole-number factor in another place": (
        "SELECT 100 * paid / due AS q FROM bill",
        "SELECT paid * 100 / due AS q FROM bill",
    ),
    "a column that is declared decimal on both sides": (
        "SELECT 100 * owed / due AS q FROM bill",
        "SELECT owed * 100 / due AS q FROM bill",
    ),
    "no quotient at all": ("SELECT paid * 1.0 AS q FROM bill WHERE due = 1", "SELECT paid AS q FROM bill WHERE due = 1"),
    "a quotient the value never shows": (
        "SELECT id FROM bill WHERE paid * 1.0 / due > 0.5",
        "SELECT id FROM bill WHERE paid / due > 0.5",
    ),
    "an average over a group of one row is the value": (
        "SELECT id, AVG(paid) AS q FROM bill GROUP BY id",
        "SELECT id, paid AS q FROM bill",
    ),
    "an average of one value across a union": (
        "SELECT id, paid FROM bill GROUP BY id HAVING COUNT(*) = 1 UNION SELECT id, paid FROM bill WHERE due = 1",
        "SELECT id, AVG(paid) FROM bill GROUP BY id HAVING COUNT(*) = 1 UNION SELECT id, paid FROM bill WHERE due = 1",
    ),
    "a quotient floored": ("SELECT FLOOR(paid * 1.0 / due) AS q FROM bill", "SELECT FLOOR(paid / due) AS q FROM bill"),
}


@pytest.mark.parametrize("name", sorted(STILL_PROVEN))
def test_pairs_that_return_the_same_scale_still_prove(name):
    left, right = STILL_PROVEN[name]
    assert any(proven(prove(prover, left, right)) for prover in (prove_equivalent_algebraic, prove_equivalent_smt))


# --- the declared scales ---------------------------------------------------------------------------------------------------
# Each of these was measured as the declared scale of ``CREATE TABLE .. AS SELECT <expression> FROM bill`` on MySQL 8.0.46
# (``bill``: ``paid`` and ``due`` INT, ``owed`` DECIMAL(10,2), ``ratio`` DOUBLE).
MEASURED = {
    "paid / due": 4,
    "paid * 1.0 / due": 5,
    "100.0 * paid / due": 5,
    "100 * paid / due": 4,
    "paid / due * 1.0": 5,
    "paid / due * 100.00": 6,
    "ROUND(100.0 * paid / due, 5)": 5,
    "ROUND(100 * paid / due, 5)": 4,
    "ROUND(100 * paid / due, 7)": 4,
    "ROUND(paid / due, 2)": 2,
    "TRUNCATE(paid * 1.0 / due, 6)": 5,
    "SUM(paid) / COUNT(*)": 4,
    "SUM(paid * 1.0) / COUNT(*)": 5,
    "AVG(paid)": 4,
    "AVG(paid * 1.0)": 5,
    "AVG(owed)": 6,
    "SUM(owed)": 2,
    "SUM(paid / due)": 4,
    "SUM(paid * 1.0 / due)": 5,
    "owed / due": 6,
    "paid / owed": 4,
    "1.0 / due": 5,
    "paid + 1.5": 1,
    "paid - owed": 2,
    "IF(paid > 1, paid / due, 0)": 4,
    "CASE WHEN paid > 1 THEN paid / due ELSE 1.00 END": 4,
    "COALESCE(paid / due, 0.5)": 4,
    "IFNULL(paid * 1.0 / due, 0)": 5,
    "FLOOR(paid / due)": 0,
    "CEIL(paid * 1.0 / due)": 0,
    "ABS(-paid / due)": 4,
    "-paid / due": 4,
    "paid % due": 0,
    "MAX(paid / due)": 4,
    "MIN(paid * 1.0 / due)": 5,
    "CAST(paid / due AS DECIMAL(10, 6))": 6,
    "CAST(paid / due AS DECIMAL(10))": 0,
    "CAST(paid / due AS SIGNED)": 0,
    "GREATEST(paid / due, 0.5)": 4,
    "NULLIF(paid / due, 0)": 4,
    "paid / due / due": 8,
    "(paid / due) * (paid / due)": 8,
    "ROUND(paid / due)": 0,
    "ROUND(100.0 * paid / due)": 0,
    "ROUND(paid * 1.0 / due, 3)": 3,
    "ROUND(owed, 4)": 2,
    "CAST(paid AS DECIMAL(10, 3)) / due": 7,
}


def scale_of(expression: str, types=TYPES):
    tree = sqlglot.parse_one(f"SELECT {expression} FROM bill", read="mysql")
    return D._Reading(tree, types).scale(tree.expressions[0].unalias(), tree)


@pytest.mark.parametrize("expression", sorted(MEASURED))
def test_the_scale_is_the_one_mysql_declares(expression):
    assert scale_of(expression) == S.constant(MEASURED[expression])


@pytest.mark.parametrize("expression", ["paid / ratio", "paid * 1e0 / due", "paid / due * ratio", "CAST(paid AS DOUBLE) / due"])
def test_a_floating_point_operand_makes_it_floating_point(expression):
    types = {"bill": {**TYPES["bill"], "ratio": "DOUBLE"}}
    assert scale_of(expression, types) == S.FLOAT


def test_a_column_the_schema_does_not_type_is_a_symbol_that_adds_up():
    first, second = scale_of("100.0 * paid / due", None), scale_of("paid * 100.0 / due", None)
    assert first == second and first is not None
    assert scale_of("paid / due", None) != scale_of("paid * 1.0 / due", None)
    assert scale_of("ROUND(paid * 1.0 / due, 5)", None) != scale_of("ROUND(paid / due, 5)", None)
    assert scale_of("ROUND(paid / due, 2)", None) == scale_of("ROUND(paid * 1.0 / due, 2)", None) == S.constant(2)


def test_a_column_through_a_derived_table_has_the_scale_of_what_it_stores():
    tree = sqlglot.parse_one("SELECT t.q FROM (SELECT SUM(paid) * 1.0 / COUNT(*) AS q FROM bill) AS t", read="mysql")
    assert D._Reading(tree, TYPES).scale(tree.expressions[0], tree) == S.constant(5)


def test_a_function_it_does_not_know_has_no_scale_and_matches_only_its_own_text():
    assert scale_of("CONCAT(paid / due, 'x')") is None
    assert D.scale_differs("SELECT CONCAT(paid / due, 'x') FROM bill", "SELECT CONCAT(paid * 1.0 / due, 'x') FROM bill", "mysql", TYPES)
    assert not D.scale_differs("SELECT CONCAT(paid / due, 'x') FROM bill", "SELECT CONCAT(paid / due, 'x') FROM bill", "mysql", TYPES)


@pytest.mark.parametrize(
    ("declared", "scale"),
    [("INT", 0), ("bigint unsigned", 0), ("DECIMAL(10,2)", 2), ("numeric(12, 5)", 5), ("DECIMAL", 0), ("DECIMAL(8)", 0), ("DOUBLE", S.FLOAT), ("FLOAT", S.FLOAT)],
)
def test_the_scale_of_a_declared_type(declared, scale):
    assert S.of_type(declared) == (S.constant(scale) if isinstance(scale, int) else scale)


def test_a_declared_type_that_is_not_a_number_has_no_scale():
    assert S.of_type("VARCHAR(20)") is None and S.of_type(None) is None


# --- which pairs are compared ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("left", "right", "differs"),
    [
        ("SELECT paid * 1.0 / due FROM bill", "SELECT paid / due FROM bill", True),
        ("SELECT paid / due FROM bill", "SELECT paid / due FROM bill", False),
        ("SELECT paid * 1.0 FROM bill", "SELECT paid FROM bill", False),  # no digit past the declared scale
        ("SELECT ROUND(paid * 1.0 / due, 2) FROM bill", "SELECT ROUND(paid / due, 2) FROM bill", False),
        ("SELECT AVG(paid) FROM bill", "SELECT paid FROM bill", False),  # the exact value fits the larger scale
        ("SELECT paid FROM bill", "SELECT AVG(paid) FROM bill", False),
        ("SELECT paid / due FROM bill", "SELECT paid * 1.00000 FROM bill", True),  # the exact side has more decimals
        ("SELECT AVG(owed) FROM bill", "SELECT owed FROM bill", False),
        ("SELECT * FROM bill", "SELECT paid / due FROM bill", False),  # outputs that cannot be listed are not compared
        ("SELECT * FROM (SELECT paid * 1.0 / due AS q FROM bill) AS t", "SELECT * FROM (SELECT paid / due AS q FROM bill) AS t", True),
        ("SELECT t.* FROM (SELECT paid * 1.0 / due AS q FROM bill) AS t, bill", "SELECT t.* FROM (SELECT paid / due AS q FROM bill) AS t, bill", True),
        ("SELECT paid / due, 1 FROM bill", "SELECT paid / due FROM bill", False),  # different widths: not this check's affair
        ("SELECT paid * 1.0 / due FROM bill UNION ALL SELECT 1", "SELECT paid / due FROM bill UNION ALL SELECT 1", True),
        ("SELECT paid * 1.0 / due FROM bill UNION ALL SELECT 1", "SELECT paid / due FROM bill UNION ALL SELECT 1.00000", False),  # both 5
    ],
)
def test_scale_differs(left, right, differs):
    assert D.scale_differs(left, right, "mysql", TYPES) is differs


def test_only_mysql_is_checked():
    assert not D.scale_differs("SELECT paid * 1.0 / due FROM bill", "SELECT paid / due FROM bill", "duckdb", TYPES)
    assert not D.scale_differs("SELECT paid * 1.0 / due FROM bill", "SELECT paid / due FROM bill", None, TYPES)


def test_a_proof_is_declined_with_the_reason_and_a_refutation_is_kept():
    left, right = DIFFERENT["a factor of 1.0"]
    options = {"dialect": "mysql", "types": TYPES}
    declined = D.proved(lambda *a, **o: SmtEquivalenceResult(SmtStatus.PROVEN_EQUIVALENT, "fake", assumptions=("x",)), left, right, options)
    assert declined.status is SmtStatus.NOT_PROVEN and declined.reason == D.DECLINED_SCALE and declined.assumptions == ("x",)
    conditional = D.proved(lambda *a, **o: SmtEquivalenceResult(SmtStatus.PROVEN_CONDITIONALLY, "fake"), left, right, options)
    assert conditional.status is SmtStatus.NOT_PROVEN
    refuted = D.proved(lambda *a, **o: SmtEquivalenceResult(SmtStatus.NOT_EQUIVALENT, "fake"), left, right, options)
    assert refuted.status is SmtStatus.NOT_EQUIVALENT


def test_the_wrapper_of_a_stored_quotient_carries_the_scale_it_is_stored_at():
    sql = "SELECT t.q * 100 FROM (SELECT SUM(paid) * 1.0 / COUNT(*) AS q FROM bill) AS t"
    found = D.reading(sql, sql.replace("* 1.0", ""), "mysql", TYPES)
    assert found is not None
    assert "KQ_STORED_QUOTIENT(SUM(paid) * 1.0 / COUNT(*), '5+')" in found.left_sql
    assert "KQ_STORED_QUOTIENT(SUM(paid) / COUNT(*), '4+')" in found.right_sql


def test_a_scale_dominates_another_when_it_is_larger_whatever_the_columns_are():
    assert S.dominates(S.constant(4), S.constant(0)) and S.dominates(S.constant(4), S.constant(4))
    assert not S.dominates(S.constant(4), S.constant(5)) and not S.dominates(S.constant(4), S.FLOAT) and not S.dominates(None, S.constant(0))
    paid, due = S.column("bill.paid"), S.column("bill.due")
    assert S.dominates(S.add_digits(paid, 4), paid)
    assert not S.dominates(S.add_digits(paid, 4), due)  # another column may declare more decimals
    assert S.dominates(S.plus(paid, due), paid)
