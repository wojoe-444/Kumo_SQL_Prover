"""A truth value compared with a number is not "never equal".

``WHERE (paid > 3) = 1`` was proven equal to ``WHERE 1 = 0`` because the SMT model keeps truth values and numbers apart
and calls them unequal. MySQL, SQLite and DuckDB read a truth value as the number 1 or 0, so the filter is ``paid > 3``;
BigQuery and PostgreSQL reject the comparison. The provers now read a truth value as 1 or 0 in MySQL, SQLite and DuckDB
(by value, so a derived-table column is covered too) and decline the pair everywhere else.
"""

import itertools
import sqlite3

import pytest

pytest.importorskip("z3")
duckdb = pytest.importorskip("duckdb")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtStatus, prove_equivalent_smt

SCHEMA = {"sale": ["item_id", "buyer", "paid"]}
PROVERS = (prove_equivalent_algebraic, prove_equivalent_smt)
LAX = ("mysql", "sqlite", "duckdb")
STRICT = ("bigquery", "postgres", "spark", "snowflake")
LEFT = "SELECT buyer FROM sale WHERE (paid > 3) = 1"


def select(where: str) -> str:
    return f"SELECT buyer FROM sale WHERE {where}"


# Pairs whose two sides differ on some table in MySQL, SQLite and DuckDB.
DIFFERENT = [
    (LEFT, select("1 = 0")),
    (select("(paid > 3) <> 1"), select("paid > 3")),
    (select("(paid > 3) > 1"), select("paid > 3")),
    (select("(paid > 3) = 2"), select("paid > 3")),
    (select("(paid > 3) = 1"), select("1 = 1")),
    (select("TRUE = 2"), select("1 = 1")),
    (select("(paid > 3) = 0"), select("paid > 3")),
    (select("(paid > 3) <> 0"), select("1 = 0")),
]

# Pairs that are equal in MySQL, SQLite and DuckDB because a truth value is 1 or 0.
EQUAL = [
    (LEFT, select("paid > 3")),
    (select("(paid > 3) > 0"), select("paid > 3")),
    (select("(paid > 3) IN (1)"), select("paid > 3")),
    (select("(paid > 3) = 0"), select("paid <= 3")),
    (select("(paid > 3) BETWEEN 1 AND 1"), select("paid > 3")),
    (select("CASE (paid > 3) WHEN 1 THEN TRUE ELSE FALSE END"), select("paid > 3")),
    (select("COALESCE(paid > 3, 0) = 1"), select("paid > 3")),
    ("SELECT buyer FROM (SELECT buyer, paid > 3 AS big FROM sale) t WHERE big = 1", select("paid > 3")),
    ("SELECT (paid > 3) = 1 FROM sale", "SELECT (paid > 3) FROM sale"),
    (select("TRUE = 1"), "SELECT buyer FROM sale"),
]

# Reached through a derived table, a CASE, an IN list or a comparison written the other way round.
DISGUISED = [
    ("SELECT buyer FROM (SELECT buyer, paid > 3 AS big FROM sale) t WHERE big = 1", select("1 = 0")),
    ("SELECT buyer FROM (SELECT buyer, paid > 3 AS big FROM sale) t WHERE big IN (1)", select("1 = 0")),
    (select("1 = (paid > 3)"), select("1 = 0")),
    (select("(CASE WHEN paid > 3 THEN paid > 4 ELSE 1 END) = 1"), select("1 = 0")),
    (select("EXISTS (SELECT 1 FROM sale s2 WHERE s2.paid > 3) = 1"), select("1 = 0")),
    (select("(paid > 3) <=> 1"), select("1 = 0")),
]


@pytest.mark.parametrize("dialect", LAX + STRICT)
@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("left,right", DIFFERENT)
def test_a_truth_value_is_not_never_equal_to_a_number(prover, dialect, left, right):
    assert prover(left, right, schema=SCHEMA, dialect=dialect).status is not SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize("dialect", ("mysql", "sqlite", "duckdb", "bigquery", "postgres"))
@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("left,right", DISGUISED)
def test_a_truth_value_read_through_another_form_is_not_never_equal(prover, dialect, left, right):
    assert prover(left, right, schema=SCHEMA, dialect=dialect).status is not SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize("dialect", LAX)
@pytest.mark.parametrize("left,right", EQUAL)
def test_a_truth_value_is_one_or_zero_where_the_engine_says_so(dialect, left, right):
    assert prove_equivalent_algebraic(left, right, schema=SCHEMA, dialect=dialect).status is SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize("dialect", STRICT)
@pytest.mark.parametrize("left,right", EQUAL[:6])
def test_other_dialects_decline_instead_of_guessing(dialect, left, right):
    assert prove_equivalent_algebraic(left, right, schema=SCHEMA, dialect=dialect).status is SmtStatus.NOT_PROVEN


@pytest.mark.parametrize("dialect", ("sqlite", "duckdb"))
def test_verdicts_agree_with_the_engine(dialect):
    """Over comparisons of a truth value with 1, 0, 2, NULL and a decimal: a proof means equal rows on a table that
    holds NULL and every value around the threshold, and a refutation means different rows."""

    rows = [(i, f"b{i}", p) for i, p in enumerate([None, 0, 1, 2, 3, 4, 5, 6, -1, 7])]

    def run(sql):
        if dialect == "sqlite":
            db = sqlite3.connect(":memory:")
            db.execute("CREATE TABLE sale (item_id, buyer, paid)")
            db.executemany("INSERT INTO sale VALUES (?, ?, ?)", rows)
            return sorted(db.execute(sql).fetchall())
        db = duckdb.connect()
        db.execute("CREATE TABLE sale (item_id INTEGER, buyer VARCHAR, paid INTEGER)")
        db.executemany("INSERT INTO sale VALUES (?, ?, ?)", rows)
        return sorted(db.execute(sql).fetchall())

    predicates = ["paid > 3", "NOT (paid > 3)", "paid <= 3", "1 = 0", "1 = 1"]
    for op, k in itertools.product(("=", "<>", "<", ">="), ("0", "1", "2", "NULL")):
        predicates.append(f"(paid > 3) {op} {k}")
    predicates += ["1 = (paid > 3)", "(paid > 3) IN (1, 5)", "(paid > 3) BETWEEN 0 AND 0"]
    for a, b in itertools.combinations(predicates, 2):
        left, right = select(a), select(b)
        try:
            expected = run(left) == run(right)
        except Exception:  # the engine rejects one side (a type error): nothing to compare
            continue
        status = prove_equivalent_algebraic(left, right, schema=SCHEMA, dialect=dialect).status
        if status is SmtStatus.PROVEN_EQUIVALENT:
            assert expected, f"false proof: {a} | {b}"
        elif status is SmtStatus.NOT_EQUIVALENT:
            assert not expected, f"refutation without a difference: {a} | {b}"
