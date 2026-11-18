"""MySQL stores a quotient or AVG a materialising derived table holds rounded to its declared scale (issue 45).

``SELECT t.q * 100 FROM (SELECT SUM(paid = due) / COUNT(*) AS q FROM bill) t`` is ``33.3300`` on MySQL 8.0.46 where
``SELECT SUM(paid = due) / COUNT(*) * 100 FROM bill`` is ``33.3333`` (three bills, one paid in full), so the provers must not
call them equal in the ``mysql`` dialect. Every table here is made up.
"""

from __future__ import annotations

import pytest

from kumo_sql_prover import derived_rounding as D
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtEquivalenceResult, SmtStatus, TableConstraints, prove_equivalent_smt
from kumo_sql_prover.uexpr import prove_bag_equivalent

pytest.importorskip("z3")

SCHEMA = {"bill": ["id", "paid", "due", "ratio"]}
TYPES = {"bill": {"id": "INT", "paid": "INT", "due": "INT", "ratio": "DOUBLE"}}
CONSTRAINTS = {"bill": TableConstraints(not_null=frozenset({"id", "paid", "due", "ratio"}), keys=(("id",),))}
PROVERS = [prove_equivalent_algebraic, prove_equivalent_smt, prove_bag_equivalent]
by_prover = pytest.mark.parametrize("prover", PROVERS, ids=lambda p: p.__name__)
PAID = "SUM(CASE WHEN paid = due THEN 1 ELSE 0 END)"


def prove(prover, left, right, dialect="mysql"):
    return prover(left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect=dialect, timeout_ms=4000)


# One side reads a quotient from a derived table MySQL materialises and computes on it; the other computes it directly.
ROUNDED = {
    "multiplied outside": (f"SELECT t.q * 100 AS p FROM (SELECT {PAID} / COUNT(*) AS q FROM bill) AS t", f"SELECT {PAID} / COUNT(*) * 100 AS p FROM bill"),
    "percentage rounded outside": (
        f"SELECT ROUND(t.pct, 2) AS p FROM (SELECT 100 * {PAID} / COUNT(*) AS pct FROM bill) AS t",
        f"SELECT ROUND(100 * {PAID} / COUNT(*), 2) AS p FROM bill",
    ),
    "average compared outside": (
        "SELECT t.due FROM (SELECT due, AVG(paid) AS a FROM bill GROUP BY due) AS t WHERE t.a = 2.3333",
        "SELECT due FROM bill GROUP BY due HAVING AVG(paid) = 2.3333",
    ),
    "average rounded outside a CTE": (
        "WITH t AS (SELECT due, AVG(paid) AS a FROM bill GROUP BY due) SELECT due, ROUND(a, 2) AS r FROM t",
        "SELECT due, ROUND(AVG(paid), 2) AS r FROM bill GROUP BY due",
    ),
    "quotient of a grouped derived table, compared": (
        "SELECT t.id FROM (SELECT id, paid / due AS q FROM bill GROUP BY id, paid, due) AS t WHERE t.q = 0.3333",
        "SELECT id FROM bill WHERE paid / due = 0.3333",
    ),
    "through a derived table that only passes it on": (
        f"SELECT m.q * 100 FROM (SELECT t.q FROM (SELECT {PAID} / COUNT(*) AS q FROM bill) AS t) AS m",
        f"SELECT {PAID} / COUNT(*) * 100 FROM bill",
    ),
}
kinds = pytest.mark.parametrize("name", sorted(ROUNDED))


@by_prover
@kinds
def test_a_quotient_a_derived_table_stores_is_not_the_direct_one_in_mysql(prover, name):
    left, right = ROUNDED[name]
    result = prove(prover, left, right)
    assert result.status not in (SmtStatus.PROVEN_EQUIVALENT, SmtStatus.PROVEN_CONDITIONALLY), result.reason


@pytest.mark.parametrize("prover", [prove_equivalent_algebraic, prove_bag_equivalent], ids=lambda p: p.__name__)
@kinds
def test_other_dialects_read_it_as_before(prover, name):
    """DuckDB divides into a floating-point number: nothing is stored rounded there (the check is for MySQL only)."""

    left, right = ROUNDED[name]
    assert prove(prover, left, right, dialect="duckdb").proven


def test_the_algebraic_prover_says_why():
    left, right = ROUNDED["multiplied outside"]
    assert "issue 45" in prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="mysql", compare_names=False).reason


# --- what still proves -----------------------------------------------------------------------------------------------------
STILL_PROVEN = {
    "a plain output is rounded to the same four decimals either way": (
        f"SELECT t.q AS p FROM (SELECT {PAID} / COUNT(*) AS q FROM bill) AS t",
        f"SELECT {PAID} / COUNT(*) AS p FROM bill",
    ),
    "a scalar subquery does not round": (
        f"SELECT (SELECT {PAID} / COUNT(*) FROM bill) * 100 AS p",
        f"SELECT {PAID} / COUNT(*) * 100 AS p FROM bill",
    ),
    "a derived table MySQL merges does not round": (
        "SELECT t.q * 100 AS p FROM (SELECT paid / due AS q FROM bill WHERE id = 1) AS t",
        "SELECT paid / due * 100 AS p FROM bill WHERE id = 1",
    ),
    "a CTE MySQL merges does not round": (
        "WITH t AS (SELECT paid / due AS q FROM bill) SELECT t.q * 100 AS p FROM t",
        "SELECT paid / due * 100 AS p FROM bill",
    ),
    "a quotient already rounded to two decimals is stored exactly": (
        f"SELECT t.q * 100 AS p FROM (SELECT ROUND({PAID} / COUNT(*), 2) AS q FROM bill) AS t",
        f"SELECT ROUND({PAID} / COUNT(*), 2) * 100 AS p FROM bill",
    ),
    "a division of floating-point numbers is stored as it is": (
        "SELECT t.q * 100 AS p FROM (SELECT SUM(ratio) / COUNT(*) AS q FROM bill) AS t",
        "SELECT SUM(ratio) / COUNT(*) * 100 AS p FROM bill",
    ),
    "the same derived table on both sides": (
        f"SELECT t.q * 100 AS p FROM (SELECT {PAID} / COUNT(*) AS q FROM bill) AS t",
        f"SELECT t.q * 100 AS p FROM (SELECT COUNT(*) AS n, {PAID} / COUNT(*) AS q FROM bill) AS t",
    ),
    "the same rounding through a CTE on one side and a derived table on the other": (
        f"WITH t AS (SELECT {PAID} / COUNT(*) AS q FROM bill) SELECT t.q * 100 AS p FROM t",
        f"SELECT t.q * 100 AS p FROM (SELECT {PAID} / COUNT(*) AS q FROM bill) AS t",
    ),
}


@pytest.mark.parametrize("name", sorted(STILL_PROVEN))
def test_pairs_whose_rounding_agrees_still_prove(name):
    left, right = STILL_PROVEN[name]
    assert any(prove(prover, left, right).proven for prover in (prove_equivalent_algebraic, prove_bag_equivalent))


def fake(by_text):
    """A prover that answers by whether it was given the wrapped text; it records what it saw."""

    seen = []

    def prove(left, right, **options):
        seen.append(left)
        return SmtEquivalenceResult(by_text["wrapped" if D.WRAPPER in left else "plain"], "fake")

    return prove, seen


@pytest.mark.parametrize(
    ("wrapped_says", "plain_says", "result", "runs"),
    [
        (SmtStatus.PROVEN_EQUIVALENT, SmtStatus.PROVEN_EQUIVALENT, SmtStatus.PROVEN_EQUIVALENT, 1),  # both sides route it alike
        (SmtStatus.PROVEN_CONDITIONALLY, SmtStatus.NOT_PROVEN, SmtStatus.PROVEN_CONDITIONALLY, 1),
        (SmtStatus.NOT_PROVEN, SmtStatus.PROVEN_EQUIVALENT, SmtStatus.NOT_PROVEN, 2),  # the proof that needed the table merged
        (SmtStatus.NOT_PROVEN, SmtStatus.PROVEN_CONDITIONALLY, SmtStatus.NOT_PROVEN, 2),
        (SmtStatus.NOT_EQUIVALENT, SmtStatus.NOT_EQUIVALENT, SmtStatus.NOT_EQUIVALENT, 2),  # a refutation of the pair as written is kept
        (SmtStatus.NOT_EQUIVALENT, SmtStatus.NOT_PROVEN, SmtStatus.NOT_PROVEN, 2),  # the function was free to differ: not a refutation
        (SmtStatus.NOT_PROVEN, SmtStatus.NOT_PROVEN, SmtStatus.NOT_PROVEN, 2),
    ],
)
def test_what_is_returned(wrapped_says, plain_says, result, runs):
    left, right = ROUNDED["multiplied outside"]
    prove_fake, seen = fake({"wrapped": wrapped_says, "plain": plain_says})
    answer = D.proved(prove_fake, left, right, {"dialect": "mysql", "types": TYPES})
    assert answer.status is result
    assert len(seen) == runs
    if result is SmtStatus.NOT_PROVEN and plain_says in (SmtStatus.PROVEN_EQUIVALENT, SmtStatus.PROVEN_CONDITIONALLY):
        assert answer.reason == D.DECLINED


def test_a_pair_with_nothing_to_wrap_is_proved_once_as_it_is():
    prove_fake, seen = fake({"wrapped": SmtStatus.NOT_PROVEN, "plain": SmtStatus.PROVEN_EQUIVALENT})
    left, right = STILL_PROVEN["a plain output is rounded to the same four decimals either way"]
    assert D.proved(prove_fake, left, right, {"dialect": "mysql"}).proven and seen == [left]


def test_a_nested_call_does_not_read_the_pair_again():
    left, right = ROUNDED["multiplied outside"]
    inner = []

    def outer(l, r, **options):
        inner.append(D.proved(lambda a, b, **o: SmtEquivalenceResult(SmtStatus.NOT_PROVEN, "inner"), l, r, options).reason)
        return SmtEquivalenceResult(SmtStatus.NOT_PROVEN, "outer")

    D.proved(outer, left, right, {"dialect": "mysql"})
    assert inner == ["inner", "inner"]


def test_the_wrapper_is_not_named_in_a_proof():
    left, right = STILL_PROVEN["the same rounding through a CTE on one side and a derived table on the other"]
    result = prove(prove_equivalent_algebraic, left, right)
    assert D.WRAPPER.lower() not in " ".join([result.reason, *result.assumptions]).lower()


# --- which outputs are wrapped ---------------------------------------------------------------------------------------------
def wrapped(sql: str, dialect: str = "mysql") -> str | None:
    found = D.reading(sql, sql, dialect, TYPES)
    return found.left_sql if found else None


WRAP = [
    "SELECT t.q * 100 FROM (SELECT SUM(paid) / COUNT(*) AS q FROM bill) AS t",
    "SELECT DISTINCT t.q FROM (SELECT SUM(paid) / COUNT(*) AS q FROM bill) AS t",
    "SELECT t.q FROM (SELECT SUM(paid) / COUNT(*) AS q FROM bill) AS t ORDER BY 1",
    "SELECT t.q FROM (SELECT SUM(paid) / COUNT(*) AS q FROM bill) AS t GROUP BY t.q",
    "SELECT t.q FROM (SELECT SUM(paid) / COUNT(*) AS q FROM bill) AS t UNION ALL SELECT 1",
    "SELECT * FROM (SELECT paid / due AS q FROM bill LIMIT 3) AS t WHERE q > 0.1",
    "SELECT x FROM (SELECT paid / due AS x FROM bill UNION ALL SELECT 1) AS t WHERE x > 0.2",
    "SELECT t.a FROM (SELECT AVG(paid) AS a FROM bill GROUP BY due) AS t JOIN (SELECT 1 AS a) AS u USING (a)",
    "SELECT t.q + 1 FROM (SELECT CASE WHEN COUNT(*) > 1 THEN SUM(paid) / COUNT(*) ELSE 0 END AS q FROM bill) AS t",
    "SELECT t.q * 2 FROM (SELECT paid / due AS q, ROW_NUMBER() OVER () AS n FROM bill) AS t",
    "SELECT m.q * 2 FROM (SELECT t.q FROM (SELECT paid / due AS q FROM bill LIMIT 3) AS t) AS m",
    "SELECT m.q * 2 FROM (SELECT DISTINCT t.q FROM (SELECT paid / due AS q FROM bill) AS t) AS m",
    "SELECT t.q * 2 FROM (SELECT SUM(paid / due) AS q FROM bill) AS t",
    "SELECT t.q * 2 FROM (SELECT 1 / 3 AS q FROM bill LIMIT 1) AS t",
    "WITH RECURSIVE t AS (SELECT paid / due AS q FROM bill) SELECT t.q * 2 FROM t",
]
LEAVE = [
    "SELECT t.q FROM (SELECT SUM(paid) / COUNT(*) AS q FROM bill) AS t",
    "SELECT t.q AS p FROM (SELECT SUM(paid) / COUNT(*) AS q FROM bill) AS t LIMIT 1",
    "SELECT t.q * 100 FROM (SELECT paid / due AS q FROM bill) AS t",
    "SELECT t.q * 100 FROM (SELECT paid / due AS q, (SELECT 1) AS z FROM bill) AS t",
    "SELECT t.q * 100 FROM (SELECT paid / due AS q FROM bill ORDER BY id) AS t",
    "WITH t AS (SELECT paid / due AS q FROM bill) SELECT a.q * 2 FROM t AS a JOIN t AS b ON a.q = b.q",
    "SELECT t.q * 100 FROM (SELECT SUM(ratio) / COUNT(*) AS q FROM bill) AS t",
    "SELECT t.q * 100 FROM (SELECT ROUND(paid / due, 2) AS q FROM bill LIMIT 3) AS t",
    "SELECT t.q * 100 FROM (SELECT ROUND(AVG(paid)) AS q FROM bill GROUP BY due) AS t",
    "SELECT t.q * 100 FROM (SELECT CAST(paid / due AS DECIMAL(10, 6)) AS q FROM bill LIMIT 3) AS t",
    "SELECT t.q * 100 FROM (SELECT paid DIV due AS q FROM bill LIMIT 3) AS t",
    "SELECT t.q * 100 FROM (SELECT 24 / 100 AS q FROM bill LIMIT 1) AS t",
    "SELECT t.n * 100 FROM (SELECT COUNT(*) AS n, SUM(paid) / COUNT(*) AS q FROM bill) AS t",
    "SELECT t.q > 0.3 FROM (SELECT paid / due > 0.5 AS q FROM bill LIMIT 3) AS t",
    "SELECT (SELECT SUM(paid) / COUNT(*) FROM bill) * 100",
    "SELECT 1 FROM bill WHERE 0.3333 IN (SELECT paid / due FROM bill GROUP BY paid, due)",
    "SELECT SUM(paid) / COUNT(*) * 100 FROM bill",
]


@pytest.mark.parametrize("sql", WRAP)
def test_outputs_that_are_stored_rounded_and_read_are_wrapped(sql):
    assert D.WRAPPER in (wrapped(sql) or "")


@pytest.mark.parametrize("sql", LEAVE)
def test_nothing_is_wrapped_where_the_rounding_cannot_be_seen_or_does_not_happen(sql):
    assert wrapped(sql) is None


@pytest.mark.parametrize("dialect", ["bigquery", "duckdb", "postgres", "sqlite"])
def test_only_mysql_is_read(dialect):
    assert wrapped(WRAP[0], dialect) is None


def test_a_wrapped_text_is_not_wrapped_again():
    once = wrapped(WRAP[0])
    assert D.reading(once, once, "mysql", TYPES) is None


def test_an_unaliased_output_keeps_the_name_it_had():
    sql = wrapped("SELECT t.`paid / due` * 2 FROM (SELECT paid / due FROM bill LIMIT 3) AS t")
    assert sql is not None and "AS `paid / due`" in sql


def test_text_that_does_not_parse_is_left_to_the_prover():
    assert D.reading("SELECT FROM (", "SELECT 1 /", "mysql") is None
