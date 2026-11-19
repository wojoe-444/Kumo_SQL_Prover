"""``IF(p, a, b)`` written as ``CASE WHEN p THEN a ELSE b END`` (``if_cases``)."""

import random
import sqlite3

import pytest
import sqlglot

from kumo_sql_prover.if_cases import write_if_calls_as_cases


def reading(sql: str, dialect: str = "mysql") -> str | None:
    tree, changed = write_if_calls_as_cases(sqlglot.parse_one(sql, read=dialect))
    return tree.sql(dialect=dialect) if changed else None


def test_each_if_call_becomes_a_searched_case():
    assert reading("SELECT SUM(IF(paid = due, 1, 0)) FROM bill") == "SELECT SUM(CASE WHEN paid = due THEN 1 ELSE 0 END) FROM bill"
    assert reading("SELECT IF(paid > 3, 'a', NULL), IF(due IS NULL, paid, due + 1) FROM bill") == "SELECT CASE WHEN paid > 3 THEN 'a' ELSE NULL END, CASE WHEN due IS NULL THEN paid ELSE due + 1 END FROM bill"
    # nested calls, and a call in WHERE and in a subquery
    assert reading("SELECT IF(paid > 3, IF(due > 1, 1, 2), 0) FROM bill") == "SELECT CASE WHEN paid > 3 THEN CASE WHEN due > 1 THEN 1 ELSE 2 END ELSE 0 END FROM bill"
    assert reading("SELECT id FROM bill WHERE IF(paid > 3, due, paid) IN (SELECT IF(due > 1, 1, 2) FROM bill)") == (
        "SELECT id FROM bill WHERE CASE WHEN paid > 3 THEN due ELSE paid END IN (SELECT CASE WHEN due > 1 THEN 1 ELSE 2 END FROM bill)"
    )
    assert reading("SELECT IIF(paid > 3, 1, 0) FROM bill", "sqlite") is not None
    assert reading("SELECT IF(paid > 3, 1, 0) FROM bill", "bigquery") == "SELECT CASE WHEN paid > 3 THEN 1 ELSE 0 END FROM bill"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT CASE WHEN paid > 3 THEN 1 ELSE 0 END FROM bill",  # the arms of a CASE are not calls
        "SELECT CASE paid WHEN 3 THEN 1 WHEN 4 THEN 2 END FROM bill",
        "SELECT IFNULL(paid, 0), NULLIF(paid, 0), COALESCE(paid, due) FROM bill",
        "SELECT paid FROM bill",
    ],
)
def test_other_expressions_are_left_alone(sql):
    assert reading(sql) is None


SPELLINGS = [
    "SELECT id, IIF(paid > 3, 1, 0), IIF(due IS NULL, paid, due + 1), IIF(paid > 3, IIF(due > 1, 'x', NULL), 'y') FROM bill ORDER BY id",
    "SELECT SUM(IIF(paid = due, 1, 0)), COUNT(IIF(paid > due, 1, NULL)), MAX(IIF(paid, due, -1)) FROM bill",
    "SELECT id FROM bill WHERE IIF(paid > 3, due, paid) IN (SELECT IIF(due > 1, 1, 2) FROM bill) ORDER BY id",
]


@pytest.mark.parametrize("seed", range(3))
def test_the_reading_returns_what_sqlite_returns(seed):
    rng = random.Random(seed)
    pairs = [(sql, reading(sql, "sqlite")) for sql in SPELLINGS]
    assert all(read is not None and "IIF" not in read for _, read in pairs)
    nulls = 0
    for _ in range(40):
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE bill (id INT, paid INT, due INT)")
        for number in range(rng.choice([0, 1, 4, 9])):
            row = (number, rng.choice([None, 0, 1, 4, 5]), rng.choice([None, 0, 1, 2, 5]))
            nulls += None in row
            con.execute("INSERT INTO bill VALUES (?, ?, ?)", row)
        for sql, read in pairs:
            assert con.execute(sql).fetchall() == con.execute(read).fetchall(), (sql, read)
    assert nulls > 30


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402

SCHEMA = {"bill": ["id", "owner", "paid", "due"]}
TYPES = {"bill": {"id": "INT", "owner": "INT", "paid": "INT", "due": "INT"}}
OPTIONS = dict(schema=SCHEMA, types=TYPES, compare_names=False, dialect="mysql")


def test_a_grouped_sum_of_ones_meets_its_case_spelling():
    by_case = "SELECT owner, SUM(CASE WHEN paid = due THEN 1 ELSE 0 END) / COUNT(*) AS share FROM bill GROUP BY owner"
    assert prove_equivalent_algebraic(by_case, "SELECT owner, SUM(IF(paid = due, 1, 0)) / COUNT(*) AS share FROM bill GROUP BY owner", **OPTIONS).proven
    assert not prove_equivalent_algebraic(by_case, "SELECT owner, SUM(IF(paid = due, 1, 2)) / COUNT(*) AS share FROM bill GROUP BY owner", **OPTIONS).proven
    assert not prove_equivalent_algebraic(by_case, "SELECT owner, SUM(IF(paid = due, 0, 1)) / COUNT(*) AS share FROM bill GROUP BY owner", **OPTIONS).proven


def test_it_is_a_rule_of_the_tenth_batch():
    assert "if_cases" in shape_fallback.TENTH_BATCH
    sql = "SELECT owner, SUM(IF(paid = due, 1, 0)) FROM bill GROUP BY owner"
    with shape_fallback.tracking() as fired:
        normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "if_cases" in fired
    with shape_fallback.without("if_cases"), shape_fallback.tracking() as fired:
        normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "if_cases" not in fired
