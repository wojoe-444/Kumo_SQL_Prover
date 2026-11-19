"""A sum of ones divided by a count (``counted_sum_ratios``)."""

import random
import sqlite3

import pytest
import sqlglot

from kumo_sql_prover.counted_sum_ratios import read_counted_sum_ratios

SUM = "SUM(CASE WHEN paid > 3 THEN 1 ELSE 0 END)"
COUNT = "COUNT(CASE WHEN paid > 3 THEN 1 END)"
RATIO = f"SELECT {SUM} / COUNT(*) AS share FROM bill"
NOT_NULL = {"bill": frozenset({"paid", "due"})}


def reading(sql: str, not_null=None, dialect: str = "mysql") -> str | None:
    result = read_counted_sum_ratios(sqlglot.parse_one(sql, read=dialect), None, not_null, dialect)
    return result.sql(dialect=dialect) if result is not None else None


def test_the_sum_under_a_count_is_written_as_a_count():
    assert reading(RATIO) == RATIO.replace(SUM, COUNT)
    assert reading(RATIO, dialect="sqlite") == RATIO.replace(SUM, COUNT)
    assert reading(RATIO.replace(SUM, COUNT)) is None
    # a product with number literals, any count as the divisor, the rest of the output kept
    assert reading(f"SELECT ROUND(100 * {SUM} / COUNT(owner), 2) AS share FROM bill") == f"SELECT ROUND(100 * {COUNT} / COUNT(owner), 2) AS share FROM bill"
    assert reading(f"SELECT ROUND(({SUM} * 100.0) / COUNT(DISTINCT owner), 2) FROM bill WHERE due > 1") == f"SELECT ROUND(({COUNT} * 100.0) / COUNT(DISTINCT owner), 2) FROM bill WHERE due > 1"
    assert reading(f"SELECT ({SUM} / COUNT(*)) * 100 FROM bill AS b") == f"SELECT ({COUNT} / COUNT(*)) * 100 FROM bill AS b"
    assert reading("SELECT SUM(IF(paid > 3, 1, 0)) / COUNT(*) FROM bill") == f"SELECT {COUNT} / COUNT(*) FROM bill"


def test_a_sum_of_a_comparison_needs_columns_that_are_never_null():
    truth = "SELECT SUM(paid = due) / COUNT(*) FROM bill"
    assert reading(truth, NOT_NULL) == "SELECT COUNT(CASE WHEN paid = due THEN 1 END) / COUNT(*) FROM bill"
    assert reading(truth.replace("paid = due", "b.paid >= 4").replace("FROM bill", "FROM bill AS b"), NOT_NULL) == "SELECT COUNT(CASE WHEN b.paid >= 4 THEN 1 END) / COUNT(*) FROM bill AS b"
    assert reading(truth) is None  # paid may be NULL on every row: the sum is NULL there and the count is 0
    assert reading(truth, {"bill": frozenset({"paid"})}) is None
    assert reading(truth.replace("FROM bill", "FROM bill JOIN owner ON owner.id = bill.owner"), NOT_NULL) is None
    assert reading(truth.replace("paid = due", "paid = due + 1"), NOT_NULL) is None
    assert reading(truth.replace("paid = due", "paid IN (1, 2)"), NOT_NULL) is None


FLAGGED = "(SELECT id, CASE WHEN paid > 3 THEN 1 ELSE 0 END AS f, IF(due > 1, 1, 0) AS g, owner FROM bill WHERE owner > 0) AS t"


def test_a_flag_column_of_the_one_derived_table_is_counted():
    count = "COUNT(CASE WHEN {} = 1 THEN 1 END)"
    assert reading(f"SELECT SUM(t.f) / COUNT(t.f) FROM {FLAGGED}") == f"SELECT {count.format('t.f')} / COUNT(t.f) FROM {FLAGGED}"
    assert reading(f"SELECT 100 * SUM(g) / COUNT(*) FROM {FLAGGED}") == f"SELECT 100 * {count.format('g')} / COUNT(*) FROM {FLAGGED}"
    assert reading(f"SELECT SUM(t.owner) / COUNT(*) FROM {FLAGGED}") is None  # not a flag
    assert reading(f"SELECT SUM(t.id) / COUNT(*) FROM {FLAGGED}") is None
    assert reading(f"SELECT SUM(t.f + 1) / COUNT(*) FROM {FLAGGED}") is None
    assert reading(f"SELECT SUM(u.f) / COUNT(*) FROM {FLAGGED}") is None
    assert reading(f"SELECT SUM(t.f) / COUNT(*) FROM bill AS b LEFT JOIN {FLAGGED} ON t.id = b.id") is None  # f is NULL on an added row
    assert reading(f"SELECT SUM(t.f) / COUNT(*) FROM {FLAGGED} JOIN bill AS b ON t.id = b.id") is None
    assert reading(f"SELECT SUM(t.f) / COUNT(*) FROM {FLAGGED}".replace("ELSE 0 END", "END")) is None  # NULL where the test fails
    assert reading(f"SELECT SUM(t.f) / COUNT(*) FROM {FLAGGED}".replace("THEN 1 ELSE 0", "THEN 2 ELSE 0")) is None
    assert reading(f"SELECT SUM(t.f) / COUNT(*) FROM {FLAGGED}".replace("AS t", "AS t(id, g, f, owner)"), dialect="sqlite") is None
    assert reading(f"SELECT SUM(t.f) FROM {FLAGGED}") is None


LEFT_ALONE = [
    f"SELECT {SUM} AS n FROM bill",  # not divided: NULL over no rows, where the count is 0
    f"SELECT {SUM} / COUNT(*) FROM bill GROUP BY owner",  # conditional_count_sums reads the grouped sum
    f"SELECT {SUM} / COUNT(*) FROM bill HAVING COUNT(*) > 1",
    f"SELECT {SUM} / SUM(paid) FROM bill",  # the divisor is not a count
    f"SELECT {SUM} / 4 FROM bill",
    f"SELECT {SUM} / (COUNT(*) + 1) FROM bill",  # 1 over no rows: the division shows the NULL
    f"SELECT COUNT(*) / {SUM} FROM bill",
    f"SELECT ({SUM} + 1) / COUNT(*) FROM bill",  # NULL + 1 is NULL, 0 + 1 is 1, and COUNT(*) is 0 either way: equal, but not read
    "SELECT SUM(CASE WHEN paid > 3 THEN 1 END) / COUNT(*) FROM bill",  # no ELSE: NULL when no row passes
    "SELECT SUM(CASE WHEN paid > 3 THEN 2 ELSE 0 END) / COUNT(*) FROM bill",
    "SELECT SUM(CASE WHEN paid > 3 THEN 1 ELSE 0 END) OVER () / COUNT(*) OVER () FROM bill",
    "SELECT SUM(DISTINCT CASE WHEN paid > 3 THEN 1 ELSE 0 END) / COUNT(*) FROM bill",
    f"SELECT {SUM} * paid / COUNT(*) FROM bill",
    f"SELECT {SUM} DIV COUNT(*) FROM bill",
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_shapes_are_left_alone(sql):
    sqlglot.parse_one(sql, read="mysql")
    assert reading(sql, NOT_NULL) is None


@pytest.mark.parametrize("dialect", ["bigquery", "postgres", "duckdb", "snowflake"])
def test_only_where_a_division_by_zero_is_null(dialect):
    assert reading(RATIO, dialect=dialect) is None


SPELLINGS = [
    RATIO,
    f"SELECT 100 * {SUM} / COUNT(owner) FROM bill",
    f"SELECT ({SUM} * 100) / COUNT(DISTINCT owner) FROM bill WHERE due > 1",
    f"SELECT ({SUM} / COUNT(*)) * 100, COUNT(*) FROM bill",
    "SELECT SUM(paid = due) / COUNT(*) FROM bill",
    "SELECT SUM(t.f) / COUNT(t.f), 100 * SUM(g) / COUNT(*) FROM (SELECT CASE WHEN paid > 3 THEN 1 ELSE 0 END AS f, IIF(due > 1, 1, 0) AS g FROM bill WHERE owner > 0) AS t",
]


@pytest.mark.parametrize("seed", range(4))
def test_the_reading_returns_what_sqlite_returns(seed):
    """SQLite divides by zero as MySQL does (NULL), and reads a comparison as 1 or 0."""

    rng = random.Random(seed)
    pairs = [(sql, reading(sql, NOT_NULL, "sqlite")) for sql in SPELLINGS]
    assert all(read is not None for _, read in pairs)
    empty = nulls = numbers = 0
    for _ in range(60):
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE bill (owner INT, paid INT NOT NULL, due INT NOT NULL)")
        for _ in range(rng.choice([0, 0, 1, 2, 5, 9])):
            con.execute("INSERT INTO bill VALUES (?, ?, ?)", [rng.choice([None, None, 1, 2]), rng.randint(0, 6), rng.randint(0, 6)])
        empty += con.execute("SELECT COUNT(*) FROM bill").fetchone()[0] == 0
        for sql, read in pairs:
            expected = con.execute(sql).fetchall()
            assert expected == con.execute(read).fetchall(), (sql, read)
            nulls += expected[0][0] is None
            numbers += expected[0][0] is not None
    assert empty > 5 and nulls > 20 and numbers > 60


def test_without_the_division_the_two_differ_on_no_rows():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE bill (owner INT, paid INT, due INT)")
    assert con.execute(f"SELECT {SUM} FROM bill").fetchall() == [(None,)]
    assert con.execute(f"SELECT {COUNT} FROM bill").fetchall() == [(0,)]
    # and a sum of a comparison that is NULL on every row is NULL beside a count of 1
    con.execute("INSERT INTO bill VALUES (1, NULL, 4)")
    assert con.execute("SELECT SUM(paid = due) / COUNT(*) FROM bill").fetchall() == [(None,)]
    assert con.execute("SELECT COUNT(CASE WHEN paid = due THEN 1 END) / COUNT(*) FROM bill").fetchall() == [(0,)]


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

SCHEMA = {"bill": ["id", "owner", "paid", "due"]}
TYPES = {"bill": {"id": "INT", "owner": "INT", "paid": "INT", "due": "INT"}}
OPTIONS = dict(schema=SCHEMA, types=TYPES, compare_names=False, dialect="mysql")
FACTS = {"bill": TableConstraints(not_null=frozenset({"id", "paid", "due"}), keys=(("id",),))}
SHARE = "SELECT ROUND((SUM(paid = due) / COUNT(*)) * 100, 2) AS share FROM bill"


def test_a_share_asked_six_ways():
    for other in (
        "SELECT ROUND((SUM(CASE WHEN paid = due THEN 1 ELSE 0 END) / COUNT(id)) * 100, 2) AS share FROM bill",
        "SELECT ROUND((SUM(IF(paid = due, 1, 0)) / COUNT(due)) * 100, 2) AS share FROM bill AS b",
        "SELECT ROUND(COUNT(IF(paid = due, 1, NULL)) / COUNT(*) * 100, 2) AS share FROM bill",
        "SELECT ROUND(((SELECT COUNT(id) FROM bill WHERE paid = due) / (SELECT COUNT(*) FROM bill)) * 100, 2) AS share",
        "SELECT ROUND(COUNT(*) / (SELECT COUNT(*) FROM bill) * 100, 2) AS share FROM bill WHERE paid = due",
        "WITH t AS (SELECT id, CASE WHEN paid = due THEN 1 ELSE 0 END AS same FROM bill) SELECT ROUND(SUM(same) / COUNT(same) * 100, 2) AS share FROM t",
    ):
        assert prove_equivalent_algebraic(SHARE, other, constraints=FACTS, **OPTIONS).proven, other


def test_what_is_not_the_same_share_stays_unproved():
    for other in (
        "SELECT ROUND(3 * SUM(CASE WHEN paid = due THEN 1 ELSE 0 END) / COUNT(*), 2) AS share FROM bill",
        "SELECT ROUND(AVG(CASE WHEN paid = due THEN 100 END), 2) AS share FROM bill",  # 100 whenever a bill is paid (with ELSE 0 it is the share: averaged_flags)
        "SELECT ROUND((SUM(CASE WHEN paid = due THEN 1 ELSE 0 END) / COUNT(owner)) * 100, 2) AS share FROM bill",  # owner may be NULL
        "SELECT ROUND((SUM(CASE WHEN paid > due THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS share FROM bill",
        "SELECT ROUND((COUNT(*) / (SELECT COUNT(*) FROM bill)) * 100, 2) AS share FROM bill WHERE paid >= due",
    ):
        assert not prove_equivalent_algebraic(SHARE, other, constraints=FACTS, **OPTIONS).proven, other
    # without the NOT NULL facts the sum of the comparison can be NULL where the count is 0
    nullable = {"bill": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),))}
    other = "SELECT ROUND((SUM(CASE WHEN paid = due THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS share FROM bill"
    assert not prove_equivalent_algebraic(SHARE, other, constraints=nullable, **OPTIONS).proven


def test_it_is_a_narrow_retry_rule_of_the_tenth_batch():
    assert "counted_sum_ratios" in shape_fallback.TENTH_BATCH
    with shape_fallback.tracking() as fired:
        normalize(RATIO, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "counted_sum_ratios" in fired
    with shape_fallback.without("counted_sum_ratios"), shape_fallback.tracking() as fired:
        normalize(RATIO, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "counted_sum_ratios" not in fired
    with shape_fallback.without("own_key_membership"), shape_fallback.tracking() as fired:  # an attempt older than the batch
        normalize(RATIO, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert not fired & shape_fallback.TENTH_BATCH
