"""A rounded percentage of two whole numbers, however the 100 is placed (``rounded_percentages``, MySQL only)."""

import random
from fractions import Fraction

import pytest
import sqlglot

from kumo_sql_prover.rounded_percentages import MAX_PLACES, MAX_SHIFT, write_rounded_percentages

PAID = "SUM(CASE WHEN paid = due THEN 1 ELSE 0 END)"
WRITTEN = f"SELECT ROUND({PAID} / COUNT(*) * 100, 2) AS share FROM bill"


def reading(sql: str, dialect: str = "mysql") -> str | None:
    tree, changed = write_rounded_percentages(sqlglot.parse_one(sql, read=dialect), dialect)
    return tree.sql(dialect=dialect) if changed else None


@pytest.mark.parametrize(
    "spelling",
    [f"100 * {PAID} / COUNT(*)", f"{PAID} * 100 / COUNT(*)", f"({PAID} / COUNT(*)) * 100", f"100 * ({PAID} / COUNT(*))", f"(100 * {PAID}) / (COUNT(*))", f"10 * {PAID} * 10 / COUNT(*)", f"10 * {PAID} / COUNT(*) * 10", f"100.0 * {PAID} / COUNT(*)", f"{PAID} / COUNT(*) * 100.00"],
)
def test_every_spelling_is_written_the_same_way(spelling):
    assert reading(f"SELECT ROUND({spelling}, 2) AS share FROM bill") == WRITTEN


def test_what_is_written_already_stays_and_other_terms_are_read():
    assert reading(WRITTEN) is None
    assert reading("SELECT ROUND(COUNT(DISTINCT owner) / COUNT(*), 2) FROM bill") is None  # no factor: nothing to place
    assert reading("SELECT ROUND(1 * COUNT(DISTINCT owner) / COUNT(*), 2) FROM bill") == "SELECT ROUND(COUNT(DISTINCT owner) / COUNT(*), 2) FROM bill"
    assert reading("SELECT owner, ROUND(100 * SUM(paid = due) / COUNT(id)) FROM bill GROUP BY owner") == "SELECT owner, ROUND(SUM(paid = due) / COUNT(id) * 100) FROM bill GROUP BY owner"
    assert reading("SELECT ROUND(1000 * SUM(IF(paid > 3, 1, 0)) / COUNT(*), 4) FROM bill") == "SELECT ROUND(SUM(IF(paid > 3, 1, 0)) / COUNT(*) * 1000, 4) FROM bill"
    scalars = "SELECT ROUND(100 * (SELECT COUNT(*) FROM bill WHERE paid = due) / (SELECT COUNT(id) FROM bill), 2) AS share"
    assert reading(scalars) == "SELECT ROUND((SELECT COUNT(*) FROM bill WHERE paid = due) / (SELECT COUNT(id) FROM bill) * 100, 2) AS share"
    nested = f"SELECT t.share FROM (SELECT owner, ROUND(100 * {PAID} / COUNT(*), 2) AS share FROM bill GROUP BY owner) AS t WHERE t.share > 5"
    assert reading(nested) == nested.replace(f"100 * {PAID} / COUNT(*)", f"{PAID} / COUNT(*) * 100")


LEFT_ALONE = [
    f"SELECT ROUND(3 * {PAID} / COUNT(*), 2) FROM bill",  # not a power of ten: (1 / 600) * 3 and 3 * 1 / 600 round apart
    f"SELECT ROUND(25 * {PAID} / COUNT(*) * 4, 2) FROM bill",
    f"SELECT ROUND(100.5 * {PAID} / COUNT(*), 2) FROM bill",
    f"SELECT ROUND(0.1 * {PAID} / COUNT(*), 2) FROM bill",
    f"SELECT ROUND(10000000 * {PAID} / COUNT(*), 2) FROM bill",  # the quotient keeps nine digits: seven and two are one too many
    f"SELECT ROUND(100 * {PAID} / COUNT(*), 7) FROM bill",
    f"SELECT ROUND(100 * {PAID} / COUNT(*), 5) FROM bill",  # a quotient declares four decimals and ROUND gives no more: 100.0 would give five
    f"SELECT ROUND(100.0 * {PAID} / COUNT(*), 5) FROM bill",
    f"SELECT ROUND(100 * {PAID} / COUNT(*), -1) FROM bill",
    f"SELECT ROUND(100 * {PAID} / COUNT(*), due) FROM bill",
    f"SELECT 100 * {PAID} / COUNT(*) FROM bill",  # not rounded: the digits past the fourth differ and can be compared
    f"SELECT TRUNCATE(100 * {PAID} / COUNT(*), 2) FROM bill",
    f"SELECT ROUND(100 * {PAID} / (COUNT(*) * 10), 2) FROM bill",  # a factor of the divisor
    f"SELECT ROUND(100 * {PAID} / COUNT(*) / 10, 2) FROM bill",  # two divisions
    f"SELECT ROUND(100 * {PAID} / COUNT(*) + 1, 2) FROM bill",
    f"SELECT ROUND(100 * ({PAID} + 1) / COUNT(*), 2) FROM bill",
    "SELECT ROUND(100 * SUM(paid) / COUNT(*), 2) FROM bill",  # a column may hold a fraction or text
    "SELECT ROUND(100 * SUM(CASE WHEN paid = due THEN paid ELSE 0 END) / COUNT(*), 2) FROM bill",
    "SELECT ROUND(100 * SUM(CASE WHEN paid = due THEN 0.5 ELSE 0 END) / COUNT(*), 2) FROM bill",
    "SELECT ROUND(100 * SUM(CASE WHEN paid = due THEN -1 ELSE 0 END) / COUNT(*), 2) FROM bill",  # negative values were not replayed
    "SELECT ROUND(100 * AVG(CASE WHEN paid = due THEN 1 ELSE 0 END), 2) FROM bill",
    "SELECT ROUND(100 * MAX(paid) / COUNT(*), 2) FROM bill",
    f"SELECT ROUND(100 * {PAID} / paid, 2) FROM bill GROUP BY paid",
    "SELECT ROUND(100 * t.n / t.d, 2) FROM (SELECT COUNT(*) AS n, COUNT(id) AS d FROM bill) AS t",  # columns: not known to be whole here
    "SELECT ROUND(100 * t.q, 2) FROM (SELECT SUM(paid) / COUNT(*) AS q FROM bill) AS t",  # a stored quotient is rounded already (issue 45)
    "SELECT ROUND(100 * (SELECT SUM(paid) / COUNT(*) FROM bill) / COUNT(*), 2) FROM bill",
    "SELECT ROUND(100 * COUNT(*) OVER () / COUNT(*) OVER (PARTITION BY owner), 2) FROM bill",
    "SELECT ROUND(100 * (SELECT COUNT(*) OVER () FROM bill LIMIT 1) / COUNT(*), 2) FROM bill",
    f"SELECT ROUND(100 * {PAID} DIV COUNT(*), 2) FROM bill",
    f"SELECT ROUND(100 * {PAID} % COUNT(*), 2) FROM bill",
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_expressions_are_left_alone(sql):
    sqlglot.parse_one(sql, read="mysql")
    assert reading(sql) is None


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb", "postgres", "bigquery", "snowflake"])
def test_only_mysql_is_read(dialect):
    """SQLite divides two whole numbers into a whole number and DuckDB into a floating-point one: the placing of the
    100 shows there."""

    assert reading(f"SELECT ROUND(100 * {PAID} / COUNT(*), 2) AS share FROM bill", dialect) is None


# MySQL's arithmetic on exact numbers, as the module doc states it: a quotient of whole numbers is cut off after nine
# fractional digits, a product is exact, and ROUND rounds half away from zero.


def cut(value: Fraction, digits: int = 9) -> Fraction:
    return Fraction(int(value * 10**digits), 10**digits)


def half_up(value: Fraction, places: int) -> Fraction:
    scaled = value * 10**places
    return Fraction((2 * scaled.numerator + scaled.denominator) // (2 * scaled.denominator), 10**places)


def spellings(n: int, d: int, k: int) -> list[Fraction]:
    return [cut(Fraction(k * n, d)), cut(Fraction(n, d)) * k, k * cut(Fraction(n, d)), cut(Fraction(n * k, d))]


@pytest.mark.parametrize("seed", range(3))
def test_the_spellings_round_alike_for_a_power_of_ten(seed):
    rng = random.Random(seed)
    checked = 0
    for _ in range(3000):
        d = rng.choice([rng.randint(1, 60), rng.randint(1, 10**4), rng.randint(1, 10**9)])
        shift, places = rng.choice([(j, r) for j in range(7) for r in range(MAX_PLACES + 1) if j + r <= MAX_SHIFT])
        tie = rng.randint(0, 10 ** (places + 2))  # a dividend next to a rounding tie of 10^shift * n / d
        near = d * (2 * tie + 1) // (2 * 10 ** (places + shift))
        for n in (rng.randint(0, 3 * d), near - 1, near, near + 1):
            if n < 0:
                continue
            exact = half_up(Fraction(10**shift * n, d), places)
            assert {half_up(value, places) for value in spellings(n, d, 10**shift)} == {exact}, (n, d, shift, places)
            checked += 1
    assert checked > 10000


def test_another_factor_or_one_digit_more_rounds_apart():
    # three times a quotient that was cut off stays below the tie that three times the dividend reaches
    assert half_up(cut(Fraction(1, 600)) * 3, 2) == 0 and half_up(cut(Fraction(3, 600)), 2) == Fraction(1, 100)
    # with j + r = 9 the moved quotient has no digit left to round on
    n, d = 1371, 1501
    assert half_up(cut(Fraction(n, d)) * 10**7, 2) != half_up(cut(Fraction(10**7 * n, d)), 2)
    assert MAX_SHIFT == 8


def test_more_places_than_the_quotient_declares_round_apart():
    """``ROUND(v, r)`` gives ``min(r, the decimals v declares)`` decimals. A quotient of whole numbers declares four and
    the literal ``100.0`` adds one, so at five places ``100 * 2 / 3`` is 66.6667 and ``100.0 * 2 / 3`` is 66.66667
    (measured on MySQL 8.0.46); up to four places every spelling declares enough."""

    value = cut(Fraction(200, 3))
    assert half_up(value, min(5, 4)) == Fraction(666667, 10**4) and half_up(value, min(5, 4 + 1)) == Fraction(6666667, 10**5)
    for places in range(MAX_PLACES + 1):
        assert half_up(value, min(places, 4)) == half_up(value, min(places, 4 + 1))
    assert MAX_PLACES == 4
    assert reading(f"SELECT ROUND(100.0 * {PAID} / COUNT(*), 4) FROM bill") == f"SELECT ROUND({PAID} / COUNT(*) * 100, 4) FROM bill"


def test_a_sum_of_hundreds_is_a_hundred_times_a_sum_of_ones():
    assert reading("SELECT ROUND(SUM(CASE WHEN paid = due THEN 100 ELSE 0 END) / COUNT(*), 2) AS share FROM bill") == WRITTEN
    assert reading("SELECT ROUND(SUM(IF(paid = due, 100, 0)) / COUNT(*), 2) AS share FROM bill") == WRITTEN
    assert reading("SELECT ROUND(10 * SUM(CASE WHEN paid = due THEN 10 END) / COUNT(*), 2) FROM bill") == "SELECT ROUND(SUM(CASE WHEN paid = due THEN 1 END) / COUNT(*) * 100, 2) FROM bill"
    # left as they are: another value beside the hundred, two arms, a sum of hundreds as the divisor
    assert reading("SELECT ROUND(SUM(CASE WHEN paid = due THEN 100 ELSE 1 END) / COUNT(*), 2) FROM bill") is None
    assert reading("SELECT ROUND(SUM(CASE WHEN paid = due THEN 100 WHEN paid > due THEN 100 ELSE 0 END) / COUNT(*), 2) FROM bill") is None
    assert reading("SELECT ROUND(COUNT(*) / SUM(CASE WHEN paid = due THEN 100 ELSE 0 END), 2) FROM bill") is None
    assert reading("SELECT ROUND(SUM(CASE WHEN paid = due THEN 100.0 ELSE 0 END) / COUNT(*), 2) FROM bill") is None
    assert reading("SELECT ROUND(SUM(CASE WHEN paid = due THEN 10000000 ELSE 0 END) / COUNT(*), 2) FROM bill") is None  # seven and two


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

SCHEMA = {"bill": ["id", "owner", "paid", "due"]}
TYPES = {"bill": {"id": "INT", "owner": "INT", "paid": "INT", "due": "INT"}}
FACTS = {"bill": TableConstraints(not_null=frozenset({"id", "paid", "due"}), keys=(("id",),))}
OPTIONS = dict(schema=SCHEMA, types=TYPES, constraints=FACTS, compare_names=False)
SHARE = f"SELECT ROUND(({PAID} / COUNT(*)) * 100, 2) AS share FROM bill"
BY_OWNER = f"SELECT owner, ROUND(100 * {PAID} / COUNT(1), 2) AS share FROM bill GROUP BY owner"


def test_a_percentage_asked_with_the_hundred_in_other_places():
    for other in (
        f"SELECT ROUND(100 * {PAID} / COUNT(*), 2) AS share FROM bill",
        f"SELECT ROUND({PAID} * 100 / COUNT(id), 2) AS share FROM bill",
        "SELECT ROUND(100 * COUNT(*) / (SELECT COUNT(*) FROM bill), 2) AS share FROM bill WHERE paid = due",
        "SELECT ROUND(100 * a.n / b.n, 2) AS share FROM (SELECT COUNT(*) AS n FROM bill WHERE paid = due) AS a, (SELECT COUNT(*) AS n FROM bill) AS b",
        "WITH t AS (SELECT CASE WHEN paid = due THEN 1 ELSE 0 END AS f FROM bill) SELECT ROUND(SUM(f) * 100 / COUNT(f), 2) AS share FROM t",
        "SELECT ROUND(100.0 * SUM(IF(paid = due, 1, 0)) / COUNT(*), 2) AS share FROM bill",
    ):
        assert prove_equivalent_algebraic(SHARE, other, dialect="mysql", **OPTIONS).proven, other
    for other in (
        f"SELECT owner, ROUND({PAID} / COUNT(*) * 100, 2) AS share FROM bill GROUP BY owner",
        "SELECT owner, ROUND((SUM(IF(paid = due, 1, 0)) / COUNT(*)) * 100, 2) AS share FROM bill GROUP BY owner",
        "SELECT owner, ROUND(COUNT(CASE WHEN paid = due THEN 1 END) / COUNT(*) * 100, 2) AS share FROM bill GROUP BY 1",
    ):
        assert prove_equivalent_algebraic(BY_OWNER, other, dialect="mysql", **OPTIONS).proven, other


def test_what_mysql_answers_differently_stays_unproved():
    for left, right in (
        (f"SELECT ROUND(({PAID} / COUNT(*)) * 3, 2) AS share FROM bill", f"SELECT ROUND(3 * {PAID} / COUNT(*), 2) AS share FROM bill"),
        (SHARE, f"SELECT ROUND(100 * {PAID} / COUNT(*), 3) AS share FROM bill"),
        (SHARE, f"SELECT ROUND(t.pct, 2) AS share FROM (SELECT 100 * {PAID} / COUNT(*) AS pct FROM bill) AS t"),  # 49.51 beside 49.50 on 101 rows
        (f"SELECT ({PAID} / COUNT(*)) * 100 AS share FROM bill", f"SELECT 100 * {PAID} / COUNT(*) AS share FROM bill"),
    ):
        assert not prove_equivalent_algebraic(left, right, dialect="mysql", **OPTIONS).proven, right
    for dialect in ("duckdb", "postgres"):
        assert not prove_equivalent_algebraic(SHARE, f"SELECT ROUND(100 * {PAID} / COUNT(*), 2) AS share FROM bill", dialect=dialect, **OPTIONS).proven, dialect


def test_it_is_a_rule_of_the_tenth_batch():
    assert "rounded_percentages" in shape_fallback.TENTH_BATCH
    sql = f"SELECT owner, ROUND(100 * {PAID} / COUNT(*), 2) AS share FROM bill GROUP BY owner"
    with shape_fallback.tracking() as fired:
        normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "rounded_percentages" in fired
    with shape_fallback.without("rounded_percentages"), shape_fallback.tracking() as fired:
        normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "rounded_percentages" not in fired
