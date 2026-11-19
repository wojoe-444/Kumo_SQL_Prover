"""A rounded average of a 1-or-0 flag is the sum of the flag over the number of rows (``averaged_flags``, MySQL only).

Measured on MySQL 8.0.46 before the rule was written: 21 rounded spellings on 1,424 groups of 1 to 40,000 rows (the
first ``x`` of ``y`` rows pass; groups next to rounding ties and on them), in five kinds of grouped select, each equal
to ``10^j * x / y`` rounded half up at ``r`` decimals for ``r <= 4`` and ``j + r <= 8``; five controls with ``r`` of 5 to 7 or
``j + r = 9`` differ. ``AVG(e)`` and ``SUM(e) / COUNT(e)`` agree in every digit the server keeps on 1,879 tables.
"""

import random
from fractions import Fraction

import pytest
import sqlglot

from kumo_sql_prover.averaged_flags import write_flag_averages
from kumo_sql_prover.rounded_percentages import MAX_PLACES, MAX_SHIFT

FLAG = "CASE WHEN paid = due THEN 1 ELSE 0 END"
QUOTIENT = f"(SUM({FLAG}) / COUNT(*))"
NOT_NULL = {"bill": {"id", "paid", "due"}}


def reading(sql: str, dialect: str = "mysql", not_null=NOT_NULL) -> str | None:
    tree, changed = write_flag_averages(sqlglot.parse_one(sql, read=dialect), not_null, dialect)
    return tree.sql(dialect=dialect) if changed else None


def test_an_average_of_a_flag_is_its_sum_over_the_count_of_rows():
    assert reading(f"SELECT ROUND(AVG({FLAG}) * 100, 2) AS share FROM bill") == f"SELECT ROUND({QUOTIENT} * 100, 2) AS share FROM bill"
    assert reading(f"SELECT ROUND(100 * AVG({FLAG}), 2) AS share FROM bill") == f"SELECT ROUND(100 * {QUOTIENT}, 2) AS share FROM bill"
    assert reading(f"SELECT ROUND((AVG({FLAG})) * 10.0 * 10, 1) FROM bill") == f"SELECT ROUND(({QUOTIENT}) * 10.0 * 10, 1) FROM bill"
    assert reading(f"SELECT owner, ROUND(AVG({FLAG}), 4) FROM bill GROUP BY owner") == f"SELECT owner, ROUND({QUOTIENT}, 4) FROM bill GROUP BY owner"
    assert reading(f"SELECT owner, ROUND(AVG({FLAG})) FROM bill GROUP BY owner") == f"SELECT owner, ROUND({QUOTIENT}) FROM bill GROUP BY owner"
    several = "CASE WHEN paid > due THEN 2 WHEN paid = due THEN 1 ELSE 0 END"
    assert reading(f"SELECT ROUND(AVG({several}), 3) FROM bill") == f"SELECT ROUND((SUM({several}) / COUNT(*)), 3) FROM bill"


def test_a_hundred_inside_the_average_comes_out_of_it():
    written = f"SELECT ROUND(({QUOTIENT} * 100), 2) AS share FROM bill"
    assert reading("SELECT ROUND(AVG(IF(paid = due, 100, 0)), 2) AS share FROM bill") == written
    assert reading("SELECT ROUND(AVG(CASE WHEN paid = due THEN 100 ELSE 0 END), 2) AS share FROM bill") == written
    assert reading("SELECT ROUND(AVG(IF(paid = due, 1, 0) * 100), 2) AS share FROM bill") == written
    assert reading("SELECT ROUND(AVG(10 * IF(paid = due, 1, 0) * 10), 2) AS share FROM bill") == written


def test_a_comparison_is_read_only_when_it_is_never_null():
    assert reading("SELECT ROUND(AVG(paid = due) * 100, 2) AS share FROM bill") == f"SELECT ROUND({QUOTIENT} * 100, 2) AS share FROM bill"
    assert reading("SELECT ROUND(AVG(b.paid < 3) * 100, 2) FROM bill AS b") == "SELECT ROUND((SUM(CASE WHEN b.paid < 3 THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) FROM bill AS b"
    # the average skips the rows on which the comparison is NULL, and COUNT(*) counts them
    assert reading("SELECT ROUND(AVG(paid = owner) * 100, 2) FROM bill") is None
    assert reading("SELECT ROUND(AVG(paid = due) * 100, 2) FROM bill", not_null={}) is None
    assert reading("SELECT ROUND(AVG(paid = due) * 100, 2) FROM bill", not_null=None) is None
    assert reading("SELECT ROUND(AVG(paid = due) * 100, 2) FROM bill JOIN owner ON owner.id = bill.owner") is None
    assert reading("SELECT ROUND(AVG(paid = due) * 100, 2) FROM bill LEFT JOIN owner ON owner.id = bill.owner") is None


def test_a_flag_column_of_the_one_derived_table():
    flags = f"(SELECT {FLAG} AS f FROM bill) AS t"
    assert reading(f"SELECT ROUND(AVG(t.f) * 100, 2) FROM {flags}") == f"SELECT ROUND((SUM(t.f) / COUNT(*)) * 100, 2) FROM {flags}"
    assert reading(f"SELECT ROUND(AVG(f) * 100, 2) FROM {flags}") == f"SELECT ROUND((SUM(f) / COUNT(*)) * 100, 2) FROM {flags}"
    assert reading("SELECT ROUND(AVG(t.f) * 100, 2) FROM (SELECT paid AS f FROM bill) AS t") is None
    assert reading("SELECT ROUND(AVG(t.f) * 100, 2) FROM (SELECT CASE WHEN paid = due THEN 1 END AS f FROM bill) AS t") is None
    assert reading(f"SELECT ROUND(AVG(t.f) * 100, 2) FROM bill AS b LEFT JOIN {flags} ON TRUE") is None


LEFT_ALONE = [
    f"SELECT AVG({FLAG}) * 100 FROM bill",  # not rounded: the digits past the fourth can be compared
    f"SELECT ROUND(AVG({FLAG}), 5) FROM bill",  # an average of whole numbers declares four decimals
    f"SELECT ROUND(AVG({FLAG}) * 100, 5) FROM bill",
    f"SELECT ROUND(AVG({FLAG}) * 10000000, 2) FROM bill",  # seven and two: one digit too many
    f"SELECT ROUND(AVG({FLAG}) * 1000000, 3) FROM bill",
    "SELECT ROUND(AVG(IF(paid = due, 10000000, 0)), 2) FROM bill",
    f"SELECT ROUND(AVG({FLAG}) * 3, 2) FROM bill",  # not a power of ten
    f"SELECT ROUND(AVG({FLAG}) + 1, 2) FROM bill",
    f"SELECT ROUND(AVG({FLAG}) / 2, 2) FROM bill",
    f"SELECT ROUND(1 - AVG({FLAG}), 2) FROM bill",
    f"SELECT TRUNCATE(AVG({FLAG}) * 100, 2) FROM bill",
    f"SELECT ROUND(AVG({FLAG}) * 100, due) FROM bill GROUP BY due",
    f"SELECT ROUND(AVG({FLAG}) * 100, -1) FROM bill",
    f"SELECT ROUND(AVG(DISTINCT {FLAG}) * 100, 2) FROM bill",
    f"SELECT ROUND(AVG({FLAG}) OVER () * 100, 2) FROM bill",
    f"SELECT ROUND(AVG({FLAG}) OVER (PARTITION BY owner) * 100, 2) FROM bill",
    "SELECT ROUND(AVG(CASE WHEN paid = due THEN 1 END) * 100, 2) FROM bill",  # NULL on the other rows: they are not counted
    "SELECT ROUND(AVG(CASE WHEN paid = due THEN 1 ELSE NULL END) * 100, 2) FROM bill",
    "SELECT ROUND(AVG(IF(paid = due, 1, NULL)) * 100, 2) FROM bill",
    "SELECT ROUND(AVG(CASE WHEN paid = due THEN -1 ELSE 0 END), 2) FROM bill",  # negative values were not replayed
    "SELECT ROUND(AVG(CASE WHEN paid = due THEN 0.5 ELSE 0 END), 2) FROM bill",
    "SELECT ROUND(AVG(CASE WHEN paid = due THEN paid ELSE 0 END), 2) FROM bill",
    "SELECT ROUND(AVG(CASE paid WHEN 1 THEN 1 ELSE 0 END), 2) FROM bill",
    "SELECT ROUND(AVG(IF(paid = due, 1, 0) * 100.0), 2) FROM bill",  # a decimal factor inside: the average is of decimals
    "SELECT ROUND(AVG(IF(paid = due, 1, 0) * 3), 2) FROM bill",
    "SELECT ROUND(AVG(paid) * 100, 2) FROM bill",  # a column may hold anything
    "SELECT ROUND(AVG(paid / due) * 100, 2) FROM bill",  # a quotient is cut before it is averaged
    "SELECT ROUND(AVG(paid IS NULL) * 100, 2) FROM bill",
    # an aggregate that reads only an outer row belongs to the outer query, and COUNT(*) would not
    "SELECT b.id, (SELECT ROUND(AVG(CASE WHEN b.paid = b.due THEN 1 ELSE 0 END) * 100, 2) FROM bill AS c) FROM bill AS b GROUP BY b.id",
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_averages_are_left_alone(sql):
    sqlglot.parse_one(sql, read="mysql")
    assert reading(sql) is None


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb", "postgres", "bigquery", "snowflake"])
def test_only_mysql_is_read(dialect):
    assert reading(f"SELECT ROUND(AVG({FLAG}) * 100, 2) AS share FROM bill", dialect) is None


# MySQL's arithmetic as measured: the average of whole numbers is their sum over their number cut off after nine
# decimals, a product is exact, and ROUND rounds half away from zero.


def cut(value: Fraction, digits: int = 9) -> Fraction:
    return Fraction(int(value * 10**digits), 10**digits)


def half_up(value: Fraction, places: int) -> Fraction:
    scaled = value * 10**places
    return Fraction((2 * scaled.numerator + scaled.denominator) // (2 * scaled.denominator), 10**places)


@pytest.mark.parametrize("seed", range(3))
def test_the_average_rounds_as_the_written_quotient_does(seed):
    """``ROUND(AVG(10^k * f) * 10^j, r)`` against ``ROUND(SUM(f) / COUNT(*) * 10^(j + k), r)`` and the exact value."""

    rng = random.Random(seed)
    checked = 0
    for _ in range(3000):
        rows = rng.choice([rng.randint(1, 60), rng.randint(1, 10**4), rng.randint(1, 10**7)])
        inside, outside, places = rng.choice([(k, j, r) for k in range(5) for j in range(7) for r in range(MAX_PLACES + 1) if k + j + r <= MAX_SHIFT])
        tie = rng.randint(0, 10 ** (places + 2))
        near = rows * (2 * tie + 1) // (2 * 10 ** (places + inside + outside))
        for passed in (rng.randint(0, rows), near - 1, near, near + 1):
            if not 0 <= passed <= rows:
                continue
            exact = half_up(Fraction(10 ** (inside + outside) * passed, rows), places)
            average = cut(Fraction(10**inside * passed, rows)) * 10**outside
            written = cut(Fraction(passed, rows)) * 10 ** (inside + outside)
            assert half_up(average, places) == half_up(written, places) == exact, (passed, rows, inside, outside, places)
            checked += 1
    assert checked > 6000


def test_one_digit_more_rounds_apart():
    passed, rows = 1371, 1501  # j + r = 9: the moved quotient has no digit left to round on
    assert half_up(cut(Fraction(passed, rows)) * 10**7, 2) != half_up(Fraction(10**7 * passed, rows), 2)


duckdb = pytest.importorskip("duckdb")

REPLAYED = [  # (the average as DuckDB reads it, the MySQL spelling the rule reads)
    (f"AVG({FLAG}) * 100", f"AVG({FLAG}) * 100"),
    ("AVG(CASE WHEN paid = due THEN 100 ELSE 0 END)", "AVG(IF(paid = due, 100, 0))"),
    ("AVG(CASE WHEN paid = due THEN 1 ELSE 0 END * 100)", "AVG(IF(paid = due, 1, 0) * 100)"),
    ("AVG(CAST(paid < due AS INT)) * 100", "AVG(paid < due) * 100"),
    ("AVG(CASE WHEN paid > due THEN 2 WHEN paid = due THEN 1 ELSE 0 END) * 10", "AVG(CASE WHEN paid > due THEN 2 WHEN paid = due THEN 1 ELSE 0 END) * 10"),
    ("AVG(CASE WHEN owner = paid THEN 10 ELSE 0 END)", "AVG(CASE WHEN owner = paid THEN 10 ELSE 0 END)"),
]


@pytest.mark.parametrize("seed", range(4))
def test_the_rewrite_keeps_the_value_on_generated_tables(seed):
    """The reading without MySQL's digits: on DuckDB (floating point) the average and what the rule writes are the same
    number, and NULL together, on tables with NULL owners, repeated rows, groups of one row and no rows at all."""

    rng = random.Random(seed)
    con = duckdb.connect()
    compared = empty = 0
    for table in range(40):
        con.execute("CREATE OR REPLACE TABLE bill (id INT, owner INT, paid INT, due INT)")
        rows = [(i, rng.choice([None, 1, 2, 3]), rng.randint(0, 3), rng.randint(0, 3)) for i in range(rng.choice([0, 1, 2, 5, 12, 40]))]
        if rows:
            con.executemany("INSERT INTO bill VALUES (?, ?, ?, ?)", rows)
        for ours, theirs in REPLAYED:
            for shape in ("SELECT ROUND({}, 2) AS v FROM bill", "SELECT owner, ROUND({}, 2) AS v FROM bill GROUP BY owner", "SELECT ROUND({}, 2) AS v FROM bill WHERE paid > 1"):
                written = reading(shape.format(theirs))
                assert written is not None, theirs
                unrounded = sqlglot.parse_one(written, read="mysql")
                for call in list(unrounded.find_all(sqlglot.exp.Round)):
                    call.replace(call.this)
                got = con.execute(unrounded.sql(dialect="duckdb") + " ORDER BY ALL").fetchall()
                want = con.execute(shape.format(ours).replace("ROUND(", "(").replace(", 2) AS v", ") AS v") + " ORDER BY ALL").fetchall()
                assert len(got) == len(want)
                for a, b in zip(got, want):
                    assert a[:-1] == b[:-1] and (a[-1] is None) == (b[-1] is None), (written, rows)
                    assert a[-1] is None or abs(float(a[-1]) - float(b[-1])) < 1e-9, (written, rows)
                    compared += 1
                    empty += a[-1] is None
    assert compared > 800 and empty > 20


def test_counting_the_rows_needs_a_flag_that_is_never_null():
    """Why a flag without ``ELSE`` and a comparison of a nullable column are left alone: one NULL row moves the count."""

    con = duckdb.connect()
    con.execute("CREATE TABLE bill (id INT, owner INT, paid INT, due INT)")
    con.execute("INSERT INTO bill VALUES (1, NULL, 1, 1), (2, 1, 1, 1), (3, 7, 1, 1)")
    average, counted = con.execute("SELECT AVG(CAST(paid = owner AS INT)), SUM(CASE WHEN paid = owner THEN 1 ELSE 0 END) / COUNT(*) FROM bill").fetchone()
    assert (average, round(counted, 6)) == (0.5, 0.333333)


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

SCHEMA = {"bill": ["id", "owner", "paid", "due"]}
TYPES = {"bill": {"id": "INT", "owner": "INT", "paid": "INT", "due": "INT"}}
FACTS = {"bill": TableConstraints(not_null=frozenset({"id", "paid", "due"}), keys=(("id",),))}
NULLABLE = {"bill": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),))}
OPTIONS = dict(schema=SCHEMA, types=TYPES, compare_names=False)
PAID = f"SUM({FLAG})"
SHARE = f"SELECT ROUND(({PAID} / COUNT(*)) * 100, 2) AS share FROM bill"
BY_OWNER = f"SELECT owner, ROUND(100 * {PAID} / COUNT(1), 2) AS share FROM bill GROUP BY owner"
AVERAGES = [f"ROUND(AVG({FLAG}) * 100, 2)", f"ROUND(100 * AVG({FLAG}), 2)", "ROUND(AVG(IF(paid = due, 100, 0)), 2)", "ROUND(AVG(IF(paid = due, 1, 0) * 100), 2)", "ROUND(AVG(CASE WHEN paid = due THEN 100 ELSE 0 END), 2)"]


@pytest.mark.parametrize("average", AVERAGES)
def test_a_percentage_asked_as_an_average(average):
    for facts in (FACTS, NULLABLE):  # a CASE with an ELSE needs no NOT NULL column
        assert prove_equivalent_algebraic(SHARE, f"SELECT {average} AS share FROM bill", dialect="mysql", constraints=facts, **OPTIONS).proven
        assert prove_equivalent_algebraic(BY_OWNER, f"SELECT owner, {average} AS share FROM bill GROUP BY owner", dialect="mysql", constraints=facts, **OPTIONS).proven


def test_an_average_of_a_comparison_needs_not_null_columns():
    for left, right in ((SHARE, "SELECT ROUND(AVG(paid = due) * 100, 2) AS share FROM bill"), (BY_OWNER, "SELECT owner, ROUND(AVG(paid = due) * 100, 2) AS share FROM bill GROUP BY owner")):
        assert prove_equivalent_algebraic(left, right, dialect="mysql", constraints=FACTS, **OPTIONS).proven, right
        assert not prove_equivalent_algebraic(left, right, dialect="mysql", constraints=NULLABLE, **OPTIONS).proven, right


def test_a_flag_column_averaged():
    other = f"WITH t AS (SELECT {FLAG} AS f FROM bill) SELECT ROUND(AVG(f) * 100, 2) AS share FROM t"
    assert prove_equivalent_algebraic(SHARE, other, dialect="mysql", constraints=FACTS, **OPTIONS).proven


def test_what_is_another_number_stays_unproved():
    for other in (
        "SELECT ROUND(AVG(CASE WHEN paid = due THEN 1 END) * 100, 2) AS share FROM bill",  # 100 whenever a bill is paid
        f"SELECT ROUND(AVG(DISTINCT {FLAG}) * 100, 2) AS share FROM bill",
        f"SELECT ROUND(AVG({FLAG}) * 100, 1) AS share FROM bill",
        "SELECT ROUND(AVG(CASE WHEN paid <= due THEN 1 ELSE 0 END) * 100, 2) AS share FROM bill",
        f"SELECT ROUND(AVG({FLAG}) * 1000 / 10, 2) AS share FROM bill",
    ):
        assert not prove_equivalent_algebraic(SHARE, other, dialect="mysql", constraints=FACTS, **OPTIONS).proven, other


def test_it_is_a_rule_of_the_tenth_batch():
    assert "averaged_flags" in shape_fallback.TENTH_BATCH
    sql = f"SELECT owner, ROUND(AVG({FLAG}) * 100, 2) AS share FROM bill GROUP BY owner"
    with shape_fallback.tracking() as fired:
        normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "averaged_flags" in fired
    with shape_fallback.without("averaged_flags"), shape_fallback.tracking() as fired:
        normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "averaged_flags" not in fired
