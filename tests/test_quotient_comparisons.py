"""In MySQL ``x / 60 < 5`` over a whole-number column is ``x < 300`` (``quotient_comparisons``)."""

import math
from fractions import Fraction

import pytest
import sqlglot
from sqlglot import exp

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.quotient_comparisons import NAME, bound, quotient_step, read_quotient_comparisons, safe
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"calls": ["id", "secs", "cost", "note"], "agents": ["id", "secs"]}
TYPES = {
    "calls": {"id": "INT", "secs": "INT", "cost": "DECIMAL(10,2)", "note": "VARCHAR(20)"},
    "agents": {"id": "BIGINT", "secs": "INT"},
}
CONSTRAINTS = {
    "calls": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
    "agents": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
}
KINDS = {"<": exp.LT, "<=": exp.LTE, ">": exp.GT, ">=": exp.GTE, "=": exp.EQ, "<>": exp.NEQ}


def written(sql: str, dialect: str = "mysql") -> str | None:
    tree, changed = read_quotient_comparisons(sqlglot.parse_one(sql, read=dialect), TYPES, dialect)
    return tree.sql(dialect=dialect) if changed else None


def prove(left: str, right: str, dialect: str = "mysql"):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect=dialect, exact_arithmetic=True, timeout_ms=5000
    )


def stored(x: int, divisor: Fraction, decimals: int, how: str) -> Fraction:
    """``x / divisor`` kept to ``decimals`` decimals, rounded half away from zero or cut toward zero."""

    scaled = Fraction(x) / divisor * 10**decimals
    if how == "cut":
        whole = math.floor(abs(scaled))
    else:
        whole = math.floor(abs(scaled) + Fraction(1, 2))
    return Fraction(whole if scaled >= 0 else -whole, 10**decimals)


def holds(kind: str, a: Fraction, b: Fraction) -> bool:
    return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "=": a == b, "<>": a != b}[kind]


@pytest.mark.parametrize("divisor", ["1", "2", "3", "7", "60", "60.0", "100", "999", "3600", "9999", "0.5", "1.5", "2.25", "12.5", "0.3"])
def test_the_bound_is_the_comparison_however_the_quotient_is_stored(divisor):
    # The argument of the module, replayed with exact fractions: whatever number of decimals (four or more) the
    # quotient keeps, and whether it is rounded or cut, comparing it with the constant is comparing x with the bound.
    c = Fraction(divisor)
    constants = ["0", "1", "5", "-3", "15", "2.5", "0.5", "-0.5", "0.25", "0.0001", "0.001", "3.3333", "4.99", "9.99", "14.99", "4.9999", "-4.99", "1.05", "0.3333", "16.6667", "4.9833"]
    given = 0
    for text in constants:
        k = Fraction(text)
        centre = math.floor(k * c)
        for kind, node in KINDS.items():
            limit = bound(node, c, k)
            if limit is None:
                continue
            given += 1
            for x in range(centre - 6, centre + 8):
                for decimals in (4, 5, 9):
                    for how in ("round", "cut"):
                        assert holds(kind, stored(x, c, decimals, how), k) == holds(kind, Fraction(x), Fraction(limit)), (x, divisor, kind, text, decimals, how)
    assert given > 20


def test_a_comparison_that_rounding_could_cross_gets_no_bound():
    assert not safe(Fraction(10000), Fraction(5))  # 49999 / 10000 = 4.9999 is 0.0001 away from 5: no margin left
    assert stored(49999, Fraction(20000), 4, "round") == Fraction(5, 2)  # 2.49995 is stored 2.5000: x / 20000 = 2.5 is not x = 50000
    assert bound(exp.EQ, Fraction(20000), Fraction("2.5")) is None
    assert not safe(Fraction(60), Fraction("4.99999"))  # five decimals: the stored quotient has four
    assert not safe(Fraction(0), Fraction(1)) and not safe(Fraction(-60), Fraction(1))
    assert bound(exp.EQ, Fraction(60), Fraction("4.99")) is None  # 299.4 is no whole number: never equal, but NULL for NULL
    assert bound(exp.LTE, Fraction(60), Fraction("4.99")) == 299 and bound(exp.GTE, Fraction(60), Fraction("4.99")) == 300
    assert not safe(Fraction(7), Fraction("0.1429"))  # 1 / 7 = 0.142857..: 0.0000429 away once stored


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT id FROM calls WHERE secs / 60 < 5", "SELECT id FROM calls WHERE secs < 300"),
        ("SELECT id FROM calls WHERE (secs / 60) >= 5 AND (secs / 60) < 10", "SELECT id FROM calls WHERE secs >= 300 AND secs < 600"),
        ("SELECT id FROM calls AS c WHERE 5 <= c.secs / 60", "SELECT id FROM calls AS c WHERE c.secs >= 300"),
        ("SELECT id FROM calls WHERE secs / 60 BETWEEN 5 AND 9.99", "SELECT id FROM calls WHERE secs BETWEEN 300 AND 599"),
        ("SELECT id FROM calls WHERE secs / 60.0 > 14.99", "SELECT id FROM calls WHERE secs > 899"),
        ("SELECT id FROM calls WHERE secs / 60 = 5", "SELECT id FROM calls WHERE secs = 300"),
        ("SELECT id FROM calls WHERE secs / 60 <> -1", "SELECT id FROM calls WHERE secs <> -60"),
        ("SELECT id FROM calls WHERE NOT secs / 60 >= 2.5", "SELECT id FROM calls WHERE NOT secs >= 150"),
        ("SELECT COUNT(secs / 60) FROM calls", "SELECT COUNT(secs) FROM calls"),
        ("SELECT CASE WHEN secs / 60 < 5 THEN 'short' ELSE 'long' END FROM calls", "SELECT CASE WHEN secs < 300 THEN 'short' ELSE 'long' END FROM calls"),
        ("SELECT t.id FROM (SELECT id, secs FROM calls) AS t WHERE t.secs / 60 < 5", "SELECT t.id FROM (SELECT id, secs FROM calls) AS t WHERE t.secs < 300"),
        ("SELECT c.id FROM calls AS c JOIN agents AS a ON a.id = c.id AND a.secs / 2 > c.id WHERE c.secs / 3600 >= 1", "SELECT c.id FROM calls AS c JOIN agents AS a ON a.id = c.id AND a.secs / 2 > c.id WHERE c.secs >= 3600"),
    ],
)
def test_the_comparison_is_written_on_the_column(sql, expected):
    assert written(sql) == expected


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM calls WHERE cost / 60 < 5",  # a decimal column keeps more decimals of the quotient
        "SELECT id FROM calls WHERE note / 60 < 5",  # text divides as a floating-point number
        "SELECT id FROM calls WHERE secs DIV 60 < 5",
        "SELECT id FROM calls WHERE ROUND(secs / 60) < 5",
        "SELECT id FROM calls WHERE secs / 60 < id",
        "SELECT id FROM calls WHERE secs / 60 < '5'",
        "SELECT id FROM calls WHERE secs / 60 < 5e0",
        "SELECT id FROM calls WHERE secs / 60 < 4.99999",
        "SELECT id FROM calls WHERE secs / 10000 < 5",
        "SELECT id FROM calls WHERE secs / 0 < 5",
        "SELECT id FROM calls WHERE secs / -60 < 5",
        "SELECT id FROM calls WHERE secs / id < 5",
        "SELECT id FROM calls WHERE (secs + 1) / 60 < 5",
        "SELECT id FROM calls WHERE secs / 60 = 4.99",
        "SELECT id FROM calls WHERE secs / 60 BETWEEN 5 AND 9.99999",
        "SELECT id FROM unknown_table WHERE secs / 60 < 5",
        "SELECT c.id FROM calls AS c JOIN agents AS a ON a.id = c.id WHERE secs / 60 < 5",  # whose secs
        "SELECT id, cost AS secs FROM calls HAVING secs / 60 < 5",  # the output named secs, not the column
        "SELECT COUNT(DISTINCT secs / 60) FROM calls",
        "WITH calls AS (SELECT id, cost AS secs FROM agents) SELECT id FROM calls WHERE secs / 60 < 5",  # the WITH name hides the table
        "SELECT (SELECT COUNT(*) FROM agents AS a WHERE c.secs / 60 < 5) FROM calls AS c",  # read from another select
    ],
)
def test_other_quotients_are_left_alone(sql):
    assert written(sql) is None


def test_only_mysql_divides_this_way():
    for dialect in ("postgres", "bigquery", "duckdb"):
        assert written("SELECT id FROM calls WHERE secs / 60 < 5", dialect) is None


def test_the_rule_on_one_select_declines_under_a_with_name():
    tree = sqlglot.parse_one("WITH calls AS (SELECT id, cost AS secs FROM agents) SELECT id FROM calls WHERE secs / 60 < 5", read="mysql")
    inner = [select for select in tree.find_all(exp.Select) if select.args.get("where") is not None][0]
    assert quotient_step(inner, TYPES, "mysql") is None
    plain = sqlglot.parse_one("SELECT id FROM calls WHERE secs / 60 < 5", read="mysql")
    assert quotient_step(plain, TYPES, "mysql").sql(dialect="mysql") == "SELECT id FROM calls WHERE secs < 300"
    assert plain.sql(dialect="mysql") == "SELECT id FROM calls WHERE secs / 60 < 5"  # the select it was given is not changed
    assert quotient_step(plain, TYPES, "postgres") is None


PAIRS = [
    ("SELECT COUNT(*) FROM calls WHERE secs / 60 < 5", "SELECT COUNT(*) FROM calls WHERE secs < 300"),
    ("SELECT COUNT(id) FROM calls WHERE secs / 60 >= 5 AND secs / 60 < 10", "SELECT COUNT(*) FROM calls WHERE secs BETWEEN 300 AND 599"),
    ("SELECT id FROM calls WHERE secs / 60 BETWEEN 5 AND 9.99", "SELECT id FROM calls WHERE secs >= 300 AND secs < 600"),
    ("SELECT COUNT(secs / 60) FROM calls WHERE secs / 3600 >= 1", "SELECT COUNT(id) FROM calls WHERE secs >= 3600"),
    ("SELECT t.id FROM (SELECT id, secs / 60 AS minutes FROM calls) AS t WHERE t.minutes < 5", "SELECT id FROM calls WHERE secs < 300"),
]


@pytest.mark.parametrize("left, right", PAIRS)
def test_pairs_proved_only_with_the_rule(left, right):
    assert prove(left, right).proven
    with shape_fallback.without(NAME):
        assert not prove(left, right).proven
    assert not prove(left, right, "postgres").proven  # there the quotient of two whole numbers is another operation


@pytest.mark.parametrize(
    "left, right",
    [
        ("SELECT id FROM calls WHERE secs / 60 < 5", "SELECT id FROM calls WHERE secs < 301"),
        ("SELECT id FROM calls WHERE secs / 60 <= 5", "SELECT id FROM calls WHERE secs < 300"),
        ("SELECT id FROM calls WHERE secs / 60 BETWEEN 5 AND 10", "SELECT id FROM calls WHERE secs BETWEEN 300 AND 599"),
        ("SELECT id FROM calls WHERE secs / 60 > 15", "SELECT id FROM calls WHERE secs >= 900"),
        ("SELECT COUNT(secs / 60) FROM calls", "SELECT COUNT(*) FROM calls"),  # secs may be NULL
        ("SELECT id FROM calls WHERE cost / 60 < 5", "SELECT id FROM calls WHERE cost < 300"),
    ],
)
def test_near_misses_stay_unproved(left, right):
    assert not prove(left, right).proven
