"""The type of a derived relation's column fixes how a decimal literal beside it is read (``derived_types``)."""

import pytest
import sqlglot

from kumo_sql_prover import derived_types, smt_values

SCHEMA = {"item": ["k", "price", "name"], "sale": ["k", "amt", "qty", "cost", "list_price"]}
TYPES = {
    "item": {"k": "INT64", "price": "NUMERIC", "name": "STRING"},
    "sale": {"k": "INT64", "amt": "NUMERIC", "qty": "INT64", "cost": "FLOAT64", "list_price": "DECIMAL(7, 2)"},
}


def classes(sql: str, types=TYPES):
    return derived_types.output_classes(sqlglot.parse_one(sql, read="bigquery"), SCHEMA, types)


def test_aggregates_of_declared_columns_have_bigquery_types():
    assert classes("SELECT k, AVG(amt) AS a, AVG(qty) AS b, AVG(cost) AS c, SUM(qty) AS d, SUM(amt) AS e, MAX(list_price) AS f, COUNT(*) AS n FROM sale GROUP BY k") == [
        smt_values.INT64, smt_values.NUMERIC, smt_values.FLOAT64, smt_values.FLOAT64, smt_values.INT64, smt_values.NUMERIC, smt_values.NUMERIC, smt_values.INT64,
    ]
    assert classes("SELECT s.k, MIN(s.cost) AS m FROM sale AS s GROUP BY s.k") == [smt_values.INT64, smt_values.FLOAT64]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT AVG(amt * 2) AS a FROM sale",  # an expression: its type is not read here
        "SELECT AVG(d.amt) AS a FROM (SELECT amt FROM sale) AS d",  # a nested derived table
        "SELECT AVG(k) AS a FROM sale, item",  # a name two tables hold
        "SELECT AVG(x.amt) AS a FROM sale",  # an alias the select does not have
        "SELECT name FROM item",  # not a number
        "SELECT AVG(amt) OVER () AS a FROM sale",  # a window
        "SELECT AVG(unknown_column) AS a FROM sale",
    ],
)
def test_anything_else_stays_unknown(sql):
    assert classes(sql) == [None]


def test_no_declared_types_no_answer():
    assert classes("SELECT AVG(amt) AS a FROM sale", types=None) == [None]
    assert classes("SELECT AVG(amt) AS a FROM sale", types={"sale": {"qty": "INT64"}}) == [None]


z3 = pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic  # noqa: E402

AVERAGE = "SELECT i.name FROM item AS i WHERE i.price > (SELECT 1.3 * AVG(amt) FROM sale WHERE sale.k = i.k{extra})"
ALIASED = "SELECT i.name FROM item AS i WHERE i.price > (SELECT 1.3 * AVG(s.amt) FROM sale AS s WHERE s.k = i.k{extra})"


def prove(left: str, right: str, **kwargs):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="bigquery", **kwargs)


@pytest.mark.parametrize("exact", [False, True])
def test_a_decimal_factor_of_a_subquery_average_proves_across_spellings(exact):
    assert prove(AVERAGE.format(extra=""), ALIASED.format(extra=""), exact_arithmetic=exact).proven
    # the two grouped relations differ in text (BETWEEN against two comparisons), so only the type ties the literal
    between = AVERAGE.format(extra=" AND list_price BETWEEN 80 AND 109")
    split = ALIASED.format(extra=" AND s.list_price >= 80 AND s.list_price <= 109")
    assert prove(between, split, exact_arithmetic=exact).proven
    flipped = "SELECT i.name FROM item AS i WHERE (SELECT 1.3 * AVG(amt) FROM sale WHERE sale.k = i.k) < i.price"
    assert prove(AVERAGE.format(extra=""), flipped, exact_arithmetic=exact).proven


def test_an_average_of_integers_is_a_float_and_still_matches_itself():
    left = "SELECT i.name FROM item AS i WHERE i.price > (SELECT 1.3 * AVG(qty) FROM sale WHERE sale.k = i.k AND list_price BETWEEN 80 AND 109)"
    right = "SELECT i.name FROM item AS i WHERE i.price > (SELECT 1.3 * AVG(s.qty) FROM sale AS s WHERE s.k = i.k AND s.list_price >= 80 AND s.list_price <= 109)"
    assert prove(left, right).proven


@pytest.mark.parametrize(
    "right",
    [
        ALIASED.format(extra="").replace("1.3", "1.4"),  # another factor
        ALIASED.format(extra="").replace("AVG(s.amt)", "SUM(s.amt)"),  # another aggregate
        ALIASED.format(extra="").replace("AVG(s.amt)", "AVG(s.cost)"),  # another column
        ALIASED.format(extra=" AND s.list_price >= 80"),  # another filter
        ALIASED.format(extra="").replace("i.price >", "i.price >="),
    ],
)
def test_different_queries_are_not_proved(right):
    assert not prove(AVERAGE.format(extra=""), right).proven
    assert not prove(AVERAGE.format(extra=""), right, exact_arithmetic=True).proven


@pytest.mark.parametrize("dialect", ["mysql", "postgres", "duckdb"])
def test_other_dialects_read_no_bigquery_types(dialect):
    # AVG and SUM have other result types elsewhere (a DECIMAL average of integers in MySQL, a DOUBLE average of
    # decimals in DuckDB), so only the text of the relation ties the two columns there.
    left, right = AVERAGE.format(extra=""), ALIASED.format(extra="")
    assert prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect=dialect).proven
    for other in (right.replace("1.3", "1.4"), right.replace("AVG(s.amt)", "SUM(s.amt)")):
        assert not prove_equivalent_algebraic(left, other, schema=SCHEMA, types=TYPES, dialect=dialect).proven
