"""The float regrouping check on tables with only some columns typed (float_regrouping.py).

A harness whose pairs carry no types declares the columns its queries use as text or as dates and leaves the
numbers untyped. The check then knows a table's columns from the declared column order, not from the types: read
from the types, an untyped column looked like a column of no source, was looked up in the selects around the
derived table, and a plain projection through a derived table was refused as a sum of partial sums.
"""

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import float_regrouping
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic

SCHEMA = {"prices": ["product", "starts", "ends", "price"], "sold": ["product", "day", "units"]}
SOME = {"prices": {"starts": "DATE", "ends": "DATE"}, "sold": {"day": "DATE"}}  # the numbers are not typed
ALL = {"prices": {"product": "INT", "starts": "DATE", "ends": "DATE", "price": "FLOAT64"}, "sold": {"product": "INT", "day": "DATE", "units": "INT"}}

JOINED = "prices AS p JOIN sold AS u ON p.product = u.product AND u.day BETWEEN p.starts AND p.ends"
FLAT = f"SELECT p.product, ROUND(SUM(units * price) / SUM(units), 2) AS mean FROM {JOINED} GROUP BY p.product"
# the same rows through a derived table that adds nothing up
PROJECTED = {
    "derived table": f"SELECT product, ROUND(SUM(price * units) / SUM(units), 2) AS mean FROM (SELECT p.product, price, units FROM {JOINED}) AS a GROUP BY product",
    "WITH and a product per row": (
        f"WITH c AS (SELECT p.product, price, units, price * units AS total FROM {JOINED}) "
        "SELECT product, ROUND(SUM(total) / SUM(units), 2) AS mean FROM c GROUP BY product"
    ),
}
REGROUPED = ("SELECT SUM(price) AS s FROM prices", "SELECT SUM(s) AS s FROM (SELECT SUM(price) AS s FROM prices GROUP BY product) AS t")


def prove(left, right, types):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=types, compare_names=False, dialect="mysql", timeout_ms=5000)


@pytest.mark.parametrize("types", [SOME, ALL, None], ids=["some columns typed", "every column typed", "no types"])
@pytest.mark.parametrize("name", list(PROJECTED))
def test_a_projection_through_a_derived_table_is_proved(name, types):
    assert prove(FLAT, PROJECTED[name], types).proven


@pytest.mark.parametrize("types", [SOME, ALL], ids=["some columns typed", "every column typed"])
def test_a_sum_of_partial_sums_of_a_column_that_may_be_float_stays_unproved(types):
    result = prove(*REGROUPED, types)
    assert not result.proven and result.reason == float_regrouping.DECLINED


def test_without_any_type_the_regrouping_is_an_assumption():
    result = prove(*REGROUPED, None)
    assert result.proven and float_regrouping.UNTYPED_ASSUMPTION in result.assumptions


def reads(sql: str, order=SCHEMA, types=SOME) -> bool:
    return float_regrouping.sum_of_sums(sqlglot.parse_one(sql, read="mysql"), types, order)


def test_a_table_owns_the_columns_of_its_declared_order():
    reader = float_regrouping._Reader(SOME, {"Prices": ["Product", "starts", "ends", "price"]})
    assert float_regrouping._columns("prices", reader) == {"product", "starts", "ends", "price"}
    assert float_regrouping._columns("sold", reader) == {"day"}  # no order for it: what the types name
    assert float_regrouping._columns("other", reader) is None
    assert float_regrouping._columns("prices", float_regrouping._Reader({}, None)) is None


def test_an_untyped_column_of_a_typed_table_is_that_tables_column():
    for sql in PROJECTED.values():
        assert not reads(sql)
    # without the order the types are all the check has, and what it cannot place still counts as a partial sum
    assert reads(PROJECTED["WITH and a product per row"], order=None)
    assert reads(REGROUPED[1]) and reads(REGROUPED[1], order=None)


def test_a_name_no_source_has_is_still_looked_up_around_the_subquery():
    partial = "(SELECT product, SUM(price) AS s FROM prices GROUP BY product) AS d"
    assert reads(f"SELECT (SELECT SUM(s + units) FROM sold) AS v FROM {partial}")  # s is no column of sold: it is d's partial sum
    assert reads(f"SELECT (SELECT SUM(d.s + units) FROM sold) AS v FROM {partial}")
    assert not reads(f"SELECT (SELECT SUM(units) FROM sold) AS v FROM {partial}")
    assert not reads(f"SELECT (SELECT SUM(d.product + units) FROM sold) AS v FROM {partial}")
