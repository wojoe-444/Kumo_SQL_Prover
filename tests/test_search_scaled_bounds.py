"""The counterexample search draws column values near the literals of the two queries. A bound on a scaled
column (``seconds / 60 >= 5``) changes at a value no literal spells (300), so the search also takes the
products of a constant divisor or factor and the whole numbers of the pair (``counterexample.constants_of``).

Without them every generated ``seconds`` stays below 61, every call falls in the first bin, and a query that
counts a ten-minute call in two bins looks equal to one that counts it once.
"""

from __future__ import annotations

import pytest

pytest.importorskip("duckdb")

from kumo_sql_prover import counterexample as cx  # noqa: E402

SPEC = cx.Spec(
    {"calls": cx.Table("calls", [cx.Column("call", "INT", not_null=True), cx.Column("seconds", "INT", not_null=True)], primary_key=("call",))}
)

HALF_OPEN = "SELECT COUNT(*) FROM calls WHERE seconds / 60 >= 5 AND seconds / 60 < 10"
CLOSED = "SELECT COUNT(*) FROM calls WHERE seconds / 60 BETWEEN 5 AND 10"
IN_SECONDS = "SELECT COUNT(*) FROM calls WHERE seconds >= 300 AND seconds < 600"
STRICT = "SELECT COUNT(*) FROM calls WHERE seconds / 60 > 5 AND seconds / 60 < 10"

# (a query, the divisors and factors read from it)
FACTORS = [
    ("SELECT seconds / 60 FROM calls", {60}),
    ("SELECT (seconds) / (60) FROM calls", {60}),
    ("SELECT seconds DIV 60 FROM calls", {60}),
    ("SELECT 60 * seconds, seconds * 24 FROM calls", {24, 60}),
    ("SELECT SUM(seconds) / 3600 FROM calls", {3600}),
    # nothing varies, nothing is scaled, or the constant is the dividend
    ("SELECT 60 * 5 FROM calls", set()),
    ("SELECT seconds / 1, seconds * 1 FROM calls", set()),
    ("SELECT 60 / seconds FROM calls", set()),
    ("SELECT seconds / 0.5 FROM calls", set()),
    ("SELECT seconds / call FROM calls", set()),
    ("SELECT seconds + 60 FROM calls", set()),
]


@pytest.mark.parametrize("sql, expected", FACTORS, ids=[sql for sql, _ in FACTORS])
def test_the_constants_a_column_is_scaled_by(sql, expected):
    import sqlglot

    assert cx._factors(sqlglot.parse_one(sql, read="mysql")) == expected


def test_scaled_bounds_are_products_of_a_factor_and_the_other_whole_numbers():
    constants = cx.constants_of(HALF_OPEN, CLOSED)
    assert constants.ints == {5, 10, 60}
    assert constants.scaled == {300, 600, 3600}


def test_a_bound_from_the_other_query_is_scaled_too():
    assert cx.constants_of("SELECT seconds / 60 FROM calls", "SELECT call FROM calls WHERE call < 7").scaled == {420, 3600}


def test_no_factor_no_scaled_bounds():
    assert cx.constants_of(IN_SECONDS, "SELECT COUNT(*) FROM calls WHERE seconds BETWEEN 300 AND 599").scaled == set()


def test_a_product_a_literal_already_spells_is_not_repeated():
    assert 300 not in cx.constants_of(HALF_OPEN, IN_SECONDS).scaled


def test_at_most_eight_scaled_bounds():
    many = "SELECT COUNT(*) FROM calls WHERE seconds / 7 IN (11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)"
    scaled = cx.constants_of(many).scaled
    assert len(scaled) == 8 and min(scaled) == 49 and 7 * 11 in scaled and 7 * 47 not in scaled


def test_the_domain_only_grows():
    column = SPEC.tables["calls"].column("seconds")
    with_scaled = cx.constants_of(HALF_OPEN, CLOSED)
    without = cx.Constants(ints=set(with_scaled.ints))
    plain, grown = cx._domain(column, without), cx._domain(column, with_scaled)
    assert set(plain) < set(grown)
    assert {299, 300, 301, 599, 600, 601} <= set(grown)
    number = cx.Column("ratio", "NUMERIC")
    assert set(cx._domain(number, without)) < set(cx._domain(number, with_scaled))


def test_a_call_counted_in_two_bins_is_found():
    found = cx.Searcher(SPEC, HALF_OPEN, CLOSED).search(200, seed=1)
    assert found is not None
    assert any(seconds == 600 for _, seconds in found.tables["calls"])


def test_a_bound_left_out_is_found():
    found = cx.Searcher(SPEC, HALF_OPEN, STRICT).search(200, seed=1)
    assert found is not None
    assert any(seconds == 300 for _, seconds in found.tables["calls"])


def test_the_same_bins_in_seconds_still_agree():
    assert cx.Searcher(SPEC, HALF_OPEN, IN_SECONDS).search(200, seed=1) is None
