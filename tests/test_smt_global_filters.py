"""A global aggregate's WHERE read as a NULL argument on the rows it drops (``smt_global_filters``)."""

import pytest

pytest.importorskip("z3")

from kumo_sql_prover.smt_equivalence import BASE_ASSUMPTIONS, SmtStatus, prove_equivalent_smt

SCHEMA = {"t": ["x", "y", "f", "c"], "u": ["a", "b"]}
TYPES = {"t": {"x": "INT64", "y": "INT64", "f": "FLOAT64", "c": "BOOL"}, "u": {"a": "INT64", "b": "INT64"}}


def prove(left, right):
    return prove_equivalent_smt(left, right, schema=SCHEMA, types=TYPES)


PROVED = [
    ("SELECT SUM(x) AS s FROM t WHERE y > 0", "SELECT SUM(IF(y > 0, x, NULL)) AS s FROM t"),
    ("SELECT SUM(x) AS s FROM t WHERE y > 0", "SELECT SUM(CASE WHEN y > 0 THEN x END) AS s FROM t"),
    ("SELECT COUNT(*) AS n FROM t WHERE y > 0", "SELECT COUNT(IF(y > 0, 1, NULL)) AS n FROM t"),
    (
        "SELECT MIN(x) AS a, MAX(x) AS b, COUNT(x) AS n FROM t WHERE c",
        "SELECT MIN(IF(c, x, NULL)) AS a, MAX(IF(c, x, NULL)) AS b, COUNT(IF(c, x, NULL)) AS n FROM t",
    ),
    # part of the filter stays a filter
    ("SELECT SUM(x) AS s FROM t WHERE y > 0 AND c", "SELECT SUM(IF(c, x, NULL)) AS s FROM t WHERE y > 0"),
    # over a join, and under a HAVING on the aggregate
    ("SELECT SUM(t.x) AS s FROM t JOIN u ON t.y = u.a WHERE u.b > 1", "SELECT SUM(IF(u.b > 1, t.x, NULL)) AS s FROM t JOIN u ON t.y = u.a"),
    (
        "SELECT SUM(x) AS s FROM t WHERE y > 0 HAVING SUM(x) > 3",
        "SELECT SUM(IF(y > 0, x, NULL)) AS s FROM t HAVING SUM(IF(y > 0, x, NULL)) > 3",
    ),
]


@pytest.mark.parametrize("left, right", PROVED)
def test_a_filter_and_a_null_argument_are_the_same_global_aggregate(left, right):
    assert prove(left, right).status is SmtStatus.PROVEN_EQUIVALENT
    assert prove(right, left).status is SmtStatus.PROVEN_EQUIVALENT


NOT_PROVED = [
    # another filter, or another value on the rows the filter drops
    ("SELECT SUM(x) AS s FROM t WHERE y > 0", "SELECT SUM(IF(y >= 0, x, NULL)) AS s FROM t"),
    ("SELECT SUM(x) AS s FROM t WHERE y > 0", "SELECT SUM(IF(y > 0, x, 0)) AS s FROM t"),  # NULL against 0 when no row passes
    ("SELECT AVG(x) AS s FROM t WHERE y > 0", "SELECT AVG(IF(y > 0, x, 0)) AS s FROM t"),
    ("SELECT COUNT(*) AS n FROM t WHERE y > 0", "SELECT COUNT(*) AS n FROM t"),
    ("SELECT COUNT(*) AS n FROM t WHERE y > 0", "SELECT COUNT(IF(y > 0, NULL, 1)) AS n FROM t"),
    # a NULL condition is dropped by WHERE and kept by IS NOT FALSE
    ("SELECT SUM(x) AS s FROM t WHERE c", "SELECT SUM(IF(c IS NOT FALSE, x, NULL)) AS s FROM t"),
    ("SELECT SUM(x) AS s FROM t WHERE y > 0", "SELECT SUM(x) AS s FROM t WHERE y > 0 OR y IS NULL"),
    # with a GROUP BY the filter also decides which groups exist
    ("SELECT y, SUM(x) AS s FROM t WHERE c GROUP BY y", "SELECT y, SUM(IF(c, x, NULL)) AS s FROM t GROUP BY y"),
    # an aggregate that keeps NULLs, and DISTINCT against plain
    ("SELECT ARRAY_AGG(x) AS s FROM t WHERE y > 0", "SELECT ARRAY_AGG(IF(y > 0, x, NULL)) AS s FROM t"),
    ("SELECT SUM(DISTINCT x) AS s FROM t WHERE y > 0", "SELECT SUM(IF(y > 0, x, NULL)) AS s FROM t"),
    # a second filter on one side only
    ("SELECT MIN(x) AS s FROM t WHERE y > 0", "SELECT MIN(IF(y > 0, x, NULL)) AS s FROM t WHERE c"),
    # the duplicates of a join are part of the sum
    ("SELECT SUM(x) AS s FROM t WHERE y > 0", "SELECT SUM(IF(t.y > 0, t.x, NULL)) AS s FROM t JOIN u ON t.y = u.a"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_near_misses_are_not_proved(left, right):
    assert not prove(left, right).proven
    assert not prove(right, left).proven


def test_the_error_comparison_is_made_on_the_queries_as_written():
    # WHERE does not keep DIV from running on a row it drops; IF does. The guarded spelling refines the other.
    filtered = "SELECT SUM(DIV(10, x)) AS s FROM t WHERE y > 0"
    guarded = "SELECT SUM(IF(y > 0, DIV(10, x), NULL)) AS s FROM t"
    assert prove(filtered, guarded).errors.verdict == "refines"
    back = prove(guarded, filtered)
    assert back.errors is None or back.errors.verdict in ("unknown", "introduces")
    plain = prove("SELECT SUM(x) AS s FROM t WHERE y > 0", "SELECT SUM(IF(y > 0, x, NULL)) AS s FROM t")
    assert plain.errors.verdict == "same"


def test_a_float_sum_keeps_the_row_order_assumption():
    result = prove("SELECT SUM(f) AS s FROM t WHERE y > 0", "SELECT SUM(IF(y > 0, f, NULL)) AS s FROM t")
    assert result.proven and BASE_ASSUMPTIONS[2] in result.assumptions
