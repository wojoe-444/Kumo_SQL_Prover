"""The float regrouping check and the SUM argument types on ``LATERAL (SELECT ..)`` (float_regrouping.py,
sum_argument_types.py).

Calcite writes every join of a rewrite as ``(SELECT * FROM a WHERE ..) AS x, LATERAL (SELECT * FROM b WHERE ..) AS y``
and nests the next join around it. Both modules gave up on a lateral source: no column of such a query had a type,
and every column read through the star of such a join counted as a partial sum. A proof whose other side is the flat
join then looked like one that folds a sum of partial sums, and was refused. A lateral subquery outputs its select
items like any derived table, and a name it takes from the sources before it is found in the select around it.
"""

import pytest
import sqlglot
from sqlglot import exp

pytest.importorskip("z3")

from kumo_sql_prover import float_regrouping, sum_argument_types
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic

SCHEMA = {"crew": ["id", "team", "pay"], "site": ["spot", "town"]}


def declared(kind):
    return {"crew": {"id": "INT64", "team": "STRING", "pay": kind}, "site": {"spot": "INT64", "town": "STRING"}}


FLOATS, WHOLE, FIXED = declared("FLOAT64"), declared("INT64"), declared("DECIMAL(7, 2)")

JOINED = "(SELECT * FROM (SELECT * FROM crew WHERE pay > 1) AS a, LATERAL (SELECT * FROM site WHERE spot > 2) AS l WHERE a.id = l.spot) AS c"
PARTIAL = "(SELECT team, SUM(pay) AS pay FROM crew GROUP BY team) AS p"

# (the query, whether its SUM or AVG may add a FLOAT64 partial sum, whether its argument has the type of crew.pay)
READS = [
    # the lateral subquery holds no sum, and the column is the base table's
    (f"SELECT c.town, SUM(c.pay) FROM {JOINED} GROUP BY c.town", False, True),
    (f"SELECT town, AVG(pay) FROM {JOINED} GROUP BY town", False, True),
    ("SELECT SUM(a.pay) FROM crew AS a, LATERAL (SELECT * FROM site WHERE spot = a.id) AS l", False, True),
    ("SELECT SUM(pay) FROM crew AS a CROSS JOIN LATERAL (SELECT town FROM site WHERE spot = a.id) AS l", False, True),
    ("SELECT SUM(pay) FROM crew AS a LEFT JOIN LATERAL (SELECT town FROM site WHERE spot = a.id) AS l ON TRUE", False, True),
    ("SELECT SUM(l.pay) FROM site AS s, LATERAL (SELECT pay FROM crew WHERE id = s.spot) AS l", False, True),
    # the lateral subquery holds, or reads, a partial sum
    (f"SELECT SUM(l.pay) FROM site AS s, LATERAL (SELECT * FROM {PARTIAL} WHERE p.team = s.town) AS l", True, True),
    (f"SELECT SUM(l.pay) FROM site AS s, LATERAL (SELECT p.pay FROM {PARTIAL} WHERE p.team = s.town) AS l", True, True),
    ("SELECT SUM(l.v) FROM site AS s, LATERAL (SELECT SUM(pay) AS v FROM crew WHERE id = s.spot) AS l", True, True),
    # a name the lateral subquery takes from a source before it: found in the select around it, never typed
    (f"SELECT SUM(v) FROM {PARTIAL}, LATERAL (SELECT p.pay AS v) AS l", True, False),
    (f"SELECT SUM(v) FROM {PARTIAL}, LATERAL (SELECT pay AS v FROM site WHERE town = p.team) AS l", True, False),
    # a column list on the lateral subquery gives positions: no output of it is a sum, and none is typed
    (f"SELECT SUM(x) FROM {PARTIAL}, LATERAL (SELECT * FROM site) AS l (x, y)", False, False),
    # a lateral function call cannot be read: only a name qualified with another source is placed
    ("SELECT SUM(g) FROM crew AS a, LATERAL generate_series(1, a.id) AS g", True, False),
    ("SELECT SUM(pay) FROM crew AS a, LATERAL generate_series(1, a.id) AS g", True, False),
    ("SELECT SUM(a.pay) FROM crew AS a, LATERAL generate_series(1, a.id) AS g", False, True),
    ("SELECT SUM(a.pay) FROM crew AS a, (VALUES (1.5)) AS v (x)", False, True),
    ("SELECT SUM(pay) FROM crew AS a, (VALUES (1.5)) AS v (x)", False, False),
]


@pytest.mark.parametrize("sql, sums, typed", READS, ids=[sql for sql, _, _ in READS])
def test_a_column_read_beside_or_through_a_lateral_subquery(sql, sums, typed):
    tree = sqlglot.parse_one(sql, read="postgres")
    assert float_regrouping.sum_of_sums(tree, FLOATS, SCHEMA) is sums
    assert float_regrouping.sum_of_sums(tree, FLOATS, None) is sums
    call = tree.find(exp.Sum, exp.Avg)
    assert sum_argument_types.is_exact(call, FLOATS, SCHEMA) is False
    assert sum_argument_types.is_exact(call, WHOLE, SCHEMA) is typed
    assert sum_argument_types.is_exact(call, FIXED, SCHEMA) is typed
    assert sum_argument_types.is_int64(call, WHOLE, SCHEMA) is typed
    assert sum_argument_types.is_int64(call, FIXED, SCHEMA) is False


def test_a_name_another_source_gives_as_a_column_is_not_skipped():
    # UNNEST(..) AS v names a column, and v.pay reads a field of it: the type of crew.pay says nothing about it
    tree = sqlglot.parse_one("SELECT SUM(v.pay) FROM crew AS a, UNNEST(a.items) AS v", read="bigquery")
    assert sum_argument_types.is_exact(tree.find(exp.Sum), WHOLE, SCHEMA) is False
    tree = sqlglot.parse_one("SELECT SUM(x.pay) FROM crew AS a, (VALUES (1.5)) AS v (x)", read="postgres")
    assert sum_argument_types.is_exact(tree.find(exp.Sum), WHOLE, SCHEMA) is False


def prove(left, right, types):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=types, compare_names=False, dialect="postgres", timeout_ms=5000)


FLAT = "crew AS a JOIN site AS l ON a.id = l.spot WHERE a.pay > 1 AND l.spot > 2"


@pytest.mark.parametrize("types", [FLOATS, WHOLE, FIXED, None], ids=["FLOAT64", "INT64", "DECIMAL", "no types"])
@pytest.mark.parametrize("call", ["SUM", "AVG"])
def test_a_join_written_with_lateral_is_proved_equal_to_the_flat_join(call, types):
    # ROLLUP is written out as one select per grouping, so the number of calls changes on both sides
    left = f"SELECT town, {call}(pay) FROM {JOINED} GROUP BY ROLLUP (town)"
    right = f"SELECT l.town, {call}(a.pay) FROM {FLAT} GROUP BY ROLLUP (l.town)"
    assert prove(left, right, types).proven


@pytest.mark.parametrize("types", [FLOATS, WHOLE, FIXED, None], ids=["FLOAT64", "INT64", "DECIMAL", "no types"])
def test_an_average_of_sums_kept_on_both_sides_is_proved(types):
    left = f"SELECT town, AVG(total) FROM (SELECT town, id, SUM(pay) AS total FROM {JOINED} GROUP BY town, id) AS s GROUP BY town"
    right = f"SELECT town, AVG(total) FROM (SELECT l.town, a.id, SUM(a.pay) AS total FROM {FLAT} GROUP BY l.town, a.id) AS s GROUP BY town"
    assert prove(left, right, types).proven


FOLDED = f"SELECT SUM(p.pay) FROM {PARTIAL}, LATERAL (SELECT * FROM site WHERE spot > 2) AS l WHERE p.team = l.town"
ONE_SUM = "SELECT SUM(c.pay) FROM crew AS c JOIN site AS l ON c.team = l.town WHERE l.spot > 2"


def test_a_partial_sum_beside_a_lateral_subquery_stays_unproved_for_floats():
    result = prove(FOLDED, ONE_SUM, FLOATS)
    assert not result.proven and result.reason == float_regrouping.DECLINED


@pytest.mark.parametrize("types", [WHOLE, FIXED], ids=["INT64", "DECIMAL"])
def test_a_partial_sum_beside_a_lateral_subquery_is_folded_for_exact_types(types):
    assert prove(FOLDED, ONE_SUM, types).proven
