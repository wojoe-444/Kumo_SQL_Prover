"""The float regrouping check on a set operation whose branches are ``SELECT *`` (float_regrouping.py).

A set operation gives its columns by position, and a star hides the positions. The check used to count every
column read from such a derived table as a possible partial sum, so ``SUM(pay)`` over
``(SELECT * FROM crew UNION ALL SELECT * FROM crew)`` looked like a sum of partial sums, and a rewrite that only
writes the columns out was refused. No position of a branch that reads base tables alone holds a sum: the check
now asks whether any output of any branch can be one.
"""

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import float_regrouping
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic

SCHEMA = {"crew": ["id", "team", "pay"], "past": ["id", "team", "pay"]}
FLOATS = {name: {"id": "INT64", "team": "STRING", "pay": "FLOAT64"} for name in SCHEMA}
WHOLE = {name: {"id": "INT64", "team": "STRING", "pay": "INT64"} for name in SCHEMA}

BOTH = "(SELECT * FROM crew UNION ALL SELECT * FROM past) AS b"
PARTIAL = "(SELECT team, SUM(pay) AS pay FROM crew GROUP BY team) AS p"

# (the query, whether some SUM or AVG of it may add a FLOAT64 partial sum)
READS = [
    # no branch holds a sum at any position
    (f"SELECT team, SUM(pay) FROM {BOTH} GROUP BY team", False),
    (f"SELECT team, AVG(pay) FROM {BOTH} GROUP BY team", False),
    ("SELECT SUM(pay) FROM (SELECT c.* FROM crew AS c UNION ALL SELECT * FROM past) AS b", False),
    ("SELECT SUM(pay) FROM (SELECT * FROM (SELECT * FROM crew) AS c UNION ALL SELECT * FROM past) AS b", False),
    ("WITH c AS (SELECT * FROM crew) SELECT SUM(pay) FROM (SELECT * FROM c UNION ALL SELECT * FROM past) AS b", False),
    ("SELECT SUM(pay) FROM (SELECT * FROM crew EXCEPT ALL SELECT * FROM past) AS b", False),
    ("SELECT SUM(v) FROM (SELECT * FROM crew UNION ALL SELECT * FROM past) AS b (k, t, v)", False),
    ("SELECT SUM(pay) FROM (SELECT *, id + 1 AS more FROM crew UNION ALL SELECT *, 0 FROM past) AS b", False),
    ("SELECT SUM(pay) FROM (SELECT * FROM crew JOIN past USING (id) UNION ALL SELECT * FROM crew JOIN past USING (id)) AS b", False),
    # a branch that can hold one, at a position the star hides
    (f"SELECT SUM(pay) FROM (SELECT * FROM {PARTIAL} UNION ALL SELECT team, pay FROM past) AS b", True),
    (f"SELECT SUM(pay) FROM (SELECT team, pay FROM past UNION ALL SELECT * FROM {PARTIAL}) AS b", True),
    (f"SELECT SUM(pay) FROM (SELECT p.* FROM {PARTIAL} UNION ALL SELECT team, pay FROM past) AS b", True),
    ("WITH p AS (SELECT team, SUM(pay) AS pay FROM crew GROUP BY team) SELECT SUM(pay) FROM (SELECT * FROM p UNION ALL SELECT team, pay FROM past) AS b", True),
    ("SELECT SUM(pay) FROM (SELECT *, (SELECT SUM(pay) FROM past) AS total FROM crew UNION ALL SELECT *, 0 FROM past) AS b", True),
    (f"SELECT SUM(pay) FROM (SELECT * FROM (SELECT * FROM {PARTIAL}) AS q UNION ALL SELECT team, pay FROM past) AS b", True),
    (f"SELECT SUM(v) FROM (SELECT * FROM {PARTIAL} UNION ALL SELECT team, pay FROM past) AS b (t, v)", True),
    # what the check cannot read counts as a sum
    ("SELECT SUM(pay) FROM (SELECT * EXCEPT (id) FROM crew UNION ALL SELECT team, pay FROM past) AS b", True),
    ("SELECT SUM(pay) FROM (SELECT * FROM UNNEST([1.5, 2.5]) AS pay UNION ALL SELECT pay FROM past) AS b", True),
]


@pytest.mark.parametrize("sql, expected", READS, ids=[sql for sql, _ in READS])
def test_a_column_past_a_star_of_a_set_operation(sql, expected):
    tree = sqlglot.parse_one(sql, read="bigquery")
    assert float_regrouping.sum_of_sums(tree, FLOATS, SCHEMA) is expected
    assert float_regrouping.sum_of_sums(tree, FLOATS, None) is expected  # the answer does not need the column order


def prove(left, right, types, dialect="bigquery"):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=types, compare_names=False, dialect=dialect, timeout_ms=5000)


WRITTEN_OUT = "(SELECT team, pay FROM crew UNION ALL SELECT team, pay FROM past) AS w"
PUSHED = "(SELECT team, SUM(pay) AS pay FROM crew GROUP BY team UNION ALL SELECT team, SUM(pay) AS pay FROM past GROUP BY team) AS s"


@pytest.mark.parametrize("types", [FLOATS, WHOLE, None], ids=["FLOAT64", "INT64", "no types"])
@pytest.mark.parametrize("call", ["SUM(pay)", "AVG(pay)", "MIN(pay)"])
@pytest.mark.parametrize("dialect", ["bigquery", "mysql", "postgres"])
def test_writing_the_columns_of_a_starred_union_out_is_proved(call, types, dialect):
    result = prove(f"SELECT team, {call} FROM {BOTH} GROUP BY team", f"SELECT team, {call} FROM {WRITTEN_OUT} GROUP BY team", types, dialect)
    assert result.proven
    assert float_regrouping.UNTYPED_ASSUMPTION not in (result.assumptions or ())


def test_a_sum_pushed_below_the_union_is_proved_for_whole_numbers():
    assert prove(f"SELECT team, SUM(pay) FROM {BOTH} GROUP BY team", f"SELECT team, SUM(pay) FROM {PUSHED} GROUP BY team", WHOLE).proven


def test_a_sum_pushed_below_the_union_stays_unproved_for_floats():
    result = prove(f"SELECT team, SUM(pay) FROM {BOTH} GROUP BY team", f"SELECT team, SUM(pay) FROM {PUSHED} GROUP BY team", FLOATS)
    assert not result.proven and result.reason == float_regrouping.DECLINED


def test_a_partial_sum_behind_a_star_of_a_union_stays_unproved_for_floats():
    starred = f"(SELECT * FROM {PARTIAL} UNION ALL SELECT * FROM (SELECT team, SUM(pay) AS pay FROM past GROUP BY team) AS q) AS s"
    result = prove(f"SELECT team, SUM(pay) FROM {WRITTEN_OUT} GROUP BY team", f"SELECT team, SUM(pay) FROM {starred} GROUP BY team", FLOATS)
    assert not result.proven
