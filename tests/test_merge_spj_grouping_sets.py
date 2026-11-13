"""``_merge_spj_source`` under GROUPING SETS / ROLLUP / CUBE (found by ``tools/rule_fuzz.py``, seed 77062, case 1306).

Folding a derived table into the select that groups it renames every outer column to the inner expression it reads.
Under grouping sets that is not always the same query: a grouping column is NULL in a subtotal row that does not list
it, so two outer columns that read one inner expression stop being two grouping expressions, and one derived column
spelled both bare and qualified is written once. The rule now declines those selects and still folds the others.
"""

import sys
from pathlib import Path

import pytest
from sqlglot import parse_one

pytest.importorskip("z3")
duckdb = pytest.importorskip("duckdb")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import rule_fuzz as fuzz  # noqa: E402

from kumo_sql_prover.algebraic_equivalence import _merge_spj_source  # noqa: E402

SCHEMA = {"p": [["id", "INT64"], ["tid", "INT64"]]}
KEYED = {"p": {"keys": [["id"]], "not_null": ["id"]}}

DECLINED = [
    # the same derived column read bare and qualified in the grouping sets
    "SELECT a.c1 AS c2 FROM (SELECT b.id AS c1 FROM p AS b) AS a GROUP BY GROUPING SETS ((c1, a.c1), (c1), ())",
    # two outer columns, one inner expression: c2 is NULL in the set (c1) before the merge and equal to it after
    "SELECT a.c2 FROM (SELECT b.id AS c1, b.id AS c2 FROM p AS b) AS a GROUP BY GROUPING SETS ((a.c1, a.c2), (a.c1), ())",
    "SELECT a.c2, COUNT(*) FROM (SELECT b.id AS c1, b.id AS c2 FROM p AS b) AS a GROUP BY ROLLUP (a.c1, a.c2)",
    "SELECT a.c2, COUNT(*) FROM (SELECT b.id AS c1, b.id AS c2 FROM p AS b) AS a GROUP BY CUBE (a.c1, a.c2)",
]
FOLDED = [
    "SELECT COUNT(*) FROM (SELECT b.id AS c1, b.tid AS c2 FROM p AS b) AS a GROUP BY GROUPING SETS ((a.c1, a.c2), (a.c1), ())",
    "SELECT a.c1, COUNT(*) FROM (SELECT b.id AS c1 FROM p AS b WHERE b.tid > 0) AS a GROUP BY ROLLUP (a.c1)",
    "SELECT a.c1, SUM(a.c2) FROM (SELECT b.id AS c1, b.tid AS c2 FROM p AS b) AS a GROUP BY a.c1",
]


def merged(sql):
    return _merge_spj_source(parse_one(sql, read="bigquery"))


@pytest.mark.parametrize("sql", DECLINED)
def test_a_grouping_sets_select_that_the_fold_would_change_is_left_alone(sql):
    assert merged(sql) is None


@pytest.mark.parametrize("sql", FOLDED)
def test_other_selects_are_still_folded(sql):
    result = merged(sql)
    assert result is not None and "(SELECT" not in result.sql()


@pytest.mark.parametrize("sql", FOLDED)
def test_a_fold_that_happens_keeps_the_result(sql):
    case = {"sql": sql, "dialect": "bigquery", "schema": SCHEMA, "constraints": KEYED, "options": {}, "source": "test"}
    result = fuzz.check_case(case, only={"_merge_spj_source"})
    assert all(f["status"] != "differs" for f in result["fires"]), result


def test_the_reduced_fuzz_case_no_longer_differs():
    sql = "SELECT a.c1 AS c2 FROM (SELECT b.id AS c1 FROM p AS b GROUP BY b.id) AS a GROUP BY ROLLUP (c1, a.c1)"
    case = {"sql": sql, "dialect": "bigquery", "schema": SCHEMA, "constraints": KEYED, "options": {}, "source": "gen:77062:1306"}
    result = fuzz.check_case(case, only={"_merge_spj_source"})
    assert all(f["status"] != "differs" for f in result["fires"]), result
