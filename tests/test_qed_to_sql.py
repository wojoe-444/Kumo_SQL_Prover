"""Checks for the QED Calcite fixture produced by tools/qed_to_sql.py."""

import json
from pathlib import Path

import pytest
import sqlglot

FIXTURES = Path(__file__).parent / "fixtures" / "qed"
CASES = [json.loads(line) for line in (FIXTURES / "qed_calcite_pairs.jsonl").read_text().splitlines()]


def test_fixture_loads():
    assert len(CASES) > 300
    assert len({c["name"] for c in CASES}) == len(CASES)
    for c in CASES:
        assert {"name", "sql_a", "sql_b", "ddl", "schema_id", "schemas"} <= set(c)
        # a few cases only use VALUES and declare no tables
        assert c["ddl"].startswith("CREATE TABLE") == bool(c["schemas"])


def test_summary_matches_fixture():
    summary = json.loads((FIXTURES / "summary.json").read_text())
    skipped = (FIXTURES / "qed_calcite_skipped.jsonl").read_text().splitlines()
    assert summary["converted"] == len(CASES)
    assert summary["skipped"] == len(skipped)
    assert all(json.loads(s)["reason"] for s in skipped)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_pair_parses(case):
    for key in ("sql_a", "sql_b"):
        tree = sqlglot.parse_one(case[key], read="mysql")
        assert isinstance(tree, sqlglot.exp.Query)
    for stmt in sqlglot.parse(case["ddl"], read="mysql") if case["ddl"] else []:
        assert isinstance(stmt, sqlglot.exp.Create)


def test_duckdb_runs_a_sample():
    duckdb = pytest.importorskip("duckdb")
    for case in CASES[:25]:
        con = duckdb.connect()
        for stmt in sqlglot.transpile(case["ddl"], read="mysql", write="duckdb"):
            con.execute(stmt)
        for key in ("sql_a", "sql_b"):
            con.execute(sqlglot.transpile(case[key], read="mysql", write="duckdb")[0]).fetchall()


# ---- cases the IR cannot state, translated from the Calcite plan text the QED file records ----

EMP = {
    "name": "CATALOG.SALES.EMP",
    "fields": ["EMPNO", "ENAME", "JOB", "MGR", "HIREDATE", "SAL", "COMM", "DEPTNO", "SLACKER"],
    "types": ["INTEGER", "VARCHAR", "VARCHAR", "INTEGER", "TIMESTAMP", "INTEGER", "INTEGER", "INTEGER", "BOOLEAN"],
    "nullable": [False, False, False, True, False, False, False, False, False],
    "key": [[0]],
    "guaranteed": [],
}
BEFORE = "LogicalAggregate(group=[{7}], N=[COUNT(DISTINCT $1, $2)])\n  LogicalTableScan(table=[[CATALOG, SALES, EMP]])\n"
AFTER = "LogicalAggregate(group=[{0}], N=[COUNT(DISTINCT $1, $2)])\n  LogicalProject(DEPTNO=[$7], ENAME=[$1], JOB=[$2])\n    LogicalTableScan(table=[[CATALOG, SALES, EMP]])\n"


def _group(keys, source):
    count = {"operator": "COUNT", "distinct": True, "type": "BIGINT", "ignoreNulls": False, "operand": [{"column": 1, "type": "VARCHAR"}, {"column": 2, "type": "VARCHAR"}]}
    return {"group": {"keys": [{"column": k, "type": "INTEGER"} for k in keys], "function": [count], "source": source}}


def _document(schema=EMP):
    project = {"project": {"source": {"scan": 0}, "target": [{"column": 7, "type": "INTEGER"}, {"column": 1, "type": "VARCHAR"}, {"column": 2, "type": "VARCHAR"}]}}
    return {"help": [BEFORE, AFTER], "schemas": [json.loads(json.dumps(schema))], "queries": [_group([7], {"scan": 0}), _group([0], project)]}


def _converter():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import qed_to_sql

    return qed_to_sql


def test_a_case_the_ir_cannot_state_is_translated_from_its_plan_text():
    conv = _converter()
    with pytest.raises(conv.Skip, match="multi-column"):
        conv.convert_ir(_document())
    case = conv.convert_case(_document())
    assert case["translated_from"] == "plan text"
    assert "COUNT(DISTINCT t1.c1, t1.c2)" in case["sql_a"] and case["sql_a"] != case["sql_b"]
    assert "CREATE TABLE EMP" in case["ddl"]


def test_the_plan_text_is_not_used_when_it_is_not_the_declared_table_or_type():
    conv = _converter()
    other_table = dict(EMP, nullable=[False] * 9)  # MGR declared NOT NULL: not the table the plan converter knows
    with pytest.raises(conv.Skip, match="multi-column"):
        conv.convert_case(_document(other_table))
    other_type = _document()
    other_type["queries"][1]["group"]["function"][0]["type"] = "INTEGER"  # a count typed as under another type system
    with pytest.raises(conv.Skip, match="multi-column"):
        conv.convert_case(other_type)
    no_text = _document()
    no_text["help"] = []
    with pytest.raises(conv.Skip, match="multi-column"):
        conv.convert_case(no_text)


def test_plan_text_cases_are_marked_in_the_fixture():
    summary = json.loads((FIXTURES / "summary.json").read_text())
    marked = [c["name"] for c in CASES if c.get("translated_from") == "plan text"]
    assert len(marked) == summary["converted_from_plan_text"] > 0
