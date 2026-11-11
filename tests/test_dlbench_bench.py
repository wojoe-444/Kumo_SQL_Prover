"""DLBench translation pairs (pinned subset): parse and proof floors, no proof contradicted by running it.

The data is pinned in tests/fixtures/dlbench (see tools/dlbench_bench.py). ``FLOORS`` only ever goes up.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.eval

pytest.importorskip("z3")
pytest.importorskip("duckdb")

_path = Path(__file__).resolve().parent.parent / "tools" / "dlbench_bench.py"
_spec = importlib.util.spec_from_file_location("dlbench_bench", _path)
bench = importlib.util.module_from_spec(_spec)
sys.modules["dlbench_bench"] = bench
_spec.loader.exec_module(bench)

FLOORS = {"parsed": 799, "exact_proven": 156}  # measured 799/807 and 156/604


def _pair(source_dbms, target_dbms, source, target, schema=(), renames=None):
    return bench.Pair("BIRDTrans", 1, "db", source_dbms, target_dbms, source, target, "exact_equivalence", list(schema), renames or {})


def test_the_subset_has_every_approximate_pair():
    pairs = bench.load_pairs()
    labels = [p.label for p in pairs]
    assert (len(pairs), labels.count("approximate")) == (807, 203)
    assert {p.label_raw for p in pairs} == {"exact_equivalence", "approximate_equivalence", "Approximate equivalence"}


def test_renamed_columns_are_read_back():
    schema = ["Table: `playstore`\nColumns:\n(`App`, text)\n(`Type`, text)\n"]
    pair = _pair("sqlite", "postgresql", "SELECT App FROM playstore WHERE Type = 'Free'",
                 'SELECT "App" FROM "playstore" WHERE "_Type" = \'Free\';', schema, {"_Type": "Type"})
    left, right = bench.as_source(pair)
    assert left == right
    assert bench.decide(pair)["outcome"] == "proven"


ORDERS = [
    "Table: `Sales Orders`\nColumns:\n(`OrderNumber`, text)\n(`Order Date`, text)\n(`_CustomerID`, integer)\n(`year`, integer)\n",
    "Table: `Customers`\nColumns:\n(`CustomerID`, integer)\n(`Customer Names`, text)\n",
]


def test_a_rename_listed_with_its_quotes_is_read_back():
    # the fixture keeps the translated schema's own quoting: "_Type" -> Type
    schema = ["Table: `playstore`\nColumns:\n(`App`, text)\n(`Type`, text)\n"]
    pair = _pair("sqlite", "duckdb", "SELECT App FROM playstore WHERE Type = 'Free'",
                 'SELECT "App" FROM "playstore" WHERE "_Type" = \'Free\'', schema, {'"_Type"': "Type", '"App"': "App"})
    left, right = bench.as_source(pair)
    assert left == right


@pytest.mark.parametrize("target", [
    'SELECT "Order_Date" FROM "Sales_Orders" WHERE "_year" = 2018',  # a space spelled _, a reserved word given a _
    'SELECT T1."Order_Date" FROM "Sales_Orders" AS T1 WHERE T1."_year" = 2018',
    'SELECT "Sales_Orders"."Order_Date" FROM "Sales_Orders" WHERE "Sales_Orders"."_year" = 2018',
])
def test_translated_spellings_are_read_back(target):
    pair = _pair("sqlite", "postgresql", "SELECT `Order Date` FROM `Sales Orders` WHERE year = 2018", target, ORDERS)
    left, right = bench.as_source(pair)
    assert right.replace("t1.", "").replace('"sales orders".', "").replace(" AS t1", "") == left
    assert bench.decide(pair)["outcome"] == "proven"


@pytest.mark.parametrize("source, target, schema", [
    # two source columns share the spelling: which one a_b is cannot be told
    ("SELECT `a b` FROM t", 'SELECT "a_b" FROM t', ["Table: `t`\nColumns:\n(`a b`, text)\n(`a-b`, text)\n"]),
    # _customerid is a column of its own, not a spelling of customerid
    ("SELECT CustomerID FROM Customers", 'SELECT "_CustomerID" FROM "Customers"', ORDERS),
    # the translation reads another column
    ("SELECT `Order Date` FROM `Sales Orders`", 'SELECT "OrderNumber" FROM "Sales_Orders"', ORDERS),
    # no source table is spelled like this one
    ("SELECT year FROM `Sales Orders`", 'SELECT "year" FROM "Sales"', ORDERS),
])
def test_other_names_are_left_alone(source, target, schema):
    assert bench.decide(_pair("sqlite", "postgresql", source, target, schema))["outcome"] != "proven"


def test_an_output_name_is_not_read_as_a_respelled_column():
    # ORDER BY _year sorts by the output (the order number), not by the year column
    pair = _pair("sqlite", "postgresql", "SELECT OrderNumber AS n FROM `Sales Orders` ORDER BY year",
                 'SELECT "OrderNumber" AS _year FROM "Sales_Orders" ORDER BY _year', ORDERS)
    left, right = bench.as_source(pair)
    assert right.startswith('SELECT ordernumber AS _year FROM "sales orders" ORDER BY _year')
    assert bench.decide(pair)["outcome"] != "proven"


def test_a_local_name_is_not_read_as_a_respelled_table():
    pair = _pair("sqlite", "postgresql", "SELECT year FROM `Sales Orders`",
                 'WITH sales_orders AS (SELECT 1 AS year) SELECT year FROM sales_orders', ORDERS)
    assert '"sales orders"' not in bench.as_source(pair)[1]
    assert bench.decide(pair)["outcome"] != "proven"


def test_replay_data_keeps_whole_numbers_in_integer_columns():
    import random

    schema = {"t": {"a": "integer", "b": "real", "c": "text"}}
    rng = random.Random(1)
    rows = [row for _ in range(200) for row in bench._random_rows(schema, rng, [0.1, 7, "x"])["t"]]
    assert rows and all(row[0] is None or isinstance(row[0], int) for row in rows)
    assert any(row[1] == 0.1 for row in rows) and any(row[0] == 7 for row in rows)


@pytest.mark.parametrize("source, target", [
    ("select hex(s1),s2 from t1 order by s1,s2; -- HEX", "SELECT HEX(s1), s2 FROM t1 ORDER BY s1, s2"),
    ('select if(1,c1,""), if(0,c1,"") from t1', "SELECT IF(1, c1, ''), IF(0, c1, '') FROM t1;"),
    ("insert ignore t1 values (1, repeat('a',110))", "INSERT IGNORE t1 VALUES (1, REPEAT('a', 110))"),
    ("create table t1 as select to_seconds(null) as s", "CREATE TABLE t1 AS SELECT TO_SECONDS(NULL) AS s"),
    ("insert into t1 (ts) values (Unix_timestamp('2002-10-27 01:00'))", "INSERT INTO t1 (ts) VALUES (UNIX_TIMESTAMP('2002-10-27 01:00'))"),
])
def test_the_same_statement_is_decided_between_mysql_and_mariadb(source, target):
    pair = _pair("mysql", "mariadb", source, target)
    assert bench.same_statement(pair)
    result = bench.decide(pair)
    assert (result["outcome"], result["wrong"]) == ("proven", False)


def test_a_statement_the_provers_cannot_read_is_decided_by_its_tokens():
    result = bench.decide(_pair("mysql", "mariadb", "insert ignore t1 values (1, repeat('a',110))", "INSERT IGNORE t1 VALUES (1, REPEAT('a', 110))"))
    assert (result["outcome"], result["detail"]) == ("proven", "same statement")


@pytest.mark.parametrize("source, target", [
    ("select hex(s1) from t1 where s2 = 'a'", "select hex(s1) from t1 where s2 = 'A'"),  # a literal's case
    ("select hex(s1) from t1 where s2 = 1", "select hex(s1) from t1 where s2 = 2"),
    ("select regexp_instr(c1, 'a+', 1) from t1", "select regexp_instr(c1, 'a+') from t1"),  # a dropped argument
    ("select hex(s1) from t1", "select hex(s2) from t1"),
    ("select hex(s1) from t1", "select distinct hex(s1) from t1"),
    ("insert into t1 values (1, 'a')", "insert ignore into t1 values (1, 'a')"),
    ("insert into t1 values (1, 'a')", "insert into t1 values (1, 'A')"),
    ("insert into t1 values (1, 'a')", "insert into t2 values (1, 'a')"),
    ("create table t1 as select 1 as a", "create table t1 as select 1 as b"),
])
def test_a_changed_statement_is_not_the_same(source, target):
    pair = _pair("mysql", "mariadb", source, target)
    assert not bench.same_statement(pair)
    assert bench.decide(pair)["outcome"] != "proven"


@pytest.mark.parametrize("source, target", [
    ("select s1 from t1 where s2 = now()", "select s1 from t1 where s2 = now()"),  # the clock
    ("insert into t1 values (now())", "insert into t1 values (now())"),
    ("select s1, rand() from t1", "select s1, rand() from t1"),
    ("select s1, unix_timestamp() from t1", "select s1, unix_timestamp() from t1"),
    ("select s1, uuid() from t1", "select s1, uuid() from t1"),
    ("select s1, @a from t1", "select s1, @a from t1"),  # a variable
    ("select s1, @@sql_mode from t1", "select s1, @@sql_mode from t1"),
    ("select s1, connection_id() from t1", "select s1, connection_id() from t1"),
    ("select group_concat(s1) from t1", "select group_concat(s1) from t1"),  # the order of the pieces is not fixed
    ("select s1, row_number() over (order by s2) from t1", "select s1, row_number() over (order by s2) from t1"),
    ("select s1 from t1 limit 1", "select s1 from t1 limit 1"),  # any row
    ("delete from t1 limit 1", "delete from t1 limit 1"),
    ("explain select s1 from t1", "explain select s1 from t1"),  # the plan is the engine's own
    ("show tables", "show tables"),
    ("select strcmp(a, 'a ') from t1", "select strcmp(a, 'a ') from t1"),  # trailing spaces: PAD SPACE against NO PAD
])
def test_the_same_statement_rule_refuses(source, target):
    # the rule's own answer: the provers may still decide some of these (they read NOW() as one value per statement)
    assert not bench.same_statement(_pair("mysql", "mariadb", source, target))


@pytest.mark.parametrize("source_dbms, target_dbms", [
    ("mysql", "postgresql"), ("mysql", "duckdb"), ("mysql", "clickhouse"), ("postgresql", "monetdb"), ("postgresql", "duckdb"), ("sqlite", "mysql"),
])
def test_the_same_text_in_another_database_is_not_the_same_statement(source_dbms, target_dbms):
    # generate_series as a column is rows in PostgreSQL and one list in DuckDB; MonetDB's leaves out the upper bound
    pair = _pair(source_dbms, target_dbms, "insert into t1 select generate_series(1, 2)", "insert into t1 select generate_series(1, 2)")
    assert not bench.same_statement(pair)
    assert bench.decide(pair)["outcome"] != "proven"


@pytest.mark.parametrize("source, target, left, right, gap", [
    ("sqlite", "postgresql", "SELECT a FROM t WHERE b LIKE 'x%'", "SELECT a FROM t WHERE b LIKE 'x%'", "LIKE and case"),
    ("sqlite", "mysql", "SELECT a FROM t", "SELECT a FROM t", "MySQL string comparison"),
    ("sqlite", "duckdb", "SELECT a / b FROM t", "SELECT a / b FROM t", "division by zero"),
    ("sqlite", "duckdb", "SELECT CAST(a AS REAL) FROM t", "SELECT CAST(a AS REAL) FROM t", "32-bit float"),
    ("sqlite", "clickhouse", "SELECT SUM(a) FROM t", "SELECT SUM(a) FROM t", "ClickHouse defaults"),
])
def test_dialect_gaps_block_proofs(source, target, left, right, gap):
    schema = ["Table: `t`\nColumns:\n(`a`, integer)\n(`b`, text)\n"]
    result = bench.decide(_pair(source, target, left, right, schema))
    assert (result["outcome"], result["detail"]) == ("unknown", f"dialect gap: {gap}")


def test_the_subset_holds_its_floors():
    results = bench.run(bench.load_pairs(), jobs=1)
    assert not [r["id"] for r in results if r["wrong"]]
    assert sum(r["parsed"] for r in results) >= FLOORS["parsed"]
    assert sum(r["outcome"] == "proven" for r in results if r["label"] == "exact") >= FLOORS["exact_proven"]
