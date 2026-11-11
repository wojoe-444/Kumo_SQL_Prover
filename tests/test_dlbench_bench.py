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
_native_spec = importlib.util.spec_from_file_location("dlbench_native", _path.with_name("dlbench_native.py"))
native = importlib.util.module_from_spec(_native_spec)
sys.modules["dlbench_native"] = native
_native_spec.loader.exec_module(native)

# measured 799/807 and 161/604 on b5192d3. Of the 156 before, 21 were removed as false-proof classes (the native
# replay showed the engines disagree); the string rule narrowed with the same replay and new prover rules added 26.
FLOORS = {"parsed": 799, "exact_proven": 161}


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
    ("select hex(s1),s2 from t1; -- HEX", "SELECT HEX(s1), s2 FROM t1"),
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
    ("sqlite", "mysql", "SELECT a FROM t WHERE b = 'x'", "SELECT a FROM t WHERE b = 'x'", "MySQL string comparison"),
    ("sqlite", "mariadb", "SELECT a FROM t ORDER BY b", "SELECT a FROM t ORDER BY b", "MySQL string comparison"),
    ("mysql", "monetdb", "SELECT a FROM t", "SELECT a FROM t", "MySQL string comparison"),  # no MonetDB engine: the coarse rule
    ("mysql", "postgresql", "SELECT a / 2 FROM t", "SELECT a / 2 FROM t", "integer division"),
    ("mysql", "duckdb", "SELECT AVG(a) FROM t", "SELECT AVG(a) FROM t", "decimal arithmetic"),
    ("mysql", "postgresql", "SELECT VARIANCE(a) FROM t", "SELECT VARIANCE(a) FROM t", "population variance"),
    ("mysql", "duckdb", "SELECT STD(a) FROM t", "SELECT STDDEV(a) FROM t", "population variance"),
    ("mysql", "postgresql", "SELECT LEAST(a, 2) FROM t", "SELECT LEAST(a, 2) FROM t", "LEAST and GREATEST with NULL"),
    ("mysql", "clickhouse", "SELECT GREATEST(a, a) FROM t", "SELECT greatest(a, a) FROM t", "LEAST and GREATEST with NULL"),
    ("mysql", "postgresql", "SELECT STR_TO_DATE(b, '%Y') FROM t", "SELECT TO_TIMESTAMP(b, 'YYYY') FROM t", "PostgreSQL time zone"),
    ("mysql", "clickhouse", "SELECT MAKEDATE(a, 1) FROM t", "SELECT makeDate(a, 1) FROM t", "ClickHouse date range"),
    ("mysql", "clickhouse", "SELECT FROM_UNIXTIME(a) FROM t", "SELECT fromUnixTimestamp(a) FROM t", "timestamp range"),  # NULL before 1970 against 1970-01-01
    ("mysql", "mariadb", "SELECT FROM_UNIXTIME(a) FROM t", "SELECT FROM_UNIXTIME(a) FROM t", "timestamp range"),
    ("mysql", "mariadb", "SELECT UNIX_TIMESTAMP('2039-01-20 01:00:00')", "SELECT UNIX_TIMESTAMP('2039-01-20 01:00:00')", "timestamp range"),
    ("mysql", "mariadb", "SELECT UNIX_TIMESTAMP('1969-12-31 16:00:00')", "SELECT UNIX_TIMESTAMP('1969-12-31 16:00:00')", "timestamp range"),
    ("mysql", "mariadb", "SELECT FROM_UNIXTIME(2147483648)", "SELECT FROM_UNIXTIME(2147483648)", "timestamp range"),
    ("mysql", "mariadb", "SELECT STRCMP(b, 'a ') FROM t", "SELECT STRCMP(b, 'a ') FROM t", "MySQL and MariaDB collations"),  # NO PAD against PAD SPACE
    ("mysql", "mariadb", "SELECT MIN(b) FROM t", "SELECT MIN(b) FROM t", "MySQL and MariaDB collations"),
    ("mysql", "mariadb", "SELECT a FROM t WHERE b = 'x'", "SELECT a FROM t WHERE b = 'x'", "MySQL and MariaDB collations"),
    ("mysql", "mariadb", "SELECT a FROM t ORDER BY b", "SELECT a FROM t ORDER BY b", "MySQL and MariaDB collations"),
    ("mysql", "mariadb", "SELECT b FROM t INTERSECT SELECT b FROM t", "SELECT b FROM t INTERSECT SELECT b FROM t", "MySQL and MariaDB collations"),
    ("mysql", "mariadb", "SELECT ROW('b', 1.5) IN (ROW('b', NULL), ROW(1, 3))", "SELECT ROW('b', 1.5) IN (ROW('b', NULL), ROW(1, 3))", "row comparison"),
    ("mysql", "mariadb", "SELECT ROW(a, b) = ROW(1, 2) FROM t", "SELECT ROW(a, b) = ROW(1, 2) FROM t", "row comparison"),  # b is text
    ("mysql", "duckdb", "SELECT ROW(1, 2) = ROW(1, 2)", "SELECT ROW(1, 2) = ROW(1, 2)", "row comparison"),  # DuckDB compares structs its own way
    ("mysql", "clickhouse", "SELECT (a, 1) IN ((1, 1)) FROM t", "SELECT (a, 1) IN ((1, 1)) FROM t", "row comparison"),
    ("sqlite", "duckdb", "SELECT a / b FROM t", "SELECT a / b FROM t", "division by zero"),
    ("sqlite", "duckdb", "SELECT CAST(a AS REAL) FROM t", "SELECT CAST(a AS REAL) FROM t", "32-bit float"),
    ("sqlite", "clickhouse", "SELECT SUM(a) FROM t", "SELECT SUM(a) FROM t", "ClickHouse defaults"),
])
def test_dialect_gaps_block_proofs(source, target, left, right, gap):
    schema = ["Table: `t`\nColumns:\n(`a`, integer)\n(`b`, text)\n"]
    result = bench.decide(_pair(source, target, left, right, schema))
    assert (result["outcome"], result["detail"]) == ("unknown", f"dialect gap: {gap}")


@pytest.mark.parametrize("source, target, left, right", [
    ("mysql", "mariadb", "SELECT a FROM t WHERE a > 1", "SELECT a FROM t WHERE 1 < a"),  # numbers only
    ("mysql", "mariadb", "SELECT 'a' = 'b', a FROM t", "SELECT 'a' = 'b', a FROM t"),  # two plain literals
    ("mysql", "mariadb", "SELECT ROW(a, 2) = ROW(1, 2) FROM t", "SELECT ROW(a, 2) = ROW(1, 2) FROM t"),  # a row of numbers
    ("mysql", "mariadb", "SELECT UNIX_TIMESTAMP('2000-01-01 00:00:00')", "SELECT UNIX_TIMESTAMP('2000-01-01 00:00:00')"),
    ("mysql", "mariadb", "SELECT FROM_UNIXTIME(100000), a FROM t", "SELECT FROM_UNIXTIME(100000), a FROM t"),
    ("mysql", "postgresql", "SELECT a FROM t WHERE a > 1", "SELECT a FROM t WHERE 1 < a"),
    ("mysql", "postgresql", "SELECT ROW(a, 2) = ROW(1, 2) FROM t", "SELECT ROW(a, 2) = ROW(1, 2) FROM t"),
    ("mysql", "duckdb", "SELECT a FROM t WHERE a > 1", "SELECT a FROM t WHERE 1 < a"),
    ("mysql", "clickhouse", "SELECT COUNT(a) FROM t WHERE a > 1", "SELECT COUNT(a) FROM t WHERE 1 < a"),
    ("mysql", "postgresql", "SELECT LEAST(1, -2, 3)", "SELECT LEAST(1, -2, 3)"),  # no argument can be NULL
    ("mysql", "postgresql", "SELECT VAR_POP(a) FROM t", "SELECT VAR_POP(a) FROM t"),  # the name says which variance
    ("sqlite", "mysql", "SELECT a FROM t WHERE a > 1", "SELECT a FROM t WHERE 1 < a"),
])
def test_narrowed_gaps_let_numbers_through(source, target, left, right):
    schema = ["Table: `t`\nColumns:\n(`a`, integer)\n(`b`, text)\n"]
    pair = _pair(source, target, left, right, schema)
    assert bench.dialect_gap(pair, *bench.as_source(pair), bench.schema_of(pair)) is None
    result = bench.decide(pair)
    assert (result["outcome"], result["wrong"]) == ("proven", False)


@pytest.mark.parametrize("sql, site", [
    ("SELECT a FROM t WHERE c = 1", True),  # a column no table declares is a string
    ("SELECT a FROM t WHERE b = 'x'", True),
    ("SELECT a FROM t WHERE a = '1'", True),  # a string literal beside a number is converted
    ("SELECT a FROM t GROUP BY b", True),
    ("SELECT DISTINCT b FROM t", True),
    ("SELECT a, b FROM t ORDER BY 2", True),  # a position stands for that output
    ("SELECT b AS a2, a FROM t ORDER BY a2", True),  # and so does an output name
    ("SELECT MAX(b) FROM t", True),
    ("SELECT a FROM t WHERE b LIKE 'x%'", True),
    ("SELECT a + b FROM t", True),  # arithmetic converts the string
    ("SELECT CAST(b AS SIGNED) FROM t", True),
    ("SELECT a FROM t WHERE b", True),  # a string as a condition
    ("SELECT a FROM t WHERE a IN (SELECT b FROM t)", True),
    ("SELECT CASE b WHEN 'x' THEN 1 END FROM t", True),
    ("SELECT INSTR(b, 'x') FROM t", True),
    ("SELECT b FROM t UNION SELECT b FROM t", True),
    ("SELECT * FROM t JOIN t AS u USING (b)", True),
    ("SELECT COUNT(*) OVER (PARTITION BY b) FROM t", True),
    ("SELECT a, b FROM t WHERE a > 1 AND a IN (1, 2) ORDER BY a", False),
    ("SELECT SUM(a), COUNT(b), LENGTH(b) FROM t GROUP BY a HAVING MAX(a) > 1", False),
    ("SELECT b FROM t UNION ALL SELECT b FROM t", False),
    ("SELECT CASE WHEN a > 1 THEN b END FROM t", False),
    ("SELECT a FROM t WHERE a BETWEEN 1 AND 2 OR a IS NULL", False),
])
def test_where_a_string_may_be_compared(sql, site):
    types = bench._Types({"t": {"a": "int(11)", "b": "varchar(10)"}})
    assert bool(bench._string_site(bench.parse(sql, "mysql"), types, literals_agree=False)) is site


def test_a_column_is_a_number_only_when_every_table_says_so():
    types = bench._Types({"t": {"a": "decimal(10,2)", "b": "bigint"}, "u": {"a": "text", "b": "smallint unsigned"}})
    assert types.numeric == {"b"}


def test_column_types_with_parentheses_are_read():
    tables = dict(bench.described_tables(["Table: `t`\nColumns:\n(`a`, decimal(10,2))\n(`b`, varchar(10), nullable)\n(`c`, int)\n"]))
    assert [column[:2] for column in tables["t"]] == [("a", "decimal(10,2)"), ("b", "varchar(10)"), ("c", "int")]


@pytest.mark.parametrize("dbms, sql, name", [
    ("postgresql", "SELECT nextval('q') = nextval('q')", "nextval"),
    ("postgresql", "SELECT clock_timestamp()", "clock_timestamp"),
    ("postgresql", "SELECT a FROM t WHERE d < CURRENT_DATE", "current_date"),
    ("postgresql", "SELECT 'today'::date", "'today'"),
    ("postgresql", "SELECT gen_random_uuid()", "gen_random_uuid"),
    ("sqlite", "SELECT randomblob(4)", "randomblob"),
    ("sqlite", "SELECT date('now')", "'now'"),
    ("sqlite", "SELECT a FROM t WHERE d < DATE()", "date()"),
    ("sqlite", "SELECT strftime('%Y', 'now') - a FROM t", "'now'"),
    ("sqlite", "SELECT random() % 2", "random"),
    ("mysql", "SELECT NOW()", "now"),
    ("mysql", "SELECT UNIX_TIMESTAMP()", "unix_timestamp()"),
    ("mysql", "SELECT sysdate(6)", "sysdate"),
    ("mysql", "SELECT uuid_short()", "uuid_short"),
    ("mysql", "SELECT @a := 1", "a variable"),
    ("mysql", "SELECT CONNECTION_ID()", "connection_id"),
    ("mariadb", "SELECT RAND(3)", "rand"),
    ("duckdb", "SELECT now()", "now"),
    ("duckdb", "SELECT get_current_timestamp()", "get_current_timestamp"),
    ("clickhouse", "SELECT generateUUIDv4()", "generateuuidv4"),
    ("clickhouse", "SELECT today()", "today"),
    ("monetdb", "SELECT a FROM t WHERE d < current_timestamp", "current_timestamp"),
])
def test_a_volatile_call_is_named(dbms, sql, name):
    assert bench.volatile(sql, dbms) == name


@pytest.mark.parametrize("dbms, sql", [
    ("sqlite", "SELECT date('2020-01-01'), now FROM t"),  # a column called now
    ("sqlite", "SELECT a FROM t WHERE b = 'nowhere'"),
    ("mysql", "SELECT UNIX_TIMESTAMP('2000-01-01 00:00:00')"),
    ("mysql", "SELECT a AS today, `user` FROM t"),
    ("postgresql", "SELECT date_trunc('year', d) FROM t"),
    ("duckdb", "SELECT a FROM t -- now()"),
])
def test_other_calls_are_not_volatile(dbms, sql):
    assert bench.volatile(sql, dbms) is None


@pytest.mark.parametrize("source, target, sql", [
    ("postgresql", "duckdb", "SELECT nextval('q') = nextval('q')"),  # the provers read it as one value: never true in fact
    ("postgresql", "duckdb", "SELECT a FROM t WHERE a < random()"),
    ("sqlite", "duckdb", "SELECT a FROM t WHERE b < date('now')"),
    ("mysql", "mariadb", "select a, now() from t"),  # and not by the same-statement rule either
])
def test_a_volatile_pair_is_not_proved(source, target, sql):
    schema = ["Table: `t`\nColumns:\n(`a`, integer)\n(`b`, text)\n"]
    result = bench.decide(_pair(source, target, sql, sql, schema))
    assert result["outcome"] == "unknown" and result["detail"].startswith("volatile: ")


# -- the native replay ------------------------------------------------------------------


@pytest.fixture
def in_process(monkeypatch):
    """The replay switched on with no server: SQLite and DuckDB only."""

    monkeypatch.setenv(native.ENV, "{}")
    yield
    native.close_all()


def _replay(source, target, source_dbms="sqlite", target_dbms="duckdb", schema=None, **options):
    schema = schema or {"t": {"a": "integer", "b": "text"}}
    trees = [bench.parse(source, source_dbms), bench.parse(target, target_dbms)]
    ordered = trees[0].args.get("order") is not None
    return native.replay(source_dbms, target_dbms, *trees, schema, bench.SQLGLOT, ordered=ordered, literals=[], **options)


def test_the_replay_is_off_unless_configured(monkeypatch):
    monkeypatch.delenv(native.ENV, raising=False)
    schema = ["Table: `t`\nColumns:\n(`a`, integer)\n(`b`, text)\n"]
    pair = _pair("sqlite", "duckdb", "SELECT a FROM t WHERE a > 1", "SELECT a FROM t WHERE 1 < a", schema)
    assert native.configured() is None and native.engine("sqlite") is None
    assert bench.native_replay(pair, bench.schema_of(pair)) is None
    result = bench.decide(pair)
    assert result["outcome"] == "proven" and "replay" not in result


def test_the_configuration_must_be_an_object(monkeypatch):
    monkeypatch.setenv(native.ENV, '"mysql"')
    with pytest.raises(ValueError):
        native.configured()


def test_a_server_that_is_not_named_is_not_used(in_process):
    assert native.engine("mysql") is None and native.engine("monetdb") is None
    assert _replay("SELECT a FROM t", "SELECT a FROM t", "mysql", "postgresql")["skipped"] == "no mysql engine"


def test_a_proved_pair_is_replayed_on_both_engines(in_process):
    schema = ["Table: `t`\nColumns:\n(`a`, integer)\n(`b`, text)\n"]
    result = bench.decide(_pair("sqlite", "duckdb", "SELECT a FROM t WHERE a > 1", "SELECT a FROM t WHERE 1 < a", schema))
    assert (result["outcome"], result["wrong"]) == ("proven", False)
    assert result["replay"] == {"route": "sqlite>duckdb", "compared": native.TRIALS, "skipped": None}


def test_a_difference_between_the_engines_is_found(in_process):
    # one text, two answers: SQLite divides whole numbers to a whole number, DuckDB to a fraction
    found = _replay("SELECT a / 2 FROM t", "SELECT a / 2 FROM t")
    assert found["difference"] and found["difference"].startswith("sqlite>duckdb: ")
    # and so are a wrong translation's
    assert _replay("SELECT a FROM t WHERE a > 1", "SELECT a FROM t WHERE a >= 1")["difference"]
    assert _replay("SELECT COUNT(*) FROM t", "SELECT COUNT(a) FROM t")["difference"]


def test_a_contradicted_proof_is_wrong(monkeypatch):
    schema = ["Table: `t`\nColumns:\n(`a`, integer)\n(`b`, text)\n"]
    pair = _pair("sqlite", "duckdb", "SELECT a FROM t WHERE a > 1", "SELECT a FROM t WHERE 1 < a", schema)
    answer = {"route": "sqlite>duckdb", "compared": 3, "difference": "sqlite>duckdb: [] against [(1,)]", "skipped": None}
    monkeypatch.setattr(bench, "native_replay", lambda pair, schema: answer)
    result = bench.decide(pair)
    assert result["wrong"] and result["detail"].startswith("sqlite>duckdb: ")


def test_the_same_query_is_no_difference(in_process):
    for sql in ("SELECT a, b FROM t WHERE a > 0", "SELECT b, COUNT(*) FROM t GROUP BY b"):
        found = _replay(sql, sql)
        assert found["difference"] is None and found["compared"] > 0, sql
    # an ordered result is compared as a list: SQLite puts NULL first and DuckDB last unless told
    assert _replay("SELECT a FROM t ORDER BY a, b", "SELECT a FROM t ORDER BY a NULLS FIRST, b NULLS FIRST")["difference"] is None
    assert _replay("SELECT a FROM t ORDER BY a, b", "SELECT a FROM t ORDER BY a, b")["difference"]


def test_ties_in_the_order_are_not_a_difference(in_process):
    # the first row by a alone is any of the tied rows: only data without a tie is compared
    same = _replay("SELECT b FROM t ORDER BY a LIMIT 1", "SELECT b FROM t ORDER BY a NULLS FIRST LIMIT 1", trials=200)
    assert same["difference"] is None and 0 < same["compared"] < 200
    # a real difference is still found, on data where a tells the rows apart
    assert _replay("SELECT b FROM t ORDER BY a LIMIT 1", "SELECT b FROM t ORDER BY a DESC NULLS FIRST LIMIT 1", trials=200)["difference"]


def test_the_order_keys_are_read_out():
    probe, positions = native.tie_probe(bench.parse("SELECT b, a AS n FROM t ORDER BY n, 1, a + 1 DESC LIMIT 3", "sqlite"))
    assert positions == [1, 0, 2]
    assert probe.sql(dialect="sqlite") == "SELECT b, a AS n, a + 1 AS dlbench_key_2 FROM t ORDER BY n, 1, a + 1 DESC"
    assert native.tie_probe(bench.parse("SELECT * FROM t ORDER BY a", "sqlite")) is None
    assert native.tie_probe(bench.parse("SELECT a FROM t", "sqlite")) is None


@pytest.mark.parametrize("source, target, schema, reason", [
    ("SELECT a FROM t LIMIT 1", "SELECT a FROM t LIMIT 1", None, "LIMIT without ORDER BY"),
    ("SELECT a FROM u", "SELECT a FROM u", None, "no definition of table u"),
    ("SELECT a FROM t", "SELECT a FROM t", {"t": {"a": "blob"}}, "a column type of t is not replayed"),
    ("INSERT INTO t VALUES (1, 'a')", "INSERT INTO t VALUES (1, 'a')", None, "not a query"),
    ("SELECT * FROM t ORDER BY a", "SELECT * FROM t ORDER BY a", None, "the ORDER BY keys cannot be read out"),
])
def test_what_is_not_replayed(in_process, source, target, schema, reason):
    found = _replay(source, target, schema=schema)
    assert (found["compared"], found["difference"], found["skipped"]) == (0, None, reason)


def test_a_query_an_engine_refuses_is_not_a_difference(in_process):
    found = _replay("SELECT a FROM t", "SELECT no_such_function(a) FROM t")
    assert found["difference"] is None and found["compared"] == 0 and found["skipped"].startswith("refused: ")


@pytest.mark.parametrize("declared, kind", [
    ("", "text"), ("varchar(10)", "text"), ("TEXT", "text"), ("int(11)", "int"), ("bigint unsigned", "int"), ("INTEGER", "int"),
    ("decimal(10,2)", "real"), ("double", "real"), ("REAL", "real"), ("date", "date"), ("datetime(6)", "datetime"),
    ("timestamp", "datetime"), ("time", None), ("blob", None), ("varbinary(4)", None), ("enum('a','b')", None), ("year", None),
    ("json", None), ("bit(1)", None), ("point", None),
])
def test_declared_types_the_replay_loads(declared, kind):
    assert native.kind_of(declared) == kind


def test_replayed_values_are_compared_in_one_form():
    import datetime
    import decimal

    assert native.normal(decimal.Decimal("1.50")) == native.normal(1.5) == 1.5
    assert native.normal(True) == 1 and native.normal(b"ab") == "ab" and native.normal(b"\xff") == "0xff"
    assert native.normal(datetime.date(2017, 1, 2)) == "2017-01-02"
    assert native.normal(datetime.datetime(2017, 1, 2, 3, 4, 5)) == "2017-01-02 03:04:05"
    assert native.normal("2017") != native.normal(2017)  # text is not a number
    assert native.normal(float("nan")) == "nan" and native.normal(0.1 + 0.2) == 0.3


def test_replay_rows_hold_the_declared_kinds():
    import random

    kinds = {"t": {"a": "int", "b": "real", "c": "text", "d": "date", "e": "datetime"}}
    seen = [row for seed in range(100) for row in native.random_rows(kinds, random.Random(seed), [0.5, 9, "x", "2020-02-02"])["t"]]
    assert all(row[0] is None or isinstance(row[0], int) for row in seen)
    assert any(row[1] == 0.5 for row in seen) and any(row[2] == "a " for row in seen) and any(row[3] == "2020-02-02" for row in seen)
    assert all(row[4] is None or len(row[4]) == 19 for row in seen)


# -- proofs a native replay contradicted ------------------------------------------------


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "native_contradictions.json"
    monkeypatch.setattr(bench, "CONTRADICTIONS", path)
    bench.contradicted.cache_clear()
    yield path
    bench.contradicted.cache_clear()


def test_a_listed_contradiction_is_never_counted(ledger):
    schema = ["Table: `t`\nColumns:\n(`a`, integer)\n(`b`, text)\n"]
    pair = _pair("sqlite", "duckdb", "SELECT a FROM t WHERE a > 1", "SELECT a FROM t WHERE 1 < a", schema)
    assert bench.decide(pair)["outcome"] == "proven"
    ledger.write_text('{"%s": {"route": "sqlite>duckdb"}}' % pair.id, encoding="utf-8")
    bench.contradicted.cache_clear()
    result = bench.decide(pair)
    assert (result["outcome"], result["detail"], result["wrong"]) == ("unknown", "contradicted natively: sqlite>duckdb", False)


def test_contradictions_are_recorded_by_id_only(ledger, in_process):
    import json

    results = [
        {"id": "BUTTERTrans/duckdb/9001", "wrong": True, "replay": {"route": "sqlite>duckdb", "compared": 4, "skipped": None},
         "detail": "sqlite>duckdb: [(1,)] against [] on {'t': [(1, 'made up')]}"},
        {"id": "BUTTERTrans/duckdb/9002", "wrong": False, "replay": {"route": "sqlite>duckdb", "compared": 40, "skipped": None}, "detail": ""},
        {"id": "BIRDTrans/duckdb/9003", "wrong": True, "detail": "SQLite: {...}"},  # found without the native replay: a bug to fix, not a line
    ]
    assert bench.record_contradictions(results) == 1
    listed = json.loads(ledger.read_text(encoding="utf-8"))
    assert list(listed) == ["BUTTERTrans/duckdb/9001"]
    assert set(listed["BUTTERTrans/duckdb/9001"]) == {"route", "engines", "recorded"}
    assert set(listed["BUTTERTrans/duckdb/9001"]["engines"]) == {"sqlite", "duckdb"}
    assert "made up" not in ledger.read_text(encoding="utf-8")
    assert bench.contradicted() == listed


def test_the_results_say_what_was_replayed(in_process, monkeypatch):
    monkeypatch.delenv(native.ENV)
    assert bench.replay_sentence([{"outcome": "proven", "wrong": False}]).startswith("No native replay in this run")
    monkeypatch.setenv(native.ENV, "{}")
    results = [
        {"outcome": "proven", "wrong": False, "replay": {"route": "sqlite>duckdb", "compared": 40, "skipped": None}},
        {"outcome": "proven", "wrong": True, "replay": {"route": "sqlite>duckdb", "compared": 2, "skipped": None}},
        {"outcome": "proven", "wrong": False, "replay": {"route": "sqlite>monetdb", "compared": 0, "skipped": "no monetdb engine"}},
        {"outcome": "proven", "wrong": False, "replay": {"route": "mysql>mariadb", "compared": 0, "skipped": "not a query"}},
        {"outcome": "unknown", "wrong": False},
    ]
    sentence = bench.replay_sentence(results)
    assert "replayed 2 of the 4 proofs" in sentence and "sqlite>duckdb 2" in sentence and ": 1 differences." in sentence
    assert "Not replayed: 1 no MonetDB engine, 1 not a query." in sentence and "duckdb " in sentence and "sqlite " in sentence


def test_the_pinned_contradictions_name_pairs_of_the_subset():
    ids = {p.id for p in bench.load_pairs()}
    listed = bench.json.loads(bench.CONTRADICTIONS.read_text(encoding="utf-8")) if bench.CONTRADICTIONS.exists() else {}
    assert set(listed) <= ids
    assert all(set(entry) == {"route", "engines", "recorded"} for entry in listed.values())


def test_the_subset_holds_its_floors():
    results = bench.run(bench.load_pairs(), jobs=1)
    assert not [r["id"] for r in results if r["wrong"]]
    assert sum(r["parsed"] for r in results) >= FLOORS["parsed"]
    assert sum(r["outcome"] == "proven" for r in results if r["label"] == "exact") >= FLOORS["exact_proven"]
