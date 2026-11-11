"""Volatile calls in every dialect (``kumo_sql_prover.volatile_calls``).

A function the provers have no rule for is read as a function of its arguments, so ``F() = F()`` is
``F() IS NOT NULL``. Until 2026-11-11 each prover kept a short list of exceptions written for BigQuery, and
``nextval('q')``, ``clock_timestamp()``, ``randomblob(4)``, ``uuid_short()`` and ``sysdate(6)`` were proved that way.
In PostgreSQL ``WHERE nextval('q') = nextval('q')`` returns no row and ``WHERE nextval('q') IS NOT NULL`` every row.
"""

from __future__ import annotations

import pytest
import sqlglot

from kumo_sql_prover import volatile_calls as V
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.equivalence import prove_equivalent
from kumo_sql_prover.refute import order_dependence
from kumo_sql_prover.smt_equivalence import UNLISTED_FUNCTION_ASSUMPTION, SmtStatus, prove_equivalent_smt
from kumo_sql_prover.uexpr import prove_bag_equivalent

z3 = pytest.importorskip("z3")

SCHEMA = {"t1": ["s1", "s2"]}

# A new value at each call, or an effect.
PER_CALL = {
    "postgres": ["nextval('q')", "nextval('q'::regclass)", "currval('q')", "clock_timestamp()", "timeofday()", "random()", "gen_random_uuid()",
                 "uuid_generate_v4()", "uuidv7()", "gen_random_bytes(4)", "random_normal()", "setseed(0.1)", "pg_sleep(1)", "pg_advisory_lock(1)",
                 "pg_backend_pid()", "crypt('x', gen_salt('bf'))"],
    "mysql": ["uuid_short()", "sysdate(6)", "sysdate()", "rand()", "rand(3)", "uuid()", "random_bytes(4)", "sleep(1)", "get_lock('a', 1)",
              "benchmark(1, 1)", "load_file('x')", "uuid_to_bin(uuid())"],
    "sqlite": ["randomblob(4)", "random()"],
    "duckdb": ["random()", "uuid()", "uuidv4()", "uuidv7()", "gen_random_uuid()", "nextval('s')", "currval('s')", "setseed(0.5)"],
    "snowflake": ["random()", "random(1)", "uuid_string()", "seq4()", "seq8()", "uniform(1, 10, random())", "normal(0, 1, random())",
                  "randstr(5, random())", "s.nextval", "encrypt(a, 'k')"],
    "tsql": ["newid()", "newsequentialid()", "rand()", "crypt_gen_random(4)", "next value for s", "checksum(newid())"],
    "spark": ["rand()", "randn()", "uuid()", "shuffle(array(1, 2))", "monotonically_increasing_id()", "spark_partition_id()", "input_file_name()",
              "java_method('a', 'b')", "reflect('a', 'b')"],
    "oracle": ["sys_guid()", "dbms_random.value", "dbms_random.value(1, 2)", "dbms_random.string('a', 3)", "s.nextval", "s.currval", "rownum"],
    "bigquery": ["rand()", "generate_uuid()", "keys.new_keyset('x')", "aead.encrypt(k, 'a', 'b')"],
    "clickhouse": ["rand()", "rand64()", "generateUUIDv4()", "generateULID()", "randomString(3)", "nowInBlock()", "rowNumberInAllBlocks()", "randNormal(1, 2)"],
    "trino": ["random()", "rand()", "uuid()", "shuffle(array[1, 2])"],
}
# One value per statement, read from the clock, the session, the connection or the server.
PER_STATEMENT = {
    "postgres": ["now()", "statement_timestamp()", "transaction_timestamp()", "txid_current()", "current_user", "session_user", "user", "current_schema",
                 "current_catalog", "localtime", "localtimestamp", "current_time", "current_date", "version()", "current_database()",
                 "current_setting('x')", "inet_client_addr()", "age(ts)", "'now'::timestamp", "cast('today' as date)", "timestamp 'now'",
                 "date 'tomorrow'", "ts > 'now'", "date_trunc('day', 'now')"],
    "mysql": ["now()", "now(3)", "curdate()", "curtime()", "utc_date()", "utc_timestamp()", "localtime()", "localtimestamp", "current_timestamp(3)",
              "unix_timestamp()", "connection_id()", "last_insert_id()", "found_rows()", "row_count()", "current_user()", "user()", "database()",
              "schema()", "version()", "system_user()", "session_user()", "current_role()"],
    "sqlite": ["date('now')", "datetime('now')", "time('now')", "julianday('now')", "strftime('%s', 'now')", "unixepoch()", "unixepoch('now')",
               "date()", "datetime()", "strftime('%s')", "date('now', '-1 day')", "datetime('now', 'localtime')", "current_timestamp",
               "last_insert_rowid()", "changes()", "total_changes()"],
    "duckdb": ["now()", "current_timestamp", "get_current_timestamp()", "today()", "get_current_time()", "transaction_timestamp()", "txid_current()",
               "current_setting('x')", "version()", "current_schema()", "current_database()", "current_localtime()", "getvariable('x')",
               "current_query()", "epoch_ms(now())"],
    "snowflake": ["current_timestamp()", "sysdate()", "getdate()", "localtimestamp()", "current_date()", "current_time()", "systimestamp()",
                  "current_user()", "current_role()", "current_session()", "current_warehouse()", "current_account()", "last_query_id()"],
    "tsql": ["getdate()", "getutcdate()", "sysdatetime()", "sysutcdatetime()", "sysdatetimeoffset()", "current_timestamp", "scope_identity()",
             "suser_sname()", "user_name()", "current_user", "system_user", "host_name()", "db_name()", "error_number()", "app_name()"],
    "spark": ["current_timestamp()", "now()", "current_date()", "curdate()", "localtimestamp()", "current_timezone()", "current_user()", "user()",
              "current_database()", "version()", "unix_timestamp()"],
    "oracle": ["sysdate", "systimestamp", "current_date", "current_timestamp", "localtimestamp", "user", "uid", "userenv('sid')",
               "sys_context('userenv', 'sid')", "sessiontimezone", "dbtimezone"],
    "bigquery": ["current_timestamp()", "current_date()", "current_datetime()", "current_time()", "session_user()", "current_date('UTC')"],
    "clickhouse": ["now()", "now64()", "today()", "yesterday()", "currentUser()", "currentDatabase()", "hostName()", "version()", "uptime()",
                   "queryID()", "timezone()"],
    "trino": ["now()", "current_timestamp", "localtimestamp", "current_date", "current_user", "current_timezone()", "current_catalog",
              "current_schema", "localtime"],
}
# Functions of their arguments, and names that are columns or parameters: never refused.
PURE = {
    "bigquery": ["upper(a)", "my_udf(a)", "ds.my_udf(a)", "concat('now', a)", "case when a = 1 then 'today' else 'tomorrow' end", "a = 'now'",
                 "date('2020-01-01')", "timestamp('2020-01-01')", "cast(a as string)", "user", "t.user", "rownum", "`current_date`",
                 "farm_fingerprint(a)", "deterministic_encrypt(k, 'a', 'b')", "format_date('%Y', d)", "@p", "unix_seconds(ts)"],
    "mysql": ["unix_timestamp(c)", "date(c)", "time(c)", "ifnull(a, 1)", "from_unixtime(c)", "date_format(c, '%Y')", "uuid_to_bin(c)", "is_uuid(c)",
              "user", "a = 'now'", "concat(a, 'today')", "@v", "md5(a)", "timezone", "version"],
    "postgres": ["age(a, b)", "timezone('UTC', ts)", "pg_typeof(a)", "date_trunc('day', ts)", "to_char(ts, 'YYYY')", "uuid_generate_v5(a, b)",
                 "'2020-01-01'::date", "'epoch'::timestamp", "md5(a)", '"user"'],
    "sqlite": ["date(c)", "datetime(c, '+1 day')", "strftime('%Y', c)", "julianday(c)", "unixepoch(c)", "abs(a)", "time(c)"],
    "duckdb": ["strftime(ts, '%Y')", "epoch_ms(c)", "date_trunc('day', ts)", "hash(a)"],
    "oracle": ["nvl(a, b)", "to_date(a, 'YYYY')", "t.rownum2", "ora_hash(c)", "trunc(d)", '"USER"'],
    "tsql": ["isnull(a, b)", "datediff(day, a, b)", "checksum(a)", "@p"],
    "spark": ["unix_timestamp(c)", "from_unixtime(c)", "hash(a)", "to_date(c)"],
    "snowflake": ["to_date(c)", "iff(a, b, c)", "hash(a)", "t.nextvalx"],
}


def _cases(table):
    return [(dialect, call) for dialect, calls in table.items() for call in calls]


def _tree(dialect, call):
    return sqlglot.parse_one(f"select {call} from t", read=dialect)


@pytest.mark.parametrize("dialect,call", _cases(PER_CALL))
def test_a_call_that_draws_a_new_value_is_listed(dialect, call):
    found = {kind for _, kind in V.sites(_tree(dialect, call), dialect)}
    assert V.PER_CALL in found, (dialect, call, found)


@pytest.mark.parametrize("dialect,call", _cases(PER_STATEMENT))
def test_a_call_that_reads_the_clock_or_the_session_is_listed(dialect, call):
    assert V.contains(_tree(dialect, call), dialect), (dialect, call)
    assert "nondeterministic" in V.reason(_tree(dialect, call), dialect)


@pytest.mark.parametrize("dialect,call", _cases(PURE))
def test_a_function_of_its_arguments_is_not_listed(dialect, call):
    assert V.reason(_tree(dialect, call), dialect) is None, (dialect, call)


def test_the_two_kinds_are_told_apart():
    assert V.kind(_tree("mysql", "now()").selects[0], "mysql") == V.PER_STATEMENT
    assert V.kind(_tree("mysql", "sysdate()").selects[0], "mysql") == V.PER_CALL  # MySQL reads the clock at each call of SYSDATE
    assert V.kind(_tree("postgres", "nextval('q')").selects[0], "postgres") == V.PER_CALL
    assert V.kind(_tree("mysql", "unix_timestamp()").selects[0], "mysql") == V.PER_STATEMENT
    assert V.kind(_tree("mysql", "unix_timestamp(c)").selects[0], "mysql") is None
    assert V.kind(_tree("postgres", "age(a)").selects[0], "postgres") == V.PER_STATEMENT
    assert V.kind(_tree("postgres", "age(a, b)").selects[0], "postgres") is None


def test_a_clock_string_counts_only_where_it_is_read_as_a_time():
    assert V.contains(_tree("sqlite", "date('now')"), "sqlite")
    assert V.contains(_tree("bigquery", "cast('now' as timestamp)"), "bigquery")
    assert not V.contains(_tree("bigquery", "a = 'now'"), "bigquery")
    assert V.contains(_tree("postgres", "a = 'now'"), "postgres")  # the literal takes the type of a, which may be a timestamp
    assert not V.contains(_tree("postgres", "a = 'noon'"), "postgres")


def test_a_keyword_value_is_a_column_where_the_dialect_says_so():
    assert not V.contains(_tree("bigquery", "user"), "bigquery")
    assert V.contains(_tree("postgres", "user"), "postgres")
    assert not V.contains(_tree("postgres", '"user"'), "postgres")
    assert V.contains(_tree("oracle", "rownum"), "oracle") and not V.contains(_tree("bigquery", "rownum"), "bigquery")
    assert V.contains(_tree("snowflake", "seq.nextval"), "snowflake") and not V.contains(_tree("snowflake", "nextval"), "snowflake")


def test_unlisted_calls_are_named():
    tree = sqlglot.parse_one("select my_udf(a), ds.other(b), upper(c), nextval('q') from t", read="bigquery")
    assert V.unlisted_calls(tree) == ("DS.OTHER", "MY_UDF")
    assert V.read("select my_udf(a) from t", "bigquery") == (None, ("MY_UDF",))
    assert V.read("select rand(", "bigquery") == (None, ())  # unreadable text is the prover's to report
    why, names = V.read("select my_udf(a), now() from t", "mysql")
    assert "NOW()" in why.upper() and names == ()


# --- the provers ---------------------------------------------------------------------------------------------------
REPORTED = [
    ("postgres", "nextval('q')"),
    ("postgres", "clock_timestamp()"),
    ("sqlite", "randomblob(4)"),
    ("mysql", "uuid_short()"),
    ("mysql", "sysdate(6)"),
    ("mysql", "now()"),
    ("mysql", "unix_timestamp()"),
    ("mysql", "connection_id()"),
    ("sqlite", "date('now')"),
    ("duckdb", "now()"),
    ("duckdb", "nextval('q')"),
    ("spark", "monotonically_increasing_id()"),
    ("spark", "randn()"),
    ("snowflake", "seq4()"),
    ("oracle", "sys_guid()"),
    ("oracle", "systimestamp"),
    ("mysql", "rand()"),
    ("bigquery", "generate_uuid()"),
]
SOLVER_PROVERS = [prove_equivalent_smt, prove_equivalent_algebraic, prove_bag_equivalent]
# One value for the whole statement in an engine that promises it: proved only under the condition that both queries
# read the same one (``kumo_sql_prover.statement_values``, tests/test_statement_values.py), never outright.
FIXED_FOR_THE_STATEMENT = {("mysql", "now()"), ("mysql", "unix_timestamp()"), ("mysql", "connection_id()"), ("duckdb", "now()")}


def _expected(dialect, call):
    return SmtStatus.PROVEN_CONDITIONALLY if (dialect, call) in FIXED_FOR_THE_STATEMENT else SmtStatus.NOT_PROVEN


@pytest.mark.parametrize("dialect,call", REPORTED)
@pytest.mark.parametrize("prover", SOLVER_PROVERS, ids=lambda p: p.__name__)
def test_two_calls_are_not_one_value(prover, dialect, call):
    left = f"select s1 from t1 where {call} = {call}"
    right = f"select s1 from t1 where {call} is not null"
    result = prover(left, right, schema=SCHEMA, dialect=dialect)
    assert result.status is _expected(dialect, call) and not result.proven, (dialect, call, result.reason)


@pytest.mark.parametrize("dialect,call", REPORTED)
@pytest.mark.parametrize("prover", SOLVER_PROVERS, ids=lambda p: p.__name__)
def test_the_solver_provers_do_not_prove_a_pair_that_holds_a_volatile_call_even_unchanged(prover, dialect, call):
    sql = f"select s1 from t1 where s2 = {call}"
    result = prover(sql, sql, schema=SCHEMA, dialect=dialect)
    assert result.status is _expected(dialect, call) and not result.proven, (dialect, call, result.reason)


def test_the_refusal_names_the_call():
    result = prove_equivalent_algebraic(
        "select s1 from t1 where nextval('q') = nextval('q')", "select s1 from t1 where nextval('q') is not null", schema=SCHEMA, dialect="postgres"
    )
    assert "nondeterministic" in result.reason and "NEXTVAL" in result.reason.upper()
    result = prove_equivalent_smt("select s1 from t1 where s2 > date('now')", "select s1 from t1 where s2 > date('now')", schema=SCHEMA, dialect="sqlite")
    assert "reads the clock or the session" in result.reason


@pytest.mark.parametrize(
    "left,right",
    [
        # The rules drop or fold the call before the solver sees the statement; the pair as written still holds it.
        ("select s1 from t1 where rand() < 2 or true", "select s1 from t1"),
        ("select s1 from t1 where case when true then 1 else rand() end = 1", "select s1 from t1"),
    ],
)
def test_a_call_the_rules_removed_still_blocks_the_proof(left, right):
    for prover in (prove_equivalent_smt, prove_equivalent_algebraic):
        result = prover(left, right, schema=SCHEMA)
        assert result.status is SmtStatus.NOT_PROVEN and "nondeterministic" in result.reason, (prover.__name__, result.reason)


def test_a_function_the_prover_does_not_know_is_an_assumption_of_the_proof():
    left = "select s1 from t1 where my_udf(s1) = my_udf(s1)"
    right = "select s1 from t1 where my_udf(s1) is not null"
    prefix = UNLISTED_FUNCTION_ASSUMPTION.split("{")[0]
    for prover in (prove_equivalent_smt, prove_equivalent_algebraic):
        result = prover(left, right, schema=SCHEMA)
        assert result.proven, result.reason
        assert [a for a in result.assumptions if a.startswith(prefix)] == [prefix + "MY_UDF"], result.assumptions
    plain = prove_equivalent_algebraic("select s1 from t1 where upper(s1) = 'A'", "select s1 from t1 where 'A' = upper(s1)", schema=SCHEMA)
    assert plain.proven and not any(a.startswith(prefix) for a in plain.assumptions)


def test_a_proof_without_such_calls_is_unchanged():
    result = prove_equivalent_algebraic("select s1 from t1 where s2 = 1 and s1 = 2", "select s1 from t1 where s1 = 2 and s2 = 1", schema=SCHEMA, dialect="mysql")
    assert result.proven


# --- the structural prover: a volatile call is accepted only identical and in place ---------------------------------------
@pytest.mark.parametrize("call", ["RAND()", "CURRENT_TIMESTAMP()", "NEXTVAL('q')", "CAST('now' AS TIMESTAMP)", "KEYS.NEW_KEYSET('AEAD_AES_GCM_256')"])
def test_the_structural_prover_keeps_an_untouched_call(call):
    sql = f"SELECT a FROM t WHERE b = {call}"
    assert prove_equivalent(sql, f"SELECT a FROM t WHERE (b = {call})").proven


@pytest.mark.parametrize("call", ["NEXTVAL('q')", "CAST('now' AS TIMESTAMP)", "CLOCK_TIMESTAMP()"])
def test_the_structural_prover_refuses_a_call_that_moved_or_changed(call):
    assert not prove_equivalent(f"SELECT a FROM t WHERE {call} = {call}", f"SELECT a FROM t WHERE {call} IS NOT NULL").proven
    assert not prove_equivalent(f"SELECT a, {call} AS v FROM t", f"SELECT a, v FROM (SELECT a, {call} AS v FROM t)").proven


# --- executed comparisons ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dialect,sql",
    [("mysql", "select now() from t"), ("postgres", "select nextval('q') from t"), ("sqlite", "select date('now') from t"), ("duckdb", "select now() from t")],
)
def test_a_difference_shown_by_a_volatile_query_is_not_a_counterexample(dialect, sql):
    assert "nondeterministic" in (order_dependence(sql, dialect) or "")
    assert order_dependence("select a from t where b = 1", dialect) is None
