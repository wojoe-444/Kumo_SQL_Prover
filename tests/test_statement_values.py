"""Queries that read the statement time or the session (``kumo_sql_prover.statement_values``).

``NOW()``, ``CURRENT_DATE`` and ``SESSION_USER()`` have one value for the whole statement, so two queries that read
them can be equal *when both read the same value*. The solver-based provers say so as ``PROVEN_CONDITIONALLY`` with
a condition naming the calls; they never call such a pair ``PROVEN_EQUIVALENT``. A call that may differ within one
statement (``RAND()``, ``nextval``, ``clock_timestamp()``) is still not proven at all.
"""

from __future__ import annotations

import pytest

from kumo_sql_prover import statement_values as S
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import UNLISTED_FUNCTION_ASSUMPTION, Counterexample, SmtEquivalenceResult, SmtStatus, prove_equivalent_smt
from kumo_sql_prover.uexpr import prove_bag_equivalent

z3 = pytest.importorskip("z3")

SCHEMA = {"t": ["a", "d", "u"], "s": ["a", "d"]}
TYPES = {"t": {"a": "INT64", "d": "DATE", "u": "STRING"}, "s": {"a": "INT64", "d": "DATE"}}
PROVERS = [prove_equivalent_smt, prove_equivalent_algebraic, prove_bag_equivalent]
by_name = pytest.mark.parametrize("prover", PROVERS, ids=lambda p: p.__name__)


def prove(prover, left, right, dialect="bigquery", **options):
    return prover(left, right, schema=SCHEMA, types=TYPES, dialect=dialect, **options)


def no_stand_in(result) -> bool:
    texts = [result.reason, *result.assumptions, *(c.text for c in result.conditions), *(n for c in result.conditions for n in c.columns)]
    return not any(S.PREFIX.lower() in text.lower() for text in texts)


# --- proved, and only under the condition ---------------------------------------------------------------------------------
REWRITES = [
    # (dialect, left, right, kind of condition, the calls it names)
    ("bigquery", "select a from t where d >= current_date() and a > 1", "select a from (select * from t where a > 1) where d >= current_date()", S.SAME_TIME, ("CURRENT_DATE",)),
    (
        "bigquery",
        "select a from t where d >= date_sub(current_date(), interval 7 day)",
        "with x as (select a, d from t) select a from x where d >= date_sub(current_date(), interval 7 day)",
        S.SAME_TIME,
        ("CURRENT_DATE",),
    ),
    ("bigquery", "select a from t where u = session_user()", "select a from t where session_user() = u", S.SAME_SESSION, ("SESSION_USER",)),
    ("bigquery", "select a from t where d < current_date('UTC')", "select a from t where current_date('UTC') > d", S.SAME_TIME, ("CURRENT_DATE('UTC')",)),
    ("postgres", "select a from t where d < now()", "select a from t where now() > d", S.SAME_TIME, ("CURRENT_TIMESTAMP",)),
    ("postgres", "select a from t where d < statement_timestamp()", "select a from t where statement_timestamp() > d", S.SAME_TIME, ("STATEMENT_TIMESTAMP",)),
    ("mysql", "select a from t where d < now()", "select a from t where current_timestamp > d", S.SAME_TIME, ("CURRENT_TIMESTAMP",)),
    ("mysql", "select a from t where d < curdate()", "select a from t where current_date > d", S.SAME_TIME, ("CURRENT_DATE",)),
    ("mysql", "select a from t where a < unix_timestamp()", "select a from t where unix_timestamp() > a", S.SAME_TIME, ("UNIX_TIMESTAMP",)),
    ("duckdb", "select a from t where d < now() and a = 1", "select a from t where a = 1 and d < get_current_timestamp()", S.SAME_TIME, ("CURRENT_TIMESTAMP",)),
    ("duckdb", "select a from t where d < today()", "select a from t where current_date > d", S.SAME_TIME, ("CURRENT_DATE",)),
    ("spark", "select a from t where d < current_date()", "select a from t where current_date() > d", S.SAME_TIME, ("CURRENT_DATE",)),
    ("trino", "select a from t where d < now()", "select a from t where current_timestamp > d", S.SAME_TIME, ("CURRENT_TIMESTAMP",)),
]


@pytest.mark.parametrize("dialect,left,right,kind,calls", REWRITES)
@by_name
def test_a_rewrite_that_keeps_the_call_is_proved_under_the_condition(prover, dialect, left, right, kind, calls):
    result = prove(prover, left, right, dialect)
    assert result.status is SmtStatus.PROVEN_CONDITIONALLY, result.reason
    assert not result.proven and result.conditionally_proven  # never counted as a proof
    assert [(c.kind, c.columns) for c in result.conditions] == [(kind, calls)]
    assert all(call in result.conditions[0].text and call in result.reason for call in calls)
    assert result.conditions[0].to_json()["check_sql"] == ""  # nothing to count in the warehouse
    assert result.counterexample is None and no_stand_in(result)


@by_name
def test_the_same_text_is_proved_only_under_the_condition(prover):
    sql = "select a, current_timestamp() as loaded_at from t where d = current_date()"
    result = prove(prover, sql, sql)
    assert result.status is SmtStatus.PROVEN_CONDITIONALLY and not result.proven
    assert result.conditions[0].columns == ("CURRENT_DATE", "CURRENT_TIMESTAMP")


def test_the_clock_and_the_session_are_two_conditions():
    left = "select a from t where d < current_date and u = current_user"
    right = "select a from t where current_user = u and current_date > d"
    result = prove(prove_equivalent_algebraic, left, right, "postgres")
    assert [(c.kind, c.columns) for c in result.conditions] == [(S.SAME_SESSION, ("CURRENT_USER",)), (S.SAME_TIME, ("CURRENT_DATE",))]
    assert "in the same session and at the same statement time" in result.reason


def test_a_condition_on_the_data_is_kept_next_to_it():
    left = "select a from t where d < current_date() and a is not null"
    right = "select a from t where d < current_date()"
    result = prove(prove_equivalent_algebraic, left, right, conditional=True)
    assert result.status is SmtStatus.PROVEN_CONDITIONALLY
    assert [(c.kind, c.table, c.columns) for c in result.conditions] == [("not_null", "t", ("a",)), (S.SAME_TIME, "", ("CURRENT_DATE",))]
    assert not prove(prove_equivalent_algebraic, left, right).conditionally_proven  # the data condition is found only when asked for


def test_a_function_the_prover_does_not_know_is_still_named_and_the_stand_in_is_not():
    left = "select a, my_udf(a) from t where d < current_date()"
    right = "select a, my_udf(a) from t where current_date() > d"
    result = prove(prove_equivalent_smt, left, right)
    assert result.status is SmtStatus.PROVEN_CONDITIONALLY
    prefix = UNLISTED_FUNCTION_ASSUMPTION.split("{")[0]
    assert [a for a in result.assumptions if a.startswith(prefix)] == [prefix + "MY_UDF"]
    alone = prove(prove_equivalent_smt, "select a from t where d < current_date()", "select a from t where current_date() > d")
    assert not any(a.startswith(prefix) for a in alone.assumptions)


# --- nothing is assumed about the value ------------------------------------------------------------------------------------
NOT_THE_SAME = [
    ("bigquery", "select a from t where d < current_date()", "select a from t where d <= current_date()"),
    ("bigquery", "select a from t where current_date() > date '2020-01-01'", "select a from t"),  # true today; not a fact
    ("bigquery", "select a from t where d < current_date()", "select a from t where d < current_date('UTC')"),
    ("bigquery", "select a from t where d < current_date()", "select a from t where d < date(current_timestamp())"),
    ("bigquery", "select a from t where d < current_date()", "select a from t where d < my_udf()"),
    ("postgres", "select a from t where d < now()", "select a from t where d < statement_timestamp()"),  # two clocks in a transaction
    ("mysql", "select a from t where d < now()", "select a from t where d < now(6)"),
    ("mysql", "select a from t where u = user()", "select a from t where u = current_user()"),  # the client's name and the account's
]


@pytest.mark.parametrize("dialect,left,right", NOT_THE_SAME)
@by_name
def test_two_different_reads_are_not_one_value(prover, dialect, left, right):
    result = prove(prover, left, right, dialect)
    assert result.status is SmtStatus.NOT_PROVEN, result.reason
    assert no_stand_in(result)


# --- calls that may differ within one statement stay unproven --------------------------------------------------------------
PER_CALL = [
    ("bigquery", "rand()"),
    ("bigquery", "generate_uuid()"),
    ("postgres", "random()"),
    ("postgres", "nextval('q')"),
    ("postgres", "clock_timestamp()"),
    ("postgres", "gen_random_uuid()"),
    ("mysql", "rand()"),
    ("mysql", "uuid()"),
    ("mysql", "sysdate()"),
    ("duckdb", "random()"),
    ("duckdb", "nextval('q')"),
    ("duckdb", "uuid()"),
    ("spark", "rand()"),
    ("spark", "monotonically_increasing_id()"),
    ("trino", "random()"),
    ("trino", "uuid()"),
]


@pytest.mark.parametrize("dialect,call", PER_CALL)
@by_name
def test_a_call_that_may_differ_within_a_statement_is_never_proved(prover, dialect, call):
    clock = "current_date()" if dialect in ("bigquery", "spark") else "current_date"
    pairs = [
        (f"select a from t where {call} = {call}", f"select a from t where {call} is not null"),
        (f"select a from t where u = {call}", f"select a from t where u = {call}"),  # the same text
        # next to a call that is fixed for the statement: the condition does not cover it
        (f"select a from t where d < {clock} and u = {call}", f"select a from t where u = {call} and d < {clock}"),
        (f"select a, {call} as v from t where d < {clock}", f"select a, {call} as v from t where {clock} > d"),
    ]
    for left, right in pairs:
        result = prove(prover, left, right, dialect)
        assert result.status is SmtStatus.NOT_PROVEN and not result.conditions, (left, right, result.reason)


# --- engines and spellings that are left out -------------------------------------------------------------------------------
LEFT_OUT = [
    # SQLite promises one value only within one step; the others were not checked
    ("sqlite", "select a from t where d < current_timestamp"),
    ("sqlite", "select a from t where d < date('now')"),
    ("tsql", "select a from t where d < getdate()"),
    ("snowflake", "select a from t where d < current_timestamp()"),
    ("oracle", "select a from t where d < sysdate"),
    ("clickhouse", "select a from t where d < now()"),
    # read by the engine, not written as a call
    ("postgres", "select a from t where d < 'now'"),
    ("postgres", "select a from t where d < 'today'::date"),
    ("postgres", "select a from t where u = user"),
    ("postgres", "select a from t where age(d) > interval '1 day'"),
    ("bigquery", "select a from t where u = session_user"),
    # not a value of the statement alone
    ("mysql", "select a from t where a < last_insert_id()"),
    ("mysql", "select a from t where a < found_rows()"),
]


@pytest.mark.parametrize("dialect,sql", LEFT_OUT)
def test_what_is_left_out_is_not_read_as_a_statement_value(dialect, sql):
    assert S.reading(sql, sql, dialect) is None
    for prover in PROVERS:
        result = prove(prover, sql, sql, dialect)
        assert not result.proven and not any(c.kind in S.KINDS for c in result.conditions), (prover.__name__, result.status, result.reason)


def test_only_the_dialects_with_a_documented_promise_are_in():
    assert S.DIALECTS == {"bigquery", "postgres", "mysql", "duckdb", "spark", "databricks", "trino", "presto"}


# --- the reading itself ----------------------------------------------------------------------------------------------------
def test_everything_but_the_call_stays_as_it_was_written():
    left = "select a from t where d < NOW( 6 ) and x = `now` and u = 'now()' -- now()\n and curdate() > d"
    found = S.reading(left, "select 1", "mysql")
    assert found is not None
    names = {call: name for name, _, call in found.names}
    assert found.left_sql == left.replace("NOW( 6 )", names["CURRENT_TIMESTAMP(6)"] + "()").replace("curdate()", names["CURRENT_DATE"] + "()")
    assert found.right_sql == "select 1"
    assert found.calls == {S.SAME_TIME: ("CURRENT_DATE", "CURRENT_TIMESTAMP(6)")}
    assert found.restore(f"{names['CURRENT_DATE']}() is NULL") == "CURRENT_DATE is NULL"


def test_two_spellings_of_one_value_get_one_name():
    found = S.reading("select today(), current_date, now(), transaction_timestamp() from t", "select get_current_timestamp()", "duckdb")
    assert [call for _, _, call in found.names] == ["CURRENT_DATE", "CURRENT_TIMESTAMP"]
    assert found.left_sql.count(found.names[0][0]) == 2 and found.left_sql.count(found.names[1][0]) == 2 and found.names[1][0] in found.right_sql


@pytest.mark.parametrize(
    "dialect,left",
    [
        ("bigquery", "select a as current_date from t where d < current_date"),  # an alias the scan cannot tell from the call
        ("bigquery", "select a from t where d < current_date; select 1"),  # two statements
        ("bigquery", "select a from t where d < kq_fixed_1_current_date() and d < current_date"),  # a stand-in name of its own
        ("bigquery", "select a from t where d < current_date("),  # not SQL
    ],
)
def test_a_text_the_scan_and_the_parser_read_differently_is_left_alone(dialect, left):
    assert S.reading(left, "select a from t", dialect) is None
    assert not prove(prove_equivalent_algebraic, left, left, dialect).proven


def test_a_column_or_a_function_of_that_name_is_not_the_call():
    found = S.reading("select t.current_date from t where d < current_date", "select a from t", "bigquery")
    assert found.left_sql == f"select t.current_date from t where d < {found.names[0][0]}()"
    # mydb.now() is somebody's own function: an unknown function like any other, named in the assumptions
    sql = "select a from t where d < mydb.now()"
    assert S.reading(sql, sql, "mysql") is None
    result = prove(prove_equivalent_smt, sql, sql, "mysql")
    assert result.proven and any("MYDB.NOW" in a for a in result.assumptions)


def test_a_text_without_such_a_call_is_not_parsed_here():
    assert S.reading("select a from t where d < date '2024-01-01'", "select nowhere, currently from t", "bigquery") is None
    result = prove(prove_equivalent_algebraic, "select a from t where a = 1 and d is null", "select a from t where d is null and a = 1")
    assert result.proven and not result.conditions


# --- no refutation ---------------------------------------------------------------------------------------------------------
def test_a_difference_is_not_reported_as_a_counterexample():
    calls = []

    def refuting(left_sql, right_sql, **kwargs):
        calls.append((left_sql, right_sql))
        example = Counterexample({"t": [{"a": 1}]}, [(1,)], [])
        return SmtEquivalenceResult(SmtStatus.NOT_EQUIVALENT, "the queries differ", counterexample=example)

    left, right = "select a from t where d < current_date()", "select a from t where d <= current_date()"
    result = S.proved(refuting, left, right, {"dialect": "bigquery"})
    assert S.PREFIX in calls[0][0] and S.PREFIX in calls[0][1]  # the prover saw the stand-ins, not the calls
    assert result.status is SmtStatus.NOT_PROVEN and result.counterexample is None
    assert "CURRENT_DATE" in result.reason and "when each query runs" in result.reason
    # a pair with no such call keeps its refutation
    plain = S.proved(refuting, "select a from t", "select a from t where a = 1", {"dialect": "bigquery"})
    assert plain.status is SmtStatus.NOT_EQUIVALENT and plain.counterexample is not None


def test_the_executed_search_does_not_refute_such_a_pair():
    left, right = "select a from t where d < current_date()", "select a from t where d <= current_date()"
    result = prove(prove_equivalent_algebraic, left, right, search_counterexample=True)
    assert result.status is SmtStatus.NOT_PROVEN and result.counterexample is None


# --- the proofs hold for any value the call could have -----------------------------------------------------------------------
@pytest.mark.parametrize("dialect,left,right,kind,calls", REWRITES)
def test_a_proved_rewrite_returns_the_same_rows_for_any_value_of_the_call(dialect, left, right, kind, calls):
    """Each stand-in is replaced by a literal (a date, a name, NULL) and the two queries are run on small random tables."""

    import random
    import re

    import sqlglot

    duckdb = pytest.importorskip("duckdb")
    values = {S.SAME_TIME: ["DATE '2024-03-05'", "DATE '2024-03-07'", "DATE '1999-01-01'", "NULL"], S.SAME_SESSION: ["'x'", "'y'", "NULL"]}
    found = S.reading(left, right, dialect)
    rng = random.Random(len(left))
    ran = 0
    for _ in range(12):
        chosen = {name: rng.choice(["1", "3", "NULL"] if "UNIX" in call else values[kind_]) for name, kind_, call in found.names}
        connection = duckdb.connect()
        connection.execute("create table t(a bigint, d date, u varchar)")
        for _ in range(rng.randint(0, 6)):
            row = [rng.choice([None, 1, 2, 3]), rng.choice([None, "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-02-27"]), rng.choice([None, "x", "y"])]
            connection.execute("insert into t values (?, ?, ?)", row)
        results = []
        for sql in (found.left_sql, found.right_sql):
            for name, literal in chosen.items():
                sql = re.sub(re.escape(name) + r"\(\)", f"({literal})", sql)
            try:
                results.append(sorted(map(repr, connection.execute(sqlglot.transpile(sql, read=dialect, write="duckdb")[0]).fetchall())))
            except duckdb.Error:
                break  # a literal of a type this spelling does not take
        if len(results) == 2:
            ran += 1
            assert results[0] == results[1], (chosen, found.left_sql, found.right_sql)
    assert ran >= 6
