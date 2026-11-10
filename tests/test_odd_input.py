"""Text and schemas the provers cannot read are declined with a result, never with an exception.

Found by fuzzing the public entry points (2026-11-10). Each case below escaped as a raw exception from at least one
of them: ``RecursionError`` from some fifty nested parentheses or a chain of 142 ``UNION ALL`` branches (with
pure-Python sqlglot; the compiled build reads and writes both, so the cases built on depth accept either), sqlglot's
``TokenError`` for an unclosed quote where only ``ParseError`` was caught, ``AttributeError`` for the empty name part
sqlglot leaves in ``t..u``, ``decimal.InvalidOperation`` for the number ``1e``, ``ValueError`` for an unterminated
SQLX block, and DuckDB's own errors for a schema it cannot hold.
"""

import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest
import sqlglot

pytest.importorskip("z3")
pytest.importorskip("duckdb")

from sqlglot.dialects.bigquery import BigQuery

from kumo_sql_prover import bounded_equivalence, check_result_equivalence, prove_equivalent, prove_equivalent_smt, result_equivalence, statement_proof, tie_determinism
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover import bigquery_syntax
from kumo_sql_prover.bigquery_syntax import MAX_NESTING, MAX_OPEN_JOINS, MAX_TREE_DEPTH, MAX_TYPE_CALLS, TooDeeplyNested, nesting_weight
from kumo_sql_prover.result_equivalence import ResultEquivalenceStatus, query_constants
from kumo_sql_prover.smt_equivalence import SmtStatus
from kumo_sql_prover.tie_witness import find_tie_witness

SOURCE = str(Path(__file__).resolve().parent.parent / "src")
SCHEMA = {"t": ["a", "b"]}
TYPES = {"t": {"a": "INT64", "b": "STRING"}}
PLAIN = "SELECT a FROM t"

DEEP = "SELECT " + "(" * 60 + "1" + ")" * 60


def _nested(levels: int) -> str:
    return "SELECT " + "(" * levels + "1" + ")" * levels


# Past the fixed limits of ``bigquery_syntax`` (MAX_NESTING, MAX_TREE_DEPTH). Without them the compiled sqlglot build
# does not raise on these: it overflows the C stack and the process dies (see the small-stack test below).
TOO_DEEP = {
    "nesting-past-the-limit": _nested(MAX_NESTING + 1),
    "five-thousand-nested-parentheses": _nested(5000),
    "a-thousand-prefix-operators": "SELECT a FROM t WHERE " + "NOT " * 1000 + "a",
    "a-thousand-nested-array-types": "SELECT CAST(a AS " + "ARRAY<" * 1000 + "INT64" + ">" * 1000 + ") FROM t",
    "a-thousand-nested-cases": "SELECT " + "CASE WHEN a > 1 THEN " * 1000 + "a" + " END" * 1000 + " FROM t",
    "five-thousand-subscripts": "SELECT a" + "[0]" * 5000 + " FROM t",
    "thirty-calls-of-a-type-name": "SELECT " + "DATE(" * 30 + "a" + ")" * 30 + " FROM t",  # hours to read, not a crash
}
UNREADABLE = {
    "sixty-nested-parentheses": DEEP,
    "nesting-at-the-limit": _nested(MAX_NESTING),
    **TOO_DEEP,
    "unclosed-string": "SELECT '",
    "unclosed-quoted-name": "SELECT `",
    "stray-parenthesis": ")",
    "empty": "",
    "unterminated-sqlx-block": "config {",
    "empty-name-part": "SELECT a FROM t..u",
    "number-with-a-bare-exponent": "SELECT 1e FROM t",
}
DIALECTS = ["bigquery", "mysql", "postgres", "duckdb"]


def _bounded(left, right, dialect="bigquery"):
    return bounded_equivalence.check_bounded(left, right, bounded_equivalence.schema_from_prover(SCHEMA, types=TYPES), dialect=dialect, rows=1)


PROOFS = {
    "structural": lambda left, right, dialect: prove_equivalent(left, right),
    "smt": lambda left, right, dialect: prove_equivalent_smt(left, right, schema=SCHEMA, dialect=dialect),
    "smt-conditional": lambda left, right, dialect: prove_equivalent_smt(left, right, schema=SCHEMA, dialect=dialect, conditional=True),
    "algebraic": lambda left, right, dialect: prove_equivalent_algebraic(left, right, schema=SCHEMA, dialect=dialect),
    "statements": lambda left, right, dialect: statement_proof.prove_statements(left, right),
    "statements-smt": lambda left, right, dialect: statement_proof.prove_statements_smt(left, right),
}


@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize("case", sorted(UNREADABLE))
@pytest.mark.parametrize("prover", sorted(PROOFS))
def test_a_prover_declines_text_it_cannot_read(prover, case, dialect):
    if dialect != "bigquery" and prover in ("structural", "statements", "statements-smt"):
        pytest.skip("this entry point reads BigQuery only")
    result = PROOFS[prover](UNREADABLE[case], PLAIN, dialect)
    assert result.status in (SmtStatus.NOT_PROVEN, SmtStatus.NOT_EQUIVALENT) or result.status.value in ("not_proven", "not_equivalent"), result


@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize("case", sorted(UNREADABLE))
def test_the_bounded_check_declines_text_it_cannot_read(case, dialect):
    result = _bounded(UNREADABLE[case], PLAIN, dialect)
    assert result.status is not bounded_equivalence.BoundedStatus.BOUNDED_EQUIVALENT, result


@pytest.mark.parametrize("case", sorted(UNREADABLE))
def test_the_executed_check_reports_an_error_for_text_it_cannot_read(case):
    for left, right in ((UNREADABLE[case], PLAIN), (PLAIN, UNREADABLE[case])):
        result = check_result_equivalence(left, right, TYPES, seeds=(1,))
        assert result.status is not ResultEquivalenceStatus.EQUIVALENT, result
    result = check_result_equivalence(UNREADABLE[case], PLAIN, TYPES, seeds=(1,), targeted=True)
    assert result.status is not ResultEquivalenceStatus.EQUIVALENT, result


@pytest.mark.parametrize("case", sorted(UNREADABLE))
def test_the_tie_analysis_and_the_tie_witness_search_do_not_raise(case):
    report = tie_determinism.analyze(UNREADABLE[case], schema=SCHEMA)
    assert report.sites == () or report.unsupported is None
    assert find_tie_witness(UNREADABLE[case], TYPES, budget=2.0) is None
    assert isinstance(query_constants(UNREADABLE[case], PLAIN), dict)


def test_a_long_chain_of_set_operations_is_answered_not_a_crash():
    """142 branches are as deep for pure-Python sqlglot's SQL writer as fifty nested parentheses are for its parser.
    The compiled build (sqlglotc) writes the chain without recursing, so there the check simply runs."""

    chain = " UNION ALL ".join([PLAIN] * 142)
    result = check_result_equivalence(chain, chain, TYPES, seeds=(1,))
    if result.status is ResultEquivalenceStatus.ERROR:
        assert "nested too deeply" in result.reason
    else:
        assert result.status is ResultEquivalenceStatus.EQUIVALENT  # the same text on both sides
    assert prove_equivalent_smt(chain, chain, schema=SCHEMA).status is SmtStatus.PROVEN_EQUIVALENT  # the provers read it


def test_a_statement_too_deep_to_write_for_duckdb_is_an_error_result(monkeypatch):
    """The same on every sqlglot build and recursion limit: the writer's ``RecursionError`` is raised by hand."""

    def too_deep(statement):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(result_equivalence, "spell_for_duckdb", too_deep)
    result = check_result_equivalence(PLAIN, PLAIN, TYPES, seeds=(1,))
    assert result.status is ResultEquivalenceStatus.ERROR and "nested too deeply" in result.reason


class _ParserOutOfStack:
    def parse(self, tokens, sql=None):
        raise RecursionError("maximum recursion depth exceeded")


def test_deep_nesting_is_a_parse_error_for_every_caller_of_the_bigquery_parser(monkeypatch):
    """One place turns the interpreter's ``RecursionError`` into the ``ParseError`` callers already handle.

    Where the stack ends depends on the build: pure-Python sqlglot stops at some fifty nested parentheses, the
    compiled one reads sixty. So that text is only required to parse or to be a ``ParseError``, and the conversion
    is tested with a parser that runs out of stack on any input."""

    try:
        assert sqlglot.parse_one(DEEP, read="bigquery") is not None
    except sqlglot.errors.ParseError as error:
        assert "nested too deeply" in str(error)
    assert sqlglot.parse_one(_nested(20), read="bigquery") is not None  # ordinary nesting parses

    monkeypatch.setattr(BigQuery, "parser", lambda self, **opts: _ParserOutOfStack())
    with pytest.raises(sqlglot.errors.ParseError, match="nested too deeply"):
        sqlglot.parse_one(PLAIN, read="bigquery")
    with pytest.raises(sqlglot.errors.ParseError, match="nested too deeply"):
        sqlglot.parse(PLAIN + "; SELECT 1", read="bigquery")
    assert prove_equivalent_smt(PLAIN, PLAIN, schema=SCHEMA).status is SmtStatus.NOT_PROVEN
    assert statement_proof.prove_statements(PLAIN, PLAIN).status.value == "not_proven"
    assert check_result_equivalence(PLAIN, PLAIN, TYPES, seeds=(1,)).status is ResultEquivalenceStatus.ERROR


@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize("case", sorted(TOO_DEEP))
def test_depth_past_the_fixed_limit_is_a_parse_error_in_every_dialect_on_every_build(case, dialect):
    """The limit is a number, not the interpreter's stack: the same answer from pure-Python and compiled sqlglot."""

    with pytest.raises(TooDeeplyNested, match="nested too deeply"):
        sqlglot.parse_one(TOO_DEEP[case], read=dialect)
    with pytest.raises(TooDeeplyNested):
        sqlglot.parse(PLAIN + "; " + TOO_DEEP[case], read=dialect)


@pytest.mark.parametrize("dialect", DIALECTS)
def test_what_is_long_but_not_deep_is_still_read(dialect):
    """The limits count what recurses. A sum of 3,000 columns, 3,000 conditions, 1,000 ``UNION ALL`` branches, a list
    of 5,000 values and hundreds of separate CASEs, array types and subscripts are flat, and stay readable."""

    def reads(sql):
        try:
            return sqlglot.parse_one(sql, read=dialect) is not None
        except TooDeeplyNested as error:  # the interpreter's own stack, never the fixed limit
            return "levels" not in str(error)

    assert reads("SELECT " + " + ".join(["a"] * 3000) + " AS v FROM t")
    assert reads("SELECT a FROM t WHERE " + " AND ".join(["a > 1"] * 3000))
    assert reads(" UNION ALL ".join([PLAIN] * 1000))
    assert reads("SELECT a IN (" + ", ".join(["1"] * 5000) + ") FROM t")
    assert reads("SELECT " + ", ".join(["CASE WHEN a > 1 THEN 1 END"] * 300) + " FROM t")
    assert reads("SELECT " + ", ".join(["a - -1", "NOT NOT a"] * 300) + " FROM t")
    assert reads("SELECT " + ", ".join(["(a)"] * 300) + " FROM t")
    assert reads("SELECT a" + "[0]" * (MAX_TREE_DEPTH // 4) + " FROM t") or dialect == "mysql"  # MySQL has no subscripts
    if dialect == "bigquery":
        assert reads("SELECT " + ", ".join(["CAST(b AS ARRAY<STRUCT<x ARRAY<INT64>>>)"] * 100) + " FROM t")


@pytest.mark.parametrize("extent", ["two-dialects", "four-dialects"])
def test_depth_does_not_end_the_process_on_a_small_stack(extent):
    """What the limits are for. The compiled build died with a segmentation fault: on a 512 KB stack reading 400 nested
    parentheses, 400 ``@`` or ``ANY`` in a row, 3,200 subscripts, and writing 300 JSON steps or lambdas; on 1 MB at about
    twice that. Each statement is read and written in four dialects on a 1 MB thread, the smallest stack a CPython
    thread gets unasked (pure-Python sqlglot needs 768 KB to write a tree at the limit), in a process of its own. The
    run over four dialects takes three seconds and is in the slow tier (``tests/order.json``)."""

    script = """
import sys, threading
import sqlglot
import kumo_sql_prover
deep = {
    "parentheses": "SELECT " + "(" * 5000 + "1" + ")" * 5000,
    "subqueries": "SELECT a FROM " + "(SELECT a FROM " * 2000 + "t" + ")" * 2000,
    "exists": "SELECT a FROM t WHERE " + "EXISTS (SELECT a FROM t WHERE " * 2000 + "TRUE" + ")" * 2000,
    "cases": "SELECT " + "CASE WHEN a > 1 THEN " * 2000 + "a" + " END" * 2000 + " FROM t",
    "not": "SELECT a FROM t WHERE " + "NOT " * 5000 + "a",
    "types": "SELECT CAST(a AS " + "ARRAY<" * 5000 + "INT64" + ">" * 5000 + ") FROM t",
    "subscripts": "SELECT a" + "[0]" * 5000 + " FROM t",
    "casts": "SELECT a" + "::INT" * 3000 + " FROM t",
    "modulo": "SELECT " + " % ".join(["a"] * 3000) + " FROM t",
    "at-signs": "SELECT " + "@ " * 5000 + "a FROM t",
    "any": "SELECT " + "ANY " * 5000 + "a FROM t",
    "interval": "SELECT " + "INTERVAL " * 5000 + "a FROM t",
    "time-zones": "SELECT a" + " AT TIME ZONE 'UTC'" * 5000 + " FROM t",
    "assignments": "SELECT " + " := ".join(["a"] * 5000) + " FROM t",
    "tildes": "SELECT " + "~ " * 5000 + "a FROM t",
    "json-steps": "SELECT a" + "->'k'" * 5000 + " FROM t",
    "lambdas": "SELECT f(" + "x -> " * 5000 + "x) FROM t",
    "powers": "SELECT " + " ^ ".join(["a"] * 5000) + " FROM t",
    "joins-without-on": "SELECT * FROM t" + " JOIN t" * 5000,
    "calls-of-a-type-name": "SELECT " + "DATE(" * 60 + "a" + ")" * 60 + " FROM t",
}
within = {
    "parentheses-at-the-limit": "SELECT " + "(" * 100 + "a" + ")" * 100 + " FROM t",
    "subqueries-at-the-limit": "SELECT a FROM " + "(SELECT a FROM " * 60 + "t" + ")" * 60,
    "subscripts-at-the-limit": "SELECT a" + "[0]" * 190 + " FROM t",
    "casts-at-the-limit": "SELECT a" + "::INT" * 190 + " FROM t",
    "json-steps-at-the-limit": "SELECT a" + "->'k'" * 190 + " FROM t",
    "lambdas-at-the-limit": "SELECT f(" + "x -> " * 190 + "x) FROM t",
    "powers-at-the-limit": "SELECT " + " ^ ".join(["a"] * 190) + " FROM t",
    "at-signs-at-the-limit": "SELECT " + "@ " * 190 + "a FROM t",
    "long-sum": "SELECT " + " + ".join(["a"] * 3000) + " FROM t",
}
dialects = DIALECTS
answers = {}
def run():
    for name, sql in {**deep, **within}.items():
        for dialect in dialects:
            try:
                tree = sqlglot.parse_one(sql, read=dialect)
                tree.sql(dialect=dialect)
                tree.sql(dialect="duckdb")
                answers[name, dialect] = "read"
            except sqlglot.errors.ParseError:
                answers[name, dialect] = "declined"
            except RecursionError:
                answers[name, dialect] = "declined"  # pure-Python sqlglot writing a tree near the limit
threading.stack_size(1024 * 1024)
thread = threading.Thread(target=run)
thread.start()
thread.join()
assert len(answers) == len(dialects) * (len(deep) + len(within)), answers
assert not [key for key, answer in answers.items() if key[0] in deep and answer != "declined"], answers
assert all(answers["long-sum", dialect] == "read" for dialect in dialects), answers
print("survived")
"""
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [SOURCE, os.environ.get("PYTHONPATH")]))}
    dialects = ("bigquery", "duckdb") if extent == "two-dialects" else ("bigquery", "duckdb", "postgres", "snowflake")
    script = script.replace("DIALECTS", repr(dialects))
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600, env=environment)
    assert done.returncode == 0 and "survived" in done.stdout, (done.returncode, done.stderr[-2000:])


# 245 tokens with no comma among them: the parser could be that deep in it, for all the tokens say (``nesting_weight``)
HEAVY = "SELECT a FROM t WHERE " + " AND ".join(f"a > {index}" for index in range(60))
# 400 tokens, never more than a dozen since the last comma
LIGHT = "SELECT " + ", ".join(f"COALESCE(a, {index}) + 1 AS c{index}" for index in range(40)) + " FROM t WHERE a > 1 AND b = 'x'"


@pytest.fixture
def roomy(monkeypatch):
    """The path the compiled build takes for a statement that could go deep, on whichever build is installed."""

    monkeypatch.setattr(bigquery_syntax, "_COMPILED", True)
    used = []
    start = threading.Thread.start

    def recording_start(thread):
        if thread.name == "deep-sql-parse":
            used.append(threading.stack_size())
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", recording_start)
    return used


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_statement_that_could_go_deep_is_read_the_same_on_a_thread_with_room(dialect, roomy, monkeypatch):
    on_a_thread = sqlglot.parse_one(HEAVY, read=dialect)
    several = sqlglot.parse(HEAVY + "; " + PLAIN, read=dialect)
    assert len(roomy) == 2 and min(roomy) >= 16 << 20  # both were read on a thread, with at least 16 MiB of stack
    assert threading.stack_size() < 16 << 20  # and the process setting was put back
    sqlglot.parse_one(PLAIN, read=dialect)
    sqlglot.parse_one(LIGHT, read=dialect)
    assert len(roomy) == 2  # a short statement and a long list are read where the caller stands

    monkeypatch.setattr(bigquery_syntax, "_COMPILED", False)
    in_place = sqlglot.parse_one(HEAVY, read=dialect)
    assert len(roomy) == 2
    assert on_a_thread == in_place and several[0] == in_place and len(several) == 2
    assert on_a_thread.sql(dialect=dialect) == in_place.sql(dialect=dialect)


@pytest.mark.parametrize("dialect", DIALECTS)
def test_an_error_in_a_statement_read_on_a_thread_is_the_same_error(dialect, roomy):
    with pytest.raises(sqlglot.errors.ParseError) as caught:
        sqlglot.parse_one(HEAVY + " GROUP BY (", read=dialect)
    assert roomy and not isinstance(caught.value, TooDeeplyNested)
    with pytest.raises(TooDeeplyNested, match="nested too deeply"):
        sqlglot.parse_one("SELECT " + " % ".join(["a"] * (MAX_TREE_DEPTH + 50)) + " FROM t", read=dialect)


def test_a_statement_is_read_where_no_thread_can_be_had(roomy, monkeypatch):
    def refuse(size=0):
        raise ValueError("size not valid")

    monkeypatch.setattr(threading, "stack_size", refuse)
    assert sqlglot.parse_one(HEAVY, read="bigquery").sql(dialect="bigquery") == sqlglot.parse_one(HEAVY, read="duckdb").sql(dialect="bigquery")
    assert not roomy


def test_statements_are_read_from_several_threads_at_once(roomy):
    """The stack size is one setting for the whole process; readers take turns setting it and do not wait on each other's parse."""

    expected = sqlglot.parse_one(HEAVY, read="bigquery")
    results, errors = [], []

    def reader():
        try:
            for _ in range(10):
                results.append(sqlglot.parse_one(HEAVY, read="bigquery") == expected)
        except Exception as error:  # reported below, in the test's own thread
            errors.append(error)

    readers = [threading.Thread(target=reader) for _ in range(4)]
    for thread in readers:
        thread.start()
    for thread in readers:
        thread.join(60)
    assert not errors and len(results) == 40 and all(results)
    assert len(roomy) == 41


def _weight(sql, dialect="bigquery"):
    return nesting_weight(sql, sqlglot.Dialect.get_or_raise(dialect).tokenize(sql))


def test_the_weight_of_a_statement_is_what_can_be_open_at_once():
    """Long lists weigh nothing; what has no comma in it weighs what it is long. The padding makes each statement long
    enough to be weighed at all."""

    padding = ", " + ", ".join(["x"] * 100)
    assert _weight("SELECT a, b FROM t") == 6  # short: its length, without looking
    assert _weight("SELECT f(a, g(b, c))" + padding) == 7  # SELECT f ( .. g ( .. c
    assert _weight("SELECT NOT NOT NOT a" + padding) == 5
    assert _weight("SELECT ((a))" + padding) == 5  # a closing bracket takes its own level's tokens away, not its parent's
    assert _weight("SELECT a[0][0]" + padding) == 6
    assert _weight("SELECT CAST(a AS ARRAY<STRUCT<x ARRAY<INT64>, y INT64>>), b" + padding) == 14  # ``>>`` closes two
    assert _weight("SELECT " + ", ".join(["CAST(b AS ARRAY<STRUCT<x ARRAY<INT64>>>)"] * 150)) == 14
    assert _weight("SELECT " + ", ".join(["CASE WHEN a > 1 THEN 1 END"] * 300)) == 9
    assert _weight("SELECT a, b FROM t WHERE x > 1; " * 50) == 7  # one statement after another: b FROM t WHERE x > 1
    assert _weight(LIGHT) <= 20 < 80 < _weight(HEAVY)
    assert _weight("SELECT " + " + ".join(["a"] * 300)) == 600  # read in a loop, but the tokens cannot say so
    assert _weight(_nested(MAX_NESTING)) == MAX_NESTING + 3
    # a comparison with a column named like a type is not the start of ``RANGE<..>``, however often it is made
    assert _weight("SELECT a FROM t WHERE " + " AND ".join(["range < 5", "list < map"] * 150)) > 80
    assert _weight("SELECT " + ", ".join(["range < 5", "list < x", "map < a.b"] * 150) + " FROM t WHERE a") < 80


@pytest.mark.parametrize("dialect", DIALECTS)
def test_light_statements_make_shallow_trees(dialect):
    """A statement read without measuring its tree cannot have made a deep one: at most two nodes below each other per token."""

    shapes = {
        "parentheses": lambda n: "SELECT " + "(" * n + "a" + ")" * n + " FROM t",
        "subqueries": lambda n: "SELECT a FROM " + "(SELECT a FROM " * n + "t" + ")" * n,
        "exists": lambda n: "SELECT a FROM t WHERE " + "EXISTS (SELECT a FROM t WHERE " * n + "TRUE" + ")" * n,
        "cases": lambda n: "SELECT " + "CASE WHEN a > 1 THEN " * n + "a" + " END" * n + " FROM t",
        "calls": lambda n: "SELECT " + "COALESCE(" * n + "a" + ")" * n + " FROM t",
        "casts": lambda n: "SELECT " + "CAST(" * n + "a" + " AS STRING)" * n + " FROM t",
        "not": lambda n: "SELECT a FROM t WHERE " + "NOT " * n + "a",
        "is-not-null": lambda n: "SELECT a" + " IS NOT NULL" * n + " FROM t",
        "dots": lambda n: "SELECT a" + ".b" * n + " FROM t",
        "modulo": lambda n: "SELECT " + " % ".join(["a"] * n) + " FROM t",
        "comparisons": lambda n: "SELECT " + " = ".join(["a"] * n) + " FROM t",
        "windows": lambda n: "SELECT " + "SUM(" * n + "a" + ") OVER (PARTITION BY b)" * n + " FROM t",
    }
    weighed = 0
    for name, shape in shapes.items():
        for size in (3, 10, 25, 78):
            sql = shape(size)
            tokens = sqlglot.Dialect.get_or_raise(dialect).tokenize(sql)
            try:
                if nesting_weight(sql, tokens) > bigquery_syntax._DIRECT_WEIGHT:
                    continue
                trees = sqlglot.parse(sql, read=dialect)
            except sqlglot.errors.ParseError:
                continue  # the dialect does not read this shape, or pure-Python sqlglot ran out of stack
            weighed += 1
            assert not bigquery_syntax._too_deep(trees), (name, size)
    assert weighed >= 20


@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize("name", ["DATE", "STRING", "STRUCT", "TIMESTAMP", "INT64"])
def test_calls_of_a_type_name_inside_each_other_are_refused_before_they_take_minutes(name, dialect):
    """sqlglot reads the inside of ``DATE(..)`` twice, as a type's parameters and as arguments: 24 inside each other took
    ninety seconds with the compiled build, 40 do not finish. Found 2026-11-10."""

    def calls(depth, inside="a"):
        return "SELECT " + f"{name}(" * depth + inside + ")" * depth + " FROM t"

    for depth in (MAX_TYPE_CALLS + 1, 40, 2000):
        with pytest.raises(TooDeeplyNested, match="calls of a type's name"):
            sqlglot.parse_one(calls(depth), read=dialect)
    assert sqlglot.parse_one(calls(MAX_TYPE_CALLS), read=dialect) is not None  # at the limit: some thousand readings of ``a``
    assert sqlglot.parse_one(calls(4), read=dialect) is not None
    # next to each other they cost nothing, however many
    assert sqlglot.parse_one("SELECT " + ", ".join([f"{name}({name}(a))"] * 200) + " FROM t", read=dialect) is not None
    assert sqlglot.parse_one("SELECT " + ", ".join(["CAST(a AS NUMERIC(10, 2))"] * 200) + " FROM t", read=dialect) is not None


CASE_CONFLICT = {"t": {"a": "INT64"}, "T": {"a": "INT64"}}  # two tables in BigQuery, one name in DuckDB
NO_COLUMNS = {"t": {}}


@pytest.mark.parametrize("schema", [CASE_CONFLICT, NO_COLUMNS], ids=["names-differ-only-by-case", "table-without-columns"])
def test_a_schema_duckdb_cannot_hold_is_an_error_result(schema):
    result = check_result_equivalence("SELECT 1", "SELECT 2", schema, seeds=(1,))
    assert result.status is ResultEquivalenceStatus.ERROR, result
    assert find_tie_witness("SELECT 1", schema, budget=2.0) is None
    names = {name: list(columns) for name, columns in schema.items()}
    bounded = bounded_equivalence.check_bounded("SELECT 1", "SELECT 1", bounded_equivalence.schema_from_prover(names, types=schema), rows=1)
    assert bounded.status in (bounded_equivalence.BoundedStatus.UNKNOWN, bounded_equivalence.BoundedStatus.BOUNDED_EQUIVALENT), bounded
    assert prove_equivalent_smt("SELECT 1", "SELECT 2", schema=names, types=schema).status is SmtStatus.NOT_EQUIVALENT


def test_an_empty_name_part_is_declined_by_both_provers_in_every_dialect():
    same = "SELECT a FROM t..u"
    for dialect in DIALECTS:
        for prove in (prove_equivalent_smt, prove_equivalent_algebraic):
            result = prove(same, same, schema=SCHEMA, dialect=dialect)
            assert result.status is SmtStatus.NOT_PROVEN and "empty part" in result.reason, (dialect, prove.__name__, result.reason)


@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize("join", ["JOIN t", "LEFT JOIN t", "INNER JOIN t AS x", "JOIN (SELECT a FROM t) AS s"])
def test_joins_in_a_row_with_no_condition_are_refused_before_they_take_all_the_memory(join, dialect):
    """sqlglot tries the joins after a join with no ON as nested inside it and reads them again when no ON follows.
    Twenty ``JOIN t`` in a row took six seconds and 1.1 GB with the compiled build, twenty-four more than 16 GB.
    Found 2026-11-10."""

    for count in (MAX_OPEN_JOINS + 1, 24, 3000):
        with pytest.raises(TooDeeplyNested, match="joins in a row with no ON or USING"):
            sqlglot.parse_one("SELECT * FROM t " + " ".join([join] * count), read=dialect)
    with pytest.raises(TooDeeplyNested, match="joins in a row"):
        sqlglot.parse_one("SELECT * FROM t " + ", t ".join([join] * 12), read=dialect)  # a comma does not end the run
    assert sqlglot.parse_one("SELECT * FROM t " + " ".join([join] * 6), read=dialect) is not None


@pytest.mark.parametrize("dialect", DIALECTS)
def test_joins_that_have_their_condition_are_read_however_many(dialect):
    def reads(sql):
        return sqlglot.parse_one(sql, read=dialect) is not None

    assert reads("SELECT * FROM t" + " JOIN t AS x ON t.a = x.a" * 300)
    assert reads("SELECT * FROM t" + " LEFT JOIN t AS x USING (a)" * 300)
    assert reads("SELECT * FROM t" + " CROSS JOIN t AS x" * 300)
    assert reads("SELECT * FROM t" + " NATURAL JOIN t AS x" * 300)
    assert reads("SELECT * FROM t" + ", t AS x" * 300)
    assert reads("SELECT * FROM t" + " JOIN (SELECT a FROM t JOIN t AS y ON TRUE JOIN t AS z ON TRUE) AS s ON TRUE" * 100)
    assert reads("SELECT * FROM t" + " JOIN t" * MAX_OPEN_JOINS + " ON TRUE" * MAX_OPEN_JOINS)  # nested joins, their ONs last
    several = sqlglot.parse("; ".join(["SELECT * FROM t" + " JOIN t AS x" * 6] * 5), read=dialect)  # each statement's joins are its own
    assert len(several) == 5


@pytest.mark.parametrize("prover", sorted(PROOFS))
def test_a_prover_declines_joins_in_a_row_with_no_condition(prover):
    endless = "SELECT * FROM t" + " JOIN t" * 30
    result = PROOFS[prover](endless, PLAIN, "bigquery")
    assert result.status in (SmtStatus.NOT_PROVEN, SmtStatus.NOT_EQUIVALENT) or result.status.value in ("not_proven", "not_equivalent"), result
