"""No counterexample is taken from a model that sets a text of one query against a number of the other (issue 60).

``SELECT id FROM t WHERE c = '1'`` against ``SELECT id FROM t WHERE c = 1`` came back as "not equivalent" with a
database that holds the number 1 in ``c``. The engines return the row from both queries on that database (replayed
below on DuckDB and SQLite with a whole-number column), so it is no counterexample; the pair is now "not proven".
"""

from __future__ import annotations

import sqlite3

import duckdb
import pytest

from kumo_sql_prover import cross_kind_witness
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtStatus, prove_equivalent_smt

SCHEMA = {"t": ["id", "c", "d", "s"]}
UNTYPED = None
FLAG = {"t": {"id": "INT", "c": "BOOL", "d": "DATE", "s": "VARCHAR"}}
WHOLE = {"t": {"id": "INT", "c": "INT", "d": "DATE", "s": "VARCHAR"}}
TEXT = {"t": {"id": "INT", "c": "VARCHAR", "d": "DATE", "s": "VARCHAR"}}
DIALECTS = ["mysql", "sqlite", "duckdb", "postgres"]

# pairs in which one value of open kind meets a string in one query and a number in the other
CLASH = [
    ("SELECT id FROM t WHERE c = '1'", "SELECT id FROM t WHERE c = 1"),
    ("SELECT id FROM t WHERE c = 'a'", "SELECT id FROM t WHERE c = 0"),  # MySQL reads 'a' as 0
    ("SELECT id FROM t WHERE c = '-1'", "SELECT id FROM t WHERE c = -1"),
    ("SELECT id FROM t WHERE c IN ('1', '2')", "SELECT x.id FROM t AS x WHERE x.c IN (1, 2)"),  # another source name
    ("SELECT id FROM t WHERE c BETWEEN '1' AND '3'", "SELECT id FROM t WHERE c BETWEEN 1 AND 3"),
    ("SELECT id FROM t WHERE COALESCE(c, id) = '1'", "SELECT id FROM t WHERE COALESCE(c, id) = 1"),
    ("SELECT id FROM t WHERE c <> '1'", "SELECT id FROM (SELECT id, c FROM t) AS u WHERE u.c <> 1"),
]
# pairs with no such clash: their counterexamples are databases, and they are kept
NO_CLASH = [
    ("SELECT id FROM t WHERE c = '1'", "SELECT id FROM t WHERE c = '2'"),
    ("SELECT id FROM t WHERE c = 1", "SELECT id FROM t WHERE c = 2"),
    ("SELECT id FROM t WHERE s = '1'", "SELECT id FROM t WHERE c = 1"),  # two columns
    ("SELECT id FROM t WHERE c = 1", "SELECT id FROM t"),
    ("SELECT id FROM t WHERE c = '1'", "SELECT id FROM t"),
]
IDS = lambda pair: f"{pair[0][17:]} / {pair[1][17:]}"  # noqa: E731


@pytest.mark.parametrize("pair", CLASH, ids=IDS)
@pytest.mark.parametrize("types", [UNTYPED, FLAG], ids=["untyped", "flag"])
def test_a_clash_across_the_pair_is_seen(pair, types):
    for dialect in DIALECTS:
        assert cross_kind_witness.clash(*pair, dialect, types, SCHEMA)
        assert cross_kind_witness.clash(*pair[::-1], dialect, types, SCHEMA)


@pytest.mark.parametrize("pair", NO_CLASH, ids=IDS)
def test_one_kind_or_two_columns_is_no_clash(pair):
    for dialect in DIALECTS:
        assert not cross_kind_witness.clash(*pair, dialect, UNTYPED, SCHEMA)


@pytest.mark.parametrize("pair", CLASH[:4], ids=IDS)
def test_a_column_declared_a_number_or_a_text_is_no_clash(pair):
    # the solver reads the comparison with the other kind as an unknown predicate and takes no counterexample already
    for types in (WHOLE, TEXT):
        assert not cross_kind_witness.clash(*pair, "mysql", types, SCHEMA)


@pytest.mark.parametrize("prover", [prove_equivalent_algebraic, prove_equivalent_smt], ids=["algebraic", "smt"])
@pytest.mark.parametrize("types", [UNTYPED, FLAG], ids=["untyped", "flag"])
@pytest.mark.parametrize("dialect", DIALECTS)
def test_the_pair_of_the_issue_is_not_proven_and_has_no_counterexample(prover, types, dialect):
    result = prover(*CLASH[0], schema=SCHEMA, types=types, dialect=dialect)
    assert result.status is SmtStatus.NOT_PROVEN and result.counterexample is None
    assert result.reason == cross_kind_witness.REASON


@pytest.mark.parametrize("pair", CLASH[1:], ids=IDS)
def test_no_clashing_pair_gets_a_counterexample(pair):
    for dialect in ("mysql", "duckdb"):
        for types in (UNTYPED, FLAG):
            result = prove_equivalent_algebraic(*pair, schema=SCHEMA, types=types, dialect=dialect, compare_names=False)
            assert result.status is SmtStatus.NOT_PROVEN and result.counterexample is None, (dialect, types)


@pytest.mark.parametrize("pair", NO_CLASH, ids=IDS)
def test_a_pair_without_a_clash_keeps_its_counterexample(pair):
    for prover in (prove_equivalent_algebraic, prove_equivalent_smt):
        result = prover(*pair, schema=SCHEMA, dialect="mysql")
        assert result.status is SmtStatus.NOT_EQUIVALENT and result.counterexample is not None


def test_a_proof_across_such_a_pair_is_kept():
    # nothing in a proof sets the text of one query against the number of the other
    left, right = "SELECT id FROM t WHERE c = '1' AND 1 = 0", "SELECT id FROM t WHERE c = 1 AND 1 = 0"
    for types in (UNTYPED, FLAG):
        assert prove_equivalent_algebraic(left, right, schema=SCHEMA, types=types, dialect="mysql").proven


def test_a_declared_whole_number_column_is_read_as_before():
    assert prove_equivalent_algebraic(*CLASH[0], schema=SCHEMA, types=WHOLE, dialect="mysql").proven
    assert prove_equivalent_algebraic(*CLASH[0], schema=SCHEMA, types=WHOLE, dialect="duckdb").proven


ROWS = [(1, 1), (2, 0), (3, 2), (4, None), (5, -1), (6, 3)]


@pytest.mark.parametrize("pair", [pair for pair in CLASH if "'a'" not in pair[0]], ids=IDS)
def test_duckdb_and_sqlite_return_the_same_rows_for_a_whole_number_column(pair):
    """The database the provers used to show (a number in ``c``) separates none of these pairs."""

    lite = sqlite3.connect(":memory:")
    duck = duckdb.connect()
    for con in (lite, duck):
        con.execute("CREATE TABLE t (id INT, c INT, d DATE, s VARCHAR)")
        con.executemany("INSERT INTO t (id, c) VALUES (?, ?)", ROWS)
    for con in (lite, duck):
        left, right = (sorted(con.execute(sql).fetchall()) for sql in pair)
        if con is lite and "COALESCE" in pair[0]:
            # the cost of the rule: SQLite converts nothing for a value that is not a plain column, so this pair does
            # differ there on a row of numbers, and the provers no longer say so ("not proven" instead)
            assert (left, right) == ([], [(1,)])
        else:
            assert left == right
    assert sorted(lite.execute(CLASH[0][0]).fetchall()) == [(1,)]  # the row with the number 1 passes c = '1'


def test_sqlite_reads_text_a_as_no_number_and_mysql_is_not_run_here():
    # SQLite does not convert 'a' to 0 for a whole-number column (MySQL does); the pair differs there on a row with 0,
    # which is a database of numbers and text read by SQLite's rules, not the one the provers showed for MySQL
    lite = sqlite3.connect(":memory:")
    lite.execute("CREATE TABLE t (id INT, c INT)")
    lite.executemany("INSERT INTO t VALUES (?, ?)", ROWS)
    assert lite.execute("SELECT id FROM t WHERE c = 'a'").fetchall() == []
    assert lite.execute("SELECT id FROM t WHERE c = 0").fetchall() == [(2,)]
