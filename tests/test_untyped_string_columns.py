"""A column the schema does not type, compared with different strings that MySQL reads as the same number (issue #613).

``WHERE t.s = 'x' AND t.s = 'y'`` is empty for a text column and returns the rows where ``t.s`` is 0 for an integer column,
since MySQL compares a number with a string as numbers and reads ``'x'`` and ``'y'`` as 0. The provers read an untyped column as
text, so they proved the pair empty. On MySQL a proof now has to hold with those strings replaced by the number they convert to
(``kumo_sql_prover.numeric_column_reading``). Other dialects refuse the comparison of an integer column with such a string, so they are
unchanged.
"""

import pytest

pytest.importorskip("z3")

from kumo_sql_prover import numeric_column_reading, string_number_compare
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtStatus, prove_equivalent_smt

PROVERS = (prove_equivalent_algebraic, prove_equivalent_smt)
NONE = "SELECT t.k FROM t WHERE FALSE"

# MySQL 8.0 returns the row t.s = 0, t.k = 1 of an INT column for each left query and none for the right
WITNESSES = [
    ("SELECT t.k FROM t WHERE t.s = 'x' AND t.s = 'y'", NONE),
    ("SELECT t.k FROM t WHERE t.s = 'a1' AND t.s = 'a2'", NONE),
    ("SELECT t.k FROM t WHERE t.s = 'x' AND t.s <> 'y'", "SELECT t.k FROM t WHERE t.s = 'x'"),
    ("SELECT t.k FROM t JOIN u ON t.k = u.k WHERE t.s = 'x' AND t.s = u.s AND u.s = 'y'", NONE),
    ("SELECT t.k FROM t WHERE t.s IN ('x', 'y') AND t.s NOT IN ('x', 'z')", NONE),
    ("SELECT t.k FROM t WHERE t.s BETWEEN 'x' AND 'y' AND t.s = 'z'", NONE),
]

# proofs that hold for a text column and for an integer column alike
NEAR_MISSES = [
    ("SELECT t.k FROM t WHERE t.s IN ('x', 'y')", "SELECT t.k FROM t WHERE t.s = 'x' OR t.s = 'y'"),
    ("SELECT t.k FROM t WHERE t.s = 'x' AND t.s = 'x'", "SELECT t.k FROM t WHERE t.s = 'x'"),
    ("SELECT t.k FROM t WHERE t.s = 'x' OR t.s = 'y'", "SELECT t.k FROM t WHERE t.s = 'y' OR t.s = 'x'"),
    ("SELECT t.k FROM t WHERE t.s = 'x'", "SELECT t.k FROM t WHERE t.s = 'x' AND TRUE"),
    # one string per column, or strings that no integer column reads alike ('1' is 1, 'x' is 0)
    ("SELECT t.k FROM t WHERE t.s = '1' AND t.s = 'x'", "SELECT t.k FROM t WHERE t.s = 'x' AND t.s = '1'"),
    (
        "SELECT t.k FROM t JOIN u ON t.k = u.k WHERE t.s = 'x' AND u.s = 'y'",
        "SELECT t.k FROM t JOIN u ON t.k = u.k WHERE u.s = 'y' AND t.s = 'x'",
    ),
]


@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("left,right", WITNESSES)
def test_witness_is_not_proven_on_mysql(prover, left, right):
    result = prover(left, right, dialect="mysql")
    assert not result.proven, (left, result.reason)


@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("left,right", NEAR_MISSES)
def test_proofs_that_do_not_need_the_strings_to_differ_stay(prover, left, right):
    assert prover(left, right, dialect="mysql").status is SmtStatus.PROVEN_EQUIVALENT, left


@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("left,right", WITNESSES[:4])
def test_other_dialects_are_unchanged(prover, left, right):
    for dialect in ("duckdb", "postgres", "bigquery"):
        assert prover(left, right, dialect=dialect).status is SmtStatus.PROVEN_EQUIVALENT, (dialect, left)


@pytest.mark.parametrize("prover", PROVERS)
def test_a_declared_text_column_keeps_the_proof(prover):
    types = {"t": {"s": "VARCHAR", "k": "INT"}}
    left, right = WITNESSES[0]
    assert prover(left, right, dialect="mysql", types=types).status is SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize("prover", PROVERS)
def test_a_declared_integer_column_is_not_proven(prover):
    types = {"t": {"s": "INT", "k": "INT"}}
    left, right = WITNESSES[0]
    assert not prover(left, right, dialect="mysql", types=types).proven


def test_mysql_reads_the_leading_number():
    number = string_number_compare.mysql_number
    assert [number(text) for text in ("x", "", "abc1", "1abc", " 12", "-3x", "+4", "1.5", ".5e1", "1e2z", "0x1A")] == [0, 0, 0, 1, 12, -3, 4, 1.5, 5, 100, 0]


def test_the_reading_replaces_the_strings_by_numbers():
    found = numeric_column_reading.reading("SELECT t.k FROM t WHERE t.s = 'x' AND t.s = 'y1' AND t.s = '-2z'", NONE, "mysql")
    assert found.pair[0] == "SELECT t.k FROM t WHERE t.s = 0 AND t.s = 0 AND t.s = -2"
    assert found.pair[1] == NONE


def test_a_string_with_no_exact_small_integer_is_declined():
    found = numeric_column_reading.reading("SELECT t.k FROM t WHERE t.s = '1.5' AND t.s = '1.5x'", NONE, "mysql")
    assert found.problem and found.pair is None


@pytest.mark.parametrize("dialect", ("duckdb", "postgres", "bigquery", "snowflake"))
def test_only_mysql_is_checked(dialect):
    assert numeric_column_reading.reading(*WITNESSES[0], dialect) is None


THROUGH_DERIVED = (
    "SELECT c FROM (SELECT s AS c FROM t) AS d WHERE c = 'x' AND c = 'y'",
    "SELECT c FROM (SELECT s AS c FROM t) AS d WHERE FALSE",
)


@pytest.mark.parametrize("prover", PROVERS)
def test_a_declared_text_column_read_by_a_bare_name_through_a_derived_table_keeps_the_proof(prover):
    """``c`` is never written ``d.c``; it is still the derived table's output, a declared VARCHAR (VeriEQL Calcite pair 147)."""

    types = {"t": {"s": "VARCHAR", "k": "INT"}}
    assert prover(*THROUGH_DERIVED, dialect="mysql", types=types).status is SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("types", [None, {"t": {"s": "INT", "k": "INT"}}, {"u": {"s": "VARCHAR"}}])
def test_an_untyped_or_integer_column_read_through_a_derived_table_is_not_proven(prover, types):
    assert not prover(*THROUGH_DERIVED, dialect="mysql", types=types).proven


@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("types", [{"t": {"s": "VARCHAR", "k": "INT"}}, {"t": {"s": "VARCHAR", "k": "INT"}, "u": {"s": "INT"}}])
def test_a_bare_name_read_from_two_derived_tables_is_not_tied_to_the_typed_one(prover, types):
    """The second branch reads ``c`` from ``u``, which is untyped or an integer: MySQL returns its rows where ``u.s`` is 0."""

    typed = "SELECT c FROM (SELECT s AS c FROM t) AS d WHERE c = 'a'"
    left = f"{typed} UNION ALL SELECT c FROM (SELECT s AS c FROM u) AS e WHERE c = 'x' AND c = 'y'"
    assert not prover(left, typed, dialect="mysql", types=types).proven


def test_a_bare_name_is_tied_only_when_one_source_can_hold_it():
    import sqlglot

    types = {"t": {"s": "VARCHAR", "k": "INT"}}

    def tied(sql):
        kinds = string_number_compare._Kinds(sqlglot.parse_one(sql, read="mysql"), types)
        return kinds.find(("", "c")) == kinds.find(("d", "c"))

    assert tied("SELECT c FROM (SELECT s AS c FROM t) AS d WHERE c = 'x'")
    assert not tied("SELECT c FROM (SELECT s AS c FROM t) AS d, u WHERE c = 'x'")  # u's columns are not known
    assert not tied("SELECT k AS c FROM (SELECT s AS c, k FROM t) AS d ORDER BY c")  # ORDER BY reads the select alias
    assert not tied("SELECT 1 FROM (SELECT s AS c FROM t) AS d WHERE EXISTS (SELECT 1 FROM t WHERE c = 'x')")  # read from the enclosing query


@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("types", [{"u": {"s": "VARCHAR"}}, {"t": {"k": "INT"}, "u": {"s": "VARCHAR"}}])
def test_another_tables_declared_type_does_not_type_a_bare_column(prover, types):
    """``s`` is read from ``t``, which the schema does not type (or types without ``s``); ``u.s VARCHAR`` says nothing
    about it. Both provers once borrowed that type for the bare name and proved the pair, which MySQL separates for an
    integer ``t.s`` that holds 0."""

    assert not prover("SELECT k FROM t WHERE s = 'x' AND s = 'y'", "SELECT k FROM t WHERE FALSE", dialect="mysql", types=types).proven


JOINED = (
    "SELECT t.k FROM t JOIN u ON t.k = u.k WHERE s = 'x' AND s = 'y'",
    "SELECT t.k FROM t JOIN u ON t.k = u.k WHERE FALSE",
)


@pytest.mark.parametrize("prover", PROVERS)
def test_a_schema_that_lists_the_other_tables_columns_types_the_bare_name(prover):
    """Only ``t.s`` is declared (a caller may type just the text columns); the schema says ``u`` has no ``s``, so ``s`` is ``t.s``."""

    types, schema = {"t": {"s": "VARCHAR"}}, {"t": ["k", "s"], "u": ["k", "v"]}
    assert prover(*JOINED, dialect="mysql", types=types, schema=schema).status is SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("types, schema", [
    ({"t": {"s": "VARCHAR"}}, None),  # nothing is known about u: it may hold an untyped s
    ({"t": {"s": "VARCHAR"}, "u": {"k": "INT"}}, None),  # u's types need not list all of its columns
])
def test_a_bare_name_another_read_table_may_hold_is_not_typed(prover, types, schema):
    assert not prover(*JOINED, dialect="mysql", types=types, schema=schema).proven


@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("types, schema", [
    ({"u": {"s": "VARCHAR", "k": "INT"}}, {"t": ["k", "s"], "u": ["k", "s"]}),
    ({"t": {"k": "INT"}, "u": {"s": "VARCHAR", "k": "INT"}}, None),
    ({"t": {"k": "INT"}, "u": {"s": "VARCHAR", "k": "INT"}}, {"t": ["k", "s"], "u": ["k", "s"]}),
])
def test_a_bare_name_of_an_untyped_column_is_not_typed_by_a_table_read_in_another_select(prover, types, schema):
    """``s`` is ``t.s``, which no type is declared for; ``u``, read only inside the EXISTS, declares its own ``s`` as text.
    MySQL returns the rows of an integer ``t.s`` that holds 0 on the left and none on the right."""

    left = "SELECT k FROM t WHERE s = 'x' AND s = 'y' AND EXISTS (SELECT 1 FROM u WHERE u.k = t.k)"
    right = "SELECT k FROM t WHERE FALSE"
    assert not prover(left, right, dialect="mysql", types=types, schema=schema).proven


# two names for one table read one column: under a key the two can be one row, and MySQL returns the row s = 0, k = 1
TWO_NAMES = "SELECT a.k FROM t AS a JOIN t AS b ON a.k = b.k WHERE a.s = 'x' AND b.s = 'y'"
ONE_NAME = "SELECT a.k FROM t AS a WHERE a.s = 'x' AND a.s = 'y'"


def _keyed():
    from kumo_sql_prover.smt_equivalence import TableConstraints

    return {"t": TableConstraints(not_null=frozenset({"k"}), keys=(("k",),))}


@pytest.mark.parametrize("prover", PROVERS)
def test_a_column_read_through_two_names_of_its_table_is_one_column(prover):
    options = dict(dialect="mysql", schema={"t": ["k", "s"]}, constraints=_keyed())
    assert not prover(TWO_NAMES, NONE, **options).proven
    assert not prover(TWO_NAMES, NONE, types={"t": {"s": "INT", "k": "INT"}}, **options).proven
    # the proof holds when the column is declared text, and on a dialect that refuses the comparison for an integer column
    assert prove_equivalent_algebraic(TWO_NAMES, NONE, types={"t": {"s": "VARCHAR", "k": "INT"}}, **options).proven
    assert prove_equivalent_algebraic(TWO_NAMES, NONE, **{**options, "dialect": "postgres"}).proven


def test_the_reading_replaces_the_strings_under_every_name_of_the_table():
    found = numeric_column_reading.reading(TWO_NAMES, ONE_NAME, "mysql")
    assert found.pair[0] == "SELECT a.k FROM t AS a JOIN t AS b ON a.k = b.k WHERE a.s = 0 AND b.s = 0"
    assert found.pair[1] == "SELECT a.k FROM t AS a WHERE a.s = 0 AND a.s = 0"
    # a subquery that reads the table again under another name
    nested = "SELECT a.k FROM t AS a WHERE a.s = 'x' AND NOT EXISTS (SELECT 1 FROM t AS b WHERE b.k = a.k AND b.s = 'y')"
    assert "'" not in numeric_column_reading.reading(nested, NONE, "mysql").pair[0]
    # two tables are two columns, as before
    assert numeric_column_reading.reading("SELECT a.k FROM t AS a JOIN u AS b ON a.k = b.k WHERE a.s = 'x' AND b.s = 'y'", NONE, "mysql") is None


@pytest.mark.parametrize("prover", PROVERS)
def test_a_proof_that_holds_for_both_readings_of_two_names_stays(prover):
    left = "SELECT a.k FROM t AS a JOIN t AS b ON a.k = b.k WHERE a.s = 'x' AND b.s = 'y'"
    right = "SELECT b.k FROM t AS b JOIN t AS a ON b.k = a.k WHERE b.s = 'y' AND a.s = 'x'"
    assert prover(left, right, dialect="mysql", schema={"t": ["k", "s"]}).status is SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize("prover", PROVERS)
def test_an_enum_column_is_read_as_its_string(prover):
    # VeriEQL declares ``ACTIVITY_TYPE`` as an ENUM of strings: two names of the table compared with two different strings
    # are two different values, not the number 0 twice (the pairs of LeetCode 1661 prove under a key on the join columns)
    options = dict(dialect="mysql", schema={"t": ["k", "s"]}, constraints=_keyed())
    assert prover(TWO_NAMES, NONE, types={"t": {"s": "ENUM", "k": "INT"}}, **options).status is SmtStatus.PROVEN_EQUIVALENT
    assert not prover(TWO_NAMES, NONE, types={"t": {"s": "INT", "k": "INT"}}, **options).proven
