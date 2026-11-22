"""A number used as a condition is not "never TRUE".

``WHERE paid - 3`` was proven equal to ``WHERE 1 = 0``: the SMT model called a value TRUE or FALSE only when it was a
Boolean, so a number in a condition's place let no row through. MySQL, SQLite and DuckDB read a number as a condition
(TRUE when it is not zero, FALSE when it is zero), so that filter keeps every row whose ``paid`` is not 3. The same
model refuted pairs that are equal there (``WHERE paid`` against ``WHERE paid <> 0``) with a database on which both
return the same rows.

The provers now read a number that way in those three dialects. A text in a condition's place is converted by the
engine in a way the model does not cover: its truth is left open, so only a pair that tests the same text on both
sides is proven through it and no counterexample is reported. Everywhere else a value known to be a number or a text
is declined as a condition (BigQuery and PostgreSQL reject it; the others have not been checked).
"""

import itertools
import sqlite3

import pytest

pytest.importorskip("z3")
duckdb = pytest.importorskip("duckdb")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtStatus, prove_equivalent_smt

SCHEMA = {"sale": ["item_id", "buyer", "paid"]}
TYPES = {"sale": {"item_id": "INT", "buyer": "VARCHAR", "paid": "INT"}}
PROVERS = (prove_equivalent_algebraic, prove_equivalent_smt)
LAX = ("mysql", "sqlite", "duckdb")
STRICT = ("bigquery", "postgres", "spark", "snowflake")
EVERY = "SELECT buyer FROM sale"


def select(where: str) -> str:
    return f"SELECT buyer FROM sale WHERE {where}"


# A value the query itself shows to be a number, in a condition's place. Each pair differs on some table in MySQL,
# SQLite and DuckDB, and no dialect may call it proven.
KNOWN = [
    (select("paid - 3"), select("1 = 0")),
    (select("NOT (paid - 3)"), select("1 = 0")),
    (select("paid * 2"), select("paid * 2 = 1")),
    (select("1"), select("1 = 0")),
    (select("0.5"), select("1 = 0")),
    (select("0"), EVERY),
    ("SELECT a.buyer FROM sale AS a JOIN sale AS b ON a.paid - b.paid", "SELECT a.buyer FROM sale AS a JOIN sale AS b ON 1 = 0"),
    (select("paid - 3 AND item_id > 3"), select("1 = 0")),
    (select("paid - 3 OR item_id > 3"), select("item_id > 3")),
    ("SELECT CASE WHEN paid - 1 THEN 'a' ELSE 'b' END FROM sale", "SELECT 'b' FROM sale"),
]

# A column in a condition's place: a number in MySQL, SQLite and DuckDB, where each pair differs on some table.
DIFFERENT = KNOWN + [
    (select("paid"), select("1 = 0")),
    (select("paid"), select("paid = 1")),
    (select("paid"), select("paid > 0")),
    (select("paid = TRUE"), select("paid")),
    (select("paid = TRUE"), select("paid <> 0")),
    (select("NOT paid"), select("1 = 0")),
    (select("paid AND item_id > 3"), select("1 = 0")),
    (select("paid OR item_id > 3"), select("item_id > 3")),
    (select("CASE WHEN paid THEN 1 ELSE 0 END = 1"), select("paid = 1")),
    ("SELECT CASE WHEN paid THEN 'a' ELSE 'b' END FROM sale", "SELECT CASE WHEN paid = 1 THEN 'a' ELSE 'b' END FROM sale"),
    ("SELECT buyer FROM sale GROUP BY buyer HAVING SUM(paid)", "SELECT buyer FROM sale GROUP BY buyer HAVING 1 = 0"),
    ("SELECT COUNT(*) FROM sale WHERE paid", "SELECT SUM(paid) FROM sale"),
]

# A text as a condition: MySQL and SQLite convert it to a number first ('1' is TRUE, 'x' is FALSE), DuckDB to a Boolean.
TEXT = [
    (select("'x'"), select("'1'")),
    (select("'1'"), select("1 = 0")),
    (select("buyer"), select("1 = 0")),
    (select("LENGTH(buyer)"), select("1 = 0")),
]

# Pairs that return the same rows in MySQL, SQLite and DuckDB because a number is TRUE when it is not zero.
EQUAL = [
    (select("paid - 3"), select("paid <> 3")),
    (select("NOT (paid - 3)"), select("paid = 3")),
    (select("paid * 2"), select("paid <> 0")),
    (select("1"), EVERY),
    (select("0"), select("1 = 0")),
    (select("2 - 2"), select("1 = 0")),
    (select("paid AND paid"), select("paid")),
    (select("NOT (NOT paid)"), select("paid")),
    (select("COALESCE(paid, 0)"), select("paid")),
]

# Equal there too, though not all proven: the least asked is that none is refuted.
NEVER_REFUTED = EQUAL + [
    (select("paid"), select("paid <> 0")),
    (select("NOT paid"), select("paid = 0")),
    (select("paid IS TRUE"), select("paid <> 0")),
    (select("paid AND item_id > 3"), select("paid <> 0 AND item_id > 3")),
]


def prove(prover, left, right, dialect, types=None):
    # exact arithmetic, as the eval harnesses ask for: ``paid - 3`` is then a number the model knows
    return prover(left, right, schema=SCHEMA, types=types, dialect=dialect, exact_arithmetic=True)


@pytest.mark.parametrize("dialect", LAX + STRICT)
@pytest.mark.parametrize("types", (None, TYPES))
@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("left,right", KNOWN + TEXT)
def test_a_number_in_a_conditions_place_is_not_never_true(prover, types, dialect, left, right):
    assert prove(prover, left, right, dialect, types).status is not SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize("dialect", LAX)
@pytest.mark.parametrize("types", (None, TYPES))
@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("left,right", DIFFERENT[len(KNOWN):])
def test_a_column_in_a_conditions_place_is_not_never_true(prover, types, dialect, left, right):
    assert prove(prover, left, right, dialect, types).status is not SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize("dialect", LAX)
@pytest.mark.parametrize("types", (None, TYPES))
@pytest.mark.parametrize("left,right", EQUAL)
def test_a_number_is_true_when_it_is_not_zero(types, dialect, left, right):
    assert prove(prove_equivalent_algebraic, left, right, dialect, types).status is SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize("dialect", LAX)
@pytest.mark.parametrize("types", (None, TYPES))
@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("left,right", NEVER_REFUTED)
def test_pairs_equal_in_the_engine_are_not_refuted(prover, types, dialect, left, right):
    result = prove(prover, left, right, dialect, types)
    assert result.status is not SmtStatus.NOT_EQUIVALENT, result.counterexample


@pytest.mark.parametrize("dialect", LAX)
@pytest.mark.parametrize("left,right", TEXT)
def test_a_text_in_a_conditions_place_gives_no_counterexample(dialect, left, right):
    # whatever the model says about the conversion is not a database
    assert prove(prove_equivalent_algebraic, left, right, dialect).status is SmtStatus.NOT_PROVEN


@pytest.mark.parametrize("dialect", STRICT)
@pytest.mark.parametrize("left,right", KNOWN[:5] + EQUAL[:3])
def test_other_dialects_decline_a_known_number(dialect, left, right):
    result = prove(prove_equivalent_algebraic, left, right, dialect)
    assert result.status is SmtStatus.NOT_PROVEN
    assert "used as a condition" in result.reason


def test_bigquery_declines_a_column_declared_a_number():
    types = {"sale": {"item_id": "INT64", "buyer": "STRING", "paid": "INT64"}}
    for where in ("paid", "buyer", "paid AND item_id > 3"):
        result = prove_equivalent_algebraic(select(where), select("1 = 0"), schema=SCHEMA, types=types, dialect="bigquery")
        assert result.status is SmtStatus.NOT_PROVEN, where


@pytest.mark.parametrize("dialect", STRICT)
def test_other_dialects_still_read_a_boolean_column(dialect):
    schema = {"sale": ["buyer", "gift", "paid"]}
    types = {"sale": {"buyer": "STRING", "gift": "BOOL", "paid": "INT64"}}
    for typed in (None, types):
        kwargs = {"schema": schema, "types": typed, "dialect": dialect}
        proved = prove_equivalent_algebraic(select("gift AND gift"), select("gift"), **kwargs)
        assert proved.status is SmtStatus.PROVEN_EQUIVALENT
        assert prove_equivalent_algebraic(select("NOT (NOT gift)"), select("gift"), **kwargs).status is SmtStatus.PROVEN_EQUIVALENT
        assert prove_equivalent_algebraic(select("gift"), select("1 = 0"), **kwargs).status is not SmtStatus.PROVEN_EQUIVALENT
        assert prove_equivalent_algebraic(select("gift AND paid > 3"), select("paid > 3 AND gift"), **kwargs).status is SmtStatus.PROVEN_EQUIVALENT


VALUES = (None, 0, 1, 2, 3, -1)


def _tables():
    """Every one-row table over the values above, and a few two-row tables."""

    for paid in VALUES:
        for item_id in (2, 5):
            yield [(item_id, "ann", paid)]
    for first, second in itertools.combinations(VALUES, 2):
        yield [(1, "ann", first), (7, "bob", second)]


def _sqlite_rows(sql: str, rows: list) -> list:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE sale (item_id INTEGER, buyer TEXT, paid INTEGER)")
    connection.executemany("INSERT INTO sale VALUES (?, ?, ?)", rows)
    return sorted(connection.execute(sql).fetchall(), key=repr)


def _duckdb_rows(sql: str, rows: list) -> list | None:
    connection = duckdb.connect()
    connection.execute("CREATE TABLE sale (item_id INTEGER, buyer VARCHAR, paid INTEGER)")
    connection.executemany("INSERT INTO sale VALUES (?, ?, ?)", rows)
    try:
        return sorted(connection.execute(sql).fetchall(), key=repr)
    except duckdb.Error:
        return None  # DuckDB rejects the query (a text that is not a Boolean, a decimal as a condition)


@pytest.mark.parametrize("engine", (_sqlite_rows, _duckdb_rows))
def test_the_labels_agree_with_the_engines(engine):
    """The lists above are what SQLite and DuckDB do: a DIFFERENT pair differs on one of the tables, an equal pair on
    none. (A pair an engine rejects is skipped for that engine.)"""

    checked = 0
    for pairs, differs in ((DIFFERENT, True), (NEVER_REFUTED, False)):
        for left, right in pairs:
            outcomes = [(engine(left, rows), engine(right, rows)) for rows in _tables()]
            if any(a is None or b is None for a, b in outcomes):
                continue
            checked += 1
            assert any(a != b for a, b in outcomes) is differs, (left, right)
    assert checked >= 25


def test_sqlite_reads_a_text_as_its_number():
    assert _sqlite_rows(select("'x'"), [(1, "ann", 1)]) == []
    assert _sqlite_rows(select("'1'"), [(1, "ann", 1)]) == [("ann",)]
    assert _sqlite_rows(select("buyer"), [(1, "1", 1), (2, "ann", 1)]) == [("1",)]
