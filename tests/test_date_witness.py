"""A counterexample puts dates in a declared date column (2026-11-17).

The SMT prover writes a date as the string it is spelled as, and the solver filled a column declared ``DATE`` with
whatever separated the two readings as text: ``'2019-06-270'`` between two days, the number 20190627, ``FALSE``. Three
answers were wrong because of it, each checked on MySQL 8.0.46 over ``act(u INT, d DATE)``:

- ``d BETWEEN '2019-06-28' AND '2019-07-27'`` and ``d > '2019-06-27' AND d <= '2019-07-27'`` return the same rows for a
  date column and were called different on the row ``d = '2019-06-270'``;
- so were ``d > '2019-06-27'`` and ``d >= '2019-06-28'`` (row ``'2019-06-27a'``);
- ``d = 20190627`` and ``d = '2019-06-27'`` return the same rows on MySQL and were called different on ``d = 20190627``.

One proof was false: ``CAST('2019-06-27 10:00:00' AS DATE)`` was read as the string inside it, so
``d = CAST('2019-06-27 10:00:00' AS DATE)`` was proved equal to ``d = '2019-06-27 10:00:00'``. For a ``DATE`` column
MySQL returns the row of the 27th for the first and nothing for the second; for a ``DATETIME`` column MySQL and DuckDB
return the midnight row for the first and the ten o'clock row for the second.
"""

import duckdb
import pytest

pytest.importorskip("z3")

from kumo_sql_prover import date_witness
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtStatus, prove_equivalent_smt

PROVERS = (prove_equivalent_algebraic, prove_equivalent_smt)
SCHEMA = {"act": ["u", "d"], "log": ["u", "d", "note"]}
KINDS = ("DATE", "DATETIME", "TIMESTAMP")


def types(kind: str | None) -> dict | None:
    if kind is None:
        return None
    return {"act": {"u": "INT", "d": kind}, "log": {"u": "INT", "d": kind, "note": "VARCHAR"}}


def where(condition: str) -> str:
    return "SELECT a.u FROM act AS a WHERE " + condition


def prove(prover, left: str, right: str, kind: str | None, dialect: str = "mysql"):
    return prover(left, right, schema=SCHEMA, types=types(kind), compare_names=False, dialect=dialect, exact_arithmetic=True, timeout_ms=4000)


def date_cells(result) -> list:
    return [row["d"] for rows in result.counterexample.tables.values() for row in rows if "d" in row]


def replay(result, left: str, right: str, kind: str) -> tuple[list, list]:
    """Both queries on DuckDB over the counterexample's tables, the date column loaded as the declared type."""

    con = duckdb.connect()
    stored = "TIMESTAMP" if kind in ("DATETIME", "TIMESTAMP") else kind
    for table, columns in SCHEMA.items():
        con.execute(f"CREATE TABLE {table} ({', '.join(c + (' ' + stored if c == 'd' else ' VARCHAR' if c == 'note' else ' INT') for c in columns)})")
        for row in result.counterexample.tables.get(table, []):
            con.execute(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in columns)})", [None if row.get(c) is None else str(row[c]) if c in ("d", "note") else int(row[c]) for c in columns])
    return tuple(sorted(con.execute(sql).fetchall(), key=repr) for sql in (left, right))


# ---- which strings a date column holds ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["2019-06-27", "2020-02-29", "1000-01-01", "9999-12-31"])
def test_a_day_is_held_by_a_date_and_by_a_timestamp_column(text):
    assert date_witness.holds(text, date_witness.DATE) and date_witness.holds(text, date_witness.STAMP)


@pytest.mark.parametrize(
    "value",
    ["2019-06-270", "2019-06-27a", "2019-6-27", "2019/06/27", "20190627", "2019-02-30", "0999-01-01", "", " 2019-06-27", 20190627, 0, False, None],
)
def test_what_is_not_a_day_is_held_by_neither(value):
    assert not date_witness.holds(value, date_witness.DATE) and not date_witness.holds(value, date_witness.STAMP)


def test_a_time_of_day_is_held_only_by_a_column_that_keeps_it_and_midnight_is_the_day_alone():
    assert date_witness.holds("2019-06-27 12:00:00", date_witness.STAMP)
    assert not date_witness.holds("2019-06-27 12:00:00", date_witness.DATE)
    assert not date_witness.holds("2019-06-27 00:00:00", date_witness.STAMP)  # one spelling a moment: the day alone
    assert not date_witness.holds("2019-06-27 24:00:00", date_witness.STAMP)
    assert not date_witness.holds("2019-06-27 12:00:00.5", date_witness.STAMP)
    assert not date_witness.holds("2019-06-27", date_witness.NODATE)


def test_the_spellings_sort_as_text_the_way_the_moments_sort_in_time():
    moments = date_witness.domain({"2019-06-27", "2019-12-31 23:59:59", "2020-01-01"}, date_witness.STAMP)
    assert moments == sorted(moments) and all(date_witness.holds(text, date_witness.STAMP) for text in moments)
    assert "2019-06-27" in moments and "2019-06-27 12:00:00" in moments and "2019-12-31 23:59:59" in moments
    days = date_witness.domain({"2019-06-27", "2019-12-31 23:59:59", "x"}, date_witness.DATE)
    assert days == sorted(days) and all(date_witness.holds(text, date_witness.DATE) for text in days)
    assert {"2019-06-26", "2019-06-27", "2019-06-28", "2019-12-31", "2020-01-01"} <= set(days)


def test_with_no_day_named_the_choice_is_a_few_real_days():
    assert date_witness.domain({"x", "10"}, date_witness.DATE) == ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04", "2000-01-05"]


# ---- which queries read their date columns the way the text follows ------------------------------------------------

READABLE = [
    where("a.d > '2019-06-27' AND a.d <= '2019-07-27'"),
    where("a.d BETWEEN '2019-06-28' AND '2019-07-27'"),
    where("a.d IN ('2019-06-28', '2019-07-27') OR a.d IS NULL"),
    where("a.d = CAST('2019-06-28' AS DATE)"),
    "SELECT a.d, COUNT(*) AS n, COUNT(DISTINCT a.d) AS days FROM act AS a GROUP BY a.d ORDER BY a.d",
    "SELECT a.u, MIN(a.d) AS first_day FROM act AS a GROUP BY a.u HAVING MAX(a.d) >= '2019-06-28'",
    "SELECT a.u FROM act AS a JOIN log AS l ON a.d = l.d AND a.u < l.u",
    "SELECT x.first_day FROM (SELECT a.u, MIN(a.d) AS first_day FROM act AS a GROUP BY a.u) AS x WHERE x.first_day < '2019-06-28'",
    "SELECT a.u FROM act AS a WHERE a.d IN (SELECT l.d FROM log AS l)",
    "SELECT a.u, ROW_NUMBER() OVER (PARTITION BY a.d ORDER BY a.u) AS r FROM act AS a",
    "SELECT a.u FROM act AS a WHERE a.u > 3",
]

NOT_READABLE = [
    where("LENGTH(a.d) = 10"),
    where("a.d = 20190627"),
    where("a.d + 1 > 20190627"),
    where("a.d = '2019/06/27'"),
    where("a.d = '2019-6-27'"),
    where("a.d = '2019-06-27 00:00:00'"),
    where("a.d = CAST('2019-06-27 10:00:00' AS DATE)"),
    where("a.d > CAST('2019-06-27' AS CHAR)"),
    where("COALESCE(a.d, '2019-06-27') = '2019-06-27'"),
    where("NULLIF(a.d, '2019-06-27') IS NULL"),
    where("CONCAT(a.d, 'x') = '2019-06-27x'"),
    "SELECT a.u FROM act AS a JOIN log AS l ON a.d = l.note",
    "SELECT a.u FROM act AS a JOIN log AS l USING (d)",
    "SELECT a.d FROM act AS a UNION ALL SELECT l.note FROM log AS l",
    "SELECT x.first_day FROM (SELECT a.u, MIN(a.d) AS first_day FROM act AS a GROUP BY a.u) AS x WHERE YEAR(x.first_day) = 2019",
    "SELECT CASE WHEN a.u > 1 THEN a.d END AS shown FROM act AS a",
    "SELECT a.u FROM act AS a JOIN other AS o ON a.d = o.d",  # a table of no declared types: its d can be text
]


@pytest.mark.parametrize("sql", READABLE)
def test_plain_uses_of_a_date_column_are_read_as_the_engines_read_them(sql):
    assert date_witness.readable(sql, "mysql", types("DATE"), SCHEMA)


@pytest.mark.parametrize("sql", NOT_READABLE)
def test_other_uses_are_not(sql):
    assert not date_witness.readable(sql, "mysql", types("DATE"), SCHEMA)


def test_a_time_of_day_literal_is_read_only_against_a_column_that_keeps_the_time():
    sql = where("a.d >= '2019-06-27 10:00:00'")
    assert not date_witness.readable(sql, "mysql", types("DATE"), SCHEMA)  # DuckDB reads the day, MySQL the moment
    assert date_witness.readable(sql, "mysql", types("DATETIME"), SCHEMA)
    assert date_witness.readable(sql, "mysql", types("TIMESTAMP"), SCHEMA)


def test_queries_over_tables_with_no_declared_date_are_not_looked_at():
    assert date_witness.readable(where("LENGTH(a.d) = 10"), "mysql", {"act": {"u": "INT", "d": "VARCHAR"}}, SCHEMA)
    assert date_witness.readable(where("LENGTH(a.d) = 10"), "mysql", None, SCHEMA)


# ---- the wrong answers ----------------------------------------------------------------------------------------------

SAME_FOR_A_DATE = [
    (where("a.d BETWEEN '2019-06-28' AND '2019-07-27'"), where("a.d > '2019-06-27' AND a.d <= '2019-07-27'")),
    (where("a.d > '2019-06-27'"), where("a.d >= '2019-06-28'")),
    (where("a.d = 20190627"), where("a.d = '2019-06-27'")),
    (where("a.d < '2019-06-28' AND a.d > '2019-06-26'"), where("a.d = '2019-06-27'")),
]


@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("left,right", SAME_FOR_A_DATE)
def test_queries_that_are_the_same_for_a_date_column_are_not_called_different(prover, left, right):
    assert prove(prover, left, right, "DATE").status is not SmtStatus.NOT_EQUIVALENT


DIFFERENT = [
    (where("a.d BETWEEN '2019-06-28' AND '2019-07-27'"), where("a.d > '2019-06-28' AND a.d <= '2019-07-27'")),
    (where("a.d >= '2019-06-27'"), where("a.d > '2019-06-27'")),
    (where("a.d <> '2019-06-27'"), where("a.d < '2019-06-27'")),
    ("SELECT a.d FROM act AS a WHERE a.u > 1 AND a.d < '2020-01-01'", "SELECT a.d FROM act AS a WHERE a.u >= 1 AND a.d < '2020-01-01'"),
    ("SELECT a.u FROM act AS a JOIN act AS b ON a.d = b.d AND a.u < b.u", "SELECT a.u FROM act AS a JOIN act AS b ON a.d <= b.d AND a.u < b.u"),
    ("SELECT a.d, COUNT(*) AS n FROM act AS a GROUP BY a.d", "SELECT a.d, COUNT(a.u) AS n FROM act AS a GROUP BY a.d"),
    ("SELECT a.u FROM act AS a JOIN log AS l ON a.d = l.d", "SELECT a.u FROM act AS a JOIN log AS l ON a.d = l.d AND a.d >= '2019-06-27'"),
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("left,right", DIFFERENT)
def test_a_difference_is_shown_on_dates_and_an_engine_agrees(prover, left, right, kind):
    result = prove(prover, left, right, kind)
    assert result.status is SmtStatus.NOT_EQUIVALENT
    stored = date_witness.kind(kind)
    assert all(cell is None or date_witness.holds(cell, stored) for cell in date_cells(result))
    left_rows, right_rows = replay(result, left, right, kind)
    assert left_rows != right_rows


@pytest.mark.parametrize("prover", PROVERS)
def test_a_timestamp_column_separates_what_a_date_column_does_not(prover):
    left, right = where("a.d > '2019-06-27'"), where("a.d >= '2019-06-28'")
    result = prove(prover, left, right, "DATETIME")
    assert result.status is SmtStatus.NOT_EQUIVALENT
    assert date_cells(result) == ["2019-06-27 12:00:00"]
    left_rows, right_rows = replay(result, left, right, "DATETIME")
    assert left_rows != right_rows


UNCOVERED = [
    (where("LENGTH(a.d) = 10"), where("a.d IS NOT NULL")),
    (where("a.d = 20190627 OR a.u = 1"), where("a.u = 1")),
    (where("a.d = '2019/06/27' OR a.u = 1"), where("a.d >= '2019/06/27' OR a.u = 2")),
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("prover", PROVERS)
@pytest.mark.parametrize("left,right", UNCOVERED)
def test_where_the_reading_of_dates_is_not_covered_a_counterexample_holds_no_date(prover, left, right, kind):
    result = prove(prover, left, right, kind)
    if result.status is SmtStatus.NOT_EQUIVALENT:
        assert all(cell is None for cell in date_cells(result))


@pytest.mark.parametrize("prover", PROVERS)
def test_a_column_with_no_declared_type_keeps_its_text_row(prover):
    result = prove(prover, where("a.d > '2019-06-27'"), where("a.d >= '2019-06-28'"), None)
    assert result.status is SmtStatus.NOT_EQUIVALENT  # for a text column the two differ, and the row is a text


@pytest.mark.parametrize("prover", PROVERS)
def test_a_declared_text_column_keeps_its_text_row(prover):
    kinds = {"act": {"u": "INT", "d": "VARCHAR"}}
    result = prover(where("a.d > '2019-06-27'"), where("a.d >= '2019-06-28'"), schema=SCHEMA, types=kinds, compare_names=False, dialect="mysql", timeout_ms=4000)
    assert result.status is SmtStatus.NOT_EQUIVALENT


# ---- the false proof --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dialect", ["mysql", "duckdb", "postgres"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("prover", PROVERS)
def test_a_cast_of_a_date_and_time_string_to_a_date_is_not_the_string(prover, kind, dialect):
    result = prove(prover, where("a.d = CAST('2019-06-27 10:00:00' AS DATE)"), where("a.d = '2019-06-27 10:00:00'"), kind, dialect)
    assert result.status is not SmtStatus.PROVEN_EQUIVALENT


def test_the_cast_and_the_string_differ_on_an_engine():
    con = duckdb.connect()
    con.execute("CREATE TABLE act (u INT, d TIMESTAMP)")
    con.execute("INSERT INTO act VALUES (1, '2019-06-27 00:00:00'), (2, '2019-06-27 10:00:00')")
    assert con.execute(where("a.d = CAST('2019-06-27 10:00:00' AS DATE)")).fetchall() == [(1,)]
    assert con.execute(where("a.d = '2019-06-27 10:00:00'")).fetchall() == [(2,)]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("prover", PROVERS)
def test_a_cast_of_a_day_is_still_the_day(prover, kind):
    assert prove(prover, where("a.d = CAST('2019-06-27' AS DATE)"), where("a.d = '2019-06-27'"), kind).status is SmtStatus.PROVEN_EQUIVALENT
