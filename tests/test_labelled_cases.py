"""A ``CASE`` of labels compared with one label is the condition that picks that label (``labelled_cases``)."""

import itertools

import duckdb
import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.labelled_cases import NAME, negate, read_labelled_cases
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"visits": ["id", "secs", "kind"], "people": ["id", "name"]}
TYPES = {"visits": {"id": "INT", "secs": "INT", "kind": "VARCHAR"}, "people": {"id": "INT", "name": "VARCHAR"}}
CONSTRAINTS = {
    "visits": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
    "people": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
}


def written(sql: str, dialect: str = "mysql") -> str | None:
    select = sqlglot.parse_one(sql, read=dialect)
    before = select.sql(dialect=dialect)
    new = read_labelled_cases(select)
    assert select.sql(dialect=dialect) == before  # the select it was given is not changed
    return None if new is None else new.sql(dialect=dialect)


def prove(left: str, right: str, dialect: str = "postgres"):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect=dialect, exact_arithmetic=True, timeout_ms=5000
    )


@pytest.mark.parametrize(
    "sql, expected",
    [
        (  # an arm: its condition, and the earlier ones are not TRUE (secs is not NULL here, so NOT is enough)
            "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'short' WHEN secs < 900 THEN 'medium' ELSE 'long' END = 'medium'",
            "SELECT id FROM visits WHERE ((secs < 900) AND secs >= 300)",
        ),
        (  # ELSE: no condition is TRUE, which a NULL does too
            "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'short' WHEN secs < 900 THEN 'medium' ELSE 'long' END = 'long'",
            "SELECT id FROM visits WHERE (secs IS NULL OR (secs >= 300 AND secs >= 900))",
        ),
        (  # one term of the filter, the label written first
            "SELECT id FROM visits WHERE 'short' = CASE WHEN secs < 300 THEN 'short' WHEN secs < 900 THEN 'medium' WHEN secs >= 900 THEN 'long' END AND id > 3",
            "SELECT id FROM visits WHERE ((secs < 300)) AND id > 3",
        ),
        (  # two arms hold the label
            "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'a' WHEN secs < 900 THEN 'a' ELSE 'c' END = 'a'",
            "SELECT id FROM visits WHERE (((secs < 300)) OR ((secs < 900) AND secs >= 300))",
        ),
        (  # conditions on different columns: "not TRUE" is spelled out
            "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'a' WHEN kind = 'x' THEN 'b' ELSE 'a' END = 'a'",
            "SELECT id FROM visits WHERE (((secs < 300)) OR ((secs >= 300 OR (secs < 300) IS NULL) AND (kind <> 'x' OR (kind = 'x') IS NULL)))",
        ),
        (  # an OR that one column alone can satisfy
            "SELECT id FROM visits WHERE CASE WHEN secs < 300 OR id > 5 THEN 'a' WHEN secs < 900 THEN 'b' ELSE 'c' END = 'c'",
            "SELECT id FROM visits WHERE (((secs >= 300 AND id <= 5) OR (secs < 300 OR id > 5) IS NULL) AND (secs >= 900 OR (secs < 900) IS NULL))",
        ),
        ("SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 1 WHEN secs < 900 THEN 2 END = 2.0", "SELECT id FROM visits WHERE ((secs < 900) AND secs >= 300)"),
        (
            "SELECT kind, COUNT(*) FROM visits GROUP BY kind HAVING CASE WHEN kind = 'x' THEN 'a' ELSE 'b' END = 'b'",
            "SELECT kind, COUNT(*) FROM visits GROUP BY kind HAVING (kind IS NULL OR (kind <> 'x'))",
        ),
        (
            "SELECT v.id FROM visits AS v JOIN people AS p ON p.id = v.id AND CASE WHEN v.secs BETWEEN 1 AND 5 THEN 'a' WHEN v.secs > 5 THEN 'b' END = 'b'",
            "SELECT v.id FROM visits AS v JOIN people AS p ON p.id = v.id AND ((v.secs > 5) AND (v.secs < 1 OR v.secs > 5))",
        ),
        (
            "SELECT id FROM visits WHERE CASE WHEN NOT secs < 300 THEN 'a' WHEN secs IS NULL THEN 'b' ELSE 'c' END = 'b'",
            "SELECT id FROM visits WHERE ((secs IS NULL) AND ((secs < 300) OR (NOT secs < 300) IS NULL))",
        ),
    ],
)
def test_the_filter_is_written_out(sql, expected):
    assert written(sql) == expected


def test_a_quoted_column_keeps_its_spelling():
    sql = """SELECT id FROM visits WHERE CASE WHEN "Secs" < 300 THEN 'a' ELSE 'b' END = 'b'"""
    assert written(sql, "postgres") == """SELECT id FROM visits WHERE ("Secs" IS NULL OR ("Secs" >= 300))"""
    # "Secs" and secs are two columns there: neither is known to be set where the other is compared
    sql = """SELECT id FROM visits WHERE CASE WHEN "Secs" < 300 THEN 'a' WHEN secs < 300 THEN 'b' END = 'b'"""
    assert written(sql, "postgres") == """SELECT id FROM visits WHERE ((secs < 300) AND ("Secs" >= 300 OR ("Secs" < 300) IS NULL))"""


CONDITIONS = [
    "secs < 300",
    "secs >= 900",
    "secs BETWEEN 300 AND 899",
    "NOT secs < 300",
    "secs < 300 OR id > 5",
    "secs < 300 AND id > 5",
    "kind = 'x'",
    "kind <> 'x' OR secs > 10",
    "secs IS NULL",
    "NOT (secs < 300 AND NOT kind = 'x')",
    "secs + id < 12",
    "secs IN (1, 300)",
    "secs / 2 > 3",
    "id > 5",
    "NOT (secs BETWEEN 2 AND 900 OR id = 3)",
]


def table():
    con = duckdb.connect()
    con.execute("CREATE TABLE visits (id INT, secs INT, kind VARCHAR)")
    for secs in (None, 1, 2, 5, 11, 299, 300, 899, 900, 5000):
        for kind in (None, "x", "y"):
            for id_ in (3, 5, 6, 11):
                con.execute("INSERT INTO visits VALUES (?, ?, ?)", [id_, secs, kind])
    return con


def test_negate_is_not_with_nulls_in_duckdb():
    con = table()
    for condition in CONDITIONS:
        pushed = negate(sqlglot.parse_one(condition, read="duckdb")).sql(dialect="duckdb")
        differ = con.execute(f"SELECT COUNT(*) FROM visits WHERE (NOT ({condition})) IS DISTINCT FROM ({pushed})").fetchone()[0]
        assert differ == 0, (condition, pushed)


def test_the_reading_returns_the_same_rows_in_duckdb():
    con = table()
    checked = 0
    for first, second, third in itertools.permutations(CONDITIONS, 3):
        if checked % 20:  # one arrangement in twenty keeps the test short
            checked += 1
            continue
        checked += 1
        for results in (("'a'", "'b'", "'a'", "'c'"), ("'a'", "'b'", "NULL", "'a'"), ("'b'", "'a'", "'a'", None), ("'c'", "'a'", "'b'", "'a'")):
            default = f" ELSE {results[3]}" if results[3] else ""
            case = f"CASE WHEN {first} THEN {results[0]} WHEN {second} THEN {results[1]} WHEN {third} THEN {results[2]}{default} END"
            sql = f"SELECT id, secs, kind FROM visits WHERE {case} = 'a'"
            new = written(sql, "duckdb")
            assert new is not None, sql
            assert sorted(con.execute(sql).fetchall(), key=repr) == sorted(con.execute(new).fetchall(), key=repr), sql
    assert checked > 2000


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM visits WHERE CASE kind WHEN 'x' THEN 'a' ELSE 'b' END = 'b'",  # a CASE with an operand
        "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN kind ELSE 'b' END = 'b'",  # a result that is no literal
        "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'a' ELSE 'b' END = kind",
        "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'a' ELSE 'b' END = NULL",
        "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'a' ELSE 1 END = 'a'",  # text and number
        "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 1 ELSE 2 END = '1'",
        "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'a' ELSE 'A' END = 'a'",  # equal under some collations
        "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'a' ELSE 'b' END = 'A'",
        "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'a ' ELSE 'b' END = 'a'",  # padded
        "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'é' ELSE 'b' END = 'b'",
        "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'a' ELSE 'b' END = 'c'",  # no arm holds the label
        "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'a' ELSE 'b' END <> 'a'",  # NULL for a NULL result
        "SELECT id FROM visits WHERE NOT CASE WHEN secs < 300 THEN 'a' END = 'a'",  # under NOT, NULL is not FALSE
        "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'a' END = 'a' OR id > 3",
        "SELECT id, CASE WHEN secs < 300 THEN 'a' END = 'a' FROM visits",  # the value is shown
        "SELECT id FROM visits WHERE CASE WHEN ABS(secs) < 300 THEN 'a' ELSE 'b' END = 'b'",  # a function call
        "SELECT id FROM visits WHERE CASE WHEN secs < (SELECT MAX(id) FROM people) THEN 'a' ELSE 'b' END = 'b'",
        "SELECT id FROM visits WHERE CASE WHEN EXISTS (SELECT 1 FROM people) THEN 'a' ELSE 'b' END = 'b'",
        "SELECT kind FROM visits GROUP BY kind HAVING CASE WHEN COUNT(*) > 2 THEN 'a' ELSE 'b' END = 'b'",  # an aggregate
        "SELECT id FROM visits WHERE CASE WHEN kind LIKE 'x%' THEN 'a' ELSE 'b' END = 'b'",
        "SELECT id FROM visits WHERE CASE WHEN id = 0 THEN 'a' WHEN secs / id > 2 THEN 'b' END = 'b'",  # the first arm shields the division
        "SELECT id FROM visits WHERE CASE WHEN secs / 0 > 2 THEN 'b' END = 'b'",
        "SELECT id FROM visits WHERE secs < 300",
    ],
)
def test_other_filters_are_left_alone(sql):
    assert written(sql) is None


PAIRS = [
    (
        "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'short' WHEN secs < 900 THEN 'medium' ELSE 'long' END = 'medium'",
        "SELECT id FROM visits WHERE secs BETWEEN 300 AND 899",
    ),
    (
        "SELECT COUNT(*) FROM (SELECT id, CASE WHEN secs < 300 THEN 'short' WHEN secs < 900 THEN 'medium' ELSE 'long' END AS bin FROM visits) AS t WHERE t.bin = 'long'",
        "SELECT COUNT(*) FROM visits WHERE secs > 899 OR secs IS NULL",
    ),
    (
        "SELECT id FROM visits WHERE CASE WHEN secs <= 299 THEN 1 WHEN secs <= 899 THEN 2 WHEN secs > 899 THEN 3 END = 2",
        "SELECT id FROM visits WHERE secs BETWEEN 300 AND 899",
    ),
]


@pytest.mark.parametrize("left, right", PAIRS)
def test_pairs_proved_only_with_the_rule(left, right):
    assert prove(left, right).proven
    with shape_fallback.without(NAME):
        assert not prove(left, right).proven


@pytest.mark.parametrize(
    "left, right",
    [
        (  # ELSE takes the rows whose secs is NULL
            "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'short' WHEN secs < 900 THEN 'medium' ELSE 'long' END = 'long'",
            "SELECT id FROM visits WHERE secs >= 900",
        ),
        (
            "SELECT id FROM visits WHERE CASE WHEN secs < 300 THEN 'short' WHEN secs < 900 THEN 'medium' ELSE 'long' END = 'medium'",
            "SELECT id FROM visits WHERE secs BETWEEN 300 AND 900",
        ),
        (  # the first arm wins
            "SELECT id FROM visits WHERE CASE WHEN secs < 900 THEN 'medium' WHEN secs < 300 THEN 'short' ELSE 'long' END = 'short'",
            "SELECT id FROM visits WHERE secs < 300",
        ),
    ],
)
def test_near_misses_stay_unproved(left, right):
    assert not prove(left, right).proven
