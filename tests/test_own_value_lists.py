"""The values of a table that a list drawn from the same table keeps are the list itself (``own_value_lists``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.own_value_lists import read_own_value_list

SCHEMA = {"views": ["viewer", "day", "article", "author"], "visits": ["viewer", "day"]}
TESTED = "SELECT DISTINCT viewer FROM views WHERE viewer IN (SELECT viewer FROM views GROUP BY viewer, day HAVING COUNT(DISTINCT article) > 1)"
LISTED = "SELECT DISTINCT viewer FROM views WHERE NOT viewer IS NULL GROUP BY viewer, day HAVING COUNT(DISTINCT article) > 1"


def rewritten(sql: str, schema=SCHEMA) -> str | None:
    result = read_own_value_list(sqlglot.parse_one(sql, read="mysql"), schema)
    return None if result is None else result.sql(dialect="mysql")


def test_a_list_of_the_tested_column_is_the_answer():
    assert rewritten(TESTED) == LISTED
    # the output keeps the name the outer select gave it, and the outer table may have another name for itself
    assert rewritten(TESTED.replace("DISTINCT viewer FROM views WHERE viewer IN", "DISTINCT v.viewer AS id FROM views AS v WHERE v.viewer IN")) == LISTED.replace("DISTINCT viewer FROM", "DISTINCT viewer AS id FROM")
    # several columns, of which the outer select may list some
    two = "SELECT DISTINCT viewer FROM views WHERE (viewer, day) IN (SELECT viewer, day FROM views GROUP BY viewer, day HAVING COUNT(DISTINCT article) >= 2)"
    assert rewritten(two) == "SELECT DISTINCT viewer FROM views WHERE NOT viewer IS NULL AND NOT day IS NULL GROUP BY viewer, day HAVING COUNT(DISTINCT article) >= 2"
    # a filter instead of a grouping, and a filter under a grouping
    assert rewritten("SELECT DISTINCT viewer FROM views WHERE viewer IN (SELECT viewer FROM views WHERE author = viewer)") == "SELECT DISTINCT viewer FROM views WHERE author = viewer AND NOT viewer IS NULL"
    assert rewritten(TESTED.replace("GROUP BY viewer, day", "WHERE day > 3 GROUP BY viewer, day")) == LISTED.replace("WHERE NOT viewer IS NULL", "WHERE day > 3 AND NOT viewer IS NULL")
    assert rewritten(TESTED.replace("SELECT viewer FROM views GROUP BY viewer, day", "SELECT DISTINCT g.viewer FROM views AS g GROUP BY g.viewer, g.day")) == "SELECT DISTINCT g.viewer FROM views AS g WHERE NOT g.viewer IS NULL GROUP BY g.viewer, g.day HAVING COUNT(DISTINCT article) > 1"


def test_the_normalizers_spelling_of_the_grouped_subquery_is_read():
    derived = "SELECT DISTINCT v.viewer FROM views AS v WHERE v.viewer IN (SELECT g.k0 FROM (SELECT viewer AS k0, COUNT(DISTINCT article) AS a0 FROM views GROUP BY viewer, day) AS g WHERE g.a0 >= 2)"
    assert rewritten(derived) == "SELECT DISTINCT g.k0 AS viewer FROM (SELECT viewer AS k0, COUNT(DISTINCT article) AS a0 FROM views GROUP BY viewer, day) AS g WHERE g.a0 >= 2 AND NOT g.k0 IS NULL"
    assert rewritten(derived.replace("SELECT g.k0 FROM", "SELECT g.a0 FROM")) is None  # a count is no viewer
    assert rewritten(derived.replace("viewer AS k0", "author AS k0").replace("GROUP BY viewer", "GROUP BY author")) is None
    assert rewritten(derived.replace("g.a0 >= 2", "g.a0 >= v.day")) is None  # reads the outer row
    assert rewritten(derived.replace("FROM views GROUP BY", "FROM visits GROUP BY")) is None


@pytest.mark.parametrize(
    "sql",
    [
        TESTED.replace("SELECT DISTINCT viewer", "SELECT viewer"),  # one row per row of the table, not per value
        TESTED.replace("viewer IN", "viewer NOT IN"),
        TESTED.replace("WHERE viewer IN", "WHERE day > 3 AND viewer IN"),  # the list's rows need not pass the other test
        TESTED.replace("WHERE viewer IN", "WHERE author IN"),  # a list of viewers tested against authors
        TESTED.replace("SELECT DISTINCT viewer FROM", "SELECT DISTINCT viewer, day FROM"),  # a column the list does not hold
        TESTED.replace("SELECT DISTINCT viewer FROM", "SELECT DISTINCT UPPER(viewer) AS viewer FROM"),
        TESTED.replace("(SELECT viewer FROM views", "(SELECT viewer FROM visits"),  # another table's viewers need not be here
        TESTED.replace("(SELECT viewer FROM views", "(SELECT viewer FROM VIEWS"),
        TESTED.replace("(SELECT viewer FROM views", "(SELECT MAX(viewer) FROM views"),
        TESTED.replace("(SELECT viewer FROM views GROUP BY viewer, day", "(SELECT viewer FROM views GROUP BY day"),
        TESTED.replace("(SELECT viewer FROM views GROUP BY", "(SELECT viewer FROM views JOIN visits USING (viewer) GROUP BY"),
        TESTED.replace("HAVING COUNT(DISTINCT article) > 1", "HAVING COUNT(DISTINCT article) > (SELECT 1)"),
        TESTED.replace("HAVING COUNT(DISTINCT article) > 1", "HAVING COUNT(DISTINCT article) > 1 LIMIT 3"),
        TESTED.replace("FROM views WHERE viewer IN", "FROM views AS v WHERE v.viewer IN").replace("HAVING COUNT(DISTINCT article) > 1", "HAVING COUNT(DISTINCT article) > v.day"),
        TESTED.replace("FROM views WHERE viewer IN", "FROM views JOIN visits USING (viewer) WHERE viewer IN"),
        TESTED + " ORDER BY viewer LIMIT 2",
        TESTED.replace("DISTINCT viewer FROM views WHERE", "DISTINCT viewer FROM (SELECT * FROM views WHERE day > 3) AS views WHERE"),
    ],
)
def test_other_selects_are_left_alone(sql):
    assert rewritten(sql) is None


def test_the_column_lists_are_needed():
    assert rewritten(TESTED, schema=None) is None
    assert rewritten(TESTED, schema={"visits": ["viewer", "day"]}) is None


SPELLINGS = [
    TESTED,
    "SELECT DISTINCT v.viewer AS id FROM views AS v WHERE (v.viewer, v.day) IN (SELECT viewer, day FROM views GROUP BY viewer, day HAVING COUNT(DISTINCT article) >= 2)",
    "SELECT DISTINCT viewer, day FROM views WHERE (viewer, day) IN (SELECT viewer, day FROM views WHERE author IS NOT NULL GROUP BY viewer, day HAVING MIN(article) < MAX(article))",
    "SELECT DISTINCT viewer FROM views WHERE viewer IN (SELECT viewer FROM views WHERE author = viewer)",
    "SELECT DISTINCT day FROM views WHERE day IN (SELECT g.k0 FROM (SELECT day AS k0, COUNT(*) AS a0 FROM views GROUP BY day, author) AS g WHERE g.a0 = 1)",
]


@pytest.mark.parametrize("seed", range(4))
def test_the_list_returns_the_rows_duckdb_returns(seed):
    """No declared fact is needed: NULLs anywhere, repeated rows, empty tables."""

    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    nonempty = 0
    for sql in SPELLINGS:
        listed = rewritten(sql)
        assert listed is not None, sql
        for _ in range(12):
            con = duckdb.connect()
            con.execute("CREATE TABLE views (viewer INT, day INT, article INT, author INT)")
            for _row in range(rng.randint(0, 8)):
                con.execute("INSERT INTO views VALUES (?, ?, ?, ?)", [rng.choice([None, 1, 1, 2]), rng.choice([None, 5, 5, 6]), rng.choice([None, 7, 7, 8, 9]), rng.choice([None, 1, 3])])
            expected = sorted(con.execute(sql).fetchall(), key=repr)
            assert expected == sorted(con.execute(listed).fetchall(), key=repr), sql
            nonempty += bool(expected)
    assert nonempty > 5


def test_the_prover_reads_the_test_as_the_list():
    pytest.importorskip("z3")
    from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
    from kumo_sql_prover.smt_equivalence import TableConstraints

    grouped = "SELECT DISTINCT viewer AS id FROM views GROUP BY day, viewer HAVING COUNT(DISTINCT article) > 1"
    facts = {"views": TableConstraints(not_null=frozenset({"viewer", "day"}))}
    for sql in (TESTED, SPELLINGS[1]):
        assert prove_equivalent_algebraic(grouped, sql, schema=SCHEMA, constraints=facts, compare_names=False, dialect="mysql").proven, sql
    # the grouped spelling lists a NULL viewer's group; the test never passes a NULL
    assert not prove_equivalent_algebraic(grouped, TESTED, schema=SCHEMA, compare_names=False, dialect="mysql").proven
    assert not prove_equivalent_algebraic(grouped.replace("> 1", "> 2"), TESTED, schema=SCHEMA, constraints=facts, compare_names=False, dialect="mysql").proven
