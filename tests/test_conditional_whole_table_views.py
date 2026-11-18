"""Candidate conditions are read through a CTE or derived table that shows one base table whole.

``WITH shared AS (SELECT * FROM policy WHERE ..) SELECT .. FROM policy WHERE pid IN (SELECT pid FROM shared)`` looks
a row up by ``pid`` among rows of its own table, which says what the pair needs only when ``pid`` is a key. The
column read from ``shared`` is ``policy.pid``, so the key on it is a candidate like the key of any other ``IN``.
"""

import pytest

from kumo_sql_prover.conditional_equivalence import candidate_conditions

SCHEMA = {"policy": ["pid", "city", "value"], "agent": ["aid", "city"]}
PLAIN = "SELECT SUM(value) FROM policy WHERE value > 3"


def texts(sql: str, schema=SCHEMA) -> set[str]:
    return {c.text for c in candidate_conditions(sql, PLAIN, schema=schema, dialect="mysql")}


def test_a_column_read_through_a_whole_table_view_is_the_tables_column():
    shared = "WITH shared AS (SELECT * FROM policy WHERE city = 'a') SELECT SUM(value) FROM policy WHERE pid IN (SELECT pid FROM shared)"
    assert "(pid) is unique in policy" in texts(shared)
    assert "policy.pid is NOT NULL" in texts(shared)
    # both sides through a view, a view named with its table alias, a derived table, and a qualified read
    both = "WITH kept AS (SELECT * FROM policy WHERE value > 1), shared AS (SELECT p.* FROM policy AS p WHERE p.city = 'a') SELECT SUM(value) FROM kept WHERE pid IN (SELECT pid FROM shared)"
    assert "(pid) is unique in policy" in texts(both)
    derived = "SELECT SUM(value) FROM policy WHERE pid IN (SELECT s.pid FROM (SELECT * FROM policy WHERE city = 'a') AS s)"
    assert "(pid) is unique in policy" in texts(derived)
    joined = "WITH shared AS (SELECT * FROM policy WHERE city = 'a') SELECT SUM(p.value) FROM policy AS p JOIN shared AS s ON s.pid = p.pid"
    assert "(pid) is unique in policy" in texts(joined)


@pytest.mark.parametrize(
    "view",
    [
        "SELECT pid, city FROM policy",  # a list of columns may rename or leave out
        "SELECT DISTINCT * FROM policy",
        "SELECT * FROM policy, agent",
        "SELECT * FROM policy LIMIT 3",
        "SELECT * FROM policy GROUP BY pid, city, value",
        "SELECT *, 1 AS one FROM policy",
        "SELECT * FROM (SELECT * FROM policy) AS p",  # one view at a time
    ],
)
def test_other_views_are_not_read_through(view):
    sql = f"WITH shared AS ({view}) SELECT SUM(value) FROM policy WHERE value IN (SELECT pid FROM shared)"
    # the same candidates as when the read goes through one more select, which is never read through
    assert texts(sql) == texts(sql.replace("FROM shared)", "FROM (SELECT * FROM shared) AS z)"))


def test_names_that_are_not_certain_are_not_read_through():
    shared = "WITH shared AS (SELECT * FROM policy) SELECT SUM(value) FROM policy WHERE value IN (SELECT pid FROM shared)"
    assert any("pid" in text for text in texts(shared))
    assert not any("pid" in text for text in texts(shared, schema={"policy": ["city", "value"]}))  # not a listed column
    assert not any("pid" in text for text in texts(shared.replace("shared AS (", "shared(pid, c, v) AS (")))  # the view renames
    twice = "WITH shared AS (SELECT * FROM policy) SELECT SUM(value) FROM policy WHERE value IN (WITH shared AS (SELECT aid AS pid FROM agent) SELECT pid FROM shared)"
    assert not any("pid" in text for text in texts(twice))  # which ``shared``?
    # a view over another view's name is not a base table
    chained = "WITH base AS (SELECT * FROM policy), shared AS (SELECT * FROM base) SELECT SUM(value) FROM policy WHERE value IN (SELECT pid FROM shared)"
    assert not any("pid" in text for text in texts(chained))
