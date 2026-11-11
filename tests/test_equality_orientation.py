"""Column equalities are written one way round (``kumo_sql_prover.equality_orientation``)."""

import pytest
import sqlglot

pytest.importorskip("z3")

from _made_up_posts import SCHEMA, TYPES, differ_on_duckdb, proved, random_databases
from kumo_sql_prover.algebraic_equivalence import normalize
from kumo_sql_prover.equality_orientation import orient_column_equalities


def oriented(sql):
    return orient_column_equalities(sqlglot.parse_one(sql, read="postgres")).sql(dialect="postgres")


def test_the_column_that_sorts_first_is_written_first():
    assert oriented("SELECT p.id FROM p JOIN u ON p.owner = u.id") == "SELECT p.id FROM p JOIN u ON u.id = p.owner"
    assert oriented("SELECT p.id FROM p JOIN u ON u.id = p.owner") == "SELECT p.id FROM p JOIN u ON u.id = p.owner"
    # the same column name: the qualifier decides
    assert oriented("SELECT p.id FROM p JOIN u ON u.id = p.id WHERE p.kind = p.day") == "SELECT p.id FROM p JOIN u ON p.id = u.id WHERE p.day = p.kind"
    assert oriented("SELECT p.id FROM p WHERE EXISTS (SELECT 1 FROM c WHERE c.post = p.id AND post = id)") == "SELECT p.id FROM p WHERE EXISTS(SELECT 1 FROM c WHERE p.id = c.post AND id = post)"


@pytest.mark.parametrize("sql", [
    "SELECT p.id FROM p JOIN u ON p.owner < u.id",  # not symmetric
    "SELECT p.id FROM p JOIN u ON p.owner <> u.id",
    "SELECT p.id FROM p JOIN u ON p.owner IS NOT DISTINCT FROM u.id",
    "SELECT p.id FROM p JOIN u ON p.owner + 1 = u.id",  # an expression operand
    "SELECT p.id FROM p WHERE p.owner = 3",
    "SELECT p.id FROM p JOIN u ON (p.owner, p.kind) = (u.id, u.rep)",
])
def test_other_comparisons_are_left_alone(sql):
    assert oriented(sql) == sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres")


COUNTED = "SELECT u.id, COUNT(DISTINCT p.id) AS n FROM u LEFT JOIN p ON p.owner = u.id GROUP BY u.id"
TARGETS = [
    # a grouped derived table kept whole: the join condition turned around was another text
    (f"SELECT x.id, d.n FROM p AS x JOIN ({COUNTED}) AS d ON x.id = d.id", f"SELECT x.id, d.n FROM p AS x JOIN ({COUNTED.replace('p.owner = u.id', 'u.id = p.owner')}) AS d ON d.id = x.id"),
    (
        "SELECT s.id, s.r FROM (SELECT p.id, RANK() OVER (ORDER BY COUNT(c.id) DESC) AS r FROM p LEFT JOIN c ON c.post = p.id GROUP BY p.id) AS s",
        "SELECT s.id, s.r FROM (SELECT p.id, RANK() OVER (ORDER BY COUNT(c.id) DESC) AS r FROM p LEFT JOIN c ON p.id = c.post GROUP BY p.id) AS s",
    ),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_turned_equalities_prove(left, right):
    assert proved(left, right)


NEAR_MISSES = [
    # another pair of columns
    (f"SELECT x.id, d.n FROM p AS x JOIN ({COUNTED}) AS d ON x.id = d.id", f"SELECT x.id, d.n FROM p AS x JOIN ({COUNTED.replace('p.owner = u.id', 'u.id = p.kind')}) AS d ON d.id = x.id"),
    # an order comparison turned around without its operator
    ("SELECT p.id FROM p JOIN u ON p.owner < u.id", "SELECT p.id FROM p JOIN u ON u.id < p.owner"),
    # the two sides of an outer join exchanged with the condition
    ("SELECT p.id, u.id AS v FROM p LEFT JOIN u ON p.owner = u.id", "SELECT p.id, u.id AS v FROM u LEFT JOIN p ON u.id = p.owner"),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right)


ORIENTED_QUERIES = [
    "SELECT p.id, u.name FROM p JOIN u ON p.owner = u.id WHERE p.kind = p.day OR u.rep = p.kind",
    f"SELECT x.id, d.n FROM p AS x LEFT JOIN ({COUNTED}) AS d ON x.owner = d.id",
    "SELECT p.id FROM p WHERE EXISTS (SELECT 1 FROM c WHERE c.post = p.id AND c.score = p.kind)",
    "SELECT p.id, CASE WHEN p.owner = p.kind THEN 1 ELSE 0 END AS same, (p.kind = p.day) IS NULL AS unknown FROM p",
]


@pytest.mark.parametrize("sql", ORIENTED_QUERIES)
def test_oriented_queries_give_the_same_rows(sql):
    out = oriented(sql)
    normalized = normalize(sql, schema=SCHEMA, types=TYPES, dialect="postgres")
    assert out != sql
    for rows in random_databases(len(sql)):
        assert not differ_on_duckdb(sql, out, rows)
        assert not differ_on_duckdb(sql, normalized, rows)
