"""A grouped table restricted to the keys of the relation it joins on that key needs no restriction (``kumo_sql_prover.joined_key_restrictions``)."""

import pytest
import sqlglot

pytest.importorskip("z3")

from _made_up_posts import SCHEMA, TYPES, differ_on_duckdb, proved, random_databases
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.joined_key_restrictions import drop_joined_key_restrictions
from kumo_sql_prover.smt_equivalence import WINDOW_SOURCE_ASSUMPTION

TOP = "SELECT p.id, p.day FROM p WHERE p.kind = 1"
RESTRICTED = f"SELECT t.id, t.day, d.n FROM ({TOP}) AS t LEFT JOIN (SELECT c.post, COUNT(*) AS n FROM c WHERE c.post IN (SELECT id FROM ({TOP}) AS x) GROUP BY c.post) AS d ON t.id = d.post"
PLAIN = f"SELECT t.id, t.day, d.n FROM ({TOP}) AS t LEFT JOIN (SELECT c.post, COUNT(*) AS n FROM c GROUP BY c.post) AS d ON t.id = d.post"
WITH = (
    "WITH top AS (SELECT p.id, p.day FROM p WHERE p.kind = 1), counts AS (SELECT c.post, COUNT(*) AS n, MAX(c.score) AS m FROM c{} GROUP BY c.post)"
    " SELECT top.id, top.day, counts.n, counts.m FROM top LEFT JOIN counts ON top.id = counts.post"
)


def dropped(sql, assumptions=None):
    return drop_joined_key_restrictions(sqlglot.parse_one(sql, read="postgres"), assumptions).sql(dialect="postgres")


def same(sql):
    return sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres")


def test_the_restriction_is_dropped():
    assert dropped(RESTRICTED) == PLAIN
    # other conjuncts of the WHERE and of the ON stay; an inner join; the key under another output name
    assert dropped(
        f"SELECT t.id, d.n FROM ({TOP}) AS t JOIN (SELECT c.post AS k, c.score, COUNT(*) AS n FROM c WHERE c.id > 1 AND c.post IN (SELECT DISTINCT x.id FROM ({TOP}) AS x) AND c.score > 0"
        " GROUP BY c.post, c.score HAVING COUNT(*) > 0) AS d ON d.k = t.id AND d.score < t.day"
    ) == (
        f"SELECT t.id, d.n FROM ({TOP}) AS t JOIN (SELECT c.post AS k, c.score, COUNT(*) AS n FROM c WHERE c.id > 1 AND c.score > 0"
        " GROUP BY c.post, c.score HAVING COUNT(*) > 0) AS d ON d.k = t.id AND d.score < t.day"
    )


def test_a_window_in_the_shared_relation_is_an_assumption():
    ranked = "SELECT p.id, ROW_NUMBER() OVER (ORDER BY p.day) AS r FROM p"
    sql = f"SELECT t.id, d.n FROM ({ranked}) AS t LEFT JOIN (SELECT c.post, COUNT(*) AS n FROM c WHERE c.post IN (SELECT id FROM ({ranked}) AS x) GROUP BY c.post) AS d ON t.id = d.post"
    assumptions = set()
    assert " IN " not in dropped(sql, assumptions) and assumptions == {WINDOW_SOURCE_ASSUMPTION}
    assumptions = set()
    assert dropped(RESTRICTED, assumptions) == PLAIN and not assumptions
    result = prove_equivalent_algebraic(sql, sql.replace(f" WHERE c.post IN (SELECT id FROM ({ranked}) AS x)", ""), schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False)
    assert result.proven and WINDOW_SOURCE_ASSUMPTION in result.assumptions


OTHER = "SELECT p.id, p.day FROM p WHERE p.kind = 2"


@pytest.mark.parametrize("sql", [
    RESTRICTED.replace(f"IN (SELECT id FROM ({TOP}) AS x)", f"IN (SELECT id FROM ({OTHER}) AS x)"),  # the keys of another relation
    RESTRICTED.replace("IN (SELECT id FROM", "IN (SELECT day FROM"),  # another column of it
    RESTRICTED.replace("ON t.id = d.post", "ON t.day = d.post"),  # joined on another column
    RESTRICTED.replace("ON t.id = d.post", "ON t.id = d.post OR d.n > 1"),
    RESTRICTED.replace("ON t.id = d.post", "ON t.id < d.post"),
    RESTRICTED.replace("ON t.id = d.post", "ON t.id = d.n"),
    RESTRICTED.replace("LEFT JOIN", "RIGHT JOIN"),  # keeps the groups that join no row
    RESTRICTED.replace("LEFT JOIN", "FULL JOIN"),
    RESTRICTED.replace("c.post IN", "c.post NOT IN"),
    RESTRICTED.replace("c.post IN", "c.score IN"),  # not the grouping key
    RESTRICTED.replace("WHERE c.post IN (", "WHERE c.id = 1 OR c.post IN (").replace(") AS x)", ") AS x) "),
    RESTRICTED.replace(" AS x)", " AS x WHERE x.day > 1)"),  # some of its keys only
    RESTRICTED.replace("SELECT c.post, COUNT(*) AS n FROM c", "SELECT COUNT(*) AS post, COUNT(*) AS n FROM c"),
    RESTRICTED.replace("GROUP BY c.post", "GROUP BY ROLLUP (c.post)"),
    RESTRICTED.replace("COUNT(*) AS n FROM c", "COUNT(*) AS n, RANK() OVER (ORDER BY COUNT(*)) AS r FROM c"),  # the rank counts the other groups
    RESTRICTED.replace("GROUP BY c.post)", "GROUP BY c.post LIMIT 3)"),
    RESTRICTED.replace(TOP, TOP + " LIMIT 3"),
    RESTRICTED.replace(TOP, TOP + " AND RANDOM() < 0.5"),
    # the restriction of a table that is not grouped, or grouped by another column
    f"SELECT t.id, d.score FROM ({TOP}) AS t LEFT JOIN (SELECT c.post, c.score FROM c WHERE c.post IN (SELECT id FROM ({TOP}) AS x)) AS d ON t.id = d.post",
    # inside a subquery the shared text may read the enclosing row
    f"SELECT u.id FROM u WHERE EXISTS ({RESTRICTED})",
])
def test_other_shapes_are_left_alone(sql):
    assert dropped(sql) == same(sql)


TARGETS = [
    (RESTRICTED, PLAIN),
    (WITH.format(" WHERE c.post IN (SELECT id FROM top)"), WITH.format("")),
    (WITH.format(" WHERE c.score > 0 AND c.post IN (SELECT top.id FROM top)"), WITH.format(" WHERE c.score > 0")),
    (WITH.format(" WHERE c.post IN (SELECT id FROM top)").replace("LEFT JOIN", "JOIN"), WITH.format("").replace("LEFT JOIN", "JOIN")),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_the_two_spellings_prove(left, right):
    assert proved(left, right)
    for rows in random_databases(len(left)):
        assert not differ_on_duckdb(left, right, rows)


NEAR_MISSES = [
    (RESTRICTED.replace(f"IN (SELECT id FROM ({TOP}) AS x)", f"IN (SELECT id FROM ({OTHER}) AS x)"), PLAIN),
    (RESTRICTED.replace("IN (SELECT id FROM", "IN (SELECT day FROM"), PLAIN),
    (RESTRICTED.replace("LEFT JOIN", "FULL JOIN"), PLAIN.replace("LEFT JOIN", "FULL JOIN")),  # the groups of other posts are kept
    (RESTRICTED.replace("c.post IN", "c.score IN"), PLAIN),
    (RESTRICTED.replace("c.post IN", "c.post NOT IN"), PLAIN),
    (RESTRICTED.replace(" AS x)", " AS x WHERE x.day > 10)"), PLAIN),
    (
        RESTRICTED.replace("COUNT(*) AS n FROM c", "RANK() OVER (ORDER BY COUNT(*), c.post DESC) AS n FROM c"),
        PLAIN.replace("COUNT(*) AS n FROM c", "RANK() OVER (ORDER BY COUNT(*), c.post DESC) AS n FROM c"),
    ),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right)


@pytest.mark.parametrize("sql", [
    RESTRICTED,
    f"SELECT t.id, d.n, d.m FROM ({TOP}) AS t JOIN (SELECT c.post AS k, c.score, COUNT(*) AS n, SUM(c.id) AS m FROM c WHERE c.post IN (SELECT x.id FROM ({TOP}) AS x) GROUP BY c.post, c.score) AS d ON d.k = t.id AND d.score > 0",
    "SELECT u.id, t.o, d.n FROM u LEFT JOIN (SELECT p.owner AS o, p.kind FROM p) AS t ON t.o = u.id LEFT JOIN (SELECT p.owner, COUNT(*) AS n FROM p WHERE p.owner IN (SELECT o FROM (SELECT p.owner AS o, p.kind FROM p) AS x) GROUP BY p.owner) AS d ON d.owner = t.o",
])
def test_dropping_gives_the_same_rows(sql):
    out = dropped(sql)
    assert " IN (" not in out
    for rows in random_databases(len(sql), count=20):
        assert not differ_on_duckdb(sql, out, rows)
