"""A correlated COUNT subquery in a select list is a left join to the counts per key (``kumo_sql_prover.counted_subqueries``)."""

import pytest
import sqlglot

pytest.importorskip("z3")

from _made_up_posts import differ_on_duckdb, proved, random_databases
from kumo_sql_prover.counted_subqueries import count_subquery_as_join

COUNTED = "SELECT p.id, (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id AND x.score = 5) AS n FROM p"
JOINED = "SELECT p.id, COALESCE(d.n, 0) AS n FROM p LEFT JOIN (SELECT c.post, COUNT(*) AS n FROM c WHERE c.score = 5 GROUP BY c.post) AS d ON d.post = p.id"


def joined(sql):
    out = count_subquery_as_join(sqlglot.parse_one(sql, read="postgres"))
    return out.sql(dialect="postgres") if out is not None else None


def test_the_subquery_becomes_a_left_join_to_the_counts():
    assert joined(COUNTED) == (
        "SELECT p.id, COALESCE(kqc0.kqv, 0) AS n FROM p"
        " LEFT JOIN (SELECT x.post AS kqk0, COUNT(*) AS kqv FROM c AS x WHERE x.score = 5 GROUP BY x.post) AS kqc0 ON p.id = kqc0.kqk0"
    )
    # one subquery per application, the first one; an alias already in the query is not reused
    twice = "SELECT kqc0.id, (SELECT COUNT(x.score) FROM c AS x WHERE kqc0.id = x.post AND x.id = kqc0.day) + 1 AS n, (SELECT COUNT(*) FROM u WHERE u.id = kqc0.owner) AS m FROM p AS kqc0 WHERE kqc0.kind = 1 ORDER BY kqc0.id LIMIT 3"
    once = joined(twice)
    assert once == (
        "SELECT kqc0.id, COALESCE(kqc1.kqv, 0) + 1 AS n, (SELECT COUNT(*) FROM u WHERE u.id = kqc0.owner) AS m FROM p AS kqc0"
        " LEFT JOIN (SELECT x.post AS kqk0, x.id AS kqk1, COUNT(x.score) AS kqv FROM c AS x GROUP BY x.post, x.id) AS kqc1 ON kqc0.id = kqc1.kqk0 AND kqc0.day = kqc1.kqk1"
        " WHERE kqc0.kind = 1 ORDER BY kqc0.id LIMIT 3"
    )
    assert joined(once).count("LEFT JOIN") == 2 and joined(joined(once)) is None


@pytest.mark.parametrize("sql", [
    "SELECT p.id, (SELECT COUNT(*) FROM c AS x WHERE x.score = 5) AS n FROM p",  # no key: one value for every row
    "SELECT p.id, (SELECT COUNT(*) FROM c AS x WHERE x.post > p.id) AS n FROM p",  # not an equality
    "SELECT p.id, (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id OR x.score = 5) AS n FROM p",
    "SELECT p.id, (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id + 1) AS n FROM p",  # the key is not a column
    "SELECT p.id, (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id AND x.score = p.kind + 1) AS n FROM p",
    "SELECT p.id, (SELECT COUNT(*) FROM c AS x WHERE post = p.id) AS n FROM p",  # a column without a qualifier
    "SELECT p.id, (SELECT COUNT(*) FROM c AS p WHERE p.post = p.id) AS n FROM p",  # the outer alias is hidden
    "SELECT p.id, (SELECT COUNT(DISTINCT x.score) FROM c AS x WHERE x.post = p.id) AS n FROM p",
    "SELECT p.id, (SELECT COUNT(*) FILTER (WHERE x.score > 1) FROM c AS x WHERE x.post = p.id) AS n FROM p",
    "SELECT p.id, (SELECT COUNT(p.day) FROM c AS x WHERE x.post = p.id) AS n FROM p",  # counts an outer column
    "SELECT p.id, (SELECT SUM(x.score) FROM c AS x WHERE x.post = p.id) AS n FROM p",  # another rule's
    "SELECT p.id, (SELECT COUNT(*) + 1 FROM c AS x WHERE x.post = p.id) AS n FROM p",
    "SELECT p.id, (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id GROUP BY x.score) AS n FROM p",
    "SELECT p.id, (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id HAVING COUNT(*) > 1) AS n FROM p",  # no row, so NULL
    "SELECT p.id, (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id LIMIT 1) AS n FROM p",
    "SELECT p.id, (SELECT COUNT(*) FROM c AS x JOIN u ON u.id = x.score WHERE x.post = p.id) AS n FROM p",
    "SELECT p.id, (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id AND x.score IN (SELECT u.id FROM u)) AS n FROM p",
    "SELECT p.id, (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id AND x.score > RANDOM()) AS n FROM p",
    "SELECT p.kind, MAX((SELECT COUNT(*) FROM c AS x WHERE x.post = p.id)) AS n FROM p GROUP BY p.kind",
    "SELECT DISTINCT p.kind, (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id) AS n FROM p",
    "SELECT *, (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id) AS n FROM p",  # the star would list the join's columns
    "SELECT p.*, (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id) AS n FROM p",
    "SELECT p.id, ROW_NUMBER() OVER (ORDER BY p.day) AS r, (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id) AS n FROM p",
    "SELECT p.id, (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id) AS n FROM p JOIN u USING (id)",
    "SELECT p.id FROM p WHERE (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id) > 1",  # not in the select list
    "SELECT u.id, (SELECT MAX((SELECT COUNT(*) FROM c AS x WHERE x.post = p.id)) FROM p WHERE p.owner = u.id) AS n FROM u",
])
def test_other_shapes_are_left_alone(sql):
    assert joined(sql) is None


TARGETS = [
    (COUNTED, JOINED),
    # the spelling a rewriter prints for the padded count
    (COUNTED, JOINED.replace("COALESCE(d.n, 0)", "CASE WHEN d.n IS NULL THEN 0 ELSE d.n END")),
    (COUNTED, JOINED.replace("COALESCE(d.n, 0)", "CASE WHEN d.n IS NOT NULL THEN CAST(d.n AS BIGINT) ELSE 0 END")),
    # two counts over the same table with different filters, under a filter of the outer select
    (
        "SELECT p.id, (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id AND x.score = 5) AS a, (SELECT COUNT(*) FROM c AS y WHERE y.post = p.id AND y.score = 0) AS b FROM p WHERE p.kind = 1",
        "SELECT p.id, COALESCE(d.n, 0) AS a, COALESCE(e.n, 0) AS b FROM p LEFT JOIN (SELECT c.post, COUNT(*) AS n FROM c WHERE c.score = 5 GROUP BY c.post) AS d ON d.post = p.id"
        " LEFT JOIN (SELECT c.post, COUNT(*) AS n FROM c WHERE c.score = 0 GROUP BY c.post) AS e ON e.post = p.id WHERE p.kind = 1",
    ),
    # the outer select already joins
    (
        "SELECT u.name, p.id, (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id) AS n FROM p JOIN u ON p.owner = u.id",
        "SELECT u.name, p.id, COALESCE(d.n, 0) AS n FROM p JOIN u ON p.owner = u.id LEFT JOIN (SELECT c.post, COUNT(*) AS n FROM c GROUP BY c.post) AS d ON d.post = p.id",
    ),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_the_two_spellings_prove(left, right):
    assert proved(left, right)
    for rows in random_databases(len(left)):
        assert not differ_on_duckdb(left, right, rows)


NEAR_MISSES = [
    (COUNTED, JOINED.replace("COALESCE(d.n, 0)", "d.n")),  # NULL, not 0, for a post without such comments
    (COUNTED, JOINED.replace("LEFT JOIN", "JOIN")),  # loses the posts without such comments
    (COUNTED, JOINED.replace("c.score = 5 GROUP", "c.score = 0 GROUP")),  # another filter
    (COUNTED, JOINED.replace("COUNT(*)", "COUNT(c.score)").replace("WHERE c.score = 5 ", "")),  # counts another set of rows
    (COUNTED, "SELECT p.id, COALESCE(d.n, 0) AS n FROM p LEFT JOIN (SELECT c.post, c.score, COUNT(*) AS n FROM c GROUP BY c.post, c.score) AS d ON d.post = p.id"),  # one row per score: repeats the post
    (COUNTED, "SELECT p.id, COUNT(*) AS n FROM p LEFT JOIN c AS x ON x.post = p.id AND x.score = 5 GROUP BY p.id"),  # counts the padded row as 1
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right)


@pytest.mark.parametrize("sql", [
    COUNTED,
    "SELECT p.id, p.owner, (SELECT COUNT(x.score) FROM c AS x WHERE x.post = p.id AND p.kind = x.id) * 2 AS n FROM p WHERE p.day > 1",
    "SELECT u.name, p.id, (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id) AS n, (SELECT COUNT(*) FROM p AS q WHERE q.owner = u.id AND q.kind = 1) AS m FROM p LEFT JOIN u ON p.owner = u.id",
    "SELECT CASE WHEN (SELECT COUNT(*) FROM c AS x WHERE x.post = p.id) > 1 THEN p.kind ELSE (SELECT COUNT(*) FROM u AS w WHERE w.id = p.owner) END AS v FROM p",
])
def test_the_join_gives_the_same_rows(sql):
    out = joined(sql)
    assert out is not None
    while (more := joined(out)) is not None:
        out = more
    assert "(SELECT COUNT" not in out
    for rows in random_databases(len(sql), count=20):
        assert not differ_on_duckdb(sql, out, rows)
