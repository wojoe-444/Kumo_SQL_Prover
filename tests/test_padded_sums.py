"""Under a GROUP BY, COALESCE(SUM(x), 0) is SUM(COALESCE(x, 0)) (``kumo_sql_prover.padded_sums``)."""

import pytest
import sqlglot

pytest.importorskip("z3")

from _made_up_posts import ROWS, differ_on_duckdb, proved, random_databases
from kumo_sql_prover.grouped_aggregate_spellings import respell
from kumo_sql_prover.padded_sums import pad_grouped_sums

INSIDE = "SELECT c.post, SUM(COALESCE(c.score, 0)) AS s FROM c GROUP BY c.post"
OUTSIDE = "SELECT c.post, COALESCE(SUM(c.score), 0) AS s FROM c GROUP BY c.post"


def padded(sql):
    out = pad_grouped_sums(sqlglot.parse_one(sql, read="postgres"))
    return out.sql(dialect="postgres") if out is not None else None


def test_the_coalesce_moves_into_a_grouped_sum():
    assert padded(OUTSIDE) == INSIDE
    # every such sum of the select, and only its own
    assert (
        padded(
            "SELECT c.post, COALESCE(SUM(c.score + c.id), 0) * 2 AS a, SUM(c.score) AS b, (SELECT COALESCE(SUM(u.rep), 0) FROM u) AS d"
            " FROM c GROUP BY c.post HAVING COALESCE(SUM(c.id), 0) > 1 ORDER BY COALESCE(SUM(c.score), 0)"
        )
        == "SELECT c.post, SUM(COALESCE(c.score + c.id, 0)) * 2 AS a, SUM(c.score) AS b, (SELECT COALESCE(SUM(u.rep), 0) FROM u) AS d"
        " FROM c GROUP BY c.post HAVING SUM(COALESCE(c.id, 0)) > 1 ORDER BY SUM(COALESCE(c.score, 0))"
    )
    assert padded(INSIDE) is None


def test_the_second_spelling_is_tried_on_every_select_of_both_queries():
    nested = f"SELECT p.id, t.s FROM p LEFT JOIN ({OUTSIDE}) AS t ON t.post = p.id WHERE p.day > COALESCE((SELECT SUM(u.rep) FROM u), 0)"
    other = "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) AS n, COALESCE(SUM(c.id), 0) AS s FROM c GROUP BY c.post"
    assert respell(nested, other, dialect="postgres") == (
        nested.replace(OUTSIDE, INSIDE),
        "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) AS n, SUM(COALESCE(c.id, 0)) AS s FROM c GROUP BY c.post",
    )
    assert respell(INSIDE, "SELECT COALESCE(SUM(c.score), 0) AS s FROM c", dialect="postgres") is None


@pytest.mark.parametrize("sql", [
    "SELECT COALESCE(SUM(c.score), 0) AS s FROM c",  # no GROUP BY: 0 over no rows, where the sum of zeros is NULL
    "SELECT COALESCE(SUM(c.score), 0) AS s FROM c HAVING COUNT(*) >= 0",
    "SELECT COALESCE(SUM(c.score), 0) AS s FROM c GROUP BY ()",
    "SELECT c.post, COALESCE(SUM(c.score), 0) AS s FROM c GROUP BY ROLLUP (c.post)",
    "SELECT c.post, COALESCE(SUM(c.score), 0) AS s FROM c GROUP BY GROUPING SETS ((c.post), ())",
    "SELECT c.post, COALESCE(SUM(c.score), 0) AS s FROM c GROUP BY CUBE (c.post)",
    "SELECT c.post, COALESCE(SUM(c.score), 1) AS s FROM c GROUP BY c.post",  # 1 for a group of NULLs
    "SELECT c.post, COALESCE(SUM(c.score), MAX(c.id), 0) AS s FROM c GROUP BY c.post",
    "SELECT c.post, COALESCE(SUM(c.score), '0') AS s FROM c GROUP BY c.post",
    "SELECT c.post, COALESCE(SUM(c.score), 0.0) AS s FROM c GROUP BY c.post",
    "SELECT c.post, COALESCE(SUM(DISTINCT c.score), 0) AS s FROM c GROUP BY c.post",
    "SELECT c.post, COALESCE(SUM(c.score) FILTER (WHERE c.id > 3), 0) AS s FROM c GROUP BY c.post",
    "SELECT c.id, COALESCE(SUM(c.score) OVER (ORDER BY c.id ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING), 0) AS s FROM c GROUP BY c.id, c.score",
    "SELECT c.post, COALESCE(AVG(c.score), 0) AS s FROM c GROUP BY c.post",  # the NULLs would count in the average
    "SELECT c.post, COALESCE(MAX(c.score), 0) AS s FROM c GROUP BY c.post",
    "SELECT c.post, COALESCE(SUM(c.score) + 1, 0) AS s FROM c GROUP BY c.post",
    "SELECT d.post FROM (SELECT c.post, COALESCE(SUM(c.score), 0) AS s FROM c GROUP BY c.post) AS d GROUP BY d.post",  # the derived table's own
])
def test_other_shapes_are_left_alone(sql):
    assert padded(sql) is None


TARGETS = [
    (INSIDE, OUTSIDE),
    # over a left join that pads the summed column
    (
        "SELECT u.id, SUM(COALESCE(d.n, 0)) AS s FROM u LEFT JOIN (SELECT p.owner, COUNT(*) AS n FROM p GROUP BY p.owner) AS d ON d.owner = u.id GROUP BY u.id",
        "SELECT u.id, COALESCE(SUM(d.n), 0) AS s FROM u LEFT JOIN (SELECT p.owner, COUNT(*) AS n FROM p GROUP BY p.owner) AS d ON d.owner = u.id GROUP BY u.id",
    ),
    (
        f"SELECT p.id, t.s FROM p LEFT JOIN ({INSIDE}) AS t ON t.post = p.id",
        f"SELECT p.id, t.s FROM p LEFT JOIN ({OUTSIDE}) AS t ON t.post = p.id",
    ),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_the_two_spellings_prove(left, right):
    assert proved(left, right)
    for rows in random_databases(len(left)):
        assert not differ_on_duckdb(left, right, rows)


NEAR_MISSES = [
    # over no rows the first is NULL and the second 0
    ("SELECT SUM(COALESCE(c.score, 0)) AS s FROM c WHERE c.id > 100", "SELECT COALESCE(SUM(c.score), 0) AS s FROM c WHERE c.id > 100"),
    (
        "SELECT c.post, SUM(COALESCE(c.score, 0)) AS s FROM c WHERE c.id > 100 GROUP BY ROLLUP (c.post)",
        "SELECT c.post, COALESCE(SUM(c.score), 0) AS s FROM c WHERE c.id > 100 GROUP BY ROLLUP (c.post)",
    ),
    (INSIDE, "SELECT c.post, SUM(c.score) AS s FROM c GROUP BY c.post"),  # NULL for a post whose scores are all NULL
    (INSIDE.replace("c.score, 0", "c.score, 1"), OUTSIDE),
    (INSIDE, OUTSIDE.replace("SUM(c.score), 0", "SUM(c.score), 1")),
    (INSIDE.replace("SUM", "AVG"), OUTSIDE.replace("SUM", "AVG")),  # a NULL counts as a 0 in the first average only
    (
        "SELECT c.post, SUM(COALESCE(c.score, 0)) FILTER (WHERE c.id > 100) AS s FROM c GROUP BY c.post",
        "SELECT c.post, COALESCE(SUM(c.score) FILTER (WHERE c.id > 100), 0) AS s FROM c GROUP BY c.post",
    ),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, {**ROWS, "c": ROWS["c"] + [(7, 4, None)]})  # with a post that has a score and a NULL


@pytest.mark.parametrize("sql", [
    OUTSIDE,
    "SELECT p.owner, p.kind, COALESCE(SUM(c.score), 0) AS a, COALESCE(SUM(p.day - c.id), 0) AS b FROM p LEFT JOIN c ON c.post = p.id GROUP BY p.owner, p.kind",
    "SELECT u.rep FROM u LEFT JOIN p ON p.owner = u.id GROUP BY u.rep HAVING COALESCE(SUM(p.kind), 0) > 1 ORDER BY COALESCE(SUM(p.day), 0)",
])
def test_the_moved_coalesce_gives_the_same_rows(sql):
    out = padded(sql)
    assert out is not None and "COALESCE(SUM(" not in out
    for rows in random_databases(len(sql), count=20):
        assert not differ_on_duckdb(sql, out, rows)


def test_a_sum_that_may_belong_to_an_enclosing_query_is_left_alone():
    def inner(sql):
        tree = sqlglot.parse_one(sql, read="postgres")
        select = max((s for s in tree.find_all(sqlglot.exp.Select) if s.args.get("group")), key=lambda s: s.depth)
        out = pad_grouped_sums(select)
        return out.sql(dialect="postgres") if out is not None else None

    # every column of the sum is the enclosing query's: over no rows of that query the first spelling is 0, the second NULL
    assert inner("SELECT (SELECT COALESCE(SUM(p.score), 0) FROM comments AS c GROUP BY c.postid) FROM posts AS p") is None
    assert inner("SELECT (SELECT COALESCE(SUM(score), 0) FROM comments AS c GROUP BY c.postid) FROM posts AS p") is None  # a bare name could be either
    # a column of the select's own source: this select's aggregate
    assert (
        inner("SELECT (SELECT COALESCE(SUM(c.score + p.score), 0) FROM comments AS c GROUP BY c.postid) FROM posts AS p")
        == "SELECT SUM(COALESCE(c.score + p.score, 0)) FROM comments AS c GROUP BY c.postid"
    )
    # a derived table sees no enclosing row
    assert inner("SELECT d.s FROM (SELECT postid, COALESCE(SUM(score), 0) AS s FROM comments GROUP BY postid) AS d") is not None
