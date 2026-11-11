"""Under a GROUP BY, SUM(CASE WHEN c THEN 1 ELSE 0 END) is COUNT(CASE WHEN c THEN 1 END) (``kumo_sql_prover.counted_cases``)."""

import pytest
import sqlglot

pytest.importorskip("z3")

from _made_up_posts import differ_on_duckdb, proved, random_databases
from kumo_sql_prover.counted_cases import count_summed_cases

SUMMED = "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) AS n FROM c GROUP BY c.post"
COUNTED = "SELECT c.post, COUNT(CASE WHEN c.score = 5 THEN 1 END) AS n FROM c GROUP BY c.post"


def counted(sql):
    out = count_summed_cases(sqlglot.parse_one(sql, read="postgres"))
    return out.sql(dialect="postgres") if out is not None else None


def test_a_grouped_sum_of_ones_is_a_count():
    assert counted(SUMMED) == COUNTED
    # every such sum of the select, and only its own
    assert (
        counted(
            "SELECT c.post, SUM(CASE WHEN c.score > 1 THEN 1 ELSE 0 END) + 1 AS a, SUM(c.score) AS b, (SELECT SUM(CASE WHEN u.rep = 1 THEN 1 ELSE 0 END) FROM u) AS d"
            " FROM c GROUP BY c.post HAVING SUM(CASE WHEN c.score IS NULL THEN 1 ELSE 0 END) = 0 ORDER BY SUM(CASE WHEN c.id IN (1, 2) THEN 1 ELSE 0 END)"
        )
        == "SELECT c.post, COUNT(CASE WHEN c.score > 1 THEN 1 END) + 1 AS a, SUM(c.score) AS b, (SELECT SUM(CASE WHEN u.rep = 1 THEN 1 ELSE 0 END) FROM u) AS d"
        " FROM c GROUP BY c.post HAVING COUNT(CASE WHEN c.score IS NULL THEN 1 END) = 0 ORDER BY COUNT(CASE WHEN c.id IN (1, 2) THEN 1 END)"
    )


@pytest.mark.parametrize("sql", [
    "SELECT SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) AS n FROM c",  # no GROUP BY: NULL over no rows
    "SELECT SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) AS n FROM c HAVING COUNT(*) >= 0",
    "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) AS n FROM c GROUP BY ROLLUP (c.post)",
    "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) AS n FROM c GROUP BY GROUPING SETS ((c.post), ())",
    "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) AS n FROM c GROUP BY CUBE (c.post)",
    "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) AS n FROM c GROUP BY c.post, ROLLUP (c.id)",
    "SELECT SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) AS n FROM c GROUP BY ()",
    "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 1 END) AS n FROM c GROUP BY c.post",  # no ELSE: NULL when never true
    "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 1 ELSE NULL END) AS n FROM c GROUP BY c.post",
    "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 2 ELSE 0 END) AS n FROM c GROUP BY c.post",
    "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 1 ELSE 1 END) AS n FROM c GROUP BY c.post",
    "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN '1' ELSE '0' END) AS n FROM c GROUP BY c.post",
    "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 1.0 ELSE 0 END) AS n FROM c GROUP BY c.post",
    "SELECT c.post, SUM(CASE c.score WHEN 5 THEN 1 ELSE 0 END) AS n FROM c GROUP BY c.post",
    "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 1 WHEN c.score = 3 THEN 1 ELSE 0 END) AS n FROM c GROUP BY c.post",
    "SELECT c.post, SUM(DISTINCT CASE WHEN c.score = 5 THEN 1 ELSE 0 END) AS n FROM c GROUP BY c.post",
    "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) FILTER (WHERE c.id > 3) AS n FROM c GROUP BY c.post",
    "SELECT c.id, SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) OVER (ORDER BY c.id ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS n FROM c GROUP BY c.id, c.score",
    "SELECT c.post, AVG(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) AS n FROM c GROUP BY c.post",
    "SELECT d.post FROM (SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) AS n FROM c GROUP BY c.post) AS d GROUP BY d.post",  # the derived table's own
])
def test_other_shapes_are_left_alone(sql):
    assert counted(sql) is None


TARGETS = [
    (SUMMED, COUNTED),
    (SUMMED, COUNTED.replace("THEN 1 END", "THEN 1 ELSE NULL END")),
    # inside a grouped derived table that a left join pads
    (
        f"SELECT p.id, COALESCE(d.n, 0) AS n FROM p LEFT JOIN ({SUMMED}) AS d ON d.post = p.id",
        f"SELECT p.id, COALESCE(d.n, 0) AS n FROM p LEFT JOIN ({COUNTED}) AS d ON d.post = p.id",
    ),
    (
        "SELECT u.id, SUM(CASE WHEN p.kind = 1 THEN 1 ELSE 0 END) AS a, SUM(CASE WHEN p.kind = 2 THEN 1 ELSE 0 END) AS b FROM u LEFT JOIN p ON p.owner = u.id GROUP BY u.id",
        "SELECT u.id, COUNT(CASE WHEN p.kind = 1 THEN 1 END) AS a, COUNT(CASE WHEN p.kind = 2 THEN 1 END) AS b FROM u LEFT JOIN p ON p.owner = u.id GROUP BY u.id",
    ),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_the_two_spellings_prove(left, right):
    assert proved(left, right)
    for rows in random_databases(len(left)):
        assert not differ_on_duckdb(left, right, rows)


NEAR_MISSES = [
    # over no rows the sum is NULL and the count 0
    ("SELECT SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) AS n FROM c WHERE c.id > 100", "SELECT COUNT(CASE WHEN c.score = 5 THEN 1 END) AS n FROM c WHERE c.id > 100"),
    # the grand total of a ROLLUP over no rows is a row
    (
        "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) AS n FROM c WHERE c.id > 100 GROUP BY ROLLUP (c.post)",
        "SELECT c.post, COUNT(CASE WHEN c.score = 5 THEN 1 END) AS n FROM c WHERE c.id > 100 GROUP BY ROLLUP (c.post)",
    ),
    (SUMMED.replace(" ELSE 0", ""), COUNTED),  # NULL for a post without such a comment
    (SUMMED.replace("THEN 1 ELSE 0", "THEN 2 ELSE 0"), COUNTED),
    (SUMMED, COUNTED.replace("c.score = 5", "c.score = 0")),
    (SUMMED, "SELECT c.post, COUNT(*) AS n FROM c GROUP BY c.post"),
    (
        "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) FILTER (WHERE c.id > 100) AS n FROM c GROUP BY c.post",
        "SELECT c.post, COUNT(CASE WHEN c.score = 5 THEN 1 END) FILTER (WHERE c.id > 100) AS n FROM c GROUP BY c.post",
    ),
    (
        "SELECT c.id, SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) OVER (ORDER BY c.id ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS n FROM c GROUP BY c.id, c.score",
        "SELECT c.id, COUNT(CASE WHEN c.score = 5 THEN 1 END) OVER (ORDER BY c.id ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS n FROM c GROUP BY c.id, c.score",
    ),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right)


@pytest.mark.parametrize("sql", [
    SUMMED,
    "SELECT p.owner, p.kind, SUM(CASE WHEN p.day > 1 OR p.id IS NULL THEN 1 ELSE 0 END) AS n, SUM(CASE WHEN c.score IS NULL THEN 1 ELSE 0 END) AS m FROM p LEFT JOIN c ON c.post = p.id GROUP BY p.owner, p.kind",
    "SELECT u.rep FROM u JOIN p ON p.owner = u.id GROUP BY u.rep HAVING SUM(CASE WHEN p.kind = 1 THEN 1 ELSE 0 END) > 1 ORDER BY SUM(CASE WHEN p.kind = 2 THEN 1 ELSE 0 END)",
])
def test_the_count_gives_the_same_rows(sql):
    out = counted(sql)
    assert out is not None and "SUM(CASE" not in out
    for rows in random_databases(len(sql), count=20):
        assert not differ_on_duckdb(sql, out, rows)
