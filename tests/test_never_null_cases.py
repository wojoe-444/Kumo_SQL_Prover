"""COALESCE around a CASE that is never NULL is the CASE (``kumo_sql_prover.never_null_cases``)."""

import pytest
import sqlglot

pytest.importorskip("z3")

from _made_up_posts import differ_on_duckdb, proved, random_databases
from kumo_sql_prover.never_null_cases import drop_never_null_padding

PADDED = "SELECT c.post, SUM(COALESCE(CASE WHEN c.score = 5 THEN 1 ELSE 0 END, 0)) AS n FROM c GROUP BY c.post"
PLAIN = "SELECT c.post, SUM(CASE WHEN c.score = 5 THEN 1 ELSE 0 END) AS n FROM c GROUP BY c.post"


def dropped(sql):
    out = drop_never_null_padding(sqlglot.parse_one(sql, read="postgres"))
    return out.sql(dialect="postgres") if out is not None else None


def test_the_padding_of_a_never_null_case_is_dropped():
    assert dropped(PADDED) == PLAIN
    # several branches, an operand, strings, nested paddings; only the select's own
    assert (
        dropped(
            "SELECT COALESCE(CASE c.score WHEN 1 THEN 'a' WHEN 2 THEN 'b' ELSE 'c' END, 'd', 'e') AS a, COALESCE(COALESCE(CASE WHEN c.id > 1 THEN 2 ELSE 3 END, 4), 5) AS b,"
            " (SELECT MAX(COALESCE(CASE WHEN u.rep = 1 THEN 1 ELSE 0 END, 0)) FROM u) AS d FROM c WHERE COALESCE(CASE WHEN c.post IS NULL THEN 1 ELSE 0 END, 7) = 1"
        )
        == "SELECT CASE c.score WHEN 1 THEN 'a' WHEN 2 THEN 'b' ELSE 'c' END AS a, COALESCE(CASE WHEN c.id > 1 THEN 2 ELSE 3 END, 5) AS b,"
        " (SELECT MAX(COALESCE(CASE WHEN u.rep = 1 THEN 1 ELSE 0 END, 0)) FROM u) AS d FROM c WHERE CASE WHEN c.post IS NULL THEN 1 ELSE 0 END = 1"
    )


@pytest.mark.parametrize("value", [
    "COALESCE(CASE WHEN c.score = 5 THEN 1 END, 0)",  # no ELSE: NULL when the test is not true
    "COALESCE(CASE WHEN c.score = 5 THEN 1 ELSE NULL END, 0)",
    "COALESCE(CASE WHEN c.score = 5 THEN NULL ELSE 0 END, 0)",
    "COALESCE(CASE WHEN c.score = 5 THEN c.id ELSE 0 END, 0)",  # a column can be NULL
    "COALESCE(CASE WHEN c.score = 5 THEN 1 ELSE c.post END, 0)",
    "COALESCE(CASE WHEN c.score = 5 THEN 1 + 1 ELSE 0 END, 0)",  # not a literal
    "COALESCE(CASE WHEN c.score = 5 THEN 1 ELSE 0 END, 0.5)",  # the other argument decides the type
    "COALESCE(CASE WHEN c.score = 5 THEN 1 ELSE 0 END, c.id)",
    "COALESCE(CASE WHEN c.score = 5 THEN 1.5 ELSE 0 END, 0)",
    "COALESCE(CASE WHEN c.score = 5 THEN 1 ELSE 'x' END, 0)",
    "COALESCE(c.score, CASE WHEN c.score = 5 THEN 1 ELSE 0 END)",  # the CASE is not the first argument
    "NULLIF(CASE WHEN c.score = 5 THEN 1 ELSE 0 END, 0)",
])
def test_other_expressions_are_left_alone(value):
    assert dropped(f"SELECT {value} AS v FROM c") is None


TARGETS = [
    (PADDED, PLAIN),
    (PADDED, "SELECT c.post, COUNT(CASE WHEN c.score = 5 THEN 1 END) AS n FROM c GROUP BY c.post"),  # with conditional_count_sums
    ("SELECT c.id, COALESCE(CASE WHEN c.score > 1 THEN 'hi' ELSE 'lo' END, 'none') AS v FROM c", "SELECT c.id, CASE WHEN c.score > 1 THEN 'hi' ELSE 'lo' END AS v FROM c"),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_the_two_spellings_prove(left, right):
    assert proved(left, right)
    for rows in random_databases(len(left)):
        assert not differ_on_duckdb(left, right, rows)


NEAR_MISSES = [
    # without the ELSE the padding is what makes the value 0
    ("SELECT c.id, COALESCE(CASE WHEN c.score = 5 THEN 1 END, 0) AS v FROM c", "SELECT c.id, CASE WHEN c.score = 5 THEN 1 END AS v FROM c"),
    ("SELECT c.id, COALESCE(CASE WHEN c.score = 5 THEN 1 ELSE c.post END, 0) AS v FROM c", "SELECT c.id, CASE WHEN c.score = 5 THEN 1 ELSE c.post END AS v FROM c"),
    ("SELECT c.id, COALESCE(CASE WHEN c.score = 5 THEN c.post ELSE 0 END, 9) AS v FROM c WHERE c.id = 4 OR c.score = 5", "SELECT c.id, CASE WHEN c.score = 5 THEN c.post ELSE 0 END AS v FROM c WHERE c.id = 4 OR c.score = 5"),
    (PADDED, PLAIN.replace("c.score = 5", "c.score = 0")),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, {"c": [(1, 1, 3), (2, 1, 0), (3, 4, 5), (4, None, 1), (5, None, 5), (6, 2, None)]})


@pytest.mark.parametrize("sql", [
    PADDED,
    "SELECT c.id, COALESCE(CASE c.score WHEN 1 THEN 'a' WHEN 2 THEN 'b' ELSE 'c' END, 'd') AS v, COALESCE(CASE WHEN c.post IS NULL THEN 7 WHEN c.post > 2 THEN 8 ELSE 9 END, 1, 2) AS w FROM c",
])
def test_dropping_gives_the_same_rows(sql):
    out = dropped(sql)
    assert out is not None and "COALESCE" not in out
    for rows in random_databases(len(sql), count=20):
        assert not differ_on_duckdb(sql, out, rows)
