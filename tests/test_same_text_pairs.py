"""A pair whose two queries are one text is proved, also when the provers cannot read the query (``same_text_pairs``)."""

from __future__ import annotations

import pytest

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.same_text_pairs import SAME_TEXT_ASSUMPTION, same_text
from kumo_sql_prover.smt_equivalence import SmtStatus

SCHEMA = {"posts": ["id", "kind", "score", "views"]}
TYPES = {"posts": {"id": "INT", "kind": "INT", "score": "INT", "views": "INT"}}
# a correlated scalar subquery over a windowed WITH table: the provers do not read it
UNREAD = (
    "WITH ranked AS (SELECT id, views, ROW_NUMBER() OVER (PARTITION BY kind ORDER BY score DESC) AS rn FROM posts) "
    "SELECT f.id, (SELECT AVG(r.views) FROM ranked AS r WHERE r.rn = f.rn) AS mean FROM ranked AS f WHERE f.rn < 5"
)

SAME = [
    ("SELECT a FROM t WHERE b = 'x'", "SELECT a FROM t WHERE b = 'x';"),
    ("SELECT a FROM t WHERE b = 'x'", "SELECT  a\nFROM t\n  WHERE b='x' ;\n"),
    ("SELECT a /* kept */ FROM t", "SELECT a /* kept */\nFROM t"),
    (UNREAD, UNREAD + ";"),
]
NOT_SAME = [
    ("SELECT a FROM t", "SELECT A FROM t"),  # a name in another case
    ("SELECT a FROM t", "select a from t"),  # spelling is compared letter for letter
    ("SELECT a FROM t WHERE b = 'x'", "SELECT a FROM t WHERE b = 'X'"),
    ("SELECT a FROM t WHERE b = 1", "SELECT a FROM t WHERE b = 1.0"),
    ("SELECT a FROM t", "SELECT a FROM t /* note */"),
    ("SELECT a FROM t /*!50000 WHERE a > 1 */", "SELECT a FROM t /*!50000 WHERE a > 2 */"),  # MySQL runs these
    ("SELECT a FROM t; SELECT b FROM t", "SELECT a FROM t; SELECT b FROM t"),  # two statements
    ("SELECT a FROM t WHERE (", "SELECT a FROM t WHERE ("),  # not a query
    ("DELETE FROM t", "DELETE FROM t"),
    ("", ""),
]


@pytest.mark.parametrize("left, right", SAME)
def test_two_texts_with_the_same_tokens_are_one_text(left, right):
    for dialect in ("postgres", "mysql", "bigquery", "duckdb", None):
        assert same_text(left, right, dialect) and same_text(right, left, dialect)


@pytest.mark.parametrize("left, right", NOT_SAME)
def test_two_texts_that_differ_or_are_no_query_are_not(left, right):
    for dialect in ("postgres", "mysql", "bigquery"):
        assert not same_text(left, right, dialect)


def _prove(left: str, right: str, dialect: str = "postgres"):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect=dialect)


def test_a_query_the_provers_cannot_read_is_equal_to_itself():
    result = _prove(UNREAD, UNREAD + ";")
    assert result.proven and SAME_TEXT_ASSUMPTION in result.assumptions
    with shape_fallback.without("same_text_pairs"):
        assert not _prove(UNREAD, UNREAD + ";").proven
    # one changed letter and the pair is as unread as it was
    assert not _prove(UNREAD, UNREAD.replace("f.rn < 5", "f.rn < 6")).proven


def test_a_pair_the_provers_read_is_proved_as_before():
    result = _prove("SELECT id FROM posts WHERE score > 1", "SELECT id FROM posts WHERE score > 1;")
    assert result.proven and SAME_TEXT_ASSUMPTION not in result.assumptions


def test_a_text_that_can_give_another_value_the_next_time_is_not_proved():
    drawn = UNREAD.replace("f.rn < 5", "f.rn < 5 AND RANDOM() < 0.5")
    assert not _prove(drawn, drawn).proven
    clock = UNREAD.replace("f.rn < 5", "f.rn < 5 AND NOW() > '2020-01-01'")
    assert _prove(clock, clock).status is not SmtStatus.PROVEN_EQUIVALENT
