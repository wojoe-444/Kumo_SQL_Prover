"""Conditional counts of a grouped query read one way (``kumo_sql_prover.grouped_conditional_counts``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.grouped_conditional_counts import grouped_conditional_counts

SCHEMA = {"a": ["id", "x", "y", "w"], "b": ["id", "n"]}
TYPES = {"a": {"id": "BIGINT", "x": "INT", "y": "INT", "w": "VARCHAR(20)"}, "b": {"id": "BIGINT", "n": "INT"}}


def read(sql):
    tree = sqlglot.parse_one(sql, read="postgres")
    out = grouped_conditional_counts(tree)
    return None if out is None else out.sql(dialect="postgres")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="postgres").proven


def test_a_sum_of_ones_becomes_a_count():
    assert read("SELECT a.id, SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) FROM a GROUP BY a.id") == (
        "SELECT a.id, COUNT(CASE WHEN a.x > 3 THEN 1 END) FROM a GROUP BY a.id"
    )
    assert read("SELECT a.id FROM a GROUP BY a.id HAVING SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) > 2 ORDER BY SUM(CASE WHEN a.y = 0 THEN 1 ELSE 0 END)") == (
        "SELECT a.id FROM a GROUP BY a.id HAVING COUNT(CASE WHEN a.x > 3 THEN 1 END) > 2 ORDER BY COUNT(CASE WHEN a.y = 0 THEN 1 END)"
    )
    assert read("SELECT b.n, SUM(CASE WHEN a.x > b.n THEN 1 ELSE 0 END) FROM a JOIN b ON a.id = b.id GROUP BY b.n") == (
        "SELECT b.n, COUNT(CASE WHEN a.x > b.n THEN 1 END) FROM a JOIN b ON a.id = b.id GROUP BY b.n"
    )
    assert read("SELECT id, SUM(CASE WHEN d > 3 THEN 1 ELSE 0 END) FROM (SELECT a.id, a.x - a.y AS d FROM a) AS s GROUP BY id") == (
        "SELECT id, COUNT(CASE WHEN d > 3 THEN 1 END) FROM (SELECT a.id, a.x - a.y AS d FROM a) AS s GROUP BY id"
    )


def test_a_count_takes_the_literal_one():
    # the literal a count's CASE returns does not matter, grouped or not
    assert read("SELECT a.id, COUNT(CASE WHEN a.x > 3 THEN 'y' ELSE NULL END) AS c FROM a GROUP BY a.id") == (
        "SELECT a.id, COUNT(CASE WHEN a.x > 3 THEN 1 END) AS c FROM a GROUP BY a.id"
    )
    assert read("SELECT COUNT(CASE WHEN a.x > 3 THEN 7 END) FROM a") == "SELECT COUNT(CASE WHEN a.x > 3 THEN 1 END) FROM a"
    assert read("SELECT COUNT(CASE WHEN o.x > 3 THEN 7 END) FROM a") == "SELECT COUNT(CASE WHEN o.x > 3 THEN 1 END) FROM a"


def test_a_count_of_a_null_test_becomes_a_count_of_the_column():
    grouped = "SELECT a.id, {} FROM a GROUP BY a.id"
    assert read(grouped.format("SUM(CASE WHEN a.x IS NOT NULL THEN 1 ELSE 0 END)")) == grouped.format("COUNT(a.x)")
    assert read(grouped.format("SUM(CASE WHEN NOT a.x IS NULL THEN 1 ELSE 0 END)")) == grouped.format("COUNT(a.x)")
    assert read(grouped.format("SUM(CASE WHEN a.x IS NULL THEN 1 ELSE 0 END)")) == grouped.format("COUNT(*) - COUNT(a.x)")
    assert read(grouped.format("SUM(CASE WHEN NOT (a.x IS NOT NULL) THEN 1 ELSE 0 END) AS c")) == grouped.format("COUNT(*) - COUNT(a.x) AS c")
    assert read(grouped.format("2 * SUM(CASE WHEN (a.x IS NULL) THEN 1 ELSE 0 END)")) == grouped.format("2 * (COUNT(*) - COUNT(a.x))")
    assert read(grouped.format("COUNT(CASE WHEN a.x IS NOT NULL THEN 1 END)")) == grouped.format("COUNT(a.x)")
    assert read(grouped.format("COUNT(CASE WHEN a.x IS NULL THEN 'n' END)")) == grouped.format("COUNT(*) - COUNT(a.x)")
    # the counts are 0 over no rows in every spelling, so they need no group
    assert read("SELECT COUNT(CASE WHEN a.x IS NULL THEN 1 END) FROM a") == "SELECT COUNT(*) - COUNT(a.x) FROM a"
    assert read("SELECT COUNT(CASE WHEN a.x IS NOT NULL THEN 1 END) FROM a") == "SELECT COUNT(a.x) FROM a"
    for left_alone in (
        "SUM(CASE WHEN a.x IS NULL THEN 2 ELSE 0 END)",
        "SUM(CASE WHEN a.x IS NULL THEN 1 ELSE 1 END)",
        "SUM(CASE WHEN a.x IS NULL THEN 1 END)",  # NULL when no row has it, not 0
        "SUM(CASE WHEN a.x IS NULL THEN 1.0 ELSE 0 END)",
        "SUM(CASE WHEN a.x IS NULL THEN '1' ELSE 0 END)",
        "SUM(CASE WHEN a.x IS NULL THEN 1 WHEN a.y IS NULL THEN 1 ELSE 0 END)",
        "SUM(CASE WHEN a.x IS TRUE THEN 1 END)",
        "SUM(DISTINCT CASE WHEN a.x IS NULL THEN 1 ELSE 0 END)",
        "SUM(CASE WHEN a.x IS NULL THEN 1 ELSE 0 END) FILTER (WHERE a.y > 0)",
        "SUM(CASE WHEN o.x IS NULL THEN 1 ELSE 0 END)",  # the enclosing query's column
        "SUM(CASE WHEN x IS NULL THEN 1 ELSE 0 END)",  # whose x is not said
        "AVG(CASE WHEN a.x IS NULL THEN 1 ELSE 0 END)",
        "COUNT(CASE WHEN o.x IS NULL THEN 1 END)",  # COUNT(*) would be this select's, the test is the enclosing query's
        "COUNT(CASE WHEN x IS NULL THEN 1 END)",
    ):
        assert read(grouped.format(left_alone)) is None, left_alone
    # an expression's NULL test stays a conditional count
    assert read(grouped.format("SUM(CASE WHEN a.x + a.y IS NULL THEN 1 ELSE 0 END)")) == grouped.format("COUNT(CASE WHEN a.x + a.y IS NULL THEN 1 END)")
    for ungrouped in ("SELECT {} FROM a", "SELECT a.id, {} FROM a GROUP BY ROLLUP (a.id)", "SELECT {} FROM a GROUP BY ()"):
        assert read(ungrouped.format("SUM(CASE WHEN a.x IS NULL THEN 1 ELSE 0 END)")) is None, ungrouped
        assert read(ungrouped.format("SUM(CASE WHEN a.x IS NOT NULL THEN 1 ELSE 0 END)")) is None, ungrouped
    assert read("SELECT a.id, SUM(CASE WHEN a.x IS NULL THEN 1 ELSE 0 END) OVER (ORDER BY a.id ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) FROM a GROUP BY a.id, a.x") is None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) FROM a",  # no group: NULL against 0 over no rows
        "SELECT SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) FROM a HAVING COUNT(*) >= 0",
        "SELECT a.id, SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) FROM a GROUP BY ROLLUP (a.id)",  # a grand total over no rows
        "SELECT a.id, SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) FROM a GROUP BY CUBE (a.id)",
        "SELECT a.id, SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) FROM a GROUP BY GROUPING SETS ((a.id), ())",
        "SELECT SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) FROM a GROUP BY ()",
        "SELECT a.id, SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) FILTER (WHERE a.y > 0) FROM a GROUP BY a.id",  # an empty filter
        "SELECT a.id, SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) OVER (ORDER BY a.id ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) FROM a GROUP BY a.id, a.x",
        "SELECT a.id, SUM(DISTINCT CASE WHEN a.x > 3 THEN 1 ELSE 0 END) FROM a GROUP BY a.id",
        "SELECT a.id, SUM(CASE WHEN a.x > 3 THEN a.y ELSE 0 END) FROM a GROUP BY a.id",
        "SELECT a.id, SUM(CASE WHEN a.x > 3 THEN 1 END) FROM a GROUP BY a.id",  # NULL when no row passes
        "SELECT a.id, SUM(CASE WHEN a.x > 3 THEN 1 ELSE 2 END) FROM a GROUP BY a.id",
        "SELECT a.id, SUM(CASE a.x WHEN 3 THEN 1 ELSE 0 END) FROM a GROUP BY a.id",  # a simple CASE
        "SELECT a.id, SUM(CASE WHEN a.x > 3 THEN 1 WHEN a.y > 3 THEN 1 ELSE 0 END) FROM a GROUP BY a.id",
        "SELECT a.id, COUNT(CASE WHEN a.x > 3 THEN 1 END) FROM a GROUP BY a.id",  # already the spelling
        "SELECT a.id, COUNT(DISTINCT CASE WHEN a.x > 3 THEN 2 END) FROM a GROUP BY a.id",
        "SELECT a.id, COUNT(CASE WHEN a.x > 3 THEN a.y END) FROM a GROUP BY a.id",  # y may be NULL
        "SELECT a.id, COUNT(CASE WHEN a.x > 3 THEN NULL END) FROM a GROUP BY a.id",
        "SELECT a.id, COUNT(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) FROM a GROUP BY a.id",  # counts every row
        "SELECT a.id, COUNT(CASE WHEN a.x > 3 THEN 7 WHEN a.y IS NULL THEN 8 END) FROM a GROUP BY a.id",  # two arms
        "SELECT a.id, COUNT(CASE WHEN a.x > 3 THEN 2 END) FILTER (WHERE a.y > 0) FROM a GROUP BY a.id",
        "SELECT a.id, COUNT(a.x) FROM a GROUP BY a.id",
        "SELECT a.id, COUNT(*) FROM a GROUP BY a.id",
        "SELECT a.id, SUM(CASE WHEN x > 3 THEN 1 ELSE 0 END) FROM a GROUP BY a.id",  # whose x is not said
        # the sum reads only the enclosing query's row, so it is that query's aggregate, which has no group
        "SELECT b.id, SUM(CASE WHEN o.x > 3 THEN 1 ELSE 0 END) FROM b GROUP BY b.id",
        "SELECT b.id, SUM(CASE WHEN EXISTS (SELECT 1 FROM b WHERE b.n = o.x) THEN 1 ELSE 0 END) FROM b GROUP BY b.id",  # the inner b
        "SELECT s.id, SUM(CASE WHEN x > 3 THEN 1 ELSE 0 END) FROM (SELECT b.id FROM b) AS s GROUP BY s.id",  # x is not the derived table's
        "SELECT s.id, SUM(CASE WHEN x > 3 THEN 1 ELSE 0 END) FROM (SELECT * FROM b) AS s GROUP BY s.id",
        "SELECT a.id, (SELECT SUM(CASE WHEN b.n > 3 THEN 1 ELSE 0 END) FROM b) FROM a GROUP BY a.id",  # the subquery's own
    ],
)
def test_what_is_left_alone(sql):
    assert read(sql) is None, sql


def test_pairs_it_proves():
    assert proved(
        "SELECT a.id, SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) FROM a GROUP BY a.id",
        "SELECT a.id, COUNT(CASE WHEN a.x > 3 THEN 1 END) FROM a GROUP BY a.id",
    )
    assert proved(
        "SELECT SUBSTRING(a.w, 1, 3), SUM(CASE WHEN a.x - a.y > 30 AND a.x - a.y <= 60 THEN 1 ELSE 0 END) "
        "FROM a, b WHERE a.id = b.id GROUP BY SUBSTRING(a.w, 1, 3)",
        "SELECT SUBSTRING(a.w, 1, 3), COUNT(CASE WHEN a.x - a.y BETWEEN 31 AND 60 THEN 1 END) "
        "FROM a JOIN b ON a.id = b.id GROUP BY SUBSTRING(a.w, 1, 3)",
    )
    nulls = "SELECT a.id, SUM(CASE WHEN b.n IS NULL THEN 1 ELSE 0 END), SUM(CASE WHEN b.n IS NOT NULL THEN 1 ELSE 0 END) FROM a LEFT JOIN b ON a.x = b.id GROUP BY a.id"
    assert proved(nulls, "SELECT a.id, COUNT(*) - COUNT(b.n), COUNT(b.n) FROM a LEFT JOIN b ON a.x = b.id GROUP BY a.id")
    assert proved(
        "SELECT a.id, SUM(CASE WHEN a.x IS NULL THEN 1 ELSE 0 END) FROM a GROUP BY a.id",
        "SELECT a.id, COUNT(*) - COUNT(a.x) FROM a GROUP BY a.id",
    )
    assert proved(
        "SELECT a.id FROM a GROUP BY a.id HAVING SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) >= 3",
        "SELECT a.id FROM a GROUP BY a.id HAVING COUNT(CASE WHEN a.x > 3 THEN 'hit' END) >= 3",
    )


def test_near_misses_stay_unproven():
    grouped = "SELECT a.id, {} FROM a GROUP BY a.id"
    total = "SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END)"
    # no group: over an empty table the sum is NULL and the count is 0
    assert not proved("SELECT SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) FROM a", "SELECT COUNT(CASE WHEN a.x > 3 THEN 1 END) FROM a")
    assert not proved(
        "SELECT a.id, SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) FROM a GROUP BY ROLLUP (a.id)",
        "SELECT a.id, COUNT(CASE WHEN a.x > 3 THEN 1 END) FROM a GROUP BY ROLLUP (a.id)",
    )
    assert not proved(grouped.format(total), grouped.format("COUNT(CASE WHEN a.x > 3 THEN a.y END)"))  # y may be NULL
    assert not proved(grouped.format(total), grouped.format("COUNT(CASE WHEN a.x > 3 THEN 1 ELSE 0 END)"))
    assert not proved(grouped.format(total), grouped.format("COUNT(CASE WHEN a.x >= 3 THEN 1 END)"))
    assert not proved(grouped.format(total), grouped.format("COUNT(CASE WHEN a.x > 3 THEN 1 END) FILTER (WHERE a.y > 0)"))
    assert not proved(grouped.format("SUM(CASE WHEN a.x > 3 THEN 2 ELSE 0 END)"), grouped.format("COUNT(CASE WHEN a.x > 3 THEN 2 END)"))
    # the null counts: no group, the other column, the other test
    assert not proved("SELECT SUM(CASE WHEN a.x IS NULL THEN 1 ELSE 0 END) FROM a", "SELECT COUNT(*) - COUNT(a.x) FROM a")
    missing = grouped.format("SUM(CASE WHEN a.x IS NULL THEN 1 ELSE 0 END)")
    assert not proved(missing, grouped.format("COUNT(a.x)"))
    assert not proved(missing, grouped.format("COUNT(*) - COUNT(a.y)"))
    assert not proved(grouped.format("SUM(CASE WHEN a.x IS NOT NULL THEN 1 ELSE 0 END)"), grouped.format("COUNT(DISTINCT a.x)"))
    assert not proved(grouped.format("SUM(CASE WHEN a.x IS NOT NULL THEN 1 ELSE 0 END)"), grouped.format("COUNT(*)"))


def test_the_engine_agrees_on_random_tables():
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(20261118)
    db = duckdb.connect()
    db.execute("CREATE TABLE a (id BIGINT, x INT, y INT, w VARCHAR)")
    db.execute("CREATE TABLE b (id BIGINT, n INT)")
    value = lambda: rng.choice([None, *range(-4, 9)])  # noqa: E731
    db.executemany("INSERT INTO a VALUES (?, ?, ?, NULL)", [(rng.choice([None, 0, 1, 2, 3, 4]), value(), value()) for _ in range(300)])
    db.executemany("INSERT INTO a VALUES (?, NULL, NULL, NULL)", [(n,) for n in (50, 50, 51)])  # groups where nothing is counted
    db.executemany("INSERT INTO b VALUES (?, ?)", [(rng.randint(0, 5), value()) for _ in range(20)])
    queries = (
        "SELECT a.id, SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) FROM a GROUP BY a.id",
        "SELECT a.id, COUNT(CASE WHEN a.x > 3 THEN 7 END), COUNT(*) FROM a GROUP BY a.id",
        "SELECT a.id FROM a GROUP BY a.id HAVING SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) > 2 OR SUM(CASE WHEN a.y > 100 THEN 1 ELSE 0 END) = 0",
        "SELECT b.n, COUNT(CASE WHEN a.x > b.n THEN 'y' ELSE NULL END) FROM a JOIN b ON a.id = b.id GROUP BY b.n",
        "SELECT a.id, SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) FROM a WHERE a.y > 1000 GROUP BY a.id",  # no rows, no groups
        "SELECT id, SUM(CASE WHEN d > 3 THEN 1 ELSE 0 END) FROM (SELECT a.id, a.x - a.y AS d FROM a) AS s GROUP BY id",
        "SELECT a.id, SUM(CASE WHEN a.x IS NULL THEN 1 ELSE 0 END), SUM(CASE WHEN NOT a.y IS NULL THEN 1 ELSE 0 END) FROM a GROUP BY a.id",
        "SELECT a.id, 2 * SUM(CASE WHEN a.x IS NULL THEN 1 ELSE 0 END) - COUNT(CASE WHEN a.y IS NOT NULL THEN 1 END) FROM a GROUP BY a.id",
        "SELECT a.id, SUM(CASE WHEN b.n IS NULL THEN 1 ELSE 0 END), SUM(CASE WHEN b.n IS NOT NULL THEN 1 ELSE 0 END) FROM a LEFT JOIN b ON a.x = b.id GROUP BY a.id",
        "SELECT a.id FROM a GROUP BY a.id HAVING SUM(CASE WHEN a.x IS NULL THEN 1 ELSE 0 END) > 3",
        "SELECT COUNT(CASE WHEN a.x IS NULL THEN 1 END), COUNT(CASE WHEN a.y IS NOT NULL THEN 'n' END), COUNT(CASE WHEN a.x > 3 THEN 9 END) FROM a",
    )
    key = lambda row: tuple((v is None, v) for v in row)  # noqa: E731
    seen = 0
    for sql in queries:
        after = read(sql)
        assert after is not None, sql
        rows = sorted(db.execute(sql).fetchall(), key=key)
        assert rows == sorted(db.execute(after).fetchall(), key=key), sql
        seen += len(rows)
    assert seen > 20
    db.execute("DELETE FROM a")
    for sql in queries:  # no rows: no groups, and the ungrouped counts are all 0
        assert db.execute(sql).fetchall() == db.execute(read(sql)).fetchall(), sql


def test_off_on_the_second_attempt():
    sql = "SELECT a.id, SUM(CASE WHEN a.x > 3 THEN 1 ELSE 0 END) FROM a GROUP BY a.id"
    assert "SUM(" not in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)
    with shape_fallback.disabled():
        assert "SUM(CASE" in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)
