"""An unread output of a derived table with a window function is dropped (``kumo_sql_prover.windowed_output_pruning``)."""

import pytest
import sqlglot

pytest.importorskip("z3")

from _made_up_posts import SCHEMA, TYPES, differ_on_duckdb, proved, random_databases
from kumo_sql_prover.algebraic_equivalence import normalize
from kumo_sql_prover.windowed_output_pruning import prune_windowed_outputs

BODY = "SELECT p.id, p.kind, COUNT(c.id) AS n, RANK() OVER (ORDER BY p.day) AS r FROM p LEFT JOIN c ON p.id = c.post GROUP BY p.id, p.kind, p.day"


def pruned(sql):
    out = prune_windowed_outputs(sqlglot.parse_one(sql, read="postgres"))
    return out.sql(dialect="postgres") if out is not None else None


def test_unread_outputs_are_dropped():
    assert pruned(f"SELECT d.id FROM ({BODY}) AS d WHERE d.r <= 2") == f"SELECT d.id FROM ({BODY.replace('p.kind, COUNT(c.id) AS n, ', '')}) AS d WHERE d.r <= 2"
    # the window itself, when nothing reads it
    assert pruned(f"SELECT d.id, d.n FROM ({BODY}) AS d") == f"SELECT d.id, d.n FROM ({BODY.replace('p.kind, ', '', 1).replace(', RANK() OVER (ORDER BY p.day) AS r', '')}) AS d"
    # a plain projection above a windowed select: the window is deeper
    assert (
        pruned("SELECT e.id FROM (SELECT w.id, w.r, w.k FROM (SELECT p.id, p.kind AS k, RANK() OVER (ORDER BY p.day) AS r FROM p) AS w) AS e WHERE e.r = 1")
        == "SELECT e.id FROM (SELECT w.id, w.r FROM (SELECT p.id, p.kind AS k, RANK() OVER (ORDER BY p.day) AS r FROM p) AS w) AS e WHERE e.r = 1"
    )
    # a select that aggregates without GROUP BY keeps an aggregate: it must stay one row
    assert (
        pruned("SELECT d.r FROM (SELECT COUNT(*) AS n, MAX(p.day) AS m, RANK() OVER (ORDER BY 1) AS r FROM p) AS d")
        == "SELECT d.r FROM (SELECT COUNT(*) AS n, RANK() OVER (ORDER BY 1) AS r FROM p) AS d"
    )


@pytest.mark.parametrize("sql", [
    f"SELECT d.id, d.kind, d.n, d.r FROM ({BODY}) AS d",  # everything is read
    f"SELECT * FROM ({BODY}) AS d",
    f"SELECT d.id FROM ({BODY.replace('SELECT', 'SELECT DISTINCT')}) AS d",  # dropping a column would merge rows
    f"SELECT d.id FROM ({BODY} ORDER BY n LIMIT 3) AS d",
    f"SELECT d.id FROM ({BODY}) AS d JOIN u USING (id)",
    f"SELECT d.id FROM ({BODY}) AS d(id, kind, n, r)",  # read by position
    "SELECT d.id FROM (SELECT p.id, p.kind FROM p) AS d",  # no window: another rule's
    f"SELECT d.id, (SELECT MAX(u.rep) FROM u WHERE u.id = d.n AND u.rep = d.kind AND u.rep <> d.r) AS m FROM ({BODY}) AS d",  # read from a subquery
    f"SELECT d.id, kind, n, r FROM ({BODY}) AS d",  # read without the alias
])
def test_other_shapes_are_left_alone(sql):
    assert pruned(sql) is None


TARGETS = [
    # the original computes a count nothing reads; the rewrite leaves it and its join out of the windowed select
    (
        "SELECT d.id FROM (SELECT p.id, COUNT(c.id) AS n, RANK() OVER (PARTITION BY p.kind ORDER BY p.day DESC) AS r FROM p JOIN u ON p.owner = u.id LEFT JOIN c ON p.id = c.post GROUP BY p.id, p.kind, p.day) AS d WHERE d.r <= 2",
        "SELECT d.id FROM (SELECT p.id, RANK() OVER (PARTITION BY p.kind ORDER BY p.day DESC) AS r FROM p JOIN u ON p.owner = u.id LEFT JOIN c ON p.id = c.post GROUP BY p.id, p.kind, p.day) AS d WHERE d.r <= 2",
    ),
    (
        "SELECT e.id FROM (SELECT w.id, w.r, w.k FROM (SELECT p.id, p.kind AS k, RANK() OVER (ORDER BY p.day) AS r FROM p) AS w) AS e WHERE e.r = 1",
        "SELECT w.id FROM (SELECT p.id, RANK() OVER (ORDER BY p.day) AS r FROM p) AS w WHERE w.r = 1",
    ),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_pruned_spellings_prove(left, right):
    assert proved(left, right)


NEAR_MISSES = [
    # DISTINCT over the pruned column: fewer columns, fewer rows
    (
        "SELECT d.k FROM (SELECT DISTINCT p.kind AS k, p.owner AS o, RANK() OVER (ORDER BY p.kind) AS r FROM p) AS d",
        "SELECT d.k FROM (SELECT DISTINCT p.kind AS k, RANK() OVER (ORDER BY p.kind) AS r FROM p) AS d",
    ),
    # the global aggregate against the plain select it would become
    ("SELECT d.r FROM (SELECT COUNT(*) AS n, RANK() OVER (ORDER BY 1) AS r FROM p) AS d", "SELECT d.r FROM (SELECT RANK() OVER (ORDER BY 1) AS r FROM p) AS d"),
    # the column is read: by the filter, by the output
    (f"SELECT d.id FROM ({BODY}) AS d WHERE d.n > 0", f"SELECT d.id FROM ({BODY}) AS d"),
    (f"SELECT d.id, d.n FROM ({BODY}) AS d", f"SELECT d.id, d.kind FROM ({BODY}) AS d"),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right)


PRUNED_QUERIES = [
    f"SELECT d.id FROM ({BODY}) AS d WHERE d.r <= 2",
    f"SELECT d.id, d.n FROM ({BODY}) AS d",
    "SELECT e.id FROM (SELECT w.id, w.r, w.k FROM (SELECT p.id, p.kind AS k, RANK() OVER (ORDER BY p.day) AS r FROM p) AS w) AS e WHERE e.r = 1",
    "SELECT d.r FROM (SELECT COUNT(*) AS n, MAX(p.day) AS m, RANK() OVER (ORDER BY 1) AS r FROM p) AS d",
    "SELECT u.name, d.r FROM u LEFT JOIN (SELECT p.owner AS o, p.day AS x, SUM(p.day) OVER (PARTITION BY p.owner) AS r FROM p WHERE p.kind = 1) AS d ON d.o = u.id",
    "SELECT d.o FROM (SELECT p.owner AS o, COUNT(*) AS n, MAX(p.day) AS m, DENSE_RANK() OVER (ORDER BY COUNT(*)) AS r FROM p GROUP BY p.owner HAVING COUNT(*) > 0) AS d",
]


@pytest.mark.parametrize("sql", PRUNED_QUERIES)
def test_pruned_queries_give_the_same_rows(sql):
    out = pruned(sql)
    normalized = normalize(sql, schema=SCHEMA, types=TYPES, dialect="postgres")
    assert out is not None and len(out) < len(sql)
    for rows in random_databases(len(sql)):
        assert not differ_on_duckdb(sql, out, rows)
        assert not differ_on_duckdb(sql, normalized, rows)
