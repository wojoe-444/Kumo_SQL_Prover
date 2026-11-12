"""Window computations over inputs the prover shows equal are read over one input (``kumo_sql_prover.window_inputs``)."""

import random

import pytest
import sqlglot
from sqlglot import exp

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.window_inputs import split, unify

SCHEMA = {"p": ["id", "owner", "kind", "day"], "u": ["id", "name", "rep"], "c": ["id", "post", "score"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


def rows_on_duckdb(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name, columns in SCHEMA.items():
        db.execute(f"CREATE TABLE {name} ({', '.join(f'{column} BIGINT' for column in columns)})")
        insert_rows(db, name, rows.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def differ_on_duckdb(left, right, rows):
    a, b = rows_on_duckdb([left, right], rows)
    return a != b


def step_one(sql):
    """The select with its windows computed over a derived table of their input, or None."""

    part = split(sqlglot.parse_one(sql, read="postgres"))
    return part.outer.sql(dialect="postgres") if part is not None else None


def test_the_input_of_a_window_is_isolated():
    assert (
        step_one("SELECT p.id AS a, RANK() OVER (PARTITION BY p.kind ORDER BY p.day DESC) AS r FROM p JOIN u ON p.owner = u.id WHERE u.rep > 1")
        == "SELECT kqwi.kqi0 AS a, RANK() OVER (PARTITION BY kqwi.kqi1 ORDER BY kqwi.kqi2 DESC) AS r"
        " FROM (SELECT p.id AS kqi0, p.kind AS kqi1, p.day AS kqi2 FROM p JOIN u ON p.owner = u.id WHERE u.rep > 1) AS kqwi"
    )
    # a grouped select: the aggregates the windows read are computed by the input, one row per group
    assert (
        step_one("SELECT p.kind AS k, COUNT(*) AS n, SUM(COUNT(*)) OVER (ORDER BY COUNT(DISTINCT p.owner), p.kind) AS s FROM p GROUP BY p.kind HAVING MAX(p.day) > 0")
        == "SELECT kqwi.kqi0 AS k, kqwi.kqi1 AS n, SUM(kqwi.kqi1) OVER (ORDER BY kqwi.kqi2, kqwi.kqi0) AS s"
        " FROM (SELECT p.kind AS kqi0, COUNT(*) AS kqi1, COUNT(DISTINCT p.owner) AS kqi2 FROM p GROUP BY p.kind HAVING MAX(p.day) > 0) AS kqwi"
    )
    # constants and the frame stay in the window
    assert (
        step_one("SELECT LAG(p.day, 1) OVER (ORDER BY p.id) AS l, SUM(p.day) OVER (ORDER BY p.id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s, COUNT(*) OVER () AS n FROM p")
        == "SELECT LAG(kqwi.kqi0, 1) OVER (ORDER BY kqwi.kqi1) AS l, SUM(kqwi.kqi0) OVER (ORDER BY kqwi.kqi1 ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s, COUNT(*) OVER () AS n"
        " FROM (SELECT p.day AS kqi0, p.id AS kqi1 FROM p) AS kqwi"
    )


def test_the_input_lists_columns_in_the_order_the_reader_uses_them():
    reader = sqlglot.parse_one("SELECT w.b, w.r, w.a FROM (SELECT p.id AS a, p.day AS b, RANK() OVER (ORDER BY p.kind) AS r FROM p) AS w WHERE w.r < 3", read="postgres")
    part = split(reader.find(exp.Subquery).this)
    assert part.input.sql(dialect="postgres") == "SELECT p.day AS kqi0, p.kind AS kqi1, p.id AS kqi2 FROM p"


@pytest.mark.parametrize("sql", [
    "SELECT p.id AS a FROM p",  # no window
    "SELECT DISTINCT p.kind AS a, RANK() OVER (ORDER BY p.kind) AS r FROM p",
    "SELECT p.id AS a, RANK() OVER (ORDER BY p.kind) AS r FROM p ORDER BY p.id LIMIT 3",
    "SELECT p.id AS a, RANK() OVER (ORDER BY p.kind) AS r FROM p QUALIFY r = 1",
    "SELECT p.id AS a, RANK() OVER w AS r FROM p WINDOW w AS (ORDER BY p.kind)",
    "SELECT p.id AS a, RANK() OVER (ORDER BY p.kind) FROM p",  # an output without a name
    "SELECT p.id AS a, p.day - LAG(p.day) OVER (ORDER BY p.id) AS d FROM p",  # a column outside the window in a window item
    "SELECT p.id AS a, SUM(p.day) FILTER (WHERE p.kind = 1) OVER (ORDER BY p.id) AS s FROM p",
    "SELECT p.id AS a, RANK() OVER (ORDER BY RANDOM()) AS r FROM p",
    "SELECT p.id AS a, RANK() OVER (ORDER BY my_udf(p.kind)) AS r FROM p",
    "SELECT p.id AS a, RANK() OVER (ORDER BY (SELECT MAX(u.rep) FROM u WHERE u.id = p.owner)) AS r FROM p",
    "SELECT p.id AS a, RANK() OVER (ORDER BY x.kind) AS r FROM p",  # a column of no source of the select
    "SELECT RANK() OVER () AS r FROM p",  # no input column at all
    "SELECT *, RANK() OVER (ORDER BY p.kind) AS r FROM p",
    "SELECT RANK() OVER (ORDER BY COUNT(*)) AS r, p.kind FROM p GROUP BY 2",  # a position: the rebuilt select list would regroup by another item
])
def test_other_shapes_are_left_alone(sql):
    assert step_one(sql) is None


def test_unify_installs_the_left_input_on_both_sides_only_after_a_proof():
    left = "SELECT w.a, w.r FROM (SELECT p.id AS a, RANK() OVER (ORDER BY p.day) AS r FROM p JOIN u ON p.owner = u.id) AS w"
    right = "SELECT v.x, v.y FROM (SELECT t.id AS x, RANK() OVER (ORDER BY t.day) AS y FROM p AS t, u AS s WHERE t.owner = s.id) AS v"
    asked = []

    def yes(a, b):
        asked.append((a, b))
        return True

    out = unify(left, right, dialect="postgres", prove=yes)
    shared = "(SELECT p.id AS kqi0n0, p.day AS kqi0n1 FROM p JOIN u ON p.owner = u.id) AS kqwi0"
    assert asked == [("SELECT p.id AS kqi0, p.day AS kqi1 FROM p JOIN u ON p.owner = u.id", "SELECT t.id AS kqi0, t.day AS kqi1 FROM p AS t, u AS s WHERE t.owner = s.id")]
    assert out == (
        f"SELECT w.a, w.r FROM (SELECT kqwi0.kqi0n0 AS a, RANK() OVER (ORDER BY kqwi0.kqi0n1) AS r FROM {shared}) AS w",
        f"SELECT v.x, v.y FROM (SELECT kqwi0.kqi0n0 AS x, RANK() OVER (ORDER BY kqwi0.kqi0n1) AS y FROM {shared}) AS v",
    )
    assert unify(left, right, dialect="postgres", prove=lambda a, b: False) is None
    # windows that read their inputs differently are never compared
    other = right.replace("ORDER BY t.day", "PARTITION BY t.day")
    assert unify(left, other, dialect="postgres", prove=yes) is None and len(asked) == 1


def test_two_unified_inputs_of_one_query_keep_separate_names():
    # the first input's first column is a number and the second's a string: under one name the comparison
    # ``w.t = 'x'`` was typed as a number against a string and the pair was declined
    schema = {"p": ["id", "owner", "title", "day"], "u": ["id", "name"]}
    types = {"p": {"id": "INT", "owner": "INT", "title": "TEXT", "day": "INT"}, "u": {"id": "INT", "name": "TEXT"}}
    left = (
        "SELECT a.id, a.r, b.t, b.s FROM (SELECT u.id, RANK() OVER (ORDER BY u.name) AS r FROM u WHERE u.id > 1) AS a"
        " JOIN (SELECT p.title AS t, p.owner AS o, ROW_NUMBER() OVER (PARTITION BY p.owner ORDER BY p.day) AS s FROM p WHERE p.day > 0) AS b"
        " ON b.o = a.id WHERE b.t = 'x'"
    )
    right = (
        "SELECT a.id, a.r, b.t, b.s FROM (SELECT u2.id, RANK() OVER (ORDER BY u2.name) AS r FROM u AS u2 WHERE 1 < u2.id) AS a"
        " JOIN (SELECT p2.title AS t, p2.owner AS o, ROW_NUMBER() OVER (PARTITION BY p2.owner ORDER BY p2.day) AS s FROM p AS p2 WHERE 0 < p2.day) AS b"
        " ON a.id = b.o AND b.t = 'x'"
    )
    out = unify(left, right, dialect="postgres", prove=lambda a, b: True)
    assert out is not None and all("kqwi0" in text and "kqwi1" in text for text in out)
    assert prove_equivalent_algebraic(left, right, schema=schema, types=types, dialect="postgres", compare_names=False).proven
    other = right.replace("0 < p2.day", "1 < p2.day")
    assert not prove_equivalent_algebraic(left, other, schema=schema, types=types, dialect="postgres", compare_names=False).proven


def test_a_window_inside_a_subquery_is_not_unified():
    # there a column can name a row of the enclosing query, and the input is not one relation
    left = "SELECT u.id FROM u WHERE u.rep IN (SELECT w.r FROM (SELECT RANK() OVER (ORDER BY p.day) AS r, p.id AS a FROM p WHERE p.owner = u.id) AS w)"
    right = left.replace("p.owner = u.id", "u.id = p.owner")
    assert unify(left, right, dialect="postgres", prove=lambda a, b: True) is None
    assert unify("WITH x AS (SELECT 1 AS a) " + left, right, dialect="postgres", prove=lambda a, b: True) is None


RANKED = "SELECT s.id, s.r FROM (SELECT p.id, RANK() OVER (PARTITION BY p.kind ORDER BY p.day DESC) AS r FROM p JOIN u ON p.owner = u.id WHERE u.rep > 1) AS s WHERE s.r <= 2"
TARGETS = [
    # the same join spelled with a comma, and as the derived table Calcite prints
    (RANKED, "SELECT s.id, s.r FROM (SELECT p.id, RANK() OVER (PARTITION BY p.kind ORDER BY p.day DESC) AS r FROM p, u WHERE p.owner = u.id AND u.rep > 1) AS s WHERE s.r <= 2"),
    (RANKED, "SELECT s.id, s.r FROM (SELECT t.id, RANK() OVER (PARTITION BY t.kind ORDER BY t.day DESC) AS r FROM (SELECT p.id, p.kind, p.day FROM p, u WHERE p.owner = u.id AND u.rep > 1) AS t) AS s WHERE s.r <= 2"),
    # the filter written twice, a predicate turned around
    (RANKED, "SELECT s.id, s.r FROM (SELECT p.id, RANK() OVER (PARTITION BY p.kind ORDER BY p.day DESC) AS r FROM p JOIN u ON u.id = p.owner AND 1 < u.rep WHERE u.rep > 1) AS s WHERE s.r <= 2"),
    # a window over a grouped select: the group keys in another order, COALESCE spelled as CASE
    (
        "SELECT s.kind, s.n, s.r FROM (SELECT p.kind, p.owner, SUM(COALESCE(c.score, 0)) AS n, DENSE_RANK() OVER (ORDER BY COUNT(c.id) DESC) AS r FROM p LEFT JOIN c ON p.id = c.post GROUP BY p.kind, p.owner) AS s",
        "SELECT s.kind, s.n, s.r FROM (SELECT p.kind, p.owner, SUM(CASE WHEN c.score IS NOT NULL THEN c.score ELSE 0 END) AS n, DENSE_RANK() OVER (ORDER BY COUNT(c.id) DESC) AS r FROM p LEFT JOIN c ON p.id = c.post GROUP BY p.owner, p.kind) AS s",
    ),
    # the window computed one level above the grouped select
    (
        "SELECT s.kind, s.n, s.r FROM (SELECT p.kind, COUNT(c.id) AS n, RANK() OVER (ORDER BY COUNT(c.id) DESC) AS r FROM p JOIN u ON p.owner = u.id LEFT JOIN c ON p.id = c.post GROUP BY p.kind) AS s",
        "SELECT s.kind, s.n, s.r FROM (SELECT g.kind, g.n, RANK() OVER (ORDER BY g.n DESC) AS r FROM (SELECT d.kind, COUNT(c.id) AS n FROM (SELECT p.id, p.kind FROM p, u WHERE p.owner = u.id) AS d LEFT JOIN c ON d.id = c.post GROUP BY d.kind) AS g) AS s",
    ),
    # a window over a window
    (
        "SELECT o.id, o.t FROM (SELECT s.id, SUM(s.r) OVER (PARTITION BY s.kind) AS t FROM (SELECT p.id, p.kind, RANK() OVER (ORDER BY p.day) AS r FROM p JOIN u ON p.owner = u.id) AS s) AS o",
        "SELECT o.id, o.t FROM (SELECT s.id, SUM(s.r) OVER (PARTITION BY s.kind) AS t FROM (SELECT p.id, p.kind, RANK() OVER (ORDER BY p.day) AS r FROM p, u WHERE u.id = p.owner) AS s) AS o",
    ),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_windows_over_equal_inputs_prove(left, right):
    assert proved(left, right)


ROWS = {
    "p": [(1, 1, 1, 10), (2, 1, 1, 20), (3, None, 1, 30), (4, 2, 2, 40), (5, 9, 1, 50), (6, 2, 1, 20), (7, 1, 2, 5), (8, 3, 1, 60)],
    "u": [(1, 7, 5), (2, 8, 6), (3, 9, 0)],
    "c": [(1, 1, 3), (2, 1, 0), (3, 4, 5), (4, None, 1), (5, 2, None)],
}
NEAR_MISSES = [
    # another input: an outer join keeps the posts without an owner, another filter keeps other rows
    (RANKED.replace(" WHERE u.rep > 1", ""), RANKED.replace(" WHERE u.rep > 1", "").replace("p JOIN u", "p LEFT JOIN u")),
    (RANKED, RANKED.replace("u.rep > 1", "u.rep > 5")),
    (RANKED, "SELECT s.id, s.r FROM (SELECT t.id, RANK() OVER (PARTITION BY t.kind ORDER BY t.day DESC) AS r FROM (SELECT DISTINCT p.kind AS id, p.kind, p.day FROM p, u WHERE p.owner = u.id AND u.rep > 1) AS t) AS s WHERE s.r <= 2"),
    # the same input read another way: partition and sort keys swapped, another direction, another function
    (RANKED, RANKED.replace("PARTITION BY p.kind ORDER BY p.day DESC", "PARTITION BY p.day ORDER BY p.kind DESC")),
    (RANKED, RANKED.replace("p.day DESC", "p.day")),
    (RANKED, RANKED.replace("RANK()", "DENSE_RANK()").replace("p JOIN u ON p.owner = u.id WHERE u.rep > 1", "p, u WHERE p.owner = u.id AND u.rep > 1").replace("p.kind ORDER", "u.rep ORDER")),
    # a running total against a total
    (
        "SELECT s.id, s.t FROM (SELECT p.id, SUM(p.day) OVER (PARTITION BY p.kind) AS t FROM p JOIN u ON p.owner = u.id) AS s",
        "SELECT s.id, s.t FROM (SELECT p.id, SUM(p.day) OVER (PARTITION BY p.kind ORDER BY p.id) AS t FROM p, u WHERE p.owner = u.id) AS s",
    ),
    # the filter above the window against the filter below it
    (RANKED, "SELECT s.id, s.r FROM (SELECT p.id, RANK() OVER (PARTITION BY p.kind ORDER BY p.day DESC) AS r FROM p JOIN u ON p.owner = u.id) AS s WHERE s.r <= 2"),
    # the window before grouping against the window after it
    (
        "SELECT s.kind, s.r FROM (SELECT p.kind, RANK() OVER (ORDER BY COUNT(*) DESC) AS r FROM p JOIN u ON p.owner = u.id GROUP BY p.kind) AS s",
        "SELECT s.kind, s.r FROM (SELECT p.kind, RANK() OVER (ORDER BY p.kind DESC) AS r FROM p, u WHERE p.owner = u.id GROUP BY p.kind) AS s",
    ),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, ROWS)


# windows whose value does not depend on how ties are ordered, so DuckDB's answer is one answer
WINDOWED = [
    "SELECT p.id AS a, RANK() OVER (PARTITION BY p.kind ORDER BY p.day DESC) AS r, COUNT(*) OVER (PARTITION BY p.owner) AS n FROM p JOIN u ON p.owner = u.id WHERE u.rep > 1",
    "SELECT p.kind AS k, COUNT(*) AS n, SUM(COUNT(*)) OVER (ORDER BY COUNT(DISTINCT p.owner), p.kind) AS s FROM p GROUP BY p.kind HAVING MAX(p.day) > 0",
    "SELECT COUNT(*) AS n, MAX(p.day) AS m, RANK() OVER (ORDER BY COUNT(*)) AS r FROM p LEFT JOIN c ON p.id = c.post",  # a global aggregate: one row
    "SELECT p.kind AS k, MIN(c.score) AS m, DENSE_RANK() OVER (ORDER BY MIN(c.score) NULLS FIRST) AS r, SUM(SUM(c.score)) OVER () AS t FROM p LEFT JOIN c ON p.id = c.post GROUP BY p.kind",
    "SELECT p.id AS a, p.day AS d, SUM(p.day) OVER (PARTITION BY p.kind ORDER BY p.day RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s, COUNT(DISTINCT p.owner) OVER (PARTITION BY p.kind) AS o FROM p",
    "SELECT p.owner AS o, 7 AS seven, MAX(p.day) OVER (PARTITION BY p.owner, p.kind) AS m, FIRST_VALUE(p.day) OVER (PARTITION BY p.owner ORDER BY p.day, p.id, p.kind) AS f FROM p WHERE p.kind IS NOT NULL",
]


def random_rows(rng):
    def table(width, size):
        rows = [tuple(rng.choice([None, 1, 2, 3]) for _ in range(width)) for _ in range(rng.randint(0, size))]
        return rows + rows[:2]  # with duplicate rows

    return {"p": table(4, 7), "u": table(3, 4), "c": table(3, 6)}


@pytest.mark.parametrize("sql", WINDOWED)
def test_the_isolated_form_gives_the_same_rows(sql):
    out = step_one(sql)
    assert out is not None and "kqwi" in out
    rng = random.Random(len(sql))
    for rows in [{}, ROWS] + [random_rows(rng) for _ in range(14)]:
        assert not differ_on_duckdb(sql, out, rows)
