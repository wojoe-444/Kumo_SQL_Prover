"""One spelling for the input of a windowed select (``kumo_sql_prover.windowed_inputs``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.windowed_inputs import spell_windowed_input

SCHEMA = {"t": ["a", "b", "c"], "u": ["a", "b", "c"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}
DOUBLES = {name: {column: "DOUBLE" for column in columns} for name, columns in SCHEMA.items()}


def spelled(sql, types=TYPES):
    out = spell_windowed_input(sqlglot.parse_one(sql, read="postgres"), types)
    return out.sql(dialect="postgres") if out is not None else None


def proved(left, right, types=TYPES):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=types, dialect="postgres", compare_names=False).proven


def rows_on_duckdb(queries, rows, kind="BIGINT"):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name in SCHEMA:
        db.execute(f"CREATE TABLE {name} (a {kind}, b {kind}, c {kind})")
        insert_rows(db, name, rows.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def differ_on_duckdb(left, right, rows, kind="BIGINT"):
    a, b = rows_on_duckdb([left, right], rows, kind)
    return a != b


WINDOW = "SUM(u.b) OVER (PARTITION BY u.c)"


def test_conditions_move_to_where_and_sources_and_conditions_get_one_order():
    want = f"SELECT t.b, {WINDOW} AS s FROM t AS t, u AS u WHERE t.a = u.a AND t.c = 10"
    assert spelled(f"SELECT t.b, {WINDOW} AS s FROM t AS t JOIN u AS u ON t.a = u.a AND t.c = 10") == want
    assert spelled(f"SELECT t.b, {WINDOW} AS s FROM u AS u, t AS t WHERE t.c = 10 AND u.a = t.a") == want
    assert spelled(f"SELECT t.b, {WINDOW} AS s FROM u AS u CROSS JOIN t AS t WHERE t.c = 10 AND u.a = t.a AND t.a = u.a") == want
    assert spelled(f"SELECT t.b, {WINDOW} AS s FROM t AS t INNER JOIN u AS u ON t.a = u.a WHERE t.c = 10") == want
    assert spelled(want) is None  # already spelled
    # the order does not depend on the aliases: the same query under other names is spelled the same way
    renamed = spelled("SELECT z.b, SUM(a.b) OVER (PARTITION BY a.c) AS s FROM u AS a JOIN t AS z ON z.c = 10 AND z.a = a.a")
    assert renamed == "SELECT z.b, SUM(a.b) OVER (PARTITION BY a.c) AS s FROM t AS z, u AS a WHERE z.a = a.a AND z.c = 10"
    # two reads of one table keep their order
    assert spelled("SELECT x.b, SUM(y.b) OVER (PARTITION BY y.c) AS s FROM t AS x JOIN t AS y ON y.a = x.a") == (
        "SELECT x.b, SUM(y.b) OVER (PARTITION BY y.c) AS s FROM t AS x, t AS y WHERE x.a = y.a"
    )


def test_integer_sums_inside_a_window_get_their_linear_form():
    def window(argument, types=TYPES):
        return spelled(f"SELECT SUM({argument}) OVER (PARTITION BY t.c) AS s FROM t AS t", types)

    want = "SELECT SUM(2 * t.b + 100) OVER (PARTITION BY t.c) AS s FROM t AS t"
    assert window("t.b + t.b + 100") == want
    assert window("t.b * 2 + 100") == want
    assert window("100 + 2 * t.b") == want
    assert window("(t.b + 50) * 2") == want
    assert window("2 * t.b + 100") is None
    assert window("t.a - t.b - 1") is None
    assert window("0 - 1 - t.b + t.a") == "SELECT SUM(t.a - t.b - 1) OVER (PARTITION BY t.c) AS s FROM t AS t"
    assert window("t.b - t.a") == "SELECT SUM(-1 * t.a + t.b) OVER (PARTITION BY t.c) AS s FROM t AS t"
    assert window("t.b - t.b") == "SELECT SUM(0 * t.b) OVER (PARTITION BY t.c) AS s FROM t AS t"  # still NULL for a NULL b
    assert window("t.b * t.b + t.b") is None  # not linear
    assert window("t.b / 2 + t.b / 2") is None
    # an integer part of a wider expression is still an integer sum
    assert window("t.b + t.b + 1.5") == "SELECT SUM((2 * t.b) + 1.5) OVER (PARTITION BY t.c) AS s FROM t AS t"
    assert window("ABS(t.b) + ABS(t.b)") is None
    assert window("t.b + t.b + 100", DOUBLES) is None  # floating sums do not reassociate
    assert window("t.b + t.b + 100", {}) is None
    # only inside the window
    assert spelled("SELECT t.b + t.b AS d, SUM(t.b) OVER (PARTITION BY t.c) AS s FROM t AS t") is None


def test_a_filtering_derived_table_is_read_as_its_table():
    want = f"SELECT x.b, {WINDOW} AS s FROM t AS x, u AS u WHERE x.a = u.a AND x.c = 10"
    assert spelled(f"SELECT x.b, {WINDOW} AS s FROM (SELECT a, b FROM t WHERE c = 10) AS x JOIN u AS u ON x.a = u.a") == want
    assert spelled(f"SELECT x.b, {WINDOW} AS s FROM (SELECT * FROM (SELECT a AS a, b AS b, c = 10 AS f FROM t) AS d WHERE f) AS x, u AS u WHERE x.a = u.a") is None  # a star
    nested = f"SELECT x.b, {WINDOW} AS s FROM (SELECT d.a, d.b, d.f FROM (SELECT a AS a, b AS b, c = 10 AS f FROM t) AS d WHERE f) AS x, u AS u WHERE x.a = u.a"
    assert spelled(nested) == want
    flagged = f"SELECT x.b, {WINDOW} AS s FROM (SELECT a, b, CASE WHEN c = 10 THEN 1 ELSE 0 END AS f FROM t) AS x JOIN u AS u ON x.a = u.a AND x.f = 1"
    assert spelled(flagged) == want
    # an output computed from the table's columns is read as that computation
    assert spelled("SELECT SUM(x.d) OVER (PARTITION BY x.c) AS s FROM (SELECT b + b AS d, c FROM t) AS x") == (
        "SELECT SUM(2 * x.b) OVER (PARTITION BY x.c) AS s FROM t AS x"
    )


@pytest.mark.parametrize("source", [
    "(SELECT DISTINCT a, b FROM t WHERE c = 10) AS x",
    "(SELECT a, b FROM t WHERE c = 10 LIMIT 3) AS x",
    "(SELECT a, MAX(b) AS b FROM t WHERE c = 10 GROUP BY a) AS x",
    "(SELECT a, b FROM t WHERE c = 10 ORDER BY b) AS x",
    "(SELECT a, ABS(b) AS b FROM t WHERE c = 10) AS x",
    "(SELECT a, b FROM t WHERE c IN (SELECT c FROM u)) AS x",
    "(SELECT t.a, t.b FROM t JOIN u AS v ON t.a = v.a) AS x",
    "(SELECT a, b FROM t WHERE c = 10) AS x (a, b)",
])
def test_other_derived_tables_stay(source):
    out = spelled(f"SELECT x.b, {WINDOW} AS s FROM {source} JOIN u AS u ON x.a = u.a")
    assert out is None or "FROM (SELECT" in out


@pytest.mark.parametrize("sql", [
    "SELECT t.b FROM t AS t JOIN u AS u ON t.a = u.a AND t.c = 10",  # no window
    f"SELECT t.b, {WINDOW} AS s FROM t AS t LEFT JOIN u AS u ON t.a = u.a AND t.c = 10",
    f"SELECT t.b, {WINDOW} AS s FROM t AS t RIGHT JOIN u AS u ON t.a = u.a AND t.c = 10",
    f"SELECT t.b, {WINDOW} AS s FROM t AS t FULL JOIN u AS u ON t.a = u.a AND t.c = 10",
    f"SELECT t.b, {WINDOW} AS s FROM t AS t JOIN u AS u USING (a)",
    f"SELECT t.b, {WINDOW} AS s FROM t AS t NATURAL JOIN u AS u",
    f"SELECT t.b, {WINDOW} AS s FROM t AS t, LATERAL (SELECT * FROM u WHERE u.a = t.a) AS u",
    f"SELECT t.b, {WINDOW} AS s FROM t AS t JOIN u AS u ON t.a = u.a AND t.c = 10 ORDER BY 1 LIMIT 3",
    f"SELECT DISTINCT t.b, {WINDOW} AS s FROM t AS t JOIN u AS u ON t.a = u.a AND t.c = 10",
    "SELECT t.b, SUM(COUNT(*)) OVER () AS s FROM t AS t JOIN u AS u ON t.a = u.a AND t.c = 10 GROUP BY t.b",
    f"SELECT t.b, {WINDOW} AS s FROM t AS t JOIN u AS u ON t.a = u.a AND t.c = 10 QUALIFY s > 1",
])
def test_other_selects_are_left_alone(sql):
    assert spelled(sql) is None


BASE = "SELECT t.b + u.c AS k, SUM(u.b + u.b + 100) OVER (PARTITION BY u.c) AS s FROM t JOIN u ON t.a = u.a AND t.c = 10"
TARGETS = [
    (BASE, "SELECT t.b + u.c AS k, SUM(u.b * 2 + 100) OVER (PARTITION BY u.c) AS s FROM t JOIN u ON t.a = u.a WHERE t.c = 10"),
    (BASE, "SELECT x.b + y.c AS k, SUM(2 * y.b + 100) OVER (PARTITION BY y.c) AS s FROM u AS y, t AS x WHERE x.c = 10 AND y.a = x.a"),
    (BASE, "SELECT x.b + y.c AS k, SUM(y.d) OVER (PARTITION BY y.c) AS s FROM (SELECT a, b FROM t WHERE c = 10) AS x JOIN (SELECT a, c, 2 * b + 100 AS d FROM u) AS y ON x.a = y.a"),
    (BASE, "SELECT x.b + y.c AS k, SUM(y.d) OVER (PARTITION BY y.c) AS s FROM (SELECT a, b, c = 10 AS f FROM t) AS x JOIN (SELECT a, c, b + b + 100 AS d FROM u) AS y ON x.a = y.a AND x.f"),
    (BASE, "SELECT x.b + y.c AS k, SUM(y.d) OVER (PARTITION BY y.c) AS s FROM (SELECT a, b, CASE WHEN c = 10 THEN 1 ELSE 0 END AS f FROM t) AS x JOIN (SELECT a, c, b * 2 + 100 AS d FROM u) AS y ON x.a = y.a AND x.f = 1"),
    (BASE, "SELECT x.b + y.c AS k, SUM(y.b + 100 + y.b) OVER (PARTITION BY y.c ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS s FROM (SELECT d.a, d.b, d.f FROM (SELECT a, b, c = 10 AS f FROM t) AS d WHERE f) AS x, u AS y WHERE x.a = y.a"),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_windowed_inputs_prove(left, right):
    assert proved(left, right)


ROWS = {
    "t": [(1, 5, 10), (1, 5, 10), (2, 7, 10), (3, 9, 2), (None, 4, 10), (4, 1, None)],
    "u": [(1, 1, 1), (1, None, 1), (1, 4, 2), (2, 2, 2), (2, None, 2), (2, None, 5), (4, 2, 2), (None, 3, 3)],
}
PLAIN = "SELECT t.b AS k, SUM(u.b) OVER (PARTITION BY u.c) AS s FROM t JOIN u ON t.a = u.a AND t.c = 10"
NEAR_MISSES = [
    # a condition moved across an outer join
    (PLAIN.replace("JOIN", "LEFT JOIN"), "SELECT t.b AS k, SUM(u.b) OVER (PARTITION BY u.c) AS s FROM t LEFT JOIN u ON t.a = u.a WHERE t.c = 10"),
    (PLAIN.replace("JOIN", "LEFT JOIN"), PLAIN),
    # another sum, another partition, another condition
    (BASE, BASE.replace("u.b + u.b + 100", "2 * u.b + 101")),
    (BASE, BASE.replace("u.b + u.b + 100", "3 * u.b + 100")),
    (PLAIN, PLAIN.replace("PARTITION BY u.c", "PARTITION BY u.a")),
    (PLAIN, PLAIN.replace("t.c = 10", "t.c = 2")),
    (PLAIN, PLAIN.replace(" AND t.c = 10", "")),
    # x - x is NULL for a NULL x, not 0
    (PLAIN.replace("SUM(u.b)", "SUM(u.b - u.b)"), PLAIN.replace("SUM(u.b)", "SUM(0)")),
    # a flag compared with its ELSE value, or one that is the same on both arms
    (PLAIN, PLAIN.replace("t.c = 10", "CASE WHEN t.c = 10 THEN 1 ELSE 0 END = 0")),
    (PLAIN.replace("t.c = 10", "NOT t.c = 10"), PLAIN.replace("t.c = 10", "CASE WHEN t.c = 10 THEN 1 ELSE 0 END = 0")),
    (PLAIN, PLAIN.replace("t.c = 10", "CASE WHEN t.c = 10 THEN 1 ELSE 1 END = 1")),
    # a derived table that removes duplicates or caps its rows is not its table
    (PLAIN, "SELECT x.b AS k, SUM(u.b) OVER (PARTITION BY u.c) AS s FROM (SELECT DISTINCT a, b FROM t WHERE c = 10) AS x JOIN u ON x.a = u.a"),
    (PLAIN, "SELECT x.b AS k, SUM(u.b) OVER (PARTITION BY u.c) AS s FROM (SELECT a, b FROM t WHERE c = 10 ORDER BY a, b LIMIT 1) AS x JOIN u ON x.a = u.a"),
    # the window frame still matters
    (PLAIN, PLAIN.replace("PARTITION BY u.c", "PARTITION BY u.c ORDER BY u.a, u.b, t.b ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW")),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, ROWS)


def test_floating_sums_are_not_reassociated():
    left = "SELECT SUM(t.a + t.b + t.b) OVER (PARTITION BY t.c) AS s FROM t"
    right = "SELECT SUM(t.a + 2 * t.b) OVER (PARTITION BY t.c) AS s FROM t"
    assert proved(left, right)  # integers
    assert not proved(left, right, DOUBLES)
    assert differ_on_duckdb(left, right, {"t": [(1e16, 1.0, 1.0)]}, kind="DOUBLE")  # (1e16 + 1) + 1 is 1e16


def test_a_cancelled_column_still_makes_the_sum_null():
    left = "SELECT SUM(t.a + t.b - t.b) OVER (PARTITION BY t.c) AS s FROM t"
    right = "SELECT SUM(t.a) OVER (PARTITION BY t.c) AS s FROM t"
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, {"t": [(1, None, 1)]})


SPELLED_QUERIES = [
    BASE,
    PLAIN,
    "SELECT x.b + y.c AS k, SUM(y.d - x.b) OVER (PARTITION BY y.c) AS s, COUNT(*) OVER (PARTITION BY x.a) AS n FROM (SELECT a, b FROM t WHERE c = 10 OR b > 4) AS x JOIN (SELECT a, c, 2 * b + 100 AS d FROM u) AS y ON x.a = y.a",
    "SELECT x.b AS k, MIN(y.b - 3 * (x.b - y.b)) OVER (PARTITION BY x.b) AS s FROM (SELECT d.a, d.b, d.f FROM (SELECT a, b, c = 10 AS f FROM t) AS d WHERE f) AS x, u AS y WHERE x.a = y.a",
    "SELECT x.b AS k, SUM(y.b) OVER (PARTITION BY y.c) AS s FROM (SELECT a, b, CASE WHEN c = 10 THEN 1 ELSE 0 END AS f FROM t) AS x JOIN u AS y ON x.a = y.a AND x.f = 1",
    "SELECT x.b AS k, MAX(y.b + y.a) OVER (PARTITION BY x.c) AS s FROM t AS x CROSS JOIN t AS y WHERE (x.a = y.b) IS TRUE AND (y.c = 10) = TRUE",
]


def random_rows(rng):
    def table(size):
        rows = [(rng.choice([None, 1, 2, 3]), rng.choice([None, 0, 1, 2, 5]), rng.choice([None, 2, 10])) for _ in range(rng.randint(0, size))]
        return rows + rows[:2]  # with duplicate rows

    return {"t": table(6), "u": table(5)}


@pytest.mark.parametrize("sql", SPELLED_QUERIES)
def test_the_spelling_gives_the_same_rows(sql):
    qualified = normalize(sql, schema=SCHEMA, types=TYPES, dialect="postgres")
    assert " JOIN " not in qualified and " ON " not in qualified  # the rule fired: one FROM list, one WHERE
    rng = random.Random(len(sql))
    for rows in [{}, ROWS] + [random_rows(rng) for _ in range(14)]:
        assert not differ_on_duckdb(sql, qualified, rows)
