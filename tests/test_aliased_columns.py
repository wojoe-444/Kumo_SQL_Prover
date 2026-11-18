"""A bare column in a select list, WHERE or ON is a source column (``kumo_sql_prover.aliased_columns``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.aliased_columns import qualify_aliased_columns
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a", "b"}), keys=(("a", "b"),))}
G = "(SELECT a AS k, MAX(c) AS top FROM t GROUP BY a) AS g"


def rewritten(sql, schema=SCHEMA):
    before = sqlglot.parse_one(sql, read="mysql")
    text = before.sql(dialect="mysql")
    after = qualify_aliased_columns(before, schema, "mysql").sql(dialect="mysql")
    return after if after != text else None


def proved(left, right, constraints=CONSTRAINTS):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=constraints, compare_names=False, dialect="mysql").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE t (a BIGINT, b BIGINT, c BIGINT, d BIGINT)")
    db.execute("CREATE TABLE u (a BIGINT, e BIGINT)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


def test_a_bare_column_under_its_own_alias_names_its_source():
    assert rewritten(f"SELECT s.b, CASE WHEN g.top > 1 THEN c * 2 ELSE c END AS c FROM t AS s JOIN {G} ON s.a = g.k") == (
        f"SELECT s.b, CASE WHEN g.top > 1 THEN s.c * 2 ELSE s.c END AS c FROM t AS s JOIN {G} ON s.a = g.k"
    )
    # every bare name with one owner, in the select list, the WHERE and the ON
    assert rewritten(f"SELECT b, top + d AS d FROM t JOIN {G} ON a = k AND c > 0 WHERE top > e OR d IS NULL") == (
        f"SELECT t.b, g.top + t.d AS d FROM t JOIN {G} ON t.a = g.k AND t.c > 0 WHERE g.top > e OR t.d IS NULL"
    )
    # a window's own clauses are read in the select list too
    assert rewritten("SELECT e, RANK() OVER (PARTITION BY b ORDER BY c) AS c FROM t JOIN u ON u.a = t.a") == (
        "SELECT u.e, RANK() OVER (PARTITION BY t.b ORDER BY t.c) AS c FROM t JOIN u ON u.a = t.a"
    )


def test_what_may_read_an_output_stays_bare():
    assert rewritten(f"SELECT s.b, top + c AS c FROM t AS s JOIN {G} ON s.a = g.k GROUP BY b, c HAVING c > 1 ORDER BY c") == (
        f"SELECT s.b, g.top + s.c AS c FROM t AS s JOIN {G} ON s.a = g.k GROUP BY b, c HAVING c > 1 ORDER BY c"
    )


@pytest.mark.parametrize("sql", [
    "SELECT b + 1 AS b FROM t",  # one source: left to the other passes
    "SELECT b + 1 AS b FROM t JOIN u USING (a)",  # a merged column belongs to no single source
    "SELECT b + 1 AS b FROM t NATURAL JOIN u",
    "SELECT b + 1 AS b FROM t JOIN (SELECT * FROM u) AS s ON s.a = t.a",  # a star: the columns of s are not known here
    "SELECT b + 1 AS b FROM t JOIN (SELECT a, e + 1 FROM u) AS s ON s.a = t.a",  # an unnamed output
    "SELECT b + 1 AS b FROM t JOIN v ON v.a = t.a",  # a table whose columns were not supplied
    "SELECT z + 1 AS z FROM t JOIN u ON u.a = t.a",  # nobody has z: a reference to an outer query
    "SELECT t.b FROM t JOIN u ON u.a = t.a WHERE EXISTS(SELECT 1 FROM u AS w WHERE w.e = b)",  # below a nested query
    "SELECT t.b FROM t JOIN u ON u.a = t.a WHERE t.c IN (SELECT e FROM u)",
    "SELECT t.b FROM t JOIN (SELECT a, e FROM u WHERE e > 0) AS s ON s.a = t.a",  # inside the derived table: one source
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_the_ambiguous_name_stays_bare():
    # a: t's or u's (the engine rejects the query); e has one owner
    assert rewritten("SELECT a, e FROM t JOIN u ON t.b = u.e") == "SELECT a, u.e FROM t JOIN u ON t.b = u.e"
    assert rewritten("SELECT b + 1 AS b FROM t JOIN u ON u.a = t.a", schema=None) is None


PROOFS = [
    # the reference spells the column bare under its own alias, the other query qualifies it
    (
        f"SELECT s.b, CASE WHEN g.top > 1 THEN c * 2 ELSE c END AS c FROM t AS s JOIN {G} ON s.a = g.k",
        f"SELECT s.b, CASE WHEN g.top > 1 THEN s.c * 2 ELSE s.c END AS c FROM t AS s JOIN {G} ON s.a = g.k",
    ),
    ("SELECT e + d AS d FROM t JOIN u ON u.a = t.a WHERE d > 0", "SELECT u.e + t.d FROM u JOIN t ON t.a = u.a WHERE t.d > 0"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    ("SELECT e + d AS d FROM t JOIN u ON u.a = t.a WHERE d > 0", "SELECT u.e + t.d FROM u JOIN t ON t.a = u.a WHERE u.e + t.d > 0"),
    ("SELECT e + d AS d FROM t JOIN u ON u.a = t.a WHERE d > 0", "SELECT u.e + t.c FROM u JOIN t ON t.a = u.a WHERE t.d > 0"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


QUERIES = [
    f"SELECT s.b, CASE WHEN g.top > 1 THEN c * 2 ELSE c END AS c FROM t AS s JOIN {G} ON s.a = g.k",
    f"SELECT b, top + d AS d FROM t JOIN {G} ON a = k AND c > 0 WHERE top > 1 OR d IS NULL",
    "SELECT e, RANK() OVER (PARTITION BY b ORDER BY c) AS c FROM t JOIN u ON u.a = t.a",
    f"SELECT s.b, top + c AS c FROM t AS s JOIN {G} ON s.a = g.k WHERE c > 0 ORDER BY c",
    "SELECT e + d AS d FROM t LEFT JOIN u ON u.a = t.a AND e > 0 WHERE d > 0 OR e IS NULL",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_qualified_query_returns_the_same_rows(sql):
    """DuckDB resolves the bare names of the written query itself; the qualified query must return the same rows."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 2, 3])  # noqa: E731
    for _ in range(150):
        keys = rng.sample([(i, j) for i in range(3) for j in range(4)], rng.choice([0, 1, 2, 4, 7]))
        tables = {
            "t": [(a, b, value(), value()) for a, b in keys],
            "u": [(value(), value()) for _ in range(rng.choice([0, 1, 3]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
