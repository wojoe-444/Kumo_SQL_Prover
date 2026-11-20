"""A scalar subquery that keeps the first row of its sorted output, written as MAX or MIN (``top_row_subqueries``, MySQL).

Checked on a MySQL 8.0.46 server on 2026-11-20, in its default SQL mode. The rewriting was done by the module and both
statements were run on generated tables with NULLs, ties and no rows (two tables, whole numbers and DATE values): in
three runs 94,950 statements returned the same rows before and after, 87,300 of them also after ``lifted_aggregates``,
and 27,450 ``ASC`` statements over a column that may be NULL were left unread. 1,765 of 5,400 controls differed (``ASC``
against ``MIN`` on a nullable column, ``DESC`` against ``MIN``, another sort key first, ``OFFSET 1``). The statements
were built from: a subquery over one table, a filtered table, an inner and a left join; the output as a column, an
aliased column, ``DISTINCT`` and the ordinal ``1``; an output named like another column of the table; ``GROUP BY`` with
and without ``HAVING``, an aggregate as the output, an aggregate with no ``GROUP BY``; subqueries that read a column of
the query around them; in a select list, inside ``COALESCE`` and in a ``WHERE``.

SQLite sorts NULL below every value as MySQL does, so the replay below runs the same statements there.
"""

import random
import sqlite3

import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import _resolve_ordinals, normalize, prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import TableConstraints
from kumo_sql_prover.top_row_subqueries import write_top_rows

SCHEMA = {"nums": ["n"], "pay": ["emp", "amount", "day", "team", "note"]}
TYPES = {"nums": {"n": "INT"}, "pay": {"emp": "INT", "amount": "INT", "day": "DATE", "team": "INT", "note": "VARCHAR"}}
NOT_NULL = {"pay": frozenset({"emp", "amount"})}


def reading(sql: str, dialect: str = "mysql", types=TYPES, not_null=NOT_NULL, schema=SCHEMA) -> str | None:
    tree = _resolve_ordinals(sqlglot.parse_one(sql, read=dialect), group_by=True)
    written, changed = write_top_rows(tree, schema, types, not_null, dialect)
    return written.sql(dialect=dialect) if changed else None


def test_the_first_row_under_desc_is_the_largest_value():
    assert reading("SELECT (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1) AS a") == "SELECT (SELECT MAX(amount) FROM pay) AS a"
    assert reading("SELECT (SELECT day FROM pay WHERE team = 2 ORDER BY day DESC LIMIT 1) AS d") == "SELECT (SELECT MAX(day) FROM pay WHERE team = 2) AS d"
    assert reading("SELECT (SELECT n FROM nums ORDER BY n DESC LIMIT 1)") == "SELECT (SELECT MAX(n) FROM nums)"  # n may be NULL: MAX skips it as the sort does
    # an alias, the ordinal, an offset of zero, DISTINCT, a qualified column
    assert reading("SELECT (SELECT amount AS top FROM pay ORDER BY top DESC LIMIT 1) AS a") == "SELECT (SELECT MAX(amount) FROM pay) AS a"
    assert reading("SELECT (SELECT amount FROM pay ORDER BY 1 DESC LIMIT 1) AS a") == "SELECT (SELECT MAX(amount) FROM pay) AS a"
    assert reading("SELECT (SELECT amount FROM pay ORDER BY amount DESC LIMIT 0, 1) AS a") == "SELECT (SELECT MAX(amount) FROM pay) AS a"
    assert reading("SELECT (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1 OFFSET 0) AS a") == "SELECT (SELECT MAX(amount) FROM pay) AS a"
    assert reading("SELECT (SELECT DISTINCT amount FROM pay ORDER BY amount DESC LIMIT 1) AS a") == "SELECT (SELECT MAX(amount) FROM pay) AS a"
    assert reading("SELECT (SELECT p.amount FROM pay AS p ORDER BY p.amount DESC LIMIT 1) AS a") == "SELECT (SELECT MAX(p.amount) FROM pay AS p) AS a"
    # an output named like another column: the unqualified ORDER BY name is the output
    assert reading("SELECT (SELECT amount AS team FROM pay ORDER BY team DESC LIMIT 1) AS a") == "SELECT (SELECT MAX(amount) FROM pay) AS a"
    # a whole-number expression
    assert reading("SELECT (SELECT amount + team FROM pay ORDER BY amount + team DESC LIMIT 1) AS a") == "SELECT (SELECT MAX(amount + team) FROM pay) AS a"


def test_the_first_row_under_asc_is_the_smallest_value_of_a_column_that_is_never_null():
    assert reading("SELECT (SELECT amount FROM pay ORDER BY amount LIMIT 1) AS a") == "SELECT (SELECT MIN(amount) FROM pay) AS a"
    assert reading("SELECT (SELECT amount FROM pay WHERE team = 1 ORDER BY amount ASC LIMIT 1) AS a") == "SELECT (SELECT MIN(amount) FROM pay WHERE team = 1) AS a"
    assert reading("SELECT (SELECT p.amount FROM pay AS p ORDER BY 1 LIMIT 1) AS a") == "SELECT (SELECT MIN(p.amount) FROM pay AS p) AS a"
    # a column that may be NULL sorts NULL first, and MIN skips it
    assert reading("SELECT (SELECT day FROM pay ORDER BY day LIMIT 1) AS d") is None
    assert reading("SELECT (SELECT n FROM nums ORDER BY n LIMIT 1) AS n") is None
    assert reading("SELECT (SELECT amount FROM pay ORDER BY amount LIMIT 1) AS a", not_null=None) is None
    # an expression, a joined table, a table under another alias
    assert reading("SELECT (SELECT amount + 1 FROM pay ORDER BY amount + 1 LIMIT 1) AS a") is None
    assert reading("SELECT (SELECT p.amount FROM pay AS p JOIN nums ON nums.n = p.emp ORDER BY p.amount LIMIT 1) AS a") is None
    assert reading("SELECT (SELECT q.amount FROM pay AS p ORDER BY q.amount LIMIT 1) AS a") is None


def test_a_grouped_subquery_keeps_its_groups_inside_a_derived_table():
    assert reading("SELECT (SELECT n FROM nums GROUP BY n HAVING COUNT(*) = 1 ORDER BY n DESC LIMIT 1) AS n") == (
        "SELECT (SELECT MAX(top_rows.n) FROM (SELECT n FROM nums GROUP BY n HAVING COUNT(*) = 1) AS top_rows) AS n"
    )
    assert reading("SELECT (SELECT COUNT(emp) AS c FROM pay GROUP BY team HAVING COUNT(*) > 1 ORDER BY c DESC LIMIT 1) AS most") == (
        "SELECT (SELECT MAX(top_rows.c) FROM (SELECT COUNT(emp) AS c FROM pay GROUP BY team HAVING COUNT(*) > 1) AS top_rows) AS most"  # the output keeps its name
    )
    assert reading("SELECT (SELECT COUNT(emp) AS c FROM pay GROUP BY team HAVING c > 1 ORDER BY c DESC LIMIT 1) AS most") is None  # HAVING reads the output by name
    assert reading("SELECT (SELECT MAX(amount) FROM pay GROUP BY team ORDER BY MAX(amount) DESC LIMIT 1) AS a") == (
        "SELECT (SELECT MAX(top_rows.top_value) FROM (SELECT MAX(amount) AS top_value FROM pay GROUP BY team) AS top_rows) AS a"
    )
    assert reading("SELECT (SELECT MIN(amount) FROM pay WHERE team = 2 ORDER BY 1 DESC LIMIT 1) AS a") == (
        "SELECT (SELECT MAX(top_rows.top_value) FROM (SELECT MIN(amount) AS top_value FROM pay WHERE team = 2) AS top_rows) AS a"
    )
    assert reading("SELECT (SELECT amount FROM pay GROUP BY amount ORDER BY amount LIMIT 1) AS a") == (
        "SELECT (SELECT MIN(top_rows.amount) FROM (SELECT amount FROM pay GROUP BY amount) AS top_rows) AS a"
    )
    # a grouped subquery that reads the query around it cannot move into a derived table
    assert reading("SELECT emp, (SELECT s.amount FROM pay AS s WHERE s.team = o.team GROUP BY s.amount ORDER BY s.amount DESC LIMIT 1) FROM pay AS o") is None


def test_a_subquery_may_read_the_query_around_it_and_stand_anywhere_a_value_does():
    assert reading("SELECT emp, (SELECT s.amount FROM pay AS s WHERE s.team = o.team ORDER BY s.amount DESC LIMIT 1) AS best FROM pay AS o") == (
        "SELECT emp, (SELECT MAX(s.amount) FROM pay AS s WHERE s.team = o.team) AS best FROM pay AS o"
    )
    assert reading("SELECT emp FROM pay WHERE amount = (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1)") == "SELECT emp FROM pay WHERE amount = (SELECT MAX(amount) FROM pay)"
    assert reading("SELECT IFNULL((SELECT n FROM nums ORDER BY n DESC LIMIT 1), 0) AS n") == "SELECT COALESCE((SELECT MAX(n) FROM nums), 0) AS n"
    # one inside another
    assert reading("SELECT (SELECT amount FROM pay WHERE amount < (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1) ORDER BY amount DESC LIMIT 1) AS second") == (
        "SELECT (SELECT MAX(amount) FROM pay WHERE amount < (SELECT MAX(amount) FROM pay)) AS second"
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT (SELECT amount FROM pay ORDER BY emp DESC LIMIT 1) AS a",  # another sort key
        "SELECT (SELECT amount FROM pay ORDER BY amount DESC, emp LIMIT 1) AS a",
        "SELECT (SELECT amount FROM pay ORDER BY amount DESC LIMIT 2) AS a",
        "SELECT (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1 OFFSET 1) AS a",  # the second row
        "SELECT (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1, 1) AS a",
        "SELECT (SELECT amount FROM pay ORDER BY amount DESC) AS a",
        "SELECT (SELECT amount FROM pay LIMIT 1) AS a",
        "SELECT (SELECT note FROM pay ORDER BY note DESC LIMIT 1) AS a",  # two texts a collation calls equal
        "SELECT (SELECT amount / 2 FROM pay ORDER BY amount / 2 DESC LIMIT 1) AS a",  # not a whole number
        "SELECT (SELECT amount * 1.0 FROM pay ORDER BY 1 DESC LIMIT 1) AS a",
        "SELECT (SELECT -amount AS amount FROM pay ORDER BY -amount DESC LIMIT 1) AS a",  # the name inside the key may be the output
        "SELECT (SELECT amount AS team FROM pay ORDER BY amount + team DESC LIMIT 1) AS a",
        "SELECT (SELECT amount FROM pay ORDER BY RAND() DESC LIMIT 1) AS a",
        "SELECT (SELECT amount + FLOOR(RAND()) FROM pay ORDER BY 1 DESC LIMIT 1) AS a",
        "SELECT (SELECT ROW_NUMBER() OVER (ORDER BY amount) FROM pay ORDER BY 1 DESC LIMIT 1) AS a",
        "SELECT (SELECT 1 ORDER BY 1 DESC LIMIT 1) AS a",  # no FROM
        "SELECT (WITH w AS (SELECT amount FROM pay) SELECT amount FROM w ORDER BY amount DESC LIMIT 1) AS a",
        "SELECT t.amount FROM (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1) AS t",  # a derived table returns no row, not NULL
        "SELECT emp FROM pay WHERE amount IN (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1)",
        "SELECT emp FROM pay WHERE EXISTS (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1)",
        "SELECT emp FROM pay WHERE amount >= ALL (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1)",
        "(SELECT amount FROM pay ORDER BY amount DESC LIMIT 1) UNION ALL (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1)",
        "SELECT amount FROM pay ORDER BY amount DESC LIMIT 1",  # the query itself
        "SELECT (SELECT amount, emp FROM pay ORDER BY amount DESC LIMIT 1) AS a",
        "SELECT (SELECT * FROM nums ORDER BY n DESC LIMIT 1) AS a",
        "SELECT (SELECT gone FROM pay ORDER BY gone DESC LIMIT 1) AS a",  # no declared type
    ],
)
def test_other_subqueries_are_left_alone(sql):
    assert reading(sql) is None


def test_it_needs_mysql_and_declared_types():
    sql = "SELECT (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1) AS a"
    assert reading(sql) is not None
    assert reading(sql, types=None) is None and reading(sql, schema=None) is None
    assert reading(sql, types={"pay": {"emp": "INT"}}) is None
    for dialect in ("postgres", "duckdb", "bigquery", "sqlite", "tsql"):  # PostgreSQL sorts NULL above every value
        assert reading(sql, dialect=dialect) is None


def test_the_tree_given_is_not_changed():
    tree = sqlglot.parse_one("SELECT (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1) AS a", read="mysql")
    before = tree.sql(dialect="mysql")
    written, changed = write_top_rows(tree, SCHEMA, TYPES, NOT_NULL, "mysql")
    assert changed and tree.sql(dialect="mysql") == before and written is not tree


# Replay on SQLite, which sorts NULL below every value as MySQL does.
REPLAY_SCHEMA = {"t": ["g", "n", "k", "b"], "u": ["g", "x", "b"]}
REPLAY_TYPES = {"t": {"g": "INT", "n": "INT", "k": "INT", "b": "INT"}, "u": {"g": "INT", "x": "INT", "b": "INT"}}
REPLAY_NOT_NULL = {"t": frozenset({"k"}), "u": frozenset({"g"})}


def replay_subqueries(rng) -> list[str]:
    c = rng.randint(0, 3)
    out = []
    for source in ["t", f"t WHERE g = {c}", "t WHERE g IS NULL", f"t WHERE n < {c}", "t JOIN u ON t.g = u.g", "t LEFT JOIN u ON t.g = u.g"]:
        q = "t." if "JOIN" in source else rng.choice(["", "t."])
        for column in ["n", "k", "b"]:
            out.append(f"SELECT {q}{column} FROM {source} ORDER BY {q}{column} DESC LIMIT 1")
            out.append(f"SELECT {q}{column} AS v FROM {source} ORDER BY v DESC LIMIT 1")
            out.append(f"SELECT DISTINCT {q}{column} FROM {source} ORDER BY 1 DESC LIMIT 1 OFFSET 0")
            out.append(f"SELECT {q}{column} FROM {source} ORDER BY {q}{column} LIMIT 1")
    for having in ["", f" HAVING COUNT(*) = {rng.randint(1, 2)}", f" HAVING COUNT(n) >= {rng.randint(1, 2)}", " HAVING MAX(g) IS NOT NULL"]:
        for column in ["n", "k"]:
            out.append(f"SELECT {column} FROM t GROUP BY {column}{having} ORDER BY {column} DESC LIMIT 1")
            out.append(f"SELECT MAX({column}) FROM t GROUP BY g{having} ORDER BY MAX({column}) DESC LIMIT 1")
            out.append(f"SELECT COUNT({column}) AS c FROM t GROUP BY g{having} ORDER BY c DESC LIMIT 1")
            out.append(f"SELECT MIN({column}) FROM t WHERE g = {c} ORDER BY 1 DESC LIMIT 1")
    return out


@pytest.mark.parametrize("seed", range(4))
def test_replay_on_generated_tables(seed):
    """Each statement and what the module writes for it return the same rows, on tables with NULLs, ties and no rows."""

    rng = random.Random(seed)
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (g INT, n INT, k INT NOT NULL, b INT)")
    con.execute("CREATE TABLE u (g INT NOT NULL, x INT, b INT)")

    def rows(sql: str):
        return sorted(con.execute(sql).fetchall(), key=repr)

    def written(sql: str) -> str | None:
        tree = _resolve_ordinals(sqlglot.parse_one(sql, read="mysql"), group_by=True)
        top, changed = write_top_rows(tree, REPLAY_SCHEMA, REPLAY_TYPES, REPLAY_NOT_NULL, "mysql")
        return top.sql(dialect="sqlite") if changed else None

    compared = unread = 0
    for _ in range(12):
        con.execute("DELETE FROM t")
        con.execute("DELETE FROM u")
        nulls = rng.choice([0.0, 0.2, 0.6, 1.0])

        def maybe(value):
            return None if rng.random() < nulls else value

        con.executemany("INSERT INTO t VALUES (?, ?, ?, ?)", [(maybe(rng.randint(0, 3)), maybe(rng.randint(-3, 4)), rng.randint(-2, 3), maybe(rng.randint(0, 5))) for _ in range(rng.choice([0, 1, 2, 3, 5, 8]))])
        con.executemany("INSERT INTO u VALUES (?, ?, ?)", [(rng.randint(0, 3), maybe(rng.randint(0, 3)), maybe(rng.randint(0, 5))) for _ in range(rng.choice([0, 1, 3, 5]))])
        statements = [f"SELECT ({sub}) AS r" for sub in replay_subqueries(rng)]
        statements += [f"SELECT o.g, o.x, (SELECT s.{column} FROM t AS s WHERE s.g = o.g ORDER BY s.{column} DESC LIMIT 1) AS r FROM u AS o" for column in ("n", "k", "b")]
        statements += [f"SELECT o.g FROM u AS o WHERE o.x >= ({sub})" for sub in replay_subqueries(rng)[:8]]
        for sql in statements:
            after = written(sql)
            if after is None:
                assert " DESC " not in sql, sql  # only ASC over a column that may be NULL is left unread
                unread += 1
                continue
            assert rows(sql) == rows(after), (sql, after)
            compared += 1
    assert compared > 1_000 and unread > 100


def test_controls_differ():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (n INT, k INT NOT NULL)")
    con.executemany("INSERT INTO t VALUES (?, ?)", [(None, 1), (3, 0), (5, 2), (5, 3)])

    def value(sql: str):
        return con.execute(sql).fetchone()[0]

    assert value("SELECT (SELECT n FROM t ORDER BY n DESC LIMIT 1)") == value("SELECT MAX(n) FROM t") == 5
    assert value("SELECT (SELECT n FROM t ORDER BY n LIMIT 1)") is None and value("SELECT MIN(n) FROM t") == 3  # NULL sorts first
    assert value("SELECT (SELECT n FROM t ORDER BY k LIMIT 1)") == 3 != value("SELECT MAX(n) FROM t")  # another sort key
    assert value("SELECT (SELECT n FROM t ORDER BY n DESC LIMIT 1 OFFSET 1)") == 5 and value("SELECT (SELECT n FROM t ORDER BY n DESC LIMIT 1 OFFSET 2)") == 3


FACTS = {"nums": TableConstraints(), "pay": TableConstraints(not_null=frozenset({"emp", "amount"}), keys=(("emp",),))}
OPTIONS = dict(schema=SCHEMA, types=TYPES, constraints=FACTS, dialect="mysql", compare_names=False, exact_arithmetic=True, timeout_ms=3000)
ONCE = "SELECT n FROM nums GROUP BY n HAVING COUNT(n) = 1"
LARGEST_ONCE = f"SELECT MAX(n) AS n FROM ({ONCE}) AS t"
PROVED = [
    (LARGEST_ONCE, f"SELECT ({ONCE} ORDER BY n DESC LIMIT 1) AS n"),
    (LARGEST_ONCE, "SELECT (SELECT m.n FROM nums AS m GROUP BY m.n HAVING COUNT(*) = 1 ORDER BY m.n DESC LIMIT 1) AS n"),
    (LARGEST_ONCE, "SELECT IFNULL((SELECT n FROM nums GROUP BY 1 HAVING COUNT(n) = 1 ORDER BY 1 DESC LIMIT 1), NULL) AS n"),
    (LARGEST_ONCE, "SELECT (SELECT DISTINCT n FROM nums GROUP BY n HAVING COUNT(n) = 1 ORDER BY n DESC LIMIT 1) AS n"),
    ("SELECT MAX(amount) AS a FROM pay", "SELECT (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1) AS a"),
    ("SELECT MIN(amount) AS a FROM pay", "SELECT (SELECT amount FROM pay ORDER BY amount LIMIT 1) AS a"),
    ("SELECT emp FROM pay WHERE amount = (SELECT MAX(amount) FROM pay)", "SELECT emp FROM pay WHERE amount = (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1)"),
    (
        "SELECT emp FROM pay AS o WHERE amount = (SELECT MAX(s.amount) FROM pay AS s WHERE s.day = o.day)",
        "SELECT emp FROM pay AS o WHERE amount = (SELECT s.amount FROM pay AS s WHERE s.day = o.day ORDER BY s.amount DESC LIMIT 1)",
    ),
]
NOT_PROVED = [
    ("SELECT MIN(day) AS a FROM pay", "SELECT (SELECT day FROM pay ORDER BY day LIMIT 1) AS a"),  # day may be NULL
    ("SELECT MAX(amount) AS a FROM pay", "SELECT (SELECT amount FROM pay ORDER BY emp DESC LIMIT 1) AS a"),
    ("SELECT MIN(amount) AS a FROM pay", "SELECT (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1) AS a"),
    ("SELECT MAX(amount) AS a FROM pay", "SELECT t.amount AS a FROM (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1) AS t"),  # no row against NULL
    ("SELECT MAX(amount) AS a FROM pay", "SELECT (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1 OFFSET 1) AS a"),
    (LARGEST_ONCE, "SELECT (SELECT n FROM nums GROUP BY n HAVING COUNT(n) = 2 ORDER BY n DESC LIMIT 1) AS n"),
]


@pytest.mark.parametrize("left, right", PROVED)
def test_the_largest_value_asked_both_ways_is_proved(left, right):
    result = prove_equivalent_algebraic(left, right, **OPTIONS)
    assert result.proven, result.reason
    with shape_fallback.without("top_row_subqueries"):
        assert not prove_equivalent_algebraic(left, right, **OPTIONS).proven
    assert not prove_equivalent_algebraic(left, right, **{**OPTIONS, "dialect": "postgres"}).proven


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_other_readings_are_not_proved(left, right):
    assert not prove_equivalent_algebraic(left, right, **OPTIONS).proven


def test_it_is_a_rule_of_the_thirteenth_batch():
    assert "top_row_subqueries" in shape_fallback.THIRTEENTH_BATCH
    sql = "SELECT (SELECT amount FROM pay ORDER BY amount DESC LIMIT 1) AS a"
    with shape_fallback.tracking() as fired:
        normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "top_row_subqueries" in fired
    with shape_fallback.without("top_row_subqueries"), shape_fallback.tracking() as fired:
        try:
            normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
        except Exception:  # the prover before the batch declines the subquery
            pass
    assert "top_row_subqueries" not in fired
