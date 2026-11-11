"""A query that reads no table is replaced by its rows (``kumo_sql_prover.closed_queries``)."""

from collections import Counter

import pytest
import sqlglot

pytest.importorskip("z3")

from closed_query_fuzz import Generator
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.closed_queries import evaluate, fold_closed_query
from kumo_sql_prover.smt_equivalence import SmtStatus

A = "(SELECT 1 AS a, 1 AS b UNION ALL SELECT 1, 2) AS t1"
C = "(SELECT 1 AS c, 1 AS d UNION ALL SELECT 1, 2) AS t2"
EMPTY = "(SELECT 1 AS e WHERE FALSE) AS t0"


def folded(sql, dialect="bigquery"):
    return fold_closed_query(sqlglot.parse_one(sql, read=dialect), dialect).sql(dialect=dialect)


def status(left, right, **kwargs):
    return prove_equivalent_algebraic(left, right, compare_names=False, **kwargs).status


@pytest.mark.parametrize("sql, rows", [
    (f"SELECT * FROM {A} LEFT JOIN {C} ON a = c AND b > d", "SELECT 1 AS a, 1 AS b, NULL AS c, NULL AS d UNION ALL SELECT 1, 2, 1, 1"),
    (f"SELECT * FROM {A} RIGHT JOIN {C} ON a = c AND b > d", "SELECT NULL AS a, NULL AS b, 1 AS c, 2 AS d UNION ALL SELECT 1, 2, 1, 1"),
    (f"SELECT * FROM {A} FULL JOIN {C} ON a = c AND b > d", "SELECT NULL AS a, NULL AS b, 1 AS c, 2 AS d UNION ALL SELECT 1, 1, NULL, NULL UNION ALL SELECT 1, 2, 1, 1"),
    (f"SELECT t1.b, t2.* FROM {A} JOIN {C} ON b < d", "SELECT 1 AS b, 1 AS c, 2 AS d"),
    (f"SELECT b FROM {A}, {C} WHERE b = d AND NOT b IS NULL", "SELECT 1 AS b UNION ALL SELECT 2"),
    (f"SELECT * FROM {A} CROSS JOIN {EMPTY}", "SELECT NULL AS a, NULL AS b, NULL AS e WHERE FALSE"),
    (f"SELECT * FROM {EMPTY} LEFT JOIN {A} ON TRUE", "SELECT NULL AS e, NULL AS a, NULL AS b WHERE FALSE"),
    (f"SELECT * FROM {EMPTY} FULL JOIN {A} ON TRUE", "SELECT NULL AS e, 1 AS a, 1 AS b UNION ALL SELECT NULL, 1, 2"),
    # a global aggregate returns a row over no rows; a grouped one does not
    (f"SELECT COUNT(*), SUM(e), MAX(e) FROM {EMPTY}", "SELECT 0, NULL, NULL"),
    (f"SELECT e, COUNT(*) FROM {EMPTY} GROUP BY e", "SELECT NULL AS e, NULL WHERE FALSE"),
    (f"SELECT a, COUNT(*), SUM(b), MIN(b), COUNT(DISTINCT a) + a FROM {A} GROUP BY a", "SELECT 1 AS a, 2, 3, 1, 2"),
    (f"SELECT COUNT(b), COUNT(DISTINCT b) FROM (SELECT 1 AS b UNION ALL SELECT 1 UNION ALL SELECT NULL) AS t", "SELECT 2, 1"),
    (f"SELECT DISTINCT a FROM {A}", "SELECT 1 AS a"),
    # NULLs: one group, one DISTINCT row, never equal in a join
    ("SELECT x, COUNT(*) FROM UNNEST([NULL, NULL, 3]) AS x GROUP BY x", "SELECT NULL AS x, 2 UNION ALL SELECT 3, 1"),
    ("SELECT DISTINCT x FROM UNNEST([NULL, NULL, 3]) AS x", "SELECT NULL AS x UNION ALL SELECT 3"),
    ("SELECT COUNT(*) FROM UNNEST([NULL, 3]) AS x JOIN UNNEST([NULL, 3]) AS y ON x = y", "SELECT 1"),
    # three-valued IN
    ("SELECT 1 IN (2, NULL), 1 IN (1, NULL), NULL IN (1), 1 IN UNNEST([2, NULL]), NOT 1 IN (2, 3)", "SELECT NULL, TRUE, NULL, NULL, TRUE"),
    (f"SELECT NULL IN (SELECT e FROM {EMPTY}), 1 IN (SELECT b FROM {A}), 3 IN (SELECT b FROM {A}), EXISTS(SELECT 1 FROM {EMPTY})", "SELECT FALSE, TRUE, FALSE, FALSE"),
    (f"SELECT b FROM {A} WHERE NOT b IN (SELECT 2 UNION ALL SELECT NULL)", "SELECT NULL AS b WHERE FALSE"),
    # a correlated subquery, a scalar subquery of no rows
    (f"SELECT b, (SELECT COUNT(*) FROM {C} WHERE d <= b), (SELECT d FROM {C} WHERE d > b + 5) FROM {A}", "SELECT 1 AS b, 1, NULL UNION ALL SELECT 2, 2, NULL"),
    ("SELECT COALESCE(NULL, 2), IF(NULL, 1, 2), CASE 2 WHEN 1 THEN 5 WHEN 2 THEN 6 END, CASE WHEN NULL THEN 1 END, NULLIF(1, 1), NULLIF(1, 2), -3 * (2 - 4), 2 BETWEEN 1 AND NULL, 0 BETWEEN 1 AND NULL",
     "SELECT 2, 2, 6, NULL, NULL, 1, 6, NULL, FALSE"),
    ("SELECT NULL AND FALSE, NULL OR TRUE, NULL AND TRUE, NOT NULL, NULL IS NULL, (1 = NULL) IS TRUE, (1 = 2) IS NOT FALSE, 'a' = 'a', 'a' != 'A'", "SELECT FALSE, TRUE, NULL, NULL, TRUE, FALSE, FALSE, TRUE, TRUE"),
    ("SELECT 1 AS x UNION ALL SELECT 1 UNION ALL SELECT 2", "SELECT 1 AS x UNION ALL SELECT 1 UNION ALL SELECT 2"),
    ("SELECT 2 AS x UNION DISTINCT SELECT 1 UNION DISTINCT SELECT 2", "SELECT 1 AS x UNION ALL SELECT 2"),
    ("(SELECT 1 AS x UNION ALL SELECT 1 UNION ALL SELECT NULL) INTERSECT DISTINCT (SELECT 1 UNION ALL SELECT NULL)", "SELECT NULL AS x UNION ALL SELECT 1"),
    ("(SELECT 1 AS x UNION ALL SELECT 1 UNION ALL SELECT NULL) EXCEPT DISTINCT (SELECT NULL)", "SELECT 1 AS x"),
    ("SELECT 'b' AS s, TRUE AS f UNION ALL SELECT 'a', NULL", "SELECT 'a' AS s, NULL AS f UNION ALL SELECT 'b', TRUE"),
    ("SELECT 9223372036854775807 AS big", "SELECT 9223372036854775807 AS big"),
    # decimals, to be compared only
    ("SELECT 1.0 IN (SELECT v FROM UNNEST([1, 2, 3]) AS v)", "SELECT TRUE"),
    ("SELECT 1.5 > 1, 0.1 = .10, 2 IN (1.5, 2.50), 3 < 2.999, 2.0 != 2, NULL = 1.5", "SELECT TRUE, TRUE, FALSE, FALSE, FALSE, NULL"),
    ("SELECT CAST(1 AS NUMERIC) IN (SELECT v FROM UNNEST([CAST(1 AS NUMERIC), CAST(2 AS NUMERIC)]) AS v) AS f, CAST(2 AS BIGNUMERIC) > 1.5 AS g", "SELECT TRUE AS f, TRUE AS g"),
    ("SELECT COUNT(DISTINCT x) AS n, COUNT(x) AS m FROM UNNEST([1.0, 1.00, 2.5, NULL]) AS x", "SELECT 2 AS n, 3 AS m"),
    ("SELECT x.v FROM (SELECT 1.5 AS d, 7 AS v UNION ALL SELECT 2.5, 8) AS x WHERE x.d > 2 AND CASE x.d WHEN 2.5 THEN TRUE ELSE FALSE END", "SELECT 8 AS v"),
    ("SELECT 999999999999999 = 999999999999999.0, 0.000000000000001 < 1", "SELECT TRUE, TRUE"),
    # an element of an array written in place
    ("SELECT [5, 6, 7][OFFSET(2)], [5, 6, 7][ORDINAL(1)], [5, 6][SAFE_ORDINAL(3)], [5, 6][SAFE_OFFSET(-1)], [5][OFFSET(CAST(NULL AS INT64))], [CAST(NULL AS INT64)][ORDINAL(1)]", "SELECT 7, 5, NULL, NULL, NULL, NULL"),
    ("SELECT a, [a * 10, a + 1][ORDINAL(a)] AS b FROM UNNEST([1, 2]) AS a", "SELECT 1 AS a, 10 AS b UNION ALL SELECT 2, 3"),
])
def test_rows(sql, rows):
    assert folded(sql) == rows


@pytest.mark.parametrize("sql", [
    "SELECT 1 FROM t",  # a table
    "WITH w AS (SELECT 1 AS a) SELECT a FROM w",  # a WITH table (the normalizer inlines these first)
    f"SELECT b FROM {A} ORDER BY b",
    f"SELECT b FROM {A} LIMIT 1",
    f"SELECT b FROM {A} ORDER BY b LIMIT 1 OFFSET 1",
    f"SELECT * FROM (SELECT b FROM {A} LIMIT 1) AS x",
    f"SELECT a, COUNT(*) FROM {A} GROUP BY a HAVING COUNT(*) > 1",
    f"SELECT b, ROW_NUMBER() OVER (ORDER BY b) FROM {A}",
    f"SELECT b FROM {A} QUALIFY ROW_NUMBER() OVER (ORDER BY b) = 1",
    f"SELECT * FROM {A} JOIN (SELECT 1 AS a) AS u USING (a)",
    f"SELECT * FROM {A}, {EMPTY} RIGHT JOIN {C} ON TRUE",  # BigQuery refuses it; engines that take it group it differently
    "SELECT 1 UNION ALL SELECT 2 INTERSECT DISTINCT SELECT 2",  # which operator binds first
    "SELECT 1 UNION ALL SELECT 2 UNION DISTINCT SELECT 2",
    "SELECT 1 INTERSECT ALL SELECT 1",
    "SELECT 1 EXCEPT ALL SELECT 1",
    "SELECT 1.5",  # decimals and floats
    "SELECT 1e2",
    "SELECT 4 / 2",
    "SELECT DIV(4, 2)",
    "SELECT MOD(4, 3)",
    "SELECT ABS(-1)",
    "SELECT RAND()",
    "SELECT CURRENT_DATE()",
    "SELECT 'a' < 'b'",  # string order
    "SELECT CONCAT('a', 'b')",
    "SELECT 1 = '1'",  # mixed kinds
    "SELECT TRUE = 1",
    "SELECT COALESCE(1, 'a')",
    "SELECT IF(TRUE, 1, 'a')",
    "SELECT 1 UNION ALL SELECT 'a'",
    "SELECT 1 UNION ALL SELECT TRUE",
    "SELECT DISTINCT x FROM (SELECT 1 AS x UNION ALL SELECT TRUE) AS t",
    "SELECT * FROM UNNEST([1, 'a']) AS x",
    "SELECT CAST(1 AS STRING)",  # a real conversion
    "SELECT CAST('1' AS INT64)",
    "SELECT SAFE_CAST(1 AS FLOAT64)",
    "SELECT 9223372036854775807 + 1",  # an overflow error in BigQuery
    "SELECT 9223372036854775808",
    "SELECT 3037000500 * 3037000500",
    f"SELECT SUM(x) FROM UNNEST([9223372036854775807, 1]) AS x",
    f"SELECT a, b FROM {A} GROUP BY a",  # neither grouped nor aggregated
    f"SELECT (SELECT d FROM {C}) FROM {A}",  # a scalar subquery of two rows is an error
    f"SELECT (SELECT c, d FROM {C} WHERE d = 1)",
    f"SELECT t1 FROM {A}",  # the alias as a whole row
    f"SELECT a FROM {A}, (SELECT 2 AS a) AS u",  # a name two sources have
    f"SELECT u.a FROM {A}",
    f"SELECT * EXCEPT (a) FROM {A}",
    f"SELECT AS STRUCT a FROM {A}",
    f"SELECT a FROM {A} GROUP BY ROLLUP (a)",
    f"SELECT COUNT(*) FILTER (WHERE b > 1) FROM {A}",
    f"SELECT ARRAY_AGG(b) FROM {A}",
    f"SELECT AVG(b) FROM {A}",
    f"SELECT (SELECT COUNT(t1.b) FROM {C}) FROM {A}",  # an aggregate over the enclosing select's column
    f"SELECT b AS a, COUNT(*) FROM {A} GROUP BY a",  # the alias or the column?
    "SELECT * FROM UNNEST([1, 2]) AS x WITH OFFSET AS o",
    "SELECT x IS DISTINCT FROM 1 FROM UNNEST([1, 2]) AS x",
    # decimals anywhere but in a comparison
    "SELECT 1.5",  # FLOAT64 or NUMERIC? the result would have to say
    "SELECT 0.1 + 0.2 = 0.3",  # FLOAT64 arithmetic rounds
    "SELECT 1.5 * 2 > 1",
    "SELECT -1.5 < 0",
    "SELECT MAX(x) > 1 FROM UNNEST([1.5, 2.5]) AS x",
    "SELECT SUM(x) > 1 FROM UNNEST([1.5, 2.5]) AS x",
    "SELECT 1e0 = 1",
    "SELECT 0.1234567890123456 = 0.1234567890123457",  # one FLOAT64 or two?
    "SELECT 1000000000000000 = 1000000000000000.0",
    "SELECT 9007199254740993 IN (1.5, 9007199254740992)",  # every operand is read as FLOAT64, where the two are one number
    "SELECT CASE 9007199254740993 WHEN 1.5 THEN 1 WHEN 9007199254740992 THEN 2 END",
    "SELECT COALESCE(1.5, 2) > 1",  # mixed kinds in one value
    "SELECT NULLIF(3, 1.5) * 3 = 9",  # the result is a FLOAT64
    "SELECT IF(TRUE, 1, 2.5) = 1",
    "SELECT x = 1 FROM (SELECT 1 AS x UNION ALL SELECT 1.5) AS t",
    "SELECT [1, 2.5][OFFSET(0)] = 1",
    "SELECT 1.0 = TRUE",
    "SELECT 1.0 = '1.0'",
    "SELECT 1.5 BETWEEN 1 AND 2",
    "SELECT CAST(1.5 AS NUMERIC) = 1.5",  # a FLOAT64 is rounded on the way
    "SELECT CAST(1.5 AS INT64) = 2",
    "SELECT CAST(100 AS NUMERIC(3, 1)) = 100",  # out of range
    "SELECT CAST(1 AS NUMERIC(3, 4)) = 1",
    "SELECT CAST(1 AS FLOAT64) = 1",
    "SELECT CAST('1' AS NUMERIC) = 1",
    # array elements
    "SELECT [1, 2][OFFSET(2)]",  # out of bounds is an error
    "SELECT [1, 2][ORDINAL(0)]",
    "SELECT [1, 'a'][OFFSET(0)]",
    "SELECT [1, 2][OFFSET(TRUE)]",
    "SELECT x[OFFSET(0)] FROM (SELECT [1, 2] AS x) AS t",  # an array is not a value here
    "SELECT [[1, 2][OFFSET(0)], 3] AS x",
])
def test_declined(sql):
    assert evaluate(sqlglot.parse_one(sql, read="bigquery")) is None


def test_too_many_rows_are_declined():
    ten = "(" + " UNION ALL ".join(f"SELECT {i} AS n" for i in range(10)) + ")"
    assert evaluate(sqlglot.parse_one(f"SELECT a.n FROM {ten} AS a, {ten} AS b", read="bigquery")) is None  # 100 output rows
    assert evaluate(sqlglot.parse_one(f"SELECT COUNT(*) FROM {ten} AS a, {ten} AS b, {ten} AS c, {ten} AS d", read="bigquery")) is None


def test_other_dialects_are_left_alone():
    sql = "SELECT 1 + 1"
    assert fold_closed_query(sqlglot.parse_one(sql, read="postgres"), "postgres").sql(dialect="postgres") == sql


@pytest.mark.parametrize("left, right", [
    (f"SELECT * FROM {A} LEFT JOIN {C} ON a = c AND b > d", "SELECT 1, 2, 1, 1 UNION ALL SELECT 1, 1, NULL, NULL"),
    (f"SELECT * FROM {A} FULL JOIN {C} ON a = c AND b < d", "SELECT * FROM (SELECT 1, 1, 1, 2 UNION ALL SELECT NULL, NULL, 1, 1 UNION ALL SELECT 1, 2, NULL, NULL) AS v"),
    (f"SELECT COUNT(*) FROM {A} JOIN {C} ON b > 2", "SELECT 0"),
    (f"SELECT a FROM {A} JOIN {C} ON b > 1", "SELECT 1 UNION ALL SELECT 1"),
    (f"SELECT * FROM {EMPTY} LEFT JOIN {A} ON TRUE", "SELECT 1, 2, 3 FROM (SELECT 1) AS one WHERE FALSE"),
    # inline rows through both rules
    ("SELECT * FROM UNNEST([STRUCT(1 AS a), STRUCT(2 AS a)]) AS t1 LEFT JOIN UNNEST([STRUCT(10 AS b), STRUCT(11 AS b)]) AS t2 ON a > b",
     "SELECT * FROM UNNEST([STRUCT(1 AS x, NULL AS y), STRUCT(2 AS x, NULL AS y)]) AS v"),
])
def test_proved(left, right):
    assert status(left, right) is SmtStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize("left, right", [
    # the padded row of the LEFT JOIN is missing
    (f"SELECT * FROM {A} LEFT JOIN {C} ON a = c AND b > d", "SELECT 1, 2, 1, 1"),
    # two matches, not one
    (f"SELECT a FROM {A} JOIN {C} ON b > 1", "SELECT 1"),
    (f"SELECT DISTINCT a FROM {A} JOIN {C} ON b > 1", "SELECT 1 UNION ALL SELECT 1"),
    # NULL is not 0
    (f"SELECT d FROM {A} LEFT JOIN {C} ON b > 5", "SELECT 0 UNION ALL SELECT 0"),
    (f"SELECT SUM(e) FROM {EMPTY}", "SELECT 0"),
    # a global COUNT over no rows is one row; the grouped one is none
    (f"SELECT COUNT(*) FROM {EMPTY}", f"SELECT COUNT(*) FROM {EMPTY} GROUP BY e"),
    (f"SELECT e FROM {EMPTY}", "SELECT NULL"),
    # NOT IN over a NULL keeps nothing
    (f"SELECT b FROM {A} WHERE NOT b IN (SELECT 2 UNION ALL SELECT NULL)", "SELECT 1"),
    # TRUE is not 1, and '1' is not 1
    ("SELECT 1 = 1", "SELECT 1"),
    ("SELECT 1", "SELECT '1'"),
    # the first row, whichever it is
    (f"SELECT b FROM {A} LIMIT 1", "SELECT 1"),
    # INNER, LEFT and FULL differ over an empty side
    (f"SELECT a FROM {A} JOIN {EMPTY} ON TRUE", f"SELECT a FROM {A} LEFT JOIN {EMPTY} ON TRUE"),
    (f"SELECT e FROM {A} RIGHT JOIN {EMPTY} ON TRUE", f"SELECT e FROM {A} LEFT JOIN {EMPTY} ON TRUE"),
])
def test_not_proved(left, right):
    assert status(left, right) is not SmtStatus.PROVEN_EQUIVALENT


def test_a_different_result_is_refuted():
    assert status(f"SELECT a FROM {A} JOIN {C} ON b > 1", "SELECT 1") is SmtStatus.NOT_EQUIVALENT


def test_a_query_over_a_table_is_not_evaluated():
    schema, types = {"t": ["k"]}, {"t": {"k": "INT64"}}
    assert folded("SELECT 1 FROM t WHERE FALSE") == "SELECT 1 FROM t WHERE FALSE"
    assert status(f"SELECT t.k FROM t JOIN {A} ON t.k = t1.b", "SELECT 1 UNION ALL SELECT 2", schema=schema, types=types) is not SmtStatus.PROVEN_EQUIVALENT


def test_the_interpreter_agrees_with_duckdb_on_random_queries():
    """Seeded; ``tests/closed_query_fuzz.py`` writes the queries. A longer run used seeds 0 to 15,000 (2026-11-11)."""

    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import run_unoptimized

    db = duckdb.connect()
    agreed = 0
    for seed in range(400):
        sql = Generator(seed).query()
        ours = evaluate(sqlglot.parse_one(sql, read="bigquery"))
        if ours is None:
            assert " CROSS JOIN " in sql or ", (" in sql, sql  # only a RIGHT or FULL join after a comma or CROSS JOIN is declined
            continue
        text = sqlglot.transpile(sql, read="bigquery", write="duckdb")[0]
        try:
            theirs = db.execute(text).fetchall()
        except duckdb.NotImplementedException:
            continue  # DuckDB has no outer join on a subquery condition
        if Counter(ours[1]) != Counter(theirs):
            theirs = run_unoptimized(db, text)[0]  # DuckDB's optimizer has returned wrong rows before
        assert Counter(ours[1]) == Counter(theirs), sql
        assert sorted(map(repr, ours[1])) == sorted(map(repr, theirs)), sql  # TRUE is not 1
        agreed += 1
    assert agreed > 330


def test_decimal_comparisons_agree_with_float64_on_duckdb():
    """A decimal of up to 15 digits is compared as written; DuckDB compares the same numbers as DOUBLE."""

    duckdb = pytest.importorskip("duckdb")
    import random
    import re

    pool = ["0", "1", "2", "3", "1.0", "1.5", "2.50", "0.1", ".10", "0.3", "2.999", "999999999999999", "999999999999999.0", "0.000000000000001", "123456.789012345", "123456.789012344", "NULL"]
    rng = random.Random(3)
    db = duckdb.connect()
    for _ in range(300):
        a, b, c, d = (rng.choice(pool) for _ in range(4))
        operator = rng.choice(["=", "!=", "<", "<=", ">", ">="])
        if "." not in a + b + c + d:
            continue
        sql = f"SELECT {a} {operator} {b}, {a} IN ({b}, {c}, {d}), {c} NOT IN ({d}, {a})"
        ours = evaluate(sqlglot.parse_one(sql, read="bigquery"))
        assert ours is not None, sql
        as_double = re.sub(r"[0-9]*\.[0-9]+", lambda m: f"CAST({m.group(0)} AS DOUBLE)", sql)
        assert list(ours[1][0]) == list(db.execute(as_double).fetchall()[0]), sql
    db.close()
