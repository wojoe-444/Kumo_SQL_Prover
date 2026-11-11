"""A select-list ``CASE`` arm the select's own filter rules out (``kumo_sql_prover.refuted_case_arms``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.refuted_case_arms import drop_refuted_case_arm

SCHEMA = {"t": ["k", "x", "y"], "u": ["k", "z"]}
TYPES = {table: {column: "BIGINT" for column in columns} for table, columns in SCHEMA.items()}

GUARDED = "CASE AVG(x) WHEN 0 THEN NULL ELSE SUM(y) / AVG(x) END"
RATIO = "CASE AVG(x) WHEN 0 THEN 0 ELSE SUM(y) / AVG(x) END > 1"
ARMS = "CASE WHEN AVG(x) = 0 THEN FALSE ELSE SUM(y) / AVG(x) > 1 END"
NULLIF = "COALESCE(SUM(y) / NULLIF(AVG(x), 0), 0) > 1"


def read(sql):
    out = drop_refuted_case_arm(sqlglot.parse_one(sql, read="postgres"))
    return None if out is None else out.sql(dialect="postgres")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="postgres").proven


def test_the_arm_goes():
    assert read(f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING AVG(x) <> 0") == "SELECT k, SUM(y) / AVG(x) AS r FROM t GROUP BY k HAVING AVG(x) <> 0"
    assert read(f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING 0 <> AVG(x) AND k > 1") is not None
    assert read(f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING NOT (AVG(x) = 0)") is not None
    # the filter spelled as the guarded ratio itself, on either side of the comparison
    assert read(f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING {RATIO}") == f"SELECT k, SUM(y) / AVG(x) AS r FROM t GROUP BY k HAVING {RATIO}"
    assert read(f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING 1 < CASE WHEN AVG(x) = 0 THEN 0 ELSE SUM(y) / AVG(x) END") is not None
    # the comparison moved into the arms, and the guard written with COALESCE and NULLIF
    assert read(f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING {ARMS}") == f"SELECT k, SUM(y) / AVG(x) AS r FROM t GROUP BY k HAVING {ARMS}"
    assert read(f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING CASE AVG(x) WHEN 0 THEN NULL ELSE 1 < SUM(y) / AVG(x) END") is not None
    assert read(f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING {NULLIF}") == f"SELECT k, SUM(y) / AVG(x) AS r FROM t GROUP BY k HAVING {NULLIF}"
    assert read(f"SELECT k, 1 + {GUARDED} AS r FROM t GROUP BY k HAVING 1 >= COALESCE(NULLIF(AVG(x), 0) * 2, 5)") == (
        "SELECT k, 1 + (SUM(y) / AVG(x)) AS r FROM t GROUP BY k HAVING 1 >= COALESCE(NULLIF(AVG(x), 0) * 2, 5)"
    )
    # per row, from WHERE; and only the first WHEN of several
    assert read("SELECT CASE WHEN x = 0 THEN NULL ELSE y / x END FROM t WHERE x <> 0") == "SELECT y / x FROM t WHERE x <> 0"
    assert read("SELECT CASE x WHEN 2 THEN NULL WHEN 3 THEN 7 ELSE y END FROM t WHERE x <> 2") == "SELECT CASE x WHEN 3 THEN 7 ELSE y END FROM t WHERE x <> 2"
    assert read("SELECT k, CASE WHEN k = 0 THEN NULL ELSE SUM(y) END FROM t GROUP BY k HAVING k <> 0") is not None


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k",  # no filter
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING AVG(x) <> 1",  # another number
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING AVG(y) <> 0",  # another expression
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING AVG(x) <> 0 OR k = 1",
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING AVG(x) >= 0",
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING AVG(x) IS NOT NULL",
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING CASE AVG(x) WHEN 0 THEN 2 ELSE SUM(y) / AVG(x) END > 1",  # the arm passes
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING CASE AVG(x) WHEN 0 THEN 0 ELSE SUM(y) END > 1",  # not NULL with AVG(x)
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING CASE AVG(x) WHEN 0 THEN 0 ELSE COALESCE(SUM(y) / AVG(x), 5) END > 1",
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING CASE AVG(x) WHEN 0 THEN 0 ELSE SUM(y) / AVG(x) END IS NOT NULL",
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING CASE WHEN AVG(x) = 0 THEN TRUE ELSE SUM(y) / AVG(x) > 1 END",  # the arm passes
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING CASE WHEN AVG(x) = 0 THEN FALSE ELSE k > 1 END",  # true with AVG(x) NULL
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING CASE WHEN AVG(x) = 0 THEN FALSE ELSE COALESCE(AVG(x), 5) > 1 END",
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING CASE WHEN AVG(x) = 0 THEN FALSE ELSE SUM(y) / AVG(x) IS NULL END",
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING CASE WHEN AVG(x) = 0 THEN FALSE WHEN k = 1 THEN TRUE ELSE AVG(x) > 1 END",
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING COALESCE(SUM(y) / NULLIF(AVG(x), 0), 2) > 1",  # the fallback passes
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING COALESCE(SUM(y) / NULLIF(AVG(x), 1), 0) > 1",  # another number
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING COALESCE(SUM(y) / AVG(x), 0) > 1",  # no NULLIF
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING COALESCE(SUM(y) / NULLIF(AVG(x), 0), SUM(y), 0) > 1",
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING COALESCE(ABS(NULLIF(AVG(x), 0)), 0) > 1 OR k = 1",
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING COALESCE(COALESCE(NULLIF(AVG(x), 0), 7), 0) > 1",  # not NULL with it
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY ROLLUP (k) HAVING AVG(x) <> 0",
        "SELECT k, CASE AVG(x) WHEN 0 THEN 0 ELSE SUM(y) / AVG(x) END AS r FROM t GROUP BY k HAVING AVG(x) <> 0",  # an arm with a value has a type
        "SELECT k, CASE AVG(x) WHEN 0 THEN NULL END AS r FROM t GROUP BY k HAVING AVG(x) <> 0",  # no ELSE
        "SELECT k, CASE WHEN AVG(x) = 0 OR k = 1 THEN NULL ELSE 1 END AS r FROM t GROUP BY k HAVING AVG(x) <> 0",
        "SELECT k, CASE WHEN k = 1 THEN 5 WHEN AVG(x) = 0 THEN NULL ELSE 1 END AS r FROM t GROUP BY k HAVING AVG(x) <> 0",  # not the first WHEN
        "SELECT k FROM t GROUP BY k HAVING AVG(x) <> 0 ORDER BY CASE AVG(x) WHEN 0 THEN NULL ELSE k END",  # not the select list
        "SELECT k, SUM(CASE WHEN x = 0 THEN NULL ELSE y END) FROM t GROUP BY k HAVING k <> 0",
        "SELECT SUM(CASE WHEN k = 0 THEN NULL ELSE y END) FROM t GROUP BY k HAVING k <> 0",  # inside an aggregate's argument
        "SELECT CASE WHEN x = 0 THEN NULL ELSE SUM(y) END FROM t WHERE x <> 0 GROUP BY x",  # a WHERE part, a CASE over groups
        "SELECT CASE WHEN 1 = 0 THEN NULL ELSE COUNT(*) END, COUNT(*) FROM t WHERE x <> 0",
        "SELECT CASE WHEN x = 0 THEN NULL ELSE y END, COUNT(*) OVER () FROM t WHERE y <> 0",
        "SELECT SUM(CASE WHEN x = 0 THEN NULL ELSE y END) OVER (PARTITION BY k) FROM t WHERE x <> 0",  # under a window
        "SELECT (SELECT CASE WHEN t.x = 0 THEN NULL ELSE u.z END FROM u LIMIT 1) FROM t WHERE t.x <> 0",  # a subquery's list
        "SELECT CASE WHEN ABS(x) = 0 THEN NULL ELSE y END FROM t WHERE ABS(x) <> 0",  # a function call
        "SELECT CASE WHEN (SELECT MAX(z) FROM u) = 0 THEN NULL ELSE y END FROM t WHERE (SELECT MAX(z) FROM u) <> 0",
        "SELECT y AS x, CASE WHEN x = 0 THEN NULL ELSE k END FROM t WHERE x <> 0",  # a bare name that is also an output
        "SELECT CASE WHEN x = 0 THEN NULL ELSE y END FROM t WHERE x <> 0 QUALIFY ROW_NUMBER() OVER (ORDER BY k) = 1",
    ],
)
def test_what_is_left_alone(sql):
    assert read(sql) is None, sql


def test_an_aggregate_without_group_by_keeps_its_arm():
    # one row comes back when nothing passed the WHERE, and a constant CASE is still evaluated on it
    assert read("SELECT CASE WHEN 5 - 5 = 0 THEN NULL ELSE 1 END, COUNT(*) FROM t WHERE 5 - 5 <> 0") is None
    assert read("SELECT CASE WHEN k = 0 THEN NULL ELSE 1 END FROM t WHERE k <> 0 GROUP BY k") is not None


def test_pairs_it_proves():
    grouped = "SELECT t.k, {} AS r FROM t JOIN u ON t.k = u.k GROUP BY t.k HAVING {}"
    guarded, plain = GUARDED.replace("(x)", "(t.x)").replace("(y)", "(t.y)"), "SUM(t.y) / AVG(t.x)"
    ratio = RATIO.replace("(x)", "(t.x)").replace("(y)", "(t.y)")
    assert proved(grouped.format(guarded, ratio), grouped.format(plain, f"AVG(t.x) <> 0 AND {plain} > 1"))
    assert proved(grouped.format(guarded, "AVG(t.x) <> 0"), grouped.format(plain, "AVG(t.x) <> 0"))
    arms = ARMS.replace("(x)", "(t.x)").replace("(y)", "(t.y)")
    assert proved(grouped.format(guarded, arms), grouped.format(plain, f"AVG(t.x) <> 0 AND {plain} > 1"))
    assert proved(grouped.format(guarded, arms), grouped.format(guarded, ratio))
    # the same through a derived table, as an inlined WITH leaves it
    derived = "SELECT g.k, {} AS r FROM (SELECT t.k, AVG(t.x) AS m, SUM(t.y) AS s FROM t GROUP BY t.k) AS g WHERE {}"
    assert proved(
        derived.format("CASE g.m WHEN 0 THEN NULL ELSE g.s / g.m END", "CASE g.m WHEN 0 THEN 0 ELSE g.s / g.m END > 1"),
        derived.format("g.s / g.m", "g.m <> 0 AND g.s / g.m > 1"),
    )


def test_near_misses_stay_unproven():
    grouped = "SELECT t.k, {} AS r FROM t GROUP BY t.k {}"
    guarded, plain = GUARDED.replace("(x)", "(t.x)").replace("(y)", "(t.y)"), "SUM(t.y) / AVG(t.x)"
    assert not proved(grouped.format(guarded, ""), grouped.format(plain, ""))
    assert not proved(grouped.format(guarded, "HAVING AVG(t.x) >= 0"), grouped.format(plain, "HAVING AVG(t.x) >= 0"))
    assert not proved(grouped.format(guarded, "HAVING AVG(t.x) <> 0"), grouped.format(plain, ""))
    assert not proved(grouped.format(guarded, "HAVING AVG(t.x) <> 0"), grouped.format("SUM(t.y)", "HAVING AVG(t.x) <> 0"))
    assert not proved(
        grouped.format("CASE AVG(t.x) WHEN 0 THEN 7 ELSE SUM(t.y) END", "HAVING AVG(t.y) <> 0"),
        grouped.format("SUM(t.y)", "HAVING AVG(t.y) <> 0"),
    )


def test_the_engine_agrees_on_random_tables():
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(20261117)
    queries = [
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING AVG(x) <> 0",
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING {RATIO}",
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING 1 >= CASE WHEN AVG(x) = 0 THEN 2 ELSE SUM(y) / AVG(x) END",
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING {ARMS}",
        f"SELECT k, {GUARDED} AS r FROM t GROUP BY k HAVING {NULLIF}",
        f"SELECT k, 1 + {GUARDED} AS r FROM t GROUP BY k HAVING 1 >= COALESCE(NULLIF(AVG(x), 0) * 2, 5)",
        "SELECT k, CASE WHEN x = 0 THEN NULL ELSE y / x END FROM t WHERE COALESCE(y / NULLIF(x, 0), 0) > 0",
        "SELECT k, CASE WHEN x = 0 THEN NULL ELSE y / x END FROM t WHERE x <> 0",
        "SELECT CASE x WHEN 2 THEN NULL WHEN 3 THEN 7 ELSE y END FROM t WHERE NOT x = 2",
        "SELECT k, CASE WHEN k = 0 THEN NULL ELSE SUM(y) END FROM t GROUP BY k HAVING k <> 0",
        "SELECT CASE WHEN x - y = 0 THEN NULL ELSE k END FROM t WHERE CASE WHEN x - y = 0 THEN 3 ELSE (x - y) * 2 END < 3",
    ]
    rows_seen = 0
    for trial in range(40):
        db = duckdb.connect()
        db.execute("CREATE TABLE t (k BIGINT, x DOUBLE, y DOUBLE)")
        for _ in range(rng.choice([0, 1, 5, 14]) if trial % 6 else 0):
            db.execute("INSERT INTO t VALUES (?, ?, ?)", [rng.choice([None, 0, 1, 2]), rng.choice([None, 0, 0, 1, 2, 3, -1]), rng.choice([None, 0, 1, 4, -2])])
        for sql in queries:
            after = read(sql)
            assert after is not None, sql
            key = lambda row: tuple((v is None, v) for v in row)  # noqa: E731
            before_rows = sorted(db.execute(sql).fetchall(), key=key)
            assert before_rows == sorted(db.execute(after).fetchall(), key=key), (sql, after)
            rows_seen += len(before_rows)
    assert rows_seen > 100


def test_off_on_the_second_attempt():
    sql = f"SELECT t.k, {GUARDED.replace('(x)', '(t.x)').replace('(y)', '(t.y)')} AS r FROM t GROUP BY t.k HAVING AVG(t.x) <> 0"
    assert "CASE" not in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)
    with shape_fallback.disabled():
        assert "CASE" in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)
