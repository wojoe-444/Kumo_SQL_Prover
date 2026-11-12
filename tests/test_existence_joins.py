"""Joins that only ask whether a matching row exists (``kumo_sql_prover.existence_joins``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.existence_joins import read_existence_joins, read_existence_joins_everywhere

SCHEMA = {"a": ["id", "x", "g"], "t": ["k", "d", "p"], "u": ["k", "d", "p"], "dd": ["dk", "y"]}
TYPES = {table: {column: "BIGINT" for column in columns} for table, columns in SCHEMA.items()}

MARKED = "(SELECT t.k, TRUE AS f FROM t JOIN dd ON t.d = dd.dk WHERE dd.y = 1 GROUP BY t.k) AS d"
PLAIN = "(SELECT t.k, TRUE AS f FROM t JOIN dd ON t.d = dd.dk WHERE dd.y = 1) AS d"
DISTINCT = "(SELECT DISTINCT t.k FROM t JOIN dd ON t.d = dd.dk WHERE dd.y = 1) AS d"
EXISTS_T = "EXISTS (SELECT 1 FROM t, dd WHERE t.k = a.x AND t.d = dd.dk AND dd.y = 1)"


def read(sql):
    tree = sqlglot.parse_one(sql, read="postgres")
    out = read_existence_joins(tree, SCHEMA)
    return None if out is None else out.sql(dialect="postgres")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="postgres").proven


def test_the_anti_join_becomes_not_exists():
    assert read("SELECT a.g FROM a LEFT JOIN t ON a.x = t.k WHERE t.k IS NULL AND a.g > 1") == (
        "SELECT a.g FROM a WHERE a.g > 1 AND NOT EXISTS(SELECT 1 FROM t WHERE a.x = t.k)"
    )
    # no de-duplication is needed: every matched row goes, however many there were
    assert read(f"SELECT a.g, COUNT(*) FROM a LEFT JOIN {PLAIN} ON a.x = d.k WHERE d.f IS NULL GROUP BY a.g") == (
        f"SELECT a.g, COUNT(*) FROM a WHERE NOT EXISTS(SELECT 1 FROM {PLAIN} WHERE a.x = d.k) GROUP BY a.g"
    )
    # a second test of the same source is true of every row that is left; an OR keeps its parentheses
    assert read("SELECT a.g FROM a LEFT JOIN t ON a.x = t.k AND t.p > a.g WHERE (a.g = 1 OR a.g = 2) AND t.p IS NULL AND t.d IS NULL") == (
        "SELECT a.g FROM a WHERE (a.g = 1 OR a.g = 2) AND NOT EXISTS(SELECT 1 FROM t WHERE a.x = t.k AND t.p > a.g)"
    )
    # joins after it stay
    assert read("SELECT a.g, u.p FROM a LEFT JOIN t ON a.x = t.k LEFT JOIN u ON u.k = a.id WHERE t.k IS NULL") == (
        "SELECT a.g, u.p FROM a LEFT JOIN u ON u.k = a.id WHERE NOT EXISTS(SELECT 1 FROM t WHERE a.x = t.k)"
    )


def test_the_semi_join_becomes_exists():
    assert read(f"SELECT a.g FROM a JOIN {MARKED} ON a.x = d.k WHERE a.g > 1") == (
        f"SELECT a.g FROM a WHERE a.g > 1 AND EXISTS(SELECT 1 FROM {MARKED} WHERE a.x = d.k)"
    )
    assert read(f"SELECT a.g FROM a JOIN {DISTINCT} ON d.k = a.x AND a.g < 5") == (
        f"SELECT a.g FROM a WHERE EXISTS(SELECT 1 FROM {DISTINCT} WHERE d.k = a.x AND a.g < 5)"
    )


def test_the_marker_join_becomes_exists_tests():
    assert read(f"SELECT a.g FROM a LEFT JOIN {MARKED} ON a.x = d.k WHERE d.f IS NOT NULL OR a.g > 1") == (
        f"SELECT a.g FROM a WHERE EXISTS(SELECT 1 FROM {MARKED} WHERE a.x = d.k) OR a.g > 1"
    )
    assert read(f"SELECT a.g FROM a LEFT JOIN {MARKED} ON a.x = d.k WHERE NOT d.f IS NULL OR NOT (d.k IS NOT NULL)") == (
        f"SELECT a.g FROM a WHERE EXISTS(SELECT 1 FROM {MARKED} WHERE a.x = d.k) OR NOT EXISTS(SELECT 1 FROM {MARKED} WHERE a.x = d.k)"
    )
    assert read(f"SELECT a.g, CASE WHEN d.k IS NULL THEN 0 ELSE 1 END FROM a LEFT JOIN {MARKED} ON a.x = d.k") == (
        f"SELECT a.g, CASE WHEN NOT EXISTS(SELECT 1 FROM {MARKED} WHERE a.x = d.k) THEN 0 ELSE 1 END FROM a"
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a.g FROM a LEFT JOIN t ON a.x = t.k WHERE t.p IS NULL",  # p may be NULL in a matched row
        "SELECT a.g FROM a LEFT JOIN t ON a.x = t.k WHERE t.k IS NOT NULL",  # the matched rows, repeated
        "SELECT a.g FROM a LEFT JOIN t ON a.x = t.k WHERE NOT t.k IS NULL",
        "SELECT a.g FROM a LEFT JOIN t ON a.x = t.k WHERE NOT (t.k IS NOT NULL)",  # true, but not read
        "SELECT a.g FROM a LEFT JOIN t ON a.x = t.k OR a.g = t.p WHERE t.k IS NULL",  # not a top-level comparison
        "SELECT a.g FROM a LEFT JOIN t ON a.x IS NOT DISTINCT FROM t.k WHERE t.k IS NULL",  # true of two NULLs
        "SELECT a.g FROM a LEFT JOIN t ON a.x = t.k WHERE t.k IS NULL OR a.g = 1",  # matched rows survive, repeated
        "SELECT a.g, t.p FROM a LEFT JOIN t ON a.x = t.k WHERE t.k IS NULL",  # the source is read
        "SELECT * FROM a LEFT JOIN t ON a.x = t.k WHERE t.k IS NULL",
        "SELECT a.g FROM a LEFT JOIN t ON a.x = t.k WHERE t.k IS NULL ORDER BY p",  # a bare column of the source
        "SELECT a.g FROM a LEFT JOIN t ON x = k WHERE t.k IS NULL",  # bare columns in ON
        "SELECT a.g FROM a LEFT JOIN t ON a.x = t.k RIGHT JOIN u ON u.k = a.id WHERE t.k IS NULL",  # padded afterwards
        "SELECT a.g FROM a LEFT JOIN t ON a.x = t.k FULL JOIN u ON u.k = a.id WHERE t.k IS NULL",
        "SELECT a.g FROM a LEFT JOIN t ON a.x = t.k JOIN u ON u.k = t.p WHERE t.k IS NULL",  # read by a later ON
        "SELECT a.g FROM a LEFT JOIN t ON a.x = t.k AND EXISTS (SELECT 1 FROM u WHERE u.k = t.p) WHERE t.k IS NULL",
        "SELECT a.g FROM a LEFT JOIN t ON a.x = t.k WHERE t.k IS NULL AND EXISTS (SELECT 1 FROM u AS t WHERE t.k = a.id)",  # the name again
        "SELECT a.g FROM a LEFT JOIN t ON a.x = t.k WHERE t.k IS NULL AND EXISTS (SELECT 1 FROM u WHERE u.k = t.p)",
        "SELECT a.g FROM a LEFT JOIN t USING (g) WHERE t.k IS NULL",
        "SELECT a.g FROM a NATURAL LEFT JOIN t WHERE t.k IS NULL",
        "SELECT a.g FROM a AS t LEFT JOIN t ON t.x = t.k WHERE t.k IS NULL",  # two sources of one name
        f"SELECT a.g FROM a JOIN {PLAIN} ON a.x = d.k",  # rows may repeat
        "SELECT a.g FROM a JOIN t ON a.x = t.k",
        "SELECT a.g FROM a JOIN (SELECT DISTINCT t.k, t.p FROM t) AS d ON a.x = d.k",  # p is not equated
        "SELECT a.g FROM a JOIN (SELECT t.k FROM t GROUP BY t.k, t.p) AS d ON a.x = d.k",  # a key that is not returned
        "SELECT a.g FROM a JOIN (SELECT t.k, COUNT(*) AS n FROM t GROUP BY t.k) AS d ON a.x = d.k AND d.n > 1 WHERE d.n < 9",
        "SELECT a.g FROM a JOIN (SELECT MAX(t.k) AS k FROM t) AS d ON a.x = d.k",
        "SELECT a.g FROM a JOIN (SELECT DISTINCT t.k FROM t LIMIT 3) AS d ON a.x = d.k WHERE d.k > 0",
        "SELECT a.g FROM a JOIN (SELECT t.k FROM t GROUP BY 1) AS d ON a.x = d.k WHERE d.k > 0",
        f"SELECT a.g, d.k FROM a JOIN {DISTINCT} ON a.x = d.k",
        f"SELECT a.g FROM a JOIN {DISTINCT} ON a.x < d.k",  # several rows can match
        f"SELECT a.g FROM a JOIN {DISTINCT} ON d.k = d.k + a.x",
        f"SELECT a.g FROM a LEFT JOIN {PLAIN} ON a.x = d.k WHERE d.f IS NOT NULL OR a.g > 1",  # a marker that repeats rows
        f"SELECT a.g FROM a LEFT JOIN {MARKED} ON a.x = d.k WHERE d.f OR a.g > 1",  # not an IS NULL test
        "SELECT a.g FROM a LEFT JOIN (SELECT t.k, t.p FROM t GROUP BY t.k, t.p) AS d ON a.x = d.k AND a.g = d.p WHERE COALESCE(d.p, 0) = 0",
        "SELECT a.g FROM a, LATERAL (SELECT t.k FROM t WHERE t.p = a.g) AS l LEFT JOIN u ON u.k = a.id WHERE u.k IS NULL",
    ],
)
def test_what_is_left_alone(sql):
    assert read(sql) is None, sql


def test_pairs_it_proves():
    grouped = "SELECT a.g, COUNT(*) FROM a WHERE {} GROUP BY a.g"
    both = grouped.format(f"{EXISTS_T} AND NOT {EXISTS_T.replace(' t,', ' u,').replace('t.', 'u.')}")
    u_marked = MARKED.replace(" t ", " u ").replace("t.", "u.").replace("AS d", "AS e")
    u_distinct = DISTINCT.replace(" t ", " u ").replace("t.", "u.").replace("AS d", "AS e")
    # the planner's spelling: a grouped marker for each test
    assert proved(both, f"SELECT a.g, COUNT(*) FROM a JOIN {MARKED} ON a.x = d.k LEFT JOIN {u_marked} ON a.x = e.k WHERE e.f IS NULL GROUP BY a.g")
    # the hand-written spelling: DISTINCT lists, the anti-join tested on its key
    assert proved(both, f"SELECT a.g, COUNT(*) FROM a JOIN {DISTINCT} ON a.x = d.k LEFT JOIN {u_distinct} ON a.x = e.k WHERE e.k IS NULL GROUP BY a.g")
    either = grouped.format(f"{EXISTS_T} OR a.g > 3")
    assert proved(either, f"SELECT a.g, COUNT(*) FROM a LEFT JOIN {MARKED} ON a.x = d.k WHERE d.f IS NOT NULL OR a.g > 3 GROUP BY a.g")


def test_near_misses_stay_unproven():
    counted = "SELECT a.g, COUNT(*) FROM a WHERE {} GROUP BY a.g"
    # a marker without GROUP BY repeats the row once per match
    assert not proved(counted.format(f"{EXISTS_T} OR a.g > 3"), f"SELECT a.g, COUNT(*) FROM a LEFT JOIN {PLAIN} ON a.x = d.k WHERE d.f IS NOT NULL OR a.g > 3 GROUP BY a.g")
    assert not proved(counted.format(EXISTS_T), f"SELECT a.g, COUNT(*) FROM a JOIN {PLAIN} ON a.x = d.k GROUP BY a.g")
    # the test must be on a column a match makes non-NULL
    assert not proved("SELECT a.g FROM a WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.k = a.x)", "SELECT a.g FROM a LEFT JOIN t ON a.x = t.k WHERE t.p IS NULL")
    # NOT EXISTS is not EXISTS, and another condition is another test
    assert not proved(counted.format(EXISTS_T), f"SELECT a.g, COUNT(*) FROM a LEFT JOIN {MARKED} ON a.x = d.k WHERE d.f IS NULL GROUP BY a.g")
    assert not proved(counted.format(f"NOT {EXISTS_T}"), f"SELECT a.g, COUNT(*) FROM a LEFT JOIN {MARKED} ON a.g = d.k WHERE d.f IS NULL GROUP BY a.g")
    assert not proved("SELECT a.g FROM a WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.k = a.x) LIMIT 2", "SELECT a.g FROM a LEFT JOIN t ON a.x = t.k WHERE t.k IS NULL LIMIT 3")


def test_the_engine_agrees_on_random_tables():
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(20261116)
    value = lambda: rng.choice([None, *range(0, 4)])  # noqa: E731
    queries = [
        "SELECT a.g FROM a LEFT JOIN t ON a.x = t.k WHERE t.k IS NULL AND a.g > 1",
        f"SELECT a.g, COUNT(*) FROM a LEFT JOIN {PLAIN} ON a.x = d.k WHERE d.f IS NULL GROUP BY a.g",
        "SELECT a.g FROM a LEFT JOIN t ON a.x = t.k AND t.p > a.g WHERE (a.g = 1 OR a.g = 2) AND t.p IS NULL AND t.d IS NULL",
        "SELECT a.g, u.p FROM a LEFT JOIN t ON a.x = t.k LEFT JOIN u ON u.k = a.id WHERE t.k IS NULL",
        "SELECT a.g, u.p FROM a LEFT JOIN u ON u.k = a.id LEFT JOIN t ON u.p = t.k WHERE t.k IS NULL",
        f"SELECT a.g FROM a JOIN {MARKED} ON a.x = d.k WHERE a.g > 1",
        f"SELECT a.g, u.p FROM a JOIN {DISTINCT} ON d.k = a.x AND a.g < 5 LEFT JOIN u ON u.k = a.id",
        "SELECT a.g FROM a JOIN (SELECT t.k, t.p, 7 AS c FROM t GROUP BY t.p, t.k) AS d ON a.x = d.k AND d.p = a.g AND d.c > a.id",
        f"SELECT a.g FROM a LEFT JOIN {MARKED} ON a.x = d.k WHERE d.f IS NOT NULL OR a.g > 1",
        f"SELECT a.g, CASE WHEN d.k IS NULL THEN 0 ELSE 1 END, COUNT(*) FROM a LEFT JOIN {MARKED} ON a.x = d.k GROUP BY 1, 2",
        f"SELECT a.g FROM a LEFT JOIN {MARKED} ON a.x = d.k WHERE NOT d.f IS NULL OR NOT (d.k IS NOT NULL) AND a.g > 2",
    ]
    rows_seen = 0
    for trial in range(40):
        db = duckdb.connect()
        for table, columns in SCHEMA.items():
            db.execute(f"CREATE TABLE {table} ({', '.join(c + ' BIGINT' for c in columns)})")
            rows = [tuple(value() for _ in columns) for _ in range(rng.choice([0, 1, 4, 9]) if trial % 5 else (0 if table == "t" else 6))]
            if rows:
                db.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in columns)})", rows)
        for sql in queries:
            after = read(sql)
            assert after is not None, sql
            key = lambda row: tuple((v is None, v) for v in row)  # noqa: E731
            before_rows = sorted(db.execute(sql).fetchall(), key=key)
            assert before_rows == sorted(db.execute(after).fetchall(), key=key), (sql, after)
            rows_seen += len(before_rows)
    assert rows_seen > 300


def test_every_select_is_read_innermost_first():
    sql = f"SELECT s.g FROM (SELECT a.g, a.x FROM a LEFT JOIN t ON a.x = t.k WHERE t.k IS NULL) AS s JOIN {DISTINCT} ON s.x = d.k"
    out = read_existence_joins_everywhere(sqlglot.parse_one(sql, read="postgres"), SCHEMA).sql(dialect="postgres")
    assert "JOIN" not in out.replace("JOIN dd", "") and out.count("EXISTS") == 2


def test_off_on_the_second_attempt():
    sql = "SELECT a.g FROM a LEFT JOIN (SELECT u.k, u.p FROM u) AS d ON a.x = d.k AND d.p > a.g WHERE d.k IS NULL"
    assert "LEFT JOIN" not in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES).upper()
    with shape_fallback.disabled():
        assert "NOT EXISTS" not in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES).upper()


def test_a_condition_that_calls_a_volatile_function_is_left_alone():
    # RANDOM() is drawn again for every pair of rows: the join is not one test per row of a
    assert read(f"SELECT a.g FROM a JOIN {DISTINCT} ON d.k = RANDOM()") is None
    assert read("SELECT a.g FROM a LEFT JOIN t ON a.x = t.k AND UNKNOWN_CALL(a.g) > 0 WHERE t.k IS NULL") is None
