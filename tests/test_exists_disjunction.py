"""Two EXISTS tests over the same FROM joined by OR read as one (``exists_disjunction``)."""

import pytest

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic

SCHEMA = {"t": ["a", "b"], "u": ["a", "c"], "v": ["a", "c"]}


def proven(left, right, dialect="bigquery"):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, dialect=dialect, compare_names=False).proven


PROVED = [
    (
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.c = 1) OR EXISTS (SELECT 1 FROM u WHERE u.c = 2)",
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.c = 1 OR u.c = 2)",
    ),
    # correlated, with another condition around it and different select lists
    (
        "SELECT a FROM t WHERE b < 10 AND (EXISTS (SELECT u.a FROM u WHERE u.a = t.a AND u.c > 5) OR EXISTS (SELECT 1 FROM u WHERE u.a = t.b))",
        "SELECT a FROM t WHERE b < 10 AND EXISTS (SELECT 1 FROM u WHERE (u.a = t.a AND u.c > 5) OR u.a = t.b)",
    ),
    # three members, and one without a filter
    (
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE c = 1) OR EXISTS (SELECT 1 FROM u WHERE c = 2) OR EXISTS (SELECT 1 FROM u WHERE c = 3)",
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE c IN (1, 2, 3))",
    ),
    (
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE c = 1) OR EXISTS (SELECT 1 FROM u)",
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u)",
    ),
    # De Morgan
    (
        "SELECT a FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.a = t.a AND u.c = 1) AND NOT EXISTS (SELECT 1 FROM u WHERE u.a = t.a AND u.c = 2)",
        "SELECT a FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.a = t.a AND (u.c = 1 OR u.c = 2))",
    ),
]


@pytest.mark.parametrize("left, right", PROVED)
@pytest.mark.parametrize("dialect", ["bigquery", "postgres"])
def test_existence_tests_over_one_source_merge(left, right, dialect):
    assert proven(left, right, dialect)


NOT_PROVED = [
    # AND is not OR: one row passing p and another passing q is not a row passing both
    (
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.c = 1) AND EXISTS (SELECT 1 FROM u WHERE u.c = 2)",
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.c = 1 AND u.c = 2)",
    ),
    (
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.c = 1) AND EXISTS (SELECT 1 FROM u WHERE u.c = 2)",
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.c = 1 OR u.c = 2)",
    ),
    # different sources
    (
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.c = 1) OR EXISTS (SELECT 1 FROM v WHERE v.c = 2)",
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.c = 1 OR u.c = 2)",
    ),
    # NOT EXISTS joined by OR
    (
        "SELECT a FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.c = 1) OR NOT EXISTS (SELECT 1 FROM u WHERE u.c = 2)",
        "SELECT a FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.c = 1 OR u.c = 2)",
    ),
    # a mix of EXISTS and NOT EXISTS
    (
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.c = 1) OR NOT EXISTS (SELECT 1 FROM u WHERE u.c = 2)",
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.c = 1 OR u.c = 2)",
    ),
    # a global aggregate returns a row whatever its filter keeps
    (
        "SELECT a FROM t WHERE EXISTS (SELECT COUNT(*) FROM u WHERE u.c = 1) OR EXISTS (SELECT COUNT(*) FROM u WHERE u.c = 2)",
        "SELECT a FROM t WHERE FALSE",
    ),
    # a HAVING decides more than the filter does
    (
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.c = 1 GROUP BY u.a HAVING COUNT(*) > 1) OR EXISTS (SELECT 1 FROM u WHERE u.c = 2 GROUP BY u.a HAVING COUNT(*) > 1)",
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.c = 1 OR u.c = 2 GROUP BY u.a HAVING COUNT(*) > 1)",
    ),
    # a NULL filter keeps no row
    (
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.c = 1) OR EXISTS (SELECT 1 FROM u WHERE u.c <> 1)",
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u)",
    ),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_near_misses_are_not_proved(left, right):
    assert not proven(left, right)
    assert not proven(right, left)


def test_duckdb_agrees_on_small_tables():
    duckdb = pytest.importorskip("duckdb")
    import itertools

    left, right = PROVED[1]
    for t_rows, u_rows in itertools.product(
        ([], [(1, 2)], [(1, 20), (None, 3)], [(3, 1), (3, 1)]),
        ([], [(1, 6)], [(1, 5), (2, None)], [(None, 9), (3, 7)], [(1, None)]),
    ):
        db = duckdb.connect()
        db.execute("CREATE TABLE t (a INT, b INT)")
        db.execute("CREATE TABLE u (a INT, c INT)")
        for row in t_rows:
            db.execute("INSERT INTO t VALUES (?, ?)", row)
        for row in u_rows:
            db.execute("INSERT INTO u VALUES (?, ?)", row)
        assert sorted(db.execute(left).fetchall(), key=repr) == sorted(db.execute(right).fetchall(), key=repr)
