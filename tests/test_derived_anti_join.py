"""A LEFT JOIN with a derived table that keeps the unmatched rows is NOT EXISTS (``kumo_sql_prover.derived_anti_join``)."""

import random

import pytest
import sqlglot
from sqlglot import exp

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.derived_anti_join import derived_anti_join
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: a is the key (NOT NULL); every other column may be NULL and repeat.  u: no facts at all.
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a"}), keys=(("a",),))}
HIT = "(SELECT p.a FROM t AS p JOIN u ON u.a = p.b WHERE u.e > p.c) AS d"
PAIRS = "(SELECT DISTINCT b AS k, c FROM t WHERE c > 0) AS d"
PLAIN = "(SELECT b AS k, c FROM t) AS d"


def rewritten(sql):
    tree = sqlglot.parse_one(sql, read="mysql")
    text = tree.sql(dialect="mysql")
    for select in list(tree.find_all(exp.Select)):
        derived_anti_join(select)
    after = tree.sql(dialect="mysql")
    return after if after != text else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=CONSTRAINTS, compare_names=False, dialect="mysql").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE t (a BIGINT, b BIGINT, c BIGINT, d BIGINT)")
    db.execute("CREATE TABLE u (a BIGINT, e BIGINT)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


def test_the_unmatched_rows_of_a_left_join_are_not_exists():
    assert rewritten(f"SELECT x.a FROM t AS x LEFT JOIN {HIT} ON x.a = d.a WHERE d.a IS NULL") == (
        f"SELECT x.a FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM {HIT} WHERE x.a = d.a)"
    )
    # joins before it, other WHERE parts, two tests, other clauses; DISTINCT is dropped under EXISTS
    assert rewritten(
        f"SELECT x.d, u.e FROM t AS x JOIN u ON u.a = x.b LEFT OUTER JOIN {PAIRS} ON d.k < x.c AND d.c = u.e "
        "WHERE x.d > 0 AND d.k IS NULL AND (u.e = 1 OR x.c = 2) AND d.c IS NULL ORDER BY x.d"
    ) == (
        "SELECT x.d, u.e FROM t AS x JOIN u ON u.a = x.b WHERE x.d > 0 AND (u.e = 1 OR x.c = 2) AND "
        "NOT EXISTS(SELECT 1 FROM (SELECT b AS k, c FROM t WHERE c > 0) AS d WHERE d.k < x.c AND d.c = u.e) ORDER BY x.d"
    )
    # a test on a column the ON condition does not compare goes too, once another one shows the row is padded
    assert rewritten(f"SELECT x.a FROM t AS x LEFT JOIN {PAIRS} ON x.b = d.k WHERE d.c IS NULL AND d.k IS NULL") == (
        "SELECT x.a FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM (SELECT b AS k, c FROM t WHERE c > 0) AS d WHERE x.b = d.k)"
    )
    # DISTINCT before a LIMIT picks the rows: kept
    assert rewritten("SELECT COUNT(*) FROM t AS x LEFT JOIN (SELECT DISTINCT b AS k FROM t ORDER BY k LIMIT 2) AS d ON d.k >= x.c WHERE d.k IS NULL GROUP BY x.b") == (
        "SELECT COUNT(*) FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM (SELECT DISTINCT b AS k FROM t ORDER BY k LIMIT 2) AS d WHERE d.k >= x.c) GROUP BY x.b"
    )


@pytest.mark.parametrize("sql", [
    f"SELECT x.a FROM t AS x JOIN {HIT} ON x.a = d.a WHERE d.a IS NULL",  # not an outer join
    f"SELECT x.a FROM t AS x RIGHT JOIN {HIT} ON x.a = d.a WHERE d.a IS NULL",
    f"SELECT x.a FROM t AS x LEFT JOIN {HIT} ON x.a = d.a WHERE NOT d.a IS NULL",
    f"SELECT x.a FROM t AS x LEFT JOIN {HIT} ON x.a = d.a WHERE d.a IS NULL OR x.b = 1",  # not a top-level part
    f"SELECT x.a FROM t AS x LEFT JOIN {HIT} ON x.a = d.a",
    f"SELECT x.a FROM t AS x LEFT JOIN {PLAIN} ON x.b = d.k WHERE d.c IS NULL",  # c may be NULL in a matched row
    f"SELECT x.a FROM t AS x LEFT JOIN {HIT} ON x.a <=> d.a WHERE d.a IS NULL",  # TRUE for two NULLs
    f"SELECT x.a FROM t AS x LEFT JOIN {HIT} ON x.a = d.a OR x.b = 1 WHERE d.a IS NULL",
    f"SELECT x.a FROM t AS x LEFT JOIN {HIT} ON x.a = COALESCE(d.a, 0) WHERE d.a IS NULL",
    f"SELECT x.a FROM t AS x LEFT JOIN {HIT} ON d.a IS NULL WHERE d.a IS NULL",
    # the derived table is read elsewhere
    f"SELECT x.a, d.a FROM t AS x LEFT JOIN {HIT} ON x.a = d.a WHERE d.a IS NULL",
    f"SELECT x.a FROM t AS x LEFT JOIN {HIT} ON x.a = d.a WHERE d.a IS NULL ORDER BY d.a",
    f"SELECT x.a FROM t AS x LEFT JOIN {PAIRS} ON x.b = d.k WHERE d.k IS NULL AND COALESCE(d.c, 1) = 1",
    f"SELECT x.a FROM t AS x LEFT JOIN {PAIRS} ON x.b = d.k WHERE d.k IS NULL AND EXISTS(SELECT 1 FROM u WHERE u.a = d.c)",
    f"SELECT x.a, k FROM t AS x LEFT JOIN {PAIRS} ON x.b = d.k WHERE d.k IS NULL",  # a bare name of the derived table
    f"SELECT x.a FROM t AS x LEFT JOIN {PAIRS} ON x.b = d.k WHERE k IS NULL",
    f"SELECT * FROM t AS x LEFT JOIN {HIT} ON x.a = d.a WHERE d.a IS NULL",
    # other shapes
    f"SELECT x.a FROM t AS x LEFT JOIN {HIT} ON x.a = d.a JOIN u ON u.a = x.b WHERE d.a IS NULL",  # not the last join
    f"SELECT x.a FROM u RIGHT JOIN t AS x ON u.a = x.b LEFT JOIN {HIT} ON x.a = d.a WHERE d.a IS NULL",
    f"SELECT x.a FROM t AS x LEFT JOIN {HIT} USING (a) WHERE d.a IS NULL",
    "SELECT x.a FROM t AS x LEFT JOIN u AS d ON x.b = d.a WHERE d.a IS NULL",  # a base table: read as written
    "SELECT x.a FROM t AS x LEFT JOIN (SELECT * FROM u) AS d ON x.b = d.a WHERE d.a IS NULL",  # outputs not listed
    "SELECT x.a FROM t AS x LEFT JOIN (SELECT a, e + 1 FROM u) AS d ON x.b = d.a WHERE d.a IS NULL",
    "SELECT x.a FROM t AS x LEFT JOIN (SELECT (a), e FROM u) AS d ON x.b = d.a WHERE d.a IS NULL",
    "SELECT x.a FROM t AS x LEFT JOIN (SELECT a FROM u UNION SELECT b FROM t) AS d ON x.b = d.a WHERE d.a IS NULL",
    "SELECT x.a FROM t AS x LEFT JOIN (SELECT a FROM u) AS d ON x.b = d.z WHERE d.z IS NULL",  # not an output
])
def test_other_joins_stay(sql):
    assert rewritten(sql) is None


ANTI = "SELECT x.a, x.d FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM u WHERE u.a = x.b AND u.e > x.c)"
PROOFS = [
    (f"SELECT x.a, x.d FROM t AS x LEFT JOIN {HIT} ON x.a = d.a WHERE d.a IS NULL", ANTI),
    ("SELECT y.a, y.d FROM t AS y LEFT JOIN (SELECT DISTINCT t.a AS k FROM t JOIN u ON u.a = t.b AND u.e > t.c) AS hit ON hit.k = y.a WHERE hit.k IS NULL", ANTI),
    (f"SELECT x.a, x.d FROM t AS x LEFT JOIN {HIT} ON x.a = d.a WHERE d.a IS NULL", "SELECT y.a, y.d FROM t AS y LEFT JOIN u ON u.a = y.b AND u.e > y.c WHERE u.a IS NULL"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # without the test, matched rows stay
    (f"SELECT x.a, x.d FROM t AS x LEFT JOIN {HIT} ON x.a = d.a", ANTI),
    # the tested column is NULL in matched rows too
    (
        f"SELECT x.a FROM t AS x LEFT JOIN {PLAIN} ON x.b = d.k WHERE d.c IS NULL",
        "SELECT x.a FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM t AS p WHERE p.b = x.b)",
    ),
    # another condition inside the derived table
    (f"SELECT x.a, x.d FROM t AS x LEFT JOIN {HIT} ON x.a = d.a WHERE d.a IS NULL", "SELECT x.a, x.d FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM u WHERE u.a = x.b AND u.e >= x.c)"),
    # a join on a column that is not the key: the derived row may be another row's
    (
        "SELECT x.a FROM t AS x LEFT JOIN (SELECT p.b FROM t AS p JOIN u ON u.a = p.c) AS d ON x.b = d.b WHERE d.b IS NULL",
        "SELECT x.a FROM t AS x WHERE NOT EXISTS(SELECT 1 FROM u WHERE u.a = x.c)",
    ),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


QUERIES = [
    f"SELECT x.a FROM t AS x LEFT JOIN {HIT} ON x.a = d.a WHERE d.a IS NULL",
    f"SELECT x.d, u.e FROM t AS x JOIN u ON u.a = x.b LEFT OUTER JOIN {PAIRS} ON d.k < x.c AND d.c = u.e WHERE x.d > 0 AND d.k IS NULL AND (u.e = 1 OR x.c = 2) AND d.c IS NULL",
    f"SELECT x.a FROM t AS x LEFT JOIN {PAIRS} ON x.b = d.k WHERE d.c IS NULL AND d.k IS NULL",
    "SELECT x.b, COUNT(*) FROM t AS x LEFT JOIN (SELECT DISTINCT b AS k FROM t ORDER BY k LIMIT 2) AS d ON d.k >= x.c WHERE d.k IS NULL GROUP BY x.b",
    "SELECT DISTINCT x.b, u.e FROM u LEFT JOIN t AS x ON x.b = u.a LEFT JOIN (SELECT e AS k, a AS m FROM u WHERE a > 0) AS d ON d.k = x.c AND d.m <> u.e WHERE d.m IS NULL",
    "SELECT x.c FROM t AS x, u LEFT JOIN (SELECT b AS k, MAX(c) AS top FROM t GROUP BY b) AS d ON d.k = u.e AND d.top > 1 WHERE d.top IS NULL AND u.a = x.b",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the rewritten one, as bags, on random tables with NULLs and duplicates in every column."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 2, 3])  # noqa: E731
    for _ in range(150):
        tables = {
            "t": [(value(), value(), value(), value()) for _ in range(rng.choice([0, 1, 2, 4, 7]))],
            "u": [(value(), value()) for _ in range(rng.choice([0, 1, 3, 5]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
