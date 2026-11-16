"""``DISTINCT`` or ``GROUP BY`` over the keys of a grouped derived table removes nothing (``kumo_sql_prover.grouped_source_dedup``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.grouped_source_dedup import drop_dedup_over_grouped, unique_key_outputs
from kumo_sql_prover.smt_equivalence import TableConstraints

# t: (a, b) is a key and both are NOT NULL; c and d may be NULL and repeat. u has no key: its rows repeat and hold NULLs.
SCHEMA = {"t": ["a", "b", "c", "d"], "u": ["a", "e"]}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a", "b"}), keys=(("a", "b"),))}


def rewritten(sql, schema=SCHEMA):
    tree = sqlglot.parse_one(sql, read="mysql")
    text = tree.sql(dialect="mysql")
    drop_dedup_over_grouped(tree, schema)
    after = tree.sql(dialect="mysql")
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


G = "(SELECT c AS k, COUNT(*) AS n FROM t GROUP BY c) AS g"
G2 = "(SELECT c, d, MAX(b) AS m FROM t GROUP BY c, d HAVING MAX(b) > 0) AS g"


def test_distinct_over_every_key_is_dropped():
    assert rewritten(f"SELECT DISTINCT g.k, g.n FROM {G} WHERE g.n > 1") == f"SELECT g.k, g.n FROM {G} WHERE g.n > 1"
    assert rewritten("SELECT DISTINCT n, k FROM (SELECT x.c AS k, COUNT(*) AS n FROM t AS x GROUP BY c) AS g") == "SELECT n, k FROM (SELECT x.c AS k, COUNT(*) AS n FROM t AS x GROUP BY c) AS g"
    assert rewritten(f"SELECT DISTINCT g.c, g.d, g.m + 1 FROM {G2} ORDER BY g.c") == f"SELECT g.c, g.d, g.m + 1 FROM {G2} ORDER BY g.c"


def test_group_by_over_every_key_is_dropped():
    assert rewritten(f"SELECT g.k, g.n FROM {G} GROUP BY g.k") == f"SELECT g.k, g.n FROM {G}"
    assert rewritten(f"SELECT g.m FROM {G2} WHERE g.m > 0 GROUP BY g.d, g.c, g.m") == f"SELECT g.m FROM {G2} WHERE g.m > 0"
    # the grouping goes first; the DISTINCT is then over rows that are all different
    assert rewritten(f"SELECT DISTINCT g.k, g.n FROM {G} GROUP BY g.k") == f"SELECT DISTINCT g.k, g.n FROM {G}"


@pytest.mark.parametrize("sql", [
    f"SELECT DISTINCT g.n FROM {G}",  # two groups may have the same count
    f"SELECT DISTINCT g.c, g.m FROM {G2}",  # one of the two keys is missing
    f"SELECT DISTINCT g.k + 0, g.n FROM {G}",  # the key only inside an expression
    "SELECT DISTINCT g.n FROM (SELECT COUNT(*) AS n FROM t GROUP BY c) AS g",  # the derived table does not output its key
    "SELECT DISTINCT g.k, g.n FROM (SELECT c + 0 AS k, COUNT(*) AS n FROM t GROUP BY c) AS g",
    "SELECT DISTINCT g.k, g.n FROM (SELECT c AS k, COUNT(*) AS n FROM t GROUP BY c WITH ROLLUP) AS g",  # the total row is a second NULL key
    "SELECT DISTINCT g.k, g.n FROM (SELECT c AS k, b AS n FROM t) AS g",  # not grouped
    "SELECT DISTINCT g.k, g.n FROM (SELECT c AS k, COUNT(*) AS n FROM t GROUP BY c LIMIT 3) AS g",
    f"SELECT DISTINCT g.k, u.e FROM {G} JOIN u ON u.a = g.k",  # the join repeats rows
    f"SELECT g.k, COUNT(*) FROM {G} GROUP BY g.k",  # an aggregate over the single-row groups
    f"SELECT g.k FROM {G} GROUP BY g.k HAVING g.k > 1",
    f"SELECT g.n FROM {G} GROUP BY g.n",  # groups by the count
    f"SELECT g.k, g.n FROM {G}",  # nothing to drop
    f"SELECT DISTINCT g.k, g.n FROM {G} LIMIT 1",
    "SELECT DISTINCT g.c, g.n FROM (SELECT d AS c, COUNT(*) AS n FROM t GROUP BY c) AS g",  # GROUP BY c: the column or the alias
    f"SELECT DISTINCT g.k, ROW_NUMBER() OVER (ORDER BY g.k) FROM {G}",
    "SELECT DISTINCT a, e FROM u",
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_which_outputs_carry_the_keys():
    def carried(sql, schema=SCHEMA):
        return unique_key_outputs(sqlglot.parse_one(f"SELECT 1 FROM {sql}", read="mysql").args["from_"].this, schema)

    assert carried(G) == [{"k"}]
    assert carried(G2) == [{"c"}, {"d"}]
    assert carried("(SELECT x.c AS k, c AS k2, x.d, COUNT(*) AS n FROM t AS x GROUP BY c, x.d) AS g") == [{"k", "k2"}, {"d"}]
    assert carried("(SELECT x.c AS k, COUNT(*) AS n FROM t AS x GROUP BY c) AS g", schema=None) is None  # x.c and a bare c: one column only by the schema
    assert carried("(SELECT c AS k, COUNT(*) AS n FROM t GROUP BY c) AS g", schema=None) == [{"k"}]
    assert carried("(SELECT c AS k, COUNT(*) AS n FROM t GROUP BY c, d) AS g") is None
    assert carried("(SELECT x.c, y.c AS c2 FROM t AS x JOIN t AS y ON x.a = y.a GROUP BY x.c) AS g") == [{"c"}]
    assert carried("t") is None


PROOFS = [
    (f"SELECT DISTINCT g.k, g.n FROM {G} WHERE g.n > 1", "SELECT c, COUNT(*) FROM t GROUP BY c HAVING COUNT(*) > 1"),
    (f"SELECT g.k, g.n FROM {G} GROUP BY g.k", "SELECT c, COUNT(*) FROM t GROUP BY c"),
    (f"SELECT DISTINCT g.m, g.c, g.d FROM {G2}", "SELECT MAX(b), c, d FROM t GROUP BY c, d HAVING MAX(b) > 0"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # two groups may have the same count
    (f"SELECT DISTINCT g.n FROM {G}", "SELECT COUNT(*) FROM t GROUP BY c"),
    # one key of two: the pairs (c, m) may repeat across d
    (f"SELECT DISTINCT g.c, g.m FROM {G2}", "SELECT c, MAX(b) FROM t GROUP BY c, d HAVING MAX(b) > 0"),
    # the join repeats the rows of the derived table
    (f"SELECT DISTINCT g.k, g.n FROM {G} JOIN u ON u.a = g.k", f"SELECT g.k, g.n FROM {G} JOIN u ON u.a = g.k"),
    # grouping by the count merges groups
    (f"SELECT g.n FROM {G} GROUP BY g.n", "SELECT COUNT(*) FROM t GROUP BY c"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


CASES = [
    f"SELECT DISTINCT g.k, g.n FROM {G} WHERE g.n > 1",
    "SELECT DISTINCT n, k FROM (SELECT x.c AS k, COUNT(*) AS n FROM t AS x GROUP BY c) AS g",
    f"SELECT DISTINCT g.c, g.d, g.m + 1 FROM {G2}",
    f"SELECT g.k, g.n FROM {G} GROUP BY g.k",
    f"SELECT g.m FROM {G2} WHERE g.m > 1 GROUP BY g.d, g.c, g.m",
    "SELECT DISTINCT g.k2, g.d FROM (SELECT x.c AS k, c AS k2, x.d, SUM(b) AS n FROM t AS x WHERE x.b > 0 GROUP BY c, x.d) AS g WHERE g.n > 2 OR g.k IS NULL",
]


@pytest.mark.parametrize("sql", CASES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """With and without the DISTINCT or GROUP BY on random tables: the keys c and d hold NULLs and repeat."""

    after = rewritten(sql)
    assert after is not None
    # DuckDB has no MySQL-style column outside the GROUP BY; ANY_VALUE picks from the group's one row
    sql = sql.replace("SELECT g.k, g.n FROM", "SELECT g.k, ANY_VALUE(g.n) FROM") if "GROUP BY g.k" in sql else sql
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 2, 3])  # noqa: E731
    for _ in range(150):
        keys = {(rng.choice([1, 1, 2, 3]), rng.choice([0, 1, 2, 3, 4])) for _ in range(rng.choice([0, 1, 2, 4, 7]))}
        tables = {"t": [(a, b, value(), value()) for a, b in sorted(keys)]}
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
