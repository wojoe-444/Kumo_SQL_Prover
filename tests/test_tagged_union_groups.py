"""A GROUP BY on a per-branch constant of a UNION ALL groups each branch on its own (``kumo_sql_prover.tagged_union_groups``)."""

import random

import pytest
import sqlglot
from sqlglot import exp

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.tagged_union_groups import split_tagged_union_group

SCHEMA = {"t": ["a", "b", "c"], "u": ["a", "b", "c"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}


def split(sql):
    out = split_tagged_union_group(sqlglot.parse_one(sql, read="postgres"))
    return out.sql(dialect="postgres") if out is not None else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


def rows_on_duckdb(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name in SCHEMA:
        db.execute(f"CREATE TABLE {name} (a BIGINT, b BIGINT, c BIGINT)")
        insert_rows(db, name, rows.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def differ_on_duckdb(left, right, rows):
    a, b = rows_on_duckdb([left, right], rows)
    return a != b


TAGGED = "(SELECT 1 AS tag, t.a AS v FROM t UNION ALL SELECT 2 AS tag, u.b AS v FROM u) AS x"
ROWS_OF = "(SELECT 1 AS tag UNION ALL SELECT 2) AS k"


def test_a_group_by_on_the_tag_splits_by_branch():
    assert split(f"SELECT x.tag, SUM(x.v) AS s FROM {TAGGED} GROUP BY x.tag") == (
        "SELECT x.tag, SUM(x.v) AS s FROM (SELECT 1 AS tag, t.a AS v FROM t) AS x GROUP BY x.tag"
        " UNION ALL SELECT x.tag, SUM(x.v) AS s FROM (SELECT 2 AS tag, u.b AS v FROM u) AS x GROUP BY x.tag"
    )
    # a bare tag when the union is the only source; other keys, a WHERE and a HAVING go along
    assert split(f"SELECT tag, v, COUNT(*) AS n FROM {TAGGED} WHERE v > 0 GROUP BY tag, v HAVING COUNT(*) > 1") == (
        "SELECT tag, v, COUNT(*) AS n FROM (SELECT 1 AS tag, t.a AS v FROM t) AS x WHERE v > 0 GROUP BY tag, v HAVING COUNT(*) > 1"
        " UNION ALL SELECT tag, v, COUNT(*) AS n FROM (SELECT 2 AS tag, u.b AS v FROM u) AS x WHERE v > 0 GROUP BY tag, v HAVING COUNT(*) > 1"
    )
    # constant rows joined to a table
    assert split(f"SELECT k.tag, COUNT(t.b) AS n FROM t CROSS JOIN {ROWS_OF} GROUP BY k.tag") == (
        "SELECT k.tag, COUNT(t.b) AS n FROM t CROSS JOIN (SELECT 1 AS tag) AS k GROUP BY k.tag"
        " UNION ALL SELECT k.tag, COUNT(t.b) AS n FROM t CROSS JOIN (SELECT 2 AS tag) AS k GROUP BY k.tag"
    )
    assert split(f"SELECT k.tag, COUNT(t.b) AS n FROM t, {ROWS_OF} WHERE t.a > k.tag GROUP BY k.tag") is not None
    assert split(f"SELECT k.tag, COUNT(t.b) AS n FROM {ROWS_OF} JOIN t ON t.a > k.tag GROUP BY k.tag") is not None


@pytest.mark.parametrize("sql", [
    f"SELECT x.v, COUNT(*) AS n FROM {TAGGED} GROUP BY x.v",  # not grouped by the tag
    f"SELECT SUM(x.v) AS s FROM {TAGGED}",
    f"SELECT x.tag, x.v FROM {TAGGED}",
    f"SELECT x.tag, SUM(x.v) AS s FROM {TAGGED} GROUP BY x.tag ORDER BY 2 LIMIT 1",
    f"SELECT x.tag, SUM(x.v) AS s FROM {TAGGED} GROUP BY x.tag LIMIT 1",
    f"SELECT DISTINCT x.tag, SUM(x.v) AS s FROM {TAGGED} GROUP BY x.tag",
    f"SELECT x.tag, SUM(x.v) AS s, SUM(SUM(x.v)) OVER () AS w FROM {TAGGED} GROUP BY x.tag",
    f"SELECT x.tag, SUM(x.v) AS s FROM {TAGGED} GROUP BY ROLLUP (x.tag)",
    f"SELECT x.tag, SUM(x.v) AS s FROM {TAGGED} GROUP BY GROUPING SETS ((x.tag), ())",
    "SELECT x.tag, SUM(x.v) AS s FROM (SELECT 1 AS tag, t.a AS v FROM t UNION ALL SELECT 1 AS tag, u.b AS v FROM u) AS x GROUP BY x.tag",  # the same tag
    "SELECT x.tag, SUM(x.v) AS s FROM (SELECT 1 AS tag, t.a AS v FROM t UNION SELECT 2 AS tag, u.b AS v FROM u) AS x GROUP BY x.tag",  # UNION
    "SELECT x.tag, SUM(x.v) AS s FROM (SELECT 'a' AS tag, t.a AS v FROM t UNION ALL SELECT 'A' AS tag, u.b AS v FROM u) AS x GROUP BY x.tag",  # strings
    "SELECT x.tag, SUM(x.v) AS s FROM (SELECT t.c AS tag, t.a AS v FROM t UNION ALL SELECT 2 AS tag, u.b AS v FROM u) AS x GROUP BY x.tag",
    "SELECT x.tag, SUM(x.v) AS s FROM (SELECT 1 AS tag, t.a AS v FROM t UNION ALL SELECT NULL AS tag, u.b AS v FROM u) AS x GROUP BY x.tag",
    "SELECT x.tag, SUM(x.v) AS s FROM (SELECT 1 AS tag, t.a AS v FROM t UNION ALL SELECT 1.0 AS tag, u.b AS v FROM u) AS x GROUP BY x.tag",
    "SELECT x.tag, SUM(x.v) AS s FROM (SELECT 1 AS tag, t.a AS v FROM t UNION ALL SELECT 2 AS tag, u.b AS v FROM u ORDER BY 2 LIMIT 3) AS x GROUP BY x.tag",
    "SELECT x.tag, SUM(x.v) AS s FROM (SELECT 1 AS tag, t.a AS v FROM t UNION ALL SELECT 2 AS tag, u.b AS v FROM u) AS x (tag, v) GROUP BY x.tag",
    f"SELECT k.tag, COUNT(t.b) AS n FROM t LEFT JOIN {ROWS_OF} ON t.a = k.tag GROUP BY k.tag",  # an outer join pads
    f"SELECT k.tag, COUNT(t.b) AS n FROM {ROWS_OF} RIGHT JOIN t ON t.a = k.tag GROUP BY k.tag",
    f"SELECT tag, COUNT(t.b) AS n FROM t CROSS JOIN {ROWS_OF} GROUP BY tag",  # a bare tag with two sources
    f"SELECT t.a FROM t WHERE t.b IN (SELECT SUM(x.v) FROM {TAGGED} GROUP BY x.tag)",
])
def test_other_shapes_are_left_alone(sql):
    tree = sqlglot.parse_one(sql, read="postgres")
    assert all(split_tagged_union_group(select) is None for select in tree.find_all(exp.Select))


TARGETS = [
    (
        f"SELECT x.tag, SUM(x.v) AS s FROM {TAGGED} GROUP BY x.tag",
        "SELECT y.tag, SUM(y.v) AS s FROM (SELECT 1 AS tag, a AS v FROM t) AS y GROUP BY y.tag UNION ALL SELECT 2 AS tag, SUM(u.b) AS s FROM u GROUP BY 1",
    ),
    # Calcite's aggregate pushed through a union of tagged branches, against a cross join with the tags
    (
        "SELECT x.tag, SUM(x.n) AS n FROM (SELECT k.tag, COUNT(t.b) AS n FROM t, (SELECT 1 AS tag) AS k GROUP BY k.tag UNION ALL SELECT k.tag, COUNT(t.b) AS n FROM t, (SELECT 2 AS tag) AS k GROUP BY k.tag) AS x GROUP BY x.tag",
        f"SELECT k.tag, COUNT(t.b) AS n FROM t CROSS JOIN {ROWS_OF} GROUP BY k.tag",
    ),
    (
        "SELECT x.tag, SUM(x.n) AS n FROM (SELECT k.tag, COUNT(t.b) AS n FROM t, (VALUES (1)) AS k (tag) GROUP BY k.tag UNION ALL SELECT k.tag, COUNT(t.b) AS n FROM t, (VALUES (2)) AS k (tag) GROUP BY k.tag) AS x GROUP BY x.tag",
        "SELECT k.tag, COUNT(t.b) AS n FROM t CROSS JOIN (VALUES (1), (2)) AS k (tag) GROUP BY k.tag",
    ),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_tagged_union_groups_prove(left, right):
    assert proved(left, right)


ROWS = {"t": [(1, 5, 1), (1, 6, 1), (2, None, 1), (None, 4, 2)], "u": [(1, 1, 1), (4, 2, 2), (None, 2, 3)]}
NO_U = {"t": ROWS["t"]}
BY_TAG = f"SELECT x.tag, SUM(x.v) AS s FROM {TAGGED} GROUP BY x.tag"
BRANCHES = "SELECT 1 AS tag, SUM(t.a) AS s FROM t GROUP BY 1 UNION ALL SELECT 2 AS tag, SUM(u.b) AS s FROM u GROUP BY 1"
NEAR_MISSES = [
    # the same tag on both branches: one group, not two
    (BY_TAG.replace("2 AS tag", "1 AS tag"), BRANCHES.replace("2 AS tag", "1 AS tag"), ROWS),
    # grouped by another column: groups mix the branches
    (f"SELECT x.v, COUNT(*) AS n FROM {TAGGED} GROUP BY x.v", "SELECT t.a AS v, COUNT(*) AS n FROM t GROUP BY t.a UNION ALL SELECT u.b AS v, COUNT(*) AS n FROM u GROUP BY u.b", ROWS),
    # no grouping: one row over both branches
    (f"SELECT SUM(x.v) AS s FROM {TAGGED}", "SELECT SUM(t.a) AS s FROM t UNION ALL SELECT SUM(u.b) AS s FROM u", ROWS),
    # an empty branch gives no group, a global aggregate still gives a row
    (BY_TAG, "SELECT 1 AS tag, SUM(t.a) AS s FROM t UNION ALL SELECT 2 AS tag, SUM(u.b) AS s FROM u", NO_U),
    # a left join keeps the unmatched rows once, not once per constant row
    (
        f"SELECT k.tag, COUNT(*) AS n FROM t LEFT JOIN {ROWS_OF} ON t.a = k.tag GROUP BY k.tag",
        "SELECT k.tag, COUNT(*) AS n FROM t LEFT JOIN (SELECT 1 AS tag) AS k ON t.a = k.tag GROUP BY k.tag UNION ALL SELECT k.tag, COUNT(*) AS n FROM t LEFT JOIN (SELECT 2 AS tag) AS k ON t.a = k.tag GROUP BY k.tag",
        ROWS,
    ),
]


@pytest.mark.parametrize("left, right, rows", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right, rows):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, rows)


SPLIT_QUERIES = [
    BY_TAG,
    f"SELECT tag, v, COUNT(*) AS n FROM {TAGGED} WHERE v > 0 GROUP BY tag, v HAVING COUNT(*) > 1",
    f"SELECT k.tag, COUNT(t.b) AS n, MIN(t.a + k.tag) AS m FROM t CROSS JOIN {ROWS_OF} GROUP BY k.tag",
    f"SELECT k.tag, t.c, SUM(t.b) AS n FROM {ROWS_OF} JOIN t ON t.a >= k.tag GROUP BY k.tag, t.c",
    "SELECT x.tag, SUM(x.n) AS n FROM (SELECT 7 AS tag, COUNT(t.b) AS n FROM t GROUP BY t.c UNION ALL SELECT 3 AS tag, COUNT(*) AS n FROM u UNION ALL SELECT 5 AS tag, u.a AS n FROM u) AS x GROUP BY x.tag",
]


def random_rows(rng):
    def table(size):
        rows = [(rng.choice([None, 1, 2, 3]), rng.choice([None, 0, 1, 2]), rng.choice([None, 1, 2])) for _ in range(rng.randint(0, size))]
        return rows + rows[:2]

    return {"t": table(6), "u": table(4)}


@pytest.mark.parametrize("sql", SPLIT_QUERIES)
def test_the_split_gives_the_same_rows(sql):
    out = split(sql)
    normalized = normalize(sql, schema=SCHEMA, types=TYPES, dialect="postgres")
    assert out is not None and " UNION ALL SELECT" in out
    rng = random.Random(len(sql))
    for rows in [{}, ROWS, {"t": ROWS["t"]}, {"u": ROWS["u"]}] + [random_rows(rng) for _ in range(14)]:
        assert not differ_on_duckdb(sql, out, rows)
        assert not differ_on_duckdb(sql, normalized, rows)
