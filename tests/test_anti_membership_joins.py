"""A NOT IN over the table's own key joined to others is NOT EXISTS over the others (``kumo_sql_prover.anti_membership_joins``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.anti_membership_joins import rewrite_anti_membership
from kumo_sql_prover.smt_equivalence import TableConstraints

# p: id is the key (NOT NULL); cid, lo, hi may be NULL; a: ts, cid may be NULL
SCHEMA = {"p": ["id", "cid", "lo", "hi"], "a": ["aid", "cid", "ts"], "b": ["bid", "cid"]}
KEYS = {"p": [("id",)], "a": [("aid",)], "b": [("bid",)]}
NOT_NULL = {"p": frozenset({"id"}), "a": frozenset({"aid"}), "b": frozenset({"bid"})}
CONSTRAINTS = {
    "p": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
    "a": TableConstraints(not_null=frozenset({"aid"}), keys=(("aid",),)),
    "b": TableConstraints(not_null=frozenset({"bid"}), keys=(("bid",),)),
}
NOT_EXISTS = "SELECT id FROM p AS q WHERE NOT EXISTS (SELECT 1 FROM a WHERE q.cid = a.cid AND a.ts >= q.lo AND a.ts <= q.hi)"
VARIANTS = [
    "SELECT id FROM p WHERE id NOT IN (SELECT x.id FROM p AS x JOIN a ON x.cid = a.cid AND a.ts >= x.lo AND a.ts <= x.hi)",
    "SELECT id FROM p WHERE id NOT IN (SELECT DISTINCT id FROM p AS x JOIN a ON x.cid = a.cid WHERE a.ts BETWEEN x.lo AND x.hi)",
    "SELECT id FROM p AS x WHERE x.id NOT IN (SELECT x.id FROM p AS x, a WHERE x.cid = a.cid AND a.ts >= x.lo AND a.ts <= x.hi)",
    "SELECT id FROM p WHERE id NOT IN (SELECT x.id FROM p AS x INNER JOIN a USING (cid) WHERE a.ts BETWEEN x.lo AND x.hi)",
    "SELECT id FROM p WHERE id NOT IN (SELECT x.id FROM p AS x LEFT JOIN a ON x.cid = a.cid WHERE a.ts >= x.lo AND a.ts <= x.hi)",
]


def rewritten(sql, keys=KEYS, not_null=NOT_NULL):
    select = sqlglot.parse_one(sql, read="mysql")
    out = rewrite_anti_membership(select, SCHEMA, keys, not_null)
    return out.sql(dialect="mysql") if out is not None else None


def proved(left, right, constraints=CONSTRAINTS):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=constraints, compare_names=False, dialect="mysql").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name, columns in {"p": "id BIGINT, cid BIGINT, lo BIGINT, hi BIGINT", "a": "aid BIGINT, cid BIGINT, ts BIGINT", "b": "bid BIGINT, cid BIGINT"}.items():
        db.execute(f"CREATE TABLE {name} ({columns})")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


def test_a_not_in_over_the_key_is_not_exists():
    assert rewritten(VARIANTS[0]) == "SELECT id FROM p WHERE NOT EXISTS(SELECT 1 FROM a WHERE p.cid = a.cid AND a.ts >= p.lo AND a.ts <= p.hi)"
    assert "NOT EXISTS" in rewritten(VARIANTS[2])
    assert "NOT EXISTS" in rewritten(VARIANTS[3])
    # the other table may be the same table under another alias
    assert rewritten("SELECT id FROM p WHERE id NOT IN (SELECT x.id FROM p AS x JOIN p AS y ON x.cid = y.cid)") == "SELECT id FROM p WHERE NOT EXISTS(SELECT 1 FROM p AS y WHERE p.cid = y.cid)"


@pytest.mark.parametrize("sql", [
    "SELECT id FROM p WHERE id NOT IN (SELECT x.id FROM p AS x JOIN a ON x.cid = a.cid) AND lo > 1",  # fine to rewrite, tested below
])
def test_other_conditions_of_the_outer_where_are_kept(sql):
    assert rewritten(sql).endswith("AND lo > 1")


@pytest.mark.parametrize("sql", [
    "SELECT id FROM p WHERE id IN (SELECT x.id FROM p AS x JOIN a ON x.cid = a.cid)",  # not NOT IN
    "SELECT id FROM p WHERE cid NOT IN (SELECT x.cid FROM p AS x JOIN a ON x.cid = a.cid)",  # not the key
    "SELECT id FROM p WHERE id NOT IN (SELECT x.cid FROM p AS x JOIN a ON x.cid = a.cid)",  # another column
    "SELECT id FROM p WHERE id NOT IN (SELECT a.aid FROM p AS x JOIN a ON x.cid = a.cid)",  # the other table's key
    "SELECT id FROM p WHERE id NOT IN (SELECT x.id FROM p AS x LEFT JOIN a ON x.cid = a.cid)",  # a LEFT JOIN keeps unmatched rows
    "SELECT id FROM p WHERE id NOT IN (SELECT x.id FROM p AS x RIGHT JOIN a ON x.cid = a.cid)",
    "SELECT id FROM p WHERE id NOT IN (SELECT x.id FROM p AS x JOIN a ON x.cid = a.cid GROUP BY x.id)",
    "SELECT id FROM p WHERE id NOT IN (SELECT x.id FROM p AS x JOIN a ON x.cid = a.cid LIMIT 1)",
    "SELECT id FROM p WHERE id NOT IN (SELECT x.id FROM p AS x JOIN a ON x.cid IN (SELECT cid FROM b))",  # a subquery inside
    "SELECT id FROM p AS o JOIN b ON o.cid = b.cid WHERE o.id NOT IN (SELECT x.id FROM p AS x JOIN a ON x.cid = a.cid)",  # a join outside
    "SELECT id FROM p WHERE id NOT IN (SELECT x.id FROM p AS x JOIN a ON x.cid = bid)",  # a column no source has
    "SELECT id FROM p WHERE id NOT IN (SELECT x.id FROM p AS x JOIN a ON cid = a.cid)",  # cid is on both
    "SELECT id FROM p WHERE id NOT IN (SELECT x.id FROM p AS x JOIN a ON x.cid = a.cid JOIN a AS a ON 1 = 1)",
    "SELECT id FROM p WHERE id NOT IN (SELECT x.id FROM p AS x)",  # nothing joined
    "SELECT id FROM p WHERE id NOT IN (SELECT x.id FROM a AS x JOIN b ON x.cid = b.cid)",  # not the outer table
    "SELECT id FROM p WHERE id NOT IN (SELECT x.id FROM p AS x JOIN a USING (cid, aid))",
    "SELECT id FROM p WHERE (id, cid) NOT IN (SELECT x.id, x.cid FROM p AS x JOIN a ON x.cid = a.cid)",
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_the_key_must_be_declared_not_null_and_single():
    assert rewritten(VARIANTS[0]) is not None
    assert rewritten(VARIANTS[0], keys=None) is None
    assert rewritten(VARIANTS[0], not_null={"p": frozenset()}) is None  # a NULL id makes NOT IN NULL
    assert rewritten(VARIANTS[0], keys={"p": [("id", "cid")]}) is None
    assert rewritten(VARIANTS[0], keys={"p": [("cid",)]}) is None


@pytest.mark.parametrize("sql", VARIANTS)
def test_variants_are_proved_equal_to_the_not_exists_form(sql):
    assert proved(sql, NOT_EXISTS)
    assert proved(NOT_EXISTS, sql)


def test_the_key_and_not_null_are_needed_for_the_proof():
    assert not proved(VARIANTS[0], NOT_EXISTS, constraints={**CONSTRAINTS, "p": TableConstraints(keys=(("id",),))})
    assert not proved(VARIANTS[0], NOT_EXISTS, constraints={**CONSTRAINTS, "p": TableConstraints(not_null=frozenset({"id"}))})


def test_a_different_condition_is_not_proved():
    other = "SELECT id FROM p AS q WHERE NOT EXISTS (SELECT 1 FROM a WHERE q.cid = a.cid AND a.ts >= q.lo AND a.ts < q.hi)"
    assert not proved(VARIANTS[0], other)
    assert not proved(other, VARIANTS[0])


@pytest.mark.parametrize("sql", VARIANTS)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """On random tables (keys unique, other columns hold NULLs and repeats) the written and the rewritten query agree."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 2, 3])  # noqa: E731
    for _ in range(60):
        tables = {
            "p": [(i, value(), value(), value()) for i in rng.sample(range(9), rng.choice([0, 1, 3, 6]))],
            "a": [(i, value(), value()) for i in rng.sample(range(9), rng.choice([0, 1, 3, 6]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
