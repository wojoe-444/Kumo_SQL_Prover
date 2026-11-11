"""Grouping by any NOT NULL key of a table is one grouping (``kumo_sql_prover.key_group_choice``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.key_group_choice import choose_key_groups
from kumo_sql_prover.smt_equivalence import TableConstraints

# p: id and name are both keys and NOT NULL; code is a key that may be NULL; o repeats. i: no keys.
SCHEMA = {"p": ["id", "name", "code", "o"], "i": ["pid", "paid"]}
KEYS = {"p": [("id",), ("name",), ("code",)]}
NOT_NULL = {"p": frozenset({"id", "name"})}
CONSTRAINTS = {"p": TableConstraints(not_null=frozenset({"id", "name"}), keys=(("id",), ("name",), ("code",))), "i": TableConstraints()}


def rewritten(sql, keys=KEYS, not_null=NOT_NULL):
    tree = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    text = tree.sql(dialect="mysql")
    after = choose_key_groups(tree, SCHEMA, keys, not_null).sql(dialect="mysql")
    return after if after != text else None


def proved(left, right, constraints=CONSTRAINTS):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=constraints, compare_names=False, dialect="mysql").proven


def test_the_key_that_sorts_first_is_named():
    assert rewritten("SELECT p.name, SUM(i.paid) FROM p LEFT JOIN i ON p.id = i.pid GROUP BY p.name") == (
        "SELECT p.name, SUM(i.paid) FROM p LEFT JOIN i ON p.id = i.pid GROUP BY p.id"
    )
    # bare names, and a plain single-table select
    assert rewritten("SELECT name, COUNT(*) FROM p GROUP BY name") == "SELECT name, COUNT(*) FROM p GROUP BY p.id"
    assert rewritten("SELECT x.name FROM p AS x JOIN i ON x.id = i.pid GROUP BY x.name") == "SELECT x.name FROM p AS x JOIN i ON x.id = i.pid GROUP BY x.id"


@pytest.mark.parametrize("sql", [
    "SELECT id, COUNT(*) FROM p GROUP BY id",  # already the first key
    "SELECT code, COUNT(*) FROM p GROUP BY code",  # a key that may be NULL: two rows with a NULL code form one group
    "SELECT o, COUNT(*) FROM p GROUP BY o",  # not a key
    "SELECT name, o, COUNT(*) FROM p GROUP BY name, o",  # more than one grouped expression
    "SELECT name, COUNT(*) FROM p GROUP BY name + 1",
    "SELECT paid, COUNT(*) FROM i GROUP BY paid",
    "SELECT name, COUNT(*) FROM (SELECT name FROM p) AS d GROUP BY name",  # a derived table has no declared key
    "SELECT name, COUNT(*) FROM p GROUP BY ROLLUP(name)",
])
def test_other_groupings_stay(sql):
    assert rewritten(sql) is None


def test_every_key_must_be_not_null_and_declared():
    sql = "SELECT name, COUNT(*) FROM p GROUP BY name"
    assert rewritten(sql) is not None
    assert rewritten(sql, not_null={"p": frozenset({"name"})}) is None  # id may be NULL, and is not a stand-in for name
    assert rewritten(sql, not_null={"p": frozenset({"id"})}) is None
    assert rewritten(sql, keys={"p": [("id", "name")]}) is None
    assert rewritten(sql, keys=None) is None


PROOFS = [
    ("SELECT name, COUNT(*) FROM p GROUP BY name", "SELECT name, COUNT(*) FROM p GROUP BY id"),
    ("SELECT p.name, SUM(i.paid) FROM p LEFT JOIN i ON p.id = i.pid GROUP BY p.name", "SELECT p.name, SUM(i.paid) FROM p LEFT JOIN i ON p.id = i.pid GROUP BY p.id"),
    ("SELECT p.name, COUNT(i.paid) FROM p JOIN i ON p.id = i.pid GROUP BY p.name", "SELECT p.name, COUNT(i.paid) FROM p JOIN i ON p.id = i.pid GROUP BY p.id"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


def test_a_key_that_may_be_null_is_not_a_stand_in():
    # two rows with a NULL code are two groups by id, one by code
    assert not proved("SELECT COUNT(*) FROM p GROUP BY code", "SELECT COUNT(*) FROM p GROUP BY id")
    assert not proved("SELECT name, COUNT(*) FROM p GROUP BY name", "SELECT name, COUNT(*) FROM p GROUP BY o")


def test_the_rewritten_grouping_returns_the_same_rows():
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    queries = [
        "SELECT name, COUNT(*) FROM p GROUP BY name",
        "SELECT p.name, SUM(i.paid) FROM p LEFT JOIN i ON p.id = i.pid GROUP BY p.name",
        "SELECT p.name, COUNT(i.paid) FROM p JOIN i ON p.id = i.pid GROUP BY p.name",
    ]
    rng = random.Random(5)
    for _ in range(100):
        ids = rng.sample(range(0, 6), rng.choice([0, 1, 3, 5]))
        names = rng.sample(range(10, 16), len(ids))
        db = duckdb.connect()
        db.execute("CREATE TABLE p (id BIGINT, name BIGINT, code BIGINT, o BIGINT)")
        db.execute("CREATE TABLE i (pid BIGINT, paid BIGINT)")
        insert_rows(db, "p", [(a, b, rng.choice([None, 1, 2]), rng.choice([0, 1])) for a, b in zip(ids, names)])
        insert_rows(db, "i", [(rng.choice([None, 0, 1, 2, 3]), rng.choice([None, 0, 5])) for _ in range(rng.choice([0, 2, 6]))])
        for sql in queries:
            after = rewritten(sql)
            assert after is not None
            # DuckDB does not know that the name follows from the id: it is added to the grouping, the one thing the rule says is free
            after = after.replace("GROUP BY p.id", "GROUP BY p.id, p.name")
            a, b = [sorted(map(repr, rows)) for rows in run_unoptimized(db, *[sqlglot.transpile(s, read="mysql", write="duckdb")[0] for s in (sql, after)])]
            assert a == b, (sql, after)
