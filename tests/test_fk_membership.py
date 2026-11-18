"""Membership of a foreign-key column in its parent's key, read as "not NULL" (``kumo_sql_prover.fk_membership``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.fk_membership import read_fk_membership
from kumo_sql_prover.smt_equivalence import TableConstraints

# c.f references p.pk and may be NULL; c.g references nothing; p.pk is not declared a key; q is another table
SCHEMA = {"c": ["id", "f", "g"], "p": ["pk", "name"], "q": ["pk"]}
FOREIGN = {"c": [(("f",), "p", ("pk",))]}
CONSTRAINTS = {"c": TableConstraints(foreign_keys=((("f",), "p", ("pk",)),))}
STRICT = {"c": TableConstraints(not_null=frozenset({"f"}), foreign_keys=((("f",), "p", ("pk",)),))}


def rewritten(sql, foreign=FOREIGN, schema=SCHEMA):
    before = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    text = before.sql(dialect="mysql")
    after = read_fk_membership(before, schema, foreign).sql(dialect="mysql")
    return after if after != text else None


def proved(left, right, constraints=CONSTRAINTS):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=constraints, compare_names=False, dialect="mysql").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE c (id BIGINT, f BIGINT, g BIGINT)")
    db.execute("CREATE TABLE p (pk BIGINT, name BIGINT)")
    db.execute("CREATE TABLE q (pk BIGINT)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


def test_membership_in_the_parent_becomes_a_null_test():
    assert rewritten("SELECT c.id FROM c WHERE c.f IN (SELECT pk FROM p)") == "SELECT c.id FROM c WHERE NOT c.f IS NULL"
    assert rewritten("SELECT id FROM c WHERE f IN (SELECT DISTINCT x.pk FROM p AS x ORDER BY x.pk)") == "SELECT id FROM c WHERE NOT f IS NULL"
    assert rewritten("SELECT y.id FROM c AS y WHERE y.g > 0 AND (y.f) IN (SELECT p.pk AS k FROM p)") == "SELECT y.id FROM c AS y WHERE y.g > 0 AND NOT y.f IS NULL"
    # SELECT * of a parent with one column
    assert rewritten("SELECT id FROM c WHERE f IN (SELECT * FROM q)", foreign={"c": [(("f",), "q", ("pk",))]}) == "SELECT id FROM c WHERE NOT f IS NULL"
    # among other sources, also on the padded side of an outer join; in a grouped select; in a subquery
    assert rewritten("SELECT q.pk FROM q LEFT JOIN c ON c.id = q.pk WHERE f IN (SELECT pk FROM p)") == "SELECT q.pk FROM q LEFT JOIN c ON c.id = q.pk WHERE NOT f IS NULL"
    assert rewritten("SELECT g, COUNT(*) FROM c WHERE f IN (SELECT pk FROM p) GROUP BY g") == "SELECT g, COUNT(*) FROM c WHERE NOT f IS NULL GROUP BY g"
    assert rewritten("SELECT pk FROM q WHERE EXISTS (SELECT 1 FROM c WHERE c.id = q.pk AND c.f IN (SELECT pk FROM p))") == (
        "SELECT pk FROM q WHERE EXISTS(SELECT 1 FROM c WHERE c.id = q.pk AND NOT c.f IS NULL)"
    )


@pytest.mark.parametrize("sql", [
    "SELECT id FROM c WHERE f NOT IN (SELECT pk FROM p)",  # NULL is not FALSE
    "SELECT id FROM c WHERE NOT f IN (SELECT pk FROM p)",
    "SELECT id FROM c WHERE f IN (SELECT pk FROM p) OR g = 1",
    "SELECT id, f IN (SELECT pk FROM p) FROM c",
    "SELECT g FROM c GROUP BY g HAVING MAX(f) IN (SELECT pk FROM p)",
    "SELECT id FROM c WHERE g IN (SELECT pk FROM p)",  # g references nothing
    "SELECT id FROM c WHERE f IN (SELECT name FROM p)",  # not the referenced column
    "SELECT id FROM c WHERE f IN (SELECT pk FROM q)",  # not the parent
    "SELECT id FROM c WHERE f + 0 IN (SELECT pk FROM p)",
    "SELECT id FROM c WHERE f IN (SELECT pk + 0 FROM p)",
    "SELECT id FROM c WHERE f IN (SELECT pk FROM p WHERE name = 1)",  # not every parent row
    "SELECT id FROM c WHERE f IN (SELECT pk FROM p LIMIT 1)",
    "SELECT id FROM c WHERE f IN (SELECT pk FROM p GROUP BY pk HAVING COUNT(*) > 1)",
    "SELECT id FROM c WHERE f IN (SELECT p.pk FROM p JOIN q ON q.pk = p.pk)",
    "SELECT id FROM c WHERE f IN (SELECT * FROM p)",  # two columns
    "SELECT id FROM c WHERE f IN (1, 2)",
    "SELECT id FROM c WHERE (f, g) IN (SELECT pk, name FROM p)",
    "SELECT d.id FROM (SELECT id, g AS f FROM c) AS d WHERE d.f IN (SELECT pk FROM p)",  # a derived table's column
    "SELECT d.id FROM (SELECT id, g AS f FROM c) AS d WHERE f IN (SELECT pk FROM p)",
    "SELECT pk FROM q WHERE EXISTS (SELECT 1 FROM p AS x WHERE c.f IN (SELECT pk FROM p))",  # no source of that select
    "SELECT pk FROM p WHERE pk IN (SELECT f FROM c)",  # the other direction
    "SELECT x.id FROM c AS x JOIN c AS y ON x.id = y.g WHERE f IN (SELECT pk FROM p)",  # f is either side's
])
def test_other_tests_stay(sql):
    assert rewritten(sql) is None


def test_the_foreign_key_has_to_be_declared():
    sql = "SELECT id FROM c WHERE f IN (SELECT pk FROM p)"
    assert rewritten(sql) is not None
    assert rewritten(sql, foreign=None) is None
    assert rewritten(sql, foreign={"c": [(("g",), "p", ("pk",))]}) is None
    assert rewritten(sql, foreign={"c": [(("f",), "p", ("name",))]}) is None
    assert rewritten(sql, foreign={"c": [(("f",), "q", ("pk",))]}) is None
    assert rewritten(sql, foreign={"c": [(("f", "g"), "p", ("pk", "name"))]}) is None
    assert rewritten(sql, foreign={"q": [(("pk",), "p", ("pk",))]}) is None
    assert rewritten(sql, schema=None) is None


PROOFS = [
    ("SELECT c.id FROM c WHERE c.f IN (SELECT pk FROM p)", "SELECT id FROM c WHERE f IS NOT NULL"),
    ("SELECT g, COUNT(*) FROM c WHERE f IN (SELECT DISTINCT pk FROM p) GROUP BY g", "SELECT g, COUNT(f) FROM c WHERE f IS NOT NULL GROUP BY g"),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


def test_with_a_not_null_column_the_test_is_gone():
    left, right = "SELECT c.id FROM c WHERE c.f IN (SELECT pk FROM p)", "SELECT id FROM c"
    assert proved(left, right, STRICT)
    assert not proved(left, right)  # f may be NULL


NOT_PROVED = [
    # no foreign key: a value of g need not exist in p
    ("SELECT id FROM c WHERE g IN (SELECT pk FROM p)", "SELECT id FROM c WHERE g IS NOT NULL"),
    # a NULL f is dropped by NOT IN and kept by IS NULL
    ("SELECT id FROM c WHERE f NOT IN (SELECT pk FROM p)", "SELECT id FROM c WHERE f IS NULL"),
    # only some parent rows
    ("SELECT id FROM c WHERE f IN (SELECT pk FROM p WHERE name = 1)", "SELECT id FROM c WHERE f IS NOT NULL"),
    ("SELECT id FROM c WHERE f IN (SELECT pk FROM q)", "SELECT id FROM c WHERE f IS NOT NULL"),
    ("SELECT id FROM c WHERE f IN (SELECT name FROM p)", "SELECT id FROM c WHERE f IS NOT NULL"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_without_the_declaration_nothing_is_proved():
    left, right = PROOFS[0]
    assert not proved(left, right, constraints=None)


def test_the_near_misses_really_differ():
    left, right = NOT_PROVED[1]
    assert run({"c": [(1, None, 0)], "p": [(5, 0)]}, left, right) == [[], ["(1,)"]]
    left, right = NOT_PROVED[2]
    assert run({"c": [(1, 5, 0)], "p": [(5, 0)]}, left, right) == [[], ["(1,)"]]


QUERIES = [
    "SELECT c.id FROM c WHERE c.f IN (SELECT pk FROM p)",
    "SELECT id FROM c WHERE f IN (SELECT DISTINCT x.pk FROM p AS x ORDER BY x.pk)",
    "SELECT y.id FROM c AS y WHERE y.g > 0 AND (y.f) IN (SELECT p.pk AS k FROM p)",
    "SELECT q.pk, c.id FROM q LEFT JOIN c ON c.id = q.pk WHERE f IN (SELECT pk FROM p)",
    "SELECT g, COUNT(*) FROM c WHERE f IN (SELECT pk FROM p) GROUP BY g",
    "SELECT pk FROM q WHERE EXISTS (SELECT 1 FROM c WHERE c.id = q.pk AND c.f IN (SELECT pk FROM p))",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the rewritten one on random tables that keep the foreign key: every f that is not NULL
    is some p.pk. p.pk may repeat or be NULL, p may be empty (then every f is NULL), c.f may be NULL."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    for _ in range(150):
        parents = [(rng.choice([None, 1, 2, 2, 3]), rng.choice([0, 1])) for _ in range(rng.choice([0, 1, 2, 4]))]
        keys = [pk for pk, _ in parents if pk is not None]
        tables = {
            "p": parents,
            "c": [(rng.choice([1, 2, 3]), rng.choice(keys + [None]), rng.choice([None, 0, 1, 7])) for _ in range(rng.choice([0, 1, 2, 4, 6]))],
            "q": [(rng.choice([None, 1, 2, 3]),) for _ in range(rng.choice([0, 1, 3]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
