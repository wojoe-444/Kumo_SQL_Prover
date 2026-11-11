"""Inside a membership test a nested ``IN`` or ``EXISTS`` is read as a join (``kumo_sql_prover.membership_joins``)."""

import random

import pytest
import sqlglot
from sqlglot import exp

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.membership_joins import join_nested_membership

SCHEMA = {"s": ["k", "name"], "ps": ["sk", "pk", "n"], "p": ["k", "f"], "l": ["pk", "sk", "q"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}


def rewritten(sql, dialect="postgres"):
    """``sql`` with the rule applied once to the first select it accepts, or ``None``."""

    tree = sqlglot.parse_one(sql, read=dialect)
    for select in list(tree.find_all(exp.Select)):
        out = join_nested_membership(select, SCHEMA)
        if out is not None:
            if select is tree:
                return out.sql(dialect=dialect)
            select.replace(out)
            return tree.sql(dialect=dialect)
    return None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


def rows_on_duckdb(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name, columns in SCHEMA.items():
        db.execute(f"CREATE TABLE {name} ({', '.join(f'{c} BIGINT' for c in columns)})")
        insert_rows(db, name, rows.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def differ_on_duckdb(left, right, rows):
    first, second = rows_on_duckdb([left, right], rows)
    return first != second


def test_a_nested_in_becomes_a_join():
    assert (
        rewritten("SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p WHERE p.f = 1))")
        == "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, p WHERE ps.pk = p.k AND p.f = 1)"
    )
    # bare columns of the nested table are qualified; a bare column it does not have keeps binding outside
    assert (
        rewritten("SELECT name FROM s WHERE EXISTS (SELECT 1 FROM ps WHERE sk = s.k AND pk IN (SELECT k FROM p WHERE f = n))")
        == "SELECT name FROM s WHERE EXISTS(SELECT 1 FROM ps, p WHERE sk = s.k AND pk = p.k AND p.f = n)"
    )
    # the other conjuncts and the joins of the nested query come along
    assert (
        rewritten("SELECT name FROM s WHERE k IN (SELECT ps.sk FROM ps WHERE ps.n > 0 AND ps.pk IN (SELECT p.k FROM p JOIN l ON l.pk = p.k WHERE l.q > 2))")
        == "SELECT name FROM s WHERE k IN (SELECT ps.sk FROM ps, p, l WHERE ps.n > 0 AND ps.pk = p.k AND l.q > 2 AND l.pk = p.k)"
    )


def test_a_nested_exists_becomes_a_join():
    assert (
        rewritten("SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE EXISTS (SELECT 1 FROM p WHERE p.k = ps.pk AND p.f = 1))")
        == "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, p WHERE p.k = ps.pk AND p.f = 1)"
    )


def test_a_name_already_in_use_gets_a_fresh_alias():
    out = rewritten("SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT ps.pk FROM ps WHERE ps.n = 1))")
    tree = sqlglot.parse_one(out, read="postgres")
    inner = tree.find(exp.In).args["query"].this
    names = [source.alias_or_name for source in [inner.args["from_"].this, *[j.this for j in inner.args["joins"]]]]
    assert names[0] == "ps" and names[1].startswith("kqm") and len(set(names)) == 2
    fresh = names[1]
    assert f"ps.pk = {fresh}.pk AND {fresh}.n = 1" in out


@pytest.mark.parametrize("sql", [
    "SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p)",  # not inside a membership test: repeats would show
    "SELECT s.name FROM s WHERE s.k = (SELECT MAX(ps.sk) FROM ps WHERE ps.pk IN (SELECT p.k FROM p))",  # a scalar subquery
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk NOT IN (SELECT p.k FROM p))",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE NOT EXISTS (SELECT 1 FROM p WHERE p.k = ps.pk))",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.n = 1 OR ps.pk IN (SELECT p.k FROM p))",  # under OR
    "SELECT s.name FROM s WHERE s.k IN (SELECT COUNT(*) FROM ps WHERE ps.pk IN (SELECT p.k FROM p))",  # repeats change the count
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p) GROUP BY ps.sk HAVING COUNT(*) > 1)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p) LIMIT 1)",
    "SELECT s.name FROM s WHERE EXISTS (SELECT ROW_NUMBER() OVER () FROM ps WHERE ps.pk IN (SELECT p.k FROM p))",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT MAX(p.k) FROM p))",  # one row over no rows
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE EXISTS (SELECT COUNT(*) FROM p WHERE p.k = ps.pk))",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p GROUP BY p.k HAVING COUNT(*) > 1))",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p LIMIT 1))",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE EXISTS (SELECT 1 FROM p WHERE p.k = ps.pk LIMIT 0))",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p LEFT JOIN l ON l.pk = p.k))",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p JOIN l USING (k)))",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT d.k FROM (SELECT k FROM p) AS d))",  # not a base table
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT u.k FROM unknown_table AS u))",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p WHERE p.f IN (SELECT l.q FROM l)))",  # innermost first
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE (ps.pk, ps.n) IN (SELECT p.k, p.f FROM p))",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT k FROM p, l))",  # bare name, two sources
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE f = 1 AND ps.pk IN (SELECT p.k FROM p))",  # bare f would bind to p
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p) ORDER BY ps.sk)",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p) UNION SELECT l.sk FROM l)",
])
def test_shapes_left_alone(sql):
    tree = sqlglot.parse_one(sql, read="postgres")
    target = None
    for select in tree.find_all(exp.Select):
        where = select.args.get("where")
        if where is not None and where.find(exp.In, exp.Exists) is not None and select.find_ancestor(exp.Select) is not None:
            target = select
            break
    target = target or tree.find(exp.Select)
    assert join_nested_membership(target, SCHEMA) is None


def test_the_innermost_nested_query_goes_first():
    # the middle query holds a nested query, so the outer one waits; the middle one is rewritten on its own turn
    sql = "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p WHERE p.f IN (SELECT l.q FROM l)))"
    assert rewritten(sql) == "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p, l WHERE p.f = l.q))"


TARGETS = [
    (
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p WHERE p.f = 1))",
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps JOIN p ON ps.pk = p.k WHERE p.f = 1)",
    ),
    (
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p WHERE p.f = 1))",
        "SELECT s.name FROM s WHERE EXISTS (SELECT 1 FROM ps JOIN p ON ps.pk = p.k WHERE p.f = 1 AND ps.sk = s.k)",
    ),
    (
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE EXISTS (SELECT 1 FROM p WHERE p.k = ps.pk AND p.f = 1))",
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps, p WHERE ps.pk = p.k AND p.f = 1)",
    ),
    (
        "SELECT s.name FROM s WHERE s.k NOT IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p WHERE p.f = 1))",
        "SELECT s.name FROM s WHERE s.k NOT IN (SELECT ps.sk FROM ps JOIN p ON ps.pk = p.k WHERE p.f = 1)",
    ),
    (
        "SELECT s.name FROM s WHERE NOT EXISTS (SELECT 1 FROM ps WHERE ps.sk = s.k AND ps.pk IN (SELECT l.pk FROM l WHERE l.q > ps.n))",
        "SELECT s.name FROM s WHERE NOT EXISTS (SELECT 1 FROM ps JOIN l ON ps.pk = l.pk AND l.q > ps.n WHERE ps.sk = s.k)",
    ),
]

NEAR_MISSES = [
    (  # the nested filter differs
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p WHERE p.f = 1))",
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps JOIN p ON ps.pk = p.k WHERE p.f = 2)",
        {"s": [(1, 10)], "ps": [(1, 5, 0)], "p": [(5, 1)]},
    ),
    (  # NOT IN is not a join
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk NOT IN (SELECT p.k FROM p WHERE p.f = 1))",
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps JOIN p ON ps.pk = p.k WHERE p.f = 1)",
        {"s": [(1, 10)], "ps": [(1, 5, 0)], "p": [(5, 1)]},
    ),
    (  # outside a membership test the join repeats rows
        "SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p)",
        "SELECT ps.sk FROM ps JOIN p ON ps.pk = p.k",
        {"ps": [(1, 5, 0)], "p": [(5, 1), (5, 2)]},
    ),
    (  # a count inside the membership test sees the repeats
        "SELECT s.name FROM s WHERE s.k IN (SELECT COUNT(*) FROM ps WHERE ps.pk IN (SELECT p.k FROM p))",
        "SELECT s.name FROM s WHERE s.k IN (SELECT COUNT(*) FROM ps JOIN p ON ps.pk = p.k)",
        {"s": [(1, 10)], "ps": [(1, 5, 0)], "p": [(5, 1), (5, 2)]},
    ),
    (  # the join compares another column
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p))",
        "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps JOIN p ON ps.pk = p.f)",
        {"s": [(1, 10)], "ps": [(1, 5, 0)], "p": [(5, 1)]},
    ),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_targets_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


@pytest.mark.parametrize("left, right, rows", NEAR_MISSES)
def test_near_misses_are_not_proved(left, right, rows):
    assert differ_on_duckdb(left, right, rows)
    assert not proved(left, right)
    assert not proved(right, left)


REPLAYED = [
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p WHERE p.f = 1))",
    "SELECT s.name FROM s WHERE s.k NOT IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.k FROM p WHERE p.f = 1))",
    "SELECT s.name, s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT p.f FROM p)) FROM s",
    "SELECT s.name FROM s WHERE EXISTS (SELECT 1 FROM ps WHERE sk = s.k AND pk IN (SELECT k FROM p WHERE f = n))",
    "SELECT s.name FROM s WHERE NOT EXISTS (SELECT ps.n FROM ps WHERE ps.sk = s.k AND EXISTS (SELECT 1 FROM l WHERE l.pk = ps.pk AND l.q > ps.n))",
    "SELECT s.name FROM s WHERE s.k IN (SELECT ps.sk FROM ps WHERE ps.pk IN (SELECT ps.pk FROM ps WHERE ps.n = 1))",
    "SELECT s.name FROM s WHERE s.k NOT IN (SELECT DISTINCT ps.n FROM ps WHERE ps.pk + 1 IN (SELECT p.k + p.f FROM p JOIN l ON l.pk = p.k))",
]


@pytest.mark.parametrize("sql", REPLAYED)
def test_the_rewrite_returns_the_same_rows(sql):
    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    value = lambda: rng.choice([None, 0, 1, 1, 2, 3])
    for _ in range(40):
        rows = {name: [tuple(value() for _ in columns) for _ in range(rng.randint(0, 4))] for name, columns in SCHEMA.items()}
        first, second = rows_on_duckdb([sql, after], rows)
        assert first == second, rows
