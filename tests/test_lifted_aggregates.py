"""A select with no FROM whose outputs each read one aggregate subquery over the same rows (``lifted_aggregates``)."""

import random

import duckdb
import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.lifted_aggregates import lift_scalar_aggregates
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"pay": ["emp", "amount", "team", "day"], "teams": ["team", "label"]}
TYPES = {"pay": {"emp": "INT", "amount": "INT", "team": "INT", "day": "DATE"}, "teams": {"team": "INT", "label": "VARCHAR"}}


def reading(sql: str, dialect: str = "mysql", schema=SCHEMA) -> str | None:
    written, changed = lift_scalar_aggregates(sqlglot.parse_one(sql, read=dialect), schema)
    return written.sql(dialect=dialect) if changed else None


def test_one_aggregate_subquery_is_the_aggregate_select():
    assert reading("SELECT (SELECT MAX(amount) FROM pay) AS top") == "SELECT MAX(amount) AS top FROM pay"
    assert reading("SELECT (SELECT MAX(amount) FROM pay WHERE team < 10) AS top") == "SELECT MAX(amount) AS top FROM pay WHERE team < 10"
    assert reading("SELECT (SELECT COUNT(*) FROM pay AS p JOIN teams AS t ON p.team = t.team WHERE t.label = 'a') AS n") == (
        "SELECT COUNT(*) AS n FROM pay AS p JOIN teams AS t ON p.team = t.team WHERE t.label = 'a'"
    )
    assert reading("SELECT (SELECT SUM(s.amount) FROM (SELECT amount FROM pay WHERE team = 1) AS s) AS total") == "SELECT SUM(s.amount) AS total FROM (SELECT amount FROM pay WHERE team = 1) AS s"
    # the expression around the subquery stays around the aggregate
    assert reading("SELECT IFNULL((SELECT MAX(amount) FROM pay), 0) AS top") == "SELECT COALESCE(MAX(amount), 0) AS top FROM pay"
    assert reading("SELECT (SELECT MAX(amount) FROM pay) + 1 AS next") == "SELECT MAX(amount) + 1 AS next FROM pay"
    # an output built around the aggregate keeps its parentheses
    assert reading("SELECT 2 * (SELECT COUNT(*) + 1 FROM pay) AS n") == "SELECT 2 * (COUNT(*) + 1) AS n FROM pay"
    assert reading("SELECT (SELECT MAX(amount) FROM pay)", dialect="postgres") == "SELECT MAX(amount) FROM pay"


def test_several_outputs_over_the_same_rows_are_one_select():
    assert reading("SELECT (SELECT MAX(amount) FROM pay WHERE team = 1) AS top, (SELECT MIN(day) FROM pay WHERE team = 1) AS first, 'a' AS label") == (
        "SELECT MAX(amount) AS top, MIN(day) AS first, 'a' AS label FROM pay WHERE team = 1"
    )
    # the same sources written with other case
    assert reading("SELECT (SELECT MAX(amount) FROM pay WHERE team = 1) AS top, (SELECT COUNT(emp) FROM PAY WHERE TEAM = 1) AS n") == "SELECT MAX(amount) AS top, COUNT(emp) AS n FROM pay WHERE team = 1"
    # other rows
    assert reading("SELECT (SELECT MAX(amount) FROM pay WHERE team = 1) AS top, (SELECT MIN(day) FROM pay WHERE team = 2) AS first") is None
    assert reading("SELECT (SELECT MAX(amount) FROM pay) AS top, (SELECT COUNT(*) FROM teams) AS n") is None
    assert reading("SELECT (SELECT MAX(amount) FROM pay) AS top, (SELECT MIN(day) FROM pay AS p) AS first") is None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT (SELECT amount FROM pay) AS a",  # no aggregate: one row is not promised
        "SELECT (SELECT MAX(amount) FROM pay GROUP BY team) AS a",
        "SELECT (SELECT MAX(amount) FROM pay HAVING COUNT(*) > 1) AS a",  # no row when the test fails
        "SELECT (SELECT MAX(amount) FROM pay ORDER BY 1 LIMIT 1) AS a",
        "SELECT (SELECT DISTINCT MAX(amount) FROM pay) AS a",
        "SELECT (SELECT AVG(amount) FROM pay) AS a",  # a quotient is not moved
        "SELECT (SELECT SUM(amount) / COUNT(*) FROM pay) AS a",
        "SELECT (SELECT SUM(amount) DIV 2 FROM pay) AS a",
        "SELECT (SELECT MAX(amount) - MIN(amount) FROM pay) AS a",  # two aggregates: aggregate_scalars writes that shape
        "SELECT (SELECT MAX(amount) + (SELECT 1) FROM pay) AS a",
        "SELECT (SELECT GROUP_CONCAT(emp) FROM pay) AS a",
        "SELECT (SELECT MAX(amount) FROM pay) + (SELECT MIN(amount) FROM pay) AS a",  # two subqueries in one output
        "SELECT (SELECT MAX(amount) FROM pay WHERE amount < (SELECT MAX(amount) FROM pay)) AS a",  # a subquery in the test
        "SELECT (SELECT MAX(amount) FROM pay WHERE amount < RAND()) AS a",
        "SELECT (SELECT MAX(amount) FROM pay) AS a, RAND() AS r",
        "SELECT (SELECT MAX(amount) FROM pay) AS a FROM teams",  # one row per row of teams
        "SELECT (SELECT MAX(amount) FROM pay) AS a WHERE 1 = 0",
        "SELECT DISTINCT (SELECT MAX(amount) FROM pay) AS a",
        "SELECT (SELECT MAX(amount) FROM pay) AS a LIMIT 0",
        "SELECT (SELECT MAX(amount) FROM pay) AS a UNION ALL SELECT 1",
        "SELECT 1 AS a",
        "SELECT 1 IN (SELECT MAX(amount) FROM pay) AS a",
        "SELECT EXISTS (SELECT MAX(amount) FROM pay) AS a",
        "SELECT MAX((SELECT MAX(amount) FROM pay)) AS a",  # inside an aggregate of the query
        "SELECT (SELECT MAX(gone) FROM pay) AS a",  # a column that does not resolve inside
        "SELECT (SELECT MAX(amount) FROM pay WHERE team = 1) AS team",  # the output name would stand next to a test on that name
        "SELECT (SELECT COUNT(*) FROM pay AS p JOIN teams AS t ON p.team = t.team AND label = 'a') AS label",
    ],
)
def test_other_selects_are_left_alone(sql):
    assert reading(sql) is None


def test_it_needs_a_schema_and_leaves_the_tree_given_alone():
    sql = "SELECT (SELECT MAX(amount) FROM pay) AS top"
    assert reading(sql, schema=None) is None
    tree = sqlglot.parse_one(sql, read="mysql")
    written, changed = lift_scalar_aggregates(tree, SCHEMA)
    assert changed and tree.sql(dialect="mysql") == sql and written is not tree


@pytest.mark.parametrize("seed", range(4))
def test_replay_on_generated_tables(seed):
    rng = random.Random(seed)
    con = duckdb.connect()
    con.execute("CREATE TABLE pay (emp INT, amount INT, team INT, day DATE)")
    con.execute("CREATE TABLE teams (team INT, label VARCHAR)")
    compared = 0
    for _ in range(10):
        con.execute("DELETE FROM pay")
        con.execute("DELETE FROM teams")
        nulls = rng.choice([0.0, 0.3, 1.0])

        def maybe(value):
            return None if rng.random() < nulls else value

        for i in range(rng.choice([0, 1, 4, 9])):
            con.execute("INSERT INTO pay VALUES (?, ?, ?, DATE '2020-01-01' + CAST(? AS INT))", (i, maybe(rng.randint(-3, 9)), maybe(rng.randint(0, 2)), rng.randint(0, 5)))
        for i in range(rng.choice([0, 2, 3])):
            con.execute("INSERT INTO teams VALUES (?, ?)", (i, rng.choice(["a", "b"])))
        c = rng.randint(0, 2)
        frames = ["pay", f"pay WHERE team = {c}", f"pay WHERE amount > {c}", "pay AS p JOIN teams AS t ON p.team = t.team", "pay AS p LEFT JOIN teams AS t ON p.team = t.team WHERE t.label = 'a'"]
        for frame in frames:
            q = "p." if " AS p " in frame else ""
            values = [f"MAX({q}amount)", f"MIN({q}emp)", "COUNT(*)", f"COUNT({q}amount)", f"SUM({q}amount)", f"COUNT(DISTINCT {q}team)", f"SUM({q}amount) + 1", f"-MAX({q}amount)"]
            for _ in range(6):
                picked = rng.sample(values, rng.randint(1, 3))
                outputs = [rng.choice(["({})", "COALESCE(({}), -1)", "({}) * 2"]).format(f"SELECT {value} FROM {frame}") + f" AS o{i}" for i, value in enumerate(picked)]
                sql = "SELECT " + ", ".join(outputs)
                written, changed = lift_scalar_aggregates(sqlglot.parse_one(sql, read="duckdb"), SCHEMA)
                assert changed, sql
                assert con.execute(sql).fetchall() == con.execute(written.sql(dialect="duckdb")).fetchall(), sql
                compared += 1
    assert compared == 300


FACTS = {"pay": TableConstraints(not_null=frozenset({"emp"}), keys=(("emp",),)), "teams": TableConstraints(not_null=frozenset({"team"}), keys=(("team",),))}
OPTIONS = dict(schema=SCHEMA, types=TYPES, constraints=FACTS, dialect="mysql", compare_names=False, exact_arithmetic=True, timeout_ms=3000)
PROVED = [
    ("SELECT MAX(amount) AS a FROM pay WHERE team > 3", "SELECT (SELECT MAX(amount) FROM pay WHERE team > 3) AS a"),
    ("SELECT MAX(amount) AS a, MIN(day) AS d FROM pay", "SELECT (SELECT MAX(amount) FROM pay) AS a, (SELECT MIN(day) FROM pay) AS d"),
    ("SELECT IFNULL(MAX(amount), 0) AS a FROM pay WHERE team = 1", "SELECT IFNULL((SELECT MAX(amount) FROM pay WHERE team = 1), 0) AS a"),
    ("SELECT COUNT(*) AS n FROM pay AS p JOIN teams AS t ON p.team = t.team", "SELECT (SELECT COUNT(*) FROM pay AS p JOIN teams AS t ON p.team = t.team) AS n"),
]
NOT_PROVED = [
    ("SELECT MAX(amount) AS a FROM pay WHERE team > 3", "SELECT (SELECT MAX(amount) FROM pay WHERE team > 4) AS a"),
    ("SELECT MAX(amount) AS a FROM pay WHERE team > 3", "SELECT (SELECT MIN(amount) FROM pay WHERE team > 3) AS a"),
    ("SELECT MAX(amount) AS a FROM pay GROUP BY team", "SELECT (SELECT MAX(amount) FROM pay) AS a"),
]


@pytest.mark.parametrize("left, right", PROVED)
def test_the_aggregate_asked_both_ways_is_proved(left, right):
    result = prove_equivalent_algebraic(left, right, **OPTIONS)
    assert result.proven, result.reason
    with shape_fallback.without("lifted_aggregates"):
        assert not prove_equivalent_algebraic(left, right, **OPTIONS).proven


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_other_aggregates_are_not_proved(left, right):
    assert not prove_equivalent_algebraic(left, right, **OPTIONS).proven


def test_it_is_a_rule_of_the_thirteenth_batch():
    assert "lifted_aggregates" in shape_fallback.THIRTEENTH_BATCH
    sql = "SELECT (SELECT MAX(amount) FROM pay WHERE team > 3) AS a"
    with shape_fallback.tracking() as fired:
        normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "lifted_aggregates" in fired
    with shape_fallback.without("lifted_aggregates"), shape_fallback.tracking() as fired:
        normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "lifted_aggregates" not in fired
