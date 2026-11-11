"""A keyed table joined to a grouped derived table, read as the grouping of the join (``kumo_sql_prover.grouped_join_keys``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.grouped_join_keys import join_before_grouping
from kumo_sql_prover.smt_equivalence import TableConstraints

# people: id is the key; games: day is the key, w1..w3 name people and may be NULL; notes has no key
SCHEMA = {"people": ["id", "name"], "games": ["day", "w1", "w2", "w3"], "notes": ["id", "name"]}
KEYS = {"people": [("id",)], "games": [("day",)]}
CONSTRAINTS = {
    "people": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
    "games": TableConstraints(not_null=frozenset({"day"}), keys=(("day",),)),
}
W = "(SELECT w1 AS who, day FROM games UNION ALL SELECT w2, day FROM games) AS w"
G = f"(SELECT w.who, COUNT(*) AS wins FROM {W} GROUP BY w.who) AS g"
SUM2 = "SUM(CASE WHEN p.id = g.w1 THEN 1 ELSE 0 END + CASE WHEN p.id = g.w2 THEN 1 ELSE 0 END)"
FLAT = f"SELECT p.id, p.name, {SUM2} AS n FROM people AS p CROSS JOIN games AS g GROUP BY p.id HAVING {SUM2} > 0"


def rewritten(sql, keys=KEYS):
    after = join_before_grouping(sqlglot.parse_one(sql, read="mysql"), SCHEMA, keys)
    return None if after is None else after.sql(dialect="mysql")


def proved(left, right, constraints=CONSTRAINTS):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=constraints, compare_names=False, dialect="mysql").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE people (id BIGINT, name VARCHAR)")
    db.execute("CREATE TABLE games (day BIGINT, w1 BIGINT, w2 BIGINT, w3 BIGINT)")
    db.execute("CREATE TABLE notes (id BIGINT, name VARCHAR)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[sqlglot.transpile(sql, read="mysql", write="duckdb")[0] for sql in queries])]


def test_the_keyed_table_joins_before_the_grouping():
    assert rewritten(f"SELECT p.id, p.name, g.wins FROM {G} JOIN people AS p ON g.who = p.id") == (
        f"SELECT p.id, p.name, COUNT(*) AS wins FROM {W} CROSS JOIN people AS p WHERE p.id = w.who GROUP BY p.id"
    )
    # the group key read from the keyed table, bare columns of the grouped input named, other conditions: on the rows, or on an aggregate
    assert rewritten(
        f"SELECT g.who, p.name, g.wins + 1 AS more FROM people AS p, (SELECT who, COUNT(*) AS wins, MAX(day) AS last FROM {W} WHERE day > 1 GROUP BY who) AS g"
        " WHERE p.id = g.who AND g.last > 3 AND p.name <> 'x'"
    ) == (
        f"SELECT p.id AS who, p.name, COUNT(*) + 1 AS more FROM {W} CROSS JOIN people AS p WHERE w.day > 1 AND p.id = w.who AND p.name <> 'x' GROUP BY p.id HAVING MAX(w.day) > 3"
    )
    # two group keys, both tied
    assert rewritten("SELECT p.name, g.n FROM people AS p JOIN (SELECT w1, w2, COUNT(DISTINCT w3) AS n FROM games GROUP BY w1, w2) AS g ON g.w1 = p.id AND g.w2 = p.id") == (
        "SELECT p.name, COUNT(DISTINCT games.w3) AS n FROM games CROSS JOIN people AS p WHERE p.id = games.w1 AND p.id = games.w2 GROUP BY p.id"
    )


@pytest.mark.parametrize("sql", [
    f"SELECT p.id, g.wins FROM {G} LEFT JOIN people AS p ON g.who = p.id",  # a group without a person stays
    f"SELECT p.id, g.wins FROM people AS p LEFT JOIN {G} ON g.who = p.id",
    f"SELECT p.name, g.wins FROM {G} JOIN people AS p ON g.who = p.name",  # the name is no key: several people per group
    f"SELECT n.id, g.wins FROM {G} JOIN notes AS n ON g.who = n.id",  # notes has no key
    f"SELECT p.id, g.wins FROM {G} JOIN people AS p ON g.who <= p.id",
    f"SELECT p.id, g.wins FROM {G} JOIN people AS p ON g.who = p.id OR g.who IS NULL",
    f"SELECT p.id, g.wins FROM {G} JOIN people AS p ON g.wins = p.id",  # tied to the aggregate, not to the key
    f"SELECT p.id, g.wins FROM {G}, people AS p",
    "SELECT p.id, g.n FROM (SELECT n.id AS who, SUM(g.day) AS n FROM notes AS n LEFT JOIN games AS g ON g.w1 = n.id GROUP BY n.id) AS g JOIN people AS p ON p.id = g.who",  # an outer join inside
    f"SELECT p.id, g.n FROM people AS p JOIN (SELECT w1, w2, COUNT(*) AS n FROM games GROUP BY w1, w2) AS g ON g.w1 = p.id",  # w2 splits a person's rows
    "SELECT p.id, g.n FROM people AS p JOIN (SELECT w1, COUNT(*) AS n FROM games GROUP BY w1, w2) AS g ON g.w1 = p.id",  # a hidden key does too
    "SELECT p.id, g.n FROM people AS p JOIN (SELECT w1, w3, COUNT(*) AS n FROM games GROUP BY w1) AS g ON g.w1 = p.id",  # a loosely grouped output
    "SELECT p.id, g.n FROM people AS p JOIN (SELECT w1, COUNT(*) AS n FROM games GROUP BY w1 HAVING COUNT(*) > 1) AS g ON g.w1 = p.id",
    "SELECT p.id, g.n FROM people AS p JOIN (SELECT w1, COUNT(*) AS n FROM games GROUP BY w1 LIMIT 2) AS g ON g.w1 = p.id",
    "SELECT p.id, g.n FROM people AS p JOIN (SELECT DISTINCT w1, COUNT(*) AS n FROM games GROUP BY w1) AS g ON g.w1 = p.id",
    "SELECT p.id, g.n FROM people AS p JOIN (SELECT w1, COUNT(*) AS n FROM games GROUP BY w1 WITH ROLLUP) AS g ON g.w1 = p.id",
    "SELECT p.id, g.n FROM people AS p JOIN (SELECT w1, COUNT(*) OVER () AS n FROM games GROUP BY w1) AS g ON g.w1 = p.id",
    "SELECT p.id, g.n FROM people AS p JOIN (SELECT w1, COUNT(*) AS n FROM games) AS g ON g.w1 = p.id",  # not grouped
    "SELECT p.id, g.n FROM people AS p JOIN (SELECT w1, COUNT(*) AS n FROM games WHERE w2 IN (SELECT id FROM notes) GROUP BY w1) AS g ON g.w1 = p.id",
    "SELECT p.id, g.n FROM people AS p JOIN (SELECT id, COUNT(*) AS n FROM notes AS p GROUP BY id) AS g ON g.id = p.id",  # the input is named like the table
    "SELECT p.id, g.n FROM people AS p JOIN (SELECT a.id, COUNT(*) AS n FROM notes AS a, notes AS b WHERE name = 'x' GROUP BY a.id) AS g ON g.id = p.id",  # name: a's or b's
    f"SELECT p.id, COUNT(*) FROM {G} JOIN people AS p ON g.who = p.id GROUP BY p.id",  # the select groups itself
    f"SELECT DISTINCT p.name, g.wins FROM {G} JOIN people AS p ON g.who = p.id",
    f"SELECT p.id, g.wins FROM {G} JOIN people AS p ON g.who = p.id ORDER BY g.wins LIMIT 1",
    f"SELECT id, wins FROM {G} JOIN people AS p ON g.who = p.id",  # bare columns
    f"SELECT p.id, g.wins FROM {G} JOIN people AS p ON g.who = p.id JOIN notes AS n ON n.id = p.id",  # a third source
    f"SELECT p.id, g.wins FROM {G} JOIN people AS p ON g.who = p.id WHERE EXISTS (SELECT 1 FROM notes AS n WHERE n.id = p.id)",
    f"SELECT * FROM {G} JOIN people AS p ON g.who = p.id",
])
def test_other_joins_stay(sql):
    assert rewritten(sql) is None


def test_the_join_needs_the_declared_key():
    sql = f"SELECT p.id, p.name, g.wins FROM {G} JOIN people AS p ON g.who = p.id"
    assert rewritten(sql) is not None
    assert rewritten(sql, keys=None) is None
    assert rewritten(sql, keys={"people": [("id", "name")]}) is None  # the name is not tied
    assert rewritten(sql, keys={"games": [("day",)]}) is None


PROOFS = [
    (f"SELECT p.id, p.name, g.wins FROM {G} JOIN people AS p ON g.who = p.id", FLAT),
    (
        f"SELECT p.id, p.name, g.wins FROM {G} JOIN people AS p ON g.who = p.id",
        f"SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN {W} ON p.id = w.who GROUP BY p.id",
    ),
    (
        f"SELECT g.who, p.name, g.wins FROM people AS p, (SELECT who, COUNT(who) AS wins FROM {W} GROUP BY who) AS g WHERE p.id = g.who AND g.wins >= 1",
        FLAT,
    ),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # the outer join keeps the group of the NULL winner
    (f"SELECT g.who, p.name, g.wins FROM {G} LEFT JOIN people AS p ON g.who = p.id", FLAT),
    # joined on the name, a group meets every person of that name
    (f"SELECT p.id, p.name, g.wins FROM {G} JOIN people AS p ON g.who = p.name", f"SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN {W} ON p.name = w.who GROUP BY p.id"),
    # grouped by the day as well: one row per person and day
    (f"SELECT p.id, p.name, g.wins FROM (SELECT w.who, COUNT(*) AS wins FROM {W} GROUP BY w.who, w.day) AS g JOIN people AS p ON g.who = p.id", FLAT),
    # only the groups of more than one win
    (f"SELECT p.id, p.name, g.wins FROM (SELECT w.who, COUNT(*) AS wins FROM {W} GROUP BY w.who HAVING COUNT(*) > 1) AS g JOIN people AS p ON g.who = p.id", FLAT),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_without_a_key_the_join_really_differs():
    # two notes with the same id: the grouped table's row meets both, the grouping of the join merges them
    tables = {"notes": [(1, "a"), (1, "b")], "games": [(1, 1, 1, None)]}
    joined, grouped = run(
        tables,
        f"SELECT n.id, g.wins FROM {G} JOIN notes AS n ON g.who = n.id",
        f"SELECT n.id, COUNT(*) FROM {W} CROSS JOIN notes AS n WHERE n.id = w.who GROUP BY n.id",
    )
    assert joined == ["(1, 2)", "(1, 2)"] and grouped == ["(1, 4)"]


QUERIES = [
    f"SELECT p.id, p.name, g.wins FROM {G} JOIN people AS p ON g.who = p.id",
    f"SELECT g.who, g.wins + 1 AS more FROM people AS p, (SELECT who, COUNT(*) AS wins, MAX(day) AS last FROM {W} WHERE day > 1 GROUP BY who) AS g"
    " WHERE p.id = g.who AND g.last > 3 AND (p.name <> 'x' OR p.name IS NULL)",
    "SELECT p.id, g.n, g.m FROM people AS p JOIN (SELECT w1, w2, COUNT(DISTINCT w3) AS n, AVG(day) AS m FROM games GROUP BY w1, w2) AS g ON g.w1 = p.id AND g.w2 = p.id",
    "SELECT p.id, g.n FROM (SELECT n.id AS who, SUM(g.day) AS n FROM notes AS n JOIN games AS g ON g.w1 = n.id GROUP BY n.id) AS g JOIN people AS p ON p.id = g.who",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the rewritten one on random tables that meet the declared facts (the two keys)."""

    after = rewritten(sql)
    assert after is not None
    # DuckDB wants every output column grouped; the name is fixed by the key the rule grouped by
    after = after.replace("GROUP BY p.id", "GROUP BY p.id, p.name") if "p.name" in after.split(" FROM ")[0] else after
    rng = random.Random(len(sql))
    person = lambda: rng.choice([None, 1, 1, 2, 3, 4])  # noqa: E731
    for _ in range(150):
        ids = rng.sample([1, 2, 3, 4, 5], rng.choice([0, 1, 2, 4]))
        tables = {
            "people": [(i, rng.choice([None, "x", "y", "y"])) for i in ids],
            "games": [(day, person(), person(), person()) for day in rng.sample(range(1, 9), rng.choice([0, 1, 2, 5]))],
            "notes": [(person(), rng.choice([None, "x", "y"])) for _ in range(rng.choice([0, 1, 3]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
