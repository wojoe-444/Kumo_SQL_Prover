"""A ``SUM`` of the counts of grouped ``UNION ALL`` branches, read over the branches' rows (``kumo_sql_prover.counted_union_branches``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.counted_union_branches import ungroup_counted_branches
from kumo_sql_prover.smt_equivalence import TableConstraints

# people: id is the key; games: day is the key, w1..w3 name people and may be NULL; notes has no key
SCHEMA = {"people": ["id", "name"], "games": ["day", "w1", "w2", "w3"], "notes": ["id", "name"]}
KEYS = {"people": [("id",)], "games": [("day",)]}
NOT_NULL = {"people": frozenset({"id"}), "games": frozenset({"day"})}
CONSTRAINTS = {
    "people": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
    "games": TableConstraints(not_null=frozenset({"day"}), keys=(("day",),)),
}
COUNTED = "(SELECT g.w1 AS who, COUNT(*) AS n FROM games AS g GROUP BY g.w1 UNION ALL SELECT g.w2, COUNT(g.w2) FROM games AS g GROUP BY g.w2) AS u"
ROWS = "(SELECT g.w1 AS who, 1 AS n FROM games AS g UNION ALL SELECT g.w2, CASE WHEN g.w2 IS NULL THEN 0 ELSE 1 END FROM games AS g) AS u"
SUM2 = "SUM(CASE WHEN p.id = g.w1 THEN 1 ELSE 0 END + CASE WHEN p.id = g.w2 THEN 1 ELSE 0 END)"
FLAT = f"SELECT p.id, p.name, {SUM2} AS n FROM people AS p CROSS JOIN games AS g GROUP BY p.id HAVING {SUM2} > 0"


def rewritten(sql, keys=KEYS, not_null=NOT_NULL):
    after = ungroup_counted_branches(sqlglot.parse_one(sql, read="mysql"), SCHEMA, keys, not_null)
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


def test_counted_branches_lose_their_grouping():
    assert rewritten(f"SELECT p.id, SUM(u.n) AS wins FROM people AS p JOIN {COUNTED} ON p.id = u.who GROUP BY p.id") == (
        f"SELECT p.id, SUM(u.n) AS wins FROM people AS p JOIN {ROWS} ON p.id = u.who GROUP BY p.id"
    )
    # a global sum, bare columns, COUNT of a literal
    assert rewritten("SELECT SUM(n) FROM (SELECT w1 AS who, COUNT(1) AS n FROM games GROUP BY w1 UNION ALL SELECT w2, COUNT(*) FROM games GROUP BY w2) AS u") == (
        "SELECT SUM(n) FROM (SELECT w1 AS who, 1 AS n FROM games UNION ALL SELECT w2, 1 FROM games) AS u"
    )
    # a GROUP BY name that is the branch's output, a SUM of integer literals, a branch that is not grouped, a filter and a HAVING on the sum
    assert rewritten(
        "SELECT who, SUM(n) FROM (SELECT w1 AS who, COUNT(w3) AS n FROM games GROUP BY who UNION ALL SELECT w2, SUM(IF(w3 > 1, 2, 0)) FROM games WHERE day > 1 GROUP BY w2"
        " UNION ALL SELECT w3, 5 FROM games) AS u WHERE who > 0 GROUP BY who HAVING SUM(n) > 1"
    ) == (
        "SELECT who, SUM(n) FROM (SELECT w1 AS who, CASE WHEN w3 IS NULL THEN 0 ELSE 1 END AS n FROM games UNION ALL SELECT w2, (IF(w3 > 1, 2, 0)) FROM games WHERE day > 1"
        " UNION ALL SELECT w3, 5 FROM games) AS u WHERE who > 0 GROUP BY who HAVING SUM(n) > 1"
    )
    # a column fixed by the branch's key rides along
    assert rewritten(
        "SELECT u.id, u.name, SUM(u.n) FROM (SELECT p.id, p.name, COUNT(g.w1) AS n FROM people AS p, games AS g WHERE p.id = g.w1 GROUP BY p.id"
        " UNION ALL SELECT p.id, p.name, COUNT(*) FROM people AS p, games AS g WHERE p.id = g.w2 GROUP BY p.id) AS u GROUP BY u.id, u.name"
    ) == (
        "SELECT u.id, u.name, SUM(u.n) FROM (SELECT p.id, p.name, CASE WHEN g.w1 IS NULL THEN 0 ELSE 1 END AS n FROM people AS p, games AS g WHERE p.id = g.w1"
        " UNION ALL SELECT p.id, p.name, 1 FROM people AS p, games AS g WHERE p.id = g.w2) AS u GROUP BY u.id, u.name"
    )


@pytest.mark.parametrize("sql", [
    f"SELECT p.id, COUNT(*) FROM people AS p JOIN {COUNTED} ON p.id = u.who GROUP BY p.id",  # counts groups, not rows
    f"SELECT p.id, SUM(u.n), COUNT(*) FROM people AS p JOIN {COUNTED} ON p.id = u.who GROUP BY p.id",
    f"SELECT p.id, MAX(u.n) FROM people AS p JOIN {COUNTED} ON p.id = u.who GROUP BY p.id",
    f"SELECT p.id, SUM(u.n + 1) FROM people AS p JOIN {COUNTED} ON p.id = u.who GROUP BY p.id",
    f"SELECT p.id, SUM(DISTINCT u.n) FROM people AS p JOIN {COUNTED} ON p.id = u.who GROUP BY p.id",
    f"SELECT p.id, SUM(u.n), MIN(u.who) FROM people AS p JOIN {COUNTED} ON p.id = u.who GROUP BY p.id",
    f"SELECT p.id, SUM(u.n) FROM people AS p JOIN {COUNTED} ON p.id = u.who WHERE u.n > 1 GROUP BY p.id",  # a filter on the counts
    f"SELECT p.id, u.n FROM people AS p JOIN {COUNTED} ON p.id = u.who",  # the count itself
    f"SELECT p.id, SUM(u.n) FROM people AS p JOIN {COUNTED} ON p.id = u.who GROUP BY p.id, u.n",
    f"SELECT p.id, SUM(u.n) FROM people AS p LEFT JOIN {COUNTED} ON p.id = u.who GROUP BY p.id",
    f"SELECT p.id, SUM(u.n) FROM people AS p JOIN {COUNTED} ON p.id = u.who GROUP BY p.id LIMIT 1",
    f"SELECT p.id, SUM(u.n) OVER () FROM people AS p JOIN {COUNTED} ON p.id = u.who",
    f"SELECT p.id, SUM(u.n) FROM people AS p JOIN {COUNTED.replace('UNION ALL', 'UNION')} ON p.id = u.who GROUP BY p.id",
    f"SELECT n.id, SUM(name) FROM notes AS n JOIN {COUNTED.replace(' AS n ', ' AS name ')} ON n.id = u.who GROUP BY n.id",  # name: the count, or the note's
    # branches the rule does not read
    "SELECT SUM(n) FROM (SELECT w1 AS who, COUNT(*) AS n FROM games GROUP BY w1 HAVING COUNT(*) > 1 UNION ALL SELECT w2, COUNT(*) FROM games GROUP BY w2) AS u",
    "SELECT SUM(n) FROM (SELECT w1 AS who, COUNT(DISTINCT day) AS n FROM games GROUP BY w1 UNION ALL SELECT w2, COUNT(*) FROM games GROUP BY w2) AS u",
    "SELECT SUM(n) FROM (SELECT w1 AS who, MAX(day) AS n FROM games GROUP BY w1 UNION ALL SELECT w2, COUNT(*) FROM games GROUP BY w2) AS u",
    "SELECT SUM(n) FROM (SELECT w1 AS who, SUM(day) AS n FROM games GROUP BY w1 UNION ALL SELECT w2, COUNT(*) FROM games GROUP BY w2) AS u",  # may be NULL
    "SELECT SUM(n) FROM (SELECT w1 AS who, COUNT(*) + 1 AS n FROM games GROUP BY w1 UNION ALL SELECT w2, COUNT(*) FROM games GROUP BY w2) AS u",
    "SELECT SUM(n) FROM (SELECT DISTINCT w1 AS who, COUNT(*) AS n FROM games GROUP BY w1 UNION ALL SELECT w2, COUNT(*) FROM games GROUP BY w2) AS u",
    "SELECT SUM(n) FROM ((SELECT w1 AS who, COUNT(*) AS n FROM games GROUP BY w1 LIMIT 1) UNION ALL SELECT w2, COUNT(*) FROM games GROUP BY w2) AS u",
    "SELECT SUM(n) FROM (SELECT w1 AS who, COUNT(*) AS n FROM games GROUP BY w1 WITH ROLLUP UNION ALL SELECT w2, COUNT(*) FROM games GROUP BY w2) AS u",
    "SELECT SUM(n) FROM (SELECT w1 AS who, COUNT(*) AS n FROM games UNION ALL SELECT w2, COUNT(*) FROM games) AS u",  # global counts: a row over no rows
    "SELECT SUM(n) FROM (SELECT w1 AS who, 1 AS n FROM games UNION ALL SELECT w2, 2 FROM games) AS u",  # nothing grouped
    "SELECT who, SUM(n) FROM (SELECT w1 AS who, w3 AS n FROM games GROUP BY w1, w3 UNION ALL SELECT w2, COUNT(*) FROM games GROUP BY w2) AS u GROUP BY who",  # a key where a count is summed
    "SELECT who, SUM(n) FROM (SELECT w2 AS who, COUNT(*) AS n FROM games GROUP BY w1 UNION ALL SELECT w2, COUNT(*) FROM games GROUP BY w2) AS u GROUP BY who",  # w2 is not fixed by w1
    "SELECT who, SUM(n) FROM (SELECT name AS who, COUNT(*) AS n FROM notes GROUP BY id UNION ALL SELECT name, COUNT(*) FROM notes GROUP BY name) AS u GROUP BY who",  # notes has no key
    "SELECT SUM(n) FROM (SELECT w1 AS who, COUNT(*) AS n FROM games GROUP BY w1) AS u",  # no union
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_a_column_fixed_by_a_key_needs_the_declared_key():
    sql = (
        "SELECT u.id, SUM(u.n) FROM (SELECT p.id, p.name, COUNT(*) AS n FROM people AS p, games AS g WHERE p.id = g.w1 GROUP BY p.id"
        " UNION ALL SELECT p.id, p.name, COUNT(*) FROM people AS p, games AS g WHERE p.id = g.w2 GROUP BY p.id) AS u GROUP BY u.id"
    )
    assert rewritten(sql) is not None
    assert rewritten(sql, keys=None) is None
    assert rewritten(sql, not_null=None) is None


PROOFS = [
    (f"SELECT p.id, p.name, SUM(u.n) FROM people AS p JOIN {COUNTED} ON p.id = u.who GROUP BY p.id", FLAT),
    (
        "SELECT u.id, u.name, SUM(u.n) FROM (SELECT p.id, p.name, COUNT(g.w1) AS n FROM people AS p, games AS g WHERE p.id = g.w1 GROUP BY p.id"
        " UNION ALL SELECT p.id, p.name, COUNT(*) FROM people AS p, games AS g WHERE p.id = g.w2 GROUP BY p.id) AS u GROUP BY u.id",
        FLAT,
    ),
    (
        f"SELECT p.id, p.name, SUM(u.n) FROM people AS p JOIN {COUNTED} ON p.id = u.who GROUP BY p.id",
        "SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN (SELECT w1 AS who FROM games UNION ALL SELECT w2 FROM games) AS u ON p.id = u.who GROUP BY p.id",
    ),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # the number of groups a person meets, not the number of wins
    (f"SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN {COUNTED} ON p.id = u.who GROUP BY p.id", FLAT),
    # the outer join from the union keeps the NULL winner's group
    (f"SELECT u.who, p.name, SUM(u.n) FROM {COUNTED} LEFT JOIN people AS p ON p.id = u.who GROUP BY u.who", FLAT),
    # COUNT(w3) counts fewer rows than COUNT(*)
    (
        "SELECT p.id, p.name, SUM(u.n) FROM people AS p JOIN (SELECT g.w1 AS who, COUNT(g.w3) AS n FROM games AS g GROUP BY g.w1"
        " UNION ALL SELECT g.w2, COUNT(*) FROM games AS g GROUP BY g.w2) AS u ON p.id = u.who GROUP BY p.id",
        FLAT,
    ),
    # only the groups with more than one row
    (
        "SELECT p.id, p.name, SUM(u.n) FROM people AS p JOIN (SELECT g.w1 AS who, COUNT(*) AS n FROM games AS g GROUP BY g.w1 HAVING COUNT(*) > 1"
        " UNION ALL SELECT g.w2, COUNT(*) FROM games AS g GROUP BY g.w2) AS u ON p.id = u.who GROUP BY p.id",
        FLAT,
    ),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_counting_the_groups_really_differs():
    tables = {"people": [(1, "a")], "games": [(1, 1, None, None), (2, 1, 1, None)]}
    groups, wins = run(
        tables,
        f"SELECT p.id, COUNT(*) FROM people AS p JOIN {COUNTED} ON p.id = u.who GROUP BY p.id",
        f"SELECT p.id, SUM(u.n) FROM people AS p JOIN {COUNTED} ON p.id = u.who GROUP BY p.id",
    )
    assert groups == ["(1, 2)"] and wins == ["(1, 3)"]


QUERIES = [
    f"SELECT p.id, p.name, SUM(u.n) AS wins FROM people AS p JOIN {COUNTED} ON p.id = u.who GROUP BY p.id, p.name",
    "SELECT SUM(n) FROM (SELECT w1 AS who, COUNT(1) AS n FROM games GROUP BY w1 UNION ALL SELECT w2, COUNT(*) FROM games GROUP BY w2) AS u",
    "SELECT who, SUM(n) FROM (SELECT w1 AS who, COUNT(w3) AS n FROM games GROUP BY w1 UNION ALL SELECT w2, SUM(IF(w3 > 1, 2, 0)) FROM games WHERE day > 1 GROUP BY w2"
    " UNION ALL SELECT w3, 5 FROM games) AS u WHERE who > 0 OR who IS NULL GROUP BY who HAVING SUM(n) > 1",
    "SELECT u.id, u.name, SUM(u.n) FROM (SELECT p.id, p.name, COUNT(g.w3) AS n FROM people AS p, games AS g WHERE p.id = g.w1 GROUP BY p.id, p.name"
    " UNION ALL SELECT p.id, p.name, COUNT(*) FROM people AS p, games AS g WHERE p.id = g.w2 GROUP BY p.id, p.name) AS u GROUP BY u.id, u.name",
    f"SELECT n.name, SUM(u.n), SUM(u.n) + 1 FROM notes AS n, {COUNTED} WHERE n.id = u.who OR u.who IS NULL GROUP BY n.name",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the rewritten one on random tables that meet the declared facts (the two keys)."""

    after = rewritten(sql)
    assert after is not None
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
