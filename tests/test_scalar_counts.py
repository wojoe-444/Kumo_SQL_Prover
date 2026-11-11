"""A filtered scalar count per row of a keyed table, read as a grouped count over a join (``kumo_sql_prover.scalar_counts``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.scalar_counts import scalar_counts_to_grouped_join
from kumo_sql_prover.smt_equivalence import TableConstraints

# people: id is the key; games: day is the key, w1..w3 name people and may be NULL; notes has no key
SCHEMA = {"people": ["id", "name"], "games": ["day", "w1", "w2", "w3"], "notes": ["id", "name"]}
KEYS = {"people": [("id",)], "games": [("day",)]}
NOT_NULL = {"people": frozenset({"id"}), "games": frozenset({"day"})}
CONSTRAINTS = {
    "people": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
    "games": TableConstraints(not_null=frozenset({"day"}), keys=(("day",),)),
}
SUM2 = "SUM(CASE WHEN p.id = g.w1 THEN 1 ELSE 0 END + CASE WHEN p.id = g.w2 THEN 1 ELSE 0 END)"
FLAT = f"SELECT p.id, p.name, {SUM2} AS n FROM people AS p CROSS JOIN games AS g GROUP BY p.id HAVING {SUM2} > 0"
ONE = "(SELECT SUM(CASE WHEN w1 = id THEN 1 ELSE 0 END) + SUM(CASE WHEN w2 = id THEN 1 ELSE 0 END) FROM games)"
TWO = "(SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) + (SELECT COUNT(g.w2) FROM games AS g WHERE g.w2 = p.id)"


def rewritten(sql, keys=KEYS, not_null=NOT_NULL):
    after = scalar_counts_to_grouped_join(sqlglot.parse_one(sql, read="mysql"), SCHEMA, keys, not_null)
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


def test_one_nested_query_becomes_the_aggregates_of_a_grouped_join():
    assert rewritten("SELECT p.id, p.name, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p HAVING wins > 0") == (
        "SELECT p.id, p.name, COUNT(*) AS wins FROM people AS p CROSS JOIN games AS g WHERE g.w1 = p.id GROUP BY p.id HAVING COUNT(*) > 0"
    )
    # bare names: the nested query's own table first, then the outer one
    assert rewritten(f"SELECT id, name, {ONE} AS wins FROM people WHERE wins <> 0") == (
        "SELECT people.id, people.name, SUM(CASE WHEN games.w1 = people.id THEN 1 ELSE 0 END) + SUM(CASE WHEN games.w2 = people.id THEN 1 ELSE 0 END) AS wins"
        " FROM people CROSS JOIN games GROUP BY people.id"
        " HAVING SUM(CASE WHEN games.w1 = people.id THEN 1 ELSE 0 END) + SUM(CASE WHEN games.w2 = people.id THEN 1 ELSE 0 END) <> 0"
    )
    # a sum of a column, a join inside, another output of the outer table
    assert rewritten(
        "SELECT p.name, UPPER(p.name) AS loud, (SELECT SUM(g.w3) FROM games AS g JOIN notes AS n ON n.id = g.w2 WHERE g.w1 = p.id) AS total FROM people AS p HAVING total >= 1"
    ) == (
        "SELECT p.name, UPPER(p.name) AS loud, SUM(g.w3) AS total FROM people AS p CROSS JOIN games AS g CROSS JOIN notes AS n"
        " WHERE g.w1 = p.id AND n.id = g.w2 GROUP BY p.id HAVING SUM(g.w3) >= 1"
    )


def test_the_same_through_a_derived_table():
    assert rewritten(f"SELECT d.id, d.name, d.wins AS n FROM (SELECT id, name, {ONE} AS wins FROM people) AS d WHERE d.wins > 0") == (
        "SELECT people.id, people.name, SUM(CASE WHEN games.w1 = people.id THEN 1 ELSE 0 END) + SUM(CASE WHEN games.w2 = people.id THEN 1 ELSE 0 END) AS n"
        " FROM people CROSS JOIN games GROUP BY people.id"
        " HAVING SUM(CASE WHEN games.w1 = people.id THEN 1 ELSE 0 END) + SUM(CASE WHEN games.w2 = people.id THEN 1 ELSE 0 END) > 0"
    )
    assert rewritten("SELECT wins, id AS who FROM (SELECT p.id, (SELECT COUNT(g.w3) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p) AS d WHERE wins <> 0") == (
        "SELECT COUNT(g.w3) AS wins, p.id AS who FROM people AS p CROSS JOIN games AS g WHERE g.w1 = p.id GROUP BY p.id HAVING COUNT(g.w3) <> 0"
    )


def test_a_sum_of_counting_queries_becomes_one_sum_of_indicators():
    assert rewritten(f"SELECT p.id, p.name, {TWO} AS n FROM people AS p HAVING n > 0") == FLAT.replace("p.id = g.w1", "g.w1 = p.id").replace("p.id = g.w2", "g.w2 = p.id")
    # different aliases for the counted table, a counted column the condition does not compare, no condition at all
    assert rewritten(
        "SELECT id, n FROM (SELECT p.id, ((SELECT COUNT(*) FROM games WHERE w1 = p.id) + (SELECT COUNT(x.w3) FROM games AS x WHERE x.w2 = p.id)"
        " + (SELECT COUNT(1) FROM games AS y)) AS n FROM people AS p) AS t WHERE t.n >= 1"
    ) == (
        "SELECT p.id, SUM(CASE WHEN games.w1 = p.id THEN 1 ELSE 0 END + CASE WHEN games.w2 = p.id AND NOT games.w3 IS NULL THEN 1 ELSE 0 END + 1) AS n"
        " FROM people AS p CROSS JOIN games GROUP BY p.id"
        " HAVING SUM(CASE WHEN games.w1 = p.id THEN 1 ELSE 0 END + CASE WHEN games.w2 = p.id AND NOT games.w3 IS NULL THEN 1 ELSE 0 END + 1) >= 1"
    )


@pytest.mark.parametrize("sql", [
    "SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p",  # no filter: people without games stay
    "SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p HAVING wins >= 0",  # keeps the zeros
    "SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p HAVING wins < 2",
    "SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p HAVING wins = 0",
    "SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p HAVING wins > 0 AND p.id > 1",
    "SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p WHERE p.id > 1 HAVING wins > 0",
    "SELECT n.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = n.id) AS wins FROM notes AS n HAVING wins > 0",  # notes has no key
    "SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS name FROM people AS p HAVING name > 0",  # the name is a column
    "SELECT p.id, (SELECT COUNT(*) FROM games AS p WHERE p.w1 = 1) AS wins FROM people AS p HAVING wins > 0",  # one alias for both
    "SELECT p.id, (SELECT MAX(g.day) FROM games AS g WHERE g.w1 = p.id) AS last FROM people AS p HAVING last > 0",
    "SELECT p.id, (SELECT COUNT(DISTINCT g.w2) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p HAVING wins > 0",
    "SELECT p.id, (SELECT COUNT(*) - 1 FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p HAVING wins > 0",
    "SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id GROUP BY g.w2) AS wins FROM people AS p HAVING wins > 0",
    "SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id LIMIT 1) AS wins FROM people AS p HAVING wins > 0",
    "SELECT p.id, (SELECT COUNT(*) FROM games AS g LEFT JOIN notes AS n ON n.id = g.w2 WHERE g.w1 = p.id) AS wins FROM people AS p HAVING wins > 0",
    "SELECT p.id, (SELECT COUNT(*) FROM (SELECT * FROM games) AS g WHERE g.w1 = p.id) AS wins FROM people AS p HAVING wins > 0",
    "SELECT p.id, (SELECT g.w2 FROM games AS g WHERE g.day = p.id) AS wins FROM people AS p HAVING wins > 0",  # not an aggregate
    # a sum of nested queries: a SUM is NULL where a filtered count is 0, and the tables must be one
    "SELECT p.id, (SELECT SUM(1) FROM games AS g WHERE g.w1 = p.id) + (SELECT COUNT(*) FROM games AS g WHERE g.w2 = p.id) AS wins FROM people AS p HAVING wins > 0",
    "SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) + (SELECT COUNT(*) FROM notes AS n WHERE n.id = p.id) AS wins FROM people AS p HAVING wins > 0",
    "SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) - (SELECT COUNT(*) FROM games AS g WHERE g.w2 = p.id) AS wins FROM people AS p HAVING wins > 0",
    "SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) + 1 AS wins FROM people AS p HAVING wins > 0",
    # two items with nested queries, grouping, DISTINCT, a join outside
    "SELECT (SELECT COUNT(*) FROM games) AS a, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p HAVING wins > 0",
    "SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p GROUP BY p.name HAVING wins > 0",
    "SELECT DISTINCT p.name, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p HAVING wins > 0",
    "SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p, notes AS n HAVING wins > 0",
    "SELECT d.id FROM (SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p) AS d WHERE d.wins > 0",  # the count is not returned
    "SELECT d.id, d.wins FROM (SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p WHERE p.id > 1) AS d WHERE d.wins > 0",
    "SELECT d.id, d.wins FROM (SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p) AS d WHERE d.id > 0",
    "SELECT d.id, d.wins FROM (SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p) AS d",
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_the_key_and_its_not_null_columns_are_needed():
    sql = "SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p HAVING wins > 0"
    assert rewritten(sql) is not None
    assert rewritten(sql, keys=None) is None
    assert rewritten(sql, keys={"games": [("day",)]}) is None
    assert rewritten(sql, not_null=None) is None
    assert rewritten(sql, not_null={"games": frozenset({"day"})}) is None


PROOFS = [
    (f"SELECT p.id, p.name, {TWO} AS n FROM people AS p HAVING n > 0", FLAT),
    (f"SELECT id, name, n FROM (SELECT id, name, {ONE} AS n FROM people) AS d WHERE n > 0", FLAT),
    (f"SELECT id, name, {ONE} AS n FROM people HAVING n <> 0", FLAT),
    (
        "SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p HAVING wins > 0",
        "SELECT p.id, COUNT(*) FROM people AS p JOIN games AS g ON g.w1 = p.id GROUP BY p.id",
    ),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # without the filter the people who never won are returned, with a 0
    (f"SELECT p.id, p.name, {TWO} AS n FROM people AS p", FLAT),
    (f"SELECT p.id, p.name, {TWO} AS n FROM people AS p HAVING n >= 0", FLAT),
    # another count
    (f"SELECT p.id, p.name, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS n FROM people AS p HAVING n > 0", FLAT),
    # notes may hold a row twice: the grouped select would return it once
    (
        "SELECT n.id, n.name, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = n.id) AS wins FROM notes AS n HAVING wins > 0",
        "SELECT n.id, n.name, COUNT(*) FROM notes AS n JOIN games AS g ON g.w1 = n.id GROUP BY n.id, n.name",
    ),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_the_filter_and_the_key_really_matter():
    tables = {"people": [(1, "a"), (2, "b")], "games": [(1, 1, None, None)], "notes": [(1, "x"), (1, "x")]}
    unfiltered, joined, per_note, grouped_notes = run(
        tables,
        "SELECT p.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = p.id) AS wins FROM people AS p",
        "SELECT p.id, COUNT(*) FROM people AS p JOIN games AS g ON g.w1 = p.id GROUP BY p.id",
        "SELECT * FROM (SELECT n.id, (SELECT COUNT(*) FROM games AS g WHERE g.w1 = n.id) AS wins FROM notes AS n) AS d WHERE wins > 0",
        "SELECT n.id, COUNT(*) FROM notes AS n JOIN games AS g ON g.w1 = n.id GROUP BY n.id",
    )
    assert unfiltered == ["(1, 1)", "(2, 0)"] and joined == ["(1, 1)"]
    assert per_note == ["(1, 1)", "(1, 1)"] and grouped_notes == ["(1, 2)"]


# DuckDB has no HAVING without GROUP BY and no alias in WHERE: the differential test writes the filter through a derived table
QUERIES = [
    f"SELECT d.id, d.name, d.n FROM (SELECT p.id, p.name, {TWO} AS n FROM people AS p) AS d WHERE d.n > 0",
    f"SELECT id, name, n AS wins FROM (SELECT id, name, {ONE} AS n FROM people) AS d WHERE n <> 0",
    "SELECT d.name, d.total FROM (SELECT p.name, (SELECT SUM(g.w3) FROM games AS g JOIN notes AS n ON n.id = g.w2 WHERE g.w1 = p.id) AS total FROM people AS p) AS d WHERE d.total >= 1",
    "SELECT d.id, d.n FROM (SELECT p.id, (SELECT COUNT(g.w3) + SUM(g.w2) + COUNT(*) FROM games AS g WHERE g.w1 = p.id OR g.w1 IS NULL) AS n FROM people AS p) AS d WHERE d.n > 0",
    "SELECT id, n FROM (SELECT p.id, ((SELECT COUNT(*) FROM games WHERE w1 = p.id) + (SELECT COUNT(x.w3) FROM games AS x WHERE x.w2 = p.id)"
    " + (SELECT COUNT(1) FROM games AS y)) AS n FROM people AS p) AS t WHERE t.n >= 1",
    "SELECT id, n FROM (SELECT p.id, ((SELECT COUNT(w3) FROM games WHERE w1 <> p.id) + (SELECT COUNT(x.w3) FROM games AS x WHERE x.w2 = p.id OR x.w3 > p.id)) AS n"
    " FROM people AS p) AS t WHERE t.n <> 0",
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
            "notes": [(rng.choice([None, 1, 2, 3]), rng.choice([None, "x", "y"])) for _ in range(rng.choice([0, 1, 3]))],
        }
        # DuckDB wants the name in the GROUP BY; id is the key of people, so the groups are the same
        strict = after.replace("GROUP BY p.id", "GROUP BY p.id, p.name").replace("GROUP BY people.id", "GROUP BY people.id, people.name")
        before_rows, after_rows = run(tables, sql, strict)
        assert before_rows == after_rows, (tables, sql, after)
