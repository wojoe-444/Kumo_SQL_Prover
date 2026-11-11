"""A ``UNION`` whose branches cannot return the same row twice, read as ``UNION ALL`` (``kumo_sql_prover.keyed_union_branches``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.keyed_union_branches import union_of_distinct_rows
from kumo_sql_prover.smt_equivalence import TableConstraints

# people: id is the key; games: day is the key, w1..w3 name people and may be NULL; notes has no key
SCHEMA = {"people": ["id", "name"], "games": ["day", "w1", "w2", "w3"], "notes": ["id", "name"]}
KEYS = {"people": [("id",)], "games": [("day",)]}
NOT_NULL = {"people": frozenset({"id"}), "games": frozenset({"day"})}
CONSTRAINTS = {
    "people": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
    "games": TableConstraints(not_null=frozenset({"day"}), keys=(("day",),)),
}
TAGGED = "SELECT day, 'first' AS slot, w1 AS who FROM games UNION SELECT day, 'second', w2 FROM games UNION SELECT day, 'third', w3 FROM games"
TAGGED_ALL = TAGGED.replace("UNION", "UNION ALL")
SUM3 = "SUM(CASE WHEN p.id = g.w1 THEN 1 ELSE 0 END + CASE WHEN p.id = g.w2 THEN 1 ELSE 0 END + CASE WHEN p.id = g.w3 THEN 1 ELSE 0 END)"
FLAT = f"SELECT p.id, p.name, {SUM3} AS n FROM people AS p CROSS JOIN games AS g GROUP BY p.id HAVING {SUM3} > 0"


def rewritten(sql, keys=KEYS, not_null=NOT_NULL):
    after = union_of_distinct_rows(sqlglot.parse_one(sql, read="mysql"), SCHEMA, keys, not_null)
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


def test_a_union_of_tagged_keyed_branches_is_written_as_union_all():
    assert rewritten(f"SELECT u.who, COUNT(*) FROM ({TAGGED}) AS u GROUP BY u.who") == f"SELECT u.who, COUNT(*) FROM ({TAGGED_ALL}) AS u GROUP BY u.who"
    # the select around it is left as it is, whatever it does
    assert rewritten(f"SELECT DISTINCT p.name FROM people AS p LEFT JOIN ({TAGGED}) AS u ON u.who = p.id WHERE u.day > 1 ORDER BY 1") == (
        f"SELECT DISTINCT p.name FROM people AS p LEFT JOIN ({TAGGED_ALL}) AS u ON u.who = p.id WHERE u.day > 1 ORDER BY 1"
    )
    # whole numbers tell the branches apart as well, and a filter keeps a branch's rows apart
    assert rewritten("SELECT * FROM (SELECT g.day, 1 AS slot, g.w1 FROM games AS g WHERE g.w1 > 2 UNION SELECT g.day, 2, g.w2 FROM games AS g) AS u") == (
        "SELECT * FROM (SELECT g.day, 1 AS slot, g.w1 FROM games AS g WHERE g.w1 > 2 UNION ALL SELECT g.day, 2, g.w2 FROM games AS g) AS u"
    )
    # branches in parentheses, and a UNION ALL among the UNIONs
    assert rewritten("SELECT * FROM ((SELECT day, 'a' AS t FROM games) UNION ALL (SELECT day, 'b' FROM games) UNION (SELECT day, 'c' FROM games)) AS u") == (
        "SELECT * FROM ((SELECT day, 'a' AS t FROM games) UNION ALL (SELECT day, 'b' FROM games) UNION ALL (SELECT day, 'c' FROM games)) AS u"
    )
    # a branch over two tables returns the keys of both
    assert rewritten(
        "SELECT * FROM (SELECT p.id, g.day, 'a' AS t FROM people AS p JOIN games AS g ON p.id = g.w1 UNION SELECT p.id, g.day, 'b' FROM people AS p, games AS g) AS u"
    ) == "SELECT * FROM (SELECT p.id, g.day, 'a' AS t FROM people AS p JOIN games AS g ON p.id = g.w1 UNION ALL SELECT p.id, g.day, 'b' FROM people AS p, games AS g) AS u"
    # different pairs of branches may differ at different positions
    assert rewritten("SELECT * FROM (SELECT day, 'a' AS s, 1 AS t FROM games UNION SELECT day, 'a', 2 FROM games UNION SELECT day, 'b', 2 FROM games) AS u") == (
        "SELECT * FROM (SELECT day, 'a' AS s, 1 AS t FROM games UNION ALL SELECT day, 'a', 2 FROM games UNION ALL SELECT day, 'b', 2 FROM games) AS u"
    )


@pytest.mark.parametrize("sql", [
    "SELECT * FROM (SELECT 'first' AS slot, w1 AS who FROM games UNION SELECT 'second', w2 FROM games) AS u",  # no key: a branch repeats a row
    "SELECT * FROM (SELECT day, w1 AS who FROM games UNION SELECT day, w2 FROM games) AS u",  # nothing tells the branches apart
    "SELECT * FROM (SELECT day, 'a' AS slot, w1 AS who FROM games UNION SELECT day, 'a', w2 FROM games) AS u",  # the same literal
    "SELECT * FROM (SELECT day, 'a' AS slot, w1 AS who FROM games UNION SELECT day, 'A ', w2 FROM games) AS u",  # the same to a collation
    "SELECT * FROM (SELECT day, 'a' AS slot, w1 AS who FROM games UNION SELECT day, 1, w2 FROM games) AS u",  # a string against a number
    "SELECT * FROM (SELECT day, 'a' AS slot, w1 AS who FROM games UNION SELECT day, 'b', w2 FROM games UNION SELECT day, name, id FROM people) AS u",  # a column
    "SELECT * FROM (SELECT day, 1.0 AS slot, w1 AS who FROM games UNION SELECT day, 1.00, w2 FROM games) AS u",  # not whole numbers
    "SELECT * FROM (SELECT day + 0 AS day, 'a' AS slot, w1 AS who FROM games UNION SELECT day, 'b', w2 FROM games) AS u",  # the key is not returned itself
    "SELECT * FROM (SELECT id, 'a' AS slot FROM notes UNION SELECT id, 'b' FROM notes) AS u",  # notes has no key
    "SELECT * FROM (SELECT g.day, 'a' AS slot FROM games AS g, people AS p UNION SELECT day, 'b' FROM games) AS u",  # each game once per person
    "SELECT * FROM (SELECT g.day, 'a' AS slot FROM games AS g LEFT JOIN people AS p ON p.id = g.w1 UNION SELECT day, 'b' FROM games) AS u",
    "SELECT * FROM (SELECT DISTINCT day, 'a' AS slot FROM games UNION SELECT day, 'b' FROM games) AS u",
    "SELECT * FROM (SELECT day, 'a' AS slot FROM games GROUP BY day UNION SELECT day, 'b' FROM games) AS u",
    "SELECT * FROM (SELECT day, 'a' AS slot FROM games LIMIT 2 UNION SELECT day, 'b' FROM games) AS u",
    "SELECT * FROM (SELECT day, 'a' AS slot FROM (SELECT * FROM games) AS g UNION SELECT day, 'b' FROM games) AS u",
    "SELECT * FROM (SELECT *, 'a' AS slot FROM games UNION SELECT *, 'b' FROM games) AS u",  # a star is not read
    "SELECT * FROM (SELECT day, 'a' AS slot FROM games EXCEPT SELECT day, 'b' FROM games) AS u",
    "SELECT * FROM (SELECT day, 'a' AS slot FROM games UNION SELECT day, 'b' FROM games ORDER BY 1 LIMIT 3) AS u",
    "SELECT * FROM (SELECT day, 'a' AS slot FROM games UNION ALL SELECT day, 'b' FROM games) AS u",  # nothing to do
    "SELECT * FROM (SELECT day, 'a' AS slot FROM games) AS u",
    "SELECT day, 'a' AS slot FROM games",
])
def test_other_unions_stay(sql):
    assert rewritten(sql) is None


def test_the_key_and_its_not_null_columns_are_needed():
    sql = f"SELECT * FROM ({TAGGED}) AS u"
    assert rewritten(sql) is not None
    assert rewritten(sql, keys=None) is None
    assert rewritten(sql, keys={"people": [("id",)]}) is None
    assert rewritten(sql, not_null=None) is None
    assert rewritten(sql, not_null={"people": frozenset({"id"})}) is None  # a key that may be NULL repeats


PROOFS = [
    (f"SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN ({TAGGED}) AS u ON p.id = u.who GROUP BY p.id", FLAT),
    (f"SELECT p.id, p.name, COUNT(u.slot) FROM people AS p, ({TAGGED}) AS u WHERE p.id = u.who GROUP BY p.id, p.name", FLAT),
    (
        f"SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN ({TAGGED}) AS u ON p.id = u.who GROUP BY p.id",
        f"SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN ({TAGGED_ALL}) AS u ON p.id = u.who GROUP BY p.id",
    ),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


UNTAGGED = "SELECT day, w1 AS who FROM games UNION SELECT day, w2 FROM games UNION SELECT day, w3 FROM games"
NOT_PROVED = [
    # one win a day at most: the UNION drops a second win of the same person on the same day
    (f"SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN ({UNTAGGED}) AS u ON p.id = u.who GROUP BY p.id", FLAT),
    # without the day, one win per slot at most
    (
        "SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN (SELECT 'first' AS slot, w1 AS who FROM games UNION SELECT 'second', w2 FROM games"
        " UNION SELECT 'third', w3 FROM games) AS u ON p.id = u.who GROUP BY p.id",
        FLAT,
    ),
    # the outer join keeps the people without a win
    (f"SELECT p.id, p.name, COUNT(u.who) FROM people AS p LEFT JOIN ({TAGGED}) AS u ON p.id = u.who GROUP BY p.id", FLAT),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_untagged_branches_really_lose_rows():
    tables = {"people": [(1, "a")], "games": [(1, 1, 1, None), (2, 1, None, None)]}
    union, union_all = run(
        tables,
        f"SELECT p.id, COUNT(*) FROM people AS p JOIN ({UNTAGGED}) AS u ON p.id = u.who GROUP BY p.id",
        f"SELECT p.id, COUNT(*) FROM people AS p JOIN ({UNTAGGED.replace('UNION', 'UNION ALL')}) AS u ON p.id = u.who GROUP BY p.id",
    )
    assert union == ["(1, 2)"] and union_all == ["(1, 3)"]


QUERIES = [
    f"SELECT u.who, u.slot, COUNT(*), MIN(u.day) FROM ({TAGGED}) AS u GROUP BY u.who, u.slot",
    f"SELECT p.id, p.name, u.day, u.slot FROM people AS p LEFT JOIN ({TAGGED}) AS u ON u.who = p.id",
    "SELECT * FROM (SELECT g.day, 1 AS slot, g.w1 FROM games AS g WHERE g.w1 > 2 UNION SELECT g.day, 2, g.w2 FROM games AS g UNION ALL SELECT g.day, 3, NULL FROM games AS g) AS u",
    "SELECT u.t, COUNT(*), COUNT(u.name) FROM (SELECT p.id, p.name, g.day, 'a' AS t FROM people AS p JOIN games AS g ON p.id = g.w1"
    " UNION SELECT p.id, p.name, g.day, 'b' FROM people AS p, games AS g WHERE g.w2 IS NULL) AS u GROUP BY u.t",
    "SELECT u.s, u.t, COUNT(*) FROM (SELECT day, 'a' AS s, 1 AS t, w1 FROM games UNION SELECT day, 'a', 2, w1 FROM games UNION SELECT day, 'b', 2, w2 FROM games) AS u GROUP BY u.s, u.t",
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
        }
        before_rows, after_rows = run(tables, sql, after)
        assert before_rows == after_rows, (tables, sql, after)
