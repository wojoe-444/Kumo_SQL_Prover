"""A grouped count over a ``UNION ALL`` of reads of one source, as a sum of indicators (``kumo_sql_prover.union_all_counts``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import TableConstraints
from kumo_sql_prover.union_all_counts import count_over_union_all

# people: id is the key; games: day is the key, w1..w3 name people and may be NULL; notes has no key
SCHEMA = {"people": ["id", "name"], "games": ["day", "w1", "w2", "w3"], "notes": ["id", "name"]}
KEYS = {"people": [("id",)], "games": [("day",)]}
NOT_NULL = {"people": frozenset({"id"}), "games": frozenset({"day"})}
CONSTRAINTS = {
    "people": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
    "games": TableConstraints(not_null=frozenset({"day"}), keys=(("day",),)),
}
U2 = "(SELECT g.w1 AS who FROM games AS g UNION ALL SELECT g.w2 AS who FROM games AS g) AS u"
U3 = "(SELECT w1 AS who FROM games UNION ALL SELECT w2 FROM games UNION ALL SELECT w3 FROM games) AS u"
SUM2 = "SUM(CASE WHEN p.id = g.w1 THEN 1 ELSE 0 END + CASE WHEN p.id = g.w2 THEN 1 ELSE 0 END)"
FLAT = f"SELECT p.id, p.name, {SUM2} AS n FROM people AS p CROSS JOIN games AS g GROUP BY p.id HAVING {SUM2} > 0"


def rewritten(sql, keys=KEYS, not_null=NOT_NULL, schema=SCHEMA):
    before = sqlglot.parse_one(sql, read="mysql")
    after = count_over_union_all(before, schema, keys, not_null)
    assert before.sql(dialect="mysql") == sqlglot.parse_one(sql, read="mysql").sql(dialect="mysql")  # the select itself is left alone
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


def test_a_count_over_an_unpivoting_union_becomes_a_sum_of_indicators():
    assert rewritten(f"SELECT p.id, p.name, COUNT(*) AS n FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id") == FLAT
    # the union's column in the select list and in the grouping is the column it equals
    assert rewritten(f"SELECT u.who, p.name, COUNT(*) AS n FROM {U2} JOIN people AS p ON u.who = p.id GROUP BY u.who") == (
        "SELECT p.id AS who, p.name, SUM(CASE WHEN g.w1 = p.id THEN 1 ELSE 0 END + CASE WHEN g.w2 = p.id THEN 1 ELSE 0 END) AS n FROM games AS g CROSS JOIN people AS p"
        " GROUP BY p.id HAVING SUM(CASE WHEN g.w1 = p.id THEN 1 ELSE 0 END + CASE WHEN g.w2 = p.id THEN 1 ELSE 0 END) > 0"
    )
    # a comma join with the condition in the WHERE, bare columns, three branches without aliases
    assert rewritten(f"SELECT id, name, COUNT(1) FROM people, {U3} WHERE who = id GROUP BY id") == (
        "SELECT people.id, people.name, SUM(CASE WHEN games.w1 = people.id THEN 1 ELSE 0 END + CASE WHEN games.w2 = people.id THEN 1 ELSE 0 END"
        " + CASE WHEN games.w3 = people.id THEN 1 ELSE 0 END) FROM people CROSS JOIN games GROUP BY people.id HAVING SUM(CASE WHEN games.w1 = people.id"
        " THEN 1 ELSE 0 END + CASE WHEN games.w2 = people.id THEN 1 ELSE 0 END + CASE WHEN games.w3 = people.id THEN 1 ELSE 0 END) > 0"
    )


def test_counts_of_a_column_and_sums_of_literals():
    # COUNT(u.who): the join condition already rejects a NULL who
    assert rewritten(f"SELECT p.id, p.name, COUNT(u.who) AS n FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id") == FLAT
    assert rewritten(f"SELECT p.id, p.name, COUNT(p.id) AS n FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id") == FLAT
    # a column that may be NULL is tested per branch, a literal carried by the branches is summed
    assert rewritten(
        "SELECT p.id, COUNT(u.other), SUM(u.k) FROM people AS p JOIN (SELECT g.w1 AS who, g.w2 AS other, 1 AS k FROM games AS g"
        " UNION ALL SELECT g.w2, g.w3, 2 FROM games AS g WHERE g.day > 3) AS u ON p.id = u.who GROUP BY p.id"
    ) == (
        "SELECT p.id, SUM(CASE WHEN p.id = g.w1 AND NOT g.w2 IS NULL THEN 1 ELSE 0 END + CASE WHEN p.id = g.w2 AND g.day > 3 AND NOT g.w3 IS NULL THEN 1 ELSE 0 END),"
        " SUM(CASE WHEN p.id = g.w1 THEN 1 ELSE 0 END + CASE WHEN p.id = g.w2 AND g.day > 3 THEN 2 ELSE 0 END) FROM people AS p CROSS JOIN games AS g GROUP BY p.id"
        " HAVING SUM(CASE WHEN p.id = g.w1 THEN 1 ELSE 0 END + CASE WHEN p.id = g.w2 AND g.day > 3 THEN 1 ELSE 0 END) > 0"
    )
    # a NULL test inside a summed value: settled where the branch's condition compares the column, kept where it does not
    assert rewritten(
        "SELECT p.id, SUM(u.k) FROM people AS p JOIN (SELECT g.w1 AS who, CASE WHEN g.w1 IS NULL THEN 0 ELSE 1 END AS k FROM games AS g"
        " UNION ALL SELECT g.w2, CASE WHEN g.w3 IS NULL THEN 0 ELSE 1 END FROM games AS g) AS u ON p.id = u.who GROUP BY p.id"
    ) == (
        "SELECT p.id, SUM(CASE WHEN p.id = g.w1 THEN 1 ELSE 0 END + CASE WHEN p.id = g.w2 THEN (CASE WHEN g.w3 IS NULL THEN 0 ELSE 1 END) ELSE 0 END)"
        " FROM people AS p CROSS JOIN games AS g GROUP BY p.id"
        " HAVING SUM(CASE WHEN p.id = g.w1 THEN 1 ELSE 0 END + CASE WHEN p.id = g.w2 THEN 1 ELSE 0 END) > 0"
    )


def test_a_union_of_joins_is_read_through_its_own_conditions():
    joined = "SELECT p.id, p.name FROM people AS p JOIN games AS g ON p.id = g.{0}"
    union = f"({joined.format('w1')} UNION ALL {joined.format('w2')}) AS u"
    assert rewritten(f"SELECT u.id, u.name, COUNT(*) AS n FROM {union} GROUP BY u.id") == FLAT
    # a branch output that the branch equates with one and the same column is that column
    by_winner = "(SELECT g.w1 AS who, p.name FROM games AS g, people AS p WHERE g.w1 = p.id UNION ALL SELECT g.w2, p.name FROM games AS g, people AS p WHERE g.w2 = p.id) AS u"
    assert rewritten(f"SELECT who, name, COUNT(*) AS n FROM {by_winner} GROUP BY who, name") == (
        "SELECT p.id AS who, p.name, SUM(CASE WHEN g.w1 = p.id THEN 1 ELSE 0 END + CASE WHEN g.w2 = p.id THEN 1 ELSE 0 END) AS n FROM games AS g CROSS JOIN people AS p"
        " GROUP BY p.id, p.name HAVING SUM(CASE WHEN g.w1 = p.id THEN 1 ELSE 0 END + CASE WHEN g.w2 = p.id THEN 1 ELSE 0 END) > 0"
    )


def test_other_conditions_having_and_order_stay():
    assert rewritten(f"SELECT p.id, COUNT(*) AS n FROM people AS p JOIN {U2} ON p.id = u.who WHERE p.name <> 'x' GROUP BY p.id HAVING COUNT(*) > 1 ORDER BY p.id") == (
        f"SELECT p.id, {SUM2} AS n FROM people AS p CROSS JOIN games AS g WHERE p.name <> 'x' GROUP BY p.id HAVING {SUM2} > 1 AND {SUM2} > 0 ORDER BY p.id"
    )
    # a HAVING that says what the added test says is not repeated
    assert rewritten(f"SELECT p.id, COUNT(*) AS n FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id HAVING COUNT(*) >= 1") == (
        f"SELECT p.id, {SUM2} AS n FROM people AS p CROSS JOIN games AS g GROUP BY p.id HAVING {SUM2} > 0"
    )
    assert rewritten(f"SELECT p.id, COUNT(*) AS n FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id HAVING n <> 0") == (
        f"SELECT p.id, {SUM2} AS n FROM people AS p CROSS JOIN games AS g GROUP BY p.id HAVING {SUM2} > 0"
    )
    # a link that is not an equality, with the union's column only inside the count
    assert rewritten(f"SELECT p.id, COUNT(*) FROM people AS p JOIN {U2} ON p.id < u.who GROUP BY p.id") == (
        "SELECT p.id, SUM(CASE WHEN p.id < g.w1 THEN 1 ELSE 0 END + CASE WHEN p.id < g.w2 THEN 1 ELSE 0 END) FROM people AS p CROSS JOIN games AS g GROUP BY p.id"
        " HAVING SUM(CASE WHEN p.id < g.w1 THEN 1 ELSE 0 END + CASE WHEN p.id < g.w2 THEN 1 ELSE 0 END) > 0"
    )


def test_names_that_would_collide_or_be_ambiguous():
    # the union's table is named like a source of the select
    assert rewritten("SELECT p.id, COUNT(*) FROM people AS p JOIN (SELECT p.w1 AS who FROM games AS p UNION ALL SELECT p.w2 FROM games AS p) AS u ON p.id = u.who GROUP BY p.id") == (
        "SELECT p.id, SUM(CASE WHEN p.id = kqu0.w1 THEN 1 ELSE 0 END + CASE WHEN p.id = kqu0.w2 THEN 1 ELSE 0 END) FROM people AS p CROSS JOIN games AS kqu0 GROUP BY p.id"
        " HAVING SUM(CASE WHEN p.id = kqu0.w1 THEN 1 ELSE 0 END + CASE WHEN p.id = kqu0.w2 THEN 1 ELSE 0 END) > 0"
    )
    # GROUP BY id: both sources have an id, and the join makes them equal
    assert rewritten("SELECT p.id, COUNT(*) FROM people AS p JOIN (SELECT w1 AS id FROM games UNION ALL SELECT w2 FROM games) AS u ON p.id = u.id GROUP BY id") == (
        "SELECT p.id, SUM(CASE WHEN p.id = games.w1 THEN 1 ELSE 0 END + CASE WHEN p.id = games.w2 THEN 1 ELSE 0 END) FROM people AS p CROSS JOIN games GROUP BY p.id"
        " HAVING SUM(CASE WHEN p.id = games.w1 THEN 1 ELSE 0 END + CASE WHEN p.id = games.w2 THEN 1 ELSE 0 END) > 0"
    )


@pytest.mark.parametrize("sql", [
    f"SELECT p.id, COUNT(*) FROM people AS p JOIN {U2.replace('UNION ALL', 'UNION')} ON p.id = u.who GROUP BY p.id",  # UNION removes repeated rows
    f"SELECT p.id, COUNT(*) FROM people AS p JOIN {U2.replace('UNION ALL', 'UNION DISTINCT')} ON p.id = u.who GROUP BY p.id",
    f"SELECT u.who, COUNT(*) FROM {U2} LEFT JOIN people AS p ON p.id = u.who GROUP BY u.who",  # a NULL who keeps its group
    f"SELECT p.id, COUNT(u.who) FROM people AS p LEFT JOIN {U2} ON p.id = u.who GROUP BY p.id",  # so does a person nobody names
    f"SELECT p.id, COUNT(*) FROM {U2} RIGHT JOIN people AS p ON p.id = u.who GROUP BY p.id",
    f"SELECT COUNT(*) FROM people AS p JOIN {U2} ON p.id = u.who",  # a global count returns a row over no rows
    f"SELECT p.id, COUNT(*) FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id LIMIT 2",
    f"SELECT p.id, COUNT(*) FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id WITH ROLLUP",
    f"SELECT p.id, MIN(u.who) FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id",
    f"SELECT p.id, COUNT(DISTINCT u.who) FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id",
    f"SELECT p.id, SUM(u.who) FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id",  # a sum of a column may be NULL
    f"SELECT p.id, SUM(p.id) FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id",
    f"SELECT p.id, AVG(1) FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id",
    f"SELECT p.id, COUNT(*) OVER () FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id",
    f"SELECT u.who, COUNT(*) FROM people AS p JOIN {U2} ON p.id <= u.who GROUP BY u.who",  # nothing the grouped column equals
    f"SELECT u.who, COUNT(*) FROM {U2} GROUP BY u.who",
    f"SELECT p.id, u.who, COUNT(*) FROM people AS p JOIN {U2} ON p.id <= u.who GROUP BY p.id",
    f"SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.name",  # id is not fixed by the name
    f"SELECT n.id, n.name, COUNT(*) FROM notes AS n JOIN {U2} ON n.id = u.who GROUP BY n.id",  # notes has no key
    f"SELECT p.id, COUNT(*) FROM people AS p JOIN {U2} ON p.id = u.who JOIN (SELECT id FROM notes) AS s ON s.id = p.id GROUP BY p.id",  # a second derived table
    f"SELECT p.id, COUNT(*) FROM people AS p JOIN {U2} USING (id) GROUP BY p.id",
    f"SELECT p.id, COUNT(*) FROM people AS p JOIN {U2} ON p.id = u.who WHERE EXISTS (SELECT 1 FROM notes AS n WHERE n.id = u.who) GROUP BY p.id",
    f"SELECT p.id, COUNT(*) FROM people AS p JOIN {U2} ON p.id = u.who WHERE p.id IN (SELECT id FROM notes) GROUP BY p.id",
    # branches the rule does not read
    "SELECT p.id, COUNT(*) FROM people AS p JOIN (SELECT DISTINCT w1 AS who FROM games UNION ALL SELECT w2 FROM games) AS u ON p.id = u.who GROUP BY p.id",
    "SELECT p.id, COUNT(*) FROM people AS p JOIN (SELECT w1 AS who FROM games GROUP BY w1 UNION ALL SELECT w2 FROM games) AS u ON p.id = u.who GROUP BY p.id",
    "SELECT p.id, COUNT(*) FROM people AS p JOIN ((SELECT w1 AS who FROM games LIMIT 1) UNION ALL SELECT w2 FROM games) AS u ON p.id = u.who GROUP BY p.id",
    "SELECT p.id, COUNT(*) FROM people AS p JOIN (SELECT w1 AS who FROM games UNION ALL SELECT w2 FROM games LIMIT 3) AS u ON p.id = u.who GROUP BY p.id",
    "SELECT p.id, COUNT(*) FROM people AS p JOIN (SELECT w1 AS who FROM games UNION ALL SELECT id FROM notes) AS u ON p.id = u.who GROUP BY p.id",  # different tables
    "SELECT p.id, COUNT(*) FROM people AS p JOIN (SELECT w1 AS who FROM games UNION ALL SELECT MAX(w2) FROM games) AS u ON p.id = u.who GROUP BY p.id",
    "SELECT p.id, COUNT(*) FROM people AS p JOIN (SELECT w1 AS who FROM games UNION ALL SELECT w2 FROM games WHERE day IN (SELECT id FROM notes)) AS u ON p.id = u.who GROUP BY p.id",
    "SELECT p.id, COUNT(*) FROM people AS p JOIN (SELECT g.w1 AS who FROM games AS g UNION ALL SELECT n.id FROM games AS g LEFT JOIN notes AS n ON n.id = g.w2) AS u ON p.id = u.who GROUP BY p.id",
    "SELECT p.id, COUNT(*) FROM people AS p JOIN (SELECT * FROM games UNION ALL SELECT * FROM games) AS u ON p.id = u.w1 GROUP BY p.id",
    "SELECT p.id, COUNT(*) FROM people AS p JOIN (SELECT w1 AS who FROM games UNION ALL SELECT RAND() FROM games) AS u ON p.id = u.who GROUP BY p.id",
    "SELECT p.id, COUNT(*) FROM people AS p JOIN (SELECT w1 AS who FROM games) AS u ON p.id = u.who GROUP BY p.id",  # no union
    # who is the union's column or an output of the select: not read
    f"SELECT p.id AS who, COUNT(*) FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id HAVING who > 1",
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_the_loosely_grouped_name_needs_the_declared_key():
    sql = f"SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id"
    assert rewritten(sql) is not None
    assert rewritten(sql, keys=None) is None
    assert rewritten(sql, not_null=None) is None  # a UNIQUE column may hold several NULLs
    assert rewritten(sql, keys={"people": [("id", "name")]}) is None  # a key the grouping does not cover
    assert rewritten(sql.replace("p.name, ", ""), keys=None, not_null=None) is not None  # nothing loose: no fact needed


PROOFS = [
    (f"SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id", FLAT),
    (f"SELECT u.who, p.name, COUNT(u.who) FROM {U2}, people AS p WHERE u.who = p.id GROUP BY u.who, p.name", FLAT),
    (
        "SELECT u.id, u.name, COUNT(*) FROM (SELECT p.id, p.name FROM people AS p JOIN games AS g ON p.id = g.w1"
        " UNION ALL SELECT p.id, p.name FROM people AS p JOIN games AS g ON p.id = g.w2) AS u GROUP BY u.id",
        FLAT,
    ),
    (
        f"SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN {U3} ON p.id = u.who GROUP BY p.id",
        "SELECT p.id, p.name, COUNT(*) FROM (SELECT g.w3 AS x FROM games AS g UNION ALL SELECT g.w1 FROM games AS g UNION ALL SELECT g.w2 FROM games AS g) AS v"
        " JOIN people AS p ON v.x = p.id GROUP BY p.id, p.name",
    ),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # a NULL winner: the outer join from the union keeps a group for it
    (f"SELECT u.who, p.name, COUNT(*) FROM {U2} LEFT JOIN people AS p ON p.id = u.who GROUP BY u.who", FLAT),
    # UNION counts a person once per distinct row
    (f"SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN {U2.replace('UNION ALL', 'UNION')} ON p.id = u.who GROUP BY p.id", FLAT),
    # every person, also with no win
    (f"SELECT p.id, p.name, COUNT(u.who) FROM people AS p LEFT JOIN {U2} ON p.id = u.who GROUP BY p.id", FLAT),
    # two columns against three
    (f"SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN {U3} ON p.id = u.who GROUP BY p.id", FLAT),
    # a count of another column, which may be NULL
    (
        "SELECT p.id, p.name, COUNT(u.other) FROM people AS p JOIN (SELECT g.w1 AS who, g.w3 AS other FROM games AS g UNION ALL SELECT g.w2, g.w3 FROM games AS g) AS u"
        " ON p.id = u.who GROUP BY p.id",
        FLAT,
    ),
    # without the HAVING a person with no win has a row of 0
    (f"SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id", FLAT.split(" HAVING ")[0]),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_the_key_is_needed_for_the_loosely_grouped_name():
    left, right = PROOFS[0]
    assert not proved(left, right, constraints={})


def test_the_outer_join_and_the_union_really_differ():
    tables = {"people": [(1, "a")], "games": [(1, 1, None, None), (2, 1, 1, None)]}
    grouped = FLAT.replace("GROUP BY p.id", "GROUP BY p.id, p.name")
    outer, union, flat = run(
        tables,
        f"SELECT u.who, p.name, COUNT(*) FROM {U2} LEFT JOIN people AS p ON p.id = u.who GROUP BY u.who, p.name",
        f"SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN {U2.replace('UNION ALL', 'UNION')} ON p.id = u.who GROUP BY p.id, p.name",
        grouped,
    )
    assert flat == ["(1, 'a', 3)"]
    assert outer == ["(1, 'a', 3)", "(None, None, 1)"]
    assert union == ["(1, 'a', 1)"]


QUERIES = [
    f"SELECT p.id, p.name, COUNT(*) AS n FROM people AS p JOIN {U2} ON p.id = u.who GROUP BY p.id, p.name",
    f"SELECT u.who, p.name, COUNT(u.who) AS n FROM {U3}, people AS p WHERE u.who = p.id GROUP BY u.who, p.name",
    "SELECT p.id, COUNT(u.other), SUM(u.k), COUNT(*) FROM people AS p JOIN (SELECT g.w1 AS who, g.w2 AS other, 1 AS k FROM games AS g"
    " UNION ALL SELECT g.w2, g.w3, 2 FROM games AS g WHERE g.day > 3) AS u ON p.id = u.who GROUP BY p.id",
    "SELECT u.id, u.name, COUNT(*) AS n FROM (SELECT p.id, p.name FROM people AS p JOIN games AS g ON p.id = g.w1"
    " UNION ALL SELECT p.id, p.name FROM people AS p JOIN games AS g ON p.id = g.w2) AS u GROUP BY u.id, u.name",
    "SELECT who, name, COUNT(*) AS n FROM (SELECT g.w1 AS who, p.name FROM games AS g, people AS p WHERE g.w1 = p.id"
    " UNION ALL SELECT g.w2, p.name FROM games AS g, people AS p WHERE g.w2 = p.id) AS u GROUP BY who, name",
    f"SELECT p.id, COUNT(*) AS n FROM people AS p JOIN {U2} ON p.id = u.who WHERE p.name <> 'x' GROUP BY p.id HAVING COUNT(*) > 1",
    "SELECT p.id, SUM(u.k) FROM people AS p JOIN (SELECT g.w1 AS who, CASE WHEN g.w1 IS NULL THEN 0 ELSE 1 END AS k FROM games AS g"
    " UNION ALL SELECT g.w2, CASE WHEN g.w3 IS NULL THEN 0 ELSE 1 END FROM games AS g) AS u ON p.id = u.who GROUP BY p.id",
    f"SELECT p.id, COUNT(*) AS n FROM people AS p JOIN {U2} ON p.id < u.who GROUP BY p.id HAVING COUNT(*) >= 1",
    f"SELECT n.name, COUNT(*), COUNT(u.who) FROM notes AS n JOIN {U3} ON n.id = u.who OR u.who IS NULL GROUP BY n.name",
    "SELECT p.id, COUNT(*) FROM people AS p JOIN (SELECT w1 AS id FROM games UNION ALL SELECT w2 FROM games WHERE w3 IS NULL UNION ALL SELECT w3 FROM games) AS u"
    " ON p.id = u.id GROUP BY p.id",
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
