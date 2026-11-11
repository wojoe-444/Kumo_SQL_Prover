"""Sums of indicators merged, and a filter they repeat read as a ``HAVING`` (``kumo_sql_prover.indicator_sums``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.indicator_sums import disjunction_to_having, indicator_sum_rules, merge_indicator_sums, positive_tests
from kumo_sql_prover.smt_equivalence import TableConstraints

# people: id is the key; games: day is the key, w1..w3 name people and may be NULL; notes has no key
SCHEMA = {"people": ["id", "name"], "games": ["day", "w1", "w2", "w3"], "notes": ["id", "name"]}
KEYS = {"people": [("id",)], "games": [("day",)]}
NOT_NULL = {"people": frozenset({"id"}), "games": frozenset({"day"})}
CONSTRAINTS = {
    "people": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
    "games": TableConstraints(not_null=frozenset({"day"}), keys=(("day",),)),
}
ONE, TWO = "CASE WHEN p.id = g.w1 THEN 1 ELSE 0 END", "CASE WHEN p.id = g.w2 THEN 1 ELSE 0 END"
SUM2 = f"SUM({ONE} + {TWO})"
FLAT = f"SELECT p.id, p.name, {SUM2} AS n FROM people AS p CROSS JOIN games AS g GROUP BY p.id HAVING {SUM2} > 0"
EITHER = "p.id = g.w1 OR p.id = g.w2"


def sql_of(tree):
    return None if tree is None else tree.sql(dialect="mysql")


def merged(sql):
    return sql_of(merge_indicator_sums(sqlglot.parse_one(sql, read="mysql")))


def lifted(sql, keys=KEYS, not_null=NOT_NULL):
    return sql_of(disjunction_to_having(sqlglot.parse_one(sql, read="mysql"), SCHEMA, keys, not_null))


def positive(sql, schema=SCHEMA):
    return sql_of(positive_tests(sqlglot.parse_one(sql, read="mysql"), schema))


def rewritten(sql):
    """The rules, as often as they apply."""

    tree, changed = sqlglot.parse_one(sql, read="mysql"), False
    for _ in range(6):
        after = indicator_sum_rules(tree, SCHEMA, KEYS, NOT_NULL)
        if after is None:
            break
        tree, changed = after, True
    return tree.sql(dialect="mysql") if changed else None


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


def test_a_sum_of_indicator_sums_is_one_sum():
    assert merged(f"SELECT p.id, SUM({ONE}) + SUM({TWO}) AS n FROM people AS p, games AS g GROUP BY p.id") == (
        f"SELECT p.id, {SUM2} AS n FROM people AS p, games AS g GROUP BY p.id"
    )
    # IF, parentheses, three sums, the HAVING and a global aggregate
    assert merged("SELECT (SUM(IF(g.w1 = 1, 1, 0)) + SUM(IF(g.w2 = 1, 2, 0))) + SUM(3) AS n FROM games AS g HAVING SUM(1) + SUM(IF(g.w3 IS NULL, 0, 1)) > 2") == (
        "SELECT SUM(CASE WHEN g.w1 = 1 THEN 1 ELSE 0 END + CASE WHEN g.w2 = 1 THEN 2 ELSE 0 END + 3) AS n FROM games AS g HAVING SUM(1 + CASE WHEN g.w3 IS NULL THEN 0 ELSE 1 END) > 2"
    )
    # an argument that is itself a sum of indicators
    assert merged(f"SELECT p.id, SUM({ONE} + {TWO}) + SUM(CASE WHEN g.w3 > 0 THEN 1 ELSE 0 END * 2) FROM people AS p, games AS g GROUP BY p.id") == (
        f"SELECT p.id, SUM(({ONE} + {TWO}) + (CASE WHEN g.w3 > 0 THEN 1 ELSE 0 END * 2)) FROM people AS p, games AS g GROUP BY p.id"
    )


def test_counts_of_a_test_join_the_sum_in_a_grouped_select():
    assert merged("SELECT p.id, COUNT(CASE WHEN p.id = g.w1 THEN 1 END) + COUNT(IF(p.id = g.w2, 'x', NULL)) + SUM(IF(p.id = g.w3, 1, 0)) FROM people AS p, games AS g GROUP BY p.id") == (
        f"SELECT p.id, SUM({ONE} + {TWO} + CASE WHEN p.id = g.w3 THEN 1 ELSE 0 END) FROM people AS p, games AS g GROUP BY p.id"
    )
    assert merged("SELECT p.id, COUNT(CASE WHEN p.id = g.w1 THEN 1 ELSE NULL END) + COUNT(CASE WHEN p.id = g.w2 THEN 1 END) FROM people AS p, games AS g GROUP BY p.id") == (
        f"SELECT p.id, {SUM2} FROM people AS p, games AS g GROUP BY p.id"
    )


@pytest.mark.parametrize("sql", [
    "SELECT p.id, SUM(p.id = g.w1) + SUM(p.id = g.w2) FROM people AS p, games AS g GROUP BY p.id",  # NULL for a group whose w1 is all NULL
    "SELECT p.id, SUM(g.w1) + SUM(g.w2) FROM people AS p, games AS g GROUP BY p.id",
    "SELECT p.id, SUM(CASE WHEN p.id = g.w1 THEN 1 END) + SUM(CASE WHEN p.id = g.w2 THEN 1 END) FROM people AS p, games AS g GROUP BY p.id",  # no ELSE: NULL
    "SELECT p.id, SUM(CASE WHEN p.id = g.w1 THEN g.w3 ELSE 0 END) + SUM(CASE WHEN p.id = g.w2 THEN 1 ELSE 0 END) FROM people AS p, games AS g GROUP BY p.id",
    f"SELECT p.id, SUM(DISTINCT {ONE}) + SUM({TWO}) FROM people AS p, games AS g GROUP BY p.id",
    f"SELECT p.id, SUM({ONE}) - SUM({TWO}) FROM people AS p, games AS g GROUP BY p.id",
    f"SELECT p.id, SUM({ONE}) + 1 FROM people AS p, games AS g GROUP BY p.id",
    f"SELECT p.id, SUM({ONE}) + COUNT(*) FROM people AS p, games AS g GROUP BY p.id",
    f"SELECT p.id, SUM({ONE}) + MAX({TWO}) FROM people AS p, games AS g GROUP BY p.id",
    f"SELECT p.id, SUM({ONE}) OVER () + SUM({TWO}) OVER () FROM people AS p, games AS g",
    f"SELECT p.id, SUM({ONE}) + SUM({TWO}) OVER () FROM people AS p, games AS g GROUP BY p.id",
    f"SELECT p.id, SUM({ONE}) FROM people AS p, games AS g GROUP BY p.id",  # one sum
    "SELECT p.id, COUNT(CASE WHEN p.id = g.w1 THEN 1 END) FROM people AS p, games AS g GROUP BY p.id",  # a lone count
    # a global COUNT is 0 over no rows, where a SUM is NULL
    "SELECT COUNT(CASE WHEN g.w1 = 1 THEN 1 END) + COUNT(CASE WHEN g.w2 = 1 THEN 1 END) FROM games AS g",
    "SELECT g.day, COUNT(CASE WHEN g.w1 = 1 THEN 1 END) + COUNT(CASE WHEN g.w2 = 1 THEN 1 END) FROM games AS g GROUP BY g.day WITH ROLLUP",
    "SELECT p.id, COUNT(CASE WHEN p.id = g.w1 THEN 1 ELSE 0 END) + COUNT(CASE WHEN p.id = g.w2 THEN 1 END) FROM people AS p, games AS g GROUP BY p.id",  # counts every row
    "SELECT p.id, COUNT(CASE WHEN p.id = g.w1 THEN g.w3 END) + COUNT(CASE WHEN p.id = g.w2 THEN 1 END) FROM people AS p, games AS g GROUP BY p.id",  # w3 may be NULL
    "SELECT p.id, COUNT(DISTINCT CASE WHEN p.id = g.w1 THEN 1 END) + COUNT(CASE WHEN p.id = g.w2 THEN 1 END) FROM people AS p, games AS g GROUP BY p.id",
    "SELECT p.id, COUNT(CASE WHEN p.id = g.w1 THEN 1 WHEN p.id = g.w2 THEN NULL END) + COUNT(CASE WHEN p.id = g.w2 THEN 1 END) FROM people AS p, games AS g GROUP BY p.id",
])
def test_other_sums_stay(sql):
    assert merged(sql) is None


def test_a_filter_the_indicators_repeat_becomes_a_having():
    assert lifted(f"SELECT p.id, p.name, {SUM2} AS n FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id") == FLAT
    assert lifted(f"SELECT p.id, p.name, {SUM2} AS n FROM people AS p, games AS g WHERE g.w1 = p.id OR (g.w2 = p.id) GROUP BY p.id") == (
        "SELECT p.id, p.name, SUM(CASE WHEN p.id = g.w1 THEN 1 ELSE 0 END + CASE WHEN p.id = g.w2 THEN 1 ELSE 0 END) AS n FROM people AS p CROSS JOIN games AS g GROUP BY p.id"
        " HAVING SUM(CASE WHEN g.w1 = p.id THEN 1 ELSE 0 END + CASE WHEN g.w2 = p.id THEN 1 ELSE 0 END) > 0"
    )
    # IN is the disjunction of its equalities; bare columns are named; other conditions and the HAVING stay
    assert lifted("SELECT id, SUM(IF(id = w1, 1, 0) + IF(id = w2, 1, 0)) AS n FROM people JOIN games ON id IN (w1, w2) WHERE day > 2 GROUP BY id HAVING n > 1") == (
        "SELECT people.id, SUM(IF(people.id = games.w1, 1, 0) + IF(people.id = games.w2, 1, 0)) AS n FROM people CROSS JOIN games WHERE games.day > 2 GROUP BY people.id"
        " HAVING n > 1 AND SUM(CASE WHEN people.id = games.w1 THEN 1 ELSE 0 END + CASE WHEN people.id = games.w2 THEN 1 ELSE 0 END) > 0"
    )
    # the indicators may cover part of the filter, in any nesting whose value is 0 when no test holds
    assert lifted(
        "SELECT p.id, SUM(CASE WHEN p.id = g.w1 THEN CASE WHEN p.id = g.w2 THEN 2 ELSE 1 END ELSE CASE WHEN p.id = g.w2 AND g.day > 1 THEN 1 ELSE 0 END END)"
        f" FROM people AS p JOIN games AS g ON {EITHER} OR p.id = g.w3 GROUP BY p.id"
    ) == (
        "SELECT p.id, SUM(CASE WHEN p.id = g.w1 THEN CASE WHEN p.id = g.w2 THEN 2 ELSE 1 END ELSE CASE WHEN p.id = g.w2 AND g.day > 1 THEN 1 ELSE 0 END END)"
        " FROM people AS p CROSS JOIN games AS g GROUP BY p.id"
        " HAVING SUM(CASE WHEN p.id = g.w1 THEN 1 ELSE 0 END + CASE WHEN p.id = g.w2 THEN 1 ELSE 0 END + CASE WHEN p.id = g.w3 THEN 1 ELSE 0 END) > 0"
    )


@pytest.mark.parametrize("sql", [
    f"SELECT p.id, SUM({ONE} + CASE WHEN p.id = g.w3 THEN 1 ELSE 0 END) FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id",  # w3 is not in the filter
    f"SELECT p.id, SUM({ONE} + 1) FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id",
    f"SELECT p.id, SUM(CASE WHEN p.id = g.w1 THEN 0 ELSE 1 END) FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id",
    f"SELECT p.id, SUM(CASE WHEN g.day > 0 THEN 0 ELSE 1 END) FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id",
    f"SELECT p.id, COUNT(*) FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id",  # counts rows
    f"SELECT p.id, {SUM2}, MAX(g.day) FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id",
    f"SELECT p.id, SUM(p.id = g.w1) + SUM(p.id = g.w2) FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id",
    f"SELECT p.id FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id",  # no aggregate
    f"SELECT {SUM2} FROM people AS p JOIN games AS g ON {EITHER}",  # no grouping
    f"SELECT p.id, {SUM2} FROM people AS p LEFT JOIN games AS g ON {EITHER} GROUP BY p.id",
    f"SELECT p.id, {SUM2} FROM people AS p JOIN games AS g ON p.id = g.w1 GROUP BY p.id",  # one test: w2 is not covered
    f"SELECT p.id, {SUM2} FROM people AS p JOIN games AS g ON p.id NOT IN (g.w1, g.w2) GROUP BY p.id",
    f"SELECT p.id, {SUM2} FROM people AS p JOIN games AS g ON p.id IN (SELECT id FROM notes) GROUP BY p.id",
    f"SELECT p.id, {SUM2} FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id LIMIT 3",
    f"SELECT p.id, {SUM2} FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id WITH ROLLUP",
    f"SELECT p.id, p.name, {SUM2} FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.name",  # id is not fixed by the name
    f"SELECT g.w3, {SUM2} FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id",  # nor is w3 by the person
    f"SELECT p.id, {SUM2} FROM (SELECT * FROM people) AS p JOIN games AS g ON {EITHER} GROUP BY p.id",
    f"SELECT id, SUM(IF(id = w1, 1, 0)) FROM people JOIN games ON id = w1 OR id = w2 JOIN notes ON notes.name = people.name GROUP BY id",  # id: people's or notes'
])
def test_other_filters_stay(sql):
    assert lifted(sql) is None


def test_the_loosely_grouped_name_needs_the_declared_key():
    sql = f"SELECT p.id, p.name, {SUM2} FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id"
    assert lifted(sql) is not None
    assert lifted(sql, keys=None) is None
    assert lifted(sql, not_null=None) is None
    assert lifted(sql.replace("p.name, ", ""), keys=None, not_null=None) is not None


def test_a_sum_of_indicators_that_is_not_zero_is_positive():
    start = f"SELECT p.id, {SUM2} AS n FROM people AS p, games AS g GROUP BY p.id HAVING "
    assert positive(start + f"{SUM2} <> 0") == start + f"{SUM2} > 0"
    assert positive(start + f"({SUM2} >= 1) AND p.id > 2") == start + f"({SUM2} > 0) AND p.id > 2"
    # by the name of the output, when no source has a column of that name
    assert positive(start + "n != 0") == start + "n > 0"
    assert positive(start + "n >= 1", schema=None) is None
    assert positive(start.replace("AS n", "AS name") + "name <> 0") is None
    assert positive(start.replace("games AS g", "(SELECT * FROM games) AS g") + "n <> 0") is None
    assert positive(start.replace("AS n", "AS n, p.id + 1 AS n") + "n <> 0") is None


@pytest.mark.parametrize("having", [
    f"{SUM2} > 0",  # already
    f"{SUM2} >= 0",
    f"{SUM2} <> 1",
    f"{SUM2} >= 2",
    f"{SUM2} <> 0 OR p.id > 3",  # not an AND part
    f"NOT {SUM2} <> 0",
    "SUM(CASE WHEN p.id = g.w1 THEN 1 ELSE -1 END) <> 0",  # may be negative
    "SUM(CASE WHEN p.id = g.w1 THEN 1 END) <> 0",  # may be NULL: not an integer value
    "SUM(1 - CASE WHEN p.id = g.w1 THEN 1 ELSE 0 END) <> 0",
    "SUM(g.w3) <> 0",
    "SUM(p.id = g.w1) <> 0",
    "SUM(DISTINCT CASE WHEN p.id = g.w1 THEN 1 ELSE 0 END) >= 1.0",
    "COUNT(g.w1) <> 0",  # counts are left to the rules that read them
    "MAX(CASE WHEN p.id = g.w1 THEN 1 ELSE 0 END) <> 0",
])
def test_other_tests_stay(having):
    assert positive(f"SELECT p.id, {SUM2} AS total FROM people AS p, games AS g GROUP BY p.id HAVING {having}") is None


PROOFS = [
    (f"SELECT p.id, p.name, {SUM2} FROM people AS p, games AS g GROUP BY p.id HAVING {SUM2} <> 0", FLAT),
    (f"SELECT p.id, p.name, SUM(IF(g.w1 = p.id, 1, 0)) + SUM(IF(g.w2 = p.id, 1, 0)) AS n FROM people AS p, games AS g GROUP BY p.id, p.name HAVING n >= 1", FLAT),
    (f"SELECT p.id, p.name, SUM({ONE}) + SUM({TWO}) FROM people AS p, games AS g GROUP BY p.id HAVING SUM({ONE}) + SUM({TWO}) > 0", FLAT),
    (f"SELECT p.id, p.name, {SUM2} FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id", FLAT),
    (f"SELECT p.id, p.name, SUM(IF(g.w1 = p.id, 1, 0)) + SUM(IF(g.w2 = p.id, 1, 0)) FROM people AS p, games AS g WHERE p.id IN (g.w1, g.w2) GROUP BY p.id, p.name", FLAT),
    (
        "SELECT p.id, p.name, COUNT(CASE WHEN p.id = g.w1 THEN 1 END) + COUNT(IF(p.id = g.w2, 1, NULL)) AS n FROM people AS p JOIN games AS g"
        f" ON {EITHER} GROUP BY p.id HAVING n <> 0",
        FLAT,
    ),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


BOOLEAN_SUMS = f"SELECT p.id, p.name, SUM(p.id = g.w1) + SUM(p.id = g.w2) FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id"
NOT_PROVED = [
    # SUM(p.id = g.w1) is NULL for a person whose rows all have a NULL w1, and so is the total
    (BOOLEAN_SUMS, FLAT),
    (f"SELECT p.id, p.name, SUM((p.id = g.w1) + (p.id = g.w2)) FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id", FLAT),
    # a count of the rows that pass the filter is not the number of wins
    (f"SELECT p.id, p.name, COUNT(*) FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id", FLAT),
    # the filter covers a third column
    (f"SELECT p.id, p.name, {SUM2} FROM people AS p JOIN games AS g ON {EITHER} OR p.id = g.w3 GROUP BY p.id", FLAT),
    # without the HAVING a person with no win has a row of 0
    (f"SELECT p.id, p.name, {SUM2} FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id", FLAT.split(" HAVING ")[0]),
    # a sum that may be negative is not zero more often than it is positive
    (
        "SELECT p.id, SUM(CASE WHEN p.id = g.w1 THEN 1 ELSE -1 END) FROM people AS p, games AS g GROUP BY p.id HAVING SUM(CASE WHEN p.id = g.w1 THEN 1 ELSE -1 END) <> 0",
        "SELECT p.id, SUM(CASE WHEN p.id = g.w1 THEN 1 ELSE -1 END) FROM people AS p, games AS g GROUP BY p.id HAVING SUM(CASE WHEN p.id = g.w1 THEN 1 ELSE -1 END) > 0",
    ),
    # a global sum is NULL over no rows, a global count 0
    ("SELECT SUM(IF(g.w1 = 1, 1, 0)) + SUM(IF(g.w2 = 1, 1, 0)) FROM games AS g", "SELECT COUNT(IF(g.w1 = 1, 1, NULL)) + COUNT(IF(g.w2 = 1, 1, NULL)) FROM games AS g"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_the_key_is_needed_for_the_loosely_grouped_name():
    left, right = PROOFS[1]
    assert not proved(left, right, constraints={})


def test_the_boolean_sums_really_differ():
    # MySQL's SUM(p.id = g.w1), spelled for DuckDB: 1, 0 or NULL per row
    boolean = "SUM(CASE WHEN p.id = g.{0} THEN 1 WHEN p.id <> g.{0} THEN 0 END)"
    left = f"SELECT p.id, p.name, {boolean.format('w1')} + {boolean.format('w2')} FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id, p.name"
    a, b = run({"people": [(1, "a")], "games": [(1, None, 1, None)]}, left, FLAT.replace("GROUP BY p.id", "GROUP BY p.id, p.name"))
    assert a == ["(1, 'a', None)"] and b == ["(1, 'a', 1)"]


QUERIES = [
    f"SELECT p.id, p.name, SUM({ONE}) + SUM({TWO}) AS n FROM people AS p, games AS g GROUP BY p.id, p.name HAVING SUM({ONE}) + SUM({TWO}) > 0",
    f"SELECT p.id, p.name, {SUM2} AS n FROM people AS p JOIN games AS g ON {EITHER} GROUP BY p.id, p.name",
    "SELECT p.id, SUM(IF(g.w1 = p.id, 1, 0)) + SUM(IF(g.w2 = p.id, 2, 0)) FROM people AS p, games AS g WHERE p.id IN (g.w1, g.w2) AND g.day > 2 GROUP BY p.id",
    f"SELECT p.id, COUNT(CASE WHEN p.id = g.w1 THEN 1 END) + COUNT(IF(p.id = g.w2, 1, NULL)) AS n FROM people AS p JOIN games AS g ON {EITHER} OR g.w3 IS NULL GROUP BY p.id",
    "SELECT p.id, SUM(CASE WHEN p.id = g.w1 THEN CASE WHEN p.id = g.w2 THEN 2 ELSE 1 END ELSE CASE WHEN p.id = g.w2 AND g.day > 1 THEN 1 ELSE 0 END END)"
    f" FROM people AS p JOIN games AS g ON {EITHER} OR p.id = g.w3 GROUP BY p.id",
    "SELECT SUM(IF(g.w1 = 1, 1, 0)) + SUM(IF(g.w2 = 1, 2, 0)) + SUM(3) AS n FROM games AS g HAVING SUM(1) + SUM(IF(g.w3 IS NULL, 0, 1)) > 2",
    f"SELECT p.id, {SUM2} AS n, SUM(g.w3) FROM people AS p, games AS g WHERE g.day > 1 GROUP BY p.id HAVING n <> 0 AND SUM(IF(p.id = g.w3, 2, 0) * 3) >= 1",
    "SELECT n.name, SUM(CASE WHEN n.id = g.w1 THEN 1 ELSE 0 END) + SUM(CASE WHEN n.id = g.w2 OR n.id = g.w3 THEN 1 ELSE 0 END) FROM notes AS n JOIN games AS g"
    " ON n.id = g.w1 OR n.id = g.w2 OR n.id = g.w3 GROUP BY n.name",
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
