"""A row tested against a grouped count of its own table is asked for another row (``own_group_membership``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.own_group_membership import read_own_group_membership

SCHEMA = {"policy": ["pid", "city", "lat", "lon", "value", "owner"], "agent": ["aid", "city"]}
KEYS = {"policy": [("pid",)]}
NOT_NULL = {"policy": frozenset({"pid"})}
ALL = {"policy": frozenset({"pid", "city", "lat", "lon", "owner"})}
MORE = "SELECT p.pid FROM policy AS p WHERE p.city IN (SELECT city FROM policy GROUP BY city HAVING COUNT(*) > 1)"
OTHER = "EXISTS(SELECT 1 FROM policy AS ogm0 WHERE ogm0.city = p.city AND ogm0.pid <> p.pid)"


def rewritten(sql: str, keys=KEYS, not_null=NOT_NULL, types=None, schema=SCHEMA) -> str | None:
    result = read_own_group_membership(sqlglot.parse_one(sql, read="mysql"), schema, keys, not_null, types)
    return None if result is None else result.sql(dialect="mysql")


def test_a_shared_value_is_another_row_with_it():
    assert rewritten(MORE) == f"SELECT p.pid FROM policy AS p WHERE {OTHER}"
    for spelled in ("COUNT(*) >= 2", "COUNT(*) <> 1", "1 < COUNT(*)", "COUNT(1) > 1", "COUNT(pid) > 1", "COUNT(city) > 1", "COUNT(DISTINCT pid) > 1", "2 <= COUNT(policy.pid)"):
        assert rewritten(MORE.replace("COUNT(*) > 1", spelled)) == f"SELECT p.pid FROM policy AS p WHERE {OTHER}", spelled
    # the other conjuncts stay where they are, and the subquery may name its table
    assert rewritten(MORE.replace("WHERE p.city", "WHERE p.value > 3 AND p.city") + " AND p.owner = 'x'") == f"SELECT p.pid FROM policy AS p WHERE p.value > 3 AND {OTHER} AND p.owner = 'x'"
    assert rewritten(MORE.replace("SELECT city FROM policy GROUP BY city", "SELECT DISTINCT q.city FROM policy AS q GROUP BY q.city")) == f"SELECT p.pid FROM policy AS p WHERE {OTHER}"


def test_a_value_of_its_own_is_no_other_row_with_it():
    one = MORE.replace("> 1", "= 1")
    assert rewritten(one) == f"SELECT p.pid FROM policy AS p WHERE NOT p.city IS NULL AND NOT {OTHER}"
    assert rewritten(one, not_null=ALL) == f"SELECT p.pid FROM policy AS p WHERE NOT {OTHER}"
    for spelled in ("COUNT(*) < 2", "COUNT(*) <= 1", "1 = COUNT(*)", "2 > COUNT(pid)"):
        assert rewritten(MORE.replace("COUNT(*) > 1", spelled), not_null=ALL) == f"SELECT p.pid FROM policy AS p WHERE NOT {OTHER}", spelled


def test_not_in_needs_the_tested_column_never_null():
    negative = MORE.replace("p.city IN", "p.city NOT IN")
    assert rewritten(negative) is None
    assert rewritten(negative, not_null=ALL) == f"SELECT p.pid FROM policy AS p WHERE NOT {OTHER}"
    assert rewritten(negative.replace("> 1", "= 1"), not_null=ALL) == f"SELECT p.pid FROM policy AS p WHERE {OTHER}"
    assert rewritten(MORE.replace("p.city IN", "NOT p.city IN"), not_null=ALL) == f"SELECT p.pid FROM policy AS p WHERE NOT {OTHER}"


def test_several_columns_and_a_bare_table():
    sql = "SELECT SUM(value) FROM policy WHERE city IN (SELECT city FROM policy GROUP BY city HAVING COUNT(*) > 1) AND (lat, lon) IN (SELECT lat, lon FROM policy GROUP BY lat, lon HAVING COUNT(*) = 1)"
    first = rewritten(sql, not_null=ALL)
    assert first == "SELECT SUM(value) FROM policy WHERE EXISTS(SELECT 1 FROM policy AS ogm0 WHERE ogm0.city = policy.city AND ogm0.pid <> policy.pid) AND (lat, lon) IN (SELECT lat, lon FROM policy GROUP BY lat, lon HAVING COUNT(*) = 1)"
    assert rewritten(first, not_null=ALL) == (
        "SELECT SUM(value) FROM policy WHERE EXISTS(SELECT 1 FROM policy AS ogm0 WHERE ogm0.city = policy.city AND ogm0.pid <> policy.pid) "
        "AND NOT EXISTS(SELECT 1 FROM policy AS ogm1 WHERE ogm1.lat = policy.lat AND ogm1.lon = policy.lon AND ogm1.pid <> policy.pid)"
    )
    # the subquery may list its grouping columns in another order than it groups by: the tuple follows the list
    swapped = "SELECT pid FROM policy WHERE (lon, lat) IN (SELECT lon, lat FROM policy GROUP BY lat, lon HAVING COUNT(*) > 1)"
    assert rewritten(swapped) == "SELECT pid FROM policy WHERE EXISTS(SELECT 1 FROM policy AS ogm0 WHERE ogm0.lon = policy.lon AND ogm0.lat = policy.lat AND ogm0.pid <> policy.pid)"


def test_counting_values_needs_no_key():
    sql = "SELECT p.pid FROM policy AS p WHERE p.city IN (SELECT city FROM policy GROUP BY city HAVING COUNT(DISTINCT owner) > 1)"
    expected = "SELECT p.pid FROM policy AS p WHERE EXISTS(SELECT 1 FROM policy AS ogm0 WHERE ogm0.city = p.city AND ogm0.owner <> p.owner)"
    assert rewritten(sql, keys=None, not_null=ALL) == expected
    assert rewritten(sql, keys=None) is None  # a row with no owner is not one of the group's values
    assert rewritten(sql.replace("> 1", "= 1"), keys=None, not_null=ALL) == expected.replace("WHERE EXISTS", "WHERE NOT EXISTS")


def test_the_normalizers_spelling_of_the_grouped_subquery_is_read():
    derived = "SELECT p.pid FROM policy AS p WHERE p.city IN (SELECT g.k0 FROM (SELECT city AS k0, COUNT(pid) AS a0 FROM policy GROUP BY city) AS g WHERE g.a0 >= 2)"
    assert rewritten(derived) == f"SELECT p.pid FROM policy AS p WHERE {OTHER}"
    two = "SELECT pid FROM policy WHERE NOT (lat, lon) IN (SELECT g.k0, g.k1 FROM (SELECT lat AS k0, lon AS k1, COUNT(*) AS a0 FROM policy GROUP BY lat, lon) AS g WHERE g.a0 >= 2)"
    assert rewritten(two, not_null=ALL) == "SELECT pid FROM policy WHERE NOT EXISTS(SELECT 1 FROM policy AS ogm0 WHERE ogm0.lat = policy.lat AND ogm0.lon = policy.lon AND ogm0.pid <> policy.pid)"
    assert rewritten(derived.replace("g.a0 >= 2", "g.a0 >= 2 AND g.k0 > 'm'")) is None
    assert rewritten(derived.replace("GROUP BY city)", "GROUP BY city HAVING MIN(value) > 3)")) is None
    assert rewritten(derived.replace("SELECT g.k0 FROM", "SELECT g.a0 FROM")) is None


def test_a_name_in_use_is_not_taken_for_the_other_row():
    assert "AS ogm1 WHERE ogm1.city" in rewritten(MORE.replace("policy AS p WHERE", "policy AS p, agent AS ogm0 WHERE"))


@pytest.mark.parametrize(
    "sql",
    [
        MORE.replace("GROUP BY city", "WHERE value > 3 GROUP BY city"),  # the row need not be in the group counted
        MORE.replace("GROUP BY city", "GROUP BY city, owner"),  # the group that passes may be another one
        MORE.replace("SELECT city FROM policy GROUP BY city", "SELECT owner FROM policy GROUP BY owner"),  # not the tested column
        MORE.replace("SELECT city FROM policy GROUP BY city", "SELECT city FROM agent GROUP BY city"),  # another table
        MORE.replace("SELECT city FROM policy GROUP BY city", "SELECT city FROM POLICY GROUP BY city"),  # another table to some engines
        MORE.replace("> 1", "> 2"),
        MORE.replace("> 1", ">= 1"),
        MORE.replace("> 1", "> 1 AND MAX(value) > 3"),
        MORE.replace("COUNT(*)", "COUNT(value)"),  # counts the rows with a value
        MORE.replace("COUNT(*)", "COUNT(DISTINCT city)"),
        MORE.replace("COUNT(*)", "SUM(value)"),
        MORE.replace("p.city IN", "UPPER(p.city) IN"),
        MORE.replace("p.city IN", "p.owner IN"),
        MORE.replace("WHERE p.city IN", "WHERE p.value > 3 OR p.city IN"),  # under OR a NULL test is not a FALSE one
        MORE.replace("FROM policy AS p WHERE", "FROM agent AS a LEFT JOIN policy AS p ON a.city = p.city WHERE"),  # a padded row is no row of the table
        MORE.replace("FROM policy AS p WHERE", "FROM policy AS p RIGHT JOIN agent AS a ON a.city = p.city WHERE"),
        MORE.replace("FROM policy AS p WHERE", "FROM (SELECT * FROM policy WHERE value > 3) AS p WHERE"),
        "SELECT pid FROM policy, agent WHERE city IN (SELECT city FROM policy GROUP BY city HAVING COUNT(*) > 1)",  # whose city?
        "SELECT p.pid FROM policy AS p WHERE p.pid IN (SELECT pid FROM policy GROUP BY pid HAVING COUNT(*) > 1)",  # a key's groups have one row
    ],
)
def test_other_selects_are_left_alone(sql):
    assert rewritten(sql, not_null=ALL) is None


def test_facts_that_are_needed():
    assert rewritten(MORE, keys=None) is None
    assert rewritten(MORE, keys={"policy": [("pid", "city")]}) is None
    assert rewritten(MORE, not_null={"policy": frozenset({"city"})}) is None  # two rows with a NULL key are not told apart
    assert rewritten(MORE, schema=None) is None
    assert rewritten(MORE, types={"policy": {"pid": "DOUBLE"}}) is None
    assert rewritten(MORE, types={"policy": {"city": "FLOAT"}}) is None
    assert rewritten(MORE, types={"policy": {"city": "VARCHAR", "pid": "INT"}}) is not None
    # an inner join keeps real rows; the row may be the joined one
    joined = MORE.replace("FROM policy AS p WHERE", "FROM agent AS a JOIN policy AS p ON a.city = p.city WHERE")
    assert rewritten(joined) == f"SELECT p.pid FROM agent AS a JOIN policy AS p ON a.city = p.city WHERE {OTHER}"


TESTS = [
    "p.city IN (SELECT city FROM policy GROUP BY city HAVING COUNT(*) > 1)",
    "p.city IN (SELECT city FROM policy GROUP BY city HAVING COUNT(pid) = 1)",
    "p.city NOT IN (SELECT city FROM policy GROUP BY city HAVING COUNT(*) > 1)",
    "p.city NOT IN (SELECT city FROM policy GROUP BY city HAVING COUNT(*) = 1)",
    "(p.lat, p.lon) IN (SELECT lat, lon FROM policy GROUP BY lat, lon HAVING COUNT(*) = 1)",
    "(p.lat, p.lon) NOT IN (SELECT lat, lon FROM policy GROUP BY lat, lon HAVING COUNT(*) >= 2)",
    "p.city IN (SELECT city FROM policy GROUP BY city HAVING COUNT(DISTINCT owner) > 1)",
    "p.city NOT IN (SELECT city FROM policy GROUP BY city HAVING COUNT(DISTINCT owner) > 1)",
    "p.city IN (SELECT city FROM policy GROUP BY city HAVING COUNT(DISTINCT owner) < 2)",
]


def _fill(con, rng, nulls: bool, keyed: bool):
    con.execute("CREATE TABLE policy (pid INT, city VARCHAR, lat INT, lon INT, value INT, owner VARCHAR)")
    for row in range(rng.randint(0, 9)):
        pick = lambda values: rng.choice(values + ([None] if nulls else []))  # noqa: E731
        con.execute("INSERT INTO policy VALUES (?, ?, ?, ?, ?, ?)", [row if keyed else rng.choice([1, 1, 2]), pick(["a", "a", "b", "c"]), pick([1, 1, 2]), pick([5, 5, 6]), rng.choice([None, 10, 20]), pick(["x", "x", "y"])])


@pytest.mark.parametrize("seed", range(4))
def test_the_reading_returns_the_rows_duckdb_returns(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    kept = dropped = 0
    for test in TESTS:
        sql = f"SELECT p.pid, p.value FROM policy AS p WHERE {test}"
        # with every fact declared and true
        read = rewritten(sql, not_null=ALL)
        assert read is not None, test
        for _ in range(10):
            con = duckdb.connect()
            _fill(con, rng, nulls=False, keyed=True)
            expected = sorted(con.execute(sql).fetchall(), key=repr)
            assert expected == sorted(con.execute(read).fetchall(), key=repr), test
            kept += bool(expected)
            dropped += len(expected) < con.execute("SELECT COUNT(*) FROM policy").fetchone()[0]
        # with the key alone declared, where the rule still fires the tested columns may be NULL
        read = rewritten(sql)
        if read is not None:
            for _ in range(10):
                con = duckdb.connect()
                _fill(con, rng, nulls=True, keyed=True)
                con.execute("UPDATE policy SET owner = 'x' WHERE owner IS NULL")
                assert sorted(con.execute(sql).fetchall(), key=repr) == sorted(con.execute(read).fetchall(), key=repr), test
    assert kept > 20 and dropped > 20


def test_the_facts_are_what_makes_it_true():
    """Each declared fact dropped from the data gives a database where the two forms differ."""

    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(7)

    def differ(test: str, nulls: bool, keyed: bool) -> bool:
        sql = f"SELECT p.pid, p.value FROM policy AS p WHERE {test}"
        read = rewritten(sql, not_null=ALL)
        for _ in range(60):
            con = duckdb.connect()
            _fill(con, rng, nulls=nulls, keyed=keyed)
            if sorted(con.execute(sql).fetchall(), key=repr) != sorted(con.execute(read).fetchall(), key=repr):
                return True
        return False

    assert differ(TESTS[0], nulls=False, keyed=False)  # two rows with one key value are not "another row"
    assert differ(TESTS[2], nulls=True, keyed=True)  # NOT IN with a NULL city
    assert differ(TESTS[6], nulls=True, keyed=True)  # a row with no owner


z3 = pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

FACTS = {"policy": TableConstraints(not_null=frozenset({"pid", "city", "lat", "lon"}), keys=(("pid",),))}
COUNTED = "SELECT SUM(value) FROM policy WHERE city IN (SELECT city FROM policy GROUP BY city HAVING COUNT(*) > 1) AND (lat, lon) IN (SELECT lat, lon FROM policy GROUP BY lat, lon HAVING COUNT(*) = 1)"


def prove(left, right, **options):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, dialect="mysql", compare_names=False, **options)


SPELLED = [
    "SELECT SUM(p.value) FROM policy p WHERE EXISTS (SELECT 1 FROM policy q WHERE q.city = p.city AND q.pid <> p.pid) AND NOT EXISTS (SELECT 1 FROM policy q WHERE q.lat = p.lat AND q.lon = p.lon AND q.pid <> p.pid)",
    "SELECT SUM(value) FROM policy WHERE city IN (SELECT city FROM policy GROUP BY city HAVING COUNT(pid) >= 2) AND (lat, lon) NOT IN (SELECT lat, lon FROM policy GROUP BY lat, lon HAVING COUNT(pid) > 1)",
    "SELECT SUM(value) FROM policy WHERE city NOT IN (SELECT city FROM policy GROUP BY city HAVING COUNT(*) = 1) AND (lat, lon) NOT IN (SELECT lat, lon FROM policy GROUP BY lat, lon HAVING COUNT(*) > 1)",
    "SELECT SUM(value) FROM policy WHERE pid IN (SELECT DISTINCT a.pid FROM policy a JOIN policy b ON a.pid <> b.pid AND a.city = b.city) AND pid NOT IN (SELECT DISTINCT a.pid FROM policy a JOIN policy b ON a.pid <> b.pid AND a.lat = b.lat AND a.lon = b.lon)",
]


@pytest.mark.parametrize("other", SPELLED)
def test_the_spellings_of_a_shared_value_are_one_query_under_the_key(other):
    assert prove(COUNTED, other, constraints=FACTS).proven


@pytest.mark.parametrize("other", SPELLED)
def test_the_spellings_stay_unproven_without_the_facts(other):  # the last one is in the slow tier
    assert not prove(COUNTED, other).proven


def test_near_misses_stay_unproven():
    exists = "SELECT SUM(p.value) FROM policy p WHERE EXISTS (SELECT 1 FROM policy q WHERE q.city = p.city AND q.pid <> p.pid) AND NOT EXISTS (SELECT 1 FROM policy q WHERE q.lat = p.lat AND q.lon = p.lon AND q.pid <> p.pid)"
    assert not prove(COUNTED.replace("COUNT(*) > 1", "COUNT(*) > 2"), exists, constraints=FACTS).proven
    assert not prove(COUNTED.replace("COUNT(*) = 1", "COUNT(*) > 1"), exists, constraints=FACTS).proven
    assert not prove(COUNTED, exists.replace("q.lon = p.lon AND ", ""), constraints=FACTS).proven
    assert not prove(COUNTED, exists, constraints={"policy": TableConstraints(not_null=frozenset({"pid", "city", "lat", "lon"}))}).proven  # no key
