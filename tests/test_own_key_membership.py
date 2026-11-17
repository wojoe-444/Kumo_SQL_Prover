"""A row whose key is looked up among the keys of its own table is looked up as itself (``own_key_membership``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.own_key_membership import read_own_key_membership

SCHEMA = {"policy": ["pid", "city", "lat", "lon", "value"], "agent": ["aid", "city"]}
KEYS = {"policy": [("pid",)]}
NOT_NULL = {"policy": frozenset({"pid"})}
LOOKED_UP = "SELECT p.pid FROM policy AS p WHERE p.pid IN (SELECT a.pid FROM policy AS a JOIN policy AS b ON a.pid <> b.pid AND a.city = b.city)"
ITSELF = "SELECT p.pid FROM policy AS p WHERE EXISTS(SELECT 1 FROM policy AS b WHERE p.pid <> b.pid AND p.city = b.city)"


def rewritten(sql: str, keys=KEYS, not_null=NOT_NULL, types=None, schema=SCHEMA) -> str | None:
    result = read_own_key_membership(sqlglot.parse_one(sql, read="mysql"), schema, keys, not_null, types)
    return None if result is None else result.sql(dialect="mysql")


def test_the_row_with_the_key_is_the_row_itself():
    assert rewritten(LOOKED_UP) == ITSELF
    assert rewritten(LOOKED_UP.replace("p.pid IN", "p.pid NOT IN")) == ITSELF.replace("WHERE EXISTS", "WHERE NOT EXISTS")
    # DISTINCT changes nothing; a comma join with the conditions in WHERE; the keyed row listed second
    assert rewritten(LOOKED_UP.replace("SELECT a.pid", "SELECT DISTINCT a.pid")) == ITSELF
    assert rewritten("SELECT p.pid FROM policy AS p WHERE p.pid IN (SELECT a.pid FROM policy AS b, policy AS a WHERE a.pid <> b.pid AND a.city = b.city)") == ITSELF
    # another table as the partner, and two partners
    assert rewritten("SELECT p.pid FROM policy AS p WHERE p.pid IN (SELECT a.pid FROM policy AS a JOIN agent AS g ON g.city = a.city)") == "SELECT p.pid FROM policy AS p WHERE EXISTS(SELECT 1 FROM agent AS g WHERE g.city = p.city)"
    three = "SELECT p.pid FROM policy AS p WHERE p.pid IN (SELECT a.pid FROM policy AS a JOIN agent AS g ON g.city = a.city JOIN policy AS b ON b.lat = a.lat WHERE b.pid <> a.pid)"
    assert rewritten(three) == "SELECT p.pid FROM policy AS p WHERE EXISTS(SELECT 1 FROM agent AS g, policy AS b WHERE g.city = p.city AND b.lat = p.lat AND b.pid <> p.pid)"
    # the other conjuncts stay, and the row may be an inner-joined source
    joined = "SELECT p.pid FROM agent AS g JOIN policy AS p ON g.city = p.city WHERE g.aid > 3 AND p.pid IN (SELECT a.pid FROM policy AS a JOIN policy AS b ON a.pid <> b.pid AND a.city = b.city)"
    assert rewritten(joined) == "SELECT p.pid FROM agent AS g JOIN policy AS p ON g.city = p.city WHERE g.aid > 3 AND EXISTS(SELECT 1 FROM policy AS b WHERE p.pid <> b.pid AND p.city = b.city)"


def test_nested_queries_see_the_same_row():
    nested = "SELECT SUM(value) FROM policy WHERE pid IN (SELECT a.pid FROM policy AS a JOIN policy AS b ON a.pid <> b.pid WHERE a.city = b.city AND a.pid NOT IN (SELECT a.pid FROM policy AS a JOIN policy AS b ON a.pid <> b.pid WHERE a.lat = b.lat))"
    # the nested query's own ``a`` is another source: only the read outside it is the outer row's
    # the nested lookup reads none of the partner's columns, so it is a test on the row, beside the ``EXISTS``
    first = rewritten(nested)
    assert first == "SELECT SUM(value) FROM policy WHERE EXISTS(SELECT 1 FROM policy AS b WHERE policy.pid <> b.pid AND policy.city = b.city) AND NOT policy.pid IN (SELECT a.pid FROM policy AS a JOIN policy AS b ON a.pid <> b.pid WHERE a.lat = b.lat)"
    assert rewritten(first) == "SELECT SUM(value) FROM policy WHERE EXISTS(SELECT 1 FROM policy AS b WHERE policy.pid <> b.pid AND policy.city = b.city) AND NOT EXISTS(SELECT 1 FROM policy AS b WHERE policy.pid <> b.pid AND policy.lat = b.lat)"
    # under NOT IN it stays inside: there a NULL counts as "no row"
    inside = rewritten("SELECT p.pid FROM policy AS p WHERE p.pid NOT IN (SELECT a.pid FROM policy AS a JOIN policy AS b ON a.pid <> b.pid WHERE a.value > 3)")
    assert inside == "SELECT p.pid FROM policy AS p WHERE NOT EXISTS(SELECT 1 FROM policy AS b WHERE p.pid <> b.pid AND p.value > 3)"
    # a nested query that reads the keyed row
    reads = "SELECT p.pid FROM policy AS p WHERE p.pid IN (SELECT a.pid FROM policy AS a JOIN agent AS g ON g.city = a.city WHERE NOT EXISTS (SELECT 1 FROM policy AS c WHERE c.lat = a.lat AND c.pid <> a.pid))"
    assert rewritten(reads) == "SELECT p.pid FROM policy AS p WHERE EXISTS(SELECT 1 FROM agent AS g WHERE g.city = p.city) AND NOT EXISTS(SELECT 1 FROM policy AS c WHERE c.lat = p.lat AND c.pid <> p.pid)"
    # one that also reads the partner stays with it
    both = reads.replace("c.pid <> a.pid", "c.pid <> a.pid AND c.city = g.city")
    assert rewritten(both) == "SELECT p.pid FROM policy AS p WHERE EXISTS(SELECT 1 FROM agent AS g WHERE g.city = p.city AND NOT EXISTS(SELECT 1 FROM policy AS c WHERE c.lat = p.lat AND c.pid <> p.pid AND c.city = g.city))"


def test_a_row_of_a_query_around_the_select_is_read_inside_a_where():
    def inner(sql: str) -> str | None:
        probe = sqlglot.parse_one(sql, read="mysql").find(sqlglot.exp.Exists).this
        result = read_own_key_membership(probe, SCHEMA, KEYS, NOT_NULL, None)
        return None if result is None else result.sql(dialect="mysql")

    around = "SELECT p.pid FROM policy AS p WHERE EXISTS (SELECT 1 FROM agent AS g WHERE g.city = p.city AND p.pid NOT IN (SELECT a.pid FROM policy AS a, policy AS b WHERE a.pid <> b.pid AND a.lat = b.lat))"
    assert inner(around) == "SELECT 1 FROM agent AS g WHERE g.city = p.city AND NOT EXISTS(SELECT 1 FROM policy AS b WHERE p.pid <> b.pid AND p.lat = b.lat)"
    # the name means a source of the select itself, a padded row, or a row seen from a select list
    assert inner(around.replace("agent AS g WHERE g.city", "agent AS p WHERE p.city")) is None
    assert inner(around.replace("FROM policy AS p WHERE EXISTS", "FROM agent AS h LEFT JOIN policy AS p ON h.city = p.city WHERE EXISTS")) is None
    assert inner("SELECT p.pid, EXISTS (SELECT 1 FROM agent AS g WHERE g.city = p.city AND p.pid NOT IN (SELECT a.pid FROM policy AS a, policy AS b WHERE a.pid <> b.pid AND a.lat = b.lat)) FROM policy AS p") is None
    assert inner("SELECT h.aid FROM agent AS h JOIN policy AS p ON EXISTS (SELECT 1 FROM agent AS g WHERE g.city = p.city AND p.pid NOT IN (SELECT a.pid FROM policy AS a, policy AS b WHERE a.pid <> b.pid AND a.lat = b.lat))") is None
    # a query in between with a source of that name: the nearest one is meant, and it is not a keyed base row here
    between = "SELECT p.pid FROM policy AS p WHERE p.value IN (SELECT p.aid FROM agent AS p WHERE EXISTS (SELECT 1 FROM agent AS g WHERE p.pid NOT IN (SELECT a.pid FROM policy AS a, policy AS b WHERE a.pid <> b.pid AND a.lat = b.lat)))"
    assert inner(between) is None


def test_a_subquery_with_the_one_source_is_its_conditions():
    alone = "SELECT SUM(value) FROM policy AS p WHERE p.pid IN (SELECT pid FROM policy WHERE value > 3 AND EXISTS (SELECT 1 FROM agent AS g WHERE g.city = policy.city))"
    assert rewritten(alone) == "SELECT SUM(value) FROM policy AS p WHERE p.value > 3 AND EXISTS(SELECT 1 FROM agent AS g WHERE g.city = p.city)"
    assert rewritten(alone.replace("p.pid IN", "p.pid NOT IN")) is None  # the conditions may be NULL
    assert rewritten("SELECT p.pid FROM policy AS p WHERE p.pid IN (SELECT pid FROM policy)") is None


@pytest.mark.parametrize(
    "sql",
    [
        LOOKED_UP.replace("p.pid IN", "p.city IN").replace("SELECT a.pid", "SELECT a.city"),  # not a key
        LOOKED_UP.replace("SELECT a.pid", "SELECT b.pid").replace("p.pid IN", "p.lat IN"),
        LOOKED_UP.replace("SELECT a.pid", "SELECT a.lat"),  # a list of something else
        LOOKED_UP.replace("SELECT a.pid", "SELECT a.pid + 1"),
        LOOKED_UP.replace("policy AS a JOIN", "agent AS a JOIN"),  # another table's keys
        LOOKED_UP.replace("policy AS a JOIN", "POLICY AS a JOIN"),
        LOOKED_UP.replace("JOIN policy AS b ON", "LEFT JOIN policy AS b ON"),
        LOOKED_UP.replace("a.city = b.city)", "a.city = b.city GROUP BY a.pid)"),
        LOOKED_UP.replace("a.city = b.city)", "a.city = b.city LIMIT 3)"),
        LOOKED_UP.replace("a.city = b.city)", "a.city = b.city AND lat > 3)"),  # whose lat?
        LOOKED_UP.replace("a.city = b.city)", "a.city = b.city AND EXISTS (SELECT 1 FROM agent WHERE city = a.city))"),  # a bare name in a nested query
        LOOKED_UP.replace("policy AS b ON", "policy AS p ON").replace("b.", "p."),  # the subquery has a source called like the row
        LOOKED_UP.replace("a.city = b.city)", "a.city = b.city AND EXISTS (SELECT 1 FROM agent AS p WHERE p.city = a.city))"),  # the row's name means another source in there
        LOOKED_UP.replace("WHERE p.pid IN", "WHERE p.value > 3 OR p.pid IN"),
        LOOKED_UP.replace("FROM policy AS p WHERE", "FROM agent AS g LEFT JOIN policy AS p ON g.city = p.city WHERE"),  # a padded row has no key
        LOOKED_UP.replace("FROM policy AS p WHERE", "FROM (SELECT * FROM policy) AS p WHERE"),
        LOOKED_UP.replace("policy AS a JOIN policy AS b ON", "policy AS a JOIN (SELECT * FROM policy AS c WHERE c.lat = a.lat) AS b ON"),
    ],
)
def test_other_selects_are_left_alone(sql):
    assert rewritten(sql) is None


def test_facts_that_are_needed():
    assert rewritten(LOOKED_UP, keys=None) is None
    assert rewritten(LOOKED_UP, keys={"policy": [("pid", "city")]}) is None
    assert rewritten(LOOKED_UP, not_null=None) is None
    assert rewritten(LOOKED_UP, not_null={"policy": frozenset({"city"})}) is None
    assert rewritten(LOOKED_UP, schema=None) is None
    assert rewritten(LOOKED_UP, types={"policy": {"pid": "DOUBLE"}}) is None
    assert rewritten(LOOKED_UP, types={"policy": {"pid": "INT"}}) == ITSELF


SPELLINGS = [
    LOOKED_UP,
    LOOKED_UP.replace("p.pid IN", "p.pid NOT IN"),
    "SELECT p.pid FROM policy AS p WHERE p.pid IN (SELECT a.pid FROM policy AS a JOIN policy AS b ON a.pid <> b.pid WHERE a.city = b.city AND a.pid NOT IN (SELECT a.pid FROM policy AS a JOIN policy AS b ON a.pid <> b.pid WHERE a.lat = b.lat))",
    "SELECT p.pid FROM policy AS p WHERE p.pid IN (SELECT pid FROM policy WHERE value > 15 AND EXISTS (SELECT 1 FROM policy AS o WHERE o.lat = policy.lat AND o.pid <> policy.pid))",
    "SELECT p.pid, q.pid FROM policy AS p JOIN policy AS q ON p.city = q.city WHERE q.pid NOT IN (SELECT a.pid FROM policy AS a, policy AS b WHERE a.lat = b.lat AND a.pid < b.pid)",
    # a condition on the row alone, which may be NULL, beside the probe and (under NOT IN) inside it
    "SELECT p.pid FROM policy AS p WHERE p.pid IN (SELECT a.pid FROM policy AS a JOIN policy AS b ON a.pid <> b.pid AND a.city = b.city WHERE a.value > 15)",
    "SELECT p.pid FROM policy AS p WHERE p.pid NOT IN (SELECT a.pid FROM policy AS a JOIN policy AS b ON a.pid <> b.pid AND a.city = b.city WHERE a.value > 15)",
]


def _fill(con, rng, keyed: bool):
    con.execute("CREATE TABLE policy (pid INT, city VARCHAR, lat INT, lon INT, value INT)")
    for row in range(rng.randint(0, 8)):
        con.execute("INSERT INTO policy VALUES (?, ?, ?, ?, ?)", [row if keyed else rng.choice([None, 1, 1, 2]), rng.choice([None, "a", "a", "b"]), rng.choice([None, 1, 1, 2]), rng.choice([5, 6]), rng.choice([None, 10, 20])])


@pytest.mark.parametrize("seed", range(4))
def test_the_reading_returns_the_rows_duckdb_returns_when_the_key_holds(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    differs_without_the_key = nonempty = 0
    for sql in SPELLINGS:
        read = rewritten(sql)
        assert read is not None, sql
        again = rewritten(read)  # a nested lookup is read on the next step
        read = again if again is not None else read
        for keyed in (True, True, True, True, False, False):
            con = duckdb.connect()
            _fill(con, rng, keyed)
            expected = sorted(con.execute(sql).fetchall(), key=repr)
            same = expected == sorted(con.execute(read).fetchall(), key=repr)
            if keyed:
                assert same, sql
                nonempty += bool(expected)
            else:
                differs_without_the_key += not same
    assert nonempty > 3 and differs_without_the_key > 0  # the key is what makes it true


AROUND = [
    "SELECT p.pid FROM policy AS p WHERE EXISTS (SELECT 1 FROM policy AS g WHERE g.city = p.city AND g.pid <> p.pid AND p.pid NOT IN (SELECT a.pid FROM policy AS a, policy AS b WHERE a.pid <> b.pid AND a.lat = b.lat))",
    "SELECT p.pid FROM policy AS p WHERE p.value IN (SELECT g.value FROM policy AS g WHERE g.pid <> p.pid AND p.pid IN (SELECT a.pid FROM policy AS a, policy AS b WHERE a.pid < b.pid AND a.lat = b.lat AND a.value > 15))",
]


@pytest.mark.parametrize("seed", range(3))
def test_the_reading_of_an_outer_row_returns_the_rows_duckdb_returns(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(100 + seed)
    differs_without_the_key = nonempty = 0
    for sql in AROUND:
        tree = sqlglot.parse_one(sql, read="mysql")
        probe = next(s for s in tree.find_all(sqlglot.exp.Select) if s is not tree and s.find_ancestor(sqlglot.exp.Select) is tree)
        read = read_own_key_membership(probe, SCHEMA, KEYS, NOT_NULL, None)
        assert read is not None, sql
        probe.replace(read)
        after = tree.sql(dialect="mysql")
        assert " IN (" not in after.replace("p.value IN (", "")
        for keyed in [True] * 5 + [False] * 25:
            con = duckdb.connect()
            _fill(con, rng, keyed)
            expected = sorted(con.execute(sql).fetchall(), key=repr)
            same = expected == sorted(con.execute(after).fetchall(), key=repr)
            if keyed:
                assert same, sql
                nonempty += bool(expected)
            else:
                differs_without_the_key += not same
    assert nonempty > 2 and differs_without_the_key > 0


z3 = pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

FACTS = {"policy": TableConstraints(not_null=frozenset({"pid", "city", "lat", "lon"}), keys=(("pid",),))}
COUNTED = "SELECT SUM(value) FROM policy WHERE city IN (SELECT city FROM policy GROUP BY city HAVING COUNT(*) > 1) AND (lat, lon) IN (SELECT lat, lon FROM policy GROUP BY lat, lon HAVING COUNT(*) = 1)"


LOOKUPS = [
    # the nested query reuses the names of the query around it
    "SELECT SUM(value) FROM policy WHERE pid IN (SELECT DISTINCT(a.pid) FROM policy a JOIN policy b ON a.pid <> b.pid WHERE (a.city) = (b.city) AND a.pid NOT IN (SELECT DISTINCT(a.pid) FROM policy a JOIN policy b ON a.pid <> b.pid WHERE (a.lat, a.lon) = (b.lat, b.lon)))",
    "SELECT SUM(value) FROM policy WHERE pid IN (SELECT a.pid FROM policy a, policy b WHERE a.pid <> b.pid AND a.city = b.city) AND pid NOT IN (SELECT c.pid FROM policy c, policy d WHERE c.pid <> d.pid AND c.lat = d.lat AND c.lon = d.lon)",
]


@pytest.mark.parametrize("other", LOOKUPS)
def test_lookups_by_key_meet_the_grouped_counts(other):
    assert prove_equivalent_algebraic(COUNTED, other, schema=SCHEMA, constraints=FACTS, compare_names=False, dialect="mysql").proven


@pytest.mark.parametrize("other", LOOKUPS)
def test_lookups_by_key_stay_unproven_without_the_facts_or_against_another_count(other):  # slow tier
    assert not prove_equivalent_algebraic(COUNTED, other, schema=SCHEMA, compare_names=False, dialect="mysql").proven
    assert not prove_equivalent_algebraic(COUNTED.replace("COUNT(*) = 1", "COUNT(*) > 1"), other, schema=SCHEMA, constraints=FACTS, compare_names=False, dialect="mysql").proven
