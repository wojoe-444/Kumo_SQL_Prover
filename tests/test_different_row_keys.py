"""Two rows of one table told apart by a key are told apart by the table's first key (``different_row_keys``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.different_row_keys import name_different_rows

KEYS = {"policy": [("pid",), ("code",)]}
NOT_NULL = {"policy": frozenset({"pid", "code"})}
BY_PID = "SELECT a.pid FROM policy AS a JOIN policy AS b ON a.city = b.city AND a.pid <> b.pid"
BY_CODE = "SELECT a.pid FROM policy AS a JOIN policy AS b ON a.city = b.city AND a.code <> b.code"


def rewritten(sql: str, keys=KEYS, not_null=NOT_NULL, types=None, pick=None) -> str | None:
    tree = sqlglot.parse_one(sql, read="mysql")
    select = tree if pick is None else pick(tree)
    result = name_different_rows(select, keys, not_null, types)
    if result is None:
        return None
    if select is tree:
        return result.sql(dialect="mysql")
    select.replace(result)
    return tree.sql(dialect="mysql")


def test_the_first_key_by_name_tells_the_rows_apart():
    assert rewritten(BY_PID) == BY_CODE
    assert rewritten(BY_CODE) is None  # already the first key
    # in the WHERE, with the sides the other way round, and twice
    assert rewritten("SELECT a.pid FROM policy AS a, policy AS b WHERE b.pid <> a.pid AND a.city = b.city") == "SELECT a.pid FROM policy AS a, policy AS b WHERE b.code <> a.code AND a.city = b.city"
    three = "SELECT a.pid FROM policy AS a JOIN policy AS b ON a.pid <> b.pid JOIN policy AS c ON c.pid <> b.pid"
    assert rewritten(three) == "SELECT a.pid FROM policy AS a JOIN policy AS b ON a.code <> b.code JOIN policy AS c ON c.code <> b.code"
    # under OR and NOT the comparison has the same value, so it is read there too
    assert rewritten("SELECT a.pid FROM policy AS a, policy AS b WHERE NOT (a.pid <> b.pid OR a.city = b.city)") == "SELECT a.pid FROM policy AS a, policy AS b WHERE NOT (a.code <> b.code OR a.city = b.city)"
    # a padded row: both comparisons are NULL
    assert rewritten(BY_PID.replace("JOIN", "LEFT JOIN")) == BY_CODE.replace("JOIN", "LEFT JOIN")


def test_a_row_of_a_query_around_the_select_is_told_apart_too():
    def probe(tree):
        return tree.find(sqlglot.exp.Exists).this

    around = "SELECT p.pid FROM policy AS p WHERE EXISTS (SELECT 1 FROM policy AS q WHERE q.city = p.city AND q.pid <> p.pid)"
    assert rewritten(around, pick=probe) == "SELECT p.pid FROM policy AS p WHERE EXISTS(SELECT 1 FROM policy AS q WHERE q.city = p.city AND q.code <> p.code)"
    assert rewritten(around) is None  # the outer select holds no comparison of its own
    # the name means a derived table out there, or the select is not in a WHERE
    assert rewritten(around.replace("FROM policy AS p WHERE", "FROM (SELECT * FROM policy) AS p WHERE"), pick=probe) is None
    listed = "SELECT p.pid, EXISTS (SELECT 1 FROM policy AS q WHERE q.city = p.city AND q.pid <> p.pid) FROM policy AS p"
    assert rewritten(listed, pick=probe) is None


@pytest.mark.parametrize(
    "sql",
    [
        BY_PID.replace("a.pid <> b.pid", "a.pid = b.pid"),
        BY_PID.replace("a.pid <> b.pid", "a.pid < b.pid"),
        BY_PID.replace("a.pid <> b.pid", "a.pid <> b.code"),  # two columns
        BY_PID.replace("a.pid <> b.pid", "a.city <> b.city"),  # not a key
        BY_PID.replace("a.pid <> b.pid", "a.pid <> a.pid"),  # one row
        BY_PID.replace("a.pid <> b.pid", "a.pid <> 3"),
        BY_PID.replace("a.pid <> b.pid", "pid <> b.pid"),  # whose pid?
        BY_PID.replace("policy AS b", "agent AS b"),  # another table
        BY_PID.replace("policy AS b", "POLICY AS b"),
        BY_PID.replace("policy AS b", "(SELECT * FROM policy) AS b"),
        "SELECT a.pid FROM policy AS a JOIN policy AS b USING (city) WHERE a.pid <> b.pid",
    ],
)
def test_other_comparisons_are_left_alone(sql):
    assert rewritten(sql) is None


def test_facts_that_are_needed():
    assert rewritten(BY_PID, keys=None) is None
    assert rewritten(BY_PID, not_null=None) is None
    assert rewritten(BY_PID, keys={"policy": [("pid",)]}) is None  # one key: nothing to choose
    assert rewritten(BY_PID, keys={"policy": [("pid",), ("code", "city")]}) is None  # a composite key tells nothing apart alone
    assert rewritten(BY_PID, not_null={"policy": frozenset({"pid"})}) is None  # code may be NULL
    assert rewritten(BY_PID, not_null={"policy": frozenset({"code"})}) is None  # pid may be NULL
    assert rewritten(BY_PID, types={"policy": {"code": "DOUBLE"}}) is None
    assert rewritten(BY_PID, types={"policy": {"pid": "DOUBLE"}}) is None
    assert rewritten(BY_PID, types={"policy": {"pid": "INT", "code": "VARCHAR"}}) == BY_CODE
    # three keys: the first by name
    assert rewritten(BY_PID, keys={"policy": [("pid",), ("code",), ("badge",)]}, not_null={"policy": frozenset({"pid", "code", "badge"})}) == BY_CODE.replace("code", "badge")


SPELLINGS = [
    BY_PID,
    "SELECT a.pid, b.pid FROM policy AS a LEFT JOIN policy AS b ON a.city = b.city AND a.pid <> b.pid",
    "SELECT a.pid, b.pid FROM policy AS a, policy AS b WHERE NOT (a.pid <> b.pid) OR a.city = b.city",
    "SELECT a.pid, (a.pid <> b.pid) IS NULL FROM policy AS a LEFT JOIN policy AS b ON a.city = b.city WHERE a.pid <> b.pid OR b.pid IS NULL",
]


@pytest.mark.parametrize("seed", range(3))
def test_the_reading_returns_the_rows_duckdb_returns_when_both_are_keys(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    differs = nonempty = 0
    for sql in SPELLINGS:
        read = rewritten(sql)
        assert read is not None, sql
        for keyed in [True] * 5 + [False] * 10:
            con = duckdb.connect()
            con.execute("CREATE TABLE policy (pid INT, code INT, city VARCHAR)")
            for row in range(rng.randint(0, 6)):
                # without the facts: a repeated or missing code
                con.execute("INSERT INTO policy VALUES (?, ?, ?)", [row, 100 - row if keyed else rng.choice([None, 1, 1, 2]), rng.choice([None, "a", "a", "b"])])
            expected = sorted(con.execute(sql).fetchall(), key=repr)
            same = expected == sorted(con.execute(read).fetchall(), key=repr)
            if keyed:
                assert same, sql
                nonempty += bool(expected)
            else:
                differs += not same
    assert nonempty > 3 and differs > 0  # the facts are what make it true


z3 = pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

SCHEMA = {"policy": ["pid", "code", "city", "value"]}
COUNTED = "SELECT SUM(value) FROM policy WHERE city IN (SELECT city FROM policy GROUP BY city HAVING COUNT(*) > 1)"
LOOKED_UP = "SELECT SUM(value) FROM policy WHERE pid IN (SELECT a.pid FROM policy a JOIN policy b ON a.pid <> b.pid AND a.city = b.city)"


def test_two_spellings_meet_whichever_key_each_names():
    both = {"policy": TableConstraints(not_null=frozenset({"pid", "code", "city"}), keys=(("pid",), ("code",)))}
    assert prove_equivalent_algebraic(COUNTED, LOOKED_UP, schema=SCHEMA, constraints=both, compare_names=False, dialect="mysql").proven
    assert prove_equivalent_algebraic(LOOKED_UP, LOOKED_UP.replace("a.pid <> b.pid", "a.code <> b.code"), schema=SCHEMA, constraints=both, compare_names=False, dialect="mysql").proven
    one = {"policy": TableConstraints(not_null=frozenset({"pid", "code", "city"}), keys=(("pid",),))}
    assert prove_equivalent_algebraic(COUNTED, LOOKED_UP, schema=SCHEMA, constraints=one, compare_names=False, dialect="mysql").proven


def test_a_column_that_is_not_a_key_tells_nothing_apart():
    one = {"policy": TableConstraints(not_null=frozenset({"pid", "code", "city"}), keys=(("pid",),))}
    assert not prove_equivalent_algebraic(LOOKED_UP, LOOKED_UP.replace("a.pid <> b.pid", "a.code <> b.code"), schema=SCHEMA, constraints=one, compare_names=False, dialect="mysql").proven
