"""A self-join that looks for a value on two different rows is a grouping with ``COUNT(*) >= 2`` (``duplicate_self_joins``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.duplicate_self_joins import regroup_duplicate_self_join

KEYS = {"person": [("id",)]}
NOT_NULL = {"person": frozenset({"id"})}
JOINED = "SELECT DISTINCT a.email FROM person AS a JOIN person AS b ON a.email = b.email AND a.id <> b.id"
GROUPED = "SELECT a.email FROM person AS a WHERE NOT a.email IS NULL GROUP BY a.email HAVING COUNT(*) >= 2"


def rewritten(sql: str, keys=KEYS, not_null=NOT_NULL, types=None) -> str | None:
    result = regroup_duplicate_self_join(sqlglot.parse_one(sql, read="mysql"), keys, not_null, types)
    return None if result is None else result.sql(dialect="mysql")


def test_two_rows_with_one_value_are_a_group_of_two():
    assert rewritten(JOINED) == GROUPED
    # a comma join with the conditions in WHERE, the comparison the other way round, and the other row listed
    assert rewritten("SELECT DISTINCT b.email FROM person AS a, person AS b WHERE b.id <> a.id AND a.email = b.email") == GROUPED
    # one of two different rows has the larger key
    assert rewritten(JOINED.replace("a.id <> b.id", "a.id < b.id")) == GROUPED
    assert rewritten(JOINED.replace("a.id <> b.id", "a.id > b.id")) == GROUPED
    # an output name is kept, and two equated columns are both grouped
    assert rewritten(JOINED.replace("a.email FROM", "a.email AS mail FROM")) == GROUPED.replace("a.email FROM", "a.email AS mail FROM")
    two = "SELECT DISTINCT a.email, a.city FROM person AS a JOIN person AS b ON a.email = b.email AND a.city = b.city WHERE a.id <> b.id"
    assert rewritten(two) == "SELECT a.email, a.city FROM person AS a WHERE NOT a.email IS NULL AND NOT a.city IS NULL GROUP BY a.email, a.city HAVING COUNT(*) >= 2"


@pytest.mark.parametrize(
    "sql, keys, not_null",
    [
        (JOINED, None, NOT_NULL),
        (JOINED, KEYS, None),
        (JOINED, KEYS, {"person": frozenset({"email"})}),  # two rows with a NULL id are not told apart, and are two rows
        (JOINED, {"person": [("id", "email")]}, NOT_NULL),  # id alone does not tell rows apart
        (JOINED, {"other": [("id",)]}, NOT_NULL),
        (JOINED.replace("SELECT DISTINCT", "SELECT"), KEYS, NOT_NULL),  # every pair is listed, twice
        (JOINED.replace("a.id <> b.id", "a.id <= b.id"), KEYS, NOT_NULL),  # a row pairs with itself
        (JOINED.replace("a.id <> b.id", "a.id = b.id"), KEYS, NOT_NULL),
        (JOINED.replace(" AND a.id <> b.id", ""), KEYS, NOT_NULL),
        (JOINED.replace("a.id <> b.id", "a.id <> b.id AND a.city = 'x'"), KEYS, NOT_NULL),  # another condition
        (JOINED.replace("a.id <> b.id", "a.id <> b.id AND a.city <> b.city"), KEYS, NOT_NULL),  # two columns must differ
        (JOINED.replace("a.email = b.email", "a.email = b.city"), KEYS, NOT_NULL),  # not the same column
        (JOINED.replace("a.email = b.email", "a.email = a.email"), KEYS, NOT_NULL),  # not across the two rows
        (JOINED.replace("a.email = b.email", "a.id = b.id"), KEYS, NOT_NULL),  # the key equal and different: other rules read it
        (JOINED.replace("DISTINCT a.email", "DISTINCT a.email, a.city"), KEYS, NOT_NULL),  # a column the two rows need not share
        (JOINED.replace("DISTINCT a.email", "DISTINCT a.id"), KEYS, NOT_NULL),
        (JOINED.replace("DISTINCT a.email", "DISTINCT UPPER(a.email) AS email"), KEYS, NOT_NULL),
        (JOINED.replace("person AS b", "people AS b"), KEYS, NOT_NULL),  # another table
        (JOINED.replace("JOIN person AS b", "LEFT JOIN person AS b"), KEYS, NOT_NULL),
        (JOINED + " ORDER BY a.email LIMIT 3", KEYS, NOT_NULL),
        (JOINED.replace("JOIN person AS b ON", "JOIN person AS b JOIN person AS c ON a.email = c.email AND"), KEYS, NOT_NULL),  # three rows
    ],
)
def test_other_selects_are_left_alone(sql, keys, not_null):
    assert rewritten(sql, keys, not_null) is None


def test_a_floating_point_key_is_left_alone():
    assert rewritten(JOINED, types={"person": {"id": "DOUBLE"}}) is None
    assert rewritten(JOINED, types={"person": {"id": "INT"}}) == GROUPED


SPELLINGS = [
    JOINED,
    "SELECT DISTINCT b.email FROM person AS a, person AS b WHERE a.id > b.id AND a.email = b.email",
    "SELECT DISTINCT a.email, b.city FROM person AS a JOIN person AS b ON a.email = b.email AND a.city = b.city AND a.id < b.id",
]


@pytest.mark.parametrize("seed", range(3))
def test_the_grouping_returns_the_rows_duckdb_returns_when_the_key_holds(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    differs_without_the_key = 0
    for sql in SPELLINGS:
        grouped = rewritten(sql)
        assert grouped is not None, sql
        for keyed in (True, True, True, False):
            con = duckdb.connect()
            con.execute("CREATE TABLE person (id INT, email VARCHAR, city VARCHAR)")
            for row in range(rng.randint(0, 9)):
                key = row if keyed else rng.choice([None, 1, 1, 2])
                con.execute("INSERT INTO person VALUES (?, ?, ?)", [key, rng.choice([None, "a", "a", "b"]), rng.choice([None, "x", "x", "y"])])
            same = sorted(con.execute(sql).fetchall(), key=repr) == sorted(con.execute(grouped).fetchall(), key=repr)
            if keyed:
                assert same, sql
            else:
                differs_without_the_key += not same
    assert differs_without_the_key > 0  # the key is what makes it true


z3 = pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

SCHEMA = {"person": ["id", "email", "city"]}
COUNTED = "SELECT email FROM person GROUP BY email HAVING COUNT(email) > 1"
KEY = {"person": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),))}


def prove(left, right, **options):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, dialect="mysql", compare_names=False, **options)


def test_the_two_spellings_of_duplicates_are_one_query_under_the_key():
    assert prove(COUNTED, JOINED, constraints=KEY).proven
    assert prove(COUNTED, JOINED.replace("a.id <> b.id", "a.id < b.id"), constraints=KEY).proven
    assert not prove(COUNTED, JOINED).proven
    assert not prove(COUNTED, JOINED, constraints={"person": TableConstraints(keys=(("id",),))}).proven  # a key that can be NULL
    assert not prove(COUNTED.replace("> 1", "> 2"), JOINED, constraints=KEY).proven


def test_the_conditional_verdict_names_the_key():
    # in the slow tier (tests/order.json): the search runs the prover a dozen times
    result = prove(COUNTED, JOINED.replace("a.id <> b.id", "a.id > b.id"), conditional=True)
    assert result.status.value == "proven_conditionally"
    assert sorted((c.kind, tuple(c.columns)) for c in result.conditions) == [("not_null", ("id",)), ("unique", ("id",))]
