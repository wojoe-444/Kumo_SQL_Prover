"""A list drawn from a table the row is joined to on its key is a test of the joined row (``joined_key_lists``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.joined_key_lists import test_joined_rows as read_joined_rows

SCHEMA = {"item": ["item_id", "label"], "sale": ["buyer", "item_id", "paid"]}
KEYS = {"item": [("item_id",)]}
LISTED = "SELECT s.buyer FROM sale AS s JOIN item AS i ON s.item_id = i.item_id WHERE s.item_id IN (SELECT x.item_id FROM item AS x WHERE x.label = 'S' OR x.label = 'I') AND s.paid > 1"
JOINED = "SELECT s.buyer FROM sale AS s JOIN item AS i ON s.item_id = i.item_id WHERE (i.label = 'S' OR i.label = 'I') AND s.paid > 1"


def reading(sql: str, keys=KEYS, types=None, schema=SCHEMA) -> str | None:
    result = read_joined_rows(sqlglot.parse_one(sql, read="mysql"), keys, schema, types)
    return result.sql(dialect="mysql") if result is not None else None


def test_the_list_is_read_on_the_joined_row():
    assert reading(LISTED) == JOINED
    assert reading(JOINED) is None
    # the key side tested, bare names in the list, a comma join, DISTINCT in the list
    assert reading(LISTED.replace("s.item_id IN", "i.item_id IN")) == JOINED
    assert reading(LISTED.replace("SELECT x.item_id FROM item AS x WHERE x.label = 'S' OR x.label = 'I'", "SELECT DISTINCT item_id FROM item WHERE label = 'S' OR label = 'I'")) == JOINED
    assert reading(LISTED.replace(" JOIN item AS i ON s.item_id = i.item_id WHERE", ", item AS i WHERE i.item_id = s.item_id AND")) == JOINED.replace(
        " JOIN item AS i ON s.item_id = i.item_id WHERE", ", item AS i WHERE i.item_id = s.item_id AND"
    )
    # a list without a WHERE asks only for a row with that key, and the joined row is one
    assert reading(LISTED.replace(" WHERE x.label = 'S' OR x.label = 'I'", "")) == "SELECT s.buyer FROM sale AS s JOIN item AS i ON s.item_id = i.item_id WHERE s.paid > 1"
    # grouping and HAVING around it stay
    grouped = reading(LISTED + " GROUP BY s.buyer HAVING COUNT(*) > 1")
    assert grouped == JOINED + " GROUP BY s.buyer HAVING COUNT(*) > 1"


LEFT_ALONE = [
    LISTED.replace("s.item_id IN", "s.item_id NOT IN"),
    LISTED.replace("s.item_id IN", "s.paid IN"),  # not the tied column
    LISTED.replace("SELECT x.item_id FROM", "SELECT x.label FROM"),  # the list is not of the key
    LISTED.replace("FROM item AS x WHERE", "FROM sale AS x WHERE").replace("x.label = 'S' OR x.label = 'I'", "x.paid = 2"),  # a list from another table
    LISTED.replace("JOIN item AS i ON", "LEFT JOIN item AS i ON"),  # the joined row may be padding
    LISTED.replace("ON s.item_id = i.item_id", "ON s.item_id < i.item_id"),  # not tied by an equality
    LISTED.replace("ON s.item_id = i.item_id", "ON s.item_id = i.item_id OR s.paid = 1"),  # the tie is not a top-level conjunct
    LISTED.replace("x.label = 'I'", "x.label = 'I' LIMIT 3"),
    LISTED.replace("x.label = 'I'", "x.label IN (SELECT y.label FROM item AS y)"),
    LISTED.replace("x.label = 'I'", "x.label = s.buyer"),  # the list reads the outer row
    LISTED.replace("SELECT x.item_id FROM item AS x WHERE", "SELECT x.item_id FROM item AS x JOIN sale AS z ON z.item_id = x.item_id WHERE"),
    LISTED.replace("SELECT x.item_id FROM", "SELECT MAX(x.item_id) FROM"),
    LISTED.replace("x.label = 'I'", "x.label = 'I' AND RAND() < 0.5"),
    LISTED.replace("s.item_id IN (", "s.item_id + 0 IN ("),
    LISTED.replace("s.item_id IN (SELECT", "(s.item_id IN (SELECT").replace("x.label = 'I')", "x.label = 'I') OR s.paid = 9)"),  # under OR
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_shapes_are_left_alone(sql):
    sqlglot.parse_one(sql, read="mysql")
    assert reading(sql) is None


def test_the_key_and_matching_types_are_needed():
    assert reading(LISTED, keys=None) is None
    assert reading(LISTED, keys={"item": [("label",)]}) is None
    assert reading(LISTED, keys={"item": [("item_id", "label")]}) is None  # a key of two columns does not make item_id unique
    assert reading(LISTED, types={"item": {"item_id": "INT"}, "sale": {"item_id": "INT"}}) == JOINED
    assert reading(LISTED, types={"item": {"item_id": "INT"}, "sale": {"item_id": "VARCHAR"}}) is None
    assert reading(LISTED, types={"item": {"item_id": "DOUBLE"}, "sale": {"item_id": "DOUBLE"}}) is None


SPELLINGS = [
    LISTED,
    LISTED.replace("s.item_id IN", "i.item_id IN"),
    LISTED.replace(" WHERE x.label = 'S' OR x.label = 'I'", ""),
    LISTED.replace("x.label = 'S' OR x.label = 'I'", "NOT x.label = 'S'"),
    LISTED + " GROUP BY s.buyer HAVING COUNT(*) > 1",
    "SELECT s.buyer, i.label FROM item AS i, sale AS s WHERE i.item_id = s.item_id AND s.item_id IN (SELECT item_id FROM item WHERE label IS NULL OR label <> 'I')",
]


@pytest.mark.parametrize("seed", range(4))
def test_the_reading_returns_the_rows_duckdb_returns_while_the_key_holds(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    pairs = [(sql, reading(sql)) for sql in SPELLINGS]
    assert all(read is not None for _, read in pairs)
    nonempty = differences = 0
    for trial in range(40):
        unique = trial % 2 == 0
        con = duckdb.connect()
        con.execute("CREATE TABLE item (item_id INT, label VARCHAR)")
        con.execute("CREATE TABLE sale (buyer INT, item_id INT, paid INT)")
        ids = rng.sample([3, 7, 8, 9], rng.randint(0, 4)) if unique else [rng.choice([3, 7, 8]) for _ in range(rng.randint(0, 5))]
        for item_id in ids + ([None, None] if rng.random() < 0.5 else []):  # NULL keys repeat freely: a key speaks of values
            con.execute("INSERT INTO item VALUES (?, ?)", [item_id, rng.choice([None, "S", "I", "T"])])
        for _ in range(rng.randint(0, 10)):
            con.execute("INSERT INTO sale VALUES (?, ?, ?)", [rng.choice([None, 1, 2, 3]), rng.choice([None, 3, 7, 8, 9]), rng.choice([None, 1, 2, 2, 3])])
        for sql, read in pairs:
            expected = sorted(con.execute(sql).fetchall(), key=repr)
            got = sorted(con.execute(read).fetchall(), key=repr)
            if unique:
                assert expected == got, (sql, read)
                nonempty += bool(expected)
            else:
                differences += expected != got
    assert nonempty > 20
    assert differences > 0  # two items under one id: the list finds the other one, the joined row does not


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402


def facts(key: bool):
    return {"item": TableConstraints(keys=(("item_id",),) if key else ()), "sale": TableConstraints()}


def test_the_two_spellings_are_proved_equal_only_with_the_key():
    assert prove_equivalent_algebraic(LISTED, JOINED, schema=SCHEMA, constraints=facts(True), compare_names=False, dialect="mysql").proven
    assert not prove_equivalent_algebraic(LISTED, JOINED, schema=SCHEMA, constraints=facts(False), compare_names=False, dialect="mysql").proven


def test_it_is_a_narrow_retry_rule():
    with shape_fallback.tracking() as fired:
        normalize(LISTED, schema=SCHEMA, dialect="mysql", keys=KEYS)
    assert "joined_key_lists" in fired
