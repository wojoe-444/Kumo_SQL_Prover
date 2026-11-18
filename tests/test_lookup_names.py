"""Bare column names around a value looked up by key are given their tables (``lookup_names``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.lookup_names import name_lookup_columns

SCHEMA = {"item": ["item_id", "label"], "sale": ["buyer", "item_id", "paid"]}
KEYS = {"item": [("label",)]}
BARE = "SELECT buyer FROM sale WHERE item_id = (SELECT item_id FROM item WHERE label = 'S')"
NAMED = "SELECT buyer FROM sale WHERE sale.item_id = (SELECT item.item_id FROM item WHERE item.label = 'S')"


def naming(sql: str, schema=SCHEMA, keys=KEYS) -> str | None:
    tree, named = name_lookup_columns(sqlglot.parse_one(sql, read="mysql"), schema, keys)
    return tree.sql(dialect="mysql") if named else None


def test_the_names_are_written_in():
    assert naming(BARE) == NAMED
    assert naming(NAMED) is None
    # the subquery on the left, another comparison, an alias, and a test under OR
    assert naming("SELECT buyer FROM sale AS s WHERE (SELECT item_id FROM item AS i WHERE label = 'S') < item_id OR paid = 2") == (
        "SELECT buyer FROM sale AS s WHERE (SELECT i.item_id FROM item AS i WHERE i.label = 'S') < s.item_id OR paid = 2"
    )
    # inside a list, where the outer select is the list's own
    assert naming(f"SELECT DISTINCT buyer FROM sale WHERE buyer IN ({BARE})") == f"SELECT DISTINCT buyer FROM sale WHERE buyer IN ({NAMED})"
    # two outer sources: the name only one of them has
    assert naming("SELECT s.buyer FROM sale AS s, item AS i WHERE paid = (SELECT item_id FROM item AS x WHERE label = 'S')") == (
        "SELECT s.buyer FROM sale AS s, item AS i WHERE s.paid = (SELECT x.item_id FROM item AS x WHERE x.label = 'S')"
    )


LEFT_ALONE = [
    BARE.replace("FROM sale WHERE item_id =", "FROM sale AS s, item AS i WHERE item_id ="),  # two outer sources have the name
    BARE.replace("WHERE item_id =", "WHERE shelf ="),  # no outer source has it: a read of a query further out
    BARE.replace("WHERE label = 'S'", "WHERE shelf = 'S'"),  # the looked-up table does not have it
    BARE.replace("FROM item WHERE", "FROM item JOIN sale AS z ON z.item_id = item.item_id WHERE"),
    BARE.replace("SELECT item_id FROM item", "SELECT MAX(item_id) FROM item"),
    BARE.replace("label = 'S'", "label = 'S' LIMIT 1"),
    BARE.replace("label = 'S'", "label = 'S' AND item_id IN (SELECT item_id FROM sale)"),
    BARE.replace("FROM sale WHERE", "FROM (SELECT * FROM sale) AS sale WHERE"),
    "WITH item AS (SELECT 1 AS item_id, 'S' AS label) " + BARE,  # a WITH table of that name
    BARE.replace("WHERE item_id =", "WHERE `item_id` ="),
    BARE.replace("WHERE item_id = (", "WHERE item_id IN ("),  # a list, not a single value
    "SELECT buyer, item_id = (SELECT item_id FROM item WHERE label = 'S') FROM sale",  # an output, not a WHERE test
    "SELECT buyer FROM sale GROUP BY buyer HAVING MAX(item_id) = (SELECT item_id FROM item WHERE label = 'S')",
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_shapes_are_left_alone(sql):
    sqlglot.parse_one(sql, read="mysql")
    assert naming(sql) is None


def test_nothing_is_named_without_columns_or_a_key():
    assert naming(BARE, schema=None) is None
    assert naming(BARE, keys=None) is None
    assert naming(BARE, keys={"sale": [("buyer",)]}) is None  # the looked-up table has no declared key
    assert naming(BARE, schema={"item": ["item_id", "label"]}) is None  # the outer table's columns are not known


@pytest.mark.parametrize("seed", range(3))
def test_the_named_query_returns_the_rows_duckdb_returns(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    queries = [BARE, "SELECT buyer FROM sale AS s WHERE (SELECT item_id FROM item AS i WHERE label = 'S') < item_id OR paid = 2", f"SELECT DISTINCT buyer FROM sale WHERE buyer IN ({BARE})"]
    nonempty = 0
    for _ in range(25):
        con = duckdb.connect()
        con.execute("CREATE TABLE item (item_id INT, label VARCHAR)")
        con.execute("CREATE TABLE sale (buyer INT, item_id INT, paid INT)")
        for label in rng.sample(["S", "I", "T"], rng.randint(0, 3)):  # each label once: the lookup finds at most one row
            con.execute("INSERT INTO item VALUES (?, ?)", [rng.choice([None, 3, 7, 8]), label])
        for _ in range(rng.randint(0, 10)):
            con.execute("INSERT INTO sale VALUES (?, ?, ?)", [rng.choice([None, 1, 2, 3]), rng.choice([None, 3, 7, 8, 9]), rng.choice([None, 1, 2])])
        for sql in queries:
            expected = sorted(con.execute(sql).fetchall(), key=repr)
            assert expected == sorted(con.execute(naming(sql)).fetchall(), key=repr), sql
            nonempty += bool(expected)
    assert nonempty > 10


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

LISTED = "SELECT s.buyer FROM sale AS s WHERE s.item_id IN (SELECT i.item_id FROM item AS i WHERE i.label = 'S')"


def facts(label_key: bool):
    return {"item": TableConstraints(keys=(("item_id",),) + ((("label",),) if label_key else ())), "sale": TableConstraints()}


def test_a_value_looked_up_by_a_bare_key_name_is_a_list():
    assert prove_equivalent_algebraic(BARE, LISTED, schema=SCHEMA, constraints=facts(True), compare_names=False, dialect="mysql").proven
    # without the key on the label the lookup may find two rows: nothing is claimed
    assert not prove_equivalent_algebraic(BARE, LISTED, schema=SCHEMA, constraints=facts(False), compare_names=False, dialect="mysql").proven
    # another label is another question
    assert not prove_equivalent_algebraic(BARE, LISTED.replace("'S'", "'T'"), schema=SCHEMA, constraints=facts(True), compare_names=False, dialect="mysql").proven


def test_it_is_a_narrow_retry_rule():
    with shape_fallback.tracking() as fired:
        normalize(BARE, schema=SCHEMA, dialect="mysql", keys={"item": [("label",)]})
    assert "lookup_names" in fired
    with shape_fallback.without("lookup_names"), shape_fallback.tracking() as fired:
        assert "= (SELECT" in normalize(BARE, schema=SCHEMA, dialect="mysql", keys={"item": [("label",)]})
    assert "lookup_names" not in fired
