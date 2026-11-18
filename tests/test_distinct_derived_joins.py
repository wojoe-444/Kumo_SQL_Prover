"""Under ``DISTINCT`` or in an ``IN`` list, a derived table that only joins and projects is read as its join (``distinct_derived_joins``)."""

import random

import pytest
import sqlglot
from sqlglot import exp

from kumo_sql_prover.distinct_derived_joins import read_derived_join

SCHEMA = {"item": ["item_id", "label"], "sale": ["buyer", "item_id", "paid"]}
DERIVED = (
    "SELECT DISTINCT x.item_id, x.label FROM (SELECT s.item_id, i.label FROM sale AS s JOIN item AS i ON s.item_id = i.item_id) AS x "
    "WHERE x.item_id NOT IN (SELECT k.item_id FROM sale AS k WHERE k.paid > 5)"
)
JOINED = (
    "SELECT DISTINCT s.item_id AS item_id, i.label AS label FROM sale AS s JOIN item AS i ON s.item_id = i.item_id "
    "WHERE NOT s.item_id IN (SELECT k.item_id FROM sale AS k WHERE k.paid > 5)"
)


def reading(sql: str, schema=SCHEMA) -> str | None:
    result = read_derived_join(sqlglot.parse_one(sql, read="mysql"), schema)
    return result.sql(dialect="mysql") if result is not None else None


def list_reading(sql: str, schema=SCHEMA) -> str | None:
    tree = sqlglot.parse_one(sql, read="mysql")
    test = tree.find(exp.In)
    result = read_derived_join(test.args["query"].this, schema)
    if result is None:
        return None
    test.args["query"].set("this", result)
    return tree.sql(dialect="mysql")


BARE = "SELECT DISTINCT item_id, label FROM (SELECT s.item_id, i.label FROM sale AS s JOIN item AS i ON s.item_id = i.item_id) AS x WHERE item_id NOT IN (SELECT item_id FROM sale WHERE paid > 5)"
RENAMED = (
    "SELECT DISTINCT x.n AS shown FROM (SELECT s.item_id AS n, i.label AS l, s.paid AS p FROM sale AS s, item AS i WHERE s.item_id = i.item_id AND i.label <> 'I') AS x "
    "WHERE x.p > 1 AND EXISTS (SELECT 1 FROM sale AS s WHERE s.item_id = x.n AND s.paid < x.p)"
)
LISTED = "SELECT b.buyer FROM sale AS b WHERE b.item_id IN (SELECT x.item_id FROM (SELECT s.item_id, i.label FROM sale AS s JOIN item AS i ON s.item_id = i.item_id) AS x WHERE x.label = 'S')"


def test_the_derived_table_is_read_as_its_join():
    assert reading(DERIVED) == JOINED
    assert reading(JOINED) is None
    # bare names: the outer ones are the derived table's, the nested ones their own table's
    assert reading(BARE) == (
        "SELECT DISTINCT s.item_id AS item_id, i.label AS label FROM sale AS s JOIN item AS i ON s.item_id = i.item_id WHERE NOT s.item_id IN (SELECT sale.item_id FROM sale WHERE sale.paid > 5)"
    )
    # renamed outputs, a WHERE inside, and a nested query that names its source like one of the join's tables
    assert reading(RENAMED) == (
        "SELECT DISTINCT s.item_id AS shown FROM sale AS s, item AS i WHERE s.item_id = i.item_id AND i.label <> 'I' AND s.paid > 1 "
        "AND EXISTS(SELECT 1 FROM sale AS kns0_s WHERE kns0_s.item_id = s.item_id AND kns0_s.paid < s.paid)"
    )
    # the select of a list, with or without DISTINCT
    assert list_reading(LISTED) == "SELECT b.buyer FROM sale AS b WHERE b.item_id IN (SELECT s.item_id AS item_id FROM sale AS s JOIN item AS i ON s.item_id = i.item_id WHERE i.label = 'S')"


LEFT_ALONE = [
    DERIVED.replace("SELECT DISTINCT x.item_id", "SELECT x.item_id"),  # no DISTINCT: the solver reads it
    DERIVED.replace(" WHERE x.item_id NOT IN (SELECT k.item_id FROM sale AS k WHERE k.paid > 5)", ""),  # no nested test
    DERIVED.replace("(SELECT s.item_id, i.label FROM", "(SELECT DISTINCT s.item_id, i.label FROM"),
    DERIVED.replace("ON s.item_id = i.item_id)", "ON s.item_id = i.item_id LIMIT 5)"),
    DERIVED.replace("ON s.item_id = i.item_id)", "ON s.item_id = i.item_id GROUP BY s.item_id, i.label)"),
    DERIVED.replace("JOIN item AS i ON", "LEFT JOIN item AS i ON"),
    DERIVED.replace("(SELECT s.item_id, i.label FROM", "(SELECT s.item_id + 0 AS item_id, i.label FROM"),
    DERIVED.replace("(SELECT s.item_id, i.label FROM", "(SELECT s.item_id, label FROM"),  # a bare name inside
    DERIVED.replace("FROM sale AS s JOIN item AS i ON s.item_id = i.item_id", "FROM sale AS s"),  # no join: other rules read a filter
    DERIVED.replace(") AS x WHERE", ") AS x JOIN item AS z ON z.item_id = x.item_id WHERE"),
    DERIVED.replace("SELECT DISTINCT x.item_id, x.label FROM", "SELECT DISTINCT x.item_id, x.label, x.paid FROM"),  # not an output of the derived table
    DERIVED.replace("WHERE k.paid > 5)", "WHERE k.paid > i.item_id)"),  # a name from further out that the join would capture
    DERIVED.replace("WHERE k.paid > 5)", "WHERE k.paid > 5 AND label = 'S')").replace("FROM sale AS k WHERE", "FROM sale AS k, item AS m WHERE"),  # a bare name in a nested join
    DERIVED + " ORDER BY x.item_id",
    DERIVED.replace("WHERE k.paid > 5)", "WHERE k.paid > 5 AND RAND() < 0.5)"),
    DERIVED.replace("SELECT DISTINCT x.item_id, x.label FROM", "SELECT DISTINCT x.* FROM"),
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_shapes_are_left_alone(sql):
    sqlglot.parse_one(sql, read="mysql")
    assert reading(sql) is None


def test_a_derived_table_elsewhere_is_left_to_the_solver():
    inner = "SELECT x.item_id FROM (SELECT s.item_id, i.label FROM sale AS s JOIN item AS i ON s.item_id = i.item_id) AS x WHERE x.label = 'S'"
    for holder in (f"SELECT b.buyer FROM sale AS b WHERE EXISTS ({inner})", f"SELECT b.buyer FROM sale AS b WHERE b.item_id = ({inner})", f"SELECT d.item_id FROM ({inner}) AS d"):
        tree = sqlglot.parse_one(holder, read="mysql")
        nested = next(node for node in tree.find_all(exp.Select) if node is not tree and isinstance(node.args.get("from_", node.args.get("from")).this, exp.Subquery))
        assert read_derived_join(nested, SCHEMA) is None, holder


@pytest.mark.parametrize("seed", range(4))
def test_the_reading_returns_the_rows_duckdb_returns(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    pairs = [(sql, reading(sql)) for sql in (DERIVED, BARE, RENAMED)] + [(LISTED, list_reading(LISTED)), (LISTED.replace("b.item_id IN", "b.item_id NOT IN"), list_reading(LISTED.replace("b.item_id IN", "b.item_id NOT IN")))]
    assert all(read is not None for _, read in pairs)
    nonempty = 0
    for _ in range(30):
        con = duckdb.connect()
        con.execute("CREATE TABLE item (item_id INT, label VARCHAR)")
        con.execute("CREATE TABLE sale (buyer INT, item_id INT, paid INT)")
        for _ in range(rng.randint(0, 6)):
            con.execute("INSERT INTO item VALUES (?, ?)", [rng.choice([None, 3, 7, 7, 8]), rng.choice([None, "S", "S", "I"])])
        for _ in range(rng.randint(0, 10)):
            con.execute("INSERT INTO sale VALUES (?, ?, ?)", [rng.choice([None, 1, 2]), rng.choice([None, 3, 7, 8, 8]), rng.choice([None, 1, 2, 4, 9])])
        for sql, read in pairs:
            expected = sorted(con.execute(sql).fetchall(), key=repr)
            assert expected == sorted(con.execute(read).fetchall(), key=repr), (sql, read)
            nonempty += bool(expected)
    assert nonempty > 30


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

ASKED = "SELECT i.item_id, i.label FROM item AS i WHERE i.item_id IN (SELECT s.item_id FROM sale AS s) AND NOT EXISTS (SELECT 1 FROM sale AS k WHERE k.paid > 5 AND k.item_id = i.item_id)"


def facts(key: bool):
    return {
        "item": TableConstraints(not_null=frozenset({"item_id", "label"}), keys=(("item_id",),) if key else ()),
        "sale": TableConstraints(not_null=frozenset({"buyer", "item_id", "paid"}), foreign_keys=((("item_id",), "item", ("item_id",)),)),
    }


def test_items_sold_and_never_above_a_price_through_a_derived_join_and_as_tests_of_the_item():
    assert prove_equivalent_algebraic(BARE, ASKED, schema=SCHEMA, constraints=facts(True), compare_names=False, dialect="mysql").proven
    assert not prove_equivalent_algebraic(BARE, ASKED, schema=SCHEMA, constraints=facts(False), compare_names=False, dialect="mysql").proven
    assert not prove_equivalent_algebraic(BARE, ASKED.replace("k.paid > 5", "k.paid > 6"), schema=SCHEMA, constraints=facts(True), compare_names=False, dialect="mysql").proven


def test_it_is_a_narrow_retry_rule():
    constraints = facts(True)
    with shape_fallback.tracking() as fired:
        normalize(BARE, schema=SCHEMA, dialect="mysql", keys={t: list(c.keys) for t, c in constraints.items()}, not_null={t: c.not_null for t, c in constraints.items()})
    assert "distinct_derived_joins" in fired
