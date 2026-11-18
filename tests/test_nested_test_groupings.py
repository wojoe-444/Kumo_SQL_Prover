"""A ``GROUP BY`` that lists exactly its outputs, around joined tables and a nested test, is ``SELECT DISTINCT`` (``nested_test_groupings``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.nested_test_groupings import group_as_distinct

GROUPED = (
    "SELECT a.item_id, b.label FROM sale AS a, item AS b WHERE a.item_id = b.item_id "
    "AND a.item_id NOT IN (SELECT k.item_id FROM sale AS k WHERE k.paid > 5) GROUP BY a.item_id, b.label"
)
DISTINCT = (
    "SELECT DISTINCT a.item_id, b.label FROM sale AS a, item AS b WHERE a.item_id = b.item_id "
    "AND NOT a.item_id IN (SELECT k.item_id FROM sale AS k WHERE k.paid > 5)"
)
DERIVED = "SELECT item_id, label FROM (SELECT s.item_id, i.label FROM sale AS s JOIN item AS i ON s.item_id = i.item_id) AS x WHERE EXISTS (SELECT 1 FROM sale AS k WHERE k.item_id = x.item_id AND k.paid > 5) GROUP BY label, item_id"


def reading(sql: str) -> str | None:
    result = group_as_distinct(sqlglot.parse_one(sql, read="mysql"))
    return result.sql(dialect="mysql") if result is not None else None


def test_the_grouping_is_read_as_distinct():
    assert reading(GROUPED) == DISTINCT
    assert reading(DISTINCT) is None
    assert reading(GROUPED.replace("SELECT a.item_id, b.label FROM", "SELECT b.label AS l, a.item_id AS i FROM")).startswith("SELECT DISTINCT b.label AS l, a.item_id AS i FROM")
    # one derived table that joins
    assert reading(DERIVED) == DERIVED.replace("SELECT item_id, label FROM", "SELECT DISTINCT item_id, label FROM").replace(" GROUP BY label, item_id", "").replace("EXISTS (", "EXISTS(")


LEFT_ALONE = [
    GROUPED.replace("SELECT a.item_id, b.label FROM", "SELECT a.item_id FROM"),  # a key that is not listed: still one row per pair
    GROUPED.replace("GROUP BY a.item_id, b.label", "GROUP BY a.item_id"),
    GROUPED.replace("SELECT a.item_id, b.label FROM", "SELECT a.item_id, b.label, COUNT(*) FROM"),
    GROUPED + " HAVING COUNT(*) > 1",
    GROUPED + " ORDER BY a.item_id",
    GROUPED + " LIMIT 2",
    GROUPED.replace("GROUP BY a.item_id, b.label", "GROUP BY a.item_id, b.label WITH ROLLUP"),
    GROUPED.replace(" AND a.item_id NOT IN (SELECT k.item_id FROM sale AS k WHERE k.paid > 5)", ""),  # no nested test: the solver reads it
    "SELECT a.item_id FROM sale AS a WHERE a.item_id NOT IN (SELECT k.item_id FROM sale AS k WHERE k.paid > 5) GROUP BY a.item_id",  # one table: other rules read it
    GROUPED.replace("SELECT a.item_id, b.label FROM", "SELECT a.item_id + 1, b.label FROM"),
    GROUPED.replace("SELECT a.item_id, b.label FROM", "SELECT DISTINCT a.item_id, b.label FROM"),
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_shapes_are_left_alone(sql):
    sqlglot.parse_one(sql, read="mysql")
    assert reading(sql) is None


@pytest.mark.parametrize("seed", range(3))
def test_the_reading_returns_the_rows_duckdb_returns(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    pairs = [(sql, reading(sql)) for sql in (GROUPED, DERIVED, GROUPED.replace("SELECT a.item_id, b.label FROM", "SELECT b.label AS l, a.item_id AS i FROM"))]
    assert all(read is not None for _, read in pairs)
    nonempty = 0
    for _ in range(30):
        con = duckdb.connect()
        con.execute("CREATE TABLE item (item_id INT, label VARCHAR)")
        con.execute("CREATE TABLE sale (buyer INT, item_id INT, paid INT)")
        for _ in range(rng.randint(0, 6)):
            con.execute("INSERT INTO item VALUES (?, ?)", [rng.choice([None, 3, 7, 7, 8]), rng.choice([None, "S", "I"])])
        for _ in range(rng.randint(0, 10)):
            con.execute("INSERT INTO sale VALUES (?, ?, ?)", [rng.choice([None, 1, 2]), rng.choice([None, 3, 7, 8, 8]), rng.choice([None, 1, 4, 9])])
        for sql, read in pairs:
            expected = sorted(con.execute(sql).fetchall(), key=repr)
            assert expected == sorted(con.execute(read).fetchall(), key=repr), (sql, read)
            nonempty += bool(expected)
    assert nonempty > 20


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

SCHEMA = {"item": ["item_id", "label"], "sale": ["buyer", "item_id", "paid"]}
ASKED = "SELECT i.item_id, i.label FROM item AS i WHERE i.item_id IN (SELECT s.item_id FROM sale AS s) AND NOT EXISTS (SELECT 1 FROM sale AS k WHERE k.paid > 5 AND k.item_id = i.item_id)"


def facts(key: bool):
    return {
        "item": TableConstraints(not_null=frozenset({"item_id", "label"}), keys=(("item_id",),) if key else ()),
        "sale": TableConstraints(not_null=frozenset({"buyer", "item_id", "paid"}), foreign_keys=((("item_id",), "item", ("item_id",)),)),
    }


def test_items_sold_and_never_above_a_price_as_a_grouped_join_and_as_tests_of_the_item():
    assert prove_equivalent_algebraic(GROUPED, ASKED, schema=SCHEMA, constraints=facts(True), compare_names=False, dialect="mysql").proven
    # without the key two items may share an id and a label: the grouping lists them once
    assert not prove_equivalent_algebraic(GROUPED, ASKED, schema=SCHEMA, constraints=facts(False), compare_names=False, dialect="mysql").proven


def test_it_is_a_narrow_retry_rule():
    constraints = facts(True)
    with shape_fallback.tracking() as fired:
        normalize(GROUPED, schema=SCHEMA, dialect="mysql", keys={t: list(c.keys) for t, c in constraints.items()}, not_null={t: c.not_null for t, c in constraints.items()})
    assert "nested_test_groupings" in fired
