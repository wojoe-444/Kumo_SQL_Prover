"""The values of a column that a list drawn from that column keeps (``own_column_lists``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.own_column_lists import read_own_column_list

SCHEMA = {"item": ["item_id", "label"], "sale": ["buyer", "item_id", "paid"]}
LISTED = (
    "SELECT DISTINCT buyer FROM sale WHERE buyer IN (SELECT s.buyer FROM sale AS s WHERE s.item_id IN (SELECT i.item_id FROM item AS i WHERE i.label = 'S')) "
    "AND buyer NOT IN (SELECT t.buyer FROM sale AS t WHERE t.item_id = 7)"
)
DIRECT = (
    "SELECT DISTINCT s.buyer AS buyer FROM sale AS s WHERE NOT s.buyer IS NULL AND s.item_id IN (SELECT i.item_id FROM item AS i WHERE i.label = 'S') "
    "AND NOT s.buyer IN (SELECT t.buyer FROM sale AS t WHERE t.item_id = 7)"
)


def reading(sql: str, types=None, schema=SCHEMA) -> str | None:
    result = read_own_column_list(sqlglot.parse_one(sql, read="mysql"), schema, types)
    return result.sql(dialect="mysql") if result is not None else None


def test_the_list_itself_is_read_with_the_other_tests_of_the_value():
    assert reading(LISTED) == DIRECT
    assert reading(DIRECT) is None
    # the outer row under a name of its own, and a list source that carries the table's name
    named = reading(LISTED.replace("SELECT DISTINCT buyer FROM sale WHERE buyer IN", "SELECT DISTINCT x.buyer FROM sale AS x WHERE x.buyer IN").replace("AND buyer NOT IN", "AND x.buyer NOT IN"))
    assert named == DIRECT
    same_name = reading("SELECT DISTINCT buyer FROM sale WHERE buyer IN (SELECT buyer FROM sale WHERE paid > 3) AND buyer <> 4")
    assert same_name is not None and " FROM sale AS " in same_name and same_name.count("FROM sale") == 1 and "<> 4" in same_name
    # a nested test that reads the outer value is moved with it
    moved = reading("SELECT DISTINCT x.buyer FROM sale AS x WHERE x.buyer IN (SELECT s.buyer FROM sale AS s WHERE s.paid > 3) AND EXISTS (SELECT 1 FROM sale AS t WHERE t.buyer = x.buyer AND t.item_id = 7)")
    assert moved == "SELECT DISTINCT s.buyer AS buyer FROM sale AS s WHERE NOT s.buyer IS NULL AND s.paid > 3 AND EXISTS(SELECT 1 FROM sale AS t WHERE t.buyer = s.buyer AND t.item_id = 7)"
    # the list alone, with a nested query in its WHERE
    alone = reading("SELECT DISTINCT buyer FROM sale WHERE buyer IN (SELECT s.buyer FROM sale AS s WHERE s.item_id IN (SELECT i.item_id FROM item AS i WHERE i.label = 'S'))")
    assert alone == "SELECT DISTINCT s.buyer AS buyer FROM sale AS s WHERE NOT s.buyer IS NULL AND s.item_id IN (SELECT i.item_id FROM item AS i WHERE i.label = 'S')"


OUTER = "SELECT DISTINCT x.buyer FROM sale AS x WHERE x.buyer IN (SELECT s.buyer FROM sale AS s WHERE s.paid > 3) AND x.buyer <> 4"
LEFT_ALONE = [
    OUTER.replace("SELECT DISTINCT", "SELECT"),  # the outer rows are counted
    OUTER.replace("x.buyer <> 4", "x.paid <> 4"),  # another column of the outer row
    OUTER.replace("x.buyer <> 4", "EXISTS (SELECT 1 FROM item AS i WHERE i.item_id = x.item_id)"),  # read inside a nested query
    OUTER.replace("SELECT DISTINCT x.buyer FROM", "SELECT DISTINCT x.buyer, x.paid FROM"),
    OUTER.replace("SELECT DISTINCT x.buyer FROM", "SELECT DISTINCT x.buyer + 1 FROM"),
    OUTER.replace("x.buyer IN", "x.buyer NOT IN"),
    OUTER.replace("x.buyer IN", "x.item_id IN"),  # the list is tested against another column
    OUTER.replace("SELECT s.buyer FROM", "SELECT s.item_id FROM"),  # a list of another column
    OUTER.replace("FROM sale AS s WHERE", "FROM item AS s WHERE").replace("s.buyer", "s.item_id").replace("s.paid", "s.item_id"),  # another table
    OUTER.replace("s.paid > 3", "s.paid > x.paid"),  # the list reads the outer row
    OUTER.replace("s.paid > 3", "s.paid > 3 GROUP BY s.buyer"),
    OUTER.replace("s.paid > 3", "s.paid > 3 LIMIT 2"),
    OUTER.replace("SELECT s.buyer FROM", "SELECT MAX(s.buyer) FROM"),
    OUTER.replace("FROM sale AS s WHERE", "FROM sale AS s JOIN item AS i ON i.item_id = s.item_id WHERE"),  # distinct_match_joins reads the join first
    OUTER.replace("FROM sale AS x WHERE", "FROM sale AS x JOIN item AS j ON j.item_id = x.item_id WHERE"),
    OUTER + " ORDER BY x.buyer",
    OUTER + " LIMIT 3",
    OUTER.replace("x.buyer <> 4", "x.buyer <> 4 OR x.buyer IN (SELECT s.buyer FROM sale AS s)").replace("x.buyer IN (SELECT s.buyer FROM sale AS s WHERE s.paid > 3) AND ", ""),  # the list under OR
    OUTER.replace("x.buyer <> 4", "x.buyer <> 4 * RAND()"),
    OUTER.replace("FROM sale AS x WHERE", "FROM (SELECT * FROM sale) AS x WHERE"),
    OUTER.replace("x.buyer <> 4", "x.buyer <> s.item_id"),  # a source further out that carries the list's name
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_shapes_are_left_alone(sql):
    sqlglot.parse_one(sql, read="mysql")
    assert reading(sql) is None


def test_bare_names_are_read_only_with_the_tables_columns():
    bare = OUTER.replace("s.paid > 3", "paid > 3").replace("SELECT s.buyer", "SELECT buyer").replace("sale AS s", "sale")
    assert reading(bare) is not None
    assert reading(bare, schema=None) is None
    assert reading(OUTER, schema=None) is None  # the outer row's columns are not known


def test_two_lists_under_one_name_get_two_names():
    read = reading(TWO_NAMED)
    assert read is not None and read.count("kns0_s.") >= 3 and "kns1_s.buyer FROM sale AS kns1_s" in read and " s." not in read


def test_a_floating_point_column_is_not_read():
    assert reading(OUTER, {"sale": {"buyer": "INT"}}) is not None
    assert reading(OUTER, {"sale": {"buyer": "DOUBLE"}}) is None


TWO_NAMED = "SELECT DISTINCT buyer FROM sale WHERE buyer IN (SELECT s.buyer FROM sale AS s WHERE s.paid > 3) AND buyer NOT IN (SELECT s.buyer FROM sale AS s WHERE s.item_id = 7)"
SPELLINGS = [
    LISTED,
    TWO_NAMED,
    OUTER.replace("x.buyer <> 4", "EXISTS (SELECT 1 FROM item AS s WHERE s.item_id = x.buyer)"),  # a deeper source with the list's name
    OUTER,
    "SELECT DISTINCT buyer FROM sale WHERE buyer IN (SELECT buyer FROM sale WHERE paid > 3) AND buyer <> 4",
    "SELECT DISTINCT x.buyer FROM sale AS x WHERE x.buyer IN (SELECT s.buyer FROM sale AS s WHERE s.paid > 3) AND EXISTS (SELECT 1 FROM sale AS t WHERE t.buyer = x.buyer AND t.item_id = 7)",
    "SELECT DISTINCT buyer FROM sale WHERE buyer IN (SELECT s.buyer FROM sale AS s WHERE s.item_id IN (SELECT i.item_id FROM item AS i WHERE i.label = 'S'))",
    "SELECT DISTINCT x.buyer AS b, x.buyer FROM sale AS x WHERE x.buyer NOT IN (SELECT t.buyer FROM sale AS t WHERE t.item_id = 7) AND x.buyer IN (SELECT DISTINCT s.buyer FROM sale AS s WHERE s.paid > 3 OR s.paid IS NULL) "
    "AND x.buyer NOT IN (SELECT t.paid FROM sale AS t WHERE t.item_id = 3)",
    "SELECT DISTINCT x.buyer FROM sale AS x WHERE x.buyer IN (SELECT s.buyer FROM sale AS s WHERE NOT EXISTS (SELECT 1 FROM sale AS t WHERE t.buyer = s.buyer AND t.paid > s.paid)) AND x.buyer IN (SELECT t.item_id FROM sale AS t)",
]


@pytest.mark.parametrize("seed", range(4))
def test_the_reading_returns_the_rows_duckdb_returns(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    pairs = []
    for sql in SPELLINGS:
        read = reading(sql)
        assert read is not None, sql
        pairs.append((sql, read))
    nonempty = 0
    for _ in range(30):
        # no fact is declared: NULLs and repeated rows everywhere
        con = duckdb.connect()
        con.execute("CREATE TABLE item (item_id INT, label VARCHAR)")
        con.execute("CREATE TABLE sale (buyer INT, item_id INT, paid INT)")
        for _ in range(rng.randint(0, 5)):
            con.execute("INSERT INTO item VALUES (?, ?)", [rng.choice([None, 3, 7, 8]), rng.choice([None, "S", "S", "I"])])
        for _ in range(rng.randint(0, 14)):
            con.execute("INSERT INTO sale VALUES (?, ?, ?)", [rng.choice([None, 1, 2, 3, 4, 5]), rng.choice([None, 3, 7, 8, 8]), rng.choice([None, 1, 4, 5, 9])])
        for sql, read in pairs:
            expected = sorted(con.execute(sql).fetchall(), key=repr)
            assert expected == sorted(con.execute(read).fetchall(), key=repr), (sql, read)
            nonempty += bool(expected)
    assert nonempty > 60


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

JOINED = (
    "SELECT DISTINCT s.buyer FROM sale AS s JOIN item AS i ON s.item_id = i.item_id WHERE i.label = 'S' "
    "AND s.buyer NOT IN (SELECT s2.buyer FROM sale AS s2 JOIN item AS i2 ON s2.item_id = i2.item_id WHERE i2.label = 'I')"
)
TWO_LISTS = (
    "SELECT DISTINCT buyer FROM sale WHERE buyer IN (SELECT s.buyer FROM sale AS s JOIN item AS i ON s.item_id = i.item_id WHERE i.label = 'S') "
    "AND buyer NOT IN (SELECT s.buyer FROM sale AS s JOIN item AS i ON s.item_id = i.item_id WHERE i.label = 'I')"
)


def facts():
    return {"item": TableConstraints(not_null=frozenset({"item_id", "label"})), "sale": TableConstraints(not_null=frozenset({"buyer", "item_id"}))}


def test_bought_one_label_and_never_another_as_a_join_and_as_two_lists():
    assert prove_equivalent_algebraic(JOINED, TWO_LISTS, schema=SCHEMA, constraints=facts(), compare_names=False, dialect="mysql").proven
    # another label in the second list is another question
    assert not prove_equivalent_algebraic(JOINED, TWO_LISTS.replace("i.label = 'I'", "i.label = 'J'"), schema=SCHEMA, constraints=facts(), compare_names=False, dialect="mysql").proven
    # and so is counting the buyers' rows
    assert not prove_equivalent_algebraic(JOINED, TWO_LISTS.replace("SELECT DISTINCT buyer", "SELECT buyer"), schema=SCHEMA, constraints=facts(), compare_names=False, dialect="mysql").proven


def test_it_is_a_narrow_retry_rule():
    constraints = facts()
    with shape_fallback.tracking() as fired:
        normalize(TWO_LISTS, schema=SCHEMA, dialect="mysql", keys={}, not_null={t: c.not_null for t, c in constraints.items()})
    assert "own_column_lists" in fired
