"""A presence test a declared foreign key answers is dropped (``referenced_presence``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.referenced_presence import drop_referenced_presence

SCHEMA = {"item": ["item_id", "label"], "sale": ["item_id", "sold_on", "buyer"]}
NOT_NULL = {"item": frozenset({"item_id"}), "sale": frozenset()}
FOREIGN = {"item": [(("item_id",), "sale", ("item_id",))]}
LISTED = "SELECT p.item_id FROM item AS p WHERE p.item_id IN (SELECT s.item_id FROM sale AS s) AND p.label <> 'x'"
ASKED = "SELECT p.item_id FROM item AS p WHERE EXISTS (SELECT 1 FROM sale AS s WHERE s.item_id = p.item_id) AND p.label <> 'x'"
PLAIN = "SELECT p.item_id FROM item AS p WHERE p.label <> 'x'"


def rewritten(sql: str, schema=SCHEMA, not_null=NOT_NULL, foreign=FOREIGN, types=None, pick=None) -> str | None:
    tree = sqlglot.parse_one(sql, read="mysql")
    select = tree if pick is None else pick(tree)
    result = drop_referenced_presence(select, schema, not_null, foreign, types)
    if result is None:
        return None
    if select is tree:
        return result.sql(dialect="mysql")
    select.replace(result)
    return tree.sql(dialect="mysql")


def test_a_test_the_foreign_key_answers_is_dropped():
    assert rewritten(LISTED) == PLAIN
    assert rewritten(ASKED) == PLAIN
    assert rewritten(PLAIN) is None
    # the only conjunct: the WHERE goes
    assert rewritten("SELECT p.item_id FROM item AS p WHERE p.item_id IN (SELECT DISTINCT item_id FROM sale)") == "SELECT p.item_id FROM item AS p"
    # spellings: an alias in the list, parentheses, the equality the other way round, bare names the lists place
    assert rewritten(LISTED.replace("SELECT s.item_id FROM sale AS s", "SELECT sale.item_id AS k0 FROM sale")) == PLAIN
    assert rewritten(LISTED.replace("p.item_id IN", "(p.item_id) IN")) == PLAIN
    assert rewritten(ASKED.replace("s.item_id = p.item_id", "p.item_id = s.item_id")) == PLAIN
    assert rewritten(ASKED.replace("s.item_id = p.item_id", "item_id = p.item_id")) == PLAIN
    assert rewritten("SELECT item_id FROM item WHERE item_id IN (SELECT item_id FROM sale)") == "SELECT item_id FROM item"
    # beside an inner join, and as the outer side of a left join, the item is still a real row
    assert rewritten("SELECT p.item_id FROM item AS p JOIN sale AS t ON t.buyer = p.label WHERE p.item_id IN (SELECT item_id FROM sale)") == "SELECT p.item_id FROM item AS p JOIN sale AS t ON t.buyer = p.label"
    assert rewritten("SELECT p.item_id FROM item AS p LEFT JOIN sale AS t ON t.buyer = p.label WHERE p.item_id IN (SELECT item_id FROM sale)") == "SELECT p.item_id FROM item AS p LEFT JOIN sale AS t ON t.buyer = p.label"
    # two rows of the child: each is tested
    two = "SELECT a.item_id FROM item AS a, item AS b WHERE a.item_id IN (SELECT item_id FROM sale) AND EXISTS (SELECT 1 FROM sale AS s WHERE s.item_id = b.item_id)"
    assert rewritten(two) == "SELECT a.item_id FROM item AS a, item AS b"


def test_a_nested_select_is_read_when_it_is_the_one_given():
    def inner(tree):
        return tree.find(sqlglot.exp.Exists).this

    around = "SELECT 1 FROM sale AS t WHERE NOT EXISTS (SELECT 1 FROM item AS p WHERE p.label = t.buyer AND p.item_id IN (SELECT item_id FROM sale))"
    assert rewritten(around, pick=inner) == "SELECT 1 FROM sale AS t WHERE NOT EXISTS(SELECT 1 FROM item AS p WHERE p.label = t.buyer)"
    assert rewritten(around) is None


@pytest.mark.parametrize(
    "sql",
    [
        LISTED.replace("p.item_id IN", "p.item_id NOT IN"),
        ASKED.replace("EXISTS", "NOT EXISTS"),
        LISTED.replace(" AND p.label <> 'x'", " OR p.label <> 'x'"),
        LISTED.replace("p.item_id IN", "p.label IN"),  # not the referencing column
        LISTED.replace("SELECT s.item_id FROM", "SELECT s.buyer FROM"),  # not the referenced column
        LISTED.replace("sale AS s)", "sale AS s WHERE s.buyer = 'b')"),  # only some sales
        LISTED.replace("sale AS s)", "sale AS s LIMIT 1)"),
        LISTED.replace("sale AS s)", "sale AS s GROUP BY s.item_id)"),
        LISTED.replace("sale AS s)", "sale AS s JOIN item AS q ON q.item_id = s.item_id)"),
        LISTED.replace("SELECT s.item_id FROM sale AS s", "SELECT s.item_id FROM sale AS s UNION SELECT 1"),
        LISTED.replace("SELECT s.item_id FROM sale AS s", "SELECT s.item_id + 0 FROM sale AS s"),
        LISTED.replace("SELECT s.item_id FROM sale AS s", "SELECT p.item_id FROM sale AS s"),  # the outer row's own value
        LISTED.replace("SELECT s.item_id FROM sale AS s", "SELECT s.item_id FROM item AS s"),  # another table
        LISTED.replace("SELECT s.item_id FROM sale AS s", "SELECT s.item_id FROM (SELECT * FROM sale) AS s"),
        LISTED.replace("p.item_id IN (SELECT s.item_id FROM sale AS s)", "p.item_id IN (1, 2)"),
        ASKED.replace("s.item_id = p.item_id", "s.item_id = p.item_id AND s.buyer = 'b'"),
        ASKED.replace("s.item_id = p.item_id", "s.item_id >= p.item_id"),
        ASKED.replace("s.item_id = p.item_id", "s.item_id = s.item_id"),  # no outer row
        ASKED.replace("s.item_id = p.item_id", "s.buyer = p.item_id"),
        ASKED.replace("s.item_id = p.item_id", "s.item_id = p.label"),
        ASKED.replace("SELECT 1 FROM sale AS s", "SELECT COUNT(*) FROM sale AS s"),
        ASKED.replace("sale AS s WHERE", "sale AS p WHERE"),  # the inner table takes the name
        # the item may be the NULL padding of an outer join
        "SELECT p.item_id FROM sale AS t LEFT JOIN item AS p ON t.buyer = p.label WHERE p.item_id IN (SELECT item_id FROM sale)",
        "SELECT p.item_id FROM item AS p RIGHT JOIN sale AS t ON t.buyer = p.label WHERE p.item_id IN (SELECT item_id FROM sale)",
        "SELECT p.item_id FROM item AS p FULL JOIN sale AS t ON t.buyer = p.label WHERE p.item_id IN (SELECT item_id FROM sale)",
        "SELECT p.item_id FROM (SELECT * FROM item) AS p WHERE p.item_id IN (SELECT item_id FROM sale)",  # a derived table
        "SELECT item_id FROM item, sale WHERE item_id IN (SELECT item_id FROM sale)",  # whose item_id?
        "SELECT p.item_id FROM item AS p JOIN sale AS t ON p.item_id IN (SELECT item_id FROM sale)",  # in an ON
    ],
)
def test_other_tests_are_left_alone(sql):
    assert rewritten(sql) is None


def test_facts_that_are_needed():
    assert rewritten(LISTED, foreign=None) is None
    assert rewritten(LISTED, not_null=None) is None
    assert rewritten(LISTED, not_null={"item": frozenset({"label"})}) is None  # a NULL item_id has no sale
    assert rewritten(LISTED, foreign={"sale": [(("item_id",), "item", ("item_id",))]}) is None  # the other direction
    assert rewritten(LISTED, foreign={"item": [(("item_id",), "sale", ("buyer",))]}) is None
    assert rewritten(LISTED, foreign={"item": [(("label",), "sale", ("item_id",))]}) is None
    assert rewritten(LISTED, foreign={"item": [(("item_id", "label"), "sale", ("item_id", "buyer"))]}) is None  # several columns
    assert rewritten(LISTED, types={"item": {"item_id": "DOUBLE"}}) is None
    assert rewritten(LISTED, types={"sale": {"item_id": "DOUBLE"}}) is None
    assert rewritten(LISTED, types={"item": {"item_id": "INT"}, "sale": {"item_id": "INT"}}) == PLAIN
    # bare names need the column lists
    bare = "SELECT item_id FROM item WHERE item_id IN (SELECT item_id FROM sale)"
    assert rewritten(bare, schema=None) is None
    assert rewritten(bare, schema={"item": ["item_id"], "sale": ["buyer"]}) is None  # the inner name is the outer row's
    assert rewritten(LISTED, schema=None) == PLAIN  # every name says whose it is


SPELLINGS = [
    LISTED,
    ASKED,
    "SELECT p.item_id, t.buyer FROM item AS p LEFT JOIN sale AS t ON t.buyer = p.label WHERE p.item_id IN (SELECT DISTINCT item_id FROM sale)",
    "SELECT COUNT(*) FROM item WHERE item_id IN (SELECT item_id FROM sale)",
]


@pytest.mark.parametrize("seed", range(3))
def test_the_reading_returns_the_rows_duckdb_returns_when_the_facts_hold(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    differs = nonempty = 0
    for sql in SPELLINGS:
        read = rewritten(sql)
        assert read is not None, sql
        for held in [True] * 5 + [False] * 10:
            con = duckdb.connect()
            con.execute("CREATE TABLE item (item_id INT, label VARCHAR)")
            con.execute("CREATE TABLE sale (item_id INT, sold_on DATE, buyer VARCHAR)")
            ids = [rng.randint(1, 5) for _ in range(rng.randint(0, 5))]
            for value in ids:
                # without the facts: an item with no id
                con.execute("INSERT INTO item VALUES (?, ?)", [value if held or rng.random() < 0.8 else None, rng.choice([None, "a", "b", "x"])])
            sold = list(ids) if held else [v for v in ids if rng.random() < 0.5]  # without the facts: an item with no sale
            for value in sold + [rng.choice([None, 7, 8]) for _ in range(rng.randint(0, 3))]:
                con.execute("INSERT INTO sale VALUES (?, DATE '2019-02-01', ?)", [value, rng.choice([None, "a", "b"])])
            expected = sorted(con.execute(sql).fetchall(), key=repr)
            same = expected == sorted(con.execute(read).fetchall(), key=repr)
            if held:
                assert same, sql
                nonempty += bool(expected)
            else:
                differs += not same
    assert nonempty > 3 and differs > 0  # the facts are what make it true


z3 = pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

TYPES = {"sale": {"sold_on": "DATE"}}
ALL_INSIDE = "SELECT p.item_id, p.label FROM item p JOIN (SELECT item_id FROM sale GROUP BY item_id HAVING MIN(sold_on) >= '2019-01-01' AND MAX(sold_on) <= '2019-03-31') s ON p.item_id = s.item_id"
NONE_OUTSIDE = "SELECT item_id, label FROM item WHERE item_id NOT IN (SELECT item_id FROM sale WHERE sold_on < '2019-01-01' OR sold_on > '2019-03-31')"


def facts(foreign=True, item_id=True):
    return {
        "item": TableConstraints(not_null=frozenset({"item_id"} if item_id else ()), foreign_keys=((("item_id",), "sale", ("item_id",)),) if foreign else ()),
        "sale": TableConstraints(not_null=frozenset({"item_id", "sold_on"})),
    }


def proved(left: str, right: str, constraints) -> bool:
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=constraints, types=TYPES, compare_names=False, dialect="mysql").proven


def test_sold_only_in_the_quarter_two_ways_when_every_item_has_a_sale():
    assert proved(ALL_INSIDE, NONE_OUTSIDE, facts())
    assert proved(LISTED, PLAIN, facts())


def test_without_the_foreign_key_or_the_not_null_nothing_is_proved():
    assert not proved(ALL_INSIDE, NONE_OUTSIDE, facts(foreign=False))  # an item with no sale is only in the second
    assert not proved(LISTED, PLAIN, facts(foreign=False))
    assert not proved(LISTED, PLAIN, facts(item_id=False))


def test_the_two_spellings_differ_on_an_item_with_no_sale():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE item (item_id INT, label VARCHAR)")
    con.execute("CREATE TABLE sale (item_id INT, sold_on DATE, buyer VARCHAR)")
    con.execute("INSERT INTO item VALUES (1, 'a'), (2, 'b')")
    con.execute("INSERT INTO sale VALUES (1, DATE '2019-02-01', 'x')")
    assert con.execute(ALL_INSIDE).fetchall() == [(1, "a")]
    assert sorted(con.execute(NONE_OUTSIDE).fetchall()) == [(1, "a"), (2, "b")]
