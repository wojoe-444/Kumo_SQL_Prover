"""A row that passes ``NOT EXISTS`` over its own tables is not itself a row the test looks for (``own_row_witness``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.own_row_witness import add_own_row_tests

NOT_NULL = {"item": frozenset({"item_id", "label"}), "sale": frozenset({"buyer", "item_id", "paid"})}
ABSENT = (
    "SELECT DISTINCT s.buyer FROM sale AS s JOIN item AS i ON s.item_id = i.item_id WHERE (i.label = 'S' OR i.label = 'I') "
    "AND NOT EXISTS (SELECT 1 FROM sale AS k JOIN item AS j ON k.item_id = j.item_id WHERE (j.label = 'S' OR j.label = 'I') AND j.label = 'I' AND k.buyer = s.buyer)"
)
ADDED = (
    "SELECT DISTINCT s.buyer FROM sale AS s JOIN item AS i ON s.item_id = i.item_id WHERE (i.label = 'S' OR i.label = 'I') "
    "AND NOT EXISTS(SELECT 1 FROM sale AS k JOIN item AS j ON k.item_id = j.item_id WHERE (j.label = 'S' OR j.label = 'I') AND j.label = 'I' AND k.buyer = s.buyer) AND NOT i.label = 'I'"
)


def reading(sql: str, not_null=NOT_NULL) -> str | None:
    result = add_own_row_tests(sqlglot.parse_one(sql, read="mysql"), not_null)
    return result.sql(dialect="mysql") if result is not None else None


def test_the_row_must_fail_what_the_subquery_looks_for():
    assert reading(ABSENT) == ADDED
    assert reading(ADDED) is None  # not added twice
    # one table, several remaining conditions
    single = reading("SELECT s.buyer FROM sale AS s WHERE s.paid < 9 AND NOT EXISTS (SELECT 1 FROM sale AS k WHERE k.buyer = s.buyer AND k.paid > 3 AND k.item_id IN (7, 8))")
    assert single is not None and single.endswith("AND NOT (s.paid > 3 AND s.item_id IN (7, 8))")
    # the subquery over one of the select's two tables
    part = reading("SELECT s.buyer FROM sale AS s, item AS i WHERE s.item_id = i.item_id AND NOT EXISTS (SELECT 1 FROM sale AS k WHERE k.buyer = s.buyer AND k.paid BETWEEN 2 AND 3)")
    assert part is not None and part.endswith("AND NOT s.paid BETWEEN 2 AND 3")
    # a tie to another column of the row stays among the conditions the row must fail
    other = reading("SELECT s.buyer FROM sale AS s WHERE NOT EXISTS (SELECT 1 FROM sale AS k WHERE k.buyer = s.buyer AND k.paid = 4 AND k.item_id > 2)".replace("k.buyer = s.buyer AND ", ""))
    assert other is not None and other.endswith("AND NOT (s.paid = 4 AND s.item_id > 2)")


LEFT_ALONE = [
    ABSENT.replace("AND NOT EXISTS", "AND EXISTS"),
    ABSENT.replace(" AND j.label = 'I' AND k.buyer = s.buyer", " AND k.buyer = s.buyer"),  # nothing remains: the row would always be found
    ABSENT.replace("JOIN item AS i ON", "LEFT JOIN item AS i ON"),  # the row's item may be padding
    ABSENT.replace("FROM sale AS k JOIN item AS j ON", "FROM sale AS k LEFT JOIN item AS j ON"),
    ABSENT.replace("j.label = 'I' AND", "j.label = s.buyer AND"),  # not a comparison with a literal
    ABSENT.replace("j.label = 'I' AND", "UPPER(j.label) = 'I' AND"),
    ABSENT.replace("AND k.buyer = s.buyer)", "AND k.buyer = s.buyer LIMIT 1)"),
    ABSENT.replace("SELECT 1 FROM sale AS k", "SELECT COUNT(*) FROM sale AS k"),
    ABSENT.replace("AND k.buyer = s.buyer)", "AND k.buyer = s.buyer GROUP BY k.buyer)"),
    ABSENT.replace("FROM sale AS k JOIN item AS j ON k.item_id = j.item_id", "FROM sale AS k JOIN item AS j ON k.item_id = j.item_id JOIN item AS j2 ON j2.item_id = k.item_id"),  # a table read twice
    ABSENT.replace("FROM sale AS k JOIN item AS j ON k.item_id = j.item_id WHERE (j.label = 'S' OR j.label = 'I') AND j.label = 'I'", "FROM sale AS k WHERE k.paid IN (SELECT m.paid FROM sale AS m)"),
    ABSENT.replace("j.label = 'I' AND", "label = 'I' AND"),  # a bare name
    ABSENT.replace("j.label = 'I' AND", "j.label = 'I' AND RAND() < 0.5 AND"),
    ABSENT.replace("NOT EXISTS (", "(s.buyer = 1 OR NOT EXISTS (") + ")",  # under OR
    "SELECT d.buyer FROM (SELECT * FROM sale) AS d WHERE NOT EXISTS (SELECT 1 FROM sale AS k WHERE k.buyer = d.buyer AND k.paid > 3)",  # the row is not a table's row
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_shapes_are_left_alone(sql):
    sqlglot.parse_one(sql, read="mysql")
    assert reading(sql) is None


def test_not_null_facts_are_needed():
    assert reading(ABSENT, None) is None
    assert reading(ABSENT, {"item": frozenset({"item_id"}), "sale": NOT_NULL["sale"]}) is None  # the label may be NULL
    assert reading(ABSENT, {"item": NOT_NULL["item"], "sale": frozenset({"item_id"})}) is None  # the tie on buyer may be NULL


SPELLINGS = [
    ABSENT,
    "SELECT s.buyer FROM sale AS s WHERE s.paid < 9 AND NOT EXISTS (SELECT 1 FROM sale AS k WHERE k.buyer = s.buyer AND k.paid > 3 AND k.item_id IN (7, 8))",
    "SELECT s.buyer FROM sale AS s, item AS i WHERE s.item_id = i.item_id AND NOT EXISTS (SELECT 1 FROM sale AS k WHERE k.buyer = s.buyer AND k.paid BETWEEN 2 AND 3)",
    "SELECT s.buyer FROM sale AS s WHERE NOT EXISTS (SELECT 1 FROM sale AS k WHERE k.paid = 4 AND k.item_id > 2)",
    "SELECT s.buyer, i.label FROM sale AS s JOIN item AS i ON s.item_id = i.item_id WHERE NOT EXISTS (SELECT 1 FROM item AS j, sale AS k WHERE k.item_id = j.item_id AND k.buyer = s.buyer AND NOT (j.label = 'S' OR k.paid < 2))",
]


@pytest.mark.parametrize("seed", range(4))
def test_the_reading_returns_the_rows_duckdb_returns_while_the_facts_hold(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    pairs = [(sql, reading(sql)) for sql in SPELLINGS]
    assert all(read is not None for _, read in pairs)
    nonempty = differences = 0
    for trial in range(40):
        filled = trial % 2 == 0
        blank = [] if filled else [None]
        con = duckdb.connect()
        con.execute("CREATE TABLE item (item_id INT, label VARCHAR)")
        con.execute("CREATE TABLE sale (buyer INT, item_id INT, paid INT)")
        for _ in range(rng.randint(0, 5)):
            con.execute("INSERT INTO item VALUES (?, ?)", [rng.choice([3, 7, 8]), rng.choice(["S", "S", "I", "T"] + blank)])
        for _ in range(rng.randint(0, 9)):
            con.execute("INSERT INTO sale VALUES (?, ?, ?)", [rng.choice([1, 2, 3] + blank), rng.choice([3, 7, 8, 9]), rng.choice([1, 2, 3, 4] + blank)])
        for sql, read in pairs:
            expected = sorted(con.execute(sql).fetchall(), key=repr)
            got = sorted(con.execute(read).fetchall(), key=repr)
            if filled:
                assert expected == got, (sql, read)
                nonempty += bool(expected)
            else:
                differences += expected != got
    assert nonempty > 20
    assert differences > 0  # a NULL makes the added test NULL on a row the NOT EXISTS keeps


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

SCHEMA = {"item": ["item_id", "label"], "sale": ["buyer", "item_id", "paid"]}
TYPES = {"item": {"label": "STRING"}}
GROUPED = "SELECT buyer FROM sale JOIN item ON sale.item_id = item.item_id WHERE label IN ('S', 'I') GROUP BY buyer HAVING SUM(CASE WHEN label = 'I' THEN 1 ELSE 0 END) = 0"
LISTS = (
    "SELECT DISTINCT sale.buyer FROM item JOIN sale ON sale.item_id = item.item_id WHERE label = 'S' "
    "AND sale.buyer NOT IN (SELECT sale.buyer FROM item JOIN sale ON sale.item_id = item.item_id WHERE label = 'I')"
)


def facts(not_null: bool = True):
    return {
        "item": TableConstraints(not_null=frozenset({"item_id", "label"} if not_null else ()), keys=(("item_id",),)),
        "sale": TableConstraints(not_null=frozenset({"buyer", "item_id"} if not_null else ()), foreign_keys=((("item_id",), "item", ("item_id",)),)),
    }


def test_never_bought_one_label_as_a_group_test_and_as_two_lists():
    options = dict(schema=SCHEMA, compare_names=False, dialect="mysql", types=TYPES)
    assert prove_equivalent_algebraic(GROUPED, LISTS, constraints=facts(), **options).proven
    assert not prove_equivalent_algebraic(GROUPED, LISTS.replace("label = 'S'", "label = 'T'"), constraints=facts(), **options).proven
    assert not prove_equivalent_algebraic(GROUPED, LISTS, constraints=facts(False), **options).proven


def test_it_is_a_narrow_retry_rule():
    constraints = facts()
    with shape_fallback.tracking() as fired:
        normalize(GROUPED, schema=SCHEMA, dialect="mysql", types=TYPES, keys={t: list(c.keys) for t, c in constraints.items()}, not_null={t: c.not_null for t, c in constraints.items()})
    assert "own_row_witness" in fired
