"""A group whose one distinct value is a given literal (``single_value_groups``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.single_value_groups import read_single_value_groups

ONE = "SELECT buyer FROM sale GROUP BY buyer HAVING COUNT(DISTINCT label) = 1 AND MIN(label) = 'S'"
COUNTS = "SELECT buyer FROM sale GROUP BY buyer HAVING COUNT(CASE WHEN label = 'S' THEN 1 END) > 0 AND COUNT(CASE WHEN label <> 'S' THEN 1 END) = 0"


def reading(sql: str, types=None) -> str | None:
    result = read_single_value_groups(sqlglot.parse_one(sql, read="mysql"), types)
    return result.sql(dialect="mysql") if result is not None else None


def test_the_two_tests_are_written_as_two_counts():
    assert reading(ONE) == COUNTS
    assert reading(COUNTS) is None
    assert reading(ONE.replace("MIN(label)", "MAX(label)")) == COUNTS
    # either order, the literal on the left, other tests kept in place
    swapped = reading("SELECT buyer FROM sale GROUP BY buyer HAVING 'S' = MAX(s.label) AND COUNT(*) > 2 AND 1 = COUNT(DISTINCT s.label)".replace("FROM sale", "FROM sale AS s"))
    assert swapped == (
        "SELECT buyer FROM sale AS s GROUP BY buyer HAVING COUNT(CASE WHEN s.label <> 'S' THEN 1 END) = 0 AND COUNT(*) > 2 AND COUNT(CASE WHEN s.label = 'S' THEN 1 END) > 0"
    )
    # a whole number beside a column declared to hold whole numbers
    numbers = ONE.replace("label", "paid").replace("'S'", "4")
    assert reading(numbers, {"sale": {"paid": "INT"}}) == COUNTS.replace("label", "paid").replace("'S'", "4")


LEFT_ALONE = [
    ONE.replace("COUNT(DISTINCT label) = 1 AND ", ""),
    ONE.replace(" AND MIN(label) = 'S'", ""),
    ONE.replace("COUNT(DISTINCT label) = 1", "COUNT(DISTINCT label) = 2"),
    ONE.replace("COUNT(DISTINCT label) = 1", "COUNT(label) = 1"),
    ONE.replace("COUNT(DISTINCT label) = 1", "COUNT(DISTINCT paid) = 1"),  # another column
    ONE.replace("MIN(label) = 'S'", "MIN(label) > 'S'"),
    ONE.replace("MIN(label) = 'S'", "MIN(label) = MAX(label)"),
    ONE.replace("MIN(label) = 'S'", "MIN(UPPER(label)) = 'S'"),
    ONE.replace(" AND MIN(label)", " OR MIN(label)"),
    ONE.replace("MIN(label) = 'S'", "MIN(label) = 4"),  # a number beside a column that may hold text: '4' and '4.0' are two values
    ONE.replace("GROUP BY buyer HAVING", "WHERE").replace("COUNT(DISTINCT label) = 1 AND MIN(label) = 'S'", "label = 'S'"),
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_shapes_are_left_alone(sql):
    sqlglot.parse_one(sql, read="mysql")
    assert reading(sql) is None


def test_declared_types_decide_number_literals_and_floats():
    numbers = ONE.replace("label", "paid").replace("'S'", "4")
    assert reading(numbers) is None
    assert reading(numbers, {"sale": {"paid": "VARCHAR"}}) is None
    assert reading(numbers, {"sale": {"paid": "DOUBLE"}}) is None
    assert reading(ONE, {"sale": {"label": "DOUBLE"}}) is None
    assert reading(ONE, {"sale": {"label": "VARCHAR"}}) == COUNTS


SPELLINGS = [
    (ONE, None),
    (ONE.replace("MIN(label)", "MAX(label)"), None),
    ("SELECT buyer, COUNT(*) FROM sale WHERE paid > 1 GROUP BY buyer HAVING 'I' = MAX(label) AND COUNT(*) > 1 AND 1 = COUNT(DISTINCT label)", None),
    (ONE.replace("label", "paid").replace("'S'", "4"), {"sale": {"paid": "INT"}}),
]


@pytest.mark.parametrize("seed", range(4))
def test_the_reading_returns_the_rows_duckdb_returns(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    pairs = [(sql, reading(sql, types)) for sql, types in SPELLINGS]
    assert all(read is not None for _, read in pairs)
    kept = dropped = 0
    for _ in range(40):
        con = duckdb.connect()
        con.execute("CREATE TABLE sale (buyer INT, label VARCHAR, paid INT)")
        for _ in range(rng.randint(0, 12)):
            con.execute("INSERT INTO sale VALUES (?, ?, ?)", [rng.choice([None, 1, 2, 3, 4]), rng.choice([None, "S", "S", "S", "I", "T"]), rng.choice([None, 2, 4, 4, 4])])
        groups = con.execute("SELECT COUNT(DISTINCT buyer) FROM sale").fetchone()[0]
        for sql, read in pairs:
            expected = sorted(con.execute(sql).fetchall(), key=repr)
            assert expected == sorted(con.execute(read).fetchall(), key=repr), (sql, read)
            kept += len(expected)
            dropped += groups - len(expected) if groups >= len(expected) else 0
    assert kept > 30 and dropped > 30


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

SCHEMA = {"sale": ["buyer", "label", "paid"]}
TYPES = {"sale": {"label": "STRING"}}
LISTS = "SELECT DISTINCT buyer FROM sale WHERE label = 'S' AND buyer NOT IN (SELECT buyer FROM sale WHERE label <> 'S')"


def facts(not_null: bool = True):
    return {"sale": TableConstraints(not_null=frozenset({"buyer", "label"} if not_null else ()))}


def test_only_one_label_as_a_group_test_and_as_a_list():
    options = dict(schema=SCHEMA, compare_names=False, dialect="mysql", types=TYPES)
    assert prove_equivalent_algebraic(ONE, LISTS, constraints=facts(), **options).proven
    assert not prove_equivalent_algebraic(ONE, LISTS.replace("label <> 'S'", "label = 'I'"), constraints=facts(), **options).proven  # a T is another label too
    assert not prove_equivalent_algebraic(ONE.replace("= 1", "= 2"), LISTS, constraints=facts(), **options).proven


def test_it_is_a_narrow_retry_rule():
    constraints = facts()
    with shape_fallback.tracking() as fired:
        normalize(ONE, schema=SCHEMA, dialect="mysql", types=TYPES, not_null={t: c.not_null for t, c in constraints.items()})
    assert "single_value_groups" in fired
