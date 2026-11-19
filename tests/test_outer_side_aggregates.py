"""Aggregates of the outer side of a left join read the matched rows only (``outer_side_aggregates``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.outer_side_aggregates import join_matched_rows, join_matched_rows_everywhere

LEFT = "SELECT COUNT(b.id) AS n FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id AND b.paid > 3"
INNER = LEFT.replace("LEFT JOIN", "JOIN")


def reading(sql: str) -> str | None:
    result = join_matched_rows(sqlglot.parse_one(sql, read="mysql"))
    return result.sql(dialect="mysql") if result is not None else None


def test_the_left_join_of_the_aggregated_source_is_an_inner_join():
    assert reading(LEFT) == INNER
    assert reading(LEFT.replace("LEFT JOIN", "LEFT OUTER JOIN")) == INNER
    assert reading(INNER) is None
    # every skipping aggregate, DISTINCT or not, and WHERE on either side
    many = "SELECT COUNT(DISTINCT b.id) / SUM(b.paid) + MIN(b.due) - MAX(b.due) * AVG(b.paid) AS x FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id WHERE o.name = 'a' OR b.due IS NULL"
    assert reading(many) == many.replace("LEFT JOIN", "JOIN")
    # a join before or after it keeps its kind
    chain = "SELECT SUM(b.paid) FROM owner AS o LEFT JOIN owner AS p ON p.id = o.id LEFT JOIN bill AS b ON b.owner = p.id JOIN owner AS q ON q.id = o.id"
    assert reading(chain) == chain.replace("LEFT JOIN bill", "JOIN bill")
    # a derived table on the outer side
    derived = "SELECT COUNT(t.owner) FROM owner AS o LEFT JOIN (SELECT owner FROM bill WHERE paid > 3) AS t ON t.owner = o.id"
    assert reading(derived) == derived.replace("LEFT JOIN", "JOIN")
    # a right join of two sources, the first one outer
    assert reading("SELECT COUNT(b.id) FROM bill AS b RIGHT JOIN owner AS o ON b.owner = o.id") == "SELECT COUNT(b.id) FROM bill AS b JOIN owner AS o ON b.owner = o.id"


def test_everywhere_reads_nested_selects():
    nested = f"SELECT (SELECT COUNT(b.id) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id) / (SELECT COUNT(*) FROM owner) AS share, t.n FROM ({LEFT}) AS t"
    tree, changed = join_matched_rows_everywhere(sqlglot.parse_one(nested, read="mysql"))
    assert changed and tree.sql(dialect="mysql") == nested.replace("LEFT JOIN", "JOIN")
    tree, changed = join_matched_rows_everywhere(sqlglot.parse_one(INNER, read="mysql"))
    assert not changed and tree.sql(dialect="mysql") == INNER


LEFT_ALONE = [
    "SELECT COUNT(*) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id",  # counts the added rows
    "SELECT COUNT(o.id) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id",  # the inner side
    "SELECT COUNT(b.id), COUNT(o.id) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id",
    "SELECT COUNT(b.id), COUNT(*) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id",
    "SELECT COUNT(id) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id",  # no alias on the column
    "SELECT COUNT(COALESCE(b.id, 0)) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id",  # not NULL on an added row
    "SELECT COUNT(b.id IS NULL) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id",
    "SELECT SUM(b.paid + 1) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id",  # NULL on an added row too, but not read
    "SELECT GROUP_CONCAT(b.id) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id",
    "SELECT COUNT(b.id) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id GROUP BY o.id",  # a group with no match would vanish
    "SELECT COUNT(b.id) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id HAVING COUNT(b.id) > 0",
    "SELECT DISTINCT COUNT(b.id) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id",
    "SELECT COUNT(b.id) OVER () FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id",
    "SELECT COUNT(b.id), SUM(b.paid) OVER () FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id",
    "SELECT COUNT(b.id), o.name FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id",  # a column outside the aggregates
    "SELECT COUNT(b.id) FROM owner AS o LEFT JOIN bill AS b USING (id)",
    "SELECT COUNT(b.id) FROM owner AS o NATURAL LEFT JOIN bill AS b",
    "SELECT COUNT(b.id) FROM owner AS o FULL JOIN bill AS b ON b.owner = o.id",
    "SELECT COUNT(b.id) FROM owner AS o RIGHT JOIN bill AS b ON b.owner = o.id",  # b is the kept side
    "SELECT COUNT(o.id) FROM owner AS o RIGHT JOIN bill AS b ON b.owner = o.id JOIN owner AS q ON q.id = b.owner",  # a right join among several
    "SELECT COUNT(b.id) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id LIMIT 1",
    "SELECT COUNT(b.id) FROM bill AS b",
    "SELECT COUNT(b.id) FROM owner AS b LEFT JOIN bill AS b ON TRUE",
    "SELECT b.id FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id",
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_shapes_are_left_alone(sql):
    sqlglot.parse_one(sql, read="mysql")
    assert reading(sql) is None


REPLAYED = [
    LEFT,
    "SELECT COUNT(DISTINCT b.owner), SUM(b.paid), MIN(b.due), MAX(b.due), AVG(b.paid) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id WHERE o.name = 'a' OR b.due IS NULL",
    "SELECT SUM(b.paid), COUNT(b.due) FROM owner AS o LEFT JOIN owner AS p ON p.id = o.id LEFT JOIN bill AS b ON b.owner = p.id JOIN owner AS q ON q.id = o.id",
    "SELECT COUNT(t.owner), MAX(t.owner) FROM owner AS o LEFT JOIN (SELECT owner FROM bill WHERE paid > 3) AS t ON t.owner = o.id",
    "SELECT COUNT(b.id), SUM(b.paid) FROM bill AS b RIGHT JOIN owner AS o ON b.owner = o.id WHERE o.name IS NULL",
    "SELECT COUNT(b.paid) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id LEFT JOIN owner AS q ON q.id = b.due",
]


@pytest.mark.parametrize("seed", range(4))
def test_the_reading_returns_the_row_duckdb_returns(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    pairs = [(sql, reading(sql)) for sql in REPLAYED]
    assert all(read is not None for _, read in pairs)
    unmatched = 0
    for _ in range(40):
        con = duckdb.connect()
        con.execute("CREATE TABLE owner (id INT, name VARCHAR)")
        con.execute("CREATE TABLE bill (id INT, owner INT, paid INT, due INT)")
        owners = rng.choice([0, 1, 3, 5])
        for number in range(owners):
            con.execute("INSERT INTO owner VALUES (?, ?)", [rng.choice([None, 0, 1, 2, 3]), rng.choice(["a", "b", None])])
        for number in range(rng.choice([0, 0, 2, 6])):
            con.execute("INSERT INTO bill VALUES (?, ?, ?, ?)", [number, rng.choice([None, 0, 1, 2]), rng.choice([None, 1, 4, 5]), rng.choice([None, 1, 2, 3])])
        unmatched += con.execute("SELECT COUNT(*) FROM owner AS o WHERE NOT EXISTS (SELECT 1 FROM bill AS b WHERE b.owner = o.id)").fetchone()[0] > 0
        for sql, read in pairs:
            assert con.execute(sql).fetchall() == con.execute(read).fetchall(), (sql, read)
    assert unmatched > 15


def test_a_count_of_every_row_differs_between_the_joins():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE owner (id INT, name VARCHAR)")
    con.execute("CREATE TABLE bill (id INT, owner INT, paid INT, due INT)")
    con.execute("INSERT INTO owner VALUES (1, 'a'), (2, 'b')")
    con.execute("INSERT INTO bill VALUES (1, 1, 5, 1)")
    star = "SELECT COUNT(*) FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id"
    assert con.execute(star).fetchall() != con.execute(star.replace("LEFT JOIN", "JOIN")).fetchall()
    assert con.execute(LEFT).fetchall() == con.execute(INNER).fetchall() == [(1,)]


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402

SCHEMA = {"owner": ["id", "name"], "bill": ["id", "owner", "paid", "due"]}
TYPES = {"owner": {"id": "INT", "name": "VARCHAR"}, "bill": {"id": "INT", "owner": "INT", "paid": "INT", "due": "INT"}}
OPTIONS = dict(schema=SCHEMA, types=TYPES, compare_names=False, dialect="mysql")


def test_the_two_joins_prove_equal_under_such_aggregates_only():
    assert prove_equivalent_algebraic(LEFT, INNER, **OPTIONS).proven
    two = "SELECT SUM(b.paid) AS s, COUNT(DISTINCT b.owner) AS n FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id WHERE o.name = 'a'"
    assert prove_equivalent_algebraic(two, two.replace("LEFT JOIN", "JOIN"), **OPTIONS).proven
    for other in (
        "SELECT COUNT(*) AS n FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id AND b.paid > 3",
        "SELECT COUNT(o.id) AS n FROM owner AS o LEFT JOIN bill AS b ON b.owner = o.id AND b.paid > 3",
    ):
        assert not prove_equivalent_algebraic(other, other.replace("LEFT JOIN", "JOIN"), **OPTIONS).proven, other


def test_it_is_a_narrow_retry_rule_of_the_tenth_batch():
    assert "outer_side_aggregates" in shape_fallback.TENTH_BATCH
    with shape_fallback.tracking() as fired:
        normalize(LEFT, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "outer_side_aggregates" in fired
    with shape_fallback.without("outer_side_aggregates"), shape_fallback.tracking() as fired:
        normalize(LEFT, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "outer_side_aggregates" not in fired
