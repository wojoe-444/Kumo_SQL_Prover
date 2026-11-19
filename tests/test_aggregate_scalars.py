"""A one-row select that combines several aggregates, one scalar subquery each (``aggregate_scalars``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.aggregate_scalars import split_one_row_select

SCHEMA = {"owner": ["id", "name"], "bill": ["id", "owner", "paid", "due"]}
JOINED = "owner AS o LEFT JOIN bill AS b ON b.owner = o.id"
SHARE = f"SELECT ROUND(COUNT(b.id) / COUNT(DISTINCT o.id), 2) AS share FROM {JOINED}"


def reading(sql: str, schema=SCHEMA) -> str | None:
    tree, changed = split_one_row_select(sqlglot.parse_one(sql, read="mysql"), schema)
    return tree.sql(dialect="mysql") if changed else None


def test_each_aggregate_becomes_a_scalar_subquery_over_the_same_rows():
    assert reading(SHARE) == f"SELECT ROUND((SELECT COUNT(b.id) FROM {JOINED}) / (SELECT COUNT(DISTINCT o.id) FROM {JOINED}), 2) AS share"
    # WHERE goes with every aggregate; a scalar subquery already written stays where it is
    assert reading("SELECT COUNT(*) / (SELECT COUNT(*) FROM bill) AS share FROM bill WHERE paid > 3") == (
        "SELECT (SELECT COUNT(*) FROM bill WHERE paid > 3) / (SELECT COUNT(*) FROM bill) AS share"
    )
    # one output that combines two aggregates is enough; the others are split with it
    assert reading("SELECT SUM(paid) - SUM(due) AS gap, COUNT(*) AS n FROM bill") == (
        "SELECT (SELECT SUM(paid) FROM bill) - (SELECT SUM(due) FROM bill) AS gap, (SELECT COUNT(*) FROM bill) AS n"
    )


def test_one_row_tables_are_read_where_their_outputs_are_used():
    counts = "(SELECT COUNT(*) AS n FROM bill WHERE paid > 3) AS a, (SELECT COUNT(*) AS n FROM bill) AS b"
    split = "SELECT (SELECT COUNT(*) FROM bill WHERE paid > 3) / (SELECT COUNT(*) FROM bill) AS share"
    assert reading(f"SELECT a.n / b.n AS share FROM {counts}") == split
    assert reading(f"SELECT a.n / b.n AS share FROM {counts}".replace(", (", " CROSS JOIN (")) == split
    assert reading("SELECT t.n / (SELECT COUNT(*) FROM owner) AS share FROM (SELECT COUNT(*) AS n FROM bill WHERE paid > 3) AS t") == (
        "SELECT (SELECT COUNT(*) FROM bill WHERE paid > 3) / (SELECT COUNT(*) FROM owner) AS share"
    )
    assert reading("SELECT t.n / (SELECT COUNT(o.id) FROM owner AS o WHERE o.name = 'a') AS share FROM (SELECT COUNT(*) AS n FROM bill) AS t") == (
        "SELECT (SELECT COUNT(*) FROM bill) / (SELECT COUNT(o.id) FROM owner AS o WHERE o.name = 'a') AS share"
    )
    assert reading("SELECT t.n / (SELECT COUNT(*) FROM owner AS o WHERE o.id = t.n) AS share FROM (SELECT COUNT(*) AS n FROM bill) AS t") is None  # reads t
    assert reading("SELECT t.n + t.m AS x, n FROM (SELECT COUNT(*) AS n, SUM(paid) * 2 AS m FROM bill) AS t") == (
        "SELECT (SELECT COUNT(*) FROM bill) + (SELECT SUM(paid) * 2 FROM bill) AS x, (SELECT COUNT(*) FROM bill) AS n"
    )


LEFT_ALONE = [
    "SELECT COUNT(*) AS n, SUM(paid) AS s FROM bill",  # one aggregate per output: compared block by block already
    "SELECT ROUND(AVG(paid), 2) FROM bill",
    "SELECT AVG(paid) * 100 + COUNT(*) FROM bill",  # an average carries digits past its scale: it stays where it is
    "SELECT SUM(paid / due) * 100 + COUNT(*) FROM bill",
    "SELECT t.q * 100 + t.n FROM (SELECT SUM(paid) / COUNT(*) AS q, COUNT(*) AS n FROM bill) AS t",  # MySQL rounds q where the table stores it
    "SELECT t.a * 3 + t.n FROM (SELECT AVG(paid) AS a, COUNT(*) AS n FROM bill) AS t",
    "SELECT t.n * 3 + t.m FROM (SELECT COUNT(*) AS n, (SELECT COUNT(*) FROM owner) AS m FROM bill) AS t",
    "SELECT t.n FROM (SELECT COUNT(*) AS n FROM bill) AS t",  # nothing combined
    "SELECT owner, SUM(paid) / COUNT(*) FROM bill GROUP BY owner",
    "SELECT SUM(paid) / COUNT(*) FROM bill HAVING COUNT(*) > 1",
    "SELECT DISTINCT SUM(paid) / COUNT(*) FROM bill",
    "SELECT SUM(paid) / COUNT(*) FROM bill ORDER BY 1",
    "SELECT SUM(paid) / COUNT(*) FROM bill LIMIT 1",
    "SELECT SUM(paid) / COUNT(*) FROM bill UNION ALL SELECT 1",
    "SELECT SUM(paid) / COUNT(*) OVER () FROM bill",
    "SELECT SUM(paid) / COUNT(*) + MAX(due) * RAND() FROM bill",  # volatile: read once, not once per aggregate
    "SELECT SUM(paid) / COUNT(*) FROM bill WHERE owner IN (SELECT id FROM owner)",  # a subquery in WHERE
    "SELECT SUM(paid) / (SELECT COUNT(*) FROM owner WHERE owner.id = bill.owner) FROM bill",  # not closed
    "SELECT SUM(paid) / COUNT(*) + paid FROM bill",  # a column outside the aggregates
    "SELECT SUM(paid) / COUNT(*) FROM bill, unknown_table",
    "SELECT SUM(x.paid) / COUNT(*) FROM bill AS x JOIN bill AS x ON TRUE",
    "SELECT a.n / b.n FROM (SELECT COUNT(*) AS n FROM bill) AS a JOIN (SELECT COUNT(*) AS n FROM bill) AS b ON a.n = b.n",  # an ON can drop the row
    "SELECT a.n / b.n FROM (SELECT COUNT(*) AS n FROM bill) AS a, (SELECT COUNT(*) AS n FROM bill GROUP BY owner) AS b",  # not one row
    "SELECT a.n / b.n FROM (SELECT COUNT(*) AS n FROM bill) AS a, (SELECT paid AS n FROM bill) AS b",
    "SELECT a.n / a.n FROM (SELECT COUNT(*) AS n FROM bill) AS a WHERE a.n > 0",  # WHERE can drop the row
    "SELECT a.n / b.id FROM (SELECT COUNT(*) AS n FROM bill) AS a, bill AS b",
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_shapes_are_left_alone(sql):
    sqlglot.parse_one(sql, read="mysql")
    assert reading(sql) is None


def test_it_reads_the_query_only_and_needs_the_schema():
    assert reading(SHARE, None) is None
    inner = sqlglot.parse_one(f"SELECT x.share FROM ({SHARE}) AS x", read="mysql")
    assert split_one_row_select(inner, SCHEMA)[1] is False


SPELLINGS = [
    f"SELECT COUNT(b.id) * 10 + COUNT(DISTINCT o.id) AS x, MAX(b.paid) AS top FROM {JOINED}",
    "SELECT COUNT(*) - (SELECT COUNT(*) FROM bill) AS x FROM bill WHERE paid > 3",
    "SELECT COALESCE(SUM(paid), -1) - COALESCE(MIN(due), -2) AS gap, COUNT(owner) AS n FROM bill WHERE due >= 2",
    "SELECT a.n * 100 + b.n AS x FROM (SELECT COUNT(*) AS n FROM bill WHERE paid > 3) AS a, (SELECT COUNT(owner) AS n FROM bill) AS b",
    "SELECT COALESCE(t.m, 0) - t.n AS x, n FROM (SELECT COUNT(*) AS n, SUM(paid) * 2 AS m FROM bill) AS t",
]


@pytest.mark.parametrize("seed", range(4))
def test_the_reading_returns_the_row_duckdb_returns(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    pairs = [(sql, reading(sql)) for sql in SPELLINGS]
    assert all(read is not None for _, read in pairs)
    empty = filled = 0
    for _ in range(40):
        con = duckdb.connect()
        con.execute("CREATE TABLE owner (id INT, name VARCHAR)")
        con.execute("CREATE TABLE bill (id INT, owner INT, paid INT, due INT)")
        for number in range(rng.choice([0, 1, 3, 4])):
            con.execute("INSERT INTO owner VALUES (?, ?)", [number, rng.choice(["a", "b", None])])
        bills = rng.choice([0, 0, 1, 4, 8])
        for number in range(bills):
            con.execute("INSERT INTO bill VALUES (?, ?, ?, ?)", [number, rng.choice([None, 0, 1, 2, 7]), rng.choice([None, 1, 4, 5]), rng.choice([None, 1, 2, 3])])
        empty += bills == 0
        filled += bills > 0
        for sql, read in pairs:
            assert con.execute(sql).fetchall() == con.execute(read).fetchall(), (sql, read)
    assert empty > 5 and filled > 15


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

TYPES = {"owner": {"id": "INT", "name": "VARCHAR"}, "bill": {"id": "INT", "owner": "INT", "paid": "INT", "due": "INT"}}
OPTIONS = dict(schema=SCHEMA, types=TYPES, compare_names=False, dialect="mysql")
FACTS = {"bill": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)), "owner": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),))}


def test_a_ratio_of_counts_written_together_and_apart():
    together = "SELECT ROUND(COUNT(CASE WHEN paid > 3 THEN 1 END) / COUNT(*), 2) AS share FROM bill"
    for other in (
        "SELECT ROUND((SELECT COUNT(*) FROM bill WHERE paid > 3) / (SELECT COUNT(id) FROM bill), 2) AS share",
        "SELECT ROUND(COUNT(*) / (SELECT COUNT(*) FROM bill), 2) AS share FROM bill WHERE paid > 3",
        "SELECT ROUND(a.n / b.n, 2) AS share FROM (SELECT COUNT(id) AS n FROM bill WHERE paid > 3) AS a, (SELECT COUNT(*) AS n FROM bill) AS b",
        "SELECT ROUND(t.n / (SELECT COUNT(*) FROM bill), 2) AS share FROM (SELECT COUNT(*) AS n FROM bill WHERE paid > 3) AS t",
    ):
        assert prove_equivalent_algebraic(together, other, constraints=FACTS, **OPTIONS).proven, other
    for other in (
        "SELECT ROUND((SELECT COUNT(*) FROM bill WHERE paid > 4) / (SELECT COUNT(id) FROM bill), 2) AS share",
        "SELECT ROUND((SELECT COUNT(*) FROM bill WHERE paid > 3) / (SELECT COUNT(owner) FROM bill), 2) AS share",  # owner may be NULL
        "SELECT ROUND((SELECT COUNT(*) FROM bill) / (SELECT COUNT(*) FROM bill WHERE paid > 3), 2) AS share",
    ):
        assert not prove_equivalent_algebraic(together, other, constraints=FACTS, **OPTIONS).proven, other


def test_it_is_a_narrow_retry_rule_of_the_tenth_batch():
    assert "aggregate_scalars" in shape_fallback.TENTH_BATCH
    with shape_fallback.tracking() as fired:
        normalize(SHARE, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "aggregate_scalars" in fired
    with shape_fallback.without("aggregate_scalars"), shape_fallback.tracking() as fired:
        normalize(SHARE, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "aggregate_scalars" not in fired
