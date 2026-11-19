"""``COUNT(DISTINCT x)`` over rows that cannot repeat ``x`` is ``COUNT(x)`` (``unique_count_arguments``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.unique_count_arguments import drop_needless_count_distinct, drop_needless_count_distinct_everywhere

KEYS = {"visit": [("owner", "day")], "owner": [("id",)]}
TYPES = {"visit": {"owner": "INT", "day": "DATE", "room": "INT", "note": "VARCHAR"}, "owner": {"id": "INT", "name": "VARCHAR", "code": "VARCHAR"}}
FIRST = "(SELECT owner, MIN(day) AS first FROM visit GROUP BY owner) AS f"
NEXT = f"SELECT COUNT(DISTINCT f.owner) AS n FROM {FIRST} JOIN visit AS v ON v.owner = f.owner AND v.day = f.first + 1"


def reading(sql: str, keys=KEYS, types=TYPES) -> str | None:
    result = drop_needless_count_distinct(sqlglot.parse_one(sql, read="mysql"), keys, types)
    return result.sql(dialect="mysql") if result is not None else None


def test_the_distinct_goes_when_no_row_repeats_the_value():
    assert reading(NEXT) == NEXT.replace("DISTINCT ", "")
    # the joined table's column, equal to the grouped one on every row; either order of the sources and of the tests
    assert reading(NEXT.replace("f.owner) AS n", "v.owner) AS n")) == NEXT.replace("DISTINCT f.owner", "v.owner")
    swapped = f"SELECT COUNT(DISTINCT v.owner), COUNT(DISTINCT f.owner) FROM visit AS v JOIN {FIRST} ON f.first + 1 = v.day AND f.owner = v.owner WHERE v.room > 2"
    assert reading(swapped) == swapped.replace("DISTINCT ", "")
    # a base table with a one-column key on the unique side, and a whole-number key column pinned by whole numbers
    keyed = "SELECT COUNT(DISTINCT o.id) / COUNT(*) FROM owner AS o INNER JOIN visit AS v ON v.owner = o.id * 2 AND v.day = v.day AND v.day = o.id"
    assert reading(keyed) == keyed.replace("DISTINCT ", "")
    # only the counts it can answer lose the DISTINCT
    mixed = NEXT.replace("COUNT(DISTINCT f.owner) AS n", "COUNT(DISTINCT f.owner), COUNT(DISTINCT v.room), COUNT(DISTINCT f.first)")
    assert reading(mixed) == mixed.replace("DISTINCT f.owner", "f.owner")


def test_everywhere_reads_nested_selects():
    nested = f"SELECT ({NEXT.replace(' AS n', '')}) / (SELECT COUNT(DISTINCT owner) FROM visit) AS share"
    tree, changed = drop_needless_count_distinct_everywhere(sqlglot.parse_one(nested, read="mysql"), KEYS, TYPES)
    assert changed and tree.sql(dialect="mysql") == nested.replace("DISTINCT f.owner", "f.owner")
    assert drop_needless_count_distinct_everywhere(sqlglot.parse_one(nested, read="mysql"), None, TYPES)[1] is False


def test_each_fact_is_needed():
    assert reading(NEXT, keys={"owner": [("id",)]}) is None  # no key on the joined table: a row of f can meet two rows
    assert reading(NEXT, keys={"visit": [("owner", "day", "room")]}) is None  # a key column the join does not pin
    assert reading(NEXT.replace(" AND v.day = f.first + 1", "")) is None
    assert reading(NEXT.replace("v.day = f.first + 1", "v.day >= f.first + 1")) is None
    assert reading(NEXT.replace("v.day = f.first + 1", "v.day = v.day")) is None  # reads its own row
    assert reading(NEXT.replace(" AND v.day", " OR v.day")) is None
    assert reading(NEXT.replace("GROUP BY owner", "GROUP BY owner, room")) is None  # f repeats an owner
    assert reading(NEXT.replace("SELECT owner, MIN", "SELECT room AS owner, MIN")) is None  # not the grouped column
    assert reading(NEXT.replace("f.owner) AS n", "f.first) AS n")) is None  # two owners can share a first day
    assert reading(NEXT.replace("f.owner) AS n", "v.room) AS n")) is None


def test_a_key_column_must_be_told_apart_the_way_the_key_does():
    by_name = "SELECT COUNT(DISTINCT o.id) FROM owner AS o JOIN visit AS v ON v.owner = o.id AND v.day = o.id"
    assert reading(by_name) is not None
    text = {**TYPES, "visit": {**TYPES["visit"], "owner": "VARCHAR"}}
    assert reading(by_name, types=text) is None  # '5' and '5.0' both equal 5: two rows of visit for one owner
    assert reading(by_name, types={}) is None
    assert reading(by_name.replace("v.owner = o.id", "v.owner = o.name")) is None  # a whole-number column beside text
    assert reading(by_name.replace("v.owner = o.id", "v.owner = o.id / 2")) is None
    assert reading(by_name.replace("v.owner = o.id", "v.owner = o.id + 0.5")) is None
    assert reading(by_name.replace("v.owner = o.id", "v.owner = o.id + RAND()")) is None
    assert reading(by_name.replace("v.owner = o.id", "v.owner = (SELECT MAX(id) FROM owner)")) is None
    # the same base column on both sides needs no type; a date column takes any arithmetic of the other side's columns
    assert reading(NEXT, types={"visit": {"day": "DATE"}}) is not None
    assert reading(NEXT, types={}) is None
    floating = {**TYPES, "visit": {**TYPES["visit"], "owner": "DOUBLE"}}
    assert reading(NEXT, types=floating) is None


LEFT_ALONE = [
    NEXT.replace("JOIN visit", "LEFT JOIN visit"),
    NEXT.replace("JOIN visit", "CROSS JOIN visit"),
    NEXT.replace("AS n FROM", "AS n, f.owner FROM") + " GROUP BY f.owner",
    NEXT + " HAVING COUNT(*) > 1",
    NEXT + " LIMIT 1",
    NEXT.replace("SELECT COUNT", "SELECT DISTINCT COUNT"),
    NEXT.replace("COUNT(DISTINCT f.owner)", "COUNT(DISTINCT f.owner) OVER ()"),
    NEXT.replace("COUNT(DISTINCT f.owner)", "COUNT(DISTINCT f.owner, f.first)"),
    NEXT.replace("COUNT(DISTINCT f.owner)", "COUNT(DISTINCT f.owner + 1)"),
    NEXT.replace("COUNT(DISTINCT f.owner)", "COUNT(DISTINCT owner)"),
    NEXT.replace("COUNT(DISTINCT f.owner)", "SUM(DISTINCT f.owner)"),
    NEXT + " JOIN owner AS o ON o.id = f.owner",  # three sources
    f"SELECT COUNT(DISTINCT f.owner) FROM {FIRST}, visit AS v WHERE v.owner = f.owner AND v.day = f.first + 1",  # the tests are not in an ON
    "SELECT COUNT(DISTINCT v.owner) FROM visit AS v JOIN visit AS w ON w.owner = v.owner AND w.day = v.day",  # no unique column on either side
    f"SELECT COUNT(DISTINCT f.owner) FROM {FIRST} JOIN (SELECT * FROM visit) AS v ON v.owner = f.owner AND v.day = f.first + 1",
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_shapes_are_left_alone(sql):
    sqlglot.parse_one(sql, read="mysql")
    assert reading(sql) is None


REPLAYED = [
    "SELECT COUNT(DISTINCT f.owner), COUNT(DISTINCT v.owner), COUNT(*) FROM (SELECT owner, MIN(day) AS first FROM visit GROUP BY owner) AS f JOIN visit AS v ON v.owner = f.owner AND v.day = f.first + 1",
    "SELECT COUNT(DISTINCT v.owner), COUNT(DISTINCT f.owner) FROM visit AS v JOIN (SELECT owner, MAX(day) AS first FROM visit WHERE room > 1 GROUP BY owner) AS f ON f.first - 2 = v.day AND f.owner = v.owner WHERE v.room > 2",
    "SELECT COUNT(DISTINCT o.id), COUNT(o.id) FROM owner AS o INNER JOIN visit AS v ON v.owner = o.id * 2 AND v.day = o.id",
]


@pytest.mark.parametrize("seed", range(4))
def test_the_reading_returns_the_row_duckdb_returns(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    types = {**TYPES, "visit": {**TYPES["visit"], "day": "INT"}}  # whole-number days: DuckDB adds no number to a date
    pairs = [(sql, reading(sql, types=types)) for sql in REPLAYED]
    assert all(read is not None for _, read in pairs)
    matched = 0
    for _ in range(40):
        con = duckdb.connect()
        con.execute("CREATE TABLE owner (id INT UNIQUE, name VARCHAR, code VARCHAR)")
        con.execute("CREATE TABLE visit (owner INT, day INT, room INT, note VARCHAR, UNIQUE (owner, day))")
        for number in rng.sample([None, None, 0, 1, 2, 3], rng.choice([0, 2, 4])):
            con.execute("INSERT INTO owner VALUES (?, 'a', 'b')", [number])
        seen = set()
        for _ in range(rng.choice([0, 6, 12, 20])):
            row = (rng.choice([None, 0, 1, 2, 4]), rng.choice([None, 0, 1, 2, 3]))
            if None not in row and row in seen:
                continue
            seen.add(row)
            con.execute("INSERT INTO visit VALUES (?, ?, ?, 'n')", [*row, rng.choice([None, 1, 2, 3])])
        matched += con.execute(REPLAYED[0]).fetchone()[2] > 1
        for sql, read in pairs:
            assert con.execute(sql).fetchall() == con.execute(read).fetchall(), (sql, read)
    assert matched > 8


def test_without_the_key_the_two_counts_differ():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE visit (owner INT, day INT, room INT, note VARCHAR)")
    con.execute("INSERT INTO visit VALUES (1, 1, 1, 'n'), (1, 2, 1, 'n'), (1, 2, 2, 'n')")  # the second day twice
    sql = REPLAYED[0]
    assert con.execute(sql).fetchone() == (1, 1, 2)
    assert con.execute(sql.replace("DISTINCT ", "")).fetchone() == (2, 2, 2)


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

SCHEMA = {"visit": ["owner", "day", "room", "note"], "owner": ["id", "name", "code"]}
KEYED = {"visit": TableConstraints(not_null=frozenset({"owner", "day"}), keys=(("owner", "day"),))}
OPTIONS = dict(schema=SCHEMA, types=TYPES, compare_names=False, dialect="mysql")


def test_the_two_counts_prove_equal_with_the_key_only():
    plain = NEXT.replace("DISTINCT ", "")
    assert prove_equivalent_algebraic(NEXT, plain, constraints=KEYED, **OPTIONS).proven
    assert not prove_equivalent_algebraic(NEXT, plain, **OPTIONS).proven
    loose = NEXT.replace("v.day = f.first + 1", "v.day > f.first")
    assert not prove_equivalent_algebraic(loose, loose.replace("DISTINCT ", ""), constraints=KEYED, **OPTIONS).proven


def test_it_is_a_narrow_retry_rule_of_the_tenth_batch():
    assert "unique_count_arguments" in shape_fallback.TENTH_BATCH
    facts = dict(schema=SCHEMA, dialect="mysql", types=TYPES, keys=KEYS, not_null={"visit": {"owner", "day"}})
    with shape_fallback.tracking() as fired:
        normalize(NEXT, **facts)
    assert "unique_count_arguments" in fired
    with shape_fallback.without("unique_count_arguments"), shape_fallback.tracking() as fired:
        normalize(NEXT, **facts)
    assert "unique_count_arguments" not in fired
