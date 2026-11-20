"""An outer join from a derived table of referencing columns to the table they reference (``referencing_union_joins``)."""

import random

import duckdb
import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.referencing_union_joins import inner_referencing_joins
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"members": ["member_id", "name"], "rounds": ["round_id", "gold", "silver", "bronze", "host"], "halls": ["hall_id", "city"]}
TYPES = {
    "members": {"member_id": "INT", "name": "VARCHAR"},
    "rounds": {"round_id": "INT", "gold": "INT", "silver": "INT", "bronze": "INT", "host": "INT"},
    "halls": {"hall_id": "INT", "city": "VARCHAR"},
}
MEDALS = ("gold", "silver", "bronze")
NOT_NULL = {"members": frozenset({"member_id"}), "rounds": frozenset({"round_id", *MEDALS}), "halls": frozenset({"hall_id"})}
FOREIGN_KEYS = {"rounds": [((medal,), "members", ("member_id",)) for medal in MEDALS] + [(("host",), "halls", ("hall_id",))]}
UNPIVOT = "SELECT gold AS m FROM rounds UNION ALL SELECT silver AS m FROM rounds UNION ALL SELECT bronze AS m FROM rounds"


def reading(sql: str, dialect: str = "mysql", schema=SCHEMA, not_null=NOT_NULL, foreign_keys=FOREIGN_KEYS) -> str | None:
    written, changed = inner_referencing_joins(sqlglot.parse_one(sql, read=dialect), schema, not_null, foreign_keys)
    return written.sql(dialect=dialect) if changed else None


def test_a_left_join_to_the_referenced_table_is_an_inner_join():
    assert reading(f"SELECT t.m, p.name FROM ({UNPIVOT}) AS t LEFT JOIN members AS p ON t.m = p.member_id") == f"SELECT t.m, p.name FROM ({UNPIVOT}) AS t JOIN members AS p ON t.m = p.member_id"
    assert reading(f"SELECT t.m, p.name FROM ({UNPIVOT}) AS t LEFT OUTER JOIN members AS p ON p.member_id = t.m") == f"SELECT t.m, p.name FROM ({UNPIVOT}) AS t JOIN members AS p ON p.member_id = t.m"
    # one select, UNION, DISTINCT and WHERE only drop rows; other outputs do not matter
    assert reading("SELECT p.name FROM (SELECT DISTINCT gold AS m FROM rounds WHERE silver > 2) AS t LEFT JOIN members AS p ON t.m = p.member_id") == (
        "SELECT p.name FROM (SELECT DISTINCT gold AS m FROM rounds WHERE silver > 2) AS t JOIN members AS p ON t.m = p.member_id"
    )
    assert reading("SELECT p.name FROM (SELECT round_id, gold AS m FROM rounds UNION SELECT host, silver FROM rounds AS r) AS t LEFT JOIN members AS p ON t.m = p.member_id") == (
        "SELECT p.name FROM (SELECT round_id, gold AS m FROM rounds UNION SELECT host, silver FROM rounds AS r) AS t JOIN members AS p ON t.m = p.member_id"
    )
    # bare names when the select has only the two sources; USING
    assert reading(f"SELECT m, name FROM ({UNPIVOT}) AS t LEFT JOIN members ON m = member_id") == f"SELECT m, name FROM ({UNPIVOT}) AS t JOIN members ON m = member_id"
    listed = UNPIVOT.replace(" AS m", " AS member_id")
    assert reading(f"SELECT name FROM ({listed}) AS t LEFT JOIN members USING (member_id)") == f"SELECT name FROM ({listed}) AS t JOIN members USING (member_id)"
    # the same join the other way round
    assert reading(f"SELECT t.m, p.name FROM members AS p RIGHT JOIN ({UNPIVOT}) AS t ON t.m = p.member_id") == f"SELECT t.m, p.name FROM members AS p JOIN ({UNPIVOT}) AS t ON t.m = p.member_id"
    # after an inner join, and inside a derived table
    assert reading(f"SELECT 1 FROM halls AS h JOIN ({UNPIVOT}) AS t ON t.m = h.hall_id LEFT JOIN members AS p ON t.m = p.member_id") == (
        f"SELECT 1 FROM halls AS h JOIN ({UNPIVOT}) AS t ON t.m = h.hall_id JOIN members AS p ON t.m = p.member_id"
    )
    assert reading(f"SELECT z.name FROM (SELECT p.name FROM ({UNPIVOT}) AS t LEFT JOIN members AS p ON t.m = p.member_id) AS z") == (
        f"SELECT z.name FROM (SELECT p.name FROM ({UNPIVOT}) AS t JOIN members AS p ON t.m = p.member_id) AS z"
    )


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT t.m FROM ({UNPIVOT}) AS t JOIN members AS p ON t.m = p.member_id",  # already inner
        f"SELECT t.m FROM ({UNPIVOT}) AS t LEFT JOIN members AS p ON t.m = p.member_id AND p.name = 'a'",  # a second test may fail
        f"SELECT t.m FROM ({UNPIVOT}) AS t LEFT JOIN members AS p ON t.m >= p.member_id",
        f"SELECT t.m FROM ({UNPIVOT}) AS t LEFT JOIN members AS p ON t.m + 0 = p.member_id",
        f"SELECT t.m FROM ({UNPIVOT}) AS t LEFT JOIN members AS p ON t.m = p.name",  # not the referenced column
        f"SELECT t.m FROM ({UNPIVOT}) AS t LEFT JOIN halls AS p ON t.m = p.hall_id",  # not the referenced table
        f"SELECT t.m FROM ({UNPIVOT}) AS t FULL JOIN members AS p ON t.m = p.member_id",
        f"SELECT t.m FROM ({UNPIVOT}) AS t RIGHT JOIN members AS p ON t.m = p.member_id",  # keeps every member
        f"SELECT t.m FROM members AS p LEFT JOIN ({UNPIVOT}) AS t ON t.m = p.member_id",
        f"SELECT t.m FROM ({UNPIVOT}) AS t NATURAL LEFT JOIN members AS p",
        # the derived table is itself padded, or a later join pads everything before it
        f"SELECT 1 FROM halls AS h LEFT JOIN ({UNPIVOT}) AS t ON t.m = h.hall_id LEFT JOIN members AS p ON t.m = p.member_id",
        f"SELECT 1 FROM ({UNPIVOT}) AS t RIGHT JOIN halls AS h ON t.m = h.hall_id LEFT JOIN members AS p ON t.m = p.member_id",
        f"SELECT 1 FROM members AS p RIGHT JOIN ({UNPIVOT}) AS t ON t.m = p.member_id JOIN halls AS h ON h.hall_id = t.m",  # three sources
        # bare names with a third source
        f"SELECT 1 FROM ({UNPIVOT}) AS t JOIN halls AS h ON h.hall_id = t.m LEFT JOIN members ON m = member_id",
        # a column that does not reference the table, or may be NULL
        "SELECT 1 FROM (SELECT gold AS m FROM rounds UNION ALL SELECT round_id FROM rounds) AS t LEFT JOIN members AS p ON t.m = p.member_id",
        "SELECT 1 FROM (SELECT gold AS m FROM rounds UNION ALL SELECT host FROM rounds) AS t LEFT JOIN members AS p ON t.m = p.member_id",
        "SELECT 1 FROM (SELECT host AS h FROM rounds) AS t LEFT JOIN halls AS p ON t.h = p.hall_id",  # host is declared to reference, and may be NULL
        # an output that is not the column as it is
        "SELECT 1 FROM (SELECT gold + 0 AS m FROM rounds) AS t LEFT JOIN members AS p ON t.m = p.member_id",
        "SELECT 1 FROM (SELECT MAX(gold) AS m FROM rounds) AS t LEFT JOIN members AS p ON t.m = p.member_id",
        "SELECT 1 FROM (SELECT gold AS m FROM rounds GROUP BY gold) AS t LEFT JOIN members AS p ON t.m = p.member_id",
        "SELECT 1 FROM (SELECT gold AS m FROM rounds LIMIT 3) AS t LEFT JOIN members AS p ON t.m = p.member_id",
        "SELECT 1 FROM (SELECT * FROM rounds) AS t LEFT JOIN members AS p ON t.gold = p.member_id",
        "SELECT 1 FROM (SELECT r.gold AS m FROM rounds AS r JOIN halls AS h ON h.hall_id = r.host) AS t LEFT JOIN members AS p ON t.m = p.member_id",
        "SELECT 1 FROM (SELECT gold AS m FROM rounds EXCEPT SELECT silver FROM rounds) AS t LEFT JOIN members AS p ON t.m = p.member_id",
        "SELECT 1 FROM (SELECT gold AS m, silver AS m FROM rounds) AS t LEFT JOIN members AS p ON t.m = p.member_id",  # two outputs of one name
        "SELECT 1 FROM (SELECT gold FROM rounds) AS t (m) LEFT JOIN members AS p ON t.m = p.member_id",
        "SELECT 1 FROM (SELECT gold AS m FROM other.rounds) AS t LEFT JOIN members AS p ON t.m = p.member_id",
        "SELECT 1 FROM rounds AS t LEFT JOIN members AS p ON t.gold = p.member_id",  # a table: join_rewrites reads that
        f"WITH w AS (SELECT 1 AS one) SELECT t.m FROM ({UNPIVOT}) AS t LEFT JOIN members AS p ON t.m = p.member_id",  # a WITH still to inline
    ],
)
def test_other_joins_are_left_alone(sql):
    assert reading(sql) is None


def test_it_needs_the_declared_facts():
    sql = f"SELECT t.m, p.name FROM ({UNPIVOT}) AS t LEFT JOIN members AS p ON t.m = p.member_id"
    assert reading(sql) is not None
    assert reading(sql, foreign_keys=None) is None and reading(sql, not_null=None) is None and reading(sql, schema=None) is None
    assert reading(sql, not_null={**NOT_NULL, "rounds": frozenset({"round_id", "gold", "silver"})}) is None  # bronze may be NULL
    assert reading(sql, foreign_keys={"rounds": FOREIGN_KEYS["rounds"][:2]}) is None  # bronze references nothing
    assert reading(sql, foreign_keys={"rounds": [(("gold", "silver"), "members", ("member_id", "name"))]}) is None  # a two-column key says nothing of one
    # names are read without case
    assert reading(sql.upper().replace("AS M", "AS m").replace("T.M", "t.m"), schema={k.upper(): [c.upper() for c in v] for k, v in SCHEMA.items()}) is not None


def test_the_tree_given_is_not_changed():
    sql = f"SELECT t.m, p.name FROM ({UNPIVOT}) AS t LEFT JOIN members AS p ON t.m = p.member_id"
    tree = sqlglot.parse_one(sql, read="mysql")
    written, changed = inner_referencing_joins(tree, SCHEMA, NOT_NULL, FOREIGN_KEYS)
    assert changed and tree.sql(dialect="mysql") == sql and written is not tree


@pytest.mark.parametrize("seed", range(4))
def test_replay_on_generated_tables(seed):
    """With the declared facts true of the rows, the outer join and the inner join the module writes return the same rows."""

    rng = random.Random(seed)
    con = duckdb.connect()
    con.execute("CREATE TABLE members (member_id INT, name VARCHAR)")
    con.execute("CREATE TABLE rounds (round_id INT, gold INT, silver INT, bronze INT, host INT)")
    con.execute("CREATE TABLE halls (hall_id INT, city VARCHAR)")
    statements = [
        f"SELECT t.m, p.name FROM ({UNPIVOT}) AS t LEFT JOIN members AS p ON t.m = p.member_id",
        f"SELECT t.m, p.name, COUNT(*) FROM ({UNPIVOT}) AS t LEFT JOIN members AS p ON t.m = p.member_id GROUP BY t.m, p.name",
        f"SELECT p.name, COUNT(t.m), COUNT(p.member_id) FROM members AS p RIGHT JOIN ({UNPIVOT}) AS t ON t.m = p.member_id GROUP BY p.name",
        "SELECT p.name FROM (SELECT DISTINCT gold AS m FROM rounds WHERE silver > 2) AS t LEFT JOIN members AS p ON t.m = p.member_id",
        "SELECT t.round_id, p.name FROM (SELECT round_id, gold AS m FROM rounds UNION SELECT host, silver FROM rounds AS r) AS t LEFT JOIN members AS p ON t.m = p.member_id",
        f"SELECT h.city, t.m, p.name FROM halls AS h JOIN ({UNPIVOT}) AS t ON t.m = h.hall_id LEFT JOIN members AS p ON t.m = p.member_id",
    ]
    compared = padded = 0
    for _ in range(12):
        for table in ("members", "rounds", "halls"):
            con.execute(f"DELETE FROM {table}")
        ids = rng.sample(range(1, 9), rng.randint(1, 6))
        for member in ids:
            for _ in range(rng.choice([1, 1, 2])):  # the referenced column need not be unique
                con.execute("INSERT INTO members VALUES (?, ?)", (member, rng.choice(["a", "b", None])))
        for i in range(rng.choice([0, 1, 3, 6])):
            con.execute("INSERT INTO rounds VALUES (?, ?, ?, ?, ?)", (i, rng.choice(ids), rng.choice(ids), rng.choice(ids), rng.choice([None, 1, 2])))
        for i in range(rng.choice([0, 2, 5])):
            con.execute("INSERT INTO halls VALUES (?, ?)", (i + 1, rng.choice(["x", "y"])))
        for sql in statements:
            written, changed = inner_referencing_joins(sqlglot.parse_one(sql, read="duckdb"), SCHEMA, NOT_NULL, FOREIGN_KEYS)
            assert changed, sql
            assert sorted(con.execute(sql).fetchall(), key=repr) == sorted(con.execute(written.sql(dialect="duckdb")).fetchall(), key=repr), sql
            compared += 1
        # the control: a column that may be NULL, where the outer join pads
        control = "SELECT t.h, p.city FROM (SELECT host AS h FROM rounds) AS t {} JOIN halls AS p ON t.h = p.hall_id"
        padded += sorted(con.execute(control.format("LEFT")).fetchall(), key=repr) != sorted(con.execute(control.format("INNER")).fetchall(), key=repr)
    assert compared == 72 and padded > 0


def facts(not_null=MEDALS):
    return {
        "members": TableConstraints(not_null=frozenset({"member_id"}), keys=(("member_id",),)),
        "rounds": TableConstraints(not_null=frozenset({"round_id", *not_null}), keys=(("round_id",),), foreign_keys=tuple(FOREIGN_KEYS["rounds"])),
        "halls": TableConstraints(not_null=frozenset({"hall_id"}), keys=(("hall_id",),)),
    }


OPTIONS = dict(schema=SCHEMA, types=TYPES, dialect="mysql", compare_names=False, exact_arithmetic=True, timeout_ms=5000)
INNER = f"SELECT t.m AS member_id, p.name, COUNT(*) AS wins FROM ({UNPIVOT}) AS t JOIN members AS p ON t.m = p.member_id GROUP BY t.m, p.name"
OUTER = INNER.replace(" JOIN members", " LEFT JOIN members")


def test_the_outer_join_is_proved_equal_to_the_inner_join_under_the_declared_facts():
    result = prove_equivalent_algebraic(INNER, OUTER, constraints=facts(), **OPTIONS)
    assert result.proven, result.reason
    with shape_fallback.without("referencing_union_joins"):
        assert not prove_equivalent_algebraic(INNER, OUTER, constraints=facts(), **OPTIONS).proven
    plain = f"SELECT t.m, p.name FROM ({UNPIVOT}) AS t {{}}JOIN members AS p ON t.m = p.member_id"
    assert prove_equivalent_algebraic(plain.format(""), plain.format("LEFT "), constraints=facts(), **OPTIONS).proven
    assert prove_equivalent_algebraic(plain.format(""), f"SELECT t.m, p.name FROM members AS p RIGHT JOIN ({UNPIVOT}) AS t ON t.m = p.member_id", constraints=facts(), **OPTIONS).proven


def test_it_is_not_proved_without_the_facts():
    assert not prove_equivalent_algebraic(INNER, OUTER, constraints=facts(not_null=("gold", "silver")), **OPTIONS).proven  # bronze may be NULL
    no_keys = {**facts(), "rounds": TableConstraints(not_null=frozenset({"round_id", *MEDALS}), keys=(("round_id",),))}
    assert not prove_equivalent_algebraic(INNER, OUTER, constraints=no_keys, **OPTIONS).proven
    assert not prove_equivalent_algebraic(INNER, OUTER.replace("t.m = p.member_id", "t.m = p.member_id AND p.name = 'a'"), constraints=facts(), **OPTIONS).proven


def test_it_is_a_rule_of_the_thirteenth_batch():
    assert "referencing_union_joins" in shape_fallback.THIRTEENTH_BATCH
    options = dict(schema=SCHEMA, dialect="mysql", types=TYPES, not_null=NOT_NULL, foreign_keys=FOREIGN_KEYS)
    with shape_fallback.tracking() as fired:
        normalize(OUTER, **options)
    assert "referencing_union_joins" in fired
    with shape_fallback.without("referencing_union_joins"), shape_fallback.tracking() as fired:
        normalize(OUTER, **options)
    assert "referencing_union_joins" not in fired
    with shape_fallback.tracking() as fired:
        normalize(OUTER, schema=SCHEMA, dialect="mysql", types=TYPES, not_null=NOT_NULL)  # no foreign keys given
    assert "referencing_union_joins" not in fired
