"""A column list on a WITH name, written as aliases on the outputs it renames (``listed_cte_names``)."""

import duckdb
import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.listed_cte_names import name_listed_cte_outputs
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"members": ["member_id", "name"], "rounds": ["round_id", "gold", "silver", "bronze"]}
TYPES = {"members": {"member_id": "INT", "name": "VARCHAR"}, "rounds": {"round_id": "INT", "gold": "INT", "silver": "INT", "bronze": "INT"}}


def reading(sql: str, dialect: str = "mysql") -> str | None:
    written, changed = name_listed_cte_outputs(sqlglot.parse_one(sql, read=dialect))
    return written.sql(dialect=dialect) if changed else None


def test_the_listed_names_become_aliases():
    assert reading("WITH t (id) AS (SELECT gold FROM rounds) SELECT t.id FROM t") == "WITH t AS (SELECT gold AS id FROM rounds) SELECT t.id FROM t"
    assert reading("WITH t (id, place) AS (SELECT gold, 1 FROM rounds) SELECT id, place FROM t") == "WITH t AS (SELECT gold AS id, 1 AS place FROM rounds) SELECT id, place FROM t"
    # a name the output already has is kept; an alias written in the query is replaced
    assert reading("WITH t (gold, n) AS (SELECT gold, COUNT(*) AS wins FROM rounds GROUP BY gold) SELECT * FROM t") == (
        "WITH t AS (SELECT gold AS gold, COUNT(*) AS n FROM rounds GROUP BY gold) SELECT * FROM t"
    )
    assert reading("WITH t (k, c) AS (SELECT gold, COUNT(*) FROM rounds GROUP BY gold) SELECT k FROM t WHERE c > 1") == (
        "WITH t AS (SELECT gold AS k, COUNT(*) AS c FROM rounds GROUP BY gold) SELECT k FROM t WHERE c > 1"
    )
    # listed names are read without case unless they are quoted
    assert reading("WITH t (ID) AS (SELECT gold FROM rounds) SELECT t.id FROM t") == "WITH t AS (SELECT gold AS id FROM rounds) SELECT t.id FROM t"
    assert reading('WITH t ("Id") AS (SELECT gold FROM rounds) SELECT t."Id" FROM t', dialect="postgres") == 'WITH t AS (SELECT gold AS "Id" FROM rounds) SELECT t."Id" FROM t'


def test_a_set_operation_takes_the_names_on_its_first_select():
    assert reading("WITH t (id) AS (SELECT gold FROM rounds UNION ALL SELECT silver FROM rounds UNION ALL SELECT bronze FROM rounds) SELECT t.id FROM t") == (
        "WITH t AS (SELECT gold AS id FROM rounds UNION ALL SELECT silver FROM rounds UNION ALL SELECT bronze FROM rounds) SELECT t.id FROM t"
    )
    assert reading("WITH t (id) AS ((SELECT gold FROM rounds) UNION (SELECT silver FROM rounds)) SELECT id FROM t") == (
        "WITH t AS ((SELECT gold AS id FROM rounds) UNION (SELECT silver FROM rounds)) SELECT id FROM t"
    )
    assert reading("WITH t (id) AS (SELECT gold FROM rounds EXCEPT SELECT silver FROM rounds) SELECT id FROM t", dialect="postgres") == (
        "WITH t AS (SELECT gold AS id FROM rounds EXCEPT SELECT silver FROM rounds) SELECT id FROM t"
    )


def test_each_listed_entry_is_read_by_itself():
    assert reading("WITH a (x) AS (SELECT gold FROM rounds), b AS (SELECT silver FROM rounds), c (y) AS (SELECT x FROM a) SELECT y FROM c") == (
        "WITH a AS (SELECT gold AS x FROM rounds), b AS (SELECT silver FROM rounds), c AS (SELECT x AS y FROM a) SELECT y FROM c"
    )
    # one that is not read leaves the other as it would be alone
    assert reading("WITH a (x) AS (SELECT gold FROM rounds), c (y, z) AS (SELECT x FROM a) SELECT y FROM c") == "WITH a AS (SELECT gold AS x FROM rounds), c(y, z) AS (SELECT x FROM a) SELECT y FROM c"
    # a WITH inside a derived table
    assert reading("SELECT d.id FROM (WITH t (id) AS (SELECT gold FROM rounds) SELECT id FROM t) AS d") == "SELECT d.id FROM (WITH t AS (SELECT gold AS id FROM rounds) SELECT id FROM t) AS d"


@pytest.mark.parametrize(
    "sql",
    [
        "WITH t AS (SELECT gold FROM rounds) SELECT gold FROM t",  # no list
        "WITH t (id, place) AS (SELECT gold FROM rounds) SELECT id FROM t",  # the list must name every output
        "WITH t (id) AS (SELECT gold, silver FROM rounds) SELECT id FROM t",
        "WITH t (id, ID) AS (SELECT gold, silver FROM rounds) SELECT 1 FROM t",  # one name twice
        "WITH t (a, b, c, d) AS (SELECT * FROM rounds) SELECT a FROM t",  # a star
        "WITH t (a) AS (SELECT rounds.* FROM rounds) SELECT a FROM t",
        "WITH t (id) AS (SELECT gold FROM rounds ORDER BY gold LIMIT 3) SELECT id FROM t",  # ORDER BY names outputs
        "WITH t (id) AS (SELECT gold FROM rounds ORDER BY 1) SELECT id FROM t",
        "WITH t (id) AS (SELECT gold FROM rounds UNION ALL SELECT silver FROM rounds ORDER BY gold) SELECT id FROM t",
        "WITH t (id) AS (SELECT gold FROM rounds UNION ALL SELECT silver FROM rounds LIMIT 2) SELECT id FROM t",
        "WITH t (id) AS (SELECT 1 FROM rounds INTERSECT SELECT 2 FROM rounds ORDER BY 1) SELECT id FROM t",
        "WITH RECURSIVE t (n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t WHERE n < 3) SELECT n FROM t",
        # a name the alias gives, read without a table inside the select: a column for one engine, the output for another
        "WITH t (silver) AS (SELECT gold FROM rounds WHERE silver > 1) SELECT silver FROM t",
        "WITH t (silver, n) AS (SELECT gold, COUNT(*) FROM rounds GROUP BY silver) SELECT n FROM t",
        "WITH t (k, n) AS (SELECT gold, COUNT(*) FROM rounds GROUP BY gold HAVING n > 1) SELECT k FROM t",
        # a name an alias written in the query loses
        "WITH t (k, n) AS (SELECT gold + 1 AS g, COUNT(*) FROM rounds GROUP BY g) SELECT k FROM t",
        "WITH t (k, n) AS (SELECT gold, COUNT(*) AS wins FROM rounds GROUP BY gold HAVING wins > 1) SELECT k FROM t",
        "WITH t (id) AS (SELECT id AS x FROM other) SELECT id FROM t",
    ],
)
def test_other_entries_are_left_alone(sql):
    assert reading(sql) is None


def test_the_tree_given_is_not_changed():
    sql = "WITH t(id) AS (SELECT gold FROM rounds) SELECT t.id FROM t"
    tree = sqlglot.parse_one(sql, read="mysql")
    written, changed = name_listed_cte_outputs(tree)
    assert changed and tree.sql(dialect="mysql") == sql and written is not tree
    same, changed = name_listed_cte_outputs(sqlglot.parse_one("SELECT gold FROM rounds", read="mysql"))
    assert not changed


REPLAYED = [
    "WITH t (id) AS (SELECT gold FROM rounds UNION ALL SELECT silver FROM rounds UNION ALL SELECT bronze FROM rounds) SELECT t.id, COUNT(*) FROM t GROUP BY t.id",
    "WITH t (k, c) AS (SELECT gold, COUNT(*) FROM rounds GROUP BY gold) SELECT k, c FROM t WHERE c > 1",
    "WITH t (gold, n) AS (SELECT gold, COUNT(*) AS wins FROM rounds GROUP BY gold) SELECT * FROM t",
    "WITH a (x) AS (SELECT gold FROM rounds), c (y) AS (SELECT x + 1 FROM a WHERE x > 1) SELECT y FROM c",
    "WITH t (id, name) AS (SELECT m.member_id, m.name FROM members AS m JOIN rounds AS r ON r.gold = m.member_id) SELECT name, COUNT(id) FROM t GROUP BY name",
    "WITH t (id) AS ((SELECT gold FROM rounds) UNION (SELECT silver FROM rounds)) SELECT id FROM t",
    "SELECT d.id FROM (WITH t (id) AS (SELECT gold FROM rounds WHERE silver > 1) SELECT id FROM t) AS d",
]


@pytest.mark.parametrize("sql", REPLAYED)
def test_replay(sql):
    con = duckdb.connect()
    con.execute("CREATE TABLE members (member_id INT, name VARCHAR)")
    con.execute("CREATE TABLE rounds (round_id INT, gold INT, silver INT, bronze INT)")
    con.execute("INSERT INTO members VALUES (1, 'a'), (2, 'b'), (3, 'a'), (4, NULL)")
    con.execute("INSERT INTO rounds VALUES (1, 1, 2, 3), (2, 1, 3, 2), (3, 2, 1, 4), (4, 4, 2, 1), (5, NULL, 2, 2)")
    written, changed = name_listed_cte_outputs(sqlglot.parse_one(sql, read="duckdb"))
    assert changed and not any(cte.args["alias"].args.get("columns") for cte in written.find_all(sqlglot.exp.CTE))
    before = con.execute(sql)
    names, rows = [d[0] for d in before.description], sorted(before.fetchall(), key=repr)
    after = con.execute(written.sql(dialect="duckdb"))
    assert [d[0] for d in after.description] == names and sorted(after.fetchall(), key=repr) == rows and rows


FACTS = {"members": TableConstraints(not_null=frozenset({"member_id"}), keys=(("member_id",),)), "rounds": TableConstraints(not_null=frozenset({"round_id"}), keys=(("round_id",),))}
OPTIONS = dict(schema=SCHEMA, types=TYPES, constraints=FACTS, dialect="mysql", compare_names=False, exact_arithmetic=True, timeout_ms=5000)
UNPIVOT = "SELECT gold AS m FROM rounds UNION ALL SELECT silver AS m FROM rounds UNION ALL SELECT bronze AS m FROM rounds"
WINS = f"SELECT t.m AS member_id, p.name, COUNT(*) AS wins FROM ({UNPIVOT}) AS t JOIN members AS p ON t.m = p.member_id GROUP BY t.m, p.name"
LISTED = "WITH t (id) AS (SELECT gold FROM rounds UNION ALL SELECT silver FROM rounds UNION ALL SELECT bronze FROM rounds) SELECT member_id, name, COUNT(t.id) AS wins FROM t INNER JOIN members AS p ON p.member_id = t.id GROUP BY p.member_id"


def test_a_query_that_uses_listed_names_is_proved():
    result = prove_equivalent_algebraic(WINS, LISTED, **OPTIONS)
    assert result.proven, result.reason
    with shape_fallback.without("listed_cte_names"):
        # the solver layer reads the list itself since batch 26 (``listed_body``; tests/test_listed_cte_reading.py)
        assert prove_equivalent_algebraic(WINS, LISTED, **OPTIONS).proven
    # the list names the outputs by position: the other medal is another query
    other = LISTED.replace("WITH t (id) AS (SELECT gold FROM rounds", "WITH t (id) AS (SELECT round_id FROM rounds")
    assert not prove_equivalent_algebraic(WINS, other, **OPTIONS).proven
    plain = "WITH t (k, c) AS (SELECT gold, COUNT(*) FROM rounds GROUP BY gold) SELECT k FROM t WHERE c > 1"
    assert prove_equivalent_algebraic(plain, "SELECT gold FROM rounds GROUP BY gold HAVING COUNT(*) > 1", **OPTIONS).proven
    assert not prove_equivalent_algebraic(plain, "SELECT gold FROM rounds GROUP BY gold HAVING COUNT(*) > 2", **OPTIONS).proven


def test_it_is_a_rule_of_the_thirteenth_batch():
    assert "listed_cte_names" in shape_fallback.THIRTEENTH_BATCH
    sql = "WITH t (id) AS (SELECT gold FROM rounds) SELECT t.id FROM t"
    with shape_fallback.tracking() as fired:
        normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "listed_cte_names" in fired
    with shape_fallback.without("listed_cte_names"), shape_fallback.tracking() as fired:
        try:
            normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
        except Exception:  # the prover before the batch declines the listed name
            pass
    assert "listed_cte_names" not in fired
