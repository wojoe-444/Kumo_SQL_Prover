"""Under ``DISTINCT``, a joined table that no output reads is only asked for a match (``distinct_match_joins``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.distinct_match_joins import ask_unlisted_joins

SCHEMA = {"patron": ["patron_id", "patron_name"], "loan": ["patron_id", "title"]}
JOINED = (
    "SELECT DISTINCT p.patron_id, p.patron_name FROM patron AS p JOIN loan AS o1 ON p.patron_id = o1.patron_id AND o1.title = 'A' "
    "JOIN loan AS o2 ON p.patron_id = o2.patron_id AND o2.title = 'B' WHERE p.patron_id NOT IN (SELECT patron_id FROM loan WHERE title = 'C')"
)
ASKED = (
    "SELECT DISTINCT p.patron_id, p.patron_name FROM patron AS p WHERE NOT p.patron_id IN (SELECT loan.patron_id FROM loan WHERE loan.title = 'C') "
    "AND EXISTS(SELECT 1 FROM loan AS o1 WHERE p.patron_id = o1.patron_id AND o1.title = 'A') AND EXISTS(SELECT 1 FROM loan AS o2 WHERE p.patron_id = o2.patron_id AND o2.title = 'B')"
)


def reading(sql: str, types=None, schema=SCHEMA) -> str | None:
    result = ask_unlisted_joins(sqlglot.parse_one(sql, read="mysql"), types, schema)
    return result.sql(dialect="mysql") if result is not None else None


def test_tables_no_output_reads_are_asked_for_a_match():
    assert reading(JOINED) == ASKED
    assert reading(ASKED) is None
    # comma joins with the conditions in the WHERE
    commas = JOINED.replace(" JOIN loan AS o1 ON ", ", loan AS o1, loan AS o2 WHERE ").replace(" JOIN loan AS o2 ON ", " AND ").replace(" WHERE p.patron_id NOT IN", " AND p.patron_id NOT IN")
    assert sqlglot.parse_one(commas, read="mysql").args.get("joins") and reading(commas) is not None
    # two asked tables a conjunct reads together are asked together
    together = reading(JOINED.replace("p.patron_id = o2.patron_id", "o1.title < o2.title"))
    assert "EXISTS(SELECT 1 FROM loan AS o1, loan AS o2 WHERE p.patron_id = o1.patron_id AND o1.title = 'A' AND o1.title < o2.title AND o2.title = 'B')" in together
    # tied only through a kept column, they are asked apart
    assert reading(JOINED.replace("p.patron_id = o2.patron_id", "o1.patron_id = o2.patron_id")) == ASKED
    # a table an output reads stays, and a nested query that reads an asked table goes with it
    listed = reading(
        "SELECT DISTINCT p.patron_id, o1.title FROM patron AS p, loan AS o1, loan AS o2, patron AS q WHERE p.patron_id = o1.patron_id AND o2.patron_id = q.patron_id "
        "AND q.patron_name = p.patron_name AND EXISTS (SELECT 1 FROM loan AS z WHERE z.title = o2.title AND z.patron_id <> p.patron_id)"
    )
    assert listed == (
        "SELECT DISTINCT p.patron_id, o1.title FROM patron AS p, loan AS o1 WHERE p.patron_id = o1.patron_id AND EXISTS(SELECT 1 FROM loan AS o2, patron AS q "
        "WHERE o2.patron_id = q.patron_id AND q.patron_name = p.patron_name AND EXISTS(SELECT 1 FROM loan AS z WHERE z.title = o2.title AND z.patron_id <> p.patron_id))"
    )


LEFT_ALONE = [
    JOINED.replace("SELECT DISTINCT", "SELECT"),  # the joins multiply the rows
    JOINED.replace(" WHERE p.patron_id NOT IN (SELECT patron_id FROM loan WHERE title = 'C')", ""),  # no nested query: other rules read it
    JOINED.replace(" JOIN loan AS o2 ON p.patron_id = o2.patron_id AND o2.title = 'B'", ""),  # two tables: keyed_semijoin reads it
    JOINED.replace("JOIN loan AS o2", "LEFT JOIN loan AS o2"),
    JOINED.replace("p.patron_id, p.patron_name FROM", "p.patron_id, o1.title, o2.title FROM"),  # every table is listed
    JOINED.replace("p.patron_id, p.patron_name FROM", "p.patron_id, UPPER(p.patron_name) FROM"),
    JOINED.replace("p.patron_id, p.patron_name FROM", "p.patron_id, COUNT(*) OVER () FROM"),
    JOINED + " LIMIT 2",
    JOINED + " ORDER BY p.patron_id",
    JOINED.replace("o2.title = 'B'", "title = 'B'"),  # a bare name
    JOINED.replace("(SELECT patron_id FROM loan WHERE title = 'C')", "(SELECT o2.patron_id FROM loan AS o2 JOIN (SELECT 1 AS x) AS d ON d.x = 1 WHERE o2.title = 'C')"),  # reuses a name
    JOINED.replace(" AND o2.title = 'B'", "").replace("p.patron_id = o2.patron_id", "p.patron_id = p.patron_id"),  # nothing reads the second table
    JOINED.replace("o2.title = 'B'", "o2.title = 'B' AND RAND() < 0.5"),
    JOINED.replace("FROM patron AS p JOIN", "FROM (SELECT * FROM patron) AS p JOIN"),
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_shapes_are_left_alone(sql):
    sqlglot.parse_one(sql, read="mysql")
    assert reading(sql) is None


def test_columns_of_different_types_are_not_read_one_for_the_other():
    through = JOINED.replace("p.patron_id = o2.patron_id", "o1.patron_id = o2.patron_id")
    mixed = reading(through, {"patron": {"patron_id": "VARCHAR"}, "loan": {"patron_id": "INT"}})
    assert mixed is not None and "FROM loan AS o1, loan AS o2" in mixed  # asked together, the equality as written


SPELLINGS = [
    JOINED,
    JOINED.replace("p.patron_id = o2.patron_id", "o1.patron_id = o2.patron_id"),
    JOINED.replace("p.patron_id = o2.patron_id", "o1.title < o2.title"),
    "SELECT DISTINCT p.patron_id, o1.title FROM patron AS p, loan AS o1, loan AS o2, patron AS q WHERE p.patron_id = o1.patron_id AND o2.patron_id = q.patron_id "
    "AND q.patron_name = p.patron_name AND EXISTS (SELECT 1 FROM loan AS z WHERE z.title = o2.title AND z.patron_id <> p.patron_id)",
    "SELECT DISTINCT p.patron_name FROM patron AS p JOIN loan AS o1 ON p.patron_id = o1.patron_id JOIN patron AS q ON q.patron_id = o1.patron_id AND q.patron_name <> p.patron_name "
    "WHERE o1.title IN (SELECT t.title FROM loan AS t WHERE t.patron_id = q.patron_id)",
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
        con.execute("CREATE TABLE patron (patron_id INT, patron_name VARCHAR)")
        con.execute("CREATE TABLE loan (patron_id INT, title VARCHAR)")
        for _ in range(rng.randint(0, 6)):
            con.execute("INSERT INTO patron VALUES (?, ?)", [rng.choice([None, 1, 2, 3, 3]), rng.choice([None, "n", "m", "A"])])
        for _ in range(rng.randint(0, 14)):
            con.execute("INSERT INTO loan VALUES (?, ?)", [rng.choice([None, 1, 2, 3]), rng.choice([None, "A", "A", "B", "B", "C"])])
        for sql, read in pairs:
            expected = sorted(con.execute(sql).fetchall(), key=repr)
            assert expected == sorted(con.execute(read).fetchall(), key=repr), (sql, read)
            nonempty += bool(expected)
    assert nonempty > 20


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

LISTS = (
    "SELECT patron_id, patron_name FROM patron WHERE patron_id IN (SELECT DISTINCT patron_id FROM loan WHERE title = 'A') "
    "AND patron_id IN (SELECT DISTINCT patron_id FROM loan WHERE title = 'B') AND patron_id NOT IN (SELECT DISTINCT patron_id FROM loan WHERE title = 'C')"
)


def facts(key: bool):
    return {
        "patron": TableConstraints(not_null=frozenset({"patron_id", "patron_name"}), keys=(("patron_id",),) if key else ()),
        "loan": TableConstraints(not_null=frozenset({"patron_id", "title"})),
    }


def test_borrowed_two_titles_and_never_a_third_as_lists_and_as_joins():
    assert prove_equivalent_algebraic(LISTS, JOINED, schema=SCHEMA, constraints=facts(True), compare_names=False, dialect="mysql").proven
    # without the key two patrons may share an id and a name: DISTINCT lists them once, the lists twice
    assert not prove_equivalent_algebraic(LISTS, JOINED, schema=SCHEMA, constraints=facts(False), compare_names=False, dialect="mysql").proven


def test_it_is_a_narrow_retry_rule():
    constraints = facts(True)
    with shape_fallback.tracking() as fired:
        normalize(JOINED, schema=SCHEMA, dialect="mysql", keys={t: list(c.keys) for t, c in constraints.items()}, not_null={t: c.not_null for t, c in constraints.items()})
    assert "distinct_match_joins" in fired
