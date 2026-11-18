"""A second copy of the outer row inside an ``EXISTS`` is the outer row itself (``own_row_copies``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.own_row_copies import drop_own_row_copies

SCHEMA = {"patron": ["patron_id", "patron_name"], "loan": ["patron_id", "title"]}
NAMED = {"patron": frozenset({"patron_name"})}
COPY = (
    "SELECT p.patron_id FROM patron AS p WHERE EXISTS (SELECT 1 FROM patron AS k, loan AS o "
    "WHERE k.patron_id = o.patron_id AND o.title = 'B' AND k.patron_id = p.patron_id AND k.patron_name = p.patron_name)"
)
PLAIN = "SELECT p.patron_id FROM patron AS p WHERE EXISTS(SELECT 1 FROM loan AS o WHERE p.patron_id = o.patron_id AND o.title = 'B')"
GUARDED = PLAIN.replace("o.title = 'B')", "o.title = 'B' AND NOT p.patron_name IS NULL)")


def reading(sql: str, not_null=None, types=None) -> str | None:
    result = drop_own_row_copies(sqlglot.parse_one(sql, read="mysql"), not_null, types)
    return result.sql(dialect="mysql") if result is not None else None


def test_the_copy_is_the_outer_row():
    # the column that nothing else compares is not NULL on the outer row; declared NOT NULL, nothing is left to say
    assert reading(COPY) == GUARDED
    assert reading(COPY, NAMED) == PLAIN
    assert reading(COPY.replace("EXISTS", "NOT EXISTS"), NAMED) == PLAIN.replace("EXISTS", "NOT EXISTS")
    assert reading(PLAIN) is None
    # joined with ON, the equalities either way round, the copy last
    joined = COPY.replace("FROM patron AS k, loan AS o WHERE k.patron_id = o.patron_id AND", "FROM loan AS o JOIN patron AS k ON o.patron_id = k.patron_id WHERE").replace("k.patron_name = p.patron_name", "p.patron_name = k.patron_name")
    assert reading(joined, NAMED) == "SELECT p.patron_id FROM patron AS p WHERE EXISTS(SELECT 1 FROM loan AS o WHERE o.patron_id = p.patron_id AND o.title = 'B')"
    # under OR, and two tests in one WHERE
    both = COPY + " OR " + COPY.split(" WHERE ", 1)[1].replace("'B'", "'C'").replace("EXISTS", "NOT EXISTS")
    assert reading(both, NAMED) == PLAIN + " OR NOT " + PLAIN.split(" WHERE ", 1)[1].replace("'B'", "'C'")
    # a nested query that reads the copy reads the outer row
    nested = COPY.replace("o.title = 'B'", "o.title = (SELECT MAX(t.title) FROM loan AS t WHERE t.patron_id = k.patron_id)")
    assert reading(nested, NAMED) == PLAIN.replace("o.title = 'B'", "o.title = (SELECT MAX(t.title) FROM loan AS t WHERE t.patron_id = p.patron_id)")
    # the outer row on the padded side of a join: its NOT NULL column may be NULL there
    padded = COPY.replace("FROM patron AS p WHERE", "FROM loan AS x LEFT JOIN patron AS p ON x.patron_id = p.patron_id WHERE")
    assert reading(padded, NAMED) == GUARDED.replace("FROM patron AS p WHERE", "FROM loan AS x LEFT JOIN patron AS p ON x.patron_id = p.patron_id WHERE")


LEFT_ALONE = [
    COPY.replace("k.patron_name = p.patron_name", "k.patron_name <> p.patron_name"),  # another row
    COPY.replace(" AND k.patron_name = p.patron_name", " AND k.patron_name = 'n'").replace("k.patron_id = p.patron_id", "k.patron_name = p.patron_id"),  # not the same column
    COPY.replace(" AND k.patron_id = p.patron_id", ""),  # the copy's id is read and not tied
    COPY.replace("k.patron_name = p.patron_name", "(k.patron_name = p.patron_name OR o.title = 'x')"),
    COPY.replace("FROM patron AS k, loan AS o", "FROM loan AS k, loan AS o"),  # not the outer row's table
    COPY.replace("FROM patron AS k, loan AS o WHERE k.patron_id = o.patron_id AND", "FROM patron AS k LEFT JOIN loan AS o ON k.patron_id = o.patron_id WHERE"),
    COPY.replace("SELECT 1 FROM", "SELECT COUNT(*) FROM"),  # always one row
    COPY.replace("k.patron_name = p.patron_name)", "k.patron_name = p.patron_name GROUP BY o.title)"),
    COPY.replace("k.patron_name = p.patron_name)", "k.patron_name = p.patron_name LIMIT 1 OFFSET 1)"),
    COPY.replace("o.title = 'B'", "title = 'B'"),  # a bare name could be the copy's
    COPY.replace("SELECT 1 FROM", "SELECT k.* FROM"),
    COPY.replace("o.title = 'B'", "o.title IN (SELECT k.title FROM loan AS k)"),  # another source under the copy's name
    COPY.replace("o.title = 'B'", "o.title IN (SELECT p.title FROM loan AS p)"),  # another source under the outer row's name
    COPY.replace("k.patron_name = p.patron_name", "k.`patron_name` = p.`patron_name`"),
    COPY.replace("o.title = 'B'", "o.title = 'B' AND RAND() < 0.5"),
    "SELECT p.patron_id FROM patron AS p WHERE EXISTS (SELECT 1 FROM patron AS k WHERE k.patron_id = p.patron_id)",  # nothing else is asked
    "SELECT p.patron_id FROM (SELECT * FROM patron) AS p WHERE EXISTS (SELECT 1 FROM patron AS k, loan AS o WHERE k.patron_id = o.patron_id AND k.patron_id = p.patron_id)",
    "SELECT p.patron_id FROM patron AS p WHERE p.patron_id IN (SELECT k.patron_id FROM patron AS k, loan AS o WHERE k.patron_id = o.patron_id AND k.patron_name = p.patron_name)",
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_shapes_are_left_alone(sql):
    sqlglot.parse_one(sql, read="mysql")
    assert reading(sql, NAMED) is None


def test_a_floating_point_column_is_not_read_for_itself():
    assert reading(COPY, NAMED, {"patron": {"patron_id": "INT", "patron_name": "DOUBLE"}}) is None
    assert reading(COPY, NAMED, {"patron": {"patron_id": "INT", "patron_name": "VARCHAR"}}) == PLAIN


SPELLINGS = [
    COPY,
    COPY.replace("EXISTS", "NOT EXISTS"),
    COPY.replace("FROM patron AS p WHERE", "FROM loan AS x LEFT JOIN patron AS p ON x.patron_id = p.patron_id WHERE"),
    COPY.replace("o.title = 'B'", "o.title = (SELECT MAX(t.title) FROM loan AS t WHERE t.patron_id = k.patron_id)"),
    "SELECT p.patron_id, p.patron_name FROM patron AS p WHERE p.patron_name <> 'n' OR NOT EXISTS (SELECT 1 FROM loan AS o JOIN patron AS k ON o.title <> k.patron_name WHERE k.patron_name = p.patron_name AND o.patron_id > 2)",
    "SELECT COUNT(*) FROM patron AS p, loan AS x WHERE x.patron_id = p.patron_id AND EXISTS (SELECT 1 FROM patron AS k, loan AS o WHERE k.patron_id = p.patron_id AND o.title = x.title AND o.patron_id <> k.patron_id)",
]


@pytest.mark.parametrize("seed", range(4))
def test_the_reading_returns_the_rows_duckdb_returns(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    pairs = []
    for sql in SPELLINGS:
        for facts in (None, NAMED):
            read = reading(sql, facts)
            assert read is not None, sql
            pairs.append((sql, read, facts is not None))
    nonempty = guarded = 0
    for _ in range(30):
        named = rng.random() < 0.5  # every patron has a name: the declared fact holds
        con = duckdb.connect()
        con.execute("CREATE TABLE patron (patron_id INT, patron_name VARCHAR)")
        con.execute("CREATE TABLE loan (patron_id INT, title VARCHAR)")
        for _ in range(rng.randint(0, 6)):
            con.execute("INSERT INTO patron VALUES (?, ?)", [rng.choice([None, 1, 2, 3, 3, 4]), rng.choice(["n", "m", "A"] if named else [None, "n", "m", "A"])])
        for _ in range(rng.randint(0, 7)):
            con.execute("INSERT INTO loan VALUES (?, ?)", [rng.choice([None, 1, 2, 3, 5]), rng.choice([None, "A", "B", "m"])])
        for sql, read, needs_names in pairs:
            expected = sorted(con.execute(sql).fetchall(), key=repr)
            same = expected == sorted(con.execute(read).fetchall(), key=repr)
            if named or not needs_names:
                assert same, (sql, read)
                nonempty += bool(expected)
            else:
                guarded += not same
    assert nonempty > 40 and guarded > 0  # without the declared NOT NULL the guard is what keeps it true


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

LISTS = (
    "SELECT patron_id, patron_name FROM patron WHERE patron_id IN (SELECT DISTINCT patron_id FROM loan WHERE title = 'A') "
    "AND patron_id IN (SELECT DISTINCT patron_id FROM loan WHERE title = 'B') AND patron_id NOT IN (SELECT DISTINCT patron_id FROM loan WHERE title = 'C')"
)
SUMS = (
    "SELECT s.patron_id, p.patron_name FROM (SELECT patron_id, SUM(CASE WHEN title = 'A' THEN 1 ELSE 0 END) AS a, SUM(CASE WHEN title = 'B' THEN 1 ELSE 0 END) AS b, "
    "SUM(CASE WHEN title = 'C' THEN 1 ELSE 0 END) AS c FROM loan GROUP BY patron_id) s JOIN patron p ON s.patron_id = p.patron_id WHERE a > 0 AND b > 0 AND c = 0"
)


def facts(named: bool):
    return {
        "patron": TableConstraints(not_null=frozenset({"patron_id"} | ({"patron_name"} if named else set())), keys=(("patron_id",),)),
        "loan": TableConstraints(not_null=frozenset({"patron_id", "title"})),
    }


def test_a_declared_not_null_name_no_longer_loses_the_proof():
    # "borrowed A and B and never C", as three lists and as a table of counts; proved with the name nullable before this rule
    for named in (False, True):
        assert prove_equivalent_algebraic(LISTS, SUMS, schema=SCHEMA, constraints=facts(named), compare_names=False, dialect="mysql").proven


def test_it_is_a_narrow_retry_rule():
    constraints = facts(True)
    with shape_fallback.tracking() as fired:
        normalize(SUMS, schema=SCHEMA, dialect="mysql", keys={t: list(c.keys) for t, c in constraints.items()}, not_null={t: c.not_null for t, c in constraints.items()})
    assert "own_row_copies" in fired
