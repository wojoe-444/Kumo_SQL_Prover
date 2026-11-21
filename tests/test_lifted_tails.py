"""A LIMIT on the driving rows of a query is the LIMIT of the query when each row stays one row (``lifted_tails``)."""

import random

import duckdb
import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.lifted_tails import NAME, lift
from kumo_sql_prover.smt_equivalence import TIE_ASSUMPTION, TableConstraints

SCHEMA = {
    "notes": ["id", "title", "made", "author", "kind"],
    "people": ["id", "name", "team"],
    "replies": ["id", "note", "body"],
    "votes": ["id", "note", "weight"],
}
TYPES = {
    "notes": {"id": "INT", "title": "TEXT", "made": "INT", "author": "INT", "kind": "INT"},
    "people": {"id": "INT", "name": "TEXT", "team": "INT"},
    "replies": {"id": "INT", "note": "INT", "body": "TEXT"},
    "votes": {"id": "INT", "note": "INT", "weight": "INT"},
}
KEYS = {table: [("id",)] for table in SCHEMA}
NOT_NULL = {table: {"id"} for table in SCHEMA}
CONSTRAINTS = {table: TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)) for table in SCHEMA}

REPLIES = "(SELECT note, COUNT(*) AS n FROM replies GROUP BY note)"
# the cut last: the query every spelling below is another way to write
LAST = f"SELECT n.title, p.name, COALESCE(r.n, 0) AS replies FROM notes AS n LEFT JOIN people AS p ON n.author = p.id LEFT JOIN {REPLIES} AS r ON n.id = r.note ORDER BY n.made DESC, n.id LIMIT 3"
TOP = "SELECT id, title, made, author FROM notes ORDER BY made DESC, id LIMIT 3"
DERIVED = f"SELECT t.title, p.name, COALESCE(r.n, 0) AS replies FROM ({TOP}) AS t LEFT JOIN people AS p ON t.author = p.id LEFT JOIN {REPLIES} AS r ON t.id = r.note"
NAMED = f"WITH top AS ({TOP}) SELECT t.title, p.name, COALESCE(r.n, 0) AS replies FROM top AS t LEFT JOIN people AS p ON t.author = p.id LEFT JOIN {REPLIES} AS r ON t.id = r.note ORDER BY t.made DESC"
LISTED = NAMED.replace(REPLIES, "(SELECT note, COUNT(*) AS n FROM replies WHERE note IN (SELECT id FROM top) GROUP BY note)")
JOINED = NAMED.replace(REPLIES, "(SELECT c.note, COUNT(*) AS n FROM replies AS c JOIN top AS k ON c.note = k.id GROUP BY c.note)")
JOINED_BY_TOP = NAMED.replace(REPLIES, "(SELECT k.id AS note, COUNT(*) AS n FROM top AS k JOIN replies AS c ON k.id = c.note WHERE c.body IS NOT NULL GROUP BY k.id)")
COUNTED_WITH = (
    f"WITH top AS ({TOP}), counted AS (SELECT c.note, COUNT(*) AS n FROM replies AS c JOIN top AS k ON c.note = k.id GROUP BY c.note) "
    "SELECT t.title, p.name, COALESCE(r.n, 0) AS replies FROM top AS t LEFT JOIN people AS p ON t.author = p.id LEFT JOIN counted AS r ON t.id = r.note"
)
NUMBERED = (
    "SELECT title, made, replies FROM (SELECT n.id, n.title, n.made, COUNT(r.id) AS replies, ROW_NUMBER() OVER (ORDER BY n.made DESC, n.id) AS rn "
    "FROM notes AS n LEFT JOIN replies AS r ON n.id = r.note GROUP BY n.id, n.title, n.made) AS sub WHERE rn <= 3 ORDER BY made"
)
BOTH_CUT = f"SELECT t.title, p.name FROM ({TOP}) AS t LEFT JOIN people AS p ON t.author = p.id ORDER BY t.made DESC, t.id LIMIT 2"
OFFSET = f"SELECT t.title, v.weight FROM ({TOP} OFFSET 1) AS t LEFT JOIN (SELECT * FROM votes WHERE weight > 0) AS v ON t.made = v.id"
STARRED = "SELECT s.title, s.made FROM (SELECT * FROM notes ORDER BY made DESC, id LIMIT 4) AS s"
READ = [DERIVED, NAMED, LISTED, JOINED, JOINED_BY_TOP, COUNTED_WITH, NUMBERED, NUMBERED.replace("rn <= 3", "sub.rn < 4"), BOTH_CUT, OFFSET, STARRED]

# each breaks one condition of a reading: nothing is lifted
LEFT_ALONE = [
    DERIVED.replace("LEFT JOIN people", "JOIN people"),  # an inner join can drop one of the first rows
    DERIVED + " WHERE p.team = 1",  # a filter above the cut
    DERIVED.replace(f"LEFT JOIN {REPLIES} AS r ON t.id = r.note", "LEFT JOIN replies AS r ON t.id = r.note").replace("COALESCE(r.n, 0)", "r.body"),  # two rows
    DERIVED.replace("ON t.author = p.id", "ON t.author = p.team"),  # no key of the joined table
    DERIVED.replace("ON t.id = r.note", "ON t.id = r.n"),  # not the grouped column
    DERIVED.replace("ON t.author = p.id", "ON t.author < p.id"),
    DERIVED.replace("ON t.author = p.id", "ON t.author = p.id AND p.team = 1"),
    DERIVED.replace("SELECT t.title", "SELECT DISTINCT t.title"),
    DERIVED.replace("COALESCE(r.n, 0) AS replies", "MAX(r.n) AS replies") + " GROUP BY t.title, p.name",
    DERIVED.replace("p.name,", "p.name, (SELECT MAX(id) FROM votes) AS top,"),
    DERIVED.replace("p.name,", "p.name, RANDOM() AS chance,"),
    DERIVED.replace("ORDER BY made DESC, id LIMIT 3", "ORDER BY kind LIMIT 3"),  # a key the driving table does not show
    DERIVED.replace("LIMIT 3", "LIMIT (SELECT 3)"),
    DERIVED.replace("ORDER BY made DESC, id LIMIT 3", "LIMIT 3"),
    DERIVED + " LIMIT 2",  # another ordering than the driving table's (none)
    BOTH_CUT.replace("ORDER BY t.made DESC, t.id LIMIT 2", "ORDER BY t.made, t.id LIMIT 2"),
    BOTH_CUT.replace("LIMIT 2", "LIMIT 2 OFFSET 1"),
    DERIVED.replace("LEFT JOIN people", "RIGHT JOIN people"),
    # the WITH query is read twice and the second reading is no restriction this module removes
    NAMED.replace(REPLIES, "(SELECT note, COUNT(*) AS n FROM replies WHERE id IN (SELECT id FROM top) GROUP BY note)"),
    NAMED.replace(REPLIES, "(SELECT note, COUNT(*) AS n FROM replies WHERE note NOT IN (SELECT id FROM top) GROUP BY note)"),
    NAMED.replace(REPLIES, "(SELECT note, COUNT(*) AS n FROM replies WHERE note IN (SELECT made FROM top) GROUP BY note)"),
    NAMED.replace(REPLIES, "(SELECT c.note, COUNT(*) AS n FROM replies AS c JOIN top AS k ON c.note = k.made GROUP BY c.note)"),
    NAMED.replace(REPLIES, "(SELECT c.note, COUNT(*) AS n FROM replies AS c LEFT JOIN top AS k ON c.note = k.id GROUP BY c.note)"),
    NAMED.replace(REPLIES, "(SELECT c.note, MAX(k.made) AS n FROM replies AS c JOIN top AS k ON c.note = k.id GROUP BY c.note)"),
    # the JOIN form needs one row of the WITH query per id
    JOINED.replace(TOP, "SELECT author AS id, title, made, author FROM notes ORDER BY made DESC, id LIMIT 3"),
    NUMBERED.replace("ROW_NUMBER() OVER (ORDER BY", "ROW_NUMBER() OVER (PARTITION BY n.kind ORDER BY").replace("n.title, n.made)", "n.title, n.made, n.kind)"),
    NUMBERED.replace("ROW_NUMBER()", "RANK()"),
    NUMBERED.replace("rn <= 3", "rn <= 3 AND made > 1"),
    NUMBERED.replace("rn <= 3", "rn >= 3"),
    NUMBERED.replace("rn <= 3", "rn < 1"),
    NUMBERED.replace("SELECT title, made, replies FROM", "SELECT title, made, rn FROM"),  # the number is shown
    NUMBERED.replace("ORDER BY n.made DESC, n.id) AS rn", "ORDER BY n.kind) AS rn"),  # a key the numbered select does not show
    STARRED.replace("ORDER BY made DESC, id", "ORDER BY made DESC, 1"),  # a position in a star
    LAST,
    "SELECT title FROM notes ORDER BY made LIMIT 3",
]


def lifted(sql: str):
    return lift(sql, "postgres", SCHEMA, KEYS, NOT_NULL)


def prove(left: str, right: str):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, constraints=CONSTRAINTS, types=TYPES, compare_names=False, dialect="postgres", exact_arithmetic=True, timeout_ms=5000
    )


def test_the_tail_of_the_driving_table_moves_to_the_end_of_the_query():
    assert lifted(DERIVED) == (
        "SELECT t.title, p.name, COALESCE(r.n, 0) AS replies FROM (SELECT id, title, made, author FROM notes) AS t "
        "LEFT JOIN people AS p ON t.author = p.id LEFT JOIN (SELECT note, COUNT(*) AS n FROM replies GROUP BY note) AS r ON t.id = r.note "
        "ORDER BY t.made DESC, t.id LIMIT 3"
    )
    assert lifted(OFFSET).endswith("ORDER BY t.made DESC, t.id LIMIT 3 OFFSET 1")
    assert lifted(STARRED) == "SELECT s.title, s.made FROM (SELECT * FROM notes) AS s ORDER BY 2 DESC, s.id LIMIT 4"


def test_a_row_number_test_becomes_the_tail_and_a_shown_key_is_named_by_its_position():
    assert lifted(NUMBERED) == (
        "SELECT title, made, replies FROM (SELECT n.id, n.title, n.made, COUNT(r.id) AS replies FROM notes AS n LEFT JOIN replies AS r ON n.id = r.note "
        "GROUP BY n.id, n.title, n.made) AS sub ORDER BY 2 DESC, sub.id LIMIT 3"
    )


def test_a_restriction_to_the_driving_rows_is_removed():
    plain = lifted(NAMED)
    assert plain.startswith("WITH top AS (SELECT id, title, made, author FROM notes) SELECT") and plain.endswith("ORDER BY t.made DESC, t.id LIMIT 3")
    assert lifted(LISTED) == plain
    assert lifted(JOINED) == plain.replace("SELECT note, COUNT(*) AS n FROM replies GROUP BY note", "SELECT c.note, COUNT(*) AS n FROM replies AS c GROUP BY c.note")
    assert "FROM replies AS c WHERE NOT c.body IS NULL GROUP BY c.note" in lifted(JOINED_BY_TOP)
    assert "counted AS (SELECT c.note, COUNT(*) AS n FROM replies AS c GROUP BY c.note)" in lifted(COUNTED_WITH)


def test_names_are_written_as_the_query_writes_them():
    sql = "SELECT T.Title FROM (SELECT Id, Title, Made AS Day FROM Notes ORDER BY Made DESC, Id LIMIT 3) AS T LEFT JOIN People AS P ON T.Id = P.Id"
    assert lift(sql, "mysql", SCHEMA, KEYS, NOT_NULL).endswith("ON T.Id = P.Id ORDER BY T.Day DESC, T.Id LIMIT 3")


def test_the_smaller_of_two_cuts_under_one_ordering_is_kept():
    assert lifted(BOTH_CUT).endswith("ORDER BY t.made DESC, t.id LIMIT 2")
    assert lifted(BOTH_CUT.replace("LIMIT 2", "LIMIT 7")).endswith("ORDER BY t.made DESC, t.id LIMIT 3")


@pytest.mark.parametrize("sql", READ)
def test_every_readable_query_is_lifted(sql):
    new = lifted(sql)
    assert new is not None and new != sqlglot.parse_one(sql, read="postgres").sql("postgres")
    assert lifted(new) is None  # once is enough


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_a_query_that_breaks_a_condition_is_left_alone(sql):
    assert lifted(sql) is None


def test_nothing_is_lifted_without_declared_tables():
    assert lift(DERIVED, "postgres", None, KEYS, NOT_NULL) is None
    assert lift(DERIVED, "postgres", SCHEMA, None, NOT_NULL) is None  # the lookup is not known to find one row
    assert lift(DERIVED, "postgres", SCHEMA, KEYS, None) is None


def _database(seed: int):
    rng = random.Random(seed)
    con = duckdb.connect()
    con.execute("CREATE TABLE notes (id INT, title TEXT, made INT, author INT, kind INT)")
    con.execute("CREATE TABLE people (id INT, name TEXT, team INT)")
    con.execute("CREATE TABLE replies (id INT, note INT, body TEXT)")
    con.execute("CREATE TABLE votes (id INT, note INT, weight INT)")

    def maybe(value):
        return None if rng.random() < 0.2 else value

    for i in range(rng.randint(0, 8)):
        con.execute("INSERT INTO notes VALUES (?, ?, ?, ?, ?)", [i, maybe(rng.choice("ab")), maybe(rng.randint(1, 4)), maybe(rng.randint(0, 4)), maybe(rng.randint(1, 2))])
    for i in range(rng.randint(0, 4)):
        con.execute("INSERT INTO people VALUES (?, ?, ?)", [i, maybe(rng.choice("xy")), maybe(rng.randint(0, 2))])
    for i in range(rng.randint(0, 14)):
        con.execute("INSERT INTO replies VALUES (?, ?, ?)", [i, maybe(rng.randint(0, 8)), maybe(rng.choice("xz"))])
    for i in range(rng.randint(0, 6)):
        con.execute("INSERT INTO votes VALUES (?, ?, ?)", [i, maybe(rng.randint(0, 8)), maybe(rng.randint(0, 3))])
    return con


def _rows(con, sql: str):
    return sorted(con.execute(sqlglot.transpile(sql, read="postgres", write="duckdb")[0]).fetchall(), key=repr)


def test_the_lifted_query_returns_the_same_rows_in_duckdb():
    # every ordering here ends in the key, so no two rows tie and each query has one result
    written = [(sql, lifted(sql)) for sql in READ]
    cut = 0
    for seed in range(14):
        con = _database(seed)
        for sql, new in written:
            rows = _rows(con, sql)
            assert rows == _rows(con, new), (seed, sql)
        cut += con.execute("SELECT COUNT(*) > 3 FROM notes").fetchone()[0]
        con.close()
    assert cut > 4  # the databases do hold more rows than the cut keeps


def test_duckdb_tells_the_refused_spellings_apart():
    # why three of the conditions are there: these lifts, done by hand, give other rows
    by_hand = [
        (DERIVED.replace("LEFT JOIN people", "JOIN people"), LAST.replace("LEFT JOIN people", "JOIN people")),
        (DERIVED + " WHERE p.team = 1", LAST.replace("ORDER BY", "WHERE p.team = 1 ORDER BY")),
        (
            JOINED.replace(TOP, "SELECT author AS id, title, made, author FROM notes ORDER BY made DESC, id LIMIT 3"),
            NAMED.replace(TOP, "SELECT author AS id, title, made, author FROM notes ORDER BY made DESC, id LIMIT 3"),
        ),
    ]
    for left, right in by_hand:
        assert any(_rows(con, left) != _rows(con, right) for con in map(_database, range(40))), left


@pytest.mark.parametrize("other", [DERIVED, LISTED, JOINED])
def test_the_cut_first_and_the_cut_last_are_proved_equal(other):
    result = prove(LAST, other)
    assert result.proven, result.reason
    assert TIE_ASSUMPTION in result.assumptions
    with shape_fallback.without(NAME):
        assert not prove(LAST, other).proven  # the step is what proves it


LIMITED = "SELECT n.title, n.made, COUNT(r.id) AS replies FROM notes AS n LEFT JOIN replies AS r ON n.id = r.note GROUP BY n.id, n.title, n.made ORDER BY n.made DESC, n.id LIMIT 3"


def test_a_row_number_cut_is_proved_equal_to_a_limit():
    result = prove(NUMBERED, LIMITED)
    assert result.proven and TIE_ASSUMPTION in result.assumptions


@pytest.mark.parametrize("tail, proven", [("ORDER BY made DESC, id LIMIT 3", True), ("ORDER BY made DESC, id LIMIT 4", False), ("ORDER BY made, id LIMIT 3", False)])
def test_a_row_number_cut_is_that_limit_and_no_other(tail, proven):
    numbered = "SELECT title FROM (SELECT title, made, id, ROW_NUMBER() OVER (ORDER BY made DESC, id) AS rn FROM notes) AS s WHERE rn <= 3"
    assert prove(numbered, f"SELECT title FROM notes {tail}").proven is proven


@pytest.mark.parametrize(
    "old, new",
    [("LIMIT 3", "LIMIT 4"), ("ORDER BY made DESC, id LIMIT 3", "ORDER BY made, id LIMIT 3"), ("LEFT JOIN people", "JOIN people")],
)
def test_another_cut_or_other_rows_are_not_proved(old, new):
    assert old in DERIVED
    assert not prove(LAST, DERIVED.replace(old, new)).proven
