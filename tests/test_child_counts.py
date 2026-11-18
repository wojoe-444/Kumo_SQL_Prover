"""Counts of a parent row's child rows are read in one spelling (``child_counts``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.child_counts import read_child_counts

SCHEMA = {"bill": ["bill_id", "owner_id", "amount"], "member": ["member_id", "member_name", "mail"], "friend": ["owner_id", "friend_name", "friend_mail"]}
KEYS = {"bill": [("bill_id",)], "member": [("member_id",), ("mail",)]}
NOT_NULL = {"bill": frozenset({"bill_id", "owner_id"})}
FOREIGN = {"bill": [(("owner_id",), "member", ("member_id",))]}

# "each bill with its owner's name, the number of the owner's friends and the number of them who are members"
GROUPED = (
    "SELECT i.bill_id, c.member_name, COUNT(f.owner_id) AS friends, COUNT(m.mail) AS trusted FROM bill AS i JOIN member AS c ON c.member_id = i.owner_id "
    "LEFT JOIN friend AS f ON c.member_id = f.owner_id LEFT JOIN member AS m ON f.friend_mail = m.mail GROUP BY i.bill_id, c.member_name"
)
COUNTED = (
    "SELECT i.bill_id, c.member_name, COALESCE(g.n, 0) AS friends, COALESCE(t.n, 0) AS trusted FROM bill AS i JOIN member AS c ON c.member_id = i.owner_id "
    "LEFT JOIN (SELECT owner_id, COUNT(*) AS n FROM friend GROUP BY owner_id) AS g ON i.owner_id = g.owner_id "
    "LEFT JOIN (SELECT owner_id, COUNT(*) AS n FROM friend WHERE friend_mail IN (SELECT mail FROM member) GROUP BY owner_id) AS t ON i.owner_id = t.owner_id"
)
PARENT_FIRST = (
    "SELECT i.bill_id, c.member_name, c.friends, c.trusted FROM bill AS i JOIN (SELECT a.member_id, a.member_name, COUNT(b.owner_id) AS friends, "
    "SUM(CASE WHEN b.friend_mail IN (SELECT mail FROM member) THEN 1 ELSE 0 END) AS trusted FROM member AS a LEFT JOIN friend AS b ON a.member_id = b.owner_id "
    "GROUP BY a.member_id, a.member_name) AS c ON i.owner_id = c.member_id"
)
READ = (
    "SELECT i.bill_id AS bill_id, {c}.member_name AS member_name, COALESCE(kqf.kqn0, 0) AS friends, COALESCE(kqf.kqn1, 0) AS trusted FROM bill AS i "
    "JOIN member AS {c} ON {c}.member_id = i.owner_id LEFT JOIN (SELECT kqs.owner_id AS owner_id, COUNT(*) AS kqn0, "
    "COUNT(CASE WHEN kqs.friend_mail IN (SELECT kqm.mail FROM member AS kqm) THEN 1 END) AS kqn1 FROM friend AS kqs GROUP BY kqs.owner_id) AS kqf ON i.owner_id = kqf.owner_id"
)
# the same counts with no lookup of the owner: flags in a view of the friends, and a list of distinct values asked for a match
FLAGGED = (
    "SELECT i.bill_id, COALESCE(g.n, 0) AS friends, COALESCE(g.t, 0) AS trusted FROM bill AS i LEFT JOIN (SELECT v.owner_id, COUNT(*) AS n, SUM(v.flag) AS t FROM "
    "(SELECT f.*, CASE WHEN m.mail IS NULL THEN 0 ELSE 1 END AS flag FROM friend AS f LEFT JOIN member AS m ON f.friend_mail = m.mail) AS v GROUP BY v.owner_id) AS g ON i.owner_id = g.owner_id"
)
LISTED = (
    "SELECT i.bill_id, COUNT(f.owner_id) AS friends, COUNT(e.mail) AS trusted FROM bill AS i LEFT JOIN friend AS f ON i.owner_id = f.owner_id "
    "LEFT JOIN (SELECT DISTINCT mail FROM member) AS e ON f.friend_mail = e.mail GROUP BY i.bill_id"
)
READ_BARE = (
    "SELECT i.bill_id AS bill_id, COALESCE(kqf.kqn0, 0) AS friends, COALESCE(kqf.kqn1, 0) AS trusted FROM bill AS i LEFT JOIN (SELECT kqs.owner_id AS owner_id, COUNT(*) AS kqn0, "
    "COUNT(CASE WHEN kqs.friend_mail IN (SELECT kqm.mail FROM member AS kqm) THEN 1 END) AS kqn1 FROM friend AS kqs GROUP BY kqs.owner_id) AS kqf ON i.owner_id = kqf.owner_id"
)


def reading(sql: str, schema=SCHEMA, keys=KEYS, not_null=NOT_NULL, foreign=FOREIGN, types=None) -> str | None:
    tree, fired = read_child_counts(sqlglot.parse_one(sql, read="mysql"), schema, keys, not_null, foreign, types)
    return tree.sql(dialect="mysql") if fired else None


def test_the_three_spellings_are_read_alike():
    assert reading(GROUPED) == READ.format(c="c")
    assert reading(COUNTED) == READ.format(c="c")
    assert reading(PARENT_FIRST) == READ.format(c="a")  # the grouped table's own name for the member
    assert reading(READ.format(c="c")) is None  # already in the one spelling


def test_flags_in_a_view_and_a_list_of_distinct_values_are_read_alike():
    assert reading(FLAGGED) == READ_BARE
    assert reading(LISTED) == READ_BARE
    assert reading(FLAGGED.replace("CASE WHEN m.mail IS NULL THEN 0 ELSE 1 END", "CASE WHEN f.friend_mail IN (SELECT mail FROM member) THEN 1 ELSE 0 END").replace(" LEFT JOIN member AS m ON f.friend_mail = m.mail", "")) == READ_BARE
    assert reading(LISTED.replace("SELECT DISTINCT mail FROM member", "SELECT mail FROM member GROUP BY mail")) == READ_BARE
    # a list of distinct values needs no declared key
    assert reading(LISTED, keys={"bill": [("bill_id",)]}) == READ_BARE


def test_other_spellings_of_the_counts():
    # the count as a sum of 0/1, a CASE for the zero, MAX of a parent column for the column
    spelled = GROUPED.replace("COUNT(f.owner_id)", "SUM(CASE WHEN f.owner_id IS NOT NULL THEN 1 ELSE 0 END)").replace("COUNT(m.mail)", "SUM(CASE WHEN m.mail IS NULL THEN 0 ELSE 1 END)")
    assert reading(spelled) == READ.format(c="c")
    assert reading(GROUPED.replace("c.member_name,", "MAX(c.member_name) AS member_name,").replace(", c.member_name", "")) == READ.format(c="c")
    assert reading(COUNTED.replace("COALESCE(g.n, 0)", "CASE WHEN g.n IS NULL THEN 0 ELSE g.n END")) == READ.format(c="c")
    # the child linked to either of two columns an inner lookup makes equal
    assert reading(GROUPED.replace("c.member_id = f.owner_id", "i.owner_id = f.owner_id")) == READ.format(c="c")
    # a test on the child row, in the ON or in the count
    tested = "SELECT i.bill_id, COUNT(CASE WHEN f.friend_name = 'x' THEN 1 END) AS n FROM bill AS i LEFT JOIN friend AS f ON i.owner_id = f.owner_id GROUP BY i.bill_id"
    moved = "SELECT i.bill_id, COUNT(f.owner_id) AS n FROM bill AS i LEFT JOIN friend AS f ON i.owner_id = f.owner_id AND f.friend_name = 'x' GROUP BY i.bill_id"
    first = "SELECT i.bill_id, COALESCE(g.n, 0) AS n FROM bill AS i LEFT JOIN (SELECT owner_id, COUNT(*) AS n FROM friend WHERE friend_name = 'x' GROUP BY owner_id) AS g ON i.owner_id = g.owner_id"
    assert reading(tested) is not None and reading(tested) == reading(moved) == reading(first)
    # the order of the whole result is not compared
    assert reading(GROUPED + " ORDER BY i.bill_id") == READ.format(c="c")


def test_a_left_lookup_the_foreign_key_fills_is_an_inner_one():
    left = COUNTED.replace("bill AS i JOIN member", "bill AS i LEFT JOIN member")
    assert reading(left) == READ.format(c="c")
    kept = reading(left, foreign=None)
    assert kept is not None and "LEFT JOIN member AS c" in kept
    assert "LEFT JOIN member AS c" in reading(left, not_null={"bill": frozenset({"bill_id"})})


LEFT_ALONE = [
    GROUPED.replace(" GROUP BY", " WHERE i.amount > 1 GROUP BY"),
    GROUPED + " HAVING COUNT(*) > 1",
    GROUPED + " LIMIT 3",
    GROUPED + " ORDER BY i.bill_id LIMIT 3",
    GROUPED.replace("SELECT i.bill_id", "SELECT DISTINCT i.bill_id"),
    GROUPED.replace("COUNT(f.owner_id)", "COUNT(*)"),  # counts the padded row too
    GROUPED.replace("COUNT(f.owner_id)", "SUM(f.owner_id)"),
    GROUPED.replace("COUNT(f.owner_id)", "COUNT(DISTINCT f.friend_name)"),
    GROUPED.replace("COUNT(f.owner_id)", "COUNT(c.member_name)"),  # a count of parent values
    GROUPED.replace("COUNT(m.mail)", "COUNT(m.member_name)"),  # NULL on a match too
    GROUPED.replace("COUNT(f.owner_id)", "COUNT(CASE WHEN f.friend_name <> 'x' THEN RAND() END)"),
    GROUPED.replace("COUNT(f.owner_id) AS friends", "COUNT(f.owner_id) OVER () AS friends"),
    GROUPED.replace("LEFT JOIN friend", "JOIN friend"),  # an owner with no friend is dropped
    GROUPED.replace("LEFT JOIN member AS m", "JOIN member AS m"),
    GROUPED.replace("f.friend_mail = m.mail", "f.friend_mail = m.member_name"),  # not a key: several rows may match
    GROUPED.replace("c.member_id = i.owner_id", "c.member_name = i.owner_id"),
    GROUPED.replace("c.member_id = f.owner_id", "c.member_id < f.owner_id"),
    GROUPED.replace("c.member_id = f.owner_id", "c.member_id = f.owner_id OR f.friend_name = 'x'"),
    GROUPED.replace("f.owner_id)", "f.`owner_id`)"),
    GROUPED.replace("GROUP BY i.bill_id, c.member_name", "GROUP BY i.amount, c.member_name").replace("SELECT i.bill_id", "SELECT i.amount"),  # two bills in one group
    GROUPED.replace("JOIN member AS c ON c.member_id = i.owner_id", "JOIN member AS c ON c.member_id = i.owner_id JOIN member AS d ON d.member_id = i.owner_id"),
    "SELECT i.bill_id, COUNT(f.friend_name) AS n FROM bill AS i LEFT JOIN friend AS f USING (owner_id) GROUP BY i.bill_id",
    "SELECT i.bill_id, COUNT(f.friend_name) AS n FROM bill AS i RIGHT JOIN friend AS f ON i.owner_id = f.owner_id GROUP BY i.bill_id",
    COUNTED.replace("COALESCE(g.n, 0)", "g.n"),  # NULL for an owner with no friend
    COUNTED.replace("COALESCE(g.n, 0)", "COALESCE(g.n, 1)"),
    COUNTED.replace("LEFT JOIN (SELECT owner_id, COUNT(*) AS n FROM friend GROUP BY", "JOIN (SELECT owner_id, COUNT(*) AS n FROM friend GROUP BY"),
    COUNTED.replace("FROM friend GROUP BY owner_id)", "FROM friend GROUP BY owner_id HAVING COUNT(*) > 1)"),
    COUNTED.replace("FROM friend GROUP BY owner_id)", "FROM friend GROUP BY owner_id, friend_name)"),
    COUNTED.replace("COUNT(*) AS n FROM friend GROUP", "MAX(friend_name) AS n FROM friend GROUP"),
    COUNTED.replace("WHERE friend_mail IN (SELECT mail FROM member)", "WHERE friend_mail IN (SELECT mail FROM member) OR friend_name = 'x'"),
    COUNTED.replace("WHERE friend_mail IN", "WHERE friend_mail NOT IN"),
    COUNTED.replace("(SELECT mail FROM member)", "(SELECT mail FROM member WHERE member_name = 'p')"),
    PARENT_FIRST.replace("ON i.owner_id = c.member_id", "ON i.owner_id = c.friends"),
    PARENT_FIRST.replace("ON i.owner_id = c.member_id", "ON i.owner_id = c.member_id AND c.friends > 0"),
    # a left join to members with counts that an inner lookup thins: a bill may find no row, and read NULL
    "SELECT i.bill_id, t.friends FROM bill AS i LEFT JOIN (SELECT a.member_id, COUNT(b.owner_id) AS friends FROM member AS a JOIN bill AS x ON x.bill_id = a.member_id "
    "LEFT JOIN friend AS b ON a.member_id = b.owner_id GROUP BY a.member_id) AS t ON i.owner_id = t.member_id",
    FLAGGED.replace("SUM(v.flag)", "COUNT(v.flag)"),  # a flag is never NULL
    FLAGGED.replace("THEN 0 ELSE 1 END", "THEN 0 ELSE 2 END"),
    FLAGGED.replace("ON f.friend_mail = m.mail) AS v", "ON f.friend_mail = m.mail WHERE f.friend_name <> 'y') AS v"),
    FLAGGED.replace("GROUP BY v.owner_id", "GROUP BY v.flag").replace("SELECT v.owner_id,", "SELECT v.flag AS owner_id,"),
    LISTED.replace("SELECT DISTINCT mail FROM member", "SELECT DISTINCT mail FROM member WHERE member_name = 'p'"),
    LISTED.replace("SELECT DISTINCT mail FROM member", "SELECT DISTINCT mail, member_name FROM member"),
    LISTED.replace("(SELECT DISTINCT mail FROM member) AS e ON f.friend_mail = e.mail", "(SELECT member_name AS mail FROM member) AS e ON f.friend_mail = e.mail"),  # may repeat
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_shapes_are_left_alone(sql):
    sqlglot.parse_one(sql, read="mysql")
    assert reading(sql) is None


def test_facts_that_are_needed():
    assert reading(GROUPED, keys=None) is None
    assert reading(GROUPED, schema=None) is None
    # grouped after the joins: the bill's key and its NOT NULL make one group per bill
    assert reading(GROUPED, keys={"member": KEYS["member"]}) is None
    assert reading(GROUPED, not_null={"bill": frozenset({"owner_id"})}) is None
    # the lookups and the table asked for a match: a one-column key, so at most one row
    assert reading(GROUPED, keys={"bill": KEYS["bill"], "member": [("mail",)]}) is None
    assert reading(GROUPED, keys={"bill": KEYS["bill"], "member": [("member_id",)]}) is None
    assert reading(GROUPED, keys={"bill": KEYS["bill"], "member": [("member_id",), ("mail", "member_name")]}) is None
    # counted first groups no parent rows: no key of the bill is needed, the lookup's key still is
    assert reading(COUNTED, keys={"member": [("member_id",)]}, not_null=None, foreign=None) == READ.format(c="c")
    assert reading(COUNTED, keys={"member": [("mail",)]}) is None
    # members with their counts first: the joined column must be their key
    assert reading(PARENT_FIRST, keys={"member": [("member_id",)]}, not_null=None, foreign=None) == READ.format(c="a")
    assert reading(PARENT_FIRST, keys={"member": [("mail",)]}) is None
    # a LEFT join to them pads a bill whose owner is no member, unless the foreign key says there is none
    padded = PARENT_FIRST.replace("bill AS i JOIN (", "bill AS i LEFT JOIN (")
    assert reading(padded) == READ.format(c="a")
    assert reading(padded, foreign=None) is None
    assert reading(padded, not_null={"bill": frozenset({"bill_id"})}) is None
    assert reading(padded, foreign={"bill": [(("owner_id",), "member", ("mail",))]}) is None
    # columns of different declared types are not put in one equality
    assert reading(GROUPED, types={"bill": {"owner_id": "INT"}, "member": {"member_id": "VARCHAR"}, "friend": {"owner_id": "INT"}}) is None
    assert reading(GROUPED, types={"bill": {"owner_id": "INT"}, "member": {"member_id": "INT"}, "friend": {"owner_id": "INT"}}) == READ.format(c="c")


SPELLINGS = [
    GROUPED,
    COUNTED,
    PARENT_FIRST,
    PARENT_FIRST.replace("bill AS i JOIN (", "bill AS i LEFT JOIN ("),
    COUNTED.replace("bill AS i JOIN member", "bill AS i LEFT JOIN member"),
    FLAGGED,
    LISTED,
    "SELECT i.bill_id, MAX(c.member_name) AS nm, SUM(CASE WHEN f.owner_id IS NOT NULL THEN 1 ELSE 0 END) AS a, SUM(CASE WHEN m.mail IS NULL THEN 0 ELSE 1 END) AS b "
    "FROM bill i JOIN member c ON i.owner_id = c.member_id LEFT JOIN friend f ON i.owner_id = f.owner_id LEFT JOIN member m ON f.friend_mail = m.mail GROUP BY i.bill_id",
    "SELECT i.bill_id, COUNT(f.friend_name) AS a, SUM(CASE WHEN f.friend_mail IN (SELECT mail FROM member) THEN 1 ELSE 0 END) AS b, "
    "COALESCE(COUNT(CASE WHEN f.friend_name = 'x' THEN 1 END), 0) AS c FROM bill i LEFT JOIN friend f ON i.owner_id = f.owner_id GROUP BY i.bill_id",
    "SELECT i.bill_id, i.amount + 1 AS up, COUNT(f.owner_id) AS a, COUNT(CASE WHEN f.friend_name <> 'x' AND f.friend_mail IS NOT NULL THEN f.friend_mail END) AS b "
    "FROM bill i LEFT JOIN friend f ON i.owner_id = f.owner_id AND f.friend_name <> 'y' GROUP BY i.bill_id, i.amount",
    "SELECT i.bill_id, CASE WHEN g.n IS NULL THEN 0 ELSE g.n END AS a, COALESCE(g.t, 0) AS b, COALESCE(g.u, 0) AS c FROM bill i LEFT JOIN (SELECT f.owner_id, COUNT(f.friend_name) AS n, "
    "COUNT(m.mail) AS t, SUM(CASE WHEN f.friend_name = 'x' THEN 1 ELSE 0 END) AS u FROM friend f LEFT JOIN member m ON f.friend_mail = m.mail GROUP BY f.owner_id) g ON i.owner_id = g.owner_id",
    "SELECT i.bill_id, c.mail, COALESCE(g.n, 0) AS a FROM bill i LEFT JOIN member c ON c.member_id = i.owner_id "
    "LEFT JOIN (SELECT owner_id, COUNT(friend_mail) AS n FROM friend WHERE friend_name <> 'x' GROUP BY owner_id) g ON c.member_id = g.owner_id",
    "SELECT i.bill_id, t.nm, COALESCE(t.friends, 0) AS a FROM bill i JOIN (SELECT a.member_id AS mid, a.member_name AS nm, COUNT(b.owner_id) AS friends FROM member a "
    "LEFT JOIN friend b ON a.member_id = b.owner_id AND b.friend_name = 'x' GROUP BY a.member_id, a.member_name) t ON t.mid = i.owner_id",
    "SELECT i.bill_id, COALESCE(g.n, 0) AS a, COALESCE(g.t, 0) AS b FROM bill i LEFT JOIN (SELECT owner_id, COUNT(friend_mail) AS n, SUM(flag) AS t FROM "
    "(SELECT owner_id, friend_mail, CASE WHEN friend_mail IN (SELECT mail FROM member) THEN 1 ELSE 0 END AS flag FROM friend) v GROUP BY owner_id) g ON i.owner_id = g.owner_id",
    "SELECT i.bill_id, COALESCE(g.t, 0) AS b FROM bill i LEFT JOIN (SELECT f.owner_id, COUNT(e.m2) AS t FROM friend f LEFT JOIN (SELECT DISTINCT mail AS m2 FROM member) e "
    "ON f.friend_mail = e.m2 GROUP BY f.owner_id) g ON i.owner_id = g.owner_id",
]


def _database(duckdb, rng, held: bool):
    """Random tables; with ``held`` they meet the declared keys, NOT NULL columns and the foreign key."""

    con = duckdb.connect()
    con.execute("CREATE TABLE bill (bill_id INT, owner_id INT, amount INT)")
    con.execute("CREATE TABLE member (member_id INT, member_name VARCHAR, mail VARCHAR)")
    con.execute("CREATE TABLE friend (owner_id INT, friend_name VARCHAR, friend_mail VARCHAR)")
    ids = rng.sample(range(1, 9), rng.randint(1, 5)) if held else [rng.randint(1, 4) for _ in range(rng.randint(1, 5))]
    mails = rng.sample(["a", "b", "c", "d", "e", "f"], 6)
    for n, value in enumerate(ids):
        mail = rng.choice([None, mails[n]]) if held else rng.choice([None, "a", "b"])  # a key column may hold NULLs
        con.execute("INSERT INTO member VALUES (?, ?, ?)", [value, rng.choice([None, "p", "q"]), mail])
    if rng.random() < 0.4:
        con.execute("INSERT INTO member VALUES (NULL, 'z', ?)", [None if held else "a"])
    for number in range(rng.randint(0, 5)):
        owner = rng.choice(ids) if held else rng.choice([None, 1, 2, 3, 9])
        con.execute("INSERT INTO bill VALUES (?, ?, ?)", [number if held else rng.choice([0, 1, None]), owner, rng.choice([None, 5, 6])])
    for _ in range(rng.randint(0, 8)):
        con.execute("INSERT INTO friend VALUES (?, ?, ?)", [rng.choice([None, 1, 2, 3, 4, 5, 9]), rng.choice([None, "x", "y", "w"]), rng.choice([None, "a", "b", "c", "zz"])])
    return con


@pytest.mark.parametrize("seed", range(4))
def test_the_reading_returns_the_rows_duckdb_returns_when_the_facts_hold(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    pairs = []
    for sql in SPELLINGS:
        read = reading(sql)
        assert read is not None, sql
        pairs.append((sqlglot.transpile(sql, read="mysql", write="duckdb")[0], sqlglot.transpile(read, read="mysql", write="duckdb")[0]))
    nonempty = differs = 0
    for held in [True] * 14 + [False] * 6:
        con = _database(duckdb, rng, held)
        for sql, read in pairs:
            expected = sorted(con.execute(sql).fetchall(), key=repr)
            same = expected == sorted(con.execute(read).fetchall(), key=repr)
            if held:
                assert same, sql
                nonempty += bool(expected)
            else:
                differs += not same
    assert nonempty > 60 and differs > 0  # the facts are what make it true


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402


def facts(bill_key=True, mail_key=True, foreign=True):
    return {
        "bill": TableConstraints(not_null=frozenset({"bill_id", "owner_id"}), keys=(("bill_id",),) if bill_key else (), foreign_keys=((("owner_id",), "member", ("member_id",)),) if foreign else ()),
        "member": TableConstraints(not_null=frozenset({"member_id"}), keys=(("member_id",),) + ((("mail",),) if mail_key else ())),
        "friend": TableConstraints(),
    }


def proved(left: str, right: str, constraints) -> bool:
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=constraints, compare_names=False, dialect="mysql").proven


PADDED = PARENT_FIRST.replace("bill AS i JOIN (", "bill AS i LEFT JOIN (")


def test_the_spellings_are_proved_equal_under_the_keys():
    assert proved(GROUPED, COUNTED, facts())
    assert proved(GROUPED, PARENT_FIRST, facts())
    assert proved(COUNTED, PADDED, facts())
    assert proved(FLAGGED, LISTED, facts())


def test_without_a_needed_fact_nothing_is_proved():
    assert not proved(GROUPED, COUNTED, facts(bill_key=False))  # two bills with one number are one group
    assert not proved(GROUPED, COUNTED, facts(mail_key=False))  # a friend's mail on two members is counted twice
    assert not proved(COUNTED, PADDED, facts(foreign=False))  # a bill whose owner is no member


def test_it_is_a_narrow_retry_rule():
    constraints = facts()
    kwargs = dict(
        schema=SCHEMA,
        dialect="mysql",
        keys={t: list(c.keys) for t, c in constraints.items()},
        not_null={t: c.not_null for t, c in constraints.items()},
        foreign_keys={t: list(c.foreign_keys) for t, c in constraints.items() if c.foreign_keys},
    )
    with shape_fallback.tracking() as fired:
        normalize(GROUPED, **kwargs)
    assert "child_counts" in fired


def test_the_spellings_differ_when_a_fact_is_broken():
    duckdb = pytest.importorskip("duckdb")

    def rows(sql: str, bills: str, members: str, friends: str):
        con = duckdb.connect()
        con.execute("CREATE TABLE bill (bill_id INT, owner_id INT, amount INT)")
        con.execute("CREATE TABLE member (member_id INT, member_name VARCHAR, mail VARCHAR)")
        con.execute("CREATE TABLE friend (owner_id INT, friend_name VARCHAR, friend_mail VARCHAR)")
        con.execute(f"INSERT INTO bill VALUES {bills}")
        con.execute(f"INSERT INTO member VALUES {members}")
        con.execute(f"INSERT INTO friend VALUES {friends}")
        return sorted(con.execute(sql).fetchall(), key=repr)

    # the facts hold: one answer
    held = ("(1, 7, 5), (2, 7, 6)", "(7, 'p', 'a'), (8, 'q', 'b')", "(7, 'x', 'b'), (7, 'y', 'zz')")
    assert rows(GROUPED, *held) == rows(COUNTED, *held) == rows(PADDED, *held) == [(1, "p", 2, 1), (2, "p", 2, 1)]
    # two bills with one number
    twice = ("(1, 7, 5), (1, 7, 6)", held[1], held[2])
    assert rows(GROUPED, *twice) == [(1, "p", 4, 2)] and rows(COUNTED, *twice) == [(1, "p", 2, 1), (1, "p", 2, 1)]
    # one mail on two members
    shared = (held[0], "(7, 'p', 'a'), (8, 'q', 'b'), (9, 'r', 'b')", held[2])
    assert rows(GROUPED, *shared) == [(1, "p", 3, 2), (2, "p", 3, 2)] and rows(COUNTED, *shared) == [(1, "p", 2, 1), (2, "p", 2, 1)]
    # a bill whose owner is no member
    orphan = ("(1, 7, 5), (2, 4, 6)", held[1], held[2])
    assert rows(COUNTED, *orphan) == [(1, "p", 2, 1)] and rows(PADDED, *orphan) == [(1, "p", 2, 1), (2, None, None, None)]
